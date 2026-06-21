"""
solvers.py

Основные функции решателей для алгоритма графовой слепой деконволюции 
изображений (GBBID).

Основано на методах:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Также включает методы неслепой деконволюции из работ:
    D. Krishnan, R. Fergus: "Fast Image Deconvolution using
    Hyper-Laplacian Priors", NIPS 2009.
    L. Xu et al.: "Unnatural L0 Sparse Representation for Natural 
    Image Deblurring", CVPR 2013.

Содержит:
    - TV_denoising : предварительное шумоподавление с TV-регуляризацией.
    - apply_denoiser : диспетчер применения алгоритмов шумоподавления.
    - Deblur_GL_CG_4 : решатель на основе графового лапласиана и сопряженных градиентов.
    - kernel_solver_L2 : оценка ядра размытия в частотной области.
    - bid_rgtv_c2f_cg : главный цикл иерархической слепой деконволюции.
    - fast_deconv : неслепая деконволюция с гиперлапласовским априорным распределением.
    - Deconvolution_FHLP : обертка для неслепой деконволюции со сглаживанием краев.
    - deblurring_adm_aniso : анизотропная TV-деконволюция (метод ADM).
    - L0Restoration : восстановление изображения с L0-нормой градиентов.
    - ringing_artifacts_removal : комплексное подавление эффекта звона.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import convolve as ndimage_convolve
from skimage.transform import resize as sk_resize

from .utils import (
    psf2otf,
    otf2psf,
    G_padding,
    Copy_Enlarge_h,
    fftconv,
    edgetaper,
    weights_computation,
    informative_edge_mask_adaptive_mine,
    kernel_centralize,
    conjgrad,
    GenerateFrameletFilter,
    FraDecMultiLevel2D,
    kernel_filter,
    solve_image,
    clear_solve_image_cache,
    opt_fft_size,
    wrap_boundary_liu,
    bilateral_filter,
)

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


blinddeconv.algorithms.mod_cython._build_pyd


def TV_denoising(I, weight, max_it):
    """
    Подавление шума с использованием априорного распределения полной 
    вариации (TV), реализованное в частотной области (формулировка Split Bregman / ADMM).

    Параметры
    ---------
    I : ndarray
        Двумерный массив входного изображения.
    weight : tuple
        Кортеж (mu, gamma) с параметрами регуляризации.
    max_it : int
        Количество итераций алгоритма.

    Возвращает
    ----------
    x : ndarray
        Изображение после фильтрации (размер совпадает с I).
    """
    h = np.full((3, 3), 1e-10, dtype=np.float64)
    h[1, 1] = 1.0
    h = h / h.sum()

    mu = weight[0]
    gamma = weight[1]

    s_h, s_w = h.shape

    I_sym, border = Copy_Enlarge_h(I, (s_h * 3, s_w * 3))
    I_sym = edgetaper(I_sym, h)
    I_h, I_w = I_sym.shape

    B = h
    Bt = h[::-1, ::-1]

    from scipy.signal import fftconvolve
    BtB = fftconvolve(Bt, B)
    S_h = 2 * s_h - 1
    S_w = 2 * s_w - 1

    vtv_w = np.array([0, -1, 2, -1, 0], dtype=np.float64)
    vtv_h = vtv_w.copy()

    VtV_w = np.zeros((S_h, S_w), dtype=np.float64)
    VtV_w[s_h - 1, s_w - 3:s_w + 2] = vtv_w

    VtV_h = np.zeros((S_h, S_w), dtype=np.float64)
    VtV_h[s_h - 3:s_h + 2, s_w - 1] = vtv_h

    Bt_tmp = np.zeros((S_h, S_w), dtype=np.float64)
    hh = (s_h - 1) // 2
    hw = (s_w - 1) // 2
    Bt_tmp[s_h - 1 - hh:s_h + hh, s_w - 1 - hw:s_w + hw] = Bt

    vt_w = np.array([0, -1, 1], dtype=np.float64)
    vt_h = vt_w.copy()

    Vt_w = np.zeros((S_h, S_w), dtype=np.float64)
    Vt_w[s_h - 1, s_w - 2:s_w + 1] = vt_w

    Vt_h = np.zeros((S_h, S_w), dtype=np.float64)
    Vt_h[s_h - 2:s_h + 1, s_w - 1] = vt_h

    v_w = np.array([[1, -1, 0]], dtype=np.float64)
    v_h = v_w.T

    z_h = np.zeros((I_h, I_w), dtype=np.float64)
    z_w = np.zeros((I_h, I_w), dtype=np.float64)
    y_h = np.zeros((I_h, I_w), dtype=np.float64)
    y_w = np.zeros((I_h, I_w), dtype=np.float64)

    fft_shape = (S_h + I_h - 1, S_w + I_w - 1)

    def _embed_fft(arr, arr_shape, total_shape):
        """
        Вспомогательная функция для встраивания массива в нуль-матрицу 
        заданного размера и выполнения двумерного БПФ.
        """
        Ftmp = np.zeros(total_shape, dtype=np.float64)
        Ftmp[:arr_shape[0], :arr_shape[1]] = arr
        return fft2(Ftmp)

    FBtB = _embed_fft(BtB, (S_h, S_w), fft_shape)
    FBt = _embed_fft(Bt_tmp, (S_h, S_w), fft_shape)
    FVtV_w = _embed_fft(VtV_w, (S_h, S_w), fft_shape)
    FVtV_h = _embed_fft(VtV_h, (S_h, S_w), fft_shape)
    FVt_w = _embed_fft(Vt_w, (S_h, S_w), fft_shape)
    FVt_h = _embed_fft(Vt_h, (S_h, S_w), fft_shape)

    Fb = _embed_fft(I_sym, (I_h, I_w), fft_shape)

    for _ in range(max_it):
        Fz_h = _embed_fft(z_h, (I_h, I_w), fft_shape)
        Fz_w = _embed_fft(z_w, (I_h, I_w), fft_shape)
        Fy_h = _embed_fft(y_h, (I_h, I_w), fft_shape)
        Fy_w = _embed_fft(y_w, (I_h, I_w), fft_shape)

        x = ifft2(
            (Fb * FBt
             + gamma * FVt_h * Fz_h + gamma * FVt_w * Fz_w
             + gamma * FVt_h * Fy_h + gamma * FVt_w * Fy_w)
            / (FBtB + gamma * FVtV_w + gamma * FVtV_h)
        )
        x = np.real(x[:I_h, :I_w])

        Vx_h = ndimage_convolve(x, v_h, mode='nearest')
        Vx_h[-1, :] = 0.0
        Vx_w = ndimage_convolve(x, v_w, mode='nearest')
        Vx_w[:, -1] = 0.0

        thr = mu / gamma

        z_h[:] = 0.0
        diff_h = Vx_h - y_h
        mask_pos = diff_h > thr
        mask_neg = diff_h < -thr
        z_h[mask_pos] = diff_h[mask_pos] - thr
        z_h[mask_neg] = diff_h[mask_neg] + thr

        z_w[:] = 0.0
        diff_w = Vx_w - y_w
        mask_pos = diff_w > thr
        mask_neg = diff_w < -thr
        z_w[mask_pos] = diff_w[mask_pos] - thr
        z_w[mask_neg] = diff_w[mask_neg] + thr

        y_h = y_h - (Vx_h - z_h)
        y_w = y_w - (Vx_w - z_w)

    x = x[border[0]:I_h - border[0], border[1]:I_w - border[1]]
    return x



def _guided_filter(I, p, radius, eps):
    """
    Фильтрация на основе направляющего изображения (Guided Filter).
    Реализует сглаживание с сохранением краев объектов (подход He et al. 2010).
    """
    from scipy.ndimage import uniform_filter
    size = 2 * radius + 1
    mean_I = uniform_filter(I, size)
    mean_p = uniform_filter(p, size)
    corr_Ip = uniform_filter(I * p, size)
    var_I = uniform_filter(I * I, size) - mean_I * mean_I
    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)
    return mean_a * I + mean_b


def apply_denoiser(img, method, **params):
    """
    Применение выбранного метода шумоподавления к двумерному изображению.

    Для метода 'act' (адаптивное пороговое ограничение в кривлет-области):
    метод устойчив к цветному и пуассоновскому шуму без использования 
    стабилизирующих преобразований. Важно передавать точную оценку дисперсии 
    в параметре `noise_var`, так как резервная слепая оценка по методу MAD 
    может занижать уровень шума в темных участках.

    Параметры
    ---------
    img : ndarray
        Двумерный массив изображения.
    method : str
        Идентификатор метода: 'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 
        'act', 'vst_bm3d' или 'none' (либо None).
    **params : dict
        Специфические параметры для каждого метода:
        - tv: mu, gamma, max_it
        - nlm: patch_size, patch_distance, h, sigma
        - bilateral: sigma_color, sigma_spatial
        - guided: radius, eps
        - bm3d: sigma_psd
        - act: noise_var, threshold_setting
        - vst_bm3d: noise_info, a, b, sigma, stage_arg

    Возвращает
    ----------
    denoised : ndarray
        Отфильтрованное изображение.
    """
    if method is None or method == 'none':
        return img.copy()

    if method == 'tv':
        mu = params.get('mu', 0.01)
        gamma = params.get('gamma', 0.1)
        max_it = params.get('max_it', 10)
        return TV_denoising(img, (mu, gamma), max_it)

    elif method == 'nlm':
        from skimage.restoration import denoise_nl_means, estimate_sigma
        sigma_est = params.get('sigma', None)
        if sigma_est is None:
            sigma_est = float(estimate_sigma(img))
        patch_size = params.get('patch_size', 5)
        patch_distance = params.get('patch_distance', 6)
        h = params.get('h', 0.8 * sigma_est)
        return denoise_nl_means(
            img, h=h, patch_size=patch_size,
            patch_distance=patch_distance, fast_mode=True)

    elif method == 'bilateral':
        from skimage.restoration import denoise_bilateral, estimate_sigma
        sigma_color = params.get('sigma_color', None)
        if sigma_color is None:
            sigma_color = float(estimate_sigma(img))
        sigma_spatial = params.get('sigma_spatial', 1.0)
        return denoise_bilateral(
            img, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    elif method == 'guided':
        radius = params.get('radius', 5)
        eps = params.get('eps', 0.01)
        return _guided_filter(img, img, radius, eps)

    elif method == 'bm3d':
        try:
            import bm3d as bm3d_lib
        except ImportError:
            raise ImportError("bm3d package required: pip install bm3d")
        from skimage.restoration import estimate_sigma
        sigma_psd = params.get('sigma_psd', None)
        if sigma_psd is None:
            sigma_psd = float(estimate_sigma(img))
        return bm3d_lib.bm3d(img, sigma_psd=sigma_psd)

    elif method == 'act':
        from blinddeconv.algorithms.mod_cython._build_pyd.act_denoise import act_denoise
        nv = params.get('noise_var', None)
        ts = params.get('threshold_setting', 's')
        result, _ = act_denoise(img, noise_var=nv, threshold_setting=ts)
        return result

    elif method == 'vst_bm3d':
        from blinddeconv.algorithms.mod_cython._build_pyd.vst import vst_bm3d_denoise
        noise_info = params.get('noise_info', None)
        a = params.get('a', None)
        b = params.get('b', None)
        sigma = params.get('sigma', None)
        result, _ = vst_bm3d_denoise(img, noise_info=noise_info,
                                     a=a, b=b, sigma=sigma)
        return result

    else:
        raise ValueError(f"Unknown denoiser method: {method}")



def Deblur_GL_CG_4(Y_b, k, W, we, max_iter):
    """
    Восстановление структурного изображения (skeleton) с использованием 
    графового лапласиана и метода сопряженных градиентов.

    Решает оптимизационную задачу:
        min_x ||k * x - Y_b||^2 + we * x^T * L * x
    где L — матрица графового лапласиана, сформированная на основе весов W.

    Параметры
    ---------
    Y_b : ndarray
        Размытое изображение (возможно с дополнением границ).
    k : ndarray
        Ядро размытия.
    W : ndarray
        Массив весов графа размерности (h*w, 4).
    we : float
        Вес графового регуляризатора (соответствует параметру mu).
    max_iter : int
        Количество итераций метода сопряженных градиентов.

    Возвращает
    ----------
    x : ndarray
        Восстановленное изображение, ограниченное диапазоном [0, 1].
    """
    d1 = np.array([[1, -1, 0]], dtype=np.float64)
    d1_c = np.array([[0, -1, 1]], dtype=np.float64)
    d2 = d1.T
    d2_c = d1_c.T
    d3 = np.array([[0, -1, 1]], dtype=np.float64)
    d3_c = np.array([[1, -1, 0]], dtype=np.float64)
    d4 = d3.T
    d4_c = d3_c.T

    Y_b_padding = Y_b
    h_p, w_p = Y_b_padding.shape
    x = Y_b_padding.copy()

    vertex, neighbours_num = W.shape
    if vertex != h_p * w_p or neighbours_num != 4:
        raise ValueError("Weights matrix W is not correct, please check.")

    k_flipped = k[::-1, ::-1]
    use_fft = max(k.shape) >= 25

    def _apply_blur(img):
        """Применение операции прямого и сопряженного размытия."""
        if use_fft:
            return fftconv(fftconv(img, k, 'same'), k_flipped, 'same')
        else:
            tmp = ndimage_convolve(img, k, mode='nearest')
            return ndimage_convolve(tmp, k_flipped, mode='nearest')

    def _apply_graph(img):
        """Применение графового регуляризатора: D^T * W * D * x."""
        w1 = W[:, 0].reshape(h_p, w_p)
        w2 = W[:, 1].reshape(h_p, w_p)
        w3 = W[:, 2].reshape(h_p, w_p)
        w4 = W[:, 3].reshape(h_p, w_p)

        out = we * ndimage_convolve(
            w1 * ndimage_convolve(img, d1, mode='nearest'),
            d1_c, mode='nearest')
        out += we * ndimage_convolve(
            w2 * ndimage_convolve(img, d2, mode='nearest'),
            d2_c, mode='nearest')
        out += we * ndimage_convolve(
            w3 * ndimage_convolve(img, d3, mode='nearest'),
            d3_c, mode='nearest')
        out += we * ndimage_convolve(
            w4 * ndimage_convolve(img, d4, mode='nearest'),
            d4_c, mode='nearest')
        return out

    if use_fft:
        b = fftconv(Y_b_padding, k_flipped, 'same')
    else:
        b = ndimage_convolve(Y_b_padding, k_flipped, mode='nearest')

    Ax = _apply_blur(x) + _apply_graph(x)

    r = b - Ax
    rho_1 = 0.0
    p = None
    for i in range(max_iter):
        rho = np.sum(r * r)

        if i > 0:
            beta_cg = rho / rho_1
            p = r + beta_cg * p
        else:
            p = r.copy()

        Ap = _apply_blur(p) + _apply_graph(p)

        q = Ap
        pq = np.sum(p * q)
        if pq == 0:
            break
        alpha_cg = rho / pq
        x = x + alpha_cg * p
        r = r - alpha_cg * q

        rho_1 = rho

        x = np.clip(x, 0.0, 1.0)

    return x


def _compute_Ax_kernel(x, p):
    """
    Вычисление произведения матрицы на вектор для системы уравнений оценки ядра.
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def kernel_solver_L2(Y, b, k_size, M, lambda_val):
    """
    Оценка ядра размытия в градиентной области с помощью метода сопряженных градиентов.

    Параметры
    ---------
    Y : ndarray
        Оцененное структурное изображение (latent).
    b : ndarray
        Размытое наблюдаемое изображение.
    k_size : int
        Размер стороны квадратного ядра размытия (нечетное число).
    M : ndarray или None
        Маска информативных краев.
    lambda_val : float
        Коэффициент регуляризации.

    Возвращает
    ----------
    psf : ndarray
        Оцененное ядро размытия размерности (k_size, k_size).
    """
    dx = np.array([[1, -1, 0]], dtype=np.float64)
    dy = dx.T

    if M is None:
        M = np.ones_like(Y)

    Yx = ndimage_convolve(Y, dx, mode='nearest') * M
    Yy = ndimage_convolve(Y, dy, mode='nearest') * M

    bx = ndimage_convolve(b, dx, mode='nearest')
    by = ndimage_convolve(b, dy, mode='nearest')

    pad_time = 3
    pad_size = int(np.floor(k_size * pad_time))

    bx_p = np.pad(bx, pad_size)
    by_p = np.pad(by, pad_size)
    Yx_p = np.pad(Yx, pad_size)
    Yy_p = np.pad(Yy, pad_size)

    Yx_f = fft2(Yx_p)
    Yy_f = fft2(Yy_p)
    bx_f = fft2(bx_p)
    by_f = fft2(by_p)

    wx = 25.0
    wy = 25.0
    psf_size = (k_size, k_size)

    b_rhs_f = wx * np.conj(Yx_f) * bx_f + wy * np.conj(Yy_f) * by_f
    b_rhs = np.real(otf2psf(b_rhs_f, psf_size))

    p = {
        'm': wx * np.conj(Yx_f) * Yx_f + wy * np.conj(Yy_f) * Yy_f,
        'img_size': bx_f.shape,
        'psf_size': psf_size,
        'lambda': lambda_val,
    }

    psf = np.ones(psf_size, dtype=np.float64) / (k_size * k_size)
    psf = conjgrad(psf, b_rhs, 20, 1e-5, _compute_Ax_kernel, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    return psf



def bid_rgtv_c2f_cg(Y_b, k_estimate_size, show_intermediate=False,
                     preprocess='tv', preprocess_params=None,
                     pre_kernel='none', pre_kernel_params=None,
                     iteration_callback=None):
    """
    Иерархическая (от грубого масштаба к точному) слепая деконволюция 
    с использованием графовой регуляризации (RGTV).

    Параметры
    ---------
    Y_b : ndarray
        Размытое изображение (после обрезки краев).
    k_estimate_size : int
        Размер оцениваемого ядра размытия.
    show_intermediate : bool
        Флаг отображения промежуточных результатов (в данной реализации игнорируется).
    preprocess : str
        Идентификатор алгоритма предварительного шумоподавления перед 
        построением пирамиды.
    preprocess_params : dict или None
        Параметры для метода предварительного шумоподавления.
    pre_kernel : str
        Идентификатор алгоритма шумоподавления перед шагом оценки ядра.
    pre_kernel_params : dict или None
        Параметры для метода шумоподавления перед оценкой ядра.
    iteration_callback : callable или None
        Функция обратного вызова для отслеживания прогресса.

    Возвращает
    ----------
    k_estimate : ndarray
        Оцененное ядро размытия.
    Y_r_rgtv_cg : ndarray
        Восстановленное структурное изображение.
    """
    scale_factor = np.log2(3)
    level_num = int(np.ceil(np.log(k_estimate_size / 7) / np.log(scale_factor))) + 1

    image_pyramid = [None] * level_num
    k_size = np.zeros(level_num, dtype=int)
    image_size = np.zeros((level_num, 2), dtype=int)

    image_pyramid[0] = apply_denoiser(Y_b, preprocess, **(preprocess_params or {}))

    k_size[0] = k_estimate_size
    image_size[0] = image_pyramid[0].shape

    for i in range(1, level_num):
        image_size[i] = np.floor(image_size[i - 1] / np.log2(3)).astype(int)
        image_pyramid[i] = sk_resize(
            image_pyramid[i - 1],
            (int(image_size[i, 0]), int(image_size[i, 1])),
            order=1,
            anti_aliasing=True,
            preserve_range=True,
        )
        k_size[i] = int(np.floor(k_size[i - 1] / np.log2(3)))
        k_size[i] = k_size[i] + (1 - k_size[i] % 2)

    frame = 1
    Level = 1
    D, R = GenerateFrameletFilter(frame)

    k_estimate = None
    Y_r_rgtv_cg = None

    for level in range(level_num - 1, -1, -1):
        mu = 0.01
        lambda_val = 0.05
        sigma = 0.1 * np.sqrt(2)

        if level >= level_num - 1:
            ks = int(k_size[level])
            k_estimate = np.zeros((ks, ks), dtype=np.float64)
            k_center = ks // 2
            k_estimate[k_center, k_center] = 1.0
        else:
            ks = int(k_size[level])
            k_estimate = sk_resize(
                k_estimate, (ks, ks),
                order=1, anti_aliasing=True, preserve_range=True)
            k_estimate[k_estimate < k_estimate.max() * 0.05] = 0.0
            k_sum = k_estimate.sum()
            if k_sum > 0:
                k_estimate = k_estimate / k_sum

        Y_b_padding, padsize = G_padding(image_pyramid[level], k_estimate, 1)
        Y_r_rgtv_cg = Y_b_padding.copy()
        h, w = Y_r_rgtv_cg.shape

        for iter_main in range(3):
            W1 = np.ones((h * w, 4), dtype=np.float64)
            W = W1.copy()

            for i in range(3):
                for j in range(3):
                    Y_r_rgtv_cg = Deblur_GL_CG_4(
                        Y_b_padding, k_estimate, W, mu, 20)
                    W = W1 * weights_computation(Y_r_rgtv_cg, None, 4, 2)

                W1 = weights_computation(Y_r_rgtv_cg, sigma, 4, 1)
                W = W1 * weights_computation(Y_r_rgtv_cg, None, 4, 2)

            Y_r_rgtv_cg = Y_r_rgtv_cg[
                padsize[0]:h - padsize[0],
                padsize[1]:w - padsize[1]
            ]

            if pre_kernel is not None and pre_kernel != 'none':
                Y_for_kernel = apply_denoiser(
                    Y_r_rgtv_cg, pre_kernel, **(pre_kernel_params or {}))
            else:
                Y_for_kernel = Y_r_rgtv_cg

            t_s = 0.1
            t_r = 0.3
            M = informative_edge_mask_adaptive_mine(Y_for_kernel, t_s, t_r, 5)
            k_estimate = kernel_solver_L2(
                Y_for_kernel, image_pyramid[level],
                int(k_size[level]), M, lambda_val)

            if level <= 1:
                Cf = FraDecMultiLevel2D(k_estimate, D, Level)
                k_estimate = kernel_filter(Cf, R, Level, 0.05)
                k_estimate[k_estimate < k_estimate.max() * 0.05] = 0.0
                k_sum = k_estimate.sum()
                if k_sum > 0:
                    k_estimate = k_estimate / k_sum
                k_estimate = kernel_centralize(k_estimate, 0.1)

            lambda_val = lambda_val / 1.2

            if iteration_callback is not None:
                iteration_callback({
                    'iteration': iter_main,
                    'scale': level,
                    'num_scales': level_num,
                    'kernel': k_estimate.copy(),
                    'image': Y_r_rgtv_cg,
                    'metrics': {
                        'lambda_val': float(lambda_val),
                        'mu': float(mu),
                    },
                })

    return k_estimate, Y_r_rgtv_cg


def _computeDenominator(y, k):
    """
    Предварительное вычисление знаменателя и части числителя для 
    гиперлапласовской деконволюции.

    Возвращает
    ----------
    Nomin1 : произведение сопряженной оптической передаточной функции на спектр y
    Denom1 : квадрат модуля оптической передаточной функции ядра
    Denom2 : сумма квадратов модулей оптических передаточных функций градиентов
    """
    sizey = y.shape
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = (np.abs(psf2otf(np.array([[1, -1]]), sizey)) ** 2
              + np.abs(psf2otf(np.array([[1], [-1]]), sizey)) ** 2)
    return Nomin1, Denom1, Denom2


def fast_deconv(yin, k, lambda_val, alpha, yout0=None):
    """
    Неслепая деконволюция с гиперлапласовским априорным распределением.

    Решает задачу минимизации вида:
        min_y (lambda_val/2) * ||k * y - yin||^2 + ||D_x y||^alpha + ||D_y y||^alpha

    Параметры
    ---------
    yin : ndarray
        Размытое изображение в полутоновом формате.
    k : ndarray
        Ядро свертки (нечетного размера).
    lambda_val : float
        Коэффициент верности данных.
    alpha : float
        Параметр степени для гиперлапласиана (0 < alpha <= 2).
    yout0 : ndarray или None
        Начальное приближение. Если None, используется yin.

    Возвращает
    ----------
    yout : ndarray
        Восстановленное изображение.
    """
    beta = 1.0
    beta_rate = 2.0 * np.sqrt(2)
    beta_max = 2 ** 8

    mit_inn = 1

    m, n = yin.shape

    if yout0 is not None:
        yout = yout0.copy()
    else:
        yout = yin.copy()

    if k.shape[0] % 2 != 1 or k.shape[1] % 2 != 1:
        raise ValueError("Blur kernel k must be odd-sized.")

    Nomin1, Denom1, Denom2 = _computeDenominator(yin, k)

    youtx = np.concatenate([np.diff(yout, 1, axis=1), yout[:, 0:1] - yout[:, -1:]], axis=1)
    youty = np.concatenate([np.diff(yout, 1, axis=0), yout[0:1, :] - yout[-1:, :]], axis=0)

    while beta < beta_max:
        gamma = beta / lambda_val
        Denom = Denom1 + gamma * Denom2

        for _ in range(mit_inn):
            Wx = solve_image(youtx, beta, alpha)
            Wy = solve_image(youty, beta, alpha)

            Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1], -np.diff(Wx, 1, axis=1)], axis=1)
            Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :], -np.diff(Wy, 1, axis=0)], axis=0)

            Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
            yout = np.real(ifft2(Fyout))
            youtx = np.concatenate([np.diff(yout, 1, axis=1), yout[:, 0:1] - yout[:, -1:]], axis=1)
            youty = np.concatenate([np.diff(yout, 1, axis=0), yout[0:1, :] - yout[-1:, :]], axis=0)

        beta = beta * beta_rate

    return yout


def Deconvolution_FHLP(y, kernel, lambda_val=2e3, alpha=0.5,
                       edgetaper_iters=4):
    """
    Обертка для неслепой деконволюции с использованием быстрых гиперлапласовских 
    априорных распределений. Включает предварительное сглаживание краев (edgetaper) 
    для устранения круговых граничных артефактов.

    Параметры
    ---------
    y : ndarray
        Размытое изображение.
    kernel : ndarray
        Оцененное ядро размытия.
    lambda_val : float
        Вес члена верности данных.
    alpha : float
        Параметр гиперлапласиана.
    edgetaper_iters : int
        Количество итераций сглаживания краев.

    Возвращает
    ----------
    x : ndarray
        Восстановленное изображение (размер совпадает с y).
    """
    kernel = kernel.copy().astype(np.float64)
    kernel[kernel == 0] = 1e-10
    kernel = kernel / kernel.sum()

    ks = (kernel.shape[0] - 1) // 2

    y_padded = np.pad(y, ks, mode='edge')

    for _ in range(edgetaper_iters):
        y_padded = edgetaper(y_padded, kernel)

    clear_solve_image_cache()

    x = fast_deconv(y_padded, kernel, lambda_val, alpha)

    x = x[ks:x.shape[0] - ks, ks:x.shape[1] - ks]

    return x



def deblurring_adm_aniso(B, k, lambda_tv, alpha=1):
    """
    Анизотропная деконволюция с TV-регуляризацией методом попеременных 
    направлений (Split Bregman / ADM).

    Параметры
    ---------
    B : ndarray
        Размытое изображение.
    k : ndarray
        Ядро размытия нечетного размера.
    lambda_tv : float
        Вес TV-регуляризатора.
    alpha : int
        Экспонента нормы (поддерживается только значение 1).

    Возвращает
    ----------
    I : ndarray
        Восстановленное изображение.
    """
    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, n=1, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                f"deblurring_adm_aniso: alpha={alpha} not implemented; "
                f"only alpha=1 supported"
            )

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, n=1, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, n=1, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)

        beta = beta / 2.0

    return I


def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """
    Восстановление изображения с использованием L0-нормы градиентов.

    Решает задачу: min_S ||S * k - B||^2 + lambda * ||grad S||_0

    Параметры
    ---------
    Im : ndarray
        Размытое изображение в исходном размере.
    kernel : ndarray
        Ядро размытия.
    lambda_grad : float
        Вес градиентного априорного распределения L0.
    kappa : float
        Множитель обновления параметра ADM (по умолчанию 2.0).

    Возвращает
    ----------
    S : ndarray
        Восстановленное изображение, обрезанное до исходного размера.
    """
    H_orig, W_orig = Im.shape[0], Im.shape[1]

    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1
    )
    Im = wrap_boundary_liu(Im, tuple(target_size))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    N, M = Im.shape[:2]
    sizeI2D = (N, M)

    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    Normin1 = np.conj(KER) * fft2(S)

    beta = 2 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1] - S[:, -1:]], axis=1)
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :] - S[-1:, :]], axis=0)

        t = (h ** 2 + v ** 2) < lambda_grad / beta
        h[t] = 0.0
        v[t] = 0.0

        Normin2_val = np.concatenate([h[:, -1:] - h[:, 0:1],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :] - v[0:1, :],
             -np.diff(v, n=1, axis=0)], axis=0)

        FS = (Normin1 + beta * fft2(Normin2_val)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa

    S = S[:H_orig, :W_orig]
    return S



def ringing_artifacts_removal(y, kernel, lambda_tv=2e-3, lambda_l0=2e-3,
                              weight_ring=0.5):
    """
    Подавление эффекта звона (ringing artifacts) после неслепой деконволюции.

    Комбинирует результаты TV-деконволюции и L0-деконволюции, применяя 
    билатеральный фильтр к их разности для выявления и вычитания звона 
    при сохранении резкости краев объектов.

    Параметры
    ---------
    y : ndarray
        Размытое изображение.
    kernel : ndarray
        Ядро размытия.
    lambda_tv : float
        Вес TV-деконволюции.
    lambda_l0 : float
        Вес L0-деконволюции.
    weight_ring : float
        Коэффициент интенсивности подавления звона (0 соответствует чистой TV-деконволюции).

    Возвращает
    ----------
    result : ndarray
        Восстановленное изображение со сниженным уровнем краевых артефактов.
    """
    H, W = y.shape[:2]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W]

    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
