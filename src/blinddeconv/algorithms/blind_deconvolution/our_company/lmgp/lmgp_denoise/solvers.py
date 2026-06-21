"""
solvers.py

Модифицированные оптимизационные методы и алгоритмы для решения задачи 
слепой деконволюции на основе локального максимального градиента.

Основано на методе:
    L. Chen, F. Fang, T. Wang, G. Zhang: "Blind Image Deblurring
    With Local Maximum Gradient Prior", CVPR, 2019.

Модуль содержит расширенные реализации алгоритмов оценки ядра размытия и 
скрытого изображения. Введены механизмы робастности: L1-мягкое отсечение градиентов,
температурное сглаживание операции максимума, эквализация гистограмм для 
пространственных производных, опциональная регуляризация Тихонова для функции 
рассеяния точки, а также интеграция априорного знания паттерн-минимума (PMP) 
для финального восстановления. Решение целевых функционалов базируется на схеме 
полуквадратичного расщепления и методе множителей Лагранжа.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates, gaussian_filter

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    LMG,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
    guided_filter,
    find_min_pixels,
    nlm_filter,
    bm3d_filter,
)


def _graythresh(img):
    """
    Вычисление глобального порога бинаризации по методу Оцу.

    Алгоритм минимизирует внутриклассовую дисперсию путем поиска порога
    на основе 256-уровневой гистограммы нормированного изображения.

    Параметры
    ---------
    img : ndarray
        Входное изображение.

    Возвращаемое значение
    ---------------------
    threshold : float
        Рассчитанный порог в диапазоне от 0.0 до 1.0.
    """
    img = np.clip(np.asarray(img, dtype=np.float64).ravel(), 0.0, 1.0)

    nbins = 256
    indices = np.round(img * (nbins - 1)).astype(np.intp)
    indices = np.clip(indices, 0, nbins - 1)
    counts = np.bincount(indices, minlength=nbins).astype(np.float64)

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    bin_mids = np.arange(nbins, dtype=np.float64) / (nbins - 1)

    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_mids)
    mu_t = mu[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b_sq = np.where(np.isfinite(sigma_b_sq), sigma_b_sq, 0.0)

    idx = int(np.argmax(sigma_b_sq))
    return bin_mids[idx]



def _compute_Ax(x, p):
    """
    Вычисление произведения матрицы системы на вектор для алгоритма сопряженных градиентов
    в задаче оценки функции рассеяния точки.
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size,
                 kernel_reg_weight=0.0):
    """
    Оценка ядра искажения в пространстве градиентов методом сопряженных градиентов.

    Формирует систему линейных уравнений в частотной области на основе градиентных карт
    оценки скрытого изображения и входного искаженного изображения. Добавлена поддержка
    опциональной регуляризации Тихонова для подавления шумовых компонент в ядре.

    Параметры
    ---------
    blurred_x, blurred_y : ndarray
        Градиентные карты искаженного изображения.
    latent_x, latent_y : ndarray
        Градиентные карты скрытого изображения.
    weight : float
        Базовый коэффициент регуляризации ядра.
    psf_size : tuple
        Размерность оцениваемого ядра размытия.
    kernel_reg_weight : float
        Дополнительный весовой коэффициент L2-регуляризации для функции рассеяния точки.

    Возвращаемое значение
    ---------------------
    psf : ndarray
        Оцененная матрица функции рассеяния точки.
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(otf2psf(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape[:2],
        'psf_size': psf_size,
        'lambda': weight + kernel_reg_weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum
    return psf



def L0_LMG_deblur(Im, kernel, lambda_lmg, wei_grad, kappa=2.0,
                  lmg_denoise_eps=None, lmg_denoise_radius=2,
                  lmg_denoise_type='guided',
                  lmg_bilateral_sigma_s=2.0, lmg_bilateral_sigma_r=0.1,
                  lmg_bm3d_sigma=0.01, lmg_nlm_h=0.01,
                  use_soft_threshold=False,
                  softmax_tau=None):
    """
    Восстановление промежуточного скрытого изображения с использованием
    априорного знания локального максимального градиента и регуляризации.

    Решает оптимизационную задачу:
    min_S ||S * K - B||_2^2 + lambda_lmg * ||2 - G_S(S)||_1 + wei_grad * ||nabla S||_p
    где p=0 или p=1 в зависимости от флага use_soft_threshold.

    Расширенная версия включает опциональную пространственную фильтрацию 
    перед вычислением оператора максимума для повышения устойчивости к шуму,
    а также температурное сглаживание выбора экстремумов.

    Параметры
    ---------
    Im : ndarray
        Искаженное изображение с заполненными краевыми областями.
    kernel : ndarray
        Матрица ядра искажения.
    lambda_lmg : float
        Весовой коэффициент регуляризатора локального максимального градиента.
    wei_grad : float
        Весовой коэффициент регуляризатора градиентов.
    kappa : float
        Множитель шага штрафного параметра бета для схемы расщепления.
    lmg_denoise_eps : float или None
        Параметр регуляризации пространственного фильтра перед вычислением оператора.
    lmg_denoise_radius : int
        Радиус окна направленного фильтра.
    lmg_denoise_type : str
        Тип применяемого пространственного фильтра.
    lmg_bilateral_sigma_s : float
        Пространственное отклонение двустороннего фильтра.
    lmg_bilateral_sigma_r : float
        Амплитудное отклонение двустороннего фильтра.
    lmg_bm3d_sigma : float
        Уровень шума для BM3D фильтрации.
    lmg_nlm_h : float
        Сила сглаживания NLM фильтрации.
    use_soft_threshold : bool
        Переключение между жестким (L0) и мягким (L1) отсечением градиентов.
    softmax_tau : float или None
        Температурный параметр для сглаживания функции максимума.

    Возвращаемое значение
    ---------------------
    S : ndarray
        Восстановленное скрытое изображение.
    """
    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    rows, cols = Im.shape[:2]
    sizeI2D = (rows, cols)

    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    Normin1 = np.conj(KER) * fft2(S)

    patch_size = 35  

    def _denoise_for_lmg(img):
        if lmg_denoise_eps is not None and lmg_denoise_eps > 0:
            if lmg_denoise_type == 'bm3d':
                return bm3d_filter(img, sigma_psd=lmg_bm3d_sigma)
            elif lmg_denoise_type == 'nlm':
                return nlm_filter(img, h=lmg_nlm_h)
            elif lmg_denoise_type == 'bilateral':
                return bilateral_filter(img, lmg_bilateral_sigma_s,
                                        lmg_bilateral_sigma_r)
            else:
                return guided_filter(img, img, lmg_denoise_radius,
                                     lmg_denoise_eps)
        return img

    S_clean = _denoise_for_lmg(S)
    gt = _graythresh(S_clean ** 2)
    if gt == 0:
        gt = 1e-10
    mybeta_pixel = lambda_lmg / gt

    n_pixels = rows * cols

    for _outer in range(4):
        S_for_lmg = _denoise_for_lmg(S)
        J, A = LMG(S_for_lmg, patch_size, softmax_tau=softmax_tau)

        t = 2.0 - J
        t2 = lambda_lmg / (2.0 * mybeta_pixel)
        t3 = np.abs(t) - t2
        t3[t3 < 0] = 0.0
        u = np.sign(t) * t3

        alpha3 = mybeta_pixel * 2.0

        AtA = A.T @ A

        for _inner in range(4):
            lhs = mybeta_pixel * AtA + alpha3 * sparse.eye(n_pixels,
                                                           format='csr')
            rhs = (mybeta_pixel * A.T @ (2.0 - u.flatten(order='F'))
                   + alpha3 * S.flatten(order='F'))
            subsitute_I_vec = spsolve(lhs, rhs)
            subsitute_I = subsitute_I_vec.reshape((rows, cols), order='F')

            beta = 2.0 * wei_grad
            while beta < betamax:
                h = np.concatenate([np.diff(S, n=1, axis=1),
                                    S[:, 0:1] - S[:, -1:]], axis=1)
                v = np.concatenate([np.diff(S, n=1, axis=0),
                                    S[0:1, :] - S[-1:, :]], axis=0)

                if use_soft_threshold:
                    lam_half = wei_grad / (2.0 * beta)
                    h = np.sign(h) * np.maximum(np.abs(h) - lam_half, 0.0)
                    v = np.sign(v) * np.maximum(np.abs(v) - lam_half, 0.0)
                else:
                    th = h ** 2 < wei_grad / beta
                    tv = v ** 2 < wei_grad / beta
                    h[th] = 0.0
                    v[tv] = 0.0

                Normin2 = np.concatenate([h[:, -1:] - h[:, 0:1],
                                          -np.diff(h, n=1, axis=1)], axis=1)
                Normin2 = Normin2 + np.concatenate(
                    [v[-1:, :] - v[0:1, :],
                     -np.diff(v, n=1, axis=0)], axis=0)

                FS = ((Normin1
                       + beta * fft2(Normin2)
                       + alpha3 * fft2(subsitute_I))
                      / (Den_KER + beta * Denormin2 + alpha3))
                S = np.real(ifft2(FS))

                beta = beta * kappa

            alpha3 = alpha3 * 4.0

        mybeta_pixel = mybeta_pixel * 4.0

    return S


def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """
    Восстановление промежуточного скрытого изображения с использованием
    L0-регуляризации градиентов без учета априорного знания локального максимума.

    Решает оптимизационную задачу:
    min_S ||S * K - B||_2^2 + lambda_grad * ||nabla S||_0
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



def blind_deconv_main(blur_B, k, lambda_lmg, lambda_grad, threshold, opts,
                      iteration_callback=None):
    """
    Выполнение одной итерации слепой деконволюции на фиксированном масштабе.

    Осуществляет цикл попеременной оптимизации функции рассеяния точки и
    скрытого изображения. Включает дополнительные механизмы робастности:
    опциональное применение эквализации гистограммы только в тракте оценки ядра,
    предварительное гауссово сглаживание градиентов входного сигнала и 
    адаптивное пороговое отсечение.

    Параметры
    ---------
    blur_B : ndarray
        Искаженное изображение на текущем масштабном уровне.
    k : ndarray
        Текущая оценка функции рассеяния точки.
    lambda_lmg : float
        Текущий вес регуляризатора локального максимального градиента.
    lambda_grad : float
        Текущий вес регуляризатора градиентов.
    threshold : float
        Порог отсечения градиентов, обновляемый на каждой итерации.
    opts : dict
        Словарь настроек, содержащий параметры фильтрации и оптимизации.
    iteration_callback : callable, опционально
        Функция обратного вызова для экспорта промежуточных состояний.

    Возвращаемое значение
    ---------------------
    k : ndarray
        Обновленная оценка функции рассеяния точки.
    lambda_lmg, lambda_grad : float
        Скорректированные веса регуляризаторов.
    S : ndarray
        Промежуточная оценка скрытого изображения.
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H = blur_B.shape[0]
    W = blur_B.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(k.shape[:2]) - 1
    )
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target_size))
    blur_B_tmp = blur_B_w[:H, :W]

    kernel_eq_method = opts.get('kernel_eq', 'none')
    kernel_eq_params = opts.get('kernel_eq_params', None) or {}

    def _apply_kernel_eq(img):
        if kernel_eq_method in (None, 'none'):
            return img
        from skimage.exposure import equalize_adapthist, equalize_hist
        img_c = np.clip(img, 0.0, 1.0)
        if kernel_eq_method == 'clahe':
            return equalize_adapthist(
                img_c,
                clip_limit=kernel_eq_params.get('clip_limit', 0.003),
                nbins=kernel_eq_params.get('nbins', 256),
                kernel_size=kernel_eq_params.get('kernel_size', None),
            )
        elif kernel_eq_method == 'global':
            return equalize_hist(img_c)
        raise ValueError(
            f"Unknown kernel_eq='{kernel_eq_method}'. "
            f"Choose from: 'clahe', 'global', 'none'")

    blur_B_for_grad = _apply_kernel_eq(blur_B_tmp)

    Bx = convolve2d(blur_B_for_grad, dx, mode='valid')
    By = convolve2d(blur_B_for_grad, dy, mode='valid')

    grad_smooth_sigma = opts.get('grad_smooth_sigma', None)
    if grad_smooth_sigma is not None and grad_smooth_sigma > 0:
        Bx = gaussian_filter(Bx, sigma=grad_smooth_sigma)
        By = gaussian_filter(By, sigma=grad_smooth_sigma)

    xk_iter = opts.get('xk_iter', 5)
    denoise_eps = opts.get('denoise_eps', None)
    denoise_radius = opts.get('denoise_radius', 2)
    ensemble_denoise = opts.get('ensemble_denoise', False)
    denoise_type = opts.get('denoise_type', 'guided')
    denoise_bilateral_sigma_s = opts.get('denoise_bilateral_sigma_s', 2.0)
    denoise_bilateral_sigma_r = opts.get('denoise_bilateral_sigma_r', 0.1)
    denoise_bm3d_sigma = opts.get('denoise_bm3d_sigma', 0.01)
    denoise_nlm_h = opts.get('denoise_nlm_h', 0.01)
    lmg_denoise_eps = opts.get('lmg_denoise_eps', None)
    lmg_denoise_radius = opts.get('lmg_denoise_radius', 2)
    lmg_denoise_type = opts.get('lmg_denoise_type', 'guided')
    lmg_bilateral_sigma_s = opts.get('lmg_bilateral_sigma_s', 2.0)
    lmg_bilateral_sigma_r = opts.get('lmg_bilateral_sigma_r', 0.1)
    lmg_bm3d_sigma = opts.get('lmg_bm3d_sigma', 0.01)
    lmg_nlm_h = opts.get('lmg_nlm_h', 0.01)
    use_soft_threshold = opts.get('use_soft_threshold', False)
    softmax_tau = opts.get('softmax_tau', None)
    kernel_reg_weight = opts.get('kernel_reg_weight', 0.0)

    for _iter in range(xk_iter):
        if lambda_lmg == 0:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S = L0_LMG_deblur(blur_B_w, k, lambda_lmg, lambda_grad, 2.0,
                              lmg_denoise_eps=lmg_denoise_eps,
                              lmg_denoise_radius=lmg_denoise_radius,
                              lmg_denoise_type=lmg_denoise_type,
                              lmg_bilateral_sigma_s=lmg_bilateral_sigma_s,
                              lmg_bilateral_sigma_r=lmg_bilateral_sigma_r,
                              lmg_bm3d_sigma=lmg_bm3d_sigma,
                              lmg_nlm_h=lmg_nlm_h,
                              use_soft_threshold=use_soft_threshold,
                              softmax_tau=softmax_tau)
            S = S[:H, :W]

        S_for_grad = _apply_kernel_eq(S)

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S_for_grad, max(k.shape), threshold,
            denoise_eps=denoise_eps, denoise_radius=denoise_radius,
            ensemble_denoise=ensemble_denoise,
            denoise_type=denoise_type,
            bilateral_sigma_s=denoise_bilateral_sigma_s,
            bilateral_sigma_r=denoise_bilateral_sigma_r,
            bm3d_sigma=denoise_bm3d_sigma,
            nlm_h=denoise_nlm_h,
        )

        k_prev = k.copy()

        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape,
                         kernel_reg_weight=kernel_reg_weight)

        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        if lambda_lmg != 0:
            lambda_lmg = max(lambda_lmg / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        if iteration_callback is not None:
            iteration_callback({
                'iteration': _iter,
                'scale': opts.get('_current_scale', 0),
                'num_scales': opts.get('scales', 1),
                'kernel': k.copy(),
                'image': S,
                'metrics': {
                    'kernel_diff': float(np.linalg.norm(k - k_prev)),
                    'lambda_lmg': lambda_lmg,
                    'lambda_grad': lambda_grad,
                },
            })

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_lmg, lambda_grad, S



def _init_kernel(minsize):
    """
    Инициализация матрицы функции рассеяния точки на самом грубом уровне.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    c = (minsize - 1) // 2      
    r = c - 1                   
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Понижающее дискретизирование изображения с предварительным гауссовым сглаживанием.
    """
    if ret == 1:
        return I.copy()

    sig = (1.0 / np.pi) * ret
    g0 = np.arange(-50, 51, dtype=np.float64) * 2 * np.pi
    sf = np.exp(-0.5 * g0 ** 2 * sig ** 2)
    sf = sf / sf.sum()
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]

    sf_row = sf.reshape(1, -1)
    sf_col = sf.reshape(-1, 1)
    I_filtered = convolve2d(I, sf_row, mode='valid')
    I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    gx_0 = gx_grid - 1.0
    gy_0 = gy_grid - 1.0

    sI = map_coordinates(I_filtered,
                         [gy_0.ravel(), gx_0.ravel()],
                         order=1, mode='nearest')
    sI = sI.reshape(gy_grid.shape)

    return sI


def _fixsize(f, nk1, nk2):
    """
    Корректировка размерности матрицы ядра искажения до целевых значений.
    """
    k1, k2 = f.shape

    while k1 != nk1 or k2 != nk2:
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]

        if k1 < nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[:k1, :] = f
                f = tf
            else:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[1:k1 + 1, :] = f
                f = tf

        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]

        if k2 < nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, :k2] = f
                f = tf
            else:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, 1:k2 + 1] = f
                f = tf

        k1, k2 = f.shape

    return f


def _resizeKer(k, ret, k1, k2):
    """
    Масштабирование функции рассеяния точки при переходе между уровнями пирамиды.
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_lmg, lambda_grad, opts, iteration_callback=None):
    """
    Многомасштабная итеративная слепая оценка ядра размытия и промежуточного
    скрытого изображения.

    Формирует пирамиду разрешений. На каждом уровне оценка функции рассеяния точки
    уточняется с использованием попеременной оптимизации и фильтрации градиентов.

    Параметры
    ---------
    y : ndarray
        Входное искаженное изображение в градациях серого.
    lambda_lmg : float
        Базовый весовой коэффициент для регуляризации локального максимального градиента.
    lambda_grad : float
        Базовый весовой коэффициент для регуляризации градиентов.
    opts : dict
        Словарь конфигурационных параметров.
    iteration_callback : callable, опционально
        Функция для логирования метрик сходимости.

    Возвращаемое значение
    ---------------------
    kernel : ndarray
        Финальная оценка ядра размытия на исходном масштабе изображения.
    interim_latent : ndarray
        Промежуточное скрытое изображение.
    """
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = np.maximum(y, 0.0) ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)    
    k2list = k1list.copy()                  

    opts['scales'] = num_scales

    threshold = None
    ks = None
    interim_latent = None

    for s_idx in range(num_scales - 1, -1, -1):
        if s_idx == num_scales - 1:
            ks = _init_kernel(int(k1list[s_idx]))
        else:
            ks = _resizeKer(ks, 1.0 / ret,
                            int(k1list[s_idx]), int(k2list[s_idx]))

        cret = retv[s_idx]
        ys = _downSmpImC(y, cret)

        if s_idx == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        opts['_current_scale'] = s_idx

        ks, lambda_lmg, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_lmg, lambda_grad, threshold, opts,
            iteration_callback=iteration_callback,
        )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

        if s_idx == 0:
            kernel = ks.copy()
            k_thresh = opts.get('k_thresh', 0)
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            kernel = kernel / kernel.sum()

    return kernel, interim_latent



def deblur_tv_pmpr(Im, kernel, lambda_pmp, mu, opts):
    """
    Неслепое восстановление изображения с использованием комбинации полной
    вариации и априорного знания паттерн-минимума (PMP).

    Обеспечивает более высокую устойчивость к зашумлению при восстановлении
    сглаженных областей по сравнению с классическим методом L0-регуляризации.

    Параметры
    ---------
    Im : ndarray
        Искаженное изображение с заполненными краевыми областями.
    kernel : ndarray
        Оцененная функция рассеяния точки.
    lambda_pmp : float
        Весовой коэффициент для регуляризатора минимумов паттерна.
    mu : float
        Весовой коэффициент для L0-регуляризации градиентов.
    opts : dict
        Параметры алгоритма PMP.

    Возвращаемое значение
    ---------------------
    S : ndarray
        Восстановленное изображение.
    """
    S = Im.copy()
    alphamax = 1e5

    M, N = Im.shape[:2]
    sizeI2D = (M, N)

    otfFh = psf2otf(np.array([[1, -1]], dtype=np.float64), sizeI2D)
    otfFv = psf2otf(np.array([[1], [-1]], dtype=np.float64), sizeI2D)
    otfKER = psf2otf(kernel, sizeI2D)

    denKER = np.abs(otfKER) ** 2
    denGrad = np.abs(otfFh) ** 2 + np.abs(otfFv) ** 2

    Fk_FI = np.conj(otfKER) * fft2(Im)

    alpha = 2.0 * mu
    K = 3
    kappa = 2

    patch_r = opts.get('r', 3)
    current_scale = opts.get('s', 1)
    total_scales = opts.get('scales', 1)
    pmp_quantile = opts.get('pmp_quantile', 0.0)

    while alpha < alphamax:
        for _k in range(K):
            Z, Md = find_min_pixels(S, patch_r, quantile=pmp_quantile)
            z = Z[Md > 0]

            if current_scale < total_scales / 2.0:
                if z.size > 0:
                    lambdat = min(max(lambda_pmp, np.mean(np.abs(z))), 0.1)
                else:
                    lambdat = lambda_pmp
                Z[np.abs(Z) < lambdat] = 0.0
            else:
                Z = np.sign(Z) * np.maximum(Z - lambda_pmp, 0.0)

            S = S * (1.0 - Md) + Z * Md

            Gh = np.concatenate([np.diff(S, n=1, axis=1),
                                 S[:, 0:1] - S[:, -1:]], axis=1)
            Gv = np.concatenate([np.diff(S, n=1, axis=0),
                                 S[0:1, :] - S[-1:, :]], axis=0)

            t = (Gh ** 2 + Gv ** 2) < mu / alpha
            Gh[t] = 0.0
            Gv[t] = 0.0

            gh = np.concatenate([Gh[:, -1:] - Gh[:, 0:1],
                                 -np.diff(Gh, n=1, axis=1)], axis=1)
            gv = np.concatenate([Gv[-1:, :] - Gv[0:1, :],
                                 -np.diff(Gv, n=1, axis=0)], axis=0)

            Fs = (Fk_FI + alpha * fft2(gh + gv)) / (denKER + alpha * denGrad)
            S = np.real(ifft2(Fs))

        alpha = alpha * kappa

    return S


def _computeDenominator(y, k):
    """
    Вычисление спектральных знаменателей для метода попеременных направлений.
    """
    sizey = y.shape[:2]
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = (np.abs(psf2otf(np.array([[1, -1]], dtype=np.float64), sizey)) ** 2
              + np.abs(psf2otf(np.array([[1], [-1]], dtype=np.float64), sizey)) ** 2)
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B, k, lambda_tv, alpha):
    """
    Неслепая деконволюция с использованием анизотропной полной вариации методом
    множителей Лагранжа на основе разделения переменных.
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


def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring,
                              use_pmp_nonblind=False, pmp_lambda=0.1,
                              pmp_patch_r=3, pmp_quantile=0.0):
    """
    Подавление пространственных артефактов на этапе финальной неслепой деконволюции.

    Объединяет результаты деконволюции на базе полной вариации и альтернативного
    метода (классического L0 или PMP). Разностная компонента двух решений
    подвергается двусторонней фильтрации для формирования профиля артефактов звона.
    """
    H, W = y.shape[:2]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    # TV deblurring
    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    if use_pmp_nonblind:
        pmp_opts = {
            'r': pmp_patch_r,
            's': 1,
            'scales': 1,
            'pmp_quantile': pmp_quantile,
        }
        Latent_pmp = deblur_tv_pmpr(y_pad, kernel, pmp_lambda, lambda_l0, pmp_opts)
        Latent_second = Latent_pmp[:H, :W]
    else:
        Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
        Latent_second = Latent_l0[:H, :W]

    diff_img = Latent_tv - Latent_second
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
