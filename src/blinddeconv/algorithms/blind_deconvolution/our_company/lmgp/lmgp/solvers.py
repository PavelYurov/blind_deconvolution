"""
solvers.py

Оптимизационные методы и алгоритмы для решения задачи слепой деконволюции 
на основе локального максимального градиента.

Основано на методе:
    L. Chen, F. Fang, T. Wang, G. Zhang: "Blind Image Deblurring
    With Local Maximum Gradient Prior", CVPR, 2019.

Модуль содержит реализации алгоритмов многомасштабной оценки ядра размытия,
субградиентных методов решения целевых функционалов с L0- и L1-регуляризаторами,
а также конвейер финальной неслепой деконволюции с подавлением эффектов звона.
Модель опирается на схему полуквадратичного расщепления (Half-Quadratic Splitting)
и метод множителей Лагранжа (ADMM).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

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
)


def _graythresh(img):
    """
    Вычисление глобального порога бинаризации по методу Оцу.

    Алгоритм минимизирует внутриклассовую дисперсию путем поиска порога
    на основе 256-уровневой гистограммы нормированного изображения.

    Параметры
    ---------
    img : ndarray
        Входное изображение произвольной размерности.

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


def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size):
    """
    Оценка ядра искажения в пространстве градиентов.

    Формирует систему линейных уравнений в частотной области и решает ее
    методом сопряженных градиентов на основе градиентных карт текущей
    оценки скрытого изображения и входного размытого изображения.

    Параметры
    ---------
    blurred_x, blurred_y : ndarray
        Градиентные карты искаженного изображения.
    latent_x, latent_y : ndarray
        Градиентные карты скрытого изображения.
    weight : float
        Коэффициент регуляризации Тихонова (норма L2 ядра).
    psf_size : tuple
        Размерность оцениваемого ядра размытия.

    Возвращаемое значение
    ---------------------
    psf : ndarray
        Матрица восстановленного ядра размытия с отсеченным шумом и
        единичной нормой суммы элементов.
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
        'lambda': weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum
    return psf


def L0_LMG_deblur(Im, kernel, lambda_lmg, wei_grad, kappa=2.0):
    """
    Восстановление промежуточного скрытого изображения с использованием
    априорного знания локального максимального градиента и L0-регуляризации.

    Решает оптимизационную задачу:
    min_S ||S * K - B||_2^2 + lambda_lmg * ||2 - G_S(S)||_1 + wei_grad * ||nabla S||_0

    Решение строится на базе схемы полуквадратичного расщепления. Внешний цикл
    обновляет оператор G_S и выполняет мягкое пороговое отсечение вектора невязок.
    Внутренний цикл попеременно решает систему линейных уравнений для попиксельной
    оценки изображения, применяет жесткое L0-отсечение к градиентам и обновляет
    общую оценку сигнала в частотной области.

    Параметры
    ---------
    Im : ndarray
        Искаженное изображение с заполненными краевыми областями размерности (H, W).
    kernel : ndarray
        Матрица ядра искажения.
    lambda_lmg : float
        Весовой коэффициент регуляризатора локального максимального градиента.
    wei_grad : float
        Весовой коэффициент L0-регуляризатора градиентов.
    kappa : float
        Множитель шага штрафного параметра бета для схемы расщепления.

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

    gt = _graythresh(S ** 2)
    if gt == 0:
        gt = 1e-10 
    mybeta_pixel = lambda_lmg / gt

    n_pixels = rows * cols

    for _outer in range(4):
        J, A = LMG(S, patch_size)

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

    Параметры
    ---------
    Im : ndarray
        Искаженное изображение оригинального размера размерности (H, W).
    kernel : ndarray
        Матрица ядра искажения.
    lambda_grad : float
        Весовой коэффициент L0-регуляризатора градиентов.
    kappa : float
        Множитель шага штрафного параметра бета для схемы расщепления.

    Возвращаемое значение
    ---------------------
    S : ndarray
        Восстановленное скрытое изображение оригинального размера.
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


def blind_deconv_main(blur_B, k, lambda_lmg, lambda_grad, threshold, opts):
    """
    Выполнение одной итерации слепой деконволюции на фиксированном масштабе.

    Осуществляет цикл попеременной оптимизации функции рассеяния точки и
    скрытого изображения. Включает адаптивное пороговое отсечение шумовых
    градиентов и очистку ядра от артефактов на основе анализа связных компонент.

    Параметры
    ---------
    blur_B : ndarray
        Искаженное изображение на текущем масштабном уровне.
    k : ndarray
        Текущая оценка функции рассеяния точки.
    lambda_lmg : float
        Текущий вес регуляризатора локального максимального градиента.
    lambda_grad : float
        Текущий вес L0-регуляризатора градиентов.
    threshold : float
        Порог отсечения градиентов, обновляемый на каждой итерации.
    opts : dict
        Словарь настроек, содержащий количество итераций оптимизации xk_iter.

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

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    xk_iter = opts.get('xk_iter', 5)

    for _iter in range(xk_iter):
        if lambda_lmg == 0:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S = L0_LMG_deblur(blur_B_w, k, lambda_lmg, lambda_grad, 2.0)
            S = S[:H, :W]

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        k_prev = k.copy()

        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

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

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_lmg, lambda_grad, S


def _init_kernel(minsize):
    """
    Инициализация матрицы функции рассеяния точки на самом грубом уровне
    масштабной пирамиды путем задания горизонтального штриха в геометрическом центре.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    c = (minsize - 1) // 2      
    r = c - 1                   
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Понижающее дискретизирование изображения с предварительным гауссовым сглаживанием
    для предотвращения эффектов пространственного алиасинга.
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
    Корректировка размерности матрицы ядра искажения до требуемых значений
    путем отсечения или симметричного дополнения нулевыми строками и столбцами
    с учетом положения энергетического центра.
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
    Масштабирование функции рассеяния точки при переходе между уровнями
    пирамиды с последующей нормализацией и фиксацией размерности.
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_lmg, lambda_grad, opts):
    """
    Многомасштабная итеративная слепая оценка ядра размытия и промежуточного
    скрытого изображения.

    Формирует пирамиду разрешений на основе заданного размера функции рассеяния точки.
    Оценка ядра передается от грубых масштабов к детальным (coarse-to-fine),
    последовательно уточняясь на каждом уровне. На финальном масштабе применяется
    порог жесткого отсечения шума ядра.

    Параметры
    ---------
    y : ndarray
        Входное искаженное изображение в градациях серого.
    lambda_lmg : float
        Начальный весовой коэффициент для регуляризации локального максимального градиента.
    lambda_grad : float
        Начальный весовой коэффициент для L0-регуляризации градиентов.
    opts : dict
        Словарь конфигурационных параметров, содержащий целевой размер ядра,
        параметр гамма-коррекции и количество итераций на уровень.

    Возвращаемое значение
    ---------------------
    kernel : ndarray
        Финальная оценка ядра размытия на исходном масштабе изображения.
    interim_latent : ndarray
        Промежуточное скрытое изображение, полученное на финальной итерации.
    """
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = y ** gamma_correct

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

        print(f'Processing scale {s_idx + 1}/{num_scales}; '
              f'kernel size {k1list[s_idx]}x{k2list[s_idx]}; '
              f'image size {ys.shape[0]}x{ys.shape[1]}')

        if s_idx == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        ks, lambda_lmg, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_lmg, lambda_grad, threshold, opts
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


def _computeDenominator(y, k):
    """
    Вычисление спектральных знаменателей для метода попеременных направлений (ADMM).
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
    множителей Лагранжа на основе разделения переменных (Split Bregman).

    Параметры
    ---------
    B : ndarray
        Искаженное полутоновое изображение.
    k : ndarray
        Матрица функции рассеяния точки.
    lambda_tv : float
        Коэффициент регуляризации полной вариации.
    alpha : float
        Параметр нормы. Поддерживается только режим мягкого отсечения (alpha = 1).

    Возвращаемое значение
    ---------------------
    I : ndarray
        Деконволированное изображение.
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



def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    """
    Подавление пространственных артефактов на этапе финальной неслепой деконволюции.

    Объединяет результаты деконволюции на базе полной вариации (с сохранением гладкости)
    и L0-градиентов (с сохранением резких структур). Разностная компонента
    двух решений подвергается двусторонней фильтрации для извлечения информации
    об артефактах звона, которая затем вычитается из финального изображения.

    Параметры
    ---------
    y : ndarray
        Входное искаженное изображение.
    kernel : ndarray
        Предварительно оцененная функция рассеяния точки.
    lambda_tv : float
        Вес регуляризации полной вариации.
    lambda_l0 : float
        Вес L0-регуляризации градиентов.
    weight_ring : float
        Коэффициент силы подавления артефактов (0.0 отключает дополнительную обработку).

    Возвращаемое значение
    ---------------------
    result : ndarray
        Финальное восстановленное изображение с подавленным эффектом звона.
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
