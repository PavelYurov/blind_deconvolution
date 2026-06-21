"""
solvers.py

Основные функции решателей для алгоритма слепой деконволюции 
на основе априорного распределения экстремальных каналов (ECP).

Основано на методе:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao: "Image Deblurring via
    Extreme Channels Prior", CVPR, 2017.

Метод ECP расширяет подход на основе темного канала (DCP) за счет 
добавления симметричного условия для светлого канала. Светлый канал 
учитывается через инверсию изображения (1 - S) внутри подзадачи 
оценки скрытого изображения. Основная часть конвейера аналогична DCP, 
специфичные для ECP изменения находятся в функции L0Deblur_dark_channel_BD.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    dark_channel,
    assign_dark_channel_to_pixel,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
    graythresh,
)



def _compute_Ax(x, p):
    """
    Вычисление произведения матрицы на вектор для системы сопряженных градиентов.
    
    Реализует оператор: y = otf2psf(m * psf2otf(x)) + lambda * x
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size):
    """
    Оценка ядра размытия на основе градиентных изображений с использованием 
    метода сопряженных градиентов в частотной области.

    Решает оптимизационную задачу:
        min_k || grad(I) * k - grad(B) ||_2^2 + weight * || k ||_2^2

    Параметры
    ---------
    blurred_x, blurred_y : ndarray
        Градиентные изображения размытого входа.
    latent_x, latent_y : ndarray
        Градиентные изображения оцененного скрытого изображения.
    weight : float
        Весовой коэффициент L2-регуляризации ядра.
    psf_size : tuple
        Ожидаемый пространственный размер ядра (kh, kw).

    Возвращает
    ----------
    psf : ndarray
        Оцененное ядро размытия, прошедшее пороговую обработку 
        и нормализацию.
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
    psf = psf / psf.sum()
    return psf



def L0Deblur_dark_channel_BD(Im, kernel, lambda_dark, wei_grad, kappa=2.0):
    """
    Восстановление изображения с совместным использованием L0-регуляризации 
    темного канала, светлого канала и градиентов изображения (ECP I-подзадача).

    Решает оптимизационную задачу:
        min_S || S * k - Im ||_2^2 + lambda_dark * || D(S) ||_0 + 
              lambda_dark * || 1 - B(S) ||_0 + wei_grad * || grad(S) ||_0

    Отличия от алгоритма DCP:
    - Добавлена пиксельная подзадача для светлого канала (вычисляется как 1 - S).
    - В знаменателе шага обновления через БПФ используется удвоенный коэффициент 
      штрафа, так как присутствуют две вспомогательные переменные.
    - Используется увеличенный размер локального окна (45 вместо 35).
    - Установлен жесткий предел максимального штрафного коэффициента.

    Параметры
    ---------
    Im : ndarray
        Размытое изображение с дополненными границами размерности (N, M) или (N, M, D).
    kernel : ndarray
        Оцененное ядро размытия.
    lambda_dark : float
        Весовой коэффициент для L0-регуляризации экстремальных каналов.
    wei_grad : float
        Весовой коэффициент для L0-регуляризации градиентов.
    kappa : float, по умолчанию 2.0
        Множитель увеличения штрафного параметра.

    Возвращает
    ----------
    S : ndarray
        Восстановленное скрытое изображение.
    """
    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
        S = S[:, :, np.newaxis]
        Im = Im[:, :, np.newaxis]
        squeeze_out = True
    else:
        N, M, D = S.shape
        squeeze_out = False

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    dark_r = 45

    mybeta_pixel = lambda_dark / graythresh(S ** 2)
    maxbeta_pixel = 8

    while mybeta_pixel < maxbeta_pixel:
        J, J_idx = dark_channel(S, dark_r)
        u = J.copy()
        t = u ** 2 < lambda_dark / mybeta_pixel
        u[t] = 0.0
        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        BS = 1.0 - S
        BJ, BJ_idx = dark_channel(BS, dark_r)
        bu = BJ.copy()
        t = bu ** 2 < lambda_dark / mybeta_pixel
        bu[t] = 0.0
        bu = assign_dark_channel_to_pixel(BS, bu, BJ_idx, dark_r)

        beta = 2.0 * wei_grad
        while beta < betamax:
            Denormin = Den_KER + beta * Denormin2 + 2.0 * mybeta_pixel

            h = np.concatenate([np.diff(S, n=1, axis=1),
                                S[:, 0:1, :] - S[:, -1:, :]], axis=1)
            v = np.concatenate([np.diff(S, n=1, axis=0),
                                S[0:1, :, :] - S[-1:, :, :]], axis=0)

            if D == 1:
                t = (h ** 2 + v ** 2)[:, :, 0] < wei_grad / beta
                t = t[:, :, np.newaxis]
            else:
                t = np.sum(h ** 2 + v ** 2, axis=2) < wei_grad / beta
                t = np.tile(t[:, :, np.newaxis], (1, 1, D))
            h[t] = 0.0
            v[t] = 0.0

            Normin2_val = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                          -np.diff(h, n=1, axis=1)], axis=1)
            Normin2_val = Normin2_val + np.concatenate(
                [v[-1:, :, :] - v[0:1, :, :],
                 -np.diff(v, n=1, axis=0)], axis=0
            )

            if D == 1:
                u_3d = u[:, :, np.newaxis] if u.ndim == 2 else u
                bu_3d = bu[:, :, np.newaxis] if bu.ndim == 2 else bu
            else:
                u_3d = u if u.ndim == 3 else np.tile(u[:, :, np.newaxis], (1, 1, D))
                bu_3d = bu if bu.ndim == 3 else np.tile(bu[:, :, np.newaxis], (1, 1, D))

            FS = (Normin1
                  + beta * fft2(Normin2_val, axes=(0, 1))
                  + mybeta_pixel * fft2(u_3d, axes=(0, 1))
                  + mybeta_pixel * fft2(1.0 - bu_3d, axes=(0, 1))) / Denormin
            S = np.real(ifft2(FS, axes=(0, 1)))

            beta = beta * kappa
            if wei_grad == 0:
                break

        mybeta_pixel = mybeta_pixel * kappa

    if squeeze_out:
        S = S[:, :, 0]
    return S


def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """
    Восстановление изображения с использованием исключительно L0-нормы градиентов.

    Производит циклическое дополнение границ до оптимального для БПФ размера, 
    решает задачу минимизации методом HQS и обрезает результат до исходных размеров.

    Параметры
    ---------
    Im : ndarray
        Входное размытое изображение.
    kernel : ndarray
        Ядро размытия.
    lambda_grad : float
        Весовой коэффициент для регуляризации градиентов.
    kappa : float, по умолчанию 2.0
        Шаг увеличения штрафного параметра.

    Возвращает
    ----------
    S : ndarray
        Восстановленное изображение исходного размера.
    """
    orig_ndim = Im.ndim
    H_orig, W_orig = Im.shape[0], Im.shape[1]

    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1
    )
    Im = wrap_boundary_liu(Im, tuple(target_size))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
        S = S[:, :, np.newaxis]
        Im = Im[:, :, np.newaxis]
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :, :] - S[-1:, :, :]], axis=0)

        if D == 1:
            t = (h ** 2 + v ** 2)[:, :, 0] < lambda_grad / beta
            t = t[:, :, np.newaxis]
        else:
            t = np.sum(h ** 2 + v ** 2, axis=2) < lambda_grad / beta
            t = np.tile(t[:, :, np.newaxis], (1, 1, D))
        h[t] = 0.0
        v[t] = 0.0

        Normin2_val = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :, :] - v[0:1, :, :],
             -np.diff(v, n=1, axis=0)], axis=0)

        FS = (Normin1 + beta * fft2(Normin2_val, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))

        beta = beta * kappa

    S = S[:H_orig, :W_orig, :]
    if orig_ndim == 2:
        S = S[:, :, 0]
    return S



def blind_deconv_main_BDF(blur_B, k, lambda_dark, lambda_grad, threshold, opts):
    """
    Выполнение слепой деконволюции (вариант ECP) на одном уровне масштабной пирамиды.

    Включает в себя чередующиеся этапы:
    1. Оценка скрытого изображения (с применением экстремальных каналов и 
       L0-регуляризации, либо только градиентной регуляризации).
    2. Адаптивное пороговое ограничение градиентов скрытого изображения.
    3. Оценка ядра размытия в частотной области по градиентам с последующей 
       фильтрацией мелких компонент связности.
    4. Обновление (уменьшение) весовых коэффициентов регуляризации.

    Параметры
    ---------
    blur_B : ndarray
        Размытое изображение.
    k : ndarray
        Текущая оценка ядра.
    lambda_dark : float
        Вес априорного распределения экстремальных каналов.
    lambda_grad : float
        Вес априорного распределения градиентов.
    threshold : float
        Порог для фильтрации градиентов.
    opts : dict
        Словарь дополнительных параметров.

    Возвращает
    ----------
    k : ndarray
        Обновленная оценка ядра.
    lambda_dark : float
        Обновленный вес экстремальных каналов.
    lambda_grad : float
        Обновленный вес градиентов.
    S : ndarray
        Оцененное промежуточное скрытое изображение.
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

    S = None
    for _iter in range(xk_iter):
        if lambda_dark != 0:
            S = L0Deblur_dark_channel_BD(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

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

        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S


def _init_kernel(minsize):
    """
    Инициализация ядра размытия на самом грубом уровне масштабной пирамиды.
    Создается матрица нулей, в центр которой помещаются два элемента по 0.5.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    c = (minsize - 1) // 2 
    r = c - 1                  
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Понижающее масштабирование изображения с применением антиалиасингового 
    гауссовского фильтра перед билинейной интерполяцией.
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
    if I.ndim == 3:
        channels = []
        for c in range(I.shape[2]):
            tmp = convolve2d(I[:, :, c], sf_row, mode='valid')
            tmp = convolve2d(tmp, sf_col, mode='valid')
            channels.append(tmp)
        I_filtered = np.stack(channels, axis=2)
    else:
        I_filtered = convolve2d(I, sf_row, mode='valid')
        I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    gx_0 = gx_grid - 1.0
    gy_0 = gy_grid - 1.0

    if I_filtered.ndim == 3:
        channels = []
        for c in range(I_filtered.shape[2]):
            sI_ch = map_coordinates(I_filtered[:, :, c],
                                    [gy_0.ravel(), gx_0.ravel()],
                                    order=1, mode='nearest')
            channels.append(sI_ch.reshape(gy_grid.shape))
        sI = np.stack(channels, axis=2)
    else:
        sI = map_coordinates(I_filtered,
                             [gy_0.ravel(), gx_0.ravel()],
                             order=1, mode='nearest')
        sI = sI.reshape(gy_grid.shape)

    return sI


def _fixsize(f, nk1, nk2):
    """
    Корректировка пространственных размеров ядра до целевых значений 
    путем добавления или удаления строк и столбцов со стороны с 
    минимальной суммой элементов.
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
    Масштабирование ядра размытия с использованием бикубической интерполяции, 
    ограничением снизу нулем и корректировкой размера с последующей нормализацией.
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_dark, lambda_grad, opts):
    """
    Многомасштабная слепая деконволюция (метод ECP).

    Формирует пирамиду изображений и последовательно уточняет ядро размытия, 
    начиная с грубого масштаба и заканчивая оригинальным разрешением.

    Параметры
    ---------
    y : ndarray
        Полутоновое размытое изображение в формате float64 [0, 1].
    lambda_dark : float
        Начальный вес регуляризации для экстремальных каналов.
    lambda_grad : float
        Начальный вес регуляризации для градиентов изображения.
    opts : dict
        Словарь параметров конфигурации:
        - 'kernel_size' : целевой размер ядра.
        - 'gamma_correct' : гамма-коррекция входа.
        - 'xk_iter' : количество итераций на одном масштабе.
        - 'k_thresh' : порог обнуления шума в итоговом ядре.

    Возвращает
    ----------
    kernel : ndarray
        Финальная оценка ядра размытия.
    interim_latent : ndarray
        Промежуточная оценка скрытого изображения на самом точном уровне.
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
    kernel = None

    for s in range(num_scales - 1, -1, -1):
        if s == num_scales - 1:
            ks = _init_kernel(int(k1list[s]))
        else:
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[s]), int(k2list[s]))

        cret = retv[s]
        ys = _downSmpImC(y, cret)

        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main_BDF(
            ys, ks, lambda_dark, lambda_grad, threshold, opts
        )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

        if s == 0:
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
    Предварительное вычисление компонентов знаменателя для метода расщепления 
    Брэгмана (ADM) в частотной области.
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
    Неслепая TV-l2 деконволюция с использованием анизотропной полной вариации.

    Решает задачу с применением метода ADM (Split Bregman). 
    В рамках текущего конвейера поддерживается только вариант с alpha = 1 
    (мягкое пороговое ограничение).

    Параметры
    ---------
    B : ndarray
        Размытое изображение.
    k : ndarray
        Ядро размытия.
    lambda_tv : float
        Вес TV-регуляризатора.
    alpha : int
        Экспонента нормы (поддерживается только 1).

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

    Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                "deblurring_adm_aniso: only alpha=1 is used in the ECP pipeline"
            )

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                    -np.diff(Wy, n=1, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

        beta = beta / 2.0

    return I


def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    """
    Комплексное подавление артефактов звона после неслепой деконволюции.

    Конвейер обработки:
    1. Расширение границ изображения до оптимального для БПФ размера.
    2. Поканальная TV-деконволюция (алгоритм ADM).
    3. Если подавление звона отключено (weight_ring == 0), возвращается результат TV.
    4. Если подавление активно, выполняется параллельная L0-деконволюция.
    5. Разница между TV и L0 фильтруется билатеральным фильтром и вычитается из 
       TV-результата для устранения звона с сохранением текстур.

    Параметры
    ---------
    y : ndarray
        Размытое входное изображение.
    kernel : ndarray
        Оцененное ядро размытия.
    lambda_tv : float
        Вес TV-регуляризации.
    lambda_l0 : float
        Вес L0-регуляризации.
    weight_ring : float
        Коэффициент силы вычитания артефактов звона.

    Возвращает
    ----------
    result : ndarray
        Итоговое восстановленное изображение.
    """
    H, W = y.shape[0], y.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    if y_pad.ndim == 2:
        Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    else:
        channels = []
        for c in range(y_pad.shape[2]):
            channels.append(
                deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1)
            )
        Latent_tv = np.stack(channels, axis=2)

    if Latent_tv.ndim == 2:
        Latent_tv = Latent_tv[:H, :W]
    else:
        Latent_tv = Latent_tv[:H, :W, :]

    if weight_ring == 0:
        return Latent_tv
    Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)

    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
