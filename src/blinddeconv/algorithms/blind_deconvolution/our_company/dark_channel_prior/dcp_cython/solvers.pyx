"""
solvers.py

Основные функции решателей для алгоритма слепой деконволюции на основе 
априорного распределения темного канала (Dark Channel Prior, DCP). 
Модифицированная версия с поддержкой промежуточных обработчиков (hooks) 
и расширенным конвейером шумоподавления.

Основано на методе:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR, 2016.

Содержит:
    - estimate_psf : оценка функции рассеяния точки (PSF) методом сопряженных градиентов.
    - L0Deblur_dark_channel : восстановление изображения с L0-регуляризацией 
      интенсивности темного канала и градиентов.
    - L0Restoration : восстановление изображения только с L0-нормой градиентов.
    - blind_deconv_main : главный цикл слепой деконволюции для одного масштаба.
    - blind_deconv : многомасштабная (иерархическая) слепая деконволюция.
    - deblurring_adm_aniso : неслепая TV-l2 деконволюция на базе метода ADM.
    - ringing_artifacts_removal : комплексное подавление эффекта звона.
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
        min_k || grad(I) * k - grad(B) ||_2^2 + gamma * || k ||_2^2
    где grad(I) и grad(B) — градиенты скрытого и размытого изображений соответственно,
    а gamma — весовой коэффициент регуляризации ядра.

    Параметры
    ---------
    blurred_x, blurred_y : ndarray
        Градиентные изображения размытого входа размерности (M, N).
    latent_x, latent_y : ndarray
        Градиентные изображения оцененного скрытого изображения размерности (M, N).
    weight : float
        Весовой коэффициент L2-регуляризации ядра (параметр gamma).
    psf_size : tuple
        Ожидаемый пространственный размер ядра (kh, kw).

    Возвращает
    ----------
    psf : ndarray
        Оцененное ядро размытия размерности (kh, kw), прошедшее пороговую 
        обработку и нормализацию.
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



def L0Deblur_dark_channel(Im, kernel, lambda_dark, wei_grad, kappa=2.0):
    """
    Восстановление изображения с совместным использованием L0-регуляризации 
    интенсивности темного канала и L0-регуляризации градиентов изображения.

    Решает оптимизационную задачу:
        min_S || S * k - Im ||_2^2 + lambda_dark * || D(S) ||_0 + wei_grad * || grad(S) ||_0
    где D(S) — темный канал изображения S, извлекаемый с использованием 
    локального окна.

    Параметры
    ---------
    Im : ndarray
        Размытое изображение размерности (N, M) или (N, M, D) с уже 
        дополненными границами.
    kernel : ndarray
        Оцененное ядро размытия размерности (kh, kw).
    lambda_dark : float
        Вес L0-априорного распределения для темного канала.
    wei_grad : float
        Вес L0-априорного распределения для градиентов изображения.
    kappa : float, по умолчанию 2.0
        Множитель для обновления штрафного параметра ADM на каждой итерации.

    Возвращает
    ----------
    S : ndarray
        Восстановленное скрытое изображение размерности (N, M) или (N, M, D).
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

    dark_r = 35
    mybeta_pixel = lambda_dark / graythresh(S ** 2)
    maxbeta_pixel = 2 ** 3

    while mybeta_pixel < maxbeta_pixel:
        J, J_idx = dark_channel(S, dark_r)
        u = J.copy()

        t = u ** 2 < lambda_dark / mybeta_pixel
        u[t] = 0.0

        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        beta = 2 * wei_grad
        while beta < betamax:
            Denormin = Den_KER + beta * Denormin2 + mybeta_pixel

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
                 -np.diff(v, n=1, axis=0)], axis=0)

            FS = (Normin1 + beta * fft2(Normin2_val, axes=(0, 1))
                  + mybeta_pixel * fft2(u, axes=(0, 1))) / Denormin
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

    Решает оптимизационную задачу:
        min_S || S * k - Im ||_2^2 + lambda_grad * || grad(S) ||_0

    Параметры
    ---------
    Im : ndarray
        Размытое изображение в исходном размере.
    kernel : ndarray
        Оцененное ядро размытия.
    lambda_grad : float
        Весовой коэффициент для L0-априорного распределения градиентов.
    kappa : float, по умолчанию 2.0
        Множитель увеличения штрафного параметра ADM.

    Возвращает
    ----------
    S : ndarray
        Восстановленное изображение, обрезанное до исходного размера.
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

    beta = 2 * lambda_grad
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



def blind_deconv_main(blur_B, k, lambda_dark, lambda_grad, threshold, opts,
                      latent_hook=None, kernel_hook=None, scale_idx=0,
                      iteration_callback=None):
    """
    Выполнение слепой деконволюции на одном уровне масштабной пирамиды.

    В модифицированной версии добавлены функции обратного вызова (hooks), 
    позволяющие применять дополнительную фильтрацию или эквализацию к 
    скрытому изображению перед вычислением градиентов, а также сглаживать 
    ядро размытия. Это снижает влияние шума на оценку без изменения 
    основной оптимизационной модели.

    Параметры
    ---------
    blur_B : ndarray
        Размытое изображение размерности (H, W) или (H, W, D).
    k : ndarray
        Текущая оценка ядра размытия размерности (kh, kw).
    lambda_dark : float
        Вес для априорного распределения интенсивности темного канала.
    lambda_grad : float
        Вес для L0-априорного распределения градиентов.
    threshold : float или None
        Порог для фильтрации градиентов, который адаптивно обновляется.
    opts : dict
        Словарь с параметрами (например, 'xk_iter').
    latent_hook : callable или None
        Функция вида f(S, k, iter_idx, scale_idx) -> S, применяемая к скрытому 
        изображению перед пороговой обработкой градиентов.
    kernel_hook : callable или None
        Функция вида f(k, S, iter_idx, scale_idx) -> k, применяемая для 
        очистки и сглаживания ядра после его оценки.
    scale_idx : int
        Индекс текущего масштаба в пирамиде.
    iteration_callback : callable или None
        Функция логирования состояния итераций.

    Возвращает
    ----------
    k : ndarray
        Обновленная оценка ядра размытия.
    lambda_dark : float
        Обновленный вес для темного канала.
    lambda_grad : float
        Обновленный вес для градиентов.
    S : ndarray
        Оцененное скрытое промежуточное изображение.
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
        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        S_for_grad = latent_hook(S.copy(), k, _iter, scale_idx) if latent_hook is not None else S

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S_for_grad, max(k.shape), threshold
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

        if kernel_hook is not None:
            k = kernel_hook(k, S, _iter, scale_idx)
            k[k < 0] = 0.0
            if k.sum() > 0:
                k = k / k.sum()

        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        S = np.clip(S, 0.0, 1.0)

        if iteration_callback is not None:
            iteration_callback({
                'iteration': _iter,
                'scale': opts.get('_current_scale', scale_idx),
                'num_scales': opts.get('_num_scales', 1),
                'kernel': k.copy(),
                'image': S,
                'metrics': {
                    'kernel_diff': float(np.linalg.norm(k - k_prev)),
                    'lambda_dark': lambda_dark,
                    'lambda_grad': lambda_grad,
                },
            })

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S



def _init_kernel(minsize):
    """Инициализация ядра размытия на самом грубом уровне масштабной пирамиды."""
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
    Корректировка пространственных размеров ядра до целевых значений (nk1, nk2) 
    путем добавления или удаления строк и столбцов на основе минимальной суммы 
    элементов по краям.
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
    Масштабирование ядра размытия (увеличение разрешения при переходе 
    от грубого масштаба к более точному) с последующей корректировкой размеров.
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_dark, lambda_grad, opts,
                 latent_hook=None, kernel_hook=None,
                 iteration_callback=None):
    """
    Многомасштабная слепая деконволюция на основе пирамиды изображений.

    Включает в себя последовательную оценку ядра от низкого разрешения 
    к исходному (coarse-to-fine), с поддержкой функций обратного вызова 
    для шумоподавления и анализа промежуточных результатов.

    Параметры
    ---------
    y : ndarray
        Полутоновое размытое изображение размерности (H, W).
    lambda_dark : float
        Начальный вес регуляризации для темного канала.
    lambda_grad : float
        Начальный вес регуляризации для градиентов изображения.
    opts : dict
        Словарь параметров ('kernel_size', 'gamma_correct', 'xk_iter', 'k_thresh').
    latent_hook, kernel_hook : callable или None
        Функции обратного вызова, передаваемые во внутренний цикл.
    iteration_callback : callable или None
        Функция для логирования итераций.

    Возвращает
    ----------
    kernel : ndarray
        Окончательно оцененное ядро размытия.
    interim_latent : ndarray
        Промежуточная оценка скрытого изображения на самом точном уровне пирамиды.
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

    for s_idx in range(num_scales - 1, -1, -1):  # MATLAB: num_scales:-1:1
        s = s_idx 

        if s == num_scales - 1:
            ks = _init_kernel(int(k1list[s]))
        else:
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[s]), int(k2list[s]))

        cret = retv[s]
        ys = _downSmpImC(y, cret)

        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        opts['_current_scale'] = s_idx  # 0 = finest
        opts['_num_scales'] = num_scales

        ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_dark, lambda_grad, threshold, opts,
            latent_hook=latent_hook, kernel_hook=kernel_hook,
            scale_idx=s_idx,
            iteration_callback=iteration_callback,
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
    Предварительное вычисление знаменателя и части числителя для 
    решателя ADM в частотной области.
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
    Неслепая TV-l2 деконволюция с анизотропной полной вариацией.

    Решает задачу:
        min_I || I * k - B ||_2^2 + lambda_tv * || grad(I) ||_1
    с использованием метода расщепления Брэгмана (Split Bregman / ADM).

    Параметры
    ---------
    B : ndarray
        Размытое одноканальное изображение.
    k : ndarray
        Ядро размытия нечетного размера.
    lambda_tv : float
        Весовой коэффициент TV-регуляризатора.
    alpha : int
        Экспонента нормы. Для анизотропной TV поддерживается только значение 1 
        (мягкое пороговое ограничение).

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
                f"deblurring_adm_aniso: alpha={alpha} not implemented; only alpha=1 supported"
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
    Комплексное подавление артефактов звона после этапа неслепой деконволюции.

    Использует комбинацию двух методов восстановления:
    1. TV-деконволюция.
    2. L0-деконволюция (регуляризация на градиенты).
    Разница между результатами пропускается через билатеральный фильтр для 
    обнаружения высокочастотного звона и вычитается из TV-результата, 
    что позволяет сохранить резкость информативных краев.

    Параметры
    ---------
    y : ndarray
        Размытое изображение размерности (H, W) или (H, W, D).
    kernel : ndarray
        Оцененное ядро размытия.
    lambda_tv : float
        Коэффициент регуляризации для TV-деконволюции.
    lambda_l0 : float
        Коэффициент регуляризации для L0-деконволюции.
    weight_ring : float
        Сила вычитания артефактов звона (0 соответствует результату только 
        по методу TV).

    Возвращает
    ----------
    result : ndarray
        Восстановленное изображение без эффекта звона.
    """
    orig_ndim = y.ndim
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
            channels.append(deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1))
        Latent_tv = np.stack(channels, axis=2)

    Latent_tv = Latent_tv[:H, :W] if Latent_tv.ndim == 2 else Latent_tv[:H, :W, :]

    if weight_ring == 0:
        return Latent_tv

    if y_pad.ndim == 2:
        Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)
    else:
        Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)

    diff_img = Latent_tv - Latent_l0

    if diff_img.ndim == 2:
        bf_diff = bilateral_filter(diff_img, 3, 0.1)
    else:
        channels = []
        for c in range(diff_img.shape[2]):
            channels.append(bilateral_filter(diff_img[:, :, c], 3, 0.1))
        bf_diff = np.stack(channels, axis=2)

    result = Latent_tv - weight_ring * bf_diff
    return result
