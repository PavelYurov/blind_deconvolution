"""
non_blind.py

Неслепая деконволюция.

Основные идеи реализации:
    1. Lp-регуляризация для градиентов изображения (гиперлапласиан, alpha от 0 до 2).
    2. Функция потерь на базе Lp-нормы для шумового члена (alpha_n оценивается через KL-дивергенцию).
    3. Пространственно-вариативная карта весов регуляризации lambda(x,y), 
       вычисляемая на основе локальной статистики градиентов и оценки 
       стандартного отклонения шума.
    4. Одномерная интерполяция параметра lambda: предварительно строится 
       библиотека восстановленных изображений для геометрической сетки значений lambda, 
       затем выполняется попиксельная интерполяция.
    5. Двухэтапный конвейер: первый проход выполняется с alpha_n = alpha 
       (априорная модель градиентов), второй проход использует оцененный 
       через KL-дивергенцию параметр alpha_n.

Зависимости: numpy, scipy, pywt (PyWavelets).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.stats import entropy

__all__ = ['adaptive_lp_deconv', 'ringing_removal']


# --- Подавление артефактов звона (ringing_removal) ---

def ringing_removal(blurred, kernel, lambda_tv=3e-3, lambda_l0=5e-4,
                    weight_ring=1.0):
    """Обертка над функцией ringing_artifacts_removal.

    Используется отложенный импорт для избежания лишней загрузки зависимостей, 
    модуль загружается только при фактическом вызове функции.
    """
    from blinddeconv.algorithms.blind_deconvolution.our_company.dark_channel_prior\
        .dcp_with_denoiser.solvers import ringing_artifacts_removal
    return ringing_artifacts_removal(
        blurred, kernel,
        float(lambda_tv), float(lambda_l0), float(weight_ring),
    )


# --- Проксимальные операторы (на основе интерполяционных таблиц LUT) ---

_LUT_RANGE = 10
_LUT_STEP = 0.0001
_XX = np.arange(-_LUT_RANGE, _LUT_RANGE + _LUT_STEP, _LUT_STEP)


def _compute_w1(v, beta):
    """Мягкое ограничение порогом (Soft-thresholding) при alpha = 1."""
    return np.sign(v) * np.maximum(np.abs(v) - 1.0 / beta, 0.0)


def _compute_w23(v, beta):
    """Метод Феррари для alpha = 2/3."""
    eps = 1e-6
    m = np.full_like(v, 8.0 / (27.0 * beta ** 3))
    t1 = (-9.0 / 8.0) * v ** 2
    t2 = (1.0 / 4.0) * v ** 3
    t3 = (-1.0 / 8.0) * m * v ** 2
    t4 = -t3 / 2.0 + np.sqrt(
        (-m ** 3 / 27.0 + m ** 2 * v ** 4 / 256.0).astype(np.complex128))
    t5 = t4 ** (1.0 / 3.0)
    t6 = 2.0 * (-5.0 / 18.0 * t1 + t5 + m / (3.0 * t5))
    t7 = np.sqrt((t1 / 3.0 + t6).astype(np.complex128))

    root = np.zeros(v.shape + (4,), dtype=np.complex128)
    root[:, 0] = 0.75 * v + (
        t7 + np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 1] = 0.75 * v + (
        t7 - np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 2] = 0.75 * v + (
        -t7 + np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0
    root[:, 3] = 0.75 * v + (
        -t7 - np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0

    c1 = np.abs(np.imag(root)) < eps
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 &= (c23 > 0.5 * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)


def _compute_w12(v, beta):
    """Метод Кардано для alpha = 1/2."""
    eps = 1e-6
    m = -np.sign(v) / (4.0 * beta ** 2)
    t1 = (2.0 / 3.0) * v
    inner = (27.0 * m ** 2 + 4.0 * m * v ** 3).astype(np.complex128)
    t2 = (-27.0 * m - 2.0 * v ** 3 + 3.0 ** 1.5 * np.sqrt(inner)) ** (1.0 / 3.0)
    t2 = np.where(np.abs(t2) < eps, eps, t2)
    t3 = v ** 2 / t2

    root = np.zeros(v.shape + (3,), dtype=np.complex128)
    root[:, 0] = t1 + (2 ** (1.0 / 3.0)) / 3.0 * t3 + t2 / (3.0 * 2 ** (1.0 / 3.0))
    root[:, 1] = (
        t1
        - ((1.0 + 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 - 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)
    root[:, 2] = (
        t1
        - ((1.0 - 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 + 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)

    root = np.where(np.isfinite(root), root, 0)
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 = np.abs(np.imag(root)) < eps
    c1 &= (c23 > (2.0 / 3.0) * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)


def _newton_w(v, beta, alpha):
    """Метод Ньютона для произвольного значения alpha."""
    w = v.copy().astype(np.float64)
    for _ in range(4):
        df = alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)
        ddf = alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta
        w -= df / ddf
    w = np.where(np.isfinite(w), w, 0)
    cost0 = (beta / 2.0) * v ** 2
    costw = np.abs(w) ** alpha + (beta / 2.0) * (w - v) ** 2
    return np.where(costw < cost0, w, 0)


def _compute_w(v, beta, alpha):
    eps = 1e-9
    if abs(alpha - 1.0) < eps:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < eps:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < eps:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)


_lut_cache = {}


def _solve_img(v, beta, alpha):
    """Применение проксимального оператора через интерполяционную таблицу (LUT)."""
    key = (beta, alpha)
    if key not in _lut_cache:
        _lut_cache[key] = _compute_w(_XX, beta, alpha)
    lut = _lut_cache[key]
    return np.interp(v.ravel(), _XX, lut).reshape(v.shape)


def _clear_lut_cache():
    _lut_cache.clear()


# --- Вспомогательные функции БПФ ---

def _psf2otf(psf, shape):
    """Функция рассеяния точки (PSF) в оптическую передаточную функцию (OTF)."""
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


# --- Базовый алгоритм деконволюции ADMM ---

def _fast_deconv_adaptive(yin, kernel, alpha, alpha_n, lam):
    """
    Неслепая деконволюция с априорным распределением Lp для градиентов 
    и Lp-функцией потерь для шума.

    Решает:
        min_x  lam * (||∇_x x||^alpha + ||∇_y x||^alpha)  +  ||H * x - y||^{alpha_n}

    через полуквадратичное расщепление (ADMM):
        - подзадача w_n: проксимальный оператор на невязку шума
        - подзадачи w_x, w_y: проксимальные операторы на градиенты
        - подзадача x: явное решение в частотной области

    Параметры
    ----------
    yin : 2D массив — размытое изображение [0, 1]
    kernel : 2D массив — ФРТ (сумма элементов равна 1)
    alpha : float — экспонента гиперлапласиана для градиентов
    alpha_n : float — экспонента гиперлапласиана для модели шума
    lam : float — вес регуляризации (скалярный, для всего изображения)

    Возвращает
    -------
    yout : 2D массив — восстановленное изображение
    """
    M, N = yin.shape

    K = _psf2otf(kernel, (M, N))
    Y = fft2(yin)
    Nomin1 = np.conj(K) * Y
    Denom1 = np.abs(K) ** 2

    gx = np.array([[1, -1]], dtype=np.float64)
    gy = np.array([[1], [-1]], dtype=np.float64)
    Gx = _psf2otf(gx, (M, N))
    Gy = _psf2otf(gy, (M, N))
    Denom2 = np.abs(Gx) ** 2 + np.abs(Gy) ** 2

    yout = yin.copy()

    youtx = np.roll(yout, -1, axis=1) - yout
    youty = np.roll(yout, -1, axis=0) - yout
    youtn = yin - np.real(ifft2(fft2(yout) * K))

    betas = np.geomspace(1, 2 ** 8, num=9)
    gamma = 1.0 / 50.0
    beta_n = betas * lam / gamma
    beta_g = betas

    for i in range(len(betas)):
        Wn = _solve_img(youtn, beta_n[i], alpha_n)
        Wx = _solve_img(youtx, beta_g[i], alpha)
        Wy = _solve_img(youty, beta_g[i], alpha)

        Wxx = np.roll(Wx, 1, axis=1) - Wx
        Wyy = np.roll(Wy, 1, axis=0) - Wy
        Wnn = np.real(ifft2(fft2(Wn) * np.conj(K)))

        W = -Wnn + gamma * (Wxx + Wyy)
        Denom = Denom1 + Denom2 * gamma
        Fyout = (fft2(W) + Nomin1) / Denom
        yout = np.real(ifft2(Fyout))
        yout = np.clip(yout, 0, 1)

        youtx = np.roll(yout, -1, axis=1) - yout
        youty = np.roll(yout, -1, axis=0) - yout
        youtn = yin - np.real(ifft2(fft2(yout) * K))

    return yout


# --- Оценка стандартного отклонения шума ---

def _dwt_hh(img):
    """Извлечение субполосы HH (диагональные детали) через двухуровневое DWT (db2)."""
    import pywt
    _, (_, _, HH) = pywt.dwt2(img, 'db2')
    return HH


def _local_std(grad_map, L=10):
    """Локальное стандартное отклонение с окном (2L+1)x(2L+1)."""
    win = 2 * L + 1
    k = np.ones((win, win), dtype=np.float64) / (win ** 2)
    ms_local = convolve2d(grad_map ** 2, k, mode='same', boundary='symm')
    return np.sqrt(ms_local)


def _x_grad(img):
    gx = np.array([[1, 0, -1]], dtype=np.float64)
    return convolve2d(img, gx, mode='same', boundary='symm')


def _y_grad(img):
    gy = np.array([[1, 0, -1]], dtype=np.float64).reshape(3, 1)
    return convolve2d(img, gy, mode='same', boundary='symm')


def _find_turning_point(sorted_std, M, N):
    """Поиск точки перегиба уровня шума в отсортированном массиве локальных отклонений."""
    original = sorted_std.copy()
    d = max(1, int(M * N / 2000))
    smooth = np.ones(2 * d + 1, dtype=np.float64)
    smooth[0] = 0
    smooth[d + 1:] = -1
    filtered = convolve1d(sorted_std, smooth, mode='reflect')

    ind = 0
    for i in range(1, len(filtered)):
        if filtered[i] < filtered[ind]:
            break
        ind = i - 1
    return original[ind]


def _estimate_noise_std(image):
    """
    Оценка стандартного отклонения шума (sigma) на одиночном изображении 
    с использованием локальной статистики градиентов на базе DWT.

    Возвращает
    -------
    sigma_n : float — оцененный уровень шума sigma.
    """
    M, N = image.shape
    HH = _dwt_hh(image)
    Bgx, Bgy = _x_grad(HH), _y_grad(HH)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)
    sort_sgx = np.sort(sgx.ravel())
    sort_sgy = np.sort(sgy.ravel())
    sgx_n = _find_turning_point(sort_sgx, M, N)
    sgy_n = _find_turning_point(sort_sgy, M, N)
    Eg = np.sum(np.array([1, 0, -1], dtype=np.float64) ** 2)
    sigma_n = np.sqrt((sgx_n ** 2 + sgy_n ** 2 + 1e-8) / Eg)
    return sigma_n


# --- Вычисление пространственно-вариативной карты lambda ---

def _compute_lambda_map(image, sigma_n, alpha):
    """
    Вычисление пространственно-вариативной карты весов регуляризации lambda(x,y) 
    на основе локальной статистики градиентов и оцененного шума sigma.

    Алгоритм:
      1. Выделение DWT-HH -> локальное стандартное отклонение градиентов -> точки перегиба sgx_n, sgy_n.
      2. Градиенты изображения -> локальное отклонение sgx, sgy.
      3. sigma_gsx = sqrt(sgx^2 - sgx_n^2),  sigma_gsy = sqrt(sgy^2 - sgy_n^2).
      4. lambda(x,y) = (sqrt(2 * sigma_n^2 / (sigma_gsx^2 + sigma_gsy^2)))^alpha.
    """
    eps = 1e-8
    M, N = image.shape

    HH = _dwt_hh(image)
    Bgx_hh, Bgy_hh = _x_grad(HH), _y_grad(HH)
    sgx_hh, sgy_hh = _local_std(Bgx_hh, 10), _local_std(Bgy_hh, 10)
    sgx_n = _find_turning_point(np.sort(sgx_hh.ravel()), M, N)
    sgy_n = _find_turning_point(np.sort(sgy_hh.ravel()), M, N)

    Bgx, Bgy = _x_grad(image), _y_grad(image)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)

    sigma_gsx_sq = sgx ** 2 - sgx_n ** 2
    sigma_gsx_sq[sigma_gsx_sq < eps] = eps
    sigma_gsx = np.sqrt(sigma_gsx_sq)

    sigma_gsy_sq = sgy ** 2 - sgy_n ** 2
    sigma_gsy_sq[sigma_gsy_sq < eps] = eps
    sigma_gsy = np.sqrt(sigma_gsy_sq)

    lam_map = (np.sqrt(2 * sigma_n ** 2 / (
        sigma_gsx ** 2 + sigma_gsy ** 2 + eps))) ** alpha
    return lam_map


# --- Оценка экспоненты шума alpha_n через KL-дивергенцию ---

def _estimate_alpha_n(blurred, restored, kernel, sigma_n):
    """
    Оценка экспоненты шума alpha_n путем минимизации KL-дивергенции между
    наблюдаемой невязкой шума и сгенерированным гиперлапласовским шумом.
    """
    import math

    noise_observed = blurred - restored

    threshold = 0.3
    best_alpha, best_kl = 0.5, np.inf

    for i in range(1, 10):
        alpha_n = round(0.1 * i, 2)

        rng = np.random.default_rng(0)
        beta_hl = sigma_n * np.sqrt(
            math.gamma(1.0 / alpha_n) / math.gamma(3.0 / alpha_n))
        T = rng.gamma(shape=1.0 / alpha_n, scale=1.0, size=blurred.shape)
        S = rng.choice([-1.0, 1.0], size=blurred.shape)
        noise_ref = beta_hl * S * (T ** (1.0 / alpha_n))

        mask = (restored >= threshold) & (restored <= 1.0 - threshold)
        noise_sample = noise_observed[mask]
        noise_ref_masked = noise_ref[mask]

        if noise_sample.size < 100:
            continue

        dx = 0.01
        bins = np.arange(-threshold, threshold + dx, dx)
        hist_s, _ = np.histogram(noise_sample, bins)
        hist_s = hist_s.astype(np.float64) / hist_s.sum() + 1e-12
        hist_r, _ = np.histogram(noise_ref_masked, bins)
        hist_r = hist_r.astype(np.float64) / hist_r.sum() + 1e-12

        kl = entropy(hist_s, hist_r)
        if kl < best_kl:
            best_alpha, best_kl = alpha_n, kl

    return best_alpha


# --- Библиотека lambda и одномерная интерполяция ---

def _build_lambda_library(alpha, C, lam_N):
    """Построение геометрической сетки значений lambda."""
    i = np.arange(lam_N, dtype=np.float64)
    return C * (2 ** ((alpha / 3.0) * i))


def _interpolate_library(blurred, kernel, alpha, alpha_n, lam_map, lam_library):
    """
    Построение библиотеки восстановленных изображений и попиксельная интерполяция.

    Для каждого значения lambda в библиотеке выполняется полная деконволюция
    (с внутренним зеркальным дополнением границ). Затем для каждого пикселя
    вычисляется взвешенное среднее двух ближайших lambda-изображений на основе
    пространственной карты lambda(x,y).
    """
    C = lam_library[0]

    I_library = {}
    sat = False
    prev_I = None
    for idx in range(len(lam_library)):
        if not sat:
            I_library[idx] = _deconv_with_padding(
                blurred, kernel, alpha, alpha_n, lam_library[idx])
        else:
            I_library[idx] = prev_I.copy()
            continue
        if prev_I is not None and np.array_equal(I_library[idx], prev_I):
            sat = True
        prev_I = I_library[idx].copy()

    M, N = lam_map.shape
    raw_idx = np.ceil((3.0 / alpha) * np.log2(
        np.maximum(lam_map / C, 1.0))).astype(int)
    raw_idx = np.clip(raw_idx, 0, len(lam_library) - 2)

    w_map = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            denom = np.log(lam_library[i + 1] / max(lam_library[i], 1e-12))
            if abs(denom) < 1e-12:
                w_map[m, n] = 0.5
            else:
                w_map[m, n] = (np.log(
                    lam_library[i + 1] / max(lam_map[m, n], 1e-12)
                ) / denom) ** 1.4

    I_opt = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            w = w_map[m, n]
            I_opt[m, n] = (
                w * I_library[i][m, n]
                + (1.0 - w) * I_library[i + 1][m, n])

    return np.clip(I_opt, 0, 1)


# --- Зеркальное дополнение краев ---

def _mirror_pad(image, pad):
    """Симметричное (зеркальное) дополнение краев изображения."""
    return np.pad(image, pad, mode='reflect')


def _mirror_unpad(image, pad, orig_shape):
    """Удаление зеркального дополнения краев."""
    return image[pad:pad + orig_shape[0], pad:pad + orig_shape[1]]


def _deconv_with_padding(blurimg, kernel, alpha, alpha_n, lam):
    """
    Деконволюция с внутренним зеркальным дополнением краев.
    Дополняет изображение, запускает ADMM и затем обрезает края.
    """
    M, N = blurimg.shape
    k_size = kernel.shape[0]
    padded = _mirror_pad(blurimg, k_size)
    result = _fast_deconv_adaptive(padded, kernel, alpha, alpha_n, lam)
    return result[k_size:k_size + M, k_size:k_size + N]


# --- Публичный API ---

def adaptive_lp_deconv(blurred, kernel, alpha=0.8, sigma_n=None,
                       two_stage=True):
    """
    Неслепая деконволюция с пространственно-вариативной Lp-регуляризацией
    и адаптивной моделью шума.

    Параметры
    ----------
    blurred : ndarray, форма HxW
        Размытое (и, возможно, зашумленное) полутоновое изображение, float64 [0, 1].
    kernel : ndarray, форма hxw
        Ядро размытия (PSF). Будет нормализовано на единичную сумму.
    alpha : float, опционально
        Экспонента гиперлапласиана для априорного распределения градиентов 
        (по умолчанию 0.8). Стандартный диапазон для естественных изображений: [0.5, 0.8].
    sigma_n : float или None, опционально
        Стандартное отклонение шума. Если None, будет оценено автоматически.
    two_stage : bool, опционально
        Если True (по умолчанию), запускает второй этап деконволюции 
        с оценкой параметра шума alpha_n на основе KL-дивергенции. 
        Если False, используется alpha_n = alpha.

    Возвращает
    -------
    restored : ndarray, форма HxW
        Восстановленное изображение, float64 [0, 1].
    """
    kernel = kernel.astype(np.float64)
    kernel = np.maximum(kernel, 1e-10)
    kernel /= kernel.sum()

    blurred = blurred.astype(np.float64)
    if blurred.max() > 1.0:
        blurred /= 255.0

    M, N = blurred.shape

    if sigma_n is None:
        sigma_n = _estimate_noise_std(blurred)
    sigma_n = max(sigma_n, 1e-8)

    lam_map = _compute_lambda_map(blurred, sigma_n, alpha)

    C = max(lam_map.min(), 1e-12)
    lam_N = int(np.ceil(3.0 / alpha * np.log2(
        max(lam_map.max() / C, 1.0))) + 2)
    lam_N = max(lam_N, 3)
    lam_library = _build_lambda_library(alpha, C, lam_N)

    alpha_n = alpha
    _clear_lut_cache()
    I_opt = _interpolate_library(
        blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    if two_stage:
        alpha_n = _estimate_alpha_n(blurred, I_opt, kernel, sigma_n)

        if sigma_n > 0.025 or alpha_n == 0.5:
            alpha_n = max(alpha_n, 0.6)
        center_val = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
        if center_val < 1e-4:
            alpha_n = 0.8

        _clear_lut_cache()
        I_opt = _interpolate_library(
            blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    return np.clip(I_opt, 0, 1)