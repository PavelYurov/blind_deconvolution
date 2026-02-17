"""
Вспомогательные функции для метода слепой деконволюции
на основе дробного порядка с PMP-априори.

Литература:
    Wu, T., Wan, S., Feng, C., Zhang, H., & Zeng, T. (2024).
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
     Meets Fractional-Order Method."
    Journal of Mathematical Imaging and Vision, 67(1), 2.
    DOI: 10.1007/s10851-024-01221-x

Содержимое модуля:
    - Вычисление коэффициентов Грюнвальда–Летникова (GL) для дробных производных
    - Построение частотных операторов дробного дифференцирования
    - Мягкое пороговое преобразование (soft-thresholding)
    - Изотропное векторное сжатие (vector shrinkage)
    - Вычисление PMP (Patch-wise Minimal Pixels) prior
    - Пирамида Гаусса для coarse-to-fine схемы
    - Утилиты для работы с ядрами размытия
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import minimum_filter, gaussian_filter, zoom
from scipy.signal import fftconvolve


# ─────────────────────────────────────────────────────────────────────────────
#  1. Дробные производные (Грюнвальд–Летников)
# ─────────────────────────────────────────────────────────────────────────────

def gl_coefficients(alpha: float, n: int) -> np.ndarray:
    """
    Коэффициенты дробной производной Грюнвальда–Летникова (GL) порядка alpha.

    Определение:
        w_0 = 1
        w_k = (1 - (alpha + 1) / k) * w_{k-1},  k = 1, 2, ..., n-1

    Эквивалентно:
        w_k = (-1)^k * C(alpha, k) = (-1)^k * Gamma(alpha + 1) / (k! * Gamma(alpha - k + 1))

    Parameters
    ----------
    alpha : float
        Порядок дробной производной (типично 1 < alpha < 2 для регуляризации).
    n : int
        Число коэффициентов.

    Returns
    -------
    w : np.ndarray, shape (n,)
        Массив GL-коэффициентов.

    References
    ----------
    Podlubny, I. (1999). Fractional Differential Equations. Academic Press.
    Oldham, K. B. & Spanier, J. (1974). The Fractional Calculus. Academic Press.
    """
    w = np.zeros(n, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = (1.0 - (alpha + 1.0) / k) * w[k - 1]
    return w


def fft_fractional_operators(shape: tuple, alpha: float, truncation: int = None):
    """
    Построение частотных представлений операторов дробного дифференцирования
    D_x^alpha и D_y^alpha для FFT-основанных солверов.

    Операторы строятся путём вычисления GL-коэффициентов, размещения их
    в массив размера shape и перехода в частотную область через fft2.

    Parameters
    ----------
    shape : tuple of (int, int)
        Размер изображения (H, W).
    alpha : float
        Порядок дробной производной.
    truncation : int or None
        Длина усечения ряда GL-коэффициентов. По умолчанию min(H, W).

    Returns
    -------
    F_Dx : np.ndarray, complex
        FFT горизонтального оператора дробной производной.
    F_Dy : np.ndarray, complex
        FFT вертикального оператора дробной производной.
    F_Dx_sq : np.ndarray, float
        |F_Dx|^2 — квадрат модуля.
    F_Dy_sq : np.ndarray, float
        |F_Dy|^2 — квадрат модуля.
    """
    H, W = shape
    if truncation is None:
        truncation = min(H, W)

    w = gl_coefficients(alpha, truncation)

    # Горизонтальный оператор: строка GL-коэффициентов
    kernel_x = np.zeros((H, W), dtype=np.float64)
    kernel_x[0, :min(W, truncation)] = w[:min(W, truncation)]

    # Вертикальный оператор: столбец GL-коэффициентов
    kernel_y = np.zeros((H, W), dtype=np.float64)
    kernel_y[:min(H, truncation), 0] = w[:min(H, truncation)]

    F_Dx = fft2(kernel_x)
    F_Dy = fft2(kernel_y)

    F_Dx_sq = np.abs(F_Dx) ** 2
    F_Dy_sq = np.abs(F_Dy) ** 2

    return F_Dx, F_Dy, F_Dx_sq, F_Dy_sq


def fft_gradient_operators(shape: tuple):
    """
    Построение FFT стандартных (целочисленных) операторов градиента
    для оценки ядра в градиентной области.

    Используются конечные разности с периодическими граничными условиями:
        dx: [1, -1]   (горизонтальная)
        dy: [1; -1]   (вертикальная)

    Parameters
    ----------
    shape : tuple of (int, int)
        Размер изображения (H, W).

    Returns
    -------
    F_dx : np.ndarray, complex
        FFT горизонтального градиентного оператора.
    F_dy : np.ndarray, complex
        FFT вертикального градиентного оператора.
    """
    H, W = shape

    dx = np.zeros((H, W), dtype=np.float64)
    dx[0, 0] = 1.0
    dx[0, -1] = -1.0

    dy = np.zeros((H, W), dtype=np.float64)
    dy[0, 0] = 1.0
    dy[-1, 0] = -1.0

    return fft2(dx), fft2(dy)


# ─────────────────────────────────────────────────────────────────────────────
#  2. Операторы проксимального отображения
# ─────────────────────────────────────────────────────────────────────────────

def soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Оператор мягкого порогового преобразования (shrinkage).

    S_t(x) = sign(x) * max(|x| - t, 0)

    Является проксимальным оператором l1-нормы:
        prox_{t ||.||_1}(x) = argmin_z { t||z||_1 + (1/2)||z - x||_2^2 }

    Parameters
    ----------
    x : np.ndarray
        Входной массив.
    threshold : float
        Пороговое значение t > 0.

    Returns
    -------
    result : np.ndarray
        Результат мягкого порогового преобразования.
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def vector_shrinkage(vx: np.ndarray, vy: np.ndarray, threshold: float):
    """
    Изотропное (векторное) сжатие для двумерного полного вариационного
    регуляризатора.

    Для каждого пикселя i с вектором v_i = (vx_i, vy_i):
        factor_i = max(||v_i|| - t, 0) / max(||v_i||, eps)
        (sx_i, sy_i) = factor_i * (vx_i, vy_i)

    Это проксимальный оператор изотропного TV:
        sum_i sqrt(vx_i^2 + vy_i^2)

    Parameters
    ----------
    vx, vy : np.ndarray
        Компоненты двумерного векторного поля.
    threshold : float
        Пороговое значение t > 0.

    Returns
    -------
    sx, sy : np.ndarray
        Компоненты сжатого векторного поля.
    """
    magnitude = np.sqrt(vx ** 2 + vy ** 2)
    factor = np.maximum(magnitude - threshold, 0.0) / np.maximum(magnitude, 1e-10)
    return vx * factor, vy * factor


# ─────────────────────────────────────────────────────────────────────────────
#  3. PMP (Patch-wise Minimal Pixels) prior
# ─────────────────────────────────────────────────────────────────────────────

def patch_minimum(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Вычисление PMP-оператора (Patch-wise Minimal Pixels).

    Для каждой позиции i вычисляется минимальное абсолютное значение
    в окрестности размером patch_size x patch_size:
        M(f)_i = min_{j in P_i} |f_j|

    Аналогичен «тёмному каналу» (dark channel) для абсолютных значений.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение.
    patch_size : int
        Размер квадратного патча.

    Returns
    -------
    dark_channel : np.ndarray
        Карта поэлементных минимумов.

    References
    ----------
    Pan, J., Sun, D., Pfister, H., & Yang, M.-H. (2016).
    Blind Image Deblurring Using Dark Channel Prior. CVPR.

    He, K., Sun, J., & Tang, X. (2011).
    Single Image Haze Removal Using Dark Channel Prior. IEEE TPAMI.
    """
    return minimum_filter(np.abs(image), size=patch_size)


def pmp_weight_map(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Вычисление карты весов PMP prior для итеративно-перевзвешенного подхода.

    Пиксель i получает вес w_i = 1, если он является (приблизительно)
    минимальным в своём патче P_i, иначе w_i = 0.

    Это позволяет аппроксимировать PMP-регуляризатор как взвешенную l1-норму:
        R_PMP(f) ≈ sum_i w_i |f_i|

    Parameters
    ----------
    image : np.ndarray
        Входное изображение.
    patch_size : int
        Размер патча.

    Returns
    -------
    weights : np.ndarray
        Бинарная карта весов (0 или 1).
    """
    abs_image = np.abs(image)
    dark_channel = minimum_filter(abs_image, size=patch_size)
    weights = (abs_image <= dark_channel + 1e-6).astype(np.float64)
    return weights


# ─────────────────────────────────────────────────────────────────────────────
#  4. Пирамида Гаусса (coarse-to-fine)
# ─────────────────────────────────────────────────────────────────────────────

def build_gaussian_pyramid(image: np.ndarray, num_scales: int,
                           scale_factor: float = None) -> list:
    """
    Построение Гауссовой пирамиды изображений для coarse-to-fine оценки.

    На каждом уровне изображение уменьшается в scale_factor раз
    с предварительным Гауссовым сглаживанием (σ = factor/2)
    и билинейной интерполяцией.

    Parameters
    ----------
    image : np.ndarray
        Исходное размытое изображение (наивысшее разрешение).
    num_scales : int
        Число уровней пирамиды.
    scale_factor : float or None
        Коэффициент масштабирования между уровнями.
        По умолчанию sqrt(2) ≈ 1.414.

    Returns
    -------
    pyramid : list of np.ndarray
        Список изображений от грубого (мелкого) к мелкому (оригинальному).

    References
    ----------
    Cho, S. & Lee, S. (2009). Fast Motion Deblurring. SIGGRAPH Asia.
    """
    if scale_factor is None:
        scale_factor = np.sqrt(2)

    pyramid = []

    for s in range(num_scales - 1, -1, -1):
        factor = scale_factor ** s
        if s == 0:
            pyramid.append(image.copy())
        else:
            new_h = max(int(np.round(image.shape[0] / factor)), 16)
            new_w = max(int(np.round(image.shape[1] / factor)), 16)
            sigma = factor / 2.0
            smoothed = gaussian_filter(image, sigma=sigma)
            scaled = zoom(smoothed,
                          (new_h / image.shape[0], new_w / image.shape[1]),
                          order=1)
            pyramid.append(scaled)

    return pyramid


# ─────────────────────────────────────────────────────────────────────────────
#  5. Утилиты для ядра размытия
# ─────────────────────────────────────────────────────────────────────────────

def center_kernel_fft(kernel: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Размещение ядра PSF в массив размера target_size для FFT.

    Центр ядра переносится в позицию (0, 0) с периодической
    обёрткой (wrap-around), что соответствует циклической свёртке.

    Parameters
    ----------
    kernel : np.ndarray
        Компактное ядро размытия (kh x kw).
    target_size : tuple of (int, int)
        Целевой размер массива (H, W).

    Returns
    -------
    padded : np.ndarray
        Ядро, центрированное для FFT-свёртки.
    """
    H, W = target_size
    kh, kw = kernel.shape

    padded = np.zeros((H, W), dtype=np.float64)
    half_kh = kh // 2
    half_kw = kw // 2

    for i in range(kh):
        for j in range(kw):
            ii = (i - half_kh) % H
            jj = (j - half_kw) % W
            padded[ii, jj] = kernel[i, j]

    return padded


def crop_kernel_from_fft(kernel_full: np.ndarray, kernel_size: tuple) -> np.ndarray:
    """
    Извлечение компактного ядра из полноразмерного массива (после IFFT).

    Находит пиковый элемент и вырезает окрестность kernel_size,
    затем проецирует на допустимое множество (>=0, сумма=1).

    Parameters
    ----------
    kernel_full : np.ndarray
        Полноразмерное ядро (H x W).
    kernel_size : tuple of (int, int)
        Требуемый размер ядра (kh, kw).

    Returns
    -------
    kernel : np.ndarray
        Компактное ядро, спроецированное на симплекс.
    """
    H, W = kernel_full.shape
    kh, kw = kernel_size

    peak_y, peak_x = np.unravel_index(np.argmax(kernel_full), kernel_full.shape)

    kernel = np.zeros((kh, kw), dtype=np.float64)
    half_kh = kh // 2
    half_kw = kw // 2

    for i in range(kh):
        for j in range(kw):
            ii = (peak_y + i - half_kh) % H
            jj = (peak_x + j - half_kw) % W
            kernel[i, j] = kernel_full[ii, jj]

    kernel = np.maximum(kernel, 0.0)
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    else:
        kernel = np.ones((kh, kw), dtype=np.float64) / (kh * kw)

    return kernel


def resize_kernel(kernel: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Масштабирование ядра к новому размеру (для coarse-to-fine).

    Используется билинейная интерполяция с последующей
    проекцией на допустимое множество.

    Parameters
    ----------
    kernel : np.ndarray
        Исходное ядро.
    new_size : tuple of (int, int)
        Новый размер (new_h, new_w).

    Returns
    -------
    resized : np.ndarray
        Масштабированное ядро (>=0, сумма=1).
    """
    old_h, old_w = kernel.shape
    new_h, new_w = new_size

    resized = zoom(kernel, (new_h / old_h, new_w / old_w), order=1)
    resized = np.maximum(resized, 0.0)
    if resized.sum() > 0:
        resized /= resized.sum()
    else:
        resized = np.ones(new_size, dtype=np.float64) / (new_h * new_w)

    return resized


def kernel_threshold(kernel: np.ndarray, threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Пороговая обработка ядра: обнуление малых элементов.

    Элементы ядра, значение которых меньше threshold_ratio * max(kernel),
    обнуляются. Затем ядро нормируется на единичную сумму.

    Parameters
    ----------
    kernel : np.ndarray
        Входное ядро.
    threshold_ratio : float
        Доля от максимума — порог обнуления.

    Returns
    -------
    kernel : np.ndarray
        Пороговое ядро (>=0, сумма=1).
    """
    max_val = kernel.max()
    if max_val > 0:
        kernel[kernel < threshold_ratio * max_val] = 0.0
    kernel = np.maximum(kernel, 0.0)
    if kernel.sum() > 0:
        kernel /= kernel.sum()
    return kernel


# ─────────────────────────────────────────────────────────────────────────────
#  6. Edge tapering (уменьшение артефактов граничных условий)
# ─────────────────────────────────────────────────────────────────────────────

def edgetaper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Ослабление граничных артефактов (edge tapering).

    Вычисляет автокорреляцию PSF и строит весовую карту alpha,
    затем выполняет линейную интерполяцию:
        result = alpha * image + (1 - alpha) * (image * kernel)

    Это снижает эффект «звона» (ringing) вблизи границ изображения,
    вызванный предположением циклической свёртки при FFT-деконволюции.

    Parameters
    ----------
    image : np.ndarray
        Входное изображение.
    kernel : np.ndarray
        Ядро размытия (PSF).

    Returns
    -------
    tapered : np.ndarray
        Изображение с ослабленными граничными артефактами.

    References
    ----------
    MATLAB, edgetaper function.
    """
    # Автокорреляция ядра
    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf /= acf.max()

    H, W = image.shape
    kh, kw = kernel.shape

    # 1D маргинальные автокорреляции (для строк и столбцов)
    acf_y = acf[:, acf.shape[1] // 2]
    acf_x = acf[acf.shape[0] // 2, :]

    # Построение карт альфа для строк и столбцов
    alpha_y = np.ones(H, dtype=np.float64)
    half_ky = len(acf_y) // 2
    for i in range(min(half_ky, H)):
        val = acf_y[half_ky - i]
        alpha_y[i] = min(alpha_y[i], val)
        alpha_y[H - 1 - i] = min(alpha_y[H - 1 - i], val)

    alpha_x = np.ones(W, dtype=np.float64)
    half_kx = len(acf_x) // 2
    for i in range(min(half_kx, W)):
        val = acf_x[half_kx - i]
        alpha_x[i] = min(alpha_x[i], val)
        alpha_x[W - 1 - i] = min(alpha_x[W - 1 - i], val)

    alpha = alpha_y[:, None] * alpha_x[None, :]

    # Интерполяция: оригинал * alpha + размытое * (1 - alpha)
    blurred = fftconvolve(image, kernel, mode='same')
    tapered = image * alpha + blurred * (1.0 - alpha)

    return tapered


# ─────────────────────────────────────────────────────────────────────────────
#  7. Оценка уровня шума
# ─────────────────────────────────────────────────────────────────────────────

def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    Оценка стандартного отклонения шума методом MAD (Median Absolute Deviation).

    Применяется лапласиан-фильтр, далее оценка по формуле Donoho:
        sigma = median(|L * image|) / 0.6745

    Parameters
    ----------
    image : np.ndarray
        Входное изображение (нормализованное к [0, 1]).

    Returns
    -------
    sigma : float
        Оценка стандартного отклонения шума.

    References
    ----------
    Donoho, D. L. (1995). De-noising by Soft-Thresholding.
    IEEE Trans. on Information Theory.
    """
    from scipy.ndimage import convolve
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    filtered = convolve(image, laplacian)
    sigma = np.median(np.abs(filtered)) / (0.6745 * np.sqrt(20.0))
    return max(sigma, 1e-4)


def compute_gradient(image: np.ndarray):
    """
    Вычисление градиента изображения (конечные разности с периодическими г.у.).

    Parameters
    ----------
    image : np.ndarray
        Входное изображение.

    Returns
    -------
    gx : np.ndarray
        Горизонтальный градиент.
    gy : np.ndarray
        Вертикальный градиент.
    """
    gx = np.roll(image, -1, axis=1) - image
    gy = np.roll(image, -1, axis=0) - image
    return gx, gy
