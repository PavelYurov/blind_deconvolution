"""
utils.py

Вспомогательные вычислительные функции для алгоритма слепой деконволюции 
на основе логарифмического априорного распределения (LIP).

Основано на методах:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", EMMCVPR 2015.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from skimage.transform import resize as sk_resize



def convn_valid(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Двумерная свертка с отсечением краев (режим 'valid'). 
    Выходной размер составляет (M - MK + 1, N - NK + 1). Требует M >= MK, N >= NK.
    """
    return fftconvolve(u, k, mode='valid')


def convn_full(u: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Двумерная свертка с полным сохранением краев (режим 'full'). 
    Выходной размер составляет (M + MK - 1, N + NK - 1).
    """
    return fftconvolve(u, k, mode='full')


def pad_replicate(f: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Дополнение границ матрицы дублированием крайних значений (режим 'edge').
    Добавляет pad_h строк сверху и снизу, а также pad_w столбцов слева и справа.
    """
    return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


def shft(u: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Вычисление конечной разности со сдвигом и нулевыми граничными условиями.

    Внутри допустимой области вычисляется разность: 
        result[i, j] = u[i + dy, j + dx] - u[i, j],
    на границах значения полагаются равными нулю.

    Параметры
    ---------
    u : ndarray
        Входной двумерный массив размерности (M, N).
    dx : int
        Смещение по горизонтали (положительное значение соответствует сдвигу вправо).
    dy : int
        Смещение по вертикали (положительное значение соответствует сдвигу вниз).

    Возвращает
    ----------
    us : ndarray
        Массив конечных разностей размерности (M, N).
    """
    M, N = u.shape
    us = np.zeros_like(u)

    r0 = max(-dy, 0)
    r1 = min(M, M - dy)
    c0 = max(-dx, 0)
    c1 = min(N, N - dx)

    sr0 = max(dy, 0)
    sr1 = min(dy + M, M)
    sc0 = max(dx, 0)
    sc1 = min(dx + N, N)

    us[r0:r1, c0:c1] = u[sr0:sr1, sc0:sc1] - u[r0:r1, c0:c1]
    return us



def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Гамма-коррекция изображения по формуле: Ic = I^gamma."""
    return np.power(img, gamma)


def make_size_odd(f: np.ndarray) -> np.ndarray:
    """
    Приведение пространственных размеров изображения к нечетным значениям.
    Если измерение имеет четную длину, последняя строка или столбец отсекаются.
    """
    if f.shape[0] % 2 == 0:
        f = f[:-1, :]
    if f.shape[1] % 2 == 0:
        f = f[:, :-1]
    return f


def imresize_matlab(img: np.ndarray, target_shape: tuple,
                    order: int = 3) -> np.ndarray:
    """
    Масштабирование двумерного изображения до целевых размеров target_shape.

    Использует бикубическую интерполяции с предварительным гауссовским 
    сглаживанием (антиалиасингом) при уменьшении разрешения. При увеличении 
    разрешения сглаживание автоматически отключается.
    """
    th, tw = int(target_shape[0]), int(target_shape[1])
    oh, ow = img.shape[:2]

    if oh == th and ow == tw:
        return img.copy()

    return sk_resize(
        img, (th, tw),
        order=order,
        anti_aliasing=True,
        preserve_range=True,
        mode='edge',
    )



def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Преобразование функции рассеяния точки (PSF) в оптическую передаточную 
    функцию (OTF).

    Алгоритм:
    1. Дополнение матрицы PSF нулями до размеров shape.
    2. Циклический сдвиг матрицы так, чтобы центр PSF оказался в точке (0, 0).
    3. Вычисление двумерного быстрого преобразования Фурье.
    """
    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def get_gradient_operators(shape: tuple):
    """
    Формирование операторов конечных разностей (градиентов) по горизонтали 
    и вертикали в частотной области.

    Возвращает
    ----------
    OTF_dx, OTF_dy, conj(OTF_dx), conj(OTF_dy)
    """
    kx = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]], dtype=np.float64)
    ky = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]], dtype=np.float64)
    OTF_dx = psf2otf(kx, shape)
    OTF_dy = psf2otf(ky, shape)
    return OTF_dx, OTF_dy, np.conj(OTF_dx), np.conj(OTF_dy)


def wiener_filter(img: np.ndarray, kernel: np.ndarray,
                  noise_snr: float = 0.01) -> np.ndarray:
    """
    Неслепая деконволюция на основе фильтра Винера.

    Модель: f = k * u + n
    Решение в частотной области:
        U = conj(K) / (|K|^2 + snr) * F
    """
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (np.abs(otf) ** 2 + noise_snr)) * F_img
    return np.real(np.fft.ifft2(F_res))


def tikhonov_filter(img: np.ndarray, kernel: np.ndarray,
                    alpha: float = 0.01) -> np.ndarray:
    """
    Неслепая деконволюция с регуляризацией Тихонова (штраф на градиенты 
    первого порядка).

    Решает оптимизационную задачу:
        min_u ||k * u - f||^2 + alpha * ||grad u||^2
    """
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    OTF_dx, OTF_dy, _, _ = get_gradient_operators((H, W))
    reg_term = np.abs(OTF_dx) ** 2 + np.abs(OTF_dy) ** 2
    denominator = np.abs(otf) ** 2 + alpha * reg_term
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))


def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:
    """
    Сглаживание краев изображения в направлении его размытой копии для 
    подавления эффекта звона при деконволюции на базе БПФ.

    Алгоритм:
    1. Вычисление автокорреляции ядра размытия на его собственном размере.
    2. Извлечение центральных профилей автокорреляции по горизонтали и вертикали 
       для формирования плавного перехода от 0 к 1.
    3. Построение двумерной весовой маски alpha, равной 1 во внутренней области 
       и плавно спадающей к нулю на границах.
    4. Циклическое размытие изображения через БПФ.
    5. Смешивание: J = alpha * I + (1 - alpha) * blur(I). Процедура повторяется 
       n_tapers раз.

    Параметры
    ---------
    img : ndarray
        Входное изображение размерности (H, W).
    kernel : ndarray
        Ядро размытия.
    n_tapers : int, по умолчанию 3
        Количество последовательных итераций сглаживания.

    Возвращает
    ----------
    tapered : ndarray
        Изображение со сглаженными границами.
    """
    H, W = img.shape
    kh, kw = kernel.shape

    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf_max = acf.max()
    if acf_max > 0:
        acf /= acf_max

    cy, cx = kh - 1, kw - 1          
    z_col = acf[:, cx]               
    z_row = acf[cy, :]               

    beta_y = np.ones(H, dtype=np.float64)
    beta_x = np.ones(W, dtype=np.float64)

    half_ky = kh - 1                  
    if half_ky > 0:
        taper = z_col[:half_ky]       
        n = min(len(taper), H // 2)   
        beta_y[:n] = taper[:n]
        beta_y[-n:] = taper[:n][::-1]

    half_kx = kw - 1
    if half_kx > 0:
        taper = z_row[:half_kx]
        n = min(len(taper), W // 2)
        beta_x[:n] = taper[:n]
        beta_x[-n:] = taper[:n][::-1]

    alpha = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]

    otf = psf2otf(kernel, (H, W))

    result = img.copy()
    for _ in range(n_tapers):
        blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(result)))
        result = alpha * result + (1.0 - alpha) * blurred

    return result


def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Симметричное дополнение изображения на полный размер ядра с каждой стороны."""
    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')


def crop_image(img: np.ndarray, original_shape: tuple,
               kernel_shape: tuple) -> np.ndarray:
    """Обрезка дополненного изображения обратно до исходных размеров."""
    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    h, w = original_shape
    return img[pad_h:pad_h + h, pad_w:pad_w + w]
