"""
utils.py

Вспомогательные функции и операторы для алгоритма слепой деконволюции 
на основе локальных минимальных значений интенсивности (PMP).

Основано на методе:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong: "A Simple Local 
    Minimal Intensity Prior and An Improved Algorithm for Blind Image 
    Deblurring", IEEE TCSVT, 2021.

Модуль включает поддержку направленной (guided) и двусторонней (bilateral) 
фильтрации, круговое сглаживание границ изображения, а также специализированный 
оператор поиска минимальных пикселей (или квантилей) в неперекрывающихся паттернах.
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.fft import dstn, idstn



def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Преобразование функции рассеяния точки в оптическую передаточную функцию.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Преобразование оптической передаточной функции в функцию рассеяния точки.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]



_OPT_FFT_LUT = None

def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """
    Построение таблицы оптимальных размеров для быстрого преобразования Фурье.
    """
    lut = np.zeros(lut_size + 1, dtype=np.int64) 

    e2 = 1
    while e2 <= lut_size:
        e3 = e2
        while e3 <= lut_size:
            e5 = e3
            while e5 <= lut_size:
                e7 = e5
                while e7 <= lut_size:
                    if e7 <= lut_size:
                        lut[e7] = e7
                    if e7 * 11 <= lut_size:
                        lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= lut_size:
                        lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2

    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = i
        else:
            lut[i] = nn
    return lut


def opt_fft_size(n) -> np.ndarray:
    """
    Вычисление оптимального размера массива для быстрого преобразования Фурье.
    """
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None:
        _OPT_FFT_LUT = _build_opt_fft_lut()

    n = np.asarray(n, dtype=np.int64)
    scalar_input = n.ndim == 0
    n = np.atleast_1d(n)

    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n)
    for i in range(n.size):
        nn = n.flat[i]
        if 1 <= nn <= lut_size:
            m.flat[i] = _OPT_FFT_LUT[nn]
        else:
            m.flat[i] = -1

    if scalar_input:
        return int(m.flat[0])
    return m



def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """
    Решение уравнения Лапласа с граничными условиями Дирихле.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    boundary_image[1:-1, 1:-1] = 0.0

    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]        # k+1
        + boundary_image[1:H - 1, 0:W - 2]    # k-1
        + boundary_image[0:H - 2, 1:W - 1]    # j-1
        + boundary_image[2:H,     1:W - 1]    # j+1
    )

    f1 = -f_bp

    f2 = f1[1:H - 1, 1:W - 1]

    f2sin = dstn(f2, type=1)

    x = np.arange(1, W - 1)   
    y = np.arange(1, H - 1)   
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom

    img_tt = idstn(f3, type=1)

    img_direct = boundary_image.copy()
    img_direct[1:H - 1, 1:W - 1] = img_tt

    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Круговое сглаживание границ изображения для Фурье-деконволюции.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H  
    W_w = W_out - W  

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]  

        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]

        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (
            (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        )
        r_A[alpha:alpha + H_w, -1] = (
            (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]
        )

        A2 = _solve_min_laplacian(r_A)
        r_A = A2
        A = r_A

        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (
            (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        )
        r_B[-1, alpha:alpha + W_w] = (
            (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]
        )

        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        A = A[:H_w, :]

        B = B[:, 1:W_w + 1]

        C = C[1:H_w + 1, 1:W_w + 1]

        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret


def find_min_pixels(I: np.ndarray, patch_size: int, quantile: float = 0.0):
    """
    Поиск минимальных пикселей (или квантилей) в неперекрывающихся паттернах.

    Разбивает изображение на неперекрывающиеся блоки заданного размера и
    идентифицирует пиксель с минимальной (или соответствующей заданному квантилю)
    интенсивностью в каждом из них. Использование малых квантилей вместо 
    абсолютного минимума повышает устойчивость к шумовым выбросам.

    Параметры
    ---------
    I : ndarray
        Полутоновое изображение размерности (M, N) в формате float64.
    patch_size : int
        Длина стороны квадратного паттерна.
    quantile : float
        Уровень квантиля в диапазоне [0, 1). Значение 0.0 соответствует
        выбору абсолютного минимума.

    Возвращаемое значение
    ---------------------
    J : ndarray
        Разреженное изображение размерности (M, N), содержащее найденные 
        минимальные значения в соответствующих координатах.
    Mask : ndarray
        Бинарная маска размерности (M, N), отмечающая позиции локальных минимумов.
    """
    M, N = I.shape
    Mp = int(np.ceil(M / patch_size))
    Np = int(np.ceil(N / patch_size))
    J = np.zeros((M, N), dtype=np.float64)
    Mask = np.zeros((M, N), dtype=np.float64)

    for m in range(Mp):
        for n in range(Np):
            r_start = m * patch_size
            r_end = min((m + 1) * patch_size, M)
            c_start = n * patch_size
            c_end = min((n + 1) * patch_size, N)

            patch = I[r_start:r_end, c_start:c_end]

            flat = patch.flatten(order='F')

            if quantile > 0 and flat.size > 1:
                q_val = np.quantile(flat, quantile)
                lin_idx = np.argmin(np.abs(flat - q_val))
                val = flat[lin_idx]
            else:
                lin_idx = np.argmin(flat)
                val = flat[lin_idx]

            pr, pc = np.unravel_index(lin_idx, patch.shape, order='F')

            J[r_start + pr, c_start + pc] = val

            Mask[r_start + pr, c_start + pc] = 1.0

    return J, Mask



def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Метод сопряженных градиентов для решения систем линейных уравнений.
    """
    x = x.copy()
    r = b - ax_func(x, func_param)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(max_it):
        Ap = ax_func(p, func_param)
        pAp = np.sum(p * Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x



def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    """
    Центрирование функции рассеяния точки.
    """
    rows, cols = psf.shape

    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                        np.arange(1, rows + 1, dtype=np.float64))

    total = np.sum(psf)
    if total == 0:
        return psf

    xc1 = np.sum(psf * X)  
    yc1 = np.sum(psf * Y)  

    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0

    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)

    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                      np.arange(cols, dtype=np.float64),
                                      indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    result = map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                             order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)


def _histc(data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Построение гистограммы распределения значений по заданным границам корзин.
    """
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def guided_filter(I: np.ndarray, p: np.ndarray,
                  radius: int, eps: float) -> np.ndarray:
    """
    Направленный (guided) фильтр с сохранением границ.

    Выполняет сглаживание изображения p, используя опорное изображение I.
    Для подавления шума в самом изображении используется режим самонаправленности
    (I совпадает с p).
    """
    ksize = 2 * radius + 1
    mean_I = uniform_filter(I.astype(np.float64), size=ksize, mode='reflect')
    mean_p = uniform_filter(p.astype(np.float64), size=ksize, mode='reflect')
    corr_I = uniform_filter((I * I).astype(np.float64), size=ksize, mode='reflect')
    corr_Ip = uniform_filter((I * p).astype(np.float64), size=ksize, mode='reflect')

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = uniform_filter(a, size=ksize, mode='reflect')
    mean_b = uniform_filter(b, size=ksize, mode='reflect')

    q = mean_a * I + mean_b
    return q


def threshold_pxpy_v1(latent: np.ndarray, psf_size,
                      threshold=None,
                      denoise_eps=None, denoise_radius=2,
                      ensemble_denoise=False):
    """
    Адаптивное пороговое отсечение шумовых градиентов изображения с 
    предварительной направленной фильтрацией.

    Перед вычислением пространственных производных к изображению может 
    применяться направленная фильтрация для подавления шума в гладких 
    областях. Опционально используется ансамблевое усреднение нескольких 
    сглаживающих окон.
    """
    b_estimate_threshold = threshold is None

    if b_estimate_threshold:
        threshold = 0.0

    if denoise_eps is not None and denoise_eps > 0:
        d_guided = guided_filter(latent, latent, denoise_radius, denoise_eps)
        if ensemble_denoise:
            r2 = max(1, denoise_radius - 1)
            d_guided2 = guided_filter(latent, latent, r2, denoise_eps * 0.5)
            d_guided3 = guided_filter(latent, latent, denoise_radius + 1, denoise_eps * 2.0)
            denoised = (d_guided + d_guided2 + d_guided3) / 3.0
        else:
            denoised = d_guided
    else:
        denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = (np.max(psf_size) if hasattr(psf_size, '__len__')
                        else psf_size)
        th = max(psf_size_val * 20, 10)

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                threshold = pm_steps[len(pm_steps) - 1 - t]
                break

    m = pm < threshold
    while np.all(m):
        threshold = threshold * 0.81
        m = pm < threshold

    px[m] = 0.0
    py[m] = 0.0

    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold


def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Генерация двумерного гауссова фильтра.
    """
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """
    Двусторонняя фильтрация изображения с сохранением границ.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1

    h, w, d = img.shape
    img = img.astype(np.float32)

    lab = img.copy()
    sigma = sigma * np.sqrt(d)

    fr = int(np.ceil(sigma_s * 3))

    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    p_lab = np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')

    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)

    spatial_weight = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * sigma

    for y in range(-fr, fr + 1):
        for x in range(-fr, fr + 1):
            w_s = spatial_weight[y + fr, x + fr]

            n_img = p_img[fr + y:fr + y + h, fr + x:fr + x + w, :]
            n_lab = p_lab[fr + y:fr + y + h, fr + x:fr + x + w, :]

            f_diff = lab - n_lab
            f_dist = np.sum(f_diff ** 2, axis=2)

            w_f = np.exp(-0.5 * f_dist / ss)
            w_t = w_s * w_f

            r_img += n_img * w_t[:, :, np.newaxis]
            w_sum += w_t

    r_img = r_img / w_sum[:, :, np.newaxis]

    if was_2d:
        return r_img[:, :, 0]
    return r_img
