# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
utils.py

Вспомогательные функции и операторы для алгоритмов слепой деконволюции 
с расширенными методами пространственной фильтрации.

Основано на методе:
    L. Chen, F. Fang, T. Wang, G. Zhang: "Blind Image Deblurring
    With Local Maximum Gradient Prior", CVPR, 2019.

Модуль включает поддержку направленной (guided), двусторонней (bilateral), 
нелокальной (NLM) и блочной (BM3D) фильтрации. Оператор поиска локального 
максимума расширен опциональной поддержкой гладкого вероятностного взвешивания 
через функцию softmax.
"""

import numpy as np
cimport numpy as cnp
from scipy import sparse
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates, uniform_filter
from scipy.fft import dstn, idstn
from skimage.restoration import denoise_nl_means, estimate_sigma
from libc.math cimport exp

try:
    import bm3d as _bm3d_mod
    _HAS_BM3D = True
except ImportError:
    _HAS_BM3D = False

def psf2otf(psf, shape):
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

def otf2psf(otf, psf_size):
    """
    Преобразование оптической передаточной функции в функцию рассеяния точки.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]

_OPT_FFT_LUT = None

def _build_opt_fft_lut(lut_size: int = 4096):
    """
    Построение таблицы оптимальных размеров для быстрого преобразования Фурье.
    Оптимальными считаются размеры, факторизуемые малыми простыми числами.
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
                    if e7 <= lut_size: lut[e7] = e7
                    if e7 * 11 <= lut_size: lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= lut_size: lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2
    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0: nn = i
        else: lut[i] = nn
    return lut

def opt_fft_size(n):
    """
    Вычисление оптимального размера массива для быстрого преобразования Фурье.
    """
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None: _OPT_FFT_LUT = _build_opt_fft_lut()
    n = np.asarray(n, dtype=np.int64)
    scalar_input = n.ndim == 0
    n = np.atleast_1d(n)
    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n)
    for i in range(n.size):
        nn = n.flat[i]
        m.flat[i] = _OPT_FFT_LUT[nn] if 1 <= nn <= lut_size else -1
    return int(m.flat[0]) if scalar_input else m

def _solve_min_laplacian(boundary_image):
    """
    Решение уравнения Лапласа с граничными условиями Дирихле на основе
    дискретного синусного преобразования первого типа.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()
    boundary_image[1:-1, 1:-1] = 0.0
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (-4.0 * boundary_image[1:H - 1, 1:W - 1]
                              + boundary_image[1:H - 1, 2:W]
                              + boundary_image[1:H - 1, 0:W - 2]
                              + boundary_image[0:H - 2, 1:W - 1]
                              + boundary_image[2:H, 1:W - 1])
    f2sin = dstn(-f_bp[1:H - 1, 1:W - 1], type=1)
    xx, yy = np.meshgrid(np.arange(1, W - 1), np.arange(1, H - 1))
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)
    img_direct = boundary_image.copy()
    img_direct[1:H - 1, 1:W - 1] = idstn(f2sin / denom, type=1)
    return img_direct

def wrap_boundary_liu(img, img_size):
    """
    Круговое сглаживание границ изображения для Фурье-деконволюции.
    """
    if img.ndim == 2: img = img[:, :, np.newaxis]
    H, W, Ch = img.shape
    H_out, W_out = int(img_size[0]), int(img_size[1])
    H_w, W_w = H_out - H, W_out - W
    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)
    for ch in range(Ch):
        HG = img[:, :, ch]
        r_A = np.zeros((2 + H_w, W), dtype=np.float64)
        r_A[:1, :], r_A[-1:, :] = HG[-1:, :], HG[:1, :]
        a = np.arange(H_w, dtype=np.float64) / (H_w - 1) if H_w > 1 else np.array([0.0])
        r_A[1:1 + H_w, 0], r_A[1:1 + H_w, -1] = (1 - a) * r_A[0, 0] + a * r_A[-1, 0], (1 - a) * r_A[0, -1] + a * r_A[-1, -1]
        A = _solve_min_laplacian(r_A)
        
        r_B = np.zeros((H, 2 + W_w), dtype=np.float64)
        r_B[:, :1], r_B[:, -1:] = HG[:, -1:], HG[:, :1]
        b = np.arange(W_w, dtype=np.float64) / (W_w - 1) if W_w > 1 else np.array([0.0])
        r_B[0, 1:1 + W_w], r_B[-1, 1:1 + W_w] = (1 - b) * r_B[0, 0] + b * r_B[0, -1], (1 - b) * r_B[-1, 0] + b * r_B[-1, -1]
        B = _solve_min_laplacian(r_B)
        
        r_C = np.zeros((2 + H_w, 2 + W_w), dtype=np.float64)
        r_C[:1, :], r_C[-1:, :] = B[-1:, :], B[:1, :]
        r_C[:, :1], r_C[:, -1:] = A[:, -1:], A[:, :1]
        C = _solve_min_laplacian(r_C)
        
        ret[:, :, ch] = np.block([[HG, B[:, 1:W_w + 1]], [A[:H_w, :], C[1:H_w + 1, 1:W_w + 1]]])
    return ret[:, :, 0] if ret.shape[2] == 1 else ret

def conjgrad(x, b, max_it, tol, ax_func, func_param):
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
        if abs(pAp) < 1e-30: break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol: break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def _matlab_round(x):
    return int(np.floor(np.abs(x) + 0.5) * np.sign(x)) if x != 0 else 0

def adjust_psf_center(psf):
    """
    Центрирование функции рассеяния точки.
    """
    rows, cols = psf.shape
    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64), np.arange(1, rows + 1, dtype=np.float64))
    if np.sum(psf) == 0: return psf
    xc1, yc1 = np.sum(psf * X), np.sum(psf * Y)
    xshift, yshift = _matlab_round((cols + 1) / 2.0 - xc1), _matlab_round((rows + 1) / 2.0 - yc1)
    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64), np.arange(cols, dtype=np.float64), indexing='ij')
    return map_coordinates(psf, [out_rows - yshift, out_cols - xshift], order=1, mode='constant', cval=0.0).reshape(rows, cols)

def _histc(data, edges):
    """
    Построение гистограммы распределения значений по заданным границам корзин.
    """
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)
    return np.bincount(indices, minlength=len(edges) + 1)[:len(edges)]

def guided_filter(I, p, radius, eps):
    """
    Направленный (guided) фильтр с сохранением границ.

    Выполняет сглаживание изображения p, используя опорное изображение I.
    Для подавления шума в самом изображении используется режим самонаправленности
    (I совпадает с p).

    Параметры
    ---------
    I : ndarray
        Опорное изображение.
    p : ndarray
        Фильтруемое изображение.
    radius : int
        Радиус локального окна фильтра.
    eps : float
        Коэффициент регуляризации.

    Возвращаемое значение
    ---------------------
    q : ndarray
        Отфильтрованное изображение.
    """
    ksize = 2 * radius + 1
    mean_I = uniform_filter(I.astype(np.float64), size=ksize, mode='reflect')
    mean_p = uniform_filter(p.astype(np.float64), size=ksize, mode='reflect')
    corr_I = uniform_filter((I * I).astype(np.float64), size=ksize, mode='reflect')
    corr_Ip = uniform_filter((I * p).astype(np.float64), size=ksize, mode='reflect')
    var_I, cov_Ip = corr_I - mean_I * mean_I, corr_Ip - mean_I * mean_p
    a, b = cov_Ip / (var_I + eps), mean_p - (cov_Ip / (var_I + eps)) * mean_I
    mean_a, mean_b = uniform_filter(a, size=ksize, mode='reflect'), uniform_filter(b, size=ksize, mode='reflect')
    return mean_a * I + mean_b

def threshold_pxpy_v1(latent, psf_size, threshold=None, denoise_eps=None, denoise_radius=2, ensemble_denoise=False, denoise_type='guided', bilateral_sigma_s=2.0, bilateral_sigma_r=0.1, bm3d_sigma=0.01, nlm_h=0.01):
    """
    Адаптивное пороговое отсечение шумовых градиентов изображения с 
    предварительной пространственной фильтрацией.

    Перед вычислением производных к изображению применяется выбранный метод
    шумоподавления для стабилизации оценки в зашумленных областях.
    """
    b_estimate_threshold = threshold is None
    if b_estimate_threshold: threshold = 0.0

    if denoise_eps is not None and denoise_eps > 0:
        if denoise_type == 'bm3d': denoised = bm3d_filter(latent, sigma_psd=bm3d_sigma)
        elif denoise_type == 'nlm': denoised = nlm_filter(latent, h=nlm_h)
        elif denoise_type == 'bilateral': denoised = bilateral_filter(latent, bilateral_sigma_s, bilateral_sigma_r)
        else:
            d_guided = guided_filter(latent, latent, denoise_radius, denoise_eps)
            denoised = (d_guided + guided_filter(latent, latent, max(1, denoise_radius - 1), denoise_eps * 0.5) + guided_filter(latent, latent, denoise_radius + 1, denoise_eps * 2.0)) / 3.0 if ensemble_denoise else d_guided
    else: denoised = latent

    dx, dy = np.array([[-1, 1], [0, 0]], dtype=np.float64), np.array([[-1, 0], [1, 0]], dtype=np.float64)
    px, py = convolve2d(denoised, dx, mode='valid'), convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        with np.errstate(divide='ignore', invalid='ignore'): pd = np.arctan(py / px)
        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]
        H1 = np.cumsum(_histc(pm[(pd >= 0) & (pd < np.pi / 4)], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[(pd >= np.pi / 4) & (pd < np.pi / 2)], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[(pd >= -np.pi / 4) & (pd < 0)], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[(pd >= -np.pi / 2) & (pd < -np.pi / 4)], pm_steps)[::-1])
        th = max((np.max(psf_size) if hasattr(psf_size, '__len__') else psf_size) * 20, 10)
        for t in range(len(pm_steps)):
            if min(H1[t], H2[t], H3[t], H4[t]) >= th:
                threshold = pm_steps[len(pm_steps) - 1 - t]
                break

    m = pm < threshold
    while np.all(m):
        threshold *= 0.81
        m = pm < threshold
    px[m], py[m] = 0.0, 0.0
    if not b_estimate_threshold: threshold /= 1.1
    return px, py, threshold

def nlm_filter(img, h=0.01, patch_size=5, patch_distance=6):
    """
    Подавление шума с использованием алгоритма нелокальных средних (NLM).
    """
    sigma_est = max(estimate_sigma(img), 1e-8)
    return denoise_nl_means(img, h=h * sigma_est / 0.01, sigma=sigma_est, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)

def bm3d_filter(img, sigma_psd=0.01):
    """
    Блочная фильтрация с трехмерным преобразованием (BM3D).
    """
    if not _HAS_BM3D: raise ImportError("bm3d package is not installed.")
    return _bm3d_mod.bm3d(img, sigma_psd=sigma_psd)

def _fspecial_gaussian(size, sigma):
    """
    Генерация двумерного гауссова фильтра.
    """
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()

def bilateral_filter(img, sigma_s, sigma):
    """
    Двусторонняя фильтрация изображения с сохранением границ.
    """
    if img.ndim == 2: img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1
    h, w, d = img.shape
    img = img.astype(np.float32)
    lab, sigma = img.copy(), sigma * np.sqrt(d)
    fr = int(np.ceil(sigma_s * 3))
    p_img, p_lab = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge'), np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    r_img, w_sum = np.zeros((h, w, d), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    spatial_weight, ss = _fspecial_gaussian(2 * fr + 1, sigma_s), sigma * sigma
    for y_off in range(-fr, fr + 1):
        for x_off in range(-fr, fr + 1):
            w_t = spatial_weight[y_off + fr, x_off + fr] * np.exp(-0.5 * np.sum((lab - p_lab[fr + y_off:fr + y_off + h, fr + x_off:fr + x_off + w, :]) ** 2, axis=2) / ss)
            r_img += p_img[fr + y_off:fr + y_off + h, fr + x_off:fr + x_off + w, :] * w_t[:, :, np.newaxis]
            w_sum += w_t
    r_img /= w_sum[:, :, np.newaxis]
    return r_img[:, :, 0] if was_2d else r_img

def find_min_pixels(I, patch_size, quantile=0.0):
    """
    Поиск минимальных пикселей в неперекрывающихся паттернах.

    При значении параметра quantile больше нуля осуществляется выбор
    значения, соответствующего заданному квантилю, что обеспечивает
    устойчивость алгоритма к экстремальным импульсным выбросам.

    Параметры
    ---------
    I : ndarray
        Полутоновое изображение.
    patch_size : int
        Размер стороны квадратного паттерна.
    quantile : float
        Уровень квантиля в диапазоне [0, 1). Значение 0 эквивалентно
        выбору абсолютного минимума.

    Возвращаемое значение
    ---------------------
    J : ndarray
        Разреженное изображение, содержащее найденные значения в 
        соответствующих координатах.
    Mask : ndarray
        Бинарная маска позиций минимальных значений.
    """
    M, N = I.shape
    Mp, Np = int(np.ceil(M / patch_size)), int(np.ceil(N / patch_size))
    J, Mask = np.zeros((M, N), dtype=np.float64), np.zeros((M, N), dtype=np.float64)
    for m in range(Mp):
        for n in range(Np):
            r_start, r_end = m * patch_size, min((m + 1) * patch_size, M)
            c_start, c_end = n * patch_size, min((n + 1) * patch_size, N)
            flat = I[r_start:r_end, c_start:c_end].flatten(order='F')
            if quantile > 0 and flat.size > 1:
                val = flat[np.argmin(np.abs(flat - np.quantile(flat, quantile)))]
            else:
                val = flat[np.argmin(flat)]
            pr, pc = np.unravel_index(np.argmin(np.abs(flat - val)) if quantile > 0 else np.argmin(flat), (r_end-r_start, c_end-c_start), order='F')
            J[r_start + pr, c_start + pc] = val
            Mask[r_start + pr, c_start + pc] = 1.0
    return J, Mask

def gen_partialmat(im_row, im_col):
    """
    Генерация разреженных матриц операторов частных производных.
    """
    M, N = im_row, im_col
    n = M * N
    all_inds = np.arange(n, dtype=np.int64)
    first_row_mask = (all_inds % M) == 0
    first_row, not_first_row = all_inds[first_row_mask], all_inds[~first_row_mask]
    r_fr, c_fr = np.repeat(first_row, 2), np.empty(2 * len(first_row), dtype=np.int64)
    c_fr[0::2], c_fr[1::2] = first_row, first_row + 1
    r_nfr, c_nfr = np.repeat(not_first_row, 2), np.empty(2 * len(not_first_row), dtype=np.int64)
    c_nfr[0::2], c_nfr[1::2] = not_first_row - 1, not_first_row
    py_mat = sparse.csr_matrix((np.concatenate([np.tile([-1.0, 1.0], len(first_row)), np.tile([-1.0, 1.0], len(not_first_row))]), (np.concatenate([r_fr, r_nfr]), np.concatenate([c_fr, c_nfr]))), shape=(n, n))
    
    first_col_mask = all_inds < M
    first_col, not_first_col = all_inds[first_col_mask], all_inds[~first_col_mask]
    r_fc, c_fc = np.repeat(first_col, 2), np.empty(2 * len(first_col), dtype=np.int64)
    c_fc[0::2], c_fc[1::2] = first_col, first_col + M
    r_nfc, c_nfc = np.repeat(not_first_col, 2), np.empty(2 * len(not_first_col), dtype=np.int64)
    c_nfc[0::2], c_nfc[1::2] = not_first_col, not_first_col - M
    px_mat = sparse.csr_matrix((np.concatenate([np.tile([-1.0, 1.0], len(first_col)), np.tile([1.0, -1.0], len(not_first_col))]), (np.concatenate([r_fc, r_nfc]), np.concatenate([c_fc, c_nfc]))), shape=(n, n))
    return px_mat, py_mat

def Abs_matrix(I):
    """
    Построение разреженной диагональной матрицы знаков элементов.
    """
    with np.errstate(divide='ignore', invalid='ignore'): abs_I = np.abs(I) / I
    abs_I = np.where(np.isfinite(abs_I), abs_I, 1.0)
    diag_vals = abs_I.flatten(order='F')
    return sparse.diags(diag_vals, 0, shape=(diag_vals.size, diag_vals.size), format='csr')


def Max_matrix(cnp.ndarray[cnp.float64_t, ndim=2] I, int patch_size, softmax_tau=None):
    """
    Построение разреженной матрицы выбора локального максимума.

    При значении параметра softmax_tau > 0 осуществляется вероятностное
    (мягкое) взвешивание соседей вместо жесткого выбора единичного максимума.
    Это преобразует оператор в непрерывную функцию от входного сигнала
    и устраняет хаотичную нестабильность при переключении индексов.

    Параметры
    ---------
    I : ndarray
        Карта полной вариации размерности (M, N).
    patch_size : int
        Размер окрестности локального поиска (нечетное число).
    softmax_tau : float, опционально
        Температурный параметр для функции softmax. Если не задан,
        применяется стандартное жесткое отсечение.

    Возвращаемое значение
    ---------------------
    max_mat : sparse.csr_matrix
        Разреженная матрица весов размерности (M*N, M*N).
    """
    cdef int M = I.shape[0]
    cdef int N = I.shape[1]
    cdef int padsize = patch_size // 2
    cdef int n_px = M * N
    cdef int m_0, n_0, r_start, r_end, c_start, c_end, r, c
    cdef double max_val, sum_w, stau
    cdef int best_r, best_c
    cdef double[:, ::1] I_view = I

    cdef int nnz
    cdef int[::1] r_view
    cdef int[::1] c_view
    cdef int[::1] sc_view
    cdef double[::1] v_view

    if softmax_tau is not None and softmax_tau > 0:
        stau = softmax_tau
        max_nnz = n_px * patch_size * patch_size
        row_arr = np.zeros(max_nnz, dtype=np.int32)
        col_arr = np.zeros(max_nnz, dtype=np.int32)
        val_arr = np.zeros(max_nnz, dtype=np.float64)
        
        r_view = row_arr
        c_view = col_arr
        v_view = val_arr
        nnz = 0

        for n_0 in range(N):
            for m_0 in range(M):
                r_start = m_0 - padsize
                if r_start < 0: r_start = 0
                r_end = m_0 + padsize
                if r_end >= M: r_end = M - 1
                c_start = n_0 - padsize
                if c_start < 0: c_start = 0
                c_end = n_0 + padsize
                if c_end >= N: c_end = N - 1

                max_val = -1e30
                for c in range(c_start, c_end + 1):
                    for r in range(r_start, r_end + 1):
                        if I_view[r, c] > max_val: max_val = I_view[r, c]

                sum_w = 0.0
                for c in range(c_start, c_end + 1):
                    for r in range(r_start, r_end + 1):
                        sum_w += exp((I_view[r, c] - max_val) / stau)

                for c in range(c_start, c_end + 1):
                    for r in range(r_start, r_end + 1):
                        r_view[nnz] = m_0 + n_0 * M
                        c_view[nnz] = r + c * M
                        v_view[nnz] = exp((I_view[r, c] - max_val) / stau) / sum_w
                        nnz += 1

        return sparse.csr_matrix((val_arr[:nnz], (row_arr[:nnz], col_arr[:nnz])), shape=(n_px, n_px))
    
    else:
        sparse_row = np.arange(n_px, dtype=np.int32)
        sparse_col = np.zeros(n_px, dtype=np.int32)
        sc_view = sparse_col

        for n_0 in range(N):
            for m_0 in range(M):
                r_start = m_0 - padsize
                if r_start < 0: r_start = 0
                r_end = m_0 + padsize
                if r_end >= M: r_end = M - 1
                c_start = n_0 - padsize
                if c_start < 0: c_start = 0
                c_end = n_0 + padsize
                if c_end >= N: c_end = N - 1

                max_val = -1e30
                best_r = -1
                best_c = -1

                for c in range(c_start, c_end + 1):
                    for r in range(r_start, r_end + 1):
                        if I_view[r, c] > max_val:
                            max_val = I_view[r, c]
                            best_r = r
                            best_c = c

                sc_view[m_0 + n_0 * M] = best_r + best_c * M

        sparse_val = np.ones(n_px, dtype=np.float64)
        return sparse.csr_matrix((sparse_val, (sparse_row, sparse_col)), shape=(n_px, n_px))

def LMG(img, patch_size, softmax_tau=None):
    """
    Вычисление локального максимального градиента и сборка оператора матрицы.

    Модифицированная реализация с опциональной поддержкой гладкого вероятностного
    взвешивания локальных экстремумов.

    Параметры
    ---------
    img : ndarray
        Полутоновое изображение размерности (M, N).
    patch_size : int
        Размер квадратной окрестности поиска максимума.
    softmax_tau : float, опционально
        Температурный параметр сглаживания операции выбора максимума.

    Возвращаемое значение
    ---------------------
    output_img : ndarray
        Карта локального максимального градиента.
    A : sparse.csr_matrix
        Разреженный линейный оператор G_S.
    """
    M, N = img.shape
    px_mat, py_mat = gen_partialmat(M, N)
    img_vec = img.flatten(order='F')
    px = (px_mat @ img_vec).reshape((M, N), order='F')
    py = (py_mat @ img_vec).reshape((M, N), order='F')
    abs_x_mat, abs_y_mat = Abs_matrix(px), Abs_matrix(py)
    tv = np.ascontiguousarray(np.abs(px) + np.abs(py), dtype=np.float64)
    max_tv_mat = Max_matrix(tv, patch_size, softmax_tau=softmax_tau)
    A = max_tv_mat @ (abs_x_mat @ px_mat + abs_y_mat @ py_mat)
    output_img = (A @ img_vec).reshape((M, N), order='F')
    return output_img, A