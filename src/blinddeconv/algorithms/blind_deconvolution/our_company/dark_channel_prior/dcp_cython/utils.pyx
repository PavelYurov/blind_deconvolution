"""
utils.py

Вспомогательные вычислительные функции для алгоритма слепой деконволюции 
на основе априорного распределения темного канала (DCP).

Основано на методах:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR, 2016.
"""

cimport cython
cimport numpy as cnp
from libc.math cimport INFINITY

import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn



def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Преобразование функции рассеяния точки (PSF) в оптическую передаточную 
    функцию (OTF).

    Алгоритм:
    1. Дополнение матрицы PSF нулями до размеров shape.
    2. Циклический сдвиг матрицы для перемещения центра ядра в начало координат (0, 0).
    3. Вычисление двумерного быстрого преобразования Фурье.
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
    Преобразование оптической передаточной функции (OTF) обратно в 
    функцию рассеяния точки (PSF).
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]



_OPT_FFT_LUT = None


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """
    Формирование справочной таблицы оптимальных размеров БПФ. Оптимальными 
    считаются размеры, раскладывающиеся на простые множители (2, 3, 5, 7, 11, 13).
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
    """Поиск оптимального размера данных для быстрого преобразования Фурье."""
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
    Решает уравнение Лапласа с граничными условиями Дирихле с помощью 
    дискретного синус-преобразования (DST). Внутренние пиксели обнуляются, 
    а на основе граничных вычисляется лапласиан, который затем инвертируется 
    в частотной области.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    boundary_image[1:-1, 1:-1] = 0.0

    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H-1, 1:W-1] = (
        -4.0 * boundary_image[1:H-1, 1:W-1]
        + boundary_image[1:H-1, 2:W]       # k+1
        + boundary_image[1:H-1, 0:W-2]     # k-1
        + boundary_image[0:H-2, 1:W-1]     # j-1
        + boundary_image[2:H,   1:W-1]     # j+1
    )

    f1 = -f_bp

    f2 = f1[1:H-1, 1:W-1]

    f2sin = dstn(f2, type=1)

    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom

    img_tt = idstn(f3, type=1)

    img_direct = boundary_image.copy()
    img_direct[1:H-1, 1:W-1] = img_tt

    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Дополнение изображения для обеспечения циклической гладкости границ 
    (алгоритм Liu & Jia). Минимизирует артефакты звона при выполнении 
    деконволюции на базе БПФ.
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
        r_A[alpha:alpha + H_w, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        r_A[alpha:alpha + H_w, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

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
        r_B[0, alpha:alpha + W_w] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        r_B[-1, alpha:alpha + W_w] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

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



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _dark_channel_kernel(
        const double[:, :, ::1] I_pad,
        double[:, ::1] J,
        long long[:, ::1] J_index,
        Py_ssize_t M, Py_ssize_t N, Py_ssize_t C,
        Py_ssize_t ps) noexcept nogil:
    cdef Py_ssize_t m, n, r, c, ch, min_idx
    cdef double min_val, val, v

    for m in range(M):
        for n in range(N):
            min_val = INFINITY
            min_idx = 0
            for c in range(ps):
                for r in range(ps):
                    val = I_pad[m + r, n + c, 0]
                    for ch in range(1, C):
                        v = I_pad[m + r, n + c, ch]
                        if v < val:
                            val = v
                    if val < min_val:
                        min_val = val
                        min_idx = r + ps * c
            J[m, n] = min_val
            J_index[m, n] = min_idx + 1 


def dark_channel(I: np.ndarray, patch_size: int):
    """
    Вычисление темного канала изображения. Для каждого пикселя вычисляется 
    минимальное значение интенсивности в пространственном окне среди всех 
    цветовых каналов. Возвращает также 1D-индекс пикселя, на котором был 
    достигнут минимум.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    cdef Py_ssize_t M = I.shape[0]
    cdef Py_ssize_t N = I.shape[1]
    cdef Py_ssize_t C = I.shape[2]
    cdef Py_ssize_t ps = patch_size
    cdef Py_ssize_t p = patch_size // 2

    I_pad_np = np.ascontiguousarray(
        np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge'),
        dtype=np.float64,
    )
    J_np = np.zeros((M, N), dtype=np.float64)
    J_index_np = np.zeros((M, N), dtype=np.int64)

    cdef double[:, :, ::1] I_pad_mv = I_pad_np
    cdef double[:, ::1] J_mv = J_np
    cdef long long[:, ::1] J_index_mv = J_index_np

    with nogil:
        _dark_channel_kernel(I_pad_mv, J_mv, J_index_mv, M, N, C, ps)

    return J_np, J_index_np


def dark_channel_fast(I: np.ndarray, patch_size: int):
    return dark_channel(I, patch_size)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _assign_dcp_kernel(
        double[:, :, ::1] S_padd,
        const double[:, ::1] refine,
        const long long[:, ::1] J_idx,
        Py_ssize_t M, Py_ssize_t N, Py_ssize_t C,
        Py_ssize_t ps) noexcept nogil:
    cdef Py_ssize_t m, n, r, c, ch, idx, rr, cc, cch
    cdef double min_val, val, ref
    cdef Py_ssize_t ps2 = ps * ps

    for m in range(M):
        for n in range(N):
            ref = refine[m, n]
            min_val = INFINITY
            for r in range(ps):
                for c in range(ps):
                    for ch in range(C):
                        val = S_padd[m + r, n + c, ch]
                        if val < min_val:
                            min_val = val

            if min_val != ref:
                idx = J_idx[m, n] - 1
                cch = idx // ps2
                idx = idx - cch * ps2
                cc = idx // ps
                rr = idx - cc * ps
                S_padd[m + rr, n + cc, cch] = ref


def assign_dark_channel_to_pixel(S: np.ndarray,
                                 dark_channel_refine: np.ndarray,
                                 dark_channel_index: np.ndarray,
                                 patch_size: int) -> np.ndarray:
    """
    Присвоение уточненных значений темного канала обратно исходным пикселям 
    изображения на основе сохраненных линейных индексов.
    """
    if S.ndim == 2:
        S_3d = S[:, :, np.newaxis]
        was_2d = True
    else:
        S_3d = S
        was_2d = False

    cdef Py_ssize_t M = S_3d.shape[0]
    cdef Py_ssize_t N = S_3d.shape[1]
    cdef Py_ssize_t C = S_3d.shape[2]
    cdef Py_ssize_t ps = patch_size
    cdef Py_ssize_t padsize = patch_size // 2

    S_padd_np = np.ascontiguousarray(
        np.pad(S_3d, ((padsize, padsize), (padsize, padsize), (0, 0)),
               mode='edge'),
        dtype=np.float64,
    )
    refine_np = np.ascontiguousarray(dark_channel_refine, dtype=np.float64)
    idx_np = np.ascontiguousarray(dark_channel_index, dtype=np.int64)

    cdef double[:, :, ::1] S_padd_mv = S_padd_np
    cdef double[:, ::1] refine_mv = refine_np
    cdef long long[:, ::1] idx_mv = idx_np

    with nogil:
        _assign_dcp_kernel(S_padd_mv, refine_mv, idx_mv, M, N, C, ps)

    outImg = S_padd_np[padsize:padsize + M, padsize:padsize + N, :]

    outImg = outImg.copy()
    S_3d_arr = np.asarray(S_3d, dtype=np.float64)
    outImg[:padsize, :, :] = S_3d_arr[:padsize, :, :]
    outImg[-padsize:, :, :] = S_3d_arr[-padsize:, :, :]
    outImg[:, :padsize, :] = S_3d_arr[:, :padsize, :]
    outImg[:, -padsize:, :] = S_3d_arr[:, -padsize:, :]

    if was_2d:
        return outImg[:, :, 0]
    return outImg


def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Решение линейной системы A*x = b методом сопряженных градиентов.
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
    Вычисление центра масс ядра и его сдвиг к геометрическому центру 
    массива с помощью билинейной интерполяции.
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
    """Гистограмма с включением граничных значений в последний интервал."""
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size,
                      threshold=None):
    """
    Адаптивное пороговое ограничение градиентов скрытого изображения.
    Слабые градиенты обнуляются для повышения надежности оценки ядра. 
    Порог оценивается по кумулятивной гистограмме магнитуд градиентов 
    в четырех направлениях.
    """
    b_estimate_threshold = threshold is None

    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        pd = np.arctan2(py, px) 

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

        psf_size_val = np.max(psf_size) if hasattr(psf_size, '__len__') else psf_size
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
    """Формирование ядра фильтра Гаусса."""
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """Билатеральная фильтрация для подавления шума с сохранением краев объектов."""
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



def graythresh(img: np.ndarray) -> float:
    """Вычисление оптимального порога бинаризации методом Оцу."""
    img_flat = img.ravel().astype(np.float64)
    img_flat = np.clip(img_flat, 0.0, 1.0)

    num_bins = 256
    counts, bin_edges = np.histogram(img_flat, bins=num_bins, range=(0.0, 1.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts.astype(np.float64) / total

    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centres)
    mu_t = mu[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))

    sigma_b_sq = np.nan_to_num(sigma_b_sq, nan=0.0)
    max_idx = np.argmax(sigma_b_sq)

    return bin_centres[max_idx]



def wiener_filter(img: np.ndarray, kernel: np.ndarray,
                  noise_snr: float = 0.01) -> np.ndarray:
    """Неслепая деконволюция на основе фильтра Винера в частотной области."""
    H, W = img.shape[:2]
    otf = psf2otf(kernel, (H, W))
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (np.abs(otf) ** 2 + noise_snr)) * F_img
    return np.real(np.fft.ifft2(F_res))


def tikhonov_filter(img: np.ndarray, kernel: np.ndarray,
                    alpha: float = 0.01) -> np.ndarray:
    """Неслепая деконволюция с регуляризацией Тихонова."""
    H, W = img.shape[:2]
    otf = psf2otf(kernel, (H, W))

    dx_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float64)
    dy_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float64)
    OTF_dx = psf2otf(dx_kernel, (H, W))
    OTF_dy = psf2otf(dy_kernel, (H, W))

    reg_term = np.abs(OTF_dx) ** 2 + np.abs(OTF_dy) ** 2
    denominator = np.abs(otf) ** 2 + alpha * reg_term

    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))


def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:
    """Сглаживание краев изображения для предотвращения артефактов звона при БПФ-деконволюции."""
    H, W = img.shape[:2]
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

    alpha_map = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]
    otf = psf2otf(kernel, (H, W))

    result = img.copy()
    for _ in range(n_tapers):
        blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(result)))
        result = alpha_map * result + (1.0 - alpha_map) * blurred

    return result


def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Симметричное дополнение изображения на размер ядра с каждой стороны."""
    pad_h, pad_w = kernel_shape[0], kernel_shape[1]
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')


def crop_image(img: np.ndarray, original_shape: tuple,
               kernel_shape: tuple) -> np.ndarray:
    """Обрезка дополненного изображения обратно до исходных размеров."""
    pad_h, pad_w = kernel_shape[0], kernel_shape[1]
    h, w = original_shape
    return img[pad_h:pad_h + h, pad_w:pad_w + w]
