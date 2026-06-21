"""
utils.py

Вспомогательные вычислительные функции для алгоритма графовой слепой 
деконволюции изображений (GBBID).

Основано на методах:
    - Y. Bai, G. Cheung, X. Liu, W. Gao: "Graph-Based Blind Image Deblurring 
      From a Single Photograph", IEEE TIP 2019.
    - D. Krishnan, R. Fergus: "Fast Image Deconvolution using Hyper-Laplacian 
      Priors", NIPS 2009.
    - Библиотека двумерного дискретного вейвлет-преобразования (Jian-Feng Cai).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from scipy.ndimage import convolve as ndimage_convolve
from scipy.ndimage import correlate as ndimage_correlate
from scipy.interpolate import interp1d
from scipy.fft import dstn, idstn



def psf2otf(psf, shape):
    """
    Преобразование функции рассеяния точки (PSF) в оптическую передаточную 
    функцию (OTF).

    Алгоритм:
    1. Дополнение матрицы PSF нулями до размеров shape.
    2. Циклический сдвиг матрицы так, чтобы центр PSF оказался в координате (0, 0).
    3. Вычисление двумерного быстрого преобразования Фурье.
    """
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf, psf_size):
    """
    Преобразование оптической передаточной функции (OTF) обратно в 
    функцию рассеяния точки (PSF).
    """
    full = np.real(ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]



def G_padding(x, k, factor):
    """
    Дополнение границ входного изображения для построения графа.
    Использует дублирование крайних пикселей (режим 'edge').

    Возвращает дополненное изображение и размер добавленных полей (padsize).
    """
    padsize = (k.shape[0] * factor, k.shape[1] * factor)
    x_padding = np.pad(x,
                       ((padsize[0], padsize[0]), (padsize[1], padsize[1])),
                       mode='edge')
    return x_padding, padsize


def Copy_Enlarge_h(I, H_size):
    """
    Симметричное увеличение изображения путем дублирования крайних строк и столбцов.

    Возвращает увеличенное изображение и размер добавленных полей (border).
    """
    s_h, s_w = int(H_size[0]), int(H_size[1])
    if s_h % 2 == 0:
        s_h += 1
    if s_w % 2 == 0:
        s_w += 1
    border = (s_h - 1, s_w - 1)
    h, w = I.shape

    left = np.tile(I[:, 0:1], (1, border[1]))
    right = np.tile(I[:, -1:], (1, border[1]))
    I2 = np.concatenate([left, I, right], axis=1)

    top = np.tile(I2[0:1, :], (border[0], 1))
    bottom = np.tile(I2[-1:, :], (border[0], 1))
    I2 = np.concatenate([top, I2, bottom], axis=0)

    return I2, border


def fftconv(I, filt, method):
    """
    Ускоренная двумерная свертка с использованием быстрого преобразования Фурье.

    Параметры
    ---------
    I : ndarray
        Двумерный массив изображения.
    filt : ndarray
        Ядро свертки.
    method : str
        Режим формирования выходного массива ('same' или 'valid').
    """
    k1, k2 = filt.shape

    I_padded, p_size = G_padding(I, filt, 1)
    n, m = I_padded.shape

    if method == 'same':
        tI = np.zeros((n + k1 - 1, m + k2 - 1), dtype=np.float64)
        tI[:n, :m] = I_padded
        I_padded = tI

    bn, bm = I_padded.shape
    fI = fft2(I_padded)
    ff = fft2(filt, s=(bn, bm))
    fI = fI * ff
    cI = np.real(ifft2(fI))

    hk1d = k1 // 2
    hk1u = k1 - hk1d - 1
    hk2d = k2 // 2
    hk2u = k2 - hk2d - 1

    if method == 'same':
        end0 = -hk1u if hk1u > 0 else None
        end1 = -hk2u if hk2u > 0 else None
        cI = cI[hk1d:end0, hk2d:end1]
        cI = cI[p_size[0]:-p_size[0] if p_size[0] > 0 else None,
                p_size[1]:-p_size[1] if p_size[1] > 0 else None]
    elif method == 'valid':
        cI = cI[hk1d + hk1u:, hk2d + hk2u:]

    return cI



def edgetaper(img, psf):
    """
    Сглаживание краев изображения путем смешивания исходного изображения 
    с его размытой копией. 

    Веса для смешивания вычисляются на основе одномерной автокорреляции 
    горизонтальных и вертикальных проекций функции рассеяния точки (PSF).
    Такой сепарабельный подход позволяет корректно сглаживать все границы 
    независимо от формы ядра (включая ядра размытия в движении по диагонали).
    """
    n, m = img.shape

    proj_row = psf.sum(axis=0)
    proj_col = psf.sum(axis=1)

    acf_row = np.real(np.fft.ifft(np.abs(np.fft.fft(proj_row, m)) ** 2))
    acf_row = acf_row / acf_row[0]

    acf_col = np.real(np.fft.ifft(np.abs(np.fft.fft(proj_col, n)) ** 2))
    acf_col = acf_col / acf_col[0]

    beta = acf_col[:, np.newaxis] * acf_row[np.newaxis, :]

    otf = psf2otf(psf, (n, m))
    blurred = np.real(ifft2(fft2(img) * otf))

    return img * (1.0 - beta) + blurred * beta


def weight_function_l1(d):
    """Вычисление весов для L1-графового лапласиана: w = 1 / max(|d|, epsilon)."""
    epsilon = 0.01
    d_abs = np.abs(d)
    d_abs = np.maximum(d_abs, epsilon)
    return 1.0 / d_abs


def weights_computation(x, sigma, nei_num, wtype):
    """
    Вычисление весов графа для пространственной регуляризации.

    Параметры
    ---------
    x : ndarray
        Текущая оценка изображения.
    sigma : float или None
        Параметр Гауссианы (используется при wtype=1).
    nei_num : int
        Количество соседей (поддерживается только 4).
    wtype : int
        Тип взвешивания:
        1 - Гауссово: w = exp(-d^2 / sigma^2)
        2 - Норма L1 (IRLS): w = 1 / |d|

    Возвращает
    ----------
    W : ndarray
        Массив весов графа размерности (h*w, 4).
    """
    h, w = x.shape

    if nei_num == 4 and wtype == 1:
        W = np.zeros((h * w, 4), dtype=np.float64)

        d1 = np.array([[1, -1, 0]], dtype=np.float64)
        d2 = d1.T
        d3 = np.array([[0, -1, 1]], dtype=np.float64)
        d4 = d3.T

        W[:, 0] = ndimage_convolve(x, d1, mode='nearest').ravel()
        W[:, 0] = np.exp(-W[:, 0] ** 2 / sigma ** 2)

        W[:, 1] = ndimage_convolve(x, d2, mode='nearest').ravel()
        W[:, 1] = np.exp(-W[:, 1] ** 2 / sigma ** 2)

        W[:, 2] = ndimage_convolve(x, d3, mode='nearest').ravel()
        W[:, 2] = np.exp(-W[:, 2] ** 2 / sigma ** 2)

        W[:, 3] = ndimage_convolve(x, d4, mode='nearest').ravel()
        W[:, 3] = np.exp(-W[:, 3] ** 2 / sigma ** 2)

    elif nei_num == 4 and wtype == 2:
        W = np.zeros((h * w, 4), dtype=np.float64)

        d1 = np.array([[1, -1, 0]], dtype=np.float64)
        d2 = d1.T
        d3 = np.array([[0, -1, 1]], dtype=np.float64)
        d4 = d3.T

        W[:, 0] = weight_function_l1(
            ndimage_convolve(x, d1, mode='nearest').ravel())
        W[:, 1] = weight_function_l1(
            ndimage_convolve(x, d2, mode='nearest').ravel())
        W[:, 2] = weight_function_l1(
            ndimage_convolve(x, d3, mode='nearest').ravel())
        W[:, 3] = weight_function_l1(
            ndimage_convolve(x, d4, mode='nearest').ravel())
    else:
        W = np.zeros(1)

    return W


def _adaptive_threshold(M, ratio, max_iter):
    """
    Бинарный поиск порогового значения, при котором заданная доля 
    пикселей (ratio) превышает этот порог.
    """
    n = M.size
    lower_bound = 0.0
    upper_bound = float(M.max())
    threshold = upper_bound / 2.0
    r = 0.0

    for _ in range(max_iter):
        M_t = np.sum(M > threshold)
        r = M_t / n
        if ratio * 0.9 < r < ratio * 1.1:
            break
        elif r <= ratio * 0.9:
            upper_bound = threshold
            threshold = (lower_bound + upper_bound) / 2.0
        else:
            lower_bound = threshold
            threshold = (lower_bound + upper_bound) / 2.0

    M_threshold = np.zeros_like(M)
    M_threshold[M > threshold] = 1.0
    return M_threshold, r


def informative_edge_mask_adaptive_mine(Y_s, t_s, t_r, h):
    """
    Выделение информативных краев и формирование соответствующей бинарной маски.

    Параметры
    ---------
    Y_s : ndarray
        Промежуточное структурное (skeleton) изображение.
    t_s : float
        Относительный порог для интенсивности градиентов.
    t_r : float
        Относительный порог для оценки когерентности краев.
    h : int
        Размер локального окна для оценки.

    Возвращает
    ----------
    M : ndarray
        Бинарная пространственная маска размерности Y_s.
    """
    Dx = np.array([[1, -1, 0]], dtype=np.float64)
    Dy = Dx.T

    Mx = ndimage_convolve(Y_s, Dx, mode='nearest')
    My = ndimage_convolve(Y_s, Dy, mode='nearest')
    M_mag = np.sqrt(Mx ** 2 + My ** 2)

    M3, _ = _adaptive_threshold(M_mag, t_s, 100)

    k_tmp = np.ones((h, h), dtype=np.float64)
    Mx2 = ndimage_convolve(Mx, k_tmp, mode='nearest')
    My2 = ndimage_convolve(My, k_tmp, mode='nearest')
    M4 = np.sqrt(Mx2 ** 2 + My2 ** 2)

    M5 = ndimage_convolve(M_mag, k_tmp, mode='nearest')
    M4 = M4 / (M5 + 0.5)

    M4_bin, _ = _adaptive_threshold(M4, t_r, 100)

    return M3 * M4_bin


def _shift_kernel(k, hw):
    """Сдвиг ядра на величину (dh, dw) пикселей."""
    h, w = k.shape
    dh, dw = int(hw[0]), int(hw[1])

    k_tmp = np.zeros_like(k)
    if dh >= 0:
        if dh < h:
            k_tmp[dh:, :] = k[:h - dh, :]
    else:
        if -dh < h:
            k_tmp[:h + dh, :] = k[-dh:, :]

    k_s = np.zeros_like(k)
    if dw >= 0:
        if dw < w:
            k_s[:, dw:] = k_tmp[:, :w - dw]
    else:
        if -dw < w:
            k_s[:, :w + dw] = k_tmp[:, -dw:]

    return k_s


def kernel_centralize(k, threshold):
    """
    Centralize restored kernel.
    MATLAB: kernel_centralize.m

    Finds the bounding box of significant kernel elements,
    computes its centre, and shifts the kernel so that this centre
    aligns with the geometric centre of the array.
    """
    h, w = k.shape
    thresh_val = k.max() * threshold

    h_begin = 0
    for i in range(h):
        if k[i, :].sum() > thresh_val:
            h_begin = i
            break

    h_end = h - 1
    for i in range(h - 1, -1, -1):
        if k[i, :].sum() > thresh_val:
            h_end = i
            break

    w_begin = 0
    for i in range(w):
        if k[:, i].sum() > thresh_val:
            w_begin = i
            break

    w_end = w - 1
    for i in range(w - 1, -1, -1):
        if k[:, i].sum() > thresh_val:
            w_end = i
            break

    h_center = int(np.floor(h_begin + (h_end - h_begin) / 2.0))
    w_center = int(np.floor(w_begin + (w_end - w_begin) / 2.0))

    kh_center = (h - 1) // 2
    kw_center = (w - 1) // 2

    dh = kh_center - h_center
    dw = kw_center - w_center

    k_c = _shift_kernel(k, (dh, dw))
    k_c_sum = k_c.sum()
    if k_c_sum > 0:
        k_c = k_c / k_c_sum
    return k_c


def k_rescale(k):
    """Масштабирование ядра по правилу min-max для визуализации в диапазоне [0, 1]."""
    k_max = k.max()
    k_min = k.min()
    if k_max == k_min:
        return np.zeros_like(k)
    return (k - k_min) / (k_max - k_min)


def conjgrad(x, b, max_it, tol, Ax_func, func_param):
    """
    Решение системы линейных уравнений A*x = b методом сопряженных градиентов.
    Матрица A задается неявно через функцию умножения матрицы на вектор (Ax_func).
    """
    r = b - Ax_func(x, func_param)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(max_it):
        Ap = Ax_func(p, func_param)
        pAp = np.sum(p * Ap)
        if pAp == 0:
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



def GenerateFrameletFilter(frame):
    """
    Генерация наборов фильтров для декомпозиции (D) и восстановления (R) 
    с использованием вейвлет-фреймов.

    Параметры
    ---------
    frame : int
        Тип базиса: 
        0 = Вейвлет Хаара
        1 = Кусочно-линейный фрейм
        3 = Кусочно-кубический фрейм

    Возвращает
    ----------
    D, R : list
        Списки фильтров, где последний элемент — строка с граничными условиями.
    """
    if frame == 0:
        D = [
            np.array([0, 1, 1], dtype=np.float64) / 2,
            np.array([0, 1, -1], dtype=np.float64) / 2,
            'cc',
        ]
        R = [
            np.array([1, 1, 0], dtype=np.float64) / 2,
            np.array([-1, 1, 0], dtype=np.float64) / 2,
            'cc',
        ]
    elif frame == 1:
        D = [
            np.array([1, 2, 1], dtype=np.float64) / 4,
            np.array([1, 0, -1], dtype=np.float64) / 4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=np.float64) / 4,
            'ccc',
        ]
        R = [
            np.array([1, 2, 1], dtype=np.float64) / 4,
            np.array([-1, 0, 1], dtype=np.float64) / 4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=np.float64) / 4,
            'ccc',
        ]
    elif frame == 3:
        D = [
            np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16,
            np.array([1, 2, 0, -2, -1], dtype=np.float64) / 8,
            np.array([-1, 0, 2, 0, -1], dtype=np.float64) / 16 * np.sqrt(6),
            np.array([-1, 2, 0, -2, 1], dtype=np.float64) / 8,
            np.array([1, -4, 6, -4, 1], dtype=np.float64) / 16,
            'ccccc',
        ]
        R = [
            np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16,
            np.array([-1, -2, 0, 2, 1], dtype=np.float64) / 8,
            np.array([-1, 0, 2, 0, -1], dtype=np.float64) / 16 * np.sqrt(6),
            np.array([1, -2, 0, 2, -1], dtype=np.float64) / 8,
            np.array([1, -4, 6, -4, 1], dtype=np.float64) / 16,
            'ccccc',
        ]
    else:
        raise ValueError(f"Unsupported frame type: {frame}")

    return D, R


def ConvSymAsym2D(A, M, b, L):
    """
    Одномерная свертка (корреляция) с учетом граничных условий, применяемая по строкам.
    """
    m, n = A.shape
    nM = len(M)
    step = 2 ** (L - 1)

    ker_len = step * (nM - 1) + 1
    ker = np.zeros(ker_len, dtype=np.float64)
    ker[::step] = M
    lker = ker_len // 2

    ker_2d = ker.reshape(-1, 1)

    if b == 'c':
        C = ndimage_correlate(A, ker_2d, mode='wrap')
    else:
        Ae = np.pad(A,
                    ((lker, lker), (0, 0)),
                    mode='symmetric')
        if b == 'a':
            Ae[:lker, :] = -Ae[:lker, :]
            Ae[m + lker:m + 2 * lker, :] = -Ae[m + lker:m + 2 * lker, :]

        from scipy.signal import convolve2d
        C = convolve2d(Ae, ker_2d, mode='valid')

    return C


def FraDec2D(A, D, L):
    """Одноуровневая сепарабельная двумерная декомпозиция на базе вейвлет-фреймов."""
    nD = len(D)
    SorAS = D[-1]
    n_filt = nD - 1

    Dec = [[None] * n_filt for _ in range(n_filt)]
    for i in range(n_filt):
        M1 = D[i]
        tempi = ConvSymAsym2D(A, M1, SorAS[i], L)
        for j in range(n_filt):
            M2 = D[j]
            tempj = ConvSymAsym2D(tempi.T, M2, SorAS[j], L)
            Dec[i][j] = tempj.T.copy()
    return Dec


def FraDecMultiLevel2D(A, D, L):
    """Многоуровневая декомпозиция на базе вейвлет-фреймов."""
    Dec = []
    kDec = A.copy()
    for k in range(1, L + 1):
        dec_k = FraDec2D(kDec, D, k)
        Dec.append(dec_k)
        kDec = dec_k[0][0].copy()
    return Dec


def FraRec2D(C, R, L):
    """Одноуровневое сепарабельное восстановление из коэффициентов вейвлет-фреймов."""
    nR = len(R)
    SorAS = R[-1]
    n_filt = nR - 1

    ImSize = C[0][0].shape
    Rec = np.zeros(ImSize, dtype=np.float64)

    for i in range(n_filt):
        temp = np.zeros(ImSize, dtype=np.float64)
        for j in range(n_filt):
            M2 = R[j]
            temp = temp + ConvSymAsym2D(C[i][j].T, M2, SorAS[j], L).T
        M1 = R[i]
        Rec = Rec + ConvSymAsym2D(temp, M1, SorAS[i], L)

    return Rec


def sort_filter(Cf, level, f_n, ratio):
    """
    Пороговое ограничение вейвлет-коэффициентов на заданном уровне декомпозиции.

    Собирает все коэффициенты, сортирует по абсолютной величине и обнуляет 
    заданную долю (1 - ratio) наименьших значений.
    """
    h, w = Cf[level][0][0].shape
    num = h * w

    v_cf = np.zeros(num * f_n * f_n, dtype=np.float64)
    n = 0
    for k in range(f_n):
        for t in range(f_n):
            v_cf[n:n + num] = Cf[level][k][t].ravel()
            n += num

    indices = np.argsort(np.abs(v_cf))
    n_zero = int(np.floor(num * f_n * f_n * (1 - ratio)))
    v_cf[indices[:n_zero]] = 0.0

    n = 0
    for k in range(f_n):
        for t in range(f_n):
            Cf[level][k][t] = v_cf[n:n + num].reshape(h, w)
            n += num

    return Cf


def kernel_filter(C, R, L, ratio):
    """
    Удаление шума из восстановленного ядра с использованием вейвлет-фильтрации.

    Параметры
    ---------
    C : list
        Коэффициенты многоуровневой декомпозиции.
    R : list
        Набор фильтров для восстановления.
    L : int
        Количество уровней декомпозиции.
    ratio : float
        Доля сохраняемых наибольших коэффициентов.
    """
    f_n = len(R) - 1

    for k in range(L, 1, -1):
        C = sort_filter(C, k - 1, f_n, ratio)
        C[k - 2][0][0] = FraRec2D(C[k - 1], R, k)

    C = sort_filter(C, 0, f_n, ratio)
    Rec = FraRec2D(C[0], R, 1)
    return Rec



_SOLVE_IMAGE_LUT = {}


def clear_solve_image_cache():
    """Очистка постоянного кэша таблицы LUT."""
    _SOLVE_IMAGE_LUT.clear()


def _compute_w1(v, beta):
    """Оценка проксимального оператора для alpha = 1 (мягкое пороговое ограничение)."""
    return np.maximum(np.abs(v) - 1.0 / beta, 0.0) * np.sign(v)


def _compute_w23(v, beta):
    """Аналитическая оценка проксимального оператора для alpha = 2/3 (метод Феррари)."""
    epsilon = 1e-6

    k_val = 8.0 / (27.0 * beta ** 3)
    m = np.full_like(v, k_val)

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    m2 = m * m
    m3 = m2 * m

    alpha_q = -1.125 * v2
    beta2 = 0.25 * v3

    q = -0.125 * (m * v2)
    disc = -m3 / 27.0 + (m2 * v4) / 256.0
    r1 = -q / 2.0 + np.sqrt(disc.astype(np.complex128))

    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.exp(np.log(r1) / 3.0)
        y = 2.0 * (-5.0 / 18.0 * alpha_q + u + (m.astype(np.complex128) / (3.0 * u)))

    W_val = np.sqrt(alpha_q.astype(np.complex128) / 3.0 + y)

    alpha_c = alpha_q.astype(np.complex128)
    beta2_c = beta2.astype(np.complex128)

    root = np.zeros((v.size, 4), dtype=np.complex128)
    v_flat = v.ravel()

    sqrt_plus = np.sqrt(-(alpha_c + y + beta2_c / W_val))
    sqrt_minus = np.sqrt(-(alpha_c + y - beta2_c / W_val))

    root[:, 0] = 0.75 * v_flat + 0.5 * (W_val + sqrt_plus)
    root[:, 1] = 0.75 * v_flat + 0.5 * (W_val - sqrt_plus)
    root[:, 2] = 0.75 * v_flat + 0.5 * (-W_val + sqrt_minus)
    root[:, 3] = 0.75 * v_flat + 0.5 * (-W_val - sqrt_minus)

    v_rep = np.repeat(v_flat[:, np.newaxis], 4, axis=1)
    sv2 = np.sign(v_rep)
    rsv2 = np.real(root) * sv2

    mask = ((np.abs(np.imag(root)) < epsilon) &
            (rsv2 > np.abs(v_rep) / 2.0) &
            (rsv2 < np.abs(v_rep)))

    filtered = mask * rsv2
    sorted_vals = np.sort(filtered, axis=1)[:, ::-1]
    w = sorted_vals[:, 0] * np.sign(v_flat)

    return np.real(w).reshape(v.shape)


def _compute_w12(v, beta):
    """Аналитическая оценка проксимального оператора для alpha = 1/2 (метод Кардано)."""
    epsilon = 1e-6

    k_val = -0.25 / beta ** 2
    m = np.full_like(v, k_val) * np.sign(v)

    t1 = (2.0 / 3.0) * v
    v2 = v * v
    v3 = v2 * v

    with np.errstate(divide='ignore', invalid='ignore'):
        inner = (-27.0 * m - 2.0 * v3
                 + 3.0 * np.sqrt(3.0)
                 * np.sqrt((27.0 * m ** 2 + 4.0 * m * v3).astype(np.complex128)))
        t2 = np.exp(np.log(inner.astype(np.complex128)) / 3.0)

        t3 = v2.astype(np.complex128) / t2

    cbrt2 = 2.0 ** (1.0 / 3.0)
    sqrt3 = np.sqrt(3.0)

    root = np.zeros((v.size, 3), dtype=np.complex128)
    v_flat = v.ravel()
    t1_flat = t1.ravel()

    with np.errstate(invalid='ignore'):
        root[:, 0] = t1_flat + (cbrt2 / 3.0) * t3 + t2 / (3.0 * cbrt2)
        root[:, 1] = (t1_flat
                      - ((1.0 + 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
                      - ((1.0 - 1j * sqrt3) / (6.0 * cbrt2)) * t2)
        root[:, 2] = (t1_flat
                      - ((1.0 - 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
                      - ((1.0 + 1j * sqrt3) / (6.0 * cbrt2)) * t2)

    bad = np.isnan(root) | np.isinf(root)
    root[bad] = 0.0

    v_rep = np.repeat(v_flat[:, np.newaxis], 3, axis=1)
    sv2 = np.sign(v_rep)
    rsv2 = np.real(root) * sv2

    mask = ((np.abs(np.imag(root)) < epsilon) &
            (rsv2 > 2.0 * np.abs(v_rep) / 3.0) &
            (rsv2 < np.abs(v_rep)))

    filtered = mask * rsv2
    sorted_vals = np.sort(filtered, axis=1)[:, ::-1]
    w = sorted_vals[:, 0] * np.sign(v_flat)

    return np.real(w).reshape(v.shape)


def _newton_w(v, beta, alpha):
    """
    Вычисление проксимального оператора методом Ньютона-Рафсона 
    для произвольных значений alpha.
    """
    iterations = 4
    x = v.copy()

    for _ in range(iterations):
        fd = alpha * np.sign(x) * np.abs(x) ** (alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.abs(x) ** (alpha - 2) + beta
        fdd[fdd == 0] = 1e-10
        x = x - fd / fdd

    x[np.isnan(x)] = 0.0

    z = beta / 2.0 * v ** 2
    f = np.abs(x) ** alpha + beta / 2.0 * (x - v) ** 2
    w = np.where(f < z, x, 0.0)
    return w


def _compute_w(v, beta, alpha):
    """Выбор оптимального метода вычисления проксимального оператора на основе alpha."""
    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    elif abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    elif abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    else:
        return _newton_w(v, beta, alpha)


def solve_image(v, beta, alpha):
    """
    Покомпонентное решение задачи минимизации: 
        min_w |w|^alpha + (beta/2)*(w - v)^2 
    с использованием интерполяции по предварительно вычисленной таблице 
    значений (LUT).
    """
    key = (beta, alpha)

    if key not in _SOLVE_IMAGE_LUT:
        range_val = 10.0
        step = 0.0001
        xx = np.arange(-range_val, range_val + step / 2, step)
        lut_vals = _compute_w(xx, beta, alpha)
        _SOLVE_IMAGE_LUT[key] = interp1d(
            xx, lut_vals, kind='linear', fill_value='extrapolate',
            assume_sorted=True)

    interp_func = _SOLVE_IMAGE_LUT[key]

    orig_shape = v.shape
    w = interp_func(v.ravel())
    return w.reshape(orig_shape)



_OPT_FFT_LUT = None


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """Формирование справочной таблицы оптимальных размеров БПФ (кратных 2,3,5,7,11,13)."""
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
    """Вычисление оптимального размера для быстрого преобразования Фурье."""
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
    Решение уравнения Лапласа с краевыми условиями Дирихле 
    с помощью дискретного синус-преобразования (DST).
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    boundary_image[1:-1, 1:-1] = 0.0

    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]
        + boundary_image[1:H - 1, 0:W - 2]
        + boundary_image[0:H - 2, 1:W - 1]
        + boundary_image[2:H,     1:W - 1]
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
    Дополнение изображения для обеспечения плавного циклического перехода 
    значений на границах (алгоритм Liu & Jia, 2008). 
    Минимизирует артефакты звона при БПФ-деконволюции.

    Параметры
    ---------
    img      : ndarray
        Входное изображение размерности (H, W) или (H, W, C).
    img_size : tuple
        Целевой размер изображения (H_out, W_out).
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
        A = A2

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
        B = B2

        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        C = C2

        A = A[:H_w, :]
        B = B[:, 1:W_w + 1]
        C = C[1:H_w + 1, 1:W_w + 1]

        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret



def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """Билатеральная фильтрация."""
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

    for y_off in range(-fr, fr + 1):
        for x_off in range(-fr, fr + 1):
            w_s = spatial_weight[y_off + fr, x_off + fr]

            n_img = p_img[fr + y_off:fr + y_off + h,
                          fr + x_off:fr + x_off + w, :]
            n_lab = p_lab[fr + y_off:fr + y_off + h,
                          fr + x_off:fr + x_off + w, :]

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
