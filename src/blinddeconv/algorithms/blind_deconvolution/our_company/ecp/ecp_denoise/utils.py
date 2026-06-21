"""
utils.py

Вспомогательные вычислительные функции для алгоритма слепой 
деконволюции на основе априорного распределения экстремальных 
каналов (ECP).

Основано на методе:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao: "Image Deblurring via
    Extreme Channels Prior", CVPR, 2017.

Метод базируется на архитектуре извлечения темного канала с добавлением 
светлого. Внутри основного решателя светлый канал реализуется через 
вычисление темного канала от инвертированного изображения (1 - S). 
В данном файле дополнительно приведена прямая функция вычисления 
светлого канала (bright_channel) для полноты API и совместимости 
со структурой других модулей.
"""

import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn



def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Преобразование функции рассеяния точки (PSF) в оптическую 
    передаточную функцию (OTF).

    Алгоритм:
    1. Дополнение матрицы ядра нулями до заданного размера shape.
    2. Циклический сдвиг для совмещения центра ядра с началом координат.
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
    Преобразование оптической передаточной функции (OTF) обратно 
    в функцию рассеяния точки (PSF) заданного размера.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]



_OPT_FFT_LUT = None  


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """
    Формирование справочной таблицы оптимальных размеров для быстрого 
    преобразования Фурье (числа, являющиеся произведением малых простых: 
    2, 3, 5, 7, 11, 13).
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
    Поиск оптимального размера данных для алгоритма быстрого 
    преобразования Фурье с использованием предварительно рассчитанной 
    справочной таблицы.
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
    Решение уравнения Пуассона с граничными условиями Дирихле 
    через двумерное дискретное синус-преобразование.
    Используется для гладкого заполнения дополненных областей изображения.
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
    Дополнение границ изображения для обеспечения циклической гладкости 
    и предотвращения краевых артефактов при выполнении деконволюции 
    на основе БПФ (метод Liu & Jia).
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



def dark_channel(I: np.ndarray, patch_size: int):
    """
    Вычисление темного канала изображения.

    Для каждого пикселя вычисляется минимальное значение интенсивности 
    в локальном окне заданного размера по всем каналам.

    Параметры
    ---------
    I : ndarray
        Входное изображение.
    patch_size : int
        Размер локального окна поиска.

    Возвращает
    ----------
    J : ndarray
        Изображение темного канала.
    J_index : ndarray
        Массив линейных индексов, указывающих позицию минимального элемента 
        внутри блока. Индексация ведется в столбцовом порядке для обеспечения 
        корректного обратного восстановления значений.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    M, N, C = I.shape
    J = np.zeros((M, N), dtype=np.float64)
    J_index = np.zeros((M, N), dtype=np.int64)

    p = patch_size // 2
    I_pad = np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge')

    for m in range(M):
        for n in range(N):
            patch = I_pad[m:m + patch_size, n:n + patch_size, :]
            tmp = np.min(patch, axis=2)
            tmp_flat = tmp.flatten(order='F')
            tmp_idx = np.argmin(tmp_flat)
            J[m, n] = tmp_flat[tmp_idx]
            J_index[m, n] = tmp_idx + 1  

    return J, J_index



def bright_channel(I: np.ndarray, patch_size: int):
    """
    Вычисление светлого канала изображения.

    Для каждого пикселя вычисляется максимальное значение интенсивности 
    в локальном окне. В основном решателе ECP вместо нее применяется 
    вычисление темного канала от инвертированного изображения (1 - S). 
    Данная функция приводится для прямого доступа к априорному условию.

    Параметры
    ---------
    I : ndarray
        Входное изображение.
    patch_size : int
        Размер локального окна поиска.

    Возвращает
    ----------
    J : ndarray
        Изображение светлого канала.
    J_index : ndarray
        Массив линейных индексов максимумов внутри блока 
        (индексация в столбцовом порядке).
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    M, N, C = I.shape
    J = np.zeros((M, N), dtype=np.float64)
    J_index = np.zeros((M, N), dtype=np.int64)

    p = patch_size // 2
    I_pad = np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge')

    for m in range(M):
        for n in range(N):
            patch = I_pad[m:m + patch_size, n:n + patch_size, :]
            tmp = np.max(patch, axis=2)
            tmp_flat = tmp.flatten(order='F')
            tmp_idx = np.argmax(tmp_flat)
            J[m, n] = tmp_flat[tmp_idx]
            J_index[m, n] = tmp_idx + 1  

    return J, J_index



def assign_dark_channel_to_pixel(S: np.ndarray,
                                 dark_channel_refine: np.ndarray,
                                 dark_channel_index: np.ndarray,
                                 patch_size: int) -> np.ndarray:
    """
    Присвоение уточненных значений экстремального канала обратно пикселям 
    изображения на основе сохраненных индексов.

    Ограничение распространяется на пиксели исходного изображения по координатам, 
    определенным на этапе вычисления темного/светлого каналов. Эта же функция 
    используется для обновления пикселей светлого канала.

    Параметры
    ---------
    S : ndarray
        Текущая оценка изображения.
    dark_channel_refine : ndarray
        Уточненные значения экстремального канала.
    dark_channel_index : ndarray
        Индексы элементов внутри блоков (в столбцовом порядке).
    patch_size : int
        Размер использованного локального окна.

    Возвращает
    ----------
    outImg : ndarray
        Обновленное изображение.
    """
    if S.ndim == 2:
        S_3d = S[:, :, np.newaxis]
    else:
        S_3d = S

    M, N, C = S_3d.shape
    padsize = patch_size // 2

    S_padd = np.pad(S_3d, ((padsize, padsize), (padsize, padsize), (0, 0)),
                    mode='edge')

    for m in range(M):
        for n in range(N):
            patch = S_padd[m:m + patch_size, n:n + patch_size, :].copy()

            if np.min(patch) != dark_channel_refine[m, n]:
                idx = int(dark_channel_index[m, n]) - 1 
                coords = np.unravel_index(idx, (patch_size, patch_size, C),
                                          order='F')
                patch[coords] = dark_channel_refine[m, n]

            S_padd[m:m + patch_size, n:n + patch_size, :] = patch

    outImg = S_padd[padsize:padsize + M, padsize:padsize + N, :]

    outImg[:padsize, :, :] = S_3d[:padsize, :, :]
    outImg[-padsize:, :, :] = S_3d[-padsize:, :, :]
    outImg[:, :padsize, :] = S_3d[:, :padsize, :]
    outImg[:, -padsize:, :] = S_3d[:, -padsize:, :]

    if S.ndim == 2:
        return outImg[:, :, 0]
    return outImg


def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Решение линейной системы A*x = b итерационным методом сопряженных градиентов.
    Операция умножения вектора на матрицу A передается через пользовательскую 
    функцию ax_func.
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
    Центрирование ядра размытия (PSF). 

    Вычисляет пространственный центр масс ядра и смещает его в геометрический 
    центр массива с использованием билинейной интерполяции.
    """
    rows, cols = psf.shape

    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                       np.arange(1, rows + 1, dtype=np.float64))

    if np.sum(psf) == 0:
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
    Вспомогательная функция для подсчета элементов гистограммы 
    с включением правого граничного значения в последний бин.
    """
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size, threshold=None):
    """
    Адаптивное пороговое ограничение градиентов для повышения устойчивости 
    при оценке функции рассеяния точки.

    Малые градиенты (шум, незначительные текстуры) обнуляются. Если порог 
    не задан, он вычисляется автоматически на основе анализа кумулятивной 
    гистограммы направлений и амплитуд градиентов.

    Параметры
    ---------
    latent : ndarray
        Скрытое изображение на текущей итерации.
    psf_size : int или array-like
        Размер ядра размытия.
    threshold : float или None
        Текущее пороговое значение.

    Возвращает
    ----------
    px, py : ndarray
        Градиенты изображения по осям X и Y после обнуления слабых перепадов.
    threshold : float
        Обновленное значение порога.
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
    """
    Формирование двумерного ядра распределения Гаусса с нормализацией суммы к 1.
    """
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """
    Билатеральный фильтр для сохранения краев при подавлении шума.
    Применяется на этапе финальной неслепой деконволюции для выделения 
    и устранения артефактов звона.
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



def graythresh(img: np.ndarray) -> float:
    """
    Вычисление оптимального порога бинаризации методом Оцу.
    Адаптировано для работы с вещественными массивами в диапазоне [0, 1].

    Возвращает
    ----------
    threshold : float
        Пороговое значение.
    """
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



def fftconv(I: np.ndarray, filt: np.ndarray, b_otf: bool = False) -> np.ndarray:
    """
    Быстрая свертка изображения с фильтром в частотной области.

    Параметры
    ---------
    I : ndarray
        Исходное изображение.
    filt : ndarray
        Ядро фильтра или оптическая передаточная функция (OTF).
    b_otf : bool, по умолчанию False
        Флаг, указывающий, что filt уже является OTF (совпадает 
        по размерности с изображением I).

    Возвращает
    ----------
    out : ndarray
        Результат свертки.
    """
    if I.ndim == 3 and I.shape[2] == 3:
        H, W, _ = I.shape
        otf = psf2otf(filt, (H, W))
        out = np.zeros_like(I, dtype=np.float64)
        for c in range(3):
            out[:, :, c] = np.real(np.fft.ifft2(
                np.fft.fft2(I[:, :, c]) * otf))
        return out

    if b_otf:
        return np.real(np.fft.ifft2(np.fft.fft2(I) * filt))
    return np.real(np.fft.ifft2(np.fft.fft2(I) * psf2otf(filt, I.shape)))
