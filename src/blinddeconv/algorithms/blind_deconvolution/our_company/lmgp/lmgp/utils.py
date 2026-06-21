"""
utils.py

Вспомогательные функции и операторы для алгоритмов слепой деконволюции.

Основано на методе:
    L. Chen, F. Fang, T. Wang, G. Zhang: "Blind Image Deblurring
    With Local Maximum Gradient Prior", CVPR, 2019.

Модуль содержит реализации базовых операций цифровой обработки сигналов:
преобразование функций рассеяния точки в оптическую передаточную функцию,
граничное сглаживание для минимизации краевых эффектов Фурье-фильтрации,
адаптивное пороговое отсечение градиентов, двустороннюю фильтрацию, а также
матричные генераторы для вычисления локального максимального градиента.
"""

import numpy as np
from scipy import sparse
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn



def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Преобразование функции рассеяния точки в оптическую передаточную функцию.

    Осуществляет дополнение входного массива нулями до заданного размера,
    выполняет круговой сдвиг для центрирования ядра в точке начала координат
    и применяет двумерное дискретное преобразование Фурье.
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

    Применяет обратное двумерное дискретное преобразование Фурье, извлекает
    действительную часть, выполняет обратный круговой сдвиг и усекает массив
    до исходных размеров функции рассеяния точки.
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

    Параметры
    ---------
    n : int или массив
        Требуемый минимальный размер массива.

    Возвращаемое значение
    ---------------------
    m : int или ndarray
        Ближайший больший или равный размер, оптимальный для алгоритмов БПФ.
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
    Решение уравнения Лапласа с граничными условиями Дирихле на основе
    дискретного синусного преобразования первого типа.
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

    Параметры
    ---------
    img : ndarray
        Входное изображение размерности (H, W) или (H, W, Ch).
    img_size : tuple
        Целевой размер изображения после дополнения (H_out, W_out).

    Возвращаемое значение
    ---------------------
    ret : ndarray
        Изображение с заполненными краевыми областями, обеспечивающими
        гладкий периодический переход.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = int(img_size[0]), int(img_size[1])
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


def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Метод сопряженных градиентов для решения систем линейных уравнений.

    Параметры
    ---------
    x : ndarray
        Начальное приближение решения.
    b : ndarray
        Вектор свободных членов (правая часть системы).
    max_it : int
        Максимальное количество итераций.
    tol : float
        Порог сходимости по норме вектора невязки.
    ax_func : callable
        Функция, вычисляющая произведение матрицы системы на вектор.
    func_param : any
        Дополнительные параметры для функции ax_func.

    Возвращаемое значение
    ---------------------
    x : ndarray
        Найденное решение системы.
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

    Вычисляет центр масс переданного массива и смещает его в геометрический
    центр с использованием билинейной интерполяции.

    Параметры
    ---------
    psf : ndarray
        Исходная матрица функции рассеяния точки.

    Возвращаемое значение
    ---------------------
    result : ndarray
        Отцентрированная матрица функции рассеяния точки.
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


def threshold_pxpy_v1(latent: np.ndarray, psf_size,
                      threshold=None):
    """
    Адаптивное пороговое отсечение шумовых градиентов изображения.

    Вычисляет горизонтальные и вертикальные градиенты. Если порог не задан,
    он оценивается автоматически на основе гистограмм амплитуд градиентов
    в четырех направлениях. Значения градиентов ниже порога обнуляются.

    Параметры
    ---------
    latent : ndarray
        Скрытое изображение для оценки градиентов.
    psf_size : int или массив
        Размер функции рассеяния точки.
    threshold : float или None
        Пороговое значение. Если None, рассчитывается автоматически.

    Возвращаемое значение
    ---------------------
    px, py : ndarray
        Карты градиентов после отсечения.
    threshold : float
        Использованное пороговое значение.
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

    Параметры
    ---------
    img : ndarray
        Входное изображение размерности (H, W) или (H, W, D).
    sigma_s : float
        Пространственное среднеквадратичное отклонение.
    sigma : float
        Амплитудное среднеквадратичное отклонение.

    Возвращаемое значение
    ---------------------
    r_img : ndarray
        Отфильтрованное изображение той же размерности.
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



def gen_partialmat(im_row: int, im_col: int):
    """
    Генерация разреженных матриц операторов частных производных.

    Матрицы формируются для вычисления градиентов через матричное умножение.
    Используется разностная схема: прямая разность на границах и обратная
    во внутренних областях. Индексация строится по столбцам.

    Параметры
    ---------
    im_row : int
        Высота изображения.
    im_col : int
        Ширина изображения.

    Возвращаемое значение
    ---------------------
    px_mat : sparse.csr_matrix
        Оператор горизонтального градиента.
    py_mat : sparse.csr_matrix
        Оператор вертикального градиента.
    """
    M, N = im_row, im_col
    n = M * N
    all_inds = np.arange(n, dtype=np.int64)

    first_row_mask = (all_inds % M) == 0
    first_row = all_inds[first_row_mask]
    not_first_row = all_inds[~first_row_mask]

    r_fr = np.repeat(first_row, 2)
    c_fr = np.empty(2 * len(first_row), dtype=np.int64)
    c_fr[0::2] = first_row
    c_fr[1::2] = first_row + 1
    v_fr = np.tile(np.array([-1.0, 1.0]), len(first_row))

    r_nfr = np.repeat(not_first_row, 2)
    c_nfr = np.empty(2 * len(not_first_row), dtype=np.int64)
    c_nfr[0::2] = not_first_row - 1
    c_nfr[1::2] = not_first_row
    v_nfr = np.tile(np.array([-1.0, 1.0]), len(not_first_row))

    rows_py = np.concatenate([r_fr, r_nfr])
    cols_py = np.concatenate([c_fr, c_nfr])
    vals_py = np.concatenate([v_fr, v_nfr])
    py_mat = sparse.csr_matrix((vals_py, (rows_py, cols_py)), shape=(n, n))

    first_col_mask = all_inds < M
    first_col = all_inds[first_col_mask]
    not_first_col = all_inds[~first_col_mask]

    r_fc = np.repeat(first_col, 2)
    c_fc = np.empty(2 * len(first_col), dtype=np.int64)
    c_fc[0::2] = first_col
    c_fc[1::2] = first_col + M
    v_fc = np.tile(np.array([-1.0, 1.0]), len(first_col))

    r_nfc = np.repeat(not_first_col, 2)
    c_nfc = np.empty(2 * len(not_first_col), dtype=np.int64)
    c_nfc[0::2] = not_first_col
    c_nfc[1::2] = not_first_col - M
    v_nfc = np.tile(np.array([1.0, -1.0]), len(not_first_col))

    rows_px = np.concatenate([r_fc, r_nfc])
    cols_px = np.concatenate([c_fc, c_nfc])
    vals_px = np.concatenate([v_fc, v_nfc])
    px_mat = sparse.csr_matrix((vals_px, (rows_px, cols_px)), shape=(n, n))

    return px_mat, py_mat


def Abs_matrix(I: np.ndarray) -> sparse.dia_matrix:
    """
    Построение разреженной диагональной матрицы знаков элементов.

    Матрица вычисляется как diag(sign(I)), при этом нулевые значения
    обрабатываются как положительные единицы для сохранения стабильности
    оптимизации.

    Параметры
    ---------
    I : ndarray
        Двумерный массив значений.

    Возвращаемое значение
    ---------------------
    Abs_mat : sparse.dia_matrix
        Диагональная матрица знаков соответствующих элементов вектора.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_I = np.abs(I) / I

    abs_I = np.where(np.isfinite(abs_I), abs_I, 1.0)

    diag_vals = abs_I.flatten(order='F')
    n = diag_vals.size
    return sparse.diags(diag_vals, 0, shape=(n, n), format='csr')


def Max_matrix(I: np.ndarray, patch_size: int) -> sparse.csr_matrix:
    """
    Построение разреженной матрицы выбора локального максимума.

    Для каждого пикселя изображения осуществляется поиск позиции максимума
    полной вариации в окрестности заданного размера. Формируется матрица
    перестановочного типа, осуществляющая выбор найденных значений
    при умножении на вектор изображения.

    Параметры
    ---------
    I : ndarray
        Карта полной вариации изображения размерности (M, N).
    patch_size : int
        Размер локальной окрестности поиска максимума (нечетное число).

    Возвращаемое значение
    ---------------------
    max_mat : sparse.csr_matrix
        Матрица выбора локального максимума размерности (M*N, M*N).
    """
    M, N = I.shape
    padsize = patch_size // 2
    h_val = (patch_size + 1) // 2 

    J_index = np.zeros((M, N), dtype=np.int64)

    for m_0 in range(M):
        m_1 = m_0 + 1 
        for n_0 in range(N):
            n_1 = n_0 + 1

            r_start_1 = max(1, m_1 - padsize)
            r_end_1 = min(M, m_1 + padsize)
            c_start_1 = max(1, n_1 - padsize)
            c_end_1 = min(N, n_1 + padsize)

            patch = I[r_start_1 - 1:r_end_1, c_start_1 - 1:c_end_1]
            h1, h2 = patch.shape

            flat = patch.flatten(order='F')
            tmp_idx = int(np.argmax(flat)) + 1
            ori_i = h_val - (patch_size - h1)
            ori_j = h_val - (patch_size - h2)

            if ori_i != h_val and m_1 > h_val:
                ori_i = h1 + 1 - ori_i
            if ori_j != h_val and n_1 > h_val:
                ori_j = h2 + 1 - ori_j

            J_need = int(np.ceil(tmp_idx / h1))
            I_need = tmp_idx - (J_need - 1) * h1

            i_quote = m_1 + I_need - ori_i
            j_quote = n_1 + J_need - ori_j

            J_index[m_0, n_0] = (i_quote - 1) + (j_quote - 1) * M

    n_px = M * N
    sparse_row = np.arange(n_px, dtype=np.int64)
    sparse_col = J_index.flatten(order='F')
    sparse_val = np.ones(n_px, dtype=np.float64)

    return sparse.csr_matrix(
        (sparse_val, (sparse_row, sparse_col)), shape=(n_px, n_px)
    )


def LMG(img: np.ndarray, patch_size: int):
    """
    Вычисление локального максимального градиента и сборка оператора матрицы.

    Реализует математическую модель априорного знания, где каждый пиксель
    отображается в градиент с наибольшей амплитудой в пределах своей окрестности.
    Сборка полного линейного оператора осуществляется из матриц частных
    производных, матриц знаков и матрицы выбора максимума.

    Параметры
    ---------
    img : ndarray
        Полутоновое изображение в формате вещественных чисел размерности (M, N).
    patch_size : int
        Размер квадратной окрестности поиска локального максимума.

    Возвращаемое значение
    ---------------------
    output_img : ndarray
        Карта локального максимального градиента.
    A : sparse.csr_matrix
        Полный разреженный линейный оператор G размерности (M*N, M*N).
    """
    M, N = img.shape
    px_mat, py_mat = gen_partialmat(M, N)
    img_vec = img.flatten(order='F')

    px = (px_mat @ img_vec).reshape((M, N), order='F')
    py = (py_mat @ img_vec).reshape((M, N), order='F')

    abs_x_mat = Abs_matrix(px)
    abs_y_mat = Abs_matrix(py)

    tv = np.abs(px) + np.abs(py)
    max_tv_mat = Max_matrix(tv, patch_size)

    A = max_tv_mat @ (abs_x_mat @ px_mat + abs_y_mat @ py_mat)
    output_vec = A @ img_vec
    output_img = output_vec.reshape((M, N), order='F')

    return output_img, A