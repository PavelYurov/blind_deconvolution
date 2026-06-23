"""
utils.py

Вспомогательные функции для алгоритма сверхразрешения на основе байесовской 
комбинации разреженных и неразреженных априорных распределений (BCSNSP-SR).

Функциональность:
    - Построение разреженных матриц операторов деградации (свертка, даунсэмплинг, 
      пространственный сдвиг и поворот с интерполяцией).
    - Разделение и комбинирование многокадровых наблюдений.
    - Быстрые операции свертки в частотной области.
    - Вычислительные утилиты для метода сопряженных градиентов без предобуславливания.

Литература:
[1] S. D. Babacan, R. Molina, A. K. Katsaggelos,
    "Bayesian Super Resolution Image Reconstruction using an l1 Prior",
    ISPA 2009 / Chapter in Bayesian Inference, 2011.
[2] J. Salvador, S. Villena, R. Molina, A. K. Katsaggelos,
    "Bayesian Combination of Sparse and Non-Sparse Priors in
    Image Super Resolution", Digital Signal Processing, 2013.
"""

import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse as sp
from scipy.signal import convolve2d
from scipy.ndimage import zoom


# --- Вспомогательные функции для ядра ---

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """Преобразование функции рассеяния точки (ФРТ) в оптическую передаточную функцию (ОТФ) через БПФ."""
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


def cent_nucleus2fft(kernel: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Центрирование ядра свертки с отражением по обеим осям и вычисление его 2D БПФ."""
    nrk, nck = kernel.shape

    if nrk > nr or nck > nc:
        border = np.maximum(0, (np.array([nrk, nck]) - np.array([nr, nc]))) // 2
        interior = kernel[border[0]:nrk - border[0],
                          border[1]:nck - border[1]]
        fac = kernel.sum() / (interior.sum() + 1e-30)
        kernel = interior * fac
        nrk, nck = kernel.shape

    kernel_flipped = kernel[::-1, ::-1]

    h = np.zeros((nr, nc), dtype=np.float64)
    h[:nrk, :nck] = kernel_flipped

    shift_r = (nrk + 1) // 2 - nrk
    shift_c = (nck + 1) // 2 - nck
    h = np.roll(h, shift_r, axis=0)
    h = np.roll(h, shift_c, axis=1)
    return fft2(h)


def tcent_nucleus2fft(kernel: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """Транспонированный вариант центрирования ядра (выполняется без пространственного отражения)."""
    nrk, nck = kernel.shape

    if nrk > nr or nck > nc:
        border = np.maximum(0, (np.array([nrk, nck]) - np.array([nr, nc]))) // 2
        interior = kernel[border[0]:nrk - border[0],
                          border[1]:nck - border[1]]
        fac = kernel.sum() / (interior.sum() + 1e-30)
        kernel = interior * fac
        nrk, nck = kernel.shape

    h = np.zeros((nr, nc), dtype=np.float64)
    h[:nrk, :nck] = kernel

    shift_r = (nrk + 1) // 2 - nrk
    shift_c = (nck + 1) // 2 - nck
    h = np.roll(h, shift_r, axis=0)
    h = np.roll(h, shift_c, axis=1)
    return fft2(h)


# --- Циркулянтные матрицы ---

def _circulant_matrix(c: np.ndarray) -> sp.csr_matrix:
    """Формирование разреженной циркулянтной матрицы, первый столбец которой задан вектором c."""
    m = len(c)
    rows = np.arange(m)
    indices = np.zeros((m, m), dtype=np.int64)
    for j in range(m):
        indices[:, j] = (rows - j) % m
    data = c[indices]
    return sp.csr_matrix(data)


def circconvmatx2(h: np.ndarray, M: int, N: int) -> sp.csr_matrix:
    """Построение разреженной матрицы круговой свертки для 2D ядра."""
    h = np.atleast_2d(h).astype(np.float64)
    if h.ndim == 0 or h.size == 1:
        return sp.eye(M * N, format='csr')

    mh, nh = h.shape
    if mh != nh:
        raise ValueError("Ядро размытия должно быть квадратным")

    centre = (nh + 1) // 2  

    block_rows = []

    for i_col in range(centre - 1, nh):  
        h0 = h[:, i_col].copy()
        h0 = h0[::-1]  

        row = np.zeros(M, dtype=np.float64)
        row[:len(h0)] = h0
        shift_amount = -(centre - 1)
        row = np.roll(row, shift_amount)

        if np.any(np.abs(row) > 0):
            H0 = _circulant_matrix(row)
        else:
            H0 = sp.csr_matrix((M, M))

        block_rows.append(H0)

    for i_col in range(centre - 2, -1, -1):  
        h0 = h[:, i_col].copy()
        h0 = h0[::-1]

        row = np.zeros(M, dtype=np.float64)
        row[:len(h0)] = h0
        shift_amount = -(centre - 1)
        row = np.roll(row, shift_amount)

        if np.any(np.abs(row) > 0):
            H0 = _circulant_matrix(row)
        else:
            H0 = sp.csr_matrix((M, M))

        block_rows.insert(0, H0)

    n_zero_cols = N - nh
    if n_zero_cols > 0:
        block_rows.append(sp.csr_matrix((M, M * n_zero_cols)))

    first_block_row = sp.hstack(block_rows, format='csr')  

    shift_blocks = -(centre - 1)
    if shift_blocks != 0:
        total_cols = M * N
        shift_cols = shift_blocks * M
        data_dense = first_block_row.toarray()
        data_dense = np.roll(data_dense, shift_cols, axis=1)
        first_block_row = sp.csr_matrix(data_dense)

    rows_list = [first_block_row]
    fbr_dense = first_block_row.toarray()
    for i in range(1, N):
        shifted = np.roll(fbr_dense, M * i, axis=1)
        rows_list.append(sp.csr_matrix(shifted))

    H = sp.vstack(rows_list, format='csr')
    return H


# --- Матрицы операторов деградации ---

def dwnsmpl_matrix(M: int, N: int, res: int) -> sp.csr_matrix:
    """Построение разреженной матрицы субдискретизации (даунсэмплинга)."""
    nopixels = M * N
    m = M // res
    n = N // res

    idx_grid = np.arange(nopixels).reshape(M, N, order='F')  
    dindices = idx_grid[::res, ::res].ravel(order='F')

    if len(dindices) != m * n:
        raise ValueError("Ошибка размера при построении матрицы субдискретизации")

    row = np.arange(m * n)
    col = dindices
    data = np.ones(m * n, dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(m * n, nopixels))
    return A


def shift_matrix(dx: np.ndarray, dy: np.ndarray) -> sp.csr_matrix:
    """Построение разреженной матрицы целочисленного пространственного сдвига."""
    M, N = dx.shape
    nopixels = M * N

    base = np.arange(1, nopixels + 1, dtype=np.int64)  
    dindices = base + dx.ravel(order='F') * M + dy.ravel(order='F')

    dindices = np.clip(dindices, 1, nopixels)
    dindices -= 1  

    row = np.arange(nopixels)
    data = np.ones(nopixels, dtype=np.float64)
    C = sp.csr_matrix((data, (row, dindices)), shape=(nopixels, nopixels))
    return C


def warp_matrix_bilinear(sx: float, sy: float, theta: float,
                         M: int, N: int):
    """
    Построение разреженной матрицы трансформации (сдвиг и поворот) 
    с билинейной интерполяцией субпиксельных значений.
    """
    if M <= N:
        x_range = np.arange(-N // 2, N // 2)
        y_range = np.arange(-N // 2, N // 2)
        X, Y = np.meshgrid(x_range, y_range)
        if M < N:
            X = X[:M, :]
            lo = (N - M) // 2
            hi = lo + M  
            lo_m = int(np.ceil((N - M) / 2))
            hi_m = N - int(np.floor((N - M) / 2))
            Y = Y[lo_m:hi_m, :]
    else:  
        x_range = np.arange(-M // 2, M // 2)
        y_range = np.arange(-M // 2, M // 2)
        X, Y = np.meshgrid(x_range, y_range)
        lo_n = int(np.ceil((M - N) / 2))
        hi_n = M - int(np.floor((M - N) / 2))
        X = X[:, lo_n:hi_n]
        Y = Y[:M, :N]

    Xf = X.ravel(order='F')
    Yf = Y.ravel(order='F')

    indices = np.vstack([Xf, Yf, np.ones(N * M)])
    S = np.array([[np.cos(theta), -np.sin(theta), sx],
                  [np.sin(theta),  np.cos(theta), sy]])
    new_indices = S @ indices

    dx_arr = new_indices[0, :] - indices[0, :]
    dy_arr = new_indices[1, :] - indices[1, :]

    a = dx_arr - np.floor(dx_arr)
    b = dy_arr - np.floor(dy_arr)

    dx_2d = dx_arr.reshape(M, N, order='F')
    dy_2d = dy_arr.reshape(M, N, order='F')

    dx_2d = dx_2d + 1e-6
    dy_2d = dy_2d + 1e-6

    floor_dx = np.floor(dx_2d).astype(np.int64)
    ceil_dx = np.ceil(dx_2d).astype(np.int64)
    floor_dy = np.floor(dy_2d).astype(np.int64)
    ceil_dy = np.ceil(dy_2d).astype(np.int64)

    Lbl = shift_matrix(floor_dx, ceil_dy)
    Lbr = shift_matrix(ceil_dx, ceil_dy)
    Ltl = shift_matrix(floor_dx, floor_dy)
    Ltr = shift_matrix(ceil_dx, floor_dy)

    nopix = N * M
    if np.sum(np.abs(a)) == 0 and np.sum(np.abs(b)) == 0:
        C = shift_matrix(floor_dx, floor_dy)
    else:
        Da_inv = sp.diags(1.0 - a, 0, shape=(nopix, nopix))
        Da = sp.diags(a, 0, shape=(nopix, nopix))
        Db_inv = sp.diags(1.0 - b, 0, shape=(nopix, nopix))
        Db = sp.diags(b, 0, shape=(nopix, nopix))

        C = (Db @ Da_inv @ Lbl
             + Db @ Da @ Lbr
             + Db_inv @ Da_inv @ Ltl
             + Db_inv @ Da @ Ltr)

    return C, Lbl, Lbr, Ltl, Ltr, a, b


# --- Работа с наблюдениями ---

def unwrap_lr(y: np.ndarray, m: int, n: int, L: int):
    """Разделение объединенного вектора наблюдений на отдельные кадры низкого разрешения."""
    npix = m * n
    ys = []
    yvecs = []
    for k in range(L):
        vec = y[npix * k: npix * (k + 1)]
        yvecs.append(vec.copy())
        ys.append(vec.reshape(m, n, order='F'))
    return ys, yvecs


# --- Инициализация SAR ---

def restore_sar(image: np.ndarray, h: np.ndarray,
                tol: float = 1e-6, max_iter: int = 50):
    """Деконволюция в частотной области на основе модели одновременной авторегрессии (SAR)."""
    image = image.astype(np.float64)
    g = fft2(image)
    M, N = image.shape
    npix = M * N

    H = cent_nucleus2fft(h, M, N)
    Ht = tcent_nucleus2fft(h, M, N)
    HtH = Ht * H

    priorn = np.array([[0, -0.25, 0],
                       [-0.25, 1, -0.25],
                       [0, -0.25, 0]], dtype=np.float64)
    priorn = convolve2d(priorn, priorn, mode='full')
    prior = cent_nucleus2fft(priorn, M, N)

    dif = g - H * g
    denom = np.sum(np.conj(dif) * dif).real
    beta0 = npix * npix / (denom + 1e-30)
    if beta0 > 1e6:
        beta0 = 1.0

    alpha0 = ((npix - 1.0) * npix /
              (np.sum(np.conj(g) * (prior * g)).real + 1e-30))

    Q = beta0 * HtH + alpha0 * prior
    f = beta0 * Ht * g / (Q + 1e-30)
    f0 = f.copy()

    alpha = alpha0
    beta = beta0

    for _ in range(max_iter):
        alpha_new = ((npix - 1.0) /
                     (np.sum(np.conj(f) * (prior * f)).real / npix
                      + np.sum(prior / (Q + 1e-30)).real + 1e-30)).real

        residual = g - H * f
        beta_new = (npix /
                    (np.sum(np.conj(residual) * residual).real / npix
                     + np.sum(HtH / (Q + 1e-30)).real + 1e-30)).real

        Q = beta_new * HtH + alpha_new * prior
        f = beta_new * Ht * g / (Q + 1e-30)

        t3 = (np.sum(np.conj(f - f0) * (f - f0)).real /
              (np.sum(np.conj(f0) * f0).real + 1e-30))
        f0 = f.copy()
        alpha = alpha_new
        beta = beta_new

        if t3 <= tol:
            break

    alpha = float(np.real(alpha))
    beta = float(np.real(beta))
    out = np.real(ifft2(f))
    return out, alpha, beta


# --- Решатели ---

def pcg_solve(A, b: np.ndarray, tol: float = 1e-10,
              max_iter: int = 100, x0: np.ndarray = None,
              min_iter: int = 10):
    """
    Решение системы линейных уравнений методом сопряженных градиентов 
    без предобуславливания.
    """
    n = b.shape[0]

    if callable(A):
        matvec = A
    else:
        matvec = lambda v: A @ v

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.copy()

    n2b = np.linalg.norm(b)
    if n2b == 0:
        return np.zeros(n), 0

    r = b - matvec(x)
    normr = np.linalg.norm(r)
    tolb = tol * n2b

    if normr <= tolb:
        return x, 0

    normr_min = normr
    x_min = x.copy()
    rho = 1.0
    flag = 1

    for it in range(1, max_iter + 1):
        z = r.copy()  
        rho_new = np.dot(r, z)

        if it == 1:
            p = z.copy()
        else:
            beta_cg = rho_new / (rho + 1e-30)
            p = z + beta_cg * p

        q = matvec(p)
        pq = np.dot(p, q)

        if pq <= 0:
            break

        alpha_cg = rho_new / pq
        x = x + alpha_cg * p
        r = r - alpha_cg * q
        rho = rho_new

        normr = np.linalg.norm(r)
        if normr < normr_min:
            normr_min = normr
            x_min = x.copy()

        if normr <= tolb and it >= min_iter:
            flag = 0
            break

    if flag == 1:
        x = x_min

    return x, flag


def get_avg_img(y: np.ndarray, W) -> np.ndarray:
    """Вычисление средневзвешенного изображения из набора наблюдений."""
    col_sums = np.array(W.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1e-30
    e = 1.0 / col_sums
    S = sp.diags(e, 0, shape=(W.shape[1], W.shape[1]))
    return S @ (W.T @ y)


# --- Генерация ядер ---

def fatmosfblur(R: float, delta: float, nr: int, nc: int) -> np.ndarray:
    """Генерация ядра размытия атмосферной турбулентности."""
    if nr % 2 == 0:
        nr += 1
    if nc % 2 == 0:
        nc += 1

    centre_r = nr // 2
    centre_c = nc // 2

    yr = centre_r - np.arange(nr)
    xc = centre_c - np.arange(nc)

    xs, ys = np.meshgrid(xc, yr)
    rs = (xs * xs + ys * ys).astype(np.float64)

    h = (rs / (R ** 2) + 1.0) ** (-delta)
    h /= h.sum()
    return h


def fspecial_average(size: int) -> np.ndarray:
    """Генерация усредняющего ядра заданного размера."""
    return np.ones((size, size), dtype=np.float64) / (size * size)


def fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """Генерация гауссовского ядра заданного размера и стандартного отклонения."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fspecial_disk(radius: float) -> np.ndarray:
    """Генерация дискового усредняющего ядра."""
    size = int(2 * radius + 1)
    ax = np.arange(size) - radius
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx ** 2 + yy ** 2) <= radius ** 2
    kernel = mask.astype(np.float64)
    kernel /= kernel.sum()
    return kernel


def imresize(image: np.ndarray, factor, order: int = 3) -> np.ndarray:
    """Изменение размера изображения с использованием сплайновой интерполяции."""
    return zoom(image, float(factor), order=order)


def get_diff_kernels():
    """Получение ядер конечных разностей для вычисления TV регуляризации."""
    dx = np.array([[0, 0, 0],
                   [-1, 1, 0],
                   [0, 0, 0]], dtype=np.float64)
    dy = np.array([[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=np.float64)
    return dx, dy


def get_sar_kernel():
    """Получение ядра регуляризации SAR (Лапласиана)."""
    return np.array([[0, -0.25, 0],
                     [-0.25, 1, -0.25],
                     [0, -0.25, 0]], dtype=np.float64)


def build_coord_grid(M: int, N: int):
    """Построение координатной сетки для процедуры пространственной регистрации."""
    if M <= N:
        x_range = np.arange(-N // 2, N // 2)
        y_range = np.arange(-N // 2, N // 2)
        X, Y = np.meshgrid(x_range, y_range)
        if M < N:
            X = X[:M, :]
            lo = int(np.ceil((N - M) / 2))
            hi = N - int(np.floor((N - M) / 2))
            Y = Y[lo:hi, :]
    else:
        x_range = np.arange(-M // 2, M // 2)
        y_range = np.arange(-M // 2, M // 2)
        X, Y = np.meshgrid(x_range, y_range)
        lo = int(np.ceil((M - N) / 2))
        hi = M - int(np.floor((M - N) / 2))
        X = X[:, lo:hi]
        Y = Y[:M, :N]

    return X.ravel(order='F'), Y.ravel(order='F')