# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
solvers_cy.pyx

Ускоренные с помощью Cython функции-решатели для алгоритма BID-HBSP.

Ключевые оптимизации по сравнению с чистым Python:
- Умножение матрицы на вектор для CG: объединено вычисление градиента 
  и взвешенной дивергенции за один проход (в 2 раза меньше аллокаций памяти).
- Оценка ядра QP: плотные циклы для индексации автокорреляции 
  (избегает накладных расходов сложной индексации NumPy).
- Шаг IRLS: объединенное умножение градиента/весов/дивергенции.

Вызовы FFT (fft2/ifft2) остаются в NumPy, так как они занимают основное 
время при больших размерах (256x256 до 1024x1024), но устраняются 
накладные расходы Python вокруг них.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, fabs

np.import_array()

from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg as scipy_cg

from .utils import (
    psf2otf,
    otf2psf,
    precompute_gradient_operators,
    forward_diff_x,
    forward_diff_y,
    adjoint_diff_x,
    adjoint_diff_y,
    compute_hs_weights,
    project_kernel,
    threshold_kernel,
    edgetaper,
)

DEF EPSILON = 1e-12


# --- Вспомогательные функции для объединенных матрично-векторных операций (fused matvec) ---

def _fused_add_grad_prior(
    double[:, :] out not None,
    double[:, :] v not None,
    double[:, :] gamma_x not None,
    double[:, :] gamma_y not None,
):
    """Добавляет D_x^T(Gamma_x * D_x * v) + D_y^T(Gamma_y * D_y * v) к out, на месте."""
    cdef Py_ssize_t H = v.shape[0]
    cdef Py_ssize_t W = v.shape[1]
    cdef Py_ssize_t i, j

    cdef double[:, :] tmp_dx = np.empty((H, W), dtype=np.float64)
    cdef double[:, :] tmp_dy = np.empty((H, W), dtype=np.float64)

    with nogil:
        # Взвешенные прямые разности
        for i in range(H):
            for j in range(W - 1):
                tmp_dx[i, j] = gamma_x[i, j] * (v[i, j + 1] - v[i, j])
            tmp_dx[i, W - 1] = gamma_x[i, W - 1] * (v[i, 0] - v[i, W - 1])

        for i in range(H - 1):
            for j in range(W):
                tmp_dy[i, j] = gamma_y[i, j] * (v[i + 1, j] - v[i, j])
        for j in range(W):
            tmp_dy[H - 1, j] = gamma_y[H - 1, j] * (v[0, j] - v[H - 1, j])

        # Сопряженный оператор прямой разности, добавляется к out
        for i in range(H):
            out[i, 0] += tmp_dx[i, W - 1] - tmp_dx[i, 0]
            for j in range(1, W):
                out[i, j] += tmp_dx[i, j - 1] - tmp_dx[i, j]

        for j in range(W):
            out[0, j] += tmp_dy[H - 1, j] - tmp_dy[0, j]
        for i in range(1, H):
            for j in range(W):
                out[i, j] += tmp_dy[i - 1, j] - tmp_dy[i, j]


def _fused_add_diag_prior(
    double[:, :] out not None,
    double[:, :] v not None,
    double[:, :] theta not None,
):
    """Добавляет diag(theta) * v к out, на месте."""
    cdef Py_ssize_t H = v.shape[0]
    cdef Py_ssize_t W = v.shape[1]
    cdef Py_ssize_t i, j

    with nogil:
        for i in range(H):
            for j in range(W):
                out[i, j] += theta[i, j] * v[i, j]


def _qp_accumulate(
    double[:, :] C_h not None,
    double[:] b_h not None,
    double[:, :] rxx not None,
    double[:, :] ryx not None,
    long long[:, :] da_mat not None,
    long long[:, :] db_mat not None,
    long long[:] a_off not None,
    long long[:] b_off not None,
    double sigma_sum,
    int K,
):
    """Накапливает вклад одного фильтра в C_h и b_h."""
    cdef Py_ssize_t ii, jj

    with nogil:
        for ii in range(K):
            for jj in range(K):
                C_h[ii, jj] += rxx[da_mat[ii, jj], db_mat[ii, jj]]
            C_h[ii, ii] += sigma_sum
            b_h[ii] += ryx[a_off[ii], b_off[ii]]


# --- Решатель CG в пространстве изображений ---

def solve_image_cg(
    y, h, x_init,
    double beta,
    gamma_x, gamma_y,
    int max_cg_iter=50,
    double cg_tol=1e-6,
    str jacobi_mode="scalar",
):
    """Решатель CG в пространстве изображений с объединенным matvec.

    (beta * H^T * H + D_x^T * Gamma_x * D_x + D_y^T * Gamma_y * D_y) * x = beta * H^T * y
    """
    cdef int H_img = y.shape[0]
    cdef int W_img = y.shape[1]
    cdef int N = H_img * W_img
    cdef double h_energy

    F_h = psf2otf(h, (H_img, W_img))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y)

    gx = np.ascontiguousarray(gamma_x, dtype=np.float64)
    gy = np.ascontiguousarray(gamma_y, dtype=np.float64)

    def _mv(v_flat):
        v = v_flat.reshape((H_img, W_img))
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        v_c = np.ascontiguousarray(v, dtype=np.float64)
        _fused_add_grad_prior(Av, v_c, gx, gy)
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_mv, dtype=np.float64)
    rhs = beta * np.real(ifft2(F_h_conj * F_y))

    x_flat, _info = scipy_cg(A_op, rhs.ravel(), x0=x_init.ravel(),
                              maxiter=max_cg_iter, atol=cg_tol)
    x_out = x_flat.reshape((H_img, W_img))

    # Аппроксимация дисперсии Якоби
    reg_strength = (np.asarray(gx) + np.roll(np.asarray(gx), 1, axis=1) +
                    np.asarray(gy) + np.roll(np.asarray(gy), 1, axis=0))

    if jacobi_mode == "perpixel":
        diag_hth = np.real(ifft2(F_h_sq))
        sigma_sq = 1.0 / (beta * diag_hth + reg_strength + EPSILON)
    else:
        h_energy = float(np.sum(np.asarray(h) ** 2))
        sigma_sq = 1.0 / (beta * h_energy + reg_strength + EPSILON)

    return np.maximum(x_out, 0.0), sigma_sq


# --- Решатель CG в пространстве фильтров ---

def solve_filtered_image_cg(
    y_n, h, x_n_init,
    double beta_n,
    theta_n,
    int max_cg_iter=50,
    double cg_tol=1e-6,
):
    """Решатель CG в пространстве фильтров для одного отфильтрованного изображения."""
    cdef int H_img = y_n.shape[0]
    cdef int W_img = y_n.shape[1]
    cdef int N = H_img * W_img
    cdef double h_energy

    F_h = psf2otf(h, (H_img, W_img))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_yn = fft2(y_n)

    tn = np.ascontiguousarray(theta_n, dtype=np.float64)

    def _mv(v_flat):
        v = v_flat.reshape((H_img, W_img))
        Av = beta_n * np.real(ifft2(F_h_sq * fft2(v)))
        v_c = np.ascontiguousarray(v, dtype=np.float64)
        _fused_add_diag_prior(Av, v_c, tn)
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_mv, dtype=np.float64)
    rhs = beta_n * np.real(ifft2(F_h_conj * F_yn))

    x_flat, _ = scipy_cg(A_op, rhs.ravel(), x0=x_n_init.ravel(),
                          maxiter=max_cg_iter, atol=cg_tol)
    x_n = x_flat.reshape((H_img, W_img))

    h_energy = float(np.sum(np.asarray(h) ** 2))
    sigma_sq_n = 1.0 / (beta_n * h_energy + np.asarray(tn) + EPSILON)

    return x_n, sigma_sq_n


# --- Решатель ядра QP (пространство фильтров) ---

def solve_kernel_qp_filterspace(
    list filtered_data,
    kernel_shape,
    double lambda_h=0.0,
    bint do_threshold=True,
    double threshold_ratio=0.05,
):
    """Оценка ядра QP на вероятностном симплексе."""
    cdef int kh = kernel_shape[0]
    cdef int kw = kernel_shape[1]
    cdef int K = kh * kw
    cdef int H_img, W_img
    cdef Py_ssize_t ii, jj
    cdef double sigma_sum, h_sum_val

    H_img = filtered_data[0][0].shape[0]
    W_img = filtered_data[0][0].shape[1]

    # Массивы координат ядра
    idx = np.arange(K, dtype=np.int64)
    a_coords_np = idx // kw
    b_coords_np = idx % kw
    cdef long long[:] a_coords = a_coords_np
    cdef long long[:] b_coords = b_coords_np

    # Предварительное вычисление матриц модульных разностей
    da_mat_np = np.empty((K, K), dtype=np.int64)
    db_mat_np = np.empty((K, K), dtype=np.int64)
    cdef long long[:, :] da_mat = da_mat_np
    cdef long long[:, :] db_mat = db_mat_np

    with nogil:
        for ii in range(K):
            for jj in range(K):
                da_mat[ii, jj] = ((a_coords[ii] - a_coords[jj]) % H_img + H_img) % H_img
                db_mat[ii, jj] = ((b_coords[ii] - b_coords[jj]) % W_img + W_img) % W_img

    # Смещения b_h
    a_off_np = np.empty(K, dtype=np.int64)
    b_off_np = np.empty(K, dtype=np.int64)
    cdef long long[:] a_off = a_off_np
    cdef long long[:] b_off = b_off_np

    with nogil:
        for ii in range(K):
            a_off[ii] = ((a_coords[ii] - kh // 2) % H_img + H_img) % H_img
            b_off[ii] = ((b_coords[ii] - kw // 2) % W_img + W_img) % W_img

    C_h_np = np.zeros((K, K), dtype=np.float64)
    b_h_np = np.zeros(K, dtype=np.float64)
    cdef double[:, :] C_h = C_h_np
    cdef double[:] b_h = b_h_np

    # Итерация по отфильтрованным данным
    for item in filtered_data:
        y_n, m_xn, sigma_sq_n = item
        F_xn = fft2(m_xn)
        F_yn = fft2(y_n)

        R_xx = np.real(ifft2(np.abs(F_xn) ** 2))
        R_yx = np.real(ifft2(np.conj(F_xn) * F_yn))

        rxx_arr = np.ascontiguousarray(R_xx, dtype=np.float64)
        ryx_arr = np.ascontiguousarray(R_yx, dtype=np.float64)
        sigma_sum = float(np.sum(sigma_sq_n))

        _qp_accumulate(C_h, b_h, rxx_arr, ryx_arr,
                        da_mat, db_mat, a_off, b_off,
                        sigma_sum, K)

    # Регуляризация Тихонова + добавка для стабильности
    if lambda_h > 0.0:
        with nogil:
            for ii in range(K):
                C_h[ii, ii] += lambda_h
    with nogil:
        for ii in range(K):
            C_h[ii, ii] += 1e-10

    # Решение
    try:
        h_flat = np.linalg.solve(C_h_np, b_h_np)
    except np.linalg.LinAlgError:
        h_flat, _, _, _ = np.linalg.lstsq(C_h_np, b_h_np, rcond=None)

    # Проекция на симплекс
    h_flat = np.maximum(h_flat, 0.0)
    if do_threshold:
        peak = np.max(h_flat)
        if peak > 0:
            h_flat[h_flat < threshold_ratio * peak] = 0.0

    h_sum_val = h_flat.sum()
    if h_sum_val > EPSILON:
        h_flat /= h_sum_val
    else:
        h_flat = np.ones(K, dtype=np.float64) / K

    return h_flat.reshape(kh, kw)


# --- Устаревший решатель ядра в пространстве изображений (Фурье) ---

def solve_kernel_fourier(
    y, x, sigma_sq, kernel_shape,
    double beta, double lambda_h,
    bint do_threshold=True,
):
    """Оценка ядра в пространстве изображений (Винер в области градиентов)."""
    cdef int H_img = y.shape[0]
    cdef int W_img = y.shape[1]

    dy_x = forward_diff_x(y)
    dy_y = forward_diff_y(y)
    dx_x = forward_diff_x(x)
    dx_y = forward_diff_y(x)

    dy_x_np = np.asarray(dy_x)
    dx_x_np = np.asarray(dx_x)
    dy_y_np = np.asarray(dy_y)
    dx_y_np = np.asarray(dx_y)
    dy_x_np[:, -1] = 0.0
    dx_x_np[:, -1] = 0.0
    dy_y_np[-1, :] = 0.0
    dx_y_np[-1, :] = 0.0

    F_dy_x = fft2(dy_x_np)
    F_dy_y = fft2(dy_y_np)
    F_dx_x = fft2(dx_x_np)
    F_dx_y = fft2(dx_y_np)

    numerator = (F_dy_x * np.conj(F_dx_x)) + (F_dy_y * np.conj(F_dx_y))
    denominator = np.abs(F_dx_x) ** 2 + np.abs(F_dx_y) ** 2

    sigma_grad_mean = 2.0 * float(np.mean(sigma_sq))
    uncertainty_term = H_img * W_img * sigma_grad_mean

    denominator += uncertainty_term + (lambda_h / beta) + EPSILON

    F_h = numerator / denominator
    h = otf2psf(F_h, kernel_shape)

    if do_threshold:
        h_typed = np.ascontiguousarray(h, dtype=np.float64)
        h = threshold_kernel(h_typed, ratio=0.05)
    else:
        h_typed = np.ascontiguousarray(h, dtype=np.float64)
        h = project_kernel(h_typed)

    return np.asarray(h)


# --- Обновление точности шума ---

def update_noise_precision(
    y, h, x,
    double beta_prev, double damping=0.5,
):
    """Обновление beta = N / RSS с демпфированием."""
    cdef int H_img = y.shape[0]
    cdef int W_img = y.shape[1]
    cdef double N_px = <double>(H_img * W_img)
    cdef double rss, beta_new, beta_out

    F_h = psf2otf(h, (H_img, W_img))
    residual = y - np.real(ifft2(F_h * fft2(x)))
    rss = float(np.sum(residual ** 2))
    beta_new = N_px / (rss + EPSILON)
    beta_out = (1.0 - damping) * beta_prev + damping * beta_new
    if beta_out < 1.0:
        beta_out = 1.0
    elif beta_out > 1e8:
        beta_out = 1e8
    return beta_out


# --- Финальная деконволюция (IRLS с паддингом) ---

def _solve_image_irls_step(
    rhs, F_h_sq,
    wx_in, wy_in,
    x_init,
    double beta,
    int cg_iter=20,
):
    """Один шаг IRLS CG с объединенным matvec."""
    cdef int H_img = x_init.shape[0]
    cdef int W_img = x_init.shape[1]
    cdef int N = H_img * W_img

    wx_c = np.ascontiguousarray(wx_in, dtype=np.float64)
    wy_c = np.ascontiguousarray(wy_in, dtype=np.float64)

    def _mv(v_flat):
        v_2d = v_flat.reshape((H_img, W_img))
        Av = beta * np.real(ifft2(F_h_sq * fft2(v_2d)))
        v_c = np.ascontiguousarray(v_2d, dtype=np.float64)
        _fused_add_grad_prior(Av, v_c, wx_c, wy_c)
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_mv, dtype=np.float64)
    x_flat, _ = scipy_cg(A_op, rhs.ravel(), x0=x_init.ravel(),
                          maxiter=cg_iter, atol=1e-5)
    return x_flat.reshape((H_img, W_img))


def final_deconvolution(y, h, double beta, double lambda_reg):
    """Неслепая деконволюция IRLS с дополнением краев (паддингом)."""
    cdef int kh = h.shape[0]
    cdef int kw = h.shape[1]
    cdef int pad_h = kh
    cdef int pad_w = kw
    cdef double p = 0.8
    cdef int irls_iters = 15
    cdef double power = (p - 2.0) / 2.0
    cdef int Hp, Wp

    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    Hp = y_padded.shape[0]
    Wp = y_padded.shape[1]

    x = y_padded.copy()

    F_h = psf2otf(h, (Hp, Wp))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y_padded)
    rhs_base = beta * np.real(ifft2(F_h_conj * F_y))

    for it in range(irls_iters):
        dx = forward_diff_x(x)
        dy = forward_diff_y(x)

        dx_np = np.asarray(dx)
        dy_np = np.asarray(dy)

        grad_sq_x = dx_np ** 2 + 1e-8
        grad_sq_y = dy_np ** 2 + 1e-8

        wx = p * (grad_sq_x ** power)
        wy = p * (grad_sq_y ** power)
        wx = np.clip(wx, 0.0, 1e4) * lambda_reg
        wy = np.clip(wy, 0.0, 1e4) * lambda_reg

        x = _solve_image_irls_step(rhs_base, F_h_sq, wx, wy, x, beta, cg_iter=15)
        x = np.clip(x, 0.0, 1.0)

    return x[pad_h:-pad_h, pad_w:-pad_w]


def update_hs_weights(x, sigma_sq, double b):
    """Удобная обертка для соответствия API на чистом Python."""
    dx = forward_diff_x(x)
    dy = forward_diff_y(x)
    return compute_hs_weights(dx, dy, sigma_sq, b)


def solve_image_irw(*args, **kwargs):
    raise NotImplementedError(
        "Решатель IRW не адаптирован для отслеживания дисперсии. Используйте 'cg'."
    )