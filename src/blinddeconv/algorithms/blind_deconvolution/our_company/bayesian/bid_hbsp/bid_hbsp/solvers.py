"""
solvers.py

Содержит:
    - Метод сопряженных градиентов (CG) для оценки оригинального изображения (solve_image_cg)
    - Метод сопряженных градиентов для пространства фильтров (solve_filtered_image_cg)
    - Оценку ядра размытия через квадратичное программирование на вероятностном симплексе (solve_kernel_qp_filterspace)
    - Упрощенную оценку ядра с помощью фильтра Винера в частотной области (solve_kernel_fourier, solve_kernel_fourier_filterspace)
    - Алгоритм неслепой деконволюции на базе перевзвешенных наименьших квадратов (final_deconvolution, IRLS)
    - Функцию обновления точности шума (update_noise_precision)

Литература:
[1] Francisco M. Castro-Macias, Fernando Perez-Bueno, et al., "Bayesian Blind 
    Image Deconvolution using a Hyperbolic-Secant prior", ICIP 2024.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg
from typing import Tuple

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
    EPSILON,
    edgetaper,
)

def solve_image_cg(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    gamma_x: np.ndarray,
    gamma_y: np.ndarray,
    max_cg_iter: int = 50,
    cg_tol: float = 1e-6,
    jacobi_mode: str = "scalar",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Решатель в пространстве изображений (метод сопряженных градиентов):
        (beta * H^T * H + D_x^T * Gamma_x * D_x + D_y^T * Gamma_y * D_y) * x = beta * H^T * y

    Параметры
    ----------
    jacobi_mode : 'scalar' | 'perpixel'
        'scalar'   - sigma^2(i) ~ 1 / (beta * ||h||^2 + reg_i + eps)  (быстро)
        'perpixel' - sigma^2(i) ~ 1 / (beta * ifft2(|F_h|^2)(i) + reg_i + eps)
    """
    H, W = y.shape
    N = H * W

    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y)

    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        Av += adjoint_diff_x(gamma_x * forward_diff_x(v))
        Av += adjoint_diff_y(gamma_y * forward_diff_y(v))
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    rhs = beta * np.real(ifft2(F_h_conj * F_y))

    x_flat, _info = cg(A_op, rhs.ravel(), x0=x_init.ravel(),
                        maxiter=max_cg_iter, atol=cg_tol)
    x_out = x_flat.reshape((H, W))

    # Аппроксимация Якоби для дисперсии: sigma^2(i) ~ 1 / diag(A)(i)
    reg_strength = (gamma_x + np.roll(gamma_x, 1, axis=1) +
                    gamma_y + np.roll(gamma_y, 1, axis=0))

    if jacobi_mode == "perpixel":
        # Попиксельно: diag(H^T * H) = ifft2(|F_h|^2)  (точная диагональ)
        diag_hth = np.real(ifft2(F_h_sq))
        sigma_sq = 1.0 / (beta * diag_hth + reg_strength + EPSILON)
    else:
        # Скалярно: diag(H^T * H) ~ ||h||^2
        h_energy = np.sum(h ** 2)
        sigma_sq = 1.0 / (beta * h_energy + reg_strength + EPSILON)

    return np.maximum(x_out, 0.0), sigma_sq


# --- Решатели в пространстве фильтров (Castro-Macias et al. 2024, Раздел IV) ---

def solve_filtered_image_cg(
    y_n: np.ndarray,
    h: np.ndarray,
    x_n_init: np.ndarray,
    beta_n: float,
    theta_n: np.ndarray,
    max_cg_iter: int = 50,
    cg_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Решает задачу для одного отфильтрованного изображения в формулировке VB в пространстве фильтров.

    Система (Уравнения 17-18 из [1]):

        (beta_n * H^T * H + diag(theta_n)) * m_xn = beta_n * H^T * y_n

    Параметры
    ----------
    y_n : (H, W) - псевдо-наблюдение y_n = F_n * y.
    h : (kh, kw) - текущая оценка ядра размытия.
    x_n_init : (H, W) - начальное приближение (warm-start) для CG.
    beta_n : float - точность шума для данного канала фильтра.
    theta_n : (H, W) - диагональные веса гиперболического секанса E[omega_n^i].
    max_cg_iter, cg_tol : критерии остановки CG.

    Возвращает
    -------
    x_n : (H, W) - апостериорное среднее m_xn.
    sigma_sq_n : (H, W) - аппроксимация Якоби для Sigma_xn(i,i).
    """
    H, W = y_n.shape
    N = H * W

    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_yn = fft2(y_n)

    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        # beta_n * H^T * H * v  (слагаемое правдоподобия, частотная область)
        Av = beta_n * np.real(ifft2(F_h_sq * fft2(v)))
        # diag(theta_n) * v  (априорное слагаемое)
        Av += theta_n * v
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    rhs = beta_n * np.real(ifft2(F_h_conj * F_yn))

    x_flat, _ = cg(A_op, rhs.ravel(), x0=x_n_init.ravel(),
                    maxiter=max_cg_iter, atol=cg_tol)
    x_n = x_flat.reshape((H, W))

    # Аппроксимация Якоби для дисперсии: Sigma(i,i) ~ 1 / A(i,i)
    h_energy = np.sum(h ** 2)
    sigma_sq_n = 1.0 / (beta_n * h_energy + theta_n + EPSILON)

    return x_n, sigma_sq_n


def solve_kernel_fourier_filterspace(
    filtered_data: list,
    kernel_shape: Tuple[int, int],
    beta: float,
    lambda_h: float,
    do_threshold: bool = True,
) -> np.ndarray:
    """
    Оценивает ядро размытия по N парам отфильтрованных изображений.

    Использует формулу закрытого типа (фильтр Винера) в частотной области с
    поправкой на неопределенность VB. Краевые строки/столбцы каждого
    отфильтрованного изображения обнуляются для подавления круговых артефактов.

    Примечание:
        Это упрощенная версия. В полной статье [1] решается
        квадратичная задача на симплексе Delta^K с матрицей C_h,
        скорректированной на ковариацию (Уравнения 20-22).

    Параметры
    ----------
    filtered_data : список кортежей (y_n, x_n, sigma_sq_n)
        Каждый кортеж содержит псевдо-наблюдение, апостериорное среднее
        отфильтрованного изображения и его диагональную ковариацию.
    kernel_shape : (kh, kw)
    beta : точность шума (в оригинальном пространстве изображения).
    lambda_h : вес регуляризации ядра.
    do_threshold : обнулить малые элементы ядра и перенормировать.
    """
    H, W = filtered_data[0][0].shape

    numerator = np.zeros((H, W), dtype=np.complex128)
    denominator = np.zeros((H, W), dtype=np.float64)
    uncertainty_total = 0.0

    for y_n, x_n, sigma_sq_n in filtered_data:
        # Маскирование границ (последняя строка и столбец заворачиваются при конечных разностях)
        ym = y_n.copy();  ym[:, -1] = 0.0;  ym[-1, :] = 0.0
        xm = x_n.copy();  xm[:, -1] = 0.0;  xm[-1, :] = 0.0

        F_yn = fft2(ym)
        F_xn = fft2(xm)

        numerator += F_yn * np.conj(F_xn)
        denominator += np.abs(F_xn) ** 2

        # VB поправка на ковариацию (грубо: Trace(X^T * X * Sigma) ~ N * mean(sigma^2))
        uncertainty_total += np.mean(sigma_sq_n)

    denominator += H * W * uncertainty_total + (lambda_h / beta) + EPSILON

    F_h = numerator / denominator
    h = otf2psf(F_h, kernel_shape)

    if do_threshold:
        h = threshold_kernel(h, ratio=0.05)
    else:
        h = project_kernel(h)

    return h


def solve_kernel_qp_filterspace(
    filtered_data: list,
    kernel_shape: Tuple[int, int],
    lambda_h: float = 0.0,
    do_threshold: bool = True,
    threshold_ratio: float = 0.05,
) -> np.ndarray:
    """
    Оценивает ядро размытия путем решения квадратичной задачи (QP) на вероятностном симплексе.

    Реализует Уравнения (20)-(22) из Castro-Macias et al. (2024):

        h_hat = argmin { h^T * C_h * h - 2 * h^T * b_h } для h в Delta^K

    C_h - это автокорреляция апостериорных средних VB m_xn (сумма по N фильтрам)
    с диагональной поправкой на ковариацию. b_h - это кросс-корреляция
    m_xn с псевдо-наблюдениями y_n.

    Обе матрицы эффективно вычисляются через БПФ (FFT); итоговая система
    C_h * h = b_h решается с помощью np.linalg.solve, а результат
    проецируется на симплекс Delta^K = {h >= 0, sum(h) = 1}.

    Параметры
    ----------
    filtered_data : список кортежей (y_n, m_xn, sigma_sq_n)
        N псевдо-наблюдений с их апостериорными средними VB и
        диагональными ковариациями.
    kernel_shape : (kh, kw)
    lambda_h : float
        Опциональная регуляризация Тихонова, добавляемая к диагонали C_h.
    do_threshold : bool
        Обнулить элементы ядра ниже threshold_ratio * max(h)
        перед нормализацией.
    threshold_ratio : float
        Доля от максимума, ниже которой элементы ядра обнуляются.
    """
    kh, kw = kernel_shape
    K = kh * kw
    H_img, W_img = filtered_data[0][0].shape

    # --- Массивы координат ядра ---
    idx = np.arange(K)
    a_coords = idx // kw                          # строка в ядре
    b_coords = idx % kw                           # столбец в ядре

    # C_h[i,j] = R_xx[(a_i - a_j) mod H, (b_i - b_j) mod W]
    da_mat = (a_coords[:, None] - a_coords[None, :]) % H_img    # (K, K)
    db_mat = (b_coords[:, None] - b_coords[None, :]) % W_img    # (K, K)

    # Смещения b_h относительно центра ядра (kh//2, kw//2)
    a_off = (a_coords - kh // 2) % H_img                        # (K,)
    b_off = (b_coords - kw // 2) % W_img                        # (K,)

    C_h = np.zeros((K, K), dtype=np.float64)
    b_h = np.zeros(K, dtype=np.float64)

    for y_n, m_xn, sigma_sq_n in filtered_data:
        F_xn = fft2(m_xn)
        F_yn = fft2(y_n)

        # Автокорреляция: R_xx[d1,d2] = sum_{r,c} x(r+d1, c+d2) * x(r, c)
        # Симметрична, поэтому направление не имеет значения.
        R_xx = np.real(ifft2(np.abs(F_xn) ** 2))

        # Кросс-корреляция для конвенции СВЕРТКИ
        # (psf2otf использует свертку: y = ifft(F_h * F_x)).
        # b_h(a,b) = sum_{r,c} x(r,c) * y(r + offset, c + offset)
        #          = ifft2(conj(F_x) * F_y)[offset]
        R_yx = np.real(ifft2(np.conj(F_xn) * F_yn))

        # --- C_h: часть автокорреляции (Уравнение 21, первое слагаемое) ---
        C_h += R_xx[da_mat, db_mat]

        # --- C_h: VB поправка на ковариацию (Уравнение 21, второе слагаемое) ---
        # При использовании Якоби (диагональной) Sigma_{x_n}: ненулевые элементы только для i == j.
        # sum_l Sigma_{x_n}(i+l, i+l) = sum(sigma^2_n) (периодические границы).
        C_h[np.diag_indices(K)] += float(np.sum(sigma_sq_n))

        # --- b_h (Уравнение 22) ---
        b_h += R_yx[a_off, b_off]

    # Опциональная регуляризация Тихонова
    if lambda_h > 0.0:
        C_h[np.diag_indices(K)] += lambda_h

    # Небольшая добавка (ridge) для численной стабильности
    C_h[np.diag_indices(K)] += 1e-10

    # --- Решение C_h * h = b_h ---
    try:
        h_flat = np.linalg.solve(C_h, b_h)
    except np.linalg.LinAlgError:
        h_flat, _, _, _ = np.linalg.lstsq(C_h, b_h, rcond=None)

    # --- Проекция на симплекс Delta^K ---
    h_flat = np.maximum(h_flat, 0.0)

    if do_threshold:
        peak = np.max(h_flat)
        if peak > 0:
            h_flat[h_flat < threshold_ratio * peak] = 0.0

    h_sum = h_flat.sum()
    if h_sum > EPSILON:
        h_flat /= h_sum
    else:
        h_flat = np.ones(K, dtype=np.float64) / K

    return h_flat.reshape(kh, kw)


# --- Устаревшие решатели (сохранены для обратной совместимости) ---

def solve_kernel_fourier(
    y: np.ndarray,
    x: np.ndarray,
    sigma_sq: np.ndarray,
    kernel_shape: Tuple[int, int],
    beta: float,
    lambda_h: float,
    do_threshold: bool = True,
) -> np.ndarray:
    """
    Оценивает ядро в пространстве градиентов с поправкой на ковариацию и маскированием границ.
    Источник: Castro-Macias (2024) Раздел IV.B и Уравнение (21).
    """
    H, W = y.shape
    
    # 1. Пространство градиентов
    dy_x = forward_diff_x(y)
    dy_y = forward_diff_y(y)
    
    dx_x = forward_diff_x(x)
    dx_y = forward_diff_y(x)
    
    # Маскирование границ
    dy_x[:, -1] = 0.0
    dx_x[:, -1] = 0.0
    dy_y[-1, :] = 0.0
    dx_y[-1, :] = 0.0
    F_dy_x = fft2(dy_x)
    F_dy_y = fft2(dy_y)
    F_dx_x = fft2(dx_x)
    F_dx_y = fft2(dx_y)
    
    # 2. Фильтр Винера
    numerator = (F_dy_x * np.conj(F_dx_x)) + (F_dy_y * np.conj(F_dx_y))
    
    # Автокорреляция X
    denominator = (np.abs(F_dx_x)**2) + (np.abs(F_dx_y)**2)
    
    # Термин VB поправки (Sigma)
    # Trace(T^T * T * Sigma) примерно равна N * mean(Sigma_grad)
    sigma_grad_mean = 2.0 * np.mean(sigma_sq)
    uncertainty_term = H * W * sigma_grad_mean
    
    # Регуляризация
    denominator += uncertainty_term + (lambda_h / beta) + EPSILON
    
    F_h = numerator / denominator
    
    # 3. Пространственное представление
    h = otf2psf(F_h, kernel_shape)
    
    if do_threshold:
        # Пороговая обработка (thresholding)
        h = threshold_kernel(h, ratio=0.05) # 0.05 0.1
    else:
        h = project_kernel(h)
        
    return h


def update_noise_precision(y: np.ndarray, h: np.ndarray, x: np.ndarray, beta_prev: float, damping: float = 0.5) -> float:
    """Обновление точности аддитивного шума (параметр beta)."""
    H, W = y.shape
    N = float(H * W)
    F_h = psf2otf(h, (H, W))
    residual = y - np.real(ifft2(F_h * fft2(x)))
    rss = float(np.sum(residual ** 2))
    beta_new = N / (rss + EPSILON)
    beta = (1.0 - damping) * beta_prev + damping * beta_new
    beta = float(np.clip(beta, 1.0, 1e8))
    return beta


def update_hs_weights(x: np.ndarray, sigma_sq: np.ndarray, b: float) -> Tuple:
    """
    Обновляет веса HS, используя вариационное математическое ожидание E[w].
    Требует sigma_sq (дисперсию) согласно Уравнению 26 в статье.
    """
    dx = forward_diff_x(x)
    dy = forward_diff_y(x)
    from .utils import compute_hs_weights
    return compute_hs_weights(dx, dy, sigma_sq, b)


def final_deconvolution(y: np.ndarray, h: np.ndarray, beta: float, lambda_reg: float) -> np.ndarray:
    """
    Неслепая деконволюция с использованием IRLS и паддингом для устранения краевых артефактов.
    Минимизирует: 0.5 * ||y - h*x||^2 + lambda * ||grad x||_p^p
    """
    # Паддинг
    kh, kw = h.shape
    pad_h = kh
    pad_w = kw
    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    Hp, Wp = y_padded.shape

    x = y_padded.copy()
    
    # IRLS
    p = 0.8
    irls_iters = 15
    
    # Пересчет OTF ядра
    F_h = psf2otf(h, (Hp, Wp))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    
    # (H^T * y)
    F_y = fft2(y_padded)
    rhs_base = beta * np.real(ifft2(F_h_conj * F_y))
    
    for i in range(irls_iters):
        dx = forward_diff_x(x)
        dy = forward_diff_y(x)
        
        # Веса IRLS: w = p * (|grad|^2 + eps)^(p/2 - 1)
        # power = (0.8 - 2) / 2 = -0.6
        power = (p - 2.0) / 2.0
        
        grad_sq_x = dx**2 + 1e-8
        grad_sq_y = dy**2 + 1e-8
        
        wx = p * (grad_sq_x ** power)
        wy = p * (grad_sq_y ** power)
        
        wx = np.clip(wx, 0.0, 1e4)
        wy = np.clip(wy, 0.0, 1e4)
        
        # Регуляризация
        wx *= lambda_reg
        wy *= lambda_reg
        
        x = _solve_image_irls_step(rhs_base, F_h_sq, wx, wy, x, beta, cg_iter=15)
        
        x = np.clip(x, 0.0, 1.0)
    
    x_final = x[pad_h:-pad_h, pad_w:-pad_w]
        
    return x_final

def _solve_image_irls_step(
    rhs: np.ndarray,
    F_h_sq: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    cg_iter: int = 20
) -> np.ndarray:
    """
    Решает один шаг алгоритма IRLS с использованием метода сопряженных градиентов.
    Система: (beta * H^T * H + D_x^T * W_x * D_x + D_y^T * W_y * D_y) * x = rhs
    """
    H, W = x_init.shape
    N = H * W
    
    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        
        # Слагаемое данных: beta * H^T * H * x
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        
        # Априорное слагаемое: D_x^T * (W_x * D_x * v)
        dx_v = forward_diff_x(v)
        dy_v = forward_diff_y(v)
        
        dx_v *= wx
        dy_v *= wy
        
        Av += adjoint_diff_x(dx_v)
        Av += adjoint_diff_y(dy_v)
        
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    
    x_flat, _ = cg(A_op, rhs.ravel(), x0=x_init.ravel(), maxiter=cg_iter, atol=1e-5)
    
    return x_flat.reshape((H, W))

# Заглушка для solve_image_irw, если необходимо избежать ошибок импорта,
# хотя 'cg' является предпочтительным решателем.
def solve_image_irw(*args, **kwargs):
    raise NotImplementedError("IRW solver is not fully adapted for the new variance tracking. Use 'cg'.")