"""
Вариационная байесовская слепая деконволюция с априори полной вариации (TV)

Литература:
    Babacan, S. D., Molina, R., & Katsaggelos, A. K. (2009).
    Variational Bayesian Blind Deconvolution Using a Total Variation Prior.
    IEEE Transactions on Image Processing, 18(1), 12-26.
    DOI: 10.1109/TIP.2008.2007354

Алгоритм использует:
    - Априори полной вариации (TV) для изображения
    - SAR (лапласиан) априори для ядра размытия
    - Гамма гиперприори для всех параметров точности
    - Вариационный вывод с majorization-minimization для TV

Ключевые особенности:
    - TV априори продвигает кусочно-гладкие изображения с резкими гранями
    - Half-quadratic регуляризация обрабатывает недифференцируемость TV
    - Автоматическая оценка гиперпараметров через вариационный Байес
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


# Параметры гиперприори для гамма-распределений (плоские/неинформативные)
# Для Gamma(x | a, b): p(x) ∝ x^(a-1) exp(-b*x), Среднее = a/b
DEFAULT_HYPERPRIORS = {
    'a_alpha_im': 1e-3,   # Априори изображения
    'b_alpha_im': 1e-3,
    'a_alpha_bl': 1e-3,   # Априори размытия
    'b_alpha_bl': 1e-3,
    'a_beta': 1e-3,       # Точность шума
    'b_beta': 1e-3,
}

TV_EPSILON = 1e-6


def _compute_horizontal_gradient(x):
    """
    Вычисляет горизонтальный градиент ∇_h x методом прямых разностей.
    
    (∇_h x)_i = x_{i+1,j} - x_{i,j}
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    
    Возвращает
    ----------
    grad_h : ndarray (H, W)
        Горизонтальный градиент.
    """
    grad_h = np.zeros_like(x)
    grad_h[:, :-1] = x[:, 1:] - x[:, :-1]
    grad_h[:, -1] = 0
    return grad_h


def _compute_vertical_gradient(x):
    """
    Вычисляет вертикальный градиент ∇_v x методом прямых разностей.
    
    (∇_v x)_i = x_{i,j+1} - x_{i,j}
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    
    Возвращает
    ----------
    grad_v : ndarray (H, W)
        Вертикальный градиент.
    """
    grad_v = np.zeros_like(x)
    grad_v[:-1, :] = x[1:, :] - x[:-1, :]
    grad_v[-1, :] = 0
    return grad_v


def _compute_divergence(p_h, p_v):
    """
    Вычисляет дивергенцию (отрицательный сопряжённый к градиенту).
    
    div(p) = ∇_h^T p_h + ∇_v^T p_v
    
    Удовлетворяет свойству сопряжённости: <∇x, p> = <x, div(p)>.
    
    Параметры
    ---------
    p_h : ndarray (H, W)
        Горизонтальная компонента.
    p_v : ndarray (H, W)
        Вертикальная компонента.
    
    Возвращает
    ----------
    div : ndarray (H, W)
        Дивергенция поля (p_h, p_v).
    """
    div = np.zeros_like(p_h)
    div[:, 1:] += p_h[:, :-1]
    div[:, :-1] -= p_h[:, :-1]
    div[1:, :] += p_v[:-1, :]
    div[:-1, :] -= p_v[:-1, :]
    return div


def _compute_tv(x, epsilon=TV_EPSILON):
    """
    Вычисляет полную вариацию изображения x.
    
    TV(x) = Σ_i √((∇_h x_i)² + (∇_v x_i)² + ε)
    
    Ссылка: Ур. (4) в Babacan et al. (2009)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    epsilon : float
        Параметр сглаживания для численной стабильности.
    
    Возвращает
    ----------
    tv : float
        Значение полной вариации.
    """
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    gradient_magnitude = np.sqrt(grad_h**2 + grad_v**2 + epsilon)
    return np.sum(gradient_magnitude)


def _compute_tv_weights(x, epsilon=TV_EPSILON):
    """
    Вычисляет вспомогательные веса u_i для half-quadratic TV регуляризации.
    
    Из подхода majorization-minimization:
        u_i = √((∇_h x_i)² + (∇_v x_i)² + ε)
    
    Эти веса преобразуют TV априори в взвешенную квадратичную форму.
    
    Ссылка: Ур. (10)-(11) в Babacan et al. (2009)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    u : ndarray (H, W)
        Вспомогательные веса.
    """
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    u = np.sqrt(grad_h**2 + grad_v**2 + epsilon)
    return u


def _pad_kernel_for_fft(h, image_shape):
    """
    Дополняет ядро h до размера изображения и центрирует для БПФ.
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Ядро размытия (PSF).
    image_shape : tuple (H, W)
        Размер целевого изображения.
    
    Возвращает
    ----------
    h_padded : ndarray (H, W)
        Дополненное и центрированное ядро.
    """
    H, W = image_shape
    kh, kw = h.shape
    h_padded = np.zeros((H, W), dtype=np.float64)
    h_padded[:kh, :kw] = h
    h_padded = np.roll(h_padded, shift=-kh//2, axis=0)
    h_padded = np.roll(h_padded, shift=-kw//2, axis=1)
    return h_padded


def _extract_kernel_from_padded(h_padded, kernel_shape):
    """
    Извлекает ядро из дополненного представления.
    
    Параметры
    ---------
    h_padded : ndarray (H, W)
        Дополненное ядро.
    kernel_shape : tuple (kh, kw)
        Размер исходного ядра.
    
    Возвращает
    ----------
    h : ndarray (kh, kw)
        Извлечённое ядро.
    """
    kh, kw = kernel_shape
    shifted = np.roll(h_padded, shift=kh//2, axis=0)
    shifted = np.roll(shifted, shift=kw//2, axis=1)
    return shifted[:kh, :kw]


def _compute_gradient_operators_fft(image_shape):
    """
    Вычисляет БПФ операторов градиента для эффективных вычислений.
    
    Возвращает |∇_h|² + |∇_v|² в частотной области.
    
    Параметры
    ---------
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    Lambda_grad : ndarray (H, W)
        Сумма квадратов модулей операторов градиента.
    D_h : ndarray (H, W), complex
        БПФ горизонтального оператора градиента.
    D_v : ndarray (H, W), complex
        БПФ вертикального оператора градиента.
    """
    H, W = image_shape
    
    d_h = np.zeros((H, W), dtype=np.float64)
    d_h[0, 0] = -1
    d_h[0, 1] = 1
    D_h = fft2(d_h)
    
    d_v = np.zeros((H, W), dtype=np.float64)
    d_v[0, 0] = -1
    d_v[1, 0] = 1
    D_v = fft2(d_v)
    
    Lambda_grad = np.abs(D_h)**2 + np.abs(D_v)**2
    
    return Lambda_grad, D_h, D_v


def _compute_laplacian_operator_fft(image_shape):
    """
    Вычисляет |C(k)|² для оператора Лапласа (априори размытия).
    
    Ядро Лапласа:
        [0, -1,  0]
        [-1, 4, -1]
        [0, -1,  0]
    
    Ссылка: Ур. (7) в Babacan et al. (2009)
    
    Параметры
    ---------
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа в частотной области.
    """
    H, W = image_shape
    laplacian = np.array([
        [0, -1,  0],
        [-1, 4, -1],
        [0, -1,  0]
    ], dtype=np.float64)
    
    C_padded = np.zeros((H, W), dtype=np.float64)
    C_padded[:3, :3] = laplacian
    C_padded = np.roll(C_padded, shift=-1, axis=0)
    C_padded = np.roll(C_padded, shift=-1, axis=1)
    
    C_fft = fft2(C_padded)
    Lambda_C = np.abs(C_fft)**2
    
    return Lambda_C


def _update_q_x_tv(Y, M_h, S_h, E_alpha_im, E_beta, u, D_h, D_v, image_shape):
    """
    Обновляет q(x) = N(x | μ_x, Σ_x) с TV априори через half-quadratic.
    
    TV априори аппроксимируется взвешенной квадратичной формой:
        p(x|α_im, u) ∝ exp(-α_im/2 Σ_i ((∇_h x_i)² + (∇_v x_i)²) / u_i)
    
    Это приводит к апостериорной ковариации:
        Σ_x^{-1} = β H^T H + α_im (D_h^T W D_h + D_v^T W D_v)
    
    где W = diag(1/u_i) — матрица весов.
    
    В частотной области (с диагональным приближением):
        S_x(k) ≈ 1 / (β E[|H(k)|²] + α_im (|D_h(k)|² + |D_v(k)|²) / ū)
    
    где ū — среднее u.
    
    Ссылка: Раздел III-A в Babacan et al. (2009)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    E_alpha_im : float
        Мат. ожидание точности априори изображения.
    E_beta : float
        Мат. ожидание точности шума.
    u : ndarray (H, W)
        Вспомогательные веса TV.
    D_h : ndarray (H, W), complex
        БПФ горизонтального оператора градиента.
    D_v : ndarray (H, W), complex
        БПФ вертикального оператора градиента.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    M_x : ndarray (H, W), complex
        Апостериорное среднее x в частотной области.
    S_x : ndarray (H, W)
        Апостериорная дисперсия x в частотной области.
    """
    H, W = image_shape
    
    E_H_squared = np.abs(M_h)**2 + S_h
    
    # Эффективный вес для TV регуляризации (среднее гармоническое как приближение)
    u_safe = np.maximum(u, TV_EPSILON)
    w_mean = 1.0 / np.mean(u_safe)
    
    Lambda_grad_weighted = (np.abs(D_h)**2 + np.abs(D_v)**2) * w_mean
    
    precision_x = E_beta * E_H_squared + E_alpha_im * Lambda_grad_weighted
    precision_x = np.maximum(precision_x, 1e-12)
    
    S_x = 1.0 / precision_x
    M_x = E_beta * S_x * np.conj(M_h) * Y
    
    return M_x, S_x


def _update_q_x_tv_spatial(y, M_h, E_alpha_im, E_beta, u, image_shape, kernel_shape,
                           max_cg_iters=50, cg_tol=1e-4):
    """
    Обновляет q(x) методом сопряжённых градиентов в пространственной области.
    
    Решает линейную систему:
        (β H^T H + α_im L_w) x = β H^T y
    
    где L_w = D_h^T W D_h + D_v^T W D_v — взвешенный лапласиан
    и W = diag(1/u_i).
    
    Этот метод даёт более точные результаты чем приближение через БПФ,
    когда веса u существенно изменяются по изображению.
    
    Ссылка: Ур. (14)-(16) в Babacan et al. (2009)
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое изображение.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    E_alpha_im : float
        Мат. ожидание точности априори изображения.
    E_beta : float
        Мат. ожидание точности шума.
    u : ndarray (H, W)
        Вспомогательные веса TV.
    image_shape : tuple (H, W)
        Размер изображения.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    max_cg_iters : int
        Максимальное число итераций метода сопряжённых градиентов.
    cg_tol : float
        Порог сходимости метода сопряжённых градиентов.
    
    Возвращает
    ----------
    x : ndarray (H, W)
        Апостериорное среднее x (пространственная область).
    """
    Ht, Wt = image_shape
    
    u_safe = np.maximum(u, TV_EPSILON)
    w = 1.0 / u_safe
    
    def apply_H(x):
        """Применяет оператор размытия H."""
        X = fft2(x)
        return np.real(ifft2(M_h * X))
    
    def apply_HT(x):
        """Применяет транспонированный оператор размытия H^T."""
        X = fft2(x)
        return np.real(ifft2(np.conj(M_h) * X))
    
    def apply_Lw(x):
        """Применяет взвешенный лапласиан L_w."""
        grad_h = _compute_horizontal_gradient(x)
        grad_v = _compute_vertical_gradient(x)
        w_grad_h = w * grad_h
        w_grad_v = w * grad_v
        return -_compute_divergence(w_grad_h, w_grad_v)
    
    def apply_A(x):
        """Применяет полный оператор A = β H^T H + α_im L_w."""
        HTHx = apply_HT(apply_H(x))
        Lwx = apply_Lw(x)
        return E_beta * HTHx + E_alpha_im * Lwx
    
    b = E_beta * apply_HT(y)
    
    # Метод сопряжённых градиентов
    x = y.copy()
    r = b - apply_A(x)
    p = r.copy()
    rs_old = np.sum(r * r)
    
    for iteration in range(max_cg_iters):
        Ap = apply_A(p)
        pAp = np.sum(p * Ap)
        
        if pAp < 1e-12:
            break
            
        alpha_cg = rs_old / pAp
        x = x + alpha_cg * p
        r = r - alpha_cg * Ap
        rs_new = np.sum(r * r)
        
        if np.sqrt(rs_new) < cg_tol * np.sqrt(np.sum(b * b)):
            break
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return np.clip(x, 0, None)


def _update_q_h(Y, M_x, S_x, E_alpha_bl, E_beta, Lambda_C):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Априори размытия — SAR:
        p(h|α_bl) ∝ α_bl^(P/2) exp(-α_bl/2 ||Ch||²)
    
    В частотной области:
        S_h(k) = 1 / (β E[|X(k)|²] + α_bl |C(k)|²)
        M_h(k) = β S_h(k) X(k)* Y(k)
    
    Ссылка: Ур. (18)-(19) в Babacan et al. (2009)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_x : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_x : ndarray (H, W)
        Дисперсия изображения в частотной области.
    E_alpha_bl : float
        Мат. ожидание точности априори размытия.
    E_beta : float
        Мат. ожидание точности шума.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    
    Возвращает
    ----------
    M_h : ndarray (H, W), complex
        Апостериорное среднее h в частотной области.
    S_h : ndarray (H, W)
        Апостериорная дисперсия h в частотной области.
    """
    E_X_squared = np.abs(M_x)**2 + S_x
    
    precision_h = E_beta * E_X_squared + E_alpha_bl * Lambda_C
    precision_h = np.maximum(precision_h, 1e-12)
    
    S_h = 1.0 / precision_h
    M_h = E_beta * S_h * np.conj(M_x) * Y
    
    return M_h, S_h


def _project_kernel_constraints(M_h, kernel_shape, image_shape):
    """
    Проецирует размытие на допустимое множество.
    
    Ограничения:
        1. Ограничение носителя (h ненулевое только на kernel_shape)
        2. Неотрицательность (h ≥ 0)
        3. Нормировка (Σ h = 1)
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    M_h_proj : ndarray (H, W), complex
        Спроецированное размытие в частотной области.
    """
    h_spatial = np.real(ifft2(M_h))
    h_kernel = _extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h_kernel = np.maximum(h_kernel, 0.0)
    
    h_sum = np.sum(h_kernel)
    if h_sum > 1e-12:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = _pad_kernel_for_fft(h_kernel, image_shape)
    return fft2(h_padded)


def _update_q_alpha_im(x, u, a_alpha_im, b_alpha_im):
    """
    Обновляет q(α_im), которое является гамма-распределением.
    
    Для TV априори с half-quadratic регуляризацией:
        ⟨α_im⟩ = (N/2 + a_α) / (E[Σ_i (∇x_i)²/(2u_i)] + b_α)
    
    Ссылка: Ур. (20) в Babacan et al. (2009)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    u : ndarray (H, W)
        Вспомогательные веса TV.
    a_alpha_im : float
        Параметр формы гамма-гиперприори.
    b_alpha_im : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_alpha_im : float
        Мат. ожидание точности априори изображения.
    """
    H, W = x.shape
    N = H * W
    
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    
    u_safe = np.maximum(u, TV_EPSILON)
    weighted_gradient_sq = (grad_h**2 + grad_v**2) / (2.0 * u_safe)
    E_tv_weighted = np.sum(weighted_gradient_sq)
    
    numerator = N / 2.0 + a_alpha_im
    denominator = E_tv_weighted + b_alpha_im
    
    E_alpha_im = numerator / max(denominator, 1e-12)
    
    return E_alpha_im


def _update_q_alpha_bl(M_h, S_h, Lambda_C, kernel_shape, a_alpha_bl, b_alpha_bl):
    """
    Обновляет q(α_bl), которое является гамма-распределением.
    
    Для SAR априори на размытие:
        ⟨α_bl⟩ = (P/2 + a_α_bl) / (E[||Ch||²]/2 + b_α_bl)
    
    где P = kh × kw — размер ядра.
    
    Ссылка: Ур. (21) в Babacan et al. (2009)
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    a_alpha_bl : float
        Параметр формы гамма-гиперприори.
    b_alpha_bl : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_alpha_bl : float
        Мат. ожидание точности априори размытия.
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw
    
    E_Ch_squared = np.sum(Lambda_C * (np.abs(M_h)**2 + S_h)) / N
    
    numerator = P / 2.0 + a_alpha_bl
    denominator = E_Ch_squared / 2.0 + b_alpha_bl
    
    E_alpha_bl = numerator / max(denominator, 1e-12)
    
    return E_alpha_bl


def _update_q_beta(Y, M_x, S_x, M_h, S_h, a_beta, b_beta):
    """
    Обновляет q(β), которое является гамма-распределением.
    
    Обновление точности шума:
        ⟨β⟩ = (N/2 + a_β) / (E[||y - Hx||²]/2 + b_β)
    
    где:
        E[||y - Hx||²] = (1/N) Σ_k { |Y(k) - M_h(k) M_x(k)|²
                                    + S_x(k) |M_h(k)|²
                                    + S_h(k) |M_x(k)|²
                                    + S_x(k) S_h(k) }
    
    Ссылка: Ур. (22) в Babacan et al. (2009)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_x : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_x : ndarray (H, W)
        Дисперсия изображения в частотной области.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    a_beta : float
        Параметр формы гамма-гиперприори.
    b_beta : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_beta : float
        Мат. ожидание точности шума.
    """
    H, W = Y.shape
    N = H * W
    
    residual_squared = np.abs(Y - M_h * M_x)**2
    
    variance_terms = (S_x * np.abs(M_h)**2 + 
                      S_h * np.abs(M_x)**2 + 
                      S_x * S_h)
    
    E_error_squared = np.sum(residual_squared + variance_terms) / N
    
    numerator = N / 2.0 + a_beta
    denominator = E_error_squared / 2.0 + b_beta
    
    E_beta = numerator / max(denominator, 1e-12)
    
    return E_beta


def _update_auxiliary_u(x, epsilon=TV_EPSILON):
    """
    Обновляет вспомогательные переменные u для half-quadratic TV.
    
    Из подхода majorization-minimization:
        u_i = √((∇_h x_i)² + (∇_v x_i)² + ε)
    
    Ссылка: Ур. (10)-(11) в Babacan et al. (2009)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    u : ndarray (H, W)
        Обновлённые вспомогательные веса.
    """
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    u = np.sqrt(grad_h**2 + grad_v**2 + epsilon)
    return u


class VBBID_TV(DeconvolutionAlgorithm):
    """
    Вариационная байесовская слепая деконволюция с TV априори.
    
    Совместно оценивает:
        - Латентное изображение x
        - Ядро размытия h
        - Гиперпараметры (α_im, α_bl, β)
    
    используя вариационный вывод с:
        - TV априори для изображения (через half-quadratic)
        - SAR априори для размытия
        - Гамма гиперприори для точностей
    
    Вариационное приближение:
        q(x, h, α_im, α_bl, β) = q(x) q(h) q(α_im) q(α_bl) q(β)
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости (относительное изменение гиперпараметров).
    hyperpriors : dict
        Параметры гамма гиперприори.
    use_spatial_solver : bool
        Использовать метод сопряжённых градиентов в пространственной области.
    max_cg_iters : int
        Максимальное число итераций метода сопряжённых градиентов.
    cg_tol : float
        Порог сходимости метода сопряжённых градиентов.
    tv_epsilon : float
        Параметр сглаживания для TV.
    initial_alpha_im : float или None
        Начальное значение точности априори изображения.
    initial_alpha_bl : float или None
        Начальное значение точности априори размытия.
    initial_beta : float или None
        Начальное значение точности шума.
    apply_kernel_constraints : bool
        Проецировать ядро на допустимое множество.
    verbose : bool
        Выводить прогресс на консоль.
    history : dict
        История сходимости (заполняется после process()).
    hyperparams : dict
        Оценённые гиперпараметры (заполняется после process()).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        hyperpriors: dict = None,
        use_spatial_solver: bool = False,
        max_cg_iters: int = 30,
        cg_tol: float = 1e-4,
        tv_epsilon: float = TV_EPSILON,
        initial_alpha_im: float = None,
        initial_alpha_bl: float = None,
        initial_beta: float = None,
        apply_kernel_constraints: bool = True,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Babacan2009.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число VB-итераций.
        tolerance : float
            Порог сходимости.
        hyperpriors : dict или None
            Параметры гамма гиперприори. Если None, используются значения
            по умолчанию (неинформативные приори).
        use_spatial_solver : bool
            Использовать метод сопряжённых градиентов (точнее, но медленнее).
        max_cg_iters : int
            Максимальное число итераций метода сопряжённых градиентов.
        cg_tol : float
            Порог сходимости метода сопряжённых градиентов.
        tv_epsilon : float
            Параметр сглаживания для TV.
        initial_alpha_im : float или None
            Начальное значение точности априори изображения.
        initial_alpha_bl : float или None
            Начальное значение точности априори размытия.
        initial_beta : float или None
            Начальное значение точности шума.
        apply_kernel_constraints : bool
            Проецировать ядро на допустимое множество.
        verbose : bool
            Выводить прогресс на консоль.
        """
        super().__init__(name='Babacan2009')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.hyperpriors = hyperpriors if hyperpriors else DEFAULT_HYPERPRIORS.copy()
        self.use_spatial_solver = use_spatial_solver
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
        self.tv_epsilon = tv_epsilon
        self.initial_alpha_im = initial_alpha_im
        self.initial_alpha_bl = initial_alpha_bl
        self.initial_beta = initial_beta
        self.apply_kernel_constraints = apply_kernel_constraints
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции с TV априори.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Входное размытое изображение в градациях серого.
            Значения пикселей могут быть в диапазоне [0, 255] или [0, 1].
        
        Возвращает
        ----------
        x_est : ndarray (H, W)
            Восстановленное изображение.
        h_est : ndarray (kh, kw)
            Оценённое ядро размытия (PSF).
        
        Исключения
        ----------
        ValueError
            Если изображение не является 2D массивом.
        
        Примечания
        ----------
        После выполнения метод заполняет атрибуты:
            - self.history: история сходимости гиперпараметров
            - self.hyperparams: финальные значения гиперпараметров
            - self.timer: время выполнения в секундах
        """
        start_time = time.time()
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        y = np.asarray(image, dtype=np.float64)
        H, W = y.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Параметры гиперприори
        a_alpha_im = self.hyperpriors.get('a_alpha_im', 1e-3)
        b_alpha_im = self.hyperpriors.get('b_alpha_im', 1e-3)
        a_alpha_bl = self.hyperpriors.get('a_alpha_bl', 1e-3)
        b_alpha_bl = self.hyperpriors.get('b_alpha_bl', 1e-3)
        a_beta = self.hyperpriors.get('a_beta', 1e-3)
        b_beta = self.hyperpriors.get('b_beta', 1e-3)
        
        # Предвычисление операторов
        Lambda_C = _compute_laplacian_operator_fft((H, W))
        Lambda_grad, D_h, D_v = _compute_gradient_operators_fft((H, W))
        
        # Инициализация изображения: x = y
        x_spatial = y.copy()
        M_x = fft2(x_spatial)
        S_x = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация размытия: дельта-функция
        h_init = np.zeros((kh, kw), dtype=np.float64)
        h_init[kh//2, kw//2] = 1.0
        h_padded = _pad_kernel_for_fft(h_init, (H, W))
        M_h = fft2(h_padded)
        S_h = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация вспомогательной переменной u
        u = _update_auxiliary_u(x_spatial, epsilon=self.tv_epsilon)
        
        # Инициализация гиперпараметров
        noise_var_estimate = max(1e-3 * np.var(y), 1e-6)
        
        if self.initial_beta is None:
            E_beta = 1.0 / noise_var_estimate
        else:
            E_beta = self.initial_beta
            
        if self.initial_alpha_im is None:
            E_alpha_im = 0.1 / noise_var_estimate
        else:
            E_alpha_im = self.initial_alpha_im
            
        if self.initial_alpha_bl is None:
            E_alpha_bl = 1.0
        else:
            E_alpha_bl = self.initial_alpha_bl
        
        # БПФ наблюдения
        Y = fft2(y)
        
        # История
        self.history = {
            'alpha_im': [E_alpha_im],
            'alpha_bl': [E_alpha_bl],
            'beta': [E_beta],
            'tv': [_compute_tv(x_spatial, self.tv_epsilon)]
        }
        
        for iteration in range(self.max_iterations):
            
            E_alpha_im_prev = E_alpha_im
            E_alpha_bl_prev = E_alpha_bl
            E_beta_prev = E_beta
            
            # Обновление вспомогательной u [Ур. 10-11]
            u = _update_auxiliary_u(x_spatial, epsilon=self.tv_epsilon)
            
            # Обновление q(x) [Раздел III-A]
            if self.use_spatial_solver:
                x_spatial = _update_q_x_tv_spatial(
                    y, M_h, E_alpha_im, E_beta, u, (H, W), self.kernel_shape,
                    max_cg_iters=self.max_cg_iters, cg_tol=self.cg_tol
                )
                M_x = fft2(x_spatial)
                _, S_x = _update_q_x_tv(Y, M_h, S_h, E_alpha_im, E_beta, u, D_h, D_v, (H, W))
            else:
                M_x, S_x = _update_q_x_tv(Y, M_h, S_h, E_alpha_im, E_beta, u, D_h, D_v, (H, W))
                x_spatial = np.real(ifft2(M_x))
                x_spatial = np.clip(x_spatial, 0, None)
            
            # Обновление q(h) [Ур. 18-19]
            M_h, S_h = _update_q_h(Y, M_x, S_x, E_alpha_bl, E_beta, Lambda_C)
            
            # Проекция на допустимое множество
            if self.apply_kernel_constraints:
                M_h = _project_kernel_constraints(M_h, self.kernel_shape, (H, W))
            
            # Обновление q(α_im) [Ур. 20]
            E_alpha_im = _update_q_alpha_im(x_spatial, u, a_alpha_im, b_alpha_im)
            
            # Обновление q(α_bl) [Ур. 21]
            E_alpha_bl = _update_q_alpha_bl(M_h, S_h, Lambda_C, self.kernel_shape, 
                                            a_alpha_bl, b_alpha_bl)
            
            # Обновление q(β) [Ур. 22]
            E_beta = _update_q_beta(Y, M_x, S_x, M_h, S_h, a_beta, b_beta)
            
            self.history['alpha_im'].append(E_alpha_im)
            self.history['alpha_bl'].append(E_alpha_bl)
            self.history['beta'].append(E_beta)
            self.history['tv'].append(_compute_tv(x_spatial, self.tv_epsilon))
            
            # Проверка сходимости
            delta_alpha_im = abs(E_alpha_im - E_alpha_im_prev) / max(E_alpha_im_prev, 1e-12)
            delta_alpha_bl = abs(E_alpha_bl - E_alpha_bl_prev) / max(E_alpha_bl_prev, 1e-12)
            delta_beta = abs(E_beta - E_beta_prev) / max(E_beta_prev, 1e-12)
            max_delta = max(delta_alpha_im, delta_alpha_bl, delta_beta)
            
            if self.verbose:
                print(f"Итерация {iteration+1:3d}: α_im={E_alpha_im:.4e}, α_bl={E_alpha_bl:.4e}, "
                      f"β={E_beta:.4e}, TV={self.history['tv'][-1]:.2f}, Δ={max_delta:.4e}")
            
            if max_delta < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
        
        # Извлечение финальных оценок
        x_est = np.clip(x_spatial, 0, None)
        
        h_spatial = np.real(ifft2(M_h))
        h_est = _extract_kernel_from_padded(h_spatial, self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = np.sum(h_est)
        if h_sum > 1e-12:
            h_est = h_est / h_sum
        
        self.hyperparams = {
            'alpha_im': E_alpha_im,
            'alpha_bl': E_alpha_bl,
            'beta': E_beta
        }
        
        self.timer = time.time() - start_time
        
        return x_est, h_est
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """
        Возвращает текущие гиперпараметры алгоритма.
        
        Возвращает
        ----------
        params : list of tuple
            Список кортежей (название_параметра, значение).
        """
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('hyperpriors', self.hyperpriors),
            ('use_spatial_solver', self.use_spatial_solver),
            ('max_cg_iters', self.max_cg_iters),
            ('cg_tol', self.cg_tol),
            ('tv_epsilon', self.tv_epsilon),
            ('initial_alpha_im', self.initial_alpha_im),
            ('initial_alpha_bl', self.initial_alpha_bl),
            ('initial_beta', self.initial_beta),
            ('apply_kernel_constraints', self.apply_kernel_constraints),
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """
        Изменяет гиперпараметры алгоритма.
        
        Параметры
        ---------
        params : dict
            Словарь с параметрами для изменения.
            Ключи — названия параметров, значения — новые значения.
        """
        if 'kernel_shape' in params:
            self.kernel_shape = tuple(params['kernel_shape'])
        if 'max_iterations' in params:
            self.max_iterations = int(params['max_iterations'])
        if 'tolerance' in params:
            self.tolerance = float(params['tolerance'])
        if 'hyperpriors' in params:
            self.hyperpriors = params['hyperpriors']
        if 'use_spatial_solver' in params:
            self.use_spatial_solver = bool(params['use_spatial_solver'])
        if 'max_cg_iters' in params:
            self.max_cg_iters = int(params['max_cg_iters'])
        if 'cg_tol' in params:
            self.cg_tol = float(params['cg_tol'])
        if 'tv_epsilon' in params:
            self.tv_epsilon = float(params['tv_epsilon'])
        if 'initial_alpha_im' in params:
            self.initial_alpha_im = params['initial_alpha_im']
        if 'initial_alpha_bl' in params:
            self.initial_alpha_bl = params['initial_alpha_bl']
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'apply_kernel_constraints' in params:
            self.apply_kernel_constraints = bool(params['apply_kernel_constraints'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """
        Возвращает историю сходимости.
        
        Возвращает
        ----------
        history : dict
            Словарь с историей значений гиперпараметров по итерациям.
            Ключи: 'alpha_im', 'alpha_bl', 'beta', 'tv'.
        """
        return self.history
    
    def get_hyperparams(self) -> dict:
        """
        Возвращает оценённые гиперпараметры.
        
        Возвращает
        ----------
        hyperparams : dict
            Словарь с финальными значениями гиперпараметров.
            Ключи: 'alpha_im', 'alpha_bl', 'beta'.
        """
        return self.hyperparams


# Обратная совместимость: функции-обёртки для старого API
def tv_blind_deconvolution(y, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры для Babacan2009.
    
    Возвращает
    ----------
    x_est : ndarray (H, W)
        Восстановленное изображение.
    h_est : ndarray (kh, kw)
        Оценённое ядро.
    hyperparams : dict
        Оценённые гиперпараметры.
    history : dict
        История сходимости.
    """
    algo = VBBID_TV(kernel_shape=kernel_shape, **kwargs)
    x_est, h_est = algo.process(y)
    return x_est, h_est, algo.hyperparams, algo.history


def run_algorithm(y, kernel_shape, **kwargs):
    """Обёртка для tv_blind_deconvolution()."""
    return tv_blind_deconvolution(y, kernel_shape, **kwargs)
