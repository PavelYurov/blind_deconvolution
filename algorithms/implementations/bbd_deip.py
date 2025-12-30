"""
Байесовская слепая деконволюция по паре изображений с разной экспозицией

Литература:
    Babacan, S. D., Wang, J., Molina, R., & Katsaggelos, A. K. (2010).
    Bayesian Blind Deconvolution From Differently Exposed Image Pairs.
    IEEE Transactions on Image Processing, 19(11), 2874-2888.
    DOI: 10.1109/TIP.2010.2052263

Алгоритм использует два изображения с разным временем экспозиции:
    - y_l: Длинная экспозиция (размытое из-за дрожания камеры, низкий шум)
    - y_s: Короткая экспозиция (резкое, но очень шумное)

Модели наблюдений:
    y_l = H * x + n_l    (длинная экспозиция: размытие H, шум n_l)
    y_s = x + n_s        (короткая экспозиция: без размытия, шум n_s)

Иерархическая байесовская модель использует:
    - TV априори для латентного изображения x
    - SAR (лапласиан) априори для ядра размытия h
    - Гамма гиперприори для всех параметров точности
    - Разные точности шума для каждого наблюдения

Ключевые особенности:
    - Использует информацию из обоих изображений (резкое-шумное и размытое-чистое)
    - Совместная оценка снижает некорректность по сравнению с одним изображением
    - Автоматическая оценка гиперпараметров через вариационный Байес
    - TV априори сохраняет грани при удалении шума
    - Опциональная регистрация изображений для выравнивания
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import shift as image_shift
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


# Параметры гиперприори для гамма-распределений (плоские/неинформативные)
DEFAULT_HYPERPRIORS = {
    'a_alpha': 1e-3,    # Априори изображения (TV)
    'b_alpha': 1e-3,
    'a_gamma': 1e-3,    # Априори размытия (SAR)
    'b_gamma': 1e-3,
    'a_beta_l': 1e-3,   # Точность шума длинной экспозиции
    'b_beta_l': 1e-3,
    'a_beta_s': 1e-3,   # Точность шума короткой экспозиции
    'b_beta_s': 1e-3,
}

TV_EPSILON = 1e-6


def _compute_horizontal_gradient(x):
    """
    Вычисляет горизонтальный градиент ∇_h x методом прямых разностей.
    
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
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    tv : float
        Значение полной вариации.
    """
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    return np.sum(np.sqrt(grad_h**2 + grad_v**2 + epsilon))


def _pad_kernel_for_fft(h, image_shape):
    """
    Дополняет ядро h до размера изображения и центрирует для БПФ.
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Ядро размытия.
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
    Вычисляет БПФ операторов градиента.
    
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


def _register_images(y_l, y_s, max_shift=20, upsample_factor=1):
    """
    Регистрирует изображение короткой экспозиции относительно длинной.
    
    Изображения с разной экспозицией могут иметь небольшое смещение
    из-за движения камеры между кадрами. Функция оценивает и корректирует
    трансляционное смещение.
    
    Ссылка: Раздел II-B в Babacan et al. (2010)
    
    Параметры
    ---------
    y_l : ndarray (H, W)
        Изображение длинной экспозиции (опорное).
    y_s : ndarray (H, W)
        Изображение короткой экспозиции (регистрируемое).
    max_shift : int
        Максимальное смещение для поиска.
    upsample_factor : int
        Фактор субпиксельной точности (не используется).
    
    Возвращает
    ----------
    y_s_registered : ndarray (H, W)
        Зарегистрированное изображение короткой экспозиции.
    shift_vec : tuple (dy, dx)
        Оценённое смещение.
    """
    from scipy.ndimage import gaussian_filter
    
    y_l_smooth = gaussian_filter(y_l, sigma=2)
    y_s_smooth = gaussian_filter(y_s, sigma=2)
    
    Y_l = fft2(y_l_smooth)
    Y_s = fft2(y_s_smooth)
    
    # Кросс-степенной спектр
    cross_power = Y_l * np.conj(Y_s)
    cross_power /= np.abs(cross_power) + 1e-12
    
    correlation = np.real(ifft2(cross_power))
    
    H, W = y_l.shape
    correlation_shifted = np.fft.fftshift(correlation)
    center = (H // 2, W // 2)
    
    search_region = correlation_shifted[
        center[0] - max_shift : center[0] + max_shift + 1,
        center[1] - max_shift : center[1] + max_shift + 1
    ]
    
    peak_idx = np.unravel_index(np.argmax(search_region), search_region.shape)
    dy = peak_idx[0] - max_shift
    dx = peak_idx[1] - max_shift
    
    y_s_registered = image_shift(y_s, shift=(-dy, -dx), mode='reflect')
    
    return y_s_registered, (dy, dx)


def _update_q_x_dual_observation(Y_l, Y_s, M_h, S_h, E_alpha, E_beta_l, E_beta_s, 
                                  u, D_h, D_v, image_shape):
    """
    Обновляет q(x) = N(x | μ_x, Σ_x) используя ОБА наблюдения.
    
    Правдоподобие объединяет обе модели наблюдений:
        p(y_l, y_s | x, h, β_l, β_s) ∝ 
            exp(-β_l/2 ||y_l - h*x||²) exp(-β_s/2 ||y_s - x||²)
    
    С TV априори (через half-quadratic приближение):
        p(x | α, u) ∝ exp(-α/2 Σ_i [(∇_h x_i)² + (∇_v x_i)²] / u_i)
    
    Апостериорная ковариация в частотной области:
        Σ_x^{-1}(k) = β_l E[|H(k)|²] + β_s + α λ_grad(k)
    
    Апостериорное среднее:
        μ_x(k) = Σ_x(k) [β_l H(k)* Y_l(k) + β_s Y_s(k)]
    
    Ссылка: Ур. (16)-(17) в Babacan et al. (2010)
    
    Параметры
    ---------
    Y_l : ndarray (H, W), complex
        БПФ изображения длинной экспозиции.
    Y_s : ndarray (H, W), complex
        БПФ изображения короткой экспозиции.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    E_alpha : float
        Мат. ожидание точности априори изображения.
    E_beta_l : float
        Мат. ожидание точности шума длинной экспозиции.
    E_beta_s : float
        Мат. ожидание точности шума короткой экспозиции.
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
    
    u_safe = np.maximum(u, TV_EPSILON)
    w_mean = 1.0 / np.mean(u_safe)
    
    Lambda_grad_weighted = (np.abs(D_h)**2 + np.abs(D_v)**2) * w_mean
    
    # Σ_x^{-1}(k) = β_l E[|H(k)|²] + β_s + α λ_grad(k)
    # Член β_s приходит от наблюдения короткой экспозиции (тождественное размытие)
    precision_x = E_beta_l * E_H_squared + E_beta_s + E_alpha * Lambda_grad_weighted
    precision_x = np.maximum(precision_x, 1e-12)
    
    S_x = 1.0 / precision_x
    
    # μ_x(k) = Σ_x(k) [β_l H(k)* Y_l(k) + β_s Y_s(k)]
    M_x = S_x * (E_beta_l * np.conj(M_h) * Y_l + E_beta_s * Y_s)
    
    return M_x, S_x


def _update_q_x_spatial_dual(y_l, y_s, M_h, E_alpha, E_beta_l, E_beta_s, u, 
                             image_shape, max_cg_iters=50, cg_tol=1e-4):
    """
    Обновляет q(x) методом сопряжённых градиентов в пространственной области.
    
    Решает линейную систему:
        (β_l H^T H + β_s I + α L_w) x = β_l H^T y_l + β_s y_s
    
    где L_w = D_h^T W D_h + D_v^T W D_v — взвешенный лапласиан.
    
    Более точный метод для пространственно изменяющихся TV весов.
    
    Ссылка: Раздел III-C в Babacan et al. (2010)
    
    Параметры
    ---------
    y_l : ndarray (H, W)
        Изображение длинной экспозиции.
    y_s : ndarray (H, W)
        Изображение короткой экспозиции.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    E_alpha : float
        Мат. ожидание точности априори изображения.
    E_beta_l : float
        Мат. ожидание точности шума длинной экспозиции.
    E_beta_s : float
        Мат. ожидание точности шума короткой экспозиции.
    u : ndarray (H, W)
        Вспомогательные веса TV.
    image_shape : tuple (H, W)
        Размер изображения.
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
        X = fft2(x)
        return np.real(ifft2(M_h * X))
    
    def apply_HT(x):
        X = fft2(x)
        return np.real(ifft2(np.conj(M_h) * X))
    
    def apply_Lw(x):
        grad_h = _compute_horizontal_gradient(x)
        grad_v = _compute_vertical_gradient(x)
        w_grad_h = w * grad_h
        w_grad_v = w * grad_v
        return -_compute_divergence(w_grad_h, w_grad_v)
    
    def apply_A(x):
        HTHx = apply_HT(apply_H(x))
        Lwx = apply_Lw(x)
        return E_beta_l * HTHx + E_beta_s * x + E_alpha * Lwx
    
    b = E_beta_l * apply_HT(y_l) + E_beta_s * y_s
    
    x = y_s.copy()
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


def _update_q_h(Y_l, M_x, S_x, E_gamma, E_beta_l, Lambda_C):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Размытие оценивается только по наблюдению длинной экспозиции:
        p(y_l | x, h, β_l) ∝ exp(-β_l/2 ||y_l - h*x||²)
    
    С SAR априори:
        p(h | γ) ∝ exp(-γ/2 ||Ch||²)
    
    В частотной области:
        S_h(k) = 1 / (β_l E[|X(k)|²] + γ |C(k)|²)
        M_h(k) = β_l S_h(k) X(k)* Y_l(k)
    
    Ссылка: Ур. (18)-(19) в Babacan et al. (2010)
    
    Параметры
    ---------
    Y_l : ndarray (H, W), complex
        БПФ изображения длинной экспозиции.
    M_x : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_x : ndarray (H, W)
        Дисперсия изображения в частотной области.
    E_gamma : float
        Мат. ожидание точности априори размытия.
    E_beta_l : float
        Мат. ожидание точности шума длинной экспозиции.
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
    
    precision_h = E_beta_l * E_X_squared + E_gamma * Lambda_C
    precision_h = np.maximum(precision_h, 1e-12)
    
    S_h = 1.0 / precision_h
    M_h = E_beta_l * S_h * np.conj(M_x) * Y_l
    
    return M_h, S_h


def _project_kernel_constraints(M_h, kernel_shape, image_shape):
    """
    Проецирует размытие на допустимое множество.
    
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


def _update_q_alpha(x, u, a_alpha, b_alpha):
    """
    Обновляет q(α), которое является гамма-распределением.
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    u : ndarray (H, W)
        Вспомогательные веса TV.
    a_alpha : float
        Параметр формы гамма-гиперприори.
    b_alpha : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_alpha : float
        Мат. ожидание точности априори изображения.
    """
    H, W = x.shape
    N = H * W
    
    grad_h = _compute_horizontal_gradient(x)
    grad_v = _compute_vertical_gradient(x)
    
    u_safe = np.maximum(u, TV_EPSILON)
    weighted_gradient_sq = (grad_h**2 + grad_v**2) / (2.0 * u_safe)
    E_tv_weighted = np.sum(weighted_gradient_sq)
    
    numerator = N / 2.0 + a_alpha
    denominator = E_tv_weighted + b_alpha
    
    E_alpha = numerator / max(denominator, 1e-12)
    
    return E_alpha


def _update_q_gamma(M_h, S_h, Lambda_C, kernel_shape, a_gamma, b_gamma):
    """
    Обновляет q(γ), которое является гамма-распределением.
    
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
    a_gamma : float
        Параметр формы гамма-гиперприори.
    b_gamma : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_gamma : float
        Мат. ожидание точности априори размытия.
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw
    
    E_Ch_squared = np.sum(Lambda_C * (np.abs(M_h)**2 + S_h)) / N
    
    numerator = P / 2.0 + a_gamma
    denominator = E_Ch_squared / 2.0 + b_gamma
    
    E_gamma = numerator / max(denominator, 1e-12)
    
    return E_gamma


def _update_q_beta_l(Y_l, M_x, S_x, M_h, S_h, a_beta_l, b_beta_l):
    """
    Обновляет q(β_l) для шума длинной экспозиции.
    
    Обновление точности шума для наблюдения длинной экспозиции:
        ⟨β_l⟩ = (N/2 + a_β_l) / (E[||y_l - Hx||²]/2 + b_β_l)
    
    Ссылка: Ур. (23) в Babacan et al. (2010)
    
    Параметры
    ---------
    Y_l : ndarray (H, W), complex
        БПФ изображения длинной экспозиции.
    M_x : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_x : ndarray (H, W)
        Дисперсия изображения в частотной области.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    a_beta_l : float
        Параметр формы гамма-гиперприори.
    b_beta_l : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_beta_l : float
        Мат. ожидание точности шума длинной экспозиции.
    """
    H, W = Y_l.shape
    N = H * W
    
    residual_squared = np.abs(Y_l - M_h * M_x)**2
    
    variance_terms = (S_x * np.abs(M_h)**2 + 
                      S_h * np.abs(M_x)**2 + 
                      S_x * S_h)
    
    E_error_squared = np.sum(residual_squared + variance_terms) / N
    
    numerator = N / 2.0 + a_beta_l
    denominator = E_error_squared / 2.0 + b_beta_l
    
    E_beta_l = numerator / max(denominator, 1e-12)
    
    return E_beta_l


def _update_q_beta_s(Y_s, M_x, S_x, a_beta_s, b_beta_s):
    """
    Обновляет q(β_s) для шума короткой экспозиции.
    
    Обновление точности шума для наблюдения короткой экспозиции:
        ⟨β_s⟩ = (N/2 + a_β_s) / (E[||y_s - x||²]/2 + b_β_s)
    
    где:
        E[||y_s - x||²] = (1/N) Σ_k { |Y_s(k) - M_x(k)|² + S_x(k) }
    
    Примечание: наблюдение короткой экспозиции не имеет размытия (H = I).
    
    Ссылка: Ур. (24) в Babacan et al. (2010)
    
    Параметры
    ---------
    Y_s : ndarray (H, W), complex
        БПФ изображения короткой экспозиции.
    M_x : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_x : ndarray (H, W)
        Дисперсия изображения в частотной области.
    a_beta_s : float
        Параметр формы гамма-гиперприори.
    b_beta_s : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_beta_s : float
        Мат. ожидание точности шума короткой экспозиции.
    """
    H, W = Y_s.shape
    N = H * W
    
    residual_squared = np.abs(Y_s - M_x)**2
    variance_terms = S_x
    
    E_error_squared = np.sum(residual_squared + variance_terms) / N
    
    numerator = N / 2.0 + a_beta_s
    denominator = E_error_squared / 2.0 + b_beta_s
    
    E_beta_s = numerator / max(denominator, 1e-12)
    
    return E_beta_s


def _update_auxiliary_u(x, epsilon=TV_EPSILON):
    """
    Обновляет вспомогательные переменные u для half-quadratic TV.
    
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


class BBD_DEIP(DeconvolutionAlgorithm):
    """
    Байесовская слепая деконволюция по паре изображений с разной экспозицией.
    
    Совместно оценивает латентное изображение x и ядро размытия h
    из пары изображений с разной экспозицией:
        - y_l: Длинная экспозиция (размытое, низкий шум)
        - y_s: Короткая экспозиция (резкое, высокий шум)
    
    Использует вариационный байесовский вывод с:
        - TV априори для изображения (через half-quadratic)
        - SAR априори для размытия
        - Гамма гиперприори для всех параметров точности
        - Раздельные модели шума для каждого наблюдения
    
    Вариационное приближение:
        q(x, h, α, γ, β_l, β_s) = q(x) q(h) q(α) q(γ) q(β_l) q(β_s)
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости.
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
    register_images_flag : bool
        Выполнять ли регистрацию изображений.
    verbose : bool
        Выводить прогресс на консоль.
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
        initial_alpha: float = None,
        initial_gamma: float = None,
        initial_beta_l: float = None,
        initial_beta_s: float = None,
        apply_kernel_constraints: bool = True,
        register_images_flag: bool = False,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Babacan2010.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число VB-итераций.
        tolerance : float
            Порог сходимости.
        hyperpriors : dict или None
            Параметры гамма гиперприори.
        use_spatial_solver : bool
            Использовать метод сопряжённых градиентов.
        max_cg_iters : int
            Максимальное число итераций CG.
        cg_tol : float
            Порог сходимости CG.
        tv_epsilon : float
            Параметр сглаживания для TV.
        initial_alpha : float или None
            Начальное значение точности априори изображения.
        initial_gamma : float или None
            Начальное значение точности априори размытия.
        initial_beta_l : float или None
            Начальное значение точности шума длинной экспозиции.
        initial_beta_s : float или None
            Начальное значение точности шума короткой экспозиции.
        apply_kernel_constraints : bool
            Проецировать ядро на допустимое множество.
        register_images_flag : bool
            Выполнять ли регистрацию изображений.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='Babacan2010')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.hyperpriors = hyperpriors if hyperpriors else DEFAULT_HYPERPRIORS.copy()
        self.use_spatial_solver = use_spatial_solver
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
        self.tv_epsilon = tv_epsilon
        self.initial_alpha = initial_alpha
        self.initial_gamma = initial_gamma
        self.initial_beta_l = initial_beta_l
        self.initial_beta_s = initial_beta_s
        self.apply_kernel_constraints = apply_kernel_constraints
        self.register_images_flag = register_images_flag
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
        
        # Для хранения второго изображения
        self._y_s = None
    
    def set_short_exposure_image(self, y_s: np.ndarray) -> None:
        """
        Устанавливает изображение короткой экспозиции.
        
        Параметры
        ---------
        y_s : ndarray (H, W)
            Изображение короткой экспозиции (резкое, шумное).
        """
        self._y_s = np.asarray(y_s, dtype=np.float64)
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции по паре изображений.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Изображение длинной экспозиции (размытое, низкий шум).
            Изображение короткой экспозиции должно быть установлено
            через set_short_exposure_image() перед вызовом.
        
        Возвращает
        ----------
        x_est : ndarray (H, W)
            Восстановленное изображение.
        h_est : ndarray (kh, kw)
            Оценённое ядро размытия.
        
        Исключения
        ----------
        ValueError
            Если изображение короткой экспозиции не установлено.
        """
        start_time = time.time()
        
        if self._y_s is None:
            raise ValueError("Изображение короткой экспозиции не установлено. "
                           "Используйте set_short_exposure_image() перед process().")
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        y_l = np.asarray(image, dtype=np.float64)
        y_s = self._y_s
        
        if y_l.shape != y_s.shape:
            raise ValueError(f"Размеры изображений должны совпадать: y_l={y_l.shape}, y_s={y_s.shape}")
        
        H, W = y_l.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Опциональная регистрация изображений
        if self.register_images_flag:
            y_s, shift_vec = _register_images(y_l, y_s)
            if self.verbose:
                print(f"Изображения выровнены со смещением: {shift_vec}")
        
        # Параметры гиперприори
        a_alpha = self.hyperpriors.get('a_alpha', 1e-3)
        b_alpha = self.hyperpriors.get('b_alpha', 1e-3)
        a_gamma = self.hyperpriors.get('a_gamma', 1e-3)
        b_gamma = self.hyperpriors.get('b_gamma', 1e-3)
        a_beta_l = self.hyperpriors.get('a_beta_l', 1e-3)
        b_beta_l = self.hyperpriors.get('b_beta_l', 1e-3)
        a_beta_s = self.hyperpriors.get('a_beta_s', 1e-3)
        b_beta_s = self.hyperpriors.get('b_beta_s', 1e-3)
        
        # Предвычисление операторов
        Lambda_C = _compute_laplacian_operator_fft((H, W))
        Lambda_grad, D_h, D_v = _compute_gradient_operators_fft((H, W))
        
        # Инициализация изображения: x = y_s (резкое, но шумное)
        x_spatial = y_s.copy()
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
        var_l = np.var(y_l)
        var_s = np.var(y_s)
        
        if self.initial_beta_l is None:
            E_beta_l = 1.0 / max(1e-4 * var_l, 1e-6)
        else:
            E_beta_l = self.initial_beta_l
            
        if self.initial_beta_s is None:
            E_beta_s = 1.0 / max(1e-2 * var_s, 1e-6)
        else:
            E_beta_s = self.initial_beta_s
            
        if self.initial_alpha is None:
            E_alpha = 0.01 / max(var_l, 1e-6)
        else:
            E_alpha = self.initial_alpha
            
        if self.initial_gamma is None:
            E_gamma = 1.0
        else:
            E_gamma = self.initial_gamma
        
        # БПФ наблюдений
        Y_l = fft2(y_l)
        Y_s = fft2(y_s)
        
        # История
        self.history = {
            'alpha': [E_alpha],
            'gamma': [E_gamma],
            'beta_l': [E_beta_l],
            'beta_s': [E_beta_s],
            'tv': [_compute_tv(x_spatial, self.tv_epsilon)]
        }
        
        for iteration in range(self.max_iterations):
            
            E_alpha_prev = E_alpha
            E_gamma_prev = E_gamma
            E_beta_l_prev = E_beta_l
            E_beta_s_prev = E_beta_s
            
            # Обновление вспомогательной u
            u = _update_auxiliary_u(x_spatial, epsilon=self.tv_epsilon)
            
            # Обновление q(x) используя оба наблюдения
            if self.use_spatial_solver:
                x_spatial = _update_q_x_spatial_dual(
                    y_l, y_s, M_h, E_alpha, E_beta_l, E_beta_s, u,
                    (H, W), max_cg_iters=self.max_cg_iters, cg_tol=self.cg_tol
                )
                M_x = fft2(x_spatial)
                _, S_x = _update_q_x_dual_observation(
                    Y_l, Y_s, M_h, S_h, E_alpha, E_beta_l, E_beta_s,
                    u, D_h, D_v, (H, W)
                )
            else:
                M_x, S_x = _update_q_x_dual_observation(
                    Y_l, Y_s, M_h, S_h, E_alpha, E_beta_l, E_beta_s,
                    u, D_h, D_v, (H, W)
                )
                x_spatial = np.real(ifft2(M_x))
                x_spatial = np.clip(x_spatial, 0, None)
            
            # Обновление q(h) только по длинной экспозиции
            M_h, S_h = _update_q_h(Y_l, M_x, S_x, E_gamma, E_beta_l, Lambda_C)
            
            # Проекция на допустимое множество
            if self.apply_kernel_constraints:
                M_h = _project_kernel_constraints(M_h, self.kernel_shape, (H, W))
            
            # Обновление гиперпараметров
            E_alpha = _update_q_alpha(x_spatial, u, a_alpha, b_alpha)
            E_gamma = _update_q_gamma(M_h, S_h, Lambda_C, self.kernel_shape, a_gamma, b_gamma)
            E_beta_l = _update_q_beta_l(Y_l, M_x, S_x, M_h, S_h, a_beta_l, b_beta_l)
            E_beta_s = _update_q_beta_s(Y_s, M_x, S_x, a_beta_s, b_beta_s)
            
            self.history['alpha'].append(E_alpha)
            self.history['gamma'].append(E_gamma)
            self.history['beta_l'].append(E_beta_l)
            self.history['beta_s'].append(E_beta_s)
            self.history['tv'].append(_compute_tv(x_spatial, self.tv_epsilon))
            
            # Проверка сходимости
            delta_alpha = abs(E_alpha - E_alpha_prev) / max(E_alpha_prev, 1e-12)
            delta_gamma = abs(E_gamma - E_gamma_prev) / max(E_gamma_prev, 1e-12)
            delta_beta_l = abs(E_beta_l - E_beta_l_prev) / max(E_beta_l_prev, 1e-12)
            delta_beta_s = abs(E_beta_s - E_beta_s_prev) / max(E_beta_s_prev, 1e-12)
            max_delta = max(delta_alpha, delta_gamma, delta_beta_l, delta_beta_s)
            
            if self.verbose:
                print(f"Итерация {iteration+1:3d}: α={E_alpha:.4e}, γ={E_gamma:.4e}, "
                      f"β_l={E_beta_l:.4e}, β_s={E_beta_s:.4e}, "
                      f"TV={self.history['tv'][-1]:.2f}, Δ={max_delta:.4e}")
            
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
            'alpha': E_alpha,
            'gamma': E_gamma,
            'beta_l': E_beta_l,
            'beta_s': E_beta_s,
            'noise_std_l': 1.0 / np.sqrt(E_beta_l),
            'noise_std_s': 1.0 / np.sqrt(E_beta_s)
        }
        
        self.timer = time.time() - start_time
        
        return x_est, h_est
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """Возвращает текущие гиперпараметры алгоритма."""
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('hyperpriors', self.hyperpriors),
            ('use_spatial_solver', self.use_spatial_solver),
            ('max_cg_iters', self.max_cg_iters),
            ('cg_tol', self.cg_tol),
            ('tv_epsilon', self.tv_epsilon),
            ('initial_alpha', self.initial_alpha),
            ('initial_gamma', self.initial_gamma),
            ('initial_beta_l', self.initial_beta_l),
            ('initial_beta_s', self.initial_beta_s),
            ('apply_kernel_constraints', self.apply_kernel_constraints),
            ('register_images_flag', self.register_images_flag),
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """Изменяет гиперпараметры алгоритма."""
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
        if 'initial_alpha' in params:
            self.initial_alpha = params['initial_alpha']
        if 'initial_gamma' in params:
            self.initial_gamma = params['initial_gamma']
        if 'initial_beta_l' in params:
            self.initial_beta_l = params['initial_beta_l']
        if 'initial_beta_s' in params:
            self.initial_beta_s = params['initial_beta_s']
        if 'apply_kernel_constraints' in params:
            self.apply_kernel_constraints = bool(params['apply_kernel_constraints'])
        if 'register_images_flag' in params:
            self.register_images_flag = bool(params['register_images_flag'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """Возвращает историю сходимости."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает оценённые гиперпараметры."""
        return self.hyperparams


# Обратная совместимость
def dual_exposure_blind_deconvolution(y_l, y_s, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    y_l : ndarray (H, W)
        Изображение длинной экспозиции.
    y_s : ndarray (H, W)
        Изображение короткой экспозиции.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры.
    
    Возвращает
    ----------
    x_est, h_est, hyperparams, history
    """
    algo = Babacan2010(kernel_shape=kernel_shape, **kwargs)
    algo.set_short_exposure_image(y_s)
    x_est, h_est = algo.process(y_l)
    return x_est, h_est, algo.hyperparams, algo.history


def single_image_blind_deconvolution(y, kernel_shape, **kwargs):
    """
    Слепая деконволюция по одному изображению с TV априори.
    
    Резервный вариант, когда доступно только одно изображение.
    Использует ту же модель, но с β_s = 0.
    """
    y_s_synthetic = y.copy()
    kwargs['initial_beta_s'] = 1e-12
    return dual_exposure_blind_deconvolution(y, y_s_synthetic, kernel_shape, **kwargs)


def run_algorithm(y_l, y_s, kernel_shape, **kwargs):
    """Обёртка для dual_exposure_blind_deconvolution()."""
    return dual_exposure_blind_deconvolution(y_l, y_s, kernel_shape, **kwargs)


def run_single_image(y, kernel_shape, **kwargs):
    """Обёртка для single_image_blind_deconvolution()."""
    return single_image_blind_deconvolution(y, kernel_shape, **kwargs)
