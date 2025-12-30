"""
Разреженная байесовская слепая деконволюция с оценкой параметров

Литература:
    Amizic, B., Molina, R., & Katsaggelos, A. K. (2012).
    Sparse Bayesian blind image deconvolution with parameter estimation.
    EURASIP Journal on Image and Video Processing, 2012(1), 20.
    DOI: 10.1186/1687-5281-2012-20

Алгоритм использует:
    - Невыпуклую ℓp квази-норму (0 < p < 1) для априори изображения (разреженность)
    - TV априори для ядра размытия
    - Majorization-Minimization (MM) для невыпуклой оптимизации
    - MAP оценку с автоматической оценкой параметров
    - Многомасштабную реализацию для улучшенной сходимости

Ключевые особенности:
    - ℓp априори продвигает разреженные градиенты (более резкие грани, чем TV)
    - Автоматическая оценка всех гиперпараметров (α, β, γ)
    - MM границы делают невыпуклую оптимизацию решаемой
    - Многомасштабный подход предотвращает локальные минимумы

Модель наблюдения:
    y = Hx + n
    
где:
    y: наблюдаемое размытое и зашумленное изображение (N × 1)
    x: оригинальное резкое изображение (N × 1)
    H: матрица размытия (циркулянтная, созданная из PSF h)
    n: гауссов шум с точностью β
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


# Параметры по умолчанию
DEFAULT_P = 0.8           # Показатель ℓp квази-нормы (0 < p < 1)
DEFAULT_LAMBDA_1 = 2/3    # Вес априори изображения
DEFAULT_LAMBDA_2 = 1e-3   # Вес априори размытия

EPSILON = 1e-10


def _compute_gradient_h(x):
    """
    Вычисляет горизонтальный градиент Δ^h x методом прямых разностей.
    
    (Δ^h x)_{i,j} = x_{i,j+1} - x_{i,j}
    
    Ссылка: Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    
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
    grad_h[:, -1] = x[:, 0] - x[:, -1]  # Периодические границы
    return grad_h


def _compute_gradient_v(x):
    """
    Вычисляет вертикальный градиент Δ^v x методом прямых разностей.
    
    (Δ^v x)_{i,j} = x_{i+1,j} - x_{i,j}
    
    Ссылка: Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    
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
    grad_v[-1, :] = x[0, :] - x[-1, :]  # Периодические границы
    return grad_v


def _compute_gradient_d1(x):
    """
    Вычисляет первый диагональный градиент Δ^{d1} x.
    
    (Δ^{d1} x)_{i,j} = x_{i+1,j+1} - x_{i,j}
    
    Ссылка: Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    
    Возвращает
    ----------
    grad_d1 : ndarray (H, W)
        Первый диагональный градиент.
    """
    grad_d1 = np.zeros_like(x)
    grad_d1[:-1, :-1] = x[1:, 1:] - x[:-1, :-1]
    grad_d1[-1, :-1] = x[0, 1:] - x[-1, :-1]
    grad_d1[:-1, -1] = x[1:, 0] - x[:-1, -1]
    grad_d1[-1, -1] = x[0, 0] - x[-1, -1]
    return grad_d1


def _compute_gradient_d2(x):
    """
    Вычисляет второй диагональный градиент Δ^{d2} x.
    
    (Δ^{d2} x)_{i,j} = x_{i+1,j-1} - x_{i,j}
    
    Ссылка: Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    
    Возвращает
    ----------
    grad_d2 : ndarray (H, W)
        Второй диагональный градиент.
    """
    grad_d2 = np.zeros_like(x)
    grad_d2[:-1, 1:] = x[1:, :-1] - x[:-1, 1:]
    grad_d2[-1, 1:] = x[0, :-1] - x[-1, 1:]
    grad_d2[:-1, 0] = x[1:, -1] - x[:-1, 0]
    grad_d2[-1, 0] = x[0, -1] - x[-1, 0]
    return grad_d2


def _compute_divergence_h(p):
    """
    Вычисляет сопряжённый (транспонированный) оператор горизонтального градиента.
    
    Параметры
    ---------
    p : ndarray (H, W)
        Входной массив.
    
    Возвращает
    ----------
    div_h : ndarray (H, W)
        Результат применения сопряжённого оператора.
    """
    div_h = np.zeros_like(p)
    div_h[:, 1:] -= p[:, :-1]
    div_h[:, :-1] += p[:, :-1]
    div_h[:, 0] -= p[:, -1]
    div_h[:, -1] += p[:, -1]
    return div_h


def _compute_divergence_v(p):
    """
    Вычисляет сопряжённый (транспонированный) оператор вертикального градиента.
    
    Параметры
    ---------
    p : ndarray (H, W)
        Входной массив.
    
    Возвращает
    ----------
    div_v : ndarray (H, W)
        Результат применения сопряжённого оператора.
    """
    div_v = np.zeros_like(p)
    div_v[1:, :] -= p[:-1, :]
    div_v[:-1, :] += p[:-1, :]
    div_v[0, :] -= p[-1, :]
    div_v[-1, :] += p[-1, :]
    return div_v


def _compute_divergence_d1(p):
    """Вычисляет сопряжённый оператор первого диагонального градиента."""
    div_d1 = np.zeros_like(p)
    div_d1[1:, 1:] -= p[:-1, :-1]
    div_d1[:-1, :-1] += p[:-1, :-1]
    div_d1[0, 1:] -= p[-1, :-1]
    div_d1[-1, :-1] += p[-1, :-1]
    div_d1[1:, 0] -= p[:-1, -1]
    div_d1[:-1, -1] += p[:-1, -1]
    div_d1[0, 0] -= p[-1, -1]
    div_d1[-1, -1] += p[-1, -1]
    return div_d1


def _compute_divergence_d2(p):
    """Вычисляет сопряжённый оператор второго диагонального градиента."""
    div_d2 = np.zeros_like(p)
    div_d2[1:, :-1] -= p[:-1, 1:]
    div_d2[:-1, 1:] += p[:-1, 1:]
    div_d2[0, :-1] -= p[-1, 1:]
    div_d2[-1, 1:] += p[-1, 1:]
    div_d2[1:, -1] -= p[:-1, 0]
    div_d2[:-1, 0] += p[:-1, 0]
    div_d2[0, -1] -= p[-1, 0]
    div_d2[-1, 0] += p[-1, 0]
    return div_d2


def _compute_lp_prior(x, p=0.8):
    """
    Вычисляет ℓp квази-норму априори для изображения.
    
    Φ(x) = Σ_{d∈D} 2^{1-ω(d)} Σ_i |Δ^d_i(x)|^p
    
    где D = {h, v, d1, d2} — направления градиента и
    ω(d) = 1 для горизонтального/вертикального, ω(d) = 2 для диагоналей.
    
    Ссылка: Ур. (5) в Amizic et al. (2012)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Входное изображение.
    p : float
        Показатель ℓp квази-нормы (0 < p < 1).
    
    Возвращает
    ----------
    phi : float
        Значение ℓp квази-нормы.
    """
    grad_h = _compute_gradient_h(x)
    grad_v = _compute_gradient_v(x)
    grad_d1 = _compute_gradient_d1(x)
    grad_d2 = _compute_gradient_d2(x)
    
    # Веса: 2^{1-ω(d)}
    # ω(h) = ω(v) = 1 → вес = 2^0 = 1
    # ω(d1) = ω(d2) = 2 → вес = 2^{-1} = 0.5
    phi = (np.sum(np.abs(grad_h)**p) + 
           np.sum(np.abs(grad_v)**p) +
           0.5 * np.sum(np.abs(grad_d1)**p) + 
           0.5 * np.sum(np.abs(grad_d2)**p))
    
    return phi


def _compute_tv(h, epsilon=EPSILON):
    """
    Вычисляет полную вариацию ядра размытия h.
    
    TV(h) = Σ_i √((Δ^h h_i)² + (Δ^v h_i)² + ε)
    
    Ссылка: Ур. (6) в Amizic et al. (2012)
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Ядро размытия.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    tv : float
        Значение полной вариации.
    """
    grad_h = _compute_gradient_h(h)
    grad_v = _compute_gradient_v(h)
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
    Вычисляет БПФ операторов градиента для эффективных вычислений.
    
    Возвращает D_h, D_v, D_d1, D_d2 в частотной области.
    
    Параметры
    ---------
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    D_h, D_v, D_d1, D_d2 : ndarray (H, W), complex
        БПФ операторов градиента.
    Lambda_sum : ndarray (H, W)
        Сумма квадратов модулей (с весами).
    """
    H, W = image_shape
    
    # Горизонтальный градиент: [−1, 1]
    d_h = np.zeros((H, W), dtype=np.float64)
    d_h[0, 0] = -1
    d_h[0, 1] = 1
    D_h = fft2(d_h)
    
    # Вертикальный градиент: [−1; 1]
    d_v = np.zeros((H, W), dtype=np.float64)
    d_v[0, 0] = -1
    d_v[1, 0] = 1
    D_v = fft2(d_v)
    
    # Первая диагональ: [-1 в (0,0), 1 в (1,1)]
    d_d1 = np.zeros((H, W), dtype=np.float64)
    d_d1[0, 0] = -1
    d_d1[1, 1] = 1
    D_d1 = fft2(d_d1)
    
    # Вторая диагональ: [-1 в (0,0), 1 в (1,-1)]
    d_d2 = np.zeros((H, W), dtype=np.float64)
    d_d2[0, 0] = -1
    d_d2[1, -1] = 1
    D_d2 = fft2(d_d2)
    
    # Сумма квадратов модулей (с весами)
    Lambda_sum = (np.abs(D_h)**2 + np.abs(D_v)**2 + 
                  0.5 * np.abs(D_d1)**2 + 0.5 * np.abs(D_d2)**2)
    
    return D_h, D_v, D_d1, D_d2, Lambda_sum


def _update_auxiliary_z(x, p=0.8, epsilon=EPSILON):
    """
    Обновляет вспомогательные переменные z_{d,i} для MM-границы ℓp априори.
    
    MM-граница для |t|^p:
        |t|^p ≤ (p/2) z^{p/2-1} t² + (1 - p/2) z^{p/2}
    
    где z = t² (из предыдущей итерации).
    
    Это приводит к:
        z_{d,i} = (Δ^d_i(x))²
    
    Ссылка: Ур. (9)-(10) в Amizic et al. (2012)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    p : float
        Показатель ℓp квази-нормы.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    z_h, z_v, z_d1, z_d2 : ndarray (H, W)
        Вспомогательные переменные для каждого направления.
    """
    grad_h = _compute_gradient_h(x)
    grad_v = _compute_gradient_v(x)
    grad_d1 = _compute_gradient_d1(x)
    grad_d2 = _compute_gradient_d2(x)
    
    z_h = grad_h**2 + epsilon
    z_v = grad_v**2 + epsilon
    z_d1 = grad_d1**2 + epsilon
    z_d2 = grad_d2**2 + epsilon
    
    return z_h, z_v, z_d1, z_d2


def _update_auxiliary_u(h, epsilon=EPSILON):
    """
    Обновляет вспомогательные переменные u_i для MM-границы TV априори.
    
    u_i = (Δ^h h_i)² + (Δ^v h_i)²
    
    Ссылка: Ур. (12) в Amizic et al. (2012)
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Текущая оценка ядра.
    epsilon : float
        Параметр сглаживания.
    
    Возвращает
    ----------
    u : ndarray (kh, kw)
        Вспомогательные переменные.
    """
    grad_h = _compute_gradient_h(h)
    grad_v = _compute_gradient_v(h)
    u = grad_h**2 + grad_v**2 + epsilon
    return u


def _update_image_fft(Y, H_fft, z_h, z_v, z_d1, z_d2, alpha, beta, p,
                      D_h, D_v, D_d1, D_d2, image_shape):
    """
    Обновляет оценку изображения x через БПФ.
    
    Решает линейную систему:
        (β H^T H + α Q) x = β H^T y
    
    где Q — взвешенный лапласиан из MM-границы:
        Q = Σ_{d∈D} 2^{1-ω(d)} (p/2) (Δ^d)^T Z_d^{p/2-1} Δ^d
    
    В частотной области:
        X(k) = β H(k)* Y(k) / (β |H(k)|² + α Σ_d 2^{1-ω(d)} (p/2) |D_d(k)|² ψ_d(k))
    
    где ψ_d(k) — БПФ от z_d^{p/2-1}.
    
    Для простоты используем диагональное приближение со средними весами.
    
    Ссылка: Ур. (11) в Amizic et al. (2012)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    H_fft : ndarray (H, W), complex
        БПФ ядра размытия.
    z_h, z_v, z_d1, z_d2 : ndarray (H, W)
        Вспомогательные переменные для каждого направления.
    alpha : float
        Точность априори изображения.
    beta : float
        Точность шума.
    p : float
        Показатель ℓp квази-нормы.
    D_h, D_v, D_d1, D_d2 : ndarray (H, W), complex
        БПФ операторов градиента.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    X : ndarray (H, W), complex
        БПФ обновлённой оценки изображения.
    """
    H, W = image_shape
    
    # Вычисление весов из z: w_d = z_d^{p/2-1}
    exp_factor = p/2 - 1  # Отрицательный для p < 2
    w_h = z_h ** exp_factor
    w_v = z_v ** exp_factor
    w_d1 = z_d1 ** exp_factor
    w_d2 = z_d2 ** exp_factor
    
    # Диагональное приближение со средними весами
    mean_w_h = np.mean(w_h)
    mean_w_v = np.mean(w_v)
    mean_w_d1 = np.mean(w_d1)
    mean_w_d2 = np.mean(w_d2)
    
    # Член априори: Σ_d 2^{1-ω(d)} (p/2) |D_d(k)|² E[w_d]
    prior_term = (alpha * p / 2) * (
        np.abs(D_h)**2 * mean_w_h +
        np.abs(D_v)**2 * mean_w_v +
        0.5 * np.abs(D_d1)**2 * mean_w_d1 +
        0.5 * np.abs(D_d2)**2 * mean_w_d2
    )
    
    denom = beta * np.abs(H_fft)**2 + prior_term
    denom = np.maximum(denom, EPSILON)
    
    numer = beta * np.conj(H_fft) * Y
    X = numer / denom
    
    return X


def _update_blur_fft(Y, X_fft, u, gamma, beta, D_h, D_v, kernel_shape, image_shape):
    """
    Обновляет оценку размытия h через БПФ.
    
    Решает:
        (β X^T X + γ L_u) h = β X^T y
    
    где L_u — взвешенный лапласиан из MM-границы TV.
    
    В частотной области:
        H(k) = β X(k)* Y(k) / (β |X(k)|² + γ (|D_h(k)|² + |D_v(k)|²) / sqrt(ū))
    
    Ссылка: Ур. (13) в Amizic et al. (2012)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    X_fft : ndarray (H, W), complex
        БПФ текущей оценки изображения.
    u : ndarray (kh, kw)
        Вспомогательные переменные для TV.
    gamma : float
        Точность априори размытия.
    beta : float
        Точность шума.
    D_h, D_v : ndarray (H, W), complex
        БПФ операторов градиента.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    H_fft : ndarray (H, W), complex
        БПФ обновлённой оценки ядра.
    """
    mean_w = np.mean(1.0 / np.sqrt(u + EPSILON))
    
    prior_term = (gamma / 2) * (np.abs(D_h)**2 + np.abs(D_v)**2) * mean_w
    
    denom = beta * np.abs(X_fft)**2 + prior_term
    denom = np.maximum(denom, EPSILON)
    
    H_fft = beta * np.conj(X_fft) * Y / denom
    
    return H_fft


def _project_blur_constraints(H_fft, kernel_shape, image_shape):
    """
    Проецирует размытие на допустимое множество.
    
    Ограничения:
        1. Носитель: h ненулевое только внутри kernel_shape
        2. Неотрицательность: h ≥ 0
        3. Нормировка: Σ h = 1
    
    Ссылка: Раздел 3.3 в Amizic et al. (2012)
    
    Параметры
    ---------
    H_fft : ndarray (H, W), complex
        БПФ ядра.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    H_fft_proj : ndarray (H, W), complex
        БПФ спроецированного ядра.
    h : ndarray (kh, kw)
        Спроецированное ядро в пространственной области.
    """
    h_spatial = np.real(ifft2(H_fft))
    h = _extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h = np.maximum(h, 0.0)
    
    h_sum = np.sum(h)
    if h_sum > EPSILON:
        h = h / h_sum
    else:
        h = np.zeros(kernel_shape)
        h[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = _pad_kernel_for_fft(h, image_shape)
    H_fft_proj = fft2(h_padded)
    
    return H_fft_proj, h


def _estimate_alpha(x, z_h, z_v, z_d1, z_d2, p, lambda_1):
    """
    Оценивает точность априори изображения α.
    
    α = (λ₁ N / p) / (Σ_{d∈D} 2^{1-ω(d)} Σ_i z_{d,i}^{p/2})
    
    Ссылка: Ур. (16) в Amizic et al. (2012)
    
    Параметры
    ---------
    x : ndarray (H, W)
        Текущая оценка изображения.
    z_h, z_v, z_d1, z_d2 : ndarray (H, W)
        Вспомогательные переменные.
    p : float
        Показатель ℓp квази-нормы.
    lambda_1 : float
        Вес априори изображения.
    
    Возвращает
    ----------
    alpha : float
        Точность априори изображения.
    """
    H, W = x.shape
    N = H * W
    
    denom = (np.sum(z_h**(p/2)) + 
             np.sum(z_v**(p/2)) +
             0.5 * np.sum(z_d1**(p/2)) + 
             0.5 * np.sum(z_d2**(p/2)))
    
    alpha = (lambda_1 * N / p) / max(denom, EPSILON)
    
    return alpha


def _estimate_beta(y, x, h, image_shape):
    """
    Оценивает точность шума β.
    
    β = N / ||y - Hx||²
    
    Ссылка: Ур. (17) в Amizic et al. (2012)
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое изображение.
    x : ndarray (H, W)
        Текущая оценка изображения.
    h : ndarray (kh, kw)
        Текущая оценка ядра.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    beta : float
        Точность шума.
    """
    H, W = image_shape
    N = H * W
    
    h_padded = _pad_kernel_for_fft(h, image_shape)
    H_fft = fft2(h_padded)
    X_fft = fft2(x)
    Y_fft = fft2(y)
    
    residual_fft = Y_fft - H_fft * X_fft
    residual_sq = np.sum(np.abs(residual_fft)**2) / N
    
    beta = N / max(residual_sq, EPSILON)
    
    return beta


def _estimate_gamma(h, u, lambda_2):
    """
    Оценивает точность априори размытия γ.
    
    γ = λ₂ N / TV(h)
    
    где TV(h) аппроксимируется через вспомогательные переменные:
        TV(h) ≈ Σ_i √(u_i)
    
    Ссылка: Ур. (18) в Amizic et al. (2012)
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Текущая оценка ядра.
    u : ndarray (kh, kw)
        Вспомогательные переменные.
    lambda_2 : float
        Вес априори размытия.
    
    Возвращает
    ----------
    gamma : float
        Точность априори размытия.
    """
    kh, kw = h.shape
    P = kh * kw
    
    tv_approx = np.sum(np.sqrt(u + EPSILON))
    
    gamma = (lambda_2 * P) / max(tv_approx, EPSILON)
    
    return gamma


def _resize_image(img, scale, order=1):
    """Изменяет размер изображения с заданным коэффициентом масштаба."""
    return zoom(img, scale, order=order)


def _resize_kernel(h, target_shape):
    """Изменяет размер ядра до заданного размера."""
    scale_h = target_shape[0] / h.shape[0]
    scale_w = target_shape[1] / h.shape[1]
    h_resized = zoom(h, (scale_h, scale_w), order=1)
    h_resized = np.maximum(h_resized, 0)
    if np.sum(h_resized) > EPSILON:
        h_resized /= np.sum(h_resized)
    return h_resized


class SB_BID_PE(DeconvolutionAlgorithm):
    """
    Разреженная байесовская слепая деконволюция с оценкой параметров.
    
    Оценивает латентное изображение x и ядро размытия h из
    размытого и зашумленного наблюдения y используя:
        - ℓp квази-норму (0 < p < 1) для разреженных градиентов изображения
        - TV априори для ядра размытия
        - Majorization-Minimization для невыпуклой оптимизации
        - Автоматическую оценку параметров (α, β, γ)
        - Опциональную многомасштабную пирамиду
    
    MAP оценка вычисляется чередованием:
        1. Обновление вспомогательных переменных z (для изображения) и u (для размытия)
        2. Обновление изображения x при заданных z, h и параметрах
        3. Обновление размытия h при заданных u, x и параметрах
        4. Обновление параметров α, β, γ
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    p : float
        Показатель ℓp (0 < p < 1). Меньшие значения усиливают разреженность.
    lambda_1 : float
        Вес априори изображения.
    lambda_2 : float
        Вес априори размытия.
    max_iterations : int
        Максимальное число итераций на масштаб.
    tolerance : float
        Порог сходимости.
    num_scales : int
        Число масштабов в многомасштабной пирамиде.
    use_multiscale : bool
        Использовать ли многомасштабный подход.
    verbose : bool
        Выводить прогресс.
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        p: float = DEFAULT_P,
        lambda_1: float = DEFAULT_LAMBDA_1,
        lambda_2: float = DEFAULT_LAMBDA_2,
        max_iterations: int = 50,
        tolerance: float = 1e-5,
        num_scales: int = 5,
        use_multiscale: bool = True,
        use_spatial_solver: bool = False,
        max_cg_iters: int = 30,
        cg_tol: float = 1e-4,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Amizic2012.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        p : float
            Показатель ℓp (0 < p < 1). По умолчанию: 0.8
        lambda_1 : float
            Вес априори изображения. По умолчанию: 2/3
        lambda_2 : float
            Вес априори размытия. По умолчанию: 1e-3
        max_iterations : int
            Максимальное число итераций на масштаб.
        tolerance : float
            Порог сходимости.
        num_scales : int
            Число масштабов в многомасштабной пирамиде.
        use_multiscale : bool
            Использовать ли многомасштабный подход.
        use_spatial_solver : bool
            Использовать ли CG (не реализовано в этой версии).
        max_cg_iters : int
            Максимальное число CG-итераций.
        cg_tol : float
            Порог сходимости CG.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='Amizic2012')
        
        self.kernel_shape = tuple(kernel_shape)
        self.p = p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_scales = num_scales
        self.use_multiscale = use_multiscale
        self.use_spatial_solver = use_spatial_solver
        self.max_cg_iters = max_cg_iters
        self.cg_tol = cg_tol
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции с ℓp априори.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Входное размытое изображение в градациях серого.
        
        Возвращает
        ----------
        x_est : ndarray (H, W)
            Восстановленное изображение.
        h_est : ndarray (kh, kw)
            Оценённое ядро размытия (PSF).
        
        Исключения
        ----------
        ValueError
            Если p не в диапазоне (0, 1).
        """
        start_time = time.time()
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        if not (0 < self.p < 1):
            raise ValueError(f"p должен быть в диапазоне (0, 1), получено {self.p}")
        
        y = np.asarray(image, dtype=np.float64)
        H, W = y.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Многомасштабная пирамида
        if self.use_multiscale and self.num_scales > 1:
            scale_factors = [(3/2) ** (0.5 * (s - self.num_scales)) 
                             for s in range(1, self.num_scales + 1)]
        else:
            scale_factors = [1.0]
            num_scales = 1
        
        # Инициализация на самом грубом масштабе
        current_scale = scale_factors[0]
        y_scaled = _resize_image(y, current_scale, order=1)
        H_s, W_s = y_scaled.shape
        
        x = y_scaled.copy()
        
        kh_s = max(3, int(kh * current_scale))
        kw_s = max(3, int(kw * current_scale))
        h = np.zeros((kh_s, kw_s))
        h[kh_s//2, kw_s//2] = 1.0
        
        z_h, z_v, z_d1, z_d2 = _update_auxiliary_z(x, self.p)
        u = _update_auxiliary_u(h)
        
        alpha = (self.lambda_1 * N / self.p) / max(_compute_lp_prior(x, self.p), EPSILON)
        beta = N / max(np.var(y_scaled) * N, EPSILON)
        gamma = (self.lambda_2 * kh_s * kw_s) / max(_compute_tv(h), EPSILON)
        
        D_h, D_v, D_d1, D_d2, Lambda_sum = _compute_gradient_operators_fft((H_s, W_s))
        
        self.history = {
            'alpha': [],
            'beta': [],
            'gamma': [],
            'residual': []
        }
        
        # Цикл по масштабам
        for scale_idx, scale in enumerate(scale_factors):
            
            if self.verbose:
                print(f"\n--- Масштаб {scale_idx + 1}/{len(scale_factors)} (фактор: {scale:.3f}) ---")
            
            if scale_idx > 0:
                new_scale = scale / scale_factors[scale_idx - 1]
                x = _resize_image(x, new_scale, order=1)
                
                kh_s = max(3, int(kh * scale))
                kw_s = max(3, int(kw * scale))
                h = _resize_kernel(h, (kh_s, kw_s))
                
                y_scaled = _resize_image(y, scale, order=1)
                H_s, W_s = y_scaled.shape
                
                z_h, z_v, z_d1, z_d2 = _update_auxiliary_z(x, self.p)
                u = _update_auxiliary_u(h)
                
                D_h, D_v, D_d1, D_d2, Lambda_sum = _compute_gradient_operators_fft((H_s, W_s))
            
            # На самом тонком масштабе отжигаем λ₁ до 1
            if scale_idx == len(scale_factors) - 1:
                lambda_1_current = 1.0
            else:
                lambda_1_current = self.lambda_1
            
            Y_fft = fft2(y_scaled)
            
            # Цикл итераций на текущем масштабе
            for iteration in range(self.max_iterations):
                
                x_prev = x.copy()
                
                # Обновление вспомогательной z [Ур. 9-10]
                z_h, z_v, z_d1, z_d2 = _update_auxiliary_z(x, self.p)
                
                # Обновление вспомогательной u [Ур. 12]
                u = _update_auxiliary_u(h)
                
                # Обновление изображения x [Ур. 11]
                h_padded = _pad_kernel_for_fft(h, (H_s, W_s))
                H_fft = fft2(h_padded)
                
                X_fft = _update_image_fft(
                    Y_fft, H_fft, z_h, z_v, z_d1, z_d2,
                    alpha, beta, self.p, D_h, D_v, D_d1, D_d2, (H_s, W_s)
                )
                x = np.real(ifft2(X_fft))
                x = np.clip(x, 0, None)
                
                # Обновление размытия h [Ур. 13]
                H_fft_new = _update_blur_fft(
                    Y_fft, X_fft, u, gamma, beta,
                    D_h, D_v, (kh_s, kw_s), (H_s, W_s)
                )
                H_fft, h = _project_blur_constraints(H_fft_new, (kh_s, kw_s), (H_s, W_s))
                
                # Обновление параметров [Ур. 16-18]
                alpha = _estimate_alpha(x, z_h, z_v, z_d1, z_d2, self.p, lambda_1_current)
                beta = _estimate_beta(y_scaled, x, h, (H_s, W_s))
                gamma = _estimate_gamma(h, u, self.lambda_2)
                
                self.history['alpha'].append(alpha)
                self.history['beta'].append(beta)
                self.history['gamma'].append(gamma)
                
                residual = np.sqrt(np.sum((x - x_prev)**2) / max(np.sum(x_prev**2), EPSILON))
                self.history['residual'].append(residual)
                
                if self.verbose and iteration % 10 == 0:
                    print(f"  Итерация {iteration+1:3d}: α={alpha:.4e}, β={beta:.4e}, "
                          f"γ={gamma:.4e}, Δx={residual:.4e}")
                
                if residual < self.tolerance:
                    if self.verbose:
                        print(f"  Сходимость достигнута на итерации {iteration+1}")
                    break
        
        # Финальные результаты
        if scale_factors[-1] != 1.0:
            x = _resize_image(x, 1.0 / scale_factors[-1], order=1)
            x = x[:H, :W]
        
        h_est = _resize_kernel(h, self.kernel_shape)
        x_est = np.clip(x, 0, None)
        
        self.hyperparams = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'noise_std': 1.0 / np.sqrt(beta),
            'p': self.p,
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2
        }
        
        self.timer = time.time() - start_time
        
        return x_est, h_est
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """Возвращает текущие гиперпараметры алгоритма."""
        return [
            ('kernel_shape', self.kernel_shape),
            ('p', self.p),
            ('lambda_1', self.lambda_1),
            ('lambda_2', self.lambda_2),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('num_scales', self.num_scales),
            ('use_multiscale', self.use_multiscale),
            ('use_spatial_solver', self.use_spatial_solver),
            ('max_cg_iters', self.max_cg_iters),
            ('cg_tol', self.cg_tol),
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """Изменяет гиперпараметры алгоритма."""
        if 'kernel_shape' in params:
            self.kernel_shape = tuple(params['kernel_shape'])
        if 'p' in params:
            self.p = float(params['p'])
        if 'lambda_1' in params:
            self.lambda_1 = float(params['lambda_1'])
        if 'lambda_2' in params:
            self.lambda_2 = float(params['lambda_2'])
        if 'max_iterations' in params:
            self.max_iterations = int(params['max_iterations'])
        if 'tolerance' in params:
            self.tolerance = float(params['tolerance'])
        if 'num_scales' in params:
            self.num_scales = int(params['num_scales'])
        if 'use_multiscale' in params:
            self.use_multiscale = bool(params['use_multiscale'])
        if 'use_spatial_solver' in params:
            self.use_spatial_solver = bool(params['use_spatial_solver'])
        if 'max_cg_iters' in params:
            self.max_cg_iters = int(params['max_cg_iters'])
        if 'cg_tol' in params:
            self.cg_tol = float(params['cg_tol'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """Возвращает историю сходимости."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает оценённые гиперпараметры."""
        return self.hyperparams


# Обратная совместимость
def sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры.
    
    Возвращает
    ----------
    x_est, h_est, params, history
    """
    algo = Amizic2012(kernel_shape=kernel_shape, **kwargs)
    x_est, h_est = algo.process(y)
    return x_est, h_est, algo.hyperparams, algo.history


def run_algorithm(y, kernel_shape, **kwargs):
    """Обёртка для sparse_bayesian_blind_deconvolution()."""
    return sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs)
