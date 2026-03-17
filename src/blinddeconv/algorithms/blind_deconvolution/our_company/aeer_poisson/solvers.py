"""
solvers.py
Реализация подзадач алгоритма AEE-BD для POISSON шума.
Адаптировано под формулы (9)-(21) из статьи "Blind Restoration of Poisson Images...".
"""

import numpy as np
from scipy.signal import convolve2d
from .utils import psf2otf, soft_threshold, compute_curvature, gaussian_kernel

def compute_adaptive_matrix_T(f: np.ndarray, iota: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    grad_x_f = np.roll(f, -1, axis=1) - f
    grad_y_f = np.roll(f, -1, axis=0) - f
    k_size = int(np.ceil(delta * 4)) * 2 + 1
    gauss = gaussian_kernel(k_size, delta)
    conv_x = convolve2d(grad_x_f, gauss, mode='same', boundary='wrap')
    conv_y = convolve2d(grad_y_f, gauss, mode='same', boundary='wrap')
    t1 = 1.0 / (1.0 + iota * np.abs(conv_x))
    t2 = 1.0 / (1.0 + iota * np.abs(conv_y))
    return t1, t2

def solve_k_subproblem(u: np.ndarray, g: np.ndarray, lambda4: np.ndarray, 
                       w: np.ndarray, lambda3: np.ndarray, 
                       r3: float, r4: float, 
                       dx_otf: np.ndarray, dy_otf: np.ndarray) -> np.ndarray:
    """
    Eq (9), (10): k update for Poisson model.
    Использует g и lambda4 (r4) вместо исходного изображения f и lambda.
    """
    H, W = g.shape
    
    # F(g - lambda4)
    F_g_l4 = np.fft.fft2(g - lambda4)
    F_u = np.fft.fft2(u)
    
    # div(w - lambda3) в частотной области: conj(dx)*Fx + conj(dy)*Fy
    term_x = w[0] - lambda3[0]
    term_y = w[1] - lambda3[1]
    F_term_x = np.fft.fft2(term_x)
    F_term_y = np.fft.fft2(term_y)
    div_term = np.conj(dx_otf) * F_term_x + np.conj(dy_otf) * F_term_y
    
    numerator = r4 * np.conj(F_u) * F_g_l4 + r3 * div_term
    laplacian_mag = np.abs(dx_otf)**2 + np.abs(dy_otf)**2
    denominator = r4 * np.abs(F_u)**2 + r3 * laplacian_mag
    
    F_k = numerator / (denominator + 1e-12)
    return np.real(np.fft.ifft2(F_k))

def solve_u_subproblem(k: np.ndarray, g: np.ndarray, lambda4: np.ndarray, 
                       p: np.ndarray, lambda1: np.ndarray, 
                       r1: float, r4: float, 
                       dx_otf: np.ndarray, dy_otf: np.ndarray) -> np.ndarray:
    """
    Eq (11), (12): u update for Poisson model.
    Использует g и lambda4.
    """
    H, W = g.shape
    
    F_g_l4 = np.fft.fft2(g - lambda4)
    F_k = psf2otf(k, (H, W)) 
    
    term_x = p[0] - lambda1[0]
    term_y = p[1] - lambda1[1]
    F_term_x = np.fft.fft2(term_x)
    F_term_y = np.fft.fft2(term_y)
    div_term = np.conj(dx_otf) * F_term_x + np.conj(dy_otf) * F_term_y
    
    numerator = r4 * np.conj(F_k) * F_g_l4 + r1 * div_term
    laplacian_mag = np.abs(dx_otf)**2 + np.abs(dy_otf)**2
    denominator = r4 * np.abs(F_k)**2 + r1 * laplacian_mag
    
    F_u = numerator / (denominator + 1e-12)
    return np.real(np.fft.ifft2(F_u))

def solve_p_subproblem(u: np.ndarray, q: np.ndarray, 
                       lambda1: np.ndarray, lambda2: np.ndarray,
                       t1: np.ndarray, t2: np.ndarray,
                       r1: float, r2: float) -> np.ndarray:
    """Eq (13)-(15): p update (без изменений)"""
    grad_x_u = np.roll(u, -1, axis=1) - u
    grad_y_u = np.roll(u, -1, axis=0) - u
    num_p1 = r1 * (grad_x_u + lambda1[0]) + r2 * t1 * (q[0] - lambda2[0])
    den_p1 = r1 + r2 * (t1**2)
    p1 = num_p1 / den_p1
    num_p2 = r1 * (grad_y_u + lambda1[1]) + r2 * t2 * (q[1] - lambda2[1])
    den_p2 = r1 + r2 * (t2**2)
    p2 = num_p2 / den_p2
    return np.stack([p1, p2])

def solve_q_subproblem(p: np.ndarray, u_prev: np.ndarray, lambda2: np.ndarray,
                       t1: np.ndarray, t2: np.ndarray,
                       alpha: float, r2: float) -> np.ndarray:
    """Eq (16), (17): q update (без изменений)"""
    term_x = t1 * p[0] + lambda2[0]
    term_y = t2 * p[1] + lambda2[1]
    kappa = compute_curvature(u_prev)
    g_kappa = 1.0 + alpha * np.abs(kappa)
    threshold = g_kappa / r2
    q1 = soft_threshold(term_x, threshold)
    q2 = soft_threshold(term_y, threshold)
    return np.stack([q1, q2])

def solve_w_subproblem(k: np.ndarray, lambda3: np.ndarray,
                       beta: float, r3: float) -> np.ndarray:
    """Eq (18), (19): w update (без изменений)"""
    grad_x_k = np.roll(k, -1, axis=1) - k
    grad_y_k = np.roll(k, -1, axis=0) - k
    term_x = grad_x_k + lambda3[0]
    term_y = grad_y_k + lambda3[1]
    threshold = beta / r3
    w1 = soft_threshold(term_x, threshold)
    w2 = soft_threshold(term_y, threshold)
    return np.stack([w1, w2])

def solve_g_subproblem(k: np.ndarray, u: np.ndarray, f: np.ndarray, 
                       lambda4: np.ndarray, lambda_val: float, r4: float) -> np.ndarray:
    """
    Eq (20), (21): НОВАЯ ПОДЗАДАЧА ДЛЯ g (Poisson Fidelity).
    """
    H, W = f.shape
    F_k = psf2otf(k, (H, W))
    F_u = np.fft.fft2(u)
    
    # Ku = K * u (Свертка)
    Ku = np.real(np.fft.ifft2(F_k * F_u))
    
    # Вспомогательная переменная из Eq (21)
    b = Ku + lambda4 - lambda_val / r4
    
    # Решение квадратного уравнения для Пуассона.
    # Чтобы избежать NaN при f=0 и отрицательном b, берем np.maximum(f, 0)
    # g_new = b / 2.0 + np.sqrt((b / 2.0)**2 + (lambda_val * np.maximum(f, 0)) / r4)
    g_new = b / 2.0 + np.sqrt((b / 2.0)**2 + (lambda_val * f) / r4)
    
    return g_new