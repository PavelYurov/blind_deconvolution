"""
solvers.py
Реализация подзадач алгоритма AEE-BD (Algorithm 3.1).
"""

import numpy as np
from scipy.signal import convolve2d
from .utils import psf2otf, soft_threshold, compute_curvature, gaussian_kernel

def compute_adaptive_matrix_T(f: np.ndarray, iota: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет матрицу T по формуле (4).
    T = 1 / (1 + iota * | G_delta * grad f |)
    """
    # 1. Градиенты f
    grad_x_f = np.roll(f, -1, axis=1) - f
    grad_y_f = np.roll(f, -1, axis=0) - f
    
    # 2. Ядро Гаусса
    k_size = int(np.ceil(delta * 4)) * 2 + 1
    gauss = gaussian_kernel(k_size, delta)
    
    # 3. Свертка градиентов с Гауссом (Boundary conditions from paper are usually wrap/periodic for FFT methods, 
    # but for T calculation 'same' is apt. Using 'wrap' to match periodic model).
    # ВНИМАНИЕ: Сначала свертка, потом модуль (как на скриншоте 4).
    conv_x = convolve2d(grad_x_f, gauss, mode='same', boundary='wrap')
    conv_y = convolve2d(grad_y_f, gauss, mode='same', boundary='wrap')
    
    # 4. Формула (4)
    t1 = 1.0 / (1.0 + iota * np.abs(conv_x))
    t2 = 1.0 / (1.0 + iota * np.abs(conv_y))
    
    return t1, t2

def solve_k_subproblem(u: np.ndarray, f: np.ndarray, w: np.ndarray, 
                       lambda3: np.ndarray, r3: float, lambda_val: float, 
                       dx_otf: np.ndarray, dy_otf: np.ndarray) -> np.ndarray:
    """
    Eq (12): k update.
    """
    H, W = f.shape
    
    F_f = np.fft.fft2(f)
    F_u = np.fft.fft2(u)
    
    # w - lambda3
    term_x = w[0] - lambda3[0]
    term_y = w[1] - lambda3[1]
    
    F_term_x = np.fft.fft2(term_x)
    F_term_y = np.fft.fft2(term_y)
    
    # Adjoint grad in freq domain: conj(dx)*Fx + conj(dy)*Fy
    div_term = np.conj(dx_otf) * F_term_x + np.conj(dy_otf) * F_term_y
    
    # Числитель: lambda * conj(F_u) * F_f + r3 * div(...)
    numerator = lambda_val * np.conj(F_u) * F_f + r3 * div_term
    
    # Знаменатель: lambda * |F_u|^2 + r3 * (|dx|^2 + |dy|^2)
    laplacian_mag = np.abs(dx_otf)**2 + np.abs(dy_otf)**2
    denominator = lambda_val * np.abs(F_u)**2 + r3 * laplacian_mag
    
    F_k = numerator / (denominator + 1e-12)
    k_new = np.real(np.fft.ifft2(F_k))
    
    return k_new

def solve_u_subproblem(k: np.ndarray, f: np.ndarray, p: np.ndarray,
                       lambda1: np.ndarray, r1: float, lambda_val: float,
                       dx_otf: np.ndarray, dy_otf: np.ndarray) -> np.ndarray:
    """
    Eq (14): u update.
    """
    H, W = f.shape
    
    F_f = np.fft.fft2(f)
    # k дополняется нулями до размера (H, W) внутри psf2otf
    F_k = psf2otf(k, (H, W)) 
    
    # p - lambda1
    term_x = p[0] - lambda1[0]
    term_y = p[1] - lambda1[1]
    
    F_term_x = np.fft.fft2(term_x)
    F_term_y = np.fft.fft2(term_y)
    
    div_term = np.conj(dx_otf) * F_term_x + np.conj(dy_otf) * F_term_y
    
    numerator = lambda_val * np.conj(F_k) * F_f + r1 * div_term
    
    laplacian_mag = np.abs(dx_otf)**2 + np.abs(dy_otf)**2
    denominator = lambda_val * np.abs(F_k)**2 + r1 * laplacian_mag
    
    F_u = numerator / (denominator + 1e-12)
    u_new = np.real(np.fft.ifft2(F_u))
    
    return u_new

def solve_p_subproblem(u: np.ndarray, q: np.ndarray, 
                       lambda1: np.ndarray, lambda2: np.ndarray,
                       t1: np.ndarray, t2: np.ndarray,
                       r1: float, r2: float) -> np.ndarray:
    """
    Eq (17): p update.
    """
    # Градиенты u
    grad_x_u = np.roll(u, -1, axis=1) - u
    grad_y_u = np.roll(u, -1, axis=0) - u
    
    # p1
    num_p1 = r1 * (grad_x_u + lambda1[0]) + r2 * t1 * (q[0] - lambda2[0])
    den_p1 = r1 + r2 * (t1**2)
    p1 = num_p1 / den_p1
    
    # p2
    num_p2 = r1 * (grad_y_u + lambda1[1]) + r2 * t2 * (q[1] - lambda2[1])
    den_p2 = r1 + r2 * (t2**2)
    p2 = num_p2 / den_p2
    
    return np.stack([p1, p2])

def solve_q_subproblem(p: np.ndarray, u_prev: np.ndarray, lambda2: np.ndarray,
                       t1: np.ndarray, t2: np.ndarray,
                       alpha: float, r2: float) -> np.ndarray:
    """
    Eq (19): q update.
    """
    term_x = t1 * p[0] + lambda2[0]
    term_y = t2 * p[1] + lambda2[1]
    
    # Кривизна от предыдущего u (u^n) - так в Eq (9) u_i,j^n используется в g(...)
    kappa = compute_curvature(u_prev)
    
    # Eq (6): g(k) = 1 + alpha * |k|
    g_kappa = 1.0 + alpha * np.abs(kappa)
    
    threshold = g_kappa / r2
    
    q1 = soft_threshold(term_x, threshold)
    q2 = soft_threshold(term_y, threshold)
    
    return np.stack([q1, q2])

def solve_w_subproblem(k: np.ndarray, lambda3: np.ndarray,
                       beta: float, r3: float) -> np.ndarray:
    """
    Eq (21): w update.
    """
    grad_x_k = np.roll(k, -1, axis=1) - k
    grad_y_k = np.roll(k, -1, axis=0) - k
    
    term_x = grad_x_k + lambda3[0]
    term_y = grad_y_k + lambda3[1]
    
    threshold = beta / r3
    
    w1 = soft_threshold(term_x, threshold)
    w2 = soft_threshold(term_y, threshold)
    
    return np.stack([w1, w2])