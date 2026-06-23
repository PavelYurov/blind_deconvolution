"""
Логика решателей для MHDM Blind Deconvolution.
Реализация формул из [Wolf et al., 2025] с защитой от численной нестабильности.
"""

import numpy as np

def solve_step_0(F_f: np.ndarray, 
                 lambda_0: float, 
                 mu_0: float, 
                 W_r: np.ndarray, 
                 W_s: np.ndarray) -> tuple:
    """
    Начальное приближение (u0, k0). Формулы (3.3)-(3.4).
    """
    abs_F_f = np.abs(F_f)
    
    # sgn(f)
    sgn_F_f = np.zeros_like(F_f)
    mask_nz = abs_F_f > 1e-12
    sgn_F_f[mask_nz] = F_f[mask_nz] / abs_F_f[mask_nz]
    
    # Расчет u0
    # Чтобы избежать деления на ноль и переполнения:
    ratio_u = (mu_0 * W_s) / (lambda_0 * W_r + 1e-12)
    
    # Порог: |f| - mu0 * Ws
    thresh_u = abs_F_f - (mu_0 * W_s)
    
    # u0 = sgn(f) * sqrt( ratio * [thresh]+ )
    term_u = np.sqrt(ratio_u * np.maximum(thresh_u, 0.0))
    F_u0 = sgn_F_f * term_u
    
    # Расчет k0
    ratio_k = (lambda_0 * W_r) / (mu_0 * W_s + 1e-12)
    thresh_k = abs_F_f - (lambda_0 * W_r)
    
    term_k = np.sqrt(ratio_k * np.maximum(thresh_k, 0.0))
    F_k0 = term_k.astype(np.complex128)
    
    # Принудительно задаем средние значения (DC component)
    F_k0[0, 0] = 1.0 + 0j
    F_u0[0, 0] = F_f[0, 0]
    
    return F_u0, F_k0

def solve_step_n(F_f: np.ndarray,
                 F_U_prev: np.ndarray,
                 F_K_prev: np.ndarray,
                 lambda_n: float,
                 mu_n: float,
                 W_r: np.ndarray,
                 W_s: np.ndarray) -> tuple:
    """
    Вычисляет приращения (u_inc, k_inc) решая полином 5-й степени.
    """
    # Гарантируем вещественность известных величин
    q_n = np.real(F_K_prev)  # Предыдущее ядро
    
    p_n = F_U_prev
    z = F_f
    
    # Коэффициенты регуляризации для текущего шага
    a_n = lambda_n * W_r
    b_n = mu_n * W_s
    
    # Предварительные расчеты
    a2 = a_n**2
    abs_pn_sq = np.real(p_n * np.conj(p_n))
    abs_z_sq = np.real(z * np.conj(z))
    Re_zp = np.real(z * np.conj(p_n))
    
    # Коэффициенты полинома P(Q) (Уравнение 3.16)
    C5 = b_n
    C4 = -b_n * q_n
    C3 = 2 * a_n * b_n
    C2 = a_n * Re_zp - 2 * a_n * b_n * q_n
    C1 = a2 * abs_pn_sq - a_n * abs_z_sq + a2 * b_n
    C0 = -a2 * (Re_zp + b_n * q_n)
    
    # Начальное приближение: предыдущее ядро
    Q = q_n.copy().astype(np.float64)
    
    # Метод Ньютона (векторизованный)
    for _ in range(10):
        Q2 = Q * Q
        Q3 = Q2 * Q
        Q4 = Q3 * Q
        
        P_val = C5*Q4*Q + C4*Q4 + C3*Q3 + C2*Q2 + C1*Q + C0
        P_der = 5*C5*Q4 + 4*C4*Q3 + 3*C3*Q2 + 2*C2*Q + C1
        
        mask = np.abs(P_der) > 1e-12
        delta = np.zeros_like(Q)
        delta[mask] = P_val[mask] / P_der[mask]
        delta = np.clip(delta, -100.0, 100.0)
        
        Q = Q - delta
        Q = np.maximum(Q, 0.0)

    # Приращение ядра
    F_k_inc = (Q - q_n).astype(np.complex128)
    
    # Восстановление изображения через формулу (3.12)
    denom = Q**2 + a_n
    denom[denom < 1e-12] = 1e-12
    
    U_next = (a_n * p_n + z * Q) / denom
    F_u_inc = U_next - F_U_prev
    
    # Фиксация среднего
    F_k_inc[0, 0] = 0.0j
    F_u_inc[0, 0] = 0.0j
    
    return F_u_inc, F_k_inc