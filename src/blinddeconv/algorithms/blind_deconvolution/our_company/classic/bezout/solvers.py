import numpy as np
from numpy.linalg import svd  # Используем numpy как вы просили
from typing import List, Tuple
from .utils import construct_bezout_matrix, build_generalized_bezout, estimate_rank, poly_div_fft

def solve_approx_univariate_gcd(p: np.ndarray, q_list: List[np.ndarray], epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 2.1: Приближенный НОД (GCD) одномерных полиномов.
    """
    n = len(p)
    
    # 1. Формируем обобщенную матрицу Безу
    B = build_generalized_bezout(p, q_list, n)
    
    # 2. Оцениваем ранг
    k = estimate_rank(B, epsilon)
    
    deg_gcd = n - k
    
    if k == 0:
        return p, np.array([1.0])

    if deg_gcd <= 0:
        return np.array([1.0]), p 

    # 3. Находим кофактор f(x). Используем SVD.
    # full_matrices=False экономит память, хотя здесь B не очень большая
    U, s_vals, Vh = svd(B, full_matrices=False)
    
    # Решение (вектор null-space) - последняя строка Vh
    # Это коэффициенты f.
    f_horner_coords = Vh[-1, :].conj()
    
    # Перевод в мономный базис через матрицу Безу(p, 1)
    one_poly = np.zeros_like(p)
    one_poly[0] = 1.0
    Bez_p_1 = construct_bezout_matrix(p, one_poly, n)
    
    f_monomial = Bez_p_1 @ f_horner_coords
    
    # 4. Вычисляем GCD = p / f
    gcd_vals = poly_div_fft(p, f_monomial)
    
    # Обрезаем
    result_len = deg_gcd + 1
    if result_len < len(gcd_vals):
        gcd_final = gcd_vals[:result_len]
    else:
        gcd_final = gcd_vals
        
    return gcd_final, f_monomial


def solve_approx_bivariate_gcd(image_stack: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Algorithm 3.1: Двумерный GCD.
    """
    m, n, s = image_stack.shape
    
    # Переход в частотную область (FFT)
    P_xt = np.fft.fft(image_stack, axis=0) # Вдоль колонок
    P_yl = np.fft.fft(image_stack, axis=1) # Вдоль строк
    
    row_gcd_vals = np.zeros((m, n), dtype=complex) # A_tl
    col_gcd_vals = np.zeros((m, n), dtype=complex) # B_tl

    # --- Step 1: Построчный GCD ---
    for t in range(m):
        coeffs_list = [P_xt[t, :, j] for j in range(s)]
        p_ref = coeffs_list[0]
        q_others = coeffs_list[1:]
        
        g_coeffs, _ = solve_approx_univariate_gcd(p_ref, q_others, epsilon)
        
        g_padded = np.zeros(n, dtype=complex)
        limit = min(len(g_coeffs), n)
        g_padded[:limit] = g_coeffs[:limit]
        
        row_gcd_vals[t, :] = np.fft.fft(g_padded)

    # --- Step 2: Постолбцовый GCD ---
    for l in range(n):
        coeffs_list = [P_yl[:, l, j] for j in range(s)]
        p_ref = coeffs_list[0]
        q_others = coeffs_list[1:]
        
        g_coeffs, _ = solve_approx_univariate_gcd(p_ref, q_others, epsilon)
        
        g_padded = np.zeros(m, dtype=complex)
        limit = min(len(g_coeffs), m)
        g_padded[:limit] = g_coeffs[:limit]
        
        col_gcd_vals[:, l] = np.fft.fft(g_padded)

    # --- Step 3: Решение системы Gamma * z = 0 ---
    # Размерность системы:
    # Уравнений: m * n (пикселей)
    # Переменных: m + n (коэффициенты a и b)
    
    # Создаем ПЛОТНУЮ матрицу для numpy.linalg.svd.
    # ВНИМАНИЕ: Размер этой матрицы (m*n) x (m+n).
    # Для 256x256: 65536 x 512.
    # Память: ~536 МБ (Complex128). Это ОК для 16 ГБ RAM.
    
    rows_count = m * n
    cols_count = m + n
    Gamma = np.zeros((rows_count, cols_count), dtype=complex)
    
    idx = 0
    for t in range(m):
        for l in range(n):
            val_A = row_gcd_vals[t, l]
            val_B = col_gcd_vals[t, l]
            
            # Gamma[idx, t] = val_A
            Gamma[idx, t] = val_A
            
            # Gamma[idx, m + l] = -val_B
            Gamma[idx, m + l] = -val_B
            
            idx += 1
            
    # --- FIX FOR MEMORY ERROR ---
    # Используем full_matrices=False.
    # Это создаст U размера (rows_count, cols_count) вместо (rows_count, rows_count).
    try:
        U, S, Vh = svd(Gamma, full_matrices=False)
        # Решение - последняя строка Vh (соответствует наименьшему сингулярному числу)
        z = Vh[-1]
    except np.linalg.LinAlgError:
        print("SVD failed to converge or other LinAlg error. Using ones.")
        z = np.ones(m + n, dtype=complex)
        
    a = z[:m]
    b = z[m:]
    
    # --- Step 4: Восстановление ---
    # p(xt, yl) = 0.5 * (A_tl * a_t + B_tl * b_l)
    
    A_scaled = row_gcd_vals * a[:, np.newaxis]
    B_scaled = col_gcd_vals * b[np.newaxis, :]
    
    P_reconstructed = 0.5 * (A_scaled + B_scaled)
    
    restored_complex = np.fft.ifft2(P_reconstructed)
    restored_image = restored_complex.real
    
    # --- Step 5: Оценка ядра ---
    F0_freq = np.fft.fft2(image_stack[:, :, 0])
    
    denom = P_reconstructed.copy()
    threshold = 1e-6 * (np.max(np.abs(denom)) + 1e-9)
    denom[np.abs(denom) < threshold] = threshold
    
    K_freq = F0_freq / denom
    kernel = np.fft.ifftshift(np.fft.ifft2(K_freq).real)
    
    # Простая нормализация
    if np.sum(kernel) != 0:
        kernel = kernel / np.sum(kernel)
    
    return restored_image, kernel