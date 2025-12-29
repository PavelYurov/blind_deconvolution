"""
Вариационный подход к байесовской слепой деконволюции (алгоритм VAR3)

Литература:
    Likas, A. C., & Galatsanos, N. P. (2004).
    A variational approach for Bayesian blind image deconvolution.
    IEEE Transactions on Signal Processing, 52(8), 2222-2233.
    DOI: 10.1109/TSP.2004.831119

Реализация подхода VAR3 с использованием вариационного приближения 
к полному байесовскому апостериорному распределению:
    - Гауссова модель наблюдений: p(g|f,h,β) ∝ exp(-β/2 ||g - h*f||²)
    - SAR априори для изображения: p(f|α) ∝ exp(-α/2 ||Cf||²)  
    - Гауссово априори для размытия: p(h|γ) ∝ exp(-γ/2 ||h||²)

Подход VAR3 учитывает неопределённость в оценках изображения и ядра
(апостериорные ковариационные члены) при вариационных обновлениях.

Реализованные уравнения:
    - Ур. (16): апостериорная ковариация f
    - Ур. (17): апостериорное среднее f
    - Ур. (18): апостериорная ковариация h
    - Ур. (19): апостериорное среднее h
    - Ур. (20-22): обновление гиперпараметров
"""

import numpy as np
from numpy.fft import fft2, ifft2


EPSILON = 1e-12


def pad_kernel_for_fft(h, image_shape):
    """Дополняет ядро h до размера изображения и центрирует для БПФ."""
    H, W = image_shape
    kh, kw = h.shape
    h_padded = np.zeros((H, W), dtype=np.float64)
    h_padded[:kh, :kw] = h
    h_padded = np.roll(h_padded, shift=-kh//2, axis=0)
    h_padded = np.roll(h_padded, shift=-kw//2, axis=1)
    return h_padded


def extract_kernel_from_padded(h_padded, kernel_shape):
    """Извлекает ядро из дополненного представления."""
    kh, kw = kernel_shape
    shifted = np.roll(h_padded, shift=kh//2, axis=0)
    shifted = np.roll(shifted, shift=kw//2, axis=1)
    return shifted[:kh, :kw]


def compute_laplacian_spectrum(image_shape):
    """
    Вычисляет |C(k)|² для оператора Лапласа в частотной области.
    
    Ядро Лапласа:
        [0, -1,  0]
        [-1, 4, -1]
        [0, -1,  0]
    """
    H, W = image_shape
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float64)
    
    C_padded = np.zeros((H, W), dtype=np.float64)
    C_padded[:3, :3] = laplacian
    C_padded = np.roll(C_padded, shift=-1, axis=0)
    C_padded = np.roll(C_padded, shift=-1, axis=1)
    
    return np.abs(fft2(C_padded))**2


def update_q_f(G, M_h, S_h, alpha, beta, Lambda_C):
    """
    Обновляет q(f) = N(f | μ_f, Σ_f).
    
    Ур. (16): Σ_f(k) = [β E[|H(k)|²] + α |C(k)|²]^{-1}
    Ур. (17): M_f(k) = β Σ_f(k) M_h(k)* G(k)
    
    где E[|H(k)|²] = |M_h(k)|² + S_h(k)
    """
    E_H_sq = np.abs(M_h)**2 + S_h
    precision = beta * E_H_sq + alpha * Lambda_C + EPSILON
    S_f = 1.0 / precision
    M_f = beta * S_f * np.conj(M_h) * G
    return M_f, S_f


def update_q_h(G, M_f, S_f, gamma, beta):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Ур. (18): Σ_h(k) = [β E[|F(k)|²] + γ]^{-1}
    Ур. (19): M_h(k) = β Σ_h(k) M_f(k)* G(k)
    
    где E[|F(k)|²] = |M_f(k)|² + S_f(k)
    """
    E_F_sq = np.abs(M_f)**2 + S_f
    precision = beta * E_F_sq + gamma + EPSILON
    S_h = 1.0 / precision
    M_h = beta * S_h * np.conj(M_f) * G
    return M_h, S_h


def project_kernel_constraints(M_h, kernel_shape, image_shape):
    """Проецирует h на допустимое множество (неотрицательность, нормировка)."""
    h_spatial = np.real(ifft2(M_h))
    h_kernel = extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h_kernel = np.maximum(h_kernel, 0.0)
    h_sum = np.sum(h_kernel)
    if h_sum > EPSILON:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = pad_kernel_for_fft(h_kernel, image_shape)
    return fft2(h_padded)


def update_alpha(M_f, S_f, Lambda_C):
    """
    Обновляет точность априори изображения α.
    
    Ур. (20): α = N / E[||Cf||²]
    
    где E[||Cf||²] = (1/N) Σ_k |C(k)|² (|M_f(k)|² + S_f(k))
    """
    H, W = M_f.shape
    N = H * W
    E_Cf_sq = np.sum(Lambda_C * (np.abs(M_f)**2 + S_f)) / N
    return N / (E_Cf_sq + EPSILON)


def update_gamma(M_h, S_h, kernel_shape):
    """
    Обновляет точность априори размытия γ.
    
    Ур. (21): γ = P / E[||h||²]
    
    где P = kh × kw — размер ядра.
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw
    E_h_sq = np.sum(np.abs(M_h)**2 + S_h) / N
    return P / (E_h_sq + EPSILON)


def update_beta(G, M_f, S_f, M_h, S_h):
    """
    Обновляет точность шума β.
    
    Ур. (22): β = N / E[||g - h*f||²]
    
    где E[||g - h*f||²] = (1/N) Σ_k {|G(k) - M_h(k)M_f(k)|² 
                                    + S_f(k)|M_h(k)|² 
                                    + S_h(k)|M_f(k)|² 
                                    + S_f(k)S_h(k)}
    """
    H, W = G.shape
    N = H * W
    
    residual_sq = np.abs(G - M_h * M_f)**2
    variance_terms = (S_f * np.abs(M_h)**2 + 
                      S_h * np.abs(M_f)**2 + 
                      S_f * S_h)
    
    E_error_sq = np.sum(residual_sq + variance_terms) / N
    return N / (E_error_sq + EPSILON)


def var3_blind_deconvolution(
    g,
    kernel_shape,
    max_iterations=50,
    tolerance=1e-6,
    initial_alpha=None,
    initial_gamma=None,
    initial_beta=None,
    apply_kernel_constraints=True,
    verbose=False
):
    """
    Алгоритм VAR3 для вариационной байесовской слепой деконволюции.
    
    Совместно оценивает латентное изображение f, ядро размытия h
    и гиперпараметры (α, β, γ) методом вариационного вывода.
    
    Вариационное приближение:
        q(f, h) = q(f) q(h)
    
    где q(f) и q(h) — гауссовы распределения с моментами, вычисляемыми
    в частотной области (диагональное приближение ковариации).
    
    Параметры
    ---------
    g : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число итераций.
    tolerance : float
        Порог сходимости.
    initial_alpha, initial_gamma, initial_beta : float, optional
        Начальные значения гиперпараметров.
    apply_kernel_constraints : bool
        Проецировать ядро на допустимое множество.
    verbose : bool
        Выводить прогресс.
    
    Возвращает
    ----------
    f_est : ndarray (H, W)
        Оценка латентного изображения.
    h_est : ndarray (kh, kw)
        Оценка PSF.
    hyperparams : dict
        Финальные значения α, β, γ.
    history : dict
        История сходимости.
    """
    g = np.asarray(g, dtype=np.float64)
    H, W = g.shape
    N = H * W
    kh, kw = kernel_shape
    
    # Предвычисление спектра Лапласа
    Lambda_C = compute_laplacian_spectrum((H, W))
    
    # Инициализация изображения: f = g
    f_init = g.copy()
    M_f = fft2(f_init)
    S_f = np.zeros((H, W), dtype=np.float64)
    
    # Инициализация размытия: дельта-функция
    h_init = np.zeros((kh, kw), dtype=np.float64)
    h_init[kh//2, kw//2] = 1.0
    h_padded = pad_kernel_for_fft(h_init, (H, W))
    M_h = fft2(h_padded)
    S_h = np.zeros((H, W), dtype=np.float64)
    
    # Инициализация гиперпараметров
    noise_var = max(1e-3 * np.var(g), EPSILON)
    
    alpha = initial_alpha if initial_alpha else 1.0 / np.var(g)
    gamma = initial_gamma if initial_gamma else 1.0
    beta = initial_beta if initial_beta else 1.0 / noise_var
    
    # БПФ наблюдения
    G = fft2(g)
    
    # История
    history = {
        'alpha': [alpha],
        'beta': [beta],
        'gamma': [gamma]
    }
    
    for iteration in range(max_iterations):
        
        alpha_prev, beta_prev, gamma_prev = alpha, beta, gamma
        
        # Обновление q(f) — Ур. (16)-(17)
        M_f, S_f = update_q_f(G, M_h, S_h, alpha, beta, Lambda_C)
        
        # Обновление q(h) — Ур. (18)-(19)
        M_h, S_h = update_q_h(G, M_f, S_f, gamma, beta)
        
        # Проекция на допустимое множество
        if apply_kernel_constraints:
            M_h = project_kernel_constraints(M_h, kernel_shape, (H, W))
        
        # Обновление гиперпараметров — Ур. (20)-(22)
        alpha = update_alpha(M_f, S_f, Lambda_C)
        gamma = update_gamma(M_h, S_h, kernel_shape)
        beta = update_beta(G, M_f, S_f, M_h, S_h)
        
        history['alpha'].append(alpha)
        history['beta'].append(beta)
        history['gamma'].append(gamma)
        
        # Проверка сходимости
        delta_alpha = abs(alpha - alpha_prev) / max(alpha_prev, EPSILON)
        delta_beta = abs(beta - beta_prev) / max(beta_prev, EPSILON)
        delta_gamma = abs(gamma - gamma_prev) / max(gamma_prev, EPSILON)
        max_delta = max(delta_alpha, delta_beta, delta_gamma)
        
        if verbose:
            print(f"Iter {iteration+1:3d}: α={alpha:.4e}, β={beta:.4e}, "
                  f"γ={gamma:.4e}, Δ={max_delta:.4e}")
        
        if max_delta < tolerance:
            if verbose:
                print(f"Сходимость на итерации {iteration+1}")
            break
    
    # Извлечение результатов
    f_est = np.real(ifft2(M_f))
    
    h_spatial = np.real(ifft2(M_h))
    h_est = extract_kernel_from_padded(h_spatial, kernel_shape)
    h_est = np.maximum(h_est, 0.0)
    h_sum = np.sum(h_est)
    if h_sum > EPSILON:
        h_est = h_est / h_sum
    
    hyperparams = {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma
    }
    
    return f_est, h_est, hyperparams, history


def run_algorithm(g, kernel_shape, **kwargs):
    """Обёртка для var3_blind_deconvolution()."""
    return var3_blind_deconvolution(g, kernel_shape, **kwargs)
