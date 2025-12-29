"""
Вариационный байесовский алгоритм слепой деконволюции

Литература:
    Molina, R., Mateos, J., & Katsaggelos, A. K. (2006).
    Blind Deconvolution Using a Variational Approach to Parameter,
    Image, and Blur Estimation.
    IEEE Transactions on Image Processing, 15(12), 3715-3727.
    DOI: 10.1109/TIP.2006.881972

Иерархическая байесовская модель:
    - Гауссова модель наблюдений: p(g|f,h,β) ∝ β^(N/2) exp(-β/2 ||g - Hf||²)
    - SAR априори изображения: p(f|α) ∝ α^(N/2) exp(-α/2 ||Cf||²)
    - SAR априори размытия: p(h|γ) ∝ γ^(P/2) exp(-γ/2 ||Dh||²)
    - Гамма гиперприори: p(α), p(β), p(γ)

Вариационное приближение факторизуется как:
    q(f, h, Ω) = q(f) q(h) q(α) q(β) q(γ)

где q(f) и q(h) — гауссовы, а q(α), q(β), q(γ) — гамма-распределения.
"""

import numpy as np
from numpy.fft import fft2, ifft2


# Параметры гиперприори для гамма-распределений (плоские/неинформативные)
# Для Gamma(x | a, b): p(x) ∝ x^(a-1) exp(-b*x), Mean = a/b
DEFAULT_HYPERPRIORS = {
    'a_alpha': 1e-3,
    'b_alpha': 1e-3,
    'a_beta': 1e-3,
    'b_beta': 1e-3,
    'a_gamma': 1e-3,
    'b_gamma': 1e-3,
}


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


def compute_laplacian_operator_fft(image_shape):
    """
    Вычисляет |C(k)|² в частотной области для оператора Лапласа C.
    
    Ядро Лапласа:
        [0, -1,  0]
        [-1, 4, -1]
        [0, -1,  0]
    
    Соответствует SAR априори ||Cf||² = f^T C^T C f.
    См. Ур. (6) в Molina et al. (2006)
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
    Lambda_C = np.abs(C_fft) ** 2
    
    return Lambda_C


def update_q_f(G, M_h, S_h, E_alpha, E_beta, Lambda_C):
    """
    Обновляет q(f) = N(f | μ_f, Σ_f).
    
    Уравнения (22)-(23):
        Σ_f = (⟨β⟩ ⟨H^T H⟩ + ⟨α⟩ C^T C)^{-1}
        μ_f = ⟨β⟩ Σ_f ⟨H⟩^T g
    
    В частотной области:
        S_f(k) = 1 / (⟨β⟩ E[|H(k)|²] + ⟨α⟩ |C(k)|²)
        M_f(k) = ⟨β⟩ S_f(k) M_h(k)^* G(k)
    
    где E[|H(k)|²] = |M_h(k)|² + S_h(k) (Ур. 24)
    """
    # E[|H(k)|²] = |M_h(k)|² + S_h(k) — Ур. (24)
    E_H_squared = np.abs(M_h) ** 2 + S_h
    
    # Σ_f^{-1}(k) = β E[|H(k)|²] + α |C(k)|² — Ур. (22)
    precision_f = E_beta * E_H_squared + E_alpha * Lambda_C
    precision_f = np.maximum(precision_f, 1e-12)
    
    S_f = 1.0 / precision_f
    
    # M_f(k) = β S_f(k) M_h(k)^* G(k) — Ур. (23)
    M_f = E_beta * S_f * np.conj(M_h) * G
    
    return M_f, S_f


def update_q_h(G, M_f, S_f, E_gamma, E_beta, Lambda_D):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Уравнения (26)-(27):
        Σ_h = (⟨β⟩ ⟨F^T F⟩ + ⟨γ⟩ D^T D)^{-1}
        μ_h = ⟨β⟩ Σ_h ⟨F⟩^T g
    
    В частотной области:
        S_h(k) = 1 / (⟨β⟩ E[|F(k)|²] + ⟨γ⟩ |D(k)|²)
        M_h(k) = ⟨β⟩ S_h(k) M_f(k)^* G(k)
    """
    # E[|F(k)|²] = |M_f(k)|² + S_f(k)
    E_F_squared = np.abs(M_f) ** 2 + S_f
    
    # Σ_h^{-1}(k) = β E[|F(k)|²] + γ |D(k)|² — Ур. (26)
    precision_h = E_beta * E_F_squared + E_gamma * Lambda_D
    precision_h = np.maximum(precision_h, 1e-12)
    
    S_h = 1.0 / precision_h
    
    # M_h(k) = β S_h(k) M_f(k)^* G(k) — Ур. (27)
    M_h = E_beta * S_h * np.conj(M_f) * G
    
    return M_h, S_h


def project_kernel_constraints(M_h, S_h, kernel_shape, image_shape):
    """
    Проецирует оценку размытия на допустимое множество:
        1. Ограничение носителя (h ненулевое только на kernel_shape)
        2. Неотрицательность (h >= 0)
        3. Нормировка (sum(h) = 1)
    """
    h_spatial = np.real(ifft2(M_h))
    h_kernel = extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h_kernel = np.maximum(h_kernel, 0.0)
    
    h_sum = np.sum(h_kernel)
    if h_sum > 1e-12:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = pad_kernel_for_fft(h_kernel, image_shape)
    M_h_proj = fft2(h_padded)
    
    return M_h_proj


def update_q_alpha(M_f, S_f, Lambda_C, a_alpha, b_alpha):
    """
    Обновляет q(α), которое является гамма-распределением.
    
    Из Ур. (29):
        ⟨α⟩ = (N/2 + a_α - 1) / (E[||Cf||²]/2 + b_α)
    
    где:
        E[||Cf||²] = (1/N) Σ_k |C(k)|² (|M_f(k)|² + S_f(k))  [по Парсевалю]
    """
    H, W = M_f.shape
    N = H * W
    
    E_Cf_squared = np.sum(Lambda_C * (np.abs(M_f) ** 2 + S_f)) / N
    
    numerator = N / 2.0 + a_alpha - 1.0
    denominator = E_Cf_squared / 2.0 + b_alpha
    
    E_alpha = numerator / denominator
    E_alpha = max(E_alpha, 1e-12)
    
    return E_alpha


def update_q_beta(G, M_f, S_f, M_h, S_h, a_beta, b_beta):
    """
    Обновляет q(β), которое является гамма-распределением.
    
    Из Ур. (30):
        ⟨β⟩ = (N/2 + a_β - 1) / (E[||g - Hf||²]/2 + b_β)
    
    где (из Ур. 25):
        E[||g - Hf||²] = (1/N) Σ_k { |G(k) - M_h(k) M_f(k)|²
                                    + S_f(k) |M_h(k)|²
                                    + S_h(k) |M_f(k)|²
                                    + S_f(k) S_h(k) }
    """
    H, W = G.shape
    N = H * W
    
    residual_squared = np.abs(G - M_h * M_f) ** 2
    
    variance_terms = (S_f * np.abs(M_h) ** 2 + 
                      S_h * np.abs(M_f) ** 2 + 
                      S_f * S_h)
    
    E_error_squared = np.sum(residual_squared + variance_terms) / N
    
    numerator = N / 2.0 + a_beta - 1.0
    denominator = E_error_squared / 2.0 + b_beta
    
    E_beta = numerator / denominator
    E_beta = max(E_beta, 1e-12)
    
    return E_beta


def update_q_gamma(M_h, S_h, Lambda_D, a_gamma, b_gamma, kernel_shape):
    """
    Обновляет q(γ), которое является гамма-распределением.
    
    Из Ур. (31):
        ⟨γ⟩ = (P/2 + a_γ - 1) / (E[||Dh||²]/2 + b_γ)
    
    где P — размерность h (число пикселей ядра).
    
    Для D = I (единичная матрица, белый гауссов приор на h):
        E[||Dh||²] = E[||h||²] = μ_h^T μ_h + Tr(Σ_h)
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw
    
    E_Dh_squared = np.sum(Lambda_D * (np.abs(M_h) ** 2 + S_h)) / N
    
    numerator = P / 2.0 + a_gamma - 1.0
    denominator = E_Dh_squared / 2.0 + b_gamma
    
    E_gamma = numerator / denominator
    E_gamma = max(E_gamma, 1e-12)
    
    return E_gamma


def variational_blind_deconvolution(
    g,
    kernel_shape,
    max_iterations=50,
    tolerance=1e-6,
    hyperpriors=None,
    initial_alpha=None,
    initial_beta=None,
    initial_gamma=None,
    apply_kernel_constraints=True,
    verbose=False
):
    """
    Вариационная байесовская слепая деконволюция — Алгоритм 1 из Molina et al. (2006).
    
    Совместно оценивает латентное изображение f, ядро размытия h
    и гиперпараметры (α, β, γ) методом вариационного вывода.
    
    Вариационное приближение:
        q(f, h, α, β, γ) = q(f) q(h) q(α) q(β) q(γ)
    
    где q(f) и q(h) — гауссовы, а q(α), q(β), q(γ) — гамма-распределения.
    
    Параметры
    ---------
    g : ndarray (H, W)
        Наблюдаемое размытое и зашумленное изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости (относительное изменение гиперпараметров).
    hyperpriors : dict, optional
        Параметры гамма гиперприори.
    initial_alpha, initial_beta, initial_gamma : float, optional
        Начальные значения мат. ожиданий гиперпараметров.
    apply_kernel_constraints : bool
        Проецировать h на допустимое множество.
    verbose : bool
        Выводить прогресс.
    
    Возвращает
    ----------
    f_est : ndarray (H, W)
        Оценка латентного изображения.
    h_est : ndarray (kh, kw)
        Оценка PSF.
    hyperparams : dict
        Финальные мат. ожидания α, β, γ.
    history : dict
        История сходимости.
    """
    g = np.asarray(g, dtype=np.float64)
    H, W = g.shape
    N = H * W
    kh, kw = kernel_shape
    
    # Параметры гиперприори
    if hyperpriors is None:
        hyperpriors = DEFAULT_HYPERPRIORS
    a_alpha = hyperpriors.get('a_alpha', 1e-3)
    b_alpha = hyperpriors.get('b_alpha', 1e-3)
    a_beta = hyperpriors.get('a_beta', 1e-3)
    b_beta = hyperpriors.get('b_beta', 1e-3)
    a_gamma = hyperpriors.get('a_gamma', 1e-3)
    b_gamma = hyperpriors.get('b_gamma', 1e-3)
    
    # Предвычисление операторов
    Lambda_C = compute_laplacian_operator_fft((H, W))
    Lambda_D = np.ones((H, W), dtype=np.float64)
    
    # Инициализация изображения: μ_f = g
    f_init = g.copy()
    M_f = fft2(f_init)
    S_f = np.zeros((H, W), dtype=np.float64)
    
    # Инициализация размытия: h = дельта-функция
    h_init = np.zeros((kh, kw), dtype=np.float64)
    h_init[kh // 2, kw // 2] = 1.0
    h_padded = pad_kernel_for_fft(h_init, (H, W))
    M_h = fft2(h_padded)
    S_h = np.zeros((H, W), dtype=np.float64)
    
    # Инициализация гиперпараметров
    noise_var_estimate = 1e-3 * np.var(g)
    
    if initial_beta is None:
        E_beta = 1.0 / max(noise_var_estimate, 1e-6)
    else:
        E_beta = initial_beta
        
    if initial_alpha is None:
        E_alpha = 1.0 / max(np.var(g), 1e-6)
    else:
        E_alpha = initial_alpha
        
    if initial_gamma is None:
        E_gamma = 1.0
    else:
        E_gamma = initial_gamma
    
    # Наблюдение в частотной области
    G = fft2(g)
    
    # История
    history = {
        'alpha': [E_alpha],
        'beta': [E_beta],
        'gamma': [E_gamma]
    }
    
    for iteration in range(max_iterations):
        
        E_alpha_prev = E_alpha
        E_beta_prev = E_beta
        E_gamma_prev = E_gamma
        
        # Обновление q(f) [Ур. 22-23]
        M_f, S_f = update_q_f(G, M_h, S_h, E_alpha, E_beta, Lambda_C)
        
        # Обновление q(h) [Ур. 26-27]
        M_h, S_h = update_q_h(G, M_f, S_f, E_gamma, E_beta, Lambda_D)
        
        # Проекция на допустимое множество
        if apply_kernel_constraints:
            M_h = project_kernel_constraints(M_h, S_h, kernel_shape, (H, W))
        
        # Обновление q(α) [Ур. 29]
        E_alpha = update_q_alpha(M_f, S_f, Lambda_C, a_alpha, b_alpha)
        
        # Обновление q(β) [Ур. 30]
        E_beta = update_q_beta(G, M_f, S_f, M_h, S_h, a_beta, b_beta)
        
        # Обновление q(γ) [Ур. 31]
        E_gamma = update_q_gamma(M_h, S_h, Lambda_D, a_gamma, b_gamma, kernel_shape)
        
        history['alpha'].append(E_alpha)
        history['beta'].append(E_beta)
        history['gamma'].append(E_gamma)
        
        # Проверка сходимости
        delta_alpha = abs(E_alpha - E_alpha_prev) / max(E_alpha_prev, 1e-12)
        delta_beta = abs(E_beta - E_beta_prev) / max(E_beta_prev, 1e-12)
        delta_gamma = abs(E_gamma - E_gamma_prev) / max(E_gamma_prev, 1e-12)
        max_delta = max(delta_alpha, delta_beta, delta_gamma)
        
        if verbose:
            print(f"Iter {iteration+1:3d}: α={E_alpha:.4e}, β={E_beta:.4e}, "
                  f"γ={E_gamma:.4e}, Δ={max_delta:.4e}")
        
        if max_delta < tolerance:
            if verbose:
                print(f"Сходимость на итерации {iteration+1}")
            break
    
    # Извлечение финальных оценок
    f_est = np.real(ifft2(M_f))
    
    h_spatial = np.real(ifft2(M_h))
    h_est = extract_kernel_from_padded(h_spatial, kernel_shape)
    h_est = np.maximum(h_est, 0.0)
    h_sum = np.sum(h_est)
    if h_sum > 1e-12:
        h_est = h_est / h_sum
    
    hyperparams = {
        'alpha': E_alpha,
        'beta': E_beta,
        'gamma': E_gamma
    }
    
    return f_est, h_est, hyperparams, history


def run_algorithm(g, kernel_shape, **kwargs):
    """Обёртка для variational_blind_deconvolution()."""
    return variational_blind_deconvolution(g, kernel_shape, **kwargs)
