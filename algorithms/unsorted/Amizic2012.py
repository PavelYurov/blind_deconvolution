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

# Параметры по умолчанию
DEFAULT_PARAMS = {
    'p': 0.8,           # Показатель ℓp квази-нормы (0 < p < 1)
    'lambda_1': 2/3,    # Вес априори изображения (отжиг начинается с 2/3)
    'lambda_2': 1e-3,   # Вес априори размытия
    'max_iterations': 50,
    'tolerance': 1e-5,
    'num_scales': 5,    # Число масштабов в пирамиде
}

EPSILON = 1e-10


def compute_gradient_h(x):
    """
    Вычисляет горизонтальный градиент Δ^h x методом прямых разностей.
    
    (Δ^h x)_{i,j} = x_{i,j+1} - x_{i,j}
    
    См. Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    """
    grad_h = np.zeros_like(x)
    grad_h[:, :-1] = x[:, 1:] - x[:, :-1]
    grad_h[:, -1] = x[:, 0] - x[:, -1]  # Периодические границы
    return grad_h


def compute_gradient_v(x):
    """
    Вычисляет вертикальный градиент Δ^v x методом прямых разностей.
    
    (Δ^v x)_{i,j} = x_{i+1,j} - x_{i,j}
    
    См. Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    """
    grad_v = np.zeros_like(x)
    grad_v[:-1, :] = x[1:, :] - x[:-1, :]
    grad_v[-1, :] = x[0, :] - x[-1, :]  # Периодические границы
    return grad_v


def compute_gradient_d1(x):
    """
    Вычисляет первый диагональный градиент Δ^{d1} x.
    
    (Δ^{d1} x)_{i,j} = x_{i+1,j+1} - x_{i,j}
    
    См. Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    """
    grad_d1 = np.zeros_like(x)
    grad_d1[:-1, :-1] = x[1:, 1:] - x[:-1, :-1]
    grad_d1[-1, :-1] = x[0, 1:] - x[-1, :-1]
    grad_d1[:-1, -1] = x[1:, 0] - x[:-1, -1]
    grad_d1[-1, -1] = x[0, 0] - x[-1, -1]
    return grad_d1


def compute_gradient_d2(x):
    """
    Вычисляет второй диагональный градиент Δ^{d2} x.
    
    (Δ^{d2} x)_{i,j} = x_{i+1,j-1} - x_{i,j}
    
    См. Раздел 2.1, Ур. (5) в Amizic et al. (2012)
    """
    grad_d2 = np.zeros_like(x)
    grad_d2[:-1, 1:] = x[1:, :-1] - x[:-1, 1:]
    grad_d2[-1, 1:] = x[0, :-1] - x[-1, 1:]
    grad_d2[:-1, 0] = x[1:, -1] - x[:-1, 0]
    grad_d2[-1, 0] = x[0, -1] - x[-1, 0]
    return grad_d2


def compute_divergence_h(p):
    """Вычисляет сопряжённый (транспонированный) оператор горизонтального градиента."""
    div_h = np.zeros_like(p)
    div_h[:, 1:] -= p[:, :-1]
    div_h[:, :-1] += p[:, :-1]
    div_h[:, 0] -= p[:, -1]
    div_h[:, -1] += p[:, -1]
    return div_h


def compute_divergence_v(p):
    """Вычисляет сопряжённый (транспонированный) оператор вертикального градиента."""
    div_v = np.zeros_like(p)
    div_v[1:, :] -= p[:-1, :]
    div_v[:-1, :] += p[:-1, :]
    div_v[0, :] -= p[-1, :]
    div_v[-1, :] += p[-1, :]
    return div_v


def compute_divergence_d1(p):
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


def compute_divergence_d2(p):
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


def compute_lp_prior(x, p=0.8):
    """
    Вычисляет ℓp квази-норму априори для изображения.
    
    Φ(x) = Σ_{d∈D} 2^{1-ω(d)} Σ_i |Δ^d_i(x)|^p
    
    где D = {h, v, d1, d2} — направления градиента и
    ω(d) = 1 для горизонтального/вертикального, ω(d) = 2 для диагоналей.
    
    См. Ур. (5) в Amizic et al. (2012)
    """
    grad_h = compute_gradient_h(x)
    grad_v = compute_gradient_v(x)
    grad_d1 = compute_gradient_d1(x)
    grad_d2 = compute_gradient_d2(x)
    
    # Веса: 2^{1-ω(d)}
    # ω(h) = ω(v) = 1 → вес = 2^0 = 1
    # ω(d1) = ω(d2) = 2 → вес = 2^{-1} = 0.5
    phi = (np.sum(np.abs(grad_h)**p) + 
           np.sum(np.abs(grad_v)**p) +
           0.5 * np.sum(np.abs(grad_d1)**p) + 
           0.5 * np.sum(np.abs(grad_d2)**p))
    
    return phi


def compute_tv(h, epsilon=EPSILON):
    """
    Вычисляет полную вариацию ядра размытия h.
    
    TV(h) = Σ_i √((Δ^h h_i)² + (Δ^v h_i)² + ε)
    
    См. Ур. (6) в Amizic et al. (2012)
    """
    grad_h = compute_gradient_h(h)
    grad_v = compute_gradient_v(h)
    return np.sum(np.sqrt(grad_h**2 + grad_v**2 + epsilon))


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


def compute_gradient_operators_fft(image_shape):
    """
    Вычисляет БПФ операторов градиента для эффективных вычислений.
    
    Возвращает D_h, D_v, D_d1, D_d2 в частотной области.
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


def update_auxiliary_z(x, p=0.8, epsilon=EPSILON):
    """
    Обновляет вспомогательные переменные z_{d,i} для MM-границы ℓp априори.
    
    MM-граница для |t|^p:
        |t|^p ≤ (p/2) z^{p/2-1} t² + (1 - p/2) z^{p/2}
    
    где z = t² (из предыдущей итерации).
    
    Это приводит к:
        z_{d,i} = (Δ^d_i(x))²
    
    См. Ур. (9)-(10) в Amizic et al. (2012)
    """
    grad_h = compute_gradient_h(x)
    grad_v = compute_gradient_v(x)
    grad_d1 = compute_gradient_d1(x)
    grad_d2 = compute_gradient_d2(x)
    
    z_h = grad_h**2 + epsilon
    z_v = grad_v**2 + epsilon
    z_d1 = grad_d1**2 + epsilon
    z_d2 = grad_d2**2 + epsilon
    
    return z_h, z_v, z_d1, z_d2


def update_auxiliary_u(h, epsilon=EPSILON):
    """
    Обновляет вспомогательные переменные u_i для MM-границы TV априори.
    
    u_i = (Δ^h h_i)² + (Δ^v h_i)²
    
    См. Ур. (12) в Amizic et al. (2012)
    """
    grad_h = compute_gradient_h(h)
    grad_v = compute_gradient_v(h)
    u = grad_h**2 + grad_v**2 + epsilon
    return u


def update_image_fft(Y, H_fft, z_h, z_v, z_d1, z_d2, alpha, beta, p,
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
    См. Ур. (11) в Amizic et al. (2012)
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


def update_image_spatial_cg(y, h, z_h, z_v, z_d1, z_d2, alpha, beta, p,
                            image_shape, x_init=None, max_iters=30, tol=1e-4):
    """
    Обновляет изображение x методом сопряжённых градиентов с точными весами.
    
    Решает: (β H^T H + α Q) x = β H^T y
    
    где Q — взвешенный оператор регуляризации.
    См. Раздел 3.2 в Amizic et al. (2012)
    """
    Ht, Wt = image_shape
    
    h_padded = pad_kernel_for_fft(h, image_shape)
    H_fft = fft2(h_padded)
    
    # Вычисление весов: w_d = z_d^{p/2-1}
    exp_factor = p/2 - 1
    w_h = z_h ** exp_factor
    w_v = z_v ** exp_factor
    w_d1 = z_d1 ** exp_factor
    w_d2 = z_d2 ** exp_factor
    
    def apply_H(x):
        return np.real(ifft2(H_fft * fft2(x)))
    
    def apply_HT(x):
        return np.real(ifft2(np.conj(H_fft) * fft2(x)))
    
    def apply_Q(x):
        # Q = Σ_d 2^{1-ω(d)} (p/2) (Δ^d)^T W_d Δ^d
        result = np.zeros_like(x)
        
        grad_h = compute_gradient_h(x)
        result += compute_divergence_h(w_h * grad_h)
        
        grad_v = compute_gradient_v(x)
        result += compute_divergence_v(w_v * grad_v)
        
        grad_d1 = compute_gradient_d1(x)
        result += 0.5 * compute_divergence_d1(w_d1 * grad_d1)
        
        grad_d2 = compute_gradient_d2(x)
        result += 0.5 * compute_divergence_d2(w_d2 * grad_d2)
        
        return -(alpha * p / 2) * result
    
    def apply_A(x):
        return beta * apply_HT(apply_H(x)) + apply_Q(x)
    
    b = beta * apply_HT(y)
    
    if x_init is None:
        x = y.copy()
    else:
        x = x_init.copy()
    
    # Метод сопряжённых градиентов
    r = b - apply_A(x)
    p_cg = r.copy()
    rs_old = np.sum(r * r)
    
    for iteration in range(max_iters):
        Ap = apply_A(p_cg)
        pAp = np.sum(p_cg * Ap)
        
        if pAp < EPSILON:
            break
            
        alpha_cg = rs_old / pAp
        x = x + alpha_cg * p_cg
        r = r - alpha_cg * Ap
        rs_new = np.sum(r * r)
        
        if np.sqrt(rs_new) < tol * np.sqrt(np.sum(b * b)):
            break
            
        p_cg = r + (rs_new / rs_old) * p_cg
        rs_old = rs_new
    
    return x


def update_blur_fft(Y, X_fft, u, gamma, beta, D_h, D_v, kernel_shape, image_shape):
    """
    Обновляет оценку размытия h через БПФ.
    
    Решает:
        (β X^T X + γ L_u) h = β X^T y
    
    где L_u — взвешенный лапласиан из MM-границы TV.
    
    В частотной области:
        H(k) = β X(k)* Y(k) / (β |X(k)|² + γ (|D_h(k)|² + |D_v(k)|²) / sqrt(ū))
    
    См. Ур. (13) в Amizic et al. (2012)
    """
    mean_w = np.mean(1.0 / np.sqrt(u + EPSILON))
    
    prior_term = (gamma / 2) * (np.abs(D_h)**2 + np.abs(D_v)**2) * mean_w
    
    denom = beta * np.abs(X_fft)**2 + prior_term
    denom = np.maximum(denom, EPSILON)
    
    H_fft = beta * np.conj(X_fft) * Y / denom
    
    return H_fft


def project_blur_constraints(H_fft, kernel_shape, image_shape):
    """
    Проецирует размытие на допустимое множество.
    
    Ограничения:
        1. Носитель: h ненулевое только внутри kernel_shape
        2. Неотрицательность: h ≥ 0
        3. Нормировка: Σ h = 1
    
    См. Раздел 3.3 в Amizic et al. (2012)
    """
    h_spatial = np.real(ifft2(H_fft))
    h = extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h = np.maximum(h, 0.0)
    
    h_sum = np.sum(h)
    if h_sum > EPSILON:
        h = h / h_sum
    else:
        h = np.zeros(kernel_shape)
        h[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = pad_kernel_for_fft(h, image_shape)
    H_fft_proj = fft2(h_padded)
    
    return H_fft_proj, h


def estimate_alpha(x, z_h, z_v, z_d1, z_d2, p, lambda_1):
    """
    Оценивает точность априори изображения α.
    
    α = (λ₁ N / p) / (Σ_{d∈D} 2^{1-ω(d)} Σ_i z_{d,i}^{p/2})
    
    См. Ур. (16) в Amizic et al. (2012)
    """
    H, W = x.shape
    N = H * W
    
    denom = (np.sum(z_h**(p/2)) + 
             np.sum(z_v**(p/2)) +
             0.5 * np.sum(z_d1**(p/2)) + 
             0.5 * np.sum(z_d2**(p/2)))
    
    alpha = (lambda_1 * N / p) / max(denom, EPSILON)
    
    return alpha


def estimate_beta(y, x, h, image_shape):
    """
    Оценивает точность шума β.
    
    β = N / ||y - Hx||²
    
    См. Ур. (17) в Amizic et al. (2012)
    """
    H, W = image_shape
    N = H * W
    
    h_padded = pad_kernel_for_fft(h, image_shape)
    H_fft = fft2(h_padded)
    X_fft = fft2(x)
    Y_fft = fft2(y)
    
    residual_fft = Y_fft - H_fft * X_fft
    residual_sq = np.sum(np.abs(residual_fft)**2) / N
    
    beta = N / max(residual_sq, EPSILON)
    
    return beta


def estimate_gamma(h, u, lambda_2):
    """
    Оценивает точность априори размытия γ.
    
    γ = λ₂ N / TV(h)
    
    где TV(h) аппроксимируется через вспомогательные переменные:
        TV(h) ≈ Σ_i √(u_i)
    
    См. Ур. (18) в Amizic et al. (2012)
    """
    kh, kw = h.shape
    P = kh * kw
    
    tv_approx = np.sum(np.sqrt(u + EPSILON))
    
    gamma = (lambda_2 * P) / max(tv_approx, EPSILON)
    
    return gamma


def resize_image(img, scale, order=1):
    """Изменяет размер изображения с заданным коэффициентом масштаба."""
    return zoom(img, scale, order=order)


def resize_kernel(h, target_shape):
    """Изменяет размер ядра до заданного размера."""
    scale_h = target_shape[0] / h.shape[0]
    scale_w = target_shape[1] / h.shape[1]
    h_resized = zoom(h, (scale_h, scale_w), order=1)
    h_resized = np.maximum(h_resized, 0)
    if np.sum(h_resized) > EPSILON:
        h_resized /= np.sum(h_resized)
    return h_resized


def sparse_bayesian_blind_deconvolution(
    y,
    kernel_shape,
    p=0.8,
    lambda_1=2/3,
    lambda_2=1e-3,
    max_iterations=50,
    tolerance=1e-5,
    num_scales=5,
    use_multiscale=True,
    use_spatial_solver=False,
    max_cg_iters=30,
    cg_tol=1e-4,
    verbose=False
):
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
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое размытое и зашумленное изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    p : float
        Показатель ℓp (0 < p < 1). Меньшие значения усиливают разреженность.
        По умолчанию: 0.8
    lambda_1 : float
        Вес априори изображения. По умолчанию: 2/3 (отжиг до 1)
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
        Использовать CG (точнее, но медленнее).
    max_cg_iters : int
        Максимальное число CG-итераций.
    cg_tol : float
        Порог сходимости CG.
    verbose : bool
        Выводить прогресс.
    
    Возвращает
    ----------
    x_est : ndarray (H, W)
        Оценка латентного изображения.
    h_est : ndarray (kh, kw)
        Оценка PSF.
    params : dict
        Финальные оценённые параметры (α, β, γ).
    history : dict
        История сходимости.
    """
    y = np.asarray(y, dtype=np.float64)
    H, W = y.shape
    N = H * W
    kh, kw = kernel_shape
    
    if not (0 < p < 1):
        raise ValueError(f"p должен быть в (0, 1), получено {p}")
    
    # Многомасштабная пирамида
    if use_multiscale and num_scales > 1:
        scale_factors = [(3/2) ** (0.5 * (s - num_scales)) 
                         for s in range(1, num_scales + 1)]
    else:
        scale_factors = [1.0]
        num_scales = 1
    
    # Инициализация на самом грубом масштабе
    current_scale = scale_factors[0]
    y_scaled = resize_image(y, current_scale, order=1)
    H_s, W_s = y_scaled.shape
    
    x = y_scaled.copy()
    
    kh_s = max(3, int(kh * current_scale))
    kw_s = max(3, int(kw * current_scale))
    h = np.zeros((kh_s, kw_s))
    h[kh_s//2, kw_s//2] = 1.0
    
    z_h, z_v, z_d1, z_d2 = update_auxiliary_z(x, p)
    u = update_auxiliary_u(h)
    
    alpha = (lambda_1 * N / p) / max(compute_lp_prior(x, p), EPSILON)
    beta = N / max(np.var(y_scaled) * N, EPSILON)
    gamma = (lambda_2 * kh_s * kw_s) / max(compute_tv(h), EPSILON)
    
    D_h, D_v, D_d1, D_d2, Lambda_sum = compute_gradient_operators_fft((H_s, W_s))
    
    history = {
        'alpha': [],
        'beta': [],
        'gamma': [],
        'residual': []
    }
    
    # Цикл по масштабам
    for scale_idx, scale in enumerate(scale_factors):
        
        if verbose:
            print(f"\n--- Масштаб {scale_idx + 1}/{num_scales} (фактор: {scale:.3f}) ---")
        
        if scale_idx > 0:
            new_scale = scale / scale_factors[scale_idx - 1]
            x = resize_image(x, new_scale, order=1)
            
            kh_s = max(3, int(kh * scale))
            kw_s = max(3, int(kw * scale))
            h = resize_kernel(h, (kh_s, kw_s))
            
            y_scaled = resize_image(y, scale, order=1)
            H_s, W_s = y_scaled.shape
            
            z_h, z_v, z_d1, z_d2 = update_auxiliary_z(x, p)
            u = update_auxiliary_u(h)
            
            D_h, D_v, D_d1, D_d2, Lambda_sum = compute_gradient_operators_fft((H_s, W_s))
        
        # На самом тонком масштабе отжигаем λ₁ до 1
        if scale_idx == num_scales - 1:
            lambda_1_current = 1.0
        else:
            lambda_1_current = lambda_1
        
        Y_fft = fft2(y_scaled)
        
        # Цикл итераций на текущем масштабе
        for iteration in range(max_iterations):
            
            x_prev = x.copy()
            h_prev = h.copy()
            alpha_prev = alpha
            beta_prev = beta
            gamma_prev = gamma
            
            # Обновление вспомогательной z [Ур. 9-10]
            z_h, z_v, z_d1, z_d2 = update_auxiliary_z(x, p)
            
            # Обновление вспомогательной u [Ур. 12]
            u = update_auxiliary_u(h)
            
            # Обновление изображения x [Ур. 11]
            h_padded = pad_kernel_for_fft(h, (H_s, W_s))
            H_fft = fft2(h_padded)
            
            if use_spatial_solver:
                x = update_image_spatial_cg(
                    y_scaled, h, z_h, z_v, z_d1, z_d2,
                    alpha, beta, p, (H_s, W_s), x_init=x,
                    max_iters=max_cg_iters, tol=cg_tol
                )
                X_fft = fft2(x)
            else:
                X_fft = update_image_fft(
                    Y_fft, H_fft, z_h, z_v, z_d1, z_d2,
                    alpha, beta, p, D_h, D_v, D_d1, D_d2, (H_s, W_s)
                )
                x = np.real(ifft2(X_fft))
                x = np.clip(x, 0, None)
            
            # Обновление размытия h [Ур. 13]
            H_fft_new = update_blur_fft(
                Y_fft, X_fft, u, gamma, beta,
                D_h, D_v, (kh_s, kw_s), (H_s, W_s)
            )
            H_fft, h = project_blur_constraints(H_fft_new, (kh_s, kw_s), (H_s, W_s))
            
            # Обновление параметров [Ур. 16-18]
            alpha = estimate_alpha(x, z_h, z_v, z_d1, z_d2, p, lambda_1_current)
            beta = estimate_beta(y_scaled, x, h, (H_s, W_s))
            gamma = estimate_gamma(h, u, lambda_2)
            
            history['alpha'].append(alpha)
            history['beta'].append(beta)
            history['gamma'].append(gamma)
            
            residual = np.sqrt(np.sum((x - x_prev)**2) / max(np.sum(x_prev**2), EPSILON))
            history['residual'].append(residual)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration+1:3d}: α={alpha:.4e}, β={beta:.4e}, "
                      f"γ={gamma:.4e}, Δx={residual:.4e}")
            
            if residual < tolerance:
                if verbose:
                    print(f"  Сходимость на итерации {iteration+1}")
                break
    
    # Финальные результаты
    if scale_factors[-1] != 1.0:
        x = resize_image(x, 1.0 / scale_factors[-1], order=1)
        x = x[:H, :W]
    
    h_est = resize_kernel(h, kernel_shape)
    x_est = np.clip(x, 0, None)
    
    params = {
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'noise_std': 1.0 / np.sqrt(beta),
        'p': p,
        'lambda_1': lambda_1,
        'lambda_2': lambda_2
    }
    
    return x_est, h_est, params, history


def run_algorithm(y, kernel_shape, **kwargs):
    """Обёртка для sparse_bayesian_blind_deconvolution()."""
    return sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs)
