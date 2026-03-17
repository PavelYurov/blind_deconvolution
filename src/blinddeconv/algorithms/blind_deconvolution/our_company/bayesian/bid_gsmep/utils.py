import numpy as np
from scipy.fft import fft2, ifft2

def get_gsm_foe_params(num_filters=25, filter_size=5):
    """
    Возвращает параметры GSM FoE (Fields of Experts), упомянутые в Разделе II.
    
    В статье сказано: "filters... are learned from natural images" [24].
    Поскольку точные численные значения весов и параметров смеси (pi, sigma) 
    в тексте статьи не даны, здесь генерируется аппроксимация на основе DCT-базиса,
    что часто используется в качестве инициализации для FoE.
    
    Returns:
        filters (np.ndarray): (K, size, size) - банк фильтров w_k
        variances (np.ndarray): (J,) - дисперсии гауссиан смеси sigma_j
        weights (np.ndarray): (J,) - веса гауссиан смеси pi_j
    """
    # Генерация DCT фильтров как аппроксимация обученных фильтров
    filters = np.zeros((num_filters, filter_size, filter_size))
    idx = 0
    for u in range(filter_size):
        for v in range(filter_size):
            if idx >= num_filters: break
            # Формула базисных функций DCT
            basis = np.fromfunction(
                lambda x, y: np.cos((2*x+1)*u*np.pi/(2*filter_size)) * 
                             np.cos((2*y+1)*v*np.pi/(2*filter_size)),
                (filter_size, filter_size)
            )
            # Нормализация (zero mean, unit norm)
            basis = basis - np.mean(basis)
            norm = np.linalg.norm(basis)
            if norm > 0:
                basis /= norm
            filters[idx] = basis
            idx += 1
            
    # Параметры GSM (имитация типичных параметров для FoE из [24])
    # J - количество шкал. Обычно J=2..4.
    variances = np.array([0.5, 1.5, 5.0]) # sigma_j
    weights = np.array([0.8, 0.15, 0.05]) # pi_j, сумма должна быть 1
    
    return filters, variances, weights

def gsm_log_pdf(x, variances, weights):
    """
    Вычисляет ln(psi(x)) согласно Eq. (5) и Eq. (4).
    psi(x) = sum_j (pi_j / sigma_j) * exp(-x^2 / (2*sigma_j^2))
    
    Args:
        x: значение отклика фильтра
    """
    # Eq. (5): psi(x) proportional to sum ...
    # Мы вычисляем логарифм этой суммы.
    # Используем log-sum-exp trick для стабильности
    
    exponents = -0.5 * (x[..., None] ** 2) / (variances ** 2)
    coeffs = weights / variances
    
    # argument of log
    args = coeffs * np.exp(exponents)
    psi = np.sum(args, axis=-1)
    
    return np.log(psi + 1e-10) # +epsilon для защиты от log(0)

def gsm_neg_log_derivative(x, variances, weights):
    """
    Вычисляет производную -ln(psi(x)) по x.
    Нужно для оптимизации подзадачи 1 (Eq. 14).
    d/dx (-ln(psi)) = - (psi') / psi
    """
    # Численности:
    # A_j = (pi_j / sigma_j) * exp(-x^2 / 2sigma^2)
    # psi = sum A_j
    # psi' = sum A_j * (-x / sigma_j^2)
    # result = - psi' / psi = x * (sum A_j/sigma_j^2) / (sum A_j)
    
    x_sq = x[..., None]**2
    inv_var_sq = 1.0 / (variances**2)
    
    exponents = np.exp(-0.5 * x_sq * inv_var_sq)
    aj = (weights / variances) * exponents
    
    numerator = np.sum(aj * inv_var_sq, axis=-1)
    denominator = np.sum(aj, axis=-1) + 1e-12
    
    return x * (numerator / denominator)

def psf2otf(psf, shape):
    """
    Преобразует PSF в OTF (Optical Transfer Function) с учетом центрирования.
    Аналог MATLAB psf2otf.
    """
    in_shape = psf.shape
    # Pad to shape
    padded = np.zeros(shape)
    padded[:in_shape[0], :in_shape[1]] = psf
    
    # Circular shift to center the PSF at (0,0) for FFT
    for axis, axis_size in enumerate(in_shape):
        padded = np.roll(padded, -int(axis_size / 2), axis=axis)
        
    return fft2(padded)

def gradient_filters():
    """Возвращает фильтры производных d1 (hor), d2 (ver)."""
    # Простейшие фильтры [-1, 1], как обычно используется в статьях
    d1 = np.array([[1, -1]])      # горизонтальный (по x)
    d2 = np.array([[1], [-1]])    # вертикальный (по y)
    return d1, d2

def threshold_gradients(grad, c):
    """
    Пороговая функция T(d_o) из Eq. (16).
    T_i(d, o) = 0 if |(d,o)_i| <= c, else (d,o)_i
    """
    mask = np.abs(grad) > c
    return grad * mask

def threshold_psf(h, c_h):
    """
    Пороговая функция T2(h) из Eq. (19).
    T2(h) = 0 if h_m <= c_h, else h_m
    """
    h_out = h.copy()
    h_out[h_out <= c_h] = 0
    # Normalize for energy preservation (text below Eq. 19)
    s = np.sum(h_out)
    if s > 0:
        h_out /= s
    return h_out

def compute_grads(img):
    """Вычисляет градиенты изображения (циклическая свертка)."""
    h, w = img.shape
    img_f = fft2(img)
    
    d1, d2 = gradient_filters()
    d1_f = psf2otf(d1, (h, w))
    d2_f = psf2otf(d2, (h, w))
    
    g1 = np.real(ifft2(img_f * d1_f))
    g2 = np.real(ifft2(img_f * d2_f))
    return g1, g2