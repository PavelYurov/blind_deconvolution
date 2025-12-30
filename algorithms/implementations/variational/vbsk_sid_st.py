"""
Вариационная байесовская слепая деконволюция с разреженным ядром 
и распределениями Стьюдента

Литература:
    Tzikas, D. G., Likas, A. C., & Galatsanos, N. P. (2009).
    Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution 
    With Student's-t Priors.
    IEEE Transactions on Image Processing, 18(4), 753-764.
    DOI: 10.1109/TIP.2008.2011757

Алгоритм использует:
    - Разреженную kernel-based модель PSF: h = Kw с разреженными весами w
    - Априори Стьюдента для весов w (продвигает разреженность через ARD)
    - Априори Стьюдента для разностей изображения (сохраняет грани)
    - Модель шума Стьюдента (устойчивость к выбросам)
    - Вариационный байесовский вывод

Ключевые особенности:
    - Автоматическая оценка носителя PSF через разреженность
    - Восстановление изображения с сохранением граней
    - Устойчивость к выбросам шума
    - Распределение Стьюдента представлено как гауссова scale-mixture
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


# Степени свободы для распределений Стьюдента (малые значения = тяжёлые хвосты)
DEFAULT_NU_W = 1e-4      # Для априори весов (очень разреженное)
DEFAULT_NU_F = 1e-4      # Для априори изображения (сохранение граней)
DEFAULT_NU_NOISE = 1e-4  # Для модели шума (устойчивость)

# Параметры базисных ядер
DEFAULT_KERNEL_SIGMA = 1.0

EPSILON = 1e-10


def _create_gaussian_kernel_basis(kernel_shape, sigma=DEFAULT_KERNEL_SIGMA):
    """
    Создаёт базис гауссовых ядер K для представления PSF.
    
    PSF моделируется как: h = Kw
    где K — матрица смещённых гауссовых ядер, w — веса.
    
    Ссылка: Раздел II-A, Ур. (3) в Tzikas et al. (2009)
    
    Параметры
    ---------
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    sigma : float
        Стандартное отклонение гауссова базиса.
    
    Возвращает
    ----------
    G : ndarray (kh, kw)
        Базисное гауссово ядро.
    """
    kh, kw = kernel_shape
    
    y = np.arange(kh) - kh // 2
    x = np.arange(kw) - kw // 2
    Y, X = np.meshgrid(y, x, indexing='ij')
    
    G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    G = G / np.sum(G)
    
    return G


def _apply_kernel_basis(w, kernel_shape, sigma=DEFAULT_KERNEL_SIGMA):
    """
    Вычисляет h = Kw, где K — базис гауссовых ядер.
    
    Для разреженной модели ядра h — сглаженная версия весов w.
    
    Параметры
    ---------
    w : ndarray (kh, kw)
        Веса PSF.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    sigma : float
        Стандартное отклонение гауссова базиса.
    
    Возвращает
    ----------
    h : ndarray (kh, kw)
        PSF = Kw.
    """
    if sigma > 0:
        h = gaussian_filter(w, sigma=sigma, mode='constant')
    else:
        h = w.copy()
    
    return h


def _student_t_scale_parameter(x, nu):
    """
    Вычисляет scale-параметр для Стьюдента как гауссовой scale-mixture.
    
    Распределение Стьюдента:
        Student's-t(x | 0, 1, ν) = ∫ N(x | 0, τ^{-1}) Gamma(τ | ν/2, ν/2) dτ
    
    Апостериорное τ при заданном x:
        q(τ) = Gamma(τ | (ν+1)/2, (ν + x²)/2)
    
    Мат. ожидание:
        E[τ] = (ν + 1) / (ν + x²)
    
    Ссылка: Ур. (6)-(7) в Tzikas et al. (2009)
    
    Параметры
    ---------
    x : ndarray
        Значения переменной.
    nu : float
        Степени свободы распределения Стьюдента.
    
    Возвращает
    ----------
    E_tau : ndarray
        Мат. ожидание параметра масштаба τ.
    """
    E_tau = (nu + 1.0) / (nu + x**2 + EPSILON)
    return E_tau


def _compute_student_t_weights_image(f, nu_f):
    """
    Вычисляет веса точности для априори изображения на основе локальных разностей.
    
    Априори изображения использует Стьюдента для разностей:
        p(f_i - f_j) ∝ Student's-t(f_i - f_j | 0, 1, ν_f)
    
    Это продвигает кусочно-гладкие изображения с резкими гранями.
    
    Ссылка: Раздел II-C в Tzikas et al. (2009)
    
    Параметры
    ---------
    f : ndarray (H, W)
        Текущая оценка изображения.
    nu_f : float
        Степени свободы для априори изображения.
    
    Возвращает
    ----------
    alpha_h : ndarray (H, W)
        Веса точности для горизонтальных разностей.
    alpha_v : ndarray (H, W)
        Веса точности для вертикальных разностей.
    """
    H, W = f.shape
    
    # Горизонтальные разности
    diff_h = np.zeros((H, W))
    diff_h[:, :-1] = f[:, 1:] - f[:, :-1]
    
    # Вертикальные разности
    diff_v = np.zeros((H, W))
    diff_v[:-1, :] = f[1:, :] - f[:-1, :]
    
    # Веса точности из scale-mixture Стьюдента
    alpha_h = _student_t_scale_parameter(diff_h, nu_f)
    alpha_v = _student_t_scale_parameter(diff_v, nu_f)
    
    return alpha_h, alpha_v


def _compute_student_t_weights_psf(w, nu_w):
    """
    Вычисляет веса точности для априори весов PSF.
    
    Априори весов использует Стьюдента:
        p(w_i) ∝ Student's-t(w_i | 0, 1, ν_w)
    
    Это продвигает разреженное представление PSF (ARD).
    
    Ссылка: Раздел II-B, Ур. (5) в Tzikas et al. (2009)
    
    Параметры
    ---------
    w : ndarray (kh, kw)
        Веса PSF.
    nu_w : float
        Степени свободы для априори весов.
    
    Возвращает
    ----------
    gamma : ndarray (kh, kw)
        Веса точности для весов PSF.
    """
    gamma = _student_t_scale_parameter(w, nu_w)
    return gamma


def _compute_student_t_weights_noise(residual, nu_noise):
    """
    Вычисляет веса точности для устойчивой модели шума.
    
    Модель шума использует Стьюдента:
        p(n_i) ∝ Student's-t(n_i | 0, σ², ν_noise)
    
    Это обеспечивает устойчивость к выбросам.
    
    Ссылка: Раздел II-D в Tzikas et al. (2009)
    
    Параметры
    ---------
    residual : ndarray (H, W)
        Невязка y - Hx.
    nu_noise : float
        Степени свободы для модели шума.
    
    Возвращает
    ----------
    beta : ndarray (H, W)
        Веса точности для шума.
    """
    beta = _student_t_scale_parameter(residual, nu_noise)
    return beta


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
    
    return D_h, D_v


def _update_image(Y, H_fft, beta_weights, alpha_h, alpha_v, D_h, D_v, image_shape):
    """
    Обновляет q(f) = N(f | μ_f, Σ_f).
    
    С априори Стьюдента, представленным как scale-mixture, апостериорное
    распределение изображения становится гауссовым с точностью:
        Σ_f^{-1} = H^T diag(β) H + D_h^T diag(α_h) D_h + D_v^T diag(α_v) D_v
    
    В частотной области с диагональным приближением:
        S_f(k) ≈ 1 / (β̄ |H(k)|² + ᾱ_h |D_h(k)|² + ᾱ_v |D_v(k)|²)
    
    где β̄, ᾱ — средние точности.
    
    Ссылка: Ур. (11)-(13) в Tzikas et al. (2009)
    
    Параметры
    ---------
    Y : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    H_fft : ndarray (H, W), complex
        БПФ ядра размытия.
    beta_weights : ndarray (H, W)
        Веса точности шума.
    alpha_h : ndarray (H, W)
        Веса точности горизонтальных разностей.
    alpha_v : ndarray (H, W)
        Веса точности вертикальных разностей.
    D_h : ndarray (H, W), complex
        БПФ горизонтального оператора градиента.
    D_v : ndarray (H, W), complex
        БПФ вертикального оператора градиента.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    M_f : ndarray (H, W), complex
        Апостериорное среднее f в частотной области.
    S_f : ndarray (H, W)
        Апостериорная дисперсия f в частотной области.
    """
    H, W = image_shape
    
    # Средние точности (диагональное приближение)
    beta_mean = np.mean(beta_weights)
    alpha_h_mean = np.mean(alpha_h)
    alpha_v_mean = np.mean(alpha_v)
    
    # Апостериорная точность в частотной области
    precision = (beta_mean * np.abs(H_fft)**2 + 
                 alpha_h_mean * np.abs(D_h)**2 + 
                 alpha_v_mean * np.abs(D_v)**2)
    precision = np.maximum(precision, EPSILON)
    
    S_f = 1.0 / precision
    M_f = beta_mean * S_f * np.conj(H_fft) * Y
    
    return M_f, S_f


def _update_psf_weights(y, f, K_sigma, kernel_shape, gamma_weights, beta_mean, image_shape):
    """
    Обновляет q(w) = N(w | μ_w, Σ_w).
    
    Апостериорное распределение весов PSF с априори Стьюдента:
        Σ_w^{-1} = β (KX)^T (KX) + diag(γ)
        μ_w = β Σ_w (KX)^T y
    
    где X — матрица свёртки, построенная из f.
    
    Для эффективности используем решение в пространственной области.
    
    Ссылка: Ур. (14)-(16) в Tzikas et al. (2009)
    
    Параметры
    ---------
    y : ndarray (H, W)
        Наблюдаемое изображение.
    f : ndarray (H, W)
        Текущая оценка изображения.
    K_sigma : float
        Стандартное отклонение гауссова базиса.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    gamma_weights : ndarray (kh, kw)
        Веса точности для весов PSF.
    beta_mean : float
        Средняя точность шума.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    w : ndarray (kh, kw)
        Обновлённые веса PSF.
    """
    H, W = image_shape
    kh, kw = kernel_shape
    
    F_fft = fft2(f)
    Y_fft = fft2(y)
    
    # Корреляция: X^T y в частотной области
    XTy = np.real(ifft2(np.conj(F_fft) * Y_fft))
    XTy_crop = _extract_kernel_from_padded(XTy, kernel_shape)
    
    # Применение сглаживания ядром: K^T X^T y
    if K_sigma > 0:
        KTXTy = gaussian_filter(XTy_crop, sigma=K_sigma, mode='constant')
    else:
        KTXTy = XTy_crop
    
    # X^T X (автокорреляция f)
    XTX = np.real(ifft2(np.abs(F_fft)**2))
    XTX_center = XTX[0, 0]  # Диагональное приближение
    
    # Диагональное приближение для Σ_w
    precision_w = beta_mean * XTX_center + gamma_weights
    precision_w = np.maximum(precision_w, EPSILON)
    
    w = beta_mean * KTXTy / precision_w
    
    return w


def _update_psf_from_weights(w, kernel_shape, K_sigma):
    """
    Вычисляет PSF h из весов w через базис ядер.
    
    h = Kw (сглаженная версия весов)
    
    Параметры
    ---------
    w : ndarray (kh, kw)
        Веса PSF.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    K_sigma : float
        Стандартное отклонение гауссова базиса.
    
    Возвращает
    ----------
    h : ndarray (kh, kw)
        PSF с применёнными ограничениями.
    """
    h = _apply_kernel_basis(w, kernel_shape, sigma=K_sigma)
    
    # Применение ограничений
    h = np.maximum(h, 0.0)  # Неотрицательность
    h_sum = np.sum(h)
    if h_sum > EPSILON:
        h = h / h_sum  # Нормировка
    else:
        h = np.zeros(kernel_shape)
        h[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    return h


def _update_noise_precision(residual, beta_weights, a_beta=1e-3, b_beta=1e-3):
    """
    Обновляет мат. ожидание точности шума.
    
    С моделью шума Стьюдента:
        E[β] = (N/2 + a_β) / (E[Σ β_i r_i²]/2 + b_β)
    
    Ссылка: Раздел III в Tzikas et al. (2009)
    
    Параметры
    ---------
    residual : ndarray (H, W)
        Невязка y - Hx.
    beta_weights : ndarray (H, W)
        Веса точности шума.
    a_beta : float
        Параметр формы гамма-гиперприори.
    b_beta : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_beta : float
        Мат. ожидание точности шума.
    """
    N = residual.size
    weighted_error = np.sum(beta_weights * residual**2)
    
    numerator = N / 2.0 + a_beta
    denominator = weighted_error / 2.0 + b_beta
    
    E_beta = numerator / max(denominator, EPSILON)
    return E_beta


class VBSK_SID_ST(DeconvolutionAlgorithm):
    """
    Вариационная байесовская слепая деконволюция с разреженным ядром
    и распределениями Стьюдента.
    
    Совместно оценивает:
        - Латентное изображение f
        - PSF h (через разреженные веса ядра w)
    
    используя вариационный вывод с априори Стьюдента для:
        - Весов PSF (разреженность)
        - Разностей изображения (сохранение граней)
        - Шума (устойчивость)
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости.
    nu_w : float
        Степени свободы для априори весов (меньше = разреженнее).
    nu_f : float
        Степени свободы для априори изображения (меньше = резче грани).
    nu_noise : float
        Степени свободы для шума (меньше = устойчивее).
    kernel_sigma : float
        Ширина базиса гауссовых ядер.
    verbose : bool
        Выводить прогресс.
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        nu_w: float = DEFAULT_NU_W,
        nu_f: float = DEFAULT_NU_F,
        nu_noise: float = DEFAULT_NU_NOISE,
        kernel_sigma: float = DEFAULT_KERNEL_SIGMA,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Tzikas2009.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число VB-итераций.
        tolerance : float
            Порог сходимости.
        nu_w : float
            Степени свободы для априори весов (меньше = разреженнее).
        nu_f : float
            Степени свободы для априори изображения (меньше = резче грани).
        nu_noise : float
            Степени свободы для шума (меньше = устойчивее).
        kernel_sigma : float
            Ширина базиса гауссовых ядер.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='Tzikas2009')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.nu_w = nu_w
        self.nu_f = nu_f
        self.nu_noise = nu_noise
        self.kernel_sigma = kernel_sigma
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции с разреженным ядром и Стьюдентом.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Входное размытое изображение в градациях серого.
        
        Возвращает
        ----------
        f_est : ndarray (H, W)
            Восстановленное изображение.
        h_est : ndarray (kh, kw)
            Оценённое ядро размытия (PSF).
        """
        start_time = time.time()
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        y = np.asarray(image, dtype=np.float64)
        H, W = y.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Инициализация изображения наблюдением
        f = y.copy()
        M_f = fft2(f)
        S_f = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация весов PSF (разреженные, в основном нули)
        w = np.zeros((kh, kw), dtype=np.float64)
        w[kh//2, kw//2] = 1.0  # Дельта-инициализация
        
        # Инициализация PSF из весов
        h = _update_psf_from_weights(w, self.kernel_shape, self.kernel_sigma)
        h_padded = _pad_kernel_for_fft(h, (H, W))
        H_fft = fft2(h_padded)
        
        # Инициализация весов точности (равномерные)
        gamma_weights = np.ones((kh, kw))  # Точности весов PSF
        alpha_h = np.ones((H, W))          # Горизонтальные точности изображения
        alpha_v = np.ones((H, W))          # Вертикальные точности изображения
        beta_weights = np.ones((H, W))     # Точности шума
        
        # Глобальная точность шума
        E_beta = 1.0 / max(1e-3 * np.var(y), EPSILON)
        
        # Предвычисление операторов
        D_h, D_v = _compute_gradient_operators_fft((H, W))
        Y = fft2(y)
        
        # История
        self.history = {
            'beta': [E_beta],
            'sparsity': [np.sum(np.abs(w) < 0.01) / w.size],
        }
        
        E_beta_prev = E_beta
        
        for iteration in range(self.max_iterations):
            
            # Обновление весов точности из Стьюдента
            
            # Невязка для весов шума
            Hf = np.real(ifft2(H_fft * M_f))
            residual = y - Hf
            beta_weights = _compute_student_t_weights_noise(residual, self.nu_noise)
            
            # Веса априори изображения
            f_spatial = np.real(ifft2(M_f))
            alpha_h, alpha_v = _compute_student_t_weights_image(f_spatial, self.nu_f)
            
            # Веса априори весов PSF
            gamma_weights = _compute_student_t_weights_psf(w, self.nu_w)
            
            # Обновление q(f)
            M_f, S_f = _update_image(Y, H_fft, beta_weights, alpha_h, alpha_v, 
                                     D_h, D_v, (H, W))
            f_spatial = np.real(ifft2(M_f))
            f_spatial = np.clip(f_spatial, 0, None)  # Неотрицательность
            M_f = fft2(f_spatial)
            
            # Обновление q(w) и h
            beta_mean = np.mean(beta_weights) * E_beta
            w = _update_psf_weights(y, f_spatial, self.kernel_sigma, self.kernel_shape,
                                    gamma_weights, beta_mean, (H, W))
            
            # Обновление h из w
            h = _update_psf_from_weights(w, self.kernel_shape, self.kernel_sigma)
            h_padded = _pad_kernel_for_fft(h, (H, W))
            H_fft = fft2(h_padded)
            
            # Обновление точности шума
            Hf = np.real(ifft2(H_fft * M_f))
            residual = y - Hf
            E_beta = _update_noise_precision(residual, beta_weights)
            
            # История
            sparsity = np.sum(np.abs(w) < 0.01 * np.max(np.abs(w))) / w.size
            self.history['beta'].append(E_beta)
            self.history['sparsity'].append(sparsity)
            
            # Проверка сходимости
            delta = abs(E_beta - E_beta_prev) / max(E_beta_prev, EPSILON)
            
            if self.verbose:
                print(f"Итерация {iteration+1:3d}: β={E_beta:.4e}, "
                      f"разреженность={sparsity:.2%}, Δ={delta:.4e}")
            
            if delta < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
            
            E_beta_prev = E_beta
        
        # Извлечение финальных оценок
        f_est = np.clip(f_spatial, 0, None)
        h_est = h.copy()
        
        self.hyperparams = {
            'beta': E_beta,
            'nu_w': self.nu_w,
            'nu_f': self.nu_f,
            'nu_noise': self.nu_noise,
        }
        
        self.timer = time.time() - start_time
        
        return f_est, h_est
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """Возвращает текущие гиперпараметры алгоритма."""
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('nu_w', self.nu_w),
            ('nu_f', self.nu_f),
            ('nu_noise', self.nu_noise),
            ('kernel_sigma', self.kernel_sigma),
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
        if 'nu_w' in params:
            self.nu_w = float(params['nu_w'])
        if 'nu_f' in params:
            self.nu_f = float(params['nu_f'])
        if 'nu_noise' in params:
            self.nu_noise = float(params['nu_noise'])
        if 'kernel_sigma' in params:
            self.kernel_sigma = float(params['kernel_sigma'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """Возвращает историю сходимости."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает оценённые гиперпараметры."""
        return self.hyperparams


# Обратная совместимость
def sparse_kernel_blind_deconvolution(y, kernel_shape, **kwargs):
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
    f_est, h_est, hyperparams, history
    """
    algo = Tzikas2009(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(y)
    return f_est, h_est, algo.hyperparams, algo.history


def run_algorithm(y, kernel_shape, **kwargs):
    """Обёртка для sparse_kernel_blind_deconvolution()."""
    return sparse_kernel_blind_deconvolution(y, kernel_shape, **kwargs)
