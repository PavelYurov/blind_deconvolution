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
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


# Параметры гиперприори для гамма-распределений (плоские/неинформативные)
# Для Gamma(x | a, b): p(x) ∝ x^(a-1) exp(-b*x), Среднее = a/b
DEFAULT_HYPERPRIORS = {
    'a_alpha': 1e-3,   # Априори изображения
    'b_alpha': 1e-3,
    'a_beta': 1e-3,    # Точность шума
    'b_beta': 1e-3,
    'a_gamma': 1e-3,   # Априори размытия
    'b_gamma': 1e-3,
}


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


def _compute_laplacian_operator_fft(image_shape):
    """
    Вычисляет |C(k)|² в частотной области для оператора Лапласа C.
    
    Ядро Лапласа:
        [0, -1,  0]
        [-1, 4, -1]
        [0, -1,  0]
    
    Соответствует SAR априори ||Cf||² = f^T C^T C f.
    
    Параметры
    ---------
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа в частотной области.
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


def _update_q_f(G, M_h, S_h, E_alpha, E_beta, Lambda_C):
    """
    Обновляет q(f) = N(f | μ_f, Σ_f).
    
    Уравнения (22)-(23):
        Σ_f = (⟨β⟩ ⟨H^T H⟩ + ⟨α⟩ C^T C)^{-1}
        μ_f = ⟨β⟩ Σ_f ⟨H⟩^T g
    
    Ссылка: Ур. (22)-(23) в Molina et al. (2006)
    
    Параметры
    ---------
    G : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    E_alpha : float
        Мат. ожидание точности априори изображения.
    E_beta : float
        Мат. ожидание точности шума.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    
    Возвращает
    ----------
    M_f : ndarray (H, W), complex
        Апостериорное среднее f в частотной области.
    S_f : ndarray (H, W)
        Апостериорная дисперсия f в частотной области.
    """
    # E[H^t H] = |E[H]|² + cov(H)
    E_H_squared = np.abs(M_h) ** 2 + S_h
    
    # M^k(f) = β * E[H^t H] + α * C^t C
    precision_f = E_beta * E_H_squared + E_alpha * Lambda_C
    precision_f = np.maximum(precision_f, 1e-12)
    
    S_f = 1.0 / precision_f
    M_f = E_beta * S_f * np.conj(M_h) * G
    
    return M_f, S_f


def _update_q_h(G, M_f, S_f, E_gamma, E_beta, Lambda_D):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Уравнения (26)-(27).
    
    Ссылка: Ур. (26)-(27) в Molina et al. (2006)
    
    Параметры
    ---------
    G : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_f : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_f : ndarray (H, W)
        Дисперсия изображения в частотной области.
    E_gamma : float
        Мат. ожидание точности априори размытия.
    E_beta : float
        Мат. ожидание точности шума.
    Lambda_D : ndarray (H, W)
        Квадрат модуля оператора априори размытия.
    
    Возвращает
    ----------
    M_h : ndarray (H, W), complex
        Апостериорное среднее h в частотной области.
    S_h : ndarray (H, W)
        Апостериорная дисперсия h в частотной области.
    """
    # E[F^t F] = |E[F]|² + cov(F)
    E_F_squared = np.abs(M_f) ** 2 + S_f
    
    # M^k(h) = β * E[F^t F] + α_bl * C^t C
    precision_h = E_beta * E_F_squared + E_gamma * Lambda_D
    precision_h = np.maximum(precision_h, 1e-12)
    
    S_h = 1.0 / precision_h
    M_h = E_beta * S_h * np.conj(M_f) * G
    
    return M_h, S_h


def _project_kernel_constraints(M_h, S_h, kernel_shape, image_shape):
    """
    Проецирует оценку размытия на допустимое множество.
    
    Ограничения:
        1. Ограничение носителя: h ненулевое только в kernel_shape
        2. Неотрицательность: h ≥ 0
        3. Нормировка: Σ h = 1
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области (не изменяется).
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    M_h_proj : ndarray (H, W), complex
        Спроецированное размытие в частотной области.
    """
    h_spatial = np.real(ifft2(M_h))
    h_kernel = _extract_kernel_from_padded(h_spatial, kernel_shape)
    
    h_kernel = np.maximum(h_kernel, 0.0)
    
    h_sum = np.sum(h_kernel)
    if h_sum > 1e-12:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = _pad_kernel_for_fft(h_kernel, image_shape)
    M_h_proj = fft2(h_padded)
    
    return M_h_proj


def _update_q_alpha(M_f, S_f, Lambda_C, a_alpha, b_alpha):
    """
    Обновляет q(α), которое является гамма-распределением.
    
    Для гамма гиперприори Gamma(α | a, b):
        ⟨α⟩ = (N/2 + a - 1) / (E[||Cf||²]/2 + b)
    
    Ссылка: Раздел III-B в Molina et al. (2006)
    
    Параметры
    ---------
    M_f : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_f : ndarray (H, W)
        Дисперсия изображения в частотной области.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    a_alpha : float
        Параметр формы гамма-гиперприори.
    b_alpha : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_alpha : float
        Мат. ожидание точности априори изображения.
    """
    H, W = M_f.shape
    N = H * W
    
     # Ур. (44): E[||Cf||²] = ||C E[f]||² + trace(C^t C cov(f))
    # В частотной области это сумма элементов.
    E_Cf_squared = np.sum(Lambda_C * (np.abs(M_f) ** 2 + S_f)) / N
    
    numerator = a_alpha + N / 2.0
    denominator = b_alpha + E_Cf_squared / 2.0
    
    E_alpha = numerator / denominator
    E_alpha = max(E_alpha, 1e-12)
    
    return E_alpha


def _update_q_beta(G, M_f, S_f, M_h, S_h, a_beta, b_beta):
    """
    Обновляет q(β), которое является гамма-распределением.
    
    Ур. (57): E[β] = (a_beta^o + N/2) / (b_beta^o + 1/2 * E[||g - Hf||²])
    
    Ссылка: Раздел III-B в Molina et al. (2006)
    
    Параметры
    ---------
    G : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_f : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_f : ndarray (H, W)
        Дисперсия изображения в частотной области.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    a_beta : float
        Параметр формы гамма-гиперприори.
    b_beta : float
        Параметр масштаба гамма-гиперприори.
    
    Возвращает
    ----------
    E_beta : float
        Мат. ожидание точности шума.
    """
    H, W = G.shape
    N = H * W
    
    # Ур. (46)-(48): E[||g - Hf||²]
    residual_squared = np.abs(G - M_h * M_f) ** 2
    variance_terms = (S_f * np.abs(M_h) ** 2 + 
                      S_h * np.abs(M_f) ** 2 + 
                      S_f * S_h)
    
    E_error_squared = np.sum(residual_squared + variance_terms) / N
    
    numerator = a_beta + N / 2.0
    denominator = b_beta + E_error_squared / 2.0
    
    E_beta = numerator / denominator
    E_beta = max(E_beta, 1e-12)
    
    return E_beta


def _update_q_gamma(M_h, S_h, Lambda_D, a_gamma, b_gamma, kernel_shape):
    """
    Обновляет q(α_bl), которое является гамма-распределением.
    
    Ур. (56): E[α_bl] = (a_bl^o + M/2) / (b_bl^o + 1/2 * E[||Ch||²])
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия.
    S_h : ndarray (H, W)
        Дисперсия размытия.
    Lambda_D : ndarray (H, W)
        Квадрат модуля оператора априори размытия.
    a_gamma : float
        Параметр формы гамма-гиперприори.
    b_gamma : float
        Параметр масштаба гамма-гиперприори.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    
    Возвращает
    ----------
    E_gamma : float
        Мат. ожидание точности априори размытия.
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw  # В статье обозначается как M
    
    # Ур. (45): E[||Ch||²] = ||C E[h]||² + trace(C^t C cov(h))
    E_Dh_squared = np.sum(Lambda_D * (np.abs(M_h) ** 2 + S_h)) / N
    
    numerator = a_gamma + P / 2.0
    denominator = b_gamma + E_Dh_squared / 2.0
    
    E_gamma = numerator / denominator
    E_gamma = max(E_gamma, 1e-12)
    
    return E_gamma


class Molina2006(DeconvolutionAlgorithm):
    """
    Вариационная байесовская слепая деконволюция.
    
    Алгоритм из Molina et al. (2006) совместно оценивает латентное 
    изображение f, ядро размытия h и гиперпараметры (α, β, γ) 
    методом вариационного вывода.
    
    Вариационное приближение:
        q(f, h, α, β, γ) = q(f) q(h) q(α) q(β) q(γ)
    
    где q(f) и q(h) — гауссовы, а q(α), q(β), q(γ) — гамма-распределения.
    Используется модель SAR как для изображения, так и для размытия.

    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости.
    hyperpriors : dict
        Параметры гамма гиперприори.
    initial_alpha : float или None
        Начальное значение точности априори изображения.
    initial_beta : float или None
        Начальное значение точности шума.
    initial_gamma : float или None
        Начальное значение точности априори размытия.
    apply_kernel_constraints : bool
        Проецировать ядро на допустимое множество.
    verbose : bool
        Выводить прогресс.
    history : dict
        История сходимости (заполняется после process()).
    hyperparams : dict
        Оценённые гиперпараметры (заполняется после process()).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        hyperpriors: dict = None,
        initial_alpha: float = None,
        initial_beta: float = None,
        initial_gamma: float = None,
        apply_kernel_constraints: bool = True,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Molina2006.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число VB-итераций.
        tolerance : float
            Порог сходимости.
        hyperpriors : dict или None
            Параметры гамма гиперприори. Если None, используются значения
            по умолчанию (неинформативные приори).
        initial_alpha : float или None
            Начальное значение точности априори изображения.
        initial_beta : float или None
            Начальное значение точности шума.
        initial_gamma : float или None
            Начальное значение точности априори размытия.
        apply_kernel_constraints : bool
            Проецировать ядро на допустимое множество.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='Molina2006')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.hyperpriors = hyperpriors if hyperpriors else DEFAULT_HYPERPRIORS.copy()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.initial_gamma = initial_gamma
        self.apply_kernel_constraints = apply_kernel_constraints
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Входное размытое изображение в градациях серого.
            Значения пикселей могут быть в диапазоне [0, 255] или [0, 1].
        
        Возвращает
        ----------
        f_est : ndarray (H, W)
            Восстановленное изображение.
        h_est : ndarray (kh, kw)
            Оценённое ядро размытия (PSF).
        
        Исключения
        ----------
        ValueError
            Если изображение не является 2D массивом.
        
        Примечания
        ----------
        После выполнения метод заполняет атрибуты:
            - self.history: история сходимости гиперпараметров
            - self.hyperparams: финальные значения гиперпараметров
            - self.timer: время выполнения в секундах
        """
        start_time = time.time()
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        g = np.asarray(image, dtype=np.float64)
        H, W = g.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Параметры гиперприори
        a_alpha = self.hyperpriors.get('a_alpha', 1e-3)
        b_alpha = self.hyperpriors.get('b_alpha', 1e-3)
        a_beta = self.hyperpriors.get('a_beta', 1e-3)
        b_beta = self.hyperpriors.get('b_beta', 1e-3)
        a_gamma = self.hyperpriors.get('a_gamma', 1e-3)
        b_gamma = self.hyperpriors.get('b_gamma', 1e-3)
        
        # Предвычисление операторов
        Lambda_C = _compute_laplacian_operator_fft((H, W))
        # Согласно (Ур. 8), для размытия также используется оператор Лапласа
        Lambda_D = Lambda_C 
        # Lambda_D = np.ones((H, W), dtype=np.float64)  # Тождественный оператор для h
        
        # Инициализация изображения: μ_f = g
        f_init = g.copy()
        M_f = fft2(f_init)
        S_f = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация размытия: h = дельта-функция
        h_init = np.zeros((kh, kw), dtype=np.float64)
        h_init[kh // 2, kw // 2] = 1.0
        h_padded = _pad_kernel_for_fft(h_init, (H, W))
        M_h = fft2(h_padded)
        S_h = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация гиперпараметров
        noise_var_estimate = 1e-3 * np.var(g)
        
        if self.initial_beta is None:
            E_beta = 1.0 / max(noise_var_estimate, 1e-6)
        else:
            E_beta = self.initial_beta
            
        if self.initial_alpha is None:
            E_alpha = 1.0 / max(np.var(g), 1e-6)
        else:
            E_alpha = self.initial_alpha
            
        if self.initial_gamma is None:
            E_gamma = 1.0
        else:
            E_gamma = self.initial_gamma
        
        # Наблюдение в частотной области
        G = fft2(g)
        
        # История
        self.history = {
            'alpha': [E_alpha],
            'beta': [E_beta],
            'gamma': [E_gamma]
        }
        
        for iteration in range(self.max_iterations):
            
            E_alpha_prev = E_alpha
            E_beta_prev = E_beta
            E_gamma_prev = E_gamma
            
            # Обновление q(f)
            M_f, S_f = _update_q_f(G, M_h, S_h, E_alpha, E_beta, Lambda_C)
            
            # Обновление q(h)
            M_h, S_h = _update_q_h(G, M_f, S_f, E_gamma, E_beta, Lambda_D)
            
            # Проекция на допустимое множество
            if self.apply_kernel_constraints:
                M_h = _project_kernel_constraints(M_h, S_h, self.kernel_shape, (H, W))
            
            # Обновление гиперпараметров
            E_alpha = _update_q_alpha(M_f, S_f, Lambda_C, a_alpha, b_alpha)
            E_beta = _update_q_beta(G, M_f, S_f, M_h, S_h, a_beta, b_beta)
            E_gamma = _update_q_gamma(M_h, S_h, Lambda_D, a_gamma, b_gamma, self.kernel_shape)
            
            self.history['alpha'].append(E_alpha)
            self.history['beta'].append(E_beta)
            self.history['gamma'].append(E_gamma)
            
            # Проверка сходимости
            delta_alpha = abs(E_alpha - E_alpha_prev) / max(E_alpha_prev, 1e-12)
            delta_beta = abs(E_beta - E_beta_prev) / max(E_beta_prev, 1e-12)
            delta_gamma = abs(E_gamma - E_gamma_prev) / max(E_gamma_prev, 1e-12)
            max_delta = max(delta_alpha, delta_beta, delta_gamma)
            
            if self.verbose:
                print(f"Итерация {iteration+1:3d}: α={E_alpha:.4e}, β={E_beta:.4e}, "
                      f"γ={E_gamma:.4e}, Δ={max_delta:.4e}")
            
            if max_delta < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
        
        # Извлечение финальных оценок
        f_est = np.real(ifft2(M_f))
        
        h_spatial = np.real(ifft2(M_h))
        h_est = _extract_kernel_from_padded(h_spatial, self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = np.sum(h_est)
        if h_sum > 1e-12:
            h_est = h_est / h_sum
        
        self.hyperparams = {
            'alpha': E_alpha,
            'beta': E_beta,
            'gamma': E_gamma
        }
        
        self.timer = time.time() - start_time
        
        return f_est, h_est
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """
        Возвращает текущие гиперпараметры алгоритма.
        
        Возвращает
        ----------
        params : list of tuple
            Список кортежей (название_параметра, значение).
        """
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('hyperpriors', self.hyperpriors),
            ('initial_alpha', self.initial_alpha),
            ('initial_beta', self.initial_beta),
            ('initial_gamma', self.initial_gamma),
            ('apply_kernel_constraints', self.apply_kernel_constraints),
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """
        Изменяет гиперпараметры алгоритма.
        
        Параметры
        ---------
        params : dict
            Словарь с параметрами для изменения.
            Ключи — названия параметров, значения — новые значения.
        """
        if 'kernel_shape' in params:
            self.kernel_shape = tuple(params['kernel_shape'])
        if 'max_iterations' in params:
            self.max_iterations = int(params['max_iterations'])
        if 'tolerance' in params:
            self.tolerance = float(params['tolerance'])
        if 'hyperpriors' in params:
            self.hyperpriors = params['hyperpriors']
        if 'initial_alpha' in params:
            self.initial_alpha = params['initial_alpha']
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'initial_gamma' in params:
            self.initial_gamma = params['initial_gamma']
        if 'apply_kernel_constraints' in params:
            self.apply_kernel_constraints = bool(params['apply_kernel_constraints'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """
        Возвращает историю сходимости.
        
        Возвращает
        ----------
        history : dict
            Словарь с историей значений гиперпараметров по итерациям.
            Ключи: 'alpha', 'beta', 'gamma'.
        """
        return self.history
    
    def get_hyperparams(self) -> dict:
        """
        Возвращает оценённые гиперпараметры.
        
        Возвращает
        ----------
        hyperparams : dict
            Словарь с финальными значениями гиперпараметров.
            Ключи: 'alpha', 'beta', 'gamma'.
        """
        return self.hyperparams


# Обратная совместимость
def variational_blind_deconvolution(g, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    g : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры для Molina2006.
    
    Возвращает
    ----------
    f_est : ndarray (H, W)
        Восстановленное изображение.
    h_est : ndarray (kh, kw)
        Оценённое ядро.
    hyperparams : dict
        Оценённые гиперпараметры.
    history : dict
        История сходимости.
    """
    algo = Molina2006(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history


def run_algorithm(g, kernel_shape, **kwargs):
    """Обёртка для variational_blind_deconvolution()."""
    return variational_blind_deconvolution(g, kernel_shape, **kwargs)
