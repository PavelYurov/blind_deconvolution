"""
Вариационный подход к байесовской слепой деконволюции (алгоритм VAR3)

Литература:
    Likas, A. C., & Galatsanos, N. P. (2004).
    A variational approach for Bayesian blind image deconvolution.
    IEEE Transactions on Signal Processing, 52(8), 2222-2233.
    DOI: 10.1109/TSP.2004.831119

Реализация подхода VAR3 с использованием вариационного приближения 
к полному байесовскому апостериорному распределению.
В отличие от VAR1, здесь используется двухэтапная процедура (Alternating Variational):
    1. Этап f: Ядро h считается детерминированным (ковариация S_h = 0).
    2. Этап h: Изображение f считается детерминированным (ковариация S_f = 0).

Модель:
    - Гауссова модель наблюдений: p(g|f,h,β) ∝ exp(-β/2 ||g - h*f||²)
    - SAR априори для изображения: p(f|α) ∝ exp(-α/2 ||Cf||²)  
    - Гауссово априори для размытия: p(h|γ) ∝ exp(-γ/2 ||h||²)

Подход VAR3 учитывает неопределённость в оценках изображения и ядра
(апостериорные ковариационные члены) при вариационных обновлениях.

Реализованные уравнения (нумерация по статье):
    - Ур. (10): Вариационная нижняя граница для случая VAR3
    - Ур. (11): Обновление шума β на этапе оценки f
    - Ур. (12): Обновление шума β на этапе оценки h
    - Appendix B: Обновление параметров q(f) и q(h) (аналог фильтра Винера)
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


EPSILON = 1e-12


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


def _compute_laplacian_spectrum(image_shape):
    """
    Вычисляет |C(k)|² для оператора Лапласа в частотной области.
    
    Ядро Лапласа:
        [0, -1,  0]
        [-1, 4, -1]
        [0, -1,  0]
    
    Используется для SAR априори: ||Cf||² = f^T C^T C f.
    
    Ссылка: Раздел II-B в Likas & Galatsanos (2004)
    
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
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float64)
    
    C_padded = np.zeros((H, W), dtype=np.float64)
    C_padded[:3, :3] = laplacian
    C_padded = np.roll(C_padded, shift=-1, axis=0)
    C_padded = np.roll(C_padded, shift=-1, axis=1)
    
    return np.abs(fft2(C_padded))**2


def _update_q_f(G, M_h, S_h, alpha, beta, Lambda_C):
    """
    Обновляет q(f) = N(f | μ_f, Σ_f).
    
    Ур. (16): Σ_f(k) = [β E[|H(k)|²] + α |C(k)|²]^{-1}
    Ур. (17): μ_f(k) = β Σ_f(k) M_h(k)* G(k)
    
    где E[|H(k)|²] = |M_h(k)|² + S_h(k).
    
    Формула Винера обобщённая на случай неопределённости в ядре.
    
    Ссылка: Ур. (16)-(17) в Likas & Galatsanos (2004)
    
    Параметры
    ---------
    G : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    alpha : float
        Точность априори изображения.
    beta : float
        Точность шума.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    
    Возвращает
    ----------
    M_f : ndarray (H, W), complex
        Апостериорное среднее f в частотной области.
    S_f : ndarray (H, W)
        Апостериорная дисперсия f в частотной области.
    """
    E_H_sq = np.abs(M_h)**2 + S_h
    precision = beta * E_H_sq + alpha * Lambda_C + EPSILON
    S_f = 1.0 / precision
    M_f = beta * S_f * np.conj(M_h) * G
    return M_f, S_f


def _update_q_h(G, M_f, S_f, gamma, beta):
    """
    Обновляет q(h) = N(h | μ_h, Σ_h).
    
    Ур. (18): Σ_h(k) = [β E[|F(k)|²] + γ]^{-1}
    Ур. (19): μ_h(k) = β Σ_h(k) M_f(k)* G(k)
    
    где E[|F(k)|²] = |M_f(k)|² + S_f(k).
    
    Ссылка: Ур. (18)-(19) в Likas & Galatsanos (2004)
    
    Параметры
    ---------
    G : ndarray (H, W), complex
        БПФ наблюдаемого изображения.
    M_f : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_f : ndarray (H, W)
        Дисперсия изображения в частотной области.
    gamma : float
        Точность априори размытия.
    beta : float
        Точность шума.
    
    Возвращает
    ----------
    M_h : ndarray (H, W), complex
        Апостериорное среднее h в частотной области.
    S_h : ndarray (H, W)
        Апостериорная дисперсия h в частотной области.
    """
    E_F_sq = np.abs(M_f)**2 + S_f
    precision = beta * E_F_sq + gamma + EPSILON
    S_h = 1.0 / precision
    M_h = beta * S_h * np.conj(M_f) * G
    return M_h, S_h


def _project_kernel_constraints(M_h, kernel_shape, image_shape):
    """
    Проецирует h на допустимое множество (неотрицательность, нормировка).
    
    Ограничения:
        1. Ограничение носителя: h ненулевое только в kernel_shape
        2. Неотрицательность: h ≥ 0
        3. Нормировка: Σ h = 1
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
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
    if h_sum > EPSILON:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0]//2, kernel_shape[1]//2] = 1.0
    
    h_padded = _pad_kernel_for_fft(h_kernel, image_shape)
    return fft2(h_padded)


def _update_alpha(M_f, S_f, Lambda_C):
    """
    Обновляет точность априори изображения α.
    
    Ур. (20): α = N / E[||Cf||²]
    
    где E[||Cf||²] = (1/N) Σ_k |C(k)|² (|M_f(k)|² + S_f(k))
    
    Параметры
    ---------
    M_f : ndarray (H, W), complex
        Среднее изображения в частотной области.
    S_f : ndarray (H, W)
        Дисперсия изображения в частотной области.
    Lambda_C : ndarray (H, W)
        Квадрат модуля оператора Лапласа.
    
    Возвращает
    ----------
    alpha : float
        Обновлённая точность априори изображения.
    """
    H, W = M_f.shape
    N = H * W
    E_Cf_sq = np.sum(Lambda_C * (np.abs(M_f)**2 + S_f)) / N
    return N / (E_Cf_sq + EPSILON)


def _update_gamma(M_h, S_h, kernel_shape):
    """
    Обновляет точность априори размытия γ.
    
    Ур. (21): γ = P / E[||h||²]
    
    где P = kh × kw — размер ядра.
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    
    Возвращает
    ----------
    gamma : float
        Обновлённая точность априори размытия.
    """
    H, W = M_h.shape
    N = H * W
    kh, kw = kernel_shape
    P = kh * kw
    E_h_sq = np.sum(np.abs(M_h)**2 + S_h) / N
    return P / (E_h_sq + EPSILON)


def _update_beta(G, M_f, S_f, M_h, S_h):
    """
    Обновляет точность шума β.
    
    Ур. (22): β = N / E[||g - h*f||²]
    
    где E[||g - h*f||²] = (1/N) Σ_k {|G(k) - M_h(k)M_f(k)|² 
                                    + S_f(k)|M_h(k)|² 
                                    + S_h(k)|M_f(k)|² 
                                    + S_f(k)S_h(k)}
    
    
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
    
    Возвращает
    ----------
    beta : float
        Обновлённая точность шума.
    """
    H, W = G.shape
    N = H * W
    
    residual_sq = np.abs(G - M_h * M_f)**2
    variance_terms = (S_f * np.abs(M_h)**2 + 
                      S_h * np.abs(M_f)**2 + 
                      S_f * S_h)
    
    E_error_sq = np.sum(residual_sq + variance_terms) / N
    return N / (E_error_sq + EPSILON)


class VABID(DeconvolutionAlgorithm):
    """
    Алгоритм VAR3 для вариационной байесовской слепой деконволюции.
    
    Совместно оценивает латентное изображение f, ядро размытия h
    и гиперпараметры (α, β, γ) методом вариационного вывода.
    
    Вариационное приближение:
        q(f, h) = q(f) q(h)
    
    где q(f) и q(h) — гауссовы распределения с моментами, вычисляемыми
    в частотной области (диагональное приближение ковариации).

    Используется схема Alternating Variational Minimization (VAR3):
    1. E-шаг и M-шаг для f при фиксированном h (h считается детерминированным).
    2. E-шаг и M-шаг для h при фиксированном f (f считается детерминированным).
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число VB-итераций.
    tolerance : float
        Порог сходимости (относительное изменение гиперпараметров).
    initial_alpha : float или None
        Начальное значение точности априори изображения.
    initial_gamma : float или None
        Начальное значение точности априори ядра.
    initial_beta : float или None
        Начальное значение точности шума.
    apply_kernel_constraints : bool
        Проецировать ядро на допустимое множество.
    verbose : bool
        Выводить прогресс на консоль.
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
        initial_alpha: float = None,
        initial_gamma: float = None,
        initial_beta: float = None,
        apply_kernel_constraints: bool = True,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма Likas2004 (VAR3).
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число VB-итераций.
        tolerance : float
            Порог сходимости.
        initial_alpha : float или None
            Начальное значение точности априори изображения.
        initial_gamma : float или None
            Начальное значение точности априори ядра.
        initial_beta : float или None
            Начальное значение точности шума.
        apply_kernel_constraints : bool
            Проецировать ядро на допустимое множество.
        verbose : bool
            Выводить прогресс на консоль.
        """
        super().__init__(name='Likas2004')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.initial_alpha = initial_alpha
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta
        self.apply_kernel_constraints = apply_kernel_constraints
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции методом VAR3.
        
        Реализует итеративный процесс, описанный в Section III-B.

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
        
        # Предвычисление спектра Лапласа
        Lambda_C = _compute_laplacian_spectrum((H, W))
        
        # Инициализация изображения: f = g
        f_init = g.copy()
        M_f = fft2(f_init)
        # S_f инициализируем, но в цикле VAR3 он будет вычисляться локально на шаге 1
        S_f = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация размытия: дельта-функция
        h_init = np.zeros((kh, kw), dtype=np.float64)
        h_init[kh//2, kw//2] = 1.0
        h_padded = _pad_kernel_for_fft(h_init, (H, W))
        M_h = fft2(h_padded)
        # S_h инициализируем, но в цикле VAR3 он будет вычисляться локально на шаге 2
        S_h = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация гиперпараметров
        noise_var = max(1e-3 * np.var(g), EPSILON)
        
        alpha = self.initial_alpha if self.initial_alpha else 1.0 / np.var(g)
        gamma = self.initial_gamma if self.initial_gamma else 1.0
        beta = self.initial_beta if self.initial_beta else 1.0 / noise_var
        
        # БПФ наблюдения
        G = fft2(g)
        
        # Нулевой массив для передачи в функции, где ковариация считается равной нулю
        zeros_shape = (H, W)
        zeros_cov = np.zeros(zeros_shape, dtype=np.float64)

        # История
        self.history = {
            'alpha': [alpha],
            'beta': [beta],
            'gamma': [gamma]
        }
        
        for iteration in range(self.max_iterations):
            
            params_prev = np.array([alpha, beta, gamma])
            
            # Обновление f (h фиксировано и детерминировано) 
            # h считается детерминированным, значит S_h = 0
            # Обновление q(f) — Ур. (16)-(17)

            # E-step f:
            M_f, S_f = _update_q_f(G, M_h, zeros_cov, alpha, beta, Lambda_C)
            
            # M-step f (обновление alpha):
            alpha = _update_alpha(M_f, S_f, Lambda_C)
            
            # Обновление шума beta (Ур. 11): используется S_f, но S_h=0
            beta = _update_beta(G, M_f, S_f, M_h, zeros_cov)

            # Обновление h (f фиксировано и детерминировано)
            # f считается детерминированным, значит S_f = 0
            # Обновление q(h) — Ур. (18)-(19)

            # E-step h:
            M_h, S_h = _update_q_h(G, M_f, zeros_cov, gamma, beta)
            
            # Проекция на допустимое множество
            if self.apply_kernel_constraints:
                M_h = _project_kernel_constraints(M_h, self.kernel_shape, (H, W))
            
            # Обновление гиперпараметров
            # M-step h (обновление gamma):
            gamma = _update_gamma(M_h, S_h, self.kernel_shape)
             # Обновление шума beta (Ур. 12): используется S_h, но S_f=0
            beta = _update_beta(G, M_f, zeros_cov, M_h, S_h)
            
            self.history['alpha'].append(alpha)
            self.history['beta'].append(beta)
            self.history['gamma'].append(gamma)
            
            # Проверка сходимости
            params_curr = np.array([alpha, beta, gamma])
            max_delta = np.max(np.abs(params_curr - params_prev) / (params_prev + EPSILON))
            
            if self.verbose:
                print(f"Итерация {iteration+1:3d}: α={alpha:.4e}, β={beta:.4e}, "
                      f"γ={gamma:.4e}, Δ={max_delta:.4e}")
            
            if max_delta < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
        
        # Извлечение результатов
        f_est = np.real(ifft2(M_f))
        
        h_spatial = np.real(ifft2(M_h))
        h_est = _extract_kernel_from_padded(h_spatial, self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = np.sum(h_est)
        if h_sum > EPSILON:
            h_est = h_est / h_sum
        
        self.hyperparams = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma
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
            ('initial_alpha', self.initial_alpha),
            ('initial_gamma', self.initial_gamma),
            ('initial_beta', self.initial_beta),
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
        if 'initial_alpha' in params:
            self.initial_alpha = params['initial_alpha']
        if 'initial_gamma' in params:
            self.initial_gamma = params['initial_gamma']
        if 'initial_beta' in params:
            self.initial_beta = params['initial_beta']
        if 'apply_kernel_constraints' in params:
            self.apply_kernel_constraints = bool(params['apply_kernel_constraints'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """
        Возвращает историю сходимости после выполнения process().
        
        Возвращает
        ----------
        history : dict
            Словарь с историей значений гиперпараметров по итерациям.
            Ключи: 'alpha', 'beta', 'gamma'.
        """
        return self.history
    
    def get_hyperparams(self) -> dict:
        """
        Возвращает оценённые гиперпараметры после выполнения process().
        
        Возвращает
        ----------
        hyperparams : dict
            Словарь с финальными значениями гиперпараметров.
            Ключи: 'alpha', 'beta', 'gamma'.
        """
        return self.hyperparams


# Обратная совместимость: функция-обёртка
def var3_blind_deconvolution(g, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    g : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры для Likas2004.
    
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
    algo = Likas2004(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history


def run_algorithm(g, kernel_shape, **kwargs):
    """Обёртка для var3_blind_deconvolution()."""
    return var3_blind_deconvolution(g, kernel_shape, **kwargs)
