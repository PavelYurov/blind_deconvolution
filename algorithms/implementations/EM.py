"""
EM-алгоритм для слепой деконволюции изображений

Реализация EM-алгоритма для задачи слепой деконволюции:
    g = h * f + n

где:
    g — наблюдаемое (размытое + зашумленное) изображение
    f — неизвестное оригинальное изображение
    h — неизвестная функция рассеяния точки (PSF)
    n — аддитивный гауссов шум

EM-алгоритм рассматривает f как латентную переменную и чередует:
    E-шаг: вычисление апостериорного распределения f при текущих оценках
    M-шаг: обновление параметров (h, дисперсия шума, регуляризация)

Литература:
    Lagendijk, R. L., Biemond, J., & Boekee, D. E. (1990).
    IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(7), 1180-1191.
    
    Katsaggelos, A. K., & Lay, K. T. (1991).
    IEEE Transactions on Signal Processing, 39(3), 729-733.
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


def _pad_kernel(h, image_shape):
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
    h_padded = np.roll(h_padded, -kh // 2, axis=0)
    h_padded = np.roll(h_padded, -kw // 2, axis=1)
    return h_padded


def _extract_kernel(h_padded, kernel_shape):
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
    shifted = np.roll(h_padded, kh // 2, axis=0)
    shifted = np.roll(shifted, kw // 2, axis=1)
    return shifted[:kh, :kw]


def _compute_laplacian_spectrum(image_shape):
    """
    Вычисляет |C(k)|² для оператора Лапласа в частотной области.
    
    Ядро Лапласа:
        [0, -1, 0]
        [-1, 4, -1]
        [0, -1, 0]
    
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
    lap = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
    C = np.zeros((H, W), dtype=np.float64)
    C[:3, :3] = lap
    C = np.roll(C, -1, axis=0)
    C = np.roll(C, -1, axis=1)
    return np.abs(fft2(C)) ** 2


def _e_step(G, M_h, S_h, alpha, beta, Lambda_C):
    """
    E-шаг: вычисляет апостериорное распределение q(f) = N(μ_f, Σ_f).
    
    В частотной области (циркулянтное приближение):
        Σ_f(k) = [β |H(k)|² + α |C(k)|²]^{-1}
        μ_f(k) = β Σ_f(k) H(k)* G(k)
    
    При учёте неопределённости H:
        |H(k)|² заменяется на E[|H(k)|²] = |M_h(k)|² + S_h(k)
    
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
    # E[|H(k)|²] = |M_h(k)|² + S_h(k)
    E_H_sq = np.abs(M_h) ** 2 + S_h
    
    # Апостериорная точность
    precision = beta * E_H_sq + alpha * Lambda_C
    precision = np.maximum(precision, 1e-12)
    
    # Апостериорная дисперсия
    S_f = 1.0 / precision
    
    # Апостериорное среднее (формула Винера)
    M_f = beta * S_f * np.conj(M_h) * G
    
    return M_f, S_f


def _m_step_image_prior(M_f, S_f, Lambda_C):
    """
    M-шаг: обновление точности априори изображения α.
    
    α = N / E[||Cf||²]
    
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
    
    E_Cf_sq = np.sum(Lambda_C * (np.abs(M_f) ** 2 + S_f)) / N
    alpha = N / max(E_Cf_sq, 1e-12)
    
    return alpha


def _m_step_noise_precision(G, M_f, S_f, M_h, S_h):
    """
    M-шаг: обновление точности шума β.
    
    β = N / E[||g - h*f||²]
    
    где:
        E[||g - h*f||²] = (1/N) Σ_k { |G(k) - M_h(k) M_f(k)|²
                                      + S_f(k) |M_h(k)|²
                                      + S_h(k) |M_f(k)|²  
                                      + S_f(k) S_h(k) }
    
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
    
    residual_sq = np.abs(G - M_h * M_f) ** 2
    variance_terms = (S_f * np.abs(M_h) ** 2 + 
                      S_h * np.abs(M_f) ** 2 + 
                      S_f * S_h)
    
    E_error_sq = np.sum(residual_sq + variance_terms) / N
    beta = N / max(E_error_sq, 1e-12)
    
    return beta


def _m_step_blur(G, M_f, S_f, gamma, beta, Lambda_D):
    """
    M-шаг: обновление оценки размытия h.
    
    Апостериорное распределение h при заданных f и g:
        Σ_h(k) = [β E[|F(k)|²] + γ |D(k)|²]^{-1}
        μ_h(k) = β Σ_h(k) E[F(k)]* G(k)
    
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
    Lambda_D : ndarray (H, W)
        Квадрат модуля оператора априори размытия.
    
    Возвращает
    ----------
    M_h : ndarray (H, W), complex
        Апостериорное среднее h в частотной области.
    S_h : ndarray (H, W)
        Апостериорная дисперсия h в частотной области.
    """
    E_F_sq = np.abs(M_f) ** 2 + S_f
    
    precision_h = beta * E_F_sq + gamma * Lambda_D
    precision_h = np.maximum(precision_h, 1e-12)
    
    S_h = 1.0 / precision_h
    M_h = beta * S_h * np.conj(M_f) * G
    
    return M_h, S_h


def _m_step_blur_prior(M_h, S_h, kernel_shape, N):
    """
    M-шаг: обновление точности априори размытия γ.
    
    γ = P / E[||h||²]
    
    где P — число пикселей в носителе ядра.
    
    Параметры
    ---------
    M_h : ndarray (H, W), complex
        Среднее размытия в частотной области.
    S_h : ndarray (H, W)
        Дисперсия размытия в частотной области.
    kernel_shape : tuple (kh, kw)
        Размер ядра.
    N : int
        Общее число пикселей изображения.
    
    Возвращает
    ----------
    gamma : float
        Обновлённая точность априори размытия.
    """
    kh, kw = kernel_shape
    P = kh * kw
    
    E_h_sq = np.sum(np.abs(M_h) ** 2 + S_h) / N
    gamma = P / max(E_h_sq, 1e-12)
    
    return gamma


def _project_kernel(M_h, kernel_shape, image_shape):
    """
    Проецирует размытие на допустимое множество.
    
    Ограничения:
        1. Ограничение носителя
        2. Неотрицательность
        3. Нормировка (сумма = 1)
    
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
    h_kernel = _extract_kernel(h_spatial, kernel_shape)
    
    # Неотрицательность
    h_kernel = np.maximum(h_kernel, 0.0)
    
    # Нормировка
    h_sum = np.sum(h_kernel)
    if h_sum > 1e-12:
        h_kernel = h_kernel / h_sum
    else:
        h_kernel = np.zeros(kernel_shape)
        h_kernel[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0
    
    h_padded = _pad_kernel(h_kernel, image_shape)
    return fft2(h_padded)


class EMBlindDeconvolution(DeconvolutionAlgorithm):
    """
    EM-алгоритм для слепой деконволюции изображений.
    
    Чередует:
        E-шаг: оценка апостериорного распределения латентного изображения f
        M-шаг: обновление размытия h и гиперпараметров (α, β, γ)
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    max_iterations : int
        Максимальное число EM-итераций.
    tolerance : float
        Порог сходимости.
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
        verbose: bool = False
    ):
        """
        Инициализация EM-алгоритма.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        max_iterations : int
            Максимальное число EM-итераций.
        tolerance : float
            Порог сходимости.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='EMBlindDeconvolution')
        
        self.kernel_shape = tuple(kernel_shape)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции методом EM.
        
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
        
        g = np.asarray(image, dtype=np.float64)
        H, W = g.shape
        N = H * W
        kh, kw = self.kernel_shape
        
        # Инициализация изображения
        f_init = g.copy()
        M_f = fft2(f_init)
        S_f = np.zeros((H, W), dtype=np.float64)
        
        # Инициализация размытия: дельта-функция
        h_init = np.zeros((kh, kw), dtype=np.float64)
        h_init[kh // 2, kw // 2] = 1.0
        h_padded = _pad_kernel(h_init, (H, W))
        M_h = fft2(h_padded)
        S_h = np.zeros((H, W), dtype=np.float64)
        
        # Гиперпараметры
        alpha = 1.0 / max(np.var(g), 1e-6)
        beta = 1.0 / (1e-3 * np.var(g))
        gamma = 1.0
        
        # Операторы
        Lambda_C = _compute_laplacian_spectrum((H, W))
        Lambda_D = np.ones((H, W), dtype=np.float64)
        
        # Наблюдение в частотной области
        G = fft2(g)
        
        prev_beta = beta
        
        # История
        self.history = {
            'alpha': [alpha],
            'beta': [beta],
            'gamma': [gamma]
        }
        
        for iteration in range(self.max_iterations):
            
            # E-шаг
            M_f, S_f = _e_step(G, M_h, S_h, alpha, beta, Lambda_C)
            
            # M-шаг: обновление h
            M_h, S_h = _m_step_blur(G, M_f, S_f, gamma, beta, Lambda_D)
            M_h = _project_kernel(M_h, self.kernel_shape, (H, W))
            
            # M-шаг: обновление гиперпараметров
            alpha = _m_step_image_prior(M_f, S_f, Lambda_C)
            beta = _m_step_noise_precision(G, M_f, S_f, M_h, S_h)
            gamma = _m_step_blur_prior(M_h, S_h, self.kernel_shape, N)
            
            self.history['alpha'].append(alpha)
            self.history['beta'].append(beta)
            self.history['gamma'].append(gamma)
            
            # Проверка сходимости
            delta = abs(beta - prev_beta) / max(prev_beta, 1e-12)
            
            if self.verbose:
                print(f"EM Итерация {iteration+1:3d}: α={alpha:.4e}, β={beta:.4e}, "
                      f"γ={gamma:.4e}, Δβ={delta:.4e}")
            
            if delta < self.tolerance:
                if self.verbose:
                    print(f"Сходимость достигнута на итерации {iteration+1}")
                break
            
            prev_beta = beta
        
        # Извлечение результатов
        f_est = np.real(ifft2(M_f))
        
        h_spatial = np.real(ifft2(M_h))
        h_est = _extract_kernel(h_spatial, self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = np.sum(h_est)
        if h_sum > 1e-12:
            h_est = h_est / h_sum
        
        self.hyperparams = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        
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
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """
        Изменяет гиперпараметры алгоритма.
        
        Параметры
        ---------
        params : dict
            Словарь с параметрами для изменения.
        """
        if 'kernel_shape' in params:
            self.kernel_shape = tuple(params['kernel_shape'])
        if 'max_iterations' in params:
            self.max_iterations = int(params['max_iterations'])
        if 'tolerance' in params:
            self.tolerance = float(params['tolerance'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        """Возвращает историю сходимости."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает оценённые гиперпараметры."""
        return self.hyperparams


# Обратная совместимость
def em_blind_deconvolution(g, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    g : ndarray (H, W)
        Наблюдаемое размытое изображение.
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    **kwargs
        Дополнительные параметры.
    
    Возвращает
    ----------
    f_est, h_est, params
    """
    algo = EMBlindDeconvolution(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams


def run_em(g, kernel_shape, **kwargs):
    """Обёртка для em_blind_deconvolution()."""
    return em_blind_deconvolution(g, kernel_shape, **kwargs)
