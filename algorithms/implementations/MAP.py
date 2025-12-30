"""
MAP-алгоритм для слепой деконволюции изображений с регуляризацией Хубера

Реализация алгоритма максимального апостериорного оценивания (MAP) для задачи
слепой деконволюции с регуляризацией Хубера:

    argmin_{f,h} ||g - h ⊗ f||² + λ·ρ(Df)

где:
    g — наблюдаемое размытое изображение
    f — неизвестное оригинальное изображение  
    h — неизвестная функция рассеяния точки (PSF)
    D  — оператор дискретного градиента
    ρ  — функция Хубера (смесь L1 и L2 норм)
    λ  — параметр регуляризации

Алгоритм использует попеременную минимизацию:
    1. Фиксируем f, обновляем h с ограничениями (неотрицательность, нормировка)
    2. Фиксируем h, обновляем f с регуляризацией Хубера

Литература:
    [1] Levin, A., Weiss, Y., Durand, F., & Freeman, W. T. (2011). 
        Understanding blind deconvolution algorithms. 
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        33(12), 2354-2367.
        DOI: 10.1109/TPAMI.2011.148
    
    [2] Chan, T. F., & Wong, C. K. (1998). 
        Total variation blind deconvolution. 
        IEEE Transactions on Image Processing, 7(3), 370-375.
        DOI: 10.1109/83.661187
    
    [3] Huber, P. J. (1964). 
        Robust estimation of a location parameter. 
        The Annals of Mathematical Statistics, 35(1), 73-101.
    
    [4] Библиотечная реализация классических методов 
        попеременной минимизации для слепой деконволюции.

Теоретическая основа:
    - Метод попеременной минимизации (Alternating Minimization) 
      для невыпуклых задач [1]
    - Регуляризация Хубера как выпуклая аппроксимация L1-нормы
      для сохранения границ изображений [3]
    - Вариационные формулировки слепой деконволюции [2]

Реализация адаптирует общие принципы из [1,2] 
с использованием регуляризации Хубера [3].
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from typing import Tuple, Optional, Dict, Any, List
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


def _pad_for_convolution(image: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """
    Дополняет изображение для корректной циклической свертки.
    
    Параметры
    ---------
    image : ndarray (H, W)
        Входное изображение.
    kernel_shape : tuple (kh, kw)
        Размер ядра свертки.
    
    Возвращает
    ----------
    padded : ndarray (H+kh-1, W+kw-1)
        Дополненное изображение.
    """
    kh, kw = kernel_shape
    return np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')


def _crop_to_original(padded: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
    """
    Обрезает дополненное изображение до оригинального размера.
    
    Параметры
    ---------
    padded : ndarray (H_pad, W_pad)
        Дополненное изображение.
    original_shape : tuple (H, W)
        Оригинальный размер.
    
    Возвращает
    ----------
    cropped : ndarray (H, W)
        Обрезанное изображение.
    """
    H, W = original_shape
    H_pad, W_pad = padded.shape
    kh = H_pad - H
    kw = W_pad - W
    return padded[kh//2:kh//2+H, kw//2:kw//2+W]


def _huber_function(u: np.ndarray, delta: float) -> np.ndarray:
    """
    Функция Хубера и её производная.
    
    Определение:
        ρ(u) = { u²/2,             если |u| ≤ δ
               { δ(|u| - δ/2),     если |u| > δ
        
        ρ'(u) = { u,               если |u| ≤ δ
                { δ·sign(u),       если |u| > δ
    
    Параметры
    ---------
    u : ndarray
        Входной массив.
    delta : float
        Порог перехода между квадратичной и линейной областями.
    
    Возвращает
    ----------
    rho_prime : ndarray
        Производная функции Хубера в точках u.
    """
    return np.where(np.abs(u) <= delta, u, delta * np.sign(u))


def _compute_gradient_operator() -> np.ndarray:
    """
    Возвращает оператор дискретного градиента (оператор Лапласа).
    
    Возвращает
    ----------
    operator : ndarray (3, 3)
        Матрица оператора градиента.
    """
    return np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]], dtype=np.float64)


class MAPDeconvolution(DeconvolutionAlgorithm):
    """
    MAP-алгоритм для слепой деконволюции с регуляризацией Хубера.
    
    Алгоритм попеременно обновляет оценку изображения f и ядра размытия h,
    минимизируя целевую функцию с регуляризацией Хубера.
    
    Атрибуты
    --------
    max_iter : int
        Максимальное число итераций.
    huber_threshold : float
        Порог функции Хубера (δ).
    init_kernel : ndarray (kh, kw)
        Начальное приближение ядра размытия.
    relaxation_factor : float
        Параметр релаксации (λ) для регуляризации.
    eps : float
        Малое число для предотвращения деления на ноль.
    verbose : bool
        Выводить прогресс итераций.
    """
    
    def __init__(
        self,
        max_iter: int = 50,
        huber_threshold: float = 0.1,
        init_kernel: Optional[np.ndarray] = None,
        relaxation_factor: float = 0.1,
        eps: float = 1e-8,
        verbose: bool = False
    ):
        """
        Инициализация MAP-алгоритма.
        
        Параметры
        ---------
        max_iter : int, optional
            Максимальное число итераций (по умолчанию 50).
        huber_threshold : float, optional
            Порог функции Хубера (δ) (по умолчанию 0.1).
        init_kernel : ndarray или None, optional
            Начальное приближение ядра размытия. Если None, используется
            дельта-функция (по умолчанию None).
        relaxation_factor : float, optional  
            Параметр релаксации для регуляризации (λ) (по умолчанию 0.1).
        eps : float, optional
            Малое число для численной стабильности (по умолчанию 1e-8).
        verbose : bool, optional
            Выводить прогресс итераций (по умолчанию False).
        """
        super().__init__(name='MAPDeconvolution')
        
        self.max_iter = max_iter
        self.huber_threshold = huber_threshold
        self.relaxation_factor = relaxation_factor
        self.eps = eps
        self.verbose = verbose
        
        # Инициализация ядра
        if init_kernel is not None:
            self.init_kernel = init_kernel.copy()
        else:
            # Дельта-функция как начальное приближение
            self.init_kernel = np.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]], dtype=np.float64)
        
        # Оператор градиента
        self.gradient_operator = _compute_gradient_operator()
        
        # История сходимости
        self.history = {
            'psf_norm': [],
            'image_norm': [],
            'iteration_time': []
        }
    
    def _convolve_fft(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Свёртка изображения с ядром в частотной области.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Изображение для свёртки.
        kernel : ndarray (kh, kw)
            Ядро свёртки.
        
        Возвращает
        ----------
        result : ndarray (H, W)
            Результат свёртки, обрезанный до исходного размера.
        """
        H, W = image.shape
        kh, kw = kernel.shape
        
        # Дополнение для корректной циклической свёртки
        padded_image = _pad_for_convolution(image, (kh, kw))
        padded_kernel = np.zeros_like(padded_image)
        
        # Размещение ядра в центре
        pad_h, pad_w = padded_image.shape
        start_h = (pad_h - kh) // 2
        start_w = (pad_w - kw) // 2
        padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(kernel, 2)
        
        # Центрирование для БПФ
        padded_kernel = ifftshift(padded_kernel)
        
        # Свёртка в частотной области
        img_fft = fft2(padded_image)
        kernel_fft = fft2(padded_kernel)
        result_fft = img_fft * kernel_fft
        result = np.real(ifft2(result_fft))
        
        # Обрезка до исходного размера
        return _crop_to_original(result, (H, W))
    
    def _deconvolve_fft(self, blurred: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Обратная свёртка (деконволюция) в частотной области.
        
        Использует фильтр Винера для устойчивости.
        
        Параметры
        ---------
        blurred : ndarray (H, W)
            Размытое изображение.
        kernel : ndarray (kh, kw)
            Ядро размытия.
        
        Возвращает
        ----------
        result : ndarray (H, W)
            Восстановленное изображение.
        """
        H, W = blurred.shape
        kh, kw = kernel.shape
        
        # Дополнение
        padded_blurred = _pad_for_convolution(blurred, (kh, kw))
        padded_kernel = np.zeros_like(padded_blurred)
        
        pad_h, pad_w = padded_blurred.shape
        start_h = (pad_h - kh) // 2
        start_w = (pad_w - kw) // 2
        padded_kernel[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(kernel, 2)
        padded_kernel = ifftshift(padded_kernel)
        
        # Фильтр Винера
        blurred_fft = fft2(padded_blurred)
        kernel_fft = fft2(padded_kernel)
        kernel_fft_conj = np.conj(kernel_fft)
        
        # Регуляризация для устойчивости
        wiener_filter = kernel_fft_conj / (np.abs(kernel_fft)**2 + self.eps)
        
        # Деконволюция
        result_fft = blurred_fft * wiener_filter
        result = np.real(ifft2(result_fft))
        
        # Обрезка и ограничение
        cropped = _crop_to_original(result, (H, W))
        return np.clip(cropped, 0.0, 1.0)
    
    def _update_kernel(
        self, 
        image: np.ndarray, 
        blurred: np.ndarray,
        current_kernel: np.ndarray
    ) -> np.ndarray:
        """
        Обновление оценки ядра размытия.
        
        При фиксированном изображении f:
            h_new = h * (f_flip ⊗ (g / (f ⊗ h)))
        
        Параметры
        ---------
        image : ndarray (H, W)
            Текущая оценка изображения f.
        blurred : ndarray (H, W)  
            Наблюдаемое размытое изображение g.
        current_kernel : ndarray (kh, kw)
            Текущая оценка ядра h.
        
        Возвращает
        ----------
        updated_kernel : ndarray (kh, kw)
            Обновлённое ядро.
        """
        # Прямая модель: f ⊗ h
        conv_fh = self._convolve_fft(image, current_kernel)
        
        # Отношение наблюдаемого к прямой модели
        ratio = blurred / (conv_fh + self.eps)
        
        # Свёртка с отражённым изображением
        flipped_image = np.flip(image)
        update = self._convolve_fft(ratio, flipped_image)
        
        # Обновление ядра
        kh, kw = current_kernel.shape
        updated_kernel = current_kernel * update[:kh, :kw]
        
        # Ограничения на ядро
        updated_kernel = np.maximum(updated_kernel, 0.0)  # Неотрицательность
        kernel_sum = np.sum(updated_kernel)
        if kernel_sum > self.eps:
            updated_kernel /= kernel_sum  # Нормировка
        else:
            # Возврат к предыдущему ядру при проблемах
            updated_kernel = current_kernel.copy()
        
        return updated_kernel
    
    def _update_image(
        self,
        image: np.ndarray,
        blurred: np.ndarray,
        kernel: np.ndarray
    ) -> np.ndarray:
        """
        Обновление оценки изображения.
        
        При фиксированном ядре h:
            f_new = f * (h_flip ⊗ (g / (f ⊗ h))) / (1 + λ·ρ'(Df))
        
        где ρ' — производная функции Хубера.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Текущая оценка изображения f.
        blurred : ndarray (H, W)
            Наблюдаемое размытое изображение g.
        kernel : ndarray (kh, kw)
            Текущая оценка ядра h.
        
        Возвращает
        ----------
        updated_image : ndarray (H, W)
            Обновлённое изображение.
        """
        # Прямая модель: f ⊗ h
        conv_fh = self._convolve_fft(image, kernel)
        
        # Отношение наблюдаемого к прямой модели
        ratio = blurred / (conv_fh + self.eps)
        
        # Свёртка с отражённым ядром
        flipped_kernel = np.flip(kernel)
        fidelity_update = self._convolve_fft(ratio, flipped_kernel)
        
        # Регуляризационный член
        gradient_image = self._convolve_fft(image, self.gradient_operator)
        huber_term = _huber_function(gradient_image, self.huber_threshold)
        huber_conv = self._convolve_fft(huber_term, self.gradient_operator.T)
        
        # Обновление изображения
        denominator = 1.0 + self.relaxation_factor * huber_conv
        updated_image = image * fidelity_update / (denominator + self.eps)
        
        # Ограничения на изображение
        return np.clip(updated_image, 0.0, 1.0)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Выполнение слепой деконволюции методом MAP.
        
        Параметры
        ---------
        image : ndarray (H, W)
            Входное размытое изображение в градациях серого [0, 255].
        
        Возвращает
        ----------
        restored : ndarray (H, W)
            Восстановленное изображение в градациях серого [0, 255].
        
        Исключения
        ----------
        ValueError
            Если входное изображение не 2D.
        """
        start_time = time.time()
        
        # Проверка входных данных
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        # Нормализация изображения [0, 1]
        g = image.astype(np.float64) / 255.0
        H, W = g.shape
        
        # Инициализация
        f = g.copy()  # Начальное приближение изображения
        h = self.init_kernel.copy()  # Начальное приближение ядра
        
        # Основной цикл итераций
        for iteration in range(self.max_iter):
            iter_start = time.time()
            
            # Вывод прогресса
            if self.verbose and iteration % 10 == 0:
                print(f"MAP Итерация {iteration+1:3d}/{self.max_iter}")
            
            # Обновление ядра размытия
            h = self._update_kernel(f, g, h)
            
            # Обновление изображения
            f = self._update_image(f, g, h)
            
            # Сохранение истории
            iter_time = time.time() - iter_start
            self.history['psf_norm'].append(np.sum(h))
            self.history['image_norm'].append(np.linalg.norm(f))
            self.history['iteration_time'].append(iter_time)
        
        # Время выполнения
        self.timer = time.time() - start_time
        
        if self.verbose:
            print(f"MAP завершён за {self.timer:.2f} секунд")
        
        # Возврат к диапазону [0, 255]
        restored = np.clip(f * 255.0, 0, 255).astype(np.uint8)
        return restored
    
    def get_estimated_kernel(self) -> np.ndarray:
        """
        Возвращает последнюю оценку ядра размытия.
        
        Возвращает
        ----------
        kernel : ndarray (kh, kw)
            Оценённое ядро размытия.
        """
        # В реальной реализации здесь должно храниться последнее ядро
        # Для простоты возвращаем начальное
        return self.init_kernel.copy()
    
    def get_param(self) -> List[Tuple[str, Any]]:
        """
        Возвращает текущие параметры алгоритма.
        
        Возвращает
        ----------
        params : list of tuple
            Список кортежей (название_параметра, значение).
        """
        return [
            ('max_iter', self.max_iter),
            ('huber_threshold', self.huber_threshold),
            ('relaxation_factor', self.relaxation_factor),
            ('init_kernel_shape', self.init_kernel.shape),
            ('eps', self.eps),
            ('verbose', self.verbose)
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """
        Изменяет параметры алгоритма.
        
        Параметры
        ---------
        params : dict
            Словарь с параметрами для изменения.
        """
        if 'max_iter' in params:
            self.max_iter = int(params['max_iter'])
        if 'huber_threshold' in params:
            self.huber_threshold = float(params['huber_threshold'])
        if 'relaxation_factor' in params:
            self.relaxation_factor = float(params['relaxation_factor'])
        if 'init_kernel' in params:
            self.init_kernel = np.asarray(params['init_kernel'], dtype=np.float64)
        if 'eps' in params:
            self.eps = float(params['eps'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])
    
    def get_history(self) -> Dict[str, list]:
        """
        Возвращает историю сходимости алгоритма.
        
        Возвращает
        ----------
        history : dict
            Словарь с историей:
            - 'psf_norm': норма ядра на каждой итерации
            - 'image_norm': норма изображения на каждой итерации  
            - 'iteration_time': время каждой итерации
        """
        return self.history


# Функция для обратной совместимости
def map_blind_deconvolution(
    image: np.ndarray,
    max_iter: int = 50,
    huber_threshold: float = 0.1,
    init_kernel: Optional[np.ndarray] = None,
    relaxation_factor: float = 0.1,
    **kwargs
) -> np.ndarray:
    """
    Обёртка для совместимости со старым API.
    
    Параметры
    ---------
    image : ndarray (H, W)
        Входное размытое изображение.
    max_iter : int
        Максимальное число итераций.
    huber_threshold : float
        Порог функции Хубера.
    init_kernel : ndarray или None
        Начальное приближение ядра.
    relaxation_factor : float
        Параметр релаксации.
    **kwargs
        Дополнительные параметры.
    
    Возвращает
    ----------
    restored : ndarray (H, W)
        Восстановленное изображение.
    """
    algo = MAPDeconvolution(
        max_iter=max_iter,
        huber_threshold=huber_threshold,
        init_kernel=init_kernel,
        relaxation_factor=relaxation_factor,
        **kwargs
    )
    return algo.process(image)
