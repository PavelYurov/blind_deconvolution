import cv2 as cv
import numpy as np
from typing import Callable, Optional
from .base import FilterBase

class DefocusBlur(FilterBase):
    """
    Фильтр размытия вне фокуса, имитирующий эффект расфокусировки камеры.
    
    Создает 2D размытие в форме колокола с использованием настраиваемой PSF-функции.
    
    Атрибуты:
        psf (Callable): Функция распределения точки (PSF), генерирующая ядро размытия
        param (float): Параметр интенсивности размытия
        kernel_size (Optional[int]): Фиксированный размер ядра размытия (None для автоопределения)
    """

    def __init__(self, 
                 psf: Callable[[np.ndarray, float], np.ndarray],
                 param: float = 5.0, 
                 kernel_size: Optional[int] = None) -> None:
        """
        Инициализация фильтра размытия вне фокуса.
        
        Аргументы:
            psf: Функция, принимающая (radius, param) и возвращающая значения ядра
            param: Параметр контроля интенсивности размытия
            kernel_size: Опциональный фиксированный размер ядра (должен быть нечетным если указан)
        """
        self.psf = psf
        self.param = param
        self.kernel_size = kernel_size
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|defocus_{self.psf.__name__}_{self.param}_{self.kernel_size}"

    def generate_kernel(self) -> np.ndarray:
        """Генерация ядра размытия."""
        size = self.kernel_size or self._calculate_kernel_size()
        y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
        radius = np.sqrt(x**2 + y**2)

        kernel = self.psf(radius, self.param)
        kernel[kernel < np.finfo(float).eps] = 0
        return kernel / kernel.sum()

    def _calculate_kernel_size(self) -> int:
        """Вычисление оптимального размера ядра на основе параметра размытия."""
        return int(6 * self.param) | 1 

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение размытия вне фокуса к изображению.
        
        Аргументы:
            image: Входное изображение для размытия
            
        Возвращает:
            Размытое изображение
        """
        return cv.filter2D(image, -1, self.generate_kernel())
    
class MotionBlur(FilterBase):
    """
    Фильтр размытия в движении, имитирующий линейное движение камеры.
    
    Создает одномерное направленное размытие, которое можно повернуть на любой угол.
    
    Атрибуты:
        psf (Callable): PSF-функция для одномерного движения
        param (float): Параметр контроля длины/интенсивности размытия
        angle (float): Направление движения в градусах (0 = горизонтальное)
        kernel_length (Optional[int]): Фиксированная длина ядра размытия
    """
    
    def __init__(self, 
                 psf: Callable[[np.ndarray, float], np.ndarray],
                 param: float = 1.0,
                 angle: float = 0,
                 kernel_length: Optional[int] = None) -> None:
        """
        Инициализация фильтра размытия в движении.
        
        Аргументы:
            psf: Функция (x, param) -> значения ядра, где x - 1D массив координат
            param: Параметр для PSF-функции
            angle: Угол направления размытия (в градусах)
            kernel_length: Длина размытия (нечетное число)
        """
        self.psf = psf
        self.param = param
        self.angle = angle
        self.kernel_length = kernel_length
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|motion_{self.psf.__name__}_{self.param}_{self.angle}_{self.kernel_length}"

    def generate_kernel(self) -> np.ndarray:
        """Генерация ядра размытия в движении"""
        length = self.kernel_length or self._calculate_kernel_length()
        if length % 2 == 0:
            raise ValueError("Длина ядра должна быть нечетной")

        x_coords = np.linspace(-length//2, length//2, length)
        psf_1d = self.psf(x_coords, self.param)
        psf_1d = psf_1d / psf_1d.sum()
        
        kernel = np.zeros((length, length))
        center = length // 2
        kernel[center, :] = psf_1d

        rotation_matrix = cv.getRotationMatrix2D((center, center), self.angle, 1)
        rotated_kernel = cv.warpAffine(kernel, rotation_matrix, (length, length))
        
        rotated_kernel[rotated_kernel < np.finfo(float).eps] = 0
        return rotated_kernel / rotated_kernel.sum()

    def _calculate_kernel_length(self) -> int:
        """Вычисление длины ядра на основе параметра размытия"""
        return max(int(4 * self.param) | 1, 3) 

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение размытия в движении к изображению."""
        return cv.filter2D(image, -1, self.generate_kernel())