
import cv2 as cv
import numpy as np
from typing import Union, Tuple
from .base import FilterBase

"""
    над кодом работал:
    Беззаборов П.И.
"""

class MeanBlur(FilterBase):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    """
    Усредняющий (боксовый) фильтр размытия.
    
    Параметры:
        kernel_size (int): Размер усредняющего ядра (должен быть нечетным и положительным)
    """

    def __init__(self, kernel_size: int) -> None:
        """
        Усредняющий (боксовый) фильтр размытия.
        
        Параметры:
            kernel_size (int): Размер усредняющего ядра (должен быть нечетным и положительным)
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть положительным нечетным числом")
        super().__init__(kernel_size, 'blur')
        self.kernel_size = kernel_size

    def discription(self) -> str:
        """Выдает название смаза с параметром."""
        return f"|meanblur_{self.kernel_size}"

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применить усредняющее размытие к изображению.
        
        Аргументы:
            img: Входное изображение (в градациях серого или цветное)
            
        Возвращает:
            Размытое изображение
        """
        return cv.blur(image, (self.kernel_size, self.kernel_size))
    

class MedianBlur(FilterBase):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    """
    Медианный фильтр (эффективен против шума "соль-перец").
    
    Параметры:
        kernel_size (int): Размер медианного ядра (должен быть нечетным и >=3)
    """

    def __init__(self, kernel_size: int) -> None:
        """
        Медианный фильтр (эффективен против шума "соль-перец").
        
        Параметры:
            kernel_size (int): Размер медианного ядра (должен быть нечетным и >=3)
        """
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным числом >=3")
        super().__init__(kernel_size, 'blur')
        self.kernel_size = kernel_size

    def discription(self) -> str:
        """Выдает название смаза с параметром."""
        return f"|medianblur_{self.kernel_size}"

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применить медианное размытие к изображению."""
        return cv.medianBlur(image, self.kernel_size)


class GaussianBlur(FilterBase):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    """
    Гауссовский фильтр размытия.
    
    Параметры:
        kernel_size (int): Размер гауссовского ядра (должен быть нечетным и положительным)
        std (float): Стандартное отклонение (0 для автоматического расчета)
    """

    def __init__(self, params: Union[int, Tuple[int, float]]) -> None:
        """
        Гауссовский фильтр размытия.
        
        Параметры:
            kernel_size (int): Размер гауссовского ядра (должен быть нечетным и положительным)
            std (float): Стандартное отклонение (0 для автоматического расчета)
        """
        if isinstance(params, int):
            kernel_size = params
            std = 0
        else:
            kernel_size, std = params
            
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Размер ядра должен быть положительным нечетным числом")
        super().__init__(params, 'blur')
        self.kernel_size = kernel_size
        self.std = std

    def discription(self) -> str:
        """Выдает название смаза с параметром."""
        return f"|gaussianblur_{self.kernel_size}_{self.std}"

    def filter(self, img: np.ndarray) -> np.ndarray:
        """Применить гауссовское размытие к изображению."""
        return cv.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.std)


class BilateralFilter(FilterBase):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    """
    Билатеральный фильтр (с сохранением границ).
    
    Параметры:
        d (int): Диаметр окрестности пикселя
        sigma_color (float): Сигма фильтр в цветовом пространстве
        sigma_space (float): Сигма фильтр в координатном пространстве
    """

    def __init__(self, params: Union[int, Tuple[int, float, float]]):
        """
        Билатеральный фильтр (с сохранением границ).
        
        Параметры:
            d (int): Диаметр окрестности пикселя
            sigma_color (float): Сигма фильтр в цветовом пространстве
            sigma_space (float): Сигма фильтр в координатном пространстве
        """
        if isinstance(params, int):
            d = params
            sigma_color = sigma_space = 75
        else:
            d, sigma_color, sigma_space = params
            
        super().__init__(params, 'blur')
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def discription(self) -> str:
        """Выдает название смаза с параметром."""
        return f"|bilateralfilter_{self.d}_{self.sigma_color}_{self.sigma_space}"

    def filter(self, img: np.ndarray) -> np.ndarray:
        """Применение билатерального фильтра к изображению."""
        return cv.bilateralFilter(img, self.d, self.sigma_color, self.sigma_space)