"""
Пакет фильтров для генерации искажений изображений.

Модули:
    base: Базовый класс FilterBase
    blur: Фильтры размытия (Defocus, Motion blur)
    noise: Фильтры шума (Gaussian, Poisson, Salt & Pepper)
    denoise: Методы уменьшения шума
    smooth: Сглаживающие фильтры
    distributions: Функции распределения для ядер
    colored_noise: Генераторы цветного шума

Авторы: Юров П.И., Беззаборов А.А.
"""

from .base import FilterBase
from .blur import DefocusBlur, MotionBlur, BSpline_blur, Kernel_convolution
from .noise import (
    GaussianNoise,
    PoissonNoise,
    SaltAndPepperNoise,
    OldPhotoNoise,
    ColoredNoise,
    Pink_Noise,
    Brown_Noise,
)
from .smooth import MeanBlur, MedianBlur, GaussianBlur, BilateralFilter

__all__ = [
    'FilterBase',
    'DefocusBlur',
    'MotionBlur',
    'BSpline_blur',
    'Kernel_convolution',
    'Identical_kernel',
    'GaussianNoise',
    'PoissonNoise',
    'SaltAndPepperNoise',
    'OldPhotoNoise',
    'ColoredNoise',
    'Pink_Noise',
    'Brown_Noise',
    'MeanBlur',
    'MedianBlur',
    'GaussianBlur',
    'BilateralFilter',
]

