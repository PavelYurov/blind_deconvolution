"""
Фильтры шумоподавления для изображений.

Автор: Юров П.И.
"""

import numpy as np
from typing import Optional, Tuple
from .base import FilterBase
from scipy.signal import lfilter, butter, sosfilt
from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv

from .colored_noise import powerlaw_psd_gaussian, pink_noise_2d
from skimage.restoration import (
    denoise_nl_means, estimate_sigma, denoise_wavelet, 
    denoise_bilateral, denoise_tv_bregman, denoise_tv_chambolle
)
from scipy.ndimage import median_filter


class LinearSmoothing(FilterBase):
    """
    Метод гауссова размытия для уменьшения шума.

    Атрибуты
    --------
    param : int
        Размер ядра размытия.
    """

    def __init__(self, param: int = 3) -> None:
        """
        Инициализация фильтра размытия.

        Параметры
        ---------
        param : int
            Размер ядра размытия.
        """
        self.param = param
        super().__init__(param, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|linearsmoothingdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет размытие к изображению.""" 
        denoised_image = cv.GaussianBlur(image, (self.param, self.param), 0)
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)
    

class AnisotropicDiffusion(FilterBase):
    """
    Метод анизотропной диффузии для уменьшения шума на изображении.

    Атрибуты
    --------
    niter : int
        Количество итераций диффузии.
    kappa : int
        Коэффициент проводимости.
    gamma : float
        Коэффициент скорости диффузии.
    option : int
        Параметр формулы диффузии.
    """

    def __init__(self, 
                 niter: int = 20, 
                 kappa: int = 50, 
                 gamma: float = 0.1, 
                 option: int = 2) -> None:
        """
        Инициализация метода анизотропной диффузии.

        Параметры
        ---------
        niter : int
            Количество итераций диффузии.
        kappa : int
            Коэффициент проводимости.
        gamma : float
            Коэффициент скорости диффузии.
        option : int
            Параметр формулы диффузии:
            1 и 2 - Диффузия Перона-Малика;
            3 - Бивесовая функция Тьюки.
        """
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.option = option
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|anisotropicdiffusiondenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода к изображению.""" 
        denoised_image = anisotropic_diffusion(image, 
                                               niter=self.niter, 
                                               kappa=self.kappa, 
                                               gamma=self.gamma, 
                                               option=self.option)
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class NonLocalMeans(FilterBase):
    """
    Метод Non-Local Means (NLM) уменьшения шума на изображении.

    Атрибуты
    --------
    fast : bool
        Если True - будет применена быстрая версия алгоритма.
    sigma_coef : float
        Параметр стандартного отклонения Гауссова шума.
    patch_size : int
        Размер фрагмента подобия.
    patch_distance : int
        Максимальное расстояние поиска фрагментов подобия.
    """

    def __init__(self, 
                 fast: bool = False, 
                 sigma_coef: float = 0.01, 
                 patch_size: int = 5, 
                 patch_distance: int = 6) -> None:
        """
        Инициализация метода Non-Local Means (NLM).

        Параметры
        ---------
        fast : bool
            Если True - будет применена быстрая версия алгоритма.
        sigma_coef : float
            Параметр стандартного отклонения Гауссова шума.
        patch_size : int
            Размер фрагмента подобия.
        patch_distance : int
            Максимальное расстояние поиска фрагментов подобия.
        """
        self.sigma_coef = sigma_coef
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.fast = fast
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|nonlocalmeansdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода NLM к изображению."""
        sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))

        denoised_image = denoise_nl_means(image,
                                          patch_size=self.patch_size,
                                          patch_distance=self.patch_distance, 
                                          h=self.sigma_coef * sigma_est, 
                                          fast_mode=self.fast)
        denoised_image *= 255.0
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)
    

class MedianFilter(FilterBase):
    """
    Метод медианной фильтрации для уменьшения шума на изображении.

    Атрибуты
    --------
    param : int
        Размер следа.
    """

    def __init__(self, param=3) -> None:
        """
        Инициализация медианного фильтра.

        Параметры
        ---------
        param : int
            Размер следа.
        """
        super().__init__(param, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|mediandenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет медианный фильтр к изображению.""" 
        denoised_image = median_filter(image, size=self.param)
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class Wavelet(FilterBase):
    """
    Метод уменьшения шума на изображении с помощью вейвлетов.

    Атрибуты
    --------
    method : str
        Метод порога: 'BayesShrink' или 'VisuShrink'.
    mode : str
        Тип порога: 'soft' или 'hard'.
    rescale_sigma : bool
        Переопределять ли стандартное отклонение.
    wavelet : str
        Тип вейвлета: 'db2', 'haar', 'sym9'.
    sigma : Optional[float]
        Стандартное отклонение.
    """

    def __init__(self, 
                 method: str = 'BayesShrink', 
                 mode: str = 'soft', 
                 wavelet: str = 'db1', 
                 sigma: Optional[float] = None, 
                 rescale_sigma: bool = True) -> None:
        """
        Инициализация вейвлет-фильтра.

        Параметры
        ---------
        method : str
            Метод порога: 'BayesShrink' или 'VisuShrink'.
        mode : str
            Тип порога: 'soft' или 'hard'.
        wavelet : str
            Тип вейвлета: 'db2', 'haar', 'sym9'.
        sigma : Optional[float]
            Стандартное отклонение.
        rescale_sigma : bool
            Переопределять ли стандартное отклонение.
        """
        self.method = method
        self.mode = mode
        self.rescale_sigma = rescale_sigma
        self.wavelet = wavelet
        self.sigma = sigma
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|waveletdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение вейвлетов для уменьшения шума.""" 
        sigma_est = self.sigma if self.sigma else estimate_sigma(image, 
                                                                 channel_axis=-1, 
                                                                 average_sigmas=True)
        denoised_image = denoise_wavelet(image,
                                         method=self.method,
                                         mode=self.mode,
                                         rescale_sigma=self.rescale_sigma, 
                                         wavelet=self.wavelet,
                                         sigma=sigma_est)
        denoised_image *= 255
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class BilateralFilter(FilterBase):
    """
    Метод билатерального преобразования для уменьшения шума на изображении.

    Атрибуты
    --------
    win_size : int
        Размер окна фильтра.
    sigma_color : Optional[float]
        Стандартное отклонение для цветов.
    sigma_spatial : float
        Стандартное отклонение для расстояния.
    bins : int
        Число дискретных значений для весов Гауссиана.
    mode : str
        Тип пэддинга: 'constant', 'edge', 'symmetric', 'reflect', 'wrap'.
    cval : float
        Значения за границей изображения (для 'constant').
    """

    def __init__(self, 
                 win_size: int = 5, 
                 sigma_color: Optional[float] = None, 
                 sigma_spatial: float = 1, 
                 bins: int = 10000, 
                 mode: str = 'constant', 
                 cval: int = 0) -> None:
        """
        Инициализация билатерального фильтра.

        Параметры
        ---------
        win_size : int
            Размер окна фильтра.
        sigma_color : Optional[float]
            Стандартное отклонение для цветов.
        sigma_spatial : float
            Стандартное отклонение для расстояния.
        bins : int
            Число дискретных значений для весов Гауссиана.
        mode : str
            Тип пэддинга: 'constant', 'edge', 'symmetric', 'reflect', 'wrap'.
        cval : int
            Значения за границей изображения (для 'constant').
        """
        self.win_size = win_size
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
        self.bins = bins
        self.mode = mode
        self.cval = cval
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|bilateralfilter_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение билатерального фильтра к изображению.""" 
        denoised_image = denoise_bilateral(image,
                                            win_size=self.win_size,
                                            sigma_color=self.sigma_color,
                                            sigma_spatial=self.sigma_spatial,
                                           bins=self.bins,
                                           mode=self.mode,
                                            cval=self.cval)
        denoised_image *= 255.0
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class TV_bregman(FilterBase):
    """
    Метод Total Variation с оптимизацией Брегмана.

    Атрибуты
    --------
    weight : Optional[float]
        Вес денойзинга.
    eps : Optional[float]
        Допуск невязки до остановки.
    max_num_iter : Optional[int]
        Максимальное число итераций.
    isotropic : Optional[bool]
        Изотропическое или анизотропическое TV уменьшение шума.
    """

    def __init__(self,
                 weight: Optional[float] = 5.0, 
                 max_num_iter: Optional[int] = 100, 
                 eps: Optional[float] = 0.001, 
                 isotropic: Optional[bool] = True) -> None:
        """
        Инициализация метода Total Variation.

        Параметры
        ---------
        weight : Optional[float]
            Вес денойзинга.
        eps : Optional[float]
            Допуск невязки до остановки.
        max_num_iter : Optional[int]
            Максимальное число итераций.
        isotropic : Optional[bool]
            Изотропическое или анизотропическое TV уменьшение шума.
        """
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.eps = eps
        self.isotropic = isotropic
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|TV1_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода TV к изображению.""" 
        denoised_image = denoise_tv_bregman(image,
                                            weight=self.weight,
                                            max_num_iter=self.max_num_iter,
                                            eps=self.eps,
                                            isotropic=self.isotropic)
        denoised_image *= 255.0
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class TV_Chambolle(FilterBase):
    """
    Метод Total Variation реализованный алгоритмом Шамболле.

    Атрибуты
    --------
    weight : Optional[float]
        Вес денойзинга.
    eps : Optional[float]
        Критерий остановки, порог невязки.
    max_num_iter : Optional[int]
        Максимальное число итераций.
    """

    def __init__(self,
                 weight: Optional[float] = 5.0, 
                 max_num_iter: Optional[int] = 100, 
                 eps: Optional[float] = 0.001):
        """
        Инициализация метода Total Variation.

        Параметры
        ---------
        weight : Optional[float]
            Вес денойзинга.
        eps : Optional[float]
            Критерий остановки, порог невязки.
        max_num_iter : Optional[int]
            Максимальное число итераций.
        """
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.eps = eps
        super().__init__(0, 'denoise')

    def description(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|TV2_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода TV к изображению.""" 
        denoised_image = denoise_tv_chambolle(image, 
                                               weight=self.weight, 
                                               max_num_iter=self.max_num_iter,
                                               eps=self.eps)
        denoised_image *= 255
        return np.clip(denoised_image, 0.0, 255.0).astype(image.dtype)


class WienerFilter(FilterBase):
    """
    Фильтр Винера для уменьшения шума (в разработке).

    TODO: Реализовать фильтр Винера.
    """

    def __init__(self) -> None:
        """Инициализация фильтра Винера."""
        raise NotImplementedError("WienerFilter еще не реализован")

    def description(self) -> str:
        """Возвращает название фильтра."""
        return "|wiener_"

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение фильтра к изображению."""
        raise NotImplementedError("WienerFilter еще не реализован")
