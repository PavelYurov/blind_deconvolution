"""
Различные методы шумоподавления

Содержит:
    - Сглаживание по гауссу (расфокус, как денойзинг)
    - Усредняющее сглаживание 
    - Метод анизотропной диффузии
    - Метод Non-Local Means (NLM)
    - Метод медианной фильтрации
    - Вейвлет преобразование
    - Метод билатерального преобразования
    - Метод Total Variation (две его реализации)

Автор: Юров П.И. Беззаборов А.А.
"""
import numpy as np
import random
from typing import Callable, Optional, Tuple, Union
from random import sample
from .base import FilterBase
from scipy.signal import lfilter, butter, sosfilt
from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv

from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet, denoise_bilateral, denoise_tv_bregman, denoise_tv_chambolle
from scipy.ndimage import median_filter


class LinearSmoothing(FilterBase):
    """
    Метод гауссова размытия для уменьшения шума.

    Аргументы:
        param (int): Размер ядра размытия
    """

    def __init__(self, param: int = 3) -> None:
        """
        Инициализация фильтра размытия.

        Аргументы:
            param (int): Размер ядра размытия
        """
        self.param = param
        super().__init__(param, 'blur') #blur - чтобы влиял на ядро

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|linearsmoothingdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет размытие к изображению.""" 
        denoised_image = cv.GaussianBlur(image, (self.param , self.param) ,0)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    

class AnisotropicDiffusion(FilterBase):
    """
    Метод анизотропной диффузии для уменьшения шума на изображении.

    Аргументы:
        niter (int): Количество итераций диффузии
        kappa (int): Коэффициент проводимости
        gamma (float): Коэффициент скорости диффузии
        option (int): Параметр формулы диффузии
    """

    def __init__(self, 
                 niter: int = 20, 
                 kappa: int = 50, 
                 gamma: float = 0.1, 
                 option: int = 2) -> None:
        """
        Инициализация метода анизотропной диффузии.

        Аргументы:
            niter (int): Количество итераций диффузии
            kappa (int): Коэффициент проводимости
            gamma (float): Коэффициент скорости диффузии
            option (int): Параметр формулы диффузии: \
                1 и 2 Диффузия Перона-Малика; \
                3 Бивесовая функция Тьюки
        """
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.option = option
        super().__init__(0, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|anisotropicdiffusiondenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применения метода к изображнию.""" 
        denoised_image = anisotropic_diffusion(image, 
                                               niter=self.niter, 
                                               kappa=self.kappa, 
                                               gamma=self.gamma, 
                                               option=self.option)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class NonLocalMeans(FilterBase):
    """
    Метод Non-Local Means (NLM) уменьшения шума на изображении.

    Аргументы:
        fast (bool): Если True - будет применена быстрая версия алгоритма, иначе - обычная
        sigma_coef (float): Параметр стандартного отклоненния Гауссова шума
        patch_size (int): Размер фрагмента подобия
        patch_distance (int): Максимальное расстояние, где нужно искать фрагменты подобия
    """

    def __init__(self, 
                 fast: bool = False, 
                 sigma_coef: float = 0.01, 
                 patch_size: int = 5, 
                 patch_distance: int = 6) -> None:
        """
        Инициализация метода Non-Local Means (NLM).

        Аргументы:
            fast (bool): Если True - будет применена быстрая версия алгоритма, иначе - обычная
            sigma_coef (float): Параметр стандартного отклоненния Гауссова шума
            patch_size (int): Размер фрагмента подобия
            patch_distance (int): Максимальное расстояние, где нужно искать фрагменты подобия
        """
        self.sigma_coef = sigma_coef
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.fast = fast
        super().__init__(0, 'denoise')

    def discription(self) -> str:
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
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    

class MedianFilter(FilterBase):
    """
    Метод медианной фильтрации для уменьшения шума на изображении.

    Аргументы:
        param (Tuple[int,int] or int): Размер следа
    """

    def __init__(self, param=3) -> None:
        """
        Инициализация медианного фильтра.

        Аргументы:
            param (Tuple[int,int] or int): Размер следа
        """
        super().__init__(param, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|mediandenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применяет медианный фильтр к изображению.""" 
        denoised_image = median_filter(image, size=self.param)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class Wavelet(FilterBase):
    """
    Метод уменьшения шума на изображении с помощью вейвлетов.

    Аргументы:
        method (str): Метод порога: 'BayesShrink' или 'VisuShrink'
        mode (str): Тип порога: 'soft' или 'hard'
        rescale_sigma (bool): Если False, то не будет переопределять стандартное отклонение
        wavelet (str): Тип вейвлета: 'db2', 'haar', 'sym9'
        sigma (Optional[float]): Стандартное отклонение
    """

    def __init__(self, 
                 method: str = 'BayesShrink', 
                 mode: str = 'soft', 
                 wavelet: str = 'db1', 
                 sigma : Optional[float] = None, 
                 rescale_sigma: bool = True) -> None:
        """
        Метод уменьшения шума на изображении с помощью вейвлетов.

        Аргументы:
            method (str): Метод порога: 'BayesShrink' или 'VisuShrink'
            mode (str): Тип порога: 'soft' или 'hard'
            rescale_sigma (bool): Если False, то не будет переопределять стандартное отклонение
            wavelet (str): Тип вейвлета: 'db2', 'haar', 'sym9'
            sigma (float): Стандартное отклонение
        """
        self.method = method
        self.mode = mode
        self.rescale_sigma = rescale_sigma
        self.wavelet = wavelet
        self.sigma = sigma
        super().__init__(0, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|waveletdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение вейвелетов для уменьшения шума.""" 
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
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class BilateralFilter_sk(FilterBase):
    """
    Метод билатерального преобразования для уменьшения шума на изображении.

    Аргументы:
        win_size (int): Размер окна фильтра
        sigma_color (Optional[float]): стандартное отклонение для цветов
        sigma_spatial (float): стандартное отклонение для расстояния
        bins (int): Число дискретных значений для весов Гауссиана
        mode (str): Тип пэддинга: ‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’
        cval (float): Значения за границей изображения, если тип пэддинга ‘constant’
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

        Аргументы:
            win_size (int): Размер окна фильтра
            sigma_color (float): стандартное отклонение для цветов
            sigma_spatial (float): стандартное отклонение для расстояния
            bins (int): Число дискретных значений для весов Гауссиана
            mode (str): Тип пэддинга: ‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’
            cval (float): Значения за границей изображения, если тип пэддинга ‘constant’
        """
        self.win_size = win_size
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial
        self.bins = bins
        self.mode = mode
        self.cval = cval
        super().__init__(0, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|bilateralfilter_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение билатерального фильтра к изображению.""" 
        denoised_image =  denoise_bilateral(image,
                                            win_size=self.win_size,
                                            sigma_color=self.sigma_color,
                                            sigma_spatial=self.sigma_spatial,
                                            bins=self.bins,
                                            mode=self.mode,
                                            cval=self.cval)
        denoised_image *= 255.0
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class TV_bregman(FilterBase):
    """
    Метод Total Variation с оптимизацией Брегмана.

    Аргументы:
        weight (Optional[float]): Вес денойзинга
        eps (Optional[float]): Допуск невязки до остановки
        max_num_iter (Optional[int]): Максимальное число итераций
        isotropic (Optional[bool]): Переключение между изотропическим и анизотропическим TV уменьшением шума
    """

    def __init__(self,
                 weight: Optional[float] = 5.0, 
                 max_num_iter: Optional[int] = 100, 
                 eps: Optional[float] = 0.001, 
                 isotropic: Optional[bool] = True) -> None:
        """
        Инициализация метода Total Variation.

        Аргументы:
            weight (Optional[float]): Вес денойзинга
            eps (Optional[float]): Допуск невязки до остановки
            max_num_iter (Optional[int]): Максимальное число итераций
            isotropic (Optional[bool]): Переключение между изотропическим и анизотропическим TV уменьшением шума
        """
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.eps = eps
        self.isotropic = isotropic
        super().__init__(0, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|TV1_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода TV к изображению.""" 
        denoised_image =  denoise_tv_bregman(image,weight=self.weight,
                                            max_num_iter=self.max_num_iter,
                                            eps=self.eps,
                                            isotropic=self.isotropic)
        denoised_image *= 255.0
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class TV_Chambolle(FilterBase):
    """
    Метод Total Variation реализованный алгоритмом Шамболле.

    Аргументы:
        weight (Optional[float]): Вес денойзинга
        eps (Optional[float]): Критерий остановки, порог невязки
        max_num_iter (Optional[int]): Максимальное число итераций
    """

    def __init__(self,
                 weight: Optional[float] = 5.0, 
                 max_num_iter: Optional[int] = 100, 
                 eps: Optional[float] = 0.001):
        """
        Инициализация метода Total Variation.

        Аргументы:
            weight (Optional[float]): Вес денойзинга
            eps (Optional[float]): Критерий остановки, порог невязки
            max_num_iter (Optional[int]): Максимальное число итераций
        """
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.eps = eps
        super().__init__(0, 'denoise')

    def discription(self) -> str:
        """Возвращает название способа уменьшения шума."""
        return f"|TV2_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение метода TV к изображению.""" 
        denoised_image =  denoise_tv_chambolle(image, 
                                               weight=self.weight, 
                                               max_num_iter=self.max_num_iter,
                                               eps=self.eps)
        denoised_image *= 255
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class MeanBlur(FilterBase):
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


class BilateralFilter_cv(FilterBase):
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





