"""
Фильтры добавления шума к изображениям.

Автор: Юров П.И.
"""

import numpy as np
import random
from typing import Tuple
from random import sample
from .base import FilterBase
from scipy.signal import lfilter, butter, sosfilt

from .colored_noise import powerlaw_psd_gaussian, pink_noise_2d


class GaussianNoise(FilterBase):
    """
    Фильтр аддитивного гауссовского шума.
    
    Добавляет нормально распределенный шум с заданным стандартным отклонением.
    
    Атрибуты
    --------
    param : float
        Стандартное отклонение гауссовского шума.
    """

    def __init__(self, param: float) -> None:
        """
        Инициализация фильтра гауссовского шума.
        
        Параметры
        ---------
        param : float
            Стандартное отклонение шума (должно быть положительным).
        """
        if param <= 0:
            raise ValueError("Стандартное отклонение должно быть положительным")
        super().__init__(param, 'noise')
        self.param = param

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|gaussiannoise_{self.param}"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение гауссовского шума к изображению.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение (любой тип, будет преобразовано в float32).
            
        Возвращает
        ----------
        np.ndarray
            Зашумленное изображение (той же формы и типа, что и входное).
        """ 

        noise = np.random.normal(0, self.param, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0.0, 255.0).astype(image.dtype)


class PoissonNoise(FilterBase):
    """
    Фильтр пуассоновского шума (шума дробления).
    
    Имитирует шум подсчета фотонов с пуассоновской статистикой.
    
    Атрибуты
    --------
    param : float
        Интенсивность шума (от 0.0 до 1.0).
    """
    
    def __init__(self, param: float) -> None:
        """
        Инициализация фильтра пуассоновского шума.
        
        Параметры
        ---------
        param : float
            Интенсивность шума (от 0.0 до 1.0).
        """
        super().__init__(param, 'noise')
        self.param = param

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|poissonnoise_{self.param}"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение пуассоновского шума к изображению.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение (любой тип).
            
        Возвращает
        ----------
        np.ndarray
            Зашумленное изображение (той же формы и типа, что и входное).
        """
        noisy = image + np.sqrt(image) * np.random.normal(0, 1, image.shape) * self.param
        return np.clip(noisy, 0, 255).astype(image.dtype)


class SaltAndPepperNoise(FilterBase):
    """
    Фильтр импульсного шума (типа "соль-перец").
    
    Добавляет случайные белые (соль) и черные (перец) пиксели к изображению.
    
    Атрибуты
    --------
    white_pixel : float
        Относительная интенсивность белых пикселей (соль).
    black_pixel : float
        Относительная интенсивность черных пикселей (перец).
    noise_amount : int
        Максимальное количество зашумляемых пикселей.
    """
    
    def __init__(self, param: Tuple[float, float, float]) -> None:
        """
        Инициализация фильтра шума "соль-перец".
        
        Параметры
        ---------
        param : Tuple[float, float, float]
            Кортеж, содержащий:
            - white_pixel: Относительное количество белых пикселей (>=0).
            - black_pixel: Относительное количество черных пикселей (>=0).
            - noise_amount: Максимальное число изменяемых пикселей (>=0).
        """
        super().__init__(param, 'noise')
        
        self.white_pixel = param[0] 
        self.black_pixel = param[1]  
        self.noise_amount = param[2]

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|saltandpappernoise_{self.param}"

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение шума "соль-перец" к входному изображению.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение (в градациях серого или цветное).
            
        Возвращает
        ----------
        np.ndarray
            Изображение с добавленным шумом "соль-перец".
        """
        
        noisy = image.copy()
        h, w = image.shape[:2]
        total_pixels = h * w

        white_count = self.param[0]
        black_count = self.param[1]
        max_noise = self.param[2]
        
        if white_count + black_count <= 0 or max_noise <= 0:
            return noisy
        
        total = white_count + black_count
        white_prob = white_count / total
        black_prob = black_count / total
        
        num_pixels = min(max_noise, total_pixels)
        indices = sample(range(total_pixels), num_pixels)
        
        noise_values = [255, 0]
        
        for idx in indices:
            selected_value = random.choices(noise_values,
                                            weights=[white_prob, black_prob],
                                            k=1)[0]
            if len(image.shape) == 3: 
                noisy[idx // w, idx % w, :] = [selected_value] * 3
            else: 
                noisy[idx // w, idx % w] = selected_value

        return noisy


class OldPhotoNoise(FilterBase):
    """
    Коричневый шум для имитации старых фотографий.

    Атрибуты
    --------
    strength : int
        Сила шума (0-255).
    f3dB : float
        Частота среза экспоненциального фильтра (0-0.5).
    fs : float
        Псевдо-частота дискретизации.
    apply_highpass : bool
        Применять ли high-pass фильтр.
    highpass_cutoff : float
        Cutoff частота для high-pass фильтра.
    """

    def __init__(self, 
                 strength: int = 30, 
                 f3dB: float = 0.05, 
                 fs: float = 1.0, 
                 apply_highpass: bool = True, 
                 highpass_cutoff: float = 0.01) -> None:
        """
        Инициализация.

        Параметры
        ---------
        strength : int
            Сила шума (0-255).
        f3dB : float
            Частота среза экспоненциального фильтра (0-0.5).
        fs : float
            Псевдо-частота дискретизации.
        apply_highpass : bool
            Применять ли high-pass фильтр.
        highpass_cutoff : float
            Cutoff частота для high-pass фильтра.
        """
        self.strength = strength
        self.f3dB = f3dB
        self.fs = fs
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff
        super().__init__(1, 'noise')

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|oldphotonoise_{self.strength}"

    def find_alpha(self, 
                   Fs: float, 
                   f3dB: float) -> float:
        """Вычислить альфу для экспоненциального фильтра."""
        return (
            np.sqrt(
                np.power(np.cos(2 * np.pi * f3dB / Fs), 2)
                - 4 * np.cos(2 * np.pi * f3dB / Fs)
                + 3
            )
            + np.cos(2 * np.pi * f3dB / Fs)
            - 1
        )

    def generate_2d_brownian_noise(self, 
                                   shape: Tuple[int, int], 
                                   alpha: float) -> np.ndarray:
        """Создает 2D коричневый шум с фильтрацией по строкам и столбцам."""
        x = np.random.normal(0, 1, shape)
        b = [alpha]
        a = [1, -(1 - alpha)]
        for i in range(shape[0]):
            x[i, :] = lfilter(b, a, x[i, :])
        for j in range(shape[1]):
            x[:, j] = lfilter(b, a, x[:, j])
        
        return x

    def high_pass_filter_2d(self, 
                            img: np.ndarray, 
                            fs: float, 
                            cutoff: float = 0.01) -> np.ndarray:
        """Применяет 2D high-pass фильтр через Butterworth."""
        sos = butter(2, cutoff, btype='highpass', fs=fs, output='sos')
        filtered = np.copy(img)
        for i in range(img.shape[0]):
            filtered[i, :] = sosfilt(sos, img[i, :])
        for j in range(img.shape[1]):
            filtered[:, j] = sosfilt(sos, filtered[:, j])
        return filtered

    def filter(self, img):
        """Применяет 2D Brownian шум к изображению."""
        img = np.array(img, dtype=np.float32)

        alpha = self.find_alpha(self.fs, self.f3dB)
        brown_noise = self.generate_2d_brownian_noise(img.shape, alpha)

        if self.apply_highpass:
            brown_noise = self.high_pass_filter_2d(brown_noise, 
                                                   self.fs, 
                                                   cutoff=self.highpass_cutoff)

        brown_noise -= brown_noise.min()
        brown_noise /= (brown_noise.max() - brown_noise.min())
        brown_noise = 2 * brown_noise - 1

        noisy_img = img + self.strength * brown_noise

        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img

    
class ColoredNoise(FilterBase):
    """
    Цветной шум (1/f)^beta.

    Атрибуты
    --------
    noise_level : float
        Уровень шума (0.0 - без шума, 1.0 - сильный шум).
    beta : float
        Параметр спектрального наклона (1.0 = розовый шум, 2.0 = коричневый).
    """

    def __init__(self, 
                 noise_level: float = 0.2, 
                 beta: float = 1.0) -> None:
        """
        Инициализация.

        Параметры
        ---------
        noise_level : float
            Уровень шума (0.0 - без шума, 1.0 - сильный шум).
        beta : float
            Параметр спектрального наклона (1.0 = розовый, 2.0 = коричневый).
        """
        self.noise_level = noise_level
        self.beta = beta
        super().__init__(1, 'noise')

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|colorednoise_{self.beta}_{self.noise_level}"

    def filter(self, img):
        """Применяет цветной шум к изображению."""
        assert img.ndim == 2, "Ожидается двумерное изображение (grayscale)."
        h, w = img.shape
        pink_noise = powerlaw_psd_gaussian(exponent=self.beta, size=(h, w))

        pink_noise = pink_noise / np.std(pink_noise)
        pink_noise = pink_noise * self.noise_level * 255.0

        noisy_img = img.astype(np.float32) + pink_noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img


class Pink_Noise(FilterBase):
    """
    Розовый шум (1/f).

    Атрибуты
    --------
    noise_level : float
        Уровень шума (0.0 - без шума, 1.0 - сильный шум).
    """

    def __init__(self, noise_level: float = 0.2) -> None:
        """
        Инициализация.

        Параметры
        ---------
        noise_level : float
            Уровень шума (0.0 - без шума, 1.0 - сильный шум).
        """
        self.noise_level = noise_level
        super().__init__(1, 'noise')

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|pinknoise_{self.noise_level}"

    def filter(self, img):
        """Применяет розовый шум к изображению."""
        noise = pink_noise_2d(img.shape, 2) * self.noise_level
        res = np.clip(img + noise, 0.0, 255.0).astype(np.int16)
        return res
        
        
class Brown_Noise(FilterBase):
    """
    Коричневый шум (1/f)^2.

    Атрибуты
    --------
    noise_level : float
        Уровень шума (0.0 - без шума, 1.0 - сильный шум).
    """

    def __init__(self, noise_level: float = 0.2) -> None:
        """
        Инициализация.

        Параметры
        ---------
        noise_level : float
            Уровень шума (0.0 - без шума, 1.0 - сильный шум).
        """
        self.noise_level = noise_level
        super().__init__(1, 'noise')

    def description(self) -> str:
        """Выдает название шума в файловой системе с параметром."""
        return f"|brownnoise_{self.noise_level}"

    def filter(self, img):
        """Применяет коричневый шум к изображению."""
        noise = pink_noise_2d(img.shape, 4) * self.noise_level
        res = np.clip(img + noise, 0.0, 255.0).astype(np.int16)
        return res
        