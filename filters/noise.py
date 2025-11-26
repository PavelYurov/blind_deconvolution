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
    
    Атрибуты:
        param (float): Стандартное отклонение гауссовского шума
    """
    def __init__(self, param):
        """
        Инициализация фильтра гауссовского шума.
        
        Аргументы:
            param: Стандартное отклонение шума (должно быть положительным)
        """
        if param <= 0:
            raise ValueError("Стандартное отклонение должно быть положительным")
        super().__init__(param, 'noise')
        self.param = param

    def discription(self) -> str:
        return f"|gaussiannoise_{self.param}"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение гауссовского шума к изображению.
        
        Аргументы:
            image: Входное изображение (любой тип, будет преобразовано в float32)
            
        Возвращает:
            Зашумленное изображение (той же формы и типа, что и входное)
        """ 

        noise = np.random.normal(0, self.param, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy,0.0,255.0).astype(image.dtype)


class PoissonNoise(FilterBase):
    """
    Фильтр пуассоновского шума (шума дробления).
    
    Имитирует шум подсчета фотонов с пуассоновской статистикой.
    
    Атрибуты:
        param (float): Интенсивность шума (от 0.0 до 1.0)
    """
    
    def __init__(self, param: float) -> None:
        """
        Инициализация фильтра пуассоновского шума.
        
        Аргументы:
            param: Интенсивность шума (от 0.0 до 1.0)
        """
        # if param <= 0 or param > 1.0:
        #     raise ValueError("Интенсивность должна быть в диапазоне (0.0, 1.0]")
        super().__init__(param, 'noise')
        self.param = param

    def discription(self) -> str:
        return f"|poissonnoise_{self.param}"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение пуассоновского шума к изображению.
        
        Аргументы:
            image: Входное изображение (любой тип)
            
        Возвращает:
            Зашумленное изображение (той же формы и типа, что и входное)
        """
        # noisy = image + np.round(np.random.poisson(image * self.param))
        noisy = image + np.sqrt(image) * np.random.normal(0, 1, image.shape) * self.param
        return np.clip(noisy, 0, 255).astype(image.dtype)
    
class SaltAndPepperNoise(FilterBase):
    """
    Фильтр импульсного шума (типа "соль-перец").
    
    Добавляет случайные белые (соль) и черные (перец) пиксели к изображению.
    
    Параметры:
        param (Tuple[float, float, float]): 
            Кортеж из трех элементов:
            - white_pixel: Относительная интенсивность белых пикселей (соль)
            - black_pixel: Относительная интенсивность черных пикселей (перец)
            - noise_amount: Максимальное количество зашумляемых пикселей (абсолютное значение)
    """
    
    def __init__(self, param: Tuple[float, float, float]):
        """
        Инициализация фильтра шума "соль-перец".
        
        Аргументы:
            param: Кортеж, содержащий:
                   - white_pixel: Относительное количество белых пикселей (>=0)
                   - black_pixel: Относительное количество черных пикселей (>=0)
                   - noise_amount: Максимальное число изменяемых пикселей (>=0)
        """
        super().__init__(param, 'noise')
        
        self.white_pixel = param[0] 
        self.black_pixel = param[1]  
        self.noise_amount = param[2]

    def discription(self) -> str:
        return f"|saltandpappernoise_{self.param}"

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение шума "соль-перец" к входному изображению.
        
        Аргументы:
            image: Входное изображение (в градациях серого или цветное) в виде numpy массива
            
        Возвращает:
            Изображение с добавленным шумом "соль-перец" (того же типа, что и входное)
        """
        
        noisy = image.copy()
        h, w = image.shape[:2]
        total_pixels = h * w

        white_count  = self.param[0]
        black_count  = self.param[1]
        max_noise = self.param[2]
        
        if white_count + black_count <= 0 or max_noise <= 0:
            return noisy
        
        total = white_count + black_count
        white_prob = white_count  / total
        black_prob = black_count  / total
        
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
    добавляет коричневый шум
        :param strength: Сила шума (0-255)
        :param f3dB: Частота среза экспоненциального фильтра (0-0.5)
        :param fs: Псевдо-частота дискретизации (не критично для изображений)
        :param apply_highpass: Применять ли high-pass фильтр
        :param highpass_cutoff: Cutoff частота для high-pass фильтра
        :return: 2D numpy-массив с добавленным шумом
    """

    def __init__(self, strength=30, f3dB=0.05, fs=1.0, apply_highpass=True, highpass_cutoff=0.01):
        """
        :param strength: Сила шума (0-255)
        :param f3dB: Частота среза экспоненциального фильтра (0-0.5)
        :param fs: Псевдо-частота дискретизации (не критично для изображений)
        :param apply_highpass: Применять ли high-pass фильтр
        :param highpass_cutoff: Cutoff частота для high-pass фильтра
        :return: 2D numpy-массив с добавленным шумом
        """
        self.strength = strength
        self.f3dB = f3dB
        self.fs=fs
        self.apply_highpass = apply_highpass
        self.highpass_cutoff = highpass_cutoff
        super().__init__(1, 'noise')

    def discription(self) -> str:
        return f"|oldphotonoise_{self.strength}"

    def find_alpha(self, Fs, f3dB):
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

    def generate_2d_brownian_noise(self, shape, alpha):
        """Создать 2D Brownian noise с фильтрацией по строкам и столбцам."""
        print("Генерация 2D Brownian шума...")
        x = np.random.normal(0, 1, shape)
        b = [alpha]
        a = [1, -(1 - alpha)]
        
        # Фильтрация по строкам
        for i in range(shape[0]):
            x[i, :] = lfilter(b, a, x[i, :])
        
        # Фильтрация по столбцам
        for j in range(shape[1]):
            x[:, j] = lfilter(b, a, x[:, j])
        
        return x

    def high_pass_filter_2d(self, img, fs, cutoff=0.01):
        """Применить 2D high-pass фильтр через Butterworth."""
        print(f"Применение 2D high-pass фильтра с порогом {cutoff}...")
        sos = butter(2, cutoff, btype='highpass', fs=fs, output='sos')
        
        filtered = np.copy(img)
        # Фильтрация по строкам
        for i in range(img.shape[0]):
            filtered[i, :] = sosfilt(sos, img[i, :])
        
        # Фильтрация по столбцам
        for j in range(img.shape[1]):
            filtered[:, j] = sosfilt(sos, filtered[:, j])
        
        return filtered

    def filter(self, img):
        """
        Применяет 2D Brownian шум к изображению.

        """
        img = np.array(img, dtype=np.float32)

        alpha = self.find_alpha(self.fs, self.f3dB)
        brown_noise = self.generate_2d_brownian_noise(img.shape, alpha)

        if self.apply_highpass:
            brown_noise = self.high_pass_filter_2d(brown_noise, self.fs, cutoff=self.highpass_cutoff)

        # Нормализуем шум к диапазону [-1, 1]
        brown_noise -= brown_noise.min()
        brown_noise /= (brown_noise.max() - brown_noise.min())
        brown_noise = 2 * brown_noise - 1

        # Применение шума
        noisy_img = img + self.strength * brown_noise

        # Ограничение значений
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img

    
class ColoredNoise(FilterBase):
    """
        Добавляет к изображению цветной шум (1/f)

        :param noise_level: уровень шума (0.0 - без шума, 1.0 - сильный шум)
        :param beta: параметр спектрального наклона (1.0 = розовый шум, 2.0 = коричневый)
        :return: зашумлённое изображение
    """
    def __init__(self, noise_level: float = 0.2, beta: float = 1.0):
        """
        :param noise_level: уровень шума (0.0 - без шума, 1.0 - сильный шум)
        :param beta: параметр спектрального наклона (1.0 = розовый шум, 2.0 = коричневый)
        :return: зашумлённое изображение
        """
        self.noise_level = noise_level
        self.beta = beta
        super().__init__(1, 'noise')

    def discription(self) -> str:
        return f"|colorednoise_{self.beta}_{self.noise_level}"

    def filter(self, img):
        
        assert img.ndim == 2, "Ожидается двумерное изображение (grayscale)."

        h, w = img.shape

        # Генерируем 2D розовый шум того же размера
        pink_noise = powerlaw_psd_gaussian(exponent=self.beta, size=(h, w))

        # Нормируем шум к диапазону значений изображения
        pink_noise = pink_noise / np.std(pink_noise)  # приведение к std=1
        pink_noise = pink_noise * self.noise_level * 255   # масштабируем под уровень шума

        # Добавляем шум к изображению
        noisy_img = img.astype(np.float32) + pink_noise

        # Клонируем в допустимые границы
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return noisy_img


class Pink_Noise(FilterBase):
    """
        
    """
    def __init__(self,noise_level: float = 0.2):
        """

        """
        self.noise_level = noise_level
        super().__init__(1, 'noise')

    def discription(self) -> str:
        return f"|pinknoise_{self.noise_level}"

    def filter(self, img):
        noise = pink_noise_2d(img.shape, 2)*self.noise_level
        res = np.clip(img + noise, 0.0, 255.0).astype(np.int16)
        return res
        
        
class Brown_Noise(FilterBase):
    """
        
    """
    def __init__(self,noise_level: float = 0.2):
        """

        """
        self.noise_level = noise_level
        super().__init__(1, 'noise')

    def discription(self) -> str:
        return f"|brownnoise_{self.noise_level}"

    def filter(self, img):
        noise = pink_noise_2d(img.shape, 4)*self.noise_level
        res = np.clip(img + noise, 0.0, 255.0).astype(np.int16)
        return res

        