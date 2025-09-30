import numpy as np
import random
from typing import Tuple
from random import sample
from .base import FilterBase

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
        if param <= 0 or param > 1.0:
            raise ValueError("Интенсивность должна быть в диапазоне (0.0, 1.0]")
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
        noisy = image + np.random.poisson(image / 255.0 * self.param) / self.param*255.0
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