from .base import DeconvolutionAlgorithm
import numpy as np
from typing import Any
from time import time


class TemplateAlgorithm(DeconvolutionAlgorithm):
    """
    Шаблонная реализация алгоритма деконволюции.
    
    Служит примером для создания новых алгоритмов.
    """

    def __init__(self, param: Any) -> None:
        """
        Инициализация шаблонного алгоритма.
        
        Аргументы:
            param: Параметры алгоритма
        """
        super().__init__('ALGORITHM_NAME')
        self.param = param
    
    def change_param(self, param):
        '''
        Изменеие параметра

        Аргументы:
            -param: словарь параметров алгоритма
        '''
        return super().change_param(param)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Обработка изображения с использованием шаблонного алгоритма.
        
        Аргументы:
            image: Входное размытое изображение в виде numpy массива
            kernel: предикт psf
            
        Возвращает:
            Восстановленное изображение в виде numpy массива
        """
        timer1 = time()
        processed_image = image.copy()
        kernel = [[0]]
        timer2 = time()
        self.timer = timer2 - timer1
        return processed_image, kernel
    
    def get_param(self):
        return []
