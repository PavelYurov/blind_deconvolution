import abc
import numpy as np
from typing import Any
import json

class DeconvolutionAlgorithm(abc.ABC):
    """
    Абстрактный базовый класс для алгоритмов деконволюции.
    
    Атрибуты:
        name (str): Название алгоритма
        params (Any): Параметры алгоритма
    """
    name = 'default' 
    param = None
    def __init__(self, name: str) -> None: 
        """
        Инициализация алгоритма деконволюции.
        
        Аргументы:
            params: Дополнительные параметры алгоритма
            name: Название алгоритма (должно быть уникальным)
        """
        super().__init__()
        self.name = name
        self.timer = -1
        pass
    
    @abc.abstractmethod
    def change_param(self, param):
        '''
        Необходимо для изменения гиперпараметров во время работы алгоритма
        '''
        pass

    @abc.abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Обработка изображения с использованием алгоритма деконволюции.
        
        Аргументы:
            image: Входное размытое изображение в виде numpy массива
            
            
        Возвращает:
            Восстановленное изображение в виде numpy массива
            kernel: предикт psf
        """
        pass

    @abc.abstractmethod
    def get_param(self):
        """
        Возвращает:
            массив текущих гиперпараметров, вместе с их названиями [('param_name', param),...]
        """
        pass

    def get_name(self) -> str:
        """Получение названия алгоритма."""
        return self.name
    
    def get_timer(self)->float:
        '''
        Возвращает:
            время работы алгоритма
        '''
        return self.timer
    
    def import_param_from_file(self, file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.change_param(data)
