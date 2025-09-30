import abc
import numpy as np
from typing import Any

class FilterBase(abc.ABC):
    """
    Абстрактный базовый класс для фильтров изображений.
    
    Атрибуты:
        param (Any): Параметры фильтра
    """
    
    param = None
    def __init__(self, param: Any, type) -> None:
        """
        Инициализация фильтра.
        
        Аргументы:
            param: Параметры фильтра
        """
        super().__init__()
        self.param = param
        self.type = type

    def get_type(self):
        return self.type

    @abc.abstractmethod
    def discription(self):
        pass
    @abc.abstractmethod
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение фильтра к изображению.
        
        Аргументы:
            image: Входное изображение в формате numpy массива
            
        Возвращает:
            Отфильтрованное изображение в формате numpy массива
        """
        pass

