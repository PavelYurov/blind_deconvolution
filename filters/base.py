import abc
import numpy as np
from typing import Any

"""
    над кодом работал:
    Юров П.И.
"""

class FilterBase(abc.ABC):
    """
        над кодом работал:
        Юров П.И.
    """
    """
    Абстрактный базовый класс для фильтров изображений.
    
    Атрибуты:
        param (Any): Параметры фильтра
        type (Any): Тип фильтра 
    """
    
    param = None
    def __init__(self, param: Any, type: str) -> None:
        """
        Инициализация фильтра.
        
        Аргументы:
            param: Параметры фильтра
            type: Nип фильтра (например, blur, noise и т.п)
        """
        super().__init__()
        self.param = param
        self.type = type

    def get_type(self) -> str:
        """Возвращает тип фильтра"""
        return self.type

    @abc.abstractmethod
    def discription(self) -> str:
        """Возвращает зашифрованное название фильтра и его параметры"""
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

