"""
Абстрактный базовый класс для фильтров изображений.

Автор: Юров П.И.
"""

import abc
import numpy as np
from typing import Any


class FilterBase(abc.ABC):
    """
    Абстрактный базовый класс для фильтров изображений.
    
    Атрибуты
    --------
    param : Any
        Параметры фильтра.
    type : str
        Тип фильтра (blur, noise, denoise и т.п.).
    """
    
    param = None

    def __init__(self, param: Any, type: str) -> None:
        """
        Инициализация фильтра.
        
        Параметры
        ---------
        param : Any
            Параметры фильтра.
        type : str
            Тип фильтра (например, blur, noise и т.п.).
        """
        super().__init__()
        self.param = param
        self.type = type

    def get_type(self) -> str:
        """Возвращает тип фильтра."""
        return self.type

    @abc.abstractmethod
    def description(self) -> str:
        """Возвращает зашифрованное название фильтра и его параметры."""
        pass

    @abc.abstractmethod
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение фильтра к изображению.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение в формате numpy массива.
            
        Возвращает
        ----------
        np.ndarray
            Отфильтрованное изображение.
        """
        pass
