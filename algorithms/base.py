"""
Базовый класс для алгоритмов деконволюции.
"""

import abc
import numpy as np
from typing import Any, Dict, List, Tuple
import json


class DeconvolutionAlgorithm(abc.ABC):
    """
    Абстрактный базовый класс для алгоритмов деконволюции.
    
    Attributes
    ----------
    name : str
        Название алгоритма.
    timer : float
        Время выполнения последнего вызова process() в секундах.
    """

    name = 'default' 
    param = None

    def __init__(self, name: str) -> None: 
        """
        Инициализация алгоритма деконволюции.
        
        Parameters
        ----------
        name : str
            Название алгоритма (должно быть уникальным).
        """
        super().__init__()
        self.name = name
        self.timer = -1
    
    @abc.abstractmethod
    def change_param(self, param: Dict[str, Any]) -> None:
        """
        Изменение гиперпараметров алгоритма.

        Parameters
        ----------
        param : dict
            Словарь с параметрами для изменения.
        """
        pass

    @abc.abstractmethod
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обработка изображения с использованием алгоритма деконволюции.
        
        Parameters
        ----------
        image : np.ndarray
            Входное размытое изображение.

        Returns
        -------
        restored : np.ndarray
            Восстановленное изображение.
        kernel : np.ndarray
            Оценённое ядро размытия (PSF).
        """
        pass

    @abc.abstractmethod
    def get_param(self) -> List[Tuple[str, Any]]:
        """
        Получение текущих гиперпараметров алгоритма.

        Returns
        -------
        params : list of tuple
            Список кортежей (название_параметра, значение).
        """
        pass

    def get_name(self) -> str:
        """
        Получение названия алгоритма.

        Returns
        -------
        name : str
            Название алгоритма.
        """
        return self.name
    
    def get_timer(self) -> float:
        """
        Получение времени работы алгоритма.

        Returns
        -------
        timer : float
            Время выполнения в секундах (-1 если не запускался).
        """
        return self.timer
    
    def import_param_from_file(self, file: str) -> None:
        """
        Загрузка параметров из JSON-файла.

        Parameters
        ----------
        file : str
            Путь к JSON-файлу с параметрами.
        """
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.change_param(data)
