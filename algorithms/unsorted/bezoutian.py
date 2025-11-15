from .base import DeconvolutionAlgorithm
import numpy as np
from typing import Any

class TemplateAlgorithm(DeconvolutionAlgorithm):
    """
    Шаблонная реализация алгоритма деконволюции.
    
    Служит примером для создания новых алгоритмов.
    """

    def __init__(self) -> None:
        """
        Инициализация шаблонного алгоритма.
        
        Аргументы:
            param: Параметры алгоритма
        """
        super().__init__('Bezoutioan')
    
    def process(self, image: np.ndarray) -> np.ndarray:
        
        processed_image = image.copy()
        return processed_image
    
