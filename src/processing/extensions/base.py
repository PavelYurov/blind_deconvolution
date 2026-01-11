"""
Базовые классы и типы для пакета расширений.

Содержит:
    - Абстрактный базовый класс ProcessingExtension
    - Классы данных для параметров и результатов
    - Перечисления для методов оптимизации и метрик

Автор: Беззаборов А.А.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ParameterRange:
    """
    Диапазон поиска гиперпараметра.
    
    Атрибуты
    --------
    name : str
        Имя параметра.
    min_value : Union[int, float]
        Минимальное значение диапазона.
    max_value : Union[int, float]
        Максимальное значение диапазона.
    log_scale : bool
        Использовать логарифмическую шкалу для параметров, 
        охватывающих несколько порядков величины.
    step : Optional[float]
        Размер дискретного шага (None для непрерывных параметров).
    """
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    log_scale: bool = False
    step: Optional[float] = None
    
    @property
    def is_integer(self) -> bool:
        """Проверка, является ли параметр целочисленным."""
        return isinstance(self.min_value, int) and isinstance(self.max_value, int)
    
    def validate(self) -> bool:
        """Проверка корректности диапазона."""
        if self.min_value >= self.max_value:
            raise ValueError(f"min_value должен быть меньше max_value для {self.name}")
        if self.log_scale and self.min_value <= 0:
            raise ValueError(f"log_scale требует положительных значений для {self.name}")
        return True


@dataclass
class OptimizationResult:
    """
    Контейнер для результатов оптимизации.
    
    Атрибуты
    --------
    best_params : Dict[str, Any]
        Лучшие найденные гиперпараметры.
    best_value : float
        Лучшее достигнутое значение целевой функции.
    n_trials : int
        Количество выполненных испытаний.
    study : Optional[Any]
        Объект исследования Optuna для дальнейшего анализа.
    history : List[Dict[str, Any]]
        История испытаний с параметрами и оценками.
    elapsed_time : float
        Затраченное время в секундах.
    """
    best_params: Dict[str, Any]
    best_value: float
    n_trials: int
    study: Optional[Any] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование результатов в словарь."""
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': self.n_trials,
            'elapsed_time': self.elapsed_time
        }


@dataclass
class ParetoPoint:
    """
    Точка в многокритериальном пространстве.
    
    Атрибуты
    --------
    objectives : Dict[str, float]
        Значения целевых функций (например, {'psnr': 25.0, 'time': 1.5}).
    parameters : Dict[str, Any]
        Связанные гиперпараметры.
    metadata : Dict[str, Any]
        Дополнительная информация (имя алгоритма, изображение и т.д.).
    is_pareto_optimal : bool
        Принадлежит ли точка фронту Парето.
    """
    objectives: Dict[str, float]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_pareto_optimal: bool = False
    
    def dominates(self, other: 'ParetoPoint', maximize: List[str] = None) -> bool:
        """
        Проверка доминирования текущей точки над другой.
        
        Параметры
        ---------
        other : ParetoPoint
            Другая точка для сравнения.
        maximize : List[str]
            Список критериев для максимизации.
            
        Возвращает
        ----------
        bool
            True если текущая точка доминирует над other.
        """
        if maximize is None:
            maximize = list(self.objectives.keys())
        
        dominated = False
        for key in self.objectives:
            if key in maximize:
                if self.objectives[key] < other.objectives.get(key, float('-inf')):
                    return False
                if self.objectives[key] > other.objectives.get(key, float('-inf')):
                    dominated = True
            else:
                if self.objectives[key] > other.objectives.get(key, float('inf')):
                    return False
                if self.objectives[key] < other.objectives.get(key, float('inf')):
                    dominated = True
        
        return dominated


class OptimizationMethod(Enum):
    """
    Поддерживаемые методы оптимизации.
    
    TPE : str
        Tree-structured Parzen Estimator (байесовская оптимизация).
        Рекомендуется для общей настройки гиперпараметров.
    RANDOM : str
        Случайный поиск.
        Эффективен для многомерных пространств и параллельных вычислений.
    GP : str
        Гауссовские процессы (требует BoTorch).
        Оптимален для дорогостоящих вычислений в малоразмерных пространствах.
    NSGA2 : str
        Non-dominated Sorting Genetic Algorithm II.
        Предназначен для многокритериальной оптимизации.
    """
    TPE = "tpe"
    RANDOM = "random"
    GP = "gp"
    NSGA2 = "nsga2"
    
    @classmethod
    def from_string(cls, value: str) -> 'OptimizationMethod':
        """Создание из строкового представления."""
        value_lower = value.lower()
        for method in cls:
            if method.value == value_lower:
                return method
        raise ValueError(f"Неизвестный метод оптимизации: {value}")


class MetricType(Enum):
    """
    Метрики качества для оценки восстановления изображений.
    
    Для всех метрик большее значение соответствует лучшему качеству.
    
    PSNR - пиковое отношение сигнал-шум (дБ)
    SSIM - индекс структурного сходства
    SHARPNESS - мера резкости на основе лапласиана
    """
    PSNR = "psnr"
    SSIM = "ssim"
    SHARPNESS = "sharpness"
    
    @classmethod
    def from_string(cls, value: str) -> 'MetricType':
        """Создание из строкового представления."""
        value_lower = value.lower()
        for metric in cls:
            if metric.value == value_lower:
                return metric
        raise ValueError(f"Неизвестная метрика: {value}")


class ProcessingExtension(ABC):
    """
    Абстрактный базовый класс для расширений обработки.
    
    Предоставляет общий функционал для расширений, 
    дополняющих основной конвейер обработки изображений.
    
    Атрибуты
    --------
    processing : Any
        Ссылка на основной экземпляр обработки.
    output_folder : Path
        Директория для сохранения результатов.
    logger : logging.Logger
        Логгер для данного расширения.
    """
    
    def __init__(self, processing_instance: Any, output_folder: str = "output"):
        """
        Инициализация расширения.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing.
        output_folder : str
            Директория для сохранения результатов.
        """
        self.processing = processing_instance
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Выполнение основного функционала расширения.
        
        Должен быть реализован в подклассах.
        """
        pass
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для вычисления метрик.
        
        Нормализует значения в диапазон [0, 1] и преобразует 
        в оттенки серого при необходимости.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение.
            
        Возвращает
        ----------
        np.ndarray
            Подготовленное изображение в диапазоне [0, 1].
        """
        image = np.asarray(image, dtype=np.float32)
        
        if image.max() > 1.0:
            image = image / 255.0
        
        if len(image.shape) == 3:
            if image.shape[2] >= 3:
                # Формула яркости: Y = 0.299R + 0.587G + 0.114B
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            else:
                image = image[..., 0]
        
        return np.clip(image, 0.0, 1.0)
    
    def _validate_processing(self) -> bool:
        """Проверка корректности объекта processing."""
        if self.processing is None:
            raise ValueError("processing_instance не может быть None")
        if not hasattr(self.processing, 'images'):
            raise AttributeError("processing_instance должен иметь атрибут 'images'")
        return True
