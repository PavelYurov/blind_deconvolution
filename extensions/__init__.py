"""
Пакет расширений для фреймворка слепой деконволюции изображений.

Модули:
    base: Базовые классы и интерфейсы
    hyperparameter_optimization: Оптимизация гиперпараметров
    pareto_analysis: Анализ фронта Парето

Автор: Беззаборов А.А.
"""

from extensions.base import (
    ProcessingExtension,
    ParameterRange,
    OptimizationResult,
    ParetoPoint,
    OptimizationMethod,
    MetricType
)

from extensions.hyperparameter_optimization import HyperparameterOptimizer
from extensions.pareto_analysis import ParetoFrontAnalyzer

__all__ = [
    'ProcessingExtension',
    'ParameterRange',
    'OptimizationResult',
    'ParetoPoint',
    'OptimizationMethod',
    'MetricType',
    'HyperparameterOptimizer',
    'ParetoFrontAnalyzer',
]

__version__ = '1.0.0'
__author__ = 'Беззаборов А.А.'
