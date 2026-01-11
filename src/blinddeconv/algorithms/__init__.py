"""
Пакет алгоритмов слепой деконволюции.

Модули:
    base: Базовый класс DeconvolutionAlgorithm
    blind_deconvolution/implementations: Собственные реализации
    blind_deconvolution/external: Внешние обёртки (original sources + wrapper)
    nonblind_deconvolution: Не-слепая деконволюция (известное ядро)
    kernel_estimation: Оценка ядра (PSF) без восстановления (если применимо)
    unsorted: Экспериментальные/черновые алгоритмы

This Python Wrapper Provides:
    - Python interface to original implementation
    - Parameter validation and type conversion
    - Automatic preprocessing (format, normalization, etc.)
    - Integration with BlindDeconvolution framework
    - Progress tracking and timing measurement
    - Error handling and fallback options

Wrapper Features:
    - Works with NumPy arrays and common image dtypes
    - Integrates into the BlindDeconvolution algorithm registry
    - Keeps bundled original source code untouched

Example:
    >>> from my_wrappers import <DeconvolutionAlgorithmClass>
    >>> processor = <DeconvolutionAlgorithmClass>(param1=value1, param2=value2)
    >>> result = processor.process(input_image)

Important Notes:
    1. Some wrappers require extra dependencies (see each wrapper's README/requirements)
    2. Original code remains unchanged
    3. This is purely an interface wrapper
    4. Check original repository for license information

Авторы: Юров П.И., Беззаборов А.А., Малыш Я.В.
"""

from .base import DeconvolutionAlgorithm
from .octave import OctaveEngine

__all__ = ['DeconvolutionAlgorithm', 'OctaveEngine']

