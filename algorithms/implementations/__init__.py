"""
Реализованные алгоритмы слепой деконволюции.

Содержит проверенные и протестированные реализации классических
алгоритмов слепой деконволюции.

Доступные алгоритмы:
    - RichardsonLucy: Алгоритм Richardson-Lucy
    - EMBlindDeconvolution: EM-алгоритм
    - MAP: Maximum a Posteriori
    - Babacan2009, Babacan2010: Байесовские методы
    - Molina2006, Likas2004, Tzikas2009: Вариационные методы
    - Amizic2012: Sparse Bayesian Blind Deconvolution

Авторы: Юров П.И., Беззаборов А.А.
"""

# Примечание: импорты выполняются явно при использовании,
# так как модули имеют разные зависимости
__all__ = [
    'RichardsonLucy',
    'EMBlindDeconvolution',
    'MAP',
    'Babacan2009',
    'Babacan2010',
    'Molina2006',
    'Likas2004',
    'Tzikas2009',
    'Amizic2012',
]

