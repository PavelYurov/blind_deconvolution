"""
lip.py

Слепая деконволюция изображений с использованием логарифмического априорного 
распределения (Logarithmic Image Prior, LIP) с нижней границей.

Основано на:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", EMMCVPR, 2015.

Реализует три метода оценки функции рассеяния точки (PSF) из оригинальной статьи:
- 'mm' : метод мажоризации-минимизации (Таблица 2). Использует градиентный 
  спуск для мажорированной задачи взвешенной полной вариации.
- 'cv' : метод прямо-двойственного расщепления Конда-Вю для мажорированной 
  задачи (вычисление верности данных через пространственные свертки без БПФ).
- 'pd' : метод прямо-двойственного расщепления (Таблица 1). Решает исходную 
  невыпуклую задачу энергии LIP напрямую (без внешней мажоризации MM).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm

from .solvers import coarse_to_fine
from .utils import (
    gamma_correction,
    make_size_odd,
    edgetaper,
    pad_image,
    crop_image,
    wiener_filter,
    tikhonov_filter,
)


class LIP_BD(DeconvolutionAlgorithm):
    """
    Класс алгоритма слепой деконволюции LIP (оригинальная базовая версия).

    Конвейер обработки:
    1. Нормализация входного изображения к диапазону float64 [0, 1].
    2. Приведение пространственных размеров к нечетным значениям.
    3. Применение гамма-коррекции (опционально).
    4. Иерархическая (от грубого масштаба к точному) оценка ядра размытия.
    5. Пороговая обработка элементов ядра (удаление шумовых компонент).
    6. Финальная неслепая деконволюция с использованием оцененного ядра 
       (фильтр Тихонова или Винера).
    7. Возврат восстановленного изображения в формате int16 [0, 255] и ядра.

    Параметры
    ---------
    kernel_shape : tuple of ints (MK, NK)
        Пространственный размер неизвестной функции рассеяния точки (PSF).
    lambda_val : float, по умолчанию 30000.0
        Вес члена верности данных (соответствует параметру beta в статье).
    tau : float, по умолчанию 1e-3
        Параметр нижней границы логарифмического априорного распределения.
    outer_iters : int, по умолчанию 140
        Количество внешних итераций EM на каждом уровне пирамиды.
    inner_iters : int, по умолчанию 5
        Количество внутренних итераций градиентного спуска на одну внешнюю.
    k_step : array-like, опционально
        Массив размеров шага для обновления ядра.
    u_step : array-like, опционально
        Массив размеров шага для обновления изображения.
    lambda_mult : float, по умолчанию 2.1
        Множитель для параметра lambda между уровнями пирамиды.
    scale_mult : float, по умолчанию 1.414 (sqrt(2))
        Делитель размера ядра между уровнями пирамиды.
    gamma_correction : bool, по умолчанию False
        Флаг применения гамма-коррекции.
    gamma : float, по умолчанию 1.0
        Экспонента гамма-коррекции.
    method : {'mm', 'pd', 'cv'}, по умолчанию 'mm'
        Метод оптимизации оценки ядра.
    kernel_threshold : float, по умолчанию 0.05
        Относительный порог (доля от максимума ядра), ниже которого 
        элементы ядра обнуляются.
    final_deconv : {'tikhonov', 'wiener'}, по умолчанию 'tikhonov'
        Метод финальной неслепой деконволюции.
    final_alpha : float, по умолчанию 0.001
        Степень регуляризации для неслепого шага.
    verbose : bool, по умолчанию False
        Флаг вывода информации о прогрессе.

    Параметры для метода PD (метод 'pd')
    ------------------------------------
    pd_outer_iters : int, по умолчанию 30
        Количество внешних итераций.
    pd_inner_iters : int, по умолчанию 50
        Количество внутренних итераций алгоритма Шамболя-Пока.
    pd_theta : float, по умолчанию 1.0
        Параметр перерелаксации (theta).
    pd_tau : float, опционально
        Шаг для прямой переменной (если None, вычисляется автоматически).
    pd_sigma : float, опционально
        Шаг для двойственной переменной (если None, равен pd_tau).
    h_mode : {'closed', 'lut'}, по умолчанию 'closed'
        Способ вычисления функции H (аналитический или через таблицу).
    h_lut_size : int, по умолчанию 4096
        Размер таблицы для функции H.
    h_lut_xi_max : float, по умолчанию 4.0
        Максимальное значение аргумента для интерполяции функции H.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_val: float = 30000.0,
        tau: float = 1e-3,
        outer_iters: int = 140,
        inner_iters: int = 5,
        k_step: Any = None,
        u_step: Any = None,
        lambda_mult: float = 2.1,
        scale_mult: float = 1.4142135623730951,  # sqrt(2)
        gamma_correction: bool = False,
        gamma: float = 1.0,
        method: str = 'mm',
        kernel_threshold: float = 0.05,
        final_deconv: str = 'tikhonov',
        final_alpha: float = 0.001,
        verbose: bool = False,
        pd_outer_iters: int = 30,
        pd_inner_iters: int = 50,
        pd_theta: float = 1.0,
        pd_tau: float = None,
        pd_sigma: float = None,
        h_mode: str = 'closed',
        h_lut_size: int = 4096,
        h_lut_xi_max: float = 4.0,
    ):
        super().__init__(name='LIP-BD')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_val = lambda_val
        self.tau = tau
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters

        if k_step is None:
            self.k_step = np.array([1e-2, 5e-3, 1e-3, 5e-4])
        else:
            self.k_step = np.atleast_1d(np.asarray(k_step, dtype=np.float64))
        if u_step is None:
            self.u_step = np.array([1e-2, 5e-3, 1e-3, 1e-3])
        else:
            self.u_step = np.atleast_1d(np.asarray(u_step, dtype=np.float64))

        self.lambda_mult = lambda_mult
        self.scale_mult = scale_mult
        self.gamma_corr = gamma_correction
        self.gamma = gamma
        self.method = method.lower()
        self.kernel_threshold = kernel_threshold
        self.final_deconv = final_deconv.lower()
        self.final_alpha = final_alpha
        self.verbose = verbose

        self.pd_outer_iters = int(pd_outer_iters)
        self.pd_inner_iters = int(pd_inner_iters)
        self.pd_theta = float(pd_theta)
        self.pd_tau = pd_tau
        self.pd_sigma = pd_sigma
        self.h_mode = str(h_mode)
        self.h_lut_size = int(h_lut_size)
        self.h_lut_xi_max = float(h_lut_xi_max)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск алгоритма слепой деконволюции.
        """
        start_time = time.time()

        MK, NK = self.kernel_shape

        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        M_orig, N_orig = f.shape

        f = make_size_odd(f)

        if self.gamma_corr:
            f = gamma_correction(f, self.gamma)

        if self.method in ('mm', 'cv'):
            blind_params = {
                'outer_iters': self.outer_iters,
                'inner_iters': self.inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
            }
            ctf_params = {
                'final_lambda': self.lambda_val,
                'lambda_mult': self.lambda_mult,
                'scale_mult': self.scale_mult,
            }
            u, k = coarse_to_fine(f, MK, NK, blind_params, ctf_params,
                                  verbose=self.verbose, method=self.method)
        elif self.method == 'pd':
            blind_params = {
                'outer_iters': self.pd_outer_iters,
                'inner_iters': self.pd_inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
                'pd_theta': self.pd_theta,
                'pd_tau': self.pd_tau,
                'pd_sigma': self.pd_sigma,
                'h_mode': self.h_mode,
                'h_lut_size': self.h_lut_size,
                'h_lut_xi_max': self.h_lut_xi_max,
            }
            ctf_params = {
                'final_lambda': self.lambda_val,
                'lambda_mult': self.lambda_mult,
                'scale_mult': self.scale_mult,
            }
            u, k = coarse_to_fine(f, MK, NK, blind_params, ctf_params,
                                  verbose=self.verbose, method='pd')
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose 'mm', 'pd', or 'cv'.")

        k[k < self.kernel_threshold * k.max()] = 0.0
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        f_pad = pad_image(f, (MK, NK))
        f_pad = edgetaper(f_pad, k)

        if self.final_deconv == 'tikhonov':
            u_restored = tikhonov_filter(f_pad, k, alpha=self.final_alpha)
        elif self.final_deconv == 'wiener':
            u_restored = wiener_filter(f_pad, k, noise_snr=self.final_alpha)
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'tikhonov' or 'wiener'."
            )

        u_final = crop_image(u_restored, (M_orig, N_orig), (MK, NK))
        u_final = np.clip(u_final, 0.0, 1.0)

        self.hyperparams = {
            'lambda': self.lambda_val,
            'tau': self.tau,
            'method': self.method,
            'final_deconv': self.final_deconv,
            'final_alpha': self.final_alpha,
            'outer_iters': self.outer_iters,
            'inner_iters': self.inner_iters,
            'time': time.time() - start_time,
        }

        x_final = u_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_val', self.lambda_val),
            ('tau', self.tau),
            ('outer_iters', self.outer_iters),
            ('inner_iters', self.inner_iters),
            ('method', self.method),
            ('kernel_threshold', self.kernel_threshold),
            ('final_deconv', self.final_deconv),
            ('final_alpha', self.final_alpha),
            ('gamma_correction', self.gamma_corr),
            ('gamma', self.gamma),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                elif key in ('k_step', 'u_step'):
                    setattr(self, key, np.atleast_1d(
                        np.asarray(value, dtype=np.float64)))
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
