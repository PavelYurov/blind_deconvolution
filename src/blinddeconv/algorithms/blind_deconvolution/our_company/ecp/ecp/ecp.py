"""
ecp.py

Слепая деконволюция изображений с использованием априорного 
распределения экстремальных каналов (Extreme Channels Prior, ECP).

Основано на методе:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao: "Image Deblurring via
    Extreme Channels Prior", CVPR, 2017.

Метод расширяет подход на основе темного канала (DCP), добавляя 
симметричное условие для светлого канала. Результирующая 
оптимизационная задача решается с помощью метода полуквадратичного 
расщепления (Half-Quadratic Splitting, HQS).

Конвейер обработки:
1. Нормализация входного изображения к диапазону float64 [0, 1].
2. Преобразование в полутоновый формат для оценки ядра размытия.
3. Многомасштабная слепая оценка ядра размытия.
4. Неслепое восстановление полноцветного изображения с подавлением 
   артефактов звона.
5. Возврат восстановленного изображения (в формате int16 [0, 255]) и 
   оцененного ядра.
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

from .solvers import blind_deconv, ringing_artifacts_removal


class ECP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе априорного распределения 
    экстремальных каналов (светлого и темного).

    Параметры
    ---------
    kernel_size : int, по умолчанию 35
        Пространственный размер неизвестной функции рассеяния точки (нечетное число).
    lambda_dark : float, по умолчанию 4e-3
        Весовой коэффициент для L0-регуляризации интенсивности экстремальных 
        каналов (применяется симметрично к темному и светлому каналам).
    lambda_grad : float, по умолчанию 4e-3
        Весовой коэффициент для L0-регуляризации градиентов.
    xk_iter : int, по умолчанию 5
        Количество итераций слепой оценки на каждом уровне пирамиды масштабов.
    gamma_correct : float, по умолчанию 1.0
        Экспонента гамма-коррекции, применяемая перед оценкой ядра. 
        Значение 1.0 означает отсутствие коррекции.
    k_thresh : float, по умолчанию 20.0
        Относительный порог для финального ядра. Значения ядра, меньшие чем 
        max(k) / k_thresh, обнуляются.
    lambda_tv : float, по умолчанию 0.001
        Весовой коэффициент TV-регуляризации для этапа неслепой деконволюции.
    lambda_l0 : float, по умолчанию 5e-4
        Весовой коэффициент L0-регуляризации для этапа неслепой деконволюции.
    weight_ring : float, по умолчанию 1.0
        Коэффициент подавления артефактов звона (0 соответствует отключению 
        дополнительного подавления).
    """

    def __init__(
        self,
        kernel_size: int = 35,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
    ):
        super().__init__(name='ECP-BD')

        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск алгоритма слепой деконволюции.

        Выполняет предварительную обработку входного изображения (нормализацию и 
        перевод в полутоновый формат), многомасштабную оценку ядра размытия и 
        последующее неслепое восстановление полноцветного изображения с подавлением 
        артефактов звона.

        Параметры
        ---------
        image : ndarray
            Входное размытое изображение.

        Возвращает
        ----------
        x_final : ndarray
            Восстановленное изображение в формате int16 [0, 255].
        kernel : ndarray
            Оцененное ядро размытия.
        """
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_dark, self.lambda_grad, opts
        )

        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_dark', self.lambda_dark),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
