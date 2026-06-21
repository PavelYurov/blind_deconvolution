"""
pmp.py

Модуль слепой деконволюции изображений на основе априорной информации 
о локальных минимальных значениях интенсивности (Patch-wise Minimal Pixels, PMP).

Основано на методе:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong: "A Simple Local 
    Minimal Intensity Prior and An Improved Algorithm for Blind Image 
    Deblurring", IEEE TCSVT, 2021.

Математическая модель опирается на наблюдение, что в небольших неперекрывающихся 
паттернах четкого естественного изображения минимальные значения интенсивности 
близки к нулю. Процесс пространственного размытия сглаживает эти минимумы, 
повышая их значения. Использование L0-нормы карты минимальных пикселей 
позволяет эффективно разделять размытые и четкие изображения в процессе оптимизации.

Общий конвейер восстановления:
1. Нормализация динамического диапазона входного сигнала к отрезку [0.0, 1.0].
2. Формирование полутоновой матрицы яркости по стандарту ITU-R BT.601 для оценки ядра.
3. Многомасштабная слепая деконволюция с использованием априорного знания PMP.
4. Финальная неслепая деконволюция исходного сигнала с подавлением пространственных 
   артефактов (эффекта Гиббса).
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


class PMP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе априорного знания локальных 
    минимальных значений интенсивности.

    Целевой минимизируемый функционал (согласно уравнению 9 в первоисточнике):
    E(I, K) = ||I * K - B||_2^2 + lambda_pmp * ||P(I)||_0 + lambda_grad * ||nabla I||_0 + gamma * ||K||_2^2
    где P(I) — оператор извлечения минимальных пикселей паттерна.

    Параметры
    ---------
    kernel_size : int, по умолчанию 25
        Линейный размер квадратного пространственного носителя функции рассеяния
        точки. Значение должно быть нечетным числом.
    lambda_pmp : float, по умолчанию 0.1
        Коэффициент регуляризации априорного члена минимальных значений (параметр alpha).
    lambda_grad : float, по умолчанию 4e-3
        Коэффициент L0-регуляризации разреженности градиентов (параметр mu).
    xk_iter : int, по умолчанию 5
        Количество итераций попеременной минимизации на каждом уровне масштабной пирамиды.
    gamma_correct : float, по умолчанию 1.0
        Коэффициент предварительного степенного преобразования сигнала перед оценкой ядра.
    k_thresh : float, по умолчанию 20.0
        Жесткий порог отсечения шума в оцененном ядре размытия.
    patch_r : int или None, по умолчанию None
        Размер локального паттерна для поиска минимальных пикселей. При значении None
        вычисляется автоматически как floor(0.025 * mean(H, W)).
    lambda_tv : float, по умолчанию 0.001
        Вес полной вариации (Total Variation) на этапе финальной неслепой деконволюции.
    lambda_l0 : float, по умолчанию 5e-4
        Вес L0-нормы градиента на этапе финальной неслепой деконволюции.
    weight_ring : float, по умолчанию 1.0
        Коэффициент силы подавления краевых эффектов звона.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        lambda_pmp: float = 0.1,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        patch_r: int = None,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
    ):
        super().__init__(name='PMP-BD')

        self.kernel_size = kernel_size
        self.lambda_pmp = lambda_pmp
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.patch_r = patch_r
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск полного цикла слепой деконволюции для переданного кадра.

        Параметры
        ---------
        image : np.ndarray
            Входное искаженное изображение (одноканальное или многоканальное).

        Возвращаемое значение
        ---------------------
        Tuple[np.ndarray, np.ndarray]
            Кортеж, содержащий восстановленное изображение в целочисленном формате
            (int16, диапазон 0-255) и оцененную матрицу ядра искажения.
        """
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_pmp, self.lambda_grad, opts,
            patch_r=self.patch_r,
        )

        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_pmp': self.lambda_pmp,
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
            ('lambda_pmp', self.lambda_pmp),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('patch_r', self.patch_r),
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
