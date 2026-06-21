"""
lmgp.py

Модуль слепой деконволюции изображений на основе априорной информации
о локальном максимальном градиенте (Local Maximum Gradient Prior, LMGP).

Математическая модель опирается на свойство процесса свертки, при котором
максимальное значение градиента внутри любого локального паттерна
искаженного изображения строго меньше аналогичного значения для исходного.

Теоретическое обоснование и вывод оптимизационного функционала описаны в:
L. Chen, F. Fang, T. Wang, G. Zhang: Blind Image Deblurring With Local
Maximum Gradient Prior, CVPR, 2019.

Общий конвейер восстановления включает:
1. Нормализацию динамического диапазона входного сигнала к отрезку [0.0, 1.0].
2. Формирование одноканальной матрицы яркости по стандарту ITU-R BT.601
   для устойчивой оценки ядра искажения.
3. Итеративную оценку матрицы ядра размытия в пространстве масштабной пирамиды.
4. Финальную неслепую деконволюцию исходного изображения с подавлением
   пространственных артефактов (эффекта звона).
5. Масштабирование восстановленного сигнала обратно в целочисленный формат.
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


class LMGP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе априорного знания локального
    максимального градиента.

    Целевой минимизируемый функционал:
    E(I, K) = ||I * K - B||_2^2 + beta * ||2 - LMG(I)||_1 + gamma * ||nabla I||_0 + tau * ||K||_2^2

    Параметры
    ---------
    kernel_size : int, по умолчанию 27
        Линейный размер квадратного пространственного носителя функции рассеяния
        точки. Значение должно быть нечетным числом.
    lambda_lmg : float, по умолчанию 4e-3
        Коэффициент регуляризации априорного члена LMG. Соответствует параметру
        beta в математической модели.
    lambda_grad : float, по умолчанию 4e-3
        Коэффициент L0-регуляризации разреженности градиента скрытого
        изображения. Соответствует параметру gamma в математической модели.
    xk_iter : int, по умолчанию 5
        Количество итераций попеременной минимизации на каждом уровне
        масштабной пирамиды.
    gamma_correct : float, по умолчанию 1.0
        Коэффициент предварительного степенного преобразования сигнала
        перед оценкой ядра. Значение 1.0 эквивалентно линейному тракту.
    k_thresh : float, по умолчанию 20.0
        Жесткий порог отсечения шума в оцененном ядре. Элементы матрицы ядра,
        значения которых меньше максимального элемента, деленного на данный порог,
        принудительно обнуляются.
    lambda_tv : float, по умолчанию 0.001
        Вес полной вариации на этапе финальной неслепой деконволюции.
    lambda_l0 : float, по умолчанию 5e-4
        Вес L0-нормы градиента на этапе финальной неслепой деконволюции.
    weight_ring : float, по умолчанию 1.0
        Коэффициент силы подавления краевых эффектов звона. Значение 0.0
        соответствует полному отключению алгоритма подавления.
    """

    def __init__(
        self,
        kernel_size: int = 27,
        lambda_lmg: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
    ):
        super().__init__(name='LMGP-BD')

        self.kernel_size = kernel_size
        self.lambda_lmg = lambda_lmg
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
            yg, self.lambda_lmg, self.lambda_grad, opts,
        )

        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_lmg': self.lambda_lmg,
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
            ('lambda_lmg', self.lambda_lmg),
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
