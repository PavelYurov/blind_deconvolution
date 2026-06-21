"""
gbbid.py

Слепая деконволюция изображений с использованием графового априорного 
распределения (Graph-Based RGTV Prior, GBBID).

Основано на методе:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Конвейер обработки:
1. Нормализация входного изображения к диапазону float64 [0, 1].
2. Преобразование в полутоновый формат и обрезка краевых пикселей.
3. Слепая оценка ядра размытия иерархическим методом (от грубого масштаба 
   к точному) с использованием регуляризации RGTV.
4. Неслепое восстановление с использованием гиперлапласовского априорного 
   распределения (метод FHLP).
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

from .solvers import bid_rgtv_c2f_cg, Deconvolution_FHLP


class GBBID(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе графовой регуляризации RGTV.

    Параметры
    ---------
    k_estimate_size : int, по умолчанию 69
        Ожидаемый пространственный размер оцениваемого ядра размытия (нечетное число).
    border : int, по умолчанию 20
        Количество краевых пикселей, обрезаемых перед началом оценки ядра 
        для снижения влияния граничных артефактов.
    lambda_fhlp : float, по умолчанию 2e3
        Вес члена верности данных для этапа неслепой деконволюции FHLP.
    alpha_fhlp : float, по умолчанию 0.5
        Экспонента гиперлапласиана для этапа неслепой деконволюции.
    """

    def __init__(
        self,
        k_estimate_size: int = 69,
        border: int = 20,
        lambda_fhlp: float = 2e3,
        alpha_fhlp: float = 0.5,
    ):
        super().__init__(name='GBBID')

        self.k_estimate_size = k_estimate_size
        self.border = border
        self.lambda_fhlp = lambda_fhlp
        self.alpha_fhlp = alpha_fhlp

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        b = self.border
        if b > 0:
            yg_cropped = yg[b:-b, b:-b]
        else:
            yg_cropped = yg

        kernel, _skeleton = bid_rgtv_c2f_cg(
            yg_cropped, self.k_estimate_size, show_intermediate=False)

        if y.ndim == 3:
            Latent = np.zeros_like(y)
            for ch in range(y.shape[2]):
                Latent[:, :, ch] = Deconvolution_FHLP(
                    y[:, :, ch], kernel,
                    lambda_val=self.lambda_fhlp,
                    alpha=self.alpha_fhlp)
        else:
            Latent = Deconvolution_FHLP(
                y, kernel,
                lambda_val=self.lambda_fhlp,
                alpha=self.alpha_fhlp)

        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'k_estimate_size': self.k_estimate_size,
            'border': self.border,
            'lambda_fhlp': self.lambda_fhlp,
            'alpha_fhlp': self.alpha_fhlp,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('k_estimate_size', self.k_estimate_size),
            ('border', self.border),
            ('lambda_fhlp', self.lambda_fhlp),
            ('alpha_fhlp', self.alpha_fhlp),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
