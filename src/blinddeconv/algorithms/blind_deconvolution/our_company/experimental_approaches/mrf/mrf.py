"""
mrf.py

Источник:
    N. Komodakis, N. Paragios: "MRF-based Blind Image Deconvolution",
    Proceedings of the 11th Asian Conference on Computer Vision (ACCV),
    Vol. 3, pp. 361-374, 2012.
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

from .solvers import blind_deconvolution, RESIZE_FACTORS

class MRF_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: tuple = (40, 40),
        mu: float = 0.4e-3,
        lam: float = 0.4e-3,
        tau: float = 1.0e-3,
        rho: float = 1.0e+3,
        n_clusters: int = 15,
        max_iter: int = 10,
        max_admm_iter: int = 10,
        convergence_thresh: float = 1.0e-8,
        resize_factors: tuple = RESIZE_FACTORS,
    ):
        super().__init__(name='MRF-BD')

        self.kernel_shape = kernel_shape
        self.mu = mu
        self.lam = lam
        self.tau = tau
        self.rho = rho
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.max_admm_iter = max_admm_iter
        self.convergence_thresh = convergence_thresh
        self.resize_factors = resize_factors

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        restored, kernel = blind_deconvolution(
            yg,
            kernel_shape=self.kernel_shape,
            mu=self.mu,
            lam=self.lam,
            tau=self.tau,
            rho=self.rho,
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            max_admm_iter=self.max_admm_iter,
            convergence_thresh=self.convergence_thresh,
            resize_factors=self.resize_factors,
            verbose=False,
        )
        restored = np.clip(restored, 0.0, 1.0)

        self.hyperparams = {
            'kernel_shape': self.kernel_shape,
            'mu': self.mu,
            'lam': self.lam,
            'tau': self.tau,
            'rho': self.rho,
            'n_clusters': self.n_clusters,
            'max_iter': self.max_iter,
            'max_admm_iter': self.max_admm_iter,
            'convergence_thresh': self.convergence_thresh,
            'resize_factors': self.resize_factors,
            'time': time.time() - start_time,
        }

        x_final = restored * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('mu', self.mu),
            ('lam', self.lam),
            ('tau', self.tau),
            ('rho', self.rho),
            ('n_clusters', self.n_clusters),
            ('max_iter', self.max_iter),
            ('max_admm_iter', self.max_admm_iter),
            ('convergence_thresh', self.convergence_thresh),
            ('resize_factors', self.resize_factors),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                elif key == 'resize_factors':
                    self.resize_factors = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
