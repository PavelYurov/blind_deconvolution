"""
lowrank_li.py

Источник:
    Li Siyao, Shiyu Zhao, Wenzhe Wang, Ping Tan:
    "Understanding Kernel Size in Blind Deconvolution", WACV 2019.
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

from .solvers import multiscaled_cry

class LowRankLi_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 185,
        lambda_: float = 80.0,
        sigma: float = 1.0,
        tx: float = 1e-2,
        tau: float = 1e-5,
        delta: float = 1e-5,
        imax: int = 5,
        ximax: int = 2,
        xjmax: int = 2,
        kmax: int = 3,
        rmax: int = 3,
        mu: float = 1.0,
        iterkrank: int = 10,
        threshold: float = 0.05,
        nb_lambda: float = 3000.0,
        nb_alpha: float = 1.0,
    ):
        super().__init__(name='LowRankLi-BD')

        self.kernel_size = kernel_size
        self.lambda_ = lambda_
        self.sigma = sigma
        self.tx = tx
        self.tau = tau
        self.delta = delta
        self.imax = imax
        self.ximax = ximax
        self.xjmax = xjmax
        self.kmax = kmax
        self.rmax = rmax
        self.mu = mu
        self.iterkrank = iterkrank
        self.threshold = threshold
        self.nb_lambda = nb_lambda
        self.nb_alpha = nb_alpha

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

        K = self.kernel_size
        if K % 2 == 0:
            K += 1

        params = {
            'lambda_': self.lambda_,
            'sigma': self.sigma,
            'tx': self.tx,
            'tau': self.tau,
            'delta': self.delta,
            'imax': self.imax,
            'ximax': self.ximax,
            'xjmax': self.xjmax,
            'kmax': self.kmax,
            'rmax': self.rmax,
            'mu': self.mu,
            'iterkrank': self.iterkrank,
            'threshold': self.threshold,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
        }

        x, kernel = multiscaled_cry(yg, K, params)

        x_final = np.clip(x, 0.0, 1.0) * 255.0
        x_final = np.round(x_final).astype(np.int16)

        self.hyperparams = {
            'kernel_size': K,
            'lambda_': self.lambda_,
            'sigma': self.sigma,
            'tx': self.tx,
            'tau': self.tau,
            'delta': self.delta,
            'imax': self.imax,
            'ximax': self.ximax,
            'xjmax': self.xjmax,
            'kmax': self.kmax,
            'rmax': self.rmax,
            'mu': self.mu,
            'iterkrank': self.iterkrank,
            'threshold': self.threshold,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
            'time': time.time() - start_time,
        }

        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_', self.lambda_),
            ('sigma', self.sigma),
            ('tx', self.tx),
            ('tau', self.tau),
            ('delta', self.delta),
            ('imax', self.imax),
            ('ximax', self.ximax),
            ('xjmax', self.xjmax),
            ('kmax', self.kmax),
            ('rmax', self.rmax),
            ('mu', self.mu),
            ('iterkrank', self.iterkrank),
            ('threshold', self.threshold),
            ('nb_lambda', self.nb_lambda),
            ('nb_alpha', self.nb_alpha),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
