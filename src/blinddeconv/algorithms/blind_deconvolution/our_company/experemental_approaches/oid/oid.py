"""
oid.py

Источник:
    L. Chen, F. Fang, J. Zhang, J. Liu, G. Zhang:
    "OID: Outlier Identifying and Discarding in Blind Image Deblurring",
    ECCV 2020.
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

from .solvers import blind_deconv, image_estimate

class OID_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 27,
        lambda_grad: float = 4e-3,
        xk_iter: int = 4,
        last_iter: int = 4,
        k_thresh: float = 20.0,
        isnoisy: int = 1,
        predeblur: str = 'L0',
        gamma_correct: float = 1.0,
        lambda_final: float = 3e-3,
    ):
        super().__init__(name='OID-BD')

        self.kernel_size = kernel_size
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.last_iter = last_iter
        self.k_thresh = k_thresh
        self.isnoisy = isnoisy
        self.predeblur = predeblur
        self.gamma_correct = gamma_correct
        self.lambda_final = lambda_final

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 2:
            yg = y
        else:
            yg = y[:, :, 0]

        opts = {
            'kernel_size':   self.kernel_size,
            'xk_iter':       self.xk_iter,
            'last_iter':     self.last_iter,
            'k_thresh':      self.k_thresh,
            'isnoisy':       self.isnoisy,
            'predeblur':     self.predeblur,
            'gamma_correct': self.gamma_correct,
        }

        kernel, _interim_latent = blind_deconv(yg, self.lambda_grad, opts)

        Latent, _w_out = image_estimate(y, kernel, self.lambda_final, False)
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size':   self.kernel_size,
            'lambda_grad':   self.lambda_grad,
            'xk_iter':       self.xk_iter,
            'last_iter':     self.last_iter,
            'k_thresh':      self.k_thresh,
            'isnoisy':       self.isnoisy,
            'predeblur':     self.predeblur,
            'gamma_correct': self.gamma_correct,
            'lambda_final':  self.lambda_final,
            'time':          time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size',   self.kernel_size),
            ('lambda_grad',   self.lambda_grad),
            ('xk_iter',       self.xk_iter),
            ('last_iter',     self.last_iter),
            ('k_thresh',      self.k_thresh),
            ('isnoisy',       self.isnoisy),
            ('predeblur',     self.predeblur),
            ('gamma_correct', self.gamma_correct),
            ('lambda_final',  self.lambda_final),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
