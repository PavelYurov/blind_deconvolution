"""
nsm.py

Источник:
    D. Krishnan, T. Tay, R. Fergus:
    "Blind Deconvolution using a Normalized Sparsity Measure", CVPR 2011.
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

from .solvers import ms_blind_deconv

class NSM_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 31,
        prescale: float = 1.0,
        min_lambda: float = 250.0,
        k_reg_wt: float = 1.0,
        gamma_correct: float = 1.0,
        k_thresh: float = 0.0,
        delta: float = 0.001,
        x_in_iter: int = 2,
        x_out_iter: int = 2,
        xk_iter: int = 21,
        nb_lambda: float = 3000.0,
        nb_alpha: float = 1.0,
        use_ycbcr: bool = True,
    ):
        super().__init__(name='NSM-BD')

        self.kernel_size = kernel_size
        self.prescale = prescale
        self.min_lambda = min_lambda
        self.k_reg_wt = k_reg_wt
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.delta = delta
        self.x_in_iter = x_in_iter
        self.x_out_iter = x_out_iter
        self.xk_iter = xk_iter
        self.nb_lambda = nb_lambda
        self.nb_alpha = nb_alpha
        self.use_ycbcr = use_ycbcr

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        opts = {
            'kernel_size': self.kernel_size,
            'prescale': self.prescale,
            'min_lambda': self.min_lambda,
            'k_reg_wt': self.k_reg_wt,
            'gamma_correct': self.gamma_correct,
            'k_thresh': self.k_thresh,
            'delta': self.delta,
            'x_in_iter': self.x_in_iter,
            'x_out_iter': self.x_out_iter,
            'xk_iter': self.xk_iter,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
            'use_ycbcr': self.use_ycbcr,
        }

        deblurred, kernel = ms_blind_deconv(y, opts)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'min_lambda': self.min_lambda,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
            'time': time.time() - start_time,
        }

        deblurred = np.clip(deblurred, 0.0, 1.0)
        x_final = deblurred * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('prescale', self.prescale),
            ('min_lambda', self.min_lambda),
            ('k_reg_wt', self.k_reg_wt),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('delta', self.delta),
            ('x_in_iter', self.x_in_iter),
            ('x_out_iter', self.x_out_iter),
            ('xk_iter', self.xk_iter),
            ('nb_lambda', self.nb_lambda),
            ('nb_alpha', self.nb_alpha),
            ('use_ycbcr', self.use_ycbcr),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
