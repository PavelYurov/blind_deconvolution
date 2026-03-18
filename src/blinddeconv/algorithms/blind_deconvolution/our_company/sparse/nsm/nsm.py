"""
nsm.py

Blind Image Deconvolution Using Normalized Sparsity Measure (NSM).

Reference:
    D. Krishnan, T. Tay, R. Fergus:
    "Blind Deconvolution using a Normalized Sparsity Measure", CVPR 2011.

Ported from MATLAB code (BlindDeconvolution-main/matlab/).

Pipeline (mirrors MATLAB ms_blind_deconv):
    1. Normalise input to float64 [0, 1].
    2. Multi-scale blind deconvolution on grayscale (kernel estimation).
    3. Non-blind restoration via Split-Bregman per channel (fast_deconv_bregman).
    4. Return restored image (int16, [0, 255]) and kernel.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# -- Framework base class import (DO NOT MODIFY) --------------------------
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
# --------------------------------------------------------------------------

from .solvers import ms_blind_deconv


class NSM_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Normalized Sparsity Measure (NSM).

    Default parameters match MATLAB test_blind_deconv.m.

    Parameters
    ----------
    kernel_size   : int   -- spatial support of the unknown PSF (square, odd).
    prescale      : float -- pre-downscale factor for large images. Default 1.0.
    min_lambda    : float -- weight on data-fidelity term. Default 250.0.
    k_reg_wt      : float -- L1 regularisation weight on kernel. Default 1.0.
    gamma_correct : float -- gamma correction exponent. Default 1.0.
    k_thresh      : float -- final kernel threshold (fraction of max). Default 0.0.
    delta         : float -- ISTA step size. Default 0.001.
    x_in_iter     : int   -- inner ISTA iterations. Default 2.
    x_out_iter    : int   -- outer ISTA iterations. Default 2.
    xk_iter       : int   -- x/k alternations per scale. Default 21.
    nb_lambda     : float -- non-blind deconv lambda. Default 3000.
    nb_alpha      : float -- non-blind deconv alpha (hyper-Laplacian exponent). Default 1.0.
    use_ycbcr     : bool  -- if True, deconvolve only Y channel of YCbCr (MATLAB default). Default True.
    """

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

    # -- Main entry point --------------------------------------------------
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Normalise to float64 [0, 1]
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # 2. Build options dict (matching MATLAB opts structure)
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

        # 3. Multi-scale blind deconvolution + non-blind step
        # Pass full image (possibly color) -- ms_blind_deconv handles
        # grayscale conversion for kernel estimation and per-channel non-blind
        deblurred, kernel = ms_blind_deconv(y, opts)

        # 4. Output
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

    # -- Interface methods -------------------------------------------------
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
