"""
oid.py

Blind Image Deblurring via Outlier Identification and Discarding (OID).

Reference:
    L. Chen, F. Fang, J. Zhang, J. Liu, G. Zhang:
    "OID: Outlier Identifying and Discarding in Blind Image Deblurring",
    ECCV 2020.

Pipeline (mirrors MATLAB outlier_public/demo_deblurring.m):
    1. Normalise input to float64 [0, 1].
    2. Optional grayscale conversion for kernel estimation.
    3. Multi-scale blind kernel estimation with outlier weights
       (blind_deconv -> fine_deblur / coarse_deblur).
    4. Non-blind restoration via the same IRLS image_estimate with
       outlier weights on the (full-resolution) input.
    5. Return restored image (int16, [0, 255]) and kernel.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# ── Framework base class import (DO NOT MODIFY) ─────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

from .solvers import blind_deconv, image_estimate


class OID_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution via Outlier Identification and Discarding (ECCV 2020).

    Parameters
    ----------
    kernel_size   : int — spatial support of the unknown PSF (square, odd).
    lambda_grad   : float — gradient-prior weight for both kernel estimation
                    and final non-blind deconvolution.
                    Default 4e-3 (from demo_deblurring.m).
    xk_iter       : int — inner (I, k) iterations per pyramid level (coarse).
                    Default 4.
    last_iter     : int — iterations at the finest pyramid level.
                    Default 4.  Increase for very noisy input.
    k_thresh      : float — final kernel threshold: entries < max(k)/k_thresh
                    are zeroed.  Default 20.
    isnoisy       : int — if 1, pre-smooth the image at coarse scales.
                    Default 1.
    predeblur     : str — 'L0' or 'Lp' image-update at coarse scales.
                    Default 'L0'.
    gamma_correct : float — gamma exponent applied before kernel estimation.
                    Default 1.0 (no correction).
    lambda_final  : float — gradient-prior weight used for the final
                    non-blind restoration on the full-size input.
                    Default 3e-3 (from demo).
    """

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

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────────
        # MATLAB demo: yg = im2double(rgb2gray(y)) for colour input
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 2:
            yg = y
        else:
            yg = y[:, :, 0]

        # ── 3. Multi-scale blind kernel estimation ──────────────────────
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

        # ── 4. Non-blind restoration on the full (possibly colour) input ─
        # MATLAB demo: Latent = image_estimate(y, kernel, 0.003, 0)
        Latent, _w_out = image_estimate(y, kernel, self.lambda_final, False)
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ───────────────────────────────────────────────────
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

    # ── Interface methods ────────────────────────────────────────────────
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
