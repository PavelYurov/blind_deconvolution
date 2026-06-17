"""
esm.py

Blind Image Deblurring with the Enhanced Sparse Model (ESM).

Reference:
    L. Chen, F. Fang, S. Lei, F. Li, G. Zhang: "Enhanced Sparse Model
    for Blind Deblurring", ECCV 2020.

Pipeline (mirrors MATLAB demo_deblurring.m):
    1. Normalise input to float64 [0, 1].
    2. Convert to grayscale for kernel estimation (if colour).
    3. Multi-scale blind deconvolution (blind_deconv) — produces the PSF.
    4. Non-blind restoration on the full image via
       ringing_artifacts_removal (TV-ℓ² + L0 + bilateral-filter ringing
       subtraction).
    5. Return restored image (int16, [0, 255]) and the PSF.
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

from .solvers import blind_deconv, ringing_artifacts_removal


class ESM_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Enhanced Sparse Model (ECCV 2020).

    Parameters
    ----------
    kernel_size   : int — spatial support of the unknown PSF (square, odd).
                    Default 35 (as in demo_deblurring.m).
    lambda_data   : float — weight of the ℓ0−ℓ1 prior on the data-gradient
                    residual  k*∇I − ∇B.  Default 4e-3.
    lambda_grad   : float — weight of the ℓ0−ℓ1 prior on ∇I.  Default 4e-3.
    theta         : float — θ parameter of the ℓ0−ℓ1 enhanced-sparse prior
                    (controls the shrinkage zone width).  Default 1.0.
    xk_iter       : int — inner I/k alternations per pyramid level.
                    Default 5.
    gamma_correct : float — gamma correction applied before kernel
                    estimation.  1.0 = no correction.  Default 1.0.
    k_thresh      : float — final kernel threshold.  Entries below
                    max(k)/k_thresh are zeroed.  Default 20.
    saturation    : bool — if True (saturated input), use the L0 + TV +
                    bilateral ring-removal pipeline for the non-blind step.
                    If False, this wrapper still uses the same
                    ringing_artifacts_removal call but the TV-only branch
                    is selected by setting weight_ring=0 externally.
                    Default False.
    lambda_tv     : float — TV weight for the non-blind step.  Default 0.002.
    lambda_l0     : float — L0 weight for the non-blind step.  Default 2e-4.
    weight_ring   : float — ringing suppression weight (0 = TV only).
                    Default 1.0.
    """

    def __init__(
        self,
        kernel_size: int = 35,
        lambda_data: float = 4e-3,
        lambda_grad: float = 4e-3,
        theta: float = 1.0,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.002,
        lambda_l0: float = 2e-4,
        weight_ring: float = 1.0,
    ):
        super().__init__(name='ESM-BD')

        self.kernel_size = kernel_size
        self.lambda_data = lambda_data
        self.lambda_grad = lambda_grad
        self.theta = theta
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

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
        # MATLAB: yg = im2double(rgb2gray(y))
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 2:
            yg = y.copy()
        else:
            yg = y[:, :, 0]

        # Ensure kernel_size is odd (matches MATLAB convention)
        ks = int(self.kernel_size)
        if ks % 2 == 0:
            ks += 1

        # ── 3. Blind kernel estimation (ECCV20 Algorithm 1) ─────────────
        opts = {
            'kernel_size': ks,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'theta': self.theta,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_data, self.lambda_grad, opts
        )

        # ── 4. Non-blind restoration on the full-resolution image ───────
        # MATLAB: Latent = ringing_artifacts_removal(y, kernel, lambda_tv,
        #                                            lambda_l0, weight_ring)
        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': ks,
            'lambda_data': self.lambda_data,
            'lambda_grad': self.lambda_grad,
            'theta': self.theta,
            'xk_iter': self.xk_iter,
            'gamma_correct': self.gamma_correct,
            'k_thresh': self.k_thresh,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_data', self.lambda_data),
            ('lambda_grad', self.lambda_grad),
            ('theta', self.theta),
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
