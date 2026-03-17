"""
pmp.py

Blind Image Deblurring Using Patch-wise Minimal Pixels (PMP) Prior.

Reference:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong:
    "A Simple Local Minimal Intensity Prior and An Improved Algorithm
    for Blind Image Deblurring", IEEE TCSVT, 2021.

Pipeline (mirrors MATLAB demo_samples.m):
    1. Normalise input to float64 [0, 1].
    2. Multi-scale blind deconvolution (blind_deconv) on grayscale input.
    3. Non-blind restoration via ringing_artifacts_removal.
    4. Return restored image (int16, [0, 255]) and kernel.
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


class PMP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Patch-wise Minimal Pixels (PMP) prior.

    Parameters
    ----------
    kernel_size   : int — spatial support of the unknown PSF (square, odd).
    lambda_pmp    : float — weight for L0 intensity (PMP) prior.
                    Default 0.1 (from demo_samples.m).
    lambda_grad   : float — weight for L0 gradient prior.
                    Default 4e-3.
    xk_iter       : int — number of blind iterations per pyramid level.
                    Default 5.
    gamma_correct : float — gamma correction exponent applied before
                    kernel estimation.  1.0 = no correction.  Default 1.0.
    k_thresh      : float — final kernel threshold.
                    kernel values < max(k)/k_thresh are zeroed.
                    Default 20.
    patch_r       : int or None — patch size for PMP prior.
                    None = auto (floor(0.025 * mean(image_size))).
                    Default None.
    lambda_tv     : float — weight for TV non-blind deconvolution.
                    Default 0.001.
    lambda_l0     : float — weight for L0 non-blind deconvolution.
                    Default 5e-4.
    weight_ring   : float — ringing suppression weight (0 = no suppression).
                    Default 1.0.
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

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────────
        # MATLAB: yg = im2double(rgb2gray(y))
        # The user specifies: process() takes 1 grayscale image.
        # Handle both 2D and 3D (single-channel) inputs gracefully.
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 3. Blind kernel estimation ──────────────────────────────────
        # MATLAB: [kernel, interim_latent] = blind_deconv(yg, lambda, lambda_grad, opts)
        # or:     [kernel, interim_latent] = blind_deconv(yg, lambda, lambda_grad, opts, patch_r)
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

        # ── 4. Non-blind restoration ────────────────────────────────────
        # MATLAB: Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ──────────────────────────────────────────────────
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

    # ── Interface methods ────────────────────────────────────────────────
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
