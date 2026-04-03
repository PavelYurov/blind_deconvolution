"""
sdp2.py

Blind Image Deconvolution via Convex Programming (Nuclear Norm Minimization).

Reference:
    A. Ahmed, B. Recht, J. Romberg: "Blind Deconvolution Using Convex
    Programming", IEEE Trans. Inform. Theory, 2014.

Pipeline (mirrors MATLAB blind2d.m):
    1. Normalise input to float64.
    2. Build subspace matrix B from blur kernel (non-zero entries).
    3. Build subspace matrix C from wavelet decomposition of blurred image.
    4. Lift to Fourier domain and build linear operator A.
    5. Solve  min ||X||_*  s.t. A·vec(X) = y_hat  via ADMM.
    6. SVD recovery: X → u (kernel coeffs), v (image coeffs).
    7. Wavelet reconstruction of the restored image.
    8. Return restored image (int16, [0, 255]) and recovered kernel.
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

from .solvers import blind_deconv_2d
from .utils import fspecial_motion


class SDP2_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution via convex programming (nuclear norm minimization).

    The algorithm lifts the bilinear blind deconvolution problem into a
    linear one over a rank-1 matrix X = h·m^T, where h are the kernel
    coefficients and m are the wavelet coefficients of the image.
    The nuclear norm ||X||_* is minimized subject to the linear
    measurement constraint A·vec(X) = y_hat (Fourier-domain).

    The solution is obtained via ADMM with Singular Value Thresholding.

    Parameters
    ----------
    kernel_size     : tuple (K1, K2) — spatial support of the blur PSF.
    motion_length   : int — length of motion blur in pixels.
                      Used to generate the initial kernel estimate via
                      fspecial('motion', length, angle).
    motion_angle    : float — angle of motion blur in degrees.
    wavelet_level   : int — wavelet decomposition depth (default 4).
    wavelet_name    : str — wavelet family (default 'db1' = Haar).
    threshold_ratio : float — wavelet coefficient threshold.
                      Coefficients with |c| > threshold_ratio * max(|c|)
                      are kept.  0.0 keeps all non-zero (default).
    admm_rho        : float — ADMM penalty parameter (default 1.0).
    admm_max_iter   : int — max ADMM iterations (default 500).
    verbose         : bool — print solver diagnostics (default False).
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 5),
        motion_length: int = 5,
        motion_angle: float = 45.0,
        wavelet_level: int = 4,
        wavelet_name: str = 'db1',
        threshold_ratio: float = 0.0,
        admm_rho: float = 1.0,
        admm_max_iter: int = 500,
        verbose: bool = False,
    ):
        super().__init__(name='SDP2-BD')

        self.kernel_size = kernel_size
        self.motion_length = motion_length
        self.motion_angle = motion_angle
        self.wavelet_level = wavelet_level
        self.wavelet_name = wavelet_name
        self.threshold_ratio = threshold_ratio
        self.admm_rho = admm_rho
        self.admm_max_iter = admm_max_iter
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blind deconvolution of a grayscale blurred image.

        Pipeline:
            1. Normalise to float64.
            2. Generate initial blur kernel estimate.
            3. Run convex blind deconvolution (nuclear norm minimization).
            4. Return restored image (int16, [0, 255]) and kernel.

        Parameters
        ----------
        image : (L1, L2) ndarray — blurred grayscale input image.

        Returns
        -------
        x_final : (L1, L2) int16 ndarray — restored image, [0, 255].
        kernel  : (K1, K2) float64 ndarray — recovered blur kernel.
        """
        start_time = time.time()

        # ── 1. Normalise to float64 ─────────────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # Handle colour images: convert to grayscale
        if y.ndim == 3 and y.shape[2] == 3:
            y = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3:
            y = y[:, :, 0]

        # Normalise by Frobenius norm (matching MATLAB blind2d.m)
        y_norm = np.linalg.norm(y, 'fro')
        if y_norm > 0:
            y /= y_norm

        # ── 2. Generate initial blur kernel ─────────────────────────────
        blur_kernel = fspecial_motion(self.motion_length, self.motion_angle)
        blur_kernel /= np.linalg.norm(blur_kernel, 'fro')

        # ── 3. Blind deconvolution via nuclear norm minimization ────────
        x_restored, kernel = blind_deconv_2d(
            blurred_image=y,
            blur_kernel=blur_kernel,
            wavelet_level=self.wavelet_level,
            wavelet_name=self.wavelet_name,
            threshold_ratio=self.threshold_ratio,
            admm_rho=self.admm_rho,
            admm_max_iter=self.admm_max_iter,
            use_dftmtx=False,
            verbose=self.verbose,
        )

        # ── 4. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'motion_length': self.motion_length,
            'motion_angle': self.motion_angle,
            'wavelet_level': self.wavelet_level,
            'wavelet_name': self.wavelet_name,
            'threshold_ratio': self.threshold_ratio,
            'admm_rho': self.admm_rho,
            'admm_max_iter': self.admm_max_iter,
            'time': time.time() - start_time,
        }

        # Scale back and convert to int16 [0, 255]
        x_restored = np.real(x_restored)
        # Renormalise: the restored image is in the normalised domain,
        # scale to [0, 1] range
        x_min = x_restored.min()
        x_max = x_restored.max()
        if x_max > x_min:
            x_restored = (x_restored - x_min) / (x_max - x_min)
        else:
            x_restored = np.zeros_like(x_restored)

        x_final = x_restored * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('motion_length', self.motion_length),
            ('motion_angle', self.motion_angle),
            ('wavelet_level', self.wavelet_level),
            ('wavelet_name', self.wavelet_name),
            ('threshold_ratio', self.threshold_ratio),
            ('admm_rho', self.admm_rho),
            ('admm_max_iter', self.admm_max_iter),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_size':
                    self.kernel_size = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
