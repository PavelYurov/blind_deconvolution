"""
cpd.py

Fast Blind Image Deblurring Based on Cross Partial Derivative (CPD).

Reference:
    K.-C. Ting, S.-J. Wang, R.-B. Hwang: "Fast Blind Image Deblurring
    Based on Cross Partial Derivative", IEEE Transactions on Image
    Processing, vol. 34, pp. 8627-8640, 2025.
    DOI: 10.1109/TIP.2025.3645574

Pipeline (mirrors MATLAB Demo_CPD_v01.m):
    1. Normalise input to float64 [0, 1].
    2. Convert to grayscale for kernel estimation.
    3. Blind kernel estimation via CPD (estimate_kernel).
    4. Non-blind Tikhonov restoration on the full (colour) image
       via reconstruct_image.
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

from .solvers import estimate_kernel, reconstruct_image


class CPD_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Cross Partial Derivative (CPD).

    Parameters
    ----------
    kernel_size : int — estimated spatial support of the unknown PSF
                  (square). Default 25.
    cpd_sigma : float — σ of the Gaussian for CPD computation.
                Default 1.5.
    nms_sparsity : float — sparsity threshold for non-maximum suppression.
                   Default 50.
    cca_scale : float — threshold scale for connected component analysis.
                Default 0.15.
    cca_connect_type : int — connectivity for CCA (4 or 8). Default 8.
    resize_factor : float — resize factor for spectrum correlation step.
                    Default 0.5.
    corr_sigma : float — σ for Gaussian smoothing of spectra.
                 Default 2.0.
    smooth_blurred_image : str — 'Y' to smooth blurred image before
                           reconstruction, 'N' otherwise. Default 'N'.
    tikhonov_factor : float — regularisation parameter kH for Tikhonov
                      deconvolution. Default 0.002.
    zero_finding_distance : int — distance for zero-crossing detection
                            in the kernel FFT. Default 1.
    num_candidates : int — number of top kernel candidates to keep.
                     Default 5.
    index_ug : int — 0-based index of the chosen candidate. Default 0
               (= best by spectrum correlation).
    """

    def __init__(
        self,
        kernel_size: int = 25,
        cpd_sigma: float = 1.5,
        nms_sparsity: float = 50.0,
        cca_scale: float = 0.15,
        cca_connect_type: int = 8,
        resize_factor: float = 0.5,
        corr_sigma: float = 2.0,
        smooth_blurred_image: str = 'N',
        tikhonov_factor: float = 0.002,
        zero_finding_distance: int = 1,
        num_candidates: int = 5,
        index_ug: int = 0,
    ):
        super().__init__(name='CPD-BD')

        self.kernel_size = kernel_size
        self.cpd_sigma = cpd_sigma
        self.nms_sparsity = nms_sparsity
        self.cca_scale = cca_scale
        self.cca_connect_type = cca_connect_type
        self.resize_factor = resize_factor
        self.corr_sigma = corr_sigma
        self.smooth_blurred_image = smooth_blurred_image
        self.tikhonov_factor = tikhonov_factor
        self.zero_finding_distance = zero_finding_distance
        self.num_candidates = num_candidates
        self.index_ug = index_ug

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
        # MATLAB: b = rgb2gray(b_RGB)
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 3. Build opts dict for solvers ──────────────────────────────
        opts = {
            'kernel_size_est': self.kernel_size,
            'cpd_sigma': self.cpd_sigma,
            'nms_sparsity': self.nms_sparsity,
            'cca_scale': self.cca_scale,
            'cca_connect_type': self.cca_connect_type,
            'resize_factor': self.resize_factor,
            'corr_sigma': self.corr_sigma,
            'smooth_blurred_image': self.smooth_blurred_image,
            'tikhonov_factor': self.tikhonov_factor,
            'zero_finding_distance': self.zero_finding_distance,
        }

        # ── 4. Blind kernel estimation ──────────────────────────────────
        kernel, _candidates, run_time = estimate_kernel(
            yg, opts, self.num_candidates, self.index_ug
        )

        # ── 5. Non-blind Tikhonov restoration ──────────────────────────
        Latent = reconstruct_image(y, kernel, opts)
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 6. Output ──────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'cpd_sigma': self.cpd_sigma,
            'nms_sparsity': self.nms_sparsity,
            'cca_scale': self.cca_scale,
            'cca_connect_type': self.cca_connect_type,
            'resize_factor': self.resize_factor,
            'corr_sigma': self.corr_sigma,
            'smooth_blurred_image': self.smooth_blurred_image,
            'tikhonov_factor': self.tikhonov_factor,
            'zero_finding_distance': self.zero_finding_distance,
            'num_candidates': self.num_candidates,
            'index_ug': self.index_ug,
            'time': elapsed,
            'time_kernel_estimation': sum(run_time),
            'time_reconstruction': elapsed - sum(run_time),
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('cpd_sigma', self.cpd_sigma),
            ('nms_sparsity', self.nms_sparsity),
            ('cca_scale', self.cca_scale),
            ('cca_connect_type', self.cca_connect_type),
            ('resize_factor', self.resize_factor),
            ('corr_sigma', self.corr_sigma),
            ('smooth_blurred_image', self.smooth_blurred_image),
            ('tikhonov_factor', self.tikhonov_factor),
            ('zero_finding_distance', self.zero_finding_distance),
            ('num_candidates', self.num_candidates),
            ('index_ug', self.index_ug),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
