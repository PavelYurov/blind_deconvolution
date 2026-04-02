"""
map.py

MAP (Maximum A Posteriori) Blind Image Deconvolution.

Reference:
    O. Whyte, J. Sivic, A. Zisserman, and J. Ponce.
    "Non-uniform Deblurring for Shaken Images". IJCV, 2012.

    O. Whyte, J. Sivic and A. Zisserman.
    "Deblurring Shaken and Partially Saturated Images".
    In Proc. CPCV Workshop at ICCV, 2011.

Pipeline:
    1. Normalise input to float64 [0, 1].
    2. Build configuration from hyperparameters.
    3. Run blind_deblur_map (coarse-to-fine MAP estimation).
    4. Final non-blind deconvolution on colour image.
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

from .solvers import blind_deblur_map, fast_deconv
from .utils import default_config


class MAP_BD(DeconvolutionAlgorithm):
    """
    MAP blind deconvolution (Whyte et al. IJCV 2012 / CPCV 2011).

    Coarse-to-fine kernel estimation with edge prediction
    (bilateral + shock filter + gradient thresholding) and
    non-blind deconvolution (Krishnan & Fergus / CG / sparse IRLS).

    Parameters
    ----------
    blur_kernel_size : int — spatial support of the unknown PSF (odd).
    kernel_method    : str — kernel estimation method:
                       'lars', 'lars_ols', 'conjgrad'. Default 'lars'.
    image_method     : str — non-blind deconvolution method:
                       'krishnan', 'conjgrad', 'sparse'. Default 'krishnan'.
    alpha            : float — regularization weight for image estimation.
                       Default 50.
    beta             : float — regularization weight for kernel estimation
                       (LARS lambda). Default 1.0.
    kf_lambda        : float — Krishnan & Fergus data-fidelity weight.
                       Default 3000.
    kf_exponent      : float — hyper-Laplacian exponent.
                       Default 0.5 (= Laplacian).
    kernel_threshold : float — kernel thresholding ratio.
                       Values < max(k)/kernel_threshold are zeroed.
                       Default 20.
    scale_ratio_k    : float — kernel pyramid downsampling ratio.
                       Default 1/sqrt(2).
    scale_ratio_i    : float — image pyramid downsampling ratio.
                       Default 1/sqrt(2).
    num_iters        : int — iterations per pyramid level.
                       Default 3.
    sat_thresh       : float — saturation threshold.
                       Default 0.98.
    verbose          : bool — print progress. Default False.
    """

    def __init__(
        self,
        blur_kernel_size: int = 25,
        kernel_method: str = 'lars',
        image_method: str = 'krishnan',
        alpha: float = 50.0,
        beta: float = 1.0,
        kf_lambda: float = 3000.0,
        kf_exponent: float = 0.5,
        kernel_threshold: float = 20.0,
        scale_ratio_k: float = None,
        scale_ratio_i: float = None,
        num_iters: int = 3,
        sat_thresh: float = 0.98,
        verbose: bool = False,
    ):
        super().__init__(name='MAP-BD')

        self.blur_kernel_size = blur_kernel_size
        self.kernel_method = kernel_method
        self.image_method = image_method
        self.alpha = alpha
        self.beta = beta
        self.kf_lambda = kf_lambda
        self.kf_exponent = kf_exponent
        self.kernel_threshold = kernel_threshold
        self.scale_ratio_k = scale_ratio_k if scale_ratio_k is not None \
            else 1.0 / np.sqrt(2)
        self.scale_ratio_i = scale_ratio_i if scale_ratio_i is not None \
            else 1.0 / np.sqrt(2)
        self.num_iters = num_iters
        self.sat_thresh = sat_thresh
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Build config ─────────────────────────────────────────────
        cfg = default_config(self.blur_kernel_size)
        cfg['kernel_method'] = self.kernel_method
        cfg['image_method'] = self.image_method
        cfg['alpha'] = self.alpha
        cfg['beta'] = self.beta
        cfg['kf_lambda'] = self.kf_lambda
        cfg['kf_exponent'] = self.kf_exponent
        cfg['kernel_threshold'] = self.kernel_threshold
        cfg['scale_ratio_k'] = self.scale_ratio_k
        cfg['scale_ratio_i'] = self.scale_ratio_i
        cfg['num_iters'] = self.num_iters
        cfg['sat_thresh'] = self.sat_thresh

        # ── 3. Blind deconvolution ──────────────────────────────────────
        L_est, kernel, hist = blind_deblur_map(y, cfg, verbose=self.verbose)

        # ── 4. Final non-blind deconvolution on colour image ────────────
        if self.image_method == 'krishnan':
            L_final = fast_deconv(y, kernel, self.kf_lambda, self.kf_exponent)
        else:
            L_final = L_est
            # For colour: apply channel-by-channel if needed
            if y.ndim == 3 and L_final.ndim == 2:
                from .solvers import deconv_sparse, deconv_L2_grad_data
                if self.image_method == 'sparse':
                    L_final = deconv_sparse(y, kernel, self.alpha)
                else:
                    L_final, pad = deconv_L2_grad_data(
                        y, kernel, self.alpha, 200)
                    from .utils import pad_image
                    L_final = pad_image(L_final, -pad)

        L_final = np.clip(L_final, 0.0, 1.0)

        # ── 5. Store history & output ───────────────────────────────────
        self.history = hist
        self.hyperparams = {
            'blur_kernel_size': self.blur_kernel_size,
            'kernel_method': self.kernel_method,
            'image_method': self.image_method,
            'alpha': self.alpha,
            'beta': self.beta,
            'kf_lambda': self.kf_lambda,
            'kf_exponent': self.kf_exponent,
            'kernel_threshold': self.kernel_threshold,
            'time': time.time() - start_time,
        }

        x_final = L_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('blur_kernel_size', self.blur_kernel_size),
            ('kernel_method', self.kernel_method),
            ('image_method', self.image_method),
            ('alpha', self.alpha),
            ('beta', self.beta),
            ('kf_lambda', self.kf_lambda),
            ('kf_exponent', self.kf_exponent),
            ('kernel_threshold', self.kernel_threshold),
            ('scale_ratio_k', self.scale_ratio_k),
            ('scale_ratio_i', self.scale_ratio_i),
            ('num_iters', self.num_iters),
            ('sat_thresh', self.sat_thresh),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
