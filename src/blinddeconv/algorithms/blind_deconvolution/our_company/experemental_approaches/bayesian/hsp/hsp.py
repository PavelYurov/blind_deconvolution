"""
hsp.py

Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior.

Reference:
    F. M. Castro-Macías, F. Pérez-Bueno, M. Vega, J. Mateos, R. Molina,
    A. K. Katsaggelos: "Bayesian Blind Image Deconvolution using a
    Hyperbolic-Secant Prior" (2024).

Pipeline (mirrors rest_oneGrayImageHSwopden.m):
    1. Normalise input to float64 [0, 1] grayscale.
    2. Coarse-to-fine variational blind kernel estimation
       (multi_stage_deconv) with the HS prior.
    3. Non-blind restoration via FIRLS + ADMM (frils_deb_ubc) using
       the estimated kernel.
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

from .utils import getfilters
from .solvers import multi_stage_deconv, frils_deb_ubc


class HSP_BD(DeconvolutionAlgorithm):
    """
    Hyperbolic-Secant Prior blind deconvolution.

    Parameters
    ----------
    kernel_size : (int, int)
        Final blur-kernel support (rows, cols), should be odd.
    alpha : sequence of float, optional
        Per-filter HS shape parameter.  Default ``(10**2.4, 10**2.15)``
        (matches MATLAB ``rest_someRealImagesHSwopden.m`` defaults).
    sigma2 : float
        Initial noise variance (default 1e-2, as in MATLAB).
    filters_name : {'fohv', 'fo', 'none'}
        High-pass filter bank.  Default 'fohv' (horiz./vert. first-order).
    max_iter : int
        Outer iterations per stage (MATLAB MAX_ITER, default 10).
    max_inner_iter : int
        Inner CG-loop iterations in update_xf_alpha (default 5).
    no_stages : int or None
        Number of pyramid levels (None → auto).

    FIRLS non-blind parameters (defaults from rest_oneGrayImageHSwopden.m):
        firls_out_iter=5, firls_inner_iter=4, firls_IF=sqrt(2),
        firls_lambda=2e-4, firls_lambda_u=0.1,
        firls_eps_min=2.55/255, firls_eps_max=2.55/255,
        firls_alpha=2/3.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (25, 25),
        alpha: Tuple[float, ...] = (10 ** 2.4, 10 ** 2.15),
        sigma2: float = 1e-2,
        filters_name: str = 'fohv',
        max_iter: int = 10,
        max_inner_iter: int = 5,
        no_stages: int = None,
        # FIRLS non-blind hyperparameters
        firls_out_iter: int = 5,
        firls_inner_iter: int = 4,
        firls_IF: float = float(np.sqrt(2.0)),
        firls_lambda: float = 2e-4,
        firls_lambda_u: float = 0.1,
        firls_eps_min: float = 2.55 / 255.0,
        firls_eps_max: float = 2.55 / 255.0,
        firls_alpha: float = 2.0 / 3.0,
    ):
        super().__init__(name='HS-BD')

        self.kernel_size = tuple(kernel_size)
        self.alpha = tuple(alpha)
        self.sigma2 = sigma2
        self.filters_name = filters_name
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.no_stages = no_stages

        self.firls_out_iter = firls_out_iter
        self.firls_inner_iter = firls_inner_iter
        self.firls_IF = firls_IF
        self.firls_lambda = firls_lambda
        self.firls_lambda_u = firls_lambda_u
        self.firls_eps_min = firls_eps_min
        self.firls_eps_max = firls_eps_max
        self.firls_alpha = firls_alpha

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1) Normalise to float64 [0, 1]
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y = y / 255.0

        # Grayscale only — pick channel 0 if multi-channel.
        if y.ndim == 3:
            y = y[:, :, 0]

        # 2) Validate filter bank ↔ alpha length consistency (MATLAB check)
        filters = getfilters(self.filters_name)
        if len(filters) != len(self.alpha):
            raise ValueError(
                f"Dimension mismatch: {len(filters)} filters vs "
                f"{len(self.alpha)} alpha values."
            )

        prior = {'name': 'Hysec', 'filter_name': 'None'}

        options = {
            'verbose': False,
            'tol': 1e-7,
            'UPDATE_NOISE': False,
            'SHOW_IMGS': False,
            'PROP_IMG_BET_STAGES': False,
            'PROP_IMG_WITHIN_STAGES': False,
            'MAX_ITER': int(self.max_iter),
            'MAX_INNER_ITER': int(self.max_inner_iter),
            'no_stages': self.no_stages,
        }

        # 3) Blind kernel estimation
        vars_, k_history = multi_stage_deconv(
            y,
            self.kernel_size,
            prior,
            filters,
            float(self.sigma2),
            self.alpha,
            options,
        )
        est_kernel = vars_['k']
        self.history['kernel_diff'] = k_history

        # 4) Non-blind restoration (FIRLS + ADMM)
        firls = {
            'out_iter': int(self.firls_out_iter),
            'inner_iter': int(self.firls_inner_iter),
            'IF': float(self.firls_IF),
            'lambda': float(self.firls_lambda),
            'lambda_u': float(self.firls_lambda_u),
            'epsilon_min': float(self.firls_eps_min),
            'epsilon_max': float(self.firls_eps_max),
            'alpha': float(self.firls_alpha),
            'beta_a': float(self.firls_lambda) * float(self.firls_alpha)
                      * (20.0 / 255.0) ** (float(self.firls_alpha) - 2.0),
            'cost_display': False,
            'isnr_display': False,
        }
        x_hat, _, _ = frils_deb_ubc(y, est_kernel, firls)

        # 5) Output
        x_hat = np.clip(x_hat, 0.0, 1.0) * 255.0
        x_hat = np.clip(x_hat, 0, 255).astype(np.int16)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'alpha': self.alpha,
            'sigma2': self.sigma2,
            'filters_name': self.filters_name,
            'max_iter': self.max_iter,
            'max_inner_iter': self.max_inner_iter,
            'no_stages': self.no_stages,
            'firls_lambda': self.firls_lambda,
            'firls_lambda_u': self.firls_lambda_u,
            'firls_alpha': self.firls_alpha,
            'time': time.time() - start_time,
        }
        return x_hat, est_kernel

    # ── Interface methods ───────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('alpha', self.alpha),
            ('sigma2', self.sigma2),
            ('filters_name', self.filters_name),
            ('max_iter', self.max_iter),
            ('max_inner_iter', self.max_inner_iter),
            ('no_stages', self.no_stages),
            ('firls_out_iter', self.firls_out_iter),
            ('firls_inner_iter', self.firls_inner_iter),
            ('firls_IF', self.firls_IF),
            ('firls_lambda', self.firls_lambda),
            ('firls_lambda_u', self.firls_lambda_u),
            ('firls_eps_min', self.firls_eps_min),
            ('firls_eps_max', self.firls_eps_max),
            ('firls_alpha', self.firls_alpha),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_size':
                    self.kernel_size = tuple(value)
                elif key == 'alpha':
                    self.alpha = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
