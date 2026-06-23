"""
lowrank_li.py

Blind Image Deblurring with Low-Rank Kernel Regularisation.

Reference:
    Li Siyao, Shiyu Zhao, Wenzhe Wang, Ping Tan:
    "Understanding Kernel Size in Blind Deconvolution", WACV 2019.

Pipeline (mirrors MATLAB test.m):
    1. Normalise input to float64 [0, 1].
    2. Multi-scale blind deconvolution (multiscaled_cry) on Y channel.
    3. Return restored image (int16, [0, 255]) and kernel.
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

from .solvers import multiscaled_cry


class LowRankLi_BD(DeconvolutionAlgorithm):
    """
    Single-image blind deconvolution with low-rank kernel regularisation.

    Hyper-parameters (defaults from test.m):

    kernel_size : int   — spatial support of the unknown PSF (odd). Default 185.
    lambda_     : float — data-fidelity weight for x-step.             Default 80.
    sigma       : float — flag / weight for low-rank kernel reg.       Default 1.
    tx          : float — initial ISTA step size.                      Default 1e-2.
    tau         : float — SVT threshold.                               Default 1e-5.
    delta       : float — log-det stabiliser for SVT.                  Default 1e-5.
    imax        : int   — outer alternations per scale.                Default 5.
    ximax       : int   — outer ISTA iterations (x-step).              Default 2.
    xjmax       : int   — inner ISTA iterations (x-step).              Default 2.
    kmax        : int   — CG iterations (k-step).                      Default 3.
    rmax        : int   — SVT iterations per k-step.                   Default 3.
    mu          : float — proximity weight for k-step.                 Default 1.
    iterkrank   : int   — k+rank sub-iterations per alternation.       Default 10.
    threshold   : float — kernel threshold factor (frac of max).       Default 0.05.
    nb_lambda   : float — non-blind data-fidelity weight.              Default 3000.
    nb_alpha    : float — non-blind sparsity exponent.                 Default 1.
    init_kernel : str   — 'line' (MATLAB default, horizontal 2-px bias)
                          or 'uniform' (isotropic).                    Default 'line'.
    multiscale  : bool  — use coarse-to-fine pyramid. If False, estimate
                          directly at the final kernel size (avoids the
                          bell-shape artefact caused by repeated bilinear
                          upsampling of the kernel).                   Default True.
    kernel_upsample : str — 'bilinear' (MATLAB default) or 'nearest'
                            (preserves kernel sharpness between scales). Default 'bilinear'.
    sharpen_after_upsample : bool — after upscaling kernel between scales,
                            apply threshold + sum-to-one renormalisation
                            (counters bilinear-induced bell shape).    Default False.
    verbose     : bool  — print progress logs.                         Default False.
    """

    def __init__(
        self,
        kernel_size: int = 185,
        lambda_: float = 80.0,
        sigma: float = 1.0,
        tx: float = 1e-2,
        tau: float = 1e-5,
        delta: float = 1e-5,
        imax: int = 5,
        ximax: int = 2,
        xjmax: int = 2,
        kmax: int = 3,
        rmax: int = 3,
        mu: float = 1.0,
        iterkrank: int = 10,
        threshold: float = 0.05,
        nb_lambda: float = 3000.0,
        nb_alpha: float = 1.0,
        init_kernel: str = 'line',
        multiscale: bool = True,
        kernel_upsample: str = 'bilinear',
        sharpen_after_upsample: bool = False,
        verbose: bool = False,
    ):
        super().__init__(name='LowRankLi-BD')

        self.kernel_size = kernel_size
        self.lambda_ = lambda_
        self.sigma = sigma
        self.tx = tx
        self.tau = tau
        self.delta = delta
        self.imax = imax
        self.ximax = ximax
        self.xjmax = xjmax
        self.kmax = kmax
        self.rmax = rmax
        self.mu = mu
        self.iterkrank = iterkrank
        self.threshold = threshold
        self.nb_lambda = nb_lambda
        self.nb_alpha = nb_alpha
        self.init_kernel = init_kernel
        self.multiscale = multiscale
        self.kernel_upsample = kernel_upsample
        self.sharpen_after_upsample = sharpen_after_upsample
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

        # ── 2. Grayscale only ───────────────────────────────────────────────
        if y.ndim == 3:
            if y.shape[2] == 1:
                yg = y[:, :, 0]
            else:
                raise ValueError(
                    f'LowRankLi-BD is grayscale-only; got image with shape {image.shape}'
                )
        else:
            yg = y

        # ── 3. Ensure kernel_size is odd ─────────────────────────────────
        K = self.kernel_size
        if K % 2 == 0:
            K += 1

        # ── 4. Build params dict ────────────────────────────────────────
        params = {
            'lambda_': self.lambda_,
            'sigma': self.sigma,
            'tx': self.tx,
            'tau': self.tau,
            'delta': self.delta,
            'imax': self.imax,
            'ximax': self.ximax,
            'xjmax': self.xjmax,
            'kmax': self.kmax,
            'rmax': self.rmax,
            'mu': self.mu,
            'iterkrank': self.iterkrank,
            'threshold': self.threshold,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
            'init_kernel': self.init_kernel,
            'multiscale': self.multiscale,
            'kernel_upsample': self.kernel_upsample,
            'sharpen_after_upsample': self.sharpen_after_upsample,
        }

        # ── 5. Multi-scale blind + non-blind deconvolution ──────────────
        x, kernel = multiscaled_cry(yg, K, params, verbose=self.verbose)

        # ── 6. Clip & convert to int16 [0, 255] ─────────────────────────
        x_final = np.clip(x, 0.0, 1.0) * 255.0
        x_final = np.round(x_final).astype(np.int16)

        self.hyperparams = {
            'kernel_size': K,
            'lambda_': self.lambda_,
            'sigma': self.sigma,
            'tx': self.tx,
            'tau': self.tau,
            'delta': self.delta,
            'imax': self.imax,
            'ximax': self.ximax,
            'xjmax': self.xjmax,
            'kmax': self.kmax,
            'rmax': self.rmax,
            'mu': self.mu,
            'iterkrank': self.iterkrank,
            'threshold': self.threshold,
            'nb_lambda': self.nb_lambda,
            'nb_alpha': self.nb_alpha,
            'init_kernel': self.init_kernel,
            'multiscale': self.multiscale,
            'kernel_upsample': self.kernel_upsample,
            'sharpen_after_upsample': self.sharpen_after_upsample,
            'time': time.time() - start_time,
        }

        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_', self.lambda_),
            ('sigma', self.sigma),
            ('tx', self.tx),
            ('tau', self.tau),
            ('delta', self.delta),
            ('imax', self.imax),
            ('ximax', self.ximax),
            ('xjmax', self.xjmax),
            ('kmax', self.kmax),
            ('rmax', self.rmax),
            ('mu', self.mu),
            ('iterkrank', self.iterkrank),
            ('threshold', self.threshold),
            ('nb_lambda', self.nb_lambda),
            ('nb_alpha', self.nb_alpha),
            ('init_kernel', self.init_kernel),
            ('multiscale', self.multiscale),
            ('kernel_upsample', self.kernel_upsample),
            ('sharpen_after_upsample', self.sharpen_after_upsample),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
