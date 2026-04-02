"""
bcsnsp_sr.py

Bayesian Combination of Sparse and Non-Sparse Priors Super-Resolution.

References:
    [1] S. D. Babacan, R. Molina, A. K. Katsaggelos,
        "Bayesian Super Resolution Image Reconstruction using an l1 Prior",
        ISPA 2009 / Chapter in Bayesian Inference, 2011.
    [2] J. Salvador, S. Villena, R. Molina, A. K. Katsaggelos,
        "Bayesian Combination of Sparse and Non-Sparse Priors in
        Image Super Resolution", Digital Signal Processing, 2013.

Pipeline:
    1. Take input image (treated as HR original).
    2. Simulate L low-resolution observations via ``create_data``.
    3. Run ``solvex_var_l4_sar`` to reconstruct.
    4. Return (enhanced_int16, dummy 3×3 kernel).
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

from .solvers import create_data, solvex_var_l4_sar
from .utils import fspecial_gaussian

from scipy.ndimage import shift as _ndshift


class BCSNSP_SR(DeconvolutionAlgorithm):
    """
    Super-resolution via Bayesian Combination of Sparse and Non-Sparse Priors.

    Modes
    -----
    ``'upscale'``  (default) — real upscaling.
        Input : LR image  (m × n).
        Output: HR image  (m*res × n*res).
        Generates L pseudo-frames from the single input via sub-pixel shifts,
        then reconstructs a larger HR image.

    ``'benchmark'`` — simulation / self-test.
        Input : HR image  (M × N).
        Output: HR image  (M × N)  — same size.
        Degrades the HR input into LR frames, then restores it.

    Parameters
    ----------
    res           : int   — magnification factor.
    L             : int   — number of (simulated) LR frames.
    sigma         : float — assumed observation noise σ.
    blur_size     : int   — PSF kernel size (odd).
    blur_sigma    : float — PSF Gaussian σ.
    lambda_prior  : float — trade-off L1-TV vs SAR, in [0, 1].
    maxit         : int   — max SR iterations.
    thr           : float — convergence threshold.
    method        : str   — ``'variational'`` or ``'degenerate'``.
    estimate_reg  : bool  — update registration per iteration.
    max_shift     : float — max sub-pixel shift (in LR pixels for upscale,
                            in HR pixels for benchmark).
    max_theta     : float — max random rotation (rad) for benchmark mode.
    pcg_thr       : float — PCG solver tolerance.
    pcg_maxit     : int   — PCG max iterations.
    pcg_minit     : int   — PCG min iterations.
    mode          : str   — ``'upscale'`` or ``'benchmark'``.
    verbose       : bool  — print iteration info.
    seed          : int or None — random seed for reproducibility.
    """

    def __init__(
        self,
        res: int = 2,
        L: int = 4,
        sigma: float = 0.01,
        blur_size: int = 3,
        blur_sigma: float = 0.5,
        lambda_prior: float = 0.5,
        maxit: int = 30,
        thr: float = 1e-4,
        method: str = 'variational',
        estimate_reg: bool = True,
        max_shift: float = 0.5,
        max_theta: float = 0.01,
        pcg_thr: float = 1e-6,
        pcg_maxit: int = 100,
        pcg_minit: int = 10,
        mode: str = 'upscale',
        verbose: bool = False,
        seed: int | None = None,
    ):
        super().__init__(name='BCSNSP-SR')

        self.res = res
        self.L = L
        self.sigma = sigma
        self.blur_size = blur_size
        self.blur_sigma = blur_sigma
        self.lambda_prior = lambda_prior
        self.maxit = maxit
        self.thr = thr
        self.method = method
        self.estimate_reg = estimate_reg
        self.max_shift = max_shift
        self.max_theta = max_theta
        self.pcg_thr = pcg_thr
        self.pcg_maxit = pcg_maxit
        self.pcg_minit = pcg_minit
        self.mode = mode
        self.verbose = verbose
        self.seed = seed

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        if self.seed is not None:
            np.random.seed(self.seed)

        # ── 1. Normalise to float64 [0, 1] grayscale ────────────────────
        img = image.astype(np.float64)
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + \
                      0.1140 * img[:, :, 2]
        if img.max() > 1.0:
            img /= 255.0

        h = fspecial_gaussian(self.blur_size, self.blur_sigma)

        if self.mode == 'benchmark':
            x_vec, out, M, N = self._process_benchmark(img, h)
        else:
            x_vec, out, M, N = self._process_upscale(img, h)

        # ── Output ───────────────────────────────────────────────────────
        x_img = x_vec.reshape(M, N, order='F')
        x_img = np.clip(x_img, 0.0, 1.0)

        self.hyperparams = {
            'res': self.res,
            'L': self.L,
            'sigma': self.sigma,
            'lambda_prior': self.lambda_prior,
            'method': self.method,
            'mode': self.mode,
            'iterations': out['iterations'],
            'input_shape': img.shape,
            'output_shape': (M, N),
            'xconv': out['xconv'],
            'time': time.time() - start_time,
        }
        if out['history']['PSNRs']:
            self.hyperparams['final_psnr'] = out['history']['PSNRs'][-1]
        self.history = out['history']

        x_final = (x_img * 255.0).clip(0, 255).astype(np.int16)
        kernel = np.zeros((3, 3), dtype=np.float64)
        return x_final, kernel

    # ── Upscale mode (real SR: input LR → output HR bigger) ─────────────
    def _process_upscale(self, img, h):
        """
        Input: single LR image (m × n).
        Creates L pseudo-frames via sub-pixel shifts.
        Output: HR image vector of size (m*res × n*res).
        """
        m, n = img.shape
        M = m * self.res
        N = n * self.res

        # Generate L frames from single input via sub-pixel shifts
        # Frame 0 = original, frames 1..L-1 = shifted copies
        sx_lr = np.zeros(self.L)   # shifts in LR pixel space
        sy_lr = np.zeros(self.L)

        frames_vec = [img.ravel(order='F')]
        for k in range(1, self.L):
            sx_lr[k] = (np.random.rand() * 2 - 1) * self.max_shift
            sy_lr[k] = (np.random.rand() * 2 - 1) * self.max_shift
            # scipy.ndimage.shift takes [row_shift, col_shift] = [dy, dx]
            shifted = _ndshift(img, [sy_lr[k], sx_lr[k]],
                               order=1, mode='reflect')
            frames_vec.append(shifted.ravel(order='F'))

        y = np.concatenate(frames_vec)

        # Convert LR-pixel shifts to HR-pixel shifts for the solver
        sx_hr = sx_lr * self.res
        sy_hr = sy_lr * self.res
        theta_hr = np.zeros(self.L)

        x_vec, out = solvex_var_l4_sar(
            y, M=M, N=N, m=m, n=n, res=self.res, L=self.L, h=h,
            sx=sx_hr, sy=sy_hr, theta=theta_hr,
            xtrue=None,
            method=self.method,
            lambda_prior=self.lambda_prior,
            maxit=self.maxit,
            thr=self.thr,
            pcg_thr=self.pcg_thr,
            pcg_maxit=self.pcg_maxit,
            pcg_minit=self.pcg_minit,
            estimate_registration=False,  # shifts are known exactly
            verbose=self.verbose,
        )
        return x_vec, out, M, N

    # ── Benchmark mode (simulation: input HR → degrade → restore HR) ────
    def _process_benchmark(self, img, h):
        """
        Input: HR image (M × N).
        Degrades into L LR frames, then reconstructs.
        Output: HR image vector (same M × N size).
        """
        M_raw, N_raw = img.shape
        m = M_raw // self.res
        n = N_raw // self.res
        M = m * self.res
        N = n * self.res
        img = img[:M, :N]

        sx_true = np.zeros(self.L)
        sy_true = np.zeros(self.L)
        theta_true = np.zeros(self.L)
        for k in range(1, self.L):
            sx_true[k] = (np.random.rand() * 2 - 1) * self.max_shift
            sy_true[k] = (np.random.rand() * 2 - 1) * self.max_shift
            theta_true[k] = (np.random.rand() * 2 - 1) * self.max_theta

        y, _W = create_data(img, h, M, N, self.res, self.L,
                            sx_true, sy_true, theta_true, self.sigma)

        sx_init = sx_true.copy()
        sy_init = sy_true.copy()
        theta_init = theta_true.copy()
        for k in range(1, self.L):
            sx_init[k] += np.random.randn() * 0.1 * self.max_shift
            sy_init[k] += np.random.randn() * 0.1 * self.max_shift
            theta_init[k] += np.random.randn() * 0.1 * self.max_theta

        x_vec, out = solvex_var_l4_sar(
            y, M=M, N=N, m=m, n=n, res=self.res, L=self.L, h=h,
            sx=sx_init, sy=sy_init, theta=theta_init,
            sx_init=sx_init.copy(), sy_init=sy_init.copy(),
            theta_init=theta_init.copy(),
            xtrue=img,
            method=self.method,
            lambda_prior=self.lambda_prior,
            maxit=self.maxit,
            thr=self.thr,
            pcg_thr=self.pcg_thr,
            pcg_maxit=self.pcg_maxit,
            pcg_minit=self.pcg_minit,
            estimate_registration=self.estimate_reg,
            verbose=self.verbose,
        )
        return x_vec, out, M, N

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('res', self.res),
            ('L', self.L),
            ('sigma', self.sigma),
            ('blur_size', self.blur_size),
            ('blur_sigma', self.blur_sigma),
            ('lambda_prior', self.lambda_prior),
            ('maxit', self.maxit),
            ('thr', self.thr),
            ('method', self.method),
            ('estimate_reg', self.estimate_reg),
            ('max_shift', self.max_shift),
            ('max_theta', self.max_theta),
            ('pcg_thr', self.pcg_thr),
            ('pcg_maxit', self.pcg_maxit),
            ('pcg_minit', self.pcg_minit),
            ('mode', self.mode),
            ('verbose', self.verbose),
            ('seed', self.seed),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
