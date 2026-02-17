"""
Blind Image Deconvolution via the Multiscale Hierarchical Decomposition
Method (MHDM).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    compute_conjugate_indices,
    estimate_noise_sigma,
)
from .solvers import blind_deconvolution_mhdm

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root "
                               "(no pyproject.toml found)")
        path = path.parent
    return path


_CURRENT_FILE  = Path(__file__).resolve()
_PROJECT_ROOT  = _find_project_root(_CURRENT_FILE)
_SRC_DIR       = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm

class MHDM(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using the Multiscale Hierarchical
    Decomposition Method (MHDM).

    Parameters:
    kernel_shape : tuple of int
        Expected (height, width) of the blur kernel.
    lambda_0 : float
        Initial image regularisation parameter.  Default from
        the MATLAB reference ``test_noisy.m``:  1.4e-4.
    mu_0 : float
        Initial kernel regularisation parameter.  Default: 6.3e5.
    r : float
        Sobolev exponent for the image penalty (default 1.0).
    s : float
        Sobolev exponent for the kernel penalty (default 0.1).
    noise_sigma : float or None
        Standard deviation of additive Gaussian noise.
        If ``None``, estimated automatically via the Robust Median
        Estimator on the Laplacian of the input.
    tau : float
        Safety factor for the discrepancy principle (>= 1).
        Default 1.001 (from ``test_noisy.m``).
    max_iter : int
        Maximum number of MHDM iterations.
    tol : float
        Numerical tolerance for polynomial root selection.
    verbose : bool
        If True, print per-iteration diagnostics.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_0: float = 1.0e-6,
        mu_0: float = 4000.0,
        r: float = 1.0,
        s: float = 0.5,
        noise_sigma: float | None = None,
        tau: float = 1.001,
        max_iter: int = 30,
        tol: float = 1e-10,
        verbose: bool = False,
    ):
        super().__init__(name='MHDM-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_0     = lambda_0
        self.mu_0         = mu_0
        self.r            = r
        self.s            = s
        self.noise_sigma  = noise_sigma
        self.tau          = tau
        self.max_iter     = max_iter
        self.tol          = tol
        self.verbose      = verbose
        self.history: Dict[str, Any]    = {'residuals': [], 'iterations': 0}
        self.hyperparams: Dict[str, Any] = {}

    def _apply_edgetaper(self, img: np.ndarray, n_taper: int = 8) -> np.ndarray:
        """
        Apply a simple boundary tapering to reduce FFT ringing artifacts (cross).
        Smoothly blends the image borders to the image mean.
        """
        h, w = img.shape
        if h < 2 * n_taper or w < 2 * n_taper:
            return img 

        idx = np.arange(n_taper)
        ramp = 0.5 * (1 - np.cos(np.pi * (idx + 1) / (n_taper + 1)))

        mask_h = np.ones(h)
        mask_h[:n_taper] = ramp
        mask_h[-n_taper:] = ramp[::-1]

        mask_w = np.ones(w)
        mask_w[:n_taper] = ramp
        mask_w[-n_taper:] = ramp[::-1]

        mask = np.outer(mask_h, mask_w)

        mean_val = np.mean(img)
        return img * mask + mean_val * (1.0 - mask)

    def _apply_edgetaper(self, image: np.ndarray, n_taper: int = 15) -> np.ndarray:
        """
        Apply a simple cosine window taper to the image borders to reduce
        boundary artifacts in the FFT.
        """
        H, W = image.shape
        taper = np.ones((H, W), dtype=np.float64)

        cos_decay = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_taper) / (n_taper - 1)))

        for i in range(n_taper):
            taper[i, :] *= cos_decay[i]
            taper[H - 1 - i, :] *= cos_decay[i]

        for j in range(n_taper):
            taper[:, j] *= cos_decay[j]
            taper[:, W - 1 - j] *= cos_decay[j]

        mean_val = np.mean(image)
        return (image - mean_val) * taper + mean_val

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        H, W = y.shape
        kh, kw = self.kernel_shape

        if H > 32 and W > 32:
             y = self._apply_edgetaper(y)

        if self.noise_sigma is not None:
            sigma = self.noise_sigma
        else:
            sigma = estimate_noise_sigma(y, sigma_floor=2.0 / 255.0)

        delta = sigma * np.sqrt(H * W)
        stopping = self.tau * delta

        if self.verbose:
            print(f"[{self.name}] Start.  Img: {H}x{W},  Ker: {kh}x{kw}")
            print(f"  sigma={sigma:.6f},  delta={delta:.4f},  "
                  f"stopping={stopping:.4f}")
            print(f"  lambda_0={self.lambda_0:.4e},  mu_0={self.mu_0:.4e},  "
                  f"r={self.r},  s={self.s}")

        f_four = np.fft.fft2(y)
        primary_idx, conjugate_idx, self_conj_idx = \
            compute_conjugate_indices(H, W)

        u_end, k_end_full, u_list, k_list, its, residuals = \
            blind_deconvolution_mhdm(
                f=y,
                f_four=f_four,
                lambda_0=self.lambda_0,
                mu_0=self.mu_0,
                r=self.r,
                s=self.s,
                tol=self.tol,
                stopping=stopping,
                maxits=self.max_iter,
                primary_idx=primary_idx,
                conjugate_idx=conjugate_idx,
                self_conj_idx=self_conj_idx,
                verbose=self.verbose,
            )

        elapsed = time.time() - start_time

        cy, cx = H // 2, W // 2
        top  = cy - kh // 2
        left = cx - kw // 2
        kernel = k_end_full[top:top + kh, left:left + kw].copy()

        kernel = np.maximum(kernel, 0.0)
        k_sum  = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        self.history = {
            'residuals':  residuals,
            'iterations': its,
        }
        self.hyperparams = {
            'lambda_0':          self.lambda_0,
            'mu_0':              self.mu_0,
            'r':                 self.r,
            's':                 self.s,
            'noise_sigma':       sigma,
            'tau':               self.tau,
            'stopping_threshold': stopping,
            'final_residual':    residuals[-1] if residuals else None,
            'iterations':        its,
            'elapsed_seconds':   elapsed,
        }

        if self.verbose:
            print(f"[{self.name}] Done.  {its} iters,  "
                  f"final residual={residuals[-1]:.6f},  "
                  f"time={elapsed:.2f}s")

        u_end = u_end * 255.0
        x_final = np.clip(u_end, 0.0, 255.0)
        x_final = np.round(x_final).astype(np.int16)

        return x_final, kernel


    def get_param(self) -> List[Tuple[str, Any]]:
        """Return current algorithm hyper-parameters."""
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_0',     self.lambda_0),
            ('mu_0',         self.mu_0),
            ('r',            self.r),
            ('s',            self.s),
            ('noise_sigma',  self.noise_sigma),
            ('tau',          self.tau),
            ('max_iter',     self.max_iter),
            ('tol',          self.tol),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        """Update algorithm hyper-parameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        """Return per-iteration diagnostics from the last ``process`` call."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Return the hyper-parameter snapshot from the last ``process`` call."""
        return self.hyperparams
