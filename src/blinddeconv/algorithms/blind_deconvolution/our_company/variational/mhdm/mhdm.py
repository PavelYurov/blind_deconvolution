"""
Blind Image Deconvolution via Multiscale Hierarchical Decomposition (MHDM).

Framework wrapper for the MHDM algorithm described in:

    Wolf, T., Kindermann, S., Resmerita, E., Vese, L.
    "Applications of multiscale hierarchical decomposition to blind
    deconvolution." arXiv:2409.08734v5, 2025.

The method operates entirely in the Fourier domain, decomposing the
observed image into a sequence of increasingly fine-scale pairs
(image increment, kernel increment).  At each scale the regularisation
parameters are reduced by a factor of 4, yielding a multiscale
regularisation path.  The iterations stop once the discrepancy principle
||f - K_n * U_n||_{L^2} <= tau * delta  is satisfied, or a maximum
iteration count is reached.

Key mathematical ingredients
----------------------------
* Sobolev-type Fourier penalties  lambda * ||u||_{H^r}^2  and
  mu * ||k||_{H^s}^2  with discrete weights
  delta_{j,l} = 1 + 2m^2(1-cos(2pi j/m)) + 2n^2(1-cos(2pi l/n))
  (Justen, "Blind Deconvolution: Theory, Regularization and
  Applications", p. 110).
* Initial step:  closed-form thresholding in Fourier space
  (Theorem 3.5 in [1]).
* MHDM step:  pointwise degree-5 polynomial root-finding for the
  kernel, closed-form image update, Hermitian-symmetry enforcement
  (Theorem 4.3 / Algorithm 1 in [1]).
* Stopping:  L^2 discrepancy principle with noise level estimated
  via the Robust Median Estimator on Laplacian residuals
  (Donoho & Johnstone 1994), or user-supplied sigma.

Modules
-------
* ``utils``   — FFT helpers, Sobolev weights, conjugate-symmetry
                indices, noise estimation, PSF/OTF conversions.
* ``solvers`` — Pure functions for the initial step, iterative step,
                and full MHDM loop.
* ``mhdm``    — This file: framework-facing class ``MHDM``.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    compute_conjugate_indices,
    otf2psf,
    estimate_noise_sigma,
)
from .solvers import blind_deconvolution_mhdm

# Robust import of base class (identical pattern to ep_em.py)
import sys
import os
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


class MHDM(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using the Multiscale Hierarchical
    Decomposition Method (MHDM).

    Implementation follows Algorithm 2 of Wolf et al. (2025) [1]:

    1. **Initial step** — closed-form Fourier-domain thresholding
       (Theorem 3.5).
    2. **Iterative steps** — pointwise polynomial root-finding for the
       kernel, analytic image update, Hermitian-symmetry enforcement
       (Theorem 4.3).  Regularisation parameters lambda, mu are divided
       by 4 at every iteration.
    3. **Stopping** — L^2 discrepancy principle with estimated or
       user-supplied noise level.

    Parameters
    ----------
    kernel_shape : tuple of int
        Expected (height, width) of the blur kernel.
    lambda_0 : float
        Initial image regularisation parameter.
    mu_0 : float
        Initial kernel regularisation parameter.
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
    max_iter : int
        Maximum number of MHDM iterations.
    tol : float
        Numerical tolerance for polynomial root selection.
    verbose : bool
        If True, print per-iteration diagnostics.

    References
    ----------
    [1] Wolf, T., Kindermann, S., Resmerita, E., Vese, L.
        "Applications of multiscale hierarchical decomposition to blind
        deconvolution." arXiv:2409.08734v5, 2025.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_0: float = 1.4e-4,
        mu_0: float = 6.3e5,
        r: float = 1.0,
        s: float = 0.1,
        noise_sigma: float | None = None,
        tau: float = 1.001,
        max_iter: int = 30,
        tol: float = 1e-10,
        verbose: bool = False,
    ):
        super().__init__(name='MHDM-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.r = r
        self.s = s
        self.noise_sigma = noise_sigma
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Diagnostics filled by process()
        self.history: Dict[str, list] = {'residuals': [], 'iterations': 0}
        self.hyperparams: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Restore a blurred greyscale image and estimate the blur kernel.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            Observed blurred (and possibly noisy) image.
            Pixel values may be in [0, 255] or [0, 1].

        Returns
        -------
        restored : ndarray, shape (H, W), int16
            Restored image scaled to [0, 255].
        kernel : ndarray, shape kernel_shape
            Estimated PSF (spatial domain, normalised to sum 1).
        """
        start_time = time.time()

        # --- 1. Prepare data ------------------------------------------------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        H, W = y.shape
        kh, kw = self.kernel_shape

        # --- 2. Noise level & stopping threshold ----------------------------
        #     The MHDM stopping rule is the L^2 discrepancy principle
        #     ||f − K_n * U_n||_{L^2}  ≤  τ · δ,   δ = σ √(H·W).
        #     A too-small σ leads to a near-zero threshold, causing the
        #     algorithm to iterate with vanishing regularisation and
        #     produce divergent (saturated) reconstructions.
        if self.noise_sigma is not None:
            sigma = self.noise_sigma
        else:
            sigma = estimate_noise_sigma(y)          # floor = 0.005

        # L2 noise norm:  delta ≈ sigma * sqrt(H * W)
        delta = sigma * np.sqrt(H * W)
        stopping = self.tau * delta

        if self.verbose:
            print(f"[{self.name}] Start.  Img: {H}×{W},  Ker: {kh}×{kw}")
            print(f"  sigma={sigma:.6f},  delta={delta:.4f},  "
                  f"stopping={stopping:.4f}")
            print(f"  lambda_0={self.lambda_0:.4e},  mu_0={self.mu_0:.4e},  "
                  f"r={self.r},  s={self.s}")

        # --- 3. Precompute DFT & conjugate indices --------------------------
        f_four = np.fft.fft2(y)
        primary_idx, conjugate_idx = compute_conjugate_indices(H, W)

        # --- 4. Run the MHDM loop -------------------------------------------
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
                verbose=self.verbose,
            )

        elapsed = time.time() - start_time

        # --- 5. Extract the kernel (crop to kernel_shape) -------------------
        #     k_end_full is the fftshifted full-size PSF; extract the
        #     central (kh × kw) region.
        cy, cx = H // 2, W // 2
        top = cy - kh // 2
        left = cx - kw // 2
        kernel = k_end_full[top:top + kh, left:left + kw].copy()

        # Enforce non-negativity and normalise to sum 1
        kernel = np.maximum(kernel, 0.0)
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        # --- 6. Store diagnostics -------------------------------------------
        self.history = {
            'residuals': residuals,
            'iterations': its,
        }
        self.hyperparams = {
            'lambda_0': self.lambda_0,
            'mu_0': self.mu_0,
            'r': self.r,
            's': self.s,
            'noise_sigma': sigma,
            'tau': self.tau,
            'stopping_threshold': stopping,
            'final_residual': residuals[-1] if residuals else None,
            'iterations': its,
            'elapsed_seconds': elapsed,
        }

        if self.verbose:
            print(f"[{self.name}] Done.  {its} iters,  "
                  f"final residual={residuals[-1]:.6f},  "
                  f"time={elapsed:.2f}s")

        # --- 7. Scale output to int16 [0, 255] ------------------------------
        #     The MHDM reconstruction is not box-constrained; the
        #     Fourier-domain solution may produce values outside [0, 1].
        #     If the overshoot is small (< 10 % of the dynamic range),
        #     standard clipping is applied.  Otherwise, min–max
        #     rescaling is used to preserve contrast and avoid the
        #     uniform-white (saturated) appearance caused by hard
        #     clipping of heavily out-of-range data.
        u_min, u_max = float(u_end.min()), float(u_end.max())
        dynamic_range = u_max - u_min if u_max > u_min else 1.0
        overshoot = max(u_max - 1.0, 0.0) + max(0.0 - u_min, 0.0)

        if overshoot / dynamic_range < 0.10:
            # Small excursion — physics-aware clip to [0, 1]
            x_final = np.clip(u_end, 0.0, 1.0)
        else:
            # Significant excursion — rescale to preserve contrast
            x_final = (u_end - u_min) / dynamic_range
            if self.verbose:
                print(f"[{self.name}] Output rescaled: "
                      f"[{u_min:.3f}, {u_max:.3f}] → [0, 1]")

        x_final = x_final * 255.0
        x_final = np.round(x_final).astype(np.int16)

        return x_final, kernel

    # ------------------------------------------------------------------
    # Parameter interface (mirrors ep_em.py)
    # ------------------------------------------------------------------

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_0', self.lambda_0),
            ('mu_0', self.mu_0),
            ('r', self.r),
            ('s', self.s),
            ('noise_sigma', self.noise_sigma),
            ('tau', self.tau),
            ('max_iter', self.max_iter),
            ('tol', self.tol),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams


# ======================================================================
# Convenience runner (matches the pattern in ep_em.py)
# ======================================================================

def run_algorithm(g, kernel_shape, **kwargs):
    """
    Functional entry point for quick experiments.

    Parameters
    ----------
    g : ndarray
        Observed blurred image.
    kernel_shape : tuple of int
        Expected PSF size.
    **kwargs
        Forwarded to :class:`MHDM`.

    Returns
    -------
    f_est : ndarray
        Restored image (int16, [0, 255]).
    h_est : ndarray
        Estimated PSF.
    hyperparams : dict
    history : dict
    """
    algo = MHDM(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
