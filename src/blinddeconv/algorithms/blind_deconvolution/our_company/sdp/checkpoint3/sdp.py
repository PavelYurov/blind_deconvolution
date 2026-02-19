"""
Steepest Descent on Quotient Manifold (SDP) — Blind Image Deconvolution.

Framework-compatible wrapper for the SDP blind deconvolution algorithm.

References
----------
[1] Zeng, So, Gillis. "Blind Deconvolution by a Steepest Descent Algorithm
    on a Quotient Manifold," 2018.
[2] Barzilai, Borwein. "Two-Point Step Size Gradient Methods."
    IMA J. Numer. Anal., 8(1):141–148, 1988.
[3] Ahmed, Recht, Romberg. "Blind Deconvolution using Convex Programming."
    IEEE Trans. Inform. Theory, 60(3):1711–1732, 2014. arXiv:1211.5608.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .solvers import sdp_multiscale, sdp_single_scale, refine_image_non_blind
from .utils import init_gaussian_kernel, edge_taper

# ── Robust import of the framework base class ──
import sys
import os
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up from *start* until a directory containing pyproject.toml is found."""
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


class SDP(DeconvolutionAlgorithm):
    """
    Steepest Descent on Quotient Manifold for Blind Image Deconvolution.

    Jointly estimates a blur kernel  h  and a latent sharp image  x  from
    a single blurred greyscale observation  y ≈ h ⊛ x.

    The algorithm performs Riemannian steepest descent on the quotient
    manifold  M / G = (R^m × R^n) / R_+  formed by quotienting out the
    inherent scaling ambiguity  (h, x) ~ (α h, x / α),  with
    Barzilai–Borwein adaptive step sizes [2].

    Pipeline (Ref: [1], Algorithm 1 + Section 4):
        1. Multi-scale (coarse-to-fine) blind estimation.
        2. At each scale: joint (h, x) descent with horizontal projection.
        3. Final non-blind deconvolution with the estimated kernel.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 0.002,
        max_iter: int = 200,
        num_scales: int = 4,
        tol: float = 1e-6,
        kernel_threshold: float = 0.02,
        multiscale: bool = True,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        kernel_shape : (kh, kw)
            Expected blur-kernel size in pixels.
        lambda_tv : float
            Total Variation regularisation weight for the image.
        max_iter : int
            Maximum descent iterations per scale.
        num_scales : int
            Number of pyramid levels for coarse-to-fine processing.
        tol : float
            Convergence tolerance on ‖Δh‖₂.
        kernel_threshold : float
            Sparsity threshold for kernel entries (fraction of max).
        multiscale : bool
            Enable multi-scale coarse-to-fine strategy.
        verbose : bool
            Print progress diagnostics.
        """
        super().__init__(name='SDP-BID')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.max_iter = max_iter
        self.num_scales = num_scales
        self.tol = tol
        self.kernel_threshold = kernel_threshold
        self.multiscale = multiscale
        self.verbose = verbose

        self.history = {}
        self.hyperparams = {}

    # ────────────────────────────────────────────────────────────────
    #  Main entry point
    # ────────────────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform blind deconvolution on a single greyscale image.

        Parameters
        ----------
        image : (H, W) ndarray
            Blurred greyscale image.  Accepts both [0, 255] (uint8 / int)
            and [0, 1] (float) dynamic ranges.

        Returns
        -------
        x_restored  : (H, W) ndarray, dtype int16
            Restored image in [0, 255].
        h_estimated : (kh, kw) ndarray, dtype float64
            Estimated blur kernel (non-negative, sums to 1).
        """
        start_time = time.time()

        # ── Normalise to [0, 1] float64 ──
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if self.verbose:
            print(f"[{self.name}] Image: {y.shape}, "
                  f"Kernel: {self.kernel_shape}, "
                  f"λ_tv={self.lambda_tv}, scales={self.num_scales}")

        # ── Blind deconvolution ──
        if self.multiscale:
            x_est, h_est, history = sdp_multiscale(
                y,
                kernel_shape=self.kernel_shape,
                lambda_tv=self.lambda_tv,
                num_scales=self.num_scales,
                iters_per_scale=self.max_iter,
                tol=self.tol,
                kernel_threshold=self.kernel_threshold,
                verbose=self.verbose,
            )
        else:
            # Single-scale fallback
            h_init = init_gaussian_kernel(self.kernel_shape)
            y_tapered = edge_taper(y, self.kernel_shape)

            x_est, h_est, history = sdp_single_scale(
                y_tapered, h_init, y.copy(),
                lambda_tv=self.lambda_tv,
                max_iter=self.max_iter,
                tol=self.tol,
                kernel_threshold=self.kernel_threshold,
                verbose=self.verbose,
            )

            # Non-blind refinement with estimated kernel
            x_est = refine_image_non_blind(
                y_tapered, h_est, x_est,
                lambda_tv=self.lambda_tv,
                max_iter=100,
                verbose=self.verbose,
            )

        elapsed = time.time() - start_time

        # ── Store metadata ──
        self.history = history
        self.hyperparams = {
            'lambda_tv': self.lambda_tv,
            'kernel_shape': self.kernel_shape,
            'num_scales': self.num_scales,
            'elapsed_time': elapsed,
            'multiscale': self.multiscale,
        }

        if self.verbose:
            print(f"[{self.name}] Done in {elapsed:.1f} s.")

        # ── Convert to framework output format ──
        x_out = x_est * 255.0
        x_out = np.round(x_out).astype(np.int16)

        return x_out, h_est

    # ────────────────────────────────────────────────────────────────
    #  Framework interface methods
    # ────────────────────────────────────────────────────────────────

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_tv', self.lambda_tv),
            ('max_iter', self.max_iter),
            ('num_scales', self.num_scales),
            ('tol', self.tol),
            ('kernel_threshold', self.kernel_threshold),
            ('multiscale', self.multiscale),
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

