"""
Blind Image Deconvolution via Fractional-Order TV
with Patch-wise Minimal Pixels (PMP) Prior.

Framework wrapper that exposes the ``process`` interface expected
by :class:`DeconvolutionAlgorithm`.

Algorithm overview (Wu et al. [1]):
  1. Build a coarse-to-fine image pyramid.
  2. At each level alternate between:
     a) **Image estimation** — ADMM with Grünwald–Letnikov
        fractional-order total variation (α ∈ (1, 2)) to suppress
        ringing artefacts ([1] Sec. 3.1, [2] Sec. 2).
     b) **Edge prediction** — gradient thresholding weighted by the
        Patch-wise Minimal Pixels map ([1] Sec. 3.2).
     c) **Kernel estimation** — closed-form spectral solver with
        simplex projection ([1] Sec. 3.3).
  3. Final non-blind deconvolution with fractional TV.

References
----------
[1] Wu, T., Wan, S., Feng, C., Zhang, H., Zeng, T.
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
    Meets Fractional-Order Method."
    J. Math. Imaging Vis., 2024.
    DOI: 10.1007/s10851-024-01221-x

[2] Pan, X., Ye, Y., Wang, J., Gao, X., Zhou, X.
    "Noncausal fractional directional differentiator and blind
    deconvolution: motion blur estimation."
    Multimedia Tools Appl., 73(3), 1485–1506, 2014.

Modules
-------
- ``utils``   : FFT helpers, GL coefficients, PMP, pyramid, kernel ops.
- ``solvers`` : ADMM image solver, kernel estimator, coarse-to-fine loop.
- ``fractional_order`` (this file) : framework wrapper class.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .solvers import coarse_to_fine_estimation, final_nonblind_deconvolution

# ---- Robust import of the abstract base class ----
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up until ``pyproject.toml`` is found."""
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root (pyproject.toml)")
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


class FractionalOrderBID(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using Fractional-Order TV + PMP Prior.

    The method combines a fractional-order variational regulariser
    (Grünwald–Letnikov derivative of order *alpha*) with a patch-wise
    minimal pixels prior to simultaneously estimate the blur kernel and
    the latent sharp image from a single blurred observation.

    Parameters
    ----------
    kernel_shape : (int, int)
        Spatial support of the blur kernel (height, width).
        Both values should be odd.
    alpha : float
        Fractional derivative order.  Values in (1, 2).
        α → 1 recovers standard TV; α → 2 approaches Laplacian.
        Typical good range: 1.2 – 1.6  ([1] Sec. 4).
    gl_truncation : int
        Number of terms in the GL finite-difference stencil.
    lambda_tv : float
        Fractional-TV regularisation weight during the blind phase.
    lambda_tv_final : float
        Fractional-TV weight for the final non-blind pass.
    mu_kernel : float
        Tikhonov weight on the kernel.
    admm_iter : int
        Inner ADMM iterations per image sub-problem.
    inner_iter : int
        Alternating-minimisation passes per pyramid level.
    rho_init : float
        Initial ADMM penalty parameter.
    grad_threshold_pct : float
        Percentile for gradient thresholding in edge prediction.
    pmp_patch_size : int
        Patch size for the PMP prior (odd).
    pmp_gamma : float
        Decay rate for the PMP weight map.
    scale_ratio : float
        Geometric ratio between successive pyramid scales.
    final_iter : int
        ADMM iterations for the final non-blind pass.
    verbose : bool
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (21, 21),
        alpha: float = 1.4,
        gl_truncation: int = 10,
        lambda_tv: float = 4e-3,
        lambda_tv_final: float = 2e-3,
        mu_kernel: float = 0.01,
        admm_iter: int = 15,
        inner_iter: int = 3,
        rho_init: float = 1.0,
        grad_threshold_pct: float = 94.0,
        pmp_patch_size: int = 5,
        pmp_gamma: float = 2.0,
        scale_ratio: float = 1.5,
        final_iter: int = 30,
        verbose: bool = False,
    ):
        super().__init__(name="FractionalOrder-PMP-BID")

        # Kernel geometry
        self.kernel_shape = tuple(kernel_shape)

        # Fractional derivative parameters  – [1] Eq. 7, [2] Sec. 2
        self.alpha = alpha
        self.gl_truncation = gl_truncation

        # Regularisation weights
        self.lambda_tv = lambda_tv
        self.lambda_tv_final = lambda_tv_final
        self.mu_kernel = mu_kernel

        # Solver controls
        self.admm_iter = admm_iter
        self.inner_iter = inner_iter
        self.rho_init = rho_init
        self.final_iter = final_iter

        # PMP prior parameters  – [1] Sec. 3.2
        self.grad_threshold_pct = grad_threshold_pct
        self.pmp_patch_size = pmp_patch_size
        self.pmp_gamma = pmp_gamma

        # Multi-scale
        self.scale_ratio = scale_ratio

        self.verbose = verbose

        # Runtime bookkeeping
        self.history: Dict[str, list] = {"kernel_diff": []}
        self.hyperparams: Dict[str, Any] = {}

    # ---------------------------------------------------------------
    #  Main entry point
    # ---------------------------------------------------------------
    def process(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run full blind deconvolution pipeline.

        Parameters
        ----------
        image : ndarray (H, W) or (H, W, C)
            Blurred observation.  Grayscale (single-channel) or colour.
            Values may be uint8 [0, 255] or float [0, 1].

        Returns
        -------
        restored : ndarray (H, W), int16, [0, 255]
            Estimated sharp image.
        kernel : ndarray (kh, kw), float64
            Estimated blur kernel (non-negative, unit sum).
        """
        t0 = time.time()

        # ------ Pre-processing ------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # Handle colour by converting to grayscale for kernel estimation
        if y.ndim == 3:
            y_gray = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            y_gray = y

        H, W = y_gray.shape

        if self.verbose:
            print(
                f"[{self.name}] Image {H}×{W}, "
                f"kernel {self.kernel_shape}, α = {self.alpha}"
            )

        # ------ Phase 1: Coarse-to-fine kernel estimation ------
        k_est, _ = coarse_to_fine_estimation(
            y_gray,
            kernel_shape=self.kernel_shape,
            alpha=self.alpha,
            L=self.gl_truncation,
            lambda_tv=self.lambda_tv,
            mu_kernel=self.mu_kernel,
            admm_iter=self.admm_iter,
            rho_init=self.rho_init,
            inner_iter=self.inner_iter,
            grad_threshold_pct=self.grad_threshold_pct,
            pmp_patch_size=self.pmp_patch_size,
            pmp_gamma=self.pmp_gamma,
            scale_ratio=self.scale_ratio,
            verbose=self.verbose,
        )

        # ------ Phase 2: Final non-blind deconvolution ------
        if self.verbose:
            print(f"[{self.name}] Final non-blind restoration …")

        if y.ndim == 3:
            # Process each channel independently with the same kernel
            channels = []
            for c in range(y.shape[2]):
                ch = final_nonblind_deconvolution(
                    y[:, :, c], k_est,
                    alpha=self.alpha,
                    L=self.gl_truncation,
                    lambda_tv=self.lambda_tv_final,
                    num_iter=self.final_iter,
                    rho_init=self.rho_init,
                )
                channels.append(ch)
            x_final = np.stack(channels, axis=-1)
        else:
            x_final = final_nonblind_deconvolution(
                y_gray, k_est,
                alpha=self.alpha,
                L=self.gl_truncation,
                lambda_tv=self.lambda_tv_final,
                num_iter=self.final_iter,
                rho_init=self.rho_init,
            )

        elapsed = time.time() - t0
        self.timer = elapsed

        self.hyperparams = {
            "alpha": self.alpha,
            "gl_truncation": self.gl_truncation,
            "lambda_tv": self.lambda_tv,
            "lambda_tv_final": self.lambda_tv_final,
            "mu_kernel": self.mu_kernel,
            "elapsed_sec": elapsed,
        }

        if self.verbose:
            print(f"[{self.name}] Done in {elapsed:.1f} s.")

        # ------ Post-processing ------
        x_final = x_final * 255.0
        x_final = np.clip(np.round(x_final), 0, 255).astype(np.int16)
        return x_final, k_est

    # ---------------------------------------------------------------
    #  Framework interface
    # ---------------------------------------------------------------
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("alpha", self.alpha),
            ("gl_truncation", self.gl_truncation),
            ("lambda_tv", self.lambda_tv),
            ("lambda_tv_final", self.lambda_tv_final),
            ("mu_kernel", self.mu_kernel),
            ("admm_iter", self.admm_iter),
            ("inner_iter", self.inner_iter),
            ("rho_init", self.rho_init),
            ("grad_threshold_pct", self.grad_threshold_pct),
            ("pmp_patch_size", self.pmp_patch_size),
            ("pmp_gamma", self.pmp_gamma),
            ("scale_ratio", self.scale_ratio),
            ("final_iter", self.final_iter),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_shape":
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams


# ===================================================================
#  Convenience function (mirrors ep_em.run_algorithm)
# ===================================================================
def run_algorithm(
    g: np.ndarray,
    kernel_shape: Tuple[int, int],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Convenience entry point for quick experiments.

    Parameters
    ----------
    g : ndarray
        Blurred image.
    kernel_shape : (kh, kw)
        Kernel support size.
    **kwargs
        Forwarded to :class:`FractionalOrderBID`.

    Returns
    -------
    (restored, kernel, hyperparams, history)
    """
    algo = FractionalOrderBID(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
