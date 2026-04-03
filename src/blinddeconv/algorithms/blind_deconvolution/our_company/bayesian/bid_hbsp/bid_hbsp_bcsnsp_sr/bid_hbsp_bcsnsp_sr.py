"""
BID-HBSP + BCSNSP-SR: Bayesian Blind Image Deconvolution with
Hyperbolic-Secant Prior, enhanced by BCSNSP-SR-inspired initialisation.

Integration strategy
--------------------
The original BID-HBSP (Castro-Macías et al., 2024) initialises the
latent image x₀ = y (blurred observation) and iteratively alternates
between image/kernel estimation via a variational EM loop.  The
quality of kernel estimation directly depends on the gradient sharpness
of x: with x₀ = y the gradients are weak and the first few kernel
estimates are poor — this can trap the EM in a bad local minimum.

This module replaces the trivial initialisation with a fast two-phase
approach inspired by the BCSNSP-SR dual-prior philosophy (Salvador et
al., 2013):

  Phase 1 — FFT-based SAR deconvolution (``restore_sar`` from BCSNSP-SR):
      Wiener-like filter with automatic α/β estimation.  Pure frequency-
      domain operations, O(N log N).

  Phase 2 — TV-IRLS refinement:
      A few iterations of anisotropic Lp (p ≈ 0.8) regularisation using
      FFT matvec.  Adds edge-preserving sparsity that SAR alone cannot
      provide, mirroring the TV component of BCSNSP-SR.

The combined initialisation produces a sharper x₀ with better-defined
edges, giving the Wiener kernel estimator a stronger gradient signal on
the very first EM iteration.  No sparse matrices are built — the entire
Stage 0 adds ~1-2 seconds even for 256×256 images.

Pipeline
--------
Stage 0  — SAR + TV initialisation  (FFT-based, ~1-2 s)
Stage 1  — Variational EM loop      (standard BID-HBSP)
Stage 2  — Final non-blind deconvolution (IRLS, p=0.8)

References
----------
[1] Castro-Macías, Pérez-Bueno et al. (2024), "Bayesian Blind Image
    Deconvolution using a Hyperbolic-Secant prior", ICIP 2024.
[2] Salvador, Villena, Molina, Katsaggelos (2013), "Bayesian Combination
    of Sparse and Non-Sparse Priors in Image Super Resolution", DSP.
[3] Babacan, Molina, Katsaggelos (2009), "Variational Bayesian Blind
    Deconvolution Using a Total Variation Prior", IEEE TIP 18(1).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    precompute_gradient_operators,
    init_gaussian_kernel,
    fft_convolve,
    sr_initial_estimate,
)
from .solvers import (
    solve_image_cg,
    solve_image_irw,
    solve_kernel_fourier,
    update_noise_precision,
    update_hs_weights,
    final_deconvolution,
)

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
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

for _p in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm


class BID_HBSP_BCSNSP_SR(DeconvolutionAlgorithm):
    """BID-HBSP with BCSNSP-SR-inspired fast initialisation.

    Inherits all BID-HBSP parameters and adds two lightweight SR ones.

    New parameters (SR initialisation — Stage 0)
    ---------------------------------------------
    sr_lambda_prior : float
        Strength of the TV refinement relative to the SAR result, in
        [0, 1].  0 = SAR-only initialisation; 1 = full TV pass (default 0.5).
    sr_tv_iters : int
        Number of TV-IRLS iterations in Phase 2 of the initialisation
        (default 5).  All FFT-based, very fast.

    Original BID-HBSP parameters
    -----------------------------
    kernel_shape, hs_scale, noise_sigma, max_iter, cg_iter, cg_tol,
    solver, irw_iter, lambda_h_init, lambda_h_min, lambda_h_decay,
    kernel_threshold, beta_update, verbose
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # ── HS prior ───────────────────────────────
        hs_scale: float = 0.5,
        noise_sigma: float = 0.01,
        max_iter: int = 40,
        # ── Image solver ───────────────────────────
        cg_iter: int = 50,
        cg_tol: float = 1e-6,
        solver: str = "cg",
        irw_iter: int = 5,
        # ── Kernel estimation ──────────────────────
        lambda_h_init: float = 1e3,
        lambda_h_min: float = 1.0,
        lambda_h_decay: float = 0.92,
        kernel_threshold: bool = True,
        # ── Noise estimation ───────────────────────
        beta_update: bool = True,
        # ── SR initialisation (Stage 0) ────────────
        sr_lambda_prior: float = 0.5,
        sr_tv_iters: int = 5,
        # ── General ────────────────────────────────
        verbose: bool = False,
    ):
        super().__init__(name="BID-HBSP+BCSNSP-SR")
        self.kernel_shape = tuple(kernel_shape)
        self.hs_scale = hs_scale
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter

        self.cg_iter = cg_iter
        self.cg_tol = cg_tol
        self.solver = solver
        self.irw_iter = irw_iter

        self.lambda_h_init = lambda_h_init
        self.lambda_h_min = lambda_h_min
        self.lambda_h_decay = lambda_h_decay
        self.kernel_threshold = kernel_threshold

        self.beta_update = beta_update

        # SR parameters (lightweight, FFT-only)
        self.sr_lambda_prior = sr_lambda_prior
        self.sr_tv_iters = sr_tv_iters

        self.verbose = verbose

        self.history: Dict[str, list] = {
            "kernel_diff": [],
            "noise_precision": [],
            "residual_norm": [],
        }
        self.hyperparams: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────
    #  Main entry point
    # ─────────────────────────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run BID-HBSP+BCSNSP-SR on a single greyscale blurred image.

        Parameters
        ----------
        image : ndarray, shape (H, W)
            Observed blurred image (uint8 [0, 255] or float [0, 1]).

        Returns
        -------
        restored : ndarray, shape (H, W), int16 [0, 255]
        kernel   : ndarray, shape (kh, kw), float64
        """
        start_time = time.time()

        # ── 1. Data preparation ──────────────────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        H, W = y.shape
        kh, kw = self.kernel_shape
        b = self.hs_scale

        # ── 2. Kernel seed ───────────────────────────────────────────────
        h = init_gaussian_kernel(self.kernel_shape)

        # ══════════════════════════════════════════════════════════════════
        #  STAGE 0: Fast SAR + TV initialisation
        #
        #  Instead of x₀ = y (blurred), we run a fast FFT-based
        #  SAR deconvolution (restore_sar from BCSNSP-SR) followed
        #  by a lightweight TV-IRLS refinement.  All operations are
        #  element-wise in the Fourier domain — O(N log N), ~1-2 s.
        #  This produces sharper edges in x₀, giving the Wiener
        #  kernel estimator a stronger gradient signal from the start.
        # ══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(
                f"[{self.name}] Stage 0: SAR + TV initialisation "
                f"(tv_iters={self.sr_tv_iters}, λ={self.sr_lambda_prior}) …"
            )

        x = sr_initial_estimate(
            y,
            h_init=h,
            lambda_prior=self.sr_lambda_prior,
            tv_iters=self.sr_tv_iters,
            verbose=self.verbose,
        )

        if self.verbose:
            # Quick quality check: gradient energy ratio
            grad_y = float(np.sum(np.abs(np.diff(y, axis=1))))
            grad_x = float(np.sum(np.abs(np.diff(x, axis=1))))
            print(
                f"  SR init done. Gradient energy: y={grad_y:.1f} → "
                f"x₀={grad_x:.1f} (×{grad_x / (grad_y + 1e-12):.2f})"
            )

        # ── 3. Remaining initialisations ─────────────────────────────────
        beta = 1.0 / (self.noise_sigma ** 2 + 1e-12)
        lambda_h = self.lambda_h_init

        F_ops = precompute_gradient_operators((H, W))

        gamma_x = np.full((H, W), 1.0 / (b * b))
        gamma_y = np.full((H, W), 1.0 / (b * b))

        if self.verbose:
            print(
                f"[{self.name}] Stage 1: EM loop — {H}×{W}, "
                f"kernel {kh}×{kw}, b={b:.3f}, β₀={beta:.1f}"
            )

        # ══════════════════════════════════════════════════════════════════
        #  STAGE 1: Variational EM loop (BID-HBSP core)
        # ══════════════════════════════════════════════════════════════════
        n_iter = 0
        sigma_sq = np.zeros_like(x)

        for it in range(self.max_iter):
            h_prev = h.copy()

            # (a) Update HS prior weights
            gamma_x, gamma_y = update_hs_weights(x, sigma_sq, b)

            # (b) Image estimation (CG with per-pixel HS weights)
            if self.solver == "cg":
                x, sigma_sq = solve_image_cg(
                    y, h, x, beta,
                    gamma_x, gamma_y,
                    max_cg_iter=self.cg_iter,
                    cg_tol=self.cg_tol,
                )
            else:
                x = solve_image_irw()
                sigma_sq = np.zeros_like(x)

            # (c) Kernel estimation (gradient-space Wiener)
            h = solve_kernel_fourier(
                y, x, sigma_sq, self.kernel_shape, beta, lambda_h,
                do_threshold=self.kernel_threshold,
            )

            # (d) Noise precision update (M-step for β)
            if self.beta_update:
                beta = update_noise_precision(y, h, x, beta)

            # Anneal kernel regularisation
            lambda_h = max(lambda_h * self.lambda_h_decay, self.lambda_h_min)

            # ── Monitoring ───────────────────────────────────────────────
            diff = float(np.linalg.norm(h - h_prev))
            residual = float(np.linalg.norm(y - fft_convolve(x, h)))
            self.history["kernel_diff"].append(diff)
            self.history["noise_precision"].append(beta)
            self.history["residual_norm"].append(residual)

            if self.verbose:
                print(
                    f"  Iter {it + 1:3d}/{self.max_iter}:  "
                    f"ΔH={diff:.2e}  β={beta:.2f}  "
                    f"λ_h={lambda_h:.2f}  ‖r‖={residual:.4f}"
                )

            n_iter = it + 1
            if diff < 1e-5 and it > 5:
                if self.verbose:
                    print(f"  Converged at iteration {n_iter}.")
                break

        # ══════════════════════════════════════════════════════════════════
        #  STAGE 2: Final non-blind deconvolution (IRLS, p=0.8)
        # ══════════════════════════════════════════════════════════════════
        lambda_final = beta * 0.0005

        if self.verbose:
            print(
                f"[{self.name}] Stage 2: Final deconvolution "
                f"(IRLS p=0.8, λ={lambda_final:.4f}) …"
            )

        x_final = final_deconvolution(y, h, beta, lambda_final)

        # ── 4. Diagnostics ───────────────────────────────────────────────
        self.timer = time.time() - start_time
        self.hyperparams = {
            "hs_scale": b,
            "noise_precision_final": beta,
            "noise_sigma_estimated": (
                1.0 / np.sqrt(beta) if beta > 0 else None
            ),
            "lambda_h_final": lambda_h,
            "iterations": n_iter,
            "sr_lambda_prior": self.sr_lambda_prior,
            "sr_tv_iters": self.sr_tv_iters,
            "time_seconds": self.timer,
        }

        # ── 5. Output ───────────────────────────────────────────────────
        x_out = x_final * 255.0
        x_out = np.round(x_out).astype(np.int16)
        return x_out, h

    # ─────────────────────────────────────────────────────────────────────
    #  Interface methods
    # ─────────────────────────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("hs_scale", self.hs_scale),
            ("noise_sigma", self.noise_sigma),
            ("max_iter", self.max_iter),
            ("solver", self.solver),
            ("lambda_h_init", self.lambda_h_init),
            ("lambda_h_decay", self.lambda_h_decay),
            ("beta_update", self.beta_update),
            ("sr_lambda_prior", self.sr_lambda_prior),
            ("sr_tv_iters", self.sr_tv_iters),
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
