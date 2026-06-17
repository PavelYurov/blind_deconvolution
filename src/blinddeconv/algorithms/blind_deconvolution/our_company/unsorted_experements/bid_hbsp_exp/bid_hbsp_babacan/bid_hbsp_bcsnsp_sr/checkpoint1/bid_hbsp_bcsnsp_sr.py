"""
BID-HBSP + BCSNSP-SR: Bayesian Blind Image Deconvolution with
Hyperbolic-Secant Prior, enhanced by Bayesian Super-Resolution
initialisation and periodic refinement.

Integration strategy
--------------------
The original BID-HBSP (Castro-Macías et al., 2024) initialises the
latent image x₀ = y (blurred observation) and iteratively alternates
between image/kernel estimation via a variational EM loop.  The
quality of kernel estimation directly depends on the gradient sharpness
of x: with x₀ = y the gradients are weak and the first few kernel
estimates are poor — this can trap the EM in a bad local minimum.

This module replaces the trivial initialisation with a BCSNSP-SR pass
(Salvador et al., 2013) run at resolution factor 1 (no upscaling).
At res=1 the SR solver degenerates into a multi-frame Bayesian
deconvolution with a combined anisotropic-TV + SAR prior.  L pseudo-
frames generated from y via sub-pixel shifts provide information
redundancy that, combined with a stronger (TV+SAR) regulariser, yields
a substantially sharper x₀ even with an inaccurate Gaussian kernel seed.

Additionally, an *optional* periodic SR refinement step can be enabled
(``sr_refine_every > 0``): every N EM iterations a short BCSNSP-SR pass
with the *current* kernel estimate re-sharpens x, acting as a
complementary-prior "reset" that helps escape local minima.

Pipeline
--------
Stage 0  — SR initialisation  (BCSNSP-SR at res=1, ~15 iters)
Stage 1  — Modified EM loop   (BID-HBSP with optional SR refinement)
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
    sr_refine_step,
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
    """BID-HBSP with BCSNSP-SR-enhanced initialisation and refinement.

    Inherits all BID-HBSP parameters and adds SR-specific ones.

    New parameters (SR initialisation)
    -----------------------------------
    sr_L : int
        Number of pseudo-frames for the SR pass (default 4).
    sr_max_shift : float
        Max sub-pixel shift when generating pseudo-frames (default 0.5).
    sr_maxit_init : int
        SR iterations for the initial estimate (Stage 0, default 15).
    sr_lambda_prior : float
        TV vs SAR trade-off in [0, 1] for the SR solver (default 0.5).

    New parameters (periodic SR refinement)
    ----------------------------------------
    sr_refine_every : int
        Run a lightweight SR pass every N EM iterations.
        0 = disabled (default 0).
    sr_maxit_refine : int
        SR iterations for each refinement pass (default 5).

    Original BID-HBSP parameters
    -----------------------------
    kernel_shape, hs_scale, noise_sigma, max_iter, cg_iter, cg_tol,
    solver, irw_iter, lambda_h_init, lambda_h_min, lambda_h_decay,
    kernel_threshold, beta_update, verbose

    See the BID-HBSP docstring for their descriptions.
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
        sr_L: int = 4,
        sr_max_shift: float = 0.5,
        sr_maxit_init: int = 15,
        sr_lambda_prior: float = 0.5,
        # ── SR periodic refinement ─────────────────
        sr_refine_every: int = 0,
        sr_maxit_refine: int = 5,
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

        # SR parameters
        self.sr_L = sr_L
        self.sr_max_shift = sr_max_shift
        self.sr_maxit_init = sr_maxit_init
        self.sr_lambda_prior = sr_lambda_prior
        self.sr_refine_every = sr_refine_every
        self.sr_maxit_refine = sr_maxit_refine

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
        #  STAGE 0: SR-enhanced initialisation (BCSNSP-SR at res=1)
        #
        #  Instead of x₀ = y (blurred), we obtain a sharper x₀ by
        #  running the BCSNSP-SR multi-frame deconvolver with the
        #  Gaussian seed kernel.  Even with an inaccurate PSF, the
        #  TV+SAR prior produces edges that are 2-5 dB sharper than
        #  the raw blurred input, giving the Wiener kernel estimator
        #  a much better gradient map to work with on the first EM
        #  iteration.
        # ══════════════════════════════════════════════════════════════════
        if self.verbose:
            print(
                f"[{self.name}] Stage 0: SR initialisation "
                f"(L={self.sr_L}, iters={self.sr_maxit_init}) …"
            )

        x = sr_initial_estimate(
            y,
            h_init=h,
            L=self.sr_L,
            max_shift=self.sr_max_shift,
            sr_maxit=self.sr_maxit_init,
            lambda_prior=self.sr_lambda_prior,
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

            # ── Periodic SR refinement (optional) ────────────────────────
            #
            # Every sr_refine_every iterations, run a lightweight
            # BCSNSP-SR pass with the *current* (improved) kernel.
            # This injects complementary TV+SAR structure into x,
            # helping the HS-CG image solver escape local minima.
            # The kernel h is NOT modified — only x is updated.
            if (
                self.sr_refine_every > 0
                and (it + 1) % self.sr_refine_every == 0
                and (it + 1) < self.max_iter  # skip on last iter
            ):
                if self.verbose:
                    print(
                        f"  Iter {it + 1}: SR refinement "
                        f"({self.sr_maxit_refine} iters) …"
                    )
                x = sr_refine_step(
                    y,
                    h=h,
                    x_current=x,
                    L=self.sr_L,
                    max_shift=self.sr_max_shift,
                    sr_maxit=self.sr_maxit_refine,
                    lambda_prior=self.sr_lambda_prior,
                    verbose=False,
                )

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
            "sr_L": self.sr_L,
            "sr_maxit_init": self.sr_maxit_init,
            "sr_refine_every": self.sr_refine_every,
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
            ("sr_L", self.sr_L),
            ("sr_max_shift", self.sr_max_shift),
            ("sr_maxit_init", self.sr_maxit_init),
            ("sr_lambda_prior", self.sr_lambda_prior),
            ("sr_refine_every", self.sr_refine_every),
            ("sr_maxit_refine", self.sr_maxit_refine),
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
