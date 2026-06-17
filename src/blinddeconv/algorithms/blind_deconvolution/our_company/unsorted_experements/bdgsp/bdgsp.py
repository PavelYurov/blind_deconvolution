"""
bdgsp.py — Framework wrapper for BDGSP.

Reference:
    S. D. Babacan, R. Molina, M. N. Do, A. K. Katsaggelos,
    "Bayesian Blind Deconvolution with General Sparse Image Priors,"
    European Conference on Computer Vision (ECCV), 2012.
    http://www.dbabacan.info/BDGSP.php

Pipeline implemented in :meth:`BDGSP.process`
---------------------------------------------
    1.  Input normalisation (float64 in [0, 1], grayscale).
    2.  Reflection padding to mitigate circular-convolution boundaries.
    3.  Build a coarse-to-fine image/kernel pyramid (ratio 1/√2).
    4.  At every level run ``num_iter`` outer VB iterations of
        Algorithm 1 (see :mod:`.solvers`):
            a. ξ_γ  = ρ'(ν_γ)/ν_γ                      (eq. 23)
            b. E[x_γ] ← CG solve                        (eq. 18)
            c. C_xγ(i,i) ← 1 / C_xγ⁻¹(i,i)             (diag. approx.)
            d. ν_γ(i) ← √(E[x_γ]² + C_xγ(i,i))
            e. k ← projected-gradient QP                (eq. 14)
        The noise variance σ² is annealed from coarse to fine.
    5.  Final non-blind image reconstruction via eq. (24).
    6.  Crop to the original size; return ``(int16 image, kernel)``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ── Framework base class import (same pattern as neighbouring algorithms) ──
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

from blinddeconv.algorithms.base import DeconvolutionAlgorithm  # noqa: E402
# ────────────────────────────────────────────────────────────────────────────

from .solvers import (  # noqa: E402
    diag_Cx_approx,
    estimate_kernel,
    solve_final_image,
    solve_x_gamma,
)
from .utils import (  # noqa: E402
    apply_filter,
    build_pyramid_sizes,
    center_kernel,
    compute_filtered,
    crop_center,
    edgetaper,
    edgetaper_pad,
    gradient_filters,
    resize_image,
    resize_kernel,
    to_grayscale,
    xi_from_nu,
)


class BDGSP(DeconvolutionAlgorithm):
    """
    Bayesian Blind Deconvolution with General Sparse Image Priors.

    Parameters
    ----------
    kernel_size : int
        Spatial support (square, odd) of the estimated PSF at the
        original resolution.  Default ``25``.
    prior : {"log", "lp", "exp", "gaussian"}
        Choice of super-Gaussian image prior.  ``"log"`` reproduces the
        best-performing configuration of the paper.
    p : float
        Exponent used when ``prior == "lp"``.  Default ``0.8``.
    sigma_r : float
        Scale parameter of the ``"exp"`` prior.  Default ``0.9``.
    num_outer : int
        Outer VB iterations per pyramid level (Algorithm 1 inner loop).
        Default ``10``.
    num_cg_x : int
        CG iterations for the image sub-problem (eq. 18) per outer step.
    num_kernel_iter : int
        Projected-gradient iterations for the kernel QP (eq. 14).
    sigma2_init, sigma2_final : float
        Noise-variance schedule: log-linearly annealed from coarsest to
        finest level (§4 of the paper, adapted from Levin 2011).
    pyramid_scale : float
        Down-sampling ratio between pyramid levels.  Default ``0.5``
        (standard coarse-to-fine step; finer ratios like 1/√2 require
        more outer iterations per level to let the kernel re-concentrate
        after upsampling).
    min_kernel : int
        Kernel size at the coarsest level.  Default ``3``.
    final_sigma2 : float
        σ² used in the final image reconstruction (eq. 24).  If
        ``None``, uses ``sigma2_final``.
    pad : int or None
        Reflection-padding width.  If ``None``, defaults to
        ``2 * kernel_size``.
    filter_order : int
        Order of the derivative filter bank.  ``1`` (default) uses only
        ``f_x = [1, -1]`` and ``f_y = [1, -1]ᵀ`` as in the reference
        MATLAB implementation of the paper (§2, "high-pass filters").
        ``2`` additionally adds second-order derivatives.  The paper
        itself is filter-agnostic.
    prune_threshold : float
        Relative threshold applied after every kernel update to prune
        tiny entries.  The paper explicitly states kernel thresholding
        is *not* used (§4, last paragraph), so default is ``0``.
    final_prune : float
        Relative threshold applied once on the final kernel (clean-up
        only; does not affect kernel estimation).  Default ``0.05``.
    use_edgetaper : bool
        Whether to apply MATLAB-style ``edgetaper`` to the observation
        at every pyramid level.  Not in the paper but standard in FFT
        based circular-convolution implementations; mitigates
        wrap-around artefacts at image borders.  Default ``True``.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        prior: str = "log",
        p: float = 0.8,
        sigma_r: float = 0.9,
        num_outer: int = 20,
        num_cg_x: int = 25,
        num_kernel_iter: int = 80,
        sigma2_init: float = 5e-2,
        sigma2_final: float = 5e-4,
        pyramid_scale: float = 0.5,
        min_kernel: int | None = 3,
        final_sigma2: float | None = 1e-4,
        pad: int | None = None,
        filter_order: int = 1,
        prune_threshold: float = 0.0,
        final_prune: float = 0.05,
        use_edgetaper: bool = True,
        xi_max: float = 1e3,
        kernel_sharpen: float = 1.5,
        kernel_init_sigma: float = 0.7,
    ) -> None:
        super().__init__(name="BDGSP")

        if kernel_size % 2 == 0:
            kernel_size += 1  # enforce odd kernel size

        self.kernel_size = int(kernel_size)
        self.prior = str(prior)
        self.p = float(p)
        self.sigma_r = float(sigma_r)
        self.num_outer = int(num_outer)
        self.num_cg_x = int(num_cg_x)
        self.num_kernel_iter = int(num_kernel_iter)
        self.sigma2_init = float(sigma2_init)
        self.sigma2_final = float(sigma2_final)
        self.pyramid_scale = float(pyramid_scale)
        # ``min_kernel=None`` disables the pyramid (single-level run at
        # the full requested kernel size).
        self.min_kernel = (
            int(min_kernel) if min_kernel is not None else int(self.kernel_size)
        )
        self.xi_max = float(xi_max)
        self.final_sigma2 = (
            float(final_sigma2) if final_sigma2 is not None else float(sigma2_final)
        )
        self.pad = pad
        self.filter_order = int(filter_order)
        self.prune_threshold = float(prune_threshold)
        self.final_prune = float(final_prune)
        self.use_edgetaper = bool(use_edgetaper)
        self.kernel_sharpen = float(kernel_sharpen)
        self.kernel_init_sigma = float(kernel_init_sigma)

        self.history: Dict[str, List[Any]] = {"sigma2": [], "level_shapes": []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ───────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Normalise to float64 grayscale in [0, 1].
        y_full = to_grayscale(image)

        # 2. Reflection-pad to suppress boundary wrap-around artefacts.
        pad = self.pad if self.pad is not None else 2 * self.kernel_size
        y_pad = edgetaper_pad(y_full, pad)

        # 3. Build pyramid sizes (coarsest → finest).
        levels = build_pyramid_sizes(
            y_pad.shape,
            self.kernel_size,
            min_kernel=self.min_kernel,
            scale=self.pyramid_scale,
        )
        self.history["level_shapes"] = [
            {"shape": shape, "kernel": ksz} for shape, ksz in levels
        ]

        filters = gradient_filters(order=self.filter_order)

        # σ² is annealed log-linearly across ALL outer iterations (summed
        # over every pyramid level) rather than just between levels.  This
        # matches Babacan's original MATLAB implementation / Levin 2011:
        # start with a large σ² so that trace(C_xγ) is big enough to keep
        # the kernel QP away from the delta solution, then shrink σ² to
        # progressively sharpen both the filtered images and the kernel.
        n_levels = len(levels)
        total_iters = n_levels * self.num_outer
        if total_iters > 1:
            sigma2_all = np.exp(
                np.linspace(
                    np.log(self.sigma2_init),
                    np.log(self.sigma2_final),
                    total_iters,
                )
            ).tolist()
        else:
            sigma2_all = [self.sigma2_final]
        self.history["sigma2"] = sigma2_all

        # 4. Coarse-to-fine VB iteration.
        k = None
        global_iter = 0
        for level_idx, ((H, W), ksz) in enumerate(levels):
            y_lvl = resize_image(y_pad, (H, W))

            # Enforce odd kernel size at this level.
            if ksz % 2 == 0:
                ksz += 1
            if k is None:
                # Gaussian initialisation breaks the perfect axis-aligned
                # symmetry that a pure δ would impose on the very first
                # kernel QP, which is the dominant cause of the "+"-shape
                # cross artefact in BDGSP-like FFT-based implementations.
                yy, xx = np.mgrid[0:ksz, 0:ksz]
                cy = cx = ksz // 2
                sig = max(self.kernel_init_sigma, 0.5)
                k = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sig * sig))
                k = k / k.sum()
            else:
                k = resize_kernel(k, (ksz, ksz))
                # Sharpen the upsampled kernel to compensate for the
                # bilinear smoothing that would otherwise leave it diffuse.
                if self.kernel_sharpen > 1.0:
                    k = np.maximum(k, 0.0) ** self.kernel_sharpen
                    s = k.sum()
                    k = k / s if s > 0 else k

            # edgetaper using the current kernel at every level: at level 0
            # the Gaussian seed is broad enough to mute boundary content
            # smoothly; at deeper levels the propagated estimate is far
            # better than δ.  This mitigates the wrap-around correlation
            # that imprints axis-aligned crosses onto the recovered k.
            y_taper = edgetaper(y_lvl, k) if self.use_edgetaper else y_lvl

            # Pre-compute filtered observations  y_γ = f_γ ⊗ y.
            y_filt = compute_filtered(y_taper, filters)

            # Initialisation per Algorithm 1:  x_γ = y_γ, C_xγ = 0.
            # (Re-initialised at every level — the natural-image gradient
            # statistics are scale-dependent and propagating x_γ across
            # levels was found to lock in axis-aligned spike artefacts.)
            x_list = [yf.copy() for yf in y_filt]
            cx_list = [np.zeros_like(yf) for yf in y_filt]

            for outer in range(self.num_outer):
                sigma2 = sigma2_all[global_iter]
                global_iter += 1

                # (a) ν_γ and ξ_γ.
                xi_list = []
                for x, cx in zip(x_list, cx_list):
                    nu = np.sqrt(x * x + cx)
                    xi = xi_from_nu(
                        nu, prior=self.prior, p=self.p, sigma_r=self.sigma_r
                    )
                    # Cap ξ from above as sanctioned by paper §3.2 (last
                    # paragraph): "ξ_γ can be bounded from above by a
                    # positive number, which makes the MAP problem
                    # well-defined".  Without it the log prior produces
                    # ξ → ∞ at flat pixels which collapses the trace
                    # correction and drives the kernel to a delta.
                    if self.xi_max > 0:
                        xi = np.minimum(xi, self.xi_max)
                    xi_list.append(xi)

                # (b) CG solve for each E[x_γ].
                new_x = []
                for x, xi, yg in zip(x_list, xi_list, y_filt):
                    new_x.append(
                        solve_x_gamma(
                            yg, k, xi, sigma2,
                            x0=x, max_iter=self.num_cg_x, tol=1e-5,
                        )
                    )
                x_list = new_x

                # (c) diagonal approximation of C_xγ.
                cx_list = [diag_Cx_approx(k, xi, sigma2) for xi in xi_list]

                # (e) kernel QP.
                k = estimate_kernel(
                    x_list, cx_list, y_filt, k,
                    num_iter=self.num_kernel_iter,
                )

                # Re-centre the kernel each iteration: any residual
                # translation in k is absorbed into x_γ on the next CG
                # solve, which keeps the kernel-image alignment stable
                # across iterations and prevents drift toward boundary.
                k = center_kernel(k)

                # Optional per-iteration pruning (off by default — the
                # paper §4 explicitly says no kernel thresholding).
                if self.prune_threshold > 0.0 and k.max() > 0.0:
                    thr = self.prune_threshold * k.max()
                    k = np.where(k < thr, 0.0, k)
                    s = k.sum()
                    if s > 0:
                        k = k / s

            # Re-centre kernel once per level (after it has settled).
            k = center_kernel(k)

        assert k is not None
        # Resize final kernel back to requested kernel_size (odd).
        final_ksz = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        if k.shape[0] != final_ksz:
            k = resize_kernel(k, (final_ksz, final_ksz))
        k = center_kernel(k)

        # Single, mild final pruning → simplex projection.
        if self.final_prune > 0.0 and k.max() > 0:
            thr = self.final_prune * k.max()
            k = np.where(k < thr, 0.0, k)
            s = k.sum()
            if s > 0:
                k = k / s

        # 5. Final non-blind reconstruction on the padded image.
        y_lvl = edgetaper(y_pad, k) if self.use_edgetaper else y_pad
        y_filt = compute_filtered(y_lvl, filters)
        # Re-estimate ξ_γ on the full resolution from the latest x_γ, c_γ
        # pair computed at the finest pyramid level.  We re-run a quick CG
        # image update on the full-resolution grid to align them with the
        # final kernel and padded image.
        x_list_final = [yf.copy() for yf in y_filt]
        cx_list_final = [np.zeros_like(yf) for yf in y_filt]
        for _ in range(2):
            xi_final = []
            for x, cx in zip(x_list_final, cx_list_final):
                nu = np.sqrt(x * x + cx)
                xi_final.append(
                    xi_from_nu(
                        nu, prior=self.prior, p=self.p, sigma_r=self.sigma_r
                    )
                )
            x_list_final = [
                solve_x_gamma(
                    yg, k, xi, self.final_sigma2,
                    x0=x, max_iter=self.num_cg_x, tol=1e-5,
                )
                for x, xi, yg in zip(x_list_final, xi_final, y_filt)
            ]
            cx_list_final = [
                diag_Cx_approx(k, xi, self.final_sigma2) for xi in xi_final
            ]

        x_hat = solve_final_image(
            y_lvl, k, xi_final, filters, self.final_sigma2,
            max_iter=200, tol=1e-5,
        )

        # 6. Undo padding and rescale to uint-like output.
        x_hat = crop_center(x_hat, pad)
        x_hat = np.clip(x_hat, 0.0, 1.0)
        x_out = np.clip(x_hat * 255.0, 0, 255).astype(np.int16)

        elapsed = time.time() - start_time
        self.hyperparams = {
            "kernel_size": self.kernel_size,
            "prior": self.prior,
            "p": self.p,
            "sigma_r": self.sigma_r,
            "num_outer": self.num_outer,
            "num_cg_x": self.num_cg_x,
            "num_kernel_iter": self.num_kernel_iter,
            "sigma2_init": self.sigma2_init,
            "sigma2_final": self.sigma2_final,
            "pyramid_scale": self.pyramid_scale,
            "min_kernel": self.min_kernel,
            "final_sigma2": self.final_sigma2,
            "pad": pad,
            "filter_order": self.filter_order,
            "prune_threshold": self.prune_threshold,
            "final_prune": self.final_prune,
            "use_edgetaper": self.use_edgetaper,
            "n_levels": n_levels,
            "time": elapsed,
        }
        return x_out, k

    # ── Framework interface ───────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_size", self.kernel_size),
            ("prior", self.prior),
            ("p", self.p),
            ("sigma_r", self.sigma_r),
            ("num_outer", self.num_outer),
            ("num_cg_x", self.num_cg_x),
            ("num_kernel_iter", self.num_kernel_iter),
            ("sigma2_init", self.sigma2_init),
            ("sigma2_final", self.sigma2_final),
            ("pyramid_scale", self.pyramid_scale),
            ("min_kernel", self.min_kernel),
            ("final_sigma2", self.final_sigma2),
            ("pad", self.pad),
            ("filter_order", self.filter_order),
            ("prune_threshold", self.prune_threshold),
            ("final_prune", self.final_prune),
            ("use_edgetaper", self.use_edgetaper),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
