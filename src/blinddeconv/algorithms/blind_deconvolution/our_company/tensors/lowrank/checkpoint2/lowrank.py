"""
Low-Rank Blind Image Deconvolution.

Main module providing the :class:`LowRankBD` class, which wraps the
multi-scale blind deconvolution algorithm with low-rank kernel
regularisation into the framework's :class:`DeconvolutionAlgorithm`
interface.

Algorithm overview
------------------
The method operates in a coarse-to-fine multi-scale framework
(§ 1–3 below), alternating between three coupled sub-problems at
each scale:

1. **Image estimation** (ISTA)
   Estimates latent gradient images *x₁*, *x₂* using the L₁/L₂
   normalised-sparsity prior, which promotes sparse gradients while
   being scale-invariant.
   See Krishnan et al. (CVPR 2011) [3].

2. **Kernel estimation** (Conjugate Gradient)
   Estimates the blur kernel *k* via least-squares on the gradient
   domain with optional Tikhonov regularisation and an exponential
   regularisation schedule.

3. **Low-rank regularisation** (iALM RPCA)
   Decomposes the kernel matrix K = L + S via inexact Augmented
   Lagrangian Method for Robust PCA, where L (low-rank) captures
   the structural blur pattern and S (sparse) captures estimation
   noise.  SVT handles the L-update; GST handles the S-update.
   See Dong et al. (Neurocomputing 2017) [6], Lin et al. (2010).

4. **Non-blind deconvolution** (Split Bregman / ADMM)
   Given the estimated kernel, recovers the sharp image using a
   hyper-Laplacian gradient prior.
   See Krishnan & Fergus (NIPS 2009) [4].

References
----------
[1] Li, S., Chu, W., & Kuo, C.-C.J. "Understanding kernel size in
    blind deconvolution." WACV, 2019.
    GitHub: https://github.com/lisiyaoATbnu/low_rank_kernel
[2] Ren, D., et al. "Image Deblurring via Enhanced Low Rank Prior."
    IEEE TIP, vol. 25, no. 7, pp. 3426–3437, 2016.
[3] Krishnan, D., Tay, T., & Fergus, R. "Blind deconvolution using
    a normalized sparsity measure." CVPR, 2011.
[4] Krishnan, D. & Fergus, R. "Fast Image Deconvolution using
    Hyper-Laplacian Priors." NIPS, 2009.
[5] Yang, J., et al. "Hyper-Laplacian Regularized Non-local Low-rank
    Prior for Blind Image Deblurring." IEEE Access, 2020.
[6] Dong, J., et al. "Multi-image blind deconvolution using low-rank
    representation." Neurocomputing, vol. 259, pp. 227–236, 2017.
    GitHub: https://github.com/crewleader/BlindDeconvolutionLowRank
[7] Zuo, W., et al. "A Generalized Iterated Shrinkage Algorithm for
    Non-convex Sparse Coding." ICCV, 2013.
[8] Lin, Z., Chen, M. & Ma, Y. "The Augmented Lagrange Multiplier
    Method for Exact Recovery of Corrupted Low-Rank Matrices."
    arXiv:1009.5055, 2010.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    compute_gradients,
    build_scale_pyramid,
    center_kernel,
    normalize_kernel,
    resize_image,
    edgetaper,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from .solvers import (
    optimize_image,
    optimize_kernel,
    low_rank_regularization,
    fast_deconv_hyper_laplacian,
)
from pathlib import Path
import sys
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


class LowRankBD(DeconvolutionAlgorithm):
    """
    Low-Rank Blind Image Deconvolution.

    Estimates both the blur kernel and the latent sharp image from a
    single blurred observation, using a multi-scale coarse-to-fine
    framework with low-rank kernel regularisation.

    The image sub-problem is solved in the *image domain* via IRLS
    (Iteratively Reweighted Least Squares) with a hyper-Laplacian
    edge prior, following [6] (``solve_image_irls.m`` /
    ``solve_image_L2_w.m``).  The kernel sub-problem is solved via
    Conjugate Gradient with Tikhonov regularisation and projection
    onto {k ≥ 0, Σk = 1}.

    Parameters
    ----------
    kernel_size : int
        Expected maximum PSF size (must be odd, ≥ 3).
    lambda_ : float
        Base edge-regularisation weight α₀ for the image step.
        Scaled per pyramid level:
        α = λ · ``alpha_multiplier`` ^ (level − 0.5).
    sigma : float
        Low-rank regularisation flag/weight.  Set > 0 to enable,
        0 to disable.
    lambda_s : float or None
        Sparse-penalty weight λ for the RPCA decomposition.
        ``None`` uses the default 1/√max(kh, kw).
    p_sparse : float
        Lp exponent for the RPCA sparse term (0 < p ≤ 1).
    rho_alm : float
        Penalty growth factor ρ for the ALM solver.
    kernel_beta : float
        Tikhonov regularisation weight β for the kernel CG step.
    max_iter : int
        Outer alternating-minimisation iterations per scale.
    max_irls : int
        IRLS outer iterations for the image step.
    max_cg : int
        CG inner iterations for the image step.
    max_iter_k : int
        CG iterations for the kernel step.
    max_iter_rank : int
        iALM iterations for the low-rank step.
    iter_k_rank : int
        Inner kernel–rank alternation count per outer iteration.
    exp_a : float
        Hyper-Laplacian exponent *p* (0 < p ≤ 2; typical 0.5–0.8).
    thr_e : float
        IRLS smoothing parameter ε (avoids division by zero).
    alpha_multiplier : float
        Factor for scaling α across pyramid levels.
    threshold : float
        Kernel thresholding fraction (relative to max element).
    nb_lambda : float
        Regularisation weight for non-blind deconvolution.
    nb_alpha : float
        Hyper-Laplacian exponent for non-blind deconvolution.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        lambda_: float = 2e-3,
        sigma: float = 1.0,
        lambda_s: float = None,
        p_sparse: float = 0.5,
        rho_alm: float = 1.5,
        kernel_beta: float = 3e-3,
        max_iter: int = 7,
        max_irls: int = 3,
        max_cg: int = 200,
        max_iter_k: int = 50,
        max_iter_rank: int = 30,
        iter_k_rank: int = 3,
        exp_a: float = 0.8,
        thr_e: float = 1.0 / 1500,
        alpha_multiplier: float = 2.0,
        threshold: float = 0.05,
        nb_lambda: float = 3000.0,
        nb_alpha: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(name='LowRank-BD')

        assert kernel_size >= 3 and kernel_size % 2 == 1, \
            "kernel_size must be odd and >= 3"

        self.kernel_size      = kernel_size
        self.lambda_          = lambda_
        self.sigma            = sigma
        self.lambda_s         = lambda_s
        self.p_sparse         = p_sparse
        self.rho_alm          = rho_alm
        self.kernel_beta      = kernel_beta
        self.max_iter         = max_iter
        self.max_irls         = max_irls
        self.max_cg           = max_cg
        self.max_iter_k       = max_iter_k
        self.max_iter_rank    = max_iter_rank
        self.iter_k_rank      = iter_k_rank
        self.exp_a            = exp_a
        self.thr_e            = thr_e
        self.alpha_multiplier = alpha_multiplier
        self.threshold        = threshold
        self.nb_lambda        = nb_lambda
        self.nb_alpha         = nb_alpha
        self.verbose          = verbose

        self.history: Dict[str, list]    = {'kernel_diff': [], 'scale': []}
        self.hyperparams: Dict[str, Any] = {}

    # ==================================================================
    #  Main entry point
    # ==================================================================

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform blind deconvolution on the input blurred image.

        **Phase 1 — Blind kernel estimation (multi-scale):**
        At each scale, alternates among image estimation (IRLS+CG),
        kernel estimation (CG+projection), and low-rank iALM RPCA
        regularisation.  All steps work in the image domain following
        [6] (Dong et al. 2017).

        **Phase 2 — Non-blind image restoration:**
        Given the estimated kernel, recovers the sharp image via the
        hyper-Laplacian Split-Bregman solver [4].

        Parameters
        ----------
        image : np.ndarray
            Blurred input image, shape ``(H, W)`` or ``(H, W, 3)``.
            Accepts ``uint8 [0, 255]`` or ``float [0, 1]``.

        Returns
        -------
        restored : np.ndarray
            Restored image, same spatial shape as input,
            dtype ``int16``, range [0, 255].
        kernel : np.ndarray
            Estimated blur kernel (PSF), float64, non-negative,
            sums to one.
        """
        start_time = time.time()

        # --- Data preparation -----------------------------------------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        is_color = (y.ndim == 3 and y.shape[2] == 3)

        if is_color:
            ycbcr  = rgb_to_ycbcr(y)
            y_gray = ycbcr[:, :, 0]
        else:
            y_gray = y.copy()

        K     = self.kernel_size
        H, W  = y_gray.shape

        if self.verbose:
            print(f"[{self.name}] Image: {H}×{W}, "
                  f"Kernel: {K}×{K}")

        # ==============================================================
        #  PHASE 1:  Multi-Scale Blind Deconvolution  (image domain)
        # ==============================================================
        # Build coarse-to-fine scale pyramid
        scales = build_scale_pyramid(K)
        num_scales = len(scales)

        if self.verbose:
            print(f"[{self.name}] Scales: {scales}")

        # --- Initialisation -------------------------------------------
        min_scale = scales[0]

        # Initial kernel: delta function (no directional bias)
        k = np.zeros((min_scale, min_scale))
        k[min_scale // 2, min_scale // 2] = 1.0

        # Latent sharp image estimate (initialised from observation
        # at the first scale)
        x = None

        # --- Process each scale ---------------------------------------
        for si, Ki in enumerate(scales):
            if self.verbose:
                print(f"[{self.name}] Scale {si + 1}/{num_scales}: "
                      f"kernel {Ki}×{Ki}")

            # Down-sample blurred observation to current scale ratio
            ratio = Ki / K
            hw = (max(int(np.floor(H * ratio)), Ki + 2),
                  max(int(np.floor(W * ratio)), Ki + 2))
            y_small = resize_image(y_gray, hw)

            # Image estimate: init from observation or carry over
            if x is None:
                x = y_small.copy()
            else:
                x = resize_image(x, hw)

            # Up-scale kernel (skip for the first scale)
            if si > 0:
                k = resize_image(k, (Ki, Ki))
                k = normalize_kernel(k)

            # Scale-dependent regularisation weight:
            # α = λ₀ · m^(level − 0.5)
            # Larger α at coarser scales → smoother images → easier
            # kernel estimation;  decreases at finer scales to
            # preserve detail.
            # [6], function_multi_image_deblurring.m:
            #   alpha = min_alpha * alpha_multiplier^(k - 0.5)
            scale_idx = num_scales - 1 - si
            alpha = self.lambda_ * self.alpha_multiplier ** (
                scale_idx - 0.5
            )

            # --- Alternating minimisation at this scale ----------------
            for it in range(self.max_iter):
                k_prev = k.copy()

                # ---- Image step (IRLS + CG, [6]) ---------------------
                # Solves: min_x ||x⊛k - y||² + α D^T W D x
                x = optimize_image(
                    x, k, y_small, alpha,
                    self.max_irls, self.max_cg,
                    self.exp_a, self.thr_e,
                )

                # ---- Kernel + Low-rank step --------------------------
                for ir in range(self.iter_k_rank):
                    # Kernel estimation (CG + projection)
                    k = optimize_kernel(
                        x, k, y_small,
                        self.kernel_beta, self.max_iter_k,
                    )

                    # Low-rank regularisation (iALM RPCA, [6], [8])
                    if self.sigma > 0:
                        k = low_rank_regularization(
                            k, self.max_iter_rank,
                            self.lambda_s, self.p_sparse,
                            self.rho_alm,
                        )

                    # Project onto feasible set: k ≥ 0, Σk = 1
                    k = normalize_kernel(k)

                # Progressive thresholding
                # [1], blinddeconv_new2_cry.m:
                #   k(k < max(k)*threshold*i/imax) = 0
                k = normalize_kernel(
                    k,
                    self.threshold * (it + 1) / self.max_iter,
                )

                # — Convergence monitoring —
                diff = np.linalg.norm(k - k_prev)
                self.history['kernel_diff'].append(diff)
                self.history['scale'].append(Ki)

                if self.verbose:
                    print(f"  Iter {it + 1}/{self.max_iter}: "
                          f"‖Δk‖ = {diff:.6f}")

            # Centre kernel (no companion images to shift)
            k = center_kernel(k)
            k = normalize_kernel(k)

        # Final threshold
        k = normalize_kernel(k, self.threshold)

        if self.verbose:
            print(f"[{self.name}] Kernel estimated in "
                  f"{time.time() - start_time:.1f} s")

        # ==============================================================
        #  PHASE 2:  Non-Blind Deconvolution
        # ==============================================================
        if self.verbose:
            print(f"[{self.name}] Non-blind deconvolution "
                  f"(λ={self.nb_lambda}, α={self.nb_alpha}) ...")

        bhs = K // 2

        if is_color:
            # Deconvolve luminance channel; keep chrominance intact
            y_pad = np.pad(ycbcr[:, :, 0], bhs, mode='edge')
            for _ in range(4):
                y_pad = edgetaper(y_pad, k)

            restored_y = fast_deconv_hyper_laplacian(
                y_pad, k, self.nb_lambda, self.nb_alpha,
            )
            restored_y = restored_y[bhs: bhs + H, bhs: bhs + W]

            result = ycbcr.copy()
            result[:, :, 0] = restored_y
            result = ycbcr_to_rgb(result)
            result = np.clip(result, 0.0, 1.0)
        else:
            y_pad = np.pad(y_gray, bhs, mode='edge')
            for _ in range(4):
                y_pad = edgetaper(y_pad, k)

            result = fast_deconv_hyper_laplacian(
                y_pad, k, self.nb_lambda, self.nb_alpha,
            )
            result = result[bhs: bhs + H, bhs: bhs + W]
            result = np.clip(result, 0.0, 1.0)

        self.timer = time.time() - start_time

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda':      self.lambda_,
            'sigma':       self.sigma,
            'lambda_s':    self.lambda_s,
            'nb_lambda':   self.nb_lambda,
            'nb_alpha':    self.nb_alpha,
            'scales':      scales,
            'iterations':  sum(
                1 for s in self.history['scale'] if s == K
            ),
            'total_time':  self.timer,
        }

        if self.verbose:
            print(f"[{self.name}] Done in {self.timer:.1f} s")

        # Framework convention: return int16 image in [0, 255]
        result = np.round(result * 255.0).astype(np.int16)
        return result, k

    # ==================================================================
    #  Framework interface
    # ==================================================================

    def get_param(self) -> List[Tuple[str, Any]]:
        """Return current hyper-parameter list."""
        return [
            ('kernel_size',      self.kernel_size),
            ('lambda',           self.lambda_),
            ('sigma',            self.sigma),
            ('lambda_s',         self.lambda_s),
            ('p_sparse',         self.p_sparse),
            ('rho_alm',          self.rho_alm),
            ('kernel_beta',      self.kernel_beta),
            ('max_iter',         self.max_iter),
            ('max_irls',         self.max_irls),
            ('max_cg',           self.max_cg),
            ('max_iter_k',       self.max_iter_k),
            ('max_iter_rank',    self.max_iter_rank),
            ('iter_k_rank',      self.iter_k_rank),
            ('exp_a',            self.exp_a),
            ('thr_e',            self.thr_e),
            ('alpha_multiplier', self.alpha_multiplier),
            ('threshold',        self.threshold),
            ('nb_lambda',        self.nb_lambda),
            ('nb_alpha',         self.nb_alpha),
            ('verbose',          self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        """Update hyper-parameters from a dictionary."""
        for key, value in params.items():
            if key == 'lambda':
                self.lambda_ = value
            elif hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        """Convergence history (per-iteration kernel changes)."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Hyper-parameters and run-time statistics after process()."""
        return self.hyperparams


# ======================================================================
#  Convenience function
# ======================================================================

def run_algorithm(
    g: np.ndarray,
    kernel_size: int,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Convenience wrapper for running LowRank-BD.

    Parameters
    ----------
    g : np.ndarray
        Blurred image.
    kernel_size : int
        Expected PSF size (odd, ≥ 3).
    **kwargs
        Forwarded to :class:`LowRankBD`.

    Returns
    -------
    f_est      : np.ndarray — restored image
    k_est      : np.ndarray — estimated kernel
    hyperparams : dict
    history     : dict
    """
    algo = LowRankBD(kernel_size=kernel_size, **kwargs)
    f_est, k_est = algo.process(g)
    return f_est, k_est, algo.get_hyperparams(), algo.get_history()
