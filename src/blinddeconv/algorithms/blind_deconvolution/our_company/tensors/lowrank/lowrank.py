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

    Parameters:
    kernel_size : int
        Expected maximum PSF size (must be odd, ≥ 3).
    lambda_ : float
        Base edge-regularisation weight α₀ for the image step.
        Scaled per pyramid level:
        α = λ · ``alpha_multiplier`` ^ (level − 0.5).
    sigma : float
        Low-rank regularisation flag/weight.  Set > 0 to enable,
        0 to disable.
    tau : float
        Proximal parameter for nuclear-norm thresholding (IRNN).
    delta : float
        Smoothing for the ``log det`` rank surrogate.
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
        IRNN iterations for the low-rank step.
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
        tau: float = 1e-5,
        delta: float = 1e-5,
        kernel_beta: float = 3e-3,
        max_iter: int = 7,
        max_irls: int = 3,
        max_cg: int = 200,
        max_iter_k: int = 50,
        max_iter_rank: int = 3,
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
        self.tau              = tau
        self.delta            = delta
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

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform blind deconvolution on the input blurred image.
        """
        start_time = time.time()

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

        scales = build_scale_pyramid(K)
        num_scales = len(scales)

        if self.verbose:
            print(f"[{self.name}] Scales: {scales}")

        min_scale = scales[0]

        k = np.zeros((min_scale, min_scale))
        k[min_scale // 2, min_scale // 2] = 1.0

        x = None

        for si, Ki in enumerate(scales):
            if self.verbose:
                print(f"[{self.name}] Scale {si + 1}/{num_scales}: "
                      f"kernel {Ki}×{Ki}")

            ratio = Ki / K
            hw = (max(int(np.floor(H * ratio)), Ki + 2),
                  max(int(np.floor(W * ratio)), Ki + 2))
            y_small = resize_image(y_gray, hw)

            if x is None:
                x = y_small.copy()
            else:
                x = resize_image(x, hw)

            if si > 0:
                k = resize_image(k, (Ki, Ki))
                k = normalize_kernel(k)

            scale_idx = num_scales - 1 - si
            alpha = self.lambda_ * self.alpha_multiplier ** (
                scale_idx - 0.5
            )

            tau_scale = self.tau * (si + 1) / num_scales


            for it in range(self.max_iter):
                k_prev = k.copy()

                x = optimize_image(
                    x, k, y_small, alpha,
                    self.max_irls, self.max_cg,
                    self.exp_a, self.thr_e,
                )

                for ir in range(self.iter_k_rank):

                    k = optimize_kernel(
                        x, k, y_small,
                        self.kernel_beta, self.max_iter_k,
                    )

                    if self.sigma > 0:
                        k = low_rank_regularization(
                            k, self.max_iter_rank,
                            tau_scale, self.delta,
                        )

                    k = normalize_kernel(k)

                k = normalize_kernel(
                    k,
                    self.threshold * (it + 1) / self.max_iter,
                )

                diff = np.linalg.norm(k - k_prev)
                self.history['kernel_diff'].append(diff)
                self.history['scale'].append(Ki)

                if self.verbose:
                    print(f"  Iter {it + 1}/{self.max_iter}: "
                          f"‖Δk‖ = {diff:.6f}")

            k = center_kernel(k)
            k = normalize_kernel(k)

        k = normalize_kernel(k, self.threshold)

        if self.verbose:
            print(f"[{self.name}] Kernel estimated in "
                  f"{time.time() - start_time:.1f} s")

        if self.verbose:
            print(f"[{self.name}] Non-blind deconvolution "
                  f"(λ={self.nb_lambda}, α={self.nb_alpha}) ...")

        bhs = K // 2

        if is_color:
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
            'tau':         self.tau,
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

        result = np.round(result * 255.0).astype(np.int16)
        return result, k

    def get_param(self) -> List[Tuple[str, Any]]:
        """Return current hyper-parameter list."""
        return [
            ('kernel_size',      self.kernel_size),
            ('lambda',           self.lambda_),
            ('sigma',            self.sigma),
            ('tau',              self.tau),
            ('delta',            self.delta),
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
