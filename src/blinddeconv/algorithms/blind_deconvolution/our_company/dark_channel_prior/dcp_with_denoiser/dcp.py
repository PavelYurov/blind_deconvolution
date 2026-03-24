"""
dcp.py

Blind Image Deblurring Using Dark Channel Prior (DCP).

Reference:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR 2016.

Pipeline (mirrors MATLAB demo_deblurring.m):
    1. Normalise input to float64 [0, 1].
    2. Convert to grayscale for kernel estimation.
    3. Multi-scale blind deconvolution (blind_deconv).
    4. Non-blind restoration on the full (colour) image
       via ringing_artifacts_removal.
    5. Return restored image (int16, [0, 255]) and kernel.
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

from .solvers import blind_deconv, ringing_artifacts_removal


class DCP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Dark Channel Prior.

    Parameters
    ----------
    kernel_size  : int — spatial support of the unknown PSF (square, odd).
    lambda_dark  : float — weight for L0 intensity (dark-channel) prior.
                   Default 4e-3 (from demo_deblurring.m).
    lambda_grad  : float — weight for L0 gradient prior.
                   Default 4e-3.
    xk_iter      : int — number of blind iterations per pyramid level.
                   Default 5.
    gamma_correct : float — gamma correction exponent applied before
                    kernel estimation.  1.0 = no correction.  Default 1.0.
    k_thresh     : float — final kernel threshold.
                   kernel values < max(k)/k_thresh are zeroed.
                   Default 20.
    lambda_tv    : float — weight for TV non-blind deconvolution.
                   Default 0.003.
    lambda_l0    : float — weight for L0 non-blind deconvolution.
                   Default 5e-4.
    weight_ring  : float — ringing suppression weight (0 = no suppression).
                   Default 1.0.
    denoise_eps   : float or None — guided filter regularisation eps
                    for denoising the intermediate latent image before
                    gradient computation in kernel estimation.
                    None = disabled (original behaviour).
                    Typical values: 0.001 – 0.02.
    denoise_radius : int — guided filter window radius.
                     Default 2.
    threshold_boost : float — multiplier for the gradient threshold on
                      the first blind iteration.  Linearly decays to 1.0
                      by the last iteration.  Values > 1 force early
                      iterations to use only the strongest edges,
                      suppressing noise-induced gradients.
                      1.0 = disabled (original behaviour).
                      Typical values for noisy images: 2.0 – 5.0.
    grad_smooth_sigma : float or None — Gaussian sigma for smoothing
                        the blurred-image gradients (Bx, By) before
                        kernel estimation.  Suppresses noise-induced
                        spurious gradients while preserving structural
                        edges.  None = disabled (original behaviour).
                        Typical values: 0.5 – 2.0.
    residual_beta : float or None — soft-thresholding level for the
                    observation-residual correction in the blind loop.
                    After each iteration, computes r = S*k − B and
                    absorbs noise via s = SoftThresh(r, β).  The next
                    iteration uses B_eff = B + s.  None = disabled.
                    Typical values for σ≈0.02 noise: 0.005 – 0.03.
    noise_lambda : float — L1 weight for the noise-residual variable
                   in the non-blind TV deconvolution step (Fang et al.).
                   Absorbs noise/outliers into an auxiliary variable s
                   so the restored image is cleaner.  0 = disabled.
                   Typical values: 0.001 – 0.01.
    gradient_confidence_tau : float or None — soft confidence weighting
                    for latent gradients before kernel estimation.
                    conf = |∇S| / (|∇S| + τ).  Strong edges → weight ≈ 1,
                    weak noise gradients → weight → 0.
                    None = disabled.  Typical values: 0.001 – 0.01.
    k_iter_clean : float or None — per-iteration kernel thresholding.
                   After each kernel estimation, zeros kernel entries
                   below max(k) / k_iter_clean.  Keeps iterative kernel
                   estimates clean.  None = disabled.
                   Typical values: 5.0 – 15.0.
    noise_adapt  : float or None — noise-adaptive threshold floor.
                   Estimates noise σ from the blurred image (MAD of
                   gradients) and raises the gradient threshold if
                   noise_adapt · σ² > auto-threshold.  None = disabled.
                   Typical values: 50 – 500.
    dc_quantile  : float — quantile for dark channel prior computation.
                   0.0 = strict minimum (original behaviour).
                   Positive values (e.g. 0.05) ignore the darkest
                   quantile of pixels in each patch, making the dark
                   channel robust to noise outliers.
                   Typical values: 0.0 – 0.1.
    pyramid_adapt_eps : bool — if True, automatically scales denoise_eps
                   per pyramid level (less denoising at coarse scales
                   where downsampling already suppresses noise, full eps
                   at the finest scale).  Default False.
    nsr          : float — Wiener-style noise-to-signal ratio added to
                   the FFT denominator in L0 solvers.  Prevents noise
                   amplification during deconvolution.  0 = disabled.
                   Typical values: 0.001 – 0.02.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.003,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        denoise_eps: float = None,
        denoise_radius: int = 2,
        threshold_boost: float = 1.0,
        grad_smooth_sigma: float = None,
        residual_beta: float = None,
        noise_lambda: float = 0.0,
        gradient_confidence_tau: float = None,
        k_iter_clean: float = None,
        noise_adapt: float = None,
        dc_quantile: float = 0.0,
        pyramid_adapt_eps: bool = False,
        nsr: float = 0.0,
    ):
        super().__init__(name='DCP-BD')

        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.denoise_eps = denoise_eps
        self.denoise_radius = denoise_radius
        self.threshold_boost = threshold_boost
        self.grad_smooth_sigma = grad_smooth_sigma
        self.residual_beta = residual_beta
        self.noise_lambda = noise_lambda
        self.gradient_confidence_tau = gradient_confidence_tau
        self.k_iter_clean = k_iter_clean
        self.noise_adapt = noise_adapt
        self.dc_quantile = dc_quantile
        self.pyramid_adapt_eps = pyramid_adapt_eps
        self.nsr = nsr

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────────
        # MATLAB: yg = im2double(rgb2gray(y))
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 3. Blind kernel estimation ──────────────────────────────────
        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'denoise_eps': self.denoise_eps,
            'denoise_radius': self.denoise_radius,
            'threshold_boost': self.threshold_boost,
            'grad_smooth_sigma': self.grad_smooth_sigma,
            'residual_beta': self.residual_beta,
            'gradient_confidence_tau': self.gradient_confidence_tau,
            'k_iter_clean': self.k_iter_clean,
            'noise_adapt': self.noise_adapt,
            'dc_quantile': self.dc_quantile,
            'pyramid_adapt_eps': self.pyramid_adapt_eps,
            'nsr': self.nsr,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_dark, self.lambda_grad, opts
        )

        # ── 4. Non-blind restoration ────────────────────────────────────
        # MATLAB: Latent = ringing_artifacts_removal(y, kernel, ...)
        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring,
            noise_lambda=self.noise_lambda,
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_dark', self.lambda_dark),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('denoise_eps', self.denoise_eps),
            ('denoise_radius', self.denoise_radius),
            ('threshold_boost', self.threshold_boost),
            ('grad_smooth_sigma', self.grad_smooth_sigma),
            ('residual_beta', self.residual_beta),
            ('noise_lambda', self.noise_lambda),
            ('gradient_confidence_tau', self.gradient_confidence_tau),
            ('k_iter_clean', self.k_iter_clean),
            ('noise_adapt', self.noise_adapt),
            ('dc_quantile', self.dc_quantile),
            ('pyramid_adapt_eps', self.pyramid_adapt_eps),
            ('nsr', self.nsr),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
