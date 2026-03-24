"""
lmgp.py

Blind Image Deblurring With Local Maximum Gradient Prior.

Reference:
    L. Chen, F. Fang, T. Wang, G. Zhang:
    "Blind Image Deblurring With Local Maximum Gradient Prior",
    CVPR, 2019.

Pipeline (mirrors MATLAB demo_deblurring.m):
    1. Normalise input to float64 [0, 1].
    2. Multi-scale blind deconvolution (blind_deconv) on grayscale input.
    3. Non-blind restoration via ringing_artifacts_removal.
    4. Return restored image (int16, [0, 255]) and kernel.
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


class LMGP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Local Maximum Gradient Prior (LMGP).

    Parameters
    ----------
    kernel_size   : int — spatial support of the unknown PSF (square, odd).
                    Default 27 (from demo_deblurring.m).
    lambda_lmg    : float — weight for LMG prior.
                    Default 4e-3.
    lambda_grad   : float — weight for L0 gradient prior.
                    Default 4e-3.
    xk_iter       : int — number of blind iterations per pyramid level.
                    Default 5.
    gamma_correct : float — gamma correction exponent applied before
                    kernel estimation.  1.0 = no correction.  Default 1.0.
    k_thresh      : float — final kernel threshold.
                    kernel values < max(k)/k_thresh are zeroed.
                    Default 20.
    lambda_tv     : float — weight for TV non-blind deconvolution.
                    Default 0.001.
    lambda_l0     : float — weight for L0 non-blind deconvolution.
                    Default 5e-4.
    weight_ring   : float — ringing suppression weight (0 = no suppression).
                    Default 1.0.
    denoise_eps   : float or None — guided-filter regularisation for
                    self-guided denoising inside gradient thresholding.
                    None = disabled.  Default None.
    denoise_radius: int — guided-filter radius (kernel = 2r+1).
                    Default 2.
    ensemble_denoise : bool — use ensemble of 3 guided filters with varied
                    (radius, eps) parameters, averaged before gradient
                    computation.  Requires denoise_eps to be set.
                    Default False.
    denoise_type  : str — denoiser type for gradient thresholding (point #3).
                    'guided' | 'bilateral' | 'bm3d' | 'nlm'.
                    Default 'guided'.
    denoise_bilateral_sigma_s : float — bilateral spatial sigma for
                    gradient thresholding denoiser.  Default 2.0.
    denoise_bilateral_sigma_r : float — bilateral range sigma for
                    gradient thresholding denoiser.  Default 0.1.
    denoise_bm3d_sigma : float — BM3D noise std for gradient thresholding
                    denoiser. Only used when denoise_type='bm3d'. Default 0.01.
    denoise_nlm_h : float — NLM filter strength for gradient thresholding
                    denoiser. Only used when denoise_type='nlm'. Default 0.01.
    grad_smooth_sigma : float or None — sigma for Gaussian smoothing of
                    blurred-image gradients Bx/By before kernel estimation.
                    None = disabled.  Default None.
    lmg_denoise_eps   : float or None — guided-filter eps applied to the
                    image before computing the LMG operator in L0_LMG_deblur.
                    Suppresses noise peaks that confuse Max_matrix.
                    None = disabled.  Default None.
    lmg_denoise_radius: int — guided-filter radius for LMG denoising.
                    Default 2.
    lmg_denoise_type : str — 'guided' | 'bilateral' | 'bm3d' | 'nlm'.
                    Which denoiser to use before LMG operator and for
                    graythresh stabilisation.  Default 'guided'.
    lmg_bilateral_sigma_s : float — bilateral filter spatial sigma.
                    Only used when lmg_denoise_type='bilateral'. Default 2.0.
    lmg_bilateral_sigma_r : float — bilateral filter range sigma.
                    Only used when lmg_denoise_type='bilateral'. Default 0.1.
    lmg_bm3d_sigma : float — BM3D noise std for LMG denoiser.
                    Only used when lmg_denoise_type='bm3d'. Default 0.01.
    lmg_nlm_h     : float — NLM filter strength for LMG denoiser.
                    Only used when lmg_denoise_type='nlm'. Default 0.01.
    use_soft_threshold : bool — use L1 soft thresholding instead of L0 hard
                    thresholding on gradients in L0_LMG_deblur.  More robust
                    to noise.  Default False.
    softmax_tau   : float or None — temperature for soft-max in Max_matrix.
                    Replaces hard argmax with weighted soft-max selection,
                    making the LMG operator a continuous function of
                    the input and eliminating chaotic instability under
                    noise.  None or 0 = original hard argmax.
                    Typical values: 0.01–0.5.  Default None.
    kernel_reg_weight : float — additional Tikhonov regularisation on PSF
                    estimation.  Penalises kernel complexity, suppressing
                    noise-induced artefacts. 0 = original.  Default 0.0.
    use_pmp_nonblind : bool — use PMP deblur_tv_pmpr for non-blind step
                    instead of L0Restoration.  Default False.
    pmp_lambda    : float — PMP prior weight for non-blind step.
                    Default 0.1.
    pmp_patch_r   : int — PMP patch size for non-blind step.
                    Default 3.
    pmp_quantile  : float — PMP quantile for noise-robust min pixels.
                    Default 0.0 (absolute minimum).
    """

    def __init__(
        self,
        kernel_size: int = 27,
        lambda_lmg: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        denoise_eps: float = None,
        denoise_radius: int = 2,
        ensemble_denoise: bool = False,
        denoise_type: str = 'guided',
        denoise_bilateral_sigma_s: float = 2.0,
        denoise_bilateral_sigma_r: float = 0.1,
        denoise_bm3d_sigma: float = 0.01,
        denoise_nlm_h: float = 0.01,
        grad_smooth_sigma: float = None,
        lmg_denoise_eps: float = None,
        lmg_denoise_radius: int = 2,
        lmg_denoise_type: str = 'guided',
        lmg_bilateral_sigma_s: float = 2.0,
        lmg_bilateral_sigma_r: float = 0.1,
        lmg_bm3d_sigma: float = 0.01,
        lmg_nlm_h: float = 0.01,
        use_soft_threshold: bool = False,
        softmax_tau: float = None,
        kernel_reg_weight: float = 0.0,
        use_pmp_nonblind: bool = False,
        pmp_lambda: float = 0.1,
        pmp_patch_r: int = 3,
        pmp_quantile: float = 0.0,
    ):
        super().__init__(name='LMGP-BD')

        self.kernel_size = kernel_size
        self.lambda_lmg = lambda_lmg
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.denoise_eps = denoise_eps
        self.denoise_radius = denoise_radius
        self.ensemble_denoise = ensemble_denoise
        self.denoise_type = denoise_type
        self.denoise_bilateral_sigma_s = denoise_bilateral_sigma_s
        self.denoise_bilateral_sigma_r = denoise_bilateral_sigma_r
        self.denoise_bm3d_sigma = denoise_bm3d_sigma
        self.denoise_nlm_h = denoise_nlm_h
        self.grad_smooth_sigma = grad_smooth_sigma
        self.lmg_denoise_eps = lmg_denoise_eps
        self.lmg_denoise_radius = lmg_denoise_radius
        self.lmg_denoise_type = lmg_denoise_type
        self.lmg_bilateral_sigma_s = lmg_bilateral_sigma_s
        self.lmg_bilateral_sigma_r = lmg_bilateral_sigma_r
        self.lmg_bm3d_sigma = lmg_bm3d_sigma
        self.lmg_nlm_h = lmg_nlm_h
        self.use_soft_threshold = use_soft_threshold
        self.softmax_tau = softmax_tau
        self.kernel_reg_weight = kernel_reg_weight
        self.use_pmp_nonblind = use_pmp_nonblind
        self.pmp_lambda = pmp_lambda
        self.pmp_patch_r = pmp_patch_r
        self.pmp_quantile = pmp_quantile

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
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 3. Blind kernel estimation ──────────────────────────────────
        # MATLAB: [kernel, interim_latent] = blind_deconv(yg, lambda_lmg, lambda_grad, opts)
        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'denoise_eps': self.denoise_eps,
            'denoise_radius': self.denoise_radius,
            'ensemble_denoise': self.ensemble_denoise,
            'denoise_type': self.denoise_type,
            'denoise_bilateral_sigma_s': self.denoise_bilateral_sigma_s,
            'denoise_bilateral_sigma_r': self.denoise_bilateral_sigma_r,
            'denoise_bm3d_sigma': self.denoise_bm3d_sigma,
            'denoise_nlm_h': self.denoise_nlm_h,
            'grad_smooth_sigma': self.grad_smooth_sigma,
            'lmg_denoise_eps': self.lmg_denoise_eps,
            'lmg_denoise_radius': self.lmg_denoise_radius,
            'lmg_denoise_type': self.lmg_denoise_type,
            'lmg_bilateral_sigma_s': self.lmg_bilateral_sigma_s,
            'lmg_bilateral_sigma_r': self.lmg_bilateral_sigma_r,
            'lmg_bm3d_sigma': self.lmg_bm3d_sigma,
            'lmg_nlm_h': self.lmg_nlm_h,
            'use_soft_threshold': self.use_soft_threshold,
            'softmax_tau': self.softmax_tau,
            'kernel_reg_weight': self.kernel_reg_weight,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_lmg, self.lambda_grad, opts,
        )

        # ── 4. Non-blind restoration ────────────────────────────────────
        # MATLAB: Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring)
        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring,
            use_pmp_nonblind=self.use_pmp_nonblind,
            pmp_lambda=self.pmp_lambda,
            pmp_patch_r=self.pmp_patch_r,
            pmp_quantile=self.pmp_quantile,
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_lmg': self.lambda_lmg,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'denoise_eps': self.denoise_eps,
            'lmg_denoise_eps': self.lmg_denoise_eps,
            'use_pmp_nonblind': self.use_pmp_nonblind,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_lmg', self.lambda_lmg),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('denoise_eps', self.denoise_eps),
            ('denoise_radius', self.denoise_radius),
            ('ensemble_denoise', self.ensemble_denoise),
            ('denoise_type', self.denoise_type),
            ('denoise_bilateral_sigma_s', self.denoise_bilateral_sigma_s),
            ('denoise_bilateral_sigma_r', self.denoise_bilateral_sigma_r),
            ('denoise_bm3d_sigma', self.denoise_bm3d_sigma),
            ('denoise_nlm_h', self.denoise_nlm_h),
            ('grad_smooth_sigma', self.grad_smooth_sigma),
            ('lmg_denoise_eps', self.lmg_denoise_eps),
            ('lmg_denoise_radius', self.lmg_denoise_radius),
            ('lmg_denoise_type', self.lmg_denoise_type),
            ('lmg_bilateral_sigma_s', self.lmg_bilateral_sigma_s),
            ('lmg_bilateral_sigma_r', self.lmg_bilateral_sigma_r),
            ('lmg_bm3d_sigma', self.lmg_bm3d_sigma),
            ('lmg_nlm_h', self.lmg_nlm_h),
            ('use_soft_threshold', self.use_soft_threshold),
            ('softmax_tau', self.softmax_tau),
            ('kernel_reg_weight', self.kernel_reg_weight),
            ('use_pmp_nonblind', self.use_pmp_nonblind),
            ('pmp_lambda', self.pmp_lambda),
            ('pmp_patch_r', self.pmp_patch_r),
            ('pmp_quantile', self.pmp_quantile),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
