"""
nscp.py

Blind Image Deblurring via a Novel Sparse Channel Prior (NSCP).

Reference:
    D. Yang, X. Wu, H. Yin: "Blind Image Deblurring via a Novel Sparse
    Channel Prior", Mathematics, 2022.
    https://www.mdpi.com/2227-7390/10/8/1238

Pipeline (mirrors the original Python implementation by D. Yang):
    1. Normalise input to float32 [0, 1].
    2. Build Gaussian pyramid (coarse-to-fine).
    3. Alternating minimisation at each scale:
       a. Bright-channel weight  w_k = mu / (||B(l)||_0 + eps)   (Eq. 13)
       b. Dark-channel prior  p  via L0 thresholding               (Eq. 19)
       c. Gradient prior  g  via L0 thresholding                   (Eq. 20)
       d. Update latent image  l  (FFT, Eq. 18)
       e. Update blur kernel  k  (FFT in gradient domain, Eq. 22)
       f. Post-process kernel (clamp, threshold, crop, normalise)
    4. Final non-blind Wiener-filter deconvolution.
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

from .utils import (
    dark_channel,
    bright_channel,
    bcpl0norm,
    compute_gradients,
    threshold_gradient,
    gaussian_pyramid,
    upsample_l,
    upsample_small_kernel,
    clean_kernel,
    make_delta_kernel,
)
from .solvers import update_l, update_kernel, final_restore


class NSCP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Novel Sparse Channel Prior (D/B).

    The D/B prior combines the dark-channel prior (DCP) and bright-channel
    prior (BCP) to enforce sparsity of the ratio D(x) / (B(x) + eps),
    which favours sharp images over blurry ones.

    Parameters
    ----------
    kernel_size    : int — initial spatial support of the unknown PSF
                     (square, odd).  Default 25.
    kernel_max_size : int — maximum kernel size to prevent drift during
                      upsampling.  Default 35.
    num_scales     : int — number of Gaussian-pyramid levels.  Default 4.
    max_iter       : int — alternating-minimisation iterations per scale.
                     Default 10.
    mu             : float — weight for the D/B prior P(l).
                     Eq. (7): controls ||D(l)||_0 / (||B(l)||_0 + eps).
                     Default 0.003 (paper value; ||B||_0 is normalised by
                     pixel count internally).
    lambda_grad    : float — weight λ for the gradient-fidelity term
                     ||∇l − g||² in the latent-image sub-problem.
                     Default 0.02.
    xi             : float — weight ξ for the dark-channel fidelity term
                     ||D(l) − p||².  Default 0.02.
    theta          : float — numerator of gradient threshold T = θ/λ.
                     Default 0.003 (paper value; original code had 0.001).
    gamma          : float — L2 kernel regularisation weight.  Adapted
                     per scale internally:  0.5 / 0.2 / 0.05.
                     Default 1.0 (initial value, overridden per scale).
    epsilon        : float — small constant for numerical stability.
                     Default 1e-6.
    dcp_window     : int — patch size for the dark channel.  Default 15.
    bcp_window     : int — patch size for the bright channel.  Default 15.
    snr_const      : float — Wiener-filter SNR constant for the final
                     non-blind deconvolution.  Default 0.015.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        kernel_max_size: int = 35,
        num_scales: int = 4,
        max_iter: int = 10,
        mu: float = 0.003,
        lambda_grad: float = 0.02,
        xi: float = 0.02,
        theta: float = 0.003,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
        dcp_window: int = 15,
        bcp_window: int = 15,
        snr_const: float = 0.015,
    ):
        super().__init__(name='NSCP-BD')

        self.kernel_size = kernel_size
        self.kernel_max_size = kernel_max_size
        self.num_scales = num_scales
        self.max_iter = max_iter
        self.mu = mu
        self.lambda_grad = lambda_grad
        self.xi = xi
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.dcp_window = dcp_window
        self.bcp_window = bcp_window
        self.snr_const = snr_const

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full NSCP blind-deconvolution pipeline.

        Parameters
        ----------
        image : np.ndarray
            Input blurred image.  Grayscale (H, W) or colour (H, W, 3).
            Accepted dtypes: uint8 [0, 255]  or  float [0, 1].

        Returns
        -------
        x_final : np.ndarray, int16, [0, 255]
            Restored image.
        kernel  : np.ndarray, float
            Estimated blur kernel (normalised, sum ≈ 1).
        """
        start_time = time.time()

        # ── 1. Normalise to float32 [0, 1] ──────────────────────────────
        b_full = image.astype(np.float32)
        if b_full.max() > 1.0:
            b_full /= 255.0

        # ── 2. Build Gaussian pyramid (coarse → fine) ───────────────────
        pyramid = gaussian_pyramid(b_full, self.num_scales)

        # ── 3. Initialise latent image l and kernel k ────────────────────
        l = pyramid[0].astype(np.float32).copy()
        if l.max() > 1.0:
            l /= 255.0

        # Scale the initial kernel size to be appropriate for the coarsest
        # pyramid level.  The original code used the full kernel_size at the
        # coarsest scale, which made the kernel almost as large as the image
        # (e.g. 31×31 on a 32×32 image) — making kernel estimation fail.
        # DCP keeps the kernel at ~12-15 % of the image at every scale.
        coarsest_h, coarsest_w = pyramid[0].shape[:2]
        init_ks = max(3, min(self.kernel_size,
                             coarsest_h // 3, coarsest_w // 3))
        if init_ks % 2 == 0:
            init_ks += 1  # ensure odd
        k = make_delta_kernel(init_ks)

        # ── 4. Coarse-to-Fine Loop (Algorithm 2) ────────────────────────
        for scale_idx in range(len(pyramid)):
            b_scaled = pyramid[scale_idx]
            H, W = b_scaled.shape[:2]

            # Adapt gamma per scale (coarser → more regularisation)
            if scale_idx == 0:
                gamma_scale = 0.5
            elif scale_idx == 1:
                gamma_scale = 0.2
            else:
                gamma_scale = 0.05

            # Upsample l and k from previous scale
            if scale_idx > 0:
                l = upsample_l(l, (H, W))

                # Cap kernel to at most half the image dimension (DCP-style)
                max_kh = min(self.kernel_max_size, H // 2)
                max_kw = min(self.kernel_max_size, W // 2)
                max_kh = max(max_kh, 3)
                max_kw = max(max_kw, 3)
                k = upsample_small_kernel(
                    k, scale_factor=2, max_size=(max_kh, max_kw)
                )

                if k.sum() <= 1e-12:
                    k = make_delta_kernel(k.shape)
                else:
                    k = k / k.sum()

            # Scale DCP/BCP window size for current image dimensions.
            # A 15×15 window on a 32×32 image covers nearly half the image,
            # making the channels overly global.  Capping at H//3 keeps
            # the window local (same idea as the paper's patch Ψ).
            eff_dcp_w = min(self.dcp_window, max(3, H // 3))
            eff_bcp_w = min(self.bcp_window, max(3, W // 3))
            if eff_dcp_w % 2 == 0:
                eff_dcp_w -= 1
            if eff_bcp_w % 2 == 0:
                eff_bcp_w -= 1

            # ── Inner iterations (Alternating Minimisation) ──────────────
            num_pixels = H * W
            for it in range(self.max_iter):

                # (A) w_k = mu / (||B(l)||_0 + eps)           Eq. (13)
                # Normalise ||B||_0 by pixel count so that the threshold
                # w_k / xi stays in [0, 1] regardless of image size.
                # Without this, at coarse pyramid levels (32×32) the raw
                # pixel count (~1024) makes the threshold >1 and zeros
                # the ENTIRE p, destroying the dark-channel prior.
                B = bright_channel(l, window_size=eff_bcp_w)
                B_l0 = bcpl0norm(B) / num_pixels  # normalised to [0, 1]
                w_k = self.mu / (B_l0 + self.epsilon)

                # (B) Dark-channel prior → auxiliary variable p  Eq. (19)
                D = dark_channel(l, window_size=eff_dcp_w)
                threshold_val = w_k / self.xi

                p = l.copy()
                # L0 proximal: zero if |D|^2 < w_k/xi (squared comparison).
                # DCP: t = u**2 < lambda_dark / mybeta_pixel.
                mask_should_be_zero = D * D < threshold_val
                if l.ndim == 3:
                    mask_broadcast = np.repeat(
                        mask_should_be_zero[:, :, np.newaxis], l.shape[2], axis=2
                    )
                    p[mask_broadcast] = 0.0
                else:
                    p[mask_should_be_zero] = 0.0

                # (C) Gradient threshold → auxiliary variable g  Eq. (20)
                gh, gv = compute_gradients(l)
                g = threshold_gradient(
                    (gh, gv), self.theta, self.lambda_grad
                )

                # (D) Update latent image l                     Eq. (18)
                l = update_l(
                    l=l, k=k, b=b_scaled, g=g, p=p,
                    lam=self.lambda_grad, xi=self.xi,
                )

                # (E) Update blur kernel k                      Eq. (22)
                k = update_kernel(
                    l=l, b=b_scaled, gamma=gamma_scale,
                    image_shape=b_scaled.shape[:2], prev_k=k,
                )

                # (F) Post-process kernel
                k = clean_kernel(k)

                if k.sum() <= 1e-12:
                    k = make_delta_kernel(k.shape)

        # ── 5. Final non-blind deconvolution (Wiener) ────────────────────
        final_image = final_restore(b_full, k, snr_const=self.snr_const)

        # ── 6. Output ────────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'kernel_max_size': self.kernel_max_size,
            'num_scales': self.num_scales,
            'max_iter': self.max_iter,
            'mu': self.mu,
            'lambda_grad': self.lambda_grad,
            'xi': self.xi,
            'theta': self.theta,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'dcp_window': self.dcp_window,
            'bcp_window': self.bcp_window,
            'snr_const': self.snr_const,
            'time': time.time() - start_time,
        }

        x_final = final_image * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('kernel_max_size', self.kernel_max_size),
            ('num_scales', self.num_scales),
            ('max_iter', self.max_iter),
            ('mu', self.mu),
            ('lambda_grad', self.lambda_grad),
            ('xi', self.xi),
            ('theta', self.theta),
            ('gamma', self.gamma),
            ('epsilon', self.epsilon),
            ('dcp_window', self.dcp_window),
            ('bcp_window', self.bcp_window),
            ('snr_const', self.snr_const),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
