"""
Fast Bayesian Blind Deconvolution with Huber Super-Gaussian Priors (FBbd-HsGp).

Framework wrapper implementing the EM-based blind deconvolution algorithm
proposed in:

    Zhou, X., Vega, M., Zhou, F., Molina, R., & Katsaggelos, A.K. (2017).
    "Fast Bayesian blind deconvolution with Huber super Gaussian priors."
    Digital Signal Processing, 60, 122–133.

The algorithm avoids the well-known trivial-solution pitfall of joint MAP
estimation (see Levin et al., 2009/2011, MIT-CSAIL-TR-2009-014) by using
**marginal** kernel estimation through the EM framework: at each iteration
the posterior over the latent image is integrated out analytically (under a
circulant covariance approximation), and the kernel is updated by maximising
the resulting marginal likelihood.

Key features:
    * Huber super-Gaussian prior on image gradients for edge-preserving
      regularisation without the singularity-at-zero problem of L1/hyper-
      Laplacian priors (Sec. 3, Eq. 10–14).
    * Fully FFT-based E-step with O(N log N) cost (Sec. 4).
    * Coarse-to-fine (multi-scale pyramid) estimation following
      Levin et al. (2011) and Almeida & Figueiredo (ICIP 2013).
    * Optional automatic update of noise precision β and regularisation α
      via evidence maximisation (Sec. 6).

Modules:
    - utils.py   : FFT helpers, gradient operators, Huber weights, pyramid,
                   kernel post-processing.
    - solvers.py : Pure numerical routines for E-step, W-step, M-step, and
                   hyperparameter updates.
    - FBbdHsGp.py: This file — main class & multi-scale EM orchestration.
"""

import numpy as np
import time
from numpy.fft import fft2, ifft2
from typing import Tuple, List, Any, Dict
from scipy.ndimage import zoom

from .utils import (precompute_gradient_operators, init_kernel, center_kernel, 
                    build_pyramid, pad_image, crop_image)
from .solvers import (compute_variational_stats, update_huber_weights_variational, 
                      solve_image_sadmm, solve_kernel_admm)

# ── Robust import of framework base class ──────────────────────────────
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward until ``pyproject.toml`` is found."""
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

from blinddeconv.algorithms.base import DeconvolutionAlgorithm  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
#  Main class
# ═══════════════════════════════════════════════════════════════════════

class FBbdHsGp(DeconvolutionAlgorithm):
    """
    Fast Bayesian blind deconvolution with Huber super Gaussian priors.
    Zhou et al., 2017.
    """
    def __init__(self, 
                 kernel_shape=(27, 27),
                 n_scales=None,
                 scale_factor=0.5,
                 n_outer=30,      # Adequate iterations
                 sigma_init=0.03, # Start with assumption of noise to regularize
                 sigma_min=0.005, # Refine later
                 beta_v=0.05,     # SADMM Image penalty (controls intermediate smoothness)
                 beta_h=200.0,    # Kernel ADMM penalty
                 epsilon=0.002):  # Huber threshold
        
        super().__init__(name="FBbdHsGp")
        
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)
        if kernel_shape[0] % 2 == 0: kernel_shape = (kernel_shape[0]+1, kernel_shape[1])
        if kernel_shape[1] % 2 == 0: kernel_shape = (kernel_shape[0], kernel_shape[1]+1)
            
        self.kernel_shape = kernel_shape
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.n_outer = n_outer
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.beta_v = beta_v 
        self.beta_h = beta_h
        self.epsilon = epsilon
        
    def process(self, image):
        # 1. Normalize
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0
        
        # 2. Pad Image (Edge Mode)
        # Avoids boundary ringing which destroys kernel estimation
        pad_sz = max(self.kernel_shape)
        y_padded = pad_image(img_float, pad_sz)
        
        # 3. Pyramid
        if self.n_scales is None:
            min_dim = min(img_float.shape)
            self.n_scales = int(np.floor(np.log(min_dim / max(self.kernel_shape)) / np.log(1/self.scale_factor)))
            self.n_scales = max(1, self.n_scales)
            
        pyramid = build_pyramid(y_padded, self.n_scales, self.scale_factor)
        
        # 4. Init Kernel
        k_est = init_kernel(self.kernel_shape)
        
        # 5. Multiscale Loop
        x_est = None
        
        for scale_idx, y_s in enumerate(pyramid):
            H_s, W_s = y_s.shape
            
            if x_est is not None:
                x_est = zoom(x_est, (H_s / x_est.shape[0], W_s / x_est.shape[1]), order=3)
            else:
                x_est = y_s.copy()
            
            F_dx, F_dy, F_dtd = precompute_gradient_operators((H_s, W_s))
            
            # Use fixed beta_v as per paper SADMM recommendation (high penalty for smoothness)
            # but we can decay sigma to trust data more later.
            
            for it in range(self.n_outer):
                prog = it / (self.n_outer - 1)
                curr_sigma = self.sigma_init * (1 - prog) + self.sigma_min * prog
                
                # --- W-Step: Estimate Weights ---
                # We need mu_x and Sigma from the distribution
                mu_x, Sigma_spec = compute_variational_stats(
                    y_s, k_est, 
                    np.ones_like(y_s), np.ones_like(y_s), # Dummy weights for first stats
                    curr_sigma, F_dx, F_dy
                )
                
                # Update weights using the Variational (Second Moment) formula
                xi_x, xi_y = update_huber_weights_variational(
                    mu_x, Sigma_spec, F_dx, F_dy, self.epsilon
                )
                
                # --- E-Step: Update Image (SADMM) ---
                x_est = solve_image_sadmm(
                    y_s, k_est, xi_x, xi_y, 
                    curr_sigma, self.beta_v, 
                    F_dx, F_dy, F_dtd, n_iters=1
                )
                
                # --- M-Step: Update Kernel ---
                # Passing Sigma_spec is CRITICAL here
                k_est = solve_kernel_admm(
                    y_s, x_est, Sigma_spec, 
                    self.kernel_shape, self.beta_h, n_iters=5
                )
                
                # Center kernel to prevent drift
                k_est = center_kernel(k_est)
                
        # 6. Final Non-Blind Deconvolution
        # High quality restoration on full image
        F_dx, F_dy, F_dtd = precompute_gradient_operators(y_padded.shape)
        
        # Calculate final accurate weights
        mu_final, Sigma_final = compute_variational_stats(
            y_padded, k_est, np.ones_like(y_padded), np.ones_like(y_padded), 
            self.sigma_min, F_dx, F_dy
        )
        xi_x, xi_y = update_huber_weights_variational(
            mu_final, Sigma_final, F_dx, F_dy, self.epsilon
        )
        
        # Solve with low beta_v (or just Wiener, but SADMM is robust)
        # Low beta_v allows sharp edges in the final output
        final_x_padded = solve_image_sadmm(
            y_padded, k_est, xi_x, xi_y, 
            self.sigma_min, beta_v=0.005, 
            F_dx=F_dx, F_dy=F_dy, F_dtd=F_dtd, n_iters=30
        )
        
        # Crop and finish
        final_x = crop_image(final_x_padded, pad_sz)
        final_x = np.clip(final_x, 0, 1)
        res_image = (final_x * 255).astype(np.int16)
        
        return res_image, k_est.astype(np.float64)

    def get_param(self):
        return [("kernel_shape", self.kernel_shape)]

    def change_param(self, params):
        for k, v in params.items():
            if hasattr(self, k): setattr(self, k, v)