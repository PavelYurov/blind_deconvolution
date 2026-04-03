"""
si.py

Shift-Invariant Blind Deblurring.

Reference:
    Hua Cheng, "Shift-Invariant Deblurring", part of Super-Resolution
    Project (SR.pdf), 2011.

    Based on works by:
        - Xu & Jia (ref 16): edge selection + coarse kernel estimation
        - Perona & Malik (ref 17): nonlinear anisotropic diffusion
        - Osher & Rudin (ref 18): shock filter
        - Shan et al. (ref 13): multi-derivative non-blind deconvolution

Pipeline (mirrors MATLAB deblur.m):
    1. Normalise input to float64 [0, 1].
    2. Multi-scale coarse-to-fine loop:
       a. Resize to current pyramid level.
       b. Pad + edgetaper for circular boundary handling.
       c. Perona–Malik diffusion (noise suppression).
       d. Osher–Rudin shock filter (edge sharpening).
       e. Compute circular gradients.
       f. M_compute: mask of significant blur structures.
       g. H_compute: mask of significant shock-filter edges.
       h. coarse_kernel_est: estimate kernel + latent image.
    3. Non-blind multi-derivative deconvolution (Shan et al.).
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

from .utils import (
    fspecial_gaussian,
    edgetaper,
    imresize,
    gradient_circular,
    perona_malik,
    shock_filter,
    m_compute,
    h_compute,
    clean_kernel,
    center_kernel,
)
from .solvers import coarse_kernel_est, multi_deriv_deconv, tv_deconv


class SI_BD(DeconvolutionAlgorithm):
    """
    Shift-Invariant Blind Deblurring (Hua Cheng, 2011).

    Multi-scale coarse-to-fine blind deconvolution with edge selection,
    followed by multi-derivative non-blind restoration.

    Parameters
    ----------
    lambda_kernel : float
        Weight for gradient fidelity in the latent-image sub-problem
        during blind kernel estimation.  Default 0.001 (MATLAB ``lambda``).
    gamma : float
        L2-regularisation weight in the kernel sub-problem.
        Default 10 (MATLAB ``ganma``).
    lambda_deconv : float
        Regularisation weight for the final non-blind deconvolution.
        Default 0.05 (MATLAB second ``lambda``).
    ratios : tuple of float
        Downsampling ratios for the coarse-to-fine pyramid.
        Default (10, 7, 5, 3, 2, 1.5, 1).
    half_kernel_sizes : tuple of int
        Half-kernel sizes at each pyramid level (kernel = 2*ks + 1).
        Default (1, 2, 4, 7, 9, 12, 17).
    pm_iter : int
        Number of Perona–Malik diffusion iterations per level.
        Default 5.
    sf_iter : int
        Number of shock-filter iterations per level.
        Default 5.
    sf_dt : float
        Shock-filter time step.  Default 0.1.
    sf_h : float
        Shock-filter spatial step.  Default 1.0.
    edgetaper_sigma : float
        Sigma for the Gaussian kernel used in edgetaper.
        Default 2.0.
    n_edgetaper : int
        Number of edgetaper passes per level.  Default 3.
    lambda_tv : float
        TV regularisation weight for the final non-blind step
        (used when use_tv=True).  Default 0.002.
    use_tv : bool
        If True, use TV-ADMM for non-blind deconvolution (recommended).
        If False, use the original multi-derivative Wiener solver.
        Default True.
    """

    def __init__(
        self,
        lambda_kernel: float = 0.001,
        gamma: float = 10.0,
        lambda_deconv: float = 0.05,
        ratios: tuple = (10, 7, 5, 3, 2, 1.5, 1),
        half_kernel_sizes: tuple = (1, 2, 4, 7, 9, 12, 17),
        pm_iter: int = 5,
        sf_iter: int = 5,
        sf_dt: float = 0.1,
        sf_h: float = 1.0,
        edgetaper_sigma: float = 2.0,
        n_edgetaper: int = 3,
        lambda_tv: float = 0.002,
        use_tv: bool = True,
    ):
        super().__init__(name='SI-BD')

        self.lambda_kernel = lambda_kernel
        self.gamma = gamma
        self.lambda_deconv = lambda_deconv
        self.ratios = ratios
        self.half_kernel_sizes = half_kernel_sizes
        self.pm_iter = pm_iter
        self.sf_iter = sf_iter
        self.sf_dt = sf_dt
        self.sf_h = sf_h
        self.edgetaper_sigma = edgetaper_sigma
        self.n_edgetaper = n_edgetaper
        self.lambda_tv = lambda_tv
        self.use_tv = use_tv

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full blind deconvolution pipeline.

        Parameters
        ----------
        image : (H, W) grayscale image (uint8 or float).

        Returns
        -------
        x_final : (H', W') restored image, int16, [0, 255].
        kernel  : (ksize, ksize) estimated blur kernel.
        """
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        # MATLAB: I = im2double(imread(...));  I = I(:,:,1);
        I = image.astype(np.float64)
        if I.max() > 1.0:
            I /= 255.0
        if I.ndim == 3:
            I = I[:, :, 0]

        ratios = list(self.ratios)
        ks_list = list(self.half_kernel_sizes)
        n_scales = len(ratios)

        # ── 2. Initialise latent image at coarsest level ────────────────
        # MATLAB: Im = imresize(I, 1/ratio(1));  I_latent = Im;
        I_latent = imresize(I, 1.0 / ratios[0])

        tau_r = 0.0
        tau_s = 0.0
        k = None

        # ── 3. Multi-scale coarse-to-fine loop ─────────────────────────
        for i in range(n_scales):
            ks_i = ks_list[i]
            ksize = 2 * ks_i + 1

            # ── 3a. Resize blurred image to current level ───────────────
            # MATLAB: Im = imresize(I, 1/ratio(i));
            Im = imresize(I, 1.0 / ratios[i])

            # Align sizes of Im and I_latent
            # MATLAB: minxi = min([xim, xil]); etc.
            min_h = min(Im.shape[0], I_latent.shape[0])
            min_w = min(Im.shape[1], I_latent.shape[1])
            Im = Im[:min_h, :min_w]
            I_latent = I_latent[:min_h, :min_w]

            # ── 3b. Pad + edgetaper ─────────────────────────────────────
            # MATLAB: padarray(Im, [1 1]*ks(i), 'replicate', 'both')
            Im = np.pad(Im, ((ks_i, ks_i), (ks_i, ks_i)), mode='edge')
            I_latent = np.pad(I_latent,
                              ((ks_i, ks_i), (ks_i, ks_i)), mode='edge')

            # MATLAB: kernel_gaus = fspecial('gaussian', ksize, sigma);
            kernel_gaus = fspecial_gaussian(ksize, self.edgetaper_sigma)

            # MATLAB: for j=1:3; Im = edgetaper(Im, kernel_gaus); ... end
            for _ in range(self.n_edgetaper):
                Im = edgetaper(Im, kernel_gaus)
                I_latent = edgetaper(I_latent, kernel_gaus)

            # ── 3c. Perona–Malik diffusion ──────────────────────────────
            # MATLAB: I_sh = perona_malik(I_latent, iter);
            I_sh = perona_malik(I_latent, self.pm_iter)

            # ── 3d. Shock filter ────────────────────────────────────────
            # MATLAB: I_sh = shock_filter(I_sh, iter, dt, h);
            I_sh = shock_filter(I_sh, self.sf_iter, self.sf_dt, self.sf_h)

            # ── 3e. Circular gradients ──────────────────────────────────
            # MATLAB circular finite differences
            xim, yim = Im.shape
            Im_x, Im_y = gradient_circular(Im)
            Ish_x, Ish_y = gradient_circular(I_sh)

            # ── 3f. M mask (significant blurred structures) ─────────────
            # MATLAB: [M tau_r] = M_compute(Im_x, Im_y, ks, i, tau_r);
            # Note: MATLAB passes the whole ks array + 1-based index i.
            # We pass ks_val = ks_list[i] and a boolean for first scale.
            M, tau_r = m_compute(Im_x, Im_y, ks_i,
                                 is_first_scale=(i == 0),
                                 tau_r=tau_r)

            # ── 3g. H mask (significant shock-filter edges) ────────────
            # MATLAB: [H tau_s] = H_compute(Ish_x, Ish_y, M, ks, i, tau_s);
            H, tau_s = h_compute(Ish_x, Ish_y, M, ks_i,
                                 is_first_scale=(i == 0),
                                 tau_s=tau_s)

            # ── 3h. Select gradients and estimate kernel ────────────────
            # MATLAB: Is_x = Ish_x .* H;  Is_y = Ish_y .* H;
            Is_x = Ish_x * H
            Is_y = Ish_y * H

            # MATLAB: [k, I_latent] = coarse_kernel_est(...)
            k, I_latent = coarse_kernel_est(
                Is_x, Is_y, Im_x, Im_y, Im,
                ksize, self.lambda_kernel, self.gamma,
            )

            # ── 3i. Crop padding + upscale for next level ──────────────
            # MATLAB: if i < ind(end)
            #   I_latent = I_latent(ks(i)+1:xim-ks(i), ks(i)+1:yim-ks(i));
            #   I_latent = imresize(I_latent, ratio(i)/ratio(i+1), 'bicubic');
            if i < n_scales - 1:
                # Crop padding  (MATLAB 1-based ks(i)+1 : xim-ks(i))
                # In 0-based:  ks_i : xim - ks_i
                I_latent = I_latent[ks_i:xim - ks_i, ks_i:yim - ks_i]
                I_latent = imresize(
                    I_latent, ratios[i] / ratios[i + 1], method='bicubic'
                )

        # ── 4. Final kernel cleanup ────────────────────────────────────
        k = clean_kernel(k, threshold_ratio=20.0)
        k = center_kernel(k)

        # ── 5. Non-blind deconvolution ──────────────────────────────────
        # Im here is already padded from the last scale iteration
        if self.use_tv:
            # Crop padding first (TV solver does its own wrap_boundary)
            ks_last = ks_list[-1]
            Im_crop = Im[ks_last:xim - ks_last, ks_last:yim - ks_last]
            I_latent = tv_deconv(Im_crop, k, lambda_tv=self.lambda_tv)
        else:
            # MATLAB: I_latent = multi_deriv_deconv(Im, k, 0.05);
            I_latent = multi_deriv_deconv(Im, k, self.lambda_deconv)
            # Crop final padding
            ks_last = ks_list[-1]
            I_latent = I_latent[ks_last:xim - ks_last,
                                ks_last:yim - ks_last]

        # ── 6. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'lambda_kernel': self.lambda_kernel,
            'gamma': self.gamma,
            'lambda_deconv': self.lambda_deconv,
            'lambda_tv': self.lambda_tv,
            'use_tv': self.use_tv,
            'n_scales': n_scales,
            'final_kernel_size': k.shape[0],
            'time': time.time() - start_time,
        }

        I_latent = np.clip(I_latent, 0.0, 1.0)
        x_final = I_latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('lambda_kernel', self.lambda_kernel),
            ('gamma', self.gamma),
            ('lambda_deconv', self.lambda_deconv),
            ('ratios', self.ratios),
            ('half_kernel_sizes', self.half_kernel_sizes),
            ('pm_iter', self.pm_iter),
            ('sf_iter', self.sf_iter),
            ('sf_dt', self.sf_dt),
            ('sf_h', self.sf_h),
            ('edgetaper_sigma', self.edgetaper_sigma),
            ('n_edgetaper', self.n_edgetaper),
            ('lambda_tv', self.lambda_tv),
            ('use_tv', self.use_tv),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key in ('ratios', 'half_kernel_sizes'):
                    setattr(self, key, tuple(value))
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
