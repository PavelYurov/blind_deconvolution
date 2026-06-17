"""
denoisers.py

Self-contained denoiser dispatcher for the HTP blind deconvolution
pipeline.  Used by the configurable hooks in ``htp.HTP_BD``:

    pre_pyramid    — once on the full normalised image, before the ROI
                     pyramid is built (most impactful single hook).
    pre_kernel     — inside the alternating loop, on the latent image
                     fed to the H-step (must be MILD).
    pre_nonblind   — once before the final non-blind FFT-CG-SR-AL step.

All defaults in ``HTP_BD`` are ``None`` so the pipeline reproduces the
original Kotera–Šroubek–Milanfar (CAIP 2013) algorithm bit-for-bit
unless a user explicitly opts in.

Available methods
-----------------
    'none' / None
    'tv'                  — scikit-image Chambolle TV
    'nlm'                 — scikit-image Non-Local Means
    'bilateral'           — scikit-image bilateral
    'guided'              — He et al. guided filter (self-guided)
    'bm3d'                — bm3d package (state-of-art AWGN denoiser)
    'act'                 — Eslahi & Aghagolzadeh curvelet thresholding
                            (good for coloured / 1-over-f noise)
    'vst_bm3d'            — Anscombe VST + BM3D for Poisson / Poisson-
                            Gaussian noise (Mäkitalo–Foi 2013)
    'screenot'            — ScreeNOT SVD shrinkage (low-rank residual)
    'adaptive_median'     — impulse-noise removal via AMF

Notes
-----
* Methods that need a noise std (``nlm``, ``bilateral``, ``bm3d``) try
  the keyword first, otherwise call ``skimage.restoration.estimate_sigma``
  on the input.  When the caller already has σ from Chen / Pyatykh, it
  should pass it explicitly via ``sigma`` / ``sigma_color`` /
  ``sigma_psd`` / ``noise_var`` to avoid double estimation.
* This file deliberately has no dependency on the gbbid pipeline; only
  on modules already present next to it under ``htp_denoise/``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ['apply_denoiser']


# ────────────────────────────────────────────────────────────────────────────
# Self-guided filter (He et al., ECCV 2010) — small enough to keep inline
# ────────────────────────────────────────────────────────────────────────────
def _guided_filter(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    size = 2 * int(radius) + 1
    mean_I = uniform_filter(I, size)
    mean_p = uniform_filter(p, size)
    corr_Ip = uniform_filter(I * p, size)
    var_I = uniform_filter(I * I, size) - mean_I * mean_I
    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)
    return mean_a * I + mean_b


# ────────────────────────────────────────────────────────────────────────────
# Public dispatcher
# ────────────────────────────────────────────────────────────────────────────
def apply_denoiser(img: np.ndarray, method, **params) -> np.ndarray:
    """
    Apply a denoiser to a 2-D float image.

    Parameters
    ----------
    img : ndarray (H, W) — input image, expected in [0, 1] (float64).
    method : str or None
        One of the keys listed in the module docstring.  ``None`` /
        ``'none'`` returns ``img.copy()``.
    **params : method-specific keyword arguments (see below).

    Returns
    -------
    denoised : ndarray (H, W) — denoised image, same dtype as input.
    """
    if method is None or method == 'none':
        return img.copy()

    # ── tv ──────────────────────────────────────────────────────────────
    if method == 'tv':
        from skimage.restoration import denoise_tv_chambolle
        weight = float(params.get('weight', 0.05))
        max_num_iter = int(params.get('max_num_iter', 100))
        return denoise_tv_chambolle(img, weight=weight, max_num_iter=max_num_iter)

    # ── nlm ─────────────────────────────────────────────────────────────
    if method == 'nlm':
        from skimage.restoration import denoise_nl_means, estimate_sigma
        sigma = params.get('sigma', None)
        if sigma is None:
            sigma = float(estimate_sigma(img))
        patch_size = int(params.get('patch_size', 5))
        patch_distance = int(params.get('patch_distance', 6))
        h = float(params.get('h', 0.8 * sigma))
        return denoise_nl_means(
            img, h=h, patch_size=patch_size,
            patch_distance=patch_distance, fast_mode=True,
            sigma=sigma,
        )

    # ── bilateral ──────────────────────────────────────────────────────
    if method == 'bilateral':
        from skimage.restoration import denoise_bilateral, estimate_sigma
        sigma_color = params.get('sigma_color', None)
        if sigma_color is None:
            sigma_color = float(estimate_sigma(img))
        sigma_spatial = float(params.get('sigma_spatial', 1.0))
        return denoise_bilateral(
            img, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    # ── guided ──────────────────────────────────────────────────────────
    if method == 'guided':
        radius = int(params.get('radius', 5))
        eps = float(params.get('eps', 0.01))
        return _guided_filter(img, img, radius, eps)

    # ── bm3d ────────────────────────────────────────────────────────────
    if method == 'bm3d':
        try:
            import bm3d as bm3d_lib
        except ImportError as e:
            raise ImportError(
                "bm3d package required for method='bm3d': pip install bm3d"
            ) from e
        from skimage.restoration import estimate_sigma
        sigma_psd = params.get('sigma_psd', None)
        if sigma_psd is None:
            sigma_psd = float(estimate_sigma(img))
        return bm3d_lib.bm3d(img, sigma_psd=sigma_psd)

    # ── act (Adaptive Curvelet Thresholding) ───────────────────────────
    if method == 'act':
        from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
        nv = params.get('noise_var', None)
        ts = params.get('threshold_setting', 's')
        result, _ = act_denoise(img, noise_var=nv, threshold_setting=ts)
        return result

    # ── vst_bm3d (Generalized Anscombe VST + BM3D) ─────────────────────
    if method == 'vst_bm3d':
        from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise
        result, _ = vst_bm3d_denoise(
            img,
            noise_info=params.get('noise_info', None),
            a=params.get('a', None),
            b=params.get('b', None),
            sigma=params.get('sigma', None),
            stage_arg=params.get('stage_arg', None),
            verbose=params.get('verbose', False),
        )
        return result

    # ── screenot (SVD shrinkage) ───────────────────────────────────────
    if method == 'screenot':
        from blinddeconv.algorithms.mod_denoise.screenot import screenot_denoise
        return screenot_denoise(
            img,
            k=int(params.get('k', 10)),
            strategy=params.get('strategy', 'i'),
            mode=params.get('mode', 'full'),
            patch_size=params.get('patch_size', None),
            stride=params.get('stride', None),
        )

    # ── adaptive_median (impulse-noise removal) ────────────────────────
    if method == 'adaptive_median':
        from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import (
            detect_impulse_noise, adaptive_median_filter,
        )
        max_window = int(params.get('max_window', 7))
        mask = params.get('impulse_mask', None)
        if mask is None:
            mask = detect_impulse_noise(
                img,
                outlier_window=int(params.get('outlier_window', 5)),
                outlier_threshold=float(params.get('outlier_threshold', 0.15)),
            )
        return adaptive_median_filter(img, mask, max_window=max_window)

    raise ValueError(f"Unknown denoiser method: {method!r}")
