"""
utils.py

Utility functions for DCP+SelfExSR integration.

This module provides blending and resizing helpers used to combine
the SR-enhanced reference image with the DCP latent estimate on
intermediate pyramid levels.  All heavy-lifting utilities are
imported from the original DCP and SelfExSR modules — nothing is
duplicated.
"""

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom


def blend_images(img_dcp: np.ndarray,
                 img_sr: np.ndarray,
                 alpha: float) -> np.ndarray:
    """
    Linearly blend a DCP latent estimate with an SR reference.

    result = alpha * img_dcp + (1 - alpha) * img_sr

    Parameters
    ----------
    img_dcp : (H, W) or (H, W, D) — DCP latent estimate (float64, [0,1])
    img_sr  : same shape           — SR reference downsampled to match
    alpha   : float in [0, 1]      — weight for the DCP estimate.
              0 → pure SR, 1 → pure DCP.

    Returns
    -------
    blended : same shape, float64, clipped to [0, 1]
    """
    blended = alpha * img_dcp + (1.0 - alpha) * img_sr
    return np.clip(blended, 0.0, 1.0)


def resize_to_match(source: np.ndarray,
                    target_shape: tuple) -> np.ndarray:
    """
    Resize *source* so that its spatial dimensions match *target_shape*.

    Uses bicubic interpolation (order=3) via scipy.ndimage.zoom,
    consistent with MATLAB imresize used elsewhere in DCP.

    Parameters
    ----------
    source       : (Hs, Ws) or (Hs, Ws, D)
    target_shape : (Ht, Wt) — desired spatial size

    Returns
    -------
    resized : (Ht, Wt) or (Ht, Wt, D), float64
    """
    Ht, Wt = target_shape[:2]
    Hs, Ws = source.shape[:2]

    if Hs == Ht and Ws == Wt:
        return source.copy()

    zoom_h = Ht / Hs
    zoom_w = Wt / Ws

    if source.ndim == 3:
        factors = (zoom_h, zoom_w, 1.0)
    else:
        factors = (zoom_h, zoom_w)

    resized = ndimage_zoom(source, factors, order=3)
    # Exact shape may differ by ±1 pixel due to rounding — crop/pad
    resized = resized[:Ht, :Wt] if resized.ndim == 2 else resized[:Ht, :Wt, :]
    return resized.astype(np.float64)


def compute_sr_blend_alpha(scale_idx: int,
                           warmup_boundary: int,
                           n_sr_levels: int,
                           alpha_min: float = 0.5,
                           alpha_max: float = 0.85) -> float:
    """
    Compute per-level blending weight for the SR reference.

    SR-enhanced levels sit between the warmup phase (coarsest) and
    the fine phase (finest).  The first SR level (right after warmup)
    gets the MOST SR influence (alpha → alpha_min) because the DCP
    kernel is still rough.  The last SR level (just before fine
    levels) gets the LEAST (alpha → alpha_max) since the kernel has
    improved.

    Parameters
    ----------
    scale_idx        : current scale index (0 = finest, higher = coarser)
    warmup_boundary  : scale index of the last warmup level
                       (SR levels are at indices warmup_boundary-1 down to
                       warmup_boundary - n_sr_levels)
    n_sr_levels      : how many levels use SR enhancement
    alpha_min        : blend alpha on the coarsest SR level (most SR trust)
    alpha_max        : blend alpha on the finest SR level (least SR trust)

    Returns
    -------
    alpha : float in [alpha_min, alpha_max]
    """
    if n_sr_levels <= 1:
        return (alpha_min + alpha_max) / 2.0

    # SR levels span: warmup_boundary-1 (coarsest SR) .. warmup_boundary-n_sr_levels (finest SR)
    # pos = 0 at the coarsest SR level, n_sr_levels-1 at the finest
    coarsest_sr = warmup_boundary - 1
    pos = coarsest_sr - scale_idx   # 0 at coarsest SR, grows toward finer
    t = pos / (n_sr_levels - 1)     # 0..1
    t = float(np.clip(t, 0.0, 1.0))

    # Linearly interpolate: coarsest SR → alpha_min, finest SR → alpha_max
    return alpha_min + t * (alpha_max - alpha_min)


def enhance_with_sr_detail(img_dcp: np.ndarray,
                           img_sr: np.ndarray,
                           beta: float,
                           sigma: float = 2.0) -> np.ndarray:
    """
    Transfer high-frequency detail from SR reference to DCP latent.

    Instead of linear pixel blending (which shifts global intensity and
    can weaken existing DCP edges), this function:

    1. Extracts high-frequency *detail* from both images by subtracting
       a Gaussian-blurred version of each.
    2. Computes a per-pixel *boost mask* that is ≈1 where the SR
       reference has strong detail but DCP does not, and ≈0 where DCP
       already has strong edges.
    3. Injects ``beta * boost_mask * sr_detail`` into the DCP image.

    This is more targeted than pixel blending: DCP's smooth regions and
    existing edges are preserved, and only genuinely missing
    high-frequency content from SR is added.

    Parameters
    ----------
    img_dcp : (H, W) — DCP latent estimate, float64 [0, 1]
    img_sr  : (H, W) — SR reference at matching resolution
    beta    : float ≥ 0 — injection strength.  0 = no change.
              Typical range: 0.15–0.5.
    sigma   : float — Gaussian sigma for low-pass separation (default 2.0).
              Larger → more conservative (coarser detail treated as
              "structure", not injected).

    Returns
    -------
    enhanced : (H, W) float64, clipped to [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    # Separate high-freq detail from low-freq structure
    sr_detail = img_sr - gaussian_filter(img_sr, sigma=sigma)
    dcp_detail = img_dcp - gaussian_filter(img_dcp, sigma=sigma)

    # Per-pixel injection weight: high where SR has detail DCP lacks
    sr_energy = np.abs(sr_detail)
    dcp_energy = np.abs(dcp_detail)
    boost_mask = sr_energy / (dcp_energy + sr_energy + 1e-10)

    enhanced = img_dcp + beta * boost_mask * sr_detail
    return np.clip(enhanced, 0.0, 1.0)
