"""
utils.py

Utility functions for DCP+SelfExSR integration.

This module provides blending and resizing helpers used to combine
the SR-enhanced reference image with the DCP latent estimate on
coarse pyramid levels. All heavy-lifting utilities are imported
from the original DCP and SelfExSR modules — nothing is duplicated.
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
                           num_scales: int,
                           n_sr_levels: int,
                           alpha_min: float = 0.3,
                           alpha_max: float = 0.7) -> float:
    """
    Compute per-level blending weight for the SR reference.

    On the coarsest level the DCP estimate is weakest, so we give
    more weight to SR (alpha → alpha_min, meaning 1-alpha → 0.7 for SR).
    As we approach the boundary where SR is no longer used, alpha
    increases toward alpha_max.

    Parameters
    ----------
    scale_idx   : current scale index (0 = finest, num_scales-1 = coarsest)
    num_scales  : total number of pyramid scales
    n_sr_levels : how many coarsest levels use SR enhancement
    alpha_min   : blend alpha at the very coarsest level (least trust in DCP)
    alpha_max   : blend alpha at the last SR-enhanced level (most trust in DCP)

    Returns
    -------
    alpha : float in [alpha_min, alpha_max]
    """
    if n_sr_levels <= 1:
        return (alpha_min + alpha_max) / 2.0

    # scale_idx counts from 0 (finest). Coarsest = num_scales - 1.
    # SR is applied on the n_sr_levels coarsest levels:
    #   indices [num_scales - n_sr_levels, ..., num_scales - 1]
    # Position within the SR range: 0 = coarsest SR level, n_sr_levels-1 = finest SR level
    coarsest_sr = num_scales - 1
    pos = coarsest_sr - scale_idx  # 0 at coarsest, grows toward finer
    t = pos / (n_sr_levels - 1)    # 0..1
    t = np.clip(t, 0.0, 1.0)

    # Linearly interpolate: coarsest → alpha_min, finest SR level → alpha_max
    return alpha_min + t * (alpha_max - alpha_min)
