import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

def blend_images(img_dcp: np.ndarray,
                 img_sr: np.ndarray,
                 alpha: float) -> np.ndarray:

    blended = alpha * img_dcp + (1.0 - alpha) * img_sr
    return np.clip(blended, 0.0, 1.0)

def resize_to_match(source: np.ndarray,
                    target_shape: tuple) -> np.ndarray:

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

    resized = resized[:Ht, :Wt] if resized.ndim == 2 else resized[:Ht, :Wt, :]
    return resized.astype(np.float64)

def compute_sr_blend_alpha(scale_idx: int,
                           warmup_boundary: int,
                           n_sr_levels: int,
                           alpha_min: float = 0.5,
                           alpha_max: float = 0.85) -> float:

    if n_sr_levels <= 1:
        return (alpha_min + alpha_max) / 2.0

    coarsest_sr = warmup_boundary - 1
    pos = coarsest_sr - scale_idx
    t = pos / (n_sr_levels - 1)
    t = float(np.clip(t, 0.0, 1.0))

    return alpha_min + t * (alpha_max - alpha_min)

def enhance_with_sr_detail(img_dcp: np.ndarray,
                           img_sr: np.ndarray,
                           beta: float,
                           sigma: float = 2.0) -> np.ndarray:

    from scipy.ndimage import gaussian_filter

    sr_detail = img_sr - gaussian_filter(img_sr, sigma=sigma)
    dcp_detail = img_dcp - gaussian_filter(img_dcp, sigma=sigma)

    sr_energy = np.abs(sr_detail)
    dcp_energy = np.abs(dcp_detail)
    boost_mask = sr_energy / (dcp_energy + sr_energy + 1e-10)

    enhanced = img_dcp + beta * boost_mask * sr_detail
    return np.clip(enhanced, 0.0, 1.0)
