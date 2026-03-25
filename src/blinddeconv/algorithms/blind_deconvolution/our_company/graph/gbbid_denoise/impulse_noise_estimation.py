"""
impulse_noise_estimation.py

Detection and removal of impulse (salt-and-pepper) noise.

Impulse noise appears as isolated pixels at extreme values (0 or 255).
It must be handled BEFORE blind deconvolution — otherwise the PSF
estimation treats impulses as image features, and the blur kernel
"smears" them into star-shaped artifacts.

Detection strategy (no reference image needed):
    1. Histogram analysis: count pixels at/near extreme values (0, 255).
    2. Local outlier detection: flag pixels whose value differs from
       their local median by more than a threshold.
    3. Combine both signals to decide presence/absence and estimate
       impulse noise density.

Removal:
    Adaptive Median Filter (AMF) — applies median only to detected
    impulse pixels, preserving all other pixels untouched.
    This is critical: a standard median filter blurs the entire image.

Dependencies: numpy, scipy (only).
"""

import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter

__all__ = [
    'detect_impulse_noise',
    'estimate_impulse_density',
    'adaptive_median_filter',
    'remove_impulse_noise',
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Detection
# ═════════════════════════════════════════════════════════════════════════════

def _histogram_extremes(image, low_thresh=0.01, high_thresh=0.99):
    """
    Count fraction of pixels at extreme values.

    For [0,1] images: values ≤ low_thresh or ≥ high_thresh.
    Salt-and-pepper noise creates spikes at the distribution tails.

    Returns
    -------
    frac_low : float — fraction of near-zero pixels
    frac_high : float — fraction of near-max pixels
    frac_total : float — total fraction of extreme pixels
    """
    total = image.size
    frac_low = np.count_nonzero(image <= low_thresh) / total
    frac_high = np.count_nonzero(image >= high_thresh) / total
    return frac_low, frac_high, frac_low + frac_high


def _local_outlier_mask(image, window_size=5, threshold=0.15):
    """
    Detect pixels that are local outliers (differ significantly from
    their neighborhood median).

    Parameters
    ----------
    image : ndarray, H×W, float [0, 1]
    window_size : int — median filter window (odd)
    threshold : float — minimum difference from local median to flag
                a pixel as an outlier (in [0, 1] scale)

    Returns
    -------
    mask : bool ndarray, H×W — True where impulse noise is suspected
    """
    local_med = _scipy_median_filter(image, size=window_size)
    diff = np.abs(image - local_med)
    return diff > threshold


def detect_impulse_noise(image, low_thresh=0.01, high_thresh=0.99,
                         outlier_window=5, outlier_threshold=0.15,
                         density_threshold=0.0005):
    """
    Detect whether an image contains impulse (salt-and-pepper) noise.

    Uses two complementary signals:
        1. Histogram extremes — fraction of pixels near 0 or 1.
        2. Local outlier detection — pixels far from local median.

    Parameters
    ----------
    image : ndarray, H×W or H×W×C, float64 [0, 1] or uint8 [0, 255]
    low_thresh : float — lower bound for "salt" pixels (in [0, 1])
    high_thresh : float — upper bound for "pepper" pixels (in [0, 1])
    outlier_window : int — window size for local median (odd)
    outlier_threshold : float — min diff from local median for outlier
    density_threshold : float — minimum estimated density to declare
                        impulse noise present (default 0.5%)

    Returns
    -------
    result : dict
        'has_impulse'    — bool: True if impulse noise detected
        'density'        — float: estimated impulse noise density [0, 1]
        'frac_low'       — float: fraction of near-zero pixels
        'frac_high'      — float: fraction of near-max pixels
        'outlier_frac'   — float: fraction flagged as local outliers
        'impulse_mask'   — bool ndarray H×W: per-pixel impulse map
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    # Work on grayscale for detection
    if img.ndim == 3:
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img

    # Signal 1: histogram extremes
    frac_low, frac_high, frac_total = _histogram_extremes(
        gray, low_thresh, high_thresh)

    # Signal 2: local outlier detection
    outlier_mask = _local_outlier_mask(
        gray, outlier_window, outlier_threshold)
    outlier_frac = np.count_nonzero(outlier_mask) / gray.size

    # Extreme pixels (near 0 or 1)
    extreme_mask = (gray <= low_thresh) | (gray >= high_thresh)

    # Impulse mask: extreme pixels that are local outliers
    impulse_mask = extreme_mask & outlier_mask

    # If very few extreme-AND-outlier pixels found, also consider
    # strong outliers alone (for cases where S&P values shifted
    # slightly from extremes due to blur or noise)
    strong_outlier = _local_outlier_mask(gray, outlier_window,
                                         outlier_threshold * 2.0)
    extreme_and_strong = extreme_mask & strong_outlier
    if np.count_nonzero(impulse_mask) < np.count_nonzero(extreme_and_strong):
        impulse_mask = extreme_and_strong

    # Density = fraction of impulse pixels
    density = np.count_nonzero(impulse_mask) / gray.size

    has_impulse = density >= density_threshold

    return {
        'has_impulse': has_impulse,
        'density': density,
        'frac_low': frac_low,
        'frac_high': frac_high,
        'outlier_frac': outlier_frac,
        'impulse_mask': impulse_mask,
    }


def estimate_impulse_density(image, **kwargs):
    """
    Convenience wrapper: estimate impulse noise density.

    Returns
    -------
    density : float — estimated density in [0, 1], or 0.0 if not detected
    """
    result = detect_impulse_noise(image, **kwargs)
    return result['density'] if result['has_impulse'] else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 2. Removal — Adaptive Median Filter
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_median_filter(image, impulse_mask, max_window=7):
    """
    Apply median filter ONLY to pixels flagged as impulse noise.
    Non-impulse pixels are left completely untouched.

    Uses progressively larger windows if the median of a smaller
    window is itself an extreme value (i.e. in a dense impulse region).

    Parameters
    ----------
    image : ndarray, H×W, float64 [0, 1]
    impulse_mask : bool ndarray, H×W — True where impulse detected
    max_window : int — maximum median filter window size (odd)

    Returns
    -------
    filtered : ndarray, H×W — image with impulse pixels replaced
    """
    filtered = image.copy()
    remaining = impulse_mask.copy()

    for wsize in range(3, max_window + 1, 2):
        if not np.any(remaining):
            break
        med = _scipy_median_filter(filtered, size=wsize)
        # Replace only pixels that are still flagged
        filtered[remaining] = med[remaining]
        # Check if the replacement is still extreme — if so, try larger window
        still_extreme = (filtered <= 0.01) | (filtered >= 0.99)
        remaining = remaining & still_extreme

    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# 3. Combined pipeline
# ═════════════════════════════════════════════════════════════════════════════

def remove_impulse_noise(image, density_threshold=0.005,
                         max_window=7, outlier_window=5,
                         outlier_threshold=0.15):
    """
    Detect and remove impulse noise from an image.

    If no impulse noise is detected (density < density_threshold),
    the original image is returned unchanged.

    Parameters
    ----------
    image : ndarray, H×W or H×W×C, float64 [0, 1] or uint8 [0, 255]
    density_threshold : float — minimum density to trigger removal
    max_window : int — max AMF window
    outlier_window : int — window for outlier detection
    outlier_threshold : float — threshold for outlier detection

    Returns
    -------
    result : dict
        'image'        — ndarray: filtered image (same shape as input)
        'has_impulse'  — bool: whether impulse noise was detected
        'density'      — float: estimated density
        'applied'      — bool: whether filtering was actually applied
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    is_color = img.ndim == 3

    # Detect on grayscale
    info = detect_impulse_noise(
        img,
        density_threshold=density_threshold,
        outlier_window=outlier_window,
        outlier_threshold=outlier_threshold,
    )

    if not info['has_impulse']:
        return {
            'image': img,
            'has_impulse': False,
            'density': info['density'],
            'applied': False,
        }

    mask = info['impulse_mask']

    if is_color:
        # For color: also detect per-channel extremes
        filtered = img.copy()
        for ch in range(img.shape[2]):
            ch_mask = mask.copy()
            ch_extreme = (img[:, :, ch] <= 0.01) | (img[:, :, ch] >= 0.99)
            ch_mask = ch_mask | (_local_outlier_mask(
                img[:, :, ch], outlier_window, outlier_threshold) & ch_extreme)
            filtered[:, :, ch] = adaptive_median_filter(
                img[:, :, ch], ch_mask, max_window)
    else:
        filtered = adaptive_median_filter(img, mask, max_window)

    return {
        'image': filtered,
        'has_impulse': True,
        'density': info['density'],
        'applied': True,
    }
