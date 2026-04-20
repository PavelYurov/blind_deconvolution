# cython: language_level=3
"""
impulse_noise_estimation.pyx

Detection and removal of impulse (salt-and-pepper) noise.
"""

import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter

__all__ = [
    'detect_impulse_noise',
    'estimate_impulse_density',
    'adaptive_median_filter',
    'remove_impulse_noise',
]


def _histogram_extremes(image, low_thresh=0.01, high_thresh=0.99):
    total = image.size
    frac_low = np.count_nonzero(image <= low_thresh) / total
    frac_high = np.count_nonzero(image >= high_thresh) / total
    return frac_low, frac_high, frac_low + frac_high


def _local_outlier_mask(image, window_size=5, threshold=0.15):
    local_med = _scipy_median_filter(image, size=window_size)
    diff = np.abs(image - local_med)
    return diff > threshold


def detect_impulse_noise(image, low_thresh=0.01, high_thresh=0.99,
                         outlier_window=5, outlier_threshold=0.15,
                         density_threshold=0.0005):
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img

    frac_low, frac_high, frac_total = _histogram_extremes(
        gray, low_thresh, high_thresh)
    outlier_mask = _local_outlier_mask(
        gray, outlier_window, outlier_threshold)
    outlier_frac = np.count_nonzero(outlier_mask) / gray.size
    extreme_mask = (gray <= low_thresh) | (gray >= high_thresh)
    impulse_mask = extreme_mask & outlier_mask

    hard_extreme = (gray <= 0.005) | (gray >= 0.995)
    hard_outlier = _local_outlier_mask(gray, outlier_window, 0.02)
    impulse_mask = impulse_mask | (hard_extreme & hard_outlier)

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
    result = detect_impulse_noise(image, **kwargs)
    return result['density'] if result['has_impulse'] else 0.0


def adaptive_median_filter(image, impulse_mask, max_window=7):
    filtered = image.copy()
    remaining = impulse_mask.copy()
    for wsize in range(3, max_window + 1, 2):
        if not np.any(remaining):
            break
        med = _scipy_median_filter(filtered, size=wsize)
        filtered[remaining] = med[remaining]
        still_extreme = (filtered <= 0.01) | (filtered >= 0.99)
        remaining = remaining & still_extreme
    return filtered


def remove_impulse_noise(image, density_threshold=0.005,
                         max_window=7, outlier_window=5,
                         outlier_threshold=0.15):
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    is_color = img.ndim == 3

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
