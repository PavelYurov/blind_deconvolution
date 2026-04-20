# cython: language_level=3
"""
noise_psd_analysis.pyx

PSD-based noise analysis, spectral filtering, and correlation detection.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import median_filter as _median_filter_nd

__all__ = [
    'analyze_noise_psd',
    'estimate_noise_psd',
    'classify_noise',
    'prewhiten',
    'notch_filter',
    'bandstop_filter',
]


def _radial_profile(psd_2d):
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    radii = np.arange(0, max_r)
    profile = np.zeros(max_r, dtype=np.float64)
    for r in radii:
        mask = R == r
        if mask.any():
            profile[r] = psd_2d[mask].mean()
    return radii, profile


def _extract_smooth_patches(image, pch_size=32, stride=None, n_patches=100):
    H, W = image.shape
    if stride is None:
        stride = max(1, pch_size // 2)
    candidates = []
    for y0 in range(0, H - pch_size + 1, stride):
        for x0 in range(0, W - pch_size + 1, stride):
            p = image[y0:y0 + pch_size, x0:x0 + pch_size]
            dx = np.diff(p, axis=1)
            dy = np.diff(p, axis=0)
            energy = float(np.var(dx) + np.var(dy))
            candidates.append((energy, p))
    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:n_patches]]


def _detrend_patch(patch):
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx.ravel().astype(np.float64),
                         yy.ravel().astype(np.float64),
                         np.ones(h * w, dtype=np.float64)])
    coef = np.linalg.lstsq(A, patch.ravel(), rcond=None)[0]
    trend = (A @ coef).reshape(h, w)
    return patch - trend


def _detect_periodic_peaks_2d(psd_2d, threshold_factor=8.0, min_radius=5,
                              max_peaks=20):
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_int = R.astype(int)
    max_r = min(cy, cx)

    radial_avg = np.zeros(max_r + 1, dtype=np.float64)
    radial_cnt = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        mask_r = R_int == r
        if mask_r.any():
            radial_avg[r] = np.median(psd_2d[mask_r])
            radial_cnt[r] = mask_r.sum()

    baseline = np.ones_like(psd_2d) * np.median(psd_2d)
    for r in range(max_r + 1):
        mask_r = R_int == r
        baseline[mask_r] = max(radial_avg[r], 1e-30)

    ratio_map = psd_2d / baseline

    border = 2
    peak_mask = (
        (ratio_map > threshold_factor) &
        (R > min_radius) &
        (R < min(cy, cx) * 0.95)
    )
    peak_mask[:border, :] = False
    peak_mask[-border:, :] = False
    peak_mask[:, :border] = False
    peak_mask[:, -border:] = False

    peaks = []
    coords = np.argwhere(peak_mask)
    if len(coords) == 0:
        return peaks

    powers = psd_2d[peak_mask]
    order = np.argsort(-powers)
    coords_sorted = coords[order]
    used = np.zeros(len(coords_sorted), dtype=bool)

    for i in range(len(coords_sorted)):
        if used[i]:
            continue
        v, u = coords_sorted[i]
        r = float(np.sqrt((u - cx) ** 2 + (v - cy) ** 2))
        peaks.append({
            'u': int(u),
            'v': int(v),
            'radius': r,
            'power': float(psd_2d[v, u]),
            'ratio': float(ratio_map[v, u]),
        })
        for j in range(i + 1, len(coords_sorted)):
            vj, uj = coords_sorted[j]
            if abs(vj - v) <= 5 and abs(uj - u) <= 5:
                used[j] = True

    return peaks[:max_peaks]


def estimate_noise_psd(image, pch_size=32, n_smooth=100):
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    H, W = img.shape[:2]
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    patches = _extract_smooth_patches(img, pch_size=pch_size, n_patches=n_smooth)
    if len(patches) < 5:
        F = fftshift(fft2(img))
        psd_full = np.abs(F) ** 2 / (H * W)
        radii, profile = _radial_profile(psd_full)
        max_r = min(H // 2, W // 2)
        return psd_full, radii / max_r, profile, psd_full

    window = np.outer(np.hanning(pch_size), np.hanning(pch_size))
    window_energy = np.sum(window ** 2)
    psd_avg = np.zeros((pch_size, pch_size), dtype=np.float64)
    for p in patches:
        p_dt = _detrend_patch(p) * window
        F = fftshift(fft2(p_dt))
        psd_avg += np.abs(F) ** 2
    psd_avg /= len(patches)
    psd_avg /= window_energy

    radii, radial_psd = _radial_profile(psd_avg)
    max_r_patch = pch_size // 2
    norm_freq = radii / max(max_r_patch, 1)

    cy, cx = H // 2, W // 2
    max_r_full = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    R_full = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    scale = max_r_patch / max(max_r_full, 1)
    R_patch = R_full * scale
    R_patch = np.clip(R_patch, 0, len(radial_psd) - 1)
    r_floor = np.floor(R_patch).astype(int)
    r_ceil = np.minimum(r_floor + 1, len(radial_psd) - 1)
    frac = R_patch - r_floor
    psd_2d_full = radial_psd[r_floor] * (1 - frac) + radial_psd[r_ceil] * frac

    return psd_2d_full, norm_freq, radial_psd, psd_avg


def _lag1_autocorrelation(image, pch_size=32, n_patches=100):
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    patches = _extract_smooth_patches(img, pch_size=pch_size,
                                      n_patches=n_patches)
    if len(patches) < 5:
        return 0.0, 0.0, False

    lag1_h_list = []
    lag1_v_list = []
    for p in patches:
        p_dt = _detrend_patch(p)
        var = np.var(p_dt)
        if var < 1e-15:
            continue
        h = np.mean(p_dt[:, :-1] * p_dt[:, 1:]) / var
        v = np.mean(p_dt[:-1, :] * p_dt[1:, :]) / var
        lag1_h_list.append(h)
        lag1_v_list.append(v)

    if not lag1_h_list:
        return 0.0, 0.0, False

    lag1_h = float(np.median(lag1_h_list))
    lag1_v = float(np.median(lag1_v_list))

    expected_std = 1.0 / pch_size
    threshold = 3.0 * expected_std
    is_correlated = (lag1_h > threshold) or (lag1_v > threshold)

    return lag1_h, lag1_v, is_correlated


def classify_noise(radial_freq, radial_psd, psd_2d_full=None,
                   peak_threshold=100.0, image=None):
    valid = (radial_freq > 0.35) & (radial_freq < 0.9)
    if valid.sum() < 3:
        beta = 0.0
    else:
        f = radial_freq[valid]
        p = np.maximum(radial_psd[valid], 1e-30)
        A = np.vstack([np.log(f), np.ones_like(f)]).T
        coeff = np.linalg.lstsq(A, np.log(p), rcond=None)[0]
        beta = float(-coeff[0])

    noise_floor = float(np.median(radial_psd[valid])) if valid.sum() >= 3 \
        else float(np.median(radial_psd))

    peaks = []
    if psd_2d_full is not None:
        peaks = _detect_periodic_peaks_2d(psd_2d_full,
                                          threshold_factor=peak_threshold)
    has_periodic = len(peaks) > 0

    lag1_h, lag1_v, is_corr = 0.0, 0.0, False
    if image is not None:
        lag1_h, lag1_v, is_corr = _lag1_autocorrelation(image)

    if has_periodic:
        noise_class = 'periodic'
    elif is_corr:
        noise_class = 'correlated'
    else:
        noise_class = 'white'

    return {
        'noise_class': noise_class,
        'beta': beta,
        'is_correlated': is_corr,
        'has_periodic': has_periodic,
        'periodic_peaks': peaks,
        'noise_floor': noise_floor,
        'lag1_h': lag1_h,
        'lag1_v': lag1_v,
    }


def analyze_noise_psd(image, pch_size=32, n_smooth=100,
                      peak_threshold=100.0):
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    psd_2d, radial_freq, radial_psd, psd_patches = estimate_noise_psd(
        img, pch_size=pch_size, n_smooth=n_smooth)

    F = fftshift(fft2(img))
    psd_full_2d = np.abs(F) ** 2 / (img.shape[0] * img.shape[1])

    classification = classify_noise(
        radial_freq, radial_psd,
        psd_2d_full=psd_full_2d,
        peak_threshold=peak_threshold,
        image=img)

    return {
        'psd_2d': psd_2d,
        'psd_2d_patches': psd_patches,
        'radial_freq': radial_freq,
        'radial_psd': radial_psd,
        **classification,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Notch Filter
# ═════════════════════════════════════════════════════════════════════════════

def notch_filter(image, peaks, notch_radius=3, rolloff=2):
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = notch_filter(img[:, :, ch], peaks,
                                         notch_radius, rolloff)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    cy, cx = H // 2, W // 2
    Y, X = np.mgrid[:H, :W]

    mask = np.ones((H, W), dtype=np.float64)
    for pk in peaks:
        if 'u' in pk and 'v' in pk:
            u0, v0 = pk['u'], pk['v']
            for (pu, pv) in [(u0, v0), (2 * cx - u0, 2 * cy - v0)]:
                D = np.sqrt((X - pu) ** 2 + (Y - pv) ** 2)
                D = np.maximum(D, 1e-10)
                notch = 1.0 - 1.0 / (1.0 + (D / max(notch_radius, 1)) ** (2 * rolloff))
                mask *= notch
        else:
            r0 = pk['radius']
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            D = np.abs(R - r0)
            D = np.maximum(D, 1e-10)
            notch = 1.0 - 1.0 / (1.0 + (D / max(notch_radius, 1)) ** (2 * rolloff))
            mask *= notch
    F = fftshift(fft2(img))
    F_filtered = F * mask
    filtered = np.real(ifft2(ifftshift(F_filtered)))
    return filtered * 255.0 if was_255 else filtered


# ═════════════════════════════════════════════════════════════════════════════
# Band-Stop Filter
# ═════════════════════════════════════════════════════════════════════════════

def bandstop_filter(image, freq_low, freq_high, order=2):
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = bandstop_filter(img[:, :, ch],
                                            freq_low, freq_high, order)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    R_norm = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_r

    f_centre = (freq_low + freq_high) / 2.0
    f_width = (freq_high - freq_low) / 2.0

    D = np.abs(R_norm - f_centre)
    D = np.maximum(D, 1e-10)
    mask = 1.0 - 1.0 / (1.0 + (D / max(f_width, 1e-6)) ** (2 * order))
    mask[cy, cx] = 1.0

    F = fftshift(fft2(img))
    F_filtered = F * mask
    filtered = np.real(ifft2(ifftshift(F_filtered)))
    return filtered * 255.0 if was_255 else filtered


# ═════════════════════════════════════════════════════════════════════════════
# Prewhitening Filter
# ═════════════════════════════════════════════════════════════════════════════

def prewhiten(image, psd_2d, reg=1e-3):
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = prewhiten(img[:, :, ch], psd_2d, reg)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    psd = np.asarray(psd_2d, dtype=np.float64)

    if psd.shape != (H, W):
        from scipy.ndimage import zoom
        psd = zoom(psd, (H / psd.shape[0], W / psd.shape[1]), order=1)

    W_filter = 1.0 / np.sqrt(psd + reg)
    med = np.median(W_filter)
    if med > 0:
        W_filter = W_filter / med

    F = fftshift(fft2(img))
    F_whitened = F * W_filter
    whitened = np.real(ifft2(ifftshift(F_whitened)))
    whitened = np.clip(whitened, 0.0, 1.0)
    return whitened * 255.0 if was_255 else whitened


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: full noise preprocessing pipeline
# ═════════════════════════════════════════════════════════════════════════════

def noise_preprocess(image, pch_size=32, n_smooth=100,
                     peak_threshold=100.0,
                     notch_radius=3):
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        work = img / 255.0
    else:
        work = img.copy()

    if work.ndim == 3:
        gray = 0.2989 * work[:, :, 0] + 0.587 * work[:, :, 1] + 0.114 * work[:, :, 2]
    else:
        gray = work

    psd_info = analyze_noise_psd(gray, pch_size=pch_size,
                                 n_smooth=n_smooth,
                                 peak_threshold=peak_threshold)
    applied = []
    processed = work.copy()

    if psd_info['has_periodic']:
        processed = notch_filter(processed, psd_info['periodic_peaks'],
                                 notch_radius=notch_radius)
        applied.append('notch')

    if was_255:
        processed = processed * 255.0

    return {
        'image': processed,
        'psd_info': psd_info,
        'applied': applied,
    }
