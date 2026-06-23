"""
noise_psd_analysis.py

PSD-based noise analysis, spectral filtering, and correlation detection.

Provides tools for:
    1. Noise PSD estimation from a single image via patch-based
       spectral analysis on the smoothest patches.
    2. Periodic noise detection (spectral peaks) — the one thing
       that reliably works on 2D images.
    3. Noise correlation detection via lag-1 autocorrelation of
       noise residuals (NOT from spectral slope — β is unreliable
       on natural images due to signal contamination).
    4. Spectral filters: notch, band-stop, prewhitening.

IMPORTANT: The spectral slope β (P ∝ f^{-β}) estimated from image
patches is UNRELIABLE for 2D images.  Even the smoothest patches
retain residual signal content with ∼1/f² spectrum, biasing β upward.
β is included in the output for informational purposes only.
Automatic decisions are based on lag-1 autocorrelation instead.

Typical usage inside GBBID pipeline:
    from noise_psd_analysis import analyze_noise_psd, notch_filter

    info = analyze_noise_psd(noisy_image)
    if info['periodic_peaks']:
        cleaned = notch_filter(noisy_image, info['periodic_peaks'])

Dependencies: numpy, scipy (only).
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


# ═════════════════════════════════════════════════════════════════════════════
# 1. PSD Estimation & Classification
# ═════════════════════════════════════════════════════════════════════════════

def _radial_profile(psd_2d):
    """Compute radial (azimuthally averaged) power spectrum from 2D PSD.

    Parameters
    ----------
    psd_2d : ndarray, shape (H, W)
        Centred (fftshift-ed) 2D power spectrum.

    Returns
    -------
    radii : ndarray, shape (max_r,)
        Frequency bins (0, 1, 2, … pixels from centre).
    profile : ndarray, shape (max_r,)
        Mean PSD at each radius.
    """
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
    """Extract the smoothest (lowest-texture) patches from an image.

    Patches are sorted by gradient energy (ascending).  The smoothest
    patches contain mostly noise with minimal signal content, making
    them ideal for noise PSD estimation.

    Parameters
    ----------
    image : ndarray, H×W
    pch_size : int
    stride : int or None — defaults to pch_size // 2.
    n_patches : int — number of smoothest patches to return.

    Returns
    -------
    patches : list of ndarray, each (pch_size, pch_size)
    """
    H, W = image.shape
    if stride is None:
        stride = max(1, pch_size // 2)

    candidates = []
    for y0 in range(0, H - pch_size + 1, stride):
        for x0 in range(0, W - pch_size + 1, stride):
            p = image[y0:y0 + pch_size, x0:x0 + pch_size]
            # Gradient energy = variance of horizontal + vertical differences
            dx = np.diff(p, axis=1)
            dy = np.diff(p, axis=0)
            energy = float(np.var(dx) + np.var(dy))
            candidates.append((energy, p))

    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:n_patches]]


def _detrend_patch(patch):
    """Remove planar trend (ax + by + c) from a 2D patch.

    This kills DC and linear gradients, leaving only the noise and
    higher-order signal residuals.  Critical for unbiased noise PSD
    estimation — without detrending, gradient ramps in smooth patches
    produce a spurious 1/f² decay.
    """
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
    """Detect isolated spectral peaks in a 2D PSD (periodic noise).

    Compares each pixel against the radial average at its frequency
    (not a local median, which can be biased by directional features).
    True periodic peaks stand out far above the radial average.

    Parameters
    ----------
    psd_2d : ndarray, shape (H, W)
        Centred 2D power spectrum.
    threshold_factor : float
        Peak-to-radial-average ratio. Default 8.0.
    min_radius : int
        Minimum distance from DC to consider. Default 5.
    max_peaks : int
        Maximum number of peaks to return. Default 20.

    Returns
    -------
    peaks : list of dict
        Each: {'u': int, 'v': int, 'radius': float, 'power': float,
               'ratio': float}.
    """
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2

    # Compute radial average as baseline (robust to directional features)
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_int = R.astype(int)
    max_r = min(cy, cx)

    # Build radial average map
    radial_avg = np.zeros(max_r + 1, dtype=np.float64)
    radial_cnt = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        mask_r = R_int == r
        if mask_r.any():
            radial_avg[r] = np.median(psd_2d[mask_r])  # median per ring
            radial_cnt[r] = mask_r.sum()

    # Build baseline map from radial averages
    baseline = np.ones_like(psd_2d) * np.median(psd_2d)
    for r in range(max_r + 1):
        mask_r = R_int == r
        baseline[mask_r] = max(radial_avg[r], 1e-30)

    ratio_map = psd_2d / baseline

    # Find peaks: high ratio, not near DC, not at edges
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

    # Cluster nearby peaks: for each connected region, keep the brightest
    peaks = []
    coords = np.argwhere(peak_mask)
    if len(coords) == 0:
        return peaks

    # Simple greedy clustering: sort by power, skip neighbours
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
        # Mark neighbours as used (radius 5 to avoid duplicates)
        for j in range(i + 1, len(coords_sorted)):
            vj, uj = coords_sorted[j]
            if abs(vj - v) <= 5 and abs(uj - u) <= 5:
                used[j] = True

    return peaks[:max_peaks]


def estimate_noise_psd(image, pch_size=32, n_smooth=100):
    """Estimate the 2D noise PSD from the smoothest image patches.

    Selects the smoothest (lowest gradient energy) patches, subtracts
    their mean (removes DC/signal), computes average periodogram.
    This avoids signal contamination that plagues full-image PSD methods.

    Parameters
    ----------
    image : ndarray, H×W, float64 [0, 1] or [0, 255].
    pch_size : int
        Patch size for spectral analysis. Default 32.
    n_smooth : int
        Number of smoothest patches to average. Default 100.

    Returns
    -------
    psd_2d : ndarray, shape (H, W)
        Estimated noise PSD (centred, fftshift convention),
        scaled back to full image dimensions.
    radial_freq : ndarray
        Normalised radial frequencies [0, 1] (patch-resolution).
    radial_psd : ndarray
        Radially averaged noise PSD from patches.
    psd_2d_patches : ndarray, shape (pch_size, pch_size)
        Raw patch-level PSD before upscaling.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    H, W = img.shape[:2]
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Extract smoothest patches
    patches = _extract_smooth_patches(img, pch_size=pch_size, n_patches=n_smooth)
    if len(patches) < 5:
        # Fallback: estimate from full image high-freq ring
        F = fftshift(fft2(img))
        psd_full = np.abs(F) ** 2 / (H * W)
        radii, profile = _radial_profile(psd_full)
        max_r = min(H // 2, W // 2)
        return psd_full, radii / max_r, profile, psd_full

    # Average periodogram from smooth + detrended patches (Bartlett's method)
    # Apply Hanning window to reduce spectral leakage from patch edges.
    window = np.outer(np.hanning(pch_size), np.hanning(pch_size))
    window_energy = np.sum(window ** 2)
    psd_avg = np.zeros((pch_size, pch_size), dtype=np.float64)
    for p in patches:
        p_dt = _detrend_patch(p) * window
        F = fftshift(fft2(p_dt))
        psd_avg += np.abs(F) ** 2
    psd_avg /= len(patches)
    psd_avg /= window_energy  # correct for window energy loss

    # Radial profile of patch-level noise PSD
    radii, radial_psd = _radial_profile(psd_avg)
    max_r_patch = pch_size // 2
    norm_freq = radii / max(max_r_patch, 1)

    # Upscale patch-level PSD to full image dimensions for filtering.
    # Interpolate the radial noise profile to full resolution.
    cy, cx = H // 2, W // 2
    max_r_full = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    R_full = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Linear interpolation of radial profile to full-image radii
    scale = max_r_patch / max(max_r_full, 1)
    R_patch = R_full * scale  # patch-space radii
    R_patch = np.clip(R_patch, 0, len(radial_psd) - 1)
    r_floor = np.floor(R_patch).astype(int)
    r_ceil = np.minimum(r_floor + 1, len(radial_psd) - 1)
    frac = R_patch - r_floor
    psd_2d_full = radial_psd[r_floor] * (1 - frac) + radial_psd[r_ceil] * frac

    return psd_2d_full, norm_freq, radial_psd, psd_avg


def _lag1_autocorrelation(image, pch_size=32, n_patches=100):
    """Test noise correlation via lag-1 autocorrelation of patch residuals.

    This is the ONLY reliable method for detecting correlated noise
    in 2D natural images.  The PSD spectral slope β is unreliable
    because even "smooth" patches retain signal content with ∼1/f².

    Method:
    1. Extract smooth patches (same as for PSD estimation).
    2. Detrend each patch (remove planar gradient).
    3. Compute normalised lag-1 autocorrelation horizontally and
       vertically.  For white noise, lag-1 ≈ 0.  For correlated
       noise, lag-1 is significantly positive.

    Returns
    -------
    lag1_h : float — mean horizontal lag-1 autocorrelation.
    lag1_v : float — mean vertical lag-1 autocorrelation.
    is_correlated : bool — True if lag-1 significantly above noise floor.
    """
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
        # Horizontal lag-1: correlation between p[y, x] and p[y, x+1]
        h = np.mean(p_dt[:, :-1] * p_dt[:, 1:]) / var
        # Vertical lag-1: correlation between p[y, x] and p[y+1, x]
        v = np.mean(p_dt[:-1, :] * p_dt[1:, :]) / var
        lag1_h_list.append(h)
        lag1_v_list.append(v)

    if not lag1_h_list:
        return 0.0, 0.0, False

    lag1_h = float(np.median(lag1_h_list))
    lag1_v = float(np.median(lag1_v_list))

    # Statistical significance: for N independent white-noise samples
    # of size pch_size², the expected std of lag-1 is ∼ 1/pch_size.
    # We use a conservative threshold of 3σ.
    expected_std = 1.0 / pch_size
    threshold = 3.0 * expected_std
    is_correlated = (lag1_h > threshold) or (lag1_v > threshold)

    return lag1_h, lag1_v, is_correlated


def classify_noise(radial_freq, radial_psd, psd_2d_full=None,
                   peak_threshold=100.0, image=None):
    """Classify noise type from periodic peak detection and autocorrelation.

    NOTE: The spectral slope β (P ∝ f^{-β}) estimated from image patches
    is UNRELIABLE for 2D natural images.  It is included for informational
    purposes only and is NOT used for the is_correlated decision.
    Correlation is determined by lag-1 autocorrelation if `image` is given.

    Parameters
    ----------
    radial_freq : ndarray
        Normalised radial frequencies [0, 1].
    radial_psd : ndarray
        Radially averaged noise PSD (from patch analysis).
    psd_2d_full : ndarray or None
        Full-image 2D PSD for periodic peak detection.
    peak_threshold : float
        Threshold for periodic peak detection.  Default 100.0.
        Real periodic peaks have ratio > 2000; false positives < 50.
    image : ndarray or None
        Original image for lag-1 autocorrelation test.
        If None, is_correlated defaults to False.

    Returns
    -------
    result : dict
        'noise_class'      — 'white' | 'periodic' | 'correlated'
        'beta'             — spectral slope (INFORMATIONAL ONLY)
        'is_correlated'    — bool (from lag-1 autocorrelation, not β)
        'has_periodic'     — bool (True if spectral peaks found)
        'periodic_peaks'   — list of peak dicts
        'noise_floor'      — estimated noise level
        'lag1_h'           — horizontal lag-1 autocorrelation
        'lag1_v'           — vertical lag-1 autocorrelation
    """
    # ── β estimation (informational only) ────────────────────────────
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

    # ── Periodic peak detection (reliable on 2D images) ─────────────
    peaks = []
    if psd_2d_full is not None:
        peaks = _detect_periodic_peaks_2d(psd_2d_full,
                                          threshold_factor=peak_threshold)
    has_periodic = len(peaks) > 0

    # ── Lag-1 autocorrelation (reliable correlation test) ───────────
    lag1_h, lag1_v, is_corr = 0.0, 0.0, False
    if image is not None:
        lag1_h, lag1_v, is_corr = _lag1_autocorrelation(image)

    # ── Classification ──────────────────────────────────────────────
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
    """Full PSD noise analysis: estimate + classify.

    Combines estimate_noise_psd(), lag-1 autocorrelation test, and
    classify_noise() into one call.

    Parameters
    ----------
    image : ndarray, H×W grayscale (float or uint8).
    pch_size : int — patch size for noise PSD estimation.
    n_smooth : int — number of smooth patches to average.
    peak_threshold : float — periodic peak detection threshold.
        Real peaks have ratio > 2000; set ≥ 100 to avoid false positives.

    Returns
    -------
    info : dict
        'psd_2d'          — 2D noise PSD (centred, H×W)
        'psd_2d_patches'  — patch-level noise PSD (pch_size × pch_size)
        'radial_freq'     — normalised radial frequencies
        'radial_psd'      — radially averaged noise PSD
        'noise_class'     — 'white' | 'periodic' | 'correlated'
        'beta'            — spectral slope (INFORMATIONAL ONLY)
        'is_correlated'   — bool (from lag-1 autocorrelation)
        'has_periodic'    — bool (spectral peaks found)
        'periodic_peaks'  — list of peak dicts (2D)
        'noise_floor'     — median noise PSD level
        'lag1_h'          — horizontal lag-1 autocorrelation
        'lag1_v'          — vertical lag-1 autocorrelation
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    psd_2d, radial_freq, radial_psd, psd_patches = estimate_noise_psd(
        img, pch_size=pch_size, n_smooth=n_smooth)

    # Full-image 2D PSD for periodic peak detection
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
# 2. Notch Filter
# ═════════════════════════════════════════════════════════════════════════════

def notch_filter(image, peaks, notch_radius=3, rolloff=2):
    """Remove periodic noise by notching out spectral peaks.

    For each detected peak, creates a point-symmetric Butterworth
    notch at (u, v) and its conjugate (-u, -v) in the 2D frequency
    domain.  If peaks only have 'radius' (no u/v), falls back to
    annular (ring) suppression.

    Parameters
    ----------
    image : ndarray, H×W, float [0,1] or [0,255].
    peaks : list of dict
        From analyze_noise_psd()['periodic_peaks'].
        Each should have 'u', 'v' (2D coords) or at least 'radius'.
    notch_radius : int
        Half-width of the notch around each peak. Default 3.
    rolloff : int
        Butterworth order for smooth transition. Default 2.

    Returns
    -------
    filtered : ndarray, same shape, float64.
    """
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

    # Build the notch mask: 1.0 everywhere, 0.0 at peaks
    mask = np.ones((H, W), dtype=np.float64)
    for pk in peaks:
        if 'u' in pk and 'v' in pk:
            # 2D peak: notch at (u, v) and its symmetric conjugate
            u0, v0 = pk['u'], pk['v']
            for (pu, pv) in [(u0, v0), (2 * cx - u0, 2 * cy - v0)]:
                D = np.sqrt((X - pu) ** 2 + (Y - pv) ** 2)
                D = np.maximum(D, 1e-10)
                notch = 1.0 - 1.0 / (1.0 + (D / max(notch_radius, 1)) ** (2 * rolloff))
                mask *= notch
        else:
            # Radial peak: annular suppression at radius r
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
# 3. Band-Stop Filter
# ═════════════════════════════════════════════════════════════════════════════

def bandstop_filter(image, freq_low, freq_high, order=2):
    """Suppress a band of radial frequencies.

    Attenuates all frequencies between `freq_low` and `freq_high`
    (normalised, 0 = DC, 1 = Nyquist) using a Butterworth band-reject.

    Parameters
    ----------
    image : ndarray, H×W, float [0,1] or [0,255].
    freq_low : float
        Lower normalised frequency of the stop band (0 to 1).
    freq_high : float
        Upper normalised frequency of the stop band (0 to 1).
    order : int
        Butterworth order. Default 2.

    Returns
    -------
    filtered : ndarray, same shape, float64.
    """
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
    R_norm = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_r  # [0, ~1.4]

    # Band centre and width (in normalised frequency)
    f_centre = (freq_low + freq_high) / 2.0
    f_width = (freq_high - freq_low) / 2.0

    # Butterworth band-reject:
    #   H(f) = 1 - 1 / (1 + ((f - f_c) / f_w)^(2n))
    # which is 0 at f = f_c and 1 far away.
    D = np.abs(R_norm - f_centre)
    D = np.maximum(D, 1e-10)
    mask = 1.0 - 1.0 / (1.0 + (D / max(f_width, 1e-6)) ** (2 * order))

    # Preserve DC
    mask[cy, cx] = 1.0

    F = fftshift(fft2(img))
    F_filtered = F * mask
    filtered = np.real(ifft2(ifftshift(F_filtered)))

    return filtered * 255.0 if was_255 else filtered


# ═════════════════════════════════════════════════════════════════════════════
# 4. Prewhitening Filter
# ═════════════════════════════════════════════════════════════════════════════

def prewhiten(image, psd_2d, reg=1e-3):
    """Prewhiten an image by dividing its spectrum by the noise PSD.

    WARNING: This function requires a PURE NOISE PSD (not signal+noise).
    The PSD from estimate_noise_psd() is estimated from image patches
    and contains signal contamination.  Using it will DESTROY the image
    (suppress low-frequency signal content, amplify high-frequency noise).

    This function is kept for manual/experimental use only.  It is NOT
    called by the auto pipeline (noise_preprocess).

    The filter is:  W(f) = 1 / sqrt(P_n(f) + reg)

    Parameters
    ----------
    image : ndarray, H×W, float [0,1] or [0,255].
    psd_2d : ndarray, H×W
        Centred (fftshift) 2D NOISE-ONLY PSD.  If this contains signal
        energy (as from estimate_noise_psd), the result will be wrong.
    reg : float
        Regularisation constant (Tikhonov). Default 1e-3.

    Returns
    -------
    whitened : ndarray, same shape, float64 [0,1] or [0,255].
    """
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

    # Resize PSD to match image if needed
    if psd.shape != (H, W):
        from scipy.ndimage import zoom
        psd = zoom(psd, (H / psd.shape[0], W / psd.shape[1]), order=1)

    # Whitening filter: W(f) = 1 / sqrt(P_n(f) + reg)
    # Ensures that after filtering, noise becomes flat-spectrum.
    W_filter = 1.0 / np.sqrt(psd + reg)

    # Normalise so that the filter doesn't change overall energy too much.
    # Scale so median of W_filter is 1.0.
    med = np.median(W_filter)
    if med > 0:
        W_filter = W_filter / med

    F = fftshift(fft2(img))
    F_whitened = F * W_filter
    whitened = np.real(ifft2(ifftshift(F_whitened)))

    # Clip to valid range
    whitened = np.clip(whitened, 0.0, 1.0)

    return whitened * 255.0 if was_255 else whitened


# ═════════════════════════════════════════════════════════════════════════════
# 5. Convenience: full noise preprocessing pipeline
# ═════════════════════════════════════════════════════════════════════════════

def noise_preprocess(image, pch_size=32, n_smooth=100,
                     peak_threshold=100.0,
                     notch_radius=3):
    """Automatic noise preprocessing: analyze → notch filter.

    In auto mode, ONLY applies notch filter for periodic noise.
    Prewhitening is NOT applied automatically because it requires
    a pure noise PSD (which cannot be reliably separated from the
    signal PSD on a single 2D image).

    Steps:
    1. Analyze noise PSD + lag-1 autocorrelation.
    2. If periodic peaks detected → notch filter.
    3. Report is_correlated flag (informational only).

    Parameters
    ----------
    image : ndarray, H×W or H×W×C, float [0,1] or [0,255].
    pch_size : int — patch size for PSD estimation.
    n_smooth : int — number of smooth patches.
    peak_threshold : float — periodic peak detection threshold.
        Default 100.0.  Real periodic peaks have ratio > 2000.
    notch_radius : int — notch half-width for periodic peaks.

    Returns
    -------
    result : dict
        'image'       — preprocessed image (same shape/scale as input)
        'psd_info'    — full analysis dict from analyze_noise_psd()
        'applied'     — list of str: which operations were applied
                        ('notch' or empty)
    """
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0

    # Work in [0, 1]
    if was_255:
        work = img / 255.0
    else:
        work = img.copy()

    # Grayscale for analysis (even if input is colour)
    if work.ndim == 3:
        gray = 0.2989 * work[:, :, 0] + 0.587 * work[:, :, 1] + 0.114 * work[:, :, 2]
    else:
        gray = work

    psd_info = analyze_noise_psd(gray, pch_size=pch_size,
                                 n_smooth=n_smooth,
                                 peak_threshold=peak_threshold)

    applied = []
    processed = work.copy()

    # Only notch filter for periodic peaks.
    # Prewhitening is NOT applied automatically — it requires a pure
    # noise PSD which cannot be separated from signal on a single image.
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
