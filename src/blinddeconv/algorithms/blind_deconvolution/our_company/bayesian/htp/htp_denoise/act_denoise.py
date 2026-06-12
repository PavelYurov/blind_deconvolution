"""
act_denoise.py

Adaptive Curvelet Thresholding (ACT) for white and colored Gaussian noise.

Based on:
    N. Eslahi, A. Aghagolzadeh:
    "Compressive Sensing Image Restoration Using Adaptive Curvelet
     Thresholding and Nonlocal Sparse Regularization",
    IEEE Trans. Image Process., vol. 25, no. 7, pp. 3126-3140, Jul. 2016.
    https://doi.org/10.1109/TIP.2016.2562563

Ported from the MATLAB implementation (v1.00, May 2022)
by Nasser Eslahi (Tampere University).

Requires: curvelets  (pure-Python UDCT implementation, pip install curvelets).

Dependencies: numpy, scipy, curvelets.
"""

import numpy as np
from scipy.ndimage import convolve as _ndconvolve
from numpy.fft import ifft2, fftshift

__all__ = ['act_denoise']


# ═════════════════════════════════════════════════════════════════════════════
# 1. UDCT operator factory
# ═════════════════════════════════════════════════════════════════════════════

def _choose_num_scales(H, W):
    """Pick the number of UDCT scales (including lowpass) for an image.

    Heuristic: ceil(log2(min(H,W))) - 2, clamped to [2, 4].
    """
    return max(2, min(4, int(np.ceil(np.log2(min(H, W)))) - 2))


def _udct_pad_multiple(num_scales):
    """Smallest factor each spatial dimension must divide for UDCT to
    reconstruct exactly.

    ``curvelets.numpy.UDCT`` silently produces large reconstruction error
    when a dimension is not divisible by ``2**(num_scales-1)``.  Padding
    each dim to a multiple of this value restores perfect reconstruction
    and eliminates banding artifacts on non-square / odd-sized images.
    """
    return 1 << max(num_scales - 1, 0)


def _make_udct(H, W, num_scales=None):
    """Create a UDCT operator. (H, W) must already be padded to multiples
    of ``_udct_pad_multiple(num_scales)`` -- otherwise the underlying UDCT
    fails silently with bad reconstruction. See :func:`act_denoise`.
    """
    from curvelets.numpy import UDCT

    if num_scales is None:
        num_scales = _choose_num_scales(H, W)

    return UDCT(shape=(H, W), num_scales=num_scales, transform_kind='real')


# ═════════════════════════════════════════════════════════════════════════════
# 2. Noise PSD propagation to curvelet domain
# ═════════════════════════════════════════════════════════════════════════════

def _compute_curvelet_noise_rootpsd(fft_psd, udct_op):
    """Compute noise root-PSD (σ) per curvelet subband.

    Mirrors MATLAB ``cmpt_DCuT_rootPSD``.

    The noise coloring kernel in the spatial domain is recovered from
    √(FFT_PSD) via inverse FFT, then transformed into the curvelet
    domain.  The RMS of each subband gives the noise std there.

    Parameters
    ----------
    fft_psd : ndarray (H, W)
        Noise FFT-PSD in standard FFT order (DC at [0,0]).
        Convention: for AWGN with variance σ², FFT_PSD = σ² × H × W.
    udct_op : curvelets.numpy.UDCT

    Returns
    -------
    rootpsd : list[list[list[float]]]
        rootpsd[J][D][W] = noise std in subband
        (scale J, direction D, wedge W).
    """
    # Spatial noise-coloring kernel: ifft2(√PSD), centred.
    # For a properly formed FFT_PSD (Hermitian symmetric for real kernel),
    # the result is real-valued.  Take .real for numerical safety.
    kernel_noise = fftshift(ifft2(np.sqrt(fft_psd.astype(np.complex128)))).real

    c_struct = udct_op.forward(kernel_noise)

    rootpsd = []
    for scale in c_struct:
        dirs = []
        for direction in scale:
            wedges = []
            for wedge in direction:
                rms = float(np.sqrt(np.mean(np.abs(wedge) ** 2)))
                wedges.append(rms)
            dirs.append(wedges)
        rootpsd.append(dirs)
    return rootpsd


# ═════════════════════════════════════════════════════════════════════════════
# 3. ML estimator for clean signal std
# ═════════════════════════════════════════════════════════════════════════════

def _ml_estimator(noisy_coeffs, noise_rootpsd, noise_type):
    """Estimate clean signal std in one curvelet subband via ML.

    Mirrors MATLAB ``ML_estimator``.

    Local variance of noisy coefficients is estimated by averaging
    |c|² over a neighbourhood (excluding the centre pixel).
    Clean variance = local noisy variance − noise variance.
    Clean std = √(max(clean_var, 0)).

    Parameters
    ----------
    noisy_coeffs : ndarray (complex)
        Curvelet coefficients at subband (J, L).
    noise_rootpsd : float
        Noise std (root-PSD) in this subband.
    noise_type : str
        'white' or 'colored'.  Controls averaging window size:
        7×7 for white, 31×31 for colored (matches MATLAB).

    Returns
    -------
    clean_std : ndarray (real, same spatial shape as noisy_coeffs)
        Spatially varying estimate of clean signal std.
    """
    if noise_type == 'white':
        # 7×7 window, exclude centre → 48 neighbours
        k = np.ones((7, 7), dtype=np.float64) / 48.0
        k[3, 3] = 0.0
    else:
        # 31×31 window, exclude centre → 960 neighbours
        k = np.ones((31, 31), dtype=np.float64) / 960.0
        k[15, 15] = 0.0

    # Local average of |c|² (circular boundary, same as MATLAB padarray+convn)
    power = (np.abs(noisy_coeffs) ** 2).astype(np.float64)
    local_var = _ndconvolve(power, k, mode='wrap')

    # Clean variance = noisy variance − noise variance
    clean_var = local_var - noise_rootpsd ** 2
    clean_var = np.maximum(clean_var, 0.0)

    return np.sqrt(clean_var)


# ═════════════════════════════════════════════════════════════════════════════
# 4. ACT: Adaptive Curvelet Thresholding
# ═════════════════════════════════════════════════════════════════════════════

def _apply_act(c_struct, rootpsd, threshold_setting, noise_type):
    """Apply ACT thresholding across all curvelet subbands.

    Mirrors MATLAB ``ACT`` subfunction.
    Coarsest scale (J=0, MATLAB J=1) is always skipped.

    Parameters
    ----------
    c_struct : list[list[list[ndarray]]]
        Curvelet coefficients from UDCT.forward().
        Structure: c_struct[scale][direction][wedge].
    rootpsd : list[list[list[float]]]
        Noise root-PSD per subband from _compute_curvelet_noise_rootpsd.
    threshold_setting : str
        's' — soft ACT, 'h' — hard ACT, 'ksigma' — k-sigma baseline.
    noise_type : str
        'white' or 'colored'.

    Returns
    -------
    denoised : list[list[list[ndarray]]]
        Thresholded curvelet coefficients (same structure).
    """
    nscales = len(c_struct)
    denoised = []

    for J in range(nscales):
        dirs = []
        for D in range(len(c_struct[J])):
            wedges = []
            for W in range(len(c_struct[J][D])):
                coeff = c_struct[J][D][W].copy()

                # Skip coarsest scale (J=0 in Python = J=1 in MATLAB)
                if J == 0:
                    wedges.append(coeff)
                    continue

                sigma_n = rootpsd[J][D][W]
                mag = np.abs(coeff)

                if threshold_setting in ('s', 'h'):
                    # ── ACT (Eslahi & Aghagolzadeh, 2016) ───────────────
                    clean_std = _ml_estimator(coeff, sigma_n, noise_type)
                    safe_std = np.maximum(clean_std, 1e-10)

                    if threshold_setting == 's':
                        # Soft adaptive threshold: T = √2 · σ_n² / σ_clean
                        threshold = np.sqrt(2.0) * (sigma_n ** 2) / safe_std
                        threshold = np.where(clean_std > 0, threshold, np.inf)

                        # Complex soft thresholding:
                        # out = (c / |c|) × max(|c| − T, 0)
                        shrunk = np.maximum(mag - threshold, 0.0)
                        coeff = np.where(
                            mag > 1e-30,
                            coeff * (shrunk / mag),
                            np.zeros_like(coeff),
                        )

                    else:  # 'h' — hard adaptive threshold
                        # Additional factor (3+δ)/√2 where δ=1 for finest scale
                        is_finest = float(J == nscales - 1)
                        threshold = ((3.0 + is_finest) * (sigma_n ** 2)
                                     / (np.sqrt(2.0) * safe_std))
                        threshold = np.where(clean_std > 0, threshold, np.inf)

                        coeff = coeff * (mag > threshold)

                else:  # 'ksigma' — Starck, Candes, Donoho (2002)
                    is_finest = float(J == nscales - 1)
                    threshold = (3.0 + is_finest) * sigma_n

                    coeff = coeff * (mag > threshold)

                wedges.append(coeff)
            dirs.append(wedges)
        denoised.append(dirs)

    return denoised


# ═════════════════════════════════════════════════════════════════════════════
# 5. Public interface
# ═════════════════════════════════════════════════════════════════════════════

def act_denoise(image, noise_var=None, threshold_setting='s'):
    """Denoise a grayscale image using Adaptive Curvelet Thresholding.

    Parameters
    ----------
    image : ndarray (H, W), float64 [0, 1]
        Noisy grayscale image.
    noise_var : None, float, or ndarray (H, W)
        - ``None``  — blind estimation via MAD on finest curvelet subband.
        - ``float`` — known AWGN variance σ².
        - ``ndarray (H, W)`` — noise FFT-PSD in standard FFT order
          (DC at [0,0]).  Convention: for AWGN σ², FFT_PSD = σ² × H × W.

        NOTE: the PSD from ``noise_psd_analysis.estimate_noise_psd()``
        uses a different convention (centred, patch-based scaling) and
        is NOT directly compatible.  Use ``None`` (blind MAD) or pass
        a known σ² from Chen / Pyatykh noise estimators instead.
    threshold_setting : str
        ``'s'``      — soft ACT (default, usually best PSNR).
        ``'h'``      — hard ACT.
        ``'ksigma'`` — k-sigma baseline (Starck/Candes/Donoho 2002).

    Returns
    -------
    denoised : ndarray (H, W), float64
        Denoised image.
    info : dict
        ``'noise_type'``         — 'white' or 'colored'
        ``'noise_var'``          — effective variance (float or 'fft_psd')
        ``'threshold_setting'``  — str
        ``'blind'``              — bool, True if variance was estimated
    """
    if threshold_setting not in ('s', 'h', 'ksigma'):
        raise ValueError(
            f"threshold_setting='{threshold_setting}': "
            f"choose from 's', 'h', 'ksigma'")

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img.shape}")
    H, W = img.shape

    # Pad to a UDCT-compatible shape.  UDCT silently breaks (huge
    # reconstruction error -> block / banding artifacts) when a spatial
    # dimension is not divisible by 2**(num_scales-1).  Pick num_scales
    # from the original size, then pad each dim to the next multiple.
    num_scales = _choose_num_scales(H, W)
    pad_mult = _udct_pad_multiple(num_scales)
    Hp = H + (-H) % pad_mult
    Wp = W + (-W) % pad_mult
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    N = Hp * Wp

    udct_op = _make_udct(Hp, Wp, num_scales=num_scales)

    # ── Forward curvelet transform ───────────────────────────────────────
    c_struct = udct_op.forward(img)

    # ── Noise variance estimation ────────────────────────────────────────
    blind = noise_var is None
    if blind:
        # MAD on the finest scale.  Non-square images give wedges of
        # very different aspect ratios, so a single wedge is biased;
        # aggregate MAD across all finest-scale wedges and take the
        # median.
        mads = []
        for direction in c_struct[-1]:
            for wedge in direction:
                vals = wedge.real.ravel() if np.iscomplexobj(wedge) \
                    else wedge.ravel()
                mads.append(
                    np.median(np.abs(vals - np.median(vals))) / 0.6745)
        noise_std = float(np.median(mads))
        noise_var = noise_std ** 2

    # ── Build FFT-PSD ────────────────────────────────────────────────────
    scalar_var = (np.isscalar(noise_var)
                  or (isinstance(noise_var, np.ndarray)
                      and noise_var.size == 1))
    if scalar_var:
        sigma2 = float(np.ravel(noise_var)[0]
                        if not np.isscalar(noise_var)
                        else noise_var)
        fft_psd = np.full((Hp, Wp), sigma2 * N, dtype=np.float64)
        noise_type = 'white'
    else:
        fft_psd = np.asarray(noise_var, dtype=np.float64)
        if fft_psd.shape == (H, W) and (pad_h or pad_w):
            fft_psd = np.pad(fft_psd, ((0, pad_h), (0, pad_w)), mode='wrap')
        if fft_psd.shape != (Hp, Wp):
            raise ValueError(
                f"FFT-PSD shape {fft_psd.shape} != image ({Hp}, {Wp})")
        # Check if essentially flat → white
        psd_range = float(fft_psd.max() - fft_psd.min())
        noise_type = 'white' if psd_range < 0.015 * N else 'colored'

    # ── Noise root-PSD per curvelet subband ──────────────────────────────
    rootpsd = _compute_curvelet_noise_rootpsd(fft_psd, udct_op)

    # ── ACT thresholding ─────────────────────────────────────────────────
    denoised_struct = _apply_act(
        c_struct, rootpsd, threshold_setting, noise_type)

    # ── Inverse curvelet transform ───────────────────────────────────────
    denoised = udct_op.backward(denoised_struct)
    if np.iscomplexobj(denoised):
        denoised = denoised.real

    # ── Crop back to original size ───────────────────────────────────────
    denoised = denoised[:H, :W]

    info = {
        'noise_type': noise_type,
        'noise_var': sigma2 if scalar_var else 'fft_psd',
        'threshold_setting': threshold_setting,
        'blind': blind,
    }
    return denoised, info
