"""
solvers.py

Core solver functions for CPD (Cross Partial Derivative) blind deconvolution.

Ported from MATLAB code by Ting, Wang & Hwang (IEEE TIP 2025).
Reference:
    K.-C. Ting, S.-J. Wang, R.-B. Hwang: "Fast Blind Image Deblurring
    Based on Cross Partial Derivative", IEEE Transactions on Image
    Processing, vol. 34, pp. 8627-8640, 2025.
    DOI: 10.1109/TIP.2025.3645574

Contains:
    matched_peaks       — Step 01: CPD peaks matched to Harris corners
                          (f_01_Matched_Peaks.m)
    estimate_kernel     — Steps 01–06: full blind kernel estimation
                          (f_00_Estimate_Kernel.m)
    reconstruct_image   — Non-blind Tikhonov deconvolution + periodic
                          noise removal (f_00_Reconstruct_Image.m)

MATLAB → Python notes:
    - MATLAB imresize(img, factor) uses bicubic interpolation.
      → scipy.ndimage.zoom(img, factor, order=3) for 2D.
      For 3D array (H, W, N), zoom each slice individually
      to avoid interpolating across channels.
    - MATLAB sum(sum(A)) for 3D applies per-slice only if A(:,:,k)
      is addressed explicitly.  In the MATLAB code,
      su0 = su0 ./ sum(sum(su0)) operates on the whole 3D array
      elementwise after resize — this normalises each pixel across
      all candidates.  Actually re-reading the code: imresize(u0, factor)
      on a 3D array resizes each slice, and sum(sum(su0)) sums ALL
      elements to a scalar.  So it normalises by the global sum.
      We replicate this exactly.
    - MATLAB rgb2ycbcr / ycbcr2rgb operate on float64 [0,1] input.
      → Manual conversion with the ITU-R BT.601 matrix.
    - MATLAB padarray(h, [p q], 'Both') pads p rows on top+bottom
      and q cols on left+right.
      → np.pad(h, ((p,p),(q,q)), mode='constant')
    - MATLAB ifftshift / fftshift → np.fft.ifftshift / fftshift.
"""

import numpy as np
import time
from scipy.ndimage import zoom

from .utils import (
    compute_cpd,
    harris_corners,
    non_max_suppress,
    filter_peaks,
    produce_masks,
    connected_component_analysis,
    adjust_psf_center,
    gaussian_smoothing,
    spectrum_correlation,
    wrap_boundary_liu,
    zero_finding,
    periodic_noise_removal,
)


# ═════════════════════════════════════════════════════════════════════════════
# matched_peaks  (from f_01_Matched_Peaks.m)
# ═════════════════════════════════════════════════════════════════════════════

def matched_peaks(b: np.ndarray, opts: dict):
    """
    Compute CPD images and find peaks that match Harris corners.

    Equivalent to MATLAB ``f_01_Matched_Peaks.m``.

    Parameters
    ----------
    b : (H, W) float64 — grayscale blurred image.
    opts : dict — algorithm parameters (must contain 'cpd_sigma',
           'kernel_size_est', 'nms_sparsity').

    Returns
    -------
    bxy_P : (H, W) — positive CPD image.
    bxy_N : (H, W) — negative CPD image.
    harris_b : (M, 2) — Harris corners [x, y].
    bxy_P_match_peak : (K1, 7) — matched positive CPD peaks.
    bxy_P_match_harris : (K1, 2) — Harris corners matching positive peaks.
    bxy_N_match_peak : (K2, 7) — matched negative CPD peaks.
    bxy_N_match_harris : (K2, 2) — Harris corners matching negative peaks.
    """
    # Harris corners of blurred image
    harris_b = harris_corners(b)

    # CPD images
    _bxy, bxy_P, bxy_N = compute_cpd(b, opts['cpd_sigma'])

    # Positive CPD peaks + match to Harris
    bxy_P_peak = non_max_suppress(
        bxy_P, opts['kernel_size_est'], opts['nms_sparsity']
    )
    bxy_P_match_peak, bxy_P_match_harris = filter_peaks(bxy_P_peak, harris_b)

    # Negative CPD peaks + match to Harris
    bxy_N_peak = non_max_suppress(
        bxy_N, opts['kernel_size_est'], opts['nms_sparsity']
    )
    bxy_N_match_peak, bxy_N_match_harris = filter_peaks(bxy_N_peak, harris_b)

    return (bxy_P, bxy_N, harris_b,
            bxy_P_match_peak, bxy_P_match_harris,
            bxy_N_match_peak, bxy_N_match_harris)


# ═════════════════════════════════════════════════════════════════════════════
# _resize_2d  — helper for MATLAB-like imresize on 2D/3D arrays
# ═════════════════════════════════════════════════════════════════════════════

def _resize_2d(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize a 2D image by *factor* using bicubic interpolation.

    MATLAB ``imresize(img, factor)`` default is bicubic (order=3).
    ``scipy.ndimage.zoom`` with order=3 matches this.
    """
    return zoom(img, factor, order=3)


def _resize_3d(stack: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize each slice of a 3D array (H, W, N) independently.

    MATLAB ``imresize(u0, factor)`` on a 3D (H, W, N) array resizes
    the spatial dimensions of every page.
    """
    N = stack.shape[2]
    # Resize first slice to determine output shape
    s0 = zoom(stack[:, :, 0], factor, order=3)
    out = np.zeros((s0.shape[0], s0.shape[1], N), dtype=np.float64)
    out[:, :, 0] = s0
    for k in range(1, N):
        out[:, :, k] = zoom(stack[:, :, k], factor, order=3)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# estimate_kernel  (from f_00_Estimate_Kernel.m)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_kernel(b: np.ndarray, opts: dict,
                    num_candidates: int = 5,
                    index_ug: int = 0):
    """
    Estimate the blur kernel from a grayscale blurred image via CPD.

    Equivalent to MATLAB ``f_00_Estimate_Kernel.m``.

    Six-step pipeline:
        01. Matched CPD peaks (Harris + NMS + filtering).
        02. Produce masks around matched peaks.
        03. Connected component analysis on each mask.
        04. Centre, normalise and merge candidates.
        05. Resize blurred image and candidates.
        06. Spectrum correlation → rank → select best kernel.

    Parameters
    ----------
    b : (H, W) float64 — grayscale blurred image in [0, 1].
    opts : dict — algorithm parameters:
        'kernel_size_est' : int
        'cpd_sigma'       : float
        'nms_sparsity'    : float
        'cca_scale'       : float
        'cca_connect_type': int (4 or 8)
        'resize_factor'   : float
        'corr_sigma'      : float
    num_candidates : int — how many top candidates to keep (default 5).
    index_ug : int — 0-based index into top candidates (default 0 = best).

    Returns
    -------
    ug : (K, K) float64 — estimated kernel.
    u1 : (K, K, M) float64 — top-M candidates.
    run_time : list of float — per-step timings.
    """
    run_time = []

    # ── Step 01: Matched CPD peaks ───────────────────────────────────────
    t0 = time.time()
    (bxy_P, bxy_N, _harris_b,
     bxy_P_match_peak, _bxy_P_match_harris,
     bxy_N_match_peak, _bxy_N_match_harris) = matched_peaks(b, opts)
    run_time.append(time.time() - t0)

    # ── Step 02: Produce masks ───────────────────────────────────────────
    t0 = time.time()
    mask_size = opts['kernel_size_est']
    Mask_P = produce_masks(bxy_P, bxy_P_match_peak, mask_size)
    Mask_N = produce_masks(bxy_N, bxy_N_match_peak, mask_size)
    run_time.append(time.time() - t0)

    # ── Step 03: Connected Component Analysis ────────────────────────────
    t0 = time.time()
    cca_scale = opts['cca_scale']
    cca_conn = opts['cca_connect_type']

    # Positive
    uP_list = []
    for h in range(Mask_P.shape[2]):
        cleaned = connected_component_analysis(
            Mask_P[:, :, h], cca_scale, cca_conn
        )
        uP_list.append(cleaned)
    if uP_list:
        uP = np.stack(uP_list, axis=2)
    else:
        uP = np.zeros((mask_size, mask_size, 0), dtype=np.float64)

    # Negative
    uN_list = []
    for h in range(Mask_N.shape[2]):
        cleaned = connected_component_analysis(
            Mask_N[:, :, h], cca_scale, cca_conn
        )
        uN_list.append(cleaned)
    if uN_list:
        uN = np.stack(uN_list, axis=2)
    else:
        uN = np.zeros((mask_size, mask_size, 0), dtype=np.float64)
    run_time.append(time.time() - t0)

    # ── Step 04: Centre, normalise and merge ─────────────────────────────
    t0 = time.time()
    uP1 = adjust_psf_center(uP, mask_size)
    uN1 = adjust_psf_center(uN, mask_size)

    # Combine positive and negative candidates
    if uP1.shape[2] > 0 and uN1.shape[2] > 0:
        u0 = np.concatenate([uP1, uN1], axis=2)
    elif uP1.shape[2] > 0:
        u0 = uP1
    elif uN1.shape[2] > 0:
        u0 = uN1
    else:
        # No candidates found — return uniform kernel
        ug = np.ones((mask_size, mask_size), dtype=np.float64)
        ug /= ug.sum()
        return ug, ug[:, :, np.newaxis], run_time
    run_time.append(time.time() - t0)

    # ── Step 05: Resize image and candidates ─────────────────────────────
    t0 = time.time()
    resize_factor = opts['resize_factor']
    sb = _resize_2d(b, resize_factor)

    su0 = _resize_3d(u0, resize_factor)
    # MATLAB: su0 = su0 ./ sum(sum(su0))
    # In MATLAB, sum(sum(A)) on a 3D array sums along dim1 twice,
    # yielding (1,1,N) — so each slice is normalised independently.
    for k in range(su0.shape[2]):
        s = su0[:, :, k].sum()
        if s != 0:
            su0[:, :, k] /= s
    run_time.append(time.time() - t0)

    # ── Step 06: Spectrum correlation ────────────────────────────────────
    t0 = time.time()
    _sB_log, _sU0_log, CORR = spectrum_correlation(
        sb, su0, opts['corr_sigma']
    )

    # Select top candidates
    num_desired = min(su0.shape[2], num_candidates)
    top_indices = CORR[:num_desired, 0].astype(int)

    # u1 from the ORIGINAL (non-resized) candidates
    u1 = u0[:, :, top_indices]

    # Select the chosen kernel (0-based index_ug)
    ug = u1[:, :, index_ug]
    run_time.append(time.time() - t0)

    return ug, u1, run_time


# ═════════════════════════════════════════════════════════════════════════════
# Color-space conversion helpers (matching MATLAB rgb2ycbcr / ycbcr2rgb)
# ═════════════════════════════════════════════════════════════════════════════

def _rgb2ycbcr(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB float64 [0,1] → YCbCr float64.

    Uses the same ITU-R BT.601 matrix as MATLAB's rgb2ycbcr for
    double/float input:
        Y  =  65.481*R + 128.553*G +  24.966*B +  16
        Cb = -37.797*R -  74.203*G + 112.0  *B + 128
        Cr = 112.0  *R -  93.786*G -  18.214*B + 128
    all divided by 255 for [0,1] input.

    MATLAB rgb2ycbcr for double: output in [16/255, 235/255] for Y,
    [16/255, 240/255] for Cb/Cr.
    """
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    Y  = ( 65.481 * R + 128.553 * G +  24.966 * B +  16.0) / 255.0
    Cb = (-37.797 * R -  74.203 * G + 112.0   * B + 128.0) / 255.0
    Cr = (112.0   * R -  93.786 * G -  18.214 * B + 128.0) / 255.0

    return np.stack([Y, Cb, Cr], axis=2)


def _ycbcr2rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr float64 → RGB float64 [0,1].

    Inverse of _rgb2ycbcr, matching MATLAB's ycbcr2rgb for double input.

    From MATLAB docs the inverse transform:
        R = 298.082/256 * C  +             0 * D + 408.583/256 * E
        G = 298.082/256 * C - 100.291/256 * D - 208.120/256 * E
        B = 298.082/256 * C + 516.412/256 * D +             0 * E
    where C = Y - 16/255, D = Cb - 128/255, E = Cr - 128/255.
    """
    Y  = ycbcr[:, :, 0]
    Cb = ycbcr[:, :, 1]
    Cr = ycbcr[:, :, 2]

    C = Y  - 16.0 / 255.0
    D = Cb - 128.0 / 255.0
    E = Cr - 128.0 / 255.0

    R = 298.082 / 256.0 * C                         + 408.583 / 256.0 * E
    G = 298.082 / 256.0 * C - 100.291 / 256.0 * D - 208.120 / 256.0 * E
    B = 298.082 / 256.0 * C + 516.412 / 256.0 * D

    return np.stack([R, G, B], axis=2)


# ═════════════════════════════════════════════════════════════════════════════
# reconstruct_image  (from f_00_Reconstruct_Image.m)
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_image(b_in: np.ndarray, h: np.ndarray, opts: dict) -> np.ndarray:
    """
    Reconstruct (deblur) an image using the estimated kernel via Tikhonov
    regularisation in the frequency domain.

    Equivalent to MATLAB ``f_00_Reconstruct_Image.m``.

    Formula:
        R = (1/H) · [ |H|² / (|H|² + kH) ] · B

    Also performs periodic noise removal to suppress ringing artifacts
    caused by zeros of the kernel's transfer function.

    Parameters
    ----------
    b_in : (H, W) or (H, W, 3) float64 — blurred image in [0, 1].
    h : (Kh, Kw) float64 — estimated kernel.
    opts : dict — parameters:
        'tikhonov_factor'       : float — kH
        'smooth_blurred_image'  : str — 'Y' or 'N'
        'cpd_sigma'             : float — σ (used if smoothing)
        'kernel_size_est'       : int — kernel size (for noise removal)
        'zero_finding_distance' : int — distance for zero detection

    Returns
    -------
    r_out : same shape as b_in, float64 — reconstructed image.
    """
    kH = opts['tikhonov_factor']
    smooth_blur = opts['smooth_blurred_image']

    # ── Read the image ───────────────────────────────────────────────────
    if b_in.ndim == 2:
        color_size = 1
        b = b_in.copy()
    elif b_in.ndim == 3 and b_in.shape[2] == 3:
        color_size = 3
        b_ycbcr = _rgb2ycbcr(b_in)
        b = b_ycbcr[:, :, 0]
        Cb = b_ycbcr[:, :, 1]
        Cr = b_ycbcr[:, :, 2]
    elif b_in.ndim == 3 and b_in.shape[2] == 1:
        color_size = 1
        b = b_in[:, :, 0]
    else:
        raise ValueError(f"Unexpected image shape: {b_in.shape}")

    Nx_b, Ny_b = b.shape
    Nx_h, Ny_h = h.shape

    # ── Image size ───────────────────────────────────────────────────────
    Nx_bh = Nx_b + Nx_h - 1
    Ny_bh = Ny_b + Ny_h - 1

    # ── Padding ──────────────────────────────────────────────────────────
    # Blurred image — wrap boundary
    b1 = wrap_boundary_liu(b, (Nx_bh, Ny_bh))

    # Kernel — pad symmetrically then crop
    # MATLAB: padarray(h, [floor(Nx_b/2), floor(Ny_b/2)], 'Both')
    pad_r = Nx_b // 2
    pad_c = Ny_b // 2
    h1 = np.pad(h, ((pad_r, pad_r), (pad_c, pad_c)),
                mode='constant', constant_values=0)
    h1 = h1[:Nx_bh, :Ny_bh]

    # Centre to upper-left
    h1 = np.fft.ifftshift(h1)

    # ── FFT ──────────────────────────────────────────────────────────────
    B1 = np.fft.fft2(b1)
    B1_shift = np.fft.fftshift(B1)

    H1 = np.fft.fft2(h1)
    H1_shift = np.fft.fftshift(H1)

    # ── Reconstruct in frequency domain ──────────────────────────────────
    # Inverse term: 1 / H
    Inverse = 1.0 / H1_shift

    # Wiener-like weighting: |H|² / (|H|² + kH)
    absH1 = np.abs(H1_shift)
    Weighting = absH1 ** 2 / (absH1 ** 2 + kH)

    # Tikhonov filter
    Filter = Inverse * Weighting

    # Reconstruction
    if smooth_blur == 'N':
        R1_shift = Filter * B1_shift
    elif smooth_blur == 'Y':
        b_smooth = gaussian_smoothing(b, opts['cpd_sigma'])
        b_smooth1 = wrap_boundary_liu(b_smooth, (Nx_bh, Ny_bh))
        B_smooth1 = np.fft.fft2(b_smooth1)
        B_smooth1_shift = np.fft.fftshift(B_smooth1)
        R1_shift = Filter * B_smooth1_shift
    else:
        R1_shift = Filter * B1_shift

    # ── Remove periodic noise ────────────────────────────────────────────
    ZeroMask = zero_finding(H1_shift, opts['zero_finding_distance'])
    R1_shift = periodic_noise_removal(
        ZeroMask, R1_shift, opts['kernel_size_est']
    )

    # ── IFFT ─────────────────────────────────────────────────────────────
    R1 = np.fft.ifftshift(R1_shift)
    r1 = np.real(np.fft.ifft2(R1))
    r2 = r1[:Nx_b, :Ny_b]

    # ── Color space back-conversion ──────────────────────────────────────
    if color_size == 1:
        r_out = r2
    else:
        r_ycbcr = np.stack([r2, Cb, Cr], axis=2)
        r_out = _ycbcr2rgb(r_ycbcr)

    return r_out
