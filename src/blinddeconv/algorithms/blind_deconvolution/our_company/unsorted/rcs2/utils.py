"""
utils.py

Utility functions for the Fergus et al. (SIGGRAPH 2006) blind deconvolution
algorithm, ported from MATLAB to Python.

Reference:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Transactions on Graphics (SIGGRAPH), 2006.

Core variational-Bayesian engine based on:
    J. Miskin, D. J. C. MacKay: "Ensemble Learning for Blind Image
    Separation and Deconvolution", Adv. in ICA, Springer-Verlag, 2000.

MATLAB -> Python critical differences handled in this port:
    ─────────────────────────────────────────────────────────────────────
    MATLAB arrays are 1-based, column-major (Fortran order).
    Python/NumPy arrays are 0-based, row-major (C order).

    MATLAB conv2(A, B, 'valid') does true convolution (flips kernel).
    -> scipy.signal.convolve2d(A, B, mode='valid')  (also flips kernel)

    MATLAB conv2(A, B, 'same') centres the output.
    -> scipy.signal.convolve2d(A, B, mode='same')

    MATLAB fft2(A, M, N) zero-pads A to size (M, N) then computes FFT.
    -> np.fft.fft2(A, s=(M, N))

    MATLAB ifft2 returns complex; real() is needed.
    -> np.real(np.fft.ifft2(...))

    MATLAB reshape(x, M, N) fills column-major.
    -> np.reshape(x, (M, N), order='F') for equivalent behaviour,
       BUT this code stores everything row-major (Python convention)
       and the ensemble vectors are packed row-major accordingly.
       Careful: we use order='C' (default) and adapt indexing.

    MATLAB imresize(A, scale, 'bilinear')
    -> cv2.resize or scipy.ndimage.zoom with order=1

    MATLAB edgetaper(I, kernel)
    -> custom implementation (no direct NumPy equivalent)

    MATLAB deconvlucy(I, kernel, iters)
    -> custom Richardson-Lucy implementation

    MATLAB histeq(I, hist_target)
    -> custom histogram matching implementation

    MATLAB rgb2gray (standard NTSC weights)
    -> custom with saturation awareness (rgb2gray_rob)

    MATLAB erfc / erfcx
    -> scipy.special.erfc / erfcx

    MATLAB gammaln
    -> scipy.special.gammaln
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.special import erfc, erfcx as _scipy_erfcx, gammaln
from scipy.ndimage import zoom
from typing import Tuple, Optional, Dict, Any


# ═════════════════════════════════════════════════════════════════════════════
# erfcx — scaled complementary error function
# ═════════════════════════════════════════════════════════════════════════════

def erfcx(x: np.ndarray) -> np.ndarray:
    """
    Scaled complementary error function:  erfcx(x) = exp(x^2) * erfc(x).

    Matches MATLAB erfcx(x).  Uses scipy.special.erfcx which is
    numerically stable across the entire input range.
    """
    return _scipy_erfcx(np.asarray(x, dtype=np.float64))


# ═════════════════════════════════════════════════════════════════════════════
# PSF / OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape[:2]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # Circular shift: move PSF centre to (0, 0)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).

    1. ifft2 → real part.
    2. Circular shift by +floor(psf_size/2) for each dim.
    3. Crop to psf_size.
    """
    full = np.real(ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# delta_kernel  (from delta_kernel.m)
# ═════════════════════════════════════════════════════════════════════════════

def delta_kernel(s: int) -> np.ndarray:
    """
    Create a delta (impulse) kernel of size s x s.
    If s is even, it is made odd (s+1).

    MATLAB equivalent: delta_kernel.m
    """
    if s % 2 == 0:
        s += 1
    out = np.zeros((s, s), dtype=np.float64)
    c = s // 2
    out[c, c] = 1.0
    return out


# ═════════════════════════════════════════════════════════════════════════════
# clip_image  (from clip_image.m)
# ═════════════════════════════════════════════════════════════════════════════

def clip_image(im: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    """
    Clip image values to [minval, maxval].

    MATLAB equivalent: clip_image.m
    """
    return np.clip(im, minval, maxval)


# ═════════════════════════════════════════════════════════════════════════════
# rgb2gray_rob  (from rgb2gray_rob.m)
# ═════════════════════════════════════════════════════════════════════════════

def rgb2gray_rob(rgb: np.ndarray, saturation_level: int = 250) -> np.ndarray:
    """
    Convert RGB image to grayscale with saturation awareness.

    Pixels where ANY channel exceeds saturation_level are set to 255 in output.
    Uses NTSC/YIQ luminance weights (same as MATLAB rgb2gray under the hood).

    MATLAB equivalent: rgb2gray_rob.m

    Parameters
    ----------
    rgb : (H, W, 3) uint8 or float array
    saturation_level : int, threshold above which a pixel is considered saturated

    Returns
    -------
    gray : (H, W) array, same dtype as input (uint8 stays uint8, float stays float)
    """
    # NTSC YIQ transform: first row of inv([1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703])
    # This is the luminance weight vector used in MATLAB's rgb2gray_rob
    T_inv = np.linalg.inv(np.array([
        [1.0, 0.956, 0.621],
        [1.0, -0.272, -0.647],
        [1.0, -1.106, 1.703]
    ]))
    weights = T_inv[0, :]  # first row

    # Find saturated pixels (any channel > saturation_level)
    sat_mask = np.any(rgb > saturation_level, axis=2)

    is_uint8 = (rgb.dtype == np.uint8)
    rgb_f = rgb.astype(np.float64)

    # Reshape to (H*W, 3) and multiply
    H, W = rgb.shape[:2]
    flat = rgb_f.reshape(H * W, 3)
    gray = (flat @ weights).reshape(H, W)

    if is_uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        gray[sat_mask] = 255
    else:
        gray = np.clip(gray, 0.0, 1.0)
        gray[sat_mask] = 255.0 if rgb.max() > 1.0 else 1.0

    return gray


# ═════════════════════════════════════════════════════════════════════════════
# normMDpdf  (from normMDpdf.m)
# ═════════════════════════════════════════════════════════════════════════════

def normMDpdf(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """
    Multivariate normal PDF.

    MATLAB equivalent: normMDpdf.m

    Parameters
    ----------
    x   : (nDims, nPoints)
    mu  : (nDims, 1) or (nDims,)
    sig : (nDims, nDims) covariance matrix

    Returns
    -------
    y : (nPoints,) density values
    """
    mu = np.asarray(mu, dtype=np.float64).flatten()
    nDims = x.shape[0]

    i_sig = np.linalg.inv(sig)
    d = ((2 * np.pi) ** (-nDims / 2.0)) / np.sqrt(np.linalg.det(sig))

    # x - mu broadcast
    tt = x - mu[:, np.newaxis]
    ttt = i_sig @ tt
    e = np.sum(tt * ttt, axis=0)

    y = d * np.exp(-0.5 * e)
    return y


# ═════════════════════════════════════════════════════════════════════════════
# reconsEdge3 + invDel2  (from reconsEdge3.m, invDel2.m — by Yair Weiss)
# Poisson reconstruction from gradients
# ═════════════════════════════════════════════════════════════════════════════

def invDel2(isize: int) -> np.ndarray:
    """
    Compute inverse Laplacian kernel in spatial domain.

    MATLAB equivalent: invDel2.m (by Yair Weiss)

    Parameters
    ----------
    isize : int, size of the kernel (should be even, typically 2*max(sx,sy))

    Returns
    -------
    invK : (isize, isize) real array — inverse Laplacian in spatial domain
    """
    K = np.zeros((isize, isize), dtype=np.float64)
    c = isize // 2  # MATLAB isize/2 (1-based equivalent)
    # MATLAB: K(isize/2, isize/2) = -4  etc.
    # In 0-based: K[c-1, c-1] = -4, but MATLAB isize/2 in 1-based = c
    # So 0-based index is c-1
    K[c - 1, c - 1] = -4.0
    K[c, c - 1] = 1.0       # K(isize/2+1, isize/2)
    K[c - 1, c] = 1.0       # K(isize/2, isize/2+1)
    K[c - 2, c - 1] = 1.0   # K(isize/2-1, isize/2)
    K[c - 1, c - 2] = 1.0   # K(isize/2, isize/2-1)

    Khat = fft2(K)
    # Avoid division by zero
    zero_mask = (Khat == 0)
    Khat[zero_mask] = 1.0
    invKhat = 1.0 / Khat
    invKhat[zero_mask] = 0.0

    invK = np.real(ifft2(invKhat))
    invK = -invK

    # Shift by one: conv2(invK, [1 0 0; 0 0 0; 0 0 0], 'same')
    shift_kernel = np.zeros((3, 3), dtype=np.float64)
    shift_kernel[0, 0] = 1.0
    invK = convolve2d(invK, shift_kernel, mode='same')

    return invK


def reconsEdge3(dx: np.ndarray, dy: np.ndarray,
                invKhat: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Poisson reconstruction of image from gradients dx, dy.

    MATLAB equivalent: reconsEdge3.m (by Yair Weiss)

    Parameters
    ----------
    dx : (sx, sy) x-gradient
    dy : (sx, sy) y-gradient
    invKhat : optional precomputed FFT of inverse Laplacian

    Returns
    -------
    im      : (sx, sy) reconstructed image
    invKhat : FFT of inverse Laplacian (for reuse)
    """
    sx, sy = dx.shape
    mxsize = max(sx, sy)

    if invKhat is None:
        invK = invDel2(2 * mxsize)
        invKhat = fft2(invK)

    # MATLAB: imX = conv2(dx, fliplr([0 1 -1]), 'same')
    # fliplr([0 1 -1]) = [-1 1 0]
    # conv2 with 'same' does convolution (flips kernel again), so effective
    # correlation with [-1 1 0], which is convolution with [0 1 -1]
    # In scipy convolve2d also flips, so we pass fliplr directly.
    kx = np.array([[-1.0, 1.0, 0.0]])    # fliplr([0 1 -1])
    ky = np.array([[-1.0], [1.0], [0.0]])  # flipud([0; 1; -1])

    imX = convolve2d(dx, kx, mode='same')
    imY = convolve2d(dy, ky, mode='same')

    imS = imX + imY

    imShat = fft2(imS, s=(2 * mxsize, 2 * mxsize))
    im = np.real(ifft2(imShat * invKhat))
    # MATLAB crops: im(mxsize+1:mxsize+sx, mxsize+1:mxsize+sy)
    # 0-based: im[mxsize:mxsize+sx, mxsize:mxsize+sy]
    im = im[mxsize:mxsize + sx, mxsize:mxsize + sy]

    return im, invKhat


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper  (MATLAB built-in equivalent)
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:
    """
    Reduce edge artifacts before deconvolution by tapering boundaries.

    Approximates MATLAB edgetaper(img, kernel).  Applies n_tapers iterations
    of alpha-blending between the image and a blurred version, where the
    blend weight is derived from the PSF autocorrelation.

    Parameters
    ----------
    img      : (H, W) or (H, W, C) image
    kernel   : (kh, kw) PSF
    n_tapers : int, number of blending passes (default 3)

    Returns
    -------
    result : same shape as img, edge-tapered
    """
    # Autocorrelation of the kernel
    k = kernel.astype(np.float64)
    # Normalise
    k = k / k.sum()

    # 1D projections for separable taper
    beta_col = k.sum(axis=1)  # (kh,)
    beta_row = k.sum(axis=0)  # (kw,)

    # Autocorrelation of 1D projections
    ac_col = np.correlate(beta_col, beta_col, mode='full')
    ac_row = np.correlate(beta_row, beta_row, mode='full')

    # Normalise to [0, 1]
    ac_col = ac_col / ac_col.max()
    ac_row = ac_row / ac_row.max()

    is_3d = (img.ndim == 3)
    if is_3d:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1
        img = img[:, :, np.newaxis]

    result = img.astype(np.float64).copy()

    for _ in range(n_tapers):
        # Blur the current result
        blurred = np.zeros_like(result)
        for c in range(C):
            blurred[:, :, c] = convolve2d(
                result[:, :, c], k, mode='same', boundary='wrap'
            )

        # Build 2D alpha mask from the 1D autocorrelations
        kh = len(ac_col)
        kw = len(ac_row)
        half_h = kh // 2
        half_w = kw // 2

        # Column taper (top and bottom edges)
        alpha_col = np.ones(H, dtype=np.float64)
        top_len = min(half_h, H)
        bot_len = min(half_h, H)
        alpha_col[:top_len] = ac_col[half_h - top_len:half_h]
        alpha_col[-bot_len:] = ac_col[half_h + 1:half_h + 1 + bot_len]

        # Row taper (left and right edges)
        alpha_row = np.ones(W, dtype=np.float64)
        left_len = min(half_w, W)
        right_len = min(half_w, W)
        alpha_row[:left_len] = ac_row[half_w - left_len:half_w]
        alpha_row[-right_len:] = ac_row[half_w + 1:half_w + 1 + right_len]

        # 2D alpha is the outer product (minimum for corners)
        alpha = alpha_col[:, np.newaxis] * alpha_row[np.newaxis, :]
        alpha = alpha[:, :, np.newaxis]  # (H, W, 1)

        result = alpha * result + (1.0 - alpha) * blurred

    if not is_3d:
        result = result[:, :, 0]

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Richardson-Lucy deconvolution  (replaces MATLAB deconvlucy)
# ═════════════════════════════════════════════════════════════════════════════

def deconvlucy(image: np.ndarray, psf: np.ndarray,
               iterations: int = 10) -> np.ndarray:
    """
    Richardson-Lucy deconvolution.

    Equivalent to MATLAB deconvlucy(image, psf, iterations).

    Parameters
    ----------
    image      : (H, W) or (H, W, C) blurred image (float64, >= 0)
    psf        : (kh, kw) point spread function (normalised, sums to 1)
    iterations : int, number of RL iterations

    Returns
    -------
    estimate : same shape as image, deconvolved result
    """
    psf = psf.astype(np.float64)
    psf_mirror = psf[::-1, ::-1]

    # Ensure non-negative
    im = np.maximum(image.astype(np.float64), 1e-12)
    estimate = im.copy()

    is_3d = (im.ndim == 3)

    for _ in range(iterations):
        if is_3d:
            reblurred = np.zeros_like(estimate)
            relative_blur = np.zeros_like(estimate)
            correction = np.zeros_like(estimate)
            for c in range(estimate.shape[2]):
                reblurred[:, :, c] = convolve2d(
                    estimate[:, :, c], psf, mode='same', boundary='wrap'
                )
            reblurred = np.maximum(reblurred, 1e-12)
            relative_blur = im / reblurred
            for c in range(estimate.shape[2]):
                correction[:, :, c] = convolve2d(
                    relative_blur[:, :, c], psf_mirror,
                    mode='same', boundary='wrap'
                )
        else:
            reblurred = convolve2d(estimate, psf, mode='same', boundary='wrap')
            reblurred = np.maximum(reblurred, 1e-12)
            relative_blur = im / reblurred
            correction = convolve2d(
                relative_blur, psf_mirror, mode='same', boundary='wrap'
            )

        estimate = estimate * correction
        estimate = np.maximum(estimate, 1e-12)

    return estimate


# ═════════════════════════════════════════════════════════════════════════════
# deconvlucy_intens  (from deconvlucy_intens.m)
# Runs Richardson-Lucy only on the Y (intensity) channel in YIQ
# ═════════════════════════════════════════════════════════════════════════════

def rgb2ntsc(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,1] float to YIQ, matching MATLAB rgb2ntsc."""
    T = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    flat = rgb.reshape(-1, 3)
    yiq = (flat @ T.T).reshape(rgb.shape)
    return yiq


def ntsc2rgb(yiq: np.ndarray) -> np.ndarray:
    """Convert YIQ to RGB [0,1] float, matching MATLAB ntsc2rgb."""
    T = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ])
    T_inv = np.linalg.inv(T)
    flat = yiq.reshape(-1, 3)
    rgb = (flat @ T_inv.T).reshape(yiq.shape)
    return rgb


def deconvlucy_intens(image_rgb: np.ndarray, kernel: np.ndarray,
                      iterations: int = 10) -> np.ndarray:
    """
    Run Richardson-Lucy only on the intensity (Y) channel of a colour image.

    MATLAB equivalent: deconvlucy_intens.m

    Parameters
    ----------
    image_rgb  : (H, W, 3) float64 RGB in [0, 1]
    kernel     : (kh, kw) PSF
    iterations : int

    Returns
    -------
    result : (H, W, 3) float64 RGB
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Color images only (H, W, 3)")

    yiq = rgb2ntsc(image_rgb)
    y_deconv = deconvlucy(yiq[:, :, 0], kernel, iterations)
    yiq_out = np.stack([y_deconv, yiq[:, :, 1], yiq[:, :, 2]], axis=2)
    rgb_out = ntsc2rgb(yiq_out)
    return rgb_out


# ═════════════════════════════════════════════════════════════════════════════
# histmatch  (from histmatch.m)
# ═════════════════════════════════════════════════════════════════════════════

def histmatch(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of `in_img` to `reference`.

    MATLAB equivalent: histmatch.m

    Parameters
    ----------
    in_img    : (H, W) or (H, W, C) float64 image in [0, 1]
    reference : (H', W') or (H', W', C) uint8 image in [0, 255]

    Returns
    -------
    out : (H, W) or (H, W, C) uint8 image in [0, 255]
    """
    # Convert both to grayscale for histogram computation
    if in_img.ndim == 3 and in_img.shape[2] == 3:
        gray_in = 0.2989 * in_img[:, :, 0] + 0.5870 * in_img[:, :, 1] + \
                  0.1140 * in_img[:, :, 2]
    else:
        gray_in = in_img if in_img.ndim == 2 else in_img[:, :, 0]

    ref_f = reference.astype(np.float64)
    if ref_f.ndim == 3 and ref_f.shape[2] == 3:
        gray_ref = 0.2989 * ref_f[:, :, 0] + 0.5870 * ref_f[:, :, 1] + \
                   0.1140 * ref_f[:, :, 2]
    else:
        gray_ref = ref_f if ref_f.ndim == 2 else ref_f[:, :, 0]

    # Compute reference histogram (256 bins, 0..255)
    hist_reference, _ = np.histogram(gray_ref.ravel(), bins=256, range=(0, 256))

    # Compute CDF of reference histogram
    hist_ref_norm = hist_reference.astype(np.float64)
    hist_ref_norm = hist_ref_norm / hist_ref_norm.sum()
    cdf_ref = np.cumsum(hist_ref_norm)

    # Build the transfer function t (MATLAB histeq equivalent)
    # t maps [0, 1] to [0, 1] via the reference CDF
    t = cdf_ref  # 256 entries mapping input level to output level

    # Apply transfer function to each channel
    if in_img.ndim == 3:
        C = in_img.shape[2]
    else:
        C = 1
        in_img = in_img[:, :, np.newaxis]

    out = np.zeros_like(in_img, dtype=np.float64)
    for c in range(C):
        # MATLAB: qm = interp1([0:255]/256, t, q(:))
        # Map input values [0,1] to bin indices
        q = in_img[:, :, c].ravel()
        bin_edges = np.arange(256) / 256.0
        qm = np.interp(q, bin_edges, t)
        out[:, :, c] = (256.0 * qm).reshape(in_img.shape[:2])

    out = np.clip(out, 0, 255).astype(np.uint8)

    if C == 1:
        out = out[:, :, 0]

    return out


# ═════════════════════════════════════════════════════════════════════════════
# fix_image  (from fix_image.m)
# ═════════════════════════════════════════════════════════════════════════════

def fix_image(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Normalise `in_img` and match its histogram to `reference`.

    MATLAB equivalent: fix_image.m

    Parameters
    ----------
    in_img    : (H, W) float image (may have negative values)
    reference : (H, W) image (positive, any numeric type)

    Returns
    -------
    out : (H, W) float image with matched histogram
    """
    SPACING = 0.05

    # Make reference [0, 1]
    ref_im = reference.astype(np.float64) / float(np.max(reference))

    # Histogram of reference
    x_bins = np.arange(0, 1.0 + SPACING, SPACING)
    hist_ref, _ = np.histogram(ref_im.ravel(), bins=x_bins)

    # Normalise input to [0, 1]
    m = np.min(in_img)
    in_shift = in_img - m
    mx = np.max(in_shift)
    if mx > 0:
        in_norm = in_shift / mx
    else:
        in_norm = in_shift

    # Simple histogram matching using CDF
    hist_ref_f = hist_ref.astype(np.float64)
    if hist_ref_f.sum() > 0:
        hist_ref_f = hist_ref_f / hist_ref_f.sum()
    cdf_ref = np.cumsum(hist_ref_f)

    # Map input values through CDF
    # Quantise input into bins
    n_bins = len(hist_ref)
    indices = np.clip(
        (in_norm.ravel() * n_bins).astype(int), 0, n_bins - 1
    )
    out = cdf_ref[indices].reshape(in_img.shape)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# automatic_patch_selector  (from automatic_patch_selector.m)
# ═════════════════════════════════════════════════════════════════════════════

def automatic_patch_selector(
    im: np.ndarray, patch_size: int,
    center_weight: float, sat_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Automatically select a high-variance, low-saturation image patch.

    MATLAB equivalent: automatic_patch_selector.m

    Parameters
    ----------
    im           : (H, W) float image
    patch_size   : int, size of square patch
    center_weight: float, weight for centre preference
    sat_mask     : (H, W) binary saturation mask

    Returns
    -------
    out_im         : (patch_size, patch_size) selected patch
    patch_location : (2,) array [sx, sy] (0-based column, row)
    """
    SMOOTH_SIGMA = 3

    H, W = im.shape

    # Centre weighting mask
    yy, xx = np.mgrid[0:H, 0:W]
    xx = xx - W // 2
    yy = yy - H // 2
    centre_weight_mask = np.exp(-center_weight / (W ** 2) * (xx ** 2 + yy ** 2))

    II = H * 2
    JJ = W * 2

    # Shift by patch_size using FFT convolution
    dk = delta_kernel(patch_size)
    centre_weight_mask = np.real(ifft2(
        fft2(centre_weight_mask, s=(II, JJ)) *
        fft2(dk, s=(II, JJ))
    ))

    # Patch mask for averaging
    pmask = np.ones((patch_size, patch_size), dtype=np.float64) / patch_size ** 2

    # Variance: E[I^2] - E[I]^2
    ei2 = np.real(ifft2(fft2(im ** 2, s=(II, JJ)) * fft2(pmask, s=(II, JJ))))
    mu2 = np.real(ifft2(fft2(im, s=(II, JJ)) * fft2(pmask, s=(II, JJ)))) ** 2
    w = ei2 - mu2

    # Saturation convolution
    q = np.real(ifft2(
        fft2(sat_mask.astype(np.float64), s=(II, JJ)) *
        fft2(pmask, s=(II, JJ))
    ))

    # Combined score: more variance, less saturation
    mean_im = np.mean(im)
    combined = centre_weight_mask * w / (q * mean_im ** 2 + 1)

    # Smooth
    from scipy.ndimage import gaussian_filter
    combined_smooth = np.real(ifft2(
        fft2(combined, s=(II, JJ)) *
        fft2(_fspecial_gaussian(8, SMOOTH_SIGMA), s=(II, JJ))
    ))

    # Crop to avoid edge effects
    combined_crop = combined_smooth[patch_size:II // 2, patch_size:JJ // 2]

    # Find max
    mm = np.argmax(combined_crop)
    sy, sx = np.unravel_index(mm, combined_crop.shape)

    patch_location = np.array([sx, sy])  # Note: 0-based, already shifted by -1

    # Extract patch
    out_im = im[sy:sy + patch_size, sx:sx + patch_size]

    return out_im, patch_location


def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel matching MATLAB fspecial('gaussian', [size size], sigma).
    """
    half = size // 2
    x = np.arange(-half, half + 1 if size % 2 else half, dtype=np.float64)
    if len(x) != size:
        x = np.arange(size, dtype=np.float64) - half
    g1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    g2d = np.outer(g1d, g1d)
    g2d /= g2d.sum()
    return g2d


# ═════════════════════════════════════════════════════════════════════════════
# GaussianMixtures1D  (from GaussianMixtures1D.m)
# EM for 1D mixture of Gaussians (zero-mean)
# ═════════════════════════════════════════════════════════════════════════════

def GaussianMixtures1D(
    x: np.ndarray, nComponents: int,
    max_iterations: int = 100,
    likelihood_threshold: float = 1e-5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    EM for 1D zero-mean mixture of Gaussians.

    MATLAB equivalent: GaussianMixtures1D.m

    Parameters
    ----------
    x             : (nPoints,) data
    nComponents   : int, number of mixture components
    max_iterations: int
    likelihood_threshold : float, convergence threshold

    Returns
    -------
    mu     : (1, nComponents) means (all zero)
    sigma  : (1, 1, nComponents) variances
    weight : (nComponents,) mixing weights
    log_likelihood : (n_iters,) log-likelihood trace
    """
    x = x.ravel().astype(np.float64)
    nPoints = len(x)

    # Initialise
    mu = np.zeros((1, nComponents), dtype=np.float64)
    sigma = np.zeros((1, 1, nComponents), dtype=np.float64)
    for a in range(nComponents):
        sigma[0, 0, a] = (1e6 - np.random.rand() * 1e6)
    sigma[0, 0, 0] = 1e6

    weight = np.ones(nComponents) / nComponents

    resp = np.zeros((nComponents, nPoints), dtype=np.float64)
    likelihoods = np.zeros((nComponents, nPoints), dtype=np.float64)
    log_likelihood_list = []
    delta_lh = np.inf

    for iteration in range(max_iterations):
        # E-step
        for c in range(nComponents):
            var_c = sigma[0, 0, c]
            if var_c <= 0:
                var_c = 1e-10
            normaliser = 1.0 / np.sqrt(2 * np.pi * var_c)
            offset = x - mu[0, c]
            exponent = offset ** 2 / var_c
            likelihoods[c, :] = weight[c] * normaliser * np.exp(-0.5 * exponent)

        # Log-likelihood
        ll = np.mean(np.log(np.sum(likelihoods, axis=0) + 1e-300))
        log_likelihood_list.append(ll)

        if iteration > 0:
            delta_lh = log_likelihood_list[-1] - log_likelihood_list[-2]

        # Normalise responsibilities
        resp_total = np.sum(likelihoods, axis=0) + 1e-300
        for c in range(nComponents):
            resp[c, :] = likelihoods[c, :] / resp_total

        # M-step
        for c in range(nComponents):
            total_resp_c = np.sum(resp[c, :])
            weight[c] = total_resp_c / nPoints

            # Mean fixed at 0
            mu[0, c] = 0.0

            # Variance
            offset = x - mu[0, c]
            sigma[0, 0, c] = np.sum(resp[c, :] * offset ** 2) / (total_resp_c + 1e-300)
            sigma[0, 0, c] += 1e-5  # regularisation

        # Fix first component to large variance for first 10 iterations
        if iteration < 10:
            sigma[0, 0, 0] = 1e6

        # Check convergence
        if iteration > 0 and delta_lh < likelihood_threshold:
            break

    log_likelihood = np.array(log_likelihood_list)
    return mu, sigma, weight, log_likelihood


# ═════════════════════════════════════════════════════════════════════════════
# estimate_priors2  (from estimate_priors2.m)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_priors(
    images: list,
    num_components: int = 4,
    num_scales: int = 1,
    gradient_mode: str = 'haar',
    max_im_size: int = 700,
    gamma_correction: float = 1.0,
    intensity_scaling: float = 1.0 / 256.0,
) -> list:
    """
    Estimate MoG prior parameters on image gradients.

    MATLAB equivalent: estimate_priors2.m

    Parameters
    ----------
    images          : list of (H, W) grayscale float images
    num_components  : int, number of MoG components
    num_scales      : int, number of scale levels
    gradient_mode   : str, 'haar' or 'steer'
    max_im_size     : int, max image dimension
    gamma_correction: float
    intensity_scaling : float

    Returns
    -------
    priors : list of dicts with keys 'pi' (weights) and 'gamma' (precisions)
    """
    SCALE_STEP = np.sqrt(2)

    priors = []

    for b in range(num_scales):
        scale = SCALE_STEP ** (-b)
        b_all = []

        for im in images:
            if im.ndim == 3:
                im_gray = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + \
                          0.1140 * im[:, :, 2]
            else:
                im_gray = im.astype(np.float64)

            H, W = im_gray.shape
            scale_factor = max_im_size / max(H, W)
            if scale_factor < 1.0:
                new_H = int(round(H * scale_factor))
                new_W = int(round(W * scale_factor))
                im_gray = zoom(im_gray, (new_H / H, new_W / W), order=1)

            im_gray = im_gray * intensity_scaling

            if gamma_correction != 1.0:
                im_gray = (im_gray ** gamma_correction) / (
                    256 ** (gamma_correction - 1)
                )

            if gradient_mode == 'haar':
                kx = np.array([[1.0, -1.0]])
                ky = np.array([[1.0], [-1.0]])
                b_x = convolve2d(im_gray, kx, mode='valid')
                b_y = convolve2d(im_gray, ky, mode='valid')

                if scale != 1.0:
                    new_shape_x = (
                        max(1, int(round(b_x.shape[0] * scale))),
                        max(1, int(round(b_x.shape[1] * scale)))
                    )
                    new_shape_y = (
                        max(1, int(round(b_y.shape[0] * scale))),
                        max(1, int(round(b_y.shape[1] * scale)))
                    )
                    b_x = zoom(b_x, (new_shape_x[0] / b_x.shape[0],
                                      new_shape_x[1] / b_x.shape[1]), order=1)
                    b_y = zoom(b_y, (new_shape_y[0] / b_y.shape[0],
                                      new_shape_y[1] / b_y.shape[1]), order=1)
            else:
                raise ValueError(f"Unsupported gradient mode: {gradient_mode}")

            b_all.extend(b_x.ravel().tolist())
            b_all.extend(b_y.ravel().tolist())

        b_all = np.array(b_all)

        mu, sigma, weight, _ = GaussianMixtures1D(b_all, num_components)

        priors.append({
            'pi': weight.copy(),
            'gamma': 1.0 / sigma[0, 0, :].copy()
        })

    return priors


# ═════════════════════════════════════════════════════════════════════════════
# Image resize helper (wraps scipy.ndimage.zoom for bilinear)
# ═════════════════════════════════════════════════════════════════════════════

def imresize(img: np.ndarray, output_shape: tuple = None,
             scale: float = None, method: str = 'bilinear') -> np.ndarray:
    """
    Resize image, approximating MATLAB imresize.

    Parameters
    ----------
    img          : (H, W) or (H, W, C) image
    output_shape : (new_H, new_W) target size (used if given)
    scale        : float scale factor (used if output_shape is None)
    method       : 'bilinear', 'bicubic', or 'nearest'

    Returns
    -------
    resized image
    """
    order_map = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}
    order = order_map.get(method, 1)

    if output_shape is not None:
        new_H, new_W = output_shape
    elif scale is not None:
        H = img.shape[0]
        W = img.shape[1]
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
    else:
        return img.copy()

    H, W = img.shape[:2]
    zoom_h = new_H / H
    zoom_w = new_W / W

    if img.ndim == 3:
        result = zoom(img, (zoom_h, zoom_w, 1), order=order)
    else:
        result = zoom(img, (zoom_h, zoom_w), order=order)

    # Ensure exact output shape
    if img.ndim == 3:
        result = result[:new_H, :new_W, :]
    else:
        result = result[:new_H, :new_W]

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Greenspan super-resolution  (from greenspan.m, create_greenspan_settings.m)
# ═════════════════════════════════════════════════════════════════════════════

def _binomial_filter(n: int) -> np.ndarray:
    """
    Binomial filter of length n (matching MATLAB binomialFilter).
    """
    if n == 1:
        return np.array([1.0])
    h = np.array([1.0, 1.0])
    for _ in range(n - 2):
        h = np.convolve(h, [1.0, 1.0])
    h = h / h.sum()
    return h


def create_greenspan_settings(
    c: float = 0.4,
    s: float = 5.0,
    bp: bool = True,
    factor: int = 1,
) -> dict:
    """
    Create parameter settings for Greenspan enhancement.

    MATLAB equivalent: create_greenspan_settings.m
    """
    lo_filt_1d = _binomial_filter(5)
    lo_filt = np.outer(lo_filt_1d, lo_filt_1d)

    return {
        'lo_filt': lo_filt,
        'c': c,
        's': s,
        'bp': bp,
        'factor': factor,
    }


def greenspan(im: np.ndarray, settings: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greenspan nonlinear image enhancement / super-resolution.

    MATLAB equivalent: greenspan.m

    Parameters
    ----------
    im       : (H, W) input image
    settings : dict from create_greenspan_settings

    Returns
    -------
    en : (zH, zW) enhanced image
    L0 : (zH, zW) high-frequency detail
    """
    S = settings
    z = 2 ** S['factor']
    lo_filt = S['lo_filt']

    # L1 = im - lowpass(im)
    im_low = convolve2d(im, lo_filt, mode='same', boundary='wrap')
    L1 = im - im_low

    # Upsample L1 by factor z
    H, W = im.shape
    L0 = imresize(L1, output_shape=(z * H, z * W), method='bilinear')
    L0 = L0 * (z ** 2)

    # Apply nonlinear clipping
    max_L0 = np.max(np.abs(L0)) if np.max(np.abs(L0)) > 0 else 1.0
    L0 = S['s'] * clip_image(L0, -(1 - S['c']) * max_L0, (1 - S['c']) * max_L0)

    # Optional bandpass
    if S['bp']:
        L0_low = convolve2d(L0, lo_filt, mode='same', boundary='wrap')
        L0 = L0 - L0_low

    # Upsample original
    en = imresize(im, output_shape=(z * H, z * W), method='bilinear')
    en = en * (z ** 2) / (z ** 2)  # normalise (MATLAB upConv scales by z^2)
    en = en + L0

    return en, L0


# ═════════════════════════════════════════════════════════════════════════════
# randnND  (from randnND.m)
# ═════════════════════════════════════════════════════════════════════════════

def randnND(mu: np.ndarray, Sigma: np.ndarray, N: int) -> np.ndarray:
    """
    Generate N samples from a multivariate Gaussian N(mu, Sigma).

    MATLAB equivalent: randnND.m

    Parameters
    ----------
    mu    : (d,) mean vector
    Sigma : (d, d) covariance matrix
    N     : int, number of samples

    Returns
    -------
    x : (d, N) samples
    """
    mu = np.asarray(mu, dtype=np.float64).flatten()
    d = len(mu)
    eigenvalues, V = np.linalg.eigh(Sigma)
    ss = np.sqrt(np.maximum(eigenvalues, 0))
    T = V @ np.diag(ss)
    x = T @ np.random.randn(d, N) + mu[:, np.newaxis]
    return x


# ═════════════════════════════════════════════════════════════════════════════
# prefZeros  (from prefZeros.m)
# ═════════════════════════════════════════════════════════════════════════════

def prefZeros(n: int, z: int) -> str:
    """
    Format integer n as string of length z with leading zeros.

    MATLAB equivalent: prefZeros.m
    """
    return str(n).zfill(z)
