"""
utils.py

Utility functions for NSCP (Novel Sparse Channel Prior) blind deconvolution.

Ported from the Python implementation by D. Yang.
Reference:
    D. Yang, X. Wu, H. Yin: "Blind Image Deblurring via a Novel Sparse
    Channel Prior", Mathematics, 2022.
    https://www.mdpi.com/2227-7390/10/8/1238

Source files mapped:
    channels.py    → dark_channel, bright_channel, dcpl0norm, bcpl0norm
    gradient.py    → gradient_h, gradient_v, compute_gradients, gradient_mag_sq
    pyramid.py     → gaussian_pyramid, upsample_kernel, upsample_small_kernel,
                     upsample_l, kernel_to_fft_size
    threshold.py   → threshold_dark_channel, threshold_gradient
    kernel_utils.py → normalise_kernel, clamp_kernel, crop_kernel,
                      clean_kernel, pad_kernel_centered, extract_kernel_center
"""

import numpy as np
import cv2
from numpy.fft import fftshift, ifftshift


# ═════════════════════════════════════════════════════════════════════════════
# Dark / Bright Channel Priors  (from channels.py)
#
# Eq. (3) of the paper:
#   D(x) = min_{y in Psi(x)} min_c I^c(y)    (dark channel)
#   B(x) = max_{y in Psi(x)} max_c I^c(y)    (bright channel)
#
# Implementation uses morphological erosion/dilation as the min/max filters
# over the rectangular patch Psi(x).
# ═════════════════════════════════════════════════════════════════════════════

def dark_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an image.

    For grayscale (H, W): skip channel-min, just apply erosion.
    For colour (H, W, C): take per-pixel min across channels, then erode.

    Parameters
    ----------
    image       : float32 array, [0, 1]
    window_size : int, size of the square structuring element

    Returns
    -------
    dark : (H, W) float32 array
    """
    if image.ndim == 2:
        min_channel = image
    else:
        min_channel = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (window_size, window_size)
    )
    dark = cv2.erode(min_channel, kernel)
    return dark.astype(np.float32)


def bright_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute the bright channel of an image.

    For grayscale (H, W): skip channel-max, just apply dilation.
    For colour (H, W, C): take per-pixel max across channels, then dilate.

    Parameters
    ----------
    image       : float32 array, [0, 1]
    window_size : int, size of the square structuring element

    Returns
    -------
    bright : (H, W) float32 array
    """
    if image.ndim == 2:
        max_channel = image
    else:
        max_channel = np.max(image, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (window_size, window_size)
    )
    bright = cv2.dilate(max_channel, kernel)
    return bright.astype(np.float32)


def dcpl0norm(dark: np.ndarray) -> int:
    """L0 norm of the dark channel (number of non-zero elements)."""
    return int(np.count_nonzero(dark))


def bcpl0norm(bright: np.ndarray) -> int:
    """L0 norm of the bright channel (number of non-zero elements)."""
    return int(np.count_nonzero(bright))


# ═════════════════════════════════════════════════════════════════════════════
# Gradient operators  (from gradient.py)
#
# Forward-difference scheme: filter [1, -1] with zero-padding at boundary.
# Used to compute ∇l for the L0 gradient prior (Eq. 20).
# ═════════════════════════════════════════════════════════════════════════════

def gradient_h(I: np.ndarray) -> np.ndarray:
    """
    Horizontal gradient via forward difference: I[:,1:] - I[:,:-1].
    Zero-padded on the right to keep the same shape as I.
    Works for both 2-D (H, W) and 3-D (H, W, C) arrays.
    """
    if I.ndim == 2:
        return np.pad(I[:, 1:] - I[:, :-1], ((0, 0), (0, 1)))
    else:
        return np.pad(I[:, 1:, :] - I[:, :-1, :], ((0, 0), (0, 1), (0, 0)))


def gradient_v(I: np.ndarray) -> np.ndarray:
    """
    Vertical gradient via forward difference: I[1:,:] - I[:-1,:].
    Zero-padded on the bottom to keep the same shape as I.
    Works for both 2-D (H, W) and 3-D (H, W, C) arrays.
    """
    if I.ndim == 2:
        return np.pad(I[1:, :] - I[:-1, :], ((0, 1), (0, 0)))
    else:
        return np.pad(I[1:, :, :] - I[:-1, :, :], ((0, 1), (0, 0), (0, 0)))


def compute_gradients(img: np.ndarray):
    """
    Compute horizontal and vertical gradients.

    Returns
    -------
    gh, gv : arrays with same shape as *img*
    """
    gh = gradient_h(img)
    gv = gradient_v(img)
    return gh, gv


def gradient_mag_sq(grad):
    """Squared gradient magnitude: |∇I|² = g_h² + g_v²."""
    gh, gv = grad
    return gh ** 2 + gv ** 2


# ═════════════════════════════════════════════════════════════════════════════
# Gaussian pyramid  (from pyramid.py)
#
# Coarse-to-fine strategy (Section 4.5, Algorithm 2):
#   Results of coarse layer are up-sampled with bilinear interpolation
#   as the initialisation of the next fine layer.
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_pyramid(img: np.ndarray, num_levels: int) -> list:
    """
    Build a Gaussian pyramid and return levels in **coarse-to-fine** order.

    Parameters
    ----------
    img        : input image, float32
    num_levels : number of levels (≥ 1)

    Returns
    -------
    list of arrays, index 0 = coarsest, index -1 = finest (original size)
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    pyr = [img.copy()]
    for _ in range(1, num_levels):
        prev = pyr[-1]
        if prev.shape[0] < 2 or prev.shape[1] < 2:
            break
        down = cv2.pyrDown(prev)
        pyr.append(down)

    # Reverse: coarsest first
    return pyr[::-1]


def upsample_kernel(k: np.ndarray, target_hw: tuple) -> np.ndarray:
    """
    Resize kernel to given (H, W) and re-normalise.

    Parameters
    ----------
    k         : 2-D kernel array
    target_hw : (height, width)

    Returns
    -------
    Resized, non-negative, normalised kernel.
    """
    th, tw = target_hw
    resized = cv2.resize(k, (tw, th), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized /= s
    else:
        resized = np.zeros((th, tw), dtype=np.float32)
        resized[th // 2, tw // 2] = 1.0
    return resized


def upsample_small_kernel(
    k_small: np.ndarray,
    scale_factor: float = 2.0,
    max_size: tuple = None,
) -> np.ndarray:
    """
    Upsample a small kernel by *scale_factor* (default 2×).
    Optionally cap the result to *max_size* = (h_max, w_max).

    Returns normalised kernel.
    """
    kh, kw = k_small.shape
    new_kh = int(kh * scale_factor)
    new_kw = int(kw * scale_factor)

    if max_size is not None:
        mh, mw = max_size
        new_kh = min(new_kh, mh)
        new_kw = min(new_kw, mw)

    new_kh = max(1, new_kh)
    new_kw = max(1, new_kw)

    resized = cv2.resize(
        k_small, (new_kw, new_kh), interpolation=cv2.INTER_LINEAR
    )
    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized = resized / s
    else:
        resized = np.zeros((new_kh, new_kw), dtype=np.float32)
        resized[new_kh // 2, new_kw // 2] = 1.0
    return resized


def upsample_l(l: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Up-sample the latent image *l* to *target_shape* = (H, W)
    using bicubic interpolation.
    """
    target_h, target_w = target_shape
    return cv2.resize(l, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def kernel_to_fft_size(
    k_small: np.ndarray, image_shape: tuple
) -> np.ndarray:
    """
    Embed a small kernel into an image-sized array (top-left corner).
    Useful for direct FFT-based convolution without ``ifftshift``.
    """
    H, W = image_shape
    out = np.zeros((H, W), dtype=k_small.dtype)
    kh, kw = k_small.shape
    out[:kh, :kw] = k_small
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Thresholding  (from threshold.py)
#
# L0 proximal operators for the auxiliary variables p and g.
#   p  → Eq. (19):  p = D(l) if |D(l)| >= sqrt(w_k / xi), else 0
#   g  → Eq. (20):  g = ∇l  if |∇l|²  >= theta / lambda,   else 0
#
# The original author uses a simplified amplitude threshold (not squared)
# and constructs p as a masked copy of *l* (RGB) rather than D(l).
# We reproduce this behaviour exactly.
# ═════════════════════════════════════════════════════════════════════════════

def threshold_dark_channel(
    l: np.ndarray,
    D: np.ndarray,
    w_k: float,
    xi: float,
) -> np.ndarray:
    """
    Construct the auxiliary variable *p* for the dark-channel sub-problem.

    Pixels where ``D < w_k / xi`` are set to zero (sparsity enforcement).
    The remaining pixels retain the latent-image value *l*.

    Parameters
    ----------
    l   : current latent image (H, W) or (H, W, C)
    D   : dark channel (H, W)
    w_k : adaptive weight  mu / (||B(l)||_0 + epsilon)
    xi  : dark-channel penalty weight

    Returns
    -------
    p : same shape as *l*
    """
    threshold = w_k / xi
    mask = D * D > threshold

    p = l.copy()
    if l.ndim == 3:
        mask_3d = mask[:, :, np.newaxis]
        p[~np.broadcast_to(mask_3d, l.shape)] = 0.0
    else:
        p[~mask] = 0.0
    return p


def threshold_gradient(g, theta: float, lam: float):
    """
    L0 gradient thresholding (Eq. 20).

    Keep gradient entries whose amplitude exceeds ``theta / lambda``;
    set the rest to zero.

    Parameters
    ----------
    g     : tuple (gh, gv) of gradient arrays
    theta : numerator of threshold
    lam   : denominator of threshold (lambda)

    Returns
    -------
    (gh_thresholded, gv_thresholded)
    """
    gh, gv = g
    mag_sq = gh * gh + gv * gv
    T = theta / (lam + 1e-8)

    # L0 proximal: keep gradient vector if ||g||^2 > T.
    # DCP does the same: (h**2 + v**2) < wei_grad / beta.
    # The original code compared sqrt(gh^2+gv^2) > T, which is
    # equivalent to gh^2+gv^2 > T^2  — far too lenient.
    mask = mag_sq > T
    return gh * mask, gv * mask


# ═════════════════════════════════════════════════════════════════════════════
# Kernel utilities  (from kernel_utils.py)
#
# Post-processing pipeline for the estimated kernel:
#   1. Clamp negatives to zero.
#   2. Threshold small values (< 0.1 % of max).
#   3. fftshift to centre the kernel spatially.
#   4. Crop to minimal bounding box of non-zero entries.
#   5. Normalise to sum = 1.
#
# Also contains:
#   - pad_kernel_centered: place kernel in centre of (H2, W2) then ifftshift
#     for correct FFT phase alignment.
#   - extract_kernel_center: after ifft2, fftshift + crop + normalise.
# ═════════════════════════════════════════════════════════════════════════════

def normalise_kernel(k: np.ndarray) -> np.ndarray:
    """Normalise kernel so that it sums to 1."""
    s = k.sum()
    if s > 1e-8:
        return k / s
    return k


def clamp_kernel(k: np.ndarray) -> np.ndarray:
    """Clamp negative values to zero."""
    return np.clip(k, 0, None)


def crop_kernel(k: np.ndarray) -> np.ndarray:
    """
    Crop kernel to its minimal bounding box of non-zero entries.
    If the kernel is entirely zero, return it unchanged.
    """
    nz = np.nonzero(k)
    if len(nz[0]) == 0:
        return k

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()
    return k[y_min : y_max + 1, x_min : x_max + 1]


def resize_kernel(k: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize kernel to *target_shape* and re-normalise."""
    resized = cv2.resize(
        k, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    resized = np.clip(resized, 0, None)
    return normalise_kernel(resized)


def clean_kernel(k: np.ndarray) -> np.ndarray:
    """
    Full kernel post-processing pipeline.

    Steps:
        1. Clamp negatives to zero.
        2. Threshold values < 0.1 % of max.
        3. ``fftshift`` to centre.
        4. Crop to bounding box.
        5. Normalise to sum = 1.  Fall back to a delta if all zeros.
    """
    # 1. Clamp
    k = np.clip(k, 0, None)

    # 2. Relative threshold
    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    # 3. Centre
    k = fftshift(k)

    # 4. Crop
    k = crop_kernel(k)

    # 5. Normalise
    s = k.sum()
    if s > 1e-12:
        k = k / s
    else:
        k = np.zeros_like(k)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    return k


def pad_and_ifftshift_kernel(
    k_small: np.ndarray, image_shape: tuple
) -> np.ndarray:
    """
    Place a small kernel at the top-left corner and ``ifftshift`` so that
    its centre is at (0, 0) in the frequency domain.
    """
    H, W = image_shape
    padded = np.zeros((H, W), dtype=np.float32)
    kh, kw = k_small.shape
    padded[:kh, :kw] = k_small
    return ifftshift(padded)


def postprocess_kernel_spatial(k_full: np.ndarray) -> np.ndarray:
    """
    Post-process a full-size kernel after inverse FFT:
    ``fftshift`` → clamp → threshold → crop → normalise.
    """
    k = fftshift(k_full.real)
    k = np.clip(k, 0, None)

    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    nz = np.nonzero(k)
    if len(nz[0]) == 0:
        return k

    y0, y1 = nz[0].min(), nz[0].max()
    x0, x1 = nz[1].min(), nz[1].max()
    k_cropped = k[y0 : y1 + 1, x0 : x1 + 1]

    s = k_cropped.sum()
    if s > 1e-12:
        k_cropped = k_cropped / s
    return k_cropped


def pad_kernel_centered(k: np.ndarray, out_shape: tuple) -> np.ndarray:
    """
    Place kernel at the **centre** of an (H2, W2) array, then ``ifftshift``
    so that the kernel origin sits at index (0, 0) for FFT-based convolution.

    Parameters
    ----------
    k         : (kh, kw) kernel
    out_shape : (H2, W2) target padded size

    Returns
    -------
    kpad : (H2, W2) float32 array ready for ``fft2``
    """
    H2, W2 = out_shape
    kh, kw = k.shape

    kpad = np.zeros((H2, W2), dtype=np.float32)
    cx = (H2 - kh) // 2
    cy = (W2 - kw) // 2
    kpad[cx : cx + kh, cy : cy + kw] = k

    kpad = np.fft.ifftshift(kpad)
    return kpad


def extract_kernel_center(
    k_full: np.ndarray, expected_size: tuple = None
) -> np.ndarray:
    """
    Extract the compact kernel from a full-size array after ``ifft2``.

    Steps:
        1. ``fftshift`` to centre the peak.
        2. Clamp negatives; threshold small values.
        3. Crop to *expected_size* (centred) or auto-bbox.
        4. Normalise.

    Parameters
    ----------
    k_full        : (H, W) real-valued ifft2 result
    expected_size : (kh, kw) or None for auto-crop

    Returns
    -------
    k_small : normalised compact kernel
    """
    k = np.real(fftshift(k_full))
    k = np.clip(k, 0, None)

    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    H, W = k.shape

    if expected_size is not None:
        kh, kw = expected_size
        cy, cx = H // 2, W // 2
        y0 = cy - kh // 2
        x0 = cx - kw // 2
        cropped = k[y0 : y0 + kh, x0 : x0 + kw]
    else:
        nz = np.nonzero(k)
        if len(nz[0]) == 0:
            return np.array([[1.0]], dtype=np.float32)
        y0, y1 = nz[0].min(), nz[0].max()
        x0, x1 = nz[1].min(), nz[1].max()
        cropped = k[y0 : y1 + 1, x0 : x1 + 1]

    s = cropped.sum()
    if s <= 1e-12:
        out = np.zeros_like(cropped, dtype=np.float32)
        out[out.shape[0] // 2, out.shape[1] // 2] = 1.0
        return out

    return (cropped / s).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Kernel initialisation helpers
# ═════════════════════════════════════════════════════════════════════════════

def make_delta_kernel(kernel_size) -> np.ndarray:
    """
    Create a delta (identity) kernel: all zeros except the centre pixel = 1.

    Parameters
    ----------
    kernel_size : int or (kh, kw)

    Returns
    -------
    k : float32 array
    """
    if isinstance(kernel_size, tuple):
        kh, kw = kernel_size
    else:
        kh = kw = int(kernel_size)

    k = np.zeros((kh, kw), dtype=np.float32)
    k[kh // 2, kw // 2] = 1.0
    return k
