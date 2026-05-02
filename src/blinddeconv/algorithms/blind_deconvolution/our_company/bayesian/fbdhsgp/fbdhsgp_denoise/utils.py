"""
utils.py

Utility (helper) functions for the FBDHSGP blind-deconvolution algorithm.

Ported from the MATLAB reference implementation (folder ``FBDHSGP/``)
accompanying the paper:

    X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
    "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
    Digital Signal Processing, 2016.

MATLAB → Python conversion notes (CRITICAL):
    ─────────────────────────────────────────────────────────────────────
    * MATLAB indexing is 1-based; Python is 0-based.
    * MATLAB ``conv2(A, B, 'valid')`` performs TRUE convolution (kernel
      flip).  In Python use ``scipy.signal.convolve2d(A, B, mode='valid')``
      which does the same.  Output size = (M-mk+1, N-nk+1).
    * MATLAB ``padarray(I,[p1 p2],'replicate','both')`` →
      ``np.pad(I,((p1,p1),(p2,p2)), mode='edge')``.
    * MATLAB ``padarray(I,[p1 p2],0,'both')`` →
      ``np.pad(I,((p1,p1),(p2,p2)), mode='constant')``.
    * MATLAB ``imresize(I,[r c],'bilinear')`` is bilinear with anti-aliasing.
      We approximate it with ``cv2.resize(..., INTER_LINEAR)`` which gives
      a faithful enough match for the multiscale pyramid (the algorithm
      is not numerically sensitive to the precise interpolant).
    * MATLAB ``psf2otf(psf, sz)``: zero-pad psf to ``sz``, circ-shift
      by ``-floor(size(psf)/2)`` so the centre lands at index (0,0),
      then ``fft2``.
    * MATLAB ``otf2psf(otf, psf_size)``: ``ifft2`` (real), circ-shift by
      ``+floor(psf_size/2)``, crop to ``psf_size``.
    * MATLAB ``rot90(A,2)`` = flip both axes → ``A[::-1, ::-1]``.
    * MATLAB ``fspecial('gaussian',[hr hc],sigma)`` constructs an
      ``hr × hc`` Gaussian centred at the middle pixel, normalised so
      ``sum == 1`` (after thresholding values below ``eps*max``).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# scikit-image is listed in requirements.txt and provides a bilinear
# ``resize`` with the anti-aliasing low-pass that MATLAB's ``imresize``
# enables by default.  We fall back to ``cv2`` only if skimage is missing.
try:  # pragma: no cover
    from skimage.transform import resize as _sk_resize
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False
import cv2


# =============================================================================
# PSF ↔ OTF conversions (psf2otf / otf2psf, MATLAB)
# =============================================================================

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Equivalent to MATLAB ``psf2otf(psf, shape)``.

    Steps:
        1. Zero-pad ``psf`` into an array of the requested ``shape``.
        2. Circularly shift by ``-floor(size(psf)/2)`` along each axis so
           that the centre of the PSF lands at index (0, 0).
        3. Return ``fft2``.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    out_h, out_w = shape
    padded = np.zeros((out_h, out_w), dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: Tuple[int, int]) -> np.ndarray:
    """
    Equivalent to MATLAB ``otf2psf(otf, psf_size)``.

    Steps:
        1. ``ifft2`` (take real part).
        2. Circularly shift by ``+floor(psf_size/2)`` along each axis.
        3. Crop to the top-left ``psf_size`` block.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# =============================================================================
# Padding helpers (padarray)
# =============================================================================

def pad_replicate(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """MATLAB ``padarray(x, [pad_h pad_w], 'replicate', 'both')``."""
    return np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")


def pad_zeros(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """MATLAB ``padarray(x, [pad_h pad_w], 0, 'both')``."""
    return np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")


# =============================================================================
# fspecial('gaussian', [hr, hc], sigma)
# =============================================================================

def fspecial_gaussian(shape: Sequence[int], sigma: float) -> np.ndarray:
    """
    2-D Gaussian filter, identical to MATLAB ``fspecial('gaussian', shape, sigma)``.

    Notes
    -----
    MATLAB's exact recipe::

        siz = (shape - 1) / 2          # may be fractional
        [x, y] = meshgrid(-siz(2):siz(2), -siz(1):siz(1))
        arg = -(x.^2 + y.^2) / (2*sigma^2)
        h = exp(arg)
        h(h < eps*max(h(:))) = 0
        sumh = sum(h(:))
        if sumh ~= 0:  h = h / sumh
    """
    hr, hc = int(shape[0]), int(shape[1])
    siz_r = (hr - 1) / 2.0
    siz_c = (hc - 1) / 2.0
    y, x = np.mgrid[-siz_r:siz_r + 1, -siz_c:siz_c + 1]
    arg = -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)
    h = np.exp(arg)
    eps = np.finfo(np.float64).eps
    h[h < eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def init_kernel(minsize: Sequence[int], sigma: float) -> np.ndarray:
    """
    Initial blur kernel at the coarsest pyramid level (FBDHSGP.m::init_kernel).

    Equivalent to MATLAB::

        k = fspecial('gaussian', minsize, Gsigma)
    """
    return fspecial_gaussian(minsize, sigma)


# =============================================================================
# Image resize (bilinear)
# =============================================================================

def imresize_bilinear(img: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    MATLAB ``imresize(img, [r c], 'bilinear')`` (with default anti-aliasing).

    Differences from a naive ``cv2.INTER_LINEAR`` resize matter for FBDHSGP:
    when the multiscale pyramid downsamples the blurred image by a large
    factor, MATLAB applies a **low-pass anti-aliasing prefilter**.  Without
    it the coarse-level data contains aliased high frequencies, which then
    cause the kernel estimate to drift off-centre at each scale and
    eventually produces "shadow" artefacts in the final restoration.

    We therefore prefer ``skimage.transform.resize`` with bilinear
    interpolation (``order=1``) and ``anti_aliasing=True`` for downsampling.
    For upsampling, anti-aliasing is implicitly disabled by skimage and the
    behaviour matches MATLAB.  We fall back to ``cv2.INTER_LINEAR`` only if
    skimage is unavailable.
    """
    r, c = int(target_shape[0]), int(target_shape[1])
    src = np.ascontiguousarray(img, dtype=np.float64)

    if _HAS_SKIMAGE:
        # skimage decides anti-aliasing automatically based on scale.
        # We pass anti_aliasing=None so it is enabled for downsampling only.
        out = _sk_resize(
            src,
            (r, c),
            order=1,                  # bilinear
            mode="edge",              # MATLAB-style boundary handling
            anti_aliasing=None,       # auto: True only when downsampling
            preserve_range=True,
        )
        return np.asarray(out, dtype=np.float64)

    # Fallback (no anti-aliasing).
    out = cv2.resize(src, (c, r), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float64)


# =============================================================================
# 4-tile UBC indices (getindex.m)
# =============================================================================

def getindex(n1: int, n2: int, hks1: int, hks2: int):
    """
    Pre-compute the four tile-index sets used by ``x_admm_ubc_bi`` to
    apply ``H`` and ``H^T`` with **undetermined boundary conditions**.

    Returns four lists of length 4.  Each entry is a tuple
    ``(rows, cols)`` of 1-D ``np.ndarray`` indices (0-based) suitable
    for use with ``np.ix_``::

        block = x[np.ix_(rows, cols)]

    Index sets reproduce the MATLAB function ``getindex.m`` 1:1.
    """
    # ---------- index1 : Px (extract padded patches of x) -----------------
    rows_lr = np.r_[n1 - hks1:n1, 0:n1, 0:hks1]      # length n1 + 2*hks1
    rows_top = np.r_[n1 - hks1:n1, 0:2 * hks1]        # length 3*hks1
    rows_bot = np.r_[n1 - 2 * hks1:n1, 0:hks1]        # length 3*hks1

    cols_left = np.r_[n2 - hks2:n2, 0:2 * hks2]       # length 3*hks2
    cols_right = np.r_[n2 - 2 * hks2:n2, 0:hks2]      # length 3*hks2
    cols_full = np.arange(n2)                         # length n2

    index1 = [
        (rows_lr,   cols_left),    # left
        (rows_lr,   cols_right),   # right
        (rows_top,  cols_full),    # top
        (rows_bot,  cols_full),    # bottom
    ]

    # ---------- index2 : where to write Hx tiles into ye ------------------
    index2 = [
        (np.arange(n1),                  np.arange(hks2)),                         # left
        (np.arange(n1),                  np.arange(n2 - hks2, n2)),                # right
        (np.arange(hks1),                np.arange(hks2, n2 - hks2)),              # top
        (np.arange(n1 - hks1, n1),       np.arange(hks2, n2 - hks2)),              # bottom
    ]

    # ---------- index3 : Py (extract padded patches of ye for H^T) --------
    cols_left3 = np.r_[n2 - hks2:n2, 0:3 * hks2]      # length 4*hks2
    cols_right3 = np.r_[n2 - 3 * hks2:n2, 0:hks2]     # length 4*hks2
    rows_top3 = np.r_[n1 - hks1:n1, 0:3 * hks1]       # length 4*hks1
    rows_bot3 = np.r_[n1 - 3 * hks1:n1, 0:hks1]       # length 4*hks1
    cols_mid = np.arange(hks2, n2 - hks2)             # length n2-2*hks2

    index3 = [
        (rows_lr,   cols_left3),   # left
        (rows_lr,   cols_right3),  # right
        (rows_top3, cols_mid),     # top
        (rows_bot3, cols_mid),     # bottom
    ]

    # ---------- index4 : where to write H^T(ye) tiles into hrye -----------
    index4 = [
        (np.arange(n1),                  np.arange(2 * hks2)),                         # left
        (np.arange(n1),                  np.arange(n2 - 2 * hks2, n2)),                # right
        (np.arange(2 * hks1),            np.arange(2 * hks2, n2 - 2 * hks2)),          # top
        (np.arange(n1 - 2 * hks1, n1),   np.arange(2 * hks2, n2 - 2 * hks2)),          # bottom
    ]

    return index1, index2, index3, index4


# =============================================================================
# Kernel centring (shift_kernel_img_space.m)
# =============================================================================

def shift_kernel_img_space(
    x: np.ndarray, k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centre the blur kernel ``k`` so the distance to its bounding-box
    boundary is minimised, and shift the latent image ``x`` by the
    opposite amount (kernels are shift-invariant: a shift in ``k`` is
    equivalent to a shift in ``x``).

    Direct port of MATLAB ``shift_kernel_img_space.m`` by Xu Zhou.
    """
    from scipy.signal import convolve2d  # local import: avoid hard top-level dep

    # --- 1. small threshold to remove noise ----------------------------------
    tao = 0.03
    threshold = min(k.max() * tao, 0.002)
    k = k.copy()
    k[k < threshold] = 0.0

    # --- 2. bounding box of nonzero entries ----------------------------------
    nz = np.argwhere(k > 0)
    if nz.size == 0:
        # fallback: nothing to do
        return x.copy(), k

    y_top, x_left = nz.min(axis=0)
    y_bottom, x_right = nz.max(axis=0)

    ksy, ksx = k.shape
    # MATLAB uses 1-based indexing: gap_left = x_left - 1.
    # In 0-based Python: gap_left = x_left, because `x_left` here is the
    # 0-based column index of the leftmost nonzero pixel.
    gap_left = x_left
    gap_right = ksx - 1 - x_right
    gap_top = y_top
    gap_bottom = ksy - 1 - y_bottom

    # --- 3. compute centring shift with small "bonus" tie-breaker -----------
    s_l = k[:, x_left].sum()
    s_r = k[:, x_right].sum()
    ratio_x = s_l / s_r if s_r != 0 else 1.0
    bonus_x = 0.01 if ratio_x >= 1 else -0.01
    shift_x = int(np.round((gap_right - gap_left + bonus_x) / 2.0))

    s_t = k[y_top, :].sum()
    s_b = k[y_bottom, :].sum()
    ratio_y = s_t / s_b if s_b != 0 else 1.0
    bonus_y = 0.01 if ratio_y >= 1 else -0.01
    shift_y = int(np.round((gap_bottom - gap_top + bonus_y) / 2.0))

    # --- 4. realise the shift via a delta-kernel convolution -----------------
    hksy = ksy // 2
    hksx = ksx // 2
    shift_filter = np.zeros((ksy, ksx), dtype=np.float64)
    # MATLAB: shift_filter(hksy+1+shift_y, hksx+1+shift_x) = 1
    # Python (0-based): index = (hksy+shift_y, hksx+shift_x)
    rr = hksy + shift_y
    cc = hksx + shift_x
    if 0 <= rr < ksy and 0 <= cc < ksx:
        shift_filter[rr, cc] = 1.0
    else:
        # Out of range — fall back to identity (no shift).
        shift_filter[hksy, hksx] = 1.0

    k_shift = convolve2d(k, shift_filter, mode="same")

    # x_shift = conv2(padarray(x,[hksy hksx],'replicate','both'),
    #                 rot90(shift_filter,2), 'valid')
    x_padded = pad_replicate(x, hksy, hksx)
    flipped = shift_filter[::-1, ::-1]
    x_shift = convolve2d(x_padded, flipped, mode="valid")

    return x_shift, k_shift


# =============================================================================
# Convenience: round-up odd
# =============================================================================

def make_odd(n: int) -> int:
    """Return ``n`` if odd, else ``n + 1`` (kernel sizes must be odd)."""
    return n if n % 2 == 1 else n + 1
