"""
utils.py

Utility functions for Low-Rank Kernel blind deconvolution.

Ported from MATLAB code by Li Siyao et al.
Reference:
    Li Siyao, Shiyu Zhao, Wenzhe Wang, Ping Tan:
    "Understanding Kernel Size in Blind Deconvolution", WACV 2019.

The non-blind deconvolution (fast_deconv_bregman / solve_image_bregman)
is from Krishnan & Fergus "Fast Image Deconvolution using
Hyper-Laplacian Priors", NIPS 2009.

Contains:
    psf2otf                — PSF to OTF conversion (MATLAB psf2otf)
    imresize               — image resize (approximation of MATLAB imresize)
    edgetaper              — edge tapering (approximation of MATLAB edgetaper)
    center_kernel_separate — centre kernel by translation, shift images
    solve_image_bregman    — LUT-based proximal operator (solve_image_bregman.m)

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    Indexing:
        MATLAB is 1-based, Python is 0-based.

    conv2(A, B, 'same'|'valid'|'full'):
        Both MATLAB conv2 and scipy.signal.convolve2d perform TRUE
        convolution (kernel is flipped internally).  Semantics match.

    rot90(k, 2):
        MATLAB rot90(k,2) = np.rot90(k, 2).  Both rotate 180 degrees.

    norm(x(:)):
        MATLAB norm for a column vector = L2 norm.
        -> np.linalg.norm(x.ravel())

    svd:
        MATLAB [U, S, V] = svd(X) : S is a diagonal MATRIX, V is NOT
        transposed.
        NumPy  U, s, Vh = np.linalg.svd(X) : s is a 1-D vector,
        Vh = V'.  CRITICAL: multiply as U @ diag(s) @ Vh.

    psf2otf(psf, shape):
        Zero-pad PSF, circularly shift centre to (0,0), then fft2.
        circshift amount: -floor(size(psf)/2) per dim.

    imresize(img, [h w], 'bilinear'):
        MATLAB uses anti-aliased bilinear interpolation.
        -> scipy.ndimage.zoom(order=1) — no anti-aliasing, but close
           enough for the multi-scale pyramid.

    sign(0):
        MATLAB sign(0) = 0, np.sign(0) = 0.  Same.

    sqrt / log of negative reals:
        MATLAB transparently returns complex.
        NumPy returns NaN unless array dtype is complex.
        -> Cast to complex128 explicitly before sqrt / log.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import zoom


# ═════════════════════════════════════════════════════════════════════════════
# psf2otf
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

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    # MATLAB circshift amounts: -floor(size(psf)/2) per dim
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


# ═════════════════════════════════════════════════════════════════════════════
# imresize  (approximation of MATLAB imresize)
# ═════════════════════════════════════════════════════════════════════════════

def imresize(img: np.ndarray, target_size, method: str = 'bilinear') -> np.ndarray:
    """
    Resize *img* to *target_size* = (height, width).

    Parameters
    ----------
    img         : 2-D ndarray
    target_size : (int, int) — target (rows, cols).
                  Also accepts a numpy array of length 2.
    method      : 'bilinear' (order=1) or 'bicubic' (order=3)
    """
    th = int(target_size[0])
    tw = int(target_size[1])
    if th <= 0 or tw <= 0:
        return np.zeros((max(th, 1), max(tw, 1)), dtype=np.float64)
    if img.size == 0:
        return np.zeros((th, tw), dtype=np.float64)

    order = 1 if method == 'bilinear' else 3
    zh = th / img.shape[0]
    zw = tw / img.shape[1]
    result = zoom(img.astype(np.float64), (zh, zw), order=order)

    # scipy.ndimage.zoom may be off by ±1 pixel — fix to exact target size
    rh, rw = result.shape
    if rh != th or rw != tw:
        out = np.zeros((th, tw), dtype=np.float64)
        mh, mw = min(rh, th), min(rw, tw)
        out[:mh, :mw] = result[:mh, :mw]
        return out
    return result


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper  (approximation of MATLAB edgetaper)
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Approximate MATLAB ``edgetaper(I, PSF)``.

    Blends image edges with a locally-blurred version using weights
    derived from the PSF autocorrelation projected onto each axis,
    so that the image is unchanged in the interior and smoothly
    transitions to the blurred version at the boundaries.
    """
    blurred = fftconvolve(img, psf, mode='same')

    kh, kw = psf.shape
    ih, iw = img.shape

    # 1-D projections of PSF onto each axis
    proj_y = psf.sum(axis=1)  # (kh,)
    proj_x = psf.sum(axis=0)  # (kw,)

    # 1-D autocorrelation of each projection — length 2k-1
    ac_y = np.correlate(proj_y, proj_y, mode='full')
    ac_x = np.correlate(proj_x, proj_x, mode='full')

    # Normalise to [0, 1]
    ac_y = ac_y / ac_y.max()
    ac_x = ac_x / ac_x.max()

    # Blend weights: 1.0 in the centre, tapering at edges
    Ly = kh - 1  # half-length of autocorrelation
    Lx = kw - 1

    wy = np.ones(ih)
    wx = np.ones(iw)

    ny = min(Ly, ih)
    nx = min(Lx, iw)

    wy[:ny] = np.minimum(wy[:ny], ac_y[:ny])
    wy[-ny:] = np.minimum(wy[-ny:], ac_y[-ny:])
    wx[:nx] = np.minimum(wx[:nx], ac_x[:nx])
    wx[-nx:] = np.minimum(wx[-nx:], ac_x[-nx:])

    alpha = np.outer(wy, wx)
    return alpha * img + (1.0 - alpha) * blurred


# ═════════════════════════════════════════════════════════════════════════════
# center_kernel_separate  (from center_kernel_separate.m)
# ═════════════════════════════════════════════════════════════════════════════

def center_kernel_separate(x: np.ndarray, y: np.ndarray,
                           k: np.ndarray):
    """
    Centre the blur kernel by translating it so that its centre of mass
    coincides with its geometric centre.  Both gradient images *x* and *y*
    are shifted in the opposite direction to compensate.

    Equivalent to MATLAB ``center_kernel_separate.m``.

    Parameters
    ----------
    x, y : 2-D arrays — gradient images (e.g. dx, dy derivatives)
    k    : 2-D array  — blur kernel (non-negative, sums to 1)

    Returns
    -------
    x, y, k : shifted copies
    """
    nrows, ncols = k.shape

    # Centre of mass in MATLAB 1-based coordinates
    # MATLAB: mu_y = sum([1:nrows] .* sum(k,2)')
    #         mu_x = sum([1:ncols] .* sum(k,1))
    mu_y = np.sum(np.arange(1, nrows + 1) * k.sum(axis=1))
    mu_x = np.sum(np.arange(1, ncols + 1) * k.sum(axis=0))

    # Offset from geometric centre (MATLAB 1-based centre = floor(n/2)+1)
    offset_x = int(np.round(np.floor(ncols / 2) + 1 - mu_x))
    offset_y = int(np.round(np.floor(nrows / 2) + 1 - mu_y))

    # Build a delta-shift kernel
    sk_h = abs(offset_y) * 2 + 1
    sk_w = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sk_h, sk_w))
    # MATLAB 1-based: shift_kernel(abs(oy)+1+oy, abs(ox)+1+ox) = 1
    # Python 0-based: subtract 1
    shift_kernel[abs(offset_y) + offset_y,
                 abs(offset_x) + offset_x] = 1.0

    # Shift kernel (convolution)
    k = convolve2d(k, shift_kernel, mode='same')

    # Shift images in the *opposite* direction
    # MATLAB: conv2(x, flipud(fliplr(shift_kernel)), 'same')
    inv_sk = shift_kernel[::-1, ::-1]
    x = convolve2d(x, inv_sk, mode='same')
    y = convolve2d(y, inv_sk, mode='same')

    return x, y, k


# ═════════════════════════════════════════════════════════════════════════════
# solve_image_bregman  (from solve_image_bregman.m)
#
# Proximal operator:  min_w |w|^alpha + (beta/2)(w - v)^2
# Uses a Look-Up Table (LUT) built once per (beta, alpha) pair.
# ═════════════════════════════════════════════════════════════════════════════

# Module-level LUT cache  (equivalent to MATLAB persistent variables)
_lut_cache: dict = {}


def _compute_w1(v: np.ndarray, beta: float) -> np.ndarray:
    """Proximal for alpha = 1: soft thresholding."""
    return np.maximum(np.abs(v) - 1.0 / beta, 0.0) * np.sign(v)


def _compute_w23(v: np.ndarray, beta: float) -> np.ndarray:
    """
    Proximal for alpha = 2/3  — solve a quartic equation
    (Ferrari's method).

    CRITICAL: MATLAB sqrt / log handle complex numbers transparently;
    NumPy does not — we must cast to complex128 explicitly.
    """
    epsilon = 1e-6

    orig_shape = v.shape
    vr = v.ravel().astype(np.complex128)
    n = vr.size

    k_val = 8.0 / (27.0 * beta ** 3)
    m = np.full(n, k_val, dtype=np.complex128)

    v2 = vr * vr
    v3 = v2 * vr
    v4 = v3 * vr
    m2 = m * m
    m3 = m2 * m

    aq = -1.125 * v2          # "alpha" in the quartic formula
    bq = 0.25 * v3            # "beta"  in the quartic formula

    q = -0.125 * m * v2
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = -q / 2.0 + np.sqrt(-m3 / 27.0 + m2 * v4 / 256.0)

        u = np.exp(np.log(r1) / 3.0)
        yy = 2.0 * (-5.0 / 18.0 * aq + u + m / (3.0 * u))

        W = np.sqrt(aq / 3.0 + yy)

        inner_p = np.sqrt(-(aq + yy + bq / W))
        inner_m = np.sqrt(-(aq + yy - bq / W))

    # All 4 roots — shape (n, 4)
    roots = np.column_stack([
        0.75 * vr + 0.5 * (W + inner_p),
        0.75 * vr + 0.5 * (W - inner_p),
        0.75 * vr + 0.5 * (-W + inner_m),
        0.75 * vr + 0.5 * (-W - inner_m),
    ])

    # ---------- Root selection (vectorised, mirrors MATLAB exactly) ----------
    sv = np.sign(v.ravel())[:, None]          # (n, 1)
    abs_v = np.abs(v.ravel())[:, None]        # (n, 1)

    rsv = np.real(roots) * sv                 # real(root) * sign(v)

    mask = ((np.abs(np.imag(roots)) < epsilon) &
            (rsv > abs_v / 2.0) &
            (rsv < abs_v))

    candidates = mask * rsv                   # 0 where invalid
    # Sort descending along the 4-root axis, take the largest
    best_rsv = np.sort(candidates, axis=1)[:, ::-1][:, 0]

    w = (best_rsv * sv.ravel()).real.astype(np.float64)
    w[np.isnan(w)] = 0.0
    return w.reshape(orig_shape)


def _compute_w12(v: np.ndarray, beta: float) -> np.ndarray:
    """
    Proximal for alpha = 1/2  — solve a cubic equation.

    CRITICAL: complex arithmetic needed.
    """
    epsilon = 1e-6

    orig_shape = v.shape
    vr = v.ravel().astype(np.complex128)
    n = vr.size

    k_val = -0.25 / (beta ** 2)
    m = np.full(n, k_val, dtype=np.complex128) * np.sign(v.ravel())

    t1 = (2.0 / 3.0) * vr
    v2 = vr * vr
    v3 = v2 * vr

    # MATLAB: t2 = exp(log(-27*m - 2*v3 + 3*sqrt(3)*sqrt(27*m.^2 + 4*m.*v3))/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        inner = -27.0 * m - 2.0 * v3 + (3.0 * np.sqrt(3.0 + 0j)) * np.sqrt(27.0 * m ** 2 + 4.0 * m * v3)
        t2 = np.exp(np.log(inner) / 3.0)
        t3 = v2 / t2

    # Pre-compute constants
    sqrt3 = np.sqrt(3.0 + 0j)
    c1 = 2.0 ** (1.0 / 3) / 3.0
    c_21 = (1.0 + 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3))
    c_22 = (1.0 - 1j * sqrt3) / (6.0 * 2.0 ** (1.0 / 3))
    c_31 = (1.0 - 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3))
    c_32 = (1.0 + 1j * sqrt3) / (6.0 * 2.0 ** (1.0 / 3))

    # 3 roots — shape (n, 3)
    with np.errstate(divide='ignore', invalid='ignore'):
        roots = np.column_stack([
            t1 + c1 * t3 + t2 / (3.0 * 2.0 ** (1.0 / 3)),
            t1 - c_21 * t3 - c_22 * t2,
            t1 - c_31 * t3 - c_32 * t2,
        ])

    # Handle NaN / Inf
    roots[np.isnan(roots) | np.isinf(roots)] = 0.0

    # ---------- Root selection (for alpha = 1/2: between 2|v|/3 and |v|) -----
    sv = np.sign(v.ravel())[:, None]
    abs_v = np.abs(v.ravel())[:, None]

    rsv = np.real(roots) * sv

    mask = ((np.abs(np.imag(roots)) < epsilon) &
            (rsv > 2.0 * abs_v / 3.0) &
            (rsv < abs_v))

    candidates = mask * rsv
    best_rsv = np.sort(candidates, axis=1)[:, ::-1][:, 0]

    w = (best_rsv * sv.ravel()).real.astype(np.float64)
    w[np.isnan(w)] = 0.0
    return w.reshape(orig_shape)


def _newton_w(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Proximal for general alpha via Newton-Raphson (4 iterations)."""
    x = v.copy().astype(np.float64)
    for _ in range(4):
        fd = alpha * np.sign(x) * np.abs(x) ** (alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.abs(x) ** (alpha - 2) + beta
        with np.errstate(divide='ignore', invalid='ignore'):
            x = x - fd / fdd
    x[np.isnan(x)] = 0.0
    return x


def _compute_w(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Dispatch to the correct proximal solver based on *alpha*."""
    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)


def solve_image_bregman(v: np.ndarray, beta: float,
                        alpha: float) -> np.ndarray:
    """
    Solve the component-wise proximal problem

        min_w  |w|^alpha  +  (beta/2) (w - v)^2

    using a pre-computed Look-Up Table (LUT), exactly as in MATLAB
    ``solve_image_bregman.m`` (Krishnan & Fergus, NIPS 2009).

    Parameters
    ----------
    v     : 2-D array — input values
    beta  : float     — quadratic penalty weight
    alpha : float     — sparsity exponent (0.5, 2/3, 1, or general)

    Returns
    -------
    w : 2-D array — proximal solution, same shape as *v*
    """
    global _lut_cache

    key = (beta, alpha)
    lut_range = 10.0
    lut_step = 0.0001

    if key not in _lut_cache:
        xx = np.arange(-lut_range, lut_range + lut_step * 0.5, lut_step)
        lookup = _compute_w(xx.copy(), beta, alpha)
        _lut_cache[key] = (xx, lookup.astype(np.float64))

    xx, lookup = _lut_cache[key]

    # MATLAB: interp1(xx', lookup', v(:), 'linear', 'extrap')
    # np.interp clamps to edge values for out-of-range queries — close
    # enough for the range [-10, 10] which covers practical gradient values.
    w = np.interp(v.ravel(), xx, lookup)
    return w.reshape(v.shape)
