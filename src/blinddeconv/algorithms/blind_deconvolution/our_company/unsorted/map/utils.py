"""
utils.py

Utility functions for the MAP (Maximum A Posteriori) blind deconvolution.

Ported from MATLAB code by Oliver Whyte et al.
References:
    O. Whyte, J. Sivic, A. Zisserman, and J. Ponce.
    "Non-uniform Deblurring for Shaken Images". IJCV, 2012.

    O. Whyte, J. Sivic and A. Zisserman.
    "Deblurring Shaken and Partially Saturated Images".
    In Proc. CPCV Workshop at ICCV, 2011.

    D. Krishnan, R. Fergus.
    "Fast Image Deconvolution using Hyper-Laplacian Priors". NIPS, 2009.

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'same'):
        MATLAB conv2 performs TRUE convolution (flips B).
        -> scipy.signal.convolve2d(A, B, mode='same', boundary='fill')

    imfilter(I, h, 'conv', 'replicate'):
        Convolution with replicated boundary.
        -> scipy.ndimage.convolve(I, h, mode='nearest')

    padarray(I, [p q], 'replicate', 'pre'/'post'):
        -> np.pad(I, ..., mode='edge')

    fft2/ifft2:
        -> numpy.fft.fft2/ifft2 (same semantics)

    psf2otf(psf, shape):
        Zero-pad, circshift center to (0,0), fft2.
        -> Custom implementation.

    MATLAB dst/idst:
        MATLAB dst is DST Type-I.
        -> scipy.fft.dstn/idstn with type=1.

    imresize(I, scale, 'bilinear'):
        -> cv2.resize or scipy.ndimage.zoom.

    MATLAB indexing is 1-based; Python is 0-based.
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import convolve as ndimage_convolve
from scipy.ndimage import binary_erosion, zoom
from scipy.fft import dstn, idstn
from scipy.linalg import expm
from scipy.interpolate import RegularGridInterpolator


# ═══════════════════════════════════════════════════════════════════════════
# PSF <-> OTF conversions
# ═══════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.

    MATLAB psf2otf circshift amounts: -floor(size(psf)/2) for each dim.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    psf = np.atleast_2d(psf)
    in_shape = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_shape[0], :in_shape[1]] = psf

    # Circular shift: move PSF centre to (0, 0)
    for ax in range(len(in_shape)):
        padded = np.roll(padded, -(in_shape[ax] // 2), axis=ax)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).
    """
    full = np.real(np.fft.ifft2(otf))
    for ax in range(2):
        full = np.roll(full, psf_size[ax] // 2, axis=ax)
    return full[:psf_size[0], :psf_size[1]]


# ═══════════════════════════════════════════════════════════════════════════
# Padding utilities  (from padImage.m, calculatePadding.m)
# ═══════════════════════════════════════════════════════════════════════════

def pad_image(im: np.ndarray, padsize: np.ndarray, padval=0) -> np.ndarray:
    """
    Pad or crop image.  Equivalent to MATLAB padImage.m.

    Parameters
    ----------
    im : ndarray, 2D or 3D (H, W) or (H, W, C)
    padsize : array-like of 4 ints [top, bottom, left, right]
        Negative values crop instead of padding.
    padval : 'edge' for replicate, or numeric value (default 0)

    Returns
    -------
    Padded or cropped image.
    """
    padsize = np.asarray(padsize, dtype=int)
    if np.any(padsize < 0):
        # Negative padsize -> crop (undo padding)
        padsize = -padsize
        top, bottom, left, right = padsize
        h, w = im.shape[0], im.shape[1]
        if im.ndim == 3:
            return im[top:h - bottom, left:w - right, :]
        return im[top:h - bottom, left:w - right]

    top, bottom, left, right = padsize
    if isinstance(padval, str) and padval in ('edge', 'replicate'):
        mode = 'edge'
    else:
        mode = 'constant'

    if im.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))

    if mode == 'constant':
        return np.pad(im, pad_width, mode='constant',
                      constant_values=padval)
    else:
        return np.pad(im, pad_width, mode='edge')


def calculate_padding(image_size: tuple, kernel: np.ndarray) -> np.ndarray:
    """
    Calculate padding for uniform blur.
    Equivalent to MATLAB calculatePadding.m with non_uniform=0.

    Parameters
    ----------
    image_size : (H, W)
    kernel : 2D kernel array (used only for its shape)

    Returns
    -------
    pad : array [top, bottom, left, right]
    """
    kh, kw = kernel.shape[:2]
    pad_top = int(np.ceil((kh - 1) / 2))
    pad_bottom = int(np.floor((kh - 1) / 2))
    pad_left = int(np.ceil((kw - 1) / 2))
    pad_right = int(np.floor((kw - 1) / 2))
    return np.array([pad_top, pad_bottom, pad_left, pad_right], dtype=int)


# ═══════════════════════════════════════════════════════════════════════════
# Joint / Cross Bilateral Filter  (from jcbfilter.m)
# ═══════════════════════════════════════════════════════════════════════════

def jcb_filter(Iin: np.ndarray,
               Iguide: np.ndarray,
               sigma_spatial: float = 1.6,
               sigma_range: float = 0.08,
               window_width: int = None) -> np.ndarray:
    """
    Joint / cross bilateral filter.  Equivalent to MATLAB jcbfilter.m.

    Parameters
    ----------
    Iin : (H, W) or (H, W, C) input image
    Iguide : (H, W) or (H, W, C) guide image
    sigma_spatial : spatial Gaussian sigma
    sigma_range : range (intensity) Gaussian sigma
    window_width : full window width (odd); default ceil(6*sigma_spatial)+1

    Returns
    -------
    Ilow : filtered image, same shape as Iin
    """
    if Iin.ndim == 2:
        Iin = Iin[:, :, np.newaxis]
        squeeze_output = True
    else:
        squeeze_output = False

    if Iguide.ndim == 2:
        Iguide = Iguide[:, :, np.newaxis]

    h, w, channels = Iin.shape

    sigmad2 = 2.0 * sigma_spatial ** 2
    sigmar2 = 2.0 * sigma_range ** 2

    if window_width is None:
        whs = int(np.ceil(3 * sigma_spatial))
    else:
        whs = int(np.ceil((window_width - 1) / 2))

    # Spatial Gaussian kernel
    wy, wx = np.mgrid[-whs:whs + 1, -whs:whs + 1]
    Gd = np.exp(-(wx ** 2 + wy ** 2) / sigmad2)

    Ilow = np.zeros((h, w, channels), dtype=np.float64)

    # Compute signal weights on full colour guide image
    for ir in range(h):
        rows_j_start = max(ir - whs, 0)
        rows_j_end = min(ir + whs, h - 1)
        # Window indices (0-based)
        rows_j = np.arange(rows_j_start, rows_j_end + 1)
        rows_win = rows_j - ir + whs  # indices into Gd

        for ic in range(w):
            cols_j_start = max(ic - whs, 0)
            cols_j_end = min(ic + whs, w - 1)
            cols_j = np.arange(cols_j_start, cols_j_end + 1)
            cols_win = cols_j - ic + whs  # indices into Gd

            # Range weight: squared difference across all channels
            sigdiff2 = np.zeros((len(rows_j), len(cols_j)), dtype=np.float64)
            for c in range(channels):
                sigdiff2 += (Iguide[ir, ic, c] -
                             Iguide[np.ix_(rows_j, cols_j)][..., c]) ** 2

            weights = (Gd[np.ix_(rows_win, cols_win)] *
                       np.exp(-sigdiff2 / (sigmar2 * channels)))
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights /= weights_sum
            for c in range(channels):
                Ilow[ir, ic, c] = np.sum(
                    weights * Iin[np.ix_(rows_j, cols_j)][..., c])

    if squeeze_output:
        return Ilow[:, :, 0]
    return Ilow


# ═══════════════════════════════════════════════════════════════════════════
# Shock Filter  (from shock_filter.m by Guy Gilboa)
# ═══════════════════════════════════════════════════════════════════════════

def shock_filter(I0: np.ndarray,
                 iters: int = 1,
                 dt: float = 0.1,
                 h: float = 1.0) -> np.ndarray:
    """
    Osher-Rudin shock filter (method='org').
    Equivalent to MATLAB shock_filter.m with meth='org'.

    Evolves image according to:
        I_t = -sign(I_nn) * |grad I| / h

    Parameters
    ----------
    I0 : 2D array (H, W)
    iters : number of iterations
    dt : time step size
    h : grid step size

    Returns
    -------
    I : filtered image, same shape as I0
    """
    ny, nx = I0.shape
    I = I0.copy().astype(np.float64)

    for _ in range(iters):
        # Neumann boundary conditions via replication indexing
        # MATLAB: I(:,[1 1:nx-1]) -> column 0 repeated, then cols 0..nx-2
        I_left = np.empty_like(I)
        I_left[:, 0] = I[:, 0]
        I_left[:, 1:] = I[:, :-1]

        I_right = np.empty_like(I)
        I_right[:, -1] = I[:, -1]
        I_right[:, :-1] = I[:, 1:]

        I_up = np.empty_like(I)
        I_up[0, :] = I[0, :]
        I_up[1:, :] = I[:-1, :]

        I_down = np.empty_like(I)
        I_down[-1, :] = I[-1, :]
        I_down[:-1, :] = I[1:, :]

        # Left/right/up/down differences
        I_mx = I - I_left      # backward difference x
        I_px = I_right - I      # forward difference x
        I_my = I - I_up          # backward difference y
        I_py = I_down - I        # forward difference y

        # Central differences
        I_x = (I_mx + I_px) / 2.0
        I_y = (I_my + I_py) / 2.0

        # Minmod operator
        Dx = np.minimum(np.abs(I_mx), np.abs(I_px))
        Dx[I_mx * I_px < 0] = 0.0
        Dy = np.minimum(np.abs(I_my), np.abs(I_py))
        Dy[I_my * I_py < 0] = 0.0

        # Second derivatives
        I_xx = I_right + I_left - 2.0 * I
        I_yy = I_down + I_up - 2.0 * I
        # Mixed derivative
        I_x_down = np.empty_like(I)
        I_x_down[-1, :] = I_x[-1, :]
        I_x_down[:-1, :] = I_x[1:, :]
        I_x_up = np.empty_like(I)
        I_x_up[0, :] = I_x[0, :]
        I_x_up[1:, :] = I_x[:-1, :]
        I_xy = (I_x_down - I_x_up) / 2.0

        # Abs gradient (minmod-based)
        a_grad_I = np.sqrt(Dx ** 2 + Dy ** 2)

        # Second derivative along gradient direction (I_nn)
        dl = 1e-8
        denom = np.abs(I_x) ** 2 + np.abs(I_y) ** 2 + dl
        I_nn = (I_xx * np.abs(I_x) ** 2 +
                2.0 * I_xy * I_x * I_y +
                I_yy * np.abs(I_y) ** 2) / denom

        # Handle zero gradient: fall back to I_xx
        zero_grad = (np.abs(I_x) + np.abs(I_y)) == 0
        I_nn[zero_grad] = I_xx[zero_grad]

        # Osher-Rudin evolution
        I_t = -np.sign(I_nn) * a_grad_I / h
        I = I + dt * I_t

    return I


# ═══════════════════════════════════════════════════════════════════════════
# Poisson Blend via FFT/DST  (from poisson_blend_fft.m)
# ═══════════════════════════════════════════════════════════════════════════

def poisson_blend_fft(gx: np.ndarray,
                      gy: np.ndarray,
                      boundary_image: np.ndarray = None) -> np.ndarray:
    """
    Recover an image from a gradient field with Dirichlet boundary conditions.
    Equivalent to MATLAB poisson_blend_fft.m.

    Uses DST Type-I (matching MATLAB dst/idst).

    Parameters
    ----------
    gx : (H, W) or (H, W, C) x-component of gradient field
    gy : (H, W) or (H, W, C) y-component of gradient field
    boundary_image : (H, W) or (H, W, C) boundary conditions (default: zeros)

    Returns
    -------
    img_direct : reconstructed image
    """
    if gx.ndim == 2:
        gx = gx[:, :, np.newaxis]
        gy = gy[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    if boundary_image is None:
        boundary_image = np.zeros_like(gx)
    elif boundary_image.ndim == 2:
        boundary_image = boundary_image[:, :, np.newaxis]

    H, W, C = boundary_image.shape
    img_direct = boundary_image.copy().astype(np.float64)

    for c in range(C):
        gxx = np.zeros((H, W), dtype=np.float64)
        gyy = np.zeros((H, W), dtype=np.float64)

        # Laplacian: gyy(j+1,k) = gy(j+1,k) - gy(j,k)  [MATLAB 1-based]
        # In Python (0-based): gyy[1:H, :W-1] = gy[1:H, :W-1, c] - gy[:H-1, :W-1, c]
        gyy[1:H, :W - 1] = gy[1:H, :W - 1, c] - gy[:H - 1, :W - 1, c]
        gxx[:H - 1, 1:W] = gx[:H - 1, 1:W, c] - gx[:H - 1, :W - 1, c]

        f = gxx + gyy

        # Boundary image: zero out interior, keep boundary
        bi = boundary_image[:, :, c].copy()
        bi[1:H - 1, 1:W - 1] = 0.0

        # Laplacian of boundary image
        f_bp = np.zeros((H, W), dtype=np.float64)
        j = slice(1, H - 1)
        k = slice(1, W - 1)
        f_bp[j, k] = (-4.0 * bi[j, k] +
                       bi[j, 2:W] + bi[j, 0:W - 2] +
                       bi[0:H - 2, k] + bi[2:H, k])

        f1 = f - f_bp

        # Interior only
        f2 = f1[1:H - 1, 1:W - 1]

        # DST Type-I
        f2sin = dstn(f2, type=1)

        # Eigenvalues of discrete Laplacian under DST-I basis
        # MATLAB: meshgrid(1:W-2, 1:H-2)
        x = np.arange(1, W - 1)
        y = np.arange(1, H - 1)
        xx, yy = np.meshgrid(x, y)
        denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0 +
                 2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

        f3 = f2sin / denom

        # Inverse DST Type-I
        img_tt = idstn(f3, type=1)

        # Put solution in inner points
        img_direct[1:H - 1, 1:W - 1, c] = img_tt

    if squeeze:
        return img_direct[:, :, 0]
    return img_direct


# ═══════════════════════════════════════════════════════════════════════════
# Gamma correction  (from linear2srgb.m, srgb2linear.m)
# ═══════════════════════════════════════════════════════════════════════════

def linear2srgb(imlinear: np.ndarray) -> np.ndarray:
    """Convert linear tristimulus values to sRGB.  Matches MATLAB linear2srgb."""
    thresh = 0.0031308
    slope = 12.92
    a = 0.055
    gamma = 2.4
    lo = imlinear * slope
    hi = (1.0 + a) * np.power(np.maximum(imlinear, 0), 1.0 / gamma) - a
    return np.where(imlinear <= thresh, lo, hi)


def srgb2linear(imsrgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear tristimulus values.  Matches MATLAB srgb2linear."""
    thresh = 0.0031308
    slope = 12.92
    a = 0.055
    gamma = 2.4
    srgb_thresh = slope * thresh
    lo = imsrgb / slope
    hi = np.power(np.maximum(imsrgb + a, 0) / (1.0 + a), gamma)
    return np.where(imsrgb <= srgb_thresh, lo, hi)


# ═══════════════════════════════════════════════════════════════════════════
# Homogeneous coordinates  (from crossmatrix.m, htranslate.m, hscale.m,
#                            hnormalise.m)
# ═══════════════════════════════════════════════════════════════════════════

def crossmatrix(v: np.ndarray) -> np.ndarray:
    """Antisymmetric (cross-product) matrix.  Matches MATLAB crossmatrix.m."""
    v = np.asarray(v).ravel()
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]], dtype=np.float64)


def htranslate(tt: np.ndarray) -> np.ndarray:
    """
    Translation matrix for homogeneous coordinates.
    Matches MATLAB htranslate.m.

    For input tt with n elements, returns (n+1) x (n+1) matrix.
    """
    tt = np.asarray(tt).ravel()
    n = len(tt)
    T = np.eye(n + 1, dtype=np.float64)
    T[:n, -1] = tt
    return T


def hscale(ss: np.ndarray) -> np.ndarray:
    """
    Scaling matrix for homogeneous coordinates.
    Matches MATLAB hscale.m.

    For input ss with n elements, returns (n+1) x (n+1) matrix.
    """
    ss = np.asarray(ss).ravel()
    n = len(ss)
    S = np.eye(n + 1, dtype=np.float64)
    for i in range(n):
        S[i, i] = ss[i]
    return S


def hnormalise(X: np.ndarray) -> np.ndarray:
    """
    Normalise homogeneous coordinates so last row is 1.
    Matches MATLAB hnormalise.m.

    Parameters
    ----------
    X : (ndims, npts) array

    Returns
    -------
    Xn : normalised array (last row all ones)
    """
    X = X.copy().astype(np.float64)
    for i in range(X.shape[0] - 1):
        X[i, :] = X[i, :] / X[-1, :]
    X[-1, :] = 1.0
    return X


# ═══════════════════════════════════════════════════════════════════════════
# Solve w-subproblem / LUT  (from solve_image.m, Krishnan & Fergus 2009)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level cache (like MATLAB persistent variables)
_solve_image_cache = {
    'lookup_v': [],
    'known_beta': [],
    'known_alpha': [],
    'xx': None,
}


def _compute_w1(v: np.ndarray, beta: float) -> np.ndarray:
    """Soft thresholding for alpha=1."""
    return np.maximum(np.abs(v) - 1.0 / beta, 0) * np.sign(v)


def _compute_w23(v: np.ndarray, beta: float) -> np.ndarray:
    """Ferrari's quartic root for alpha=2/3."""
    epsilon = 1e-6

    k_val = 8.0 / (27.0 * beta ** 3)
    m = np.full_like(v, k_val)

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    m2 = m * m
    m3 = m2 * m

    alpha_q = -1.125 * v2
    beta2 = 0.25 * v3

    q = -0.125 * (m * v2)
    # Handle complex arithmetic
    inner = -m3 / 27.0 + (m2 * v4) / 256.0
    r1 = -q / 2.0 + np.sqrt(inner.astype(np.complex128))

    u = np.exp(np.log(r1) / 3.0)
    y = 2.0 * (-5.0 / 18.0 * alpha_q + u + (m.astype(np.complex128) / (3.0 * u)))

    W_sq = alpha_q / 3.0 + y
    W = np.sqrt(W_sq.astype(np.complex128))

    root = np.zeros((*v.shape, 4), dtype=np.complex128)
    inner1 = np.sqrt(-(alpha_q + y + beta2 / W).astype(np.complex128))
    inner2 = np.sqrt(-(alpha_q + y - beta2 / W).astype(np.complex128))
    root[..., 0] = 0.75 * v + 0.5 * (W + inner1)
    root[..., 1] = 0.75 * v + 0.5 * (W - inner1)
    root[..., 2] = 0.75 * v + 0.5 * (-W + inner2)
    root[..., 3] = 0.75 * v + 0.5 * (-W - inner2)

    # Pick correct root
    v_rep = np.repeat(v[..., np.newaxis], 4, axis=-1)
    sv = np.sign(v_rep)
    rsv = np.real(root) * sv

    # Valid: real, between |v|/2 and |v|
    valid = ((np.abs(np.imag(root)) < epsilon) &
             (rsv > np.abs(v_rep) / 2.0) &
             (rsv < np.abs(v_rep)))
    # Set invalid roots to -inf for sorting
    scores = np.where(valid, rsv, -np.inf)
    # Take the best (highest score)
    best_idx = np.argmax(scores, axis=-1)
    # Gather the best root (real part, correct sign)
    w = np.take_along_axis(
        np.real(root) * sv, best_idx[..., np.newaxis], axis=-1
    )[..., 0] * np.sign(v)

    # Where no valid root was found, return 0
    w[np.all(~valid, axis=-1)] = 0.0
    return w


def _compute_w12(v: np.ndarray, beta: float) -> np.ndarray:
    """Cubic root for alpha=1/2."""
    epsilon = 1e-6

    k_val = -0.25 / beta ** 2
    m = np.full_like(v, k_val, dtype=np.complex128) * np.sign(v)

    t1 = (2.0 / 3.0) * v

    v2 = v.astype(np.complex128) * v
    v3 = v2 * v

    inner = 27.0 * m ** 2 + 4.0 * m * v3
    sqrt_inner = np.sqrt(3.0) * np.sqrt(inner)
    t2 = np.exp(np.log(-27.0 * m - 2.0 * v3 + sqrt_inner) / 3.0)

    # Avoid division by zero
    t2_safe = np.where(np.abs(t2) < 1e-30, 1e-30, t2)
    t3 = v2 / t2_safe

    root = np.zeros((*v.shape, 3), dtype=np.complex128)
    c1 = 2.0 ** (1.0 / 3.0)
    c2 = 1j * np.sqrt(3.0)

    root[..., 0] = t1 + (c1 / 3.0) * t3 + (t2 / (3.0 * c1))
    root[..., 1] = t1 - ((1.0 + c2) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3 - \
        ((1.0 - c2) / (6.0 * c1)) * t2
    root[..., 2] = t1 - ((1.0 - c2) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3 - \
        ((1.0 + c2) / (6.0 * c1)) * t2

    # Replace NaN/Inf with 0
    root[np.isnan(root) | np.isinf(root)] = 0.0

    # Pick correct root
    v_rep = np.repeat(v[..., np.newaxis], 3, axis=-1)
    sv = np.sign(v_rep)
    rsv = np.real(root) * sv

    valid = ((np.abs(np.imag(root)) < epsilon) &
             (rsv > 2.0 * np.abs(v_rep) / 3.0) &
             (rsv < np.abs(v_rep)))
    scores = np.where(valid, rsv, -np.inf)
    best_idx = np.argmax(scores, axis=-1)
    w = np.take_along_axis(
        np.real(root) * sv, best_idx[..., np.newaxis], axis=-1
    )[..., 0] * np.sign(v)

    w[np.all(~valid, axis=-1)] = 0.0
    return w


def _newton_w(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Newton-Raphson for general alpha."""
    x = v.copy().astype(np.float64)

    for _ in range(4):
        abs_x = np.abs(x)
        fd = alpha * np.sign(x) * np.power(abs_x, alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.power(abs_x, alpha - 2) + beta
        # Avoid division by zero
        fdd = np.where(np.abs(fdd) < 1e-30, 1e-30, fdd)
        x = x - fd / fdd

    x[np.isnan(x)] = 0.0

    # Check whether zero is a better solution
    z = beta / 2.0 * v ** 2
    f = np.power(np.abs(x), alpha) + beta / 2.0 * (x - v) ** 2
    w = np.where(f < z, x, 0.0)
    return w


def _compute_w_dispatch(v: np.ndarray,
                        beta: float,
                        alpha: float) -> np.ndarray:
    """Dispatch to specific solver based on alpha value."""
    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    elif abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    elif abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    else:
        return _newton_w(v, beta, alpha)


def solve_image(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """
    Solve the w-subproblem:
        min |w|^alpha + (beta/2) * (w - v)^2

    Uses a LUT (lookup table) for efficiency.
    Equivalent to MATLAB solve_image.m (Krishnan & Fergus 2009).

    Parameters
    ----------
    v : ndarray — input values
    beta : regularization weight
    alpha : sparsity exponent (0 < alpha <= 1)

    Returns
    -------
    w : ndarray — optimal w values, same shape as v
    """
    cache = _solve_image_cache
    lut_range = 10.0
    step = 0.0001

    # Check if LUT already exists for this (beta, alpha)
    found = False
    for i, (kb, ka) in enumerate(zip(cache['known_beta'],
                                      cache['known_alpha'])):
        if kb == beta and ka == alpha:
            found = True
            lut = cache['lookup_v'][i]
            break

    if cache['xx'] is None:
        cache['xx'] = np.arange(-lut_range, lut_range + step / 2, step)

    if not found:
        # Compute new LUT
        lut = _compute_w_dispatch(cache['xx'], beta, alpha)
        cache['lookup_v'].append(lut)
        cache['known_beta'].append(beta)
        cache['known_alpha'].append(alpha)

    # Interpolate from LUT
    original_shape = v.shape
    v_flat = v.ravel()
    w = np.interp(v_flat, cache['xx'], lut)
    return w.reshape(original_shape)


def reset_solve_image_cache():
    """Reset the LUT cache for solve_image."""
    _solve_image_cache['lookup_v'] = []
    _solve_image_cache['known_beta'] = []
    _solve_image_cache['known_alpha'] = []
    _solve_image_cache['xx'] = None


# ═══════════════════════════════════════════════════════════════════════════
# Kernel pyramid  (from make_kernel_pyramid.m — uniform path only)
# ═══════════════════════════════════════════════════════════════════════════

def make_kernel_pyramid(blur_x_lims: np.ndarray,
                        blur_y_lims: np.ndarray,
                        scale_ratio_k: float,
                        max_levels: int,
                        init_kernel: np.ndarray = None
                        ) -> tuple:
    """
    Build a multi-scale pyramid of kernels for uniform blur.
    Simplified from MATLAB make_kernel_pyramid.m (non_uniform=false).

    For uniform blur, tgs=[1,1,1], theta_z_lims=[0,0], and the kernel is
    a standard 2D PSF whose elements correspond to pixel offsets.

    Parameters
    ----------
    blur_x_lims : [x_min, x_max] offset range (pixels, symmetric e.g. [-k, k])
    blur_y_lims : [y_min, y_max] offset range (pixels, symmetric e.g. [-k, k])
    scale_ratio_k : downscale ratio (e.g. 1/sqrt(2))
    max_levels : maximum number of pyramid levels
    init_kernel : optional initial kernel at finest scale

    Returns
    -------
    pyr_kernel : list of 2D kernel arrays (level 0 = finest)
    pyr_tt : list of dicts with 'tty', 'ttx' meshgrid arrays
    pyr_tgs : list of [tgs_y, tgs_x] per level
    """
    # For uniform blur, grid spacing is 1 pixel
    tgs = np.array([1.0, 1.0])

    # Half-sizes at finest level
    khh = int(np.ceil((blur_y_lims[1] - blur_y_lims[0]) / 2.0 - 0.5))
    khw = int(np.ceil((blur_x_lims[1] - blur_x_lims[0]) / 2.0 - 0.5))
    # Center of the kernel (should be integer for uniform)
    ttc_y = (blur_y_lims[1] + blur_y_lims[0]) / 2.0
    ttc_x = (blur_x_lims[1] + blur_x_lims[0]) / 2.0

    pyr_kernel = []
    pyr_tt = []
    pyr_tgs = []

    actual_levels = max_levels
    for s in range(max_levels):
        scale_factor = scale_ratio_k ** s
        tgs_s = tgs  # For uniform blur, grid spacing stays 1

        # Scale the half-sizes
        khh_s = int(np.ceil(scale_factor * (khh + 0.5) - 0.5))
        khw_s = int(np.ceil(scale_factor * (khw + 0.5) - 0.5))

        if max(khh_s, khw_s) < 2 and s > 0:
            actual_levels = s
            break

        # Build meshgrid of pixel offsets
        # MATLAB: meshgrid(-khw_s:khw_s, -khh_s:khh_s) * tgs_s + ttc
        ttx = np.arange(-khw_s, khw_s + 1, dtype=np.float64) * tgs_s[1] + ttc_x
        tty = np.arange(-khh_s, khh_s + 1, dtype=np.float64) * tgs_s[0] + ttc_y
        ttx_grid, tty_grid = np.meshgrid(ttx, tty)

        kernel_shape = (2 * khh_s + 1, 2 * khw_s + 1)
        kernel = np.zeros(kernel_shape, dtype=np.float64)

        if init_kernel is not None and s == 0:
            # Use provided kernel at finest scale
            if isinstance(init_kernel, str) and init_kernel == 'delta':
                kernel[khh_s, khw_s] = 1.0
            else:
                # Crop or pad init_kernel to match kernel_shape
                ikh, ikw = init_kernel.shape
                kh, kw = kernel_shape
                # Centers
                cy_src, cx_src = ikh // 2, ikw // 2
                cy_dst, cx_dst = kh // 2, kw // 2
                # Compute overlapping region
                r_start_src = max(0, cy_src - cy_dst)
                r_start_dst = max(0, cy_dst - cy_src)
                r_end_src = min(ikh, cy_src + (kh - cy_dst))
                r_end_dst = min(kh, cy_dst + (ikh - cy_src))
                c_start_src = max(0, cx_src - cx_dst)
                c_start_dst = max(0, cx_dst - cx_src)
                c_end_src = min(ikw, cx_src + (kw - cx_dst))
                c_end_dst = min(kw, cx_dst + (ikw - cx_src))
                kernel[r_start_dst:r_end_dst,
                       c_start_dst:c_end_dst] = init_kernel[
                           r_start_src:r_end_src, c_start_src:c_end_src]
        elif init_kernel is not None and s > 0:
            # Downsample from previous level via bilinear interpolation
            prev_kernel = pyr_kernel[s - 1]
            prev_tt = pyr_tt[s - 1]
            kernel = _downsample_kernel_uniform(
                prev_kernel, prev_tt['tty'], prev_tt['ttx'],
                tty_grid, ttx_grid)

        pyr_kernel.append(kernel)
        pyr_tt.append({'tty': tty_grid, 'ttx': ttx_grid})
        pyr_tgs.append(tgs_s.copy())

    # Trim to actual number of levels
    pyr_kernel = pyr_kernel[:actual_levels]
    pyr_tt = pyr_tt[:actual_levels]
    pyr_tgs = pyr_tgs[:actual_levels]

    return pyr_kernel, pyr_tt, pyr_tgs


def _downsample_kernel_uniform(kernel: np.ndarray,
                               tty_src: np.ndarray,
                               ttx_src: np.ndarray,
                               tty_dst: np.ndarray,
                               ttx_dst: np.ndarray) -> np.ndarray:
    """Downsample a uniform 2D kernel using bilinear interpolation."""
    # Source grid (must be 1D and monotonically increasing)
    y_src = tty_src[:, 0]  # column of y-values
    x_src = ttx_src[0, :]  # row of x-values

    interp_fn = RegularGridInterpolator(
        (y_src, x_src), kernel,
        method='linear', bounds_error=False, fill_value=0.0)

    # Destination query points
    pts = np.column_stack([tty_dst.ravel(), ttx_dst.ravel()])
    result = interp_fn(pts).reshape(tty_dst.shape)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Upsample kernel  (from upsample_kernel_map.m — uniform path)
# ═══════════════════════════════════════════════════════════════════════════

def upsample_kernel_map(kernel: np.ndarray,
                        tt_src: dict,
                        tt_dst: dict,
                        scale_ratio_k: float = 1.0) -> np.ndarray:
    """
    Upsample a 2D kernel from coarser to finer scale via bilinear interpolation.
    Simplified from MATLAB upsample_kernel_map.m for uniform (2D) blur.

    Parameters
    ----------
    kernel : 2D array — kernel at coarse scale
    tt_src : dict with 'tty', 'ttx' — coordinate grids at coarse scale
    tt_dst : dict with 'tty', 'ttx' — coordinate grids at fine scale
    scale_ratio_k : ratio to multiply tt_dst coordinates by
        (for uniform blur, used to align grids between scales)

    Returns
    -------
    kernel_up : 2D array — upsampled kernel at fine scale
    """
    tty_src = tt_src['tty']
    ttx_src = tt_src['ttx']
    tty_dst = tt_dst['tty'] * scale_ratio_k
    ttx_dst = tt_dst['ttx'] * scale_ratio_k

    y_src = tty_src[:, 0]
    x_src = ttx_src[0, :]

    interp_fn = RegularGridInterpolator(
        (y_src, x_src), kernel,
        method='linear', bounds_error=False, fill_value=0.0)

    pts = np.column_stack([tty_dst.ravel(), ttx_dst.ravel()])
    kernel_up = interp_fn(pts).reshape(tty_dst.shape)
    return kernel_up


# ═══════════════════════════════════════════════════════════════════════════
# imfilter_conv_replicate — MATLAB imfilter(...,'conv','replicate')
# ═══════════════════════════════════════════════════════════════════════════

def imfilter_conv_replicate(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Replicate MATLAB's imfilter(im, h, 'conv', 'replicate').

    MATLAB imfilter with 'conv' flips the kernel then does correlation.
    This is standard convolution with 'replicate' (nearest) boundary.

    scipy.ndimage.convolve does convolution (with kernel flip) and
    mode='nearest' corresponds to 'replicate'.

    Parameters
    ----------
    im : 2D or 3D array
    kernel : 2D kernel

    Returns
    -------
    result : same shape as im
    """
    if im.ndim == 3:
        result = np.empty_like(im)
        for c in range(im.shape[2]):
            result[:, :, c] = ndimage_convolve(
                im[:, :, c], kernel, mode='nearest')
        return result
    return ndimage_convolve(im, kernel, mode='nearest')


# ═══════════════════════════════════════════════════════════════════════════
# Colour conv2 helper — conv2 per channel
# ═══════════════════════════════════════════════════════════════════════════

def colour_conv2(im: np.ndarray, h: np.ndarray,
                 mode: str = 'same') -> np.ndarray:
    """
    Apply conv2(..., mode) to each channel of a multi-channel image.
    Equivalent to MATLAB's per-channel conv2 pattern.

    Parameters
    ----------
    im : (H, W) or (H, W, C)
    h : 2D kernel
    mode : 'same', 'valid', 'full'

    Returns
    -------
    result : filtered image
    """
    if im.ndim == 2:
        return convolve2d(im, h, mode=mode, boundary='fill', fillvalue=0)
    channels = im.shape[2]
    out_list = []
    for c in range(channels):
        out_list.append(
            convolve2d(im[:, :, c], h, mode=mode, boundary='fill', fillvalue=0))
    return np.stack(out_list, axis=2)


# ═══════════════════════════════════════════════════════════════════════════
# Image resizing (equivalent to MATLAB imresize)
# ═══════════════════════════════════════════════════════════════════════════

def imresize(im: np.ndarray, scale_or_size, method: str = 'bilinear') -> np.ndarray:
    """
    Resize image.  Approximates MATLAB imresize.

    Parameters
    ----------
    im : 2D or 3D image
    scale_or_size : float (scale factor) or (H_out, W_out) tuple
    method : 'bilinear' or 'bicubic'

    Returns
    -------
    Resized image.
    """
    if isinstance(scale_or_size, (int, float)):
        scale = float(scale_or_size)
        h_out = max(1, int(round(im.shape[0] * scale)))
        w_out = max(1, int(round(im.shape[1] * scale)))
    else:
        h_out, w_out = int(scale_or_size[0]), int(scale_or_size[1])

    # Compute zoom factors
    zoom_h = h_out / im.shape[0]
    zoom_w = w_out / im.shape[1]

    order = 1 if method == 'bilinear' else 3  # 1=bilinear, 3=bicubic

    if im.ndim == 3:
        # Zoom spatial dims only, not channels
        return zoom(im, (zoom_h, zoom_w, 1), order=order)
    return zoom(im, (zoom_h, zoom_w), order=order)


# ═══════════════════════════════════════════════════════════════════════════
# Derivative filters  (defined in blind_deblur_map.m)
# ═══════════════════════════════════════════════════════════════════════════

def get_derivative_filters() -> dict:
    """
    Return the derivative filter kernels used throughout the algorithm.
    Matches MATLAB definitions in blind_deblur_map.m:
        kx = [0, 1,-1, 0, 0];  ky = kx';
        kxx = conv2(kx,kx,'same'); kyy=kxx'; kxy=conv2(kx,ky,'full');
        kxt = rot90(rot90(kx));   kyt = kxt';
        kxxt = rot90(rot90(kxx)); kxyt = rot90(rot90(kxy)); kyyt = kxxt';
    """
    kx = np.array([[0, 1, -1, 0, 0]], dtype=np.float64)
    ky = kx.T

    kxx = convolve2d(kx, kx, mode='same')
    kyy = kxx.T
    kxy = convolve2d(kx, ky, mode='full')

    # rot90 twice = 180-degree rotation = flipud(fliplr(...))
    kxt = np.rot90(kx, 2)
    kyt = kxt.T
    kxxt = np.rot90(kxx, 2)
    kxyt = np.rot90(kxy, 2)
    kyyt = kxxt.T

    return {
        'kx': kx, 'ky': ky,
        'kxx': kxx, 'kyy': kyy, 'kxy': kxy,
        'kxt': kxt, 'kyt': kyt,
        'kxxt': kxxt, 'kxyt': kxyt, 'kyyt': kyyt,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Default configuration  (from default_config.m)
# ═══════════════════════════════════════════════════════════════════════════

def default_config(blur_kernel_size: int = 17) -> dict:
    """
    Return default parameters for the MAP blind deconvolution algorithm.
    Equivalent to MATLAB default_config.m (uniform blur case).

    Parameters
    ----------
    blur_kernel_size : initial kernel size in pixels (odd integer)

    Returns
    -------
    cfg : dict of all algorithm parameters
    """
    half_k = (blur_kernel_size - 1) // 2

    cfg = {
        # Kernel size
        'blur_kernel_size': blur_kernel_size,
        'blur_x_lims': np.array([-half_k, half_k]),
        'blur_y_lims': np.array([-half_k, half_k]),

        # Multiscale parameters
        'scale_ratio_i': 1.0 / np.sqrt(2.0),
        'scale_ratio_k': 1.0 / np.sqrt(2.0),
        'max_levels': 9,
        'first_level': -1,     # -1 means start from coarsest
        'final_level': 1,      # 1 means end at finest

        # Number of iterations at each scale
        'num_iters': 5,

        # Bilateral filter parameters
        'bi_sigma_spatial0': 2.0 / np.sqrt(2.0),
        'bi_sigma_range0': 0.5,
        'bi_size': 5,

        # Shock filter parameters
        'shock_dt0': 1.0,
        'shock_iters': 1,

        # Parameter decrease rate (Cho & Lee)
        'param_decrease': 0.9,

        # Gradient thresholding parameters
        'grad_dir_bins': 4,
        'grad_thresh_decrease': 0.9,
        'r': 2,  # factor to multiply necessary number of retained gradients

        # Shan et al. 2008 gradient data term weights
        'omega0': 1.0,
        'omega1': 0.5,
        'omega2': 0.25,

        # Regularization weights
        'alpha': 0.0005,        # latent image regularization
        'kf_lambda': 8e3,       # Krishnan & Fergus lambda
        'kf_exponent': 0.5,     # Krishnan & Fergus sparse gradient exponent

        # Kernel estimation parameters
        'kernel_threshold': 20,
        'beta': 5.0,            # kernel regularization weight
        'num_cg_iters': 5,

        # Kernel processing
        'recenter_kernel': True,
        'kernel_dilate_radius': 1,
        'threshold_kernel': True,

        # Saturation threshold
        'sat_thresh': 235.5 / 256.0,

        # Methods
        'kernel_method': 'lars',     # 'lars', 'lars_ols', 'conjgrad'
        'image_method': 'conjgrad',  # 'conjgrad', 'krishnan', 'sparse'

        # Estimation flags
        'do_estimate_kernel': True,
        'estimate_kernel_from': 'blind',

        # Non-blind deconvolution parameters
        'deconv_maxit': 20,

        # Max image dimension at finest scale
        'max_dim': 1024,

        # Gamma correction
        'israw': False,
    }
    return cfg
