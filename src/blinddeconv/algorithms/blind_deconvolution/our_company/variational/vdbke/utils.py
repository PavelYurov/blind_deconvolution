"""
utils.py

Utility functions for VDBKE (Variational Dirichlet Blur Kernel Estimation)
blind deconvolution.

Ported from MATLAB code by X. Zhou et al.
Reference:
    X. Zhou, J. Mateos, F. Zhou, R. Molina, A.K. Katsaggelos:
    "Variational Dirichlet Blur Kernel Estimation",
    IEEE TIP, vol. 24, no. 12, pp. 5127-5139, 2015.

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    psf2otf(psf, shape):
        MATLAB: zero-pad, circshift by -floor(size/2), fft2.
        Python: identical logic via np.roll.

    conv2(A, B, mode):
        MATLAB conv2 and scipy.signal.convolve2d both perform TRUE
        convolution (kernel flipped).  Output sizes for 'valid', 'same',
        'full' match between MATLAB and scipy.

    fspecial('gaussian', hsize, sigma):
        Custom implementation below.  MATLAB generates on a centred
        meshgrid and normalises to sum=1.

    psi(x)      [digamma]  -> scipy.special.digamma(x)
    psi(1,x)    [trigamma]  -> scipy.special.polygamma(1, x)
    gammaln(x)              -> scipy.special.gammaln(x)

    diff(x, 1, 1)  -> np.diff(x, n=1, axis=0)   [MATLAB dim 1 = rows]
    diff(x, 1, 2)  -> np.diff(x, n=1, axis=1)   [MATLAB dim 2 = cols]

    padarray(x, [r c], val, 'both'):
        -> np.pad(x, ((r,r),(c,c)), mode='constant', constant_values=val)
    padarray(x, [r c], 'replicate', 'both'):
        -> np.pad(x, ((r,r),(c,c)), mode='edge')
    padarray(x, [r c], 'circular', 'both'):
        -> np.pad(x, ((r,r),(c,c)), mode='wrap')
    padarray(x, [r c], 0, 'post'):
        -> np.pad(x, ((0,r),(0,c)), mode='constant', constant_values=0)

    rot90(x, 2)  -> np.rot90(x, 2)   [180-degree rotation]

    imresize(x, [r c], 'bilinear'):
        scipy.ndimage.zoom with order=1.
        Note: MATLAB applies antialiasing when downsampling; scipy does not.
        For this algorithm the difference is negligible.

    rgb2gray, rgb2ycbcr, ycbcr2rgb:
        Custom implementations matching MATLAB's exact coefficients.

    interp2(X, Y, V, Xq, Yq):
        -> scipy.ndimage.map_coordinates(V, coords, order=1)
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, map_coordinates


# ═════════════════════════════════════════════════════════════════════════════
# psf2otf — PSF to OTF conversion
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert point-spread function to optical transfer function.
    Equivalent to MATLAB ``psf2otf(psf, shape)``.

    Steps:
        1. Zero-pad *psf* into an array of *shape*.
        2. Circularly shift so that the centre of the PSF lands at (0, 0).
           Shift amounts: ``-floor(size(psf, dim) / 2)`` per dimension.
        3. Return ``fft2`` of the result.

    Parameters
    ----------
    psf   : 2-D ndarray — point-spread function.
    shape : tuple (H, W) — desired output size.

    Returns
    -------
    otf : 2-D complex ndarray of *shape*.
    """
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # Circular shift: MATLAB circshift(psf, -floor(size/2), dim)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)


# ═════════════════════════════════════════════════════════════════════════════
# fspecial_gaussian — Gaussian filter
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_gaussian(hsize, sigma: float) -> np.ndarray:
    """
    Create a rotationally symmetric Gaussian lowpass filter.
    Equivalent to MATLAB ``fspecial('gaussian', hsize, sigma)``.

    Parameters
    ----------
    hsize : int or (int, int)
        Size of the filter.  Scalar → square filter.
    sigma : float
        Standard deviation.

    Returns
    -------
    h : 2-D ndarray, sum == 1.
    """
    if np.isscalar(hsize):
        m, n = int(hsize), int(hsize)
    else:
        m, n = int(hsize[0]), int(hsize[1])

    # MATLAB: siz = (hsize-1)/2;  [x,y] = meshgrid(-siz(2):siz(2), -siz(1):siz(1));
    sy = (m - 1) / 2.0
    sx = (n - 1) / 2.0
    y = np.arange(m, dtype=np.float64) - sy          # row coords
    x = np.arange(n, dtype=np.float64) - sx          # col coords
    X, Y = np.meshgrid(x, y)                         # 'xy' indexing (default)

    arg = -(X * X + Y * Y) / (2.0 * sigma * sigma)
    h = np.exp(arg)

    # MATLAB: h(h < eps*max(h(:))) = 0
    h[h < np.finfo(np.float64).eps * h.max()] = 0.0

    s = h.sum()
    if s != 0:
        h /= s
    return h


# ═════════════════════════════════════════════════════════════════════════════
# imresize — Image resizing
# ═════════════════════════════════════════════════════════════════════════════

def imresize(img: np.ndarray, output_size, method: str = 'bilinear') -> np.ndarray:
    """
    Resize an image.
    Approximates MATLAB ``imresize(img, [rows cols], method)``.

    Uses ``scipy.ndimage.zoom`` internally.
    Note: MATLAB applies antialiasing during downsampling by default;
    this implementation does not, but the difference is negligible for
    the multi-scale BID pipeline.

    Parameters
    ----------
    img         : 2-D or 3-D ndarray.
    output_size : (rows, cols) — desired spatial size.
    method      : 'bilinear' (order=1) or 'bicubic' (order=3).

    Returns
    -------
    out : ndarray of shape (rows, cols [, channels]).
    """
    oh, ow = int(output_size[0]), int(output_size[1])
    h, w = img.shape[0], img.shape[1]

    if h == oh and w == ow:
        return img.copy()

    order = 1 if method == 'bilinear' else 3

    zoom_h = oh / h
    zoom_w = ow / w

    if img.ndim == 3:
        result = zoom(img, (zoom_h, zoom_w, 1), order=order)
    else:
        result = zoom(img, (zoom_h, zoom_w), order=order)

    # Guarantee exact output shape (zoom may round differently)
    if result.shape[0] > oh:
        result = result[:oh]
    if result.shape[1] > ow:
        result = result[:, :ow]
    return result


# ═════════════════════════════════════════════════════════════════════════════
# valid_conv_by_fft — Valid convolution using pre-computed FFT
# ═════════════════════════════════════════════════════════════════════════════

def valid_conv_by_fft(X_fft: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Compute valid linear convolution using pre-computed FFT.
    Equivalent to the inner ``valid_conv_by_fft`` in
    ``dirichlet_Adbc_fft.m``.

    MATLAB::

        function y = valid_conv_by_fft(X, h)
        [M1,M2] = size(X);
        [s1,s2] = size(h);
        H = fft2(padarray(h, [M1-s1, M2-s2], 0, 'post'));
        temp = real(ifft2(X .* H));
        y = temp(s1:M1, s2:M2);       % 1-indexed

    Parameters
    ----------
    X_fft : (M1, M2) complex ndarray — FFT of the first operand.
    h     : (s1, s2) real ndarray — second operand (spatial domain).

    Returns
    -------
    y : (M1 - s1 + 1, M2 - s2 + 1) real ndarray — valid convolution.
    """
    M1, M2 = X_fft.shape
    s1, s2 = h.shape

    # Zero-pad h to (M1, M2) — MATLAB: padarray(h, [M1-s1, M2-s2], 0, 'post')
    h_padded = np.zeros((M1, M2), dtype=np.float64)
    h_padded[:s1, :s2] = h
    H = fft2(h_padded)

    temp = np.real(ifft2(X_fft * H))

    # MATLAB 1-indexed: temp(s1:M1, s2:M2) → Python 0-indexed: temp[s1-1:, s2-1:]
    return temp[s1 - 1:, s2 - 1:]


# ═════════════════════════════════════════════════════════════════════════════
# Color-space conversions
# ═════════════════════════════════════════════════════════════════════════════

def rgb2gray(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    Matches MATLAB ``rgb2gray`` for double [0,1] input:
        gray = 0.2989*R + 0.5870*G + 0.1140*B

    Parameters
    ----------
    img : (H, W, 3) float64 ndarray in [0, 1].

    Returns
    -------
    gray : (H, W) float64 ndarray.
    """
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def rgb2ycbcr(img: np.ndarray) -> np.ndarray:
    """
    Convert RGB to YCbCr.
    Matches MATLAB ``rgb2ycbcr`` for double [0,1] input.

    MATLAB formula (double)::

        T = [65.481  128.553  24.966;
            -37.797  -74.203 112.0  ;
            112.0    -93.786 -18.214];
        offset = [16; 128; 128];
        ycbcr = reshape(rgb * T', size) / 255 + offset' / 255;

    Parameters
    ----------
    img : (H, W, 3) float64 ndarray in [0, 1].

    Returns
    -------
    ycbcr : (H, W, 3) float64 ndarray.
    """
    T = np.array([
        [ 65.481, 128.553,  24.966],
        [-37.797, -74.203, 112.0  ],
        [112.0,   -93.786, -18.214]
    ], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)

    H, W = img.shape[:2]
    rgb_flat = img.reshape(-1, 3)                    # (H*W, 3)
    ycbcr_flat = rgb_flat @ T.T / 255.0 + offset / 255.0
    return ycbcr_flat.reshape(H, W, 3)


def ycbcr2rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert YCbCr to RGB.
    Matches MATLAB ``ycbcr2rgb`` for double input.

    Inverse of ``rgb2ycbcr``.

    Parameters
    ----------
    img : (H, W, 3) float64 ndarray (YCbCr).

    Returns
    -------
    rgb : (H, W, 3) float64 ndarray.
    """
    T = np.array([
        [ 65.481, 128.553,  24.966],
        [-37.797, -74.203, 112.0  ],
        [112.0,   -93.786, -18.214]
    ], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)

    invT = np.linalg.inv(T)

    H, W = img.shape[:2]
    ycbcr_flat = img.reshape(-1, 3)                  # (H*W, 3)
    # Inverse: rgb = (ycbcr - offset/255) * 255 * invT'
    rgb_flat = (ycbcr_flat - offset / 255.0) * 255.0 @ invT.T
    return np.clip(rgb_flat.reshape(H, W, 3), 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# comp_upto_shift — Evaluation metric (SSD with sub-pixel alignment)
# ═════════════════════════════════════════════════════════════════════════════

def comp_upto_shift(I1: np.ndarray, I2: np.ndarray):
    """
    Compute sum of squared differences between two images after finding
    the best sub-pixel shift.  Accounts for shift invariance of the
    kernel reconstruction.

    Ported from ``comp_upto_shift.m`` (Anat Levin).

    MATLAB::

        maxshift = 5;
        shifts = [-5:0.25:5];
        I2 = I2(16:end-15, 16:end-15);
        I1 = I1(16-maxshift:end-15+maxshift, 16-maxshift:end-15+maxshift);
        % grid search over sub-pixel shifts, interp2

    Parameters
    ----------
    I1, I2 : (H, W) float64 — images to compare.

    Returns
    -------
    ssde : float — minimum SSD.
    tI1  : (N1, N2) ndarray — I1 shifted to best-align with I2.
    """
    maxshift = 5
    shifts = np.arange(-5.0, 5.0 + 0.25, 0.25)  # MATLAB: -5:0.25:5

    # Crop: MATLAB 1-indexed: I2(16:end-15, 16:end-15)
    # Python 0-indexed: I2[15:-15, 15:-15]
    I2c = I2[15:-15, 15:-15].copy()

    # MATLAB 1-indexed: I1(16-maxshift:end-15+maxshift, ...)
    # = I1(11:end-10, 11:end-10)
    # Python 0-indexed: I1[10:-10, 10:-10]
    I1c = I1[10:-10, 10:-10].copy()

    N1, N2 = I2c.shape
    ns = len(shifts)
    ssdem = np.full((ns, ns), np.inf, dtype=np.float64)

    # Build base query grids (0-indexed into I1c)
    # In MATLAB, I2's pixel [r,c] (1-indexed: r=1..N1, c=1..N2)
    # maps to I1c at [r + maxshift, c + maxshift] (1-indexed)
    # In Python 0-indexed: [r + maxshift, c + maxshift] → same formula
    base_r = np.arange(N1, dtype=np.float64) + maxshift  # (N1,)
    base_c = np.arange(N2, dtype=np.float64) + maxshift  # (N2,)

    for i in range(ns):
        for j in range(ns):
            coords_r = base_r + shifts[j]   # row shift
            coords_c = base_c + shifts[i]   # col shift
            rr, cc = np.meshgrid(coords_r, coords_c, indexing='ij')
            tI1 = map_coordinates(I1c, [rr, cc], order=1, mode='constant',
                                  cval=0.0)
            ssdem[i, j] = np.sum((tI1 - I2c) ** 2)

    ssde = ssdem.min()
    idx = np.unravel_index(ssdem.argmin(), ssdem.shape)
    best_i, best_j = idx

    coords_r = base_r + shifts[best_j]
    coords_c = base_c + shifts[best_i]
    rr, cc = np.meshgrid(coords_r, coords_c, indexing='ij')
    tI1 = map_coordinates(I1c, [rr, cc], order=1, mode='constant', cval=0.0)

    return ssde, tI1
