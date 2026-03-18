"""
utils.py

Utility functions for NSM (Normalized Sparsity Measure) blind deconvolution.

Ported from MATLAB code by Krishnan & Fergus (BlindDeconvolution-main/matlab/).
Reference:
    D. Krishnan, T. Tay, R. Fergus:
    "Blind Deconvolution using a Normalized Sparsity Measure", CVPR 2011.

MATLAB -> Python conversion notes (CRITICAL differences):
    conv2(A, B, 'valid'/'same'/'full'):
        MATLAB conv2 is TRUE CONVOLUTION (kernel is flipped internally).
        -> scipy.signal.fftconvolve(A, B, mode=...) is also true convolution.
        Output sizes match exactly:
            'valid': (m-p+1, n-q+1)
            'same' : (m, n)
            'full' : (m+p-1, n+q-1)

    flipud(fliplr(k)):
        Flips both axes -> k[::-1, ::-1] in numpy.

    psf2otf(k, size):
        Circularly shifts kernel center to (0,0) then FFT.
        -> np.roll based placement, then fft2.

    Derivative filters in MATLAB code:
        ms_blind_deconv:  dx = [-1 1; 0 0] (2x2), dy = [-1 0; 1 0] (2x2)
        fast_deconv_bregman: dx = [1 -1] (1x2), dy = [1; -1] (2x1)
        These are NOT Sobel filters (the C++ port incorrectly used Sobel).

    edgetaper(I, PSF):
        MATLAB built-in -- blends image edges with blurred version
        using autocorrelation-derived weights. Implemented below.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve


# =========================================================================
# Convolution wrappers -- matching MATLAB conv2 (true convolution)
# =========================================================================

def conv2_valid(A, B):
    """MATLAB conv2(A, B, 'valid') -- true convolution, valid output."""
    return fftconvolve(A, B, mode='valid')


def conv2_same(A, B):
    """MATLAB conv2(A, B, 'same') -- true convolution, same-size output."""
    return fftconvolve(A, B, mode='same')


def conv2_full(A, B):
    """MATLAB conv2(A, B, 'full') -- true convolution, full output."""
    return fftconvolve(A, B, mode='full')


# =========================================================================
# psf2otf -- place PSF into image-sized array for FFT
# =========================================================================

def psf2otf(psf, shape):
    """
    Equivalent to MATLAB psf2otf.

    1. Zero-pad psf into array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.
    """
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    # Circular shift: move PSF centre to (0,0)
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


# =========================================================================
# compute_constants -- precompute FFT constants for non-blind deconv
# =========================================================================

def compute_constants(f, k, dx, dy):
    """
    Precompute Fourier-domain constants for fast_deconv_bregman.
    Matches MATLAB computeConstants().

    Returns: Ktf, KtK, DtD
    """
    shape = f.shape[:2]
    otfk = psf2otf(k, shape)

    Ktf = np.conj(otfk) * fft2(f)
    KtK = np.abs(otfk) ** 2

    otf_dx = psf2otf(dx, shape)
    otf_dy = psf2otf(dy, shape)
    DtD = np.abs(otf_dx) ** 2 + np.abs(otf_dy) ** 2

    return Ktf, KtK, DtD


# =========================================================================
# init_kernel -- initialise kernel at coarsest level
# =========================================================================

def init_kernel(minsize):
    """
    Initialise a tiny 2-pixel wide kernel at the coarsest level.
    Matches MATLAB init_kernel() inside ms_blind_deconv.m:
        k = zeros(minsize, minsize);
        k((minsize-1)/2, (minsize-1)/2:(minsize-1)/2+1) = 1/2;
    Note: MATLAB is 1-indexed. minsize is always >= 5 and odd.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    # MATLAB 1-indexed center: (minsize-1)/2
    # Python 0-indexed: (minsize-1)//2 - 1
    c = (minsize - 1) // 2  # MATLAB 1-indexed value
    k[c - 1, c - 1] = 0.5
    k[c - 1, c] = 0.5
    return k


# =========================================================================
# center_kernel_separate -- center kernel by centre-of-mass
# =========================================================================

def center_kernel_separate(x, y, k):
    """
    Center the kernel by translation so that boundary issues are mitigated.
    Also shift images x, y in the opposite direction.

    Ported from MATLAB center_kernel_separate.m.
    Uses conv2 with a shift kernel (delta function) for translation.

    Parameters
    ----------
    x : 2D array -- gradient image (channel 1)
    y : 2D array -- gradient image (channel 2)
    k : 2D array -- blur kernel

    Returns
    -------
    x_shifted, y_shifted, k_shifted
    """
    # Centre of mass (1-indexed, matching MATLAB)
    rows = np.arange(1, k.shape[0] + 1, dtype=np.float64)
    cols = np.arange(1, k.shape[1] + 1, dtype=np.float64)

    mu_y = np.sum(rows * k.sum(axis=1))   # sum(k, 2)' in MATLAB = sum along cols
    mu_x = np.sum(cols * k.sum(axis=0))   # sum(k, 1) in MATLAB = sum along rows

    # Offset to centre (MATLAB: floor(size/2) + 1 because of 1-indexing)
    offset_x = int(round(np.floor(k.shape[1] / 2.0) + 1 - mu_x))
    offset_y = int(round(np.floor(k.shape[0] / 2.0) + 1 - mu_y))

    # Create shift kernel (delta function at the offset position)
    sh_rows = abs(offset_y) * 2 + 1
    sh_cols = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sh_rows, sh_cols), dtype=np.float64)
    # MATLAB 1-indexed: (abs(oy)+1+oy, abs(ox)+1+ox)
    # Python 0-indexed: (abs(oy)+oy, abs(ox)+ox)
    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    # Shift kernel via convolution
    k_shifted = conv2_same(k, shift_kernel)

    # Shift images in opposite direction: conv2(x, flipud(fliplr(shift_kernel)), 'same')
    flipped_sk = shift_kernel[::-1, ::-1]
    x_shifted = conv2_same(x, flipped_sk)
    y_shifted = conv2_same(y, flipped_sk)

    return x_shifted, y_shifted, k_shifted


# =========================================================================
# edgetaper -- replicate MATLAB built-in edgetaper
# =========================================================================

def edgetaper(img, psf):
    """
    Replicate MATLAB's edgetaper(I, PSF).

    Blends the edges of image I with a blurred version using weights
    derived from the autocorrelation of the PSF projections.

    J = beta * I + (1 - beta) * blurred
    where beta = 1 in interior, < 1 at edges.
    """
    sn, sm = psf.shape
    n, m = img.shape

    # Project PSF onto each axis and compute autocorrelation
    proj_y = psf.sum(axis=1)   # sum along columns -> (sn,)
    proj_x = psf.sum(axis=0)   # sum along rows    -> (sm,)

    z_y = np.correlate(proj_y, proj_y, mode='full')   # length 2*sn-1
    z_x = np.correlate(proj_x, proj_x, mode='full')   # length 2*sm-1

    # Normalize to [0, 1]
    z_y = z_y / z_y.max()
    z_x = z_x / z_x.max()

    # Embed autocorrelation into image-sized arrays with circular shift
    w_y = np.zeros(n, dtype=np.float64)
    if len(z_y) <= n:
        w_y[:len(z_y)] = z_y
    else:
        w_y[:] = z_y[sn - 1 : sn - 1 + n]
    w_y = np.roll(w_y, -(sn - 1))
    w_y = np.maximum(w_y, 0)

    w_x = np.zeros(m, dtype=np.float64)
    if len(z_x) <= m:
        w_x[:len(z_x)] = z_x
    else:
        w_x[:] = z_x[sm - 1 : sm - 1 + m]
    w_x = np.roll(w_x, -(sm - 1))
    w_x = np.maximum(w_x, 0)

    # 2D blending weight: 1 in interior, 0 at edges
    beta = 1.0 - np.outer(w_y, w_x)

    # Blurred image via circular convolution (FFT)
    blurred = np.real(ifft2(fft2(img) * psf2otf(psf, img.shape)))

    return beta * img + (1.0 - beta) * blurred
