"""
Utility functions for Low-Rank Blind Deconvolution.

Provides FFT-based convolution operators, gradient computation,
multiscale pyramid construction, kernel post-processing helpers,
and color-space conversions.

References
----------
[1] Li, S., Chu, W., & Kuo, C.-C.J. "Understanding kernel size in blind
    deconvolution." WACV, 2019.
    GitHub: https://github.com/lisiyaoATbnu/low_rank_kernel
[2] Ren, D., et al. "Image Deblurring via Enhanced Low Rank Prior."
    IEEE TIP, vol. 25, no. 7, pp. 3426–3437, 2016.
[3] Krishnan, D. & Fergus, R. "Fast Image Deconvolution using
    Hyper-Laplacian Priors." NIPS, 2009.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from typing import Tuple, List, Optional


# =============================================================================
#   FFT / Convolution Helpers
# =============================================================================

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).

    Equivalent to MATLAB's ``psf2otf``: zero-pads the PSF to the target
    shape, circularly shifts so that the center of the PSF aligns with
    the (0, 0) frequency bin, and computes the 2-D DFT.

    Parameters
    ----------
    psf : np.ndarray, shape (kh, kw)
        Point spread function.
    shape : tuple of int
        Target spatial dimensions (H, W) for the OTF.

    Returns
    -------
    otf : np.ndarray, complex, shape (H, W)
        Optical transfer function in frequency domain.

    Notes
    -----
    Standard operation in Fourier-based image deconvolution
    (see [3], Appendix).
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    padded = np.zeros(shape)
    padded[:psf.shape[0], :psf.shape[1]] = psf

    # Circular shift: center of PSF → origin (0, 0)
    shift_y = -(psf.shape[0] // 2)
    shift_x = -(psf.shape[1] // 2)
    padded = np.roll(np.roll(padded, shift_y, axis=0), shift_x, axis=1)

    return np.fft.fft2(padded)


def convolve2d(image: np.ndarray, kernel: np.ndarray,
               mode: str = 'same') -> np.ndarray:
    """
    2-D convolution via FFT.

    Thin wrapper around ``scipy.signal.fftconvolve`` performing
    mathematical convolution (with kernel flipping), consistent
    with MATLAB's ``conv2``.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    kernel : np.ndarray, shape (kh, kw)
    mode : {'full', 'same', 'valid'}

    Returns
    -------
    result : np.ndarray
    """
    return fftconvolve(image, kernel, mode=mode)


# =============================================================================
#   Gradient Operators
# =============================================================================

def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute horizontal and vertical image gradients.

    Applies the 2×2 forward-difference operators::

        dx = [[1, -1],      dy = [[ 1,  0],
              [0,  0]]            [-1,  0]]

    with ``'valid'`` boundary handling, reducing each spatial dimension
    by 1.  This follows the standard gradient-domain formulation used in
    blind deconvolution ([1], multiscaled_cry.m; [3], Sec. 3).

    Parameters
    ----------
    image : np.ndarray, shape (H, W)

    Returns
    -------
    grad_x : np.ndarray, shape (H-1, W-1)
        Horizontal (x-direction) gradient.
    grad_y : np.ndarray, shape (H-1, W-1)
        Vertical (y-direction) gradient.

    References
    ----------
    Follows the gradient filter design of Krishnan et al. (CVPR 2011):
    ``dx = [1, -1; 0, 0]``, ``dy = [1, 0; -1, 0]``.
    """
    dx = np.array([[1.0, -1.0],
                   [0.0,  0.0]])
    dy = np.array([[ 1.0, 0.0],
                   [-1.0, 0.0]])

    grad_x = fftconvolve(image, dx, mode='valid')
    grad_y = fftconvolve(image, dy, mode='valid')

    return grad_x, grad_y


# =============================================================================
#   Multi-Scale Pyramid
# =============================================================================

def build_scale_pyramid(kernel_size: int) -> List[int]:
    """
    Build a coarse-to-fine scale pyramid for multi-scale blind
    deconvolution.

    Starting from a minimum scale (derived from ``kernel_size``),
    each subsequent scale is obtained by multiplying by √2 and
    rounding to the nearest odd integer.  The final entry equals
    ``kernel_size``.

    Parameters
    ----------
    kernel_size : int
        Target (maximum) kernel size.  Must be odd and ≥ 3.

    Returns
    -------
    scales : list of int
        Ascending list of odd kernel sizes.

    References
    ----------
    [1] Li et al. (WACV 2019), multiscaled_cry.m:
        "We follow Krishnan CVPR 2011 code to design layers of the
        scaling pyramid (/16)."
    """
    assert kernel_size >= 3 and kernel_size % 2 == 1, \
        "kernel_size must be odd and >= 3"

    # Minimum scale: max(2*floor((K-1)/32)+1, 3)
    min_scale = max(2 * ((kernel_size - 1) // 32) + 1, 3)

    scales: List[int] = []
    layer = min_scale
    step = np.sqrt(2.0)

    while layer < kernel_size:
        # Ensure odd
        if layer % 2 == 0:
            layer += 1
        scales.append(int(layer))
        layer = int(np.floor(layer * step))
        if layer % 2 == 0:
            layer += 1

    # Always include the target size as the final scale
    scales.append(kernel_size)
    return scales


# =============================================================================
#   Kernel Post-Processing
# =============================================================================

def center_kernel(
    kernel: np.ndarray,
    images: Optional[Tuple[np.ndarray, ...]] = None
) -> Tuple:
    """
    Center the kernel by its center-of-mass and shift associated images.

    Translates the kernel so that its centre of mass coincides with
    the geometric center.  If companion images are provided, they are
    shifted in the *opposite* direction (maintaining the convolution
    relation ``y = x ⊛ k``).

    Parameters
    ----------
    kernel : np.ndarray, shape (kh, kw)
    images : tuple of np.ndarray, optional
        Gradient-domain images to be shifted (opposite direction).

    Returns
    -------
    kernel_centered : np.ndarray
    *images_shifted  : np.ndarray  (only when *images* is given)

    References
    ----------
    ``center_kernel_separate.m`` from [1].
    """
    kh, kw = kernel.shape
    total = kernel.sum()

    if total < 1e-10:
        if images is not None:
            return (kernel,) + images
        return kernel

    # Center of mass (0-indexed)
    mu_y = np.sum(np.arange(kh) * kernel.sum(axis=1)) / total
    mu_x = np.sum(np.arange(kw) * kernel.sum(axis=0)) / total

    # Integer offset from geometric center
    offset_y = int(np.round(kh // 2 - mu_y))
    offset_x = int(np.round(kw // 2 - mu_x))

    if offset_y == 0 and offset_x == 0:
        if images is not None:
            return (kernel,) + images
        return kernel

    # Build a delta-function translation kernel
    shift_h = 2 * abs(offset_y) + 1
    shift_w = 2 * abs(offset_x) + 1
    shift_kern = np.zeros((shift_h, shift_w))
    shift_kern[abs(offset_y) + offset_y,
               abs(offset_x) + offset_x] = 1.0

    # Shift the PSF towards center
    kernel_centered = fftconvolve(kernel, shift_kern, mode='same')

    if images is not None:
        # The inverse shift for the images (flip the translation kernel)
        inv_kern = shift_kern[::-1, ::-1]
        shifted = tuple(
            fftconvolve(img, inv_kern, mode='same') for img in images
        )
        return (kernel_centered,) + shifted

    return kernel_centered


def normalize_kernel(kernel: np.ndarray,
                     threshold: float = 0.0) -> np.ndarray:
    """
    Project the kernel onto the feasible set: non-negative, unit-sum,
    with optional small-value thresholding.

    Parameters
    ----------
    kernel : np.ndarray
    threshold : float
        Fraction of ``max(kernel)`` below which elements are zeroed.

    Returns
    -------
    kernel : np.ndarray
    """
    kernel = np.clip(kernel, 0, None)

    if threshold > 0 and kernel.max() > 0:
        kernel[kernel < kernel.max() * threshold] = 0.0

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def edgetaper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Taper image edges to reduce boundary ringing in deconvolution.

    Blends the boundary region with a blurred version, using a smooth
    weight map derived from the kernel support.  Approximates MATLAB's
    ``edgetaper``.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
    kernel : np.ndarray, shape (kh, kw)

    Returns
    -------
    tapered : np.ndarray, shape (H, W)
    """
    blurred = fftconvolve(image, kernel, mode='same')

    kh, kw = kernel.shape
    H, W = image.shape

    # Build a 2-D weight that is 1 in the interior and fades to 0
    # over a border region whose width matches half the kernel support.
    wy = np.ones(H)
    wx = np.ones(W)

    border_y = max(kh // 2, 1)
    border_x = max(kw // 2, 1)

    # Raised-cosine (Hann) window → C¹-smooth transition.
    # Linear ramp has a derivative discontinuity at the junction with
    # the interior, causing spectral leakage / boundary ringing.
    t_y = np.linspace(0, 1, border_y, endpoint=False)
    t_x = np.linspace(0, 1, border_x, endpoint=False)
    ramp_y = 0.5 * (1.0 - np.cos(np.pi * t_y))
    ramp_x = 0.5 * (1.0 - np.cos(np.pi * t_x))

    wy[:border_y] = ramp_y
    wy[-border_y:] = ramp_y[::-1]
    wx[:border_x] = ramp_x
    wx[-border_x:] = ramp_x[::-1]

    weight = wy[:, None] * wx[None, :]

    return weight * image + (1.0 - weight) * blurred


# =============================================================================
#   Image Resizing
# =============================================================================

def resize_image(image: np.ndarray,
                 target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear-interpolation resize (wrapper around ``scipy.ndimage.zoom``).

    Parameters
    ----------
    image : np.ndarray, 2-D
    target_shape : (H_new, W_new)

    Returns
    -------
    resized : np.ndarray
    """
    h_in, w_in = image.shape[:2]
    h_out, w_out = target_shape

    if h_in == h_out and w_in == w_out:
        return image.copy()

    factors = (h_out / h_in, w_out / w_in)
    return zoom(image, factors, order=1)   # order=1 → bilinear


# =============================================================================
#   Color-Space Conversions
# =============================================================================

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image (float, [0, 1]) to YCbCr.

    Uses the ITU-R BT.601 conversion matrix, consistent with
    MATLAB's ``rgb2ycbcr`` for double-precision inputs.

    Parameters
    ----------
    image : np.ndarray, shape (H, W, 3), float in [0, 1]

    Returns
    -------
    ycbcr : np.ndarray, shape (H, W, 3)
    """
    M = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.500   ],
        [ 0.500,    -0.418688, -0.081312]
    ])
    ycbcr = image @ M.T
    ycbcr[:, :, 1:] += 0.5
    return ycbcr


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """
    Convert a YCbCr image to RGB.

    Inverse of :func:`rgb_to_ycbcr`.

    Parameters
    ----------
    ycbcr : np.ndarray, shape (H, W, 3)

    Returns
    -------
    rgb : np.ndarray, shape (H, W, 3)
    """
    ycbcr = ycbcr.copy()
    ycbcr[:, :, 1:] -= 0.5

    M_inv = np.array([
        [1.0,  0.0,       1.402   ],
        [1.0, -0.344136, -0.714136],
        [1.0,  1.772,     0.0     ]
    ])
    return ycbcr @ M_inv.T
