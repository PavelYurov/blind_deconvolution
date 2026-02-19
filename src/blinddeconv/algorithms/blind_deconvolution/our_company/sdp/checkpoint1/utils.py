"""
Utility Functions for Steepest Descent on Quotient Manifold (SDP)
Blind Image Deconvolution.
"""

import numpy as np
from numpy.fft import fft2, ifft2


# ═══════════════════════════════════════════════════════════════════════
#  FFT-Based Linear Operators
# ═══════════════════════════════════════════════════════════════════════

def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular 2-D convolution via FFT."""
    H, W = image.shape
    kh, kw = kernel.shape

    h_pad = np.zeros((H, W), dtype=np.float64)
    h_pad[:kh, :kw] = kernel
    # Centre kernel at origin for proper circular convolution
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)

    return np.real(ifft2(fft2(image) * fft2(h_pad)))


def fft_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular 2-D cross-correlation via FFT."""
    H, W = image.shape
    kh, kw = kernel.shape

    h_pad = np.zeros((H, W), dtype=np.float64)
    h_pad[:kh, :kw] = kernel
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)

    return np.real(ifft2(fft2(image) * np.conj(fft2(h_pad))))


# ═══════════════════════════════════════════════════════════════════════
#  Kernel Constraint Operators
# ═══════════════════════════════════════════════════════════════════════

def project_kernel(h: np.ndarray) -> np.ndarray:
    """Euclidean projection onto simplex."""
    h = np.maximum(h, 0.0)
    total = h.sum()
    if total > 1e-15:
        h /= total
    return h


def threshold_kernel(h: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Hard thresholding for kernel sparsity."""
    peak = h.max()
    if peak > 1e-15:
        h = np.where(h < fraction * peak, 0.0, h)
    return project_kernel(h)


def center_kernel(h: np.ndarray) -> np.ndarray:
    """Center of mass alignment."""
    kh, kw = h.shape
    total = h.sum()
    if total < 1e-15:
        return h

    yy, xx = np.mgrid[:kh, :kw]
    cy = (yy * h).sum() / total
    cx = (xx * h).sum() / total

    shift_y = int(np.round(kh / 2.0 - cy))
    shift_x = int(np.round(kw / 2.0 - cx))

    return np.roll(np.roll(h, shift_y, axis=0), shift_x, axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Total Variation Differential Operators
# ═══════════════════════════════════════════════════════════════════════

def tv_gradient(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Gradient of Isotropic Total Variation."""
    # Forward differences
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x

    mag = np.sqrt(dx ** 2 + dy ** 2 + epsilon ** 2)
    nx = dx / mag
    ny = dy / mag

    # Divergence (backward differences)
    div_x = nx - np.roll(nx, 1, axis=1)
    div_y = ny - np.roll(ny, 1, axis=0)

    return -(div_x + div_y)


# ═══════════════════════════════════════════════════════════════════════
#  Boundary Handling (CRITICAL FIX)
# ═══════════════════════════════════════════════════════════════════════

def edge_taper(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """
    Apply edge tapering to reduce boundary artifacts in FFT convolution.
    
    Instead of multiplying by a window (which creates black borders), 
    we blend the edges towards the image mean to enforce periodicity 
    without destroying the image content.
    """
    kh, kw = kernel_shape
    H, W = image.shape
    
    # Calculate taper width
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Create 1D tapers
    def _get_taper_1d(size, pad):
        t = np.ones(size)
        if pad > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(pad) / pad))
            t[:pad] = ramp
            t[-pad:] = ramp[::-1]
        return t

    wy = _get_taper_1d(H, pad_h).reshape(-1, 1)
    wx = _get_taper_1d(W, pad_w).reshape(1, -1)
    
    # Create 2D mask (1 in center, 0 at edges)
    mask = wy * wx
    
    # FIX: Blend towards the mean instead of zero (black)
    # This prevents the "black frame" artifact and associated ringing.
    mean_val = np.mean(image)
    tapered = image * mask + mean_val * (1.0 - mask)
    
    return tapered


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Scale Utilities
# ═══════════════════════════════════════════════════════════════════════

def downsample(image: np.ndarray, factor: int = 2) -> np.ndarray:
    """Downsample with averaging."""
    H, W = image.shape
    nH = (H // factor) * factor
    nW = (W // factor) * factor
    cropped = image[:nH, :nW]
    return cropped.reshape(nH // factor, factor,
                           nW // factor, factor).mean(axis=(1, 3))


def resize_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Bilinear resize."""
    h_in, w_in = image.shape
    H, W = target_shape

    row_idx = np.linspace(0, h_in - 1, H)
    col_idx = np.linspace(0, w_in - 1, W)

    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, h_in - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, w_in - 1)

    dr = (row_idx - r0)[:, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :]

    return (image[np.ix_(r0, c0)] * (1 - dr) * (1 - dc) +
            image[np.ix_(r0, c1)] * (1 - dr) * dc +
            image[np.ix_(r1, c0)] * dr * (1 - dc) +
            image[np.ix_(r1, c1)] * dr * dc)


def upsample_kernel(h: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Upsample kernel and re-project."""
    return project_kernel(resize_image(h, target_shape))


def build_pyramid(y: np.ndarray, num_scales: int = 4, min_size: int = 32) -> list:
    """Build image pyramid."""
    levels = [y]
    for _ in range(num_scales - 1):
        prev = levels[-1]
        if prev.shape[0] < 2 * min_size or prev.shape[1] < 2 * min_size:
            break
        levels.append(downsample(prev, factor=2))
    levels.reverse()
    return levels


def kernel_shape_for_level(kernel_shape: tuple, level: int, num_levels: int) -> tuple:
    """Calculate kernel size for pyramid level."""
    ratio = 2.0 ** (num_levels - 1 - level)
    kh = max(3, int(np.round(kernel_shape[0] / ratio)))
    kw = max(3, int(np.round(kernel_shape[1] / ratio)))
    # Force odd sizes
    kh = kh if kh % 2 == 1 else kh + 1
    kw = kw if kw % 2 == 1 else kw + 1
    return (kh, kw)


def init_gaussian_kernel(shape: tuple, sigma: float = None) -> np.ndarray:
    """Init Gaussian kernel."""
    kh, kw = shape
    if sigma is None:
        sigma = max(kh, kw) / 6.0
    cy, cx = kh // 2, kw // 2
    yy, xx = np.mgrid[:kh, :kw]
    h = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    h /= h.sum()
    return h