import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

EPSILON = 1e-12

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert Point Spread Function to Optical Transfer Function.
    Centers the kernel and pads it to image size.
    """
    in_shape = psf.shape
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:in_shape[0], :in_shape[1]] = psf
    
    # Circular shift to center the kernel at (0,0)
    psf_padded = np.roll(psf_padded, -in_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -in_shape[1] // 2, axis=1)
    return fft2(psf_padded)

def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """Convert OTF back to PSF."""
    psf_padded = np.real(ifft2(otf))
    psf_padded = np.roll(psf_padded, out_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, out_shape[1] // 2, axis=1)
    return psf_padded[:out_shape[0], :out_shape[1]]

def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Soft thresholding operator: sign(x) * max(|x| - thresh, 0)."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def compute_spatial_gradient(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes gradients in spatial domain using forward differences."""
    # Roll -1 is equivalent to x[i+1] - x[i] with wrap around
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    return dx, dy

def precompute_gradient_operators(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns FFT of gradient operators (dx, dy) and squared magnitude.
    """
    H, W = shape
    dx = np.zeros((H, W)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((H, W)); dy[0, 0] = -1; dy[1, 0] = 1
    
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
    return F_dx, F_dy, F_grad_sq