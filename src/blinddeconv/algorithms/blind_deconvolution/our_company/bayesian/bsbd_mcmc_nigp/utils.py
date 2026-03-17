"""
Utility functions for Bayesian Sparse Blind Deconvolution.
Implements Fourier-domain operators and mathematical helpers.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert Point Spread Function to Optical Transfer Function.
    
    Correctly shifts the kernel so its center aligns with (0,0) in frequency domain.
    This prevents the reconstructed kernel from drifting to the top-left corner.
    
    Args:
        psf: Spatial kernel (assumed odd size, centered).
        shape: Target shape (H, W) of the image.
    """
    in_h, in_w = psf.shape
    out_h, out_w = shape
    
    # Pad to output shape
    pad_h = out_h - in_h
    pad_w = out_w - in_w
    
    if pad_h < 0 or pad_w < 0:
        raise ValueError("PSF must be smaller than the target shape")
        
    padded = np.pad(psf, ((0, pad_h), (0, pad_w)), mode='constant')
    
    # Circularly shift so that the center of the kernel moves to (0,0)
    # Assumes kernel center is at (in_h//2, in_w//2)
    shifted = np.roll(padded, (-in_h//2, -in_w//2), axis=(0, 1))
    
    return fft2(shifted)

def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Compute convolution y = x * h using FFT.
    Uses psf2otf to handle phase correctly.
    """
    H, W = image.shape
    otf = psf2otf(kernel, (H, W))
    xf = fft2(image)
    return np.real(ifft2(xf * otf))

def circshift(x: np.ndarray, shift: Tuple[int, int]) -> np.ndarray:
    """
    Circular shift of an array.
    Used for Time-Shift Compensation.
    """
    return np.roll(x, shift, axis=(0, 1))

def compute_gradient_matrix_operators(h_shape: Tuple[int, int], y_shape: Tuple[int, int]):
    """
    Precompute indices for mapping between dense kernel matrix and image autocorrelation.
    Used to construct Sigma_h efficiently.
    """
    kh, kw = h_shape
    H, W = y_shape
    K = kh * kw
    
    # Coordinates of kernel pixels
    coords = [(i // kw, i % kw) for i in range(K)]
    
    # For each pair of kernel pixels (k1, k2), calculating the lag (du, dv)
    # to fetch from the autocorrelation matrix of x.
    indices = []
    for k1 in range(K):
        u1, v1 = coords[k1]
        row_indices = []
        for k2 in range(K):
            u2, v2 = coords[k2]
            du = u1 - u2
            dv = v1 - v2
            # Handle wrapping for negative indices to match FFT output
            row_indices.append((du % H, dv % W))
        indices.append(row_indices)
        
    return np.array(indices)