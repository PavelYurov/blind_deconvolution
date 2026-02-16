import numpy as np
from numpy.fft import fft2, ifft2

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert Point Spread Function to Optical Transfer Function.
    Centers the kernel and pads it to image size.
    """
    in_shape = psf.shape
    # Pad to image size
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:in_shape[0], :in_shape[1]] = psf
    
    # Circularly shift to center the kernel at (0,0)
    for axis, axis_size in enumerate(in_shape):
        padded = np.roll(padded, -int(axis_size / 2), axis=axis)
        
    return fft2(padded)

def get_grad_operators(shape: tuple) -> tuple:
    """
    Returns the Optical Transfer Functions (OTF) of the forward difference operators.
    Used for implementing Eq. (5) and Eq. (13) in frequency domain.
    
    Returns:
        F_dx, F_dy: Complex arrays of shape `shape`.
    """
    h, w = shape
    # Forward difference kernel x: [0, -1, 1] (centered such that -1 is at 0)
    # Corresponds to x[i, j+1] - x[i, j]
    dx = np.zeros((h, w), dtype=np.float32)
    dx[0, 0] = -1
    dx[0, 1] = 1
    
    dy = np.zeros((h, w), dtype=np.float32)
    dy[0, 0] = -1
    dy[1, 0] = 1
    
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    
    return F_dx, F_dy

def compute_gradients(img: np.ndarray) -> tuple:
    """
    Compute gradients in spatial domain using forward differences with periodic boundary.
    Matches the F_dx, F_dy operators defined above.
    """
    # dx: x[i, j+1] - x[i, j] -> roll(-1) - x
    dx = np.roll(img, -1, axis=1) - img
    dy = np.roll(img, -1, axis=0) - img
    return dx, dy

def compute_divergence(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Compute divergence (adjoint of gradient).
    Used for the regularization term in IRLS (TV).
    D^T v corresponds to backward difference with sign flip.
    """
    # D_x^T * v = v[i, j-1] - v[i, j] (Backward diff)
    dx = np.roll(vx, 1, axis=1) - vx
    dy = np.roll(vy, 1, axis=0) - vy