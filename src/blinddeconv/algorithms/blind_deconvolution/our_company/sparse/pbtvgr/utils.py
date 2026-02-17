import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import center_of_mass, shift

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert Point Spread Function to Optical Transfer Function.
    Centers the kernel and pads it to image size.
    """
    in_shape = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:in_shape[0], :in_shape[1]] = psf
    
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
    dx = np.roll(img, -1, axis=1) - img
    dy = np.roll(img, -1, axis=0) - img
    return dx, dy

def compute_divergence(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """
    Compute divergence (adjoint of gradient).
    Used for the regularization term in IRLS (TV).
    D^T v corresponds to backward difference with sign flip.
    """
    dx = np.roll(vx, 1, axis=1) - vx
    dy = np.roll(vy, 1, axis=0) - vy

    return dx + dy

def adjust_psf(h: np.ndarray) -> np.ndarray:
    """
    Post-processing for Kernel:
    1. Thresholding: Sets small values to 0 to remove noise.
    2. Centering: Shifts center of mass to the geometric center.
    Required to resolve shift ambiguity in Blind Deconvolution.
    """
    h = h.copy()
    
    threshold = h.max() * 0.05
    h[h < threshold] = 0
    
    cy, cx = center_of_mass(h)
    kh, kw = h.shape

    shift_y = (kh // 2) - cy
    shift_x = (kw // 2) - cx
    
    if not np.isnan(shift_y) and not np.isnan(shift_x):
        h = shift(h, (shift_y, shift_x), order=1, mode='constant', cval=0.0)
    
    h = np.maximum(h, 0)
    h_sum = np.sum(h)
    if h_sum > 1e-12:
        h /= h_sum
        
    return h