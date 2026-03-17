"""
utils.py

Auxiliary functions for Blind Deconvolution.
"""

import numpy as np
import cv2
from scipy.ndimage import center_of_mass, shift

def pad_for_kernel(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Pads image to handle convolution boundary conditions."""
    h, w = kernel_shape
    pad_h = h // 2
    pad_w = w // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

def crop_center(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    y, x = img.shape
    ty, tx = target_shape
    starty = y // 2 - ty // 2
    startx = x // 2 - tx // 2
    return img[starty:starty+ty, startx:startx+tx]

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """Converts PSF to OTF with centering."""
    in_height, in_width = shape
    psf_height, psf_width = psf.shape
    padded_psf = np.zeros((in_height, in_width), dtype=psf.dtype)
    padded_psf[:psf_height, :psf_width] = psf
    padded_psf = np.roll(padded_psf, -psf_height // 2, axis=0)
    padded_psf = np.roll(padded_psf, -psf_width // 2, axis=1)
    return np.fft.fft2(padded_psf)

def compute_gradient(img: np.ndarray) -> np.ndarray:
    """Forward differences. Returns (2, H, W)."""
    h, w = img.shape
    grad = np.zeros((2, h, w), dtype=img.dtype)
    grad[0, :, :-1] = img[:, 1:] - img[:, :-1]
    grad[1, :-1, :] = img[1:, :] - img[:-1, :]
    return grad

def compute_divergence(p: np.ndarray) -> np.ndarray:
    """Backward differences (adjoint of gradient)."""
    p_x = p[0]
    p_y = p[1]
    
    div_x = np.zeros_like(p_x)
    div_x[:, 1:-1] = p_x[:, 1:-1] - p_x[:, :-2]
    div_x[:, 0] = p_x[:, 0]
    div_x[:, -1] = -p_x[:, -2]

    div_y = np.zeros_like(p_y)
    div_y[1:-1, :] = p_y[1:-1, :] - p_y[:-2, :]
    div_y[0, :] = p_y[0, :]
    div_y[-1, :] = -p_y[-2, :]

    return div_x + div_y

def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0: return img
    H, W = img.shape
    new_H, new_W = int(H * scale), int(W * scale)
    # Make dimensions even to simplify pyramid operations slightly
    if new_H % 2 != 0: new_H -= 1
    if new_W % 2 != 0: new_W -= 1
    return cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)

def resize_kernel(k: np.ndarray, size: int) -> np.ndarray:
    """Resizes kernel and re-normalizes."""
    k_resized = cv2.resize(k, (size, size), interpolation=cv2.INTER_CUBIC)
    k_resized[k_resized < 0] = 0
    s = k_resized.sum()
    if s > 0: k_resized /= s
    return k_resized

def center_kernel(k: np.ndarray) -> np.ndarray:
    """
    Shifts the kernel so its center of mass is at the geometric center.
    CRITICAL for blind deconvolution stability.
    """
    h, w = k.shape
    cy, cx = center_of_mass(k)
    shift_y = (h // 2) - cy
    shift_x = (w // 2) - cx
    
    # Cyclic shift is okay because the kernel support should be padded/small enough
    k_centered = shift(k, shift=(shift_y, shift_x), mode='constant', cval=0.0)
    
    # Re-normalize just in case
    k_centered[k_centered < 0] = 0
    s = k_centered.sum()
    if s > 0: k_centered /= s
    return k_centered