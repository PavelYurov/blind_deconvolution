"""
Вспомогательные функции для алгоритма Blind Deconvolution.
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom, center_of_mass, shift

def pad_image(img: np.ndarray, pad_width: tuple, mode='wrap') -> np.ndarray:
    return np.pad(img, pad_width, mode=mode)

def crop_image(img: np.ndarray, crop_width: tuple) -> np.ndarray:
    h, w = img.shape
    ph, pw = crop_width
    return img[ph:h-ph, pw:w-pw]

def compute_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """D operator (Forward difference with cyclic boundary)."""
    dy = np.roll(img, -1, axis=0) - img
    dx = np.roll(img, -1, axis=1) - img
    return dy, dx

def compute_divergence(py: np.ndarray, px: np.ndarray) -> np.ndarray:
    """D^T operator (Negative Backward difference)."""
    dty = np.roll(py, 1, axis=0) - py
    dtx = np.roll(px, 1, axis=1) - px
    return dty + dtx

def get_boundary_mask(shape: tuple[int, int], kernel_shape: tuple[int, int]) -> np.ndarray:
    """Mask to ignore boundary effects (Gamma=0 at borders)."""
    H, W = shape
    kh, kw = kernel_shape
    mask = np.zeros((H, W), dtype=np.float32)
    ph = kh // 2 + 1
    pw = kw // 2 + 1
    if H > 2*ph and W > 2*pw:
        mask[ph:-ph, pw:-pw] = 1.0
    else:
        mask[:, :] = 1.0
    return mask

def center_kernel(h: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Center kernel mass to prevent image drift."""
    kh, kw = h.shape
    cy, cx = center_of_mass(h)
    target_cy, target_cx = kh // 2, kw // 2
    
    shift_y = int(round(target_cy - cy))
    shift_x = int(round(target_cx - cx))
    
    if shift_y == 0 and shift_x == 0:
        return h, (0, 0)
        
    h_centered = shift(h, (shift_y, shift_x), order=0, mode='constant', cval=0.0)
    
    # Re-normalize after shift (interpolation might change sum)
    s = h_centered.sum()
    if s > 1e-12:
        h_centered /= s
        
    return h_centered, (shift_y, shift_x)

def shift_image(img: np.ndarray, shift_val: tuple[int, int]) -> np.ndarray:
    """Cyclic shift of image."""
    dy, dx = shift_val
    return np.roll(img, (-dy, -dx), axis=(0, 1))

def resize_image(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    curr_h, curr_w = img.shape
    req_h, req_w = shape
    if (curr_h, curr_w) == (req_h, req_w):
        return img
    return zoom(img, (req_h / curr_h, req_w / curr_w), order=3)

def resize_kernel(ker: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    curr_h, curr_w = ker.shape
    req_h, req_w = shape
    if (curr_h, curr_w) == (req_h, req_w):
        return ker
    res = zoom(ker, (req_h / curr_h, req_w / curr_w), order=1)
    res = np.maximum(res, 0)
    s = res.sum()
    if s > 1e-12:
        res /= s
    return res