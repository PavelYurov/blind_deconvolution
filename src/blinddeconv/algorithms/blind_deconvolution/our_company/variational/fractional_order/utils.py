import numpy as np
from scipy.ndimage import minimum_filter, gaussian_filter, generate_binary_structure, binary_dilation
from numpy.fft import fft2, ifft2
from typing import Tuple, List, Any, Dict
def compute_gl_coefficients(alpha: float, n: int) -> np.ndarray:
    c = np.zeros(n, dtype=np.float64)
    c[0] = 1.0
    for k in range(1, n):
        c[k] = (1.0 - (alpha + 1.0) / k) * c[k - 1]
    return c

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    if psf.size == 0:
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    shift_y = -(ph // 2)
    shift_x = -(pw // 2)
    padded = np.roll(padded, shift_y, axis=0)
    padded = np.roll(padded, shift_x, axis=1)
    return fft2(padded)

def fractional_gradient_x(f: np.ndarray, alpha: float) -> np.ndarray:
    M, N = f.shape
    filt = np.zeros((M, N), dtype=np.float64)
    coeffs = compute_gl_coefficients(alpha, N)
    filt[0, :len(coeffs)] = coeffs
    return np.real(ifft2(fft2(f) * fft2(filt)))

def fractional_gradient_y(f: np.ndarray, alpha: float) -> np.ndarray:
    M, N = f.shape
    filt = np.zeros((M, N), dtype=np.float64)
    coeffs = compute_gl_coefficients(alpha, M)
    filt[:len(coeffs), 0] = coeffs
    return np.real(ifft2(fft2(f) * fft2(filt)))

def fractional_otf_x(alpha: float, shape: tuple) -> np.ndarray:
    M, N = shape
    filt = np.zeros((M, N), dtype=np.float64)
    coeffs = compute_gl_coefficients(alpha, N)
    filt[0, :len(coeffs)] = coeffs
    return fft2(filt)

def fractional_otf_y(alpha: float, shape: tuple) -> np.ndarray:
    M, N = shape
    filt = np.zeros((M, N), dtype=np.float64)
    coeffs = compute_gl_coefficients(alpha, M)
    filt[:len(coeffs), 0] = coeffs
    return fft2(filt)

def pmp_operator(f: np.ndarray, patch_size: int = 3) -> np.ndarray:
    return minimum_filter(f, size=patch_size, mode='reflect')

def pmp_mask(f: np.ndarray, patch_size: int = 3) -> np.ndarray:
    pmp_vals = minimum_filter(f, size=patch_size, mode='reflect')
    return (np.abs(f - pmp_vals) < 1e-12).astype(np.float64)

def soft_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)

def hard_threshold(z: np.ndarray, tau: float) -> np.ndarray:
    return z * (np.abs(z) > tau).astype(np.float64)

def apply_hysteresis_threshold(mag: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Applies hysteresis thresholding to gradient magnitudes.
    1. Select strong edges >= high.
    2. Keep weak edges >= low ONLY if they are connected to strong edges.
    """
    # Create masks
    strong_mask = mag >= high
    weak_mask = (mag >= low) & (mag < high)
    
    # Use morphological reconstruction (dilation) to connect weak to strong
    # We iteratively dilate the strong mask into the weak mask
    # Ideally, we use scipy.ndimage.binary_dilation or label, but simple loop is robust
    
    # Structure for connectivity (8-neighbors)
    struct = generate_binary_structure(2, 2)
    
    # The 'markers' are the strong edges. We grow them inside 'mask' (strong + weak)
    total_mask = strong_mask | weak_mask
    
    # Reconstruction by dilation
    # Start with strong
    connected_mask = strong_mask.copy()
    
    # Iterative dilation (can be slow for very spiral shapes, but usually fast for edges)
    # Optimization: scipy has binary_propagation but it's not always exposed simply.
    # We will do a loop which converges quickly for image edges.
    for _ in range(100): # Limit iterations to prevent hang
        new_mask = binary_dilation(connected_mask, structure=struct) & total_mask
        if np.array_equal(new_mask, connected_mask):
            break
        connected_mask = new_mask
        
    return connected_mask.astype(np.float64)

def gradient_otfs(shape: tuple):
    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    return psf2otf(dx, shape), psf2otf(dy, shape)

def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.roll(image, -1, axis=1) - image
    gy = np.roll(image, -1, axis=0) - image
    return gx, gy

def edgetaper(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel_shape
    kh = min(kh, h)
    kw = min(kw, w)
    
    alpha_h = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, kh)))
    alpha_w = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, kw)))
    
    weight_h = np.ones(h)
    weight_h[:kh] = alpha_h
    weight_h[-kh:] = alpha_h[::-1]
    
    weight_w = np.ones(w)
    weight_w[:kw] = alpha_w
    weight_w[-kw:] = alpha_w[::-1]
    
    weight = np.outer(weight_h, weight_w)
    blurred = gaussian_filter(image, sigma=3.0)
    return image * weight + blurred * (1 - weight)

def pad_image_for_kernel(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    h_pad = kernel_shape[0] // 2
    w_pad = kernel_shape[1] // 2
    return np.pad(image, ((h_pad, h_pad), (w_pad, w_pad)), mode='edge')

def build_image_pyramid(image: np.ndarray, num_scales: int,
                        scale_factor: float = 1.0 / np.sqrt(2.0)) -> list:
    from skimage.transform import resize
    pyramid = [image]
    for _ in range(1, num_scales):
        h, w = pyramid[-1].shape
        new_h = max(int(round(h * scale_factor)), 16)
        new_w = max(int(round(w * scale_factor)), 16)
        if new_h == h and new_w == w:
            break
        down = resize(pyramid[-1], (new_h, new_w),
                      anti_aliasing=True, preserve_range=True)
        pyramid.append(down)
    pyramid.reverse()
    return pyramid

def resize_kernel(kernel: np.ndarray, new_shape: tuple) -> np.ndarray:
    from skimage.transform import resize
    resized = resize(kernel, new_shape, anti_aliasing=False, preserve_range=True)
    resized = np.maximum(resized, 0.0)
    s = resized.sum()
    if s > 0: resized /= s
    return resized

def kernel_threshold_and_normalize(h: np.ndarray, rel_threshold: float = 0.05) -> np.ndarray:
    h = np.maximum(h, 0.0)
    peak = h.max()
    if peak > 0:
        h[h < rel_threshold * peak] = 0.0
    s = h.sum()
    if s > 0: h /= s
    return h

def center_kernel(h: np.ndarray) -> np.ndarray:
    from scipy.ndimage import center_of_mass, shift as ndi_shift
    cy, cx = center_of_mass(h)
    target_y, target_x = (h.shape[0] - 1) / 2.0, (h.shape[1] - 1) / 2.0
    h_shifted = ndi_shift(h, (target_y - cy, target_x - cx), order=1, mode='constant')
    h_shifted = np.maximum(h_shifted, 0.0)
    s = h_shifted.sum()
    if s > 0: h_shifted /= s
    return h_shifted