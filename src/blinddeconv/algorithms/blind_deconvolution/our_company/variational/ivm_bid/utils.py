import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple
from scipy.ndimage import gaussian_filter

EPSILON = 1e-10

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert PSF to OTF with centering."""
    h, w = shape
    kh, kw = psf.shape
    padded = np.zeros((h, w), dtype=psf.dtype)
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -int(kh // 2), axis=0)
    padded = np.roll(padded, -int(kw // 2), axis=1)
    return fft2(padded)

def otf2psf(otf: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Convert OTF back to PSF."""
    kh, kw = kernel_shape
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, int(kh // 2), axis=0)
    psf_full = np.roll(psf_full, int(kw // 2), axis=1)
    return psf_full[:kh, :kw]

def project_kernel(h: np.ndarray) -> np.ndarray:
    """Constraints: non-negative and sum to 1."""
    h = np.maximum(h, 0.0)
    total = np.sum(h)
    if total > EPSILON:
        h /= total
    else:
        h[:] = 1.0 / h.size
    return h

def precompute_gradient_operators(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectral gradient operators."""
    H, W = shape
    dx = np.zeros((H, W)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((H, W)); dy[0, 0] = -1; dy[1, 0] = 1
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
    return F_dx, F_dy, F_grad_sq

def initialize_kernel(kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Gaussian initialization."""
    kh, kw = kernel_shape
    sigma = min(kh, kw) / 6.0 
    y, x = np.ogrid[-(kh//2):(kh//2 + kh%2), -(kw//2):(kw//2 + kw%2)]
    h = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return project_kernel(h)

def compute_residual_energy(g: np.ndarray, f: np.ndarray, h: np.ndarray) -> float:
    """MSE in image domain."""
    H, W = g.shape
    F_g = fft2(g)
    F_f = fft2(f)
    F_h = psf2otf(h, (H, W))
    diff = F_g - F_h * F_f
    return float(np.mean(np.abs(ifft2(diff))**2))

def edgetaper(img: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Smooth image edges to avoid FFT boundary artifacts."""
    h, w = img.shape
    kh, kw = kernel_shape
    taper_h = min(h, kh * 2)
    taper_w = min(w, kw * 2)
    
    window_v = np.ones(h)
    window_h = np.ones(w)
    
    if taper_h > 0:
        ramp_v = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_h)))
        window_v[:taper_h//2] = ramp_v[:taper_h//2]
        window_v[-taper_h//2:] = ramp_v[taper_h//2:]
    
    if taper_w > 0:
        ramp_h = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_w)))
        window_h[:taper_w//2] = ramp_h[:taper_w//2]
        window_h[-taper_w//2:] = ramp_h[taper_w//2:]
        
    window = np.outer(window_v, window_h)
    mean_val = np.mean(img)
    return img * window + mean_val * (1 - window)

def compute_strong_gradients(f: np.ndarray, threshold_quantile: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns gradients of f, but zeroed out where magnitude is small.
    This helps the kernel solver focus on main edges.
    """
    dx = np.roll(f, -1, axis=1) - f
    dy = np.roll(f, -1, axis=0) - f
    
    mag = np.sqrt(dx**2 + dy**2)
    # Find robust threshold
    threshold = np.quantile(mag, threshold_quantile)
    
    # Create mask
    mask = mag > threshold
    
    return dx * mask, dy * mask