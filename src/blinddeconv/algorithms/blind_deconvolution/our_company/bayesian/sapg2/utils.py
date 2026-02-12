import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, Union, Optional

EPSILON = 1e-12

def psf2otf(psf: np.ndarray, shape: Tuple[int, int], boundary: str = 'periodic') -> np.ndarray:
    """
    Convert PSF to OTF with specified boundary condition.
    boundary: 'periodic' (default) or 'reflect' (zero-padding with reflection).
    """
    h, w = psf.shape
    H, W = shape
    if boundary == 'periodic':
        psf_pad = np.zeros((H, W), dtype=psf.dtype)
        psf_pad[:h, :w] = psf
        # Circular shift to center
        psf_pad = np.roll(psf_pad, -h // 2, axis=0)
        psf_pad = np.roll(psf_pad, -w // 2, axis=1)
        return fft2(psf_pad)
    elif boundary == 'reflect':
        # Zero-padding without circular shift (for reflective boundaries)
        psf_pad = np.zeros((H, W), dtype=psf.dtype)
        psf_pad[:h, :w] = psf
        return fft2(psf_pad)
    else:
        raise ValueError("boundary must be 'periodic' or 'reflect'")

def otf2psf(otf: np.ndarray, shape: Tuple[int, int], boundary: str = 'periodic') -> np.ndarray:
    """Inverse of psf2otf."""
    h, w = shape
    psf_pad = np.real(ifft2(otf))
    if boundary == 'periodic':
        psf_pad = np.roll(psf_pad, h // 2, axis=0)
        psf_pad = np.roll(psf_pad, w // 2, axis=1)
    return psf_pad[:h, :w]

def compute_spatial_gradient(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Forward differences with periodic boundaries."""
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    return dx, dy

def precompute_gradient_operators(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FFT of gradient operators and squared magnitude."""
    H, W = shape
    dx = np.zeros((H, W)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((H, W)); dy[0, 0] = -1; dy[1, 0] = 1
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
    return F_dx, F_dy, F_grad_sq

def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """Soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def tv_prox(x: np.ndarray, lambda_tv: float, max_iter: int = 20) -> np.ndarray:
    """
    Proximal operator for isotropic TV: prox_{λ·TV}(x).
    Chambolle's algorithm.
    """
    H, W = x.shape
    p_x = np.zeros((H, W))
    p_y = np.zeros((H, W))
    tau = 1.0 / 8.0
    for _ in range(max_iter):
        div_p = np.roll(p_x, -1, axis=1) - p_x + np.roll(p_y, -1, axis=0) - p_y
        grad_x, grad_y = compute_spatial_gradient(x + lambda_tv * div_p)
        denom = np.sqrt(grad_x**2 + grad_y**2 + EPSILON)
        p_x = (p_x + tau * grad_x) / denom
        p_y = (p_y + tau * grad_y) / denom
    div_p = np.roll(p_x, -1, axis=1) - p_x + np.roll(p_y, -1, axis=0) - p_y
    return x - lambda_tv * div_p

def gaussian_psf(alpha: Union[float, Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Gaussian PSF. 
    If alpha is float: isotropic with inverse width = alpha.
    If alpha is tuple (ah, av): anisotropic with horizontal and vertical inverse widths.
    h(u,v) ∝ exp( -0.5 * (ah² * u² + av² * v²) )
    """
    h, w = shape
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)
    if isinstance(alpha, (int, float)):
        alpha = float(alpha)
        r2 = U**2 + V**2
        kernel_raw = np.exp(-0.5 * alpha**2 * r2)
    else:
        ah, av = alpha
        kernel_raw = np.exp(-0.5 * (ah**2 * U**2 + av**2 * V**2))
    kernel = kernel_raw / (np.sum(kernel_raw) + EPSILON)
    return kernel

def gaussian_psf_deriv_alpha(alpha: Union[float, Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Derivative of normalized Gaussian PSF w.r.t. alpha.
    If alpha is scalar: derivative w.r.t. that scalar.
    If alpha is tuple (ah, av): returns tuple (dh/dah, dh/dav).
    """
    h, w = shape
    u = np.arange(h) - h // 2
    v = np.arange(w) - w // 2
    U, V = np.meshgrid(v, u)
    
    if isinstance(alpha, (int, float)):
        alpha = float(alpha)
        r2 = U**2 + V**2
        kernel_raw = np.exp(-0.5 * alpha**2 * r2)
        S = np.sum(kernel_raw) + EPSILON
        kernel_norm = kernel_raw / S
        d_raw = -alpha * r2 * kernel_raw
        dS = np.sum(d_raw)
        d_norm = (d_raw * S - kernel_raw * dS) / (S**2)
        return d_norm
    else:
        ah, av = alpha
        kernel_raw = np.exp(-0.5 * (ah**2 * U**2 + av**2 * V**2))
        S = np.sum(kernel_raw) + EPSILON
        kernel_norm = kernel_raw / S
        
        # d/dah
        d_raw_ah = -ah * U**2 * kernel_raw
        dS_ah = np.sum(d_raw_ah)
        d_norm_ah = (d_raw_ah * S - kernel_raw * dS_ah) / (S**2)
        
        # d/dav
        d_raw_av = -av * V**2 * kernel_raw
        dS_av = np.sum(d_raw_av)
        d_norm_av = (d_raw_av * S - kernel_raw * dS_av) / (S**2)
        
        return d_norm_ah, d_norm_av

def project_param(value: Union[float, np.ndarray], bounds: Tuple[float, float]) -> Union[float, np.ndarray]:
    """Project scalar or array onto [min, max]."""
    lo, hi = bounds
    if isinstance(value, np.ndarray):
        return np.clip(value, lo, hi)
    else:
        return max(lo, min(value, hi))

def tv_norm(x: np.ndarray) -> float:
    """Isotropic TV norm."""
    dx, dy = compute_spatial_gradient(x)
    return np.sum(np.sqrt(dx**2 + dy**2))

def estimate_lipschitz_alpha(alpha: Union[float, Tuple[float, float]], 
                             kernel_shape: Tuple[int, int],
                             image_shape: Tuple[int, int],
                             sigma2_min: float) -> float:
    """
    Estimate Lipschitz constant L_α = ||H(α)||^2 / σ²_min.
    Uses maximum singular value via OTF.
    """
    h = gaussian_psf(alpha, kernel_shape)
    F_h = psf2otf(h, image_shape)
    max_sv = np.max(np.abs(F_h))
    return (max_sv**2) / sigma2_min