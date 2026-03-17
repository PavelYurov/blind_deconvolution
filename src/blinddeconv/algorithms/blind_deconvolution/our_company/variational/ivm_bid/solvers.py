import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

from .utils import (
    psf2otf,
    otf2psf,
    project_kernel,
    EPSILON,
)

def solve_image_tikhonov(
    g: np.ndarray,
    h: np.ndarray,
    lambda_f: float,
    F_ops: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Solves image step using H1 (Tikhonov) regularization as in Laaziri et al. 2022.
    Equation: (H*H + lambda_f*D*D) f = H*g
    This is closed-form, fast, and avoids 'cartoon' artifacts.
    """
    H, W = g.shape
    _, _, F_grad_sq = F_ops # |Dx|^2 + |Dy|^2

    F_g = fft2(g)
    F_h = psf2otf(h, (H, W))
    
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2

    # Eq. 23 from Laaziri et al. 2022 (simplified notation)
    # Numerator: H_conj * G
    # Denominator: |H|^2 + lambda_f * |D|^2
    
    numer = F_h_conj * F_g
    denom = F_h_sq + lambda_f * F_grad_sq + EPSILON
    
    f = np.real(ifft2(numer / denom))
    
    # Projection to valid range [0, 1]
    return np.clip(f, 0.0, 1.0)

def solve_kernel_gradient_domain(
    g: np.ndarray,
    f: np.ndarray,
    kernel_shape: Tuple[int, int],
    lambda_h: float,
    F_ops: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    Solves for kernel in GRADIENT domain.
    Minimizes: || grad(f)*h - grad(g) ||^2 + lambda ||h||^2
    """
    F_dx, F_dy, _ = F_ops
    
    # Compute gradients in frequency domain
    F_g = fft2(g)
    F_f = fft2(f)
    
    # Gradient of Image and Gradient of Blurry Observation
    F_gx = F_dx * F_g
    F_gy = F_dy * F_g
    
    F_fx = F_dx * F_f
    F_fy = F_dy * F_f
    
    # Analytical solution for least squares
    # We combine X and Y gradients to robustness
    numer = (np.conj(F_fx) * F_gx) + (np.conj(F_fy) * F_gy)
    denom = (np.abs(F_fx)**2 + np.abs(F_fy)**2) + lambda_h + EPSILON
    
    F_h = numer / denom
    h = otf2psf(F_h, kernel_shape)
    
    # --- Post-Processing ---
    # 1. Thresholding to remove noise floor (very important for stability)
    max_val = np.max(h)
    # Очищаем все, что меньше 5% от пика - убирает "шум"
    h[h < 0.05 * max_val] = 0.0
    
    # 2. Constraints
    h = project_kernel(h)
    
    return h