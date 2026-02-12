import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import cg
from typing import Tuple
from .utils import psf2otf, soft_threshold, EPSILON, truncated_gaussian_moments

def solve_image_hqs(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    noise_sigma: float,
    lambda_tv: float,
    beta_max: float,
    inner_iter: int,
    F_ops: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    E-Step (Mean): Solves MAP estimation for Image using Half-Quadratic Splitting.
    Minimizes: ||y - h*x||^2 / (2*sigma^2) + lambda_tv * ||grad x||_1
    """
    H, W = y.shape
    F_dx, F_dy, F_grad_sq = F_ops
    
    # Precompute constant FFT terms
    F_y = fft2(y)
    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h)**2
    
    alpha = 1.0 / (noise_sigma**2 + EPSILON)
    
    x = x_init.copy()
    z_x = np.zeros_like(x)
    z_y = np.zeros_like(x)
    
    beta = 1.0 # Starting beta
    
    while beta < beta_max:
        for _ in range(inner_iter):
            # 1. Linear System Solve (FFT)
            # (alpha * H'H + beta * D'D) x = alpha * H'y + beta * D'z
            rhs = alpha * F_h_conj * F_y + beta * (
                np.conj(F_dx) * fft2(z_x) + np.conj(F_dy) * fft2(z_y)
            )
            lhs = alpha * F_h_sq + beta * F_grad_sq
            
            x = np.real(ifft2(rhs / (lhs + EPSILON)))
            x = np.maximum(x, 0.0) # Projection
            
            # 2. Shrinkage Step
            grad_x = np.real(ifft2(F_dx * fft2(x)))
            grad_y = np.real(ifft2(F_dy * fft2(x)))
            
            thresh = lambda_tv / beta
            z_x = soft_threshold(grad_x, thresh)
            z_y = soft_threshold(grad_y, thresh)
        
        beta *= 2.0 # Continuation scheme
        
    return x

def estimate_uncertainty(
    h: np.ndarray,
    noise_sigma: float,
    lambda_eff: float,
    shape: Tuple[int, int],
    F_grad_sq: np.ndarray,
    strategy: str = 'fast',
    num_probes: int = 10
) -> Tuple[float, np.ndarray]:
    """
    E-Step (Variance): Estimates marginal variance using Spectral Approximation or RBMC.
    Returns scalar uncertainty (mean variance) and the full autocovariance function r (from spectral).
    """
    H, W = shape
    F_h = psf2otf(h, (H, W))
    F_h_sq = np.abs(F_h)**2
    
    alpha = 1.0 / (noise_sigma**2 + EPSILON)
    
    # Inverse Hessian Spectrum
    inv_hessian = 1.0 / (alpha * F_h_sq + lambda_eff * F_grad_sq + EPSILON)
    r = np.real(ifft2(inv_hessian))
    
    if strategy == 'fast':
        # Scalar uncertainty (mean of diagonal)
        uncertainty = float(np.mean(inv_hessian))
    elif strategy == 'rbmc':
        def matvec(v):
            return alpha * np.real(ifft2(F_h_sq * fft2(v))) + lambda_eff * np.real(ifft2(F_grad_sq * fft2(v)))
        
        accum = np.zeros(shape)
        for _ in range(num_probes):
            r_probe = np.random.randn(H, W)
            v, info = cg(matvec, r_probe, atol=1e-6, maxiter=200)
            if info != 0:
                print(f"CG did not converge: {info}")
            accum += v * r_probe
        diag_est = accum / num_probes
        uncertainty = float(np.mean(diag_est))
    else:
        raise ValueError("Unknown strategy")
    
    return uncertainty, r

def non_neg_ep(
    x: np.ndarray,
    uncertainty: float
) -> np.ndarray:
    """
    Incorporate non-negativity constraint using EP approximation.
    Updates the mean image x with soft truncation.
    Assumes uniform variance.
    """
    H, W = x.shape
    v = uncertainty
    for i in range(H):
        for j in range(W):
            m = x[i, j]
            mean_t, var_t = truncated_gaussian_moments(m, v)
            if var_t < EPSILON:
                sigma_inv = 1e10
                mu_site = mean_t * sigma_inv
            else:
                sigma_inv = 1 / var_t - 1 / v
                mu_site = mean_t / var_t - m / v
            v_new = 1 / (1 / v + sigma_inv + EPSILON)
            m_new = v_new * (m / v + mu_site)
            x[i, j] = m_new
    return x

def solve_kernel_pgd(
    y: np.ndarray,
    x: np.ndarray,
    h_init: np.ndarray,
    D_x: np.ndarray,       
    inner_iter: int,
    momentum: float = 0.9   
) -> np.ndarray:
    """
    M-Step: Solves Kernel estimation using Projected Gradient Descent (PGD).
    Minimizes: ||y - x*h||^2 + h^T D_x h
    Subject to: h >= 0, sum(h) = 1
    
    Uses adaptive step size based on Lipschitz constant for stability.
    D_x is the full regularization matrix incorporating correlations.
    """
    H, W = y.shape
    kh, kw = h_init.shape
    
    # Precompute X terms
    F_x = fft2(x)
    F_y = fft2(y)
    F_xx = np.abs(F_x)**2
    F_xy = np.conj(F_x) * F_y
    
    # Calculate Lipschitz constant for step size stability
    # L = 2 * max_eig(X^T X) + 2 * max_eig(D_x)
    max_eig_xx = np.max(F_xx)
    max_eig_Dx = np.linalg.eigvalsh(D_x).max()
    L = 2.0 * max_eig_xx + 2.0 * max_eig_Dx
    step_size = 1.0 / (L + EPSILON)
    
    h = h_init.copy()
    h_old = h.copy()
    
    for _ in range(inner_iter):
        # Nesterov momentum
        y_mom = h + momentum * (h - h_old)
        h_old = h.copy()

        # 1. Gradient Calculation
        F_h = psf2otf(h, (H, W))
        
        # Grad_fidelity = 2 * (X^T X h - X^T y)
        grad_freq = 2.0 * (F_xx * F_h - F_xy)
        grad_spatial = np.real(ifft2(grad_freq))
        
        # Crop to kernel support and center
        grad_spatial = np.roll(grad_spatial, kh//2, axis=0)
        grad_spatial = np.roll(grad_spatial, kw//2, axis=1)
        grad_h = grad_spatial[:kh, :kw]
        
        # Grad_regularization = 2 * D_x h (flattened)
        h_flat = h.flatten()
        grad_reg_flat = 2.0 * (D_x @ h_flat)
        grad_h += grad_reg_flat.reshape((kh, kw))
        
        # 2. Gradient Descent Step
        h = h - step_size * grad_h
        
        # 3. Projection (Simplex Constraint)
        h = np.maximum(h, 0.0) # Non-negativity
        
        h_sum = np.sum(h)
        if h_sum > EPSILON:
            h /= h_sum
        else:
            h = np.ones_like(h) / (kh * kw)
            
    return h