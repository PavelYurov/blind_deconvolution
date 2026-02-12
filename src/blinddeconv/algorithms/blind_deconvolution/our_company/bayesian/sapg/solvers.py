import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, Callable
from .utils import psf2otf, compute_spatial_gradient, tv_prox, gaussian_psf, EPSILON, project_param, soft_threshold

def data_fidelity_grad(y: np.ndarray, alpha: float, sigma2: float, x: np.ndarray, psf_func: Callable) -> np.ndarray:
    """
    Gradient of f_{y,alpha,sigma2}(x) = (1/(2 sigma2)) ||y - H(alpha) x||^2
    """
    h = psf_func(alpha, (5, 5))  # Assume small kernel support
    F_h = psf2otf(h, y.shape)
    F_x = fft2(x)
    conv = np.real(ifft2(F_h * F_x))
    residual = conv - y
    F_res = fft2(residual)
    grad = np.real(ifft2(np.conj(F_h) * F_res)) / sigma2
    return grad

def data_fidelity_alpha_grad(y: np.ndarray, alpha: float, sigma2: float, x: np.ndarray, psf_func: Callable, psf_alpha_deriv: Callable) -> float:
    """
    Gradient w.r.t. alpha of f_{y,alpha,sigma2}(x)
    For Gaussian PSF, derivative is analytic.
    """
    h = psf_func(alpha, (5, 5))
    dh_dalpha = psf_alpha_deriv(alpha, (5, 5))
    F_h = psf2otf(h, y.shape)
    F_x = fft2(x)
    conv_hx = np.real(ifft2(F_h * F_x))
    F_dh = psf2otf(dh_dalpha, y.shape)
    conv_dhx = np.real(ifft2(F_dh * F_x))
    residual = conv_hx - y
    grad_alpha = np.sum(residual * conv_dhx) / sigma2
    return grad_alpha

def data_fidelity_sigma2_grad(y: np.ndarray, alpha: float, sigma2: float, x: np.ndarray, psf_func: Callable) -> float:
    """
    Gradient w.r.t. sigma2 of f_{y,alpha,sigma2}(x)
    """
    h = psf_func(alpha, (5, 5))
    F_h = psf2otf(h, y.shape)
    F_x = fft2(x)
    conv = np.real(ifft2(F_h * F_x))
    residual = conv - y
    grad_sigma2 = -0.5 * np.sum(residual**2) / sigma2**2
    return grad_sigma2

def myula_sampler(
    y: np.ndarray,
    alpha: float,
    sigma2: float,
    theta: float,
    x_init: np.ndarray,
    gamma: float,
    lam: float,
    m: int,
    burn_in: int,
    psf_func: Callable,
    prox_g: Callable,
    grad_f: Callable
) -> np.ndarray:
    """
    Moreau-Yosida regularized unadjusted Langevin MCMC (MYULA) sampler.
    Samples from approximate posterior.
    Returns array of m samples after burn-in.
    """
    x = x_init.copy()
    samples = []
    for k in range(m + burn_in):
        grad = grad_f(y, alpha, sigma2, x, psf_func)
        prox_term = prox_g(x, lam * theta)
        x = (1 - gamma * lam) * x - gamma * grad + gamma * lam * prox_term + np.sqrt(2 * gamma) * np.random.randn(*x.shape)
        if k >= burn_in:
            samples.append(x.copy())
    return np.array(samples)

def sapg_update_theta(samples_posterior: np.ndarray, samples_prior: np.ndarray, theta: float, delta: float, d: int, q: float = 1.0, bounds: Tuple[float, float] = (1e-3, 1.0)) -> float:
    """
    SAPG update for theta.
    """
    m = len(samples_posterior)
    avg_g_post = np.mean([tv_norm(s) for s in samples_posterior])
    avg_g_prior = np.mean([tv_norm(s) for s in samples_prior])
    delta_theta = delta * (avg_g_prior - avg_g_post)
    theta_new = project_param(theta + delta_theta, bounds)
    return theta_new

def sapg_update_alpha(samples: np.ndarray, y: np.ndarray, alpha: float, sigma2: float, delta: float, psf_func: Callable, psf_alpha_deriv: Callable, bounds: Tuple[float, float] = (0.1, 10.0)) -> float:
    """
    SAPG update for alpha.
    """
    m = len(samples)
    avg_grad_alpha = np.mean([data_fidelity_alpha_grad(y, alpha, sigma2, s, psf_func, psf_alpha_deriv) for s in samples])
    alpha_new = project_param(alpha - delta * avg_grad_alpha, bounds)
    return alpha_new

def sapg_update_sigma2(samples: np.ndarray, y: np.ndarray, alpha: float, sigma2: float, delta: float, d: int, psf_func: Callable, bounds: Tuple[float, float] = (1e-4, 1.0)) -> float:
    """
    SAPG update for sigma2.
    """
    m = len(samples)
    avg_grad_sigma2 = np.mean([data_fidelity_sigma2_grad(y, alpha, sigma2, s, psf_func) for s in samples]) + d / (2 * sigma2)
    sigma2_new = project_param(sigma2 - delta * avg_grad_sigma2, bounds)
    return sigma2_new

def tv_norm(x: np.ndarray) -> float:
    """Total Variation norm g(x) = ||grad x||_1"""
    dx, dy = compute_spatial_gradient(x)
    return np.sum(np.sqrt(dx**2 + dy**2))

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