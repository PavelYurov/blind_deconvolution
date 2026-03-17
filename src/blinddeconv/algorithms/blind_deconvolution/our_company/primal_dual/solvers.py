"""
solvers.py

Mathematical solvers for Logarithmic Prior Blind Deconvolution.
Revised to use Gradient-Domain Kernel Estimation to prevent 'blob' artifacts.
"""

import numpy as np
from .utils import psf2otf, compute_gradient, compute_divergence

def solve_scalar_prox_log(z_norm: np.ndarray, sigma: float, mu: float, 
                          epsilon: float, p: float) -> np.ndarray:
    """
    Solves the scalar shrinkage problem for the Logarithmic Prior using Newton's method.
    Target equation derived from setting derivative to zero.
    
    The subproblem is equivalent to:
    min_{rho}  (1 / 2*sigma) * (rho - 1)^2 * ||z||^2 + mu * log(rho^2 ||z||^2 + eps^2)
    
    Note: 'mu' here represents the effective weight of the prior relative to the prox quadratic term.
    """
    z_sq = z_norm**2
    # Avoid numerical instability for very small gradients
    mask = z_sq > 1e-8
    
    rho = np.ones_like(z_norm)
    
    # We solve f(rho) = 0
    # where f(rho) comes from d/drho of the objective
    
    # Pre-calculate constants for masked elements
    # In the paper, the regularization weight is lambda.
    # The PD update separates the fidelity (lambda) and prior (1).
    # The effective weight passed to this solver is scaled by sigma.
    
    # Let's perform the Newton iterations only on relevant pixels
    z2_m = z_sq[mask]
    rho_m = rho[mask]
    eps_sq = epsilon**2
    
    # Optimization: 5 iterations is usually sufficient for quadratic convergence
    for _ in range(5):
        # Current term: rho^2 * z^2 + epsilon^2
        denom = rho_m**2 * z2_m + eps_sq
        
        # Objective derivative (f to minimize)
        # term1 = (1/sigma) * z^2 * (rho - 1)
        # term2 = p * rho * z^2 / denom
        # We divide entire equation by (z^2 / sigma) to simplify:
        # F(rho) = (rho - 1) + (sigma * p * rho) / denom = 0
        
        term_reg = (sigma * p * rho_m) / denom
        f_val = (rho_m - 1.0) + term_reg
        
        # Jacobian (F'(rho))
        # d/drho [ rho ] + d/drho [ C * rho * (A * rho^2 + B)^-1 ]
        # derivative of rho / (A rho^2 + B) is (B - A rho^2) / (A rho^2 + B)^2
        # Here A = z2, B = eps^2
        
        term_deriv = (sigma * p) * (eps_sq - z2_m * rho_m**2) / (denom**2)
        f_prime = 1.0 + term_deriv
        
        # Newton step
        step = f_val / (f_prime + 1e-10)
        rho_m = rho_m - step
        
        # Project to valid range [0, 1] (shrinkage implies 0 <= rho <= 1)
        rho_m = np.clip(rho_m, 0.0, 1.0)
    
    rho[mask] = rho_m
    # For very small z, rho stays 1 (no shrinkage), or 0 if dominated by epsilon.
    # Usually shrinkage -> 0 for noise. Let's force 0 if z is tiny.
    rho[~mask] = 0.0
    
    return rho

def solve_image_primal_dual(
    f: np.ndarray, 
    k: np.ndarray, 
    u_init: np.ndarray, 
    lambda_reg: float, 
    epsilon: float,
    p: float,
    pd_iter: int = 30
) -> np.ndarray:
    """
    Primal-Dual solver for Image u.
    Minimizes: lambda/2 ||k*u - f||^2 + sum log(||grad u||^p + eps)
    """
    H, W = f.shape
    u = u_init.astype(np.float32)
    u_bar = u.copy()
    
    # Dual variables
    q = np.zeros_like(f)         # For data term
    z = np.zeros((2, H, W), dtype=f.dtype) # For gradient term
    
    # Calculate operator norm for step sizes
    # Norm of K is approx 1. Norm of Grad is sqrt(8).
    # L^2 = ||K||^2 + ||Grad||^2 approx 1 + 8 = 9
    L_sq = 9.0
    L = np.sqrt(L_sq)
    
    # Standard P-D step sizes
    tau = 1.0 / L
    sigma = 1.0 / L
    
    k_otf = psf2otf(k, (H, W))
    k_conj = np.conj(k_otf)
    
    for _ in range(pd_iter):
        # --- Dual Update ---
        
        # 1. Data fidelity dual (q)
        # Prox_F*(y) where F(x) = lambda/2 ||x - f||^2
        # Argument: q + sigma * K * u_bar
        Ku_bar = np.real(np.fft.ifft2(k_otf * np.fft.fft2(u_bar)))
        q_in = q + sigma * Ku_bar
        
        # Analytical prox for conjugate of quadratic
        # prox_{sigma F*}(v) = (v - sigma * f) / (1 + sigma / lambda)
        q = (q_in - sigma * f) / (1.0 + sigma / lambda_reg)
        
        # 2. Gradient prior dual (z)
        # Argument: z + sigma * Grad * u_bar
        grad_u_bar = compute_gradient(u_bar)
        z_in = z + sigma * grad_u_bar
        
        # Prox G* via Moreau Identity
        # prox_{sigma G*}(w) = w - sigma * prox_{1/sigma G}(w / sigma)
        # We need to solve the primal scalar problem for:
        # min 1/2||x - w/sigma||^2 + (1/sigma)*Prior(x)
        
        # Compute magnitude of input vector (w/sigma)
        w_scaled = z_in / sigma
        w_norm = np.sqrt(np.sum(w_scaled**2, axis=0))
        
        # Solve scalar shrinkage factor rho
        # Effective weight for log term in this subproblem is 1.0 (relative to 1/sigma scaling)
        # Actually, in solve_scalar_prox_log derivation:
        # The weight 'mu' is the coefficient in front of log. Here it is (1/sigma).
        # Wait, previous logic: "A = mu/sigma".
        # Here: Objective is sigma/2 (...) + 1 * log(...) -> A = 1/sigma * 2 ??
        # Let's trust the consistent derivation: 
        # Prox_tau_phi(y). Here tau = 1/sigma. phi(x) = log(...).
        # We pass sigma_val = 1/sigma to the solver logic? No.
        # Let's use the inputs directly.
        # We want prox of: log(||x||^p + eps). Step size is (1/sigma).
        # call solver with z_norm=w_norm, sigma=(1/sigma), mu=1.0.
        
        rho = solve_scalar_prox_log(w_norm, sigma=(1.0/sigma), mu=1.0, epsilon=epsilon, p=p)
        
        # Primal prox result: x = rho * w_scaled
        # Dual prox result: z = z_in - sigma * x = z_in - sigma * rho * z_in/sigma = z_in * (1 - rho)
        z = z_in * (1.0 - rho)
        
        # --- Primal Update ---
        # u = u - tau * (K* q + div z)
        K_adj_q = np.real(np.fft.ifft2(k_conj * np.fft.fft2(q)))
        div_z = compute_divergence(z)
        
        u_next = u - tau * (K_adj_q + div_z)
        u_next = np.clip(u_next, 0.0, 1.0)
        
        # --- Extrapolation ---
        u_bar = 2.0 * u_next - u
        u = u_next

    return u

def solve_kernel_pgd(
    image: np.ndarray, 
    latent: np.ndarray, 
    k_init: np.ndarray, 
    iters: int = 10
) -> np.ndarray:
    """
    Estimates kernel using Gradient-Domain optimization.
    Minimizing in gradient domain prevents low-frequency bias (blobs).
    
    Problem: min_k || grad(u) * k - grad(f) ||^2 + constraint(k)
    """
    k_est = k_init.copy()
    H, W = image.shape
    kh, kw = k_est.shape
    
    # 1. Compute Gradients
    # Using gradients is CRITICAL for stability in Blind Deconv
    # (Cho & Lee, Xu & Jia, etc. all do this)
    grad_f = compute_gradient(image) # (2, H, W)
    grad_u = compute_gradient(latent) # (2, H, W)
    
    # Convert to frequency domain
    # We solve the system for k in Fourier domain
    # F_x * k = F_y  (where F_x is FFT of grad_u, F_y is FFT of grad_f)
    
    # Sum over x and y gradients
    # grad_f_x, grad_f_y = grad_f[0], grad_f[1]
    # grad_u_x, grad_u_y = grad_u[0], grad_u[1]
    
    # Precompute FFTs
    Gx_u = np.fft.fft2(grad_u[0])
    Gy_u = np.fft.fft2(grad_u[1])
    Gx_f = np.fft.fft2(grad_f[0])
    Gy_f = np.fft.fft2(grad_f[1])
    
    # Helper to pad/unpad kernel
    def pad_k(kern):
        p = np.zeros((H, W), dtype=kern.dtype)
        p[:kh, :kw] = kern
        # Circular shift to center for FFT
        p = np.roll(p, -kh//2, axis=0)
        p = np.roll(p, -kw//2, axis=1)
        return p
    
    def unpad_k(kern_big):
        kern_big = np.roll(kern_big, kh//2, axis=0)
        kern_big = np.roll(kern_big, kw//2, axis=1)
        return kern_big[:kh, :kw]

    # Lipschitz constant estimation
    # L <= max( |Gx_u|^2 + |Gy_u|^2 )
    denom = np.abs(Gx_u)**2 + np.abs(Gy_u)**2
    L = np.max(denom) + 1e-6
    step_size = 1.0 / L
    
    for _ in range(iters):
        K_big = np.fft.fft2(pad_k(k_est))
        
        # Compute gradient of the data term
        # Grad = Sum_i (F_ui * K - F_fi) * conj(F_ui)
        # i in {x, y}
        
        # X-component residual
        R_x = K_big * Gx_u - Gx_f
        # Y-component residual
        R_y = K_big * Gy_u - Gy_f
        
        Grad_freq = R_x * np.conj(Gx_u) + R_y * np.conj(Gy_u)
        Grad_spatial = np.real(np.fft.ifft2(Grad_freq))
        
        grad_k = unpad_k(Grad_spatial)
        
        # Descent
        k_est = k_est - step_size * grad_k
        
        # Constraints
        # 1. Non-negativity
        k_est = np.maximum(k_est, 0)
        
        # 2. Thresholding / Sparsity (CRITICAL)
        # Remove small noise to prevent the blob effect
        max_val = np.max(k_est)
        k_est[k_est < 0.05 * max_val] = 0.0
        
        # 3. Normalization
        s = np.sum(k_est)
        if s > 1e-9:
            k_est /= s
        else:
            k_est = np.ones_like(k_est) / k_est.size
            
    return k_est