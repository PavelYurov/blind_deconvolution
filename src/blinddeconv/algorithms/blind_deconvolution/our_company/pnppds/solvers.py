"""
Primal-Dual Solvers for Blind Deconvolution.
Based on Condat (2013) Alg 3.1 and Chambolle-Pock (2011) Alg 1.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from .utils import (
    psf2otf, apply_conv, apply_corr, compute_gradient, compute_divergence,
    prox_tv_dual, prox_box, project_simplex, threshold_kernel,
    huber_gradient, EPSILON
)

def solve_image_pds(
    y: np.ndarray, 
    h: np.ndarray, 
    x_init: np.ndarray,
    alpha: float,       
    lambda_tv: float,   
    huber_delta: float, 
    num_iter: int,
    F_dh: np.ndarray, 
    F_dv: np.ndarray, 
    rho: float = 1.0
) -> np.ndarray:
    """
    Image Estimation: Condat's Primal-Dual Splitting (2013).
    
    Problem: min_x  F(x) + G(x) + H(Lx)
    
    F(x) = alpha * Huber(h*x - y)  [Smooth, Lipschitz gradient]
    G(x) = Indicator_{[0,1]}(x)    [Proximable]
    H(Lx) = lambda_tv * ||Grad x||_1
    """
    
    H, W = y.shape
    F_h = psf2otf(h, (H, W))
    max_h_sq = np.max(np.abs(F_h) ** 2)
    beta = alpha * max_h_sq
    

    L_sq = 8.0
    
    sigma = 1.0 / (2.0 * L_sq)
    tau = 0.9 / (beta / 2.0 + sigma * L_sq) 


    x = x_init.copy()
    ph = np.zeros_like(x)
    pv = np.zeros_like(x) 

    for _ in range(num_iter):
        Hx = apply_conv(x, F_h)
        residual = Hx - y
        
        grad_fidelity = huber_gradient(residual, huber_delta)
        
        grad_f = alpha * apply_corr(grad_fidelity, F_h)

        div_p = compute_divergence(ph, pv, F_dh, F_dv)

        x_tilde = prox_box(x - tau * (grad_f + div_p))

        x_bar = 2.0 * x_tilde - x

        gh, gv = compute_gradient(x_bar, F_dh, F_dv)
        ph_new = ph + sigma * gh
        pv_new = pv + sigma * gv
        
        ph_new, pv_new = prox_tv_dual(ph_new, pv_new, lambda_tv)

        x = rho * x_tilde + (1 - rho) * x
        ph = rho * ph_new + (1 - rho) * ph
        pv = rho * pv_new + (1 - rho) * pv

    return x

def solve_kernel_pds(
    y: np.ndarray, 
    x: np.ndarray, 
    h_init: np.ndarray,
    alpha: float,       
    num_iter: int,
    theta: float = 1.0,
    kernel_threshold: float = 0.0
) -> np.ndarray:
    """
    Kernel Estimation: Chambolle-Pock (2011).
    
    Problem: min_h alpha/2 * ||x*h - y||^2 + Indicator_Simplex(h)
    
    Here we keep L2 norm because kernel estimation is generally 
    an over-determined system and less prone to outlier ringing than image step.
    """
    H, W = y.shape
    kh, kw = h_init.shape
    
    F_x = fft2(x)
    F_x_conj = np.conj(F_x)
    
    K_norm = np.sqrt(np.max(np.abs(F_x)**2)) + EPSILON
    
    sigma_pd = 0.9 / K_norm
    tau_pd = 0.9 / K_norm
    
    h = h_init.copy()
    h_bar = h.copy()
    v = np.zeros((H, W)) 

    for _ in range(num_iter):

        F_hbar = psf2otf(h_bar, (H, W))
        Kh_bar = np.real(ifft2(F_x * F_hbar))
        
        v_tilde = v + sigma_pd * Kh_bar
        v_new = (v_tilde - sigma_pd * y) / (1.0 + sigma_pd / alpha)
        
        KTv_full = np.real(ifft2(F_x_conj * fft2(v_new)))
        
        KTv_crop = np.roll(KTv_full, kh // 2, axis=0)
        KTv_crop = np.roll(KTv_crop, kw // 2, axis=1)
        KTv_crop = KTv_crop[:kh, :kw]
        
        h_new = project_simplex(h - tau_pd * KTv_crop)

        h_bar = h_new + theta * (h_new - h)
        
        h = h_new
        v = v_new


    if kernel_threshold > 0:
        h = threshold_kernel(h, kernel_threshold)
        h = project_simplex(h)
        
    return h