"""
EP Solvers: Image EP (E-Step) and Kernel PGD (M-Step)
With Strategy III (RBMC) Variance Estimation and Moment Matching.
Numeric stability fixes applied.
"""
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.special import ndtr, log_ndtr
from typing import Tuple, Dict
from .utils import psf2otf, EPSILON

def update_ep_sites(
    mu_cav: np.ndarray, 
    v_cav: np.ndarray, 
    lambda_tv: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the moments of the tilted distribution and updates site parameters 
    for a Laplace prior f(x) = exp(-lambda |x|).
    
    Robust implementation to prevent overflow in exp().
    """

    v_cav = np.maximum(v_cav, 1e-10)
    sqrt_v = np.sqrt(v_cav)
    

    z_minus = -(mu_cav + v_cav * lambda_tv) / sqrt_v
    z_plus  =  (mu_cav - v_cav * lambda_tv) / sqrt_v
    

    c_minus =  lambda_tv * mu_cav
    c_plus  = -lambda_tv * mu_cav
    

    log_I_minus = c_minus + log_ndtr(z_minus)
    log_I_plus  = c_plus  + log_ndtr(z_plus)
    
    
    log_ratio = log_I_plus - log_I_minus

    w_plus = np.zeros_like(log_ratio)
    
    mask_pos = log_ratio > 0
    w_plus[mask_pos] = 1.0 / (1.0 + np.exp(-log_ratio[mask_pos]))
    
    mask_neg = ~mask_pos
    exp_lr = np.exp(log_ratio[mask_neg])
    w_plus[mask_neg] = exp_lr / (1.0 + exp_lr)
    
    w_minus = 1.0 - w_plus
    
    
    def get_ratio_phi_Phi_robust(z):
        """
        Computes R(z) = phi(z) / Phi(z) = exp(log_phi - log_Phi)
        Robust against overflow for large negative z.
        As z -> -inf, R(z) -> |z|
        As z -> +inf, R(z) -> 0
        """
        res = np.zeros_like(z)
        mask_neg = z < -30
        res[mask_neg] = -z[mask_neg] 
        
        mask_ok = ~mask_neg
        if np.any(mask_ok):
            z_ok = z[mask_ok]
            log_phi = -0.5 * z_ok**2 - 0.5 * np.log(2 * np.pi)
            log_Phi = log_ndtr(z_ok)
            res[mask_ok] = np.exp(log_phi - log_Phi)
            
        return res

    R_minus = get_ratio_phi_Phi_robust(z_minus)
    R_plus  = get_ratio_phi_Phi_robust(z_plus)
    
    mu_minus = mu_cav + v_cav * lambda_tv
    mu_plus  = mu_cav - v_cav * lambda_tv
    
    E_minus = mu_minus - sqrt_v * R_minus
    E_plus  = mu_plus  + sqrt_v * R_plus
    

    mu_hat = w_minus * E_minus + w_plus * E_plus
    

    E2_minus = v_cav + mu_minus**2 - sqrt_v * mu_minus * R_minus
    E2_plus  = v_cav + mu_plus**2  + sqrt_v * mu_plus * R_plus
    
    E2_hat = w_minus * E2_minus + w_plus * E2_plus
    v_hat = E2_hat - mu_hat**2
    
    v_hat = np.maximum(v_hat, 1e-10)
    

    gamma_site = (1.0 / v_hat) - (1.0 / v_cav)
    

    beta_site  = (mu_hat / v_hat) - (mu_cav / v_cav)
    

    if np.any(~np.isfinite(gamma_site)) or np.any(~np.isfinite(beta_site)):

        gamma_site = np.nan_to_num(gamma_site, nan=lambda_tv, posinf=lambda_tv, neginf=lambda_tv)
        beta_site = np.nan_to_num(beta_site, nan=0.0)
    
    return gamma_site, beta_site, mu_hat, v_hat

def solve_image_ep(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    noise_sigma: float,
    lambda_tv: float,
    ep_iter: int,
    F_ops: Tuple[np.ndarray, np.ndarray, np.ndarray],
    ep_state: Dict[str, np.ndarray] = None,
    strategy: str = 'fast',
    num_probes: int = 10,
    damping: float = 0.8
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """
    E-Step (Mean & Variance).
    """
    H, W = y.shape
    F_dx, F_dy, F_grad_sq = F_ops
    
    if ep_state is None:
        ep_state = {
            'gamma_h': np.full((H, W), lambda_tv, dtype=np.float64),
            'gamma_v': np.full((H, W), lambda_tv, dtype=np.float64),
            'beta_h': np.zeros((H, W), dtype=np.float64),
            'beta_v': np.zeros((H, W), dtype=np.float64)
        }
    
    gamma_h = ep_state['gamma_h']
    gamma_v = ep_state['gamma_v']
    beta_h = ep_state['beta_h']
    beta_v = ep_state['beta_v']
    
    F_y = fft2(y)
    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h)**2
    
    alpha = 1.0 / (noise_sigma**2 + EPSILON) 
    
    mu_x = x_init.copy()
    var_x_mean = 0.0
    
    for _ in range(ep_iter):

        avg_gamma_h = np.mean(gamma_h)
        avg_gamma_v = np.mean(gamma_v)

        avg_gamma_h = max(avg_gamma_h, 1e-6)
        avg_gamma_v = max(avg_gamma_v, 1e-6)
        
        F_LHS = alpha * F_h_sq + avg_gamma_h * (np.abs(F_dx)**2) + avg_gamma_v * (np.abs(F_dy)**2)
        
        rhs_freq = alpha * F_h_conj * F_y + \
                   np.conj(F_dx) * fft2(beta_h) + \
                   np.conj(F_dy) * fft2(beta_v)
        

        safe_LHS = F_LHS + 1e-6 
        
        mu_x = np.real(ifft2(rhs_freq / safe_LHS))
        

        inv_hessian_freq = 1.0 / safe_LHS 
        
        if strategy == 'fast':
            val_dx = np.mean(np.real(ifft2(np.abs(F_dx)**2 * inv_hessian_freq)))
            val_dy = np.mean(np.real(ifft2(np.abs(F_dy)**2 * inv_hessian_freq)))
            v_dx = np.full((H, W), val_dx)
            v_dy = np.full((H, W), val_dy)
            var_x_mean = np.mean(np.real(ifft2(inv_hessian_freq)))
            
        elif strategy == 'rbmc':
            accum_dx = np.zeros((H, W))
            accum_dy = np.zeros((H, W))
            spec_std = np.sqrt(np.maximum(inv_hessian_freq, 0))
            
            for _ in range(num_probes):
                noise = np.random.randn(H, W) + 1j * np.random.randn(H, W)
                noise_spatial = np.random.randn(H, W)
                noise_freq = fft2(noise_spatial)
                
                sample_x_freq = noise_freq * spec_std
                g_dx = np.real(ifft2(F_dx * sample_x_freq))
                g_dy = np.real(ifft2(F_dy * sample_x_freq))
                accum_dx += g_dx**2
                accum_dy += g_dy**2
            
            v_dx = accum_dx / num_probes
            v_dy = accum_dy / num_probes
            var_x_mean = np.mean(np.real(ifft2(inv_hessian_freq)))
        else:
            v_dx = np.full((H, W), 1e-3)
            v_dy = np.full((H, W), 1e-3)
            var_x_mean = 1e-3

        g_x_h = np.real(ifft2(F_dx * fft2(mu_x)))
        g_x_v = np.real(ifft2(F_dy * fft2(mu_x)))
        
        def compute_site_updates(mu_marg_g, v_marg_g, gamma_old, beta_old):
            v_marg_g = np.maximum(v_marg_g, 1e-12)
            prec_marg = 1.0 / v_marg_g
            prec_cav = prec_marg - gamma_old
            
            prec_cav = np.maximum(prec_cav, 1e-6)
            v_cav = 1.0 / prec_cav
            mu_cav = v_cav * (mu_marg_g * prec_marg - beta_old)
            
            g_new, b_new, _, _ = update_ep_sites(mu_cav, v_cav, lambda_tv)
            return g_new, b_new

        gh_new, bh_new = compute_site_updates(g_x_h, v_dx, gamma_h, beta_h)
        gv_new, bv_new = compute_site_updates(g_x_v, v_dy, gamma_v, beta_v)
        
        gamma_h = damping * gh_new + (1 - damping) * gamma_h
        gamma_v = damping * gv_new + (1 - damping) * gamma_v
        beta_h = damping * bh_new + (1 - damping) * beta_h
        beta_v = damping * bv_new + (1 - damping) * beta_v
        
        gamma_h = np.maximum(gamma_h, 1e-6)
        gamma_v = np.maximum(gamma_v, 1e-6)

    ep_state['gamma_h'] = gamma_h
    ep_state['gamma_v'] = gamma_v
    ep_state['beta_h'] = beta_h
    ep_state['beta_v'] = beta_v
    
    return mu_x, ep_state, var_x_mean

def estimate_uncertainty(
    h: np.ndarray,
    noise_sigma: float,
    lambda_eff: float,
    shape: Tuple[int, int],
    F_grad_sq: np.ndarray,
    strategy: str = 'fast'
) -> Tuple[float, np.ndarray]:
    """
    Computes approximation of posterior covariance for use in M-step.
    """
    H, W = shape
    F_h = psf2otf(h, (H, W))
    F_h_sq = np.abs(F_h)**2
    
    alpha = 1.0 / (noise_sigma**2 + EPSILON)
    inv_hessian = 1.0 / (alpha * F_h_sq + lambda_eff * F_grad_sq + EPSILON)
    
    uncertainty = float(np.mean(inv_hessian))
    r = np.real(ifft2(inv_hessian))
    return uncertainty, r

def non_neg_ep(x: np.ndarray, uncertainty: float) -> np.ndarray:
    return np.maximum(x, 0.0)

def solve_kernel_pgd(
    y: np.ndarray,
    x: np.ndarray,
    h_init: np.ndarray,
    D_x: np.ndarray,       
    inner_iter: int,
    momentum: float = 0.9,
    var_x: float = 0.0
) -> np.ndarray:
    """
    M-Step (Kernel).
    """
    H, W = y.shape
    kh, kw = h_init.shape
    
    F_x = fft2(x)
    F_y = fft2(y)
    
    F_xx = np.abs(F_x)**2
    F_xy = np.conj(F_x) * F_y
    
    cov_reg = var_x * (H * W) 
    max_eig_xx = np.max(F_xx) + cov_reg
    
    try:
        max_eig_Dx = np.linalg.eigvalsh(D_x).max()
    except:
        max_eig_Dx = 1.0
        
    L = 2.0 * max_eig_xx + 2.0 * max_eig_Dx
    step_size = 1.0 / (L + 1e-12)
    
    h = h_init.copy()
    h_old = h.copy()
    
    cy, cx = H//2, W//2
    
    for _ in range(inner_iter):
        y_mom = h + momentum * (h - h_old)
        h_old = h.copy()
        curr_h = y_mom
        
        F_h = psf2otf(curr_h, (H, W))
        grad_freq = 2.0 * ((F_xx + cov_reg) * F_h - F_xy)
        
        grad_spatial = np.real(ifft2(grad_freq))
        grad_spatial = np.roll(grad_spatial, kh//2, axis=0)
        grad_spatial = np.roll(grad_spatial, kw//2, axis=1)
        grad_h_data = grad_spatial[:kh, :kw]
        
        h_flat = curr_h.flatten()
        grad_reg_flat = 2.0 * (D_x @ h_flat)
        grad_h_reg = grad_reg_flat.reshape((kh, kw))
        
        total_grad = grad_h_data + grad_h_reg
        
        h = curr_h - step_size * total_grad
        
        h = np.maximum(h, 0.0)
        h_sum = np.sum(h)
        if h_sum > 1e-12:
            h /= h_sum
        else:
            h = np.ones_like(h) / (kh * kw)
            
    return h
