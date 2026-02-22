import numpy as np
import torch
from .utils_ml import get_blur_operator_ml, proj_l2_ball, proj_box

def solve_image_pnp_pds(
    y_obs, 
    k_curr, 
    x_init, 
    denoiser_model, 
    params
):
    """
    ML PnP-PDS solver (DnCNN).
    Used for final non-blind image restoration.
    
    Matches the original Gaussian-PnPPDS iteration from iteration.py:
        x_n   = denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
        y_n   = y_n + gamma2 * phi(2*x_n - x_prev) 
                    - gamma2 * proj_l2_ball(y_n/gamma2 + phi(2*x_n - x_prev), ...)
        y2_n  = y2_n + gamma2 * (2*x_n - x_prev) 
                    - gamma2 * proj_box(y2_n/gamma2 + (2*x_n - x_prev))
    """
    # Parameters (matched to original main_gaussian.py defaults)
    gamma1 = params.get('gamma1', 0.5)
    gamma2 = params.get('gamma2', 0.99)
    alpha_n = params.get('alpha_n', 0.82)
    noise_std = params.get('noise_sigma', 0.01)
    max_iter = params.get('ml_iter', 200)
    
    phi, adj_phi = get_blur_operator_ml(x_init, k_curr)
    
    x_n = x_init.copy()
    y_n = np.zeros_like(x_n)
    y2_n = np.zeros_like(x_n)
    
    for i in range(max_iter):
        x_prev = x_n
        
        # 1. Primal Update: denoise(x - gamma1 * (adj_phi(y) + y2))
        # Matches original: x_n = denoiser.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
        x_n = denoiser_model.denoise(x_n - gamma1 * (adj_phi(y_n) + y2_n))
        
        # Relaxation: x_bar = 2*x_n - x_prev
        x_bar = 2 * x_n - x_prev
        
        # 2. Dual Update (Data Fidelity)
        # Original: y_n = y_n + gamma2*phi(x_bar); y_n = y_n - gamma2*proj_l2_ball(y_n/gamma2,...)
        y_n = y_n + gamma2 * phi(x_bar)
        y_n = y_n - gamma2 * proj_l2_ball(y_n / gamma2, alpha_n, noise_std, 0, y_obs, r=1)
        
        # 3. Dual Update (Box Constraint)
        # Original: y2_n = y2_n + gamma2*x_bar; y2_n = y2_n - gamma2*proj_C(y2_n/gamma2)
        y2_n = y2_n + gamma2 * x_bar
        y2_n = y2_n - gamma2 * proj_box(y2_n / gamma2)
        
    return x_n