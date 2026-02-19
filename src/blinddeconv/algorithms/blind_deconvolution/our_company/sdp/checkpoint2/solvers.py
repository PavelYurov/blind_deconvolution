import numpy as np
from numpy.fft import fft2, ifft2

from .utils import (
    fft_convolve,
    project_kernel,
    threshold_kernel,
    center_kernel,
    tv_gradient,
    edge_taper,
    upsample_kernel,
    build_pyramid,
    kernel_shape_for_level,
    init_gaussian_kernel
)

# Переиспользуем функцию градиентов из прошлого ответа или определим заново, 
# если она была внутри utils (в моем прошлом ответе она была в solvers)
# Я продублирую её здесь для полноты файла.

def compute_euclidean_gradients(y, h, x):
    H_img, W_img = y.shape
    kh, kw = h.shape
    h_pad = np.zeros((H_img, W_img), dtype=np.float64)
    h_pad[:kh, :kw] = h
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)

    F_h = fft2(h_pad)
    F_x = fft2(x)
    F_y = fft2(y)
    F_r = F_y - F_h * F_x # Residual

    grad_x = -np.real(ifft2(np.conj(F_h) * F_r))
    
    grad_h_full = -np.real(ifft2(np.conj(F_x) * F_r))
    grad_h_full = np.roll(grad_h_full, kh // 2, axis=0)
    grad_h_full = np.roll(grad_h_full, kw // 2, axis=1)
    grad_h = grad_h_full[:kh, :kw]
    
    return grad_h, grad_x

def horizontal_projection(grad_h, grad_x, h, x):
    # Scale ambiguity invariance projection
    inner_hg = np.sum(grad_h * h)
    inner_xg = np.sum(grad_x * x)
    norm_sq = np.sum(h ** 2) + np.sum(x ** 2)
    beta = (inner_hg - inner_xg) / (norm_sq + 1e-25)
    return grad_h - beta * h, grad_x + beta * x

def barzilai_borwein_stepsize(s_h, s_x, z_h, z_x, iteration, alpha_min=1e-15, alpha_max=10.0):
    # NOTE: Reduced alpha_max from 1.0/100.0 to 10.0 to prevent explosions
    ss = np.sum(s_h ** 2) + np.sum(s_x ** 2)
    sz = np.sum(s_h * z_h) + np.sum(s_x * z_x)
    zz = np.sum(z_h ** 2) + np.sum(z_x ** 2)

    if sz <= 1e-25 or zz <= 1e-25:
        return alpha_min * 100 # Fallback small step

    alpha_bb1 = ss / sz
    alpha_bb2 = sz / zz

    # Safe guard: if gradient flipped direction drastically (sz < 0), use small step
    if sz < 0:
        return alpha_min

    # Alternate, but prioritize stability
    alpha = alpha_bb1 if iteration % 2 == 0 else alpha_bb2
    return float(np.clip(alpha, alpha_min, alpha_max))

def sdp_single_scale(
    y: np.ndarray,
    h_init: np.ndarray,
    x_init: np.ndarray,
    lambda_tv: float = 0.002,
    max_iter: int = 300,
    tol: float = 1e-6,
    kernel_threshold: float = 0.0,
    verbose: bool = False,
) -> tuple:
    h = h_init.copy()
    x = x_init.copy()
    
    info = {'cost': [], 'kernel_diff': [], 'step_size': []}
    
    h_prev, x_prev = None, None
    g_h_prev, g_x_prev = None, None
    
    # Conservative start
    alpha = 1e-4 

    for it in range(max_iter):
        h_old = h.copy()

        # 1. Gradients
        grad_h, grad_x = compute_euclidean_gradients(y, h, x)

        # 2. TV Regularization
        if lambda_tv > 0.0:
            grad_x = grad_x + lambda_tv * tv_gradient(x)

        # 3. Horizontal Projection
        g_h, g_x = horizontal_projection(grad_h, grad_x, h, x)

        # 4. BB Step Size calculation
        if h_prev is not None:
            s_h = h - h_prev
            s_x = x - x_prev
            z_h = g_h - g_h_prev
            z_x = g_x - g_x_prev
            
            # Dynamic alpha_max based on iteration
            # Early iterations need smaller steps to establish structure
            curr_max = 5.0 if it > 20 else 0.5
            alpha = barzilai_borwein_stepsize(s_h, s_x, z_h, z_x, it, alpha_max=curr_max)

        h_prev, x_prev = h.copy(), x.copy()
        g_h_prev, g_x_prev = g_h.copy(), g_x.copy()

        # 5. Descent Step
        h = h - alpha * g_h
        x = x - alpha * g_x

        # 6. Projections / Constraints
        # Soft Thresholding for Kernel:
        # Calculate dynamic threshold based on current max intensity
        # kernel_threshold input is treated as a ratio (e.g. 0.02)
        if kernel_threshold > 0.0:
            # Gradually increase thresholding strictness? 
            # Or keep constant relative to peak.
            # Using Soft Thresholding (L1 proxy)
            thresh_val = kernel_threshold * h.max() * 0.1 # Scale down for Soft Threshold
            h = threshold_kernel(h, threshold=thresh_val)
        else:
            h = project_kernel(h) # Just simplex constraint

        x = np.clip(x, 0.0, 1.0)

        # 7. Convergence Check
        diff = np.linalg.norm(h - h_old) / (np.linalg.norm(h) + 1e-20)
        
        # Calculate cost occasionally to save compute
        if verbose and (it % 20 == 0 or it == max_iter - 1):
            cost = 0.5 * np.sum((y - fft_convolve(x, h)) ** 2)
            if lambda_tv > 0:
                # Add approximate TV cost for logging
                dx = np.roll(x, -1, axis=1) - x
                dy = np.roll(x, -1, axis=0) - x
                cost += lambda_tv * np.sum(np.sqrt(dx**2 + dy**2))
                
            print(f"    iter {it + 1:4d}: cost={cost:.4e}  rel_ΔH={diff:.2e}  α={alpha:.2e}")
            info['cost'].append(cost)

        if diff < tol and it > 20:
            if verbose: print(f"    Converged.")
            break

    h = center_kernel(h)
    h = project_kernel(h)
    return x, h, info

def sdp_multiscale(
    y: np.ndarray,
    kernel_shape: tuple,
    lambda_tv: float = 0.002,
    num_scales: int = 4,
    iters_per_scale: int = 200,
    tol: float = 1e-6,
    kernel_threshold: float = 0.02,
    verbose: bool = False,
) -> tuple:
    pyramid = build_pyramid(y, num_scales=num_scales)
    num_levels = len(pyramid)

    if verbose:
        print(f"[SDP] Multi-scale. Scales: {num_levels}, Kernel: {kernel_shape}")

    h_est = None
    history = {'scales': []}

    for level in range(num_levels):
        y_level = pyramid[level]
        ker_shape_l = kernel_shape_for_level(kernel_shape, level, num_levels)

        if verbose:
            print(f"\n  ── Scale {level} (img={y_level.shape}, ker={ker_shape_l}) ──")

        # Use the fixed edge_taper with blending
        y_tapered = edge_taper(y_level, ker_shape_l)

        # Initialize Kernel
        if h_est is None:
            h_level = init_gaussian_kernel(ker_shape_l)
        else:
            h_level = upsample_kernel(h_est, ker_shape_l)

        # Initialize Image
        # Using the observation y is robust.
        # Alternatively, one could upsample the previous x_est, but 
        # in blind deconv, fresh initialization at new scale often avoids local minima.
        x_level = y_tapered.copy()

        # Adaptive TV: High regularization at coarse scales prevents noise overfitting
        scale_factor = (level + 1) / num_levels
        # Example schedule: 3x TV at coarsest, 1x at finest
        lambda_tv_level = lambda_tv * (1.0 + 2.0 * (1.0 - scale_factor))

        x_level, h_level, info = sdp_single_scale(
            y_tapered, h_level, x_level,
            lambda_tv=lambda_tv_level,
            max_iter=iters_per_scale,
            tol=tol,
            kernel_threshold=kernel_threshold,
            verbose=verbose,
        )

        h_est = h_level
        history['scales'].append(info)

    # Final Kernel Processing
    if h_est.shape != kernel_shape:
        h_est = upsample_kernel(h_est, kernel_shape)
    
    h_est = center_kernel(h_est)
    h_est = threshold_kernel(h_est, threshold=kernel_threshold * h_est.max() * 0.05) # Final light cleanup
    h_est = project_kernel(h_est)

    if verbose:
        print("\n  ── Final non-blind refinement ──")

    # Use original Y (tapered)
    y_final_tapered = edge_taper(y, kernel_shape)
    
    # We can run a few iterations of non-blind with lower TV to bring back details
    x_est = refine_image_non_blind(
        y_final_tapered, h_est, y.copy(),
        lambda_tv=lambda_tv * 0.8, # Slightly less TV for final detail
        max_iter=100,
        verbose=verbose,
    )

    return x_est, h_est, history

def refine_image_non_blind(y, h, x_init, lambda_tv, max_iter=100, verbose=False):
    # Standard Gradient Descent / FISTA could be used here
    # Using simple GD with fixed step for robustness
    H_img, W_img = y.shape
    kh, kw = h.shape
    h_pad = np.zeros((H_img, W_img), dtype=np.float64)
    h_pad[:kh, :kw] = h
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)

    F_h = fft2(h_pad)
    F_hc = np.conj(F_h)
    F_y = fft2(y)

    # Lipschitz constant
    L = np.max(np.abs(F_h) ** 2) + 1e-8
    step = 1.0 / L

    x = x_init.copy()
    
    for it in range(max_iter):
        F_x = fft2(x)
        F_r = F_y - F_h * F_x
        grad = -np.real(ifft2(F_hc * F_r))
        
        if lambda_tv > 0.0:
            grad += lambda_tv * tv_gradient(x)
            
        x = x - step * grad
        x = np.clip(x, 0.0, 1.0)
        
    if verbose:
        res = y - fft_convolve(x, h)
        print(f"    Refinement MSE: {np.mean(res**2):.4e}")
        
    return x