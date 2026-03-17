"""
Solver routines for Blind Image Deconvolution.
Now supports Hysteresis Gradient Thresholding, Richardson-Lucy restoration,
and correct Boundary Handling (Pad/Crop).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Dict, Any, Tuple, List

from .utils import (
    psf2otf,
    fractional_gradient_x,
    fractional_gradient_y,
    fractional_otf_x,
    fractional_otf_y,
    pmp_operator,
    pmp_mask,
    soft_threshold,
    hard_threshold,
    gradient_otfs,
    kernel_threshold_and_normalize,
    center_kernel,
    build_image_pyramid,
    resize_kernel,
    edgetaper,
    compute_gradients,
    apply_hysteresis_threshold
)

_EPS = 1e-8


# ═══════════════════════════════════════════════════════════════════════════
#  Sub-problems
# ═══════════════════════════════════════════════════════════════════════════

def solve_u_subproblem(f: np.ndarray, alpha: float,
                       lam: float, rho1: float) -> np.ndarray:
    grad_x = fractional_gradient_x(f, alpha)
    tau = np.sqrt(2.0 * lam / rho1)
    return hard_threshold(grad_x, tau)


def solve_v_subproblem(f: np.ndarray, alpha: float,
                       lam: float, rho2: float) -> np.ndarray:
    grad_y = fractional_gradient_y(f, alpha)
    tau = np.sqrt(2.0 * lam / rho2)
    return hard_threshold(grad_y, tau)


def solve_w_subproblem(f: np.ndarray, gamma: float, rho3: float,
                       patch_size: int) -> np.ndarray:
    pmp_f = pmp_operator(f, patch_size)
    tau = np.sqrt(2.0 * gamma / rho3)
    return hard_threshold(pmp_f, tau)


def solve_f_subproblem_fft(
    g: np.ndarray,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    alpha: float,
    mu: float,
    rho1: float,
    rho2: float,
) -> np.ndarray:
    shape = g.shape

    G = fft2(g)
    U = fft2(u)
    V = fft2(v)

    H = psf2otf(h, shape)
    Dx = fractional_otf_x(alpha, shape)
    Dy = fractional_otf_y(alpha, shape)

    numer = (mu * np.conj(H) * G
             + rho1 * np.conj(Dx) * U
             + rho2 * np.conj(Dy) * V)

    denom = (mu * np.abs(H) ** 2
             + rho1 * np.abs(Dx) ** 2
             + rho2 * np.abs(Dy) ** 2
             + _EPS)

    f_new = np.real(ifft2(numer / denom))
    return f_new


def apply_pmp_constraint(
    f: np.ndarray,
    w: np.ndarray,
    patch_size: int
) -> np.ndarray:
    mask = pmp_mask(f, patch_size)
    f_new = f.copy()
    f_new[mask > 0] = w[mask > 0]
    return f_new


def solve_h_subproblem(
    g: np.ndarray,
    f: np.ndarray,
    beta: float,
    mu: float,
    kernel_shape: Tuple[int, int],
    params: Dict[str, Any], 
) -> np.ndarray:
    shape = g.shape
    
    grad_thresh_factor = params.get('grad_threshold_factor', 2.0)
    hysteresis_ratio = params.get('hysteresis_ratio', 0.5) 
    border_w = params.get('border_width', 5)
    rel_thresh = params.get('kernel_threshold', 0.05)
    
    fx, fy = compute_gradients(f)
    gx, gy = compute_gradients(g)
    
    # Border Suppression
    if shape[0] > 2*border_w and shape[1] > 2*border_w:
        fx[:, :border_w] = 0; fx[:, -border_w:] = 0
        gx[:, :border_w] = 0; gx[:, -border_w:] = 0
        fy[:border_w, :] = 0; fy[-border_w:, :] = 0
        gy[:border_w, :] = 0; gy[-border_w:, :] = 0
        fx[:border_w, :] = 0; fx[-border_w:, :] = 0
        gx[:border_w, :] = 0; gx[-border_w:, :] = 0
        fy[:, :border_w] = 0; fy[:, -border_w:] = 0
        gy[:, :border_w] = 0; gy[:, -border_w:] = 0

    mag = np.sqrt(fx**2 + fy**2)
    
    # HYSTERESIS THRESHOLDING
    if grad_thresh_factor > 0:
        high_t = grad_thresh_factor * np.mean(mag)
        low_t = high_t * hysteresis_ratio
        mask = apply_hysteresis_threshold(mag, low_t, high_t)
    else:
        mask = np.ones_like(mag)
    
    fx = fx * mask
    fy = fy * mask
    gx = gx * mask
    gy = gy * mask
    
    FX = fft2(fx)
    FY = fft2(fy)
    GX = fft2(gx)
    GY = fft2(gy)
    
    otf_dx, otf_dy = gradient_otfs(shape)
    
    numer = (np.conj(FX) * GX + np.conj(FY) * GY)
    denom = (np.abs(FX) ** 2 + np.abs(FY) ** 2
             + (beta / mu) * (np.abs(otf_dx) ** 2 + np.abs(otf_dy) ** 2)
             + _EPS)

    h_full = np.real(ifft2(numer / denom))

    h_cropped = _crop_kernel_center(h_full, kernel_shape[0], kernel_shape[1])
    h_cropped = kernel_threshold_and_normalize(h_cropped, rel_thresh)
    return h_cropped


# ═══════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def blind_deconv_single_scale(
    g: np.ndarray,
    h_init: np.ndarray,
    params: Dict[str, Any],
    max_iter: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    
    alpha      = params['alpha']
    mu         = params['mu']
    lam        = params['lam']
    gamma      = params['gamma']
    beta       = params['beta']
    patch_size = params['patch_size']
    rho1       = params['rho1_init']
    rho2       = params['rho2_init']
    rho3       = params['rho3_init']
    rho_factor = params['rho_factor']
    
    h = h_init.copy()
    f = g.copy()

    kernel_diffs: List[float] = []

    for _ in range(max_iter):
        h_old = h.copy()

        # Latent image
        u = solve_u_subproblem(f, alpha, lam, rho1)
        v = solve_v_subproblem(f, alpha, lam, rho2)
        f_temp = solve_f_subproblem_fft(g, h, u, v, alpha, mu, rho1, rho2)
        
        w = solve_w_subproblem(f_temp, gamma, rho3, patch_size)
        f = apply_pmp_constraint(f_temp, w, patch_size)
        np.clip(f, 0.0, 1.0, out=f)

        # Kernel
        h = solve_h_subproblem(g, f, beta, mu, h.shape, params)

        diff = np.linalg.norm(h - h_old)
        kernel_diffs.append(float(diff))

        rho1 *= rho_factor
        rho2 *= rho_factor

    return f, h, kernel_diffs


def blind_deconv_multiscale(
    g: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, list]]:
    
    kernel_shape = tuple(params['kernel_shape'])
    num_scales   = params.get('num_scales', 5)
    iter_per_sc  = params.get('iter_per_scale', 5)
    scale_factor = params.get('scale_factor', 0.707)
    boundary_mode = params.get('boundary_mode', 'edgetaper')
    final_mode    = params.get('final_restoration_mode', None)

    pyramid = build_image_pyramid(g, num_scales, scale_factor)

    coarse_ratio = pyramid[0].shape[0] / g.shape[0]
    kh0 = max(3, int(round(kernel_shape[0] * coarse_ratio)) | 1)
    kw0 = max(3, int(round(kernel_shape[1] * coarse_ratio)) | 1)
    h = np.zeros((kh0, kw0), dtype=np.float64)
    h[kh0 // 2, kw0 // 2] = 1.0

    history: Dict[str, list] = {'kernel_diff': []}

    # --- Blind Phase (Kernel Estimation) ---
    for s, g_s in enumerate(pyramid):
        scale_ratio = g_s.shape[0] / g.shape[0]
        kh_s = max(3, int(round(kernel_shape[0] * scale_ratio)) | 1)
        kw_s = max(3, int(round(kernel_shape[1] * scale_ratio)) | 1)

        # Apply boundary handling for kernel estimation phase
        if boundary_mode == 'edgetaper':
            g_s_proc = edgetaper(g_s, (kh_s, kw_s))
        elif boundary_mode == 'pad':
            # Padding for estimation is tricky if not unpadded, sticking to edgetaper logic 
            # or simple wrap for estimation usually suffices. 
            # Let's use edgetaper for estimation stability by default if pad is selected,
            # or manually pad/unpad inside single scale (complex). 
            # Safe bet: Edgetaper during estimation is usually robust enough.
            # But let's respect the param if we can.
            pad_h = kh_s // 2
            pad_w = kw_s // 2
            g_s_proc = np.pad(g_s, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        else:
            g_s_proc = g_s

        if h.shape != (kh_s, kw_s):
            h = resize_kernel(h, (kh_s, kw_s))
            h = center_kernel(h) 

        scale_params = dict(params)
        if s == 0:
             scale_params['iter_per_scale'] = max(iter_per_sc, 15)

        # Run estimation
        f_s, h, kdiffs = blind_deconv_single_scale(
            g_s_proc, h, scale_params, max_iter=scale_params['iter_per_scale'],
        )
        history['kernel_diff'].extend(kdiffs)

    # Finalize kernel
    if h.shape != kernel_shape:
        h = resize_kernel(h, kernel_shape)
    h = center_kernel(h)
    h = kernel_threshold_and_normalize(h, params.get('kernel_threshold', 0.05))

    # --- Phase 2: Final Non-blind Restoration (WITH PADDING/CROPPING) ---
    
    pad_h, pad_w = 0, 0
    g_final = g
    
    # 1. Prepare image (Pad or Edgetaper)
    if boundary_mode == 'pad':
        pad_h = kernel_shape[0] 
        pad_w = kernel_shape[1] 
        # Using 'edge' (replicate) padding moves the boundary discontinuity far away
        g_final = np.pad(g, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    elif boundary_mode == 'edgetaper':
        g_final = edgetaper(g, kernel_shape)

    # 2. Run Restoration
    if final_mode == 'tikhonov':
        reg_weight = params.get('lam_nonblind', 2e-3)
        # Increased regularization slightly to combat ringing
        f_est = final_restore_tikhonov(g_final, h, reg_weight)
    elif final_mode == 'richardson':
        f_est = final_restore_richardson_lucy(g_final, h, iterations=30)
    else:
        # Default iterative L1
        f_est = final_nonblind_deconv(g_final, h, params)

    # 3. Unpad (Crop) if necessary
    if boundary_mode == 'pad':
        # Crop back to original size
        f_est = f_est[pad_h:-pad_h, pad_w:-pad_w]

    return f_est, h, history


def final_restore_tikhonov(
    g: np.ndarray,
    h: np.ndarray,
    reg_weight: float = 5e-3
) -> np.ndarray:
    """
    Tikhonov with Gradient Regularization.
    reg_weight: Higher values = smoother image, less ringing.
    """
    shape = g.shape
    H = psf2otf(h, shape)
    G = fft2(g)
    otf_dx, otf_dy = gradient_otfs(shape)
    # Magnitude of gradient filter in freq domain
    laplacian_kernel = np.abs(otf_dx)**2 + np.abs(otf_dy)**2
    
    numer = np.conj(H) * G
    # Added extra epsilon 1e-3 to prevent division by near-zero at low freqs
    denom = np.abs(H)**2 + reg_weight * laplacian_kernel + 1e-3
    
    f = np.real(ifft2(numer / denom))
    np.clip(f, 0.0, 1.0, out=f)
    return f


def final_restore_richardson_lucy(
    g: np.ndarray,
    h: np.ndarray,
    iterations: int = 30
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution.
    Inherently non-negative, often produces sharper edges with less ringing 
    than linear methods (Tikhonov/Wiener) for impulse kernels.
    """
    # RL works best with non-negative data
    g = np.maximum(g, 1e-6)
    h = np.maximum(h, 1e-9) # Avoid div by zero in kernel
    if h.sum() > 0:
        h /= h.sum()
        
    # Initial estimate
    u = g.copy()
    
    # For FFT convolution, we need OTFs
    H = psf2otf(h, u.shape)
    H_conj = np.conj(H)
    
    for _ in range(iterations):
        # 1. Blur current estimate: H * u
        est_blur = np.real(ifft2(H * fft2(u)))
        est_blur = np.maximum(est_blur, 1e-6)
        
        # 2. Ratio: g / (H * u)
        ratio = g / est_blur
        
        # 3. Correction: H_conj * ratio
        correction = np.real(ifft2(H_conj * fft2(ratio)))
        
        # 4. Update: u = u * correction
        u = u * correction
        np.clip(u, 0.0, 1.0, out=u)
        
    return u


def final_nonblind_deconv(
    g: np.ndarray,
    h: np.ndarray,
    params: Dict[str, Any],
    max_iter: int = 15,
) -> np.ndarray:
    """Non-blind restoration using L1 fractional TV."""
    alpha  = params['alpha']
    mu_nb  = params.get('mu_nonblind', 50.0)
    lam_nb = params.get('lam_nonblind', 2e-3)
    rho    = 1.0
    
    shape = g.shape
    H  = psf2otf(h, shape)
    Dx = fractional_otf_x(alpha, shape)
    Dy = fractional_otf_y(alpha, shape)
    G  = fft2(g)
    
    # Wiener init with stronger dampening
    f = np.real(ifft2(np.conj(H)*G / (np.abs(H)**2 + 1e-2)))
    np.clip(f, 0, 1, out=f)

    for _ in range(max_iter):
        grad_x = fractional_gradient_x(f, alpha)
        grad_y = fractional_gradient_y(f, alpha)
        u = soft_threshold(grad_x, lam_nb / rho)
        v = soft_threshold(grad_y, lam_nb / rho)

        U = fft2(u)
        V = fft2(v)

        numer = mu_nb * np.conj(H) * G + rho * (np.conj(Dx) * U + np.conj(Dy) * V)
        denom = mu_nb * np.abs(H) ** 2 + rho * (np.abs(Dx) ** 2 + np.abs(Dy) ** 2) + _EPS

        f = np.real(ifft2(numer / denom))
        np.clip(f, 0.0, 1.0, out=f)
        rho *= 1.5

    return f


def _crop_kernel_center(h_full: np.ndarray, kh: int, kw: int) -> np.ndarray:
    M, N = h_full.shape
    cy, cx = M // 2, N // 2
    h_half = kh // 2
    w_half = kw // 2
    top  = cy - h_half
    left = cx - w_half
    h_shifted = np.fft.fftshift(h_full)
    return h_shifted[top:top + kh, left:left + kw].copy()