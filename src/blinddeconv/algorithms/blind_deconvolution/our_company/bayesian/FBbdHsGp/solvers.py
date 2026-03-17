import numpy as np
from numpy.fft import fft2, ifft2
from .utils import psf2otf, project_simplex

def get_huber_weights(grad_magnitude, eps=0.005):
    """
    Calculate Huber weights xi = rho'(s)/s.
    For Huber: 1.0 if |s| < eps, else eps/|s|.
    """
    w = np.ones_like(grad_magnitude)
    mask = grad_magnitude > eps
    w[mask] = eps / (grad_magnitude[mask] + 1e-12)
    return w

def compute_variational_stats(y, k_est, xi_x, xi_y, sigma_n, F_dx, F_dy):
    """
    Computes Posterior Mean (mu_x) and Spectral Covariance (Sigma_spec).
    This corresponds to the E-step logic but returns the full statistics needed for VB.
    
    Based on Zhou et al. Eq 16 (Precision) and 17 (Mean).
    """
    H, W = y.shape
    F_y = fft2(y)
    F_k = psf2otf(k_est, (H, W))
    
    # Precision Alpha = 1/sigma^2
    alpha = 1.0 / (sigma_n**2 + 1e-10)
    
    # Mean Field Approximation: Use scalar average of spatial weights for the Circulant Preconditioner
    # This is standard in Fast VB methods (Babacan 2012, Zhou 2017) to allow FFT inversion.
    avg_xi_x = np.mean(xi_x)
    avg_xi_y = np.mean(xi_y)
    
    # Denominator of the Wiener Filter (Precision Spectrum)
    # Lambda = alpha * |H|^2 + Sum(avg_xi * |D|^2)
    denom = alpha * (np.abs(F_k)**2) + avg_xi_x * (np.abs(F_dx)**2) + avg_xi_y * (np.abs(F_dy)**2)
    Sigma_spec = 1.0 / (denom + 1e-10)
    
    # Posterior Mean (Eq 17)
    # mu = Sigma * (alpha * H' * y)
    numerator = alpha * np.conj(F_k) * F_y
    F_mu = numerator * Sigma_spec
    mu_x = np.real(ifft2(F_mu))
    
    return mu_x, Sigma_spec

def update_huber_weights_variational(mu_x, Sigma_spec, F_dx, F_dy, epsilon):
    """
    W-Step: Update Huber weights based on Expected Second Moment.
    Eq. 18: nu_gamma = sqrt( E[ (D_gamma x)^2 ] )
    """
    N = mu_x.size
    F_mu = fft2(mu_x)
    
    # 1. Gradient of the mean
    dx = np.real(ifft2(F_dx * F_mu))
    dy = np.real(ifft2(F_dy * F_mu))
    
    # 2. Uncertainty term (Trace of Covariance * D'D)
    # Via Parseval: sum(|F_d|^2 * Sigma) / N
    var_x = np.sum(np.abs(F_dx)**2 * Sigma_spec) / N
    var_y = np.sum(np.abs(F_dy)**2 * Sigma_spec) / N
    
    # 3. Expected magnitude (nu)
    nu_x = np.sqrt(dx**2 + var_x)
    nu_y = np.sqrt(dy**2 + var_y)
    
    # 4. Weights
    xi_x = get_huber_weights(nu_x, epsilon)
    xi_y = get_huber_weights(nu_y, epsilon)
    
    return xi_x, xi_y

def solve_image_sadmm(y, k_est, xi_x, xi_y, sigma_n, beta_v, F_dx, F_dy, F_dtd, n_iters=1):
    """
    Algorithm 1: Image Estimation via ADMM.
    Finds a 'smoother' estimate than the pure Wiener filter by enforcing the
    Huber prior via variable splitting.
    """
    H, W = y.shape
    F_y = fft2(y)
    F_k = psf2otf(k_est, (H, W))
    F_ktk = np.abs(F_k)**2
    
    # Denominator for x update (Eq 22)
    # H'H + beta * D'D
    # Note: Zhou Eq 22 omits 1/sigma^2 factor on data term, implying implicit scaling.
    # We stick to standard ADMM form: 1/2||Hx-y||^2 + beta/2||Dx-v||^2
    denom = F_ktk + beta_v * F_dtd + 1e-10
    
    x = y.copy()
    v_x = np.zeros_like(y)
    v_y = np.zeros_like(y)
    dv_x = np.zeros_like(y)
    dv_y = np.zeros_like(y)
    
    sigma_sq = sigma_n**2
    
    for _ in range(n_iters):
        # 1. Update x
        rhs_x = np.conj(F_dx) * fft2(v_x + dv_x)
        rhs_y = np.conj(F_dy) * fft2(v_y + dv_y)
        
        num = np.conj(F_k) * F_y + beta_v * (rhs_x + rhs_y)
        x = np.real(ifft2(num / denom))
        
        # 2. Update v (Reweighted shrinkage, Eq 23)
        F_x = fft2(x)
        gx = np.real(ifft2(F_dx * F_x))
        gy = np.real(ifft2(F_dy * F_x))
        
        # Shrinkage factor
        # The paper derives this for the quadratic upper bound of Huber.
        # scale = beta / (beta + sigma^2 * xi)
        scale_x = beta_v / (beta_v + sigma_sq * xi_x + 1e-10)
        scale_y = beta_v / (beta_v + sigma_sq * xi_y + 1e-10)
        
        v_x = scale_x * (gx - dv_x)
        v_y = scale_y * (gy - dv_y)
        
        # 3. Update dual
        dv_x = dv_x + v_x - gx
        dv_y = dv_y + v_y - gy
        
    return x

def solve_kernel_admm(y, mu_x, Sigma_spec, k_shape, beta_h, n_iters=5):
    """
    Algorithm 2: Kernel Estimation.
    CRITICAL: Uses Sigma_spec (Variational term) to prevent delta-solution.
    """
    H, W = y.shape
    kh, kw = k_shape
    
    F_y = fft2(y)
    F_x = fft2(mu_x)
    
    # E[X'X] = |mu|^2 + N * Sigma (Eq 16 in Alg 2 context)
    # This extra term 'Sigma' acts as regularization, preventing the kernel
    # from simply inverting the image.
    F_xtx_expected = np.abs(F_x)**2 + (Sigma_spec * (H * W))
    
    F_xty = np.conj(F_x) * F_y
    
    H_var = np.zeros((H, W), dtype=np.complex64)
    dH = np.zeros((H, W), dtype=np.complex64)
    
    k_est = np.zeros(k_shape, dtype=np.float32)
    k_est[kh//2, kw//2] = 1.0
    
    for _ in range(n_iters):
        # 1. Update H (Freq domain) - Eq 29
        F_Pk = psf2otf(k_est, (H, W))
        
        num = F_xty + beta_h * (F_Pk - dH)
        denom = F_xtx_expected + beta_h + 1e-10
        H_var = num / denom
        
        # 2. Update h (Spatial) - Eq 30
        spatial_target = np.real(ifft2(H_var + dH))
        
        # Unroll cyclic shift (center of image -> center of kernel)
        spatial_target = np.roll(spatial_target, H//2, axis=0)
        spatial_target = np.roll(spatial_target, W//2, axis=1)
        
        # Crop to support
        start_h = (H - kh) // 2
        start_w = (W - kw) // 2
        k_crop = spatial_target[start_h:start_h+kh, start_w:start_w+kw]
        
        # Project to simplex (Non-negativity + Sum=1)
        # Note: We do NOT use aggressive thresholding here to stay true to the math.
        # The Sigma term should handle the regularization naturally.
        k_est = project_simplex(k_crop)
        
        # 3. Update dual
        F_Pk_new = psf2otf(k_est, (H, W))
        dH = dH + H_var - F_Pk_new
        
    return k_est