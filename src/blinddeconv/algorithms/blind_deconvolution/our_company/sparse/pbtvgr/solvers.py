import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import cg, LinearOperator
from .utils import psf2otf, compute_gradients, compute_divergence

def solve_u_subproblem(g: np.ndarray, h: np.ndarray, o: np.ndarray, 
                       beta: float, lambda_: float) -> np.ndarray:
    """
    Algorithm 3: Solving for u sub-problem.
    Eq. (16) - (18) in Dong et al. (2021).
    """
    H, W = g.shape
    
    F_h = psf2otf(h, (H, W))
    F_o = fft2(o)
    ho = np.real(ifft2(F_h * F_o))
    
    b_term = lambda_ - 2 * beta * ho
    delta = b_term**2 + 8 * beta * lambda_ * g
    
    sqrt_delta = np.sqrt(np.maximum(delta, 0))
    u = (2 * beta * ho - lambda_ + sqrt_delta) / (4 * beta + 1e-12)
    
    return np.maximum(u, 1e-8)

def solve_h_subproblem(u: np.ndarray, o: np.ndarray, h_prev: np.ndarray, 
                       beta: float, mu: float, F_dx: np.ndarray, F_dy: np.ndarray) -> np.ndarray:
    """
    Algorithm 2: IRLS for solving h sub-problem.
    """
    H, W = u.shape
    kh, kw = h_prev.shape
    
    dx_o, dy_o = compute_gradients(o)
    dx_u, dy_u = compute_gradients(u)
    
    F_dx_o = fft2(dx_o)
    F_dy_o = fft2(dy_o)
    F_dx_u = fft2(dx_u)
    F_dy_u = fft2(dy_u)

    rhs_freq = beta * (np.conj(F_dx_o) * F_dx_u + np.conj(F_dy_o) * F_dy_u)
    b_spatial = np.real(ifft2(rhs_freq))
    
    b_spatial = np.roll(b_spatial, kh//2, axis=0)
    b_spatial = np.roll(b_spatial, kw//2, axis=1)
    b_vec = b_spatial[:kh, :kw].flatten()
    
    h_curr = h_prev.copy()
    l_max = 5
    
    for l in range(l_max):
        dh_x, dh_y = compute_gradients(h_curr)
        grad_mag = np.sqrt(dh_x**2 + dh_y**2 + 1e-6)
        W_inv = 1.0 / grad_mag 
        def matvec(v_flat):
            v = v_flat.reshape((kh, kw))
            F_v = psf2otf(v, (H, W))
            
            res_x = F_dx_o * F_v
            res_y = F_dy_o * F_v

            res_freq = beta * (np.conj(F_dx_o) * res_x + np.conj(F_dy_o) * res_y)
            res_spatial = np.real(ifft2(res_freq))
            
            res_spatial = np.roll(res_spatial, kh//2, axis=0)
            res_spatial = np.roll(res_spatial, kw//2, axis=1)
            term1 = res_spatial[:kh, :kw]
            
            dv_x, dv_y = compute_gradients(v)
            wdv_x = W_inv * dv_x
            wdv_y = W_inv * dv_y
            term2 = mu * compute_divergence(wdv_x, wdv_y)
            
            return (term1 + term2).flatten()
            
        A_op = LinearOperator((kh*kw, kh*kw), matvec=matvec)
        
        h_flat, _ = cg(A_op, b_vec, x0=h_curr.flatten(), maxiter=50, atol=1e-5)
        h_curr = h_flat.reshape((kh, kw))
        
        h_curr = np.maximum(h_curr, 0)
        h_sum = np.sum(h_curr)
        if h_sum > 1e-12:
            h_curr /= h_sum
            
    return h_curr

def solve_o_subproblem(u: np.ndarray, h: np.ndarray, o_prev: np.ndarray, 
                       beta: float, F_dx: np.ndarray, F_dy: np.ndarray) -> np.ndarray:
    """
    Algorithm 4: Solving for o sub-problem using L0 minimization.
    Based on Xu et al. (2011) "Image Smoothing via L0 Gradient Minimization".
    """
    H, W = u.shape
    o = o_prev.copy()
    
    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_u = fft2(u)
    
    denom_fidelity = beta * np.abs(F_h)**2
    denom_grad = np.abs(F_dx)**2 + np.abs(F_dy)**2

    gamma = 2.0 * beta 
    gamma_max = 1e5
    kappa = 2.0
    
    while gamma < gamma_max:
        dx_o, dy_o = compute_gradients(o)
        mag_sq = dx_o**2 + dy_o**2
        
        mask = mag_sq > (2.0 / gamma)
        w_x = np.where(mask, dx_o, 0)
        w_y = np.where(mask, dy_o, 0)
        
        F_wx = fft2(w_x)
        F_wy = fft2(w_y)
        
        rhs = beta * F_h_conj * F_u + gamma * (np.conj(F_dx) * F_wx + np.conj(F_dy) * F_wy)
        lhs = denom_fidelity + gamma * denom_grad
        
        o = np.real(ifft2(rhs / (lhs + 1e-12)))
        o = np.maximum(o, 0)
        
        gamma *= kappa
        
    return o

def solve_nonblind_tv(g: np.ndarray, h: np.ndarray, o_init: np.ndarray, 
                      xi: float, eta_init: float, 
                      F_dx: np.ndarray, F_dy: np.ndarray) -> np.ndarray:
    """
    Algorithm 5: Non-blind deconvolution with TV regularization.
    Uses variable splitting from Wang et al. (2008).
    """
    H, W = g.shape
    o = o_init.copy()
    y = o.copy()
    
    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    denom_h = np.abs(F_h)**2
    
    eta = eta_init
    zeta = 2.0 
    max_iter = 10 
    
    for k in range(max_iter):
        ho = np.real(ifft2(F_h * fft2(o)))
        b_term = xi - 2 * eta * ho
        delta = b_term**2 + 8 * xi * eta * g
        x = (2 * eta * ho - xi + np.sqrt(np.maximum(delta, 0))) / (4 * eta + 1e-12)
        x = np.maximum(x, 1e-8)
        
        y = _solve_ftvd(o, eta, F_dx, F_dy)
        
        F_x = fft2(x)
        F_y = fft2(y)
        
        num = eta * (F_h_conj * F_x + F_y)
        den = eta * (denom_h + 1.0)
        o = np.real(ifft2(num / (den + 1e-12)))
        o = np.maximum(o, 0)
        
        eta += zeta * np.linalg.norm(o - y)**2
        
    return o

def _solve_ftvd(f: np.ndarray, mu: float, F_dx: np.ndarray, F_dy: np.ndarray) -> np.ndarray:
    """
    Fast Total Variation Deconvolution (FTVd) for Denoising.
    Implementation of Wang et al. (2008).
    """
    u = f.copy()
    beta = 1.0
    beta_max = 256.0
    
    F_f = fft2(f)
    denom_laplace = np.abs(F_dx)**2 + np.abs(F_dy)**2
    
    while beta < beta_max:
        dx_u, dy_u = compute_gradients(u)
        mag = np.sqrt(dx_u**2 + dy_u**2)
        
        mask = mag > (1.0 / beta)
        scale = np.zeros_like(mag)
        scale[mask] = 1.0 - 1.0 / (beta * mag[mask])
        
        w_x = dx_u * scale
        w_y = dy_u * scale
        
        F_wx = fft2(w_x)
        F_wy = fft2(w_y)
        
        rhs = mu * F_f + beta * (np.conj(F_dx) * F_wx + np.conj(F_dy) * F_wy)
        lhs = mu + beta * denom_laplace
        
        u = np.real(ifft2(rhs / (lhs + 1e-12)))
        
        beta *= 2.0
        
    return u