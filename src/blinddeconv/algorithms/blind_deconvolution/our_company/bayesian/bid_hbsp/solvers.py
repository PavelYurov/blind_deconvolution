"""
Solver functions for the BID-HBSP algorithm.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg
from typing import Tuple

from .utils import (
    psf2otf,
    otf2psf,
    precompute_gradient_operators,
    forward_diff_x,
    forward_diff_y,
    adjoint_diff_x,
    adjoint_diff_y,
    compute_hs_weights,
    project_kernel,
    threshold_kernel,
    EPSILON,
    edgetaper,
)

def solve_image_cg(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    gamma_x: np.ndarray,
    gamma_y: np.ndarray,
    max_cg_iter: int = 50,
    cg_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates the image mean and approximates the pixel-wise variance (covariance diagonal).
    """
    H, W = y.shape
    N = H * W

    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y)

    # Operator A for Ax = b
    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        # Likelihood term: beta * H^T H
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        # Prior term: D^T Gamma D (approximated Laplacian-like structure)
        Av += adjoint_diff_x(gamma_x * forward_diff_x(v))
        Av += adjoint_diff_y(gamma_y * forward_diff_y(v))
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    rhs = beta * np.real(ifft2(F_h_conj * F_y))

    x_flat, _info = cg(A_op, rhs.ravel(), x0=x_init.ravel(), maxiter=max_cg_iter, atol=cg_tol)
    x_out = x_flat.reshape((H, W))
    
    # Variational Variance Approximation
    # Sigma^2 = 1 / diag(A)
    # Approximation: diag(H^T H) ~ ||h||^2 and spatial interactions
    h_energy = np.sum(h**2)
    reg_strength = (gamma_x + np.roll(gamma_x, 1, axis=1) + 
                    gamma_y + np.roll(gamma_y, 1, axis=0))
    
    sigma_sq = 1.0 / (beta * h_energy + reg_strength + EPSILON)

    return np.maximum(x_out, 0.0), sigma_sq


def solve_kernel_fourier(
    y: np.ndarray,
    x: np.ndarray,
    sigma_sq: np.ndarray,
    kernel_shape: Tuple[int, int],
    beta: float,
    lambda_h: float,
    do_threshold: bool = True,
) -> np.ndarray:
    """
    Estimates kernel in Gradient Space with Covariance correction and Boundary Masking.
    Ref: Castro-Macías (2024) Section IV.B & Eq (21).
    """
    H, W = y.shape
    
    # 1. Gradient domain
    dy_x = forward_diff_x(y)
    dy_y = forward_diff_y(y)
    
    dx_x = forward_diff_x(x)
    dx_y = forward_diff_y(x)
    
    # Boundary Masking
    dy_x[:, -1] = 0.0
    dx_x[:, -1] = 0.0
    dy_y[-1, :] = 0.0
    dx_y[-1, :] = 0.0
    F_dy_x = fft2(dy_x)
    F_dy_y = fft2(dy_y)
    F_dx_x = fft2(dx_x)
    F_dx_y = fft2(dx_y)
    
    # 2. Wiener filter
    numerator = (F_dy_x * np.conj(F_dx_x)) + (F_dy_y * np.conj(F_dx_y))
    
    # Autocorrelation X
    denominator = (np.abs(F_dx_x)**2) + (np.abs(F_dx_y)**2)
    
    # VB Correction term (Sigma)
    # Trace(T^T T Sigma) approx N * mean(Sigma_grad)
    sigma_grad_mean = 2.0 * np.mean(sigma_sq)
    uncertainty_term = H * W * sigma_grad_mean
    
    # regularization
    denominator += uncertainty_term + (lambda_h / beta) + EPSILON
    
    F_h = numerator / denominator
    
    # 3. Spatial domain
    h = otf2psf(F_h, kernel_shape)
    
    if do_threshold:
        # thresholding
        h = threshold_kernel(h, ratio=0.1) #0.05
    else:
        h = project_kernel(h)
        
    return h


def update_noise_precision(y: np.ndarray, h: np.ndarray, x: np.ndarray, beta_prev: float, damping: float = 0.5) -> float:
    H, W = y.shape
    N = float(H * W)
    F_h = psf2otf(h, (H, W))
    residual = y - np.real(ifft2(F_h * fft2(x)))
    rss = float(np.sum(residual ** 2))
    beta_new = N / (rss + EPSILON)
    beta = (1.0 - damping) * beta_prev + damping * beta_new
    beta = float(np.clip(beta, 1.0, 1e8))
    return beta


def update_hs_weights(x: np.ndarray, sigma_sq: np.ndarray, b: float) -> Tuple:
    """
    Update HS weights using the variational expectation E[w].
    Needs sigma_sq (variance) as per Eq. 26 in the paper.
    """
    dx = forward_diff_x(x)
    dy = forward_diff_y(x)
    from .utils import compute_hs_weights
    return compute_hs_weights(dx, dy, sigma_sq, b)


def final_deconvolution(y: np.ndarray, h: np.ndarray, beta: float, lambda_reg: float) -> np.ndarray:
    """
    Non-blind deconvolution using IRLS with Padding to remove boundary artifacts.
    Minimizes: 0.5 * ||y - h*x||^2 + lambda * ||grad x||_p^p
    """
    # Padding
    kh, kw = h.shape
    pad_h = kh
    pad_w = kw
    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    Hp, Wp = y_padded.shape

    x = y_padded.copy()
    
    # IRLS
    p = 0.8
    irls_iters = 15
    
    # Recalculating kernel OTF
    F_h = psf2otf(h, (Hp, Wp))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    
    # (H^T * y)
    F_y = fft2(y_padded)
    rhs_base = beta * np.real(ifft2(F_h_conj * F_y))
    
    for i in range(irls_iters):
        dx = forward_diff_x(x)
        dy = forward_diff_y(x)
        
        # Weights IRLS: w = p * (|grad|^2 + eps)^(p/2 - 1)
        # power = (0.8 - 2) / 2 = -0.6
        power = (p - 2.0) / 2.0
        
        grad_sq_x = dx**2 + 1e-8
        grad_sq_y = dy**2 + 1e-8
        
        wx = p * (grad_sq_x ** power)
        wy = p * (grad_sq_y ** power)
        
        wx = np.clip(wx, 0.0, 1e4)
        wy = np.clip(wy, 0.0, 1e4)
        
        # regularization
        wx *= lambda_reg
        wy *= lambda_reg
        
        x = _solve_image_irls_step(rhs_base, F_h_sq, wx, wy, x, beta, cg_iter=15)
        
        x = np.clip(x, 0.0, 1.0)
    
    x_final = x[pad_h:-pad_h, pad_w:-pad_w]
        
    return x_final

def _solve_image_irls_step(
    rhs: np.ndarray,
    F_h_sq: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    cg_iter: int = 20
) -> np.ndarray:
    """
    Solves one step of IRLS using Conjugate Gradient.
    System: (beta * H^T H + Dx^T Wx Dx + Dy^T Wy Dy) x = rhs
    """
    H, W = x_init.shape
    N = H * W
    
    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        
        # Data term: beta * H^T H x
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        
        # Prior term: Dx^T (Wx * Dx * v)
        dx_v = forward_diff_x(v)
        dy_v = forward_diff_y(v)
        
        dx_v *= wx
        dy_v *= wy
        
        Av += adjoint_diff_x(dx_v)
        Av += adjoint_diff_y(dy_v)
        
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    
    x_flat, _ = cg(A_op, rhs.ravel(), x0=x_init.ravel(), maxiter=cg_iter, atol=1e-5)
    
    return x_flat.reshape((H, W))

# Placeholder for solve_image_irw if needed to avoid import errors, 
# though 'cg' is the preferred solver.
def solve_image_irw(*args, **kwargs):
    raise NotImplementedError("IRW solver is not fully adapted for the new variance tracking. Use 'cg'.")