"""
Utility functions for VB-TV-BID, Checkpoint 5 (Edgetaper added).
"""

import numpy as np
from scipy.special import digamma, polygamma
from scipy.ndimage import center_of_mass, shift
from scipy.signal.windows import tukey


# =============================================================================
# Edgetapering (Boundary Artifact Removal)
# =============================================================================

def edgetaper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Smooth the edges of the image to reduce ringing artifacts in FFT.
    Mimics Matlab's edgetaper.
    
    It blends the image edges with a blurred version of the image, 
    forcing the boundary discontinuity to approach zero in the frequency domain.
    """
    h, w = image.shape
    kh, kw = kernel.shape

    # 1. Blur the image with the kernel (circular convolution is fine here 
    # because we only care about the edges matching the kernel's frequency response)
    # Using existing fft_convolve from this module
    blurred = fft_convolve(image, kernel)

    # 2. Create the weighting mask (alpha)
    # We use a Tukey window (cosine-tapered window).
    # The taper width should roughly match the kernel half-size.
    
    alpha_h = tukey(h, alpha=min(1.0, (kh * 2.0) / h))
    alpha_w = tukey(w, alpha=min(1.0, (kw * 2.0) / w))
    
    # Outer product to create 2D mask
    mask = np.outer(alpha_h, alpha_w)

    # 3. Blend
    # Center is original image (mask ~ 1), edges are blurred image (mask ~ 0).
    return image * mask + blurred * (1.0 - mask)


# =============================================================================
# FFT-Based Convolution Utilities
# =============================================================================

def psf_to_otf(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    kh, kw = kernel.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = kernel
    
    padded = np.roll(padded, shift=-(kh // 2), axis=0)
    padded = np.roll(padded, shift=-(kw // 2), axis=1)
    return np.fft.fft2(padded)


def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    otf = psf_to_otf(kernel, image.shape)
    return np.real(np.fft.ifft2(otf * np.fft.fft2(image)))


def fft_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    otf = psf_to_otf(kernel, image.shape)
    return np.real(np.fft.ifft2(np.conj(otf) * np.fft.fft2(image)))


# =============================================================================
# First-Order Difference Operators
# =============================================================================

def forward_diff_h(f: np.ndarray) -> np.ndarray:
    return np.roll(f, -1, axis=1) - f


def forward_diff_v(f: np.ndarray) -> np.ndarray:
    return np.roll(f, -1, axis=0) - f


def gradient_power_spectrum(shape: tuple) -> tuple:
    d_h = np.zeros(shape, dtype=np.float64)
    d_h[0, 0] = -1.0;  d_h[0, 1] = 1.0
    Dh_sq = np.abs(np.fft.fft2(d_h))**2

    d_v = np.zeros(shape, dtype=np.float64)
    d_v[0, 0] = -1.0;  d_v[1, 0] = 1.0
    Dv_sq = np.abs(np.fft.fft2(d_v))**2

    return Dh_sq, Dv_sq


# =============================================================================
# Isotropic Total Variation and MM Weights
# =============================================================================

def compute_tv(f: np.ndarray, epsilon: float = 1e-4) -> float:
    dh = forward_diff_h(f)
    dv = forward_diff_v(f)
    return float(np.sum(np.sqrt(dh**2 + dv**2 + epsilon)))


def compute_mm_weights(f: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    dh = forward_diff_h(f)
    dv = forward_diff_v(f)
    grad_mag = np.sqrt(dh**2 + dv**2 + epsilon)
    return 1.0 / (2.0 * np.maximum(grad_mag, 1e-12))


def apply_tv_precision_operator(f: np.ndarray, w: np.ndarray) -> np.ndarray:
    dh = forward_diff_h(f)
    dv = forward_diff_v(f)

    wh = w * dh
    wv = w * dv

    Qf_h = np.roll(wh, 1, axis=1) - wh
    Qf_v = np.roll(wv, 1, axis=0) - wv

    return Qf_h + Qf_v


def tv_quadratic_form(f: np.ndarray, w: np.ndarray) -> float:
    dh = forward_diff_h(f)
    dv = forward_diff_v(f)
    return float(np.sum(w * (dh**2 + dv**2)))


# =============================================================================
# Kernel Utilities
# =============================================================================

def extract_centered_kernel(h_full: np.ndarray, h_shape: tuple) -> np.ndarray:
    kh, kw = h_shape
    h_rolled = np.roll(h_full, kh // 2, axis=0)
    h_rolled = np.roll(h_rolled, kw // 2, axis=1)
    return h_rolled[:kh, :kw].copy()


def project_kernel(h: np.ndarray) -> np.ndarray:
    h_proj = np.maximum(h, 1e-12)
    total = h_proj.sum()
    if total > 1e-12:
        return h_proj / total
    return np.ones_like(h) / h.size


def center_kernel_mass(h: np.ndarray) -> np.ndarray:
    kh, kw = h.shape
    cy, cx = center_of_mass(h)
    dy = (kh // 2) - cy
    dx = (kw // 2) - cx
    h_shifted = shift(h, (dy, dx), order=1, mode='constant', cval=0.0, prefilter=False)
    return project_kernel(h_shifted)


# =============================================================================
# Trace Estimation
# =============================================================================

def hutchinson_trace_estimate(
    matvec_Ainv: callable,
    shape: tuple,
    matvec_B: callable = None,
    n_probes: int = 5,
    rng: np.random.Generator = None,
) -> float:
    if rng is None:
        rng = np.random.default_rng(42)

    H, W = shape
    total = 0.0

    for _ in range(n_probes):
        z = rng.choice([-1.0, 1.0], size=(H, W))
        if matvec_B is not None:
            Bz = matvec_B(z)
        else:
            Bz = z
        AinvBz = matvec_Ainv(Bz)
        total += np.sum(z * AinvBz)

    return total / n_probes


def spectral_trace(H_otf_sq: np.ndarray,
                   Q_spec: np.ndarray,
                   alpha: float,
                   beta: float,
                   B_spec: np.ndarray) -> float:
    A_spec = beta * H_otf_sq + alpha * Q_spec
    A_spec = np.maximum(A_spec, 1e-12)
    return float(np.sum(B_spec / A_spec))

def spectral_log_det(H_otf_sq: np.ndarray,
                     Q_spec: np.ndarray,
                     alpha: float,
                     beta: float) -> float:
    A_spec = beta * H_otf_sq + alpha * Q_spec
    A_spec = np.maximum(A_spec, 1e-30)
    return float(np.sum(np.log(A_spec)))

# =============================================================================
# ELBO Computation
# =============================================================================

def compute_elbo(y: np.ndarray,
                 f: np.ndarray,
                 h: np.ndarray,
                 alpha: float,
                 beta: float,
                 delta_h: float,
                 w: np.ndarray,
                 tr_Sigma_f_Q: float,
                 tr_Sigma_f_HtH: float,
                 log_det_Sigma_f: float,
                 h_cov_trace: float = 0.0) -> float:
    
    N = float(y.size)
    K = float(h.size)

    residual = y - fft_convolve(f, h)
    res_sq = float(np.sum(residual**2))
    data_term = -0.5 * beta * (res_sq + tr_Sigma_f_HtH)

    fQf = tv_quadratic_form(f, w)
    prior_f = -0.5 * alpha * (fQf + tr_Sigma_f_Q)

    prior_h = -0.5 * delta_h * (float(np.sum(h**2)) + h_cov_trace)

    entropy_f = 0.5 * log_det_Sigma_f
    entropy_h = 0.5 * K * np.log(2.0 * np.pi * np.e)

    logZ = 0.5 * N * np.log(max(beta, 1e-30)) \
         + 0.5 * N * np.log(max(alpha, 1e-30)) \
         + 0.5 * K * np.log(max(delta_h, 1e-30))

    return data_term + prior_f + prior_h + entropy_f + entropy_h + logZ