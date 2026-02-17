"""
Utility functions for Primal-Dual Splitting (PDS) Blind Deconvolution.
Based on O'Connor (2015) and standard FFT-based imaging techniques.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter

EPSILON = 1e-14

def edgetaper(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """
    Smooths image boundaries to reduce Gibbs ringing in FFT operations.
    Conceptually similar to MATLAB's edgetaper.
    """
    h, w = img.shape
    kh, kw = kernel_shape
    
    alpha_h = np.ones(h)
    alpha_w = np.ones(w)
    
    for i in range(kh):
        val = 0.5 * (1 - np.cos(np.pi * (i + 1) / (kh + 1)))
        alpha_h[i] = val
        alpha_h[h - 1 - i] = val
        
    for j in range(kw):
        val = 0.5 * (1 - np.cos(np.pi * (j + 1) / (kw + 1)))
        alpha_w[j] = val
        alpha_w[w - 1 - j] = val
        
    mask = np.outer(alpha_h, alpha_w)
    
    sigma = max(kh, kw) / 3.0
    blurred = gaussian_filter(img, sigma=sigma, mode='wrap')
    
    return img * mask + blurred * (1.0 - mask)

def huber_gradient(residual: np.ndarray, delta: float) -> np.ndarray:
    """
    Computes the gradient of the Huber loss function.
    
    phi(x) = 0.5 * x^2              if |x| <= delta
             delta * (|x| - 0.5*delta)  if |x| > delta
             
    phi'(x) = x                     if |x| <= delta
              delta * sign(x)       if |x| > delta
              
    This acts as a "soft clipper" for gradients, preventing ringing artifacts
    caused by large errors at edges.
    """
    return np.clip(residual, -delta, delta)

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Converts PSF to OTF (Optical Transfer Function) with correct centering.
    """
    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:kh, :kw] = psf
    
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)

def apply_conv(x: np.ndarray, F_h: np.ndarray) -> np.ndarray:
    """ Computes h * x using FFT. """
    return np.real(ifft2(F_h * fft2(x)))

def apply_corr(x: np.ndarray, F_h: np.ndarray) -> np.ndarray:
    """ Computes h_adjoint * x (Correlation) using FFT. """
    return np.real(ifft2(np.conj(F_h) * fft2(x)))

def build_gradient_filters(shape: tuple):
    """ Precomputes Finite Difference operators in Frequency Domain. """
    dh = np.zeros(shape)
    dh[0, 0] = -1; dh[0, 1] = 1
    dv = np.zeros(shape)
    dv[0, 0] = -1; dv[1, 0] = 1
    return fft2(dh), fft2(dv)

def compute_gradient(x: np.ndarray, F_dh: np.ndarray, F_dv: np.ndarray):
    """ Computes (Grad_h x, Grad_v x). """
    Fx = fft2(x)
    return np.real(ifft2(F_dh * Fx)), np.real(ifft2(F_dv * Fx))

def compute_divergence(ph: np.ndarray, pv: np.ndarray, F_dh: np.ndarray, F_dv: np.ndarray):
    """ Computes -Div(p) = L_adjoint * p. """
    return np.real(ifft2(np.conj(F_dh) * fft2(ph) + np.conj(F_dv) * fft2(pv)))

def prox_box(x: np.ndarray) -> np.ndarray:
    """ Projection onto [0, 1]. """
    return np.clip(x, 0.0, 1.0)

def prox_tv_dual(ph: np.ndarray, pv: np.ndarray, lam: float):
    """ Projection of dual variables onto L2 ball (for TV). """
    magnitude = np.sqrt(ph**2 + pv**2 + EPSILON)
    scale = np.maximum(1.0, magnitude / lam)
    return ph / scale, pv / scale

def project_simplex(h: np.ndarray) -> np.ndarray:
    """ Project vector onto simplex (sum=1, non-negative). """
    h_flat = h.ravel()
    n = len(h_flat)
    u = np.sort(h_flat)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u > cssv / ind
    
    if np.any(cond):
        rho = np.count_nonzero(cond)
        theta = cssv[rho - 1] / rho
        return np.maximum(h - theta, 0.0)
    
    return np.maximum(h, 0) / (np.sum(np.maximum(h, 0)) + EPSILON)

def init_kernel(shape: tuple, mode='gaussian') -> np.ndarray:
    """ Initialize kernel estimate. """
    kh, kw = shape
    if mode == 'gaussian':
        sigma = max(kh, kw) / 8.0
        y, x = np.ogrid[:kh, :kw]
        cy, cx = kh // 2, kw // 2
        g = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return g / g.sum()
    else:
        k = np.ones(shape)
        return k / k.sum()

def center_kernel(h: np.ndarray) -> np.ndarray:
    """ Shifts center of mass to the geometric center. """
    kh, kw = h.shape
    y, x = np.indices((kh, kw))
    m = h.sum() + EPSILON
    cy = np.sum(y * h) / m
    cx = np.sum(x * h) / m
    dy = int(round(kh // 2 - cy))
    dx = int(round(kw // 2 - cx))
    return np.roll(h, (dy, dx), axis=(0, 1))

def threshold_kernel(h: np.ndarray, thr: float) -> np.ndarray:
    """ Hard thresholding for kernel sparsity. """
    m = h.max() * thr
    h_out = h * (h > m)
    s = h_out.sum()
    if s > EPSILON:
        h_out /= s
    return h_out

def resize_image(img: np.ndarray, shape: tuple) -> np.ndarray:
    from scipy.ndimage import zoom
    return zoom(img, (shape[0]/img.shape[0], shape[1]/img.shape[1]), order=1)

def resize_kernel(ker: np.ndarray, shape: tuple) -> np.ndarray:
    from scipy.ndimage import zoom
    k = zoom(ker, (shape[0]/ker.shape[0], shape[1]/ker.shape[1]), order=1)
    k = np.maximum(k, 0)
    return k / (k.sum() + EPSILON)