import numpy as np
from numpy.fft import fft2, ifft2
from scipy.special import erf
from scipy.signal import fftconvolve
from typing import Tuple

EPSILON = 1e-12

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    in_shape = psf.shape
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:in_shape[0], :in_shape[1]] = psf
    psf_padded = np.roll(psf_padded, -in_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -in_shape[1] // 2, axis=1)
    return fft2(psf_padded)

def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    psf_padded = np.real(ifft2(otf))
    psf_padded = np.roll(psf_padded, out_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, out_shape[1] // 2, axis=1)
    return psf_padded[:out_shape[0], :out_shape[1]]

def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

def compute_spatial_gradient(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    return dx, dy

def precompute_gradient_operators(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = shape
    dx = np.zeros((H, W)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((H, W)); dy[0, 0] = -1; dy[1, 0] = 1
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
    return F_dx, F_dy, F_grad_sq

def truncated_gaussian_moments(m: float, v: float) -> Tuple[float, float]:
    if v <= 0: return max(m, 0.0), 0.0
    sqrt_v = np.sqrt(v)
    z = m / sqrt_v
    if z < -10: return 0.0, 0.0
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
    if Phi < EPSILON: return 0.0, 0.0
    ratio = phi / Phi
    mean_t = m + sqrt_v * ratio
    var_t = v * (1 + z * ratio - ratio**2)
    return mean_t, var_t

def edgetaper(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Slightly blurs the edges of the image to avoid ringing artifacts in FFT-based deconvolution.
    Simulates Matlab's edgetaper.
    """
    alpha = 0.5 
    H, W = img.shape
    kh, kw = kernel.shape
    
    pad_h = kh // 2
    pad_w = kw // 2
    
    if pad_h == 0 or pad_w == 0:
        return img

    blurred = fftconvolve(img, kernel, mode='same')
    
    w_y = np.ones(H)
    w_x = np.ones(W)
    

    taper_h = np.hanning(pad_h * 4) 
    taper_w = np.hanning(pad_w * 4)
    
    w_y[:pad_h*2] = taper_h[:pad_h*2]
    w_y[-pad_h*2:] = taper_h[-pad_h*2:]
    
    w_x[:pad_w*2] = taper_w[:pad_w*2]
    w_x[-pad_w*2:] = taper_w[-pad_w*2:]
    
    mask = w_y[:, None] * w_x[None, :]
    
    return img * mask + blurred * (1 - mask)