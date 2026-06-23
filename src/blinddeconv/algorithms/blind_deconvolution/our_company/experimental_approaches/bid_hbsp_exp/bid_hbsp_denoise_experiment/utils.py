import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

EPSILON = 1e-12

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:

    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:kh, :kw] = psf

    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)

def otf2psf(otf: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:

    kh, kw = kernel_shape
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, kh // 2, axis=0)
    psf_full = np.roll(psf_full, kw // 2, axis=1)
    return psf_full[:kh, :kw]

def precompute_gradient_operators(
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    H, W = shape

    dx = np.zeros(shape)
    dx[0, 0] = -1
    dx[0, 1] = 1

    dy = np.zeros(shape)
    dy[0, 0] = -1
    dy[1, 0] = 1

    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx) ** 2 + np.abs(F_dy) ** 2
    return F_dx, F_dy, F_grad_sq

def forward_diff_x(u: np.ndarray) -> np.ndarray:

    return np.roll(u, -1, axis=1) - u

def forward_diff_y(u: np.ndarray) -> np.ndarray:

    return np.roll(u, -1, axis=0) - u

def adjoint_diff_x(v: np.ndarray) -> np.ndarray:

    return np.roll(v, 1, axis=1) - v

def adjoint_diff_y(v: np.ndarray) -> np.ndarray:

    return np.roll(v, 1, axis=0) - v

def compute_hs_weights(
    dx: np.ndarray,
    dy: np.ndarray,
    sigma_x: np.ndarray,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:

    sigma_grad = 2.0 * sigma_x

    nu_x = np.sqrt(dx**2 + sigma_grad + EPSILON)
    nu_y = np.sqrt(dy**2 + sigma_grad + EPSILON)

    alpha = 1.0 / b

    gamma_x = (alpha * np.tanh(alpha * nu_x)) / nu_x
    gamma_y = (alpha * np.tanh(alpha * nu_y)) / nu_y

    return gamma_x, gamma_y

def compute_hs_weights_scalar(
    x_n: np.ndarray,
    sigma_sq_n: np.ndarray,
    alpha_n: float,
) -> np.ndarray:

    xi = np.sqrt(x_n ** 2 + sigma_sq_n + EPSILON)
    theta = (alpha_n * np.tanh(alpha_n * xi)) / xi
    return theta

def project_kernel(h: np.ndarray) -> np.ndarray:

    h = np.maximum(h, 0.0)
    h_sum = h.sum()
    if h_sum > EPSILON:
        h /= h_sum
    else:
        h = np.ones_like(h) / h.size
    return h

def threshold_kernel(
    h: np.ndarray,
    ratio: float = 0.05
) -> np.ndarray:

    h = np.maximum(h, 0.0)
    h[h < ratio * np.max(h)] = 0.0
    return project_kernel(h)

def init_gaussian_kernel(
    shape: Tuple[int, int],
    sigma: float = None
) -> np.ndarray:

    kh, kw = shape
    if sigma is None:
        sigma = max(kh, kw) / 6.0
    cy, cx = kh // 2, kw // 2
    y, x = np.ogrid[-cy: kh - cy, -cx: kw - cx]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel

def fft_convolve(
    x: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:

    F_h = psf2otf(h, x.shape)
    return np.real(ifft2(F_h * fft2(x)))

from scipy.signal import fftconvolve

def edgetaper(img: np.ndarray, kernel: np.ndarray, n_taper: int = None) -> np.ndarray:

    h, w = img.shape
    kh, kw = kernel.shape

    if n_taper is None:
        n_taper = max(kh, kw)

    dx = np.arange(w)
    wx = np.ones(w)

    wx[dx < n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx < n_taper] - n_taper) / n_taper))

    wx[dx >= w - n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx >= w - n_taper] - (w - n_taper - 1)) / n_taper))

    dy = np.arange(h)
    wy = np.ones(h)

    wy[dy < n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy < n_taper] - n_taper) / n_taper))

    wy[dy >= h - n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy >= h - n_taper] - (h - n_taper - 1)) / n_taper))

    W = np.outer(wy, wx)

    blurred = fftconvolve(img, kernel, mode='same')

    return img * W + blurred * (1 - W)
