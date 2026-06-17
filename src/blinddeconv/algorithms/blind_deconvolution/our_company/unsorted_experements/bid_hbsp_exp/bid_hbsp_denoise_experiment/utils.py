"""
Utility functions for BID-HBSP: Bayesian Blind Image Deconvolution
with Hyperbolic-Secant Prior.

Provides:
    - FFT-based convolution utilities (psf2otf, otf2psf)
    - Spatial gradient operators and their adjoints
    - Hyperbolic-Secant prior weight computation (Gaussian Scale Mixture)
    - Kernel projection, thresholding, and initialization

References
[1] Castro-Macías, Pérez-Bueno, et al. (2024), "Bayesian Blind Image
    Deconvolution using a Hyperbolic-Secant prior", ICIP 2024.
[2] Babacan, Molina, Katsaggelos (2009), "Variational Bayesian Blind
    Deconvolution Using a Total Variation Prior", IEEE TIP, 18(1).
[3] Polson & Scott (2016), "Mixtures, envelopes and hierarchical duality",
    J. R. Statist. Soc. B, 78(3), pp. 701–727.
[4] Datta, Ghosh & Polson (2024), "Bayesian ICA with super-Gaussian
    Source Priors", arXiv:2406.17058v3, Sec. 3.1 & Appendix B.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

EPSILON = 1e-12



def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert Point Spread Function (PSF) to Optical Transfer Function (OTF).

    The PSF is zero-padded to the target *shape* and circularly shifted so
    that the kernel centre sits at index (0, 0) before taking the 2-D DFT.

    Parameters
    psf : ndarray, shape (kh, kw)
        Point spread function (blur kernel).
    shape : tuple (H, W)
        Target spatial dimensions of the OTF.

    Returns
    otf : ndarray, shape (H, W), complex
        Optical transfer function.
    """
    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:kh, :kw] = psf
    # Centre the kernel at the origin for correct phase
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Recover a spatial PSF from its OTF by inverse DFT and cropping.

    Parameters
    otf : ndarray, shape (H, W), complex
    kernel_shape : (kh, kw)

    Returns
    psf : ndarray, shape (kh, kw), real
    """
    kh, kw = kernel_shape
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, kh // 2, axis=0)
    psf_full = np.roll(psf_full, kw // 2, axis=1)
    return psf_full[:kh, :kw]


def precompute_gradient_operators(
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute DFT representations of first-order finite-difference operators.

    Forward differences with periodic (wrap-around) boundaries::

        (C_x u)[i, j] = u[i, j+1] - u[i, j]      (horizontal)
        (C_y u)[i, j] = u[i+1, j] - u[i, j]      (vertical)

    Returns
    F_dx : ndarray, complex — DFT of horizontal difference kernel
    F_dy : ndarray, complex — DFT of vertical   difference kernel
    F_grad_sq : ndarray, real — |F_dx|² + |F_dy|²  (Laplacian spectrum)
    """
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
    """Horizontal forward difference: (C_x u)[i,j] = u[i, j+1] - u[i, j]."""
    return np.roll(u, -1, axis=1) - u


def forward_diff_y(u: np.ndarray) -> np.ndarray:
    """Vertical forward difference: (C_y u)[i,j] = u[i+1, j] - u[i, j]."""
    return np.roll(u, -1, axis=0) - u


def adjoint_diff_x(v: np.ndarray) -> np.ndarray:
    r"""Adjoint of the horizontal forward difference operator.

    .. math::
        (C_x^\top v)[i,j] = v[i, j-1] - v[i, j]
    """
    return np.roll(v, 1, axis=1) - v


def adjoint_diff_y(v: np.ndarray) -> np.ndarray:
    r"""Adjoint of the vertical forward difference operator.

    .. math::
        (C_y^\top v)[i,j] = v[i-1, j] - v[i, j]
    """
    return np.roll(v, 1, axis=0) - v



def compute_hs_weights(
    dx: np.ndarray, 
    dy: np.ndarray, 
    sigma_x: np.ndarray,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes E[w] for the HS prior using the variational approximation.
    Ref: Castro-Macías et al. (2024), Eq. (26).
    """
    sigma_grad = 2.0 * sigma_x
    
    # Second moment E[u^2] = mean^2 + var
    # Need an argument for tanh: sqrt(E[u^2])
    # Ref. Appendix C or Eq (26) where ksi = sqrt(E[x^2])
    
    nu_x = np.sqrt(dx**2 + sigma_grad + EPSILON)
    nu_y = np.sqrt(dy**2 + sigma_grad + EPSILON)
    
    # alpha = 1/b. Formula: (alpha * tanh(alpha * nu)) / nu
    alpha = 1.0 / b
    
    gamma_x = (alpha * np.tanh(alpha * nu_x)) / nu_x
    gamma_y = (alpha * np.tanh(alpha * nu_y)) / nu_y
    
    return gamma_x, gamma_y


def compute_hs_weights_scalar(
    x_n: np.ndarray,
    sigma_sq_n: np.ndarray,
    alpha_n: float,
) -> np.ndarray:
    r"""Compute HS weights :math:`E[\omega]` for one filtered image (filter space).

    In the filter-space VB formulation the prior is placed directly on
    the pixels of the filtered image :math:`x_n = F_n x`, so the weight
    update is a scalar (per-pixel) formula without gradient operators.

    .. math::
        \xi_n^i = \sqrt{m_{x_n}^2(i) + \Sigma_{x_n}(i,i)}, \qquad
        E[\omega_n^i] = \frac{\alpha_n \tanh(\alpha_n \xi_n^i)}{\xi_n^i}.

    Parameters
    ----------
    x_n : ndarray (H, W)
        Posterior mean of the *n*-th filtered image :math:`m_{x_n}`.
    sigma_sq_n : ndarray (H, W)
        Diagonal of the posterior covariance :math:`\Sigma_{x_n}(i,i)`.
    alpha_n : float
        HS scale parameter :math:`\alpha_n = 1/b`.

    Returns
    -------
    theta_n : ndarray (H, W)
        Diagonal HS weights :math:`E[\omega_n^i]`.

    Reference: Castro-Macías et al. (2024), Eq. (26).
    """
    xi = np.sqrt(x_n ** 2 + sigma_sq_n + EPSILON)
    theta = (alpha_n * np.tanh(alpha_n * xi)) / xi
    return theta


def project_kernel(h: np.ndarray) -> np.ndarray:
    r"""Project a kernel onto the probability simplex :math:`h \ge 0,\;\sum h = 1`."""
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
    """Threshold small kernel values (promote sparsity) then re-normalise.

    Elements below ``ratio * max(h)`` are zeroed out.

    Parameters
    ----------
    h : ndarray — kernel (non-negative expected)
    ratio : float — fraction of peak below which values are zeroed
    """
    h = np.maximum(h, 0.0)
    h[h < ratio * np.max(h)] = 0.0
    return project_kernel(h)


def init_gaussian_kernel(
    shape: Tuple[int, int],
    sigma: float = None
) -> np.ndarray:
    """Create a Gaussian kernel normalised to unit sum.

    Parameters
    ----------
    shape : (kh, kw)
    sigma : float, optional
        Standard deviation; defaults to ``max(kh, kw) / 6``.
    """
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
    """Circular convolution :math:`h * x` via the FFT.

    Parameters
    x : ndarray (H, W) — image
    h : ndarray (kh, kw) — kernel

    Returns
    y : ndarray (H, W) — convolved image
    """
    F_h = psf2otf(h, x.shape)
    return np.real(ifft2(F_h * fft2(x)))


from scipy.signal import fftconvolve

def edgetaper(img: np.ndarray, kernel: np.ndarray, n_taper: int = None) -> np.ndarray:
    """
    Smooths the edges of the image to reduce ringing artifacts in FFT-based deconvolution.
    Simulates Matlab's edgetaper.
    """
    h, w = img.shape
    kh, kw = kernel.shape
    
    if n_taper is None:
        n_taper = max(kh, kw)
        
    # Create tapering weights (hanning window-like)
    # 1. Horizontal
    dx = np.arange(w)
    wx = np.ones(w)
    # Left edge
    wx[dx < n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx < n_taper] - n_taper) / n_taper))
    # Right edge
    wx[dx >= w - n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx >= w - n_taper] - (w - n_taper - 1)) / n_taper))
    
    # 2. Vertical
    dy = np.arange(h)
    wy = np.ones(h)
    # Top edge
    wy[dy < n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy < n_taper] - n_taper) / n_taper))
    # Bottom edge
    wy[dy >= h - n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy >= h - n_taper] - (h - n_taper - 1)) / n_taper))
    
    # 2D weights
    W = np.outer(wy, wx)
    
    # Blur the image with the kernel (to match boundary conditions)
    blurred = fftconvolve(img, kernel, mode='same')
    
    # Blend: Center is original image, Borders are blurred version
    # This makes the image cyclic-consistent for FFT
    return img * W + blurred * (1 - W)