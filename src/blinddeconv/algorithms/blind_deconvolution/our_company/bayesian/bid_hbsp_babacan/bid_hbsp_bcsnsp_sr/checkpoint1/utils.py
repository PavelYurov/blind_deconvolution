"""
Utility functions for BID-HBSP + BCSNSP-SR integration.

This module combines:
  - All utility functions from the original BID-HBSP (psf2otf, gradient
    operators, HS weights, kernel ops, edgetaper, fft_convolve).
  - A new ``sr_initial_estimate`` function that uses the BCSNSP-SR solver
    (solvex_var_l4_sar) at resolution factor 1 to produce a sharper
    initial image estimate for the EM loop.

The original BID-HBSP and BCSNSP-SR source files are NOT modified.

References
----------
[1] Castro-Macías et al. (2024), "Bayesian Blind Image Deconvolution
    using a Hyperbolic-Secant prior", ICIP 2024.
[2] Salvador, Villena, Molina, Katsaggelos (2013), "Bayesian Combination
    of Sparse and Non-Sparse Priors in Image Super Resolution", DSP.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from scipy.ndimage import shift as _ndshift
from typing import Tuple

EPSILON = 1e-12


# ═════════════════════════════════════════════════════════════════════════════
#  FFT-based convolution utilities (from BID-HBSP utils)
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Convert PSF to OTF (zero-pad + circshift + fft2)."""
    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Recover spatial PSF from OTF by inverse DFT and cropping."""
    kh, kw = kernel_shape
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, kh // 2, axis=0)
    psf_full = np.roll(psf_full, kw // 2, axis=1)
    return psf_full[:kh, :kw]


# ═════════════════════════════════════════════════════════════════════════════
#  Gradient operators (from BID-HBSP utils)
# ═════════════════════════════════════════════════════════════════════════════

def precompute_gradient_operators(
    shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute DFT of first-order finite-difference operators."""
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
    """Horizontal forward difference."""
    return np.roll(u, -1, axis=1) - u


def forward_diff_y(u: np.ndarray) -> np.ndarray:
    """Vertical forward difference."""
    return np.roll(u, -1, axis=0) - u


def adjoint_diff_x(v: np.ndarray) -> np.ndarray:
    """Adjoint of horizontal forward difference."""
    return np.roll(v, 1, axis=1) - v


def adjoint_diff_y(v: np.ndarray) -> np.ndarray:
    """Adjoint of vertical forward difference."""
    return np.roll(v, 1, axis=0) - v


# ═════════════════════════════════════════════════════════════════════════════
#  HS prior weights (from BID-HBSP utils)
# ═════════════════════════════════════════════════════════════════════════════

def compute_hs_weights(
    dx: np.ndarray,
    dy: np.ndarray,
    sigma_x: np.ndarray,
    b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    E[w] for the HS prior via variational approximation.
    Ref: Castro-Macías et al. (2024), Eq. (26).
    """
    sigma_grad = 2.0 * sigma_x
    nu_x = np.sqrt(dx ** 2 + sigma_grad + EPSILON)
    nu_y = np.sqrt(dy ** 2 + sigma_grad + EPSILON)
    alpha = 1.0 / b
    gamma_x = (alpha * np.tanh(alpha * nu_x)) / nu_x
    gamma_y = (alpha * np.tanh(alpha * nu_y)) / nu_y
    return gamma_x, gamma_y


# ═════════════════════════════════════════════════════════════════════════════
#  Kernel helpers (from BID-HBSP utils)
# ═════════════════════════════════════════════════════════════════════════════

def project_kernel(h: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex (h >= 0, sum = 1)."""
    h = np.maximum(h, 0.0)
    h_sum = h.sum()
    if h_sum > EPSILON:
        h /= h_sum
    else:
        h = np.ones_like(h) / h.size
    return h


def threshold_kernel(h: np.ndarray, ratio: float = 0.05) -> np.ndarray:
    """Zero entries below ratio*max, then re-normalise."""
    h = np.maximum(h, 0.0)
    h[h < ratio * np.max(h)] = 0.0
    return project_kernel(h)


def init_gaussian_kernel(
    shape: Tuple[int, int], sigma: float = None,
) -> np.ndarray:
    """Gaussian kernel normalised to unit sum."""
    kh, kw = shape
    if sigma is None:
        sigma = max(kh, kw) / 6.0
    cy, cx = kh // 2, kw // 2
    y, x = np.ogrid[-cy : kh - cy, -cx : kw - cx]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fft_convolve(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Circular convolution h * x via the FFT."""
    F_h = psf2otf(h, x.shape)
    return np.real(ifft2(F_h * fft2(x)))


def edgetaper(
    img: np.ndarray, kernel: np.ndarray, n_taper: int = None,
) -> np.ndarray:
    """Smooth image edges for FFT-based deconvolution (cf. MATLAB edgetaper)."""
    h, w = img.shape
    kh, kw = kernel.shape
    if n_taper is None:
        n_taper = max(kh, kw)
    dx = np.arange(w)
    wx = np.ones(w)
    wx[dx < n_taper] = 0.5 * (
        1 + np.cos(np.pi * (dx[dx < n_taper] - n_taper) / n_taper)
    )
    wx[dx >= w - n_taper] = 0.5 * (
        1
        + np.cos(
            np.pi
            * (dx[dx >= w - n_taper] - (w - n_taper - 1))
            / n_taper
        )
    )
    dy = np.arange(h)
    wy = np.ones(h)
    wy[dy < n_taper] = 0.5 * (
        1 + np.cos(np.pi * (dy[dy < n_taper] - n_taper) / n_taper)
    )
    wy[dy >= h - n_taper] = 0.5 * (
        1
        + np.cos(
            np.pi
            * (dy[dy >= h - n_taper] - (h - n_taper - 1))
            / n_taper
        )
    )
    W = np.outer(wy, wx)
    blurred = fftconvolve(img, kernel, mode="same")
    return img * W + blurred * (1 - W)


# ═════════════════════════════════════════════════════════════════════════════
#  NEW: SR-based initial image estimate via BCSNSP-SR solver
# ═════════════════════════════════════════════════════════════════════════════

def sr_initial_estimate(
    y: np.ndarray,
    h_init: np.ndarray,
    L: int = 4,
    max_shift: float = 0.5,
    sr_maxit: int = 15,
    lambda_prior: float = 0.5,
    pcg_thr: float = 1e-6,
    pcg_maxit: int = 80,
    pcg_minit: int = 5,
    verbose: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """Produce a sharper initial image estimate using BCSNSP-SR at res=1.

    Instead of initialising the BID-HBSP EM loop with x₀ = y (the blurred
    observation), we run the BCSNSP-SR solver with resolution factor 1 — 
    effectively a **multi-frame Bayesian deconvolution** with an anisotropic
    TV + SAR prior.  This yields a significantly sharper x₀ whose gradient
    map ∇x gives a better first kernel estimate in the Wiener step.

    The trick: with res=1 the downsampling matrix A is the identity, so the
    system matrix becomes W_k = H · C_k (convolution + sub-pixel shift).
    L pseudo-frames created from y via small sub-pixel shifts provide
    information redundancy, enabling robust regularised inversion even with
    an inaccurate initial kernel h_init.

    Parameters
    ----------
    y          : (H, W) blurred image, float64, [0, 1].
    h_init     : (kh, kw) initial kernel estimate (Gaussian seed).
    L          : number of pseudo-frames to generate.
    max_shift  : maximum sub-pixel shift magnitude (pixels).
    sr_maxit   : number of SR iterations.
    lambda_prior : TV vs SAR trade-off in [0, 1].
    pcg_thr    : PCG solver tolerance.
    pcg_maxit  : PCG max iterations.
    pcg_minit  : PCG min iterations.
    verbose    : print SR iteration info.
    seed       : random seed for reproducible shifts.

    Returns
    -------
    x0 : (H, W) sharper initial image estimate, float64, clipped to [0, 1].
    """
    # --- Import BCSNSP-SR solver (original, unmodified) ---
    from blinddeconv.algorithms.super_resolution.our_company.bcsnsp_sr.solvers import (
        solvex_var_l4_sar,
    )
    from blinddeconv.algorithms.super_resolution.our_company.bcsnsp_sr.utils import (
        fspecial_gaussian,
    )

    rng = np.random.RandomState(seed)

    H, W = y.shape
    # res=1 → M=H, N=W, m=H, n=W (no downsampling)
    M, N = H, W
    m, n = H, W
    res = 1

    # --- Generate L pseudo-frames via sub-pixel shifts ---
    sx = np.zeros(L)
    sy = np.zeros(L)
    theta = np.zeros(L)

    frames = [y.ravel(order="F")]
    for k in range(1, L):
        sx[k] = (rng.rand() * 2 - 1) * max_shift
        sy[k] = (rng.rand() * 2 - 1) * max_shift
        shifted = _ndshift(y, [sy[k], sx[k]], order=1, mode="reflect")
        frames.append(shifted.ravel(order="F"))

    y_stacked = np.concatenate(frames)

    # --- Run BCSNSP-SR solver at res=1 ---
    x_vec, _out = solvex_var_l4_sar(
        y_stacked,
        M=M, N=N, m=m, n=n,
        res=res, L=L, h=h_init,
        sx=sx, sy=sy, theta=theta,
        xtrue=None,
        method="variational",
        lambda_prior=lambda_prior,
        maxit=sr_maxit,
        thr=1e-4,
        pcg_thr=pcg_thr,
        pcg_maxit=pcg_maxit,
        pcg_minit=pcg_minit,
        estimate_registration=False,  # shifts are known exactly
        verbose=verbose,
    )

    x0 = x_vec.reshape(M, N, order="F")
    x0 = np.clip(x0, 0.0, 1.0)

    return x0
