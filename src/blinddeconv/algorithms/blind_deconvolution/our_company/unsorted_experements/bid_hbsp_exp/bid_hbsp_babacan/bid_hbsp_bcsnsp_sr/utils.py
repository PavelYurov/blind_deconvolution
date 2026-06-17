"""
Utility functions for BID-HBSP + BCSNSP-SR integration.

This module combines:
  - All utility functions from the original BID-HBSP (psf2otf, gradient
    operators, HS weights, kernel ops, edgetaper, fft_convolve).
  - A fast FFT-based ``sr_initial_estimate`` that uses the SAR deconvolver
    from BCSNSP-SR (restore_sar — pure frequency-domain, O(N log N)) followed
    by a lightweight TV-IRLS refinement to sharpen the initial image estimate.

No sparse matrices are built — all operations are element-wise in the
Fourier domain, so the initialization adds only ~1-2 seconds even for
large images.

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
#  NEW: Fast FFT-based initial image estimate (SAR + TV-IRLS)
#
#  Uses restore_sar from BCSNSP-SR (pure frequency-domain Wiener filter
#  with SAR prior and automatic hyperparameter estimation) followed by
#  a few IRLS iterations with an anisotropic TV prior.
#
#  Complexity: O(N log N) per iteration — typically <1-2 s for 256×256.
#  Contrast with the old approach that called solvex_var_l4_sar which
#  builds O(N²) sparse matrices and took ~1 hour on the same image.
# ═════════════════════════════════════════════════════════════════════════════

def sr_initial_estimate(
    y: np.ndarray,
    h_init: np.ndarray,
    lambda_prior: float = 0.5,
    tv_iters: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Produce a sharper initial image estimate via FFT-based SAR + TV.

    Two-phase approach inspired by BCSNSP-SR's dual-prior philosophy
    (Salvador et al., 2013), but implemented entirely in the frequency
    domain for speed:

    Phase 1 — **SAR deconvolution** (``restore_sar`` from BCSNSP-SR):
        Frequency-domain Wiener filter with a Simultaneous Auto-Regressive
        (Laplacian) prior.  Automatically estimates regularisation (α) and
        noise precision (β) via EM — no manual tuning needed.  Produces a
        moderately sharpened image with estimated hyperparameters.

    Phase 2 — **TV-IRLS refinement** (anisotropic Lp, p ≈ 0.8):
        A few iterations of half-quadratic / IRLS with an Lp gradient
        penalty, all solved via element-wise Fourier operations (CG on
        the normal equation with FFT matvec).  This adds edge-preserving
        sparsity that SAR alone cannot provide, mirroring the TV component
        of the BCSNSP-SR combined prior.

    The ``lambda_prior`` parameter blends the two phases:
        - 1.0 → full TV refinement after SAR
        - 0.0 → SAR only (no TV step)

    Parameters
    ----------
    y            : (H, W) blurred image, float64, [0, 1].
    h_init       : (kh, kw) initial kernel estimate (Gaussian seed).
    lambda_prior : TV strength in [0, 1] — weight of the TV refinement
                   relative to the SAR result.
    tv_iters     : number of TV-IRLS iterations (Phase 2).
    verbose      : print diagnostics.

    Returns
    -------
    x0 : (H, W) sharper initial image, float64, [0, 1].
    """
    # --- Import FFT-based SAR deconvolver from BCSNSP-SR utils ---
    from blinddeconv.algorithms.super_resolution.our_company.bcsnsp_sr.utils import (
        restore_sar,
    )

    H, W = y.shape

    # ── Phase 1: SAR deconvolution (frequency-domain Wiener) ─────────
    x_sar, alpha_sar, beta_sar = restore_sar(y, h_init)
    x_sar = np.clip(x_sar, 0.0, 1.0)

    if verbose:
        print(
            f"  SAR init: α={alpha_sar:.2f}, β={beta_sar:.2f}"
        )

    if lambda_prior <= 0.0 or tv_iters <= 0:
        return x_sar

    # ── Phase 2: TV-IRLS refinement (FFT-based, no sparse matrices) ──
    #
    # Solve:  min_x  β/2 ||y - h*x||² + λ_tv Σ |∇x|^p
    # via IRLS (iteratively reweighted least squares) with p = 0.8.
    # Each iteration solves:
    #   (β H^T H + D^T W D) x = β H^T y
    # where W = diag(p |∇x_prev|^{p-2}) are the IRLS weights,
    # and the matvec uses FFT for the H^T H term.
    from scipy.sparse.linalg import LinearOperator, cg as sp_cg

    F_h = psf2otf(h_init, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y)

    beta = beta_sar
    # TV regularisation weight — scaled by lambda_prior and alpha_sar
    # so the TV term is commensurate with the data fidelity
    lambda_tv = lambda_prior * alpha_sar * 0.5

    rhs_base = beta * np.real(ifft2(F_h_conj * F_y))

    p = 0.8
    x = x_sar.copy()
    N_px = H * W

    for _i in range(tv_iters):
        dx = forward_diff_x(x)
        dy = forward_diff_y(x)

        power = (p - 2.0) / 2.0
        wx = lambda_tv * p * np.clip((dx ** 2 + 1e-8) ** power, 0.0, 1e4)
        wy = lambda_tv * p * np.clip((dy ** 2 + 1e-8) ** power, 0.0, 1e4)

        def _matvec(v_flat, _wx=wx, _wy=wy):
            v = v_flat.reshape((H, W))
            Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
            Av += adjoint_diff_x(_wx * forward_diff_x(v))
            Av += adjoint_diff_y(_wy * forward_diff_y(v))
            return Av.ravel()

        A_op = LinearOperator(
            shape=(N_px, N_px), matvec=_matvec, dtype=np.float64,
        )
        x_flat, _ = sp_cg(
            A_op, rhs_base.ravel(), x0=x.ravel(), maxiter=15, atol=1e-5,
        )
        x = np.clip(x_flat.reshape((H, W)), 0.0, 1.0)

    if verbose:
        grad_sar = float(np.sum(np.abs(np.diff(x_sar, axis=1))))
        grad_tv = float(np.sum(np.abs(np.diff(x, axis=1))))
        print(
            f"  TV refine: grad energy {grad_sar:.1f} → {grad_tv:.1f}"
        )

    return x
