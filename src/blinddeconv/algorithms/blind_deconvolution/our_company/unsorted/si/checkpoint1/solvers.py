"""
solvers.py

Core solver functions for the Shift-Invariant Blind Deblurring algorithm.

Ported from MATLAB code by Hua Cheng (2011).
Reference:
    Hua Cheng, "Shift-Invariant Deblurring", part of Super-Resolution
    Project (SR.pdf), 2011.

Contains:
    coarse_kernel_est      — kernel + latent image estimation (coarse_kernel_est.m)
    multi_deriv_deconv     — multi-derivative non-blind deconvolution
                             (multi_deriv_deconv.m, Shan et al.)

MATLAB -> Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    MATLAB fft2 / ifft2 / conj -> np.fft.fft2 / np.fft.ifft2 / np.conj
        Both operate identically on 2-D arrays.

    MATLAB .* (element-wise multiply) -> Python * (on ndarrays).

    MATLAB conj(A).*B = np.conj(A) * B.

    MATLAB ifft2 on real-valued problems returns complex with tiny
    imaginary residuals.  MATLAB silently drops them when stored into
    a double variable; Python requires explicit np.real().

    MATLAB k(find(k <= 0)) = 0 -> k[k <= 0] = 0.0

    MATLAB sum(k(:)) -> k.sum()

    MATLAB psf2otf / otf2psf -> our utils.psf2otf / utils.otf2psf
        These are exact ports (zero-pad + circshift + fft2 / inverse).

    MATLAB [1,-1] is a (1,2) row vector.
    Python np.array([[1, -1]]) is shape (1,2) — must be 2-D for psf2otf.

    MATLAB [1;-1] is a (2,1) column vector.
    Python np.array([[1], [-1]]) is shape (2,1).
"""

import numpy as np
from typing import Tuple

from .utils import psf2otf, otf2psf


# ═════════════════════════════════════════════════════════════════════════════
# coarse_kernel_est  (from coarse_kernel_est.m)
# ═════════════════════════════════════════════════════════════════════════════

def coarse_kernel_est(
    Ish_x: np.ndarray,
    Ish_y: np.ndarray,
    Im_x: np.ndarray,
    Im_y: np.ndarray,
    Im: np.ndarray,
    ksize: int,
    lam: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coarse-to-fine kernel estimation + latent image update.

    Exact port of MATLAB ``coarse_kernel_est.m``.

    Solves two optimisation problems in closed form via FFT:

    **Kernel estimation:**

    .. math::

        \\min_k  \\|\\nabla I_{sh} * k - \\nabla I_m\\|^2
                + \\gamma \\|k\\|^2

    **Latent image estimation:**

    .. math::

        \\min_I  \\|I * k - I_m\\|^2
                + \\lambda \\|\\nabla I - \\nabla I_{sh}\\|^2

    Parameters
    ----------
    Ish_x, Ish_y : (H, W) selected gradient images of the shock-filtered
                   prediction  (``Is_x = Ish_x .* H``, already masked).
    Im_x, Im_y   : (H, W) gradient images of the blurred input.
    Im            : (H, W) blurred input image.
    ksize         : spatial support of the kernel (square, ksize x ksize).
    lam           : weight for gradient fidelity in the image sub-problem
                    (MATLAB variable ``lambda``).
    gamma         : weight for L2 regularisation in the kernel sub-problem
                    (MATLAB variable ``ganma``).

    Returns
    -------
    k        : (ksize, ksize) estimated kernel (non-negative, sums to 1).
    I_latent : (H, W) updated latent image estimate.

    Notes
    -----
    MATLAB ``ifft2(Nomin./Denom)`` returns complex with negligible
    imaginary part; we take ``np.real()``.
    """
    xim, yim = Im.shape

    # Derivative filters  (MATLAB: F1 = [1,-1];  F2 = [1;-1])
    F1 = np.array([[1.0, -1.0]])          # shape (1, 2)
    F2 = np.array([[1.0], [-1.0]])         # shape (2, 1)

    # FFTs
    FFtIsh_x = np.fft.fft2(Ish_x)
    FFtIsh_y = np.fft.fft2(Ish_y)
    FFtIm_x  = np.fft.fft2(Im_x)
    FFtIm_y  = np.fft.fft2(Im_y)
    FFtF1    = psf2otf(F1, (xim, yim))
    FFtF2    = psf2otf(F2, (xim, yim))
    FFtIm    = np.fft.fft2(Im)

    # ── Kernel sub-problem ──────────────────────────────────────────────
    # Nomin = conj(FFtIsh_x).*FFtIm_x + conj(FFtIsh_y).*FFtIm_y
    Nomin = np.conj(FFtIsh_x) * FFtIm_x + np.conj(FFtIsh_y) * FFtIm_y
    # Denom = conj(FFtIsh_x).*FFtIsh_x + conj(FFtIsh_y).*FFtIsh_y + ganma
    Denom = (np.conj(FFtIsh_x) * FFtIsh_x +
             np.conj(FFtIsh_y) * FFtIsh_y + gamma)
    FFtk = Nomin / Denom

    k = otf2psf(FFtk, (ksize, ksize))

    # Threshold negative values and normalise
    k[k <= 0] = 0.0
    k_sum = k.sum()
    if k_sum > 0:
        k = k / k_sum

    # ── Latent image sub-problem ────────────────────────────────────────
    FFtk = psf2otf(k, (xim, yim))
    # Nomin = conj(FFtk).*FFtIm
    #       + lambda * (conj(FFtF1).*FFtIsh_x + conj(FFtF2).*FFtIsh_y)
    Nomin = (np.conj(FFtk) * FFtIm +
             lam * (np.conj(FFtF1) * FFtIsh_x +
                    np.conj(FFtF2) * FFtIsh_y))
    # Denom = conj(FFtk).*FFtk
    #       + lambda * (conj(FFtF1).*FFtF1 + conj(FFtF2).*FFtF2)
    Denom = (np.conj(FFtk) * FFtk +
             lam * (np.conj(FFtF1) * FFtF1 +
                    np.conj(FFtF2) * FFtF2))

    I_latent = np.real(np.fft.ifft2(Nomin / Denom))

    return k, I_latent


# ═════════════════════════════════════════════════════════════════════════════
# multi_deriv_deconv  (from multi_deriv_deconv.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_denominator(
    y: np.ndarray,
    k: np.ndarray,
    weit: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    F3: np.ndarray,
    F4: np.ndarray,
    F5: np.ndarray,
    F6: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute numerator, denominator, and gradient-penalty denominator
    for multi-derivative deconvolution.

    Exact port of MATLAB ``computeDenominator`` sub-function inside
    ``multi_deriv_deconv.m``.

    Parameters
    ----------
    y    : (H, W) blurred input.
    k    : (kh, kw) blur kernel.
    weit : length-7 weight vector [w0, w1, w2, w3, w4, w5, w6].
    F1..F6 : derivative filter arrays.

    Returns
    -------
    Nomin  : numerator in the Fourier domain.
    Denom  : denominator (data-fidelity part).
    Denom2 : denominator (gradient-regularisation part).
    """
    sizey = y.shape
    otfk = psf2otf(k, sizey)

    FF1 = psf2otf(F1, sizey)
    FF2 = psf2otf(F2, sizey)
    FF3 = psf2otf(F3, sizey)
    FF4 = psf2otf(F4, sizey)
    FF5 = psf2otf(F5, sizey)
    FF6 = psf2otf(F6, sizey)

    Y = np.fft.fft2(y)
    conj_otfk = np.conj(otfk)

    # ── Numerator ───────────────────────────────────────────────────────
    # Nomin = w0 * conj(K) * Y
    #       + w1 * conj(F1) * conj(K) * F1 * Y
    #       + ...  (for each derivative filter)
    Nomin = weit[0] * conj_otfk * Y
    Nomin += weit[1] * np.conj(FF1) * conj_otfk * FF1 * Y
    Nomin += weit[2] * np.conj(FF2) * conj_otfk * FF2 * Y
    Nomin += weit[3] * np.conj(FF3) * conj_otfk * FF3 * Y
    Nomin += weit[4] * np.conj(FF4) * conj_otfk * FF4 * Y
    Nomin += weit[5] * np.conj(FF5) * conj_otfk * FF5 * Y
    Nomin += weit[6] * np.conj(FF6) * conj_otfk * FF6 * Y

    # ── Denominator (data fidelity) ─────────────────────────────────────
    # Denom = w0 * conj(K) * K
    #       + w1 * conj(F1) * conj(K) * K * F1
    #       + ...
    Denom = weit[0] * conj_otfk * otfk
    Denom += weit[1] * np.conj(FF1) * conj_otfk * otfk * FF1
    Denom += weit[2] * np.conj(FF2) * conj_otfk * otfk * FF2
    Denom += weit[3] * np.conj(FF3) * conj_otfk * otfk * FF3
    Denom += weit[4] * np.conj(FF4) * conj_otfk * otfk * FF4
    Denom += weit[5] * np.conj(FF5) * conj_otfk * otfk * FF5
    Denom += weit[6] * np.conj(FF6) * conj_otfk * otfk * FF6

    # ── Gradient-regularisation denominator ─────────────────────────────
    # Denom2 = |F1|^2 + |F2|^2
    Denom2 = np.conj(FF1) * FF1 + np.conj(FF2) * FF2

    return Nomin, Denom, Denom2


def multi_deriv_deconv(
    yin: np.ndarray,
    k: np.ndarray,
    lam: float,
) -> np.ndarray:
    """
    Multi-derivative non-blind deconvolution (Shan et al.).

    Exact port of MATLAB ``multi_deriv_deconv.m``.

    Solves:

    .. math::

        \\min_I  \\sum_j w_j \\|F_j * (I * k - I_m)\\|^2
                + \\lambda (\\|\\partial_x I\\|^2 + \\|\\partial_y I\\|^2)

    where F_j are 1st- and 2nd-order derivative filters with decreasing
    weights.

    Parameters
    ----------
    yin : (H, W) blurred input image (float64).
    k   : (kh, kw) estimated blur kernel (odd-sized).
    lam : regularisation weight (``lambda`` in MATLAB).

    Returns
    -------
    yout : (H, W) restored image (float64).
    """
    # Check odd size
    if k.shape[0] % 2 != 1 or k.shape[1] % 2 != 1:
        raise ValueError("Blur kernel k must be odd-sized.")

    # ── Weights ─────────────────────────────────────────────────────────
    # MATLAB: weight=2; weit=[weight, weight/2, weight/2,
    #                          weight/4, weight/4, weight/4, weight/4]
    weight = 2.0
    weit = np.array([weight,
                     weight / 2, weight / 2,
                     weight / 4, weight / 4, weight / 4, weight / 4])

    # ── Derivative filters ──────────────────────────────────────────────
    # MATLAB shapes:
    #   F1 = [1,-1]          -> (1,2)
    #   F2 = [1;-1]          -> (2,1)
    #   F3 = [1,-2,1]        -> (1,3)
    #   F4 = [1;-2;1]        -> (3,1)
    #   F5 = [1,-2;0,1]      -> (2,2)
    #   F6 = [1,0;-2,1]      -> (2,2)
    F1 = np.array([[1.0, -1.0]])                     # (1,2)
    F2 = np.array([[1.0], [-1.0]])                    # (2,1)
    F3 = np.array([[1.0, -2.0, 1.0]])                 # (1,3)
    F4 = np.array([[1.0], [-2.0], [1.0]])              # (3,1)
    F5 = np.array([[1.0, -2.0], [0.0, 1.0]])          # (2,2)
    F6 = np.array([[1.0, 0.0], [-2.0, 1.0]])          # (2,2)

    # ── Compute Fourier-domain quantities ───────────────────────────────
    Nomin1, Denom1, Denom2 = _compute_denominator(
        yin, k, weit, F1, F2, F3, F4, F5, F6
    )

    # ── Solve ───────────────────────────────────────────────────────────
    # MATLAB: gamma = lambda; Denom = Denom1 + gamma*Denom2;
    Denom = Denom1 + lam * Denom2
    Fyout = Nomin1 / Denom
    yout = np.real(np.fft.ifft2(Fyout))

    return yout
