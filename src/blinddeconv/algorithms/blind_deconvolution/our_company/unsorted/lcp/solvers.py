"""
solvers.py

Core solver functions for the LCP (Log-Concave Prior) Bayesian image
restoration under Poisson noise.

Ported from MATLAB code by Maxime Vono (ICASSP 2019).
Reference:
    M. Vono, N. Dobigeon, P. Chainais, "Bayesian image restoration under
    Poisson noise and log-concave prior", ICASSP 2019, Brighton, UK.

Contains:
    SPA_analysis  — Split-and-Augmented Gibbs Sampler with TV prior
                    (analysis formulation).  From SPA_analysis.m.
    SPA_synthesis — Split-and-Augmented Gibbs Sampler with ℓ1-wavelet
                    prior (synthesis formulation).  From SPA_synthesis.m.

MATLAB → Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    randi([0 M], N, N):
        MATLAB randi([lo hi], rows, cols) generates integer uniform
        samples in [lo, hi] *inclusive*.
        → np.random.randint(0, M + 1, (N, N)) because NumPy's high
          bound is *exclusive*.

    randn(N, N):
        Both MATLAB and NumPy use standard normal N(0,1).

    sqrt(-1) * randn(…):
        MATLAB uses sqrt(-1) to get the imaginary unit i.
        → 1j * np.random.randn(…) in Python.

    fft2 / ifft2 / real:
        Identical semantics in MATLAB and NumPy.

    max(z, 0):
        Element-wise maximum.  → np.maximum(z, 0).

    chambolle_prox_TV_stop(z, 'lambda', val, 'maxiter', val):
        MATLAB name-value pairs → keyword arguments in Python.

    sign(z) .* max(abs(z) - t, zeros(…)):
        Soft-thresholding.  → np.sign(z) * np.maximum(np.abs(z) - t, 0).
"""

import numpy as np
import time
from typing import Tuple, Optional, Callable
from numpy.fft import fft2, ifft2

from .utils import chambolle_prox_TV_stop


# ═════════════════════════════════════════════════════════════════════════════
# SPA_analysis  (from SPA_analysis.m)
# ═════════════════════════════════════════════════════════════════════════════

def SPA_analysis(rho: float,
                 beta: float,
                 N: int,
                 N_MC: int,
                 N_bi: int,
                 FBC: np.ndarray,
                 F2B: np.ndarray,
                 FB: np.ndarray,
                 obs: np.ndarray,
                 M: float,
                 seed: Optional[int] = None,
                 callback: Optional[Callable] = None,
                 ) -> Tuple[np.ndarray, float]:
    """
    Split-and-Augmented Gibbs Sampler — **analysis** formulation (TV prior).

    Exact port of SPA_analysis.m.

    The posterior model (analysis):
        p(x | y) ∝ Π_i  (Hx)_i^{y_i} exp(-(Hx)_i) / y_i!
                    · exp(-β · TV(x))
                    · 1_{x ≥ 0}

    The algorithm introduces three pairs of auxiliary variables
    (z_j, u_j), j=1,2,3, and iteratively samples:
        x   — Gaussian conditional (closed-form in Fourier domain)
        z1  — P-MYULA step (Poisson likelihood proximal)
        z2  — P-MYULA step (TV proximal via Chambolle)
        z3  — P-MYULA step (non-negativity proximal)
        u1, u2, u3 — Gaussian conditionals (augmented variables)

    Parameters
    ----------
    rho   : splitting parameter ρ.
    beta  : TV regularisation weight β.
    N     : image size (square N×N).
    N_MC  : total number of MCMC iterations.
    N_bi  : number of burn-in iterations (samples before this are discarded).
    FBC   : conj(FB) — conjugate of FFT of the PSF.
    F2B   : |FB|^2 — squared modulus of FFT of the PSF.
    FB    : FFT of the zero-padded, centred PSF (from HXconv).
    obs   : (N, N) observed (Poisson-corrupted, blurred) image.
    M     : peak value used for initialisation (dynamic range).
    seed  : optional RNG seed for reproducibility.
    callback : optional callable(t, X) invoked every iteration for progress.

    Returns
    -------
    X_MC     : (N, N, N_MC - N_bi) array of post-burn-in samples.
    time_SPA : elapsed wall-clock time (seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    # ── Parameters ───────────────────────────────────────────────────────
    alpha = rho                     # augmentation parameter
    lambdaPMYULA = rho ** 2         # P-MYULA parameter
    gammaPMYULA = rho ** 2 / 4.0    # P-MYULA step-size

    N_MC = int(N_MC)
    N_bi = int(N_bi)

    # ── Initialisation ──────────────────────────────────────────────────
    # MATLAB: randi([0 M], N, N) — integers in [0, M] inclusive
    M_int = int(M)
    z1 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    z2 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    z3 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    u1 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    u2 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    u3 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)

    n_samples = N_MC - N_bi
    X_MC = np.zeros((N, N, n_samples), dtype=np.float64)

    t_start = time.time()

    for t in range(1, N_MC + 1):

        # ── Sample x (Gaussian in Fourier domain) ───────────────────────
        # MATLAB:
        #   cov = rho^2 ./ (F2B + 2);
        #   moy = (1/rho^2) * cov .* (FBC .* fft2(z1-u1) + fft2(z2+z3-u2-u3));
        #   eps = sqrt(0.5) * (randn(N,N) + sqrt(-1)*randn(N,N));
        #   X = real(ifft2(moy + eps .* sqrt(cov)));
        cov = (rho ** 2) / (F2B + 2.0)
        moy = (1.0 / rho ** 2) * cov * (
            FBC * fft2(z1 - u1) + fft2(z2 + z3 - u2 - u3)
        )
        eps = np.sqrt(0.5) * (
            np.random.randn(N, N) + 1j * np.random.randn(N, N)
        )
        X = np.real(ifft2(moy + eps * np.sqrt(cov)))

        if t > N_bi:
            X_MC[:, :, t - N_bi - 1] = X

        # ── Sample z1 (P-MYULA — Poisson likelihood) ────────────────────
        u_noise = np.random.randn(N, N)
        gradH1 = (1.0 / rho ** 2) * np.real(
            ifft2(fft2(z1 - u1) - FB * fft2(X))
        )
        proxH1 = 0.5 * (
            z1 - lambdaPMYULA
            + np.sqrt((z1 - lambdaPMYULA) ** 2 + 4.0 * lambdaPMYULA * obs)
        )
        z1 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z1
              - gammaPMYULA * gradH1
              + (gammaPMYULA / lambdaPMYULA) * proxH1
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample z2 (P-MYULA — TV proximal) ───────────────────────────
        u_noise = np.random.randn(N, N)
        gradH2 = (1.0 / rho ** 2) * (z2 - X - u2)
        proxH2, _, _ = chambolle_prox_TV_stop(
            z2, lam=beta * lambdaPMYULA, maxiter=10
        )
        z2 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z2
              - gammaPMYULA * gradH2
              + (gammaPMYULA / lambdaPMYULA) * proxH2
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample z3 (P-MYULA — non-negativity) ────────────────────────
        u_noise = np.random.randn(N, N)
        gradH3 = (1.0 / rho ** 2) * (z3 - X - u3)
        proxH3 = np.maximum(z3, 0.0)
        z3 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z3
              - gammaPMYULA * gradH3
              + (gammaPMYULA / lambdaPMYULA) * proxH3
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample u1 (Gaussian) ────────────────────────────────────────
        coeff_mean = alpha ** 2 / (alpha ** 2 + rho ** 2)
        coeff_std = alpha * rho / np.sqrt(alpha ** 2 + rho ** 2)
        u1 = (coeff_mean * (z1 - np.real(ifft2(FB * fft2(X))))
              + coeff_std * np.random.randn(N, N))

        # ── Sample u2 (Gaussian) ────────────────────────────────────────
        u2 = (coeff_mean * (z2 - X)
              + coeff_std * np.random.randn(N, N))

        # ── Sample u3 (Gaussian) ────────────────────────────────────────
        u3 = (coeff_mean * (z3 - X)
              + coeff_std * np.random.randn(N, N))

        if callback is not None:
            callback(t, X)

    time_SPA = time.time() - t_start
    return X_MC, time_SPA


# ═════════════════════════════════════════════════════════════════════════════
# SPA_synthesis  (from SPA_synthesis.m)
# ═════════════════════════════════════════════════════════════════════════════

def SPA_synthesis(rho: float,
                  beta: float,
                  N: int,
                  k: int,
                  N_MC: int,
                  N_bi: int,
                  FBC: np.ndarray,
                  F2B: np.ndarray,
                  FB: np.ndarray,
                  obs: np.ndarray,
                  W: Callable,
                  WT: Callable,
                  M: float,
                  seed: Optional[int] = None,
                  callback: Optional[Callable] = None,
                  ) -> Tuple[np.ndarray, float]:
    """
    Split-and-Augmented Gibbs Sampler — **synthesis** formulation
    (ℓ1-wavelet prior).

    Exact port of SPA_synthesis.m.

    The posterior model (synthesis):
        x = W θ,
        p(θ | y) ∝ p(y | W θ) · exp(-β ‖θ‖₁) · 1_{Wθ ≥ 0}

    Parameters
    ----------
    rho   : splitting parameter ρ.
    beta  : ℓ1 regularisation weight β.
    N     : spatial image size (square N×N).
    k     : number of wavelet-coefficient columns (= size(WT(im), 2)).
    N_MC  : total number of MCMC iterations.
    N_bi  : number of burn-in iterations.
    FBC   : conj(FB).
    F2B   : |FB|^2.
    FB    : FFT of the zero-padded, centred PSF.
    obs   : (N, N) observed Poisson-corrupted blurred image.
    W     : inverse wavelet transform  θ → x  (callable, N×k → N×N).
    WT    : forward wavelet transform  x → θ  (callable, N×N → N×k).
    M     : peak value (dynamic range).
    seed  : optional RNG seed.
    callback : optional callable(t, X_image) invoked every iteration.

    Returns
    -------
    X_MC     : (N, N, N_MC - N_bi) post-burn-in *image* samples (= W(θ)).
    time_SPA : elapsed wall-clock time (seconds).
    """
    if seed is not None:
        np.random.seed(seed)

    # ── Parameters ───────────────────────────────────────────────────────
    alpha = 0.1 * rho               # augmentation parameter (note: 0.1 * rho)
    lambdaPMYULA = rho ** 2
    gammaPMYULA = rho ** 2 / 4.0

    N_MC = int(N_MC)
    N_bi = int(N_bi)

    # ── Initialisation ──────────────────────────────────────────────────
    M_int = int(M)
    z1 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    z2 = np.random.randint(0, M_int + 1, (N, k)).astype(np.float64)
    z3 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    u1 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)
    u2 = np.random.randint(0, M_int + 1, (N, k)).astype(np.float64)
    u3 = np.random.randint(0, M_int + 1, (N, N)).astype(np.float64)

    n_samples = N_MC - N_bi
    X_MC = np.zeros((N, N, n_samples), dtype=np.float64)

    t_start = time.time()

    for t in range(1, N_MC + 1):

        # ── Sample θ (Woodbury / wavelet-domain) ────────────────────────
        # MATLAB:
        #   eta1 = z1 - u1 + rho * randn(N,N);
        #   eta2 = z2 - u2 + rho * randn(N,k);
        #   eta3 = z3 - u3 + rho * randn(N,N);
        #   eta = WT(real(ifft2(FBC .* fft2(eta1)))) + eta2 + WT(eta3);
        #   ratio = 1 ./ (1 + 1 ./ (F2B + 1));
        #   X = eta - WT(real(ifft2(ratio .* fft2(W(eta)))));
        eta1 = z1 - u1 + rho * np.random.randn(N, N)
        eta2 = z2 - u2 + rho * np.random.randn(N, k)
        eta3 = z3 - u3 + rho * np.random.randn(N, N)

        eta = (WT(np.real(ifft2(FBC * fft2(eta1))))
               + eta2
               + WT(eta3))

        ratio = 1.0 / (1.0 + 1.0 / (F2B + 1.0))
        X = eta - WT(np.real(ifft2(ratio * fft2(W(eta)))))

        if t > N_bi:
            X_MC[:, :, t - N_bi - 1] = W(X)

        # ── Sample z1 (P-MYULA — Poisson likelihood) ────────────────────
        u_noise = np.random.randn(N, N)
        WX = W(X)   # cache W(X) — used multiple times below
        gradH1 = (1.0 / rho ** 2) * np.real(
            ifft2(fft2(z1 - u1) - FB * fft2(WX))
        )
        proxH1 = 0.5 * (
            z1 - lambdaPMYULA
            + np.sqrt((z1 - lambdaPMYULA) ** 2 + 4.0 * lambdaPMYULA * obs)
        )
        z1 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z1
              - gammaPMYULA * gradH1
              + (gammaPMYULA / lambdaPMYULA) * proxH1
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample z2 (P-MYULA — ℓ1 soft-thresholding) ──────────────────
        u_noise = np.random.randn(N, k)
        gradH2 = (1.0 / rho ** 2) * (z2 - X - u2)
        # Soft-thresholding: sign(z2) .* max(|z2| - β·λ_PMYULA, 0)
        proxH2 = np.sign(z2) * np.maximum(
            np.abs(z2) - beta * lambdaPMYULA, 0.0
        )
        z2 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z2
              - gammaPMYULA * gradH2
              + (gammaPMYULA / lambdaPMYULA) * proxH2
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample z3 (P-MYULA — non-negativity) ────────────────────────
        u_noise = np.random.randn(N, N)
        gradH3 = (1.0 / rho ** 2) * (z3 - WX - u3)
        proxH3 = np.maximum(z3, 0.0)
        z3 = ((1.0 - gammaPMYULA / lambdaPMYULA) * z3
              - gammaPMYULA * gradH3
              + (gammaPMYULA / lambdaPMYULA) * proxH3
              + np.sqrt(2.0 * gammaPMYULA) * u_noise)

        # ── Sample u1 (Gaussian) ────────────────────────────────────────
        coeff_mean = alpha ** 2 / (alpha ** 2 + rho ** 2)
        coeff_std = alpha * rho / np.sqrt(alpha ** 2 + rho ** 2)
        u1 = (coeff_mean * (z1 - np.real(ifft2(FB * fft2(WX))))
              + coeff_std * np.random.randn(N, N))

        # ── Sample u2 (Gaussian) ────────────────────────────────────────
        u2 = (coeff_mean * (z2 - X)
              + coeff_std * np.random.randn(N, k))

        # ── Sample u3 (Gaussian) ────────────────────────────────────────
        u3 = (coeff_mean * (z3 - WX)
              + coeff_std * np.random.randn(N, N))

        if callback is not None:
            callback(t, WX)

    time_SPA = time.time() - t_start
    return X_MC, time_SPA
