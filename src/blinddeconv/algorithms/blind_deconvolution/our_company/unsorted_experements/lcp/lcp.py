"""
lcp.py

Bayesian Image Restoration under Poisson Noise and Log-Concave Prior (LCP).

Reference:
    M. Vono, N. Dobigeon, P. Chainais, "Bayesian image restoration under
    Poisson noise and log-concave prior", ICASSP 2019, Brighton, UK.

Pipeline (mirrors MATLAB exp_analysis.m / exp_synthesis.m):
    1. Normalise input to float64, scale to [0, M].
    2. Construct PSF and precompute FFT representations (HXconv).
    3. Run Split-and-Augmented Gibbs Sampler (SPA):
       - analysis mode → TV prior (SPA_analysis)
       - synthesis mode → ℓ1-wavelet prior (SPA_synthesis)
    4. Compute MMSE estimate (posterior mean over post-burn-in samples).
    5. Return restored image (int16, [0, 255]) and the known kernel.

NOTE: This is a *non-blind* restoration algorithm — the blur kernel
is assumed known and passed as a parameter.  The framework interface
still returns the kernel for consistency.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# ── Framework base class import (DO NOT MODIFY) ─────────────────────────────
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm
# ─────────────────────────────────────────────────────────────────────────────

from .solvers import SPA_analysis, SPA_synthesis
from .utils import (
    HXconv,
    fspecial_gaussian,
    daubcqf,
    mrdwt_TI2D,
    mirdwt_TI2D,
)


class LCP_BD(DeconvolutionAlgorithm):
    """
    Bayesian image restoration under Poisson noise with a log-concave prior,
    using the Split-and-Augmented Gibbs Sampler (SPA).

    Two modes are supported:

    * **analysis** — Total Variation (TV) prior on the image.
      Matches ``exp_analysis.m`` / ``SPA_analysis.m``.

    * **synthesis** — ℓ1-sparsity prior in a redundant wavelet frame
      (Haar, translation-invariant).
      Matches ``exp_synthesis.m`` / ``SPA_synthesis.m``.

    Parameters
    ----------
    mode          : str — 'analysis' (TV) or 'synthesis' (ℓ1-wavelet).
                    Default 'analysis'.
    kernel_size   : int — size of the Gaussian PSF (square).  Default 8.
    kernel_sigma  : float — standard deviation of the Gaussian PSF.
                    Default 1.0.
    beta          : float — regularisation weight.
                    For analysis (TV): recommended 0.1–1.0, default 1.0.
                    For synthesis (ℓ1): recommended 0.1–1.0, default 0.1.
    rho           : float — splitting parameter ρ.  Default 1.0.
    N_MC          : int — total number of MCMC samples.  Default 100 000.
    N_bi          : int — number of burn-in iterations.  Default 50 000.
    peak_value    : float — peak pixel value M for scaling.
                    Default 30 (analysis) or 255 (synthesis).
                    If None, auto-selected based on mode.
    wav_N         : int — Daubechies filter length for synthesis mode.
                    Default 2 (Haar wavelet).
    wav_levels    : int — number of wavelet decomposition levels.
                    Default 4.
    seed          : int or None — RNG seed for reproducibility.
    """

    def __init__(
        self,
        mode: str = 'analysis',
        kernel_size: int = 8,
        kernel_sigma: float = 1.0,
        beta: float = None,
        rho: float = 1.0,
        N_MC: int = 100_000,
        N_bi: int = 50_000,
        peak_value: float = None,
        wav_N: int = 2,
        wav_levels: int = 4,
        seed: int = None,
    ):
        super().__init__(name='LCP-BD')

        self.mode = mode
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.rho = rho
        self.N_MC = N_MC
        self.N_bi = N_bi
        self.wav_N = wav_N
        self.wav_levels = wav_levels
        self.seed = seed

        # Defaults that depend on mode
        if beta is None:
            self.beta = 1.0 if mode == 'analysis' else 0.1
        else:
            self.beta = beta

        if peak_value is None:
            self.peak_value = 30.0 if mode == 'analysis' else 255.0
        else:
            self.peak_value = peak_value

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64, scale to [0, M] ────────────────────
        # MATLAB: im = M * im ./ max(im(:));
        im = image.astype(np.float64)
        if im.max() > 0:
            im = self.peak_value * im / im.max()

        N = im.shape[0]  # assumes square image
        M = self.peak_value

        # ── 2. Construct PSF and precompute FFTs ────────────────────────
        # MATLAB: B = fspecial('gaussian', 8, 1);
        #         [FB, FBC, F2B, Bx] = HXconv(im, B, 'Hx');
        B = fspecial_gaussian(self.kernel_size, self.kernel_sigma)
        FB, FBC, F2B, _ = HXconv(im, B)

        # ── 3. Run SPA sampler ──────────────────────────────────────────
        if self.mode == 'analysis':
            X_MC, time_SPA = SPA_analysis(
                rho=self.rho,
                beta=self.beta,
                N=N,
                N_MC=self.N_MC,
                N_bi=self.N_bi,
                FBC=FBC,
                F2B=F2B,
                FB=FB,
                obs=im,
                M=M,
                seed=self.seed,
            )
        elif self.mode == 'synthesis':
            # Set up wavelet operators
            # MATLAB:
            #   wav = daubcqf(2);       % Haar
            #   levels = 4;
            #   W  = @(x) mirdwt_TI2D(x, wav, levels);   % inverse
            #   WT = @(x) mrdwt_TI2D(x, wav, levels);    % forward
            #   k  = size(WT(im), 2);
            wav, _ = daubcqf(self.wav_N)

            def W(x):
                return mirdwt_TI2D(x, wav, self.wav_levels)

            def WT(x):
                return mrdwt_TI2D(x, wav, self.wav_levels)

            k = WT(im).shape[1]

            X_MC, time_SPA = SPA_synthesis(
                rho=self.rho,
                beta=self.beta,
                N=N,
                k=k,
                N_MC=self.N_MC,
                N_bi=self.N_bi,
                FBC=FBC,
                F2B=F2B,
                FB=FB,
                obs=im,
                W=W,
                WT=WT,
                M=M,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'analysis' or 'synthesis'.")

        # ── 4. MMSE estimate (posterior mean) ────────────────────────────
        # MATLAB: X_MMSE = mean(X_MC, 3);
        X_MMSE = np.mean(X_MC, axis=2)

        # ── 5. Output ──────────────────────────────────────────────────
        # Compute MAE for diagnostics
        # MATLAB: MAE = sum(sum(abs(im - mean(X_MC,3)))) / N^2;
        MAE = np.sum(np.abs(im - X_MMSE)) / (N ** 2)
        MAE_norm = MAE / M

        self.hyperparams = {
            'mode': self.mode,
            'kernel_size': self.kernel_size,
            'kernel_sigma': self.kernel_sigma,
            'beta': self.beta,
            'rho': self.rho,
            'N_MC': self.N_MC,
            'N_bi': self.N_bi,
            'peak_value': M,
            'MAE_MMSE': float(MAE),
            'MAE_MMSE_norm': float(MAE_norm),
            'time_SPA': time_SPA,
            'time': time.time() - start_time,
        }

        # Scale back to [0, 255] and return as int16
        x_final = X_MMSE / M * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, B

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('mode', self.mode),
            ('kernel_size', self.kernel_size),
            ('kernel_sigma', self.kernel_sigma),
            ('beta', self.beta),
            ('rho', self.rho),
            ('N_MC', self.N_MC),
            ('N_bi', self.N_bi),
            ('peak_value', self.peak_value),
            ('wav_N', self.wav_N),
            ('wav_levels', self.wav_levels),
            ('seed', self.seed),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
