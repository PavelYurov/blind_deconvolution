"""
sdp.py

Blind Image Deconvolution via Convex Programming (SDP / Nuclear-Norm).

Reference:
    A. Ahmed, B. Recht, J. Romberg: "Blind Deconvolution Using Convex
    Programming", IEEE Trans. Inform. Theory, 2014. (arXiv:1211.5608)

Pipeline (mirrors MATLAB DeblurAlgorithm1.m):
    1. Normalise input to float64 [0, 1].
    2. Build kernel subspace B from kernel_shape (uniform initialization).
    3. Build image wavelet subspace C from blurred image.
    4. Solve convex program via ALM with Burer-Monteiro factorisation.
    5. Extract rank-1 estimates via SVD.
    6. Reconstruct image and kernel.
    7. Return restored image (int16, [0, 255]) and kernel.

Note:
    The original MATLAB code requires the ground-truth kernel support
    (non-zero pattern) to build matrix B.  In a truly blind setting
    we do not have this.  Instead, we initialise B as a full kernel of
    the given kernel_shape — every pixel in the kernel window is
    assumed potentially non-zero.  This is equivalent to setting the
    kernel subspace to the full space of (kernel_shape) images.
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

from .utils import (
    vec, mat,
    build_kernel_subspace,
    build_image_subspace,
    make_CC_operator,
    make_BB_operator,
    extract_estimates,
)
from .solvers import blind_deconvolve_implicit_2d


class ConvexSDP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution via convex programming (nuclear-norm relaxation).

    Parameters
    ----------
    kernel_shape : tuple of int
        (height, width) of the blur kernel window.  Every pixel in this
        window is assumed potentially non-zero.  Default (11, 11).
    wavelet_level : int
        Number of wavelet decomposition levels for the image subspace.
        Default 4 (matches the paper).
    wavelet : str
        Wavelet family.  Default 'db1' (Haar).
    threshold_ratio : float
        Wavelet coefficient threshold: coefficients with
        |c| > threshold_ratio * max(|c|) are kept.  Default 0.0005.
    maxrank : int
        Rank of the Burer-Monteiro factorisation.  Default 4.
    max_out_iter : int
        Maximum outer ALM iterations.  Default 25.
    rmse_tol : float
        RMSE convergence tolerance.  Default 1e-8.
    sigma_init : float
        Initial ALM penalty parameter.  Default 1e4.
    max_fun_evals : int
        Maximum L-BFGS function evaluations per inner solve.
        Default 50000.
    verbose : bool
        Print solver diagnostics.  Default False.
    oracle_kernel : ndarray or None
        If provided, a 2-D array containing the ground-truth blur kernel.
        Its non-zero support will be used to build matrix B (as in the
        original MATLAB code), which greatly reduces the kernel subspace
        dimension K and improves recovery quality.  Default None (blind).
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (11, 11),
        wavelet_level: int = 4,
        wavelet: str = 'db1',
        threshold_ratio: float = 0.0005,
        maxrank: int = 4,
        max_out_iter: int = 25,
        rmse_tol: float = 1e-8,
        sigma_init: float = 1e4,
        max_fun_evals: int = 50000,
        verbose: bool = False,
        oracle_kernel: np.ndarray = None,
    ):
        super().__init__(name='Convex-SDP-BD')

        self.kernel_shape = kernel_shape
        self.wavelet_level = wavelet_level
        self.wavelet = wavelet
        self.threshold_ratio = threshold_ratio
        self.maxrank = maxrank
        self.max_out_iter = max_out_iter
        self.rmse_tol = rmse_tol
        self.sigma_init = sigma_init
        self.max_fun_evals = max_fun_evals
        self.verbose = verbose
        self.oracle_kernel = oracle_kernel

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # Ensure grayscale 2-D
        if y.ndim == 3 and y.shape[2] == 3:
            y = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3:
            y = y[:, :, 0]

        L1, L2 = y.shape
        L = L1 * L2

        # ── 2. Build kernel subspace B ───────────────────────────────────
        #  If oracle_kernel is provided (2D array same size as y),
        #  build B from its TRUE non-zero support — matching MATLAB's
        #  DeblurAlgorithm1.m behaviour.  Otherwise, use a full
        #  kernel_shape window as the support.
        kh, kw = self.kernel_shape
        if self.oracle_kernel is not None:
            # Oracle: embed in (L1, L2) and use actual support
            w_oracle = np.zeros((L1, L2), dtype=np.float64)
            oh, ow = self.oracle_kernel.shape
            w_oracle[:oh, :ow] = self.oracle_kernel
            B, _h_init, K = build_kernel_subspace(w_oracle)
        else:
            w_init = np.zeros((L1, L2), dtype=np.float64)
            w_init[:kh, :kw] = 1.0 / (kh * kw)
            B, _h_init, K = build_kernel_subspace(w_init)
        BB, BBT = make_BB_operator(B, L1, L2)

        if self.verbose:
            print(f'Kernel subspace: K = {K}  (kernel_shape = {self.kernel_shape})')

        # ── 3. Build image wavelet subspace C ────────────────────────────
        C, m_blurred, N, bookkeeping = build_image_subspace(
            y, L1, L2,
            level=self.wavelet_level,
            wavelet=self.wavelet,
            threshold_ratio=self.threshold_ratio,
        )
        CC, CCT = make_CC_operator(C, bookkeeping, self.wavelet)

        if self.verbose:
            print(f'Image subspace: N = {N}  (wavelet_level = {self.wavelet_level})')

        # ── 4. Solve convex program ──────────────────────────────────────
        #  MATLAB: [M, H] = blindDeconvolve_implicit_2D(vec(y), CC, BB, 4, CCT, BBT)
        conv_zh = vec(y)

        solver_pars = {
            'maxOutIter': self.max_out_iter,
            'rmseTol': self.rmse_tol,
            'sigmaInit': self.sigma_init,
            'maxFunEvals': self.max_fun_evals,
        }

        Z, H = blind_deconvolve_implicit_2d(
            conv_zh=conv_zh,
            C1=CC, C2=BB,
            maxrank=self.maxrank,
            C1T=CCT, C2T=BBT,
            n1=N, n2=K,
            L1=L1, L2=L2,
            pars=solver_pars,
            verbose=self.verbose,
        )

        # ── 5. Extract rank-1 estimates via SVD ─────────────────────────
        mEst, hEst = extract_estimates(Z, H)

        # ── 6. Reconstruct image and kernel ──────────────────────────────
        #  MATLAB: xEst = CC(mEst)
        xEst = CC(mEst)

        #  MATLAB: xEst = (x(1,1)/xEst(1,1)) * (xEst - min(min(xEst)))
        #  The nuclear-norm minimisation has a sign ambiguity:
        #  (mEst, hEst) and (-mEst, -hEst) yield the same product.
        #  Resolve by assuming the image should have positive mean.
        if np.mean(xEst) < 0:
            xEst = -xEst
            hEst = -hEst

        #  Without ground-truth x(1,1), we normalise to [0, 1].
        xEst = xEst - np.min(xEst)
        x_max = np.max(xEst)
        if x_max > 0:
            xEst = xEst / x_max

        #  MATLAB: wEst = BB(hEst)
        wEst = BB(hEst)

        # Extract the kernel window and normalise
        kernel = wEst[:kh, :kw].copy()
        k_sum = np.sum(kernel)
        if k_sum > 0:
            kernel = kernel / k_sum
        elif np.sum(np.abs(kernel)) > 0:
            kernel = np.abs(kernel)
            kernel = kernel / np.sum(kernel)

        # ── 7. Output ───────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self.hyperparams = {
            'kernel_shape': self.kernel_shape,
            'wavelet_level': self.wavelet_level,
            'wavelet': self.wavelet,
            'threshold_ratio': self.threshold_ratio,
            'maxrank': self.maxrank,
            'max_out_iter': self.max_out_iter,
            'rmse_tol': self.rmse_tol,
            'sigma_init': self.sigma_init,
            'K': K,
            'N': N,
            'time': elapsed,
        }

        x_final = np.clip(xEst * 255.0, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('wavelet_level', self.wavelet_level),
            ('wavelet', self.wavelet),
            ('threshold_ratio', self.threshold_ratio),
            ('maxrank', self.maxrank),
            ('max_out_iter', self.max_out_iter),
            ('rmse_tol', self.rmse_tol),
            ('sigma_init', self.sigma_init),
            ('max_fun_evals', self.max_fun_evals),
            ('verbose', self.verbose),
            ('oracle_kernel', self.oracle_kernel is not None),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
