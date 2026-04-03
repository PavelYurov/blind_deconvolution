"""
dcp_selfexsr.py

Blind Image Deblurring Using Dark Channel Prior
with SelfExSR-Enhanced Kernel Estimation.

This algorithm extends the DCP blind deconvolution method
(Pan et al., CVPR 2016) by integrating SelfExSR super-resolution
(Huang et al., CVPR 2015) to improve kernel estimation accuracy
on intermediate pyramid levels.

Motivation:
    The standard DCP pipeline estimates blur kernels in a coarse-to-
    fine manner.  On coarse levels, the heavily downsampled image has
    weak gradients and lost high-frequency detail, leading to poor
    initial kernel estimates that cascade errors to finer levels.
    SelfExSR recovers high-frequency content from the image's own
    internal patch recurrence statistics — exactly the information
    DCP lacks at coarse scales.

Integration — three-phase "mid-loop SR injection":
    Phase 1 (WARMUP): Run standard DCP on the N coarsest levels to
        obtain a rough but directionally correct kernel estimate.
    Bridge: Upsample the warmup kernel to full resolution, Wiener-
        deconvolve the original blurred image, then run SelfExSR on
        the partially-deblurred result.  This produces an SR reference
        with genuine scene structure (not blur artefacts).
    Phase 2 (SR-ENHANCED): On intermediate levels, blend the DCP
        latent estimate with the SR reference before gradient
        extraction:  S_enh = α·S_dcp + (1-α)·S_sr
    Phase 3 (FINE): Standard DCP on the finest level(s).

Key design choice — why NOT SR on the raw blurred image:
    SelfExSR assumes a clean low-resolution input.  On a blurred
    image, self-similar patches are blurred too, so SR hallucinates
    FALSE high-frequency detail.  Blending these false edges with
    the DCP latent corrupts gradient extraction and makes the kernel
    WORSE — especially on complex/large blur kernels.  By first
    partially deblurring with a rough Wiener filter, SelfExSR
    operates on meaningful scene structure.

Parameters unique to this variant (vs. original DCP):
    SRF              — SelfExSR upscaling factor (default 2)
    sr_num_iter      — PatchMatch iterations in SelfExSR (default 5)
    sr_n_iter_bp     — back-projection iterations in SelfExSR (default 10)
    n_warmup_levels  — coarsest DCP levels before SR injection (default 2)
    n_sr_levels      — intermediate levels with SR blending (default 2)
    sr_alpha_min     — DCP blend weight at coarsest SR level (default 0.5)
    sr_alpha_max     — DCP blend weight at finest SR level (default 0.85)
    wiener_snr       — Wiener filter noise parameter (default 0.01)
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

from .solvers import blind_deconv_sr, ringing_artifacts_removal


class DCP_SelfExSR_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using Dark Channel Prior with SelfExSR-enhanced
    kernel estimation on intermediate pyramid levels.

    Parameters
    ----------
    kernel_size      : int — spatial support of the unknown PSF (square, odd).
    lambda_dark      : float — weight for L0 intensity (dark-channel) prior.
    lambda_grad      : float — weight for L0 gradient prior.
    xk_iter          : int — blind iterations per pyramid level.
    gamma_correct    : float — gamma correction exponent. 1.0 = no correction.
    k_thresh         : float — final kernel threshold.
    lambda_tv        : float — weight for TV non-blind deconvolution.
    lambda_l0        : float — weight for L0 non-blind deconvolution.
    weight_ring      : float — ringing suppression weight.
    SRF              : int — SelfExSR super-resolution factor (default 2).
    sr_num_iter      : int — PatchMatch iterations for SelfExSR (default 5).
    sr_n_iter_bp     : int — back-projection iterations for SelfExSR (default 10).
    n_warmup_levels  : int — coarsest pyramid levels with standard DCP
                       before SR injection (default 2).
    n_sr_levels      : int — intermediate pyramid levels enhanced by
                       SR blending (default 2).
    sr_alpha_min     : float — DCP blend weight at coarsest SR level (default 0.5).
    sr_alpha_max     : float — DCP blend weight at finest SR level (default 0.85).
    wiener_snr       : float — Wiener filter noise-to-signal ratio for the
                       initial deblur before SR (default 0.01).
    """

    def __init__(
        self,
        kernel_size: int = 25,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.003,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        SRF: int = 2,
        sr_num_iter: int = 5,
        sr_n_iter_bp: int = 10,
        n_warmup_levels: int = 2,
        n_sr_levels: int = 2,
        sr_alpha_min: float = 0.5,
        sr_alpha_max: float = 0.85,
        wiener_snr: float = 0.01,
    ):
        super().__init__(name='DCP-SelfExSR-BD')

        # ── DCP parameters (same as original) ────────────────────────
        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

        # ── SelfExSR integration parameters ──────────────────────────
        self.SRF = SRF
        self.sr_num_iter = sr_num_iter
        self.sr_n_iter_bp = sr_n_iter_bp
        self.n_warmup_levels = n_warmup_levels
        self.n_sr_levels = n_sr_levels
        self.sr_alpha_min = sr_alpha_min
        self.sr_alpha_max = sr_alpha_max
        self.wiener_snr = wiener_snr

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 3. SR-enhanced blind kernel estimation ──────────────────
        dcp_opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'n_warmup_levels': self.n_warmup_levels,
            'n_sr_levels': self.n_sr_levels,
            'sr_alpha_min': self.sr_alpha_min,
            'sr_alpha_max': self.sr_alpha_max,
            'wiener_snr': self.wiener_snr,
        }

        sr_opts = {
            'SRF': self.SRF,
            'numIter': self.sr_num_iter,
            'nIterBP': self.sr_n_iter_bp,
        }

        kernel, interim_latent = blind_deconv_sr(
            yg, self.lambda_dark, self.lambda_grad, dcp_opts,
            sr_opts=sr_opts,
        )

        # ── 4. Non-blind restoration (same as original DCP) ────────
        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ──────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'SRF': self.SRF,
            'sr_num_iter': self.sr_num_iter,
            'sr_n_iter_bp': self.sr_n_iter_bp,
            'n_warmup_levels': self.n_warmup_levels,
            'n_sr_levels': self.n_sr_levels,
            'sr_alpha_min': self.sr_alpha_min,
            'sr_alpha_max': self.sr_alpha_max,
            'wiener_snr': self.wiener_snr,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_dark', self.lambda_dark),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('SRF', self.SRF),
            ('sr_num_iter', self.sr_num_iter),
            ('sr_n_iter_bp', self.sr_n_iter_bp),
            ('n_warmup_levels', self.n_warmup_levels),
            ('n_sr_levels', self.n_sr_levels),
            ('sr_alpha_min', self.sr_alpha_min),
            ('sr_alpha_max', self.sr_alpha_max),
            ('wiener_snr', self.wiener_snr),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
