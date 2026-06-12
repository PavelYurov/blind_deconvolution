"""
htp.py

Blind Image Deblurring Using Heavy-Tailed Priors (HTP).

Reference:
    J. Kotera, F. Sroubek, P. Milanfar:
    "Blind Deconvolution Using Alternating Maximum a Posteriori
     Estimation with Heavy-tailed Priors", CAIP 2013.

Pipeline (mirrors MATLAB demo.m / MCrestoration.m):
    1. Normalise input to float64 [0, 1].
    2. Build coarse-to-fine pyramid of the central ROI
       (green channel for RGB, full image for grayscale).
    3. Multi-scale alternating MAP for (u, h) with heavy-tailed Lp prior
       on image gradients (p < 1) and L1 prior on the PSF, solved via
       half-quadratic splitting + Bregman iterations in the FFT domain
       (psf_estim_lno_rgrad at each scale).
    4. Final non-blind deconvolution on the full image (fft_cg_sr_al)
       with stronger data-term and TV-like prior (Lp_nonblind = 1).
    5. Return restored image (int16, [0, 255]) and kernel.
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

from .solvers import mc_restoration


class HTP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution with heavy-tailed priors (Kotera et al., CAIP 2013).

    Parameters
    ----------
    kernel_size : int
        Spatial support of the unknown PSF (square, equals MATLAB hsize).
        Default 31 (matching demo_levin.m).
    Lp : float
        Lp-norm exponent for the gradient prior on the latent image
        during PSF estimation, 0 < p <= 1.  Default 0.3 (heavy-tailed).
    gamma : float
        Data-term weight during PSF estimation.  Should be tuned to the
        noise level (10 dB → 1e1, 20 dB → 1e2, ...).  Default 1e2.
    alpha_u : float
        Image-prior weight relative scale (multiplied by gamma).
        Default 1e-2.
    beta_u : float
        Coupling (split-Bregman) weight relative scale (× gamma).
        Default 1e0.
    alpha_h : float
        PSF L1-prior weight relative scale (× gamma).  Default 1e1.
    beta_h : float
        PSF coupling weight relative scale (× gamma).  Default 1e4.
    centering_threshold : float
        Threshold used in PSF centering between iterations.  Default
        20/255.  <= 0 disables centering.
    gamma_nonblind : float
        Data-term weight for the final non-blind deconvolution
        (relative to gamma).  Default 2e1.
    beta_u_nonblind : float
        Coupling weight for the final non-blind step (× gamma_nonblind).
        Default 1e-2.
    Lp_nonblind : float
        Lp exponent for the final non-blind step.  Default 1.0 (TV-like).
    maxiter : int
        Outer alternating iterations per pyramid level.  Default 10.
    maxiter_u : int
        Inner u-step iterations.  Default 10.
    maxiter_h : int
        Inner h-step iterations.  Default 10.
    ccreltol : float
        Relative-change stop criterion for inner loops.  Default 1e-3.
    MSlevels : int
        Number of multiscale levels (>= 1).  Default 4.
    maxROIsize : tuple of int
        Central ROI used for kernel estimation.  Default (1024, 1024).
    verbose : int
        0 = silent, 1 = progress messages.  Default 0.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        Lp: float = 0.3,
        gamma: float = 1e2,
        alpha_u: float = 1e-2,
        beta_u: float = 1e0,
        alpha_h: float = 1e1,
        beta_h: float = 1e4,
        centering_threshold: float = 20.0 / 255.0,
        gamma_nonblind: float = 2e1,
        beta_u_nonblind: float = 1e-2,
        Lp_nonblind: float = 1.0,
        maxiter: int = 10,
        maxiter_u: int = 10,
        maxiter_h: int = 10,
        ccreltol: float = 1e-3,
        MSlevels: int = 4,
        maxROIsize: Tuple[int, int] = (1024, 1024),
        verbose: int = 0,
        kernel_flip: str = 'none',
        auto_recenter: bool = False,
        recenter_mode: str = 'centroid',
        kernel_thresh: float = 0.0,
        iterative_recenter: bool = True,
    ):
        super().__init__(name='HTP-BD')

        self.kernel_size = int(kernel_size)
        self.Lp = float(Lp)
        self.gamma = float(gamma)
        self.alpha_u = float(alpha_u)
        self.beta_u = float(beta_u)
        self.alpha_h = float(alpha_h)
        self.beta_h = float(beta_h)
        self.centering_threshold = float(centering_threshold)
        self.gamma_nonblind = float(gamma_nonblind)
        self.beta_u_nonblind = float(beta_u_nonblind)
        self.Lp_nonblind = float(Lp_nonblind)
        self.maxiter = int(maxiter)
        self.maxiter_u = int(maxiter_u)
        self.maxiter_h = int(maxiter_h)
        self.ccreltol = float(ccreltol)
        self.MSlevels = int(MSlevels)
        self.maxROIsize = tuple(maxROIsize)
        self.verbose = int(verbose)
        if kernel_flip not in ('none', 'lr', 'ud', 'rot180'):
            raise ValueError(
                f"kernel_flip must be one of 'none','lr','ud','rot180', got {kernel_flip!r}"
            )
        self.kernel_flip = kernel_flip
        if recenter_mode not in ('centroid', 'peak', 'masscentroid'):
            raise ValueError(
                f"recenter_mode must be 'centroid','peak' or 'masscentroid', got {recenter_mode!r}"
            )
        self.auto_recenter = bool(auto_recenter)
        self.recenter_mode = recenter_mode
        self.kernel_thresh = float(kernel_thresh)
        self.iterative_recenter = bool(iterative_recenter)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Build the PAR dict expected by solvers (mirrors parameters.m) ────
    def _build_par(self) -> Dict[str, Any]:
        gamma = self.gamma
        gamma_nb = self.gamma_nonblind * gamma
        return {
            'verbose': self.verbose,
            'gamma': gamma,
            'Lp': self.Lp,
            # PSF prior (relative scales × gamma, exactly as in parameters.m)
            'beta_h': self.beta_h * gamma,
            'alpha_h': self.alpha_h * gamma,
            'centering_threshold': self.centering_threshold,
            # Image prior (relative scales × gamma)
            'beta_u': self.beta_u * gamma,
            'alpha_u': self.alpha_u * gamma,
            # Non-blind final step (× gamma_nonblind)
            'gamma_nonblind': gamma_nb,
            'beta_u_nonblind': self.beta_u_nonblind * gamma_nb,
            'Lp_nonblind': self.Lp_nonblind,
            # Iteration limits
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'maxiter': self.maxiter,
            'ccreltol': self.ccreltol,
            # Iterative-improvement knobs (HTP-internal, not in original MATLAB)
            'kernel_thresh': self.kernel_thresh,
            'iterative_recenter': self.iterative_recenter,
        }

    # ── Auto-recentering helper ──────────────────────────────────────────
    def _recenter_kernel_and_image(
        self, H: np.ndarray, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shift the kernel to put its centre at the window centre, and
        shift the image by the OPPOSITE amount so g = h * u is preserved.

        Centring uses the **bounding box** of the thresholded kernel
        (mirrors FBDHSGP's ``shift_kernel_img_space``):

            shift = round((gap_far - gap_near + bonus) / 2)

        BBox-based centring is much more robust than a naive centroid
        for diffuse / heavy-tailed kernels (defocus rings, dendritic
        traces, V-shapes), because the centroid is biased by the
        negative-tail noise floor and by long thin tails.

        The image counter-shift is realised by replicate-padding on the
        far side and cropping on the near side — NOT by ``np.roll`` —
        so no wrap-around.

        Modes:
          * 'centroid'      – bbox of H clipped at >=20% of max  [default]
          * 'masscentroid'  – mass-centroid of |H| (legacy)
          * 'peak'          – argmax(H)
        """
        kh, kw = H.shape
        cy_int = kh // 2
        cx_int = kw // 2

        # ---- determine target offset (sy, sx): how to move kernel ------
        if self.recenter_mode == 'peak':
            iy, ix = np.unravel_index(int(np.argmax(H)), H.shape)
            sy, sx = int(cy_int - iy), int(cx_int - ix)

        elif self.recenter_mode == 'masscentroid':
            Hp = np.maximum(H, 0.0)
            s = Hp.sum()
            if s <= 0:
                return H, U
            ys = np.arange(kh)[:, None]
            xs = np.arange(kw)[None, :]
            iy = (Hp * ys).sum() / s
            ix = (Hp * xs).sum() / s
            sy = int(round((kh - 1) / 2.0 - iy))
            sx = int(round((kw - 1) / 2.0 - ix))

        else:  # 'centroid' — bbox-based (FBDHSGP style)
            Hp = np.maximum(H, 0.0)
            m = Hp.max()
            if m <= 0:
                return H, U
            # Threshold: keep pixels above max(0.03*max, small floor),
            # exactly the same recipe FBDHSGP uses.
            tao = 0.03
            thr = min(m * tao, 0.002)
            mask = Hp >= thr
            if not mask.any():
                return H, U
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y_top, y_bot = int(rows[0]), int(rows[-1])
            x_left, x_right = int(cols[0]), int(cols[-1])

            gap_left = x_left
            gap_right = (kw - 1) - x_right
            gap_top = y_top
            gap_bot = (kh - 1) - y_bot

            # Tie-breaker bonus toward the heavier edge column/row
            s_l = Hp[:, x_left].sum()
            s_r = Hp[:, x_right].sum()
            bonus_x = 0.01 if (s_l >= s_r) else -0.01
            s_t = Hp[y_top, :].sum()
            s_b = Hp[y_bot, :].sum()
            bonus_y = 0.01 if (s_t >= s_b) else -0.01

            sx = int(round((gap_right - gap_left + bonus_x) / 2.0))
            sy = int(round((gap_bot - gap_top + bonus_y) / 2.0))

        if sy == 0 and sx == 0:
            return H, U

        # ---- shift kernel with zero padding (no wrap) ------------------
        H_new = np.zeros_like(H)
        src_r0 = max(0, -sy); src_r1 = min(kh, kh - sy)
        src_c0 = max(0, -sx); src_c1 = min(kw, kw - sx)
        dst_r0 = max(0, sy);  dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, sx);  dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            H_new[dst_r0:dst_r1, dst_c0:dst_c1] = H[src_r0:src_r1, src_c0:src_c1]
        s_h = H_new.sum()
        if s_h > 0:
            H_new = H_new / s_h

        # ---- counter-shift image: pad-edge on far side, crop on near ---
        # If kernel moved by (sy, sx), image must move by (-sy, -sx).
        # Use replicate-edge boundary so we don't introduce wrap-around
        # or black borders.
        Mh, Mw = U.shape
        py0 = max(0, sy);  py1 = max(0, -sy)
        px0 = max(0, sx);  px1 = max(0, -sx)
        U_padded = np.pad(U, ((py0, py1), (px0, px1)), mode='edge')
        U_new = U_padded[py1:py1 + Mh, px1:px1 + Mw].copy()

        return H_new, U_new


    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise input to float64 [0, 1] ────────────────────────
        y = np.asarray(image, dtype=np.float64)
        if y.max() > 1.0:
            y = y / 255.0

        # ── 2. Build parameter dict and run the multiscale pipeline ─────
        PAR = self._build_par()
        hsize = (self.kernel_size, self.kernel_size)

        U, H, _report = mc_restoration(
            y,
            hsize=hsize,
            PAR=PAR,
            MSlevels=self.MSlevels,
            maxROIsize=self.maxROIsize,
        )
        U = np.clip(U, 0.0, 1.0)

        # ── 2b. Auto-recenter the kernel (translation-ambiguity fix) ────
        # Blind deconvolution is translation-invariant: (h(x), u(x)) and
        # (h(x-d), u(x+d)) explain the same observation g.  In practice
        # the recovered kernel often drifts off-center (typically up).
        # We compensate by computing the kernel's "centre" (centroid of
        # the thresholded mass, robust to noise floor), shifting the
        # kernel to put it at the window centre, and shifting the image
        # by the OPPOSITE amount so that g = h * u remains invariant.
        if self.auto_recenter:
            H, U = self._recenter_kernel_and_image(H, U)

        # ── 3. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'Lp': self.Lp,
            'gamma': self.gamma,
            'alpha_u': self.alpha_u,
            'beta_u': self.beta_u,
            'alpha_h': self.alpha_h,
            'beta_h': self.beta_h,
            'gamma_nonblind': self.gamma_nonblind,
            'beta_u_nonblind': self.beta_u_nonblind,
            'Lp_nonblind': self.Lp_nonblind,
            'MSlevels': self.MSlevels,
            'maxROIsize': self.maxROIsize,
            'maxiter': self.maxiter,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'time': time.time() - start_time,
        }

        x_final = U * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        if self.kernel_flip == 'lr':
            H_out = H[:, ::-1].copy()
        elif self.kernel_flip == 'ud':
            H_out = H[::-1, :].copy()
        elif self.kernel_flip == 'rot180':
            H_out = H[::-1, ::-1].copy()
        else:
            H_out = H
        return x_final, H_out

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('Lp', self.Lp),
            ('gamma', self.gamma),
            ('alpha_u', self.alpha_u),
            ('beta_u', self.beta_u),
            ('alpha_h', self.alpha_h),
            ('beta_h', self.beta_h),
            ('centering_threshold', self.centering_threshold),
            ('gamma_nonblind', self.gamma_nonblind),
            ('beta_u_nonblind', self.beta_u_nonblind),
            ('Lp_nonblind', self.Lp_nonblind),
            ('maxiter', self.maxiter),
            ('maxiter_u', self.maxiter_u),
            ('maxiter_h', self.maxiter_h),
            ('ccreltol', self.ccreltol),
            ('MSlevels', self.MSlevels),
            ('maxROIsize', self.maxROIsize),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'maxROIsize':
                    self.maxROIsize = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
