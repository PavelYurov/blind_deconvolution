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
from typing import Tuple, List, Any, Dict, Optional, Callable

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
        # ── Noise-aware extensions (all OFF by default; original behaviour) ──
        pre_pyramid: Optional[str] = None,
        pre_pyramid_params: Optional[Dict[str, Any]] = None,
        pre_kernel: Optional[str] = None,
        pre_kernel_params: Optional[Dict[str, Any]] = None,
        pre_nonblind: Optional[str] = None,
        pre_nonblind_params: Optional[Dict[str, Any]] = None,
        noise_estimation: str = 'none',
        noise_estimation_params: Optional[Dict[str, Any]] = None,
        # ── Impulse-noise preprocessing (orthogonal to denoiser hooks) ──────
        # 'auto' detects density via histogram + local outliers and runs
        # an Adaptive Median Filter ONLY on detected pixels.  Runs BEFORE
        # noise_estimation, because impulse spikes badly skew Chen/Pyatykh
        # variance estimates.  Default 'none' → unchanged.
        impulse_preprocess: str = 'none',
        impulse_params: Optional[Dict[str, Any]] = None,
        # ── Auto-config for denoiser hooks (denoisers ONLY) ─────────────────
        # 'auto' AND noise_estimation != 'none' ⇒ any hook left as
        # ``None`` is filled in based on the estimated noise (σ and, for
        # Pyatykh, the inferred ``noise_type``).  HOOKS THE USER SET
        # EXPLICITLY ARE NEVER OVERRIDDEN.  Algorithm parameters
        # (gamma, Lp, alpha_*, ...) are NEVER touched by auto_mode.
        # Naming kept consistent with the other algorithms' interface.
        auto_mode: str = 'off',
        auto_mode_overrides: Optional[Dict[str, Any]] = None,
        # ── Iteration callback (gbbid-style) ──────────────────────────────
        # Called once per outer iteration of psf_estim_lno_rgrad with a
        # dict {iteration, scale, num_scales, kernel, image, metrics}.
        iteration_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        # ── Alternative non-blind step: ringing removal ──────────────
        # 'fft_cg_sr_al'    — original (default, bit-exact behaviour)
        # 'ringing_removal' — TV-ADM + L0 + bilateral-merge pipeline
        #                     shared with gbbid / dcp / ecp / pmp / lip /
        #                     vdbke (defined locally in non_blind.py).
        # 'firls'           — FIRLS-UBC (Zhou et al., DSP 2016) shared
        #                     with FBDHSGP / BID-HBSP.  Best on clean or
        #                     mildly noisy data — produces sharper output
        #                     than fft_cg_sr_al with no extra ringing.
        nonblind_method: str = 'fft_cg_sr_al',
        lambda_tv: float = 4e-3,
        lambda_l0: float = 2e-3,
        weight_ring: float = 0.5,
        firls_params: Optional[Dict[str, Any]] = None,
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

        # ── Denoiser hooks ───────────────────────────────────────────────────────
        # All hooks default to None → the algorithm runs unchanged from
        # the original Kotera–Šroubek–Milanfar (CAIP 2013) pipeline.
        self.pre_pyramid = pre_pyramid
        self.pre_pyramid_params = dict(pre_pyramid_params or {})
        self.pre_kernel = pre_kernel
        self.pre_kernel_params = dict(pre_kernel_params or {})
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = dict(pre_nonblind_params or {})

        # ── Noise estimator (off by default) ─────────────────────────────────
        if noise_estimation not in ('none', 'chen', 'pyatykh'):
            raise ValueError(
                f"noise_estimation must be 'none','chen' or 'pyatykh', "
                f"got {noise_estimation!r}"
            )
        self.noise_estimation = noise_estimation
        self.noise_estimation_params = dict(noise_estimation_params or {})
        # Populated by ``process``:
        self.noise_sigma: Optional[float] = None
        self.noise_info: Optional[Dict[str, Any]] = None

        # ── Impulse / auto / callback ─────────────────────────────────────
        if impulse_preprocess not in ('none', 'auto'):
            raise ValueError(
                f"impulse_preprocess must be 'none' or 'auto', got "
                f"{impulse_preprocess!r}"
            )
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = dict(impulse_params or {})
        # Populated by ``process`` if impulse preprocessing ran:
        self.impulse_info: Optional[Dict[str, Any]] = None

        if auto_mode not in ('off', 'auto'):
            raise ValueError(
                f"auto_mode must be 'off' or 'auto', got {auto_mode!r}"
            )
        self.auto_mode = auto_mode
        self.auto_mode_overrides = dict(auto_mode_overrides or {})
        # Populated by ``process``: snapshot of hook config actually used.
        self.auto_mode_applied: Optional[Dict[str, Any]] = None

        self.iteration_callback = iteration_callback

        # ── Non-blind step variant ─────────────────────────────────────────
        if nonblind_method not in ('fft_cg_sr_al', 'ringing_removal', 'firls'):
            raise ValueError(
                f"nonblind_method must be 'fft_cg_sr_al', 'ringing_removal' "
                f"or 'firls', got {nonblind_method!r}"
            )
        self.nonblind_method = nonblind_method
        self.lambda_tv = float(lambda_tv)
        self.lambda_l0 = float(lambda_l0)
        self.weight_ring = float(weight_ring)
        self.firls_params = dict(firls_params or {})

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
            # Denoiser hooks (None ⇒ unchanged behaviour)
            'pre_pyramid': self.pre_pyramid,
            'pre_pyramid_params': self.pre_pyramid_params,
            'pre_kernel': self.pre_kernel,
            'pre_kernel_params': self.pre_kernel_params,
            'pre_nonblind': self.pre_nonblind,
            'pre_nonblind_params': self.pre_nonblind_params,
            # Iteration callback (gbbid-style payload, see solvers.py)
            'iteration_callback': self.iteration_callback,
            # Non-blind step variant
            'nonblind_method': self.nonblind_method,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'firls_params': self.firls_params,
        }

    # ── Auto-config helper for denoiser hooks (denoisers ONLY) ──────────
    @staticmethod
    def _auto_denoiser_config(noise_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map estimated noise statistics to a denoiser-hook configuration.

        Rules (heuristic; ALGORITHM PARAMETERS UNTOUCHED):

          * Pyatykh ``noise_type=='poisson'`` or ``'poisson_gaussian'``
            with a > 0  → use Generalised-Anscombe VST + BM3D
            (``vst_bm3d``) at Hook 1 and Hook 3.  Hook 2 = mild bilateral.
          * Otherwise (white / coloured / unknown):
                σ < 0.01           → nothing (default behaviour preserved)
                0.01 ≤ σ            → ACT (Eslahi-Aghagolzadeh adaptive
                                      curvelet thresholding) at Hook 1 &
                                      Hook 3, bilateral at Hook 2 with
                                      sigma_color = σ/2.  ACT outperforms
                                      BM3D on coloured / 1-over-f and
                                      camera-pipeline residual noise.
        """
        sigma = float(noise_info.get('sigma_norm',
                                     noise_info.get('sigma', 0.0)) or 0.0)
        ntype = str(noise_info.get('noise_type', 'gaussian')).lower()
        a = float(noise_info.get('a', 0.0) or 0.0)

        cfg: Dict[str, Any] = {}

        # Poisson / Poisson-Gaussian path — VST does the heavy lifting
        if 'poisson' in ntype and a > 0.0:
            ni = dict(noise_info)
            cfg['pre_pyramid'] = 'vst_bm3d'
            cfg['pre_pyramid_params'] = {'noise_info': ni}
            cfg['pre_kernel'] = 'bilateral'
            cfg['pre_kernel_params'] = {
                'sigma_color': max(sigma * 0.5, 0.01),
                'sigma_spatial': 1.0,
            }
            cfg['pre_nonblind'] = 'vst_bm3d'
            cfg['pre_nonblind_params'] = {'noise_info': ni}
            return cfg

        # Pure Gaussian / unknown — ACT (curvelet-domain shrinkage)
        if sigma < 0.01:
            return cfg                                  # leave defaults

        cfg['pre_pyramid'] = 'act'
        cfg['pre_pyramid_params'] = {'noise_var': float(sigma ** 2),
                                     'threshold_setting': 's'}
        cfg['pre_kernel'] = 'bilateral'
        cfg['pre_kernel_params'] = {
            'sigma_color': float(max(sigma * 0.5, 0.005)),
            'sigma_spatial': 1.0,
        }
        cfg['pre_nonblind'] = 'act'
        cfg['pre_nonblind_params'] = {'noise_var': float(sigma ** 2),
                                      'threshold_setting': 's'}
        return cfg

    # ── Per-method default params from σ / noise_info ───────────────────
    @staticmethod
    def _default_params_for(
        method: Optional[str],
        sigma: float,
        noise_info: Optional[Dict[str, Any]] = None,
        hook: str = 'pre_pyramid',
    ) -> Dict[str, Any]:
        """
        Sensible default ``**params`` for ``apply_denoiser(img, method, ...)``
        derived from the estimated noise σ (in [0, 1] scale) and, when
        available, the full Pyatykh ``noise_info`` dict.

        Used by ``auto_mode='auto'`` to fill EMPTY ``*_params`` for
        methods the user picked manually, so e.g. setting
        ``pre_pyramid='bm3d'`` without ``pre_pyramid_params`` no longer
        falls back to ``estimate_sigma`` on the blurred input (which is
        usually wrong) — it uses the proper Chen / Pyatykh σ instead.

        ``hook`` lets the resolver tune strength per location:
            * pre_pyramid  — full strength (slight over-smoothing OK)
            * pre_kernel   — MILD (multiplied by 0.5) so we don't kill
                             gradients the H-step relies on
            * pre_nonblind — full strength
        """
        if method in (None, 'none'):
            return {}
        sigma = float(max(sigma or 0.0, 1e-6))
        # Strength multiplier per hook
        scale = 0.5 if hook == 'pre_kernel' else 1.0
        s = sigma * scale

        if method == 'tv':
            # Chambolle weight ~ σ works well in [0,1] images
            return {'weight': float(max(s, 0.005))}

        if method == 'nlm':
            return {
                'sigma': float(s),
                'h': float(0.8 * s),
                'patch_size': 5,
                'patch_distance': 6,
            }

        if method == 'bilateral':
            return {
                'sigma_color': float(max(s, 0.005)),
                'sigma_spatial': 1.0,
            }

        if method == 'guided':
            return {
                'radius': 4 if hook == 'pre_kernel' else 5,
                'eps': float(max(s ** 2, 1e-4)),
            }

        if method == 'bm3d':
            # On strong noise (σ≥0.05) at the pyramid stage a 10% bump
            # helps the kernel — same rule as in _auto_denoiser_config.
            if hook == 'pre_pyramid' and sigma >= 0.05:
                return {'sigma_psd': float(1.1 * sigma)}
            return {'sigma_psd': float(s if hook == 'pre_kernel' else sigma)}

        if method == 'vst_bm3d':
            return {'noise_info': dict(noise_info)} if noise_info else {}

        if method == 'act':
            # noise_var = σ² in [0,1]² scale
            return {'noise_var': float(sigma ** 2)}

        if method == 'screenot':
            # leave structural defaults; ScreeNOT auto-detects rank
            return {}

        if method == 'adaptive_median':
            return {}

        return {}

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

        # ── 1a. Optional impulse-noise preprocessing (orthogonal hook) ──
        # Runs BEFORE everything else (incl. noise estimation), because
        # impulse spikes (salt-and-pepper, RS, hot/dead pixels) destroy
        # any further variance-based statistics and produce a star-shaped
        # kernel under blind deconvolution.  Detection is histogram-based;
        # an Adaptive Median Filter is applied ONLY to detected pixels,
        # so flat regions and edges are preserved.  Default 'none' \u21d2 NOP.
        self.impulse_info = None
        if self.impulse_preprocess == 'auto':
            from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import (
                detect_impulse_noise, adaptive_median_filter,
            )
            ip = dict(self.impulse_params)
            density_threshold = float(ip.pop('density_threshold', 0.005))
            max_window = int(ip.pop('max_window', 7))
            # `ip` is forwarded to detect_impulse_noise unchanged.
            def _impulse_one(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
                info = detect_impulse_noise(arr, **ip)
                if info['has_impulse'] and info['density'] >= density_threshold:
                    cleaned = adaptive_median_filter(
                        arr, info['impulse_mask'], max_window=max_window,
                    )
                    return cleaned, info
                return arr, info

            if y.ndim == 3:
                cleaned = np.empty_like(y)
                infos: List[Dict[str, Any]] = []
                for c in range(y.shape[2]):
                    ch_out, ch_info = _impulse_one(y[..., c])
                    cleaned[..., c] = ch_out
                    infos.append(ch_info)
                # Stash a compact summary; per-channel masks omitted to
                # keep the attribute small.
                self.impulse_info = {
                    'per_channel': [
                        {k: v for k, v in d.items() if k != 'impulse_mask'}
                        for d in infos
                    ],
                    'mean_density': float(np.mean(
                        [d['density'] for d in infos]
                    )),
                }
                y = cleaned
            else:
                y_clean, info = _impulse_one(y)
                self.impulse_info = {
                    k: v for k, v in info.items() if k != 'impulse_mask'
                }
                y = y_clean
            if self.verbose:
                print(f'[HTP_BD] impulse_preprocess=auto \u2192 '
                      f'{self.impulse_info}')

        # ── 1b. Optional noise estimation ──────────────────────────
        # Runs ONCE on the (luminance of) the input image; results are
        # stored in ``self.noise_sigma`` and ``self.noise_info`` for the
        # caller to inspect or feed into denoiser params manually.  The
        # estimate is NOT auto-applied to any algorithm parameter — the
        # user explicitly passes it (or anything else) into the hook
        # ``*_params`` dicts.
        self.noise_sigma = None
        self.noise_info = None
        if self.noise_estimation != 'none':
            y_lum = y.mean(axis=2) if y.ndim == 3 else y
            if self.noise_estimation == 'chen':
                from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level
                pch_size = int(self.noise_estimation_params.get('pch_size', 8))
                sigma = float(estimate_noise_level(y_lum, pch_size=pch_size))
                self.noise_sigma = sigma
                self.noise_info = {'method': 'chen', 'sigma': sigma}
            elif self.noise_estimation == 'pyatykh':
                from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params
                blocksize = int(self.noise_estimation_params.get('blocksize', 7))
                result = estimate_noise_params(y_lum, blocksize=blocksize)
                # ``estimate_noise_params`` returns a dict with keys
                # 'a', 'b', 'sigma' (in [0,255] scale), 'sigma_norm'
                # (in [0,1] scale) and 'noise_type'.  We expose the
                # [0,1]-scale sigma since the rest of the pipeline works
                # in normalised intensities.
                self.noise_sigma = float(result.get('sigma_norm',
                                                    result.get('sigma', 0.0) / 255.0))
                self.noise_info = {
                    'method': 'pyatykh',
                    'a': float(result.get('a', 0.0)),
                    'b': float(result.get('b', 0.0)),
                    'sigma': float(result.get('sigma', 0.0)),
                    'sigma_norm': self.noise_sigma,
                    'noise_type': result.get('noise_type', 'unknown'),
                }
            if self.verbose:
                print(f'[HTP_BD] noise_estimation={self.noise_estimation} '
                      f'→ {self.noise_info}')

        # ── 1c. Auto-config for denoiser hooks ──────────────────────────
        # Two distinct things, both gated on ``auto_mode == 'auto'``:
        #
        #   (a) FILL METHOD     — for hooks the user left as None/'none',
        #                         pick a method based on the noise stats
        #                         (rules in ``_auto_denoiser_config``).
        #   (b) FILL PARAMS     — for ANY hook whose method is set
        #                         (whether by (a) or by the user) but
        #                         whose ``*_params`` is None / empty,
        #                         derive sensible defaults from σ via
        #                         ``_default_params_for``.  This fixes
        #                         the case "I picked pre_pyramid='bm3d'
        #                         but didn't pass sigma_psd" — without
        #                         it BM3D falls back to ``estimate_sigma``
        #                         on the blurred input, which is wrong.
        #
        # Algorithm parameters (gamma, Lp, alpha_*, beta_*, ...) are
        # NEVER touched here.  Manual ``*_params`` set by the user are
        # NEVER overridden.
        self.auto_mode_applied = None
        if self.auto_mode == 'auto' and self.noise_info is not None:
            cfg = self._auto_denoiser_config(self.noise_info)
            # Manual user overrides win over rule output:
            cfg.update(self.auto_mode_overrides or {})
            sigma = float(self.noise_sigma or 0.0)
            applied: Dict[str, Any] = {}
            for hook in ('pre_pyramid', 'pre_kernel', 'pre_nonblind'):
                # (a) METHOD — only fill if user didn't choose one
                user_method = getattr(self, hook)
                if user_method in (None, 'none') and cfg.get(hook) is not None:
                    setattr(self, hook, cfg[hook])
                    applied[hook] = cfg[hook]
                # (b) PARAMS — fill if missing/empty, regardless of who
                #              chose the method (user or rule (a))
                method_now = getattr(self, hook)
                if method_now in (None, 'none'):
                    continue
                pkey = hook + '_params'
                user_params = getattr(self, pkey)
                if not user_params:                      # None or empty dict
                    # If our rule produced a method+params pair AND the
                    # method matches what's currently set, prefer those
                    # (they may include richer fields like noise_info);
                    # otherwise derive from the per-method resolver.
                    if cfg.get(hook) == method_now and cfg.get(pkey):
                        params = dict(cfg[pkey])
                    else:
                        params = self._default_params_for(
                            method_now, sigma, self.noise_info, hook=hook,
                        )
                    setattr(self, pkey, params)
                    applied[pkey] = params
            self.auto_mode_applied = applied
            if self.verbose and applied:
                print(f'[HTP_BD] auto_mode applied: {applied}')

            # ── (c) NON-BLIND VARIANT — noise-aware swap ───────────────
            # Two regimes (only when user kept the default fft_cg_sr_al
            # AND did not explicitly override 'nonblind_method'):
            #   σ < 0.01   → 'firls'            (FBDHSGP / BID-HBSP
            #                routine — sharper than fft_cg_sr_al on
            #                clean data; same prior family but UBC
            #                boundary + IRLS schedule)
            #   σ ≥ 0.01   → 'ringing_removal'  (TV-ADM + L0 + bilateral;
            #                suppresses the strong ringing fft_cg_sr_al
            #                produces near OTF zeros under noise)
            sigma = float(self.noise_sigma or 0.0)
            user_locked_nb = (
                'nonblind_method' in (self.auto_mode_overrides or {})
            )
            if (self.nonblind_method == 'fft_cg_sr_al'
                    and not user_locked_nb):
                if sigma < 0.01:
                    self.nonblind_method = 'firls'
                    if self.verbose:
                        print(f'[HTP_BD] auto_mode \u2192 nonblind_method='
                              f'firls (sigma={sigma:.3f}, clean image)')
                else:
                    self.nonblind_method = 'ringing_removal'
                    if self.verbose:
                        print(f'[HTP_BD] auto_mode \u2192 nonblind_method='
                              f'ringing_removal (sigma={sigma:.3f})')

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
