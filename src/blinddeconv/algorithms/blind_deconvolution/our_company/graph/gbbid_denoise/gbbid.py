"""
gbbid.py

Blind Image Deblurring using Graph-Based RGTV Prior (GBBID).

Reference:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Pipeline (mirrors MATLAB graph_blind_main.m):
    1. Normalise input to float64 [0, 1].
    2. Convert to grayscale, crop borders.
    3. Blind kernel estimation via bid_rgtv_c2f_cg (coarse-to-fine RGTV).
    4. Non-blind restoration via Deconvolution_FHLP (Krishnan & Fergus NIPS 2009).
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

from .solvers import (
    bid_rgtv_c2f_cg,
    Deconvolution_FHLP,
    apply_denoiser,
    deblurring_adm_aniso,
    L0Restoration,
    ringing_artifacts_removal,
)
from blinddeconv.algorithms.mod_denoise.non_blind import adaptive_lp_deconv
from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from .utils import opt_fft_size, wrap_boundary_liu


class GBBID(DeconvolutionAlgorithm):
    """
    Blind deconvolution using Graph-Based RGTV prior.

    Parameters
    ----------
    k_estimate_size : int — estimated blur kernel size (odd). Default 69.
    border          : int — boundary pixels to crop. Default 20.
    preprocess      : str — denoiser before pyramid building:
                      'tv'|'nlm'|'bilateral'|'guided'|'bm3d'|'none'.
                      Default 'tv' (original behaviour).
    preprocess_params : dict or None — kwargs for preprocess denoiser.
                      TV defaults: {'mu': 0.01, 'gamma': 0.1, 'max_it': 10}.
    pre_kernel      : str — denoiser before kernel estimation step.
                      Same options. Default 'none'.
    pre_kernel_params : dict or None — kwargs for pre_kernel denoiser.
    nonblind_method : str — non-blind deconvolution method:
                      'fhlp'     — Fast Hyper-Laplacian Prior (default, Krishnan & Fergus NIPS 2009)
                      'tv_adm'   — TV-l2 via ADM/Split Bregman with wrap_boundary_liu
                      'l0'       — L0 gradient prior with wrap_boundary_liu
                      'ringing_removal' — TV + L0 + bilateral blend (Pan et al.)
                      'adaptive_lp' — Space-variant Lp with adaptive noise model (Wang et al.)
    nonblind_params : dict or None — method-specific kwargs.
                      fhlp defaults:     {'lambda_val': 2e3,  'alpha': 0.5,  'edgetaper_iters': 4}
                      tv_adm defaults:   {'lambda_tv': 2e-3,  'alpha': 1}
                      l0 defaults:       {'lambda_grad': 2e-3, 'kappa': 2.0}
                      ringing_removal:   {'lambda_tv': 2e-3,  'lambda_l0': 2e-3, 'weight_ring': 0.5}
                      adaptive_lp:       {'alpha': 0.8, 'sigma_n': None, 'two_stage': True}
    lambda_fhlp     : float — (legacy) FHLP data-fidelity weight. Default 2e3.
    alpha_fhlp      : float — (legacy) hyper-Laplacian exponent. Default 0.5.
    edgetaper_iters : int — (legacy) number of edgetaper passes in FHLP.
                      Default 4.
    noise_estimation : str — noise estimation method before processing:
                      'chen'    — PCA eigenvalue method (Chen et al. ICCV 2015)
                      'pyatykh' — PCA + VST + kurtosis (Pyatykh et al. TIP 2013)
                      'none'    — disabled (default)
    auto_params     : bool — if True and noise_estimation != 'none',
                      automatically adapt preprocess / pre_kernel / nonblind
                      parameters based on estimated σ. Only fills parameters
                      that were not explicitly set (None). Default False.
    noise_preprocess : str — PSD-based spectral noise preprocessing:
                      'auto'    — analyze PSD, apply notch + prewhiten as needed
                      'prewhiten' — force prewhitening only
                      'notch'   — force notch filter only
                      'bandstop' — band-stop filter (requires bandstop_params)
                      'none'    — disabled (default)
    noise_preprocess_params : dict or None — kwargs for noise_preprocess.
                      auto defaults: {'pch_size': 32, 'n_smooth': 100,
                                      'peak_threshold': 5.0,
                                      'prewhiten_reg': 1e-3, 'notch_radius': 3}
                      bandstop requires: {'freq_low': ..., 'freq_high': ..., 'order': 2}
    impulse_preprocess : str — impulse noise detection and removal:
                      'auto'  — detect and remove if density > threshold (default)
                      'none'  — disabled
    impulse_params : dict or None — kwargs for impulse preprocessing.
                      Defaults: {'density_threshold': 0.005, 'max_window': 7,
                                 'outlier_window': 5, 'outlier_threshold': 0.15}
    screenot_preprocess : str — ScreeNOT SVD thresholding denoising:
                      'auto'  — apply patch-based ScreeNOT (default)
                      'none'  — disabled
    screenot_params : dict or None — kwargs for ScreeNOT.
                      Defaults: {'k': 10, 'strategy': 'i', 'mode': 'full'}
                      mode='full': treat image as matrix (no artifacts).
                      mode='patch': patch-based (needs patch_size, stride).
    act_preprocess : str — ACT curvelet denoising (Eslahi & Aghagolzadeh TIP 2016):
                      'auto'  — apply ACT (requires curvelops)
                      'none'  — disabled (default)
                      Cannot be used together with screenot_preprocess='auto'.
    act_params : dict or None — kwargs for ACT.
                      Defaults: {'noise_var': None, 'threshold_setting': 's'}
                      noise_var=None: blind MAD estimation.
                      noise_var=float: known AWGN variance σ².
                      noise_var=ndarray: FFT-PSD (DC at [0,0], scale σ²×N).
                      threshold_setting: 's' (soft), 'h' (hard), 'ksigma'.
                      If noise_var is None AND noise_estimation is enabled,
                      σ² from Chen/Pyatykh is automatically used instead
                      of blind MAD (much more accurate for correlated noise).
    pre_nonblind : str — denoiser applied to y BEFORE non-blind step:
                      'bm3d'|'nlm'|'bilateral'|'guided'|'tv'|'act'|'none'.
                      Default 'none'.  For correlated noise, 'bm3d' is
                      recommended — non-blind methods assume white noise
                      and produce color artifacts otherwise.
    pre_nonblind_params : dict or None — kwargs for pre_nonblind denoiser.
                      bm3d: {'sigma_psd': auto from noise_estimation}
                      act:  {'noise_var': auto, 'threshold_setting': 's'}
                      Other: same as preprocess params.
    auto_mode : str — high-level noise-aware orchestrator:
                      'off'    — disabled (default).
                      'robust' — soft-weighted auto configuration of the
                                 entire noise pipeline (preprocess /
                                 pre_kernel / pre_nonblind / nonblind_method /
                                 lambda_fhlp / alpha_fhlp) from estimated σ.
                                 Conservative compared to LIP: avoids
                                 aggressive in-loop smoothing that would
                                 erase informative edges needed by
                                 ``kernel_solver_L2``.
                                 Forces ``noise_estimation='pyatykh'`` if
                                 the user left it as 'none'.
    auto_mode_params : dict or None — orchestrator tuning:
        sigma_clean        — σ below which defaults are preserved verbatim.
                             Default 0.005.
        sigma_heavy        — σ at which the noise-robust config is fully
                             applied (smooth blend in between).  Default 0.05.
        force_heavy_sigma  — minimum σ to force heavy branch for signal-
                             dependent noise (poisson / poisson_gaussian).
                             Default 0.01.
        k_lambda_fhlp      — coefficient for FHLP λ ∝ k_lambda_fhlp / σ.
                             Default 50.0  (≈ scales 2e3 default at σ=0.025).
        k_alpha_fhlp       — exponent shrinkage at higher σ.
                             Default 0.6  (caps α to 0.5..0.6 when noisy).
        ensemble_members   — denoisers used in pre_nonblind ensemble at
                             heavy regime.  Default ('bm3d', 'nlm',
                             'bilateral').
        nonblind_auto_heavy — non-blind method to use in heavy regime when
                             ``nonblind_method='auto'``.
                             Default 'ringing_removal'.
        poisson_denoiser   — denoiser used for Poisson-like noise in
                             ``preprocess`` and ``pre_nonblind``:
                             'act'      — Adaptive Curvelet Thresholding
                                          (default; safe for edge preservation
                                          only in pre_nonblind — preprocess
                                          stays bilateral for GBBID).
                             'vst_bm3d' — Generalized Anscombe VST + BM3D
                                          (Mäkitalo–Foi 2013); applies GAT
                                          before BM3D so that σ≈1 everywhere,
                                          then inverts.  More accurate than
                                          ACT for moderate-to-heavy Poisson.
        act_preprocess_gaussian — bool (default False).  When True, use ACT
                             instead of bilateral/bm3d for the preprocess
                             stage on Gaussian/correlated noise.  ACT
                             operates per curvelet subband and adapts to
                             colored (1/f, 1/f²) spectra naturally, making
                             it a good fit for pink/brown noise.
                             WARNING: in GBBID this is risky — aggressive
                             curvelet thresholding can erase edges that
                             ``kernel_solver_L2`` relies on.  Enabled only
                             when the kernel is large (k_estimate_size≥51)
                             and σ is well-estimated.  Disabled by default.
        act_pre_nonblind_gaussian — bool (default False).  When True, use
                             ACT instead of BM3D for pre_nonblind on
                             Gaussian/correlated noise.  This is safe
                             (kernel already estimated) and well-suited for
                             correlated noise: ACT's per-subband threshold
                             captures both white and colored components,
                             avoiding the color artifacts that BM3D can
                             introduce when the noise is not white.
    """

    def __init__(
        self,
        k_estimate_size: int = 69,
        border: int = 20,
        preprocess: str = 'tv',
        preprocess_params: dict = None,
        pre_kernel: str = 'none',
        pre_kernel_params: dict = None,
        nonblind_method: str = 'fhlp',
        nonblind_params: dict = None,
        lambda_fhlp: float = 2e3,
        alpha_fhlp: float = 0.5,
        edgetaper_iters: int = 4,
        noise_estimation: str = 'none',
        auto_params: bool = False,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        impulse_preprocess: str = 'auto',
        impulse_params: dict = None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
    ):
        super().__init__(name='GBBID')

        self.k_estimate_size = k_estimate_size
        self.border = border
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.pre_kernel = pre_kernel
        self.pre_kernel_params = pre_kernel_params
        self.nonblind_method = nonblind_method
        self.nonblind_params = nonblind_params
        self.lambda_fhlp = lambda_fhlp
        self.alpha_fhlp = alpha_fhlp
        self.edgetaper_iters = edgetaper_iters
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        # Snapshot of user-supplied defaults — used by the robust
        # orchestrator so that soft-blending always starts from the
        # values the user passed at construction, not from values that
        # were overwritten on a previous process() call.
        self._defaults_snapshot = {
            'lambda_fhlp': float(lambda_fhlp),
            'alpha_fhlp': float(alpha_fhlp),
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'pre_kernel': pre_kernel,
            'pre_kernel_params': pre_kernel_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'nonblind_method': nonblind_method,
            'nonblind_params': nonblind_params,
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────────
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # ── 2½. Impulse noise detection & removal ────────────────
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                yg,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_window=ip.get('outlier_window', 5),
                outlier_threshold=ip.get('outlier_threshold', 0.15),
            )
            if impulse_info['has_impulse']:
                yg = adaptive_median_filter(
                    yg, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))
                # Also filter the full image for non-blind step
                if y.ndim == 3:
                    for ch in range(y.shape[2]):
                        ch_info = detect_impulse_noise(
                            y[:, :, ch],
                            density_threshold=ip.get('density_threshold', 0.005),
                            outlier_window=ip.get('outlier_window', 5),
                            outlier_threshold=ip.get('outlier_threshold', 0.15),
                        )
                        if ch_info['has_impulse']:
                            y[:, :, ch] = adaptive_median_filter(
                                y[:, :, ch], ch_info['impulse_mask'],
                                max_window=ip.get('max_window', 7))
                else:
                    y = yg.copy()

        # ── 2¾. ScreeNOT SVD thresholding denoising ─────────────────
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from blinddeconv.algorithms.mod_denoise.screenot import screenot_denoise
            sp = self.screenot_params or {}
            yg, screenot_info = screenot_denoise(
                yg,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            # Also denoise the full image for non-blind step
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = screenot_denoise(
                        y[:, :, ch],
                        k=sp.get('k', 10),
                        strategy=sp.get('strategy', 'i'),
                        mode=sp.get('mode', 'full'),
                        patch_size=sp.get('patch_size', 8),
                        stride=sp.get('stride', 3),
                    )
            else:
                y = yg.copy()

        # ── 2¾a. Noise estimation (moved BEFORE ACT) ────────────────
        #    So that σ² is available for ACT and pre_nonblind.
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
        elif self.auto_mode == 'robust':
            # Orchestrator requires σ — force PCA estimator if user left it off.
            self.noise_estimation = 'pyatykh'
            noise_info = self._estimate_noise(yg)

        # ── 2¾a½. Robust orchestrator (soft-weighted auto config) ───
        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info, image=yg)

        # ── 2¾b. ACT curvelet denoising ──────────────────────────────
        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            ap = self.act_params or {}
            # If user did not specify noise_var AND we have a noise
            # estimate, use σ² from Chen/Pyatykh instead of blind MAD.
            # This is critical for correlated noise where MAD on the
            # finest curvelet scale severely underestimates total σ.
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            yg, act_info = act_denoise(
                yg,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            # Also denoise the full image for non-blind step
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = act_denoise(
                        y[:, :, ch],
                        noise_var=act_noise_var,
                        threshold_setting=ap.get('threshold_setting', 's'),
                    )
            else:
                y = yg.copy()

        # ── 3. Crop borders ─────────────────────────────────────────────
        # MATLAB: Y_b(border+1:end-border, border+1:end-border)
        b = self.border
        if b > 0:
            yg_cropped = yg[b:-b, b:-b]
        else:
            yg_cropped = yg

        # ── 3¼. PSD-based noise preprocessing ───────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            yg, psd_info = self._apply_noise_preprocess(yg)
            if b > 0:
                yg_cropped = yg[b:-b, b:-b]
            else:
                yg_cropped = yg

        # ── Effective parameters (auto-adapted or user-specified) ────
        eff_pp = self.preprocess_params
        eff_pkp = self.pre_kernel_params
        eff_nbp = self.nonblind_params
        if self.auto_params and noise_info is not None:
            sigma = noise_info.get('sigma_norm', 0.0)
            eff_pp, eff_pkp, eff_nbp = self._compute_adaptive_params(
                sigma, eff_pp, eff_pkp, eff_nbp)

        # ── 4. Blind kernel estimation ──────────────────────────────────
        kernel, _skeleton = bid_rgtv_c2f_cg(
            yg_cropped, self.k_estimate_size,
            show_intermediate=False,
            preprocess=self.preprocess,
            preprocess_params=eff_pp,
            pre_kernel=self.pre_kernel,
            pre_kernel_params=eff_pkp,
            iteration_callback=self._callback,
        )

        # ── 4½. Pre-nonblind denoising ───────────────────────────────
        #    Non-blind methods (FHLP, TV, L0) assume white noise.
        #    Correlated noise (1/f, 1/f²) causes color artifacts.
        #    Applying a denoiser to y before non-blind suppresses this.
        if self.pre_nonblind not in (None, 'none'):
            y = self._apply_pre_nonblind(y, noise_info)

        # ── 5. Non-blind restoration ──────────────────────────────────
        nb = self.nonblind_method
        nb_p = eff_nbp or {}

        if y.ndim == 3:
            Latent = np.zeros_like(y)
            for ch in range(y.shape[2]):
                Latent[:, :, ch] = self._nonblind_single(
                    y[:, :, ch], kernel, nb, nb_p)
        else:
            Latent = self._nonblind_single(y, kernel, nb, nb_p)

        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 6. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'k_estimate_size': self.k_estimate_size,
            'border': self.border,
            'preprocess': self.preprocess,
            'preprocess_params': self.preprocess_params,
            'pre_kernel': self.pre_kernel,
            'pre_kernel_params': self.pre_kernel_params,
            'nonblind_method': self.nonblind_method,
            'nonblind_params': self.nonblind_params,
            'lambda_fhlp': self.lambda_fhlp,
            'alpha_fhlp': self.alpha_fhlp,
            'edgetaper_iters': self.edgetaper_iters,
            'noise_estimation': self.noise_estimation,
            'auto_params': self.auto_params,
            'noise_preprocess': self.noise_preprocess,
            'noise_preprocess_params': self.noise_preprocess_params,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': {k: v for k, v in (impulse_info or {}).items()
                            if k != 'impulse_mask'} if impulse_info else None,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_params': self.screenot_params,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_params': self.act_params,
            'act_info': act_info,
            'pre_nonblind': self.pre_nonblind,
            'pre_nonblind_params': self.pre_nonblind_params,
            'noise_info': noise_info,
            'psd_info': {k: v for k, v in (psd_info or {}).items()
                         if k != 'psd_2d'} if psd_info else None,
            'effective_preprocess_params': eff_pp,
            'effective_pre_kernel_params': eff_pkp,
            'effective_nonblind_params': eff_nbp,
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Non-blind dispatch ───────────────────────────────────────────────
    def _nonblind_single(self, y_ch, kernel, method, params):
        """Run non-blind deconvolution on a single channel."""
        if method == 'fhlp':
            return Deconvolution_FHLP(
                y_ch, kernel,
                lambda_val=params.get('lambda_val', self.lambda_fhlp),
                alpha=params.get('alpha', self.alpha_fhlp),
                edgetaper_iters=params.get('edgetaper_iters',
                                           self.edgetaper_iters))
        elif method == 'tv_adm':
            H, W = y_ch.shape
            target_size = opt_fft_size(
                np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
            y_pad = wrap_boundary_liu(y_ch, tuple(target_size))
            result = deblurring_adm_aniso(
                y_pad, kernel,
                lambda_tv=params.get('lambda_tv', 2e-3),
                alpha=params.get('alpha', 1))
            return result[:H, :W]
        elif method == 'l0':
            return L0Restoration(
                y_ch, kernel,
                lambda_grad=params.get('lambda_grad', 2e-3),
                kappa=params.get('kappa', 2.0))
        elif method == 'ringing_removal':
            return ringing_artifacts_removal(
                y_ch, kernel,
                lambda_tv=params.get('lambda_tv', 2e-3),
                lambda_l0=params.get('lambda_l0', 2e-3),
                weight_ring=params.get('weight_ring', 0.5))
        elif method == 'adaptive_lp':
            return adaptive_lp_deconv(
                y_ch, kernel,
                alpha=params.get('alpha', 0.8),
                sigma_n=params.get('sigma_n', None),
                two_stage=params.get('two_stage', True))
        else:
            raise ValueError(
                f"Unknown nonblind_method='{method}'. "
                f"Choose from: 'fhlp', 'tv_adm', 'l0', 'ringing_removal', "
                f"'adaptive_lp'")

    # ── Robust orchestrator ──────────────────────────────────────────
    def _orchestrate_robust(self, noise_info, image=None):
        """Soft-weighted auto configuration of the GBBID noise pipeline.

        Conservative mirror of LIP's orchestrator.  Designed to keep
        ``kernel_solver_L2`` happy: avoids aggressive smoothing inside the
        coarse-to-fine kernel-estimation loop, where over-denoised images
        wipe out the fine gradients that ``informative_edge_mask_adaptive_mine``
        relies on.

        Policy:
            • Clean regime  (σ ≤ σ_clean) — defaults are kept verbatim.
              Only resolves ``nonblind_method='auto'`` to ``'fhlp'``.
            • Heavy regime  (σ  > σ_clean) — soft blend toward σ-aware
              configuration:
                – preprocess (single-pass, level-0 of pyramid):
                    poisson-like → ACT, gaussian-mild → bilateral,
                    gaussian-heavy → bm3d.
                – pre_kernel (per-iteration smoothing of skeleton):
                    only enabled at w ≥ 0.5, mild bilateral
                    (sigma_color = 0.5·σ).
                – pre_nonblind: same routing as LIP.
                – nonblind_method='auto' → ringing_removal.
                – λ_FHLP scaled as k_lambda_fhlp / σ; α_FHLP shrunk.
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        # ── 1) Reset from snapshot — avoid sticky state between calls.
        self.lambda_fhlp = snap['lambda_fhlp']
        self.alpha_fhlp = snap['alpha_fhlp']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.pre_kernel = snap['pre_kernel']
        self.pre_kernel_params = snap['pre_kernel_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.nonblind_method = snap['nonblind_method']
        self.nonblind_params = snap['nonblind_params']

        # ── 2) Read σ.
        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.01))

        # Force heavy branch for signal-dependent noise even when σ is
        # small — pca's sigma_norm is taken at mean brightness and may
        # understate the Poisson component.
        nt = (noise_info or {}).get('noise_type', None)
        force_heavy = (nt in ('poisson', 'poisson_gaussian')
                       and sigma >= force_heavy_sigma)

        # ── 3) Clean branch — DO NOT alter denoisers or parameters.
        #      Only resolve ``nonblind_method='auto'``.
        if sigma <= sigma_clean and not force_heavy:
            if snap['nonblind_method'] == 'auto':
                self.nonblind_method = 'fhlp'
            if self.verbose if hasattr(self, 'verbose') else False:
                pass  # GBBID has no verbose flag — silent by design
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'nonblind_method': self.nonblind_method,
                'preprocess': self.preprocess,
                'pre_kernel': self.pre_kernel,
                'pre_nonblind': self.pre_nonblind,
                'lambda_fhlp': float(self.lambda_fhlp),
                'alpha_fhlp': float(self.alpha_fhlp),
            }

        # ── 4) Heavy branch — smooth weight between σ_clean and σ_heavy.
        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / (sigma_heavy - sigma_clean))
        regime = 'heavy' if w > 0.95 else 'medium'

        noise_type = nt or 'gaussian'
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        # ── 4a) λ_FHLP / α_FHLP: soft blend.
        # FHLP λ controls data-fidelity weight in non-blind step.
        # Higher σ → smaller λ (let prior win).  k_lambda_fhlp/σ gives
        # ≈ 50/0.025 = 2000 at σ=0.025 — matches the snap default.
        k_lambda_fhlp = float(amp.get('k_lambda_fhlp', 50.0))
        k_alpha_fhlp = float(amp.get('k_alpha_fhlp', 0.6))
        lam_noisy = float(np.clip(k_lambda_fhlp / max(sigma, 1e-6),
                                  100.0, 1e5))
        # α stays in [0.5, snap].  Heavier σ → α → 0.5 (more sparsity).
        alpha_noisy = max(0.5, k_alpha_fhlp)

        self.lambda_fhlp = (1.0 - w) * snap['lambda_fhlp'] + w * lam_noisy
        self.alpha_fhlp = (1.0 - w) * snap['alpha_fhlp'] + w * alpha_noisy

        # ── 4b) preprocess (single-pass, level-0 of pyramid).
        #   Poisson-like → bilateral (default) OR vst_bm3d (optional).
        #     Bilateral with sigma_color tuned to peak Poisson σ is the
        #     conservative default: it preserves edges needed by
        #     ``kernel_solver_L2`` / ``informative_edge_mask_adaptive_mine``.
        #     vst_bm3d applies GAT → BM3D → inverse, which is more
        #     accurate for moderate-to-heavy Poisson at the cost of
        #     some edge softening.  Safe to use when the kernel size is
        #     large enough that a slightly softer input still resolves
        #     the PSF support.
        #     ACT is NOT offered for preprocess in GBBID: curvelet
        #     thresholding can wipe out mid-tone edges when σ² is
        #     misestimated — see comments in LIP orchestrator.
        #   Gaussian, w<0.6 → bilateral.
        #   Gaussian, w≥0.6 → bm3d.
        poisson_denoiser = str(amp.get('poisson_denoiser', 'act')).lower()
        if poisson_denoiser not in ('act', 'vst_bm3d'):
            poisson_denoiser = 'act'

        gauss_act_preprocess = bool(amp.get('act_preprocess_gaussian', False))
        gauss_act_pre_nonblind = bool(amp.get('act_pre_nonblind_gaussian', False))


        if poisson_like:
            if poisson_denoiser == 'vst_bm3d':
                self.preprocess = 'vst_bm3d'
                self.preprocess_params = {'noise_info': noise_info}
            else:
                # Default: bilateral — gentle, preserves edges for kernel solver.
                # Effective σ in bright regions of a Poisson image is larger
                # than the global σ_norm (which is averaged).  Boost
                # sigma_color to handle the worst case without losing edges.
                self.preprocess = 'bilateral'
                self.preprocess_params = {
                    'sigma_color': float(max(2.0 * sigma, 0.02)),
                    'sigma_spatial': 3.0,
                }
        elif gauss_act_preprocess:
            # ACT for Gaussian/correlated noise: use Pyatykh scalar σ²
            # (white noise assumption — avoids blind-MAD underestimation
            # for 1/f noise where finest curvelet scale has less energy).
            # NOTE: opt-in only — ACT can erase fine edges in GBBID.
            self.preprocess = 'act'
            self.preprocess_params = {
                'noise_var': float(sigma ** 2),
                'threshold_setting': 's',
            }
        elif w < 0.6:
            self.preprocess = 'bilateral'
            self.preprocess_params = {
                'sigma_color': float(max(sigma, 0.01)),
                'sigma_spatial': 3.0,
            }
        else:
            self.preprocess = 'bm3d'
            self.preprocess_params = {'sigma_psd': float(sigma)}

        # ── 4c) pre_kernel (in-loop denoiser of skeleton before kernel
        # solver).  This is the DANGEROUS knob: smoothing inside the
        # kernel loop can erase the informative edges that
        # `kernel_solver_L2` and `informative_edge_mask_adaptive_mine`
        # rely on.
        #   Poisson-like → ALWAYS 'none'.  Per-iteration smoothing of a
        #     signal-dependent-noise image with a single-σ filter erodes
        #     dark-region detail and leaves bright-region noise alone
        #     — worst of both worlds for kernel estimation.
        #   Gaussian, w<0.5 → 'none' (defaults).
        #   Gaussian, w≥0.5 → mild bilateral (sigma_color = 0.5·σ).
        if (not poisson_like) and w >= 0.5:
            self.pre_kernel = 'bilateral'
            self.pre_kernel_params = {
                'sigma_color': float(max(0.5 * sigma, 0.005)),
                'sigma_spatial': 2.0,
            }
        # else: pre_kernel stays as snap['pre_kernel'] (default 'none')

        # ── 4d) pre_nonblind (denoise of y before non-blind solve).
        # Outside kernel loop — can be aggressive.
        #   Poisson-like → ACT (default) or vst_bm3d (if poisson_denoiser
        #     set in auto_mode_params).  Both are safe here: the kernel is
        #     already estimated and ringing_removal handles residuals.
        #     ACT: explicit noise_var=σ² avoids blind-MAD underestimation.
        #     vst_bm3d: GAT→BM3D→inverse; more accurate for heavier Poisson.
        #   Gaussian, w<0.6 → BM3D with sigma_psd=σ (or ACT via flag).
        #   Gaussian, w≥0.6 → BM3D (or ACT via flag).
        #   ACT (act_pre_nonblind_gaussian=True): per-subband curvelet
        #     threshold handles correlated (pink/brown) noise well because
        #     each curvelet subband has its own energy-based threshold.
        #     noise_var=σ² from Pyatykh prevents blind-MAD underestimation.
        if poisson_like:
            if poisson_denoiser == 'vst_bm3d':
                self.pre_nonblind = 'vst_bm3d'
                self.pre_nonblind_params = {'noise_info': noise_info}
            else:
                self.pre_nonblind = 'act'
                self.pre_nonblind_params = {
                    'noise_var': float(sigma ** 2),
                    'threshold_setting': 's',
                }
        elif gauss_act_pre_nonblind:
            # ACT for Gaussian/correlated noise in pre_nonblind.
            # Safe here (kernel already estimated).  Use Pyatykh scalar σ²
            # — more accurate than blind-MAD for 1/f noise.
            self.pre_nonblind = 'act'
            self.pre_nonblind_params = {
                'noise_var': float(sigma ** 2),
                'threshold_setting': 's',
            }
        elif w < 0.6:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma_psd': float(max(sigma, 0.01))}
        else:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma_psd': float(sigma)}

        # ── 4e) nonblind_method: heavy noise → ringing_removal when 'auto'.
        nb_auto_heavy = amp.get('nonblind_auto_heavy', 'ringing_removal')
        if snap['nonblind_method'] == 'auto':
            self.nonblind_method = nb_auto_heavy

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'poisson_denoiser': poisson_denoiser,
            'act_preprocess_gaussian': gauss_act_preprocess,
            'act_pre_nonblind_gaussian': gauss_act_pre_nonblind,
            'act_noise_var_type': 'pyatykh_scalar',
            'lambda_fhlp': float(self.lambda_fhlp),
            'alpha_fhlp': float(self.alpha_fhlp),
            'preprocess': self.preprocess,
            'pre_kernel': self.pre_kernel,
            'pre_nonblind': self.pre_nonblind,
            'nonblind_method': self.nonblind_method,
        }
        return info

    # ── PSD-based noise preprocessing ────────────────────────────────────
    def _apply_noise_preprocess(self, yg):
        """Analyze noise PSD and apply spectral filtering.

        In 'auto' mode, only notch filter is applied (for periodic noise).
        Prewhitening is NOT applied automatically — it requires a pure
        noise PSD which cannot be obtained from a single image.

        Returns
        -------
        yg_filtered : ndarray — preprocessed grayscale image [0, 1]
        psd_info : dict — PSD analysis results
        """
        from blinddeconv.algorithms.mod_denoise.noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess,
            prewhiten, notch_filter, bandstop_filter,
        )
        p = self.noise_preprocess_params or {}
        mode = self.noise_preprocess

        if mode == 'auto':
            result = noise_preprocess(
                yg,
                pch_size=p.get('pch_size', 32),
                n_smooth=p.get('n_smooth', 100),
                peak_threshold=p.get('peak_threshold', 100.0),
                notch_radius=p.get('notch_radius', 3),
            )
            return result['image'], result['psd_info']

        # For explicit modes, always run analysis first
        psd_info = analyze_noise_psd(
            yg,
            pch_size=p.get('pch_size', 32),
            n_smooth=p.get('n_smooth', 100),
            peak_threshold=p.get('peak_threshold', 100.0),
        )

        if mode == 'prewhiten':
            import warnings
            warnings.warn(
                "Prewhitening uses patch-based PSD which contains signal "
                "contamination. This WILL distort the image. Use with caution.",
                stacklevel=2)
            yg_out = prewhiten(yg, psd_info['psd_2d'],
                               reg=p.get('prewhiten_reg', 1e-3))
        elif mode == 'notch':
            peaks = psd_info['periodic_peaks']
            if peaks:
                yg_out = notch_filter(yg, peaks,
                                      notch_radius=p.get('notch_radius', 3))
            else:
                yg_out = yg
        elif mode == 'bandstop':
            yg_out = bandstop_filter(
                yg,
                freq_low=p.get('freq_low', 0.3),
                freq_high=p.get('freq_high', 0.5),
                order=p.get('order', 2),
            )
        else:
            raise ValueError(
                f"Unknown noise_preprocess='{mode}'. "
                f"Choose from: 'auto', 'prewhiten', 'notch', 'bandstop', 'none'")

        return yg_out, psd_info

    # ── Pre-nonblind denoising ────────────────────────────────────────
    def _apply_pre_nonblind(self, y, noise_info):
        """Denoise y before non-blind deconvolution.

        Non-blind methods (FHLP / TV-ADM / L0 / ringing_removal) all
        assume white Gaussian noise.  Correlated noise (1/f, 1/f²)
        violates this assumption and causes structured artifacts
        ('wrong colors', ringing amplification).

        Applying a denoiser to y here suppresses the correlated noise
        component, letting the non-blind step work correctly.

        Parameters
        ----------
        y : ndarray, H×W or H×W×C, float64 [0,1]
        noise_info : dict or None — from _estimate_noise()

        Returns
        -------
        y_denoised : ndarray, same shape as y
        """
        method = self.pre_nonblind
        params = dict(self.pre_nonblind_params or {})

        # Auto-fill sigma from noise estimation if available
        sigma = None
        if noise_info is not None:
            sigma = noise_info.get('sigma_norm', None)

        if method == 'act':
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            nv = params.get('noise_var', None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = params.get('threshold_setting', 's')
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = act_denoise(
                        y[:, :, ch], noise_var=nv,
                        threshold_setting=ts)
            else:
                y, _ = act_denoise(y, noise_var=nv,
                                   threshold_setting=ts)
            return y

        if method == 'vst_bm3d':
            from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise
            ni = params.get('noise_info', None)
            a = params.get('a', None)
            b = params.get('b', None)
            sig = params.get('sigma', sigma)
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = vst_bm3d_denoise(
                        y[:, :, ch], noise_info=ni,
                        a=a, b=b, sigma=sig)
            else:
                y, _ = vst_bm3d_denoise(y, noise_info=ni,
                                        a=a, b=b, sigma=sig)
            return y

        # For standard denoisers (bm3d, nlm, bilateral, guided, tv),
        # auto-fill sigma_psd / h / sigma_color from noise estimation.
        if method == 'bm3d' and 'sigma_psd' not in params and sigma is not None:
            params['sigma_psd'] = sigma
        elif method == 'nlm' and 'h' not in params and sigma is not None:
            params['h'] = 0.8 * sigma
        elif method == 'bilateral' and 'sigma_color' not in params and sigma is not None:
            params['sigma_color'] = sigma
        elif method == 'guided' and 'eps' not in params and sigma is not None:
            params['eps'] = sigma ** 2 * 4

        if y.ndim == 3:
            for ch in range(y.shape[2]):
                y[:, :, ch] = apply_denoiser(
                    y[:, :, ch], method, **params)
        else:
            y = apply_denoiser(y, method, **params)
        return y

    # ── Noise estimation helpers ─────────────────────────────────────────
    def _estimate_noise(self, yg):
        """Estimate noise level from grayscale image (float64 [0, 1])."""
        if self.noise_estimation == 'chen':
            from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pyatykh':
            from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pyatykh'
            return result
        return None

    def _compute_adaptive_params(self, sigma, pp, pkp, nbp):
        """Adapt processing parameters based on estimated noise σ (in [0,1] scale).

        Only fills parameters that are None (not explicitly set by user).
        Formulas are initial heuristics — tune per dataset.
        """
        if sigma < 1e-6:
            return pp, pkp, nbp

        if pp is None:
            if self.preprocess == 'nlm':
                pp = {'h': sigma}
            elif self.preprocess == 'tv':
                pp = {'mu': sigma * 0.5, 'gamma': 0.1, 'max_it': 10}
            elif self.preprocess == 'bilateral':
                pp = {'sigma_spatial': 3, 'sigma_range': sigma * 2}
            elif self.preprocess == 'guided':
                pp = {'radius': 2, 'eps': sigma ** 2 * 4}

        if pkp is None:
            if self.pre_kernel == 'guided':
                pkp = {'radius': 2, 'eps': sigma ** 2 * 4}
            elif self.pre_kernel == 'nlm':
                pkp = {'h': sigma}
            elif self.pre_kernel == 'bilateral':
                pkp = {'sigma_spatial': 2, 'sigma_range': sigma * 2}

        if nbp is None:
            if self.nonblind_method == 'ringing_removal':
                nbp = {
                    'lambda_tv': sigma * 0.5,
                    'lambda_l0': sigma * 0.025,
                    'weight_ring': min(1.0, max(0.3, sigma * 50)),
                }
            elif self.nonblind_method == 'tv_adm':
                nbp = {'lambda_tv': sigma * 0.5}
            elif self.nonblind_method == 'l0':
                nbp = {'lambda_grad': sigma * 0.1}
            elif self.nonblind_method == 'adaptive_lp':
                nbp = {'alpha': 0.8, 'sigma_n': sigma, 'two_stage': True}

        return pp, pkp, nbp

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('k_estimate_size', self.k_estimate_size),
            ('border', self.border),
            ('preprocess', self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('pre_kernel', self.pre_kernel),
            ('pre_kernel_params', self.pre_kernel_params),
            ('nonblind_method', self.nonblind_method),
            ('nonblind_params', self.nonblind_params),
            ('lambda_fhlp', self.lambda_fhlp),
            ('alpha_fhlp', self.alpha_fhlp),
            ('edgetaper_iters', self.edgetaper_iters),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params', self.screenot_params),
            ('act_preprocess', self.act_preprocess),
            ('act_params', self.act_params),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Keep the orchestrator's default-snapshot in sync with
                # parameters the user updates after construction —
                # otherwise robust mode would keep blending toward stale
                # values from the original __init__.
                if key in self._defaults_snapshot:
                    self._defaults_snapshot[key] = (
                        float(value) if key in ('lambda_fhlp', 'alpha_fhlp')
                        else value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
