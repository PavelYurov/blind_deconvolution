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
from .non_blind import adaptive_lp_deconv
from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
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

        # ── 3½. Noise estimation ────────────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)

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
        )

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
            'noise_info': noise_info,
            'psd_info': {k: v for k, v in (psd_info or {}).items()
                         if k != 'psd_2d'} if psd_info else None,
            'effective_preprocess_params': eff_pp,
            'effective_pre_kernel_params': eff_pkp,
            'effective_nonblind_params': eff_nbp,
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
        from .noise_psd_analysis import (
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

    # ── Noise estimation helpers ─────────────────────────────────────────
    def _estimate_noise(self, yg):
        """Estimate noise level from grayscale image (float64 [0, 1])."""
        if self.noise_estimation == 'chen':
            from .chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pyatykh':
            from .pyatykh_noise_reconstruction import estimate_noise_params
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
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
