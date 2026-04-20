"""
lip.py

Blind Image Deconvolution via Lower-Bounded Logarithmic Image Priors (LIP).

Reference:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", International Conference on
    Energy Minimization Methods in Computer Vision and Pattern Recognition
    (EMMCVPR), 2015.

Implements two methods from the paper:
    MM  — Majorization-Minimization (Table 2): gradient descent on the
          EM-majorised weighted-TV subproblem.
    PD  — Primal-Dual / Condat-Vũ (Table 1): solves the same
          weighted-TV subproblem with Condat-Vũ primal-dual splitting
          (gradient of data fidelity, no FFT).
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

from .solvers import coarse_to_fine, ringing_artifacts_removal
from .utils import (
    gamma_correction,
    make_size_odd,
    edgetaper,
    pad_image,
    crop_image,
    wiener_filter,
    tikhonov_filter,
)
from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter


class LIP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Logarithmic Image Prior (MM algorithm).

    Pipeline (mirrors MATLAB ``deblur.m`` → ``coarseToFine.m`` → ``blind.m``):
        1. Normalise input to float64 [0, 1].
        2. Trim image to odd dimensions.
        3. (Optional) gamma correction.
        4. Noise pipeline (impulse → estimation → auto-params → ScreeNOT/ACT
           → preprocess → PSD filtering).
        5. Coarse-to-fine PSF estimation (MM or PD method) with optional
           blind_denoise before each kernel step.
        6. Pre-nonblind denoising (optional).
        7. Non-blind restoration: Tikhonov / Wiener / ringing_removal /
           adaptive_lp.
        8. Return restored image (int16, [0, 255]) and kernel.

    Parameters
    ----------
    kernel_shape : (MK, NK) — spatial support of the unknown PSF.
    lambda_val   : data-fidelity weight (β in the paper).
                   Default 30000 (from main_levin.m benchmark).
    tau          : lower-bound parameter of the log prior (default 1e-3).
    outer_iters  : EM outer iterations per pyramid level (default 140).
    inner_iters  : gradient-descent inner iterations per outer (default 5).
    k_step       : kernel step-size schedule (list/array).
    u_step       : image step-size schedule (list/array).
    lambda_mult  : λ multiplier between pyramid levels (default 2.1).
    scale_mult   : kernel-size divider between pyramid levels (default √2).
    gamma_correction : whether to apply gamma correction (default False).
    gamma        : gamma exponent (used when gamma_correction=True).
    method       : 'mm' (gradient-descent, Table 2) or 'pd' (Condat-Vũ, Table 1).
    kernel_threshold : fraction of max(k) below which kernel values are zeroed (default 0.05).
    final_deconv : 'tikhonov', 'wiener', 'ringing_removal', or 'adaptive_lp'.
    final_alpha  : regularisation strength for the non-blind step.
    verbose      : print progress during coarse-to-fine.

    Noise Pipeline Parameters (all disabled by default)
    ---------------------------------------------------
    impulse_preprocess : str
        'auto' — detect & remove impulse (salt-and-pepper) noise before
        blind deconvolution.  'none' — skip.
    impulse_params : dict or None
        Keys: 'density_threshold' (float, default 0.0005),
              'outlier_threshold' (float, default 0.08),
              'max_window' (int, default 7).
    noise_estimation : str
        'chen' — Chen et al. (ICCV 2015) wavelet-based σ estimation.
        'pca'  — Pyatykh et al. (TIP 2013) PCA + VST + kurtosis.
        'none' — skip.
    screenot_preprocess : str
        'auto' — apply ScreeNOT SVD denoising before blind step.
        'none' — skip.  Mutually exclusive with act_preprocess.
    screenot_params : dict or None
        Keys: 'k', 'strategy', 'mode', 'patch_size', 'stride'.
    act_preprocess : str
        'auto' — apply Adaptive Curvelet Thresholding before blind step.
        'none' — skip.  Mutually exclusive with screenot_preprocess.
    act_params : dict or None
        Keys: 'noise_var', 'threshold_setting' ('s'/'h').
    preprocess : str
        Spatial denoiser name: 'tv', 'nlm', 'bilateral', 'guided',
        'bm3d', 'act', or 'none'.
    preprocess_params : dict or None
        Denoiser-specific parameters.
    noise_preprocess : str
        PSD-based noise filter: 'auto', 'notch', 'bandstop', or 'none'.
    noise_preprocess_params : dict or None
        Keys: 'pch_size', 'n_smooth', 'peak_threshold', 'notch_radius',
              'freq_low', 'freq_high', 'order'.
    blind_denoise : str
        Denoiser applied to u before each kernel update inside the blind
        loop: 'tv', 'nlm', 'bilateral', 'guided', 'bm3d', or 'none'.
    blind_denoise_params : dict or None
        Denoiser-specific parameters.
    pre_nonblind : str
        Denoiser applied to the blurry image before the non-blind step.
        Same options as preprocess.
    pre_nonblind_params : dict or None
        Denoiser-specific parameters.
    auto_params : dict or None
        If not None and noise_estimation succeeds, auto-tune
        lambda_val, tau, and final_alpha from σ.
        Keys: 'k_lambda' (float, default 5000.0),
              'k_tau'    (float, default 10.0),
              'k_alpha'  (float, default 0.1).
    nb_params : dict or None
        Parameters for ringing_removal non-blind method:
        'lambda_tv' (float), 'lambda_l0' (float), 'weight_ring' (float).
        Parameters for adaptive_lp non-blind method:
        'alpha' (float), 'two_stage' (bool).
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_val: float = 30000.0,
        tau: float = 1e-3,
        outer_iters: int = 140,
        inner_iters: int = 5,
        k_step: Any = None,
        u_step: Any = None,
        lambda_mult: float = 2.1,
        scale_mult: float = 1.4142135623730951,  # sqrt(2)
        gamma_correction: bool = False,
        gamma: float = 1.0,
        method: str = 'mm',
        kernel_threshold: float = 0.05,
        final_deconv: str = 'tikhonov',
        final_alpha: float = 0.001,
        verbose: bool = False,
        # ── Noise pipeline (all disabled by default) ────────────────────
        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        preprocess: str = 'none',
        preprocess_params: dict = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        blind_denoise: str = 'none',
        blind_denoise_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        auto_params: dict = None,
        nb_params: dict = None,
    ):
        super().__init__(name='LIP-BD')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_val = lambda_val
        self.tau = tau
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters

        # Step-size schedules — defaults from deblur.m
        if k_step is None:
            self.k_step = np.array([1e-2, 5e-3, 1e-3, 5e-4])
        else:
            self.k_step = np.atleast_1d(np.asarray(k_step, dtype=np.float64))
        if u_step is None:
            self.u_step = np.array([1e-2, 5e-3, 1e-3, 1e-3])
        else:
            self.u_step = np.atleast_1d(np.asarray(u_step, dtype=np.float64))

        self.lambda_mult = lambda_mult
        self.scale_mult = scale_mult
        self.gamma_corr = gamma_correction
        self.gamma = gamma
        self.method = method.lower()
        self.kernel_threshold = kernel_threshold
        self.final_deconv = final_deconv.lower()
        self.final_alpha = final_alpha
        self.verbose = verbose

        # Noise pipeline
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.blind_denoise = blind_denoise
        self.blind_denoise_params = blind_denoise_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.auto_params = auto_params
        self.nb_params = nb_params

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        MK, NK = self.kernel_shape

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        M_orig, N_orig = f.shape

        # ── 2. Trim to odd dimensions ───────────────────────────────────
        f = make_size_odd(f)

        # ── 3. Gamma correction (optional) ──────────────────────────────
        if self.gamma_corr:
            f = gamma_correction(f, self.gamma)

        # ── 4a. Impulse noise detection & removal ───────────────────────
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                f,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
            )
            if impulse_info['has_impulse']:
                f = adaptive_median_filter(
                    f, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))

        # ── 4b. Noise estimation ────────────────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(f)

        # ── 4b½. Auto-params (λ, τ, α) from σ ─────────────────────────
        if self.auto_params is not None and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lambda = ap.get('k_lambda', 5000.0)
                k_tau = ap.get('k_tau', 10.0)
                k_alpha = ap.get('k_alpha', 0.1)
                self.lambda_val = max(100.0, k_lambda / max(sigma_n, 1e-6))
                self.tau = max(1e-6, k_tau * sigma_n ** 2)
                self.final_alpha = max(1e-5, k_alpha * sigma_n)
                if self.verbose:
                    print(f"[{self.name}] auto_params(σ={sigma_n:.5f}): "
                          f"λ={self.lambda_val:.1f}, "
                          f"τ={self.tau:.6f}, "
                          f"α={self.final_alpha:.5f}")

        # ── 4c. ScreeNOT SVD denoising ──────────────────────────────────
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            f, screenot_info = screenot_denoise(
                f,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )

        # ── 4d. ACT curvelet denoising ──────────────────────────────────
        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from .act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            f, act_info = act_denoise(
                f,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )

        # ── 4e. Pre-pyramid denoising ──────────────────────────────────
        if self.preprocess not in (None, 'none'):
            f = self._apply_denoise(
                f, self.preprocess, self.preprocess_params, noise_info)

        # ── 4f. PSD-based noise preprocessing ──────────────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            f, psd_info = self._apply_noise_preprocess(f)

        # ── 5. Build blind_denoise callback ─────────────────────────────
        blind_denoise_fn = None
        if self.blind_denoise not in (None, 'none'):
            def blind_denoise_fn(u_arr):
                return self._apply_blind_denoise(u_arr, noise_info)

        # ── 6. PSF estimation (blind step) ──────────────────────────────
        if self.method in ('mm', 'pd'):
            blind_params = {
                'outer_iters': self.outer_iters,
                'inner_iters': self.inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
            }
            ctf_params = {
                'final_lambda': self.lambda_val,
                'lambda_mult': self.lambda_mult,
                'scale_mult': self.scale_mult,
            }
            u, k = coarse_to_fine(
                f, MK, NK, blind_params, ctf_params,
                verbose=self.verbose, method=self.method,
                blind_denoise_fn=blind_denoise_fn)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Choose 'mm' or 'pd'.")

        # ── 6b. Kernel thresholding ─────────────────────────────────────
        k[k < self.kernel_threshold * k.max()] = 0.0
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        # ── 7. Pre-nonblind denoising ──────────────────────────────────
        if self.pre_nonblind not in (None, 'none'):
            f = self._apply_pre_nonblind(f, noise_info)

        # ── 8. Non-blind restoration ────────────────────────────────────
        if self.final_deconv == 'ringing_removal':
            from .solvers import ringing_artifacts_removal
            nbp = self.nb_params or {}
            u_restored = ringing_artifacts_removal(
                f, k,
                lambda_tv=nbp.get('lambda_tv', 1e-3),
                lambda_l0=nbp.get('lambda_l0', 2e-3),
                weight_ring=nbp.get('weight_ring', 1.0),
            )
        elif self.final_deconv == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            u_restored = adaptive_lp_deconv(
                f, k,
                alpha=nbp.get('alpha', 0.8),
                sigma_n=sigma_n,
                two_stage=nbp.get('two_stage', True),
            )
        elif self.final_deconv in ('tikhonov', 'wiener'):
            f_pad = pad_image(f, (MK, NK))
            f_pad = edgetaper(f_pad, k)

            if self.final_deconv == 'tikhonov':
                u_restored = tikhonov_filter(f_pad, k, alpha=self.final_alpha)
            else:
                u_restored = wiener_filter(f_pad, k, noise_snr=self.final_alpha)

            u_restored = crop_image(u_restored, (M_orig, N_orig), (MK, NK))
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'tikhonov', 'wiener', 'ringing_removal', "
                "or 'adaptive_lp'."
            )

        # ── Restore original dimensions (make_size_odd may have trimmed) ─
        rh, rw = u_restored.shape[:2]
        if rh < M_orig or rw < N_orig:
            u_restored = np.pad(
                u_restored,
                ((0, max(0, M_orig - rh)), (0, max(0, N_orig - rw))),
                mode='edge')
        u_restored = u_restored[:M_orig, :N_orig]

        u_final = np.clip(u_restored, 0.0, 1.0)

        # ── 9. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'lambda': self.lambda_val,
            'tau': self.tau,
            'method': self.method,
            'final_deconv': self.final_deconv,
            'final_alpha': self.final_alpha,
            'outer_iters': self.outer_iters,
            'inner_iters': self.inner_iters,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': {k_: v for k_, v in (impulse_info or {}).items()
                            if k_ != 'impulse_mask'} if impulse_info else None,
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'preprocess': self.preprocess,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': {k_: v for k_, v in (psd_info or {}).items()
                         if k_ != 'psd_2d'} if psd_info else None,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'auto_params': self.auto_params,
            'time': time.time() - start_time,
        }

        x_final = u_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    # ── Guided filter (box-filter variant, He et al. 2013) ─────────────
    @staticmethod
    def _guided_filter(I, p, r, eps):
        from scipy.ndimage import uniform_filter
        size = 2 * r + 1
        def box(x):
            return uniform_filter(x, size=size, mode='reflect')
        mean_I = box(I)
        mean_p = box(p)
        corr_Ip = box(I * p)
        var_I = box(I * I) - mean_I ** 2
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return box(a) * I + box(b)

    # ── Universal denoiser dispatch ───────────────────────────────────
    def _apply_denoise(self, img, method, params, noise_info):
        """Apply a spatial denoiser to a single-channel image [0, 1]."""
        if method is None or method == 'none':
            return img
        p = dict(params or {})
        sigma = noise_info.get('sigma_norm', None) if noise_info else None

        if method == 'tv':
            from skimage.restoration import denoise_tv_chambolle
            w = p.get('weight', max(0.01, sigma * 2) if sigma else 0.1)
            return denoise_tv_chambolle(img, weight=w)

        elif method == 'nlm':
            from skimage.restoration import (
                denoise_nl_means, estimate_sigma as _est_sig)
            sig = p.get('sigma', sigma)
            if sig is None:
                sig = float(np.mean(_est_sig(img)))
            h = p.get('h', 0.8 * sig)
            return denoise_nl_means(
                img, h=h,
                patch_size=p.get('patch_size', 5),
                patch_distance=p.get('patch_distance', 6),
                fast_mode=True)

        elif method == 'bilateral':
            import cv2
            d = p.get('d', 5)
            sc = p.get('sigma_color', sigma if sigma else 0.1)
            ss = p.get('sigma_space', 5.0)
            return cv2.bilateralFilter(
                img.astype(np.float32), d, float(sc), float(ss)
            ).astype(np.float64)

        elif method == 'guided':
            r = p.get('radius', 4)
            eps = p.get('eps',
                        sigma ** 2 * 4 if sigma else 0.01)
            return self._guided_filter(img, img, r, eps)

        elif method == 'bm3d':
            import bm3d as bm3d_lib
            sig = p.get('sigma', sigma if sigma else 0.05)
            return bm3d_lib.bm3d(img, sigma_psd=sig)

        elif method == 'act':
            from .act_denoise import act_denoise
            nv = p.get('noise_var', None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = p.get('threshold_setting', 's')
            result, _ = act_denoise(img, noise_var=nv,
                                    threshold_setting=ts)
            return result

        else:
            raise ValueError(
                f"Unknown denoiser='{method}'. Choose from: "
                f"'tv', 'nlm', 'bilateral', 'guided', 'bm3d', "
                f"'act', 'none'")

    # ── Noise estimation ─────────────────────────────────────────────
    def _estimate_noise(self, yg):
        if self.noise_estimation == 'chen':
            from .chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pca':
            from .pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pca'
            return result
        return None

    # ── PSD-based noise preprocessing ────────────────────────────────
    def _apply_noise_preprocess(self, yg):
        from .noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess,
            notch_filter, bandstop_filter,
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

        psd_info = analyze_noise_psd(
            yg,
            pch_size=p.get('pch_size', 32),
            n_smooth=p.get('n_smooth', 100),
            peak_threshold=p.get('peak_threshold', 100.0),
        )

        if mode == 'notch':
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
                f"Choose from: 'auto', 'notch', 'bandstop', 'none'")

        return yg_out, psd_info

    # ── Blind-loop denoiser (u before kernel step) ───────────────────
    def _apply_blind_denoise(self, x, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    # ── Pre-nonblind denoiser ───────────────────────────────────────
    def _apply_pre_nonblind(self, img, noise_info):
        return self._apply_denoise(
            img, self.pre_nonblind, self.pre_nonblind_params, noise_info)

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_val', self.lambda_val),
            ('tau', self.tau),
            ('outer_iters', self.outer_iters),
            ('inner_iters', self.inner_iters),
            ('method', self.method),
            ('kernel_threshold', self.kernel_threshold),
            ('final_deconv', self.final_deconv),
            ('final_alpha', self.final_alpha),
            ('gamma_correction', self.gamma_corr),
            ('gamma', self.gamma),
            ('verbose', self.verbose),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('noise_estimation', self.noise_estimation),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params', self.screenot_params),
            ('act_preprocess', self.act_preprocess),
            ('act_params', self.act_params),
            ('preprocess', self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('blind_denoise', self.blind_denoise),
            ('blind_denoise_params', self.blind_denoise_params),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('auto_params', self.auto_params),
            ('nb_params', self.nb_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                elif key in ('k_step', 'u_step'):
                    setattr(self, key, np.atleast_1d(
                        np.asarray(value, dtype=np.float64)))
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
