"""
pmp.py

Blind Image Deblurring Using Patch-wise Minimal Pixels (PMP) Prior.

Reference:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong:
    "A Simple Local Minimal Intensity Prior and An Improved Algorithm
    for Blind Image Deblurring", IEEE TCSVT, 2021.

Pipeline:
    1.  Normalise input to float64 [0, 1].
    2.  Grayscale conversion (if RGB).
    3a. Impulse noise detection & removal (optional).
    3b. Noise σ estimation (optional).
    3c. Auto-params from σ (optional).
    3d. ScreeNOT SVD denoising (optional, mutually exclusive with ACT).
    3e. ACT curvelet denoising (optional, mutually exclusive with ScreeNOT).
    3f. Spatial pre-blind denoising (optional).
    3g. PSD-based noise filtering (optional).
    3h. Histogram equalization (optional).
    4.  Build blind_denoise callback (optional).
    5.  Multi-scale blind deconvolution (blind_deconv).
    6.  Pre-nonblind denoising (optional).
    7.  Non-blind restoration (ringing_removal / adaptive_lp / wiener /
        tikhonov).
    8.  Return restored image (int16, [0, 255]) and kernel.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict, Optional


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


from .solvers import blind_deconv, ringing_artifacts_removal
from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter


class PMP_BD(DeconvolutionAlgorithm):


    def __init__(
        self,
        kernel_size: int = 25,
        lambda_pmp: float = 0.1,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        patch_r: int = None,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        denoise_eps: float = None,
        denoise_radius: int = 2,
        grad_smooth_sigma: float = None,
        pmp_quantile: float = 0.0,
        ensemble_denoise: bool = False,
        estimate_noise_internal: bool = False,
        noise_sigma_mult: float = 10.0,

        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        auto_params: dict = None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        preprocess: str = 'none',
        preprocess_params: dict = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        histogram_eq: str = 'none',
        histogram_eq_params: dict = None,
        blind_denoise: str = 'none',
        blind_denoise_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,

        final_deconv: str = 'ringing_removal',
        nb_params: dict = None,

        auto_mode: str = 'off',
        auto_mode_params: Optional[dict] = None,
        verbose: bool = False,
    ):
        super().__init__(name='PMP-BD')


        self.kernel_size = kernel_size
        self.lambda_pmp = lambda_pmp
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.patch_r = patch_r
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.denoise_eps = denoise_eps
        self.denoise_radius = denoise_radius
        self.grad_smooth_sigma = grad_smooth_sigma
        self.pmp_quantile = pmp_quantile
        self.ensemble_denoise = ensemble_denoise
        self.estimate_noise_internal = estimate_noise_internal
        self.noise_sigma_mult = noise_sigma_mult


        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.histogram_eq = histogram_eq
        self.histogram_eq_params = histogram_eq_params
        self.blind_denoise = blind_denoise
        self.blind_denoise_params = blind_denoise_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params


        self.final_deconv = final_deconv.lower()
        self.nb_params = nb_params
        self.verbose = verbose


        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params


        self._heavy_snapshot = {
            'lambda_pmp': float(lambda_pmp),
            'lambda_grad': float(lambda_grad),
            'pmp_quantile': float(pmp_quantile),
            'denoise_eps': denoise_eps,
            'denoise_radius': int(denoise_radius),
            'grad_smooth_sigma': grad_smooth_sigma,
            'ensemble_denoise': bool(ensemble_denoise),
            'estimate_noise_internal': bool(estimate_noise_internal),
            'noise_sigma_mult': float(noise_sigma_mult),
            'lambda_tv': float(lambda_tv),
            'lambda_l0': float(lambda_l0),
            'weight_ring': float(weight_ring),
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'final_deconv': self.final_deconv,
            'nb_params': nb_params,
        }


        self._clean_preset_default = {
            'lambda_pmp': 4e-3,
            'lambda_grad': 4e-3,
            'pmp_quantile': 0.0,
            'denoise_eps': None,
            'denoise_radius': 2,
            'grad_smooth_sigma': None,
            'ensemble_denoise': False,
            'estimate_noise_internal': False,
            'noise_sigma_mult': 10.0,
            'lambda_tv': 1e-3,
            'lambda_l0': 5e-4,
            'weight_ring': 1.0,
            'preprocess': 'none',
            'preprocess_params': None,
            'blind_denoise': 'none',
            'blind_denoise_params': None,
            'pre_nonblind': 'none',
            'pre_nonblind_params': None,
            'final_deconv': 'ringing_removal',
            'nb_params': None,
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}


    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()


        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0


        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]


        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                yg,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
                outlier_window=ip.get('outlier_window', 5),
            )
            if impulse_info['has_impulse']:
                if self.verbose:
                    print(f"[PMP-BD] Impulse noise detected "
                          f"(density={impulse_info['density']:.4f}), "
                          f"applying adaptive median filter")
                yg = adaptive_median_filter(
                    yg, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))


        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
            if self.verbose and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[PMP-BD] Noise estimation ({self.noise_estimation}): "
                      f"σ={sigma:.5f} (σ_255={sigma * 255:.2f})")
        elif self.auto_mode == 'robust':

            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(yg)
            if self.verbose and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[PMP-BD] auto_mode='robust' → PCA noise est.: "
                      f"σ={sigma:.5f} (σ_255={sigma * 255:.2f})")


        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)


        if (self.auto_mode != 'robust'
                and self.auto_params is not None
                and noise_info is not None):
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lp = ap.get('k_lambda_pmp', 5.0)
                k_lg = ap.get('k_lambda_grad', 0.2)
                k_q = ap.get('k_quantile', 3.0)
                k_tv = ap.get('k_lambda_tv', 0.05)
                k_l0 = ap.get('k_lambda_l0', 0.025)
                self.lambda_pmp = max(1e-2, k_lp * sigma_n)
                self.lambda_grad = max(1e-4, k_lg * sigma_n)
                self.pmp_quantile = min(0.2, k_q * sigma_n)
                self.lambda_tv = max(1e-4, k_tv * sigma_n)
                self.lambda_l0 = max(1e-5, k_l0 * sigma_n)
                if self.verbose:
                    print(f"[PMP-BD] auto_params(σ={sigma_n:.5f}): "
                          f"λ_pmp={self.lambda_pmp:.4f}, "
                          f"λ_grad={self.lambda_grad:.5f}, "
                          f"quantile={self.pmp_quantile:.3f}, "
                          f"λ_tv={self.lambda_tv:.5f}, "
                          f"λ_l0={self.lambda_l0:.6f}")


        screenot_info = None
        if self.screenot_preprocess == 'auto':
            if self.act_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            yg, screenot_info = screenot_denoise(
                yg,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if self.verbose:
                print(f"[PMP-BD] ScreeNOT applied "
                      f"(rank={screenot_info.get('rank', '?')})")


        act_info = None
        if self.act_preprocess == 'auto':
            from .act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            yg, act_info = act_denoise(
                yg,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if self.verbose:
                print("[PMP-BD] ACT curvelet denoising applied")


        if self.preprocess not in (None, 'none'):
            yg = self._apply_denoise(
                yg, self.preprocess, self.preprocess_params, noise_info)
            if self.verbose:
                print(f"[PMP-BD] Pre-blind denoise: {self.preprocess}")


        psd_info = None
        if self.noise_preprocess != 'none':
            yg, psd_info = self._apply_noise_preprocess(yg)
            if self.verbose:
                print(f"[PMP-BD] PSD noise preprocess: {self.noise_preprocess}")


        yg_for_restore = yg
        if self.histogram_eq not in (None, 'none'):
            yg = self._apply_histogram_eq(yg)
            if self.verbose:
                print(f"[PMP-BD] Histogram equalization: {self.histogram_eq}")


        blind_denoise_fn = None
        if self.blind_denoise not in (None, 'none'):
            def blind_denoise_fn(s_arr):
                return self._apply_blind_denoise(s_arr, noise_info)


        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'denoise_eps': self.denoise_eps,
            'denoise_radius': self.denoise_radius,
            'grad_smooth_sigma': self.grad_smooth_sigma,
            'pmp_quantile': self.pmp_quantile,
            'ensemble_denoise': self.ensemble_denoise,
            'estimate_noise': self.estimate_noise_internal,
            'noise_sigma_mult': self.noise_sigma_mult,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_pmp, self.lambda_grad, opts,
            patch_r=self.patch_r,
            blind_denoise_fn=blind_denoise_fn,
            iteration_callback=self._callback,
        )


        y_nb = yg_for_restore
        if self.pre_nonblind not in (None, 'none'):
            y_nb = self._apply_denoise(
                y_nb, self.pre_nonblind, self.pre_nonblind_params, noise_info)
            if self.verbose:
                print(f"[PMP-BD] Pre-nonblind denoise: {self.pre_nonblind}")


        if self.final_deconv == 'ringing_removal':
            Latent = ringing_artifacts_removal(
                y_nb, kernel,
                self.lambda_tv, self.lambda_l0, self.weight_ring,
            )
        elif self.final_deconv == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            Latent = adaptive_lp_deconv(
                y_nb, kernel,
                alpha=nbp.get('alpha', 0.8),
                sigma_n=sigma_n,
                two_stage=nbp.get('two_stage', True),
            )
        elif self.final_deconv == 'wiener':
            Latent = self._wiener_filter(
                y_nb, kernel, noise_info)
        elif self.final_deconv == 'tikhonov':
            Latent = self._tikhonov_filter(
                y_nb, kernel, noise_info)
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'ringing_removal', 'adaptive_lp', "
                "'wiener', or 'tikhonov'.")

        Latent = np.clip(Latent, 0.0, 1.0)


        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_pmp': self.lambda_pmp,
            'lambda_grad': self.lambda_grad,
            'pmp_quantile': self.pmp_quantile,
            'final_deconv': self.final_deconv,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
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
            'histogram_eq': self.histogram_eq,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'auto_params': self.auto_params,
            'nb_params': self.nb_params,
            'auto_mode': self.auto_mode,
            'orchestrator_info': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel


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


    def _apply_denoise(self, img, method, params, noise_info):

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


    def _apply_histogram_eq(self, img):

        from skimage.exposure import equalize_adapthist, equalize_hist
        method = self.histogram_eq
        p = self.histogram_eq_params or {}

        if method == 'clahe':
            return equalize_adapthist(
                img,
                clip_limit=p.get('clip_limit', 0.01),
                nbins=p.get('nbins', 256),
                kernel_size=p.get('kernel_size', None),
            )
        elif method == 'global':
            return equalize_hist(img)
        else:
            raise ValueError(
                f"Unknown histogram_eq='{method}'. "
                f"Choose from: 'clahe', 'global', 'none'")


    def _apply_blind_denoise(self, s, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(s, self.blind_denoise, p, noise_info)


    def _orchestrate_robust(self, noise_info):


        heavy = self._heavy_snapshot
        clean = dict(self._clean_preset_default)
        amp = dict(self.auto_mode_params or {})
        if isinstance(amp.get('clean_preset'), dict):
            clean.update(amp['clean_preset'])


        for k, v in heavy.items():
            setattr(self, k, v)


        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))

        force_heavy = False
        nt = (noise_info or {}).get('noise_type', None)
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.01))
        if nt in ('poisson', 'poisson_gaussian') and sigma >= force_heavy_sigma:
            force_heavy = True


        _numeric_keys = (
            'lambda_pmp', 'lambda_grad', 'pmp_quantile',
            'lambda_tv', 'lambda_l0', 'weight_ring',
            'noise_sigma_mult',
        )

        _discrete_keys = (
            'denoise_eps', 'grad_smooth_sigma', 'ensemble_denoise',
            'estimate_noise_internal',
            'preprocess', 'preprocess_params',
            'blind_denoise', 'blind_denoise_params',
            'pre_nonblind', 'pre_nonblind_params',
            'final_deconv', 'nb_params',
        )


        if sigma <= sigma_clean and not force_heavy:
            for k in _numeric_keys + _discrete_keys:
                if k in clean:
                    setattr(self, k, clean[k])
            info = {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'noise_type': nt,
                'lambda_pmp': float(self.lambda_pmp),
                'lambda_grad': float(self.lambda_grad),
                'pmp_quantile': float(self.pmp_quantile),
                'lambda_tv': float(self.lambda_tv),
                'lambda_l0': float(self.lambda_l0),
                'denoise_eps': self.denoise_eps,
                'ensemble_denoise': self.ensemble_denoise,
                'grad_smooth_sigma': self.grad_smooth_sigma,
                'preprocess': self.preprocess,
                'blind_denoise': self.blind_denoise,
                'pre_nonblind': self.pre_nonblind,
                'final_deconv': self.final_deconv,
            }
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"clean preset → λ_pmp={self.lambda_pmp:.4f}, "
                      f"λ_grad={self.lambda_grad:.5f}, "
                      f"q={self.pmp_quantile:.3f}, "
                      f"denoise_eps={self.denoise_eps}, "
                      f"ensemble={self.ensemble_denoise}, "
                      f"σ_grad={self.grad_smooth_sigma}, "
                      f"blind={self.blind_denoise}, "
                      f"pre={self.preprocess}, pre_nb={self.pre_nonblind}")
            return info


        if sigma >= sigma_heavy or force_heavy:
            w = 1.0
            regime = 'heavy'
        else:
            w = (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-9)
            w = float(np.clip(w, 0.0, 1.0))
            regime = 'medium' if w < 0.95 else 'heavy'


        for k in _numeric_keys:
            c_val = float(clean.get(k, heavy[k]))
            h_val = float(heavy[k])
            setattr(self, k, (1.0 - w) * c_val + w * h_val)


        use_heavy_discrete = w >= 0.5
        for k in _discrete_keys:
            src = heavy if use_heavy_discrete else clean
            if k in src:
                setattr(self, k, src[k])


        if use_heavy_discrete and isinstance(heavy.get('denoise_eps'), (int, float)):
            self.denoise_eps = float(heavy['denoise_eps']) * float(w)


        if use_heavy_discrete and isinstance(heavy.get('grad_smooth_sigma'),
                                              (int, float)):
            self.grad_smooth_sigma = float(heavy['grad_smooth_sigma']) * float(w)

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': nt,
            'lambda_pmp': float(self.lambda_pmp),
            'lambda_grad': float(self.lambda_grad),
            'pmp_quantile': float(self.pmp_quantile),
            'lambda_tv': float(self.lambda_tv),
            'lambda_l0': float(self.lambda_l0),
            'denoise_eps': self.denoise_eps,
            'ensemble_denoise': self.ensemble_denoise,
            'grad_smooth_sigma': self.grad_smooth_sigma,
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={nt}): "
                  f"λ_pmp={self.lambda_pmp:.4f}, "
                  f"λ_grad={self.lambda_grad:.5f}, "
                  f"q={self.pmp_quantile:.3f}, "
                  f"λ_tv={self.lambda_tv:.5f}, λ_l0={self.lambda_l0:.6f}, "
                  f"denoise_eps={self.denoise_eps}, "
                  f"ensemble={self.ensemble_denoise}, "
                  f"σ_grad={self.grad_smooth_sigma}, "
                  f"blind={self.blind_denoise}, "
                  f"pre={self.preprocess}, pre_nb={self.pre_nonblind}, "
                  f"final={self.final_deconv}")
        return info


    @staticmethod
    def _psf2otf(psf, shape):
        padded = np.zeros(shape, dtype=np.float64)
        ph, pw = psf.shape
        padded[:ph, :pw] = psf
        padded = np.roll(padded, -(ph // 2), axis=0)
        padded = np.roll(padded, -(pw // 2), axis=1)
        return np.fft.fft2(padded)

    def _wiener_filter(self, img, kernel, noise_info):
        nbp = self.nb_params or {}
        noise_snr = nbp.get('noise_snr', 0.01)

        H_otf = self._psf2otf(kernel, img.shape)
        H_conj = np.conj(H_otf)
        G = np.fft.fft2(img)
        denom = np.abs(H_otf) ** 2 + noise_snr
        result = np.real(np.fft.ifft2(H_conj * G / denom))
        return result

    def _tikhonov_filter(self, img, kernel, noise_info):
        nbp = self.nb_params or {}
        alpha = nbp.get('alpha', 0.01)

        H_otf = self._psf2otf(kernel, img.shape)
        H_conj = np.conj(H_otf)
        G = np.fft.fft2(img)

        dx = np.array([[1, -1]], dtype=np.float64)
        dy = np.array([[1], [-1]], dtype=np.float64)
        Dx = self._psf2otf(dx, img.shape)
        Dy = self._psf2otf(dy, img.shape)
        reg = np.abs(Dx) ** 2 + np.abs(Dy) ** 2

        denom = np.abs(H_otf) ** 2 + alpha * reg
        result = np.real(np.fft.ifft2(H_conj * G / denom))
        return result


    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_pmp', self.lambda_pmp),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('patch_r', self.patch_r),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('denoise_eps', self.denoise_eps),
            ('denoise_radius', self.denoise_radius),
            ('grad_smooth_sigma', self.grad_smooth_sigma),
            ('pmp_quantile', self.pmp_quantile),
            ('ensemble_denoise', self.ensemble_denoise),
            ('estimate_noise_internal', self.estimate_noise_internal),
            ('noise_sigma_mult', self.noise_sigma_mult),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
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
            ('final_deconv', self.final_deconv),
            ('nb_params', self.nb_params),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'final_deconv':
                    setattr(self, key, value.lower())
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
