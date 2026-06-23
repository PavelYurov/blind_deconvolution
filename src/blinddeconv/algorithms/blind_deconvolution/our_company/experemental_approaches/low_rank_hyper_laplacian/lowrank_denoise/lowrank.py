import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    compute_gradients,
    build_scale_pyramid,
    center_kernel,
    normalize_kernel,
    resize_image,
    edgetaper,
    rgb_to_ycbcr,
    ycbcr_to_rgb,
)
from .solvers import (
    optimize_image,
    optimize_kernel,
    low_rank_regularization,
    fast_deconv_hyper_laplacian,
    ringing_artifacts_removal,
)
from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from pathlib import Path
import sys
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

class LowRankBD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 31,
        lambda_: float = 2e-3,
        sigma: float = 1.0,
        tau: float = 1e-5,
        delta: float = 1e-5,
        kernel_beta: float = 3e-3,
        max_iter: int = 7,
        max_irls: int = 3,
        max_cg: int = 200,
        max_iter_k: int = 50,
        max_iter_rank: int = 3,
        iter_k_rank: int = 3,
        exp_a: float = 0.8,
        thr_e: float = 1.0 / 1500,
        alpha_multiplier: float = 2.0,
        threshold: float = 0.05,
        nb_lambda: float = 3000.0,
        nb_alpha: float = 0.5,
        verbose: bool = False,

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
        nb_method: str = 'hyper_laplacian',
        nb_params: dict = None,
    ):
        super().__init__(name='LowRank-BD')

        assert kernel_size >= 3 and kernel_size % 2 == 1,\
            "kernel_size must be odd and >= 3"

        self.kernel_size      = kernel_size
        self.lambda_          = lambda_
        self.sigma            = sigma
        self.tau              = tau
        self.delta            = delta
        self.kernel_beta      = kernel_beta
        self.max_iter         = max_iter
        self.max_irls         = max_irls
        self.max_cg           = max_cg
        self.max_iter_k       = max_iter_k
        self.max_iter_rank    = max_iter_rank
        self.iter_k_rank      = iter_k_rank
        self.exp_a            = exp_a
        self.thr_e            = thr_e
        self.alpha_multiplier = alpha_multiplier
        self.threshold        = threshold
        self.nb_lambda        = nb_lambda
        self.nb_alpha         = nb_alpha
        self.verbose          = verbose
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
        self.nb_method = nb_method
        self.nb_params = nb_params

        self.history: Dict[str, list]    = {'kernel_diff': [], 'scale': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        is_color = (y.ndim == 3 and y.shape[2] == 3)

        if is_color:
            ycbcr  = rgb_to_ycbcr(y)
            y_gray = ycbcr[:, :, 0].copy()
        else:
            y_gray = y.copy()

        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                y_gray,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
            )
            if impulse_info['has_impulse']:
                y_gray = adaptive_median_filter(
                    y_gray, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))
                if is_color:
                    for ch in range(3):
                        ch_info = detect_impulse_noise(
                            ycbcr[:, :, ch],
                            density_threshold=ip.get(
                                'density_threshold', 0.0005),
                            outlier_threshold=ip.get(
                                'outlier_threshold', 0.08),
                        )
                        if ch_info['has_impulse']:
                            ycbcr[:, :, ch] = adaptive_median_filter(
                                ycbcr[:, :, ch], ch_info['impulse_mask'],
                                max_window=ip.get('max_window', 7))

        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(y_gray)

        if self.auto_params is not None and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lambda = ap.get('k_lambda', 0.2)
                k_thr_e  = ap.get('k_thr_e', 4.0)
                k_nb     = ap.get('k_nb', 30.0)
                self.lambda_   = max(1e-4, k_lambda * sigma_n)
                self.thr_e     = max(1e-6, k_thr_e * sigma_n ** 2)
                self.nb_lambda = max(100.0, k_nb / max(sigma_n, 1e-6))
                if self.verbose:
                    print(f"[{self.name}] auto_params(σ={sigma_n:.5f}): "
                          f"λ={self.lambda_:.4f}, "
                          f"ε={self.thr_e:.6f}, "
                          f"nb_λ={self.nb_lambda:.0f}")

        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            y_gray, screenot_info = screenot_denoise(
                y_gray,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )

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
            y_gray, act_info = act_denoise(
                y_gray,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )

        if self.preprocess not in (None, 'none'):
            y_gray = self._apply_denoise(
                y_gray, self.preprocess, self.preprocess_params,
                noise_info)

        psd_info = None
        if self.noise_preprocess != 'none':
            y_gray, psd_info = self._apply_noise_preprocess(y_gray)

        if is_color:
            ycbcr[:, :, 0] = y_gray

        K     = self.kernel_size
        H, W  = y_gray.shape

        if self.verbose:
            print(f"[{self.name}] Image: {H}×{W}, "
                  f"Kernel: {K}×{K}")

        scales = build_scale_pyramid(K)
        num_scales = len(scales)

        if self.verbose:
            print(f"[{self.name}] Scales: {scales}")

        min_scale = scales[0]

        k = np.zeros((min_scale, min_scale))
        k[min_scale // 2, min_scale // 2] = 1.0

        x = None

        for si, Ki in enumerate(scales):
            if self.verbose:
                print(f"[{self.name}] Scale {si + 1}/{num_scales}: "
                      f"kernel {Ki}×{Ki}")

            ratio = Ki / K
            hw = (max(int(np.floor(H * ratio)), Ki + 2),
                  max(int(np.floor(W * ratio)), Ki + 2))
            y_small = resize_image(y_gray, hw)

            if x is None:
                x = y_small.copy()
            else:
                x = resize_image(x, hw)

            if si > 0:
                k = resize_image(k, (Ki, Ki))
                k = normalize_kernel(k)

            scale_idx = num_scales - 1 - si
            alpha = self.lambda_ * self.alpha_multiplier ** (
                scale_idx - 0.5
            )

            tau_scale = self.tau * (si + 1) / num_scales

            for it in range(self.max_iter):
                k_prev = k.copy()

                x = optimize_image(
                    x, k, y_small, alpha,
                    self.max_irls, self.max_cg,
                    self.exp_a, self.thr_e,
                )

                if self.blind_denoise not in (None, 'none'):
                    x_dk = self._apply_blind_denoise(x, noise_info)
                else:
                    x_dk = x

                for ir in range(self.iter_k_rank):

                    k = optimize_kernel(
                        x_dk, k, y_small,
                        self.kernel_beta, self.max_iter_k,
                    )

                    if self.sigma > 0:
                        k = low_rank_regularization(
                            k, self.max_iter_rank,
                            tau_scale, self.delta,
                        )

                    k = normalize_kernel(k)

                k = normalize_kernel(
                    k,
                    self.threshold * (it + 1) / self.max_iter,
                )

                diff = np.linalg.norm(k - k_prev)
                self.history['kernel_diff'].append(diff)
                self.history['scale'].append(Ki)

                if self.verbose:
                    print(f"  Iter {it + 1}/{self.max_iter}: "
                          f"‖Δk‖ = {diff:.6f}")

            k = center_kernel(k)
            k = normalize_kernel(k)

        k = normalize_kernel(k, self.threshold)

        if self.verbose:
            print(f"[{self.name}] Kernel estimated in "
                  f"{time.time() - start_time:.1f} s")

        if self.pre_nonblind not in (None, 'none'):
            if is_color:
                ycbcr[:, :, 0] = self._apply_pre_nonblind(
                    ycbcr[:, :, 0], noise_info)
            else:
                y_gray = self._apply_pre_nonblind(y_gray, noise_info)

        if self.verbose:
            print(f"[{self.name}] Non-blind deconvolution "
                  f"(method={self.nb_method}, "
                  f"λ={self.nb_lambda}, α={self.nb_alpha}) ...")

        if self.nb_method == 'ringing_removal':

            nbp = self.nb_params or {}
            y_nb = ycbcr[:, :, 0] if is_color else y_gray
            restored_ch = ringing_artifacts_removal(
                y_nb, k,
                lambda_tv=nbp.get('lambda_tv', 1e-3),
                lambda_l0=nbp.get('lambda_l0', 2e-3),
                weight_ring=nbp.get('weight_ring', 1.0),
            )
            if is_color:
                result = ycbcr.copy()
                result[:, :, 0] = np.clip(restored_ch, 0.0, 1.0)
                result = ycbcr_to_rgb(result)
                result = np.clip(result, 0.0, 1.0)
            else:
                result = np.clip(restored_ch, 0.0, 1.0)
        else:

            bhs = K // 2
            if is_color:
                y_pad = np.pad(ycbcr[:, :, 0], bhs, mode='edge')
                for _ in range(4):
                    y_pad = edgetaper(y_pad, k)

                restored_y = fast_deconv_hyper_laplacian(
                    y_pad, k, self.nb_lambda, self.nb_alpha,
                )
                restored_y = restored_y[bhs: bhs + H, bhs: bhs + W]

                result = ycbcr.copy()
                result[:, :, 0] = restored_y
                result = ycbcr_to_rgb(result)
                result = np.clip(result, 0.0, 1.0)
            else:
                y_pad = np.pad(y_gray, bhs, mode='edge')
                for _ in range(4):
                    y_pad = edgetaper(y_pad, k)

                result = fast_deconv_hyper_laplacian(
                    y_pad, k, self.nb_lambda, self.nb_alpha,
                )
                result = result[bhs: bhs + H, bhs: bhs + W]
                result = np.clip(result, 0.0, 1.0)

        self.timer = time.time() - start_time

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda':      self.lambda_,
            'sigma':       self.sigma,
            'tau':         self.tau,
            'nb_lambda':   self.nb_lambda,
            'nb_alpha':    self.nb_alpha,
            'scales':      scales,
            'iterations':  sum(
                1 for s in self.history['scale'] if s == K
            ),
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
            'nb_method': self.nb_method,
            'total_time':  self.timer,
        }

        if self.verbose:
            print(f"[{self.name}] Done in {self.timer:.1f} s")

        result = np.round(result * 255.0).astype(np.int16)
        return result, k

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

    def _apply_blind_denoise(self, x, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    def _apply_pre_nonblind(self, img, noise_info):
        return self._apply_denoise(
            img, self.pre_nonblind, self.pre_nonblind_params, noise_info)

    def get_param(self) -> List[Tuple[str, Any]]:

        return [
            ('kernel_size',      self.kernel_size),
            ('lambda',           self.lambda_),
            ('sigma',            self.sigma),
            ('tau',              self.tau),
            ('delta',            self.delta),
            ('kernel_beta',      self.kernel_beta),
            ('max_iter',         self.max_iter),
            ('max_irls',         self.max_irls),
            ('max_cg',           self.max_cg),
            ('max_iter_k',       self.max_iter_k),
            ('max_iter_rank',    self.max_iter_rank),
            ('iter_k_rank',      self.iter_k_rank),
            ('exp_a',            self.exp_a),
            ('thr_e',            self.thr_e),
            ('alpha_multiplier', self.alpha_multiplier),
            ('threshold',        self.threshold),
            ('nb_lambda',        self.nb_lambda),
            ('nb_alpha',         self.nb_alpha),
            ('verbose',          self.verbose),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params',   self.impulse_params),
            ('noise_estimation', self.noise_estimation),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params',  self.screenot_params),
            ('act_preprocess',   self.act_preprocess),
            ('act_params',       self.act_params),
            ('preprocess',       self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('blind_denoise',    self.blind_denoise),
            ('blind_denoise_params', self.blind_denoise_params),
            ('pre_nonblind',     self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('auto_params',      self.auto_params),
            ('nb_method',        self.nb_method),
            ('nb_params',        self.nb_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:

        for key, value in params.items():
            if key == 'lambda':
                self.lambda_ = value
            elif hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:

        return self.history

    def get_hyperparams(self) -> dict:

        return self.hyperparams
