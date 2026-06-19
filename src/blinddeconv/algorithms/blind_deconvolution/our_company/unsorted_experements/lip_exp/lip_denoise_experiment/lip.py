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


    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_val: float = 30000.0,
        tau: float = 1e-3,
        outer_iters: int = 140,
        inner_iters: int = 5,
        k_step: Any = None,
        u_step: Any = None,
        pd_tau: float = None,
        pd_sigma: float = None,
        h_mode: str = 'closed',
        h_lut_size: int = 4096,
        h_lut_xi_max: float = 4.0,
        lambda_mult: float = 2.1,
        scale_mult: float = 1.4142135623730951,
        gamma_correction: bool = False,
        gamma: float = 1.0,
        method: str = 'mm',
        kernel_threshold: float = 0.05,
        final_deconv: str = 'tikhonov',
        final_alpha: float = 0.001,
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
        nb_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
    ):
        super().__init__(name='LIP-BD')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_val = lambda_val
        self.tau = tau
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters


        if k_step is None:
            self.k_step = np.array([1e-2, 5e-3, 1e-3, 5e-4])
        else:
            self.k_step = np.atleast_1d(np.asarray(k_step, dtype=np.float64))
        if u_step is None:
            self.u_step = np.array([1e-2, 5e-3, 1e-3, 1e-3])
        else:
            self.u_step = np.atleast_1d(np.asarray(u_step, dtype=np.float64))


        self.pd_tau = pd_tau
        self.pd_sigma = pd_sigma
        self.h_mode = h_mode
        self.h_lut_size = h_lut_size
        self.h_lut_xi_max = h_lut_xi_max

        self.lambda_mult = lambda_mult
        self.scale_mult = scale_mult
        self.gamma_corr = gamma_correction
        self.gamma = gamma
        self.method = method.lower()
        self.kernel_threshold = kernel_threshold
        self.final_deconv = final_deconv.lower()
        self.final_alpha = final_alpha
        self.verbose = verbose


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
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params


        self._defaults_snapshot = {
            'lambda_val': float(lambda_val),
            'tau': float(tau),
            'final_alpha': float(final_alpha),
            'final_deconv': self.final_deconv,
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}


    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        MK, NK = self.kernel_shape


        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        M_orig, N_orig = f.shape


        f = make_size_odd(f)


        if self.gamma_corr:
            f = gamma_correction(f, self.gamma)


        sanitation_result = None
        impulse_info = None
        noise_info = None
        orchestrator_info = None
        if self.auto_mode == 'sanitation':
            from blinddeconv.algorithms.blind_deconvolution.our_company\
                ._sanitation_cascade import sanitize as _sanitize
            sanitation_result = _sanitize(f, verbose=self.verbose)
            f = sanitation_result.image_clean.astype(np.float64, copy=False)
            noise_info = dict(sanitation_result.noise_info)
            self.last_sanitation = sanitation_result
            if self.verbose:
                print(f"[{self.name}] sanitation: branch="
                      f"{sanitation_result.branch}, "
                      f"residual_σ={sanitation_result.residual_sigma:.5f}, "
                      f"residual_type={sanitation_result.residual_type}")
        else:

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


            if self.noise_estimation != 'none':
                noise_info = self._estimate_noise(f)
            elif self.auto_mode == 'robust':


                self.noise_estimation = 'pca'
                noise_info = self._estimate_noise(f)


            if self.auto_mode == 'robust':
                orchestrator_info = self._orchestrate_robust(noise_info)


        if self.auto_params is not None and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lambda = ap.get('k_lambda', 5000.0)
                k_tau = ap.get('k_tau', 10.0)
                k_alpha = ap.get('k_alpha', 0.1)


                sigma_floor = ap.get('sigma_floor', 0.002)
                if sigma_n >= sigma_floor:


                    lambda_cap = ap.get('lambda_cap', None)
                    lam_new = k_lambda / max(sigma_n, 1e-6)
                    if lambda_cap is not None:
                        lam_new = min(lam_new, float(lambda_cap))
                    self.lambda_val = max(100.0, lam_new)
                    self.tau = max(1e-6, k_tau * sigma_n ** 2)
                    self.final_alpha = max(1e-5, k_alpha * sigma_n)
                    if self.verbose:
                        print(f"[{self.name}] auto_params(σ={sigma_n:.5f}): "
                              f"λ={self.lambda_val:.1f}, "
                              f"τ={self.tau:.6f}, "
                              f"α={self.final_alpha:.5f}")
                elif self.verbose:
                    print(f"[{self.name}] auto_params skipped "
                          f"(σ={sigma_n:.5f} < floor={sigma_floor})")


        screenot_info = None
        if self.screenot_preprocess == 'auto' and self.auto_mode != 'sanitation':
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


        act_info = None
        if self.act_preprocess == 'auto' and self.auto_mode != 'sanitation':
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


        if self.preprocess not in (None, 'none') and self.auto_mode != 'sanitation':
            f = self._apply_denoise(
                f, self.preprocess, self.preprocess_params, noise_info)


        psd_info = None
        if self.noise_preprocess != 'none' and self.auto_mode != 'sanitation':
            f, psd_info = self._apply_noise_preprocess(f)


        blind_denoise_fn = None
        if (self.blind_denoise not in (None, 'none')
                and self.auto_mode != 'sanitation'):
            def blind_denoise_fn(u_arr):
                return self._apply_blind_denoise(u_arr, noise_info)


        if self.method in ('mm', 'pd', 'cv'):
            blind_params = {
                'outer_iters': self.outer_iters,
                'inner_iters': self.inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
                'pd_tau': self.pd_tau,
                'pd_sigma': self.pd_sigma,
                'h_mode': self.h_mode,
                'h_lut_size': self.h_lut_size,
                'h_lut_xi_max': self.h_lut_xi_max,
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
            raise ValueError(
                f"Unknown method '{self.method}'. Choose 'mm', 'pd', or 'cv'.")


        k[k < self.kernel_threshold * k.max()] = 0.0
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum


        if (self.pre_nonblind not in (None, 'none')
                and self.auto_mode != 'sanitation'):
            f = self._apply_pre_nonblind(f, noise_info)


        if self.final_deconv == 'ringing_removal':
            from .solvers import ringing_artifacts_removal
            nbp = self.nb_params or {}
            u_restored = ringing_artifacts_removal(
                f, k,
                lambda_tv=nbp.get('lambda_tv', 1e-3),
                lambda_l0=nbp.get('lambda_l0', 2e-3),
                weight_ring=nbp.get('weight_ring', 1.0),
            )
        elif self.final_deconv == 'blend':


            from .solvers import ringing_artifacts_removal
            nbp = self.nb_params or {}
            blend_w = float(nbp.get('blend_weight', 0.5))

            f_pad = pad_image(f, (MK, NK))
            f_pad = edgetaper(f_pad, k)
            u_tik = tikhonov_filter(f_pad, k, alpha=self.final_alpha)
            u_tik = crop_image(u_tik, (M_orig, N_orig), (MK, NK))

            u_rr = ringing_artifacts_removal(
                f, k,
                lambda_tv=nbp.get('lambda_tv', 1e-3),
                lambda_l0=nbp.get('lambda_l0', 2e-3),
                weight_ring=nbp.get('weight_ring', 1.0),
            )


            h = min(u_tik.shape[0], u_rr.shape[0])
            w = min(u_tik.shape[1], u_rr.shape[1])
            u_restored = ((1.0 - blend_w) * u_tik[:h, :w]
                          + blend_w * u_rr[:h, :w])
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
                "'adaptive_lp', or 'blend'."
            )


        rh, rw = u_restored.shape[:2]
        if rh < M_orig or rw < N_orig:
            u_restored = np.pad(
                u_restored,
                ((0, max(0, M_orig - rh)), (0, max(0, N_orig - rw))),
                mode='edge')
        u_restored = u_restored[:M_orig, :N_orig]

        u_final = np.clip(u_restored, 0.0, 1.0)


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
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'sanitation': ({
                'branch': sanitation_result.branch,
                'residual_sigma': sanitation_result.residual_sigma,
                'residual_type': sanitation_result.residual_type,
                'actions': sanitation_result.actions,
                'noise_info': sanitation_result.noise_info,
            } if sanitation_result is not None else None),
            'time': time.time() - start_time,
        }

        x_final = u_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k


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


        if p.pop('use_vst', False) and noise_info is not None:
            return self._apply_denoise_vst(img, method, p, noise_info)

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

        elif method == 'ensemble':


            sig = p.get('sigma', sigma if sigma else 0.05)
            members = p.get('members', ('bm3d', 'nlm', 'bilateral'))
            weights = p.get('weights', None)
            outs = []
            for m in members:
                sub_params = dict(p)
                sub_params['sigma'] = sig

                if m == 'ensemble':
                    continue
                outs.append(self._apply_denoise(img, m, sub_params, noise_info))
            if not outs:
                return img
            if weights is None:
                return np.mean(np.stack(outs, axis=0), axis=0)
            ws = np.asarray(weights, dtype=np.float64)
            ws = ws / ws.sum()
            stacked = np.stack(outs, axis=0)
            return np.tensordot(ws, stacked, axes=(0, 0))

        else:
            raise ValueError(
                f"Unknown denoiser='{method}'. Choose from: "
                f"'tv', 'nlm', 'bilateral', 'guided', 'bm3d', "
                f"'act', 'ensemble', 'none'")


    def _apply_denoise_vst(self, img, method, params, noise_info):


        a = float(noise_info.get('a', 0.0) or 0.0) if noise_info else 0.0
        b = float(noise_info.get('b', 0.0) or 0.0) if noise_info else 0.0
        A = a / 255.0
        B = b / (255.0 ** 2)


        if A < 1e-8:
            sub = dict(params)
            sub.setdefault('sigma', noise_info.get('sigma_norm', None)
                           if noise_info else None)
            return self._apply_denoise(img, method, sub, noise_info)


        inner = np.maximum(A * img + (3.0 / 8.0) * A * A + B, 0.0)
        z = (2.0 / np.sqrt(A)) * np.sqrt(inner)


        zmax = float(max(z.max(), 1e-6))
        z_scaled = z / zmax
        sigma_scaled = 1.0 / zmax

        sub = dict(params)
        sub['sigma'] = sigma_scaled
        sub['weight'] = max(0.005, 2.0 * sigma_scaled)
        sub['sigma_color'] = sigma_scaled


        inner_info = {'sigma_norm': sigma_scaled, 'method': 'vst'}
        denoised_scaled = self._apply_denoise(
            z_scaled, method, sub, inner_info)

        z_clean = denoised_scaled * zmax


        x_rec = (z_clean / 2.0) ** 2 - (3.0 / 8.0) * A - B / max(A, 1e-8)
        return np.clip(x_rec, 0.0, 1.0)


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


    def _orchestrate_robust(self, noise_info):


        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})


        self.lambda_val = snap['lambda_val']
        self.tau = snap['tau']
        self.final_alpha = snap['final_alpha']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.blind_denoise = snap['blind_denoise']
        self.blind_denoise_params = snap['blind_denoise_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']


        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))
        blend_weight_clean = float(amp.get('blend_weight', 0.5))


        force_heavy = False
        nt = (noise_info or {}).get('noise_type', None)
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.01))
        if nt in ('poisson', 'poisson_gaussian') and sigma >= force_heavy_sigma:
            force_heavy = True


        if sigma <= sigma_clean and not force_heavy:
            w = 0.0
            regime = 'clean'
            if snap['final_deconv'] == 'auto':
                self.final_deconv = 'blend'

                if self.nb_params is None:
                    self.nb_params = {'blend_weight': blend_weight_clean}
                elif 'blend_weight' not in self.nb_params:
                    self.nb_params = dict(self.nb_params)
                    self.nb_params['blend_weight'] = blend_weight_clean
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"defaults kept, final_deconv={self.final_deconv}")
            return {
                'sigma_norm': sigma, 'w': w, 'regime': regime,
                'final_deconv': self.final_deconv,
                'preprocess': self.preprocess,
                'blind_denoise': self.blind_denoise,
                'pre_nonblind': self.pre_nonblind,
                'lambda_val': float(self.lambda_val),
                'tau': float(self.tau),
                'final_alpha': float(self.final_alpha),
            }


        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / (sigma_heavy - sigma_clean))
        regime = 'heavy' if w > 0.95 else 'medium'


        noise_type = (noise_info or {}).get('noise_type', 'gaussian')
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')


        k_lambda = float(amp.get('k_lambda', 5000.0))
        k_tau = float(amp.get('k_tau', 10.0))
        k_alpha = float(amp.get('k_alpha', 0.1))
        lam_cap = float(amp.get('lambda_cap', 1e5))
        tau_floor = float(amp.get('tau_floor', 1e-4))

        lam_noisy = float(np.clip(k_lambda / max(sigma, 1e-6),
                                  100.0, lam_cap))
        tau_noisy = max(tau_floor, k_tau * sigma ** 2)
        alpha_noisy = max(1e-5, k_alpha * sigma)

        self.lambda_val = (1.0 - w) * snap['lambda_val'] + w * lam_noisy
        self.tau = max(tau_floor, (1.0 - w) * snap['tau'] + w * tau_noisy)
        self.final_alpha = max(1e-5,
                               (1.0 - w) * snap['final_alpha']
                               + w * alpha_noisy)


        if w < 0.5:
            self.blind_denoise = 'bilateral'
            self.blind_denoise_params = {
                'sigma_color': float(max(sigma, 0.01)),
                'sigma_space': 5.0,
            }
        else:
            self.blind_denoise = 'bilateral'
            self.blind_denoise_params = {
                'sigma_color': float(max(2.0 * sigma, 0.02)),
                'sigma_space': 7.0,
            }


        if poisson_like:
            self.preprocess = 'act'
            self.preprocess_params = {'threshold_setting': 's'}
        elif w < 0.6:
            self.preprocess = 'bilateral'
            self.preprocess_params = {
                'sigma_color': float(sigma),
                'sigma_space': 5.0,
            }
        else:
            self.preprocess = 'bm3d'
            self.preprocess_params = {'sigma': float(sigma)}


        if poisson_like:
            self.pre_nonblind = 'act'
            self.pre_nonblind_params = {'threshold_setting': 's'}
        elif w < 0.6:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma': float(max(sigma, 0.01))}
        else:
            self.pre_nonblind = 'ensemble'
            self.pre_nonblind_params = {
                'sigma': float(sigma),
                'members': amp.get('ensemble_members',
                                   ('bm3d', 'nlm', 'bilateral')),
            }


        if snap['final_deconv'] == 'auto':
            self.final_deconv = 'ringing_removal'

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'lambda_val': float(self.lambda_val),
            'tau': float(self.tau),
            'final_alpha': float(self.final_alpha),
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={noise_type}): "
                  f"λ={self.lambda_val:.1f}, "
                  f"τ={self.tau:.6f}, α={self.final_alpha:.5f}, "
                  f"pre={self.preprocess}, blind={self.blind_denoise}, "
                  f"pre_nb={self.pre_nonblind}, "
                  f"final={self.final_deconv}")
        return info


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
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
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


                if key in self._defaults_snapshot:
                    self._defaults_snapshot[key] = (
                        float(value) if key in ('lambda_val', 'tau',
                                                 'final_alpha')
                        else value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
