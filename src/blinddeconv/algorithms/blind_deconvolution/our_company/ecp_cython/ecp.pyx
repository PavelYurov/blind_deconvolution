"""
ecp.py

Blind Image Deblurring via Extreme Channels Prior (ECP).

Reference:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao: "Image Deblurring via
    Extreme Channels Prior", CVPR 2017.

ECP extends the Dark Channel Prior (Pan et al., CVPR 2016) with a
symmetric Bright Channel term and solves the resulting energy with HQS.

Pipeline (mirrors MATLAB demo_deblurring.m from ECP-main):
    1. Normalise input to float64 [0, 1].
    2. Convert to grayscale for kernel estimation.
    3. (Optional) noise pipeline — disabled by default; behaviour is
       bit-for-bit identical to the original ECP demo when every noise
       toggle is left at ``'none'`` and ``auto_mode='off'``.
    4. Multi-scale ECP blind kernel estimation (blind_deconv).
    5. Non-blind restoration on the full (colour) image
       via ``ringing_artifacts_removal`` (default) or ``adaptive_lp``.
    6. Return restored image (int16, [0, 255]) and kernel.

The optional noise pipeline mirrors the LIP module so that both methods
expose the same interface for noise-aware experiments.  The ECP *core*
parameters (``lambda_dark``, ``lambda_grad``, ``xk_iter``, ``k_thresh``,
``kernel_size``) are kept **paper-faithful** in every regime — the
robust orchestrator only adapts denoisers and the non-blind step,
because the ECP paper does not motivate σ-driven scaling of the prior
weights.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict, Callable, Optional

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

from .solvers import blind_deconv, ringing_artifacts_removal
from .impulse_noise_estimation import (
    detect_impulse_noise,
    adaptive_median_filter,
    remove_impulse_noise,
)


class ECP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Extreme Channels Prior (dark + bright).

    Pipeline (mirrors MATLAB ``demo_deblurring.m`` from ECP-main):
        1. Normalise input to float64 [0, 1].
        2. Convert to grayscale for kernel estimation.
        3. Noise pipeline (impulse → estimation → orchestrator →
           ScreeNOT/ACT → preprocess → PSD filtering).
        4. Multi-scale ECP blind kernel estimation (blind_deconv) with
           optional ``blind_denoise`` before each kernel step.
        5. Pre-nonblind denoising (optional, on the full colour image).
        6. Non-blind restoration: ``ringing_removal`` (default) or
           ``adaptive_lp``.
        7. Return restored image (int16, [0, 255]) and kernel.

    With every noise toggle left at ``'none'`` and ``auto_mode='off'``
    the pipeline is bit-for-bit identical to the original ECP demo.

    Parameters
    ----------
    kernel_size  : spatial support of the unknown PSF (default 35).
    lambda_dark  : dark-channel prior weight λ_D (default 4e-3).
    lambda_grad  : gradient prior weight λ_g (default 4e-3).
    xk_iter      : number of inner kernel-estimation iterations per
                   pyramid level (default 5).
    gamma_correct : gamma exponent applied to the input (default 1.0,
                   i.e. no correction).
    k_thresh     : kernel-mass threshold used inside the multi-scale
                   solver (default 20.0).
    lambda_tv    : TV weight for the ``ringing_removal`` non-blind step
                   (default 1e-3).
    lambda_l0    : L0-gradient weight for ``ringing_removal``
                   (default 5e-4).
    weight_ring  : ringing-suppression weight for ``ringing_removal``
                   (default 1.0).
    final_deconv : ``'ringing_removal'`` (default, paper) or
                   ``'adaptive_lp'`` (Lp-norm prior, signal-aware).
    verbose      : print orchestrator decisions when ``auto_mode='robust'``.
    progress_callback : optional ``callable(event_dict)`` invoked by the
                   multi-scale solver — useful for plotting kernel
                   evolution.  Events: ``scale_start``, ``iter`` (with
                   kernel snapshot, ``kernel_diff``, λ values), ``scale_end``.

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
        'pca'  — Pyatykh et al. (TIP 2013) PCA + VST + kurtosis
                 (preferred, also returns ``noise_type``).
        'chen' — Chen et al. (ICCV 2015) wavelet-based σ estimation
                 (kept as fallback).
        'none' — skip.  Auto-promoted to 'pca' when ``auto_mode='robust'``.
    act_preprocess : str
        'auto' — Adaptive Curvelet Thresholding before blind step
                 (preferred sanitation in this module).
        'none' — skip.  Mutually exclusive with ``screenot_preprocess``.
    act_params : dict or None
        Keys: 'noise_var', 'threshold_setting' ('s'/'h').
    screenot_preprocess : str
        'auto' — ScreeNOT SVD denoising before blind step.
        'none' — skip.  Mutually exclusive with ``act_preprocess``.
        Kept as an optional fallback only.
    screenot_params : dict or None
        Keys: 'k', 'strategy', 'mode', 'patch_size', 'stride'.
    preprocess : str
        Spatial denoiser applied to the grayscale kernel-estimation
        image before the pyramid: 'tv', 'nlm', 'bilateral', 'guided',
        'bm3d', 'act', 'ensemble', or 'none'.
    preprocess_params : dict or None
        Denoiser-specific parameters.  Pass ``{'use_vst': True}`` to
        wrap the denoiser in a Generalized Anscombe VST (requires PCA
        noise estimation with ``a, b``).
    noise_preprocess : str
        PSD-based noise filter: 'auto', 'notch', 'bandstop', or 'none'.
    noise_preprocess_params : dict or None
        Keys: 'pch_size', 'n_smooth', 'peak_threshold', 'notch_radius',
              'freq_low', 'freq_high', 'order'.
    blind_denoise : str
        Denoiser applied to S before each kernel update inside the
        blind loop: same options as ``preprocess``.
    blind_denoise_params : dict or None
        Denoiser-specific parameters.
    pre_nonblind : str
        Denoiser applied to the (full colour) blurry image before the
        non-blind step.  Same options as ``preprocess``.  Colour
        channels are denoised independently with the same parameters.
    pre_nonblind_params : dict or None
        Denoiser-specific parameters.
    nb_params : dict or None
        Override dict for the non-blind solver.
        ``ringing_removal`` keys: 'lambda_tv', 'lambda_l0', 'weight_ring'.
        ``adaptive_lp``     keys: 'alpha', 'two_stage'.
    auto_mode : str
        'off' (default) or 'robust'.  In 'robust' mode the orchestrator
        estimates σ (PCA) and soft-blends the non-blind weights and
        denoiser choices between the user defaults and noise-robust
        values.  ECP core weights (λ_D, λ_g, kernel_size, xk_iter) are
        **never** modified — the paper has no σ rule for them.
    auto_mode_params : dict or None
        Orchestrator knobs.  Keys:
        'sigma_clean' (default 0.005) — σ below which clean defaults are kept.
        'sigma_heavy' (default 0.05)  — σ above which the heavy regime is full-strength.
        'force_heavy_sigma' (default 0.01) — for poisson-like noise, force heavy
                                              regime above this σ.
        'k_lambda_tv'    (default 0.05),
        'k_lambda_l0'    (default 0.01),
        'k_weight_ring'  (default 1.0)  — σ-coupling slopes for the
                                          ringing-removal weights.
        'k_alpha'        (default 0.1)  — slope used for adaptive_lp's α.
        'ensemble_members' (default ('bm3d','nlm','bilateral')).
    """

    def __init__(
        self,
        kernel_size: int = 35,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        final_deconv: str = 'ringing_removal',
        verbose: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        # ── Noise pipeline (all disabled by default) ────────────────
        impulse_preprocess: str = 'none',
        impulse_params: Optional[dict] = None,
        noise_estimation: str = 'none',
        screenot_preprocess: str = 'none',
        screenot_params: Optional[dict] = None,
        act_preprocess: str = 'none',
        act_params: Optional[dict] = None,
        preprocess: str = 'none',
        preprocess_params: Optional[dict] = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: Optional[dict] = None,
        blind_denoise: str = 'none',
        blind_denoise_params: Optional[dict] = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: Optional[dict] = None,
        nb_params: Optional[dict] = None,
        auto_mode: str = 'off',
        auto_mode_params: Optional[dict] = None,
    ):
        super().__init__(name='ECP-BD')

        # Core (paper-faithful)
        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.final_deconv = (final_deconv or 'ringing_removal').lower()
        self.verbose = verbose
        self.progress_callback = progress_callback

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
        self.nb_params = nb_params
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        # Snapshot of defaults used by the robust orchestrator so that
        # soft-blending always starts from values the user supplied,
        # not from values overwritten on a previous process() call.
        self._defaults_snapshot = {
            'lambda_tv': float(lambda_tv),
            'lambda_l0': float(lambda_l0),
            'weight_ring': float(weight_ring),
            'final_deconv': self.final_deconv,
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'nb_params': nb_params,
        }

        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'iterations': [],
            'scale_kernels': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # Reset run-local history (don't leak across process() calls).
        self.history = {
            'kernel_diff': [],
            'iterations': [],
            'scale_kernels': [],
        }

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # ── 2. Grayscale for kernel estimation ──────────────────────────
        # MATLAB: yg = im2double(rgb2gray(y))
        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        # The non-blind step operates on the full (colour) input.  All
        # pre-processors that clean ``yg`` for kernel estimation are
        # mirrored per-channel onto ``y_full`` so the non-blind solver
        # sees the same denoised stream — matching LIP/PMP semantics
        # (where the single denoised stream IS the non-blind input).
        # After each yg-modifying hook ``yg = luma(y_full)`` is re-derived
        # so the two streams stay consistent.
        y_full = y

        def _luma(im):
            if im.ndim == 2:
                return im
            if im.shape[2] == 3:
                return (0.2989 * im[:, :, 0]
                        + 0.5870 * im[:, :, 1]
                        + 0.1140 * im[:, :, 2])
            return im[:, :, 0]

        def _per_channel(im, fn):
            """Apply a 2D → 2D function ``fn`` per channel of ``im``."""
            if im.ndim == 2:
                return fn(im)
            out = np.empty_like(im)
            for _c in range(im.shape[2]):
                out[:, :, _c] = fn(im[:, :, _c])
            return out

        # ── 3a. Impulse noise detection & removal ──────────────────────
        # Run the colour-aware ``remove_impulse_noise`` on the full image
        # (it detects on grayscale, then per-channel re-detects single-
        # channel hits invisible on the luma map) and re-derive yg from
        # the cleaned y_full so kernel estimation and the non-blind solver
        # see a consistent input.
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            res = remove_impulse_noise(
                y_full,
                density_threshold=ip.get('density_threshold', 0.0005),
                max_window=ip.get('max_window', 7),
                outlier_window=ip.get('outlier_window', 5),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
            )
            impulse_info = {
                'has_impulse': res['has_impulse'],
                'density': res['density'],
                'applied': res['applied'],
            }
            if res['applied']:
                y_full = res['image']
                if y_full.ndim == 3 and y_full.shape[2] == 3:
                    yg = (0.2989 * y_full[:, :, 0]
                          + 0.5870 * y_full[:, :, 1]
                          + 0.1140 * y_full[:, :, 2])
                elif y_full.ndim == 3:
                    yg = y_full[:, :, 0]
                else:
                    yg = y_full

        # ── 3b. Noise estimation ────────────────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
        elif self.auto_mode == 'robust':
            # Orchestrator needs σ — switch on PCA estimator if user
            # forgot to (preferred path: PCA over Chen).
            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(yg)

        # ── 3c. Robust orchestrator ─────────────────────────────────────
        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)

        # ── 3d. ScreeNOT SVD denoising (optional, fallback) ─────────────
        # Per-channel onto y_full so non-blind also sees denoised image
        # (LIP/PMP-style propagation). yg = luma(y_full).
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            sn_kw = dict(
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if y_full.ndim == 2:
                y_full, screenot_info = screenot_denoise(y_full, **sn_kw)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = screenot_denoise(
                        y_full[:, :, _c], **sn_kw)
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        screenot_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        # ── 3e. ACT curvelet denoising (preferred sanitation) ──────────
        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from .act_denoise import act_denoise as _act
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            act_kw = dict(
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if y_full.ndim == 2:
                y_full, act_info = _act(y_full, **act_kw)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = _act(y_full[:, :, _c], **act_kw)
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        act_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        # ── 3f. Pre-pyramid spatial denoising ──────────────────────────
        if self.preprocess not in (None, 'none'):
            y_full = _per_channel(
                y_full,
                lambda im: self._apply_denoise(
                    im, self.preprocess, self.preprocess_params, noise_info),
            )
            yg = _luma(y_full)

        # ── 3g. PSD-based noise preprocessing ──────────────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            if y_full.ndim == 2:
                y_full, psd_info = self._apply_noise_preprocess(y_full)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = self._apply_noise_preprocess(
                        y_full[:, :, _c])
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        psd_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        # ── 4. Build optional blind-loop denoiser + telemetry ──────────
        blind_denoise_fn = None
        if self.blind_denoise not in (None, 'none'):
            def blind_denoise_fn(s_arr, _info=noise_info):
                return self._apply_blind_denoise(s_arr, _info)

        progress_proxy = self._make_progress_proxy()

        # ── 5. Blind kernel estimation (ECP multi-scale) ───────────────
        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_dark, self.lambda_grad, opts,
            blind_denoise_fn=blind_denoise_fn,
            progress_callback=progress_proxy,
        )

        # ── 6. Pre-nonblind denoising (operates on full image) ─────────
        if self.pre_nonblind not in (None, 'none'):
            y_full = self._apply_pre_nonblind(y_full, noise_info)

        # ── 7. Non-blind restoration ───────────────────────────────────
        Latent = self._final_deconv(y_full, kernel, noise_info)
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 8. Output / telemetry ──────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'final_deconv': self.final_deconv,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': ({k_: v for k_, v in (impulse_info or {}).items()
                              if k_ != 'impulse_mask'}
                             if impulse_info else None),
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'preprocess': self.preprocess,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': ({k_: v for k_, v in (psd_info or {}).items()
                          if k_ != 'psd_2d'} if psd_info else None),
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Non-blind solver dispatch ────────────────────────────────────────
    def _final_deconv(self, y_full, kernel, noise_info):
        nbp = self.nb_params or {}
        method = self.final_deconv

        # 'auto' may survive into _final_deconv when the user sets
        # final_deconv='auto' without auto_mode='robust' (orchestrator
        # is the only place that resolves it).  Fall back to the paper
        # default rather than raise.
        if method == 'auto':
            method = 'ringing_removal'
            self.final_deconv = method

        if method == 'ringing_removal':
            return ringing_artifacts_removal(
                y_full, kernel,
                nbp.get('lambda_tv', self.lambda_tv),
                nbp.get('lambda_l0', self.lambda_l0),
                nbp.get('weight_ring', self.weight_ring),
            )

        if method == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            alpha = nbp.get('alpha', 0.8)
            two_stage = nbp.get('two_stage', True)

            if y_full.ndim == 2:
                return adaptive_lp_deconv(
                    y_full, kernel, alpha=alpha,
                    sigma_n=sigma_n, two_stage=two_stage)
            chans = []
            for c in range(y_full.shape[2]):
                chans.append(adaptive_lp_deconv(
                    y_full[:, :, c], kernel, alpha=alpha,
                    sigma_n=sigma_n, two_stage=two_stage))
            return np.stack(chans, axis=2)

        raise ValueError(
            f"Unknown final_deconv='{self.final_deconv}'. "
            "Choose 'ringing_removal' or 'adaptive_lp'.")

    # ── Progress callback proxy ─────────────────────────────────────────
    def _make_progress_proxy(self):
        """Wrap the user-supplied callback so we also record telemetry
        in ``self.history``.  Always returns a callable so the multi-scale
        solver records kernel snapshots even when no user callback is
        supplied."""
        user_cb = self.progress_callback
        history = self.history

        def proxy(event):
            try:
                if event.get('event') == 'iter':
                    history['kernel_diff'].append(event.get('kernel_diff'))
                    history['iterations'].append({
                        k: v for k, v in event.items() if k != 'kernel'
                    })
                elif event.get('event') == 'scale_end':
                    history['scale_kernels'].append({
                        'scale': event.get('scale'),
                        'kernel': event.get('kernel'),
                    })
            except Exception:
                pass
            if user_cb is not None:
                try:
                    user_cb(event)
                except Exception:
                    pass

        return proxy

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

    # ── Universal denoiser dispatch ─────────────────────────────────────
    def _apply_denoise(self, img, method, params, noise_info):
        """Apply a spatial denoiser to a single-channel image [0, 1].

        Mirrors ``LIP_BD._apply_denoise`` so robustness experiments can
        share the same denoiser configuration across both methods.
        Supports an optional ``use_vst`` flag in ``params``.
        """
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
                outs.append(self._apply_denoise(
                    img, m, sub_params, noise_info))
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

    # ── Generalized Anscombe VST wrapper ────────────────────────────────
    def _apply_denoise_vst(self, img, method, params, noise_info):
        """Generalized Anscombe VST → denoise → inverse VST.

        For Poisson-Gaussian noise  Var[y] = a·y + b  (pyatykh scale,
        [0,255]) the transform with  A = a/255, B = b/255²  yields
        approximately unit variance.  We rescale into [0,1] so generic
        denoisers behave, then invert.
        """
        a = float(noise_info.get('a', 0.0) or 0.0) if noise_info else 0.0
        b = float(noise_info.get('b', 0.0) or 0.0) if noise_info else 0.0
        A = a / 255.0
        B = b / (255.0 ** 2)

        if A < 1e-8:
            sub = dict(params)
            sub.setdefault('sigma',
                           noise_info.get('sigma_norm', None)
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

    # ── Noise estimation ────────────────────────────────────────────────
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

    # ── Robust orchestrator ─────────────────────────────────────────────
    def _orchestrate_robust(self, noise_info):
        """Soft-weighted auto configuration of the noise pipeline.

        Same overall logic as ``LIP_BD._orchestrate_robust`` with two
        deliberate ECP-specific deviations:

          1. The ECP *core* (λ_D, λ_g, kernel_size, xk_iter) is **never**
             touched.  The Yan et al. paper does not motivate σ-driven
             adaptation of these weights.
          2. λ_tv / λ_l0 / weight_ring (ringing_removal weights) **are**
             σ-blended, because they directly absorb noise amplification
             from the inverse filter.

        Always starts by resetting mutable fields from the __init__
        snapshot so repeated process() calls are deterministic.
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        # 1) Reset from snapshot — avoid sticky state between calls.
        self.lambda_tv = snap['lambda_tv']
        self.lambda_l0 = snap['lambda_l0']
        self.weight_ring = snap['weight_ring']
        self.final_deconv = snap['final_deconv']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.blind_denoise = snap['blind_denoise']
        self.blind_denoise_params = snap['blind_denoise_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.nb_params = snap['nb_params']

        # 2) Read σ.  If estimator failed or returned 0 — treat as clean.
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

        # 3) Clean branch — DO NOT alter denoisers or parameters.
        if sigma <= sigma_clean and not force_heavy:
            # Resolve final_deconv='auto' even on the clean path so the
            # non-blind step has a concrete method.  ECP's clean default
            # is the paper's ringing-aware solver.
            if snap['final_deconv'] == 'auto':
                self.final_deconv = 'ringing_removal'
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"defaults kept, final_deconv={self.final_deconv}")
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'final_deconv': self.final_deconv,
                'preprocess': self.preprocess,
                'blind_denoise': self.blind_denoise,
                'pre_nonblind': self.pre_nonblind,
                'lambda_tv': float(self.lambda_tv),
                'lambda_l0': float(self.lambda_l0),
                'weight_ring': float(self.weight_ring),
            }

        # 4) Heavy branch — smooth weight between σ_clean and σ_heavy.
        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / (sigma_heavy - sigma_clean))
        regime = 'heavy' if w > 0.95 else 'medium'

        noise_type = (noise_info or {}).get('noise_type', 'gaussian')
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        # 4a) σ-driven non-blind weights (ECP-specific).
        k_lambda_tv = float(amp.get('k_lambda_tv', 0.05))
        k_lambda_l0 = float(amp.get('k_lambda_l0', 0.01))
        k_weight_ring = float(amp.get('k_weight_ring', 1.0))
        k_alpha = float(amp.get('k_alpha', 0.1))

        lam_tv_noisy = max(snap['lambda_tv'], k_lambda_tv * sigma)
        lam_l0_noisy = max(snap['lambda_l0'], k_lambda_l0 * sigma)
        wring_noisy = min(2.0, snap['weight_ring'] + k_weight_ring * sigma)

        self.lambda_tv = (1.0 - w) * snap['lambda_tv'] + w * lam_tv_noisy
        self.lambda_l0 = (1.0 - w) * snap['lambda_l0'] + w * lam_l0_noisy
        self.weight_ring = (1.0 - w) * snap['weight_ring'] + w * wring_noisy

        # 4b) Blind-loop denoiser — cheap edge-preserving bilateral.
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

        # 4c) Pre-pyramid denoiser.
        # Poisson-like → ACT (locally adaptive curvelet).
        # Pure Gaussian → bilateral at moderate w, BM3D at heavy.
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

        # 4d) Pre-nonblind denoiser.
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

        # 4e) Final deconv routing for snapshot=='auto'.
        if snap['final_deconv'] == 'auto':
            if poisson_like and w >= 0.5:
                self.final_deconv = 'adaptive_lp'
                if self.nb_params is None:
                    self.nb_params = {
                        'alpha': max(0.5, 0.8 - k_alpha * sigma),
                        'two_stage': True,
                    }
            else:
                self.final_deconv = 'ringing_removal'

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'lambda_tv': float(self.lambda_tv),
            'lambda_l0': float(self.lambda_l0),
            'weight_ring': float(self.weight_ring),
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={noise_type}): "
                  f"λ_tv={self.lambda_tv:.5f}, λ_l0={self.lambda_l0:.5f}, "
                  f"w_ring={self.weight_ring:.3f}, "
                  f"pre={self.preprocess}, blind={self.blind_denoise}, "
                  f"pre_nb={self.pre_nonblind}, "
                  f"final={self.final_deconv}")
        return info

    # ── PSD-based noise preprocessing ───────────────────────────────────
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

    # ── Blind-loop denoiser (S before kernel step) ──────────────────────
    def _apply_blind_denoise(self, x, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    # ── Pre-nonblind denoiser ──────────────────────────────────────────
    def _apply_pre_nonblind(self, img, noise_info):
        """Apply the pre-nonblind denoiser.  Handles colour by denoising
        each channel independently with the same parameters."""
        if img.ndim == 2:
            return self._apply_denoise(
                img, self.pre_nonblind, self.pre_nonblind_params, noise_info)
        chans = []
        for c in range(img.shape[2]):
            chans.append(self._apply_denoise(
                img[:, :, c], self.pre_nonblind,
                self.pre_nonblind_params, noise_info))
        return np.stack(chans, axis=2)

    # ── Interface methods ────────────────────────────────────────────────
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
            ('final_deconv', self.final_deconv),
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
            ('nb_params', self.nb_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Keep the orchestrator's snapshot in sync with parameters
                # the user updates after construction — otherwise robust
                # mode would keep blending toward stale values.
                if key in self._defaults_snapshot:
                    self._defaults_snapshot[key] = (
                        float(value)
                        if key in ('lambda_tv', 'lambda_l0', 'weight_ring')
                        else value
                    )

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
