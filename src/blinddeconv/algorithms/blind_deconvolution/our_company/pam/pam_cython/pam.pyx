"""
pam.py

Blind Image Deconvolution via Total Variation (Perrone-Favaro CVPR 2014)
with optional noise-aware preprocessing, LIP-style robust orchestrator
and post-CTF non-blind restoration.

Reference (paper core):
    D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", CVPR 2014.

Pipeline (default = paper-pure if all noise knobs left at 'none'):
    1.  Normalise to float64 [0, 1] and ensure grayscale.
    2.  Impulse noise detection & removal (optional).
    3.  Noise sigma estimation (optional, auto-promoted in robust mode).
    4.  Auto-params from sigma (user-driven, optional): lam = k_lam * sigma.
    5.  LIP-style robust orchestrator (optional).
    6.  ScreeNOT SVD denoising (optional).
    7.  ACT curvelet denoising (optional, mutually exclusive with ScreeNOT).
    8.  Spatial pre-blind denoising (optional).
    9.  PSD-based noise filtering (optional).
    10. Histogram equalization (optional).
    11. Coarse-to-fine blind deconvolution (deblur, paper core).
    12. Optional post-CTF non-blind restoration.
    13. Crop padding, resize back to original (H, W), clip to int16 [0, 255].
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# -- Framework base class import (DO NOT MODIFY) ---------------------------
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
# --------------------------------------------------------------------------

from .solvers import deblur, dec


class PAM_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using Total Variation (Perrone & Favaro, CVPR 2014).

    Operates on grayscale images.  When `auto_mode='off'` and all noise
    pipeline knobs are at 'none', the algorithm is identical to the
    original paper port (`our_company/pam/pam`).

    Paper-core parameters (do NOT auto-tune)
    ----------------------------------------
    kernel_shape  : (int, int) PSF support, both values odd.  Default (25, 25).
    lam           : TV regularisation weight.
                    Typical: 3e-4 .. 6e-4.  Noisy: 1e-3 .. 3e-3.  Default 3e-4.
    iters         : iterations per blind/non-blind call per pyramid scale.
                    Default 1000.
    gamma_correct : bool, gamma correction before kernel estimation.
                    Default False.
    gamma         : gamma exponent (used iff gamma_correct).  Default 1.0.

    Noise pipeline (orthogonal to orchestrator, never auto-tuned)
    -------------------------------------------------------------
    impulse_preprocess  : 'none' | 'auto'.  Default 'none'.
    noise_estimation    : 'none' | 'pca' | 'chen'.  Default 'none'.
                          Auto-promoted to 'pca' when auto_mode='robust'.
    auto_params         : None or dict of {k_lam, lam_min, lam_max}.  When set,
                          lam is rescaled as
                              lam = clip(k_lam * sigma, lam_min, lam_max).
                          Defaults if dict is given but keys missing:
                              k_lam=0.05, lam_min=5e-4, lam_max=5e-3.
                          When None (default), lam stays as user-provided.
    screenot_preprocess : 'none' | 'auto'.  Default 'none'.
    noise_preprocess    : 'none' | 'auto' | 'notch' | 'bandstop'.  Default 'none'.
    histogram_eq        : 'none' | 'clahe' | 'global'.  Default 'none'.

    Orchestrator-managed groups (controlled by auto_mode)
    -----------------------------------------------------
    preprocess        : 'none' | 'tv' | 'nlm' | 'bilateral' | 'guided' | 'bm3d'
                        | 'act'.  Default 'none'.
    act_preprocess    : 'none' | 'auto'.  Default 'none'.
    pre_nonblind      : same options as preprocess; applied to the input
                        before the post-CTF non-blind step.  Default 'none'.
                        Effective only when final_nb != 'none'.
    final_nb          : 'none' | 'dec' | 'ringing_removal' | 'adaptive_lp'
                        | 'wiener' | 'tikhonov'.  Default 'none' (paper-pure).
                        - 'none'             : output the coarse_to_fine result.
                        - 'dec'              : run paper PF dec() with the
                                               estimated kernel on a (possibly
                                               denoised) input.
                        - 'ringing_removal'  : Pan et al. TV+L0+bilateral diff.
                        - 'adaptive_lp'      : Wang et al. (non_blind.adaptive_lp_deconv).
                        - 'wiener', 'tikhonov': simple FFT-based fallbacks.
    nb_params         : dict of method-specific params (see _run_final_nb).

    LIP-style robust orchestrator
    -----------------------------
    auto_mode         : 'off' | 'robust'.  Default 'off'.
    auto_mode_params  : dict of orchestrator thresholds.  Defaults:
        sigma_clean             = 0.005
        sigma_heavy             = 0.05
        force_heavy_sigma       = 0.01    (poisson-like only)
        prefer_act_for_gaussian = False
    """

    def __init__(
        self,
        # -- Paper core (untouched by orchestrator) --
        kernel_shape: tuple = (25, 25),
        lam: float = 3e-4,
        iters: int = 1000,
        gamma_correct: bool = False,
        gamma: float = 1.0,
        # -- Orthogonal noise pipeline --
        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        auto_params: dict = None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        histogram_eq: str = 'none',
        histogram_eq_params: dict = None,
        # -- Orchestrator-managed --
        preprocess: str = 'none',
        preprocess_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        final_nb: str = 'none',
        nb_params: dict = None,
        # -- LIP-style orchestrator switch --
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
        visualize: bool = False,
    ):
        super().__init__(name='PAM-BD')

        # Paper core
        self.kernel_shape = tuple(kernel_shape)
        self.lam = lam
        self.iters = iters
        self.gamma_correct = gamma_correct
        self.gamma = gamma

        # Orthogonal pipeline
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.histogram_eq = histogram_eq
        self.histogram_eq_params = histogram_eq_params

        # Orchestrator-managed
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.final_nb = (final_nb or 'none').lower()
        self.nb_params = nb_params

        # LIP-style
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        self.visualize = visualize

        # Snapshot for orchestrator (used to restore on clean branch and
        # to compute the (1-w)*default + w*noisy lam blend on heavy branch).
        self._defaults_snapshot = {
            'preprocess':          preprocess,
            'preprocess_params':   preprocess_params,
            'act_preprocess':      act_preprocess,
            'act_params':          act_params,
            'pre_nonblind':        pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'final_nb':            self.final_nb,
            'nb_params':           nb_params,
            'lam':                 lam,
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ============================================================
    # Main entry point
    # ============================================================
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # -- 1. Normalise & coerce to grayscale float64 [0, 1] -----------
        f = image.astype(np.float64)
        if f.max() > 1.0:
            f = f / 255.0
        if f.ndim == 3:
            if f.shape[2] == 1:
                f = f[:, :, 0]
            else:
                f = (0.2989 * f[:, :, 0]
                     + 0.5870 * f[:, :, 1]
                     + 0.1140 * f[:, :, 2])
        # Keep the raw input for post-CTF NB (where pre_nonblind may be
        # different from the pre-blind preprocess chain).
        f_raw = f.copy()
        orig_H, orig_W = f.shape[:2]

        # -- 2. Impulse noise detection & removal ------------------------
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            from .impulse_noise_estimation import (
                detect_impulse_noise, adaptive_median_filter,
            )
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                f,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
                outlier_window=ip.get('outlier_window', 5),
            )
            if impulse_info['has_impulse']:
                if self.visualize:
                    print(f"[PAM-BD] Impulse noise detected "
                          f"(density={impulse_info['density']:.4f}), "
                          f"applying adaptive median filter")
                f = adaptive_median_filter(
                    f, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))
                f_raw = f.copy()

        # -- 3. Noise estimation -----------------------------------------
        if self.auto_mode == 'robust' and self.noise_estimation == 'none':
            self.noise_estimation = 'pca'
            if self.visualize:
                print("[PAM-BD] auto_mode='robust' -> "
                      "forcing noise_estimation='pca'")
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(f)
            if self.visualize and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[PAM-BD] Noise estimation "
                      f"({self.noise_estimation}): "
                      f"sigma={sigma:.5f}  (sigma_255={sigma * 255:.2f})")

        # -- 4. User auto_params: sigma -> lam ---------------------------
        if self.auto_params is not None and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lam = float(ap.get('k_lam', 0.05))
                lam_min = float(ap.get('lam_min', 5e-4))
                lam_max = float(ap.get('lam_max', 5e-3))
                self.lam = float(np.clip(k_lam * sigma_n, lam_min, lam_max))
                if self.visualize:
                    print(f"[PAM-BD] auto_params(sigma={sigma_n:.5f}): "
                          f"lam={self.lam:.5e}")

        # -- 5. LIP-style robust orchestrator ----------------------------
        orchestrator_info = self._orchestrate_robust(noise_info)

        # -- 6. ScreeNOT SVD denoising -----------------------------------
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            if self.act_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
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
            if self.visualize:
                print(f"[PAM-BD] ScreeNOT applied "
                      f"(rank={screenot_info.get('rank', '?')})")

        # -- 7. ACT curvelet denoising -----------------------------------
        act_info = None
        if self.act_preprocess == 'auto':
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
            if self.visualize:
                print("[PAM-BD] ACT curvelet denoising applied")

        # -- 8. Spatial pre-blind denoising ------------------------------
        if self.preprocess not in (None, 'none'):
            f = self._apply_denoise(f, self.preprocess,
                                    self.preprocess_params, noise_info)
            if self.visualize:
                print(f"[PAM-BD] Pre-blind denoise: {self.preprocess}")

        # -- 9. PSD-based noise preprocessing ----------------------------
        psd_info = None
        if self.noise_preprocess != 'none':
            f, psd_info = self._apply_noise_preprocess(f)
            if self.visualize:
                print(f"[PAM-BD] PSD noise preprocess: "
                      f"{self.noise_preprocess}")

        # -- 10. Histogram equalization ----------------------------------
        if self.histogram_eq not in (None, 'none'):
            f = self._apply_histogram_eq(f)
            if self.visualize:
                print(f"[PAM-BD] Histogram equalization: {self.histogram_eq}")

        # -- 11. Coarse-to-fine blind deconvolution (paper core) ---------
        MK, NK = self.kernel_shape
        u, kernel = deblur(
            f,
            MK, NK,
            lam=self.lam,
            iters=self.iters,
            gamma_correct=self.gamma_correct,
            gamma=self.gamma,
            visualize=self.visualize,
        )

        # -- 12. Crop padding back to original-image domain --------------
        # deblur() returns u of size (M_odd + MK - 1, N_odd + NK - 1)
        pad_h = MK // 2
        pad_w = NK // 2
        u = u[pad_h:u.shape[0] - pad_h, pad_w:u.shape[1] - pad_w]

        # -- 13. Optional post-CTF non-blind restoration -----------------
        if self.final_nb not in (None, 'none'):
            # pre_nonblind is applied to the RAW input (not the pre-blind
            # denoised one) so users can decouple the two stages.
            f_pre_nb = f_raw
            if self.pre_nonblind not in (None, 'none'):
                f_pre_nb = self._apply_denoise(
                    f_pre_nb, self.pre_nonblind,
                    self.pre_nonblind_params, noise_info)
                if self.visualize:
                    print(f"[PAM-BD] Pre-NB denoise: {self.pre_nonblind}")
            u = self._run_final_nb(f_pre_nb, kernel, noise_info)
            if self.visualize:
                print(f"[PAM-BD] Final NB: {self.final_nb}")

        # -- 14. Resize back if even->odd cropping happened in deblur ----
        if u.shape[0] != orig_H or u.shape[1] != orig_W:
            from .utils import imresize
            u = imresize(u, (orig_H, orig_W), method='bicubic')

        # -- 15. Output --------------------------------------------------
        u = np.clip(u, 0.0, 1.0)
        self.hyperparams = {
            'kernel_shape': self.kernel_shape,
            'lam': self.lam,
            'iters': self.iters,
            'gamma_correct': self.gamma_correct,
            'gamma': self.gamma,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': {
                k_: v for k_, v in (impulse_info or {}).items()
                if k_ != 'impulse_mask'
            } if impulse_info else None,
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'auto_params': self.auto_params,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'preprocess': self.preprocess,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': {
                k_: v for k_, v in (psd_info or {}).items()
                if k_ != 'psd_2d'
            } if psd_info else None,
            'histogram_eq': self.histogram_eq,
            'pre_nonblind': self.pre_nonblind,
            'final_nb': self.final_nb,
            'nb_params': self.nb_params,
            'auto_mode': self.auto_mode,
            'auto_mode_params': self.auto_mode_params,
            'orchestrator_info': orchestrator_info,
            'time': time.time() - start_time,
        }
        x_final = u * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ============================================================
    # LIP-style robust orchestrator (schema A)
    # ============================================================
    def _orchestrate_robust(self, noise_info):
        """Adjust orchestrator-managed groups based on estimated noise.

        Manages: preprocess, act_preprocess, pre_nonblind, final_nb, nb_params,
        and the paper-core lam (via blend, only if user's auto_params is
        not active so we don't double-scale).

        Untouched: kernel_shape, iters, gamma, gamma_correct (paper core);
        impulse_preprocess, screenot_preprocess, noise_preprocess,
        histogram_eq (orthogonal user knobs).
        """
        info = {'triggered': False, 'mode': self.auto_mode}

        if self.auto_mode != 'robust':
            return info

        snap = self._defaults_snapshot
        p = self.auto_mode_params or {}
        sigma_clean = float(p.get('sigma_clean', 0.005))
        sigma_heavy = float(p.get('sigma_heavy', 0.05))
        force_heavy_sigma = float(p.get('force_heavy_sigma', 0.01))
        prefer_act = bool(p.get('prefer_act_for_gaussian', False))

        sigma = None
        ntype = 'unknown'
        if noise_info is not None:
            sigma = noise_info.get('sigma_norm', None)
            ntype = noise_info.get('noise_type', 'unknown') or 'unknown'
        if sigma is None:
            sigma = 0.0

        poisson_like = ntype in ('poisson', 'poisson_gaussian', 'unknown')
        heavy = (sigma > sigma_clean) or (poisson_like and sigma > force_heavy_sigma)

        info.update({
            'triggered': True,
            'sigma': float(sigma),
            'noise_type': ntype,
            'poisson_like': poisson_like,
            'sigma_clean': sigma_clean,
            'sigma_heavy': sigma_heavy,
            'force_heavy_sigma': force_heavy_sigma,
            'prefer_act_for_gaussian': prefer_act,
        })

        # -- Clean branch: restore user values, paper-pure run -----------
        if not heavy:
            info['branch'] = 'clean'
            self.preprocess          = snap['preprocess']
            self.preprocess_params   = snap['preprocess_params']
            self.act_preprocess      = snap['act_preprocess']
            self.act_params          = snap['act_params']
            self.pre_nonblind        = snap['pre_nonblind']
            self.pre_nonblind_params = snap['pre_nonblind_params']
            self.final_nb            = snap['final_nb']
            self.nb_params           = snap['nb_params']
            self.lam                 = snap['lam']
            if self.visualize:
                print(f"[PAM-BD][orchestrator] clean (sigma={sigma:.5f}) "
                      "-> paper defaults restored")
            return info

        # -- Heavy branch -----------------------------------------------
        info['branch'] = 'heavy'
        sigma_eff = max(sigma, 1e-3)

        if poisson_like or prefer_act:
            info['route'] = 'act'
            self.preprocess        = 'none'
            self.preprocess_params = None
            self.act_preprocess    = 'auto'
            self.act_params        = {'noise_var': sigma_eff ** 2}
            self.pre_nonblind      = 'act'
            self.pre_nonblind_params = {'noise_var': sigma_eff ** 2}
            self.final_nb          = 'ringing_removal'
            self.nb_params         = None
        else:
            w = (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            info['blend_weight'] = w

            if w < 0.6:
                info['route'] = 'gauss_light'
                self.preprocess = 'bilateral'
                self.preprocess_params = {
                    'sigma_color': sigma_eff * 2.0,
                    'sigma_space': 5.0,
                }
                self.act_preprocess    = 'none'
                self.act_params        = None
                self.pre_nonblind      = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma_eff}
                self.final_nb          = 'ringing_removal'
                self.nb_params         = None
            else:
                info['route'] = 'gauss_strong'
                self.preprocess = 'bm3d'
                self.preprocess_params = {'sigma': sigma_eff}
                self.act_preprocess    = 'none'
                self.act_params        = None
                self.pre_nonblind      = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma_eff * 1.5}
                self.final_nb          = 'ringing_removal'
                self.nb_params         = None

        # -- lam blend: only if user auto_params is NOT active ----------
        if self.auto_params is None:
            w = (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            lam_noisy = float(np.clip(0.05 * sigma_eff, 5e-4, 5e-3))
            self.lam = (1 - w) * snap['lam'] + w * lam_noisy
            info['lam_blend_applied'] = True
            info['lam_blend_weight'] = w
            info['lam'] = self.lam
        else:
            info['lam_blend_applied'] = False
            info['lam_blend_skipped_reason'] = 'user_auto_params_active'

        if self.visualize:
            print(f"[PAM-BD][orchestrator] heavy/{info['route']} "
                  f"sigma={sigma:.5f} type={ntype} "
                  f"preprocess={self.preprocess} "
                  f"act={self.act_preprocess} "
                  f"pre_nb={self.pre_nonblind} "
                  f"final_nb={self.final_nb} "
                  f"lam_blend={info['lam_blend_applied']}")

        return info

    # ============================================================
    # Final non-blind dispatch
    # ============================================================
    def _run_final_nb(self, blurred, kernel, noise_info):
        method = self.final_nb
        nbp = self.nb_params or {}

        if method == 'dec':
            iters = int(nbp.get('iters', self.iters))
            lam_nb = float(nbp.get('lam', self.lam))
            u = dec(blurred, kernel, lam=lam_nb,
                    iters=iters, visualize=self.visualize)
            # dec() returns padded image (M+MK-1, N+NK-1); crop back.
            MK, NK = kernel.shape
            pad_h = MK // 2
            pad_w = NK // 2
            return u[pad_h:u.shape[0] - pad_h, pad_w:u.shape[1] - pad_w]

        if method == 'ringing_removal':
            from .non_blind import ringing_removal
            return ringing_removal(
                blurred, kernel,
                lambda_tv=float(nbp.get('lambda_tv', 3e-3)),
                lambda_l0=float(nbp.get('lambda_l0', 5e-4)),
                weight_ring=float(nbp.get('weight_ring', 1.0)),
            )

        if method == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            return adaptive_lp_deconv(
                blurred, kernel,
                alpha=float(nbp.get('alpha', 0.8)),
                sigma_n=sigma_n,
                two_stage=bool(nbp.get('two_stage', True)),
            )

        if method == 'wiener':
            return self._wiener_filter(blurred, kernel,
                                       float(nbp.get('noise_snr', 0.01)))

        if method == 'tikhonov':
            return self._tikhonov_filter(blurred, kernel,
                                         float(nbp.get('alpha', 0.01)))

        raise ValueError(
            f"Unknown final_nb='{method}'. Choose from: "
            "'none', 'dec', 'ringing_removal', 'adaptive_lp', "
            "'wiener', 'tikhonov'."
        )

    # ============================================================
    # Wiener / Tikhonov fallbacks (FFT-based, single channel)
    # ============================================================
    @staticmethod
    def _wiener_filter(b, k, noise_snr):
        H, W = b.shape
        K = np.fft.fft2(k, s=(H, W))
        B = np.fft.fft2(b)
        K_conj = np.conj(K)
        return np.real(np.fft.ifft2(K_conj * B / (np.abs(K) ** 2 + noise_snr)))

    @staticmethod
    def _tikhonov_filter(b, k, alpha):
        H, W = b.shape
        K = np.fft.fft2(k, s=(H, W))
        B = np.fft.fft2(b)
        # Differential operator (Laplacian) in FFT domain.
        cy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        L = np.fft.fft2(cy, s=(H, W))
        K_conj = np.conj(K)
        return np.real(np.fft.ifft2(
            K_conj * B / (np.abs(K) ** 2 + alpha * np.abs(L) ** 2)
        ))

    # ============================================================
    # Universal denoiser dispatch (single-channel)
    # ============================================================
    def _apply_denoise(self, img, method, params, noise_info):
        if method is None or method == 'none':
            return img
        p = dict(params or {})
        sigma = noise_info.get('sigma_norm', None) if noise_info else None

        if method == 'tv':
            from skimage.restoration import denoise_tv_chambolle
            w = p.get('weight', max(0.01, sigma * 2) if sigma else 0.1)
            return denoise_tv_chambolle(img, weight=w)

        if method == 'nlm':
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

        if method == 'bilateral':
            import cv2
            d = p.get('d', 5)
            sc = p.get('sigma_color', sigma if sigma else 0.1)
            ss = p.get('sigma_space', 5.0)
            return cv2.bilateralFilter(
                img.astype(np.float32), d, float(sc), float(ss)
            ).astype(np.float64)

        if method == 'guided':
            r = p.get('radius', 4)
            eps = p.get('eps', sigma ** 2 * 4 if sigma else 0.01)
            return self._guided_filter(img, img, r, eps)

        if method == 'bm3d':
            import bm3d as bm3d_lib
            sig = p.get('sigma', sigma if sigma else 0.05)
            return bm3d_lib.bm3d(img, sigma_psd=sig)

        if method == 'act':
            from .act_denoise import act_denoise
            nv = p.get('noise_var', None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = p.get('threshold_setting', 's')
            result, _ = act_denoise(img, noise_var=nv,
                                    threshold_setting=ts)
            return result

        raise ValueError(
            f"Unknown denoiser='{method}'. Choose from: "
            "'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'act', 'none'"
        )

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

    # ============================================================
    # Noise estimation
    # ============================================================
    def _estimate_noise(self, yg):
        if self.noise_estimation == 'chen':
            from .chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        if self.noise_estimation == 'pca':
            from .pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pca'
            return result
        return None

    # ============================================================
    # PSD-based noise preprocessing (orthogonal)
    # ============================================================
    def _apply_noise_preprocess(self, yg):
        from .noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess as _npp,
        )
        npp = self.noise_preprocess_params or {}
        psd_info = analyze_noise_psd(yg)
        method = self.noise_preprocess
        if method == 'auto':
            if psd_info.get('has_periodic', False):
                method = 'notch'
            elif psd_info.get('color_label', 'white') in ('pink', 'brown'):
                method = 'bandstop'
            else:
                return yg, psd_info
        return _npp(yg, method, npp), psd_info

    # ============================================================
    # Histogram equalization (orthogonal)
    # ============================================================
    def _apply_histogram_eq(self, yg):
        method = self.histogram_eq
        hp = self.histogram_eq_params or {}
        yg_clipped = np.clip(yg, 0, 1)
        if method == 'clahe':
            from skimage.exposure import equalize_adapthist
            return equalize_adapthist(
                yg_clipped,
                clip_limit=hp.get('clip_limit', 0.01),
                nbins=hp.get('nbins', 256))
        if method == 'global':
            from skimage.exposure import equalize_hist
            return equalize_hist(yg_clipped)
        raise ValueError(
            f"Unknown histogram_eq='{method}'. "
            "Choose 'clahe', 'global', or 'none'.")

    # ============================================================
    # Interface methods
    # ============================================================
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lam', self.lam),
            ('iters', self.iters),
            ('gamma_correct', self.gamma_correct),
            ('gamma', self.gamma),
            ('impulse_preprocess', self.impulse_preprocess),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
            ('screenot_preprocess', self.screenot_preprocess),
            ('noise_preprocess', self.noise_preprocess),
            ('histogram_eq', self.histogram_eq),
            ('preprocess', self.preprocess),
            ('act_preprocess', self.act_preprocess),
            ('pre_nonblind', self.pre_nonblind),
            ('final_nb', self.final_nb),
            ('nb_params', self.nb_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
            ('visualize', self.visualize),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
