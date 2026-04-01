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
    """
    Low-Rank Blind Image Deconvolution.

    Parameters:
    kernel_size : int
        Expected maximum PSF size (must be odd, ≥ 3).
    lambda_ : float
        Base edge-regularisation weight α₀ for the image step.
        Scaled per pyramid level:
        α = λ · ``alpha_multiplier`` ^ (level − 0.5).
    sigma : float
        Low-rank regularisation flag/weight.  Set > 0 to enable,
        0 to disable.
    tau : float
        Proximal parameter for nuclear-norm thresholding (IRNN).
    delta : float
        Smoothing for the ``log det`` rank surrogate.
    kernel_beta : float
        Tikhonov regularisation weight β for the kernel CG step.
    max_iter : int
        Outer alternating-minimisation iterations per scale.
    max_irls : int
        IRLS outer iterations for the image step.
    max_cg : int
        CG inner iterations for the image step.
    max_iter_k : int
        CG iterations for the kernel step.
    max_iter_rank : int
        IRNN iterations for the low-rank step.
    iter_k_rank : int
        Inner kernel–rank alternation count per outer iteration.
    exp_a : float
        Hyper-Laplacian exponent *p* (0 < p ≤ 2; typical 0.5–0.8).
    thr_e : float
        IRLS smoothing parameter ε (avoids division by zero).
    alpha_multiplier : float
        Factor for scaling α across pyramid levels.
    threshold : float
        Kernel thresholding fraction (relative to max element).
    nb_lambda : float
        Regularisation weight for non-blind deconvolution.
    nb_alpha : float
        Hyper-Laplacian exponent for non-blind deconvolution.
    verbose : bool
        Print progress messages.

    Noise pipeline (all disabled by default — existing behaviour unchanged):
    impulse_preprocess : str — 'auto' | 'none'.  Default 'none'.
    impulse_params : dict or None — settings for impulse noise detection/removal.
        density_threshold : float — min fraction of outlier pixels to trigger
                           filtering (default 0.0005). Lower = more sensitive.
        outlier_threshold : float — pixel deviation from local median to flag
                           as impulse (default 0.08, normalised [0,1] range).
        max_window : int — maximum adaptive median window size (default 7).
                     Must be odd.  Larger = heavier smoothing.
        All keys optional; omitted keys use defaults above.
    noise_estimation : str — 'chen' | 'pca' | 'none'.  Default 'none'.
        'chen'  — Chen et al. wavelet-based σ estimator (fast, single value).
        'pca'   — Pyatykh PCA-based reconstruction (returns sigma + more).
        Result stored in noise_info dict, accessible via get_hyperparams().
    screenot_preprocess : str — 'auto' | 'none'.  Default 'none'.
    screenot_params : dict or None — settings for ScreeNOT SVD denoising.
        k          : int — number of singular values to keep (default 10).
        strategy   : str — 'i' (improved) | 'c' (classic) (default 'i').
        mode       : str — 'full' | 'patch' (default 'full').
        patch_size : int — patch side if mode='patch' (default 8).
        stride     : int — patch stride (default 3).
        All keys optional; omitted keys use defaults above.
    act_preprocess : str — 'auto' | 'none'.  Default 'none'.
                     Mutually exclusive with screenot_preprocess.
    act_params : dict or None — settings for ACT curvelet denoising.
        noise_var          : float or None — noise variance in [0,1] scale.
                             Auto-filled from σ² when noise_estimation enabled
                             and not explicitly set.
        threshold_setting  : str — 's' (soft) | 'h' (hard) (default 's').
        All keys optional; omitted keys / None auto-inferred from noise_info.
    preprocess : str — 'bm3d' | 'nlm' | 'bilateral' | 'guided' | 'tv'
                 | 'none'.  Default 'none'.
    preprocess_params : dict or None — settings for the chosen spatial denoiser.
        For 'tv':
            weight : float — TV weight (default auto ~2σ or 0.1).
        For 'nlm':
            sigma          : float — noise σ (default from noise_info).
            h              : float — filter strength (default 0.8×σ).
            patch_size     : int — default 5.
            patch_distance : int — search window (default 6).
        For 'bilateral':
            d           : int — pixel neighbourhood diameter (default 5).
            sigma_color : float — colour range σ (default from noise_info).
            sigma_space : float — spatial σ (default 5.0).
        For 'guided':
            radius : int — box filter radius (default 4).
            eps    : float — regularisation (default 4σ² or 0.01).
        For 'bm3d':
            sigma : float — noise σ (default from noise_info or 0.05).
        All keys optional; omitted keys auto-inferred from noise_info.
    noise_preprocess : str — 'auto' | 'notch' | 'bandstop' | 'none'.
                       Default 'none'.
    noise_preprocess_params : dict or None — settings for PSD-based filtering.
        pch_size       : int — PSD patch size (default 32).
        n_smooth       : int — PSD smoothing iterations (default 100).
        peak_threshold : float — periodic peak detection threshold
                         (default 100.0).
        notch_radius   : int — notch filter radius (default 3).
                         Used when mode='auto' or 'notch'.
        freq_low       : float — bandstop low cutoff (default 0.3).
        freq_high      : float — bandstop high cutoff (default 0.5).
        order          : int — Butterworth order (default 2).
        All keys optional; omitted keys use defaults above.
    blind_denoise : str — 'guided' | 'bilateral' | 'bm3d' | 'nlm'
                    | 'none'.  Default 'none'.
                    Applied to x BEFORE passing to optimize_kernel
                    at each blind iteration.
    blind_denoise_params : dict or None — same keys as preprocess_params
        for the chosen method.  Differences from preprocess defaults:
        For 'guided': radius default 2 (smaller for iterative use).
        All keys optional; omitted keys use defaults.
    pre_nonblind : str — 'bm3d' | 'nlm' | 'bilateral' | 'guided' | 'tv'
                   | 'act' | 'none'.  Default 'none'.
    pre_nonblind_params : dict or None — same keys as preprocess_params
        for the chosen method (see preprocess_params above).
        For 'act': same keys as act_params.
        All keys optional; omitted keys auto-inferred from noise_info.
    auto_params : dict or None — σ-dependent tuning of λ, ε, nb_λ.
                  Default None (disabled — all values fully manual).
                  Requires noise_estimation != 'none' to have σ.
                  When set, the following formula is applied:
                    λ       = k_lambda  × σ
                    ε       = k_thr_e   × σ²
                    nb_λ    = k_nb      / σ
                  Dict keys (all optional, defaults shown):
                    k_lambda : float — λ multiplier  (default 0.2).
                    k_thr_e  : float — ε multiplier  (default 4.0).
                    k_nb     : float — nb_λ numerator (default 30.0).
                  Example: auto_params={'k_lambda': 0.1, 'k_nb': 50.0}
                  Tips: increase k_lambda for noisier images (stronger prior),
                  decrease k_nb to trust the prior more in non-blind step.
    nb_method : str — non-blind deconvolution solver. Default 'hyper_laplacian'.
        'hyper_laplacian' — ADMM hyper-Laplacian prior (Krishnan & Fergus).
            Uses edge-replicate padding + edgetaper + FFT circular deconv.
            Fast, preserves sharp edges, but may produce ringing artefacts
            (especially with large kernels or strong noise).
            Controlled by nb_lambda and nb_alpha.
        'ringing_removal' — TV + L0 + bilateral (Pan et al. CVPR 2014).
            Uses wrap_boundary_liu for FFT-compatible periodic boundaries
            (eliminates the main source of ringing).  Then subtracts
            the bilateral-filtered difference between TV and L0 deconv
            to further suppress ringing.  Slower but better quality.
            Controlled by nb_params.
    nb_params : dict or None — extra parameters for the non-blind solver.
        For nb_method='hyper_laplacian': None (uses nb_lambda, nb_alpha).
        For nb_method='ringing_removal':
            lambda_tv   : float — TV weight (default 1e-3, range ~[5e-4, 1e-2]).
                          Higher → smoother result, less texture.
            lambda_l0   : float — L0 gradient weight (default 2e-3,
                          range ~[5e-4, 5e-3]). Higher → piecewise constant.
            weight_ring : float — ringing suppression (default 1.0).
                          0.0 = pure TV deconv, no ringing post-processing.
                          1.0 = full subtraction.  >1.0 aggressive removal.
        All keys optional; omitted keys use defaults above.
    """

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
        nb_method: str = 'hyper_laplacian',
        nb_params: dict = None,
    ):
        super().__init__(name='LowRank-BD')

        assert kernel_size >= 3 and kernel_size % 2 == 1, \
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
        """
        Perform blind deconvolution on the input blurred image.
        """
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

        # ── 2a. Impulse noise detection & removal ──────────────────────
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

        # ── 2b. Noise estimation ─────────────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(y_gray)

        # ── 2b½. Auto-params (α, ε, nb_λ) from σ ────────────────────
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

        # ── 2c. ScreeNOT SVD denoising ───────────────────────────────
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

        # ── 2d. ACT curvelet denoising ───────────────────────────────
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

        # ── 2e. Pre-pyramid denoising ───────────────────────────────
        if self.preprocess not in (None, 'none'):
            y_gray = self._apply_denoise(
                y_gray, self.preprocess, self.preprocess_params,
                noise_info)

        # ── 2f. PSD-based noise preprocessing ───────────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            y_gray, psd_info = self._apply_noise_preprocess(y_gray)

        # Sync preprocessed Y channel back to ycbcr for non-blind step
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

                # Denoise x before kernel estimation
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

        # ── 3½. Pre-nonblind denoising ────────────────────────────────
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
            # ── Pan et al. 2014: TV + L0 + bilateral ──────────────
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
            # ── Default: hyper-Laplacian ADMM ─────────────────────
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

    # ── Blind-loop denoiser (x before kernel step) ───────────────────
    def _apply_blind_denoise(self, x, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)   # smaller for iterative use
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    # ── Pre-nonblind denoiser ───────────────────────────────────────
    def _apply_pre_nonblind(self, img, noise_info):
        return self._apply_denoise(
            img, self.pre_nonblind, self.pre_nonblind_params, noise_info)

    # ── Interface methods ───────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        """Return current hyper-parameter list."""
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
        """Update hyper-parameters from a dictionary."""
        for key, value in params.items():
            if key == 'lambda':
                self.lambda_ = value
            elif hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        """Convergence history (per-iteration kernel changes)."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Hyper-parameters and run-time statistics after process()."""
        return self.hyperparams
