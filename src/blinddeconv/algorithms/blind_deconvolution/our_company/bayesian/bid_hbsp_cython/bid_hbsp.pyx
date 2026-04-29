# cython: language_level=3
"""
bid_hbsp.pyx

Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior (BID-HBSP).

Implements the VB-EM algorithm from Castro-Macías et al. (2024) ICIP
with two solver modes:

    **filter_space** (default, paper formulation):
        Decompose into N=2 independent filtered-image problems
        (Eq. 17–18 of Castro-Macías et al. 2024).

    **image_space** (alternative):
        Estimate x as a single image via CG with D^T Γ D prior, then
        compute gradients for kernel estimation.  Babacan et al. (2009)
        style; kept for comparison.

References
[1] Castro-Macías, Pérez-Bueno, et al. (2024), ICIP 2024.
[2] Babacan, Molina, Katsaggelos (2009), IEEE TIP 18(1).
[4] Datta, Ghosh & Polson (2024), arXiv:2406.17058v3.
"""

import numpy as np
import time
import scipy.ndimage as ndimage
from typing import Tuple, List, Any, Dict

from numpy.fft import fft2, ifft2
from .utils import (
    precompute_gradient_operators,
    init_gaussian_kernel,
    fft_convolve,
    forward_diff_x,
    forward_diff_y,
    adjoint_diff_x,
    adjoint_diff_y,
    compute_hs_weights,
    compute_hs_weights_scalar,
    edgetaper,
)
from .solvers import (
    solve_image_cg,
    solve_filtered_image_cg,
    solve_kernel_fourier,
    solve_kernel_qp_filterspace,
    update_noise_precision,
    final_deconvolution,
)
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root (pyproject.toml)")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _p in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm


# --- PYRAMID HELPERS ---

def _init_kernel(kh, kw):
    k = np.zeros((kh, kw), dtype=np.float64)
    cy = (kh - 1) // 2
    cx = (kw - 1) // 2
    k[cy - 1, cx - 1 : cx + 1] = 0.5
    return k

def _fixsize(f, nk1, nk2):
    k1, k2 = f.shape
    while k1 != nk1 or k2 != nk2:
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]: f = f[1:, :]
            else: f = f[:-1, :]
        if k1 < nk1:
            s = f.sum(axis=1)
            tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
            if s[0] < s[-1]: tf[:k1, :] = f
            else: tf[1:k1 + 1, :] = f
            f = tf
        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]: f = f[:, 1:]
            else: f = f[:, :-1]
        if k2 < nk2:
            s = f.sum(axis=0)
            tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
            if s[0] < s[-1]: tf[:, :k2] = f
            else: tf[:, 1:k2 + 1] = f
            f = tf
        k1, k2 = f.shape
    return f

def _resizeKer(k, ret, k1, k2):
    k = ndimage.zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.sum() > 0:
        k = k / k.sum()
    return k

def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    rows, cols = psf.shape
    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                       np.arange(1, rows + 1, dtype=np.float64))
    total = np.sum(psf)
    if total == 0: return psf
    xc1 = np.sum(psf * X) / total
    yc1 = np.sum(psf * Y) / total
    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0
    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)
    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                     np.arange(cols, dtype=np.float64),
                                     indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift
    result = ndimage.map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                                     order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)

# --- END HELPERS ---

class BID_HBSP(DeconvolutionAlgorithm):
    """Bayesian Blind Image Deconvolution with Hyperbolic-Secant Prior."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        hs_scale: float = 0.01,
        noise_sigma: float = 0.005,
        max_iter: int = 40,
        cg_iter: int = 50,
        cg_tol: float = 1e-6,
        irw_iter: int = 5,
        # Architecture options
        kernel_init: str = "asymmetric",
        solver_mode: str = "filter_space",
        kernel_solver: str = "qp",
        boundary_mode: str = "padding",
        jacobi_mode: str = "scalar",
        center_kernel: bool = True,
        # Kernel estimation
        lambda_h_init: float = 100.0,
        lambda_h_min: float = 1.0,
        lambda_h_decay: float = 0.92,
        kernel_threshold: bool = True,
        # Noise
        beta_update: bool = False,
        beta_n_factor: float = 2.0,
        # ── Noise pipeline (all disabled by default) ────────────────────
        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        auto_beta: bool = False,
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
        # ── Non-blind restoration ───────────────────────────────────────
        final_deconv: str = 'irls',
        nb_params: dict = None,
        # ── Robust orchestrator (LIP-style) ────────────────────────────────
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
        # General
        verbose: bool = False,
    ):
        super().__init__(name="BID-HBSP")
        self.kernel_shape = tuple(kernel_shape)
        self.hs_scale = hs_scale
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter
        self.cg_iter = cg_iter
        self.cg_tol = cg_tol
        self.irw_iter = irw_iter

        self.kernel_init = kernel_init
        self.solver_mode = solver_mode
        self.kernel_solver = kernel_solver
        self.boundary_mode = boundary_mode
        self.jacobi_mode = jacobi_mode
        self.center_kernel = center_kernel

        self.lambda_h_init = lambda_h_init
        self.lambda_h_min = lambda_h_min
        self.lambda_h_decay = lambda_h_decay
        self.kernel_threshold = kernel_threshold

        self.beta_update = beta_update
        self.beta_n_factor = beta_n_factor

        # Noise pipeline
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.auto_beta = auto_beta
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

        # Non-blind
        self.final_deconv = final_deconv.lower()
        self.nb_params = nb_params

        # Robust orchestrator (LIP-style scheme)
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        # Snapshot of mutable fields from __init__.  The orchestrator
        # resets to this at the start of every process() call so
        # repeated runs are deterministic.  HBSP-specific extras
        # (screenot/act/noise_preprocess/histogram_eq) are NOT managed
        # by the orchestrator — they remain whatever the user set.
        self._defaults_snapshot = {
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'final_deconv': self.final_deconv,
            'nb_params': nb_params,
        }

        self.verbose = verbose

        self.history: Dict[str, list] = {
            "kernel_diff": [],
            "noise_precision": [],
            "residual_norm": [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run the full BID-HBSP pipeline."""
        start_time = time.time()

        # ── 1. Normalise ───────────────────────────────────────
        y_full = image.astype(np.float64)
        if y_full.max() > 1.0:
            y_full /= 255.0

        # ── 2. Grayscale ──────────────────────────────────────
        if y_full.ndim == 3 and y_full.shape[2] == 3:
            y_full = (0.2989 * y_full[:, :, 0]
                      + 0.5870 * y_full[:, :, 1]
                      + 0.1140 * y_full[:, :, 2])
        elif y_full.ndim == 3 and y_full.shape[2] == 1:
            y_full = y_full[:, :, 0]

        # ── 3a. Impulse noise detection & removal ──────────────
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            from .impulse_noise_estimation import (
                detect_impulse_noise, adaptive_median_filter)
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                y_full,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
                outlier_window=ip.get('outlier_window', 5),
            )
            if impulse_info['has_impulse']:
                if self.verbose:
                    print(f"[{self.name}] Impulse noise detected "
                          f"(density={impulse_info['density']:.4f}), "
                          f"applying adaptive median filter")
                y_full = adaptive_median_filter(
                    y_full, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))

        # ── 3b. Noise σ estimation ─────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(y_full)
            if self.verbose and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[{self.name}] Noise estimation "
                      f"({self.noise_estimation}): "
                      f"σ={sigma:.5f} (σ_255={sigma * 255:.2f})")
            if (self.auto_beta and noise_info is not None
                    and noise_info.get('sigma_norm', 0) > 0):
                self.noise_sigma = noise_info['sigma_norm']
                if self.verbose:
                    print(f"[{self.name}] auto_beta: noise_sigma "
                          f"overridden → {self.noise_sigma:.5f}")
        elif self.auto_mode == 'robust':
            # Orchestrator needs σ — auto-promote to PCA estimator.
            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(y_full)
            if self.verbose and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[{self.name}] auto_mode='robust' → PCA noise "
                      f"est.: σ={sigma:.5f} (σ_255={sigma * 255:.2f})")

        # ── 3b'. Robust orchestrator ───────────────────────────
        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)

        # ── 3c. ScreeNOT SVD denoising ─────────────────────────
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            if self.act_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one.")
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            y_full, screenot_info = screenot_denoise(
                y_full,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if self.verbose:
                print(f"[{self.name}] ScreeNOT applied "
                      f"(rank={screenot_info.get('rank', '?')})")

        # ── 3d. ACT curvelet denoising ─────────────────────────
        act_info = None
        if self.act_preprocess == 'auto':
            from .act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            y_full, act_info = act_denoise(
                y_full,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if self.verbose:
                print(f"[{self.name}] ACT curvelet denoising applied")

        # ── 3e. Spatial pre-blind denoise ──────────────────────
        if self.preprocess not in (None, 'none'):
            y_full = self._apply_denoise(
                y_full, self.preprocess, self.preprocess_params, noise_info)
            if self.verbose:
                print(f"[{self.name}] Pre-blind denoise: {self.preprocess}")

        # ── 3f. PSD-based noise preprocessing ──────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            y_full, psd_info = self._apply_noise_preprocess(y_full)
            if self.verbose:
                print(f"[{self.name}] PSD noise preprocess: "
                      f"{self.noise_preprocess}")

        # ── 3g. Histogram equalization ─────────────────────────
        y_for_restore = y_full.copy()
        if self.histogram_eq not in (None, 'none'):
            y_full = self._apply_histogram_eq(y_full)
            if self.verbose:
                print(f"[{self.name}] Histogram equalization: "
                      f"{self.histogram_eq}")

        # ── 4. Coarse-to-fine blind kernel estimation ──────────
        kh_full, kw_full = self.kernel_shape
        b = self.hs_scale
        alpha = 1.0 / b

        min_k = min(kh_full, kw_full)
        ret = np.sqrt(0.5)
        maxitr = max(int(np.floor(np.log(5.0 / min_k) / np.log(ret))), 0)
        num_scales = maxitr + 1

        retv = ret ** np.arange(num_scales)
        k1list = np.ceil(kh_full * retv).astype(int)
        k1list = k1list + (k1list % 2 == 0)
        k2list = np.ceil(kw_full * retv).astype(int)
        k2list = k2list + (k2list % 2 == 0)

        h = None
        beta = 1.0 / (self.noise_sigma ** 2 + 1e-12)
        self._beta_init = beta

        if self.verbose:
            print(f"[{self.name}] {num_scales} scales, "
                  f"β={beta:.1f}, α={alpha:.3f}, "
                  f"mode={self.solver_mode}, "
                  f"kernel_solver={self.kernel_solver}, "
                  f"boundary={self.boundary_mode}")

        n_iter = 0
        for s in range(num_scales - 1, -1, -1):
            kh, kw = int(k1list[s]), int(k2list[s])
            cret = retv[s]

            # Kernel init / upscale
            if s == num_scales - 1:
                if self.kernel_init == 'asymmetric':
                    h = _init_kernel(kh, kw)
                elif self.kernel_init == 'delta':
                    h = np.zeros((kh, kw), dtype=np.float64)
                    h[kh // 2, kw // 2] = 1.0
                else:  # 'gaussian'
                    h = init_gaussian_kernel((kh, kw))
            else:
                h = _resizeKer(h, 1.0 / ret, kh, kw)

            # Image at this scale
            if s == 0:
                y_level = y_full.copy()
            else:
                y_level = ndimage.zoom(y_full, cret, order=1)
            H_img, W_img = y_level.shape

            # Boundary handling (once per scale)
            if self.boundary_mode == 'edgetaper':
                y_work = edgetaper(y_level, h)
            else:
                y_work = y_level.copy()

            lambda_h = self.lambda_h_init

            if self.verbose:
                print(f"\n  Scale {num_scales - s}/{num_scales}  "
                      f"img {H_img}×{W_img}  kernel {kh}×{kw}")

            # ── Dispatch to solver mode ────────────────────────
            if self.solver_mode == 'filter_space':
                h, beta, n_iter = self._run_filter_space(
                    y_work, y_level, h, beta, alpha,
                    (kh, kw), lambda_h, s, num_scales,
                    noise_info)
            else:
                h, beta, n_iter = self._run_image_space(
                    y_work, y_level, h, beta, b,
                    (kh, kw), lambda_h, s, num_scales,
                    noise_info)

            # Centre kernel after each scale to prevent drift
            if self.center_kernel:
                h = adjust_psf_center(h)
                h[h < 0] = 0.0
                if h.sum() > 0:
                    h /= h.sum()

        # ── 5. Pre-nonblind denoising ──────────────────────────
        y_nb = y_for_restore
        if self.pre_nonblind not in (None, 'none'):
            y_nb = self._apply_denoise(
                y_nb, self.pre_nonblind, self.pre_nonblind_params,
                noise_info)
            if self.verbose:
                print(f"[{self.name}] Pre-nonblind denoise: "
                      f"{self.pre_nonblind}")

        # ── 6. Non-blind restoration ───────────────────────────
        lambda_final = beta * 0.0005
        if self.final_deconv == 'irls':
            if self.verbose:
                print(f"\n[{self.name}] Non-blind IRLS "
                      f"(p=0.8, λ={lambda_final:.6f})")
            x_final = final_deconvolution(y_nb, h, beta, lambda_final)
        elif self.final_deconv == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            x_final = adaptive_lp_deconv(
                y_nb, h,
                alpha=nbp.get('alpha', 0.8),
                sigma_n=sigma_n,
                two_stage=nbp.get('two_stage', True),
            )
        elif self.final_deconv == 'wiener':
            x_final = self._wiener_filter(y_nb, h, noise_info)
        elif self.final_deconv == 'tikhonov':
            x_final = self._tikhonov_filter(y_nb, h, noise_info)
        elif self.final_deconv == 'ringing':
            from .non_blind import ringing_artifacts_removal
            nbp = self.nb_params or {}
            x_final = ringing_artifacts_removal(
                y_nb, h,
                lambda_tv=nbp.get('lambda_tv', 4e-3),
                lambda_l0=nbp.get('lambda_l0', 2e-3),
                weight_ring=nbp.get('weight_ring', 0.5),
            )
            if self.verbose:
                print(f"[{self.name}] Non-blind: ringing_artifacts_removal "
                      f"(λ_tv={nbp.get('lambda_tv', 4e-3)}, "
                      f"λ_l0={nbp.get('lambda_l0', 2e-3)}, "
                      f"w_ring={nbp.get('weight_ring', 0.5)})")
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'irls', 'adaptive_lp', 'wiener', 'tikhonov', "
                "or 'ringing'.")

        x_final = np.clip(x_final, 0.0, 1.0)

        # ── 7. Diagnostics ─────────────────────────────────────
        self.timer = time.time() - start_time
        self.hyperparams = {
            "hs_scale": b,
            "alpha": alpha,
            "noise_precision_final": beta,
            "noise_sigma_estimated": (
                1.0 / np.sqrt(beta) if beta > 0 else None),
            "lambda_h_final": lambda_h,
            "iterations": n_iter,
            "time_seconds": self.timer,
            "final_deconv": self.final_deconv,
            # Noise pipeline info
            "impulse_preprocess": self.impulse_preprocess,
            "impulse_info": ({k_: v for k_, v in impulse_info.items()
                             if k_ != 'impulse_mask'}
                             if impulse_info else None),
            "noise_estimation": self.noise_estimation,
            "noise_info": noise_info,
            "screenot_preprocess": self.screenot_preprocess,
            "screenot_info": screenot_info,
            "act_preprocess": self.act_preprocess,
            "act_info": act_info,
            "preprocess": self.preprocess,
            "noise_preprocess": self.noise_preprocess,
            "psd_info": ({k_: v for k_, v in psd_info.items()
                         if k_ != 'psd_2d'} if psd_info else None),
            "histogram_eq": self.histogram_eq,
            "blind_denoise": self.blind_denoise,
            "pre_nonblind": self.pre_nonblind,
            "auto_mode": self.auto_mode,
            "orchestrator_info": orchestrator_info,
        }

        x_out = np.clip(np.round(x_final * 255.0), 0, 255).astype(np.int16)
        return x_out, h

    # ───────────────────────────────────────────────────────────
    #  IMAGE-SPACE solver
    # ───────────────────────────────────────────────────────────
    def _run_image_space(self, y_work, y_level, h, beta, b,
                         kernel_shape, lambda_h, s, num_scales,
                         noise_info=None):
        kh, kw = kernel_shape
        H_img, W_img = y_level.shape
        use_padding = (self.boundary_mode == 'padding')

        if use_padding:
            pad_h = kh // 2 + 1
            pad_w = kw // 2 + 1
            y_pad = np.pad(y_level, ((pad_h, pad_h), (pad_w, pad_w)),
                           mode='edge')
            x_est = y_pad.copy()
            sigma_sq = np.zeros_like(y_pad)
        else:
            pad_h = pad_w = 0
            x_est = y_level.copy()
            sigma_sq = np.zeros_like(x_est)

        n_iter = 0

        for it in range(self.max_iter):
            h_prev = h.copy()

            if use_padding:
                y_cg = y_pad
            elif self.boundary_mode == 'edgetaper_iter':
                y_cg = edgetaper(y_level, h)
            else:
                y_cg = y_work

            gamma_x, gamma_y = compute_hs_weights(
                forward_diff_x(x_est), forward_diff_y(x_est),
                sigma_sq, b)

            x_est, sigma_sq = solve_image_cg(
                y_cg, h, x_est, beta,
                gamma_x, gamma_y,
                max_cg_iter=self.cg_iter,
                cg_tol=self.cg_tol,
                jacobi_mode=self.jacobi_mode,
            )

            if self.blind_denoise not in (None, 'none'):
                x_den = self._apply_blind_denoise(
                    np.clip(x_est, 0.0, 1.0), noise_info)
                x_est = x_den

            if use_padding:
                x_inner = x_est[pad_h:-pad_h, pad_w:-pad_w]
                sig_inner = sigma_sq[pad_h:-pad_h, pad_w:-pad_w]
            else:
                x_inner = x_est
                sig_inner = sigma_sq

            use_thr = self.kernel_threshold and (s == 0)
            if self.kernel_solver == 'qp':
                dx_est = forward_diff_x(x_inner)
                dy_est = forward_diff_y(x_inner)
                sigma_grad = 2.0 * sig_inner
                y_dx = forward_diff_x(y_level)
                y_dy = forward_diff_y(y_level)
                filt_data = [
                    (y_dx, dx_est, sigma_grad),
                    (y_dy, dy_est, sigma_grad),
                ]
                h = solve_kernel_qp_filterspace(
                    filt_data, (kh, kw),
                    lambda_h=lambda_h,
                    do_threshold=use_thr,
                )
            else:
                h = solve_kernel_fourier(
                    y_level, x_inner, sig_inner, (kh, kw),
                    beta, lambda_h,
                    do_threshold=use_thr,
                )

            if self.beta_update:
                beta = update_noise_precision(
                    y_level, h, x_inner, beta)
                beta = float(np.clip(
                    beta, self._beta_init * 0.1, self._beta_init * 50.0))

            lambda_h = max(lambda_h * self.lambda_h_decay,
                           self.lambda_h_min)

            diff = float(np.linalg.norm(h - h_prev))
            n_iter = it + 1

            if s == 0:
                res = float(np.linalg.norm(
                    y_level - fft_convolve(x_inner, h)))
                self.history["kernel_diff"].append(diff)
                self.history["noise_precision"].append(beta)
                self.history["residual_norm"].append(res)

            if self._callback is not None:
                self._callback({
                    'iteration': it + 1,
                    'scale': num_scales - s,
                    'num_scales': num_scales,
                    'kernel': h.copy(),
                    'image': x_inner.copy() if s == 0 else None,
                    'beta': beta,
                    'lambda_h': lambda_h,
                    'metrics': {
                        'kernel_diff': diff,
                        'residual_norm': res if s == 0 else None,
                    },
                })

            if self.verbose:
                print(f"    it {it+1:3d}  ΔH={diff:.2e}  "
                      f"β={beta:.1f}  λ_h={lambda_h:.2f}")

            if diff < 1e-5 and it > 5:
                if self.verbose:
                    print(f"    converged at iteration {n_iter}")
                break

        return h, beta, n_iter

    # ───────────────────────────────────────────────────────────
    #  FILTER-SPACE solver  (paper formulation, Sec. IV)
    # ───────────────────────────────────────────────────────────
    def _run_filter_space(self, y_work, y_level, h, beta, alpha,
                          kernel_shape, lambda_h, s, num_scales,
                          noise_info=None):
        kh, kw = kernel_shape
        H_img, W_img = y_level.shape
        beta_n = beta / self.beta_n_factor
        use_padding = (self.boundary_mode == 'padding')

        alpha_list = [alpha, alpha]
        N_FILT = 2

        if use_padding:
            pad_h = kh // 2 + 1
            pad_w = kw // 2 + 1
            y_pad = np.pad(y_level, ((pad_h, pad_h), (pad_w, pad_w)),
                           mode='edge')
        else:
            pad_h = pad_w = 0
            if self.boundary_mode == 'edgetaper_iter':
                y_pad = edgetaper(y_level, h)
            else:
                y_pad = y_work

        y_filt = [forward_diff_x(y_pad), forward_diff_y(y_pad)]
        x_filt = [yf.copy() for yf in y_filt]
        sig_sq = [np.zeros_like(y_pad) for _ in range(N_FILT)]
        n_iter = 0

        for it in range(self.max_iter):
            h_prev = h.copy()

            if self.boundary_mode == 'edgetaper_iter':
                y_obs = edgetaper(y_level, h)
                y_filt = [forward_diff_x(y_obs), forward_diff_y(y_obs)]

            theta = [
                compute_hs_weights_scalar(
                    x_filt[n], sig_sq[n], alpha_list[n])
                for n in range(N_FILT)
            ]

            for n in range(N_FILT):
                x_filt[n], sig_sq[n] = solve_filtered_image_cg(
                    y_filt[n], h, x_filt[n],
                    beta_n, theta[n],
                    max_cg_iter=self.cg_iter,
                    cg_tol=self.cg_tol,
                )

            if self.blind_denoise not in (None, 'none'):
                for n in range(N_FILT):
                    x_filt[n] = self._apply_blind_denoise(
                        x_filt[n], noise_info)

            if use_padding:
                x_inner = [xf[pad_h:-pad_h, pad_w:-pad_w]
                           for xf in x_filt]
                sig_inner = [ss[pad_h:-pad_h, pad_w:-pad_w]
                             for ss in sig_sq]
                y_filt_inner = [yf[pad_h:-pad_h, pad_w:-pad_w]
                                for yf in y_filt]
            else:
                x_inner = x_filt
                sig_inner = sig_sq
                y_filt_inner = y_filt

            use_thr = self.kernel_threshold and (s == 0)
            filt_data = [
                (y_filt_inner[n], x_inner[n], sig_inner[n])
                for n in range(N_FILT)
            ]
            h = solve_kernel_qp_filterspace(
                filt_data, (kh, kw),
                lambda_h=lambda_h,
                do_threshold=use_thr,
            )

            if self.beta_update:
                div_field = (adjoint_diff_x(x_inner[0])
                             + adjoint_diff_y(x_inner[1]))
                _, _, F_grad_sq = precompute_gradient_operators(
                    (H_img, W_img))
                F_div = fft2(div_field)
                F_x_recon = F_div / (F_grad_sq + 1e-12)
                F_x_recon[0, 0] = np.mean(y_level) * H_img * W_img
                x_recon = np.clip(np.real(ifft2(F_x_recon)), 0.0, 1.0)
                beta = update_noise_precision(
                    y_level, h, x_recon, beta)
                beta = float(np.clip(
                    beta, self._beta_init * 0.1, self._beta_init * 50.0))
                beta_n = beta / self.beta_n_factor

            lambda_h = max(lambda_h * self.lambda_h_decay,
                           self.lambda_h_min)

            diff = float(np.linalg.norm(h - h_prev))
            n_iter = it + 1

            res_filt = None
            if s == 0:
                res_filt = sum(
                    float(np.sum(
                        (y_filt_inner[n] - fft_convolve(
                            x_inner[n], h)) ** 2))
                    for n in range(N_FILT)
                )
                self.history["kernel_diff"].append(diff)
                self.history["noise_precision"].append(beta)
                self.history["residual_norm"].append(
                    float(np.sqrt(res_filt)))

            if self._callback is not None:
                cb_image = None
                if s == 0:
                    div_cb = (adjoint_diff_x(x_inner[0])
                              + adjoint_diff_y(x_inner[1]))
                    _, _, F_gs = precompute_gradient_operators(
                        (H_img, W_img))
                    F_d = fft2(div_cb)
                    F_xr = F_d / (F_gs + 1e-12)
                    F_xr[0, 0] = np.mean(y_level) * H_img * W_img
                    cb_image = np.clip(
                        np.real(ifft2(F_xr)), 0.0, 1.0)
                self._callback({
                    'iteration': it + 1,
                    'scale': num_scales - s,
                    'num_scales': num_scales,
                    'kernel': h.copy(),
                    'image': cb_image,
                    'beta': beta,
                    'lambda_h': lambda_h,
                    'metrics': {
                        'kernel_diff': diff,
                        'residual_norm': (
                            float(np.sqrt(res_filt))
                            if res_filt is not None else None),
                    },
                })

            if self.verbose:
                print(f"    it {it+1:3d}  ΔH={diff:.2e}  "
                      f"β={beta:.1f}  λ_h={lambda_h:.2f}")

            if diff < 1e-5 and it > 5:
                if self.verbose:
                    print(f"    converged at iteration {n_iter}")
                break

        return h, beta, n_iter

    # ═══════════════════════════════════════════════════════════
    #  Private helpers: noise pipeline
    # ═══════════════════════════════════════════════════════════

    # ── Robust orchestrator (LIP-style) ───────────────────────
    def _orchestrate_robust(self, noise_info):
        """Soft-weighted auto configuration of the noise pipeline.

        Mirrors ``LIP_BD._orchestrate_robust``.  HBSP-core (β / α /
        λ_h / cg / max_iter) and HBSP-specific aux denoisers
        (screenot/act/noise_preprocess/histogram_eq/impulse) are NEVER
        modified.  Only ``preprocess`` / ``blind_denoise`` /
        ``pre_nonblind`` and the shared NB weights
        (``lambda_tv`` / ``lambda_l0`` / ``weight_ring``) are touched.
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        # 1) Reset from snapshot — avoid sticky state between calls.
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.blind_denoise = snap['blind_denoise']
        self.blind_denoise_params = snap['blind_denoise_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.final_deconv = snap['final_deconv']
        self.nb_params = snap['nb_params']

        # 2) Read σ; missing/zero ⇒ treat as clean.
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
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"defaults kept, final_deconv={self.final_deconv}")
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'noise_type': nt,
                'preprocess': self.preprocess,
                'blind_denoise': self.blind_denoise,
                'pre_nonblind': self.pre_nonblind,
                'final_deconv': self.final_deconv,
            }

        # 4) Heavy branch — smooth weight between σ_clean and σ_heavy.
        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-9))
        w = float(np.clip(w, 0.0, 1.0))
        regime = 'heavy' if w > 0.95 else 'medium'

        noise_type = nt if nt is not None else 'gaussian'
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        # 4a) Blind-loop denoiser — cheap edge-preserving bilateral.
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

        # 4b) Pre-pyramid global denoiser.
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

        # 4c) Pre-non-blind denoiser.
        if poisson_like:
            self.pre_nonblind = 'act'
            self.pre_nonblind_params = {'threshold_setting': 's'}
        elif w < 0.6:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma': float(max(sigma, 0.01))}
        else:
            # Heavy gaussian: BM3D with slightly inflated σ.  HBSP has
            # no 'ensemble' denoiser available — just stronger BM3D here.
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {
                'sigma': float(max(1.5 * sigma, 0.02)),
            }

        # 4d) σ-blend of shared non-blind weights (ringing_removal).
        nb_default = dict(snap['nb_params'] or {})
        lam_tv0 = float(nb_default.get('lambda_tv', 0.005))
        lam_l00 = float(nb_default.get('lambda_l0', 0.002))
        wring0 = float(nb_default.get('weight_ring', 0.5))

        k_lambda_tv = float(amp.get('k_lambda_tv', 0.05))
        k_lambda_l0 = float(amp.get('k_lambda_l0', 0.01))
        k_weight_ring = float(amp.get('k_weight_ring', 1.0))

        lam_tv_noisy = max(lam_tv0, k_lambda_tv * sigma)
        lam_l0_noisy = max(lam_l00, k_lambda_l0 * sigma)
        wring_noisy = min(2.0, wring0 + k_weight_ring * sigma)

        nb_blended = dict(nb_default)
        nb_blended['lambda_tv'] = (1.0 - w) * lam_tv0 + w * lam_tv_noisy
        nb_blended['lambda_l0'] = (1.0 - w) * lam_l00 + w * lam_l0_noisy
        nb_blended['weight_ring'] = (1.0 - w) * wring0 + w * wring_noisy
        self.nb_params = nb_blended

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
            'nb_lambda_tv': float(nb_blended['lambda_tv']),
            'nb_lambda_l0': float(nb_blended['lambda_l0']),
            'nb_weight_ring': float(nb_blended['weight_ring']),
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={noise_type}): "
                  f"pre={self.preprocess}, blind={self.blind_denoise}, "
                  f"pre_nb={self.pre_nonblind}, "
                  f"nb(λtv={nb_blended['lambda_tv']:.4f}, "
                  f"λl0={nb_blended['lambda_l0']:.4f}, "
                  f"wring={nb_blended['weight_ring']:.3f})")
        return info

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
            h_val = p.get('h', 0.8 * sig)
            return denoise_nl_means(
                img, h=h_val,
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

    def _apply_blind_denoise(self, img, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(
            img, self.blind_denoise, p, noise_info)

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
                yg_out = notch_filter(
                    yg, peaks,
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
        return np.real(np.fft.ifft2(H_conj * G / denom))

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
        return np.real(np.fft.ifft2(H_conj * G / denom))

    # ═══════════════════════════════════════════════════════════
    #  Interface methods
    # ═══════════════════════════════════════════════════════════

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("hs_scale", self.hs_scale),
            ("noise_sigma", self.noise_sigma),
            ("max_iter", self.max_iter),
            ("solver_mode", self.solver_mode),
            ("kernel_solver", self.kernel_solver),
            ("kernel_init", self.kernel_init),
            ("boundary_mode", self.boundary_mode),
            ("jacobi_mode", self.jacobi_mode),
            ("center_kernel", self.center_kernel),
            ("lambda_h_init", self.lambda_h_init),
            ("lambda_h_decay", self.lambda_h_decay),
            ("kernel_threshold", self.kernel_threshold),
            ("beta_update", self.beta_update),
            ("beta_n_factor", self.beta_n_factor),
            # Noise pipeline
            ("impulse_preprocess", self.impulse_preprocess),
            ("impulse_params", self.impulse_params),
            ("noise_estimation", self.noise_estimation),
            ("auto_beta", self.auto_beta),
            ("screenot_preprocess", self.screenot_preprocess),
            ("screenot_params", self.screenot_params),
            ("act_preprocess", self.act_preprocess),
            ("act_params", self.act_params),
            ("preprocess", self.preprocess),
            ("preprocess_params", self.preprocess_params),
            ("noise_preprocess", self.noise_preprocess),
            ("noise_preprocess_params", self.noise_preprocess_params),
            ("histogram_eq", self.histogram_eq),
            ("histogram_eq_params", self.histogram_eq_params),
            ("blind_denoise", self.blind_denoise),
            ("blind_denoise_params", self.blind_denoise_params),
            ("pre_nonblind", self.pre_nonblind),
            ("pre_nonblind_params", self.pre_nonblind_params),
            ("final_deconv", self.final_deconv),
            ("nb_params", self.nb_params),
            ("verbose", self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_shape":
                    self.kernel_shape = tuple(value)
                elif key == "final_deconv":
                    self.final_deconv = value.lower()
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        """Return per-iteration convergence history."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Return estimated / final hyper-parameters."""
        return self.hyperparams

    # ═══════════════════════════════════════════════════════════
    #  Interface methods
    # ═══════════════════════════════════════════════════════════

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("hs_scale", self.hs_scale),
            ("noise_sigma", self.noise_sigma),
            ("max_iter", self.max_iter),
            ("solver_mode", self.solver_mode),
            ("kernel_solver", self.kernel_solver),
            ("kernel_init", self.kernel_init),
            ("boundary_mode", self.boundary_mode),
            ("jacobi_mode", self.jacobi_mode),
            ("center_kernel", self.center_kernel),
            ("lambda_h_init", self.lambda_h_init),
            ("lambda_h_decay", self.lambda_h_decay),
            ("kernel_threshold", self.kernel_threshold),
            ("beta_update", self.beta_update),
            ("beta_n_factor", self.beta_n_factor),
            # Noise pipeline
            ("impulse_preprocess", self.impulse_preprocess),
            ("impulse_params", self.impulse_params),
            ("noise_estimation", self.noise_estimation),
            ("auto_beta", self.auto_beta),
            ("screenot_preprocess", self.screenot_preprocess),
            ("screenot_params", self.screenot_params),
            ("act_preprocess", self.act_preprocess),
            ("act_params", self.act_params),
            ("preprocess", self.preprocess),
            ("preprocess_params", self.preprocess_params),
            ("noise_preprocess", self.noise_preprocess),
            ("noise_preprocess_params", self.noise_preprocess_params),
            ("histogram_eq", self.histogram_eq),
            ("histogram_eq_params", self.histogram_eq_params),
            ("blind_denoise", self.blind_denoise),
            ("blind_denoise_params", self.blind_denoise_params),
            ("pre_nonblind", self.pre_nonblind),
            ("pre_nonblind_params", self.pre_nonblind_params),
            ("final_deconv", self.final_deconv),
            ("nb_params", self.nb_params),
            ("verbose", self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_shape":
                    self.kernel_shape = tuple(value)
                elif key == "final_deconv":
                    self.final_deconv = value.lower()
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        """Return per-iteration convergence history."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Return estimated / final hyper-parameters."""
        return self.hyperparams
