"""
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
    """Асимметричная инициализация ядра (ломает симметрию)."""
    k = np.zeros((kh, kw), dtype=np.float64)
    cy = (kh - 1) // 2
    cx = (kw - 1) // 2
    k[cy - 1, cx - 1 : cx + 1] = 0.5
    return k

def _fixsize(f, nk1, nk2):
    """Точная подгонка размера при переходе между масштабами."""
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
    """Апскейлинг ядра (bicubic) с точной подгонкой размера."""
    k = ndimage.zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.sum() > 0:
        k = k / k.sum()
    return k

def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    """Центрирование по центру масс (предотвращает уплывание ядра за края)."""
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
    """Bayesian Blind Image Deconvolution with Hyperbolic-Secant Prior.

    Parameters
    ----------
    kernel_shape : (kh, kw)
    hs_scale : float
        Scale *b* of HS distribution. Smaller → stronger sparsity.
    noise_sigma : float
        Initial noise std. β₀ = 1/σ².
    max_iter : int
    cg_iter, cg_tol : CG parameters.
    kernel_init : 'gaussian' | 'delta' | 'asymmetric'
        Kernel initialization.  'asymmetric' breaks symmetry (recommended).
    solver_mode : 'filter_space' | 'image_space'
        'filter_space' — N=2 independent CG per filter (paper Eq.17-18, default).
        'image_space' — single CG for x, then gradients for h (Babacan 2009).
    kernel_solver : 'fourier' | 'qp'
        'qp' — quadratic programme on simplex (paper Eq.20-22, recommended).
        'fourier' — Wiener filter in gradient domain (fast, approximate).
    boundary_mode : 'none' | 'edgetaper' | 'edgetaper_iter' | 'padding'
        How to handle FFT circular-boundary artefacts.
        'padding' — edge-pad the image before CG, crop for kernel step
                    (recommended, eliminates wrap-around).
        'edgetaper' — apply edgetaper once per scale.
        'edgetaper_iter' — recompute edgetaper every iteration.
        'none' — no boundary handling.
    jacobi_mode : 'scalar' | 'perpixel'
        Variance approximation for diag(H^T H).
    center_kernel : bool
        Re-centre kernel after each scale (prevents drift).
    beta_update : bool
    beta_n_factor : float
        Divisor for filter-space noise precision: β_n = β / factor.
    """

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

        # ── 1. Data preparation ────────────────────────────────
        y_full = image.astype(np.float64)
        if y_full.max() > 1.0:
            y_full /= 255.0

        kh_full, kw_full = self.kernel_shape
        b = self.hs_scale
        alpha = 1.0 / b

        # ── 2. Coarse-to-fine pyramid ──────────────────────────
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
                  f"mode={self.solver_mode}, kernel_solver={self.kernel_solver}, "
                  f"boundary={self.boundary_mode}")

        # ── 3. Multi-scale loop ────────────────────────────────
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
                    (kh, kw), lambda_h, s, num_scales)
            else:
                h, beta, n_iter = self._run_image_space(
                    y_work, y_level, h, beta, b,
                    (kh, kw), lambda_h, s, num_scales)

            # Centre kernel after each scale to prevent drift
            if self.center_kernel:
                h = adjust_psf_center(h)
                h[h < 0] = 0.0
                if h.sum() > 0:
                    h /= h.sum()

        # ── 4. Final non-blind deconvolution ───────────────────
        lambda_final = beta * 0.0005
        if self.verbose:
            print(f"\n[{self.name}] Non-blind IRLS "
                  f"(p=0.8, λ={lambda_final:.6f})")
        x_final = final_deconvolution(y_full, h, beta, lambda_final)

        # ── 5. Diagnostics ─────────────────────────────────────
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
        }

        x_out = np.clip(np.round(x_final * 255.0), 0, 255).astype(np.int16)
        return x_out, h

    # ───────────────────────────────────────────────────────────
    #  IMAGE-SPACE solver  (recommended)
    # ───────────────────────────────────────────────────────────
    def _run_image_space(self, y_work, y_level, h, beta, b,
                         kernel_shape, lambda_h, s, num_scales):
        """Image-space CG for x, then gradient-domain kernel estimation."""
        kh, kw = kernel_shape
        H_img, W_img = y_level.shape
        use_padding = (self.boundary_mode == 'padding')

        # ── Padding setup ──────────────────────────────────────
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

            # ── Observation for CG (padded domain) ─────────────
            if use_padding:
                y_cg = y_pad
            elif self.boundary_mode == 'edgetaper_iter':
                y_cg = edgetaper(y_level, h)
            else:
                y_cg = y_work  # 'none' or 'edgetaper'

            # (a) HS weights from gradients of current x
            gamma_x, gamma_y = compute_hs_weights(
                forward_diff_x(x_est), forward_diff_y(x_est),
                sigma_sq, b)

            # (b) Image-space CG  (on padded or original domain)
            x_est, sigma_sq = solve_image_cg(
                y_cg, h, x_est, beta,
                gamma_x, gamma_y,
                max_cg_iter=self.cg_iter,
                cg_tol=self.cg_tol,
                jacobi_mode=self.jacobi_mode,
            )

            # ── Crop to original size for kernel estimation ────
            if use_padding:
                x_inner = x_est[pad_h:-pad_h, pad_w:-pad_w]
                sig_inner = sigma_sq[pad_h:-pad_h, pad_w:-pad_w]
            else:
                x_inner = x_est
                sig_inner = sigma_sq

            # (c) Kernel estimation  (uses original-size y + x)
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
            else:  # 'fourier'
                h = solve_kernel_fourier(
                    y_level, x_inner, sig_inner, (kh, kw),
                    beta, lambda_h,
                    do_threshold=use_thr,
                )

            # (d) Noise precision update  (on original domain)
            if self.beta_update:
                beta = update_noise_precision(
                    y_level, h, x_inner, beta)
                beta = float(np.clip(
                    beta, self._beta_init * 0.1, self._beta_init * 50.0))

            # (e) λ_h annealing
            lambda_h = max(lambda_h * self.lambda_h_decay,
                           self.lambda_h_min)

            # Convergence monitoring
            diff = float(np.linalg.norm(h - h_prev))
            n_iter = it + 1

            if s == 0:
                res = float(np.linalg.norm(
                    y_level - fft_convolve(x_inner, h)))
                self.history["kernel_diff"].append(diff)
                self.history["noise_precision"].append(beta)
                self.history["residual_norm"].append(res)

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
                          kernel_shape, lambda_h, s, num_scales):
        """Filter-space VB: N=2 independent CG + QP kernel (2024 paper)."""
        kh, kw = kernel_shape
        H_img, W_img = y_level.shape
        beta_n = beta / self.beta_n_factor
        use_padding = (self.boundary_mode == 'padding')

        alpha_list = [alpha, alpha]  # same α for ∂x, ∂y
        N_FILT = 2

        # ── Padding / boundary setup ───────────────────────────
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
                y_pad = y_work  # edgetaper (applied once) or none

        # Filtered pseudo-observations on (possibly padded) domain
        y_filt = [forward_diff_x(y_pad), forward_diff_y(y_pad)]
        x_filt = [yf.copy() for yf in y_filt]
        sig_sq = [np.zeros_like(y_pad) for _ in range(N_FILT)]
        n_iter = 0

        for it in range(self.max_iter):
            h_prev = h.copy()

            # Re-taper per iteration (only for edgetaper_iter)
            if self.boundary_mode == 'edgetaper_iter':
                y_obs = edgetaper(y_level, h)
                y_filt = [forward_diff_x(y_obs), forward_diff_y(y_obs)]

            # (a) HS weights per filter
            theta = [
                compute_hs_weights_scalar(
                    x_filt[n], sig_sq[n], alpha_list[n])
                for n in range(N_FILT)
            ]

            # (b) CG per filtered image (on padded domain)
            for n in range(N_FILT):
                x_filt[n], sig_sq[n] = solve_filtered_image_cg(
                    y_filt[n], h, x_filt[n],
                    beta_n, theta[n],
                    max_cg_iter=self.cg_iter,
                    cg_tol=self.cg_tol,
                )

            # ── Crop to original size for kernel estimation ────
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

            # (c) Kernel estimation (QP on simplex, Eq. 20-22)
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

            # (d) Noise precision update via Poisson reconstruction
            #     Reconstruct x from filtered images:
            #     ∇²x = D_x^T(x_filt_dx) + D_y^T(x_filt_dy)
            #     Solve in Fourier domain, then use image-domain residual
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
                # Clip to reasonable range around initial estimate
                beta = float(np.clip(
                    beta, self._beta_init * 0.1, self._beta_init * 50.0))
                beta_n = beta / self.beta_n_factor

            # (e) λ_h annealing
            lambda_h = max(lambda_h * self.lambda_h_decay,
                           self.lambda_h_min)

            # Convergence
            diff = float(np.linalg.norm(h - h_prev))
            n_iter = it + 1

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

            if self.verbose:
                print(f"    it {it+1:3d}  ΔH={diff:.2e}  "
                      f"β={beta:.1f}  λ_h={lambda_h:.2f}")

            if diff < 1e-5 and it > 5:
                if self.verbose:
                    print(f"    converged at iteration {n_iter}")
                break

        return h, beta, n_iter


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
            ("beta_update", self.beta_update),
            ("beta_n_factor", self.beta_n_factor),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_shape":
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        """Return per-iteration convergence history."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Return estimated / final hyper-parameters."""
        return self.hyperparams


