"""
Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior (BID-HBSP).

The method implements a variational Expectation–Maximisation (EM)
loop that alternates between:

    (a) **MM / Envelope step** — update spatially-adaptive HS prior weights
        :math:`\\gamma_i = \\tanh(|d_i|/b)\\,/\\,(|d_i|\\,b)` from the
        current gradient map.
    (b) **E-step (image)** — solve the GSM-regularised normal equation
        via CG with per-pixel weights.
    (c) **E-step (kernel)** — Fourier-domain Wiener filter + simplex
        projection.
    (d) **M-step (noise)** — update noise precision :math:`\\beta` from
        the reconstruction residual.

The Hyperbolic-Secant distribution
:math:`p(d) \\propto 1\\!/\\!\\cosh(d/b)` is represented as a
**Gaussian Scale Mixture** (GSM) with Pólya-Gamma mixing density
([4], Eq. 4–6), which keeps all conditional distributions Gaussian
and enables closed-form EM updates.

Observation model
.. math::
    y = h \\ast x + n, \\qquad n \\sim \\mathcal{N}(0,\\,\\beta^{-1}I).

Prior on image gradients
.. math::
    p(\\nabla x_i) \\propto \\frac{1}{\\cosh(\\nabla x_i \\,/\\, b)}.

References
[1] Castro-Macías, Pérez-Bueno, et al. (2024), "Bayesian Blind Image
    Deconvolution using a Hyperbolic-Secant prior", ICIP 2024.
[2] Babacan, Molina, Katsaggelos (2009), "Variational Bayesian Blind
    Deconvolution Using a Total Variation Prior", IEEE TIP 18(1).
[3] Polson & Scott (2016), "Mixtures, envelopes and hierarchical
    duality", J. R. Statist. Soc. B 78(3), pp. 701–727.
[4] Datta, Ghosh & Polson (2024), "Bayesian ICA with super-Gaussian
    Source Priors", arXiv:2406.17058v3.
"""

import numpy as np
import time
import scipy.ndimage as ndimage
from typing import Tuple, List, Any, Dict

from .utils import (
    precompute_gradient_operators,
    init_gaussian_kernel,
    fft_convolve,
)
from .solvers import (
    solve_image_cg,
    solve_image_irw,
    solve_kernel_fourier,
    update_noise_precision,
    update_hs_weights,
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


# --- DCP PYRAMID HELPERS ---

def _init_kernel(kh, kw):
    """Строго асимметричная инициализация ядра (как в DCP)."""
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
    kernel_shape : (kh, kw)
        Spatial dimensions of the blur kernel to estimate.
    hs_scale : float
        Scale parameter *b* of the HS distribution
        :math:`p(d) \\propto 1/\\cosh(d/b)`.
        Smaller values yield stronger sparsity (heavier tails);
        larger values approach Gaussian (L2) behaviour.
    noise_sigma : float
        Initial estimate of the noise standard deviation (:math:`\\sigma_n`).
        Used to seed :math:`\\beta_0 = 1/\\sigma_n^2`.
    max_iter : int
        Maximum number of outer EM iterations.
    cg_iter : int
        Maximum conjugate-gradient iterations per image solve.
    cg_tol : float
        CG convergence tolerance (absolute residual).
    lambda_h_init : float
        Initial kernel regularisation (Gaussian prior precision).
    lambda_h_min : float
        Minimum kernel regularisation (lower bound during annealing).
    lambda_h_decay : float
        Multiplicative decay applied to *lambda_h* each EM iteration
        (annealing schedule; 1.0 = constant).
    beta_update : bool
        Whether to update the noise precision :math:`\\beta` during the
        M-step.  If False, :math:`\\beta` is held at its initial value.
    kernel_threshold : bool
        Whether to zero-out small kernel values (< 5 % of peak) each
        iteration, promoting kernel sparsity.
    solver : ``'cg'`` | ``'irw'``
        Image solver backend:
        ``'cg'``  — per-pixel HS weights via CG (exact, recommended);
        ``'irw'`` — iteratively-reweighted Wiener (faster, approximate).
    verbose : bool
        Print per-iteration diagnostics.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        hs_scale: float = 0.5,
        noise_sigma: float = 0.01,
        max_iter: int = 40,
        # Image solver parameters
        cg_iter: int = 50,
        cg_tol: float = 1e-6,
        solver: str = "cg",
        irw_iter: int = 5,
        # Kernel estimation parameters
        lambda_h_init: float = 1e3,
        lambda_h_min: float = 1.0,
        lambda_h_decay: float = 0.92,
        kernel_threshold: bool = True,
        # Noise estimation
        beta_update: bool = True,
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
        self.solver = solver
        self.irw_iter = irw_iter

        self.lambda_h_init = lambda_h_init
        self.lambda_h_min = lambda_h_min
        self.lambda_h_decay = lambda_h_decay
        self.kernel_threshold = kernel_threshold

        self.beta_update = beta_update
        self.verbose = verbose

        self.history: Dict[str, list] = {
            "kernel_diff": [],
            "noise_precision": [],
            "residual_norm": [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Data preparation
        y_full = image.astype(np.float64)
        if y_full.max() > 1.0:
            y_full /= 255.0

        kh_full, kw_full = self.kernel_shape
        b = self.hs_scale

        # 2. Инициализация пирамиды строго как в DCP коде
        min_k = min(kh_full, kw_full)
        ret = np.sqrt(0.5)
        # Спускаемся пока размер ядра не станет ~5
        maxitr = max(int(np.floor(np.log(5.0 / min_k) / np.log(ret))), 0)
        num_scales = maxitr + 1

        retv = ret ** np.arange(0, num_scales)
        k1list = np.ceil(kh_full * retv).astype(int)
        k1list = k1list + (k1list % 2 == 0) # ensure odd
        k2list = np.ceil(kw_full * retv).astype(int)
        k2list = k2list + (k2list % 2 == 0)

        h = None
        beta = 1.0 / (self.noise_sigma ** 2 + 1e-12)

        if self.verbose:
            print(f"[{self.name}] Starting Coarse-to-Fine processing with {num_scales} scales.")

        # 3. Идем от маленького (num_scales-1) к большому (0)
        for s in range(num_scales - 1, -1, -1):
            kh, kw = k1list[s], k2list[s]
            current_kernel_shape = (kh, kw)
            cret = retv[s]
            
            # Инициализация / Апскейл ядра
            if s == num_scales - 1:
                h = _init_kernel(kh, kw) # ломаем симметрию!
            else:
                h = _resizeKer(h, 1.0 / ret, kh, kw)

            # Даунсэмпл картинки
            if s == 0:
                y_level = y_full.copy()
            else:
                y_level = ndimage.zoom(y_full, cret, order=1)

            H, W = y_level.shape
            x = y_level.copy()
            sigma_sq = np.zeros_like(x)
            lambda_h = self.lambda_h_init

            if self.verbose:
                print(f"\n--- Scale {num_scales - s}/{num_scales} | Img: {H}x{W} | Kernel: {kh}x{kw} ---")

            n_iter = 0
            for it in range(self.max_iter):
                h_prev = h.copy()

                gamma_x, gamma_y = update_hs_weights(x, sigma_sq, b)

                if self.solver == "cg":
                    x, sigma_sq = solve_image_cg(
                        y_level, h, x, beta,
                        gamma_x, gamma_y,
                        max_cg_iter=self.cg_iter,
                        cg_tol=self.cg_tol,
                    )
                else:
                    raise NotImplementedError("Only CG solver is strictly supported")

                # Thresholding применяем ТОЛЬКО на оригинальном размере (s == 0)
                # Иначе мелкое ядро 5x5 мгновенно схлопнется в точку.
                use_threshold = self.kernel_threshold if (s == 0) else False

                h = solve_kernel_fourier(
                    y_level, x, sigma_sq, current_kernel_shape, beta, lambda_h,
                    do_threshold=use_threshold,
                )

                if self.beta_update:
                    beta = update_noise_precision(y_level, h, x, beta)

                lambda_h = max(lambda_h * self.lambda_h_decay, self.lambda_h_min)

                diff = float(np.linalg.norm(h - h_prev))
                residual = float(np.linalg.norm(y_level - fft_convolve(x, h)))

                if s == 0:
                    self.history["kernel_diff"].append(diff)
                    self.history["noise_precision"].append(beta)
                    self.history["residual_norm"].append(residual)

                if self.verbose:
                    print(
                        f"  Iter {it + 1:3d}/{self.max_iter}:  "
                        f"ΔH={diff:.2e}  β={beta:.2f}  "
                        f"λ_h={lambda_h:.2f}  ‖r‖={residual:.4f}"
                    )

                n_iter = it + 1
                if diff < 1e-5 and it > 5:
                    if self.verbose: print(f"  Converged at iteration {n_iter}.")
                    break

            # Обязательное центрирование ядра в конце каждого уровня, 
            # чтобы оно не уплыло за границы при следующем апскейле
            h = adjust_psf_center(h)
            h[h < 0] = 0.0
            h /= h.sum()

        # 4. Final non-blind deconvolution
        lambda_final = beta * 0.0005
        
        if self.verbose:
            print(f"\n[{self.name}] Final non-blind deconvolution (IRLS p=0.8) (λ_reg={lambda_final:.4f}) …")
        x_final = final_deconvolution(y_full, h, beta, lambda_final)

        # 5. Store diagnostics
        self.timer = time.time() - start_time
        self.hyperparams = {
            "hs_scale": b,
            "noise_precision_final": beta,
            "noise_sigma_estimated": 1.0 / np.sqrt(beta) if beta > 0 else None,
            "lambda_h_final": lambda_h,
            "iterations": n_iter,
            "time_seconds": self.timer,
        }

        # 6. Output conversion
        x_out = x_final * 255.0
        x_out = np.clip(np.round(x_out), 0, 255).astype(np.int16)
        return x_out, h


    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("hs_scale", self.hs_scale),
            ("noise_sigma", self.noise_sigma),
            ("max_iter", self.max_iter),
            ("solver", self.solver),
            ("lambda_h_init", self.lambda_h_init),
            ("lambda_h_decay", self.lambda_h_decay),
            ("beta_update", self.beta_update),
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


