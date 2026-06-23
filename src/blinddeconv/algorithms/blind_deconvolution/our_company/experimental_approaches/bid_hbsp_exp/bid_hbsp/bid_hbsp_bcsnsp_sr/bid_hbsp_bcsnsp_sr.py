import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    precompute_gradient_operators,
    init_gaussian_kernel,
    fft_convolve,
    sr_initial_estimate,
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

class BID_HBSP_BCSNSP_SR(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: Tuple[int, int],

        hs_scale: float = 0.5,
        noise_sigma: float = 0.01,
        max_iter: int = 40,

        cg_iter: int = 50,
        cg_tol: float = 1e-6,
        solver: str = "cg",
        irw_iter: int = 5,

        lambda_h_init: float = 1e3,
        lambda_h_min: float = 1.0,
        lambda_h_decay: float = 0.92,
        kernel_threshold: bool = True,

        beta_update: bool = True,

        sr_lambda_prior: float = 0.5,
        sr_tv_iters: int = 5,

        verbose: bool = False,
    ):
        super().__init__(name="BID-HBSP+BCSNSP-SR")
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

        self.sr_lambda_prior = sr_lambda_prior
        self.sr_tv_iters = sr_tv_iters

        self.verbose = verbose

        self.history: Dict[str, list] = {
            "kernel_diff": [],
            "noise_precision": [],
            "residual_norm": [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        H, W = y.shape
        kh, kw = self.kernel_shape
        b = self.hs_scale

        h = init_gaussian_kernel(self.kernel_shape)

        if self.verbose:
            print(
                f"[{self.name}] Stage 0: SAR + TV initialisation "
                f"(tv_iters={self.sr_tv_iters}, λ={self.sr_lambda_prior}) …"
            )

        x = sr_initial_estimate(
            y,
            h_init=h,
            lambda_prior=self.sr_lambda_prior,
            tv_iters=self.sr_tv_iters,
            verbose=self.verbose,
        )

        if self.verbose:

            grad_y = float(np.sum(np.abs(np.diff(y, axis=1))))
            grad_x = float(np.sum(np.abs(np.diff(x, axis=1))))
            print(
                f"  SR init done. Gradient energy: y={grad_y:.1f} → "
                f"x₀={grad_x:.1f} (×{grad_x / (grad_y + 1e-12):.2f})"
            )

        beta = 1.0 / (self.noise_sigma ** 2 + 1e-12)
        lambda_h = self.lambda_h_init

        F_ops = precompute_gradient_operators((H, W))

        gamma_x = np.full((H, W), 1.0 / (b * b))
        gamma_y = np.full((H, W), 1.0 / (b * b))

        if self.verbose:
            print(
                f"[{self.name}] Stage 1: EM loop — {H}×{W}, "
                f"kernel {kh}×{kw}, b={b:.3f}, β₀={beta:.1f}"
            )

        n_iter = 0
        sigma_sq = np.zeros_like(x)

        for it in range(self.max_iter):
            h_prev = h.copy()

            gamma_x, gamma_y = update_hs_weights(x, sigma_sq, b)

            if self.solver == "cg":
                x, sigma_sq = solve_image_cg(
                    y, h, x, beta,
                    gamma_x, gamma_y,
                    max_cg_iter=self.cg_iter,
                    cg_tol=self.cg_tol,
                )
            else:
                x = solve_image_irw()
                sigma_sq = np.zeros_like(x)

            h = solve_kernel_fourier(
                y, x, sigma_sq, self.kernel_shape, beta, lambda_h,
                do_threshold=self.kernel_threshold,
            )

            if self.beta_update:
                beta = update_noise_precision(y, h, x, beta)

            lambda_h = max(lambda_h * self.lambda_h_decay, self.lambda_h_min)

            diff = float(np.linalg.norm(h - h_prev))
            residual = float(np.linalg.norm(y - fft_convolve(x, h)))
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
                if self.verbose:
                    print(f"  Converged at iteration {n_iter}.")
                break

        lambda_final = beta * 0.0005

        if self.verbose:
            print(
                f"[{self.name}] Stage 2: Final deconvolution "
                f"(IRLS p=0.8, λ={lambda_final:.4f}) …"
            )

        x_final = final_deconvolution(y, h, beta, lambda_final)

        self.timer = time.time() - start_time
        self.hyperparams = {
            "hs_scale": b,
            "noise_precision_final": beta,
            "noise_sigma_estimated": (
                1.0 / np.sqrt(beta) if beta > 0 else None
            ),
            "lambda_h_final": lambda_h,
            "iterations": n_iter,
            "sr_lambda_prior": self.sr_lambda_prior,
            "sr_tv_iters": self.sr_tv_iters,
            "time_seconds": self.timer,
        }

        x_out = x_final * 255.0
        x_out = np.round(x_out).astype(np.int16)
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
            ("sr_lambda_prior", self.sr_lambda_prior),
            ("sr_tv_iters", self.sr_tv_iters),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_shape":
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
