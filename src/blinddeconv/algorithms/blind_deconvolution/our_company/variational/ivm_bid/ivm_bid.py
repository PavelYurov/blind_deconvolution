"""
Blind Image Deconvolution via TV-Regularized Alternating Minimization.

This implementation uses a powerful variational framework:
1. Image Estimation: Total Variation (TV) regularization solved via Split Bregman.
   (More robust to ringing than the H1 method in Laaziri et al. 2022).
2. Kernel Estimation: Tikhonov regularization solved via Fourier domain (FFT).
3. Parameter Strategy:
   - Lambda_f (Image): Cooling schedule (coarse-to-fine strategy).
   - Lambda_h (Kernel): GCV or fixed strategy.

References for the algorithmic structure:
    T. Goldstein, S. Osher, "The Split Bregman Method for L1-Regularized Problems,"
    SIAM J. Imaging Sci., 2009.
    
    (Structure inspired by Laaziri et al., 2022, but with improved TV model).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    precompute_gradient_operators,
    initialize_kernel,
    compute_residual_energy,
    edgetaper,
    compute_strong_gradients
)
from .solvers import (
    solve_image_tikhonov,
    solve_kernel_gradient_domain
    
)

# ---- Robust import of the framework base class ----
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

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm


class IVM_BID(DeconvolutionAlgorithm):
    """
    Implementation of Laaziri et al. 2022 (H1 Regularization).
    Improved with Gradient-Domain kernel estimation for stability.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # Настройки для H1 регуляризации
        lambda_f_start: float = 1.0,   # Начинаем с сильного подавления шума
        lambda_f_end: float = 1e-3,    # Заканчиваем тонкими деталями
        lambda_h: float = 0.5,         # Регуляризация ядра
        max_iter: int = 50,
        tol: float = 1e-6,
        verbose: bool = False,
    ):
        super().__init__(name="IVM-BID-H1")
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_f_start = lambda_f_start
        self.lambda_f_end = lambda_f_end
        self.lambda_h = lambda_h
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.history = {}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Normalize
        g = image.astype(np.float64)
        if g.max() > 1.0: g /= 255.0
        
        # 2. Edgetaper (Обязательно!)
        g_tapered = edgetaper(g, self.kernel_shape)
        H, W = g.shape

        # 3. Initialization
        h = initialize_kernel(self.kernel_shape)
        # В H1 методе лучше стартовать с самой размытой картинки
        f = g_tapered.copy() 

        F_ops = precompute_gradient_operators((H, W))
        
        # Cooling schedule for lambda_f (Геометрическая прогрессия)
        if self.max_iter > 1:
            decay = (self.lambda_f_end / self.lambda_f_start) ** (1 / (self.max_iter - 1))
        else:
            decay = 1.0
        
        curr_lambda_f = self.lambda_f_start

        self.history = {"diff": []}

        # 4. Alternating Minimization Loop
        n_iter = 0
        for it in range(self.max_iter):
            h_prev = h.copy()

            # --- Step A: Image Estimation (H1 Tikhonov) ---
            # Быстрое решение в частотной области. Не дает "мультяшности".
            f = solve_image_tikhonov(
                g_tapered, h, 
                lambda_f=curr_lambda_f,
                F_ops=F_ops
            )

            # --- Step B: Kernel Estimation (Gradient Domain) ---
            # Ищем ядро, сравнивая градиенты f и g
            h = solve_kernel_gradient_domain(
                g_tapered, f,
                kernel_shape=self.kernel_shape,
                lambda_h=self.lambda_h,
                F_ops=F_ops
            )

            # --- Step C: Parameter Update ---
            curr_lambda_f = max(curr_lambda_f * decay, self.lambda_f_end)

            # --- Monitoring ---
            diff = np.linalg.norm(h - h_prev) / (np.linalg.norm(h) + 1e-12)
            self.history["diff"].append(diff)
            
            if self.verbose:
                print(f"Iter {it+1}: dH={diff:.2e}, λ_f={curr_lambda_f:.2e}")

            if diff < self.tol and it > 10:
                break
            n_iter += 1

        # 5. Final Refinement (Non-blind Wiener)
        # Финальный проход с очень малой лямбдой для четкости
        f_final = solve_image_tikhonov(
            g_tapered, h, 
            lambda_f=1e-4, 
            F_ops=F_ops
        )

        elapsed = time.time() - start_time
        self.hyperparams = {"iter": n_iter, "time": elapsed}

        f_out = np.clip(f_final * 255.0, 0, 255).astype(np.int16)
        return f_out, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("lambda_f_start", self.lambda_f_start),
            ("lambda_f_end", self.lambda_f_end),
            ("lambda_h", self.lambda_h),
            ("max_iter", self.max_iter)
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

def run_algorithm(g, kernel_shape, **kwargs):
    algo = IVM_BID(kernel_shape, **kwargs)
    return algo.process(g)