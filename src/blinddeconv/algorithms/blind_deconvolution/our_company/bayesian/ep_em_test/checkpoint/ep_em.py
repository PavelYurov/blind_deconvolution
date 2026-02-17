"""
Blind Image Deconvolution using Expectation Propagation (EP-EM).
Implementation Strategy: Fast-Cx (Spectral Uncertainty) + HQS + PGD.

Based on:
    Abdulaziz, A., et al. "Blind deconvolution of images corrupted by Gaussian noise 
    using Expectation Propagation." EUSIPCO 2021.

Modules:
    - utils: FFT helpers and math operators.
    - solvers: Pure functions for Image (HQS), Uncertainty (Spectral), and Kernel (PGD).
    - ep_em: Main class managing the EM loop.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict
from scipy.signal import convolve2d
from .utils import precompute_gradient_operators, compute_spatial_gradient, edgetaper
from .solvers import solve_image_hqs, estimate_uncertainty, solve_kernel_pgd, non_neg_ep


try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class EP_EM(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 6.7,     
        noise_sigma: float = 0.05, 
        max_iter: int = 30,          
        hqs_iter: int = 5,          
        pgd_iter: int = 20,          
        pgd_momentum: float = 0.9,
        beta_max: float = 512.0,     
        strategy: str = 'fast',
        num_probes: int = 10,
        non_neg: bool = True,
        verbose: bool = False
    ):
        super().__init__(name='EP-EM-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.hqs_iter = hqs_iter
        self.pgd_iter = pgd_iter
        self.pgd_momentum = pgd_momentum
        self.beta_max = beta_max
        self.strategy = strategy
        self.num_probes = num_probes
        self.non_neg = non_neg
        
        self.history = {'kernel_diff': []}
        self.hyperparams = {}

    def _build_Dx_fast(self, r: np.ndarray, kh: int, kw: int, H: int, W: int) -> np.ndarray:
        """
        Быстрое построение матрицы Dx (ковариация) без вложенных циклов.
        Математически эквивалентно оригиналу, но работает за доли секунды.
        """
        k = kh * kw
        I = np.arange(k)
        y_idx = I // kw
        x_idx = I % kw
        

        dy = (y_idx[:, None] - y_idx[None, :]) % H
        dx = (x_idx[:, None] - x_idx[None, :]) % W
        

        D_x = (H * W) * r[dy, dx]
        return D_x

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_full = image.astype(np.float64)
        if y_full.max() > 1.0:
            y_full /= 255.0
            
        H, W = y_full.shape
        kh, kw = self.kernel_shape

        sig = max(kh, kw) / 8.0 
        grid_y, grid_x = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2 * sig**2))
        h /= h.sum()
        
        y = edgetaper(y_full, h)
        x = y.copy()

        F_ops = precompute_gradient_operators((H, W))
        _, _, F_grad_sq = F_ops
        
        if self.verbose:
            print(f"[{self.name}] Start. Img: {H}x{W}, Ker: {kh}x{kw}")

        for it in range(self.max_iter):
            h_prev = h.copy()
            
            x = solve_image_hqs(
                y, h, x, 
                self.noise_sigma, self.lambda_tv, 
                self.beta_max, self.hqs_iter, F_ops
            )
            

            grad_x, grad_y = compute_spatial_gradient(x)
            mean_grad = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)) + 1e-6
            lambda_eff = self.lambda_tv / (mean_grad * 0.5)

            uncertainty, r = estimate_uncertainty(
                h, self.noise_sigma, lambda_eff, (H, W), F_grad_sq,
                strategy=self.strategy
            )
            
            if self.non_neg:
                x = non_neg_ep(x, uncertainty)


            D_x = self._build_Dx_fast(r, kh, kw, H, W)

            h = solve_kernel_pgd(
                y, x, h, 
                D_x, self.pgd_iter, momentum=self.pgd_momentum
            )
            

            diff = np.linalg.norm(h - h_prev)
            if self.verbose:
                print(f"Iter {it+1}: dH={diff:.6f}, Uncert={uncertainty:.2e}")
            
            if diff < 1e-6:
                break
        
        y_final_taper = edgetaper(y_full, h)
        x_final = solve_image_hqs(
            y_final_taper, h, x, 
            self.noise_sigma, self.lambda_tv * 0.8, 
            self.beta_max * 4.0, self.hqs_iter * 2, F_ops
        )
        
        x_final = np.clip(x_final * 255.0, 0, 255).astype(np.uint8)
        return x_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_tv', self.lambda_tv),
            ('noise_sigma', self.noise_sigma),
            ('max_iter', self.max_iter),
            ('strategy', self.strategy),
            ('non_neg', self.non_neg)
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

def run_algorithm(g, kernel_shape, **kwargs):
    algo = EP_EM(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
