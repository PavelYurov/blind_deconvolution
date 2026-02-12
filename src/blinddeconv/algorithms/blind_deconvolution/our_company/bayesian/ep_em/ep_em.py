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
from .utils import precompute_gradient_operators, compute_spatial_gradient
from .solvers import solve_image_hqs, estimate_uncertainty, solve_kernel_pgd, non_neg_ep

# Robust import of base class
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class EP_EM(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using Expectation Propagation (EP-EM).
    
    Implements the strategy described in Abdulaziz et al. (2021):
    1. Image Estimation: HQS (Half-Quadratic Splitting).
    2. Uncertainty: Fast-Cx or RBMC (Spectral or Monte Carlo Approximation) with adaptive lambda_eff.
    3. Kernel Estimation: Nesterov-accelerated PGD with covariance using full D_x.
    4. Non-negativity: Optional EP update for soft constraint.
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 6.7,
        noise_sigma: float = 0.05,
        max_iter: int = 30,
        # Solver params
        hqs_iter: int = 5,
        pgd_iter: int = 20,
        pgd_momentum: float = 0.9,
        beta_max: float = 1024.0,
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
        
        # Internal solver settings
        self.hqs_iter = hqs_iter
        self.pgd_iter = pgd_iter
        self.pgd_momentum = pgd_momentum
        self.beta_max = beta_max
        self.strategy = strategy
        self.num_probes = num_probes
        self.non_neg = non_neg
        
        self.history = {'kernel_diff': []}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # 1. Prepare Data
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
        
        H, W = y.shape
        kh, kw = self.kernel_shape
        
        # 2. Initialization
        # Adaptive Gaussian Kernel Initialization
        sig = max(kh, kw) / 10.0
        grid_y, grid_x = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2 * sig**2))
        h /= h.sum()
        
        x = y.copy()
        
        # Precompute Gradient Operators (Shared resource)
        F_ops = precompute_gradient_operators((H, W))
        _, _, F_grad_sq = F_ops
        
        if self.verbose:
            print(f"[{self.name}] Start. Img: {H}x{W}, Ker: {kh}x{kw}, Sigma: {self.noise_sigma}")

        # 3. Main EM Loop
        for it in range(self.max_iter):
            h_prev = h.copy()
            
            # E-STEP 
            # A. Estimate Image Mean (MAP) via HQS
            x = solve_image_hqs(
                y, h, x, 
                self.noise_sigma, self.lambda_tv, 
                self.beta_max, self.hqs_iter, F_ops
            )
            
            # Compute adaptive lambda_eff based on current estimate
            grad_x, grad_y = compute_spatial_gradient(x)
            mean_abs_grad = (np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))) / 2 + 1e-3
            lambda_eff = self.lambda_tv / mean_abs_grad

            # B. Estimate Uncertainty (Variance) and Autocovariance via Spectral Method
            uncertainty, r = estimate_uncertainty(
                h, self.noise_sigma, lambda_eff, (H, W), F_grad_sq,
                strategy=self.strategy, num_probes=self.num_probes
            )
            
            # C. Incorporate non-negativity if enabled
            if self.non_neg:
                x = non_neg_ep(x, uncertainty)

            # Compute full D_x matrix using autocovariance r
            k = kh * kw
            D_x = np.zeros((k, k))
            for ii in range(k):
                y1 = ii // kw
                x1 = ii % kw
                for jj in range(k):
                    y2 = jj // kw
                    x2 = jj % kw
                    ly = y1 - y2
                    lx = x1 - x2
                    D_x[ii, jj] = (H * W) * r[ly % H, lx % W]
            
            # M-STEP
            # Estimate Kernel via PGD with full D_x
            h = solve_kernel_pgd(
                y, x, h, 
                D_x, self.pgd_iter, momentum=self.pgd_momentum
            )
            
            # Monitoring
            diff = np.linalg.norm(h - h_prev)
            self.history['kernel_diff'].append(diff)
            
            if self.verbose:
                print(f"Iter {it+1}/{self.max_iter}: dH={diff:.6f}, Uncert={uncertainty:.2e}")
            
            if diff < 1e-5 and it > 5:
                if self.verbose: print("Converged.")
                break
        
        # 4. Final Non-Blind Pass
        # Run HQS one last time with full beta range to ensure sharpness
        if self.verbose: print("Final Refinement...")
        x_final = solve_image_hqs(
            y, h, x, 
            self.noise_sigma, self.lambda_tv, 
            self.beta_max, self.hqs_iter * 2, F_ops
        )
        
        self.hyperparams = {
            'lambda_tv': self.lambda_tv,
            'noise_sigma': self.noise_sigma,
            'final_uncertainty': uncertainty,
            'final_lambda_eff': lambda_eff,
            'iterations': it + 1
        }
        
        x_final = x_final * 255.0
        x_final = np.round(x_final).astype(np.int16)
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
