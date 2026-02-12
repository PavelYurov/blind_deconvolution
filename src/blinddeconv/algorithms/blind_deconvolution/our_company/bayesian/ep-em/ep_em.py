"""
Blind Image Deconvolution using Expectation Propagation (EP-EM).
Implementation Strategy: Fast-Cx (Spectral Uncertainty) + HQS + PGD.

Based on:
    Abdulaziz, A., et al. "Blind deconvolution of images corrupted by Gaussian noise 
    using Expectation Propagation." EUSIPCO 2021.

Modules:
    - Utils: FFT helpers and math operators.
    - Solvers: Pure functions for Image (HQS), Uncertainty (Spectral), and Kernel (PGD).
    - Algorithm: Main class managing the EM loop.
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict, Optional
import sys
import os

# Robust import of base class
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class EPEM_BID(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using Expectation Propagation (EP-EM).
    
    Implements the strategy described in Abdulaziz et al. (2021):
    1. Image Estimation: HQS (Half-Quadratic Splitting).
    2. Uncertainty: Fast-Cx (Spectral Approximation).
    3. Kernel Estimation: PGD (Projected Gradient Descent).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 0.003,
        noise_sigma: float = 0.01,
        max_iter: int = 30,
        # Solver params
        hqs_iter: int = 5,
        pgd_iter: int = 20,
        beta_max: float = 1024.0,
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
        self.beta_max = beta_max
        
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
            
            # B. Estimate Uncertainty (Variance) via Spectral Method
            uncertainty = estimate_uncertainty_spectral(
                h, self.noise_sigma, self.lambda_tv, 
                (H, W), F_grad_sq
            )
            
            # M-STEP
            # Estimate Kernel via PGD
            # Regularization weight depends on image uncertainty
            # Trace(h^T D h) approx sum(Var) * ||h||^2
            reg_weight = uncertainty * (H * W)
            
            h = solve_kernel_pgd(
                y, x, h, 
                reg_weight, self.pgd_iter
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
            'iterations': it + 1
        }
        
        return x_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_tv', self.lambda_tv),
            ('noise_sigma', self.noise_sigma),
            ('max_iter', self.max_iter)
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
    algo = EPEM_BID(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
