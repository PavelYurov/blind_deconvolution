"""
Bayesian Sparse Blind Deconvolution Wrapper.
Implements the framework interface.
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

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
from .utils import compute_gradient_matrix_operators
from .solvers import (
    update_hyperparams,
    sample_latent_variances,
    sample_image_cg,
    sample_kernel,
    sample_noise_variance_marginalized,
    mh_shift_compensation
)

class BSBD_MCMC_NIGP(DeconvolutionAlgorithm):
    """
    Bayesian Sparse Blind Deconvolution using MCMC (NIG-Prior).
    Strict implementation of Civek & Ertin (2022).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        num_iter: int = 150,
        burn_in: int = 50,
        noise_sigma_init: float = 0.01,
        verbose: bool = False
    ):
        super().__init__(name='BSBD-MCMC-NIGP')
        self.kernel_shape = tuple(kernel_shape)
        self.num_iter = num_iter
        self.burn_in = burn_in
        self.noise_sigma_init = noise_sigma_init
        self.verbose = verbose
        
        self.hyperparams = {}
        self.history = {'noise_var': [], 'alpha': []}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # 1. Preprocessing
        y = image.astype(np.float64)
        if y.max() > 1.0: y /= 255.0
        H, W = y.shape
        kh, kw = self.kernel_shape
        
        # Precompute matrix indices for kernel operations
        indices = compute_gradient_matrix_operators(self.kernel_shape, (H, W))
        
        # 2. Initialization
        x = y.copy()
        
        # Initialize kernel (Gaussian centered)
        h = np.zeros(self.kernel_shape)
        cy, cx = kh//2, kw//2
        yg, xg = np.ogrid[-cy:kh-cy, -cx:kw-cx]
        h = np.exp(-(yg**2 + xg**2)/(2*2.0))
        h /= h.sum()
        
        # Initialize variances (sparse prior)
        grads = np.gradient(y)
        grad_mag = np.sqrt(grads[0]**2 + grads[1]**2)
        sigma_sq_x = grad_mag**2 + 1e-3
        
        # Params
        alpha_x = 1.0
        beta_x = 0.1
        sigma_sq_v = self.noise_sigma_init**2
        sigma_sq_gamma = 100.0 # Fixed large prior for kernel (Section III-D-1)
        
        x_accum = np.zeros_like(x)
        h_accum = np.zeros_like(h)
        samples_cnt = 0
        
        if self.verbose:
            print(f"[{self.name}] Start. {H}x{W}, K:{kh}x{kw}")

        # 3. Main Gibbs Loop (Table I + Fixes)
        for it in range(self.num_iter):
            t0 = time.time()
            
            # Step 1 & 2: Update Hyperparams
            alpha_x, beta_x = update_hyperparams(sigma_sq_x, alpha_x)
            
            # Step 3: Sample Latent Variances
            sigma_sq_x = sample_latent_variances(x, alpha_x, beta_x)
            
            # Step 4: Sample Image x
            # Clipping sigma to avoid numerical issues in CG
            sigma_sq_x_cl = np.clip(sigma_sq_x, 1e-10, 1e3)
            x = sample_image_cg(y, h, sigma_sq_x_cl, sigma_sq_v)
            x = np.maximum(x, 0.0)
            
            # Step 5: Sample Kernel h
            h = sample_kernel(y, x, sigma_sq_v, sigma_sq_gamma, self.kernel_shape, indices)
            h = np.maximum(h, 0.0)
            
            # Scale Fix (Section III-D-1): Fixing gamma variance implies scaling.
            # We explicitly normalize h to sum to 1 to resolve ambiguity quickly.
            h_sum = np.sum(h)
            if h_sum > 1e-6:
                h /= h_sum
                x *= h_sum # maintain y ~ h*x
                sigma_sq_x *= (h_sum**2)
            
            # Step 6: Sample Noise Variance (Marginalized)
            # This is robust against local optima
            sigma_sq_v = sample_noise_variance_marginalized(
                sigma_sq_v, y, x, sigma_sq_gamma, self.kernel_shape, indices
            )
            
            # Extra: Shift Compensation (Section III-D-2)
            # This aligns the kernel to the center
            if it % 2 == 0: # Do every other iteration to save time
                x, h, sigma_sq_x = mh_shift_compensation(
                    y, x, h, sigma_sq_x, sigma_sq_v, sigma_sq_gamma, indices
                )
            
            # History
            self.history['noise_var'].append(sigma_sq_v)
            self.history['alpha'].append(alpha_x)
            
            # Accumulate
            if it >= self.burn_in:
                x_accum += x
                h_accum += h
                samples_cnt += 1
                
            if self.verbose and it % 10 == 0:
                print(f"Iter {it}: Sv={np.sqrt(sigma_sq_v):.4f}, Alpha={alpha_x:.2f}, Time={time.time()-t0:.2f}s")

        # 4. Final Estimate
        if samples_cnt > 0:
            x_est = x_accum / samples_cnt
            h_est = h_accum / samples_cnt
        else:
            x_est = x
            h_est = h
            
        if h_est.sum() > 0:
            h_est /= h_est.sum()
            
        x_final = np.clip(x_est * 255.0, 0, 255).astype(np.float32)
        
        self.hyperparams = {
            'sigma_v': np.sqrt(sigma_sq_v),
            'alpha': alpha_x
        }
        
        return x_final, h_est

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('num_iter', self.num_iter),
            ('burn_in', self.burn_in),
            ('noise_sigma_init', self.noise_sigma_init)
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
    algo = BSBD_MCMC_NIGP(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history