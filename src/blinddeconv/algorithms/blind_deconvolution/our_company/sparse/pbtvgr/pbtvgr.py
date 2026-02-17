import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict
from .utils import get_grad_operators, psf2otf, adjust_psf
from .solvers import solve_h_subproblem, solve_u_subproblem, solve_o_subproblem, solve_nonblind_tv

import sys
import os
from pathlib import Path

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

class PBTVGR(DeconvolutionAlgorithm):
    """
    Blind Deconvolution for Poissonian Blurred Image 
    With Total Variation and L0-Norm Gradient Regularizations.
    
    Reference: Dong et al., IEEE TIP 2021.
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_: float = 5e4,
        mu: float = 0.1,
        beta_init: float = 1.0,
        T: float = 50.0,
        tau: float = 10.0,
        xi: float = 5e4,
        eta_init: float = 1.0,
        max_iter: int = 40,
        tol: float = 1e-4,
        verbose: bool = False
    ):
        super().__init__(name='PBTVGR')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_ = lambda_      
        self.mu = mu                
        self.beta_init = beta_init  
        self.T = T                  
        self.tau = tau              
        
        self.xi = xi
        self.eta_init = eta_init
        
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.history = {'error': []}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        g = image.astype(np.float64)
        img_min, img_max = g.min(), g.max()
        
        scale_factor = img_max - img_min + 1e-8
        g = (g - img_min) / scale_factor
        g = np.maximum(g, 1e-8) 
        
        H, W = g.shape
        kh, kw = self.kernel_shape
        
        sigma = max(kh, kw) / 15.0
        x_grid = np.linspace(-kh//2, kh//2, kh)
        y_grid = np.linspace(-kw//2, kw//2, kw)
        X, Y = np.meshgrid(x_grid, y_grid)
        h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
        h /= h.sum()
        
        u = g.copy()
        o = g.copy()
        
        beta = self.beta_init
        F_dx, F_dy = get_grad_operators((H, W))
        
        if self.verbose:
            print(f"[{self.name}] Start. Img: {H}x{W}, Ker: {kh}x{kw}")
        
        for k in range(self.max_iter):
            h_prev = h.copy()
            
            h = solve_h_subproblem(u, o, h, beta, self.mu, F_dx, F_dy)

            h = adjust_psf(h)
            
            u = solve_u_subproblem(g, h, o, beta, self.lambda_)
            
            o = solve_o_subproblem(u, h, o, beta, F_dx, F_dy)
            
            ho = np.real(ifft2(psf2otf(h, (H, W)) * fft2(o)))
            diff_norm = np.linalg.norm(ho - u)**2
            
            beta = beta + min(self.T, self.tau * diff_norm)
            
            h_diff = np.linalg.norm(h - h_prev) / (np.linalg.norm(h_prev) + 1e-12)
            self.history['error'].append(diff_norm)
            
            if self.verbose and k % 5 == 0:
                print(f"Iter {k}: beta={beta:.2f}, err={diff_norm:.2e}, dh={h_diff:.2e}")
            
            if h_diff < self.tol and k > 10:
                if self.verbose: print("Converged.")
                break
        
        if self.verbose:
            print("Blind stage finished. Starting non-blind refinement...")
        
        o_final = solve_nonblind_tv(g, h, o, self.xi, self.eta_init, F_dx, F_dy)
        
        o_final = o_final * scale_factor + img_min
        
        self.hyperparams = {
            'lambda': self.lambda_,
            'mu': self.mu,
            'beta_final': beta,
            'iterations': k + 1
        }
        
        return o_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_', self.lambda_),
            ('mu', self.mu),
            ('beta_init', self.beta_init),
            ('xi', self.xi),
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
    algo = PBTVGR(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history