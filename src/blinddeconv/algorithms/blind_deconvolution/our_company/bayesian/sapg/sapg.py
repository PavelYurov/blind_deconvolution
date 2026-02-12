import numpy as np
import time
from typing import Tuple, List, Any, Dict
from .utils import precompute_gradient_operators, compute_spatial_gradient, gaussian_psf, project_param
from .solvers import myula_sampler, sapg_update_theta, sapg_update_alpha, sapg_update_sigma2, tv_prox, data_fidelity_grad, solve_image_hqs

# Robust import of base class
try:
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class SAPG(DeconvolutionAlgorithm):
    """
    Semi-Blind Image Deconvolution using Stochastic Approximation Proximal Gradient (SAPG).
    
    Implements the strategy described in Mbakam et al. (2024):
    1. Hyperparameter Estimation: SAPG with MYULA MCMC for marginal likelihood gradients.
    2. Image Estimation: Proximal Gradient (e.g., HQS) for MAP.
    Assumes Gaussian PSF parameterized by alpha (variance), TV prior with theta.
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (5, 5),
        theta_init: float = 1.0,
        sigma2_init: float = 0.0025,
        alpha_init: float = 1.0,
        max_iter: int = 50,
        # Sampler params
        myula_gamma: float = 1e-3,
        myula_lam: float = 1e-2,
        m_per_iter: int = 100,
        burn_in: int = 50,
        # SAPG params
        delta_base: float = 1e-2,
        strategy: str = 'decreasing',  # 'constant' or 'decreasing'
        # Bounds
        theta_bounds: Tuple[float, float] = (1e-3, 1.0),
        alpha_bounds: Tuple[float, float] = (0.1, 10.0),
        sigma2_bounds: Tuple[float, float] = (1e-4, 1.0),
        verbose: bool = False
    ):
        super().__init__(name='SAPG-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.theta = theta_init  # Regularization param
        self.sigma2 = sigma2_init
        self.alpha = alpha_init  # Blur param
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Internal settings
        self.myula_gamma = myula_gamma
        self.myula_lam = myula_lam
        self.m_per_iter = m_per_iter
        self.burn_in = burn_in
        self.delta_base = delta_base
        self.strategy = strategy
        self.theta_bounds = theta_bounds
        self.alpha_bounds = alpha_bounds
        self.sigma2_bounds = sigma2_bounds
        
        self.history = {'theta': [], 'alpha': [], 'sigma2': []}
        self.hyperparams = {}
        self.deltas = []

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # 1. Prepare Data
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
        
        H, W = y.shape
        d = H * W
        x = y.copy()  # Initial image estimate
        
        # Precompute for TV/HQS
        F_ops = precompute_gradient_operators((H, W))
        
        if self.verbose:
            print(f"[{self.name}] Start. Img: {H}x{W}, Init alpha: {self.alpha}, sigma2: {self.sigma2}")

        # 3. Main SAPG Loop for hyperparameters
        for it in range(self.max_iter):
            # Compute step size
            if self.strategy == 'decreasing':
                delta = self.delta_base / (it + 1)**0.6  # a=0.6 as suggested
            else:
                delta = self.delta_base
            self.deltas.append(delta)
            
            # A. Sample from posterior p(x | y, theta, alpha, sigma2)
            samples_post = myula_sampler(
                y, self.alpha, self.sigma2, self.theta, x,
                self.myula_gamma, self.myula_lam, self.m_per_iter, self.burn_in,
                gaussian_psf, tv_prox, data_fidelity_grad
            )
            
            # B. Sample from prior p(x | theta) - set grad_f=0
            def zero_grad(*args): return np.zeros_like(x)
            samples_prior = myula_sampler(
                y, self.alpha, self.sigma2, self.theta, np.random.randn(H, W),
                self.myula_gamma, self.myula_lam, self.m_per_iter, self.burn_in,
                gaussian_psf, tv_prox, zero_grad
            )
            
            # C. Update hyperparameters
            self.theta = sapg_update_theta(samples_post, samples_prior, self.theta, delta, d, 1.0, self.theta_bounds)
            self.alpha = sapg_update_alpha(samples_post, y, self.alpha, self.sigma2, delta, gaussian_psf, gaussian_psf_deriv_alpha, self.alpha_bounds)
            self.sigma2 = sapg_update_sigma2(samples_post, y, self.alpha, self.sigma2, delta, d, gaussian_psf, self.sigma2_bounds)
            
            # Monitoring
            self.history['theta'].append(self.theta)
            self.history['alpha'].append(self.alpha)
            self.history['sigma2'].append(self.sigma2)
            
            if self.verbose:
                print(f"Iter {it+1}/{self.max_iter}: theta={self.theta:.4f}, alpha={self.alpha:.4f}, sigma2={self.sigma2:.4f}")
            
            # Warm start x with mean of samples
            x = np.mean(samples_post, axis=0)
        
        # Weighted averages for final hyperparameters
        weights = np.array(self.deltas)
        sum_weights = np.sum(weights)
        bar_theta = np.sum(weights * np.array(self.history['theta'])) / sum_weights
        bar_alpha = np.sum(weights * np.array(self.history['alpha'])) / sum_weights
        bar_sigma2 = np.sum(weights * np.array(self.history['sigma2'])) / sum_weights
        
        # 4. Final MAP for x
        if self.verbose: print("Final MAP Estimation...")
        h_final = gaussian_psf(bar_alpha, self.kernel_shape)
        x_final = solve_image_hqs(
            y, h_final, x, 
            np.sqrt(bar_sigma2), bar_theta, 
            1024.0, 10, F_ops
        )
        
        self.hyperparams = {
            'theta': bar_theta,
            'alpha': bar_alpha,
            'sigma2': bar_sigma2,
            'iterations': it + 1
        }
        
        return x_final, h_final

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('theta_init', self.theta),
            ('sigma2_init', self.sigma2),
            ('alpha_init', self.alpha),
            ('max_iter', self.max_iter)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_history(self) -> dict:
        return self.history
    
    def get_hyperparams(self) -> dict:
        return self.hyperparams

def run_algorithm(g, kernel_shape, **kwargs):
    algo = SAPG(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history