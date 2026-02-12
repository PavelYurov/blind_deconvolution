"""
Blind Image Deconvolution using Hybrid EP-EM Strategy.

This implementation combines two powerful approaches:
1. E-Step (Image Estimation): Uses Half-Quadratic Splitting (HQS) with Spectral Uncertainty approximation.
   This provides stability and speed compared to full variational or EP site updates.
   Based on the provided HQS logic.
2. M-Step (Kernel Estimation): Uses FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
   This provides significantly faster convergence for the kernel than standard PGD.

Reference:
- Abdulaziz, A., et al. "Blind deconvolution of images corrupted by Gaussian noise using Expectation Propagation." EUSIPCO 2021.
- Beck, A., & Teboulle, M. "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM J. Imaging Sci., 2009.
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict, Optional
import sys
import os

# Robust import of base class
def _import_base():
    current = os.path.dirname(os.path.abspath(__file__))
    while len(current) > 3: # Stop at root
        if os.path.exists(os.path.join(current, 'base.py')):
            if current not in sys.path:
                sys.path.append(current)
            return
        current = os.path.dirname(current)

_import_base()

try:
    from base import DeconvolutionAlgorithm
except ImportError:
    # Fallback/Dummy for static analysis or if environment is not set up standardly
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

EPSILON = 1e-10

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts PSF (Point Spread Function) to OTF (Optical Transfer Function).
    Centers the kernel and pads it for FFT.
    """
    in_shape = psf.shape
    # Pad with zeros
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:in_shape[0], :in_shape[1]] = psf
    
    # Circular shift to center the kernel at (0,0) for FFT
    psf_padded = np.roll(psf_padded, -in_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, -in_shape[1] // 2, axis=1)
    
    return fft2(psf_padded)

def otf2psf(otf: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    """
    Inverse conversion from OTF to PSF.
    """
    psf_padded = np.real(ifft2(otf))
    
    # Reverse circular shift
    psf_padded = np.roll(psf_padded, out_shape[0] // 2, axis=0)
    psf_padded = np.roll(psf_padded, out_shape[1] // 2, axis=1)
    
    return psf_padded[:out_shape[0], :out_shape[1]]

class EPEM_Hybrid(DeconvolutionAlgorithm):
    """
    Hybrid EP-EM Blind Deconvolution.
    
    Combines HQS for robust image estimation and FISTA for fast kernel estimation.
    Approximates posterior uncertainty using spectral mean (Fast-Cx strategy).
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 0.003,
        noise_sigma: float = 0.01,
        max_iter: int = 30,
        inner_iter_x: int = 5,
        inner_iter_h: int = 20, # Increased due to FISTA speed
        beta_min: float = 1.0,
        beta_max: float = 1024.0,
        verbose: bool = False
    ):
        """
        Initialize the algorithm.

        Args:
            kernel_shape: Size of the blur kernel (h, w).
            lambda_tv: Weight of TV regularization.
            noise_sigma: Standard deviation of additive Gaussian noise.
            max_iter: Number of outer EM iterations.
            inner_iter_x: Number of iterations for HQS (Image step).
            inner_iter_h: Number of iterations for FISTA (Kernel step).
            beta_min: Starting penalty parameter for HQS.
            beta_max: Maximum penalty parameter for HQS.
            verbose: If True, prints progress.
        """
        super().__init__(name='EP-EM-Hybrid')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter
        self.inner_iter_x = inner_iter_x
        self.inner_iter_h = inner_iter_h
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.verbose = verbose
        
        self.history = {'loss': [], 'kernel_diff': []}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main execution method.
        """
        start_time = time.time()
        
        # Normalize and prepare input
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
            
        H, W = y.shape
        kh, kw = self.kernel_shape
        
        # 1. Initialization
        # Initialize kernel with a Gaussian guess
        sig = max(kh, kw) / 10.0
        grid_y, grid_x = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2 * sig**2))
        h /= h.sum()
        
        # Initialize latent image x with observation y
        x = y.copy()
        
        # Precompute gradient operators in Fourier domain
        dx = np.zeros((H, W))
        dx[0, 0] = -1; dx[0, 1] = 1
        dy = np.zeros((H, W))
        dy[0, 0] = -1; dy[1, 0] = 1
        
        F_dx = fft2(dx)
        F_dy = fft2(dy)
        # |Dx|^2 + |Dy|^2
        F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
        
        if self.verbose:
            print(f"Start EP-EM Hybrid. Image: {H}x{W}, Kernel: {kh}x{kw}, Sigma: {self.noise_sigma}")

        # --- Main EM Loop ---
        for it in range(self.max_iter):
            h_prev = h.copy()
            
            # === E-Step: Image Estimation & Uncertainty ===
            # We seek q(x) ~ N(m_x, C_x).
            # m_x is estimated via HQS (MAP estimate).
            
            m_x = self._estimate_image_hqs(y, h, x, F_dx, F_dy, F_grad_sq)
            
            # Estimate Uncertainty (Trace of C_x)
            # Cov ~ inv( (1/sigma^2)*H^T H + lambda_eff * D^T D )
            # We use spectral approximation (Fast strategy)
            uncertainty_scalar = self._estimate_uncertainty_spectral(h, F_grad_sq, (H, W))
            
            # === M-Step: Kernel Estimation ===
            # Minimize: ||y - m_x * h||^2 + Trace(h^T D_x h)
            # The trace term acts as Tikhonov regularization scaled by uncertainty.
            
            reg_weight = uncertainty_scalar * (H * W)
            
            # Use FISTA for faster convergence than PGD
            h = self._update_kernel_fista(y, m_x, h, reg_weight)
            
            # Convergence check
            diff = np.linalg.norm(h - h_prev) / (np.linalg.norm(h_prev) + EPSILON)
            self.history['kernel_diff'].append(diff)
            
            if self.verbose:
                print(f"Iter {it+1}/{self.max_iter}: |dh|={diff:.6f}, Uncertainty={uncertainty_scalar:.2e}")
            
            # Update x for warm start
            x = m_x
            
            if diff < 1e-4 and it > 3:
                if self.verbose: print("Converged.")
                break

        # Final Non-blind Deconvolution Step
        # Run HQS with tighter constraints/more iterations to get the final sharp image
        if self.verbose: print("Final Non-blind Deconvolution...")
        x_final = self._estimate_image_hqs(
            y, h, x, F_dx, F_dy, F_grad_sq, 
            final_pass=True
        )
        
        self.hyperparams = {
            'lambda_tv': self.lambda_tv,
            'noise_sigma': self.noise_sigma,
            'final_uncertainty': uncertainty_scalar,
            'iterations': it + 1
        }
        
        return x_final, h

    def _estimate_image_hqs(
        self, 
        y: np.ndarray, 
        h: np.ndarray, 
        x_init: np.ndarray, 
        F_dx: np.ndarray, 
        F_dy: np.ndarray, 
        F_grad_sq: np.ndarray,
        final_pass: bool = False
    ) -> np.ndarray:
        """
        Solves the sub-problem for x (E-step mean) using Half-Quadratic Splitting.
        Model: argmin ||y - h*x||^2 / (2*sigma^2) + lambda_tv * ||grad x||_1
        """
        H, W = y.shape
        x = x_init.copy()
        
        # Precompute in Fourier
        F_y = fft2(y)
        F_h = psf2otf(h, (H, W))
        F_h_conj = np.conj(F_h)
        F_h_sq = np.abs(F_h)**2
        
        # HQS Parameters
        beta = self.beta_min
        beta_step = 2.0
        # For final pass, we want to go deeper into the penalty method
        max_beta = self.beta_max if final_pass else (self.beta_max / 2)
        
        # Auxiliary variables for gradients (z_x, z_y)
        z_x = np.zeros_like(x)
        z_y = np.zeros_like(x)
        
        # Data fidelity weight
        alpha = 1.0 / (self.noise_sigma**2)
        
        while beta < max_beta:
            # Number of inner iterations can be small for blind stages
            iter_count = 5 if final_pass else self.inner_iter_x
            
            for _ in range(iter_count):
                # 1. x-update (FFT solution of linear system)
                # (alpha * H^T H + beta * D^T D) x = alpha * H^T y + beta * D^T z
                
                rhs = alpha * F_h_conj * F_y + beta * (
                    np.conj(F_dx) * fft2(z_x) + np.conj(F_dy) * fft2(z_y)
                )
                lhs = alpha * F_h_sq + beta * F_grad_sq
                
                x = np.real(ifft2(rhs / (lhs + EPSILON)))
                x = np.maximum(x, 0.0) # Projection to non-negative constraint
                
                # 2. z-update (Soft Thresholding)
                # z = argmin beta/2 ||Du - z||^2 + lambda ||z||_1
                
                # Calculate gradients of current x
                grad_x = np.real(ifft2(F_dx * fft2(x)))
                grad_y = np.real(ifft2(F_dy * fft2(x)))
                
                threshold = self.lambda_tv / beta
                z_x = self._soft_threshold(grad_x, threshold)
                z_y = self._soft_threshold(grad_y, threshold)
            
            beta *= beta_step
            
        return x

    def _estimate_uncertainty_spectral(
        self, 
        h: np.ndarray, 
        F_grad_sq: np.ndarray, 
        shape: Tuple[int, int]
    ) -> float:
        """
        Estimates the mean pixel variance (EP update).
        Uses diagonal approximation of the inverse Hessian in Fourier domain.
        
        Cov ~ ( (1/sigma^2) H^T H + lambda_eff * D^T D )^-1
        """
        H, W = shape
        F_h = psf2otf(h, (H, W))
        
        alpha = 1.0 / (self.noise_sigma**2)
        
        # Effective lambda for TV. 
        # TV behaves like Laplacian with weight depending on local gradient.
        # We use a heuristic scaling for the spectral approximation.
        lambda_eff = self.lambda_tv * 100.0 
        
        # Spectrum of Inverse Hessian
        inv_hessian_spectrum = 1.0 / (alpha * np.abs(F_h)**2 + lambda_eff * F_grad_sq + EPSILON)
        
        # Mean variance (Sum in Fourier = Sum in Time, divided by N for mean)
        mean_variance = np.mean(inv_hessian_spectrum)
        
        return float(mean_variance)

    def _update_kernel_fista(
        self, 
        y: np.ndarray, 
        x: np.ndarray, 
        h_init: np.ndarray, 
        reg_weight: float
    ) -> np.ndarray:
        """
        Solves M-step for kernel h using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
        Minimizes: ||y - x*h||^2 + reg_weight * ||h||^2
        Subject to: h >= 0, sum(h) = 1
        """
        H, W = y.shape
        kh, kw = h_init.shape
        
        # Precompute in Fourier
        F_x = fft2(x)
        F_y = fft2(y)
        
        # Autocorrelation of X and Cross-correlation XY
        # These are constant during the h-update
        F_xx = np.abs(F_x)**2
        F_xy = np.conj(F_x) * F_y
        
        # Lipsitz constant for step size
        # L <= 2 * max(eig(X^T X)) + 2 * reg_weight
        lipschitz = 2.0 * np.max(F_xx) + 2.0 * reg_weight
        step_size = 1.0 / (lipschitz + EPSILON)
        
        # Initialization for FISTA
        zk = h_init.copy() # Momentum variable
        xk = h_init.copy() # Current estimate
        tk = 1.0           # Step size multiplier
        
        for _ in range(self.inner_iter_h):
            xk_prev = xk.copy()
            
            # 1. Gradient Step on auxiliary variable zk
            # Grad = 2 * (X^T X h - X^T y) + 2 * reg * h
            
            F_zk = psf2otf(zk, (H, W))
            grad_freq = 2.0 * (F_xx * F_zk - F_xy)
            grad_spatial = np.real(ifft2(grad_freq))
            
            # Crop and center gradient
            grad_spatial = np.roll(grad_spatial, kh//2, axis=0)
            grad_spatial = np.roll(grad_spatial, kw//2, axis=1)
            grad_h = grad_spatial[:kh, :kw]
            
            # Add regularization gradient
            grad_h += 2.0 * reg_weight * zk
            
            # Gradient Descent
            zk_new = zk - step_size * grad_h
            
            # 2. Projection (Proximal Operator)
            # Non-negativity constraint
            xk = np.maximum(zk_new, 0.0)
            
            # 3. FISTA Momentum Update
            tk_new = (1.0 + np.sqrt(1.0 + 4.0 * tk**2)) / 2.0
            zk = xk + ((tk - 1.0) / tk_new) * (xk - xk_prev)
            tk = tk_new
            
        # Final Normalization (Sum = 1)
        h_sum = np.sum(xk)
        if h_sum > EPSILON:
            xk /= h_sum
        else:
            xk = np.ones_like(xk) / (kh * kw)
            
        return xk

    def _soft_threshold(self, x: np.ndarray, thresh: float) -> np.ndarray:
        """Soft thresholding operator: sign(x) * max(|x| - thresh, 0)"""
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_tv', self.lambda_tv),
            ('noise_sigma', self.noise_sigma),
            ('max_iter', self.max_iter),
            ('beta_max', self.beta_max)
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

# Wrapper functions for compatibility
def epem_hybrid_deconvolution(g, kernel_shape, **kwargs):
    algo = EPEM_Hybrid(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history

def run_algorithm(g, kernel_shape, **kwargs):
    return epem_hybrid_deconvolution(g, kernel_shape, **kwargs)
