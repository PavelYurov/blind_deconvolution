"""
Blind Image Deconvolution Algorithm Wrapper.
Implementation of Perrone & Favaro (2015).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict
import cv2

import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path: raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path

_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path: sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm
from .utils import resize_image, resize_kernel, center_kernel
from .solvers import solve_image_primal_dual, solve_kernel_pgd

class LogTotalVariationBlindDeconv(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # Lambda needs to be high for float images (err ~ 1e-4) to balance with Log(~1)
        lambda_reg: float = 5000.0, 
        epsilon: float = 1e-2, 
        p_norm: float = 0.5,        
        num_scales: int = 5,        
        scale_factor: float = 0.7,  # Gentle downsampling
        outer_iters: int = 5,
        image_iters: int = 50,
        kernel_iters: int = 30,
    ):
        super().__init__(name='LogTV-Perrone2015')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        self.p_norm = p_norm
        self.num_scales = num_scales
        self.scale_factor = scale_factor
        self.outer_iters = outer_iters
        self.image_iters = image_iters
        self.kernel_iters = kernel_iters
        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # Normalize to [0, 1]
        y = image.astype(np.float32)
        if y.max() > 1.0: y /= 255.0
        
        # Build Pyramid
        pyramid_y = []
        curr = y
        for _ in range(self.num_scales):
            pyramid_y.append(curr)
            curr = resize_image(curr, self.scale_factor)
        pyramid_y = pyramid_y[::-1] # Coarse to fine

        # Initialize Kernel
        # Use a small Gaussian to break symmetry but keep it tight
        # Do not initialize with 3x3 at coarsest if coarsest is very small
        kh_final, kw_final = self.kernel_shape
        
        # Initial kernel guess (3x3)
        k_curr = np.zeros((3, 3), dtype=np.float32)
        k_curr[1, 1] = 1.0 
        # Slight blur to allow gradients to flow
        k_curr = cv2.GaussianBlur(k_curr, (3, 3), 0.5)
        k_curr /= k_curr.sum()
        
        u_curr = pyramid_y[0].copy()

        for scale_idx, y_level in enumerate(pyramid_y):
            H_l, W_l = y_level.shape
            
            # Update dimensions for this level
            ratio = self.scale_factor ** (self.num_scales - 1 - scale_idx)
            kh_l = int(kh_final * ratio)
            kw_l = int(kw_final * ratio)
            
            # Ensure odd and minimum size 3
            kh_l = max(3, kh_l if kh_l % 2 != 0 else kh_l + 1)
            kw_l = max(3, kw_l if kw_l % 2 != 0 else kw_l + 1)
            
            if scale_idx > 0:
                k_curr = resize_kernel(k_curr, max(kh_l, kw_l))
                u_curr = cv2.resize(u_curr, (W_l, H_l), interpolation=cv2.INTER_CUBIC)
                u_curr = np.clip(u_curr, 0, 1)
            
            # Center kernel at start of scale
            k_curr = center_kernel(k_curr)

            # Alternating Minimization
            for it in range(self.outer_iters):
                # 1. Image Update (Primal Dual)
                u_curr = solve_image_primal_dual(
                    y_level, k_curr, u_curr,
                    lambda_reg=self.lambda_reg,
                    epsilon=self.epsilon,
                    p=self.p_norm,
                    pd_iter=self.image_iters
                )
                
                # 2. Kernel Update (PGD in Gradient Domain)
                k_curr = solve_kernel_pgd(
                    y_level, u_curr, k_curr,
                    iters=self.kernel_iters
                )
                
                # 3. Center Kernel
                k_curr = center_kernel(k_curr)
                
        # Final formatting
        if k_curr.shape != self.kernel_shape:
            k_curr = resize_kernel(k_curr, max(self.kernel_shape))
            # Crop to exact center
            h, w = k_curr.shape
            th, tw = self.kernel_shape
            start_h = (h - th) // 2
            start_w = (w - tw) // 2
            k_curr = k_curr[start_h:start_h+th, start_w:start_w+tw]
            k_curr /= k_curr.sum()

        self.hyperparams['time'] = time.time() - start_time
        
        # One last high-quality non-blind deconv can be done here if needed
        # but process() returns the blind result.
        
        x_final = np.clip(u_curr * 255.0, 0, 255).astype(np.uint8)
        return x_final, k_curr

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda', self.lambda_reg),
            ('epsilon', self.epsilon),
            ('p', self.p_norm),
            ('scales', self.num_scales),
            ('outer', self.outer_iters)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_history(self) -> dict: return self.history
    def get_hyperparams(self) -> dict: return self.hyperparams