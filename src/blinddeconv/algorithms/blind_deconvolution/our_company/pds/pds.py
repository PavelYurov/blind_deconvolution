"""
Blind Image Deconvolution via Primal-Dual Splitting (PDS).
Adaptation of O'Connor (2015) and Condat (2013).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict
import sys
from pathlib import Path

from .utils import (
    build_gradient_filters,
    init_kernel,
    resize_image,
    resize_kernel,
    center_kernel,
    threshold_kernel,
    edgetaper,
    EPSILON,
    project_simplex
)
from .solvers import solve_image_pds, solve_kernel_pds

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

class PDS(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 5e-3,        
        noise_sigma: float = 0.03,
        huber_delta: float = 0.04,
        max_iter: int = 20,
        image_pds_iter: int = 80,
        kernel_pds_iter: int = 80,
        num_scales: int = 3,            
        rho: float = 1.0,               
        theta: float = 1.0,            
        kernel_threshold: float = 0.02, 
        verbose: bool = True,
    ):
        super().__init__(name="PDS-OConnor")
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.noise_sigma = noise_sigma
        self.huber_delta = huber_delta
        self.max_iter = max_iter
        self.image_pds_iter = image_pds_iter
        self.kernel_pds_iter = kernel_pds_iter
        self.num_scales = num_scales
        self.rho = rho
        self.theta = theta
        self.kernel_threshold = kernel_threshold
        self.verbose = verbose
        
        self.history: Dict[str, list] = {"kernel_diff": []}
        self.hyperparams: Dict[str, Any] = {}

    def _build_scales(self, image_shape, kernel_shape):
        """ Creates multiscale pyramid dimensions. """
        H, W = image_shape
        kh, kw = kernel_shape
        
        min_dim = max(kh, kw) * 1.5
        if self.num_scales <= 1 or min(H, W) < min_dim:
            return [(image_shape, kernel_shape)]
        
        factors = np.geomspace(0.5, 1.0, self.num_scales)
        scales = []
        for s in factors:
            Hs, Ws = int(H * s), int(W * s)
            if Hs < min_dim or Ws < min_dim: continue
            
            khs, kws = int(kh * s), int(kw * s)
            if khs % 2 == 0: khs += 1
            if kws % 2 == 0: kws += 1
            scales.append(((Hs, Ws), (khs, kws)))

        if scales[-1][0] != image_shape:
            scales.append((image_shape, kernel_shape))
            
        return scales

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y_original = image.astype(np.float64)
        if y_original.max() > 1.0:
            y_original /= 255.0

        y = edgetaper(y_original, self.kernel_shape)
        
        H, W = y.shape
        scales = self._build_scales((H, W), self.kernel_shape)
        
        _, ks_coarse = scales[0]
        h = init_kernel(ks_coarse, mode='gaussian')
        x = None 
        
        alpha_base = 1.0 / (self.noise_sigma ** 2 + EPSILON)


        for level, (img_shape, ker_shape) in enumerate(scales):

            y_s = resize_image(y, img_shape) if img_shape != (H, W) else y
            

            if h.shape != ker_shape:
                h = resize_kernel(h, ker_shape)
                h = center_kernel(h)
            
            if x is None:
                x_s = y_s.copy()
            else:
                x_s = resize_image(x, img_shape)
            
            F_dh, F_dv = build_gradient_filters(img_shape)
            
            scale_ratio = img_shape[0] / H

            lambda_tv_s = self.lambda_tv / (scale_ratio ** 0.5)
            

            huber_delta_s = self.huber_delta * scale_ratio 
            
            if self.verbose:
                print(f"Scale {level+1}/{len(scales)}: {img_shape}, K:{ker_shape}")
                print(f"  Params: TV={lambda_tv_s:.4f}, Huber={huber_delta_s:.4f}")


            for it in range(self.max_iter):
                h_prev = h.copy()
                

                x_s = solve_image_pds(
                    y_s, h, x_s,
                    alpha_base, lambda_tv_s,
                    num_iter=self.image_pds_iter,
                    huber_delta=huber_delta_s,
                    F_dh=F_dh, F_dv=F_dv,
                    rho=self.rho
                )
                
                if np.any(np.isnan(x_s)):
                    print("  [Warning] NaN in image step. Resetting.")
                    x_s = y_s.copy()

                h = solve_kernel_pds(
                    y_s, x_s, h,
                    alpha=alpha_base * 0.1, 
                    num_iter=self.kernel_pds_iter,
                    theta=self.theta,
                    kernel_threshold=self.kernel_threshold
                )
                
                diff = np.linalg.norm(h - h_prev)
                self.history["kernel_diff"].append(float(diff))
                
                if self.verbose and (it % 5 == 0):
                    print(f"    Iter {it}: dH={diff:.6f}")
                    
                if diff < 1e-6 and it > 5:
                    break
            
            x = x_s 

        
        if self.verbose: print("Final Non-Blind Deconvolution...")
        

        h = center_kernel(h)
        h = threshold_kernel(h, max(0.01, self.kernel_threshold))
        h = project_simplex(h)
        
        F_dh, F_dv = build_gradient_filters((H, W))

        x_final = solve_image_pds(
            y, h, x,
            alpha=alpha_base, 
            lambda_tv=self.lambda_tv * 0.8,
            huber_delta=self.huber_delta,
            num_iter=self.image_pds_iter * 3,
            F_dh=F_dh, F_dv=F_dv,
            rho=self.rho
        )


        x_final = np.clip(x_final, 0, 1) * 255.0
        x_final = np.round(x_final).astype(np.uint8)
        
        self.hyperparams["elapsed"] = time.time() - start_time
        return x_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("lambda_tv", self.lambda_tv),
            ("noise_sigma", self.noise_sigma),
            ("huber_delta", self.huber_delta),
            ("max_iter", self.max_iter),
            ("scales", self.num_scales),
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

def run_algorithm(g: np.ndarray, kernel_shape: Tuple[int, int], **kwargs) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    algo = PDS(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history