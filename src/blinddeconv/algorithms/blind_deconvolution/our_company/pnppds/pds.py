import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Any, Dict, List

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

try:
    from models.denoiser import Denoiser
except ImportError as e:
    try:
        from .models.denoiser import Denoiser
    except ImportError:
        raise ImportError(f"Error importing models: {e}")

from blinddeconv.algorithms.base import DeconvolutionAlgorithm

from .solvers import solve_image_pds as solve_image_tv, solve_kernel_pds
from .utils import (
    build_gradient_filters, init_kernel, resize_image, resize_kernel, 
    center_kernel, threshold_kernel, edgetaper, project_simplex
)
from .solvers_ml import solve_image_pnp_pds
from .utils_ml import align_kernel_and_image

class PnP_PDS_Blind(DeconvolutionAlgorithm):
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # Math/TV Params (Kernel Estimation) — defaults aligned with original PDS
        lambda_tv: float = 5e-3,
        huber_delta: float = 0.04,
        # Common Params
        noise_sigma: float = 0.01,
        # ML Params (aligned with original PnP-PDS paper defaults)
        model_path: str = str(_CURRENT_FILE.parent / 'nn/DnCNN_nobn_nch_1_nlev_0.01.pth'), 
        alpha_n: float = 0.82,
        gamma1: float = 0.5,
        gamma2: float = 0.99,
        # Iteration Params
        num_scales: int = 3,
        max_iter: int = 20,
        tv_inner_iter: int = 80,
        kernel_inner_iter: int = 80,
        ml_final_iter: int = 200,
        # PDS solver params (from original PDS)
        rho: float = 1.0,
        theta: float = 1.0,
        kernel_threshold: float = 0.02,
        
        channels: int = 1,
        verbose: bool = False
    ):
        super().__init__(name='PnP-PDS-Hybrid')
        self.kernel_shape = tuple(kernel_shape)
        self.model_path = model_path
        self.noise_sigma = noise_sigma
        
        # Params for Math loop (kernel estimation via TV-PDS)
        self.lambda_tv = lambda_tv
        self.huber_delta = huber_delta
        self.num_scales = num_scales
        self.max_iter = max_iter
        self.tv_inner_iter = tv_inner_iter
        self.kernel_inner_iter = kernel_inner_iter
        self.rho = rho
        self.theta = theta
        self.kernel_threshold = kernel_threshold
        
        # ML Params template (for final PnP-PDS restoration)
        self.ml_params_template = {
            'gamma1': gamma1,
            'gamma2': gamma2,           # gamma1*gamma2 < 1 required for convergence
            'alpha_n': alpha_n,
            'ml_iter': ml_final_iter
        }
        
        self.channels = channels
        self.verbose = verbose
        
        self.denoiser = None
        self.history = {'kernel_diff': []}
        self.hyperparams = {}

    def _load_model(self):
        if self.denoiser is None:
            if Denoiser is None:
                return 

            weight_path = Path(self.model_path)
            if weight_path.exists():
                final_path = str(weight_path)
            else:
                alt_path = _PROJECT_ROOT / self.model_path
                if alt_path.exists():
                    final_path = str(alt_path)
                else:
                    print(f"Warning: Model weights not found. Skipping ML.")
                    return
            
            self.denoiser = Denoiser(file_name=final_path, ch=self.channels)
    
    def _build_scales(self, image_shape):
        H, W = image_shape
        kh, kw = self.kernel_shape
        min_dim = max(kh, kw) * 1.5
        if self.num_scales <= 1 or min(H, W) < min_dim:
            return [(image_shape, self.kernel_shape)]
        
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
            scales.append((image_shape, self.kernel_shape))
        return scales

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._load_model()
        start_time = time.time()
        
        # 1. Prepare Data
        y_original = image.astype(np.float64)
        if y_original.max() > 1.0:
            y_original /= 255.0
        
        # Convert to grayscale FIRST (edgetaper requires 2D), then edgetaper
        if y_original.ndim == 3:
            y_gray = np.mean(y_original, axis=2)
        else:
            y_gray = y_original.copy()
            
        y = edgetaper(y_gray, self.kernel_shape)
            
        H, W = y.shape
        scales = self._build_scales((H, W))
        
        # 2. Initialize Kernel
        _, ks_coarse = scales[0]
        h = init_kernel(ks_coarse, mode='gaussian')
        x_curr = None 
        
        from .utils import EPSILON
        alpha_base = 1.0 / (self.noise_sigma ** 2 + EPSILON)

        # 1. Kernel Estimation (Math TV-PDS)

        for level, (img_shape, ker_shape) in enumerate(scales):
            
            y_s = resize_image(y, img_shape) if img_shape != (H, W) else y
            
            if h.shape != ker_shape:
                h = resize_kernel(h, ker_shape)
                h = center_kernel(h)
            
            if x_curr is None:
                x_s = y_s.copy()
            else:
                x_s = resize_image(x_curr, img_shape)
            
            F_dh, F_dv = build_gradient_filters(img_shape)
            
            scale_ratio = img_shape[0] / H
            # Stronger TV at coarser scales
            lambda_tv_s = self.lambda_tv / (scale_ratio ** 0.5)
            huber_delta_s = self.huber_delta * scale_ratio 
            
            if self.verbose:
                print(f"Scale {level+1}/{len(scales)} ({img_shape}): TV_reg={lambda_tv_s:.5f}, Huber={huber_delta_s:.4f}")

            for it in range(self.max_iter):
                h_prev = h.copy()
                
                # A. Update Image (TV-PDS)
                x_s = solve_image_tv(
                    y_s, h, x_s,
                    alpha_base, lambda_tv_s,
                    huber_delta_s,
                    num_iter=self.tv_inner_iter,
                    F_dh=F_dh, F_dv=F_dv,
                    rho=self.rho
                )
                
                # NaN safety check (from original PDS)
                if np.any(np.isnan(x_s)):
                    if self.verbose:
                        print("  [Warning] NaN in image step. Resetting.")
                    x_s = y_s.copy()
                
                # B. Update Kernel (Chambolle-Pock)
                h = solve_kernel_pds(
                    y_s, x_s, h,
                    alpha=alpha_base * 0.1, 
                    num_iter=self.kernel_inner_iter,
                    theta=self.theta,
                    kernel_threshold=self.kernel_threshold
                )
                
                # C. Align kernel and image to prevent drift
                h, x_s = align_kernel_and_image(h, x_s)
                
                diff = np.linalg.norm(h - h_prev)
                self.history["kernel_diff"].append(float(diff))
                
                if self.verbose and (it % 5 == 0):
                    print(f"    Iter {it}: dH={diff:.6f}")
                    
                if diff < 1e-6 and it > 5:
                    break
            
            x_curr = x_s 

        # Final kernel cleanup
        h = center_kernel(h)
        h = threshold_kernel(h, max(0.01, self.kernel_threshold))
        h = project_simplex(h)

        # 2. Image Restoration (ML PnP-PDS)
        if self.verbose: 
            print("Final Restoration with DnCNN...")

        # Prepare input in (C, H, W) format required by DnCNN
        if y_original.ndim == 3:
            # Color: (H, W, C) -> (C, H, W)
            y_final_in = np.moveaxis(y_original, -1, 0)
            x_init_in = np.moveaxis(y_original, -1, 0).copy()
        else:
            # Grayscale: (H, W) -> (1, H, W) — channel dim required by DnCNN
            y_final_in = y_original[np.newaxis, ...]
            x_init_in = y_original[np.newaxis, ...].copy()

        if self.denoiser:
            ml_params = self.ml_params_template.copy()
            ml_params['noise_sigma'] = self.noise_sigma 
            
            x_out_raw = solve_image_pnp_pds(
                y_obs=y_final_in,
                k_curr=h,
                x_init=x_init_in,
                denoiser_model=self.denoiser,
                params=ml_params
            )
        else:
            # Fallback to TV restoration
            if self.verbose:
                print("Warning: No denoiser loaded. Falling back to TV restoration.")
            F_dh, F_dv = build_gradient_filters((H, W))
            x_fallback = solve_image_tv(
                y, h, x_curr if x_curr is not None else y.copy(),
                alpha=alpha_base,
                lambda_tv=self.lambda_tv * 0.8,
                huber_delta=self.huber_delta,
                num_iter=self.tv_inner_iter * 3,
                F_dh=F_dh, F_dv=F_dv,
                rho=self.rho
            )
            # Convert to same format as ML output
            if y_original.ndim == 3:
                x_out_raw = np.stack([x_fallback]*3, axis=0)
            else:
                x_out_raw = x_fallback[np.newaxis, ...]

        # Output formatting: convert back from (C, H, W) to original shape
        if y_original.ndim == 3:
            x_out = np.moveaxis(x_out_raw, 0, -1)   # (C,H,W) -> (H,W,C)
        else:
            x_out = x_out_raw[0]                      # (1,H,W) -> (H,W)
            
        x_out = np.clip(x_out, 0, 1) * 255.0
        x_out = np.round(x_out).astype(np.uint8)
        
        self.hyperparams["elapsed"] = time.time() - start_time
        return x_out, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_shape", self.kernel_shape),
            ("lambda_tv", self.lambda_tv),
            ("noise_sigma", self.noise_sigma),
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