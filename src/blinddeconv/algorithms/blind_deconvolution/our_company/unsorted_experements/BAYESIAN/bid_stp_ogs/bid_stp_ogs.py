import numpy as np
import time
from typing import Tuple, List, Any, Dict

# ── Robust import of the framework base class ──────────────────────
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up the directory tree until ``pyproject.toml`` is found."""
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
from .solvers import TzikasVB2009Solver
from .utils import wiener_deconvolution, tikhonov_deconvolution, edgetaper, keep_largest_component, force_center_mass
from scipy.ndimage import zoom

# ===================================================================
# Main algorithm class
# ===================================================================

class BID_STP_OGS(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using Student-t Priors (STP).
    """
    def __init__(
        self,
        kernel_size: int = 35,
        scales: int = 3,
        scale_factor: float = 0.5,
        max_iter: int = 40,
        cg_iter: int = 20,
        gamma_val: float = 100.0,
        alpha_val: float = 1e-8,
        beta_val: float = 0.01,
        init_beta: float = 100.0,
        kernel_threshold: float = 0.1, 
        
        # New parameters for final step
        final_method: str = "wiener", # "wiener" or "tikhonov"
        final_param: float = 50.0,    # SNR for Wiener (e.g. 50.0), Alpha for Tikhonov (e.g. 0.01)
        
        verbose: bool = True
    ):
        super().__init__(name="BID-STP-OGS")
        
        self.kernel_size = kernel_size
        self.scales = scales
        self.scale_factor = scale_factor
        self.max_iter = max_iter
        self.cg_iter = cg_iter
        
        self.gamma_ab = (1.0, gamma_val)
        self.alpha_ab = (1e-8, alpha_val)
        self.beta_ab = (1e-2, beta_val)
        self.init_beta = init_beta
        self.kernel_threshold = kernel_threshold
        
        self.final_method = final_method
        self.final_param = final_param
        
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {k: v for k, v in locals().items() if k != 'self' and k != '__class__'}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_in = image.astype(np.float32)
        if img_in.max() > 1.0: img_in /= 255.0
        
        if self.verbose:
            print(f"=== BID-STP-OGS: Edgetaper (Blind) -> Padding (Final) ===")
        
        start_time = time.time()
        
        # 1. BLIND ESTIMATION (Uses Edgetaper internally in solver)
        # We perform estimation on the original image. The solver handles boundary artifacts
        # for estimation using edgetaper, which is robust for kernel finding.
        pyramid_imgs = [img_in]
        for _ in range(self.scales - 1):
            pyramid_imgs.append(zoom(pyramid_imgs[-1], self.scale_factor, order=1))
        pyramid_imgs = pyramid_imgs[::-1]
        
        current_kernel = None
        
        for level, img_level in enumerate(pyramid_imgs):
            scale_ratio = self.scale_factor ** (self.scales - 1 - level)
            k_sz = int(np.ceil(self.kernel_size * scale_ratio))
            if k_sz % 2 == 0: k_sz += 1
            k_sz = max(3, k_sz)
            
            if self.verbose:
                print(f"\n[Scale {level+1}] Kernel: {k_sz}x{k_sz}")
            
            if current_kernel is not None:
                current_kernel = zoom(current_kernel, k_sz / current_kernel.shape[0], order=1)
                current_kernel = np.maximum(current_kernel, 0)
                current_kernel /= current_kernel.sum()
            
            solver = TzikasVB2009Solver(
                kernel_shape=(k_sz, k_sz),
                max_iter=self.max_iter,
                cg_iter=self.cg_iter,
                gamma_ab=self.gamma_ab,
                alpha_ab=self.alpha_ab,
                beta_ab=self.beta_ab,
                init_beta=self.init_beta,
                kernel_threshold=self.kernel_threshold,
                verbose=self.verbose
            )
            
            _, current_kernel = solver.solve(img_level, init_kernel=current_kernel)
            
        if current_kernel.shape[0] != self.kernel_size:
            current_kernel = zoom(current_kernel, self.kernel_size / current_kernel.shape[0], order=1)
            current_kernel /= current_kernel.sum()

        # 2. FINAL KERNEL CLEANUP
        current_kernel = keep_largest_component(current_kernel, threshold=self.kernel_threshold)
        current_kernel /= (current_kernel.sum() + 1e-12)
        current_kernel = force_center_mass(current_kernel, threshold=0.0)

        # 3. FINAL RESTORATION (Padding Strategy)
        # We employ "Pad -> Deconvolve -> Crop" to avoid gray borders from edgetaper.
        
        pad_w = self.kernel_size 
        # 'reflect' mode extends texture naturally, preventing ringing at new borders
        img_padded = np.pad(img_in, ((pad_w, pad_w), (pad_w, pad_w)), mode='reflect')
        
        if self.final_method == "tikhonov":
            # Tikhonov usually needs a small alpha, e.g., 0.01 to 0.05
            # final_param here is alpha.
            if self.verbose: print(f"Final Step: Tikhonov (alpha={self.final_param})")
            restored_padded = tikhonov_deconvolution(img_padded, current_kernel, alpha=self.final_param)
            
        else: # "wiener"
            # Wiener uses SNR, e.g., 50.0 to 100.0
            # final_param here is SNR.
            if self.verbose: print(f"Final Step: Wiener (SNR={self.final_param})")
            restored_padded = wiener_deconvolution(img_padded, current_kernel, snr=self.final_param)
        
        # 4. CROP
        restored_float = restored_padded[pad_w:-pad_w, pad_w:-pad_w]
        
        elapsed = time.time() - start_time
        self.history["runtime"] = elapsed
        
        restored_img = np.clip(restored_float * 255.0, 0.0, 255.0).astype(np.int16)
        
        return restored_img, current_kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return list(self.hyperparams.items())

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in ["gamma_val", "alpha_val", "beta_val"]:
                self.gamma_ab = (1.0, self.gamma_val)
                self.alpha_ab = (1e-8, self.alpha_val)
                self.beta_ab = (1e-2, self.beta_val)
            self.hyperparams[key] = value

    def get_history(self) -> dict: return self.history
    def get_hyperparams(self) -> dict: return self.hyperparams