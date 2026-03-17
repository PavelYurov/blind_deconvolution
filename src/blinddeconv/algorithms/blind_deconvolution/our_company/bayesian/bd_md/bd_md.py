"""
Blind Deconvolution with Model Discrepancies (Kotera et al. 2017).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import resize_image, resize_kernel, get_boundary_mask, crop_image, pad_image, center_kernel
from .solvers import VBSolver

# ── Framework base class import ──────────────────────────────────────────────
import sys
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
# ─────────────────────────────────────────────────────────────────────────────


class BlindDeconvMD(DeconvolutionAlgorithm):
    """
    Blind Deconvolution with Model Discrepancies (VB-ARD).
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        max_iter: int = 30,
        scales: int = 4, # Standard for multiscale
        scale_factor: float = 1.5,
        b_lambda: float = 1e-4, # Для изображения
        b_beta: float = 1e-5,   # Для ядра (поменьше, чтобы стимулировать sparsity)
        verbose: bool = False,
    ):
        super().__init__(name='BD-MD-2017')
        self.kernel_shape = tuple(kernel_shape)
        self.max_iter = max_iter
        self.scales = scales
        self.scale_factor = scale_factor
        self.b_lambda = b_lambda
        self.b_beta = b_beta
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Normalize
        y = image.astype(np.float64)
        if y.max() > 1.0: y /= 255.0
        
        # 2. Pad to avoid FFT artifacts
        pad_h = self.kernel_shape[0] // 2 + 8
        pad_w = self.kernel_shape[1] // 2 + 8
        y_padded = pad_image(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap') # 'wrap' лучше для FFT
        
        # 3. Pyramid
        pyramid = []
        curr = y_padded
        for s in range(self.scales):
            pyramid.append(curr)
            if s < self.scales - 1:
                nh = int(curr.shape[0] / self.scale_factor)
                nw = int(curr.shape[1] / self.scale_factor)
                if nh < self.kernel_shape[0] + 5 or nw < self.kernel_shape[1] + 5:
                    break
                curr = resize_image(y_padded, (nh, nw))
        pyramid = pyramid[::-1]
        
        # 4. Init Kernel
        # Start small
        scale_0 = pyramid[0].shape[0] / y_padded.shape[0]
        k0_h = max(3, int(self.kernel_shape[0] * scale_0))
        k0_w = max(3, int(self.kernel_shape[1] * scale_0))
        if k0_h % 2 == 0: k0_h += 1
        if k0_w % 2 == 0: k0_w += 1
        
        h_curr = self._create_initial_kernel((k0_h, k0_w))
        x_res = None

        # 5. Multiscale Loop
        for i, img in enumerate(pyramid):
            kh, kw = h_curr.shape
            if self.verbose:
                print(f"Scale {i}: Img {img.shape}, Ker {h_curr.shape}")
                
            mask = get_boundary_mask(img.shape, (kh, kw))
            
            solver = VBSolver(img, (kh, kw), mask, self.b_lambda, self.b_beta)
            solver.initialize_h(h_curr)
            
            if x_res is not None:
                x_res = resize_image(x_res, img.shape)
                solver.u = x_res
            
            # Iterations
            for it in range(self.max_iter):
                h_prev = solver.h.copy()
                solver.run_step()
                
                # Convergence check
                diff = np.linalg.norm(solver.h - h_prev)
                if diff < 1e-5 and it > 5: break
            
            h_curr = solver.h
            x_res = solver.u
            
            # Upsample kernel for next scale
            if i < len(pyramid) - 1:
                # Center before resize
                h_curr, shift_val = center_kernel(h_curr)
                if shift_val != (0,0):
                    x_res = np.roll(x_res, (-shift_val[0], -shift_val[1]), axis=(0,1))
                
                # Next size
                next_scale = pyramid[i+1].shape[0] / y_padded.shape[0]
                nkh = int(self.kernel_shape[0] * next_scale)
                nkw = int(self.kernel_shape[1] * next_scale)
                if nkh % 2 == 0: nkh += 1
                if nkw % 2 == 0: nkw += 1
                
                h_curr = resize_kernel(h_curr, (nkh, nkw))

        # 6. Final Size Adjust
        if h_curr.shape != self.kernel_shape:
            h_curr = resize_kernel(h_curr, self.kernel_shape)
        h_curr, _ = center_kernel(h_curr)
        
        # 7. Final Deconvolution (Non-blind)
        # Use simple Wiener or just a few steps of the solver with fixed h
        mask = get_boundary_mask(y_padded.shape, self.kernel_shape)
        final_solver = VBSolver(y_padded, self.kernel_shape, mask, self.b_lambda, self.b_beta)
        final_solver.initialize_h(h_curr)
        if x_res.shape != y_padded.shape:
            x_res = resize_image(x_res, y_padded.shape)
        final_solver.u = x_res
        final_solver.alpha = 200.0 # High trust in data
        
        for _ in range(5):
            final_solver.run_step()
            final_solver.h = h_curr # Keep kernel fixed
            
        x_final = final_solver.u
        h_final = h_curr
        
        # Crop padding
        x_final = crop_image(x_final, (pad_h, pad_w))
        x_final = np.clip(x_final * 255.0, 0, 255).astype(np.uint8)
        
        return x_final, h_final

    def _create_initial_kernel(self, shape):
        kh, kw = shape
        sig = max(kh, kw) / 8.0
        cy, cx = kh // 2, kw // 2
        y, x = np.mgrid[-cy:kh-cy, -cx:kw-cx]
        h = np.exp(-(x**2 + y**2) / (2 * sig**2 + 1e-9))
        return h / h.sum()

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('max_iter', self.max_iter),
            ('scales', self.scales),
            ('b_lambda', self.b_lambda),
            ('b_beta', self.b_beta)
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