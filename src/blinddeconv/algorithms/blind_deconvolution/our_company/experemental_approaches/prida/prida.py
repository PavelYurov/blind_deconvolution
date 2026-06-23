"""
prida.py

Источник:
Ravi, S. N., Mehta, R., & Singh, V. (2018).
"Robust Blind Deconvolution via Mirror Descent."
arXiv:1803.08137 [cs.CV].
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .solvers import coarse_to_fine
import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root (pyproject.toml)")
        path = path.parent
    return path

_CURRENT_FILE  = Path(__file__).resolve()
_PROJECT_ROOT  = _find_project_root(_CURRENT_FILE)
_SRC_DIR       = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from blinddeconv.algorithms.base import DeconvolutionAlgorithm

class PRIDA(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 5e-3,
        n_iters: int = 1000,
        lambda_multiplier: float = 1.9,
        max_lambda: float = 0.11,
        scale_multiplier: float = 1.1,
        verbose: bool = False,
    ):
        super().__init__(name='PRIDA')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.n_iters = n_iters
        self.lambda_multiplier = lambda_multiplier
        self.max_lambda = max_lambda
        self.scale_multiplier = scale_multiplier
        self.verbose = verbose

        self.history: Dict[str, Any] = {}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        H_orig, W_orig = y.shape

        rpad = 1 if H_orig % 2 == 0 else 0
        cpad = 1 if W_orig % 2 == 0 else 0
        if rpad or cpad:
            y = y[: H_orig - rpad, : W_orig - cpad]

        H, W = y.shape
        MK, NK = self.kernel_shape

        if self.verbose:
            print(f"[{self.name}] Input: {H_orig}×{W_orig} "
                  f"(work {H}×{W}), "
                  f"Kernel: {MK}×{NK}, λ={self.lambda_tv}")

        u_padded, k = coarse_to_fine(
            image=y,
            kernel_shape=(MK, NK),
            lambda_val=self.lambda_tv,
            n_iters=self.n_iters,
            lambda_multiplier=self.lambda_multiplier,
            max_lambda=self.max_lambda,
            scale_multiplier=self.scale_multiplier,
            verbose=self.verbose,
        )

        pad_top  = MK // 2
        pad_left = NK // 2
        u_cropped = u_padded[pad_top: pad_top + H,
                             pad_left: pad_left + W]

        if u_cropped.shape != (H_orig, W_orig):
            restored_full = np.zeros((H_orig, W_orig), dtype=np.float64)
            restored_full[:H, :W] = u_cropped

            if rpad:
                restored_full[H:, :W] = u_cropped[-1:, :]
            if cpad:
                restored_full[:H, W:] = u_cropped[:, -1:]
            if rpad and cpad:
                restored_full[H:, W:] = u_cropped[-1, -1]
            u_cropped = restored_full

        elapsed = time.time() - start_time
        self.timer = elapsed

        self.hyperparams = {
            'lambda_tv':         self.lambda_tv,
            'n_iters':           self.n_iters,
            'lambda_multiplier': self.lambda_multiplier,
            'max_lambda':        self.max_lambda,
            'scale_multiplier':  self.scale_multiplier,
            'elapsed_time':      elapsed,
            'image_shape':       (H_orig, W_orig),
            'kernel_shape':      (MK, NK),
        }

        restored = np.clip(u_cropped * 255.0, 0.0, 255.0)
        restored = np.round(restored).astype(np.int16)

        return restored, k

    def get_param(self) -> List[Tuple[str, Any]]:

        return [
            ('kernel_shape',      self.kernel_shape),
            ('lambda_tv',         self.lambda_tv),
            ('n_iters',           self.n_iters),
            ('lambda_multiplier', self.lambda_multiplier),
            ('max_lambda',        self.max_lambda),
            ('scale_multiplier',  self.scale_multiplier),
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
