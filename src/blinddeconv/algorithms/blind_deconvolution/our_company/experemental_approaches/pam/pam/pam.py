"""
pam.py

Источник:
    D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2014.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

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

from .solvers import deblur

class PAM_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: tuple = (25, 25),
        lam: float = 3e-4,
        iters: int = 1000,
        gamma_correct: bool = False,
        gamma: float = 1.0,
        visualize: bool = False,
    ):
        super().__init__(name='PAM-BD')

        self.kernel_shape = tuple(kernel_shape)
        self.lam = lam
        self.iters = iters
        self.gamma_correct = gamma_correct
        self.gamma = gamma
        self.visualize = visualize

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        MK, NK = self.kernel_shape
        orig_H, orig_W = image.shape[:2]

        u, kernel = deblur(
            image,
            MK, NK,
            lam=self.lam,
            iters=self.iters,
            gamma_correct=self.gamma_correct,
            gamma=self.gamma,
            visualize=self.visualize,
        )

        pad_h = MK // 2
        pad_w = NK // 2
        u = u[pad_h:u.shape[0] - pad_h, pad_w:u.shape[1] - pad_w]

        if u.shape[0] != orig_H or u.shape[1] != orig_W:
            from .utils import imresize
            u = imresize(u, (orig_H, orig_W), method='bicubic')

        u = np.clip(u, 0.0, 1.0)

        self.hyperparams = {
            'kernel_shape': self.kernel_shape,
            'lam': self.lam,
            'iters': self.iters,
            'gamma_correct': self.gamma_correct,
            'gamma': self.gamma,
            'time': time.time() - start_time,
        }

        x_final = u * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lam', self.lam),
            ('iters', self.iters),
            ('gamma_correct', self.gamma_correct),
            ('gamma', self.gamma),
            ('visualize', self.visualize),
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
