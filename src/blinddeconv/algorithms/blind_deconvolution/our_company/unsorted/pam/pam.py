"""
pam.py

Blind Image Deconvolution via Total Variation (Perrone–Favaro).

Reference:
    D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", IEEE Conference on Computer Vision
    and Pattern Recognition (CVPR), 2014.
    Technical Report: perrone2014tvTR.pdf

Pipeline:
    1. Normalise input to float64 [0, 1].
    2. Ensure odd image dimensions.
    3. Optional gamma correction.
    4. Multi-scale coarse-to-fine blind deconvolution (deblur).
    5. Clip and return restored image (int16, [0, 255]) and kernel.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# ── Framework base class import (DO NOT MODIFY) ─────────────────────────────
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

from .solvers import deblur


class PAM_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using Total Variation (Perrone & Favaro, CVPR 2014).

    The algorithm maximises TV to avoid the trivial delta-kernel solution,
    using a coarse-to-fine pyramid scheme with alternating gradient descent
    on the sharp image and kernel.

    Parameters
    ----------
    kernel_shape : tuple(int, int) — spatial support (height, width) of the
                   unknown PSF.  Both values should be odd.
                   Default (25, 25).
    lam : float — TV regularisation weight (lambda).
          Typical: 3e-4 .. 6e-4.  Noisy images: 1e-3 .. 3e-3.
          Default 3e-4.
    iters : int — number of gradient-descent iterations per blind/non-blind
            call at each pyramid scale.
            Default 1000.
    gamma_correct : bool — whether to apply gamma correction before
                    kernel estimation.  Default False.
    gamma : float — gamma exponent (used only when gamma_correct is True).
            Default 1.0.
    visualize : bool — print diagnostic information during optimisation.
                Default False.
    """

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

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── deblur() handles:
        #    1. Normalise to float64 [0, 1]
        #    2. Ensure odd dimensions
        #    3. Optional gamma correction
        #    4. Coarse-to-fine blind deconvolution
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

        # ── Crop padding ────────────────────────────────────────────────
        # deblur() returns u of size (M_odd + MK - 1, N_odd + NK - 1)
        # where M_odd, N_odd are the (cropped-to-odd) input dimensions.
        # We need to remove the MK//2 and NK//2 padding on each side,
        # then resize back to the original (H, W) if needed.
        pad_h = MK // 2
        pad_w = NK // 2
        u = u[pad_h:u.shape[0] - pad_h, pad_w:u.shape[1] - pad_w]

        # Handle even→odd dimension mismatch: resize to original size
        if u.shape[0] != orig_H or u.shape[1] != orig_W:
            from .utils import imresize
            u = imresize(u, (orig_H, orig_W), method='bicubic')

        # ── Clip to [0, 1] and convert to int16 [0, 255] ───────────────
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

    # ── Interface methods ────────────────────────────────────────────────
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
