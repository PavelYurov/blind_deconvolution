"""
selfexsr.py

Framework wrapper for SelfExSR — Single Image Super-Resolution
Using Transformed Self-Exemplars (Huang et al., CVPR 2015).

Interface hack: accepts an image as if it were blurred, runs super-
resolution, and returns the HR result + a dummy 3×3 zero kernel.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

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

from .solvers import sr_demo, sr_init_opt


class SelfExSR(DeconvolutionAlgorithm):
    """
    Single Image Super-Resolution Using Transformed Self-Exemplars.

    Parameters
    ----------
    SRF        : int — super-resolution factor (2, 3, 4, or 8). Default 2.
    numIter    : int — PatchMatch iterations at first level. Default 15.
    nIterBP    : int — back-projection iterations. Default 20.
    usePlaneGuide : bool — use planar structure guidance. Default False.
    useAffine  : bool — use affine PatchMatch. Default True.
    """

    def __init__(
        self,
        SRF: int = 2,
        numIter: int = 15,
        nIterBP: int = 20,
        usePlaneGuide: bool = False,
        useAffine: bool = True,
    ):
        super().__init__(name='SelfExSR')

        self.SRF = SRF
        self.numIter = numIter
        self.nIterBP = nIterBP
        self.usePlaneGuide = usePlaneGuide
        self.useAffine = useAffine

        self.history: Dict[str, list] = {}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # Build options
        opt = sr_init_opt(self.SRF)
        opt['numIter'] = self.numIter
        opt['nIterBP'] = self.nIterBP
        opt['usePlaneGuide'] = self.usePlaneGuide
        opt['useAffine'] = self.useAffine

        # Run super-resolution
        img_hr = sr_demo(image, self.SRF, opt=opt)

        elapsed = time.time() - start_time

        self.hyperparams = {
            'SRF': self.SRF,
            'numIter': self.numIter,
            'nIterBP': self.nIterBP,
            'usePlaneGuide': self.usePlaneGuide,
            'useAffine': self.useAffine,
            'time': elapsed,
        }

        # Convert to int16 [0, 255]
        img_hr = np.clip(img_hr * 255.0, 0, 255).astype(np.int16)

        # Dummy kernel (interface requirement)
        kernel = np.zeros((3, 3), dtype=np.float64)
        kernel[1, 1] = 1.0

        return img_hr, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('SRF', self.SRF),
            ('numIter', self.numIter),
            ('nIterBP', self.nIterBP),
            ('usePlaneGuide', self.usePlaneGuide),
            ('useAffine', self.useAffine),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
