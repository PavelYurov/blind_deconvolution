"""
Blind Image Deconvolution — Algorithm Template.

Replace this docstring with a description of your algorithm and references.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# --- Your algorithm-specific imports here ---
# from .utils import ...
# from .solvers import ...

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


class BlindDeconvTemplate(DeconvolutionAlgorithm):
    """
    TODO: Rename the class and describe your algorithm here.
    """

    def __init__(
        self,
        # TODO: add your hyperparameters
    ):
        super().__init__(name='TODO-AlgorithmName')

        # TODO: store your hyperparameters
        # self.param = param

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        self.hyperparams = {
            'time': time.time() - start_time,
            # TODO: add relevant hyperparams / diagnostics
        }

        x_final = x_final * 255.0
        x_final = np.round(x_final).astype(np.int16)
        return x_final, h #restored image, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape), #for example
            # TODO: expose your hyperparameters
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
