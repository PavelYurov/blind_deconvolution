"""
Blind Image Deconvolution framework wrapper.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict, Optional

from .solvers import blind_deconv_multiscale

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


class FractionalOrderPMPDeconv(DeconvolutionAlgorithm):


    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (35, 35),

        alpha: float = 1.0,
        mu: float = 1.0,
        lam: float = 2e-3,
        gamma: float = 4e-3,
        beta: float = 10.0,

        rho1_init: float = 2.0,
        rho2_init: float = 2.0,
        rho3_init: float = 50.0,
        rho_factor: float = 1.5,

        num_scales: int = 5,
        scale_factor: float = 0.8,
        iter_per_scale: int = 5,

        kernel_threshold: float = 0.05,
        grad_threshold_factor: float = 2.0,
        hysteresis_ratio: float = 0.5,
        border_width: int = 5,
        boundary_mode: str = 'pad',

        patch_size: int = 3,

        final_restoration_mode: Optional[str] = None,
        mu_nonblind: float = 50.0,
        lam_nonblind: float = 2e-3,
    ):
        super().__init__(name='FractionalOrderPMP')

        self.kernel_shape    = tuple(kernel_shape)
        self.alpha           = float(alpha)
        self.mu              = float(mu)
        self.lam             = float(lam)
        self.gamma           = float(gamma)
        self.beta            = float(beta)

        self.rho1_init       = float(rho1_init)
        self.rho2_init       = float(rho2_init)
        self.rho3_init       = float(rho3_init)
        self.rho_factor      = float(rho_factor)

        self.num_scales      = int(num_scales)
        self.scale_factor    = float(scale_factor)
        self.iter_per_scale  = int(iter_per_scale)

        self.kernel_threshold      = float(kernel_threshold)
        self.grad_threshold_factor = float(grad_threshold_factor)
        self.hysteresis_ratio      = float(hysteresis_ratio)
        self.border_width          = int(border_width)
        self.boundary_mode         = boundary_mode

        self.patch_size      = int(patch_size)

        self.final_restoration_mode = final_restoration_mode
        self.mu_nonblind     = float(mu_nonblind)
        self.lam_nonblind    = float(lam_nonblind)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        g = image.astype(np.float64) / 255.0
        params = self._build_params()

        f_est, h_est, history = blind_deconv_multiscale(g, params)

        self.history = history
        elapsed = time.time() - start_time
        self.hyperparams = {'time': elapsed, 'params': params}

        x_final = f_est * 255.0
        x_final = np.clip(x_final, 0, 255)
        x_final = np.round(x_final).astype(np.int16)

        if h_est.sum() > 0:
            h_est = h_est / h_est.sum()

        return x_final, h_est

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape',     self.kernel_shape),
            ('beta',             self.beta),
            ('kernel_threshold', self.kernel_threshold),
            ('grad_threshold_factor', self.grad_threshold_factor),
            ('hysteresis_ratio', self.hysteresis_ratio),
            ('scale_factor',     self.scale_factor),
            ('final_restoration_mode', self.final_restoration_mode),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict: return self.history
    def get_hyperparams(self) -> dict: return self.hyperparams

    def _build_params(self) -> Dict[str, Any]:
        return {
            'kernel_shape':     self.kernel_shape,
            'alpha':            self.alpha,
            'mu':               self.mu,
            'lam':              self.lam,
            'gamma':            self.gamma,
            'beta':             self.beta,
            'rho1_init':        self.rho1_init,
            'rho2_init':        self.rho2_init,
            'rho3_init':        self.rho3_init,
            'rho_factor':       self.rho_factor,
            'num_scales':       self.num_scales,
            'scale_factor':     self.scale_factor,
            'iter_per_scale':   self.iter_per_scale,
            'kernel_threshold': self.kernel_threshold,
            'grad_threshold_factor': self.grad_threshold_factor,
            'hysteresis_ratio': self.hysteresis_ratio,
            'border_width':     self.border_width,
            'boundary_mode':    self.boundary_mode,
            'patch_size':       self.patch_size,
            'final_restoration_mode': self.final_restoration_mode,
            'mu_nonblind':      self.mu_nonblind,
            'lam_nonblind':     self.lam_nonblind,
        }
