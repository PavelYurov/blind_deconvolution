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

from .solvers import blind_deconv_sr, ringing_artifacts_removal

class DCP_SelfExSR_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 25,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.003,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        SRF: int = 2,
        sr_num_iter: int = 3,
        sr_n_iter_bp: int = 5,
        n_warmup_levels: int = 2,
        n_sr_levels: int = 2,
        sr_alpha_min: float = 0.5,
        sr_alpha_max: float = 0.85,
        wiener_snr: float = 0.05,
        sr_downscale: float = 0.75,
    ):
        super().__init__(name='DCP-SelfExSR-BD')

        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

        self.SRF = SRF
        self.sr_num_iter = sr_num_iter
        self.sr_n_iter_bp = sr_n_iter_bp
        self.n_warmup_levels = n_warmup_levels
        self.n_sr_levels = n_sr_levels
        self.sr_alpha_min = sr_alpha_min
        self.sr_alpha_max = sr_alpha_max
        self.wiener_snr = wiener_snr
        self.sr_downscale = sr_downscale

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        dcp_opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'n_warmup_levels': self.n_warmup_levels,
            'n_sr_levels': self.n_sr_levels,
            'sr_alpha_min': self.sr_alpha_min,
            'sr_alpha_max': self.sr_alpha_max,
            'wiener_snr': self.wiener_snr,
            'sr_downscale': self.sr_downscale,
        }

        sr_opts = {
            'SRF': self.SRF,
            'numIter': self.sr_num_iter,
            'nIterBP': self.sr_n_iter_bp,
        }

        kernel, interim_latent = blind_deconv_sr(
            yg, self.lambda_dark, self.lambda_grad, dcp_opts,
            sr_opts=sr_opts,
        )

        Latent = ringing_artifacts_removal(
            y, kernel, self.lambda_tv, self.lambda_l0, self.weight_ring
        )
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'SRF': self.SRF,
            'sr_num_iter': self.sr_num_iter,
            'sr_n_iter_bp': self.sr_n_iter_bp,
            'n_warmup_levels': self.n_warmup_levels,
            'n_sr_levels': self.n_sr_levels,
            'sr_alpha_min': self.sr_alpha_min,
            'sr_alpha_max': self.sr_alpha_max,
            'wiener_snr': self.wiener_snr,
            'sr_downscale': self.sr_downscale,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_dark', self.lambda_dark),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('SRF', self.SRF),
            ('sr_num_iter', self.sr_num_iter),
            ('sr_n_iter_bp', self.sr_n_iter_bp),
            ('n_warmup_levels', self.n_warmup_levels),
            ('n_sr_levels', self.n_sr_levels),
            ('sr_alpha_min', self.sr_alpha_min),
            ('sr_alpha_max', self.sr_alpha_max),
            ('wiener_snr', self.wiener_snr),
            ('sr_downscale', self.sr_downscale),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
