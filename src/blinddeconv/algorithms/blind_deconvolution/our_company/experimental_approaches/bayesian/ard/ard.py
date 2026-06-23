"""
ard.py

Источник:
    J. Kotera, F. Sroubek, V. Smidl,
    "Blind Deconvolution with Model Discrepancies",
    IEEE Transactions on Image Processing, 2017.
"""

import numpy as np
import time
import copy
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

from .solvers import mc_restoration, vb_deconv, frils_deb_ubc

def _params_ard2() -> Dict[str, Any]:

    return {
        'gamma_corr': 1.0,
        'psf_method': 'ard',
        'PAR': {
            'maxROIsize': (512, 512),
            'MSlevels': 5,
            'factor': 1.5,
            'ARDnoise': 2,
            'deltaPDF': 0,
            'verbose': 0,
            'srf': 1,
            'gamma': 1e1,
            'alpha': 1.0,
            'd': 1.0,
            'reltol': 1e-4,
            'ccreltol': 1e-2,
            'uprior': {'type': 0, 'model': (0.0, 1e-7)},
            'gammamodel': (0.0, 1e-8),
            'betamodel': (1.0, 1e-6),
            'maxiter': 100,

            'gamma_nonblind': 1e6,
            'gammamodel_nonblind': (0.0, 1e-10),
            'uprior_nonblind': {'type': 0, 'model': (0.0, 2e-4)},
            'maxiter_u': 10,
        },
    }

def _params_ard3() -> Dict[str, Any]:

    return {
        'gamma_corr': 1.0,
        'psf_method': 'ard',
        'PAR': {
            'maxROIsize': (512, 512),
            'MSlevels': 5,
            'factor': 1.5,
            'ARDnoise': 3,
            'deltaPDF': 0,
            'verbose': 0,
            'srf': 1,
            'gamma': 1.0,
            'alpha': 1e1,
            'd': 1.0,
            'reltol': 1e-4,
            'ccreltol': 1e-2,
            'uprior': {'type': 0, 'model': (0.0, 1e-7)},
            'dmodel': (0.0, 1e-3),
            'gammamodel': (0.0, 1e-8),
            'betamodel': (1.0, 1e-7),
            'maxiter': 100,

            'gamma_nonblind': 1e6,
            'gammamodel_nonblind': (0.0, 1e-10),
            'uprior_nonblind': {'type': 0, 'model': (0.0, 1e-4)},
            'maxiter_u': 10,
        },
    }

class ARD_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 33,
        method: str = 'ard3',
        ms_levels: int = 5,
        factor: float = 1.5,
        max_roi_size: Tuple[int, int] = (512, 512),
        maxiter: int = 100,
        maxiter_u: int = 10,
        reltol: float = 1e-4,
        ccreltol: float = 1e-2,
        gamma_corr: float = 1.0,
        verbose: int = 0,
        kernel_thresh: float = 0.05,
        beta_a0: float = 1.0,
        nonblind_method: str = 'firls',
        firls_lambda: float = 2e-4,
        firls_alpha: float = 2.0 / 3.0,
        firls_epsilon_min: float = 2.55 / 255.0,
        firls_epsilon_max: float | None = None,
        firls_out_iter: int = 5,
        firls_inner_iter: int = 4,
        firls_IF: float = float(np.sqrt(2.0)),
        firls_lambda_u: float = 0.1,
    ):
        super().__init__(name='ARD-BD')

        self.kernel_size = int(kernel_size)
        self.method = str(method)
        self.ms_levels = int(ms_levels)
        self.factor = float(factor)
        self.max_roi_size = tuple(max_roi_size)
        self.maxiter = int(maxiter)
        self.maxiter_u = int(maxiter_u)
        self.reltol = float(reltol)
        self.ccreltol = float(ccreltol)
        self.gamma_corr = float(gamma_corr)
        self.verbose = int(verbose)
        self.kernel_thresh = float(kernel_thresh)
        self.beta_a0 = float(beta_a0)
        self.nonblind_method = str(nonblind_method)
        self.firls_lambda = float(firls_lambda)
        self.firls_alpha = float(firls_alpha)
        self.firls_epsilon_min = float(firls_epsilon_min)
        self.firls_epsilon_max = (
            float(firls_epsilon_max)
            if firls_epsilon_max is not None
            else float(firls_epsilon_min)
        )
        self.firls_out_iter = int(firls_out_iter)
        self.firls_inner_iter = int(firls_inner_iter)
        self.firls_IF = float(firls_IF)
        self.firls_lambda_u = float(firls_lambda_u)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def _build_params(self) -> Dict[str, Any]:
        if self.method == 'ard2':
            params = _params_ard2()
        elif self.method == 'ard3':
            params = _params_ard3()
        else:
            raise ValueError(f"Unknown method {self.method!r}; "
                             "use 'ard2' or 'ard3'.")
        params = copy.deepcopy(params)
        params['gamma_corr'] = self.gamma_corr
        PAR = params['PAR']
        PAR['MSlevels'] = self.ms_levels
        PAR['factor'] = self.factor
        PAR['maxROIsize'] = self.max_roi_size
        PAR['maxiter'] = self.maxiter
        PAR['maxiter_u'] = self.maxiter_u
        PAR['reltol'] = self.reltol
        PAR['ccreltol'] = self.ccreltol
        PAR['verbose'] = self.verbose

        PAR['betamodel'] = (self.beta_a0, PAR['betamodel'][1])

        params['pyramid_thresh'] = self.kernel_thresh * 0.5
        return params

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = np.asarray(image).astype(np.float64)
        if y.max() > 1.0:
            y = y / 255.0
        if y.ndim == 3:

            y = y.mean(axis=2)

        ksz = self.kernel_size
        if ksz % 2 == 0:
            ksz += 1
        hsize = (ksz, ksz)

        params = self._build_params()

        kernel, _gamma_vec = mc_restoration(y, hsize, params)
        if kernel.ndim == 3:
            kernel = kernel[:, :, 0]

        if self.kernel_thresh > 0.0:
            thresh = kernel.max() * self.kernel_thresh
            kernel = np.where(kernel >= thresh, kernel, 0.0)
            ksum = kernel.sum()
            if ksum > 1e-10:
                kernel = kernel / ksum

        if self.nonblind_method == 'firls':
            firls_opt = {
                'lambda': self.firls_lambda,
                'alpha': self.firls_alpha,
                'epsilon_min': self.firls_epsilon_min,
                'epsilon_max': self.firls_epsilon_max,
                'out_iter': self.firls_out_iter,
                'inner_iter': self.firls_inner_iter,
                'IF': self.firls_IF,
                'lambda_u': self.firls_lambda_u,
                'beta_a': (self.firls_lambda * self.firls_alpha
                           * (20.0 / 255.0) ** (self.firls_alpha - 2.0)),
            }

            latent = frils_deb_ubc(y, kernel, firls_opt)
            latent = np.clip(latent, 0.0, 1.0)
        else:

            hh0 = kernel.shape[0] // 2
            hh1 = kernel.shape[1] // 2
            if hh0 > 0 or hh1 > 0:
                y_padded = np.pad(y, ((hh0, hh0), (hh1, hh1)), mode='reflect')
            else:
                y_padded = y
            latent, _report = vb_deconv([y_padded], [kernel], params)
            latent = latent[hh0: hh0 + y.shape[0], hh1: hh1 + y.shape[1]]
            latent = np.clip(latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': ksz,
            'method': self.method,
            'ms_levels': self.ms_levels,
            'factor': self.factor,
            'maxiter': self.maxiter,
            'maxiter_u': self.maxiter_u,
            'reltol': self.reltol,
            'ccreltol': self.ccreltol,
            'gamma_corr': self.gamma_corr,
            'kernel_thresh': self.kernel_thresh,
            'time': time.time() - start_time,
        }

        latent = np.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=0.0)
        x_final = latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('method', self.method),
            ('ms_levels', self.ms_levels),
            ('factor', self.factor),
            ('max_roi_size', self.max_roi_size),
            ('maxiter', self.maxiter),
            ('maxiter_u', self.maxiter_u),
            ('reltol', self.reltol),
            ('ccreltol', self.ccreltol),
            ('gamma_corr', self.gamma_corr),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'max_roi_size':
                    self.max_roi_size = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
