"""
amape_htp.py

Источник:
    J. Kotera, F. Sroubek, P. Milanfar:
    "Blind deconvolution using alternating maximum a posteriori estimation
    with heavy-tailed priors", DOI: 10.1007/978-3-642-40246-3_8
"""

import numpy as np
import math
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

from .utils import (
    normalizeImage,
    matNormalize,
    getROI,
    createROI,
    doublePSF,
    copy_mat_2_cmat,
)
from .solvers import PSFestimaLnoRgrad, fftCGSRaL

class AMAPE_HTP(DeconvolutionAlgorithm):

    def __init__(
        self,
        psf_size: int = 32,
        maxROIsize_r: int = 1024,
        maxROIsize_c: int = 1024,
        MSlevels: int = 4,
        gamma: float = 1e2,
        Lp: float = 0.3,
        beta_h_factor: float = 1e4,
        alpha_h_factor: float = 1e1,
        centering_threshold: float = 30.0 / 255.0,
        beta_u_factor: float = 1e0,
        alpha_u_factor: float = 1e-2,
        gamma_nonblind_factor: float = 2e3,
        beta_u_nonblind_factor: float = 1.0,
        Lp_nonblind: float = 0.0,
        maxiter_u: int = 10,
        maxiter_h: int = 10,
        maxiter: int = 5,
        ccreltol: float = 1e-3,
    ):
        super().__init__(name='AMAPE-HTP')

        self.psf_size = psf_size
        self.maxROIsize_r = maxROIsize_r
        self.maxROIsize_c = maxROIsize_c
        self.MSlevels = MSlevels
        self.gamma = gamma
        self.Lp = Lp
        self.beta_h_factor = beta_h_factor
        self.alpha_h_factor = alpha_h_factor
        self.centering_threshold = centering_threshold
        self.beta_u_factor = beta_u_factor
        self.alpha_u_factor = alpha_u_factor
        self.gamma_nonblind_factor = gamma_nonblind_factor
        self.beta_u_nonblind_factor = beta_u_nonblind_factor
        self.Lp_nonblind = Lp_nonblind
        self.maxiter_u = maxiter_u
        self.maxiter_h = maxiter_h
        self.maxiter = maxiter
        self.ccreltol = ccreltol

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def _build_param(self) -> dict:

        gamma = self.gamma
        gamma_nonblind = self.gamma_nonblind_factor * gamma

        return {
            'gamma': gamma,
            'Lp': self.Lp,
            'beta_h': self.beta_h_factor * gamma,
            'alpha_h': self.alpha_h_factor * gamma,
            'centering_threshold': self.centering_threshold,
            'beta_u': self.beta_u_factor * gamma,
            'alpha_u': self.alpha_u_factor * gamma,
            'gamma_nonblind': gamma_nonblind,
            'beta_u_nonblind': self.beta_u_nonblind_factor * gamma_nonblind,
            'Lp_nonblind': self.Lp_nonblind,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'maxiter': self.maxiter,
            'ccreltol': self.ccreltol,
        }

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()
        param = self._build_param()

        Src = image.astype(np.float64)

        Src, _min, _max = normalizeImage(Src)

        ROI = getROI(Src, self.maxROIsize_r, self.maxROIsize_c)

        h_rows = self.psf_size >> (self.MSlevels - 1)
        h_cols = self.psf_size >> (self.MSlevels - 1)

        h = np.zeros((h_rows, h_cols), dtype=np.float64)
        cen_r = int(math.floor((h_rows + 1) / 2 - 1))
        cen_c = int(math.floor((h_cols + 1) / 2 - 1))
        h[cen_r, cen_c] = 1.0

        for L in range(1, self.MSlevels + 1):

            h = matNormalize(h)

            tmp = ROI.copy()
            tmp = createROI(tmp, L, self.MSlevels)

            cROI = copy_mat_2_cmat(tmp, (tmp.shape[0], tmp.shape[1]),
                                   tmp.shape[0], tmp.shape[1])

            h = PSFestimaLnoRgrad(h, cROI, param, L)

            if L != self.MSlevels:
                h = doublePSF(h)

        h = matNormalize(h)

        g_x = 0.0
        g_y = 0.0
        nr, nc = h.shape
        for x in range(nc):
            for y in range(nr):
                g_x += x * h[x, y]
                g_y += y * h[x, y]

        shift_x = nr // 2 - int(math.floor(g_x))
        shift_y = nc // 2 - int(math.floor(g_y))

        tmp_h = np.zeros_like(h)
        for x in range(nr):
            for y in range(nc):
                nx = x + shift_x
                ny = y + shift_y
                if 0 <= nx < nr and 0 <= ny < nc:
                    tmp_h[nx, ny] = h[x, y]
        h = tmp_h

        U = fftCGSRaL(Src, h, param)

        restored = np.real(U) * (_max - _min) + _min

        restored = np.clip(restored, 0.0, 255.0)
        x_final = np.round(restored).astype(np.int16)

        self.hyperparams = {
            'psf_size': self.psf_size,
            'MSlevels': self.MSlevels,
            'gamma': self.gamma,
            'Lp': self.Lp,
            'Lp_nonblind': self.Lp_nonblind,
            'maxiter': self.maxiter,
            'time': time.time() - start_time,
        }

        return x_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('psf_size', self.psf_size),
            ('maxROIsize_r', self.maxROIsize_r),
            ('maxROIsize_c', self.maxROIsize_c),
            ('MSlevels', self.MSlevels),
            ('gamma', self.gamma),
            ('Lp', self.Lp),
            ('beta_h_factor', self.beta_h_factor),
            ('alpha_h_factor', self.alpha_h_factor),
            ('centering_threshold', self.centering_threshold),
            ('beta_u_factor', self.beta_u_factor),
            ('alpha_u_factor', self.alpha_u_factor),
            ('gamma_nonblind_factor', self.gamma_nonblind_factor),
            ('beta_u_nonblind_factor', self.beta_u_nonblind_factor),
            ('Lp_nonblind', self.Lp_nonblind),
            ('maxiter_u', self.maxiter_u),
            ('maxiter_h', self.maxiter_h),
            ('maxiter', self.maxiter),
            ('ccreltol', self.ccreltol),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
