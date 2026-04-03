"""
pansharpening.py

Single-Image Super-Resolution via Variational Bayesian Pansharpening
with Super-Gaussian / TV Sparse Image Priors.

Reference:
    Pérez-Bueno, F., Vega, M., Mateos, J., Molina, R., & Katsaggelos, A. K.
    (2020). Variational Bayesian Pansharpening with Super-Gaussian Sparse
    Image Priors. Sensors, 20(18), 5308.

Pipeline (single grayscale image → super-resolved image + dummy kernel):
    1.  Normalise input to float64 [0, 1].
    2.  Build pseudo-PAN image (bicubic upsample to target HR size).
    3.  Run variational Bayesian SR (restSGME_Sens or TVME_Sens).
    4.  Return upscaled image (int16, [0, 255]) and dummy 3×3 zero kernel.
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

from .utils import get_psf, getfilters, getkappa, image_normalize, image_denormalize
from .solvers import (
    restoreSAR, alfaTVpvini, alfaSGlogvini, alfaSGlpvini,
    restSGME_Sens, TVME_Sens,
)


class SGPansharpening(DeconvolutionAlgorithm):
    """
    Single-Image Bayesian Super-Resolution using pansharpening machinery.

    The input LR grayscale image is treated as a 1-band MS observation.
    A bicubic up-sample serves as a pseudo-PAN guide.
    The output is the HR image at (ratio × ratio) the input resolution
    together with a dummy 3×3 zero kernel (since this is not deblurring).

    Parameters
    ----------
    ratio          : int — super-resolution factor (default 2).
    prior_type     : str — 'log', 'lp', or 'tv' (default 'log').
    filtersetname  : str — 'fohv' or 'fo' (SG priors only, default 'fohv').
    lp_p           : float — exponent for 'lp' prior (default 0.8).
    sensor         : str — PSF type: 'none' (box), 'gaussian', etc. (default 'none').
    eps_map        : float — convergence threshold (default 1e-4).
    itmax_map      : int — max outer iterations (default 50).
    itmin_map      : int — min outer iterations (default 2).
    eps_y          : float — CG tolerance (default 1e-7).
    itmax_y        : int — CG max iterations (default 30).
    gamma_gamma    : float — PAN hyperprior confidence (default 0.0).
    verbose        : bool — print iteration info (default False).
    """

    def __init__(
        self,
        ratio: int = 2,
        prior_type: str = 'log',
        filtersetname: str = 'fohv',
        lp_p: float = 0.8,
        sensor: str = 'none',
        eps_map: float = 1e-4,
        itmax_map: int = 50,
        itmin_map: int = 2,
        eps_y: float = 1e-7,
        itmax_y: int = 30,
        gamma_gamma: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(name='SG-Pansharpening')

        self.ratio = ratio
        self.prior_type = prior_type
        self.filtersetname = filtersetname
        self.lp_p = lp_p
        self.sensor = sensor
        self.eps_map = eps_map
        self.itmax_map = itmax_map
        self.itmin_map = itmin_map
        self.eps_y = eps_y
        self.itmax_y = itmax_y
        self.gamma_gamma = gamma_gamma
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 1:
            y = y[:, :, 0]
        elif y.ndim == 3 and y.shape[2] == 3:
            y = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]

        lr_h, lr_w = y.shape[:2]
        hr_h, hr_w = lr_h * self.ratio, lr_w * self.ratio
        nbands = 1

        # ── 2. Build pseudo-PAN, normalise, prepare observations ────────
        from scipy.ndimage import zoom
        Y_LR = y[:, :, np.newaxis]  # (lr_h, lr_w, 1)
        x_pan = zoom(y, self.ratio, order=3)
        x_pan = np.clip(x_pan, 0.0, 1.0)

        Y_norm, x_norm, facY, facx = image_normalize(Y_LR, x_pan)

        # Lambda coefficients (for 1 band always [1.0])
        lam = np.array([1.0])

        # PSF
        psf = get_psf(self.ratio, self.sensor)

        # ── 3. Initial hyperparameter estimates ─────────────────────────
        _, alpha_sar, beta_sar = restoreSAR(Y_norm[:, :, 0], np.array([[1.0]]))

        if self.prior_type == 'tv':
            alpha_init = alfaTVpvini(x_norm, 2)
            alpha_mode = np.array([alpha_init])
        elif self.prior_type == 'log':
            alpha_init = alfaSGlogvini(Y_norm, self.filtersetname)
            alpha_mode = alpha_init  # list of (1,) arrays
        elif self.prior_type == 'lp':
            alpha_init = alfaSGlpvini(Y_norm, self.lp_p, self.filtersetname)
            alpha_mode = alpha_init
        else:
            raise ValueError(f"Unknown prior_type: {self.prior_type!r}")

        beta_mode = np.array([beta_sar])
        gamma_mode = alpha_sar

        # ── 4. Run the solver ───────────────────────────────────────────
        if self.prior_type == 'tv':
            y_hr, alpha_out, beta_out, gamma_out, W_out = TVME_Sens(
                Y_norm, x_norm, lam, psf, nbands,
                eps_map=self.eps_map, itmax_map=self.itmax_map,
                itmin_map=self.itmin_map,
                alpha_mode=alpha_mode, beta_mode=beta_mode,
                gamma_mode=gamma_mode, gamma_gamma=self.gamma_gamma,
                eps_y=self.eps_y, itmax_y=self.itmax_y,
                verbose=self.verbose,
            )
        else:
            kappa = getkappa(self.prior_type,
                             self.lp_p if self.prior_type == 'lp' else None)
            y_hr, alpha_out, beta_out, gamma_out, W_out = restSGME_Sens(
                Y_norm, x_norm, lam, kappa, self.filtersetname, psf, nbands,
                eps_map=self.eps_map, itmax_map=self.itmax_map,
                itmin_map=self.itmin_map,
                alpha_mode=alpha_mode, beta_mode=beta_mode,
                gamma_mode=gamma_mode, gamma_gamma=self.gamma_gamma,
                eps_y=self.eps_y, itmax_y=self.itmax_y,
                verbose=self.verbose,
            )

        # ── 5. De-normalise and return ──────────────────────────────────
        if y_hr.ndim == 3:
            y_hr = y_hr[:, :, 0]

        y_hr = image_denormalize(y_hr, facY)

        elapsed = time.time() - start_time
        self.hyperparams = {
            'ratio': self.ratio,
            'prior_type': self.prior_type,
            'filtersetname': self.filtersetname,
            'eps_map': self.eps_map,
            'itmax_map': self.itmax_map,
            'alpha': alpha_out,
            'beta': beta_out,
            'gamma': gamma_out,
            'time': elapsed,
        }

        x_final = y_hr * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        dummy_kernel = np.zeros((3, 3), dtype=np.float64)
        return x_final, dummy_kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('ratio', self.ratio),
            ('prior_type', self.prior_type),
            ('filtersetname', self.filtersetname),
            ('lp_p', self.lp_p),
            ('sensor', self.sensor),
            ('eps_map', self.eps_map),
            ('itmax_map', self.itmax_map),
            ('itmin_map', self.itmin_map),
            ('eps_y', self.eps_y),
            ('itmax_y', self.itmax_y),
            ('gamma_gamma', self.gamma_gamma),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
