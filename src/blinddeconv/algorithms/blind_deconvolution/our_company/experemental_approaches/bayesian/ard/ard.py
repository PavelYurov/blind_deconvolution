"""
ard.py

Blind Image Deconvolution via Variational Bayes with ARD priors.

Reference:
    J. Kotera, F. Sroubek, V. Smidl,
    "Blind Deconvolution with Model Discrepancies",
    IEEE Transactions on Image Processing, 2017.

Pipeline (mirrors MATLAB demo_run.m):
    1. Normalise input to float64 [0, 1] grayscale.
    2. Optional gamma correction.
    3. Multiscale blind PSF estimation (mc_restoration).
    4. Non-blind VB deconvolution on the full image (vb_deconv).
    5. Return restored image (int16, [0, 255]) and PSF.
"""

import numpy as np
import time
import copy
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

from .solvers import mc_restoration, vb_deconv, frils_deb_ubc


# ═════════════════════════════════════════════════════════════════════════════
# Default parameter sets (ports of params_ard2.m / params_ard3.m)
# ═════════════════════════════════════════════════════════════════════════════

def _params_ard2() -> Dict[str, Any]:
    """Port of params_ard2.m — Ours_gamma method.

    NOTE on ``gammamodel``: in the original MATLAB ``params_ard2.m`` the
    field ``params.PAR.gammamodel`` is assigned **twice** — first to
    ``[0 1e-4]`` (the documented "blind" value) and then *overwritten* to
    ``[0 1e-8]`` in the nonblind block.  Because of MATLAB struct semantics
    only the last assignment survives, so the blind PSF estimator
    actually runs with ``gammamodel = [0, 1e-8]``.  The nonblind step
    looks for a ``gammamodel_nonblind`` field that never exists, so it
    falls back to ``[0, 1e-10]`` (see ``VBdeconv.m``).  We replicate the
    *actual* MATLAB behaviour here.
    """
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
            'gammamodel': (0.0, 1e-8),     # <-- MATLAB final value (overwritten)
            'betamodel': (1.0, 1e-6),  # beta_a0=1.0: 3× stronger ARD prior on PSF
            'maxiter': 100,
            # nonblind
            'gamma_nonblind': 1e6,
            'gammamodel_nonblind': (0.0, 1e-10),  # <-- MATLAB VBdeconv fallback
            'uprior_nonblind': {'type': 0, 'model': (0.0, 2e-4)},
            'maxiter_u': 10,
        },
    }


def _params_ard3() -> Dict[str, Any]:
    """Port of params_ard3.m — Ours_alpha_gamma method (Student-t noise).

    See note on ``gammamodel`` in :func:`_params_ard2`.  Same MATLAB
    overwrite happens here — final blind value is ``[0, 1e-8]``,
    nonblind falls back to ``[0, 1e-10]``.
    """
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
            'gammamodel': (0.0, 1e-8),     # <-- MATLAB final value (overwritten)
            'betamodel': (1.0, 1e-7),  # beta_a0=1.0: 3× stronger ARD prior on PSF
            'maxiter': 100,
            # nonblind
            'gamma_nonblind': 1e6,
            'gammamodel_nonblind': (0.0, 1e-10),  # <-- MATLAB VBdeconv fallback
            'uprior_nonblind': {'type': 0, 'model': (0.0, 1e-4)},
            'maxiter_u': 10,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm wrapper
# ═════════════════════════════════════════════════════════════════════════════

class ARD_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution via Variational Bayes + ARD priors.

    Parameters
    ----------
    kernel_size : int
        Spatial support (square, odd) of the unknown PSF.  Default 33.
    method : str
        Either ``'ard2'`` (Ours_gamma) or ``'ard3'`` (Ours_alpha_gamma —
        Student-t noise model).  Default ``'ard3'``.
    ms_levels : int
        Number of pyramid levels (1 disables multiscale).  Default 5.
    factor : float
        Pyramid downsampling factor between levels.  Default 1.5.
    max_roi_size : tuple of int
        Central ROI size used for PSF estimation.  Default (512, 512).
    maxiter : int
        Outer VB iterations per pyramid level.  Default 100.
    maxiter_u : int
        VB iterations of the non-blind restoration.  Default 10.
    reltol : float
        Relative tolerance of the inner CG solver.  Default 1e-4.
    ccreltol : float
        Convergence tolerance of the outer VB loop.  Default 1e-2.
    gamma_corr : float
        Gamma correction applied to the input before PSF estimation.
        ``1.0`` means no correction.  Default 1.0.
    verbose : int
        ``0`` silent, ``1`` text logs.  Default 0.
    nonblind_method : str
        Non-blind deconvolution step to use after PSF estimation.
        ``'vb'``  — original Variational Bayes (Gaussian prior on image).
        ``'firls'`` — Fast IRLS with Lp/Huber prior (from FBDHSGP); sharper
        edges and fewer ringing artefacts because it uses a sparse
        (non-Gaussian) prior on image gradients.  Requires the ``fbdhsgp``
        package to be importable.  Default ``'firls'``.
    firls_lambda : float
        Data-fidelity weight for FIRLS (default ``2e-4``).
    firls_alpha : float
        Lp exponent for FIRLS prior (default ``2/3``; set to ``0.8`` for
        milder regularisation on difficult images).
    firls_epsilon_min : float
        Huber ε floor (default ``2.55/255 ≈ 0.01``).  Smaller → closer to
        true Lp, sharper, but noisier.  Increase to ``5/255`` for noisy input.
    firls_epsilon_max : float or None
        Starting ε for the continuation scheme (``None`` → same as
        ``firls_epsilon_min``, which is the standard choice).
    firls_out_iter : int
        Outer continuation iterations on β (default ``5``).
    firls_inner_iter : int
        Inner ADMM iterations per β level (default ``4``).
    firls_IF : float
        β multiplicative continuation factor (default ``√2``).
    firls_lambda_u : float
        FOV-constraint penalty (default ``0.1``).

    Notes
    -----
    **Computation time** with default settings (5 pyramid levels, 100 VB
    iterations, 512×512 ROI, CG up to 1000 steps) mirrors the original
    MATLAB code and typically takes **several minutes** on a modern CPU.

    For quick testing use::

        ARD_BD(kernel_size=15, ms_levels=2, max_roi_size=(128, 128),
               maxiter=20, maxiter_u=3)

    This reduces runtime to ~10–30 seconds with a visible quality drop.
    """

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

    # ── Build the parameter dict for the solvers ────────────────────────
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
        # Allow per-instance override of beta_a0 (PSF ARD prior shape parameter).
        # Default 1.0 (3× stronger than MATLAB's 0.0).  Larger → sparser PSF.
        PAR['betamodel'] = (self.beta_a0, PAR['betamodel'][1])
        # pyramid_thresh: applied before each upsampling step in mc_restoration to
        # suppress halo pixels before the anti-alias Gaussian spreads them to the
        # next (finer) level.  Use half of kernel_thresh so it is conservative.
        params['pyramid_thresh'] = self.kernel_thresh * 0.5
        return params

    # ── Main entry point ────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] grayscale ────────────────────
        y = np.asarray(image).astype(np.float64)
        if y.max() > 1.0:
            y = y / 255.0
        if y.ndim == 3:
            # collapse to grayscale (mean over channels — matches demo_run.m)
            y = y.mean(axis=2)

        # Hard upper bound on PSF size — odd
        ksz = self.kernel_size
        if ksz % 2 == 0:
            ksz += 1
        hsize = (ksz, ksz)

        params = self._build_params()

        # ── 2. Blind PSF estimation (multiscale) ────────────────────────
        kernel, _gamma_vec = mc_restoration(y, hsize, params)
        if kernel.ndim == 3:
            kernel = kernel[:, :, 0]

        # ── 3. Kernel denoising ──────────────────────────────────────────
        # VB estimation leaves small scattered values around the true PSF
        # support ("shadow").  Hard-threshold everything below kernel_thresh
        # fraction of the peak and renormalise so the PSF still sums to 1.
        # kernel_thresh=0.0 disables thresholding entirely.
        if self.kernel_thresh > 0.0:
            thresh = kernel.max() * self.kernel_thresh
            kernel = np.where(kernel >= thresh, kernel, 0.0)
            ksum = kernel.sum()
            if ksum > 1e-10:
                kernel = kernel / ksum

        # ── 4. Non-blind deconvolution on full image ────────────────────
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
            # frils_deb_ubc handles padding internally (pad_replicate by
            # kernel half-size) and returns the FOV-cropped image directly.
            latent = frils_deb_ubc(y, kernel, firls_opt)
            latent = np.clip(latent, 0.0, 1.0)
        else:
            # Original VB non-blind step.
            # The ARD mask zeros gamma_vec on a border of width
            # floor(kernel_size/2), producing a black frame.  Fix: pad with
            # reflect before deconvolution, then crop off.
            hh0 = kernel.shape[0] // 2
            hh1 = kernel.shape[1] // 2
            if hh0 > 0 or hh1 > 0:
                y_padded = np.pad(y, ((hh0, hh0), (hh1, hh1)), mode='reflect')
            else:
                y_padded = y
            latent, _report = vb_deconv([y_padded], [kernel], params)
            latent = latent[hh0: hh0 + y.shape[0], hh1: hh1 + y.shape[1]]
            latent = np.clip(latent, 0.0, 1.0)

        # ── 4. Output ───────────────────────────────────────────────────
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

        # Safety: nan_to_num before cast — should not trigger if the
        # algorithm is numerically healthy, but prevents a silent black image.
        latent = np.nan_to_num(latent, nan=0.0, posinf=1.0, neginf=0.0)
        x_final = latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ───────────────────────────────────────────────
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
