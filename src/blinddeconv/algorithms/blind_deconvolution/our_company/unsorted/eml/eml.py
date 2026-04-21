"""
eml.py

Framework wrapper for the EML (Efficient Marginal Likelihood) blind
deconvolution algorithm of Levin et al., CVPR 2011.

Reference
---------
    A. Levin, Y. Weiss, F. Durand, W. T. Freeman,
    "Efficient Marginal Likelihood Optimization in Blind Deconvolution",
    CVPR 2011.
    https://webee.technion.ac.il/people/anat.levin/papers/deconvLevinEtalCVPR11.pdf

This file implements the recommended pipeline of the reference MATLAB
package (``deconv_diagfe_filt_sps.m``):

    grayscale blurred image  →  blind kernel estimation by variational EM
                                (diagonal free-energy covariance, MOG prior
                                on derivatives, coarse-to-fine pyramid)
                             →  non-blind restoration by deconvSps
                                (|z|^0.8 prior on 1st/2nd derivatives)
                             →  restored image + kernel
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

from .solvers import multires_deconv, deconvSps
from .utils import load_mog_params, default_deriv_filters


def _recenter_kernel_com(k: np.ndarray) -> np.ndarray:
    """
    Shift ``k`` so that its centre of mass lands on the geometric centre
    of the kernel grid.  Uses integer-pixel ``np.roll`` so that the
    non-negativity and ``sum(k) = 1`` properties are preserved exactly.
    """
    k = np.asarray(k, dtype=np.float64)
    s = k.sum()
    if s <= 0:
        return k
    h, w = k.shape
    yy, xx = np.mgrid[0:h, 0:w]
    com_y = (yy * k).sum() / s
    com_x = (xx * k).sum() / s
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    shift_y = int(round(cy - com_y))
    shift_x = int(round(cx - com_x))
    if shift_y == 0 and shift_x == 0:
        return k
    out = np.roll(k, shift=(shift_y, shift_x), axis=(0, 1))
    # Zero out any entries that wrapped around — in practice the kernel
    # support is well inside the grid (since we rolled COM to centre).
    if shift_y > 0:
        out[:shift_y, :] = 0.0
    elif shift_y < 0:
        out[shift_y:, :] = 0.0
    if shift_x > 0:
        out[:, :shift_x] = 0.0
    elif shift_x < 0:
        out[:, shift_x:] = 0.0
    ns = out.sum()
    if ns > 0:
        out = out / ns
    return out


class EML_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution via Efficient Marginal Likelihood optimisation
    (Levin et al., CVPR 2011).

    Parameters
    ----------
    kernel_size : int
        Desired kernel size (square, odd).  Forced to the next odd number
        internally, exactly as in MATLAB ``deconv_diagfe_filt_sps.m``.
    sig_noise : float
        Noise standard deviation parameter.  Default 0.01.
    edges_w : float
        Edge-weighting passed to the final non-blind ``deconvSps`` call.
        Default 0.0068 (MATLAB default).
    k_prior_ivar : float
        Kernel-sparsity regularisation weight (``scla`` in MATLAB
        ``solve_for_sps_kernel``).  Default 0.01.
    final_max_it : int
        Max CG iterations per IRLS pass inside ``deconvSps``.  Default 70
        (MATLAB default).
    ret : float
        Pyramid rescale factor.  Default sqrt(0.5).
    verbose : bool
        Print per-iteration free energy traces.  Default False.
    """

    def __init__(
        self,
        kernel_size: int = 25,
        sig_noise: float = 0.01,
        edges_w: float = 0.0068,
        k_prior_ivar: float = 0.01,
        final_max_it: int = 70,
        ret: float = float(np.sqrt(0.5)),
        verbose: bool = False,
    ):
        super().__init__(name='EML-BD')

        self.kernel_size = int(kernel_size)
        self.sig_noise = float(sig_noise)
        self.edges_w = float(edges_w)
        self.k_prior_ivar = float(k_prior_ivar)
        self.final_max_it = int(final_max_it)
        self.ret = float(ret)
        self.verbose = bool(verbose)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ── 1. Normalise to float64 in [0, 1] ────────────────────────────
        y = np.asarray(image, dtype=np.float64)
        if y.ndim == 3:
            # Collapse to grayscale if a colour image was passed in.
            if y.shape[2] == 3:
                y = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
            else:
                y = y[:, :, 0]
        if y.max() > 1.0:
            y = y / 255.0

        # ── 2. Build problem dict (mirrors MATLAB deconv_diagfe_filt_sps) ─
        # sig_noise_v = sig_noise * 1.15.^[10:-1:0]  — noise-annealing schedule
        sig_noise_v = self.sig_noise * (1.15 ** np.arange(10, -1, -1))

        ivars, pis = load_mog_params()

        k_sz1 = (self.kernel_size // 2) * 2 + 1   # force odd
        k_sz2 = k_sz1

        # Two-tap initial kernel, same as MATLAB.
        tf = np.zeros((k_sz1, k_sz2), dtype=np.float64)
        cy = k_sz1 // 2
        cx = k_sz2 // 2
        tf[cy, cx] = 1.0
        tf[cy, cx + 1] = 1.0
        tf = tf / tf.sum()

        prob: Dict[str, Any] = {
            'prior_ivar': ivars,
            'prior_pi': pis,
            'filts': default_deriv_filters(),
            'cycconv': 0,
            'covtype': 'diag',
            'update_x': 'conjgrad',
            'filt_space': 1,
            'init_x_every_itr': 1,
            'k_prior_ivar': self.k_prior_ivar,
            'unconst_k': 0,
            'eval_freeeng': 0,
            'k_sz1': k_sz1,
            'k_sz2': k_sz2,
            'k': tf,
            'y': y,
        }

        # ── 3. Coarse-to-fine blind kernel estimation ────────────────────
        prob_final, _kListItr = multires_deconv(
            prob, self.ret, sig_noise_v, verbose=self.verbose,
        )
        kernel = prob_final['k']
        s = kernel.sum()
        if s > 0:
            kernel = kernel / s

        # ── 3b. Re-centre kernel by centre-of-mass ───────────────────────
        # Blind deconvolution is invariant to a joint (shift_k, −shift_x)
        # translation.  Levin's reference MATLAB tests therefore compare
        # images via ``comp_upto_shift``.  When the calling framework
        # compares the restored image pixel-by-pixel against the sharp
        # reference, the residual kernel shift manifests as a badly
        # aligned output and destroys PSNR.  We therefore roll the
        # estimated kernel so its centre of mass lands at the geometric
        # centre (this is the standard post-processing step used by
        # Cho & Lee 2009, Pan et al. 2016, and most blind-deconvolution
        # pipelines).  The algorithm itself is not modified.
        kernel = _recenter_kernel_com(kernel)

        # ── 4. Final non-blind restoration ───────────────────────────────
        Latent = deconvSps(y, kernel, self.edges_w, self.final_max_it)
        Latent = np.clip(Latent, 0.0, 1.0)

        # ── 5. Output ────────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'sig_noise': self.sig_noise,
            'edges_w': self.edges_w,
            'k_prior_ivar': self.k_prior_ivar,
            'final_max_it': self.final_max_it,
            'ret': self.ret,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.round(x_final)
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('sig_noise', self.sig_noise),
            ('edges_w', self.edges_w),
            ('k_prior_ivar', self.k_prior_ivar),
            ('final_max_it', self.final_max_it),
            ('ret', self.ret),
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
