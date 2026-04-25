"""
lip.py

Blind Image Deconvolution via Lower-Bounded Logarithmic Image Priors (LIP).

Reference:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", International Conference on
    Energy Minimization Methods in Computer Vision and Pattern Recognition
    (EMMCVPR), 2015.

Implements three methods from the paper:
    MM  — Majorization-Minimization (Table 2): gradient descent on the
          EM-majorised weighted-TV subproblem.
    CV  — Condat-Vũ splitting on the MM-majorised weighted-TV subproblem
          (data fidelity via spatial convolutions, no FFT).
    PD  — Paper-faithful Primal-Dual (Table 1): solves the non-convex
          log-TV energy directly via Chambolle-Pock / Möllenhoff
          splitting (no MM outer majorisation).
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

from .solvers import coarse_to_fine
from .utils import (
    gamma_correction,
    make_size_odd,
    edgetaper,
    pad_image,
    crop_image,
    wiener_filter,
    tikhonov_filter,
)


class LIP_BD(DeconvolutionAlgorithm):
    """
    Blind deconvolution using the Logarithmic Image Prior (MM algorithm).

    Pipeline (mirrors MATLAB ``deblur.m`` → ``coarseToFine.m`` → ``blind.m``):
        1. Normalise input to float64 [0, 1].
        2. Trim image to odd dimensions.
        3. (Optional) gamma correction.
        4. Coarse-to-fine PSF estimation (MM or PD method).
        5. Non-blind restoration with the estimated kernel (Tikhonov or Wiener).
        6. Return restored image (int16, [0, 255]) and kernel.

    Parameters
    ----------
    kernel_shape : (MK, NK) — spatial support of the unknown PSF.
    lambda_val   : data-fidelity weight (β in the paper).
                   Default 30000 (from main_levin.m benchmark).
    tau          : lower-bound parameter of the log prior (default 1e-3).
    outer_iters  : EM outer iterations per pyramid level (default 140).
    inner_iters  : gradient-descent inner iterations per outer (default 5).
    k_step       : kernel step-size schedule (list/array).
    u_step       : image step-size schedule (list/array).
    lambda_mult  : λ multiplier between pyramid levels (default 2.1).
    scale_mult   : kernel-size divider between pyramid levels (default √2).
    gamma_correction : whether to apply gamma correction (default False).
    gamma        : gamma exponent (used when gamma_correction=True).
    method       : 'mm' (gradient-descent, Table 2) or 'pd' (Condat-Vũ, Table 1).
    kernel_threshold : fraction of max(k) below which kernel values are zeroed (default 0.05).
    final_deconv : 'tikhonov' or 'wiener'.
    final_alpha  : regularisation strength for the non-blind step.
    verbose      : print progress during coarse-to-fine.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_val: float = 30000.0,
        tau: float = 1e-3,
        outer_iters: int = 140,
        inner_iters: int = 5,
        k_step: Any = None,
        u_step: Any = None,
        lambda_mult: float = 2.1,
        scale_mult: float = 1.4142135623730951,  # sqrt(2)
        gamma_correction: bool = False,
        gamma: float = 1.0,
        method: str = 'mm',
        kernel_threshold: float = 0.05,
        final_deconv: str = 'tikhonov',
        final_alpha: float = 0.001,
        verbose: bool = False,
        # ── paper-faithful PD (method='pd') extras ──
        pd_outer_iters: int = 30,
        pd_inner_iters: int = 50,
        pd_theta: float = 1.0,
        pd_tau: float = None,
        pd_sigma: float = None,
        h_mode: str = 'closed',
        h_lut_size: int = 4096,
        h_lut_xi_max: float = 4.0,
    ):
        super().__init__(name='LIP-BD')

        self.kernel_shape = tuple(kernel_shape)
        self.lambda_val = lambda_val
        self.tau = tau
        self.outer_iters = outer_iters
        self.inner_iters = inner_iters

        # Step-size schedules — defaults from deblur.m
        if k_step is None:
            self.k_step = np.array([1e-2, 5e-3, 1e-3, 5e-4])
        else:
            self.k_step = np.atleast_1d(np.asarray(k_step, dtype=np.float64))
        if u_step is None:
            self.u_step = np.array([1e-2, 5e-3, 1e-3, 1e-3])
        else:
            self.u_step = np.atleast_1d(np.asarray(u_step, dtype=np.float64))

        self.lambda_mult = lambda_mult
        self.scale_mult = scale_mult
        self.gamma_corr = gamma_correction
        self.gamma = gamma
        self.method = method.lower()
        self.kernel_threshold = kernel_threshold
        self.final_deconv = final_deconv.lower()
        self.final_alpha = final_alpha
        self.verbose = verbose

        # PD-specific (method='pd', paper-faithful)
        self.pd_outer_iters = int(pd_outer_iters)
        self.pd_inner_iters = int(pd_inner_iters)
        self.pd_theta = float(pd_theta)
        self.pd_tau = pd_tau
        self.pd_sigma = pd_sigma
        self.h_mode = str(h_mode)
        self.h_lut_size = int(h_lut_size)
        self.h_lut_xi_max = float(h_lut_xi_max)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        MK, NK = self.kernel_shape

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        # MATLAB: f = im2double(f)
        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        M_orig, N_orig = f.shape

        # ── 2. Trim to odd dimensions ───────────────────────────────────
        # MATLAB: if mod(M,2)==0, f=f(1:end-1,:); ...
        f = make_size_odd(f)

        # ── 3. Gamma correction (optional) ──────────────────────────────
        if self.gamma_corr:
            f = gamma_correction(f, self.gamma)

        # ── 4. PSF estimation (blind step) ──────────────────────────────
        if self.method in ('mm', 'cv'):
            blind_params = {
                'outer_iters': self.outer_iters,
                'inner_iters': self.inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
            }
            ctf_params = {
                'final_lambda': self.lambda_val,
                'lambda_mult': self.lambda_mult,
                'scale_mult': self.scale_mult,
            }
            u, k = coarse_to_fine(f, MK, NK, blind_params, ctf_params,
                                  verbose=self.verbose, method=self.method)
        elif self.method == 'pd':
            # Paper-faithful PD has its own iteration counts
            blind_params = {
                'outer_iters': self.pd_outer_iters,
                'inner_iters': self.pd_inner_iters,
                'tau': self.tau,
                'k_step': self.k_step,
                'u_step': self.u_step,
                'pd_theta': self.pd_theta,
                'pd_tau': self.pd_tau,
                'pd_sigma': self.pd_sigma,
                'h_mode': self.h_mode,
                'h_lut_size': self.h_lut_size,
                'h_lut_xi_max': self.h_lut_xi_max,
            }
            ctf_params = {
                'final_lambda': self.lambda_val,
                'lambda_mult': self.lambda_mult,
                'scale_mult': self.scale_mult,
            }
            u, k = coarse_to_fine(f, MK, NK, blind_params, ctf_params,
                                  verbose=self.verbose, method='pd')
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. Choose 'mm', 'pd', or 'cv'.")

        # ── 4b. Kernel thresholding ─────────────────────────────────────
        # Remove low-intensity noise from the estimated kernel.
        # Standard post-processing step (Cho & Lee 2009, Krishnan 2011).
        k[k < self.kernel_threshold * k.max()] = 0.0
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        # ── 5. Non-blind restoration ────────────────────────────────────
        # The MATLAB code refers to an external non-blind step
        # (deconvSps from Levin et al.).  Here we use Tikhonov / Wiener.
        #
        # NOTE: The rot90(k,2) in the MATLAB main_levin.m was needed for
        # the specific convention of deconvSps.  Our Tikhonov / Wiener
        # filters expect the PSF in the standard convolution form, which
        # is exactly what coarse_to_fine returns.  No rotation needed.

        # Pad image and taper edges to reduce FFT boundary ringing
        f_pad = pad_image(f, (MK, NK))
        f_pad = edgetaper(f_pad, k)

        if self.final_deconv == 'tikhonov':
            u_restored = tikhonov_filter(f_pad, k, alpha=self.final_alpha)
        elif self.final_deconv == 'wiener':
            u_restored = wiener_filter(f_pad, k, noise_snr=self.final_alpha)
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'tikhonov' or 'wiener'."
            )

        # Crop back to original size
        u_final = crop_image(u_restored, (M_orig, N_orig), (MK, NK))
        u_final = np.clip(u_final, 0.0, 1.0)

        # ── 6. Output ──────────────────────────────────────────────────
        self.hyperparams = {
            'lambda': self.lambda_val,
            'tau': self.tau,
            'method': self.method,
            'final_deconv': self.final_deconv,
            'final_alpha': self.final_alpha,
            'outer_iters': self.outer_iters,
            'inner_iters': self.inner_iters,
            'time': time.time() - start_time,
        }

        x_final = u_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_val', self.lambda_val),
            ('tau', self.tau),
            ('outer_iters', self.outer_iters),
            ('inner_iters', self.inner_iters),
            ('method', self.method),
            ('kernel_threshold', self.kernel_threshold),
            ('final_deconv', self.final_deconv),
            ('final_alpha', self.final_alpha),
            ('gamma_correction', self.gamma_corr),
            ('gamma', self.gamma),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                elif key in ('k_step', 'u_step'):
                    setattr(self, key, np.atleast_1d(
                        np.asarray(value, dtype=np.float64)))
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
