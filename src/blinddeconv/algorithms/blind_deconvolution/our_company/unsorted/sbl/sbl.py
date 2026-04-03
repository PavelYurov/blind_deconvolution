"""
sbl.py

Generalized Sparse Bayesian Learning (SBL) for Image Deconvolution.

Reference:
    J. Glaubitz, A. Gelb, G. Song: "Generalized sparse Bayesian learning
    and application to image reconstruction",
    SIAM/ASA J. Uncertainty Quantification, 11(1):262-284, 2023.
    arXiv:2201.07061

Pipeline (mirrors MATLAB script_deconvolution_2d.m):
    1. Normalise input to float64 [0, 1].
    2. Build 1-D forward operator F_1d (Gaussian convolution kernel
       parameterised by *gamma*).
    3. Build TV regularisation operator D of given *order*.
    4. Reconstruct via BCD_2d (Bayesian Coordinate Descent).
    5. Construct the 2-D spatial PSF from *gamma* for output.
    6. Return restored image (int16, [0, 255]) and kernel.

Note:
    This algorithm is *non-blind* — it assumes a Gaussian blur model
    with parameter gamma.  The kernel is NOT estimated from the image;
    it is constructed from the user-supplied gamma.  The algorithm
    estimates the *image* given the (assumed known) blur model.
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

from .utils import construct_F_deconvolution, TV_operator
from .solvers import BCD_2d


class SBL_BD(DeconvolutionAlgorithm):
    """
    Image deconvolution via Generalized Sparse Bayesian Learning (BCD).

    The algorithm models blur as convolution with a Gaussian kernel
    (parameter *gamma*) and recovers the image by solving a hierarchical
    Bayesian inverse problem with a TV-type sparsity prior.

    Parameters
    ----------
    gamma : float
        Blurring parameter (std-dev of the Gaussian convolution kernel).
        Controls how "wide" the assumed PSF is.  Larger values mean
        heavier blur.  Default 0.015 (from script_deconvolution_2d.m).
    tv_order : int
        Order of the Total-Variation regularisation operator (1, 2, or 3).
        Default 2 (from script_deconvolution_2d.m).
    c : float
        Hyper-hyper-parameter of the Gamma prior on inverse variances.
        Default 1.0.
    d : float
        Hyper-hyper-parameter of the Gamma prior on inverse variances.
        Default 1e-2 (from script_deconvolution_2d.m).
    kernel_size : int
        Spatial size of the output PSF kernel (square, odd).
        This only affects the *returned* kernel visualisation, not the
        reconstruction itself (which uses the full n×n forward operator).
        Default 31.
    quiet : bool
        Suppress iteration log.  Default True.
    """

    def __init__(
        self,
        gamma: float = 0.015,
        tv_order: int = 2,
        c: float = 1.0,
        d: float = 1e-2,
        kernel_size: int = 31,
        quiet: bool = True,
    ):
        super().__init__(name='SBL-BCD')

        self.gamma = gamma
        self.tv_order = tv_order
        self.c = c
        self.d = d
        self.kernel_size = kernel_size
        self.quiet = quiet

        self.history: Dict[str, list] = {'abs_error': [], 'rel_error': []}
        self.hyperparams: Dict[str, Any] = {}

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct a grayscale image from blurred observations.

        Parameters
        ----------
        image : (H, W) ndarray — blurred grayscale image (uint8 or float).

        Returns
        -------
        x_final : (H, W) int16 — restored image, values in [0, 255].
        kernel  : (kernel_size, kernel_size) float64 — assumed PSF.
        """
        start_time = time.time()

        # ── 1. Normalise to float64 [0, 1] ──────────────────────────────
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        # Ensure 2-D grayscale
        if y.ndim == 3:
            y = y[:, :, 0]

        H, W = y.shape
        n = H  # The forward operator is built for an n×n image

        # ── 2. Build forward operator ────────────────────────────────────
        # MATLAB: F_1d = construct_F_deconvolution(n, gamma)
        # The forward model is separable: Y = F_1d * X * F_1d' + noise
        # Here Y (the blurred image) is supplied as input.
        #
        # For non-square images we build separate operators per axis.
        # The MATLAB code assumes square images (n×n).
        if H != W:
            # For non-square: build separate 1-D operators, but BCD_2d
            # assumes square.  Crop/pad to square for now.
            n = min(H, W)
            y = y[:n, :n]

        F_1d = construct_F_deconvolution(n, self.gamma)

        # ── 3. Build regularisation operator ─────────────────────────────
        # MATLAB: D = TV_operator(n, order)
        D = TV_operator(n, self.tv_order)

        # ── 4. Reconstruct via BCD_2d ────────────────────────────────────
        # MATLAB: [Mu, alpha, B1, B2, history] = BCD_2d(F_1d, Y, D, c, d)
        Mu, alpha, B1, B2, bcd_history = BCD_2d(
            F_1d, y, D, self.c, self.d, quiet=self.quiet,
        )

        self.history = bcd_history

        # Clip to [0, 1]
        Mu = np.clip(Mu, 0.0, 1.0)

        # ── 5. Construct the spatial PSF kernel for output ───────────────
        kernel = self._build_psf_kernel(self.gamma, self.kernel_size)

        # ── 6. Output ────────────────────────────────────────────────────
        self.hyperparams = {
            'gamma': self.gamma,
            'tv_order': self.tv_order,
            'c': self.c,
            'd': self.d,
            'alpha': float(alpha),
            'iterations': len(bcd_history['abs_error']),
            'time': time.time() - start_time,
        }

        # Pad back if we cropped
        if H != W:
            result = np.zeros((H, W), dtype=np.float64)
            result[:n, :n] = Mu
            Mu = result

        x_final = Mu * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    # ── Helper: build a 2-D Gaussian PSF ─────────────────────────────────
    @staticmethod
    def _build_psf_kernel(gamma: float, size: int) -> np.ndarray:
        """
        Construct a 2-D Gaussian PSF kernel matching the 1-D kernel
        used in construct_F_deconvolution.

        The 1-D kernel is:
            k(t) = exp(-t^2 / (2*gamma^2)) / sqrt(2*pi*gamma^2)

        The 2-D separable kernel is k(x)*k(y), normalised to sum = 1.

        Parameters
        ----------
        gamma : float — blur parameter.
        size  : int   — kernel spatial extent (pixels).

        Returns
        -------
        psf : (size, size) float64, normalised to sum = 1.
        """
        # The MATLAB grid covers [0, 1]; pixel spacing = 1/(n-1).
        # For the PSF we centre at 0 and use the same scale.
        # A reasonable spatial extent: [-0.5, 0.5] mapped to *size* pixels.
        half = (size - 1) / 2.0
        ax = np.arange(size, dtype=np.float64) - half
        # Scale: map pixel indices to [0,1] coordinates.
        # In the MATLAB code, grid spacing = 1/(n-1).  We use a matching
        # scale so that gamma has the same meaning.
        # For a kernel_size-pixel window centred at 0, using spacing 1/(size-1):
        ax = ax / (size - 1) if size > 1 else ax
        xx, yy = np.meshgrid(ax, ax)
        psf = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * gamma ** 2))
        psf /= psf.sum()
        return psf

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('gamma', self.gamma),
            ('tv_order', self.tv_order),
            ('c', self.c),
            ('d', self.d),
            ('kernel_size', self.kernel_size),
            ('quiet', self.quiet),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
