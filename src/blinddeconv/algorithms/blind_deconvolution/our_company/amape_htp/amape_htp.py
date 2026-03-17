"""
amape_htp.py

Blind Image Deblurring Using Alternating Maximum A Posteriori Estimation
with Heavy-Tailed Priors (AMAPE-HTP).

Reference:
    J. Kotera, F. Sroubek, P. Milanfar:
    "Blind deconvolution using alternating maximum a posteriori estimation
    with heavy-tailed priors", DOI: 10.1007/978-3-642-40246-3_8

Ported from C++ code by Suzuki Hironobu (Blind-Deblurring-master).

Pipeline (mirrors C++ BlindDeblur::exe):
    1. Normalise input to float64 [0, 1].
    2. Crop ROI to maxROIsize.
    3. Multi-scale blind deconvolution:
       for L = 1..MSlevels:
         a. Normalize PSF
         b. Downsample ROI for scale L
         c. PSFestimaLnoRgrad (Ustep + Hstep alternation)
         d. Double PSF for next scale
    4. Center-of-gravity centering of final PSF.
    5. Non-blind deconvolution (fftCGSRaL) on full image.
    6. Denormalize and return restored image (int16) and kernel.
"""

import numpy as np
import math
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
    """
    Blind deconvolution using Alternating MAP Estimation with
    Heavy-Tailed Priors (Kotera et al., 2013).

    Parameters
    ----------
    psf_size        : int — spatial support of the unknown PSF (square).
                      Default 32 (from C++ init_param).
    maxROIsize_r    : int — max ROI rows for kernel estimation.
                      Default 1024.
    maxROIsize_c    : int — max ROI cols for kernel estimation.
                      Default 1024.
    MSlevels        : int — number of multi-scale pyramid levels.
                      Default 4.
    gamma           : float — initial ALM penalty parameter.
                      Default 1e2.
    Lp              : float — Lp exponent for image prior (0 < Lp < 1).
                      Default 0.3.
    beta_h_factor   : float — beta_h = beta_h_factor * gamma.
                      Default 1e4.
    alpha_h_factor  : float — alpha_h = alpha_h_factor * gamma.
                      Default 1e1.
    centering_threshold : float — threshold for PSF centering.
                      Default 30/255.
    beta_u_factor   : float — beta_u = beta_u_factor * gamma.
                      Default 1e0.
    alpha_u_factor  : float — alpha_u = alpha_u_factor * gamma.
                      Default 1e-2.
    gamma_nonblind_factor : float — gamma_nonblind = gamma_nonblind_factor * gamma.
                      Default 2e3.
    beta_u_nonblind_factor : float — beta_u_nonblind = beta_u_nonblind_factor * gamma_nonblind.
                      Default 1.0.
    Lp_nonblind     : float — Lp exponent for non-blind restoration.
                      Default 0.0.
    maxiter_u       : int — max inner iterations for U-step.
                      Default 10.
    maxiter_h       : int — max inner iterations for H-step.
                      Default 10.
    maxiter         : int — max outer alternating iterations per scale.
                      Default 5.
    ccreltol        : float — relative convergence tolerance.
                      Default 1e-3.
    """

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
        """Build the param dict consumed by solvers, from stored attributes."""
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

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the full AMAPE-HTP blind deconvolution pipeline.

        Parameters
        ----------
        image : np.ndarray
            Grayscale blurred image, 2D (H x W), uint8 or float.

        Returns
        -------
        restored : np.ndarray (int16)
            Restored image, values in [0, 255].
        kernel  : np.ndarray (float64)
            Estimated PSF, normalized to sum=1.
        """
        start_time = time.time()
        param = self._build_param()

        # ── 1. Convert to float64 ─────────────────────────────────────
        # C++: img_g.convertTo(src, CV_64F)  — raw pixel values as doubles
        Src = image.astype(np.float64)

        # ── 2. Normalize to [0, 1] ────────────────────────────────────
        # C++: normalizeImage(Src, _min, _max)
        Src, _min, _max = normalizeImage(Src)

        # ── 3. Get ROI ────────────────────────────────────────────────
        # C++: MatrixXd ROI = getROI(Src, param)
        ROI = getROI(Src, self.maxROIsize_r, self.maxROIsize_c)

        # ── 4. Create initial PSF ─────────────────────────────────────
        # C++: h_rows = psf_size_x >> (MSlevels - 1)
        #      h_cols = psf_size_y >> (MSlevels - 1)
        #      h = Zero(h_rows, h_cols)
        #      cen_r = floor((h.rows()+1)/2 - 1)
        #      cen_c = floor((h.cols()+1)/2 - 1)
        #      h(cen_r, cen_c) = 1.0
        h_rows = self.psf_size >> (self.MSlevels - 1)
        h_cols = self.psf_size >> (self.MSlevels - 1)

        h = np.zeros((h_rows, h_cols), dtype=np.float64)
        cen_r = int(math.floor((h_rows + 1) / 2 - 1))
        cen_c = int(math.floor((h_cols + 1) / 2 - 1))
        h[cen_r, cen_c] = 1.0

        # ── 5. Multi-scale estimation loop ────────────────────────────
        # C++: for (int L = 1; L <= param.MSlevels; L++)
        for L in range(1, self.MSlevels + 1):
            # Normalize PSF
            # C++: matNormalize(h)
            h = matNormalize(h)

            # Create ROI for this scale
            # C++: MatrixXd tmp = ROI; createROI(tmp, L, param);
            tmp = ROI.copy()
            tmp = createROI(tmp, L, self.MSlevels)

            # Convert to complex (with off-by-one from C++ copy_mat_2_cmat)
            # C++: MatrixXcd cROI = Zero(tmp.rows(), tmp.cols());
            #      copy_mat_2_cmat(tmp, cROI, tmp.rows(), tmp.cols());
            cROI = copy_mat_2_cmat(tmp, (tmp.shape[0], tmp.shape[1]),
                                   tmp.shape[0], tmp.shape[1])

            # Estimate PSF at this scale
            # C++: PSFestimaLnoRgrad(h, cROI, param, L)
            h = PSFestimaLnoRgrad(h, cROI, param, L)

            # Double PSF for next scale (except last)
            # C++: if (L != param.MSlevels) doublePSF(h)
            if L != self.MSlevels:
                h = doublePSF(h)

        # ── 6. Final PSF normalization ────────────────────────────────
        # C++: matNormalize(h)
        h = matNormalize(h)

        # ── 7. Center-of-gravity centering ────────────────────────────
        # C++ lines 232-254 in exe():
        #   double g_x = 0.0, g_y = 0.0;
        #   for (int x = 0; x < h.cols(); x++)
        #     for (int y = 0; y < h.rows(); y++) {
        #       g_x += x * h(x,y); g_y += y * h(x,y); }
        #   int shift_x = h.rows()/2 - (int)floor(g_x);
        #   int shift_y = h.cols()/2 - (int)floor(g_y);
        #   ... shift h by (shift_x, shift_y)
        #
        # NOTE: The C++ loop has x < h.cols() and y < h.rows() but indexes
        # h(x,y) — this means x is the row index going up to cols(), and
        # y is the row index going up to rows(). This is a bug in the C++
        # (rows/cols swapped in loop bounds) but we replicate exactly.
        g_x = 0.0
        g_y = 0.0
        nr, nc = h.shape
        for x in range(nc):      # C++: x < h.cols()
            for y in range(nr):   # C++: y < h.rows()
                g_x += x * h[x, y]
                g_y += y * h[x, y]

        # C++: int shift_x = h.rows()/2 - (int)floor(g_x)
        #      int shift_y = h.cols()/2 - (int)floor(g_y)
        # (int) cast in C++ truncates toward zero; for positive values
        # floor() already gives int, so int(floor()) is fine.
        shift_x = nr // 2 - int(math.floor(g_x))
        shift_y = nc // 2 - int(math.floor(g_y))

        # C++: tmp(x+shift_x, y+shift_y) = h(x,y)
        tmp_h = np.zeros_like(h)
        for x in range(nr):
            for y in range(nc):
                nx = x + shift_x
                ny = y + shift_y
                if 0 <= nx < nr and 0 <= ny < nc:
                    tmp_h[nx, ny] = h[x, y]
        h = tmp_h

        # ── 8. Non-blind deconvolution ────────────────────────────────
        # C++ calls divide_rgb_images to split channels, normalize each,
        # then fftCGSRaL per channel. For grayscale, we already have
        # Src normalized to [0,1], so we call fftCGSRaL directly.
        # C++: fftCGSRaL(g_src, h, gU, param)
        U = fftCGSRaL(Src, h, param)

        # ── 9. Denormalize ────────────────────────────────────────────
        # C++ make_image: val = real(U(x,y)) * (max - min) + min
        #                 then cv::normalize to [0,1]
        # For our framework: denormalize to original range, then scale to [0,255]
        restored = np.real(U) * (_max - _min) + _min
        # Clamp to [0, 255] and convert to int16
        restored = np.clip(restored, 0.0, 255.0)
        x_final = np.round(restored).astype(np.int16)

        # ── 10. Record hyperparams ────────────────────────────────────
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

    # ── Interface methods ────────────────────────────────────────────────
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
