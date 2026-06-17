"""
vdbke.py — Variational Dirichlet Blur Kernel Estimation.

Multi-scale blind deconvolution framework wrapper.

Ported from ``ms_ngm_dirichlet_ubc_img.m`` by X. Zhou et al.
Reference:
    X. Zhou, J. Mateos, F. Zhou, R. Molina, A.K. Katsaggelos:
    "Variational Dirichlet Blur Kernel Estimation",
    IEEE TIP, vol. 24, no. 12, pp. 5127-5139, 2015.
"""

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
from .utils import (psf2otf, fspecial_gaussian, imresize,
                    rgb2gray, rgb2ycbcr, ycbcr2rgb)
from .solvers import (center_kernel_img_space, ss_ngm_dirichlet_ubc_img,
                      firls_deb_ubc)
from scipy.signal import convolve2d


class VDBKE(DeconvolutionAlgorithm):
    """
    Variational Dirichlet Blur Kernel Estimation (VDBKE).

    Multi-scale blind deconvolution followed by non-blind deconvolution.
    Ported from ``ms_ngm_dirichlet_ubc_img.m``.
    """

    def __init__(
        self,
        kernel_size=(25, 25),
        gamma_correct: float = 1.0,
        use_ycbcr: bool = True,
        kernel_est_win=None,
        # ── kernel estimation parameters ──
        kernel_lambda: float = 1e-6,
        kernel_max_iter: int = 20,
        kernel_back_alpha: float = 0.01,
        kernel_back_beta: float = 0.5,
        kernel_lower_bound: float = 1.0,
        kernel_ng_min: float = 1e-5,
        kernel_cost_display: int = 0,
        kernel_mode: int = 0,
        kernel_Laplacian_filter=None,
        # NOTE on ``kernel_lambda_C``.
        # Originally I set this to 100.0 to match Sun-dataset/real-data
        # tests in the paper, but those tests use real-world 640×640
        # blurred photos with motion kernels of 13–27 px.  The user's
        # pipeline applies *synthetic* motion/defocus blur on much
        # smaller images, where ``lambda_C=100`` over-regularises the
        # kernel and introduces a sub-pixel shift bias («ступеньки» /
        # «звон» in the deblurred image).  Default 0 = pure Dirichlet
        # prior; advanced users can set 0.01 (Levin-style) or 100
        # (Sun-style) explicitly.
        kernel_lambda_C: float = 0.0,
        # ── image estimation parameters ──
        # NOTE on ``img_lambda1``.
        # Lowering this to 0.0002 (Sun default) on the user's pipeline
        # under-regularises the latent image so the kernel absorbs
        # noise-driven gradients and develops a faint halo («тень»).
        # 0.002 is a calibrated middle ground that works well for the
        # synthetic-blur scenarios used here.
        img_lambda1: float = 0.002,
        img_lambda_min: float = 0.01,
        img_lambda_max: float = 1.0,
        img_IF: float = None,
        img_N1: int = 20,
        img_N2: int = 2,
        img_lambda_u: float = 0.1,
        img_xv_iter: int = 1,
        img_cost_display: int = 0,
        # ── alternating iteration parameters ──
        xk_iter: int = 20,
        k_tol: float = 5e-4,
        # ── non-blind deconvolution (FIRLS) parameters ──
        # NOTE on ``firls_lambda``.
        # 0.0002 (Sun) under-regularises the inversion on the user's
        # pipeline and produces visible ringing / staircase artefacts
        # near edges.  0.002 keeps the inversion stable while still
        # leaving high-frequency detail thanks to the hyper-Laplacian
        # prior (alpha = 2/3).
        firls_lambda: float = 0.002,
        firls_alpha: float = 2.0 / 3.0,
        firls_out_iter: int = 5,
        firls_inner_iter: int = 4,
    ):
        super().__init__(name='VDBKE')

        self.kernel_size = tuple(kernel_size) if not isinstance(kernel_size, tuple) else kernel_size
        self.gamma_correct = gamma_correct
        self.use_ycbcr = use_ycbcr
        self.kernel_est_win = kernel_est_win

        # Kernel estimation
        self.kernel_lambda = kernel_lambda
        self.kernel_max_iter = kernel_max_iter
        self.kernel_back_alpha = kernel_back_alpha
        self.kernel_back_beta = kernel_back_beta
        self.kernel_lower_bound = kernel_lower_bound
        self.kernel_ng_min = kernel_ng_min
        self.kernel_cost_display = kernel_cost_display
        self.kernel_mode = kernel_mode
        # Default Laplacian filter: identity (Gaussian prior on kernel)
        if kernel_Laplacian_filter is None:
            self.kernel_Laplacian_filter = np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        else:
            self.kernel_Laplacian_filter = np.asarray(kernel_Laplacian_filter, dtype=np.float64)
        self.kernel_lambda_C = kernel_lambda_C

        # Image estimation
        self.img_lambda1 = img_lambda1
        self.img_lambda_min = img_lambda_min
        self.img_lambda_max = img_lambda_max
        self.img_IF = img_IF if img_IF is not None else np.sqrt(2)
        self.img_N1 = img_N1
        self.img_N2 = img_N2
        self.img_lambda_u = img_lambda_u
        self.img_xv_iter = img_xv_iter
        self.img_cost_display = img_cost_display

        # Alternating
        self.xk_iter = xk_iter
        self.k_tol = k_tol

        # FIRLS
        self.firls_lambda = firls_lambda
        self.firls_alpha = firls_alpha
        self.firls_out_iter = firls_out_iter
        self.firls_inner_iter = firls_inner_iter

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────
    # process — main entry point  (← ms_ngm_dirichlet_ubc_img.m)
    # ─────────────────────────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # Convert to float64 [0, 1]
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        yorig = y.copy()

        # Gamma correction
        y = y ** self.gamma_correct

        # Convert to grayscale for kernel estimation
        if self.kernel_est_win is not None:
            w = self.kernel_est_win  # (r1, c1, r2, c2) 0-indexed
            if y.ndim == 3 and y.shape[2] == 3:
                y = rgb2gray(y[w[0]:w[2], w[1]:w[3], :])
            else:
                y = y[w[0]:w[2], w[1]:w[3]]
        else:
            if y.ndim == 3 and y.shape[2] == 3:
                y = rgb2gray(y)

        blur_size = self.kernel_size  # (ks1, ks2)

        # ── Determine kernel sizes at each scale ──
        # MATLAB: [max_ks, ind1] = max(opts.kernel_size)
        max_ks = max(blur_size)
        ind1 = 0 if blur_size[0] >= blur_size[1] else 1
        ind2 = 1 - ind1

        minsize = [0, 0]
        minsize[ind1] = max(3, 2 * ((max_ks - 1) // 64) + 1)
        temp = int(np.floor(blur_size[ind2] / blur_size[ind1] * minsize[ind1]))
        if temp % 2 == 0:
            temp += 1
        minsize[ind2] = max(temp, 3)

        print(f'Kernel size at coarsest level is [{minsize[0]}, {minsize[1]}]')

        resize_step = np.sqrt(2)
        # Build ksize list for each scale
        ksize = []
        tmp = minsize[ind1]
        while tmp < max_ks:
            ks_entry = [0, 0]
            ks_entry[ind1] = tmp
            tmp2 = int(np.ceil(blur_size[ind2] / blur_size[ind1] * tmp))
            if tmp2 % 2 == 0:
                tmp2 += 1
            ks_entry[ind2] = max(tmp2, 3)
            ksize.append(tuple(ks_entry))

            tmp = int(np.ceil(tmp * resize_step))
            if tmp % 2 == 0:
                tmp += 1

        ksize.append(tuple(blur_size))
        num_scales = len(ksize)

        # Storage per scale
        ks = [None] * num_scales
        alphas = [None] * num_scales
        ls = [None] * num_scales

        lambda_C = self.kernel_lambda_C

        # ── Multi-scale loop ──
        for s in range(num_scales):
            k1, k2 = ksize[s]

            if s == 0:
                # Coarsest level: initialise kernel as Gaussian
                Gsigma = 1.0 if max_ks > 50 else 0.5
                ks[s] = fspecial_gaussian((k1, k2), Gsigma)
                alphas[s] = ks[s] + self.kernel_lower_bound
            else:
                # Up-sample kernel from previous level
                tmp_k = ks[s - 1].copy()
                tmp_k[tmp_k < 0] = 0
                tmp_k /= tmp_k.sum()
                ks[s] = imresize(tmp_k, (k1, k2), 'bilinear')
                alphas[s] = imresize(alphas[s - 1], (k1, k2), 'bilinear')
                ks[s][ks[s] < 0] = 0
                ks[s] /= ks[s].sum()

            # Image size at this level
            r = int(np.floor(y.shape[0] * k1 / blur_size[0]))
            c = int(np.floor(y.shape[1] * k2 / blur_size[1]))
            if s == num_scales - 1:
                r, c = y.shape[0], y.shape[1]

            print(f'Processing scale {s + 1}/{num_scales}; '
                  f'kernel size {k1}x{k2}; image size {r}x{c}')

            # Resize y to current scale
            ys = imresize(y, (r, c), 'bilinear')

            if s == 0:
                ls[s] = ys.copy()
            else:
                ls[s] = imresize(ls[s - 1], (r, c), 'bilinear')

            # Lambda_C schedule
            if s == num_scales - 1:
                cur_lambda_C = lambda_C
            else:
                cur_lambda_C = (lambda_C * ksize[s][0] * ksize[s][1]
                                / (ksize[-1][0] * ksize[-1][1]))

            # Centre the kernel
            ls[s], ks[s], shift_kernel = center_kernel_img_space(ls[s], ks[s])
            alphas[s] = np.maximum(
                convolve2d(alphas[s], shift_kernel, 'same'),
                self.kernel_lower_bound)

            # Build parameter dicts for this scale
            kernel_pars = {
                'lambda': self.kernel_lambda,
                'max_iter': self.kernel_max_iter,
                'back_alpha': self.kernel_back_alpha,
                'back_beta': self.kernel_back_beta,
                'lower_bound': self.kernel_lower_bound,
                'ng_min': self.kernel_ng_min,
                'cost_display': self.kernel_cost_display,
                'mode': self.kernel_mode,
                'Laplacian_filter': self.kernel_Laplacian_filter,
                'lambda_C': cur_lambda_C,
                'alpha0': alphas[s],
            }

            img_pars = {
                'lambda1': self.img_lambda1,
                'lambda_min': self.img_lambda_min,
                'lambda_max': self.img_lambda_max,
                'IF': self.img_IF,
                'N1': self.img_N1,
                'N2': self.img_N2,
                'lambda_u': self.img_lambda_u,
                'xv_iter': self.img_xv_iter,
                'cost_display': self.img_cost_display,
                'x0': ls[s].copy(),
            }

            pars = {
                'xk_iter': self.xk_iter,
                'img_pars': img_pars,
                'kernel_pars': kernel_pars,
                'k_tol': self.k_tol,
            }

            # Single-scale alternating estimation
            ls[s], ks[s], alphas[s] = ss_ngm_dirichlet_ubc_img(
                ys, ls[s], ks[s], alphas[s], pars)

            # At finest scale, extract final kernel
            if s == num_scales - 1:
                kernel = alphas[s] - self.kernel_lower_bound
                kernel = kernel / kernel.sum()

        # ── Non-blind deconvolution ──
        firls_opts = {
            'lambda': self.firls_lambda,
            'alpha': self.firls_alpha,
            'out_iter': self.firls_out_iter,
            'inner_iter': self.firls_inner_iter,
        }

        if self.use_ycbcr:
            if yorig.ndim == 3 and yorig.shape[2] == 3:
                ycbcr = rgb2ycbcr(yorig)
            else:
                ycbcr = yorig.copy()

            if ycbcr.ndim == 3 and ycbcr.shape[2] == 3:
                x_fov, _, _ = firls_deb_ubc(ycbcr[:, :, 0], kernel, firls_opts)
                deblur = ycbcr.copy()
                deblur[:, :, 0] = x_fov
                deblur[:, :, 1:3] = ycbcr[:, :, 1:3]
                deblur = ycbcr2rgb(deblur)
            else:
                x_fov, _, _ = firls_deb_ubc(ycbcr, kernel, firls_opts)
                deblur = x_fov
        else:
            if yorig.ndim == 3:
                deblur = yorig.copy()
                for j in range(yorig.shape[2]):
                    x_fov, _, _ = firls_deb_ubc(yorig[:, :, j], kernel, firls_opts)
                    deblur[:, :, j] = x_fov
            else:
                x_fov, _, _ = firls_deb_ubc(yorig, kernel, firls_opts)
                deblur = x_fov

        deblur = np.clip(deblur, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'kernel_lambda': self.kernel_lambda,
            'img_lambda1': self.img_lambda1,
            'firls_lambda': self.firls_lambda,
            'time': time.time() - start_time,
        }

        # Output: int16 [0, 255], kernel
        x_final = np.clip(deblur * 255.0, 0, 255).astype(np.int16)
        return x_final, kernel

    # ─────────────────────────────────────────────────────────────────────
    # Framework interface methods
    # ─────────────────────────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('gamma_correct', self.gamma_correct),
            ('use_ycbcr', self.use_ycbcr),
            ('kernel_lambda', self.kernel_lambda),
            ('kernel_max_iter', self.kernel_max_iter),
            ('kernel_lower_bound', self.kernel_lower_bound),
            ('kernel_lambda_C', self.kernel_lambda_C),
            ('img_lambda1', self.img_lambda1),
            ('img_lambda_min', self.img_lambda_min),
            ('img_lambda_max', self.img_lambda_max),
            ('xk_iter', self.xk_iter),
            ('k_tol', self.k_tol),
            ('firls_lambda', self.firls_lambda),
            ('firls_alpha', self.firls_alpha),
            ('firls_out_iter', self.firls_out_iter),
            ('firls_inner_iter', self.firls_inner_iter),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_size':
                    self.kernel_size = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
