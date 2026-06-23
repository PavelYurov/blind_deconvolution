"""
Variational Bayesian Blind Image Deconvolution with Total Variation Prior.

Источник:
[1] Babacan, S.D., Molina, R., Katsaggelos, A.K.
    "Variational Bayesian blind deconvolution using a total
    variation prior."
    IEEE Trans. Image Processing, 18(1), 12-26, 2009.
    DOI: 10.1109/TIP.2008.2005443

[2] Chantas, G., Galatsanos, N.P., Molina, R., Katsaggelos, A.K.
    "Variational Bayesian image restoration with a product of
    spatially weighted total variation image priors."
    IEEE Trans. Image Processing, 19(2), 351-362, 2010.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .utils import fft_convolve, compute_tv, compute_elbo, center_kernel_mass, edgetaper
from .solvers import (
    compute_tv_weights,
    solve_image_vb,
    solve_kernel_vb,
    update_hyperparameters_vb,
)

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

class CGMRF_VB_TV_BID(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        noise_sigma: float = 0.01,
        max_iter: int = 40,
        cg_max_iter: int = 30,
        cg_tol: float = 1e-6,
        epsilon: float = 1e-4,
        delta_h_init: float = 10.0,
        kernel_threshold: float = 0.05,
        n_trace_probes: int = 0,
        update_hyperparams: bool = True,
        use_edgetaper: bool = True,
        verbose: bool = False,
    ):
        super().__init__(name='VB-TV-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.epsilon = epsilon
        self.delta_h_init = delta_h_init
        self.kernel_threshold = kernel_threshold
        self.n_trace_probes = n_trace_probes
        self.update_hyperparams_flag = update_hyperparams
        self.use_edgetaper = use_edgetaper
        self.verbose = verbose

        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'alpha': [],
            'beta': [],
            'delta_h': [],
            'residual': [],
            'tv_value': [],
            'elbo': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        rng = np.random.default_rng(42)

        y_orig = image.astype(np.float64)
        if y_orig.max() > 1.0:
            y_orig /= 255.0

        H, W = y_orig.shape
        kh, kw = self.kernel_shape

        sig = max(kh, kw) / 8.0
        grid_y, grid_x = np.ogrid[-kh // 2:kh // 2, -kw // 2:kw // 2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sig**2))
        h /= h.sum()

        if self.use_edgetaper:
            y = edgetaper(y_orig, h)
        else:
            y = y_orig

        f = y.copy()

        beta = 1.0 / (self.noise_sigma**2 + 1e-9)
        alpha = 5.0 * beta / (H * W)
        if alpha < 1e-3: alpha = 1e-3

        delta_h = self.delta_h_init

        w = compute_tv_weights(f, self.epsilon)

        tr_Sigma_Q = 0.0
        tr_Sigma_HtH = 0.0
        log_det_Sigma = 0.0
        h_energy = float(np.sum(h**2))
        h_cov_trace = 0.0

        if self.verbose:
            print(f"[{self.name}] Start. Size: {H}x{W}, Edgetaper: {self.use_edgetaper}")

        n_iter = 0
        current_thresh = self.kernel_threshold

        for it in range(self.max_iter):
            n_iter = it + 1
            h_prev = h.copy()

            w = compute_tv_weights(f, self.epsilon)

            f, img_info = solve_image_vb(
                y, h, f, alpha, beta, w,
                cg_max_iter=self.cg_max_iter,
                cg_tol=self.cg_tol,
                n_trace_probes=self.n_trace_probes,
                rng=rng,
            )
            tr_Sigma_Q = img_info['tr_Sigma_Q']
            tr_Sigma_HtH = img_info['tr_Sigma_HtH']
            log_det_Sigma = img_info['log_det_Sigma']

            h, kern_info = solve_kernel_vb(
                y, f, self.kernel_shape, delta_h, beta,
                threshold_ratio=current_thresh
            )
            h_energy = kern_info['h_energy']
            h_cov_trace = kern_info['h_cov_trace']

            h = center_kernel_mass(h)

            if self.update_hyperparams_flag:
                alpha, beta, delta_h = update_hyperparameters_vb(
                    y, f, h, w, alpha, beta, delta_h,
                    tr_Sigma_Q, tr_Sigma_HtH,
                    h_energy, h_cov_trace,
                )

            elbo = compute_elbo(
                y, f, h, alpha, beta, delta_h, w,
                tr_Sigma_Q, tr_Sigma_HtH, log_det_Sigma,
                h_cov_trace,
            )

            diff = float(np.linalg.norm(h - h_prev))
            residual_norm = float(np.linalg.norm(y - fft_convolve(f, h)))
            tv_val = compute_tv(f, self.epsilon)

            self.history['kernel_diff'].append(diff)
            self.history['alpha'].append(alpha)
            self.history['beta'].append(beta)
            self.history['delta_h'].append(delta_h)
            self.history['residual'].append(residual_norm)
            self.history['tv_value'].append(tv_val)
            self.history['elbo'].append(elbo)

            if self.verbose:
                print(f"Iter {it+1}: dH={diff:.5f}, a={alpha:.2e}, b={beta:.2e}, dh={delta_h:.2e}, ELBO={elbo:.2e}")

            if diff < 1e-7 and it > 15:
                break

        w = compute_tv_weights(f, self.epsilon)
        f_final, _ = solve_image_vb(
            y, h, f, alpha, beta, w,
            cg_max_iter=self.cg_max_iter * 3,
            cg_tol=1e-7,
            n_trace_probes=0,
            rng=rng,
        )

        elapsed = time.time() - start_time

        self.hyperparams = {
            'alpha': alpha,
            'beta': beta,
            'delta_h': delta_h,
            'noise_sigma': self.noise_sigma,
            'kernel_threshold': self.kernel_threshold,
            'iterations': n_iter,
        }

        f_final = np.clip(f_final * 255.0, 0, 255)
        return f_final.astype(np.int16), h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('noise_sigma', self.noise_sigma),
            ('max_iter', self.max_iter),
            ('delta_h_init', self.delta_h_init),
            ('kernel_threshold', self.kernel_threshold),
            ('use_edgetaper', self.use_edgetaper),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams

def run_algorithm(g, kernel_shape, **kwargs):
    algo = CGMRF_VB_TV_BID(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
