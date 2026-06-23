"""
Источник::
    Tzikas D.G., Likas A.C., Galatsanos N.P. (2009).
    "Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution
     With Student's-t Priors."  IEEE Trans. Image Process., 18(1), 200–208.

    Chantas G., Galatsanos N., Likas A., Saunders M. (2008).
    "Variational Bayesian Image Restoration Based on a Product of
     t-Distributions Image Prior."  IEEE Trans. Image Process., 17(10).
"""

import numpy as np
from numpy.fft import fft2, ifft2
import time
from typing import Tuple, List, Any, Dict

from .utils import (
    EPSILON,
    psf2otf,
    build_rbf_basis,
    build_gradient_filters,
    build_initial_psf,
    initial_wiener,
    compute_cross_correlation_matrix,
    compute_cross_correlation_vector,
    compute_covariance_matrix_Df,
    spectral_covariance,
    compute_autocovariance,
    edge_taper,
)
from .solvers import (
    solve_image_pcg,
    solve_weights_direct,
    update_alpha,
    update_gamma,
    update_beta,
    prune_ard,
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

class VBSKB_BID_STP(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: Tuple[int, int],

        sigma_phi_sq: float = 0.1,

        sigma_h_sq: float = 3.0,
        init_alpha: float = 1e-16,
        init_beta: float = 1e3,
        init_gamma: float = 1e2,

        a0_alpha: float = 1e-6,
        b0_alpha: float = 1e-6,
        a0_gamma: float = 1e-6,
        b0_gamma: float = 1e-6,
        a0_beta: float = 1e-6,
        b0_beta: float = 1e-6,

        max_iter: int = 100,
        tol: float = 1e-5,
        min_iter: int = 5,
        num_filters: int = 2,
        ard_threshold: float = 1e10,
        cg_maxiter: int = 100,
        cg_tol: float = 1e-6,
        gamma_max: float = 1e6,
        nonblind_iters: int = 20,
        verbose: bool = False,
    ):
        super().__init__(name='VBSKB-BID-STP')

        self.kernel_shape = tuple(kernel_shape)
        self.sigma_phi_sq = sigma_phi_sq
        self.sigma_h_sq = sigma_h_sq
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.init_gamma = init_gamma
        self.a0_alpha = a0_alpha
        self.b0_alpha = b0_alpha
        self.a0_gamma = a0_gamma
        self.b0_gamma = b0_gamma
        self.a0_beta = a0_beta
        self.b0_beta = b0_beta
        self.max_iter = max_iter
        self.tol = tol
        self.min_iter = min_iter
        self.num_filters = num_filters
        self.ard_threshold = ard_threshold
        self.cg_maxiter = cg_maxiter
        self.cg_tol = cg_tol
        self.gamma_max = gamma_max
        self.nonblind_iters = nonblind_iters
        self.verbose = verbose

        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'beta': [],
            'num_bases': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        g = image.astype(np.float64)
        if g.max() > 1.0:
            g /= 255.0
        H, W = g.shape
        N = H * W
        kh, kw = self.kernel_shape

        if self.verbose:
            print(f"[{self.name}] Image {H}×{W}, Kernel {kh}×{kw}")

        g_orig = g.copy()

        g = edge_taper(g, self.kernel_shape)

        Phi = build_rbf_basis(self.kernel_shape, self.sigma_phi_sq)
        M = Phi.shape[1]

        Q_fft, Q_fft_sq = build_gradient_filters((H, W), self.num_filters)
        K = len(Q_fft)

        h0 = build_initial_psf(self.kernel_shape, self.sigma_h_sq)

        mu_w = np.linalg.lstsq(Phi, h0.ravel(), rcond=None)[0]

        alpha = np.full(M, self.init_alpha)
        beta = self.init_beta
        gamma_bar = np.full(K, self.init_gamma)

        gamma_maps = [np.full((H, W), self.init_gamma) for _ in range(K)]

        mu_f = initial_wiener(g, h0, beta, gamma_bar, Q_fft_sq)

        Sigma_w = np.eye(M) / (self.init_alpha + EPSILON)

        if self.verbose:
            print(f"[{self.name}] Init done. M={M}, K={K}")

        mu_w_prev = mu_w.copy()

        for it in range(self.max_iter):

            h_vec = Phi @ mu_w
            h_2d = h_vec.reshape(self.kernel_shape)
            h_fft = psf2otf(h_2d, (H, W))

            mu_f = solve_image_pcg(
                g, h_fft, mu_f, beta,
                gamma_maps, Q_fft, Q_fft_sq,
                cg_maxiter=self.cg_maxiter,
                cg_tol=self.cg_tol,
            )

            Sigma_f_hat = spectral_covariance(
                h_fft, gamma_bar, Q_fft_sq, beta
            )
            r_f = compute_autocovariance(Sigma_f_hat)

            FtF = compute_cross_correlation_matrix(mu_f, self.kernel_shape)
            D_f = compute_covariance_matrix_Df(r_f, self.kernel_shape)
            FtF_plus_Df = FtF + D_f

            Ftg = compute_cross_correlation_vector(mu_f, g, self.kernel_shape)

            mu_w, Sigma_w = solve_weights_direct(
                Phi, FtF_plus_Df, Ftg, alpha, beta
            )

            alpha = update_alpha(
                mu_w, Sigma_w, self.a0_alpha, self.b0_alpha
            )

            gamma_maps, gamma_bar = update_gamma(
                mu_f, Q_fft, Sigma_f_hat, Q_fft_sq,
                self.a0_gamma, self.b0_gamma,
                gamma_max=self.gamma_max,
            )

            beta = update_beta(
                g, h_fft, mu_f, Sigma_f_hat,
                Sigma_w, Phi,
                self.a0_beta, self.b0_beta,
            )

            alpha, Phi, mu_w, keep_mask = prune_ard(
                alpha, Phi, mu_w, threshold=self.ard_threshold
            )
            if not np.all(keep_mask):

                Sigma_w = Sigma_w[np.ix_(keep_mask, keep_mask)]
                M = mu_w.shape[0]
                if self.verbose:
                    print(f"  Iter {it+1}: pruned → M={M}")

            delta_w = np.linalg.norm(mu_w - mu_w_prev[:mu_w.shape[0]]) / (
                np.linalg.norm(mu_w) + EPSILON
            )
            self.history['kernel_diff'].append(delta_w)
            self.history['beta'].append(beta)
            self.history['num_bases'].append(M)

            if self.verbose:
                gb_str = ', '.join(f'{v:.1e}' for v in gamma_bar)
                print(
                    f"  Iter {it+1}/{self.max_iter}:  Δw={delta_w:.2e}  "
                    f"β={beta:.2e}  M={M}  γ̄=[{gb_str}]"
                )

            if delta_w < self.tol and it >= self.min_iter:
                if self.verbose:
                    print(f"  Converged at iteration {it+1}.")
                break

            mu_w_prev = np.zeros_like(mu_w)
            mu_w_prev[:] = mu_w

        h_vec = Phi @ mu_w
        h_est = h_vec.reshape(self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = h_est.sum()
        if h_sum > EPSILON:
            h_est /= h_sum

        ph, pw_ = kh, kw
        g_pad = np.pad(g_orig, ((ph, ph), (pw_, pw_)), mode='reflect')
        Hp, Wp = g_pad.shape
        Np = Hp * Wp

        Q_nb_fft, Q_nb_fft_sq = build_gradient_filters(
            (Hp, Wp), self.num_filters
        )
        K_nb = len(Q_nb_fft)
        h_est_fft_p = psf2otf(h_est, (Hp, Wp))
        prior_weight_nb = 1.0 / K_nb

        gamma_bar_nb = gamma_bar.copy()
        gamma_maps_nb = [
            np.full((Hp, Wp), gamma_bar[k]) for k in range(K_nb)
        ]

        mu_f_pad = initial_wiener(
            g_pad, h_est, beta, gamma_bar_nb, Q_nb_fft_sq,
            prior_weight=prior_weight_nb,
        )

        if self.verbose:
            print(f"[{self.name}] Non-blind refinement "
                  f"({self.nonblind_iters} iters, pad={ph},{pw_}, "
                  f"1/P={prior_weight_nb:.3f}) …")

        for nb_it in range(self.nonblind_iters):
            mu_f_prev = mu_f_pad.copy()

            mu_f_pad = solve_image_pcg(
                g_pad, h_est_fft_p, mu_f_pad, beta,
                gamma_maps_nb, Q_nb_fft, Q_nb_fft_sq,
                cg_maxiter=self.cg_maxiter * 2,
                cg_tol=self.cg_tol * 0.1,
                prior_weight=prior_weight_nb,
            )

            Sigma_f_hat_p = spectral_covariance(
                h_est_fft_p, gamma_bar_nb, Q_nb_fft_sq, beta,
                prior_weight=prior_weight_nb,
            )

            gamma_maps_nb, gamma_bar_nb = update_gamma(
                mu_f_pad, Q_nb_fft, Sigma_f_hat_p, Q_nb_fft_sq,
                self.a0_gamma, self.b0_gamma,
                gamma_max=self.gamma_max,
                prior_weight=prior_weight_nb,
            )

            H_mu_f_p = np.real(ifft2(h_est_fft_p * fft2(mu_f_pad)))
            residual_sq = float(np.sum((g_pad - H_mu_f_p) ** 2))
            trace_term = float(
                np.sum(np.abs(h_est_fft_p) ** 2 * Sigma_f_hat_p)
            )
            a_tilde_beta = self.a0_beta + 0.5 * Np
            b_tilde_beta = self.b0_beta + 0.5 * (residual_sq + trace_term)
            beta = a_tilde_beta / (b_tilde_beta + EPSILON)
            beta = float(np.clip(beta, 1e-1, 1e8))

            delta_f = np.linalg.norm(mu_f_pad - mu_f_prev) / (
                np.linalg.norm(mu_f_pad) + EPSILON
            )

            if self.verbose:
                gb_str = ', '.join(f'{v:.1e}' for v in gamma_bar_nb)
                print(
                    f"  NB {nb_it+1}/{self.nonblind_iters}:  "
                    f"β={beta:.2e}  Δf={delta_f:.2e}  γ̄=[{gb_str}]"
                )

            if delta_f < 1e-5 and nb_it >= 2:
                if self.verbose:
                    print(f"  Non-blind converged at iteration {nb_it+1}.")
                break

        mu_f = mu_f_pad[ph:ph+H, pw_:pw_+W]

        elapsed = time.time() - start_time
        self.timer = elapsed

        self.hyperparams = {
            'beta': beta,
            'gamma_bar': gamma_bar.tolist(),
            'num_active_bases': int(M),
            'iterations': it + 1,
            'elapsed_sec': elapsed,
        }

        f_est = np.clip(mu_f, 0.0, 1.0) * 255.0
        f_est = np.round(f_est).astype(np.int16)
        return f_est, h_est

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('sigma_phi_sq', self.sigma_phi_sq),
            ('sigma_h_sq', self.sigma_h_sq),
            ('init_beta', self.init_beta),
            ('init_gamma', self.init_gamma),
            ('max_iter', self.max_iter),
            ('num_filters', self.num_filters),
            ('ard_threshold', self.ard_threshold),
            ('tol', self.tol),
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

def run_algorithm(
    g: np.ndarray,
    kernel_shape: Tuple[int, int],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:

    algo = VBSKB_BID_STP(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
