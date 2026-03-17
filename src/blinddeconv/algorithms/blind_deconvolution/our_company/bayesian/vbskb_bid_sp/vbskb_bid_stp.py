"""
Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution
with Student's-t Priors  (VBSKB-BID-STP).

Framework wrapper following the ``DeconvolutionAlgorithm`` interface.

Based on:
    Tzikas D.G., Likas A.C., Galatsanos N.P. (2009).
    "Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution
     With Student's-t Priors."  IEEE Trans. Image Process., 18(1), 200–208.

    Chantas G., Galatsanos N., Likas A., Saunders M. (2008).
    "Variational Bayesian Image Restoration Based on a Product of
     t-Distributions Image Prior."  IEEE Trans. Image Process., 17(10).

Modules:
    - utils   : FFT helpers, RBF basis, gradient filters, spectral approx.
    - solvers : PCG image solver, direct weight solver, hyperparameter updates.
    - vbskb_bid_stp (this file) : orchestrating class ``VBSKB_BID_STP``.
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

# ── Robust import of the project-wide base class ─────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
#  Main algorithm class
# ═══════════════════════════════════════════════════════════════════════════

class VBSKB_BID_STP(DeconvolutionAlgorithm):
    """Blind Image Deconvolution via Variational Bayes with Sparse Kernel
    and Student's-t Priors.

    The algorithm alternates:

    1. **VE-step**
       a) Update image posterior  q(f) = N(μ_f, Σ_f)  via preconditioned CG
          with per-pixel Student-t gradient priors  [Eq. 34–35].
       b) Update kernel-weight posterior  q(w) = N(μ_w, Σ_w)  via direct
          Cholesky solve with RVM-style ARD prior  [Eq. 32–33].

    2. **VM-step**
       a) Update ARD precisions  ⟨α_i⟩  [Eq. 36–37, 42].
       b) Update gradient precisions  ⟨γ_j^k⟩  [Eq. 38–39, 44].
       c) Update noise precision  ⟨β⟩  [Eq. 40–41, 46].

    3. Optionally **prune** RBF basis functions whose α_i exceeds a
       threshold (Automatic Relevance Determination, Tipping 2001).

    Parameters are initialised following Sec. IV-D of the paper.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # ── RBF basis ────────────────────────────────────────────────
        sigma_phi_sq: float = 0.1,
        # ── Initialisation (Sec. IV-D) ──────────────────────────────
        sigma_h_sq: float = 3.0,
        init_alpha: float = 1e-16,
        init_beta: float = 1e3,
        init_gamma: float = 1e2,
        # ── Prior hyper-hyper-parameters (non-informative) ──────────
        a0_alpha: float = 1e-6,
        b0_alpha: float = 1e-6,
        a0_gamma: float = 1e-6,
        b0_gamma: float = 1e-6,
        a0_beta: float = 1e-6,
        b0_beta: float = 1e-6,
        # ── Algorithm control ────────────────────────────────────────
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

        # Store all settings
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

        # Diagnostics
        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'beta': [],
            'num_bases': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    #  process()  —  main entry point
    # ──────────────────────────────────────────────────────────────────────

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Deconvolve a single greyscale blurred image.

        Parameters
        ----------
        image : ndarray (H, W)
            Observed blurred image (uint8 [0,255] or float [0,1]).

        Returns
        -------
        restored : ndarray (H, W), int16, [0, 255]
            Restored image.
        kernel   : ndarray (kh, kw), float64
            Estimated PSF, normalised, non-negative.
        """
        start_time = time.time()

        # ── 0. Prepare data ──────────────────────────────────────────────
        g = image.astype(np.float64)
        if g.max() > 1.0:
            g /= 255.0
        H, W = g.shape
        N = H * W
        kh, kw = self.kernel_shape

        if self.verbose:
            print(f"[{self.name}] Image {H}×{W}, Kernel {kh}×{kw}")

        # Keep original for non-blind (padding instead of tapering)
        g_orig = g.copy()

        # Edge tapering to mitigate circular-convolution boundary artefacts
        g = edge_taper(g, self.kernel_shape)

        # ── 1. Build RBF basis Φ  (Eq. 4–7) ─────────────────────────────
        Phi = build_rbf_basis(self.kernel_shape, self.sigma_phi_sq)
        M = Phi.shape[1]                       # number of basis functions

        # ── 2. Precompute gradient filters  Q̂^k, |Q̂^k|²  ─────────────
        Q_fft, Q_fft_sq = build_gradient_filters((H, W), self.num_filters)
        K = len(Q_fft)

        # ── 3. Initialisation  (Sec. IV-D) ──────────────────────────────
        # PSF
        h0 = build_initial_psf(self.kernel_shape, self.sigma_h_sq)
        # Initial weights via least-squares fit  w₀ ≈ (Φ^T Φ)⁻¹ Φ^T h₀
        mu_w = np.linalg.lstsq(Phi, h0.ravel(), rcond=None)[0]

        # Hyperparameters
        alpha = np.full(M, self.init_alpha)     # ARD precisions
        beta = self.init_beta                    # noise precision
        gamma_bar = np.full(K, self.init_gamma)  # mean gradient precisions

        # Per-pixel gamma maps  (all constant initially)
        gamma_maps = [np.full((H, W), self.init_gamma) for _ in range(K)]

        # Image mean  (Wiener warm-start, matches Eq. 60 preconditioner)
        mu_f = initial_wiener(g, h0, beta, gamma_bar, Q_fft_sq)

        # Sigma_w placeholder (will be computed in the first iteration)
        Sigma_w = np.eye(M) / (self.init_alpha + EPSILON)

        if self.verbose:
            print(f"[{self.name}] Init done. M={M}, K={K}")

        # ── 4. Main VB loop ──────────────────────────────────────────────
        mu_w_prev = mu_w.copy()

        for it in range(self.max_iter):

            # ····· Current PSF from weights ·····························
            h_vec = Phi @ mu_w                          # h = Φ μ_w
            h_2d = h_vec.reshape(self.kernel_shape)
            h_fft = psf2otf(h_2d, (H, W))

            # ============================================================
            #  VE-STEP  (a)  Update q(f)   [Eq. 34–35]
            # ============================================================
            mu_f = solve_image_pcg(
                g, h_fft, mu_f, beta,
                gamma_maps, Q_fft, Q_fft_sq,
                cg_maxiter=self.cg_maxiter,
                cg_tol=self.cg_tol,
            )

            # Spectral covariance approximation  Σ̂_f  (Eq. 54–55)
            Sigma_f_hat = spectral_covariance(
                h_fft, gamma_bar, Q_fft_sq, beta
            )
            r_f = compute_autocovariance(Sigma_f_hat)

            # ============================================================
            #  VE-STEP  (b)  Update q(w)   [Eq. 32–33]
            # ============================================================
            # Build  F̄^T F̄  and  D_f  (Eq. 54)
            FtF = compute_cross_correlation_matrix(mu_f, self.kernel_shape)
            D_f = compute_covariance_matrix_Df(r_f, self.kernel_shape)
            FtF_plus_Df = FtF + D_f

            # Build  F̄^T g
            Ftg = compute_cross_correlation_vector(mu_f, g, self.kernel_shape)

            mu_w, Sigma_w = solve_weights_direct(
                Phi, FtF_plus_Df, Ftg, alpha, beta
            )

            # ============================================================
            #  VM-STEP  (a)  Update α_i  (ARD)  [Eq. 36–37, 42]
            # ============================================================
            alpha = update_alpha(
                mu_w, Sigma_w, self.a0_alpha, self.b0_alpha
            )

            # ============================================================
            #  VM-STEP  (b)  Update γ_j^k   [Eq. 38–39, 44]
            # ============================================================
            gamma_maps, gamma_bar = update_gamma(
                mu_f, Q_fft, Sigma_f_hat, Q_fft_sq,
                self.a0_gamma, self.b0_gamma,
                gamma_max=self.gamma_max,
            )

            # ============================================================
            #  VM-STEP  (c)  Update β   [Eq. 40–41, 46]
            # ============================================================
            beta = update_beta(
                g, h_fft, mu_f, Sigma_f_hat,
                Sigma_w, Phi,
                self.a0_beta, self.b0_beta,
            )

            # ============================================================
            #  ARD pruning  (Tipping 2001)
            # ============================================================
            alpha, Phi, mu_w, keep_mask = prune_ard(
                alpha, Phi, mu_w, threshold=self.ard_threshold
            )
            if not np.all(keep_mask):
                # Shrink Sigma_w accordingly
                Sigma_w = Sigma_w[np.ix_(keep_mask, keep_mask)]
                M = mu_w.shape[0]
                if self.verbose:
                    print(f"  Iter {it+1}: pruned → M={M}")

            # ============================================================
            #  Convergence monitoring
            # ============================================================
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

            # Prepare for next iteration
            mu_w_prev = np.zeros_like(mu_w)
            mu_w_prev[:] = mu_w

        # ── 5. Reconstruct final PSF ─────────────────────────────────────
        h_vec = Phi @ mu_w
        h_est = h_vec.reshape(self.kernel_shape)
        h_est = np.maximum(h_est, 0.0)
        h_sum = h_est.sum()
        if h_sum > EPSILON:
            h_est /= h_sum

        # ── 6. Non-blind VB refinement  (symmetric padding) ─────────
        #  Use reflect-padded original image (no edge taper) so that
        #  circular convolution wraps in the mirrored border region
        #  instead of producing visible boundary artefacts.
        ph, pw_ = kh, kw             # pad by full kernel size
        g_pad = np.pad(g_orig, ((ph, ph), (pw_, pw_)), mode='reflect')
        Hp, Wp = g_pad.shape
        Np = Hp * Wp

        # Filters and PSF OTF for the padded domain
        Q_nb_fft, Q_nb_fft_sq = build_gradient_filters(
            (Hp, Wp), self.num_filters
        )
        K_nb = len(Q_nb_fft)
        h_est_fft_p = psf2otf(h_est, (Hp, Wp))
        prior_weight_nb = 1.0 / K_nb

        # Initialise gamma maps for padded size from blind-loop averages
        gamma_bar_nb = gamma_bar.copy()
        gamma_maps_nb = [
            np.full((Hp, Wp), gamma_bar[k]) for k in range(K_nb)
        ]

        # Wiener warm-start on padded image
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

            # VE-step q(f)  —  Chantas Eq. (3.10)–(3.11)
            mu_f_pad = solve_image_pcg(
                g_pad, h_est_fft_p, mu_f_pad, beta,
                gamma_maps_nb, Q_nb_fft, Q_nb_fft_sq,
                cg_maxiter=self.cg_maxiter * 2,
                cg_tol=self.cg_tol * 0.1,
                prior_weight=prior_weight_nb,
            )

            # Spectral covariance  Σ̂_f  (padded domain)
            Sigma_f_hat_p = spectral_covariance(
                h_est_fft_p, gamma_bar_nb, Q_nb_fft_sq, beta,
                prior_weight=prior_weight_nb,
            )

            # VM-step γ  —  Chantas Eq. (3.14)–(3.15)
            gamma_maps_nb, gamma_bar_nb = update_gamma(
                mu_f_pad, Q_nb_fft, Sigma_f_hat_p, Q_nb_fft_sq,
                self.a0_gamma, self.b0_gamma,
                gamma_max=self.gamma_max,
                prior_weight=prior_weight_nb,
            )

            # VM-step β  —  Chantas 2008: H is known, no kernel uncertainty.
            H_mu_f_p = np.real(ifft2(h_est_fft_p * fft2(mu_f_pad)))
            residual_sq = float(np.sum((g_pad - H_mu_f_p) ** 2))
            trace_term = float(
                np.sum(np.abs(h_est_fft_p) ** 2 * Sigma_f_hat_p)
            )
            a_tilde_beta = self.a0_beta + 0.5 * Np
            b_tilde_beta = self.b0_beta + 0.5 * (residual_sq + trace_term)
            beta = a_tilde_beta / (b_tilde_beta + EPSILON)
            beta = float(np.clip(beta, 1e-1, 1e8))

            # Convergence check
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

        # Crop back to original image size
        mu_f = mu_f_pad[ph:ph+H, pw_:pw_+W]

        # ── 7. Post-processing ───────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    #  Framework interface methods
    # ──────────────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone convenience function
# ═══════════════════════════════════════════════════════════════════════════

def run_algorithm(
    g: np.ndarray,
    kernel_shape: Tuple[int, int],
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """Run VBSKB-BID-STP on a single greyscale image.

    Parameters
    ----------
    g : ndarray (H, W)
        Blurred observation.
    kernel_shape : (kh, kw)
        Assumed PSF support size.
    **kwargs
        Forwarded to ``VBSKB_BID_STP.__init__``.

    Returns
    -------
    f_est      : ndarray (H, W), int16
    h_est      : ndarray (kh, kw)
    hyperparams : dict
    history     : dict
    """
    algo = VBSKB_BID_STP(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
