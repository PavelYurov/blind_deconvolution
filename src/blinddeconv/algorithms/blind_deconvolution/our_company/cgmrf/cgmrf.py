"""
Variational Bayesian Blind Image Deconvolution with Total Variation Prior.

Checkpoint 2 — enhanced version of the CGMRF framework (checkpoint 1)
incorporating the methodology of:

    [5] Babacan, S.D., Molina, R., Katsaggelos, A.K.
        "Variational Bayesian blind deconvolution using a total
        variation prior."
        IEEE Trans. Image Processing, 18(1), 12-26, 2009.
        DOI: 10.1109/TIP.2008.2005443

    [6] Chantas, G., Galatsanos, N.P., Molina, R., Katsaggelos, A.K.
        "Variational Bayesian image restoration with a product of
        spatially weighted total variation image priors."
        IEEE Trans. Image Processing, 19(2), 351-362, 2010.

Earlier references (checkpoint 1):
    [1] Molina, R. et al., Pattern Recognition 33(4), 2000.
    [2] Molina, R. et al., IEEE TIP 12(12), 2003.
    [4] MacKay, D.J.C., Neural Computation 4(3), 1992.

Summary of improvements over checkpoint 1
==========================================

1. **Isotropic TV prior via Majorization-Minimization (MM).**
   Checkpoint 1 uses a CGMRF prior with separate horizontal and vertical
   line-process variables l_h, l_v, giving an anisotropic quadratic penalty
   weighted by per-pixel Gamma posteriors.  Checkpoint 2 replaces this with
   a smoothed isotropic Total Variation prior:

       TV_ε(f) = Σ_i √( (Δ_h f)_i² + (Δ_v f)_i² + ε )

   and applies the MM upper bound (half-quadratic splitting) to obtain a
   tight spatially-reweighted quadratic majoriser at each outer iteration:

       w_i = 1 / (2 √( (Δ_h f)_i² + (Δ_v f)_i² + ε ))

   This couples both gradient directions, preventing directional bias at
   diagonal edges, and guarantees monotonic decrease of the TV objective.
   [Ref. 5, Proposition 1;  Ref. 6, Eq. (8)]

2. **Variational Bayesian (VB) inference replacing Empirical Bayes.**
   Checkpoint 1 estimates hyperparameters via the Evidence Analysis
   framework (type-II ML), which treats the image and kernel as point
   estimates.  Checkpoint 2 maintains approximate posterior distributions

       q(f) = N(μ_f, Σ_f),   q(h) = N(μ_h, Σ_h)

   and updates hyperparameters by maximising the Evidence Lower Bound (ELBO)
   rather than the marginal likelihood.  The ELBO accounts for posterior
   uncertainty through covariance trace terms.
   [Ref. 5, Section II-D;  Ref. 6, Section II-B]

3. **ELBO-based hyperparameter updates with covariance traces.**
   The VB α, β updates place the trace term in the *denominator*:

       α = N / (f^T Q(w) f + tr(Σ_f Q)),
       β = N / (‖y − Hf‖²  + tr(Σ_f H^T H))

   rather than in the numerator as an effective-dimension subtraction
   (N_eff = N − α·tr(A⁻¹Q) in checkpoint 1).  This is always positive
   and numerically more stable.
   [Ref. 5, Eq. (22)–(23)]

4. **Automatic kernel regularisation (δ_h).**
   The fixed Tikhonov parameter δ_h of checkpoint 1 is replaced by a
   learned kernel precision estimated from the ELBO:

       δ_h = K / (‖h‖² + tr(Σ_h))

   When the kernel is sparse (most entries near zero), δ_h grows and
   enforces compactness of the PSF support automatically.
   [Ref. 5, Eq. (24)]

5. **ELBO convergence monitoring.**
   Checkpoint 1 monitors convergence via kernel-change ΔH and residual
   norm only.  Checkpoint 2 additionally tracks the ELBO, a principled
   lower bound on ln p(y) that must be non-decreasing across iterations.
   Any ELBO decrease signals a numerical issue rather than convergence.
   [Ref. 5, Section II-D]

6. **Stochastic trace estimation (Hutchinson probing).**
   Checkpoint 1 approximates tr(A⁻¹ B) spectrally by replacing spatially
   varying weights with their mean—biased when edges are strong.
   Checkpoint 2 can use Hutchinson's random-probe estimator, which is
   **unbiased** for any spatial weight distribution.
   [Ref. 7, Hutchinson 1990;  Ref. 5, Section III-C]

7. **Reduced hyperparameter count.**
   Checkpoint 1 estimates (α, β, γ_h, γ_v) — four hyperparameters.
   Checkpoint 2 estimates (α, β, δ_h) — three hyperparameters.
   The two Gamma shape parameters γ_h, γ_v are no longer needed because
   the line-process variables are replaced by deterministic MM weights.

Modules
-------
    utils   — FFT helpers, TV & MM utilities, trace estimation, ELBO.
    solvers — Pure solver functions (TV weights, VB image & kernel, VB hypers).
    cgmrf   — Framework wrapper class (this file).
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

# ── Robust import of the framework base class ───────────────────────────────
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk up the directory tree to locate the project root (pyproject.toml)."""
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


# =============================================================================
# Main Algorithm Class
# =============================================================================

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
        use_edgetaper: bool = True, # Flag for edgetaper
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

        # 1. Data Preparation
        y_orig = image.astype(np.float64)
        if y_orig.max() > 1.0:
            y_orig /= 255.0

        H, W = y_orig.shape
        kh, kw = self.kernel_shape

        # 2. Initialisation of Kernel (Gaussian)
        sig = max(kh, kw) / 8.0 
        grid_y, grid_x = np.ogrid[-kh // 2:kh // 2, -kw // 2:kw // 2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sig**2))
        h /= h.sum()

        # --- EDGETAPER ---
        # Apply edgetapering to the observation using the initial kernel guess.
        # This reduces ringing artifacts at boundaries without expanding image size.
        if self.use_edgetaper:
            y = edgetaper(y_orig, h)
        else:
            y = y_orig

        # Initial Image (start from tapered observation)
        f = y.copy()

        # Initial Hyperparameters
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

        # 3. Main Loop
        n_iter = 0
        current_thresh = self.kernel_threshold

        for it in range(self.max_iter):
            n_iter = it + 1
            h_prev = h.copy()

            # Step 1: Weights
            w = compute_tv_weights(f, self.epsilon)

            # Step 2: Image VB (PCG)
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

            # Step 3: Kernel VB (Gradient Domain + Thresholding)
            h, kern_info = solve_kernel_vb(
                y, f, self.kernel_shape, delta_h, beta,
                threshold_ratio=current_thresh
            )
            h_energy = kern_info['h_energy']
            h_cov_trace = kern_info['h_cov_trace']
            
            h = center_kernel_mass(h)

            # Step 4: Hyperparameters
            if self.update_hyperparams_flag:
                alpha, beta, delta_h = update_hyperparameters_vb(
                    y, f, h, w, alpha, beta, delta_h,
                    tr_Sigma_Q, tr_Sigma_HtH,
                    h_energy, h_cov_trace,
                )

            # Step 5: ELBO
            elbo = compute_elbo(
                y, f, h, alpha, beta, delta_h, w,
                tr_Sigma_Q, tr_Sigma_HtH, log_det_Sigma,
                h_cov_trace,
            )

            # Monitoring
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

        # 4. Final Refinement (Non-blind)
        # We perform the final restoration using the final estimated kernel 
        # but on the ORIGINAL (non-tapered) image if possible, OR keep using tapered
        # to avoid ringing. Standard practice: use tapered for blind steps, 
        # then restore original with edgetaper again or sophisticated boundary handling.
        # Here we re-run solver on the tapered image for consistency.
        
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