"""
Shift-Variant Blind Deconvolution Using a Field of Kernels.

Sonogashira et al. (2017), IEICE Trans. Inf. & Syst., E100-D(9), 1971-1983.
DOI: 10.1587/transinf.2016PCP0013

Key features:
    - L basis kernels {k_l} with per-pixel weight maps {w_l}  (Eq. 1-3).
    - Full Variational Bayesian inference via CAVI  (mean + variance
      for x, k_l, w_l; see Fig. 3).
    - FFT-accelerated convolutions and Laplacian.
    - Automatic hyperparameter estimation  (beta, alpha, gamma, eta;
      Eq. 15-18).
    - Uncertainty-aware updates prevent trivial delta-kernel solution
      (following Levin et al. [1]).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict
import sys
from pathlib import Path

# --------------- framework boilerplate ---------------
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

from .utils import (
    edgetaper, laplacian_fft_spectrum, compute_tv_weights, fft_conv2d,
    forward_diff_h, forward_diff_v, EPSILON,
)
from .solvers import (
    forward_blur, adjoint_blur,
    update_image, update_kernel_l, update_weight_l,
    update_beta, update_alpha, update_gamma, update_eta,
)

# =======================================================================

class SVBD_FK(DeconvolutionAlgorithm):
    """
    Shift-Variant Blind Deconvolution Using a Field of Kernels.

    Parameters
    ----------
    kernel_shape : (kh, kw)
        Spatial support of each basis kernel.
    n_basis : int
        Number of basis kernels L.
    max_iter : int
        Outer VB iterations.
    cg_iter_x : int
        CG iterations for image update.
    cg_iter_w : int
        CG iterations for weight update.
    verbose : bool
        Print per-iteration diagnostics.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (11, 11),
        n_basis: int = 3,
        max_iter: int = 30,
        cg_iter_x: int = 30,
        cg_iter_w: int = 20,
        verbose: bool = False,
    ):
        super().__init__(name='SVBD-FK')
        self.kernel_shape = tuple(kernel_shape)
        self.n_basis = n_basis
        self.max_iter = max_iter
        self.cg_iter_x = cg_iter_x
        self.cg_iter_w = cg_iter_w
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': [], 'beta': [], 'alpha': []}
        self.hyperparams: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run SVBD-FK on a single-channel blurred image.

        Implements the CAVI loop from Fig. 3 of Sonogashira et al. (2017):
          1. TV weights via Jaakkola bound  (Babacan [31])
          2. Image update  q(x) — Eq. (9)-(10)
          3. Per-basis kernel update  q(k_l) — Eq. (11)-(12)
             and weight update  q(w_l) — Eq. (13)-(14)
          4. Hyperparameter updates — Eq. (15)-(18)

        Parameters
        ----------
        image : np.ndarray, shape (H, W)
            Single-channel blurred observation in [0, 255] or [0, 1].

        Returns
        -------
        x_final : np.ndarray, shape (H, W), dtype int16
            Restored image in [0, 255].
        h_repr  : np.ndarray, shape (kh, kw)
            Representative (centre-pixel) effective PSF.
        """
        start_time = time.time()

        # ---- 1. Preprocessing ----------------------------------------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        y = edgetaper(y, max(self.kernel_shape))
        H, W = y.shape
        N = H * W
        kh, kw = self.kernel_shape
        K2 = kh * kw
        L = self.n_basis

        # ---- 2. Precompute Laplacian spectrum -------------------------
        F_L = laplacian_fft_spectrum((H, W))

        # ---- 3. Initialization  (Section 3.4 of the paper) -----------
        # Image: mean = observed, variance = small
        mu_x = y.copy()
        sigma_x = np.full((H, W), 1e-4, dtype=np.float64)

        # Kernels: first = small Gaussian, rest = delta (identity)
        mu_k: List[np.ndarray] = []
        sigma_k: List[np.ndarray] = []

        sig = max(kh, kw) / 10.0
        cy, cx = kh // 2, kw // 2
        gy, gx = np.mgrid[0:kh, 0:kw]
        gy = gy - cy; gx = gx - cx
        gauss_k = np.exp(-(gx ** 2 + gy ** 2) / (2 * sig ** 2 + EPSILON))
        gauss_k /= gauss_k.sum() + EPSILON

        mu_k.append(gauss_k.copy())
        sigma_k.append(np.full(K2, 1e-4))

        delta_k = np.zeros(self.kernel_shape)
        delta_k[kh // 2, kw // 2] = 1.0
        for _ in range(1, L):
            mu_k.append(delta_k.copy())
            sigma_k.append(np.full(K2, 1e-4))

        # Weight maps: uniform 1/L, variance small
        mu_w: List[np.ndarray] = [
            np.full((H, W), 1.0 / L, dtype=np.float64) for _ in range(L)
        ]
        sigma_w: List[np.ndarray] = [
            np.full((H, W), 1e-4, dtype=np.float64) for _ in range(L)
        ]

        # Hyperparameters (non-informative)
        a0 = 1e-6
        beta  = 1.0    # noise precision
        alpha = 1.0    # image smoothness
        gamma = [1.0] * L   # kernel sparsity
        eta   = [1.0] * L   # weight smoothness

        if self.verbose:
            print(f"[{self.name}] Image {H}x{W}, K={kh}x{kw}, L={L}")

        # ---- 4. Main VB loop  (Fig. 3) --------------------------------
        n_iter_done = 0
        for it in range(self.max_iter):
            k_prev = [k.copy() for k in mu_k]

            # Step 1: TV weights (Babacan bound)
            lam_h, lam_v = compute_tv_weights(mu_x)

            # Step 2: Update image q(x) — Eq. (9)-(10)
            mu_x, sigma_x = update_image(
                y, mu_k, mu_w, sigma_w, sigma_k,
                lam_h, lam_v, beta, alpha,
                mu_x, self.cg_iter_x, F_L,
            )

            # Step 3: Update kernels and weights per basis
            for l in range(L):
                # 3a: Kernel q(k_l) — Eq. (11)-(12)
                other_blur = np.zeros((H, W))
                for m in range(L):
                    if m != l:
                        other_blur += mu_w[m] * fft_conv2d(mu_x, mu_k[m])

                mu_k[l], sigma_k[l] = update_kernel_l(
                    y, mu_x, sigma_x,
                    mu_w[l], sigma_w[l],
                    mu_k[l], sigma_k[l],
                    other_blur,
                    beta, gamma[l],
                    self.kernel_shape,
                )

                # 3b: Weight q(w_l) — Eq. (13)-(14)
                other_w_contrib = np.zeros((H, W))
                for m in range(L):
                    if m != l:
                        other_w_contrib += mu_w[m] * fft_conv2d(mu_x, mu_k[m])

                mu_w[l], sigma_w[l] = update_weight_l(
                    y, mu_x, sigma_x,
                    mu_k[l], sigma_k[l],
                    mu_w[l],
                    other_w_contrib,
                    beta, eta[l], F_L,
                    self.cg_iter_w,
                )

            # Step 4: Hyperparameters — Eq. (15)-(18)
            beta  = update_beta(y, mu_x, sigma_x, mu_k, mu_w, sigma_w, a0, a0)
            alpha = update_alpha(mu_x, sigma_x, a0, a0)
            for l in range(L):
                gamma[l] = update_gamma(mu_k[l], sigma_k[l], a0, a0)
                eta[l]   = update_eta(mu_w[l], sigma_w[l], F_L, a0, a0)

            # Monitoring
            k_diff = sum(np.linalg.norm(mu_k[l] - k_prev[l]) for l in range(L))
            self.history['kernel_diff'].append(k_diff)
            self.history['beta'].append(beta)
            self.history['alpha'].append(alpha)
            n_iter_done = it + 1

            if self.verbose:
                print(
                    f"Iter {it + 1}/{self.max_iter}: "
                    f"dK={k_diff:.6f}, beta={beta:.1f}, alpha={alpha:.2f}"
                )

            if k_diff < 1e-5 and it > 5:
                if self.verbose:
                    print("Converged (kernel change < 1e-5).")
                break

        # ---- 5. Final non-blind refinement ----------------------------
        # Extra image-only pass with double CG iterations for accuracy.
        lam_h, lam_v = compute_tv_weights(mu_x)
        mu_x, _ = update_image(
            y, mu_k, mu_w, sigma_w, sigma_k,
            lam_h, lam_v, beta, alpha,
            mu_x, self.cg_iter_x * 2, F_L,
        )

        # ---- 6. Output formatting ------------------------------------
        x_final = np.clip(mu_x, 0.0, 1.0)
        x_final = (x_final * 255.0).round().astype(np.int16)

        # Representative kernel: h_center(j) = sum_l w_l(center) * k_l(j)
        cy, cx = H // 2, W // 2
        h_repr = sum(mu_w[l][cy, cx] * mu_k[l] for l in range(L))
        h_repr = np.maximum(h_repr, 0.0)
        s = h_repr.sum()
        if s > EPSILON:
            h_repr /= s

        self.hyperparams = {
            'time': time.time() - start_time,
            'iterations': n_iter_done,
            'beta': beta,
            'alpha': alpha,
            'gamma': gamma,
            'eta': eta,
            'noise_std_est': 1.0 / np.sqrt(beta + EPSILON),
        }

        return x_final, h_repr

    # ------------------------------------------------------------------ #
    #  Framework interface                                                 #
    # ------------------------------------------------------------------ #
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('n_basis', self.n_basis),
            ('max_iter', self.max_iter),
            ('cg_iter_x', self.cg_iter_x),
            ('cg_iter_w', self.cg_iter_w),
            ('verbose', self.verbose),
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
    """
    Convenience function matching the framework convention.

    Parameters
    ----------
    g            : (H, W) blurred observation
    kernel_shape : (kh, kw) kernel support size
    **kwargs     : forwarded to SVBD_FK constructor

    Returns
    -------
    f_est      : restored image (int16, 0..255)
    h_est      : representative PSF
    hyperparams : dict of learned hyperparameters
    history     : dict of convergence history
    """
    algo = SVBD_FK(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
