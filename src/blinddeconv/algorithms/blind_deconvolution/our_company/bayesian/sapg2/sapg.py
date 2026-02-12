import numpy as np
import time
from typing import Tuple, List, Any, Dict, Optional, Union
from .utils import (
    precompute_gradient_operators, project_param,
    gaussian_psf, gaussian_psf_deriv_alpha, tv_prox, tv_norm,
    estimate_lipschitz_alpha
)
from .solvers import (
    myula_sampler,
    posterior_grad_factory, prior_grad_factory, prox_factory,
    sapg_update_theta_homogeneous, sapg_update_theta_inhomogeneous,
    sapg_update_alpha, sapg_update_sigma2,
    data_fidelity_grad_fft, solve_image_hqs
)

try:
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class SAPG(DeconvolutionAlgorithm):
    """
    SAPG + MYULA for semi-blind deconvolution (Mbakam et al. 2024).

    Features:
    - Isotropic or anisotropic Gaussian PSF.
    - Automatic estimation of α (blur width), σ² (noise variance), θ (regularization).
    - TV prior (homogeneous, degree 1).
    - Adaptive MYULA step size based on Lipschitz constant.
    - Warm-up phase before SAPG iterations.
    - Weighted averaging of final parameters (δ_n as weights).
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (5, 5),
        alpha_init: Union[float, Tuple[float, float]] = 0.5,
        sigma2_init: float = 0.0025,
        theta_init: float = 0.1,
        max_iter: int = 50,
        anisotropic: bool = False,          # False -> isotropic, True -> anisotropic (alpha tuple)
        homogeneous_prior: bool = True,
        q: float = 1.0,
        # Warm-up
        warm_up: int = 1000,               # M0 = 3e4 in paper; here small for demo
        warm_up_burnin: int = 500,
        # MYULA parameters (auto if auto_params=True)
        myula_gamma: Optional[float] = None,
        myula_lam: Optional[float] = None,
        auto_params: bool = True,
        m_per_iter: int = 1,
        burn_in: int = 50,
        clip_samples: bool = False,        # Clip MCMC samples to [0,1]
        # SAPG parameters
        delta_base: float = 0.01,
        delta_decay: float = 0.6,
        weighted_average: bool = True,     # Use weights δ_n for final estimate
        # Parameter bounds
        theta_bounds: Tuple[float, float] = (1e-3, 1.0),
        alpha_bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = (0.1, 3.0),
        sigma2_bounds: Tuple[float, float] = (1e-5, 0.1),
        sigma2_min_est: Optional[float] = None,  # for Lipschitz estimation
        boundary: str = 'periodic',        # 'periodic' or 'reflect'
        verbose: bool = False,
        seed: int = 42,
    ):
        super().__init__(name='SAPG-BID')
        np.random.seed(seed)

        self.kernel_shape = tuple(kernel_shape)
        self.anisotropic = anisotropic
        if anisotropic:
            if not isinstance(alpha_init, (tuple, list)) or len(alpha_init) != 2:
                raise ValueError("anisotropic=True requires alpha_init as tuple (ah, av)")
            self.alpha = tuple(alpha_init)
            # For anisotropic, alpha_bounds must be tuple of two tuples
            if not isinstance(alpha_bounds, tuple) or len(alpha_bounds) != 2:
                raise ValueError("anisotropic: alpha_bounds must be ((ah_min,ah_max),(av_min,av_max))")
            self.alpha_bounds = alpha_bounds
        else:
            self.alpha = float(alpha_init)
            self.alpha_bounds = alpha_bounds

        self.sigma2 = sigma2_init
        self.theta = theta_init
        self.max_iter = max_iter
        self.homogeneous_prior = homogeneous_prior
        self.q = q

        # Warm-up
        self.warm_up = warm_up
        self.warm_up_burnin = warm_up_burnin

        # MYULA
        self.myula_gamma = myula_gamma
        self.myula_lam = myula_lam
        self.auto_params = auto_params
        self.m_per_iter = m_per_iter
        self.burn_in = burn_in
        self.clip_samples = clip_samples

        # SAPG
        self.delta_base = delta_base
        self.delta_decay = delta_decay
        self.weighted_average = weighted_average

        self.theta_bounds = theta_bounds
        self.sigma2_bounds = sigma2_bounds
        self.sigma2_min_est = sigma2_min_est
        self.boundary = boundary

        self.verbose = verbose

        # Internal state
        self.history = {'theta': [], 'alpha': [], 'sigma2': [], 'deltas': []}
        self.hyperparams = {}
        self._d = None
        self._H = None
        self._W = None
        self._L_alpha = None

    def _adapt_myula_params(self, y_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Compute λ and γ according to paper recommendations."""
        sigma2_min = self.sigma2_min_est if self.sigma2_min_est is not None else self.sigma2_bounds[0]
        L_alpha = estimate_lipschitz_alpha(self.alpha, self.kernel_shape, y_shape, sigma2_min)
        self._L_alpha = L_alpha
        lam_rec = min(5.0 / L_alpha, 2.0)
        gamma_rec = 0.98 / (L_alpha + 1.0 / lam_rec)
        if self.verbose:
            print(f"[{self.name}] L_α = {L_alpha:.3f}, λ = {lam_rec:.5f}, γ = {gamma_rec:.5f}")
        return lam_rec, gamma_rec

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # ------------------------------------------------------------
        # 1. Data preparation
        # ------------------------------------------------------------
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0
        self._H, self._W = y.shape
        self._d = self._H * self._W

        # Precompute gradient operators for TV
        self._F_ops = precompute_gradient_operators((self._H, self._W))

        # ------------------------------------------------------------
        # 2. Warm-up phase (MYULA at fixed initial parameters)
        # ------------------------------------------------------------
        if self.warm_up > 0:
            if self.verbose:
                print(f"\n[{self.name}] Warm-up: {self.warm_up} iterations with burn-in {self.warm_up_burnin}")
            if self.auto_params:
                lam, gamma = self._adapt_myula_params(y.shape)
            else:
                lam = self.myula_lam if self.myula_lam is not None else 1e-2
                gamma = self.myula_gamma if self.myula_gamma is not None else 1e-3

            # Posterior gradient and prox for warm-up (with current parameters)
            grad_post = posterior_grad_factory(y, self.alpha, self.sigma2, gaussian_psf,
                                               self.kernel_shape, self.boundary)
            prox = prox_factory(self.theta, tv_prox)

            # Run MYULA to get initial samples
            samples_warm = myula_sampler(
                x_init=y.copy(),
                gamma=gamma,
                lam=lam,
                m=self.warm_up,
                burn_in=self.warm_up_burnin,
                grad_log_potential=grad_post,
                prox=prox,
                args_grad=(),
                args_prox=(lam,),
                clip=self.clip_samples,
                clip_range=(0.0, 1.0)
            )
            # Warm start image for SAPG loop
            x = np.mean(samples_warm, axis=0)
        else:
            x = y.copy()

        # ------------------------------------------------------------
        # 3. Main SAPG loop
        # ------------------------------------------------------------
        if self.verbose:
            print(f"\n[{self.name}] Starting SAPG with {self.max_iter} iterations")

        for it in range(self.max_iter):
            # Step size
            delta = self.delta_base / (it + 1) ** self.delta_decay
            self.history['deltas'].append(delta)

            # Update MYULA parameters if auto
            if self.auto_params:
                lam, gamma = self._adapt_myula_params(y.shape)
            else:
                lam = self.myula_lam if self.myula_lam is not None else 1e-2
                gamma = self.myula_gamma if self.myula_gamma is not None else 1e-3

            # ---- A. Sample from posterior ----
            grad_post = posterior_grad_factory(y, self.alpha, self.sigma2, gaussian_psf,
                                               self.kernel_shape, self.boundary)
            prox = prox_factory(self.theta, tv_prox)
            samples_post = myula_sampler(
                x_init=x,
                gamma=gamma,
                lam=lam,
                m=self.m_per_iter,
                burn_in=self.burn_in,
                grad_log_potential=grad_post,
                prox=prox,
                args_grad=(),
                args_prox=(lam,),
                clip=self.clip_samples,
                clip_range=(0.0, 1.0)
            )

            # ---- B. Sample from prior if inhomogeneous ----
            samples_prior = None
            if not self.homogeneous_prior:
                grad_prior = prior_grad_factory()
                prox_prior = prox_factory(self.theta, tv_prox)   # same prox
                samples_prior = myula_sampler(
                    x_init=np.random.randn(self._H, self._W),
                    gamma=gamma,      # same gamma, but could be different; use same for simplicity
                    lam=lam,
                    m=self.m_per_iter,
                    burn_in=self.burn_in,
                    grad_log_potential=grad_prior,
                    prox=prox_prior,
                    args_grad=(),
                    args_prox=(lam,),
                    clip=self.clip_samples,
                    clip_range=(0.0, 1.0)
                )

            # ---- C. Update hyperparameters ----
            # θ
            if self.homogeneous_prior:
                self.theta = sapg_update_theta_homogeneous(
                    samples_post, self.theta, delta, self._d, self.q, self.theta_bounds
                )
            else:
                self.theta = sapg_update_theta_inhomogeneous(
                    samples_post, samples_prior, self.theta, delta, self.theta_bounds
                )

            # α
            if self.anisotropic:
                self.alpha = sapg_update_alpha(
                    samples_post, y, self.alpha, self.sigma2, delta,
                    gaussian_psf, gaussian_psf_deriv_alpha, self.kernel_shape,
                    self.alpha_bounds, self.boundary
                )
            else:
                self.alpha = sapg_update_alpha(
                    samples_post, y, self.alpha, self.sigma2, delta,
                    gaussian_psf, gaussian_psf_deriv_alpha, self.kernel_shape,
                    self.alpha_bounds, self.boundary
                )

            # σ²
            self.sigma2 = sapg_update_sigma2(
                samples_post, y, self.alpha, self.sigma2, delta, self._d,
                gaussian_psf, self.kernel_shape, self.sigma2_bounds, self.boundary
            )

            # ---- D. Record and warm start ----
            self.history['theta'].append(self.theta)
            self.history['alpha'].append(self.alpha)
            self.history['sigma2'].append(self.sigma2)
            x = np.mean(samples_post, axis=0)

            if self.verbose and (it + 1) % 10 == 0:
                if self.anisotropic:
                    print(f"[{self.name}] Iter {it+1:3d}: α=({self.alpha[0]:.3f},{self.alpha[1]:.3f}), "
                          f"σ²={self.sigma2:.5f}, θ={self.theta:.3f}, δ={delta:.6f}")
                else:
                    print(f"[{self.name}] Iter {it+1:3d}: α={self.alpha:.3f}, "
                          f"σ²={self.sigma2:.5f}, θ={self.theta:.3f}, δ={delta:.6f}")

        # ------------------------------------------------------------
        # 4. Final parameter estimation (weighted average after burn-in)
        # ------------------------------------------------------------
        N0 = int(0.8 * self.max_iter)   # 80% burn-in
        if self.max_iter - N0 > 0:
            if self.weighted_average:
                # Use δ_n as weights (3.16-3.18)
                weights = np.array(self.history['deltas'][N0:])
                sum_w = np.sum(weights)
                bar_theta = np.sum(weights * np.array(self.history['theta'][N0:])) / sum_w
                if self.anisotropic:
                    alpha_array = np.array(self.history['alpha'][N0:])
                    bar_alpha = (np.sum(weights * alpha_array[:, 0]) / sum_w,
                                 np.sum(weights * alpha_array[:, 1]) / sum_w)
                else:
                    bar_alpha = np.sum(weights * np.array(self.history['alpha'][N0:])) / sum_w
                bar_sigma2 = np.sum(weights * np.array(self.history['sigma2'][N0:])) / sum_w
            else:
                bar_theta = np.mean(self.history['theta'][N0:])
                if self.anisotropic:
                    alpha_array = np.array(self.history['alpha'][N0:])
                    bar_alpha = (np.mean(alpha_array[:, 0]), np.mean(alpha_array[:, 1]))
                else:
                    bar_alpha = np.mean(self.history['alpha'][N0:])
                bar_sigma2 = np.mean(self.history['sigma2'][N0:])
        else:
            bar_theta, bar_alpha, bar_sigma2 = self.theta, self.alpha, self.sigma2

        self.hyperparams = {
            'theta': bar_theta,
            'alpha': bar_alpha,
            'sigma2': bar_sigma2,
            'theta_history': self.history['theta'],
            'alpha_history': self.history['alpha'],
            'sigma2_history': self.history['sigma2'],
            'iterations': self.max_iter,
            'time': time.time() - start_time,
            'L_alpha': self._L_alpha
        }

        # ------------------------------------------------------------
        # 5. Non-blind MAP reconstruction
        # ------------------------------------------------------------
        if self.verbose:
            print(f"\n[{self.name}] Final MAP reconstruction...")
        h_final = gaussian_psf(bar_alpha, self.kernel_shape)
        x_final = solve_image_hqs(
            y=y,
            h=h_final,
            x_init=x,
            sigma=np.sqrt(bar_sigma2),
            theta=bar_theta,
            max_iter=50,
            beta0=1.0,
            beta_factor=1.1,
            boundary=self.boundary
        )

        if self.verbose:
            print(f"[{self.name}] Completed in {time.time()-start_time:.2f}s")

        return x_final, h_final

    # ---------- Interface methods ----------
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('alpha_init', self.alpha),
            ('sigma2_init', self.sigma2),
            ('theta_init', self.theta),
            ('max_iter', self.max_iter),
            ('anisotropic', self.anisotropic),
            ('homogeneous_prior', self.homogeneous_prior),
            ('q', self.q),
            ('warm_up', self.warm_up),
            ('myula_gamma', self.myula_gamma),
            ('myula_lam', self.myula_lam),
            ('auto_params', self.auto_params),
            ('m_per_iter', self.m_per_iter),
            ('burn_in', self.burn_in),
            ('delta_base', self.delta_base),
            ('delta_decay', self.delta_decay),
            ('weighted_average', self.weighted_average),
            ('theta_bounds', self.theta_bounds),
            ('alpha_bounds', self.alpha_bounds),
            ('sigma2_bounds', self.sigma2_bounds),
            ('boundary', self.boundary),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if self.verbose:
                    print(f"[{self.name}] Changing {key}: {getattr(self, key)} -> {value}")
                setattr(self, key, value)
            else:
                print(f"[{self.name}] Warning: unknown parameter '{key}'")

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams

# ---------------------------------------------------------------------
# Standalone runner function
# ---------------------------------------------------------------------
def run_algorithm(g, kernel_shape, **kwargs):
    algo = SAPG(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.get_hyperparams(), algo.get_history()