"""
rcs2.py

Blind Image Deconvolution via Variational Bayesian Inference
(Fergus et al., SIGGRAPH 2006).

Reference:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Transactions on Graphics (SIGGRAPH), 2006.

Pipeline (mirrors MATLAB deblur.m):
    1.  Normalise input; optional grayscale conversion.
    2.  Gamma correction → gradient computation (Haar/Laplace).
    3.  Multi-scale pyramid construction for gradients and kernels.
    4.  Per-scale variational Bayesian kernel estimation
        (train_ensemble_main6).
    5.  Poisson reconstruction of the latent image from estimated
        gradients (reconsEdge3).
    6.  Non-blind Richardson-Lucy deconvolution on the full image
        (fiddle_lucy3 / fiddle_lucy4).
    7.  Return restored image (int16, [0, 255]) and kernel.
"""

import numpy as np
import time
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from typing import Tuple, List, Any, Dict, Optional

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

from . import utils
from . import solvers


# ─────────────────────────────────────────────────────────────────────────────
# Simple namespace to pass priors with .pi / .gamma attributes
# ─────────────────────────────────────────────────────────────────────────────
class _PriorNamespace:
    """Thin wrapper so that prior dicts can be accessed as prior.pi / prior.gamma."""
    def __init__(self, d: dict):
        self.pi = d['pi']
        self.gamma = d['gamma']


class RCS2(DeconvolutionAlgorithm):
    """
    Blind deconvolution via ensemble learning / variational Bayesian
    inference (Fergus et al. SIGGRAPH 2006).

    Parameters
    ----------
    kernel_size : int
        Spatial support of the unknown PSF (square, odd).
    num_scales : int
        Number of pyramid levels.
    resize_step : float
        Scale factor between adjacent levels (default √2).
    resize_mode : str
        Interpolation mode for pyramid ('matlab_bilinear', etc.).
    gradient_mode : str
        Gradient filter: 'haar' or 'laplace'.
    gamma_correction : float
        Gamma exponent applied before estimation.
    prescale : float
        Pre-scaling factor for the input image (1.0 = no scaling).
    convergence : float
        VB convergence criterion.
    max_iterations : int
        Maximum VB iterations per scale.
    noise_init : float
        Initial noise standard deviation.
    init_mode_blur : str
        Blur initialisation at scale 1 ('delta', 'direct', etc.).
    init_mode_image : str
        Image initialisation at scale 1 ('slight_blur_obs', etc.).
    init_prescision : float
        Initial precision for the ensemble.
    blur_prior : int
        Prior type for the blur (1 = exponential).
    image_prior : int
        Prior type for image gradients (0 = Gaussian).
    blur_components : int
        Number of MoG components for the blur prior.
    image_components : int
        Number of MoG components for the image prior.
    blur_lock : int
        Whether to lock the blur prior (0/1).
    fft_mode : int
        Use FFT convolution (1) or direct convolution (0).
    rescale_then_grad : bool
        Rescale-then-gradient (True) vs gradient-then-rescale (False).
    upsample_mode : str
        Mode for upsampling between scales.
    center_blur : bool
        Centre the kernel by its centre of mass between scales.
    image_reconstruction : str
        'lucy' (fiddle_lucy3) or 'lucy_intens' (fiddle_lucy4).
    lucy_iterations : int
        Number of Richardson-Lucy iterations.
    kernel_threshold : float
        Dynamic threshold for the final kernel (% of max).
    scale_offset : int
        Number of scales to step back for final kernel selection.
    prior_images : list or None
        Training images for MoG prior estimation (via estimate_priors).
    """

    def __init__(
        self,
        kernel_size: int = 25,
        num_scales: int = 5,
        resize_step: float = np.sqrt(2),
        resize_mode: str = 'matlab_bilinear',
        gradient_mode: str = 'haar',
        gamma_correction: float = 2.2,
        prescale: float = 1.0,
        convergence: float = 1e-4,
        max_iterations: int = 200,
        noise_init: float = 1.0,
        init_mode_blur: str = 'delta',
        init_mode_image: str = 'slight_blur_obs',
        init_prescision: float = 1.0,
        blur_prior: int = 1,
        image_prior: int = 0,
        blur_components: int = 4,
        image_components: int = 4,
        blur_lock: int = 0,
        fft_mode: int = 1,
        rescale_then_grad: bool = True,
        upsample_mode: str = 'matlab_bilinear',
        center_blur: bool = False,
        image_reconstruction: str = 'lucy',
        lucy_iterations: int = 10,
        kernel_threshold: float = 7.0,
        scale_offset: int = 0,
        prior_images: Optional[list] = None,
    ):
        super().__init__(name='RCS2-Fergus2006')

        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.resize_step = resize_step
        self.resize_mode = resize_mode
        self.gradient_mode = gradient_mode
        self.gamma_correction = gamma_correction
        self.prescale = prescale
        self.convergence = convergence
        self.max_iterations = max_iterations
        self.noise_init = noise_init
        self.init_mode_blur = init_mode_blur
        self.init_mode_image = init_mode_image
        self.init_prescision = init_prescision
        self.blur_prior = blur_prior
        self.image_prior = image_prior
        self.blur_components = blur_components
        self.image_components = image_components
        self.blur_lock = blur_lock
        self.fft_mode = fft_mode
        self.rescale_then_grad = rescale_then_grad
        self.upsample_mode = upsample_mode
        self.center_blur = center_blur
        self.image_reconstruction = image_reconstruction
        self.lucy_iterations = lucy_iterations
        self.kernel_threshold = kernel_threshold
        self.scale_offset = scale_offset
        self.prior_images = prior_images

        self.history: Dict[str, list] = {'D_log': [], 'gamma_log': []}
        self.hyperparams: Dict[str, Any] = {}

    # ══════════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════════
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        obs_im_orig = np.array(image, dtype=np.float64)

        # ── 1. Grayscale conversion ──────────────────────────────────────
        COLOR = False
        obs_imz = 1

        if obs_im_orig.ndim == 3 and obs_im_orig.shape[2] >= 3:
            obs_im = utils.rgb2gray_rob(obs_im_orig)
        elif obs_im_orig.ndim == 3 and obs_im_orig.shape[2] == 1:
            obs_im = obs_im_orig[:, :, 0].copy()
        else:
            obs_im = obs_im_orig.copy()

        # ── 2. Prescale ──────────────────────────────────────────────────
        if self.prescale != 1.0 and self.prescale != 0.0:
            obs_im = utils.imresize(obs_im, scale=self.prescale)
            obs_im_orig = utils.imresize(obs_im_orig, scale=self.prescale)

        # ── 3. Gamma correction ──────────────────────────────────────────
        if self.gamma_correction != 1.0:
            obs_im = (obs_im.astype(np.float64) ** self.gamma_correction) / (
                256.0 ** (self.gamma_correction - 1.0))
        else:
            obs_im = obs_im.astype(np.float64)

        # ── 4. Gradient computation ──────────────────────────────────────
        if self.gradient_mode == 'haar':
            kx = np.array([[1.0, -1.0]])
            ky = np.array([[1.0], [-1.0]])
        elif self.gradient_mode == 'laplace':
            kx = np.array([[1.0, -2.0, 1.0]])
            ky = np.array([[1.0], [-2.0], [1.0]])
        else:
            raise ValueError(f"Unknown gradient mode: {self.gradient_mode}")

        obs_grad_all_x = convolve2d(obs_im, kx, mode='valid')
        obs_grad_all_y = convolve2d(obs_im, ky, mode='valid')

        yy = min(obs_grad_all_x.shape[0], obs_grad_all_y.shape[0])
        xx = min(obs_grad_all_x.shape[1], obs_grad_all_y.shape[1])
        obs_grad_all = np.concatenate([
            obs_grad_all_x[:yy, :xx], obs_grad_all_y[:yy, :xx]
        ], axis=1)

        # Full-image patch (no cropping — use entire image)
        obs_grad_x = obs_grad_all_x
        obs_grad_y = obs_grad_all_y

        # ── 5. Multi-scale pyramid — gradients ──────────────────────────
        obs_grad_scale_x = [None] * self.num_scales
        obs_grad_scale_y = [None] * self.num_scales
        obs_grad_scale = [None] * self.num_scales
        mask_scale = [None] * self.num_scales

        if self.rescale_then_grad:
            obs_im_scale = [None] * self.num_scales
            obs_im_scale[self.num_scales - 1] = obs_im.copy()
            mask_scale[self.num_scales - 1] = np.zeros(
                (obs_im.shape[0] - 2, obs_im.shape[1] - 2))

            for s_idx in range(1, self.num_scales):
                s = self.num_scales - s_idx  # scale index (0 = coarsest)
                factor = (1.0 / self.resize_step) ** s_idx
                obs_im_scale[s - 1] = utils.imresize(obs_im, scale=factor)
                mask_scale[s - 1] = np.zeros((
                    max(1, obs_im_scale[s - 1].shape[0] - 2),
                    max(1, obs_im_scale[s - 1].shape[1] - 2)))

            # Gradient filter for rescale-then-grad
            if self.gradient_mode == 'haar':
                kx_g = np.array([[0.0, 1.0, -1.0]])
                ky_g = np.array([[0.0], [1.0], [-1.0]])
            else:
                kx_g = kx
                ky_g = ky

            for s in range(self.num_scales):
                src = obs_im_scale[s]
                obs_grad_scale_x[s] = convolve2d(src, kx_g, mode='valid')
                obs_grad_scale_y[s] = convolve2d(src, ky_g, mode='valid')

            for s in range(self.num_scales):
                gx = obs_grad_scale_x[s]
                gy = obs_grad_scale_y[s]
                obs_grad_scale[s] = np.concatenate([
                    gx[1:-1, :] if self.rescale_then_grad else gx,
                    gy[:, 1:-1] if self.rescale_then_grad else gy
                ], axis=1)

        else:
            # Gradient then rescale
            obs_grad_scale_x[self.num_scales - 1] = obs_grad_x
            obs_grad_scale_y[self.num_scales - 1] = obs_grad_y
            mask_scale[self.num_scales - 1] = np.zeros_like(obs_grad_x)

            for s_idx in range(1, self.num_scales):
                factor = (1.0 / self.resize_step) ** s_idx
                idx = self.num_scales - s_idx - 1
                obs_grad_scale_x[idx] = utils.imresize(obs_grad_x, scale=factor)
                obs_grad_scale_y[idx] = utils.imresize(obs_grad_y, scale=factor)
                mask_scale[idx] = np.zeros_like(obs_grad_scale_x[idx])

            for s in range(self.num_scales):
                obs_grad_scale[s] = np.concatenate([
                    obs_grad_scale_x[s], obs_grad_scale_y[s]
                ], axis=1)

        # ── 6. Multi-scale pyramid — blur kernels ───────────────────────
        # Create initial kernel (delta or provided)
        ks = self.kernel_size
        if ks % 2 == 0:
            ks += 1
        blur_kernel = utils.delta_kernel(ks)

        blur_kernel_scale = [None] * self.num_scales
        blur_kernel_scale[self.num_scales - 1] = blur_kernel.copy()

        for s_idx in range(1, self.num_scales):
            dims = np.array(blur_kernel.shape[:2], dtype=np.float64) * (
                1.0 / self.resize_step) ** s_idx
            dims = dims.astype(int)
            dims = dims + (1 - dims % 2)  # make odd
            dims = np.maximum(dims, 3)

            idx = self.num_scales - s_idx - 1
            if min(dims) < 4:
                h = utils._fspecial_gaussian(int(dims[0]), 1.0)
                blurred = convolve2d(blur_kernel, h, mode='same')
                blur_kernel_scale[idx] = utils.imresize(blurred, tuple(dims))
            else:
                blur_kernel_scale[idx] = utils.imresize(
                    blur_kernel, tuple(dims))

            bk = blur_kernel_scale[idx]
            bk_sum = np.sum(bk)
            if bk_sum > 0:
                blur_kernel_scale[idx] = bk / bk_sum

        # ── 7. Estimate MoG priors ───────────────────────────────────────
        if self.prior_images is not None:
            priors_list = utils.estimate_priors(
                self.prior_images,
                num_components=self.image_components,
                num_scales=self.num_scales,
                gradient_mode=self.gradient_mode,
                gamma_correction=self.gamma_correction,
            )
        else:
            # Use flat uninformative priors
            priors_list = []
            for _ in range(self.num_scales):
                priors_list.append({
                    'pi': np.ones((obs_imz, self.image_components)) / self.image_components,
                    'gamma': np.ones((obs_imz, self.image_components)),
                })

        # ── 8. FFT pre-processing (shift observations) ──────────────────
        if self.fft_mode:
            for s in range(self.num_scales):
                db = utils.delta_kernel(blur_kernel_scale[s].shape[0])
                shifted = np.real(ifft2(
                    fft2(obs_grad_scale[s])
                    * fft2(db, s=obs_grad_scale[s].shape)
                ))
                obs_grad_scale[s] = shifted

        # ── 9. Size arrays and masks ─────────────────────────────────────
        K_arr = np.zeros(self.num_scales, dtype=int)
        L_arr = np.zeros(self.num_scales, dtype=int)
        M_arr = np.zeros(self.num_scales, dtype=int)
        N_arr = np.zeros(self.num_scales, dtype=int)
        I_arr = np.zeros(self.num_scales, dtype=int)
        J_arr = np.zeros(self.num_scales, dtype=int)
        D_list = [None] * self.num_scales
        Dp_list = [None] * self.num_scales
        spatial_blur_mask = [None] * self.num_scales
        spatial_image_mask = [None] * self.num_scales

        for s in range(self.num_scales):
            K_arr[s] = blur_kernel_scale[s].shape[0]
            L_arr[s] = blur_kernel_scale[s].shape[1]
            M_arr[s] = obs_grad_scale[s].shape[0]
            N_arr[s] = obs_grad_scale[s].shape[1]

            hK = K_arr[s] // 2
            hL = L_arr[s] // 2

            # Blur spatial mask (log-normal weighting)
            sbm = np.zeros((self.blur_components, int(K_arr[s] * L_arr[s])))
            spatial_blur_mask[s] = sbm

            # Image spatial mask (no saturation masking by default)
            if mask_scale[s] is not None:
                sim = np.concatenate([
                    np.zeros_like(mask_scale[s]),
                    np.zeros_like(mask_scale[s])
                ], axis=1)
            else:
                sim = np.zeros((M_arr[s], N_arr[s]))
            spatial_image_mask[s] = sim

            if self.fft_mode:
                I_arr[s] = 2 * M_arr[s]
                J_arr[s] = 2 * N_arr[s]
                Dp = np.zeros((I_arr[s], J_arr[s]))
                # MATLAB 1-based: Dp(K:M, L:N/2) → 0-based: K-1:M, L-1:N//2
                Dp[K_arr[s] - 1:M_arr[s], L_arr[s] - 1:N_arr[s] // 2] = 1
                Dp[K_arr[s] - 1:M_arr[s], L_arr[s] - 1 + N_arr[s] // 2:N_arr[s]] = 1
                Dp_list[s] = Dp
                D_list[s] = np.pad(
                    obs_grad_scale[s],
                    ((0, M_arr[s]), (0, N_arr[s])),
                    mode='constant')
            else:
                I_arr[s] = M_arr[s]
                J_arr[s] = N_arr[s]
                Dp = np.zeros((I_arr[s], J_arr[s]))
                Dp[hK:M_arr[s] - hK, hL:N_arr[s] // 2 - hL] = 1
                Dp[hK:M_arr[s] - hK,
                   N_arr[s] // 2 + hL:N_arr[s] - hL] = 1
                Dp_list[s] = Dp
                D_list[s] = obs_grad_scale[s].copy()

        # ── 10. MAIN LOOP — multi-scale VB inference ─────────────────────
        me_est = [None] * self.num_scales
        mx_est = [None] * self.num_scales
        new_grad = [None] * self.num_scales
        new_blur = [None] * self.num_scales

        for s in range(self.num_scales):
            # Select prior for this scale (coarsest first)
            prior_idx = min(self.num_scales - s - 1, len(priors_list) - 1)
            prior_s = _PriorNamespace(priors_list[prior_idx])

            K_s = int(K_arr[s])
            L_s = int(L_arr[s])
            M_s = int(M_arr[s])
            N_s = int(N_arr[s])
            I_s = int(I_arr[s])
            J_s = int(J_arr[s])

            dimensions = np.array([
                [1, 1, 1, 0, 0, 1],
                [1, K_s * L_s, self.blur_components, self.blur_prior,
                 self.blur_lock, 1],
                [obs_imz, M_s * N_s, self.image_components,
                 self.image_prior, 1, 1],
            ], dtype=np.float64)

            # ── Initialise ──
            if s == 0:
                x1, x2 = solvers.initialize_parameters2(
                    obs=obs_grad_scale[s],
                    blur=blur_kernel_scale[s],
                    im=obs_grad_scale[s],
                    true_blur=blur_kernel_scale[s],
                    true_im=obs_grad_scale[s],
                    pres=self.init_prescision,
                    prior_type=self.image_prior,
                    prior_num=self.image_components,
                    mode_im=self.init_mode_image,
                    mode_blur=self.init_mode_blur,
                    obs_im=obs_im,
                    big_blur=blur_kernel_scale[self.num_scales - 1],
                    spatial_mask=spatial_image_mask[s],
                    priors=prior_s,
                    fft_mode=self.fft_mode,
                    color=COLOR,
                    n_layers=1,
                )
            else:
                x1, x2 = solvers.initialize_parameters2(
                    obs=obs_grad_scale[s],
                    blur=new_blur[s - 1],
                    im=new_grad[s - 1],
                    true_blur=None,
                    true_im=None,
                    pres=self.init_prescision,
                    prior_type=self.image_prior,
                    prior_num=self.image_components,
                    mode_im='direct',
                    mode_blur='direct',
                    obs_im=obs_im,
                    big_blur=blur_kernel_scale[self.num_scales - 1],
                    spatial_mask=spatial_image_mask[s],
                    priors=prior_s,
                    fft_mode=self.fft_mode,
                    color=COLOR,
                    n_layers=1,
                )

            # ── Run VB inference ──
            ensemble, D_log_s, gamma_log_s = solvers.train_ensemble_main6(
                dimensions, x1, x2, '', f'Scale={s + 1}',
                [self.convergence, 0, self.noise_init, 0, 0,
                 self.max_iterations, 0],
                D_list[s], Dp_list[s],
                I_s, J_s, K_s, L_s, M_s, N_s,
                prior_s, self.fft_mode,
                spatial_blur_mask[s],
                1.0 - (spatial_image_mask[s].ravel() > 0).astype(np.float64),
            )

            self.history['D_log'].append(D_log_s)
            self.history['gamma_log'].append(gamma_log_s)

            # Extract estimates
            me_est[s] = solvers.train_ensemble_get(
                1, dimensions, ensemble['mx']).reshape(K_s, L_s)
            mx_est[s] = solvers.train_ensemble_get(
                2, dimensions, ensemble['mx']).reshape(M_s, N_s)

            # Upsample for next scale
            if s < self.num_scales - 1:
                K_next = int(K_arr[s + 1])
                L_next = int(L_arr[s + 1])
                M_next = int(M_arr[s + 1])
                N_next = int(N_arr[s + 1])
                new_grad[s], new_blur[s] = solvers.move_level(
                    mx_est[s], me_est[s],
                    K_next, L_next, M_next, N_next,
                    self.upsample_mode, self.resize_step,
                    self.center_blur,
                )

        # ── 11. Poisson reconstruction ───────────────────────────────────
        top_s = self.num_scales - 1
        M_top = int(M_arr[top_s])
        N_top = int(N_arr[top_s])

        final_dx = mx_est[top_s][:, :N_top // 2]
        final_dy = mx_est[top_s][:, N_top // 2:N_top]

        # Zero boundaries
        final_dx[0, :] = 0; final_dx[-1, :] = 0
        final_dx[:, 0] = 0; final_dx[:, -1] = 0
        final_dy[0, :] = 0; final_dy[-1, :] = 0
        final_dy[:, 0] = 0; final_dy[:, -1] = 0

        obs_im_recon, _ = utils.reconsEdge3(final_dx, final_dy)

        # ── 12. Non-blind Richardson-Lucy on full image ──────────────────
        if self.image_reconstruction == 'lucy':
            deblurred, kernel_out = solvers.fiddle_lucy3(
                me_est=me_est,
                obs_im=obs_im_orig,
                gamma_correction=self.gamma_correction,
                prescale=self.prescale,
                lucy_its=self.lucy_iterations,
                scale_offset=self.scale_offset,
                threshold=self.kernel_threshold,
            )
        elif self.image_reconstruction == 'lucy_intens':
            deblurred, kernel_out = solvers.fiddle_lucy4(
                me_est=me_est,
                obs_im=obs_im_orig,
                gamma_correction=self.gamma_correction,
                prescale=self.prescale,
                lucy_its=self.lucy_iterations,
                scale_offset=self.scale_offset,
                threshold=self.kernel_threshold,
                edge_crop=False,
                brighten=1.0,
            )
        else:
            raise ValueError(
                f"Unknown reconstruction: {self.image_reconstruction}")

        # ── 13. Output ───────────────────────────────────────────────────
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'num_scales': self.num_scales,
            'gamma_correction': self.gamma_correction,
            'blur_prior': self.blur_prior,
            'image_prior': self.image_prior,
            'blur_components': self.blur_components,
            'image_components': self.image_components,
            'lucy_iterations': self.lucy_iterations,
            'kernel_threshold': self.kernel_threshold,
            'time': time.time() - start_time,
        }

        # histmatch returns uint8 [0,255]; just cast to int16
        x_final = np.clip(np.array(deblurred, dtype=np.float64), 0, 255).astype(np.int16)
        return x_final, kernel_out

    # ══════════════════════════════════════════════════════════════════════
    # Interface methods
    # ══════════════════════════════════════════════════════════════════════
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('num_scales', self.num_scales),
            ('resize_step', self.resize_step),
            ('resize_mode', self.resize_mode),
            ('gradient_mode', self.gradient_mode),
            ('gamma_correction', self.gamma_correction),
            ('prescale', self.prescale),
            ('convergence', self.convergence),
            ('max_iterations', self.max_iterations),
            ('noise_init', self.noise_init),
            ('init_mode_blur', self.init_mode_blur),
            ('init_mode_image', self.init_mode_image),
            ('blur_prior', self.blur_prior),
            ('image_prior', self.image_prior),
            ('blur_components', self.blur_components),
            ('image_components', self.image_components),
            ('image_reconstruction', self.image_reconstruction),
            ('lucy_iterations', self.lucy_iterations),
            ('kernel_threshold', self.kernel_threshold),
            ('scale_offset', self.scale_offset),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
