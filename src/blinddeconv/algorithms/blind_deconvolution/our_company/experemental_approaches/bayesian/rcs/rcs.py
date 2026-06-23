"""
rcs.py

Источник:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Trans. Graphics (SIGGRAPH), 2006.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2

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

from .solvers import (
    train_ensemble_main6,
    initialize_parameters2,
    richardson_lucy,
)
from .utils import (
    rgb2gray_rob,
    delta_kernel,
    normMDpdf,
    automatic_patch_selector,
    imresize,
    move_level,
    reconsEdge3,
    train_ensemble_get,
    estimate_priors_from_images,
    get_default_priors,
    load_matlab_priors,
)

class RCS_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 25,
        num_scales: int = None,
        resize_step: float = np.sqrt(2),
        resize_mode: str = 'matlab_bilinear',
        gamma_correction: float = 2.2,
        prescale: float = 1.0,
        convergence: float = 5e-4,
        max_iterations: int = 50000,
        noise_init: float = 1.0,
        blur_components: int = 4,
        image_components: int = 4,
        blur_prior: int = 1,
        image_prior: int = 0,
        init_prescision: float = 1e4,
        first_init_mode_image: str = 'variational',
        first_init_mode_blur: str = 'delta',
        init_mode_image: str = 'direct',
        init_mode_blur: str = 'direct',
        upsample_mode: str = 'matlab_bilinear',
        center_blur: bool = True,
        gradient_mode: str = 'haar',
        rescale_then_grad: bool = False,
        lucy_its: int = 10,
        kernel_threshold: float = 10.0,
        scale_offset: int = 0,
        fft_mode: bool = True,
        blur_lock: bool = True,
        blur_mask: bool = False,
        blur_mask_variances: list = None,
        saturation_mask: int = 0,
        saturation_threshold: float = 250.0,
        automatic_patch: bool = False,
        automatic_patch_center_weight: float = 1.0,
        patch_size: tuple = None,
        patch_location: tuple = None,
        priors: list = None,
    ):
        super().__init__(name='RCS-BD')

        self.kernel_size = kernel_size

        if num_scales is None:
            self.num_scales = int(np.ceil(-np.log(3.0 / kernel_size)
                                          / np.log(np.sqrt(2)))) + 1
        else:
            self.num_scales = num_scales
        self.resize_step = resize_step
        self.resize_mode = resize_mode
        self.gamma_correction = gamma_correction
        self.prescale = prescale
        self.convergence = convergence
        self.max_iterations = max_iterations
        self.noise_init = noise_init
        self.blur_components = blur_components
        self.image_components = image_components
        self.blur_prior = blur_prior
        self.image_prior = image_prior
        self.init_prescision = init_prescision
        self.first_init_mode_image = first_init_mode_image
        self.first_init_mode_blur = first_init_mode_blur
        self.init_mode_image = init_mode_image
        self.init_mode_blur = init_mode_blur
        self.upsample_mode = upsample_mode
        self.center_blur = center_blur
        self.gradient_mode = gradient_mode
        self.rescale_then_grad = rescale_then_grad
        self.lucy_its = lucy_its
        self.kernel_threshold = kernel_threshold
        self.scale_offset = scale_offset
        self.fft_mode = fft_mode
        self.blur_lock = blur_lock
        self.blur_mask = blur_mask
        self.blur_mask_variances = blur_mask_variances or [1.0, 0.5, 0.1, 0.05]
        self.saturation_mask = saturation_mask
        self.saturation_threshold = saturation_threshold
        self.automatic_patch = automatic_patch
        self.automatic_patch_center_weight = automatic_patch_center_weight
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.priors = priors

        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'D_log': [],
            'gamma_log': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        COLOR = False
        DEFAULT_GAMMA = 2.2

        obs_im_orig = np.asarray(image, dtype=np.float64).copy()

        if obs_im_orig.ndim == 3 and obs_im_orig.shape[2] >= 3:
            obs_im = rgb2gray_rob(obs_im_orig)
        else:
            obs_im = obs_im_orig.copy() if obs_im_orig.ndim == 2\
                else obs_im_orig[:, :, 0].copy()

        obs_imz = 1

        if self.prescale and self.prescale != 1.0:
            obs_im = imresize(obs_im, self.prescale, 'bilinear')
            obs_im_orig = imresize(obs_im_orig, self.prescale, 'bilinear')

        obs_im_raw = obs_im.copy()

        if self.gamma_correction != 1.0:
            obs_im = (obs_im.astype(np.float64) ** self.gamma_correction) /\
                      (256.0 ** (self.gamma_correction - 1.0))
        else:
            obs_im = obs_im.astype(np.float64)

        if self.saturation_mask:
            sat = obs_im > self.saturation_threshold

            bk_init = delta_kernel(self.kernel_size)
            q = convolve2d(sat.astype(np.float64),
                           np.ones_like(bk_init), mode='same',
                           boundary='fill')
            mask_all = (q > 0).astype(np.float64)
        else:
            mask_all = np.zeros(obs_im.shape[:2], dtype=np.float64)

        if self.automatic_patch:
            obs_im_disp = obs_im ** (1.0 / DEFAULT_GAMMA)
            _, self.patch_location = automatic_patch_selector(
                obs_im_disp, max(self.patch_size or (obs_im.shape[0],)),
                self.automatic_patch_center_weight, mask_all)

        obs_im_all = obs_im.copy()

        if self.gradient_mode == 'haar':
            kx = np.array([[1, -1]], dtype=np.float64)
            ky = np.array([[1], [-1]], dtype=np.float64)
        elif self.gradient_mode == 'laplace':
            kx = np.array([[1, -2, 1]], dtype=np.float64)
            ky = np.array([[1], [-2], [1]], dtype=np.float64)
        else:
            raise ValueError(f"Unknown gradient mode: {self.gradient_mode}")

        if self.patch_location is not None and self.patch_size is not None:
            py, px = 0, 0
            pl = self.patch_location
            ps = self.patch_size
            if isinstance(ps, int):
                ps = (ps, ps)
            obs_im = obs_im_all[
                pl[1] - py:pl[1] + ps[1] - 1 + py + 1,
                pl[0] - px:pl[0] + ps[0] - 1 + px + 1,
            ]
            mask_all_patch = mask_all[
                pl[1] - py:pl[1] + ps[1] - 1 + py + 1,
                pl[0] - px:pl[0] + ps[0] - 1 + px + 1,
            ]
        else:
            mask_all_patch = mask_all

        NUM_SCALES = self.num_scales
        RESIZE_STEP = self.resize_step

        blur_kernel = delta_kernel(self.kernel_size)

        blur_kernel_scale = [None] * NUM_SCALES
        blur_kernel_scale[NUM_SCALES - 1] = blur_kernel.copy()

        for s_rev in range(1, NUM_SCALES):
            s_idx = NUM_SCALES - 1 - s_rev
            dims = np.array(blur_kernel.shape) * (1.0 / RESIZE_STEP) ** s_rev
            dims = dims.astype(int)
            dims = dims + (1 - dims % 2)
            dims = np.maximum(dims, 3)

            if min(dims) < 4:
                from scipy.ndimage import gaussian_filter
                h_sigma = 1.0
                h_size = tuple(dims)
                blurred_k = gaussian_filter(blur_kernel, sigma=h_sigma)
                blur_kernel_scale[s_idx] = imresize(
                    blurred_k, tuple(dims), 'nearest')
            else:
                blur_kernel_scale[s_idx] = imresize(
                    blur_kernel, tuple(dims), 'bilinear')

            bk_sum = np.sum(blur_kernel_scale[s_idx])
            if bk_sum > 0:
                blur_kernel_scale[s_idx] /= bk_sum

        if self.rescale_then_grad:

            obs_im_scale = [None] * NUM_SCALES
            obs_im_scale[NUM_SCALES - 1] = obs_im.copy()
            mask_scale = [None] * NUM_SCALES
            mask_scale[NUM_SCALES - 1] = mask_all_patch[1:-1, 1:-1]\
                if mask_all_patch.shape[0] > 2 else mask_all_patch.copy()

            for s_rev in range(1, NUM_SCALES):
                s_idx = NUM_SCALES - 1 - s_rev
                sf = (1.0 / RESIZE_STEP) ** s_rev
                obs_im_scale[s_idx] = imresize(obs_im, sf, 'bilinear')
                raw_mask = np.ceil(np.abs(imresize(
                    mask_all_patch, sf, 'nearest')))

                if raw_mask.shape[0] > 2 and raw_mask.shape[1] > 2:
                    mask_scale[s_idx] = raw_mask[1:-1, 1:-1]
                else:
                    mask_scale[s_idx] = raw_mask

            if self.gradient_mode == 'haar':
                kxg = np.array([[0, 1, -1]], dtype=np.float64)
                kyg = np.array([[0], [1], [-1]], dtype=np.float64)
            else:
                kxg = kx
                kyg = ky

            obs_grad_scale_x = [None] * NUM_SCALES
            obs_grad_scale_y = [None] * NUM_SCALES
            obs_grad_scale = [None] * NUM_SCALES
            for s_idx in range(NUM_SCALES):
                obs_grad_scale_x[s_idx] = convolve2d(
                    obs_im_scale[s_idx], kxg, mode='valid', boundary='fill')
                obs_grad_scale_y[s_idx] = convolve2d(
                    obs_im_scale[s_idx], kyg, mode='valid', boundary='fill')

            for s_idx in range(NUM_SCALES):
                trimmed_gx = obs_grad_scale_x[s_idx][1:-1, :]
                trimmed_gy = obs_grad_scale_y[s_idx][:, 1:-1]
                obs_grad_scale[s_idx] = np.concatenate([
                    trimmed_gx, trimmed_gy,
                ], axis=1)

                gh, gw_half = trimmed_gx.shape
                if mask_scale[s_idx].shape != (gh, gw_half):
                    mask_scale[s_idx] = imresize(
                        mask_scale[s_idx].astype(np.float64),
                        (gh, gw_half), 'nearest')

        else:

            obs_grad_all_x = convolve2d(obs_im, kx, mode='valid',
                                        boundary='fill')
            obs_grad_all_y = convolve2d(obs_im, ky, mode='valid',
                                        boundary='fill')

            yy = min(obs_grad_all_x.shape[0], obs_grad_all_y.shape[0])
            xx = min(obs_grad_all_x.shape[1], obs_grad_all_y.shape[1])
            obs_grad_all_x = obs_grad_all_x[:yy, :xx]
            obs_grad_all_y = obs_grad_all_y[:yy, :xx]

            obs_grad_scale_x = [None] * NUM_SCALES
            obs_grad_scale_y = [None] * NUM_SCALES
            obs_grad_scale = [None] * NUM_SCALES
            mask_scale = [None] * NUM_SCALES

            obs_grad_scale_x[NUM_SCALES - 1] = obs_grad_all_x.copy()
            obs_grad_scale_y[NUM_SCALES - 1] = obs_grad_all_y.copy()
            mask_scale[NUM_SCALES - 1] = mask_all_patch[:yy, :xx].copy()

            for s_rev in range(1, NUM_SCALES):
                s_idx = NUM_SCALES - 1 - s_rev
                sf = (1.0 / RESIZE_STEP) ** s_rev
                obs_grad_scale_x[s_idx] = imresize(
                    obs_grad_all_x, sf, 'bilinear')
                obs_grad_scale_y[s_idx] = imresize(
                    obs_grad_all_y, sf, 'bilinear')
                mask_scale[s_idx] = np.ceil(np.abs(imresize(
                    mask_all_patch[:yy, :xx], sf, 'nearest')))

            for s_idx in range(NUM_SCALES):
                gx = obs_grad_scale_x[s_idx]
                gy = obs_grad_scale_y[s_idx]
                obs_grad_scale[s_idx] = np.concatenate([gx, gy], axis=1)

        if self.fft_mode:
            for s_idx in range(NUM_SCALES):
                dk = delta_kernel(blur_kernel_scale[s_idx].shape[0])
                gs = obs_grad_scale[s_idx]
                gs_shifted = np.real(
                    ifft2(fft2(gs) * fft2(dk, s=gs.shape)))
                obs_grad_scale[s_idx] = gs_shifted

                mask_scale[s_idx] = convolve2d(
                    mask_scale[s_idx].astype(np.float64), dk,
                    mode='same', boundary='fill')

        K_arr = np.zeros(NUM_SCALES, dtype=int)
        L_arr = np.zeros(NUM_SCALES, dtype=int)
        M_arr = np.zeros(NUM_SCALES, dtype=int)
        N_arr = np.zeros(NUM_SCALES, dtype=int)
        I_arr = np.zeros(NUM_SCALES, dtype=int)
        J_arr = np.zeros(NUM_SCALES, dtype=int)

        D_list = [None] * NUM_SCALES
        Dp_list = [None] * NUM_SCALES
        spatial_blur_mask = [None] * NUM_SCALES
        spatial_image_mask = [None] * NUM_SCALES

        for s_idx in range(NUM_SCALES):
            K_arr[s_idx] = blur_kernel_scale[s_idx].shape[0]
            L_arr[s_idx] = blur_kernel_scale[s_idx].shape[1]
            M_arr[s_idx] = obs_grad_scale[s_idx].shape[0]
            N_arr[s_idx] = obs_grad_scale[s_idx].shape[1]

            Ks = int(K_arr[s_idx])
            Ls = int(L_arr[s_idx])
            Ms = int(M_arr[s_idx])
            Ns = int(N_arr[s_idx])
            hK = Ks // 2
            hL = Ls // 2

            if self.blur_mask:
                sbm = np.zeros((self.blur_components, Ks * Ls))
                for a in range(self.blur_components):
                    xx, yy = np.meshgrid(
                        np.arange(-hK, hK + 1),
                        np.arange(-hL, hL + 1))
                    pts = np.stack([xx.ravel(), yy.ravel()], axis=0)
                    mu = np.array([0.0, 0.0])
                    cov = np.eye(2) * Ks * self.blur_mask_variances[
                        min(a, len(self.blur_mask_variances) - 1)]
                    sbm[a, :] = np.log(normMDpdf(pts, mu, cov))
                spatial_blur_mask[s_idx] = sbm
            else:
                spatial_blur_mask[s_idx] = np.zeros(
                    (self.blur_components, Ks * Ls))

            if self.saturation_mask == 2:
                sat_p = self.init_prescision * 10.0
                sim = np.concatenate([
                    mask_scale[s_idx] * sat_p,
                    mask_scale[s_idx] * sat_p,
                ], axis=1)
                spatial_image_mask[s_idx] = sim
            else:
                spatial_image_mask[s_idx] = np.zeros(
                    (Ms, Ns), dtype=np.float64)

            if self.fft_mode:
                Is = 2 * Ms
                Js = 2 * Ns
                Dp = np.zeros((Is, Js), dtype=np.float64)

                Dp[Ks - 1:Ms, Ls - 1:Ns // 2] = 1.0
                Dp[Ks - 1:Ms, Ls - 1 + Ns // 2:Ns] = 1.0

                D_data = np.pad(obs_grad_scale[s_idx],
                                ((0, Ms), (0, Ns)), mode='constant')

                if self.saturation_mask == 1:
                    mask_concat = np.concatenate([
                        mask_scale[s_idx], mask_scale[s_idx]], axis=1)
                    Dp = Dp * np.pad(
                        1.0 - mask_concat, ((0, Ms), (0, Ns)),
                        mode='constant')
            else:
                Is = Ms
                Js = Ns
                Dp = np.zeros((Is, Js), dtype=np.float64)
                Dp[hK:Ms - hK, hL:Ns // 2 - hL] = 1.0
                Dp[hK:Ms - hK, Ns // 2 + hL:Ns - hL] = 1.0

                D_data = obs_grad_scale[s_idx].copy()

                if self.saturation_mask == 1:
                    mask_concat = np.concatenate([
                        mask_scale[s_idx], mask_scale[s_idx]], axis=1)
                    Dp = Dp * (1.0 - mask_concat)

            I_arr[s_idx] = Is
            J_arr[s_idx] = Js
            D_list[s_idx] = D_data
            Dp_list[s_idx] = Dp

        if self.priors is not None:
            priors_list = self.priors
        else:

            priors_list = get_default_priors(
                'street', self.image_components)

        me_est = [None] * NUM_SCALES
        mx_est = [None] * NUM_SCALES
        new_grad = [None] * NUM_SCALES
        new_blur = [None] * NUM_SCALES
        D_log_all = []

        for s_idx in range(NUM_SCALES):
            Ks = int(K_arr[s_idx])
            Ls = int(L_arr[s_idx])
            Ms = int(M_arr[s_idx])
            Ns = int(N_arr[s_idx])
            Is = int(I_arr[s_idx])
            Js = int(J_arr[s_idx])

            prior_idx = NUM_SCALES - s_idx - 1
            if prior_idx >= len(priors_list):
                prior_idx = len(priors_list) - 1
            if prior_idx > 7:
                prior_idx = 7
            cur_priors = priors_list[prior_idx]

            dimensions = np.array([
                [1,      1,          1,
                 0, 0, 1],
                [1,      Ks * Ls,    self.blur_components,
                 self.blur_prior, int(self.blur_lock), 1],
                [obs_imz, Ms * Ns,   self.image_components,
                 self.image_prior, 1, 1],
            ], dtype=np.float64)

            if s_idx == 0:

                x1, x2 = initialize_parameters2(
                    obs_grad_scale[s_idx],
                    blur_kernel_scale[s_idx],
                    obs_grad_scale[s_idx],
                    blur_kernel_scale[s_idx],
                    obs_grad_scale[s_idx],
                    self.init_prescision,
                    self.image_prior,
                    self.image_components,
                    self.first_init_mode_image,
                    self.first_init_mode_blur,
                    obs_im,
                    blur_kernel_scale[NUM_SCALES - 1],
                    spatial_image_mask[s_idx],
                    cur_priors,
                    self.fft_mode,
                    COLOR,
                    1,
                )
            else:

                x1, x2 = initialize_parameters2(
                    obs_grad_scale[s_idx],
                    new_blur[s_idx - 1],
                    new_grad[s_idx - 1],
                    None,
                    None,
                    self.init_prescision,
                    self.image_prior,
                    self.image_components,
                    self.init_mode_image,
                    self.init_mode_blur,
                    obs_im,
                    blur_kernel_scale[NUM_SCALES - 1],
                    spatial_image_mask[s_idx],
                    cur_priors,
                    self.fft_mode,
                    COLOR,
                    1,
                )

            print(f"Scale={s_idx + 1}/{NUM_SCALES}  "
                  f"K={Ks}x{Ls}  M={Ms}x{Ns}")

            options = [
                self.convergence, 0, self.noise_init,
                0, 0, self.max_iterations, 0,
            ]

            sat_mask_flat = spatial_image_mask[s_idx].ravel()
            image_mask_vec = 1.0 - (sat_mask_flat > 0).astype(np.float64)

            ensemble, D_log_s, gamma_log_s = train_ensemble_main6(
                dimensions, x1, x2, options,
                D_list[s_idx], Dp_list[s_idx],
                Is, Js, Ks, Ls, Ms, Ns,
                cur_priors, int(self.fft_mode),
                spatial_blur_mask[s_idx], image_mask_vec,
            )

            D_log_all.append(D_log_s)

            me_est[s_idx] = train_ensemble_get(
                2, dimensions, ensemble['mx']).reshape(Ks, Ls)
            mx_est[s_idx] = train_ensemble_get(
                3, dimensions, ensemble['mx']).reshape(Ms, Ns)

            print(f"  Kernel[{s_idx}]: min={me_est[s_idx].min():.6g}, "
                  f"max={me_est[s_idx].max():.6g}, "
                  f"sum={me_est[s_idx].sum():.6g}, "
                  f"shape={me_est[s_idx].shape}")

            if s_idx < NUM_SCALES - 1:
                Kn = int(K_arr[s_idx + 1])
                Ln = int(L_arr[s_idx + 1])
                Mn = int(M_arr[s_idx + 1])
                Nn = int(N_arr[s_idx + 1])

                new_grad[s_idx], new_blur[s_idx] = move_level(
                    mx_est[s_idx], me_est[s_idx],
                    Kn, Ln, Mn, Nn,
                    self.upsample_mode, RESIZE_STEP,
                    self.center_blur,
                )

        self.history['D_log'] = D_log_all

        final_N = int(N_arr[NUM_SCALES - 1])
        final_dx = mx_est[NUM_SCALES - 1][:, :final_N // 2].copy()
        final_dy = mx_est[NUM_SCALES - 1][:, final_N // 2:final_N].copy()

        final_dx[0, :] = 0;  final_dx[-1, :] = 0
        final_dx[:, 0] = 0;  final_dx[:, -1] = 0
        final_dy[0, :] = 0;  final_dy[-1, :] = 0
        final_dy[:, 0] = 0;  final_dy[:, -1] = 0

        obs_im_recon, _ = reconsEdge3(final_dx, final_dy)

        deblurred, kernel_out = richardson_lucy(
            obs_im_orig,
            me_est,
            gamma_correction=self.gamma_correction,
            prescale=1.0,
            lucy_its=self.lucy_its,
            threshold=self.kernel_threshold,
            scale_offset=self.scale_offset,
            resize_step=RESIZE_STEP,
        )

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'num_scales': self.num_scales,
            'gamma_correction': self.gamma_correction,
            'lucy_its': self.lucy_its,
            'kernel_threshold': self.kernel_threshold,
            'blur_components': self.blur_components,
            'image_components': self.image_components,
            'convergence': self.convergence,
            'max_iterations': self.max_iterations,
            'time': time.time() - start_time,
        }

        x_final = np.asarray(deblurred, dtype=np.float64)
        if x_final.max() <= 1.0:
            x_final = x_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        return x_final, kernel_out

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('num_scales', self.num_scales),
            ('resize_step', self.resize_step),
            ('resize_mode', self.resize_mode),
            ('gamma_correction', self.gamma_correction),
            ('prescale', self.prescale),
            ('convergence', self.convergence),
            ('max_iterations', self.max_iterations),
            ('noise_init', self.noise_init),
            ('blur_components', self.blur_components),
            ('image_components', self.image_components),
            ('blur_prior', self.blur_prior),
            ('image_prior', self.image_prior),
            ('init_prescision', self.init_prescision),
            ('first_init_mode_image', self.first_init_mode_image),
            ('first_init_mode_blur', self.first_init_mode_blur),
            ('init_mode_image', self.init_mode_image),
            ('init_mode_blur', self.init_mode_blur),
            ('upsample_mode', self.upsample_mode),
            ('center_blur', self.center_blur),
            ('gradient_mode', self.gradient_mode),
            ('rescale_then_grad', self.rescale_then_grad),
            ('lucy_its', self.lucy_its),
            ('kernel_threshold', self.kernel_threshold),
            ('scale_offset', self.scale_offset),
            ('fft_mode', self.fft_mode),
            ('blur_lock', self.blur_lock),
            ('blur_mask', self.blur_mask),
            ('saturation_mask', self.saturation_mask),
            ('automatic_patch', self.automatic_patch),
            ('patch_size', self.patch_size),
            ('patch_location', self.patch_location),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams

    @staticmethod
    def train_priors(image_paths: list,
                     num_components: int = 4,
                     num_scales: int = 8) -> list:

        import cv2 as cv
        images = []
        for p in image_paths:
            im = cv.imread(str(p), cv.IMREAD_GRAYSCALE)
            if im is None:
                raise FileNotFoundError(f"Cannot read image: {p}")
            images.append(im.astype(np.float64))
        return estimate_priors_from_images(images, num_components, num_scales)
