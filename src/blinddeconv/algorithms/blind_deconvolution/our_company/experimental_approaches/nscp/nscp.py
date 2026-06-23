"""
nscp.py

Источник:
    D. Yang, X. Wu, H. Yin: "Blind Image Deblurring via a Novel Sparse
    Channel Prior", Mathematics, 2022.
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

from .utils import (
    dark_channel,
    bright_channel,
    bcpl0norm,
    compute_gradients,
    threshold_gradient,
    gaussian_pyramid,
    upsample_l,
    upsample_small_kernel,
    clean_kernel,
    make_delta_kernel,
)
from .solvers import update_l, update_kernel, final_restore

class NSCP_BD(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_size: int = 25,
        kernel_max_size: int = 35,
        num_scales: int = 4,
        max_iter: int = 10,
        mu: float = 0.003,
        lambda_grad: float = 0.02,
        xi: float = 0.02,
        theta: float = 0.003,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
        dcp_window: int = 15,
        bcp_window: int = 15,
        snr_const: float = 0.015,
    ):
        super().__init__(name='NSCP-BD')

        self.kernel_size = kernel_size
        self.kernel_max_size = kernel_max_size
        self.num_scales = num_scales
        self.max_iter = max_iter
        self.mu = mu
        self.lambda_grad = lambda_grad
        self.xi = xi
        self.theta = theta
        self.gamma = gamma
        self.epsilon = epsilon
        self.dcp_window = dcp_window
        self.bcp_window = bcp_window
        self.snr_const = snr_const

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        start_time = time.time()

        b_full = image.astype(np.float32)
        if b_full.max() > 1.0:
            b_full /= 255.0

        pyramid = gaussian_pyramid(b_full, self.num_scales)

        l = pyramid[0].astype(np.float32).copy()
        if l.max() > 1.0:
            l /= 255.0

        coarsest_h, coarsest_w = pyramid[0].shape[:2]
        init_ks = max(3, min(self.kernel_size,
                             coarsest_h // 3, coarsest_w // 3))
        if init_ks % 2 == 0:
            init_ks += 1
        k = make_delta_kernel(init_ks)

        for scale_idx in range(len(pyramid)):
            b_scaled = pyramid[scale_idx]
            H, W = b_scaled.shape[:2]

            if scale_idx == 0:
                gamma_scale = 0.5
            elif scale_idx == 1:
                gamma_scale = 0.2
            else:
                gamma_scale = 0.05

            if scale_idx > 0:
                l = upsample_l(l, (H, W))

                max_kh = min(self.kernel_max_size, H // 2)
                max_kw = min(self.kernel_max_size, W // 2)
                max_kh = max(max_kh, 3)
                max_kw = max(max_kw, 3)
                k = upsample_small_kernel(
                    k, scale_factor=2, max_size=(max_kh, max_kw)
                )

                if k.sum() <= 1e-12:
                    k = make_delta_kernel(k.shape)
                else:
                    k = k / k.sum()

            eff_dcp_w = min(self.dcp_window, max(3, H // 3))
            eff_bcp_w = min(self.bcp_window, max(3, W // 3))
            if eff_dcp_w % 2 == 0:
                eff_dcp_w -= 1
            if eff_bcp_w % 2 == 0:
                eff_bcp_w -= 1

            num_pixels = H * W
            for it in range(self.max_iter):

                B = bright_channel(l, window_size=eff_bcp_w)
                B_l0 = bcpl0norm(B) / num_pixels
                w_k = self.mu / (B_l0 + self.epsilon)

                D = dark_channel(l, window_size=eff_dcp_w)
                threshold_val = w_k / self.xi

                p = l.copy()

                mask_should_be_zero = D * D < threshold_val
                if l.ndim == 3:
                    mask_broadcast = np.repeat(
                        mask_should_be_zero[:, :, np.newaxis], l.shape[2], axis=2
                    )
                    p[mask_broadcast] = 0.0
                else:
                    p[mask_should_be_zero] = 0.0

                gh, gv = compute_gradients(l)
                g = threshold_gradient(
                    (gh, gv), self.theta, self.lambda_grad
                )

                l = update_l(
                    l=l, k=k, b=b_scaled, g=g, p=p,
                    lam=self.lambda_grad, xi=self.xi,
                )

                k = update_kernel(
                    l=l, b=b_scaled, gamma=gamma_scale,
                    image_shape=b_scaled.shape[:2], prev_k=k,
                )

                k = clean_kernel(k)

                if k.sum() <= 1e-12:
                    k = make_delta_kernel(k.shape)

        final_image = final_restore(b_full, k, snr_const=self.snr_const)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'kernel_max_size': self.kernel_max_size,
            'num_scales': self.num_scales,
            'max_iter': self.max_iter,
            'mu': self.mu,
            'lambda_grad': self.lambda_grad,
            'xi': self.xi,
            'theta': self.theta,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'dcp_window': self.dcp_window,
            'bcp_window': self.bcp_window,
            'snr_const': self.snr_const,
            'time': time.time() - start_time,
        }

        x_final = final_image * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, k

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('kernel_max_size', self.kernel_max_size),
            ('num_scales', self.num_scales),
            ('max_iter', self.max_iter),
            ('mu', self.mu),
            ('lambda_grad', self.lambda_grad),
            ('xi', self.xi),
            ('theta', self.theta),
            ('gamma', self.gamma),
            ('epsilon', self.epsilon),
            ('dcp_window', self.dcp_window),
            ('bcp_window', self.bcp_window),
            ('snr_const', self.snr_const),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
