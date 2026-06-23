import numpy as np
import time
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Any, Dict
import scipy.ndimage

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
from .utils import (get_gradient_operators, edgetaper, pad_image, crop_image,
                    wiener_filter, tikhonov_filter, psf2otf)
from .solvers import (compute_adaptive_matrix_T, solve_k_subproblem,
                      solve_u_subproblem, solve_p_subproblem,
                      solve_q_subproblem, solve_w_subproblem, solve_g_subproblem)

class AEER_BD(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],

        lambda_val: float = 2000.0,
        beta: float = 5.0,
        alpha: float = 0.05,
        iota: float = 5.0,
        delta: float = 1.0,

        r1: float = 50.0,
        r2: float = 50.0,
        r3: float = 200.0,
        r4: float = 10.0,

        xi: float = 0.05,
        max_iter: int = 50,
        tol: float = 1e-5,

        clean_kernel: bool = False,
        boundary_handling: str = 'padding',
        grad_threshold: float = 0.0,

        final_deconv: str = 'admm',
        nb_iter: int = 30,

        debug: bool = True,
        debug_dir: str = "debug_aeer_poisson"
    ):
        super().__init__(name='AEE-BD-Poisson')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_val = lambda_val
        self.beta = beta
        self.alpha = alpha
        self.iota = iota
        self.delta = delta
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.xi = xi
        self.max_iter = max_iter
        self.tol = tol

        self.clean_kernel = clean_kernel
        self.boundary_handling = boundary_handling
        self.grad_threshold = grad_threshold
        self.final_deconv = final_deconv
        self.nb_iter = nb_iter

        self.debug = debug
        self.debug_dir = debug_dir

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def _save_debug_image(self, img: np.ndarray, name: str, iteration: int):
        if not self.debug: return
        if not os.path.exists(self.debug_dir): os.makedirs(self.debug_dir)
        fname = os.path.join(self.debug_dir, f"iter_{iteration:03d}_{name}.png")
        tmp = img.copy()
        tmp -= tmp.min()
        if tmp.max() > 0: tmp /= tmp.max()
        plt.imsave(fname, tmp, cmap='gray')

    def _clean_kernel_func(self, k: np.ndarray) -> np.ndarray:
        max_val = k.max()
        if max_val <= 0: return k
        mask = k > (self.xi * max_val)
        labeled, num_features = scipy.ndimage.label(mask)
        if num_features <= 1: return k
        sizes = scipy.ndimage.sum(k, labeled, range(num_features + 1))
        largest_label = np.argmax(sizes[1:]) + 1
        k_clean = k.copy()
        k_clean[labeled != largest_label] = 0
        return k_clean

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        f_orig = image.astype(np.float64)
        if f_orig.max() > 1.0: f_orig /= 255.0
        H_orig, W_orig = f_orig.shape
        kh, kw = self.kernel_shape

        f= edgetaper(f_orig, (kh, kw))
        H, W = f.shape

        OTF_dx, OTF_dy, _, _ = get_gradient_operators((H, W))
        t1, t2 = compute_adaptive_matrix_T(f, self.iota, self.delta)

        u = f.copy()
        g = f.copy()

        k = np.zeros((H, W))
        ks_init = max(kh, kw) // 2
        ax = np.arange(-ks_init, ks_init + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel_init = np.exp(-(xx**2 + yy**2) / (2. * (ks_init/3)**2))
        kernel_init /= kernel_init.sum()

        cy, cx = H // 2, W // 2
        sh, sw = kernel_init.shape
        k[cy - sh//2 : cy - sh//2 + sh, cx - sw//2 : cx - sw//2 + sw] = kernel_init

        p = np.zeros((2, H, W))
        q = np.zeros((2, H, W))
        w = np.zeros((2, H, W))

        lambda1 = np.zeros((2, H, W))
        lambda2 = np.zeros((2, H, W))
        lambda3 = np.zeros((2, H, W))
        lambda4 = np.zeros((H, W))

        if self.debug:
            self._save_debug_image(f, "processed_input", 0)

        for n in range(self.max_iter):
            k_prev = k.copy()
            u_prev = u.copy()

            k_raw = solve_k_subproblem(u, g, lambda4, w, lambda3, self.r3, self.r4, OTF_dx, OTF_dy)
            k = np.fft.fftshift(k_raw)
            k = np.maximum(k, 0)

            max_idx = np.unravel_index(np.argmax(k, axis=None), k.shape)
            shift_y, shift_x = H // 2 - max_idx[0], W // 2 - max_idx[1]
            k = np.roll(k, (shift_y, shift_x), axis=(0, 1))

            if self.clean_kernel and n > 5:
                k = self._clean_kernel_func(k)

            mask = np.zeros_like(k)
            sy, sx = H // 2 - kh // 2, W // 2 - kw // 2
            mask[sy : sy + kh, sx : sx + kw] = 1.0
            k *= mask

            k[k < self.xi * k.max()] = 0
            s = np.sum(k)
            if s > 1e-12: k /= s
            else: k[H//2, W//2] = 1.0

            u = solve_u_subproblem(k, g, lambda4, p, lambda1, self.r1, self.r4, OTF_dx, OTF_dy)
            u = np.maximum(u, 0.0)
            u = np.minimum(u, 1.0)

            if self.debug and (n % 5 == 0):
                vis_u = crop_image(u, (H_orig, W_orig), (kh, kw)) if self.boundary_handling == 'padding' else u
                self._save_debug_image(vis_u, "u", n+1)
                vis_size = max(kh, kw) + 20
                vis_k = k[H//2-vis_size:H//2+vis_size, W//2-vis_size:W//2+vis_size]
                self._save_debug_image(vis_k, "k", n+1)

            p = solve_p_subproblem(u, q, lambda1, lambda2, t1, t2, self.r1, self.r2)
            q = solve_q_subproblem(p, u_prev, lambda2, t1, t2, self.alpha, self.r2)
            w = solve_w_subproblem(k, lambda3, self.beta, self.r3)

            g = solve_g_subproblem(k, u, f, lambda4, self.lambda_val, self.r4)

            grad_x_u = np.roll(u, -1, axis=1) - u
            grad_y_u = np.roll(u, -1, axis=0) - u

            if self.grad_threshold > 0:
                grad_mag = np.sqrt(grad_x_u**2 + grad_y_u**2)
                noise_mask = grad_mag < self.grad_threshold
                grad_x_u[noise_mask] = 0
                grad_y_u[noise_mask] = 0

            grad_u = np.stack([grad_x_u, grad_y_u])
            Tp = np.stack([t1 * p[0], t2 * p[1]])

            grad_x_k = np.roll(k, -1, axis=1) - k
            grad_y_k = np.roll(k, -1, axis=0) - k
            grad_k = np.stack([grad_x_k, grad_y_k])

            F_k = psf2otf(k, (H, W))
            Ku = np.real(np.fft.ifft2(F_k * np.fft.fft2(u)))

            lambda1 += (grad_u - p)
            lambda2 += (Tp - q)
            lambda3 += (grad_k - w)
            lambda4 += (Ku - g)

            diff = np.linalg.norm(k - k_prev) / (np.linalg.norm(k) + 1e-12)
            self.history['kernel_diff'].append(diff)
            if diff < self.tol and n > 10:
                print(f"Kernel converged at iter {n}")
                break

        cy, cx = H // 2, W // 2
        sy, sx = cy - kh // 2, cx - kw // 2
        kernel_small = k[sy : sy + kh, sx : sx + kw]

        f_pad = pad_image(f_orig, (kh, kw)) if self.boundary_handling == 'padding' else f_orig.copy()
        H_pad, W_pad = f_pad.shape

        k_final_pad = np.zeros((H_pad, W_pad))
        cy_p, cx_p = H_pad // 2, W_pad // 2
        k_final_pad[cy_p - kh//2 : cy_p - kh//2 + kh, cx_p - kw//2 : cx_p - kw//2 + kw] = kernel_small
        k_final_pad /= k_final_pad.sum()

        print(f"Running final deconvolution: {self.final_deconv}...")

        if self.final_deconv == 'wiener':
            u_final_pad = wiener_filter(f_pad, k_final_pad, noise_snr=0.01)

        elif self.final_deconv == 'tikhonov':
            u_final_pad = tikhonov_filter(f_pad, k_final_pad, alpha=0.01)

        elif self.final_deconv == 'admm':

            OTF_dx_p, OTF_dy_p, _, _ = get_gradient_operators((H_pad, W_pad))
            t1_p, t2_p = compute_adaptive_matrix_T(f_pad, self.iota, self.delta)

            u_p = f_pad.copy()
            g_p = f_pad.copy()
            p_p = np.zeros((2, H_pad, W_pad))
            q_p = np.zeros((2, H_pad, W_pad))
            l1_p = np.zeros((2, H_pad, W_pad))
            l2_p = np.zeros((2, H_pad, W_pad))
            l4_p = np.zeros((H_pad, W_pad))

            for i in range(self.nb_iter):
                u_prev_p = u_p.copy()

                u_p = solve_u_subproblem(k_final_pad, g_p, l4_p, p_p, l1_p, self.r1, self.r4, OTF_dx_p, OTF_dy_p)
                u_p = np.clip(u_p, 0, 1)

                p_p = solve_p_subproblem(u_p, q_p, l1_p, l2_p, t1_p, t2_p, self.r1, self.r2)
                q_p = solve_q_subproblem(p_p, u_prev_p, l2_p, t1_p, t2_p, self.alpha, self.r2)

                g_p = solve_g_subproblem(k_final_pad, u_p, f_pad, l4_p, self.lambda_val, self.r4)

                grad_x = np.roll(u_p, -1, axis=1) - u_p
                grad_y = np.roll(u_p, -1, axis=0) - u_p
                grad = np.stack([grad_x, grad_y])
                Tp = np.stack([t1_p * p_p[0], t2_p * p_p[1]])

                F_k_p = psf2otf(k_final_pad, (H_pad, W_pad))
                Ku_p = np.real(np.fft.ifft2(F_k_p * np.fft.fft2(u_p)))

                l1_p += (grad - p_p)
                l2_p += (Tp - q_p)
                l4_p += (Ku_p - g_p)

            u_final_pad = u_p
        else:
            u_final_pad = f_pad

        if self.boundary_handling == 'padding':
            u_final = crop_image(u_final_pad, (H_orig, W_orig), (kh, kw))
        else:
            u_final = u_final_pad

        x_final = u_final * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        self.hyperparams = {
            'lambda': self.lambda_val,
            'iter': n,
            'final_deconv': self.final_deconv,
            'time': time.time() - start_time
        }

        return x_final, kernel_small

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lam', self.lam),
            ('beta', self.beta),
            ('alpha', self.alpha),
            ('iota', self.iota),
            ('delta', self.delta),
            ('xi', self.xi),
            ('max_iter', self.max_iter)
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
