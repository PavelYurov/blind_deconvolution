import numpy as np
import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

from .utils import (compute_sobolev_weights, normalize_min_max,
                   pad_image, crop_center, edgetaper)
from .solvers import solve_step_0, solve_step_n

def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path

try:
    _CURRENT_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
    _SRC_DIR = _PROJECT_ROOT / "src"
    _ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"
    for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from blinddeconv.algorithms.base import DeconvolutionAlgorithm
except Exception:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class MHDMBlind(DeconvolutionAlgorithm):

    def __init__(
        self,
        kernel_shape: Tuple[int, int],

        lambda_0: float = 1e-4,
        mu_0: float = 1e-3,
        r: float = 0.8,
        s: float = 0.1,
        scaling_factor: float = 1.3,
        noise_level: float = 0.01,
        tau: float = 1.01,
        max_iter: int = 50,

        kernel_threshold: float = 0.05,

        final_deconv_method: str = 'tikhonov',
        final_alpha: float = 1e-2,

        auto_scale_params: bool = True
    ):
        super().__init__(name='MHDM-Blind-SingleScale')
        self.kernel_shape = kernel_shape
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.r = r
        self.s = s
        self.scaling_factor = scaling_factor
        self.noise_level = noise_level
        self.tau = tau
        self.max_iter = max_iter

        self.kernel_threshold = kernel_threshold
        self.final_deconv_method = final_deconv_method
        self.final_alpha = final_alpha
        self.auto_scale_params = auto_scale_params

        self.history = {'residual': []}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0: y /= 255.0
        orig_H, orig_W = y.shape

        y_for_est = edgetaper(y, self.kernel_shape)

        H, W = y_for_est.shape
        F_f = np.fft.fft2(y_for_est)

        W_r = compute_sobolev_weights((H, W), self.r)
        W_s = compute_sobolev_weights((H, W), self.s)

        lambda_curr = self.lambda_0
        mu_curr = self.mu_0

        if self.auto_scale_params:
            mean_signal = np.mean(np.abs(F_f))
            mean_Wr = np.mean(W_r)
            mean_Ws = np.mean(W_s)

            mu_curr = 0.002 * (mean_signal / mean_Ws)

            lambda_curr = mu_curr * 0.1

        F_U = np.zeros_like(F_f, dtype=np.complex128)
        F_K = np.zeros_like(F_f, dtype=np.complex128)

        stop_threshold = self.tau * (H * W) * (self.noise_level ** 2)
        self.history['residual'] = []

        converged_iter = self.max_iter
        for n in range(self.max_iter):
            if n == 0:

                F_u_inc, F_k_inc = solve_step_0(F_f, lambda_curr, mu_curr, W_r, W_s)
            else:

                lambda_curr /= self.scaling_factor
                mu_curr /= self.scaling_factor

                F_u_inc, F_k_inc = solve_step_n(F_f, F_U, F_K, lambda_curr, mu_curr, W_r, W_s)

            F_U += F_u_inc
            F_K += F_k_inc

            res_sq = np.sum(np.abs(F_f - F_K * F_U)**2) / (H * W)
            self.history['residual'].append(res_sq)

            if res_sq <= stop_threshold and n > 15:
                converged_iter = n

                break

        k_full = np.real(np.fft.ifft2(F_K))

        k_shifted = np.fft.fftshift(k_full)

        kh, kw = self.kernel_shape
        cy, cx = H // 2, W // 2
        sy, sx = cy - kh // 2, cx - kw // 2
        k_final = k_shifted[sy:sy+kh, sx:sx+kw]

        k_final = np.maximum(k_final, 0)

        thresh_val = k_final.max() * self.kernel_threshold
        k_final[k_final < thresh_val] = 0

        k_sum = k_final.sum()
        if k_sum > 1e-12:
            k_final /= k_sum
        else:

            k_final[kh//2, kw//2] = 1.0

        pad_h = kh + 2
        pad_w = kw + 2
        y_padded = pad_image(y, pad_h, mode='reflect')

        u_restored_padded = self.run_final_deconv(y_padded, k_final)

        u_final = crop_center(u_restored_padded, (orig_H, orig_W))

        u_final = np.clip(u_final, 0, 1)
        u_out = (u_final * 255.0).astype(np.uint8)

        self.hyperparams['time'] = time.time() - start_time
        self.hyperparams['iterations'] = converged_iter

        return u_out, k_final

    def run_final_deconv(self, y: np.ndarray, k: np.ndarray) -> np.ndarray:

        H, W = y.shape
        kh, kw = k.shape

        k_pad = np.zeros((H, W))
        cy, cx = H//2, W//2
        sy, sx = cy - kh//2, cx - kw//2
        k_pad[sy:sy+kh, sx:sx+kw] = k
        k_pad = np.fft.ifftshift(k_pad)

        F_y = np.fft.fft2(y)
        F_k = np.fft.fft2(k_pad)

        F_k_abs2 = np.abs(F_k)**2

        if self.final_deconv_method == 'wiener':

            denom = F_k_abs2 + self.final_alpha
            F_u = np.conj(F_k) * F_y / denom

        elif self.final_deconv_method == 'tikhonov':

            freq_y = np.fft.fftfreq(H).reshape(-1, 1)
            freq_x = np.fft.fftfreq(W).reshape(1, -1)
            L_sq = (freq_y**2 + freq_x**2)

            L_sq /= (L_sq.max() + 1e-12)

            denom = F_k_abs2 + self.final_alpha * L_sq + 1e-9
            F_u = np.conj(F_k) * F_y / denom

        else:

            return y

        return np.real(np.fft.ifft2(F_u))

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('scaling_factor', self.scaling_factor),
            ('s', self.s),
            ('kernel_threshold', self.kernel_threshold),
            ('final_deconv_method', self.final_deconv_method),
            ('final_alpha', self.final_alpha),
            ('max_iter', self.max_iter)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
