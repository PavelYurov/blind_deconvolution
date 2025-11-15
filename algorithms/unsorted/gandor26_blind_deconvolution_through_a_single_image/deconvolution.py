import math
from typing import Tuple

import numpy as np
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from .compressed_sense import CompressedSense


class DeConvolution:
    def __init__(
        self,
        blurred: np.ndarray,
        kernel_shape: Tuple[int, int],
        tol: float = 1e-5,
        max_steps: int = 8,
    ) -> None:
        if blurred.ndim != 2:
            raise ValueError("DeConvolution expects a single-channel image")

        self.blurred = blurred.astype(np.float64, copy=False)
        self.height, self.width = self.blurred.shape
        self.kernel_shape = (int(kernel_shape[0]), int(kernel_shape[1]))
        self.tol = float(tol)
        self.max_steps = int(max_steps)

        self.lambda_global = 0.05
        self.lambda_local = 20.0
        self.gamma_init = 2.0
        self.gamma = self.gamma_init
        self.kappa_1 = 1.2
        self.kappa_2 = 1.5
        self.weight_init = 50.0

        self.psi_x = np.zeros_like(self.blurred)
        self.psi_y = np.zeros_like(self.blurred)
        self.original = self.blurred.copy()
        self.kernel = self._initial_kernel()

        self.grad_x_blurred = self._gradient(self.blurred, axis=1)
        self.grad_y_blurred = self._gradient(self.blurred, axis=0)
        self.smooth_region = self._estimate_smooth_region(window_size=7, threshold=0.02)

        self.F_grad_x = self._frequency_of_kernel(self._finite_difference_kernel(axis=1))
        self.F_grad_y = self._frequency_of_kernel(self._finite_difference_kernel(axis=0))
        self.F_blurred = np.fft.fft2(self.blurred)
        self.F_delta = self._build_frequency_weight()

    def run(self) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(self.max_steps):
            prev_kernel = self.kernel.copy()
            F_kernel = self._frequency_of_kernel(self.kernel)

            while True:
                prev_original = self.original.copy()
                prev_psi_x = self.psi_x.copy()
                prev_psi_y = self.psi_y.copy()

                grad_x = self._gradient(self.original, axis=1)
                grad_y = self._gradient(self.original, axis=0)

                self.psi_x = self._update_psi(self.grad_x_blurred, grad_x)
                self.psi_y = self._update_psi(self.grad_y_blurred, grad_y)

                self.original = self._update_original(self.psi_x, self.psi_y, F_kernel)

                delta_L = np.linalg.norm(self.original - prev_original)
                dx = np.linalg.norm(self.psi_x - prev_psi_x)
                dy = np.linalg.norm(self.psi_y - prev_psi_y)
                delta_psi = math.hypot(dx, dy)
                self.gamma *= 2.0
                if delta_L < self.tol and delta_psi < self.tol:
                    break

            solver = CompressedSense(self.kernel_shape, (self.height, self.width))
            solver.mat2vector(self.original, self.blurred)
            solver.solve()
            self.kernel = solver.vector2mat()
            self.gamma = self.gamma_init

            kernel_delta = np.linalg.norm(self.kernel - prev_kernel)
            if kernel_delta < self.tol:
                break
            self.lambda_global /= self.kappa_1
            self.lambda_local /= self.kappa_2

        restored = np.clip(self.original, 0.0, 1.0)
        kernel_out = np.clip(self.kernel, 0.0, 1.0)
        return restored, kernel_out

    def _estimate_smooth_region(self, window_size: int, threshold: float) -> np.ndarray:
        if cv2 is not None:
            win = (window_size, window_size)
            mean = cv2.blur(self.blurred, win)
            sq_mean = cv2.blur(self.blurred * self.blurred, win)
        else:
            mean = self._naive_box_blur(self.blurred, window_size)
            sq_mean = self._naive_box_blur(self.blurred * self.blurred, window_size)
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        mask = (std < threshold).astype(np.float64)
        return mask
    def _naive_box_blur(self, image: np.ndarray, window_size: int) -> np.ndarray:
        radius = window_size // 2
        padded = np.pad(image, radius, mode="edge")
        out = np.empty_like(image)
        for y in range(self.height):
            y0 = y
            y1 = y + window_size
            for x in range(self.width):
                x0 = x
                x1 = x + window_size
                out[y, x] = padded[y0:y1, x0:x1].mean()
        return out

    def _gradient(self, image: np.ndarray, axis: int) -> np.ndarray:
        if axis == 0:
            shifted = np.roll(image, -1, axis=0)
            grad = shifted - image
            grad[-1, :] = 0.0
        else:
            shifted = np.roll(image, -1, axis=1)
            grad = shifted - image
            grad[:, -1] = 0.0
        return grad

    def _update_psi(self, grad_blurred: np.ndarray, grad_original: np.ndarray) -> np.ndarray:
        a = 6.1e-4
        b = 5.0
        k = 2.7
        l_t = 1.8526

        M = self.smooth_region
        I = grad_blurred
        L = grad_original

        denom_common = self.lambda_local * M + self.gamma
        denom_common = np.where(denom_common == 0.0, 1e-8, denom_common)
        psi_star0 = (self.lambda_local * M * I + self.gamma * L + self.lambda_global * k / 2.0) / denom_common
        psi_star1 = (self.lambda_local * M * I + self.gamma * L - self.lambda_global * k / 2.0) / denom_common
        denom_quad = self.lambda_global * a + self.lambda_local * M + self.gamma
        denom_quad = np.where(denom_quad == 0.0, 1e-8, denom_quad)
        psi_star2 = (self.lambda_local * M * I + self.gamma * L) / denom_quad

        candidates = np.empty((*I.shape, 3), dtype=np.float64)
        candidates[..., 0] = np.clip(psi_star0, -l_t, 0.0)
        candidates[..., 1] = np.clip(psi_star1, 0.0, l_t)
        psi2 = psi_star2.copy()
        mask_between = (psi2 >= -l_t) & (psi2 <= l_t)
        psi2 = np.where(mask_between, psi2, np.where(psi2 > l_t, l_t, -l_t))
        candidates[..., 2] = psi2

        def calc_energy(x: np.ndarray, flag: int) -> np.ndarray:
            if flag == 2:
                prior = self.lambda_global * (a * x * x + b)
            else:
                prior = self.lambda_global * k * np.abs(x)
            return prior + self.lambda_local * M * (x - I) ** 2 + self.gamma * (x - L) ** 2

        energy = np.stack([
            calc_energy(candidates[..., 0], 0),
            calc_energy(candidates[..., 1], 1),
            calc_energy(candidates[..., 2], 2),
        ], axis=-1)

        best_idx = np.argmin(energy, axis=-1)
        out = np.take_along_axis(candidates, best_idx[..., None], axis=-1)[..., 0]
        return out

    def _update_original(self, psi_x: np.ndarray, psi_y: np.ndarray, F_kernel: np.ndarray) -> np.ndarray:
        F_psi_x = np.fft.fft2(psi_x)
        F_psi_y = np.fft.fft2(psi_y)

        numerator = (
            np.conj(F_kernel) * self.F_blurred * self.F_delta
            + self.gamma * np.conj(self.F_grad_x) * F_psi_x
            + self.gamma * np.conj(self.F_grad_y) * F_psi_y
        )
        denominator = (
            (np.abs(F_kernel) ** 2) * self.F_delta
            + self.gamma * (np.abs(self.F_grad_x) ** 2 + np.abs(self.F_grad_y) ** 2)
        )
        denominator = np.where(denominator == 0.0, 1e-8, denominator)
        F_L = numerator / denominator
        restored = np.fft.ifft2(F_L)
        return np.abs(restored)

    def _build_frequency_weight(self) -> np.ndarray:
        filters = []
        w = self.weight_init

        base = np.zeros_like(self.blurred)
        base[0, 0] = 1.0
        filters.append((base.copy(), w))

        w /= 2.0
        base = np.zeros_like(self.blurred)
        base[0, 0] = 1.0
        base[0, 1] = -1.0
        filters.append((base.copy(), w))

        base = np.zeros_like(self.blurred)
        base[0, 0] = 1.0
        base[1, 0] = -1.0
        filters.append((base.copy(), w))

        w /= 2.0
        base = np.zeros_like(self.blurred)
        base[0, 0] = -1.0
        if self.width > 1:
            base[0, 1] = 2.0
        if self.width > 2:
            base[0, 2] = -1.0
        filters.append((base.copy(), w))

        base = np.zeros_like(self.blurred)
        base[0, 0] = -1.0
        if self.height > 1:
            base[1, 0] = 2.0
        if self.height > 2:
            base[2, 0] = -1.0
        filters.append((base.copy(), w))

        base = np.zeros_like(self.blurred)
        base[0, 0] = -1.0
        if self.height > 1:
            base[1, 0] = 1.0
        if self.width > 1:
            base[0, 1] = 1.0
        if self.height > 1 and self.width > 1:
            base[1, 1] = -1.0
        filters.append((base.copy(), w))

        F_delta = np.zeros_like(self.blurred)
        for kernel, weight in filters:
            spectrum = np.fft.fft2(kernel)
            F_delta += np.abs(spectrum) ** 2 * weight
        return F_delta

    def _finite_difference_kernel(self, axis: int) -> np.ndarray:
        kernel = np.zeros_like(self.blurred)
        kernel[0, 0] = 1.0
        if axis == 1:
            kernel[0, 1] = -1.0
        else:
            kernel[1, 0] = -1.0
        return kernel

    def _frequency_of_kernel(self, kernel: np.ndarray) -> np.ndarray:
        pad = np.zeros_like(self.blurred)
        kh, kw = kernel.shape
        cy = kh // 2
        cx = kw // 2
        for y in range(kh):
            for x in range(kw):
                val = kernel[y, x]
                if val == 0.0:
                    continue
                ty = (y - cy) % self.height
                tx = (x - cx) % self.width
                pad[ty, tx] = val
        return np.fft.fft2(pad)

    def _initial_kernel(self) -> np.ndarray:
        kernel = np.zeros(self.kernel_shape, dtype=np.float64)
        center_y = self.kernel_shape[0] // 2
        center_x = self.kernel_shape[1] // 2
        kernel[center_y, center_x] = 1.0
        return kernel


__all__ = ["DeConvolution"]
