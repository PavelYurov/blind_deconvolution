from __future__ import annotations

from time import time
from typing import Any, Dict, Tuple

import numpy as np

from ..base import DeconvolutionAlgorithm

from .source.run import update_gamma, update_qx, update_theta


def _init_kernel(radius: int, rng: np.random.Generator) -> np.ndarray:
    size = 2 * radius + 1
    kernel = rng.random((size, size), dtype=np.float64)
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        kernel[...] = 1.0 / (size * size)
    else:
        kernel /= kernel_sum
    return kernel


def _integrate_gradients(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Reconstruct an image whose gradients best match (gx, gy)."""
    gy_fft = np.fft.fft2(gy)
    gx_fft = np.fft.fft2(gx)

    h, w = gx.shape
    ky = 2 * np.pi * np.fft.fftfreq(h).reshape(-1, 1)
    kx = 2 * np.pi * np.fft.fftfreq(w).reshape(1, -1)

    denom = kx**2 + ky**2
    numerator = 1j * (kx * gx_fft + ky * gy_fft)

    potential_fft = np.zeros_like(gx_fft, dtype=np.complex128)
    mask = denom > 1e-12
    potential_fft[mask] = numerator[mask] / denom[mask]
    potential_fft[0, 0] = 0.0

    reconstructed = np.fft.ifft2(potential_fft).real
    reconstructed -= reconstructed.min()
    max_val = reconstructed.max()
    if max_val > 0:
        reconstructed /= max_val
    return reconstructed


class Deu439SblBlindDeconvolution(DeconvolutionAlgorithm):
    """Variational Bayesian blind deconvolution inspired by the deu439 implementation."""

    def __init__(
        self,
        kernel_radius: int = 5,
        noise_precision: float = 1.0,
        iterations: int = 1,
        random_seed: int | None = 0,
    ) -> None:
        super().__init__('VariationalSBL')
        self.kernel_radius = int(kernel_radius)
        if self.kernel_radius <= 0:
            raise ValueError("kernel_radius must be positive")
        self.noise_precision = float(noise_precision)
        self.iterations = int(iterations)
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        self.random_seed = random_seed

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'kernel_radius' in param and param['kernel_radius'] is not None:
            self.kernel_radius = int(param['kernel_radius'])
        if 'kernel_size' in param and param['kernel_size'] is not None:
            size = int(param['kernel_size'])
            self.kernel_radius = max(1, size // 2)
        if 'noise_precision' in param and param['noise_precision'] is not None:
            self.noise_precision = float(param['noise_precision'])
        if 'lambda' in param and param['lambda'] is not None:
            self.noise_precision = float(param['lambda'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = int(param['iterations'])
        if 'random_seed' in param:
            value = param['random_seed']
            self.random_seed = None if value in (None, '') else int(value)

        return super().change_param(param)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start = time()
        original_dtype = image.dtype

        if image.ndim == 3 and image.shape[2] != 1:
            working = image.mean(axis=2)
        else:
            working = image.squeeze()

        working = working.astype(np.float64, copy=False)
        if working.size and float(working.max()) > 1.5:
            working = working / 255.0

        gx, gy = np.gradient(working)
        y = np.stack([gx, gy])

        rng = np.random.default_rng(self.random_seed)
        theta = _init_kernel(self.kernel_radius, rng)
        gamma = np.ones_like(y)

        mu = np.zeros_like(y)
        sigma2 = np.ones_like(working)
        
        kernel = np.asarray(theta, dtype=np.float32)
        for i in range(self.iterations):
            mu, sigma2 = update_qx(y, theta, gamma, self.noise_precision)
            theta = update_theta(y, mu, sigma2, self.kernel_radius)
            gamma = update_gamma(mu, sigma2)
            print('Iteration:', i)
            kernel = kernel + np.asarray(theta, dtype=np.float32)

        # kernel = np.asarray(theta, dtype=np.float32)
        kernel = np.clip(kernel, 0.0, None)
        kernel_sum = float(kernel.sum())
        print(f'kernel_sum: {kernel_sum}')
        # if kernel_sum > 0:
        #     kernel /= kernel_sum

        restored = _integrate_gradients(mu[0], mu[1])

        self.timer = time() - start

        restored = np.clip(restored, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.integer):
            restored_out = (restored * 255.0).round().astype(original_dtype)
        else:
            restored_out = restored.astype(original_dtype, copy=False)

        return restored_out, kernel

    def get_param(self):
        return [
            ('kernel_radius', self.kernel_radius),
            ('noise_precision', self.noise_precision),
            ('iterations', self.iterations),
            ('random_seed', self.random_seed),
        ]

__all__ = ["Deu439SblBlindDeconvolution"]
