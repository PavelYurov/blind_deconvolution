from __future__ import annotations

from time import time
from typing import Any, Tuple

from collections.abc import Iterable

import numpy as np

from ..base import DeconvolutionAlgorithm
from .source.Codes.algoviolet import violetBD

Array2D = np.ndarray


def _ensure_odd(value: int) -> int:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return size


def _prepare_image(image: Array2D) -> Tuple[Array2D, np.dtype, Tuple[int, ...]]:
    original_dtype = image.dtype
    original_shape = image.shape

    if image.ndim == 3 and image.shape[2] != 1:
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float64, copy=False)
    if working.size == 0:
        raise ValueError("Empty image provided to VioletBD.")

    max_value = float(working.max())
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)

    if working.ndim != 2:
        raise ValueError("VioletBD expects a 2-D grayscale image after preprocessing.")

    return working, original_dtype, original_shape


def _restore_dtype(image: Array2D, original_dtype: np.dtype, original_shape: Tuple[int, ...]) -> Array2D:
    clipped = np.clip(image, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        restored = (clipped * 255.0).round().astype(original_dtype)
    else:
        restored = clipped.astype(original_dtype, copy=False)

    if len(original_shape) == 3:
        channels = original_shape[2]
        if channels == 1:
            restored = restored[..., None]
        else:
            restored = np.repeat(restored[..., None], channels, axis=2)

    return restored


def _normalise_kernel(kernel: Array2D) -> Array2D:
    normalised = np.clip(kernel, 0.0, None)
    total = float(normalised.sum())
    if total <= 0:
        return normalised
    return normalised / total


class CeciledellavalleBlindDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        alpha: float = 2e-2,
        mu: float = 5e-2,
        gamma: float = 1.0,
        iterations: int = 50,
        kernel_size: int | Tuple[int, int] = 9,
        coeff_kernel: float = 1.0,
        coeff_image: float = 1.0,
        proj_simplex: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__('VioletBD')
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.iterations = int(iterations)
        self.kernel_size = self._coerce_kernel_size(kernel_size)
        self.coeff_kernel = float(coeff_kernel)
        self.coeff_image = float(coeff_image)
        self.proj_simplex = bool(proj_simplex)
        self.verbose = bool(verbose)
        self._last_kernel: Array2D | None = None

    def _coerce_kernel_size(self, size: int | Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
            seq = list(size)
            if len(seq) < 2:
                raise ValueError('kernel_size must provide two dimensions.')
            return (_ensure_odd(seq[0]), _ensure_odd(seq[1]))
        scalar = int(size)
        return (_ensure_odd(scalar), _ensure_odd(scalar))

    def _initial_kernel(self) -> Array2D:
        kx, ky = self.kernel_size
        kernel = np.zeros((kx, ky), dtype=np.float64)
        kernel[kx // 2, ky // 2] = 1.0
        return _normalise_kernel(kernel)

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'alpha' in param and param['alpha'] is not None:
                self.alpha = float(param['alpha'])
            if 'mu' in param and param['mu'] is not None:
                self.mu = float(param['mu'])
            if 'gamma' in param and param['gamma'] is not None:
                self.gamma = float(param['gamma'])
            if 'iterations' in param and param['iterations'] is not None:
                self.iterations = int(param['iterations'])
            if 'kernel_size' in param and param['kernel_size'] is not None:
                self.kernel_size = self._coerce_kernel_size(param['kernel_size'])
            if 'coeff_kernel' in param and param['coeff_kernel'] is not None:
                self.coeff_kernel = float(param['coeff_kernel'])
            if 'coeff_image' in param and param['coeff_image'] is not None:
                self.coeff_image = float(param['coeff_image'])
            if 'proj_simplex' in param and param['proj_simplex'] is not None:
                self.proj_simplex = bool(param['proj_simplex'])
            if 'verbose' in param and param['verbose'] is not None:
                self.verbose = bool(param['verbose'])
        return super().change_param(param)

    def get_param(self):
        return [
            ('alpha', self.alpha),
            ('mu', self.mu),
            ('gamma', self.gamma),
            ('iterations', self.iterations),
            ('kernel_size', self.kernel_size),
            ('coeff_kernel', self.coeff_kernel),
            ('coeff_image', self.coeff_image),
            ('proj_simplex', self.proj_simplex),
            ('verbose', self.verbose),
        ]

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        working, original_dtype, original_shape = _prepare_image(image)
        x_init = working.astype(np.float64, copy=True)
        kernel_init = self._initial_kernel()

        start = time()
        kernel, restored, _, _ = violetBD(
            x_init,
            kernel_init,
            working,
            self.alpha,
            self.mu,
            gamma=self.gamma,
            niter=self.iterations,
            coeffK=self.coeff_kernel,
            coeffx=self.coeff_image,
            proj_simplex=self.proj_simplex,
            verbose=self.verbose,
        )
        self.timer = time() - start

        kernel = _normalise_kernel(kernel)
        restored = np.clip(restored, 0.0, 1.0)
        self._last_kernel = kernel

        return _restore_dtype(restored, original_dtype, original_shape), kernel

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ["CeciledellavalleBlindDeconvolution"]

__all__ = ["CeciledellavalleBlindDeconvolution"]
