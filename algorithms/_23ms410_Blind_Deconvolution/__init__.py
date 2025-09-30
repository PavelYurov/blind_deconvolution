from __future__ import annotations

from time import time
from typing import Any, Iterable, Tuple

import numpy as np

from ..base import DeconvolutionAlgorithm
from .source.admm_numpy import blind_deconvolution_admm

Array2D = np.ndarray


def _ensure_odd(size: int) -> int:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return size


def _prepare_image(image: Array2D) -> tuple[Array2D, np.dtype, tuple[int, ...]]:
    original_dtype = image.dtype
    original_shape = image.shape

    if image.ndim == 3 and image.shape[2] != 1:
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float64, copy=False)
    if working.size == 0:
        raise ValueError('Empty image provided to _23ms410BlindDeconvolution.')

    max_value = float(working.max())
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)

    if working.ndim != 2:
        raise ValueError('_23ms410BlindDeconvolution expects a 2-D grayscale image after preprocessing.')

    return working, original_dtype, original_shape


def _restore_dtype(image: Array2D, original_dtype: np.dtype, original_shape: tuple[int, ...]) -> Array2D:
    image = np.clip(image, 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        restored = (image * 255.0).round().astype(original_dtype)
    else:
        restored = image.astype(original_dtype, copy=False)
    if len(original_shape) == 3:
        channels = original_shape[2]
        if channels == 1:
            restored = restored[..., None]
        else:
            restored = np.repeat(restored[..., None], channels, axis=2)
    return restored


class _23ms410BlindDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        iterations: int = 8,
        kernel_size: int | Tuple[int, int] = 15,
        lambda_tv: float = 0.01,
        rho: float = 2.0,
        admm_iters: int = 25,
        epsilon: float = 1e-3,
        weight_strength: float = 5.0,
    ) -> None:
        super().__init__('ADMM23ms410')
        self.iterations = int(iterations)
        self.kernel_size = self._coerce_kernel_size(kernel_size)
        self.lambda_tv = float(lambda_tv)
        self.rho = float(rho)
        self.admm_iters = int(admm_iters)
        self.epsilon = float(epsilon)
        self.weight_strength = float(weight_strength)
        self._last_kernel: Array2D | None = None

    def _coerce_kernel_size(self, size: int | Tuple[int, int]) -> Tuple[int, int]:
        if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
            seq = list(size)
            if len(seq) < 2:
                raise ValueError('kernel_size must provide two dimensions.')
            return (_ensure_odd(seq[0]), _ensure_odd(seq[1]))
        scalar = int(size)
        return (_ensure_odd(scalar), _ensure_odd(scalar))

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'iterations' in param and param['iterations'] is not None:
                self.iterations = int(param['iterations'])
            if 'kernel_size' in param and param['kernel_size'] is not None:
                self.kernel_size = self._coerce_kernel_size(param['kernel_size'])
            if 'lambda_tv' in param and param['lambda_tv'] is not None:
                self.lambda_tv = float(param['lambda_tv'])
            if 'rho' in param and param['rho'] is not None:
                self.rho = float(param['rho'])
            if 'admm_iters' in param and param['admm_iters'] is not None:
                self.admm_iters = int(param['admm_iters'])
            if 'epsilon' in param and param['epsilon'] is not None:
                self.epsilon = float(param['epsilon'])
            if 'weight_strength' in param and param['weight_strength'] is not None:
                self.weight_strength = float(param['weight_strength'])
        return super().change_param(param)

    def get_param(self):
        return [
            ('iterations', self.iterations),
            ('kernel_size', self.kernel_size),
            ('lambda_tv', self.lambda_tv),
            ('rho', self.rho),
            ('admm_iters', self.admm_iters),
            ('epsilon', self.epsilon),
            ('weight_strength', self.weight_strength),
        ]

    def process(self, image: Array2D) -> tuple[Array2D, Array2D]:
        working, original_dtype, original_shape = _prepare_image(image)
        start = time()
        latent, kernel = blind_deconvolution_admm(
            working,
            self.kernel_size,
            max(1, self.iterations),
            max(1e-6, self.lambda_tv),
            max(1e-6, self.rho),
            max(1, self.admm_iters),
            max(1e-8, self.epsilon),
            max(0.0, self.weight_strength),
        )
        self.timer = time() - start
        self._last_kernel = kernel
        restored = _restore_dtype(latent, original_dtype, original_shape)
        return restored, kernel

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ['_23ms410BlindDeconvolution']

__all__ = ["_23ms410BlindDeconvolution"]
