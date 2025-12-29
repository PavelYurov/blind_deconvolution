# https://github.com/jeffreysblake/funsearch-blind-deconvolution
from __future__ import annotations
from dataclasses import dataclass
from time import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage

from algorithms.base import DeconvolutionAlgorithm

from .source.blind_deconvolution.deconvolution_utils import (
    lucy_richardson_iteration,
    compute_image_gradient_norm,
    compute_residual_norm,
)


KernelSpec = Union[int, Sequence[int], np.ndarray]


def _coerce_kernel_shape(size: KernelSpec) -> Tuple[int, int]:
    if isinstance(size, np.ndarray):
        if size.ndim != 2:
            raise ValueError("Custom kernel must be 2-D.")
        return size.shape
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("Kernel size must be positive.")
        return (size, size)
    values = tuple(int(v) for v in size)  # type: ignore[arg-type]
    if len(values) == 0:
        raise ValueError("Kernel size specification can not be empty.")
    if len(values) == 1:
        return _coerce_kernel_shape(values[0])
    h, w = values[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    return (h, w)


def _gaussian_kernel(size: Tuple[int, int], sigma: float) -> np.ndarray:
    y = np.arange(size[0]) - (size[0] - 1) / 2.0
    x = np.arange(size[1]) - (size[1] - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = np.clip(kernel, 0.0, None)
    kernel /= kernel.sum() + 1e-12
    return kernel.astype(np.float32)


def _motion_kernel(size: Tuple[int, int], angle: float) -> np.ndarray:
    kernel = np.zeros(size, dtype=np.float32)
    center_y = size[0] // 2
    kernel[center_y, :] = 1.0
    rotated = np.asarray(ndimage.rotate(kernel, angle=angle, reshape=False))
    rotated = np.clip(rotated, 0.0, None)
    total = rotated.sum()
    if total > 0:
        rotated /= total
    else:
        rotated[center_y, size[1] // 2] = 1.0
    return rotated.astype(np.float32)


def _prepare_image(image: np.ndarray) -> Tuple[np.ndarray, bool, float]:
    scale = 255.0 if image.max(initial=0.0) > 1.5 else 1.0
    arr = image.astype(np.float32, copy=False) / scale
    needs_squeeze = False
    if arr.ndim == 2:
        arr = arr[None, ...]
        needs_squeeze = True
    elif arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        raise ValueError("Unsupported image dimensionality for Lucy-Richardson.")
    return arr, needs_squeeze, scale


def _restore_image(stack: np.ndarray, needs_squeeze: bool, original_shape: Tuple[int, ...]) -> np.ndarray:
    if needs_squeeze:
        restored = stack[0]
    else:
        restored = np.transpose(stack, (1, 2, 0))
        if original_shape[-1] == 1:
            restored = restored[..., 0]
    return restored


def _pad_kernel(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float32)
    kh, kw = kernel.shape
    start_y = (shape[0] - kh) // 2
    start_x = (shape[1] - kw) // 2
    padded[start_y:start_y + kh, start_x:start_x + kw] = kernel
    return padded


@dataclass
class _KernelConfig:
    mode: str
    kernel_size: Tuple[int, int]
    sigma: float
    angle: float
    custom_kernel: Optional[np.ndarray] = None

    def build(self) -> np.ndarray:
        if self.custom_kernel is not None:
            kernel = np.asarray(self.custom_kernel, dtype=np.float32)
            kernel = np.clip(kernel, 0.0, None)
            total = float(kernel.sum())
            if total == 0:
                raise ValueError("Custom kernel must have non-zero sum.")
            return kernel / total
        if self.mode == 'gaussian':
            return _gaussian_kernel(self.kernel_size, self.sigma)
        if self.mode == 'motion':
            return _motion_kernel(self.kernel_size, self.angle)
        raise ValueError(f"Unsupported kernel mode '{self.mode}'.")


class JeffreysblakeFunsearchBlindDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: KernelSpec = 15,
        kernel_mode: str = 'gaussian',
        sigma: float = 3.0,
        angle: float = 0.0,
        max_iterations: int = 60,
        min_iterations: int = 5,
        residual_threshold: float = 5e-4,
        gradient_threshold: float = 0.05,
        stagnation_window: int = 5,
        stagnation_delta: float = 1e-4,
        clip: bool = True,
        kernel: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__('FunsearchRLStopping')
        kernel_shape = _coerce_kernel_shape(kernel if kernel is not None else kernel_size)
        self.config = _KernelConfig(
            mode=str(kernel_mode).lower(),
            kernel_size=kernel_shape,
            sigma=float(sigma),
            angle=float(angle),
            custom_kernel=None if kernel is None else np.asarray(kernel, dtype=np.float32),
        )
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if min_iterations < 0:
            raise ValueError("min_iterations must be non-negative")
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.residual_threshold = float(residual_threshold)
        self.gradient_threshold = float(gradient_threshold)
        self.stagnation_window = max(1, int(stagnation_window))
        self.stagnation_delta = float(stagnation_delta)
        self.clip = bool(clip)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.config.kernel_size = _coerce_kernel_shape(param['kernel_size'])
        if 'kernel_mode' in param and param['kernel_mode']:
            self.config.mode = str(param['kernel_mode']).lower()
        if 'sigma' in param and param['sigma'] is not None:
            self.config.sigma = float(param['sigma'])
        if 'angle' in param and param['angle'] is not None:
            self.config.angle = float(param['angle'])
        if 'kernel' in param and param['kernel'] is not None:
            self.config.custom_kernel = np.asarray(param['kernel'], dtype=np.float32)
        if 'max_iterations' in param and param['max_iterations'] is not None:
            value = int(param['max_iterations'])
            if value <= 0:
                raise ValueError('max_iterations must be positive')
            self.max_iterations = value
        if 'min_iterations' in param and param['min_iterations'] is not None:
            value = int(param['min_iterations'])
            if value < 0:
                raise ValueError('min_iterations must be non-negative')
            self.min_iterations = value
        if 'residual_threshold' in param and param['residual_threshold'] is not None:
            self.residual_threshold = float(param['residual_threshold'])
        if 'gradient_threshold' in param and param['gradient_threshold'] is not None:
            self.gradient_threshold = float(param['gradient_threshold'])
        if 'stagnation_window' in param and param['stagnation_window'] is not None:
            self.stagnation_window = max(1, int(param['stagnation_window']))
        if 'stagnation_delta' in param and param['stagnation_delta'] is not None:
            self.stagnation_delta = float(param['stagnation_delta'])
        if 'clip' in param and param['clip'] is not None:
            self.clip = bool(param['clip'])

        return super().change_param(param)

    def process(self, image: np.ndarray):
        start = time()
        channels, needs_squeeze, scale = _prepare_image(image)
        kernel = self.config.build()
        padded_kernel = _pad_kernel(kernel, channels.shape[1:])

        restored_channels = []
        for observed in channels:
            estimate = observed.copy()
            residual_history = []
            gradient_history = []

            for iteration in range(self.max_iterations):
                updated = lucy_richardson_iteration(estimate, observed, padded_kernel)
                if self.clip:
                    updated = np.clip(updated, 0.0, 1.0)

                residual = float(compute_residual_norm(updated, estimate))
                gradient = float(compute_image_gradient_norm(updated))
                residual_history.append(residual)
                gradient_history.append(gradient)

                estimate = updated

                if iteration + 1 < self.min_iterations:
                    continue

                stagnated = False
                if len(residual_history) >= self.stagnation_window:
                    recent = residual_history[-self.stagnation_window:]
                    if max(recent) - min(recent) < self.stagnation_delta:
                        stagnated = True

                if (
                    residual <= self.residual_threshold
                    and gradient <= self.gradient_threshold
                ) or stagnated:
                    break

            restored_channels.append(estimate)

        restored_stack = np.stack(restored_channels, axis=0)
        restored = _restore_image(restored_stack, needs_squeeze, image.shape)
        restored = np.clip(restored, 0.0, 1.0)

        if np.issubdtype(image.dtype, np.integer):
            restored_out = (restored * scale).round().astype(image.dtype)
        else:
            restored_out = (restored * scale).astype(image.dtype, copy=False)

        self.timer = time() - start
        print(np.sum(kernel))
        return restored_out, kernel

    def get_param(self):
        return [
            ('kernel_size', self.config.kernel_size),
            ('kernel_mode', self.config.mode),
            ('sigma', self.config.sigma),
            ('angle', self.config.angle),
            ('max_iterations', self.max_iterations),
            ('min_iterations', self.min_iterations),
            ('residual_threshold', self.residual_threshold),
            ('gradient_threshold', self.gradient_threshold),
            ('stagnation_window', self.stagnation_window),
            ('stagnation_delta', self.stagnation_delta),
            ('clip', self.clip),
        ]

__all__ = ["_KernelConfig", "JeffreysblakeFunsearchBlindDeconvolution"]
