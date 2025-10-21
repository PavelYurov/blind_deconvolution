from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _gaussian_kernel(size: int, sigma: float) -> Array2D:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        return np.full((size, size), 1.0 / (size * size), dtype=np.float64)
    return kernel / total


def _flip_kernel(kernel: Array2D) -> Array2D:
    return np.flip(kernel, axis=(0, 1))


def _laplacian(arr: Array2D) -> Array2D:
    return (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )


def _project_kernel(kernel: Array2D) -> Array2D:
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        side = kernel.shape[0]
        return np.full_like(kernel, 1.0 / (side * side))
    return kernel / total


def _compute_residual(sharp: Array2D, kernel: Array2D, observed: Array2D) -> Array2D:
    return fftconvolve(sharp, kernel, mode="same") - observed


def _robust_weight(residual: Array2D, epsilon: float) -> Array2D:
    return residual / np.maximum(np.sqrt(residual * residual + epsilon), epsilon)


def _extract_center(arr: Array2D, shape: Tuple[int, int]) -> Array2D:
    target_h, target_w = shape
    center_h = arr.shape[0] // 2
    center_w = arr.shape[1] // 2
    half_h = target_h // 2
    half_w = target_w // 2
    top = max(center_h - half_h, 0)
    left = max(center_w - half_w, 0)
    return arr[top : top + target_h, left : left + target_w]


def _low_rank_project(image: Array2D, rank: Optional[int], shrinkage: float) -> Array2D:
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    if shrinkage > 0.0:
        s = np.maximum(s - shrinkage, 0.0)
    if rank is not None and 0 < rank < len(s):
        s[rank:] = 0.0
    return (u * s) @ vt


def _update_sharp(
    sharp: Array2D,
    kernel: Array2D,
    observed: Array2D,
    step_size: float,
    epsilon: float,
    tv_weight: float,
    rank: Optional[int],
    shrinkage: float,
) -> Array2D:
    residual = _compute_residual(sharp, kernel, observed)
    weight = _robust_weight(residual, epsilon)
    grad_data = fftconvolve(weight, _flip_kernel(kernel), mode="same")
    grad_tv = tv_weight * _laplacian(sharp)
    updated = sharp - step_size * (grad_data + grad_tv)
    updated = _low_rank_project(updated, rank, shrinkage)
    return np.clip(updated, 0.0, 1.0)


def _update_kernel(
    sharp: Array2D,
    kernel: Array2D,
    observed: Array2D,
    step_size: float,
    epsilon: float,
    smooth_weight: float,
    kernel_size: int,
) -> Array2D:
    residual = _compute_residual(sharp, kernel, observed)
    weight = _robust_weight(residual, epsilon)
    grad_full = fftconvolve(weight, _flip_kernel(sharp), mode="same")
    grad_data = _extract_center(grad_full, kernel.shape)
    grad_smooth = smooth_weight * _laplacian(kernel)
    updated = kernel - step_size * (grad_data + grad_smooth)
    return _project_kernel(updated)


def _resize_kernel(kernel: Array2D, kernel_size: int) -> Array2D:
    if kernel.shape == (kernel_size, kernel_size):
        return kernel
    resized = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kh, kw = kernel.shape
    half_h = kernel_size // 2
    half_w = kernel_size // 2
    center_h = kh // 2
    center_w = kw // 2
    top = max(center_h - half_h, 0)
    left = max(center_w - half_w, 0)
    crop = kernel[top : top + kernel_size, left : left + kernel_size]
    resized[: crop.shape[0], : crop.shape[1]] = crop
    return _project_kernel(resized)


def _ensure_odd(value: int) -> int:
    return int(value) | 1


class CrewleaderBlindDeconvolutionLowRank(DeconvolutionAlgorithm):
    """Low-rank blind deconvolution via alternating minimisation (Python port)."""

    def __init__(
        self,
        kernel_size: int = 17,
        iterations: int = 15,
        image_step: float = 0.3,
        kernel_step: float = 0.05,
        epsilon: float = 1e-4,
        tv_weight: float = 0.002,
        kernel_smooth: float = 5e-4,
        low_rank: Optional[int] = 40,
        shrinkage: float = 0.01,
        init_sigma: Optional[float] = None,
    ) -> None:
        super().__init__('LowRankBlindDeconvolution')
        self.kernel_size = _ensure_odd(kernel_size)
        self.iterations = max(1, int(iterations))
        self.image_step = float(image_step)
        self.kernel_step = float(kernel_step)
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.kernel_smooth = float(kernel_smooth)
        self.low_rank = None if low_rank is None else max(1, int(low_rank))
        self.shrinkage = float(shrinkage)
        self.init_sigma = float(init_sigma) if init_sigma is not None else None
        self._last_kernel: Optional[Array2D] = None
        self._last_output: Optional[Array2D] = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = _ensure_odd(param['kernel_size'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = max(1, int(param['iterations']))
        if 'image_step' in param and param['image_step'] is not None:
            self.image_step = float(param['image_step'])
        if 'kernel_step' in param and param['kernel_step'] is not None:
            self.kernel_step = float(param['kernel_step'])
        if 'epsilon' in param and param['epsilon'] is not None:
            self.epsilon = float(param['epsilon'])
        if 'tv_weight' in param and param['tv_weight'] is not None:
            self.tv_weight = float(param['tv_weight'])
        if 'kernel_smooth' in param and param['kernel_smooth'] is not None:
            self.kernel_smooth = float(param['kernel_smooth'])
        if 'low_rank' in param:
            value = param['low_rank']
            self.low_rank = None if value is None else max(1, int(value))
        if 'shrinkage' in param and param['shrinkage'] is not None:
            self.shrinkage = float(param['shrinkage'])
        if 'init_sigma' in param:
            value = param['init_sigma']
            self.init_sigma = None if value is None else float(value)
        return super().change_param(param)

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('iterations', self.iterations),
            ('image_step', self.image_step),
            ('kernel_step', self.kernel_step),
            ('epsilon', self.epsilon),
            ('tv_weight', self.tv_weight),
            ('kernel_smooth', self.kernel_smooth),
            ('low_rank', self.low_rank),
            ('shrinkage', self.shrinkage),
            ('init_sigma', self.init_sigma),
            ('last_kernel_shape', None if self._last_kernel is None else self._last_kernel.shape),
            ('last_output_shape', None if self._last_output is None else self._last_output.shape),
        ]

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        if image is None:
            raise ValueError('Input image is None.')

        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError('Expected a 2D grayscale image.')

        original_dtype = arr.dtype
        float_img = arr.astype(np.float64, copy=False)
        if float_img.max() > 1.5:
            float_img = float_img / 255.0
        float_img = np.clip(float_img, 0.0, 1.0)

        sharp = float_img.copy()
        sigma = self.init_sigma if self.init_sigma is not None else max(1.0, self.kernel_size / 6.0)
        kernel = _gaussian_kernel(self.kernel_size, sigma)

        start = time()
        for _ in range(self.iterations):
            sharp = _update_sharp(
                sharp,
                kernel,
                float_img,
                self.image_step,
                self.epsilon,
                self.tv_weight,
                self.low_rank,
                self.shrinkage,
            )
            kernel = _update_kernel(
                sharp,
                kernel,
                float_img,
                self.kernel_step,
                self.epsilon,
                self.kernel_smooth,
                self.kernel_size,
            )
            kernel = _resize_kernel(kernel, self.kernel_size)
        self.timer = time() - start

        restored = sharp
        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(restored * 255.0, 0, 255).round().astype(original_dtype)
        else:
            restored = restored.astype(original_dtype, copy=False)

        self._last_kernel = kernel
        self._last_output = restored
        return restored, kernel

    def get_kernel(self) -> Array2D | None:
        return None if self._last_kernel is None else self._last_kernel.copy()


__all__ = ['CrewleaderBlindDeconvolutionLowRank']
