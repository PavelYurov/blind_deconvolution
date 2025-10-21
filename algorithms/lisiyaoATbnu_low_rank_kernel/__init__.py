from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import correlate2d, fftconvolve

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _ensure_odd(value: int) -> int:
    value = int(value)
    return value | 1


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


def _svd_shrink(matrix: Array2D, lam: float) -> Array2D:
    if lam <= 0.0:
        return matrix
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s_shrink = np.maximum(s - lam, 0.0)
    return (u * s_shrink) @ vt


def _update_image(
    image: Array2D,
    kernel: Array2D,
    observed: Array2D,
    step_size: float,
    tv_weight: float,
) -> Array2D:
    residual = fftconvolve(image, kernel, mode="same") - observed
    grad_data = correlate2d(residual, _flip_kernel(kernel), mode="same")
    grad_tv = tv_weight * _laplacian(image)
    updated = image - step_size * (grad_data + grad_tv)
    return np.clip(updated, 0.0, 1.0)


def _update_kernel(
    image: Array2D,
    kernel: Array2D,
    observed: Array2D,
    step_size: float,
    smooth_weight: float,
    nuclear_weight: float,
    kernel_size: int,
) -> Array2D:
    residual = fftconvolve(image, kernel, mode="same") - observed
    grad_data = correlate2d(image, residual, mode="valid")
    grad_smooth = smooth_weight * _laplacian(kernel)
    updated = kernel - step_size * (grad_data + grad_smooth)
    updated = _svd_shrink(updated, nuclear_weight)
    updated = _project_kernel(updated)
    if updated.shape[0] != kernel_size:
        updated = _resize_kernel(updated, kernel_size)
    return updated


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


def _wiener_deconv(image: Array2D, kernel: Array2D, reg: float) -> Array2D:
    kernel = _project_kernel(kernel)
    kh, kw = kernel.shape
    pad = np.zeros_like(image)
    pad[:kh, :kw] = kernel
    pad = np.roll(pad, -kh // 2, axis=0)
    pad = np.roll(pad, -kw // 2, axis=1)

    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(pad)
    denom = np.abs(kernel_fft) ** 2 + reg
    restored_fft = image_fft * np.conj(kernel_fft) / denom
    restored = np.real(np.fft.ifft2(restored_fft))
    return np.clip(restored, 0.0, 1.0)


class LisiyaoATbnuLowRankKernel(DeconvolutionAlgorithm):
    """Blind deconvolution with low-rank kernel regularisation (Python port)."""

    def __init__(
        self,
        kernel_size: int = 21,
        iterations: int = 20,
        image_step: float = 0.2,
        kernel_step: float = 0.05,
        tv_weight: float = 0.002,
        kernel_smooth: float = 1e-3,
        nuclear_weight: float = 0.01,
        regularization: float = 1e-3,
        init_sigma: Optional[float] = None,
    ) -> None:
        super().__init__('LowRankKernelBlindDeconv')
        self.kernel_size = _ensure_odd(kernel_size)
        self.iterations = max(1, int(iterations))
        self.image_step = float(image_step)
        self.kernel_step = float(kernel_step)
        self.tv_weight = float(tv_weight)
        self.kernel_smooth = float(kernel_smooth)
        self.nuclear_weight = float(nuclear_weight)
        self.regularization = float(regularization)
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
        if 'tv_weight' in param and param['tv_weight'] is not None:
            self.tv_weight = float(param['tv_weight'])
        if 'kernel_smooth' in param and param['kernel_smooth'] is not None:
            self.kernel_smooth = float(param['kernel_smooth'])
        if 'nuclear_weight' in param and param['nuclear_weight'] is not None:
            self.nuclear_weight = float(param['nuclear_weight'])
        if 'regularization' in param and param['regularization'] is not None:
            self.regularization = float(param['regularization'])
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
            ('tv_weight', self.tv_weight),
            ('kernel_smooth', self.kernel_smooth),
            ('nuclear_weight', self.nuclear_weight),
            ('regularization', self.regularization),
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

        sigma = self.init_sigma if self.init_sigma is not None else max(1.0, self.kernel_size / 6.0)
        kernel = _gaussian_kernel(self.kernel_size, sigma)
        latent = float_img.copy()

        start = time()
        for _ in range(self.iterations):
            latent = _update_image(latent, kernel, float_img, self.image_step, self.tv_weight)
            kernel = _update_kernel(latent, kernel, float_img, self.kernel_step, self.kernel_smooth, self.nuclear_weight, self.kernel_size)

        restored = _wiener_deconv(float_img, kernel, self.regularization)
        self.timer = time() - start

        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(restored * 255.0, 0, 255).round().astype(original_dtype)
        else:
            restored = restored.astype(original_dtype, copy=False)

        self._last_kernel = kernel
        self._last_output = restored
        return restored, kernel

    def get_kernel(self) -> Array2D | None:
        return None if self._last_kernel is None else self._last_kernel.copy()


__all__ = ['LisiyaoATbnuLowRankKernel']
