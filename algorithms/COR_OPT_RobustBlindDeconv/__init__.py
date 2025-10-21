from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import fftconvolve, correlate2d

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _gaussian_kernel(size: int, sigma: float) -> Array2D:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel[kernel < 0] = 0.0
    kernel_sum = kernel.sum()
    if kernel_sum <= 0:
        return np.ones((size, size), dtype=np.float64) / (size * size)
    return kernel / kernel_sum


def _flip_kernel(kernel: Array2D) -> Array2D:
    return np.flip(kernel, axis=(0, 1))


def _laplacian(arr: Array2D) -> Array2D:
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    return lap


def _project_kernel(kernel: Array2D) -> Array2D:
    kernel = np.clip(kernel, 0.0, None)
    total = kernel.sum()
    if total <= 0:
        side = kernel.shape[0]
        return np.ones_like(kernel) / (side * side)
    return kernel / total


def _pad_kernel_like(kernel: Array2D, shape: Tuple[int, int]) -> Array2D:
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = kernel.shape
    ph, pw = (shape[0] - kh) // 2, (shape[1] - kw) // 2
    padded[ph : ph + kh, pw : pw + kw] = kernel
    return padded


def _compute_residual(sharp: Array2D, kernel: Array2D, observed: Array2D) -> Array2D:
    return fftconvolve(sharp, kernel, mode="same") - observed


def _robust_weight(residual: Array2D, epsilon: float) -> Array2D:
    denom = np.sqrt(residual * residual + epsilon)
    return residual / np.maximum(denom, epsilon)


def _update_sharp(
    sharp: Array2D,
    kernel: Array2D,
    observed: Array2D,
    step_size: float,
    epsilon: float,
    tv_weight: float,
) -> Array2D:
    residual = _compute_residual(sharp, kernel, observed)
    weight = _robust_weight(residual, epsilon)
    grad_data = fftconvolve(weight, _flip_kernel(kernel), mode="same")
    grad_tv = tv_weight * _laplacian(sharp)
    updated = sharp - step_size * (grad_data + grad_tv)
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
    flipped_sharp = _flip_kernel(sharp)
    grad_full = fftconvolve(weight, flipped_sharp, mode='same')
    center_h = grad_full.shape[0] // 2
    center_w = grad_full.shape[1] // 2
    half = kernel_size // 2
    top = max(center_h - half, 0)
    left = max(center_w - half, 0)
    grad_data = grad_full[top:top + kernel_size, left:left + kernel_size]
    if grad_data.shape != kernel.shape:
        resized = np.zeros_like(kernel)
        sh, sw = grad_data.shape
        resized[:sh, :sw] = grad_data
        grad_data = resized
    grad_smooth = smooth_weight * _laplacian(kernel)
    grad = grad_data + grad_smooth
    updated = kernel - step_size * grad
    return _project_kernel(updated)


def _resize_kernel(kernel: Array2D, kernel_size: int) -> Array2D:
    kh, kw = kernel.shape
    if kh == kernel_size and kw == kernel_size:
        return kernel
    center = (kh // 2, kw // 2)
    offset_h = kernel_size // 2
    offset_w = kernel_size // 2
    top = max(center[0] - offset_h, 0)
    left = max(center[1] - offset_w, 0)
    bottom = min(top + kernel_size, kh)
    right = min(left + kernel_size, kw)
    cropped = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    ch, cw = bottom - top, right - left
    cropped[:ch, :cw] = kernel[top:bottom, left:right]
    return _project_kernel(cropped)


class COROPTRobustBlindDeconv(DeconvolutionAlgorithm):
    """Robust blind deconvolution via alternating robust updates (Python port)."""

    def __init__(
        self,
        kernel_size: int = 15,
        iterations: int = 25,
        image_step: float = 0.4,
        kernel_step: float = 0.1,
        epsilon: float = 1e-4,
        tv_weight: float = 0.005,
        kernel_smooth: float = 0.001,
        init_sigma: Optional[float] = None,
    ) -> None:
        super().__init__("RobustBlindDeconv")
        self.kernel_size = max(3, int(kernel_size) | 1)
        self.iterations = max(1, int(iterations))
        self.image_step = float(image_step)
        self.kernel_step = float(kernel_step)
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.kernel_smooth = float(kernel_smooth)
        self.init_sigma = float(init_sigma) if init_sigma is not None else None
        self._last_kernel: Optional[Array2D] = None
        self._last_output: Optional[Array2D] = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_size = max(3, int(param["kernel_size"]) | 1)
        if "iterations" in param and param["iterations"] is not None:
            self.iterations = max(1, int(param["iterations"]))
        if "image_step" in param and param["image_step"] is not None:
            self.image_step = float(param["image_step"])
        if "kernel_step" in param and param["kernel_step"] is not None:
            self.kernel_step = float(param["kernel_step"])
        if "epsilon" in param and param["epsilon"] is not None:
            self.epsilon = float(param["epsilon"])
        if "tv_weight" in param and param["tv_weight"] is not None:
            self.tv_weight = float(param["tv_weight"])
        if "kernel_smooth" in param and param["kernel_smooth"] is not None:
            self.kernel_smooth = float(param["kernel_smooth"])
        if "init_sigma" in param:
            value = param["init_sigma"]
            self.init_sigma = None if value is None else float(value)
        return super().change_param(param)

    def get_param(self):
        return [
            ("kernel_size", self.kernel_size),
            ("iterations", self.iterations),
            ("image_step", self.image_step),
            ("kernel_step", self.kernel_step),
            ("epsilon", self.epsilon),
            ("tv_weight", self.tv_weight),
            ("kernel_smooth", self.kernel_smooth),
            ("init_sigma", self.init_sigma),
            ("last_kernel_shape", None if self._last_kernel is None else self._last_kernel.shape),
            ("last_output_shape", None if self._last_output is None else self._last_output.shape),
        ]

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        if image is None:
            raise ValueError("Input image is None.")
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError("Expected a 2D grayscale image.")

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
            sharp = _update_sharp(sharp, kernel, float_img, self.image_step, self.epsilon, self.tv_weight)
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

        self._last_output = restored
        self._last_kernel = kernel
        return restored, kernel

    def get_kernel(self) -> Array2D | None:
        return None if self._last_kernel is None else self._last_kernel.copy()


__all__ = ["COROPTRobustBlindDeconv"]
