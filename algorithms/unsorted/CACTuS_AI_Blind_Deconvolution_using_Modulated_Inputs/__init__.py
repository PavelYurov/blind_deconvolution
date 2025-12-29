# https://github.com/CACTuS-AI/Blind-Deconvolution-using-Modulated-Inputs
from __future__ import annotations
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import correlate2d, fftconvolve

from algorithms.base import DeconvolutionAlgorithm

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        return np.full((size, size), 1.0 / (size * size), dtype=np.float64)
    return kernel / total


def _flip_kernel(kernel: np.ndarray) -> np.ndarray:
    return np.flip(kernel, axis=(0, 1))


def _laplacian(arr: np.ndarray) -> np.ndarray:
    return (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )


def _project_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        side = kernel.shape[0]
        return np.full_like(kernel, 1.0 / (side * side))
    return kernel / total


def _update_sharp(
    sharp: np.ndarray,
    kernel: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    step_size: float,
    epsilon: float,
    tv_weight: float,
) -> np.ndarray:
    modulated = mask * sharp
    predicted = fftconvolve(modulated, kernel, mode="same")
    residual = predicted - observed

    grad_data = mask * fftconvolve(residual, _flip_kernel(kernel), mode="same")
    grad_tv = tv_weight * _laplacian(sharp)
    updated = sharp - step_size * (grad_data + grad_tv)
    return np.clip(updated, 0.0, 1.0)


def _update_kernel(
    sharp: np.ndarray,
    kernel: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    step_size: float,
    epsilon: float,
    smooth_weight: float,
    kernel_size: int,
) -> np.ndarray:
    modulated = mask * sharp
    predicted = fftconvolve(modulated, kernel, mode="same")
    residual = predicted - observed

    grad_data = correlate2d(modulated, residual, mode="valid")
    grad_smooth = smooth_weight * _laplacian(kernel)
    grad = grad_data + grad_smooth
    updated = kernel - step_size * grad
    return _project_kernel(updated)


def _resize_kernel(kernel: np.ndarray, kernel_size: int) -> np.ndarray:
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


class CACTuSAIBlindDeconvolutionUsingModulatedInputs(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: int = 21,
        iterations: int = 20,
        image_step: float = 0.2,
        kernel_step: float = 0.05,
        epsilon: float = 1e-4,
        tv_weight: float = 0.001,
        kernel_smooth: float = 5e-4,
        init_sigma: Optional[float] = None,
        modulation_seed: Optional[int] = 1234,
        modulation_prob: float = 1.0,
    ) -> None:
        super().__init__('ModulatedInputsBlindDeconvolution')
        self.kernel_size = _ensure_odd(kernel_size)
        self.iterations = max(1, int(iterations))
        self.image_step = float(image_step)
        self.kernel_step = float(kernel_step)
        self.epsilon = float(epsilon)
        self.tv_weight = float(tv_weight)
        self.kernel_smooth = float(kernel_smooth)
        self.init_sigma = float(init_sigma) if init_sigma is not None else None
        self.modulation_seed = modulation_seed
        self.modulation_prob = float(np.clip(modulation_prob, 0.0, 1.0))
        self._last_kernel: Optional[np.ndarray] = None
        self._last_output: Optional[np.ndarray] = None

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
        if 'init_sigma' in param:
            value = param['init_sigma']
            self.init_sigma = None if value is None else float(value)
        if 'modulation_seed' in param:
            self.modulation_seed = None if param['modulation_seed'] is None else int(param['modulation_seed'])
        if 'modulation_prob' in param and param['modulation_prob'] is not None:
            self.modulation_prob = float(np.clip(param['modulation_prob'], 0.0, 1.0))
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
            ('init_sigma', self.init_sigma),
            ('modulation_seed', self.modulation_seed),
            ('modulation_prob', self.modulation_prob),
            ('last_kernel_shape', None if self._last_kernel is None else self._last_kernel.shape),
            ('last_output_shape', None if self._last_output is None else self._last_output.shape),
        ]

    def _make_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        if self.modulation_prob <= 0.0:
            return np.ones(shape, dtype=np.float64)
        rng = np.random.default_rng(self.modulation_seed)
        mask = rng.choice([-1.0, 1.0], size=shape, p=[0.5, 0.5])
        keep = rng.random(shape) < self.modulation_prob
        mask[~keep] = 1.0
        return mask

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        mask = self._make_mask(float_img.shape)

        start = time()
        for _ in range(self.iterations):
            sharp = _update_sharp(
                sharp,
                kernel,
                float_img,
                mask,
                self.image_step,
                self.epsilon,
                self.tv_weight,
            )
            kernel = _update_kernel(
                sharp,
                kernel,
                float_img,
                mask,
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

    def get_kernel(self) -> np.ndarray | None:
        return None if self._last_kernel is None else self._last_kernel.copy()


__all__ = ['CACTuSAIBlindDeconvolutionUsingModulatedInputs']
