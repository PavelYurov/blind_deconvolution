from __future__ import annotations

import math
from dataclasses import dataclass
from time import time
from typing import Any, Tuple

import numpy as np

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray


@dataclass
class _ImagePreparation:
    working: Array2D
    original_dtype: np.dtype
    original_shape: tuple[int, ...]


def _prepare_image(image: Array2D) -> _ImagePreparation:
    original_dtype = image.dtype
    original_shape = image.shape

    if image.ndim == 3 and image.shape[2] != 1:
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float64, copy=False)
    max_value = float(working.max()) if working.size else 0.0
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)

    if working.ndim != 2:
        raise ValueError('MaxMB Wiener filter expects a 2-D grayscale image.')

    return _ImagePreparation(working, original_dtype, original_shape)


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


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _centre_slice(total: int, size: int, use_round: bool = False) -> slice:
    if size > total:
        raise ValueError('Requested size exceeds padded workspace.')
    delta = (total - size) / 2.0
    start = _round_half_up(delta) if use_round else int(math.floor(delta))
    end = start + size
    return slice(start, end)


def _pad_to_square(image: Array2D, target: int) -> tuple[Array2D, slice, slice]:
    if image.shape[0] > target or image.shape[1] > target:
        raise ValueError('Target size must be at least as large as the image.')
    padded = np.zeros((target, target), dtype=np.float64)
    row_slice = _centre_slice(target, image.shape[0], use_round=False)
    col_slice = _centre_slice(target, image.shape[1], use_round=False)
    padded[row_slice, col_slice] = image
    return np.fft.fftshift(padded), row_slice, col_slice


def _embed_kernel(kernel: Array2D, target: int) -> Array2D:
    if kernel.shape[0] > target or kernel.shape[1] > target:
        raise ValueError('Kernel cannot be larger than the padded workspace.')
    padded = np.zeros((target, target), dtype=np.float64)
    row_slice = _centre_slice(target, kernel.shape[0], use_round=True)
    col_slice = _centre_slice(target, kernel.shape[1], use_round=True)
    padded[row_slice, col_slice] = kernel
    return np.fft.fftshift(padded)


def _next_power_of_two(value: int) -> int:
    value = int(value)
    if value <= 0:
        return 1
    return 1 << (value - 1).bit_length()


def _normalise_kernel(kernel: Array2D) -> Array2D:
    kernel = np.clip(kernel.astype(np.float64, copy=False), 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        return kernel
    return kernel / total


def _gaussian_kernel(size: Tuple[int, int], sigma: float) -> Array2D:
    rows, cols = size
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    diff_r = grid_r - center_r
    diff_c = grid_c - center_c
    kernel = np.exp(-(diff_r ** 2 + diff_c ** 2) / (2.0 * float(sigma) ** 2))
    return _normalise_kernel(kernel)


def _coerce_shape(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError('kernel_size must define two dimensions.')
        return max(1, int(value[0])), max(1, int(value[1]))
    size = max(1, int(value))
    return size, size


class MaxMBImageRestorationWienerBlind(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: int | Tuple[int, int] = 32,
        kernel_sigma: float = 3.0,
        snr_db: float = 10.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__('MaxMBWienerBlind')
        self.kernel_size = _coerce_shape(kernel_size)
        self.kernel_sigma = float(kernel_sigma)
        self.snr_db = float(snr_db)
        self.gamma = float(gamma)
        self._last_kernel: Array2D | None = None

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'kernel_size' in param and param['kernel_size'] is not None:
                self.kernel_size = _coerce_shape(param['kernel_size'])
            if 'kernel_sigma' in param and param['kernel_sigma'] is not None:
                self.kernel_sigma = float(param['kernel_sigma'])
            if 'snr_db' in param and param['snr_db'] is not None:
                self.snr_db = float(param['snr_db'])
            if 'gamma' in param and param['gamma'] is not None:
                self.gamma = float(param['gamma'])
        return super().change_param(param)

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('kernel_sigma', self.kernel_sigma),
            ('snr_db', self.snr_db),
            ('gamma', self.gamma),
        ]

    def process(self, image: Array2D) -> tuple[Array2D, Array2D]:
        start = time()
        prepared = _prepare_image(image)
        working = prepared.working

        kernel = _gaussian_kernel(self.kernel_size, self.kernel_sigma)
        pad_size = _next_power_of_two(max(working.shape[0], working.shape[1], kernel.shape[0], kernel.shape[1]))

        g_shifted, row_slice, col_slice = _pad_to_square(working, pad_size)
        G = np.fft.fft2(g_shifted)

        h_shifted = _embed_kernel(kernel, pad_size)
        H = np.fft.fft2(h_shifted)
        H2 = np.abs(H) ** 2

        k_ratio = 10.0 ** (-self.snr_db / 10.0)
        denom = H2 + self.gamma * k_ratio
        denom = np.where(denom == 0.0, np.finfo(np.float64).eps, denom)

        Fe = np.conj(H) * G / denom
        restored_shifted = np.fft.ifft2(Fe)
        restored_padded = np.fft.ifftshift(restored_shifted).real
        restored = restored_padded[row_slice, col_slice]
        if restored.size and restored.max() > 0.0:
            restored = restored / restored.max()

        restored = _restore_dtype(restored, prepared.original_dtype, prepared.original_shape)
        self._last_kernel = kernel
        self.timer = time() - start
        return restored, kernel

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ['MaxMBImageRestorationWienerBlind']

__all__ = ["MaxMBImageRestorationWienerBlind"]
