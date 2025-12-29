# https://github.com/ADY-YDA/Iterative-Blind-Image-Deconvolution/blob/main/Expectation-Maximization.ipynb
from __future__ import annotations

from time import time
from typing import Any, Optional, Tuple

import numpy as np
from scipy.signal import wiener

from algorithms.base import DeconvolutionAlgorithm


def _prepare_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype, Optional[int]]:
    original_dtype = image.dtype
    channels: Optional[int] = None

    if image.ndim == 3 and image.shape[2] != 1:
        channels = int(image.shape[2])
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float64, copy=False)
    if working.size and working.max() > 1.5:
        working = working / 255.0

    if working.ndim != 2:
        raise ValueError("Expectation-Maximisation requires a grayscale image (2-D array).")

    return working, original_dtype, channels


def _restore_dtype(restored: np.ndarray, original_dtype: np.dtype, channels: Optional[int]) -> np.ndarray:
    clipped = np.clip(restored, 0.0, 1.0)

    if np.issubdtype(original_dtype, np.integer):
        output = (clipped * 255.0).round().astype(original_dtype)
    else:
        output = clipped.astype(original_dtype, copy=False)

    if channels is not None and channels > 1:
        output = np.repeat(output[..., None], channels, axis=2)

    return output


def _covariance(image: np.ndarray) -> float:
    shifted = np.roll(image, shift=1, axis=0)
    a = image.reshape(-1)
    b = shifted.reshape(-1)
    cov = np.cov(a, b)[0, 1]
    if not np.isfinite(cov):
        return 0.0
    return float(cov)


def _normalize_kernel(kernel: np.ndarray, eps: float) -> np.ndarray:
    kernel = np.maximum(kernel, 0.0)
    total = float(kernel.sum())
    if total < eps:
        return kernel
    return kernel / total


def _center_crop(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    h, w = array.shape

    if target_h >= h and target_w >= w:
        return array

    start_y = max((h - target_h) // 2, 0)
    start_x = max((w - target_w) // 2, 0)
    end_y = start_y + min(target_h, h)
    end_x = start_x + min(target_w, w)
    cropped = array[start_y:end_y, start_x:end_x]

    if cropped.shape != shape:
        pad_y = max(target_h - cropped.shape[0], 0)
        pad_x = max(target_w - cropped.shape[1], 0)
        cropped = np.pad(
            cropped,
            ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
            mode="constant",
        )
    return cropped


def _estimate_psf(
    observed: np.ndarray,
    estimate: np.ndarray,
    noise_ratio: float,
    filter_size: int,
    eps: float,
) -> np.ndarray:
    size = observed.shape
    estimate_fft = np.fft.fft2(estimate, s=size)

    size_filter = max(3, int(round(filter_size)))
    if size_filter % 2 == 0:
        size_filter += 1
    filtered = wiener(estimate, mysize=size_filter, noise=noise_ratio)
    filtered_fft = np.fft.fft2(filtered, s=size)
    denom = np.sum(filtered_fft)
    if abs(denom) < eps:
        denom = eps
    filtered_fft = filtered_fft / denom

    log_spectrum = np.log10(1.0 + np.abs(np.fft.fftshift(estimate_fft)))
    variance = _covariance(log_spectrum)
    total = float(size[0] * size[1])
    energy = float(np.sum(np.abs(filtered_fft) ** 2))
    noise_variance = variance + (energy / total**2)
    denominator = total * noise_variance + eps

    H = (estimate_fft * np.conj(filtered_fft)) / denominator
    sum_h = np.sum(H)
    if abs(sum_h) < eps:
        sum_h = eps
    return H / sum_h


def _psf_from_frequency(psf_fft: np.ndarray, kernel_shape: Tuple[int, int], eps: float) -> np.ndarray:
    psf = np.real(np.fft.ifft2(psf_fft))
    psf = np.fft.ifftshift(psf)
    psf = _normalize_kernel(psf, eps)
    cropped = _center_crop(psf, kernel_shape)
    cropped = _normalize_kernel(cropped, eps)
    return cropped


def _stretch_to_unit(image: np.ndarray) -> np.ndarray:
    min_val = float(image.min())
    max_val = float(image.max())
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val - min_val < 1e-12:
        return np.clip(image, 0.0, 1.0)
    scaled = (image - min_val) / (max_val - min_val)
    return np.clip(scaled, 0.0, 1.0)


class ADYYDAIterativeBlindImageDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        iterations: int = 10,
        noise_ratio: float = 1e-3,
        filter_size: int = 5,
        epsilon: float = 1e-8,
        kernel_size: Tuple[int, int] = (25, 25),
        relative_tolerance: float = 1e-4,
    ) -> None:
        super().__init__('EMBlindDeconvolution')
        self.iterations = int(iterations)
        self.noise_ratio = float(noise_ratio)
        self.filter_size = int(filter_size)
        self.epsilon = float(epsilon)
        if isinstance(kernel_size, (tuple, list)):
            if len(kernel_size) < 2:
                raise ValueError('kernel_size must have at least two elements.')
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            size = int(kernel_size)
            self.kernel_size = (size, size)
        self.relative_tolerance = float(relative_tolerance)
        self._last_kernel: Optional[np.ndarray] = None

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'iterations' in param and param['iterations'] is not None:
                self.iterations = int(param['iterations'])
            if 'noise_ratio' in param and param['noise_ratio'] is not None:
                self.noise_ratio = float(param['noise_ratio'])
            if 'filter_size' in param and param['filter_size'] is not None:
                self.filter_size = int(param['filter_size'])
            if 'epsilon' in param and param['epsilon'] is not None:
                self.epsilon = float(param['epsilon'])
            if 'kernel_size' in param and param['kernel_size'] is not None:
                value = param['kernel_size']
                if isinstance(value, (tuple, list)) and len(value) >= 2:
                    self.kernel_size = (int(value[0]), int(value[1]))
                else:
                    size = int(value)
                    self.kernel_size = (size, size)
            if 'relative_tolerance' in param and param['relative_tolerance'] is not None:
                self.relative_tolerance = float(param['relative_tolerance'])
            return super().change_param(param)
        return super().change_param(param)


    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, original_dtype, channels = _prepare_image(image)
        estimate = working.copy()
        eps = float(self.epsilon)

        start = time()
        kernel_fft = np.ones_like(estimate)

        prev_estimate = estimate.copy()
        for iteration in range(max(1, int(self.iterations))):
            kernel_fft = _estimate_psf(working, estimate, self.noise_ratio, self.filter_size, eps)

            estimate_fft = np.fft.fft2(estimate)
            denominator = kernel_fft.copy()
            small = np.abs(denominator) < eps
            denominator[small] = eps
            decon_fft = estimate_fft / denominator
            estimate = np.real(np.fft.ifft2(decon_fft))
            estimate = _stretch_to_unit(estimate)

            delta = np.linalg.norm(estimate - prev_estimate) / (np.linalg.norm(prev_estimate) + eps)
            prev_estimate = estimate.copy()
            if delta < self.relative_tolerance:
                break

        kernel = _psf_from_frequency(kernel_fft, self.kernel_size, eps)
        self._last_kernel = kernel.astype(np.float32)
        self.timer = time() - start
        self._last_kernel = self._last_kernel*10000
        restored = _restore_dtype(estimate, original_dtype, channels)
        return restored, self._last_kernel

    def get_param(self):
        return [
            ('iterations', self.iterations),
            ('noise_ratio', self.noise_ratio),
            ('filter_size', self.filter_size),
            ('epsilon', self.epsilon),
            ('kernel_size', self.kernel_size),
            ('relative_tolerance', self.relative_tolerance),
        ]


__all__ = ['ADYYDAIterativeBlindImageDeconvolution']

__all__ = ["ADYYDAIterativeBlindImageDeconvolution"]
