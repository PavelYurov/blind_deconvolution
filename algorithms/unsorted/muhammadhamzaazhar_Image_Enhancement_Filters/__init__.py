from __future__ import annotations

from time import time
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
from skimage.restoration import richardson_lucy, wiener

from ..base import DeconvolutionAlgorithm

KernelSpec = Union[int, Tuple[int, int], Iterable[int]]


def _normalize_kernel_shape(spec: KernelSpec) -> Tuple[int, int]:
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("Kernel size must be positive.")
        return (spec, spec)

    try:
        values = tuple(int(v) for v in spec)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - invalid types
        raise ValueError("Unsupported kernel size specification.") from exc

    if len(values) == 0:
        raise ValueError("Kernel size can not be empty.")
    if len(values) == 1:
        return _normalize_kernel_shape(values[0])

    h, w = values[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    return (h, w)


def _gaussian_kernel(shape: Tuple[int, int], sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive.")

    h, w = shape
    y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * float(sigma) ** 2))
    kernel_sum = float(kernel.sum())
    if kernel_sum == 0.0:
        raise ValueError("Gaussian kernel sum can not be zero.")
    kernel /= kernel_sum
    return kernel.astype(np.float32, copy=False)


def _prepare_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype, float]:
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or RGB image array.")

    original_dtype = image.dtype
    working = image.astype(np.float32, copy=False)
    scale = 255.0 if working.max(initial=0.0) > 1.5 else 1.0
    if scale != 1.0:
        working /= scale
    working = np.clip(working, 0.0, 1.0)
    return working, original_dtype, scale


def _restore_dtype(image: np.ndarray, dtype: np.dtype, scale: float) -> np.ndarray:
    clipped = np.clip(image, 0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        restored = (clipped * scale).round().astype(dtype)
    else:
        restored = clipped.astype(dtype, copy=False)
    return restored


class MuhammadhamzaazharImageEnhancementFilters(DeconvolutionAlgorithm):
    """Wiener deconvolution with a Gaussian point spread function."""

    def __init__(
        self,
        kernel_size: KernelSpec = 5,
        sigma: float = 1.5,
        balance: float = 0.01,
        clip: bool = True,
    ) -> None:
        super().__init__("WienerFilter")
        self.kernel_shape = _normalize_kernel_shape(kernel_size)
        self.sigma = float(sigma)
        self.balance = float(balance)
        self.clip = bool(clip)
        self._psf_cache: np.ndarray | None = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_size"])
            self._psf_cache = None
        if "kernel_shape" in param and param["kernel_shape"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_shape"])
            self._psf_cache = None
        if "sigma" in param and param["sigma"] is not None:
            self.sigma = float(param["sigma"])
            self._psf_cache = None
        if "balance" in param and param["balance"] is not None:
            self.balance = float(param["balance"])
        if "clip" in param and param["clip"] is not None:
            self.clip = bool(param["clip"])

        return super().change_param(param)

    def _psf(self) -> np.ndarray:
        if self._psf_cache is None:
            self._psf_cache = _gaussian_kernel(self.kernel_shape, self.sigma)
        return self._psf_cache

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, dtype, scale = _prepare_image(np.asarray(image))
        psf = self._psf()

        start = time()
        if working.ndim == 2:
            restored = wiener(working, psf, balance=self.balance, clip=self.clip)
        else:
            channels = [
                wiener(working[..., c], psf, balance=self.balance, clip=self.clip)
                for c in range(working.shape[2])
            ]
            restored = np.stack(channels, axis=2)
        self.timer = time() - start

        restored = restored.astype(np.float32, copy=False)
        output = _restore_dtype(restored, dtype, scale)
        return output, psf.copy()

    def get_param(self):
        return [
            ("kernel_shape", self.kernel_shape),
            ("sigma", self.sigma),
            ("balance", self.balance),
            ("clip", self.clip),
        ]


class MuhammadhamzaazharImageEnhancementFiltersRichardsonLucy(DeconvolutionAlgorithm):
    """Richardson-Lucy deconvolution with a Gaussian point spread function."""

    def __init__(
        self,
        kernel_size: KernelSpec = 5,
        sigma: float = 1.5,
        iterations: int = 20,
        clip: bool = True,
        filter_epsilon: float = 1e-7,
    ) -> None:
        super().__init__("RichardsonLucy")
        self.kernel_shape = _normalize_kernel_shape(kernel_size)
        self.sigma = float(sigma)
        self.iterations = int(iterations)
        self.clip = bool(clip)
        self.filter_epsilon = float(filter_epsilon)
        self._psf_cache: np.ndarray | None = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_size"])
            self._psf_cache = None
        if "kernel_shape" in param and param["kernel_shape"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_shape"])
            self._psf_cache = None
        if "sigma" in param and param["sigma"] is not None:
            self.sigma = float(param["sigma"])
            self._psf_cache = None
        if "iterations" in param and param["iterations"] is not None:
            self.iterations = int(param["iterations"])
        if "clip" in param and param["clip"] is not None:
            self.clip = bool(param["clip"])
        if "filter_epsilon" in param and param["filter_epsilon"] is not None:
            self.filter_epsilon = float(param["filter_epsilon"])

        return super().change_param(param)

    def _psf(self) -> np.ndarray:
        if self._psf_cache is None:
            self._psf_cache = _gaussian_kernel(self.kernel_shape, self.sigma)
        return self._psf_cache

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, dtype, scale = _prepare_image(np.asarray(image))
        psf = self._psf()

        start = time()
        if working.ndim == 2:
            restored = richardson_lucy(
                working,
                psf,
                self.iterations,
                clip=self.clip,
                filter_epsilon=self.filter_epsilon,
            )
        else:
            channels = [
                richardson_lucy(
                    working[..., c],
                    psf,
                    self.iterations,
                    clip=self.clip,
                    filter_epsilon=self.filter_epsilon,
                )
                for c in range(working.shape[2])
            ]
            restored = np.stack(channels, axis=2)
        self.timer = time() - start

        restored = restored.astype(np.float32, copy=False)
        output = _restore_dtype(restored, dtype, scale)
        return output, psf.copy()

    def get_param(self):
        return [
            ("kernel_shape", self.kernel_shape),
            ("sigma", self.sigma),
            ("iterations", self.iterations),
            ("clip", self.clip),
            ("filter_epsilon", self.filter_epsilon),
        ]

__all__ = ["MuhammadhamzaazharImageEnhancementFilters", "MuhammadhamzaazharImageEnhancementFiltersRichardsonLucy"]
