from __future__ import annotations

from time import time
from typing import Any, Iterable, Tuple

import numpy as np

from algorithms.base import DeconvolutionAlgorithm
from .source.libs.deconv import PSFest, deconv

ArrayND = np.ndarray
Array2D = np.ndarray


def _ensure_odd(value: int) -> int:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return size


def _coerce_kernel_size(kernel_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(kernel_size, tuple):
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be int or tuple[int, int].")
        return (_ensure_odd(kernel_size[0]), _ensure_odd(kernel_size[1]))
    return (_ensure_odd(kernel_size), _ensure_odd(kernel_size))


def _normalise_kernel(kernel: Array2D) -> Array2D:
    kernel = np.clip(kernel.astype(np.float64, copy=False), 0.0, None)
    total = float(kernel.sum())
    if total <= 0:
        return kernel
    return kernel / total


def _prepare_image(image: ArrayND) -> Tuple[ArrayND, np.dtype, Tuple[int, ...], bool]:
    original_dtype = image.dtype
    original_shape = image.shape

    if image.ndim == 2:
        working = image.astype(np.float64, copy=False)
        is_color = False
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        is_color = image.shape[2] != 1
        working = image.astype(np.float64, copy=False)
    else:
        raise ValueError("Expected grayscale (H,W) or color (H,W,C) image.")

    if working.size == 0:
        raise ValueError("Empty image provided.")

    max_value = float(working.max())
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)

    return working, original_dtype, original_shape, is_color


def _restore_dtype(image01: ArrayND, original_dtype: np.dtype, original_shape: Tuple[int, ...]) -> ArrayND:
    clipped = np.clip(image01, 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        restored = (clipped * 255.0).round().astype(original_dtype)
    else:
        restored = clipped.astype(original_dtype, copy=False)

    if restored.shape != original_shape:
        if len(original_shape) == 3 and original_shape[2] == 1 and restored.ndim == 2:
            restored = restored[..., None]
    return restored


class YenhsunlinBlindDeconv(DeconvolutionAlgorithm):
    """
    Обёртка над экспериментальным Bayesian blind deconvolution.

    Источник: https://github.com/yenhsunlin/blind_deconv
    """

    def __init__(
        self,
        kernel_size: int | Tuple[int, int] = 15,
        max_it: int = 5,
        max_it_u: int = 5,
        max_it_h: int = 5,
        gamma: float = 3e2,
        Lp: float = 0.0,
        ccreltol: float = 1e-3,
        return_nonblind_deconv: bool = True,
    ) -> None:
        super().__init__("yenhsunlin_blind_deconv")
        self.kernel_size = _coerce_kernel_size(kernel_size)
        self.max_it = int(max_it)
        self.max_it_u = int(max_it_u)
        self.max_it_h = int(max_it_h)
        self.gamma = float(gamma)
        self.Lp = float(Lp)
        self.ccreltol = float(ccreltol)
        self.return_nonblind_deconv = bool(return_nonblind_deconv)
        self._last_kernel: Array2D | None = None

    def _initial_kernel(self) -> Array2D:
        kx, ky = self.kernel_size
        kernel = np.zeros((kx, ky), dtype=np.float64)
        kernel[kx // 2, ky // 2] = 1.0
        return kernel

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if "kernel_size" in param and param["kernel_size"] is not None:
                value = param["kernel_size"]
                if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                    seq = list(value)
                    self.kernel_size = _coerce_kernel_size((seq[0], seq[1]))
                else:
                    self.kernel_size = _coerce_kernel_size(int(value))
            if "max_it" in param and param["max_it"] is not None:
                self.max_it = int(param["max_it"])
            if "max_it_u" in param and param["max_it_u"] is not None:
                self.max_it_u = int(param["max_it_u"])
            if "max_it_h" in param and param["max_it_h"] is not None:
                self.max_it_h = int(param["max_it_h"])
            if "gamma" in param and param["gamma"] is not None:
                self.gamma = float(param["gamma"])
            if "Lp" in param and param["Lp"] is not None:
                self.Lp = float(param["Lp"])
            if "ccreltol" in param and param["ccreltol"] is not None:
                self.ccreltol = float(param["ccreltol"])
            if "return_nonblind_deconv" in param and param["return_nonblind_deconv"] is not None:
                self.return_nonblind_deconv = bool(param["return_nonblind_deconv"])
        return None

    def get_param(self):
        return [
            ("kernel_size", self.kernel_size),
            ("max_it", self.max_it),
            ("max_it_u", self.max_it_u),
            ("max_it_h", self.max_it_h),
            ("gamma", self.gamma),
            ("Lp", self.Lp),
            ("ccreltol", self.ccreltol),
            ("return_nonblind_deconv", self.return_nonblind_deconv),
        ]

    def process(self, image: ArrayND) -> tuple[ArrayND, Array2D]:
        working, original_dtype, original_shape, is_color = _prepare_image(image)

        kernel_init = self._initial_kernel()

        if is_color:
            gray = working[..., :3].mean(axis=2)
        else:
            gray = np.squeeze(working)

        start = time()
        _, kernel = PSFest(
            gray,
            kernel_init,
            max_it=self.max_it,
            max_it_u=self.max_it_u,
            max_it_h=self.max_it_h,
            gamma=self.gamma,
            Lp=self.Lp,
            ccreltol=self.ccreltol,
        )
        kernel = _normalise_kernel(kernel)
        self._last_kernel = kernel

        if not self.return_nonblind_deconv:
            self.timer = time() - start
            restored01 = gray
            if is_color:
                restored01 = np.repeat(restored01[..., None], original_shape[2], axis=2)
            return _restore_dtype(restored01, original_dtype, original_shape), kernel

        if is_color:
            restored_channels = []
            for channel_index in range(min(3, working.shape[2])):
                restored_channels.append(
                    deconv(
                        working[..., channel_index],
                        kernel,
                        max_it_u=self.max_it,
                        gamma=self.gamma,
                        ccreltoltol=self.ccreltol,
                    )
                )
            restored01 = np.stack(restored_channels, axis=2)
            if working.shape[2] == 4:
                restored01 = np.concatenate([restored01, working[..., 3:4]], axis=2)
        else:
            restored01 = deconv(
                gray,
                kernel,
                max_it_u=self.max_it,
                gamma=self.gamma,
                ccreltoltol=self.ccreltol,
            )

        self.timer = time() - start
        return _restore_dtype(restored01, original_dtype, original_shape), kernel

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ["YenhsunlinBlindDeconv"]
