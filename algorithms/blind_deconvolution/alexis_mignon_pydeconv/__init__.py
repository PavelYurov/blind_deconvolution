# https://github.com/alexis-mignon/pydeconv
from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Iterable

import numpy as np

from algorithms.base import DeconvolutionAlgorithm

from .source.pydeconv import objective


def _ensure_odd(value: int) -> int:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return size


def _coerce_kernel_size(kernel_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(kernel_size, tuple):
        if len(kernel_size) != 2:
            raise ValueError("kernel_size must be int or tuple[int, int].")
        return (_ensure_odd(kernel_size[0]), _ensure_odd(kernel_size[1]))
    return (_ensure_odd(kernel_size), _ensure_odd(kernel_size))


def _prepare_image(image: np.ndarray) -> tuple[np.ndarray, np.dtype, tuple[int, ...]]:
    original_dtype = image.dtype
    original_shape = image.shape

    if image.ndim == 2:
        working = image.astype(np.float64, copy=False)
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        working = image.astype(np.float64, copy=False)
    else:
        raise ValueError("Expected grayscale (H,W) or color (H,W,C) image.")

    if working.size == 0:
        raise ValueError("Empty image provided.")

    max_value = float(working.max())
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)
    return working, original_dtype, original_shape


def _restore_dtype(image01: np.ndarray, original_dtype: np.dtype, original_shape: tuple[int, ...]) -> np.ndarray:
    clipped = np.clip(image01, 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        restored = (clipped * 255.0).round().astype(original_dtype)
    else:
        restored = clipped.astype(original_dtype, copy=False)

    if restored.shape != original_shape:
        if len(original_shape) == 3 and original_shape[2] == 1 and restored.ndim == 2:
            restored = restored[..., None]
    return restored


def _initial_kernel(size: tuple[int, int]) -> np.ndarray:
    kx, ky = size
    kernel = np.zeros((kx, ky), dtype=np.float64)
    kernel[kx // 2, ky // 2] = 1.0
    return kernel


def _normalise_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.clip(kernel.astype(np.float64, copy=False), 0.0, None)
    total = float(kernel.sum())
    if total <= 0:
        return kernel
    return kernel / total


class AlexisMignonPydeconv(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 15,
        maxiter: int = 20,
        niter_latent: int = 3,
        niter_psf: int = 5,
        w0: float = 50.0,
        lambda1: float = 1e-3,
        lambda2: float = 10.0,
        a: float = 1.0,
        b: float = 3.0,
        t: float = 5.0,
        method: str = "gd",
        verbose: bool = False,
    ) -> None:
        super().__init__("alexis_mignon_pydeconv")
        self.kernel_size = _coerce_kernel_size(kernel_size)
        self.maxiter = int(maxiter)
        self.niter_latent = int(niter_latent)
        self.niter_psf = int(niter_psf)
        self.w0 = float(w0)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.a = float(a)
        self.b = float(b)
        self.t = float(t)
        self.method = str(method)
        self.verbose = bool(verbose)
        self._last_kernel: np.ndarray | None = None

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return None

        if "kernel_size" in param and param["kernel_size"] is not None:
            value = param["kernel_size"]
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                seq = list(value)
                self.kernel_size = _coerce_kernel_size((int(seq[0]), int(seq[1])))
            else:
                self.kernel_size = _coerce_kernel_size(int(value))

        for key in ("maxiter", "niter_latent", "niter_psf"):
            if key in param and param[key] is not None:
                setattr(self, key, int(param[key]))

        for key in ("w0", "lambda1", "lambda2", "a", "b", "t"):
            if key in param and param[key] is not None:
                setattr(self, key, float(param[key]))

        if "method" in param and param["method"] is not None:
            self.method = str(param["method"])
        if "verbose" in param and param["verbose"] is not None:
            self.verbose = bool(param["verbose"])

        return None

    def get_param(self):
        return [
            ("kernel_size", self.kernel_size),
            ("maxiter", self.maxiter),
            ("niter_latent", self.niter_latent),
            ("niter_psf", self.niter_psf),
            ("w0", self.w0),
            ("lambda1", self.lambda1),
            ("lambda2", self.lambda2),
            ("a", self.a),
            ("b", self.b),
            ("t", self.t),
            ("method", self.method),
            ("verbose", self.verbose),
        ]

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        working, original_dtype, original_shape = _prepare_image(image)
        if working.ndim == 3 and working.shape[2] == 4:
            working_rgb = working[..., :3]
            alpha = working[..., 3:4]
        else:
            working_rgb = working
            alpha = None

        psf0 = _initial_kernel(self.kernel_size)

        start = time()
        func = objective.ObjFunc(
            working_rgb,
            self.w0,
            self.lambda1,
            self.lambda2,
            self.a,
            self.b,
            t=self.t,
        )

        kernel = psf0
        latent = working_rgb
        for _ in range(self.maxiter):
            func.set_psf(kernel)
            latent = func.optimize_latent(
                latent,
                maxiter=self.niter_latent,
                method=self.method,
                verbose=self.verbose,
            )
            func.set_latent(latent)
            kernel = func.optimize_psf(
                kernel,
                maxiter=self.niter_psf,
                method=self.method,
                verbose=self.verbose,
                alpha_0=1e-5,
            )
        self.timer = time() - start

        kernel = _normalise_kernel(np.asarray(kernel, dtype=np.float64))
        self._last_kernel = kernel

        latent = np.asarray(latent, dtype=np.float64)
        if alpha is not None and latent.ndim == 3 and alpha.shape[:2] == latent.shape[:2]:
            latent = np.concatenate([latent, alpha], axis=2)

        return _restore_dtype(latent, original_dtype, original_shape), kernel

    def get_kernel(self) -> np.ndarray | None:
        return self._last_kernel


__all__ = ["AlexisMignonPydeconv"]
