"""
LuczengMotionBlur - Python Wrapper

Original Implementation:
    luczeng (GitHub)
    GitHub Repository: https://github.com/luczeng/MotionBlur
    Language/Framework: Python

Reference Paper (if applicable):
    Based on method described in the repository without published paper

Algorithm Description:
    - Motion blur kernel estimation (often parametric) with restoration loop
    - Outputs a restored image (and kernel estimate when available)
    - Outputs a restored image (and kernel estimate when available)

This Python Wrapper Provides:
    - DeconvolutionAlgorithm-compatible Python class
    - Bridges and calls bundled original implementation
    - Basic parameter coercion and sanity checks

Wrapper Features:
    - Works with NumPy arrays and common image dtypes
    - Integrates into the BlindDeconvolution algorithm registry
    - Keeps bundled original source code untouched

Example:
    >>> from my_wrappers import LuczengMotionBlur
    >>> processor = LuczengMotionBlur()
    >>> result = processor.process(input_image)

Important Notes:
    1. Requires: numpy
    2. Original Python code remains unchanged
    3. This is purely an interface wrapper
    4. Check original repository for license information

Author: AUTHOR_PROJECT
Wrapper Version: 1.0.0
Original Author: luczeng
Original License: Unknown (check repository)
"""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import sys

from algorithms.base import DeconvolutionAlgorithm

_SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from .source.motion_blur.libs.inverse_problems.wiener import Wiener
from .source.motion_blur.libs.forward_models.kernels.motion import motion_kernel


def _normalize_image(image: np.ndarray) -> tuple[np.ndarray, float]:
    if image.size == 0:
        return image.astype(np.float32, copy=False), 1.0

    peak = float(image.max())
    if peak == 0:
        return image.astype(np.float32, copy=False), 1.0

    if peak > 1.5:
        return (image / 255.0).astype(np.float32), 255.0

    return image.astype(np.float32, copy=False), 1.0


class LuczengMotionBlur(DeconvolutionAlgorithm):
    def __init__(
        self,
        *,
        length: int = 15,
        theta: float = 0.0,
        regularization: float = 1e-3,
        name: str = "WienerMotionBlur",
    ) -> None:
        super().__init__(name)
        self.length = self._ensure_length(length)
        self.theta = float(theta)
        self.regularization = float(regularization)
        self._kernel_override: np.ndarray | None = None

    @staticmethod
    def _ensure_length(length: int) -> int:
        length = max(int(length), 1)
        if length % 2 == 0:
            length += 1
        return length

    def change_param(self, param: Dict[str, Any] | Iterable[Tuple[str, Any]] | None):
        if not param:
            return super().change_param(param)

        if isinstance(param, dict):
            items = param.items()
        else:
            try:
                items = list(param)
            except TypeError:
                return super().change_param(param)

        for key, value in items:
            if key in {"length", "kernel_length"} and value is not None:
                self.length = self._ensure_length(int(value))
            elif key in {"theta", "angle"} and value is not None:
                self.theta = float(value)
            elif key in {"lambda", "regularization", "reg"} and value is not None:
                self.regularization = float(value)
            elif key == "kernel" and value is not None:
                array = np.asarray(value, dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError("Kernel override must be a 2-D array.")
                self._kernel_override = array
            elif key == "reset_kernel" and value:
                self._kernel_override = None

        return super().change_param(param)

    def _get_kernel(self) -> np.ndarray:
        if self._kernel_override is not None:
            return self._kernel_override
        kernel = motion_kernel(theta=self.theta % 180.0, L=self.length)
        return kernel.astype(np.float32, copy=False)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image.ndim > 2:
            working = image.squeeze()
            if working.ndim == 3:
                working = working.mean(axis=2)
        else:
            working = image

        working = np.asarray(working)
        original_dtype = working.dtype

        normalized, scale = _normalize_image(working)
        kernel = self._get_kernel()

        start = time()
        restored = Wiener(normalized, kernel, self.regularization)
        self.timer = time() - start

        restored = np.clip(restored, 0.0, 1.0)
        if scale != 1.0:
            restored = (restored * scale).clip(0.0, scale)

        if np.issubdtype(original_dtype, np.integer):
            restored_out = restored.round().astype(original_dtype)
        else:
            restored_out = restored.astype(original_dtype, copy=False)
        print(np.max(kernel))
        return restored_out, kernel

    def get_param(self):
        return [
            ("length", self.length),
            ("theta", self.theta),
            ("regularization", self.regularization),
            ("kernel_override", self._kernel_override is not None),
        ]

__all__ = ["LuczengMotionBlur"]
