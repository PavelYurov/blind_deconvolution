# https://github.com/idiap/semiblindpsfdeconv
from __future__ import annotations
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple

import numpy as np

from algorithms.base import DeconvolutionAlgorithm

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from data_utils import gaussian_kernel  # type: ignore
from deconvolution import rl_deconv_all  # type: ignore


def _as_positive_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


class IdiapSemiblindpsfdeconv(DeconvolutionAlgorithm):
    def __init__(
        self,
        psf_size: int = 63,
        fwhm_x: float = 3.0,
        fwhm_y: float = 3.0,
        iterations: int = 20,
        regularization: float = 0.1,
        normalize_input: bool = True,
    ) -> None:
        super().__init__('SemiBlindSpatiallyVariantRL')
        self.psf_size = int(psf_size)
        self.fwhm_x = float(fwhm_x)
        self.fwhm_y = float(fwhm_y)
        self.iterations = int(iterations)
        self.regularization = float(regularization)
        self.normalize_input = bool(normalize_input)
        self._last_kernel: np.ndarray | None = None
        self._last_output: np.ndarray | None = None

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return

        if 'psf_size' in param:
            self.psf_size = _as_positive_int(param['psf_size'], self.psf_size)
        if 'fwhm_x' in param and param['fwhm_x'] is not None:
            self.fwhm_x = float(param['fwhm_x'])
        if 'fwhm_y' in param and param['fwhm_y'] is not None:
            self.fwhm_y = float(param['fwhm_y'])
        if 'iterations' in param:
            self.iterations = _as_positive_int(param['iterations'], self.iterations)
        if 'regularization' in param and param['regularization'] is not None:
            self.regularization = float(param['regularization'])
        if 'normalize_input' in param and param['normalize_input'] is not None:
            self.normalize_input = bool(param['normalize_input'])

        return

    def get_param(self):
        return [
            ('psf_size', self.psf_size),
            ('fwhm_x', self.fwhm_x),
            ('fwhm_y', self.fwhm_y),
            ('iterations', self.iterations),
            ('regularization', self.regularization),
            ('normalize_input', self.normalize_input),
            ('last_kernel_shape', None if self._last_kernel is None else self._last_kernel.shape),
            ('last_output_shape', None if self._last_output is None else self._last_output.shape),
        ]

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image is None:
            raise ValueError('Input image is None.')

        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError('Expected a 2D grayscale image.')

        original_dtype = arr.dtype
        working = arr.astype(np.float32, copy=False)
        min_level = float(working.min()) if working.size else 0.0
        working = working - min_level
        max_level = float(working.max()) if working.size else 1.0
        if self.normalize_input and max_level > 0:
            working = working / max_level

        kernel_size = max(3, int(self.psf_size) | 1)
        kernel = gaussian_kernel(kernel_size, self.fwhm_x, self.fwhm_y)

        start = time()
        restored = rl_deconv_all([working], [kernel], iterations=max(1, int(self.iterations)), lbd=float(self.regularization))
        self.timer = time() - start

        restored = np.asarray(restored, dtype=np.float32)
        if self.normalize_input and max_level > 0:
            restored = restored * max_level
        restored = restored + min_level

        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(restored, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
            restored_out = restored.round().astype(original_dtype)
        else:
            restored_out = restored.astype(original_dtype, copy=False)

        self._last_kernel = kernel
        self._last_output = restored_out
        return restored_out, kernel

    def get_kernel(self) -> np.ndarray | None:
        return self._last_kernel


__all__ = ['IdiapSemiblindpsfdeconv']
