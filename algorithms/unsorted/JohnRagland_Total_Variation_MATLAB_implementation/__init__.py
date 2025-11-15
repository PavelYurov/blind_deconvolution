from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Tuple

import numpy as np
from scipy.signal import convolve2d

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _rot180(arr: Array2D) -> Array2D:
    return np.flip(arr, axis=(0, 1))


def _tv_gradient(arr: Array2D, epsilon: float = 1e-8) -> Array2D:
    """Compute the gradient of isotropic TV energy via divergence of normalized gradients."""
    grad_x = np.zeros_like(arr, dtype=np.float32)
    grad_y = np.zeros_like(arr, dtype=np.float32)

    grad_x[:, :-1] = arr[:, :-1] - arr[:, 1:]
    grad_y[:-1, :] = arr[:-1, :] - arr[1:, :]

    denom = np.sqrt(grad_x * grad_x + grad_y * grad_y + epsilon)
    grad_x /= denom
    grad_y /= denom

    div = np.zeros_like(arr, dtype=np.float32)
    div[:, :-1] += grad_x[:, :-1]
    div[:, 1:] -= grad_x[:, :-1]
    div[:-1, :] += grad_y[:-1, :]
    div[1:, :] -= grad_y[:-1, :]
    return div


def _ensure_odd(value: int) -> int:
    value = int(value)
    return value | 1


class JohnRaglandTotalVariationMATLABImplementation(DeconvolutionAlgorithm):
    """Python port of John Ragland's MATLAB TV blind deconvolution."""

    def __init__(
        self,
        alpha1: float = 5e-6,
        alpha2: float = 1e-4,
        gamma: float = 1e-1,
        beta: float = 1e-5,
        iterations: int = 200,
        kernel_size: int = 7,
    ) -> None:
        super().__init__('TotalVariationBlindDeconvolution')
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.iterations = max(1, int(iterations))
        self.kernel_size = _ensure_odd(kernel_size)
        self._last_kernel: Array2D | None = None
        self._last_output: Array2D | None = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'alpha1' in param and param['alpha1'] is not None:
            self.alpha1 = float(param['alpha1'])
        if 'alpha2' in param and param['alpha2'] is not None:
            self.alpha2 = float(param['alpha2'])
        if 'gamma' in param and param['gamma'] is not None:
            self.gamma = float(param['gamma'])
        if 'beta' in param and param['beta'] is not None:
            self.beta = float(param['beta'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = max(1, int(param['iterations']))
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = _ensure_odd(int(param['kernel_size']))

        return super().change_param(param)

    def get_param(self):
        return [
            ('alpha1', self.alpha1),
            ('alpha2', self.alpha2),
            ('gamma', self.gamma),
            ('beta', self.beta),
            ('iterations', self.iterations),
            ('kernel_size', self.kernel_size),
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
        working = arr.astype(np.float32, copy=False)

        kernel_shape = (self.kernel_size, self.kernel_size)
        pad_y = kernel_shape[0] // 2
        pad_x = kernel_shape[1] // 2
        if working.shape[0] <= 2 * pad_y or working.shape[1] <= 2 * pad_x:
            raise ValueError('Image is too small for the requested kernel size.')

        z = working[pad_y: working.shape[0] - pad_y, pad_x: working.shape[1] - pad_x]

        un = working.copy()
        kn = np.zeros(kernel_shape, dtype=np.float32)

        start = time()
        for _ in range(self.iterations):
            conv_un_kn_valid = convolve2d(un, kn, mode='valid')
            residual = conv_un_kn_valid - z

            dkn = convolve2d(_rot180(un), residual, mode='valid') - self.alpha2 * _tv_gradient(kn)
            kn = kn - self.beta * dkn
            kn = np.clip(kn, 0.0, None)
            kn = 0.5 * (kn + _rot180(kn))
            kn_sum = float(kn.sum())
            if kn_sum > 1e-12:
                kn = kn / kn_sum

            dun = convolve2d(residual, _rot180(kn), mode='full') - self.alpha1 * _tv_gradient(un)
            un = un - self.gamma * dun
            un = np.clip(un, 0.0, None)

        self.timer = time() - start

        restored = un
        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(restored, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
            restored = restored.round().astype(original_dtype)
        else:
            restored = restored.astype(original_dtype, copy=False)

        self._last_kernel = kn
        self._last_output = restored
        return restored, kn

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ['JohnRaglandTotalVariationMATLABImplementation']
