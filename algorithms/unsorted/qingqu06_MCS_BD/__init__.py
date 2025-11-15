from __future__ import annotations

import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def _normalize(arr: Array2D) -> Array2D:
    norm_val = float(np.linalg.norm(arr))
    if norm_val <= 1e-12:
        return arr
    return arr / norm_val


def _orth_proj(u: Array2D, v: Array2D) -> Array2D:
    denom = float(np.sum(u * u))
    if denom <= 1e-12:
        return v
    return v - float(np.sum(u * v)) * u / denom


def _multi_conv(a: np.ndarray, b: np.ndarray, adj: Optional[str] = None) -> np.ndarray:
    if a.ndim == 2:
        a = a[..., np.newaxis]
    if b.ndim == 2:
        b = b[..., np.newaxis]
    fft_shape = a.shape[:2]
    a_hat = np.fft.fft2(a, axes=(0, 1))
    b_hat = np.fft.fft2(b, s=fft_shape, axes=(0, 1))
    if adj == 'adj-left':
        a_hat = np.conj(a_hat)
    elif adj == 'adj-right':
        b_hat = np.conj(b_hat)
    elif adj == 'adj-both':
        a_hat = np.conj(a_hat)
        b_hat = np.conj(b_hat)
    if a_hat.shape[2] == 1 and b_hat.shape[2] > 1:
        a_hat = np.repeat(a_hat, b_hat.shape[2], axis=2)
    if b_hat.shape[2] == 1 and a_hat.shape[2] > 1:
        b_hat = np.repeat(b_hat, a_hat.shape[2], axis=2)
    result_hat = a_hat * b_hat
    return np.real(np.fft.ifft2(result_hat, axes=(0, 1)))


def _project_kernel(kernel: Array2D) -> Array2D:
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        side = kernel.shape[0]
        return np.full_like(kernel, 1.0 / (side * side))
    return kernel / total


def _extract_patches(image: Array2D, patch_size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    h, w = image.shape
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='edge')
        h, w = image.shape
    patches = np.zeros((patch_size, patch_size, count), dtype=np.float64)
    for i in range(count):
        top = 0 if h == patch_size else int(rng.integers(0, h - patch_size + 1))
        left = 0 if w == patch_size else int(rng.integers(0, w - patch_size + 1))
        patches[:, :, i] = image[top : top + patch_size, left : left + patch_size]
    patches -= patches.mean(axis=(0, 1), keepdims=True)
    return patches


def _wiener_deconv(image: Array2D, kernel: Array2D, reg: float) -> Array2D:
    kernel = _project_kernel(kernel)
    kh, kw = kernel.shape
    pad_kernel = np.zeros_like(image)
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -kh // 2, axis=0)
    pad_kernel = np.roll(pad_kernel, -kw // 2, axis=1)
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(pad_kernel)
    denom = np.abs(kernel_fft) ** 2 + reg
    restored_fft = image_fft * np.conj(kernel_fft) / denom
    restored = np.real(np.fft.ifft2(restored_fft))
    return np.clip(restored, 0.0, 1.0)


class _HuberObjective:
    def __init__(self, y: np.ndarray, mu: float) -> None:
        self.y = y
        self.mu = float(mu)
        self.shape = y.shape[:2]
        self.count = y.shape[2]
        self.normalizer = 1.0 / (self.shape[0] * self.shape[1] * self.count)

    def oracle(self, z: Array2D) -> Tuple[float, Array2D]:
        conv = _multi_conv(self.y, z)
        abs_conv = np.abs(conv)
        mu_val = self.mu
        huber_val = np.where(
            abs_conv >= mu_val,
            abs_conv,
            (mu_val / 2.0) + (conv ** 2) / (2.0 * mu_val),
        )
        fval = self.normalizer * float(np.sum(huber_val))
        grad_arg = np.where(abs_conv >= mu_val, np.sign(conv), conv / mu_val)
        grad = self.normalizer * np.sum(_multi_conv(self.y, grad_arg, adj='adj-left'), axis=2)
        return fval, grad


def _linesearch(objective: _HuberObjective, z: Array2D, fval: float, grad: Array2D, tau: float) -> Tuple[Array2D, float]:
    beta = 0.8
    eta = 1e-3
    tau = max(tau, 1e-12) * 2.0
    grad_norm_sq = float(np.sum(grad * grad))
    if grad_norm_sq <= 1e-20:
        return z, tau
    tau_threshold = 1e-12
    while True:
        candidate = _normalize(z - tau * grad)
        cand_val = objective.oracle(candidate)[0]
        if cand_val <= fval - eta * tau * grad_norm_sq or tau <= tau_threshold:
            return candidate, max(tau, tau_threshold)
        tau *= beta


def _grad_descent(objective: _HuberObjective, opts: Dict[str, Any]) -> Array2D:
    z = np.array(opts['Z_init'], dtype=np.float64)
    tau = float(opts.get('tau', 1e-2))
    max_iter = int(opts.get('MaxIter', 200))
    use_linesearch = bool(opts.get('islinesearch', True))
    line_tau = float(opts.get('line_tau', 0.1))
    for _ in range(max_iter):
        fval, grad = objective.oracle(z)
        grad = _orth_proj(z, grad)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm > 1e2:
            grad = grad * (1e2 / grad_norm)
        if use_linesearch:
            z, line_tau = _linesearch(objective, z, fval, grad, line_tau)
        else:
            z = _normalize(z - tau * grad)
    return z


class Qingqu06MCSBD(DeconvolutionAlgorithm):
    """Python port of the MCS-BD Riemannian gradient method (simplified)."""

    def __init__(
        self,
        kernel_size: int = 17,
        iterations: int = 200,
        mu: float = 1e-2,
        step_size: float = 1e-2,
        line_search: bool = True,
        num_patches: int = 64,
        regularization: float = 1e-3,
        random_state: Optional[int] = 0,
    ) -> None:
        super().__init__('MCSBlindDeconvolution')
        self.kernel_size = int(kernel_size)
        self.iterations = int(iterations)
        self.mu = float(mu)
        self.step_size = float(step_size)
        self.line_search = bool(line_search)
        self.num_patches = max(1, int(num_patches))
        self.regularization = float(regularization)
        self.random_state = random_state
        self._last_kernel: Optional[Array2D] = None
        self._last_output: Optional[Array2D] = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = int(param['kernel_size'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = int(param['iterations'])
        if 'mu' in param and param['mu'] is not None:
            self.mu = float(param['mu'])
        if 'step_size' in param and param['step_size'] is not None:
            self.step_size = float(param['step_size'])
        if 'line_search' in param and param['line_search'] is not None:
            self.line_search = bool(param['line_search'])
        if 'num_patches' in param and param['num_patches'] is not None:
            self.num_patches = max(1, int(param['num_patches']))
        if 'regularization' in param and param['regularization'] is not None:
            self.regularization = float(param['regularization'])
        if 'random_state' in param:
            self.random_state = None if param['random_state'] is None else int(param['random_state'])
        return super().change_param(param)

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('iterations', self.iterations),
            ('mu', self.mu),
            ('step_size', self.step_size),
            ('line_search', self.line_search),
            ('num_patches', self.num_patches),
            ('regularization', self.regularization),
            ('random_state', self.random_state),
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
        float_img = arr.astype(np.float64, copy=False)
        if float_img.max() > 1.5:
            float_img = float_img / 255.0
        float_img = np.clip(float_img, 0.0, 1.0)

        rng = np.random.default_rng(self.random_state)
        patch_size = max(3, self.kernel_size | 1)
        patches = _extract_patches(float_img, patch_size, self.num_patches, rng)

        fft_patches = np.fft.fft2(patches, axes=(0, 1))
        power_spectrum = np.sum(np.abs(fft_patches) ** 2, axis=2)
        power_spectrum = power_spectrum / (patch_size ** 2 * patches.shape[2])
        V = np.power(power_spectrum + 1e-8, -0.5)
        Y_p = np.fft.ifft2(fft_patches * V[..., np.newaxis], axes=(0, 1))
        Y_p = np.real(Y_p)

        Z_init = rng.normal(size=(patch_size, patch_size))
        Z_init = _normalize(Z_init)
        opts = {
            'Z_init': Z_init,
            'tau': self.step_size,
            'MaxIter': self.iterations,
            'islinesearch': self.line_search,
            'line_tau': 0.1,
        }
        objective = _HuberObjective(Y_p, self.mu)

        start = time()
        Z_est = _grad_descent(objective, opts)
        kernel_precond = np.real(np.fft.ifft2(V * np.fft.fft2(Z_est)))
        kernel_est = _project_kernel(kernel_precond)

        restored = _wiener_deconv(float_img, kernel_est, self.regularization)
        self.timer = time() - start

        if np.issubdtype(original_dtype, np.integer):
            output = np.clip(restored * 255.0, 0, 255).round().astype(original_dtype)
        else:
            output = restored.astype(original_dtype, copy=False)

        self._last_kernel = kernel_est
        self._last_output = output
        return output, kernel_est

    def get_kernel(self) -> Array2D | None:
        return None if self._last_kernel is None else self._last_kernel.copy()


__all__ = ['Qingqu06MCSBD']
