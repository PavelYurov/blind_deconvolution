from __future__ import annotations

from time import time
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ...base import DeconvolutionAlgorithm

Array2D = np.ndarray
KernelSpec = Tuple[int, int]


def _normalize_kernel_shape(kernel: Any, default: KernelSpec = (9, 9)) -> KernelSpec:
    if kernel is None:
        return default
    if isinstance(kernel, int):
        if kernel <= 0:
            raise ValueError("Kernel size must be positive.")
        return (int(kernel), int(kernel))
    if isinstance(kernel, (tuple, list)):
        if len(kernel) == 1:
            return _normalize_kernel_shape(kernel[0], default)
        if len(kernel) >= 2:
            h, w = kernel[:2]
            if h <= 0 or w <= 0:
                raise ValueError("Kernel size entries must be positive.")
            return (int(h), int(w))
    raise ValueError("Unsupported kernel specification.")


def _prepare_image(image: Array2D) -> Tuple[Array2D, np.dtype, Optional[int]]:
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
        raise ValueError("MHDM expects a grayscale image (2-D array).")

    return working, original_dtype, channels


def _restore_dtype(restored: Array2D, original_dtype: np.dtype, channels: Optional[int]) -> Array2D:
    clipped = np.clip(restored, 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        output = (clipped * 255.0).round().astype(original_dtype)
    else:
        output = clipped.astype(original_dtype, copy=False)

    if channels is not None and channels > 1:
        output = np.repeat(output[..., None], channels, axis=2)

    return output


def _l2_norm(array: Array2D) -> float:
    return float(np.sqrt(np.sum(np.abs(array) ** 2)))


def _fourier_weights(shape: KernelSpec) -> Array2D:
    m, n = shape
    yy = 2 * (np.array(range(m)) / m)
    xx = 2 * (np.array(range(n)) / n)
    cos_y = 1 - np.cos(np.pi * yy)
    cos_x = 1 - np.cos(np.pi * xx)
    weights = 1 + (2 * (m ** 2) * cos_y)[:, None] + (2 * (n ** 2) * cos_x)[None, :]
    return weights.astype(np.float64)


def _hermitian_pairs(shape: KernelSpec) -> np.ndarray:
    m, n = shape
    total = m * n
    pairs: list[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for idx in range(total):
        row, col = np.unravel_index(idx, (m, n), order='F')
        partner = ((m - row) % m, (n - col) % n)
        partner_idx = np.ravel_multi_index(partner, (m, n), order='F')
        key = tuple(sorted((idx, partner_idx)))
        if key in seen:
            continue
        seen.add(key)
        if idx == 0:
            continue
        pairs.append((partner_idx, idx))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def _mhdm_initial(f_four: Array2D, lambda_: float, mu: float, r: float, s: float, weights: Array2D) -> Tuple[Array2D, Array2D]:
    ratio = np.sqrt((mu / lambda_) * np.power(weights, s - r))
    penalty = mu * np.power(weights, s)
    value = np.maximum(ratio * np.abs(f_four) - penalty, 0.0)
    u_four = np.sqrt(value)
    with np.errstate(invalid="ignore"):
        signs = np.divide(f_four, np.abs(f_four), out=np.zeros_like(f_four, dtype=np.complex128), where=np.abs(f_four) > 0)
    u_four = (u_four * signs).astype(np.complex128)

    ratio_k = np.sqrt((lambda_ / mu) * np.power(weights, r - s))
    penalty_k = lambda_ * np.power(weights, r)
    value_k = np.maximum(ratio_k * np.abs(f_four) - penalty_k, 0.0)
    k_four = np.sqrt(value_k).astype(np.complex128)

    u_four[0, 0] = f_four[0, 0]
    k_four[0, 0] = 1.0
    return u_four, k_four


def _select_root(coeffs: np.ndarray, shift: complex, objective, tol: float) -> complex:
    coeffs = np.trim_zeros(coeffs, trim='f')
    if coeffs.size == 0:
        return 0.0
    roots = np.roots(coeffs)
    if roots.size == 0:
        return 0.0
    best: Optional[complex] = None
    best_val: float = float("inf")
    for root in roots:
        candidate = root - shift
        if not np.isfinite(candidate):
            continue
        if candidate.real < -tol:
            continue
        val = objective(candidate)
        if not np.isfinite(val):
            continue
        if val < best_val:
            best_val = val
            best = candidate
    return 0.0 if best is None else best


def _mhdm_step(
    u_prev: Array2D,
    k_prev: Array2D,
    f_four: Array2D,
    lambda_: float,
    mu: float,
    r: float,
    s: float,
    tol: float,
    indices: np.ndarray,
    weights: Array2D,
) -> Tuple[Array2D, Array2D]:
    m, n = f_four.shape
    total = m * n

    flat_weights = weights.reshape(total, order='F')
    u_prev_flat = u_prev.reshape(total, order='F')
    k_prev_flat = k_prev.reshape(total, order='F')
    f_flat = f_four.reshape(total, order='F')

    u_inc = np.zeros(total, dtype=np.complex128)
    k_inc = np.zeros(total, dtype=np.complex128)

    if indices.size:
        num_rows = indices.shape[0]
        u_vals = np.zeros(num_rows, dtype=np.complex128)
        k_vals = np.zeros(num_rows, dtype=np.complex128)
        for row_idx, (partner_idx, idx) in enumerate(indices):
            a_n = lambda_ * np.power(flat_weights[idx], r)
            b_n = mu * np.power(flat_weights[idx], s)
            u_n = u_prev_flat[idx]
            k_n = k_prev_flat[idx]
            f_val = f_flat[idx]

            def objective(q: complex) -> float:
                denom = (np.abs(q + k_n) ** 2) + a_n
                if denom == 0:
                    denom = 1e-12
                term1 = a_n / denom * (np.abs(u_n * (q + k_n) - f_val) ** 2)
                term2 = b_n * (np.abs(q) ** 2)
                return float(np.real(term1 + term2))

            coeffs = np.array(
                [
                    b_n,
                    -k_n * b_n,
                    2 * a_n * b_n,
                    a_n * f_val * np.conj(u_n) - 2 * a_n * b_n * k_n,
                    (a_n ** 2) * (np.abs(u_n) ** 2) - a_n * (np.abs(f_val) ** 2) + (a_n ** 2) * b_n,
                    - (a_n ** 2) * (f_val * np.conj(u_n) + b_n * k_n),
                ],
                dtype=np.complex128,
            )
            coeffs = np.real(coeffs)
            k_delta = _select_root(coeffs, k_n, objective, tol)
            numerator = (a_n * u_n) + f_val * np.conj(k_delta + k_n)
            denom = (np.abs(k_delta + k_n) ** 2) + a_n
            if denom == 0:
                denom = 1e-12
            u_delta = numerator / denom - u_n

            k_vals[row_idx] = k_delta
            u_vals[row_idx] = u_delta

        target_indices = indices[:, 1]
        partner_indices = indices[:, 0]
        k_inc[target_indices] = k_vals
        u_inc[target_indices] = u_vals
        k_inc[partner_indices] = np.conj(k_vals)
        u_inc[partner_indices] = np.conj(u_vals)

    k_inc[0] = 1.0 - k_prev_flat[0]
    u_inc[0] = f_flat[0] - u_prev_flat[0]

    u_inc = u_inc.reshape((m, n), order='F')
    k_inc = k_inc.reshape((m, n), order='F')
    return u_inc, k_inc


def _fft_convolve(u_four: Array2D, k_four: Array2D) -> Array2D:
    return np.real(np.fft.ifft2(u_four * k_four))


def _otf_to_psf(otf: Array2D) -> Array2D:
    psf = np.real(np.fft.ifft2(otf))
    psf = np.fft.ifftshift(psf)
    psf = np.maximum(psf, 0.0)
    norm = psf.sum()
    if norm > 0:
        psf = psf / norm
    return psf


def _ensure_complex(array: Array2D) -> Array2D:
    return np.asarray(array, dtype=np.complex128)


class TobiasWolfMathBlindDeconvolutionMHDM(DeconvolutionAlgorithm):
    def __init__(
        self,
        lambda0: float = 14e-5,
        mu0: float = 63e4,
        r: float = 1.0,
        s: float = 1e-1,
        tol: float = 1e-10,
        max_iter: int = 5,
        stopping_factor: float = 0.0,
        noise_level: Optional[float] = None,
        kernel_size: KernelSpec = (9, 9),
    ) -> None:
        super().__init__('MHDM')
        self.lambda0 = float(lambda0)
        self.mu0 = float(mu0)
        self.r = float(r)
        self.s = float(s)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.stopping_factor = float(stopping_factor)
        self.noise_level = None if noise_level is None else float(noise_level)
        self.kernel_size = _normalize_kernel_shape(kernel_size)

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'lambda0' in param and param['lambda0'] is not None:
                self.lambda0 = float(param['lambda0'])
            if 'mu0' in param and param['mu0'] is not None:
                self.mu0 = float(param['mu0'])
            if 'r' in param and param['r'] is not None:
                self.r = float(param['r'])
            if 's' in param and param['s'] is not None:
                self.s = float(param['s'])
            if 'tol' in param and param['tol'] is not None:
                self.tol = float(param['tol'])
            if 'max_iter' in param and param['max_iter'] is not None:
                self.max_iter = int(param['max_iter'])
            if 'stopping_factor' in param and param['stopping_factor'] is not None:
                self.stopping_factor = float(param['stopping_factor'])
            if 'noise_level' in param:
                value = param['noise_level']
                self.noise_level = None if value is None else float(value)
            if 'kernel_size' in param and param['kernel_size'] is not None:
                self.kernel_size = _normalize_kernel_shape(param['kernel_size'])
            return super().change_param(param)
        return super().change_param(param)


    def _stopping_threshold(self, image: Array2D) -> float:
        if self.noise_level is None or self.noise_level <= 0:
            return 0.0
        return self.stopping_factor * self.noise_level * np.sqrt(image.size)

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        working, original_dtype, channels = _prepare_image(image)
        start = time()

        f_four = np.fft.fft2(working)
        weights = _fourier_weights(working.shape)
        u_four, k_four = _mhdm_initial(
            f_four,
            self.lambda0,
            self.mu0,
            self.r,
            self.s,
            weights,
        )
        indices = _hermitian_pairs(working.shape)
        stopping_threshold = self._stopping_threshold(working)

        residual = _l2_norm(working - _fft_convolve(u_four, k_four))
        lambda_val = self.lambda0
        mu_val = self.mu0

        iteration = 1
        while residual > stopping_threshold and iteration < self.max_iter:
            lambda_val /= 4.0
            mu_val /= 4.0
            u_inc, k_inc = _mhdm_step(
                u_four,
                k_four,
                f_four,
                lambda_val,
                mu_val,
                self.r,
                self.s,
                self.tol,
                indices,
                weights,
            )
            u_four = _ensure_complex(u_four + u_inc)
            k_four = _ensure_complex(k_four + k_inc)
            residual = _l2_norm(working - _fft_convolve(u_four, k_four))
            iteration += 1

        restored = np.real(np.fft.ifft2(u_four))
        kernel = _otf_to_psf(k_four)

        self.timer = time() - start
        return _restore_dtype(restored, original_dtype, channels), kernel.astype(np.float32)

    def get_param(self):
        return [
            ('lambda0', self.lambda0),
            ('mu0', self.mu0),
            ('r', self.r),
            ('s', self.s),
            ('tol', self.tol),
            ('max_iter', self.max_iter),
            ('stopping_factor', self.stopping_factor),
            ('noise_level', self.noise_level),
            ('kernel_size', self.kernel_size),
        ]


__all__ = ['TobiasWolfMathBlindDeconvolutionMHDM']

__all__ = ["TobiasWolfMathBlindDeconvolutionMHDM"]
