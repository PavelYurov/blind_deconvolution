from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Iterable, Sequence

import cv2
import cvxpy as cp
import numpy as np

from ..base import DeconvolutionAlgorithm


ArrayLike = np.ndarray


@dataclass
class _ImagePreparation:
    working: ArrayLike
    original_dtype: np.dtype
    original_shape: tuple[int, ...]


def _to_grayscale(image: ArrayLike) -> ArrayLike:
    if image.ndim == 3 and image.shape[2] != 1:
        return image.mean(axis=2)
    return np.squeeze(image)


def _prepare_image(image: ArrayLike) -> _ImagePreparation:
    original_dtype = image.dtype
    original_shape = image.shape
    working = _to_grayscale(image)
    working = working.astype(np.float64, copy=False)
    max_value = float(working.max()) if working.size else 0.0
    if max_value > 1.5:
        working /= 255.0
    working = np.clip(working, 0.0, 1.0)
    return _ImagePreparation(working, original_dtype, original_shape)


def _resize_for_workspace(image: ArrayLike, workspace_max: int) -> tuple[ArrayLike, tuple[int, int]]:
    height, width = image.shape
    if workspace_max is None or max(height, width) <= workspace_max:
        return image, (height, width)

    scale = workspace_max / float(max(height, width))
    new_height = max(8, int(round(height * scale)))
    new_width = max(8, int(round(width * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, (height, width)


def _dft_matrix(n: int) -> ArrayLike:
    idx = np.arange(n, dtype=np.float64)
    root = np.exp(-2j * np.pi / n)
    return np.power(root, np.outer(idx, idx)) / np.sqrt(n)


def _kernel_support_indices(shape: tuple[int, int], kernel_size: tuple[int, int]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    height, width = shape
    k_height, k_width = kernel_size
    if k_height % 2 == 0 or k_width % 2 == 0:
        raise ValueError('Kernel dimensions must be odd.')

    row_start = (height - k_height) // 2
    col_start = (width - k_width) // 2

    indices: list[int] = []
    rel_coords: list[tuple[int, int]] = []
    for i in range(k_height):
        for j in range(k_width):
            row = row_start + i
            col = col_start + j
            if 0 <= row < height and 0 <= col < width:
                indices.append(row * width + col)
                rel_coords.append((i, j))
    if not indices:
        raise ValueError('Kernel support is empty for the current workspace size.')
    return np.asarray(indices, dtype=np.int32), rel_coords


def _select_support(flat: ArrayLike, max_coeffs: int) -> np.ndarray:
    count = int(max(1, min(max_coeffs, flat.size)))
    magnitudes = np.abs(flat)
    if count >= flat.size:
        selected = np.arange(flat.size, dtype=np.int32)
    else:
        partition_idx = np.argpartition(-magnitudes, count - 1)[:count]
        selected = np.sort(partition_idx.astype(np.int32))
    if selected.size > 0 and selected[0] != 0:
        selected[0] = 0
    return selected


def _restore_dtype(image: ArrayLike, original_dtype: np.dtype, original_shape: tuple[int, ...]) -> ArrayLike:
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


def _normalise_kernel(kernel: ArrayLike) -> ArrayLike:
    kernel = kernel.astype(np.float64, copy=False)
    min_val = float(kernel.min()) if kernel.size else 0.0
    if min_val < 0.0:
        kernel = kernel - min_val
    kernel = np.clip(kernel, 0.0, None)
    total = float(kernel.sum())
    if total <= 0.0:
        kernel = np.abs(kernel)
        total = float(kernel.sum())
        if total <= 0.0:
            return kernel
    return kernel / total


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


class WarrenzhaBlindDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: int | Sequence[int] = (5, 5),
        workspace_max: int = 16,
        max_image_coeffs: int = 64,
        solver: str = 'SCS',
        solver_max_iters: int = 4000,
        solver_verbose: bool = False,
        consistency_weight: float = 10.0,
    ) -> None:
        super().__init__('ZhaoBlindDeconv')
        self.consistency_weight = max(0.0, float(consistency_weight))
        if isinstance(kernel_size, Iterable) and not isinstance(kernel_size, (str, bytes)):
            size = list(kernel_size)
            if len(size) < 2:
                raise ValueError('kernel_size must specify two dimensions.')
            self.kernel_size = (_ensure_odd(size[0]), _ensure_odd(size[1]))
        else:
            odd = _ensure_odd(int(kernel_size))
            self.kernel_size = (odd, odd)
        self.workspace_max = max(8, int(workspace_max))
        self.max_image_coeffs = max(1, int(max_image_coeffs))
        self.solver = solver
        self.solver_max_iters = max(100, int(solver_max_iters))
        self.solver_verbose = bool(solver_verbose)
        self._last_kernel: ArrayLike | None = None

    def change_param(self, param: Any):
        if isinstance(param, dict):
            if 'kernel_size' in param and param['kernel_size'] is not None:
                size = param['kernel_size']
                if isinstance(size, Iterable) and not isinstance(size, (str, bytes)):
                    size = list(size)
                    if len(size) < 2:
                        raise ValueError('kernel_size must specify two dimensions.')
                    self.kernel_size = (_ensure_odd(size[0]), _ensure_odd(size[1]))
                else:
                    odd = _ensure_odd(int(size))
                    self.kernel_size = (odd, odd)
            if 'workspace_max' in param and param['workspace_max'] is not None:
                self.workspace_max = max(8, int(param['workspace_max']))
            if 'max_image_coeffs' in param and param['max_image_coeffs'] is not None:
                self.max_image_coeffs = max(1, int(param['max_image_coeffs']))
            if 'solver' in param and param['solver']:
                self.solver = str(param['solver'])
            if 'solver_max_iters' in param and param['solver_max_iters'] is not None:
                self.solver_max_iters = max(100, int(param['solver_max_iters']))
            if 'solver_verbose' in param and param['solver_verbose'] is not None:
                self.solver_verbose = bool(param['solver_verbose'])
            if 'consistency_weight' in param and param['consistency_weight'] is not None:
                self.consistency_weight = max(0.0, float(param['consistency_weight']))
        return super().change_param(param)

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('workspace_max', self.workspace_max),
            ('max_image_coeffs', self.max_image_coeffs),
            ('solver', self.solver),
            ('solver_max_iters', self.solver_max_iters),
            ('solver_verbose', self.solver_verbose),
            ('consistency_weight', self.consistency_weight),
        ]

    def _solve_convex_program(
        self,
        y_hat: ArrayLike,
        B_hat: ArrayLike,
        C_hat: ArrayLike,
        matrix_scale: float,
        penalty_weight: float,
    ) -> np.ndarray:
        K = B_hat.shape[1]
        N = C_hat.shape[1]
        if K == 0 or N == 0:
            raise RuntimeError('Invalid operator dimensions for convex program.')
        X = cp.Variable((K, N), complex=True)
        B_const = cp.Constant(B_hat)
        C_const = cp.Constant(matrix_scale * C_hat)
        y_const = cp.Constant(y_hat)

        lhs = cp.sum(cp.multiply(C_const, B_const @ X), axis=1)
        if penalty_weight <= 0.0:
            objective = cp.Minimize(cp.normNuc(X))
            constraints = [lhs == y_const]
        else:
            residual = lhs - y_const
            objective = cp.Minimize(cp.normNuc(X) + penalty_weight * cp.norm(residual, 2))
            constraints = []
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(
                solver=self.solver,
                verbose=self.solver_verbose,
                max_iters=self.solver_max_iters,
            )
        except cp.SolverError as exc:
            if penalty_weight > 0.0:
                return self._solve_convex_program(y_hat, B_hat, C_hat, matrix_scale, 0.0)
            raise RuntimeError('Convex program solver error.') from exc

        statuses_ok = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT}
        if X.value is None or problem.status not in statuses_ok:
            if penalty_weight > 0.0 and problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
                return self._solve_convex_program(y_hat, B_hat, C_hat, matrix_scale, 0.0)
            raise RuntimeError(f'Convex program failed: {problem.status}.')
        return np.asarray(X.value)


    def process(self, image: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        start = time()
        prepared = _prepare_image(image)
        workspace, original_hw = _resize_for_workspace(prepared.working, self.workspace_max)
        height, width = workspace.shape
        L = height * width

        kernel_indices, rel_coords = _kernel_support_indices((height, width), self.kernel_size)
        flat_workspace = workspace.reshape(L)
        F = _dft_matrix(L)
        y_hat = F @ flat_workspace

        pad_kernel = F[:, kernel_indices]
        image_indices = _select_support(flat_workspace, self.max_image_coeffs)
        C_hat = F[:, image_indices]

        X_val = self._solve_convex_program(
            y_hat, pad_kernel, C_hat, np.sqrt(L), self.consistency_weight
        )

        U, s, Vh = np.linalg.svd(X_val, full_matrices=False)
        if s.size == 0 or s[0] <= 0:
            raise RuntimeError('Convex recovery returned a degenerate solution.')
        scale = np.sqrt(s[0])
        h_vec = (U[:, 0] * scale).real
        m_vec = (Vh.conj().T[:, 0] * scale).real

        kernel_patch = np.zeros(self.kernel_size, dtype=np.float64)
        for value, (i, j) in zip(h_vec, rel_coords):
            kernel_patch[i, j] = value
        kernel_patch = _normalise_kernel(kernel_patch)

        image_vec = np.zeros(L, dtype=np.float64)
        for idx, pos in enumerate(image_indices):
            image_vec[pos] = m_vec[idx]
        restored_small = image_vec.reshape((height, width))
        restored_small = np.clip(restored_small, 0.0, None)
        if restored_small.max() > 0:
            restored_small /= restored_small.max()

        restored = cv2.resize(restored_small, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR)
        self.timer = time() - start

        restored = _restore_dtype(restored, prepared.original_dtype, prepared.original_shape)
        self._last_kernel = kernel_patch
        return restored, kernel_patch

    def get_kernel(self) -> ArrayLike | None:
        return self._last_kernel


__all__ = ['WarrenzhaBlindDeconvolution']

__all__ = ["WarrenzhaBlindDeconvolution"]

class pizda:
	pass
