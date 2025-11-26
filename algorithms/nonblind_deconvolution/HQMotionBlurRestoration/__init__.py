from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, Tuple

import numpy as np

from ...base import DeconvolutionAlgorithm
from .convolve import create_line_psf
from .deblur import computeLocalPrior, updatePsi, computeL, updatef


@dataclass
class HQMotionParams:
    gamma: float = 2.0
    lambda1: float = 0.5
    lambda2: float = 25.0
    kernel_scale: float = 1.0


class HQMotionBlindDeconvolution(DeconvolutionAlgorithm):
    """High-quality motion blind deconvolution (Stuani & Lacroix, 2008)."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (27, 27),
        angle: float = 0.0,
        params: HQMotionParams | None = None,
        max_iterations: int = 3,
        local_threshold: float = 5.0,
        n_rows: int = 260,
    ) -> None:
        super().__init__('HQMotionBlindDeconv')
        self.kernel_shape = tuple(int(v) for v in kernel_shape)
        self.angle = float(angle)
        self.params = params or HQMotionParams()
        self.max_iterations = int(max_iterations)
        self.local_threshold = float(local_threshold)
        self.n_rows = int(n_rows)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'kernel_shape' in param and param['kernel_shape'] is not None:
            shape = param['kernel_shape']
            if isinstance(shape, (tuple, list)) and len(shape) >= 2:
                self.kernel_shape = (int(shape[0]), int(shape[1]))
        if 'angle' in param and param['angle'] is not None:
            self.angle = float(param['angle'])
        if 'gamma' in param and param['gamma'] is not None:
            self.params.gamma = float(param['gamma'])
        if 'lambda1' in param and param['lambda1'] is not None:
            self.params.lambda1 = float(param['lambda1'])
        if 'lambda2' in param and param['lambda2'] is not None:
            self.params.lambda2 = float(param['lambda2'])
        if 'kernel_scale' in param and param['kernel_scale'] is not None:
            self.params.kernel_scale = float(param['kernel_scale'])
        if 'max_iterations' in param and param['max_iterations'] is not None:
            self.max_iterations = int(param['max_iterations'])
        if 'local_threshold' in param and param['local_threshold'] is not None:
            self.local_threshold = float(param['local_threshold'])
        if 'n_rows' in param and param['n_rows'] is not None:
            self.n_rows = int(param['n_rows'])
        return super().change_param(param)

    def process(self, image: np.ndarray):
        start = time()
        original_dtype = image.dtype
        img = image.astype(np.float32, copy=False)
        if img.max(initial=0.0) > 1.5:
            img = img / 255.0

        # Work with grayscale but keep channel dimension for the legacy helpers.
        gray = np.atleast_3d(img.mean(axis=2) if img.ndim == 3 else img)

        kernel = create_line_psf(
            theta=self.angle,
            scale=self.params.kernel_scale,
            sz=self.kernel_shape,
        )

        # local priors per channel
        M = np.zeros_like(gray)
        for channel in range(gray.shape[2]):
            M[:, :, channel] = computeLocalPrior(gray[:, :, channel], kernel.shape, self.local_threshold)

        I_d = [np.gradient(gray[:, :, c], axis=(1, 0)) for c in range(gray.shape[2])]
        Psi = [[np.zeros(gray.shape[:2]), np.zeros(gray.shape[:2])] for _ in range(gray.shape[2])]
        L = gray.copy()

        gamma = self.params.gamma
        for _ in range(self.max_iterations):
            for c in range(L.shape[2]):
                L_d = np.gradient(L[:, :, c], axis=(1, 0))
                Psi[c] = updatePsi(
                    I_d[c],
                    L_d,
                    M[:, :, c],
                    self.params.lambda1,
                    self.params.lambda2,
                    gamma,
                )
                L[:, :, c] = computeL(L[:, :, c], gray[:, :, c], kernel, Psi[c], gamma)

            kernel = updatef(L, gray, kernel, n_rows=self.n_rows, k_cut_ratio=0.0)
            kernel = np.clip(kernel, 0.0, None)
            kernel_sum = kernel.sum()
            if kernel_sum > 0:
                kernel /= kernel_sum
            gamma *= 2.0

        restored = np.clip(L.squeeze(), 0.0, 1.0)
        if np.issubdtype(original_dtype, np.integer):
            restored_out = (restored * 255.0).round().astype(original_dtype)
        else:
            restored_out = restored.astype(original_dtype, copy=False)

        self.timer = time() - start
        return restored_out, kernel.astype(np.float32)

    def get_param(self):
        return [
            ('kernel_shape', self.kernel_shape),
            ('angle', self.angle),
            ('gamma', self.params.gamma),
            ('lambda1', self.params.lambda1),
            ('lambda2', self.params.lambda2),
            ('kernel_scale', self.params.kernel_scale),
            ('max_iterations', self.max_iterations),
            ('local_threshold', self.local_threshold),
            ('n_rows', self.n_rows),
        ]

__all__ = ["HQMotionBlindDeconvolution", "HQMotionParams"]
