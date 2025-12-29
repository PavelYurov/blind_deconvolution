# https://github.com/fabioviggiano/BlindDeconvolution
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from algorithms.base import DeconvolutionAlgorithm
from .shan_impl import deblurShanPyramidal


class FabioviggianoBlindDeconvolution(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: int = 35,
        iterations: int = 15,
        lambda_prior: float = 5e-3,
        lambda_kernel_reg: float = 1e-3,
        num_levels: int = 4,
    ) -> None:
        super().__init__('Shan')
        self.kernel_size = int(kernel_size)
        self.iterations = int(iterations)
        self.lambda_prior = float(lambda_prior)
        self.lambda_kernel_reg = float(lambda_kernel_reg)
        self.num_levels = int(num_levels)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return

        if 'kernel_size' in param and param['kernel_size']:
            self.kernel_size = int(param['kernel_size'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = int(param['iterations'])
        if 'num_iterations' in param and param['num_iterations'] is not None:
            self.iterations = int(param['num_iterations'])
        if 'lambda_prior' in param and param['lambda_prior'] is not None:
            self.lambda_prior = float(param['lambda_prior'])
        if 'lambda_kernel_reg' in param and param['lambda_kernel_reg'] is not None:
            self.lambda_kernel_reg = float(param['lambda_kernel_reg'])
        if 'lambda_kernel' in param and param['lambda_kernel'] is not None:
            self.lambda_kernel_reg = float(param['lambda_kernel'])
        if 'num_levels' in param and param['num_levels'] is not None:
            self.num_levels = int(param['num_levels'])
        if 'pyramid_levels' in param and param['pyramid_levels'] is not None:
            self.num_levels = int(param['pyramid_levels'])

        return

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('iterations', self.iterations),
            ('lambda_prior', self.lambda_prior),
            ('lambda_kernel_reg', self.lambda_kernel_reg),
            ('num_levels', self.num_levels),
        ]

    def process(self, image: np.ndarray):
        orig_dtype = image.dtype
        img = image.astype(np.float32)
        if img.max() > 1.5:
            img = img / 255.0

        restored, kernel = deblurShanPyramidal(
            img,
            self.kernel_size,
            num_iterations=self.iterations,
            lambda_prior=self.lambda_prior,
            lambda_kernel_reg=self.lambda_kernel_reg,
            num_levels=self.num_levels,
        )

        if np.issubdtype(orig_dtype, np.integer):
            restored_out = (np.clip(restored, 0, 1) * 255.0).astype(orig_dtype)
        else:
            restored_out = restored.astype(orig_dtype, copy=False)

        return restored_out, kernel

__all__ = ["FabioviggianoBlindDeconvolution"]
