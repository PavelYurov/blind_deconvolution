from __future__ import annotations

import sys
from time import time
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np

from ...base import DeconvolutionAlgorithm

# Ensure legacy absolute imports inside source modules resolve correctly.
from .source import L1_support as _l1_support
from .source import blur_kernels as _blur_kernels

sys.modules.setdefault("L1_support", _l1_support)
sys.modules.setdefault("blur_kernels", _blur_kernels)

from .source import blind_deconvolution as _blind_deconvolution
from .source import blind_deconv_devil_in_details as _blind_deconv_devil_in_details

sys.modules.setdefault("blind_deconvolution", _blind_deconvolution)

KernelSpec = Union[int, Tuple[int, int], Iterable[int]]


def _normalize_kernel_shape(spec: KernelSpec) -> Tuple[int, int]:
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("Kernel size must be positive.")
        return (spec, spec)

    try:
        values = tuple(int(v) for v in spec)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Unsupported kernel size specification.") from exc

    if len(values) == 0:
        raise ValueError("Kernel size can not be empty.")
    if len(values) == 1:
        return _normalize_kernel_shape(values[0])

    h, w = values[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    return (h, w)


def _prepare_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype, float]:
    if image.ndim != 2:
        raise ValueError("Dragos2001TotalVariationBlindDeconvolution expects a 2-D grayscale image.")

    original_dtype = image.dtype
    working = image.astype(np.float32, copy=False)
    scale = 255.0 if working.max(initial=0.0) > 1.5 else 1.0
    if scale != 1.0:
        working /= scale
    working = np.clip(working, 0.0, 1.0)
    return working, original_dtype, scale


def _restore_dtype(image: np.ndarray, dtype: np.dtype, scale: float) -> np.ndarray:
    clipped = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        restored = (clipped * scale).round().astype(dtype)
    else:
        restored = clipped.astype(dtype, copy=False)
    return restored


class Dragos2001TotalVariationBlindDeconvolution(DeconvolutionAlgorithm):
    """Wrapper around the source TV blind deconvolution routine."""

    def __init__(
        self,
        sigma = 2,
        kernel_size: KernelSpec = 9,
        alfa1: float = 5e-3,
        alfa2: float = 5e-3,
        theta: float = 0.5,
        deviation: float = 1e-4,
        step: float = 0.1,
        epsilon: float = 2e-3,
        criterium: float = 0.1,
        num_iterations_primary: int = 10,
        num_iterations_secondary: int = 5,
    ) -> None:
        super().__init__("TotalVariationBlindTV")
        self.kernel_shape = _normalize_kernel_shape(kernel_size)
        self.alfa1 = float(alfa1)
        self.alfa2 = float(alfa2)
        self.theta = float(theta)
        self.deviation = float(deviation)
        self.step = float(step)
        self.epsilon = float(epsilon)
        self.criterium = float(criterium)
        self.num_iterations_primary = int(num_iterations_primary)
        self.num_iterations_secondary = int(num_iterations_secondary)
        self.sigma = sigma

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_size"])
        if "kernel_shape" in param and param["kernel_shape"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_shape"])
        if "alfa1" in param and param["alfa1"] is not None:
            self.alfa1 = float(param["alfa1"])
        if "alfa2" in param and param["alfa2"] is not None:
            self.alfa2 = float(param["alfa2"])
        if "theta" in param and param["theta"] is not None:
            self.theta = float(param["theta"])
        if "deviation" in param and param["deviation"] is not None:
            self.deviation = float(param["deviation"])
        if "step" in param and param["step"] is not None:
            self.step = float(param["step"])
        if "epsilon" in param and param["epsilon"] is not None:
            self.epsilon = float(param["epsilon"])
        if "criterium" in param and param["criterium"] is not None:
            self.criterium = float(param["criterium"])
        if "num_iterations_primary" in param and param["num_iterations_primary"] is not None:
            self.num_iterations_primary = max(1, int(param["num_iterations_primary"]))
        if "num_iterations_secondary" in param and param["num_iterations_secondary"] is not None:
            self.num_iterations_secondary = max(1, int(param["num_iterations_secondary"]))
        if "sigma" in param and param["sigma"] is not None:
            self.sigma = param["sigma"]

        return super().change_param(param)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, dtype, scale = _prepare_image(np.asarray(image))

        start = time()
        restored, kernel = _blind_deconvolution.total_variation_deconvolution(
            working,
            sigma=self.sigma,
            kernel_size=self.kernel_shape,
            alfa1=self.alfa1,
            alfa2=self.alfa2,
            theta=self.theta,
            deviation=self.deviation,
            step=self.step,
            epsilon=self.epsilon,
            criterium=self.criterium,
            num_iterations_primary=self.num_iterations_primary,
            num_iterations_secondary=self.num_iterations_secondary,
        )
        self.timer = time() - start

        restored = np.asarray(restored, dtype=np.float32)
        kernel = np.asarray(kernel, dtype=np.float32)

        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        print(kernel_sum)
        output = _restore_dtype(restored, dtype, scale)
        return output, kernel

    def get_param(self):
        return [
            ("kernel_shape", self.kernel_shape),
            ("sigma", self.sigma),
            ("alfa1", self.alfa1),
            ("alfa2", self.alfa2),
            ("theta", self.theta),
            ("deviation", self.deviation),
            ("step", self.step),
            ("epsilon", self.epsilon),
            ("criterium", self.criterium),
            ("num_iterations_primary", self.num_iterations_primary),
            ("num_iterations_secondary", self.num_iterations_secondary),
        ]


class Dragos2001blinddeconvolutiondevilindetails(DeconvolutionAlgorithm):
    """Wrapper around the source TV blind deconvolution routine."""

    def __init__(
        self,
        kernel_size: KernelSpec = 9,
        lambda_v:float = 0.1,
        deviation: float = 1e-4,
        step: float = 0.1,
        criterium: float = 0.1,
        num_iterations_primary: int = 10,
        num_iterations_secondary: int = 5,
    ) -> None:
        super().__init__("TotalVariationBlindTVdevilindetails")
        self.kernel_shape = _normalize_kernel_shape(kernel_size)
        self.lambda_v = lambda_v
        self.deviation = float(deviation)
        self.step = float(step)
        self.criterium = float(criterium)
        self.num_iterations_primary = int(num_iterations_primary)
        self.num_iterations_secondary = int(num_iterations_secondary)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if "kernel_size" in param and param["kernel_size"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_size"])
        if "kernel_shape" in param and param["kernel_shape"] is not None:
            self.kernel_shape = _normalize_kernel_shape(param["kernel_shape"])
        if "lambda_v" in param and param["lambda_v"] is not None:
            self.lambda_v = float(param["lambda_v"])
        if "deviation" in param and param["deviation"] is not None:
            self.deviation = float(param["deviation"])
        if "step" in param and param["step"] is not None:
            self.step = float(param["step"])
        if "criterium" in param and param["criterium"] is not None:
            self.criterium = float(param["criterium"])
        if "num_iterations_primary" in param and param["num_iterations_primary"] is not None:
            self.num_iterations_primary = max(1, int(param["num_iterations_primary"]))
        if "num_iterations_secondary" in param and param["num_iterations_secondary"] is not None:
            self.num_iterations_secondary = max(1, int(param["num_iterations_secondary"]))

        return super().change_param(param)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        working, dtype, scale = _prepare_image(np.asarray(image))

        start = time()
        restored, kernel = _blind_deconv_devil_in_details.total_variation_deconvolution(
            working,
            kernel_size=self.kernel_shape,
            lambda_v=self.lambda_v,
            deviation=self.deviation,
            step=self.step,
            criterium=self.criterium,
            num_iterations_primary=self.num_iterations_primary,
            num_iterations_secondary=self.num_iterations_secondary,
        )
        self.timer = time() - start

        restored = np.asarray(restored, dtype=np.float32)
        kernel = np.asarray(kernel, dtype=np.float32)

        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        print(kernel_sum)
        output = _restore_dtype(restored, dtype, scale)
        return output, kernel

    def get_param(self):
        return [
            ("kernel_shape", self.kernel_shape),
            ("lambda_v", self.lambda_v),
            ("deviation", self.deviation),
            ("step", self.step),
            ("criterium", self.criterium),
            ("num_iterations_primary", self.num_iterations_primary),
            ("num_iterations_secondary", self.num_iterations_secondary),
        ]

__all__ = ["Dragos2001TotalVariationBlindDeconvolution"]
# __all__ = ["Dragos2001blinddeconvolutiondevilindetails"]

