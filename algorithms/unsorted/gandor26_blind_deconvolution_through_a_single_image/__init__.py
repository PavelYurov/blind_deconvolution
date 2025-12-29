#https://github.com/Gandor26/Blind-Deconvolution-through-a-Single-Image
from typing import Any, Dict, Tuple

import numpy as np
try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from algorithms.base import DeconvolutionAlgorithm
from .deconvolution import DeConvolution


class Gandor26BlindDeconvolutionThroughASingleImageDeconvolutionAlgorithm(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (11, 11),
        max_steps: int = 5,
        tol: float = 1e-4,
    ) -> None:
        super().__init__("Blind Deconvolution through a Single Image")
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.max_steps = int(max_steps)
        self.tol = float(tol)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if "kernel_size" in param and param["kernel_size"] is not None:
            value = param["kernel_size"]
            if isinstance(value, (tuple, list)) and len(value) == 2:
                self.kernel_size = (int(value[0]), int(value[1]))
            else:
                size = int(value)
                self.kernel_size = (size, size)
        if "max_steps" in param and param["max_steps"] is not None:
            self.max_steps = int(param["max_steps"])
        if "tol" in param and param["tol"] is not None:
            self.tol = float(param["tol"])
        return super().change_param(param)

    def get_param(self):
        return [
            ("kernel_size", self.kernel_size),
            ("max_steps", self.max_steps),
            ("tol", self.tol),
        ]

    def process(self, image: np.ndarray):
        if image.ndim == 3:
            if cv2 is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = (0.114 * image[..., 0] + 0.587 * image[..., 1] + 0.299 * image[..., 2]).astype(image.dtype)
        else:
            gray = image
        orig_dtype = gray.dtype
        gray_f = gray.astype(np.float64)
        if gray_f.max() > 1.5:
            gray_f /= 255.0

        solver = DeConvolution(
            gray_f,
            kernel_shape=self.kernel_size,
            tol=self.tol,
            max_steps=self.max_steps,
        )
        restored, kernel = solver.run()

        if np.issubdtype(orig_dtype, np.integer):
            restored_img = (np.clip(restored, 0.0, 1.0) * 255.0).astype(orig_dtype)
        else:
            restored_img = restored.astype(orig_dtype, copy=False)
        return restored_img, kernel


__all__ = [
    "Gandor26BlindDeconvolutionThroughASingleImageDeconvolutionAlgorithm",
]
