# https://github.com/Tmodrzyk/richardson-lucy-python
from __future__ import annotations
from time import time
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch

from algorithms.base import DeconvolutionAlgorithm

from .source.src.algorithms.richardson_lucy import blind_richardson_lucy


KernelSpec = Union[int, Tuple[int, int], Iterable[int]]


def _normalize_kernel_size(spec: KernelSpec) -> Tuple[int, int]:
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("Kernel size must be positive.")
        return (spec, spec)

    try:
        values = tuple(int(v) for v in spec)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - invalid types
        raise ValueError("Unsupported kernel size specification.") from exc

    if len(values) == 0:
        raise ValueError("Kernel size can not be empty.")
    if len(values) == 1:
        return _normalize_kernel_size(values[0])

    h, w = values[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Kernel dimensions must be positive.")
    return (h, w)


class TmodrzykRichardsonLucyPython(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: KernelSpec = 15,
        outer_iterations: int = 10,
        image_iterations: int = 5,
        kernel_iterations: int = 5,
        clip: bool = True,
        filter_epsilon: float = 1e-10,
        use_total_variation: bool = False,
        device: str | None = None,
    ) -> None:
        super().__init__('RichardsonLucyTorchV2')
        self.kernel_shape = _normalize_kernel_size(kernel_size)
        self.outer_iterations = int(outer_iterations)
        self.image_iterations = int(image_iterations)
        self.kernel_iterations = int(kernel_iterations)
        self.clip = bool(clip)
        self.filter_epsilon = float(filter_epsilon)
        self.use_total_variation = bool(use_total_variation)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_shape = _normalize_kernel_size(param['kernel_size'])
        if 'kernel_shape' in param and param['kernel_shape'] is not None:
            self.kernel_shape = _normalize_kernel_size(param['kernel_shape'])
        if 'outer_iterations' in param and param['outer_iterations'] is not None:
            self.outer_iterations = int(param['outer_iterations'])
        if 'iterations' in param and param['iterations'] is not None:
            self.outer_iterations = int(param['iterations'])
        if 'image_iterations' in param and param['image_iterations'] is not None:
            self.image_iterations = int(param['image_iterations'])
        if 'kernel_iterations' in param and param['kernel_iterations'] is not None:
            self.kernel_iterations = int(param['kernel_iterations'])
        if 'clip' in param and param['clip'] is not None:
            self.clip = bool(param['clip'])
        if 'filter_epsilon' in param and param['filter_epsilon'] is not None:
            self.filter_epsilon = float(param['filter_epsilon'])
        if 'use_total_variation' in param and param['use_total_variation'] is not None:
            self.use_total_variation = bool(param['use_total_variation'])
        if 'tv' in param and param['tv'] is not None:
            self.use_total_variation = bool(param['tv'])
        if 'device' in param and param['device']:
            self.device = str(param['device'])

        return super().change_param(param)

    def _initial_kernel(self) -> torch.Tensor:
        h, w = self.kernel_shape
        kernel = np.ones((h, w), dtype=np.float32)
        kernel /= kernel.sum()
        return torch.from_numpy(kernel)[None, None, ...]

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start = time()
        original_dtype = image.dtype
        arr = image.astype(np.float32, copy=False)

        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError("Unsupported image shape for Richardson-Lucy algorithm.")

        scale = 255.0 if arr.max(initial=0.0) > 1.5 else 1.0
        arr = arr / scale
        arr = np.clip(arr, 0.0, 1.0)

        obs = torch.from_numpy(arr[None, ...])
        device = self.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'

        obs = obs.to(device)
        x0 = obs.clone()
        k0 = self._initial_kernel().to(device)

        restored_tensor, kernel_tensor = blind_richardson_lucy(
            observation=obs,
            x_0=x0,
            k_0=k0,
            steps=self.outer_iterations,
            x_steps=self.image_iterations,
            k_steps=self.kernel_iterations,
            clip=self.clip,
            filter_epsilon=self.filter_epsilon,
            tv=self.use_total_variation,
        )

        self.timer = time() - start

        restored = restored_tensor.detach().cpu().numpy()
        kernel = kernel_tensor.detach().cpu().numpy()

        restored = np.clip(restored, 0.0, 1.0)
        restored = restored.squeeze(0)
        if restored.shape[0] in (1, 3):
            if image.ndim == 2:
                restored = restored[0]
            else:
                restored = np.transpose(restored, (1, 2, 0))

        if np.issubdtype(original_dtype, np.integer):
            restored_out = (restored * 255.0).round().astype(original_dtype)
        else:
            restored_out = restored.astype(original_dtype, copy=False)

        kernel = kernel.squeeze()
        if kernel.ndim == 0:
            kernel = np.array([[float(kernel)]], dtype=np.float32)
        elif kernel.ndim == 1:
            kernel = kernel[np.newaxis, :]

        kernel = np.asarray(kernel, dtype=np.float32)
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel /= kernel_sum

        return restored_out, kernel

    def get_param(self):
        return [
            ('kernel_shape', self.kernel_shape),
            ('outer_iterations', self.outer_iterations),
            ('image_iterations', self.image_iterations),
            ('kernel_iterations', self.kernel_iterations),
            ('clip', self.clip),
            ('filter_epsilon', self.filter_epsilon),
            ('use_total_variation', self.use_total_variation),
            ('device', self.device),
        ]

__all__ = ["TmodrzykRichardsonLucyPython"]
