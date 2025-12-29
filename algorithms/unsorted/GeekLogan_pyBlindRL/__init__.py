# https://github.com/GeekLogan/pyBlindRL
from __future__ import annotations
from contextlib import contextmanager
from time import time
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch

from algorithms.base import DeconvolutionAlgorithm

try:
    from .source.src.pyBlindRL.commands import (
        RL_deconv_blind,
        generate_initial_psf_smaller,
        normalize_psf,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "Could not import pyBlindRL.commands from the wrapped repository."
    ) from exc

KernelShape = Union[int, Tuple[int, int], Tuple[int, int, int], Iterable[int]]


def _to_3d_kernel_shape(spec: KernelShape) -> Tuple[int, int, int]:
    if isinstance(spec, int):
        if spec <= 0:
            raise ValueError("Kernel dimension must be positive.")
        return (spec, spec, spec)

    try:
        values = tuple(int(v) for v in spec)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Unsupported kernel shape specification.") from exc

    if len(values) == 0:
        raise ValueError("Kernel shape can not be empty.")
    if len(values) == 1:
        return _to_3d_kernel_shape(values[0])
    if len(values) == 2:
        h, w = values
        return (1, h, w)

    z, h, w = values[:3]
    if min(z, h, w) <= 0:
        raise ValueError("Kernel shape entries must be positive.")
    return (z, h, w)


@contextmanager
def _safe_cuda_memory(device: str):
    if device.startswith('cuda'):
        yield
        return

    original = torch.cuda.memory_allocated

    def _dummy(_device: Any = None):  # pragma: no cover - runtime guard
        return 0

    torch.cuda.memory_allocated = _dummy
    try:
        yield
    finally:
        torch.cuda.memory_allocated = original


@contextmanager
def _silence_tqdm():
    import tqdm

    original_trange = tqdm.trange

    def _patched_trange(*args, **kwargs):
        kwargs.setdefault('disable', True)
        return original_trange(*args, **kwargs)

    tqdm.trange = _patched_trange
    try:
        yield
    finally:
        tqdm.trange = original_trange


class GeekLoganPyBlindRL(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: KernelShape = (1, 25, 25),
        iterations: int = 10,
        rl_iterations: int = 5,
        eps: float = 1e-9,
        reg_factor: float = 0.01,
        device: str | None = None,
    ) -> None:
        super().__init__('RichardsonLucyTorch')
        self.kernel_shape = _to_3d_kernel_shape(kernel_shape)
        self.iterations = int(iterations)
        self.rl_iterations = int(rl_iterations)
        self.eps = float(eps)
        self.reg_factor = float(reg_factor)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'kernel_shape' in param and param['kernel_shape'] is not None:
            self.kernel_shape = _to_3d_kernel_shape(param['kernel_shape'])
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_shape = _to_3d_kernel_shape(param['kernel_size'])
        if 'iterations' in param and param['iterations'] is not None:
            self.iterations = int(param['iterations'])
        if 'rl_iterations' in param and param['rl_iterations'] is not None:
            self.rl_iterations = int(param['rl_iterations'])
        if 'eps' in param and param['eps'] is not None:
            self.eps = float(param['eps'])
        if 'reg_factor' in param and param['reg_factor'] is not None:
            self.reg_factor = float(param['reg_factor'])
        if 'device' in param and param['device']:
            self.device = str(param['device'])

        return super().change_param(param)

    def process(self, image: np.ndarray):
        original_dtype = image.dtype
        start = time()
        if image.ndim == 3 and image.shape[2] not in (1, self.kernel_shape[0]):
            working = image.mean(axis=2)
        else:
            working = image.squeeze()
        working = working.astype(np.float32, copy=False)
        if working.size and float(working.max()) > 1.5:
            working = working / 255.0
        if working.ndim == 2:
            vol = working[None, ...]
        else:
            vol = working
        vol_tensor = torch.from_numpy(vol.astype(np.float32))
        out_tensor = vol_tensor.clone()
        psf_guess = generate_initial_psf_smaller(vol_tensor.numpy(), self.kernel_shape)
        psf_guess = normalize_psf(psf_guess)
        psf_tensor = torch.from_numpy(psf_guess.astype(np.float32))

        device = self.device
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'

        with _safe_cuda_memory(device), _silence_tqdm():
            restored_vol, kernel_vol, _ = RL_deconv_blind(
                vol_tensor,
                out_tensor,
                psf_tensor,
                iterations=self.iterations,
                rl_iter=self.rl_iterations,
                eps=self.eps,
                reg_factor=self.reg_factor,
                target_device=device,
            )

        self.timer = time() - start
        restored = np.asarray(restored_vol, dtype=np.float32)
        kernel = np.asarray(kernel_vol, dtype=np.float32)
        restored = np.clip(restored, 0.0, 1.0)
        restored2d = restored.squeeze()
        kernel2d = kernel.squeeze()
        if np.issubdtype(original_dtype, np.integer):
            restored_out = (restored2d * 255.0).round().astype(original_dtype)
        else:
            restored_out = restored2d.astype(original_dtype, copy=False)
        if image.ndim == 3 and image.shape[2] > 1:
            restored_out = np.repeat(restored_out[..., None], image.shape[2], axis=2)

        return restored_out, kernel2d

    def get_param(self):
        return [
            ('kernel_shape', self.kernel_shape),
            ('iterations', self.iterations),
            ('rl_iterations', self.rl_iterations),
            ('eps', self.eps),
            ('reg_factor', self.reg_factor),
            ('device', self.device),
        ]

__all__ = ["GeekLoganPyBlindRL"]
