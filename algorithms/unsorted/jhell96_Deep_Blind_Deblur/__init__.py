# https://github.com/jhell96/Deep-Blind-Deblur
from __future__ import annotations
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import importlib.util
import functools
from types import ModuleType
import torch

from algorithms.base import DeconvolutionAlgorithm

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

MODELS_ROOT = SOURCE_ROOT / "model" / "DeblurGAN-master"
WEIGHTS_PATH = MODELS_ROOT / "checkpoints" / "experiment_name" / "latest_net_G.pth"


_NETWORKS_MODULE: Optional[ModuleType] = None

def _load_networks():
    global _NETWORKS_MODULE
    if _NETWORKS_MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "deep_blind_deblur_networks", str(MODELS_ROOT / "models" / "networks.py"))
        if spec is None or spec.loader is None:
            raise ImportError("Unable to load DeblurGAN networks module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _NETWORKS_MODULE = module
    return _NETWORKS_MODULE


def _ensure_odd(value: int) -> int:
    return int(value) | 1


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    normalized = (image * 2.0) - 1.0
    tensor = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1)
    return tensor.unsqueeze(0)


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = (array + 1.0) / 2.0
    array = np.clip(array, 0.0, 1.0)
    return array.mean(axis=2)


class Jhell96DeepBlindDeblur(DeconvolutionAlgorithm):
    def __init__(
        self,
        use_gpu: bool = False,
        kernel_size: int = 21,
    ) -> None:
        super().__init__('DeepBlindDeblur')
        self.use_gpu = bool(use_gpu and torch.cuda.is_available())
        self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
        self.kernel_size = _ensure_odd(kernel_size)
        self._model: Optional[torch.nn.Module] = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'use_gpu' in param and param['use_gpu'] is not None:
            requested = bool(param['use_gpu'])
            self.use_gpu = requested and torch.cuda.is_available()
            self.device = torch.device('cuda:0' if self.use_gpu else 'cpu')
            if self._model is not None:
                self._model.to(self.device)
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = _ensure_odd(param['kernel_size'])
        return super().change_param(param)

    def get_param(self):
        return [
            ('use_gpu', self.use_gpu),
            ('device', str(self.device)),
            ('kernel_size', self.kernel_size),
            ('weights_path_exists', WEIGHTS_PATH.exists()),
        ]

    def _ensure_model(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model
        networks = _load_networks()
        norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=True, track_running_stats=True)
        model = networks.ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm_layer=norm_layer,
            n_blocks=9,
        )
        state_dict = torch.load(WEIGHTS_PATH, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self._model = model
        return self._model

            input_nc=3,
            output_nc=3,
            ngf=64,
        )

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if image is None:
            raise ValueError('Input image is None.')
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError('Expected a 2D grayscale image.')

        original_dtype = arr.dtype
        float_img = arr.astype(np.float32, copy=False)
        if float_img.max() > 1.5:
            float_img = float_img / 255.0
        float_img = np.clip(float_img, 0.0, 1.0)

        model = self._ensure_model()
        start = time()
        with torch.no_grad():
            tensor = _to_tensor(float_img).to(self.device)
            output = model(tensor)
        restored = _to_image(output)
        self.timer = time() - start

        if np.issubdtype(original_dtype, np.integer):
            restored = np.clip(restored * 255.0, 0, 255).round().astype(original_dtype)
        else:
            restored = restored.astype(original_dtype, copy=False)

        return restored, None

    def get_kernel(self) -> np.ndarray | None:
        return None


__all__ = ['Jhell96DeepBlindDeblur']
