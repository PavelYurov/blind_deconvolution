from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Dict, Literal, Optional, Tuple
from .source import SBD
from .source.model import ActivationNet,LISTA

import numpy as np

from ..base import DeconvolutionAlgorithm

import torch

SOURCE_ROOT = Path(__file__).resolve().parent / "source"

@dataclass
class SBDModelConfig:
    model: Literal['lista', 'cnn'] = 'lista'
    use_topography: bool = False
    trained_weights: Optional[Path] = None
    crop_kernel: bool = True


class DrorharushSBD(DeconvolutionAlgorithm):
    """Neural sparse blind deconvolution (Drorharush/SBD)."""

    def __init__(
        self,
        model: Literal['lista', 'cnn'] = 'lista',
        use_topography: bool = False,
        trained_weights: Optional[str | Path] = None,
        crop_kernel: bool = True,
        kernel_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__('SBDNeural')
        self.config = SBDModelConfig(
            model=model,
            use_topography=use_topography,
            trained_weights=Path(trained_weights) if trained_weights else None,
            crop_kernel=bool(crop_kernel),
        )
        self.kernel_size = kernel_size
        self._net = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'model' in param and param['model']:
            self.config.model = param['model']
            self._net = None  # force reload
        if 'use_topography' in param and param['use_topography'] is not None:
            self.config.use_topography = bool(param['use_topography'])
        if 'trained_weights' in param and param['trained_weights']:
            self.config.trained_weights = Path(param['trained_weights'])
            self._net = None
        if 'crop_kernel' in param and param['crop_kernel'] is not None:
            self.config.crop_kernel = bool(param['crop_kernel'])
        if 'kernel_size' in param and param['kernel_size'] is not None:
            size = param['kernel_size']
            if isinstance(size, (tuple, list)) and len(size) >= 2:
                self.kernel_size = (int(size[0]), int(size[1]))
        return super().change_param(param)

    def process(self, image: np.ndarray):
        if torch is None:
            raise ImportError("DrorharushSBD requires PyTorch to load the trained networks.")

        national_kernel_size = self.kernel_size or (max(8, image.shape[0] // 16), max(8, image.shape[1] // 16))

        density = SBD.normalize_measurement(image)
        if density.ndim == 2:
            density = density[np.newaxis, ...]
        measurement = SBD.Measurement(
            density_of_states=density,
            kernel_size=national_kernel_size,
        )

        activation = self._predict_activation(measurement)
        kernel = SBD.measurement_to_ker(measurement, activation)
        if self.config.crop_kernel:
            kernel = SBD.crop_to_center(kernel, measurement.kernel_size)

        restored = activation
        self.timer = 0.0
        return restored, kernel

    def get_param(self):
        return [
            ('model', self.config.model),
            ('use_topography', self.config.use_topography),
            ('trained_weights', str(self.config.trained_weights) if self.config.trained_weights else None),
            ('crop_kernel', self.config.crop_kernel),
            ('kernel_size', self.kernel_size),
        ]

    # ------------------------------------------------------------------
    def _predict_activation(self, measurement):
        if self._net is None:
            self._net = self._load_network(SBD)
        net = self._net
        net.eval()
        net.cpu()
        net.double()

        dos = measurement.topography if self.config.use_topography else measurement.density_of_states
        if dos.ndim == 3:
            activations = []
            for level in range(dos.shape[0]):
                temp_mes = dos[level][np.newaxis, :, :]
                measurement_tensor = SBD.ndarray_to_tensor(temp_mes)
                with torch.no_grad():
                    temp_act = net(measurement_tensor)[0][0].cpu().numpy()
                temp_act = np.clip(temp_act, 0.0, None)
                total = temp_act.sum()
                if total > 0:
                    temp_act /= total
                activations.append(temp_act)
            activation = SBD.combine_activation_maps(np.stack(activations, axis=0))
        else:
            measurement_tensor = SBD.ndarray_to_tensor(np.expand_dims(dos, 0))
            with torch.no_grad():
                activation = net(measurement_tensor)[0][0].cpu().numpy()

        activation = np.clip(activation, 0.0, None)
        max_val = activation.max(initial=0.0)
        if max_val > 0:
            activation[activation < max_val / 100.0] = 0.0
            activation /= activation.sum() if activation.sum() > 0 else 1.0
        return activation

    def _load_network(self, SBD):
        if self.config.model == 'cnn':
            net = ActivationNet()
            default_path = SOURCE_ROOT / 'trained_model_norm.pt'
        elif self.config.model == 'lista':
            net = LISTA(5, iter_num=10)
            default_path = SOURCE_ROOT / 'trained_lista_5layers.pt'
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported model '{self.config.model}'")

        weights_path = self.config.trained_weights or default_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Could not find trained weights at {weights_path}. Provide the path via 'trained_weights'."
            )
        state_dict = torch.load(weights_path, map_location='cpu')
        net.load_state_dict(state_dict)
        return net

__all__ = ["SBDModelConfig", "DrorharushSBD"]
