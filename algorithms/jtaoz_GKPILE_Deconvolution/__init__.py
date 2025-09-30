from __future__ import annotations

import math
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import DeconvolutionAlgorithm

SOURCE_ROOT = Path(__file__).resolve().parent / "source"
try:  # optional heavy dependencies
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

_utils_module = None
_networks_module = None
_ssim_module = None


def _lazy_import_utils():
    global _utils_module, _networks_module, _ssim_module
    if _utils_module is not None:
        return _utils_module
    if torch is None:
        raise ImportError("PyTorch is required for GKPILE wrapper.")

    from .source.utils import common_utils as _common_utils
    from .source.networks import knet as _knet
    from .source.networks import skip as _skip
    from .source import SSIM as _ssim

    _utils_module = _common_utils
    _networks_module = {
        'skip': _skip.skip,
        'Generator': _knet.Generator,
        'ResNet18': _knet.ResNet18,
    }
    _ssim_module = _ssim.SSIM
    return _utils_module


class JtaozGKPILEDeconvolution(DeconvolutionAlgorithm):
    """Generative kernel prior blind deconvolution (GKPILE)."""

    def __init__(
        self,
        kernel_size: int = 21,
        models_dir: Optional[str | Path] = None,
        num_iterations: int = 1000,
        reg_noise_std: float = 1e-3,
        learning_rate: float = 0.01,
        weight_lr: float = 5e-4,
        milestone1: int = 2000,
        milestone2: int = 3000,
        milestone3: int = 4000,
        mse_switch: int = 500,
        device: Optional[str] = None,
    ) -> None:
        super().__init__('GKPILE')
        self.kernel_size = int(kernel_size)
        self.models_dir = Path(models_dir) if models_dir else SOURCE_ROOT / 'models'
        self.num_iterations = int(num_iterations)
        self.reg_noise_std = float(reg_noise_std)
        self.learning_rate = float(learning_rate)
        self.weight_lr = float(weight_lr)
        self.milestone1 = int(milestone1)
        self.milestone2 = int(milestone2)
        self.milestone3 = int(milestone3)
        self.mse_switch = int(mse_switch)
        self.device_override = device

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = int(param['kernel_size'])
        if 'models_dir' in param and param['models_dir']:
            self.models_dir = Path(param['models_dir'])
        if 'num_iterations' in param and param['num_iterations'] is not None:
            self.num_iterations = int(param['num_iterations'])
        if 'reg_noise_std' in param and param['reg_noise_std'] is not None:
            self.reg_noise_std = float(param['reg_noise_std'])
        if 'learning_rate' in param and param['learning_rate'] is not None:
            self.learning_rate = float(param['learning_rate'])
        if 'weight_lr' in param and param['weight_lr'] is not None:
            self.weight_lr = float(param['weight_lr'])
        if 'milestone1' in param and param['milestone1'] is not None:
            self.milestone1 = int(param['milestone1'])
        if 'milestone2' in param and param['milestone2'] is not None:
            self.milestone2 = int(param['milestone2'])
        if 'milestone3' in param and param['milestone3'] is not None:
            self.milestone3 = int(param['milestone3'])
        if 'mse_switch' in param and param['mse_switch'] is not None:
            self.mse_switch = int(param['mse_switch'])
        return super().change_param(param)

    def process(self, image: np.ndarray):
        if torch is None:
            raise ImportError('JtaozGKPILEDeconvolution requires PyTorch.')
        utils = _lazy_import_utils()
        skip = _networks_module['skip']
        Generator = _networks_module['Generator']
        ResNet18 = _networks_module['ResNet18']
        SSIM = _ssim_module

        device = torch.device(self.device_override or ('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = torch.float32

        start = time()
        arr = image.astype(np.float32, copy=False)
        scale = 255.0 if arr.max(initial=0.0) > 1.5 else 1.0
        arr = np.clip(arr / scale, 0.0, 1.0)
        if arr.ndim == 2:
            gray = arr
            color = np.repeat(arr[..., None], 3, axis=2)
        elif arr.shape[2] == 1:
            gray = arr[..., 0]
            color = np.repeat(arr, 3, axis=2)
        else:
            color = arr
            gray = 0.114 * arr[..., 0] + 0.587 * arr[..., 1] + 0.299 * arr[..., 2]

        img_chw = color.transpose(2, 0, 1)
        blur_tensor = utils.np_to_torch(img_chw).type(dtype).to(device)
        gray_chw = gray[np.newaxis, ...]
        blur_gray_tensor = utils.np_to_torch(gray_chw).type(dtype).to(device)

        padh = self.kernel_size - 1
        padw = self.kernel_size - 1
        target_h, target_w = arr.shape[0], arr.shape[1]
        img_size = (target_h, target_w)

        kernel_dir = self.models_dir
        netG_path = kernel_dir / f'netG_{self.kernel_size}.pth'
        netE_path = kernel_dir / f'netE_{self.kernel_size}.pth'
        if not netG_path.exists() or not netE_path.exists():
            raise FileNotFoundError(
                f"Pretrained GKPILE models not found for kernel size {self.kernel_size}."
                f" Expecting {netG_path.name} and {netE_path.name} in {kernel_dir}."
            )

        netE = ResNet18().to(device)
        netE.load_state_dict(torch.load(netE_path, map_location=device))
        netE.eval()

        netG = Generator(self.kernel_size).to(device)
        netG.load_state_dict(torch.load(netG_path, map_location=device))
        netG.eval()

        input_depth = 8
        net_input = utils.get_noise(input_depth, 'noise', (target_h + padh, target_w + padw)).type(dtype).to(device)
        net_input_saved = net_input.detach().clone()

        net = skip(
            input_depth,
            3,
            num_channels_down=[128, 128, 128, 128, 128],
            num_channels_up=[128, 128, 128, 128, 128],
            num_channels_skip=[16, 16, 16, 16, 16],
            upsample_mode='bilinear',
            need_sigmoid=True,
            need_bias=True,
            pad='reflection',
            act_fun='LeakyReLU',
        ).to(device)

        with torch.no_grad():
            z = netE(blur_gray_tensor)
            w = netG.g1(z).detach()
        w.requires_grad_(True)

        mse_loss = torch.nn.MSELoss().to(device)
        ssim_metric = SSIM().to(device)

        optimizer = torch.optim.Adam(
            [{'params': net.parameters()}, {'params': [w], 'lr': self.weight_lr}],
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.milestone1, self.milestone2, self.milestone3],
            gamma=0.5,
        )

        best_img = None
        best_kernel = None

        for step in range(self.num_iterations):
            optimizer.zero_grad()

            net_input = net_input_saved + self.reg_noise_std * torch.randn_like(net_input_saved)
            out_x = net(net_input)
            out_k = netG.Gk(w)
            kernel_tensor = out_k.repeat(3, 1, 1, 1)
            out_img = F.conv2d(out_x, kernel_tensor, groups=3)

            if step < self.mse_switch:
                total_loss = mse_loss(out_img, blur_tensor)
            else:
                total_loss = 1.0 - ssim_metric(out_img, blur_tensor)

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if (step + 1) % max(1, self.num_iterations // 10) == 0 or step == self.num_iterations - 1:
                with torch.no_grad():
                    out_x_np = utils.torch_to_np(out_x).transpose(1, 2, 0)
                    out_k_np = utils.torch_to_np(out_k)[0]
                    out_k_np = np.clip(out_k_np / (out_k_np.max() + 1e-12), 0.0, 1.0)
                    current = out_x_np[padh // 2:padh // 2 + target_h, padw // 2:padw // 2 + target_w, :]
                    best_img = current
                    best_kernel = out_k_np

        if best_img is None or best_kernel is None:
            raise RuntimeError('GKPILE optimization failed to produce results.')

        restored = np.clip(best_img, 0.0, 1.0)
        kernel = np.clip(best_kernel, 0.0, None)
        s = kernel.sum()
        if s > 0:
            kernel /= s

        restored = (restored * scale).astype(image.dtype, copy=False)
        if image.ndim == 2 and restored.ndim == 3:
            restored = restored[..., 0]
        elif image.ndim == 3 and image.shape[2] == 1 and restored.ndim == 3:
            restored = restored[..., :1]

        self.timer = time() - start
        return restored, kernel.astype(np.float32)

    def get_param(self):
        return [
            ('kernel_size', self.kernel_size),
            ('models_dir', str(self.models_dir)),
            ('num_iterations', self.num_iterations),
            ('reg_noise_std', self.reg_noise_std),
            ('learning_rate', self.learning_rate),
            ('weight_lr', self.weight_lr),
            ('milestone1', self.milestone1),
            ('milestone2', self.milestone2),
            ('milestone3', self.milestone3),
            ('mse_switch', self.mse_switch),
        ]

__all__ = ["JtaozGKPILEDeconvolution"]
