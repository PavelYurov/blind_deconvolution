from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..base import DeconvolutionAlgorithm

SOURCE_ROOT = Path(__file__).resolve().parent / "source"

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

iterative_scheme = None  # type: ignore[assignment]
P4IP_Net = None  # type: ignore[assignment]
P4IP_Denoiser = None  # type: ignore[assignment]
_iterative_module = None

if torch is not None:
    import importlib

    _iterative_module = importlib.import_module(
        'algorithms.sanghviyashiitb_photon_limited_blind.source.utils.iterative_scheme'
    )
    iterative_scheme = _iterative_module.iterative_scheme  # type: ignore[assignment]
    from .source.models.network_p4ip import P4IP_Net  # type: ignore[attr-defined]
    from .source.models.network_p4ip_denoiser import P4IP_Denoiser  # type: ignore[attr-defined]

KernelSpec = Union[int, Sequence[int]]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    if image.ndim == 3:
        return image.mean(axis=2)
    raise ValueError("Unsupported image shape for photon-limited deconvolution.")


def _coerce_kernel_size(size: KernelSpec) -> int:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("Kernel size must be positive.")
        return size
    values = tuple(int(v) for v in size)
    if len(values) == 0:
        raise ValueError("Kernel size specification can not be empty.")
    return max(1, values[0])


@dataclass
class _PhotonModelPaths:
    solver: Path
    denoiser: Path

    @staticmethod
    def with_defaults(base: Path) -> "_PhotonModelPaths":
        return _PhotonModelPaths(
            solver=base / "model_zoo" / "p4ip_100epoch.pth",
            denoiser=base / "model_zoo" / "denoiser_p4ip_100epoch.pth",
        )


class SanghviyashiitbPhotonLimitedBlind(DeconvolutionAlgorithm):
    """Wrapper for the photon-limited Poisson blind deconvolution pipeline (sanghviyashiitb/photon-limited-blind).

    This adapter requires PyTorch with CUDA support and the pretrained checkpoint files supplied by the original
    repository. Provide ``solver_weights`` and ``denoiser_weights`` if the defaults are not available locally.
    """

    def __init__(
        self,
        photon_level: float = 20.0,
        kernel_size: KernelSpec = 35,
        solver_weights: Union[str, Path, None] = None,
        denoiser_weights: Union[str, Path, None] = None,
        mode: str = "symmetric",
        max_iterations: int = 150,
        tolerance: float = 1e-3,
        step_size: float = 2.0,
        update_rho: bool = True,
        use_kernel_estimate: bool = True,
        verbose: bool = False,
    ) -> None:
        super().__init__('PhotonLimitedP4IP')
        self.photon_level = float(photon_level)
        self.kernel_size = _coerce_kernel_size(kernel_size)
        default_paths = _PhotonModelPaths.with_defaults(SOURCE_ROOT)
        self.solver_weights = Path(solver_weights) if solver_weights else default_paths.solver
        self.denoiser_weights = Path(denoiser_weights) if denoiser_weights else default_paths.denoiser
        self.mode = mode
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.step_size = float(step_size)
        self.update_rho = bool(update_rho)
        self.use_kernel_estimate = bool(use_kernel_estimate)
        self.verbose = bool(verbose)

        self._p4ip = None
        self._denoiser = None
        self._device = None

    # ------------------------------------------------------------------
    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'photon_level' in param and param['photon_level'] is not None:
            self.photon_level = float(param['photon_level'])
        if 'kernel_size' in param and param['kernel_size'] is not None:
            self.kernel_size = _coerce_kernel_size(param['kernel_size'])
        if 'mode' in param and param['mode']:
            self.mode = str(param['mode'])
        if 'max_iterations' in param and param['max_iterations'] is not None:
            self.max_iterations = int(param['max_iterations'])
        if 'tolerance' in param and param['tolerance'] is not None:
            self.tolerance = float(param['tolerance'])
        if 'step_size' in param and param['step_size'] is not None:
            self.step_size = float(param['step_size'])
        if 'update_rho' in param and param['update_rho'] is not None:
            self.update_rho = bool(param['update_rho'])
        if 'use_kernel_estimate' in param and param['use_kernel_estimate'] is not None:
            self.use_kernel_estimate = bool(param['use_kernel_estimate'])
        if 'verbose' in param and param['verbose'] is not None:
            self.verbose = bool(param['verbose'])
        if 'solver_weights' in param and param['solver_weights']:
            self.solver_weights = Path(param['solver_weights'])
        if 'denoiser_weights' in param and param['denoiser_weights']:
            self.denoiser_weights = Path(param['denoiser_weights'])

        return super().change_param(param)

    # ------------------------------------------------------------------
    def process(self, image: np.ndarray):
        if torch is None or iterative_scheme is None or P4IP_Net is None:
            raise ImportError(
                "SanghviyashiitbPhotonLimitedBlind requires PyTorch and the upstream repository modules."
            )
        if not torch.cuda.is_available():  # pragma: no cover - runtime guard
            raise RuntimeError(
                "SanghviyashiitbPhotonLimitedBlind currently supports only CUDA-capable environments (torch.cuda available)."
            )

        self._ensure_models()

        grayscale = _to_grayscale(image)
        original_dtype = grayscale.dtype
        scale = 255.0 if grayscale.max(initial=0.0) > 1.5 else 1.0
        normalized = grayscale.astype(np.float32, copy=False) / scale
        normalized = np.clip(normalized, 0.0, 1.0)
        photon_counts = normalized * self.photon_level

        opts = {
            'MODE': self.mode,
            'K_N': self.kernel_size,
            'MAX_ITERS': self.max_iterations,
            'TOL': self.tolerance,
            'STEP_SIZE': self.step_size,
            'UPDATE_RHO': self.update_rho,
            'USE_KERNEL_EST': self.use_kernel_estimate,
            'VERBOSE': self.verbose,
        }

        # Align the iterative module device selection
        _iterative_module.device = self._device  # type: ignore[assignment]

        x_history, k_history = iterative_scheme(  # type: ignore[misc]
            photon_counts,
            self.photon_level,
            self._p4ip,
            self._denoiser,
            opts,
        )

        if not x_history or not k_history:
            raise RuntimeError("Iterative scheme returned empty results.")

        restored = np.clip(x_history[-1], 0.0, 1.0)
        kernel = np.array(k_history[-1], dtype=np.float32)

        restored = (restored * scale).astype(original_dtype, copy=False)

        return restored, kernel

    # ------------------------------------------------------------------
    def get_param(self):
        return [
            ('photon_level', self.photon_level),
            ('kernel_size', self.kernel_size),
            ('mode', self.mode),
            ('max_iterations', self.max_iterations),
            ('tolerance', self.tolerance),
            ('step_size', self.step_size),
            ('update_rho', self.update_rho),
            ('use_kernel_estimate', self.use_kernel_estimate),
            ('verbose', self.verbose),
            ('solver_weights', str(self.solver_weights)),
            ('denoiser_weights', str(self.denoiser_weights)),
        ]

    # ------------------------------------------------------------------
    def _ensure_models(self):
        if self._p4ip is not None and self._denoiser is not None:
            return

        if not self.solver_weights.exists():
            raise FileNotFoundError(
                f"Solver weights not found at {self.solver_weights}. Provide the pretrained checkpoint path."
            )
        if not self.denoiser_weights.exists():
            raise FileNotFoundError(
                f"Denoiser weights not found at {self.denoiser_weights}. Provide the pretrained checkpoint path."
            )

        device = torch.device('cuda')
        self._device = device

        p4ip = P4IP_Net()  # type: ignore[call-arg]
        p4ip.load_state_dict(torch.load(self.solver_weights, map_location=device))
        p4ip.to(device)
        p4ip.eval()

        denoiser = P4IP_Denoiser()  # type: ignore[call-arg]
        denoiser.load_state_dict(torch.load(self.denoiser_weights, map_location=device))
        denoiser.to(device)
        denoiser.eval()

        self._p4ip = p4ip
        self._denoiser = denoiser

__all__ = ["SanghviyashiitbPhotonLimitedBlind"]
