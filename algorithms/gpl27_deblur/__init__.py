from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

_SOURCE_DIR = Path(__file__).resolve().parent / 'source'
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

try:  # pragma: no cover - optional dependency
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

from ..base import DeconvolutionAlgorithm
from .source import convolve, deblur


def _ensure_results_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


def _safe_write_image(path: str, image: np.ndarray, transform=None) -> None:
    _ensure_results_dir(path)
    # cv2.imwrite silently fails if the directory is missing; keeping behaviour consistent
    if cv2 is None:
        return
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(path, image)


def _as_float_image(image: np.ndarray) -> Tuple[np.ndarray, np.dtype]:
    orig_dtype = image.dtype
    work_image = image.astype(np.float64, copy=False)
    if np.issubdtype(orig_dtype, np.integer):
        return work_image, orig_dtype
    if work_image.max() <= 1.0:
        work_image *= 255.0
    return work_image, orig_dtype


def _to_original_dtype(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        return np.clip(image, 0, 255).astype(dtype)
    scaled = image / 255.0
    return scaled.astype(dtype)


deblur.write_image = _safe_write_image  # type: ignore[attr-defined]


class Gpl27DeblurDeconvolutionAlgorithm(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (21, 21),
        initial_angle: float = -np.pi / 4,
        initial_scale: float = 1.0,
        max_iterations: int = 1,
        inner_iterations: int = 1,
        gamma_start: float = 2.0,
        gamma_multiplier: float = 2.0,
        lambda1: float = 0.5,
        lambda2: float = 25.0,
        lambda1_decay: float = 1.1,
        lambda2_decay: float = 1.5,
        local_prior_threshold: float = 5.0,
        n_rows: int = 64,
        kernel_cut_ratio: float = 1e-5,
    ) -> None:
        super().__init__("High-Quality Motion Deblurring (gpl27)")
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.initial_angle = float(initial_angle)
        self.initial_scale = float(initial_scale)
        self.max_iterations = int(max_iterations)
        self.inner_iterations = max(1, int(inner_iterations))
        self.gamma_start = float(gamma_start)
        self.gamma_multiplier = float(gamma_multiplier)
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda1_decay = max(lambda1_decay, 1.0)
        self.lambda2_decay = max(lambda2_decay, 1.0)
        self.local_prior_threshold = float(local_prior_threshold)
        self.n_rows = max(1, int(n_rows))
        self.kernel_cut_ratio = float(kernel_cut_ratio)

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)
        for key, value in param.items():
            if value is None:
                continue
            if key == "kernel_size":
                if isinstance(value, Iterable):
                    seq = list(value)
                    if len(seq) == 2:
                        self.kernel_size = (int(seq[0]), int(seq[1]))
                    continue
                size = int(value)
                self.kernel_size = (size, size)
            elif key in {"initial_angle", "initial_scale", "gamma_start", "gamma_multiplier", "lambda1", "lambda2", "lambda1_decay", "lambda2_decay", "local_prior_threshold", "kernel_cut_ratio"}:
                setattr(self, key, float(value))
            elif key in {"max_iterations", "inner_iterations", "n_rows"}:
                setattr(self, key, max(1, int(value)))
        return super().change_param(param)

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_size", self.kernel_size),
            ("initial_angle", self.initial_angle),
            ("initial_scale", self.initial_scale),
            ("max_iterations", self.max_iterations),
            ("inner_iterations", self.inner_iterations),
            ("gamma_start", self.gamma_start),
            ("gamma_multiplier", self.gamma_multiplier),
            ("lambda1", self.lambda1),
            ("lambda2", self.lambda2),
            ("lambda1_decay", self.lambda1_decay),
            ("lambda2_decay", self.lambda2_decay),
            ("local_prior_threshold", self.local_prior_threshold),
            ("n_rows", self.n_rows),
            ("kernel_cut_ratio", self.kernel_cut_ratio),
        ]

    def _init_kernel(self) -> np.ndarray:
        psf = convolve.create_line_psf(self.initial_angle, self.initial_scale, self.kernel_size)
        if psf.sum() == 0:
            psf = np.ones(self.kernel_size, dtype=np.float64)
        return psf.astype(np.float64, copy=False)

    def _prepare_local_prior(self, image: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
        if cv2 is None:
            return np.ones(image.shape[:2], dtype=np.uint8)
        k_h, k_w = int(max(1, kernel_shape[0])), int(max(1, kernel_shape[1]))
        if k_h == 1 and k_w == 1:
            return np.ones(image.shape[:2], dtype=np.uint8)
        img32 = image.astype(np.float32, copy=False)
        mean = cv2.blur(img32, (k_w, k_h))
        sq_mean = cv2.blur(img32 * img32, (k_w, k_h))
        variance = np.maximum(sq_mean - mean * mean, 0.0)
        std_dev = np.sqrt(variance).astype(np.float32, copy=False)
        mask = (std_dev < self.local_prior_threshold).astype(np.uint8)
        return mask

    def _resolve_n_rows(self, image_shape: Tuple[int, int]) -> int:
        limit = max(1, min(image_shape) // 2 - 1)
        return max(1, min(self.n_rows, limit))

    def process(self, image: np.ndarray):
        if image.ndim not in {2, 3}:
            raise ValueError("image must be 2D or 3D numpy array")
        work_image, orig_dtype = _as_float_image(image)
        if work_image.ndim == 2:
            work_image = work_image[..., None]
        image_f = np.ascontiguousarray(work_image)

        kernel = self._init_kernel()
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        gamma = self.gamma_start

        latent = image_f.copy()
        blurred = image_f.copy()
        prior_masks = np.zeros_like(image_f)
        for idx in range(image_f.shape[2]):
            prior_masks[:, :, idx] = self._prepare_local_prior(image_f[:, :, idx], kernel.shape)

        observed_gradients = [np.gradient(blurred[:, :, idx], axis=(1, 0)) for idx in range(image_f.shape[2])]

        for _ in range(max(1, self.max_iterations)):
            gamma = self.gamma_start
            for _ in range(max(1, self.inner_iterations)):
                for idx in range(image_f.shape[2]):
                    latent_deriv = np.gradient(latent[:, :, idx], axis=(1, 0))
                    new_grad = deblur.updatePsi(
                        observed_gradients[idx],
                        latent_deriv,
                        prior_masks[:, :, idx],
                        lambda1,
                        lambda2,
                        gamma,
                    )
                    latent[:, :, idx] = deblur.computeL(
                        latent[:, :, idx],
                        blurred[:, :, idx],
                        kernel,
                        new_grad,
                        gamma,
                    )
                gamma *= self.gamma_multiplier

            n_rows = self._resolve_n_rows(image_f.shape[:2])
            kernel = deblur.updatef(latent, blurred, kernel, n_rows=n_rows, k_cut_ratio=self.kernel_cut_ratio)
            lambda1 /= self.lambda1_decay
            lambda2 /= self.lambda2_decay

        restored = latent.squeeze()
        if restored.ndim == 2:
            restored_image = _to_original_dtype(restored, orig_dtype)
        else:
            restored_image = _to_original_dtype(restored, orig_dtype)
        return restored_image, kernel


__all__ = ["Gpl27DeblurDeconvolutionAlgorithm"]
