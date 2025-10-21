from __future__ import annotations

import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import time
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np

from ..base import DeconvolutionAlgorithm

Array2D = np.ndarray

SOURCE_ROOT = Path(__file__).resolve().parent / 'source'


def _ensure_path_on_syspath(path: Path) -> None:
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _as_tuple(value: Iterable[int] | Tuple[int, int] | int, default: Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        seq = list(value)
        if len(seq) >= 2:
            return int(seq[0]), int(seq[1])
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    scalar = int(value)
    return (scalar, scalar) if scalar > 0 else default


@dataclass
class _ModelCache:
    angle_model: Any | None = None
    length_model: Any | None = None


class AdamalaviBlindMotionDeblurringForLicensePlates(DeconvolutionAlgorithm):
    """CNN-based motion PSF estimation with Wiener deblurring."""

    ANGLE_CLASSES = 180
    LENGTH_OUTPUTS = 1

    def __init__(
        self,
        angle_model_path: str | Path | None = None,
        length_model_path: str | Path | None = None,
        noise: float = 0.01,
        psf_size: int = 200,
        inference_size: Tuple[int, int] = (640, 480),
        fft_size: Tuple[int, int] = (224, 224),
        angle_top_k: int = 3,
    ) -> None:
        super().__init__('BlindMotionCNN')
        self._source_root = SOURCE_ROOT
        self.angle_model_path = Path(angle_model_path) if angle_model_path else self._source_root / 'pretrained_models' / 'angle_model.hdf5'
        self.length_model_path = Path(length_model_path) if length_model_path else self._source_root / 'pretrained_models' / 'length_model.hdf5'

        # self.angle_model_path = Path('source\\pretrained_models\\angle_model.hdf5')
        # self.length_model_path = Path('source\\pretrained_models\\length_model.hdf5')

        self.noise = float(noise)
        self.psf_size = int(psf_size)
        self.inference_size = _as_tuple(inference_size, (640, 480))
        self.fft_size = _as_tuple(fft_size, (224, 224))
        self.angle_top_k = max(1, int(angle_top_k))

        self._models = _ModelCache()
        self._last_kernel: Array2D | None = None
        self._last_angle: float | None = None
        self._last_length: float | None = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return super().change_param(param)

        if 'angle_model_path' in param and param['angle_model_path']:
            self.angle_model_path = Path(param['angle_model_path'])
            self._models.angle_model = None
        if 'length_model_path' in param and param['length_model_path']:
            self.length_model_path = Path(param['length_model_path'])
            self._models.length_model = None
        if 'noise' in param and param['noise'] is not None:
            self.noise = float(param['noise'])
        if 'psf_size' in param and param['psf_size'] is not None:
            self.psf_size = int(param['psf_size'])
        if 'inference_size' in param and param['inference_size'] is not None:
            self.inference_size = _as_tuple(param['inference_size'], self.inference_size)
        if 'fft_size' in param and param['fft_size'] is not None:
            self.fft_size = _as_tuple(param['fft_size'], self.fft_size)
        if 'angle_top_k' in param and param['angle_top_k'] is not None:
            self.angle_top_k = max(1, int(param['angle_top_k']))

        return super().change_param(param)

    def get_param(self):
        return [
            ('angle_model_path', str(self.angle_model_path)),
            ('length_model_path', str(self.length_model_path)),
            ('noise', self.noise),
            ('psf_size', self.psf_size),
            ('inference_size', self.inference_size),
            ('fft_size', self.fft_size),
            ('angle_top_k', self.angle_top_k),
            ('last_angle', self._last_angle),
            ('last_length', self._last_length),
        ]

    def _import_builders(self):
        _ensure_path_on_syspath(self._source_root)
        angle_module = import_module('sidekick.nn.conv.angle_model')
        length_module = import_module('sidekick.nn.conv.length_model')
        return angle_module.MiniVgg, length_module.MiniVgg

    def _ensure_models(self) -> None:
        if self._models.angle_model is not None and self._models.length_model is not None:
            return

        try:
            import tensorflow  # noqa: F401  # ensure dependency is available
        except ImportError as exc:
            raise ImportError(
                'TensorFlow is required for AdamalaviBlindMotionDeblurringForLicensePlates.'
            ) from exc

        AngleVgg, LengthVgg = self._import_builders()

        fft_w, fft_h = int(self.fft_size[0]), int(self.fft_size[1])

        if self._models.angle_model is None:
            if not self.angle_model_path.exists():
                raise FileNotFoundError(f'Angle model not found at {self.angle_model_path!s}')
            model = AngleVgg.build(fft_w, fft_h, 1, self.ANGLE_CLASSES)
            model.load_weights(str(self.angle_model_path))
            model.trainable = False
            self._models.angle_model = model

        if self._models.length_model is None:
            if not self.length_model_path.exists():
                raise FileNotFoundError(f'Length model not found at {self.length_model_path!s}')
            model = LengthVgg.build(fft_w, fft_h, 1, self.LENGTH_OUTPUTS)
            model.load_weights(str(self.length_model_path))
            model.trainable = False
            self._models.length_model = model

    def _prepare_input(self, image: Array2D) -> Tuple[Array2D, np.dtype, Tuple[int, ...], Tuple[int, ...]]:
        original_dtype = image.dtype
        original_shape = image.shape

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.squeeze(image)
        grayscale_shape = gray.shape

        gray = gray.astype(np.float32, copy=False)
        max_value = float(gray.max()) if gray.size else 0.0
        if max_value > 1.5:
            gray /= 255.0
        gray = np.clip(gray, 0.0, 1.0)

        infer_w, infer_h = int(self.inference_size[0]), int(self.inference_size[1])
        resized = cv2.resize(gray, (max(infer_w, 1), max(infer_h, 1)), interpolation=cv2.INTER_AREA)
        return resized, original_dtype, original_shape, grayscale_shape

    def _predict_psf_parameters(self, fft_image: Array2D) -> Tuple[float, float]:
        self._ensure_models()

        fft_resized = cv2.resize(
            fft_image.astype(np.float32),
            (int(self.fft_size[0]), int(self.fft_size[1])),
            interpolation=cv2.INTER_AREA,
        )
        fft_resized = np.clip(fft_resized, 0.0, None)
        input_tensor = np.expand_dims(fft_resized, axis=(0, -1)).astype(np.float32) / 255.0

        angle_logits = self._models.angle_model.predict(input_tensor, verbose=0)[0]
        top_indices = np.argsort(angle_logits)[-self.angle_top_k :]
        angle_value = float(top_indices.mean())

        length_pred = self._models.length_model.predict(input_tensor, verbose=0)
        length_value = float(length_pred.reshape(-1)[0])
        return angle_value, length_value

    def _create_fft(self, image: Array2D) -> Array2D:
        image = image.astype(np.float32, copy=False)
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        spectrum = np.asarray(spectrum, dtype=np.float32)
        spectrum -= spectrum.min()
        max_val = spectrum.max()
        if max_val > 0:
            spectrum = spectrum * (255.0 / max_val)
        return spectrum

    def _run_wiener(self, image: Array2D, length: float, angle_deg: float) -> Tuple[Array2D, Array2D]:
        noise = max(1e-6, float(self.noise))
        psf_length = max(1, int(round(length)))
        psf_size = max(1, int(self.psf_size))
        angle_rad = (float(angle_deg) * np.pi) / 180.0

        psf = np.ones((1, psf_length), np.float32)
        cos_term, sin_term = np.cos(angle_rad), np.sin(angle_rad)
        affine = np.float32([[-cos_term, sin_term, 0], [sin_term, cos_term, 0]])
        half = psf_size // 2
        affine[:, 2] = (half, half) - np.dot(affine[:, :2], ((psf_length - 1) * 0.5, 0))
        psf = cv2.warpAffine(psf, affine, (psf_size, psf_size), flags=cv2.INTER_CUBIC)

        working = image.astype(np.float32, copy=False)
        gray_dft = cv2.dft(working, flags=cv2.DFT_COMPLEX_OUTPUT)
        psf /= psf.sum() + 1e-8

        psf_canvas = np.zeros_like(working, dtype=np.float32)
        h, w = working.shape[:2]
        psf_canvas[: min(psf_size, h), : min(psf_size, w)] = psf[: min(psf_size, h), : min(psf_size, w)]
        psf_dft = cv2.dft(psf_canvas, flags=cv2.DFT_COMPLEX_OUTPUT)

        psf_sq = (psf_dft ** 2).sum(-1)
        filt = psf_dft / (psf_sq + noise)[..., np.newaxis]
        result_dft = cv2.mulSpectrums(gray_dft, filt, 0)
        restored = cv2.idft(result_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        restored = np.roll(restored, -psf_size // 2, axis=0)
        restored = np.roll(restored, -psf_size // 2, axis=1)
        return restored, psf

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        print(self.angle_model_path)
        if image is None or image.size == 0:
            raise ValueError('Empty image provided to AdamalaviBlindMotionDeblurringForLicensePlates.')

        resized, original_dtype, original_shape, grayscale_shape = self._prepare_input(image)
        start = time()
        fft_image = self._create_fft(resized)
        angle_value, length_value = self._predict_psf_parameters(fft_image)
        restored_resized, kernel = self._run_wiener(resized, length_value, angle_value)
        self.timer = time() - start

        self._last_kernel = kernel
        self._last_angle = angle_value
        self._last_length = length_value

        restored_resized = restored_resized.astype(np.float32, copy=False)
        min_val = float(restored_resized.min()) if restored_resized.size else 0.0
        restored_resized -= min_val
        max_val = float(restored_resized.max()) if restored_resized.size else 0.0
        if max_val > 1e-8:
            restored_resized /= max_val

        restored_resized = cv2.resize(
            restored_resized,
            (int(grayscale_shape[1]) if len(grayscale_shape) > 1 else 1,
             int(grayscale_shape[0]) if len(grayscale_shape) > 0 else 1),
            interpolation=cv2.INTER_CUBIC,
        )

        if np.issubdtype(original_dtype, np.integer):
            restored_out = (np.clip(restored_resized, 0.0, 1.0) * 255.0).round().astype(original_dtype)
        else:
            restored_out = restored_resized.astype(original_dtype, copy=False)

        if image.ndim == 3 and image.shape[2] != 1:
            restored_out = np.repeat(restored_out[..., None], image.shape[2], axis=2)

        return restored_out, kernel

    def get_kernel(self) -> Array2D | None:
        return self._last_kernel


__all__ = ['AdamalaviBlindMotionDeblurringForLicensePlates']
