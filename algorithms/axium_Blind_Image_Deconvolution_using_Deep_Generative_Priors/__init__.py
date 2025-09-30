from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..base import DeconvolutionAlgorithm


# SOURCE_ROOT = Path(__file__).resolve().parent / 'source'
SOURCE_ROOT = Path('.source')

KERAS_BACKEND_ENV = 'KERAS_BACKEND'


def _ensure_source_on_path() -> None:
    resolved = str(SOURCE_ROOT)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


@dataclass
class _ModelBundle:
    image_generator: Any
    blur_decoder: Any
    latent_image_dim: int
    latent_blur_dim: int


class AxiumBlindImageDeconvolutionUsingDeepGenerativePriors(DeconvolutionAlgorithm):
    """Single-image variant of Algorithm 1 with pretrained generative priors."""

    def __init__(
        self,
        dataset: str = 'celeba',
        steps: int = 200,
        random_restarts: int = 3,
        regularizers: Tuple[float, float] = (0.01, 0.01),
        step_scale: float = 0.01,
        step_decay: float = 1000.0,
        noise_std: float = 0.01,
    ) -> None:
        super().__init__('DeepGenerativePriorsAlgorithm1')
        self.dataset = dataset.lower()
        self.steps = max(1, int(steps))
        self.random_restarts = max(1, int(random_restarts))
        self.regularizers = (float(regularizers[0]), float(regularizers[1]))
        self.step_scale = float(step_scale)
        self.step_decay = float(step_decay)
        self.noise_std = float(noise_std)
        self.image_range = (-1.0, 1.0)
        self.grad_clip = 5.0

        self._models: _ModelBundle | None = None
        self._last_kernel: np.ndarray | None = None

    def change_param(self, param: Dict[str, Any]):
        if not isinstance(param, dict):
            return

        if 'dataset' in param and param['dataset']:
            self.dataset = str(param['dataset']).lower()
        if 'steps' in param and param['steps'] is not None:
            self.steps = max(1, int(param['steps']))
        if 'random_restarts' in param and param['random_restarts'] is not None:
            self.random_restarts = max(1, int(param['random_restarts']))
        if 'regularizers' in param and param['regularizers'] is not None:
            reg = param['regularizers']
            if isinstance(reg, (tuple, list)) and len(reg) >= 2:
                self.regularizers = (float(reg[0]), float(reg[1]))
        if 'step_scale' in param and param['step_scale'] is not None:
            self.step_scale = float(param['step_scale'])
        if 'step_decay' in param and param['step_decay'] is not None:
            self.step_decay = float(param['step_decay'])
        if 'noise_std' in param and param['noise_std'] is not None:
            self.noise_std = float(param['noise_std'])

    def get_param(self):
        return [
            ('dataset', self.dataset),
            ('steps', self.steps),
            ('random_restarts', self.random_restarts),
            ('regularizers', self.regularizers),
            ('step_scale', self.step_scale),
            ('step_decay', self.step_decay),
            ('noise_std', self.noise_std),
        ]

    def _ensure_models(self) -> _ModelBundle:
        if self._models is not None:
            return self._models

        os.environ.setdefault(KERAS_BACKEND_ENV, 'tensorflow')
        _ensure_source_on_path()

        try:
            import tensorflow as tf  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ImportError('TensorFlow 2.17+ is required for AxiumBlindImageDeconvolutionUsingDeepGenerativePriors.') from exc

        from source.generators.CelebAGenerator import CelebAGenerator
        from source.generators.MotionBlurGenerator import MotionBlur

        image_gen = CelebAGenerator()
        image_gen.weights_path = str(SOURCE_ROOT / 'model weights' / 'celeba.h5')
        # image_gen.weights_path = str('source' / 'model weights' / 'celeba.h5')

        image_gen.GenerateModel()
        image_gen.LoadWeights()

        blur_gen = MotionBlur()
        blur_gen.weights_path = str(SOURCE_ROOT / 'model weights' / 'motionblur.h5')
        # blur_gen.weights_path = str('source' / 'model weights' / 'motionblur.h5')

        blur_gen.GenerateModel()
        blur_gen.LoadWeights()
        _, _, blur_decoder = blur_gen.GetModels()

        self._models = _ModelBundle(
            image_generator=image_gen.GetModels(),
            blur_decoder=blur_decoder,
            latent_image_dim=image_gen.latent_dim,
            latent_blur_dim=blur_gen.latent_dim,
        )
        return self._models

    def _prepare_observation(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...], np.dtype]:
        if image is None or image.size == 0:
            raise ValueError('Empty image provided to AxiumBlindImageDeconvolutionUsingDeepGenerativePriors.')

        original_dtype = image.dtype
        if image.ndim == 2:
            image = image[..., np.newaxis]
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        elif image.ndim == 3 and image.shape[2] == 1:
            pass
        else:
            raise ValueError(f'Unsupported image shape: {image.shape}')

        grayscale = image.astype(np.float32, copy=False)
        max_value = float(grayscale.max()) if grayscale.size else 0.0
        if max_value > 1.5:
            grayscale /= 255.0
        grayscale = np.clip(grayscale, 0.0, 1.0)

        target_res = 64 if self.dataset != 'svhn' else 32
        resized = cv2.resize(grayscale, (target_res, target_res), interpolation=cv2.INTER_AREA)
        resized = resized[..., np.newaxis]
        tiled = np.repeat(resized, repeats=3, axis=2)
        return tiled, image.shape, original_dtype

    def _step_size(self, iteration: int) -> float:
        return self.step_scale * float(np.exp(-float(iteration) / max(self.step_decay, 1e-6)))

    @staticmethod
    def _pad_kernel_tensor(kernel, target_size):
        import tensorflow as tf

        kernel = tf.cast(kernel, tf.float32)
        target_h, target_w = target_size
        k_shape = tf.shape(kernel)
        pad_h = target_h - k_shape[1]
        pad_w = target_w - k_shape[2]
        pad_top = tf.maximum(pad_h // 2, 0)
        pad_bottom = tf.maximum(pad_h - pad_top, 0)
        pad_left = tf.maximum(pad_w // 2, 0)
        pad_right = tf.maximum(pad_w - pad_left, 0)
        paddings = tf.stack([
            tf.zeros((2,), dtype=tf.int32),
            tf.stack([pad_top, pad_bottom]),
            tf.stack([pad_left, pad_right])
        ])
        padded = tf.pad(kernel, paddings, mode='CONSTANT')
        if pad_h < 0 or pad_w < 0:
            padded = padded[:, -target_h:, -target_w:]
        return padded

    def _run_solver(self, observation: np.ndarray, models: _ModelBundle) -> Tuple[np.ndarray, np.ndarray]:
        import tensorflow as tf

        rr = self.random_restarts
        latents_image = tf.Variable(tf.random.normal((rr, models.latent_image_dim), stddev=0.5, dtype=tf.float32))
        latents_blur = tf.Variable(tf.random.normal((rr, models.latent_blur_dim), stddev=0.5, dtype=tf.float32))

        y_batch = np.repeat(observation[None, ...], rr, axis=0)
        y_tensor = tf.convert_to_tensor(y_batch, dtype=tf.float32)

        target_h, target_w = observation.shape[0], observation.shape[1]
        reg_blur, reg_image = self.regularizers

        def forward(img_latent: tf.Tensor, blur_latent: tf.Tensor):
            decoded_image = models.image_generator(img_latent, training=False)
            decoded_image = tf.clip_by_value(decoded_image, self.image_range[0], self.image_range[1])
            decoded_image = (decoded_image - self.image_range[0]) / (self.image_range[1] - self.image_range[0])
            decoded_image = tf.clip_by_value(decoded_image, 0.0, 1.0)

            decoded_kernel = models.blur_decoder(blur_latent, training=False)
            decoded_kernel = tf.nn.relu(decoded_kernel)
            decoded_kernel = tf.squeeze(decoded_kernel, axis=-1)
            decoded_kernel /= tf.reduce_sum(decoded_kernel, axis=[1, 2], keepdims=True) + 1e-6

            padded_kernel = self._pad_kernel_tensor(decoded_kernel, (target_h, target_w))
            padded_kernel = tf.expand_dims(padded_kernel, axis=-1)

            x_fft = tf.signal.fft2d(tf.cast(decoded_image, tf.complex64))
            k_fft = tf.signal.fft2d(tf.cast(padded_kernel, tf.complex64))
            pred_fft = x_fft * k_fft
            predicted = tf.math.real(tf.signal.ifft2d(pred_fft))
            predicted = tf.clip_by_value(predicted, 0.0, 1.0)
            return decoded_image, decoded_kernel, predicted

        for step in range(self.steps):
            lr = self._step_size(step)
            with tf.GradientTape() as tape:
                tape.watch([latents_image, latents_blur])
                decoded_image, decoded_kernel, predicted = forward(latents_image, latents_blur)
                residual = tf.reduce_mean(tf.square(predicted - y_tensor), axis=[1, 2, 3])
                loss = tf.reduce_mean(
                    residual + reg_blur * tf.reduce_sum(latents_blur ** 2, axis=1) + reg_image * tf.reduce_sum(latents_image ** 2, axis=1)
                )
            grads = tape.gradient(loss, [latents_image, latents_blur])
            for var, grad in zip((latents_image, latents_blur), grads):
                if grad is None:
                    continue
                grad = tf.clip_by_norm(grad, self.grad_clip)
                var.assign_sub(lr * grad)

        decoded_image, decoded_kernel, predicted = forward(latents_image, latents_blur)
        final_residual = tf.reduce_mean(tf.square(predicted - y_tensor), axis=[1, 2, 3]).numpy()
        best_idx = int(np.argmin(final_residual))
        return decoded_image.numpy()[best_idx], decoded_kernel.numpy()[best_idx]

    def _postprocess(self, restored_rgb: np.ndarray, original_shape: Tuple[int, ...], original_dtype: np.dtype) -> np.ndarray:
        restored_rgb = np.clip(restored_rgb, 0.0, 1.0)
        if restored_rgb.ndim == 3 and restored_rgb.shape[2] == 3:
            restored_gray = 0.2989 * restored_rgb[..., 0] + 0.5870 * restored_rgb[..., 1] + 0.1140 * restored_rgb[..., 2]
        else:
            restored_gray = np.squeeze(restored_rgb)

        target_height, target_width = original_shape[0], original_shape[1]
        restored_resized = cv2.resize(restored_gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        restored_resized = np.clip(restored_resized, 0.0, 1.0)

        if np.issubdtype(original_dtype, np.integer):
            return np.round(restored_resized * 255.0).astype(original_dtype)
        return restored_resized.astype(original_dtype, copy=False)

    def _reshape_kernel(self, kernel: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        if kernel.ndim == 3:
            kernel = np.squeeze(kernel, axis=-1)
        kernel = np.clip(kernel, 0.0, None)
        kernel_sum = float(kernel.sum())
        if kernel_sum <= 0:
            kernel_sum = 1.0
        kernel /= kernel_sum

        target_height, target_width = original_shape[0], original_shape[1]
        pad_h = max(target_height - kernel.shape[0], 0)
        pad_w = max(target_width - kernel.shape[1], 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        kernel_padded = np.pad(kernel, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        if kernel_padded.shape[0] > target_height or kernel_padded.shape[1] > target_width:
            kernel_padded = kernel_padded[:target_height, :target_width]
        return kernel_padded.astype(np.float32)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        models = self._ensure_models()
        prepared, original_shape, original_dtype = self._prepare_observation(image)

        start = time.time()
        restored_rgb, kernel = self._run_solver(prepared, models)
        self.timer = time.time() - start

        restored = self._postprocess(restored_rgb, original_shape, original_dtype)
        kernel_out = self._reshape_kernel(kernel, original_shape)
        self._last_kernel = kernel_out
        return restored, kernel_out

    def get_kernel(self) -> np.ndarray | None:
        return self._last_kernel


__all__ = ['AxiumBlindImageDeconvolutionUsingDeepGenerativePriors']
