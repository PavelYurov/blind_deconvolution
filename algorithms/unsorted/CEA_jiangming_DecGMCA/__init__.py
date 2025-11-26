from __future__ import annotations

from time import time
from typing import Any, Optional, Tuple

import numpy as np

from ...base import DeconvolutionAlgorithm
from .source.pyDecGMCA.algoDecG import DecGMCA

Array2D = np.ndarray


def _prepare_image(image: Array2D) -> Tuple[Array2D, np.dtype, Optional[int]]:
    original_dtype = image.dtype
    channels: Optional[int] = None

    if image.ndim == 3 and image.shape[2] != 1:
        channels = int(image.shape[2])
        working = image.mean(axis=2)
    else:
        working = np.squeeze(image)

    working = working.astype(np.float64, copy=False)
    if working.size and working.max() > 1.5:
        working = working / 255.0

    if working.ndim != 2:
        raise ValueError("DecGMCA expects a 2-D grayscale image.")

    return working, original_dtype, channels


def _stretch_to_unit(image: Array2D) -> Array2D:
    min_val = float(image.min())
    max_val = float(image.max())
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val - min_val < 1e-12:
        return np.clip(image, 0.0, 1.0)
    scaled = (image - min_val) / (max_val - min_val)
    return np.clip(scaled, 0.0, 1.0)


def _restore_dtype(restored: Array2D, original_dtype: np.dtype, channels: Optional[int]) -> Array2D:
    clipped = np.clip(restored, 0.0, 1.0)
    if np.issubdtype(original_dtype, np.integer):
        output = (clipped * 255.0).round().astype(original_dtype)
    else:
        output = clipped.astype(original_dtype, copy=False)

    if channels is not None and channels > 1:
        output = np.repeat(output[..., None], channels, axis=2)

    return output


def _normalised_kernel(kernel_size: Tuple[int, int]) -> Array2D:
    kernel = np.ones(kernel_size, dtype=np.float32)
    total = float(kernel.sum())
    if total > 0:
        kernel /= total
    return kernel


class CEAJiangmingDecGMCA(DeconvolutionAlgorithm):
    def __init__(
        self,
        sources: int = 1,
        max_iter: int = 25,
        epsilon: float = 1e-4,
        epsilon_f: float = 1e-5,
        wavelet: bool = True,
        scale: int = 4,
        mask: bool = False,
        deconv: bool = False,
        wavelet_name: str = 'starlet',
        threshold_strategy: int = 2,
        ft_plane: bool = False,
        cutoff_frequency: float = 1.0 / 32.0,
        logistic: bool = False,
        post_process: int = 0,
        post_process_iter: int = 20,
        ksig: float = 0.6,
        positivity_sources: bool = False,
        positivity_mixing: bool = False,
        kernel_size: Tuple[int, int] = (9, 9),
    ) -> None:
        super().__init__('DecGMCA')
        self.sources = int(sources)
        self.max_iter = int(max_iter)
        self.epsilon = float(epsilon)
        self.epsilon_f = float(epsilon_f)
        self.wavelet = bool(wavelet)
        self.scale = int(scale)
        self.mask = bool(mask)
        self.deconv = bool(deconv)
        self.wavelet_name = str(wavelet_name)
        self.threshold_strategy = int(threshold_strategy)
        self.ft_plane = bool(ft_plane)
        self.cutoff_frequency = float(cutoff_frequency)
        self.logistic = bool(logistic)
        self.post_process = int(post_process)
        self.post_process_iter = int(post_process_iter)
        self.ksig = float(ksig)
        self.positivity_sources = bool(positivity_sources)
        self.positivity_mixing = bool(positivity_mixing)
        if isinstance(kernel_size, (tuple, list)):
            if len(kernel_size) < 2:
                raise ValueError('kernel_size must have at least two elements.')
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            size = int(kernel_size)
            self.kernel_size = (size, size)
        self._last_kernel: Optional[Array2D] = None

    def change_param(self, param: Any):
        if not isinstance(param, dict):
            return super().change_param(param)

        def _to_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {'true', '1', 'yes', 'y'}:
                    return True
                if lowered in {'false', '0', 'no', 'n'}:
                    return False
            return bool(value)

        int_fields = ('sources', 'max_iter', 'scale', 'threshold_strategy', 'post_process', 'post_process_iter')
        float_fields = ('epsilon', 'epsilon_f', 'cutoff_frequency', 'ksig')
        bool_fields = ('wavelet', 'mask', 'deconv', 'ft_plane', 'logistic', 'positivity_sources', 'positivity_mixing')

        for key in int_fields:
            if key in param and param[key] is not None:
                setattr(self, key, int(param[key]))
        for key in float_fields:
            if key in param and param[key] is not None:
                setattr(self, key, float(param[key]))
        for key in bool_fields:
            if key in param and param[key] is not None:
                setattr(self, key, _to_bool(param[key]))

        if 'wavelet_name' in param and param['wavelet_name'] is not None:
            self.wavelet_name = str(param['wavelet_name'])
        if 'kernel_size' in param and param['kernel_size'] is not None:
            value = param['kernel_size']
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                self.kernel_size = (int(value[0]), int(value[1]))
            else:
                size = int(value)
                self.kernel_size = (size, size)
        return super().change_param(param)

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        working, original_dtype, channels = _prepare_image(image)
        start = time()

        nx, ny = working.shape
        pixels = nx * ny

        V = working.reshape(1, pixels)
        M = np.ones_like(V)

        sources = max(1, int(self.sources))
        try:
            S_est, A_est = DecGMCA(
                V,
                M,
                sources,
                nx,
                ny,
                int(self.max_iter),
                float(self.epsilon),
                float(self.epsilon_f),
                2,
                self.wavelet,
                int(self.scale),
                self.mask,
                self.deconv,
                wname=self.wavelet_name,
                thresStrtg=int(self.threshold_strategy),
                FTPlane=self.ft_plane,
                fc=float(self.cutoff_frequency),
                logistic=self.logistic,
                postProc=int(self.post_process),
                postProcImax=int(self.post_process_iter),
                Ksig=float(self.ksig),
                positivityS=self.positivity_sources,
                positivityA=self.positivity_mixing,
            )
        except Exception as exc:
            raise RuntimeError('DecGMCA execution failed') from exc

        sources_arr = np.asarray(S_est)
        if sources_arr.ndim == 3:
            restored = sources_arr[0]
        elif sources_arr.ndim == 2:
            restored = sources_arr[0].reshape(nx, ny)
        else:
            restored = sources_arr.reshape(nx, ny)

        restored = _stretch_to_unit(np.real(restored).astype(np.float64, copy=False))
        restored_out = _restore_dtype(restored, original_dtype, channels)

        self._last_kernel = _normalised_kernel(self.kernel_size)
        # self._last_kernel = self.kernel_size
        self.timer = time() - start
        return restored_out, self._last_kernel

    def get_param(self):
        return [
            ('sources', self.sources),
            ('max_iter', self.max_iter),
            ('epsilon', self.epsilon),
            ('epsilon_f', self.epsilon_f),
            ('wavelet', self.wavelet),
            ('scale', self.scale),
            ('mask', self.mask),
            ('deconv', self.deconv),
            ('wavelet_name', self.wavelet_name),
            ('threshold_strategy', self.threshold_strategy),
            ('ft_plane', self.ft_plane),
            ('cutoff_frequency', self.cutoff_frequency),
            ('logistic', self.logistic),
            ('post_process', self.post_process),
            ('post_process_iter', self.post_process_iter),
            ('ksig', self.ksig),
            ('positivity_sources', self.positivity_sources),
            ('positivity_mixing', self.positivity_mixing),
            ('kernel_size', self.kernel_size),
        ]


__all__ = ['CEAJiangmingDecGMCA']

__all__ = ["CEAJiangmingDecGMCA"]
