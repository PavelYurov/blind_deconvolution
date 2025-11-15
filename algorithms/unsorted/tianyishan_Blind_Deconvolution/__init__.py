from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..base import DeconvolutionAlgorithm


Array2D = np.ndarray


@dataclass
class _BlindParams:
    MK: int
    NK: int
    niters: int


@dataclass
class _CTFParams:
    lambda_multiplier: float
    max_lambda: float
    final_lambda: float
    kernel_size_multiplier: float


def _is_gray_image(image: Array2D) -> bool:
    if image.ndim == 2:
        return True
    if image.shape[2] == 1:
        return True
    b, g, r = cv2.split(image.astype(np.float64))
    return not (np.count_nonzero(cv2.absdiff(b, g)) or np.count_nonzero(cv2.absdiff(b, r)))


def _ensure_three_channels(image: Array2D) -> Array2D:
    if image.ndim == 2:
        return np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def _rotate_180(mat: Array2D) -> Array2D:
    return np.flipud(np.fliplr(mat))


def _conv2(image: Array2D, kernel: Array2D, mode: str) -> Array2D:
    if image.ndim == 3:
        channels = [
            _conv2(image[..., c], kernel, mode) for c in range(image.shape[2])
        ]
        return np.stack(channels, axis=2)

    kernel_flipped = np.flipud(np.fliplr(kernel))
    pad_y = kernel.shape[0] - 1
    pad_x = kernel.shape[1] - 1

    if mode == 'full':
        padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant')
    elif mode == 'valid':
        padded = image
    else:
        raise ValueError(f'Unsupported convolution mode: {mode}')

    filtered = cv2.filter2D(padded, cv2.CV_64F, kernel_flipped, borderType=cv2.BORDER_CONSTANT)
    start_y = pad_y // 2
    start_x = pad_x // 2
    end_y = start_y + (filtered.shape[0] - pad_y)
    end_x = start_x + (filtered.shape[1] - pad_x)
    return filtered[start_y:end_y, start_x:end_x]


def _grad_tv_cc(image: Array2D, channel_mode: int) -> Array2D:
    if image.ndim == 2:
        working = image[..., None]
    else:
        working = image

    fxforw = np.empty_like(working)
    fxforw[:-1] = working[1:]
    fxforw[-1] = working[-1]
    fxforw = fxforw - working

    fyforw = np.empty_like(working)
    fyforw[:, :-1] = working[:, 1:]
    fyforw[:, -1] = working[:, -1]
    fyforw = fyforw - working

    fxback = np.empty_like(working)
    fxback[1:] = working[:-1]
    fxback[0] = working[0]

    fyback = np.empty_like(working)
    fyback[:, 1:] = working[:, :-1]
    fyback[:, 0] = working[:, 0]

    fxmixd_base = working[1:, :-1]
    fxmixd = np.pad(fxmixd_base, ((0, 1), (1, 0), (0, 0)), mode='edge')
    fxmixd = fxmixd - fyback

    fymixd_base = working[:-1, 1:]
    fymixd = np.pad(fymixd_base, ((1, 0), (0, 1), (0, 0)), mode='edge')
    fymixd = fymixd - fxback

    fyback = working - fyback
    fxback = working - fxback

    sqtforw = np.sqrt(fxforw * fxforw + fyforw * fyforw)
    sqtmixed = np.sqrt(fxback * fxback + fymixd * fymixd)
    sqtback = np.sqrt(fxmixd * fxmixd + fyback * fyback)

    max1 = np.maximum(sqtforw, 1e-3)
    max2 = np.maximum(sqtmixed, 1e-3)
    max3 = np.maximum(sqtback, 1e-3)

    dest = (fxforw + fyforw) / max1
    dest -= fxback / max2
    dest -= fyback / max3

    if channel_mode == 1 and dest.shape[2] == 1:
        dest = np.repeat(dest, 3, axis=2)
    return dest


def _build_pyramid(
    image: Array2D,
    params: _BlindParams,
    ctf_params: _CTFParams,
) -> Dict[str, Any]:
    smallest_scale = 3
    scales = 1
    mkpnext = float(params.MK)
    nkpnext = float(params.NK)
    lamnext = float(ctf_params.final_lambda)

    M, N = image.shape[:2]

    while (
        mkpnext > smallest_scale
        and nkpnext > smallest_scale
        and lamnext * ctf_params.lambda_multiplier < ctf_params.max_lambda
    ):
        scales += 1
        lamprev = lamnext
        mkpprev = mkpnext
        nkpprev = nkpnext

        lamnext = lamprev * ctf_params.lambda_multiplier
        mkpnext = round(mkpprev / ctf_params.kernel_size_multiplier)
        nkpnext = round(nkpprev / ctf_params.kernel_size_multiplier)

        if mkpnext % 2 == 0:
            mkpnext -= 1
        if nkpnext % 2 == 0:
            nkpnext -= 1
        if nkpnext == nkpprev:
            nkpnext -= 2
        if mkpnext == mkpprev:
            mkpnext -= 2
        if nkpnext < smallest_scale:
            nkpnext = smallest_scale
        if mkpnext < smallest_scale:
            mkpnext = smallest_scale

    fp: List[Array2D] = [None] * scales
    Mp: List[int] = [0] * scales
    Np: List[int] = [0] * scales
    MKp: List[int] = [0] * scales
    NKp: List[int] = [0] * scales
    lambdas: List[float] = [0.0] * scales

    fp[0] = image
    Mp[0] = M
    Np[0] = N
    MKp[0] = int(params.MK)
    NKp[0] = int(params.NK)
    lambdas[0] = float(ctf_params.final_lambda)

    for s in range(1, scales):
        lambdas[s] = lambdas[s - 1] * ctf_params.lambda_multiplier
        MKp[s] = max(3, int(round(MKp[s - 1] / ctf_params.kernel_size_multiplier)))
        NKp[s] = max(3, int(round(NKp[s - 1] / ctf_params.kernel_size_multiplier)))

        if MKp[s] % 2 == 0:
            MKp[s] -= 1
        if NKp[s] % 2 == 0:
            NKp[s] -= 1
        if NKp[s] == NKp[s - 1]:
            NKp[s] = max(3, NKp[s] - 2)
        if MKp[s] == MKp[s - 1]:
            MKp[s] = max(3, MKp[s] - 2)

        factorM = MKp[s - 1] / MKp[s]
        factorN = NKp[s - 1] / NKp[s]

        Mp[s] = max(3, int(round(Mp[s - 1] / factorM)))
        Np[s] = max(3, int(round(Np[s - 1] / factorN)))

        if Mp[s] % 2 == 0:
            Mp[s] -= 1
        if Np[s] % 2 == 0:
            Np[s] -= 1

        fp[s] = cv2.resize(image, (Np[s], Mp[s]), interpolation=cv2.INTER_LINEAR)

    return {
        'fp': fp,
        'Mp': Mp,
        'Np': Np,
        'MKp': MKp,
        'NKp': NKp,
        'lambdas': lambdas,
        'scales': scales,
    }


def _prida(
    f: Array2D,
    u: Array2D,
    k: Array2D,
    lam: float,
    params: _BlindParams,
    channel_mode: int,
) -> Tuple[Array2D, Array2D]:
    nk = int(params.NK)
    mk = int(params.MK)
    niters = int(params.niters)

    for _ in range(niters):
        if channel_mode == 1:
            tmp = _conv2(u[..., 0], k, 'valid') - f[..., 0]
            grad_u_single = _conv2(tmp, _rotate_180(k), 'full')
            grad_u = np.repeat(grad_u_single[..., None], 3, axis=2)
        else:
            grads = []
            for c in range(f.shape[2]):
                tmp = _conv2(u[..., c], k, 'valid') - f[..., c]
                grad_c = _conv2(tmp, _rotate_180(k), 'full')
                grads.append(grad_c)
            grad_u = np.stack(grads, axis=2)

        grad_tv = _grad_tv_cc(u, channel_mode)
        grad_u = grad_u - lam * grad_tv

        max_u = float(np.max(u)) if u.size else 0.0
        max_grad_u = float(np.max(np.abs(grad_u))) if grad_u.size else 0.0
        step_u = 1e-3 * max_u / max(1e-31, max_grad_u)
        u_new = u - step_u * grad_u

        grad_k = np.zeros_like(k, dtype=np.float64)
        channels_to_use = 1 if channel_mode == 1 else f.shape[2]
        for c in range(channels_to_use):
            tmp = _conv2(u[..., c], k, 'valid') - f[..., c]
            rot_u = _rotate_180(u[..., c])
            grad_k += _conv2(rot_u, tmp, 'valid')

        max_k = float(np.max(k)) if k.size else 0.0
        max_grad_k = float(np.max(np.abs(grad_k))) if grad_k.size else 0.0
        step_k = 1e-3 * max_k / max(1e-31, max_grad_k)
        etai = step_k / (k + np.finfo(np.float64).eps)
        exp_term = np.exp(np.clip(-(etai * grad_k), -np.inf, math.log(1000.0)))
        mds = k * exp_term
        k_new = mds / max(np.sum(mds), np.finfo(np.float64).eps)

        u = u_new
        k = k_new

    return u, k


def _coarse_to_fine(
    image: Array2D,
    blind_params: _BlindParams,
    ctf_params: _CTFParams,
    channel_mode: int,
) -> Tuple[Array2D, Array2D]:
    MK = int(blind_params.MK)
    NK = int(blind_params.NK)

    top = MK // 2
    left = NK // 2
    u = cv2.copyMakeBorder(image, top, top, left, left, borderType=cv2.BORDER_REPLICATE)
    k = np.ones((MK, NK), dtype=np.float64)
    k /= np.sum(k)

    pyramid = _build_pyramid(image, blind_params, ctf_params)

    for idx in range(pyramid['scales'] - 1, -1, -1):
        Ms = pyramid['Mp'][idx]
        Ns = pyramid['Np'][idx]
        MKs = pyramid['MKp'][idx]
        NKs = pyramid['NKp'][idx]
        lam = pyramid['lambdas'][idx]
        fs = pyramid['fp'][idx]

        u = cv2.resize(u, (Ns + NKs - 1, Ms + MKs - 1), interpolation=cv2.INTER_LINEAR)
        k = cv2.resize(k, (NKs, MKs), interpolation=cv2.INTER_LINEAR)
        k = k / max(np.sum(k), np.finfo(np.float64).eps)

        blind_params.MK = MKs
        blind_params.NK = NKs
        u, k = _prida(fs, u, k, lam, blind_params, channel_mode)

    return u, k


def _blind_deconv(
    image: Array2D,
    lam: float,
    params: _BlindParams,
    channel_mode: int,
) -> Tuple[Array2D, Array2D]:
    working = image.astype(np.float64).copy()
    if working.max() > 1.5:
        working /= 255.0

    rows, cols = working.shape[:2]
    rpad = 1 if rows % 2 == 0 else 0
    cpad = 1 if cols % 2 == 0 else 0
    working = working[: rows - rpad or rows, : cols - cpad or cols]

    ctf_params = _CTFParams(
        lambda_multiplier=1.9,
        max_lambda=1.1e-1,
        final_lambda=lam,
        kernel_size_multiplier=1.1,
    )

    return _coarse_to_fine(working, params, ctf_params, channel_mode)


class TianyishanBlindDeconvolutionDeconvolutionAlgorithm(DeconvolutionAlgorithm):
    """Python translation of the PRIDA blind deconvolution algorithm."""

    def __init__(
        self,
        lam: float = 6e-4,
        kernel_size: int = 19,
        iterations: int = 100,
    ) -> None:
        super().__init__('PRIDA')
        self.lam = float(lam)
        self.kernel_size = int(max(3, kernel_size | 1))
        self.iterations = int(max(1, iterations))
        self._last_kernel: Array2D | None = None

    def change_param(self, param: Dict[str, Any]):
        if isinstance(param, dict):
            if 'lambda' in param and param['lambda'] is not None:
                self.lam = float(param['lambda'])
            if 'kernel_size' in param and param['kernel_size'] is not None:
                value = int(param['kernel_size'])
                self.kernel_size = max(3, value | 1)
            if 'iterations' in param and param['iterations'] is not None:
                self.iterations = max(1, int(param['iterations']))
        return super().change_param(param)

    def get_param(self):
        return [
            ('lambda', self.lam),
            ('kernel_size', self.kernel_size),
            ('iterations', self.iterations),
            ('last_kernel_shape', None if self._last_kernel is None else self._last_kernel.shape),
        ]

    def process(self, image: Array2D) -> Tuple[Array2D, Array2D]:
        if image is None or image.size == 0:
            raise ValueError('Empty image provided to PRIDA implementation.')

        original_dtype = image.dtype
        original_shape = image.shape

        image_bgr = _ensure_three_channels(image)
        channel_mode = 1 if _is_gray_image(image_bgr) else image_bgr.shape[2]
        image_bgr = image_bgr.astype(np.float64)

        params = _BlindParams(self.kernel_size, self.kernel_size, self.iterations)

        start = time.time()
        restored_u, kernel = _blind_deconv(image_bgr, self.lam, params, channel_mode)
        self.timer = time.time() - start

        restored_u = np.clip(restored_u, 0.0, 1.0)
        kernel = np.clip(kernel, 0.0, None)
        kernel /= max(np.sum(kernel), np.finfo(np.float64).eps)
        self._last_kernel = kernel.copy()

        if channel_mode == 1:
            restored_u = restored_u[..., 0]

        if np.issubdtype(original_dtype, np.integer):
            restored_u = (restored_u * 255.0).round().astype(original_dtype)
        else:
            restored_u = restored_u.astype(original_dtype, copy=False)

        if len(original_shape) == 2:
            restored_u = restored_u[: original_shape[0], : original_shape[1]]
        else:
            restored_u = restored_u[: original_shape[0], : original_shape[1]]
            if restored_u.ndim == 2:
                restored_u = np.repeat(restored_u[..., None], original_shape[2], axis=2)

        return restored_u, kernel


__all__ = ['TianyishanBlindDeconvolutionDeconvolutionAlgorithm']
