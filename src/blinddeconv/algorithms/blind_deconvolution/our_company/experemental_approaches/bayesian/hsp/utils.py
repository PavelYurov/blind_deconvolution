from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label as nd_label
from scipy.optimize import minimize

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

def flipprev(x: np.ndarray) -> np.ndarray:

    return np.flipud(np.fliplr(x)).copy()

def getfilters(name: str) -> list[np.ndarray]:

    if name == 'none':
        return [np.array([[1.0]])]

    if name == 'fohv':

        return [
            np.array([[0.0, 1.0, -1.0]]),
            np.array([[0.0], [1.0], [-1.0]]),
        ]

    if name == 'fo':
        return [
            np.array([[0.0, 1.0, -1.0]]),
            np.array([[0.0], [1.0], [-1.0]]),
            np.array([[0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, -1.0]]),
            np.array([[0.0, 0.0, -1.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]]),
        ]

    return [np.array([[1.0]])]

def conv2_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:

    return convolve2d(a, k, mode='same', boundary='fill', fillvalue=0.0)

def conv2_valid(a: np.ndarray, k: np.ndarray) -> np.ndarray:

    return convolve2d(a, k, mode='valid', boundary='fill', fillvalue=0.0)

def pad_replicate(a: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:

    return np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

def pad_circular(a: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:

    return np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')

def imresize_bilinear(a: np.ndarray, target) -> np.ndarray:

    return _imresize(a, target, method='bilinear')

def imresize_bicubic(a: np.ndarray, target) -> np.ndarray:

    return _imresize(a, target, method='bicubic')

def _imresize(a: np.ndarray, target, method: str) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float64)
    in_h, in_w = a.shape[:2]

    if np.isscalar(target):
        scale = float(target)
        out_h = int(np.ceil(in_h * scale))
        out_w = int(np.ceil(in_w * scale))
    else:
        out_h, out_w = int(target[0]), int(target[1])

    if out_h == in_h and out_w == in_w:
        return a.copy()

    if not _HAS_CV2:
        from scipy.ndimage import zoom as _zoom
        order = 1 if method == 'bilinear' else 3
        zh = out_h / in_h
        zw = out_w / in_w
        return _zoom(a, (zh, zw), order=order, mode='nearest')

    is_down = (out_h < in_h) or (out_w < in_w)
    if method == 'bilinear':
        interp = cv2.INTER_AREA if is_down else cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_AREA if is_down else cv2.INTER_CUBIC
    return cv2.resize(a, (out_w, out_h), interpolation=interp).astype(np.float64)

def im2col_sliding(a: np.ndarray, block: Tuple[int, int]) -> np.ndarray:

    a = np.ascontiguousarray(a, dtype=np.float64)
    M, N = a.shape
    m, n = block
    out_rows = m * n
    out_cols = (M - m + 1) * (N - n + 1)

    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, (m, n))

    blocks = np.transpose(win, (0, 1, 3, 2))
    blocks = blocks.reshape(M - m + 1, N - n + 1, m * n)

    out = np.transpose(blocks, (1, 0, 2))
    out = out.reshape(out_cols, out_rows)
    return out.T

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:

    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)

def otf2psf(otf: np.ndarray, psf_size: Tuple[int, int]) -> np.ndarray:

    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]

def cg_solve(x0: np.ndarray,
             a_func: Callable[[np.ndarray], np.ndarray],
             rhs: np.ndarray,
             max_iter: int = 15,
             tol: float = 1e-5) -> Tuple[np.ndarray, int]:

    x = x0.copy()
    r = rhs - a_func(x)
    d = r.copy()
    delta_new = float(np.dot(r.ravel(), r.ravel()))

    for it in range(1, max_iter + 1):
        q = a_func(d)
        denom = float(np.dot(d.ravel(), q.ravel()))
        if denom == 0.0:
            break
        alfa = delta_new / denom
        x_new = x + alfa * d
        r = r - alfa * q
        delta_old = delta_new
        delta_new = float(np.dot(r.ravel(), r.ravel()))
        beta = delta_new / delta_old if delta_old != 0.0 else 0.0
        d = r + beta * d

        x_norm_sq = float(np.sum(np.abs(x.ravel()) ** 2))
        decr = (float(np.sum(np.abs((x_new - x).ravel()) ** 2))
                / x_norm_sq) if x_norm_sq > 0.0 else np.inf

        if it > 25 and decr < tol:
            x = x_new
            break

        x = x_new

    return x, it

def center_kernel(k: np.ndarray,
                  xf: Optional[Sequence[np.ndarray]] = None
                  ) -> Tuple[np.ndarray, Optional[list]]:

    kh, kw = k.shape

    cx = float(np.sum(np.arange(1, kw + 1) * k.sum(axis=0)))
    cy = float(np.sum(np.arange(1, kh + 1) * k.sum(axis=1)))

    def _mr(x: float) -> int:
        return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))

    offset_x = _mr(np.floor(kw / 2.0) + 1 - cx)
    offset_y = _mr(np.floor(kh / 2.0) + 1 - cy)

    if offset_x == 0 and offset_y == 0:
        return k.copy(), (list(xf) if xf is not None else None)

    sh_rows = abs(offset_y) * 2 + 1
    sh_cols = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sh_rows, sh_cols), dtype=np.float64)

    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    k_centered = conv2_same(k, shift_kernel)

    if xf is None:
        return k_centered, None

    xf_centered = [conv2_same(x, shift_kernel) for x in xf]
    return k_centered, xf_centered

def clean_kernel_ecp(k: np.ndarray) -> np.ndarray:

    k = k.copy()

    structure = np.ones((3, 3), dtype=bool)
    labelled, n = nd_label(k > 0, structure=structure)
    for i in range(1, n + 1):
        mask = labelled == i
        if k[mask].sum() < 0.1:
            k[mask] = 0.0

    k[k < 0] = 0.0
    s = k.sum()
    if s > 0:
        k = k / s
    return k

def quadprog_nonneg(H: np.ndarray,
                    f: np.ndarray,
                    x0: Optional[np.ndarray] = None,
                    max_iter: int = 200,
                    tol: float = 1e-8) -> np.ndarray:

    n = H.shape[0]
    H = 0.5 * (H + H.T)

    def obj(x):
        Hx = H @ x
        val = 0.5 * float(x @ Hx) + float(f @ x)
        grad = Hx + f
        return val, grad

    if x0 is None:
        x0 = np.full(n, 1.0 / n)
    x0 = np.clip(x0, 0.0, None)

    bounds = [(0.0, None)] * n
    res = minimize(
        obj, x0, jac=True, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol},
    )
    return res.x

__all__ = [
    'flipprev',
    'getfilters',
    'conv2_same',
    'conv2_valid',
    'pad_replicate',
    'pad_circular',
    'imresize_bilinear',
    'imresize_bicubic',
    'im2col_sliding',
    'psf2otf',
    'otf2psf',
    'cg_solve',
    'center_kernel',
    'clean_kernel_ecp',
    'quadprog_nonneg',
]
