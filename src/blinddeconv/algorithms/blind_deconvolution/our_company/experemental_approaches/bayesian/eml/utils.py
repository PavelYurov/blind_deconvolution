from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import zoom, map_coordinates

def flp(I: np.ndarray) -> np.ndarray:

    return I[::-1, ::-1]

def zero_pad(M: np.ndarray, zp1: int, zp2: int = None) -> np.ndarray:

    if zp2 is None:
        zp2 = zp1
    zp1 = int(zp1)
    zp2 = int(zp2)
    if M.ndim == 2:
        n, m = M.shape
        zM = np.zeros((n + 2 * zp1, m + 2 * zp2), dtype=M.dtype)
        zM[zp1:zp1 + n, zp2:zp2 + m] = M
    else:
        n, m, k = M.shape
        zM = np.zeros((n + 2 * zp1, m + 2 * zp2, k), dtype=M.dtype)
        zM[zp1:zp1 + n, zp2:zp2 + m, :] = M
    return zM

def zero_pad2(M: np.ndarray, zp1d: int, zp1u: int,
              zp2d: int, zp2u: int) -> np.ndarray:

    zp1d, zp1u, zp2d, zp2u = int(zp1d), int(zp1u), int(zp2d), int(zp2u)
    if M.ndim == 2:
        n, m = M.shape
        zM = np.zeros((n + zp1u + zp1d, m + zp2d + zp2u), dtype=M.dtype)
        zM[zp1d:zp1d + n, zp2d:zp2d + m] = M
    else:
        n, m, k = M.shape
        zM = np.zeros((n + zp1u + zp1d, m + zp2d + zp2u, k), dtype=M.dtype)
        zM[zp1d:zp1d + n, zp2d:zp2d + m, :] = M
    return zM

def fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:

    f = np.asarray(f, dtype=np.float64).copy()
    k1, k2 = f.shape
    while (k1 != nk1) or (k2 != nk2):

        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]

        if k1 < nk1:
            s = f.sum(axis=1)
            tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:k1, :] = f
            else:
                tf[1:k1 + 1, :] = f
            f = tf

        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]

        if k2 < nk2:
            s = f.sum(axis=0)
            tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:, :k2] = f
            else:
                tf[:, 1:k2 + 1] = f
            f = tf

        k1, k2 = f.shape

    return f

def _prime_factors(n: int) -> list:

    n = int(n)
    factors = []
    d = 2
    while n > 1:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
        if d * d > n and n > 1:
            factors.append(n)
            break
    return factors

def goodfactor(N: int) -> int:

    N = int(N)
    while max(_prime_factors(N)) > 7:
        N += 1
    return N

def normexp(logp: np.ndarray) -> np.ndarray:

    logp = np.asarray(logp, dtype=np.float64)
    row_max = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - row_max)
    p = p / p.sum(axis=1, keepdims=True)
    return p

def _cycconv(x: np.ndarray, k: np.ndarray) -> np.ndarray:

    X = fft2(x)

    h, w = x.shape
    kh, kw = k.shape
    padded = np.zeros((h, w), dtype=np.float64)
    padded[:kh, :kw] = k
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    K = fft2(padded)
    return np.real(ifft2(X * K))

def convfun(x: np.ndarray, k: np.ndarray, cycconvv: int,
            convshape: str = 'valid') -> np.ndarray:

    if cycconvv:
        return _cycconv(x, k)
    return convolve2d(x, k, mode=convshape)

def fftconvf(I: np.ndarray, k: np.ndarray, K: np.ndarray,
             method: str = None) -> np.ndarray:

    N1, N2 = I.shape
    k1, k2 = k.shape
    hk1 = (k1 - 1) // 2
    hk2 = (k2 - 1) // 2
    bk1, bk2 = K.shape

    hdiff1d = int(np.ceil((bk1 - N1) / 2))
    hdiff1u = int(np.floor((bk1 - N1) / 2))
    hdiff2d = int(np.ceil((bk2 - N2) / 2))
    hdiff2u = int(np.floor((bk2 - N2) / 2))

    Ip = zero_pad2(I, hdiff1d, hdiff1u, hdiff2d, hdiff2u)

    fI = fft2(ifftshift(Ip))
    cI = np.real(fftshift(ifft2(fI * K)))

    if method is not None:
        if method == 'same':
            cI = cI[hdiff1d:cI.shape[0] - hdiff1u,
                    hdiff2d:cI.shape[1] - hdiff2u]
        elif method == 'valid':
            cI = cI[hdiff1d:cI.shape[0] - hdiff1u,
                    hdiff2d:cI.shape[1] - hdiff2u]
            cI = cI[hk1:cI.shape[0] - hk1, hk2:cI.shape[1] - hk2]

    return cI

def downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:

    if ret == 1:
        return I.copy()

    sig = 1.0 / np.pi * ret

    g0 = np.arange(-50, 51, dtype=np.float64) * 2.0 * np.pi
    sf = np.exp(-0.5 * (g0 ** 2) * (sig ** 2))
    sf = sf / sf.sum()

    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]

    kernel2d = np.outer(sf, sf)
    I_blur = convolve2d(I, kernel2d, mode='valid')

    n1, n2 = I_blur.shape

    xs = np.arange(1.0, n2 + 1e-12, 1.0 / ret)
    ys = np.arange(1.0, n1 + 1e-12, 1.0 / ret)

    gx, gy = np.meshgrid(xs, ys)

    coords = np.vstack([(gy - 1).ravel(), (gx - 1).ravel()])
    sampled = map_coordinates(I_blur, coords, order=1,
                              mode='constant', cval=np.nan)
    return sampled.reshape(gx.shape)

def _imresize(I: np.ndarray, ret: float) -> np.ndarray:

    I = np.asarray(I, dtype=np.float64)
    in_h, in_w = I.shape
    out_h = max(1, int(round(in_h * ret)))
    out_w = max(1, int(round(in_w * ret)))

    try:
        from skimage.transform import resize as _sk_resize
        out = _sk_resize(
            I, (out_h, out_w),
            order=3, mode='reflect',
            anti_aliasing=(ret < 1.0),
            preserve_range=True,
        )
    except ImportError:

        zh = out_h / in_h
        zw = out_w / in_w
        out = zoom(I, (zh, zw), order=3, mode='reflect', prefilter=True)
        if out.shape != (out_h, out_w):
            h = min(out.shape[0], out_h)
            w = min(out.shape[1], out_w)
            res = np.zeros((out_h, out_w), dtype=np.float64)
            res[:h, :w] = out[:h, :w]
            out = res
    return np.asarray(out, dtype=np.float64)

def resizeKer(k: np.ndarray, ret: float, k1: int, k2: int) -> np.ndarray:

    k = _imresize(k, ret)
    k = np.maximum(k, 0.0)
    k = fixsize(k, k1, k2)
    s = k.sum()
    if s > 0:
        k = k / s
    return k

def set_sizes(prob: dict) -> dict:

    k = prob['k']
    y = prob['y']
    prob['k_sz1'] = k.shape[0]
    prob['k_sz2'] = k.shape[1]
    prob['k_sz'] = prob['k_sz1'] * prob['k_sz2']
    prob['y_sz1'] = y.shape[0]
    prob['y_sz2'] = y.shape[1]
    prob['y_sz'] = prob['y_sz1'] * prob['y_sz2']
    return prob

def filt_y(prob: dict) -> dict:

    if not prob.get('filt_space', 0):
        return prob

    filts = prob['filts']
    y = prob['y']
    if y.ndim == 2:
        y = y[:, :, None]

    nf = filts.shape[2]
    nc = y.shape[2]

    results = []
    for j in range(nf):
        for i in range(nc):
            results.append(
                convolve2d(y[:, :, i], filts[:, :, j], mode='valid')
            )

    filty = np.stack(results, axis=2)
    prob['filty'] = filty
    return prob

_MOG_CACHE: dict = {}

def _find_mog_params_mat() -> Path:

    here = Path(__file__).resolve()

    root = here
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            raise FileNotFoundError("Project root not found")
        root = root.parent
    candidate = (root / "LevinEtalCVPR2011Code"
                      / "BlindDeconvCode" / "MOGparams.mat")
    if not candidate.exists():
        raise FileNotFoundError(
            f"MOGparams.mat not found at expected path {candidate}"
        )
    return candidate

def load_mog_params() -> Tuple[np.ndarray, np.ndarray]:

    if 'ivars' in _MOG_CACHE:
        return _MOG_CACHE['ivars'], _MOG_CACHE['pis']

    from scipy.io import loadmat
    mat = loadmat(str(_find_mog_params_mat()))
    ivars = np.asarray(mat['ivars'], dtype=np.float64).ravel()
    pis = np.asarray(mat['pis'], dtype=np.float64).ravel()
    _MOG_CACHE['ivars'] = ivars
    _MOG_CACHE['pis'] = pis
    return ivars, pis

def default_deriv_filters() -> np.ndarray:

    filts = np.zeros((2, 2, 2), dtype=np.float64)
    filts[:, :, 0] = np.array([[-1.0, 1.0], [0.0, 0.0]])
    filts[:, :, 1] = np.array([[-1.0, 0.0], [1.0, 0.0]])
    return filts
