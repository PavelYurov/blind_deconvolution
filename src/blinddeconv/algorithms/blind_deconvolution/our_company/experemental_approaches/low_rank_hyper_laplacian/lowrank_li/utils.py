import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import zoom

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:

    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)

def imresize(img: np.ndarray, target_size, method: str = 'bilinear') -> np.ndarray:

    th = int(target_size[0])
    tw = int(target_size[1])
    if th <= 0 or tw <= 0:
        return np.zeros((max(th, 1), max(tw, 1)), dtype=np.float64)
    if img.size == 0:
        return np.zeros((th, tw), dtype=np.float64)

    order = 1 if method == 'bilinear' else 3
    zh = th / img.shape[0]
    zw = tw / img.shape[1]
    result = zoom(img.astype(np.float64), (zh, zw), order=order)

    rh, rw = result.shape
    if rh != th or rw != tw:
        out = np.zeros((th, tw), dtype=np.float64)
        mh, mw = min(rh, th), min(rw, tw)
        out[:mh, :mw] = result[:mh, :mw]
        return out
    return result

def edgetaper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:

    blurred = fftconvolve(img, psf, mode='same')

    kh, kw = psf.shape
    ih, iw = img.shape

    proj_y = psf.sum(axis=1)
    proj_x = psf.sum(axis=0)

    ac_y = np.correlate(proj_y, proj_y, mode='full')
    ac_x = np.correlate(proj_x, proj_x, mode='full')

    ac_y = ac_y / ac_y.max()
    ac_x = ac_x / ac_x.max()

    Ly = kh - 1
    Lx = kw - 1

    wy = np.ones(ih)
    wx = np.ones(iw)

    ny = min(Ly, ih)
    nx = min(Lx, iw)

    wy[:ny] = np.minimum(wy[:ny], ac_y[:ny])
    wy[-ny:] = np.minimum(wy[-ny:], ac_y[-ny:])
    wx[:nx] = np.minimum(wx[:nx], ac_x[:nx])
    wx[-nx:] = np.minimum(wx[-nx:], ac_x[-nx:])

    alpha = np.outer(wy, wx)
    return alpha * img + (1.0 - alpha) * blurred

def center_kernel_separate(x: np.ndarray, y: np.ndarray,
                           k: np.ndarray):

    nrows, ncols = k.shape

    mu_y = np.sum(np.arange(1, nrows + 1) * k.sum(axis=1))
    mu_x = np.sum(np.arange(1, ncols + 1) * k.sum(axis=0))

    offset_x = int(np.round(np.floor(ncols / 2) + 1 - mu_x))
    offset_y = int(np.round(np.floor(nrows / 2) + 1 - mu_y))

    sk_h = abs(offset_y) * 2 + 1
    sk_w = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sk_h, sk_w))

    shift_kernel[abs(offset_y) + offset_y,
                 abs(offset_x) + offset_x] = 1.0

    k = convolve2d(k, shift_kernel, mode='same')

    inv_sk = shift_kernel[::-1, ::-1]
    x = convolve2d(x, inv_sk, mode='same')
    y = convolve2d(y, inv_sk, mode='same')

    return x, y, k

_lut_cache: dict = {}

def _compute_w1(v: np.ndarray, beta: float) -> np.ndarray:

    return np.maximum(np.abs(v) - 1.0 / beta, 0.0) * np.sign(v)

def _compute_w23(v: np.ndarray, beta: float) -> np.ndarray:

    epsilon = 1e-6

    orig_shape = v.shape
    vr = v.ravel().astype(np.complex128)
    n = vr.size

    k_val = 8.0 / (27.0 * beta ** 3)
    m = np.full(n, k_val, dtype=np.complex128)

    v2 = vr * vr
    v3 = v2 * vr
    v4 = v3 * vr
    m2 = m * m
    m3 = m2 * m

    aq = -1.125 * v2
    bq = 0.25 * v3

    q = -0.125 * m * v2
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = -q / 2.0 + np.sqrt(-m3 / 27.0 + m2 * v4 / 256.0)

        u = np.exp(np.log(r1) / 3.0)
        yy = 2.0 * (-5.0 / 18.0 * aq + u + m / (3.0 * u))

        W = np.sqrt(aq / 3.0 + yy)

        inner_p = np.sqrt(-(aq + yy + bq / W))
        inner_m = np.sqrt(-(aq + yy - bq / W))

    roots = np.column_stack([
        0.75 * vr + 0.5 * (W + inner_p),
        0.75 * vr + 0.5 * (W - inner_p),
        0.75 * vr + 0.5 * (-W + inner_m),
        0.75 * vr + 0.5 * (-W - inner_m),
    ])

    sv = np.sign(v.ravel())[:, None]
    abs_v = np.abs(v.ravel())[:, None]

    rsv = np.real(roots) * sv

    mask = ((np.abs(np.imag(roots)) < epsilon) &
            (rsv > abs_v / 2.0) &
            (rsv < abs_v))

    candidates = mask * rsv

    best_rsv = np.sort(candidates, axis=1)[:, ::-1][:, 0]

    w = (best_rsv * sv.ravel()).real.astype(np.float64)
    w[np.isnan(w)] = 0.0
    return w.reshape(orig_shape)

def _compute_w12(v: np.ndarray, beta: float) -> np.ndarray:

    epsilon = 1e-6

    orig_shape = v.shape
    vr = v.ravel().astype(np.complex128)
    n = vr.size

    k_val = -0.25 / (beta ** 2)
    m = np.full(n, k_val, dtype=np.complex128) * np.sign(v.ravel())

    t1 = (2.0 / 3.0) * vr
    v2 = vr * vr
    v3 = v2 * vr

    with np.errstate(divide='ignore', invalid='ignore'):
        inner = -27.0 * m - 2.0 * v3 + (3.0 * np.sqrt(3.0 + 0j)) * np.sqrt(27.0 * m ** 2 + 4.0 * m * v3)
        t2 = np.exp(np.log(inner) / 3.0)
        t3 = v2 / t2

    sqrt3 = np.sqrt(3.0 + 0j)
    c1 = 2.0 ** (1.0 / 3) / 3.0
    c_21 = (1.0 + 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3))
    c_22 = (1.0 - 1j * sqrt3) / (6.0 * 2.0 ** (1.0 / 3))
    c_31 = (1.0 - 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3))
    c_32 = (1.0 + 1j * sqrt3) / (6.0 * 2.0 ** (1.0 / 3))

    with np.errstate(divide='ignore', invalid='ignore'):
        roots = np.column_stack([
            t1 + c1 * t3 + t2 / (3.0 * 2.0 ** (1.0 / 3)),
            t1 - c_21 * t3 - c_22 * t2,
            t1 - c_31 * t3 - c_32 * t2,
        ])

    roots[np.isnan(roots) | np.isinf(roots)] = 0.0

    sv = np.sign(v.ravel())[:, None]
    abs_v = np.abs(v.ravel())[:, None]

    rsv = np.real(roots) * sv

    mask = ((np.abs(np.imag(roots)) < epsilon) &
            (rsv > 2.0 * abs_v / 3.0) &
            (rsv < abs_v))

    candidates = mask * rsv
    best_rsv = np.sort(candidates, axis=1)[:, ::-1][:, 0]

    w = (best_rsv * sv.ravel()).real.astype(np.float64)
    w[np.isnan(w)] = 0.0
    return w.reshape(orig_shape)

def _newton_w(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:

    x = v.copy().astype(np.float64)
    for _ in range(4):
        fd = alpha * np.sign(x) * np.abs(x) ** (alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.abs(x) ** (alpha - 2) + beta
        with np.errstate(divide='ignore', invalid='ignore'):
            x = x - fd / fdd
    x[np.isnan(x)] = 0.0

    z = beta / 2.0 * v ** 2
    f = np.abs(x) ** alpha + beta / 2.0 * (x - v) ** 2
    x = np.where(f < z, x, 0.0)
    return x

def _compute_w(v: np.ndarray, beta: float, alpha: float) -> np.ndarray:

    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)

def solve_image_bregman(v: np.ndarray, beta: float,
                        alpha: float) -> np.ndarray:

    key = (beta, alpha)
    lut_range = 10.0
    lut_step = 0.0001

    if key not in _lut_cache:
        xx = np.arange(-lut_range, lut_range + lut_step * 0.5, lut_step)
        lookup = _compute_w(xx.copy(), beta, alpha)
        _lut_cache[key] = (xx, lookup.astype(np.float64))

    xx, lookup = _lut_cache[key]

    w = np.interp(v.ravel(), xx, lookup)
    return w.reshape(v.shape)
