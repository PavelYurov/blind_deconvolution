"""
utils.py

Utility functions for the EML (Efficient Marginal Likelihood) blind
deconvolution algorithm of Levin et al., CVPR 2011:

    A. Levin, Y. Weiss, F. Durand, W. T. Freeman,
    "Efficient Marginal Likelihood Optimization in Blind Deconvolution",
    CVPR 2011.

Ported from the reference MATLAB package by Anat Levin
(LevinEtalCVPR2011Code/BlindDeconvCode).

─────────────────────────────────────────────────────────────────────────────
CRITICAL MATLAB vs Python / NumPy differences that are preserved here:

* MATLAB conv2(A, B, shape) performs TRUE convolution (flips B internally).
  scipy.signal.convolve2d(A, B, mode=shape) also flips B.  They match
  bit-for-bit for float64 inputs.

* MATLAB conv2(u, v, A, shape) — *separable* form — first convolves each
  column of A with the vector u (axis=0), then each row of the result with
  v (axis=1).  We replicate this by building the outer product u*v and
  calling convolve2d, or by two 1-D convolutions.

* MATLAB fft2/ifft2 match np.fft.fft2/ifft2; MATLAB ifftshift/fftshift match
  np.fft.ifftshift / np.fft.fftshift.

* MATLAB indexing is 1-based, Python is 0-based.  Ranges  `1:N` become
  `np.arange(1, N+1)` when used as *values* (e.g. meshgrid for interp2),
  but become `slice(0, N)` when used for indexing.

* MATLAB reshape and `(:)` use Fortran (column-major) order.  We use
  .ravel(order='F') / .reshape(..., order='F') wherever column-major
  semantics matter.

* MATLAB `factor(N)` returns the list of prime factors (with multiplicity).
  We re-implement it with a simple trial division since SymPy is overkill.

* MATLAB `imresize(A, ret)` uses bicubic interpolation with antialiasing by
  default.  We approximate with a bicubic resize (scipy.ndimage.zoom,
  order=3) followed by optional low-pass pre-filter for downscale.  For the
  small kernel resizes used in the pyramid, the result is visually and
  numerically very close.

* MATLAB `interp2(I, gx, gy, 'bilinear')` returns NaN for out-of-bound
  queries; we use scipy.ndimage.map_coordinates(order=1, mode='constant',
  cval=NaN).  Coordinates are converted from MATLAB 1-based (gx, gy) to
  NumPy 0-based (col, row).

* MATLAB `meshgrid([a:b], [c:d])` produces X of shape (len_d, len_b) (i.e.
  the first output varies along axis=1).  np.meshgrid(x, y) gives the same
  shape by default (indexing='xy').
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.ndimage import zoom, map_coordinates


# ════════════════════════════════════════════════════════════════════════════
# flp  (from flp.m)
# ════════════════════════════════════════════════════════════════════════════

def flp(I: np.ndarray) -> np.ndarray:
    """
    Flip a 2-D array by 180°.

    MATLAB: I = fliplr(flipud(I))
    Python: equivalent to I[::-1, ::-1].
    """
    return I[::-1, ::-1]


# ════════════════════════════════════════════════════════════════════════════
# zero_pad / zero_pad2  (from zero_pad.m, zero_pad2.m)
# ════════════════════════════════════════════════════════════════════════════

def zero_pad(M: np.ndarray, zp1: int, zp2: int = None) -> np.ndarray:
    """
    Symmetric zero-padding.  Equivalent to MATLAB zero_pad(M, zp1, zp2).
    Adds zp1 rows on top/bottom and zp2 columns on left/right.
    """
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
    """
    Asymmetric zero-padding.  Equivalent to MATLAB zero_pad2.

    Adds zp1d rows on TOP (low-index side), zp1u rows on BOTTOM,
    zp2d columns on LEFT, zp2u columns on RIGHT.

    Note MATLAB indexing:  zM(zp1d+1:end-zp1u, zp2d+1:end-zp2u) = M
    so the first argument "d" (down) is the low-index (top) padding.
    """
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


# ════════════════════════════════════════════════════════════════════════════
# fixsize  (from fixsize.m)
# ════════════════════════════════════════════════════════════════════════════

def fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:
    """
    Resize a 2-D filter to (nk1, nk2) by iteratively adding or removing
    single rows / columns from whichever side has a smaller marginal sum.

    Equivalent to MATLAB fixsize.m.
    """
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


# ════════════════════════════════════════════════════════════════════════════
# goodfactor  (from goodfactor.m)
# ════════════════════════════════════════════════════════════════════════════

def _prime_factors(n: int) -> list:
    """Return the list of prime factors of n with multiplicity (MATLAB factor)."""
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
    """
    Find the smallest integer >= N whose prime factors are all <= 7,
    i.e. a fast FFT size.  Equivalent to MATLAB goodfactor.m.
    """
    N = int(N)
    while max(_prime_factors(N)) > 7:
        N += 1
    return N


# ════════════════════════════════════════════════════════════════════════════
# normexp  (from normexp.m)
# ════════════════════════════════════════════════════════════════════════════

def normexp(logp: np.ndarray) -> np.ndarray:
    """
    Row-wise softmax with numerical stabilisation by subtracting the
    per-row max.  Equivalent to MATLAB normexp.m:

        logp = logp - max(logp, [], 2) * ones(1, m);
        p    = exp(logp);
        p    = p ./ (sum(p, 2) * ones(1, m));
    """
    logp = np.asarray(logp, dtype=np.float64)
    row_max = logp.max(axis=1, keepdims=True)
    p = np.exp(logp - row_max)
    p = p / p.sum(axis=1, keepdims=True)
    return p


# ════════════════════════════════════════════════════════════════════════════
# convfun  (from convfun.m)  — wrapper around linear / cyclic convolution
# ════════════════════════════════════════════════════════════════════════════

def _cycconv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Cyclic 2-D convolution, equivalent to MATLAB cycconv(x, k).

    Implemented via FFT:  ifft2( fft2(x) * fft2(psf2otf(k, size(x))) ).
    Preserves the "same"-size output and the MATLAB sign convention
    (true convolution, i.e. kernel is flipped implicitly by correlation
    with conj, OR equivalently we multiply the OTF that already represents
    the convolution kernel).
    """
    X = fft2(x)
    # Build an OTF of k padded to size(x) with the PSF centre at (0, 0):
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
    """
    Linear (conv2) or cyclic convolution, depending on cycconvv flag.
    Equivalent to MATLAB convfun.m.

    convshape in {'valid', 'same', 'full'}, default 'valid'.
    """
    if cycconvv:
        return _cycconv(x, k)
    return convolve2d(x, k, mode=convshape)


# ════════════════════════════════════════════════════════════════════════════
# fftconvf  (from fftconvf.m)  — fast FFT-based convolution with cached OTF
# ════════════════════════════════════════════════════════════════════════════

def fftconvf(I: np.ndarray, k: np.ndarray, K: np.ndarray,
             method: str = None) -> np.ndarray:
    """
    Fast convolution via a pre-computed, zero-padded FFT of k (``K``).

    Equivalent to MATLAB fftconvf.m:
        1. Zero-pad I up to size(K) (centred).
        2. fI = fft2(ifftshift(I_padded))
        3. cI = fftshift(ifft2(fI .* K))
        4. If method == 'same', crop back to size(I).
           If method == 'valid', crop to size(I) - size(k) + 1.

    Parameters
    ----------
    I        : (N1, N2) image
    k        : (k1, k2) kernel (used only for sizes)
    K        : (bk1, bk2) precomputed fft2(ifftshift(zero_padded_k))
    method   : None, 'same' or 'valid'

    Returns
    -------
    cI : convolved (and optionally cropped) image.
    """
    N1, N2 = I.shape
    k1, k2 = k.shape
    hk1 = (k1 - 1) // 2
    hk2 = (k2 - 1) // 2
    bk1, bk2 = K.shape

    # MATLAB: ceil((bk1 - N1) / 2) on top (= "1d"), floor on bottom (= "1u").
    # Our zero_pad2(M, zp1d, zp1u, zp2d, zp2u) uses the same convention.
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


# ════════════════════════════════════════════════════════════════════════════
# downSmpImC  (from downSmpImC.m)
# ════════════════════════════════════════════════════════════════════════════

def downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:
    """
    Downsample an image by factor `ret` (0 < ret <= 1) using a separable
    Gaussian pre-filter followed by bilinear interpolation at
    MATLAB-style 1-based sample locations [1:1/ret:size(I, d)].

    Equivalent to MATLAB downSmpImC.m.  The Gaussian is built exactly as
    in the MATLAB code: sigma = 1/(pi * ret), support truncated where the
    cumulative weight is <= 0.05 on either tail.
    """
    if ret == 1:
        return I.copy()

    sig = 1.0 / np.pi * ret

    # g0 = [-50:50] * 2 * pi  (matches MATLAB exactly)
    g0 = np.arange(-50, 51, dtype=np.float64) * 2.0 * np.pi
    sf = np.exp(-0.5 * (g0 ** 2) * (sig ** 2))
    sf = sf / sf.sum()

    # Truncate the filter to where both tails of csf exceed 0.05:
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]
    # (sum(sf) in MATLAB is a no-op print.)

    # MATLAB: conv2(sf, sf', I, 'valid')  — separable 2-D convolution.
    # First convolve each column of I with row-vector sf (axis=0),
    # then each row with column-vector sf' (axis=1).  Equivalent to
    # building the 2-D kernel = sf[:, None] * sf[None, :] and doing a
    # single 'valid' conv2.  convolve2d with true convolution also flips
    # the kernel, but here sf is symmetric, so it does not matter.
    kernel2d = np.outer(sf, sf)
    I_blur = convolve2d(I, kernel2d, mode='valid')

    # MATLAB: [gx, gy] = meshgrid([1:1/ret:size(I,2)], [1:1/ret:size(I,1)]);
    # Note: the grid uses the BLURRED image dimensions (after 'valid' conv).
    n1, n2 = I_blur.shape
    # Sample locations in MATLAB 1-based indexing, step 1/ret.
    xs = np.arange(1.0, n2 + 1e-12, 1.0 / ret)
    ys = np.arange(1.0, n1 + 1e-12, 1.0 / ret)
    # Clip to MATLAB's interp2 behaviour: values strictly inside [1, n].
    # (interp2 returns NaN outside — we allow that here, as in MATLAB.)
    gx, gy = np.meshgrid(xs, ys)  # both shape (len(ys), len(xs))

    # scipy.ndimage.map_coordinates uses (row, col) 0-based coords.
    coords = np.vstack([(gy - 1).ravel(), (gx - 1).ravel()])
    sampled = map_coordinates(I_blur, coords, order=1,
                              mode='constant', cval=np.nan)
    return sampled.reshape(gx.shape)


# ════════════════════════════════════════════════════════════════════════════
# imresize  — MATLAB-like bicubic resize (used in resizeKer)
# ════════════════════════════════════════════════════════════════════════════

def _imresize(I: np.ndarray, ret: float) -> np.ndarray:
    """
    Approximation of MATLAB ``imresize(I, ret)`` with the default bicubic
    kernel and antialiasing.  Used by ``resizeKer`` on small kernels
    between pyramid levels.

    MATLAB's ``imresize`` with the default ``'bicubic'`` method applies an
    anti-aliasing low-pass filter before the bicubic interpolation when
    downscaling.  ``scipy.ndimage.zoom`` does NOT anti-alias, which
    introduces asymmetries and negative over-/undershoots in the
    downsampled kernel.  Those artefacts survive the subsequent
    ``max(k, 0)``, ``fixsize`` and renormalisation in ``resizeKer`` and
    manifest as a shifted "shadow" copy of the kernel in the final
    estimate.  ``skimage.transform.resize`` with ``anti_aliasing=True``
    and ``order=3`` matches MATLAB's bicubic-with-antialias behaviour
    closely (symmetric output, no negative overshoot for smooth inputs).

    The output shape follows MATLAB's default rounding convention
    ``round(size * ret)``.
    """
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
        # Fallback: scipy cubic spline (worse quality, asymmetric at small ret)
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


# ════════════════════════════════════════════════════════════════════════════
# resizeKer  (from resizeKer.m)
# ════════════════════════════════════════════════════════════════════════════

def resizeKer(k: np.ndarray, ret: float, k1: int, k2: int) -> np.ndarray:
    """
    Resize a kernel between two pyramid levels.  Equivalent to
    MATLAB resizeKer.m:

        k = imresize(k, ret);
        k = max(k, 0);
        k = fixsize(k, k1, k2);
        k = k / sum(k(:));
    """
    k = _imresize(k, ret)
    k = np.maximum(k, 0.0)
    k = fixsize(k, k1, k2)
    s = k.sum()
    if s > 0:
        k = k / s
    return k


# ════════════════════════════════════════════════════════════════════════════
# set_sizes / filt_y  (from set_sizes.m, filt_y.m)
# ════════════════════════════════════════════════════════════════════════════

def set_sizes(prob: dict) -> dict:
    """
    Populate size fields in the problem dict.  Equivalent to set_sizes.m.
    """
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
    """
    Apply each filter in prob['filts'] to y, producing prob['filty'].
    Equivalent to filt_y.m.  If filt_space is 0, this is a no-op.

    MATLAB:
        prob.filty(:,:,ind) = conv2(prob.y(:,:,i), prob.filts(:,:,j),'valid');
    """
    if not prob.get('filt_space', 0):
        return prob

    filts = prob['filts']      # (h, w, nf) array
    y = prob['y']              # (H, W) or (H, W, C)
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
    # Stack along 3rd axis, matching MATLAB ordering: outer loop j, inner i.
    filty = np.stack(results, axis=2)
    prob['filty'] = filty
    return prob


# ════════════════════════════════════════════════════════════════════════════
# MOG parameters loader
# ════════════════════════════════════════════════════════════════════════════

_MOG_CACHE: dict = {}


def _find_mog_params_mat() -> Path:
    """Locate MOGparams.mat shipped with the reference MATLAB package."""
    here = Path(__file__).resolve()
    # Walk up until we find the project root (has pyproject.toml), then
    # jump into LevinEtalCVPR2011Code/BlindDeconvCode/MOGparams.mat.
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
    """
    Load the pre-trained MOG prior parameters (pis, ivars) used by the
    sparse-prior variants of the algorithm.  Equivalent to MATLAB:

        load MOGparams        % defines variables `ivars` and `pis`

    Returns
    -------
    ivars : (L,) inverse variances of the MOG components.
    pis   : (L,) mixture weights (summing to 1).
    """
    if 'ivars' in _MOG_CACHE:
        return _MOG_CACHE['ivars'], _MOG_CACHE['pis']

    from scipy.io import loadmat
    mat = loadmat(str(_find_mog_params_mat()))
    ivars = np.asarray(mat['ivars'], dtype=np.float64).ravel()
    pis = np.asarray(mat['pis'], dtype=np.float64).ravel()
    _MOG_CACHE['ivars'] = ivars
    _MOG_CACHE['pis'] = pis
    return ivars, pis


# ════════════════════════════════════════════════════════════════════════════
# Convenience: derivative filter bank used by the reference algorithms.
# ════════════════════════════════════════════════════════════════════════════

def default_deriv_filters() -> np.ndarray:
    """
    Return the default 2-filter derivative bank used in the MATLAB code:
        filts(:,:,1) = [-1 1; 0 0];
        filts(:,:,2) = [-1 0; 1 0];

    Shape: (2, 2, 2)  (h, w, nf).
    """
    filts = np.zeros((2, 2, 2), dtype=np.float64)
    filts[:, :, 0] = np.array([[-1.0, 1.0], [0.0, 0.0]])
    filts[:, :, 1] = np.array([[-1.0, 0.0], [1.0, 0.0]])
    return filts
