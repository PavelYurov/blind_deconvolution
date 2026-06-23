"""
utils.py

Utility functions for the ARD (Automatic Relevance Determination) Variational
Bayes blind deconvolution algorithm.

Ported from the MATLAB code accompanying:
    J. Kotera, F. Sroubek, V. Smidl,
    "Blind Deconvolution with Model Discrepancies",
    IEEE Transactions on Image Processing, 2017.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    convn(A, h, 'valid'|'same'|'full'):
        MATLAB convn performs true N-D convolution (flips the kernel).
        → scipy.signal.convolve(A, h, mode=...) (also flips).

    conv2(A, h, 'same') for blurring:
        Same as convn for 2-D inputs; we use scipy.signal.convolve2d.

    fft2 / ifft2:
        MATLAB fft2(A, M, N) zero-pads the input to (M, N) before transform.
        → np.fft.fft2(A, s=(M, N)).
        np.fft.fft2 by default returns a complex array of the input size.

    interp2(V, Xq, Yq, 'linear', extrapval):
        Bilinear interpolation, V indexed as V(row, col) where Xq are
        column coords and Yq are row coords (1-based).  Returns extrapval
        for out-of-bound queries.
        → scipy.ndimage.map_coordinates with order=1.  We convert
          1-based MATLAB coords to 0-based Python coords.

    padarray(S, [1 1], 'circular', 'post'):
        Append 1 row at bottom and 1 col on right with circular wrap
        (i.e. copy of first row / first column).
        → np.pad(S, ((0,1),(0,1),...), mode='wrap').

    diff(M, 1, k):
        First-order difference along dim k (1-based).
        → np.diff(M, n=1, axis=k-1).

    size(S):
        Always returns at least 2 elements; for an N-D array squeezes
        nothing.  Trailing singleton dims may matter for repmat.

    repmat(A, [1 1 P]):
        Tile/broadcast a 2-D array P times along the new third axis.
        → np.broadcast_to(A[..., None], (..., P)).copy() or np.tile.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve, convolve2d
from scipy.ndimage import map_coordinates


# ═════════════════════════════════════════════════════════════════════════════
# vec / unvec  (from vec.m, unvec.m)
# ═════════════════════════════════════════════════════════════════════════════

def vec(arr: np.ndarray) -> np.ndarray:
    """
    Vectorise an array using MATLAB column-major (Fortran) order.

    MATLAB ``A(:)`` stacks columns; this matters for round-trips through
    ``unvec``.  We therefore use ``order='F'``.
    """
    return np.reshape(arr, (-1,), order='F')


def unvec(v: np.ndarray, shape) -> np.ndarray:
    """
    Inverse of :func:`vec`.

    Parameters
    ----------
    v     : 1-D array.
    shape : tuple of ints, target shape.
    """
    if np.isscalar(shape):
        raise ValueError("shape must be a sequence, e.g. (rows, cols, ...)")
    return np.reshape(v, tuple(shape), order='F')


# ═════════════════════════════════════════════════════════════════════════════
# initblur  (from initblur.m)
# ═════════════════════════════════════════════════════════════════════════════

def _fpulse(i: np.ndarray, c: float, w: float) -> np.ndarray:
    """
    Integral of the unit-area rectangular pulse of width ``w`` centred at ``c``.

    Mirrors the local ``fpulse`` helper in initblur.m.
    """
    i = np.asarray(i, dtype=np.float64) - c
    y = np.zeros_like(i, dtype=np.float64)
    mz = i >= 0
    mw = (i <= w / 2.0) & (i > -w / 2.0)
    y[mw] = i[mw] / w
    y[mz & ~mw] = 0.5
    y[(~mz) & (~mw)] = -0.5
    return y


def initblur(s, d=None, hw=(1, 1)) -> np.ndarray:
    """
    Initial PSF: rectangular mask of size ``hw = [h, w]`` shifted by
    ``d`` inside an ``s = [H, W]`` window.  Equivalent to MATLAB
    ``initblur(s, d, hw)``.

    The default ``hw = (1, 1)`` produces a delta function at the centre.
    """
    if np.isscalar(s):
        s = (int(s), int(s))
    else:
        s = (int(s[0]), int(s[1]))
    if np.isscalar(hw):
        hw = (float(hw), float(hw))
    else:
        hw = (float(hw[0]), float(hw[1]))
    if d is None:
        d = ((s[0] + 1) / 2.0, (s[1] + 1) / 2.0)
    # MATLAB: center = [-0.5 -0.5] + d
    center = (d[0] - 0.5, d[1] - 0.5)

    # rows direction (MATLAB hr corresponds to columns; mirror exactly)
    cols_top = np.arange(0, s[1])           # 0 : s(2)-1
    cols_bot = np.arange(1, s[1] + 1)       # 1 : s(2)
    pulse_cols = np.vstack([_fpulse(cols_top, center[1], hw[1]),
                            _fpulse(cols_bot, center[1], hw[1])])
    hr = np.diff(pulse_cols, n=1, axis=0).ravel()
    if hr.sum() == 0:
        if _fpulse(np.array([1.0]), center[1], hw[1])[0] > 0:
            hr[0] = 1.0
        else:
            hr[-1] = 1.0

    rows_top = np.arange(0, s[0])
    rows_bot = np.arange(1, s[0] + 1)
    pulse_rows = np.vstack([_fpulse(rows_top, center[0], hw[0]),
                            _fpulse(rows_bot, center[0], hw[0])])
    hc = np.diff(pulse_rows, n=1, axis=0).ravel()
    if hc.sum() == 0:
        if _fpulse(np.array([1.0]), center[0], hw[0])[0] > 0:
            hc[0] = 1.0
        else:
            hc[-1] = 1.0

    h = np.outer(hc, hr)
    h = h / h.sum()
    return h


# ═════════════════════════════════════════════════════════════════════════════
# dsample  (from dsample.m)
# ═════════════════════════════════════════════════════════════════════════════

def _gauss2d(vx: float, vy: float) -> np.ndarray:
    """
    2-D Gaussian anti-alias kernel mirroring the local ``gauss2d`` in dsample.m.

    Variances chosen so that vx (vy) ratio of Fourier coefs along x (y) are
    above ``exp(-0.5)``; spatial extent so that the boundary value is roughly
    1/20 of the centre.
    """
    v = 1.0 / (2.0 * np.pi * np.array([vy, vx]))
    j = np.round(np.sqrt(-2.0 * np.log(0.05)) /
                 (2.0 * np.pi * np.array([vy, vx]))).astype(int)
    yy, xx = np.meshgrid(np.arange(-j[0], j[0] + 1),
                         np.arange(-j[1], j[1] + 1), indexing='ij')
    h = (1.0 / ((2.0 * np.pi) * np.prod(v))) * \
        np.exp(-0.5 * ((xx ** 2) / v[1] ** 2 + (yy ** 2) / v[0] ** 2))
    h = h / h.sum()
    return h


def dsample(S, nsp, conv_type: str = 'same', fsize=None):
    """
    Resample ``S`` from sampling period ``[1, 1]`` to new period
    ``[nsp, nsp]``.  Mirrors MATLAB ``dsample(S, nsp, type, fsize)``.

    With ``S=None`` returns ``sum(h(:).^2)`` of the anti-alias kernel — the
    factor by which signal variance shrinks under the corresponding
    downsampling operator.
    """
    sp = np.array([1.0, 1.0])
    nsp_v = np.array([float(nsp), float(nsp)])
    r = nsp_v / sp
    v = 0.5 / r
    h = _gauss2d(v[1], v[0])

    if S is None:
        return float(np.sum(h ** 2))

    if nsp_v[0] == 1:
        return np.array(S)

    S = np.asarray(S, dtype=np.float64)
    # apply anti-alias filter on every channel
    if S.ndim == 2:
        Sb = convolve2d(S, h, mode=conv_type)
        Sb = Sb[..., None]
        squeeze = True
    else:
        # convolve along first two axes only
        Sb = np.empty(_convn_shape_2d(S.shape, h.shape, conv_type),
                      dtype=np.float64)
        for c in range(S.shape[2]):
            Sb[:, :, c] = convolve2d(S[:, :, c], h, mode=conv_type)
        squeeze = False

    rows, cols = Sb.shape[0], Sb.shape[1]
    # centre coords (0-based equivalent of MATLAB ((size(S)-1)/2))
    cy = (rows - 1) / 2.0
    cx = (cols - 1) / 2.0

    # X = [fliplr(center(2):-nsp(2):0), center(2)+nsp(2):nsp(2):size(S,2)-1]
    X_left = np.flip(np.arange(cx, -1e-9, -nsp_v[1]))
    X_right = np.arange(cx + nsp_v[1], cols - 1 + 1e-9, nsp_v[1])
    X = np.concatenate([X_left, X_right])

    Y_left = np.flip(np.arange(cy, -1e-9, -nsp_v[0]))
    Y_right = np.arange(cy + nsp_v[0], rows - 1 + 1e-9, nsp_v[0])
    Y = np.concatenate([Y_left, Y_right])

    # crop / extend X, Y to fsize (if given)
    if fsize is not None:
        X = _adjust_axis(X, fsize[1], nsp_v[1], cols)
        Y = _adjust_axis(Y, fsize[0], nsp_v[0], rows)

    # Sample with bilinear interpolation; use circular wrap so positions
    # slightly above (rows-1, cols-1) match MATLAB's circular post-pad.
    YY, XX = np.meshgrid(Y, X, indexing='ij')
    coords = np.vstack([YY.ravel(), XX.ravel()])

    out = np.empty((Y.size, X.size, Sb.shape[2]), dtype=np.float64)
    for c in range(Sb.shape[2]):
        out[:, :, c] = map_coordinates(
            Sb[:, :, c], coords, order=1, mode='wrap', cval=0.0
        ).reshape(Y.size, X.size)

    if squeeze:
        out = out[:, :, 0]
    return out


def _convn_shape_2d(in_shape, k_shape, mode: str):
    """Output shape of scipy.signal.convolve2d along axes 0/1, preserving axis 2."""
    H, W = in_shape[0], in_shape[1]
    kh, kw = k_shape[0], k_shape[1]
    if mode == 'full':
        out = (H + kh - 1, W + kw - 1)
    elif mode == 'same':
        out = (H, W)
    elif mode == 'valid':
        out = (H - kh + 1, W - kw + 1)
    else:
        raise ValueError(f"Unknown conv mode: {mode}")
    if len(in_shape) == 2:
        return out
    return out + tuple(in_shape[2:])


def _adjust_axis(coord: np.ndarray, target_len: int, step: float,
                 axis_len: int) -> np.ndarray:
    """
    Replicate the fsize crop/extend logic of MATLAB ``dsample``.  The
    extension uses circular addressing so we ``mod`` by ``axis_len`` at
    the end, matching MATLAB's ``mod(...,size(S,2))`` semantics.
    """
    n = coord.size
    if n > target_len:
        dl = (n - target_len) // 2
        dr = (n - target_len) - dl
        coord = coord[dl:n - dr]
        if dl != dr:
            coord = coord + step / 2.0
    elif n < target_len:
        dl = (target_len - n) // 2
        dr = (target_len - n) - dl
        left_ext = (np.linspace(step, dl * step, dl) - (dl + 1) * step
                    + coord[0]) if dl > 0 else np.array([])
        right_ext = (np.linspace(step, dr * step, dr) + coord[-1]) \
            if dr > 0 else np.array([])
        coord = np.concatenate([left_ext, coord, right_ext])
        if dl != dr:
            coord = coord - step / 2.0
        coord = np.mod(coord, axis_len)
    return coord


# ═════════════════════════════════════════════════════════════════════════════
# mycg  (from mycg.m)
# ═════════════════════════════════════════════════════════════════════════════

def mycg(fh, b: np.ndarray, defrelres: float, noiter: int,
         ph=None, x0: np.ndarray = None):
    """
    Custom Conjugate Gradient solver mirroring MATLAB ``mycg.m``.

    Solves ``A(x) = b`` where ``A`` is supplied as a function handle ``fh``.
    Optional preconditioner ``ph(r)`` mirrors the original API.

    Returns
    -------
    x       : solution.
    flag    : 0 if converged, 1 otherwise.
    relres  : final relative residual.
    iter_   : number of iterations performed.
    resvec  : last residual vector.
    """
    flag = 1
    b = np.asarray(b)
    nb = np.sqrt(np.vdot(b, b).real)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = np.array(x0, copy=True)

    r = b - fh(x)
    z = r if ph is None else ph(r)
    p = z.copy()
    rzold = np.vdot(r, z)
    relres = 1.0
    i = 0
    for i in range(1, noiter + 1):
        Ap = fh(p)
        denom = np.vdot(p, Ap)
        if denom == 0:
            break
        alpha = rzold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        relres = float(np.sqrt(np.vdot(r, r).real) / (nb if nb > 0 else 1.0))
        if not np.isfinite(relres):
            break
        if relres < defrelres:
            flag = 0
            break
        z = r if ph is None else ph(r)
        rznew = np.vdot(r, z)
        if rzold == 0:
            break
        p = z + (rznew / rzold) * p
        rzold = rznew
    return x, flag, relres, i, r


# ═════════════════════════════════════════════════════════════════════════════
# uConstr  (from uConstr.m)
# ═════════════════════════════════════════════════════════════════════════════

def u_constr(U: np.ndarray, vrange: np.ndarray) -> np.ndarray:
    """
    Clip image ``U`` to the per-channel intensity range ``vrange``.

    ``vrange`` has shape ``(C, 2)`` where each row is ``[lo, hi]``.
    For 2-D input ``vrange`` may also be ``(1, 2)`` or a length-2 vector.
    """
    out = np.array(U, copy=True)
    vr = np.atleast_2d(np.asarray(vrange, dtype=np.float64))
    if out.ndim == 2:
        lo, hi = vr[0]
        np.clip(out, lo, hi, out=out)
    else:
        for c in range(out.shape[2]):
            lo, hi = vr[c]
            out[:, :, c] = np.clip(out[:, :, c], lo, hi)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# updateGPrior  (from updateGPrior.m)
# ═════════════════════════════════════════════════════════════════════════════

def update_g_prior(DI: np.ndarray, covU: np.ndarray, model) -> np.ndarray:
    """
    Variational-Bayes update of the per-pixel ARD precision on the image
    gradient prior.  Mirrors MATLAB ``updateGPrior.m``.

    Posterior is ``Gamma(a_0 + 1/2, b_0 + 1/2*((D_i u)^2 + tr{D_i^T D_i covU}))``
    yielding mean ``lambda = a_k / b_k``.

    Parameters
    ----------
    DI    : (H, W, P) for grayscale, P = #derivatives (e.g. 2 for [Dx, Dy]).
            For colour use (H, W, C, P).
    covU  : (H, W) or (H, W, C) — diagonal of cov(u).
    model : (a_0, b_0) — Gamma-hyperprior.

    Returns
    -------
    prcmap : same shape as ``DI`` — per-pixel precision λ.
    """
    a0, b0 = float(model[0]), float(model[1])

    if DI.ndim == 3:
        usize = DI.shape[:2]
        P = DI.shape[2]
        # FS = cat(3, fft2([1 1], H, W), fft2([1; 1], H, W))
        FS = np.zeros((usize[0], usize[1], 2), dtype=np.complex128)
        FS[:, :, 0] = fft2(np.array([[1.0, 1.0]]), s=usize)
        FS[:, :, 1] = fft2(np.array([[1.0], [1.0]]), s=usize)
        if P != 2:
            # If P doesn't match (e.g. only Dx), tile the matching slice.
            FS = np.broadcast_to(FS[:, :, :1], (usize[0], usize[1], P)).copy()
        FcovU = fft2(covU)
        # CRITICAL: must specify axes=(0,1) so numpy transforms along (H,W),
        # not along the last two dims (W, 2) which is numpy's default for 3-D arrays.
        # MATLAB ifft2 on a 3-D array always transforms the first two dims.
        trace_term = np.real(ifft2(FS * FcovU[:, :, None], axes=(0, 1)))
        prcmap = (a0 + 0.5) / (b0 + 0.5 * (DI ** 2 + trace_term))
        return prcmap

    if DI.ndim == 4:
        usize = DI.shape[:2]
        C = DI.shape[2]
        P = DI.shape[3]
        FS = np.zeros((usize[0], usize[1], C, 2), dtype=np.complex128)
        FS[:, :, :, 0] = np.broadcast_to(
            fft2(np.array([[1.0, 1.0]]), s=usize)[:, :, None], (usize[0], usize[1], C))
        FS[:, :, :, 1] = np.broadcast_to(
            fft2(np.array([[1.0], [1.0]]), s=usize)[:, :, None], (usize[0], usize[1], C))
        if P != 2:
            FS = np.broadcast_to(FS[:, :, :, :1],
                                 (usize[0], usize[1], C, P)).copy()
        # covU may be (H,W) (gray) or (H,W,C)
        if covU.ndim == 2:
            FcovU = fft2(covU)[:, :, None]
        else:
            FcovU = fft2(covU, axes=(0, 1))
        trace_term = np.real(ifft2(FS * FcovU[..., None], axes=(0, 1)))
        prcmap = (a0 + 0.5) / (b0 + 0.5 * (DI ** 2 + trace_term))
        return prcmap

    raise ValueError(f"DI must be 3-D or 4-D, got shape {DI.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# PSF ↔ OTF helpers  (ported from FBDHSGP/utils.py for use by frils_deb_ubc)
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape) -> np.ndarray:
    """
    MATLAB ``psf2otf(psf, shape)``:
      1. Zero-pad ``psf`` to ``shape``.
      2. Circularly shift by ``-floor(size(psf)/2)`` so the centre lands at (0,0).
      3. Return ``fft2``.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    in_h, in_w = psf.shape
    out_h, out_w = shape
    padded = np.zeros((out_h, out_w), dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size) -> np.ndarray:
    """
    MATLAB ``otf2psf(otf, psf_size)``:
      1. ``ifft2`` (take real part).
      2. Circularly shift by ``+floor(psf_size/2)``.
      3. Crop to ``psf_size``.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


def pad_replicate(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """MATLAB ``padarray(x, [pad_h pad_w], 'replicate', 'both')``."""
    return np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")


# ═════════════════════════════════════════════════════════════════════════════
# project_to_simplex   (positivity + sum-to-1 ADMM projection)
# ═════════════════════════════════════════════════════════════════════════════

def project_to_simplex(v: np.ndarray, n_per_plane: int) -> np.ndarray:
    """
    Euclidean projection of a vector onto the probability simplex
    ``{x >= 0, sum(x) = 1}`` for each contiguous block of ``n_per_plane``
    elements.  Used in the optional ADMM constraint on PSF in the original
    MATLAB code (``project2Set`` helper).

    For a single PSF (P = 1) this projects ``v`` onto the simplex.  For
    multiple PSFs the input is split into ``len(v) / n_per_plane`` blocks.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    n_blocks = v.size // n_per_plane
    out = np.empty_like(v)
    for b in range(n_blocks):
        seg = v[b * n_per_plane:(b + 1) * n_per_plane]
        u = np.sort(seg)[::-1]
        css = np.cumsum(u)
        rng = np.arange(1, n_per_plane + 1)
        cond = u - (css - 1.0) / rng > 0
        rho = np.nonzero(cond)[0]
        if rho.size == 0:
            out[b * n_per_plane:(b + 1) * n_per_plane] = np.maximum(seg, 0)
        else:
            rho_idx = rho[-1]
            theta = (css[rho_idx] - 1.0) / (rho_idx + 1)
            out[b * n_per_plane:(b + 1) * n_per_plane] = np.maximum(seg - theta, 0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper  (MATLAB Image Processing Toolbox edgetaper)
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Replicates MATLAB ``edgetaper(I, PSF)``.

    The image is blended towards a circularly-blurred version of itself in a
    region whose width is governed by the PSF, suppressing border ringing
    introduced by FFT-based deconvolution.  The original algorithm
    (Gonzalez & Woods) constructs a per-axis weight from the autocorrelation
    of the PSF projected on that axis.
    """
    img = np.asarray(img, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)
    if img.ndim == 2:
        return _edgetaper2d(img, psf)
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _edgetaper2d(img[:, :, c], psf)
    return out


def _edgetaper2d(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    H, W = img.shape
    psf = psf / psf.sum()

    # Per-axis projections of the PSF
    proj_r = psf.sum(axis=1)   # length kH
    proj_c = psf.sum(axis=0)   # length kW

    # Autocorrelation via FFT, take real part
    def _autocorr(p, n):
        P = np.fft.fft(p, n=n)
        a = np.real(np.fft.ifft(P * np.conj(P)))
        a = a / a.max() if a.max() > 0 else a
        return a

    a_r = _autocorr(proj_r, H)
    a_c = _autocorr(proj_c, W)

    # alpha = 1 - (1 - a_r) outer (1 - a_c)  (MATLAB definition)
    alpha = 1.0 - np.outer(1.0 - a_r, 1.0 - a_c)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Blurred copy of the image (circular conv with PSF) for the boundary
    img_pad_fft = np.fft.fft2(img)
    psf_otf = np.fft.fft2(psf, s=(H, W))
    blurred = np.real(np.fft.ifft2(img_pad_fft * psf_otf))

    return alpha * img + (1.0 - alpha) * blurred
