"""
utils.py

Utility helpers for the HSP (Hyperbolic-Secant Prior) blind deconvolution.

Ported from MATLAB code accompanying:
    F. M. Castro-Macias, F. Perez-Bueno, M. Vega, J. Mateos, R. Molina,
    A. K. Katsaggelos: "Bayesian Blind Image Deconvolution using a
    Hyperbolic-Secant prior" (2024).

Original MATLAB source files mirrored here:
    flipprev.m              -> flipprev
    getfilters.m            -> getfilters
    center_kernel.m         -> center_kernel
    clean_kernel_ECP.m      -> clean_kernel_ecp
    CG.m                    -> cg_solve

Plus low-level MATLAB-equivalent primitives used throughout the toolbox:
    conv2 'same'/'valid'    -> conv2_same / conv2_valid
    padarray 'replicate'    -> pad_replicate
    padarray 'circular'     -> pad_circular
    imresize 'bilinear'     -> imresize_bilinear
    im2col 'sliding'        -> im2col_sliding
    psf2otf / otf2psf       -> psf2otf / otf2psf
    quadprog (x >= 0)       -> quadprog_nonneg

CRITICAL MATLAB vs. Python notes:
    * MATLAB conv2(A,B,'same') performs TRUE 2-D convolution (B is flipped).
      scipy.signal.convolve2d(A,B,'same') does the same.  So
      conv2_same / conv2_valid simply delegate to scipy.signal.
    * MATLAB conv2 'valid' result size: (M-kM+1, N-kN+1).
    * MATLAB padarray(A,[p q],'replicate','both') replicates the border.
      np.pad(A, ((p,p),(q,q)), mode='edge') gives identical output.
    * MATLAB padarray(A,[p q],'circular','both') wraps around.
      np.pad(A, ..., mode='wrap') is identical.
    * MATLAB imresize with the default 'bilinear' method uses an
      ANTI-ALIASED bilinear filter when DOWNSAMPLING (Antialiasing=true
      by default).  cv2.resize(INTER_AREA) for downsampling and
      INTER_LINEAR for upsampling is the closest practical match;
      we use the explicit "antialiasing on downsample" rule.
    * MATLAB im2col(A,[m n],'sliding') stores blocks in COLUMN-MAJOR
      order: each column is a block flattened by columns, and blocks
      themselves are enumerated column-first (i.e., the top-left
      corner runs over rows first, then columns).
    * MATLAB quadprog(H, f, ..., lb, []): minimises 0.5 x'Hx + f'x
      subject to lb <= x.  We replicate with scipy.optimize.minimize
      method='L-BFGS-B' (analytic gradient, lb=0, no upper bound).
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label as nd_label
from scipy.optimize import minimize

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


# ═══════════════════════════════════════════════════════════════════════════
# Trivial helpers
# ═══════════════════════════════════════════════════════════════════════════

def flipprev(x: np.ndarray) -> np.ndarray:
    """
    MATLAB: y = fliplr(flipud(x))   (== rot90(x, 2))

    Used everywhere to obtain the adjoint of the convolution operator,
    since for a real PSF k:  H^T y == conv2(y, flipud(fliplr(k)), 'same').
    """
    return np.flipud(np.fliplr(x)).copy()


def getfilters(name: str) -> list[np.ndarray]:
    """
    MATLAB getfilters.m — high-pass filter banks used as F_n in eq. (4).

    Parameters
    ----------
    name : 'none' | 'fohv' | 'fo'

    Returns
    -------
    list of 2-D float64 ndarrays, in the same order as the MATLAB cells.
    """
    if name == 'none':
        return [np.array([[1.0]])]

    if name == 'fohv':
        # first order, horizontal + vertical differences
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

    # default
    return [np.array([[1.0]])]


# ═══════════════════════════════════════════════════════════════════════════
# Convolutions (MATLAB-equivalent wrappers around scipy.signal.convolve2d)
# ═══════════════════════════════════════════════════════════════════════════

def conv2_same(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """MATLAB conv2(A, k, 'same') — true 2-D convolution, output size A."""
    return convolve2d(a, k, mode='same', boundary='fill', fillvalue=0.0)


def conv2_valid(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """MATLAB conv2(A, k, 'valid') — convolution, output size (M-kM+1, N-kN+1)."""
    return convolve2d(a, k, mode='valid', boundary='fill', fillvalue=0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Padding (MATLAB padarray equivalents)
# ═══════════════════════════════════════════════════════════════════════════

def pad_replicate(a: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """MATLAB padarray(A, [pad_h pad_w], 'replicate', 'both')."""
    return np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


def pad_circular(a: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """MATLAB padarray(A, [pad_h pad_w], 'circular', 'both')."""
    return np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')


# ═══════════════════════════════════════════════════════════════════════════
# imresize — best practical equivalent of MATLAB imresize bilinear
# ═══════════════════════════════════════════════════════════════════════════

def imresize_bilinear(a: np.ndarray, target) -> np.ndarray:
    """
    Bilinear resize approximating MATLAB ``imresize(A, target, 'bilinear')``.

    Parameters
    ----------
    a : 2-D ndarray
    target : float | (rows, cols) | (rows, cols) tuple
        * scalar  ->  scale factor (same for both dimensions);
        * tuple/list of length 2  ->  explicit output (rows, cols).

    Notes
    -----
    MATLAB's imresize uses an anti-aliased bilinear filter when the scale
    is < 1.  We emulate that by switching to ``cv2.INTER_AREA`` for
    downsampling and ``cv2.INTER_LINEAR`` for upsampling, which is the
    closest practical match available in OpenCV.
    """
    return _imresize(a, target, method='bilinear')


def imresize_bicubic(a: np.ndarray, target) -> np.ndarray:
    """
    Bicubic resize approximating MATLAB ``imresize(A, target)`` (default
    method since R2007+ is bicubic with antialias=true on downsample).

    Used for kernel upsampling between pyramid stages — see
    ``multi_stage_deconv_alphaden.m`` which calls plain ``imresize(k, sz)``.
    """
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
    else:  # bicubic
        interp = cv2.INTER_AREA if is_down else cv2.INTER_CUBIC
    return cv2.resize(a, (out_w, out_h), interpolation=interp).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# im2col 'sliding'  (column-major MATLAB layout)
# ═══════════════════════════════════════════════════════════════════════════

def im2col_sliding(a: np.ndarray, block: Tuple[int, int]) -> np.ndarray:
    """
    Replicate MATLAB im2col(A, [m n], 'sliding') exactly.

    The output has shape (m*n, (M-m+1)*(N-n+1)) where:
      - each column is one m-by-n block of A, flattened in COLUMN-MAJOR
        ('F') order;
      - columns are enumerated in COLUMN-MAJOR order over the block
        top-left positions:  first all positions with j=0 (rows 0..M-m),
        then j=1, etc.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    M, N = a.shape
    m, n = block
    out_rows = m * n
    out_cols = (M - m + 1) * (N - n + 1)

    # Build a (M-m+1, N-n+1, m, n) sliding-window view.
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(a, (m, n))            # (M-m+1, N-n+1, m, n)
    # Inside each block the elements must be stored column-first (MATLAB F
    # order), so we swap the last two axes before flattening with C order.
    blocks = np.transpose(win, (0, 1, 3, 2))        # (i, j, n_cols, m_rows)
    blocks = blocks.reshape(M - m + 1, N - n + 1, m * n)
    # Block positions themselves are enumerated column-major (i runs first):
    out = np.transpose(blocks, (1, 0, 2))           # (j, i, m*n)
    out = out.reshape(out_cols, out_rows)
    return out.T                                    # (m*n, out_cols)


# ═══════════════════════════════════════════════════════════════════════════
# PSF <-> OTF
# ═══════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    MATLAB psf2otf(psf, shape):
        zero-pad PSF into ``shape``; circularly shift so that the PSF
        centre lands at index (0, 0); return fft2.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: Tuple[int, int]) -> np.ndarray:
    """MATLAB otf2psf(otf, psf_size)."""
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═══════════════════════════════════════════════════════════════════════════
# Conjugate gradient (linear solver) — CG.m
# ═══════════════════════════════════════════════════════════════════════════

def cg_solve(x0: np.ndarray,
             a_func: Callable[[np.ndarray], np.ndarray],
             rhs: np.ndarray,
             max_iter: int = 15,
             tol: float = 1e-5) -> Tuple[np.ndarray, int]:
    """
    Conjugate gradient for symmetric positive-definite linear systems.

    Direct port of CG.m.  Solves  A x = rhs  where the linear operator
    ``A`` is applied through ``a_func(x)``.

    Notes
    -----
    The MATLAB original contains a small typo (`if i>25 && decr<CG_TOL`
    where the loop counter is `it`), which means the early-exit branch
    never fires for the default MAX_ITER=15.  We faithfully reproduce
    that behaviour: the inner check is kept, but is intentionally only
    effective for `it > 25` AND `decr < tol`, matching the MATLAB code.
    """
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

        # MATLAB has: if i>25 && decr<CG_TOL, break; end  (i is undefined →
        # condition never True for default MAX_ITER=15).  We mirror that.
        if it > 25 and decr < tol:
            x = x_new
            break

        x = x_new

    return x, it


# ═══════════════════════════════════════════════════════════════════════════
# Kernel centering — center_kernel.m
# ═══════════════════════════════════════════════════════════════════════════

def center_kernel(k: np.ndarray,
                  xf: Optional[Sequence[np.ndarray]] = None
                  ) -> Tuple[np.ndarray, Optional[list]]:
    """
    Shift kernel so its centre of mass lands at the geometric centre.

    Mirrors center_kernel.m.  When ``xf`` is given, the same shift is
    applied to each filtered image.

    Returns
    -------
    k_centered : ndarray
    xf_centered : list[ndarray] | None
    """
    kh, kw = k.shape
    # MATLAB uses 1-based indices: c_x = sum( (1:size(k,2)) .* sum(k,1) )
    cx = float(np.sum(np.arange(1, kw + 1) * k.sum(axis=0)))
    cy = float(np.sum(np.arange(1, kh + 1) * k.sum(axis=1)))

    # MATLAB round() is half-away-from-zero (banker's rounding is NOT used)
    def _mr(x: float) -> int:
        return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))

    offset_x = _mr(np.floor(kw / 2.0) + 1 - cx)
    offset_y = _mr(np.floor(kh / 2.0) + 1 - cy)

    if offset_x == 0 and offset_y == 0:
        return k.copy(), (list(xf) if xf is not None else None)

    sh_rows = abs(offset_y) * 2 + 1
    sh_cols = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sh_rows, sh_cols), dtype=np.float64)
    # MATLAB: shift_kernel(abs(offset_y)+1+offset_y, abs(offset_x)+1+offset_x) = 1
    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    k_centered = conv2_same(k, shift_kernel)

    if xf is None:
        return k_centered, None

    xf_centered = [conv2_same(x, shift_kernel) for x in xf]
    return k_centered, xf_centered


# ═══════════════════════════════════════════════════════════════════════════
# Kernel cleanup — clean_kernel_ECP.m
# ═══════════════════════════════════════════════════════════════════════════

def clean_kernel_ecp(k: np.ndarray) -> np.ndarray:
    """
    Drop weak connected components and renormalise — clean_kernel_ECP.m.

    For 8-connected components of ``k`` whose summed weight is below 0.1
    of the total mass, all pixels are zeroed.  Negative values are
    clipped to zero and the kernel is normalised so that ``sum(k) == 1``.
    """
    k = k.copy()
    # 8-connectivity == 3x3 structuring element of ones.
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


# ═══════════════════════════════════════════════════════════════════════════
# Bound-constrained quadratic programming  — MATLAB quadprog
# ═══════════════════════════════════════════════════════════════════════════

def quadprog_nonneg(H: np.ndarray,
                    f: np.ndarray,
                    x0: Optional[np.ndarray] = None,
                    max_iter: int = 200,
                    tol: float = 1e-8) -> np.ndarray:
    """
    Replicate MATLAB ``quadprog(H, f, [], [], [], [], lb=0, [])``.

    Minimises  0.5 * x' H x + f' x   subject to  x >= 0.

    Implementation
    --------------
    Uses scipy.optimize.minimize (L-BFGS-B) with analytic gradient and
    lower bound 0.  This is a standard quadratic program on a non-negative
    orthant; L-BFGS-B converges to the exact MATLAB solution to machine
    precision when ``H`` is symmetric PSD (as in our kernel-estimation
    sub-problem).
    """
    n = H.shape[0]
    H = 0.5 * (H + H.T)  # symmetrise for safety

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
