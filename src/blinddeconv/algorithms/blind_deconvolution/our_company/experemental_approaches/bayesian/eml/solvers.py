"""
solvers.py

Core solvers of the EML (Efficient Marginal Likelihood) blind deconvolution
algorithm of Levin et al., CVPR 2011.

This module ports the recommended pipeline of the reference MATLAB package
(``LevinEtalCVPR2011Code/BlindDeconvCode``):

    deconv_diagfe_filt_sps.m
        → multires_deconv.m
            → deconv1.m
                → update_x_conjgrad_diagfe_filt_space.m
                    → conjgrad_deconv_g.m
                → update_k.m
                    → getAutoCor.m / getCory.m / getCorAbDiagCov.m
                    → solve_for_sps_kernel.m (positivity-constrained QP)
        → deconvSps.m  (final non-blind restoration, Levin SIGGRAPH 2007)
            → deconvL2_w.m

All functions match the MATLAB originals as closely as possible.
See utils.py for the MATLAB↔Python convention notes.

Implementation notes
────────────────────
* The positivity-constrained QP ``solve_for_sps_kernel`` is solved with
  ``scipy.optimize.minimize(method='L-BFGS-B')`` with bounds ``k ≥ 0``.  The
  objective is the same convex quadratic  ``½ kᵀAk − bᵀk``  as MATLAB's
  ``quadprog(A, -b, [], [], [], [], zeros(...))``.

* ``getAutoCor`` returns an A-matrix whose linear index ordering is the
  MATLAB column-major one (``i = i2·k_sz1 + i1 + 1``).  Therefore the
  solution vector is reshaped to ``(k_sz1, k_sz2)`` with
  ``order='F'``, matching MATLAB's ``reshape(k, k_sz1, k_sz2)``.

* The problem dict ``prob`` is the Python equivalent of the MATLAB struct
  of the same name (see README.txt of the reference package for the list
  of fields).  Mutations happen in place on the supplied dict.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d
from scipy.optimize import minimize

from .utils import (
    flp,
    zero_pad2,
    goodfactor,
    normexp,
    fftconvf,
    downSmpImC,
    resizeKer,
    set_sizes,
    filt_y,
)


# ════════════════════════════════════════════════════════════════════════════
# conjgrad_deconv_g  (from conjgrad_deconv_g.m)
# ════════════════════════════════════════════════════════════════════════════

def _pad_replicate(ty: np.ndarray, hfs_y1: int, hfs_y2: int,
                   hfs_x1: int, hfs_x2: int) -> np.ndarray:
    """
    Replicate-pad an image, equivalent to MATLAB
        ty([ones(1,hfs_y1), 1:end, end*ones(1,hfs_y2)],
           [ones(1,hfs_x1), 1:end, end*ones(1,hfs_x2)]).
    """
    return np.pad(ty,
                  ((hfs_y1, hfs_y2), (hfs_x1, hfs_x2)),
                  mode='edge')


def conjgrad_deconv_g(y: np.ndarray, k: np.ndarray, we: float,
                      max_it: int = 200,
                      weight_i: np.ndarray = None,
                      x: np.ndarray = None) -> np.ndarray:
    """
    Weighted conjugate-gradient deconvolution in the gradient domain.

    Solves  (1/σ²·Kᵀ M K + diag(w)) x = 1/σ²·Kᵀ M y  by CG, where M is a
    mask that zeroes the boundary padding.

    Equivalent to MATLAB conjgrad_deconv_g.m.

    Parameters
    ----------
    y        : (N1_orig, N2_orig) blurred (gradient) image
    k        : (k1, k2) kernel
    we       : σ²  (noise variance scalar)  [same name as MATLAB ``we``]
    max_it   : CG iterations (default 200)
    weight_i : (N1, N2) per-pixel Tikhonov weights, where
               N1 = N1_orig + k1 - 1, N2 = N2_orig + k2 - 1.
    x        : (N1, N2) initial guess.  Default: replicate-pad y.
    """
    y = np.asarray(y, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    N1o, N2o = y.shape
    fs_y, fs_x = k.shape
    hfs1_x1 = (fs_x - 1) // 2            # MATLAB floor((sz-1)/2)
    hfs1_x2 = (fs_x - 1) - hfs1_x1        # MATLAB ceil((sz-1)/2)
    hfs1_y1 = (fs_y - 1) // 2
    hfs1_y2 = (fs_y - 1) - hfs1_y1
    hfs_x1, hfs_x2 = hfs1_x1, hfs1_x2
    hfs_y1, hfs_y2 = hfs1_y1, hfs1_y2

    N1 = N1o + hfs_y1 + hfs_y2
    N2 = N2o + hfs_x1 + hfs_x2
    mask = np.zeros((N1, N2), dtype=np.float64)
    mask[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2] = 1.0

    if weight_i is None:
        weight_i = np.ones((N1, N2), dtype=np.float64)

    ty = y
    y = np.zeros((N1, N2), dtype=np.float64)
    y[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2] = ty

    if x is None:
        x = _pad_replicate(ty, hfs_y1, hfs_y2, hfs_x1, hfs_x2).astype(np.float64)
    else:
        x = np.asarray(x, dtype=np.float64).copy()

    # b = conv2(y .* mask, k, 'same')
    b = convolve2d(y * mask, k, mode='same')

    # Pad k to a good-FFT size for fast convolution.
    N1p = goodfactor(N1 + hfs1_y1 + hfs1_y2)
    N2p = goodfactor(N2 + hfs1_x1 + hfs1_x2)
    K_pad = zero_pad2(
        k,
        int(np.ceil((N1p - fs_y) / 2)), int(np.floor((N1p - fs_y) / 2)),
        int(np.ceil((N2p - fs_x) / 2)), int(np.floor((N2p - fs_x) / 2)),
    )
    K = fft2(ifftshift(K_pad))

    small_kernel = max(fs_y, fs_x) <= 5

    def _apply_A(v: np.ndarray) -> np.ndarray:
        if small_kernel:
            inner = convolve2d(v, flp(k), mode='same') * mask
            out = convolve2d(inner, k, mode='same')
        else:
            inner = fftconvf(v, flp(k), np.conj(K), 'same') * mask
            out = fftconvf(inner, k, K, 'same')
        return out + we * weight_i * v

    Ax = _apply_A(x)
    r = b - Ax
    rho = float(r.ravel() @ r.ravel())
    p = None
    rho_1 = None

    for it in range(max_it):
        rho = float(r.ravel() @ r.ravel())
        if rho < 1e-8:
            break
        if it > 0:
            beta = rho / rho_1
            p = r + beta * p
        else:
            p = r.copy()
        q = _apply_A(p)
        alpha = rho / float(p.ravel() @ q.ravel())
        x = x + alpha * p
        r = r - alpha * q
        rho_1 = rho

    return x


# ════════════════════════════════════════════════════════════════════════════
# deconvL2_w  (from deconvL2_w.m)
# ════════════════════════════════════════════════════════════════════════════

def _full_conv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``conv2(a, b)`` without shape arg == 'full' mode."""
    return convolve2d(a, b, mode='full')


def deconvL2_w(I: np.ndarray, k: np.ndarray, we: float, max_it: int,
               weight_x: np.ndarray, weight_y: np.ndarray,
               weight_xx: np.ndarray, weight_yy: np.ndarray,
               weight_xy: np.ndarray) -> np.ndarray:
    """
    Weighted L2 non-blind deconvolution with 1st and 2nd order gradient
    regularisers.  Equivalent to MATLAB ``deconvL2_w.m``.

    The output has the size of the replicate-padded image (N1, N2),
    exactly as in MATLAB (the caller crops back).
    """
    I = np.asarray(I, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    N1o, N2o = I.shape
    fs_y, fs_x = k.shape
    hfs1_x1 = (fs_x - 1) // 2
    hfs1_x2 = (fs_x - 1) - hfs1_x1
    hfs1_y1 = (fs_y - 1) // 2
    hfs1_y2 = (fs_y - 1) - hfs1_y1
    hfs_x1, hfs_x2 = hfs1_x1, hfs1_x2
    hfs_y1, hfs_y2 = hfs1_y1, hfs1_y2

    N1 = N1o + hfs_y1 + hfs_y2
    N2 = N2o + hfs_x1 + hfs_x2
    mask = np.zeros((N1, N2), dtype=np.float64)
    mask[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2] = 1.0

    tI = I
    I = np.zeros((N1, N2), dtype=np.float64)
    I[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2] = tI
    x = _pad_replicate(tI, hfs_y1, hfs_y2, hfs_x1, hfs_x2).astype(np.float64)

    b = convolve2d(x * mask, k, mode='same')

    N1p = goodfactor(N1 + hfs1_y1 + hfs1_y2)
    N2p = goodfactor(N2 + hfs1_x1 + hfs1_x2)
    K_pad = zero_pad2(
        k,
        int(np.ceil((N1p - fs_y) / 2)), int(np.floor((N1p - fs_y) / 2)),
        int(np.ceil((N2p - fs_x) / 2)), int(np.floor((N2p - fs_x) / 2)),
    )
    K = fft2(ifftshift(K_pad))

    # Derivative filters (identical to MATLAB).
    dxf = np.array([[1.0, -1.0]])
    dyf = np.array([[1.0], [-1.0]])
    dyyf = np.array([[-1.0], [2.0], [-1.0]])
    dxxf = np.array([[-1.0, 2.0, -1.0]])
    dxyf = np.array([[-1.0, 1.0], [1.0, -1.0]])

    small_kernel = max(fs_y, fs_x) <= 5

    def _apply_A(v: np.ndarray) -> np.ndarray:
        if small_kernel:
            inner = convolve2d(v, flp(k), mode='same') * mask
            out = convolve2d(inner, k, mode='same')
        else:
            inner = fftconvf(v, flp(k), np.conj(K), 'same') * mask
            out = fftconvf(inner, k, K, 'same')
        # Regularisation: ∑ we · conv2(w_f · conv2(v, flp(f), 'valid'), f, 'full')
        out = out + we * _full_conv(weight_x * convolve2d(v, flp(dxf), 'valid'), dxf)
        out = out + we * _full_conv(weight_y * convolve2d(v, flp(dyf), 'valid'), dyf)
        out = out + we * _full_conv(weight_xx * convolve2d(v, flp(dxxf), 'valid'), dxxf)
        out = out + we * _full_conv(weight_yy * convolve2d(v, flp(dyyf), 'valid'), dyyf)
        out = out + we * _full_conv(weight_xy * convolve2d(v, flp(dxyf), 'valid'), dxyf)
        return out

    Ax = _apply_A(x)
    r = b - Ax
    p = None
    rho_1 = None

    for it in range(max_it):
        rho = float(r.ravel() @ r.ravel())
        if it > 0:
            beta = rho / rho_1
            p = r + beta * p
        else:
            p = r.copy()
        q = _apply_A(p)
        denom = float(p.ravel() @ q.ravel())
        if denom == 0.0:
            break
        alpha = rho / denom
        x = x + alpha * p
        r = r - alpha * q
        rho_1 = rho

    return x


# ════════════════════════════════════════════════════════════════════════════
# deconvSps  (from deconvSps.m) — final non-blind deconvolution
# ════════════════════════════════════════════════════════════════════════════

def deconvSps(I: np.ndarray, k: np.ndarray, we: float,
              max_it: int = 200) -> np.ndarray:
    """
    Sparse non-blind deconvolution with |z|^0.8 prior on 1st/2nd derivatives.
    Equivalent to MATLAB deconvSps.m (Levin SIGGRAPH 2007).

    Parameters
    ----------
    I      : (H, W) blurred image
    k      : (k1, k2) kernel (odd dimensions)
    we     : regularisation / noise scale (MATLAB ``edges_w``, default 0.0068)
    max_it : CG iterations per IRLS pass (default 200)

    Returns
    -------
    x : (H, W) restored image, cropped back to the size of I.
    """
    I = np.asarray(I, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    N1o, N2o = I.shape
    fs_y, fs_x = k.shape
    hfs1_x1 = (fs_x - 1) // 2
    hfs1_x2 = (fs_x - 1) - hfs1_x1
    hfs1_y1 = (fs_y - 1) // 2
    hfs1_y2 = (fs_y - 1) - hfs1_y1
    hfs_x1, hfs_x2 = hfs1_x1, hfs1_x2
    hfs_y1, hfs_y2 = hfs1_y1, hfs1_y2

    N1 = N1o + hfs_y1 + hfs_y2
    N2 = N2o + hfs_x1 + hfs_x2

    # I padded with zeros (for likelihood term), x initialised to same
    tI = I
    I_pad = np.zeros((N1, N2), dtype=np.float64)
    I_pad[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2] = tI

    # Derivative filters (for re-weighting)
    dxf = np.array([[1.0, -1.0]])
    dyf = np.array([[1.0], [-1.0]])
    dyyf = np.array([[-1.0], [2.0], [-1.0]])
    dxxf = np.array([[-1.0, 2.0, -1.0]])
    dxyf = np.array([[-1.0, 1.0], [1.0, -1.0]])

    # First pass: unit weights
    weight_x = np.ones((N1, N2 - 1), dtype=np.float64)
    weight_y = np.ones((N1 - 1, N2), dtype=np.float64)
    weight_xx = np.ones((N1, N2 - 2), dtype=np.float64)
    weight_yy = np.ones((N1 - 2, N2), dtype=np.float64)
    weight_xy = np.ones((N1 - 1, N2 - 1), dtype=np.float64)

    x = deconvL2_w(tI, k, we, max_it,
                   weight_x, weight_y, weight_xx, weight_yy, weight_xy)

    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01

    for _ in range(2):
        dy = convolve2d(x, flp(dyf), mode='valid')
        dx = convolve2d(x, flp(dxf), mode='valid')
        dyy = convolve2d(x, flp(dyyf), mode='valid')
        dxx = convolve2d(x, flp(dxxf), mode='valid')
        dxy = convolve2d(x, flp(dxyf), mode='valid')

        weight_x = w0 * np.maximum(np.abs(dx), thr_e) ** (exp_a - 2)
        weight_y = w0 * np.maximum(np.abs(dy), thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        x = deconvL2_w(
            I_pad[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2],
            k, we, max_it,
            weight_x, weight_y, weight_xx, weight_yy, weight_xy,
        )

    x = x[hfs_y1:N1 - hfs_y2, hfs_x1:N2 - hfs_x2]
    return x


# ════════════════════════════════════════════════════════════════════════════
# getAutoCor / getCory  (from getAutoCor.m, getCory.m)
# ════════════════════════════════════════════════════════════════════════════

def getAutoCor(x: np.ndarray, k_sz1: int, k_sz2: int) -> np.ndarray:
    """
    Efficient computation of the auto-correlation matrix A of all
    (k_sz1 × k_sz2) sliding windows in x.  Equivalent to
    MATLAB ``getAutoCor.m``.

    The linear index ordering is MATLAB column-major:
        i = i2 * k_sz1 + i1   (0-based),
    so the returned A can be used as-is in a system whose solution k is
    reshaped with ``order='F'``.
    """
    x = np.asarray(x, dtype=np.float64)
    M1, M2 = x.shape
    sM1 = M1 - k_sz1 + 1
    sM2 = M2 - k_sz2 + 1
    k_sz = k_sz1 * k_sz2

    A = np.zeros((k_sz, k_sz), dtype=np.float64)

    # ── Block 1: shifts (+d1, +d2), d1 = 0..k_sz1-1, d2 = 0..k_sz2-1 ──
    for d2 in range(k_sz2):
        for d1 in range(k_sz1):
            # xx = x(1:end-d1, 1:end-d2) .* x(d1+1:end, d2+1:end)
            if d1 == 0:
                xa_r = x
                xb_r = x
            else:
                xa_r = x[:-d1, :]
                xb_r = x[d1:, :]
            if d2 == 0:
                xa = xa_r
                xb = xb_r
            else:
                xa = xa_r[:, :-d2]
                xb = xb_r[:, d2:]
            xx = xa * xb
            cs = np.cumsum(np.cumsum(xx, axis=0), axis=1)

            for j2 in range(k_sz2):
                for j1 in range(k_sz1):
                    i1 = j1 + d1
                    i2 = j2 + d2
                    i = i2 * k_sz1 + i1
                    j = j2 * k_sz1 + j1
                    if (i >= k_sz) or (i1 >= k_sz1) or (i2 >= k_sz2) \
                            or (i1 < 0) or (i2 < 0):
                        continue
                    ts = cs[j1 + sM1 - 1, j2 + sM2 - 1]
                    if j1 > 0:
                        ts -= cs[j1 - 1, j2 + sM2 - 1]
                    if j2 > 0:
                        ts -= cs[j1 + sM1 - 1, j2 - 1]
                    if (j1 > 0) and (j2 > 0):
                        ts += cs[j1 - 1, j2 - 1]
                    A[j, i] = ts
                    A[i, j] = ts

    # ── Block 2: shifts (-d1, +d2), d1 = 1..k_sz1-1, d2 = 0..k_sz2-1 ──
    # MATLAB:
    #   xx = x(d1+1:end, 1:end-d2) .* x(1:end-d1, d2+1:end)
    #   for j2=0..k_sz2-1, for j1=d1..k_sz1-1:
    #     i1 = j1 - d1;  i2 = j2 + d2;
    #     ts = cs(j1-d1+sM1, j2+sM2) [minus boundary terms]
    for d2 in range(k_sz2):
        for d1 in range(1, k_sz1):
            # x(d1+1:end, ...) → 0-based x[d1:, ...]
            # x(1:end-d1, ...) → 0-based x[:-d1, ...]
            xa_r = x[d1:, :]
            xb_r = x[:-d1, :]
            if d2 == 0:
                xa = xa_r
                xb = xb_r
            else:
                xa = xa_r[:, :-d2]
                xb = xb_r[:, d2:]
            xx = xa * xb
            cs = np.cumsum(np.cumsum(xx, axis=0), axis=1)

            for j2 in range(k_sz2):
                for j1 in range(d1, k_sz1):
                    i1 = j1 - d1
                    i2 = j2 + d2
                    i = i2 * k_sz1 + i1
                    j = j2 * k_sz1 + j1
                    if (i >= k_sz) or (i1 >= k_sz1) or (i2 >= k_sz2) \
                            or (i1 < 0) or (i2 < 0):
                        continue
                    # MATLAB 1-based cs(j1-d1+sM1, j2+sM2) → 0-based
                    ts = cs[(j1 - d1) + sM1 - 1, j2 + sM2 - 1]
                    if j1 > d1:
                        ts -= cs[(j1 - d1) - 1, j2 + sM2 - 1]
                    if j2 > 0:
                        ts -= cs[(j1 - d1) + sM1 - 1, j2 - 1]
                    if (j1 > d1) and (j2 > 0):
                        ts += cs[(j1 - d1) - 1, j2 - 1]
                    A[j, i] = ts
                    A[i, j] = ts

    return A


def getCory(x: np.ndarray, y: np.ndarray,
            k_sz1: int, k_sz2: int) -> np.ndarray:
    """
    Cross-correlation vector b = Xᵀ y for all (k_sz1 × k_sz2) sliding
    windows of x against the smaller y.  Equivalent to ``getCory.m``.

    Linear index ordering is MATLAB column-major (matches ``getAutoCor``).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n1, n2 = x.shape
    k_sz = k_sz1 * k_sz2
    b = np.zeros(k_sz, dtype=np.float64)

    # Python end indices: MATLAB (1+d:end-k_sz+1+d) == (d : n-k_sz+1+d) 0-based
    d = 0
    for d2 in range(k_sz2):
        for d1 in range(k_sz1):
            x_slice = x[d1:n1 - k_sz1 + 1 + d1, d2:n2 - k_sz2 + 1 + d2]
            b[d] = np.sum(x_slice * y)
            d += 1
    return b


# ════════════════════════════════════════════════════════════════════════════
# getCorAbDiagCov  (from getCorAbDiagCov.m)
# ════════════════════════════════════════════════════════════════════════════

def getCorAbDiagCov(x: np.ndarray, y: np.ndarray, xcov: np.ndarray,
                    k_sz1: int, k_sz2: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build (A, b, c) for the kernel M-step under a diagonal covariance
    approximation.  Equivalent to ``getCorAbDiagCov.m``.

    A = getAutoCor(x) + diag-contributions of xcov
    b = getCory(x, y)
    c = ‖y‖²
    """
    A = getAutoCor(x, k_sz1, k_sz2)
    b = getCory(x, y, k_sz1, k_sz2)

    M1, M2 = x.shape
    N1, N2 = y.shape

    # cs = cumsum(cumsum(xcov, 1), 2)
    cs = np.cumsum(np.cumsum(xcov, axis=0), axis=1)

    ind = 0
    for i2 in range(1, k_sz2 + 1):      # MATLAB i2 = 1..k_sz2
        for i1 in range(1, k_sz1 + 1):  # MATLAB i1 = 1..k_sz1
            # ts = cs(i1 + N1 - 1, i2 + N2 - 1)   (1-based)
            r = (i1 + N1 - 1) - 1
            c_ = (i2 + N2 - 1) - 1
            ts = cs[r, c_]
            if i1 > 1:
                ts -= cs[i1 - 2, c_]
            if i2 > 1:
                ts -= cs[r, i2 - 2]
            if (i1 > 1) and (i2 > 1):
                ts += cs[i1 - 2, i2 - 2]
            A[ind, ind] = A[ind, ind] + ts
            ind += 1

    c = float(np.sum(np.abs(y) ** 2))
    return A, b, c


# ════════════════════════════════════════════════════════════════════════════
# solve_for_sps_kernel / solve_for_sps_kernel_unconst
# ════════════════════════════════════════════════════════════════════════════

def _qp_nonneg(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve  min  ½ kᵀA k − bᵀk   s.t. k ≥ 0.
    A is assumed symmetric positive semi-definite.
    Equivalent to MATLAB  ``quadprog(A, -b, [], [], [], [], zeros(n, 1))``.
    """
    n = A.shape[0]
    A = 0.5 * (A + A.T)  # symmetrise

    def _f(v):
        Av = A @ v
        return 0.5 * float(v @ Av) - float(b @ v), Av - b

    x0 = np.zeros(n, dtype=np.float64)
    bounds = [(0.0, None)] * n
    res = minimize(_f, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
    return np.maximum(res.x, 0.0)


def solve_for_sps_kernel(A: np.ndarray, b: np.ndarray,
                         k_sz1: int, k_sz2: int,
                         scla: float = 0.005) -> np.ndarray:
    """
    Kernel M-step with positivity and ``|k|^0.5``-style IRLS sparse prior.
    Equivalent to MATLAB ``solve_for_sps_kernel.m``.
    """
    exp_a = 0.5
    thr_0 = 1e-4

    A0 = 0.5 * (A + A.T)
    k = _qp_nonneg(A0, b)

    for _ in range(2):
        w = np.maximum(np.abs(k), thr_0) ** (exp_a - 2.0)
        k = _qp_nonneg(A0 + scla * np.diag(w), b)

    return k.reshape((k_sz1, k_sz2), order='F')


def solve_for_sps_kernel_unconst(A: np.ndarray, b: np.ndarray,
                                 k_sz1: int, k_sz2: int,
                                 scla: float = 0.005) -> np.ndarray:
    """
    Unconstrained version (no positivity), via linear solve.  Equivalent
    to MATLAB ``solve_for_sps_kernel_unconst.m``.
    """
    exp_a = 0.5
    thr_0 = 1e-4
    A0 = 0.5 * (A + A.T)

    k = np.linalg.solve(A0, b)
    for _ in range(2):
        w = np.maximum(np.abs(k), thr_0) ** (exp_a - 2.0)
        k = np.linalg.solve(A0 + scla * np.diag(w), b)
    return k.reshape((k_sz1, k_sz2), order='F')


# ════════════════════════════════════════════════════════════════════════════
# update_x_conjgrad_diagfe_filt_space  (main E-step of the recommended algo)
# ════════════════════════════════════════════════════════════════════════════

def update_x_conjgrad_diagfe_filt_space(prob: dict) -> dict:
    """
    E-step: update x (and its diagonal covariance) under a MOG prior on
    filter-space variables, with the free-energy diagonal approximation
    and conjugate-gradient solver.

    Equivalent to MATLAB ``update_x_conjgrad_diagfe_filt_space.m``.
    """
    sig_noise = prob['sig_noise']
    filty = prob['filty']
    N1, N2, N3 = filty.shape

    k_sz1 = prob['k_sz1']
    k_sz2 = prob['k_sz2']
    M1 = N1 + k_sz1 - 1
    M2 = N2 + k_sz2 - 1
    M = M1 * M2

    prior_ivar = np.asarray(prob['prior_ivar'], dtype=np.float64).ravel()
    prior_pi = np.asarray(prob['prior_pi'], dtype=np.float64).ravel()
    L = prior_ivar.size

    # mask for A_1 diagonal:  zero_pad(ones(N1,N2), (k_sz1-1)/2, (k_sz2-1)/2)
    pad_h = (k_sz1 - 1) // 2
    pad_w = (k_sz2 - 1) // 2
    mask = np.zeros((M1, M2), dtype=np.float64)
    mask[pad_h:pad_h + N1, pad_w:pad_w + N2] = 1.0
    k = prob['k']
    da1 = (1.0 / sig_noise ** 2) * convolve2d(mask, np.abs(k) ** 2, mode='same')

    init_iv = float(np.sum(prior_ivar * prior_pi))

    # itrN = 2*(L > 1) + 1
    itrN = 2 * (1 if L > 1 else 0) + 1

    use_prev_x = ((not prob.get('init_x_every_itr', 1))
                  and (prob.get('filtx', None) is not None
                       and np.size(prob.get('filtx', np.array([]))) > 0)
                  and (L > 1))

    # Allocate outputs
    filtx_out = np.zeros((M1, M2, N3), dtype=np.float64)
    filtxcov_out: List[np.ndarray] = [None] * N3
    freeeng_qlogp_ycx = np.zeros(N3)
    freeeng_qlogp_x = np.zeros(N3)
    freeeng_qpilogqpi = np.zeros(N3)
    freeeng_qxlogqx = np.zeros(N3)

    # Pre-flipped k used inside the free-energy computation.
    k_flip = flp(k)

    for j in range(N3):

        if use_prev_x:
            x = prob['filtx'][:, :, j].astype(np.float64).copy()
            xcov = prob['filtxcov'][j].astype(np.float64).copy()
        else:
            x = None
            xcov = None

        cpi = None
        da2 = None
        w = None

        for itr in range(itrN):
            if (itr == 0) and (not use_prev_x):
                w = init_iv * np.ones((M1, M2), dtype=np.float64)
                cpi = np.tile(prior_pi[np.newaxis, :], (M, 1))
            else:
                ex2 = np.abs(x).ravel(order='F') ** 2 + xcov.ravel(order='F')
                # logpi = -0.5 * ex2 * prior_ivar + log(prior_pi) + 0.5*log(prior_ivar)
                logpi = (-0.5 * np.outer(ex2, prior_ivar)
                         + np.ones((M, 1)) * (np.log(prior_pi)
                                              + 0.5 * np.log(prior_ivar)))
                cpi = normexp(logpi)
                w_vec = cpi @ prior_ivar            # (M,)
                w = w_vec.reshape((M1, M2), order='F')

            # Solve weighted deconvolution for the mean μ_j = x.
            x = conjgrad_deconv_g(filty[:, :, j], k, sig_noise ** 2,
                                  15, w)
            da2 = w
            xcov = 1.0 / (da1 + da2)

        # Free-energy terms (for monitoring; replicate MATLAB code).
        sumA1xcov = float(np.sum(da1 * xcov))
        sumA2xcov = float(np.sum(da2 * xcov))
        xA1x = (1.0 / sig_noise ** 2) * float(np.sum(
            np.abs(convolve2d(x, k_flip, mode='valid')) ** 2))
        xA2x = float(np.sum(np.abs(da2 * x ** 2)))

        xb = (1.0 / sig_noise ** 2) * float(np.sum(
            x.conj() * convolve2d(filty[:, :, j], k, mode='full')
        ))
        ynorm = (1.0 / sig_noise ** 2) * float(
            np.sum(np.abs(filty[:, :, j]) ** 2))

        filtx_out[:, :, j] = x
        filtxcov_out[j] = xcov

        freeeng_qlogp_ycx[j] = 0.5 * (sumA1xcov + xA1x - 2 * xb + ynorm)
        freeeng_qlogp_x[j] = (
            0.5 * (sumA2xcov + xA2x)
            + float(np.sum(cpi @ (-0.5 * np.log(prior_ivar)
                                  - np.log(prior_pi))))
        )
        freeeng_qpilogqpi[j] = float(
            np.sum(cpi * np.log(np.maximum(cpi, 1e-15))))
        freeeng_qxlogqx[j] = (
            -0.5 * float(np.sum(np.log(np.abs(xcov))))
            - (1.0 + np.log(2.0 * np.pi)) * M / 2.0
        )

    prob['filtx'] = filtx_out
    prob['filtxcov'] = filtxcov_out
    prob['freeeng_qlogp_ycx'] = freeeng_qlogp_ycx
    prob['freeeng_qlogp_x'] = freeeng_qlogp_x
    prob['freeeng_qpilogqpi'] = freeeng_qpilogqpi
    prob['freeeng_qxlogqx'] = freeeng_qxlogqx
    prob['freeeng'] = float(freeeng_qlogp_ycx.sum()
                            + freeeng_qlogp_x.sum()
                            + freeeng_qpilogqpi.sum()
                            + freeeng_qxlogqx.sum())
    return prob


# ════════════════════════════════════════════════════════════════════════════
# update_k  (from update_k.m) — diag covariance branch only
# ════════════════════════════════════════════════════════════════════════════

def update_k(prob: dict) -> dict:
    """
    M-step: update kernel k.  Equivalent to ``update_k.m`` restricted to
    the diagonal-covariance, filter-space branch (the recommended algo).
    """
    sig_noise = prob['sig_noise']
    k_sz1 = prob['k_sz1']
    k_sz2 = prob['k_sz2']

    if prob.get('filt_space', 0):
        x = prob['filtx']
        xcov_list = prob['filtxcov']
        y = prob['filty']
    else:
        x = prob['x']
        xcov_list = prob['xcov']
        y = prob['y']

    covtype = prob.get('covtype', 'diag')
    if covtype != 'diag':
        raise NotImplementedError(
            f"update_k: covtype='{covtype}' not implemented in this port."
        )

    k_sz = prob['k_sz']
    A = np.zeros((k_sz, k_sz), dtype=np.float64)
    b = np.zeros(k_sz, dtype=np.float64)
    c = 0.0
    AL: List[np.ndarray] = []
    bL: List[np.ndarray] = []
    cL: List[float] = []

    N3 = x.shape[2]
    for j in range(N3):
        tA, tb, tc = getCorAbDiagCov(x[:, :, j], y[:, :, j],
                                     xcov_list[j], k_sz1, k_sz2)
        A = A + tA
        b = b + tb
        c = c + tc
        AL.append(tA)
        bL.append(tb)
        cL.append(tc)

    k_prior_ivar = float(prob.get('k_prior_ivar', 0.01))
    if prob.get('unconst_k', 0):
        k_new = solve_for_sps_kernel_unconst(A, b, k_sz1, k_sz2, k_prior_ivar)
    else:
        k_new = solve_for_sps_kernel(A, b, k_sz1, k_sz2, k_prior_ivar)
    prob['k'] = k_new
    k_vec = k_new.ravel(order='F')

    # Recompute the likelihood term of the free energy.
    freeeng_qlogp_ycx = np.zeros(N3)
    for j in range(N3):
        freeeng_qlogp_ycx[j] = (
            1.0 / (2.0 * sig_noise ** 2)
            * (k_vec @ AL[j] @ k_vec - 2.0 * k_vec @ bL[j] + cL[j])
        )
    prob['freeeng_qlogp_ycx'] = freeeng_qlogp_ycx
    prob['freeeng'] = float(freeeng_qlogp_ycx.sum()
                            + prob['freeeng_qlogp_x'].sum()
                            + prob['freeeng_qpilogqpi'].sum()
                            + prob['freeeng_qxlogqx'].sum())
    return prob


# ════════════════════════════════════════════════════════════════════════════
# deconv1  (from deconv1.m)  — inner EM loop for one pyramid level
# ════════════════════════════════════════════════════════════════════════════

def deconv1(prob: dict, sig_noise_v: np.ndarray,
            verbose: bool = False) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    EM loop on one pyramid level.  Equivalent to MATLAB ``deconv1.m``.

    Only dispatches the combination actually used by the recommended
    diagfe_filt_sps algorithm; raises NotImplementedError otherwise.

    Returns
    -------
    prob    : updated problem dict
    kList   : (k_sz1, k_sz2, maxItr) history of kernels per iteration
    freeeng : (2, maxItr) free energies after x- and k-updates
    """
    sig_noise_v = np.asarray(sig_noise_v, dtype=np.float64).ravel()
    maxItr = sig_noise_v.size

    combo = (prob.get('update_x'), prob.get('covtype'),
             int(prob.get('filt_space', 0)))
    if combo != ('conjgrad', 'diag', 1):
        raise NotImplementedError(
            f"deconv1: algorithm combo {combo} not ported; "
            "only ('conjgrad','diag', filt_space=1) is supported."
        )

    kList = np.zeros((prob['k_sz1'], prob['k_sz2'], maxItr), dtype=np.float64)
    freeeng = np.zeros((2, maxItr), dtype=np.float64)

    for itr in range(maxItr):
        prob['sig_noise'] = float(sig_noise_v[itr])

        prob = update_x_conjgrad_diagfe_filt_space(prob)
        if verbose:
            print(f"itr={itr+1:02d}, free eng after x update: "
                  f"{prob['freeeng']:.4f}")
        freeeng[0, itr] = prob['freeeng']

        prob = update_k(prob)
        if verbose:
            print(f"itr={itr+1:02d}, free eng after k update: "
                  f"{prob['freeeng']:.4f}")
        freeeng[1, itr] = prob['freeeng']

        kList[:, :, itr] = prob['k']

    return prob, kList, freeeng


# ════════════════════════════════════════════════════════════════════════════
# multires_deconv  (from multires_deconv.m)
# ════════════════════════════════════════════════════════════════════════════

def multires_deconv(prob: dict, ret: float, sig_noise_v: np.ndarray,
                    verbose: bool = False) -> Tuple[dict, List[np.ndarray]]:
    """
    Coarse-to-fine blind deconvolution.  Equivalent to ``multires_deconv.m``.

    Parameters
    ----------
    prob        : initial problem dict (with ``y``, ``k``, ``k_sz1/2``, ...)
    ret         : pyramid rescale factor, typically 0.5**0.5.
    sig_noise_v : vector of noise std per EM iteration.

    Returns
    -------
    prob     : final problem dict (finest level)
    kListItr : list of kernel-histories per pyramid level (length = #levels)
    """
    k1 = prob['k_sz1']
    k2 = prob['k_sz2']
    # maxitr = max(floor(log(5 / min(k1, k2)) / log(ret)), 0)
    maxitr = max(int(np.floor(np.log(5.0 / min(k1, k2)) / np.log(ret))), 0)

    # retv = ret.^[0:maxitr]
    retv = np.power(ret, np.arange(0, maxitr + 1))

    # Kernel sizes per level, forced to be odd (MATLAB-exact).
    k1list = np.ceil(k1 * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0).astype(int)
    k2list = np.ceil(k2 * retv).astype(int)
    k2list = k2list + (k2list % 2 == 0).astype(int)

    # Start at the coarsest level.
    cret = float(retv[-1])
    k = resizeKer(prob['k'], cret, int(k1list[-1]), int(k2list[-1]))

    kListItr: List[np.ndarray] = [None] * (maxitr + 1)
    tprob = prob

    for itr in range(maxitr, -1, -1):
        cret = float(retv[itr])
        sy = downSmpImC(prob['y'], cret)

        tprob = dict(prob)  # shallow copy of top-level fields
        tprob['y'] = sy
        tprob['k'] = k
        tprob = set_sizes(tprob)
        tprob['filtx'] = None
        tprob['x'] = None
        tprob = filt_y(tprob)

        tprob, kList, _freeeng = deconv1(tprob, sig_noise_v, verbose=verbose)
        kListItr[itr] = kList

        if itr > 0:
            k = resizeKer(tprob['k'], 1.0 / ret,
                          int(k1list[itr - 1]), int(k2list[itr - 1]))

    return tprob, kListItr
