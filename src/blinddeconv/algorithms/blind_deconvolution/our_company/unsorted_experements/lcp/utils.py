"""
utils.py

Utility functions for the LCP (Log-Concave Prior) Bayesian image restoration
under Poisson noise, ported from MATLAB code.

Reference:
    M. Vono, N. Dobigeon, P. Chainais, "Bayesian image restoration under
    Poisson noise and log-concave prior", ICASSP 2019, Brighton, UK.

Ported MATLAB files:
    HXconv.m                — FFT-based convolution
    chambolle_prox_TV_stop.m — Proximal TV operator (Chambolle projection)
    daubcqf.m               — Daubechies conjugate quadrature filters
    mrdwt_TI2D.m            — Forward redundant DWT (TI, 2D wrapper)
    mirdwt_TI2D.m           — Inverse redundant DWT (TI, 2D wrapper)

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    fft2 / ifft2:
        Both MATLAB and NumPy use the same DFT convention.  However,
        MATLAB's fftshift swaps quadrants; here we replicate MATLAB's
        zero-padding + fftshift for the PSF exactly as in HXconv.m.

    padarray(B, [p1 p2], 'pre' / 'post'):
        MATLAB padarray with 'pre' pads BEFORE the array,
        'post' pads AFTER the array.  We match this exactly with
        np.pad using ((p_before, 0), …) and ((0, p_after), …).

    fftshift:
        MATLAB fftshift swaps quadrants of a 2-D array.
        np.fft.fftshift does the same.

    MATLAB floor / round:
        MATLAB floor rounds toward −∞ (same as Python int(math.floor(…))).
        MATLAB round rounds to nearest, ties to even — but the values
        here are always half-integers so we use Python's built-in round()
        to match (which also rounds half-to-even).

    fspecial('gaussian', hsize, sigma):
        Produces an hsize×hsize Gaussian kernel, normalised to sum = 1.
        → We build it manually to match MATLAB exactly.

    MATLAB conv / poly / roots:
        In daubcqf we replicate the polynomial arithmetic with numpy.

    mrdwt / mirdwt (Rice Wavelet Toolbox MEX):
        The MEX functions implement the *redundant* (undecimated,
        stationary, translation-invariant) DWT à trous.
        → We reimplement using PyWavelets' swt2 / iswt2 for the core
          transform and then apply the same scaling as the MATLAB
          wrappers mrdwt_TI2D / mirdwt_TI2D.

    Chambolle TV:
        The sub-functions DivergenceIm and GradientIm use MATLAB
        indexing with explicit boundary handling.  We replicate this
        exactly with NumPy slicing.
"""

import numpy as np
from typing import Tuple, Optional
import pywt


# ═════════════════════════════════════════════════════════════════════════════
# HXconv  (from HXconv.m — Author: Ningning Zhao, University of Toulouse)
# ═════════════════════════════════════════════════════════════════════════════

def HXconv(x: np.ndarray,
           B: np.ndarray,
           conv: Optional[str] = None
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    FFT-based convolution of image *x* with PSF *B*.

    Replicates MATLAB HXconv.m exactly:
        1. Zero-pad B to the size of x (pre + post, matching MATLAB
           padarray behaviour with floor/round).
        2. fftshift the padded kernel.
        3. Compute BF = fft2(Bpad), BCF = conj(BF), B2F = |BF|^2.
        4. If *conv* is given, also compute the convolution product.

    Parameters
    ----------
    x    : (m, n) image (real).
    B    : (m0, n0) PSF in spatial domain.
    conv : None, 'Hx', 'HTx', or 'HTHx'.

    Returns
    -------
    BF   : fft2 of centred, zero-padded PSF.
    BCF  : conj(BF).
    B2F  : |BF|^2.
    y    : convolution result (only when *conv* is not None).

    When *conv* is None the function returns (BF, BCF, B2F, None).
    """
    m, n = x.shape
    m0, n0 = B.shape

    # ── Replicate MATLAB padarray logic ──────────────────────────────────
    # MATLAB:
    #   Bpad = padarray(B, floor([m-m0+1, n-n0+1]/2), 'pre');
    #   Bpad = padarray(Bpad, round([m-m0-1, n-n0-1]/2), 'post');
    #
    # floor([a b]/2) in MATLAB operates element-wise.
    # round([a b]/2) in MATLAB rounds half-to-even (same as Python round()).
    pre_row = int(np.floor((m - m0 + 1) / 2))
    pre_col = int(np.floor((n - n0 + 1) / 2))
    post_row = int(np.round((m - m0 - 1) / 2))
    post_col = int(np.round((n - n0 - 1) / 2))

    Bpad = np.pad(B, ((pre_row, post_row), (pre_col, post_col)),
                  mode='constant', constant_values=0)

    # MATLAB: Bpad = fftshift(Bpad);
    Bpad = np.fft.fftshift(Bpad)

    BF = np.fft.fft2(Bpad)
    BCF = np.conj(BF)
    B2F = np.abs(BF) ** 2

    if conv is None:
        return BF, BCF, B2F, None

    if conv == 'Hx':
        y = np.real(np.fft.ifft2(BF * np.fft.fft2(x)))
    elif conv == 'HTx':
        y = np.real(np.fft.ifft2(BCF * np.fft.fft2(x)))
    elif conv == 'HTHx':
        y = np.real(np.fft.ifft2(B2F * np.fft.fft2(x)))
    else:
        raise ValueError(f"Unknown conv mode: {conv!r}")

    return BF, BCF, B2F, y


# ═════════════════════════════════════════════════════════════════════════════
# fspecial_gaussian  (matching MATLAB fspecial('gaussian', hsize, sigma))
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_gaussian(hsize: int, sigma: float) -> np.ndarray:
    """
    Create a Gaussian kernel identical to MATLAB's
    ``fspecial('gaussian', hsize, sigma)``.

    The kernel is *hsize* × *hsize*, centred, and normalised to sum = 1.
    """
    half = (hsize - 1) / 2.0
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    h = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    h = h / h.sum()
    return h


# ═════════════════════════════════════════════════════════════════════════════
# Chambolle proximal TV  (from chambolle_prox_TV_stop.m)
# ═════════════════════════════════════════════════════════════════════════════

def _gradient_im(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete image gradient matching MATLAB GradientIm exactly.

    MATLAB:
        z = u(2:end, :) - u(1:end-1,:);
        dux = [z;  zeros(1,size(z,2))];

        z = u(:,2:end) - u(:,1:end-1);
        duy = [z zeros(size(z,1),1)];
    """
    # dux — vertical (row) differences, zero-padded at bottom
    z_row = u[1:, :] - u[:-1, :]
    dux = np.vstack([z_row, np.zeros((1, z_row.shape[1]))])

    # duy — horizontal (column) differences, zero-padded at right
    z_col = u[:, 1:] - u[:, :-1]
    duy = np.hstack([z_col, np.zeros((z_col.shape[0], 1))])

    return dux, duy


def _divergence_im(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Discrete divergence matching MATLAB DivergenceIm exactly.

    MATLAB:
        z = p2(:,2:end-1) - p2(:,1:end-2);
        v = [p2(:,1) z -p2(:,end)];

        z = p1(2:end-1, :) - p1(1:end-2,:);
        u = [p1(1,:); z;  -p1(end,:)];

        divp = v + u;
    """
    # Horizontal part (from p2)
    z_h = p2[:, 1:-1] - p2[:, :-2]
    v = np.hstack([p2[:, 0:1], z_h, -p2[:, -1:]])

    # Vertical part (from p1)
    z_v = p1[1:-1, :] - p1[:-2, :]
    u = np.vstack([p1[0:1, :], z_v, -p1[-1:, :]])

    return v + u


def chambolle_prox_TV_stop(g: np.ndarray,
                           lam: float = 1.0,
                           maxiter: int = 10,
                           tol: float = 1e-3,
                           tau: float = 0.249
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proximal operator for the Total Variation regulariser via
    Chambolle's projection algorithm.

    Solves:
        argmin_x  (1/2)||g - x||^2 + lambda * TV(x)

    Exact port of chambolle_prox_TV_stop.m.

    Parameters
    ----------
    g       : (ny, nx) noisy image.
    lam     : regularisation parameter (called 'lambda' in MATLAB).
    maxiter : maximum number of iterations.
    tol     : tolerance for stopping criterion.
    tau     : algorithm step-size parameter (default 0.249).

    Returns
    -------
    f  : (ny, nx) denoised image.
    px : (ny, nx) dual variable (row direction).
    py : (ny, nx) dual variable (column direction).
    """
    px = np.zeros_like(g)
    py = np.zeros_like(g)
    cont = True
    k = 0

    while cont:
        k += 1
        # Divergence of (px, py)
        divp = _divergence_im(px, py)
        u = divp - g / lam
        # Gradient of u
        upx, upy = _gradient_im(u)
        tmp = np.sqrt(upx ** 2 + upy ** 2)
        # Error (matching MATLAB sum(…)^0.5 — i.e. L2 norm of the vector)
        err = np.sqrt(np.sum((-upx + tmp * px) ** 2 + (-upy + tmp * py) ** 2))
        # Update dual variables
        px = (px + tau * upx) / (1.0 + tau * tmp)
        py = (py + tau * upy) / (1.0 + tau * tmp)
        cont = (k < maxiter) and (err > tol)

    f = g - lam * _divergence_im(px, py)
    return f, px, py


# ═════════════════════════════════════════════════════════════════════════════
# daubcqf  (from daubcqf.m — Rice Wavelet Toolbox)
# ═════════════════════════════════════════════════════════════════════════════

def daubcqf(N: int, filter_type: str = 'min') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Daubechies' scaling and wavelet filters (normalised to sqrt(2)).

    Exact port of daubcqf.m from the Rice Wavelet Toolbox.

    Parameters
    ----------
    N           : Filter length (must be even).
    filter_type : 'min' (minimum phase, default), 'max', or 'mid'.

    Returns
    -------
    h_0 : (N,) scaling filter.
    h_1 : (N,) wavelet filter.
    """
    if N % 2 != 0:
        raise ValueError("No Daubechies filter exists for ODD length")

    K = N // 2
    a = 1.0
    # p and q start as 1-element polynomial coefficient arrays
    p = np.array([1.0])
    q = np.array([1.0])
    h_0 = np.array([1.0, 1.0])

    for j in range(1, K):
        a = -a * 0.25 * (j + K - 1) / j
        # h_0 = [0 h_0] + [h_0 0]  (polynomial addition with shift)
        h_0 = np.concatenate([[0.0], h_0]) + np.concatenate([h_0, [0.0]])
        # p = [0 -p] + [p 0]  (twice)
        p = np.concatenate([[0.0], -p]) + np.concatenate([p, [0.0]])
        p = np.concatenate([[0.0], -p]) + np.concatenate([p, [0.0]])
        # q = [0 q 0] + a*p
        q = np.concatenate([[0.0], q, [0.0]]) + a * p

    q_roots = np.sort(np.roots(q))
    qt = q_roots[:K - 1]

    if filter_type == 'mid':
        if K % 2 == 1:
            idx = np.concatenate([np.arange(0, N - 2, 4), np.arange(1, N - 2, 4)])
            qt = q_roots[idx]
        else:
            idx = np.concatenate([
                [0],
                np.arange(3, K - 1, 4),
                np.arange(4, K - 1, 4),
                np.arange(N - 4, K - 1, -4),
                np.arange(N - 5, K - 1, -4),
            ])
            qt = q_roots[idx]

    h_0 = np.convolve(h_0, np.real(np.poly(qt)))
    h_0 = np.sqrt(2) * h_0 / np.sum(h_0)

    if filter_type == 'max':
        h_0 = h_0[::-1]

    if abs(np.sum(h_0 ** 2) - 1.0) > 1e-4:
        raise ValueError("Numerically unstable for this value of N.")

    # Wavelet filter: h_1 = rot90(h_0,2) then negate odd-indexed entries
    # MATLAB: h_1 = rot90(h_0,2); h_1(1:2:N) = -h_1(1:2:N);
    # rot90(row_vector, 2) reverses the vector.
    h_1 = h_0[::-1].copy()
    h_1[0::2] = -h_1[0::2]

    return h_0, h_1


# ═════════════════════════════════════════════════════════════════════════════
# Redundant (Stationary / Translation-Invariant) DWT — 2D
#
# The original MATLAB code uses MEX functions mrdwt / mirdwt from the
# Rice Wavelet Toolbox.  These implement the *à trous* (undecimated,
# redundant, stationary) DWT.
#
# Convention (mrdwt for an N×N image with L levels):
#   [yl, yh] = mrdwt(x, h, L)
#   yl : N×N   — low-pass approximation coefficients at level L
#   yh : N×(3·N·L) — detail coefficients stored as
#        [LH1 HL1 HH1 | LH2 HL2 HH2 | … | LH_L HL_L HH_L]
#        each sub-band is N×N, concatenated horizontally.
#
# mirdwt is the inverse: x = mirdwt(yl, yh, h, L).
#
# The wrappers mrdwt_TI2D / mirdwt_TI2D apply per-level scaling
# so that the resulting operators form a transpose pair (needed for
# the synthesis formulation).
#
# We replicate this with PyWavelets' swt2 (stationary wavelet
# transform, 2D) and iswt2.
# ═════════════════════════════════════════════════════════════════════════════

def _pywt_wavelet_from_h(h: np.ndarray) -> pywt.Wavelet:
    """
    Create a PyWavelets Wavelet object from the scaling filter *h*
    produced by daubcqf (normalised to sqrt(2)).

    daubcqf normalises filters to sqrt(2), but PyWavelets expects
    filters normalised to sum = sqrt(2) for orthogonal wavelets
    (the standard convention).  daubcqf already does this, so we
    can use the filter directly.
    """
    # daubcqf returns h_0 (dec_lo) and h_1 (dec_hi).
    # For a PyWavelets Wavelet we need: dec_lo, dec_hi, rec_lo, rec_hi.
    # Orthogonal QMF: rec_lo = dec_lo[::-1], rec_hi = dec_hi[::-1].
    h_0 = h
    h_1 = h_0[::-1].copy()
    h_1[0::2] = -h_1[0::2]

    rec_lo = h_0[::-1]
    rec_hi = h_1[::-1]

    w = pywt.Wavelet('custom_haar', filter_bank=[
        h_0.tolist(), h_1.tolist(),
        rec_lo.tolist(), rec_hi.tolist()
    ])
    return w


def mrdwt(x: np.ndarray, h: np.ndarray, L: int
          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Redundant (stationary / undecimated) 2-D DWT, matching the Rice
    Wavelet Toolbox ``mrdwt(x, h, L)`` MEX function.

    Parameters
    ----------
    x : (N, N) input image.
    h : 1-D scaling filter (from daubcqf, normalised to sqrt(2)).
    L : number of decomposition levels.

    Returns
    -------
    yl : (N, N) low-pass approximation at level L.
    yh : (N, 3*N*L) detail coefficients, stored as
         [LH1 HL1 HH1 | LH2 HL2 HH2 | … | LH_L HL_L HH_L]
         (each sub-band is N×N, concatenated along columns).
    """
    w = _pywt_wavelet_from_h(h)
    N = x.shape[1]

    # PyWavelets swt2 returns a list from coarsest to finest:
    #   [(cA_L, (cH_L, cV_L, cD_L)), ..., (cA_1, (cH_1, cV_1, cD_1))]
    # Note: swt2 level ordering is REVERSED relative to what we need.
    coeffs = pywt.swt2(x, w, level=L, trim_approx=True)
    # coeffs with trim_approx=True:
    #   [cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]
    # That is: coeffs[0] = cA_L, coeffs[l] = (cH_{L-l+1}, cV_{L-l+1}, cD_{L-l+1})

    yl = coeffs[0]  # approximation at level L

    yh = np.zeros((x.shape[0], 3 * N * L), dtype=np.float64)
    for ll in range(1, L + 1):
        # coeffs[ll] corresponds to level (L - ll + 1) in the MATLAB ordering
        # We need level ll in MATLAB ordering → index = L - ll + 1 in coeffs list
        detail = coeffs[L - ll + 1]  # (cH, cV, cD) at MATLAB level ll
        cH, cV, cD = detail
        # MATLAB stores: [LH HL HH] = [cV, cH, cD] per level
        # Rice convention: yh(:, (l-1)*3N+1 : l*3N) = [LH_l, HL_l, HH_l]
        # where LH = horizontal details (cV in PyWavelets convention — vertical
        # pass low, horizontal pass high → these are "horizontal" edges),
        # HL = vertical details (cH), HH = diagonal (cD).
        #
        # Actually, Rice mrdwt stores sub-bands in the order:
        #   LH (low rows, high cols) = cH (in PyWavelets: detail horizontal)
        #   HL (high rows, low cols) = cV (in PyWavelets: detail vertical)
        #   HH (high rows, high cols) = cD (in PyWavelets: detail diagonal)
        #
        # PyWavelets swt2 detail order: (cH, cV, cD) where
        #   cH = "horizontal detail" (low along columns, high along rows)
        #   cV = "vertical detail" (high along columns, low along rows)
        #   cD = "diagonal detail"
        #
        # Rice mrdwt yh stores: [LH, HL, HH] per level where
        #   LH = passed through low-pass vertically, high-pass horizontally
        #   HL = passed through high-pass vertically, low-pass horizontally
        #   HH = passed through high-pass both ways
        #
        # So: LH(Rice) = cH(PyWavelets), HL(Rice) = cV(PyWavelets), HH = cD
        start = (ll - 1) * 3 * N
        yh[:, start:start + N] = cH
        yh[:, start + N:start + 2 * N] = cV
        yh[:, start + 2 * N:start + 3 * N] = cD

    return yl, yh


def mirdwt(yl: np.ndarray, yh: np.ndarray, h: np.ndarray, L: int
           ) -> np.ndarray:
    """
    Inverse redundant (stationary / undecimated) 2-D DWT, matching the
    Rice Wavelet Toolbox ``mirdwt(yl, yh, h, L)`` MEX function.

    Parameters
    ----------
    yl : (N, N) low-pass approximation at level L.
    yh : (N, 3*N*L) detail coefficients (same layout as mrdwt output).
    h  : 1-D scaling filter.
    L  : number of decomposition levels.

    Returns
    -------
    x  : (N, N) reconstructed image.
    """
    w = _pywt_wavelet_from_h(h)
    N = yl.shape[1]

    # Build coefficients list for iswt2.
    # PyWavelets iswt2 expects: [cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]
    coeffs = [yl]
    for ll_rev in range(L, 0, -1):
        # ll_rev goes L, L-1, ..., 1 (MATLAB level ordering)
        start = (ll_rev - 1) * 3 * N
        cH = yh[:, start:start + N]
        cV = yh[:, start + N:start + 2 * N]
        cD = yh[:, start + 2 * N:start + 3 * N]
        coeffs.append((cH, cV, cD))

    x = pywt.iswt2(coeffs, w)
    return x


# ═════════════════════════════════════════════════════════════════════════════
# mrdwt_TI2D  (from mrdwt_TI2D.m — wrapper by Mario Figueiredo)
# ═════════════════════════════════════════════════════════════════════════════

def mrdwt_TI2D(v: np.ndarray, h: np.ndarray, levels: int) -> np.ndarray:
    """
    Forward Translation-Invariant redundant DWT (2D) with per-level
    scaling, matching MATLAB mrdwt_TI2D.m.

    This is a wrapper around mrdwt that rescales coefficients so that
    the forward and inverse transforms form a transpose pair
    (W^T and W, respectively).

    MATLAB code:
        scalefactor = 2;
        [temp1, temp2] = mrdwt(v, h, levels-1);
        temp1 = temp1 * scalefactor^(-(levels-1));
        for ll = 1:levels-1
            temp2(:, (ll-1)*n*3+1 : ll*n*3) *= scalefactor^(-ll);
        end
        z = [temp1 temp2];

    Parameters
    ----------
    v      : (m, n) input image.
    h      : 1-D scaling filter.
    levels : number of decomposition levels.

    Returns
    -------
    z : (m, n + 3*n*(levels-1)) = (m, n*(1 + 3*(levels-1)))
        Concatenation [scaled_approx, scaled_details].
    """
    scalefactor = 2
    m, n = v.shape
    temp1, temp2 = mrdwt(v, h, levels - 1)

    # Scale approximation coefficients
    temp1 = temp1 * (scalefactor ** (-(levels - 1)))

    # Scale detail coefficients per level
    for ll in range(1, levels):
        start = (ll - 1) * n * 3
        end_ = ll * n * 3
        temp2[:, start:end_] = temp2[:, start:end_] * (scalefactor ** (-ll))

    z = np.hstack([temp1, temp2])
    return z


def mirdwt_TI2D(v: np.ndarray, h: np.ndarray, levels: int) -> np.ndarray:
    """
    Inverse Translation-Invariant redundant DWT (2D) with per-level
    scaling, matching MATLAB mirdwt_TI2D.m.

    MATLAB code:
        scalefactor = 2;
        n = min(n1, n2);
        t1 = v(:, 1:n) * scalefactor^(levels-1);
        for ll = 1:levels-1
            t2(:, (ll-1)*n*3+1 : ll*n*3) =
                v(:, n+(ll-1)*n*3+1 : n+ll*n*3) * scalefactor^(ll);
        end
        z = mirdwt(t1, t2, h, levels-1);

    Parameters
    ----------
    v      : (m, k) where k = n*(1 + 3*(levels-1)).
             First n columns are approximation, rest are details.
    h      : 1-D scaling filter.
    levels : number of decomposition levels.

    Returns
    -------
    z : (m, n) reconstructed image.
    """
    scalefactor = 2
    n1, n2 = v.shape
    n = min(n1, n2)

    # Approximation: first n columns, scaled up
    t1 = v[:, :n] * (scalefactor ** (levels - 1))

    # Details: remaining columns, scaled up per level
    t2 = np.zeros((n1, 3 * n * (levels - 1)), dtype=np.float64)
    for ll in range(1, levels):
        src_start = n + (ll - 1) * n * 3
        src_end = n + ll * n * 3
        dst_start = (ll - 1) * n * 3
        dst_end = ll * n * 3
        t2[:, dst_start:dst_end] = v[:, src_start:src_end] * (scalefactor ** ll)

    z = mirdwt(t1, t2, h, levels - 1)
    return z
