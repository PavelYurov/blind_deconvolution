"""
utils.py

Utility functions for the ECP (Extreme Channels Prior) blind deconvolution.

Ported from MATLAB code by Yanyang Yan, Wenqi Ren et al. (CVPR 2017).

Reference:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao, "Image Deblurring via
    Extreme Channels Prior", CVPR 2017.

The ECP method builds on the DCP framework of Pan et al. (CVPR 2016) by
adding a Bright Channel term.  Most utilities are therefore identical to
the DCP port; the only extra helper exposed here is ``bright_channel``
(the ``bright_channel.m`` file in the ECP repository).  Internally, the
ECP solver re-uses ``dark_channel`` on ``1 - S`` rather than calling the
bright-channel primitive directly, exactly as in the MATLAB code.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'valid'):
        MATLAB conv2 performs TRUE convolution (kernel flipped).
        → scipy.signal.convolve2d(A, B, mode='valid') — also true conv.

    padarray(I, [p p], 'replicate'):
        → np.pad(I, ((p,p),(p,p)), mode='edge')

    MATLAB indexing is 1-based, column-major (Fortran order).  In
    dark_channel / assign_dark_channel_to_pixel the linear index returned
    by ``min(tmp(:))`` is a COLUMN-MAJOR 1-based index into the patch.
    We preserve that convention (``flatten(order='F')`` /
    ``np.unravel_index(..., order='F')``) so the round-trip matches
    MATLAB exactly.

    graythresh(img):  Otsu on [0,1] float → manual 256-bin version.
    fspecial('gaussian', hsize, sigma):  manual Gaussian, sum = 1.
    histc(x, edges):  last bin includes right edge; output length == len(edges).
    dst / idst (Liu boundary Poisson solver):  MATLAB's dst is DST-I.
        scipy.fft.dstn / idstn with type=1 round-trip matches MATLAB.
    psf2otf / otf2psf:  zero-pad, circshift by -floor(psf/2), fft2 (and inverse).
    interp2 'linear' with out-of-bound NaNs:  scipy.ndimage.map_coordinates
        with cval=0 (MATLAB replaces NaNs by 0 in adjust_psf_center).
"""

cimport cython
cimport numpy as cnp
from libc.math cimport INFINITY

import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn


# ═════════════════════════════════════════════════════════════════════════════
# PSF ↔ OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# opt_fft_size  (from cho_code/opt_fft_size.m)
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None  # module-level cache (MATLAB `persistent`)


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """Build the LUT of optimal FFT sizes (products of primes ≤ 13)."""
    lut = np.zeros(lut_size + 1, dtype=np.int64)

    e2 = 1
    while e2 <= lut_size:
        e3 = e2
        while e3 <= lut_size:
            e5 = e3
            while e5 <= lut_size:
                e7 = e5
                while e7 <= lut_size:
                    if e7 <= lut_size:
                        lut[e7] = e7
                    if e7 * 11 <= lut_size:
                        lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= lut_size:
                        lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2

    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = i
        else:
            lut[i] = nn
    return lut


def opt_fft_size(n) -> np.ndarray:
    """
    Compute optimal FFT data length(s).  Equivalent to MATLAB opt_fft_size.m.
    """
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None:
        _OPT_FFT_LUT = _build_opt_fft_lut()

    n = np.asarray(n, dtype=np.int64)
    scalar_input = n.ndim == 0
    n = np.atleast_1d(n)

    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n)
    for i in range(n.size):
        nn = n.flat[i]
        if 1 <= nn <= lut_size:
            m.flat[i] = _OPT_FFT_LUT[nn]
        else:
            m.flat[i] = -1

    if scalar_input:
        return int(m.flat[0])
    return m


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu  (from cho_code/wrap_boundary_liu.m)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """
    Solve a Poisson equation with Dirichlet boundary via 2-D DST-I.
    Equivalent to the nested solve_min_laplacian helper in
    wrap_boundary_liu.m.

    MATLAB's dst is DST-I.  scipy.fft.dstn / idstn with type=1 round-trip
    correctly (forward scale factor cancels in forward + inverse).
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    # Keep only the boundary; zero the interior
    boundary_image[1:-1, 1:-1] = 0.0

    # Laplacian of the boundary image at interior points
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]
        + boundary_image[1:H - 1, 0:W - 2]
        + boundary_image[0:H - 2, 1:W - 1]
        + boundary_image[2:H,     1:W - 1]
    )

    f1 = -f_bp
    f2 = f1[1:H - 1, 1:W - 1]

    f2sin = dstn(f2, type=1)

    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom
    img_tt = idstn(f3, type=1)

    img_direct = boundary_image.copy()
    img_direct[1:H - 1, 1:W - 1] = img_tt
    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image so boundaries are circularly smooth for FFT deconvolution.
    Equivalent to MATLAB wrap_boundary_liu.m (Cho; based on Liu & Jia, ICIP'08).

    NOTE: the MATLAB code hard-codes alpha=1; this port does the same.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        # ── r_A: (2*alpha + H_w) × W ────────────────────────────────────
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]

        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        r_A[alpha:alpha + H_w, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

        A2 = _solve_min_laplacian(r_A)
        r_A = A2
        A = r_A

        # ── r_B: H × (2*alpha + W_w) ────────────────────────────────────
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        r_B[-1, alpha:alpha + W_w] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        # ── r_C: (2*alpha + H_w) × (2*alpha + W_w) ──────────────────────
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        # Crop (MATLAB uses alpha=1 throughout)
        A = A[:H_w, :]
        B = B[:, 1:W_w + 1]
        C = C[1:H_w + 1, 1:W_w + 1]

        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# dark_channel  (from dark_channel.m)
# ═════════════════════════════════════════════════════════════════════════════

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _dark_channel_kernel(
        const double[:, :, ::1] I_pad,
        double[:, ::1] J,
        long long[:, ::1] J_index,
        Py_ssize_t M, Py_ssize_t N, Py_ssize_t C,
        Py_ssize_t ps) noexcept nogil:
    """
    Sliding-window dark channel over padded I_pad.  Bit-exact with the
    reference Python implementation:
        J[m,n]       = min over (ps×ps) patch of (min over channels)
        J_index[m,n] = 1-based column-major first-occurrence index of
                       that minimum into the (ps,ps) patch layout —
                       matches np.argmin(tmp.flatten(order='F')) + 1.

    Iteration order (c outer, r inner) + strict `<` keep the first
    occurrence on ties, matching numpy's argmin tie-break.  min() over
    finite floats is associative under IEEE-754 so the value itself is
    independent of scan order — only argmin tie-breaking depends on it.
    """
    cdef Py_ssize_t m, n, r, c, ch, min_idx
    cdef double min_val, val, v

    for m in range(M):
        for n in range(N):
            min_val = INFINITY
            min_idx = 0
            for c in range(ps):
                for r in range(ps):
                    val = I_pad[m + r, n + c, 0]
                    for ch in range(1, C):
                        v = I_pad[m + r, n + c, ch]
                        if v < val:
                            val = v
                    if val < min_val:
                        min_val = val
                        min_idx = r + ps * c
            J[m, n] = min_val
            J_index[m, n] = min_idx + 1  # 1-based, matching MATLAB


def dark_channel(I: np.ndarray, patch_size: int):
    """
    Compute the dark channel of an image.  Equivalent to MATLAB dark_channel.m.

    Returns
    -------
    J       : (M, N) dark channel
    J_index : (M, N) int — 1-based COLUMN-MAJOR linear index into the patch
              where the minimum was found (matches MATLAB exactly so that
              assign_dark_channel_to_pixel.patch(idx) round-trips).

    NOTE: Cython-optimised.  Bit-exact with the original NumPy version.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    cdef Py_ssize_t M = I.shape[0]
    cdef Py_ssize_t N = I.shape[1]
    cdef Py_ssize_t C = I.shape[2]
    cdef Py_ssize_t ps = patch_size
    cdef Py_ssize_t p = patch_size // 2

    I_pad_np = np.ascontiguousarray(
        np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge'),
        dtype=np.float64,
    )
    J_np = np.zeros((M, N), dtype=np.float64)
    J_index_np = np.zeros((M, N), dtype=np.int64)

    cdef double[:, :, ::1] I_pad_mv = I_pad_np
    cdef double[:, ::1] J_mv = J_np
    cdef long long[:, ::1] J_index_mv = J_index_np

    with nogil:
        _dark_channel_kernel(I_pad_mv, J_mv, J_index_mv, M, N, C, ps)

    return J_np, J_index_np


# ═════════════════════════════════════════════════════════════════════════════
# bright_channel  (from bright_channel.m)
# ═════════════════════════════════════════════════════════════════════════════

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _bright_channel_kernel(
        const double[:, :, ::1] I_pad,
        double[:, ::1] J,
        long long[:, ::1] J_index,
        Py_ssize_t M, Py_ssize_t N, Py_ssize_t C,
        Py_ssize_t ps) noexcept nogil:
    """
    Sliding-window bright channel over padded I_pad.  Mirrors the
    dark-channel kernel with max (strict `>`) — keeps the first
    occurrence on ties, matching np.argmax(tmp.flatten(order='F')).
    """
    cdef Py_ssize_t m, n, r, c, ch, max_idx
    cdef double max_val, val, v
    # Use -INFINITY as the running max sentinel.  All finite inputs win.
    cdef double NEG_INF = -INFINITY

    for m in range(M):
        for n in range(N):
            max_val = NEG_INF
            max_idx = 0
            for c in range(ps):
                for r in range(ps):
                    val = I_pad[m + r, n + c, 0]
                    for ch in range(1, C):
                        v = I_pad[m + r, n + c, ch]
                        if v > val:
                            val = v
                    if val > max_val:
                        max_val = val
                        max_idx = r + ps * c
            J[m, n] = max_val
            J_index[m, n] = max_idx + 1


def bright_channel(I: np.ndarray, patch_size: int):
    """
    Compute the bright channel of an image.  Equivalent to MATLAB
    bright_channel.m — max over channel AND patch.

    NOTE: the ECP solver does NOT actually call this function; it instead
    computes the bright channel via ``dark_channel(1 - S)``, exactly
    mirroring L0Deblur_dark_chanelBD.m.  This primitive is exposed here
    purely to preserve the one-to-one mapping with the MATLAB repository.

    Cython-optimised.  Bit-exact with the original NumPy version.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    cdef Py_ssize_t M = I.shape[0]
    cdef Py_ssize_t N = I.shape[1]
    cdef Py_ssize_t C = I.shape[2]
    cdef Py_ssize_t ps = patch_size
    cdef Py_ssize_t p = patch_size // 2

    I_pad_np = np.ascontiguousarray(
        np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge'),
        dtype=np.float64,
    )
    J_np = np.zeros((M, N), dtype=np.float64)
    J_index_np = np.zeros((M, N), dtype=np.int64)

    cdef double[:, :, ::1] I_pad_mv = I_pad_np
    cdef double[:, ::1] J_mv = J_np
    cdef long long[:, ::1] J_index_mv = J_index_np

    with nogil:
        _bright_channel_kernel(I_pad_mv, J_mv, J_index_mv, M, N, C, ps)

    return J_np, J_index_np


# ═════════════════════════════════════════════════════════════════════════════
# assign_dark_channel_to_pixel  (from assign_dark_channel_to_pixel.m)
# ═════════════════════════════════════════════════════════════════════════════

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void _assign_dcp_kernel(
        double[:, :, ::1] S_padd,
        const double[:, ::1] refine,
        const long long[:, ::1] J_idx,
        Py_ssize_t M, Py_ssize_t N, Py_ssize_t C,
        Py_ssize_t ps) noexcept nogil:
    """
    In-place equivalent of the reference Python loop:

        patch = S_padd[m:m+ps, n:n+ps, :].copy()
        if np.min(patch) != refine[m, n]:
            idx = J_idx[m, n] - 1
            coords = np.unravel_index(idx, (ps, ps, C), order='F')
            patch[coords] = refine[m, n]
        S_padd[m:m+ps, n:n+ps, :] = patch

    Since the copy → write-back is a no-op when nothing is changed and
    modifies exactly one pixel otherwise, this reduces to:

        if min(S_padd[m:m+ps, n:n+ps, :]) != refine[m,n]:
            r, c, ch = unravel_F(J_idx[m,n]-1, (ps, ps, C))
            S_padd[m+r, n+c, ch] = refine[m,n]

    The sequential dependency (writes to S_padd are visible to later
    m,n) is preserved exactly by iterating m outer, n inner in order.
    """
    cdef Py_ssize_t m, n, r, c, ch, idx, rr, cc, cch
    cdef double min_val, val, ref
    cdef Py_ssize_t ps2 = ps * ps

    for m in range(M):
        for n in range(N):
            ref = refine[m, n]
            min_val = INFINITY
            for r in range(ps):
                for c in range(ps):
                    for ch in range(C):
                        val = S_padd[m + r, n + c, ch]
                        if val < min_val:
                            min_val = val

            if min_val != ref:
                idx = J_idx[m, n] - 1  # to 0-based column-major
                # Unravel into (ps, ps, C) Fortran order:
                #   flat = rr + ps*cc + ps*ps*cch
                cch = idx // ps2
                idx = idx - cch * ps2
                cc = idx // ps
                rr = idx - cc * ps
                S_padd[m + rr, n + cc, cch] = ref


def assign_dark_channel_to_pixel(S: np.ndarray,
                                 dark_channel_refine: np.ndarray,
                                 dark_channel_index: np.ndarray,
                                 patch_size: int) -> np.ndarray:
    """
    Assign refined dark-channel values back to pixel positions.
    Equivalent to MATLAB assign_dark_channel_to_pixel.m.

    ``dark_channel_index`` must be 1-based column-major, as returned by
    ``dark_channel`` / ``bright_channel``.

    The same function is used by the bright-channel branch in the ECP
    solver (called on ``1 - S``), matching MATLAB's behaviour: the update
    rule only depends on the stored indices, not on min-vs-max semantics.

    NOTE: Cython-optimised.  Bit-exact with the original Python loop.
    """
    if S.ndim == 2:
        S_3d = S[:, :, np.newaxis]
        was_2d = True
    else:
        S_3d = S
        was_2d = False

    cdef Py_ssize_t M = S_3d.shape[0]
    cdef Py_ssize_t N = S_3d.shape[1]
    cdef Py_ssize_t C = S_3d.shape[2]
    cdef Py_ssize_t ps = patch_size
    cdef Py_ssize_t padsize = patch_size // 2

    S_padd_np = np.ascontiguousarray(
        np.pad(S_3d, ((padsize, padsize), (padsize, padsize), (0, 0)),
               mode='edge'),
        dtype=np.float64,
    )
    refine_np = np.ascontiguousarray(dark_channel_refine, dtype=np.float64)
    idx_np = np.ascontiguousarray(dark_channel_index, dtype=np.int64)

    cdef double[:, :, ::1] S_padd_mv = S_padd_np
    cdef double[:, ::1] refine_mv = refine_np
    cdef long long[:, ::1] idx_mv = idx_np

    with nogil:
        _assign_dcp_kernel(S_padd_mv, refine_mv, idx_mv, M, N, C, ps)

    outImg = S_padd_np[padsize:padsize + M, padsize:padsize + N, :].copy()

    # Boundary processing: restore the original border values (MATLAB
    # behaviour). copy() above ensures outImg is owned, not a view.
    S_3d_arr = np.asarray(S_3d, dtype=np.float64)
    outImg[:padsize, :, :] = S_3d_arr[:padsize, :, :]
    outImg[-padsize:, :, :] = S_3d_arr[-padsize:, :, :]
    outImg[:, :padsize, :] = S_3d_arr[:, :padsize, :]
    outImg[:, -padsize:, :] = S_3d_arr[:, -padsize:, :]

    if was_2d:
        return outImg[:, :, 0]
    return outImg


# ═════════════════════════════════════════════════════════════════════════════
# conjgrad  (from cho_code/conjgrad.m)
# ═════════════════════════════════════════════════════════════════════════════

def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Conjugate gradient solver.  Equivalent to cho_code/conjgrad.m.
    Solves A·x = b where A is supplied implicitly by ``ax_func``.
    """
    x = x.copy()
    r = b - ax_func(x, func_param)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(max_it):
        Ap = ax_func(p, func_param)
        pAp = np.sum(p * Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# ═════════════════════════════════════════════════════════════════════════════
# adjust_psf_center  (from cho_code/adjust_psf_center.m)
# ═════════════════════════════════════════════════════════════════════════════

def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    """
    Centre the PSF by shifting its centre-of-mass to the geometric centre.
    Equivalent to MATLAB adjust_psf_center.m.
    """
    rows, cols = psf.shape

    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                       np.arange(1, rows + 1, dtype=np.float64))

    if np.sum(psf) == 0:
        return psf

    xc1 = np.sum(psf * X)
    yc1 = np.sum(psf * Y)
    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0

    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)

    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                     np.arange(cols, dtype=np.float64),
                                     indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    result = map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                             order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)


# ═════════════════════════════════════════════════════════════════════════════
# threshold_pxpy_v1  (from cho_code/threshold_pxpy_v1.m)
# ═════════════════════════════════════════════════════════════════════════════

def _histc(data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Equivalent to MATLAB histc(data, edges):
    bin k counts values where edges[k] <= x < edges[k+1]; the last bin
    also includes x == edges[-1]; output length == len(edges).
    """
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size, threshold=None):
    """
    Gradient thresholding for kernel estimation.
    Equivalent to MATLAB cho_code/threshold_pxpy_v1.m.
    """
    b_estimate_threshold = threshold is None
    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    # MATLAB conv2(..., 'valid') is true convolution → scipy.convolve2d matches.
    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        # MATLAB uses atan(py./px) — we mirror that (NOT arctan2).
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = np.max(psf_size) if hasattr(psf_size, '__len__') else psf_size
        th = max(psf_size_val * 20, 10)

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                # MATLAB: threshold = pm_steps(end - t + 1)  (t is 1-based there)
                threshold = pm_steps[len(pm_steps) - 1 - t]
                break

    m = pm < threshold
    while np.all(m):
        threshold = threshold * 0.81
        m = pm < threshold

    px[m] = 0.0
    py[m] = 0.0

    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold


# ═════════════════════════════════════════════════════════════════════════════
# bilateral_filter  (from bilateral_filter.m)
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Equivalent to MATLAB fspecial('gaussian', size, sigma).
    size×size Gaussian, sum normalised to 1.
    """
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """
    Bilateral filter (grayscale / multi-channel non-RGB path of MATLAB's
    bilateral_filter.m).  Called as bilateral_filter(diff, 3, 0.1) inside
    ringing_artifacts_removal.m.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1

    h, w, d = img.shape
    img = img.astype(np.float32)

    # Non-RGB branch of the MATLAB code
    lab = img.copy()
    sigma = sigma * np.sqrt(d)

    fr = int(np.ceil(sigma_s * 3))

    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    p_lab = np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')

    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)

    spatial_weight = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * sigma

    for y in range(-fr, fr + 1):
        for x in range(-fr, fr + 1):
            w_s = spatial_weight[y + fr, x + fr]

            n_img = p_img[fr + y:fr + y + h, fr + x:fr + x + w, :]
            n_lab = p_lab[fr + y:fr + y + h, fr + x:fr + x + w, :]

            f_diff = lab - n_lab
            f_dist = np.sum(f_diff ** 2, axis=2)

            w_f = np.exp(-0.5 * f_dist / ss)
            w_t = w_s * w_f

            r_img += n_img * w_t[:, :, np.newaxis]
            w_sum += w_t

    r_img = r_img / w_sum[:, :, np.newaxis]

    if was_2d:
        return r_img[:, :, 0]
    return r_img


# ═════════════════════════════════════════════════════════════════════════════
# graythresh  (Otsu's method, matching MATLAB)
# ═════════════════════════════════════════════════════════════════════════════

def graythresh(img: np.ndarray) -> float:
    """
    Otsu's threshold.  Equivalent to MATLAB graythresh(img) on [0,1] float.
    Uses a 256-bin histogram over [0,1] and returns a threshold in [0,1].
    """
    img_flat = img.ravel().astype(np.float64)
    img_flat = np.clip(img_flat, 0.0, 1.0)

    num_bins = 256
    counts, bin_edges = np.histogram(img_flat, bins=num_bins, range=(0.0, 1.0))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts.astype(np.float64) / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centres)
    mu_t = mu[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))

    sigma_b_sq = np.nan_to_num(sigma_b_sq, nan=0.0)
    max_idx = np.argmax(sigma_b_sq)
    return bin_centres[max_idx]


# ═════════════════════════════════════════════════════════════════════════════
# fftconv  (from fftconv.m) — lightweight FFT-based convolution helper
# ═════════════════════════════════════════════════════════════════════════════

def fftconv(I: np.ndarray, filt: np.ndarray, b_otf: bool = False) -> np.ndarray:
    """
    FFT-based convolution.  Equivalent to MATLAB fftconv.m.
    If ``b_otf`` is True, ``filt`` is already an OTF of the same shape as I.
    """
    if I.ndim == 3 and I.shape[2] == 3:
        H, W, _ = I.shape
        otf = psf2otf(filt, (H, W))
        out = np.zeros_like(I, dtype=np.float64)
        for c in range(3):
            out[:, :, c] = np.real(np.fft.ifft2(
                np.fft.fft2(I[:, :, c]) * otf))
        return out

    if b_otf:
        return np.real(np.fft.ifft2(np.fft.fft2(I) * filt))
    return np.real(np.fft.ifft2(np.fft.fft2(I) * psf2otf(filt, I.shape)))
