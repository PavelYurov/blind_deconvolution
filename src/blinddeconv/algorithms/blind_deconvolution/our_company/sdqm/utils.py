"""
Utility functions for SDQM blind deconvolution.

Haar wavelet transforms (1D/2D), FFT wrappers, and B/C operator construction
ported from ROPTLIB (C++ / Wen Huang).
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────
# Haar Wavelet Transforms  (direct port from Others/wavelet/wavelet.cpp)
# ─────────────────────────────────────────────────────────────────────

def haar_fwt_1d(v: np.ndarray) -> np.ndarray:
    """Forward 1-D Haar wavelet transform (in-place style, returns new array).
    Input: complex array of length n (must be power of 2).
    Matches haarFWT_1d in wavelet.cpp.
    """
    v = v.copy().astype(np.complex128)
    n = v.shape[0]
    r2 = np.sqrt(2.0)

    j = 1
    while j * 2 <= n:
        j *= 2

    while j > 1:
        j //= 2
        tmp = np.empty(2 * j, dtype=np.complex128)
        tmp[:j] = (v[:2*j:2] + v[1:2*j:2]) / r2
        tmp[j:2*j] = (v[:2*j:2] - v[1:2*j:2]) / r2
        v[:2*j] = tmp
    return v


def haar_fwt_1d_inverse(v: np.ndarray) -> np.ndarray:
    """Inverse 1-D Haar wavelet transform.
    Matches haarFWT_1d_inverse in wavelet.cpp.
    """
    v = v.copy().astype(np.complex128)
    n = v.shape[0]
    r2 = np.sqrt(2.0)

    j = 1
    while j * 2 <= n:
        tmp = np.empty(2 * j, dtype=np.complex128)
        tmp[0::2] = (v[:j] + v[j:2*j]) / r2
        tmp[1::2] = (v[:j] - v[j:2*j]) / r2
        v[:2*j] = tmp[:2*j]
        j *= 2
    return v


def haar_fwt_2d(vv: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Forward 2-D Haar wavelet transform (column-major layout like C).
    vv: complex array of shape (n1, n2)  (Fortran order internally in C, but
    we work with row-major numpy and keep shapes intuitive).
    Matches haarFWT_2d in wavelet.cpp.
    """
    vv = vv.copy().astype(np.complex128)
    r2 = np.sqrt(2.0)

    # Transform along rows (first dimension)
    k = 1
    while k * 2 <= n1:
        k *= 2
    while k > 1:
        k //= 2
        tmp = np.empty_like(vv[:2*k, :])
        tmp[:k, :] = (vv[0:2*k:2, :] + vv[1:2*k:2, :]) / r2
        tmp[k:2*k, :] = (vv[0:2*k:2, :] - vv[1:2*k:2, :]) / r2
        vv[:2*k, :] = tmp

    # Transform along columns (second dimension)
    k = 1
    while k * 2 <= n2:
        k *= 2
    while k > 1:
        k //= 2
        tmp = np.empty_like(vv[:, :2*k])
        tmp[:, :k] = (vv[:, 0:2*k:2] + vv[:, 1:2*k:2]) / r2
        tmp[:, k:2*k] = (vv[:, 0:2*k:2] - vv[:, 1:2*k:2]) / r2
        vv[:, :2*k] = tmp
    return vv


def haar_fwt_2d_inverse(vv: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Inverse 2-D Haar wavelet transform.
    Matches haarFWT_2d_inverse in wavelet.cpp.
    """
    vv = vv.copy().astype(np.complex128)
    r2 = np.sqrt(2.0)

    # Inverse along columns (second dimension) first
    k = 1
    while k * 2 <= n2:
        tmp = np.empty_like(vv[:, :2*k])
        tmp[:, 0::2] = (vv[:, :k] + vv[:, k:2*k]) / r2
        tmp[:, 1::2] = (vv[:, :k] - vv[:, k:2*k]) / r2
        vv[:, :2*k] = tmp
        k *= 2

    # Inverse along rows (first dimension)
    k = 1
    while k * 2 <= n1:
        tmp = np.empty_like(vv[:2*k, :])
        tmp[0::2, :] = (vv[:k, :] + vv[k:2*k, :]) / r2
        tmp[1::2, :] = (vv[:k, :] - vv[k:2*k, :]) / r2
        vv[:2*k, :] = tmp
        k *= 2
    return vv


# ─────────────────────────────────────────────────────────────────────
# Operator construction for the blind deconvolution model
# ─────────────────────────────────────────────────────────────────────

def build_B_kernel(kernel_shape, image_shape):
    """Build the B operator for the kernel subspace.

    The kernel h lives in R^{k1 x k2} and is embedded into the full image
    grid via zero-padding.  In the blind-deconvolution model the observation
    is y = diag(F B h) .* conj(F C m) where F is the 2D-DFT.

    B is stored as a sparse-like mapping: given coefficient vector w of
    length K = k1*k2, B @ w  produces a vector of length L = H*W that is
    the zero-padded kernel placed in the top-left corner (matching the
    circular convolution convention).

    Returns
    -------
    B_op : callable  w(K,) -> complex(L,)
    BH_op : callable  m(L,) -> complex(K,)   (adjoint B^H)
    K : int
    """
    kh, kw = kernel_shape
    H, W = image_shape
    L = H * W
    K = kh * kw

    def B_op(w):
        """Embed kernel coefficients into full image grid (top-left corner)."""
        k2d = w.reshape(kh, kw)
        full = np.zeros((H, W), dtype=np.complex128)
        full[:kh, :kw] = k2d
        return full.ravel()

    def BH_op(m):
        """Extract top-left kh x kw block (adjoint of zero-padding)."""
        m2d = m.reshape(H, W)
        return m2d[:kh, :kw].ravel().copy()

    return B_op, BH_op, K


def build_C_wavelet(blurred, N):
    """Build the C operator using 2-D Haar wavelet basis.

    C maps N wavelet coefficients -> image of size L = H*W.
    When N < L, only the first N wavelet coefficients are used (truncated basis).
    When N == L, C is the full inverse wavelet transform.

    The model uses  conj(F) C m, so the C operator is the inverse wavelet
    transform applied to the coefficient vector (zero-padded to L if N < L),
    matching the C++ code where C==nullptr triggers wavelet mode.

    Returns
    -------
    C_op : callable  v(N,) -> complex(L,)
    CH_op : callable  m(L,) -> complex(N,)   (adjoint C^H, i.e. forward wavelet + truncate)
    """
    H, W = blurred.shape
    L = H * W
    coeffs = haar_fwt_2d(blurred.astype(np.complex128), H, W)
    flat = coeffs.ravel()
    top_idx = np.argsort(np.abs(flat))[-N:]          # самые большие!

    def C_op(v):
        full = np.zeros(L, dtype=np.complex128)
        full[top_idx] = v
        return haar_fwt_2d_inverse(full.reshape(H, W), H, W).ravel()

    def CH_op(m):
        coeffs = haar_fwt_2d(m.reshape(H, W), H, W).ravel()
        return coeffs[top_idx].copy()

    return C_op, CH_op


def build_C_identity(image_shape):
    """Build a trivial C = Identity operator (no subspace constraint on image).

    Returns
    -------
    C_op : callable
    CH_op : callable
    """
    L = image_shape[0] * image_shape[1]

    def C_op(v):
        return v.copy()

    def CH_op(m):
        return m.copy()

    return C_op, CH_op


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p
