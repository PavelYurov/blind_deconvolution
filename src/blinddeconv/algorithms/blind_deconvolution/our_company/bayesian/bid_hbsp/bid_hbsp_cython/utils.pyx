# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
utils_cy.pyx — Cython-accelerated utility functions for BID-HBSP.

Provides fast typed-memoryview implementations of:
    - forward_diff_x / forward_diff_y  (gradient operators)
    - adjoint_diff_x / adjoint_diff_y  (adjoint gradient operators)
    - compute_hs_weights               (image-space HS weights)
    - compute_hs_weights_scalar        (filter-space HS weights)
    - project_kernel / threshold_kernel
    - psf2otf, otf2psf, fft_convolve   (thin wrappers — FFT stays in NumPy)
    - precompute_gradient_operators
    - init_gaussian_kernel
    - edgetaper

All pixel-level loops use typed memoryviews and release the GIL.
FFT calls delegate to numpy.fft (FFTPACK/MKL) — Cython can't beat
LAPACK for large transforms, so we optimize the *surrounding* code.
"""

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, tanh, exp, fabs, cos, floor, round as c_round

np.import_array()

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

DEF EPSILON = 1e-12
DEF M_PI = 3.14159265358979323846


# ═══════════════════════════════════════════════════════════════
#  Gradient operators — tight C loops
# ═══════════════════════════════════════════════════════════════

def forward_diff_x(double[:, :] u not None):
    """Horizontal forward difference: out[i,j] = u[i, j+1] - u[i, j]."""
    cdef Py_ssize_t H = u.shape[0]
    cdef Py_ssize_t W = u.shape[1]
    cdef Py_ssize_t i, j
    cdef double[:, :] out = np.empty((H, W), dtype=np.float64)

    with nogil:
        for i in range(H):
            for j in range(W - 1):
                out[i, j] = u[i, j + 1] - u[i, j]
            out[i, W - 1] = u[i, 0] - u[i, W - 1]  # periodic
    return np.asarray(out)


def forward_diff_y(double[:, :] u not None):
    """Vertical forward difference: out[i,j] = u[i+1, j] - u[i, j]."""
    cdef Py_ssize_t H = u.shape[0]
    cdef Py_ssize_t W = u.shape[1]
    cdef Py_ssize_t i, j
    cdef double[:, :] out = np.empty((H, W), dtype=np.float64)

    with nogil:
        for i in range(H - 1):
            for j in range(W):
                out[i, j] = u[i + 1, j] - u[i, j]
        for j in range(W):
            out[H - 1, j] = u[0, j] - u[H - 1, j]  # periodic
    return np.asarray(out)


def adjoint_diff_x(double[:, :] v not None):
    """Adjoint of horizontal forward difference: out[i,j] = v[i, j-1] - v[i, j]."""
    cdef Py_ssize_t H = v.shape[0]
    cdef Py_ssize_t W = v.shape[1]
    cdef Py_ssize_t i, j
    cdef double[:, :] out = np.empty((H, W), dtype=np.float64)

    with nogil:
        for i in range(H):
            out[i, 0] = v[i, W - 1] - v[i, 0]  # periodic
            for j in range(1, W):
                out[i, j] = v[i, j - 1] - v[i, j]
    return np.asarray(out)


def adjoint_diff_y(double[:, :] v not None):
    """Adjoint of vertical forward difference: out[i,j] = v[i-1, j] - v[i, j]."""
    cdef Py_ssize_t H = v.shape[0]
    cdef Py_ssize_t W = v.shape[1]
    cdef Py_ssize_t i, j
    cdef double[:, :] out = np.empty((H, W), dtype=np.float64)

    with nogil:
        for j in range(W):
            out[0, j] = v[H - 1, j] - v[0, j]  # periodic
        for i in range(1, H):
            for j in range(W):
                out[i, j] = v[i - 1, j] - v[i, j]
    return np.asarray(out)


# ═══════════════════════════════════════════════════════════════
#  HS weights — fused gradient + tanh in a single pass
# ═══════════════════════════════════════════════════════════════

def compute_hs_weights(
    double[:, :] dx not None,
    double[:, :] dy not None,
    double[:, :] sigma_x not None,
    double b,
):
    """Compute E[ω] for the HS prior (image-space formulation).

    Returns (gamma_x, gamma_y) — diagonal weight arrays.
    """
    cdef Py_ssize_t H = dx.shape[0]
    cdef Py_ssize_t W = dx.shape[1]
    cdef Py_ssize_t i, j
    cdef double alpha = 1.0 / b
    cdef double sg, nu_x, nu_y
    cdef double[:, :] gx = np.empty((H, W), dtype=np.float64)
    cdef double[:, :] gy = np.empty((H, W), dtype=np.float64)

    with nogil:
        for i in range(H):
            for j in range(W):
                sg = 2.0 * sigma_x[i, j]
                nu_x = sqrt(dx[i, j] * dx[i, j] + sg + EPSILON)
                nu_y = sqrt(dy[i, j] * dy[i, j] + sg + EPSILON)
                gx[i, j] = (alpha * tanh(alpha * nu_x)) / nu_x
                gy[i, j] = (alpha * tanh(alpha * nu_y)) / nu_y

    return np.asarray(gx), np.asarray(gy)


def compute_hs_weights_scalar(
    double[:, :] x_n not None,
    double[:, :] sigma_sq_n not None,
    double alpha_n,
):
    """Compute HS weights for filter-space formulation.

    θ_n(i) = α_n · tanh(α_n · ξ_n(i)) / ξ_n(i),
    where ξ_n(i) = sqrt(x_n²(i) + σ²_n(i) + ε).
    """
    cdef Py_ssize_t H = x_n.shape[0]
    cdef Py_ssize_t W = x_n.shape[1]
    cdef Py_ssize_t i, j
    cdef double xi
    cdef double[:, :] theta = np.empty((H, W), dtype=np.float64)

    with nogil:
        for i in range(H):
            for j in range(W):
                xi = sqrt(x_n[i, j] * x_n[i, j]
                          + sigma_sq_n[i, j] + EPSILON)
                theta[i, j] = (alpha_n * tanh(alpha_n * xi)) / xi

    return np.asarray(theta)


# ═══════════════════════════════════════════════════════════════
#  Kernel utilities
# ═══════════════════════════════════════════════════════════════

def project_kernel(double[:, :] h not None):
    """Project kernel onto probability simplex h≥0, Σh=1."""
    cdef Py_ssize_t kh = h.shape[0]
    cdef Py_ssize_t kw = h.shape[1]
    cdef Py_ssize_t i, j
    cdef double s = 0.0
    cdef double val

    result = np.asarray(h).copy()
    cdef double[:, :] r = result

    with nogil:
        for i in range(kh):
            for j in range(kw):
                if r[i, j] < 0.0:
                    r[i, j] = 0.0
                s += r[i, j]

    if s > EPSILON:
        with nogil:
            for i in range(kh):
                for j in range(kw):
                    r[i, j] /= s
    else:
        val = 1.0 / <double>(kh * kw)
        with nogil:
            for i in range(kh):
                for j in range(kw):
                    r[i, j] = val

    return result


def threshold_kernel(double[:, :] h not None, double ratio=0.05):
    """Threshold small kernel values, then project onto simplex."""
    cdef Py_ssize_t kh = h.shape[0]
    cdef Py_ssize_t kw = h.shape[1]
    cdef Py_ssize_t i, j
    cdef double peak = 0.0, thr

    result = np.maximum(np.asarray(h), 0.0).copy()
    cdef double[:, :] r = result

    with nogil:
        for i in range(kh):
            for j in range(kw):
                if r[i, j] > peak:
                    peak = r[i, j]
        thr = ratio * peak
        for i in range(kh):
            for j in range(kw):
                if r[i, j] < thr:
                    r[i, j] = 0.0

    return project_kernel(r)


# ═══════════════════════════════════════════════════════════════
#  FFT wrappers  (delegate to numpy — can't beat MKL/FFTPACK)
# ═══════════════════════════════════════════════════════════════

from numpy.fft import fft2, ifft2


def psf2otf(psf, shape):
    """PSF → OTF via zero-padding + circshift + fft2."""
    cdef int kh = psf.shape[0]
    cdef int kw = psf.shape[1]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf, kernel_shape):
    """OTF → PSF via ifft2 + circshift + crop."""
    cdef int kh = kernel_shape[0]
    cdef int kw = kernel_shape[1]
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, kh // 2, axis=0)
    psf_full = np.roll(psf_full, kw // 2, axis=1)
    return psf_full[:kh, :kw]


def precompute_gradient_operators(shape):
    """DFT of first-order finite-difference operators."""
    cdef int H = shape[0]
    cdef int W = shape[1]
    dx = np.zeros((H, W), dtype=np.float64)
    dx[0, 0] = -1.0
    dx[0, 1] = 1.0
    dy = np.zeros((H, W), dtype=np.float64)
    dy[0, 0] = -1.0
    dy[1, 0] = 1.0
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx) ** 2 + np.abs(F_dy) ** 2
    return F_dx, F_dy, F_grad_sq


def init_gaussian_kernel(shape, sigma=None):
    """Gaussian kernel normalised to unit sum."""
    cdef int kh = shape[0]
    cdef int kw = shape[1]
    if sigma is None:
        sigma = max(kh, kw) / 6.0
    cy, cx = kh // 2, kw // 2
    y, x = np.ogrid[-cy:kh - cy, -cx:kw - cx]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fft_convolve(x, h):
    """Circular convolution h*x via FFT."""
    F_h = psf2otf(h, (x.shape[0], x.shape[1]))
    return np.real(ifft2(F_h * fft2(x)))


def edgetaper(img, kernel, n_taper=None):
    """Edge tapering for FFT boundary artefact suppression."""
    from scipy.signal import fftconvolve

    cdef int h_img = img.shape[0]
    cdef int w_img = img.shape[1]
    cdef int kh = kernel.shape[0]
    cdef int kw = kernel.shape[1]

    if n_taper is None:
        n_taper = max(kh, kw)

    cdef int nt = n_taper
    cdef int i

    dx = np.arange(w_img, dtype=np.float64)
    wx = np.ones(w_img, dtype=np.float64)
    for i in range(w_img):
        if i < nt:
            wx[i] = 0.5 * (1.0 + cos(M_PI * (i - nt) / nt))
        elif i >= w_img - nt:
            wx[i] = 0.5 * (1.0 + cos(M_PI * (i - (w_img - nt - 1)) / nt))

    dy = np.arange(h_img, dtype=np.float64)
    wy = np.ones(h_img, dtype=np.float64)
    for i in range(h_img):
        if i < nt:
            wy[i] = 0.5 * (1.0 + cos(M_PI * (i - nt) / nt))
        elif i >= h_img - nt:
            wy[i] = 0.5 * (1.0 + cos(M_PI * (i - (h_img - nt - 1)) / nt))

    W = np.outer(wy, wx)
    blurred = fftconvolve(img, kernel, mode='same')
    return img * W + blurred * (1.0 - W)
