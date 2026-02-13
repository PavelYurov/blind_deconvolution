"""
Utility functions for the MHDM (Multiscale Hierarchical Decomposition Method)
blind deconvolution algorithm.

Provides Fourier-domain helpers: Sobolev weights, conjugate symmetry index
computation, PSF/OTF conversions, and noise level estimation.

References
----------
[1] Wolf, T., Kindermann, S., Resmerita, E., Vese, L.
    "Applications of multiscale hierarchical decomposition to blind
    deconvolution." arXiv:2409.08734v5, 2025.
[2] Justen, L. "Blind Deconvolution: Theory, Regularization and
    Applications." PhD thesis, p. 110 — Sobolev Fourier weights.
"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Fourier-domain Sobolev weights
# ---------------------------------------------------------------------------

def compute_fourier_weights(m: int, n: int) -> np.ndarray:
    r"""
    Compute discrete Sobolev-type Fourier weights delta(j, l).

    The weights are the eigenvalues of (I - Delta) on the periodic grid,
    discretised for 2D DFT of size (m, n):

    .. math::
        \delta_{j,l} = 1
            + 2 m^2 \bigl(1 - \cos(2\pi j / m)\bigr)
            + 2 n^2 \bigl(1 - \cos(2\pi l / n)\bigr),
        \qquad j=0,\dots,m{-}1,\; l=0,\dots,n{-}1.

    Ref: [2], page 110, adapted to ``numpy.fft.fft2`` index ordering.

    Parameters
    ----------
    m, n : int
        Spatial dimensions (rows, columns).

    Returns
    -------
    delta : ndarray, shape (m, n)
        Weight array.  delta[0, 0] = 1 (DC component).
    """
    j = np.arange(m)
    l = np.arange(n)
    # Column vector for row-frequencies, row vector for col-frequencies
    row_term = 2.0 * m**2 * (1.0 - np.cos(2.0 * np.pi * j / m))  # (m,)
    col_term = 2.0 * n**2 * (1.0 - np.cos(2.0 * np.pi * l / n))  # (n,)
    delta = 1.0 + row_term[:, None] + col_term[None, :]
    return delta


# ---------------------------------------------------------------------------
# Conjugate-symmetry index pairs
# ---------------------------------------------------------------------------

def compute_conjugate_indices(m: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute unique conjugate-symmetry index pairs for a real-valued 2D DFT.

    For a real signal, the DFT satisfies Hermitian symmetry:
        F[j, l] = conj(F[(m-j) mod m, (n-l) mod n]).

    This function returns two integer arrays ``primary`` and ``conjugate``
    such that, for every non-self-conjugate frequency (j, l), the value at
    ``primary[k]`` is computed and the value at ``conjugate[k]`` is set to
    its complex conjugate.  The DC component (0, 0) is excluded (handled
    separately in the algorithm).

    Parameters
    ----------
    m, n : int
        Spatial dimensions.

    Returns
    -------
    primary : ndarray, shape (N, 2)
        Array of (row, col) indices for the "primary" half.
    conjugate : ndarray, shape (N, 2)
        Corresponding conjugate indices.
    """
    primary = []
    conjugate = []
    visited = set()

    for j in range(m):
        for l in range(n):
            cj = (m - j) % m
            cl = (n - l) % n

            # Skip DC component — handled separately
            if j == 0 and l == 0:
                continue

            # Skip self-conjugate points (e.g. Nyquist frequencies)
            if (j, l) == (cj, cl):
                continue

            pair = frozenset(((j, l), (cj, cl)))
            if pair not in visited:
                visited.add(pair)
                primary.append((j, l))
                conjugate.append((cj, cl))

    return np.array(primary, dtype=np.intp), np.array(conjugate, dtype=np.intp)


# ---------------------------------------------------------------------------
# PSF <-> OTF conversions  (equivalents of MATLAB psf2otf / otf2psf)
# ---------------------------------------------------------------------------

def psf2otf(psf: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a point-spread function (PSF) to an optical transfer function
    (OTF), replicating MATLAB's ``psf2otf``.

    The PSF is zero-padded to *output_size* and circularly shifted so that
    the centre of the PSF sits at index (0, 0) before taking the 2D FFT.

    Parameters
    ----------
    psf : ndarray
        Spatial-domain PSF (arbitrary size).
    output_size : tuple of int
        Desired (rows, cols) of the output OTF.

    Returns
    -------
    otf : ndarray (complex)
        Frequency-domain OTF of shape *output_size*.
    """
    ph, pw = psf.shape
    padded = np.zeros(output_size, dtype=psf.dtype)
    padded[:ph, :pw] = psf

    # Circular shift so that the centre of the PSF is at (0, 0)
    shift_y = -(ph // 2)
    shift_x = -(pw // 2)
    padded = np.roll(padded, shift=(shift_y, shift_x), axis=(0, 1))

    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray,
            psf_size: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Convert an OTF back to the spatial-domain PSF, replicating MATLAB's
    ``otf2psf``.

    Parameters
    ----------
    otf : ndarray (complex)
        Frequency-domain OTF.
    psf_size : tuple of int or None
        If given, crop the result to this size (centred).
        If None, return the full-size PSF.

    Returns
    -------
    psf : ndarray (real)
        Spatial-domain PSF.
    """
    m, n = otf.shape
    psf_full = np.real(np.fft.ifft2(otf))

    # Circularly shift the PSF so that the peak is at the centre
    psf_full = np.fft.fftshift(psf_full)

    if psf_size is None:
        return psf_full

    kh, kw = psf_size
    cy, cx = m // 2, n // 2
    top = cy - kh // 2
    left = cx - kw // 2
    return psf_full[top:top + kh, left:left + kw]


# ---------------------------------------------------------------------------
# Noise level estimation (Robust Median Estimator)
# ---------------------------------------------------------------------------

def estimate_noise_sigma(image: np.ndarray) -> float:
    r"""
    Estimate the standard deviation of additive white Gaussian noise
    from a single (possibly blurred) image using the Robust Median
    Estimator on high-frequency Laplacian residuals.

    .. math::
        \hat\sigma = \frac{\mathrm{median}\bigl(|\nabla^2 y|\bigr)}{0.6745
        \,\sqrt{2}}

    This is a standard non-parametric estimator (Donoho & Johnstone, 1994)
    applied to the Laplacian rather than wavelet coefficients.

    Parameters
    ----------
    image : ndarray, shape (H, W)
        Observed (noisy, possibly blurred) image in [0, 1].

    Returns
    -------
    sigma : float
        Estimated noise standard deviation.
    """
    # 3×3 Laplacian kernel
    laplacian = np.array([[0,  1, 0],
                          [1, -4, 1],
                          [0,  1, 0]], dtype=np.float64)

    from scipy.signal import fftconvolve
    residual = fftconvolve(image, laplacian, mode='valid')

    # MAD estimator with correction factor for Gaussian
    sigma = np.median(np.abs(residual)) / (0.6745 * np.sqrt(2.0))
    return float(max(sigma, 1e-8))


# ---------------------------------------------------------------------------
# Complex-valued sign (MATLAB-compatible)
# ---------------------------------------------------------------------------

def complex_sign(z: np.ndarray) -> np.ndarray:
    r"""
    Element-wise complex sign, matching MATLAB's ``sign`` for complex input:

    .. math::
        \operatorname{sign}(z) = \begin{cases}
            z / |z|, & z \neq 0, \\
            0,       & z = 0.
        \end{cases}

    Parameters
    ----------
    z : ndarray (complex or real)
        Input array.

    Returns
    -------
    s : ndarray
        Complex sign of *z*.
    """
    magnitude = np.abs(z)
    out = np.zeros_like(z)
    nonzero = magnitude > 0.0
    out[nonzero] = z[nonzero] / magnitude[nonzero]
    return out
