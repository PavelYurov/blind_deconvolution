"""
Utility functions for the MHDM (Multiscale Hierarchical Decomposition Method)
blind deconvolution algorithm.
"""

import numpy as np
from typing import Tuple

def compute_fourier_weights(m: int, n: int) -> np.ndarray:
    """
    Compute discrete Sobolev-type Fourier weights.
    Parameters:
    m, n : int
        Spatial dimensions (rows, columns).
    Returns:
    delta : ndarray, shape (m, n), float64
        Weight array.  ``delta[0, 0] == 1``.
    """
    j = np.arange(m, dtype=np.float64)
    l = np.arange(n, dtype=np.float64)
    row_term = 2.0 * m ** 2 * (1.0 - np.cos(2.0 * np.pi * j / m))   # (m,)
    col_term = 2.0 * n ** 2 * (1.0 - np.cos(2.0 * np.pi * l / n))   # (n,)
    delta = 1.0 + row_term[:, None] + col_term[None, :]
    return delta


def compute_conjugate_indices(
    m: int, n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute index sets for enforcing Hermitian symmetry on a 2D DFT grid.
    Parameters:
    m, n : int
        Spatial dimensions (rows, columns).

    Returns:
    primary : ndarray, shape (N_pair, 2), intp
        Primary 2D indices for each conjugate pair.
    conjugate : ndarray, shape (N_pair, 2), intp
        Corresponding conjugate 2D indices.
    self_conjugate : ndarray, shape (N_self, 2), intp
        Self-conjugate indices (excluding DC).
    """
    primary_list = []
    conjugate_list = []
    self_conj_list = []
    visited = set()

    for j in range(m):
        for l in range(n):
            if j == 0 and l == 0:
                continue

            cj = (m - j) % m
            cl = (n - l) % n
            if (j, l) == (cj, cl):
                self_conj_list.append((j, l))
                continue

            pair = frozenset(((j, l), (cj, cl)))
            if pair not in visited:
                visited.add(pair)
                primary_list.append((j, l))
                conjugate_list.append((cj, cl))

    def _to_array(lst):
        if lst:
            return np.array(lst, dtype=np.intp)
        return np.empty((0, 2), dtype=np.intp)

    return _to_array(primary_list), _to_array(conjugate_list), _to_array(self_conj_list)


def psf2otf(psf: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a spatial-domain PSF to a frequency-domain OTF.
    Parameters:
    psf : ndarray, shape (ph, pw)
        Spatial-domain point-spread function.
    output_size : tuple of (int, int)
        Desired (rows, cols) of the output OTF.
    Returns:
    otf : ndarray, shape *output_size*, complex128
        Frequency-domain optical transfer function.
    """
    ph, pw = psf.shape
    padded = np.zeros(output_size, dtype=np.float64)
    padded[:ph, :pw] = psf

    shift_y = -(ph // 2)
    shift_x = -(pw // 2)
    padded = np.roll(padded, shift=(shift_y, shift_x), axis=(0, 1))

    return np.fft.fft2(padded)


def otf2psf(
    otf: np.ndarray,
    psf_size: Tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Convert a frequency-domain OTF to a spatial-domain PSF.
    Parameters:
    otf : ndarray, shape (m, n), complex
        Frequency-domain OTF.
    psf_size : tuple of (int, int) or None
        If given, crop the output to this centred (rows, cols) region.
        If None, return the full-size PSF.

    Returns:
    psf : ndarray, shape psf_size or (m, n), float64
        Spatial-domain PSF.
    """
    m, n = otf.shape
    psf_full = np.real(np.fft.ifft2(otf))
    psf_full = np.fft.fftshift(psf_full)

    if psf_size is None:
        return psf_full

    kh, kw = psf_size
    cy, cx = m // 2, n // 2
    top = cy - kh // 2
    left = cx - kw // 2
    return psf_full[top:top + kh, left:left + kw]


def estimate_noise_sigma(
    image: np.ndarray,
    sigma_floor: float = 2.0,
) -> float:
    """
    Estimate the standard deviation of additive white Gaussian noise
    from a single image.

    Parameters:
    image : ndarray, shape (H, W)
        Observed image. The estimator is scale-invariant (linear).
        If image is in [0, 1], returns sigma in [0, 1].
        If image is in [0, 255], returns sigma in [0, 255].
    sigma_floor : float
        Minimum returned sigma.
        For [0, 1] images, default 0.005 is appropriate.
        For [0, 255] images, use ~1.0.

    Returns:
    sigma : float
        Estimated noise standard deviation.
    """
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ], dtype=np.float64)

    from scipy.signal import fftconvolve
    residual = fftconvolve(image, laplacian_kernel, mode='valid')

    sigma = np.median(np.abs(residual)) / (0.6745 * np.sqrt(20.0))
    return float(max(sigma, sigma_floor))


def complex_sign(z: np.ndarray) -> np.ndarray:
    """
    Parameters:
    z : ndarray (complex or real)
        Input array.

    Returns:
    s : ndarray
        Complex sign, same shape and dtype as input.
    """
    magnitude = np.abs(z)
    out = np.zeros_like(z)
    mask = magnitude > 0.0
    out[mask] = z[mask] / magnitude[mask]
    return out
