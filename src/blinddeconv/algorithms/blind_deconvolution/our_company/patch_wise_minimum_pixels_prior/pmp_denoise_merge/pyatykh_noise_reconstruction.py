"""
pyatykh_noise_reconstruction.py

Poisson-Gaussian noise parameter estimation via PCA + VST + kurtosis.

Reference:
    Pyatykh S., Hesser J., Zheng L.:
    "Image Noise Level Estimation by Principal Component Analysis",
    IEEE Transactions on Image Processing, vol. 22, no. 2, 2013.

Based on NoiseParameterEstimation_Fast.py from the original authors.

Noise model:  y = Poisson(x / a) * a + N(0, b)
    a — Poisson (signal-dependent) noise parameter
    b — Gaussian (signal-independent) noise variance

Usage:
    from pyatykh_noise_reconstruction import estimate_noise_params
    result = estimate_noise_params(noisy_image)
    # result = {'a': ..., 'b': ..., 'sigma': ..., 'sigma_norm': ..., 'noise_type': ...}
"""

import numpy as np
import scipy.linalg
from scipy.stats import kurtosis
from scipy.optimize import fminbound

__all__ = ['estimate_noise_params']


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _im2col(image, m1, m2):
    """Sliding-window patch extraction (like MATLAB im2col 'sliding')."""
    rows, cols = image.shape
    s0, s1 = image.strides
    n_rows = rows - m1 + 1
    n_cols = cols - m2 + 1
    out = np.lib.stride_tricks.as_strided(
        image, shape=(m1, m2, n_rows, n_cols), strides=(s0, s1, s0, s1))
    return out.reshape(m1 * m2, -1)


def _get_valid_block_index(image, m1, m2):
    """Find indices of non-degenerate blocks (no constant/saturated pixels)."""
    block = _im2col(image, m1, m2)
    minimums = np.min(block, axis=0)
    maximums = np.max(block, axis=0)
    equal_minmax = minimums == maximums
    invalid_grayvalue = np.unique(minimums[equal_minmax])
    # A pixel is invalid if it matches a constant-block value, <= 0, or >= 255
    invalid_mask = (np.isin(block, invalid_grayvalue) |
                    (block <= 0) | (block >= 255))
    # A block is valid only if ALL its pixels are valid
    blocks_ok = ~invalid_mask.any(axis=0)
    valid_block_index = np.where(blocks_ok)
    return np.array(valid_block_index).T


def _vst(image, phi):
    """Variance Stabilizing Transform parameterized by angle phi.

    phi = 0       → pure Poisson  (a = cos(0) = 1, b = sin(0) = 0)
    phi = pi/2    → pure Gaussian (a = cos(pi/2) = 0, b = sin(pi/2) = 1)
    """
    a = np.cos(phi)
    b = np.sin(phi)
    if a > np.finfo(float).eps:
        return (2.0 / a) * np.sqrt(np.maximum(a * image + b, 0.0))
    return image / np.sqrt(max(b, np.finfo(float).eps))


def _get_blocks(image, phi, row_parity, valid_block_index, m1, m2):
    """Extract half-row blocks from VST-transformed image."""
    block = _im2col(_vst(image, phi), m1, m2)
    block = block[row_parity - 1::2, valid_block_index]
    return np.squeeze(block).T


def _pca_svd_score(data):
    """PCA scores: U * S from SVD of centered data."""
    centered = data - np.mean(data, axis=0)
    U, s, _ = scipy.linalg.svd(centered, full_matrices=False,
                                check_finite=False)
    return U * s


def _pca_svd_latent(data):
    """PCA eigenvalues (variance explained) via SVD."""
    centered = data - np.mean(data, axis=0)
    s = scipy.linalg.svd(centered, full_matrices=False,
                          compute_uv=False, check_finite=False)
    return (s ** 2) / (data.shape[0] - 1)


def _sort_blocks(image, phi, valid_block_index, m1, m2):
    """Sort blocks by texture complexity (ascending — smoothest first)."""
    block = _get_blocks(image, phi, 2, valid_block_index, m1, m2)
    scores = _pca_svd_score(block)
    # Texture energy = sum of squared scores from 4th component onwards
    energy = np.sum(np.square(scores[:, 3:]), axis=1)
    t = np.column_stack((valid_block_index, energy))
    t = t[np.argsort(t[:, 1])]
    return t[:, 0]


def _compute_kurtosis(phi, image, tau, block_count, m1, m2):
    """Kurtosis of the noise component in VST domain."""
    block = _get_blocks(image, phi, 1, tau[:block_count], m1, m2)
    scores = _pca_svd_score(block)
    g = (kurtosis(scores[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return g


def _compute_kurtosis_and_block(phi, image, tau, block_count, m1, m2):
    """Kurtosis + return block data for subsequent std computation."""
    block = _get_blocks(image, phi, 1, tau[:block_count], m1, m2)
    scores = _pca_svd_score(block)
    g = (kurtosis(scores[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return g, block


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def estimate_noise_params(image, blocksize=7):
    """
    Estimate Poisson-Gaussian noise parameters from a single image.

    Uses PCA + Variance Stabilizing Transform + kurtosis optimization
    to decompose noise into a Poisson (signal-dependent) component and
    a Gaussian (signal-independent) component.

    Parameters
    ----------
    image : ndarray
        Noisy image. Accepts:
        - H×W grayscale, uint8 [0, 255]
        - H×W grayscale, float [0, 1] (auto-scaled to [0, 255])
        - H×W×C color (auto-converted to grayscale)
    blocksize : int, optional
        Block size for patch analysis (default: 7).

    Returns
    -------
    result : dict
        'a'          — Poisson noise parameter (float)
        'b'          — Gaussian noise variance (float)
        'sigma'      — estimated noise σ in [0, 255] pixel scale (float)
        'sigma_norm' — estimated noise σ in [0, 1] scale (float)
        'noise_type' — 'gaussian' | 'poisson' | 'poisson_gaussian' | 'unknown'
    """
    img = np.asarray(image, dtype=np.float64)

    # Handle color → grayscale
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1]
                   + 0.1140 * img[:, :, 2])
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[2]}")

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after conversion, got ndim={img.ndim}")

    # Scale to [0, 255] if input is [0, 1]
    if img.max() <= 1.0:
        img = img * 255.0

    m1, m2 = blocksize, blocksize
    valid_block_index = _get_valid_block_index(img, m1, m2)

    _empty = {'a': 0.0, 'b': 0.0, 'sigma': 0.0, 'sigma_norm': 0.0,
              'noise_type': 'unknown'}

    if len(valid_block_index) < 1000:
        return _empty

    tau = _sort_blocks(img, 0.0, valid_block_index, m1, m2).astype(int)

    block_count = min(20000, len(tau))
    curr_phi = 0.0
    curr_sigma = 0.0

    while block_count <= len(tau):
        opt_phi = fminbound(
            _compute_kurtosis, 0.0, np.pi / 2 - 0.001,
            args=(img, tau, block_count, m1, m2),
            xtol=0.01, maxfun=10000, disp=0)

        opt_kurtosis, block = _compute_kurtosis_and_block(
            opt_phi, img, tau, block_count, m1, m2)

        if opt_kurtosis < 3 or curr_phi == 0:
            phi_converged = abs(opt_phi - curr_phi) < 0.0005
            curr_phi = opt_phi
            latent = _pca_svd_latent(block)
            curr_sigma = float(np.sqrt(max(latent[-1], 0.0)))
            if phi_converged:
                break
        else:
            break
        block_count += 5000

    a = curr_sigma ** 2 * np.cos(curr_phi)
    b = curr_sigma ** 2 * np.sin(curr_phi)

    # Classify noise type based on a/b ratio
    if a < 1e-6 and b < 1e-6:
        noise_type = 'unknown'
    elif a < 1e-6:
        noise_type = 'gaussian'
    elif b / max(a, 1e-10) > 10:
        noise_type = 'gaussian'
    elif a / max(b, 1e-10) > 10:
        noise_type = 'poisson'
    else:
        noise_type = 'poisson_gaussian'

    # Noise variance is signal-dependent: Var[y] = a*x + b.
    # Evaluate σ at mean brightness for a representative scalar estimate.
    mean_brightness = float(np.mean(img))
    sigma_255 = float(np.sqrt(max(a * mean_brightness + b, 0.0)))

    return {
        'a': float(a),
        'b': float(b),
        'sigma': sigma_255,
        'sigma_norm': sigma_255 / 255.0,
        'sigma_gaussian': float(np.sqrt(max(b, 0.0))),
        'noise_type': noise_type,
    }
