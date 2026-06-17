"""
chen_noise_estimate.py

Noise level estimation using PCA eigenvalue analysis of image patches.

Reference:
    Chen G., Zhu F., Heng P.A.:
    "An Efficient Statistical Method for Image Noise Level Estimation",
    ICCV 2015.

Original implementation by Zongsheng Yue (2019).
Cleaned up and adapted for framework integration.

Usage:
    from chen_noise_estimate import estimate_noise_level
    sigma = estimate_noise_level(noisy_image)  # σ in [0, 1] scale
"""

import numpy as np

__all__ = ['estimate_noise_level']


def _im2patch(im, pch_size, stride=1):
    """
    Extract patches from a C×H×W image tensor.

    Parameters
    ----------
    im : ndarray, shape (C, H, W)
    pch_size : int
    stride : int

    Returns
    -------
    patches : ndarray, shape (C, pch_size, pch_size, num_patches)
    """
    pch_H = pch_W = int(pch_size)
    stride_H = stride_W = int(stride)

    C, H, W = im.shape
    num_H = len(range(0, H - pch_H + 1, stride_H))
    num_W = len(range(0, W - pch_W + 1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H * pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H - pch_H + ii + 1:stride_H,
                       jj:W - pch_W + jj + 1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def estimate_noise_level(image, pch_size=8):
    """
    Estimate additive white Gaussian noise σ from a single image
    using PCA eigenvalue analysis of patches.

    The method extracts overlapping patches, computes their covariance
    matrix, and finds the noise floor from the smallest eigenvalues
    using a median-based stopping criterion.

    Parameters
    ----------
    image : ndarray
        H×W (grayscale) or H×W×C (color).
        Float [0, 1] or uint8 [0, 255] — auto-detected.
    pch_size : int, optional
        Patch size (default 8).

    Returns
    -------
    sigma : float
        Estimated noise σ in [0, 1] scale.
        Multiply by 255 for pixel-domain σ.
        Returns 0.0 if estimation fails.
    """
    im = np.asarray(image, dtype=np.float64)

    # Normalize to [0, 1]
    if im.max() > 1.0:
        im = im / 255.0

    # Convert to C×H×W
    if im.ndim == 3:
        im = im.transpose((2, 0, 1))   # H×W×C → C×H×W
    elif im.ndim == 2:
        im = im[np.newaxis, :, :]       # H×W → 1×H×W
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={im.ndim}")

    # Extract patches with stride 3
    pch = _im2patch(im, pch_size, stride=3)
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))    # d × num_pch
    d = pch.shape[0]

    if num_pch < d:
        return 0.0

    # Sample covariance matrix
    mu = pch.mean(axis=1, keepdims=True)
    X = pch - mu
    sigma_X = X @ X.T / num_pch

    # Eigenvalue decomposition (eigh — faster for symmetric matrices)
    sig_values, _ = np.linalg.eigh(sigma_X)
    sig_values.sort()

    # Median condition: peel away largest eigenvalues (signal energy),
    # find the subset where mean == median (pure noise eigenvalues).
    for ii in range(-1, -d - 1, -1):
        subset = sig_values[:ii]
        if len(subset) == 0:
            break
        tau = np.mean(subset)
        if tau < 0:
            break
        if np.sum(subset > tau) == np.sum(subset < tau):
            return float(np.sqrt(max(tau, 0.0)))

    # Fallback: minimum eigenvalue
    min_eig = float(sig_values[0])
    if min_eig > 0:
        return float(np.sqrt(min_eig))
    return 0.0
