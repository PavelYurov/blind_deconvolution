"""
screenot.py

Image denoising via ScreeNOT (optimal adaptive SVD thresholding).

Based on:
    Donoho, Gavish, Romanov:
    "ScreeNOT: Exact MSE-optimal singular value thresholding in correlated noise."
    Annals of Statistics (2023).

ScreeNOT finds the MSE-optimal hard threshold for singular values of a matrix
Y = X + Z, where X is low-rank signal and Z is additive noise with arbitrary
(unknown) correlation structure.  The threshold is computed adaptively from
the observed singular values — no knowledge of noise statistics is needed.

Two modes:
    'full'  — treat the entire image as a matrix Y (H×W) and apply
              ScreeNOT directly.  Simple, fast, no artifacts.
    'patch' — extract overlapping patches into a matrix, apply ScreeNOT,
              aggregate back.  Better for high-texture images but slower.

Dependencies: numpy (only).
"""

import numpy as np
from numpy.linalg import svd

__all__ = [
    'adaptive_hard_thresholding',
    'screenot_denoise',
]


def _Phi(y, fZ):

    return np.mean(y / (y ** 2 - fZ ** 2))


def _Phid(y, fZ):

    return np.mean(-(y ** 2 + fZ ** 2) / (y ** 2 - fZ ** 2) ** 2)


def _D(y, fZ, gamma):

    phi = _Phi(y, fZ)
    return phi * (gamma * phi + (1 - gamma) / y)


def _Dd(y, fZ, gamma):

    phi = _Phi(y, fZ)
    phid = _Phid(y, fZ)
    return (phid * (gamma * phi + (1 - gamma) / y)
            + phi * (gamma * phid - (1 - gamma) / y ** 2))


def _F(y, fZ, gamma):

    d = _D(y, fZ, gamma)
    dd = _Dd(y, fZ, gamma)
    return y * dd / d


def _create_pseudo_noise(fY, k, strategy='i'):


    fZ = np.sort(fY)
    p = fZ.size
    if k >= p:
        raise ValueError('k too large: requires k < min(n, p)')

    if k > 0:
        if strategy == '0':
            fZ[-k:] = 0
        elif strategy == 'w':
            fZ[-k:] = fZ[-k - 1]
        elif strategy == 'i':
            if 2 * k + 1 >= p:
                raise ValueError(
                    'k too large for imputation: requires 2*k+1 < min(n, p)')
            diff = fZ[-k - 1] - fZ[-2 * k - 1]
            for l in range(1, k + 1):
                a = (1 - ((l - 1) / k) ** (2 / 3)) / (2 ** (2 / 3) - 1)
                fZ[-l] = fZ[-k - 1] + a * diff
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}', use 'i', 'w', or '0'")
    return fZ


def _compute_opt_threshold(fZ, gamma):

    low = np.max(fZ)
    high = low + 2.0
    while _F(high, fZ, gamma) < -4:
        low = high
        high = 2 * high

    eps = 1e-6
    while high - low > eps:
        mid = (high + low) / 2
        if _F(mid, fZ, gamma) < -4:
            low = mid
        else:
            high = mid
    return (high + low) / 2


def adaptive_hard_thresholding(Y, k, strategy='i'):


    U, fY, Vt = svd(Y, full_matrices=False)
    gamma = min(Y.shape[0] / Y.shape[1], Y.shape[1] / Y.shape[0])

    fZ = _create_pseudo_noise(fY, k, strategy=strategy)
    Topt = _compute_opt_threshold(fZ, gamma)

    fY_new = fY * (fY > Topt)
    Xest = U @ np.diag(fY_new) @ Vt
    r = int(np.sum(fY_new > 0))

    return Xest, Topt, r


def screenot_denoise(image, k=10, strategy='i', mode='full',
                     patch_size=8, stride=3):


    if image.ndim != 2:
        raise ValueError(f'Expected 2D image, got shape {image.shape}')

    if mode == 'full':
        return _denoise_full(image, k, strategy)
    elif mode == 'patch':
        return _denoise_patch(image, k, strategy, patch_size, stride)
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'full' or 'patch'")


def _denoise_full(image, k, strategy):

    H, W = image.shape
    min_dim = min(H, W)


    max_k = min_dim // 2 - 1
    if k > max_k:
        k = max(1, max_k)

    try:
        denoised, Topt, r = adaptive_hard_thresholding(
            image, k, strategy=strategy)
    except ValueError:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'mode': 'full', 'skipped': True,
        }

    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised, {
        'Topt': float(Topt),
        'rank': r,
        'mode': 'full',
        'image_shape': (H, W),
        'skipped': False,
    }


def _extract_patches(image, patch_size, stride):

    H, W = image.shape
    positions = []
    rows = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
            rows.append(patch.ravel())
            positions.append((y0, x0))
    return np.array(rows, dtype=np.float64), positions


def _aggregate_patches(patches, positions, patch_size, image_shape):

    H, W = image_shape
    accum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for i, (y0, x0) in enumerate(positions):
        patch_2d = patches[i].reshape(patch_size, patch_size)
        accum[y0:y0 + patch_size, x0:x0 + patch_size] += patch_2d
        count[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0
    count = np.maximum(count, 1.0)
    return accum / count


def _denoise_patch(image, k, strategy, patch_size, stride):

    H, W = image.shape
    if H < patch_size or W < patch_size:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'mode': 'patch', 'skipped': True,
        }

    patches, positions = _extract_patches(image, patch_size, stride)
    n_patches, dim = patches.shape

    max_k = min(n_patches, dim) // 2 - 1
    if k > max_k:
        k = max(1, max_k)

    try:
        denoised_patches, Topt, r = adaptive_hard_thresholding(
            patches, k, strategy=strategy)
    except ValueError:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'n_patches': n_patches,
            'patch_matrix_shape': patches.shape, 'mode': 'patch',
            'skipped': True,
        }

    denoised = _aggregate_patches(
        denoised_patches, positions, patch_size, (H, W))
    denoised = np.clip(denoised, 0.0, 1.0)

    return denoised, {
        'Topt': float(Topt),
        'rank': r,
        'n_patches': n_patches,
        'patch_matrix_shape': patches.shape,
        'mode': 'patch',
        'skipped': False,
    }
