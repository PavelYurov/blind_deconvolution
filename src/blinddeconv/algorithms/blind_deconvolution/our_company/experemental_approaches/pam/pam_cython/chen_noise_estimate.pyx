import numpy as np

__all__ = ['estimate_noise_level']

def _im2patch(im, pch_size, stride=1):

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

    im = np.asarray(image, dtype=np.float64)

    if im.max() > 1.0:
        im = im / 255.0

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    elif im.ndim == 2:
        im = im[np.newaxis, :, :]
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={im.ndim}")

    pch = _im2patch(im, pch_size, stride=3)
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))
    d = pch.shape[0]

    if num_pch < d:
        return 0.0

    mu = pch.mean(axis=1, keepdims=True)
    X = pch - mu
    sigma_X = X @ X.T / num_pch

    sig_values, _ = np.linalg.eigh(sigma_X)
    sig_values.sort()

    for ii in range(-1, -d - 1, -1):
        subset = sig_values[:ii]
        if len(subset) == 0:
            break
        tau = np.mean(subset)
        if tau < 0:
            break
        if np.sum(subset > tau) == np.sum(subset < tau):
            return float(np.sqrt(max(tau, 0.0)))

    min_eig = float(sig_values[0])
    if min_eig > 0:
        return float(np.sqrt(min_eig))
    return 0.0
