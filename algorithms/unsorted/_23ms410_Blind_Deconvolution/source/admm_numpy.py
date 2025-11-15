from __future__ import annotations

import numpy as np


def psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape, dtype=np.float64)
    psf_shape = psf.shape
    padded[: psf_shape[0], : psf_shape[1]] = psf
    for axis, size in enumerate(psf_shape):
        padded = np.roll(padded, -size // 2, axis=axis)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_shape: tuple[int, int]) -> np.ndarray:
    spatial = np.fft.ifft2(otf)
    spatial = np.roll(spatial, psf_shape[0] // 2, axis=0)
    spatial = np.roll(spatial, psf_shape[1] // 2, axis=1)
    return np.real(spatial[: psf_shape[0], : psf_shape[1]])


def _soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def _gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_x = np.roll(image, -1, axis=1) - image
    grad_y = np.roll(image, -1, axis=0) - image
    grad_x[:, -1] = 0.0
    grad_y[-1, :] = 0.0
    return grad_x, grad_y


def _divergence(grad_x: np.ndarray, grad_y: np.ndarray) -> np.ndarray:
    div = grad_x - np.roll(grad_x, 1, axis=1) + grad_y - np.roll(grad_y, 1, axis=0)
    div[:, 0] = grad_x[:, 0]
    div[0, :] += grad_y[0, :]
    div[:, -1] -= grad_x[:, -1]
    div[-1, :] -= grad_y[-1, :]
    return div


def _weighted_kernel_estimate(
    latent: np.ndarray,
    observed: np.ndarray,
    kernel_size: tuple[int, int],
    epsilon: float,
    weight_strength: float,
) -> np.ndarray:
    grad_x, grad_y = _gradient(latent)
    magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    weights = np.exp(-(magnitude ** 2) * weight_strength)
    Fx = np.fft.fft2(latent * weights)
    Fg = np.fft.fft2(observed * weights)
    numerator = np.conj(Fx) * Fg
    denominator = np.maximum(np.abs(Fx) ** 2, epsilon)
    kernel_full = np.real(np.fft.ifft2(numerator / denominator))
    return _crop_and_project_kernel(kernel_full, kernel_size)


def _crop_and_project_kernel(kernel_full: np.ndarray, kernel_size: tuple[int, int]) -> np.ndarray:
    h, w = kernel_full.shape
    kh, kw = kernel_size
    center_y = h // 2
    center_x = w // 2
    start_y = center_y - kh // 2
    start_x = center_x - kw // 2
    cropped = kernel_full[start_y : start_y + kh, start_x : start_x + kw]
    cropped = np.clip(cropped, 0.0, None)
    total = float(cropped.sum())
    if total <= 0.0:
        cropped = np.zeros_like(cropped)
        cropped[kh // 2, kw // 2] = 1.0
    else:
        cropped /= total
    return cropped


def tv_admm_deblur(
    observed: np.ndarray,
    kernel: np.ndarray,
    lambda_tv: float,
    rho: float,
    admm_iters: int,
    epsilon: float,
) -> np.ndarray:
    shape = observed.shape
    K = psf2otf(kernel, shape)
    G = np.fft.fft2(observed)
    otf_dx = psf2otf(np.array([[1.0, -1.0]]), shape)
    otf_dy = psf2otf(np.array([[1.0], [-1.0]]), shape)

    denom_base = np.abs(K) ** 2 + rho * (np.abs(otf_dx) ** 2 + np.abs(otf_dy) ** 2) + epsilon

    x = observed.copy()
    z_x, z_y = _gradient(x)
    u_x = np.zeros_like(z_x)
    u_y = np.zeros_like(z_y)

    for _ in range(admm_iters):
        rhs = (
            np.conj(K) * G
            + rho * (
                np.conj(otf_dx) * np.fft.fft2(z_x - u_x)
                + np.conj(otf_dy) * np.fft.fft2(z_y - u_y)
            )
        )
        x = np.real(np.fft.ifft2(rhs / denom_base))
        grad_x, grad_y = _gradient(x)
        z_x = _soft_threshold(grad_x + u_x, lambda_tv / rho)
        z_y = _soft_threshold(grad_y + u_y, lambda_tv / rho)
        u_x += grad_x - z_x
        u_y += grad_y - z_y

    return np.clip(x, 0.0, 1.0)


def blind_deconvolution_admm(
    observed: np.ndarray,
    kernel_size: tuple[int, int],
    iterations: int,
    lambda_tv: float,
    rho: float,
    admm_iters: int,
    epsilon: float,
    weight_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    kh, kw = kernel_size
    kernel = np.zeros((kh, kw), dtype=np.float64)
    kernel[kh // 2, kw // 2] = 1.0
    latent = observed.copy()

    for _ in range(iterations):
        latent = tv_admm_deblur(observed, kernel, lambda_tv, rho, admm_iters, epsilon)
        kernel = _weighted_kernel_estimate(latent, observed, kernel_size, epsilon, weight_strength)

    return latent, kernel
