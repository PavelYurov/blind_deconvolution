"""
utils.py
Вспомогательные функции.
Добавлен Tikhonov filter.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    in_shape = psf.shape
    padded_psf = np.zeros(shape, dtype=psf.dtype)
    center_y, center_x = in_shape[0] // 2, in_shape[1] // 2
    padded_psf[:in_shape[0], :in_shape[1]] = psf
    padded_psf = np.roll(padded_psf, -center_y, axis=0)
    padded_psf = np.roll(padded_psf, -center_x, axis=1)
    return np.fft.fft2(padded_psf)

def get_gradient_operators(shape: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    ky = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    OTF_dx = psf2otf(kx, shape)
    OTF_dy = psf2otf(ky, shape)
    OTF_dx_conj = np.conj(OTF_dx)
    OTF_dy_conj = np.conj(OTF_dy)
    return OTF_dx, OTF_dy, OTF_dx_conj, OTF_dy_conj

def compute_curvature(u: np.ndarray) -> np.ndarray:
    dx_u = np.roll(u, -1, axis=1) - u
    dy_u = np.roll(u, -1, axis=0) - u
    norm_grad = np.sqrt(dx_u**2 + dy_u**2 + 1e-12)
    nx = dx_u / norm_grad
    ny = dy_u / norm_grad
    div_x = nx - np.roll(nx, 1, axis=1)
    div_y = ny - np.roll(ny, 1, axis=0)
    return div_x + div_y

def soft_threshold(x: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def edgetaper(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    ks = max(kernel_shape)
    sigma = ks / 2.0
    h, w = img.shape
    alpha = np.zeros((h, w))
    border = min(ks, min(h, w) // 4)
    alpha[border:-border, border:-border] = 1.0
    window = gaussian_filter(alpha, sigma=sigma/2)
    window = np.clip(window, 0, 1)
    blurred_img = gaussian_filter(img, sigma=sigma)
    return img * window + blurred_img * (1.0 - window)

def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    pad_h = kernel_shape[0] // 2 + 1
    pad_w = kernel_shape[1] // 2 + 1
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')

def crop_image(img: np.ndarray, original_shape: tuple, kernel_shape: tuple) -> np.ndarray:
    pad_h = kernel_shape[0] // 2 + 1
    pad_w = kernel_shape[1] // 2 + 1
    h, w = original_shape
    return img[pad_h : pad_h + h, pad_w : pad_w + w]

def wiener_filter(img: np.ndarray, kernel: np.ndarray, noise_snr: float = 0.01) -> np.ndarray:
    """Фильтр Винера."""
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    otf_conj = np.conj(otf)
    numerator = otf_conj
    denominator = np.abs(otf)**2 + noise_snr
    F_img = np.fft.fft2(img)
    F_res = (numerator / denominator) * F_img
    return np.real(np.fft.ifft2(F_res))

def tikhonov_filter(img: np.ndarray, kernel: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Неслепая деконволюция с регуляризацией Тихонова (1-го порядка).
    Min ||k*u - f||^2 + alpha * ||grad(u)||^2
    """
    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    otf_conj = np.conj(otf)
    
    # Получаем операторы градиента для регуляризации
    OTF_dx, OTF_dy, _, _ = get_gradient_operators((H, W))
    
    # |Dx|^2 + |Dy|^2 - лапласиан в частотной области (штраф за шероховатость)
    reg_term = np.abs(OTF_dx)**2 + np.abs(OTF_dy)**2
    
    numerator = otf_conj
    # Знаменатель: отклик ядра + альфа * штраф градиентов
    denominator = np.abs(otf)**2 + alpha * reg_term
    
    F_img = np.fft.fft2(img)
    F_res = (numerator / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))