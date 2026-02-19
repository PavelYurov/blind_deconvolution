import numpy as np
from numpy.fft import fft2, ifft2

# ... (Оставляем функции fft_convolve, fft_correlate без изменений) ...
def fft_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular 2-D convolution via FFT."""
    H, W = image.shape
    kh, kw = kernel.shape
    h_pad = np.zeros((H, W), dtype=np.float64)
    h_pad[:kh, :kw] = kernel
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)
    return np.real(ifft2(fft2(image) * fft2(h_pad)))

def fft_correlate(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Circular 2-D cross-correlation via FFT."""
    H, W = image.shape
    kh, kw = kernel.shape
    h_pad = np.zeros((H, W), dtype=np.float64)
    h_pad[:kh, :kw] = kernel
    h_pad = np.roll(h_pad, -(kh // 2), axis=0)
    h_pad = np.roll(h_pad, -(kw // 2), axis=1)
    return np.real(ifft2(fft2(image) * np.conj(fft2(h_pad))))

# ═══════════════════════════════════════════════════════════════════════
#  Kernel Constraint Operators
# ═══════════════════════════════════════════════════════════════════════

def project_kernel(h: np.ndarray) -> np.ndarray:
    """Euclidean projection onto simplex (Sum=1, Non-negative)."""
    h = np.maximum(h, 0.0)
    total = h.sum()
    if total > 1e-15:
        h /= total
    return h

def threshold_kernel(h: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Soft Thresholding Operator (Proximal operator for L1 norm).
    Mathematically: S_lambda(x) = sign(x) * max(|x| - lambda, 0)
    Since h >= 0, this simplifies to max(h - threshold, 0).
    
    This promotes sparsity much more stably than hard cutoff.
    """
    if threshold <= 0:
        return h
    
    # Apply soft thresholding
    h = np.maximum(h - threshold, 0.0)
    
    # Re-normalize immediately
    return project_kernel(h)

def center_kernel(h: np.ndarray) -> np.ndarray:
    """Center of mass alignment."""
    kh, kw = h.shape
    total = h.sum()
    if total < 1e-15:
        return h

    yy, xx = np.mgrid[:kh, :kw]
    cy = (yy * h).sum() / total
    cx = (xx * h).sum() / total

    shift_y = int(np.round(kh / 2.0 - cy))
    shift_x = int(np.round(kw / 2.0 - cx))

    return np.roll(np.roll(h, shift_y, axis=0), shift_x, axis=1)

# ... (Оставляем tv_gradient, edge_taper, и функции пирамид без изменений из моего предыдущего ответа) ...
# Убедитесь, что edge_taper используется тот, который я прислал в прошлом сообщении (с blending, а не windowing)

def tv_gradient(x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(x, -1, axis=0) - x
    mag = np.sqrt(dx ** 2 + dy ** 2 + epsilon ** 2)
    nx = dx / mag
    ny = dy / mag
    div_x = nx - np.roll(nx, 1, axis=1)
    div_y = ny - np.roll(ny, 1, axis=0)
    return -(div_x + div_y)

def edge_taper(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    kh, kw = kernel_shape
    H, W = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    def _get_taper_1d(size, pad):
        t = np.ones(size)
        if pad > 0:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(pad) / pad))
            t[:pad] = ramp
            t[-pad:] = ramp[::-1]
        return t

    wy = _get_taper_1d(H, pad_h).reshape(-1, 1)
    wx = _get_taper_1d(W, pad_w).reshape(1, -1)
    mask = wy * wx
    mean_val = np.mean(image)
    return image * mask + mean_val * (1.0 - mask)

def downsample(image: np.ndarray, factor: int = 2) -> np.ndarray:
    H, W = image.shape
    nH = (H // factor) * factor
    nW = (W // factor) * factor
    cropped = image[:nH, :nW]
    return cropped.reshape(nH // factor, factor, nW // factor, factor).mean(axis=(1, 3))

def resize_image(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    h_in, w_in = image.shape
    H, W = target_shape
    row_idx = np.linspace(0, h_in - 1, H)
    col_idx = np.linspace(0, w_in - 1, W)
    r0 = np.floor(row_idx).astype(int)
    r1 = np.minimum(r0 + 1, h_in - 1)
    c0 = np.floor(col_idx).astype(int)
    c1 = np.minimum(c0 + 1, w_in - 1)
    dr = (row_idx - r0)[:, np.newaxis]
    dc = (col_idx - c0)[np.newaxis, :]
    return (image[np.ix_(r0, c0)] * (1 - dr) * (1 - dc) +
            image[np.ix_(r0, c1)] * (1 - dr) * dc +
            image[np.ix_(r1, c0)] * dr * (1 - dc) +
            image[np.ix_(r1, c1)] * dr * dc)

def upsample_kernel(h: np.ndarray, target_shape: tuple) -> np.ndarray:
    return project_kernel(resize_image(h, target_shape))

def build_pyramid(y: np.ndarray, num_scales: int = 4, min_size: int = 32) -> list:
    levels = [y]
    for _ in range(num_scales - 1):
        prev = levels[-1]
        if prev.shape[0] < 2 * min_size or prev.shape[1] < 2 * min_size:
            break
        levels.append(downsample(prev, factor=2))
    levels.reverse()
    return levels

def kernel_shape_for_level(kernel_shape: tuple, level: int, num_levels: int) -> tuple:
    ratio = 2.0 ** (num_levels - 1 - level)
    kh = max(3, int(np.round(kernel_shape[0] / ratio)))
    kw = max(3, int(np.round(kernel_shape[1] / ratio)))
    kh = kh if kh % 2 == 1 else kh + 1
    kw = kw if kw % 2 == 1 else kw + 1
    return (kh, kw)

def init_gaussian_kernel(shape: tuple, sigma: float = None) -> np.ndarray:
    kh, kw = shape
    if sigma is None:
        sigma = max(kh, kw) / 6.0 # Slightly wider init helps
    cy, cx = kh // 2, kw // 2
    yy, xx = np.mgrid[:kh, :kw]
    h = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    h /= h.sum()
    return h