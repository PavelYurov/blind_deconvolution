import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from typing import Tuple, List, Optional

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    padded = np.zeros(shape)
    padded[:psf.shape[0], :psf.shape[1]] = psf

    shift_y = -(psf.shape[0] // 2)
    shift_x = -(psf.shape[1] // 2)
    padded = np.roll(np.roll(padded, shift_y, axis=0), shift_x, axis=1)

    return np.fft.fft2(padded)

def convolve2d(image: np.ndarray, kernel: np.ndarray,
               mode: str = 'same') -> np.ndarray:
    return fftconvolve(image, kernel, mode=mode)

def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = np.array([[1.0, -1.0],
                   [0.0,  0.0]])
    dy = np.array([[ 1.0, 0.0],
                   [-1.0, 0.0]])

    grad_x = fftconvolve(image, dx, mode='valid')
    grad_y = fftconvolve(image, dy, mode='valid')

    return grad_x, grad_y

def build_scale_pyramid(kernel_size: int) -> List[int]:
    assert kernel_size >= 3 and kernel_size % 2 == 1,\
        "kernel_size must be odd and >= 3"

    min_scale = max(2 * ((kernel_size - 1) // 32) + 1, 3)

    scales: List[int] = []
    layer = min_scale
    step = np.sqrt(2.0)

    while layer < kernel_size:
        if layer % 2 == 0:
            layer += 1
        scales.append(int(layer))
        layer = int(np.floor(layer * step))
        if layer % 2 == 0:
            layer += 1

    scales.append(kernel_size)
    return scales

def center_kernel(
    kernel: np.ndarray,
    images: Optional[Tuple[np.ndarray, ...]] = None
) -> Tuple:
    kh, kw = kernel.shape
    total = kernel.sum()

    if total < 1e-10:
        if images is not None:
            return (kernel,) + images
        return kernel

    mu_y = np.sum(np.arange(kh) * kernel.sum(axis=1)) / total
    mu_x = np.sum(np.arange(kw) * kernel.sum(axis=0)) / total

    offset_y = int(np.round(kh // 2 - mu_y))
    offset_x = int(np.round(kw // 2 - mu_x))

    if offset_y == 0 and offset_x == 0:
        if images is not None:
            return (kernel,) + images
        return kernel

    shift_h = 2 * abs(offset_y) + 1
    shift_w = 2 * abs(offset_x) + 1
    shift_kern = np.zeros((shift_h, shift_w))
    shift_kern[abs(offset_y) + offset_y,
               abs(offset_x) + offset_x] = 1.0

    kernel_centered = fftconvolve(kernel, shift_kern, mode='same')

    if images is not None:
        inv_kern = shift_kern[::-1, ::-1]
        shifted = tuple(
            fftconvolve(img, inv_kern, mode='same') for img in images
        )
        return (kernel_centered,) + shifted

    return kernel_centered

def normalize_kernel(kernel: np.ndarray,
                     threshold: float = 0.0) -> np.ndarray:
    kernel = np.clip(kernel, 0, None)

    if threshold > 0 and kernel.max() > 0:
        kernel[kernel < kernel.max() * threshold] = 0.0

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel

def edgetaper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    blurred = fftconvolve(image, kernel, mode='same')

    kh, kw = kernel.shape
    H, W = image.shape

    wy = np.ones(H)
    wx = np.ones(W)

    border_y = max(kh // 2, 1)
    border_x = max(kw // 2, 1)

    ramp_y = np.linspace(0, 1, border_y, endpoint=False)
    ramp_x = np.linspace(0, 1, border_x, endpoint=False)

    wy[:border_y] = ramp_y
    wy[-border_y:] = ramp_y[::-1]
    wx[:border_x] = ramp_x
    wx[-border_x:] = ramp_x[::-1]

    weight = wy[:, None] * wx[None, :]

    return weight * image + (1.0 - weight) * blurred

def resize_image(image: np.ndarray,
                 target_shape: Tuple[int, int]) -> np.ndarray:
    h_in, w_in = image.shape[:2]
    h_out, w_out = target_shape

    if h_in == h_out and w_in == w_out:
        return image.copy()

    factors = (h_out / h_in, w_out / w_in)
    return zoom(image, factors, order=1)

def rgb_to_ycbcr(image: np.ndarray) -> np.ndarray:
    M = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.500   ],
        [ 0.500,    -0.418688, -0.081312]
    ])
    ycbcr = image @ M.T
    ycbcr[:, :, 1:] += 0.5
    return ycbcr

def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    ycbcr = ycbcr.copy()
    ycbcr[:, :, 1:] -= 0.5

    M_inv = np.array([
        [1.0,  0.0,       1.402   ],
        [1.0, -0.344136, -0.714136],
        [1.0,  1.772,     0.0     ]
    ])
    return ycbcr @ M_inv.T
