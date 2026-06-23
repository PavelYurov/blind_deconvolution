import numpy as np
import cv2
from numpy.fft import fftshift, ifftshift

def dark_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:

    if image.ndim == 2:
        min_channel = image
    else:
        min_channel = np.min(image, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (window_size, window_size)
    )
    dark = cv2.erode(min_channel, kernel)
    return dark.astype(np.float32)

def bright_channel(image: np.ndarray, window_size: int = 15) -> np.ndarray:

    if image.ndim == 2:
        max_channel = image
    else:
        max_channel = np.max(image, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (window_size, window_size)
    )
    bright = cv2.dilate(max_channel, kernel)
    return bright.astype(np.float32)

def dcpl0norm(dark: np.ndarray) -> int:

    return int(np.count_nonzero(dark))

def bcpl0norm(bright: np.ndarray) -> int:

    return int(np.count_nonzero(bright))

def gradient_h(I: np.ndarray) -> np.ndarray:

    if I.ndim == 2:
        return np.pad(I[:, 1:] - I[:, :-1], ((0, 0), (0, 1)))
    else:
        return np.pad(I[:, 1:, :] - I[:, :-1, :], ((0, 0), (0, 1), (0, 0)))

def gradient_v(I: np.ndarray) -> np.ndarray:

    if I.ndim == 2:
        return np.pad(I[1:, :] - I[:-1, :], ((0, 1), (0, 0)))
    else:
        return np.pad(I[1:, :, :] - I[:-1, :, :], ((0, 1), (0, 0), (0, 0)))

def compute_gradients(img: np.ndarray):

    gh = gradient_h(img)
    gv = gradient_v(img)
    return gh, gv

def gradient_mag_sq(grad):

    gh, gv = grad
    return gh ** 2 + gv ** 2

def gaussian_pyramid(img: np.ndarray, num_levels: int) -> list:

    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")

    pyr = [img.copy()]
    for _ in range(1, num_levels):
        prev = pyr[-1]
        if prev.shape[0] < 2 or prev.shape[1] < 2:
            break
        down = cv2.pyrDown(prev)
        pyr.append(down)

    return pyr[::-1]

def upsample_kernel(k: np.ndarray, target_hw: tuple) -> np.ndarray:

    th, tw = target_hw
    resized = cv2.resize(k, (tw, th), interpolation=cv2.INTER_LINEAR)
    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized /= s
    else:
        resized = np.zeros((th, tw), dtype=np.float32)
        resized[th // 2, tw // 2] = 1.0
    return resized

def upsample_small_kernel(
    k_small: np.ndarray,
    scale_factor: float = 2.0,
    max_size: tuple = None,
) -> np.ndarray:

    kh, kw = k_small.shape
    new_kh = int(kh * scale_factor)
    new_kw = int(kw * scale_factor)

    if max_size is not None:
        mh, mw = max_size
        new_kh = min(new_kh, mh)
        new_kw = min(new_kw, mw)

    new_kh = max(1, new_kh)
    new_kw = max(1, new_kw)

    resized = cv2.resize(
        k_small, (new_kw, new_kh), interpolation=cv2.INTER_LINEAR
    )
    resized = np.clip(resized, 0, None)
    s = resized.sum()
    if s > 1e-12:
        resized = resized / s
    else:
        resized = np.zeros((new_kh, new_kw), dtype=np.float32)
        resized[new_kh // 2, new_kw // 2] = 1.0
    return resized

def upsample_l(l: np.ndarray, target_shape: tuple) -> np.ndarray:

    target_h, target_w = target_shape
    return cv2.resize(l, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

def kernel_to_fft_size(
    k_small: np.ndarray, image_shape: tuple
) -> np.ndarray:

    H, W = image_shape
    out = np.zeros((H, W), dtype=k_small.dtype)
    kh, kw = k_small.shape
    out[:kh, :kw] = k_small
    return out

def threshold_dark_channel(
    l: np.ndarray,
    D: np.ndarray,
    w_k: float,
    xi: float,
) -> np.ndarray:

    threshold = w_k / xi
    mask = D * D > threshold

    p = l.copy()
    if l.ndim == 3:
        mask_3d = mask[:, :, np.newaxis]
        p[~np.broadcast_to(mask_3d, l.shape)] = 0.0
    else:
        p[~mask] = 0.0
    return p

def threshold_gradient(g, theta: float, lam: float):

    gh, gv = g
    mag_sq = gh * gh + gv * gv
    T = theta / (lam + 1e-8)

    mask = mag_sq > T
    return gh * mask, gv * mask

def normalise_kernel(k: np.ndarray) -> np.ndarray:

    s = k.sum()
    if s > 1e-8:
        return k / s
    return k

def clamp_kernel(k: np.ndarray) -> np.ndarray:

    return np.clip(k, 0, None)

def crop_kernel(k: np.ndarray) -> np.ndarray:

    nz = np.nonzero(k)
    if len(nz[0]) == 0:
        return k

    y_min, y_max = nz[0].min(), nz[0].max()
    x_min, x_max = nz[1].min(), nz[1].max()
    return k[y_min : y_max + 1, x_min : x_max + 1]

def resize_kernel(k: np.ndarray, target_shape: tuple) -> np.ndarray:

    resized = cv2.resize(
        k, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    resized = np.clip(resized, 0, None)
    return normalise_kernel(resized)

def clean_kernel(k: np.ndarray) -> np.ndarray:

    k = np.clip(k, 0, None)

    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    k = fftshift(k)

    k = crop_kernel(k)

    s = k.sum()
    if s > 1e-12:
        k = k / s
    else:
        k = np.zeros_like(k)
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    return k

def pad_and_ifftshift_kernel(
    k_small: np.ndarray, image_shape: tuple
) -> np.ndarray:

    H, W = image_shape
    padded = np.zeros((H, W), dtype=np.float32)
    kh, kw = k_small.shape
    padded[:kh, :kw] = k_small
    return ifftshift(padded)

def postprocess_kernel_spatial(k_full: np.ndarray) -> np.ndarray:

    k = fftshift(k_full.real)
    k = np.clip(k, 0, None)

    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    nz = np.nonzero(k)
    if len(nz[0]) == 0:
        return k

    y0, y1 = nz[0].min(), nz[0].max()
    x0, x1 = nz[1].min(), nz[1].max()
    k_cropped = k[y0 : y1 + 1, x0 : x1 + 1]

    s = k_cropped.sum()
    if s > 1e-12:
        k_cropped = k_cropped / s
    return k_cropped

def pad_kernel_centered(k: np.ndarray, out_shape: tuple) -> np.ndarray:

    H2, W2 = out_shape
    kh, kw = k.shape

    kpad = np.zeros((H2, W2), dtype=np.float32)
    cx = (H2 - kh) // 2
    cy = (W2 - kw) // 2
    kpad[cx : cx + kh, cy : cy + kw] = k

    kpad = np.fft.ifftshift(kpad)
    return kpad

def extract_kernel_center(
    k_full: np.ndarray, expected_size: tuple = None
) -> np.ndarray:

    k = np.real(fftshift(k_full))
    k = np.clip(k, 0, None)

    thr = max(1e-8, 1e-3 * k.max())
    k[k < thr] = 0.0

    H, W = k.shape

    if expected_size is not None:
        kh, kw = expected_size
        cy, cx = H // 2, W // 2
        y0 = cy - kh // 2
        x0 = cx - kw // 2
        cropped = k[y0 : y0 + kh, x0 : x0 + kw]
    else:
        nz = np.nonzero(k)
        if len(nz[0]) == 0:
            return np.array([[1.0]], dtype=np.float32)
        y0, y1 = nz[0].min(), nz[0].max()
        x0, x1 = nz[1].min(), nz[1].max()
        cropped = k[y0 : y1 + 1, x0 : x1 + 1]

    s = cropped.sum()
    if s <= 1e-12:
        out = np.zeros_like(cropped, dtype=np.float32)
        out[out.shape[0] // 2, out.shape[1] // 2] = 1.0
        return out

    return (cropped / s).astype(np.float32)

def make_delta_kernel(kernel_size) -> np.ndarray:

    if isinstance(kernel_size, tuple):
        kh, kw = kernel_size
    else:
        kh = kw = int(kernel_size)

    k = np.zeros((kh, kw), dtype=np.float32)
    k[kh // 2, kw // 2] = 1.0
    return k
