import numpy as np
from numpy.fft import fft2, ifft2
from scipy.fft import next_fast_len
from scipy.signal import convolve2d
from scipy.ndimage import zoom as ndimage_zoom
from scipy.ndimage import gaussian_filter

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:

    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape[:2]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    padded = np.roll(np.roll(padded, -(ph // 2), axis=0),
                     -(pw // 2), axis=1)
    return fft2(padded)

def otf2psf(otf: np.ndarray, psf_shape: tuple) -> np.ndarray:

    full = np.real(ifft2(otf))

    sh = psf_shape[0] // 2
    sw = psf_shape[1] // 2
    full = np.roll(np.roll(full, sh, axis=0), sw, axis=1)

    return full[:psf_shape[0], :psf_shape[1]]

def optimal_fft_shape(im_shape: tuple, ker_shape: tuple) -> tuple:

    h = next_fast_len(im_shape[0] + ker_shape[0])
    w = next_fast_len(im_shape[1] + ker_shape[1])
    return (h, w)

def pad_to_fft(image: np.ndarray, fft_shape: tuple,
               mode: str = 'edge') -> np.ndarray:

    pad_h = fft_shape[0] - image.shape[0]
    pad_w = fft_shape[1] - image.shape[1]
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if mode == 'edge':
        return np.pad(image, ((top, bottom), (left, right)), mode='edge')
    else:
        return np.pad(image, ((top, bottom), (left, right)),
                      mode='constant', constant_values=0)

def crop_center(array: np.ndarray, target_shape: tuple) -> np.ndarray:

    h, w = target_shape[:2]
    H, W = array.shape[:2]
    top = (H - h) // 2
    left = (W - w) // 2
    return array[top:top + h, left:left + w].copy()

def convolve_fft(image: np.ndarray, kernel: np.ndarray,
                 padding: str = 'edge') -> np.ndarray:

    fft_shape = optimal_fft_shape(image.shape, kernel.shape)
    im_padded = pad_to_fft(image, fft_shape, mode=padding)
    im_fft = fft2(im_padded)
    ker_otf = psf2otf(kernel, fft_shape)

    result = np.real(ifft2(im_fft * ker_otf))
    return crop_center(result, image.shape)

def normalize_kernel(kernel: np.ndarray) -> np.ndarray:

    k = np.copy(kernel)
    k[k < 0] = 0.0
    s = k.sum()
    if s > 0:
        k /= s
    return k

def create_delta_kernel(shape: tuple) -> np.ndarray:

    k = np.zeros(shape, dtype=np.float64)
    cy, cx = shape[0] // 2, shape[1] // 2
    k[cy, cx] = 1.0
    return k

def resize_kernel(kernel: np.ndarray, factor: float) -> np.ndarray:

    if factor == 1.0:
        return normalize_kernel(kernel)
    resized = ndimage_zoom(kernel, factor, order=1)
    return normalize_kernel(resized)

def sobel_h() -> np.ndarray:

    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

def sobel_v() -> np.ndarray:

    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)

_LAPLACIAN_KERNEL_3 = np.array([[ 2,  0,  2],
                                [ 0, -8,  0],
                                [ 2,  0,  2]], dtype=np.float64)

def compute_laplacian_abs(image: np.ndarray) -> np.ndarray:

    lap = convolve2d(image, _LAPLACIAN_KERNEL_3, mode='same', boundary='symm')

    result = np.abs(lap)
    result = np.clip(result, 0, 255).astype(np.uint8).astype(np.float64)
    return result

def kmeans_quantize(image: np.ndarray,
                    n_clusters: int = 15,
                    max_iter: int = 10) -> tuple:

    from scipy.cluster.vq import kmeans2

    h, w = image.shape
    pixels = image.ravel().astype(np.float64)

    data = pixels.reshape(-1, 1)

    centres, labels = kmeans2(data, n_clusters, minit='points',
                              iter=max_iter, seed=42)
    centres = centres.ravel()

    unique_labels = np.unique(labels)
    if len(unique_labels) < n_clusters:
        old_to_new = np.zeros(n_clusters, dtype=np.int32)
        for new_idx, old_idx in enumerate(unique_labels):
            old_to_new[old_idx] = new_idx
        labels = old_to_new[labels]
        centres = centres[unique_labels]
        n_clusters = len(unique_labels)

    order = np.argsort(centres)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_clusters)
    centres = centres[order]
    labels = inv_order[labels]

    return labels.reshape(h, w).astype(np.int32), centres.astype(np.float64)

def resize_image(image: np.ndarray, factor: float) -> np.ndarray:

    return ndimage_zoom(image, factor, order=1)

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:

    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))

def compute_psnr(img1: np.ndarray, img2: np.ndarray,
                 max_val: float = 255.0) -> float:

    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))

def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                 max_val: float = 255.0) -> float:

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)

    _trunc = 5.0 / 1.5

    mu1 = gaussian_filter(I1, sigma=1.5, truncate=_trunc)
    mu2 = gaussian_filter(I2, sigma=1.5, truncate=_trunc)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(I1 * I1, sigma=1.5, truncate=_trunc) - mu1_sq
    sigma2_sq = gaussian_filter(I2 * I2, sigma=1.5, truncate=_trunc) - mu2_sq
    sigma12 = gaussian_filter(I1 * I2, sigma=1.5, truncate=_trunc) - mu1_mu2

    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))
