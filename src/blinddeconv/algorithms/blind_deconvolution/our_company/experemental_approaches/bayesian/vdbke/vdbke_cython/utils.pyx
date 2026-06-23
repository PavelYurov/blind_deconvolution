import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, map_coordinates, gaussian_filter

cimport numpy as cnp
cimport cython

cnp.import_array()

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)

def fspecial_gaussian(hsize, sigma: float) -> np.ndarray:
    if np.isscalar(hsize):
        m, n = int(hsize), int(hsize)
    else:
        m, n = int(hsize[0]), int(hsize[1])

    sy = (m - 1) / 2.0
    sx = (n - 1) / 2.0
    y = np.arange(m, dtype=np.float64) - sy
    x = np.arange(n, dtype=np.float64) - sx
    X, Y = np.meshgrid(x, y)

    arg = -(X * X + Y * Y) / (2.0 * sigma * sigma)
    h = np.exp(arg)

    h[h < np.finfo(np.float64).eps * h.max()] = 0.0

    s = h.sum()
    if s != 0:
        h /= s
    return h

def imresize(img: np.ndarray, output_size, method: str = 'bilinear') -> np.ndarray:
    oh, ow = int(output_size[0]), int(output_size[1])
    h, w = img.shape[0], img.shape[1]

    if h == oh and w == ow:
        return img.copy()

    order = 1 if method == 'bilinear' else 3
    zoom_h = oh / h
    zoom_w = ow / w

    # Anti-aliasing prefilter for downscaling (matches MATLAB imresize Antialiasing=true).
    # sigma = (1/scale - 1)/pi  (gentle, matches the tent-kernel cut-off).
    src = img
    if zoom_h < 1.0 or zoom_w < 1.0:
        sigma_h = max(0.0, (1.0 / zoom_h - 1.0) / np.pi) if zoom_h < 1.0 else 0.0
        sigma_w = max(0.0, (1.0 / zoom_w - 1.0) / np.pi) if zoom_w < 1.0 else 0.0
        if sigma_h > 1e-3 or sigma_w > 1e-3:
            if img.ndim == 3:
                src = gaussian_filter(img, sigma=(sigma_h, sigma_w, 0.0), mode='nearest')
            else:
                src = gaussian_filter(img, sigma=(sigma_h, sigma_w), mode='nearest')

    if src.ndim == 3:
        result = zoom(src, (zoom_h, zoom_w, 1), order=order)
    else:
        result = zoom(src, (zoom_h, zoom_w), order=order)

    if result.shape[0] > oh:
        result = result[:oh]
    if result.shape[1] > ow:
        result = result[:, :ow]
    if result.shape[0] < oh or result.shape[1] < ow:
        pad_r = oh - result.shape[0]
        pad_c = ow - result.shape[1]
        if result.ndim == 3:
            result = np.pad(result, ((0, max(0, pad_r)), (0, max(0, pad_c)), (0, 0)),
                            mode='edge')
        else:
            result = np.pad(result, ((0, max(0, pad_r)), (0, max(0, pad_c))),
                            mode='edge')
    return result

def valid_conv_by_fft(cnp.ndarray[cnp.complex128_t, ndim=2] X_fft, cnp.ndarray[cnp.float64_t, ndim=2] h):
    cdef int M1 = X_fft.shape[0]
    cdef int M2 = X_fft.shape[1]
    cdef int s1 = h.shape[0]
    cdef int s2 = h.shape[1]

    cdef cnp.ndarray[cnp.float64_t, ndim=2] h_padded = np.zeros((M1, M2), dtype=np.float64)
    h_padded[:s1, :s2] = h
    H = fft2(h_padded)

    temp = np.real(ifft2(X_fft * H))
    return temp[s1 - 1:, s2 - 1:]

def rgb2gray(img: np.ndarray) -> np.ndarray:
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]

def rgb2ycbcr(img: np.ndarray) -> np.ndarray:
    T = np.array([[ 65.481, 128.553,  24.966],[-37.797, -74.203, 112.0  ],
        [112.0,   -93.786, -18.214]
    ], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)

    H, W = img.shape[:2]
    rgb_flat = img.reshape(-1, 3)
    ycbcr_flat = rgb_flat @ T.T / 255.0 + offset / 255.0
    return ycbcr_flat.reshape(H, W, 3)

def ycbcr2rgb(img: np.ndarray) -> np.ndarray:
    T = np.array([[ 65.481, 128.553,  24.966],[-37.797, -74.203, 112.0  ],[112.0,   -93.786, -18.214]
    ], dtype=np.float64)
    offset = np.array([16.0, 128.0, 128.0], dtype=np.float64)

    invT = np.linalg.inv(T)
    H, W = img.shape[:2]
    ycbcr_flat = img.reshape(-1, 3)
    rgb_flat = (ycbcr_flat - offset / 255.0) * 255.0 @ invT.T
    return np.clip(rgb_flat.reshape(H, W, 3), 0.0, 1.0)

def comp_upto_shift(I1: np.ndarray, I2: np.ndarray):
    maxshift = 5
    shifts = np.arange(-5.0, 5.0 + 0.25, 0.25)
    
    I2c = I2[15:-15, 15:-15].copy()
    I1c = I1[10:-10, 10:-10].copy()

    N1, N2 = I2c.shape
    ns = len(shifts)
    ssdem = np.full((ns, ns), np.inf, dtype=np.float64)

    base_r = np.arange(N1, dtype=np.float64) + maxshift
    base_c = np.arange(N2, dtype=np.float64) + maxshift

    for i in range(ns):
        for j in range(ns):
            coords_r = base_r + shifts[j]
            coords_c = base_c + shifts[i]
            rr, cc = np.meshgrid(coords_r, coords_c, indexing='ij')
            tI1 = map_coordinates(I1c, [rr, cc], order=1, mode='constant', cval=0.0)
            ssdem[i, j] = np.sum((tI1 - I2c) ** 2)

    ssde = ssdem.min()
    idx = np.unravel_index(ssdem.argmin(), ssdem.shape)
    best_i, best_j = idx

    coords_r = base_r + shifts[best_j]
    coords_c = base_c + shifts[best_i]
    rr, cc = np.meshgrid(coords_r, coords_c, indexing='ij')
    tI1 = map_coordinates(I1c,[rr, cc], order=1, mode='constant', cval=0.0)

    return ssde, tI1

def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** gamma

def tikhonov_filter(image: np.ndarray, kernel: np.ndarray, alpha: float = 0.001) -> np.ndarray:
    H, W = image.shape
    K = psf2otf(kernel, (H, W))
    Kt = np.conj(K)
    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    Dx = psf2otf(dx, (H, W))
    Dy = psf2otf(dy, (H, W))
    L = np.abs(Dx) ** 2 + np.abs(Dy) ** 2

    Y = fft2(image)
    X = (Kt * Y) / (Kt * K + alpha * L + 1e-10)
    return np.real(ifft2(X))

def wiener_filter(image: np.ndarray, kernel: np.ndarray, noise_snr: float = 0.001) -> np.ndarray:
    H, W = image.shape
    K = psf2otf(kernel, (H, W))
    Kt = np.conj(K)
    Y = fft2(image)
    X = (Kt * Y) / (np.abs(K) ** 2 + noise_snr + 1e-10)
    return np.real(ifft2(X))

def pad_image(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    ph = kernel_shape[0] // 2
    pw = kernel_shape[1] // 2
    return np.pad(image, ((ph, ph), (pw, pw)), mode='edge')

def crop_image(image: np.ndarray, orig_shape: tuple, kernel_shape: tuple) -> np.ndarray:
    ph = kernel_shape[0] // 2
    pw = kernel_shape[1] // 2
    return image[ph:ph + orig_shape[0], pw:pw + orig_shape[1]]

def edgetaper(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve
    H, W = image.shape
    k = kernel.copy()
    acf = fftconvolve(k, k[::-1, ::-1], mode='full')
    acf /= acf.max()
    ah, aw = acf.shape

    row_profile = acf[ah // 2, :]
    col_profile = acf[:, aw // 2]
    row_w = np.ones(W, dtype=np.float64)
    col_w = np.ones(H, dtype=np.float64)

    hw = min(len(row_profile) // 2, W // 2)
    hh = min(len(col_profile) // 2, H // 2)

    rp_half = row_profile[len(row_profile) // 2:][:hw]
    cp_half = col_profile[len(col_profile) // 2:][:hh]

    for i in range(hw):
        v = rp_half[i]
        row_w[i] = min(row_w[i], v)
        row_w[W - 1 - i] = min(row_w[W - 1 - i], v)
    for i in range(hh):
        v = cp_half[i]
        col_w[i] = min(col_w[i], v)
        col_w[H - 1 - i] = min(col_w[H - 1 - i], v)

    weight = col_w[:, None] * row_w[None, :]

    blurred = fftconvolve(image, kernel, mode='same')
    return weight * image + (1 - weight) * blurred