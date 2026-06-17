"""
utils.py

Utility functions for MRF-based Blind Image Deconvolution.

Ported from C++ (OpenCV) code based on:
    N. Komodakis, N. Paragios: "MRF-based Blind Image Deconvolution",
    Proceedings of the 11th Asian Conference on Computer Vision (ACCV),
    Vol. 3, pp. 361-374, 2012.

C++ (OpenCV) → Python (NumPy/SciPy) conversion notes:
─────────────────────────────────────────────────────────────────────────
    cv::dft(input, output, 0):
        OpenCV DFT returns a 2-channel real Mat (Re, Im planes).
        → np.fft.fft2(input) returns a native complex128 ndarray.
        No manual 2-channel split / merge is needed.

    cv::idft(input, output, DFT_INVERSE | DFT_SCALE | DFT_REAL_OUTPUT):
        → np.fft.ifft2(input).real
        NumPy ifft2 always applies 1/N normalisation (SCALE is implicit).

    cv::mulSpectrums(A, B, result, 0, false):
        Element-wise complex multiplication: result = A ⊙ B.
        → A * B  (NumPy complex arrays).

    cv::mulSpectrums(A, B, result, 0, true):
        Conjugate multiply: result = A ⊙ conj(B).
        → A * np.conj(B).

    cv::getOptimalDFTSize(n):
        Smallest N ≥ n that factors into 2^a · 3^b · 5^c.
        → scipy.fft.next_fast_len(n)
        SciPy returns sizes efficient for its FFT back-end; both give
        O(N log N) transforms.

    FTMat class (toVector / toMatrix / DFT / iDFT):
        The C++ FTMat class converts 2-D images ↔ 1-D vectors with
        centre-origin indexing, zero-pads, and manages DFT round-trips.
        In Python, *all of this* is replaced by:
        • np.fft.fft2 with the *s* parameter for zero-padding, or
        • explicit np.pad + np.fft.fft2
        • psf2otf() for kernels / small filters.

    FTMat.toVector(ZeroPoint=1, IndexOrder=1, PaddingWay=0):
        Centre-origin, reverse-raster, zero-pad.
        Used for kernels and "filter" signals in the C++ code.
        Equivalent to psf2otf().

    FTMat.toVector(ZeroPoint=1, IndexOrder=0, PaddingWay=1):
        Centre-origin, normal raster, edge-replicate.
        Used for images.
        Equivalent to np.pad(image, ..., mode='edge') + np.fft.fft2.

    cv::Laplacian(src, dst, CV_64F, ksize=3):
        Internally: Sobel(dx=2,dy=0,ksize=3) + Sobel(dx=0,dy=2,ksize=3).
        The combined 3×3 kernel is [[2,0,2],[0,-8,0],[2,0,2]].
        → scipy.signal.convolve2d with the same kernel.

    cv::convertScaleAbs(src, dst, alpha=1, beta=0):
        dst = |src * alpha + beta|, saturated to uint8.
        → np.abs(src)

    cv::resize(src, dst, Size(), fx, fy, INTER_LINEAR):
        → scipy.ndimage.zoom(src, factor, order=1)

    cv::copyMakeBorder(src, dst, top, bot, left, right, BORDER_CONSTANT, 0):
        → np.pad(src, ((top, bot), (left, right)), 'constant',
                  constant_values=0)
    cv::copyMakeBorder(... , BORDER_REPLICATE):
        → np.pad(src, ..., mode='edge')

    cv::kmeans(data, K, labels, criteria, attempts, flags, centres):
        → scipy.cluster.vq.kmeans2

    cv::normalize(src, dst, 0, 255, NORM_MINMAX):
        → (src - src.min()) / (src.max() - src.min()) * 255

    cv::norm(A, B, NORM_L2):
        → np.linalg.norm(A - B)
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.fft import next_fast_len
from scipy.signal import convolve2d
from scipy.ndimage import zoom as ndimage_zoom
from scipy.ndimage import gaussian_filter


# ═════════════════════════════════════════════════════════════════════════════
# PSF ↔ OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert point-spread function to optical transfer function.

    Zero-pads *psf* to *shape*, circularly shifts so that the PSF centre
    (psf_shape // 2) moves to position (0, 0), then computes fft2.

    Replaces the C++ FTMat class pipeline for kernels / small filters:
        FTMat.toVector(1, 1, 0, Nsize, Msize) + FTMat.DFT()
    The C++ "reverse-raster from centre" indexing achieves the same
    circular shift as np.roll here.

    Parameters
    ----------
    psf   : (kh, kw) float64 kernel.
    shape : (H, W) output OTF size (typically the FFT-padded shape).

    Returns
    -------
    otf : (H, W) complex128 — the kernel's frequency response.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape[:2]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    # Circular shift: move centre of PSF to (0, 0).
    padded = np.roll(np.roll(padded, -(ph // 2), axis=0),
                     -(pw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, psf_shape: tuple) -> np.ndarray:
    """
    Convert optical transfer function back to a point-spread function.

    Inverse of psf2otf: ifft2, undo circular shift, crop to *psf_shape*.

    Parameters
    ----------
    otf       : (H, W) complex128.
    psf_shape : (kh, kw) desired kernel size.

    Returns
    -------
    psf : (kh, kw) float64.
    """
    full = np.real(ifft2(otf))

    # Undo the circular shift performed by psf2otf.
    sh = psf_shape[0] // 2
    sw = psf_shape[1] // 2
    full = np.roll(np.roll(full, sh, axis=0), sw, axis=1)

    return full[:psf_shape[0], :psf_shape[1]]


# ═════════════════════════════════════════════════════════════════════════════
# FFT size & padding helpers
# ═════════════════════════════════════════════════════════════════════════════

def optimal_fft_shape(im_shape: tuple, ker_shape: tuple) -> tuple:
    """
    Compute optimal padded size for FFT-based convolution / deconvolution.

    Mirrors ``cv::getOptimalDFTSize(im_size + ker_size)`` used in every
    C++ function that touches DFT (Convolution, UpdateDeconvImage, etc.).

    Parameters
    ----------
    im_shape  : (H, W) of the image.
    ker_shape : (kh, kw) of the kernel.

    Returns
    -------
    (fft_h, fft_w) : efficient FFT dimensions ≥ im + ker.
    """
    h = next_fast_len(im_shape[0] + ker_shape[0])
    w = next_fast_len(im_shape[1] + ker_shape[1])
    return (h, w)


def pad_to_fft(image: np.ndarray, fft_shape: tuple,
               mode: str = 'edge') -> np.ndarray:
    """
    Symmetrically pad *image* to *fft_shape*.

    Mirrors the C++ FTMat::toVector padding step:
        PaddingWay=1 (BORDER_REPLICATE)  →  mode='edge'
        PaddingWay=0 (BORDER_CONSTANT)   →  mode='constant'

    The image is placed at the centre of the padded array,
    matching the C++ ZeroPoint=1 (centre-origin) convention.

    Parameters
    ----------
    image     : (H, W) float64 array.
    fft_shape : (fft_H, fft_W) target padded shape.
    mode      : 'edge' or 'constant' (zero).

    Returns
    -------
    padded : (fft_H, fft_W) float64 array.
    """
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
    """
    Extract the central region of *array*.

    Mirrors the C++ FTMat::toMatrix crop step, which removes the
    symmetric padding added before DFT:
        Half_width  = (Xsize - Cols) / 2
        Half_height = (Ysize - Rows) / 2
        ImgMat = ImgMat_sub(Rect(Half_width, Half_height, Cols, Rows))

    Parameters
    ----------
    array        : (H, W) or larger ndarray.
    target_shape : (h, w) desired output size.

    Returns
    -------
    cropped : (h, w) ndarray (contiguous copy).
    """
    h, w = target_shape[:2]
    H, W = array.shape[:2]
    top = (H - h) // 2
    left = (W - w) // 2
    return array[top:top + h, left:left + w].copy()


# ═════════════════════════════════════════════════════════════════════════════
# Convolution via FFT
# ═════════════════════════════════════════════════════════════════════════════

def convolve_fft(image: np.ndarray, kernel: np.ndarray,
                 padding: str = 'edge') -> np.ndarray:
    """
    Convolve *image* with *kernel* via FFT.

    Mirrors the C++ ``Convolution::convolved`` method:
        1. Pad image to optimal FFT size (edge-replicate by default).
        2. psf2otf(kernel, fft_shape).
        3. Multiply in frequency domain.
        4. iFFT, crop to original image size.

    Parameters
    ----------
    image   : (H, W) float64 input image.
    kernel  : (kh, kw) float64 convolution kernel.
    padding : 'edge' (BORDER_REPLICATE) or 'constant' (zero-pad).

    Returns
    -------
    result : (H, W) float64 convolved image.
    """
    fft_shape = optimal_fft_shape(image.shape, kernel.shape)
    im_padded = pad_to_fft(image, fft_shape, mode=padding)
    im_fft = fft2(im_padded)
    ker_otf = psf2otf(kernel, fft_shape)

    result = np.real(ifft2(im_fft * ker_otf))
    return crop_center(result, image.shape)


# ═════════════════════════════════════════════════════════════════════════════
# Kernel utilities
# ═════════════════════════════════════════════════════════════════════════════

def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Enforce kernel constraints: non-negative values, sum = 1.

    Mirrors C++ KERNEL::normalization():
        1. Clamp negatives to 0.
        2. Divide by total sum so that Σ k_ij = 1.

    Parameters
    ----------
    kernel : (kh, kw) float64 array.

    Returns
    -------
    k : (kh, kw) normalised kernel (new array).
    """
    k = np.copy(kernel)
    k[k < 0] = 0.0
    s = k.sum()
    if s > 0:
        k /= s
    return k


def create_delta_kernel(shape: tuple) -> np.ndarray:
    """
    Create a delta (identity) kernel — 1.0 at the centre, 0 elsewhere.

    Equivalent to C++ KERNEL(FastOrder) when the kernel is a single
    central pixel (FastOrder=0 in the C++ code uses a 2×2 block; here
    we use a single pixel for a true delta).

    Parameters
    ----------
    shape : (kh, kw) kernel dimensions.

    Returns
    -------
    k : (kh, kw) float64 delta kernel.
    """
    k = np.zeros(shape, dtype=np.float64)
    cy, cx = shape[0] // 2, shape[1] // 2
    k[cy, cx] = 1.0
    return k


def resize_kernel(kernel: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize kernel by *factor* and re-normalise.

    Mirrors C++ ``KERNEL::resizeTo`` which calls ``cv::resize``
    (bilinear) and then the KERNEL(Mat&) constructor (normalises).

    Parameters
    ----------
    kernel : (kh, kw) float64 kernel.
    factor : scale factor (e.g. 2.0 = double the size).

    Returns
    -------
    resized : new kernel, bilinear-interpolated, non-negative, sum = 1.
    """
    if factor == 1.0:
        return normalize_kernel(kernel)
    resized = ndimage_zoom(kernel, factor, order=1)
    return normalize_kernel(resized)


# ═════════════════════════════════════════════════════════════════════════════
# Gradient operators
# ═════════════════════════════════════════════════════════════════════════════

def sobel_h() -> np.ndarray:
    """
    3×3 horizontal Sobel filter (∂/∂x approximation).

    Identical to the C++ code's ``grad_h`` used in UpdateDeconvImage:
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]
    """
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)


def sobel_v() -> np.ndarray:
    """
    3×3 vertical Sobel filter (∂/∂y approximation).

    Identical to the C++ code's ``grad_v`` used in UpdateDeconvImage:
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]
    """
    return np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# Edge detection for MRF weights
# ═════════════════════════════════════════════════════════════════════════════

# OpenCV Laplacian(ksize=3) = Sobel(dx=2,ksize=3) + Sobel(dy=2,ksize=3)
# which yields the kernel [[2,0,2],[0,-8,0],[2,0,2]].
_LAPLACIAN_KERNEL_3 = np.array([[ 2,  0,  2],
                                [ 0, -8,  0],
                                [ 2,  0,  2]], dtype=np.float64)


def compute_laplacian_abs(image: np.ndarray) -> np.ndarray:
    """
    Compute absolute-value Laplacian of a grayscale image.

    Matches the C++ pipeline used in UpdateQuantizedImage_wighted():
        cv::Laplacian(gray_Img, contrust, CV_64F, 3);
        cv::convertScaleAbs(contrust, contrust, 1, 0);
        contrust.convertTo(contrust, CV_64FC1);

    OpenCV Laplacian with ksize=3 uses the Sobel-based kernel
    [[2,0,2],[0,-8,0],[2,0,2]].  convertScaleAbs takes abs() and
    converts to uint8.  The final convertTo brings it back to float64.
    We skip the uint8 saturation (precision loss) since we stay in
    float64 throughout, but round-trip through uint8 to match C++.

    Parameters
    ----------
    image : (H, W) float64, values in [0, 1].

    Returns
    -------
    edge_map : (H, W) float64, absolute Laplacian.
    """
    # convolve2d with 'symm' boundary matches OpenCV's default
    # BORDER_REFLECT_101 for Laplacian.
    lap = convolve2d(image, _LAPLACIAN_KERNEL_3, mode='same', boundary='symm')
    # Match C++ convertScaleAbs: abs → uint8 → float64.
    result = np.abs(lap)
    result = np.clip(result, 0, 255).astype(np.uint8).astype(np.float64)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# K-means quantization
# ═════════════════════════════════════════════════════════════════════════════

def kmeans_quantize(image: np.ndarray,
                    n_clusters: int = 15,
                    max_iter: int = 10) -> tuple:
    """
    Quantise a grayscale image using k-means clustering.

    Mirrors the C++ code's k-means step inside UpdateQuantizedImage:
        1. Reshape pixel values into a feature matrix.
        2. OpenCV kmeans with KMEANS_RANDOM_CENTERS, 10 iterations.
        3. Build a quantised image using cluster centre values.

    In the C++ code the input is a 3-channel (BGR) image reshaped to
    (N, 3).  For our grayscale pipeline, pixels are 1-D scalars.

    Parameters
    ----------
    image      : (H, W) float64, pixel values in [0, 1].
    n_clusters : number of quantisation levels (C++: MAX_CLUSTERS = 15).
    max_iter   : maximum k-means iterations (C++: TermCriteria::COUNT=10).

    Returns
    -------
    labels  : (H, W) int32 — cluster index per pixel.
    centres : (n_clusters,) float64 — sorted cluster centre values.
    """
    from scipy.cluster.vq import kmeans2

    h, w = image.shape
    pixels = image.ravel().astype(np.float64)

    # scipy.cluster.vq.kmeans2 expects (n_samples, n_features).
    data = pixels.reshape(-1, 1)

    # minit='points' picks initial centres from data points,
    # analogous to OpenCV KMEANS_RANDOM_CENTERS.
    centres, labels = kmeans2(data, n_clusters, minit='points',
                              iter=max_iter, seed=42)
    centres = centres.ravel()

    # Handle possible empty clusters: keep only used centres.
    unique_labels = np.unique(labels)
    if len(unique_labels) < n_clusters:
        old_to_new = np.zeros(n_clusters, dtype=np.int32)
        for new_idx, old_idx in enumerate(unique_labels):
            old_to_new[old_idx] = new_idx
        labels = old_to_new[labels]
        centres = centres[unique_labels]
        n_clusters = len(unique_labels)

    # Sort centres so that label ordering is deterministic.
    order = np.argsort(centres)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_clusters)
    centres = centres[order]
    labels = inv_order[labels]

    return labels.reshape(h, w).astype(np.int32), centres.astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# Image scaling
# ═════════════════════════════════════════════════════════════════════════════

def resize_image(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize image by *factor* using bilinear interpolation.

    Mirrors ``cv::resize(src, dst, Size(), factor, factor)``
    called in BlindDeconvolution::initialization for every pyramid level.

    Parameters
    ----------
    image  : (H, W) float64.
    factor : scale factor (e.g. 0.5 = half size).

    Returns
    -------
    resized : bilinear-interpolated image.
    """
    return ndimage_zoom(image, factor, order=1)


# ═════════════════════════════════════════════════════════════════════════════
# Image quality metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Mean Squared Error.

    Mirrors the C++ Evaluation_MSE_PSNR_SSIM MSE block:
        MSE = Σ (img1 - img2)² / N

    Parameters
    ----------
    img1, img2 : arrays of the same shape (any dtype; cast to float64).

    Returns
    -------
    mse : float.
    """
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(img1: np.ndarray, img2: np.ndarray,
                 max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Mirrors C++:
        PSNR = 20·log10(MAX_INTENSE) − 10·log10(MSE)

    Returns ``float('inf')`` for identical images.

    Parameters
    ----------
    img1, img2 : arrays of the same shape.
    max_val    : peak value (255 for uint8, 1.0 for normalised).

    Returns
    -------
    psnr : float (dB).
    """
    mse = compute_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(max_val) - 10.0 * np.log10(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                 max_val: float = 255.0) -> float:
    """
    Structural Similarity Index (SSIM).

    Matches the C++ SSIMcalc function:
        - Gaussian window 11×11, σ = 1.5
        - C1 = (0.01 · max_val)²,  C2 = (0.03 · max_val)²

    OpenCV GaussianBlur(I, mu, Size(11,11), 1.5) uses an 11×11
    Gaussian kernel with σ=1.5.  In scipy.ndimage.gaussian_filter,
    truncate = 5/1.5 ≈ 3.333 gives kernel radius 5 → size 11.

    Parameters
    ----------
    img1, img2 : (H, W) arrays of the same shape.
    max_val    : peak value (255 for uint8, 1.0 for normalised images).

    Returns
    -------
    ssim : float in [-1, 1].
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    I1 = img1.astype(np.float64)
    I2 = img2.astype(np.float64)

    # truncate = 5.0 / 1.5 ≈ 3.333 → kernel radius = int(3.333*1.5+0.5) = 5
    # → kernel size = 2*5 + 1 = 11, matching OpenCV's Size(11,11).
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
