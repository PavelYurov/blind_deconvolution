import numpy as np
from numpy.fft import fft2, ifft2

from .utils import (
    psf2otf,
    otf2psf,
    optimal_fft_shape,
    pad_to_fft,
    crop_center,
    normalize_kernel,
    create_delta_kernel,
    resize_kernel,
    resize_image,
    sobel_h,
    sobel_v,
    compute_laplacian_abs,
    kmeans_quantize,
)

MAX_ITERATION = 10
MAX_ITERATION_ADMM = 10
MAX_CLUSTERS = 15
MYU = 0.4e-3
RAMBDA = 0.4e-3
TAU = 1.0e-3
PENALTY_PARAMETER = 1.0e+3
CONVERGENCE_THRESHOLD = 1.0e-8

PYRAMID_NUM = 8
RESIZE_FACTORS = (0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)

def update_quantized_image(
    deconv_image: np.ndarray,
    quantized_image: np.ndarray,
    mu: float = MYU,
    n_clusters: int = MAX_CLUSTERS,
    n_mrf_iterations: int = MAX_ITERATION,
) -> np.ndarray:

    h, w = quantized_image.shape

    labels, centres = kmeans_quantize(quantized_image, n_clusters)
    n_actual = len(centres)

    edge_map = compute_laplacian_abs(deconv_image)

    weights = np.zeros_like(edge_map)
    mask = edge_map != 0.0
    weights[mask] = 1.0 - 1.0 / edge_map[mask]

    data_terms = np.empty((n_actual, h, w), dtype=np.float64)
    for l in range(n_actual):
        data_terms[l] = mu * (centres[l] - deconv_image) ** 2

    for _icm in range(n_mrf_iterations):
        energies = data_terms.copy()

        for l in range(n_actual):

            energies[l, :, 1:] += weights[:, :-1] * (labels[:, :-1] != l)

            energies[l, :, :-1] += weights[:, 1:] * (labels[:, 1:] != l)

            energies[l, 1:, :] += weights[:-1, :] * (labels[:-1, :] != l)

            energies[l, :-1, :] += weights[1:, :] * (labels[1:, :] != l)

        labels = np.argmin(energies, axis=0).astype(np.int32)

    return centres[labels]

def update_image_fft(
    blurred: np.ndarray,
    quantized: np.ndarray,
    kernel: np.ndarray,
    lam: float = RAMBDA,
    mu: float = MYU,
) -> np.ndarray:

    fft_shape = optimal_fft_shape(blurred.shape, kernel.shape)

    Y_fft = fft2(pad_to_fft(blurred, fft_shape, 'edge'))
    Xt_fft = fft2(pad_to_fft(quantized, fft_shape, 'edge'))

    K_fft = psf2otf(kernel, fft_shape)
    Dh_fft = psf2otf(sobel_h(), fft_shape)
    Dv_fft = psf2otf(sobel_v(), fft_shape)

    numerator = np.conj(K_fft) * Y_fft + mu * Xt_fft

    denominator = (np.abs(K_fft) ** 2
                   + lam * (np.abs(Dh_fft) ** 2 + np.abs(Dv_fft) ** 2)
                   + mu)

    X_fft = numerator / denominator
    x = np.real(ifft2(X_fft))
    return crop_center(x, blurred.shape)

def update_kernel_admm(
    blurred: np.ndarray,
    quantized: np.ndarray,
    kernel: np.ndarray,
    tau: float = TAU,
    rho: float = PENALTY_PARAMETER,
    n_admm_iter: int = MAX_ITERATION_ADMM,
) -> np.ndarray:

    fft_shape = optimal_fft_shape(blurred.shape, kernel.shape)

    Y_fft = fft2(pad_to_fft(blurred, fft_shape, 'edge'))
    Xt_fft = fft2(pad_to_fft(quantized, fft_shape, 'edge'))

    Xt_conj_Y = np.conj(Xt_fft) * Y_fft
    Xt_sq = np.abs(Xt_fft) ** 2
    rho_half = rho / 2.0
    threshold = tau / rho

    z = np.zeros_like(kernel)
    k = kernel.copy()

    for _admm in range(n_admm_iter):

        kz_fft = psf2otf(k + z, fft_shape)
        b_fft = Xt_conj_Y + rho_half * kz_fft
        A_fft = Xt_sq + rho_half
        K_sub_fft = b_fft / A_fft
        k_sub = otf2psf(K_sub_fft, kernel.shape)

        v = k_sub - z
        k = np.maximum(v - threshold, 0.0)
        k = normalize_kernel(k)

        z = z - k_sub + k

    return k

def _match_size(image: np.ndarray, target_shape: tuple) -> np.ndarray:

    th, tw = target_shape[:2]
    ih, iw = image.shape[:2]
    if ih == th and iw == tw:
        return image
    result = np.zeros(target_shape, dtype=image.dtype)
    ch = min(th, ih)
    cw = min(tw, iw)
    result[:ch, :cw] = image[:ch, :cw]
    return result

def blind_deconvolution(
    blurred: np.ndarray,
    kernel_shape: tuple = (40, 40),
    *,
    mu: float = MYU,
    lam: float = RAMBDA,
    tau: float = TAU,
    rho: float = PENALTY_PARAMETER,
    n_clusters: int = MAX_CLUSTERS,
    max_iter: int = MAX_ITERATION,
    max_admm_iter: int = MAX_ITERATION_ADMM,
    convergence_thresh: float = CONVERGENCE_THRESHOLD,
    resize_factors: tuple = RESIZE_FACTORS,
    verbose: bool = False,
) -> tuple:

    n_levels = len(resize_factors)

    blurred_255 = blurred * 255.0

    blurred_levels = [resize_image(blurred_255, f) for f in resize_factors]

    kernel = create_delta_kernel(kernel_shape)

    deconv_pyr = blurred_levels[0].copy()
    quant_pyr = blurred_levels[0].copy()
    kernel_pyr = resize_kernel(kernel, resize_factors[0])

    for pyr in range(n_levels):
        blurred_pyr = blurred_levels[pyr]

        if verbose:
            print(
                f"Pyramid level {pyr}/{n_levels - 1}  "
                f"image {blurred_pyr.shape}  "
                f"kernel {kernel_pyr.shape}"
            )

        for it in range(max_iter):

            quant_pyr = update_quantized_image(
                deconv_pyr, quant_pyr, mu, n_clusters, max_iter,
            )

            deconv_pyr = update_image_fft(
                blurred_pyr, quant_pyr, kernel_pyr, lam, mu,
            )
            deconv_pyr = np.clip(deconv_pyr, 0.0, 255.0)

            kernel_before = kernel_pyr.copy()
            kernel_pyr = update_kernel_admm(
                blurred_pyr, quant_pyr, kernel_pyr, tau, rho,
                max_admm_iter,
            )

            diff = np.linalg.norm(kernel_pyr - kernel_before) / kernel_pyr.size
            if verbose:
                print(f"  iter {it}: kernel_diff = {diff:.2e}")
            if diff < convergence_thresh:
                break

        if pyr < n_levels - 1:
            up_factor = resize_factors[pyr + 1] / resize_factors[pyr]
            next_shape = blurred_levels[pyr + 1].shape

            deconv_pyr = _match_size(
                resize_image(deconv_pyr, up_factor), next_shape,
            )
            quant_pyr = _match_size(
                resize_image(quant_pyr, up_factor), next_shape,
            )
            kernel_pyr = resize_kernel(kernel_pyr, up_factor)

    return deconv_pyr / 255.0, kernel_pyr
