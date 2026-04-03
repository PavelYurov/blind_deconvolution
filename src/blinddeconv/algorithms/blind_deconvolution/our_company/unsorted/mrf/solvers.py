"""
solvers.py

Core solver functions for MRF-based Blind Image Deconvolution.

Ported from C++ (OpenCV) code based on:
    N. Komodakis, N. Paragios: "MRF-based Blind Image Deconvolution",
    Proceedings of the 11th Asian Conference on Computer Vision (ACCV),
    Vol. 3, pp. 361-374, 2012.

Function mapping to C++ source
──────────────────────────────────────────────────────────────────────
    update_quantized_image  ←  BlindDeconvolution::UpdateQuantizedImage_wighted()
    update_image_fft        ←  BlindDeconvolution::UpdateDeconvImage_FFT()
    update_kernel_admm      ←  BlindDeconvolution::UpdateKarnel()  +  CG_method_k()
    blind_deconvolution     ←  BlindDeconvolution::initialization()
                               + BlindDeconvolution::deblurring()
                               + BlindDeconvolution::Upsampling()

Differences from the C++ implementation
────────────────────────────────────────────────────────────────────────
1. Grayscale only (C++ code processes 3-channel BGR images; the user's
   requirement is single-channel grayscale input/output).

2. Image update (update_image_fft):
   The C++ code performs an iDFT → spatial addition → DFT round-trip to
   build the denominator *A* and numerator *b*.  When the constant μ is
   added in spatial domain and then DFT'd back, only the DC component
   receives the penalty (μ·N·δ[0]) instead of every frequency bin.
   Our implementation adds μ directly in the frequency domain, matching
   the correct closed-form solution from the paper:
       X̂ = (K̂*·Ŷ + μ·X̃̂) / (|K̂|² + λ(|D̂ₕ|²+|D̂ᵥ|²) + μ)

3. Kernel update (update_kernel_admm):
   The C++ CG_method_k uses |K̂|² for the operator matrix A; the correct
   normal-equation operator is |X̃̂|².  Our code uses a direct frequency-
   domain division (since A is diagonal in freq. domain), avoiding CG
   altogether while being both simpler and correct.

4. MRF-ICM (update_quantized_image):
   The C++ ICM compares the *current* pixel value against neighbours
   (δ(x̃_p_current ≠ x̃_q)), making the smoothness term independent of
   the candidate label.  Our code compares the *candidate* value with
   neighbours (δ(candidate ≠ x̃_q)), so the MRF actually penalises
   label disagreements and produces spatially coherent quantisations.

5. No mean subtraction before DFT (C++ subtracts settingAverageColor).
   The frequency-domain formula handles the DC component correctly via μ.
"""

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


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm constants  (from C++ Blind_Deconvolution.h / main.h)
# ═════════════════════════════════════════════════════════════════════════════

MAX_ITERATION = 10          # outer alternating-minimisation iterations
MAX_ITERATION_ADMM = 10     # inner ADMM iterations for kernel update
MAX_CLUSTERS = 15           # k-means cluster count
MYU = 0.4e-3                # μ  — quantised / deconvolved coupling weight
RAMBDA = 0.4e-3             # λ  — gradient (Sobel) regularisation weight
TAU = 1.0e-3                # τ  — L1 sparsity weight on the kernel
PENALTY_PARAMETER = 1.0e+3  # ρ  — ADMM augmented-Lagrangian penalty
CONVERGENCE_THRESHOLD = 1.0e-8  # kernel-change stopping criterion

PYRAMID_NUM = 8
RESIZE_FACTORS = (0.1, 0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Update quantised image  x̃   (k-means + MRF-ICM)
# ═════════════════════════════════════════════════════════════════════════════

def update_quantized_image(
    deconv_image: np.ndarray,
    quantized_image: np.ndarray,
    mu: float = MYU,
    n_clusters: int = MAX_CLUSTERS,
    n_mrf_iterations: int = MAX_ITERATION,
) -> np.ndarray:
    """
    Update the quantised image x̃ via k-means clustering + MRF-ICM.

    Mirrors C++ ``UpdateQuantizedImage_wighted()``.

    Pipeline
    --------
    1. K-means on the current quantised image → cluster centres.
    2. Laplacian edge map from the deconvolved image → MRF weights.
    3. ICM iterations:  for every pixel, pick the cluster centre that
       minimises  μ·(c − x_p)² + Σ_q w_pq·δ(c ≠ x̃_q).

    C++ note
    --------
    The original code uses the *current* pixel value in the δ function,
    making the smoothness term independent of the candidate.  We use
    the *candidate* instead so that the MRF actually enforces spatial
    coherence (see module docstring, item 4).

    Parameters
    ----------
    deconv_image    : (H, W) float64 in [0, 255] — current restored image x.
    quantized_image : (H, W) float64 in [0, 255] — current quantised image x̃.
    mu              : MRF data-term weight (tuned for [0, 255] range).
    n_clusters      : number of k-means levels (C++: MAX_CLUSTERS = 15).
    n_mrf_iterations: ICM sweeps (C++: MAX_Iteration = 10).

    Returns
    -------
    quantized : (H, W) float64 in [0, 255] — updated x̃.
    """
    h, w = quantized_image.shape

    # ── Step 1: k-means quantisation ─────────────────────────────────────
    labels, centres = kmeans_quantize(quantized_image, n_clusters)
    n_actual = len(centres)

    # ── Step 2: edge weights from |Laplacian| of deconvolved image ───────
    # C++ applies Laplacian on [0, 255] grayscale images.
    # Caller must pass images in [0, 255] range.
    edge_map = compute_laplacian_abs(deconv_image)

    # w_pq = 1 − 1/|Laplacian|  (0 at flat regions, → 1 at strong edges)
    weights = np.zeros_like(edge_map)
    mask = edge_map != 0.0
    weights[mask] = 1.0 - 1.0 / edge_map[mask]

    # ── Step 3: ICM optimisation (vectorised over pixels) ────────────────
    # Pre-compute data term for every candidate label.
    # data_terms[l, y, x] = μ · (centres[l] − deconv[y, x])²
    data_terms = np.empty((n_actual, h, w), dtype=np.float64)
    for l in range(n_actual):
        data_terms[l] = mu * (centres[l] - deconv_image) ** 2

    for _icm in range(n_mrf_iterations):
        energies = data_terms.copy()

        # Smoothness:  w_pq · δ(candidate ≠ neighbour label)
        for l in range(n_actual):
            # Left neighbour (y, x−1)
            energies[l, :, 1:] += weights[:, :-1] * (labels[:, :-1] != l)
            # Right neighbour (y, x+1)
            energies[l, :, :-1] += weights[:, 1:] * (labels[:, 1:] != l)
            # Top neighbour (y−1, x)
            energies[l, 1:, :] += weights[:-1, :] * (labels[:-1, :] != l)
            # Bottom neighbour (y+1, x)
            energies[l, :-1, :] += weights[1:, :] * (labels[1:, :] != l)

        labels = np.argmin(energies, axis=0).astype(np.int32)

    return centres[labels]


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Update deconvolved image  x   (closed-form FFT)
# ═════════════════════════════════════════════════════════════════════════════

def update_image_fft(
    blurred: np.ndarray,
    quantized: np.ndarray,
    kernel: np.ndarray,
    lam: float = RAMBDA,
    mu: float = MYU,
) -> np.ndarray:
    """
    Update the deconvolved image x via closed-form FFT division.

    Mirrors C++ ``UpdateDeconvImage_FFT()``.

    Closed-form solution (derived by setting ∂E/∂x = 0 in freq. domain):

    .. math::

        \\hat{x} = \\frac{\\bar{\\hat{k}} \\cdot \\hat{y}
                          + \\mu \\, \\hat{\\tilde{x}}}
                         {|\\hat{k}|^2
                          + \\lambda\\,(|\\hat{\\partial}_h|^2
                                       + |\\hat{\\partial}_v|^2)
                          + \\mu}

    All operations are element-wise in the DFT domain.

    Parameters
    ----------
    blurred   : (H, W) float64 — blurred input y.
    quantized : (H, W) float64 — quantised image x̃.
    kernel    : (kh, kw) float64 — current kernel estimate k.
    lam       : gradient regularisation weight  λ  (C++: Rambda = 0.4e-3).
    mu        : data coupling weight  μ  (C++: Myu = 0.4e-3).

    Returns
    -------
    deconv : (H, W) float64 — updated restored image x.
    """
    fft_shape = optimal_fft_shape(blurred.shape, kernel.shape)

    # FFT of images (edge-padded  ↔  C++ BORDER_REPLICATE)
    Y_fft = fft2(pad_to_fft(blurred, fft_shape, 'edge'))
    Xt_fft = fft2(pad_to_fft(quantized, fft_shape, 'edge'))

    # OTF of kernel and gradient filters
    K_fft = psf2otf(kernel, fft_shape)
    Dh_fft = psf2otf(sobel_h(), fft_shape)
    Dv_fft = psf2otf(sobel_v(), fft_shape)

    # Numerator:  K̂*·Ŷ  +  μ·X̃̂
    numerator = np.conj(K_fft) * Y_fft + mu * Xt_fft

    # Denominator:  |K̂|² + λ·(|D̂ₕ|² + |D̂ᵥ|²) + μ
    denominator = (np.abs(K_fft) ** 2
                   + lam * (np.abs(Dh_fft) ** 2 + np.abs(Dv_fft) ** 2)
                   + mu)

    X_fft = numerator / denominator
    x = np.real(ifft2(X_fft))
    return crop_center(x, blurred.shape)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Update kernel  k   (ADMM  +  direct FFT sub-problem)
# ═════════════════════════════════════════════════════════════════════════════

def update_kernel_admm(
    blurred: np.ndarray,
    quantized: np.ndarray,
    kernel: np.ndarray,
    tau: float = TAU,
    rho: float = PENALTY_PARAMETER,
    n_admm_iter: int = MAX_ITERATION_ADMM,
) -> np.ndarray:
    """
    Update the blur kernel k using ADMM with L1 regularisation.

    Mirrors C++ ``UpdateKarnel()`` + ``CG_method_k()``.

    Minimises  ‖y − k∗x̃‖² + τ‖k‖₁   via ADMM splitting:

    k'-subproblem (quadratic → closed-form in freq. domain)::

        K̂' = (X̃̂*·Ŷ + ρ/2 · \\widehat{k+z}) / (|X̃̂|² + ρ/2)

    k-subproblem (correct proximal L1 + non-negativity)::

        k = max(k' − z − τ/ρ,  0)     (correct ADMM proximal)
        k = k / Σk                            (normalise)

    Dual update::

        z ← z − k' + k

    C++ note
    --------
    The C++ CG solver uses |K̂|² (kernel power spectrum) as the
    operator A; the correct operator derived from ∂E/∂k = 0 is |X̃̂|².
    We use the correct formulation (see module docstring, item 3).

    Parameters
    ----------
    blurred    : (H, W) float64 — blurred image y.
    quantized  : (H, W) float64 — quantised image x̃.
    kernel     : (kh, kw) float64 — current kernel estimate.
    tau        : L1 sparsity weight  τ  (C++: Tau = 1.0e-3).
    rho        : ADMM penalty  ρ  (C++: PenaltyParameter = 1.0e+3).
    n_admm_iter: ADMM iterations (C++: MAX_Iteration_ADMM = 10).

    Returns
    -------
    kernel_new : (kh, kw) float64 — updated kernel (non-negative, sum = 1).
    """
    fft_shape = optimal_fft_shape(blurred.shape, kernel.shape)

    # FFT of images (edge-padded)
    Y_fft = fft2(pad_to_fft(blurred, fft_shape, 'edge'))
    Xt_fft = fft2(pad_to_fft(quantized, fft_shape, 'edge'))

    # Pre-compute fixed terms
    Xt_conj_Y = np.conj(Xt_fft) * Y_fft   # cross-correlation  X̃̂*·Ŷ
    Xt_sq = np.abs(Xt_fft) ** 2            # power spectrum  |X̃̂|²
    rho_half = rho / 2.0
    threshold = tau / rho

    # ADMM variables  (C++: Kernel_sub = k', TransVector = z)
    z = np.zeros_like(kernel)           # dual variable
    k = kernel.copy()                   # primal variable

    for _admm in range(n_admm_iter):
        # ── k'-subproblem (closed-form in frequency domain) ──────────
        kz_fft = psf2otf(k + z, fft_shape)
        b_fft = Xt_conj_Y + rho_half * kz_fft
        A_fft = Xt_sq + rho_half
        K_sub_fft = b_fft / A_fft
        k_sub = otf2psf(K_sub_fft, kernel.shape)

        # ── k-subproblem (correct ADMM proximal operator) ────────
        # Proximal of (τ/ρ)·‖·‖₁ + I_{≥0} at point (k_sub - z):
        #   k = max(k_sub - z - τ/ρ, 0)
        # The C++ uses |k'-k_old| which is a bug (see module docstring).
        v = k_sub - z
        k = np.maximum(v - threshold, 0.0)
        k = normalize_kernel(k)

        # ── dual variable update ─────────────────────────────────────
        z = z - k_sub + k

    return k


# ═════════════════════════════════════════════════════════════════════════════
# Helper: match array size after upsampling (handles ±1 pixel rounding)
# ═════════════════════════════════════════════════════════════════════════════

def _match_size(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Crop or zero-pad *image* so that it has exactly *target_shape*.

    Upsampling with ``scipy.ndimage.zoom`` can produce arrays that differ
    by ±1 pixel from the expected size due to rounding.  This helper
    silently adjusts for that difference.
    """
    th, tw = target_shape[:2]
    ih, iw = image.shape[:2]
    if ih == th and iw == tw:
        return image
    result = np.zeros(target_shape, dtype=image.dtype)
    ch = min(th, ih)
    cw = min(tw, iw)
    result[:ch, :cw] = image[:ch, :cw]
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Full coarse-to-fine blind deconvolution pipeline
# ═════════════════════════════════════════════════════════════════════════════

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
    """
    Full MRF-based blind image deconvolution pipeline.

    Mirrors C++ ``BlindDeconvolution::initialization()`` followed by
    ``BlindDeconvolution::deblurring()`` with ``Upsampling()`` between
    pyramid levels.

    Coarse-to-fine strategy
    -----------------------
    The algorithm processes the blurred image at increasing resolutions
    (controlled by *resize_factors*).  At each pyramid level:

    1. Resize the original blurred image to the current scale.
    2. Initialise deconvolved / quantised images from the previous
       level by upsampling (first level uses the blurred image).
    3. Alternate:
       a. Update quantised image  x̃   (k-means + MRF-ICM).
       b. Update restored image   x   (closed-form FFT).
       c. Update kernel           k   (ADMM + L1).
       until convergence or *max_iter* reached.
    4. Upsampling results to the next pyramid level.

    Parameters
    ----------
    blurred        : (H, W) float64 in [0, 1] — grayscale blurred image.
                     Internally scaled to [0, 255] to match C++ parameters.
    kernel_shape   : (kh, kw) — maximum blur kernel size
                     (C++ default: 40 × 40).
    mu             : μ  — quantised ↔ deconvolved coupling weight.
                     Designed for [0, 255] pixel range.
    lam            : λ  — gradient (Sobel) regularisation weight.
    tau            : τ  — L1 kernel sparsity weight.
    rho            : ρ  — ADMM penalty parameter.
    n_clusters     : k-means cluster count.
    max_iter       : outer alternating-minimisation iterations per level.
    max_admm_iter  : ADMM iterations inside kernel update.
    convergence_thresh : stop early when kernel change < this value.
    resize_factors : tuple of scale factors for the image pyramid
                     (C++ default: 0.1 … 1.0 in 8 levels).
    verbose        : print progress to stdout.

    Returns
    -------
    restored : (H, W) float64 — restored (deblurred) image.
    kernel   : (kh, kw) float64 — estimated blur kernel
               (non-negative, sum = 1).
    """
    n_levels = len(resize_factors)

    # ── Scale to [0, 255] (C++ convention) ────────────────────────────────
    # The C++ code operates on CV_64FC3 images in [0, 255] range.
    # All parameters (μ, λ, τ, ρ) are tuned for this scale.  Working at
    # [0, 1] would make the MRF data-term 65 025× too small and the
    # ADMM penalty 65 025× too strong relative to the data fit.
    blurred_255 = blurred * 255.0

    # ── Initialisation (mirrors C++ initialization()) ────────────────────
    # Pre-resize the blurred image for every pyramid level.
    blurred_levels = [resize_image(blurred_255, f) for f in resize_factors]

    # The initial kernel is a delta at the requested size.
    # C++ uses KERNEL(KernelFastOrder=1), a predefined motion-blur
    # kernel; a delta is a more general starting point.
    kernel = create_delta_kernel(kernel_shape)

    # At the first level, deconv and quantised images start as the
    # (resized) blurred image — identical to C++ initialisation:
    #   DeconvImg[pyr] = FTMat3D(NowdoubleConvIMG)
    #   QuantImg[pyr]  = FTMat3D(NowdoubleConvIMG)
    deconv_pyr = blurred_levels[0].copy()
    quant_pyr = blurred_levels[0].copy()
    kernel_pyr = resize_kernel(kernel, resize_factors[0])

    # ── Coarse-to-fine loop (mirrors C++ deblurring()) ───────────────────
    for pyr in range(n_levels):
        blurred_pyr = blurred_levels[pyr]

        if verbose:
            print(
                f"Pyramid level {pyr}/{n_levels - 1}  "
                f"image {blurred_pyr.shape}  "
                f"kernel {kernel_pyr.shape}"
            )

        # ── Alternating minimisation ─────────────────────────────────
        for it in range(max_iter):
            # (a) Update x̃  —  k-means + MRF-ICM
            quant_pyr = update_quantized_image(
                deconv_pyr, quant_pyr, mu, n_clusters, max_iter,
            )

            # (b) Update x  —  closed-form FFT
            deconv_pyr = update_image_fft(
                blurred_pyr, quant_pyr, kernel_pyr, lam, mu,
            )
            deconv_pyr = np.clip(deconv_pyr, 0.0, 255.0)

            # (c) Update k  —  ADMM
            kernel_before = kernel_pyr.copy()
            kernel_pyr = update_kernel_admm(
                blurred_pyr, quant_pyr, kernel_pyr, tau, rho,
                max_admm_iter,
            )

            # Convergence check (C++: diff_Kernel / size < 1e-8)
            diff = np.linalg.norm(kernel_pyr - kernel_before) / kernel_pyr.size
            if verbose:
                print(f"  iter {it}: kernel_diff = {diff:.2e}")
            if diff < convergence_thresh:
                break

        # ── Upsampling to next level (mirrors C++ Upsampling()) ──────
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

    # ── Return full-resolution results ───────────────────────────────────
    # Scale back to [0, 1].
    return deconv_pyr / 255.0, kernel_pyr
