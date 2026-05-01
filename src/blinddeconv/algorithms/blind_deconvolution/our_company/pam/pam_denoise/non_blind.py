"""
non_blind.py

Non-blind image deconvolution with space-variant regularization
and adaptive noise modelling.

Reference:
    "Adaptive Non-Blind Image Deblurring with Space-Variant Gradient
     and Noise Modelling" — Qingsong Wang et al.

Key ideas ported from the reference implementation:
    1. Lp regularization on image gradients (hyper-Laplacian, α ∈ (0, 2]).
    2. Lp fidelity for the noise term (α_n via KL divergence estimation).
    3. Space-variant λ(x,y) per-pixel regularization weight derived from
       local gradient statistics vs. estimated noise standard deviation.
    4. 1D λ-interpolation: build a library of restored images for a
       geometric grid of λ values, then interpolate per-pixel.
    5. Two-stage pipeline: first pass with α_n = α (gradient prior),
       second pass with KL-estimated α_n.

Dependencies: numpy, scipy, pywt (PyWavelets).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.stats import entropy

__all__ = ['adaptive_lp_deconv', 'ringing_removal']


# ═════════════════════════════════════════════════════════════════════════════
# Ringing removal — re-exported from DCP solvers (TV + L0 + bilateral diff)
# ═════════════════════════════════════════════════════════════════════════════
def ringing_removal(blurred, kernel, lambda_tv=3e-3, lambda_l0=5e-4,
                    weight_ring=1.0):
    """Thin wrapper around DCP's ringing_artifacts_removal.

    Equivalent to MATLAB ringing_artifacts_removal.m by Pan et al. (CVPR 2016).

    Parameters
    ----------
    blurred     : (H, W) blurred image (float, [0, 1]).
    kernel      : (Mk, Nk) PSF.
    lambda_tv   : TV weight (typical 1e-3 .. 1e-2).
    lambda_l0   : L0 weight (typical 1e-4 .. 2e-3).
    weight_ring : ringing suppression weight (0 = TV-only).

    Returns
    -------
    (H, W) restored image.
    """
    # Lazy import — DCP package is heavy and only needed when this is called.
    from blinddeconv.algorithms.blind_deconvolution.our_company.dark_channel_prior\
        .dcp_with_denoiser.solvers import ringing_artifacts_removal
    return ringing_artifacts_removal(
        blurred, kernel,
        float(lambda_tv), float(lambda_l0), float(weight_ring),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Proximal operators (LUT-based, Krishnan & Fergus NIPS 2009)
# ═════════════════════════════════════════════════════════════════════════════

_LUT_RANGE = 10
_LUT_STEP = 0.0001
_XX = np.arange(-_LUT_RANGE, _LUT_RANGE + _LUT_STEP, _LUT_STEP)


def _compute_w1(v, beta):
    """Soft-thresholding (α = 1)."""
    return np.sign(v) * np.maximum(np.abs(v) - 1.0 / beta, 0.0)


def _compute_w23(v, beta):
    """Ferrari's method for α = 2/3 (Alg. 3 in NIPS paper)."""
    eps = 1e-6
    m = np.full_like(v, 8.0 / (27.0 * beta ** 3))
    t1 = (-9.0 / 8.0) * v ** 2
    t2 = (1.0 / 4.0) * v ** 3
    t3 = (-1.0 / 8.0) * m * v ** 2
    t4 = -t3 / 2.0 + np.sqrt(
        (-m ** 3 / 27.0 + m ** 2 * v ** 4 / 256.0).astype(np.complex128))
    t5 = t4 ** (1.0 / 3.0)
    t6 = 2.0 * (-5.0 / 18.0 * t1 + t5 + m / (3.0 * t5))
    t7 = np.sqrt((t1 / 3.0 + t6).astype(np.complex128))

    root = np.zeros(v.shape + (4,), dtype=np.complex128)
    root[:, 0] = 0.75 * v + (
        t7 + np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 1] = 0.75 * v + (
        t7 - np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 2] = 0.75 * v + (
        -t7 + np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0
    root[:, 3] = 0.75 * v + (
        -t7 - np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0

    c1 = np.abs(np.imag(root)) < eps
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 &= (c23 > 0.5 * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)


def _compute_w12(v, beta):
    """Cardano's method for α = 1/2 (Alg. 2 in NIPS paper)."""
    eps = 1e-6
    m = -np.sign(v) / (4.0 * beta ** 2)
    t1 = (2.0 / 3.0) * v
    inner = (27.0 * m ** 2 + 4.0 * m * v ** 3).astype(np.complex128)
    t2 = (-27.0 * m - 2.0 * v ** 3 + 3.0 ** 1.5 * np.sqrt(inner)) ** (1.0 / 3.0)
    t2 = np.where(np.abs(t2) < eps, eps, t2)
    t3 = v ** 2 / t2

    root = np.zeros(v.shape + (3,), dtype=np.complex128)
    root[:, 0] = t1 + (2 ** (1.0 / 3.0)) / 3.0 * t3 + t2 / (3.0 * 2 ** (1.0 / 3.0))
    root[:, 1] = (
        t1
        - ((1.0 + 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 - 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)
    root[:, 2] = (
        t1
        - ((1.0 - 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 + 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)

    root = np.where(np.isfinite(root), root, 0)
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 = np.abs(np.imag(root)) < eps
    c1 &= (c23 > (2.0 / 3.0) * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)


def _newton_w(v, beta, alpha):
    """Newton's method fallback for general α."""
    w = v.copy().astype(np.float64)
    for _ in range(4):
        df = alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)
        ddf = alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta
        w -= df / ddf
    w = np.where(np.isfinite(w), w, 0)
    cost0 = (beta / 2.0) * v ** 2
    costw = np.abs(w) ** alpha + (beta / 2.0) * (w - v) ** 2
    return np.where(costw < cost0, w, 0)


def _compute_w(v, beta, alpha):
    eps = 1e-9
    if abs(alpha - 1.0) < eps:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < eps:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < eps:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)


# Module-level LUT cache
_lut_cache = {}


def _solve_img(v, beta, alpha):
    """Proximal operator via LUT interpolation."""
    key = (beta, alpha)
    if key not in _lut_cache:
        _lut_cache[key] = _compute_w(_XX, beta, alpha)
    lut = _lut_cache[key]
    return np.interp(v.ravel(), _XX, lut).reshape(v.shape)


def _clear_lut_cache():
    _lut_cache.clear()


# ═════════════════════════════════════════════════════════════════════════════
# FFT helpers
# ═════════════════════════════════════════════════════════════════════════════

def _psf2otf(psf, shape):
    """PSF to OTF: zero-pad, circshift centre to (0,0), fft2."""
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


# ═════════════════════════════════════════════════════════════════════════════
# Core ADMM deconvolution with Lp gradient + Lp noise fidelity
# ═════════════════════════════════════════════════════════════════════════════

def _fast_deconv_adaptive(yin, kernel, alpha, alpha_n, lam):
    """
    Non-blind deconvolution with Lp gradient prior AND Lp noise fidelity.

    Solves:
        min_x  λ · (||∇_x x||^α + ||∇_y x||^α)  +  ||Hx - y||^{α_n}

    via half-quadratic splitting (ADMM):
        - w_n subproblem: proximal operator on noise residual
        - w_x, w_y subproblems: proximal operator on gradients
        - x subproblem: closed-form in Fourier domain

    Parameters
    ----------
    yin : 2D array — blurred image [0, 1]
    kernel : 2D array — PSF (sum = 1)
    alpha : float — hyper-Laplacian exponent for gradients
    alpha_n : float — hyper-Laplacian exponent for noise fidelity
    lam : float — regularization weight (scalar, for this single image)

    Returns
    -------
    yout : 2D array — restored image
    """
    M, N = yin.shape

    K = _psf2otf(kernel, (M, N))
    Y = fft2(yin)
    Nomin1 = np.conj(K) * Y       # K^T · B
    Denom1 = np.abs(K) ** 2        # |K|^2

    gx = np.array([[1, -1]], dtype=np.float64)
    gy = np.array([[1], [-1]], dtype=np.float64)
    Gx = _psf2otf(gx, (M, N))
    Gy = _psf2otf(gy, (M, N))
    Denom2 = np.abs(Gx) ** 2 + np.abs(Gy) ** 2

    yout = yin.copy()

    # Gradients and noise residual
    youtx = np.roll(yout, -1, axis=1) - yout
    youty = np.roll(yout, -1, axis=0) - yout
    youtn = yin - np.real(ifft2(fft2(yout) * K))

    # Continuation schedule
    betas = np.geomspace(1, 2 ** 8, num=9)
    gamma = 1.0 / 50.0     # = beta_g / beta_n
    beta_n = betas * lam / gamma
    beta_g = betas

    for i in range(len(betas)):
        # w-subproblems (proximal operators)
        Wn = _solve_img(youtn, beta_n[i], alpha_n)
        Wx = _solve_img(youtx, beta_g[i], alpha)
        Wy = _solve_img(youty, beta_g[i], alpha)

        # x-subproblem (Fourier domain closed-form)
        Wxx = np.roll(Wx, 1, axis=1) - Wx
        Wyy = np.roll(Wy, 1, axis=0) - Wy
        Wnn = np.real(ifft2(fft2(Wn) * np.conj(K)))

        W = -Wnn + gamma * (Wxx + Wyy)
        Denom = Denom1 + Denom2 * gamma
        Fyout = (fft2(W) + Nomin1) / Denom
        yout = np.real(ifft2(Fyout))
        yout = np.clip(yout, 0, 1)

        # Update gradients and residual
        youtx = np.roll(yout, -1, axis=1) - yout
        youty = np.roll(yout, -1, axis=0) - yout
        youtn = yin - np.real(ifft2(fft2(yout) * K))

    return yout


# ═════════════════════════════════════════════════════════════════════════════
# Noise std estimation (DWT-based, from reference implementation)
# ═════════════════════════════════════════════════════════════════════════════

def _dwt_hh(img):
    """Extract HH (diagonal detail) subband via 2-level DWT (db2)."""
    import pywt
    _, (_, _, HH) = pywt.dwt2(img, 'db2')
    return HH


def _local_std(grad_map, L=10):
    """Local standard deviation with (2L+1)×(2L+1) window."""
    win = 2 * L + 1
    k = np.ones((win, win), dtype=np.float64) / (win ** 2)
    ms_local = convolve2d(grad_map ** 2, k, mode='same', boundary='symm')
    return np.sqrt(ms_local)


def _x_grad(img):
    gx = np.array([[1, 0, -1]], dtype=np.float64)
    return convolve2d(img, gx, mode='same', boundary='symm')


def _y_grad(img):
    gy = np.array([[1, 0, -1]], dtype=np.float64).reshape(3, 1)
    return convolve2d(img, gy, mode='same', boundary='symm')


def _find_turning_point(sorted_std, M, N):
    """Find noise-floor turning point in sorted local-std array."""
    original = sorted_std.copy()
    d = max(1, int(M * N / 2000))
    smooth = np.ones(2 * d + 1, dtype=np.float64)
    smooth[0] = 0
    smooth[d + 1:] = -1
    filtered = convolve1d(sorted_std, smooth, mode='reflect')

    ind = 0
    for i in range(1, len(filtered)):
        if filtered[i] < filtered[ind]:
            break
        ind = i - 1
    return original[ind]


def _estimate_noise_std(image):
    """
    Estimate additive noise σ from a single image using DWT-based
    local gradient statistics.

    Returns
    -------
    sigma_n : float — estimated noise σ (in image scale)
    """
    M, N = image.shape
    HH = _dwt_hh(image)
    Bgx, Bgy = _x_grad(HH), _y_grad(HH)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)
    sort_sgx = np.sort(sgx.ravel())
    sort_sgy = np.sort(sgy.ravel())
    sgx_n = _find_turning_point(sort_sgx, M, N)
    sgy_n = _find_turning_point(sort_sgy, M, N)
    Eg = np.sum(np.array([1, 0, -1], dtype=np.float64) ** 2)
    sigma_n = np.sqrt((sgx_n ** 2 + sgy_n ** 2 + 1e-8) / Eg)
    return sigma_n


# ═════════════════════════════════════════════════════════════════════════════
# Space-variant λ map
# ═════════════════════════════════════════════════════════════════════════════

def _compute_lambda_map(image, sigma_n, alpha):
    """
    Compute per-pixel regularization weight λ(x,y) from local gradient
    statistics and estimated noise σ.

    Matches the original implementation:
      1. DWT-HH → local std of x/y gradients → turning points sgx_n, sgy_n
      2. Image gradients → local std sgx, sgy
      3. σ_gsx = sqrt(sgx² − sgx_n²),  σ_gsy = sqrt(sgy² − sgy_n²)
      4. λ(x,y) = (√(2σ_n² / (σ_gsx² + σ_gsy²)))^α
    """
    eps = 1e-8
    M, N = image.shape

    # Step 1: noise floor per direction (from DWT-HH)
    HH = _dwt_hh(image)
    Bgx_hh, Bgy_hh = _x_grad(HH), _y_grad(HH)
    sgx_hh, sgy_hh = _local_std(Bgx_hh, 10), _local_std(Bgy_hh, 10)
    sgx_n = _find_turning_point(np.sort(sgx_hh.ravel()), M, N)
    sgy_n = _find_turning_point(np.sort(sgy_hh.ravel()), M, N)

    # Step 2: image gradients → local std
    Bgx, Bgy = _x_grad(image), _y_grad(image)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)

    # Step 3: subtract per-direction noise floor (as in original sigma_gs)
    sigma_gsx_sq = sgx ** 2 - sgx_n ** 2
    sigma_gsx_sq[sigma_gsx_sq < eps] = eps
    sigma_gsx = np.sqrt(sigma_gsx_sq)

    sigma_gsy_sq = sgy ** 2 - sgy_n ** 2
    sigma_gsy_sq[sigma_gsy_sq < eps] = eps
    sigma_gsy = np.sqrt(sigma_gsy_sq)

    # Step 4: λ map
    lam_map = (np.sqrt(2 * sigma_n ** 2 / (
        sigma_gsx ** 2 + sigma_gsy ** 2 + eps))) ** alpha
    return lam_map


# ═════════════════════════════════════════════════════════════════════════════
# α_n estimation via KL divergence
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_alpha_n(blurred, restored, kernel, sigma_n):
    """
    Estimate noise exponent α_n by minimizing KL divergence between
    observed noise residual and simulated hyper-Laplacian noise.
    """
    import math

    noise_observed = blurred - restored

    threshold = 0.3
    best_alpha, best_kl = 0.5, np.inf

    for i in range(1, 10):
        alpha_n = round(0.1 * i, 2)

        # Generate hyper-Laplacian reference noise
        rng = np.random.default_rng(0)
        beta_hl = sigma_n * np.sqrt(
            math.gamma(1.0 / alpha_n) / math.gamma(3.0 / alpha_n))
        T = rng.gamma(shape=1.0 / alpha_n, scale=1.0, size=blurred.shape)
        S = rng.choice([-1.0, 1.0], size=blurred.shape)
        noise_ref = beta_hl * S * (T ** (1.0 / alpha_n))

        # Mask out near-boundary pixels (clipping artifacts)
        mask = (restored >= threshold) & (restored <= 1.0 - threshold)
        noise_sample = noise_observed[mask]
        noise_ref_masked = noise_ref[mask]

        if noise_sample.size < 100:
            continue

        # Compare histograms via KL divergence
        dx = 0.01
        bins = np.arange(-threshold, threshold + dx, dx)
        hist_s, _ = np.histogram(noise_sample, bins)
        hist_s = hist_s.astype(np.float64) / hist_s.sum() + 1e-12
        hist_r, _ = np.histogram(noise_ref_masked, bins)
        hist_r = hist_r.astype(np.float64) / hist_r.sum() + 1e-12

        kl = entropy(hist_s, hist_r)
        if kl < best_kl:
            best_alpha, best_kl = alpha_n, kl

    return best_alpha


# ═════════════════════════════════════════════════════════════════════════════
# Lambda library + 1D interpolation
# ═════════════════════════════════════════════════════════════════════════════

def _build_lambda_library(alpha, C, lam_N):
    """Geometric grid of λ values: C · 2^{(α/3)·i}."""
    i = np.arange(lam_N, dtype=np.float64)
    return C * (2 ** ((alpha / 3.0) * i))


def _interpolate_library(blurred, kernel, alpha, alpha_n, lam_map, lam_library):
    """
    Build per-λ restored images and interpolate per-pixel.

    For each λ in lam_library, run full-image deconvolution (with
    internal mirror-padding). Then for each pixel, blend the two
    nearest λ-images based on the per-pixel λ weight.

    All images and lam_map are at ORIGINAL (unpadded) resolution.
    """
    C = lam_library[0]

    # Build image library (with saturation detection for speed)
    I_library = {}
    sat = False
    prev_I = None
    for idx in range(len(lam_library)):
        if not sat:
            I_library[idx] = _deconv_with_padding(
                blurred, kernel, alpha, alpha_n, lam_library[idx])
        else:
            I_library[idx] = prev_I.copy()
            continue
        if prev_I is not None and np.array_equal(I_library[idx], prev_I):
            sat = True
        prev_I = I_library[idx].copy()

    # Per-pixel interpolation (vectorized index computation, per-pixel blend)
    M, N = lam_map.shape
    raw_idx = np.ceil((3.0 / alpha) * np.log2(
        np.maximum(lam_map / C, 1.0))).astype(int)
    raw_idx = np.clip(raw_idx, 0, len(lam_library) - 2)

    w_map = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            denom = np.log(lam_library[i + 1] / max(lam_library[i], 1e-12))
            if abs(denom) < 1e-12:
                w_map[m, n] = 0.5
            else:
                w_map[m, n] = (np.log(
                    lam_library[i + 1] / max(lam_map[m, n], 1e-12)
                ) / denom) ** 1.4

    I_opt = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            w = w_map[m, n]
            I_opt[m, n] = (
                w * I_library[i][m, n]
                + (1.0 - w) * I_library[i + 1][m, n])

    return np.clip(I_opt, 0, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Mirror padding (replicate boundary conditions)
# ═════════════════════════════════════════════════════════════════════════════

def _mirror_pad(image, pad):
    """Symmetric (mirror) padding to handle boundary conditions."""
    return np.pad(image, pad, mode='reflect')


def _mirror_unpad(image, pad, orig_shape):
    """Remove mirror padding."""
    return image[pad:pad + orig_shape[0], pad:pad + orig_shape[1]]


def _deconv_with_padding(blurimg, kernel, alpha, alpha_n, lam):
    """
    Deconvolve with internal mirror-padding (matches original deconv()).
    Pads the image, runs ADMM on padded domain, then un-pads.
    """
    M, N = blurimg.shape
    k_size = kernel.shape[0]
    padded = _mirror_pad(blurimg, k_size)
    result = _fast_deconv_adaptive(padded, kernel, alpha, alpha_n, lam)
    return result[k_size:k_size + M, k_size:k_size + N]


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

def adaptive_lp_deconv(blurred, kernel, alpha=0.8, sigma_n=None,
                       two_stage=True):
    """
    Non-blind deconvolution with space-variant Lp regularization
    and adaptive noise modelling.

    Parameters
    ----------
    blurred : ndarray, H×W
        Blurred (and possibly noisy) grayscale image, float64 [0, 1].
    kernel : ndarray, h×w
        Blur kernel (PSF). Will be normalized to sum = 1.
    alpha : float, optional
        Hyper-Laplacian exponent for image gradient prior (default 0.8).
        Literature suggests α ∈ [0.5, 0.8] for natural images.
    sigma_n : float or None, optional
        Noise standard deviation (in [0, 1] image scale).
        If None, estimated automatically via DWT-based method.
    two_stage : bool, optional
        If True (default), run a second deconvolution pass with
        KL-estimated noise exponent α_n. If False, use α_n = α.

    Returns
    -------
    restored : ndarray, H×W
        Restored image, float64 [0, 1].

    Notes
    -----
    This method is significantly slower than single-pass FHLP due to
    building a library of N_λ deconvolved images. Typical N_λ ≈ 10–30,
    so expect 10–30× the cost of a single FHLP call.
    """
    kernel = kernel.astype(np.float64)
    kernel = np.maximum(kernel, 1e-10)
    kernel /= kernel.sum()

    blurred = blurred.astype(np.float64)
    if blurred.max() > 1.0:
        blurred /= 255.0

    M, N = blurred.shape

    # Step 1: Estimate noise σ
    if sigma_n is None:
        sigma_n = _estimate_noise_std(blurred)
    sigma_n = max(sigma_n, 1e-8)

    # Step 2: Compute space-variant λ map on ORIGINAL (unpadded) image
    lam_map = _compute_lambda_map(blurred, sigma_n, alpha)

    # Step 3: Build λ library (geometric grid)
    C = max(lam_map.min(), 1e-12)
    lam_N = int(np.ceil(3.0 / alpha * np.log2(
        max(lam_map.max() / C, 1.0))) + 2)
    lam_N = max(lam_N, 3)
    lam_library = _build_lambda_library(alpha, C, lam_N)

    # Step 4: First pass — α_n = α (gradient prior as noise model)
    #         Each deconv call handles padding/unpadding internally.
    alpha_n = alpha
    _clear_lut_cache()
    I_opt = _interpolate_library(
        blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    # Step 5: Second pass — estimate α_n via KL divergence
    if two_stage:
        alpha_n = _estimate_alpha_n(blurred, I_opt, kernel, sigma_n)

        # Robustness heuristics (from reference implementation)
        if sigma_n > 0.025 or alpha_n == 0.5:
            alpha_n = max(alpha_n, 0.6)
        center_val = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
        if center_val < 1e-4:
            alpha_n = 0.8

        # Re-run with estimated α_n
        _clear_lut_cache()
        I_opt = _interpolate_library(
            blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    return np.clip(I_opt, 0, 1)
