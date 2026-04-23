"""
solvers.py

Core solver functions for LMGP (Local Maximum Gradient Prior) blind deconvolution.

Ported from MATLAB code by Chen, Liang et al.
Reference:
    L. Chen, F. Fang, T. Wang, G. Zhang:
    "Blind Image Deblurring With Local Maximum Gradient Prior",
    CVPR, 2019.

Original MATLAB code based on Jinshan Pan's DCP framework (CVPR 2016).

Contains:
    estimate_psf           — PSF estimation via conjugate gradient
                             (estimate_psf.m)
    L0_LMG_deblur          — L0 deblurring with LMG prior
                             (L0_LMG_deblur.m)  [KEY LMG CONTRIBUTION]
    L0Restoration          — L0 deblurring with gradient prior only
                             (L0Restoration.m)
    blind_deconv_main      — single-scale blind deconvolution loop
                             (blind_deconv_main.m)
    blind_deconv           — multi-scale coarse-to-fine blind deconvolution
                             (blind_deconv.m)
    deblurring_adm_aniso   — TV-l2 deblurring via ADM / Split Bregman
                             (deblurring_adm_aniso.m)
    ringing_artifacts_removal — artifact removal post-processing
                             (standard non-blind restoration wrapper)

MATLAB -> Python notes  (see utils.py for full table):
    - diff(S,1,2)  ->  np.diff(S, n=1, axis=1)
    - [diff(S,1,2), S(:,1,:)-S(:,end,:)]  ->
        np.concatenate([np.diff(S,1,axis=1), S[:,0:1]-S[:,-1:]], axis=1)
    - conv2(A,B,'valid')  ->  scipy.signal.convolve2d(A,B,'valid')
    - bwconncomp(k,8)  ->  scipy.ndimage.label(k, structure=ones(3,3))
    - imresize(k,ret)  ->  scipy.ndimage.zoom(k, ret, order=3)
    - graythresh(I)  ->  _graythresh(I)  (Otsu threshold, custom impl)
    - speye(M,N)  ->  scipy.sparse.eye(M, N)
    - A'*A  ->  A.T @ A
    - A \\ b  ->  scipy.sparse.linalg.spsolve(A, b)

    LMG sparse operator (utils.LMG) uses COLUMN-MAJOR ordering:
    all flatten/reshape in L0_LMG_deblur use order='F'.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates, gaussian_filter

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    LMG,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
    guided_filter,
    find_min_pixels,
    nlm_filter,
    bm3d_filter,
)


# ═════════════════════════════════════════════════════════════════════════════
# _graythresh  (equivalent to MATLAB graythresh — Otsu's method)
# ═════════════════════════════════════════════════════════════════════════════

def _graythresh(img):
    """
    Compute Otsu's threshold.  Equivalent to MATLAB graythresh for double images.

    MATLAB graythresh for doubles:
        1. Clips input to [0, 1]
        2. Computes 256-bin histogram (bin centres at k/255, k=0..255)
        3. Finds threshold maximising inter-class variance (Otsu's method)

    Returns scalar threshold in [0, 1].
    """
    img = np.clip(np.asarray(img, dtype=np.float64).ravel(), 0.0, 1.0)

    nbins = 256
    # MATLAB imhist for doubles: bins centred at k/(nbins-1), k = 0..nbins-1
    indices = np.round(img * (nbins - 1)).astype(np.intp)
    indices = np.clip(indices, 0, nbins - 1)
    counts = np.bincount(indices, minlength=nbins).astype(np.float64)

    total = counts.sum()
    if total == 0:
        return 0.0

    p = counts / total
    bin_mids = np.arange(nbins, dtype=np.float64) / (nbins - 1)

    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_mids)
    mu_t = mu[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sigma_b_sq = np.where(np.isfinite(sigma_b_sq), sigma_b_sq, 0.0)

    idx = int(np.argmax(sigma_b_sq))
    return bin_mids[idx]


# ═════════════════════════════════════════════════════════════════════════════
# estimate_psf  (from estimate_psf.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_Ax(x, p):
    """
    Matrix-vector product for the PSF estimation CG system.

    MATLAB:
        x_f = psf2otf(x, p.img_size);
        y   = otf2psf(p.m .* x_f, p.psf_size);
        y   = y + p.lambda * x;
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size,
                 kernel_reg_weight=0.0):
    """
    Estimate blur kernel from gradient images via conjugate gradient.
    Equivalent to MATLAB estimate_psf.m.

    Parameters
    ----------
    blurred_x, blurred_y : gradient images of blurred input
    latent_x, latent_y   : gradient images of latent estimate
    weight               : regularisation weight (gamma)
    psf_size             : (kh, kw) kernel size
    kernel_reg_weight    : float — additional Tikhonov regularisation on
                           the kernel (noise-aware). 0 = original.

    Returns
    -------
    psf : (kh, kw) estimated kernel, thresholded and normalised
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(otf2psf(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape[:2],
        'psf_size': psf_size,
        'lambda': weight + kernel_reg_weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# L0_LMG_deblur  (from L0_LMG_deblur.m)
# — KEY LMG CONTRIBUTION: L0-TV + Local Maximum Gradient Prior
# ═════════════════════════════════════════════════════════════════════════════

def L0_LMG_deblur(Im, kernel, lambda_lmg, wei_grad, kappa=2.0,
                  lmg_denoise_eps=None, lmg_denoise_radius=2,
                  lmg_denoise_type='guided',
                  lmg_bilateral_sigma_s=2.0, lmg_bilateral_sigma_r=0.1,
                  lmg_bm3d_sigma=0.01, lmg_nlm_h=0.01,
                  use_soft_threshold=False,
                  softmax_tau=None):
    """
    Image restoration with L0 gradient prior and LMG prior.
    Equivalent to MATLAB L0_LMG_deblur.m — the core of the LMG paper.

    Solves (via half-quadratic splitting / ADMM):
        min_S  ||S*k - B||^2  +  lambda * ||2 - G_S(S)||_1
               + wei_grad * ||nabla S||_0

    where G_S is the Local Maximum Gradient operator (Eq. 3-4 of the paper).

    Parameters
    ----------
    Im                    : (H, W) blurred image (ALREADY boundary-wrapped)
    kernel                : (kh, kw) blur kernel
    lambda_lmg            : weight for LMG prior
    wei_grad              : weight for L0 gradient prior
    kappa                 : ADM update ratio (default 2.0)
    lmg_denoise_eps       : float or None — guided filter eps / enable flag
                            for bilateral. None = no denoising.
    lmg_denoise_radius    : int — guided filter radius (default 2)
    lmg_denoise_type      : 'guided' | 'bilateral' — which denoiser before LMG
    lmg_bilateral_sigma_s : float — bilateral spatial sigma (default 2.0)
    lmg_bilateral_sigma_r : float — bilateral range sigma (default 0.1)
    use_soft_threshold    : bool — use L1 soft threshold instead of L0 hard
                            threshold on gradients (default False)

    Returns
    -------
    S : (H, W) restored image (same size as Im, caller must crop)

    CRITICAL: The LMG sparse operator uses COLUMN-MAJOR (Fortran) ordering.
    All flatten/reshape in this function use order='F'.
    """
    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    # MATLAB: [N,M,D] = size(Im)  — N=rows, M=cols
    rows, cols = Im.shape[:2]
    sizeI2D = (rows, cols)

    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    # Data fidelity term — computed once from original image
    Normin1 = np.conj(KER) * fft2(S)

    # ── Pixel sub-problem setup ──────────────────────────────
    patch_size = 35  # Fixed size (Sec. 4.1 of the paper)

    # ── Helper: denoise S for LMG / graythresh ───────────────
    def _denoise_for_lmg(img):
        if lmg_denoise_eps is not None and lmg_denoise_eps > 0:
            if lmg_denoise_type == 'bm3d':
                return bm3d_filter(img, sigma_psd=lmg_bm3d_sigma)
            elif lmg_denoise_type == 'nlm':
                return nlm_filter(img, h=lmg_nlm_h)
            elif lmg_denoise_type == 'bilateral':
                return bilateral_filter(img, lmg_bilateral_sigma_s,
                                        lmg_bilateral_sigma_r)
            else:
                return guided_filter(img, img, lmg_denoise_radius,
                                     lmg_denoise_eps)
        return img

    # MATLAB: mybeta_pixel = lambda / graythresh(S.^2)
    # Use denoised S for graythresh to stabilise under noise
    S_clean = _denoise_for_lmg(S)
    gt = _graythresh(S_clean ** 2)
    if gt == 0:
        gt = 1e-10  # avoid division by zero
    mybeta_pixel = lambda_lmg / gt

    n_pixels = rows * cols

    # ── Outer loop (4 iterations) ────────────────────────────
    for _outer in range(4):
        # Denoise before LMG to suppress noise peaks
        S_for_lmg = _denoise_for_lmg(S)

        # Compute LMG operator on (denoised) image
        # J: (rows, cols) LMG map;  A: (n_pixels, n_pixels) sparse operator
        J, A = LMG(S_for_lmg, patch_size, softmax_tau=softmax_tau)

        # Soft-thresholding: u = shrink(2 - J, lambda_lmg / (2*mybeta_pixel))
        t = 2.0 - J
        t2 = lambda_lmg / (2.0 * mybeta_pixel)
        t3 = np.abs(t) - t2
        t3[t3 < 0] = 0.0
        u = np.sign(t) * t3

        alpha3 = mybeta_pixel * 2.0

        # Pre-compute A^T @ A (reused across inner iterations)
        AtA = A.T @ A

        # ── Inner loop (4 iterations) ────────────────────────
        for _inner in range(4):
            # Pixel sub-problem: sparse linear solve
            # MATLAB: subsitute_I = (mybeta_pixel*(A'*A) + alpha3*speye) \
            #             (mybeta_pixel*A'*(2-u(:)) + alpha3*S(:))
            lhs = mybeta_pixel * AtA + alpha3 * sparse.eye(n_pixels,
                                                           format='csr')
            rhs = (mybeta_pixel * A.T @ (2.0 - u.flatten(order='F'))
                   + alpha3 * S.flatten(order='F'))
            subsitute_I_vec = spsolve(lhs, rhs)
            subsitute_I = subsitute_I_vec.reshape((rows, cols), order='F')

            # Gradient sub-problem: L0 hard-threshold on gradients
            beta = 2.0 * wei_grad
            while beta < betamax:
                # Circular differences
                h = np.concatenate([np.diff(S, n=1, axis=1),
                                    S[:, 0:1] - S[:, -1:]], axis=1)
                v = np.concatenate([np.diff(S, n=1, axis=0),
                                    S[0:1, :] - S[-1:, :]], axis=0)

                if use_soft_threshold:
                    # L1 soft thresholding (more noise-robust)
                    lam_half = wei_grad / (2.0 * beta)
                    h = np.sign(h) * np.maximum(np.abs(h) - lam_half, 0.0)
                    v = np.sign(v) * np.maximum(np.abs(v) - lam_half, 0.0)
                else:
                    # L0 hard thresholding (original)
                    th = h ** 2 < wei_grad / beta
                    tv = v ** 2 < wei_grad / beta
                    h[th] = 0.0
                    v[tv] = 0.0

                # Divergence (backward differences)
                Normin2 = np.concatenate([h[:, -1:] - h[:, 0:1],
                                          -np.diff(h, n=1, axis=1)], axis=1)
                Normin2 = Normin2 + np.concatenate(
                    [v[-1:, :] - v[0:1, :],
                     -np.diff(v, n=1, axis=0)], axis=0)

                # Image sub-problem (Fourier domain)
                FS = ((Normin1
                       + beta * fft2(Normin2)
                       + alpha3 * fft2(subsitute_I))
                      / (Den_KER + beta * Denormin2 + alpha3))
                S = np.real(ifft2(FS))

                beta = beta * kappa

            alpha3 = alpha3 * 4.0

        mybeta_pixel = mybeta_pixel * 4.0

    return S


# ═════════════════════════════════════════════════════════════════════════════
# L0Restoration  (from L0Restoration.m)
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """
    Image restoration with L0 gradient prior.
    Equivalent to MATLAB L0Restoration.m.

    Solves:  S* = argmin_S  ||S*k - B||^2 + lambda * |nabla S|_0

    Parameters
    ----------
    Im          : (H, W) blurred image (original size, NOT wrapped)
    kernel      : (kh, kw) blur kernel
    lambda_grad : weight for L0 gradient prior
    kappa       : ADM update ratio (default 2.0)

    Returns
    -------
    S : (H, W) restored image cropped to original size
    """
    H_orig, W_orig = Im.shape[0], Im.shape[1]

    # MATLAB: Im = wrap_boundary_liu(Im, opt_fft_size([H W]+size(kernel)-1))
    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1
    )
    Im = wrap_boundary_liu(Im, tuple(target_size))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    N, M = Im.shape[:2]
    sizeI2D = (N, M)

    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    Normin1 = np.conj(KER) * fft2(S)

    beta = 2 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        # Circular differences
        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1] - S[:, -1:]], axis=1)
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :] - S[-1:, :]], axis=0)

        # MATLAB: t = (h.^2+v.^2) < lambda/beta  (D==1 case, JOINT threshold)
        t = (h ** 2 + v ** 2) < lambda_grad / beta
        h[t] = 0.0
        v[t] = 0.0

        # Divergence
        Normin2_val = np.concatenate([h[:, -1:] - h[:, 0:1],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :] - v[0:1, :],
             -np.diff(v, n=1, axis=0)], axis=0)

        FS = (Normin1 + beta * fft2(Normin2_val)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa

    S = S[:H_orig, :W_orig]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main  (from blind_deconv_main.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main(blur_B, k, lambda_lmg, lambda_grad, threshold, opts,
                      iteration_callback=None):
    """
    Single-scale blind deconvolution.
    Equivalent to MATLAB blind_deconv_main.m.

    Alternates between:
        1. Latent image estimation (L0_LMG_deblur or L0Restoration)
        2. Gradient thresholding (threshold_pxpy_v1)
        3. Kernel estimation (estimate_psf)
        4. Kernel cleanup (connected components, normalisation)

    Parameters
    ----------
    blur_B      : (H, W) blurred image
    k           : (kh, kw) current kernel estimate
    lambda_lmg  : weight for LMG prior
    lambda_grad : weight for L0 gradient prior
    threshold   : gradient threshold (updated per iteration)
    opts        : dict with 'xk_iter', and optional denoise params:
                  'denoise_eps', 'denoise_radius', 'ensemble_denoise',
                  'grad_smooth_sigma', 'lmg_denoise_eps', 'lmg_denoise_radius'

    Returns
    -------
    k           : updated kernel
    lambda_lmg  : updated lambda_lmg
    lambda_grad : updated lambda_grad
    S           : intermediate latent image

    LMG-specific differences from DCP / PMP:
        - Calls L0_LMG_deblur (not deblur_tv_pmpr / L0Deblur_dark_channel)
        - lambda_lmg floor = 1e-4 (PMP uses 1e-2, DCP uses 1e-4)
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H = blur_B.shape[0]
    W = blur_B.shape[1]

    # MATLAB: blur_B_w = wrap_boundary_liu(blur_B, opt_fft_size([H W]+size(k)-1))
    target_size = opt_fft_size(
        np.array([H, W]) + np.array(k.shape[:2]) - 1
    )
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target_size))
    blur_B_tmp = blur_B_w[:H, :W]

    # Surgical histogram equalization for kernel estimation only.
    # Applied symmetrically to both blur_B_tmp (for Bx/By) and S (for
    # latent_x/latent_y) to keep grad(B) vs grad(S)*k consistent.
    # L0_LMG_deblur input (blur_B_w) and non-blind stage stay untouched.
    kernel_eq_method = opts.get('kernel_eq', 'none')
    kernel_eq_params = opts.get('kernel_eq_params', None) or {}

    def _apply_kernel_eq(img):
        if kernel_eq_method in (None, 'none'):
            return img
        from skimage.exposure import equalize_adapthist, equalize_hist
        img_c = np.clip(img, 0.0, 1.0)
        if kernel_eq_method == 'clahe':
            return equalize_adapthist(
                img_c,
                clip_limit=kernel_eq_params.get('clip_limit', 0.003),
                nbins=kernel_eq_params.get('nbins', 256),
                kernel_size=kernel_eq_params.get('kernel_size', None),
            )
        elif kernel_eq_method == 'global':
            return equalize_hist(img_c)
        raise ValueError(
            f"Unknown kernel_eq='{kernel_eq_method}'. "
            f"Choose from: 'clahe', 'global', 'none'")

    blur_B_for_grad = _apply_kernel_eq(blur_B_tmp)

    Bx = convolve2d(blur_B_for_grad, dx, mode='valid')
    By = convolve2d(blur_B_for_grad, dy, mode='valid')

    # Pre-smooth blurred-image gradients to suppress noise
    grad_smooth_sigma = opts.get('grad_smooth_sigma', None)
    if grad_smooth_sigma is not None and grad_smooth_sigma > 0:
        Bx = gaussian_filter(Bx, sigma=grad_smooth_sigma)
        By = gaussian_filter(By, sigma=grad_smooth_sigma)

    xk_iter = opts.get('xk_iter', 5)
    denoise_eps = opts.get('denoise_eps', None)
    denoise_radius = opts.get('denoise_radius', 2)
    ensemble_denoise = opts.get('ensemble_denoise', False)
    denoise_type = opts.get('denoise_type', 'guided')
    denoise_bilateral_sigma_s = opts.get('denoise_bilateral_sigma_s', 2.0)
    denoise_bilateral_sigma_r = opts.get('denoise_bilateral_sigma_r', 0.1)
    denoise_bm3d_sigma = opts.get('denoise_bm3d_sigma', 0.01)
    denoise_nlm_h = opts.get('denoise_nlm_h', 0.01)
    lmg_denoise_eps = opts.get('lmg_denoise_eps', None)
    lmg_denoise_radius = opts.get('lmg_denoise_radius', 2)
    lmg_denoise_type = opts.get('lmg_denoise_type', 'guided')
    lmg_bilateral_sigma_s = opts.get('lmg_bilateral_sigma_s', 2.0)
    lmg_bilateral_sigma_r = opts.get('lmg_bilateral_sigma_r', 0.1)
    lmg_bm3d_sigma = opts.get('lmg_bm3d_sigma', 0.01)
    lmg_nlm_h = opts.get('lmg_nlm_h', 0.01)
    use_soft_threshold = opts.get('use_soft_threshold', False)
    softmax_tau = opts.get('softmax_tau', None)
    kernel_reg_weight = opts.get('kernel_reg_weight', 0.0)

    for _iter in range(xk_iter):
        # ── 1. Latent image estimation ────────────────────────
        if lambda_lmg == 0:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S = L0_LMG_deblur(blur_B_w, k, lambda_lmg, lambda_grad, 2.0,
                              lmg_denoise_eps=lmg_denoise_eps,
                              lmg_denoise_radius=lmg_denoise_radius,
                              lmg_denoise_type=lmg_denoise_type,
                              lmg_bilateral_sigma_s=lmg_bilateral_sigma_s,
                              lmg_bilateral_sigma_r=lmg_bilateral_sigma_r,
                              lmg_bm3d_sigma=lmg_bm3d_sigma,
                              lmg_nlm_h=lmg_nlm_h,
                              use_soft_threshold=use_soft_threshold,
                              softmax_tau=softmax_tau)
            S = S[:H, :W]

        # ── 2. Gradient thresholding ──────────────────────────
        # Symmetric eq: the same transformation applied to blur_B_tmp
        # above is applied to S here, so estimate_psf sees a consistent
        # grad(B) / grad(S) pair.  Original S is not mutated.
        S_for_grad = _apply_kernel_eq(S)

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S_for_grad, max(k.shape), threshold,
            denoise_eps=denoise_eps, denoise_radius=denoise_radius,
            ensemble_denoise=ensemble_denoise,
            denoise_type=denoise_type,
            bilateral_sigma_s=denoise_bilateral_sigma_s,
            bilateral_sigma_r=denoise_bilateral_sigma_r,
            bm3d_sigma=denoise_bm3d_sigma,
            nlm_h=denoise_nlm_h,
        )

        k_prev = k.copy()

        # ── 3. Kernel estimation ──────────────────────────────
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape,
                         kernel_reg_weight=kernel_reg_weight)

        # ── 4. Kernel cleanup ─────────────────────────────────
        # MATLAB: CC = bwconncomp(k, 8)  ->  8-connected components
        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        # ── 5. Parameter updating ─────────────────────────────
        # LMG: lambda floor = 1e-4
        if lambda_lmg != 0:
            lambda_lmg = max(lambda_lmg / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        # ── Callback ──────────────────────────────────────────
        if iteration_callback is not None:
            iteration_callback({
                'iteration': _iter,
                'scale': opts.get('_current_scale', 0),
                'num_scales': opts.get('scales', 1),
                'kernel': k.copy(),
                'image': S,
                'metrics': {
                    'kernel_diff': float(np.linalg.norm(k - k_prev)),
                    'lambda_lmg': lambda_lmg,
                    'lambda_grad': lambda_grad,
                },
            })

    # Final cleanup
    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_lmg, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv  (from blind_deconv.m)  +  helper sub-functions
# ═════════════════════════════════════════════════════════════════════════════

def _init_kernel(minsize):
    """
    Initialise kernel at coarsest level.
    Equivalent to MATLAB init_kernel(minsize).

    MATLAB (1-based):
        k = zeros(minsize, minsize);
        k((minsize-1)/2, (minsize-1)/2:(minsize-1)/2+1) = 1/2;

    Example: minsize=5 -> MATLAB k(2, 2:3)=0.5  ->  Python k[1, 1:3]=0.5
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    c = (minsize - 1) // 2      # MATLAB: (minsize-1)/2, but 1-based row index
    r = c - 1                   # Convert to 0-based row index
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Downsample image by factor *ret* (0 < ret <= 1) using Gaussian
    anti-aliasing followed by bilinear interpolation.
    Equivalent to MATLAB downSmpImC (from Levin's code).

    MATLAB:
        sig = 1/pi * ret
        g0 = [-50:50]*2*pi
        sf = exp(-0.5 * g0.^2 * sig^2), normalised, trimmed at csf>0.05
        I  = conv2(sf, sf', I, 'valid')    — separable Gaussian blur
        [gx,gy] = meshgrid(1:1/ret:cols, 1:1/ret:rows)
        sI = interp2(I, gx, gy, 'bilinear')
    """
    if ret == 1:
        return I.copy()

    sig = (1.0 / np.pi) * ret
    g0 = np.arange(-50, 51, dtype=np.float64) * 2 * np.pi
    sf = np.exp(-0.5 * g0 ** 2 * sig ** 2)
    sf = sf / sf.sum()
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]

    # Separable convolution: MATLAB conv2(sf, sf', I, 'valid')
    sf_row = sf.reshape(1, -1)
    sf_col = sf.reshape(-1, 1)
    I_filtered = convolve2d(I, sf_row, mode='valid')
    I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    # MATLAB: [gx,gy] = meshgrid(1:1/ret:size(I,2), 1:1/ret:size(I,1))
    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    # Convert to 0-based for map_coordinates
    gx_0 = gx_grid - 1.0
    gy_0 = gy_grid - 1.0

    sI = map_coordinates(I_filtered,
                         [gy_0.ravel(), gx_0.ravel()],
                         order=1, mode='nearest')
    sI = sI.reshape(gy_grid.shape)

    return sI


def _fixsize(f, nk1, nk2):
    """
    Adjust array *f* to target size (nk1, nk2) by trimming or padding
    rows/cols on the side with smaller sum.
    Equivalent to MATLAB fixsize.
    """
    k1, k2 = f.shape

    while k1 != nk1 or k2 != nk2:
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]

        if k1 < nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[:k1, :] = f
                f = tf
            else:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[1:k1 + 1, :] = f
                f = tf

        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]

        if k2 < nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, :k2] = f
                f = tf
            else:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, 1:k2 + 1] = f
                f = tf

        k1, k2 = f.shape

    return f


def _resizeKer(k, ret, k1, k2):
    """
    Resize kernel by factor *ret*, then fix to target size (k1, k2).
    Equivalent to MATLAB resizeKer.

    MATLAB: k = imresize(k, ret);  k = max(k,0);  k = fixsize(k,k1,k2);
    MATLAB imresize default = bicubic -> scipy.ndimage.zoom(order=3).
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_lmg, lambda_grad, opts, iteration_callback=None):
    """
    Multi-scale blind deconvolution.
    Equivalent to MATLAB blind_deconv.m.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image
    lambda_lmg  : weight for LMG prior
    lambda_grad : weight for L0 gradient prior
    opts        : dict with keys:
                    'kernel_size'   : int — target kernel size (square, odd)
                    'gamma_correct' : float — gamma correction exponent
                    'xk_iter'       : int — iterations per scale
                    'k_thresh'      : float — final kernel threshold

    Returns
    -------
    kernel         : (kernel_size, kernel_size) estimated kernel
    interim_latent : intermediate latent image from finest scale
    """
    # ── Gamma correction ──────────────────────────────────────
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = np.maximum(y, 0.0) ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    # ── Multi-scale pyramid ───────────────────────────────────
    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)    # ensure odd
    k2list = k1list.copy()                  # square kernels

    # Expose pyramid depth to blind_deconv_main (for callback state)
    opts['scales'] = num_scales

    threshold = None
    ks = None
    interim_latent = None

    # ── Coarse-to-fine loop ───────────────────────────────────
    for s_idx in range(num_scales - 1, -1, -1):
        if s_idx == num_scales - 1:
            ks = _init_kernel(int(k1list[s_idx]))
        else:
            ks = _resizeKer(ks, 1.0 / ret,
                            int(k1list[s_idx]), int(k2list[s_idx]))

        # Downsample image
        cret = retv[s_idx]
        ys = _downSmpImC(y, cret)

        # At coarsest level, estimate initial threshold
        if s_idx == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        # Expose 0-based scale index (0 = finest) for the callback.
        opts['_current_scale'] = s_idx

        ks, lambda_lmg, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_lmg, lambda_grad, threshold, opts,
            iteration_callback=iteration_callback,
        )

        # Centre and clean kernel
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

        # Final scale: threshold small kernel elements
        if s_idx == 0:
            kernel = ks.copy()
            k_thresh = opts.get('k_thresh', 0)
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            kernel = kernel / kernel.sum()

    return kernel, interim_latent


# ═════════════════════════════════════════════════════════════════════════════
# deblur_tv_pmpr  (from PMP deblur_tv_pmpr.m)
# — Non-blind restoration with L0-TV + Patch Minimum Prior
# ═════════════════════════════════════════════════════════════════════════════

def deblur_tv_pmpr(Im, kernel, lambda_pmp, mu, opts):
    """
    Image restoration with L0-TV regularisation and PMP thresholding.
    Ported from PMP (Wen et al., TCSVT 2021) deblur_tv_pmpr.m.

    Used here as an improved non-blind restoration step:
        min_S  ||S*k - B||^2  +  mu * ||nabla S||_0  +  lambda * ||PMP(S)||_0

    Parameters
    ----------
    Im          : (M, N) blurred image (already boundary-wrapped)
    kernel      : (kh, kw) blur kernel
    lambda_pmp  : weight for PMP (L0 intensity) prior
    mu          : weight for L0 gradient prior
    opts        : dict with keys:
                    'r'      : int — patch size for find_min_pixels
                    's'      : int — current scale index (1-based)
                    'scales' : int — total number of scales
                    'pmp_quantile' : float — quantile for PMP (default 0.0)

    Returns
    -------
    S : (M, N) restored image (same size as Im)
    """
    S = Im.copy()
    alphamax = 1e5

    M, N = Im.shape[:2]
    sizeI2D = (M, N)

    otfFh = psf2otf(np.array([[1, -1]], dtype=np.float64), sizeI2D)
    otfFv = psf2otf(np.array([[1], [-1]], dtype=np.float64), sizeI2D)
    otfKER = psf2otf(kernel, sizeI2D)

    denKER = np.abs(otfKER) ** 2
    denGrad = np.abs(otfFh) ** 2 + np.abs(otfFv) ** 2

    Fk_FI = np.conj(otfKER) * fft2(Im)

    alpha = 2.0 * mu
    K = 3
    kappa = 2

    patch_r = opts.get('r', 3)
    current_scale = opts.get('s', 1)
    total_scales = opts.get('scales', 1)
    pmp_quantile = opts.get('pmp_quantile', 0.0)

    while alpha < alphamax:
        for _k in range(K):
            # ── 1. PMP sub-problem ────────────────────────────────
            Z, Md = find_min_pixels(S, patch_r, quantile=pmp_quantile)
            z = Z[Md > 0]

            if current_scale < total_scales / 2.0:
                if z.size > 0:
                    lambdat = min(max(lambda_pmp, np.mean(np.abs(z))), 0.1)
                else:
                    lambdat = lambda_pmp
                Z[np.abs(Z) < lambdat] = 0.0
            else:
                Z = np.sign(Z) * np.maximum(Z - lambda_pmp, 0.0)

            S = S * (1.0 - Md) + Z * Md

            # ── 2. Gradient sub-problem (L0 on gradients) ────────
            Gh = np.concatenate([np.diff(S, n=1, axis=1),
                                 S[:, 0:1] - S[:, -1:]], axis=1)
            Gv = np.concatenate([np.diff(S, n=1, axis=0),
                                 S[0:1, :] - S[-1:, :]], axis=0)

            t = (Gh ** 2 + Gv ** 2) < mu / alpha
            Gh[t] = 0.0
            Gv[t] = 0.0

            # ── 3. Image sub-problem (Fourier domain) ────────────
            gh = np.concatenate([Gh[:, -1:] - Gh[:, 0:1],
                                 -np.diff(Gh, n=1, axis=1)], axis=1)
            gv = np.concatenate([Gv[-1:, :] - Gv[0:1, :],
                                 -np.diff(Gv, n=1, axis=0)], axis=0)

            Fs = (Fk_FI + alpha * fft2(gh + gv)) / (denKER + alpha * denGrad)
            S = np.real(ifft2(Fs))

        alpha = alpha * kappa

    return S


# ═════════════════════════════════════════════════════════════════════════════
# deblurring_adm_aniso  (from deblurring_adm_aniso.m)
# ═════════════════════════════════════════════════════════════════════════════

def _computeDenominator(y, k):
    """
    Compute spectral denominator components for ADM solver.
    Equivalent to MATLAB computeDenominator(y, k).

    Returns
    -------
    Nomin1 : conj(F(k)) * F(y)
    Denom1 : |F(k)|^2
    Denom2 : |F(dx)|^2 + |F(dy)|^2
    """
    sizey = y.shape[:2]
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = (np.abs(psf2otf(np.array([[1, -1]], dtype=np.float64), sizey)) ** 2
              + np.abs(psf2otf(np.array([[1], [-1]], dtype=np.float64), sizey)) ** 2)
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B, k, lambda_tv, alpha):
    """
    TV-l2 deblurring via ADM / Split Bregman with anisotropic TV.
    Equivalent to MATLAB deblurring_adm_aniso.m.

    Parameters
    ----------
    B          : (m, n) blurred image (single channel)
    k          : blur kernel (odd-sized)
    lambda_tv  : regularisation weight
    alpha      : norm exponent (1 = aniso TV with soft threshold)

    Returns
    -------
    I : (m, n) deblurred image
    """
    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, n=1, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                f"deblurring_adm_aniso: alpha={alpha} not implemented; "
                f"only alpha=1 supported"
            )

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, n=1, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, n=1, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)

        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# ringing_artifacts_removal
# (standard non-blind restoration wrapper from Pan's codebase)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring,
                              use_pmp_nonblind=False, pmp_lambda=0.1,
                              pmp_patch_r=3, pmp_quantile=0.0):
    """
    Remove ringing artifacts in non-blind deconvolution.

    Combines TV deconvolution and a second deconvolution method, using a
    bilateral filter on the difference to suppress ringing while
    preserving edges.

    Parameters
    ----------
    y                : (H, W) blurred image
    kernel           : blur kernel
    lambda_tv        : weight for TV deconvolution
    lambda_l0        : weight for L0 deconvolution (used when use_pmp_nonblind=False)
    weight_ring      : ringing suppression weight (0 = TV only)
    use_pmp_nonblind : bool — if True, use PMP deblur_tv_pmpr instead of
                       L0Restoration for the second estimate (default False)
    pmp_lambda       : float — PMP prior weight (only when use_pmp_nonblind=True)
    pmp_patch_r      : int — PMP patch size (only when use_pmp_nonblind=True)
    pmp_quantile     : float — PMP quantile (only when use_pmp_nonblind=True)

    Returns
    -------
    result : (H, W) deblurred image
    """
    H, W = y.shape[:2]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    # TV deblurring
    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    if use_pmp_nonblind:
        # PMP-based non-blind deblurring (better noise robustness)
        pmp_opts = {
            'r': pmp_patch_r,
            's': 1,        # finest scale
            'scales': 1,   # single scale for non-blind
            'pmp_quantile': pmp_quantile,
        }
        Latent_pmp = deblur_tv_pmpr(y_pad, kernel, pmp_lambda, lambda_l0, pmp_opts)
        Latent_second = Latent_pmp[:H, :W]
    else:
        # Original L0 non-blind deblurring
        Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
        Latent_second = Latent_l0[:H, :W]

    diff_img = Latent_tv - Latent_second
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
