"""
solvers.py

Core solver functions for PMP (Patch-wise Minimal Pixels) blind deconvolution.

Ported from MATLAB code by Fei Wen et al.
Reference:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong:
    "A Simple Local Minimal Intensity Prior and An Improved Algorithm
    for Blind Image Deblurring", IEEE TCSVT, 2021.

Original MATLAB code based on Jinshan Pan's DCP framework (CVPR 2016).

Contains:
    estimate_psf           — PSF estimation via conjugate gradient
                             (estimate_psf.m)
    deblur_tv_pmpr         — L0-TV deblurring with PMP thresholding
                             (deblur_tv_pmpr.m)  [KEY PMP CONTRIBUTION]
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
    - conv2(sf,sf',I,'valid')  ->  two separable convolve2d calls
    - interp2(I,gx,gy,'bilinear')  ->  map_coordinates(order=1)
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates, gaussian_filter

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    find_min_pixels,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
)


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


def estimate_psf(blurred_x, blurred_y, latent_x, latent_y, weight, psf_size):
    """
    Estimate blur kernel from gradient images via conjugate gradient.
    Equivalent to MATLAB estimate_psf.m.

    Parameters
    ----------
    blurred_x, blurred_y : gradient images of blurred input
    latent_x, latent_y   : gradient images of latent estimate
    weight               : regularisation weight (gamma)
    psf_size             : (kh, kw) kernel size

    Returns
    -------
    psf : (kh, kw) estimated kernel, thresholded and normalised
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    # b_f = conj(latent_xf) .* blurred_xf + conj(latent_yf) .* blurred_yf
    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(otf2psf(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape[:2],
        'psf_size': psf_size,
        'lambda': weight,
    }

    # MATLAB: psf = ones(psf_size) / prod(psf_size)
    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    # MATLAB: psf(psf < max(psf(:))*0.05) = 0;  psf = psf / sum(psf(:))
    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# deblur_tv_pmpr  (from deblur_tv_pmpr.m)
# — KEY PMP CONTRIBUTION: L0-TV + Patch Minimum Prior
# ═════════════════════════════════════════════════════════════════════════════

def deblur_tv_pmpr(Im, kernel, lambda_pmp, mu, opts):
    """
    Image restoration with L0-TV regularisation and PMP thresholding.
    Equivalent to MATLAB deblur_tv_pmpr.m — the core of the PMP paper.

    Solves (via half-quadratic splitting / ADMM):
        min_S  ||S*k - B||^2  +  mu * ||nabla S||_0  +  lambda * ||PMP(S)||_0

    The outer loop increases alpha from 2*mu to alphamax=1e5 (factor kappa=2).
    Each outer iteration runs K=3 inner iterations of:
        1. PMP sub-problem:  find_min_pixels -> threshold -> update S at mask
        2. Gradient sub-problem:  L0 hard-threshold on circular gradients
        3. Image sub-problem:  Fourier-domain closed-form update

    PMP thresholding strategy depends on scale (opts['s'] vs opts['scales']):
        - Fine scales  (s < scales/2):  hard threshold with adaptive lambdat
        - Coarse scales (s >= scales/2): soft threshold sign(Z)*max(Z-lambda,0)

    Parameters
    ----------
    Im          : (M, N) blurred image (already boundary-wrapped)
    kernel      : (kh, kw) blur kernel
    lambda_pmp  : weight for PMP (L0 intensity) prior
    mu          : weight for L0 gradient prior
    opts        : dict with keys:
                    'r'      : int — patch size for find_min_pixels
                    's'      : int — current scale index (1-based, MATLAB convention)
                    'scales' : int — total number of scales

    Returns
    -------
    S : (M, N) restored image (same size as Im)
    """
    S = Im.copy()
    alphamax = 1e5

    M, N = Im.shape[:2]
    sizeI2D = (M, N)

    # Pre-compute OTFs
    otfFh = psf2otf(np.array([[1, -1]], dtype=np.float64), sizeI2D)
    otfFv = psf2otf(np.array([[1], [-1]], dtype=np.float64), sizeI2D)
    otfKER = psf2otf(kernel, sizeI2D)

    denKER = np.abs(otfKER) ** 2
    denGrad = np.abs(otfFh) ** 2 + np.abs(otfFv) ** 2

    # MATLAB: Fk_FI = conj(otfKER).*fft2(Im)
    Fk_FI = np.conj(otfKER) * fft2(Im)

    alpha = 2.0 * mu
    K = 3
    kappa = 2

    patch_r = opts['r']
    current_scale = opts['s']       # 1-based (MATLAB convention)
    total_scales = opts['scales']
    pmp_quantile = opts.get('pmp_quantile', 0.0)

    while alpha < alphamax:
        for _k in range(K):
            # ── 1. PMP sub-problem ────────────────────────────────
            # MATLAB: [Z, Md] = find_min_pixels(S, opts.r)
            Z, Md = find_min_pixels(S, patch_r, quantile=pmp_quantile)

            # MATLAB: z = Z(Md>0)   — extract PMP values
            z = Z[Md > 0]

            if current_scale < total_scales / 2.0:
                # Fine scales: hard thresholding with adaptive lambdat
                # MATLAB: lambdat = min(max(lambda, mean(abs(z))), 0.1)
                if z.size > 0:
                    lambdat = min(max(lambda_pmp, np.mean(np.abs(z))), 0.1)
                else:
                    lambdat = lambda_pmp
                # MATLAB: Z(abs(Z) < lambdat) = 0
                Z[np.abs(Z) < lambdat] = 0.0
            else:
                # Coarse scales: soft thresholding
                # MATLAB: Z = sign(Z).*max(Z - lambda, 0)
                Z = np.sign(Z) * np.maximum(Z - lambda_pmp, 0.0)

            # MATLAB: S = S.*(1-Md) + Z.*Md
            S = S * (1.0 - Md) + Z * Md

            # ── 2. Gradient sub-problem (L0 on gradients) ────────
            # MATLAB: Gh = [diff(S,1,2), S(:,1,:) - S(:,end,:)]
            Gh = np.concatenate([np.diff(S, n=1, axis=1),
                                 S[:, 0:1] - S[:, -1:]], axis=1)
            # MATLAB: Gv = [diff(S,1,1); S(1,:,:) - S(end,:,:)]
            Gv = np.concatenate([np.diff(S, n=1, axis=0),
                                 S[0:1, :] - S[-1:, :]], axis=0)

            # MATLAB: t = (Gh.^2 + Gv.^2) < mu/alpha;
            #         Gh(t)=0; Gv(t)=0;
            t = (Gh ** 2 + Gv ** 2) < mu / alpha
            Gh[t] = 0.0
            Gv[t] = 0.0

            # ── 3. Image sub-problem (Fourier domain) ────────────
            # MATLAB: gh = [Gh(:,end,:)-Gh(:,1,:), -diff(Gh,1,2)]
            gh = np.concatenate([Gh[:, -1:] - Gh[:, 0:1],
                                 -np.diff(Gh, n=1, axis=1)], axis=1)
            # MATLAB: gv = [Gv(end,:,:)-Gv(1,:,:); -diff(Gv,1,1)]
            gv = np.concatenate([Gv[-1:, :] - Gv[0:1, :],
                                 -np.diff(Gv, n=1, axis=0)], axis=0)

            # MATLAB: Fs = (Fk_FI + alpha*fft2(gh+gv)) ./ (denKER + alpha*denGrad)
            Fs = (Fk_FI + alpha * fft2(gh + gv)) / (denKER + alpha * denGrad)
            S = np.real(ifft2(Fs))

        alpha = alpha * kappa

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

    # MATLAB: [N,M,D] = size(Im)  — for 2D, D=1
    N, M = Im.shape[:2]
    sizeI2D = (N, M)

    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    # MATLAB: Normin1 = conj(KER).*fft2(S)
    Normin1 = np.conj(KER) * fft2(S)

    beta = 2 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        # Circular differences
        # MATLAB: h = [diff(S,1,2), S(:,1,:) - S(:,end,:)]
        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1] - S[:, -1:]], axis=1)
        # MATLAB: v = [diff(S,1,1); S(1,:,:) - S(end,:,:)]
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :] - S[-1:, :]], axis=0)

        # MATLAB: t = (h.^2+v.^2) < lambda/beta  (D==1 case)
        t = (h ** 2 + v ** 2) < lambda_grad / beta
        h[t] = 0.0
        v[t] = 0.0

        # Divergence (backward differences)
        # MATLAB: Normin2 = [h(:,end,:)-h(:,1,:), -diff(h,1,2)]
        Normin2_val = np.concatenate([h[:, -1:] - h[:, 0:1],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        # MATLAB: Normin2 = Normin2 + [v(end,:,:)-v(1,:,:); -diff(v,1,1)]
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :] - v[0:1, :],
             -np.diff(v, n=1, axis=0)], axis=0)

        FS = (Normin1 + beta * fft2(Normin2_val)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa

    # MATLAB: S = S(1:H, 1:W, :)
    S = S[:H_orig, :W_orig]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main  (from blind_deconv_main.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main(blur_B, k, lambda_pmp, lambda_grad, threshold, opts,
                      blind_denoise_fn=None, iteration_callback=None):
    """
    Single-scale blind deconvolution.
    Equivalent to MATLAB blind_deconv_main.m.

    Alternates between:
        1. Latent image estimation (deblur_tv_pmpr or L0Restoration)
        2. Gradient thresholding (threshold_pxpy_v1)
        3. Kernel estimation (estimate_psf)
        4. Kernel cleanup (connected components, normalisation)

    Parameters
    ----------
    blur_B      : (H, W) blurred image
    k           : (kh, kw) current kernel estimate
    lambda_pmp  : weight for PMP (L0 intensity) prior
    lambda_grad : weight for L0 gradient prior
    threshold   : gradient threshold (updated per iteration)
    opts        : dict with 'xk_iter', 'r', 's', 'scales'
    blind_denoise_fn : callable or None
        If not None, called as ``blind_denoise_fn(S)`` after latent image
        estimation and before gradient thresholding + kernel estimation.
        Returns denoised S used for computing gradients for kernel step.

    Returns
    -------
    k           : updated kernel
    lambda_pmp  : updated lambda_pmp
    lambda_grad : updated lambda_grad
    S           : intermediate latent image

    PMP-specific differences from DCP:
        - Calls deblur_tv_pmpr (not L0Deblur_dark_channel)
        - lambda_pmp floor = 1e-2 (DCP uses 1e-4)
        - No clipping of S (DCP clips to [0,1])
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
    # MATLAB: blur_B_tmp = blur_B_w(1:H, 1:W, :)
    blur_B_tmp = blur_B_w[:H, :W]

    # MATLAB: Bx = conv2(blur_B_tmp, dx, 'valid'); By = conv2(blur_B_tmp, dy, 'valid')
    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    # Pre-smooth blurred-image gradients to suppress noise
    grad_smooth_sigma = opts.get('grad_smooth_sigma', None)
    if grad_smooth_sigma is not None and grad_smooth_sigma > 0:
        Bx = gaussian_filter(Bx, sigma=grad_smooth_sigma)
        By = gaussian_filter(By, sigma=grad_smooth_sigma)

    xk_iter = opts.get('xk_iter', 5)
    denoise_eps = opts.get('denoise_eps', None)
    denoise_radius = opts.get('denoise_radius', 2)
    ensemble_denoise = opts.get('ensemble_denoise', False)
    estimate_noise = opts.get('estimate_noise', False)

    # ── Interpretation 2: adaptive grad_smooth from noise estimate ─────
    noise_sigma_mult = opts.get('noise_sigma_mult', 10.0)
    if estimate_noise and denoise_eps is not None and denoise_eps > 0:
        from .utils import guided_filter
        d1 = guided_filter(blur_B_tmp, blur_B_tmp, denoise_radius, denoise_eps)
        d2 = guided_filter(blur_B_tmp, blur_B_tmp, denoise_radius + 1, denoise_eps * 0.5)
        sig1 = np.std(blur_B_tmp - d1)
        sig2 = np.std(blur_B_tmp - d2)
        sigma_est = (sig1 + sig2) / 2.0
        # Auto-set grad_smooth_sigma if not manually specified
        if grad_smooth_sigma is None or grad_smooth_sigma <= 0:
            grad_smooth_sigma = sigma_est * noise_sigma_mult
            Bx = gaussian_filter(Bx, sigma=grad_smooth_sigma)
            By = gaussian_filter(By, sigma=grad_smooth_sigma)

    for _iter in range(xk_iter):
        # ── 1. Latent image estimation ────────────────────────
        if lambda_pmp != 0:
            # MATLAB: S = deblur_tv_pmpr(blur_B_w, k, lambda, lambda_grad, opts)
            S = deblur_tv_pmpr(blur_B_w, k, lambda_pmp, lambda_grad, opts)
            # MATLAB: S = S(1:H, 1:W, :)
            S = S[:H, :W]
        else:
            # Fallback: L0 deblurring without PMP
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        # ── 1b. Blind denoise (optional) ───────────────────
        S_for_kernel = blind_denoise_fn(S) if blind_denoise_fn is not None else S

        # ── 2. Gradient thresholding ──────────────────────────
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S_for_kernel, max(k.shape), threshold,
            denoise_eps=denoise_eps, denoise_radius=denoise_radius,
            ensemble_denoise=ensemble_denoise
        )

        k_prev = k.copy()

        # ── 3. Kernel estimation ──────────────────────────────
        # MATLAB: k = estimate_psf(Bx, By, latent_x, latent_y, 2, size(k_prev))
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

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
        # PMP: lambda floor = 1e-2 (DCP uses 1e-4)
        if lambda_pmp != 0:
            lambda_pmp = max(lambda_pmp / 1.1, 1e-2)
        # lambda_grad floor = 1e-4
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        # NOTE: No clip of S  (MATLAB code has it commented out)

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
                    'lambda_pmp': lambda_pmp,
                    'lambda_grad': lambda_grad,
                },
            })

    # Final cleanup
    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_pmp, lambda_grad, S


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
    # conv2(hcol, hrow, A) first convolves columns with hcol, then rows with hrow.
    # Since sf is the same for both directions, order doesn't matter.
    sf_row = sf.reshape(1, -1)
    sf_col = sf.reshape(-1, 1)
    I_filtered = convolve2d(I, sf_row, mode='valid')
    I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    # MATLAB: [gx,gy] = meshgrid(1:1/ret:size(I,2), 1:1/ret:size(I,1))
    # I here is the post-convolution (smaller) result
    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    # MATLAB grid is 1-based: 1 to cols (at most) in steps of 1/ret
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    # Convert to 0-based for map_coordinates
    gx_0 = gx_grid - 1.0   # column coords (0-based)
    gy_0 = gy_grid - 1.0   # row coords (0-based)

    # MATLAB interp2(I, gx, gy, 'bilinear')
    # map_coordinates: [row_coords, col_coords]
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
            s = f.sum(axis=1)  # row sums
            if s[0] < s[-1]:
                f = f[1:, :]       # remove first row (less energy)
            else:
                f = f[:-1, :]      # remove last row

        if k1 < nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                # MATLAB: tf(1:k1,:) = f  — pad zero row at bottom
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[:k1, :] = f
                f = tf
            else:
                # MATLAB: tf(2:k1+1,:) = f  — pad zero row at top
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[1:k1 + 1, :] = f
                f = tf

        if k2 > nk2:
            s = f.sum(axis=0)  # col sums
            if s[0] < s[-1]:
                f = f[:, 1:]       # remove first col
            else:
                f = f[:, :-1]      # remove last col

        if k2 < nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                # Pad zero col at right
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, :k2] = f
                f = tf
            else:
                # Pad zero col at left
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
    k = zoom(k, ret, order=3)       # bicubic to match MATLAB imresize
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_pmp, lambda_grad, opts, patch_r=None,
                 blind_denoise_fn=None, iteration_callback=None):
    """
    Multi-scale blind deconvolution.
    Equivalent to MATLAB blind_deconv.m.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image
    lambda_pmp  : weight for PMP (L0 intensity) prior
    lambda_grad : weight for L0 gradient prior
    opts        : dict with keys:
                    'kernel_size'   : int — target kernel size (square, odd)
                    'gamma_correct' : float — gamma correction exponent
                    'xk_iter'       : int — iterations per scale
                    'k_thresh'      : float — final kernel threshold
    patch_r     : int or None — patch size for PMP.
                  If None: floor(0.025 * mean(image_size))
    blind_denoise_fn : callable or None
        If not None, called as ``blind_denoise_fn(S)`` inside each
        blind_deconv_main iteration to denoise the latent image before
        gradient thresholding and kernel estimation.

    Returns
    -------
    kernel         : (kernel_size, kernel_size) estimated kernel
    interim_latent : intermediate latent image from finest scale
    """
    # ── Gamma correction ──────────────────────────────────────
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = y ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    # ── Multi-scale pyramid ───────────────────────────────────
    ret = np.sqrt(0.5)
    # MATLAB: maxitr = max(floor(log(5/min(opts.kernel_size))/log(ret)), 0)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    # MATLAB: retv = ret.^[0:maxitr]
    retv = ret ** np.arange(0, maxitr + 1)
    # MATLAB: k1list = ceil(kernel_size * retv);  make odd
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)    # ensure odd
    k2list = k1list.copy()                  # square kernels

    # ── PMP patch size ────────────────────────────────────────
    # MATLAB: opts.r = floor(0.025 * mean(size(y)))
    if patch_r is None:
        opts['r'] = max(1, int(np.floor(0.025 * np.mean(y.shape[:2]))))
    else:
        opts['r'] = int(patch_r)

    opts['scales'] = num_scales

    threshold = None     # will be estimated at coarsest scale
    ks = None
    interim_latent = None

    # ── Coarse-to-fine loop ───────────────────────────────────
    # MATLAB: for s = num_scales:-1:1   (1-based, coarsest to finest)
    # Python: s_idx from num_scales-1 down to 0
    for s_idx in range(num_scales - 1, -1, -1):
        # MATLAB-equivalent 1-based scale index
        s_matlab = s_idx + 1

        if s_idx == num_scales - 1:
            # Coarsest level: initialise kernel
            ks = _init_kernel(int(k1list[s_idx]))
        else:
            # Upsample kernel from previous level
            ks = _resizeKer(ks, 1.0 / ret,
                            int(k1list[s_idx]), int(k2list[s_idx]))

        # Downsample image
        cret = retv[s_idx]
        ys = _downSmpImC(y, cret)

        # At coarsest level, estimate initial threshold
        if s_idx == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        # Pass 1-based scale index so deblur_tv_pmpr can check
        # opts.s < opts.scales/2  (MATLAB convention)
        opts['s'] = s_matlab
        opts['_current_scale'] = s_idx  # 0 = finest

        ks, lambda_pmp, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_pmp, lambda_grad, threshold, opts,
            blind_denoise_fn=blind_denoise_fn,
            iteration_callback=iteration_callback,
        )

        # Centre and clean kernel
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

        # Final scale: threshold small kernel elements
        # MATLAB: if (s == 1)   — finest scale
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

    # Circular differences
    # MATLAB: Ix = [diff(I,1,2), I(:,1)-I(:,n)]
    Ix = np.concatenate([np.diff(I, n=1, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    # MATLAB: Iy = [diff(I,1,1); I(1,:)-I(m,:)]
    Iy = np.concatenate([np.diff(I, n=1, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            # Soft-thresholding (anisotropic TV)
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                f"deblurring_adm_aniso: alpha={alpha} not implemented; "
                f"only alpha=1 supported"
            )

        # Divergence
        # MATLAB: Wxx = [Wx(:,n)-Wx(:,1), -diff(Wx,1,2)]
        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        # MATLAB: Wxx = Wxx + [Wy(m,:)-Wy(1,:); -diff(Wy,1,1)]
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, n=1, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        # Update gradients
        Ix = np.concatenate([np.diff(I, n=1, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)

        # MATLAB: beta = beta/2
        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# ringing_artifacts_removal
# (standard non-blind restoration wrapper from Pan's codebase)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    """
    Remove ringing artifacts in non-blind deconvolution.

    Combines TV deconvolution and L0 deconvolution, using a bilateral
    filter on the difference to suppress ringing while preserving edges.

    Parameters
    ----------
    y           : (H, W) blurred image
    kernel      : blur kernel
    lambda_tv   : weight for TV deconvolution
    lambda_l0   : weight for L0 deconvolution
    weight_ring : ringing suppression weight (0 = TV only)

    Returns
    -------
    result : (H, W) deblurred image
    """
    H, W = y.shape[:2]

    # Wrap boundaries for TV deconvolution
    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    # TV deblurring
    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    # MATLAB: Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2);
    # Note: MATLAB passes y_pad (already wrapped), and L0Restoration wraps again
    # internally. We must replicate this double-wrapping to match MATLAB exactly.
    Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W]

    # Bilateral filter on the difference
    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
