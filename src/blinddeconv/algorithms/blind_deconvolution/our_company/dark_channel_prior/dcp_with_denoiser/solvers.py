"""
solvers.py

Core solver functions for DCP (Dark Channel Prior) blind deconvolution.

Ported from MATLAB code by Jinshan Pan (CVPR 2016).
Reference:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR 2016.

Contains:
    estimate_psf           — PSF estimation via conjugate gradient (estimate_psf.m)
    L0Deblur_dark_channel  — L0 deblurring with dark-channel + gradient priors
                             (L0Deblur_dark_chanel.m)
    L0Restoration          — L0 deblurring with gradient prior only
                             (L0Restoration.m)
    blind_deconv_main      — single-scale blind deconvolution loop
                             (blind_deconv_main.m)
    blind_deconv           — multi-scale coarse-to-fine blind deconvolution
                             (blind_deconv.m)
    deblurring_adm_aniso   — TV-ℓ² deblurring via ADM / Split Bregman
                             (deblurring_adm_aniso.m)
    ringing_artifacts_removal — artifact removal post-processing
                             (ringing_artifacts_removal.m)

MATLAB → Python notes:
    - MATLAB conv2(A,B,'valid') → scipy.signal.convolve2d(A,B,'valid')
      Both do true convolution (kernel flip).
    - MATLAB diff(S,1,2) → np.diff(S, n=1, axis=1)
      MATLAB dim 2 = Python axis 1 (columns).
      MATLAB diff(S,1,1) → np.diff(S, n=1, axis=0).
    - MATLAB S(:,1,:)-S(:,end,:) → S[:,0,:]-S[:,-1,:]
    - MATLAB repmat(A,[1,1,D]) → np.tile(A, (1,1,D)) or np.broadcast_to
    - MATLAB fft2/ifft2/conj → np.fft.fft2/ifft2/conj
    - MATLAB bwconncomp(k,8) → scipy.ndimage.label(k, structure=np.ones((3,3)))
    - MATLAB imresize(k,ret) → scipy.ndimage.zoom (or cv2.resize)
    - MATLAB interp2(I,gx,gy,'bilinear') → scipy.ndimage.map_coordinates
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
    dark_channel,
    assign_dark_channel_to_pixel,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
    graythresh,
    estimate_noise_sigma,
)


# ═════════════════════════════════════════════════════════════════════════════
# estimate_psf  (from estimate_psf.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_Ax(x, p):
    """
    Matrix-vector product for the PSF estimation CG system.
    MATLAB: y = otf2psf(p.m .* psf2otf(x, p.img_size), p.psf_size) + p.lambda * x
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
    blurred_x, blurred_y : (M', N') gradient images of blurred input
    latent_x, latent_y   : (M', N') gradient images of latent estimate
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

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf = psf / psf.sum()
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# L0Deblur_dark_channel  (from L0Deblur_dark_chanel.m)
# ═════════════════════════════════════════════════════════════════════════════

def L0Deblur_dark_channel(Im, kernel, lambda_dark, wei_grad, kappa=2.0,
                          dc_quantile=0.0, nsr=0.0):
    """
    Image restoration with L0 regularised intensity (dark channel) and
    gradient prior.

    Equivalent to MATLAB L0Deblur_dark_chanel.m.

    Parameters
    ----------
    Im      : (N, M) or (N, M, D) blurred image (already boundary-wrapped)
    kernel  : (kh, kw) blur kernel
    lambda_dark : weight for L0 intensity prior
    wei_grad    : weight for L0 gradient prior
    kappa       : ADM update ratio (default 2.0)
    dc_quantile : float — quantile for dark channel (0 = strict min).
    nsr         : float — Wiener-style noise-to-signal term added to
                  the FFT denominator.  Stabilises inversion when noise
                  is present.  0 = disabled.

    Returns
    -------
    S : (N, M) or (N, M, D) restored image
    """
    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
        S = S[:, :, np.newaxis]
        Im = Im[:, :, np.newaxis]
        squeeze_out = True
    else:
        N, M, D = S.shape
        squeeze_out = False

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    # Always expand to 3D for consistent broadcasting with S (N,M,D)
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    # Dark-channel pixel sub-problem
    dark_r = 35  # fixed size in MATLAB code
    # mybeta_pixel = lambda_dark / graythresh(S ** 2)
    # MATLAB: graythresh((S).^2) — applied to entire (possibly 3D) array
    mybeta_pixel = lambda_dark / graythresh(S ** 2)
    maxbeta_pixel = 2 ** 3

    while mybeta_pixel < maxbeta_pixel:
        # Dark channel: J is always 2D (N, M), J_idx is (N, M)
        J, J_idx = dark_channel(S, dark_r, dc_quantile)
        u = J.copy()  # 2D dark channel

        # MATLAB: for D==1, t = u.^2 < lambda/mybeta_pixel (u is 2D)
        #         for D>1,  t = sum(u.^2,3) < lambda/mybeta_pixel (u is 2D dark channel)
        # u is always 2D from dark_channel, so:
        t = u ** 2 < lambda_dark / mybeta_pixel
        u[t] = 0.0

        # assign_dark_channel_to_pixel returns (N,M,D) matching S
        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        # Gradient sub-problem (inner loop)
        beta = 2 * wei_grad
        while beta < betamax:
            Denormin = Den_KER + beta * Denormin2 + mybeta_pixel + nsr

            # MATLAB: h = [diff(S,1,2), S(:,1,:) - S(:,end,:)]
            h = np.concatenate([np.diff(S, n=1, axis=1),
                                S[:, 0:1, :] - S[:, -1:, :]], axis=1)
            # MATLAB: v = [diff(S,1,1); S(1,:,:) - S(end,:,:)]
            v = np.concatenate([np.diff(S, n=1, axis=0),
                                S[0:1, :, :] - S[-1:, :, :]], axis=0)

            if D == 1:
                t = (h ** 2 + v ** 2)[:, :, 0] < wei_grad / beta
                t = t[:, :, np.newaxis]
            else:
                t = np.sum(h ** 2 + v ** 2, axis=2) < wei_grad / beta
                t = np.tile(t[:, :, np.newaxis], (1, 1, D))
            h[t] = 0.0
            v[t] = 0.0

            # MATLAB: Normin2 = [h(:,end,:)-h(:,1,:), -diff(h,1,2)]
            Normin2_val = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                          -np.diff(h, n=1, axis=1)], axis=1)
            # MATLAB: Normin2 = Normin2 + [v(end,:,:)-v(1,:,:); -diff(v,1,1)]
            Normin2_val = Normin2_val + np.concatenate(
                [v[-1:, :, :] - v[0:1, :, :],
                 -np.diff(v, n=1, axis=0)], axis=0)

            FS = (Normin1 + beta * fft2(Normin2_val, axes=(0, 1))
                  + mybeta_pixel * fft2(u, axes=(0, 1))) / Denormin
            S = np.real(ifft2(FS, axes=(0, 1)))

            beta = beta * kappa
            if wei_grad == 0:
                break

        mybeta_pixel = mybeta_pixel * kappa

    if squeeze_out:
        S = S[:, :, 0]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# L0Restoration  (from L0Restoration.m)
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0, nsr=0.0):
    """
    Image restoration with L0 gradient prior.

    Equivalent to MATLAB L0Restoration.m.

    Parameters
    ----------
    Im      : (H, W) or (H, W, D) blurred image (original size, NOT wrapped)
    kernel  : (kh, kw) blur kernel
    lambda_grad : weight for the L0 gradient prior
    kappa       : ADM update ratio (default 2.0)
    nsr         : float — Wiener-style noise-to-signal term added to
                  the FFT denominator.  0 = disabled.

    Returns
    -------
    S : (H, W) or (H, W, D) restored image cropped to original size
    """
    orig_ndim = Im.ndim
    H_orig, W_orig = Im.shape[0], Im.shape[1]

    # Pad image boundaries
    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1
    )
    Im = wrap_boundary_liu(Im, tuple(target_size))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
        S = S[:, :, np.newaxis]
        Im = Im[:, :, np.newaxis]
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2

    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    # Always expand to 3D for consistent broadcasting with S (N,M,D)
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    beta = 2 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2 + nsr

        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :, :] - S[-1:, :, :]], axis=0)

        if D == 1:
            t = (h ** 2 + v ** 2)[:, :, 0] < lambda_grad / beta
            t = t[:, :, np.newaxis]
        else:
            t = np.sum(h ** 2 + v ** 2, axis=2) < lambda_grad / beta
            t = np.tile(t[:, :, np.newaxis], (1, 1, D))
        h[t] = 0.0
        v[t] = 0.0

        Normin2_val = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :, :] - v[0:1, :, :],
             -np.diff(v, n=1, axis=0)], axis=0)

        FS = (Normin1 + beta * fft2(Normin2_val, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))
        beta = beta * kappa

    # Crop back to original size
    S = S[:H_orig, :W_orig, :]
    if orig_ndim == 2:
        S = S[:, :, 0]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main  (from blind_deconv_main.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main(blur_B, k, lambda_dark, lambda_grad, threshold, opts):
    """
    Single-scale blind deconvolution.

    Equivalent to MATLAB blind_deconv_main.m.

    Parameters
    ----------
    blur_B       : (H, W) or (H, W, D) blurred image
    k            : (kh, kw) current kernel estimate
    lambda_dark  : weight for L0 intensity prior
    lambda_grad  : weight for L0 gradient prior
    threshold    : gradient threshold (updated per iteration)
    opts         : dict with 'xk_iter' (int) — number of iterations

    Returns
    -------
    k            : updated kernel
    lambda_dark  : updated lambda_dark
    lambda_grad  : updated lambda_grad
    S            : intermediate latent image
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H = blur_B.shape[0]
    W = blur_B.shape[1]

    # Boundary wrapping for FFT
    target_size = opt_fft_size(
        np.array([H, W]) + np.array(k.shape[:2]) - 1
    )
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target_size))
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
    threshold_boost = opts.get('threshold_boost', 1.0)
    residual_beta = opts.get('residual_beta', None)
    gradient_confidence_tau = opts.get('gradient_confidence_tau', None)
    k_iter_clean = opts.get('k_iter_clean', None)
    noise_adapt = opts.get('noise_adapt', None)
    dc_quantile = opts.get('dc_quantile', 0.0)
    nsr = opts.get('nsr', 0.0)
    s_res = np.zeros((H, W), dtype=np.float64)

    # (D) Estimate noise level once for noise-adaptive threshold floor
    noise_sigma = None
    if noise_adapt is not None and noise_adapt > 0:
        noise_sigma = estimate_noise_sigma(blur_B)

    for _iter in range(xk_iter):
        # Latent image estimation (with noise-corrected observation)
        if lambda_dark != 0:
            if residual_beta and _iter > 0:
                blur_eff = blur_B_w.copy()
                blur_eff[:H, :W] += s_res
            else:
                blur_eff = blur_B_w
            S = L0Deblur_dark_channel(blur_eff, k, lambda_dark, lambda_grad, 2.0,
                                      dc_quantile=dc_quantile, nsr=nsr)
            S = S[:H, :W]
        else:
            if residual_beta and _iter > 0:
                S = L0Restoration(blur_B + s_res, k, lambda_grad, 2.0, nsr=nsr)
            else:
                S = L0Restoration(blur_B, k, lambda_grad, 2.0, nsr=nsr)

        # Gradient thresholding
        # MATLAB (commented-out): bilateral_filter denoising before gradient
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold,
            denoise_eps=denoise_eps, denoise_radius=denoise_radius
        )

        # Dynamic threshold boost: on early iterations use a higher
        # threshold so only the strongest gradients guide kernel
        # estimation, linearly decaying to 1.0 on the last iteration.
        if threshold_boost > 1.0 and xk_iter > 1:
            # iter 0 → full boost, iter xk_iter-1 → boost = 1.0
            t = _iter / (xk_iter - 1)  # 0..1
            current_boost = threshold_boost * (1.0 - t) + 1.0 * t
            boosted_threshold = threshold * current_boost
            m_boost = (latent_x ** 2 + latent_y ** 2) < boosted_threshold
            latent_x[m_boost] = 0.0
            latent_y[m_boost] = 0.0

        # (D) Noise-adaptive threshold floor: if estimated noise power
        #     exceeds the auto-threshold, raise threshold to noise floor
        if noise_adapt is not None and noise_sigma is not None:
            noise_thr = noise_adapt * noise_sigma ** 2
            if noise_thr > threshold:
                extra_mask = (latent_x ** 2 + latent_y ** 2) < noise_thr
                latent_x[extra_mask] = 0.0
                latent_y[extra_mask] = 0.0

        # (A) Gradient confidence weighting: strong edges get weight ≈ 1,
        #     weak residual gradients get weight → 0
        if gradient_confidence_tau is not None and gradient_confidence_tau > 0:
            mag = np.sqrt(latent_x ** 2 + latent_y ** 2)
            conf = mag / (mag + gradient_confidence_tau)
            latent_x = latent_x * conf
            latent_y = latent_y * conf

        k_prev = k.copy()

        # Kernel estimation
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

        # Prune isolated noise in kernel (MATLAB: bwconncomp + filtering)
        # scipy.ndimage.label with 8-connectivity = structure of ones(3,3)
        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        # (C) Per-iteration kernel cleaning: zero values below max/k_iter_clean
        if k_iter_clean is not None and k_iter_clean > 1:
            k[k < k.max() / k_iter_clean] = 0.0
            k[k < 0] = 0.0
            k = k / k.sum()

        # Parameter updating
        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        S = np.clip(S, 0.0, 1.0)

        # Update noise residual via soft-thresholding (Fang et al.)
        if residual_beta and residual_beta > 0:
            Sk = np.real(ifft2(psf2otf(k, (H, W)) * fft2(S)))
            r = Sk - blur_B_tmp
            s_res = np.sign(r) * np.maximum(np.abs(r) - residual_beta, 0.0)

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv  (from blind_deconv.m)  +  helper sub-functions
# ═════════════════════════════════════════════════════════════════════════════

def _init_kernel(minsize):
    """
    Initialise kernel at coarsest level.
    MATLAB: k = zeros(minsize); k((minsize-1)/2, (minsize-1)/2:(minsize-1)/2+1) = 1/2;
    Note: MATLAB 1-based indexing.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    # MATLAB row (minsize-1)/2 (1-based) → Python (minsize-1)//2 - 1
    # MATLAB: k((ms-1)/2, (ms-1)/2:(ms-1)/2+1) = 0.5
    # With ms=5: k(2, 2:3) = 0.5  (1-based) → k[1, 1:3] (0-based)
    c = (minsize - 1) // 2  # MATLAB: (minsize-1)/2 in 1-based → c-1 in 0-based
    r = c - 1  # row index (0-based), since MATLAB row is c (1-based) = c-1 (0-based)
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Downsample image by factor *ret* (0 < ret ≤ 1) using Gaussian
    anti-aliasing followed by bilinear interpolation.
    Equivalent to MATLAB downSmpImC (from Levin's code).

    MATLAB notes:
        sig = 1/pi * ret
        Gaussian filter = exp(-0.5 * g0^2 * sig^2) over g0 = [-50:50]*2*pi
        I = conv2(sf, sf', I, 'valid')
        Then interp2 on sub-sampled grid.
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

    # Separable convolution: conv2(sf, sf', I, 'valid')
    # MATLAB conv2(h_row, h_col, A, 'valid') = convolve A first with row vec,
    # then with col vec, keeping 'valid' region.
    # scipy convolve2d with 2D kernel sf_row * sf_col
    sf_row = sf.reshape(1, -1)
    sf_col = sf.reshape(-1, 1)
    if I.ndim == 3:
        # per-channel
        channels = []
        for c in range(I.shape[2]):
            tmp = convolve2d(I[:, :, c], sf_row, mode='valid')
            tmp = convolve2d(tmp, sf_col, mode='valid')
            channels.append(tmp)
        I_filtered = np.stack(channels, axis=2)
    else:
        I_filtered = convolve2d(I, sf_row, mode='valid')
        I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    # MATLAB: [gx,gy] = meshgrid(1:1/ret:size(I,2), 1:1/ret:size(I,1))
    # I here is post-convolution I_filtered
    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    # MATLAB 1-based grid: 1 to cols, step 1/ret
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    # Convert to 0-based for map_coordinates
    gx_0 = gx_grid - 1.0
    gy_0 = gy_grid - 1.0

    if I_filtered.ndim == 3:
        channels = []
        for c in range(I_filtered.shape[2]):
            sI_ch = map_coordinates(I_filtered[:, :, c],
                                    [gy_0.ravel(), gx_0.ravel()],
                                    order=1, mode='nearest')
            channels.append(sI_ch.reshape(gy_grid.shape))
        sI = np.stack(channels, axis=2)
    else:
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
            s = f.sum(axis=0)  # col sums
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
    MATLAB: k = imresize(k, ret); k = max(k,0); k = fixsize(k,k1,k2); normalise.

    scipy.ndimage.zoom is used as a MATLAB imresize equivalent (bilinear/bicubic).
    MATLAB imresize default uses bicubic interpolation.
    """
    k = zoom(k, ret, order=3)  # bicubic to match MATLAB imresize default
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_dark, lambda_grad, opts):
    """
    Multi-scale blind deconvolution.

    Equivalent to MATLAB blind_deconv.m.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image
    lambda_dark : weight for L0 intensity prior
    lambda_grad : weight for L0 gradient prior
    opts        : dict with keys:
                    'kernel_size' : int — target kernel size (square, odd)
                    'gamma_correct' : float — gamma correction exponent
                    'xk_iter' : int — iterations per scale
                    'k_thresh' : float — final kernel threshold
                                 (>0: threshold at max/k_thresh; <=0: just clip negatives)

    Returns
    -------
    kernel         : (kernel_size, kernel_size) estimated kernel
    interim_latent : intermediate latent image from finest scale
    """
    # Gamma correction
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = y ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    # Multi-scale pyramid
    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)  # ensure odd
    k2list = k1list.copy()  # square kernels

    threshold = None  # will be estimated at coarsest scale

    ks = None
    interim_latent = None

    for s_idx in range(num_scales - 1, -1, -1):  # MATLAB: num_scales:-1:1
        s = s_idx  # 0-based index

        if s == num_scales - 1:
            # Coarsest level: initialise kernel
            ks = _init_kernel(int(k1list[s]))
        else:
            # Upsample kernel from previous level
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[s]), int(k2list[s]))

        cret = retv[s]
        ys = _downSmpImC(y, cret)

        # At coarsest level, estimate initial threshold
        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        # (C) Pyramid-adaptive denoise_eps: scale denoising strength
        #     with pyramid level.  Coarsest (downsampled) images have
        #     less noise → less denoising needed.  Finest = full eps.
        pyramid_adapt = opts.get('pyramid_adapt_eps', False)
        base_eps = opts.get('denoise_eps', None)
        if pyramid_adapt and base_eps is not None and num_scales > 1:
            # s_idx: num_scales-1 = coarsest, 0 = finest
            # scale factor: 0.3 at coarsest → 1.0 at finest
            scale = 0.3 + 0.7 * (1.0 - s_idx / (num_scales - 1))
            opts_level = dict(opts)
            opts_level['denoise_eps'] = base_eps * scale
        else:
            opts_level = opts

        ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_dark, lambda_grad, threshold, opts_level
        )

        # Centre and clean kernel
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

        # Final scale thresholding
        if s == 0:
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
    MATLAB: [Nomin1, Denom1, Denom2] = computeDenominator(y, k)

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


def deblurring_adm_aniso(B, k, lambda_tv, alpha, noise_lambda=0.0):
    """
    TV-ℓ² deblurring via ADM / Split Bregman with anisotropic TV.

    Equivalent to MATLAB deblurring_adm_aniso.m.

    Parameters
    ----------
    B      : (m, n) blurred image (single channel)
    k      : blur kernel (odd-sized)
    lambda_tv : regularisation weight
    alpha     : norm exponent (1 = anisotropic TV with soft threshold)
    noise_lambda : float — L1 weight for the noise residual variable s.
                   When > 0, solves  min ||I*k - B - s||² + λ_tv TV(I) + μ||s||₁
                   absorbing noise into s via soft-thresholding (Fang et al.).
                   0 = disabled (original behaviour).  Default 0.

    Returns
    -------
    I : (m, n) deblurred image
    """
    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    # Noise residual (Fang et al.)
    if noise_lambda > 0:
        otfk = psf2otf(k, (m, n))
        conj_otfk = np.conj(otfk)
        s = np.zeros((m, n), dtype=np.float64)

    # Circular differences
    Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            # Soft-thresholding (anisotropic TV)
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            # For alpha != 1 (hyper-Laplacian), MATLAB calls solve_image
            # which uses a lookup table.  For DCP, alpha=1 is always used
            # in ringing_artifacts_removal, so we raise here for safety.
            raise NotImplementedError(
                f"deblurring_adm_aniso: alpha={alpha} not implemented; only alpha=1 supported"
            )

        # MATLAB: Wxx = [Wx(:,end)-Wx(:,1), -diff(Wx,1,2)]
        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        # MATLAB: Wxx = Wxx + [Wy(end,:)-Wy(1,:); -diff(Wy,1,1)]
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, n=1, axis=0)], axis=0)

        # Data fidelity (with noise-corrected observation if enabled)
        Nomin1_use = conj_otfk * fft2(B + s) if noise_lambda > 0 else Nomin1
        Fyout = (Nomin1_use + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        # Update noise residual s (Fang et al.)
        if noise_lambda > 0:
            Ik = np.real(ifft2(otfk * fft2(I)))
            r = Ik - B
            s = np.sign(r) * np.maximum(np.abs(r) - noise_lambda, 0.0)

        # Update gradients
        Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# ringing_artifacts_removal  (from ringing_artifacts_removal.m)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring,
                              noise_lambda=0.0):
    """
    Remove ringing artifacts in non-blind deconvolution.

    Equivalent to MATLAB ringing_artifacts_removal.m.

    Parameters
    ----------
    y           : (H, W) or (H, W, D) blurred image
    kernel      : blur kernel
    lambda_tv   : weight for TV deconvolution [1e-3, 1e-2]
    lambda_l0   : weight for L0 deconvolution, typically ~2e-3 [1e-4, 2e-3]
    weight_ring : ringing suppression weight (0 = no suppression)
    noise_lambda : float — L1 weight for noise residual in TV solver.
                   See deblurring_adm_aniso.  0 = disabled.  Default 0.

    Returns
    -------
    result : (H, W) or (H, W, D) deblurred image
    """
    orig_ndim = y.ndim
    H, W = y.shape[0], y.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    # TV deblurring per channel
    if y_pad.ndim == 2:
        Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1, noise_lambda)
    else:
        channels = []
        for c in range(y_pad.shape[2]):
            channels.append(deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1, noise_lambda))
        Latent_tv = np.stack(channels, axis=2)

    Latent_tv = Latent_tv[:H, :W] if Latent_tv.ndim == 2 else Latent_tv[:H, :W, :]

    if weight_ring == 0:
        return Latent_tv

    # L0 deblurring (uses wrap_boundary_liu internally)
    if y_pad.ndim == 2:
        Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)
    else:
        Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)
    # L0Restoration already crops to original size internally

    diff_img = Latent_tv - Latent_l0

    # Bilateral filter on the difference (per-channel for multi-channel)
    if diff_img.ndim == 2:
        bf_diff = bilateral_filter(diff_img, 3, 0.1)
    else:
        channels = []
        for c in range(diff_img.shape[2]):
            channels.append(bilateral_filter(diff_img[:, :, c], 3, 0.1))
        bf_diff = np.stack(channels, axis=2)

    result = Latent_tv - weight_ring * bf_diff
    return result
