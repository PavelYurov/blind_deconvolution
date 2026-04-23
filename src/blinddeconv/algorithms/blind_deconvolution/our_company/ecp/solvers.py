"""
solvers.py

Core solver functions for ECP (Extreme Channels Prior) blind deconvolution.

Ported from MATLAB code by Yan, Ren, Guo, Wang, Cao (CVPR 2017).

Reference:
    Y. Yan, W. Ren, Y. Guo, R. Wang, X. Cao, "Image Deblurring via
    Extreme Channels Prior", CVPR 2017.

ECP augments the Dark Channel Prior (Pan et al., CVPR 2016) with a
symmetric Bright Channel term, realised via ``dark_channel(1 - S)``
inside the I-sub-problem.  The bulk of the pipeline mirrors DCP; the
ECP-specific change lives in ``L0Deblur_dark_channel_BD``.

File mapping (MATLAB → Python):
    estimate_psf.m              → estimate_psf
    L0Deblur_dark_chanelBD.m    → L0Deblur_dark_channel_BD   (ECP I-sub)
    L0Restoration.m             → L0Restoration
    blind_deconv_mainBDF.m      → blind_deconv_main_BDF      (single-scale)
    blind_deconvBDF.m           → blind_deconv               (multi-scale wrapper)
    deblurring_adm_aniso.m      → deblurring_adm_aniso
    ringing_artifacts_removal.m → ringing_artifacts_removal

MATLAB → Python notes:
    conv2(A,B,'valid')        → scipy.signal.convolve2d(A,B,'valid')
    diff(S,1,2)               → np.diff(S, n=1, axis=1)
    diff(S,1,1)               → np.diff(S, n=1, axis=0)
    S(:,1,:) - S(:,end,:)     → S[:, 0:1, :] - S[:, -1:, :]
    repmat(A,[1,1,D])         → np.tile(A, (1, 1, D))
    fft2/ifft2/conj           → np.fft.fft2 / ifft2 / conj
    bwconncomp(k,8)           → scipy.ndimage.label(k, structure=np.ones((3,3)))
    imresize(k, ret)          → scipy.ndimage.zoom (bicubic, order=3)
    interp2(I,gx,gy,'bilin')  → scipy.ndimage.map_coordinates
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

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
)


# ═════════════════════════════════════════════════════════════════════════════
# estimate_psf  (from estimate_psf.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_Ax(x, p):
    """
    Matrix–vector product for the PSF estimation CG system.
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
        'lambda': weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _compute_Ax, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf = psf / psf.sum()
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# L0Deblur_dark_channel_BD  (from L0Deblur_dark_chanelBD.m) — ECP I-sub-problem
# ═════════════════════════════════════════════════════════════════════════════

def L0Deblur_dark_channel_BD(Im, kernel, lambda_dark, wei_grad, kappa=2.0):
    """
    Image restoration with L0 dark-channel + L0 bright-channel + L0 gradient
    prior.  This is the ECP I-sub-problem.

    Equivalent to MATLAB L0Deblur_dark_chanelBD.m.

    Objective:
        min_S ||S*k - Im||^2 + λ_D |D(S)|_0 + λ_B |1 - B(S)|_0 + μ |∇S|_0

    Notes relative to the DCP version (L0Deblur_dark_chanel.m):
        • extra pixel sub-problem for the bright channel (``1 - S``),
        • the denominator of the FFT update contains ``2 * mybeta_pixel``
          (two pixel-channel auxiliary variables instead of one),
        • the dark-channel patch size is ``dark_r = 45`` (vs 35 in DCP),
        • ``maxbeta_pixel = 8`` (same numerical value as ``2**3``).

    Parameters
    ----------
    Im      : (N, M) or (N, M, D) blurred image (already boundary-wrapped)
    kernel  : (kh, kw) blur kernel
    lambda_dark : weight for the L0 dark/bright channel priors
    wei_grad    : weight for the L0 gradient prior
    kappa       : ADM update ratio (default 2.0)

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
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    dark_r = 45  # MATLAB L0Deblur_dark_chanelBD.m hard-codes this

    # MATLAB: mybeta_pixel = lambda / graythresh((S).^2)
    mybeta_pixel = lambda_dark / graythresh(S ** 2)
    maxbeta_pixel = 8  # MATLAB: 2^3

    while mybeta_pixel < maxbeta_pixel:
        # ── Dark-channel pixel sub-problem ──────────────────────────────
        J, J_idx = dark_channel(S, dark_r)
        u = J.copy()
        # MATLAB: if D==1:  t = u.^2 < lambda/mybeta_pixel
        #         else:     t = sum(u.^2, 3) < lambda/mybeta_pixel
        # J from dark_channel is always 2D (min already taken across channels),
        # so a 2D threshold is always what we want.
        t = u ** 2 < lambda_dark / mybeta_pixel
        u[t] = 0.0
        u = assign_dark_channel_to_pixel(S, u, J_idx, dark_r)

        # ── Bright-channel pixel sub-problem (operate on 1 - S) ─────────
        BS = 1.0 - S
        BJ, BJ_idx = dark_channel(BS, dark_r)
        bu = BJ.copy()
        t = bu ** 2 < lambda_dark / mybeta_pixel
        bu[t] = 0.0
        bu = assign_dark_channel_to_pixel(BS, bu, BJ_idx, dark_r)

        # ── Gradient sub-problem ────────────────────────────────────────
        beta = 2.0 * wei_grad
        while beta < betamax:
            # Two pixel-channel auxiliaries → ``2 * mybeta_pixel`` in denom.
            Denormin = Den_KER + beta * Denormin2 + 2.0 * mybeta_pixel

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

            # MATLAB: Normin2 = [h(:,end,:) - h(:,1,:), -diff(h,1,2)]
            Normin2_val = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                          -np.diff(h, n=1, axis=1)], axis=1)
            # MATLAB: Normin2 = Normin2 + [v(end,:,:) - v(1,:,:); -diff(v,1,1)]
            Normin2_val = Normin2_val + np.concatenate(
                [v[-1:, :, :] - v[0:1, :, :],
                 -np.diff(v, n=1, axis=0)], axis=0
            )

            # Broadcast 2D pixel-channel auxiliaries to (N, M, D) for fft2
            if D == 1:
                u_3d = u[:, :, np.newaxis] if u.ndim == 2 else u
                bu_3d = bu[:, :, np.newaxis] if bu.ndim == 2 else bu
            else:
                u_3d = u if u.ndim == 3 else np.tile(u[:, :, np.newaxis], (1, 1, D))
                bu_3d = bu if bu.ndim == 3 else np.tile(bu[:, :, np.newaxis], (1, 1, D))

            FS = (Normin1
                  + beta * fft2(Normin2_val, axes=(0, 1))
                  + mybeta_pixel * fft2(u_3d, axes=(0, 1))
                  + mybeta_pixel * fft2(1.0 - bu_3d, axes=(0, 1))) / Denormin
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

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """
    Image restoration with L0 gradient prior only.

    Equivalent to MATLAB L0Restoration.m.  Boundary-wraps the input to an
    FFT-friendly size, runs HQS on the gradient prior, and crops back to
    the original dimensions.
    """
    orig_ndim = Im.ndim
    H_orig, W_orig = Im.shape[0], Im.shape[1]

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
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

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

    S = S[:H_orig, :W_orig, :]
    if orig_ndim == 2:
        S = S[:, :, 0]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main_BDF  (from blind_deconv_mainBDF.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main_BDF(blur_B, k, lambda_dark, lambda_grad, threshold, opts):
    """
    Single-scale blind deconvolution — ECP variant.

    Equivalent to MATLAB blind_deconv_mainBDF.m.  Alternates between:
      • I-sub-problem ``L0Deblur_dark_channel_BD`` (dark + bright + ∇) when
        ``lambda_dark != 0``, else ``L0Restoration`` (fallback),
      • gradient thresholding ``threshold_pxpy_v1``,
      • k-sub-problem ``estimate_psf`` + connected-component pruning,
      • continuation on ``lambda_dark`` and ``lambda_grad``.
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H = blur_B.shape[0]
    W = blur_B.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(k.shape[:2]) - 1
    )
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target_size))
    blur_B_tmp = blur_B_w[:H, :W]

    # MATLAB: Bx = conv2(blur_B_tmp, dx, 'valid'); By = conv2(blur_B_tmp, dy, 'valid')
    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    xk_iter = opts.get('xk_iter', 5)

    S = None
    for _iter in range(xk_iter):
        # I-sub-problem
        if lambda_dark != 0:
            S = L0Deblur_dark_channel_BD(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        # Gradient thresholding
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        k_prev = k.copy()
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

        # MATLAB: bwconncomp(k, 8) + prune components whose sum < 0.1
        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        # Continuation
        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv  (from blind_deconvBDF.m)  +  helper sub-functions
# ═════════════════════════════════════════════════════════════════════════════

def _init_kernel(minsize):
    """
    Initialise kernel at coarsest level.
    MATLAB: k = zeros(minsize); k((minsize-1)/2, (minsize-1)/2:(minsize-1)/2+1) = 1/2;
    (1-based indexing.)
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    c = (minsize - 1) // 2     # MATLAB (minsize-1)/2 is 1-based
    r = c - 1                  # → 0-based
    k[r, r:r + 2] = 0.5
    return k


def _downSmpImC(I, ret):
    """
    Gaussian pre-filter + bilinear down-sample by factor *ret* (0 < ret ≤ 1).
    Equivalent to MATLAB downSmpImC (Levin's code).
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

    sf_row = sf.reshape(1, -1)
    sf_col = sf.reshape(-1, 1)
    if I.ndim == 3:
        channels = []
        for c in range(I.shape[2]):
            tmp = convolve2d(I[:, :, c], sf_row, mode='valid')
            tmp = convolve2d(tmp, sf_col, mode='valid')
            channels.append(tmp)
        I_filtered = np.stack(channels, axis=2)
    else:
        I_filtered = convolve2d(I, sf_row, mode='valid')
        I_filtered = convolve2d(I_filtered, sf_col, mode='valid')

    rows, cols = I_filtered.shape[0], I_filtered.shape[1]
    gx_1based = np.arange(1, cols + 1e-9, 1.0 / ret)
    gy_1based = np.arange(1, rows + 1e-9, 1.0 / ret)
    gx_grid, gy_grid = np.meshgrid(gx_1based, gy_1based)

    # 0-based for map_coordinates
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
    Adjust array *f* to target size (nk1, nk2) by trimming/padding on the
    side with smaller sum.  Equivalent to MATLAB fixsize (Levin's code).
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
    Resize kernel by factor *ret*, clamp to ≥ 0, fix to (k1, k2), normalise.
    MATLAB ``imresize`` default is bicubic → ``scipy.ndimage.zoom(order=3)``.
    """
    k = zoom(k, ret, order=3)
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    if k.max() > 0:
        k = k / k.sum()
    return k


def blind_deconv(y, lambda_dark, lambda_grad, opts):
    """
    Multi-scale ECP blind deconvolution.

    Equivalent to MATLAB blind_deconvBDF.m.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image, float64 in [0, 1]
    lambda_dark : weight for L0 dark+bright channel priors
    lambda_grad : weight for L0 gradient prior
    opts        : dict with keys
                    'kernel_size'   : int — target kernel size (square, odd)
                    'gamma_correct' : float — gamma correction exponent
                    'xk_iter'       : int — iterations per scale
                    'k_thresh'      : float — final kernel threshold
                                      (>0: zero entries < max/k_thresh)

    Returns
    -------
    kernel         : (kernel_size, kernel_size) estimated kernel
    interim_latent : intermediate latent image at the finest scale
    """
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = y ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)  # odd sizes
    k2list = k1list.copy()

    threshold = None
    ks = None
    interim_latent = None
    kernel = None

    for s in range(num_scales - 1, -1, -1):  # MATLAB: num_scales:-1:1
        if s == num_scales - 1:
            ks = _init_kernel(int(k1list[s]))
        else:
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[s]), int(k2list[s]))

        cret = retv[s]
        ys = _downSmpImC(y, cret)

        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main_BDF(
            ys, ks, lambda_dark, lambda_grad, threshold, opts
        )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        ks = ks / ks.sum()

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
    Compute spectral denominator components for the ADM / Split-Bregman
    TV-ℓ² solver.  MATLAB: computeDenominator(y, k).

    Returns
    -------
    Nomin1 : conj(F(k)) · F(y)
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
    TV-ℓ² deblurring via ADM / Split Bregman (anisotropic TV).

    Equivalent to MATLAB deblurring_adm_aniso.m.

    In the ECP pipeline this is always invoked with ``alpha = 1`` from
    ``ringing_artifacts_removal``, so only the soft-thresholding branch
    is needed.  The α ≠ 1 branch (hyper-Laplacian via lookup table) is
    not used and raises ``NotImplementedError`` on request.
    """
    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        if alpha == 1:
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                "deblurring_adm_aniso: only alpha=1 is used in the ECP pipeline"
            )

        # MATLAB: Wxx = [Wx(:,end) - Wx(:,1), -diff(Wx,1,2)]
        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        # MATLAB: Wxx = Wxx + [Wy(end,:) - Wy(1,:); -diff(Wy,1,1)]
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                    -np.diff(Wy, n=1, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, n=1, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# ringing_artifacts_removal  (from ringing_artifacts_removal.m)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    """
    Non-blind deconvolution with TV + L0 + bilateral-based ringing removal.

    Equivalent to MATLAB ringing_artifacts_removal.m.

    Pipeline:
        1. wrap_boundary_liu to an FFT-friendly size.
        2. Per-channel TV-ℓ² deblurring (``deblurring_adm_aniso``).
        3. If ``weight_ring == 0`` return the TV result.
        4. L0 deblurring (``L0Restoration``) of the original-size image.
        5. ``result = Latent_tv - weight_ring * bilateral(Latent_tv - Latent_l0)``.
    """
    H, W = y.shape[0], y.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1
    )
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    # Per-channel TV deblurring
    if y_pad.ndim == 2:
        Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    else:
        channels = []
        for c in range(y_pad.shape[2]):
            channels.append(
                deblurring_adm_aniso(y_pad[:, :, c], kernel, lambda_tv, 1)
            )
        Latent_tv = np.stack(channels, axis=2)

    if Latent_tv.ndim == 2:
        Latent_tv = Latent_tv[:H, :W]
    else:
        Latent_tv = Latent_tv[:H, :W, :]

    if weight_ring == 0:
        return Latent_tv

    # L0 deblurring; L0Restoration handles boundary wrapping + crop internally
    Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)

    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
