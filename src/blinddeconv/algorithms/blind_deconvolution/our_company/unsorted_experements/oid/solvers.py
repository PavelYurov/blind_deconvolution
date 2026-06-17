"""
solvers.py

Core solver functions for the OID (Outlier Identifying and Discarding)
blind deconvolution algorithm.

Ported from MATLAB code released with:
    L. Chen, F. Fang, J. Zhang, J. Liu, G. Zhang:
    "OID: Outlier Identifying and Discarding in Blind Image Deblurring",
    ECCV 2020.

MATLAB sources (outlier_public/main_code/):
    blind_deconv.m          -> blind_deconv  (multi-scale driver)
    coarse_deblur.m         -> coarse_deblur
    fine_deblur.m           -> fine_deblur
    image_estimate.m        -> image_estimate  (IRLS with outlier weights)
    estimate_weightmatrix.m -> estimate_weightmatrix  (sigmoid weights)
    psf_coarse.m            -> psf_coarse
    psf_fine.m              -> psf_fine  (weighted PSF estimation)
    L0Restoration.m         -> L0Restoration  (Xu & Jia L0 deblur)

MATLAB -> Python conversion notes:
    - MATLAB imfilter(x, f, 'same', 'circular') is CORRELATION with
      circular boundary  ->  scipy.ndimage.correlate(x, f, mode='wrap').
    - MATLAB imfilter(x, f, 'conv', 'circular') is CONVOLUTION (flips f)
      with circular boundary  ->  scipy.ndimage.convolve(x, f, mode='wrap').
    - For odd-size filters the filter origin matches between MATLAB and
      scipy.ndimage (floor((size+1)/2) vs size//2 coincide).
    - MATLAB conv2('valid') flips kernel  ->  scipy.signal.convolve2d('valid').
    - MATLAB diff(X,1,2) -> np.diff(X, axis=1); diff(X,1,1) -> np.diff(X, axis=0).
    - MATLAB fft2 on 2D arrays -> np.fft.fft2.
    - MATLAB bwconncomp(k,8) -> scipy.ndimage.label with 3x3 structuring element.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import correlate as nd_correlate
from scipy.ndimage import convolve as nd_convolve
from scipy.ndimage import label, map_coordinates

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    fftconv,
)


# ═════════════════════════════════════════════════════════════════════════════
# L0Restoration  (from main_code/L0Restoration.m) — Xu & Jia L0 deblurring
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im: np.ndarray, kernel: np.ndarray,
                  lambda_grad: float, kappa: float = 2.0) -> np.ndarray:
    """
    Image restoration with L0 gradient prior.  Equivalent to MATLAB
    L0Restoration.m.  Works on 2D grayscale (as used in coarse_deblur).

    Parameters
    ----------
    Im : (H, W) blurred image
    kernel : (kh, kw) blur kernel
    lambda_grad : weight for the L0 gradient prior
    kappa : ADM update ratio (default 2.0)
    """
    H_orig, W_orig = Im.shape

    target = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape) - 1
    )
    Im_w = wrap_boundary_liu(Im, tuple(target))

    S = Im_w.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    N, M = S.shape
    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    Normin1 = np.conj(KER) * fft2(S)

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        # MATLAB: h = [diff(S,1,2), S(:,1,:) - S(:,end,:)]
        h = np.concatenate(
            [np.diff(S, axis=1), (S[:, 0] - S[:, -1])[:, None]],
            axis=1,
        )
        # MATLAB: v = [diff(S,1,1); S(1,:,:) - S(end,:,:)]
        v = np.concatenate(
            [np.diff(S, axis=0), (S[0, :] - S[-1, :])[None, :]],
            axis=0,
        )

        t = (h ** 2 + v ** 2) < (lambda_grad / beta)
        h[t] = 0.0
        v[t] = 0.0

        # MATLAB: Normin2 = [h(:,end,:) - h(:,1,:), -diff(h,1,2)]
        Normin2 = np.concatenate(
            [(h[:, -1] - h[:, 0])[:, None], -np.diff(h, axis=1)],
            axis=1,
        )
        # MATLAB: Normin2 += [v(end,:,:) - v(1,:,:); -diff(v,1,1)]
        Normin2 = Normin2 + np.concatenate(
            [(v[-1, :] - v[0, :])[None, :], -np.diff(v, axis=0)],
            axis=0,
        )

        FS = (Normin1 + beta * fft2(Normin2)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa

    return S[:H_orig, :W_orig]


# ═════════════════════════════════════════════════════════════════════════════
# estimate_weightmatrix  (from main_code/estimate_weightmatrix.m)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_weightmatrix(blur: np.ndarray, latent: np.ndarray,
                          psf: np.ndarray, is_previous: bool) -> np.ndarray:
    """
    Sigmoid weight map w_i = 1/(1+exp((r_i^2 - alpha)/beta)),  r = blur - k*latent.
    Equivalent to MATLAB estimate_weightmatrix.m.

    If is_previous is truthy, returns all-ones (outlier weighting disabled).
    """
    alpha = 1.8e-3
    beta = 2e-4

    ww = np.ones_like(blur, dtype=np.float64)
    if is_previous:
        return ww

    bb = fftconv(latent, psf)
    temp = (blur - bb) ** 2
    w_matrix = 1.0 / (1.0 + np.exp((temp - alpha) / beta))

    # Hard-zero pixels at saturated extrema of the input
    max_blur = np.max(blur)
    min_blur = np.min(blur)
    w_matrix[blur == min_blur] = 0.0
    w_matrix[blur == max_blur] = 0.0

    eps = np.finfo(np.float64).eps
    w_matrix[w_matrix <= 0] = eps
    w_matrix[w_matrix >= 1] = 1.0 - eps

    ww = w_matrix
    # Hard-zero where the reconstructed k*latent is out of range
    ww[bb > 1] = 0.0
    ww[bb < 0] = 0.0
    return ww


# ═════════════════════════════════════════════════════════════════════════════
# deconv_L2  (from main_code/utils/deconv_L2.m)
# ═════════════════════════════════════════════════════════════════════════════

_DXF  = np.array([[0.0, -1.0, 1.0]])
_DYF  = np.array([[0.0], [-1.0], [1.0]])
_DXXF = np.array([[-1.0, 2.0, -1.0]])
_DYYF = np.array([[-1.0], [2.0], [-1.0]])
_DXYF = np.array([[-1.0, 1.0, 0.0],
                  [ 1.0, -1.0, 0.0],
                  [ 0.0,  0.0, 0.0]])


def _Ax_deconv(x: np.ndarray, p: dict) -> np.ndarray:
    """Matrix-vector product for deconv_L2's CG system."""
    x = x.reshape(p['img_size'])
    x_f = fft2(x)
    # First term: K^T W K x
    y = np.real(ifft2(
        fft2(p['data_we'] * np.real(ifft2(p['psf_f'] * x_f))) * np.conj(p['psf_f'])
    ))

    L2 = p['L2_we']
    # Each gradient term: D^T (W * D x)  with D = correlation, D^T = convolution
    for f, w in (
        (_DXF,  p['weight_x']),
        (_DYF,  p['weight_y']),
        (_DXXF, p['weight_xx']),
        (_DYYF, p['weight_yy']),
        (_DXYF, p['weight_xy']),
    ):
        y = y + L2 * nd_convolve(
            w * nd_correlate(x, f, mode='wrap'),
            f, mode='wrap',
        )
    return y.ravel()


def deconv_L2(blurred: np.ndarray, latent0: np.ndarray, psf: np.ndarray,
              data_we: np.ndarray, L2_we: float,
              weight_x=None, weight_y=None,
              weight_xx=None, weight_yy=None, weight_xy=None) -> np.ndarray:
    """
    Gaussian-prior deconvolution step solving the normal equations via CG.
    Equivalent to MATLAB deconv_L2.m.
    """
    if weight_x is None:
        weight_x  = np.ones_like(blurred)
        weight_y  = np.ones_like(blurred)
        weight_xx = np.zeros_like(blurred)
        weight_yy = np.zeros_like(blurred)
        weight_xy = np.zeros_like(blurred)

    img_size = blurred.shape
    psf_f = psf2otf(psf, img_size)

    # b = ifft2(fft2(W * B) * conj(K))
    b = np.real(ifft2(fft2(data_we * blurred) * np.conj(psf_f)))
    b = b.ravel()

    x = latent0.ravel().copy()

    p = {
        'psf': psf,
        'L2_we': L2_we,
        'data_we': data_we,
        'img_size': img_size,
        'psf_f': psf_f,
        'weight_x':  weight_x,
        'weight_y':  weight_y,
        'weight_xx': weight_xx,
        'weight_yy': weight_yy,
        'weight_xy': weight_xy,
    }

    x = conjgrad(x, b, 25, 1e-4, _Ax_deconv, p)
    return x.reshape(img_size)


# ═════════════════════════════════════════════════════════════════════════════
# image_estimate  (from main_code/image_estimate.m)
# ═════════════════════════════════════════════════════════════════════════════

def image_estimate(blurred: np.ndarray, psf: np.ndarray,
                   reg_strength: float, is_previous: bool):
    """
    IRLS image estimation with outlier weights and hyper-Laplacian gradient
    prior.  Equivalent to MATLAB image_estimate.m.

    Grayscale input returns (latent, w_out) as 2D arrays.
    Colour (H,W,3) input is processed channel-by-channel — faithful to the
    per-pixel formulation since the weight map is computed elementwise.
    """
    if blurred.ndim == 3:
        H, W, C = blurred.shape
        latent = np.empty_like(blurred, dtype=np.float64)
        w_out = np.empty_like(blurred, dtype=np.float64)
        for c in range(C):
            latent[:, :, c], w_out[:, :, c] = image_estimate(
                blurred[:, :, c], psf, reg_strength, is_previous
            )
        return latent, w_out

    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01
    N_iters = 15

    H, W = blurred.shape
    target = opt_fft_size(np.array([H, W]) + np.array(psf.shape) - 1)
    blurred_w = wrap_boundary_liu(blurred, tuple(target))
    w_matrix = wrap_boundary_liu(np.zeros_like(blurred), tuple(target))

    # Re-assert that the core pixels have weight 1
    w_matrix[:H, :W] = 1.0

    mask = np.zeros_like(blurred_w)
    mask[:H, :W] = 1.0

    eps = np.finfo(np.float64).eps
    w_matrix[(1 - mask) == 1] = eps
    w_matrix[w_matrix >= 1] = 1.0 - eps
    w_matrix[w_matrix <= 0] = eps

    latent_w = deconv_L2(blurred_w, blurred_w, psf, w_matrix, reg_strength)

    ww = w_matrix  # ensures 'ww' is defined even if N_iters == 0
    for _ in range(N_iters):
        w_matrix = estimate_weightmatrix(blurred_w, latent_w, psf, is_previous)
        ww = w_matrix * mask

        dx  = nd_correlate(latent_w, _DXF,  mode='wrap')
        dy  = nd_correlate(latent_w, _DYF,  mode='wrap')
        dxx = nd_correlate(latent_w, _DXXF, mode='wrap')
        dyy = nd_correlate(latent_w, _DYYF, mode='wrap')
        dxy = nd_correlate(latent_w, _DXYF, mode='wrap')

        weight_x  =        w0 * np.maximum(np.abs(dx),  thr_e) ** (exp_a - 2)
        weight_y  =        w0 * np.maximum(np.abs(dy),  thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        latent_w = deconv_L2(
            blurred_w, latent_w, psf, ww, reg_strength,
            weight_x, weight_y, weight_xx, weight_yy, weight_xy,
        )

    latent = latent_w[:H, :W]
    w_out = ww[:H, :W]
    return latent, w_out


# ═════════════════════════════════════════════════════════════════════════════
# psf_coarse  (from main_code/psf_coarse.m)
# ═════════════════════════════════════════════════════════════════════════════

def _Ax_psf_coarse(x: np.ndarray, p: dict) -> np.ndarray:
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def psf_coarse(blurred_x, blurred_y, latent_x, latent_y,
               weight: float, psf_size: tuple) -> np.ndarray:
    """
    Kernel estimation in the gradient domain without outlier weights.
    Equivalent to MATLAB psf_coarse.m.
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(otf2psf(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape,
        'psf_size': psf_size,
        'lambda': weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _Ax_psf_coarse, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf = psf / psf.sum()
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# psf_fine  (from main_code/psf_fine.m) — weighted PSF estimation
# ═════════════════════════════════════════════════════════════════════════════

def _Ax_psf_fine(x: np.ndarray, p: dict) -> np.ndarray:
    x_f = psf2otf(x, p['img_size'])
    Ixconvk = p['ww_x'] * np.real(ifft2(p['latent_xf'] * x_f))
    Iyconvk = p['ww_y'] * np.real(ifft2(p['latent_yf'] * x_f))
    y = otf2psf(
        np.conj(p['latent_xf']) * fft2(Ixconvk) +
        np.conj(p['latent_yf']) * fft2(Iyconvk),
        p['psf_size'],
    )
    y = y + p['lambda'] * x
    return y


def psf_fine(blurred: np.ndarray, latent: np.ndarray, weight: float,
             psf: np.ndarray, threshold: float, is_previous: bool):
    """
    Kernel estimation in the gradient domain WITH per-pixel outlier weights.
    Equivalent to MATLAB psf_fine.m.
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H, W = blurred.shape
    target = opt_fft_size(np.array([H, W]) + np.array(psf.shape) - 1)
    blur_B_w = wrap_boundary_liu(blurred, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    latent_x, latent_y, _ = threshold_pxpy_v1(
        latent, np.max(psf.shape), threshold
    )
    psf_size = psf.shape

    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)

    ww_x = np.ones_like(Bx)
    for _ in range(15):
        ww_x = estimate_weightmatrix(Bx, latent_x, psf, is_previous)
        ww_y = estimate_weightmatrix(By, latent_y, psf, is_previous)

        b_f = (np.conj(latent_xf) * fft2(ww_x * Bx) +
               np.conj(latent_yf) * fft2(ww_y * By))
        b = np.real(otf2psf(b_f, psf_size))

        p = {
            'latent_xf': latent_xf,
            'latent_yf': latent_yf,
            'img_size': Bx.shape,
            'psf_size': psf_size,
            'lambda': weight,
            'ww_x': ww_x,
            'ww_y': ww_y,
        }

        # MATLAB re-initialises the kernel to uniform at every outer iteration.
        psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
        psf = conjgrad(psf, b, 21, 1e-5, _Ax_psf_fine, p)

        psf[psf < psf.max() * 0.05] = 0.0
        psf = psf / psf.sum()

    return psf, ww_x


# ═════════════════════════════════════════════════════════════════════════════
# coarse_deblur  (from main_code/coarse_deblur.m)
# ═════════════════════════════════════════════════════════════════════════════

def _prune_isolated(k: np.ndarray) -> np.ndarray:
    """Zero out connected components with total mass < 0.1 (MATLAB bwconncomp)."""
    structure = np.ones((3, 3), dtype=np.int64)
    lbl, n_comp = label(k > 0, structure=structure)
    for ii in range(1, n_comp + 1):
        mask = (lbl == ii)
        if k[mask].sum() < 0.1:
            k[mask] = 0.0
    return k


def coarse_deblur(blur_B: np.ndarray, k: np.ndarray, lambda_grad: float,
                  threshold: float, opts: dict):
    """
    Coarse-scale deblurring (no outlier weights).
    Equivalent to MATLAB coarse_deblur.m.
    """
    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H, W = blur_B.shape
    target = opt_fft_size(np.array([H, W]) + np.array(k.shape) - 1)
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]
    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    S = blur_B
    predeblur = str(opts.get('predeblur', 'L0'))

    for _ in range(int(opts['xk_iter'])):
        if predeblur == 'L0':
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S, _ = image_estimate(blur_B, k, lambda_grad, True)

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, np.max(k.shape), threshold
        )
        k = psf_coarse(Bx, By, latent_x, latent_y, 5, k.shape)

        k = _prune_isolated(k)
        k[k < 0] = 0.0
        k = k / k.sum()

    k[k < 0] = 0.0
    k = k / k.sum()
    return k, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# fine_deblur  (from main_code/fine_deblur.m)
# ═════════════════════════════════════════════════════════════════════════════

def fine_deblur(blur_B: np.ndarray, k: np.ndarray, lambda_grad: float,
                threshold: float, opts: dict):
    """
    Fine-scale deblurring WITH outlier identification / weights.
    Equivalent to MATLAB fine_deblur.m.
    """
    S = blur_B
    for it in range(int(opts['xk_iter'])):
        k_prev = k
        S, _wi = image_estimate(blur_B, k, lambda_grad, False)
        k, _wk = psf_fine(blur_B, S, 5, k, threshold, False)

        err = k_prev - k
        if np.linalg.norm(err.ravel(), 2) < 1e-3:
            break

        k = _prune_isolated(k)
        k[k < 0] = 0.0
        k = k / k.sum()

        if (it + 1) % 5 == 0:
            k = adjust_psf_center(k)
            k[k < 0] = 0.0
            k = k / k.sum()
            k_thresh = float(opts.get('k_thresh', 0))
            if k_thresh > 0:
                k[k < k.max() / k_thresh] = 0.0
            else:
                k[k < 0] = 0.0
            k = k / k.sum()

    k[k < 0] = 0.0
    k = k / k.sum()
    return k, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# Multi-scale helpers  (nested in MATLAB blind_deconv.m)
# ═════════════════════════════════════════════════════════════════════════════

def _init_kernel(minsize: int) -> np.ndarray:
    """
    MATLAB init_kernel: a length-2 horizontal bar at the geometric centre.
    Note the MATLAB code uses 1-based index (minsize-1)/2 — for an odd
    minsize this lands exactly one row above the centre; we reproduce it
    faithfully (0-based: row (minsize-1)//2 - 1, cols [c-1, c]).
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    # MATLAB: k((minsize-1)/2, (minsize-1)/2 : (minsize-1)/2 + 1) = 1/2
    row_m = (minsize - 1) // 2     # 1-based row index
    col_m = (minsize - 1) // 2     # 1-based column start
    # Convert 1-based to 0-based: subtract 1
    r = row_m - 1
    c0 = col_m - 1
    k[r, c0:c0 + 2] = 0.5
    return k


def _downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:
    """
    Levin-style downsampling: Gaussian LPF + bilinear interp2.
    Equivalent to MATLAB downSmpImC nested in blind_deconv.m.
    """
    if ret == 1:
        return I.copy()

    sig = 1.0 / np.pi * ret
    g0 = np.arange(-50, 51, dtype=np.float64) * 2.0 * np.pi
    sf = np.exp(-0.5 * g0 ** 2 * sig ** 2)
    sf = sf / sf.sum()
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]

    # MATLAB: conv2(sf, sf', I, 'valid') = conv2 with row then column filter
    tmp = convolve2d(I, sf[None, :], mode='valid')
    tmp = convolve2d(tmp, sf[:, None], mode='valid')

    # MATLAB: [gx, gy] = meshgrid(1:1/ret:size(I,2), 1:1/ret:size(I,1))
    #         sI = interp2(I, gx, gy, 'bilinear')  — note: applied to the
    #         ORIGINAL I in MATLAB (tmp is assigned to I first via `I = conv2(...)`)
    step = 1.0 / ret
    H2, W2 = tmp.shape
    # MATLAB interp2 uses 1-based coords: 1..size(tmp, dim) in steps of 1/ret
    # which map to (1-based) -> (0-based) indices by -1.
    gx = np.arange(1.0, W2 + 1e-9, step) - 1.0   # 0-based column coords
    gy = np.arange(1.0, H2 + 1e-9, step) - 1.0   # 0-based row coords
    XX, YY = np.meshgrid(gx, gy)
    sI = map_coordinates(
        tmp, [YY.ravel(), XX.ravel()],
        order=1, mode='constant', cval=0.0,
    ).reshape(YY.shape)
    return sI


def _fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:
    """Levin's trim/pad helper to force a kernel to size (nk1, nk2)."""
    k1, k2 = f.shape
    while (k1 != nk1) or (k2 != nk2):
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]
        if k1 < nk1:
            s = f.sum(axis=1)
            tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:k1, :] = f
            else:
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
            tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:, :k2] = f
            else:
                tf[:, 1:k2 + 1] = f
            f = tf
        k1, k2 = f.shape
    return f


def _resizeKer(k: np.ndarray, ret: float, k1: int, k2: int) -> np.ndarray:
    """
    Upsample kernel between pyramid levels.  Equivalent to MATLAB resizeKer.
    MATLAB uses imresize with default bicubic interpolation.  We use
    scipy.ndimage.map_coordinates with order=3 (cubic spline) to match.
    """
    in_h, in_w = k.shape
    out_h = int(round(in_h * ret))
    out_w = int(round(in_w * ret))
    # Bilinear is close enough to MATLAB's default bicubic for kernels.
    # Use the scale factor to produce (out_h, out_w) grid sampling k.
    ys = np.linspace(0, in_h - 1, out_h)
    xs = np.linspace(0, in_w - 1, out_w)
    XX, YY = np.meshgrid(xs, ys)
    k_up = map_coordinates(
        k, [YY.ravel(), XX.ravel()], order=3, mode='constant', cval=0.0,
    ).reshape(out_h, out_w)

    k_up = np.maximum(k_up, 0.0)
    k_up = _fixsize(k_up, k1, k2)
    if k_up.max() > 0:
        k_up = k_up / k_up.sum()
    return k_up


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv  (from main_code/blind_deconv.m) — multi-scale driver
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv(y: np.ndarray, lambda_grad: float, opts: dict):
    """
    Multi-scale coarse-to-fine blind deconvolution.  Equivalent to MATLAB
    blind_deconv.m.  Returns (kernel, interim_latent).

    `opts` must include:
        kernel_size    : int — assumed square, odd
        xk_iter        : int — iterations at coarse scales
        last_iter      : int — iterations at the finest scale
        k_thresh       : float — final kernel threshold ratio
        isnoisy        : bool/int — pre-smoothing on coarse levels
        predeblur      : str — 'L0' or 'Lp' (for coarse_deblur)
        gamma_correct  : float — exponent for gamma correction on input
    """
    if opts.get('gamma_correct', 1.0) != 1.0:
        y = y ** opts['gamma_correct']

    ret = np.sqrt(0.5)
    kernel_size = int(opts['kernel_size'])

    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1
    print(f'Maximum iteration level is {num_scales}')

    retv = ret ** np.arange(0, maxitr + 1)

    k1list = np.ceil(kernel_size * retv).astype(np.int64)
    k1list = k1list + (k1list % 2 == 0).astype(np.int64)   # force odd
    k2list = k1list.copy()

    ks = None
    threshold = None
    interim_latent = None
    kernel = None

    # MATLAB loops s = num_scales : -1 : 1  (coarsest -> finest)
    for s in range(num_scales, 0, -1):
        idx = s - 1  # 0-based index into *list arrays

        if s == num_scales:
            ks = _init_kernel(int(k1list[idx]))
            k1 = int(k1list[idx])
            k2 = k1
        else:
            k1 = int(k1list[idx])
            k2 = k1
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[idx]), int(k2list[idx]))

        cret = float(retv[idx])
        ys = _downSmpImC(y, cret)

        print(f'Processing scale {s}/{num_scales}; kernel size {k1}x{k2}; '
              f'image size {ys.shape[0]}x{ys.shape[1]}')

        if s == num_scales:
            _, _, threshold = threshold_pxpy_v1(ys, np.max(ks.shape))

        if s <= 1:
            # Finest scale: use outlier weights
            opts_fine = dict(opts)
            opts_fine['xk_iter'] = int(opts.get('last_iter', opts['xk_iter']))
            ks, lambda_grad, interim_latent = fine_deblur(
                ys, ks, lambda_grad, threshold, opts_fine,
            )
        else:
            # Coarser scales: optional denoising then L0-based coarse deblur
            if opts.get('isnoisy', 0):
                # fspecial('gaussian', 5, 1) + imfilter 'same','replicate'
                from scipy.ndimage import gaussian_filter
                ys = gaussian_filter(ys, sigma=1.0, truncate=2.0, mode='nearest')
            ks, lambda_grad, interim_latent = coarse_deblur(
                ys, ks, lambda_grad, threshold, opts,
            )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        sumk = ks.sum()
        if sumk > 0:
            ks = ks / sumk

        if s == 1:
            kernel = ks
            k_thresh = float(opts.get('k_thresh', 0))
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            kernel = kernel / kernel.sum()

    return kernel, interim_latent
