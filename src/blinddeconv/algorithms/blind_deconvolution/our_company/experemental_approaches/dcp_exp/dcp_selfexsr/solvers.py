import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

from ..utils import (
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
    wiener_filter,
)

from ..solvers import (
    estimate_psf,
    L0Deblur_dark_channel,
    L0Restoration,
    blind_deconv_main,
    _init_kernel,
    _downSmpImC,
    _fixsize,
    _resizeKer,
    ringing_artifacts_removal,
)

from blinddeconv.algorithms.super_resolution.our_company.selfexsr.solvers import (
    sr_demo,
    sr_init_opt,
)

from .utils import blend_images, resize_to_match, compute_sr_blend_alpha

def blind_deconv_main_sr_enhanced(blur_B, k, lambda_dark, lambda_grad,
                                  threshold, opts, sr_ref, alpha):

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H = blur_B.shape[0]
    W = blur_B.shape[1]

    target_size = opt_fft_size(
        np.array([H, W]) + np.array(k.shape[:2]) - 1
    )
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target_size))
    blur_B_tmp = blur_B_w[:H, :W]

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    xk_iter = opts.get('xk_iter', 5)

    if sr_ref.shape[0] != H or sr_ref.shape[1] != W:
        sr_ref = resize_to_match(sr_ref, (H, W))

    sr_x = convolve2d(sr_ref, dx, mode='valid')
    sr_y = convolve2d(sr_ref, dy, mode='valid')
    sr_mag = sr_x ** 2 + sr_y ** 2

    injection_weight = 1.0 - alpha

    for _iter in range(xk_iter):

        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        dcp_zero = (latent_x == 0) & (latent_y == 0)
        sr_above_thresh = sr_mag >= threshold
        fill_mask = dcp_zero & sr_above_thresh

        latent_x[fill_mask] = injection_weight * sr_x[fill_mask]
        latent_y[fill_mask] = injection_weight * sr_y[fill_mask]

        k_prev = k.copy()

        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        S = np.clip(S, 0.0, 1.0)

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S

def blind_deconv_sr(y, lambda_dark, lambda_grad, opts,
                    sr_opts=None):

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
    k1list = k1list + (k1list % 2 == 0)
    k2list = k1list.copy()

    sr_alpha_min = opts.get('sr_alpha_min', 0.5)
    sr_alpha_max = opts.get('sr_alpha_max', 0.85)
    wiener_snr = opts.get('wiener_snr', 0.01)
    sr_downscale = opts.get('sr_downscale', 0.5)

    if sr_opts is None:
        sr_opts = {}
    SRF = sr_opts.get('SRF', 2)
    sr_num_iter = sr_opts.get('numIter', 3)
    sr_n_iter_bp = sr_opts.get('nIterBP', 5)

    if num_scales <= 2:
        n_warmup = num_scales
        n_sr_levels = 0
    else:
        n_warmup = min(opts.get('n_warmup_levels', 2), num_scales - 2)
        remaining = num_scales - n_warmup

        n_sr_levels = min(opts.get('n_sr_levels', 2), remaining - 1)

    warmup_boundary = num_scales - n_warmup

    sr_boundary = warmup_boundary - n_sr_levels

    threshold = None
    ks = None
    interim_latent = None
    y_sr_ref = None

    for s_idx in range(num_scales - 1, -1, -1):
        s = s_idx

        if s == num_scales - 1:
            ks = _init_kernel(int(k1list[s]))
        else:
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[s]), int(k2list[s]))

        cret = retv[s]
        ys = _downSmpImC(y, cret)

        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        if s >= warmup_boundary:
            ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
                ys, ks, lambda_dark, lambda_grad, threshold, opts
            )

            if s == warmup_boundary and n_sr_levels > 0:
                y_sr_ref = _build_sr_reference(
                    y, ks, k1list, k2list, ret, retv, s,
                    SRF, sr_num_iter, sr_n_iter_bp, wiener_snr,
                    sr_downscale=sr_downscale,
                )

        elif s >= sr_boundary and y_sr_ref is not None:
            sr_ref_s = _downSmpImC(y_sr_ref, cret)

            alpha = compute_sr_blend_alpha(
                scale_idx=s,
                warmup_boundary=warmup_boundary,
                n_sr_levels=n_sr_levels,
                alpha_min=sr_alpha_min,
                alpha_max=sr_alpha_max,
            )

            ks, lambda_dark, lambda_grad, interim_latent =\
                blind_deconv_main_sr_enhanced(
                    ys, ks, lambda_dark, lambda_grad,
                    threshold, opts, sr_ref_s, alpha
                )

        else:
            ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
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

def _build_sr_reference(y, ks_warmup, k1list, k2list, ret, retv, warmup_level,
                        SRF, sr_num_iter, sr_n_iter_bp, wiener_snr,
                        sr_downscale=0.75):

    target_k1 = int(k1list[0])
    target_k2 = int(k2list[0])
    current_k_size = ks_warmup.shape[0]

    if current_k_size < target_k1:
        zoom_factor = target_k1 / current_k_size
        k_full = _resizeKer(ks_warmup, zoom_factor, target_k1, target_k2)
    else:
        k_full = ks_warmup.copy()

    S_deconv = wiener_filter(y, k_full, noise_snr=wiener_snr)
    S_deconv = np.clip(S_deconv, 0.0, 1.0)

    if sr_downscale < 1.0:
        H_ds = max(int(y.shape[0] * sr_downscale), 32)
        W_ds = max(int(y.shape[1] * sr_downscale), 32)
        S_for_sr = resize_to_match(S_deconv, (H_ds, W_ds))
    else:
        S_for_sr = S_deconv

    sr_opt = sr_init_opt(SRF)
    sr_opt['numIter'] = sr_num_iter
    sr_opt['nIterBP'] = sr_n_iter_bp

    y_sr_hr = sr_demo(S_for_sr, SRF, opt=sr_opt)

    if y_sr_hr.ndim == 3:
        y_sr_gray = (0.2989 * y_sr_hr[:, :, 0]
                     + 0.5870 * y_sr_hr[:, :, 1]
                     + 0.1140 * y_sr_hr[:, :, 2])
    else:
        y_sr_gray = y_sr_hr

    y_sr_ref = resize_to_match(y_sr_gray.astype(np.float64), y.shape[:2])
    y_sr_ref = np.clip(y_sr_ref, 0.0, 1.0)

    return y_sr_ref
