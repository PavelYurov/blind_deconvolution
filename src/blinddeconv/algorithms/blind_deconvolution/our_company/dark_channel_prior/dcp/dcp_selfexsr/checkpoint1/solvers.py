"""
solvers.py

SR-Enhanced DCP blind deconvolution solvers.

This module extends the original DCP blind deconvolution pipeline
(Pan et al., CVPR 2016) by integrating SelfExSR (Huang et al.,
CVPR 2015) to improve latent image estimation on coarse pyramid
levels — where the standard DCP struggles due to loss of high-
frequency detail.

Integration strategy:
    ─────────────────────────────────────────────────────────────
    1. Before the multi-scale loop, run SelfExSR ONCE on the
       blurred input (SRF=2) to obtain an "SR reference" — an
       image with restored high-frequency self-similar details.
    2. On the N coarsest pyramid levels, after L0Deblur_dark_channel
       produces the latent estimate S, we blend S with a
       downsampled version of the SR reference:
           S_enhanced = α·S + (1-α)·S_sr
       where α varies from ~0.3 (coarsest, most SR trust) to
       ~0.7 (less SR trust) as we ascend the pyramid.
    3. The enhanced S_enhanced is then passed to threshold_pxpy_v1
       and estimate_psf, yielding sharper gradients and therefore
       a more accurate kernel estimate.
    4. Finer pyramid levels run the standard DCP pipeline (α=1)
       because by then the kernel estimate is already good.

Why this works:
    SelfExSR exploits internal patch recurrence across scales —
    exactly the high-frequency content that vanishes when DCP
    downsamples the image for the coarse pyramid. By injecting
    this information early, we break the error cascade that
    otherwise propagates from coarse to fine levels.

All unmodified functions are imported from the original DCP solvers.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

# ── Import EVERYTHING from original DCP solvers & utils ──────────────────
# These are used as-is without modification.
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

# ── Import SelfExSR pipeline ─────────────────────────────────────────────
from blinddeconv.algorithms.super_resolution.our_company.selfexsr.solvers import (
    sr_demo,
    sr_init_opt,
)

# ── Local blending utilities ─────────────────────────────────────────────
from .utils import blend_images, resize_to_match, compute_sr_blend_alpha


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main_sr_enhanced
#   Modified version of blind_deconv_main that blends the latent estimate
#   with a downsampled SR reference before gradient thresholding & kernel
#   estimation.
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main_sr_enhanced(blur_B, k, lambda_dark, lambda_grad,
                                  threshold, opts, sr_ref, alpha):
    """
    Single-scale blind deconvolution with SR-enhanced latent estimate.

    Identical to the original blind_deconv_main EXCEPT:
    after computing S via L0Deblur_dark_channel (or L0Restoration),
    we blend S with *sr_ref* using weight *alpha* before extracting
    gradients. This gives sharper edges on coarse levels → better
    kernel estimation.

    Additional Parameters
    ---------------------
    sr_ref : (H, W) — SR reference image downsampled to match blur_B
             spatial size.  Already in [0, 1] float64.
    alpha  : float in [0, 1] — DCP weight.  0 = pure SR, 1 = pure DCP.

    Returns
    -------
    k, lambda_dark, lambda_grad, S  — same as blind_deconv_main
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

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    xk_iter = opts.get('xk_iter', 5)

    # Ensure sr_ref matches (H, W)
    if sr_ref.shape[0] != H or sr_ref.shape[1] != W:
        sr_ref = resize_to_match(sr_ref, (H, W))

    for _iter in range(xk_iter):
        # ── Step 1: Latent image estimation (standard DCP) ───────────
        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        # ── Step 2 (NEW): Blend with SR reference ────────────────────
        # The SR reference contains restored high-frequency detail from
        # self-similar patches.  Blending injects this information into
        # the latent estimate before gradient extraction, giving the
        # kernel estimator sharper, more reliable edges to work with.
        S_for_grad = blend_images(S, sr_ref, alpha)

        # ── Step 3: Gradient thresholding (uses enhanced S) ──────────
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S_for_grad, max(k.shape), threshold
        )

        k_prev = k.copy()

        # ── Step 4: Kernel estimation (unchanged) ────────────────────
        k = estimate_psf(Bx, By, latent_x, latent_y, 2, k_prev.shape)

        # Prune isolated noise in kernel
        labeled, num_features = label(k, structure=np.ones((3, 3)))
        for ii in range(1, num_features + 1):
            mask = labeled == ii
            if k[mask].sum() < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        k = k / k.sum()

        # Parameter updating (same schedule as original)
        if lambda_dark != 0:
            lambda_dark = max(lambda_dark / 1.1, 1e-4)
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)

        S = np.clip(S, 0.0, 1.0)

    k[k < 0] = 0.0
    k = k / k.sum()

    return k, lambda_dark, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_sr
#   Modified multi-scale blind deconvolution that:
#   (a) runs SelfExSR once to build an SR reference,
#   (b) uses blind_deconv_main_sr_enhanced on coarse levels,
#   (c) falls back to standard blind_deconv_main on fine levels.
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_sr(y, lambda_dark, lambda_grad, opts,
                    sr_opts=None):
    """
    Multi-scale blind deconvolution with SelfExSR-enhanced coarse levels.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image, float64 [0,1]
    lambda_dark : float — L0 intensity prior weight
    lambda_grad : float — L0 gradient prior weight
    opts        : dict — standard DCP options:
                    'kernel_size', 'gamma_correct', 'xk_iter', 'k_thresh'
                  PLUS new SR-specific keys:
                    'n_sr_levels' : int — how many coarsest levels use SR
                                    enhancement. Default 3.
                    'sr_alpha_min': float — blend alpha on coarsest level.
                                    Default 0.3.
                    'sr_alpha_max': float — blend alpha on finest SR level.
                                    Default 0.7.
    sr_opts     : dict or None — SelfExSR parameter overrides:
                    'SRF'       : int   — SR factor (default 2)
                    'numIter'   : int   — PatchMatch iterations (default 5)
                    'nIterBP'   : int   — back-projection iterations (default 10)

    Returns
    -------
    kernel         : (kernel_size, kernel_size) estimated kernel
    interim_latent : intermediate latent image from finest scale
    """
    # ── Gamma correction ─────────────────────────────────────────────
    gamma_correct = opts.get('gamma_correct', 1.0)
    if gamma_correct != 1:
        y = y ** gamma_correct

    kernel_size = opts['kernel_size']
    if isinstance(kernel_size, (list, tuple, np.ndarray)):
        kernel_size = int(kernel_size[0])

    # ── SR parameters ────────────────────────────────────────────────
    n_sr_levels = opts.get('n_sr_levels', 3)
    sr_alpha_min = opts.get('sr_alpha_min', 0.3)
    sr_alpha_max = opts.get('sr_alpha_max', 0.7)

    if sr_opts is None:
        sr_opts = {}
    SRF = sr_opts.get('SRF', 2)
    sr_num_iter = sr_opts.get('numIter', 5)
    sr_n_iter_bp = sr_opts.get('nIterBP', 10)

    # ── Phase 0: Build SR reference (one-time cost) ──────────────────
    # SelfExSR restores high-frequency self-similar detail from the
    # blurred input.  We run it at a modest SRF=2 then downsample back
    # to the original resolution — the goal is NOT super-resolution but
    # to have a sharper version with restored internal textures.
    sr_opt = sr_init_opt(SRF)
    sr_opt['numIter'] = sr_num_iter
    sr_opt['nIterBP'] = sr_n_iter_bp

    # sr_demo expects [0,1] float, returns (H*SRF, W*SRF, 3) float32
    y_sr_hr = sr_demo(y, SRF, opt=sr_opt)

    # Convert to grayscale and downsample back to original resolution
    if y_sr_hr.ndim == 3:
        y_sr_gray = (0.2989 * y_sr_hr[:, :, 0]
                     + 0.5870 * y_sr_hr[:, :, 1]
                     + 0.1140 * y_sr_hr[:, :, 2])
    else:
        y_sr_gray = y_sr_hr

    # Downscale SR output back to the original image size.
    # The SR reference should be at the same resolution as y.
    y_sr_ref = resize_to_match(y_sr_gray.astype(np.float64), y.shape[:2])
    y_sr_ref = np.clip(y_sr_ref, 0.0, 1.0)

    # ── Multi-scale pyramid (same as original blind_deconv) ──────────
    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)  # ensure odd
    k2list = k1list.copy()

    threshold = None
    ks = None
    interim_latent = None

    # ── Coarse-to-fine loop ──────────────────────────────────────────
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

        # ── Decide: SR-enhanced or standard DCP ──────────────────────
        # SR enhancement is applied on the n_sr_levels coarsest levels.
        # The coarsest level has index num_scales-1, decreasing toward 0.
        use_sr = (s >= num_scales - n_sr_levels)

        if use_sr:
            # Downsample SR reference to match this pyramid level
            sr_ref_s = _downSmpImC(y_sr_ref, cret)

            alpha = compute_sr_blend_alpha(
                scale_idx=s,
                num_scales=num_scales,
                n_sr_levels=n_sr_levels,
                alpha_min=sr_alpha_min,
                alpha_max=sr_alpha_max,
            )

            ks, lambda_dark, lambda_grad, interim_latent = \
                blind_deconv_main_sr_enhanced(
                    ys, ks, lambda_dark, lambda_grad,
                    threshold, opts, sr_ref_s, alpha
                )
        else:
            # Standard DCP — no SR blending
            ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
                ys, ks, lambda_dark, lambda_grad, threshold, opts
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
