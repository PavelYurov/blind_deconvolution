"""
solvers.py

SR-Enhanced DCP blind deconvolution solvers.

This module extends the original DCP blind deconvolution pipeline
(Pan et al., CVPR 2016) by integrating SelfExSR (Huang et al.,
CVPR 2015) to improve latent image estimation on intermediate
pyramid levels.

Integration strategy (REVISED — "mid-loop SR injection"):
    ─────────────────────────────────────────────────────────────
    The original (v1) approach ran SelfExSR on the RAW BLURRED
    input.  This is fundamentally wrong: SelfExSR assumes a clean
    low-resolution input and recovers detail from self-similar
    patches.  On a blurred image the patches are blurred too, so
    SR hallucinates FALSE high-frequency content.  Blending this
    with DCP's latent estimate adds fake edges that mislead the
    kernel estimator — especially on complex kernels.

    CORRECTED approach — three-phase pipeline:

    Phase 1 — WARMUP (n_warmup coarsest levels):
        Run standard DCP to obtain a rough but reasonable kernel.
        At coarse levels the kernel is small (5–9 px) and DCP
        handles this well without help.

    Phase 1→2 bridge — SR REFERENCE CONSTRUCTION (once):
        After warmup, upsample the rough kernel to full resolution
        and Wiener-deconvolve the original blurred image.  The
        result is a partially-deblurred image with real scene
        structure (not blur artefacts).  Run SelfExSR on THIS
        image to get an SR reference with genuinely enhanced
        self-similar textures.

    Phase 2 — SR-ENHANCED (n_sr_levels intermediate levels):
        After L0Deblur_dark_channel produces latent S, blend
        S with the downsampled SR reference before gradient
        extraction:   S_enh = α·S + (1-α)·S_sr
        α is conservative (0.5–0.85) and increases from
        coarser to finer SR levels.

    Phase 3 — FINE (remaining levels, including finest):
        Standard DCP, no SR blending.  The kernel estimate is
        already good from the enhanced intermediate levels.

Why this works:
    - SR input is partially deblurred → self-similar patches
      reflect real scene structure, not blur artefacts.
    - SR runs only ONCE and on a reasonably clean image.
    - Coarsest levels (small kernels) don't need SR.
    - Finest level uses pure DCP → no SR contamination in
      the final kernel.
    - Conservative blending prevents SR from dominating.

All unmodified functions are imported from the original DCP solvers.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

# ── Import from original DCP solvers & utils (used as-is) ───────────────
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

# ── Import SelfExSR pipeline ─────────────────────────────────────────────
from blinddeconv.algorithms.super_resolution.our_company.selfexsr.solvers import (
    sr_demo,
    sr_init_opt,
)

# ── Local blending utilities ─────────────────────────────────────────────
from .utils import blend_images, resize_to_match, compute_sr_blend_alpha


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main_sr_enhanced
#   Modified version of blind_deconv_main that fills gradient gaps from an
#   SR reference AFTER thresholding: where DCP gradients were zeroed out
#   (below threshold) but SR gradients are above threshold, inject SR
#   gradients.  DCP's existing gradient evidence is NEVER overwritten.
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main_sr_enhanced(blur_B, k, lambda_dark, lambda_grad,
                                  threshold, opts, sr_ref, alpha):
    """
    Single-scale blind deconvolution with SR gradient gap-filling.

    Identical to the original blind_deconv_main EXCEPT:
    after threshold_pxpy_v1 zeroes weak DCP gradients, we check
    the SR reference's gradients at those zeroed positions.  If an
    SR gradient is above the current threshold, it is injected
    (scaled by ``1 – alpha``) into the gradient map.

    This is safer than pixel-domain blending because:
    - DCP's existing strong edges are NEVER modified.
    - SR can only ADD gradient evidence where DCP has none.
    - The threshold still applies: weak SR gradients are ignored.

    Additional Parameters
    ---------------------
    sr_ref : (H, W) — SR reference image downsampled to match blur_B
             spatial size.  Already in [0, 1] float64.
    alpha  : float in [0, 1] — DCP weight.
             Controls SR injection strength: injected_grad *= (1 - alpha).
             0 → full SR injection, 1 → no SR injection.

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

    # Pre-compute SR gradients (constant across iterations)
    sr_x = convolve2d(sr_ref, dx, mode='valid')
    sr_y = convolve2d(sr_ref, dy, mode='valid')
    sr_mag = sr_x ** 2 + sr_y ** 2

    injection_weight = 1.0 - alpha  # how strongly to inject SR gradients

    for _iter in range(xk_iter):
        # ── Step 1: Latent image estimation (standard DCP) ───────────
        if lambda_dark != 0:
            S = L0Deblur_dark_channel(blur_B_w, k, lambda_dark, lambda_grad, 2.0)
            S = S[:H, :W]
        else:
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)

        # ── Step 2: DCP gradient thresholding ────────────────────────
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        # ── Step 3: Gradient gap-filling from SR ─────────────────────
        # Where DCP zeroed gradients (below threshold) but SR has
        # meaningful gradients (above threshold), inject SR evidence.
        # DCP's existing gradients are NEVER touched.
        dcp_zero = (latent_x == 0) & (latent_y == 0)
        sr_above_thresh = sr_mag >= threshold
        fill_mask = dcp_zero & sr_above_thresh

        latent_x[fill_mask] = injection_weight * sr_x[fill_mask]
        latent_y[fill_mask] = injection_weight * sr_y[fill_mask]

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
#   Three-phase multi-scale blind deconvolution:
#     Phase 1: warmup (standard DCP on coarsest levels)
#     Bridge:  build SR reference from partially-deblurred image
#     Phase 2: SR-enhanced DCP on intermediate levels
#     Phase 3: standard DCP on fine levels
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_sr(y, lambda_dark, lambda_grad, opts,
                    sr_opts=None):
    """
    Multi-scale blind deconvolution with SelfExSR-enhanced intermediate
    pyramid levels.

    Parameters
    ----------
    y           : (H, W) grayscale blurred image, float64 [0,1]
    lambda_dark : float — L0 intensity prior weight
    lambda_grad : float — L0 gradient prior weight
    opts        : dict — standard DCP options:
                    'kernel_size', 'gamma_correct', 'xk_iter', 'k_thresh'
                  PLUS new SR-specific keys:
                    'n_warmup_levels' : int — coarsest levels with pure DCP
                                        before SR injection (default 2).
                    'n_sr_levels'     : int — intermediate levels using SR-
                                        enhanced blending (default 2).
                    'sr_alpha_min'    : float — blend alpha on coarsest SR
                                        level (default 0.5).
                    'sr_alpha_max'    : float — blend alpha on finest SR
                                        level (default 0.85).
                    'wiener_snr'      : float — Wiener filter noise-to-signal
                                        ratio for initial deblur (default 0.01).
    sr_opts     : dict or None — SelfExSR parameter overrides:
                    'SRF'       : int   — SR factor (default 2)
                    'numIter'   : int   — PatchMatch iterations (default 5)
                    'nIterBP'   : int   — back-projection iters (default 10)

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

    # ── Multi-scale pyramid (identical to original blind_deconv) ─────
    ret = np.sqrt(0.5)
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list = k1list + (k1list % 2 == 0)  # ensure odd
    k2list = k1list.copy()

    # ── SR integration parameters ────────────────────────────────────
    sr_alpha_min = opts.get('sr_alpha_min', 0.5)
    sr_alpha_max = opts.get('sr_alpha_max', 0.85)
    wiener_snr = opts.get('wiener_snr', 0.01)
    sr_downscale = opts.get('sr_downscale', 0.5)

    if sr_opts is None:
        sr_opts = {}
    SRF = sr_opts.get('SRF', 2)
    sr_num_iter = sr_opts.get('numIter', 3)
    sr_n_iter_bp = sr_opts.get('nIterBP', 5)

    # ── Compute phase boundaries ─────────────────────────────────────
    # Edge case: if pyramid is too small, fall back to pure DCP.
    if num_scales <= 2:
        n_warmup = num_scales
        n_sr_levels = 0
    else:
        n_warmup = min(opts.get('n_warmup_levels', 2), num_scales - 2)
        remaining = num_scales - n_warmup
        # Reserve at least 1 level (finest) for pure DCP
        n_sr_levels = min(opts.get('n_sr_levels', 2), remaining - 1)

    # warmup_boundary: scale indices >= this are warmup (standard DCP)
    warmup_boundary = num_scales - n_warmup
    # sr_boundary: scale indices >= this AND < warmup_boundary are SR-enhanced
    sr_boundary = warmup_boundary - n_sr_levels

    # ── State ────────────────────────────────────────────────────────
    threshold = None
    ks = None
    interim_latent = None
    y_sr_ref = None  # built after warmup

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

        # ────────────────────────────────────────────────────────────
        # Phase 1: WARMUP — standard DCP on the coarsest levels
        # ────────────────────────────────────────────────────────────
        if s >= warmup_boundary:
            ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
                ys, ks, lambda_dark, lambda_grad, threshold, opts
            )

            # After the LAST warmup level: build SR reference
            if s == warmup_boundary and n_sr_levels > 0:
                y_sr_ref = _build_sr_reference(
                    y, ks, k1list, k2list, ret, retv, s,
                    SRF, sr_num_iter, sr_n_iter_bp, wiener_snr,
                    sr_downscale=sr_downscale,
                )

        # ────────────────────────────────────────────────────────────
        # Phase 2: SR-ENHANCED — blend DCP latent with SR reference
        # ────────────────────────────────────────────────────────────
        elif s >= sr_boundary and y_sr_ref is not None:
            sr_ref_s = _downSmpImC(y_sr_ref, cret)

            alpha = compute_sr_blend_alpha(
                scale_idx=s,
                warmup_boundary=warmup_boundary,
                n_sr_levels=n_sr_levels,
                alpha_min=sr_alpha_min,
                alpha_max=sr_alpha_max,
            )

            ks, lambda_dark, lambda_grad, interim_latent = \
                blind_deconv_main_sr_enhanced(
                    ys, ks, lambda_dark, lambda_grad,
                    threshold, opts, sr_ref_s, alpha
                )

        # ────────────────────────────────────────────────────────────
        # Phase 3: FINE — standard DCP, no SR blending
        # ────────────────────────────────────────────────────────────
        else:
            ks, lambda_dark, lambda_grad, interim_latent = blind_deconv_main(
                ys, ks, lambda_dark, lambda_grad, threshold, opts
            )

        # Centre and clean kernel (same as original)
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


def _build_sr_reference(y, ks_warmup, k1list, k2list, ret, retv, warmup_level,
                        SRF, sr_num_iter, sr_n_iter_bp, wiener_snr,
                        sr_downscale=0.75):
    """
    Build the SR reference image after the warmup phase.

    1. Upsample the warmup kernel to full resolution.
    2. Wiener-deconvolve the original blurred image with conservative
       SNR → partially clean image (mild deblurring, minimal ringing).
    3. Optionally downscale for faster SelfExSR processing.
    4. Run SelfExSR on the partially clean image → enhanced reference.
    5. Convert to grayscale and resize to original dimensions.

    Parameters
    ----------
    y              : (H, W) blurred image, float64 [0,1]
    ks_warmup      : kernel estimate from the warmup phase
    k1list, k2list : kernel size lists for each pyramid level
    ret            : pyramid downsampling ratio (sqrt(0.5))
    retv           : array of cumulative ratios per level
    warmup_level   : scale index of the last warmup level
    SRF            : SelfExSR upscaling factor
    sr_num_iter    : PatchMatch iterations
    sr_n_iter_bp   : back-projection iterations
    wiener_snr     : Wiener filter noise-to-signal ratio (higher = more
                     conservative deblurring, less ringing)
    sr_downscale   : float in (0, 1] — resolution factor for the image
                     passed to SelfExSR.  0.75 = 75% resolution
                     (≈1.8× fewer pixels → faster PatchMatch).
                     (default 0.75)

    Returns
    -------
    y_sr_ref : (H, W) float64 [0,1] — SR-enhanced reference at original
               image resolution.
    """
    # ── Step 1: Upsample warmup kernel to full-resolution size ───────
    # The warmup kernel has size k1list[warmup_level].
    # The full-resolution kernel size is k1list[0].
    target_k1 = int(k1list[0])
    target_k2 = int(k2list[0])
    current_k_size = ks_warmup.shape[0]

    if current_k_size < target_k1:
        zoom_factor = target_k1 / current_k_size
        k_full = _resizeKer(ks_warmup, zoom_factor, target_k1, target_k2)
    else:
        k_full = ks_warmup.copy()

    # ── Step 2: Wiener deconvolution → partially clean image ──────────
    # Conservative SNR (0.05) ensures mild deblurring: the rough warmup
    # kernel is inaccurate so aggressive deconvolution would create
    # ringing artefacts that SelfExSR would hallucinate around.
    # Mild deblurring preserves real scene structure while partially
    # sharpening edges — exactly what SelfExSR needs for its
    # cross-scale self-similar patch search.
    S_deconv = wiener_filter(y, k_full, noise_snr=wiener_snr)
    S_deconv = np.clip(S_deconv, 0.0, 1.0)

    # ── Step 3: Optionally downscale for faster SR processing ────────
    # The SR reference is only used at intermediate pyramid levels
    # (50–35 % of original resolution), so running SelfExSR at full
    # resolution is wasteful.  Downscaling before SR and resizing
    # the output back is much faster with negligible quality loss
    # at the scales where the reference is actually used.
    if sr_downscale < 1.0:
        H_ds = max(int(y.shape[0] * sr_downscale), 32)
        W_ds = max(int(y.shape[1] * sr_downscale), 32)
        S_for_sr = resize_to_match(S_deconv, (H_ds, W_ds))
    else:
        S_for_sr = S_deconv

    # ── Step 4: Run SelfExSR on the (possibly downscaled) image ──────
    sr_opt = sr_init_opt(SRF)
    sr_opt['numIter'] = sr_num_iter
    sr_opt['nIterBP'] = sr_n_iter_bp

    # sr_demo expects [0,1] float, returns (H*SRF, W*SRF, 3) float32
    y_sr_hr = sr_demo(S_for_sr, SRF, opt=sr_opt)

    # ── Step 5: Convert to grayscale, resize to original ─────────────
    if y_sr_hr.ndim == 3:
        y_sr_gray = (0.2989 * y_sr_hr[:, :, 0]
                     + 0.5870 * y_sr_hr[:, :, 1]
                     + 0.1140 * y_sr_hr[:, :, 2])
    else:
        y_sr_gray = y_sr_hr

    y_sr_ref = resize_to_match(y_sr_gray.astype(np.float64), y.shape[:2])
    y_sr_ref = np.clip(y_sr_ref, 0.0, 1.0)

    return y_sr_ref
