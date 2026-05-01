"""
utils.py

Utility functions for the PAM (Perrone-Favaro) TV blind deconvolution.

Ported from MATLAB code by Daniele Perrone and Paolo Favaro.
Reference:
    D. Perrone and P. Favaro: "Total Variation Blind Deconvolution:
    The Devil is in the Details", CVPR 2014.
    Technical Report: perrone2014tvTR.pdf

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'valid'):
        MATLAB conv2 performs TRUE convolution (flips kernel B).
        → scipy.signal.convolve2d(A, B, mode='valid') does the same.
        Result sizes match: 'full'  → (M+Mk-1, N+Nk-1)
                            'same'  → (M, N)
                            'valid' → (M-Mk+1, N-Nk+1)

    conv2fft (custom):
        Uses fftshift/ifftshift in MATLAB:
            ahat = fftshift(fftn(a, padded_size));
            bhat = fftshift(fftn(b, padded_size));
            c = real(ifftn(ifftshift(ahat .* bhat)));
        fftshift(A).*fftshift(B) == fftshift(A.*B), so the shifts
        cancel.  We keep them for exact correspondence.

    Indexing differences:
        MATLAB f([2:end end], :, :) → shift rows down, replicate last
            → np.concatenate([f[1:], f[-1:]], axis=0)
        MATLAB f([1 1:end-1], :, :) → shift rows up, replicate first
            → np.concatenate([f[:1], f[:-1]], axis=0)
        Analogous for columns on axis=1.

        MATLAB 1-based slice f(a:b) includes b.
            → Python 0-based slice f[a-1 : b]

    round(x):
        MATLAB R2014a: rounds half away from zero (round(2.5)=3).
        Python round(): banker's rounding (round(2.5)=2).
        → We use int(floor(x + 0.5)) for MATLAB-compatible rounding.

    padarray(I, [p1 p2], 'replicate'):
        → np.pad(I, pad_width, mode='edge')
        For 3D: np.pad(I, ((p1,p1),(p2,p2),(0,0)), mode='edge')

    imresize(I, [M N], 'Method', 'bicubic'):
        MATLAB uses Keys cubic kernel with anti-aliasing on downsample.
        → skimage.transform.resize(order=3, anti_aliasing=True)
        is the closest available approximation.

    rot90(k, 2):
        Rotate 180°.  → np.rot90(k, 2)  (identical semantics)

    interp2(X, Y, V, Xq, Yq):
        MATLAB: X = column coords, Y = row coords.
        → scipy.interpolate.RegularGridInterpolator((y_1d, x_1d), V)
        query with (row, col) pairs.
"""

import numpy as np
from scipy.signal import convolve2d as _scipy_convolve2d
from scipy.fft import fft2 as _sp_fft2, ifft2 as _sp_ifft2, next_fast_len as _sp_next_fast_len


# ═════════════════════════════════════════════════════════════════════════════
# MATLAB-compatible round  (rounds half away from zero)
# ═════════════════════════════════════════════════════════════════════════════

def _matlab_round(x):
    """Round half away from zero, matching MATLAB R2014a round()."""
    return int(np.floor(x + 0.5))


# ═════════════════════════════════════════════════════════════════════════════
# conv2fft  (from lib/conv2fft.m)
# ═════════════════════════════════════════════════════════════════════════════

def conv2fft(a: np.ndarray, b: np.ndarray, mode: str = 'full') -> np.ndarray:
    """
    2D convolution via FFT, matching MATLAB conv2fft.m by Paolo Favaro.

    Performs true convolution (equivalent to MATLAB conv2 but via FFT).
    Both inputs must be 2D.

    Implementation note (vs original MATLAB):
        The original conv2fft.m uses fftshift/ifftshift around the FFT
        product.  When the FFTs are padded to exactly the linear-convolution
        size  full_shape = (M+Mk-1, N+Nk-1)  (which they always are here),
        that shift dance is redundant: plain  ifft(fft(a)*fft(b))  already
        yields the linear convolution starting at index 0.  We drop it and
        instead pad to  next_fast_len  to get FFT-friendly dimensions, then
        crop back to full_shape.  This is mathematically equivalent up to
        FP rounding and ~2-3x faster (pocketfft via scipy.fft + composite
        sizes vs near-prime sizes that would arise from raw M+Mk-1).

    Parameters
    ----------
    a : (M, N) input array
    b : (Mk, Nk) kernel
    mode : 'full', 'same', or 'valid'

    Returns
    -------
    c : convolution result
        'full'  → (M+Mk-1, N+Nk-1)
        'same'  → (M, N)
        'valid' → (M-Mk+1, N-Nk+1)
    """
    Nx1, Nx2 = a.shape
    NKx1, NKx2 = b.shape

    full_h = Nx1 + NKx1 - 1
    full_w = Nx2 + NKx2 - 1

    # FFT-friendly padded size (composite, fast for pocketfft).
    pad_h = _sp_next_fast_len(full_h)
    pad_w = _sp_next_fast_len(full_w)

    A = _sp_fft2(a, s=(pad_h, pad_w))
    B = _sp_fft2(b, s=(pad_h, pad_w))
    c = _sp_ifft2(A * B).real[:full_h, :full_w]

    if mode == 'same':
        r0 = NKx1 // 2
        c0 = NKx2 // 2
        c = c[r0:r0 + Nx1, c0:c0 + Nx2]
    elif mode == 'valid':
        c = c[NKx1 - 1:Nx1, NKx2 - 1:Nx2]

    return c


# ═════════════════════════════════════════════════════════════════════════════
# conv2_matlab  (scipy wrapper matching MATLAB conv2 interface)
# ═════════════════════════════════════════════════════════════════════════════

def conv2_matlab(a: np.ndarray, b: np.ndarray,
                 mode: str = 'full') -> np.ndarray:
    """
    2D convolution matching MATLAB conv2(a, b, mode).

    Both MATLAB conv2 and scipy convolve2d perform TRUE convolution
    (kernel is flipped).  This is a thin wrapper for readability.
    Inputs must be 2D.
    """
    return _scipy_convolve2d(a, b, mode=mode)


# ═════════════════════════════════════════════════════════════════════════════
# grad_tv_cc  (from lib/gradTVcc.m)
# ═════════════════════════════════════════════════════════════════════════════

def grad_tv_cc(f: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """
    Gradient of Total Variation, channel-by-channel.

    Computes ∂TV/∂u using forward/backward/mixed finite differences
    with replicated-boundary conditions.  Each colour channel is
    processed independently.

    Matches MATLAB gradTVcc.m by Paolo Favaro.

    Parameters
    ----------
    f : (M, N) or (M, N, C) image
    epsilon : smoothing constant for numerical stability

    Returns
    -------
    divTV : same shape as f
    """
    squeeze_out = (f.ndim == 2)
    if f.ndim == 2:
        f = f[:, :, np.newaxis]

    # ── Forward differences (replicate last row / last col) ──────────
    # MATLAB: f([2:end end],:,:) - f
    fxforw = np.concatenate([f[1:, :, :], f[-1:, :, :]], axis=0) - f
    # MATLAB: f(:,[2:end end],:) - f
    fyforw = np.concatenate([f[:, 1:, :], f[:, -1:, :]], axis=1) - f

    # ── Backward differences (replicate first row / first col) ───────
    # MATLAB: f - f([1 1:end-1],:,:)
    fxback = f - np.concatenate([f[:1, :, :], f[:-1, :, :]], axis=0)
    # MATLAB: f - f(:,[1 1:end-1],:)
    fyback = f - np.concatenate([f[:, :1, :], f[:, :-1, :]], axis=1)

    # ── Mixed differences ────────────────────────────────────────────
    # MATLAB: fxmixd = f([2:end end],[1 1:end-1],:) - f(:,[1 1:end-1],:)
    #   → (shift rows down, replicate last) then (shift cols left, replicate first)
    #     minus (shift cols left, replicate first)
    f_down = np.concatenate([f[1:, :, :], f[-1:, :, :]], axis=0)
    f_left = np.concatenate([f[:, :1, :], f[:, :-1, :]], axis=1)
    f_down_left = np.concatenate(
        [f_down[:, :1, :], f_down[:, :-1, :]], axis=1
    )
    fxmixd = f_down_left - f_left

    # MATLAB: fymixd = f([1 1:end-1],[2:end end],:) - f([1 1:end-1],:,:)
    #   → (shift rows up, replicate first) then (shift cols right, replicate last)
    #     minus (shift rows up, replicate first)
    f_up = np.concatenate([f[:1, :, :], f[:-1, :, :]], axis=0)
    f_up_right = np.concatenate(
        [f_up[:, 1:, :], f_up[:, -1:, :]], axis=1
    )
    fymixd = f_up_right - f_up

    # ── Divergence (channel-by-channel, matching MATLAB loop) ────────
    divTV = np.zeros_like(f)
    for cc in range(f.shape[2]):
        fxf = fxforw[:, :, cc]
        fyf = fyforw[:, :, cc]
        fxb = fxback[:, :, cc]
        fyb = fyback[:, :, cc]
        fxm = fxmixd[:, :, cc]
        fym = fymixd[:, :, cc]

        divTV[:, :, cc] = (
            (fxf + fyf)
            / np.maximum(epsilon, np.sqrt(fxf ** 2 + fyf ** 2))
            - fxb
            / np.maximum(epsilon, np.sqrt(fxb ** 2 + fym ** 2))
            - fyb
            / np.maximum(epsilon, np.sqrt(fxm ** 2 + fyb ** 2))
        )

    if squeeze_out:
        divTV = divTV[:, :, 0]

    return divTV


# ═════════════════════════════════════════════════════════════════════════════
# gamma_correction  (from lib/gammaCorrection.m)
# ═════════════════════════════════════════════════════════════════════════════

def gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma correction: I_c = I^gamma.  Matches gammaCorrection.m."""
    return np.power(image, gamma)


# ═════════════════════════════════════════════════════════════════════════════
# imresize  (MATLAB-compatible image resize)
# ═════════════════════════════════════════════════════════════════════════════

def imresize(image: np.ndarray, target_size: tuple,
             method: str = 'bicubic') -> np.ndarray:
    """
    Resize image to target_size = (rows, cols).

    Approximates MATLAB imresize behaviour:
      - 'bicubic' uses cubic interpolation with anti-aliasing on
        downsample (MATLAB default).
      - We use skimage.transform.resize which provides anti-aliasing.

    Parameters
    ----------
    image : (M, N) or (M, N, C)
    target_size : (new_M, new_N)
    method : 'bicubic', 'bilinear', or 'nearest'

    Returns
    -------
    resized : (new_M, new_N) or (new_M, new_N, C)
    """
    from skimage.transform import resize as sk_resize

    order_map = {'bicubic': 3, 'bilinear': 1, 'nearest': 0}
    order = order_map.get(method, 3)

    is_downsampling = (target_size[0] < image.shape[0]
                       or target_size[1] < image.shape[1])

    if image.ndim == 3:
        out_shape = (target_size[0], target_size[1], image.shape[2])
    else:
        out_shape = target_size

    return sk_resize(
        image, out_shape, order=order,
        anti_aliasing=is_downsampling,
        preserve_range=True,
        mode='edge',
    )


# ═════════════════════════════════════════════════════════════════════════════
# build_pyramid  (from lib/buildPyramid.m)
# ═════════════════════════════════════════════════════════════════════════════

def build_pyramid(f: np.ndarray, MK: int, NK: int,
                  final_lambda: float, lambda_multiplier: float,
                  interp_method: str = 'bicubic',
                  scale_multiplier: float = 1.1,
                  largest_lambda: float = 0.11):
    """
    Build coarse-to-fine pyramid of images, kernel sizes, and λ values.

    Matches MATLAB buildPyramid.m by Daniele Perrone.

    Pyramid layout:
        Index 0 = finest scale (original resolution).
        Index num_scales-1 = coarsest scale.
    Iteration in coarseToFine goes from coarsest → finest.

    Parameters
    ----------
    f : (M, N) or (M, N, C) blurry image
    MK, NK : kernel height, width at finest scale
    final_lambda : λ at finest scale
    lambda_multiplier : λ growth factor per coarser level
    interp_method : interpolation method string
    scale_multiplier : kernel shrink ratio per coarser level
    largest_lambda : upper bound for λ

    Returns
    -------
    fp       : list[ndarray]  — images at each scale
    Mp, Np   : list[int]      — image rows, cols per scale
    MKp, NKp : list[int]      — kernel rows, cols per scale
    lambdas  : list[float]    — λ per scale
    num_scales : int           — total number of pyramid levels
    """
    M, N = f.shape[:2]
    smallest_scale = 3

    fp = [f]
    Mp = [M]
    Np = [N]
    MKp = [MK]
    NKp = [NK]
    lambdas = [final_lambda]

    while (MKp[-1] > smallest_scale
           and NKp[-1] > smallest_scale
           and lambdas[-1] * lambda_multiplier < largest_lambda):

        # λ for the coarser level
        new_lambda = lambdas[-1] * lambda_multiplier

        # Kernel size at coarser level
        # MATLAB: round(MKp{scales-1}/scaleMultiplier)
        new_MK = _matlab_round(MKp[-1] / scale_multiplier)
        new_NK = _matlab_round(NKp[-1] / scale_multiplier)

        # Make kernel dimensions odd
        if new_MK % 2 == 0:
            new_MK -= 1
        if new_NK % 2 == 0:
            new_NK -= 1

        # Prevent stagnation (same size as previous level)
        if new_NK == NKp[-1]:
            new_NK -= 2
        if new_MK == MKp[-1]:
            new_MK -= 2

        # Enforce minimum kernel size
        if new_NK < smallest_scale:
            new_NK = smallest_scale
        if new_MK < smallest_scale:
            new_MK = smallest_scale

        # Image size scaled proportionally to kernel size change
        # MATLAB: factorM = MKp{scales-1}/MKp{scales}
        factor_M = MKp[-1] / new_MK
        factor_N = NKp[-1] / new_NK

        # MATLAB: round(Mp{scales-1}/factorM)
        new_M = _matlab_round(Mp[-1] / factor_M)
        new_N = _matlab_round(Np[-1] / factor_N)

        # Make image dimensions odd
        if new_M % 2 == 0:
            new_M -= 1
        if new_N % 2 == 0:
            new_N -= 1

        # Resize from ORIGINAL image (not previous coarser scale)
        resized = imresize(f, (new_M, new_N), method=interp_method)

        fp.append(resized)
        Mp.append(new_M)
        Np.append(new_N)
        MKp.append(new_MK)
        NKp.append(new_NK)
        lambdas.append(new_lambda)

    num_scales = len(fp)
    return fp, Mp, Np, MKp, NKp, lambdas, num_scales


# ═════════════════════════════════════════════════════════════════════════════
# comp_upto_shift  (from comp_upto_shift.m by Anat Levin)
# ═════════════════════════════════════════════════════════════════════════════

def comp_upto_shift(I1: np.ndarray, I2: np.ndarray):
    """
    SSD comparison invariant to sub-pixel shift.

    Matches MATLAB comp_upto_shift.m by Anat Levin.
    Used only for evaluation on the Levin benchmark, not by the core
    blind-deconvolution pipeline.

    Parameters
    ----------
    I1, I2 : (M, N) grayscale images to compare

    Returns
    -------
    ssde : float — minimum SSD across all tested shifts
    tI1  : (M', N') — I1 shifted to best-match I2
    """
    from scipy.interpolate import RegularGridInterpolator

    maxshift = 5
    # MATLAB: shifts = -5:0.25:5  (41 values)
    shifts = np.arange(-5, 5.25, 0.25)

    # Crop images (convert MATLAB 1-based to Python 0-based)
    # MATLAB: I2 = I2(16:end-15, 16:end-15)
    I2c = I2[15:-15, 15:-15].copy()
    # MATLAB: I1 = I1(16-maxshift:end-15+maxshift, ...)
    #       = I1(11:end-10, 11:end-10)
    I1c = I1[15 - maxshift:I1.shape[0] - 15 + maxshift,
             15 - maxshift:I1.shape[1] - 15 + maxshift].copy()

    N1, N2 = I2c.shape

    # Grid for I1c in 1-based MATLAB coordinates
    # MATLAB: meshgrid(1-maxshift : N2+maxshift, 1-maxshift : N1+maxshift)
    x_1d = np.arange(1 - maxshift, N2 + maxshift + 1, dtype=np.float64)
    y_1d = np.arange(1 - maxshift, N1 + maxshift + 1, dtype=np.float64)

    # RegularGridInterpolator expects (row_coords, col_coords)
    interp_func = RegularGridInterpolator(
        (y_1d, x_1d), I1c.astype(np.float64),
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    # Query base grid (1-based, size of I2c)
    # MATLAB: meshgrid(1:N2, 1:N1)
    gx0, gy0 = np.meshgrid(
        np.arange(1, N2 + 1, dtype=np.float64),
        np.arange(1, N1 + 1, dtype=np.float64),
    )

    # Brute-force search over all shift combinations
    ssdem = np.full((len(shifts), len(shifts)), np.inf)
    for i, si in enumerate(shifts):
        for j, sj in enumerate(shifts):
            gxn = gx0 + si
            gyn = gy0 + sj
            pts = np.stack([gyn.ravel(), gxn.ravel()], axis=-1)
            tI1_flat = interp_func(pts)
            tI1_tmp = tI1_flat.reshape(N1, N2)
            ssdem[i, j] = np.nansum((tI1_tmp - I2c) ** 2)

    # Find minimum
    ssde = ssdem.min()
    idx = np.unravel_index(ssdem.argmin(), ssdem.shape)

    # Reconstruct best-shifted I1
    gxn = gx0 + shifts[idx[0]]
    gyn = gy0 + shifts[idx[1]]
    pts = np.stack([gyn.ravel(), gxn.ravel()], axis=-1)
    tI1 = interp_func(pts).reshape(N1, N2)

    return ssde, tI1
