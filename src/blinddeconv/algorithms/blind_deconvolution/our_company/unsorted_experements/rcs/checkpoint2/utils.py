"""
utils.py

Utility functions for the Fergus et al. (SIGGRAPH 2006) blind deconvolution.

Ported from MATLAB distribution code v1.2 by Rob Fergus, with contributions
from James Miskin, David MacKay, Yair Weiss, and Bryan Russell.

Reference:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Trans. Graphics (SIGGRAPH), 2006.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'same') / conv2(A, B, 'valid'):
        MATLAB conv2 performs TRUE 2-D convolution (flips the kernel).
        → scipy.signal.convolve2d(A, B, mode=..., boundary='fill')

    fft2 / ifft2:
        MATLAB fft2(A, M, N) zero-pads A to (M, N) then computes FFT.
        → np.fft.fft2(A, s=(M, N))

    Indexing:
        MATLAB is 1-based.  We convert to 0-based throughout.

    Column-major flattening:
        MATLAB A(:) flattens column-major (Fortran order).
        → A.ravel(order='F')  or  A.flatten(order='F')

    rgb2gray:
        MATLAB uses NTSC/YIQ luminance weights via the matrix
        T = inv([1.0 0.956 0.621; 1.0 -0.272 -0.647; 1.0 -1.106 1.703])
        First row of T gives: [0.2989, 0.5870, 0.1140].

    histeq / histmatch:
        MATLAB histeq(image, target_histogram) returns [J, T] where T
        is the cumulative mapping from [0,1] to [0,1].
        → Reimplemented manually.

    erfcx:
        The scaled complementary error function: erfcx(x) = exp(x^2)*erfc(x).
        → scipy.special.erfcx

    padarray(I, [p q], 'replicate'):
        → np.pad(I, ((p, p), (q, q)), mode='edge')

    padarray(I, [p q], 0, 'post'):
        → np.pad(I, ((0, p), (0, q)), mode='constant')

    psf2otf(psf, shape):
        Zero-pad, circshift centre to (0,0), then fft2.
        → Manual implementation matching MATLAB exactly.

    edgetaper(I, K):
        → Simulated via repeated blending at boundaries.

    imresize(I, scale, 'bilinear'):
        MATLAB's imresize uses antialiasing by default.
        → We use scipy.ndimage.zoom or cv2.resize, being careful
          about antialiasing behaviour.  For the multiscale pipeline
          the exact interpolation matters less than kernel normalisation.
"""

import numpy as np
from scipy.special import erfcx, erfc, gammaln
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import zoom
from numpy.fft import fft2, ifft2


# ═════════════════════════════════════════════════════════════════════════════
# PSF / OTF / kernel helpers
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape[:2]
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # MATLAB circshift amounts: -floor(size(psf)/2)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)


def delta_kernel(s: int) -> np.ndarray:
    """
    Create a delta (identity) kernel of size s×s.
    If s is even, size is made odd (s+1).

    Equivalent to MATLAB delta_kernel.m.
    """
    if s % 2 == 0:
        s = s + 1
    out = np.zeros((s, s), dtype=np.float64)
    c = s // 2
    out[c, c] = 1.0
    return out


# ═════════════════════════════════════════════════════════════════════════════
# rgb2gray_rob  (from rgb2gray_rob.m)
# ═════════════════════════════════════════════════════════════════════════════

def rgb2gray_rob(rgb: np.ndarray, saturation_level: float = 250.0) -> np.ndarray:
    """
    Convert RGB to grayscale using NTSC/YIQ luminance.
    Pixels that exceed *saturation_level* in ANY channel are set to 255.

    MATLAB: uses T = inv([1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703]),
    first row of T ≈ [0.2989, 0.5870, 0.1140].

    Input:  (H, W, 3) uint8 or float.
    Output: (H, W) same dtype conceptually, but returned as float64.
    """
    r = np.asarray(rgb, dtype=np.float64)
    sat_mask = (
        (r[:, :, 0] > saturation_level)
        | (r[:, :, 1] > saturation_level)
        | (r[:, :, 2] > saturation_level)
    )

    # NTSC luminance weights (first row of inv(NTSC matrix))
    T_row = np.array([0.29893602, 0.58704307, 0.11402090])
    gray = r[:, :, 0] * T_row[0] + r[:, :, 1] * T_row[1] + r[:, :, 2] * T_row[2]
    gray = np.clip(gray, 0.0, 255.0)

    # Set saturated pixels to 255
    gray[sat_mask] = 255.0
    return gray


# ═════════════════════════════════════════════════════════════════════════════
# Image reconstruction from gradients (Yair Weiss)
# ═════════════════════════════════════════════════════════════════════════════

def invDel2(isize: int) -> np.ndarray:
    """
    Inverse of 2D discrete Laplacian in Fourier domain.
    Returns a real-space kernel of size (isize, isize).

    Equivalent to MATLAB invDel2.m (Yair Weiss).
    """
    K = np.zeros((isize, isize), dtype=np.float64)
    c = isize // 2  # MATLAB uses isize/2 (1-based → 0-based: isize//2 - 1)
    # MATLAB: K(isize/2, isize/2) = -4  → 0-based: (c-1, c-1)
    K[c - 1, c - 1] = -4.0
    K[c, c - 1] = 1.0      # (isize/2+1, isize/2)
    K[c - 1, c] = 1.0      # (isize/2, isize/2+1)
    K[c - 2, c - 1] = 1.0  # (isize/2-1, isize/2)
    K[c - 1, c - 2] = 1.0  # (isize/2, isize/2-1)

    Khat = fft2(K)
    # Avoid division by zero
    zero_mask = (Khat == 0)
    Khat_safe = np.where(zero_mask, 1.0, Khat)
    invKhat = np.where(zero_mask, 0.0, 1.0 / Khat_safe)

    invK = np.real(ifft2(invKhat))
    invK = -invK

    # MATLAB: conv2(invK, [1 0 0; 0 0 0; 0 0 0], 'same') shifts by (0,0)
    # This is equivalent to np.roll by (-1, -1) then crop, or simply:
    shift_kernel = np.zeros((3, 3), dtype=np.float64)
    shift_kernel[0, 0] = 1.0
    invK = convolve2d(invK, shift_kernel, mode='same', boundary='fill')

    return invK


def reconsEdge3(dx: np.ndarray, dy: np.ndarray,
                invKhat: np.ndarray = None):
    """
    Poisson reconstruction from gradients dx, dy.
    Returns (im, invKhat).

    Equivalent to MATLAB reconsEdge3.m (Yair Weiss).

    Parameters
    ----------
    dx, dy   : (H, W) gradient images
    invKhat  : precomputed FFT of inverse Laplacian (optional)
    """
    sx, sy = dx.shape
    mxsize = max(sx, sy)

    if invKhat is None:
        invK = invDel2(2 * mxsize)
        invKhat = fft2(invK)

    # MATLAB: conv2(dx, fliplr([0 1 -1]), 'same')
    # fliplr([0 1 -1]) = [-1 1 0]
    imX = convolve2d(dx, np.array([[-1, 1, 0]], dtype=np.float64),
                     mode='same', boundary='fill')
    # MATLAB: conv2(dy, flipud([0;1;-1]), 'same')
    # flipud([0;1;-1]) = [[-1],[1],[0]]
    imY = convolve2d(dy, np.array([[-1], [1], [0]], dtype=np.float64),
                     mode='same', boundary='fill')

    imS = imX + imY
    imShat = fft2(imS, s=(2 * mxsize, 2 * mxsize))
    im = np.real(ifft2(imShat * invKhat))

    # MATLAB: im = im(mxsize+1:mxsize+sx, mxsize+1:mxsize+sy)
    # 0-based: im[mxsize:mxsize+sx, mxsize:mxsize+sy]
    im = im[mxsize:mxsize + sx, mxsize:mxsize + sy]
    return im, invKhat


# ═════════════════════════════════════════════════════════════════════════════
# normMDpdf  (from normMDpdf.m)
# ═════════════════════════════════════════════════════════════════════════════

def normMDpdf(x: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """
    Multivariate Gaussian PDF (log version for numerical stability
    is NOT used here — we return the actual PDF as in MATLAB).

    Parameters
    ----------
    x   : (nDims, nPoints)
    mu  : (nDims,) or (nDims, 1)
    sig : (nDims, nDims) covariance matrix

    Returns
    -------
    y   : (nPoints,)  probability density
    """
    mu = mu.ravel()
    nDims = x.shape[0]
    nPoints = x.shape[1]

    i_sig = np.linalg.inv(sig)
    det_sig = np.linalg.det(sig)
    d = ((2 * np.pi) ** (-nDims / 2.0)) / np.sqrt(det_sig)

    tt = x - mu[:, np.newaxis]  # (nDims, nPoints)
    ttt = i_sig @ tt            # (nDims, nPoints)
    e = np.sum(tt * ttt, axis=0)  # (nPoints,)

    y = d * np.exp(-0.5 * e)
    return y


# ═════════════════════════════════════════════════════════════════════════════
# clip_image  (from clip_image.m)
# ═════════════════════════════════════════════════════════════════════════════

def clip_image(im: np.ndarray, minval: float, maxval: float) -> np.ndarray:
    """Clip image to [minval, maxval]."""
    return np.clip(im, minval, maxval)


# ═════════════════════════════════════════════════════════════════════════════
# histmatch  (from histmatch.m)
# ═════════════════════════════════════════════════════════════════════════════

def _histeq_mapping(gray_in: np.ndarray, target_hist: np.ndarray):
    """
    Reimplementation of MATLAB histeq(I, hgram).

    Parameters
    ----------
    gray_in     : (H, W) float64 image in [0, 1]
    target_hist : (256,) target histogram (counts)

    Returns
    -------
    J : (H, W) uint8 equalised image
    T : (256,) mapping from [0..255] index → [0, 1] output
    """
    # Normalise target histogram to a CDF
    target_cdf = np.cumsum(target_hist).astype(np.float64)
    if target_cdf[-1] > 0:
        target_cdf /= target_cdf[-1]

    # Compute CDF of input image
    gray_uint8 = np.clip(np.round(gray_in * 255.0), 0, 255).astype(np.int32)
    input_hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    input_cdf = np.cumsum(input_hist)
    if input_cdf[-1] > 0:
        input_cdf /= input_cdf[-1]

    # Build mapping T:  for each input level i find closest target level j
    # such that target_cdf[j] >= input_cdf[i]
    T = np.zeros(256, dtype=np.float64)
    for i in range(256):
        j = np.searchsorted(target_cdf, input_cdf[i])
        j = min(j, 255)
        T[i] = j / 256.0  # MATLAB returns T in [0, 1], scaled by 1/256

    J = T[gray_uint8]
    return J, T


def histmatch(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of *in_img* (float64 [0,1]) to *reference* (uint8 [0,255]).
    Returns uint8 image [0, 255].

    Equivalent to MATLAB histmatch.m.
    """
    in_f = np.asarray(in_img, dtype=np.float64)
    ref = np.asarray(reference)

    # Convert to grayscale if needed
    if in_f.ndim == 3 and in_f.shape[2] != 1:
        gray_in = 0.2989 * in_f[:, :, 0] + 0.5870 * in_f[:, :, 1] + 0.1140 * in_f[:, :, 2]
    else:
        gray_in = in_f if in_f.ndim == 2 else in_f[:, :, 0]

    if ref.ndim == 3 and ref.shape[2] != 1:
        gray_ref = (0.2989 * ref[:, :, 0] + 0.5870 * ref[:, :, 1] + 0.1140 * ref[:, :, 2]).astype(np.float64)
    else:
        gray_ref = ref.astype(np.float64) if ref.ndim == 2 else ref[:, :, 0].astype(np.float64)

    # Reference histogram
    hist_reference = np.bincount(
        np.clip(np.round(gray_ref).astype(np.int32).ravel(), 0, 255),
        minlength=256
    ).astype(np.float64)

    # Get mapping T via histeq of gray input
    _, T = _histeq_mapping(gray_in, hist_reference)

    # Apply T to each channel
    nch = in_f.shape[2] if in_f.ndim == 3 else 1
    if in_f.ndim == 2:
        in_f = in_f[:, :, np.newaxis]

    out = np.zeros_like(in_f)
    for a in range(nch):
        q = in_f[:, :, a]
        # MATLAB: qm = interp1([0:255]/256, T, q(:))
        # Linearly interpolate T at query points q
        x_knots = np.arange(256) / 256.0
        qm = np.interp(q.ravel(), x_knots, T)
        out[:, :, a] = (256.0 * qm.reshape(q.shape))

    out = np.clip(out, 0, 255).astype(np.uint8)
    if nch == 1:
        out = out[:, :, 0]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# fix_image  (from fix_image.m)
# ═════════════════════════════════════════════════════════════════════════════

def fix_image(in_img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Normalise *in_img* to [0,1] then histogram-equalise to *reference*.

    Equivalent to MATLAB fix_image.m.
    """
    SPACING = 0.05

    ref_im = reference.astype(np.float64)
    ref_max = ref_im.max()
    if ref_max > 0:
        ref_im = ref_im / ref_max

    x_bins = np.arange(0, 1.0 + SPACING, SPACING)
    hist_ref, _ = np.histogram(ref_im.ravel(), bins=np.append(x_bins, np.inf))
    hist_ref = hist_ref[:len(x_bins)]

    m = in_img.min()
    in_shift = in_img - m
    in_max = in_shift.max()
    if in_max > 0:
        in_norm = in_shift / in_max
    else:
        in_norm = in_shift

    # Simple histogram equalisation with the reference histogram
    # Using the same _histeq_mapping approach
    out, _ = _histeq_mapping(in_norm, hist_ref)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Automatic patch selector  (from automatic_patch_selector.m)
# ═════════════════════════════════════════════════════════════════════════════

def automatic_patch_selector(im: np.ndarray, patch_size: int,
                             weight: float,
                             sat_mask: np.ndarray):
    """
    Automatically select the most informative image patch for kernel
    inference.  Maximises variance × (1/(saturation)) × centre-weight.

    Parameters
    ----------
    im         : (H, W) float64 image (assumed [0, 255])
    patch_size : int, size of square patch
    weight     : float, centre bias strength
    sat_mask   : (H, W) binary mask of saturated pixels

    Returns
    -------
    out_im         : (patch_size, patch_size) extracted patch
    patch_location : (x, y) — 0-based coordinates
    """
    SMOOTH_SIGMA = 3

    II, JJ = im.shape

    # Centre weighting mask
    yy, xx = np.mgrid[0:II, 0:JJ]
    xx = xx - round(JJ / 2)
    yy = yy - round(II / 2)
    centre_weight_mask = np.exp(-weight / (JJ ** 2) * (xx ** 2 + yy ** 2))

    II2 = II * 2
    JJ2 = JJ * 2

    # Shift by patch_size using delta_kernel convolution in FFT domain
    dk = delta_kernel(patch_size)
    centre_weight_mask = np.real(
        ifft2(fft2(centre_weight_mask, s=(II2, JJ2))
              * fft2(dk, s=(II2, JJ2)))
    )

    # Patch mask (averaging filter)
    pmask = np.ones((patch_size, patch_size), dtype=np.float64) / (patch_size ** 2)

    # Variance within each patch
    ei2 = np.real(ifft2(fft2(im ** 2, s=(II2, JJ2)) * fft2(pmask, s=(II2, JJ2))))
    mu2 = np.real(ifft2(fft2(im, s=(II2, JJ2)) * fft2(pmask, s=(II2, JJ2)))) ** 2
    w = ei2 - mu2

    # Saturation convolution
    q = np.real(ifft2(fft2(sat_mask.astype(np.float64), s=(II2, JJ2))
                      * fft2(pmask, s=(II2, JJ2))))

    mean_im = im.mean()
    combined = centre_weight_mask * w / (q * mean_im ** 2 + 1.0)

    # Smooth response
    from scipy.ndimage import gaussian_filter
    combined_smooth = np.real(
        ifft2(
            fft2(combined, s=(II2, JJ2))
            * fft2(
                _fspecial_gaussian(8, SMOOTH_SIGMA, (II2, JJ2)),
            )
        )
    )

    # Crop to avoid edge effects
    # MATLAB: combined(patch_size:II/2, patch_size:JJ/2) — 1-based
    # → 0-based: [patch_size-1 : II, patch_size-1 : JJ]
    combined_crop = combined_smooth[patch_size - 1:II, patch_size - 1:JJ]

    mm = np.argmax(combined_crop)
    sy, sx = np.unravel_index(mm, combined_crop.shape)

    # MATLAB: patch_location = [sx sy] - 1;  (1-based → offset)
    # In Python 0-based, the crop already shifted by (patch_size-1),
    # so actual location in original image:
    patch_location = np.array([sx, sy])  # 0-based within cropped array
    # Adjust back: sx_orig = sx + patch_size - 1, etc.  But MATLAB does
    # patch_location = [sx sy] - 1  then uses it directly.
    # To match MATLAB: patch_location is used as (x-1)-based start index.

    # Chop out patch — MATLAB:  im(sy-1:sy-2+patch_size, sx-1:sx-2+patch_size)
    # In 0-based terms:
    py = sy  # 0-based row in cropped → row in original = sy + patch_size - 1
    px = sx
    out_im = im[py:py + patch_size, px:px + patch_size]

    return out_im, patch_location


def _fspecial_gaussian(hsize: int, sigma: float,
                       fft_shape: tuple = None) -> np.ndarray:
    """
    Gaussian filter kernel, equivalent to fspecial('gaussian', hsize, sigma).
    If fft_shape is given, zero-pad for use in FFT domain.
    """
    half = hsize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]
    # NOTE: np.mgrid gives hsize+1 if hsize is even; we slice to hsize
    if y.shape[0] > hsize:
        y = y[:hsize, :hsize]
        x = x[:hsize, :hsize]
    g = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    if fft_shape is not None:
        padded = np.zeros(fft_shape, dtype=np.float64)
        padded[:g.shape[0], :g.shape[1]] = g
        return padded

    return g


# ═════════════════════════════════════════════════════════════════════════════
# GaussianMixtures1D — EM for 1D Mixture of Gaussians (from GaussianMixtures1D.m)
# ═════════════════════════════════════════════════════════════════════════════

def GaussianMixtures1D(x: np.ndarray, nComponents: int):
    """
    EM for 1D zero-mean Mixture of Gaussians.

    Parameters
    ----------
    x           : (nPoints,) data
    nComponents : number of mixture components

    Returns
    -------
    mu     : (1, nComponents)  means (always 0)
    sigma  : (1, 1, nComponents) variances
    weight : (nComponents,) mixture weights
    log_likelihood : list of log-likelihoods per iteration
    """
    MAX_ITERATIONS = 100
    LIKELIHOOD_CHANGE_THRESHOLD = 1e-5

    x = x.ravel().astype(np.float64)
    nPoints = len(x)

    mu = np.zeros((1, nComponents), dtype=np.float64)
    sigma = np.zeros((1, 1, nComponents), dtype=np.float64)
    weight = np.ones(nComponents, dtype=np.float64) / nComponents

    # Initialise variances: random large values
    for a in range(nComponents):
        sigma[0, 0, a] = 1e6 - np.random.rand() * 1e6
        if sigma[0, 0, a] <= 0:
            sigma[0, 0, a] = 1.0

    sigma[0, 0, 0] = 1e6  # First component very wide

    resp = np.zeros((nComponents, nPoints), dtype=np.float64)
    likelihoods = np.zeros((nComponents, nPoints), dtype=np.float64)
    log_likelihood_list = []
    delta_lh = np.inf

    for iteration in range(MAX_ITERATIONS):
        # ── E-Step ──
        for c in range(nComponents):
            s = sigma[0, 0, c]
            if s <= 0:
                s = 1e-10
            normaliser = 1.0 / np.sqrt(2 * np.pi * s)
            offset = x - mu[0, c]
            exponent = offset ** 2 / s
            likelihoods[c, :] = weight[c] * normaliser * np.exp(-0.5 * exponent)

        # Log-likelihood
        total = np.sum(likelihoods, axis=0)
        total = np.maximum(total, 1e-300)
        ll = np.mean(np.log(total))
        log_likelihood_list.append(ll)

        if iteration > 0:
            delta_lh = log_likelihood_list[-1] - log_likelihood_list[-2]

        # Responsibilities
        for c in range(nComponents):
            resp[c, :] = likelihoods[c, :] / total

        # ── M-Step ──
        for c in range(nComponents):
            total_resp_c = np.sum(resp[c, :])
            if total_resp_c < 1e-10:
                total_resp_c = 1e-10

            weight[c] = total_resp_c / nPoints
            mu[0, c] = 0.0  # Fixed zero mean

            offset = x - mu[0, c]
            u = np.sqrt(resp[c, :])
            new_sigma = np.sum((u * offset) ** 2) / total_resp_c
            sigma[0, 0, c] = new_sigma + 1e-5

        # Keep first component very wide for first 10 iterations
        if iteration < 9:
            sigma[0, 0, 0] = 1e6

        # ── Convergence check ──
        if delta_lh < LIKELIHOOD_CHANGE_THRESHOLD and iteration > 0:
            break

    return mu, sigma, weight, log_likelihood_list


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper — approximate MATLAB edgetaper
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(im: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 1) -> np.ndarray:
    """
    Approximate MATLAB's edgetaper: blend boundaries of *im* using *kernel*
    autocorrelation, to reduce ringing in FFT-based deconvolution.

    MATLAB's built-in edgetaper does a single-pass blend, so n_tapers=1.

    Parameters
    ----------
    im      : (H, W) or (H, W, C) image
    kernel  : 2-D kernel
    n_tapers : number of tapering iterations (default 1, matching MATLAB)

    Returns
    -------
    tapered : same shape as im
    """
    # Compute kernel autocorrelation
    kh, kw = kernel.shape
    ac = convolve2d(kernel, kernel[::-1, ::-1], mode='full')
    ac = ac / ac.max()

    # Build 1D taper profiles from the central row/col of autocorrelation
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    taper_y = ac[:, cx]
    taper_x = ac[cy, :]

    result = im.astype(np.float64).copy()
    for _ in range(n_tapers):
        if result.ndim == 2:
            blurred = _fft_convolve_same(result, kernel)
        else:
            blurred = np.stack(
                [_fft_convolve_same(result[:, :, c], kernel)
                 for c in range(result.shape[2])],
                axis=2
            )
        # Build 2D blend mask
        H, W = result.shape[:2]
        alpha_y = np.ones(H, dtype=np.float64)
        alpha_x = np.ones(W, dtype=np.float64)

        half_ky = len(taper_y) // 2
        half_kx = len(taper_x) // 2
        for i in range(min(half_ky, H)):
            v = taper_y[half_ky - i]
            alpha_y[i] = min(alpha_y[i], v)
            alpha_y[H - 1 - i] = min(alpha_y[H - 1 - i], v)
        for j in range(min(half_kx, W)):
            v = taper_x[half_kx - j]
            alpha_x[j] = min(alpha_x[j], v)
            alpha_x[W - 1 - j] = min(alpha_x[W - 1 - j], v)

        alpha = alpha_y[:, np.newaxis] * alpha_x[np.newaxis, :]
        if result.ndim == 3:
            alpha = alpha[:, :, np.newaxis]

        result = alpha * result + (1.0 - alpha) * blurred

    return result


def _fft_convolve_same(im: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """FFT-based 'same'-size convolution (2D)."""
    out = fftconvolve(im, kernel, mode='same')
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Ensemble Learning helpers (from Miskin & MacKay code)
# ═════════════════════════════════════════════════════════════════════════════

def train_ensemble_get(c: int, dimensions: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Extract data for class *c* (1-based!) from flat vector *x*.

    MATLAB: if (c>1) start=dimensions(1:c-1,1)'*dimensions(1:c-1,2);
    Returns: (dimensions(c,1), dimensions(c,2)) reshaped array.
    """
    c_idx = c - 1  # Convert to 0-based
    if c_idx > 0:
        start = int(np.sum(dimensions[:c_idx, 0] * dimensions[:c_idx, 1]))
    else:
        start = 0
    n_rows = int(dimensions[c_idx, 0])
    n_cols = int(dimensions[c_idx, 1])
    return x[start:start + n_rows * n_cols].reshape(n_rows, n_cols)


def train_ensemble_put(c: int, dimensions: np.ndarray,
                       x: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """
    Put *cx* data for class *c* (1-based!) back into flat vector *x*.
    """
    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(dimensions[:c_idx, 0] * dimensions[:c_idx, 1]))
    else:
        start = 0
    n_rows = int(dimensions[c_idx, 0])
    n_cols = int(dimensions[c_idx, 1])
    x_out = x.copy()
    x_out[start:start + n_rows * n_cols] = cx.reshape(n_rows * n_cols)
    return x_out


def train_ensemble_get_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray) -> np.ndarray:
    """
    Extract mixture weight data for class *c* (1-based!).
    Returns: (dim[c,0], dim[c,1], dim[c,2]) reshaped array.
    """
    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(np.prod(dimensions[:c_idx, :3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c_idx, :3]))
    d0 = int(dimensions[c_idx, 0])
    d1 = int(dimensions[c_idx, 1])
    d2 = int(dimensions[c_idx, 2])
    return log_lambda_x[start:start + n].reshape(d0, d1, d2)


def train_ensemble_put_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray,
                              c_log_lambda_x: np.ndarray) -> np.ndarray:
    """
    Put mixture weight data for class *c* (1-based!) back into flat vector.
    """
    c_idx = c - 1
    if c_idx > 0:
        start = int(np.sum(np.prod(dimensions[:c_idx, :3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c_idx, :3]))
    out = log_lambda_x.copy()
    out[start:start + n] = c_log_lambda_x.ravel()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# train_ensemble_rectified5 (from train_ensemble_rectified5.m)
# ═════════════════════════════════════════════════════════════════════════════

def train_ensemble_rectified5(x1: np.ndarray, x2: np.ndarray,
                              dist_type: int):
    """
    Evaluate expectations under ensemble distributions.

    Parameters
    ----------
    x1, x2   : natural parameters of Q(x)
    dist_type : 0=Gaussian, 1=Laplacian(→rectified Gaussian),
                2=Rectified Gaussian, 3=Discrete{-1,+1}

    Returns
    -------
    Hx  : <log Q(x)> minus constants from P(x)
    mx  : <x>
    mx2 : <x^2>
    """
    # Clamp x2 away from zero to prevent divide-by-zero / NaN
    x2 = np.maximum(np.asarray(x2, dtype=np.float64), 1e-300)
    x1 = np.asarray(x1, dtype=np.float64)

    if dist_type == 0:
        # Gaussian
        mx = x1 / x2
        mx2 = x1 ** 2 / x2 ** 2 + 1.0 / x2
        Hx = -0.5 + 0.5 * np.log(x2)

    elif dist_type == 1 or dist_type == 2:
        # Laplacian (type 1) or Rectified Gaussian (type 2)
        # posterior is rectified Gaussian in both cases
        sqrt_2x2 = np.sqrt(2.0 * x2)
        t = -x1 / sqrt_2x2  # can be very negative or very positive
        erf_table = erfcx(t)  # erfcx(t) = exp(t^2)*erfc(t)

        mask_low = (t <= 25)

        # Avoid division by zero in erf_table
        safe_erf = np.where(erf_table == 0, 1e-300, erf_table)
        # Use a moderate floor so that 1/safe_x1 stays finite
        safe_x1 = np.where(np.abs(x1) < 1e-100, np.copysign(1e-100, x1 + 1e-300), x1)

        # np.where evaluates both branches; suppress warnings from
        # the unselected branch to avoid false alarms.
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            # Low-t branch (t <= 25): standard formulas
            mx_low = x1 / x2 + np.sqrt(2.0 / (np.pi * x2)) / safe_erf
            mx2_low = (x1 ** 2 / x2 ** 2 + 1.0 / x2
                        + 2.0 * x1 / x2 / np.sqrt(2.0 * np.pi * x2) / safe_erf)

            # High-t branch (t > 25): asymptotic expansions
            mx_high = (-1.0 / safe_x1 + 2.0 * x2 / safe_x1 ** 3
                        - 10.0 * x2 ** 2 / safe_x1 ** 5)
            mx2_high = (2.0 / safe_x1 ** 2 - 10.0 * x2 / safe_x1 ** 4
                         + 74.0 * x2 ** 2 / safe_x1 ** 6)

        mx = np.where(mask_low, mx_low, mx_high)
        mx2 = np.where(mask_low, mx2_low, mx2_high)

        # Compute erfc(min(t,25)) via scipy.special.erfc directly
        # (avoids the NaN-prone exp(-t^2)*erfcx(t) for very negative t)
        erfc_clamped = np.maximum(erfc(np.minimum(t, 25.0)), 1e-300)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Hx_low = (-np.log(erfc_clamped)
                       + 0.5 * np.log(2.0 * x2 / np.pi) - 0.5
                       + x1 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            Hx_high = (np.log(np.maximum(np.abs(safe_x1), 1e-300)) - 1.0
                        + 2.0 * x2 / safe_x1 ** 2
                        - 15.0 * x2 ** 2 / safe_x1 ** 4 / 2.0
                        + 148.0 * x2 ** 3 / safe_x1 ** 6 / 3.0)

        Hx = np.where(t < 25, Hx_low, Hx_high)

        if dist_type == 2:
            Hx = Hx + 0.5 * np.log(np.pi / 2.0)

    elif dist_type == 3:
        # Discrete {-1, +1}
        mx = np.tanh(x1)
        mx2 = np.ones_like(x1)
        Hx = x1 * mx - np.abs(x1) - np.log(1.0 + np.exp(-2.0 * np.abs(x1))) + np.log(2.0)

    elif dist_type == 4:
        # Laplacian prior — two rectified Gaussians (same formulas as type 2)
        sqrt_2x2 = np.sqrt(2.0 * x2)
        t = -x1 / sqrt_2x2
        erf_table = erfcx(t)

        mask_low = (t <= 25)
        safe_erf = np.where(erf_table == 0, 1e-300, erf_table)
        safe_x1 = np.where(np.abs(x1) < 1e-100, np.copysign(1e-100, x1 + 1e-300), x1)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            mx_low = x1 / x2 + np.sqrt(2.0 / (np.pi * x2)) / safe_erf
            mx2_low = (x1 ** 2 / x2 ** 2 + 1.0 / x2
                        + 2.0 * x1 / x2 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            mx_high = (-1.0 / safe_x1 + 2.0 * x2 / safe_x1 ** 3
                        - 10.0 * x2 ** 2 / safe_x1 ** 5)
            mx2_high = (2.0 / safe_x1 ** 2 - 10.0 * x2 / safe_x1 ** 4
                         + 74.0 * x2 ** 2 / safe_x1 ** 6)

        mx = np.where(mask_low, mx_low, mx_high)
        mx2 = np.where(mask_low, mx2_low, mx2_high)

        erfc_clamped = np.maximum(erfc(np.minimum(t, 25.0)), 1e-300)

        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            Hx_low = (-np.log(erfc_clamped)
                       + 0.5 * np.log(2.0 * x2 / np.pi) - 0.5
                       + x1 / np.sqrt(2.0 * np.pi * x2) / safe_erf)
            Hx_high = (np.log(np.maximum(np.abs(safe_x1), 1e-300)) - 1.0
                        + 2.0 * x2 / safe_x1 ** 2
                        - 15.0 * x2 ** 2 / safe_x1 ** 4 / 2.0
                        + 148.0 * x2 ** 3 / safe_x1 ** 6 / 3.0)

        Hx = np.where(t < 25, Hx_low, Hx_high)
        Hx = Hx + 0.5 * np.log(np.pi / 2.0)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return Hx, mx, mx2


# ═════════════════════════════════════════════════════════════════════════════
# move_level  (from move_level.m)
# ═════════════════════════════════════════════════════════════════════════════

def move_level(mx: np.ndarray, me: np.ndarray,
               K: int, L: int, M: int, N: int,
               mode: str = 'matlab_bilinear',
               resize_step: float = np.sqrt(2),
               center: bool = False) -> tuple:
    """
    Upsample kernel and image estimates to the next scale level.

    Parameters
    ----------
    mx : (h, w) or (h, w, c) — image gradient estimate
    me : (kh, kw) — kernel estimate
    K, L       : target kernel size
    M, N       : target image size
    mode       : interpolation mode (default 'matlab_bilinear')
    resize_step : scale factor (default sqrt(2))
    center     : if True, centre the kernel by its centre of mass

    Returns
    -------
    mx_new, me_new
    """
    if center:
        me = me / me.sum()
        rows = np.arange(me.shape[0])
        cols = np.arange(me.shape[1])
        mu_y = np.sum(rows * me.sum(axis=1))
        mu_x = np.sum(cols * me.sum(axis=0))

        offset_y = round(me.shape[0] // 2 - mu_y)
        offset_x = round(me.shape[1] // 2 - mu_x)

        shift_kernel = np.zeros((abs(offset_y) * 2 + 1,
                                 abs(offset_x) * 2 + 1), dtype=np.float64)
        shift_kernel[abs(offset_y) + offset_y,
                     abs(offset_x) + offset_x] = 1.0

        me = convolve2d(me, shift_kernel, mode='same', boundary='fill')

        if mx.ndim == 3:
            for c in range(mx.shape[2]):
                mx[:, :, c] = convolve2d(
                    mx[:, :, c], shift_kernel[::-1, ::-1],
                    mode='same', boundary='fill'
                )
        else:
            mx = convolve2d(mx, shift_kernel[::-1, ::-1],
                            mode='same', boundary='fill')

    # Resize
    if mx.ndim == 2:
        zoom_y = M / mx.shape[0]
        zoom_x = N / mx.shape[1]
        mx_new = zoom(mx, (zoom_y, zoom_x), order=1)
    else:
        zoom_y = M / mx.shape[0]
        zoom_x = N / mx.shape[1]
        mx_new = np.stack(
            [zoom(mx[:, :, c], (zoom_y, zoom_x), order=1)
             for c in range(mx.shape[2])],
            axis=2
        )

    zoom_ky = K / me.shape[0]
    zoom_kx = L / me.shape[1]
    me_new = zoom(me, (zoom_ky, zoom_kx), order=1)

    # Crop to exact size (zoom might produce ±1 pixel mismatch)
    if mx_new.ndim == 2:
        mx_new = mx_new[:M, :N]
    else:
        mx_new = mx_new[:M, :N, :]
    me_new = me_new[:K, :L]

    # Normalise kernel
    me_sum = me_new.sum()
    if me_sum > 0:
        me_new = me_new / me_sum

    return mx_new, me_new


# ═════════════════════════════════════════════════════════════════════════════
# imresize helper
# ═════════════════════════════════════════════════════════════════════════════

def imresize(im: np.ndarray, scale_or_shape, method: str = 'bilinear') -> np.ndarray:
    """
    Resize image, approximating MATLAB imresize behaviour.

    Parameters
    ----------
    im             : 2D or 3D array
    scale_or_shape : float (scale factor) or tuple (target_rows, target_cols)
    method         : 'bilinear' (order=1) or 'nearest' (order=0) or 'bicubic' (order=3)
    """
    order = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}.get(method, 1)

    if isinstance(scale_or_shape, (int, float)):
        scale = float(scale_or_shape)
        if im.ndim == 2:
            return zoom(im, scale, order=order)
        else:
            return np.stack(
                [zoom(im[:, :, c], scale, order=order)
                 for c in range(im.shape[2])],
                axis=2
            )
    else:
        target_h, target_w = scale_or_shape
        if im.ndim == 2:
            zoom_h = target_h / im.shape[0]
            zoom_w = target_w / im.shape[1]
            out = zoom(im, (zoom_h, zoom_w), order=order)
            return out[:target_h, :target_w]
        else:
            zoom_h = target_h / im.shape[0]
            zoom_w = target_w / im.shape[1]
            out = np.stack(
                [zoom(im[:, :, c], (zoom_h, zoom_w), order=order)
                 for c in range(im.shape[2])],
                axis=2
            )
            return out[:target_h, :target_w, :]


# ═════════════════════════════════════════════════════════════════════════════
# estimate_priors2  (from estimate_priors2.m)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_priors_from_images(images: list, num_components: int,
                                num_scales: int,
                                gradient_type: str = 'haar') -> list:
    """
    Estimate MoG prior parameters from a set of sharp images.

    Parameters
    ----------
    images         : list of (H, W) float64 images
    num_components : number of Gaussian components
    num_scales     : number of scale levels
    gradient_type  : 'haar' or 'steer'

    Returns
    -------
    priors : list of dicts with keys 'pi' and 'gamma'
             priors[s]['pi']    : (1, num_components) mixture weights
             priors[s]['gamma'] : (1, num_components) inverse variances
    """
    SCALE_STEP = np.sqrt(2)
    MAX_IM_SIZE = 700
    STEP_SIZE = 1

    x_bins = np.arange(-200, 201, STEP_SIZE).astype(np.float64)

    priors = []
    for b in range(num_scales):
        scale = SCALE_STEP ** (-b)
        b_all = []

        for im in images:
            if im.ndim == 3:
                im = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]
            im = im.astype(np.float64)

            imy, imx = im.shape
            scale_factor = MAX_IM_SIZE / max(imx, imy)
            im = imresize(im, scale_factor, 'bilinear')

            if gradient_type == 'haar':
                b_x = convolve2d(im, np.array([[1, -1]], dtype=np.float64),
                                 mode='valid')
                b_y = convolve2d(im, np.array([[1], [-1]], dtype=np.float64),
                                 mode='valid')
                if scale != 1.0:
                    b_x = imresize(b_x, scale, 'bilinear')
                    b_y = imresize(b_y, scale, 'bilinear')
            else:
                raise NotImplementedError("Steerable pyramid not implemented")

            b_all.extend(b_x.ravel().tolist())
            b_all.extend(b_y.ravel().tolist())

        b_all = np.array(b_all, dtype=np.float64)
        mu, sigma, weight, ll = GaussianMixtures1D(b_all, num_components)

        prior_entry = {
            'pi': weight.reshape(1, -1).copy(),
            'gamma': (1.0 / sigma[0, 0, :]).reshape(1, -1).copy(),
        }
        priors.append(prior_entry)

    return priors


# ═════════════════════════════════════════════════════════════════════════════
# Greenspan super-resolution  (from greenspan.m / create_greenspan_settings.m)
# ═════════════════════════════════════════════════════════════════════════════

def create_greenspan_settings(**kwargs) -> dict:
    """
    Create default settings for Greenspan nonlinear enhancement.

    Possible keyword arguments:
        lo_filt, c, s, bp, factor
    """
    # binomial5 filter (MATLAB: binomialFilter(5) * binomialFilter(5)')
    bfilt = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    lo_filt = np.outer(bfilt, bfilt)

    S = {
        'lo_filt': lo_filt,
        'c': 0.4,
        's': 5,
        'bp': 1,
        'factor': 1,
    }
    S.update(kwargs)
    return S


def greenspan(im: np.ndarray, S: dict):
    """
    Greenspan–Anderson–Akber nonlinear frequency-space enhancement.

    Parameters
    ----------
    im : (H, W) image
    S  : settings dict from create_greenspan_settings

    Returns
    -------
    en : enhanced image
    L0 : inferred high-frequency band
    """
    z = 2 ** S['factor']
    lo_filt = S['lo_filt']

    # L1 = im - rconv2(im, lo_filt)  — reflected-boundary convolution
    im_smooth = convolve2d(im, lo_filt, mode='same', boundary='symm')
    L1 = im - im_smooth

    # upConv: upsample by z with reflected boundary
    target_shape = (z * im.shape[0], z * im.shape[1])
    L0 = imresize(L1, target_shape, 'bilinear') * (z ** 2)

    maxL0 = np.abs(L0).max() if np.abs(L0).max() > 0 else 1.0
    L0 = S['s'] * clip_image(L0, -(1 - S['c']) * maxL0, (1 - S['c']) * maxL0)

    if S['bp']:
        L0_smooth = convolve2d(L0, lo_filt, mode='same', boundary='symm')
        L0 = L0 - L0_smooth

    en = imresize(im, target_shape, 'bilinear') * (z ** 2) + L0
    return en, L0
