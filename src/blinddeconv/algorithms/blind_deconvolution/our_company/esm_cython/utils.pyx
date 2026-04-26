"""
utils.py

Utility functions for the ESM (Enhanced Sparse Model) blind deconvolution.

Ported from MATLAB code by Chen et al. (ECCV 2020).
Reference:
    L. Chen, F. Fang, S. Lei, F. Li, G. Zhang: "Enhanced Sparse Model
    for Blind Deblurring", ECCV 2020.

These utilities are the shared "cho_code" + bilateral_filter helpers from
the MATLAB reference implementation located at
    ECCV20_enhanced_sparse_model/cho_code/*
    ECCV20_enhanced_sparse_model/bilateral_filter.m

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'valid'):
        MATLAB conv2 performs TRUE convolution (flips kernel B).
        → scipy.signal.convolve2d(A, B, mode='valid')
        Both produce the same output size (M-mk+1, N-nk+1).

    padarray(I, [p p], 'replicate'):
        → np.pad(I, ((p,p),(p,p)), mode='edge')

    fspecial('gaussian', hsize, sigma):
        hsize×hsize Gaussian kernel, normalised to sum = 1.
        → Manual construction (_fspecial_gaussian).

    histc(x, edges):
        Same length as edges; last bin includes right edge exactly.
        → _histc helper using np.searchsorted.

    psf2otf(psf, shape):
        Zero-pad PSF, circularly shift centre to (0,0), fft2.
        → Manual implementation matching MATLAB exactly.

    dst / idst (Discrete Sine Transform — Type-I):
        MATLAB's dst is Type-I DST.  scipy.fft.dstn type=1 matches
        up to a constant factor that cancels in forward/inverse.

    interp2(..., 'linear') (MATLAB):
        NaN for out-of-bound samples.
        → scipy.ndimage.map_coordinates with cval=0.
"""

import numpy as np
from scipy.signal import convolve2d, fftconvolve
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn


# ═════════════════════════════════════════════════════════════════════════════
# PSF ↔ OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0, 0).
    3. Return fft2.

    MATLAB circshift amounts: -floor(size(psf)/2) for each dim.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# fftconv  (from cho_code/fftconv.m)
# ═════════════════════════════════════════════════════════════════════════════

def fftconv(I: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """
    FFT-based circular convolution.
    Equivalent to MATLAB cho_code/fftconv.m (uniform-blur branch).

        cI = real(ifft2(fft2(I) .* psf2otf(filt, size(I))))

    For 2-D input I (single channel).  The MATLAB file also has a 3-channel
    dispatch branch; we handle that via a simple loop for completeness.

    Parameters
    ----------
    I : (H, W) or (H, W, D) float array
    filt : 2-D kernel

    Returns
    -------
    cI : same shape as I
    """
    if I.ndim == 3:
        out = np.zeros_like(I)
        otf = psf2otf(filt, I.shape[:2])
        for c in range(I.shape[2]):
            out[:, :, c] = np.real(np.fft.ifft2(np.fft.fft2(I[:, :, c]) * otf))
        return out

    otf = psf2otf(filt, I.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(I) * otf))


# ═════════════════════════════════════════════════════════════════════════════
# opt_fft_size  (from cho_code/opt_fft_size.m)
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None  # module-level cache (like MATLAB persistent)


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """Build the LUT of optimal FFT sizes (products of 2,3,5,7 * {1,11,13})."""
    lut = np.zeros(lut_size + 1, dtype=np.int64)  # 1-indexed

    e2 = 1
    while e2 <= lut_size:
        e3 = e2
        while e3 <= lut_size:
            e5 = e3
            while e5 <= lut_size:
                e7 = e5
                while e7 <= lut_size:
                    if e7 <= lut_size:
                        lut[e7] = e7
                    if e7 * 11 <= lut_size:
                        lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= lut_size:
                        lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2

    # Fill gaps: for each position, use the next larger valid size
    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = lut[i]
        else:
            lut[i] = nn
    return lut


def opt_fft_size(n) -> np.ndarray:
    """
    Compute optimal FFT data length(s).  Equivalent to MATLAB opt_fft_size.m.

    Returns -1 for sizes above LUT range.
    """
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None:
        _OPT_FFT_LUT = _build_opt_fft_lut()

    n = np.asarray(n, dtype=np.int64)
    scalar_input = n.ndim == 0
    n = np.atleast_1d(n)

    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n)
    for i in range(n.size):
        nn = n.flat[i]
        if 1 <= nn <= lut_size:
            m.flat[i] = _OPT_FFT_LUT[nn]
        else:
            m.flat[i] = -1

    if scalar_input:
        return int(m.flat[0])
    return m


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu  (from cho_code/wrap_boundary_liu.m)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """
    Solve Laplace equation with Dirichlet boundary conditions via DST-I.
    Matches the nested solve_min_laplacian in wrap_boundary_liu.m.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    # Keep only boundary values
    boundary_image[1:-1, 1:-1] = 0.0

    # Discrete Laplacian of boundary at interior points
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]
        + boundary_image[1:H - 1, 0:W - 2]
        + boundary_image[0:H - 2, 1:W - 1]
        + boundary_image[2:H,     1:W - 1]
    )
    f1 = -f_bp

    # Interior only
    f2 = f1[1:H - 1, 1:W - 1]

    # 2-D DST-I (forward).  The implicit scale factor cancels with idstn.
    f2sin = dstn(f2, type=1)

    # Eigenvalues of the 5-point Laplacian under DST-I
    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom

    img_tt = idstn(f3, type=1)

    img_direct = boundary_image.copy()
    img_direct[1:H - 1, 1:W - 1] = img_tt

    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image with circularly smooth boundaries for FFT-based deconvolution.
    Equivalent to MATLAB wrap_boundary_liu.m (Liu & Jia, ICIP 2008).

    alpha = 1 always (hard-coded in MATLAB source).
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        # --- r_A: (2*alpha + H_w) × W ---------------------------------
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]

        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        r_A[alpha:alpha + H_w, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

        # MATLAB (alpha=1): r_A(alpha:end-alpha+1,:) covers all rows
        r_A = _solve_min_laplacian(r_A)
        A = r_A

        # --- r_B: H × (2*alpha + W_w) ---------------------------------
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        r_B[-1, alpha:alpha + W_w] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

        r_B = _solve_min_laplacian(r_B)
        B = r_B

        # --- r_C: (2*alpha + H_w) × (2*alpha + W_w) -------------------
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        r_C = _solve_min_laplacian(r_C)
        C = r_C

        # Strip wrapping rows/columns (alpha=1 ⇒ drop one border row/col
        # in the right places to reassemble the final padded image)
        A = A[:H_w, :]
        B = B[:, 1:W_w + 1]
        C = C[1:H_w + 1, 1:W_w + 1]

        # MATLAB: ret = [img, B; A, C]
        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# Conjugate Gradient  (from cho_code/conjgrad.m)
# ═════════════════════════════════════════════════════════════════════════════

def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Conjugate gradient.  Solves A x = b with A defined by ax_func(x, param).
    """
    x = x.copy()
    r = b - ax_func(x, func_param)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(max_it):
        Ap = ax_func(p, func_param)
        pAp = np.sum(p * Ap)
        if abs(pAp) < 1e-30:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x


# ═════════════════════════════════════════════════════════════════════════════
# adjust_psf_center  (from cho_code/adjust_psf_center.m)
# ═════════════════════════════════════════════════════════════════════════════

def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    """
    Shift the PSF so its centre of mass coincides with the geometric centre.

    MATLAB uses 1-based meshgrid, integer-rounded shifts, bilinear interp2,
    and NaN → 0 for out-of-bound samples.  We replicate this with
    map_coordinates(cval=0, order=1).
    """
    rows, cols = psf.shape

    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                       np.arange(1, rows + 1, dtype=np.float64))

    total = np.sum(psf)
    if total == 0:
        return psf

    xc1 = np.sum(psf * X)
    yc1 = np.sum(psf * Y)

    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0

    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)

    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                     np.arange(cols, dtype=np.float64),
                                     indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    result = map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                             order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)


# ═════════════════════════════════════════════════════════════════════════════
# threshold_pxpy_v1  (from cho_code/threshold_pxpy_v1.m)
# ═════════════════════════════════════════════════════════════════════════════

def _histc(data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    MATLAB histc(data, edges).

    Counts values where edges[k] <= x < edges[k+1], and the last bin
    additionally includes x == edges[-1].  Output length = len(edges).
    """
    data = np.asarray(data).ravel()
    if data.size == 0:
        return np.zeros(len(edges), dtype=np.int64)

    indices = np.searchsorted(edges, data, side='right') - 1
    # Values exactly at the last edge go to the last bin
    indices[data == edges[-1]] = len(edges) - 1
    # Out of range → a sentinel slot we will drop
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size, threshold=None):
    """
    Selective gradient thresholding used inside the kernel-estimation loop.
    Equivalent to MATLAB cho_code/threshold_pxpy_v1.m.

    Returns (px, py, threshold) where px, py are weak-gradient-zeroed
    derivatives of *latent*.

    Derivative filters:   dx = [-1 1; 0 0]   dy = [-1 0; 1 0]
    Applied via 'valid' convolution (true convolution, kernel flipped).
    """
    b_estimate_threshold = threshold is None
    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    dy = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        # MATLAB: pd = atan(py./px) — principal branch, range (-pi/2, pi/2)
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        # MATLAB: pm_steps = 0:0.00006:2
        pm_steps = np.arange(0.0, 2.0 + 0.00006 / 2.0, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        # MATLAB: cumsum(flipud(histc(...)))
        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = (np.max(psf_size)
                        if hasattr(psf_size, '__len__') else psf_size)
        th = max(psf_size_val * 20, 10)

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                # MATLAB: threshold = pm_steps(end - t + 1)
                # t is 1-based in MATLAB → Python loop index t (0-based) maps
                # to MATLAB's (t+1).  pm_steps(end - t) in 0-based.
                threshold = pm_steps[len(pm_steps) - 1 - t]
                break

    m = pm < threshold
    while np.all(m):
        threshold = threshold * 0.81
        m = pm < threshold

    px[m] = 0.0
    py[m] = 0.0

    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold


# ═════════════════════════════════════════════════════════════════════════════
# bilateral_filter  (from bilateral_filter.m)
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """MATLAB fspecial('gaussian', size, sigma) — centred, sum-normalised."""
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """
    Bilateral filter for grayscale / multi-channel non-RGB images.
    Equivalent to MATLAB bilateral_filter.m when the colour branch is not
    taken (i.e. D != 3 or a diff image is passed).  ESM's pipeline calls:
        bilateral_filter(diff, 3, 0.1)
    where ``diff`` is the difference of two deconvolution results, typically
    3-channel; we follow the non-LAB branch exactly:
        lab = img;  sigma *= sqrt(d)
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1

    h, w, d = img.shape
    img = img.astype(np.float32)

    lab = img.copy()
    sigma = sigma * np.sqrt(d)

    fr = int(np.ceil(sigma_s * 3))

    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    p_lab = np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')

    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)

    spatial_weight = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * sigma

    for y in range(-fr, fr + 1):
        for x in range(-fr, fr + 1):
            w_s = spatial_weight[y + fr, x + fr]

            n_img = p_img[fr + y:fr + y + h, fr + x:fr + x + w, :]
            n_lab = p_lab[fr + y:fr + y + h, fr + x:fr + x + w, :]

            f_diff = lab - n_lab
            f_dist = np.sum(f_diff ** 2, axis=2)

            w_f = np.exp(-0.5 * f_dist / ss)
            w_t = w_s * w_f

            r_img += n_img * w_t[:, :, np.newaxis]
            w_sum += w_t

    r_img = r_img / w_sum[:, :, np.newaxis]

    if was_2d:
        return r_img[:, :, 0]
    return r_img
