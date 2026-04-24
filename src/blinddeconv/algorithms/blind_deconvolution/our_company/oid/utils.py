"""
utils.py

Utility functions for the OID (Outlier Identifying and Discarding) blind
deconvolution algorithm.

Ported from MATLAB code released with the paper:
    L. Chen, F. Fang, J. Zhang, J. Liu, G. Zhang:
    "OID: Outlier Identifying and Discarding in Blind Image Deblurring",
    ECCV 2020.

Original MATLAB sources:
    outlier_public/main_code/utils/
        adjust_psf_center.m
        conjgrad.m
        dst.m / idst.m
        fftconv.m
        opt_fft_size.m
        threshold_pxpy_v1.m
        wrap_boundary_liu.m

MATLAB -> Python conversion notes (CRITICAL):
    - MATLAB conv2(A,B,'valid') is TRUE convolution (flips kernel B).
      scipy.signal.convolve2d(A,B,'valid') does the same.
    - MATLAB 'diff(A,1,2)' -> np.diff(A, n=1, axis=1)  (columns)
      MATLAB 'diff(A,1,1)' -> np.diff(A, n=1, axis=0)  (rows)
    - MATLAB psf2otf centres the PSF at (1,1) (1-based = (0,0) 0-based)
      via circshift by -floor(size(psf)/2).
    - MATLAB's dst / idst in this project are used only by wrap_boundary_liu
      (the Poisson solver); the scaling cancels out in forward/inverse.
    - MATLAB indexing is column-major (Fortran order); 1-based.
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn


# ═════════════════════════════════════════════════════════════════════════════
# PSF <-> OTF conversions  (MATLAB built-ins psf2otf / otf2psf)
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at (0, 0).
    3. Return fft2.

    MATLAB psf2otf circshift amounts: -floor(size(psf)/2) per dim.
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
# fftconv  (from outlier_public/main_code/utils/fftconv.m)
# ═════════════════════════════════════════════════════════════════════════════

def fftconv(I: np.ndarray, filt: np.ndarray, b_otf: bool = False) -> np.ndarray:
    """
    FFT-based circular convolution.  Equivalent to MATLAB fftconv.m.

    MATLAB:
        cI = real(ifft2(fft2(I) .* psf2otf(filt, size(I))))
    If *filt* is already an OTF of the same shape as *I*, pass b_otf=True.
    For colour input (H,W,3) the operation is applied channel-wise.
    """
    if I.ndim == 3 and I.shape[2] == 3:
        H, W = I.shape[:2]
        if b_otf:
            otf = filt
        else:
            otf = psf2otf(filt, (H, W))
        out = np.empty_like(I, dtype=np.float64)
        for ch in range(3):
            out[:, :, ch] = np.real(np.fft.ifft2(np.fft.fft2(I[:, :, ch]) * otf))
        return out

    if b_otf:
        return np.real(np.fft.ifft2(np.fft.fft2(I) * filt))
    return np.real(np.fft.ifft2(np.fft.fft2(I) * psf2otf(filt, I.shape)))


# ═════════════════════════════════════════════════════════════════════════════
# opt_fft_size  (from outlier_public/main_code/utils/opt_fft_size.m)
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None  # module-level cache (MATLAB persistent)


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """Build the LUT of FFT-friendly sizes (products of primes 2,3,5,7,11,13)."""
    lut = np.zeros(lut_size + 1, dtype=np.int64)

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

    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = i
        else:
            lut[i] = nn
    return lut


def opt_fft_size(n):
    """
    Optimal FFT data length(s).  Equivalent to MATLAB opt_fft_size.m.
    Accepts int or array-like; returns same shape.
    """
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None:
        _OPT_FFT_LUT = _build_opt_fft_lut()

    n_arr = np.asarray(n, dtype=np.int64)
    scalar_input = n_arr.ndim == 0
    n_arr = np.atleast_1d(n_arr)

    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n_arr)
    for i in range(n_arr.size):
        nn = int(n_arr.flat[i])
        if 1 <= nn <= lut_size:
            m.flat[i] = _OPT_FFT_LUT[nn]
        else:
            m.flat[i] = -1

    if scalar_input:
        return int(m.flat[0])
    return m


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu  (from outlier_public/main_code/utils/wrap_boundary_liu.m)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """
    Poisson solver with Dirichlet boundary conditions via 2-D DST.
    Equivalent to the nested MATLAB solve_min_laplacian.

    MATLAB's dst / idst are DST-I.  scipy.fft.dstn(..., type=1) is DST-I
    as well; it differs by a constant scaling factor from MATLAB, but
    the factor cancels exactly in the forward/inverse composition used
    here (dstn -> divide by eigenvalues -> idstn).
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    # Keep only boundary pixels; zero the interior
    boundary_image[1:-1, 1:-1] = 0.0

    # Laplacian applied to boundary pixels at interior locations
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]
        + boundary_image[1:H - 1, 0:W - 2]
        + boundary_image[0:H - 2, 1:W - 1]
        + boundary_image[2:H,     1:W - 1]
    )

    f1 = -f_bp
    f2 = f1[1:H - 1, 1:W - 1]

    # 2-D DST-I in the interior
    f2sin = dstn(f2, type=1)

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
    Pad an image so its boundaries are circularly smooth (for FFT deconv).
    Equivalent to MATLAB wrap_boundary_liu.m.  Always uses alpha = 1.

    Parameters
    ----------
    img : (H, W) or (H, W, Ch) float image
    img_size : (H_out, W_out) target padded size

    Returns
    -------
    ret : padded image, shape (H_out, W_out) or (H_out, W_out, Ch)
    """
    squeeze_out = (img.ndim == 2)
    if squeeze_out:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = int(img_size[0]), int(img_size[1])
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        # --- r_A: (2 + H_w) x W ---
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]

        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        r_A[alpha:alpha + H_w, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

        A2 = _solve_min_laplacian(r_A)
        r_A = A2
        A = r_A

        # --- r_B: H x (2 + W_w) ---
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        r_B[-1, alpha:alpha + W_w] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        # --- r_C: (2 + H_w) x (2 + W_w) ---
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        # MATLAB trims: A -> H_w rows; B -> W_w cols; C -> H_w x W_w
        A = A[:H_w, :]
        B = B[:, 1:W_w + 1]
        C = C[1:H_w + 1, 1:W_w + 1]

        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if squeeze_out:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# conjgrad  (from outlier_public/main_code/utils/conjgrad.m)
# ═════════════════════════════════════════════════════════════════════════════

def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Conjugate gradient solver for A*x = b with A defined implicitly by
    ax_func(x, func_param).  Ports MATLAB conjgrad.m exactly.
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
# adjust_psf_center  (from outlier_public/main_code/utils/adjust_psf_center.m)
# ═════════════════════════════════════════════════════════════════════════════

def adjust_psf_center(psf: np.ndarray) -> np.ndarray:
    """
    Shift PSF so that its centre of mass coincides with the geometric centre.
    Equivalent to MATLAB adjust_psf_center.m (uses bilinear interp with
    NaN -> 0 for out-of-bounds samples).
    """
    rows, cols = psf.shape

    X, Y = np.meshgrid(np.arange(1, cols + 1, dtype=np.float64),
                       np.arange(1, rows + 1, dtype=np.float64))

    total = np.sum(psf)
    if total == 0:
        return psf.copy()

    xc1 = np.sum(psf * X)
    yc1 = np.sum(psf * Y)

    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0

    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)

    out_rows, out_cols = np.meshgrid(
        np.arange(rows, dtype=np.float64),
        np.arange(cols, dtype=np.float64),
        indexing='ij',
    )
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    result = map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                             order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)


# ═════════════════════════════════════════════════════════════════════════════
# threshold_pxpy_v1  (from outlier_public/main_code/utils/threshold_pxpy_v1.m)
# ═════════════════════════════════════════════════════════════════════════════

def _histc(data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Equivalent to MATLAB histc(data, edges).

    MATLAB bin k counts values where edges(k) <= x < edges(k+1); the last
    bin also includes values equal to edges(end).  Output length == len(edges).
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    edges = np.asarray(edges, dtype=np.float64)
    n = len(edges)

    indices = np.searchsorted(edges, data, side='right') - 1
    # Values equal to the last edge go into the last bin
    indices[data == edges[-1]] = n - 1
    # Anything outside [edges[0], edges[-1]] is discarded
    valid = (data >= edges[0]) & (data <= edges[-1])
    indices = indices[valid]

    counts = np.bincount(indices, minlength=n)
    return counts[:n].astype(np.int64)


def threshold_pxpy_v1(latent: np.ndarray, psf_size, threshold=None):
    """
    Gradient thresholding used in kernel estimation.
    Equivalent to MATLAB cho_code/threshold_pxpy_v1.m.

    Returns
    -------
    px, py : gradient images with weak gradients zeroed
    threshold : possibly updated threshold value
    """
    b_estimate_threshold = threshold is None
    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    # MATLAB conv2 is TRUE convolution; scipy convolve2d does the same.
    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        # MATLAB uses atan (not atan2) -> range (-pi/2, pi/2).
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        pm_steps = np.arange(0.0, 2.0 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        # MATLAB: H = cumsum(flipud(histc(pm(mask), pm_steps)))
        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = np.max(psf_size) if hasattr(psf_size, '__len__') else psf_size
        th = max(psf_size_val * 20, 10)

        n_steps = len(pm_steps)
        for t in range(n_steps):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                # MATLAB: threshold = pm_steps(end - t + 1) with t 1-based.
                # Python t is 0-based -> index is (n_steps - 1 - t).
                threshold = pm_steps[n_steps - 1 - t]
                break

    # Thresholding
    m = pm < threshold
    while np.all(m):
        threshold = threshold * 0.81
        m = pm < threshold

    px = px.copy()
    py = py.copy()
    px[m] = 0.0
    py[m] = 0.0

    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold
