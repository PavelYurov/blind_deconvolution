"""
utils.py

Utility functions for LMGP (Local Maximum Gradient Prior) blind deconvolution.

Ported from MATLAB code by Chen, Liang et al.
Reference:
    L. Chen, F. Fang, T. Wang, G. Zhang:
    "Blind Image Deblurring With Local Maximum Gradient Prior",
    CVPR, 2019.

Original MATLAB code based on Jinshan Pan's DCP framework (CVPR 2016)
and Cho & Lee (SIGGRAPH 2009).

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    Column-major vs row-major:
        MATLAB flattens arrays in COLUMN-MAJOR (Fortran) order.
        The sparse-matrix-based LMG operator (gen_partialmat, Abs_matrix,
        Max_matrix, LMG) relies on column-major indexing throughout.
        -> All flatten(order='F') / reshape(..., order='F') in LMG code.

    Indexing:
        MATLAB is 1-based, Python is 0-based.

    max(patch(:)):
        MATLAB flattens in COLUMN-MAJOR order.
        -> Use patch.flatten(order='F') + np.argmax to match MATLAB
        tie-breaking (first-in-column-major wins).

    conv2(A, B, 'valid'):
        Both MATLAB conv2 and scipy.signal.convolve2d perform TRUE
        convolution (kernel flipped).  Result size: (M-mk+1, N-nk+1).

    diff(S, 1, 2):
        MATLAB dim 2 = Python axis=1.
        MATLAB diff(S,1,1) -> np.diff(S, n=1, axis=0).

    S(:,1,:) - S(:,end,:):
        -> S[:, 0:1, :] - S[:, -1:, :] (slicing preserves dimensions).

    fft2/ifft2/conj:
        -> np.fft.fft2 / np.fft.ifft2 / np.conj  (identical semantics).

    dst / idst (Discrete Sine Transform):
        MATLAB's dst/idst = DST-I.
        scipy.fft.dstn(type=1) computes DST-I with a factor of 2 per
        dimension vs MATLAB.  This factor cancels in the roundtrip
        dstn -> divide_by_eigenvalues -> idstn, so the result is
        identical.

    psf2otf(psf, shape):
        Zero-pad PSF, circularly shift centre to (0,0), then fft2.
        circshift amount: -floor(size(psf)/2) per dim.

    interp2(X,Y,V,Xq,Yq,'linear'):
        Returns NaN for out-of-bound -> replaced with 0.
        -> scipy.ndimage.map_coordinates(mode='constant', cval=0.0).

    padarray(I, [p p], 'replicate'):
        -> np.pad(I, ((p,p),(p,p)), mode='edge').

    bwconncomp(k, 8):
        -> scipy.ndimage.label(k, structure=np.ones((3,3))).

    imresize(k, ret):
        MATLAB default = bicubic.
        -> scipy.ndimage.zoom(k, ret, order=3).

    sparse(row, col, val, M, N):
        MATLAB sparse uses 1-based indices.
        -> scipy.sparse.csr_matrix uses 0-based indices.

Contains:
    ── Shared utilities (from cho_code/) ──────────────────────────────
    psf2otf / otf2psf         — PSF <-> OTF conversions
    opt_fft_size               — optimal FFT data length
    wrap_boundary_liu          — circular boundary padding (Liu & Jia)
    conjgrad                   — conjugate gradient solver
    adjust_psf_center          — PSF centre-of-mass alignment
    threshold_pxpy_v1          — adaptive gradient thresholding
    bilateral_filter           — bilateral filter

    ── LMG-specific (from chen code/) ─────────────────────────────────
    gen_partialmat             — sparse gradient operator matrices
    Abs_matrix                 — diagonal sign matrix
    Max_matrix                 — local-maximum selection matrix
    LMG                        — full Local Maximum Gradient computation
"""

import numpy as np
from scipy import sparse
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
from scipy.fft import dstn, idstn


# ═════════════════════════════════════════════════════════════════════════════
# PSF <-> OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.

    MATLAB psf2otf circshift amounts: -floor(size(psf)/2) for each dim.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # Circular shift: move PSF centre to (0,0)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).

    1. ifft2 -> real part.
    2. Circular shift by +floor(psf_size/2) for each dim.
    3. Crop to psf_size.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# opt_fft_size  (from cho_code/opt_fft_size.m)
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None  # module-level cache (like MATLAB persistent)


def _build_opt_fft_lut(lut_size: int = 4096) -> np.ndarray:
    """Build LUT of optimal FFT sizes (products of small primes 2,3,5,7
    with optional single factors of 11 or 13)."""
    lut = np.zeros(lut_size + 1, dtype=np.int64)  # 1-indexed internally

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

    # Fill gaps: for each position without a valid size, use next larger
    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = i
        else:
            lut[i] = nn
    return lut


def opt_fft_size(n) -> np.ndarray:
    """
    Compute optimal FFT data length(s).
    Equivalent to MATLAB opt_fft_size.m.

    Parameters
    ----------
    n : int or array-like of ints

    Returns
    -------
    m : ndarray of optimal sizes (same shape as input)
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
    Solve Laplace equation with Dirichlet boundary conditions via DST.
    Equivalent to the nested solve_min_laplacian in wrap_boundary_liu.m.

    MATLAB uses dst()/idst() which are DST-I / IDST-I.
    scipy.fft.dstn(type=1) computes DST-I with a factor of 2 per
    dimension compared to MATLAB.  This factor cancels in the roundtrip
    dstn -> divide -> idstn, so the result is identical.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    # Zero out interior — keep only boundary values
    boundary_image[1:-1, 1:-1] = 0.0

    # Compute Laplacian of the boundary image at interior points
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]        # k+1
        + boundary_image[1:H - 1, 0:W - 2]    # k-1
        + boundary_image[0:H - 2, 1:W - 1]    # j-1
        + boundary_image[2:H,     1:W - 1]    # j+1
    )

    f1 = -f_bp

    # Interior only
    f2 = f1[1:H - 1, 1:W - 1]

    # 2-D DST-I
    f2sin = dstn(f2, type=1)

    # Eigenvalues of the discrete Laplacian under DST-I basis
    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom

    # 2-D inverse DST-I
    img_tt = idstn(f3, type=1)

    img_direct = boundary_image.copy()
    img_direct[1:H - 1, 1:W - 1] = img_tt

    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image so boundaries are circularly smooth for FFT-based deconvolution.
    Equivalent to MATLAB wrap_boundary_liu.m (Cho, based on Liu & Jia ICIP 2008).

    Parameters
    ----------
    img      : (H, W) or (H, W, Ch) input image
    img_size : (H_out, W_out) target padded size

    Returns
    -------
    ret : (H_out, W_out[, Ch]) boundary-wrapped image
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = int(img_size[0]), int(img_size[1])
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        # --- Build r_A: (2*alpha + H_w) x W ---
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]

        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (
            (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        )
        r_A[alpha:alpha + H_w, -1] = (
            (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]
        )

        A2 = _solve_min_laplacian(r_A)
        r_A = A2
        A = r_A

        # --- Build r_B: H x (2*alpha + W_w) ---
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (
            (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        )
        r_B[-1, alpha:alpha + W_w] = (
            (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]
        )

        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        # --- Build r_C: (2*alpha + H_w) x (2*alpha + W_w) ---
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        # MATLAB: A = A(alpha:end-alpha-1, :)
        A = A[:H_w, :]
        # MATLAB: B = B(:, alpha+1:end-alpha)
        B = B[:, 1:W_w + 1]
        # MATLAB: C = C(alpha+1:end-alpha, alpha+1:end-alpha)
        C = C[1:H_w + 1, 1:W_w + 1]

        # MATLAB: ret(:,:,ch) = [img(:,:,ch) B; A C]
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
    Conjugate gradient solver.
    Equivalent to MATLAB cho_code/conjgrad.m (Sunghyun Cho).

    Solves A*x = b where A is defined implicitly by ax_func(x, param).

    Parameters
    ----------
    x          : initial guess (2D array)
    b          : right-hand side (same shape)
    max_it     : maximum iterations
    tol        : convergence tolerance on ||r||
    ax_func    : callable(x, param) -> A*x
    func_param : parameters passed to ax_func

    Returns
    -------
    x : solution array
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
    Centre the PSF by shifting its centre-of-mass to the geometric centre.
    Equivalent to MATLAB adjust_psf_center.m (Sunghyun Cho).

    MATLAB:
        [X,Y] = meshgrid(1:cols, 1:rows)  — 1-based coords
        xc1 = sum(psf(:) .* X(:));  yc1 = sum(psf(:) .* Y(:))
        xc2 = (cols+1)/2;           yc2 = (rows+1)/2
        shift = round(xc2 - xc1), round(yc2 - yc1)
        warpProjective2(psf, [1 0 -xshift; 0 1 -yshift])
        interp2(..., 'linear'), NaN -> 0
    """
    rows, cols = psf.shape

    # 1-based coordinate grids (matching MATLAB)
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

    # MATLAB warpProjective2: for each output pixel at 1-based (x,y),
    # sample input at (x - xshift, y - yshift).
    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                      np.arange(cols, dtype=np.float64),
                                      indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    # order=1 = bilinear; cval=0 matches MATLAB NaN -> 0
    result = map_coordinates(psf, [in_rows.ravel(), in_cols.ravel()],
                             order=1, mode='constant', cval=0.0)
    return result.reshape(rows, cols)


# ═════════════════════════════════════════════════════════════════════════════
# threshold_pxpy_v1  (from cho_code/threshold_pxpy_v1.m)
# ═════════════════════════════════════════════════════════════════════════════

def _histc(data: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Equivalent to MATLAB histc(data, edges).

    MATLAB histc: bin(k) counts values where edges(k) <= x < edges(k+1),
    except the last bin also includes x == edges(end).
    Output length = len(edges).
    """
    indices = np.searchsorted(edges, data, side='right') - 1
    indices[data == edges[-1]] = len(edges) - 1
    indices[indices < 0] = len(edges)
    indices[indices >= len(edges)] = len(edges)

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size,
                      threshold=None):
    """
    Gradient thresholding for kernel estimation.
    Equivalent to MATLAB cho_code/threshold_pxpy_v1.m.

    Computes image gradients (px, py) and applies an adaptive threshold
    to suppress small gradients. If no threshold is given, estimates one
    by building histograms of gradient magnitudes across four directional
    bins.

    Parameters
    ----------
    latent    : (M, N) image
    psf_size  : scalar or array-like — kernel size (max used)
    threshold : float or None — if None, estimate from histogram

    Returns
    -------
    px, py    : gradient images with weak gradients zeroed
    threshold : updated threshold value

    MATLAB notes:
        dx = [-1 1; 0 0]; dy = [-1 0; 1 0];
        px = conv2(denoised, dx, 'valid');  — true convolution (flips kernel)
        pd = atan(py./px)  — NOT atan2! gives [-pi/2, pi/2]
    """
    b_estimate_threshold = threshold is None

    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    # MATLAB conv2(denoised, dx, 'valid') — true convolution
    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        # MATLAB: pd = atan(py./px) — gives [-pi/2, pi/2], NOT atan2
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        # Reverse cumulative histograms
        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = (np.max(psf_size) if hasattr(psf_size, '__len__')
                        else psf_size)
        th = max(psf_size_val * 20, 10)

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                threshold = pm_steps[len(pm_steps) - 1 - t]
                break

    # Thresholding
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
    """Equivalent to MATLAB fspecial('gaussian', size, sigma)."""
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()


def bilateral_filter(img: np.ndarray, sigma_s: float,
                     sigma: float) -> np.ndarray:
    """
    Bilateral filter.
    Equivalent to MATLAB bilateral_filter.m for grayscale images.

    For grayscale (d==1) the MATLAB code uses:
        lab = img;  sigma = sigma * sqrt(d) = sigma * 1
    so no colour conversion is needed.

    Parameters
    ----------
    img     : (H, W) or (H, W, D) float image
    sigma_s : spatial sigma
    sigma   : range sigma

    Returns
    -------
    r_img : filtered image, same shape
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


# ═════════════════════════════════════════════════════════════════════════════
# gen_partialmat  (from chen code/gen_partialmat.m)
# ═════════════════════════════════════════════════════════════════════════════

def gen_partialmat(im_row: int, im_col: int):
    """
    Generate sparse gradient operator matrices (partial derivatives).
    Equivalent to MATLAB chen code/gen_partialmat.m.

    CRITICAL: Uses COLUMN-MAJOR (Fortran) linear indexing to match MATLAB.
    When using these matrices, image vectors must be flattened/reshaped
    with order='F'.

    MATLAB:
        ind = i + (j-1)*im_row   (1-based column-major)

        py_mat (vertical / row gradient):
            i==1: forward  difference: py(i,j) = I(i+1,j) - I(i,j)
            else: backward difference: py(i,j) = I(i,j) - I(i-1,j)

        px_mat (horizontal / column gradient):
            j==1: forward  difference: px(i,j) = I(i,j+1) - I(i,j)
            else: backward difference: px(i,j) = I(i,j) - I(i,j-1)

    Parameters
    ----------
    im_row : int — number of rows (M)
    im_col : int — number of columns (N)

    Returns
    -------
    px_mat : (M*N, M*N) sparse CSR matrix — horizontal gradient
    py_mat : (M*N, M*N) sparse CSR matrix — vertical gradient
    """
    M, N = im_row, im_col
    n = M * N
    all_inds = np.arange(n, dtype=np.int64)

    # -- py_mat (vertical / row gradient) --
    # In column-major ordering, first-row pixels are at indices 0, M, 2M, ...
    first_row_mask = (all_inds % M) == 0
    first_row = all_inds[first_row_mask]
    not_first_row = all_inds[~first_row_mask]

    # First row: forward difference  (ind, ind) = -1;  (ind, ind+1) = +1
    r_fr = np.repeat(first_row, 2)
    c_fr = np.empty(2 * len(first_row), dtype=np.int64)
    c_fr[0::2] = first_row
    c_fr[1::2] = first_row + 1
    v_fr = np.tile(np.array([-1.0, 1.0]), len(first_row))

    # Other rows: backward difference  (ind, ind-1) = -1;  (ind, ind) = +1
    r_nfr = np.repeat(not_first_row, 2)
    c_nfr = np.empty(2 * len(not_first_row), dtype=np.int64)
    c_nfr[0::2] = not_first_row - 1
    c_nfr[1::2] = not_first_row
    v_nfr = np.tile(np.array([-1.0, 1.0]), len(not_first_row))

    rows_py = np.concatenate([r_fr, r_nfr])
    cols_py = np.concatenate([c_fr, c_nfr])
    vals_py = np.concatenate([v_fr, v_nfr])
    py_mat = sparse.csr_matrix((vals_py, (rows_py, cols_py)), shape=(n, n))

    # -- px_mat (horizontal / column gradient) --
    # In column-major ordering, first-column pixels are at indices 0..M-1
    first_col_mask = all_inds < M
    first_col = all_inds[first_col_mask]
    not_first_col = all_inds[~first_col_mask]

    # First column: forward difference  (ind, ind) = -1;  (ind, ind+M) = +1
    r_fc = np.repeat(first_col, 2)
    c_fc = np.empty(2 * len(first_col), dtype=np.int64)
    c_fc[0::2] = first_col
    c_fc[1::2] = first_col + M
    v_fc = np.tile(np.array([-1.0, 1.0]), len(first_col))

    # Other columns: backward difference  (ind, ind) = +1;  (ind, ind-M) = -1
    r_nfc = np.repeat(not_first_col, 2)
    c_nfc = np.empty(2 * len(not_first_col), dtype=np.int64)
    c_nfc[0::2] = not_first_col
    c_nfc[1::2] = not_first_col - M
    v_nfc = np.tile(np.array([1.0, -1.0]), len(not_first_col))

    rows_px = np.concatenate([r_fc, r_nfc])
    cols_px = np.concatenate([c_fc, c_nfc])
    vals_px = np.concatenate([v_fc, v_nfc])
    px_mat = sparse.csr_matrix((vals_px, (rows_px, cols_px)), shape=(n, n))

    return px_mat, py_mat


# ═════════════════════════════════════════════════════════════════════════════
# Abs_matrix  (from chen code/Abs_matrix.m)
# ═════════════════════════════════════════════════════════════════════════════

def Abs_matrix(I: np.ndarray) -> sparse.dia_matrix:
    """
    Build diagonal sign matrix.
    Equivalent to MATLAB chen code/Abs_matrix.m.

    Computes diag(sign(I)), where sign(0) = 1 (not 0).

    MATLAB:
        abs_I = abs(I) ./ I;          % = sign(I), NaN for zeros
        abs_I(isnan(abs_I)) = 1;      % zeros -> 1
        Abs_mat = sparse(1:MN, 1:MN, abs_I(:), MN, MN);

    CRITICAL: The diagonal values are stored in COLUMN-MAJOR order
    to match MATLAB's abs_I(:).

    Parameters
    ----------
    I : (M, N) 2D array

    Returns
    -------
    Abs_mat : (M*N, M*N) sparse diagonal matrix
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        abs_I = np.abs(I) / I

    # Replace NaN (from 0/0) with 1; match MATLAB abs_I(isnan(abs_I)) = 1
    abs_I = np.where(np.isfinite(abs_I), abs_I, 1.0)

    # Flatten in column-major to match MATLAB abs_I(:)
    diag_vals = abs_I.flatten(order='F')
    n = diag_vals.size
    return sparse.diags(diag_vals, 0, shape=(n, n), format='csr')


# ═════════════════════════════════════════════════════════════════════════════
# Max_matrix  (from chen code/Max_matrix.m)
# ═════════════════════════════════════════════════════════════════════════════

def Max_matrix(I: np.ndarray, patch_size: int) -> sparse.csr_matrix:
    """
    Build sparse matrix that maps each pixel to the position of the local
    maximum total variation in its neighbourhood.
    Equivalent to MATLAB chen code/Max_matrix.m.

    For each pixel (m, n), find the position of the maximum value
    within a patch_size x patch_size window centred at (m, n), and
    build a permutation-like sparse matrix that "picks" that value.

    CRITICAL: Uses COLUMN-MAJOR (Fortran) indexing internally to match
    MATLAB. The argmax uses column-major flattening to replicate
    MATLAB's max(patch(:)) tie-breaking behaviour.

    Parameters
    ----------
    I          : (M, N) 2D array — typically abs(px) + abs(py) (TV map)
    patch_size : int — must be odd

    Returns
    -------
    max_mat : (M*N, M*N) sparse CSR matrix
    """
    M, N = I.shape
    padsize = patch_size // 2
    h_val = (patch_size + 1) // 2  # ceil(patch_size/2), MATLAB 1-based centre

    J_index = np.zeros((M, N), dtype=np.int64)

    for m_0 in range(M):
        m_1 = m_0 + 1  # MATLAB 1-based
        for n_0 in range(N):
            n_1 = n_0 + 1  # MATLAB 1-based

            # MATLAB: patch = I(max(1,m-padsize):min(M,m+padsize), ...)
            r_start_1 = max(1, m_1 - padsize)
            r_end_1 = min(M, m_1 + padsize)
            c_start_1 = max(1, n_1 - padsize)
            c_end_1 = min(N, n_1 + padsize)

            # Convert to 0-based Python slicing
            patch = I[r_start_1 - 1:r_end_1, c_start_1 - 1:c_end_1]
            h1, h2 = patch.shape

            # Find max in column-major flattening (MATLAB convention)
            flat = patch.flatten(order='F')
            tmp_idx = int(np.argmax(flat)) + 1  # 1-based

            # Compute origin offset (accounts for boundary truncation)
            ori_i = h_val - (patch_size - h1)
            ori_j = h_val - (patch_size - h2)

            if ori_i != h_val and m_1 > h_val:
                ori_i = h1 + 1 - ori_i
            if ori_j != h_val and n_1 > h_val:
                ori_j = h2 + 1 - ori_j

            # Convert column-major flat index to 2D (1-based)
            J_need = int(np.ceil(tmp_idx / h1))
            I_need = tmp_idx - (J_need - 1) * h1

            # Global 1-based coordinates of the max position
            i_quote = m_1 + I_need - ori_i
            j_quote = n_1 + J_need - ori_j

            # Store 0-based column-major global index
            J_index[m_0, n_0] = (i_quote - 1) + (j_quote - 1) * M

    n_px = M * N
    sparse_row = np.arange(n_px, dtype=np.int64)
    sparse_col = J_index.flatten(order='F')  # column-major to match MATLAB
    sparse_val = np.ones(n_px, dtype=np.float64)

    return sparse.csr_matrix(
        (sparse_val, (sparse_row, sparse_col)), shape=(n_px, n_px)
    )


# ═════════════════════════════════════════════════════════════════════════════
# LMG  (from chen code/LMG.m)  — Local Maximum Gradient
# ═════════════════════════════════════════════════════════════════════════════

def LMG(img: np.ndarray, patch_size: int):
    """
    Compute Local Maximum Gradient map and operator.
    Equivalent to MATLAB chen code/LMG.m.

    For each pixel p, the LMG prior maps it to the gradient magnitude
    at the position q* within a patch_size x patch_size neighbourhood
    that has the largest total variation:

        LMG(I)_p = |nabla I_{q*}|,  q* = argmax_{q in Omega(p)} |nabla I_q|

    This encodes the prior that sharp images have high local maximum
    gradients (Sec. 3.1, Eq. 3-4 of Chen et al., CVPR 2019).

    CRITICAL: All flatten/reshape use order='F' (column-major) to match
    MATLAB's img(:).

    Parameters
    ----------
    img        : (M, N) 2D grayscale image, float64
    patch_size : int — neighbourhood size (should be odd)

    Returns
    -------
    output_img : (M, N) LMG map
    A          : (M*N, M*N) sparse matrix — the full LMG operator G_S,
                 such that A @ img.flatten('F') = output_img.flatten('F')
    """
    M, N = img.shape
    px_mat, py_mat = gen_partialmat(M, N)
    img_vec = img.flatten(order='F')

    # Compute gradient images
    px = (px_mat @ img_vec).reshape((M, N), order='F')
    py = (py_mat @ img_vec).reshape((M, N), order='F')

    # Diagonal sign matrices
    abs_x_mat = Abs_matrix(px)
    abs_y_mat = Abs_matrix(py)

    # Local maximum TV selection matrix
    tv = np.abs(px) + np.abs(py)
    max_tv_mat = Max_matrix(tv, patch_size)

    # Full LMG operator:  G = M_max * (|sign(px)| * D_x + |sign(py)| * D_y)
    A = max_tv_mat @ (abs_x_mat @ px_mat + abs_y_mat @ py_mat)
    output_vec = A @ img_vec
    output_img = output_vec.reshape((M, N), order='F')

    return output_img, A