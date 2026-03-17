"""
utils.py

Utility functions for PMP (Patch-wise Minimal Pixels) blind deconvolution.

Ported from MATLAB code by Fei Wen et al.
Reference:
    F. Wen, R. Ying, Y. Liu, P. Liu, T.-K. Truong:
    "A Simple Local Minimal Intensity Prior and An Improved Algorithm
    for Blind Image Deblurring", IEEE TCSVT, 2021.

Original MATLAB code based on Jinshan Pan's DCP framework (CVPR 2016).

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    Indexing:
        MATLAB is 1-based, Python is 0-based.

    min(patch(:)):
        MATLAB flattens in COLUMN-MAJOR (Fortran) order.
        -> Use patch.flatten(order='F') + np.unravel_index(..., order='F')
        to match MATLAB exactly when tie-breaking matters.

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
        MATLAB's dst/idst = DST-I / IDST-I.
        scipy.fft.dstn(type=1) also computes DST-I but with a factor of
        2 per dimension vs MATLAB.  This factor cancels in the roundtrip
        dstn -> divide_by_eigenvalues -> idstn, so the final result is
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
"""

import numpy as np
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

    # Zero out interior -- keep only boundary values
    boundary_image[1:-1, 1:-1] = 0.0

    # Compute Laplacian of the boundary image at interior points
    # MATLAB: f_bp(j,k) = -4*bi(j,k) + bi(j,k+1) + bi(j,k-1)
    #                      + bi(j-1,k) + bi(j+1,k)
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * boundary_image[1:H - 1, 1:W - 1]
        + boundary_image[1:H - 1, 2:W]        # k+1
        + boundary_image[1:H - 1, 0:W - 2]    # k-1
        + boundary_image[0:H - 2, 1:W - 1]    # j-1
        + boundary_image[2:H,     1:W - 1]    # j+1
    )

    # f = zeros - f_bp
    f1 = -f_bp

    # Interior only: MATLAB f2 = f1(2:end-1, 2:end-1)
    f2 = f1[1:H - 1, 1:W - 1]

    # 2-D DST-I (MATLAB: tt = dst(f2); f2sin = dst(tt')';)
    f2sin = dstn(f2, type=1)

    # Eigenvalues of the discrete Laplacian under DST-I basis
    # MATLAB: [x,y] = meshgrid(1:W-2, 1:H-2);
    #         denom = (2*cos(pi*x/(W-1))-2) + (2*cos(pi*y/(H-1))-2);
    x = np.arange(1, W - 1)   # 1 .. W-2
    y = np.arange(1, H - 1)   # 1 .. H-2
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    # Divide in transform domain
    f3 = f2sin / denom

    # 2-D inverse DST-I
    img_tt = idstn(f3, type=1)

    # Put solution in inner points; boundary from boundary_image
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
    ret : (H_out, W_out) or (H_out, W_out, Ch) boundary-wrapped image

    Algorithm:
        ret = [Original | B ]    where B = right padding strip
              [   A     | C ]          A = bottom strip, C = corner
        Each strip is filled by solving a minimum-Laplacian problem (Poisson
        equation) with boundary conditions that stitch opposite image edges.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H  # extra rows
    W_w = W_out - W  # extra cols

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]  # (H, W)

        # --- Build r_A: (2*alpha + H_w) x W ---
        # With alpha=1: size = (2 + H_w, W)
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        # MATLAB: r_A(1:alpha,:) = HG(end-alpha+1:end,:)
        r_A[:alpha, :] = HG[-alpha:, :]
        # MATLAB: r_A(end-alpha+1:end,:) = HG(1:alpha,:)
        r_A[-alpha:, :] = HG[:alpha, :]

        # Linear interpolation of boundary columns in middle rows
        # MATLAB: a = ((1:H_w)-1)/(H_w-1)  ->  [0, 1/(H_w-1), ..., 1]
        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        # MATLAB: r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
        # With alpha=1 (0-based): mid rows [alpha .. alpha+H_w-1]
        # r_A[alpha-1, 0] = r_A[0, 0] = HG[-1, 0]
        # r_A[-alpha, 0]  = r_A[-1, 0] = HG[0, 0]
        r_A[alpha:alpha + H_w, 0] = (
            (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        )
        r_A[alpha:alpha + H_w, -1] = (
            (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]
        )

        # MATLAB: A2 = solve_min_laplacian(r_A(alpha:end-alpha+1,:))
        # With alpha=1: this spans ALL rows of r_A
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
        # With alpha=1, A size = (2+H_w):
        #   rows 1..H_w (1-based) = 0..H_w-1 (0-based)
        A = A[:H_w, :]

        # MATLAB: B = B(:, alpha+1:end-alpha)
        # With alpha=1, B cols = (2+W_w):
        #   cols 2..W_w+1 (1-based) = 1..W_w (0-based)
        B = B[:, 1:W_w + 1]

        # MATLAB: C = C(alpha+1:end-alpha, alpha+1:end-alpha)
        C = C[1:H_w + 1, 1:W_w + 1]

        # MATLAB: ret(:,:,ch) = [img(:,:,ch) B; A C]
        # Top:    [HG (H,W), B (H,W_w)]     = (H, W_out)
        # Bottom: [A (H_w,W), C (H_w,W_w)]  = (H_w, W_out)
        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# find_min_pixels  (from find_min_pixels.m)
# — PMP prior: patch-wise minimal pixels
# ═════════════════════════════════════════════════════════════════════════════

def find_min_pixels(I: np.ndarray, patch_size: int):
    """
    Find the minimum pixel in each non-overlapping patch.
    Equivalent to MATLAB find_min_pixels.m.

    This is the core of the PMP (Patch-wise Minimal Pixels) prior:
    the image is divided into non-overlapping patches of size
    patch_size x patch_size, and in each patch the minimum-valued
    pixel is identified.

    Parameters
    ----------
    I          : (M, N) grayscale image, float64
    patch_size : int, side length of each non-overlapping patch

    Returns
    -------
    J    : (M, N) sparse image — zero everywhere except at the positions
           of patch-wise minima, where it holds the minimum value.
    Mask : (M, N) binary mask — 1 at positions of patch-wise minima.

    MATLAB notes:
        [val, idx] = min(patch(:))  — column-major flattening.
        We use order='F' to match MATLAB tie-breaking exactly.

    MATLAB code:
        for m = 1:Mp
            for n = 1:Np
                idx1 = [1,patch_size]+(m-1)*patch_size;
                idx2 = [1,patch_size]+(n-1)*patch_size;
                patch = I(idx1(1):min(idx1(2),M), idx2(1):min(idx2(2),N));
                [val,idx] = min(patch(:));
                cur_patch = zeros(size(patch));
                cur_patch(idx) = val;
                J(...) = cur_patch;
                mask_patch = zeros(...);
                mask_patch(idx) = 1;
                Mask(...) = mask_patch;
    """
    M, N = I.shape
    Mp = int(np.ceil(M / patch_size))
    Np = int(np.ceil(N / patch_size))
    J = np.zeros((M, N), dtype=np.float64)
    Mask = np.zeros((M, N), dtype=np.float64)

    for m in range(Mp):
        for n in range(Np):
            # MATLAB (1-based): idx1 = [1,ps]+(m-1)*ps -> rows m*ps+1 to min((m+1)*ps, M)
            # Python (0-based): rows m*ps to min((m+1)*ps, M)-1
            r_start = m * patch_size
            r_end = min((m + 1) * patch_size, M)
            c_start = n * patch_size
            c_end = min((n + 1) * patch_size, N)

            patch = I[r_start:r_end, c_start:c_end]

            # MATLAB: [val, idx] = min(patch(:))   — column-major flat
            flat = patch.flatten(order='F')
            lin_idx = np.argmin(flat)
            val = flat[lin_idx]

            # Convert column-major linear index back to 2D position
            pr, pc = np.unravel_index(lin_idx, patch.shape, order='F')

            # J: place minimum value at its position
            J[r_start + pr, c_start + pc] = val

            # Mask: 1 at the minimum position
            Mask[r_start + pr, c_start + pc] = 1.0

    return J, Mask


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

    xc1 = np.sum(psf * X)  # centre of mass, x (1-based)
    yc1 = np.sum(psf * Y)  # centre of mass, y (1-based)

    xc2 = (cols + 1) / 2.0
    yc2 = (rows + 1) / 2.0

    xshift = round(xc2 - xc1)
    yshift = round(yc2 - yc1)

    # MATLAB warpProjective2: for each output pixel at 1-based (x,y),
    # sample input at (x - xshift, y - yshift).
    # In 0-based coords the shift amount is the same.
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
    # Values exactly equal to the last edge go into the last bin
    indices[data == edges[-1]] = len(edges) - 1
    # Values outside range: flag them so they are not counted
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
    bins and finding the value where each bin has at least
    max(psf_size)*20 significant pixels.

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
        # MATLAB: pd = atan(py./px)  — gives [-pi/2, pi/2], NOT atan2
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)

        # MATLAB: pm_steps = 0:0.00006:2
        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        # Build masks for 4 direction bins
        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        # Reverse cumulative histograms
        # MATLAB: H1 = cumsum(flipud(histc(pm(mask), pm_steps)))
        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = (np.max(psf_size) if hasattr(psf_size, '__len__')
                        else psf_size)
        th = max(psf_size_val * 20, 10)

        # MATLAB: for t=1:numel(pm_steps)
        #           min_h = min([H1(t)...]); if min_h >= th ...
        #           threshold = pm_steps(end - t + 1)
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

    # Update threshold
    # MATLAB: if b_estimate_threshold -> keep as is; else -> / 1.1
    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold


# ═════════════════════════════════════════════════════════════════════════════
# bilateral_filter  (from bilateral_filter.m)
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Equivalent to MATLAB fspecial('gaussian', size, sigma).

    Creates a size x size Gaussian kernel normalised to sum = 1.
    Grid centred at the middle pixel: -(size-1)/2 to +(size-1)/2.
    """
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

    # For grayscale: lab = img, sigma = sigma * sqrt(d) = sigma * 1
    lab = img.copy()
    sigma = sigma * np.sqrt(d)

    fr = int(np.ceil(sigma_s * 3))

    # MATLAB: padarray(img, [fr fr], 'replicate')
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
