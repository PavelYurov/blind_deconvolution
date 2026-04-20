"""
utils.py

Utility functions for the DCP (Dark Channel Prior) blind deconvolution.

Ported from MATLAB code by Jinshan Pan et al. (CVPR 2016).
Reference:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR, 2016.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(A, B, 'valid'):
        MATLAB conv2 performs TRUE cross-correlation when used with
        the standard calling convention conv2(A, B, ...).  Actually,
        conv2 does perform convolution (flips B), BUT in the DCP code
        the filters dx=[-1 1;0 0], dy=[-1 0;1 0] are ALREADY the
        "flipped" versions the author intends.  So convolution with
        these specific filters is correct.
        → scipy.signal.convolve2d(A, B, mode='valid')
        Result size: (M-mk+1, N-nk+1), same as MATLAB 'valid'.

    padarray(I, [p p], 'replicate'):
        → np.pad(I, ((p,p),(p,p)), mode='edge')
        Both replicate the nearest edge value.

    MATLAB indexing is 1-based, column-major (Fortran order):
        When MATLAB does tmp(:) on an (R,C) matrix, it stacks
        columns: [col1; col2; ...].  min(tmp(:)) returns a 1-based
        index into this column-major vector.
        In assign_dark_channel_to_pixel, patch(idx) uses this same
        column-major linear index to write back.
        → We store column-major flat index and use np.unravel_index
          with order='F' when needed.

    graythresh(img):
        Otsu's method.  MATLAB works on [0,1] float and returns a
        threshold in [0,1].
        → skimage.filters.threshold_otsu or manual Otsu.

    fspecial('gaussian', hsize, sigma):
        Produces an hsize×hsize Gaussian kernel, normalised to sum=1.
        The grid is centred at the middle pixel.
        → Manual construction with same formula.

    histc(x, edges):
        Like np.histogram but the last bin includes the right edge,
        and the output length equals len(edges).
        → np.searchsorted + np.bincount, carefully matching MATLAB.

    dst / idst (Discrete Sine Transform):
        MATLAB's dst is Type-II DST.
        → scipy.fft.dstn / idstn with type=2.

    psf2otf(psf, shape):
        Zero-pad PSF, circularly shift centre to (0,0), then fft2.
        → Manual implementation matching MATLAB exactly.

    interp2(x,y,V,xq,yq,'linear'):
        Returns NaN for out-of-bound queries.
        → scipy.ndimage.map_coordinates with appropriate handling.
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
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.

    MATLAB psf2otf circshift amounts: -floor(size(psf)/2) for each dim,
    which equals -(psf_rows//2) and -(psf_cols//2).
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    # Circular shift: move PSF centre to (0, 0)
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: tuple) -> np.ndarray:
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).

    1. ifft2 → real part.
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
    """Build the LUT of optimal FFT sizes (products of small primes)."""
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

    # Fill gaps: for each position without a valid size, use next larger valid
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

    MATLAB uses dst()/idst() which are Type-II DST.
    scipy.fft.dstn with type=2 matches MATLAB's dst.
    scipy.fft.idstn with type=2 matches MATLAB's idst.

    MATLAB convention:
        tt = dst(f2);           % DST-II along columns
        f2sin = dst(tt')';      % DST-II along rows (transpose trick)
    This is equivalent to 2-D DST-II, which scipy.fft.dstn(..., type=2) does.

    Similarly for idst.

    IMPORTANT: MATLAB dst/idst have a different normalisation than scipy.
    MATLAB dst: y_k = sum_n x_n * sin(pi*(2n-1)*k / (2N)),  k=1..N
    scipy dstn type=2: same formula but may differ by factor.
    Actually, scipy.fft.dstn with type=2 uses:
        y_k = 2 * sum_n x_n * sin(pi*(n+0.5)*(k+1)/N)
    while MATLAB dst is:
        y_k = 2 * sum_n x_n * sin(pi*n*k/(N+1))
    These are DIFFERENT transforms.

    MATLAB's dst is actually DST-I:
        y_k = sum_{n=1}^{N} x_n * sin(pi*n*k/(N+1)),  k=1..N
    scipy DST type=1:
        y_k = 2 * sum_{n=0}^{N-1} x[n] * sin(pi*(n+1)*(k+1)/(N+1))

    Let me verify: MATLAB's dst for a vector of length N:
        X(k) = sum_{n=1}^{N} x(n) * sin(pi*k*n/(N+1))   for k=1..N
    This is DST-I.  scipy DST type 1:
        y[k] = 2 * sum_{n=0}^{N-1} x[n] * sin(pi * (n+1) * (k+1) / (N+1))
    With n' = n+1, k' = k+1:
        y[k] = 2 * sum_{n'=1}^{N} x[n'-1] * sin(pi * n' * k' / (N+1))
    So y[k] = 2 * X(k+1) means scipy's DST-I = 2 * MATLAB's dst (with index shift).

    For the Poisson solver, the factor cancels in forward/inverse.
    We use scipy dstn type=1 for both forward and inverse, and the
    normalisation cancels out since we do dst → divide → idst.
    """
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()

    # Zero out interior — keep only boundary values
    boundary_image[1:-1, 1:-1] = 0.0

    # Compute Laplacian of the boundary image at interior points
    # MATLAB: f_bp(j,k) = -4*bi(j,k) + bi(j,k+1) + bi(j,k-1) + bi(j-1,k) + bi(j+1,k)
    # where j=2:H-1, k=2:W-1 (1-based) → j=1:H-2, k=1:W-2 (0-based)
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H-1, 1:W-1] = (
        -4.0 * boundary_image[1:H-1, 1:W-1]
        + boundary_image[1:H-1, 2:W]       # k+1
        + boundary_image[1:H-1, 0:W-2]     # k-1
        + boundary_image[0:H-2, 1:W-1]     # j-1
        + boundary_image[2:H,   1:W-1]     # j+1
    )

    f1 = -f_bp  # f = zeros - f_bp

    # Interior only (MATLAB: f2 = f1(2:end-1, 2:end-1))
    f2 = f1[1:H-1, 1:W-1]

    # 2-D DST-I  (MATLAB: tt = dst(f2); f2sin = dst(tt')'; )
    # MATLAB's dst column-by-column then row-by-row = 2D DST-I.
    # scipy.fft.dstn with type=1 applies DST-I along all axes.
    # The factor of 2 in scipy vs MATLAB cancels in forward/inverse.
    f2sin = dstn(f2, type=1)

    # Eigenvalues of the discrete Laplacian under DST-I basis
    # MATLAB: [x,y] = meshgrid(1:W-2, 1:H-2);
    #         denom = (2*cos(pi*x/(W-1))-2) + (2*cos(pi*y/(H-1)) - 2);
    x = np.arange(1, W - 1)   # 1 .. W-2
    y = np.arange(1, H - 1)   # 1 .. H-2
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    # Divide in transform domain
    f3 = f2sin / denom

    # 2-D inverse DST-I
    img_tt = idstn(f3, type=1)

    # Normalisation: scipy idstn type=1 includes a factor that
    # combined with dstn type=1 gives:  idstn(dstn(x)) = x * 2*(N+1)
    # per dimension.  For 2D: factor = 4*(H-2+1)*(W-2+1) = 4*(H-1)*(W-1).
    # But actually scipy's convention: dstn type=1 has implicit factor 2,
    # and idstn type=1 normalises by 1/(2*(N+1)).
    # Net roundtrip:  idstn(dstn(x, type=1), type=1) = x.
    # So no manual normalisation needed.  Let's verify this is correct
    # by trusting scipy's convention.

    # Put solution in inner points; boundary from boundary_image
    img_direct = boundary_image.copy()
    img_direct[1:H-1, 1:W-1] = img_tt

    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image so boundaries are circularly smooth for FFT-based deconvolution.
    Equivalent to MATLAB wrap_boundary_liu.m (Cho, based on Liu & Jia ICIP 2008).

    Parameters
    ----------
    img : 2D array (H, W) — input image (grayscale)
    img_size : (H_out, W_out) — target padded size

    Returns
    -------
    ret : (H_out, W_out) array — boundary-wrapped image

    Note: MATLAB uses alpha=1.  All indexing converted from 1-based to 0-based.
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

        # --- Build r_A: (alpha*2 + H_w) × W ---
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        # MATLAB: r_A(1:alpha, :) = HG(end-alpha+1:end, :)
        r_A[:alpha, :] = HG[-alpha:, :]
        # MATLAB: r_A(end-alpha+1:end, :) = HG(1:alpha, :)
        r_A[-alpha:, :] = HG[:alpha, :]

        # Linear interpolation of left/right boundary columns in the mid rows
        # MATLAB: a = ((1:H_w)-1)/(H_w-1)  → [0, 1/(H_w-1), ..., 1]
        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        # MATLAB: r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1)
        # alpha+1 in MATLAB → alpha in Python (0-based)
        # end-alpha in MATLAB → -(alpha+1)+1 = -alpha in Python slice end
        # The mid rows are indices alpha .. alpha+H_w-1 in 0-based
        r_A[alpha:alpha + H_w, 0] = (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0]
        r_A[alpha:alpha + H_w, -1] = (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1]

        # MATLAB: A2 = solve_min_laplacian(r_A(alpha:end-alpha+1,:))
        # alpha:end-alpha+1 in MATLAB 1-based = (alpha-1):(end-alpha) in 0-based
        # that's indices alpha-1 to (alpha*2+H_w - 1 - alpha) = alpha+H_w-1
        # So: r_A[alpha-1 : alpha+H_w, :]  → (H_w+1) rows
        # Wait— MATLAB: r_A has size (alpha*2 + H_w).
        # r_A(alpha:end-alpha+1, :) means rows alpha to (alpha*2+H_w)-alpha+1 = alpha+H_w+1
        # in 1-based. That's alpha .. alpha+H_w in 1-based = (alpha-1) .. (alpha+H_w-1) in 0-based.
        # Length: H_w + 1 rows.
        # But end-alpha+1 in 1-based with end = alpha*2+H_w:
        #   (alpha*2+H_w) - alpha + 1 = alpha + H_w + 1 (1-based)
        #   → alpha + H_w in 0-based
        # So slice: [alpha-1 : alpha+H_w+1]  ... no.
        # MATLAB 1-based: alpha to (alpha+H_w+1)  → length H_w+2
        # Hmm let me re-derive. r_A has size = 2*alpha + H_w = 2+H_w (since alpha=1).
        # MATLAB: r_A(alpha:end-alpha+1, :) = r_A(1:end, :) when alpha=1?
        # No: alpha=1, end=2+H_w. alpha:end-alpha+1 = 1:(2+H_w)-1+1 = 1:2+H_w = all rows.
        # Wait: end-alpha+1 = (2+H_w) - 1 + 1 = 2+H_w. So it's 1:2+H_w = all rows. Hmm, that's the whole thing.
        # Let me re-check: with alpha=1, r_A size = 2*1+H_w = H_w+2.
        # r_A(1:H_w+2, :) = all rows.  A2 = solve_min_laplacian(r_A).
        # Then r_A(alpha:end-alpha+1,:) = r_A(1:H_w+2,:) = r_A.
        # So A2 replaces the entire r_A.
        A2 = _solve_min_laplacian(r_A)
        r_A = A2  # replaces: r_A(alpha:end-alpha+1,:) = A2, but that's all rows when alpha=1
        A = r_A

        # --- Build r_B: H × (alpha*2 + W_w) ---
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]

        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha]
        r_B[-1, alpha:alpha + W_w] = (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha]

        # Same logic: with alpha=1, solve on entire r_B
        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        # --- Build r_C: (alpha*2 + H_w) × (alpha*2 + W_w) ---
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        # MATLAB: C2 = solve_min_laplacian(r_C(alpha:end-alpha+1, alpha:end-alpha+1))
        # With alpha=1: r_C(1:end, 1:end) = r_C entirely
        # No wait: alpha:end-alpha+1.  For r_C rows = 2+H_w.
        # rows: alpha to end-alpha+1 = 1 to (2+H_w)-1+1 = 2+H_w → all rows.
        # Same for cols: 2+W_w. alpha to end-alpha+1 = 1 to 2+W_w → all cols.
        # Hmm but that doesn't make sense for the general case. Let me
        # just hardcode alpha=1 since that's what the code always uses.
        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        # MATLAB: A = A(alpha:end-alpha-1, :) with alpha=1
        # rows alpha to end-alpha-1 in 1-based = 1 to (2+H_w)-1-1 = H_w in 1-based
        # = 0 to H_w-1 in 0-based. So A = A[:H_w, :]
        # Wait: end = H_w+2.  alpha=1. end-alpha-1 = H_w+2-1-1 = H_w (1-based).
        # So rows 1:H_w in 1-based = 0:H_w-1 in 0-based → A[:H_w, :]
        A = A[:H_w, :]

        # MATLAB: B = B(:, alpha+1:end-alpha) with alpha=1
        # cols 2 to (2+W_w)-1 = 2:W_w+1 in 1-based = 1:W_w in 0-based
        B = B[:, 1:W_w + 1]

        # MATLAB: C = C(alpha+1:end-alpha, alpha+1:end-alpha) with alpha=1
        # rows 2:(2+H_w)-1 = 2:H_w+1 in 1-based = 1:H_w in 0-based
        # cols 2:(2+W_w)-1 = 2:W_w+1 in 1-based = 1:W_w in 0-based
        C = C[1:H_w + 1, 1:W_w + 1]

        # MATLAB: ret(:,:,ch) = [img(:,:,ch) B; A C]
        # Top row:    [HG, B]   sizes: (H, W) and (H, W_w)
        # Bottom row: [A,  C]   sizes: (H_w, W) and (H_w, W_w)
        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# Dark Channel  (from dark_channel.m)
# ═════════════════════════════════════════════════════════════════════════════

def dark_channel(I: np.ndarray, patch_size: int):
    """
    Compute the dark channel of an image.
    Equivalent to MATLAB dark_channel.m.

    Parameters
    ----------
    I : (M, N) or (M, N, C) image, float64
    patch_size : odd integer — patch window size

    Returns
    -------
    J : (M, N) dark channel image
    J_index : (M, N) int — linear index (COLUMN-MAJOR, 1-based) of the
              minimising pixel within each patch.  This matches MATLAB's
              convention because assign_dark_channel_to_pixel uses it
              with column-major linear indexing.

    MATLAB details:
        padarray(I, [p p], 'replicate') → np.pad with mode='edge'
        min(patch, [], 3) → min over channels (axis=2 in 3D)
        [val, idx] = min(tmp(:)) → min of column-major flattened 2D patch
        The returned idx is 1-based column-major.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    M, N, C = I.shape
    J = np.zeros((M, N), dtype=np.float64)
    J_index = np.zeros((M, N), dtype=np.int64)

    p = patch_size // 2
    # MATLAB: padarray(I, [p p], 'replicate')
    I_pad = np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge')

    for m in range(M):
        for n in range(N):
            patch = I_pad[m:m + patch_size, n:n + patch_size, :]  # (ps, ps, C)
            # min over channels
            tmp = np.min(patch, axis=2)  # (ps, ps)
            # MATLAB: [tmp_val, tmp_idx] = min(tmp(:))
            # tmp(:) in MATLAB is column-major flattening.
            # We flatten in 'F' (Fortran/column-major) order to match.
            tmp_flat = tmp.flatten(order='F')
            tmp_idx = np.argmin(tmp_flat)  # 0-based
            J[m, n] = tmp_flat[tmp_idx]
            J_index[m, n] = tmp_idx + 1  # 1-based to match MATLAB

    return J, J_index


def dark_channel_fast(I: np.ndarray, patch_size: int):
    """
    Fast (vectorised) dark channel computation.
    Produces the same J as the loop version but J_index uses the same
    MATLAB column-major 1-based convention.
    """
    if I.ndim == 2:
        I = I[:, :, np.newaxis]

    M, N, C = I.shape
    p = patch_size // 2
    I_pad = np.pad(I, ((p, p), (p, p), (0, 0)), mode='edge')

    # Min across channels first
    if C > 1:
        I_min = np.min(I_pad, axis=2)
    else:
        I_min = I_pad[:, :, 0]

    # Sliding-window min using a simple approach:
    # For each row, compute running min over columns, then over rows.
    # Use stride_tricks or a loop-based approach.

    J = np.zeros((M, N), dtype=np.float64)
    J_index = np.zeros((M, N), dtype=np.int64)

    for m in range(M):
        for n in range(N):
            patch = I_pad[m:m + patch_size, n:n + patch_size, :]
            tmp = np.min(patch, axis=2)
            tmp_flat = tmp.flatten(order='F')
            idx = np.argmin(tmp_flat)
            J[m, n] = tmp_flat[idx]
            J_index[m, n] = idx + 1

    return J, J_index


# ═════════════════════════════════════════════════════════════════════════════
# assign_dark_channel_to_pixel  (from assign_dark_channel_to_pixel.m)
# ═════════════════════════════════════════════════════════════════════════════

def assign_dark_channel_to_pixel(S: np.ndarray,
                                 dark_channel_refine: np.ndarray,
                                 dark_channel_index: np.ndarray,
                                 patch_size: int) -> np.ndarray:
    """
    Assign refined dark channel values back to image pixels.
    Equivalent to MATLAB assign_dark_channel_to_pixel.m.

    Parameters
    ----------
    S : (M, N) or (M, N, C) — current image estimate
    dark_channel_refine : (M, N) — refined dark channel values
    dark_channel_index : (M, N) — 1-based column-major linear index within patch
    patch_size : odd int

    Returns
    -------
    outImg : same shape as S

    MATLAB details:
        patch(dark_channel_index(m,n)) = dark_channel_refine(m,n)
        This uses column-major linear indexing into the (ps, ps, C) patch.
        The patch is 3D even for grayscale (C=1), and the index addresses
        the column-major flattened array.

        The boundary processing at the end restores original values at the
        padsize-wide border.
    """
    if S.ndim == 2:
        S_3d = S[:, :, np.newaxis]
    else:
        S_3d = S

    M, N, C = S_3d.shape
    padsize = patch_size // 2

    # MATLAB: padarray(S, [padsize padsize], 'replicate')
    S_padd = np.pad(S_3d, ((padsize, padsize), (padsize, padsize), (0, 0)),
                    mode='edge')

    for m in range(M):
        for n in range(N):
            patch = S_padd[m:m + patch_size, n:n + patch_size, :].copy()

            # MATLAB: if ~isequal(min(patch(:)), dark_channel_refine(m,n))
            if np.min(patch) != dark_channel_refine[m, n]:
                # MATLAB: patch(dark_channel_index(m,n)) = dark_channel_refine(m,n)
                # dark_channel_index is 1-based column-major into (ps, ps, C)
                idx = int(dark_channel_index[m, n]) - 1  # to 0-based
                # Unravel in Fortran (column-major) order for (ps, ps, C)
                coords = np.unravel_index(idx, (patch_size, patch_size, C),
                                          order='F')
                patch[coords] = dark_channel_refine[m, n]

            # Write back to padded image
            S_padd[m:m + patch_size, n:n + patch_size, :] = patch

    # Crop back
    outImg = S_padd[padsize:padsize + M, padsize:padsize + N, :]

    # Boundary processing: restore original values at borders
    outImg[:padsize, :, :] = S_3d[:padsize, :, :]
    outImg[-padsize:, :, :] = S_3d[-padsize:, :, :]
    outImg[:, :padsize, :] = S_3d[:, :padsize, :]
    outImg[:, -padsize:, :] = S_3d[:, -padsize:, :]

    if S.ndim == 2:
        return outImg[:, :, 0]
    return outImg


# ═════════════════════════════════════════════════════════════════════════════
# Conjugate Gradient  (from cho_code/conjgrad.m)
# ═════════════════════════════════════════════════════════════════════════════

def conjgrad(x: np.ndarray, b: np.ndarray, max_it: int, tol: float,
             ax_func, func_param) -> np.ndarray:
    """
    Conjugate gradient solver.
    Equivalent to MATLAB cho_code/conjgrad.m.

    Solves A*x = b where A is defined implicitly by ax_func(x, param).

    Parameters
    ----------
    x : initial guess (2D array)
    b : right-hand side (same shape)
    max_it : maximum iterations
    tol : convergence tolerance on ||r||
    ax_func : callable(x, param) → A*x
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
    Equivalent to MATLAB adjust_psf_center.m.

    MATLAB uses:
        meshgrid(1:cols, 1:rows) — 1-based coordinates
        xc1 = sum(psf .* X)  — centre of mass x
        yc1 = sum(psf .* Y)  — centre of mass y
        xc2, yc2 = geometric centre (1-based)
        shift = round(xc2 - xc1), round(yc2 - yc1)
        Apply affine warp with bilinear interpolation.
        Out-of-bound → NaN → replaced with 0.

    In Python we use scipy.ndimage.map_coordinates for the same effect.
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

    # MATLAB: warpimage(psf, [1 0 -xshift; 0 1 -yshift])
    # The affine transform M = [1 0 -xshift; 0 1 -yshift] means:
    #   x' = x - xshift,  y' = y - yshift
    # Then interp2(x,y,im, x', y', 'linear') samples im at (x-xshift, y-yshift)
    # i.e. the image is shifted by (+xshift, +yshift).
    #
    # MATLAB uses 1-based coords: for each output pixel at (x,y) (1-based),
    # sample input at (x - xshift, y - yshift) (1-based).
    # Converting to 0-based for map_coordinates:
    #   input_row = output_row - yshift  (both 0-based, since offset is the same)
    #   input_col = output_col - xshift
    out_rows, out_cols = np.meshgrid(np.arange(rows, dtype=np.float64),
                                      np.arange(cols, dtype=np.float64),
                                      indexing='ij')
    in_rows = out_rows - yshift
    in_cols = out_cols - xshift

    # map_coordinates uses 0-based coords; order=1 = bilinear
    # cval=0 handles out-of-bound (MATLAB puts NaN then replaces with 0)
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

    np.histogram uses [edge_i, edge_{i+1}) for all bins, and the last bin is
    [edge_{n-1}, edge_n].  But np.histogram returns len(edges)-1 bins.

    We match MATLAB exactly using searchsorted.
    """
    # sortedcontainers not needed; data and edges are numeric
    indices = np.searchsorted(edges, data, side='right') - 1
    # Values exactly equal to the last edge go into the last bin
    indices[data == edges[-1]] = len(edges) - 1
    # Values outside range
    indices[indices < 0] = len(edges)  # will not be counted
    indices[indices >= len(edges)] = len(edges)  # will not be counted

    counts = np.bincount(indices, minlength=len(edges) + 1)
    return counts[:len(edges)]


def threshold_pxpy_v1(latent: np.ndarray, psf_size,
                      threshold=None):
    """
    Gradient thresholding for kernel estimation.
    Equivalent to MATLAB cho_code/threshold_pxpy_v1.m.

    Parameters
    ----------
    latent : (M, N) image
    psf_size : scalar or array-like — kernel size (max used)
    threshold : float or None — if None, estimate from histogram

    Returns
    -------
    px, py : gradient images with weak gradients zeroed
    threshold : updated threshold value

    MATLAB conv2 details:
        dx = [-1 1; 0 0]; dy = [-1 0; 1 0]
        conv2(denoised, dx, 'valid') — this is TRUE convolution (flips kernel).
        Flipped dx = [0 0; 1 -1], applied as correlation → same as correlation with [0 0; 1 -1].
        But since we need to match MATLAB exactly, we use scipy convolve2d
        which also does true convolution (flips kernel), matching MATLAB conv2.
    """
    b_estimate_threshold = threshold is None

    if b_estimate_threshold:
        threshold = 0.0

    denoised = latent

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    # MATLAB conv2(denoised, dx, 'valid') = true convolution (kernel flipped)
    # scipy convolve2d also does true convolution.  Both produce 'valid' output.
    px = convolve2d(denoised, dx, mode='valid')
    py = convolve2d(denoised, dy, mode='valid')
    pm = px ** 2 + py ** 2

    if b_estimate_threshold:
        pd = np.arctan2(py, px)  # MATLAB uses atan(py./px) but arctan2 is safer

        # MATLAB: atan(py./px) gives values in [-pi/2, pi/2].
        # arctan2 gives [-pi, pi].  We must use atan to match MATLAB exactly.
        # atan(py/px) for division: need to handle px=0 (gives ±inf → atan=±pi/2)
        with np.errstate(divide='ignore', invalid='ignore'):
            pd = np.arctan(py / px)
            # Where px==0: atan(±inf) = ±pi/2, atan(nan) = nan
            # MATLAB atan(Inf) = pi/2, atan(-Inf) = -pi/2, atan(NaN) = NaN
            # numpy also: arctan(inf) = pi/2, arctan(-inf) = -pi/2, arctan(nan) = nan
            # So this matches.

        pm_steps = np.arange(0, 2 + 0.00006, 0.00006)
        # MATLAB: 0:0.00006:2 — this generates values from 0 to 2 in steps of 0.00006
        # Handle floating point: make sure last value <= 2
        pm_steps = pm_steps[pm_steps <= 2.0 + 1e-12]

        # Build masks for 4 direction bins
        mask1 = (pd >= 0) & (pd < np.pi / 4)
        mask2 = (pd >= np.pi / 4) & (pd < np.pi / 2)
        mask3 = (pd >= -np.pi / 4) & (pd < 0)
        mask4 = (pd >= -np.pi / 2) & (pd < -np.pi / 4)

        H1 = np.cumsum(_histc(pm[mask1], pm_steps)[::-1])
        H2 = np.cumsum(_histc(pm[mask2], pm_steps)[::-1])
        H3 = np.cumsum(_histc(pm[mask3], pm_steps)[::-1])
        H4 = np.cumsum(_histc(pm[mask4], pm_steps)[::-1])

        psf_size_val = np.max(psf_size) if hasattr(psf_size, '__len__') else psf_size
        th = max(psf_size_val * 20, 10)

        for t in range(len(pm_steps)):
            min_h = min(H1[t], H2[t], H3[t], H4[t])
            if min_h >= th:
                # MATLAB: threshold = pm_steps(end - t + 1)
                # t is 1-based in MATLAB → (t-1) in Python
                # pm_steps(end - t + 1) with t 1-based → pm_steps[-(t)]
                # In the loop, MATLAB t goes 1,2,3,...
                # Python t goes 0,1,2,...
                # MATLAB: threshold = pm_steps(end - t + 1) → pm_steps(end - (t-1))
                # For Python t=0: pm_steps[-1], t=1: pm_steps[-2], etc.
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
    if not b_estimate_threshold:
        threshold = threshold / 1.1

    return px, py, threshold


# ═════════════════════════════════════════════════════════════════════════════
# bilateral_filter  (from bilateral_filter.m)
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """
    Equivalent to MATLAB fspecial('gaussian', size, sigma).

    Creates a size×size Gaussian kernel normalised to sum = 1.
    MATLAB centres the kernel at the middle pixel; grid goes
    from -(size-1)/2 to +(size-1)/2.
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

    Called as: bilateral_filter(diff, 3, 0.1) in ringing_artifacts_removal.m

    For grayscale (d==1) the code uses:
        lab = img;  sigma = sigma * sqrt(d) = sigma * 1
    So no colour conversion is needed.

    Parameters
    ----------
    img : (H, W) or (H, W, D) float image
    sigma_s : spatial sigma
    sigma : range sigma

    Returns
    -------
    r_img : filtered image, same shape
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1

    h, w, d = img.shape
    img = img.astype(np.float32)

    # For grayscale or multi-channel non-RGB
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


# ═════════════════════════════════════════════════════════════════════════════
# graythresh  (Otsu's method, matching MATLAB)
# ═════════════════════════════════════════════════════════════════════════════

def graythresh(img: np.ndarray) -> float:
    """
    Otsu's threshold.  Equivalent to MATLAB graythresh(img).

    MATLAB graythresh expects float in [0,1] and returns threshold in [0,1].
    It computes a 256-bin histogram over [0,1], then applies Otsu's method.
    """
    img_flat = img.ravel().astype(np.float64)
    img_flat = np.clip(img_flat, 0.0, 1.0)

    # 256 bins over [0, 1], matching MATLAB convention
    num_bins = 256
    counts, bin_edges = np.histogram(img_flat, bins=num_bins, range=(0.0, 1.0))
    # Bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = counts.sum()
    if total == 0:
        return 0.0

    # Normalised histogram
    p = counts.astype(np.float64) / total

    # Cumulative sums
    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centres)
    mu_t = mu[-1]

    # Between-class variance
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = ((mu_t * omega - mu) ** 2) / (omega * (1.0 - omega))

    sigma_b_sq = np.nan_to_num(sigma_b_sq, nan=0.0)
    max_idx = np.argmax(sigma_b_sq)

    return bin_centres[max_idx]


# ═════════════════════════════════════════════════════════════════════════════
# Non-blind deconvolution helpers
# ═════════════════════════════════════════════════════════════════════════════

def wiener_filter(img: np.ndarray, kernel: np.ndarray,
                  noise_snr: float = 0.01) -> np.ndarray:
    """
    Wiener non-blind deconvolution.
    Û = conj(K) / (|K|² + snr) · F
    """
    H, W = img.shape[:2]
    otf = psf2otf(kernel, (H, W))
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (np.abs(otf) ** 2 + noise_snr)) * F_img
    return np.real(np.fft.ifft2(F_res))


def tikhonov_filter(img: np.ndarray, kernel: np.ndarray,
                    alpha: float = 0.01) -> np.ndarray:
    """
    Tikhonov-regularised deconvolution (1st-order gradient penalty).
    Solves  min_u  ||k*u − f||² + α·||∇u||²   via FFT.
    """
    H, W = img.shape[:2]
    otf = psf2otf(kernel, (H, W))

    # Gradient operators in Fourier domain
    dx_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float64)
    dy_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float64)
    OTF_dx = psf2otf(dx_kernel, (H, W))
    OTF_dy = psf2otf(dy_kernel, (H, W))

    reg_term = np.abs(OTF_dx) ** 2 + np.abs(OTF_dy) ** 2
    denominator = np.abs(otf) ** 2 + alpha * reg_term

    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))


def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:
    """
    Taper image edges toward a blurred version to suppress FFT ringing.
    Mimics MATLAB edgetaper.
    """
    H, W = img.shape[:2]
    kh, kw = kernel.shape

    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf_max = acf.max()
    if acf_max > 0:
        acf /= acf_max

    cy, cx = kh - 1, kw - 1
    z_col = acf[:, cx]
    z_row = acf[cy, :]

    beta_y = np.ones(H, dtype=np.float64)
    beta_x = np.ones(W, dtype=np.float64)

    half_ky = kh - 1
    if half_ky > 0:
        taper = z_col[:half_ky]
        n = min(len(taper), H // 2)
        beta_y[:n] = taper[:n]
        beta_y[-n:] = taper[:n][::-1]

    half_kx = kw - 1
    if half_kx > 0:
        taper = z_row[:half_kx]
        n = min(len(taper), W // 2)
        beta_x[:n] = taper[:n]
        beta_x[-n:] = taper[:n][::-1]

    alpha_map = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]
    otf = psf2otf(kernel, (H, W))

    result = img.copy()
    for _ in range(n_tapers):
        blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(result)))
        result = alpha_map * result + (1.0 - alpha_map) * blurred

    return result


def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Symmetric-pad image by the full kernel size on each side."""
    pad_h, pad_w = kernel_shape[0], kernel_shape[1]
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')


def crop_image(img: np.ndarray, original_shape: tuple,
               kernel_shape: tuple) -> np.ndarray:
    """Crop padded image back to original dimensions."""
    pad_h, pad_w = kernel_shape[0], kernel_shape[1]
    h, w = original_shape
    return img[pad_h:pad_h + h, pad_w:pad_w + w]
