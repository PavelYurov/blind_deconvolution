"""
utils.py

Utility functions for GBBID (Graph-Based Blind Image Deblurring).

Ported from MATLAB code by Yuanchao Bai et al.
Reference:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Also includes utilities from:
    - D. Krishnan, R. Fergus: "Fast Image Deconvolution using
      Hyper-Laplacian Priors", NIPS 2009.
    - Jian-Feng Cai: 2D Tight Wavelet Frame Transform library.

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    imfilter(x, h, 'conv', 'replicate'):
        MATLAB imfilter with 'conv' flag performs TRUE convolution
        (kernel is flipped internally), with replicate boundary padding.
        -> scipy.ndimage.convolve(x, h, mode='nearest')
           scipy.ndimage.convolve also performs true convolution and
           mode='nearest' matches MATLAB's 'replicate'.

    imfilter(x, h, 'circular'):
        MATLAB imfilter DEFAULT = correlation (no flip), circular padding.
        -> scipy.ndimage.correlate(x, h, mode='wrap')

    conv2(A, B, 'valid'/'same'/'full'):
        MATLAB conv2 is TRUE CONVOLUTION (kernel is flipped internally).
        -> scipy.signal.fftconvolve(A, B, mode=...) also true convolution.
        Output sizes:
            'valid': (M-k+1, N-l+1)
            'same' : (M, N) for first argument
            'full' : (M+k-1, N+l-1)

    rot90(k, 2):
        Rotate 180 degrees = flip both dims -> k[::-1, ::-1]

    psf2otf(psf, shape):
        Zero-pad, circshift centre to (0,0), then fft2.

    otf2psf(otf, psf_size):
        ifft2 -> real, circshift by +floor(psf_size/2), crop.

    edgetaper(I, PSF):
        MATLAB built-in; reimplemented below.

    padarray(A, padsize, 'symmetric', 'both'):
        -> np.pad(A, ..., mode='symmetric')
        Both MATLAB and numpy 'symmetric' repeat the edge element.

    Indexing:
        MATLAB is 1-based, Python is 0-based.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from scipy.ndimage import convolve as ndimage_convolve
from scipy.ndimage import correlate as ndimage_correlate
from scipy.interpolate import interp1d


# ═════════════════════════════════════════════════════════════════════════════
# PSF <-> OTF conversions
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf, shape):
    """
    Convert PSF to OTF.  Equivalent to MATLAB psf2otf(psf, shape).

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.
    """
    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf, psf_size):
    """
    Convert OTF back to PSF.  Equivalent to MATLAB otf2psf(otf, psf_size).

    1. ifft2 -> real part.
    2. Circular shift by +floor(psf_size/2) for each dim.
    3. Crop to psf_size.
    """
    full = np.real(ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# ═════════════════════════════════════════════════════════════════════════════
# Padding utilities
# ═════════════════════════════════════════════════════════════════════════════

def G_padding(x, k, factor):
    """
    Padding the input image for graph construction.
    MATLAB: G_padding.m

    Returns (x_padding, padsize) where padsize = (row_pad, col_pad).
    """
    padsize = (k.shape[0] * factor, k.shape[1] * factor)
    x_padding = np.pad(x,
                       ((padsize[0], padsize[0]), (padsize[1], padsize[1])),
                       mode='edge')
    return x_padding, padsize


def Copy_Enlarge_h(I, H_size):
    """
    Symmetric padding image by replicating edge rows/columns.
    MATLAB: Copy_Enlarge_h.m

    Returns (I2, border).
    """
    s_h, s_w = int(H_size[0]), int(H_size[1])
    if s_h % 2 == 0:
        s_h += 1
    if s_w % 2 == 0:
        s_w += 1
    border = (s_h - 1, s_w - 1)
    h, w = I.shape

    # Pad columns: replicate first and last column
    left = np.tile(I[:, 0:1], (1, border[1]))
    right = np.tile(I[:, -1:], (1, border[1]))
    I2 = np.concatenate([left, I, right], axis=1)

    # Pad rows: replicate first and last row
    top = np.tile(I2[0:1, :], (border[0], 1))
    bottom = np.tile(I2[-1:, :], (border[0], 1))
    I2 = np.concatenate([top, I2, bottom], axis=0)

    return I2, border


# ═════════════════════════════════════════════════════════════════════════════
# FFT-based convolution (for large kernels)
# ═════════════════════════════════════════════════════════════════════════════

def fftconv(I, filt, method):
    """
    Convolution with a large kernel accelerated by FFT.
    MATLAB: fftconv.m

    Parameters
    ----------
    I : 2D array
    filt : 2D array -- convolution kernel
    method : str -- 'same' or 'valid'
    """
    k1, k2 = filt.shape

    I_padded, p_size = G_padding(I, filt, 1)
    n, m = I_padded.shape

    if method == 'same':
        tI = np.zeros((n + k1 - 1, m + k2 - 1), dtype=np.float64)
        tI[:n, :m] = I_padded
        I_padded = tI

    bn, bm = I_padded.shape
    fI = fft2(I_padded)
    ff = fft2(filt, s=(bn, bm))
    fI = fI * ff
    cI = np.real(ifft2(fI))

    hk1d = k1 // 2
    hk1u = k1 - hk1d - 1
    hk2d = k2 // 2
    hk2u = k2 - hk2d - 1

    if method == 'same':
        # MATLAB: cI(hk1d+1:end-hk1u, hk2d+1:end-hk2u)
        end0 = -hk1u if hk1u > 0 else None
        end1 = -hk2u if hk2u > 0 else None
        cI = cI[hk1d:end0, hk2d:end1]
        # MATLAB: cI(p_size+1:end-p_size, p_size+1:end-p_size)
        cI = cI[p_size[0]:-p_size[0] if p_size[0] > 0 else None,
                p_size[1]:-p_size[1] if p_size[1] > 0 else None]
    elif method == 'valid':
        # MATLAB: cI(hk1d+hk1u+1:end, hk2d+hk2u+1:end)
        cI = cI[hk1d + hk1u:, hk2d + hk2u:]

    return cI


# ═════════════════════════════════════════════════════════════════════════════
# edgetaper -- replicate MATLAB built-in edgetaper
# ═════════════════════════════════════════════════════════════════════════════

def edgetaper(img, psf):
    """
    Replicate MATLAB's edgetaper(I, PSF).

    Blends the edges of image I with a blurred version using the
    autocorrelation function of the PSF as weighting.

    Uses the full 2D autocorrelation via FFT (matching MATLAB built-in),
    not the separable 1D projection approach.

    Formula: J = I * (1 - acf) + blurred * acf
    where acf = normalized autocorrelation of PSF:
      - acf ≈ 1 near edges/corners  → use blurred (reduces boundary discontinuity)
      - acf ≈ 0 in center           → keep original
    """
    n, m = img.shape

    # 2D autocorrelation of PSF via power spectrum
    otf = psf2otf(psf, (n, m))
    acf = np.real(ifft2(np.abs(otf) ** 2))
    acf = acf / acf.max()

    # Circular convolution: blurred = PSF * img
    blurred = np.real(ifft2(fft2(img) * otf))

    return img * (1.0 - acf) + blurred * acf


# ═════════════════════════════════════════════════════════════════════════════
# Graph weight computation
# ═════════════════════════════════════════════════════════════════════════════

def weight_function_l1(d):
    """
    Compute weights for l1-Graph Laplacian.
    MATLAB: weight_function_l1.m

    w = 1 / max(|d|, epsilon),  epsilon = 0.01.
    """
    epsilon = 0.01
    d_abs = np.abs(d)
    d_abs = np.maximum(d_abs, epsilon)
    return 1.0 / d_abs


def weights_computation(x, sigma, nei_num, wtype):
    """
    Weight computation for graph-based deblurring.
    MATLAB: weights_computation.m

    Parameters
    ----------
    x : 2D array -- current image estimate
    sigma : float or None -- Gaussian sigma (used for type=1)
    nei_num : int -- number of neighbours (must be 4)
    wtype : int -- weight type:
        1 = Gaussian: w = exp(-d^2/sigma^2)
        2 = IRLS/L1:  w = 1/|d|

    Returns
    -------
    W : (h*w, 4) array of weights

    Notes
    -----
    MATLAB uses imfilter(x, d, 'conv', 'replicate') which is true convolution
    with replicate boundary. This matches scipy.ndimage.convolve(x, d, mode='nearest').

    The 4 directions are:
        d1 = [1, -1, 0]   (horizontal left)
        d2 = d1'           (vertical up)
        d3 = [0, -1, 1]   (horizontal right)
        d4 = d3'           (vertical down)
    """
    h, w = x.shape

    if nei_num == 4 and wtype == 1:
        W = np.zeros((h * w, 4), dtype=np.float64)

        d1 = np.array([[1, -1, 0]], dtype=np.float64)
        d2 = d1.T
        d3 = np.array([[0, -1, 1]], dtype=np.float64)
        d4 = d3.T

        W[:, 0] = ndimage_convolve(x, d1, mode='nearest').ravel()
        W[:, 0] = np.exp(-W[:, 0] ** 2 / sigma ** 2)

        W[:, 1] = ndimage_convolve(x, d2, mode='nearest').ravel()
        W[:, 1] = np.exp(-W[:, 1] ** 2 / sigma ** 2)

        W[:, 2] = ndimage_convolve(x, d3, mode='nearest').ravel()
        W[:, 2] = np.exp(-W[:, 2] ** 2 / sigma ** 2)

        W[:, 3] = ndimage_convolve(x, d4, mode='nearest').ravel()
        W[:, 3] = np.exp(-W[:, 3] ** 2 / sigma ** 2)

    elif nei_num == 4 and wtype == 2:
        W = np.zeros((h * w, 4), dtype=np.float64)

        d1 = np.array([[1, -1, 0]], dtype=np.float64)
        d2 = d1.T
        d3 = np.array([[0, -1, 1]], dtype=np.float64)
        d4 = d3.T

        W[:, 0] = weight_function_l1(
            ndimage_convolve(x, d1, mode='nearest').ravel())
        W[:, 1] = weight_function_l1(
            ndimage_convolve(x, d2, mode='nearest').ravel())
        W[:, 2] = weight_function_l1(
            ndimage_convolve(x, d3, mode='nearest').ravel())
        W[:, 3] = weight_function_l1(
            ndimage_convolve(x, d4, mode='nearest').ravel())
    else:
        W = np.zeros(1)

    return W


# ═════════════════════════════════════════════════════════════════════════════
# Informative edge mask
# ═════════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(M, ratio, max_iter):
    """
    Find a threshold such that approximately `ratio` fraction of pixels
    exceed it. Uses binary search.
    MATLAB: adaptive_threshold (nested in informative_edge_mask_adaptive_mine.m)
    """
    n = M.size
    lower_bound = 0.0
    upper_bound = float(M.max())
    threshold = upper_bound / 2.0
    r = 0.0

    for _ in range(max_iter):
        M_t = np.sum(M > threshold)
        r = M_t / n
        if ratio * 0.9 < r < ratio * 1.1:
            break
        elif r <= ratio * 0.9:
            upper_bound = threshold
            threshold = (lower_bound + upper_bound) / 2.0
        else:
            lower_bound = threshold
            threshold = (lower_bound + upper_bound) / 2.0

    M_threshold = np.zeros_like(M)
    M_threshold[M > threshold] = 1.0
    return M_threshold, r


def informative_edge_mask_adaptive_mine(Y_s, t_s, t_r, h):
    """
    Find informative edge and generate mask.
    MATLAB: informative_edge_mask_adaptive_mine.m

    Parameters
    ----------
    Y_s : 2D array -- skeleton image
    t_s : float -- strength threshold ratio (e.g. 0.1)
    t_r : float -- ratio threshold (e.g. 0.3)
    h : int -- local window size (e.g. 5)

    Returns
    -------
    M : binary mask (same size as Y_s)
    """
    # MATLAB: Dx = rot90([0,-1,1], 2) = [1,-1,0]
    Dx = np.array([[1, -1, 0]], dtype=np.float64)
    Dy = Dx.T

    # imfilter(Y_s, Dx, 'conv', 'replicate')  =  true convolution, replicate boundary
    Mx = ndimage_convolve(Y_s, Dx, mode='nearest')
    My = ndimage_convolve(Y_s, Dy, mode='nearest')
    M_mag = np.sqrt(Mx ** 2 + My ** 2)

    # Strength threshold: keep top t_s fraction
    M3, _ = _adaptive_threshold(M_mag, t_s, 100)

    # Coherence ratio
    k_tmp = np.ones((h, h), dtype=np.float64)
    Mx2 = ndimage_convolve(Mx, k_tmp, mode='nearest')
    My2 = ndimage_convolve(My, k_tmp, mode='nearest')
    M4 = np.sqrt(Mx2 ** 2 + My2 ** 2)

    M5 = ndimage_convolve(M_mag, k_tmp, mode='nearest')
    M4 = M4 / (M5 + 0.5)

    M4_bin, _ = _adaptive_threshold(M4, t_r, 100)

    return M3 * M4_bin


# ═════════════════════════════════════════════════════════════════════════════
# Kernel utilities
# ═════════════════════════════════════════════════════════════════════════════

def _shift_kernel(k, hw):
    """
    Shift kernel by (dh, dw) pixels.
    MATLAB: shift_kernel (nested in kernel_centralize.m)
    """
    h, w = k.shape
    dh, dw = int(hw[0]), int(hw[1])

    # Vertical shift
    k_tmp = np.zeros_like(k)
    if dh >= 0:
        if dh < h:
            k_tmp[dh:, :] = k[:h - dh, :]
    else:
        if -dh < h:
            k_tmp[:h + dh, :] = k[-dh:, :]

    # Horizontal shift
    k_s = np.zeros_like(k)
    if dw >= 0:
        if dw < w:
            k_s[:, dw:] = k_tmp[:, :w - dw]
    else:
        if -dw < w:
            k_s[:, :w + dw] = k_tmp[:, -dw:]

    return k_s


def kernel_centralize(k, threshold):
    """
    Centralize restored kernel.
    MATLAB: kernel_centralize.m

    Finds the bounding box of significant kernel elements,
    computes its centre, and shifts the kernel so that this centre
    aligns with the geometric centre of the array.
    """
    h, w = k.shape
    thresh_val = k.max() * threshold

    # Find bounding box
    h_begin = 0
    for i in range(h):
        if k[i, :].sum() > thresh_val:
            h_begin = i
            break

    h_end = h - 1
    for i in range(h - 1, -1, -1):
        if k[i, :].sum() > thresh_val:
            h_end = i
            break

    w_begin = 0
    for i in range(w):
        if k[:, i].sum() > thresh_val:
            w_begin = i
            break

    w_end = w - 1
    for i in range(w - 1, -1, -1):
        if k[:, i].sum() > thresh_val:
            w_end = i
            break

    # Centre of bounding box (0-indexed, same shift as MATLAB 1-indexed)
    h_center = int(np.floor(h_begin + (h_end - h_begin) / 2.0))
    w_center = int(np.floor(w_begin + (w_end - w_begin) / 2.0))

    # Geometric centre of array
    # MATLAB: ceil(h/2) (1-indexed) → (h-1)//2 (0-indexed)
    kh_center = (h - 1) // 2
    kw_center = (w - 1) // 2

    dh = kh_center - h_center
    dw = kw_center - w_center

    k_c = _shift_kernel(k, (dh, dw))
    k_c_sum = k_c.sum()
    if k_c_sum > 0:
        k_c = k_c / k_c_sum
    return k_c


def k_rescale(k):
    """
    Rescale kernel for display (min-max normalization to [0, 1]).
    MATLAB: k_rescale.m
    """
    k_max = k.max()
    k_min = k.min()
    if k_max == k_min:
        return np.zeros_like(k)
    return (k - k_min) / (k_max - k_min)


# ═════════════════════════════════════════════════════════════════════════════
# Conjugate gradient solver
# ═════════════════════════════════════════════════════════════════════════════

def conjgrad(x, b, max_it, tol, Ax_func, func_param):
    """
    Conjugate gradient optimization.
    MATLAB: conjgrad (from kernel_solver_L2.m)

    Solves A*x = b where A is defined implicitly by Ax_func.
    """
    r = b - Ax_func(x, func_param)
    p = r.copy()
    rsold = np.sum(r * r)

    for _ in range(max_it):
        Ap = Ax_func(p, func_param)
        pAp = np.sum(p * Ap)
        if pAp == 0:
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
# 2D Tight Wavelet Frame Transform
# (Ported from 2DTWFT library by Jian-Feng Cai)
# ═════════════════════════════════════════════════════════════════════════════

def GenerateFrameletFilter(frame):
    """
    Generate framelet decomposition (D) and reconstruction (R) filter banks.
    MATLAB: GenerateFrameletFilter.m

    Parameters
    ----------
    frame : int
        0 = Haar Wavelet
        1 = Piecewise Linear Framelet
        3 = Piecewise Cubic Framelet

    Returns
    -------
    D : list — decomposition filters, last element is boundary-condition string
    R : list — reconstruction filters, last element is boundary-condition string
    """
    if frame == 0:
        D = [
            np.array([0, 1, 1], dtype=np.float64) / 2,
            np.array([0, 1, -1], dtype=np.float64) / 2,
            'cc',
        ]
        R = [
            np.array([1, 1, 0], dtype=np.float64) / 2,
            np.array([-1, 1, 0], dtype=np.float64) / 2,
            'cc',
        ]
    elif frame == 1:
        D = [
            np.array([1, 2, 1], dtype=np.float64) / 4,
            np.array([1, 0, -1], dtype=np.float64) / 4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=np.float64) / 4,
            'ccc',
        ]
        R = [
            np.array([1, 2, 1], dtype=np.float64) / 4,
            np.array([-1, 0, 1], dtype=np.float64) / 4 * np.sqrt(2),
            np.array([-1, 2, -1], dtype=np.float64) / 4,
            'ccc',
        ]
    elif frame == 3:
        D = [
            np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16,
            np.array([1, 2, 0, -2, -1], dtype=np.float64) / 8,
            np.array([-1, 0, 2, 0, -1], dtype=np.float64) / 16 * np.sqrt(6),
            np.array([-1, 2, 0, -2, 1], dtype=np.float64) / 8,
            np.array([1, -4, 6, -4, 1], dtype=np.float64) / 16,
            'ccccc',
        ]
        R = [
            np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16,
            np.array([-1, -2, 0, 2, 1], dtype=np.float64) / 8,
            np.array([-1, 0, 2, 0, -1], dtype=np.float64) / 16 * np.sqrt(6),
            np.array([1, -2, 0, 2, -1], dtype=np.float64) / 8,
            np.array([1, -4, 6, -4, 1], dtype=np.float64) / 16,
            'ccccc',
        ]
    else:
        raise ValueError(f"Unsupported frame type: {frame}")

    return D, R


def ConvSymAsym2D(A, M, b, L):
    """
    1D convolution/correlation with boundary conditions, applied along rows.
    MATLAB: ConvSymAsym2D.m

    Parameters
    ----------
    A : 2D array
    M : 1D filter array
    b : char — boundary condition: 'c' (circular), 's' (symmetric), 'a' (antisymmetric)
    L : int — decomposition level (determines upsampling step = 2^(L-1))

    Notes
    -----
    For 'c' (circular): MATLAB uses imfilter(A, ker, 'circular') which is
    CORRELATION (default, no 'conv' flag) with circular boundary.
    -> scipy.ndimage.correlate(A, ker, mode='wrap')

    For 's'/'a': MATLAB uses conv2(Ae, ker, 'valid') which is true CONVOLUTION.
    -> scipy.signal.convolve2d(Ae, ker, 'valid')
    """
    m, n = A.shape
    nM = len(M)
    step = 2 ** (L - 1)

    # Build upsampled kernel (column vector)
    ker_len = step * (nM - 1) + 1
    ker = np.zeros(ker_len, dtype=np.float64)
    ker[::step] = M
    lker = ker_len // 2

    # Reshape to column filter (ker_len, 1)
    ker_2d = ker.reshape(-1, 1)

    if b == 'c':
        # Circular boundary: MATLAB imfilter default = correlation
        C = ndimage_correlate(A, ker_2d, mode='wrap')
    else:
        # Symmetric or antisymmetric boundary
        # MATLAB: padarray(A, lker, 'symmetric', 'both') — pads ROWS only
        # (scalar padsize in MATLAB pads only the first dimension)
        Ae = np.pad(A,
                    ((lker, lker), (0, 0)),
                    mode='symmetric')
        if b == 'a':
            # Negate the padded regions
            Ae[:lker, :] = -Ae[:lker, :]
            Ae[m + lker:m + 2 * lker, :] = -Ae[m + lker:m + 2 * lker, :]

        # MATLAB: conv2(Ae, ker, 'valid') — true convolution
        from scipy.signal import convolve2d
        C = convolve2d(Ae, ker_2d, mode='valid')

    return C


def FraDec2D(A, D, L):
    """
    Single-level 2D framelet decomposition (separable).
    MATLAB: FraDec2D.m

    Returns a list-of-lists Dec where Dec[i][j] is the coefficient
    for filter pair (i, j).
    """
    nD = len(D)
    SorAS = D[-1]  # boundary condition string
    n_filt = nD - 1

    Dec = [[None] * n_filt for _ in range(n_filt)]
    for i in range(n_filt):
        M1 = D[i]
        tempi = ConvSymAsym2D(A, M1, SorAS[i], L)
        for j in range(n_filt):
            M2 = D[j]
            tempj = ConvSymAsym2D(tempi.T, M2, SorAS[j], L)
            Dec[i][j] = tempj.T.copy()
    return Dec


def FraDecMultiLevel2D(A, D, L):
    """
    Multi-level 2D framelet decomposition.
    MATLAB: FraDecMultiLevel2D.m

    Returns a list Dec of length L, where Dec[k] is the single-level
    decomposition at level k+1 (0-indexed).
    """
    Dec = []
    kDec = A.copy()
    for k in range(1, L + 1):
        dec_k = FraDec2D(kDec, D, k)
        Dec.append(dec_k)
        kDec = dec_k[0][0].copy()  # low-frequency component
    return Dec


def FraRec2D(C, R, L):
    """
    Single-level 2D framelet reconstruction (separable).
    MATLAB: FraRec2D.m
    """
    nR = len(R)
    SorAS = R[-1]
    n_filt = nR - 1

    ImSize = C[0][0].shape
    Rec = np.zeros(ImSize, dtype=np.float64)

    for i in range(n_filt):
        temp = np.zeros(ImSize, dtype=np.float64)
        for j in range(n_filt):
            M2 = R[j]
            temp = temp + ConvSymAsym2D(C[i][j].T, M2, SorAS[j], L).T
        M1 = R[i]
        Rec = Rec + ConvSymAsym2D(temp, M1, SorAS[i], L)

    return Rec


# ═════════════════════════════════════════════════════════════════════════════
# Wavelet-domain kernel filtering
# ═════════════════════════════════════════════════════════════════════════════

def sort_filter(Cf, level, f_n, ratio):
    """
    Threshold wavelet coefficients at a given decomposition level.
    MATLAB: sort_filter.m

    Collects all coefficients, sorts by magnitude, zeros out the
    bottom (1-ratio) fraction.

    Parameters
    ----------
    Cf : list of list-of-lists — multi-level framelet coefficients
    level : int — 0-indexed level to filter
    f_n : int — number of filter pairs (len(R) - 1)
    ratio : float — fraction of coefficients to keep
    """
    h, w = Cf[level][0][0].shape
    num = h * w

    # Collect all coefficients into a flat vector
    v_cf = np.zeros(num * f_n * f_n, dtype=np.float64)
    n = 0
    for k in range(f_n):
        for t in range(f_n):
            v_cf[n:n + num] = Cf[level][k][t].ravel()
            n += num

    # Sort by absolute value and zero out the smallest
    indices = np.argsort(np.abs(v_cf))
    n_zero = int(np.floor(num * f_n * f_n * (1 - ratio)))
    v_cf[indices[:n_zero]] = 0.0

    # Put back
    n = 0
    for k in range(f_n):
        for t in range(f_n):
            Cf[level][k][t] = v_cf[n:n + num].reshape(h, w)
            n += num

    return Cf


def kernel_filter(C, R, L, ratio):
    """
    Filter noise on the restored kernel using wavelet thresholding.
    MATLAB: kernel_filter.m

    Parameters
    ----------
    C : list — multi-level framelet coefficients (from FraDecMultiLevel2D)
    R : list — reconstruction filter bank
    L : int — number of decomposition levels
    ratio : float — fraction of coefficients to keep

    Returns
    -------
    Rec : 2D array — filtered kernel
    """
    f_n = len(R) - 1  # number of filter pairs

    for k in range(L, 1, -1):
        # MATLAB: k goes from L down to 2 (1-indexed)
        C = sort_filter(C, k - 1, f_n, ratio)           # 0-indexed
        C[k - 2][0][0] = FraRec2D(C[k - 1], R, k)      # reconstruct level k

    C = sort_filter(C, 0, f_n, ratio)
    Rec = FraRec2D(C[0], R, 1)
    return Rec


# ═════════════════════════════════════════════════════════════════════════════
# solve_image — LUT-based solver for w-subproblem in fast_deconv
# (from solve_image.m, Krishnan & Fergus NIPS 2009)
# ═════════════════════════════════════════════════════════════════════════════

_SOLVE_IMAGE_LUT = {}  # cache: (beta, alpha) -> interp1d function


def clear_solve_image_cache():
    """Clear the persistent LUT cache (equivalent to MATLAB `clear persistent`)."""
    _SOLVE_IMAGE_LUT.clear()


def _compute_w1(v, beta):
    """alpha = 1: soft thresholding."""
    return np.maximum(np.abs(v) - 1.0 / beta, 0.0) * np.sign(v)


def _compute_w23(v, beta):
    """
    alpha = 2/3: quartic equation via Ferrari's method.
    MATLAB: compute_w23 in solve_image.m
    """
    epsilon = 1e-6

    k_val = 8.0 / (27.0 * beta ** 3)
    m = np.full_like(v, k_val)

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    m2 = m * m
    m3 = m2 * m

    alpha_q = -1.125 * v2
    beta2 = 0.25 * v3

    q = -0.125 * (m * v2)
    disc = -m3 / 27.0 + (m2 * v4) / 256.0
    r1 = -q / 2.0 + np.sqrt(disc.astype(np.complex128))

    # Cube root via exp(log/3)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.exp(np.log(r1) / 3.0)
        y = 2.0 * (-5.0 / 18.0 * alpha_q + u + (m.astype(np.complex128) / (3.0 * u)))

    W_val = np.sqrt(alpha_q.astype(np.complex128) / 3.0 + y)

    alpha_c = alpha_q.astype(np.complex128)
    beta2_c = beta2.astype(np.complex128)

    # 4 roots
    root = np.zeros((v.size, 4), dtype=np.complex128)
    v_flat = v.ravel()

    sqrt_plus = np.sqrt(-(alpha_c + y + beta2_c / W_val))
    sqrt_minus = np.sqrt(-(alpha_c + y - beta2_c / W_val))

    root[:, 0] = 0.75 * v_flat + 0.5 * (W_val + sqrt_plus)
    root[:, 1] = 0.75 * v_flat + 0.5 * (W_val - sqrt_plus)
    root[:, 2] = 0.75 * v_flat + 0.5 * (-W_val + sqrt_minus)
    root[:, 3] = 0.75 * v_flat + 0.5 * (-W_val - sqrt_minus)

    # Pick the correct root
    v_rep = np.repeat(v_flat[:, np.newaxis], 4, axis=1)
    sv2 = np.sign(v_rep)
    rsv2 = np.real(root) * sv2

    mask = ((np.abs(np.imag(root)) < epsilon) &
            (rsv2 > np.abs(v_rep) / 2.0) &
            (rsv2 < np.abs(v_rep)))

    # MATLAB: sort(mask .* rsv2, 3, 'descend') .* sv2;  w = result(:,:,1)
    filtered = mask * rsv2
    sorted_vals = np.sort(filtered, axis=1)[:, ::-1]  # descending
    w = sorted_vals[:, 0] * np.sign(v_flat)

    return np.real(w).reshape(v.shape)


def _compute_w12(v, beta):
    """
    alpha = 1/2: cubic equation.
    MATLAB: compute_w12 in solve_image.m
    """
    epsilon = 1e-6

    k_val = -0.25 / beta ** 2
    m = np.full_like(v, k_val) * np.sign(v)

    t1 = (2.0 / 3.0) * v
    v2 = v * v
    v3 = v2 * v

    with np.errstate(divide='ignore', invalid='ignore'):
        inner = (-27.0 * m - 2.0 * v3
                 + 3.0 * np.sqrt(3.0)
                 * np.sqrt((27.0 * m ** 2 + 4.0 * m * v3).astype(np.complex128)))
        t2 = np.exp(np.log(inner.astype(np.complex128)) / 3.0)

        t3 = v2.astype(np.complex128) / t2

    cbrt2 = 2.0 ** (1.0 / 3.0)
    sqrt3 = np.sqrt(3.0)

    root = np.zeros((v.size, 3), dtype=np.complex128)
    v_flat = v.ravel()
    t1_flat = t1.ravel()

    with np.errstate(invalid='ignore'):
        root[:, 0] = t1_flat + (cbrt2 / 3.0) * t3 + t2 / (3.0 * cbrt2)
        root[:, 1] = (t1_flat
                      - ((1.0 + 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
                      - ((1.0 - 1j * sqrt3) / (6.0 * cbrt2)) * t2)
        root[:, 2] = (t1_flat
                      - ((1.0 - 1j * sqrt3) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3
                      - ((1.0 + 1j * sqrt3) / (6.0 * cbrt2)) * t2)

    # Handle NaN/Inf
    bad = np.isnan(root) | np.isinf(root)
    root[bad] = 0.0

    # Pick the correct root
    v_rep = np.repeat(v_flat[:, np.newaxis], 3, axis=1)
    sv2 = np.sign(v_rep)
    rsv2 = np.real(root) * sv2

    mask = ((np.abs(np.imag(root)) < epsilon) &
            (rsv2 > 2.0 * np.abs(v_rep) / 3.0) &
            (rsv2 < np.abs(v_rep)))

    filtered = mask * rsv2
    sorted_vals = np.sort(filtered, axis=1)[:, ::-1]  # descending
    w = sorted_vals[:, 0] * np.sign(v_flat)

    return np.real(w).reshape(v.shape)


def _newton_w(v, beta, alpha):
    """
    General alpha: Newton-Raphson solver.
    MATLAB: newton_w in solve_image.m
    """
    iterations = 4
    x = v.copy()

    for _ in range(iterations):
        fd = alpha * np.sign(x) * np.abs(x) ** (alpha - 1) + beta * (x - v)
        fdd = alpha * (alpha - 1) * np.abs(x) ** (alpha - 2) + beta
        fdd[fdd == 0] = 1e-10
        x = x - fd / fdd

    x[np.isnan(x)] = 0.0

    # Check whether the zero solution is better
    z = beta / 2.0 * v ** 2
    f = np.abs(x) ** alpha + beta / 2.0 * (x - v) ** 2
    w = np.where(f < z, x, 0.0)
    return w


def _compute_w(v, beta, alpha):
    """Dispatch to the appropriate solver for a given alpha."""
    if abs(alpha - 1.0) < 1e-9:
        return _compute_w1(v, beta)
    elif abs(alpha - 2.0 / 3.0) < 1e-9:
        return _compute_w23(v, beta)
    elif abs(alpha - 0.5) < 1e-9:
        return _compute_w12(v, beta)
    else:
        return _newton_w(v, beta, alpha)


def solve_image(v, beta, alpha):
    """
    Solve component-wise:  min_w |w|^alpha + (beta/2)*(w - v)^2
    using a Look-Up Table (LUT) with linear interpolation.
    MATLAB: solve_image.m (Krishnan & Fergus, NIPS 2009)

    The LUT is built once per (beta, alpha) pair and cached.
    """
    key = (beta, alpha)

    if key not in _SOLVE_IMAGE_LUT:
        range_val = 10.0
        step = 0.0001
        xx = np.arange(-range_val, range_val + step / 2, step)
        lut_vals = _compute_w(xx, beta, alpha)
        _SOLVE_IMAGE_LUT[key] = interp1d(
            xx, lut_vals, kind='linear', fill_value='extrapolate',
            assume_sorted=True)

    interp_func = _SOLVE_IMAGE_LUT[key]

    orig_shape = v.shape
    w = interp_func(v.ravel())
    return w.reshape(orig_shape)
