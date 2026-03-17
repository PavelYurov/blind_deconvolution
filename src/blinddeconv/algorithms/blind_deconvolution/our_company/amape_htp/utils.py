"""
utils.py

Utility / auxiliary functions for AMAPE-HTP blind deconvolution.

Ported from C++ code by Suzuki Hironobu (Blind-Deblurring-master).

Reference:
    J. Kotera, F. Sroubek, P. Milanfar:
    "Blind deconvolution using alternating maximum a posteriori estimation
    with heavy-tailed priors", DOI: 10.1007/978-3-642-40246-3_8

C++ -> Python conversion notes:
    - Eigen .block(startRow, startCol, numRows, numCols) copies numRows x numCols.
      The C++ code uses .block(0,0,rows-1,cols-1) which copies (rows-1) x (cols-1)
      elements. This is an off-by-one in the original C++ — we replicate it exactly.
    - FFTW3 forward: no normalization; backward: 1/(rows*cols).
      numpy fft2/ifft2 match this behavior.
    - OpenCV imresize uses INTER_LANCZOS4 for sizes >= 8, INTER_CUBIC for >= 4.
      We replicate with cv2.resize.
    - C++ floor() on positive doubles matches Python math.floor / int(). For
      negative doubles, C++ floor() rounds toward -inf (same as Python math.floor).
    - sfix(d): C++ floor for d>0, ceil for d<=0 — truncation toward zero,
      equivalent to Python int() for finite values.

Contains:
    copy_mat_2_mat_zeros      — zero-pad real matrix into larger real matrix
    copy_mat_2_cmat_zeros     — zero-pad real matrix into larger complex matrix
    copy_mat_2_cmat           — copy real matrix into complex matrix (same size)
    copy_cmat_2_cmat_zeros    — zero-pad complex matrix into larger complex matrix
    normalizeImage            — normalize intensity to [0, 1]
    matNormalize              — normalize matrix to sum=1
    sfix                      — truncation toward zero
    sumabs2                   — sum of |x|^2
    power                     — exp(y * log(x))
    imresize_to_shape         — resize matrix to target (rows, cols) via OpenCV
    imresize_by_factor        — resize matrix by scalar factor via OpenCV
    getROI                    — crop image to max ROI size
    createROI                 — downsample ROI for coarse scale
    doublePSF                 — double PSF size via resize
    get_margins_mat           — find bounding box of non-zero region
    make_mask_mat             — binary mask by threshold
    mat2gray                  — normalize to [0, 1]
    bwmorph_clean             — remove isolated pixels (8-connected)
    centerPSF                 — center and threshold PSF
    set_Vh                    — clamp Vh < 0 and zero outside PSF region
    set_vrange                — get (min, max) of matrix
    uConstr                   — clip complex matrix real part to [min, max]
    newton                    — Newton's method for root finding
    snx                       — helper for asetLnorm (derivative condition)
    snx2                      — helper for asetLnorm (energy condition)
    dsnx2                     — derivative helper (same as snx)
    asetLnorm                 — compute (v_star, u_star) thresholds for Lp prior
    aLn                       — apply Lp thresholding (proximal operator)
    create_gaussian_kernel    — 2D Gaussian kernel
    normalized_autocorrelation — normalized autocorrelation of PSF
    create_weights            — weight map for edgetaper
    create_blurred            — boundary-aware convolution for edgetaper
    edgetaper                 — edge tapering to reduce boundary artifacts
"""

import numpy as np
import cv2
import math

from numpy.fft import fft2, ifft2


# ═══════════════════════════════════════════════════════════════════════════════
# Matrix copy utilities (from mat_utils.c)
#
# IMPORTANT: The original C++ code uses Eigen .block(0,0,rows-1,cols-1) which
# copies (rows-1) rows and (cols-1) columns. This is an off-by-one bug in the
# C++ code. We replicate it exactly to match behavior.
# ═══════════════════════════════════════════════════════════════════════════════

def copy_mat_2_mat_zeros(a, b_shape, rows, cols):
    """
    Copy real matrix `a` into a zero matrix of `b_shape`,
    placing a[:rows-1, :cols-1] into b[:rows-1, :cols-1].

    C++ (mat_utils.c):
        b.setZero();
        b.block(0,0,rows-1,cols-1) = a.block(0,0,rows-1,cols-1);
    """
    b = np.zeros(b_shape, dtype=np.float64)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1]
    return b


def copy_mat_2_cmat_zeros(a, b_shape, rows, cols):
    """
    Copy real matrix `a` into a zero complex matrix of `b_shape`,
    placing into real part of b[:rows-1, :cols-1].

    C++ (mat_utils.c):
        b.setZero();
        b.block(0,0,rows-1,cols-1).real() = a.block(0,0,rows-1,cols-1);
    """
    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1].astype(np.complex128)
    return b


def copy_mat_2_cmat(a, b_shape, rows, cols):
    """
    Copy real matrix `a` into complex matrix of `b_shape` (same size path).

    C++ (mat_utils.c):
        b.setZero();
        b.block(0,0,rows-1,cols-1).real() = a.block(0,0,rows-1,cols-1);

    Note: same off-by-one as other functions.
    """
    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1].astype(np.complex128)
    return b


def copy_cmat_2_cmat_zeros(a, b_shape, rows, cols):
    """
    Copy complex matrix `a` into a zero complex matrix of `b_shape`,
    placing a[:rows-1, :cols-1] into b[:rows-1, :cols-1].

    C++ (mat_utils.c):
        b.setZero();
        b.block(0,0,rows-1,cols-1) = a.block(0,0,rows-1,cols-1);
    """
    b = np.zeros(b_shape, dtype=np.complex128)
    b[:rows - 1, :cols - 1] = a[:rows - 1, :cols - 1]
    return b


# ═══════════════════════════════════════════════════════════════════════════════
# Basic math utilities
# ═══════════════════════════════════════════════════════════════════════════════

def power(x, y):
    """
    C++: exp(y * log(x)).  assert(x != 0.0).
    """
    assert x != 0.0
    return math.exp(y * math.log(x))


def sfix(d):
    """
    Truncation toward zero.
    C++: d > 0 ? floor(d) : ceil(d)
    Equivalent to Python int() for finite values.
    """
    if d > 0:
        return int(math.floor(d))
    return int(math.ceil(d))


def sumabs2(m):
    """
    Sum of |x|^2 for complex matrix.
    C++: tmp = conj(m) * m; return real(tmp.sum());
    """
    return np.sum(np.real(np.conj(m) * m))


# ═══════════════════════════════════════════════════════════════════════════════
# Image normalization utilities
# ═══════════════════════════════════════════════════════════════════════════════

def normalizeImage(m):
    """
    Normalize matrix to [0, 1].

    C++:
        min = m.minCoeff();
        max = m.maxCoeff();
        scale = max - min;
        m -= min;
        m /= scale;

    Returns
    -------
    m_out : normalized matrix
    min_val : original minimum
    max_val : original maximum
    """
    min_val = m.min()
    max_val = m.max()
    scale = max_val - min_val
    if scale < 0.0:
        scale *= -1.0
    assert scale != 0.0
    m_out = (m - min_val) / scale
    return m_out, min_val, max_val


def matNormalize(h):
    """
    Normalize matrix so that sum = 1.
    C++: s = h.sum(); if (s > 0) h /= s;

    Returns
    -------
    h_out : normalized matrix (or copy if sum <= 0)
    """
    h_out = h.copy()
    s = h_out.sum()
    if s > 0.0:
        h_out /= s
    return h_out


def mat2gray(m):
    """
    Normalize to [0, 1] (calls normalizeImage, discards min/max).
    C++: normalizeImage(m, min, max);
    """
    m_out, _, _ = normalizeImage(m)
    return m_out


# ═══════════════════════════════════════════════════════════════════════════════
# Image resize utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _get_interpolation_type(rsize, csize):
    """
    C++:
        if (8 <= rsize && 8 <= csize) type = INTER_LANCZOS4;
        else if (4 <= rsize && 4 <= csize) type = INTER_CUBIC;
        else type = INTER_LINEAR;
    """
    if 8 <= rsize and 8 <= csize:
        return cv2.INTER_LANCZOS4
    elif 4 <= rsize and 4 <= csize:
        return cv2.INTER_CUBIC
    else:
        return cv2.INTER_LINEAR


def imresize_to_shape(in_mat, sx, sy):
    """
    Resize matrix to target shape (sx, sy).

    C++ imresize(in, out, sx, sy):
        resize(src, dst, Size(sx, sy), 0, 0, type);

    IMPORTANT: OpenCV Size(width, height) but sx is rows, sy is cols
    in the C++ code:
        Mat dst = Mat(sx, sy, CV_64F);
        resize(src, dst, Size(sx,sy), ...);
    Here Size(sx,sy) means width=sx, height=sy.

    But wait — in the C++ call sites:
        imresize(hi, h, hx*2, hy*2)  where hx=h.rows(), hy=h.cols()
    So sx = target_rows, sy = target_cols.

    OpenCV resize(src, dst, Size(width, height)) — Size takes (w, h).
    But Mat(sx, sy) creates sx rows, sy cols.
    And Size(sx, sy) would be width=sx, height=sy.

    This is inconsistent in the C++ code. The Mat is (sx rows, sy cols)
    but Size(sx,sy) is (width=sx, height=sy), meaning the resize target
    would be height=sy, width=sx — which would be (sy rows, sx cols).

    Since the only call site is doublePSF with hx*2, hy*2 where hx=rows,
    hy=cols, and the output Mat is (hx*2, hy*2), the actual resize target
    in OpenCV is Size(hy*2, hx*2) = width=hy*2, height=hx*2 = (hx*2 rows, hy*2 cols).

    Actually looking more carefully at the C++ code:
        Mat dst = Mat(sx, sy, CV_64F);    // sx rows, sy cols
        resize(src, dst, Size(sx,sy), 0, 0, type);
    Here Size(sx,sy) = width=sx, height=sy. So the resize target is
    (sy rows, sx cols). But dst is (sx rows, sy cols).

    This is a bug in the C++ — Size and Mat have swapped arguments.
    But since the PSF is always square (hx == hy), it doesn't matter.
    We replicate by using (sx rows, sy cols) as the target.
    """
    rsize = in_mat.shape[0]
    csize = in_mat.shape[1]
    interp = _get_interpolation_type(rsize, csize)

    src = in_mat.astype(np.float64)
    # cv2.resize takes dsize=(width, height), so (sy, sx)
    dst = cv2.resize(src, (sy, sx), interpolation=interp)
    return dst


def imresize_by_factor(in_mat, factor):
    """
    Resize matrix by scalar factor.

    C++ imresize(in, out, float m):
        Mat dst = Mat(sfix(rsize*m), sfix(csize*m), CV_64F);
        resize(src, dst, Size(), m, m, type);
    """
    rsize = in_mat.shape[0]
    csize = in_mat.shape[1]
    interp = _get_interpolation_type(rsize, csize)

    src = in_mat.astype(np.float64)
    # cv2.resize with fx, fy
    dst = cv2.resize(src, None, fx=factor, fy=factor, interpolation=interp)
    return dst


# ═══════════════════════════════════════════════════════════════════════════════
# ROI and multi-scale utilities
# ═══════════════════════════════════════════════════════════════════════════════

def getROI(m, maxROIsize_r, maxROIsize_c):
    """
    Crop image to max ROI size.

    C++:
        if (m_r > maxROIsize_r) margin_r = floor((m_r - maxROIsize_r) / 2); m_r = maxROIsize_r;
        if (m_c > maxROIsize_c) margin_c = floor((m_c - maxROIsize_c) / 2); m_c = maxROIsize_c;
        return m.block(margin_r, margin_c, m_r + margin_r, m_c + margin_c);

    Note: Eigen .block(startR, startC, numRows, numCols).
    The C++ uses (m_r + margin_r) as numRows — this means the block is NOT
    centered; it takes from margin_r to margin_r + (m_r + margin_r) - 1.
    This is likely a bug (should be just m_r), but we replicate it.

    Actually, re-reading: m_r is reassigned to maxROIsize_r, and margin_r
    is about half the excess. So numRows = maxROIsize_r + margin_r, which
    is larger than maxROIsize_r. We replicate exactly.
    """
    m_r = m.shape[0]
    m_c = m.shape[1]
    margin_r = 0
    margin_c = 0

    if m_r > maxROIsize_r:
        margin_r = int(math.floor((m_r - maxROIsize_r) / 2))
        m_r = maxROIsize_r
    if m_c > maxROIsize_c:
        margin_c = int(math.floor((m_c - maxROIsize_c) / 2))
        m_c = maxROIsize_c

    # Eigen .block(margin_r, margin_c, m_r + margin_r, m_c + margin_c)
    # = rows [margin_r .. margin_r + (m_r+margin_r) - 1]
    # = cols [margin_c .. margin_c + (m_c+margin_c) - 1]
    end_r = margin_r + m_r + margin_r
    end_c = margin_c + m_c + margin_c
    # Clamp to actual matrix bounds
    end_r = min(end_r, m.shape[0])
    end_c = min(end_c, m.shape[1])
    return m[margin_r:end_r, margin_c:end_c].copy()


def createROI(tmp, L, MSlevels):
    """
    Downsample ROI for a given scale level L.

    C++:
        if (L != MSlevels) {
            float m = 1.0;
            for (i = L; i < MSlevels; i++) { tmp_x *= 0.5; tmp_y *= 0.5; m *= 0.5; }
            imresize(tmp, tmp2, m);
            tmp = tmp2;
        }

    Parameters
    ----------
    tmp : 2D array (ROI)
    L   : current level (1-based, 1..MSlevels)
    MSlevels : total number of levels

    Returns
    -------
    downsampled ROI (or copy if L == MSlevels)
    """
    if L != MSlevels:
        factor = 1.0
        for _ in range(L, MSlevels):
            factor *= 0.5
        tmp2 = imresize_by_factor(tmp, factor)
        return tmp2
    return tmp.copy()


def doublePSF(h):
    """
    Double the PSF size via resize.

    C++:
        hi = h;
        h.resize(hx*2, hy*2);
        imresize(hi, h, hx*2, hy*2);
    """
    hx = h.shape[0]
    hy = h.shape[1]
    return imresize_to_shape(h, hx * 2, hy * 2)


# ═══════════════════════════════════════════════════════════════════════════════
# PSF centering and morphology
# ═══════════════════════════════════════════════════════════════════════════════

def get_margins_mat(m, threshold):
    """
    Find bounding box of pixels > threshold.

    C++ uses 1-based indexing for m_top, m_bottom, m_left, m_right.
    We convert to 0-based internally but return the same logical values.

    Returns (m_left, m_right, m_top, m_bottom) in 1-based (matching C++)
    """
    rows, cols = m.shape

    # defaults (C++: 1-based)
    m_left = 1
    m_right = rows   # note: C++ uses m.rows() for m_right default
    m_top = 1
    m_bottom = cols  # C++ uses m.cols() for m_bottom default

    # m_top: first row with any pixel > threshold
    found = False
    for x in range(rows):
        for y in range(cols):
            if m[x, y] > threshold:
                m_top = x + 1  # 1-based
                found = True
                break
        if found:
            break

    # m_bottom: last row with any pixel > threshold
    found = False
    for x in range(rows - 1, -1, -1):
        for y in range(cols):
            if m[x, y] > threshold:
                m_bottom = x + 1  # 1-based
                found = True
                break
        if found:
            break

    # m_left: first col with any pixel > threshold
    found = False
    for y in range(cols):
        for x in range(rows):
            if m[x, y] > threshold:
                m_left = y + 1  # 1-based
                found = True
                break
        if found:
            break

    # m_right: last col with any pixel > threshold
    found = False
    for y in range(cols - 1, -1, -1):
        for x in range(rows):
            if m[x, y] > threshold:
                m_right = y + 1  # 1-based
                found = True
                break
        if found:
            break

    return m_left, m_right, m_top, m_bottom


def make_mask_mat(m, threshold):
    """
    Binary mask: 1 where m > threshold, 0 elsewhere.
    C++: mask = (m.array() > threshold).select(1.0, m*0.0);
    """
    return np.where(m > threshold, 1.0, 0.0)


def bwmorph_clean(mask):
    """
    Remove isolated pixels: if all 8 neighbors are 0, set pixel to 0.

    C++: pad with zeros, check all 8 neighbors for center pixel.
    """
    m_x, m_y = mask.shape
    mask_out = mask.copy()

    # Pad with zeros
    temp = np.zeros((m_x + 2, m_y + 2), dtype=np.float64)
    temp[1:m_x + 1, 1:m_y + 1] = mask

    for x in range(1, m_x + 1):
        for y in range(1, m_y + 1):
            if (temp[x - 1, y - 1] == 0.0 and
                temp[x - 1, y] == 0.0 and
                temp[x - 1, y + 1] == 0.0 and
                temp[x, y - 1] == 0.0 and
                temp[x, y + 1] == 0.0 and
                temp[x + 1, y - 1] == 0.0 and
                temp[x + 1, y] == 0.0 and
                    temp[x + 1, y + 1] == 0.0):
                mask_out[x - 1, y - 1] = 0.0

    return mask_out


def set_Vh(vh, rows, cols):
    """
    Clamp negative values of Vh and zero everything outside PSF region.

    C++:
        tmp = vh.block(0,0,rows-1, cols-1);     // (rows-1) x (cols-1)
        tmp = (tmp < 0) ? 0 : tmp;
        vh.setZero();
        vh.block(0,0,rows-1,cols-1) = tmp;

    Note: off-by-one from C++ — copies (rows-1) x (cols-1).
    """
    vh_out = np.zeros_like(vh)
    tmp = vh[:rows - 1, :cols - 1].copy()
    tmp[tmp < 0.0] = 0.0
    vh_out[:rows - 1, :cols - 1] = tmp
    return vh_out


def centerPSF(H, centering_threshold):
    """
    Center PSF: normalize to [0,1], threshold, remove isolated pixels,
    find bounding box, crop, re-normalize.

    C++:
        mat2gray(H);
        make_mask_mat(H, mask, threshold);
        bwmorph_clean(mask);
        get_margins_mat(mask, ...);
        ... compute shift, crop, zero small values, normalize.
    """
    H = mat2gray(H)
    mask = make_mask_mat(H, centering_threshold)
    mask = bwmorph_clean(mask)
    m_left, m_right, m_top, m_bottom = get_margins_mat(
        mask, centering_threshold
    )

    rows, cols = H.shape

    topleft_x = m_top   # 1-based
    topleft_y = m_left   # 1-based

    begin_x = max(topleft_x, 1) - 1   # convert to 0-based
    end_x = min(topleft_x + rows - 1, rows) - 1
    begin_y = max(topleft_y, 1) - 1
    end_y = min(topleft_y + cols - 1, cols) - 1

    # C++: shift_x = (begin_x + end_x - H.rows())/2
    # In C++ begin_x and end_x are 0-based at this point
    shift_x = (begin_x + end_x - rows) // 2
    shift_y = (begin_y + end_y - cols) // 2

    tmp = np.zeros((rows, cols), dtype=np.float64)

    for x in range(begin_x, end_x + 1):
        for y in range(begin_y, end_y + 1):
            if H[x, y] <= centering_threshold:
                tmp[x, y] = 0.0
            else:
                tmp[x, y] = H[x, y]

    H = tmp.copy()
    s = H.sum()
    if s != 0.0:
        H /= s

    return H


# ═══════════════════════════════════════════════════════════════════════════════
# Value range utilities
# ═══════════════════════════════════════════════════════════════════════════════

def set_vrange(m):
    """
    Get (min, max) of matrix.
    C++: vrange.min = m.minCoeff(); vrange.max = m.maxCoeff();
    """
    return m.min(), m.max()


def uConstr(m, vrange_min, vrange_max):
    """
    Constrain complex matrix: clip real part to [min, max], zero imag.

    C++:
        for each element:
            if real(m(x,y)) < min: m(x,y) = complex(min, 0)
            if real(m(x,y)) > max: m(x,y) = complex(max, 0)
    """
    m_out = m.copy()
    re = np.real(m_out)
    mask_lo = re < vrange_min
    mask_hi = re > vrange_max
    m_out[mask_lo] = complex(vrange_min, 0.0)
    m_out[mask_hi] = complex(vrange_max, 0.0)
    return m_out


# ═══════════════════════════════════════════════════════════════════════════════
# Newton solver and Lp prior (Sec. 3.1 of paper)
# ═══════════════════════════════════════════════════════════════════════════════

def snx(x, alpha, beta, q):
    """
    C++: y = -x + (1-q)*q*alpha/beta * power(x, q-1);
    """
    return -x + (1.0 - q) * q * alpha / beta * power(x, q - 1.0)


def snx2(x, alpha, beta, q):
    """
    C++: y = -x*x/2 + (1-q)*alpha/beta * power(x, q);
    """
    return -x * x / 2.0 + (1.0 - q) * alpha / beta * power(x, q)


def dsnx2(x, alpha, beta, q):
    """
    C++: return snx(s);   (derivative of snx2 is snx)
    """
    return snx(x, alpha, beta, q)


def newton(f, df, alpha, beta, q, x0, eps1, eps2, max_iter):
    """
    Newton's method for root finding.

    C++:
        x1 = x0; x = x1; ind = 0; sw = 0;
        while (sw == 0 && ind >= 0):
            sw = 1; ind++;
            g = f(x1);
            if |g| > eps2:
                if ind <= max:
                    dg = df(x1);
                    if |dg| > eps2:
                        x = x1 - g/dg;
                        if |x-x1| > eps1 && |x-x1| > eps1*|x|:
                            x1 = x; sw = 0;
                    else: ind = -1;
                else: ind = -1;
        return x;

    Returns
    -------
    x : root, ind : iteration count (negative means failure)
    """
    x1 = x0
    x = x1
    ind = 0
    sw = 0

    while sw == 0 and ind >= 0:
        sw = 1
        ind += 1

        g = f(x1, alpha, beta, q)

        if abs(g) > eps2:
            if ind <= max_iter:
                dg = df(x1, alpha, beta, q)
                if abs(dg) > eps2:
                    x = x1 - g / dg
                    if abs(x - x1) > eps1 and abs(x - x1) > eps1 * abs(x):
                        x1 = x
                        sw = 0
                else:
                    ind = -1
            else:
                ind = -1

    return x, ind


def asetLnorm(q, alpha, beta):
    """
    Compute (v_star, u_star) thresholds for Lp proximal operator.

    C++:
        if q == 1: v_star = 0, u_star = alpha/beta
        elif q == 0: v_star = u_star = sqrt(2*alpha/beta)
        else: Newton method

    Corresponds to solving equation (3) in the paper.

    Returns
    -------
    v_star, u_star
    """
    if q == 1.0:
        v_star = 0.0
        u_star = alpha / beta
    elif q == 0.0:
        v_star = math.sqrt(2.0 * alpha / beta)
        u_star = math.sqrt(2.0 * alpha / beta)
    else:
        eps1 = 2.0e-16
        eps2 = 2.0e-16
        max_iter = 100
        x0 = 0.1

        v_star, _ = newton(snx2, dsnx2, alpha, beta, q,
                           x0, eps1, eps2, max_iter)
        u_star = v_star + alpha / beta * q * power(v_star, q - 1.0)

    return v_star, u_star


def aLn(DU, normDU, v_star, u_star):
    """
    Apply Lp thresholding (proximal operator).

    C++:
        k = u_star - v_star;
        mask = (normDU > u_star) ? 1 : 0;
        V = DU * (normDU - k) / normDU * mask;

    V is zero where normDU <= u_star.
    Where normDU > u_star: V = DU * (normDU - k) / normDU.

    Corresponds to line 4 of Algorithm in Sec. 3.1.
    """
    k = u_star - v_star

    V = np.zeros_like(DU)
    mask = normDU > u_star

    # Avoid division by zero
    safe_norm = np.where(mask, normDU, 1.0)
    V[mask] = DU[mask] * (safe_norm[mask] - k) / safe_norm[mask]

    return V


# ═══════════════════════════════════════════════════════════════════════════════
# Edgetaper (from edgetaper.c)
# ═══════════════════════════════════════════════════════════════════════════════

def create_gaussian_kernel(psf_x, psf_y, sigma):
    """
    Create 2D Gaussian kernel, normalized to sum=1.

    C++:
        factor = 1/(sqrt(2*pi)*sigma)
        hh = psf_x / 2.0;  hw = psf_y / 2.0;
        psf(x,y) = exp(-((x-hh)^2 + (y-hw)^2) / (2*sigma^2)) * factor;
        normalize to sum=1.
    """
    psf = np.zeros((psf_x, psf_y), dtype=np.float64)
    factor = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    hh = psf_x / 2.0
    hw = psf_y / 2.0

    for x in range(psf_x):
        tmp_x = (x - hh) / sigma
        for y in range(psf_y):
            tmp_y = (y - hw) / sigma
            psf[x, y] = math.exp(-(tmp_x * tmp_x + tmp_y * tmp_y) / 2.0) * factor

    s = psf.sum()
    if s > 0.0:
        psf /= s
    return psf


def normalized_autocorrelation(psf):
    """
    Normalized autocorrelation of PSF.

    C++:
        xpsf size: (2*psf_x-1, 2*psf_y-1)
        for each (x,y):
            c = sum over (u,v) of psf(u,v) * psf(psf_x-1-x+u, psf_y-1-y+v)
            (where indices are in bounds)
        normalize by max value.
    """
    psf_x, psf_y = psf.shape
    xpsf_x = 2 * psf_x - 1
    xpsf_y = 2 * psf_y - 1
    xpsf = np.zeros((xpsf_x, xpsf_y), dtype=np.float64)

    max_val = -np.inf
    for x in range(xpsf_x):
        for y in range(xpsf_y):
            c = 0.0
            for u in range(psf_x):
                for v in range(psf_y):
                    ix = x - u
                    iy = y - v
                    if 0 <= ix < psf_x and 0 <= iy < psf_y:
                        c += psf[u, v] * psf[psf_x - 1 - ix, psf_y - 1 - iy]
            xpsf[x, y] = c
            if c > max_val:
                max_val = c

    if max_val != 0.0:
        xpsf /= max_val

    return xpsf


def create_weights(img_x, img_y, psf):
    """
    Create weight map for edgetaper.

    C++:
        mergin_x = psf_x - 1;  mergin_y = psf_y - 1;
        xpsf = normalized_autocorrelation(psf);
        9 regions: corners use xpsf values, center = 1.0, edges use
        border values of xpsf.
    """
    psf_x, psf_y = psf.shape
    mergin_x = psf_x - 1
    mergin_y = psf_y - 1
    w_x = img_x
    w_y = img_y

    xpsf = normalized_autocorrelation(psf)

    weights = np.zeros((w_x, w_y), dtype=np.float64)

    for x in range(w_x):
        for y in range(w_y):
            if x < mergin_x:
                if y < mergin_y:
                    # (I) top-left corner
                    weights[x, y] = xpsf[x, y]
                elif mergin_y <= y < (w_y - mergin_y):
                    # (IV) top edge
                    weights[x, y] = xpsf[x, mergin_y]
                else:
                    # (VII) top-right corner
                    weights[x, y] = xpsf[x, y - (w_y - 2 * mergin_y) + 1]
            elif mergin_x <= x < (w_x - mergin_x):
                if y < mergin_y:
                    # (II) left edge
                    weights[x, y] = xpsf[mergin_x, y]
                elif mergin_y <= y < (w_y - mergin_y):
                    # (V) center
                    weights[x, y] = 1.0
                else:
                    # (VIII) right edge
                    weights[x, y] = xpsf[mergin_x, y - (w_y - 2 * mergin_y) + 1]
            else:
                if y < mergin_y:
                    # (III) bottom-left corner
                    weights[x, y] = xpsf[x - (w_x - 2 * mergin_x) + 1, y]
                elif mergin_y <= y < (w_y - mergin_y):
                    # (VI) bottom edge
                    weights[x, y] = xpsf[x - (w_x - 2 * mergin_x) + 1, mergin_y]
                else:
                    # (IX) bottom-right corner
                    weights[x, y] = xpsf[
                        x - (w_x - 2 * mergin_x) + 1,
                        y - (w_y - 2 * mergin_y) + 1
                    ]

    return weights


def create_blurred(img, psf):
    """
    Boundary-aware convolution for edgetaper.

    C++:
        m_x = ((psf_x+1) >> 1) - 1
        m_y = ((psf_y+1) >> 1) - 1
        For interior pixels (away from border by psf_size): blurred = img
        For border pixels: direct convolution with boundary clamping.

    Parameters
    ----------
    img : 2D complex array
    psf : 2D real array

    Returns
    -------
    blurred : 2D complex array
    """
    img_x, img_y = img.shape
    psf_x, psf_y = psf.shape

    m_x = ((psf_x + 1) >> 1) - 1
    m_y = ((psf_y + 1) >> 1) - 1

    blurred = np.zeros_like(img)

    for x in range(img_x):
        for y in range(img_y):
            # C++: if ((psf_x < x && x <= (img_x - psf_x)) && (psf_y < y && y <= (img_y - psf_y)))
            if (psf_x < x <= (img_x - psf_x)) and (psf_y < y <= (img_y - psf_y)):
                blurred[x, y] = img[x, y]
            else:
                val = 0.0 + 0.0j
                for i in range(psf_x):
                    for j in range(psf_y):
                        u = i - m_x
                        v = j - m_y
                        if 0 <= (x + u) < img_x and 0 <= (y + v) < img_y:
                            val += psf[i, j] * img[x + u, y + v]
                blurred[x, y] = val

    return blurred


def edgetaper(img):
    """
    Edge tapering to reduce boundary artifacts before FFT.

    C++:
        psf = 10x10 Gaussian (sigma=1.0)
        blurred = create_blurred(img, psf)
        weights = create_weights(psf)
        edge = img * weights + blurred * (1 - weights)

    Parameters
    ----------
    img : 2D complex array (ROI as complex)

    Returns
    -------
    edge : 2D complex array with tapered edges
    """
    img_x, img_y = img.shape

    psf_x = 10
    psf_y = 10
    psf = create_gaussian_kernel(psf_x, psf_y, 1.0)

    blurred = create_blurred(img, psf)
    weights = create_weights(img_x, img_y, psf)

    edge = np.zeros_like(img)
    for x in range(img_x):
        for y in range(img_y):
            if weights[x, y] != 1.0:
                edge[x, y] = img[x, y] * weights[x, y] + blurred[x, y] * (1.0 - weights[x, y])
            else:
                edge[x, y] = img[x, y]

    return edge
