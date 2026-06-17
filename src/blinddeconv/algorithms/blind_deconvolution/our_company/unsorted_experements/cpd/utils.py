"""
utils.py

Utility functions for the CPD (Cross Partial Derivative) blind deconvolution.

Ported from MATLAB code by Ting, Wang & Hwang (IEEE TIP 2025).
Reference:
    K.-C. Ting, S.-J. Wang, R.-B. Hwang: "Fast Blind Image Deblurring
    Based on Cross Partial Derivative", IEEE Transactions on Image
    Processing, vol. 34, pp. 8627-8640, 2025.
    DOI: 10.1109/TIP.2025.3645574

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    conv2(Img, kernel, 'same'):
        MATLAB conv2 performs true convolution (flips kernel), 'same'
        returns center part of same size as first argument.
        → scipy.signal.convolve2d(Img, kernel, mode='same', boundary='fill')

    detectHarrisFeatures / cornerPoints:
        MATLAB uses its own Harris implementation with FilterSize and
        MinQuality parameters. Corner.Location returns [x, y] (col, row).
        → cv2.cornerHarris + manual NMS, or skimage, returning [x, y]
          to match MATLAB's convention.

    bwconncomp + labelmatrix:
        MATLAB bwconncomp with connectivity 4 or 8, then labelmatrix.
        → scipy.ndimage.label with appropriate structure.

    fspecial('Gaussian', Win, Sigma):
        MATLAB: hsize×hsize, centred, normalised to sum=1.
        → Manual construction.

    dst / idst (Discrete Sine Transform):
        MATLAB's dst is DST-I.
        → scipy.fft.dstn / idstn with type=1.

    imresize(img, factor):
        MATLAB uses bicubic by default.
        → cv2.resize with INTER_CUBIC or scipy.ndimage.zoom with order=3.

    corr2(A, B):
        MATLAB: Pearson correlation between two 2D matrices (treating
        them as 1D vectors). Returns scalar in [-1, 1].
        → Manual: np.corrcoef(A.ravel(), B.ravel())[0, 1]

    fftshift / ifftshift:
        Identical in numpy: np.fft.fftshift / np.fft.ifftshift.

    rgb2gray:
        MATLAB: 0.2989*R + 0.5870*G + 0.1140*B (same as ITU-R BT.601).
        → Weighted sum with same coefficients.

    wrap_boundary_liu.m:
        Sunghyun Cho's boundary wrapping (Liu & Jia, ICIP 2008).
        Uses solve_min_laplacian with DST-I.
        → Direct port, same as in DCP utils.py.
"""

import numpy as np
from scipy.signal import convolve2d
from scipy.fft import dstn, idstn
from scipy.ndimage import label, zoom


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu  (from wrap_boundary_liu.m — Cho / Liu & Jia ICIP 2008)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """
    Solve Laplace equation with Dirichlet boundary conditions via DST-I.

    Equivalent to the nested ``solve_min_laplacian`` inside
    ``wrap_boundary_liu.m``.

    MATLAB's ``dst``/``idst`` are DST type-I.
    ``scipy.fft.dstn(x, type=1)`` matches (up to a factor that cancels
    in the forward→divide→inverse round-trip).
    """
    H, W = boundary_image.shape
    bi = boundary_image.copy()

    # Keep only boundary values
    bi[1:-1, 1:-1] = 0.0

    # Laplacian of boundary image at interior points
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H - 1, 1:W - 1] = (
        -4.0 * bi[1:H - 1, 1:W - 1]
        + bi[1:H - 1, 2:W]
        + bi[1:H - 1, 0:W - 2]
        + bi[0:H - 2, 1:W - 1]
        + bi[2:H, 1:W - 1]
    )

    f1 = -f_bp

    # Interior only
    f2 = f1[1:H - 1, 1:W - 1]

    # 2-D DST-I
    f2sin = dstn(f2, type=1)

    # Eigenvalues of the discrete Laplacian
    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom

    # 2-D inverse DST-I
    img_tt = idstn(f3, type=1)

    # Put solution in inner points
    img_direct = bi.copy()
    img_direct[1:H - 1, 1:W - 1] = img_tt
    return img_direct


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image so boundaries are circularly smooth for FFT-based
    deconvolution.

    Equivalent to MATLAB ``wrap_boundary_liu.m``
    (Cho implementation of Liu & Jia, ICIP 2008).

    Parameters
    ----------
    img : (H, W) or (H, W, Ch) array — input image.
    img_size : (H_out, W_out) — target padded size.

    Returns
    -------
    ret : (H_out, W_out) or (H_out, W_out, Ch) array.
    """
    squeeze = False
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        squeeze = True

    H, W, Ch = img.shape
    H_out, W_out = int(img_size[0]), int(img_size[1])
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch].copy()

        # --- r_A: (2*alpha + H_w) × W ---
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

        # --- r_B: H × (2*alpha + W_w) ---
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

        # --- r_C: (2*alpha + H_w) × (2*alpha + W_w) ---
        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]

        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        # Trim exactly as MATLAB does (alpha=1):
        # A = A(alpha:end-alpha-1, :) → rows 0..H_w-1
        A = A[:H_w, :]
        # B = B(:, alpha+1:end-alpha) → cols 1..W_w
        B = B[:, 1:W_w + 1]
        # C = C(alpha+1:end-alpha, alpha+1:end-alpha) → [1:H_w+1, 1:W_w+1]
        C = C[1:H_w + 1, 1:W_w + 1]

        # ret = [img, B; A, C]
        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if squeeze:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# f_Define_fxy  — Cross Partial Derivative computation
# ═════════════════════════════════════════════════════════════════════════════

def compute_cpd(img: np.ndarray, cpd_sigma: float):
    """
    Compute the Cross Partial Derivative (CPD) image.

    Equivalent to MATLAB ``f_Define_fxy.m``.

    Constructs the 2nd-order mixed partial derivative of a Gaussian
    kernel  ∂²G_σ / (∂x ∂y)  and convolves with the input image.

    Parameters
    ----------
    img : (H, W) float64 — input grayscale image.
    cpd_sigma : float — σ of the Gaussian.

    Returns
    -------
    fxy   : (H, W) — full CPD image.
    fxy_P : (H, W) — positive CPD (negative values zeroed).
    fxy_N : (H, W) — absolute value of negative CPD (positive values zeroed).
    """
    # Build the kernel window
    win = int(np.ceil(cpd_sigma * 3)) * 2 + 1
    half = win // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    AZ, EL = np.meshgrid(coords, coords)  # AZ=x, EL=y

    # ∂²G / (∂x ∂y) = G(x,y) * (x / σ²) * (y / σ²)
    # MATLAB: grad2_fxy = 1/(2*pi*sigma^2) * exp(-(AZ^2+EL^2)/(2*sigma^2))
    #                     * (-AZ/sigma^2) * (-EL/sigma^2)
    s2 = cpd_sigma ** 2
    gauss = (1.0 / (2.0 * np.pi * s2)) * np.exp(
        -(AZ ** 2 + EL ** 2) / (2.0 * s2)
    )
    grad2_fxy = gauss * (-AZ / s2) * (-EL / s2)

    # Normalise positive and negative parts separately
    pos_mask = grad2_fxy > 0
    neg_mask = grad2_fxy < 0

    pos_sum = grad2_fxy[pos_mask].sum()
    if pos_sum != 0:
        grad2_fxy[pos_mask] = grad2_fxy[pos_mask] / pos_sum

    neg_sum = grad2_fxy[neg_mask].sum()
    if neg_sum != 0:
        grad2_fxy[neg_mask] = grad2_fxy[neg_mask] / neg_sum * (-1.0)

    # Convolve — MATLAB conv2(Img, kernel, 'same') = true convolution
    fxy = convolve2d(img, grad2_fxy, mode='same', boundary='fill', fillvalue=0)

    # Positive CPD
    fxy_P = fxy.copy()
    fxy_P[fxy < 0] = 0.0

    # Negative CPD (absolute value)
    fxy_N = fxy.copy()
    fxy_N[fxy > 0] = 0.0
    fxy_N = np.abs(fxy_N)

    return fxy, fxy_P, fxy_N


# ═════════════════════════════════════════════════════════════════════════════
# f_Harris_Corners  — Harris corner detection
# ═════════════════════════════════════════════════════════════════════════════

def harris_corners(img: np.ndarray,
                   filter_size: int = 13,
                   min_quality: float = 0.01) -> np.ndarray:
    """
    Detect Harris corners on a grayscale image.

    Equivalent to MATLAB ``f_Harris_Corners.m``.

    MATLAB's ``detectHarrisFeatures`` with FilterSize and MinQuality.
    Returns corner locations as (N, 2) array with columns [x, y]
    (i.e. [col, row]) — matching MATLAB's ``Corners.Location`` convention.

    Parameters
    ----------
    img : (H, W) float64 — grayscale image in [0, 1].
    filter_size : int — Gaussian smoothing window for Harris (default 13,
                  matching MATLAB code).
    min_quality : float — minimum quality threshold (default 0.01).

    Returns
    -------
    corners : (N, 2) float64 — corner coordinates [x, y] (col, row).
    """
    import cv2

    # cv2.cornerHarris:
    #   blockSize — size of neighbourhood for computing M (structure tensor)
    #   ksize — Sobel aperture for gradients
    #   k — Harris detector free parameter
    #
    # MATLAB's detectHarrisFeatures with FilterSize=13:
    #   FilterSize is the Gaussian smoothing window applied to the
    #   structure tensor. MATLAB default k = 0.04.
    #   In cv2, blockSize is the averaging neighbourhood (similar role
    #   to FilterSize). ksize is Sobel kernel size (must be odd).
    #
    # To closely match MATLAB with FilterSize=13:
    #   blockSize=13, ksize=3 (Sobel), k=0.04

    img_f32 = img.astype(np.float32)
    harris_response = cv2.cornerHarris(
        img_f32, blockSize=filter_size, ksize=3, k=0.04
    )

    # Threshold: MATLAB MinQuality is relative to max response
    threshold = min_quality * harris_response.max()
    harris_response[harris_response < threshold] = 0

    # Non-maximum suppression via dilation
    dilated = cv2.dilate(harris_response, None)
    corner_mask = (harris_response == dilated) & (harris_response > 0)

    # Get coordinates — MATLAB returns [x, y] = [col, row]
    rows, cols = np.where(corner_mask)
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    # Sort by descending response (like MATLAB selectStrongest)
    responses = harris_response[rows, cols]
    order = np.argsort(-responses)
    rows = rows[order]
    cols = cols[order]

    # Return [x, y] = [col, row] to match MATLAB Corners.Location
    corners = np.column_stack([cols.astype(np.float64),
                               rows.astype(np.float64)])
    return corners


# ═════════════════════════════════════════════════════════════════════════════
# f_Reduce_Matrix_Size  — max-pooling downsampling
# ═════════════════════════════════════════════════════════════════════════════

def reduce_matrix_size(img: np.ndarray, divisor: int) -> np.ndarray:
    """
    Downsample a 2D matrix by max-pooling with block size ``divisor``.

    Equivalent to MATLAB ``f_Reduce_Matrix_Size.m``.

    Parameters
    ----------
    img : (Nx, Ny) float64.
    divisor : int — block size.

    Returns
    -------
    img2 : reduced matrix.
    """
    Nx1, Ny1 = img.shape
    N1 = max(Nx1, Ny1)

    # Pad to make size divisible
    remainder = N1 % divisor
    if remainder != 0:
        Nx1_1 = N1 + (divisor - remainder)
    else:
        Nx1_1 = N1
    Ny1_1 = Nx1_1  # square target

    # Pad image
    img_pad = np.zeros((Nx1_1, Ny1_1), dtype=np.float64)
    img_pad[:Nx1, :Ny1] = img

    # Max-pooling
    out_h = Nx1_1 // divisor
    out_w = Ny1_1 // divisor
    img2 = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            block = img_pad[i * divisor:(i + 1) * divisor,
                            j * divisor:(j + 1) * divisor]
            img2[i, j] = block.max()

    return img2


# ═════════════════════════════════════════════════════════════════════════════
# f_Non_Max_Suppress  — sparse CPD peak detection
# ═════════════════════════════════════════════════════════════════════════════

def non_max_suppress(cpd_img: np.ndarray, kernel_size_est: int,
                     nms_sparsity: float) -> np.ndarray:
    """
    Find sparse CPD peaks via non-maximum suppression + sparsity screening.

    Equivalent to MATLAB ``f_Non_Max_Suppress.m``.

    Parameters
    ----------
    cpd_img : (H, W) float64 — positive or negative CPD image.
    kernel_size_est : int — estimated kernel size.
    nms_sparsity : float — sparsity threshold.

    Returns
    -------
    record : (N, 7) float64 — sparse CPD peak records.
             Columns: [row, col, value, idx, reduced_row, reduced_col, idx].
             Sorted by value descending.
    """
    # Parameters
    cut_value = int(np.ceil(kernel_size_est / 2)) + 1
    threshold = 0.01
    reduce_sizes = [3, 3]
    radius = int(np.ceil((np.prod(reduce_sizes) * 2 - 1) / 2))

    # Normalise & filter
    cpd_max = cpd_img.max()
    if cpd_max == 0:
        return np.zeros((0, 7), dtype=np.float64)
    cpd_norm = cpd_img / cpd_max
    cpd_norm[cpd_norm < threshold] = 0.0

    # Cut edges
    Nx, Ny = cpd_norm.shape
    cpd1 = np.zeros_like(cpd_norm)
    cpd1[cut_value:Nx - cut_value,
         cut_value:Ny - cut_value] = \
        cpd_norm[cut_value:Nx - cut_value,
                 cut_value:Ny - cut_value]

    # Find non-zero points in cpd1
    nz_rows, nz_cols = np.where(cpd1 > 0)
    if len(nz_rows) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    nz_vals = cpd1[nz_rows, nz_cols]
    # cpd1_nonzero: [row, col, value]
    cpd1_nz = np.column_stack([nz_rows, nz_cols, nz_vals])

    # Multi-level max-pooling reduction
    cpd2 = cpd1.copy()
    for divisor in reduce_sizes:
        cpd2 = reduce_matrix_size(cpd2, divisor)

    # Find non-zero points in cpd2
    nz2_rows, nz2_cols = np.where(cpd2 > 0)
    if len(nz2_rows) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    nz2_vals = cpd2[nz2_rows, nz2_cols]
    cpd2_nz = np.column_stack([nz2_rows, nz2_cols, nz2_vals])

    # Intersect by value
    # MATLAB: [C, ia, ib] = intersect(cpd1_nz(:,3), cpd2_nz(:,3), 'stable', 'row')
    # 'stable' keeps order of first input; returns UNIQUE common values only.
    vals1 = cpd1_nz[:, 2]
    vals2 = cpd2_nz[:, 2]
    # Build map of first occurrence in vals2
    ib_map = {}
    for idx2, v2 in enumerate(vals2):
        if v2 not in ib_map:
            ib_map[v2] = idx2

    # Match each unique value from vals1 (first occurrence only) to vals2
    ia_list = []
    seen_vals = set()
    for idx1, v1 in enumerate(vals1):
        if v1 in ib_map and v1 not in seen_vals:
            ia_list.append((idx1, ib_map[v1]))
            seen_vals.add(v1)

    if len(ia_list) == 0:
        return np.zeros((0, 7), dtype=np.float64)

    ia = np.array([p[0] for p in ia_list])
    ib = np.array([p[1] for p in ia_list])

    intersect_cpd1 = cpd1_nz[ia]
    intersect_cpd2 = cpd2_nz[ib]

    # NMS: Property 2 and Property 3 checks
    record_list = []
    Nx2, Ny2 = cpd2.shape

    for h in range(len(ia)):
        r2 = int(intersect_cpd2[h, 0])
        c2 = int(intersect_cpd2[h, 1])
        r1 = int(intersect_cpd1[h, 0])
        c1 = int(intersect_cpd1[h, 1])

        # Bounds check for cpd2 3×3 neighbourhood
        if r2 < 1 or r2 >= Nx2 - 1 or c2 < 1 or c2 >= Ny2 - 1:
            continue

        # Property 3: local max in 3×3 of cpd2
        local_max_3x3 = cpd2[r2 - 1:r2 + 2, c2 - 1:c2 + 2].max()
        if cpd2[r2, c2] == local_max_3x3:
            k = len(record_list) + 1
            record_list.append([
                intersect_cpd1[h, 0], intersect_cpd1[h, 1],
                intersect_cpd1[h, 2], k,
                intersect_cpd2[h, 0], intersect_cpd2[h, 1], k
            ])
        else:
            # Property 2: local max in radius neighbourhood of cpd1
            r1_lo = max(r1 - radius, 0)
            r1_hi = min(r1 + radius + 1, Nx)
            c1_lo = max(c1 - radius, 0)
            c1_hi = min(c1 + radius + 1, Ny)
            local_max_r = cpd1[r1_lo:r1_hi, c1_lo:c1_hi].max()
            if cpd1[r1, c1] == local_max_r:
                k = len(record_list) + 1
                record_list.append([
                    intersect_cpd1[h, 0], intersect_cpd1[h, 1],
                    intersect_cpd1[h, 2], k,
                    intersect_cpd2[h, 0], intersect_cpd2[h, 1], k
                ])

    if len(record_list) == 0:
        return np.zeros((0, 7), dtype=np.float64)

    record = np.array(record_list, dtype=np.float64)

    # Sparse weighting
    # MATLAB mask: 7×7 with center=0, inner 3×3=500, middle ring=25, outer=1
    mask = np.ones((7, 7), dtype=np.float64)
    mask[1:6, 1:6] = 25.0
    mask[2:5, 2:5] = 500.0
    mask[3, 3] = 0.0

    # Build flag and value maps on cpd2
    cpd2_flag = np.zeros_like(cpd2)
    cpd2_value = np.zeros_like(cpd2)

    for w in range(record.shape[0]):
        r = int(record[w, 4])
        c = int(record[w, 5])
        cpd2_flag[r, c] = 1.0
        cpd2_value[r, c] = record[w, 2]

    # Exclude peaks close to edge of cpd2
    to_remove = set()
    for w in range(record.shape[0]):
        r = int(record[w, 4])
        c = int(record[w, 5])
        if r < 3 or c < 3 or r > Nx2 - 4 or c > Ny2 - 4:
            to_remove.add(w)

    if to_remove:
        keep = [w for w in range(record.shape[0]) if w not in to_remove]
        if len(keep) == 0:
            return np.zeros((0, 7), dtype=np.float64)
        record = record[keep]
        # Rebuild flag/value
        cpd2_flag[:] = 0.0
        cpd2_value[:] = 0.0
        for w in range(record.shape[0]):
            r = int(record[w, 4])
            c = int(record[w, 5])
            cpd2_flag[r, c] = 1.0
            cpd2_value[r, c] = record[w, 2]

    # Sparsity calculation
    for q in range(record.shape[0]):
        rq = int(record[q, 4])
        cq = int(record[q, 5])
        for i in range(-3, 4):
            for j in range(-3, 4):
                ri = rq + i
                ci = cq + j
                if 0 <= ri < Nx2 and 0 <= ci < Ny2:
                    if cpd2_flag[ri, ci] != 0:
                        if cpd2_value[rq, cq] >= cpd2_value[ri, ci]:
                            cpd2_flag[ri, ci] += mask[i + 3, j + 3]

    # Exclude peaks exceeding sparsity threshold
    cpd2_flag[cpd2_flag > nms_sparsity] = 0.0
    cpd3 = cpd2 * np.sign(cpd2_flag)

    # Find surviving points
    nz4_rows, nz4_cols = np.where(cpd3 > 0)
    if len(nz4_rows) == 0:
        return np.zeros((0, 7), dtype=np.float64)
    nz4_vals = cpd3[nz4_rows, nz4_cols]
    cpd4_nz = np.column_stack([nz4_rows, nz4_cols, nz4_vals])

    # Intersect with original cpd1 non-zero by value
    vals4 = set(cpd4_nz[:, 2].tolist())
    ib4_map = {}
    for idx4, v4 in enumerate(cpd4_nz[:, 2]):
        if v4 not in ib4_map:
            ib4_map[v4] = idx4

    final_list = []
    for idx1, v1 in enumerate(cpd1_nz[:, 2]):
        if v1 in vals4 and v1 in ib4_map:
            idx4 = ib4_map[v1]
            final_list.append(np.concatenate([
                cpd1_nz[idx1],  # [row, col, val]
                [0.0],
                cpd4_nz[idx4]   # [row2, col2, val2]
            ]))

    if len(final_list) == 0:
        return np.zeros((0, 7), dtype=np.float64)

    result = np.array(final_list, dtype=np.float64)

    # Sort by value (column 2) descending
    order = np.argsort(-result[:, 2])
    result = result[order]

    return result


# ═════════════════════════════════════════════════════════════════════════════
# f_Filter_Peaks  — match CPD peaks to Harris corners
# ═════════════════════════════════════════════════════════════════════════════

def filter_peaks(peak: np.ndarray, harris: np.ndarray,
                 radius: int = 10):
    """
    Filter CPD peaks: keep only those near a Harris corner.

    Equivalent to MATLAB ``f_Filter_Peaks.m``.

    MATLAB convention: Peak columns are [row, col, value, ...] (7 cols).
    Harris columns are [x, y] = [col, row].

    Parameters
    ----------
    peak : (N, 7) float64 — CPD peak records from non_max_suppress.
    harris : (M, 2) float64 — Harris corners [x, y] = [col, row].
    radius : int — search radius (default 10).

    Returns
    -------
    peak2 : (K, 7) float64 — matched CPD peaks.
    harris2 : (K, 2) float64 — matched Harris corners.
    """
    if peak.shape[0] == 0 or harris.shape[0] == 0:
        return np.zeros((0, 7), dtype=np.float64), \
               np.zeros((0, 2), dtype=np.float64)

    num_peak = peak.shape[0]

    peak2_list = []
    harris2_list = []

    for i in range(num_peak):
        # Peak(i,1) = row, Peak(i,2) = col (0-indexed here)
        # MATLAB: SearchRange_X1 = Peak(i,2) - Radius  → col of peak
        #         SearchRange_Y1 = Peak(i,1) - Radius  → row of peak
        # Harris(:,1) = x = col, Harris(:,2) = y = row
        peak_row = peak[i, 0]
        peak_col = peak[i, 1]

        # Search range for Harris [x, y] = [col, row]
        x_lo = peak_col - radius
        x_hi = peak_col + radius
        y_lo = peak_row - radius
        y_hi = peak_row + radius

        # Find Harris corners within range
        mask = ((harris[:, 0] >= x_lo) & (harris[:, 0] <= x_hi) &
                (harris[:, 1] >= y_lo) & (harris[:, 1] <= y_hi))
        indices = np.where(mask)[0]

        if len(indices) > 0:
            harris_subset = harris[indices]

            # Distance: sqrt((peak_col - harris_x)^2 + (peak_row - harris_y)^2)
            distances = np.sqrt(
                (peak_col - harris_subset[:, 0]) ** 2 +
                (peak_row - harris_subset[:, 1]) ** 2
            )
            best = np.argmin(distances)

            peak2_list.append(peak[i])
            harris2_list.append(harris_subset[best])

    if len(peak2_list) == 0:
        return np.zeros((0, 7), dtype=np.float64), \
               np.zeros((0, 2), dtype=np.float64)

    peak2 = np.array(peak2_list, dtype=np.float64)
    harris2 = np.array(harris2_list, dtype=np.float64)
    return peak2, harris2


# ═════════════════════════════════════════════════════════════════════════════
# f_02_Produce_Masks  — extract patches around CPD peaks
# ═════════════════════════════════════════════════════════════════════════════

def produce_masks(cpd_img: np.ndarray, peak: np.ndarray,
                  mask_size: int) -> np.ndarray:
    """
    Extract square patches (masks) from the CPD image centred on each peak.

    Equivalent to MATLAB ``f_02_Produce_Masks.m``.

    Parameters
    ----------
    cpd_img : (H, W) float64 — CPD image (positive or negative).
    peak : (N, 7) float64 — matched CPD peaks.
    mask_size : int — size of each square mask.

    Returns
    -------
    masks : (mask_size, mask_size, N) float64 — extracted masks.
    """
    num = peak.shape[0]
    if num == 0:
        return np.zeros((mask_size, mask_size, 0), dtype=np.float64)

    masks = np.zeros((mask_size, mask_size, num), dtype=np.float64)
    half = mask_size // 2

    is_even = (mask_size % 2 == 0)

    for h in range(num):
        r = int(peak[h, 0])  # row
        c = int(peak[h, 1])  # col

        if is_even:
            # MATLAB: Peak(h,1)-floor(M/2) : Peak(h,1)+floor(M/2)-1
            r_lo = r - half
            r_hi = r + half
            c_lo = c - half
            c_hi = c + half
        else:
            # MATLAB: Peak(h,1)-floor(M/2) : Peak(h,1)+floor(M/2)
            r_lo = r - half
            r_hi = r + half + 1
            c_lo = c - half
            c_hi = c + half + 1

        # Bounds check
        H, W = cpd_img.shape
        if r_lo >= 0 and r_hi <= H and c_lo >= 0 and c_hi <= W:
            masks[:, :, h] = cpd_img[r_lo:r_hi, c_lo:c_hi]

    return masks


# ═════════════════════════════════════════════════════════════════════════════
# f_03_Connected_Component_Analysis  — CCA noise removal
# ═════════════════════════════════════════════════════════════════════════════

def connected_component_analysis(mask: np.ndarray, cca_scale: float,
                                 cca_connect_type: int) -> np.ndarray:
    """
    Clean a mask via connected component analysis — keep only the
    component containing the centre pixel.

    Equivalent to MATLAB ``f_03_Connected_Component_Analysis.m``.

    MATLAB bwconncomp connectivity:
        4 → scipy structure = [[0,1,0],[1,1,1],[0,1,0]]
        8 → scipy structure = [[1,1,1],[1,1,1],[1,1,1]]

    Parameters
    ----------
    mask : (K, K) float64 — input mask (may contain noise).
    cca_scale : float — threshold scale relative to max(mask).
    cca_connect_type : int — connectivity (4 or 8).

    Returns
    -------
    mask_out : (K, K) float64 — cleaned mask.
    """
    Nx, Ny = mask.shape
    mask = mask.copy()

    # Threshold
    cca_thresh = mask.max() * cca_scale
    mask[mask < cca_thresh] = 0.0

    # Binarise (any nonzero → True)
    binary = mask > 0

    # Structure for connectivity
    if cca_connect_type == 8:
        struct = np.ones((3, 3), dtype=np.int32)
    else:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

    labeled, num_features = label(binary, structure=struct)

    # MATLAB: Index = labeled(ceil((Nx+1)/2), ceil((Ny+1)/2))
    # MATLAB ceil((Nx+1)/2) with 1-based → center pixel
    # For 0-based: center = (Nx-1)//2 for odd, Nx//2 for even ... let's
    # match MATLAB exactly: ceil((N+1)/2) in 1-based = (N+1+1)//2 = (N+2)//2
    # in 1-based, then -1 for 0-based.
    cr = int(np.ceil((Nx + 1) / 2)) - 1
    cc = int(np.ceil((Ny + 1) / 2)) - 1
    index = labeled[cr, cc]

    # If centre has no label, use mode of non-zero labels
    if index == 0:
        nonzero_labels = labeled[labeled > 0]
        if len(nonzero_labels) > 0:
            # mode: most frequent label
            counts = np.bincount(nonzero_labels)
            index = np.argmax(counts)
        else:
            return np.zeros((Nx, Ny), dtype=np.float64)

    # Zero out everything except the selected component
    component_mask = (labeled == index).astype(np.float64)
    mask_out = mask * component_mask

    return mask_out


# ═════════════════════════════════════════════════════════════════════════════
# f_04_Adjust_PSF_Center  — centre, normalise and merge candidates
# ═════════════════════════════════════════════════════════════════════════════

def adjust_psf_center(psf_in: np.ndarray,
                      mask_size: int) -> np.ndarray:
    """
    Centre and normalise PSF candidates; discard those touching the border.

    Equivalent to MATLAB ``f_04_Adjust_PSF_Center.m``.

    Parameters
    ----------
    psf_in : (K, K, N) float64 — input masks after CCA.
    mask_size : int — kernel size estimate.

    Returns
    -------
    psf_out : (K, K, M) float64 — centred, normalised candidates (M ≤ N).
    """
    Nx, Ny, Num = psf_in.shape

    # Reference mask: border = 1, interior = 0
    reference = np.ones((Nx, Ny), dtype=np.float64)
    reference[1:-1, 1:-1] = 0.0

    psf_out_list = []

    for h in range(Num):
        psf_h = psf_in[:, :, h]

        # Skip if PSF touches the border
        if np.sum(psf_h * reference) != 0:
            continue

        # Find nonzero pixels
        nz_rows, nz_cols = np.where(psf_h != 0)
        if len(nz_rows) == 0:
            continue

        # Centre of mass
        mean_x = int(np.ceil(np.mean(nz_rows)))
        mean_y = int(np.ceil(np.mean(nz_cols)))

        # Shift to centre of mask
        # MATLAB: ceil(MaskSize/2) — this is 1-based centre
        # Convert to 0-based: subtract 1
        center = int(np.ceil(mask_size / 2)) - 1
        delta_x = center - mean_x
        delta_y = center - mean_y

        # MATLAB condition: all shifted coords must be in [1, Nx-1] (1-based)
        # i.e. in [0, Nx-2] (0-based) ... actually MATLAB checks > 0 and < Nx
        # which in 1-based means [1, Nx-1]. In 0-based: [0, Nx-2].
        # But the MATLAB code checks:
        #   sum(logical(A(:,1)+Delta_X > 0 & A(:,1)+Delta_X < Nx))
        #   - sum(logical(A(:,2)+Delta_Y > 0 & A(:,2)+Delta_Y < Ny)) == 0
        # This checks that the number of valid row shifts equals the number
        # of valid col shifts. If they differ, skip.
        shifted_rows = nz_rows + delta_x
        shifted_cols = nz_cols + delta_y

        valid_rows = np.sum((shifted_rows > 0) & (shifted_rows < Nx))
        valid_cols = np.sum((shifted_cols > 0) & (shifted_cols < Ny))

        if valid_rows - valid_cols != 0:
            continue

        # Build shifted PSF
        psf_shifted = np.zeros((Nx, Ny), dtype=np.float64)
        for idx in range(len(nz_rows)):
            sr = shifted_rows[idx]
            sc = shifted_cols[idx]
            if 0 <= sr < Nx and 0 <= sc < Ny:
                psf_shifted[sr, sc] = psf_h[nz_rows[idx], nz_cols[idx]]

        # Normalise
        s = psf_shifted.sum()
        if s != 0:
            psf_shifted = psf_shifted / s

        psf_out_list.append(psf_shifted)

    if len(psf_out_list) == 0:
        return np.zeros((Nx, Ny, 0), dtype=np.float64)

    psf_out = np.stack(psf_out_list, axis=2)
    return psf_out


# ═════════════════════════════════════════════════════════════════════════════
# f_Gaussian_Smoothing  — FFT-based Gaussian smoothing
# ═════════════════════════════════════════════════════════════════════════════

def gaussian_smoothing(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth an image with a Gaussian kernel via FFT.

    Equivalent to MATLAB ``f_Gaussian_Smoothing.m``.

    Parameters
    ----------
    img : (H, W) float64 — input image.
    sigma : float — standard deviation of the Gaussian.

    Returns
    -------
    out : (H, W) float64 — smoothed image.
    """
    win = int(np.ceil(sigma * 3)) * 2 + 1

    # Build Gaussian kernel (same as MATLAB fspecial('Gaussian', win, sigma))
    half = win // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(coords, coords)
    h = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    h = h / h.sum()

    Nx_b, Ny_b = img.shape
    Nx_h, Ny_h = h.shape

    # Padding sizes
    Nx_out = Nx_b + Nx_h
    Ny_out = Ny_b + Ny_h

    # Pad image using wrap_boundary_liu
    b1 = wrap_boundary_liu(img, (Nx_out, Ny_out))

    # Pad kernel
    h1 = np.pad(h,
                ((Nx_b // 2, Nx_b // 2), (Ny_b // 2, Ny_b // 2)),
                mode='constant', constant_values=0)

    # Adjust sizes to match b1
    if h1.shape[0] != b1.shape[0]:
        h1 = np.pad(h1, ((0, 1), (0, 0)), mode='constant')
    if h1.shape[1] != b1.shape[1]:
        h1 = np.pad(h1, ((0, 0), (0, 1)), mode='constant')

    # Trim if larger
    h1 = h1[:b1.shape[0], :b1.shape[1]]

    # Centre to upper-left (ifftshift)
    h1 = np.fft.ifftshift(h1)

    # FFT convolution
    B1 = np.fft.fft2(b1)
    H1 = np.fft.fft2(h1)
    B2 = H1 * B1
    b2 = np.real(np.fft.ifft2(B2))

    return b2[:Nx_b, :Ny_b]


# ═════════════════════════════════════════════════════════════════════════════
# f_06_Spectrum_Correlation  — candidate ranking
# ═════════════════════════════════════════════════════════════════════════════

def spectrum_correlation(b: np.ndarray, h: np.ndarray,
                         corr_sigma: float):
    """
    Rank kernel candidates by spectral correlation with the blurred image.

    Equivalent to MATLAB ``f_06_Spectrum_Correlation.m``.

    Parameters
    ----------
    b : (H, W) float64 — (resized) blurred image.
    h : (Kh, Kw, N) float64 — (resized) kernel candidates.
    corr_sigma : float — σ for Gaussian smoothing of spectra.

    Returns
    -------
    log_absB : (H_pad, W_pad) — log spectrum of blurred image.
    log_absH : (H_pad, W_pad, N) — log spectra of candidates.
    corr_sorted : (N, 2) — [index, correlation] sorted descending.
    """
    Nx_b, Ny_b = b.shape[:2]
    if h.ndim == 2:
        h = h[:, :, np.newaxis]
    Nx_h, Ny_h, Num_h = h.shape

    Nx_bh = Nx_b + Nx_h - 1
    Ny_bh = Ny_b + Ny_h - 1

    # Blurred image spectrum
    b1 = wrap_boundary_liu(b, (Nx_bh, Ny_bh))
    B1 = np.fft.fft2(b1)
    B1_shift = np.fft.fftshift(B1)
    absB1 = np.abs(B1_shift)
    absB1 = gaussian_smoothing(absB1, corr_sigma)
    log_absB = np.log10(np.maximum(absB1, 1e-300))

    # Candidates
    log_absH = np.zeros((Nx_bh, Ny_bh, Num_h), dtype=np.float64)
    corr = np.zeros((Num_h, 2), dtype=np.float64)

    for k in range(Num_h):
        # Pad kernel
        h1 = np.pad(h[:, :, k],
                     ((Nx_b // 2, Nx_b // 2), (Ny_b // 2, Ny_b // 2)),
                     mode='constant')
        h1 = h1[:Nx_bh, :Ny_bh]

        # Centre to upper-left
        h1 = np.fft.ifftshift(h1)

        H1 = np.fft.fft2(h1)
        H1_shift = np.fft.fftshift(H1)
        absH1 = np.abs(H1_shift)
        absH1 = gaussian_smoothing(absH1, corr_sigma)
        log_absH1 = np.log10(np.maximum(absH1, 1e-300))

        # 2D correlation (corr2 in MATLAB)
        a_flat = log_absB.ravel()
        b_flat = log_absH1.ravel()

        # Handle degenerate cases
        a_std = np.std(a_flat)
        b_std = np.std(b_flat)
        if a_std == 0 or b_std == 0:
            correlation = 0.0
        else:
            correlation = np.corrcoef(a_flat, b_flat)[0, 1]

        if np.isnan(correlation):
            corr[k, :] = [k, 0.0]
        else:
            corr[k, :] = [k, correlation]

        log_absH[:, :, k] = log_absH1

    # Sort by correlation descending
    order = np.argsort(-corr[:, 1])
    corr_sorted = corr[order]

    return log_absB, log_absH, corr_sorted


# ═════════════════════════════════════════════════════════════════════════════
# f_Zero_Finding  — locate zeros of kernel transfer function
# ═════════════════════════════════════════════════════════════════════════════

def zero_finding(H1_shift: np.ndarray, distance: int) -> np.ndarray:
    """
    Find zero crossings of the kernel's frequency response.

    Equivalent to MATLAB ``f_Zero_Finding.m``.

    Checks sign changes in the kernel's FFT along horizontal and
    vertical directions at the given distance.

    Parameters
    ----------
    H1_shift : (H, W) complex128 — fftshift of the kernel FFT.
    distance : int — search distance.

    Returns
    -------
    zero_mask : (H, W) float64 — binary mask (1 at zero crossings).
    """
    Nx, Ny = H1_shift.shape
    zero_mask = np.zeros((Nx, Ny), dtype=np.float64)

    d = distance

    # Vectorised implementation of the MATLAB nested loops
    # u = 1+d .. Nx-d-1,  v = 1+d .. Ny-d-1  (0-based)
    u_slice = slice(d, Nx - d)
    v_slice = slice(d, Ny - d)

    # Horizontal check: real(H[u+d,v])*real(H[u-d,v]) + imag(H[u+d,v])*imag(H[u-d,v])
    H_up = H1_shift[d + d:Nx, v_slice]       # H[u+d, v]
    H_down = H1_shift[0:Nx - 2 * d, v_slice]  # H[u-d, v]
    p = (np.real(H_up) * np.real(H_down) +
         np.imag(H_up) * np.imag(H_down))

    # Vertical check
    H_right = H1_shift[u_slice, d + d:Ny]      # H[u, v+d]
    H_left = H1_shift[u_slice, 0:Ny - 2 * d]   # H[u, v-d]
    q = (np.real(H_right) * np.real(H_left) +
         np.imag(H_right) * np.imag(H_left))

    zero_mask[u_slice, v_slice] = ((p < 0) | (q < 0)).astype(np.float64)

    return zero_mask


# ═════════════════════════════════════════════════════════════════════════════
# f_Periodic_Noise_Removal  — suppress ringing from kernel zeros
# ═════════════════════════════════════════════════════════════════════════════

def periodic_noise_removal(zero_mask: np.ndarray, R1_shift: np.ndarray,
                           kernel_size_est: int) -> np.ndarray:
    """
    Remove periodic noise caused by zeros in the kernel transfer function.

    Equivalent to MATLAB ``f_Periodic_Noise_Removal.m``.

    Uses local linear regression to estimate and subtract the noise
    component associated with the zero-crossing frequencies.

    Parameters
    ----------
    zero_mask : (H, W) float64 — binary mask of zero crossings.
    R1_shift : (H, W) complex128 — fftshift of the reconstructed image FFT.
    kernel_size_est : int — kernel size (used as averaging window).

    Returns
    -------
    R2_shift : (H, W) complex128 — cleaned frequency domain.
    """
    win = kernel_size_est

    # Extract noise component
    G1 = np.fft.ifftshift(R1_shift)
    g1 = np.fft.ifft2(G1)

    N1_shift = zero_mask * R1_shift
    N1 = np.fft.ifftshift(N1_shift)
    n1 = np.fft.ifft2(N1)

    # 1D box-filter kernel
    kernel_1d = np.ones((1, win), dtype=np.float64) / win

    # Local statistics via separable convolution
    # MATLAB: conv2(conv2(Term, Kernel', 'same'), Kernel, 'same')
    # Kernel' is (win, 1), Kernel is (1, win) → separable box filter
    kernel_col = kernel_1d.T  # (win, 1)
    kernel_row = kernel_1d    # (1, win)

    term1 = g1 * n1
    term2 = g1
    term3 = n1
    term4 = n1 ** 2

    def box_2d(arr):
        """Separable 2D box filter matching MATLAB conv2 chain."""
        tmp = convolve2d(arr, kernel_col, mode='same', boundary='fill')
        return convolve2d(tmp, kernel_row, mode='same', boundary='fill')

    t1 = box_2d(term1)
    t2 = box_2d(term2)
    t3 = box_2d(term3)
    t4 = box_2d(term4)

    # Local regression weight
    denom = t4 - t3 ** 2
    # Avoid division by zero
    denom[np.abs(denom) < 1e-30] = 1e-30
    w1 = (t1 - t2 * t3) / denom

    gg1 = g1 - w1 * n1

    # Normalisation
    mean_g1 = np.mean(np.real(g1))
    mean_gg1 = np.mean(np.real(gg1))
    if np.abs(mean_gg1) > 1e-30:
        gg1 = gg1 * mean_g1 / mean_gg1

    G2 = np.fft.fft2(gg1)
    R2_shift = np.fft.fftshift(G2)

    return R2_shift
