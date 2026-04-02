"""
utils.py

Utility functions for SelfExSR (Single Image Super-Resolution from
Transformed Self-Exemplars, Huang et al. CVPR 2015).

Ported from MATLAB code:
  https://github.com/jbhuang0604/SelfExSR

MATLAB -> Python conversion notes (CRITICAL differences):
    ──────────────────────────────────────────────────────────────────
    Indexing:
        MATLAB is 1-based column-major; Python is 0-based row-major.
        ALL index arrays in this port use 0-based row-major convention:
            ind = row * W + col   (not col * H + row).
        Functions sub2ind / ind2sub are NOT used directly; instead we
        use np.ravel_multi_index / np.unravel_index consistently.

    im2col(M, [p,p], 'sliding'):
        MATLAB scans columns first (column-major patch order).
        We use sliding_window_view and then flatten in row-major order
        (patches enumerated row-by-row).  This is fine as long as
        get_uvpix() enumerates patch centres in the **same** order
        (row-by-row), which it does.

    sub2ind([H,W], row, col):
        MATLAB:  (col-1)*H + row     (column-major, 1-based)
        Python:   row * W + col       (row-major, 0-based)

    find(mask):
        MATLAB returns column-major linear indices.
        np.nonzero / np.argwhere return row-major.
        -> get_uvpix uses np.argwhere for consistency.

    vgg_interp2(img, x, y, 'linear', 0):
        -> scipy.ndimage.map_coordinates(order=1, mode='constant', cval=0)
        NOTE: map_coordinates takes (row, col) = (y, x).
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import map_coordinates


# ═════════════════════════════════════════════════════════════════════════════
# 1. clamp  (from sr_clamp.m)
# ═════════════════════════════════════════════════════════════════════════════

def clamp(v, lo, hi):
    """Element-wise clamp: max(min(v, hi), lo)."""
    return np.clip(v, lo, hi)


# ═════════════════════════════════════════════════════════════════════════════
# 2. fspecial_gaussian
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_gaussian(size, sigma):
    """
    2-D Gaussian kernel, equivalent to MATLAB fspecial('gaussian', [size, size], sigma).
    Returns array of shape (size, size), sum == 1.
    """
    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h = w = int(size)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float64) - cy
    x = np.arange(w, dtype=np.float64) - cx
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


# ═════════════════════════════════════════════════════════════════════════════
# 3. im2col_sliding  (MATLAB im2col(..., 'sliding'))
# ═════════════════════════════════════════════════════════════════════════════

def im2col_sliding(img_2d, patch_size):
    """
    Extract sliding patches from a 2-D array, mimicking MATLAB
    im2col(img, [p, p], 'sliding').

    Parameters
    ----------
    img_2d : (H, W) ndarray
    patch_size : int or (ph, pw)

    Returns
    -------
    cols : (ph*pw, num_patches) ndarray, float32
        Each column is a flattened patch.
        Patches are enumerated in ROW-MAJOR order (left-to-right,
        top-to-bottom), which matches get_uvpix() enumeration.

    NOTE: MATLAB im2col enumerates in column-major order.  We
    intentionally use row-major here; all index bookkeeping
    (get_uvpix, trgPatchInd, etc.) is adapted accordingly.
    """
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size
    # sliding_window_view gives shape (H-ph+1, W-pw+1, ph, pw)
    windows = sliding_window_view(img_2d, (ph, pw))
    # Reshape: (num_patches, ph*pw), then transpose → (ph*pw, num_patches)
    n_rows, n_cols = windows.shape[0], windows.shape[1]
    cols = windows.reshape(n_rows * n_cols, ph * pw).T
    return cols.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# 4. update_uvMap  (from sr_update_uvMap.m)
# ═════════════════════════════════════════════════════════════════════════════

def update_uvMap(map_arr, data, pix_ind):
    """
    Write *data* into *map_arr* at linear indices *pix_ind*.

    map_arr : (H, W) or (H, W, C) — the map to update (modified in-place).
    data    : (N,) or (N, C) or scalar — values to write.
    pix_ind : (N,) int64 — linear indices into the H×W plane (row-major).

    MATLAB equivalent (column-major):
        offset = uint64((0:nCh-1)*H*W);
        uvPixInd = bsxfun(@plus, uvPixInd, offset);
        map(uvPixInd) = data;

    Python: we reshape map to (H*W, C), index, and write.
    """
    shape = map_arr.shape
    if map_arr.ndim == 2:
        # (H, W) — single channel
        flat = map_arr.ravel()
        flat[pix_ind] = data if np.isscalar(data) else np.asarray(data).ravel()
        # in-place — already modified via flat view
    else:
        # (H, W, C)
        H, W, C = shape
        flat = map_arr.reshape(H * W, C)
        if np.isscalar(data):
            flat[pix_ind] = data
        else:
            d = np.asarray(data)
            if d.ndim == 1:
                # broadcast single value per pixel across channels
                flat[pix_ind] = d[:, None] if C > 1 else d
            else:
                flat[pix_ind] = d
    return map_arr


# ═════════════════════════════════════════════════════════════════════════════
# 5. uvMat_from_uvMap  (from sr_uvMat_from_uvMap.m)
# ═════════════════════════════════════════════════════════════════════════════

def uvMat_from_uvMap(uv_map, pix_ind):
    """
    Read data from *uv_map* at linear indices *pix_ind*.

    uv_map  : (H, W) or (H, W, C)
    pix_ind : (N,) int64  — row-major linear indices into H×W

    Returns
    -------
    mat : (N,) or (N, C)
    """
    if uv_map.ndim == 2:
        return uv_map.ravel()[pix_ind]
    else:
        H, W, C = uv_map.shape
        flat = uv_map.reshape(H * W, C)
        return flat[pix_ind]


# ═════════════════════════════════════════════════════════════════════════════
# 6. get_uvpix  (from sr_init_lvl_nnf.m, inner function sr_get_uvpix)
# ═════════════════════════════════════════════════════════════════════════════

def get_uvpix(img_size, prad):
    """
    Get patch-centre pixel positions for PatchMatch.

    Parameters
    ----------
    img_size : (H, W)
    prad     : int — patch radius = floor(patchSize / 2)

    Returns
    -------
    uvPix : dict with keys
        'sub'       : (N, 2) float32, each row = (col_x, row_y)  0-based
        'ind'       : (N,) int64, row-major linear index
        'mask'      : (H, W) bool, True for valid patch centres
        'numUvPix'  : int

    MATLAB note: MATLAB sr_get_uvpix returns sub as (X, Y) i.e. (col, row).
    We keep the same convention: sub[:, 0] = x (col), sub[:, 1] = y (row).
    ind = row * W + col   (0-based row-major).

    Enumeration order: np.argwhere scans row-major (top-to-bottom, left-to-right),
    same order as im2col_sliding patch enumeration.
    """
    H, W = img_size
    mask = np.ones((H, W), dtype=bool)
    # Zero out border of width prad
    mask[:prad, :] = False
    mask[H - prad:, :] = False
    mask[:, :prad] = False
    mask[:, W - prad:] = False

    # argwhere returns (row, col) pairs, sorted row-major
    rc = np.argwhere(mask)  # (N, 2): row, col
    rows = rc[:, 0]
    cols = rc[:, 1]

    sub = np.column_stack([cols, rows]).astype(np.float32)  # (N, 2): x, y
    ind = (rows * W + cols).astype(np.int64)

    return {
        'sub': sub,
        'ind': ind,
        'mask': mask,
        'numUvPix': len(ind),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. scale_tform  (from sr_scale_tform.m)
# ═════════════════════════════════════════════════════════════════════════════

def scale_tform(H):
    """
    Estimate approximate scale from homography columns.

    H : (N, 9) — each row is a flattened 3×3 homography stored as
        [h1, h2, h3, h4, h5, h6, h7, h8, h9]
        which represents the matrix:
            [[h1, h4, h7],
             [h2, h5, h8],
             [h3, h6, h9]]

    Returns (N,) scale values.

    MATLAB:
        uvTformScale = (H(:,1) - H(:,7).*H(:,3)).* (H(:,5) - H(:,8).*H(:,6))
                     - (H(:,4) - H(:,7).*H(:,6)).* (H(:,2) - H(:,8).*H(:,3));
        uvTformScale = sqrt(abs(uvTformScale));
    """
    H = np.asarray(H, dtype=np.float32)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    h4, h5, h6 = H[:, 3], H[:, 4], H[:, 5]
    h7, h8 = H[:, 6], H[:, 7]
    det = (h1 - h7 * h3) * (h5 - h8 * h6) - (h4 - h7 * h6) * (h2 - h8 * h3)
    return np.sqrt(np.abs(det))


# ═════════════════════════════════════════════════════════════════════════════
# 8. trans_tform  (from sr_trans_tform.m)
# ═════════════════════════════════════════════════════════════════════════════

def trans_tform(uv_tform, d):
    """
    Apply a translation offset *d* to homography transformation(s).

    uv_tform : (N, 9) float32
    d        : (2,) or (N, 2) — displacement (dx, dy)

    Returns (N, 9) updated tform.

    MATLAB stores the homography as:
        [h1 h2 h3 h4 h5 h6 h7 h8 h9]
    Columns 7,8,9 are the translation row.

    Translation update:
        h7_new = h1*dx + h4*dy + h7
        h8_new = h2*dx + h5*dy + h8
        h9_new = h3*dx + h6*dy + h9
    Then normalise by h9.
    """
    out = uv_tform.copy()
    d = np.asarray(d, dtype=np.float32)

    if d.ndim == 1:
        dx, dy = d[0], d[1]
        out[:, 6] = uv_tform[:, 0] * dx + uv_tform[:, 3] * dy + uv_tform[:, 6]
        out[:, 7] = uv_tform[:, 1] * dx + uv_tform[:, 4] * dy + uv_tform[:, 7]
        out[:, 8] = uv_tform[:, 2] * dx + uv_tform[:, 5] * dy + uv_tform[:, 8]
    else:
        dx = d[:, 0]
        dy = d[:, 1]
        out[:, 6] = uv_tform[:, 0] * dx + uv_tform[:, 3] * dy + uv_tform[:, 6]
        out[:, 7] = uv_tform[:, 1] * dx + uv_tform[:, 4] * dy + uv_tform[:, 7]
        out[:, 8] = uv_tform[:, 2] * dx + uv_tform[:, 5] * dy + uv_tform[:, 8]

    # Normalise by h9
    h9 = out[:, 8] + 1e-10
    out = out / h9[:, None]
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 9. check_valid_pos  (from sr_update_NNF.m, inner sr_check_valid_pos)
# ═════════════════════════════════════════════════════════════════════════════

def check_valid_pos(pos, img_size, prad):
    """
    Check if source patch positions are within valid bounds.

    pos      : (N, 2) — (x=col, y=row), 0-based
    img_size : (H, W)
    prad     : int

    Returns (N,) bool.

    MATLAB (1-based):
        valid = (x <= W - prad) & (x >= prad + 1) &
                (y >= prad + 1) & (y <= H - prad);

    Python (0-based):
        valid = (x <= W - 1 - prad) & (x >= prad) &
                (y >= prad) & (y <= H - 1 - prad);
    """
    H, W = img_size
    x = pos[:, 0]
    y = pos[:, 1]
    return (x >= prad) & (x <= W - 1 - prad) & (y >= prad) & (y <= H - 1 - prad)


# ═════════════════════════════════════════════════════════════════════════════
# 10. prep_plane_prob_acc  (from sr_prep_plane_prob_acc.m)
# ═════════════════════════════════════════════════════════════════════════════

def prep_plane_prob_acc(plane_prob, pix_ind):
    """
    Compute cumulative probability for inverse-CDF plane sampling.

    plane_prob : (H, W, numPlane)
    pix_ind    : (N,) int64 — row-major linear indices

    Returns (N, numPlane + 1) float32 — accumulative probabilities starting from 0.
    """
    num_plane = plane_prob.shape[2]
    N = len(pix_ind)
    acc = np.zeros((N, num_plane + 1), dtype=np.float32)

    for i in range(num_plane):
        prob_i = plane_prob[:, :, i].ravel()[pix_ind]
        acc[:, i + 1] = prob_i
        if i > 0:
            acc[:, i + 1] += acc[:, i]

    return acc


# ═════════════════════════════════════════════════════════════════════════════
# 11. draw_plane_id  (from sr_draw_plane_id.m)
# ═════════════════════════════════════════════════════════════════════════════

def draw_plane_id(plane_prob_acc):
    """
    Sample a plane ID for each pixel from the cumulative probability.

    plane_prob_acc : (N, numPlane + 1)

    Returns (N,) uint8 — 0-based plane indices.

    MATLAB returns 1-based plane ID.  We return **0-based**.
    """
    N = plane_prob_acc.shape[0]
    num_plane = plane_prob_acc.shape[1] - 1
    rand_sample = np.random.rand(N).astype(np.float32)
    plane_id = np.zeros(N, dtype=np.uint8)

    for p in range(num_plane):
        mask = (plane_prob_acc[:, p] < rand_sample) & (plane_prob_acc[:, p + 1] >= rand_sample)
        plane_id[mask] = p  # 0-based

    return plane_id


# ═════════════════════════════════════════════════════════════════════════════
# 12. vgg_interp2  (bilinear interpolation, replaces external/imrender)
# ═════════════════════════════════════════════════════════════════════════════

def vgg_interp2(img, x_coords, y_coords):
    """
    Bilinear interpolation of a multi-channel image at sub-pixel positions.

    Equivalent to MATLAB:
        vgg_interp2(img, x, y, 'linear', 0)
    where out-of-bounds pixels are set to 0.

    Parameters
    ----------
    img       : (H, W, C) float32
    x_coords  : (pNumPix, 1, N) or (pNumPix, N) — x (column) coordinates, 0-based
    y_coords  : (pNumPix, 1, N) or (pNumPix, N) — y (row) coordinates, 0-based

    Returns
    -------
    result : (pNumPix, N, C) float32
    """
    x = np.squeeze(x_coords)  # (pNumPix, N) or (pNumPix,)
    y = np.squeeze(y_coords)

    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]

    pNumPix, N = x.shape
    C = img.shape[2] if img.ndim == 3 else 1

    result = np.zeros((pNumPix, N, C), dtype=np.float32)

    for c in range(C):
        ch = img[:, :, c] if img.ndim == 3 else img
        # map_coordinates expects (row, col) coordinates
        coords = np.array([y.ravel(), x.ravel()], dtype=np.float64)
        interped = map_coordinates(ch, coords, order=1, mode='constant', cval=0.0)
        result[:, :, c] = interped.reshape(pNumPix, N).astype(np.float32)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 13. apply_affine_tform  (from sr_update_NNF.m, inner sr_apply_affine_tform)
# ═════════════════════════════════════════════════════════════════════════════

def apply_affine_tform(tform_a, tform_d):
    """
    Apply affine perturbation *tform_d* to current affine params *tform_a*.

    tform_a : (N, 4) — [a1, a2, a3, a4] representing 2×2 matrix [[a1,a2],[a3,a4]]
    tform_d : (N, 4) — [d1, d2, d3, d4] perturbation matrix

    Returns (N, 4) — result = tform_d @ tform_a  (componentwise, per-pixel).

    MATLAB:
        cand(:,1:2) = D(:,1:2)*A(:,1) + D(:,3:4)*A(:,2)
        cand(:,3:4) = D(:,1:2)*A(:,3) + D(:,3:4)*A(:,4)
    """
    cand = np.zeros_like(tform_a)
    cand[:, 0] = tform_d[:, 0] * tform_a[:, 0] + tform_d[:, 2] * tform_a[:, 1]
    cand[:, 1] = tform_d[:, 1] * tform_a[:, 0] + tform_d[:, 3] * tform_a[:, 1]
    cand[:, 2] = tform_d[:, 0] * tform_a[:, 2] + tform_d[:, 2] * tform_a[:, 3]
    cand[:, 3] = tform_d[:, 1] * tform_a[:, 2] + tform_d[:, 3] * tform_a[:, 3]
    return cand


# ═════════════════════════════════════════════════════════════════════════════
# 14. draw_rand_sample  (from sr_update_NNF.m, inner sr_draw_rand_sample)
# ═════════════════════════════════════════════════════════════════════════════

def draw_rand_sample(search_pos_rad, num_uv_pix, iter_, opt):
    """
    Draw random PatchMatch candidates: position offset + affine perturbation.

    Parameters
    ----------
    search_pos_rad : float — current position search radius
    num_uv_pix     : int
    iter_          : int — iteration counter (>= 1), used to shrink transform radius
    opt            : dict with 'scaleRadA', 'rotRadA', 'shRadA'

    Returns
    -------
    src_pos_offset : (N, 2) float32 — position offsets
    uv_tform_d     : (N, 4) float32 — affine perturbation [d1,d2,d3,d4]
    """
    N = num_uv_pix

    # Position offset: uniform in [-search_pos_rad, +search_pos_rad]
    src_pos_offset = (2.0 * search_pos_rad * (np.random.rand(N, 2) - 0.5)).astype(np.float32)

    # Affine perturbation
    scale = opt['scaleRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    scale = scale + 1.0  # perturb around 1
    theta = opt['rotRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    sh_x = opt['shRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    sh_y = opt['shRadA'] * (np.random.rand(N, 1) - 0.5) / iter_

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    d = np.zeros((N, 4), dtype=np.float32)
    d[:, 0:1] = cos_t - sin_t * sh_y
    d[:, 1:2] = sin_t + cos_t * sh_y
    d[:, 2:3] = cos_t * sh_x - sin_t
    d[:, 3:4] = sin_t * sh_x + cos_t

    d *= scale
    return src_pos_offset, d
