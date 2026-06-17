"""
solvers.py

Core solver functions for SelfExSR (Single Image Super-Resolution from
Transformed Self-Exemplars, Huang et al. CVPR 2015).

Ported from MATLAB code:
  https://github.com/jbhuang0604/SelfExSR

Contains:
    sr_init_opt               — parameter initialisation         (sr_init_opt.m)
    sr_extract_plane_simple   — simplified plane model           (sr_extract_plane.m)
    sr_create_img_pyramid     — image pyramid construction       (sr_create_img_pyramid.m)
    sr_planar_structure_pyramid — plane model per pyramid level  (sr_planar_structure_pyramid.m)
    sr_init_nnf / sr_upsample / sr_init_lvl_nnf — NNF init      (sr_init_lvl_nnf.m)
    sr_prep_target_patch      — target patch extraction          (sr_prep_target_patch.m)
    sr_prep_source_patch      — source patch extraction          (sr_prep_source_patch.m)
    sr_patch_cost_app         — appearance patch cost            (sr_patch_cost_app.m)
    sr_patch_cost_plane       — plane compatibility cost         (sr_patch_cost_plane.m)
    sr_src_domain_tform       — perspective-aware homography     (sr_src_domain_tform.m)
    sr_random_search / sr_propagate / sr_update_NNF — PatchMatch (sr_update_NNF.m)
    sr_pass                   — iterative NNF estimation         (sr_pass.m)
    sr_voting                 — weighted patch voting            (sr_voting.m)
    sr_backprojection         — iterative back-projection        (sr_backprojection.m)
    sr_synthesis              — coarse-to-fine synthesis          (sr_synthesis.m)
    sr_demo                   — main pipeline                    (sr_demo.m)

All arrays use 0-based row-major indexing.
sub = (x=col, y=row), ind = row * W + col.
"""

import numpy as np
import cv2
from .utils import (
    clamp,
    fspecial_gaussian,
    im2col_sliding,
    update_uvMap,
    uvMat_from_uvMap,
    get_uvpix,
    scale_tform,
    trans_tform,
    check_valid_pos,
    prep_plane_prob_acc,
    draw_plane_id,
    vgg_interp2,
    apply_affine_tform,
    draw_rand_sample,
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. sr_init_opt  (from sr_init_opt.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_init_opt(SRF):
    """
    Initialise parameters for the SelfExSR algorithm.

    Parameters
    ----------
    SRF : int — super-resolution factor (2, 3, 4, or 8).

    Returns
    -------
    opt : dict — all algorithm parameters.
    """
    opt = {}
    opt['SRF'] = SRF

    # --- Multi-resolution pyramid ---
    if SRF % 2 == 0:
        opt['nLvlToRedRes'] = 3
        opt['alpha'] = (1 / 2) ** (1 / opt['nLvlToRedRes'])
        opt['coarseLvlImgScale'] = 1 / 8
    else:
        opt['nLvlToRedRes'] = 5
        opt['alpha'] = (1 / 3) ** (1 / opt['nLvlToRedRes'])
        opt['coarseLvlImgScale'] = 1 / 9

    opt['nPyrLowLvl'] = round(np.log(opt['coarseLvlImgScale']) / np.log(opt['alpha']))
    opt['nPyrLvl'] = 2 * opt['nPyrLowLvl'] + 1
    opt['origResLvl'] = round(opt['nPyrLvl'] / 2)  # 0-based: subtract 1 later
    opt['resampleKernel'] = 'bicubic'

    if SRF % 2 == 0:
        opt['topLevel'] = opt['origResLvl'] - int(
            (np.log(SRF) / np.log(2)) * opt['nLvlToRedRes']
        )
    else:
        opt['topLevel'] = opt['origResLvl'] - int(
            (np.log(SRF) / np.log(3)) * opt['nLvlToRedRes']
        )

    # --- Method parameters ---
    opt['scaleThres'] = 1.0 / opt['alpha']
    opt['lambdaScale'] = 1e-3
    opt['lambdaPlane'] = 1e-3

    opt['numIter'] = 15
    opt['numIterDec'] = 3
    opt['numIterMin'] = 3
    opt['nIterBP'] = 20
    opt['bpKernelSigma'] = 1.0

    opt['useScaleCost'] = True
    opt['usePlaneGuide'] = False  # No VP detection in Python port
    opt['useAffine'] = True
    opt['useBiasCorrection'] = True

    # --- Patch parameters ---
    opt['pSize'] = 5
    opt['pRad'] = opt['pSize'] // 2
    opt['pNumPix'] = opt['pSize'] ** 2
    opt['pMidPix'] = opt['pNumPix'] // 2

    # --- Affine search parameters ---
    opt['scaleRadA'] = 1.0
    opt['rotRadA'] = np.pi / 4
    opt['shRadA'] = 0.05

    opt['minScale'] = 1.0 / opt['alpha']
    opt['maxScale'] = 8.0

    opt['minBias'] = -0.25
    opt['maxBias'] = 0.25

    opt['costType'] = 'L2'

    # Gaussian patch weight
    w_patch = fspecial_gaussian(opt['pSize'], 3.0)
    opt['wPatch'] = w_patch.reshape(opt['pNumPix'], 1, 1).astype(np.float32)

    # Reference patch positions: (pNumPix, 3) — [dx, dy, 1]
    r = opt['pRad']
    Y, X = np.mgrid[-r:r + 1, -r:r + 1]
    opt['refPatchPos'] = np.column_stack([X.ravel(), Y.ravel(),
                                          np.ones(opt['pNumPix'])]).astype(np.float32)

    # Propagation directions: (dx, dy)
    opt['propDir'] = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.int32)

    opt['errThres'] = 0.0

    opt['fpPlaneProb'] = 1e-4
    opt['filterSize'] = 100
    opt['filterSigma'] = 50
    opt['numFilterIter'] = 5

    return opt


# ═════════════════════════════════════════════════════════════════════════════
# 2. sr_extract_plane_simple — simplified (no VP detection)
# ═════════════════════════════════════════════════════════════════════════════

def sr_extract_plane_simple(img_shape, opt):
    """
    Create a trivial plane model with a single fronto-parallel plane.

    This replaces the full sr_extract_plane which requires a VP detection
    binary.  With usePlaneGuide=False the algorithm still works (all
    patches are assigned to the fronto-parallel plane).

    Parameters
    ----------
    img_shape : (H, W) or (H, W, C)
    opt       : dict

    Returns
    -------
    model_plane : dict
    """
    H = img_shape[0]
    W = img_shape[1]

    num_plane = 1  # only fronto-parallel
    plane = {
        'vLine': np.array([0.0, 0.0, 1.0]),
        'imgPlaneProb': opt['fpPlaneProb'] * np.ones((H, W), dtype=np.float64),
    }

    # Posterior probability: just the single plane → all 1.0
    post_prob = np.ones((H, W, num_plane), dtype=np.float64)

    return {
        'numPlane': num_plane,
        'plane': [plane],
        'postProb': post_prob,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. sr_create_img_pyramid  (from sr_create_img_pyramid.m)
# ═════════════════════════════════════════════════════════════════════════════

def _imresize(img, size_hw, interp=cv2.INTER_CUBIC):
    """Resize image to (H, W). Works for 2-D and 3-D images."""
    H, W = int(size_hw[0]), int(size_hw[1])
    if img.ndim == 3:
        return cv2.resize(img, (W, H), interpolation=interp).reshape(H, W, img.shape[2])
    return cv2.resize(img, (W, H), interpolation=interp)


def sr_create_img_pyramid(img, opt):
    """
    Create high-freq and low-freq image pyramids.

    Parameters
    ----------
    img : (H, W, 3) float32 — input LR image, [0, 1]
    opt : dict

    Returns
    -------
    img_pyr_h     : list of (Hi, Wi, 3) arrays (or None)
    img_pyr_l     : list of (Hi, Wi, 3) arrays (or None)
    scale_img_pyr : list of dicts {'imgScale': float, 'imgSize': (Hi, Wi)}
    """
    H, W = img.shape[:2]
    n_pyr_lvl = opt['nPyrLvl']
    n_low = opt['nPyrLowLvl']
    orig_lvl = opt['origResLvl']
    top_level = opt['topLevel']
    alpha = opt['alpha']

    # scales: alpha^(-nPyrLowLvl) ... alpha^0 ... alpha^(nPyrLowLvl)
    exponents = np.linspace(-n_low, n_low, n_pyr_lvl)
    scale_pyr = alpha ** exponents
    # Add dummy level for low-pass
    scale_pyr = np.append(scale_pyr, scale_pyr[-1] * alpha)

    img_h_pyr = np.round(H * scale_pyr).astype(int)
    img_w_pyr = np.round(W * scale_pyr).astype(int)

    # Allocate: use None for unused levels (0-indexed → lists of length n_pyr_lvl)
    img_pyr_h = [None] * (n_pyr_lvl + 2)
    img_pyr_l = [None] * (n_pyr_lvl + 2)
    scale_img_pyr = [None] * (n_pyr_lvl + 2)

    # Fill scale info for all relevant levels
    for k in range(top_level, n_pyr_lvl + 2):
        if k < len(img_h_pyr):
            scale_img_pyr[k] = {
                'imgScale': scale_pyr[k],
                'imgSize': (int(img_h_pyr[k]), int(img_w_pyr[k])),
            }

    # Fill high-freq pyramid from origResLvl upwards (larger scales)
    for k in range(orig_lvl, n_pyr_lvl + 2):
        if k < len(img_h_pyr):
            img_pyr_h[k] = _imresize(img, (img_h_pyr[k], img_w_pyr[k]))

    # Original resolution
    img_pyr_h[orig_lvl] = img.copy()
    scale_img_pyr[orig_lvl] = {'imgScale': 1.0, 'imgSize': (H, W)}

    # Low-freq pyramid: downscale then upscale
    for k in range(orig_lvl, n_pyr_lvl + 1):
        if img_pyr_h[k + 1] is not None:
            down = _imresize(img_pyr_h[k + 1], (img_h_pyr[k], img_w_pyr[k]))
            img_pyr_l[k] = _imresize(down, (img_h_pyr[k], img_w_pyr[k]))
        else:
            img_pyr_l[k] = _imresize(img_pyr_h[k], (img_h_pyr[k], img_w_pyr[k]))

    return img_pyr_h, img_pyr_l, scale_img_pyr


# ═════════════════════════════════════════════════════════════════════════════
# 4. sr_planar_structure_pyramid  (from sr_planar_structure_pyramid.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_planar_structure_pyramid(scale_img_pyr, model_plane, top_level):
    """
    Rescale plane model for each pyramid level.

    Returns list of per-level plane models.
    """
    num_levels = len(scale_img_pyr)
    model_plane_pyr = [None] * num_levels

    for lvl in range(top_level, num_levels):
        if scale_img_pyr[lvl] is None:
            continue
        scale_cur = scale_img_pyr[lvl]['imgScale']
        size_cur = scale_img_pyr[lvl]['imgSize']

        lvl_model = {'numPlane': model_plane['numPlane'], 'rectMat': []}

        for ip in range(model_plane['numPlane']):
            H_mat = np.eye(3, dtype=np.float64)
            vline = model_plane['plane'][ip]['vLine'].copy()
            # Scale the vanishing line
            vline[0] /= scale_cur
            vline[1] /= scale_cur
            H_mat[2, :] = vline
            lvl_model['rectMat'].append(H_mat)

        # Resize posterior probability
        post = model_plane['postProb']
        # Resize each plane channel
        resized_planes = []
        for ip in range(model_plane['numPlane']):
            ch = post[:, :, ip]
            resized_planes.append(
                cv2.resize(ch, (size_cur[1], size_cur[0]),
                           interpolation=cv2.INTER_CUBIC)
            )
        plane_prob = np.stack(resized_planes, axis=-1)
        # Normalise
        plane_prob_sum = plane_prob.sum(axis=-1, keepdims=True)
        plane_prob_sum[plane_prob_sum == 0] = 1.0
        plane_prob = plane_prob / plane_prob_sum

        lvl_model['planeProb'] = plane_prob.astype(np.float32)
        lvl_model['mLogLPlaneProb'] = -np.log(
            np.clip(plane_prob, 1e-10, None)
        ).astype(np.float32)

        model_plane_pyr[lvl] = lvl_model

    return model_plane_pyr


# ═════════════════════════════════════════════════════════════════════════════
# 5–7. NNF initialisation  (from sr_init_lvl_nnf.m)
# ═════════════════════════════════════════════════════════════════════════════

def _build_trg_patch_ind(imgH, imgW, pSize):
    """
    Build index array for extracting target patches from a flat image.

    Returns (pSize*pSize, num_patches) int64 — indices into flat (H*W) array.
    Patches enumerated row-major (same as im2col_sliding and get_uvpix).
    """
    pix_map = np.arange(imgH * imgW, dtype=np.int64).reshape(imgH, imgW)
    # im2col_sliding returns (pSize*pSize, num_patches)
    return im2col_sliding(pix_map.astype(np.float32), pSize).astype(np.int64)


def _build_w_sum_img(imgH, imgW, pSize, w_patch, trg_patch_ind):
    """Compute per-pixel sum of Gaussian weights from overlapping patches."""
    w_sum = np.zeros(imgH * imgW, dtype=np.float32)
    n_patches = trg_patch_ind.shape[1]
    for i in range(n_patches):
        w_sum[trg_patch_ind[:, i]] += w_patch
    return w_sum.reshape(imgH, imgW)


def sr_src_domain_tform(plane_id, model_plane, tform_a, src_pos, trg_pos):
    """
    Compute perspective-aware homography for each pixel.

    Parameters
    ----------
    plane_id    : (N,) uint8 — 0-based plane index
    model_plane : dict with 'numPlane', 'rectMat'
    tform_a     : (N, 4) float32 — affine params
    src_pos     : (N, 2) float32 — source position (x, y), 0-based
    trg_pos     : (N, 2) float32 — target position (x, y), 0-based

    Returns
    -------
    uv_tform_t : (N, 9) float32 — final domain transformation

    NOTE: MATLAB uses 1-based coords internally (subtracts 1 then adds back).
    In Python we're already 0-based, so we skip the -1/+1 offset.
    """
    N = src_pos.shape[0]
    uv_tform_h = np.zeros((N, 9), dtype=np.float32)
    I_flat = np.eye(3, dtype=np.float32).ravel()  # [1,0,0,0,1,0,0,0,1]

    for ip in range(model_plane['numPlane']):
        rect_mat = model_plane['rectMat'][ip]
        h7 = np.float32(rect_mat[2, 0])
        h8 = np.float32(rect_mat[2, 1])

        mask = plane_id == ip
        n_cur = mask.sum()
        if n_cur == 0:
            continue

        # Target and source in rectified domain
        trg_cur = trg_pos[mask]  # (n_cur, 2) — already 0-based
        src_cur = src_pos[mask]

        # Apply H to get rectified coords: x_r = x / (x*h7 + y*h8 + 1)
        def _apply_H(pts):
            denom = pts[:, 0] * h7 + pts[:, 1] * h8 + 1.0
            return pts / (denom[:, None] + 1e-10)

        trg_r = _apply_H(trg_cur)
        src_r = _apply_H(src_cur)

        d_rect = src_r - trg_r  # (n_cur, 2)
        dx = d_rect[:, 0]
        dy = d_rect[:, 1]

        t = np.zeros((n_cur, 9), dtype=np.float32)
        # cols [0,3,6] = dx * [h7, h8, 1]
        t[:, 0] = dx * h7
        t[:, 3] = dx * h8
        t[:, 6] = dx * 1.0
        # cols [1,4,7] = dy * [h7, h8, 1]
        t[:, 1] = dy * h7
        t[:, 4] = dy * h8
        t[:, 7] = dy * 1.0
        # cols [2,5,8] = -(dx*h7 + dy*h8) * [h7, h8, 1]
        d_temp = dx * h7 + dy * h8
        t[:, 2] = -d_temp * h7
        t[:, 5] = -d_temp * h8
        t[:, 8] = -d_temp * 1.0

        t += I_flat[None, :]

        # Apply translation offset (Eqn 9): trans_tform(t, trg_cur)
        t_shifted = trans_tform(t, trg_cur)
        # MATLAB adds +1 to cols 7:8 (converting back to 1-based).
        # In 0-based we don't need that — the result is already correct.

        uv_tform_h[mask] = t_shifted

    # Apply affine: T = H * A
    uv_tform_t = uv_tform_h.copy()
    uv_tform_t[:, 0:3] = (uv_tform_h[:, 0:3] * tform_a[:, 0:1]
                           + uv_tform_h[:, 3:6] * tform_a[:, 1:2])
    uv_tform_t[:, 3:6] = (uv_tform_h[:, 3:6] * tform_a[:, 2:3]
                           + uv_tform_h[:, 3:6] * tform_a[:, 3:4])

    # Normalise by h9
    h9 = uv_tform_t[:, 8:9] + 1e-10
    uv_tform_t = uv_tform_t / h9

    return uv_tform_t


def sr_init_nnf(img_size, model_plane, opt):
    """
    Initialise NNF for the coarsest level (random sampling).

    Returns NNF dict.
    """
    imgH, imgW = img_size
    pSize = opt['pSize']
    pRad = opt['pRad']

    NNF = {}
    NNF['imgH'] = imgH
    NNF['imgW'] = imgW

    # --- uvPix ---
    NNF['uvPix'] = get_uvpix(img_size, pRad)

    # --- uvPixN: 4-connected neighbours ---
    uvPixN = []
    for i in range(4):
        d = opt['propDir'][i]
        n_sub = NNF['uvPix']['sub'] - d.astype(np.float32)
        n_row = n_sub[:, 1].astype(int)
        n_col = n_sub[:, 0].astype(int)
        n_ind = (n_row * imgW + n_col).astype(np.int64)
        # valid if neighbour is also a valid patch centre
        valid = NNF['uvPix']['mask'].ravel()[n_ind]
        uvPixN.append({'sub': n_sub, 'ind': n_ind, 'validInd': valid})
    NNF['uvPixN'] = uvPixN

    # --- trgPatchInd ---
    num_pixel = imgH * imgW
    ind_trg = _build_trg_patch_ind(imgH, imgW, pSize)  # (pSize^2, num_patches)
    # For 3 channels: stack channel offsets
    NNF['trgPatchInd'] = np.concatenate([
        ind_trg,
        ind_trg + num_pixel,
        ind_trg + 2 * num_pixel,
    ], axis=0).astype(np.int64)

    num_uv = NNF['uvPix']['numUvPix']
    num_plane = model_plane['numPlane']

    # --- uvPlaneID ---
    # Initialise all pixels to the last plane (fronto-parallel) = numPlane - 1 (0-based)
    NNF['uvPlaneID'] = {
        'data': np.full(num_uv, num_plane - 1, dtype=np.uint8),
        'map': np.full((imgH, imgW), num_plane - 1, dtype=np.uint8),
        'numPlane': num_plane,
    }
    if model_plane.get('planeProb') is not None:
        NNF['uvPlaneID']['planeProbAcc'] = prep_plane_prob_acc(
            model_plane['planeProb'], NNF['uvPix']['ind']
        )
    else:
        # Uniform
        pp = np.ones((imgH, imgW, num_plane), dtype=np.float32) / num_plane
        NNF['uvPlaneID']['planeProbAcc'] = prep_plane_prob_acc(
            pp, NNF['uvPix']['ind']
        )

    # --- uvTformA: initial affine = 2× scale (identity direction) ---
    NNF['uvTformA'] = {
        'data': np.zeros((num_uv, 4), dtype=np.float32),
        'map': np.zeros((imgH, imgW, 4), dtype=np.float32),
    }
    NNF['uvTformA']['data'][:, 0] = 2.0
    NNF['uvTformA']['data'][:, 3] = 2.0
    update_uvMap(NNF['uvTformA']['map'], NNF['uvTformA']['data'], NNF['uvPix']['ind'])

    # --- uvTformH: domain transformation ---
    NNF['uvTformH'] = {
        'data': sr_src_domain_tform(
            NNF['uvPlaneID']['data'], model_plane,
            NNF['uvTformA']['data'],
            NNF['uvPix']['sub'], NNF['uvPix']['sub']
        ),
        'map': np.zeros((imgH, imgW, 9), dtype=np.float32),
    }
    update_uvMap(NNF['uvTformH']['map'], NNF['uvTformH']['data'], NNF['uvPix']['ind'])

    # --- uvBias ---
    NNF['uvBias'] = {
        'data': np.zeros((1, 3, num_uv), dtype=np.float32),
        'map': np.zeros((imgH, imgW, 3), dtype=np.float32),
    }

    # --- uvCost ---
    NNF['uvCost'] = {
        'data': np.zeros(num_uv, dtype=np.float32),
        'map': np.zeros((imgH, imgW), dtype=np.float32),
    }

    # --- Patch weights ---
    w = fspecial_gaussian(pSize, 3.0).ravel().astype(np.float32)
    NNF['wPatch'] = w
    ind_trg_single = _build_trg_patch_ind(imgH, imgW, pSize)
    NNF['wSumImg'] = _build_w_sum_img(imgH, imgW, pSize, w, ind_trg_single)

    # --- update tracking ---
    NNF['update'] = {
        'data': np.zeros(num_uv, dtype=bool),
        'map': np.zeros((imgH, imgW), dtype=bool),
    }

    return NNF


def sr_upsample(img_size, nnf_l, model_plane, opt):
    """
    Upsample NNF from a lower-resolution level.
    """
    imgH_H, imgW_H = img_size
    imgH_L, imgW_L = nnf_l['imgH'], nnf_l['imgW']
    pSize = opt['pSize']
    pRad = opt['pRad']

    NNF = {}
    NNF['imgH'] = imgH_H
    NNF['imgW'] = imgW_H

    # --- uvPix ---
    NNF['uvPix'] = get_uvpix(img_size, pRad)
    num_uv = NNF['uvPix']['numUvPix']

    # --- uvPixN ---
    uvPixN = []
    for i in range(4):
        d = opt['propDir'][i]
        n_sub = NNF['uvPix']['sub'] - d.astype(np.float32)
        n_row = n_sub[:, 1].astype(int)
        n_col = n_sub[:, 0].astype(int)
        n_row_c = np.clip(n_row, 0, imgH_H - 1)
        n_col_c = np.clip(n_col, 0, imgW_H - 1)
        n_ind = (n_row_c * imgW_H + n_col_c).astype(np.int64)
        valid = NNF['uvPix']['mask'].ravel()[n_ind]
        uvPixN.append({'sub': n_sub, 'ind': n_ind, 'validInd': valid})
    NNF['uvPixN'] = uvPixN

    # --- trgPatchInd ---
    num_pixel = imgH_H * imgW_H
    ind_trg = _build_trg_patch_ind(imgH_H, imgW_H, pSize)
    NNF['trgPatchInd'] = np.concatenate([
        ind_trg, ind_trg + num_pixel, ind_trg + 2 * num_pixel,
    ], axis=0).astype(np.int64)

    # --- Correspondence to low-res level ---
    sX = imgW_L / imgW_H
    sY = imgH_L / imgH_H
    pix_l_sub = NNF['uvPix']['sub'] * np.array([[sX, sY]], dtype=np.float32)
    pix_l_sub = np.round(pix_l_sub).astype(np.float32)
    pix_l_sub[:, 0] = clamp(pix_l_sub[:, 0], pRad, imgW_L - 1 - pRad)
    pix_l_sub[:, 1] = clamp(pix_l_sub[:, 1], pRad, imgH_L - 1 - pRad)
    pix_l_col = pix_l_sub[:, 0].astype(int)
    pix_l_row = pix_l_sub[:, 1].astype(int)
    pix_l_ind = (pix_l_row * imgW_L + pix_l_col).astype(np.int64)

    # --- uvPlaneID ---
    plane_data = uvMat_from_uvMap(nnf_l['uvPlaneID']['map'], pix_l_ind)
    plane_data = plane_data.astype(np.uint8)
    plane_data[plane_data >= model_plane['numPlane']] = 0
    NNF['uvPlaneID'] = {
        'data': plane_data,
        'map': np.zeros((imgH_H, imgW_H), dtype=np.uint8),
        'numPlane': model_plane['numPlane'],
    }
    update_uvMap(NNF['uvPlaneID']['map'], plane_data, NNF['uvPix']['ind'])

    if model_plane.get('planeProb') is not None:
        NNF['uvPlaneID']['planeProbAcc'] = prep_plane_prob_acc(
            model_plane['planeProb'], NNF['uvPix']['ind']
        )

    # --- uvTformA ---
    tform_a = uvMat_from_uvMap(nnf_l['uvTformA']['map'], pix_l_ind).astype(np.float32)
    NNF['uvTformA'] = {
        'data': tform_a,
        'map': np.zeros((imgH_H, imgW_H, 4), dtype=np.float32),
    }
    update_uvMap(NNF['uvTformA']['map'], tform_a, NNF['uvPix']['ind'])

    # --- uvTformH ---
    tform_h = uvMat_from_uvMap(nnf_l['uvTformH']['map'], pix_l_ind).astype(np.float32)
    # Rescale source position (cols 6,7 = x,y)
    tform_h[:, 6] /= sX
    tform_h[:, 7] /= sY

    # Refine: adjust for pixel position difference
    refine_vec = NNF['uvPix']['sub'] - pix_l_sub * np.array([[1.0 / sX, 1.0 / sY]], dtype=np.float32)
    tform_h = trans_tform(tform_h, refine_vec)
    tform_h[:, 6] = clamp(tform_h[:, 6], pRad, imgW_H - 1 - pRad)
    tform_h[:, 7] = clamp(tform_h[:, 7], pRad, imgH_H - 1 - pRad)

    I_flat = np.eye(3, dtype=np.float32).ravel()
    NNF['uvTformH'] = {
        'data': tform_h,
        'map': np.tile(I_flat, (imgH_H, imgW_H, 1)).reshape(imgH_H, imgW_H, 9).copy(),
    }
    update_uvMap(NNF['uvTformH']['map'], tform_h, NNF['uvPix']['ind'])

    # --- uvBias ---
    bias = uvMat_from_uvMap(nnf_l['uvBias']['map'], pix_l_ind).astype(np.float32)
    NNF['uvBias'] = {
        'data': bias.reshape(1, 3, num_uv) if bias.ndim == 2 else np.zeros((1, 3, num_uv), dtype=np.float32),
        'map': np.zeros((imgH_H, imgW_H, 3), dtype=np.float32),
    }
    if bias.ndim == 2:
        update_uvMap(NNF['uvBias']['map'], bias, NNF['uvPix']['ind'])

    # --- uvCost ---
    cost = uvMat_from_uvMap(nnf_l['uvCost']['map'], pix_l_ind).astype(np.float32)
    NNF['uvCost'] = {
        'data': cost,
        'map': np.zeros((imgH_H, imgW_H), dtype=np.float32),
    }
    update_uvMap(NNF['uvCost']['map'], cost, NNF['uvPix']['ind'])

    # --- Patch weights ---
    w = fspecial_gaussian(pSize, 3.0).ravel().astype(np.float32)
    NNF['wPatch'] = w
    ind_trg_single = _build_trg_patch_ind(imgH_H, imgW_H, pSize)
    NNF['wSumImg'] = _build_w_sum_img(imgH_H, imgW_H, pSize, w, ind_trg_single)

    # --- update tracking ---
    NNF['update'] = {
        'data': np.zeros(num_uv, dtype=bool),
        'map': np.zeros((imgH_H, imgW_H), dtype=bool),
    }

    return NNF


def sr_init_lvl_nnf(img_size, nnf, model_plane, opt):
    """Dispatcher: init or upsample NNF depending on level."""
    if opt['iLvl'] == opt['origResLvl'] - 1:
        return sr_init_nnf(img_size, model_plane, opt)
    else:
        return sr_upsample(img_size, nnf, model_plane, opt)


# ═════════════════════════════════════════════════════════════════════════════
# 8. sr_prep_target_patch  (from sr_prep_target_patch.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_prep_target_patch(img, patch_size):
    """
    Extract axis-aligned target patches.

    Parameters
    ----------
    img : (H, W, 3) float32
    patch_size : int

    Returns
    -------
    trg_patch : (pSize*pSize, 3, num_patches) float32
    """
    C = img.shape[2] if img.ndim == 3 else 1
    patches = []
    for c in range(C):
        ch = img[:, :, c] if img.ndim == 3 else img
        cols = im2col_sliding(ch, patch_size)  # (pSize^2, num_patches)
        patches.append(cols)
    # Stack: (pSize^2, C, num_patches)
    return np.stack(patches, axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 9. sr_prep_source_patch  (from sr_prep_source_patch.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_prep_source_patch(img_pyr, uv_tform, opt):
    """
    Sample source patches from the image pyramid using the homography tform.

    Parameters
    ----------
    img_pyr : list of (H, W, 3) arrays
    uv_tform : (N, 9) float32 — homography per patch
    opt      : dict

    Returns
    -------
    src_patch      : (pNumPix, 3, N) float32
    src_patch_scale : (N,) float32
    """
    alpha = opt['alpha']
    N = uv_tform.shape[0]
    pNumPix = opt['pNumPix']
    pMidPix = opt['pMidPix']

    src_patch_scale = scale_tform(uv_tform)

    # Closest level in pyramid
    uv_scale_lvl = np.round(-np.log(src_patch_scale + 1e-10) / np.log(alpha)).astype(int)
    uv_scale_lvl = clamp(uv_scale_lvl, 1, opt['nPyrLowLvl'])

    # Adjust tform scale
    src_patch_scaleQ = alpha ** uv_scale_lvl.astype(np.float32)
    # Columns [0,3,6, 1,4,7] are multiplied by scaleQ
    tform_adj = uv_tform.copy()
    for col in [0, 3, 6, 1, 4, 7]:
        tform_adj[:, col] *= src_patch_scaleQ

    src_patch = np.zeros((pNumPix, N, 3), dtype=np.float32)

    ref_pos = opt['refPatchPos']  # (pNumPix, 3): [dx, dy, 1]

    for lvl_cur in range(1, opt['nPyrLowLvl'] + 1):
        scale_mask = uv_scale_lvl == lvl_cur
        n_cur = scale_mask.sum()
        if n_cur == 0:
            continue

        pyr_idx = opt['iLvl'] + lvl_cur
        if pyr_idx >= len(img_pyr) or img_pyr[pyr_idx] is None:
            continue
        img = img_pyr[pyr_idx]

        tf = tform_adj[scale_mask]  # (n_cur, 9)

        # Reshape transform columns for batch multiply
        c1 = tf[:, 0:3]  # (n_cur, 3): columns of the homography
        c2 = tf[:, 3:6]
        c3 = tf[:, 6:9]

        # src_pos[p, :, n] = refPatchPos[p, 0] * c1[n, :] + refPatchPos[p, 1] * c2[n, :] + c3[n, :]
        # Shape: (pNumPix, n_cur, 3)
        src_pos = (ref_pos[:, 0:1, None] * c1.T[None, :, :]
                   + ref_pos[:, 1:2, None] * c2.T[None, :, :]
                   + c3.T[None, :, :])
        # src_pos shape: (pNumPix, 3, n_cur), swap to (pNumPix, n_cur, 3)
        src_pos = src_pos.transpose(0, 2, 1)

        # Convert from homogeneous to Euclidean
        w = src_pos[:, :, 2:3]
        non_unit = np.abs(w[:, :, 0] - 1.0) > 1e-6
        # Only divide where w != 1 (at midpix check)
        mid_non_unit = non_unit[pMidPix, :]
        if mid_non_unit.any():
            inds = np.where(mid_non_unit)[0]
            src_pos[:, inds, 0:2] /= (w[:, inds, :] + 1e-10)

        x_coords = src_pos[:, :, 0]  # (pNumPix, n_cur)
        y_coords = src_pos[:, :, 1]

        # Bilinear interpolation per channel
        interped = vgg_interp2(img, x_coords, y_coords)  # (pNumPix, n_cur, C)
        src_patch[:, scale_mask, :] = interped

    # Permute to (pNumPix, 3, N) to match target patch layout
    src_patch = src_patch.transpose(0, 2, 1)
    return src_patch, src_patch_scale


# ═════════════════════════════════════════════════════════════════════════════
# 10. sr_patch_cost_app  (from sr_patch_cost_app.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_patch_cost_app(trg_patch, src_patch, opt):
    """
    Appearance-based patch matching cost (L2 + bias correction).

    trg_patch, src_patch : (pNumPix, 3, N) float32
    opt : dict

    Returns
    -------
    uv_cost : (N,) float32
    uv_bias : (1, 3, N) float32 or None
    """
    uv_bias = None
    if opt['useBiasCorrection']:
        mean_trg = trg_patch.mean(axis=0, keepdims=True)  # (1, 3, N)
        mean_src = src_patch.mean(axis=0, keepdims=True)
        uv_bias = mean_trg - mean_src
        uv_bias = clamp(uv_bias, opt['minBias'], opt['maxBias'])
        src_patch = src_patch + uv_bias

    diff = trg_patch - src_patch
    if opt['costType'] == 'L2':
        diff = diff ** 2
    else:
        diff = np.abs(diff)

    # Apply Gaussian weight
    w = opt['wPatch']  # (pNumPix, 1, 1)
    weighted = diff * w
    uv_cost = weighted.sum(axis=0).sum(axis=0)  # (N,)

    return uv_cost, uv_bias


# ═════════════════════════════════════════════════════════════════════════════
# 11. sr_patch_cost_plane  (from sr_patch_cost_plane.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_patch_cost_plane(mlog_plane_prob, plane_id_data, trg_pix_ind, src_pix_sub):
    """
    Plane compatibility cost.

    mlog_plane_prob : (H, W, numPlane) float32
    plane_id_data   : (N,) uint8 — 0-based plane index
    trg_pix_ind     : (N,) int64 — row-major linear index of target pixels
    src_pix_sub     : (N, 2) — source positions (x, y), 0-based

    Returns (N,) float32.
    """
    H, W, num_plane = mlog_plane_prob.shape

    src_sub = np.round(src_pix_sub).astype(int)
    src_sub[:, 0] = clamp(src_sub[:, 0], 0, W - 1)
    src_sub[:, 1] = clamp(src_sub[:, 1], 0, H - 1)

    # Source index into 3D array (row, col, plane)
    src_row = src_sub[:, 1]
    src_col = src_sub[:, 0]

    # Flatten mlog to (H*W, numPlane)
    flat = mlog_plane_prob.reshape(H * W, num_plane)

    # Target cost
    cost_trg = flat[trg_pix_ind, plane_id_data.astype(int)]
    # Source cost
    src_ind = src_row * W + src_col
    cost_src = flat[src_ind, plane_id_data.astype(int)]

    return cost_trg + cost_src


# ═════════════════════════════════════════════════════════════════════════════
# 13. sr_random_search  (from sr_update_NNF.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_random_search(trg_patch, img_pyr, NNF, model_plane, opt):
    """
    PatchMatch random search: coarse-to-fine random sampling.

    Returns (NNF, n_update_total).
    """
    imgH, imgW = NNF['imgH'], NNF['imgW']
    uvPix = NNF['uvPix']
    num_uv = uvPix['numUvPix']

    search_rad = max(imgH, imgW) / 4.0
    n_update_total = 0
    iter_ = 1

    while search_rad > 1.0:
        iter_ += 1
        search_rad /= 2.0

        # Current transform data
        uv_tform_h_cand = uvMat_from_uvMap(NNF['uvTformH']['map'], uvPix['ind']).astype(np.float32)
        uv_tform_a_cand = uvMat_from_uvMap(NNF['uvTformA']['map'], uvPix['ind']).astype(np.float32)

        # Draw random samples
        src_pos_offset, uv_tform_d = draw_rand_sample(search_rad, num_uv, iter_, opt)

        # Apply position offset
        src_pos = uv_tform_h_cand[:, 6:8] + src_pos_offset
        # Apply affine perturbation
        uv_tform_a_cand = apply_affine_tform(uv_tform_a_cand, uv_tform_d)

        # Draw plane ID
        if opt['usePlaneGuide']:
            plane_id_cand = draw_plane_id(NNF['uvPlaneID']['planeProbAcc'])
        else:
            plane_id_cand = np.full(num_uv, model_plane['numPlane'] - 1, dtype=np.uint8)

        # Compute domain transformation
        uv_tform_h_cand = sr_src_domain_tform(
            plane_id_cand, model_plane, uv_tform_a_cand, src_pos, uvPix['sub']
        )

        # --- Reject invalid ---
        h_scale = scale_tform(uv_tform_h_cand)
        valid_scale = (h_scale >= opt['minScale']) & (h_scale <= opt['maxScale'])
        valid_pos = check_valid_pos(uv_tform_h_cand[:, 6:8], (imgH, imgW), opt['pRad'])
        valid_err = NNF['uvCost']['data'] > opt['errThres']
        valid_all = valid_scale & valid_pos & valid_err

        valid_idx = np.where(valid_all)[0]
        if len(valid_idx) == 0:
            continue

        # Subset
        trg_sub = trg_patch[:, :, valid_all]
        cost_cur = NNF['uvCost']['data'][valid_all]
        tf_h_sub = uv_tform_h_cand[valid_all]
        tf_a_sub = uv_tform_a_cand[valid_all]
        plane_sub = plane_id_cand[valid_all]

        pix_sub = uvPix['sub'][valid_all]
        pix_ind = uvPix['ind'][valid_all]

        # Compute cost
        src_patch, src_scale = sr_prep_source_patch(img_pyr, tf_h_sub, opt)
        cost_cand, bias_cand = sr_patch_cost_app(trg_sub, src_patch, opt)

        if opt['useScaleCost']:
            cost_scale = opt['lambdaScale'] * np.maximum(0, opt['scaleThres'] - src_scale)
            cost_cand += cost_scale

        if opt['usePlaneGuide'] and model_plane.get('mLogLPlaneProb') is not None:
            cost_plane = sr_patch_cost_plane(
                model_plane['mLogLPlaneProb'], plane_sub, pix_ind,
                src_pos[valid_all]
            )
            cost_cand += opt['lambdaPlane'] * cost_plane

        # Update where cost improved
        update_mask = cost_cand < cost_cur
        if not update_mask.any():
            continue

        up_global = valid_idx[update_mask]

        NNF['uvTformH']['data'][up_global] = tf_h_sub[update_mask]
        NNF['uvTformA']['data'][up_global] = tf_a_sub[update_mask]
        NNF['uvPlaneID']['data'][up_global] = plane_sub[update_mask]
        NNF['uvCost']['data'][up_global] = cost_cand[update_mask]

        if opt['useBiasCorrection'] and bias_cand is not None:
            NNF['uvBias']['data'][:, :, up_global] = bias_cand[:, :, update_mask]

        NNF['update']['data'][up_global] = True

        # Update maps
        up_pix_ind = pix_ind[update_mask]
        update_uvMap(NNF['uvTformH']['map'], tf_h_sub[update_mask], up_pix_ind)
        update_uvMap(NNF['uvTformA']['map'], tf_a_sub[update_mask], up_pix_ind)
        update_uvMap(NNF['uvPlaneID']['map'], plane_sub[update_mask], up_pix_ind)
        update_uvMap(NNF['uvCost']['map'], cost_cand[update_mask], up_pix_ind)

        if opt['useBiasCorrection'] and bias_cand is not None:
            b = np.squeeze(bias_cand[:, :, update_mask])  # (3, n_update) or (3,)
            if b.ndim == 1:
                b = b.reshape(1, -1)
            else:
                b = b.T  # (n_update, 3)
            update_uvMap(NNF['uvBias']['map'], b, up_pix_ind)

        update_uvMap(NNF['update']['map'], True, up_pix_ind)
        n_update_total += update_mask.sum()

    return NNF, n_update_total


# ═════════════════════════════════════════════════════════════════════════════
# 14. sr_propagate  (from sr_update_NNF.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_propagate(trg_patch, img_pyr, NNF, model_plane, opt, direction):
    """
    PatchMatch propagation in one direction.

    Returns (NNF, n_update_total).
    """
    imgH, imgW = NNF['imgH'], NNF['imgW']
    n_update_total = 0

    uvPixN = NNF['uvPixN'][direction]

    # Valid: neighbour is valid, was updated, and current cost is above threshold
    valid_cost = NNF['uvCost']['data'] > opt['errThres']
    update_map_flat = NNF['update']['map'].ravel()
    valid = uvPixN['validInd'] & update_map_flat[uvPixN['ind']] & valid_cost

    n_valid = valid.sum()

    while n_valid > 0:
        n_valid = 0
        v_idx = np.where(valid)[0]

        pix_sub = NNF['uvPix']['sub'][valid]
        pix_ind = NNF['uvPix']['ind'][valid]

        n_sub = uvPixN['sub'][valid]
        n_ind = uvPixN['ind'][valid]

        trg_sub_patch = trg_patch[:, :, valid]
        src_pos_cur = NNF['uvTformH']['data'][valid, 6:8]
        cost_cur = NNF['uvCost']['data'][valid]
        plane_id_cand = uvMat_from_uvMap(NNF['uvPlaneID']['map'], n_ind).astype(np.uint8)

        # Candidate tform: take neighbour's tform and shift by propDir
        tf_a_cand = uvMat_from_uvMap(NNF['uvTformA']['map'], n_ind).astype(np.float32)
        tf_h_cand = uvMat_from_uvMap(NNF['uvTformH']['map'], n_ind).astype(np.float32)
        src_pos_neigh = tf_h_cand[:, 6:8]

        # Apply propagation shift
        tf_h_cand = trans_tform(tf_h_cand, opt['propDir'][direction].astype(np.float32))

        # Validate
        valid_src = check_valid_pos(src_pos_neigh, (imgH, imgW), opt['pRad'])
        diff_pos = np.abs(src_pos_neigh - src_pos_cur)
        valid_dist = (diff_pos[:, 0] > 1) | (diff_pos[:, 1] > 1)
        valid_err = cost_cur > opt['errThres']

        ok = valid_src & valid_dist & valid_err
        if not ok.any():
            break

        ok_idx = np.where(ok)[0]
        ok_global = v_idx[ok]

        trg_ok = trg_sub_patch[:, :, ok]
        tf_h_ok = tf_h_cand[ok]
        tf_a_ok = tf_a_cand[ok]
        plane_ok = plane_id_cand[ok]
        cost_ok = cost_cur[ok]
        pix_sub_ok = pix_sub[ok]
        pix_ind_ok = pix_ind[ok]

        # Compute cost
        src_p, src_s = sr_prep_source_patch(img_pyr, tf_h_ok, opt)
        cost_cand, bias_cand = sr_patch_cost_app(trg_ok, src_p, opt)

        if opt['useScaleCost']:
            cost_cand += opt['lambdaScale'] * np.maximum(0, opt['scaleThres'] - src_s)

        if opt['usePlaneGuide'] and model_plane.get('mLogLPlaneProb') is not None:
            cost_cand += opt['lambdaPlane'] * sr_patch_cost_plane(
                model_plane['mLogLPlaneProb'], plane_ok, pix_ind_ok,
                src_pos_cur[ok]
            )

        update = cost_cand < cost_ok
        if not update.any():
            break

        up_global = ok_global[update]
        up_pix_ind = pix_ind_ok[update]

        NNF['uvTformH']['data'][up_global] = tf_h_ok[update]
        NNF['uvTformA']['data'][up_global] = tf_a_ok[update]
        NNF['uvPlaneID']['data'][up_global] = plane_ok[update]
        NNF['uvCost']['data'][up_global] = cost_cand[update]

        if opt['useBiasCorrection'] and bias_cand is not None:
            NNF['uvBias']['data'][:, :, up_global] = bias_cand[:, :, update]

        NNF['update']['data'][up_global] = True

        update_uvMap(NNF['uvTformH']['map'], tf_h_ok[update], up_pix_ind)
        update_uvMap(NNF['uvTformA']['map'], tf_a_ok[update], up_pix_ind)
        update_uvMap(NNF['uvPlaneID']['map'], plane_ok[update], up_pix_ind)
        update_uvMap(NNF['uvCost']['map'], cost_cand[update], up_pix_ind)

        if opt['useBiasCorrection'] and bias_cand is not None:
            b = np.squeeze(bias_cand[:, :, update])
            if b.ndim == 1:
                b = b.reshape(1, -1)
            else:
                b = b.T
            update_uvMap(NNF['uvBias']['map'], b, up_pix_ind)

        update_uvMap(NNF['update']['map'], True, up_pix_ind)
        n_update_total += update.sum()

        # Prepare next round: propagate further in same direction
        next_sub = pix_sub_ok[update] + opt['propDir'][direction].astype(np.float32)
        next_col = next_sub[:, 0].astype(int)
        next_row = next_sub[:, 1].astype(int)
        next_col = np.clip(next_col, 0, imgW - 1)
        next_row = np.clip(next_row, 0, imgH - 1)
        next_ind = (next_row * imgW + next_col).astype(np.int64)

        # Reset valid mask
        valid = np.zeros(NNF['uvPix']['numUvPix'], dtype=bool)
        # Mark pixels that can still propagate
        for ni in next_ind:
            pos = np.where(NNF['uvPix']['ind'] == ni)[0]
            if len(pos) > 0:
                valid[pos[0]] = True
        valid = valid & uvPixN['validInd']
        n_valid = valid.sum()

    return NNF, n_update_total


# ═════════════════════════════════════════════════════════════════════════════
# 15. sr_update_NNF  (from sr_update_NNF.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_update_NNF(trg_patch, img_pyr, NNF, model_plane, opt):
    """
    One iteration of PatchMatch: random search + 4-directional propagation.

    Returns (NNF, n_update) where n_update = [prop_updates, rand_updates].
    """
    n_update = np.zeros(2, dtype=int)

    NNF['update']['data'][:] = False
    NNF['update']['map'][:] = False

    # Random search
    NNF, n_rand = sr_random_search(trg_patch, img_pyr, NNF, model_plane, opt)
    n_update[1] += n_rand

    # 4-directional propagation
    for d in range(4):
        NNF, n_prop = sr_propagate(trg_patch, img_pyr, NNF, model_plane, opt, d)
        n_update[0] += n_prop

    return NNF, n_update


# ═════════════════════════════════════════════════════════════════════════════
# 16. sr_pass  (from sr_pass.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_pass(img_trg, img_pyr_l, NNF, model_plane_cur, num_iter, opt):
    """
    Run PatchMatch for *num_iter* iterations at the current level.

    img_trg : (H, W, 3) — the target image (upscaled from previous level)
    img_pyr_l : list — low-freq source pyramid

    Returns updated NNF.
    """
    # Prepare target patches
    trg_patch = sr_prep_target_patch(img_trg, opt['pSize'])

    # Prepare source patches and compute initial cost
    src_patch, src_scale = sr_prep_source_patch(img_pyr_l, NNF['uvTformH']['data'], opt)
    cost_app, bias = sr_patch_cost_app(trg_patch, src_patch, opt)
    NNF['uvCost']['data'] = cost_app
    if opt['useBiasCorrection'] and bias is not None:
        NNF['uvBias']['data'] = bias

    if opt['useScaleCost']:
        cost_s = opt['lambdaScale'] * np.maximum(0, opt['scaleThres'] - src_scale)
        NNF['uvCost']['data'] += cost_s

    if opt['usePlaneGuide'] and model_plane_cur.get('mLogLPlaneProb') is not None:
        cost_p = sr_patch_cost_plane(
            model_plane_cur['mLogLPlaneProb'],
            NNF['uvPlaneID']['data'],
            NNF['uvPix']['ind'],
            NNF['uvTformH']['data'][:, 6:8],
        )
        NNF['uvCost']['data'] += opt['lambdaPlane'] * cost_p

    update_uvMap(NNF['uvCost']['map'], NNF['uvCost']['data'], NNF['uvPix']['ind'])

    # Iterative PatchMatch
    for it in range(1, num_iter + 1):
        NNF, n_upd = sr_update_NNF(trg_patch, img_pyr_l, NNF, model_plane_cur, opt)
        avg_cost = NNF['uvCost']['data'].mean()
        print(f"    {it:3d}\t{n_upd[0]:12d}\t{n_upd[1]:12d}\t{avg_cost:14.6f}")

    return NNF


# ═════════════════════════════════════════════════════════════════════════════
# 17. sr_voting  (from sr_voting.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_voting(img_pyr_h, NNF, opt):
    """
    Reconstruct image from source patches via weighted voting + back-projection.

    Returns (H, W, 3) float32.
    """
    imgH = NNF['imgH']
    imgW = NNF['imgW']
    pSize = opt['pSize']

    # Source patches
    src_patch, _ = sr_prep_source_patch(img_pyr_h, NNF['uvTformH']['data'], opt)
    # src_patch: (pNumPix, 3, N)

    # Bias correction
    if opt['useBiasCorrection']:
        src_patch = src_patch + NNF['uvBias']['data']

    # Gaussian weight
    w = NNF['wPatch']  # (pNumPix,)
    src_patch = src_patch * w[:, None, None]

    # Weighted average via accumulation
    imgAcc = np.zeros((imgH, imgW, 3), dtype=np.float32)
    out_H = imgH - pSize + 1
    out_W = imgW - pSize + 1
    num_patches = out_H * out_W

    for c in range(3):
        ch_data = src_patch[:, c, :]  # (pSize^2, N)
        # Scatter-add each pixel of each patch
        for pi in range(pSize * pSize):
            pr, pc_ = divmod(pi, pSize)
            # Each patch n corresponds to top-left corner (row_n, col_n)
            for n in range(num_patches):
                row_n = n // out_W
                col_n = n % out_W
                imgAcc[row_n + pr, col_n + pc_, c] += ch_data[pi, n]

    # Normalise
    w_sum = NNF['wSumImg'][:, :, None]
    w_sum = np.maximum(w_sum, 1e-10)
    img_rec = imgAcc / w_sum

    # Back-projection
    img_rec = sr_backprojection(
        img_rec, img_pyr_h[opt['origResLvl']], opt['bpKernelSigma'], opt['nIterBP']
    )
    return img_rec


def _sr_voting_fast(img_pyr_h, NNF, opt):
    """
    Optimised voting using vectorised scatter-add.
    """
    imgH = NNF['imgH']
    imgW = NNF['imgW']
    pSize = opt['pSize']

    src_patch, _ = sr_prep_source_patch(img_pyr_h, NNF['uvTformH']['data'], opt)

    if opt['useBiasCorrection']:
        src_patch = src_patch + NNF['uvBias']['data']

    w = NNF['wPatch']
    src_patch = src_patch * w[:, None, None]

    imgAcc = np.zeros((imgH, imgW, 3), dtype=np.float32)
    out_H = imgH - pSize + 1
    out_W = imgW - pSize + 1

    # Build index array: for each patch pixel (pi), for each patch (n),
    # compute the target pixel in the image
    patch_rows = np.arange(pSize)
    patch_cols = np.arange(pSize)
    n_rows = np.arange(out_H)
    n_cols = np.arange(out_W)

    for pi in range(pSize * pSize):
        pr = pi // pSize
        pc_ = pi % pSize
        # Target rows/cols for all patches
        rows = n_rows + pr  # (out_H,)
        cols = n_cols + pc_  # (out_W,)
        # Create grid of (row, col) for all patches
        R, C = np.meshgrid(rows, cols, indexing='ij')  # (out_H, out_W)
        for c in range(3):
            vals = src_patch[pi, c, :].reshape(out_H, out_W)
            imgAcc[R, C, c] += vals

    w_sum = NNF['wSumImg'][:, :, None]
    w_sum = np.maximum(w_sum, 1e-10)
    img_rec = imgAcc / w_sum

    img_rec = sr_backprojection(
        img_rec, img_pyr_h[opt['origResLvl']], opt['bpKernelSigma'], opt['nIterBP']
    )
    return img_rec


# ═════════════════════════════════════════════════════════════════════════════
# 18. sr_backprojection  (from sr_backprojection.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_backprojection(img_hr, img_lr, sigma, n_iter):
    """
    Iterative back-projection (Irani & Peleg).

    img_hr : (H_h, W_h, C) float32
    img_lr : (H_l, W_l, C) float32
    sigma  : float — Gaussian kernel sigma
    n_iter : int

    Returns corrected img_hr.
    """
    H_l, W_l = img_lr.shape[:2]
    H_h, W_h = img_hr.shape[:2]

    f = fspecial_gaussian(5, sigma)
    f = f ** 2
    f /= f.sum()

    for _ in range(n_iter):
        # Downscale HR → LR size
        lr_sim = _imresize(img_hr, (H_l, W_l))
        # Difference
        diff_lr = img_lr - lr_sim
        # Upscale diff to HR size
        diff_hr = _imresize(diff_lr, (H_h, W_h))
        # Apply Gaussian filter per channel
        if diff_hr.ndim == 3:
            for c in range(diff_hr.shape[2]):
                diff_hr[:, :, c] = cv2.filter2D(diff_hr[:, :, c], -1, f,
                                                  borderType=cv2.BORDER_REPLICATE)
        else:
            diff_hr = cv2.filter2D(diff_hr, -1, f, borderType=cv2.BORDER_REPLICATE)

        img_hr = img_hr + diff_hr
        img_hr = clamp(img_hr, 0.0, 1.0)

    return img_hr


# ═════════════════════════════════════════════════════════════════════════════
# 19. sr_synthesis  (from sr_synthesis.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_synthesis(img_pyr_h, img_pyr_l, scale_pyr, model_plane_pyr, opt):
    """
    Coarse-to-fine super-resolution synthesis.

    Returns updated img_pyr_h.
    """
    orig_lvl = opt['origResLvl']
    top_level = opt['topLevel']

    pyr_levels = list(range(orig_lvl - 1, top_level - 1, -1))
    num_iter_lvl = opt['numIter']
    NNF = None

    for lvl in pyr_levels:
        model_cur = model_plane_pyr[lvl]
        img_size_cur = scale_pyr[lvl]['imgSize']
        opt['iLvl'] = lvl

        print(f"--- Initialise NNF at level {lvl}")
        NNF = sr_init_lvl_nnf(img_size_cur, NNF, model_cur, opt)

        num_iter_lvl = max(num_iter_lvl - opt['numIterDec'], opt['numIterMin'])

        print(f"--- Pass... level: {lvl}, #Iter: {num_iter_lvl}, "
              f"#uvPixels: {NNF['uvPix']['numUvPix']:7d}")
        print(f"--- {'iter':>3s}\t{'#PropUpdate':>12s}\t{'#RandUpdate':>12s}\t{'AvgCost':>10s}")

        # Target: upscale from level+1
        img_trg = _imresize(img_pyr_h[lvl + 1], img_size_cur)
        NNF = sr_pass(img_trg, img_pyr_l, NNF, model_cur, num_iter_lvl, opt)

        # Reconstruct
        img_pyr_h[lvl] = _sr_voting_fast(img_pyr_h, NNF, opt)

        # Update low-pass pyramid
        next_size = scale_pyr[lvl + 1]['imgSize'] if scale_pyr[lvl + 1] is not None else img_size_cur
        down = _imresize(img_pyr_h[lvl], next_size)
        img_pyr_l[lvl] = _imresize(down, img_size_cur)

    # Final voting at top level
    img_pyr_h[top_level] = _sr_voting_fast(img_pyr_h, NNF, opt)

    return img_pyr_h


# ═════════════════════════════════════════════════════════════════════════════
# 20. sr_demo  (from sr_demo.m)
# ═════════════════════════════════════════════════════════════════════════════

def sr_demo(img, SRF, opt=None):
    """
    Main SelfExSR pipeline.

    Parameters
    ----------
    img : (H, W) or (H, W, 3) — input LR image, uint8 or float [0,1]
    SRF : int — super-resolution factor
    opt : dict or None — parameters (default: sr_init_opt(SRF))

    Returns
    -------
    img_hr : (H*SRF, W*SRF, 3) float32 in [0, 1]
    """
    if opt is None:
        opt = sr_init_opt(SRF)

    # Normalise to float32 [0, 1]
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # --- Phase 1: Plane extraction (simplified) ---
    print("- Extract planar structures")
    model_plane = sr_extract_plane_simple(img.shape, opt)

    # --- Phase 2: Image pyramid ---
    print("- Construct image pyramid")
    img_pyr_h, img_pyr_l, scale_pyr = sr_create_img_pyramid(img, opt)

    print("- Construct plane pyramid")
    model_plane_pyr = sr_planar_structure_pyramid(scale_pyr, model_plane, opt['topLevel'])

    # --- Phase 3: Synthesis ---
    print("- Single Image Super-Resolution using Transformed Self-Exemplars")
    img_pyr_h = sr_synthesis(img_pyr_h, img_pyr_l, scale_pyr, model_plane_pyr, opt)

    # Get the desired level
    if SRF % 2 == 0:
        lvl_ind = opt['origResLvl'] - int(
            (np.log(SRF) / np.log(2)) * opt['nLvlToRedRes']
        )
    else:
        lvl_ind = opt['origResLvl'] - int(
            (np.log(SRF) / np.log(3)) * opt['nLvlToRedRes']
        )

    return img_pyr_h[lvl_ind]
