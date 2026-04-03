"""
solvers.py

Core solver functions for MAP (Maximum A Posteriori) blind deconvolution.

Ported from MATLAB code by Oliver Whyte et al.
References:
    O. Whyte, J. Sivic, A. Zisserman, and J. Ponce.
    "Non-uniform Deblurring for Shaken Images". IJCV, 2012.

    O. Whyte, J. Sivic and A. Zisserman.
    "Deblurring Shaken and Partially Saturated Images".
    In Proc. CPCV Workshop at ICCV, 2011.

    D. Krishnan, R. Fergus.
    "Fast Image Deconvolution using Hyper-Laplacian Priors". NIPS, 2009.

Contains:
    lars                — LARS-LASSO for kernel estimation (lars.m)
    BtB_uni             — Gram matrix & cross-corr vector for uniform kernel
                          estimation (BtB_uni.m)
    deconv_L2_w         — L2 CG deconvolution with derivative weights
                          (deconvL2NonUni_w.m, uniform path)
    deconv_L2_grad_data — L2 CG deconvolution with gradient data term
                          (deconvL2NonUni_w_gradData.m, uniform path)
    deconv_sparse       — Sparse IRLS deconvolution
                          (deconvSpsNonUni.m, uniform path)
    fast_deconv         — Krishnan & Fergus hyper-Laplacian deconvolution
                          (fast_deconv.m)
    blind_deblur_map    — Main MAP blind deconvolution pipeline
                          (blind_deblur_map.m, uniform path)

MATLAB -> Python notes:
    imfilter(I, h, 'same', 'conv') -> scipy.ndimage.convolve(I, h, mode='nearest')
    imfilter(I, h, 'same', 'corr') -> scipy.ndimage.correlate(I, h, mode='nearest')
    conv2(A, B, 'same'/'valid'/'full') -> scipy.signal.convolve2d(A, B, mode=...)
    fliplr(flipud(h)) = rot180(h) = h[::-1, ::-1]
    imerode(mask, ones(k)) -> scipy.ndimage.binary_erosion(mask, structure=np.ones((k,k)))
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import (
    convolve as ndimage_convolve,
    correlate as ndimage_correlate,
    binary_erosion,
)

from .utils import (
    psf2otf,
    otf2psf,
    pad_image,
    calculate_padding,
    solve_image,
    reset_solve_image_cache,
    jcb_filter,
    shock_filter,
    poisson_blend_fft,
    imfilter_conv_replicate,
    colour_conv2,
    imresize,
    make_kernel_pyramid,
    upsample_kernel_map,
    htranslate,
    hscale,
    get_derivative_filters,
    default_config,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: per-channel conv2 (like MATLAB colourconv2)
# ═══════════════════════════════════════════════════════════════════════════

def _colour_conv2(im: np.ndarray, h: np.ndarray, mode: str) -> np.ndarray:
    """Apply scipy convolve2d per channel. Same as colour_conv2 in utils."""
    return colour_conv2(im, h, mode)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: apply D'*D operator (data-fidelity Hessian)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_DtD(im: np.ndarray, omega: np.ndarray,
               dxf: np.ndarray, dyf: np.ndarray,
               dxxf: np.ndarray, dyyf: np.ndarray,
               dxyf: np.ndarray) -> np.ndarray:
    """
    Apply the gradient data-term operator:
        omega[0]*I + omega[1]*(Dx'Dx + Dy'Dy) + omega[2]*(Dxx'Dxx + Dyy'Dyy + Dxy'Dxy)

    where D'D = conv2(conv2(im, h, 'valid'), rot180(h), 'full').

    MATLAB fliplr(flipud(h)) = h[::-1, ::-1].
    """
    dxf_t = dxf[::-1, ::-1]
    dyf_t = dyf[::-1, ::-1]
    dxxf_t = dxxf[::-1, ::-1]
    dyyf_t = dyyf[::-1, ::-1]
    dxyf_t = dxyf[::-1, ::-1]

    result = omega[0] * im
    result = result + omega[1] * _colour_conv2(
        _colour_conv2(im, dxf, 'valid'), dxf_t, 'full')
    result = result + omega[1] * _colour_conv2(
        _colour_conv2(im, dyf, 'valid'), dyf_t, 'full')
    result = result + omega[2] * _colour_conv2(
        _colour_conv2(im, dxxf, 'valid'), dxxf_t, 'full')
    result = result + omega[2] * _colour_conv2(
        _colour_conv2(im, dyyf, 'valid'), dyyf_t, 'full')
    result = result + omega[2] * _colour_conv2(
        _colour_conv2(im, dxyf, 'valid'), dxyf_t, 'full')
    return result


def _apply_weighted_reg(im: np.ndarray,
                        weight_x: np.ndarray, weight_y: np.ndarray,
                        weight_xx: np.ndarray, weight_yy: np.ndarray,
                        weight_xy: np.ndarray,
                        dxf: np.ndarray, dyf: np.ndarray,
                        dxxf: np.ndarray, dyyf: np.ndarray,
                        dxyf: np.ndarray) -> np.ndarray:
    """
    Apply spatially weighted regularisation operator:
        sum_d conv2(weight_d .* conv2(im, rot180(d), 'valid'), d, 'full')

    MATLAB: we * colourconv2(weight_x .* colourconv2(x, rot180(dxf), 'valid'), dxf, 'full')
    """
    dxf_t = dxf[::-1, ::-1]
    dyf_t = dyf[::-1, ::-1]
    dxxf_t = dxxf[::-1, ::-1]
    dyyf_t = dyyf[::-1, ::-1]
    dxyf_t = dxyf[::-1, ::-1]

    result = _colour_conv2(
        weight_x * _colour_conv2(im, dxf_t, 'valid'), dxf, 'full')
    result = result + _colour_conv2(
        weight_y * _colour_conv2(im, dyf_t, 'valid'), dyf, 'full')
    result = result + _colour_conv2(
        weight_xx * _colour_conv2(im, dxxf_t, 'valid'), dxxf, 'full')
    result = result + _colour_conv2(
        weight_yy * _colour_conv2(im, dyyf_t, 'valid'), dyyf, 'full')
    result = result + _colour_conv2(
        weight_xy * _colour_conv2(im, dxyf_t, 'valid'), dxyf, 'full')
    return result


# ═══════════════════════════════════════════════════════════════════════════
# LARS-LASSO  (from lars.m)
# ═══════════════════════════════════════════════════════════════════════════

def lars(Gram: np.ndarray,
         Xty: np.ndarray,
         nonneg: int = 1,
         stop: float = 0,
         use_gram: int = 1,
         precompute_gram: int = 1,
         trace: int = 0,
         mode: int = 2,
         max_active: int = None) -> tuple:
    """
    LARS-LASSO algorithm for kernel estimation.
    Equivalent to MATLAB lars.m (Karl Skoglund / Oliver Whyte).

    Operates on precomputed Gram matrix and X'y vector.

    Parameters
    ----------
    Gram : (p, p) precomputed X'X Gram matrix
    Xty : (p,) precomputed X'y vector
    nonneg : 1 for non-negative LASSO (kernel weights >= 0)
    stop : regularization parameter (depends on mode)
    use_gram : 1 to use precomputed Gram (always 1 here)
    precompute_gram : 1 if Gram is precomputed (always 1 here)
    trace : 0 = silent
    mode : 0 = L1 constraint, 2 = L1 penalty (lambda*||w||_1)
    max_active : maximum number of active variables

    Returns
    -------
    beta : (p,) final coefficient vector
    beta_path : (k, p) full regularization path
    """
    if max_active is None:
        max_active = np.inf

    p = Gram.shape[0]
    nvars = p

    max_k = 8 * nvars

    if stop == 0:
        beta = np.zeros((2 * nvars, p), dtype=np.float64)
    elif stop < 0:
        beta = np.zeros((2 * int(round(-stop)), p), dtype=np.float64)
    else:
        beta = np.zeros((100, p), dtype=np.float64)

    mu = np.zeros(p, dtype=np.float64)  # Xt_mu
    Xt_mu = np.zeros(p, dtype=np.float64)
    I_set = list(range(p))  # inactive set (indices)
    A_set = []              # active set (indices)

    lasso_cond = False
    stop_cond = False
    k = 0
    n_vars = 0
    lam = np.zeros(p + 1, dtype=np.float64)

    while n_vars <= nvars and not stop_cond and k < max_k:
        # Compute correlations
        c = Xty - Xt_mu

        if len(I_set) == 0:
            C = 0.0
            j_idx = 0
        else:
            if nonneg:
                c_I = c[I_set]
                j_idx = int(np.argmax(c_I))
                C = c_I[j_idx]
            else:
                c_I = np.abs(c[I_set])
                j_idx = int(np.argmax(c_I))
                C = c_I[j_idx]

        lam[k] = max(C, 0.0)

        # Early stopping at specified regularization weight (mode=2)
        if mode == 2 and stop >= lam[k]:
            if k == 0:
                beta[k, :] = 0.0
                break
            t1 = lam[k]
            t2 = lam[k - 1]
            if abs(t2 - t1) < 1e-30:
                s = 0.0
            else:
                s = (stop - t1) / (t2 - t1)
            beta[k, :] = beta[k, :] + s * (beta[k - 1, :] - beta[k, :])
            break

        if n_vars == nvars or (len(I_set) > 0 and np.min(np.abs(c)) <= 1e-11):
            break

        k += 1
        j = I_set[j_idx]

        if not lasso_cond:
            A_set.append(j)
            I_set.remove(j)
            n_vars += 1

        # Signs of correlations
        if nonneg:
            s_vec = np.ones(len(A_set), dtype=np.float64)
        else:
            s_vec = np.sign(c[A_set])

        # Compute equiangular direction
        A_arr = np.array(A_set, dtype=int)
        Gram_AA = Gram[np.ix_(A_arr, A_arr)]

        if nonneg:
            try:
                GA1 = np.linalg.solve(Gram_AA, np.ones(n_vars))
            except np.linalg.LinAlgError:
                GA1 = np.linalg.lstsq(Gram_AA, np.ones(n_vars), rcond=None)[0]
        else:
            S_mat = np.outer(s_vec, s_vec)
            try:
                GA1 = np.linalg.solve(Gram_AA * S_mat, np.ones(n_vars))
            except np.linalg.LinAlgError:
                GA1 = np.linalg.lstsq(Gram_AA * S_mat, np.ones(n_vars),
                                       rcond=None)[0]

        AA = 1.0 / np.sqrt(np.sum(GA1))
        sw = AA * GA1 * s_vec

        d_vec = np.zeros(p, dtype=np.float64)
        d_vec[A_arr] = sw

        # Xt_u = Gram[:, A_set] @ sw
        Xt_u = Gram[:, A_arr] @ sw

        # Compute step size gamma
        if n_vars == nvars:
            gamma = C / AA
        else:
            a_vec = Xt_u
            I_arr = np.array(I_set, dtype=int)
            if nonneg:
                temp = (C - c[I_arr]) / (AA - a_vec[I_arr] + 1e-30)
            else:
                temp1 = (C - c[I_arr]) / (AA - a_vec[I_arr] + 1e-30)
                temp2 = (C + c[I_arr]) / (AA + a_vec[I_arr] + 1e-30)
                temp = np.concatenate([temp1, temp2])
            pos_temp = temp[temp > 0]
            if len(pos_temp) > 0:
                gamma = min(np.min(pos_temp), C / AA)
            else:
                gamma = C / AA

        # LASSO modification
        lasso_cond = False
        beta_A = beta[k - 1, A_arr] if k > 0 else np.zeros(len(A_arr))
        d_A = d_vec[A_arr]
        temp_lasso = np.full(len(A_arr), np.inf)
        mask_neg = d_A != 0
        temp_lasso[mask_neg] = -beta_A[mask_neg] / d_A[mask_neg]

        pos_mask = temp_lasso > 0
        if np.any(pos_mask):
            gamma_tilde = np.min(temp_lasso[pos_mask])
            j_lasso = int(np.argmin(temp_lasso[pos_mask]))
            # Map back to index in the positive subset
            pos_indices = np.where(pos_mask)[0]
            j_lasso = pos_indices[j_lasso]
        else:
            gamma_tilde = gamma
            j_lasso = 0

        if gamma_tilde < gamma:
            gamma = gamma_tilde
            lasso_cond = True

        # Update
        Xt_mu = Xt_mu + gamma * Xt_u

        # Ensure beta has enough rows
        if beta.shape[0] <= k:
            beta = np.vstack([beta, np.zeros_like(beta)])

        if k > 0:
            beta[k, :] = beta[k - 1, :]
        beta[k, A_arr] = beta[k, A_arr] + gamma * d_A

        # Early stopping at specified bound on L1 norm (mode=0)
        if mode == 0 and stop > 0:
            t2 = np.sum(np.abs(beta[k, :]))
            if t2 >= stop:
                t1 = np.sum(np.abs(beta[k - 1, :])) if k > 0 else 0.0
                if abs(t2 - t1) < 1e-30:
                    s_interp = 0.0
                else:
                    s_interp = (stop - t1) / (t2 - t1)
                prev = beta[k - 1, :] if k > 0 else np.zeros(p)
                beta[k, :] = prev + s_interp * (beta[k, :] - prev)
                stop_cond = True

        # LASSO: drop variable
        if lasso_cond:
            drop_idx = j_lasso
            dropped_var = A_set[drop_idx]
            I_set.append(dropped_var)
            A_set.pop(drop_idx)
            n_vars -= 1

        # Early stopping on number of variables
        if stop < 0:
            stop_cond = np.count_nonzero(beta[k, :]) >= -stop
        if np.isfinite(max_active) and max_active > 0:
            stop_cond = np.count_nonzero(beta[k, :]) >= max_active

    # Trim
    beta_path = beta[:k + 1, :]
    beta_final = beta_path[-1, :] if beta_path.shape[0] > 0 else np.zeros(p)

    return beta_final, beta_path


# ═══════════════════════════════════════════════════════════════════════════
# BtB_uni — Gram matrix for uniform kernel estimation
# ═══════════════════════════════════════════════════════════════════════════

def BtB_uni(Pall: np.ndarray,
            Ball: np.ndarray,
            kernel_shape: tuple,
            mask: np.ndarray = None) -> tuple:
    """
    Compute the Gram matrix B'B and the cross-correlation vector B'g
    for uniform blur kernel estimation.

    For uniform blur, B*w = conv2(P, w) and the kernel estimation
    solves:  min ||B*w - g||^2 + beta*||w||^2

    where P = predicted sharp gradients and g = observed blurry gradients,
    and w is the blur kernel vectorised.

    Parameters
    ----------
    Pall : (H, W, D) predicted gradient images (D derivative channels)
    Ball : (H, W, D) observed blurry gradient images
    kernel_shape : (kh, kw) shape of the kernel being estimated
    mask : (H, W) observation mask (optional)

    Returns
    -------
    BtB : (kh*kw, kh*kw) Gram matrix
    Btg : (kh*kw,) cross-correlation vector
    """
    kh, kw = kernel_shape
    nk = kh * kw

    if Pall.ndim == 2:
        Pall = Pall[:, :, np.newaxis]
    if Ball.ndim == 2:
        Ball = Ball[:, :, np.newaxis]

    H, W, D = Pall.shape
    pad = 2 * calculate_padding((H, W), np.zeros((kh, kw)))
    Pall_pad = pad_image(Pall, pad, 0)
    Ball_pad = pad_image(Ball, pad, 0)

    if mask is not None:
        if mask.ndim == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], D, axis=2)
        else:
            mask_3d = mask
        mask_pad = pad_image(mask_3d, pad, 0)
        Pall_pad = Pall_pad * mask_pad
        Ball_pad = Ball_pad * mask_pad

    Hp, Wp = Pall_pad.shape[:2]

    # Compute BtB via FFT auto-correlation
    # BtB[i,j] = sum_d sum_{x,y} P_d(x-di, y-dj) * P_d(x-dj, y-dj) ... etc
    # In Fourier domain: autocorr(shift) = ifft2(|fft2(P)|^2)

    # Compute power spectrum summed over all derivative channels
    P_fft = fft2(Pall_pad, axes=(0, 1))
    P2_fft = np.sum(np.conj(P_fft) * P_fft, axis=2)  # (Hp, Wp) complex
    autocorr = np.real(ifft2(P2_fft))  # (Hp, Wp)

    # Build BtB from auto-correlation at kernel offsets
    # For kernel element at position (r1,c1) and (r2,c2), the
    # Gram entry is autocorr[r1-r2, c1-c2] (with wrapping).
    BtB = np.zeros((nk, nk), dtype=np.float64)
    for i in range(nk):
        ri, ci = divmod(i, kw)
        for j in range(i, nk):
            rj, cj = divmod(j, kw)
            dr = ri - rj
            dc = ci - cj
            # Wrap negative indices
            dr_w = dr % Hp
            dc_w = dc % Wp
            val = autocorr[dr_w, dc_w]
            BtB[i, j] = val
            BtB[j, i] = val

    # Compute Btg = B' * g (cross-correlation between P and B)
    # Btg[i] = sum_d sum_{x,y} P_d(x-di, y-di) * B_d(x, y)
    B_fft = fft2(Ball_pad, axes=(0, 1))
    PB_fft = np.sum(np.conj(P_fft) * B_fft, axis=2)
    crosscorr = np.real(ifft2(PB_fft))

    Btg = np.zeros(nk, dtype=np.float64)
    for i in range(nk):
        ri, ci = divmod(i, kw)
        # The cross-correlation at the kernel offset
        Btg[i] = crosscorr[ri, ci]

    return BtB, Btg


# ═══════════════════════════════════════════════════════════════════════════
# L2 CG deconvolution with derivative weights
# (from deconvL2NonUni_w.m, uniform path only)
# ═══════════════════════════════════════════════════════════════════════════

def deconv_L2_w(imblur: np.ndarray,
                kernel: np.ndarray,
                we: float,
                max_it: int = 200,
                weight_x: np.ndarray = None,
                weight_y: np.ndarray = None,
                weight_xx: np.ndarray = None,
                weight_yy: np.ndarray = None,
                weight_xy: np.ndarray = None,
                saturation_mask: np.ndarray = None,
                iminit: np.ndarray = None) -> tuple:
    """
    L2 conjugate gradient deconvolution with spatially varying derivative weights.
    Equivalent to MATLAB deconvL2NonUni_w.m (uniform path, no gradient data term).

    Solves:
        min ||A*x - b||^2 + we * sum_d w_d * ||D_d * x||^2

    via CG, where A is the blur operator and D_d are derivative operators.

    Parameters
    ----------
    imblur : (H, W) or (H, W, C) blurry image
    kernel : 2D blur kernel
    we : regularization weight (alpha)
    max_it : max CG iterations
    weight_x,...,weight_xy : spatially varying weights for each derivative
    saturation_mask : (H, W) or (H, W, C) binary mask of reliable pixels
    iminit : initial estimate (default: imblur)

    Returns
    -------
    x : deconvolved image (padded)
    pad : padding applied [top, bottom, left, right]
    """
    if imblur.ndim == 2:
        imblur = imblur[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n, m, channels = imblur.shape

    if saturation_mask is None:
        saturation_mask = np.ones((n, m, channels), dtype=np.float64)
    elif saturation_mask.ndim == 2:
        saturation_mask = np.repeat(saturation_mask[:, :, np.newaxis],
                                    channels, axis=2)

    if iminit is None:
        iminit = imblur.copy()
    elif iminit.ndim == 2:
        iminit = iminit[:, :, np.newaxis]

    pad = calculate_padding((n, m), kernel)

    # Pad images
    imblur = pad_image(imblur, pad, 'edge')
    iminit = pad_image(iminit, pad, 'edge')

    n_p = n + pad[0] + pad[1]
    m_p = m + pad[2] + pad[3]

    # Blur functions (uniform)
    def blurfn(im):
        return imfilter_conv_replicate(im, kernel)

    def conjfn(im):
        # imfilter with 'corr' = correlate
        if im.ndim == 3:
            res = np.empty_like(im)
            for c in range(im.shape[2]):
                res[:, :, c] = ndimage_correlate(
                    im[:, :, c], kernel, mode='nearest')
            return res
        return ndimage_correlate(im, kernel, mode='nearest')

    # Build mask
    mask = np.zeros((n_p, m_p, channels), dtype=np.float64)
    mask[pad[0]:n_p - pad[1], pad[2]:m_p - pad[3], :] = saturation_mask
    # Erode mask
    struct = np.ones((5, 5), dtype=bool)
    for c in range(channels):
        mask[:, :, c] = binary_erosion(mask[:, :, c] > 0.5,
                                       structure=struct).astype(np.float64)
    mask[0, :, :] = 0
    mask[-1, :, :] = 0
    mask[:, 0, :] = 0
    mask[:, -1, :] = 0

    # Derivative filters
    dxf = np.array([[1, -1]], dtype=np.float64)
    dyf = np.array([[1], [-1]], dtype=np.float64)
    dxxf = np.array([[-1, 2, -1]], dtype=np.float64)
    dyyf = np.array([[-1], [2], [-1]], dtype=np.float64)
    dxyf = np.array([[-1, 1], [1, -1]], dtype=np.float64)

    # Default weights: ones
    if weight_x is None:
        weight_x = np.ones((n_p, m_p - 1, channels))
    if weight_y is None:
        weight_y = np.ones((n_p - 1, m_p, channels))
    if weight_xx is None:
        weight_xx = np.ones((n_p, m_p - 2, channels))
    if weight_yy is None:
        weight_yy = np.ones((n_p - 2, m_p, channels))
    if weight_xy is None:
        weight_xy = np.ones((n_p - 1, m_p - 1, channels))

    x = iminit.copy()

    # Initial residual
    ax = blurfn(x)
    DtBall = imblur  # No gradient data term in this version
    DtDax = ax

    r = conjfn(DtBall * mask - DtDax * mask)
    r = r - we * _apply_weighted_reg(x, weight_x, weight_y,
                                     weight_xx, weight_yy, weight_xy,
                                     dxf, dyf, dxxf, dyyf, dxyf)

    d = r.copy()
    rho = np.sum(r ** 2, axis=(0, 1))  # per-channel
    if np.all(rho < np.finfo(float).eps):
        if squeeze:
            return x[:, :, 0], pad
        return x, pad

    for it in range(max_it):
        Ad = blurfn(d)
        DtDAd = Ad
        AtDtDAd = conjfn(DtDAd * mask)
        AtDtDAd = AtDtDAd + we * _apply_weighted_reg(
            d, weight_x, weight_y, weight_xx, weight_yy, weight_xy,
            dxf, dyf, dxxf, dyyf, dxyf)

        step_length = rho / (np.sum(d * AtDtDAd, axis=(0, 1)) + 1e-30)

        for c in range(channels):
            x[:, :, c] += step_length[c] * d[:, :, c]
            r[:, :, c] -= step_length[c] * AtDtDAd[:, :, c]

        rho_old = rho.copy()
        rho = np.sum(r ** 2, axis=(0, 1))
        beta_cg = rho / (rho_old + 1e-30)

        for c in range(channels):
            d[:, :, c] = r[:, :, c] + beta_cg[c] * d[:, :, c]

    if squeeze:
        return x[:, :, 0], pad
    return x, pad


# ═══════════════════════════════════════════════════════════════════════════
# L2 CG deconvolution with gradient data term
# (from deconvL2NonUni_w_gradData.m, uniform path)
# ═══════════════════════════════════════════════════════════════════════════

def deconv_L2_grad_data(imblur: np.ndarray,
                        kernel: np.ndarray,
                        we: float,
                        max_it: int = 200,
                        weight_x: float = 1,
                        weight_y: float = 1,
                        weight_xx: float = 0,
                        weight_yy: float = 0,
                        weight_xy: float = 0,
                        saturation_mask: np.ndarray = None,
                        iminit: np.ndarray = None,
                        omega: np.ndarray = None) -> tuple:
    """
    L2 CG deconvolution with Shan et al. gradient data term.
    Equivalent to MATLAB deconvL2NonUni_w_gradData.m (uniform path).

    Uses omega[0]*||Ax-b||^2 + omega[1]*||DxAx-Dxb||^2 + ...
    (gradient fidelity from Shan et al. 2008).

    Parameters
    ----------
    imblur : (H, W) or (H, W, C)
    kernel : 2D kernel
    we : regularization weight
    max_it : max CG iterations
    weight_x/y/xx/yy/xy : scalar regularization weights (used as multipliers)
    saturation_mask : binary mask
    iminit : initial estimate
    omega : [omega0, omega1, omega2] gradient weights (default [1, 0.5, 0.25])

    Returns
    -------
    x : deconvolved image (padded)
    pad : padding applied
    """
    if omega is None:
        omega = np.array([1.0, 0.5, 0.25])
    else:
        omega = np.asarray(omega)

    if imblur.ndim == 2:
        imblur = imblur[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    n, m, channels = imblur.shape

    if saturation_mask is None:
        saturation_mask = np.ones((n, m, channels), dtype=np.float64)
    elif saturation_mask.ndim == 2:
        saturation_mask = np.repeat(saturation_mask[:, :, np.newaxis],
                                    channels, axis=2)

    if iminit is None:
        iminit = imblur.copy()
    elif iminit.ndim == 2:
        iminit = iminit[:, :, np.newaxis]

    pad = calculate_padding((n, m), kernel)
    imblur = pad_image(imblur, pad, 'edge')
    iminit = pad_image(iminit, pad, 'edge')

    n_p = n + pad[0] + pad[1]
    m_p = m + pad[2] + pad[3]

    def blurfn(im):
        return imfilter_conv_replicate(im, kernel)

    def conjfn(im):
        if im.ndim == 3:
            res = np.empty_like(im)
            for c in range(im.shape[2]):
                res[:, :, c] = ndimage_correlate(
                    im[:, :, c], kernel, mode='nearest')
            return res
        return ndimage_correlate(im, kernel, mode='nearest')

    # Build mask
    mask = np.zeros((n_p, m_p, channels), dtype=np.float64)
    mask[pad[0]:n_p - pad[1], pad[2]:m_p - pad[3], :] = saturation_mask
    struct = np.ones((5, 5), dtype=bool)
    for c in range(channels):
        mask[:, :, c] = binary_erosion(mask[:, :, c] > 0.5,
                                       structure=struct).astype(np.float64)
    mask[0, :, :] = 0
    mask[-1, :, :] = 0
    mask[:, 0, :] = 0
    mask[:, -1, :] = 0

    x = iminit.copy()

    # Derivative filters
    dxf = np.array([[1, -1]], dtype=np.float64)
    dyf = np.array([[1], [-1]], dtype=np.float64)
    dxxf = np.array([[-1, 2, -1]], dtype=np.float64)
    dyyf = np.array([[-1], [2], [-1]], dtype=np.float64)
    dxyf = np.array([[-1, 1], [1, -1]], dtype=np.float64)

    # Scalar regularization weights -> uniform spatial weights
    w_x = weight_x * np.ones((n_p, m_p - 1, channels))
    w_y = weight_y * np.ones((n_p - 1, m_p, channels))
    w_xx = weight_xx * np.ones((n_p, m_p - 2, channels))
    w_yy = weight_yy * np.ones((n_p - 2, m_p, channels))
    w_xy = weight_xy * np.ones((n_p - 1, m_p - 1, channels))

    # D'*B (gradient data term applied to blurry image)
    DtBall = _apply_DtD(imblur, omega, dxf, dyf, dxxf, dyyf, dxyf)

    # Initial A*x and D'D*A*x
    ax = blurfn(x)
    DtDax = _apply_DtD(ax, omega, dxf, dyf, dxxf, dyyf, dxyf)

    # Initial residual
    r = conjfn(DtBall * mask - DtDax * mask)
    r = r - we * _apply_weighted_reg(x, w_x, w_y, w_xx, w_yy, w_xy,
                                     dxf, dyf, dxxf, dyyf, dxyf)

    d = r.copy()
    rho = np.sum(r ** 2, axis=(0, 1))
    if np.all(rho < np.finfo(float).eps):
        if squeeze:
            return x[:, :, 0], pad
        return x, pad

    for it in range(max_it):
        Ad = blurfn(d)
        DtDAd = _apply_DtD(Ad, omega, dxf, dyf, dxxf, dyyf, dxyf)
        AtDtDAd = conjfn(DtDAd * mask)
        AtDtDAd = AtDtDAd + we * _apply_weighted_reg(
            d, w_x, w_y, w_xx, w_yy, w_xy,
            dxf, dyf, dxxf, dyyf, dxyf)

        step_length = rho / (np.sum(d * AtDtDAd, axis=(0, 1)) + 1e-30)
        for c in range(channels):
            x[:, :, c] += step_length[c] * d[:, :, c]
            r[:, :, c] -= step_length[c] * AtDtDAd[:, :, c]

        rho_old = rho.copy()
        rho = np.sum(r ** 2, axis=(0, 1))
        beta_cg = rho / (rho_old + 1e-30)
        for c in range(channels):
            d[:, :, c] = r[:, :, c] + beta_cg[c] * d[:, :, c]

    if squeeze:
        return x[:, :, 0], pad
    return x, pad


# ═══════════════════════════════════════════════════════════════════════════
# Sparse IRLS deconvolution  (from deconvSpsNonUni.m, uniform path)
# ═══════════════════════════════════════════════════════════════════════════

def deconv_sparse(imblur: np.ndarray,
                  kernel: np.ndarray,
                  we: float,
                  max_it: int = 200,
                  saturation_mask: np.ndarray = None) -> np.ndarray:
    """
    Sparse (IRLS) deconvolution.
    Equivalent to MATLAB deconvSpsNonUni.m (uniform path).

    Iteratively reweighted least squares: run L2 CG deconvolution,
    recompute derivative weights based on current estimate (2 outer iters).

    Parameters
    ----------
    imblur : (H, W) or (H, W, C)
    kernel : 2D kernel
    we : regularization weight
    max_it : max CG iterations per inner loop
    saturation_mask : binary mask

    Returns
    -------
    x : deconvolved image
    """
    if imblur.ndim == 2:
        imblur_3d = imblur[:, :, np.newaxis]
    else:
        imblur_3d = imblur

    n, m, channels = imblur_3d.shape
    pad = calculate_padding((n, m), kernel)
    n_p = n + pad[0] + pad[1]
    m_p = m + pad[2] + pad[3]

    # Initial weights: all ones
    weight_x = np.ones((n_p, m_p - 1, channels))
    weight_y = np.ones((n_p - 1, m_p, channels))
    weight_xx = np.ones((n_p, m_p - 2, channels))
    weight_yy = np.ones((n_p - 2, m_p, channels))
    weight_xy = np.ones((n_p - 1, m_p - 1, channels))

    # First L2 deconvolution
    x, _ = deconv_L2_w(imblur, kernel, we, max_it,
                        weight_x, weight_y, weight_xx, weight_yy, weight_xy,
                        saturation_mask)

    # Derivative filters (for computing rot180 'd' and conv2(...,'valid'))
    dxf = np.array([[1, -1]], dtype=np.float64)
    dyf = np.array([[1], [-1]], dtype=np.float64)
    dxxf = np.array([[-1, 2, -1]], dtype=np.float64)
    dyyf = np.array([[-1], [2], [-1]], dtype=np.float64)
    dxyf = np.array([[-1, 1], [1, -1]], dtype=np.float64)

    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01

    for t in range(2):
        # Compute derivatives using rot180 of filter (fliplr(flipud(...)))
        dx = _colour_conv2(x, dxf[::-1, ::-1], 'valid')
        dy = _colour_conv2(x, dyf[::-1, ::-1], 'valid')
        dxx = _colour_conv2(x, dxxf[::-1, ::-1], 'valid')
        dyy = _colour_conv2(x, dyyf[::-1, ::-1], 'valid')
        dxy = _colour_conv2(x, dxyf[::-1, ::-1], 'valid')

        # Update weights (IRLS)
        weight_x = w0 * np.maximum(np.abs(dx), thr_e) ** (exp_a - 2)
        weight_y = w0 * np.maximum(np.abs(dy), thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        # Re-run deconvolution with updated weights
        x, _ = deconv_L2_w(imblur, kernel, we, max_it,
                            weight_x, weight_y, weight_xx, weight_yy, weight_xy,
                            saturation_mask)

    # Remove padding
    x = pad_image(x, -pad)
    return x


# ═══════════════════════════════════════════════════════════════════════════
# Krishnan & Fergus fast deconvolution  (from fast_deconv.m)
# ═══════════════════════════════════════════════════════════════════════════

def fast_deconv(yin: np.ndarray,
                k: np.ndarray,
                lam: float,
                alpha: float,
                yout0: np.ndarray = None) -> np.ndarray:
    """
    Fast non-blind deconvolution using hyper-Laplacian priors.
    Equivalent to MATLAB fast_deconv.m (Krishnan & Fergus, NIPS 2009).

    Solves:
        min (lambda/2)||k*y - b||^2 + sum_d ||D_d y||^alpha

    via half-quadratic splitting with continuation scheme on beta.

    Parameters
    ----------
    yin : (H, W) or (H, W, C) blurry input
    k : 2D blur kernel (odd-sized)
    lam : data-fidelity weight (lambda)
    alpha : sparsity exponent (e.g. 0.5)
    yout0 : optional initial estimate

    Returns
    -------
    yout : deconvolved image
    """
    # Continuation parameters
    beta = 1.0
    beta_rate = 2 * np.sqrt(2)
    beta_max = 2 ** 8
    mit_inn = 1

    # Padding (4x the kernel-based padding)
    pad_repl = 4 * calculate_padding(yin.shape[:2], k)
    yin = pad_image(yin, pad_repl, 'edge')

    if yin.ndim == 2:
        yin = yin[:, :, np.newaxis]
        squeeze = True
    else:
        squeeze = False

    m_p, n_p, channels = yin.shape

    if yout0 is not None:
        yout = pad_image(yout0, pad_repl, 'edge')
        if yout.ndim == 2:
            yout = yout[:, :, np.newaxis]
    else:
        yout = yin.copy()

    # Kernel must be odd-sized
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("Blur kernel k must be odd-sized.")

    # Precompute frequency-domain quantities
    sizey = (m_p, n_p)
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk)[:, :, np.newaxis] * fft2(yin, axes=(0, 1))
    Denom1 = np.abs(otfk) ** 2
    # Gradient filters in Fourier domain
    Denom2 = (np.abs(psf2otf(np.array([[1, -1]]), sizey)) ** 2 +
              np.abs(psf2otf(np.array([[1], [-1]]), sizey)) ** 2)

    # Circular gradients of initial estimate
    youtx = np.concatenate([np.diff(yout, axis=1),
                            yout[:, 0:1, :] - yout[:, -1:, :]], axis=1)
    youty = np.concatenate([np.diff(yout, axis=0),
                            yout[0:1, :, :] - yout[-1:, :, :]], axis=0)

    while beta < beta_max:
        gamma = beta / lam
        Denom = Denom1 + gamma * Denom2

        for _ in range(mit_inn):
            # w-subproblem
            Wx = solve_image(youtx, beta, alpha)
            Wy = solve_image(youty, beta, alpha)

            # x-subproblem (Fourier domain)
            # Transpose of circular gradients:
            # Wxx = [Wx(:,n) - Wx(:,1), -diff(Wx,1,2)]
            Wxx = np.concatenate(
                [Wx[:, -1:, :] - Wx[:, 0:1, :],
                 -np.diff(Wx, axis=1)], axis=1)
            # + [Wy(m,:) - Wy(1,:); -diff(Wy,1,1)]
            Wxx = Wxx + np.concatenate(
                [Wy[-1:, :, :] - Wy[0:1, :, :],
                 -np.diff(Wy, axis=0)], axis=0)

            Fyout = (Nomin1 + gamma * fft2(Wxx, axes=(0, 1))) / \
                Denom[:, :, np.newaxis]
            yout = np.real(ifft2(Fyout, axes=(0, 1)))

            # Update gradients
            youtx = np.concatenate([np.diff(yout, axis=1),
                                    yout[:, 0:1, :] - yout[:, -1:, :]], axis=1)
            youty = np.concatenate([np.diff(yout, axis=0),
                                    yout[0:1, :, :] - yout[-1:, :, :]], axis=0)

        beta *= beta_rate

    yout = pad_image(yout, -pad_repl)

    if squeeze:
        return yout[:, :, 0]
    return yout


# ═══════════════════════════════════════════════════════════════════════════
# Main MAP blind deconvolution pipeline
# (from blind_deblur_map.m, uniform path only)
# ═══════════════════════════════════════════════════════════════════════════

def blind_deblur_map(im_blurry: np.ndarray,
                     cfg: dict = None,
                     verbose: bool = False) -> tuple:
    """
    MAP blind deconvolution with coarse-to-fine multi-scale estimation.
    Equivalent to MATLAB blind_deblur_map.m (uniform blur path).

    Pipeline (per scale, per iteration):
        1. Edge prediction: bilateral filter -> shock filter -> gradient thresholding
        2. Kernel estimation: LARS-LASSO or CG on gradient images
        3. Non-blind deconvolution: CG, Krishnan & Fergus, or sparse

    Parameters
    ----------
    im_blurry : (H, W) or (H, W, C) blurry image in [0, 1]
    cfg : configuration dict (from default_config or custom)
    verbose : print progress

    Returns
    -------
    L_final : (H, W) or (H, W, C) deblurred image
    k_final : 2D estimated blur kernel
    history : dict with intermediate results
    """
    # Default config
    if cfg is None:
        cfg = default_config()

    # Copy config values
    blur_kernel_size = cfg['blur_kernel_size']
    blur_x_lims = cfg['blur_x_lims']
    blur_y_lims = cfg['blur_y_lims']
    scale_ratio_i = cfg['scale_ratio_i']
    scale_ratio_k = cfg['scale_ratio_k']
    max_levels = cfg['max_levels']
    num_iters_cfg = cfg['num_iters']
    bi_sigma_spatial0 = cfg['bi_sigma_spatial0']
    bi_sigma_range0 = cfg['bi_sigma_range0']
    bi_size = cfg['bi_size']
    shock_dt0 = cfg['shock_dt0']
    shock_iters = cfg['shock_iters']
    param_decrease = cfg['param_decrease']
    grad_dir_bins = cfg['grad_dir_bins']
    grad_dir_quant = np.pi / grad_dir_bins
    grad_thresh_decrease = cfg['grad_thresh_decrease']
    r_factor = cfg['r']
    omega0 = cfg['omega0']
    omega1 = cfg['omega1']
    omega2 = cfg['omega2']
    alpha_reg = cfg['alpha']
    kf_lambda = cfg['kf_lambda']
    kf_exponent = cfg['kf_exponent']
    kernel_threshold = cfg['kernel_threshold']
    beta_ker = cfg['beta']
    recenter_kernel = cfg['recenter_kernel']
    kernel_dilate_radius = cfg['kernel_dilate_radius']
    threshold_kernel = cfg.get('threshold_kernel', True)
    sat_thresh = cfg['sat_thresh']
    kernel_method = cfg['kernel_method']
    image_method = cfg['image_method']
    deconv_maxit = cfg['deconv_maxit']
    do_estimate_kernel = cfg.get('do_estimate_kernel', True)

    # Derivative filters
    filters = get_derivative_filters()
    kx = filters['kx']
    ky = filters['ky']
    kxx = filters['kxx']
    kyy = filters['kyy']
    kxy = filters['kxy']

    first_level = cfg.get('first_level', -1)
    final_level = cfg.get('final_level', 1)

    # Grayscale conversion
    if im_blurry.ndim == 3 and im_blurry.shape[2] == 3:
        im_blurry_grey = np.mean(im_blurry, axis=2)
    else:
        im_blurry_grey = im_blurry.copy()

    im_blurry_colour = im_blurry.copy() if im_blurry.ndim == 3 else \
        im_blurry[:, :, np.newaxis]

    # Saturation mask
    if im_blurry.ndim == 3:
        blurry_unsaturated = np.max(im_blurry, axis=2) < sat_thresh
    else:
        blurry_unsaturated = im_blurry < sat_thresh

    # --- Build kernel pyramid ---
    pyr_kernel, pyr_tt, pyr_tgs = make_kernel_pyramid(
        blur_x_lims, blur_y_lims, scale_ratio_k, max_levels)
    num_levels = len(pyr_kernel)

    # --- Build image pyramid ---
    pyr_blurry = [None] * num_levels
    pyr_blurry[0] = im_blurry_grey.copy()
    pyr_sat_mask = [None] * num_levels
    pyr_sat_mask[0] = blurry_unsaturated.copy()

    for s in range(1, num_levels):
        scale_factor = scale_ratio_i ** s
        ib_s = imresize(pyr_blurry[0], scale_factor, 'bilinear')
        if max(ib_s.shape) < 25:
            num_levels = s
            pyr_blurry = pyr_blurry[:num_levels]
            pyr_sat_mask = pyr_sat_mask[:num_levels]
            pyr_kernel = pyr_kernel[:num_levels]
            pyr_tt = pyr_tt[:num_levels]
            pyr_tgs = pyr_tgs[:num_levels]
            break
        pyr_blurry[s] = ib_s
        mask_uint8 = (blurry_unsaturated.astype(np.float64) * 256).astype(
            np.uint8)
        h_s, w_s = ib_s.shape[:2]
        pyr_sat_mask[s] = imresize(mask_uint8, (h_s, w_s),
                                   'bilinear') == 255

    # Set bilateral range sigma relative to image range
    tmp = np.sort(pyr_blurry[0].ravel())
    n_pix = len(tmp)
    range_blurry = (tmp[int(np.ceil(0.9 * n_pix)) - 1] -
                    tmp[int(np.floor(0.1 * n_pix))]) / 0.8
    bi_sigma_range0 = bi_sigma_range0 * range_blurry

    # Initialise deblurred images
    pyr_deblurred = [b.copy() for b in pyr_blurry]

    # Determine first/final level
    if first_level == -1:
        first_level = num_levels - 1
    else:
        first_level = min(first_level, num_levels - 1)
    final_level_idx = max(final_level - 1, 0)  # Convert 1-based to 0-based

    # num_iters per level
    if isinstance(num_iters_cfg, int):
        num_iters = [num_iters_cfg] * num_levels
    else:
        num_iters = list(num_iters_cfg)
        while len(num_iters) < num_levels:
            num_iters.append(num_iters[-1])

    history = {'kernels': [], 'images': []}

    # max_nonzeros_w for LARS
    max_nonzeros_w = cfg.get('max_nonzeros_w', blur_kernel_size ** 2)

    # Main loop: coarse to fine
    for s in range(first_level, final_level_idx - 1, -1):
        if verbose:
            print(f"Scale {s + 1} of {num_levels}")

        scale_factor_i = scale_ratio_i ** s
        B = pyr_blurry[s].copy()
        hB, wB = B.shape[:2]

        # Blurry gradients
        Bx = imfilter_conv_replicate(B, kx)
        By = imfilter_conv_replicate(B, ky)
        Bxx = imfilter_conv_replicate(B, kxx)
        Bxy = imfilter_conv_replicate(B, kxy)
        Byy = imfilter_conv_replicate(B, kyy)

        # Estimated latent image at this scale
        L_new = pyr_deblurred[s].copy()

        # Kernel at this scale
        if s == first_level and do_estimate_kernel:
            w_new = np.zeros_like(pyr_kernel[s])
            center = tuple(d // 2 for d in w_new.shape)
            w_new[center] = 1.0
        else:
            w_new = pyr_kernel[s].copy()

        BLUR_KERNEL_SIZE_s = int(
            np.ceil((blur_kernel_size - 1) / 2 * scale_factor_i)) * 2 + 1

        # Mask edges
        mask_kernel = np.ones_like(B)
        if w_new.shape[0] > 0 and w_new.shape[1] > 0:
            conv_ones = convolve2d(
                np.ones_like(B),
                np.ones((BLUR_KERNEL_SIZE_s, BLUR_KERNEL_SIZE_s)),
                mode='same')
            mask_kernel = (conv_ones == BLUR_KERNEL_SIZE_s ** 2).astype(
                np.float64)
        M = binary_erosion(mask_kernel > 0.5,
                           structure=np.ones((3, 3))).astype(np.float64)
        M = M * binary_erosion(pyr_sat_mask[s],
                               structure=np.ones((3, 3))).astype(np.float64)

        # Use_rotations mask for kernel
        if s == first_level:
            if kernel_dilate_radius < np.inf:
                dil_k = np.ones((1 + 2 * kernel_dilate_radius,
                                 1 + 2 * kernel_dilate_radius))
                use_rot = convolve2d((w_new > 0).astype(np.float64),
                                     dil_k, mode='same') >= 0.9
            else:
                use_rot = np.ones_like(w_new, dtype=bool)
        else:
            use_rot = w_new > 0

        use_rot = use_rot.astype(bool)

        # m = number of needed gradients
        m_grad = BLUR_KERNEL_SIZE_s ** 2

        # Reset parameters
        bi_sigma_range = bi_sigma_range0
        bi_sigma_spatial = bi_sigma_spatial0
        shock_dt = shock_dt0
        grad_thresh = None

        for iteration in range(num_iters[s]):
            if verbose:
                print(f"  Iteration {iteration + 1} of {num_iters[s]}")

            L_old = L_new.copy()
            w_old = w_new.copy()

            final_iter_this_level = (iteration == num_iters[s] - 1)
            is_final_iter = (s == final_level_idx and final_iter_this_level)

            # ─── Edge Prediction ───────────────────────────────────────
            if do_estimate_kernel:
                # Bilateral filtering
                Lb = jcb_filter(L_old, L_old, bi_sigma_spatial,
                                bi_sigma_range, bi_size)
                # Shock filtering
                Lp = shock_filter(Lb, shock_iters, shock_dt)

                # Gradient thresholding
                Lpx = imfilter_conv_replicate(Lp, kx)
                Lpy = imfilter_conv_replicate(Lp, ky)
                Lpmag = np.sqrt(Lpx ** 2 + Lpy ** 2)

                # Mask out edges
                erode_struct = np.ones(
                    (BLUR_KERNEL_SIZE_s, BLUR_KERNEL_SIZE_s), dtype=bool)
                Lpmag_mask = binary_erosion(
                    M > 0.5, structure=erode_struct).astype(np.float64)
                Lpmag = Lpmag * Lpmag_mask

                if iteration == 0:
                    # Compute threshold to keep r*m pixels per direction bin
                    Lparg = np.arctan2(Lpy, Lpx)
                    Lpbin = np.mod(Lparg, np.pi)
                    Lpbin = np.ceil(Lpbin / grad_dir_quant).astype(int)
                    Lpbin = np.clip(Lpbin, 1, grad_dir_bins)

                    sorted_idx = np.argsort(Lpmag.ravel())[::-1]
                    Lpmagsorted = Lpmag.ravel()[sorted_idx]
                    Lpbinsorted = Lpbin.ravel()[sorted_idx]

                    ix_thresh = 0
                    for b in range(1, grad_dir_bins + 1):
                        bin_mask = Lpbinsorted == b
                        cumsum = np.cumsum(bin_mask)
                        needed = int(np.ceil(r_factor * m_grad))
                        idx_bin = np.where(cumsum >= needed)[0]
                        if len(idx_bin) > 0:
                            ix_thresh = max(ix_thresh, idx_bin[0])

                    if ix_thresh < len(Lpmagsorted):
                        grad_thresh = Lpmagsorted[ix_thresh]
                    else:
                        grad_thresh = 0.0
                else:
                    grad_thresh = grad_thresh_decrease * grad_thresh

                # Apply threshold
                Px = Lpx.copy()
                Py = Lpy.copy()
                below = Lpmag < grad_thresh
                Px[below] = 0.0
                Py[below] = 0.0

                # Delete isolated non-zeros
                Pnz = (Px != 0) | (Py != 0)
                neighbor_kernel = pad_image(np.zeros((3, 3)), np.array([1, 1, 1, 1]), 1)
                Pnz_filtered = Pnz & (convolve2d(
                    Pnz.astype(np.float64), neighbor_kernel,
                    mode='same') > 0)
                Px[~Pnz_filtered] = 0.0
                Py[~Pnz_filtered] = 0.0

            # Decrease parameters
            bi_sigma_range *= param_decrease
            bi_sigma_spatial /= param_decrease
            shock_dt *= param_decrease

            # ─── Kernel Estimation ─────────────────────────────────────
            if do_estimate_kernel:
                Pxx = imfilter_conv_replicate(Px, kx)
                Pxy_im = imfilter_conv_replicate(Px, ky)
                Pyy = imfilter_conv_replicate(Py, ky)
                Pyx = imfilter_conv_replicate(Py, kx)

                # Build Pall, Ball
                sq_o1 = np.sqrt(omega1)
                sq_o2 = np.sqrt(omega2)
                Pall = np.stack([sq_o1 * Px, sq_o1 * Py,
                                 sq_o2 * Pxx, sq_o2 * Pyy,
                                 sq_o2 * (Pxy_im + Pyx) / 2.0], axis=2)
                Ball = np.stack([sq_o1 * Bx, sq_o1 * By,
                                 sq_o2 * Bxx, sq_o2 * Byy,
                                 sq_o2 * Bxy], axis=2)

                # Observation mask for kernel
                Pnz_conv = convolve2d(
                    (np.sum(np.abs(Pall), axis=2) > 0).astype(np.float64),
                    w_new.reshape(w_new.shape), mode='same') > 0
                kernel_obs_mask = (M > 0.5) & Pnz_conv

                # Compute BtB and Btg
                Gram, Btg = BtB_uni(Pall, Ball, w_new.shape, kernel_obs_mask)

                # Kernel estimation method
                if 'lars' in kernel_method:
                    w_lars, _ = lars(Gram, Btg, nonneg=1, stop=beta_ker / 2,
                                     use_gram=1, precompute_gram=1,
                                     trace=0, mode=2, max_active=max_nonzeros_w)

                    if 'ols' in kernel_method:
                        # Final least squares on active set
                        active = w_lars != 0
                        if np.any(active):
                            A_idx = np.where(active)[0]
                            G_sub = Gram[np.ix_(A_idx, A_idx)]
                            b_sub = Btg[A_idx]
                            try:
                                w_lars[active] = np.linalg.solve(G_sub, b_sub)
                            except np.linalg.LinAlgError:
                                w_lars[active] = np.linalg.lstsq(
                                    G_sub, b_sub, rcond=None)[0]

                    w_new = w_lars.reshape(w_new.shape)

                elif kernel_method == 'conjgrad':
                    # CG solver
                    from scipy.sparse.linalg import cg as sp_cg
                    from scipy.sparse import eye as sp_eye, csr_matrix

                    nk = Gram.shape[0]
                    A_cg = csr_matrix(Gram) + beta_ker * sp_eye(nk)
                    w_flat, _ = sp_cg(A_cg, Btg, maxiter=cfg.get('num_cg_iters', 5))
                    w_new = w_flat.reshape(w_new.shape)

                # Threshold kernel
                if threshold_kernel:
                    w_new[w_new < np.max(w_new) / kernel_threshold] = 0.0

                # Check for all-zero kernel
                if np.sum(w_new) == 0:
                    w_new = np.zeros_like(w_new)
                    center = tuple(d // 2 for d in w_new.shape)
                    w_new[center] = 1.0

                # Recenter kernel
                if recenter_kernel and s > final_level_idx:
                    w_sum = np.sum(w_new)
                    if w_sum > 0:
                        w_new /= w_sum
                    kh_s, kw_s = w_new.shape
                    yy, xx = np.mgrid[0:kh_s, 0:kw_s]
                    mu_y = np.sum(yy * w_new)
                    mu_x = np.sum(xx * w_new)
                    offset_y = int(round(kh_s // 2 - mu_y))
                    offset_x = int(round(kw_s // 2 - mu_x))
                    w_new = np.roll(w_new, offset_y, axis=0)
                    w_new = np.roll(w_new, offset_x, axis=1)

                    # Update use_rotations
                    use_rot = w_new > 0

                # Dilate kernel region
                if kernel_dilate_radius > 0 and kernel_dilate_radius < np.inf:
                    dil_size = 1 + 2 * kernel_dilate_radius
                    dil_k = np.ones((dil_size, dil_size))
                    use_rot = convolve2d(
                        (w_new > 0).astype(np.float64),
                        dil_k, mode='same') >= 0.9
                elif kernel_dilate_radius >= np.inf:
                    use_rot = np.ones_like(w_new, dtype=bool)

                use_rot = use_rot.astype(bool)

                # Normalise
                w_sum = np.sum(w_new)
                if w_sum > 0:
                    w_new /= w_sum

            # ─── Non-blind Deconvolution ───────────────────────────────
            if verbose:
                print(f"  Deconvolution ({image_method})...")

            sat_mask = pyr_sat_mask[s] if s < len(pyr_sat_mask) else None

            if image_method == 'conjgrad':
                L_deblur, deconv_pad = deconv_L2_grad_data(
                    B, w_new, alpha_reg, deconv_maxit,
                    1, 1, 0, 0, 0,
                    saturation_mask=sat_mask,
                    omega=np.array([omega0, omega1, omega2]))
                L_deblur = pad_image(L_deblur, -deconv_pad)

            elif image_method == 'krishnan':
                L_deblur = fast_deconv(B, w_new, kf_lambda, kf_exponent)

            elif image_method == 'sparse':
                L_deblur = deconv_sparse(B, w_new, alpha_reg,
                                         deconv_maxit, sat_mask)
            else:
                raise ValueError(f"Unknown image method: {image_method}")

            L_new = L_deblur

        # Store results
        pyr_deblurred[s] = L_new.copy()
        pyr_kernel[s] = w_new.copy()

        history['kernels'].append(w_new.copy())
        history['images'].append(L_new.copy())

        # ─── Upsample to next (finer) scale ───────────────────────────
        if s > final_level_idx:
            # Upsample image
            target_shape = pyr_blurry[s - 1].shape[:2]
            pyr_deblurred[s - 1] = imresize(
                pyr_deblurred[s], target_shape, 'bicubic')

            # Upsample kernel
            if do_estimate_kernel:
                pyr_kernel[s - 1] = upsample_kernel_map(
                    pyr_kernel[s], pyr_tt[s], pyr_tt[s - 1], scale_ratio_k)

    # Final results
    L_final = pyr_deblurred[final_level_idx]
    k_final = pyr_kernel[final_level_idx]

    return L_final, k_final, history
