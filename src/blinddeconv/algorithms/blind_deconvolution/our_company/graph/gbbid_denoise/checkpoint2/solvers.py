"""
solvers.py

Core solver functions for GBBID (Graph-Based Blind Image Deblurring).

Ported from MATLAB code by Yuanchao Bai et al.
Reference:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Also includes non-blind deconvolution from:
    D. Krishnan, R. Fergus: "Fast Image Deconvolution using
    Hyper-Laplacian Priors", NIPS 2009.

Contains:
    TV_denoising            — Split-Bregman TV denoising preprocessing
                              (TV_denoising.m)
    Deblur_GL_CG_4          — Graph-Laplacian CG solver for x-subproblem
                              (Deblur_GL_CG_4.m)
    kernel_solver_L2         — Kernel estimation via CG in frequency domain
                              (kernel_solver_L2.m)
    bid_rgtv_c2f_cg          — Main coarse-to-fine blind deconvolution
                              (bid_rgtv_c2f_cg.m)
    fast_deconv              — Non-blind deconv with hyper-Laplacian prior
                              (fast_deconv.m)
    Deconvolution_FHLP       — Non-blind deconv wrapper with edgetaper
                              (Deconvolution_FHLP.m)

MATLAB -> Python notes (see utils.py for full table):
    imfilter(x, h, 'conv', 'replicate') -> scipy.ndimage.convolve(x, h, mode='nearest')
    imfilter(x, h, 'circular')          -> scipy.ndimage.correlate(x, h, mode='wrap')
    rot90(h, 2)                         -> h[::-1, ::-1]
    conv2(A, B, 'same')                 -> scipy.signal.fftconvolve(A, B, mode='same')
    padarray(A, [p,p], 'both')          -> np.pad(A, p) (zero-pad default)
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import convolve as ndimage_convolve
from skimage.transform import resize as sk_resize

from .utils import (
    psf2otf,
    otf2psf,
    G_padding,
    Copy_Enlarge_h,
    fftconv,
    edgetaper,
    weights_computation,
    informative_edge_mask_adaptive_mine,
    kernel_centralize,
    conjgrad,
    GenerateFrameletFilter,
    FraDecMultiLevel2D,
    kernel_filter,
    solve_image,
    clear_solve_image_cache,
)


# ═════════════════════════════════════════════════════════════════════════════
# TV_denoising  (from TV_denoising.m)
# ═════════════════════════════════════════════════════════════════════════════

def TV_denoising(I, weight, max_it):
    """
    Denoising with TV prior implemented in the frequency domain
    (Split Bregman / ADMM formulation).

    MATLAB: TV_denoising.m (by Yuanchao Bai)

    Parameters
    ----------
    I : 2D array — input image
    weight : (mu, gamma) — regularization parameters
    max_it : int — number of iterations

    Returns
    -------
    x : 2D array — denoised image (same size as I)
    """
    # Identity-like blur kernel (near-delta)
    h = np.full((3, 3), 1e-10, dtype=np.float64)
    h[1, 1] = 1.0
    h = h / h.sum()

    mu = weight[0]
    gamma = weight[1]

    s_h, s_w = h.shape

    # Pad image
    I_sym, border = Copy_Enlarge_h(I, (s_h * 3, s_w * 3))
    I_sym = edgetaper(I_sym, h)
    I_h, I_w = I_sym.shape

    # Precompute filter arrays in the larger support domain
    B = h
    Bt = h[::-1, ::-1]  # rot90(h, 2)

    # MATLAB: BtB = conv2(Bt, B) — full convolution of two small kernels
    from scipy.signal import fftconvolve
    BtB = fftconvolve(Bt, B)
    S_h = 2 * s_h - 1
    S_w = 2 * s_w - 1

    # TV filters: second-order differences
    vtv_w = np.array([0, -1, 2, -1, 0], dtype=np.float64)
    vtv_h = vtv_w.copy()

    VtV_w = np.zeros((S_h, S_w), dtype=np.float64)
    # MATLAB: VtV_w(s_h, s_w-2:s_w+2) = vtv_w  (1-indexed)
    VtV_w[s_h - 1, s_w - 3:s_w + 2] = vtv_w

    VtV_h = np.zeros((S_h, S_w), dtype=np.float64)
    # MATLAB: VtV_h(s_h-2:s_h+2, s_w) = vtv_h  (1-indexed)
    VtV_h[s_h - 3:s_h + 2, s_w - 1] = vtv_h

    Bt_tmp = np.zeros((S_h, S_w), dtype=np.float64)
    # MATLAB: Bt_tmp(s_h-(s_h-1)/2:s_h+(s_h-1)/2, s_w-(s_w-1)/2:s_w+(s_w-1)/2) = Bt
    hh = (s_h - 1) // 2
    hw = (s_w - 1) // 2
    Bt_tmp[s_h - 1 - hh:s_h + hh, s_w - 1 - hw:s_w + hw] = Bt

    # First-order TV filters (transpose)
    vt_w = np.array([0, -1, 1], dtype=np.float64)
    vt_h = vt_w.copy()

    Vt_w = np.zeros((S_h, S_w), dtype=np.float64)
    # MATLAB: Vt_w(s_h, s_w-1:s_w+1) = vt_w  (1-indexed)
    Vt_w[s_h - 1, s_w - 2:s_w + 1] = vt_w

    Vt_h = np.zeros((S_h, S_w), dtype=np.float64)
    # MATLAB: Vt_h(s_h-1:s_h+1, s_w) = vt_h  (1-indexed)
    Vt_h[s_h - 2:s_h + 1, s_w - 1] = vt_h

    # Forward difference filters
    v_w = np.array([[1, -1, 0]], dtype=np.float64)
    v_h = v_w.T

    # Auxiliary variables
    z_h = np.zeros((I_h, I_w), dtype=np.float64)
    z_w = np.zeros((I_h, I_w), dtype=np.float64)
    y_h = np.zeros((I_h, I_w), dtype=np.float64)
    y_w = np.zeros((I_h, I_w), dtype=np.float64)

    # FFT of padded filter arrays
    fft_shape = (S_h + I_h - 1, S_w + I_w - 1)

    def _embed_fft(arr, arr_shape, total_shape):
        Ftmp = np.zeros(total_shape, dtype=np.float64)
        Ftmp[:arr_shape[0], :arr_shape[1]] = arr
        return fft2(Ftmp)

    FBtB = _embed_fft(BtB, (S_h, S_w), fft_shape)
    FBt = _embed_fft(Bt_tmp, (S_h, S_w), fft_shape)
    FVtV_w = _embed_fft(VtV_w, (S_h, S_w), fft_shape)
    FVtV_h = _embed_fft(VtV_h, (S_h, S_w), fft_shape)
    FVt_w = _embed_fft(Vt_w, (S_h, S_w), fft_shape)
    FVt_h = _embed_fft(Vt_h, (S_h, S_w), fft_shape)

    # FFT of image
    Fb = _embed_fft(I_sym, (I_h, I_w), fft_shape)

    # Solver loop
    for _ in range(max_it):
        Fz_h = _embed_fft(z_h, (I_h, I_w), fft_shape)
        Fz_w = _embed_fft(z_w, (I_h, I_w), fft_shape)
        Fy_h = _embed_fft(y_h, (I_h, I_w), fft_shape)
        Fy_w = _embed_fft(y_w, (I_h, I_w), fft_shape)

        # x-subproblem (frequency domain)
        x = ifft2(
            (Fb * FBt
             + gamma * FVt_h * Fz_h + gamma * FVt_w * Fz_w
             + gamma * FVt_h * Fy_h + gamma * FVt_w * Fy_w)
            / (FBtB + gamma * FVtV_w + gamma * FVtV_h)
        )
        x = np.real(x[:I_h, :I_w])

        # Compute gradients: imfilter(x, v_h/v_w, 'conv', 'replicate')
        Vx_h = ndimage_convolve(x, v_h, mode='nearest')
        Vx_h[-1, :] = 0.0
        Vx_w = ndimage_convolve(x, v_w, mode='nearest')
        Vx_w[:, -1] = 0.0

        # z-subproblem: soft shrinkage
        thr = mu / gamma

        z_h[:] = 0.0
        diff_h = Vx_h - y_h
        mask_pos = diff_h > thr
        mask_neg = diff_h < -thr
        z_h[mask_pos] = diff_h[mask_pos] - thr
        z_h[mask_neg] = diff_h[mask_neg] + thr

        z_w[:] = 0.0
        diff_w = Vx_w - y_w
        mask_pos = diff_w > thr
        mask_neg = diff_w < -thr
        z_w[mask_pos] = diff_w[mask_pos] - thr
        z_w[mask_neg] = diff_w[mask_neg] + thr

        # Bregman update
        y_h = y_h - (Vx_h - z_h)
        y_w = y_w - (Vx_w - z_w)

    # Crop borders
    x = x[border[0]:I_h - border[0], border[1]:I_w - border[1]]
    return x


# ═════════════════════════════════════════════════════════════════════════════
# Deblur_GL_CG_4  (from Deblur_GL_CG_4.m)
# ═════════════════════════════════════════════════════════════════════════════

def Deblur_GL_CG_4(Y_b, k, W, we, max_iter):
    """
    Restore skeleton image using Graph-Laplacian regularized CG.

    MATLAB: Deblur_GL_CG_4.m (by Yuanchao Bai)

    Solves:
        min_x  ||k*x - Y_b||^2  +  we * x^T L x
    where L is the graph Laplacian from W, using conjugate gradient.

    Parameters
    ----------
    Y_b : 2D array — blurred image (possibly padded)
    k : 2D array — blur kernel
    W : (h*w, 4) array — graph weights
    we : float — graph regularization weight (mu)
    max_iter : int — CG iterations

    Returns
    -------
    x : 2D array — restored image, clamped to [0, 1]
    """
    # Directional filters (4 neighbours)
    d1 = np.array([[1, -1, 0]], dtype=np.float64)
    d1_c = np.array([[0, -1, 1]], dtype=np.float64)
    d2 = d1.T
    d2_c = d1_c.T
    d3 = np.array([[0, -1, 1]], dtype=np.float64)
    d3_c = np.array([[1, -1, 0]], dtype=np.float64)
    d4 = d3.T
    d4_c = d3_c.T

    Y_b_padding = Y_b
    h_p, w_p = Y_b_padding.shape
    x = Y_b_padding.copy()

    vertex, neighbours_num = W.shape
    if vertex != h_p * w_p or neighbours_num != 4:
        raise ValueError("Weights matrix W is not correct, please check.")

    k_flipped = k[::-1, ::-1]  # rot90(k, 2)
    use_fft = max(k.shape) >= 25

    def _apply_blur(img):
        """Apply k, then k^T (adjoint) with mask = ones."""
        if use_fft:
            return fftconv(fftconv(img, k, 'same'), k_flipped, 'same')
        else:
            tmp = ndimage_convolve(img, k, mode='nearest')
            return ndimage_convolve(tmp, k_flipped, mode='nearest')

    def _apply_graph(img):
        """Apply weighted graph Laplacian D^T W D x."""
        w1 = W[:, 0].reshape(h_p, w_p)
        w2 = W[:, 1].reshape(h_p, w_p)
        w3 = W[:, 2].reshape(h_p, w_p)
        w4 = W[:, 3].reshape(h_p, w_p)

        out = we * ndimage_convolve(
            w1 * ndimage_convolve(img, d1, mode='nearest'),
            d1_c, mode='nearest')
        out += we * ndimage_convolve(
            w2 * ndimage_convolve(img, d2, mode='nearest'),
            d2_c, mode='nearest')
        out += we * ndimage_convolve(
            w3 * ndimage_convolve(img, d3, mode='nearest'),
            d3_c, mode='nearest')
        out += we * ndimage_convolve(
            w4 * ndimage_convolve(img, d4, mode='nearest'),
            d4_c, mode='nearest')
        return out

    # b = K^T * Y_b
    if use_fft:
        b = fftconv(Y_b_padding, k_flipped, 'same')
    else:
        b = ndimage_convolve(Y_b_padding, k_flipped, mode='nearest')

    # Ax = K^T K x  +  graph term
    Ax = _apply_blur(x) + _apply_graph(x)

    r = b - Ax
    rho_1 = 0.0
    p = None #fixing cl
    for i in range(max_iter):
        rho = np.sum(r * r)

        if i > 0:
            beta_cg = rho / rho_1
            p = r + beta_cg * p
        else:
            p = r.copy()

        # Ap = K^T K p  +  graph term
        Ap = _apply_blur(p) + _apply_graph(p)

        q = Ap
        pq = np.sum(p * q)
        if pq == 0:
            break
        alpha_cg = rho / pq
        x = x + alpha_cg * p
        r = r - alpha_cg * q

        rho_1 = rho

        # Clamp to [0, 1]
        x = np.clip(x, 0.0, 1.0)

    return x


# ═════════════════════════════════════════════════════════════════════════════
# kernel_solver_L2  (from kernel_solver_L2.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_Ax_kernel(x, p):
    """
    Matrix-vector product for kernel estimation CG system.

    MATLAB:
        x_f = psf2otf(x, p.img_size);
        y   = otf2psf(p.m .* x_f, p.psf_size);
        y   = y + p.lambda * x;
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def kernel_solver_L2(Y, b, k_size, M, lambda_val):
    """
    Solve for blur kernel in gradient domain via conjugate gradient.

    MATLAB: kernel_solver_L2.m (by Yuanchao Bai)

    Parameters
    ----------
    Y : 2D array — restored skeleton (latent) image
    b : 2D array — blurred observation (same scale)
    k_size : int — kernel size (odd)
    M : 2D array or None — informative edge mask
    lambda_val : float — regularization weight

    Returns
    -------
    k : 2D array (k_size, k_size) — estimated blur kernel
    """
    dx = np.array([[1, -1, 0]], dtype=np.float64)   # rot90([0,-1,1], 2)
    dy = dx.T

    if M is None:
        M = np.ones_like(Y)

    # Gradient images of latent estimate, masked
    Yx = ndimage_convolve(Y, dx, mode='nearest') * M
    Yy = ndimage_convolve(Y, dy, mode='nearest') * M

    # Gradient images of blurred observation
    bx = ndimage_convolve(b, dx, mode='nearest')
    by = ndimage_convolve(b, dy, mode='nearest')

    # Pad for frequency-domain computation
    pad_time = 3
    pad_size = int(np.floor(k_size * pad_time))

    bx_p = np.pad(bx, pad_size)
    by_p = np.pad(by, pad_size)
    Yx_p = np.pad(Yx, pad_size)
    Yy_p = np.pad(Yy, pad_size)

    Yx_f = fft2(Yx_p)
    Yy_f = fft2(Yy_p)
    bx_f = fft2(bx_p)
    by_f = fft2(by_p)

    wx = 25.0
    wy = 25.0
    psf_size = (k_size, k_size)

    # RHS: b_f = wx * conj(Yx_f) .* bx_f + wy * conj(Yy_f) .* by_f
    b_rhs_f = wx * np.conj(Yx_f) * bx_f + wy * np.conj(Yy_f) * by_f
    b_rhs = np.real(otf2psf(b_rhs_f, psf_size))

    # LHS parameters
    p = {
        'm': wx * np.conj(Yx_f) * Yx_f + wy * np.conj(Yy_f) * Yy_f,
        'img_size': bx_f.shape,
        'psf_size': psf_size,
        'lambda': lambda_val,
    }

    # Initial guess: uniform
    psf = np.ones(psf_size, dtype=np.float64) / (k_size * k_size)
    psf = conjgrad(psf, b_rhs, 20, 1e-5, _compute_Ax_kernel, p)

    # Threshold small values and normalize
    psf[psf < psf.max() * 0.05] = 0.0
    psf_sum = psf.sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    return psf


# ═════════════════════════════════════════════════════════════════════════════
# bid_rgtv_c2f_cg  (from bid_rgtv_c2f_cg.m)
# ═════════════════════════════════════════════════════════════════════════════

def bid_rgtv_c2f_cg(Y_b, k_estimate_size, show_intermediate=False):
    """
    Blind image deblurring from coarse to fine using RGTV.

    MATLAB: bid_rgtv_c2f_cg.m (by Yuanchao Bai)

    Parameters
    ----------
    Y_b : 2D array — blurred image (border-cropped)
    k_estimate_size : int — estimated kernel size (odd)
    show_intermediate : bool — (ignored in Python, no display)

    Returns
    -------
    k_estimate : 2D array — estimated blur kernel
    Y_r_rgtv_cg : 2D array — restored skeleton image
    """
    scale_factor = np.log2(3)
    level_num = int(np.ceil(np.log(k_estimate_size / 7) / np.log(scale_factor))) + 1

    # Build image pyramid and kernel sizes
    image_pyramid = [None] * level_num
    k_size = np.zeros(level_num, dtype=int)
    image_size = np.zeros((level_num, 2), dtype=int)

    # Level 0 (finest): TV denoising
    image_pyramid[0] = TV_denoising(Y_b, (0.01, 0.1), 10)

    k_size[0] = k_estimate_size
    image_size[0] = image_pyramid[0].shape

    for i in range(1, level_num):
        image_size[i] = np.floor(image_size[i - 1] / np.log2(3)).astype(int)
        image_pyramid[i] = sk_resize(
            image_pyramid[i - 1],
            (int(image_size[i, 0]), int(image_size[i, 1])),
            order=1,     # bilinear
            anti_aliasing=True,
            preserve_range=True,
        )
        k_size[i] = int(np.floor(k_size[i - 1] / np.log2(3)))
        k_size[i] = k_size[i] + (1 - k_size[i] % 2)  # ensure odd

    # Framelet filter setup for kernel filtering
    frame = 1   # piecewise linear
    Level = 1
    D, R = GenerateFrameletFilter(frame)

    # Main RGTV blind loop: coarse to fine
    k_estimate = None
    Y_r_rgtv_cg = None

    for level in range(level_num - 1, -1, -1):
        mu = 0.01
        lambda_val = 0.05
        sigma = 0.1 * np.sqrt(2)

        if level >= level_num - 1:
            # Coarsest level: delta kernel
            ks = int(k_size[level])
            k_estimate = np.zeros((ks, ks), dtype=np.float64)
            k_center = ks // 2
            k_estimate[k_center, k_center] = 1.0
        else:
            # Upsample kernel from coarser level
            ks = int(k_size[level])
            k_estimate = sk_resize(
                k_estimate, (ks, ks),
                order=1, anti_aliasing=False, preserve_range=True)
            k_estimate[k_estimate < k_estimate.max() * 0.05] = 0.0
            k_sum = k_estimate.sum()
            if k_sum > 0:
                k_estimate = k_estimate / k_sum

        # Pad image for graph construction
        Y_b_padding, padsize = G_padding(image_pyramid[level], k_estimate, 1)
        Y_r_rgtv_cg = Y_b_padding.copy()
        h, w = Y_r_rgtv_cg.shape

        for iter_main in range(3):
            W1 = np.ones((h * w, 4), dtype=np.float64)
            W = W1.copy()

            for i in range(3):
                for j in range(3):
                    Y_r_rgtv_cg = Deblur_GL_CG_4(
                        Y_b_padding, k_estimate, W, mu, 20)
                    W = W1 * weights_computation(Y_r_rgtv_cg, None, 4, 2)

                W1 = weights_computation(Y_r_rgtv_cg, sigma, 4, 1)
                W = W1 * weights_computation(Y_r_rgtv_cg, None, 4, 2)

            # Crop padding
            Y_r_rgtv_cg = Y_r_rgtv_cg[
                padsize[0]:h - padsize[0],
                padsize[1]:w - padsize[1]
            ]

            # Kernel estimation
            t_s = 0.1
            t_r = 0.3
            M = informative_edge_mask_adaptive_mine(Y_r_rgtv_cg, t_s, t_r, 5)
            k_estimate = kernel_solver_L2(
                Y_r_rgtv_cg, image_pyramid[level],
                int(k_size[level]), M, lambda_val)

            # Wavelet filtering at finer levels
            if level <= 1:
                Cf = FraDecMultiLevel2D(k_estimate, D, Level)
                k_estimate = kernel_filter(Cf, R, Level, 0.05)
                k_estimate[k_estimate < k_estimate.max() * 0.05] = 0.0
                k_sum = k_estimate.sum()
                if k_sum > 0:
                    k_estimate = k_estimate / k_sum
                k_estimate = kernel_centralize(k_estimate, 0.1)

            lambda_val = lambda_val / 1.2

    return k_estimate, Y_r_rgtv_cg


# ═════════════════════════════════════════════════════════════════════════════
# fast_deconv  (from fast_deconv.m, Krishnan & Fergus NIPS 2009)
# ═════════════════════════════════════════════════════════════════════════════

def _computeDenominator(y, k):
    """
    Compute denominator and part of numerator for Equation (3) of the paper.

    MATLAB: computeDenominator in fast_deconv.m

    Returns
    -------
    Nomin1 : F(K)' * F(y)
    Denom1 : |F(K)|^2
    Denom2 : |F(D^1)|^2 + |F(D^2)|^2
    """
    sizey = y.shape
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = (np.abs(psf2otf(np.array([[1, -1]]), sizey)) ** 2
              + np.abs(psf2otf(np.array([[1], [-1]]), sizey)) ** 2)
    return Nomin1, Denom1, Denom2


def fast_deconv(yin, k, lambda_val, alpha, yout0=None):
    """
    Non-blind deconvolution with hyper-Laplacian prior.

    MATLAB: fast_deconv.m (Krishnan & Fergus, NIPS 2009)

    Solves:
        min_y  (lambda/2)*||k*y - yin||^2  +  ||D_x y||^alpha  +  ||D_y y||^alpha

    Parameters
    ----------
    yin : 2D array — blurred input (grayscale)
    k : 2D array — convolution kernel (odd-sized)
    lambda_val : float — data-fidelity weight
    alpha : float — hyper-Laplacian exponent (0 < alpha <= 2)
    yout0 : 2D array or None — initialization (default: yin)

    Returns
    -------
    yout : 2D array — deblurred image
    """
    # Continuation parameters
    beta = 1.0
    beta_rate = 2.0 * np.sqrt(2)
    beta_max = 2 ** 8

    mit_inn = 1  # inner iterations per outer iteration

    m, n = yin.shape

    if yout0 is not None:
        yout = yout0.copy()
    else:
        yout = yin.copy()

    # Check kernel is odd-sized
    if k.shape[0] % 2 != 1 or k.shape[1] % 2 != 1:
        raise ValueError("Blur kernel k must be odd-sized.")

    # Compute constant quantities (Eqn. 3 of paper)
    Nomin1, Denom1, Denom2 = _computeDenominator(yin, k)

    # Circular-boundary gradients
    # diff(yout, 1, 2)  ->  np.diff(yout, n=1, axis=1)
    # [diff(yout,1,2), yout(:,1)-yout(:,end)]
    youtx = np.concatenate([np.diff(yout, 1, axis=1), yout[:, 0:1] - yout[:, -1:]], axis=1)
    youty = np.concatenate([np.diff(yout, 1, axis=0), yout[0:1, :] - yout[-1:, :]], axis=0)

    # Main continuation loop
    while beta < beta_max:
        gamma = beta / lambda_val
        Denom = Denom1 + gamma * Denom2

        for _ in range(mit_inn):
            # w-subproblem: eqn (5)
            Wx = solve_image(youtx, beta, alpha)
            Wy = solve_image(youty, beta, alpha)

            # x-subproblem: eqn (3)
            # Transpose of x,y gradients
            # Wxx = [Wx(:,n)-Wx(:,1), -diff(Wx,1,2)]
            Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1], -np.diff(Wx, 1, axis=1)], axis=1)
            # Wxx += [Wy(m,:)-Wy(1,:); -diff(Wy,1,1)]
            Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :], -np.diff(Wy, 1, axis=0)], axis=0)

            Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
            yout = np.real(ifft2(Fyout))

            # Update gradients
            youtx = np.concatenate([np.diff(yout, 1, axis=1), yout[:, 0:1] - yout[:, -1:]], axis=1)
            youty = np.concatenate([np.diff(yout, 1, axis=0), yout[0:1, :] - yout[-1:, :]], axis=0)

        beta = beta * beta_rate

    return yout


# ═════════════════════════════════════════════════════════════════════════════
# Deconvolution_FHLP  (from Deconvolution_FHLP.m)
# ═════════════════════════════════════════════════════════════════════════════

def Deconvolution_FHLP(y, kernel, lambda_val=2e3, alpha=0.5):
    """
    Non-blind deconvolution using Fast Hyper-Laplacian Priors (NIPS 2009).

    MATLAB: Deconvolution_FHLP.m

    Parameters
    ----------
    y : 2D array — blurred image
    kernel : 2D array — estimated blur kernel
    lambda_val : float — data-fidelity weight (default 2e3)
    alpha : float — hyper-Laplacian exponent (default 0.5)

    Returns
    -------
    x : 2D array — deblurred image (same size as y)
    """
    # Avoid zeros in kernel
    kernel = kernel.copy().astype(np.float64)
    kernel[kernel == 0] = 1e-10
    kernel = kernel / kernel.sum()

    ks = (kernel.shape[0] - 1) // 2

    # Pad with replicate boundaries
    y_padded = np.pad(y, ks, mode='edge')

    # Edgetaper to handle circular boundary conditions
    for _ in range(4):
        y_padded = edgetaper(y_padded, kernel)

    # Clear persistent LUT cache (matches MATLAB: clear persistent)
    clear_solve_image_cache()

    # Run non-blind deconvolution
    x = fast_deconv(y_padded, kernel, lambda_val, alpha)

    # Remove padding
    x = x[ks:x.shape[0] - ks, ks:x.shape[1] - ks]

    return x
