"""
solvers.py

Core solver functions for the ESM (Enhanced Sparse Model) blind deconvolution.

Ported from MATLAB code by Chen et al. (ECCV 2020).
Reference:
    L. Chen, F. Fang, S. Lei, F. Li, G. Zhang: "Enhanced Sparse Model
    for Blind Deblurring", ECCV 2020.
    @inproceedings{DBLP:conf/eccv/ChenFLLZ20,
        title = {Enhanced Sparse Model for Blind Deblurring},
        booktitle = {ECCV},
        year = {2020}
    }

Contains direct Python ports of the following MATLAB files located in
`ECCV20_enhanced_sparse_model/`:

    L0Restoration_HS.m          → L0Restoration_HS      (I-subproblem,
                                                         enhanced sparse prior)
    estimate_psf_l0.m           → estimate_psf_l0       (k-subproblem, with
                                                         ℓ0−ℓ1 data-gradient
                                                         prior)
    L0Restoration.m             → L0Restoration         (plain L0 gradient
                                                         prior, used inside
                                                         ringing removal)
    blind_deconv_main.m         → blind_deconv_main     (single-scale loop)
    blind_deconv.m              → blind_deconv          (coarse-to-fine
                                                         pyramid)
    deblurring_adm_aniso.m      → deblurring_adm_aniso  (TV-ℓ² via ADM,
                                                         alpha=1 branch only —
                                                         matches actual usage)
    ringing_artifacts_removal.m → ringing_artifacts_removal  (final non-blind
                                                              post-processing)

MATLAB → Python mapping highlights:
    diff(S, 1, 2)             → np.diff(S, n=1, axis=1)
    diff(S, 1, 1)             → np.diff(S, n=1, axis=0)
    S(:,1,:) - S(:,end,:)     → S[:, 0:1, ...] - S[:, -1:, ...]
    fft2 / ifft2 on 3D        → np.fft.fft2 with axes=(0, 1)
    bwconncomp(k, 8)          → scipy.ndimage.label(k, structure=3x3 ones)
    imresize(k, ret)          → scipy.ndimage.zoom(..., order=3) (cubic)
    conv2(A, B, 'valid')      → scipy.signal.convolve2d(A, B, mode='valid')
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    fftconv,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
)


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers for circular gradients (match MATLAB element-by-element)
# ═════════════════════════════════════════════════════════════════════════════

def _circ_diff_x(S: np.ndarray) -> np.ndarray:
    """
    MATLAB: [diff(S,1,2), S(:,1,:) - S(:,end,:)]  (forward circular ∂x)

    Works for both 2-D (H,W) and 3-D (H,W,D) arrays.
    """
    d = np.diff(S, n=1, axis=1)
    wrap = S[:, 0:1, ...] - S[:, -1:, ...]
    return np.concatenate([d, wrap], axis=1)


def _circ_diff_y(S: np.ndarray) -> np.ndarray:
    """
    MATLAB: [diff(S,1,1); S(1,:,:) - S(end,:,:)]  (forward circular ∂y)
    """
    d = np.diff(S, n=1, axis=0)
    wrap = S[0:1, :, ...] - S[-1:, :, ...]
    return np.concatenate([d, wrap], axis=0)


def _adjoint_diff_x(h: np.ndarray) -> np.ndarray:
    """
    MATLAB: [h(:,end,:) - h(:,1,:), -diff(h,1,2)]

    The divergence-style adjoint of _circ_diff_x used by L0Restoration /
    L0Restoration_HS to assemble Normin2.
    """
    head = h[:, -1:, ...] - h[:, 0:1, ...]
    rest = -np.diff(h, n=1, axis=1)
    return np.concatenate([head, rest], axis=1)


def _adjoint_diff_y(v: np.ndarray) -> np.ndarray:
    """
    MATLAB: [v(end,:,:) - v(1,:,:); -diff(v,1,1)]
    """
    head = v[-1:, :, ...] - v[0:1, :, ...]
    rest = -np.diff(v, n=1, axis=0)
    return np.concatenate([head, rest], axis=0)


def _fft2_planes(S: np.ndarray) -> np.ndarray:
    """MATLAB's fft2 on 3-D array: FFT over the first two axes."""
    if S.ndim == 2:
        return fft2(S)
    return fft2(S, axes=(0, 1))


def _ifft2_planes(F: np.ndarray) -> np.ndarray:
    """MATLAB's ifft2 on 3-D array: iFFT over the first two axes."""
    if F.ndim == 2:
        return ifft2(F)
    return ifft2(F, axes=(0, 1))


# ═════════════════════════════════════════════════════════════════════════════
# L0Restoration_HS  (from L0Restoration_HS.m) — Enhanced Sparse Model I-step
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration_HS(Im: np.ndarray,
                     kernel: np.ndarray,
                     lambda_data: float,
                     lambda_grad: float,
                     theta: float,
                     kappa: float = 2.0) -> np.ndarray:
    """
    Latent-image update for the ESM model.

    Equivalent to MATLAB L0Restoration_HS.m.

    Solves, with half-quadratic splitting and continuation on (beta1, tau1):

        min_I  ||k*I - B||_2^2
             + λ_grad * ( ||∇I||_0  −  ||∇I||_1 )
             + λ_data * ( ||k*∇I − ∇B||_0  −  ||k*∇I − ∇B||_1 )

    Parameters
    ----------
    Im : (H, W) float64 array — blurred image.
    kernel : (kh, kw) PSF.
    lambda_data : λ_data from the paper (data-gradient sparsity weight).
    lambda_grad : λ_grad from the paper (gradient sparsity weight).
    theta : θ parameter of the enhanced ℓ0−ℓ1 prior.
    kappa : geometric growth factor for (beta1, tau1).  MATLAB default 2.0.

    Returns
    -------
    S : (H, W) float64 array — restored latent image cropped to input size.
    """
    H, W = Im.shape[:2]

    # ── Pad image boundaries (wrap_boundary_liu) ────────────────────────────
    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    Im = wrap_boundary_liu(Im, tuple(target))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1.0, -1.0]], dtype=np.float64)
    fy = np.array([[1.0], [-1.0]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
        KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
        Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))
        otfFx_b = np.tile(otfFx[:, :, np.newaxis], (1, 1, D))
        otfFy_b = np.tile(otfFy[:, :, np.newaxis], (1, 1, D))
    else:
        otfFx_b = otfFx
        otfFy_b = otfFy

    Normin1 = np.conj(KER) * _fft2_planes(S)

    beta1 = 2.0 * lambda_grad
    tau1 = 2.0 * lambda_data

    # Gradients of the (fixed) padded input
    B_h = _circ_diff_x(Im)
    B_v = _circ_diff_y(Im)

    KG_h = otfFx_b * KER
    KG_v = otfFy_b * KER
    KG = np.abs(KG_h) ** 2 + np.abs(KG_v) ** 2

    while beta1 < betamax:
        Denormin = Den_KER + beta1 * Denormin2 + tau1 * KG

        S_h = _circ_diff_x(S)
        S_v = _circ_diff_y(S)

        # ── q subproblem: prox of  λ_data (‖·‖₀ − ‖·‖₁) on (B_h - k*S_h) ──
        q_h = B_h - fftconv(S_h, kernel)
        q_h = np.sign(q_h) * np.maximum(
            np.abs(q_h) - lambda_data * theta / (2.0 * tau1), 0.0
        )
        q_v = B_v - fftconv(S_v, kernel)
        q_v = np.sign(q_v) * np.maximum(
            np.abs(q_v) - lambda_data * theta / (2.0 * tau1), 0.0
        )
        t_h = q_h ** 2 < lambda_data / tau1
        t_v = q_v ** 2 < lambda_data / tau1
        q_h[t_h] = 0.0
        q_v[t_v] = 0.0

        # ── g subproblem: prox of  λ_grad (‖·‖₀ − ‖·‖₁) on ∇S ───────────────
        g_h = S_h.copy()
        g_v = S_v.copy()
        g_h = np.sign(g_h) * np.maximum(
            np.abs(g_h) - lambda_grad * theta / (2.0 * beta1), 0.0
        )
        g_v = np.sign(g_v) * np.maximum(
            np.abs(g_v) - lambda_grad * theta / (2.0 * beta1), 0.0
        )
        t_h = g_h ** 2 < lambda_grad / beta1
        t_v = g_v ** 2 < lambda_grad / beta1
        g_h[t_h] = 0.0
        g_v[t_v] = 0.0

        # ── I subproblem: closed-form in Fourier domain ─────────────────────
        Normin2 = _adjoint_diff_x(g_h) + _adjoint_diff_y(g_v)
        Normin3 = np.conj(KG_h) * _fft2_planes(B_h - q_h) \
                + np.conj(KG_v) * _fft2_planes(B_v - q_v)

        FS = (Normin1 + beta1 * _fft2_planes(Normin2) + tau1 * Normin3) \
             / Denormin
        S = np.real(_ifft2_planes(FS))

        beta1 = beta1 * kappa
        tau1 = tau1 * kappa

    return S[:H, :W, ...]


# ═════════════════════════════════════════════════════════════════════════════
# L0Restoration  (from L0Restoration.m) — plain L0 gradient prior
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im: np.ndarray,
                  kernel: np.ndarray,
                  lambda_grad: float,
                  kappa: float = 2.0) -> np.ndarray:
    """
    Non-blind restoration with L0 gradient prior only.

    Equivalent to MATLAB L0Restoration.m.  Used inside
    ringing_artifacts_removal as the "sharp" reference latent image.

    Parameters
    ----------
    Im : (H, W) or (H, W, D) blurred image.
    kernel : PSF.
    lambda_grad : L0-gradient regularisation weight.
    kappa : continuation growth factor for beta.
    """
    H, W = Im.shape[:2]

    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    Im = wrap_boundary_liu(Im, tuple(target))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1.0, -1.0]], dtype=np.float64)
    fy = np.array([[1.0], [-1.0]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
        KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
        Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * _fft2_planes(S)

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        h = _circ_diff_x(S)
        v = _circ_diff_y(S)

        if D == 1:
            t = (h ** 2 + v ** 2) < lambda_grad / beta
        else:
            # Per-pixel L0 on combined channel energy
            t = np.sum(h ** 2 + v ** 2, axis=2) < lambda_grad / beta
            t = np.tile(t[:, :, np.newaxis], (1, 1, D))

        h[t] = 0.0
        v[t] = 0.0

        Normin2 = _adjoint_diff_x(h) + _adjoint_diff_y(v)

        FS = (Normin1 + beta * _fft2_planes(Normin2)) / Denormin
        S = np.real(_ifft2_planes(FS))

        beta = beta * kappa

    return S[:H, :W, ...]


# ═════════════════════════════════════════════════════════════════════════════
# estimate_psf_l0  (from estimate_psf_l0.m)
# ═════════════════════════════════════════════════════════════════════════════

def _compute_Ax_psf(x: np.ndarray, p: dict) -> np.ndarray:
    """
    Matrix-vector product for the k-subproblem CG solver.

    MATLAB:
        x_f = psf2otf(x, p.img_size);
        y   = otf2psf(p.m .* x_f, p.psf_size);
        y   = y + p.lambda * x;
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def estimate_psf_l0(blurred_x: np.ndarray,
                    blurred_y: np.ndarray,
                    latent_x: np.ndarray,
                    latent_y: np.ndarray,
                    weight: float,
                    tau: float,
                    k_prev: np.ndarray,
                    theta: float) -> np.ndarray:
    """
    Kernel update step of the ESM model.

    Equivalent to MATLAB estimate_psf_l0.m.  Uses CG on a linear system
    whose Fourier-domain form is:

        ( |F(∇I)|^2 * (1 + τ1) + λ I )  k  =  RHS

    where RHS aggregates  F(∇I)^* F(∇B)  and  τ1 * F(∇I)^* F(∇B − g).
    The auxiliary variable g takes the same soft→hard ``ℓ0−ℓ1`` shrinkage
    as in L0Restoration_HS.

    Parameters
    ----------
    blurred_x, blurred_y : derivatives of blurred input.
    latent_x, latent_y   : derivatives of latent estimate.
    weight : Tikhonov weight on k (p.lambda).  MATLAB default: 2.
    tau    : λ_data from the paper.
    k_prev : previous kernel estimate (initial guess for CG).
    theta  : θ parameter of the ℓ0−ℓ1 prior.

    Returns
    -------
    psf : updated kernel, thresholded and normalised to sum = 1.
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    psf_size = k_prev.shape
    tau1 = 2.0 * tau
    psf = k_prev.copy()
    iter_max = 5  # MATLAB default; can be reduced for speed

    for _ in range(iter_max):
        # ── g subproblem (ℓ0−ℓ1 on residual B_∇ - k*I_∇) ────────────────────
        g_h = blurred_x - fftconv(latent_x, psf)
        g_h = np.sign(g_h) * np.maximum(
            np.abs(g_h) - tau * theta / (2.0 * tau1), 0.0
        )
        g_v = blurred_y - fftconv(latent_y, psf)
        g_v = np.sign(g_v) * np.maximum(
            np.abs(g_v) - tau * theta / (2.0 * tau1), 0.0
        )
        t_h = g_h ** 2 < tau / tau1
        t_v = g_v ** 2 < tau / tau1
        g_h[t_h] = 0.0
        g_v[t_v] = 0.0

        # ── k subproblem (CG) ───────────────────────────────────────────────
        temp = np.conj(latent_xf) * fft2(blurred_x - g_h) \
             + np.conj(latent_yf) * fft2(blurred_y - g_v)
        b_f = tau1 * temp + np.conj(latent_xf) * blurred_xf \
                          + np.conj(latent_yf) * blurred_yf
        b = np.real(otf2psf(b_f, psf_size))

        p = {
            'm': (np.conj(latent_xf) * latent_xf
                  + np.conj(latent_yf) * latent_yf) * (1.0 + tau1),
            'img_size': blurred_xf.shape[:2],
            'psf_size': psf_size,
            'lambda': weight,
        }
        psf = conjgrad(psf, b, 8, 1e-5, _compute_Ax_psf, p)

        tau1 = tau1 * 2.0

    # Post-process: zero small entries, normalise
    max_val = psf.max()
    if max_val > 0:
        psf[psf < max_val * 0.05] = 0.0
        s = psf.sum()
        if s > 0:
            psf = psf / s
    return psf


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv_main  (from blind_deconv_main.m) — single scale loop
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_main(blur_B: np.ndarray,
                      k: np.ndarray,
                      lambda_data: float,
                      lambda_grad: float,
                      threshold: float,
                      opts: dict):
    """
    One-scale ESM blind deconvolution loop.

    Equivalent to MATLAB blind_deconv_main.m.

    Returns (k, lambda_data, lambda_grad, S) — updated kernel, the
    (continuation-decayed) regularisers, and the latest latent image.
    """
    dx = np.array([[-1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    dy = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    # Pad blurred image once, compute its derivatives
    H, W = blur_B.shape[:2]
    target = opt_fft_size(np.array([H, W]) + np.array(k.shape[:2]) - 1)
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]

    # conv2('valid') with dx, dy as in MATLAB (true convolution, kernel flipped)
    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    theta = opts['theta']
    xk_iter = opts['xk_iter']

    S = blur_B.copy()
    for _it in range(xk_iter):
        # 1) Latent update (enhanced sparse model)
        S = L0Restoration_HS(blur_B, k, lambda_data, lambda_grad, theta)

        # 2) Gradient selection (Cho-style salient edges)
        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        # 3) Kernel update (ℓ0−ℓ1 data-gradient prior + Tikhonov)
        k = estimate_psf_l0(Bx, By, latent_x, latent_y,
                            2.0, lambda_data, k, theta)

        # 4) Prune isolated connected components (MATLAB bwconncomp(k, 8))
        structure = np.ones((3, 3), dtype=np.int32)  # 8-connectivity
        labeled, n_comp = label(k > 0, structure=structure)
        for ii in range(1, n_comp + 1):
            mask = labeled == ii
            currsum = k[mask].sum()
            if currsum < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        s = k.sum()
        if s > 0:
            k = k / s

        # 5) Continuation on regularisers
        if lambda_data != 0:
            lambda_data = max(lambda_data / 1.1, 1e-4)
        else:
            lambda_data = 0.0
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)
        else:
            lambda_grad = 0.0

    k[k < 0] = 0.0
    s = k.sum()
    if s > 0:
        k = k / s
    return k, lambda_data, lambda_grad, S


# ═════════════════════════════════════════════════════════════════════════════
# blind_deconv  (from blind_deconv.m) — coarse-to-fine pyramid
# ═════════════════════════════════════════════════════════════════════════════

def _init_kernel(minsize: int) -> np.ndarray:
    """
    MATLAB:
        k((minsize-1)/2, (minsize-1)/2 : (minsize-1)/2+1) = 1/2
    A horizontal 2-pixel delta at the geometric centre (1-based).
    In 0-based indexing: row = (minsize-1)//2 - 1   (since MATLAB (n-1)/2
    with odd n yields an integer and refers to the 1-based row).

    For odd `minsize`:  MATLAB row (minsize-1)/2  →  0-based (minsize-1)//2 - 1?
    No — (minsize-1)/2 in MATLAB 1-based addresses the middle-ish row.
    For odd minsize, centre row 1-based is (minsize+1)/2.
    MATLAB chooses (minsize-1)/2, which is one above centre.
    We replicate exactly: 0-based row index = (minsize-1)//2 - 1 when
    (minsize-1)/2 >= 1, i.e. minsize >= 3.
    Columns: (minsize-1)/2 and (minsize-1)/2 + 1  →  0-based  a-1 and a.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    a = (minsize - 1) // 2  # MATLAB 1-based index value
    # MATLAB row index a (1-based) → 0-based a - 1
    # MATLAB col indices a, a+1 (1-based) → 0-based a - 1, a
    row = a - 1
    col_start = a - 1
    col_end = a  # inclusive in MATLAB; Python slice end-exclusive = a+1
    k[row, col_start:col_end + 1] = 0.5
    return k


def _downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:
    """
    Gaussian low-pass + bilinear resample, matching Levin/MATLAB downSmpImC.

    MATLAB:
        sig = 1/pi*ret;
        g0  = [-50:50]*2*pi;
        sf  = exp(-0.5*g0.^2*sig^2); sf = sf/sum(sf);
        csf = cumsum(sf); csf = min(csf, csf(end:-1:1));
        ii  = find(csf > 0.05);
        sf  = sf(ii);
        I   = conv2(sf, sf', I, 'valid');
        [gx,gy] = meshgrid(1:1/ret:size(I,2), 1:1/ret:size(I,1));
        sI = interp2(I, gx, gy, 'bilinear');
    """
    if ret == 1.0:
        return I

    sig = ret / np.pi
    g0 = np.arange(-50, 51) * 2 * np.pi
    sf = np.exp(-0.5 * g0 * g0 * sig * sig)
    sf = sf / sf.sum()

    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]
    # Separable: conv2(sf, sf', I, 'valid') = apply sf as row kernel then sf' as column
    # MATLAB form `conv2(h1, h2, I, 'valid')` with h1 a row vector and h2 a column
    # vector: equivalent to outer-product kernel h2*h1, applied via true 2-D
    # convolution.  We use convolve2d with the outer-product kernel.
    kern_row = sf.reshape(1, -1)          # sf
    kern_col = sf.reshape(-1, 1)          # sf'
    kern = kern_col @ kern_row            # (len(sf), len(sf)) outer product
    Ic = convolve2d(I, kern, mode='valid')

    # interp2(I, gx, gy, 'bilinear') with gx = 1 : 1/ret : cols, gy = 1 : 1/ret : rows
    Hc, Wc = Ic.shape
    # Column sample positions (MATLAB 1-based): 1 : 1/ret : Wc
    gx = np.arange(1.0, Wc + 1e-12, 1.0 / ret)
    gy = np.arange(1.0, Hc + 1e-12, 1.0 / ret)
    # Convert to 0-based for map_coordinates
    gx0 = gx - 1.0
    gy0 = gy - 1.0
    GX, GY = np.meshgrid(gx0, gy0)
    sI = map_coordinates(Ic, [GY.ravel(), GX.ravel()],
                         order=1, mode='nearest')
    return sI.reshape(gy0.size, gx0.size)


def _fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:
    """
    Exact port of the MATLAB fixsize helper: iteratively crop / zero-pad
    rows and columns to match the target size (nk1, nk2), keeping the
    heavier side intact.
    """
    k1, k2 = f.shape
    while (k1 != nk1) or (k2 != nk2):
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]
        if k1 < nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[:k1, :] = f
                f = tf
            else:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[1:k1 + 1, :] = f
                f = tf
        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]
        if k2 < nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, :k2] = f
                f = tf
            else:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, 1:k2 + 1] = f
                f = tf
        k1, k2 = f.shape
    return f


def _resizeKer(k: np.ndarray, ret_inv: float, k1: int, k2: int) -> np.ndarray:
    """
    MATLAB:
        k = imresize(k, ret);   k = max(k, 0);   k = fixsize(k, k1, k2);
        if max(k(:))>0, k = k/sum(k(:)); end

    Here `ret_inv` corresponds to MATLAB's 1/ret (the upscale factor).
    We use scipy.ndimage.zoom with cubic interpolation, which is the closest
    deterministic Python equivalent to MATLAB's default imresize (bicubic).
    """
    k = zoom(k, ret_inv, order=3, mode='nearest')
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    m = k.max()
    if m > 0:
        k = k / k.sum()
    return k


def blind_deconv(y: np.ndarray,
                 lambda_data: float,
                 lambda_grad: float,
                 opts: dict):
    """
    Coarse-to-fine blind deconvolution pyramid.

    Equivalent to MATLAB blind_deconv.m.

    Parameters
    ----------
    y : (H, W) float64 grayscale image in [0, 1].
    lambda_data, lambda_grad : ESM regularisation weights.
    opts : dict with keys
        kernel_size   : int, odd, support of the PSF
        gamma_correct : float, gamma correction exponent
        xk_iter       : int, inner iterations per scale
        k_thresh      : float, final kernel thresholding parameter
        theta         : float, θ of the ℓ0−ℓ1 prior

    Returns
    -------
    kernel : (k1, k1) estimated PSF, normalised.
    interim_latent : (H, W) latent image from the finest scale.
    """
    if opts.get('gamma_correct', 1.0) != 1.0:
        y = y ** opts['gamma_correct']

    ret = np.sqrt(0.5)
    kernel_size = opts['kernel_size']
    k_thresh = opts.get('k_thresh', 20)
    opts_with_theta = dict(opts)
    if 'theta' not in opts_with_theta:
        opts_with_theta['theta'] = 1.0

    # MATLAB: maxitr = max( floor(log(5/min(opts.kernel_size))/log(ret)), 0 );
    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)  # retv[0]=1 … retv[maxitr]=ret^maxitr
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list[k1list % 2 == 0] += 1  # force odd

    ks = None
    threshold = 0.0
    interim_latent = None

    # MATLAB: for s = num_scales:-1:1  (coarsest first, indices descending 1-based)
    # Python: iterate from coarsest (highest pyramid index = num_scales-1 0-based)
    # to finest (0).  MATLAB 1-based s maps to 0-based idx = s - 1.
    for s in range(num_scales - 1, -1, -1):
        k1 = int(k1list[s])
        k2 = k1
        cret = retv[s]

        if s == num_scales - 1:
            ks = _init_kernel(k1)
        else:
            # Upsample from previous (coarser) level.
            # MATLAB calls resizeKer(ks, 1/ret, k1, k2).
            ks = _resizeKer(ks, 1.0 / ret, k1, k2)

        ys = _downSmpImC(y, cret)

        print(f'Processing scale {s + 1}/{num_scales}; '
              f'kernel size {k1}x{k2}; image size {ys.shape[0]}x{ys.shape[1]}',
              flush=True)

        if s == num_scales - 1:
            # MATLAB: [~,~, threshold] = threshold_pxpy_v1(ys, max(size(ks)));
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        ks, lambda_data, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_data, lambda_grad, threshold, opts_with_theta
        )

        # Centre, clip negatives, normalise
        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        sk = ks.sum()
        if sk > 0:
            ks = ks / sk

        if s == 0:
            # Final kernel thresholding
            kernel = ks
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            ssum = kernel.sum()
            if ssum > 0:
                kernel = kernel / ssum
            return kernel, interim_latent

    # Should not reach here
    return ks, interim_latent


# ═════════════════════════════════════════════════════════════════════════════
# deblurring_adm_aniso  (from deblurring_adm_aniso.m) — TV-ℓ² via ADM
# ═════════════════════════════════════════════════════════════════════════════

def _computeDenominator(y: np.ndarray, k: np.ndarray):
    """
    MATLAB computeDenominator helper:
        Nomin1 = conj(F(k)) .* F(y)
        Denom1 = |F(k)|^2
        Denom2 = |F([1,-1])|^2 + |F([1;-1])|^2
    """
    sizey = y.shape
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = np.abs(psf2otf(np.array([[1.0, -1.0]]), sizey)) ** 2 \
           + np.abs(psf2otf(np.array([[1.0], [-1.0]]), sizey)) ** 2
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B: np.ndarray,
                         k: np.ndarray,
                         lambda_tv: float,
                         alpha: float = 1.0) -> np.ndarray:
    """
    Anisotropic TV deblurring via ADM / Split-Bregman.

    Equivalent to MATLAB deblurring_adm_aniso.m.  Only the alpha == 1
    branch is implemented because the actual pipeline
    (ringing_artifacts_removal) always calls with alpha = 1.
    """
    if alpha != 1.0:
        raise NotImplementedError(
            "deblurring_adm_aniso is only ported for alpha = 1 "
            "(the sole branch used by the ESM pipeline)."
        )

    # Kernel must be odd-sized
    if (k.shape[0] % 2 != 1) or (k.shape[1] % 2 != 1):
        raise ValueError('Blur kernel k must be odd-sized.')

    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = _circ_diff_x(I)
    Iy = _circ_diff_y(I)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        # alpha == 1  → anisotropic TV, simple soft-threshold
        Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
        Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)

        Wxx = _adjoint_diff_x(Wx) + _adjoint_diff_y(Wy)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = _circ_diff_x(I)
        Iy = _circ_diff_y(I)

        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# ringing_artifacts_removal  (from ringing_artifacts_removal.m)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y: np.ndarray,
                              kernel: np.ndarray,
                              lambda_tv: float,
                              lambda_l0: float,
                              weight_ring: float) -> np.ndarray:
    """
    Final non-blind deconvolution with ringing suppression.

    Equivalent to MATLAB ringing_artifacts_removal.m.  Reference:
        J. Pan, Z. Hu, Z. Su, M.-H. Yang: "Deblurring Text Images via
        L0-Regularized Intensity and Gradient Prior", CVPR 2014.

    Pipeline:
        1. TV-ℓ² deconvolution (deblurring_adm_aniso, per channel).
        2. If weight_ring > 0:
           L0 deconvolution → ring = bilateral_filter(TV − L0)
           result = TV − weight_ring * ring.
        3. Otherwise return TV result.
    """
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
        was_2d = True
    else:
        was_2d = False

    H, W, Ch = y.shape

    # Pad once
    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    y_pad = wrap_boundary_liu(y, tuple(target))
    if y_pad.ndim == 2:
        y_pad = y_pad[:, :, np.newaxis]

    # TV-ℓ² per channel
    Latent_tv = np.zeros_like(y_pad)
    for c in range(Ch):
        Latent_tv[:, :, c] = deblurring_adm_aniso(
            y_pad[:, :, c], kernel, lambda_tv, 1.0
        )
    Latent_tv = Latent_tv[:H, :W, :]

    if weight_ring == 0:
        result = Latent_tv
        if was_2d:
            return result[:, :, 0]
        return result

    # L0 non-blind deconvolution (operates on the padded multichannel image)
    if y_pad.shape[2] == 1:
        Latent_l0 = L0Restoration(y_pad[:, :, 0], kernel, lambda_l0, 2.0)
        Latent_l0 = Latent_l0[:H, :W]
        Latent_l0 = Latent_l0[:, :, np.newaxis]
    else:
        Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2.0)
        Latent_l0 = Latent_l0[:H, :W, :]

    diff = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff, 3.0, 0.1)
    # bilateral_filter returns float32; cast back for consistency
    bf_diff = np.asarray(bf_diff, dtype=np.float64)
    if bf_diff.ndim == 2 and diff.ndim == 3:
        bf_diff = bf_diff[:, :, np.newaxis]

    result = Latent_tv - weight_ring * bf_diff

    if was_2d:
        return result[:, :, 0]
    return result
