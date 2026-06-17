import numpy as np
from scipy.signal import fftconvolve
from scipy.fft import dstn, idstn
from numpy.fft import fft2, ifft2

from .utils import psf2otf


def optimize_image(
    x: np.ndarray,
    kernel: np.ndarray,
    blurred: np.ndarray,
    reg_weight: float,
    max_irls: int = 3,
    max_cg: int = 200,
    exp_a: float = 0.8,
    thr_e: float = 1.0 / 1500,
) -> np.ndarray:
    """
    Estimate the latent sharp image via IRLS with hyper-Laplacian prior.

    Parameters:
    x : np.ndarray, shape (H, W)
        Current sharp-image estimate (initialised to blurred image).
    kernel : np.ndarray, shape (kh, kw)
        Current blur-kernel estimate.
    blurred : np.ndarray, shape (H, W)
        Observed blurred image.
    reg_weight : float
        Edge-regularisation weight α.
    max_irls : int
        IRLS outer iterations.
    max_cg : int
        CG inner iterations per IRLS step.
    exp_a : float
        Hyper-Laplacian exponent *p* (0 < p ≤ 2; typical 0.5–0.8).
    thr_e : float
        Smoothing parameter ε to avoid division by zero in weights.

    Returns:
    x : np.ndarray, shape (H, W)
        Updated sharp-image estimate.
    """
    dxf = np.array([[1.0, -1.0]])       
    dyf = np.array([[1.0], [-1.0]])      


    dxf_t = dxf[::-1, ::-1]            
    dyf_t = dyf[::-1, ::-1]             

    kernel_rot = np.rot90(kernel, 2)     

    for irls_it in range(max_irls):
        dx = fftconvolve(x, dxf, mode='valid')  
        dy = fftconvolve(x, dyf, mode='valid')   

        weight_x = (thr_e + dx ** 2) ** (exp_a / 2.0 - 1.0)
        weight_y = (thr_e + dy ** 2) ** (exp_a / 2.0 - 1.0)

        b = fftconvolve(blurred, kernel_rot, mode='same')

        def _apply_A(v):
            Kv = fftconvolve(v, kernel, mode='same')
            result = fftconvolve(Kv, kernel_rot, mode='same')

            vx = fftconvolve(v, dxf, mode='valid')
            vy = fftconvolve(v, dyf, mode='valid')
            result += reg_weight * fftconvolve(
                vx * weight_x, dxf_t, mode='full')
            result += reg_weight * fftconvolve(
                vy * weight_y, dyf_t, mode='full')

            return result

        x_prev = x.copy()
        r = b - _apply_A(x)
        p = r.copy()
        rho = np.sum(r * r)

        for _ in range(max_cg):
            Ap = _apply_A(p)
            pAp = np.sum(p * Ap)
            if abs(pAp) < 1e-30:
                break

            alpha_cg = rho / pAp
            x = x + alpha_cg * p
            r = r - alpha_cg * Ap

            rho_new = np.sum(r * r)

            if np.sum((alpha_cg * p) ** 2) / max(x.size, 1) < 1e-7:
                break

            p = r + (rho_new / (rho + 1e-30)) * p
            rho = rho_new

        if np.sum((x - x_prev) ** 2) / max(x.size, 1) < 1e-7:
            break

    return x

def optimize_kernel(
    x: np.ndarray,
    kernel: np.ndarray,
    blurred: np.ndarray,
    beta: float = 3e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Parameters:
    x : np.ndarray, shape (H, W)
        Current sharp-image estimate.
    kernel : np.ndarray, shape (kh, kw)
        Current kernel estimate.
    blurred : np.ndarray, shape (H, W)
        Observed blurred image.
    beta : float
        Tikhonov regularisation weight.
    max_iter : int
        Maximum CG iterations.

    Returns:
    kernel : np.ndarray, shape (kh, kw)
        Updated kernel estimate (non-negative, sums to one).
    """
    kh, kw = kernel.shape
    bhs_y, bhs_x = kh // 2, kw // 2

    if bhs_y > 0 and bhs_x > 0:
        y_crop = blurred[bhs_y:-bhs_y, bhs_x:-bhs_x]
    else:
        y_crop = blurred.copy()

    x_rot = np.rot90(x, 2)
    k0 = kernel.copy()

    def _apply_A(k):
        """Compute  (X^T X + β I) k."""
        Xk = fftconvolve(x, k, mode='valid')
        XtXk = fftconvolve(x_rot, Xk, mode='valid')
        return XtXk + beta * k
    b = fftconvolve(x_rot, y_crop, mode='valid') + beta * k0

    r = b - _apply_A(kernel)
    d = r.copy()
    rr = np.sum(r * r)
    rr0 = rr + 1e-30

    for i in range(max_iter):
        if rr < 1e-8 * rr0:
            break

        Ad = _apply_A(d)
        dAd = np.sum(d * Ad)
        if abs(dAd) < 1e-30:
            break

        alpha_cg = rr / dAd
        kernel = kernel + alpha_cg * d

        if (i + 1) % 50 == 0:
            r = b - _apply_A(kernel)
        else:
            r = r - alpha_cg * Ad

        rr_new = np.sum(r * r)
        d = r + (rr_new / (rr + 1e-30)) * d
        rr = rr_new

    kernel = np.clip(kernel, 0.0, None)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel

def low_rank_regularization(
    kernel: np.ndarray,
    max_iter: int = 3,
    tau: float = 1e-5,
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Low-rank kernel regularisation via Iteratively Reweighted Nuclear
    Norm (IRNN) minimisation.
    Parameters:
    kernel : np.ndarray, shape (kh, kw)
        Current kernel estimate.
    max_iter : int
        Number of MM iterations (typically 3–10).
    tau : float
        Proximal parameter (smaller ⟹ solution stays closer to *k₀*).
    delta : float
        Smoothing parameter for the ``log det`` surrogate.

    Returns:
    kernel : np.ndarray
        Low-rank–regularised kernel.
    """
    X = kernel.copy()

    w = np.ones(min(X.shape))

    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        S_thresh = np.maximum(S - tau * w, 0.0)
        X = (U * S_thresh) @ Vt

        sigma = np.linalg.svd(X, compute_uv=False)
        w = 1.0 / (sigma + delta)

    return X


def fast_deconv_hyper_laplacian(
    blurred: np.ndarray,
    kernel: np.ndarray,
    lambda_: float = 3000.0,
    alpha: float = 0.5,
    beta: float = 400.0,
    max_outer: int = 50,
    max_inner: int = 1,
) -> np.ndarray:
    """
    Non-blind image deconvolution with hyper-Laplacian prior.

    Solves the MAP restoration problem ([4], Eq. (1)):
    Parameters:
    blurred : np.ndarray, shape (H, W)
        Observed blurred image.
    kernel  : np.ndarray, shape (kh, kw)
        Estimated PSF.
    lambda_ : float
        Data-fidelity weight.
    alpha   : float
        Hyper-Laplacian exponent.
        α = 1: Laplacian (L₁);  α = 2/3 or 1/2: hyper-Laplacian.
    beta    : float
        ADMM penalty (augmented Lagrangian parameter).
    max_outer : int
        Outer ADMM iterations.
    max_inner : int
        Inner iterations per ADMM step.

    Returns:
    restored : np.ndarray, shape (H, W)
    """
    H, W = blurred.shape
    g = blurred.copy()

    dx  = np.array([[1.0, -1.0]])
    dy  = np.array([[1.0], [-1.0]])
    dxt = dx[::-1, ::-1]
    dyt = dy[::-1, ::-1]

    otf_k = psf2otf(kernel, (H, W))
    Ktf   = np.conj(otf_k) * np.fft.fft2(blurred)   
    KtK   = np.abs(otf_k) ** 2                        

    Fdx = np.abs(psf2otf(dx, (H, W))) ** 2
    Fdy = np.abs(psf2otf(dy, (H, W))) ** 2
    DtD = Fdx + Fdy                                   


    gx = fftconvolve(g, dx, mode='valid')
    gy = fftconvolve(g, dy, mode='valid')

    bx = np.zeros_like(gx)
    by = np.zeros_like(gy)
    wx = gx.copy()
    wy = gy.copy()

    for _ in range(max_outer):
        for _ in range(max_inner):
            if alpha == 1.0:
                wx = _soft_threshold(gx + bx, 1.0 / beta)
                wy = _soft_threshold(gy + by, 1.0 / beta)
            else:
                wx = _hyper_laplacian_proximal(gx + bx, beta, alpha)
                wy = _hyper_laplacian_proximal(gy + by, beta, alpha)

            bx = bx + gx - wx
            by = by + gy - wy
            wx_full = fftconvolve(wx - bx, dxt, mode='full')
            wy_full = fftconvolve(wy - by, dyt, mode='full')

            numer = lambda_ * Ktf + beta * np.fft.fft2(wx_full + wy_full)
            denom = lambda_ * KtK + beta * DtD

            g = np.real(np.fft.ifft2(numer / (denom + 1e-10)))

            # Recompute gradients of g
            gx = fftconvolve(g, dx, mode='valid')
            gy = fftconvolve(g, dy, mode='valid')

    return g



def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """
    Soft-thresholding — proximal operator of the L1 norm.
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _hyper_laplacian_proximal(
    v: np.ndarray,
    beta: float,
    alpha: float,
    n_lut: int = 5000,
) -> np.ndarray:
    signs = np.sign(v)
    v_abs = np.abs(v)
    w_min = (alpha * (1.0 - alpha) / beta) ** (1.0 / (2.0 - alpha))

    v_max = float(v_abs.max()) + 1.0
    w_max = v_max  

    w_lut = np.linspace(w_min, w_max, n_lut)
    v_lut = w_lut + (alpha / beta) * w_lut ** (alpha - 1.0)

    cost_w = (beta / 2.0) * (w_lut - v_lut) ** 2 + w_lut ** alpha
    cost_0 = (beta / 2.0) * v_lut ** 2
    valid  = cost_w < cost_0

    if not np.any(valid):
        return np.zeros_like(v)

    first_valid = int(np.argmax(valid))
    v_thresh = v_lut[first_valid] if first_valid > 0 else v_lut[0]

    v_mono = v_lut[valid]
    w_mono = w_lut[valid]

    w_out = np.zeros_like(v_abs)
    mask = v_abs >= v_thresh

    if np.any(mask):
        w_out[mask] = np.interp(v_abs[mask], v_mono, w_mono)

    return signs * w_out


# ═════════════════════════════════════════════════════════════════════════════
# Optimal FFT size
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None

def _build_opt_fft_lut(max_n: int = 4096) -> list:
    """Build look-up table mapping n → smallest efficient FFT length ≥ n."""
    from itertools import count
    efficient = set()
    for i2 in count():
        v2 = 2 ** i2
        if v2 > max_n * 2:
            break
        for i3 in count():
            v23 = v2 * 3 ** i3
            if v23 > max_n * 2:
                break
            for i5 in count():
                v235 = v23 * 5 ** i5
                if v235 > max_n * 2:
                    break
                for i7 in count():
                    v = v235 * 7 ** i7
                    if v > max_n * 2:
                        break
                    efficient.add(v)
    efficient = sorted(efficient)
    lut = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        for e in efficient:
            if e >= n:
                lut[n] = e
                break
    return lut


def opt_fft_size(n) -> np.ndarray:
    """Optimal FFT data length(s) — smallest efficient size ≥ n."""
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
            m.flat[i] = int(nn)
    if scalar_input:
        return int(m.flat[0])
    return m


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu  (Liu & Jia ICIP 2008, Cho implementation)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """Solve Laplace eq. with Dirichlet BC via DST-I (Poisson solver)."""
    H, W = boundary_image.shape
    bi = boundary_image.copy()
    bi[1:-1, 1:-1] = 0.0

    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H-1, 1:W-1] = (
        -4.0 * bi[1:H-1, 1:W-1]
        + bi[1:H-1, 2:W] + bi[1:H-1, 0:W-2]
        + bi[0:H-2, 1:W-1] + bi[2:H, 1:W-1]
    )
    f1 = -f_bp
    f2 = f1[1:H-1, 1:W-1]
    f2sin = dstn(f2, type=1)

    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom
    img_tt = idstn(f3, type=1)

    result = bi.copy()
    result[1:H-1, 1:W-1] = img_tt
    return result


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """
    Pad image so boundaries are circularly smooth for FFT-based deconv.
    Based on Liu & Jia (ICIP 2008), Cho implementation.
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H
    W_w = W_out - W

    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)

    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]

        # --- vertical wrap ---
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]
        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (
            (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0])
        r_A[alpha:alpha + H_w, -1] = (
            (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1])

        A2 = _solve_min_laplacian(
            r_A[alpha - 1: alpha + H_w + 1, :])

        # --- horizontal wrap ---
        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]
        if W_w > 1:
            b = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            b = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (
            (1 - b) * r_B[0, alpha - 1] + b * r_B[0, -alpha])
        r_B[-1, alpha:alpha + W_w] = (
            (1 - b) * r_B[-1, alpha - 1] + b * r_B[-1, -alpha])

        B2 = _solve_min_laplacian(
            r_B[:, alpha - 1: alpha + W_w + 1])

        # --- assemble ---
        ret[:H, :W, ch] = HG
        ret[H:, :W, ch] = A2[1:-1, :]
        ret[:H, W:, ch] = B2[:, 1:-1]

        if H_w > 0 and W_w > 0:
            r_C = np.zeros((H_w + 2, W_w + 2), dtype=np.float64)
            # Boundary values with periodic wrap-around at the far corner
            # Top row: [ret[H-1, W-1], ..., ret[H-1, W_out-1], ret[H-1, 0]]
            r_C[0, :-1] = ret[H - 1, W - 1:, ch]
            r_C[0, -1]  = ret[H - 1, 0, ch]
            # Bottom row: wraps to row 0
            r_C[-1, :-1] = ret[0, W - 1:, ch]
            r_C[-1, -1]  = ret[0, 0, ch]
            # Left col: [ret[H-1, W-1], ..., ret[H_out-1, W-1], ret[0, W-1]]
            r_C[:-1, 0]  = ret[H - 1:, W - 1, ch]
            r_C[-1, 0]   = ret[0, W - 1, ch]
            # Right col: wraps to col 0
            r_C[:-1, -1] = ret[H - 1:, 0, ch]
            r_C[-1, -1]  = ret[0, 0, ch]
            C2 = _solve_min_laplacian(r_C)
            ret[H:, W:, ch] = C2[1:-1, 1:-1]

    if Ch == 1:
        return ret[:, :, 0]
    return ret


# ═════════════════════════════════════════════════════════════════════════════
# TV deblurring (ADM anisotropic — Split Bregman)
# ═════════════════════════════════════════════════════════════════════════════

def _computeDenominator(B, k):
    """Pre-compute frequency-domain terms for ADM TV deblurring."""
    m, n = B.shape
    otf_k = psf2otf(k, (m, n))
    Nomin1 = np.conj(otf_k) * fft2(B)
    Denom1 = np.abs(otf_k) ** 2

    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    Denom2 = (np.abs(psf2otf(dx, (m, n))) ** 2 +
              np.abs(psf2otf(dy, (m, n))) ** 2)
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B, k, lambda_tv, alpha):
    """TV-ℓ² deblurring via ADM/Split Bregman with anisotropic TV."""
    beta = 1.0 / lambda_tv
    beta_min = 0.001
    m, n = B.shape
    I = B.copy()
    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
        Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)
        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# L0 gradient restoration
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """Image restoration with L0 gradient prior."""
    H_orig, W_orig = Im.shape[0], Im.shape[1]
    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1)
    Im_w = wrap_boundary_liu(Im, tuple(target_size))

    if Im_w.ndim == 2:
        Im_w = Im_w[:, :, np.newaxis]
    N, M, D = Im_w.shape

    S = Im_w.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    otfFx = psf2otf(fx, (N, M))
    otfFy = psf2otf(fy, (N, M))
    KER = psf2otf(kernel, (N, M))
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    beta_val = 2 * lambda_grad
    while beta_val < betamax:
        Denormin = Den_KER + beta_val * Denormin2

        h = np.concatenate([np.diff(S, axis=1),
                            S[:, 0:1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, axis=0),
                            S[0:1, :, :] - S[-1:, :, :]], axis=0)

        grad_sq = np.sum(h ** 2 + v ** 2, axis=2)
        t = grad_sq < lambda_grad / beta_val
        t3 = np.tile(t[:, :, np.newaxis], (1, 1, D))
        h[t3] = 0
        v[t3] = 0

        Normin2 = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                  -np.diff(h, axis=1)], axis=1)
        Normin2 += np.concatenate([v[-1:, :, :] - v[0:1, :, :],
                                   -np.diff(v, axis=0)], axis=0)

        FS = (Normin1 + beta_val * fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))
        beta_val *= kappa

    S = S[:H_orig, :W_orig, :]
    if D == 1:
        S = S[:, :, 0]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# Bilateral filter (spatial × photometric Gaussian)
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size, sigma):
    """2-D Gaussian kernel (like MATLAB fspecial('gaussian', ...))."""
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    h = np.outer(g, g)
    return h / h.sum()


def bilateral_filter(img, sigma_s, sigma):
    """Bilateral filter for grayscale images."""
    was_2d = img.ndim == 2
    if was_2d:
        img = img[:, :, np.newaxis]
    h, w, d = img.shape
    img = img.astype(np.float32)
    lab = img.copy()
    sigma = sigma * np.sqrt(d)
    fr = int(np.ceil(sigma_s * 3))

    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    p_lab = np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')

    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)
    spatial_weight = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * sigma

    for yy in range(-fr, fr + 1):
        for xx in range(-fr, fr + 1):
            w_s = spatial_weight[yy + fr, xx + fr]
            n_img = p_img[fr + yy:fr + yy + h, fr + xx:fr + xx + w, :]
            n_lab = p_lab[fr + yy:fr + yy + h, fr + xx:fr + xx + w, :]
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
# Ringing artifacts removal  (Pan et al. CVPR 2014)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv=1e-3,
                              lambda_l0=2e-3, weight_ring=1.0):
    """
    Non-blind deconvolution with ringing suppression.

    Uses TV deconv + L0 deconv + bilateral filter on their difference
    to identify and subtract ringing artifacts.

    Parameters
    ----------
    y           : (H, W) blurred image (single channel, float [0,1])
    kernel      : blur kernel
    lambda_tv   : TV regularisation weight
    lambda_l0   : L0 gradient prior weight
    weight_ring : ringing suppression strength (0 = TV only)

    Returns
    -------
    result : (H, W) deblurred image
    """
    H, W = y.shape[:2]
    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)

    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)
    result = Latent_tv - weight_ring * bf_diff
    return result
