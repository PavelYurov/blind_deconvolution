import numpy as np
from scipy.signal import fftconvolve

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
