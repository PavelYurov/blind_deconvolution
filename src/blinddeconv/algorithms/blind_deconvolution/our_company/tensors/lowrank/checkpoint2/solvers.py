"""
Solver functions for Low-Rank Blind Deconvolution.

Contains the core optimisation routines for the sub-problems arising
in the alternating-minimisation framework:

1. **Image estimation** — IRLS (Iteratively Reweighted Least Squares)
   with hyper-Laplacian edge prior, inner solve via CG.
2. **Kernel estimation** — Conjugate-Gradient (CG) descent with
   Tikhonov regularisation and projection onto the feasible set
   {k ≥ 0, Σk = 1}.
3. **Low-rank kernel regularisation** — inexact Augmented Lagrangian
   Method (iALM) for Robust PCA (K = L + S) with SVT for the
   low-rank component and GST for the sparse component.
4. **Non-blind deconvolution** — Split-Bregman / ADMM with a
   hyper-Laplacian gradient prior; proximal operator via GST
   (Zuo et al. 2013).

References
----------
[1] Li, S., Chu, W., & Kuo, C.-C.J. "Understanding kernel size in
    blind deconvolution." WACV, 2019.
    GitHub: https://github.com/lisiyaoATbnu/low_rank_kernel
[2] Ren, D., et al. "Image Deblurring via Enhanced Low Rank Prior."
    IEEE TIP, vol. 25, no. 7, pp. 3426–3437, 2016.
[3] Krishnan, D., Tay, T., & Fergus, R. "Blind deconvolution using a
    normalized sparsity measure." CVPR, 2011.
[4] Krishnan, D. & Fergus, R. "Fast Image Deconvolution using
    Hyper-Laplacian Priors." NIPS, 2009.
[5] Yang, J., et al. "Hyper-Laplacian Regularized Non-local Low-rank
    Prior for Blind Image Deblurring." IEEE Access, 2020.
[6] Dong, J., et al. "Multi-image blind deconvolution using low-rank
    representation." Neurocomputing, vol. 259, pp. 227–236, 2017.
    GitHub: https://github.com/crewleader/BlindDeconvolutionLowRank
[7] Zuo, W., et al. "A Generalized Iterated Shrinkage Algorithm for
    Non-convex Sparse Coding." ICCV, 2013.
[8] Lin, Z., Chen, M. & Ma, Y. "The Augmented Lagrange Multiplier
    Method for Exact Recovery of Corrupted Low-Rank Matrices."
    arXiv:1009.5055, 2010.
"""

import numpy as np
from scipy.signal import fftconvolve

from .utils import psf2otf


# ======================================================================
#  1.  IMAGE ESTIMATION  (IRLS + CG)
# ======================================================================

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

    Solves the MAP problem ([6], Sec. 3; [5], Sec. III-C):

    .. math::

        \\min_x\\;
            \\|x \\ast k - y\\|^2
          + \\alpha \\sum_i
              \\bigl(\\varepsilon + |\\nabla_i x|^2\\bigr)^{p/2}

    via IRLS (Iteratively Reweighted Least Squares).  At each outer
    iteration the exponent is linearised into spatially-varying
    weights, converting the problem into a *weighted* least-squares
    that is solved exactly by Conjugate Gradient (CG).

    **IRLS outer loop** (``solve_image_irls.m`` from [6]):

    .. math::

        w_i^{(t)} =
            \\bigl(\\varepsilon + (\\nabla_i x^{(t)})^2\\bigr)^{p/2 - 1}

    **CG inner loop** (``solve_image_L2_w.m`` from [6]):

    Solves the linear system

    .. math::

        \\bigl(K^\\top K
             + \\alpha\\, D^\\top W^{(t)} D\\bigr)\\, x
        = K^\\top y

    where *D* is the discrete gradient operator and *W* is the
    diagonal weight matrix.

    Parameters
    ----------
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

    Returns
    -------
    x : np.ndarray, shape (H, W)
        Updated sharp-image estimate.
    """
    # Forward-difference gradient operators  ([6], solve_image_irls.m)
    dxf = np.array([[1.0, -1.0]])        # horizontal  (1×2)
    dyf = np.array([[1.0], [-1.0]])      # vertical    (2×1)

    # Adjoint operators (rot180):  D_x^T, D_y^T
    dxf_t = dxf[::-1, ::-1]             # [-1, 1]     (1×2)
    dyf_t = dyf[::-1, ::-1]             # [[-1],[1]]  (2×1)

    kernel_rot = np.rot90(kernel, 2)     # K^T (adjoint of K)

    for irls_it in range(max_irls):
        # ---- Compute IRLS weights ------------------------------------
        # [6], solve_image_irls.m:
        #   weight_x = (thr_e + dx.^2).^(exp_a/2 - 1)
        dx = fftconvolve(x, dxf, mode='valid')   # (H, W-1)
        dy = fftconvolve(x, dyf, mode='valid')   # (H-1, W)

        weight_x = (thr_e + dx ** 2) ** (exp_a / 2.0 - 1.0)
        weight_y = (thr_e + dy ** 2) ** (exp_a / 2.0 - 1.0)

        # ---- CG for  (K^T K + α D^T W D) x  =  K^T y ---------------
        # Right-hand side:  b = K^T y
        # [6], solve_image_L2_w.m:  b = conv2(I, rfilt1)
        b = fftconvolve(blurred, kernel_rot, mode='same')

        # Linear operator  A·v = K^T K v + α D^T W D v
        def _apply_A(v):
            # K^T K v   (data-fidelity Hessian)
            Kv = fftconvolve(v, kernel, mode='same')
            result = fftconvolve(Kv, kernel_rot, mode='same')

            # D^T W D v  (edge-preserving regularisation)
            # [6], solve_image_L2_w.m:
            #   Ax += we * conv2(conv2(x, dxf, 'valid') .* weight_x, drxf)
            vx = fftconvolve(v, dxf, mode='valid')
            vy = fftconvolve(v, dyf, mode='valid')
            result += reg_weight * fftconvolve(
                vx * weight_x, dxf_t, mode='full')
            result += reg_weight * fftconvolve(
                vy * weight_y, dyf_t, mode='full')

            return result

        # CG iteration  ([6], solve_image_L2_w.m)
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

            # Convergence check from [6]:
            #   sum((alpha*p).^2)/numel(p) < thres
            if np.sum((alpha_cg * p) ** 2) / max(x.size, 1) < 1e-7:
                break

            p = r + (rho_new / (rho + 1e-30)) * p
            rho = rho_new

        # IRLS convergence check  ([6], solve_image_irls.m)
        if np.sum((x - x_prev) ** 2) / max(x.size, 1) < 1e-7:
            break

    return x


# ======================================================================
#  2.  KERNEL ESTIMATION  (Conjugate Gradient + Projection)
# ======================================================================

def optimize_kernel(
    x: np.ndarray,
    kernel: np.ndarray,
    blurred: np.ndarray,
    beta: float = 3e-3,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Estimate the blur kernel via CG with Tikhonov regularisation and
    projection onto the PSF feasible set {k ≥ 0, Σk = 1}.

    Solves the quadratic sub-problem ([6], ``solve_psf_constrained.m``):

    .. math::

        \\min_k\\;
            \\|x \\ast k - y\\|^2_{\\text{valid}}
          + \\beta\\,\\|k - k_0\\|^2

    where the residual is evaluated over the valid convolution domain
    (interior pixels), avoiding boundary artefacts.  After the CG
    solution, the kernel is projected onto the non-negative simplex.

    The Tikhonov term β‖k − k₀‖² anchors the solution to the
    previous estimate, preventing drift in ill-conditioned settings
    (analogous to the ``we`` parameter in ``solve_psf_constrained.m``).

    Parameters
    ----------
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

    Returns
    -------
    kernel : np.ndarray, shape (kh, kw)
        Updated kernel estimate (non-negative, sums to one).

    Notes
    -----
    The valid-domain approach follows [6], where the convolution
    matrix is built from ``im2col_sliding``.  Here we use the
    equivalent convolution operators for memory efficiency:

    - Forward:  X · k  =  ``fftconvolve(x, k, 'valid')``
    - Adjoint:  X^T · v  =  ``fftconvolve(rot90(x,2), v, 'valid')``
    """
    kh, kw = kernel.shape
    bhs_y, bhs_x = kh // 2, kw // 2

    # Crop observation to the valid convolution region
    # [6], solve_psf_constrained.m:  I = adjust_size(I, size(filt1)-1)
    if bhs_y > 0 and bhs_x > 0:
        y_crop = blurred[bhs_y:-bhs_y, bhs_x:-bhs_x]
    else:
        y_crop = blurred.copy()

    x_rot = np.rot90(x, 2)
    k0 = kernel.copy()

    # ------------------------------------------------------------------
    # Normal equations:  (X^T X + β I) k  =  X^T y_crop + β k₀
    # ------------------------------------------------------------------
    def _apply_A(k):
        """Compute  (X^T X + β I) k."""
        Xk = fftconvolve(x, k, mode='valid')
        XtXk = fftconvolve(x_rot, Xk, mode='valid')
        return XtXk + beta * k

    # Right-hand side:  b = X^T y + β k₀
    b = fftconvolve(x_rot, y_crop, mode='valid') + beta * k0

    # CG initialisation
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

        # Residual recompute every 50 steps for numerical stability
        # (following [1], optimizek.m)
        if (i + 1) % 50 == 0:
            r = b - _apply_A(kernel)
        else:
            r = r - alpha_cg * Ad

        rr_new = np.sum(r * r)
        d = r + (rr_new / (rr + 1e-30)) * d
        rr = rr_new

    # ---- Projection onto  {k ≥ 0,  Σ k = 1}  -----------------------
    kernel = np.clip(kernel, 0.0, None)
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


# ======================================================================
#  3.  LOW-RANK KERNEL REGULARISATION  (iALM RPCA)
# ======================================================================

def low_rank_regularization(
    kernel: np.ndarray,
    max_iter: int = 50,
    lambda_s=None,
    p: float = 0.5,
    rho: float = 1.5,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Low-rank kernel regularisation via inexact Augmented Lagrangian
    Method (iALM) for Robust PCA.

    Decomposes the kernel matrix K = L + S, where L is low-rank
    (structural blur pattern) and S is sparse (estimation noise),
    and returns the cleaned low-rank component L.

    Solves ([6], Algorithm 8; [8], Algorithm 6):

    .. math::

        \\min_{L,S}\\; \\|L\\|_* + \\lambda\\,\\|S\\|_p^p
        \\quad\\text{s.t.}\\quad K = L + S

    Sub-problems per ALM iteration:

    - **L-update** (SVT, §3.4):
      :math:`L \\leftarrow U\\,\\max(\\Sigma - 1/\\mu,\\,0)\\,V^\\top`
      applied to :math:`K - S + Y/\\mu`.
    - **S-update** (GST, [7]):
      :math:`S \\leftarrow \\mathrm{GST}_{\\lambda/\\mu}^{p}
      (K - L + Y/\\mu)`.
    - **Dual update**
      :math:`Y \\leftarrow Y + \\mu(K - L - S)`.
    - **Penalty growth**
      :math:`\\mu \\leftarrow \\min(\\rho\\,\\mu,\\;\\bar{\\mu})`.

    Parameters
    ----------
    kernel : np.ndarray, shape (kh, kw)
        Current kernel estimate.
    max_iter : int
        Maximum ALM iterations.
    lambda_s : float or None
        Sparse-penalty weight λ.  ``None`` ⟹ auto 1/√max(kh, kw).
    p : float
        Lp exponent for the sparse term (0 < p ≤ 1).
    rho : float
        Penalty growth factor ρ > 1.
    tol : float
        Convergence tolerance (primal residual).

    Returns
    -------
    kernel : np.ndarray
        Low-rank–regularised kernel (the L component of RPCA).

    References
    ----------
    [6] Dong et al. (Neurocomputing 2017), Sec. 4.2.1, Algorithm 8.
    [7] Zuo et al. (ICCV 2013) — GST.
    [8] Lin et al. (arXiv 2010), Algorithm 6 — iALM for RPCA.
    """
    K = kernel.astype(np.float64)
    m, n = K.shape

    # Default λ: standard RPCA choice (Candès & Recht)
    if lambda_s is None:
        lambda_s = 1.0 / np.sqrt(max(m, n))

    norm_fro = np.linalg.norm(K, 'fro')
    if norm_fro < 1e-12:
        return K.copy()

    norm_2 = np.linalg.norm(K, ord=2)
    norm_inf = np.max(np.abs(K))

    # Initialisation ([8], Algorithm 6)
    J = max(norm_2, norm_inf / (lambda_s + 1e-30))
    Y = K / (J + 1e-30)
    S = np.zeros_like(K)
    mu = 1.25 / (norm_2 + 1e-30)
    mu_max = mu * 1e7

    for _ in range(max_iter):
        # ---- L update: SVT_{1/μ}(K − S + Y/μ) ----------------------
        Z_L = K - S + Y / mu
        U, sigma, Vt = np.linalg.svd(Z_L, full_matrices=False)
        sigma_t = np.maximum(sigma - 1.0 / mu, 0.0)
        L = (U * sigma_t) @ Vt

        # ---- S update: GST_{λ/μ}^{p}(K − L + Y/μ) ------------------
        Z_S = K - L + Y / mu
        S = _generalized_soft_threshold(Z_S, lambda_s / mu, p)

        # ---- Dual / penalty update ----------------------------------
        residual = K - L - S
        Y = Y + mu * residual
        mu = min(rho * mu, mu_max)

        # ---- Convergence check --------------------------------------
        if np.linalg.norm(residual, 'fro') / norm_fro < tol:
            break

    return L


# ======================================================================
#  4.  NON-BLIND DECONVOLUTION  (Split Bregman / ADMM)
# ======================================================================

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

    .. math::

        \\min_g\\;
            \\frac{\\lambda}{2}\\|g \\ast k - f\\|^2
          + \\sum_i \\|\\nabla_i g\\|^\\alpha

    via Split-Bregman (ADMM) with auxiliary-variable splitting.
    The augmented Lagrangian reads:

    .. math::

        \\mathcal{L}(g, w, b)
          = \\frac{\\lambda}{2}\\|g \\ast k - f\\|^2
          + \\sum |w_i|^\\alpha
          + \\frac{\\beta}{2}\\|w - \\nabla g - b\\|^2

    **Sub-problems:**

    - *g*-update: solved in the Fourier domain (closed-form,
      [4] Eq. (8)).
    - *w*-update: element-wise proximal operator of |·|^α via
      Generalised Soft Thresholding (GST; [7]).
    - *b*-update: Bregman (dual) gradient ascent.

    Parameters
    ----------
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

    Returns
    -------
    restored : np.ndarray, shape (H, W)

    References
    ----------
    [4] Krishnan & Fergus (NIPS 2009), fast_deconv_bregman.m from [1].
    [5] Yang et al. (IEEE Access 2020), Sec. III-C.
    """
    H, W = blurred.shape
    g = blurred.copy()

    # Difference operators
    dx  = np.array([[1.0, -1.0]])
    dy  = np.array([[1.0], [-1.0]])
    dxt = dx[::-1, ::-1]
    dyt = dy[::-1, ::-1]

    # ----- Precompute spectral constants ([4], Eq. (8)) ----------------
    otf_k = psf2otf(kernel, (H, W))
    Ktf   = np.conj(otf_k) * np.fft.fft2(blurred)   # K^H f
    KtK   = np.abs(otf_k) ** 2                        # |K|²

    Fdx = np.abs(psf2otf(dx, (H, W))) ** 2
    Fdy = np.abs(psf2otf(dy, (H, W))) ** 2
    DtD = Fdx + Fdy                                   # |D_x|² + |D_y|²

    # Current gradients
    gx = fftconvolve(g, dx, mode='valid')
    gy = fftconvolve(g, dy, mode='valid')

    # Bregman variables
    bx = np.zeros_like(gx)
    by = np.zeros_like(gy)
    wx = gx.copy()
    wy = gy.copy()

    for _ in range(max_outer):
        for _ in range(max_inner):

            # === w-sub-problem ========================================
            # prox_{|·|^α / β}(∇g + b)  via GST ([7])
            wx = _generalized_soft_threshold(gx + bx, 1.0 / beta, alpha)
            wy = _generalized_soft_threshold(gy + by, 1.0 / beta, alpha)

            # === b-update (Bregman iteration) =========================
            bx = bx + gx - wx
            by = by + gy - wy

            # === g-sub-problem (Fourier closed-form) ==================
            # [4], Eq. (8):
            #   G = (λ K^H f  + β D^H(w − b)) / (λ|K|² + β|D|²)
            wx_full = fftconvolve(wx - bx, dxt, mode='full')
            wy_full = fftconvolve(wy - by, dyt, mode='full')

            numer = lambda_ * Ktf + beta * np.fft.fft2(wx_full + wy_full)
            denom = lambda_ * KtK + beta * DtD

            g = np.real(np.fft.ifft2(numer / (denom + 1e-10)))

            # Recompute gradients of g
            gx = fftconvolve(g, dx, mode='valid')
            gy = fftconvolve(g, dy, mode='valid')

    return g


# ------------------------------------------------------------------
#  Proximal operators (private helpers)
# ------------------------------------------------------------------

def _soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    """
    Soft-thresholding — proximal operator of the L₁ norm.

    .. math::
        \\operatorname{prox}_{t |\\cdot|}(x)
            = \\operatorname{sign}(x)\\,\\max(|x| - t,\\, 0)

    See Parikh & Boyd (2014), Sec. 6.5.2.
    """
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def _generalized_soft_threshold(
    v: np.ndarray,
    lam: float,
    p: float = 0.5,
    max_iter: int = 10,
) -> np.ndarray:
    """
    Generalised Soft Thresholding (GST) — Algorithm 3 from [7].

    Solves the element-wise proximal problem:

    .. math::
        w^\\star = \\arg\\min_w\\;
            \\lambda\\,|w|^p + \\tfrac{1}{2}(w - v)^2

    Closed-form solutions are used for p = 1 (standard soft
    thresholding) and p = 1/2 (Theorem 1 in [7]).  For other
    0 < p < 1, the Generalised Iterated Shrinkage Algorithm
    (GISA) is applied.

    Parameters
    ----------
    v     : np.ndarray
        Input array.
    lam   : float
        Regularisation weight λ > 0.
    p     : float
        Exponent (0 < p ≤ 1).
    max_iter : int
        Maximum GISA iterations (only for p ∉ {0.5, 1}).

    Returns
    -------
    w : np.ndarray
        Proximal solution, same shape as *v*.

    References
    ----------
    [7] Zuo, W., Meng, D., Zhang, L., Feng, X. & Zhang, D.
        "A Generalized Iterated Shrinkage Algorithm for Non-convex
        Sparse Coding."  ICCV, 2013.
    """
    if lam <= 0:
        return v.copy()

    signs = np.sign(v)
    va = np.abs(v)
    x = np.zeros_like(v)

    if abs(p - 1.0) < 1e-12:
        # ---- p = 1: standard soft thresholding -----------------------
        x = np.maximum(va - lam, 0.0)

    elif abs(p - 0.5) < 1e-12:
        # ---- p = 1/2: closed-form (Theorem 1, [7]) ------------------
        # Threshold: τ_{1/2} = (3/2)(2λ)^{2/3}
        threshold = 1.5 * (2.0 * lam) ** (2.0 / 3.0)
        mask = va > threshold
        if np.any(mask):
            vm = va[mask]
            phi = np.arccos(
                (lam / 4.0) * (vm / 3.0) ** (-1.5)
            )
            x[mask] = (2.0 / 3.0) * vm * (
                1.0 + np.cos(2.0 * np.pi / 3.0 - 2.0 * phi / 3.0)
            )

    else:
        # ---- General 0 < p < 1: GISA (iterative) --------------------
        x_it = va.copy()
        for _ in range(max_iter):
            nz = x_it > 1e-15
            w = np.full_like(x_it, np.inf)
            w[nz] = lam * p * x_it[nz] ** (p - 1.0)
            x_new = np.maximum(va - w, 0.0)
            # Non-convex safeguard: keep 0 when it has lower cost
            cost_x = lam * np.abs(x_new) ** p + 0.5 * (x_new - va) ** 2
            cost_0 = 0.5 * va ** 2
            x_new[cost_x >= cost_0] = 0.0

            if np.sum((x_new - x_it) ** 2) < 1e-12 * (
                np.sum(x_it ** 2) + 1e-30
            ):
                x_it = x_new
                break
            x_it = x_new
        x = x_it

    return signs * x
