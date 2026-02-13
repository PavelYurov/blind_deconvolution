"""
Solver functions for Low-Rank Blind Deconvolution.

Contains the core optimisation routines for the sub-problems arising
in the alternating-minimisation framework:

1. **Image estimation** — IRLS (Iteratively Reweighted Least Squares)
   with hyper-Laplacian edge prior, inner solve via CG.
2. **Kernel estimation** — Conjugate-Gradient (CG) descent with
   Tikhonov regularisation and projection onto the feasible set
   {k ≥ 0, Σk = 1}.
3. **Low-rank kernel regularisation** — Iteratively Reweighted Nuclear
   Norm (IRNN) minimisation using the ``log det`` surrogate for rank.
4. **Non-blind deconvolution** — Split-Bregman / ADMM with a
   hyper-Laplacian gradient prior (Krishnan & Fergus, NIPS 2009).

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
#  3.  LOW-RANK KERNEL REGULARISATION  (IRNN)
# ======================================================================

def low_rank_regularization(
    kernel: np.ndarray,
    max_iter: int = 3,
    tau: float = 1e-5,
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Low-rank kernel regularisation via Iteratively Reweighted Nuclear
    Norm (IRNN) minimisation.

    Solves the proximal problem ([1], optimizerank_new.m; [2], Eq. (12)):

    .. math::

        \\min_k\\;
            \\frac{1}{2\\tau}\\|k - k_0\\|_F^2
          + \\operatorname{rank}(k)

    using the ``log det`` surrogate for rank:

    .. math::

        \\operatorname{rank}(K) \\approx
            \\log\\det\\bigl(\\Sigma(K) + \\delta I\\bigr)
          = \\sum_i \\log(\\sigma_i + \\delta)

    This is iteratively linearised (Majorise–Minimise), yielding a
    *weighted* nuclear-norm problem at each step whose solution is
    obtained via SVD soft-thresholding:

    .. math::

        K^{(t+1)} = U\\,\\max\\bigl(\\Sigma - \\tau\\,\\mathrm{diag}(w),
                                     0\\bigr)\\,V^\\top

    with weights  w_i = 1 / (σ_i^{(t)} + δ).

    Parameters
    ----------
    kernel : np.ndarray, shape (kh, kw)
        Current kernel estimate.
    max_iter : int
        Number of MM iterations (typically 3–10).
    tau : float
        Proximal parameter (smaller ⟹ solution stays closer to *k₀*).
    delta : float
        Smoothing parameter for the ``log det`` surrogate.

    Returns
    -------
    kernel : np.ndarray
        Low-rank–regularised kernel.

    References
    ----------
    [1] Li et al. (WACV 2019), optimizerank_new.m.
    [2] Ren et al. (IEEE TIP 2016), Sec. III-B: "Enhanced low rank
        prior."
    [5] Yang et al. (IEEE Access 2020), Sec. III-A.
    Weighted nuclear norm: Gu, S., et al. "Weighted nuclear norm
    minimization with application to image denoising." CVPR, 2014.
    """
    X = kernel.copy()

    # Uniform initial weights (before first SVD)
    w = np.ones(min(X.shape))

    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Weighted soft-thresholding of singular values
        # Eq. (8) in [2]: S_thresh = max(Σ − τ·diag(w), 0)
        S_thresh = np.maximum(S - tau * w, 0.0)
        X = (U * S_thresh) @ Vt

        # Update weights:  w_i = 1/(σ_i + δ)
        # Captures the derivative of  log(σ + δ)
        sigma = np.linalg.svd(X, compute_uv=False)
        w = 1.0 / (sigma + delta)

    return X


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
    - *w*-update: element-wise proximal operator of |·|^α.
      For α = 1 this is soft-thresholding; for α < 1 a lookup table
      is used ([4], Sec. 3.1).
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
            # prox_{|·|^α / β}(∇g + b)
            if alpha == 1.0:
                # Soft-thresholding  ([4], Eq. (6))
                wx = _soft_threshold(gx + bx, 1.0 / beta)
                wy = _soft_threshold(gy + by, 1.0 / beta)
            else:
                # Hyper-Laplacian proximal  ([4], Sec. 3.1)
                wx = _hyper_laplacian_proximal(gx + bx, beta, alpha)
                wy = _hyper_laplacian_proximal(gy + by, beta, alpha)

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


def _hyper_laplacian_proximal(
    v: np.ndarray,
    beta: float,
    alpha: float,
    n_lut: int = 5000,
) -> np.ndarray:
    """
    Proximal operator for the hyper-Laplacian penalty |w|^α  (0 < α < 1).

    Solves element-wise:

    .. math::
        w^\\star = \\arg\\min_w\\;
            \\frac{\\beta}{2}(w - v)^2 + |w|^\\alpha

    using a lookup-table (LUT) approach for efficiency.

    For α < 1 the problem is *non-convex*: below a critical |v| the
    global minimum is *w = 0*; above it the solution jumps to a
    non-trivial branch.  The LUT is built on the monotonically
    increasing portion of the mapping  w ↦ v(w), ensuring a well-defined
    inverse.

    Parameters
    ----------
    v     : np.ndarray
    beta  : float    — quadratic penalty weight
    alpha : float    — hyper-Laplacian exponent (0 < α < 1)
    n_lut : int      — lookup-table resolution

    Returns
    -------
    w : np.ndarray

    References
    ----------
    [4] Krishnan & Fergus (NIPS 2009), Sec. 3.1,
        ``solve_image_bregman.m`` from [1].
    """
    signs = np.sign(v)
    v_abs = np.abs(v)

    # ------------------------------------------------------------------
    # Turning point of the mapping  v(w) = w + (α/β) w^{α−1}
    #   v'(w) = 0  ⟹  w_min = [α(1−α)/β]^{1/(2−α)}
    # We build the LUT on the *increasing* branch  w > w_min.
    # ------------------------------------------------------------------
    w_min = (alpha * (1.0 - alpha) / beta) ** (1.0 / (2.0 - alpha))

    v_max = float(v_abs.max()) + 1.0
    w_max = v_max   # generous upper bound

    w_lut = np.linspace(w_min, w_max, n_lut)
    # Optimality condition:  v = w + (α/β) w^{α−1}
    v_lut = w_lut + (alpha / beta) * w_lut ** (alpha - 1.0)

    # Cost comparison:  f(w*) vs f(0) = β/2 · v²
    cost_w = (beta / 2.0) * (w_lut - v_lut) ** 2 + w_lut ** alpha
    cost_0 = (beta / 2.0) * v_lut ** 2
    valid  = cost_w < cost_0

    if not np.any(valid):
        return np.zeros_like(v)

    # Threshold |v| below which w = 0 is globally optimal
    first_valid = int(np.argmax(valid))
    v_thresh = v_lut[first_valid] if first_valid > 0 else v_lut[0]

    # Monotone sub-table for interpolation
    v_mono = v_lut[valid]
    w_mono = w_lut[valid]

    w_out = np.zeros_like(v_abs)
    mask = v_abs >= v_thresh

    if np.any(mask):
        w_out[mask] = np.interp(v_abs[mask], v_mono, w_mono)

    return signs * w_out
