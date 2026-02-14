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

from .utils import psf2otf, block_matching


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
    z_nlr: np.ndarray = None,
    mu_nlr: float = 0.0,
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
    z_nlr : np.ndarray or None, optional
        Non-local low-rank prior image (WNNM-denoised).
        When provided together with ``mu_nlr > 0``, adds the
        coupling term μ‖x − z‖² to the objective.
    mu_nlr : float
        Weight for the non-local low-rank coupling term.

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

    use_nlr = z_nlr is not None and mu_nlr > 0.0

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
        if use_nlr:
            b = b + mu_nlr * z_nlr

        # Linear operator  A·v = K^T K v + α D^T W D v  [+ μ v]
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

            if use_nlr:
                result = result + mu_nlr * v

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
#  2b.  KERNEL ESTIMATION  (FFT Closed-Form, Gradient Domain)
# ======================================================================

def optimize_kernel_fft(
    x: np.ndarray,
    blurred: np.ndarray,
    kernel_size: int,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    FFT-based kernel estimation in the gradient domain.

    Solves the least-squares kernel problem using horizontal and
    vertical image gradients (Ren et al. 2016, Eq. 23):

    .. math::

        k = \\mathcal{F}^{-1}\\!\\left(
            \\frac{\\sum_{i \\in \\{h,v\\}}
                   \\overline{\\mathcal{F}(\\partial_i x)}
                   \\,\\cdot\\,
                   \\mathcal{F}(\\partial_i y)}
                 {\\sum_{i \\in \\{h,v\\}}
                   |\\mathcal{F}(\\partial_i x)|^2 + \\gamma}
        \\right)

    Then projects onto the PSF feasible set {k ≥ 0, Σk = 1}.

    Parameters
    ----------
    x : np.ndarray, shape (H, W)
        Current sharp-image estimate.
    blurred : np.ndarray, shape (H, W)
        Observed blurred image.
    kernel_size : int
        Target kernel size (must be odd).
    gamma : float
        Tikhonov regularisation weight.

    Returns
    -------
    kernel : np.ndarray, shape (kernel_size, kernel_size)

    References
    ----------
    [2] Ren et al. (IEEE TIP 2016), Eq. (23)–(24).
    """
    H, W = blurred.shape
    ks = kernel_size
    half = ks // 2

    # Gradient operators
    dx = np.array([[1.0, -1.0]])
    dy = np.array([[1.0], [-1.0]])

    # Image & observation gradients
    gx_x = fftconvolve(x, dx, mode='same')
    gy_x = fftconvolve(x, dy, mode='same')
    gx_y = fftconvolve(blurred, dx, mode='same')
    gy_y = fftconvolve(blurred, dy, mode='same')

    # 2-D FFTs
    Fgx_x = np.fft.fft2(gx_x)
    Fgy_x = np.fft.fft2(gy_x)
    Fgx_y = np.fft.fft2(gx_y)
    Fgy_y = np.fft.fft2(gy_y)

    # Closed-form solution  (Ren 2016, Eq. 23)
    numer = np.conj(Fgx_x) * Fgx_y + np.conj(Fgy_x) * Fgy_y
    denom = np.abs(Fgx_x) ** 2 + np.abs(Fgy_x) ** 2 + gamma

    k_full = np.real(np.fft.ifft2(numer / denom))

    # The PSF center sits at (0, 0) in FFT convention;
    # shift to array center, then crop to target size.
    k_full = np.fft.fftshift(k_full)
    cy, cx = H // 2, W // 2
    kernel = k_full[cy - half: cy + half + 1,
                    cx - half: cx + half + 1].copy()

    # Project onto feasible set:  k ≥ 0,  Σk = 1
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

    for _ in range(max_iter):
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # MM weights from CURRENT singular values:  w_i = 1/(σ_i + δ)
        # ([1], optimizerank_new.m;  derivative of  log(σ + δ))
        w = 1.0 / (S + delta)

        # Weighted soft-thresholding of singular values
        # Eq. (8) in [2]: S_thresh = max(Σ − τ·diag(w), 0)
        S_thresh = np.maximum(S - tau * w, 0.0)
        X = (U * S_thresh) @ Vt

    return X


# ======================================================================
#  3b.  NON-LOCAL LOW-RANK IMAGE REGULARISATION  (WNNM)
# ======================================================================

def wnnm_regularization(
    image: np.ndarray,
    patch_size: int = 6,
    search_window: int = 20,
    num_similar: int = 60,
    stride: int = 3,
    wnnm_C: float = 0.05,
    delta: float = 1e-6,
) -> np.ndarray:
    """
    Non-local low-rank image regularisation via Weighted Nuclear Norm
    Minimisation (WNNM).

    Adapts the multi-image low-rank idea to a *single* image by
    exploiting **Non-Local Self-Similarity** (NLSS): similar patches
    within the image are stacked into a matrix that is approximately
    low-rank, then denoised via WNNM.

    Pipeline
    --------
    1. **Block matching** (BM3D-style): for each reference patch,
       find the ``num_similar`` most similar patches in a local window.
    2. **WNNM**: for each patch group, form a data matrix
       X ∈ R^{d × n}, compute SVD, and apply weighted
       soft-thresholding of singular values.
    3. **Aggregation**: average overlapping denoised patches.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Current image estimate.
    patch_size : int
        Side length of square patches.
    search_window : int
        Half-side of the search area around each reference patch.
    num_similar : int
        Maximum similar patches per group (including reference).
    stride : int
        Spacing between reference patch positions.
    wnnm_C : float
        Weight scaling constant for WNNM:
        w_i = C · √n / (σ_i + δ).
    delta : float
        Smoothing constant to avoid division by zero.

    Returns
    -------
    denoised : np.ndarray, shape (H, W)

    References
    ----------
    [2] Ren et al. (IEEE TIP 2016), Sec. III — Enhanced Low-Rank Prior.
    [5] Yang et al. (IEEE Access 2020), Sec. III-A — Non-local
        low-rank prior.
    Gu, S., et al. "Weighted nuclear norm minimization with
    application to image denoising." CVPR, 2014.
    Dabov, K., et al. "Image denoising by sparse 3-D transform-domain
    collaborative filtering." IEEE TIP, 2007 (BM3D grouping).
    """
    H, W = image.shape
    ps = patch_size

    # Guard: if the image is too small for meaningful block matching,
    # return it unchanged.
    if H < 3 * ps or W < 3 * ps:
        return image.copy()

    # ---- Step 1: Block Matching  (Dabov et al., BM3D) ----------------
    groups = block_matching(image, ps, search_window, num_similar, stride)

    # ---- Step 2: WNNM on each group  (Gu et al. 2014) ---------------
    result = np.zeros_like(image)
    count = np.zeros_like(image)
    d = ps * ps

    for positions in groups:
        n = len(positions)
        if n < 2:
            continue

        # Form patch matrix  X ∈ R^{d × n}
        X = np.empty((d, n))
        for j in range(n):
            r, c = positions[j]
            X[:, j] = image[r:r + ps, c:c + ps].ravel()

        # Subtract column mean (improves low-rank approximation)
        mean_col = X.mean(axis=1, keepdims=True)
        X_c = X - mean_col

        # SVD
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)

        # WNNM weights:  w_i = C √n / (σ_i + δ)
        w = wnnm_C * np.sqrt(n) / (S + delta)

        # Weighted soft-thresholding of singular values
        S_new = np.maximum(S - w, 0.0)

        # Reconstruct denoised patches
        X_den = (U * S_new) @ Vt + mean_col

        # Accumulate into output
        for j in range(n):
            r, c = positions[j]
            result[r:r + ps, c:c + ps] += X_den[:, j].reshape(ps, ps)
            count[r:r + ps, c:c + ps] += 1.0

    # ---- Step 3: Normalise by overlap count -------------------------
    mask = count > 0
    result[mask] /= count[mask]
    result[~mask] = image[~mask]

    return result


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

    # ------------------------------------------------------------------
    # β-continuation: start with a smaller penalty and ramp up to the
    # target β geometrically.  Early iterations act as regularised
    # pre-conditioning (strong prior → smooth, no ringing); later
    # iterations refine detail.
    # Ref: Boyd et al. (2011), Sec. 3.4.1; Krishnan & Fergus (2009).
    # ------------------------------------------------------------------
    beta_min = max(1.0, beta / 64.0)
    beta_rate = (beta / beta_min) ** (1.0 / max(max_outer - 1, 1))

    for outer_it in range(max_outer):
        cur_beta = min(beta_min * beta_rate ** outer_it, beta)

        for _ in range(max_inner):

            # === w-sub-problem ========================================
            # prox_{|·|^α / β}(∇g + b)
            if alpha == 1.0:
                # Soft-thresholding  ([4], Eq. (6))
                wx = _soft_threshold(gx + bx, 1.0 / cur_beta)
                wy = _soft_threshold(gy + by, 1.0 / cur_beta)
            else:
                # Hyper-Laplacian proximal  ([4], Sec. 3.1)
                wx = _hyper_laplacian_proximal(gx + bx, cur_beta, alpha)
                wy = _hyper_laplacian_proximal(gy + by, cur_beta, alpha)

            # === b-update (Bregman iteration) =========================
            bx = bx + gx - wx
            by = by + gy - wy

            # === g-sub-problem (Fourier closed-form) ==================
            # [4], Eq. (8):
            #   G = (λ K^H f  + β D^H(w − b)) / (λ|K|² + β|D|²)
            wx_full = fftconvolve(wx - bx, dxt, mode='full')
            wy_full = fftconvolve(wy - by, dyt, mode='full')

            numer = lambda_ * Ktf + cur_beta * np.fft.fft2(
                wx_full + wy_full)
            denom = lambda_ * KtK + cur_beta * DtD

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
