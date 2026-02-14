"""
Solvers for Fractional-Order Blind Image Deconvolution
with Patch-wise Minimal Pixels (PMP) Prior.

This module contains the core numerical routines:

1. **Image sub-problem** – ADMM with fractional-order total-variation
   regularisation (Sec. 3.1 of [1]).
2. **Kernel sub-problem** – closed-form in the Fourier domain with
   simplex projection (Sec. 3.3 of [1]).
3. **Edge prediction** – gradient thresholding guided by the PMP map
   (Sec. 3.2 of [1]).
4. **Coarse-to-fine loop** – multi-scale blind estimation
   (Algorithm 2 in [1]; standard scheme of [3, 4]).

References
----------
[1] Wu, T., Wan, S., Feng, C., Zhang, H., Zeng, T.
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
    Meets Fractional-Order Method."
    J. Math. Imaging Vis., 2024.  DOI: 10.1007/s10851-024-01221-x

[2] Pan, X., Ye, Y., Wang, J., Gao, X., Zhou, X.
    "Noncausal fractional directional differentiator and blind
    deconvolution: motion blur estimation."
    Multimedia Tools Appl., 73(3), 1485–1506, 2014.

[3] Cho, S., Lee, S.
    "Fast motion deblurring."
    ACM Trans. Graphics (SIGGRAPH Asia), 28(5), 2009.

[4] Xu, L., Jia, J.
    "Two-phase kernel estimation for robust motion deblurring."
    ECCV 2010, pp. 157–170.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, Optional

from .utils import (
    EPSILON,
    FractionalOperators,
    IntegerGradientOperators,
    psf2otf,
    soft_threshold,
    precompute_fractional_operators,
    precompute_gradient_operators,
    predict_edges_with_pmp,
    build_scale_list,
    downscale_image,
    resize_kernel,
    threshold_kernel,
    center_kernel,
    make_initial_kernel,
)


# ===================================================================
# 1.  Image sub-problem  –  ADMM with fractional-order TV
# ===================================================================
def solve_image_fractional_tv(
    g: np.ndarray,
    k: np.ndarray,
    u_init: np.ndarray,
    frac_ops: FractionalOperators,
    lambda_tv: float,
    rho: float = 1.0,
    num_iter: int = 15,
    rho_scale: float = 1.05,
) -> np.ndarray:
    r"""
    Solve the image sub-problem via ADMM with fractional-order TV.

    The optimisation problem ([1], Sec. 3.1, Eq. 8) is

    .. math::
        \min_{u}\;
        \tfrac{1}{2}\|k \ast u - g\|_2^2
        + \lambda\bigl(\|v_x\|_1 + \|v_y\|_1\bigr)

    subject to  :math:`v_x = D_x^{\alpha} u`,
    :math:`v_y = D_y^{\alpha} u`.

    **ADMM splitting** introduces scaled dual variables
    :math:`b_x, b_y` and penalty :math:`\rho`:

    *u-update* (linear system in Fourier domain – [1], Eq. 9):

    .. math::
        \hat{u} = \frac{
            \overline{\hat{k}}\,\hat{g}
            + \rho\bigl(
                \overline{\hat{C}_x}(\hat{v}_x - \hat{b}_x)
              + \overline{\hat{C}_y}(\hat{v}_y - \hat{b}_y)
            \bigr)
        }{
            |\hat{k}|^2
            + \rho\bigl(|\hat{C}_x|^2 + |\hat{C}_y|^2\bigr)
        }

    *v-update* (soft thresholding – [1], Eq. 10):

    .. math::
        v_d = \mathrm{shrink}\!\bigl(D_d^{\alpha} u + b_d,\;
              \lambda / \rho\bigr), \quad d \in \{x, y\}

    *Dual update* :

    .. math::
        b_d \leftarrow b_d + D_d^{\alpha} u - v_d

    Parameters
    ----------
    g : ndarray (H, W)        – blurred observation.
    k : ndarray (kh, kw)      – current kernel estimate.
    u_init : ndarray (H, W)   – warm-start for the image.
    frac_ops : FractionalOperators
        Pre-computed fractional gradient FFTs.
    lambda_tv : float          – regularisation weight.
    rho : float                – initial ADMM penalty.
    num_iter : int             – number of ADMM iterations.
    rho_scale : float          – multiplicative increase of rho per iter.

    Returns
    -------
    u : ndarray (H, W)
        Estimated latent image (values clipped to [0, 1]).
    """
    H, W = g.shape
    F_Cx, F_Cy, F_frac_sq = frac_ops

    F_k = psf2otf(k, (H, W))
    F_k_conj = np.conj(F_k)
    F_k_sq = np.abs(F_k) ** 2
    F_g = fft2(g)

    # Initialise primal and dual variables
    u = u_init.copy()
    vx = np.zeros((H, W), dtype=np.float64)
    vy = np.zeros((H, W), dtype=np.float64)
    bx = np.zeros((H, W), dtype=np.float64)
    by = np.zeros((H, W), dtype=np.float64)

    for _ in range(num_iter):
        # -------- u-update (Fourier solve) --------
        rhs = F_k_conj * F_g + rho * (
            np.conj(F_Cx) * fft2(vx - bx)
            + np.conj(F_Cy) * fft2(vy - by)
        )
        denom = F_k_sq + rho * F_frac_sq + EPSILON
        u = np.real(ifft2(rhs / denom))

        # -------- v-update (shrinkage) --------
        Dx_u = np.real(ifft2(F_Cx * fft2(u)))
        Dy_u = np.real(ifft2(F_Cy * fft2(u)))

        vx = soft_threshold(Dx_u + bx, lambda_tv / rho)
        vy = soft_threshold(Dy_u + by, lambda_tv / rho)

        # -------- dual update --------
        bx += Dx_u - vx
        by += Dy_u - vy

        # Increase penalty (continuation)
        rho *= rho_scale

    u = np.clip(u, 0.0, 1.0)
    return u


# ===================================================================
# 2.  Kernel sub-problem  –  spectral solve + simplex projection
# ===================================================================
def estimate_kernel_from_edges(
    pred_dx: np.ndarray,
    pred_dy: np.ndarray,
    g: np.ndarray,
    kernel_shape: Tuple[int, int],
    mu: float = 0.01,
    int_ops: Optional[IntegerGradientOperators] = None,
) -> np.ndarray:
    r"""
    Estimate the blur kernel from predicted edge maps.

    Given predicted gradients :math:`\hat\partial_x u`,
    :math:`\hat\partial_y u` and the blurred image :math:`g`,
    the kernel sub-problem ([1], Sec. 3.3) reads

    .. math::
        \min_{k}\;
        \sum_{d \in \{x,y\}}
          \lVert \hat\partial_d u \ast k - \partial_d g \rVert_2^2
        + \mu\,\|k\|_2^2
        \quad \text{s.t.}\;\; k \ge 0,\;\sum k = 1.

    Closed-form solution in the frequency domain:

    .. math::
        \hat{k} = \frac{
            \sum_d \overline{\widehat{\hat\partial_d u}}\;
                   \widehat{\partial_d g}
        }{
            \sum_d \bigl|\widehat{\hat\partial_d u}\bigr|^2 + \mu
        }

    The spatial kernel is then projected onto the probability simplex
    (non-negativity + unit sum), thresholded, and centred.

    Parameters
    ----------
    pred_dx, pred_dy : ndarray (H, W)
        Predicted salient gradient maps.
    g : ndarray (H, W)
        Blurred observation.
    kernel_shape : (kh, kw)
    mu : float
        Tikhonov regularisation for kernel smoothness.
    int_ops : IntegerGradientOperators or None
        If *None*, operators are recomputed internally.

    Returns
    -------
    k : ndarray (kh, kw)
        Estimated kernel (non-negative, unit sum).
    """
    H, W = g.shape

    if int_ops is None:
        int_ops = precompute_gradient_operators((H, W))

    F_dx, F_dy, _ = int_ops

    # Gradients of blurred image g
    F_g = fft2(g)
    F_gx = F_dx * F_g      # ∂_x g  in Fourier
    F_gy = F_dy * F_g

    # Predicted edge maps in Fourier
    F_px = fft2(pred_dx)
    F_py = fft2(pred_dy)

    # Closed-form ([1] Eq. 12-like)
    numerator = np.conj(F_px) * F_gx + np.conj(F_py) * F_gy
    denominator = np.abs(F_px) ** 2 + np.abs(F_py) ** 2 + mu

    F_k = numerator / denominator
    k_full = np.real(ifft2(F_k))

    # Crop to kernel support (centred)
    kh, kw = kernel_shape
    k_full = np.roll(k_full, kh // 2, axis=0)
    k_full = np.roll(k_full, kw // 2, axis=1)
    k = k_full[:kh, :kw]

    # Simplex projection:  k ≥ 0, Σk = 1
    k = np.maximum(k, 0.0)
    k = threshold_kernel(k, rel_threshold=0.05)
    k = center_kernel(k)
    return k


# ===================================================================
# 3.  Single-scale blind estimation step
# ===================================================================
def single_scale_step(
    g: np.ndarray,
    k: np.ndarray,
    kernel_shape: Tuple[int, int],
    alpha: float,
    L: int,
    lambda_tv: float,
    mu_kernel: float,
    admm_iter: int,
    rho_init: float,
    grad_threshold_pct: float,
    pmp_patch_size: int,
    pmp_gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    One iteration of the alternating minimisation at a single
    pyramid level.

    1. **Image estimation** with fractional-order TV (ADMM).
    2. **Edge prediction** via PMP-guided gradient thresholding.
    3. **Kernel estimation** from predicted edges (Fourier solve).

    Parameters
    ----------
    g : ndarray (H, W) – blurred image at this scale.
    k : ndarray        – current kernel estimate.
    kernel_shape       – desired kernel size at this scale.
    alpha, L           – fractional derivative parameters.
    lambda_tv, mu_kernel – regularisation weights.
    admm_iter          – inner ADMM iterations for image step.
    rho_init           – initial ADMM penalty.
    grad_threshold_pct – gradient thresholding percentile.
    pmp_patch_size     – PMP patch size.
    pmp_gamma          – PMP decay parameter.

    Returns
    -------
    (k_new, u) : tuple
        Updated kernel and intermediate image.
    """
    H, W = g.shape

    # Pre-compute operators for this image size
    frac_ops = precompute_fractional_operators((H, W), alpha, L)
    int_ops = precompute_gradient_operators((H, W))

    # 1. Image estimation (ADMM with fractional TV)  –  [1] Sec. 3.1
    u = solve_image_fractional_tv(
        g, k, g.copy(), frac_ops,
        lambda_tv=lambda_tv,
        rho=rho_init,
        num_iter=admm_iter,
    )

    # 2. Edge prediction with PMP  –  [1] Sec. 3.2
    pred_dx, pred_dy = predict_edges_with_pmp(
        u,
        grad_threshold_percentile=grad_threshold_pct,
        patch_size=pmp_patch_size,
        pmp_gamma=pmp_gamma,
    )

    # 3. Kernel estimation  –  [1] Sec. 3.3
    k_new = estimate_kernel_from_edges(
        pred_dx, pred_dy, g, kernel_shape,
        mu=mu_kernel, int_ops=int_ops,
    )

    return k_new, u


# ===================================================================
# 4.  Coarse-to-fine multi-scale loop
# ===================================================================
def coarse_to_fine_estimation(
    g: np.ndarray,
    kernel_shape: Tuple[int, int],
    alpha: float = 1.4,
    L: int = 10,
    lambda_tv: float = 4e-3,
    mu_kernel: float = 0.01,
    admm_iter: int = 15,
    rho_init: float = 1.0,
    inner_iter: int = 3,
    grad_threshold_pct: float = 94.0,
    pmp_patch_size: int = 5,
    pmp_gamma: float = 2.0,
    scale_ratio: float = 1.5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Multi-scale coarse-to-fine blind deconvolution
    ([1], Algorithm 2; see also [3, 4] for the standard pyramid scheme).

    The image pyramid is built so that the effective kernel at the
    coarsest level is ≈ 3 pixels wide.  At each level the algorithm
    alternates between:

    * fractional-TV image estimation  (ADMM, Sec. 3.1),
    * PMP-guided edge prediction      (Sec. 3.2),
    * spectral kernel estimation      (Sec. 3.3).

    The kernel estimate is up-sampled between levels.

    Parameters
    ----------
    g : ndarray (H, W)
        Full-resolution blurred image, float64 in [0, 1].
    kernel_shape : (kh, kw)
        Maximum kernel support at the finest level.
    alpha : float
        Fractional derivative order (1 < α < 2).
    L : int
        GL truncation length.
    lambda_tv : float
        Weight of the fractional-TV regulariser.
    mu_kernel : float
        Tikhonov weight on the kernel.
    admm_iter : int
        ADMM iterations per image-estimation step.
    rho_init : float
        Initial ADMM penalty parameter.
    inner_iter : int
        Number of alternating-minimisation passes per scale.
    grad_threshold_pct : float
        Percentile threshold for edge prediction.
    pmp_patch_size : int
        PMP prior patch size.
    pmp_gamma : float
        PMP weight decay.
    scale_ratio : float
        Ratio between successive pyramid scales.
    verbose : bool

    Returns
    -------
    (k_est, u_est) : tuple
        Estimated kernel and latent image at the finest scale.
    """
    kh, kw = kernel_shape
    max_k = max(kh, kw)

    # Build scale list (coarse → fine)
    scales = build_scale_list(max_k, min_kernel_dim=3,
                              scale_ratio=scale_ratio)

    # Initialise a small Gaussian kernel at the coarsest scale
    init_k_size = max(3, int(np.round(max_k * scales[0])))
    init_k_size = init_k_size if init_k_size % 2 == 1 else init_k_size + 1
    k_est = make_initial_kernel((init_k_size, init_k_size))

    u_est = g.copy()

    for s_idx, scale in enumerate(scales):
        # ---- Down-scale image to current pyramid level ----
        g_s = downscale_image(g, scale)
        H_s, W_s = g_s.shape

        # Effective kernel size at this scale
        cur_kh = max(3, int(np.round(kh * scale)))
        cur_kw = max(3, int(np.round(kw * scale)))
        cur_kh = cur_kh if cur_kh % 2 == 1 else cur_kh + 1
        cur_kw = cur_kw if cur_kw % 2 == 1 else cur_kw + 1
        cur_kshape = (cur_kh, cur_kw)

        # Up-sample kernel from previous (coarser) level
        k_s = resize_kernel(k_est, cur_kshape)

        if verbose:
            print(
                f"  Scale {s_idx + 1}/{len(scales)}: "
                f"img {H_s}×{W_s}, kernel {cur_kh}×{cur_kw}, "
                f"factor {scale:.3f}"
            )

        # ---- Alternating minimisation at this scale ----
        for it in range(inner_iter):
            k_s, u_s = single_scale_step(
                g_s, k_s, cur_kshape,
                alpha=alpha, L=L,
                lambda_tv=lambda_tv,
                mu_kernel=mu_kernel,
                admm_iter=admm_iter,
                rho_init=rho_init,
                grad_threshold_pct=grad_threshold_pct,
                pmp_patch_size=pmp_patch_size,
                pmp_gamma=pmp_gamma,
            )

        k_est = k_s
        u_est = u_s

    # Resize final kernel to requested shape if not exact
    if k_est.shape != kernel_shape:
        k_est = resize_kernel(k_est, kernel_shape)

    return k_est, u_est


# ===================================================================
# 5.  Final non-blind deconvolution with fractional TV
# ===================================================================
def final_nonblind_deconvolution(
    g: np.ndarray,
    k: np.ndarray,
    alpha: float = 1.4,
    L: int = 10,
    lambda_tv: float = 2e-3,
    num_iter: int = 30,
    rho_init: float = 1.0,
) -> np.ndarray:
    r"""
    Final non-blind restoration using the estimated kernel and
    fractional-order TV regularisation (ADMM).

    This solves ([1], Eq. 8) one last time with more iterations and
    a smaller :math:`\lambda` for fine detail recovery.

    Parameters
    ----------
    g : ndarray (H, W)  – blurred observation.
    k : ndarray          – estimated kernel.
    alpha : float        – fractional order.
    L : int              – GL truncation length.
    lambda_tv : float    – regularisation weight (typically smaller
                           than in the blind phase).
    num_iter : int       – ADMM iterations.
    rho_init : float     – initial ADMM penalty.

    Returns
    -------
    u : ndarray (H, W)  – restored image, clipped to [0, 1].
    """
    H, W = g.shape
    frac_ops = precompute_fractional_operators((H, W), alpha, L)

    u = solve_image_fractional_tv(
        g, k, g.copy(), frac_ops,
        lambda_tv=lambda_tv,
        rho=rho_init,
        num_iter=num_iter,
        rho_scale=1.02,
    )
    return u
