"""
Core PRIDA solvers for blind image deconvolution.

Implements the alternating minimisation scheme for the energy

.. math::
    E(u, k) = \\tfrac{1}{2}\\|k * u - f\\|^2 + \\lambda\\,\\mathrm{TV}(u)
    \\quad\\text{s.t.}\\quad k \\ge 0,\\; \\sum_i k_i = 1

Two principle routines are exposed:

1. ``prida_single_scale`` — alternating gradient descent (image) and
   exponentiated / mirror-descent gradient descent (kernel) at a fixed
   resolution.
2. ``coarse_to_fine`` — multi-scale pyramid wrapper that invokes
   ``prida_single_scale`` from the coarsest to the finest level.

Reference
Ravi, S. N., Mehta, R., & Singh, V. (2018).
"Robust Blind Deconvolution via Mirror Descent."
arXiv:1803.08137 [cs.CV].

Original C++ implementation: main.cpp by Tianyi Shan (2018).
"""

import numpy as np
from typing import Tuple

from .utils import (
    conv2,
    rot180,
    grad_tv,
    resize_2d,
    build_pyramid,
)

#  Single-Scale Alternating Minimisation

def prida_single_scale(
    f: np.ndarray,
    u: np.ndarray,
    k: np.ndarray,
    lambda_val: float,
    n_iters: int,
    kernel_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    PRIDA alternating minimisation at a **single** scale.

    At every iteration two blocks are updated *simultaneously*
    (Jacobi-style — both gradients are evaluated at the *old* iterate):

    **(a) Image update** — gradient descent with adaptive step-size
    (Ravi et al. 2018, Sec. 3; C++ ``prida()``, lines 195–245):

    .. math::
        \nabla_u E
          = \underbrace{K^\top(Ku - f)}_{\text{data fidelity}}
            \;-\; \lambda\;\operatorname{div}\!\Bigl(\frac{\nabla u}{|\nabla u|}\Bigr)

    .. math::
        \eta_u = 10^{-3}\,\frac{\max|u|}{\max|\nabla_u E|}
        ,\qquad
        u^{+} = u - \eta_u\,\nabla_u E

    **(b) Kernel update** — exponentiated gradient / mirror descent
    (Ravi et al. 2018, Eq. 5–6; C++ ``prida()``, lines 250–290).
    The multiplicative form guarantees :math:`k \ge 0`:

    .. math::
        g = U^\top(Ku - f)

    .. math::
        \eta_k = \frac{10^{-3}\,\max|k|}{\max|g|}
        ,\quad
        \eta_i = \frac{\eta_k}{k_i + \varepsilon}

    .. math::
        k_i^{+} \propto k_i\,\min\!\bigl(\exp(-\eta_i\,g_i),\;10^3\bigr)
        ,\qquad
        k^{+} = k^{+} / \sum k^{+}

    Parameters
    f : np.ndarray, shape (M, N)
        Observed blurred image at the current scale.
    u : np.ndarray, shape (M + MK - 1, N + NK - 1)
        Current sharp-image estimate (padded domain so that
        ``conv2(u, k, 'valid')`` has the same shape as *f*).
    k : np.ndarray, shape (MK, NK)
        Current kernel estimate (non-negative, sums to 1).
    lambda_val : float
        TV regularisation weight at this scale.
    n_iters : int
        Number of alternating-minimisation iterations.
    kernel_shape : (int, int)
        ``(MK, NK)`` — kernel dimensions (for documentation only;
        already implied by ``k.shape``).

    Returns
    u : np.ndarray
        Updated image estimate (same shape as input *u*).
    k : np.ndarray
        Updated kernel estimate (same shape as input *k*).
    """
    _EPS_FLOAT = np.finfo(np.float64).eps    # ≈ 2.2e-16
    _CLAMP     = 1000.0                       # exp-term stability bound
    _STEP_INIT = 1e-3                         # base relative step-size

    for _ in range(n_iters):

        # (a) IMAGE UPDATE — TV-regularised gradient descent

        # Data-fidelity gradient:  K^T (K u - f)
        #   conv2(u, k, 'valid') has shape (M, N) — same as f.
        #   conv2(residual, K_rot, 'full') has shape (M+MK-1, N+NK-1) — same as u.
        residual = conv2(u, k, mode='valid') - f
        k_rot    = rot180(k)
        grad_data_u = conv2(residual, k_rot, mode='full')

        # TV term:  -∂TV/∂u  =  div(∇u / |∇u|)  (returned by grad_tv)
        tv_term = grad_tv(u)

        # Combined gradient:  ∂E/∂u  =  data_grad - λ·(−∂TV/∂u)
        grad_u = grad_data_u - lambda_val * tv_term

        # Adaptive step-size  (C++ lines 235–241)
        max_u  = np.max(np.abs(u))
        max_gu = np.max(np.abs(grad_u))
        step_u = _STEP_INIT * max_u / max(1e-31, max_gu)

        u_new = u - step_u * grad_u

        # (b) KERNEL UPDATE — Mirror / Exponentiated Gradient Descent

        # Kernel gradient:  U^T (K u - f)
        #   conv2(u_rot, residual, 'valid') maps to kernel-sized output.
        residual_k = conv2(u, k, mode='valid') - f
        u_rot      = rot180(u)
        grad_k     = conv2(u_rot, residual_k, mode='valid')

        # Adaptive base step
        max_k  = np.max(np.abs(k))
        max_gk = np.max(np.abs(grad_k))
        step_k = _STEP_INIT * max_k / max(1e-31, max_gk)

        # Element-wise adaptive rate 
        eta = step_k / (k + _EPS_FLOAT)

        # Multiplicative update
        exp_term = np.exp(-eta * grad_k)
        exp_term = np.minimum(exp_term, _CLAMP)

        k_new = k * exp_term
        k_sum = np.sum(k_new)
        if k_sum > 0:
            k_new /= k_sum              # project onto probability simplex
        else:
            k_new = np.ones_like(k) / k.size

        # Simultaneous (Jacobi) assignment
        u = u_new
        k = k_new

    return u, k

#  Coarse-to-Fine Multi-Scale Wrapper

def coarse_to_fine(
    image: np.ndarray,
    kernel_shape: Tuple[int, int],
    lambda_val: float,
    n_iters: int = 1000,
    lambda_multiplier: float = 1.9,
    max_lambda: float = 0.11,
    scale_multiplier: float = 1.1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Multi-scale coarse-to-fine blind deconvolution.

    Constructs a Gaussian scale-space pyramid and runs
    ``prida_single_scale`` from the **coarsest** to the **finest** level,
    progressively refining both the image and the kernel estimates.

    Pyramid strategy (Ravi et al. 2018, Sec. 4):
        * At coarser levels, :math:`\lambda` is *larger* → stronger TV
          regularisation suppresses noise and prevents trivial solutions.
        * Kernel size shrinks by ``scale_multiplier`` per level.
        * Image is down-sampled proportionally.

    Initialisation:
        * :math:`u^{(0)}`:  input image padded with replicate borders by
          ``⌊MK/2⌋`` rows and ``⌊NK/2⌋`` columns on each side, so that
          ``conv2(u, k, 'valid')`` yields an output matching the
          observation size.
        * :math:`k^{(0)}`:  uniform kernel (``1 / (MK·NK)``).

    Parameters
    image : np.ndarray, shape (H, W)
        Blurred input image (float64, range [0, 1]).
    kernel_shape : (int, int)
        ``(MK, NK)`` — blur kernel dimensions at the finest scale.
    lambda_val : float
        TV regularisation weight at the finest scale.
    n_iters : int
        Alternating-minimisation iterations **per pyramid level**.
    lambda_multiplier : float
        Factor by which λ grows per coarser level (default 1.9).
    max_lambda : float
        Upper bound for λ in pyramid (default 0.11).
    scale_multiplier : float
        Kernel-size reduction factor per level (default 1.1).
    verbose : bool
        Print per-level progress.

    Returns
    u : np.ndarray, shape (H + MK - 1, W + NK - 1)
        Estimated sharp image in the padded domain.
    k : np.ndarray, shape (MK, NK)
        Estimated blur kernel (non-negative, sums to 1).
    """
    MK, NK = kernel_shape

    # Initialise image estimate (replicate-border padding)
    # Matches C++ blind_deconv / coarseToFine initialisation.
    pad_top  = MK // 2
    pad_left = NK // 2
    u = np.pad(image, ((pad_top, pad_top), (pad_left, pad_left)), mode='edge')

    # Initialise kernel as uniform distribution 
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    # Build multi-scale pyramid (finest first)
    pyramid = build_pyramid(
        image, MK, NK,
        lambda_val, lambda_multiplier, max_lambda, scale_multiplier,
    )
    n_scales = len(pyramid)

    if verbose:
        print(f"[PRIDA] Pyramid: {n_scales} scale(s), "
              f"Image: {image.shape[0]}×{image.shape[1]}, "
              f"Kernel: {MK}×{NK}")

    # Process from coarsest (last index) to finest (index 0)
    for idx in range(n_scales - 1, -1, -1):
        level = pyramid[idx]
        f_s  = level['image']
        M_s  = level['M']
        N_s  = level['N']
        MK_s = level['MK']
        NK_s = level['NK']
        lam_s = level['lambda']

        # Resize u and k to current pyramid level
        u_target_h = M_s + MK_s - 1
        u_target_w = N_s + NK_s - 1
        u = resize_2d(u, (u_target_h, u_target_w), order=1)
        k = resize_2d(k, (MK_s, NK_s), order=1)

        # Re-normalise kernel after interpolation
        k_sum = np.sum(k)
        if k_sum > 0:
            k /= k_sum
        else:
            k = np.ones((MK_s, NK_s), dtype=np.float64) / (MK_s * NK_s)

        if verbose:
            print(f"  Scale {n_scales - idx}/{n_scales}: "
                  f"img {M_s}×{N_s}, ker {MK_s}×{NK_s}, λ={lam_s:.5f}")

        # Run alternating minimisation at this scale
        u, k = prida_single_scale(
            f_s, u, k, lam_s, n_iters, (MK_s, NK_s),
        )

    return u, k
