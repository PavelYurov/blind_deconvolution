"""
solvers.py

Core solver functions for the LIP (Logarithmic Image Prior) blind deconvolution.

Ported from MATLAB code by D. Perrone, P. Favaro (2014).
Reference:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", EMMCVPR 2015.

Contains:
    grad_tv_em   — gradient of the EM-majorised log-TV prior (gradTVEM.m)
    blind        — core MM gradient-descent loop (blind.m, Table 2)
    blind_pd     — core PD (Chambolle-Pock) loop (Table 1)
    build_pyramid — multi-scale pyramid construction (buildPyramid.m)
    coarse_to_fine — multi-scale coarse-to-fine wrapper (coarseToFine.m)

MATLAB → Python notes checked in each function below.
"""

import numpy as np

from .utils import (
    shft,
    convn_valid,
    convn_full,
    pad_replicate,
    imresize_matlab,
    psf2otf,
)


# ─────────────────────────────────────────────────────────────────────────────
# gradTVEM.m  →  grad_tv_em
# ─────────────────────────────────────────────────────────────────────────────

def grad_tv_em(u: np.ndarray, ut: np.ndarray,
               epsilon: float = 1e-3, tau: float = 1e-1) -> np.ndarray:
    """
    Gradient of the EM-majorised Total Variation regulariser.

    Equivalent to MATLAB ``gradTVEM.m`` by P. Favaro (2014).

    Uses a symmetric formulation with 4 neighbourhood structures
    (WS, ES, WN, EN) × 3 shifted cases = 12 terms, averaged by 1/4.

    The regulariser being majorised is  log(|∇u| + τ), and the
    EM majorant at the previous iterate uᵗ is:

        |∇u| / (|∇uᵗ| + τ)   (+ const)

    Parameters
    ----------
    u       : (M, N) current sharp-image estimate
    ut      : (M, N) previous iterate (for the majorant denominator)
    epsilon : smoothing constant to avoid division by zero in √(…)
    tau     : lower-bound parameter of the log prior

    Returns
    -------
    grad : (M, N) gradient of the majorised TV w.r.t. u

    MATLAB correspondence
    ---------------------
    deltas{1} = [ 1  1];   % WS        →  (dx= 1, dy= 1)
    deltas{2} = [-1  1];   % ES        →  (dx=-1, dy= 1)
    deltas{3} = [ 1 -1];   % WN        →  (dx= 1, dy=-1)
    deltas{4} = [-1 -1];   % EN        →  (dx=-1, dy=-1)

    For each (dx, dy) three cases are computed:
        case 1 (no shift)  : du = -(ux + uy)
        case 2 (x shift)   : du =  ux
        case 3 (y shift)   : du =  uy
    where ux, uy are appropriate finite-difference terms via ``shft``.

    Each contribution:  du / TV / (τ + TVt)
    where TV  = √(ε + ux² + uy²)   from u
          TVt = √(ε + utx² + uty²) from uᵗ

    Final result is averaged over the 4 neighbourhoods (/ 4).
    """
    # MATLAB: deltas{1}=[1 1]; deltas{2}=[-1 1]; ...
    deltas = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

    grad = np.zeros_like(u)

    for dx, dy in deltas:
        # ── case 1: no shift ──
        # MATLAB: ux{1,ns} = shft(u,dx,0) - shft(u,0,0);
        ux1 = shft(u, dx, 0) - shft(u, 0, 0)
        uy1 = shft(u, 0, dy) - shft(u, 0, 0)
        TV1 = np.sqrt(epsilon + ux1 ** 2 + uy1 ** 2)
        du1 = -ux1 - uy1

        utx1 = shft(ut, dx, 0) - shft(ut, 0, 0)
        uty1 = shft(ut, 0, dy) - shft(ut, 0, 0)
        TVt1 = np.sqrt(epsilon + utx1 ** 2 + uty1 ** 2)

        # ── case 2: x shift ──
        # MATLAB: ux{2,ns} = shft(u,0,0) - shft(u,-dx,0);
        ux2 = shft(u, 0, 0) - shft(u, -dx, 0)
        uy2 = shft(u, -dx, dy) - shft(u, -dx, 0)
        TV2 = np.sqrt(epsilon + ux2 ** 2 + uy2 ** 2)
        du2 = ux2

        utx2 = shft(ut, 0, 0) - shft(ut, -dx, 0)
        uty2 = shft(ut, -dx, dy) - shft(ut, -dx, 0)
        TVt2 = np.sqrt(epsilon + utx2 ** 2 + uty2 ** 2)

        # ── case 3: y shift ──
        # MATLAB: ux{3,ns} = shft(u,dx,-dy) - shft(u,0,-dy);
        ux3 = shft(u, dx, -dy) - shft(u, 0, -dy)
        uy3 = shft(u, 0, 0) - shft(u, 0, -dy)
        TV3 = np.sqrt(epsilon + ux3 ** 2 + uy3 ** 2)
        du3 = uy3

        utx3 = shft(ut, dx, -dy) - shft(ut, 0, -dy)
        uty3 = shft(ut, 0, 0) - shft(ut, 0, -dy)
        TVt3 = np.sqrt(epsilon + utx3 ** 2 + uty3 ** 2)

        # ── accumulate:  du / TV / (tau + TVt) ──
        grad += du1 / TV1 / (tau + TVt1)
        grad += du2 / TV2 / (tau + TVt2)
        grad += du3 / TV3 / (tau + TVt3)

    # Average over 4 neighbourhood structures
    grad /= 4.0
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# blind.m  →  blind
# ─────────────────────────────────────────────────────────────────────────────

def blind(f: np.ndarray, MK: int, NK: int, beta: float,
          u: np.ndarray, k: np.ndarray,
          outer_iters: int = 140,
          inner_iters: int = 5,
          tau: float = 1e-3,
          k_step: float = 1e-3,
          u_step: float = 1e-3,
          blind_denoise_fn=None,
          progress_callback=None) -> tuple:
    """
    Core MM blind-deconvolution loop (Table 2 of the paper).

    Equivalent to MATLAB ``blind.m``.

    Minimises:
        min_{u,k}  (1 / 2β) · ||k ⊛ u − f||²  +  Σ_{i,j} |∇u_{i,j}| / (|∇uᵗ_{i,j}| + τ)

    The algorithm alternates:
        1. Gradient descent on u (sharp image)
        2. Gradient descent on k (kernel) with projection onto
           the set { k ≥ 0, Σk = 1 }

    Parameters
    ----------
    f           : (M, N) blurry image
    MK, NK      : kernel spatial support (rows, cols)
    beta        : data-fidelity weight  (called ``lambda`` at higher level)
    u           : (M+MK-1, N+NK-1) initial sharp-image estimate
    k           : (MK, NK) initial kernel estimate
    outer_iters : number of outer (majorisation-update) iterations
    inner_iters : number of inner gradient-descent iterations per outer iter
    tau         : lower-bound parameter τ of the log prior
    k_step      : kernel step-size scaling factor
    u_step      : image step-size scaling factor
    blind_denoise_fn : callable or None
        If not None, called as ``blind_denoise_fn(u)`` before each kernel
        update step; returns denoised u used for kernel gradient only.

    Returns
    -------
    u : (M+MK-1, N+NK-1) estimated sharp image  (padded)
    k : (MK, NK) estimated kernel

    MATLAB → Python notes
    ---------------------
    * ``convn(u,k,'valid')`` →  ``convn_valid(u, k)``   — both true convolutions.
    * ``convn(err, rot90(k,2), 'full')`` →  ``convn_full(err, np.rot90(k,2))``
      MATLAB rot90(k,2) rotates by 180° → same as np.rot90(k,2).
      convn(err, rot90(k,2), 'full') computes the "transpose-convolution"
      needed for the gradient w.r.t. u.
    * ``convn(rot90(u,2), err, 'valid')`` → ``convn_valid(np.rot90(u,2), err)``
      This yields the gradient w.r.t. k.
    * Adaptive step sizes:
        MATLAB:  dt = u_step * (max(u(:)) + 1/numel(u)) / max(abs(gradu(:)) + 1e-30)
        Python:  dt = u_step * (u.max()   + 1/u.size)   / (np.abs(gradu).max() + 1e-30)
      ``max(u(:))`` in MATLAB is global max → ``u.max()`` in NumPy (scalar).
      ``numel(u)`` → ``u.size``.
    * Kernel projection: max(k,0) → np.maximum(k, 0); k/sum(k(:)) → k/k.sum()
    """
    epsilon = 1e-3  # TV smoothing constant (hardcoded in MATLAB)

    for it in range(outer_iters):
        ut = u.copy()  # freeze majorant reference
        for itt in range(inner_iters):
            # ── sharp-image step ──
            synth = convn_valid(u, k)          # k ⊛ u  (valid)
            err = synth - f                     # residual
            # gradient w.r.t. u (data + TV prior)
            #   MATLAB: beta*convn(err, rot90(k,2), 'full') + gradTVEM(…)
            gradu = (beta * convn_full(err, np.rot90(k, 2))
                     + grad_tv_em(u, ut, epsilon, tau))
            # adaptive step
            dt = u_step * (u.max() + 1.0 / u.size) / (np.abs(gradu).max() + 1e-30)
            u = u - dt * gradu

            # ── kernel step ──
            synth = convn_valid(u, k)
            err = synth - f
            #   MATLAB: gradk = convn(rot90(u,2), err, 'valid')
            u_dk = blind_denoise_fn(u) if blind_denoise_fn is not None else u
            gradk = convn_valid(np.rot90(u_dk, 2), err)
            alpha = k_step * (k.max() + 1.0 / k.size) / (np.abs(gradk).max() + 1e-30)
            k = k - alpha * gradk
            # projection: non-negative, sum-to-one
            k = np.maximum(k, 0.0)
            k_sum = k.sum()
            if k_sum > 0:
                k /= k_sum

        if progress_callback is not None:
            try:
                progress_callback({
                    'event': 'iter',
                    'iter': it,
                    'kernel': k.copy(),
                    'beta': beta,
                })
            except Exception:
                pass

    return u, k


# ─────────────────────────────────────────────────────────────────────────────
# Condat-Vũ variant on MM-majorant  →  blind_cv
# (own construction — NOT the PD algorithm from the paper)
# ─────────────────────────────────────────────────────────────────────────────

def blind_cv(f: np.ndarray, MK: int, NK: int, beta: float,
             u: np.ndarray, k: np.ndarray,
             outer_iters: int = 140,
             inner_iters: int = 5,
             tau_param: float = 1e-3,
             k_step: float = 1e-3,
             blind_denoise_fn=None,
             progress_callback=None) -> tuple:
    """
    Condat-Vũ primal-dual splitting on the MM-majorised weighted-TV subproblem.

    This is **not** the PD algorithm from Perrone & Favaro (Table 1) — see
    ``blind_pd`` for the paper-faithful variant.  Here the log prior is first
    majorised (as in ``blind``) and the resulting weighted-TV subproblem is
    solved with Condat-Vũ splitting (data fidelity handled via its gradient,
    TV handled via a single dual variable).

    Key improvements over a pure Chambolle-Pock implementation:
        * Data-fidelity gradient uses **spatial convolutions** (no FFT),
          avoiding circular-boundary artefacts.
          [Chen & Huang, Inverse Problems 2013; Chen, SIOPT 2014]
        * Kernel update is performed **outside** the inner PD loop,
          preserving operator-splitting convergence guarantees.
        * Positivity constraint u ≥ 0 is enforced at each primal step.
        * Step sizes satisfy the **strict** Condat convergence condition:
          1/τ − σ·‖∇‖² > L_f / 2.

    Weighted-TV subproblem at outer iteration t:

        min_u  (β/2)·‖k*u − f‖²  +  Σ_{i,j}  |∇u_{i,j}| / (|∇uᵗ_{i,j}| + τ)

    The primal step uses ∇f(u) = β·kᵀ*(k*u − f) computed via spatial
    (valid/full) convolutions, so boundary conditions are linear (not
    circular).  The dual constraint is a per-pixel ball projection with
    radius 1 / (|∇uᵗ| + τ).

    Parameters
    ----------
    f           : (M, N)  blurry image
    MK, NK      : kernel spatial support
    beta        : data-fidelity weight
    u           : (M+MK-1, N+NK-1)  initial sharp-image estimate
    k           : (MK, NK)  initial kernel estimate
    outer_iters : number of MM (majorant-update) iterations
    inner_iters : number of Condat-Vũ iterations per outer step
    tau_param   : lower-bound τ of the log prior
    k_step      : kernel gradient-descent step-size scaling
    blind_denoise_fn : callable or None
        If not None, called as ``blind_denoise_fn(u)`` before the kernel
        update step; returns denoised u used for kernel gradient only.

    Returns
    -------
    u : (M+MK-1, N+NK-1)  estimated sharp image (padded)
    k : (MK, NK)  estimated kernel
    """
    epsilon = 1e-3
    Mu, Nu = u.shape

    # Dual variable: (p_x, p_y) per pixel
    p = np.zeros((2, Mu, Nu))

    # Grid for center-of-mass computation (allocated once)
    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target = (MK - 1) / 2.0
    cx_target = (NK - 1) / 2.0

    for it in range(outer_iters):
        # ── MM step: compute weights from current u ──
        grad_x_ut = np.roll(u, -1, axis=1) - u
        grad_y_ut = np.roll(u, -1, axis=0) - u
        w = np.sqrt(epsilon + grad_x_ut ** 2 + grad_y_ut ** 2)
        # Dual constraint radius: 1 / (w + τ)
        radius = 1.0 / (w + tau_param)

        u_bar = u.copy()

        # ── Step sizes (Condat-Vũ convergence condition) ──────────────
        # L_f = β · ‖k‖²_op.  For normalised non-negative k:
        # ‖k‖_op = max|FFT(k)| = sum(k) = 1,  so  L_f = β.
        L_f = beta
        # ‖∇‖² = 8.  Condition: 1/τ − 8σ > L_f / 2  (strict).
        # Balanced: 8σ = L_f/2  →  σ = L_f/16,  τ = 1/L_f.
        # 0.99 safety factor for strict inequality.
        sigma_pd = 0.99 * L_f / 16.0
        tau_pd = 0.99 / L_f
        theta = 1.0

        for itt in range(inner_iters):
            # ── Dual step: p ← proj( p + σ·∇ū ) ──
            grad_x = np.roll(u_bar, -1, axis=1) - u_bar
            grad_y = np.roll(u_bar, -1, axis=0) - u_bar

            ptx = p[0] + sigma_pd * grad_x
            pty = p[1] + sigma_pd * grad_y

            # Project onto ball of radius 1/(w + τ) at each pixel
            norm_pt = np.sqrt(ptx ** 2 + pty ** 2 + 1e-30)
            proj_scale = np.minimum(1.0, radius / norm_pt)
            p[0] = ptx * proj_scale
            p[1] = pty * proj_scale

            # ── Primal step (Condat-Vũ: gradient + divergence) ──
            u_old = u.copy()
            # Gradient of data fidelity via spatial convolutions
            # ∇f(u) = β · kᵀ * (k * u − f)
            synth = convn_valid(u, k)
            err = synth - f
            grad_data = beta * convn_full(err, np.rot90(k, 2))
            # Divergence of dual variable
            div_p = (p[0] - np.roll(p[0], 1, axis=1)) + \
                    (p[1] - np.roll(p[1], 1, axis=0))
            # Condat-Vũ update: u ← u − τ·∇f(u) + τ·div(p)
            u = u - tau_pd * grad_data + tau_pd * div_p
            # Positivity constraint (proximal of ι_{u≥0})
            u = np.maximum(u, 0.0)

            # Overrelaxation
            u_bar = u + theta * (u - u_old)

        # ── Kernel step (OUTSIDE inner loop for convergence) ──
        # [Chen, SIOPT 2014]: operator splitting requires fixed operators
        # within each block; updating k inside violates this.
        u_dk = blind_denoise_fn(u) if blind_denoise_fn is not None else u
        synth = convn_valid(u_dk, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u_dk, 2), err)
        alpha_k = k_step * (k.max() + 1.0 / k.size) / \
                  (np.abs(gradk).max() + 1e-30)
        k = k - alpha_k * gradk
        k = np.maximum(k, 0.0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        # ── Center-of-mass kernel centering ──
        # Keeps kernel centred for proper coarse-to-fine resizing.
        k_sum_c = k.sum()
        if k_sum_c > 0:
            cy = (ys * k).sum() / k_sum_c
            cx = (xs * k).sum() / k_sum_c
            dy = int(round(cy_target - cy))
            dx = int(round(cx_target - cx))
            if dy != 0 or dx != 0:
                k = np.roll(k, dy, axis=0)
                k = np.roll(k, dx, axis=1)
                # Compensate u/p to keep spatial alignment.
                u = np.roll(u, -dy, axis=0)
                u = np.roll(u, -dx, axis=1)
                p[0] = np.roll(p[0], -dy, axis=0)
                p[0] = np.roll(p[0], -dx, axis=1)
                p[1] = np.roll(p[1], -dy, axis=0)
                p[1] = np.roll(p[1], -dx, axis=1)

        if progress_callback is not None:
            try:
                progress_callback({
                    'event': 'iter',
                    'iter': it,
                    'kernel': k.copy(),
                    'beta': beta,
                })
            except Exception:
                pass

    return u, k


# ─────────────────────────────────────────────────────────────────────────────
# Table 1 of Perrone & Favaro (2016)  →  blind_pd
# Paper-faithful primal-dual solver for the non-convex log-TV prior.
# ─────────────────────────────────────────────────────────────────────────────

def _grad_neumann(u: np.ndarray):
    """Forward differences with Neumann (zero at last row/col) boundary."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div_neumann(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    Divergence operator — adjoint of ``_grad_neumann`` under ``<∇u, p> = -<u, div p>``.

    Standard Chambolle-2004 construction:
        div p[i, j] = (p_x[i, j]   − p_x[i, j-1])    (boundary-corrected)
                    + (p_y[i, j]   − p_y[i-1, j])
    """
    dx = np.zeros_like(px)
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, 0]    = px[:, 0]
    dx[:, -1]   = -px[:, -2]

    dy = np.zeros_like(py)
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[0, :]    = py[0, :]
    dy[-1, :]   = -py[-2, :]
    return dx + dy


def _h_function(xi: np.ndarray, mu: float, eps: float,
                sigma: float) -> np.ndarray:
    """
    Solve the 1-D sub-problem of eq. (25) in Perrone & Favaro (2016):

        H(ξ, μ, ε, σ) = argmin_ρ  (μ / 2σ)·(ρ − 1)²·ξ²  +  log(ρ²·ξ² + ε²)

    The first-order optimality condition is a depressed cubic in ρ:

        ρ³ − ρ² + c·ρ + d = 0,
        where  c = ε²/ξ² + 2σ/(μ·ξ²),   d = −ε²/ξ².

    Solved per-pixel (vectorised) via Cardano's formula; when the
    discriminant is positive we have three real roots and pick the one
    that minimises the objective (the paper suggests a LUT; the closed
    form here is fast enough for our problem sizes).

    Parameters
    ----------
    xi    : (…) array — non-negative magnitudes  ‖z₂ + σ·∇ū‖ / σ
    mu    : scalar μ  (= 2·λ/p  in the paper; we use μ = 2·β below)
    eps   : scalar ε of the log prior
    sigma : scalar PD dual step-size

    Returns
    -------
    rho : (…) array with values in [0, 1].
    """
    xi = np.asarray(xi, dtype=np.float64)
    eps2 = eps * eps

    rho = np.zeros_like(xi)

    # ξ ≈ 0 → prox factor (1 − H) acts on z ≈ 0, value of ρ is irrelevant;
    # we keep ρ = 0 (equivalently any value) so output stays ≈ 0.
    safe = xi > 1e-12
    if not np.any(safe):
        return rho

    xi_s = xi[safe]
    xi2 = xi_s * xi_s

    c = eps2 / xi2 + 2.0 * sigma / (mu * xi2)
    d = -eps2 / xi2

    # Depress cubic: substitute ρ = t + 1/3   ⇒   t³ + p·t + q = 0
    p = c - 1.0 / 3.0
    q = -2.0 / 27.0 + c / 3.0 + d

    # Cardano discriminant (three real roots ⇔ Δ > 0 ⇔ q²/4 + p³/27 < 0).
    disc = q * q / 4.0 + (p ** 3) / 27.0

    rho_s = np.empty_like(xi_s)
    three_real = disc < 0
    one_real = ~three_real

    if np.any(three_real):
        p_t = p[three_real]
        q_t = q[three_real]
        xi2_t = xi2[three_real]
        # Trigonometric form (requires p < 0, guaranteed when disc < 0):
        #   t_k = 2·√(−p/3)·cos((1/3)·arccos((3q/2p)·√(−3/p)) − 2πk/3)
        r = 2.0 * np.sqrt(-p_t / 3.0)
        arg = (3.0 * q_t) / (2.0 * p_t) * np.sqrt(-3.0 / p_t)
        arg = np.clip(arg, -1.0, 1.0)
        phi = np.arccos(arg) / 3.0
        t0 = r * np.cos(phi)
        t1 = r * np.cos(phi - 2.0 * np.pi / 3.0)
        t2 = r * np.cos(phi - 4.0 * np.pi / 3.0)
        r0 = t0 + 1.0 / 3.0
        r1 = t1 + 1.0 / 3.0
        r2 = t2 + 1.0 / 3.0

        # Minimiser selection
        def _obj(rr):
            return (mu / (2.0 * sigma)) * (rr - 1.0) ** 2 * xi2_t \
                   + np.log(rr * rr * xi2_t + eps2)

        o0 = _obj(r0)
        o1 = _obj(r1)
        o2 = _obj(r2)
        stack_r = np.stack([r0, r1, r2], axis=0)
        stack_o = np.stack([o0, o1, o2], axis=0)
        best = np.argmin(stack_o, axis=0)
        rho_s[three_real] = np.take_along_axis(
            stack_r, best[None], axis=0)[0]

    if np.any(one_real):
        p_o = p[one_real]
        q_o = q[one_real]
        disc_o = disc[one_real]
        sqd = np.sqrt(np.maximum(disc_o, 0.0))
        u3 = -q_o / 2.0 + sqd
        v3 = -q_o / 2.0 - sqd
        t = np.cbrt(u3) + np.cbrt(v3)
        rho_s[one_real] = t + 1.0 / 3.0

    # Clip to [0, 1]: out of this range the prox (1 − ρ) loses meaning,
    # and a tiny numerical overshoot is possible near the endpoints.
    rho_s = np.clip(rho_s, 0.0, 1.0)
    rho[safe] = rho_s
    return rho


def _build_h_lut(mu: float, eps: float, sigma: float,
                 xi_max: float = 2.0, n_grid: int = 4096) -> tuple:
    """
    Pre-compute H(ξ) on a uniform grid for fast interpolation.

    The paper recommends a look-up table (LUT) because, inside the inner
    loop, ``μ, ε, σ`` are fixed and only ``ξ`` changes per pixel.  We
    sample ξ on ``[0, xi_max]`` (values of ξ above that threshold give
    H ≈ 1 to numerical precision).

    Returns
    -------
    xi_grid : (n_grid,) ndarray — sample points
    h_grid  : (n_grid,) ndarray — H(xi_grid)
    """
    xi_grid = np.linspace(0.0, xi_max, n_grid, dtype=np.float64)
    h_grid = _h_function(xi_grid, mu=mu, eps=eps, sigma=sigma)
    return xi_grid, h_grid


def _h_lut_apply(xi: np.ndarray, xi_grid: np.ndarray,
                 h_grid: np.ndarray) -> np.ndarray:
    """Vectorised 1-D linear interpolation of a pre-computed H-LUT."""
    return np.interp(xi, xi_grid, h_grid,
                     left=h_grid[0], right=h_grid[-1])


def blind_pd(f: np.ndarray, MK: int, NK: int, beta: float,
             u: np.ndarray, k: np.ndarray,
             outer_iters: int = 30,
             inner_iters: int = 50,
             tau_param: float = 1e-3,
             k_step: float = 1e-3,
             theta: float = 1.0,
             pd_tau: float = None,
             pd_sigma: float = None,
             h_mode: str = 'closed',
             h_lut_size: int = 4096,
             h_lut_xi_max: float = 4.0,
             blind_denoise_fn=None,
             progress_callback=None) -> tuple:
    """
    Primal-dual blind deconvolution  —  Table 1 of Perrone & Favaro (2016),
    "A Logarithmic Image Prior for Blind Deconvolution", IJCV 117.

    Solves the non-convex log-TV energy (eq. 12) directly via the
    Chambolle-Pock / Möllenhoff primal-dual splitting (no MM outer
    majorisation).  For each outer iteration, a fixed kernel k is used
    and the inner loop performs N₀ primal-dual steps (Table 1):

        z₁^{n+1} = (z₁^n + σ·(k * ū^n − f)) / (1 + σ)

        ζ        = z₂^n + σ·∇ū^n
        ξ        = ‖ζ‖ / σ                                   (per-pixel)
        z₂^{n+1} = (1 − H(ξ, μ, ε, σ)) · ζ

        ũ^{n+1}  = ũ^n − τ·( k₋ * z₁^{n+1}  +  ∇* z₂^{n+1} )
        ū^{n+1}  = ũ^{n+1} + θ·(ũ^{n+1} − ũ^n)

    where ``∇* = −div``.  After the inner loop the kernel step follows
    ``blind.m`` — projected gradient descent onto the simplex.

    Parameters
    ----------
    f           : (M, N) blurry image
    MK, NK      : kernel spatial support
    beta        : data-fidelity weight λ (eq. 12).  Drives μ = β
                  (see "Note on μ" below).
    u           : (M+MK-1, N+NK-1) initial (padded) sharp-image estimate
    k           : (MK, NK) initial kernel
    outer_iters : T — outer iterations (kernel updates).  Paper leaves T
                  symbolic; default 30 is empirically sufficient at full
                  resolution (more is needed for coarse-to-fine pyramids
                  via ``coarse_to_fine``).
    inner_iters : N₀ — inner PD iterations per outer step.  Paper leaves
                  N₀ symbolic; for PD splitting the inner saddle-point
                  must actually be approached, so the default is large
                  (50) — much larger than for the MM solver, where the
                  inner gradient-descent loop only needs a few steps.
    tau_param   : ε of the log prior (eq. 11).  Following the MATLAB
                  reference we expose this as ``tau`` for parity with
                  the MM solver.
    k_step      : kernel gradient-descent step-size scaling
    theta       : PD over-relaxation parameter, θ ∈ (0, 1] (paper: 1)
    pd_tau      : primal step.  ``None`` → balanced default 0.99/√‖K‖²
                  ≈ 0.33 (see step-size note below).
    pd_sigma    : dual step.  ``None`` → equal to ``pd_tau``.
    blind_denoise_fn : optional callable applied to u before the kernel
                  gradient (same contract as in ``blind``/``blind_cv``).

    Returns
    -------
    u : (M+MK-1, N+NK-1) estimated sharp image (still padded)
    k : (MK, NK) estimated kernel

    Notes on step sizes
    -------------------
    The PD scheme has K = (k*, ∇), so
        ‖K‖² ≤ ‖k‖_op² + ‖∇‖² ≤ 1 + 8 = 9
    for a normalised non-negative k and Neumann forward-difference ∇.
    The Chambolle-Pock convergence condition (Perrone & Favaro eq. 19,
    Chambolle-Pock 2010 Thm. 1) is

        τ · σ · ‖K‖² < 1

    The paper does NOT give numerical defaults for τ, σ — it only states
    the constraint above.  We pick the **balanced** choice

        τ = σ = 0.99 / √‖K‖² = 0.99 / 3 ≈ 0.33

    which gives τσ‖K‖² ≈ 0.98 — strictly below 1, but as close as
    practical (Chambolle-Pock 2010 §6.2 recommends operating near the
    boundary for fastest convergence on negative-curvature problems).

    Important: β does NOT enter the primal step nor F₁* (see "Note on
    μ" below); it only enters the dual update via H(·, μ=β, ε, σ).
    Therefore the step sizes do NOT need to scale with β — the small
    ``τ = 0.005`` used in earlier code revisions was a (slow) safety
    fallback, not a paper recommendation.

    Note on μ
    ---------
    In problem (12) of the paper,  λ  multiplies the data term.  After
    rescaling to problem (16) — the form actually solved by the PD
    saddle-point — the data-fidelity coefficient becomes ½ and the log
    prior is weighted by ``1/μ`` with ``μ = 2λ/p``.  For the standard
    ``p = 1`` this gives ``μ = 2λ``.

    Throughout this codebase (following ``blind.m``) the parameter
    ``beta`` plays the role of ``2·λ`` of problem (12).  Indeed the MM
    gradient ``β · k₋ · (k*u−f)`` is the gradient of
    ``(β/2)·‖k*u−f‖² + prior``, so the effective  λ_problem-12  is
    ``β/2``.  To keep the **same** regularisation-to-data ratio between
    MM and PD solvers (so that the same ``lambda_val`` parameter gives
    comparable behaviour), we set

            μ = β   (=  2 · (β/2)  =  2 · λ_effective ).
    """
    epsilon = float(tau_param)   # ε of the log prior (eq. 11)
    # ``beta`` here plays the role of 2·λ in problem (12) — see docstring.
    mu = float(beta)

    # ── Step sizes ──────────────────────────────────────────────────────
    # Paper-faithful balanced default:  τ = σ = 0.99 / √‖K‖² ≈ 0.33,
    # giving τσ‖K‖² ≈ 0.98 < 1  (eq. 19, paper).  See step-size note in
    # the docstring for why β does NOT enter the choice of τ, σ.
    K_norm2 = 9.0  # ‖k*‖² + ‖∇‖² ≤ 1 + 8 = 9
    default_step = 0.99 / np.sqrt(K_norm2)
    tau_pd = float(pd_tau) if pd_tau is not None else default_step
    sigma_pd = float(pd_sigma) if pd_sigma is not None else tau_pd

    # ── H-evaluator (closed-form vs LUT) ────────────────────────────────
    # Paper (sec. 5) recommends a look-up table for speed: μ, ε, σ are
    # fixed inside the inner loop, so H(ξ) can be precomputed once.
    # Mathematically LUT ≡ closed-form; the only difference is speed
    # (LUT ≈ O(1) per pixel via np.interp) and a controllable
    # interpolation error.
    h_mode_l = str(h_mode).lower()
    if h_mode_l == 'lut':
        _xi_grid, _h_grid = _build_h_lut(
            mu=mu, eps=epsilon, sigma=sigma_pd,
            xi_max=float(h_lut_xi_max), n_grid=int(h_lut_size))
        def _H(xi_arr):
            return _h_lut_apply(xi_arr, _xi_grid, _h_grid)
    elif h_mode_l == 'closed':
        def _H(xi_arr):
            return _h_function(xi_arr, mu, epsilon, sigma_pd)
    else:
        raise ValueError(f"h_mode must be 'closed' or 'lut', got {h_mode!r}")

    # ── Dual / primal state ─────────────────────────────────────────────
    Mu, Nu = u.shape
    z1 = np.zeros_like(f)
    z2x = np.zeros((Mu, Nu))
    z2y = np.zeros((Mu, Nu))
    u_tilde = u.copy()
    u_bar = u.copy()    

    # ── Kernel-centering grids (as in blind_cv) ─────────────────────────
    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target = (MK - 1) / 2.0
    cx_target = (NK - 1) / 2.0

    for it in range(outer_iters):
        for itt in range(inner_iters):
            # ── Dual update z₁ (data),  Table 1, line 1:
            #   z₁ ← (z₁ + σ(Kū − f)) / (1 + σ)
            # The paper normalises the data term to F₁(v)=½‖v−f‖², so
            # F₁*(z)=½‖z‖²+⟨z,f⟩  and  prox_{σF₁*}(w) = (w−σf)/(1+σ).
            # All of the regularisation strength (λ/β) is carried by μ in the
            # H-function below, not here.
            Kub = convn_valid(u_bar, k)                    # k * ū, valid
            z1 = (z1 + sigma_pd * (Kub - f)) / (1.0 + sigma_pd)

            # ── Dual update z₂ (gradient / log prior)
            gx, gy = _grad_neumann(u_bar)
            zx = z2x + sigma_pd * gx
            zy = z2y + sigma_pd * gy
            # ξ = ‖ζ‖ / σ   per pixel
            xi = np.sqrt(zx * zx + zy * zy) / sigma_pd
            H = _H(xi)
            scale = 1.0 - H
            z2x = scale * zx
            z2y = scale * zy

            # ── Primal update
            #   ũ ← ũ − τ·( k₋ * z₁  +  ∇* z₂ ),   ∇* = −div
            Kstar_z1 = convn_full(z1, np.rot90(k, 2))
            div_z2 = _div_neumann(z2x, z2y)
            u_new = u_tilde - tau_pd * (Kstar_z1 - div_z2)

            # Over-relaxation
            u_bar = u_new + theta * (u_new - u_tilde)
            u_tilde = u_new

        u = u_tilde

        # ── Kernel step (Chan-Wong / blind.m style) ────────────────────
        u_dk = blind_denoise_fn(u) if blind_denoise_fn is not None else u
        synth = convn_valid(u_dk, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u_dk, 2), err)
        alpha_k = k_step * (k.max() + 1.0 / k.size) / \
                  (np.abs(gradk).max() + 1e-30)
        k = k - alpha_k * gradk
        k = np.maximum(k, 0.0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        # ── Kernel centre-of-mass re-centring (same trick as blind_cv) ─
        k_sum_c = k.sum()
        if k_sum_c > 0:
            cy = (ys * k).sum() / k_sum_c
            cx = (xs * k).sum() / k_sum_c
            dy = int(round(cy_target - cy))
            dx = int(round(cx_target - cx))
            if dy != 0 or dx != 0:
                k = np.roll(np.roll(k, dy, axis=0), dx, axis=1)
                u = np.roll(np.roll(u, -dy, axis=0), -dx, axis=1)
                u_tilde = np.roll(np.roll(u_tilde, -dy, axis=0), -dx, axis=1)
                u_bar = np.roll(np.roll(u_bar, -dy, axis=0), -dx, axis=1)
                z2x = np.roll(np.roll(z2x, -dy, axis=0), -dx, axis=1)
                z2y = np.roll(np.roll(z2y, -dy, axis=0), -dx, axis=1)

        if progress_callback is not None:
            try:
                progress_callback({
                    'event': 'iter',
                    'iter': it,
                    'kernel': k.copy(),
                    'beta': beta,
                })
            except Exception:
                pass

    return u, k


# ─────────────────────────────────────────────────────────────────────────────
# buildPyramid.m  →  build_pyramid
# ─────────────────────────────────────────────────────────────────────────────

def _make_odd(val: int) -> int:
    """Force integer to odd by subtracting 1 if even."""
    return val - 1 if val % 2 == 0 else val


def build_pyramid(f: np.ndarray, MK: int, NK: int,
                  lam: float, lambda_mult: float,
                  scale_mult: float = 1.4142135623730951):
    """
    Build a coarse-to-fine pyramid of images, kernel sizes, and λ values.

    Equivalent to MATLAB ``buildPyramid.m``.

    Parameters
    ----------
    f           : (M, N) blurry image at full resolution
    MK, NK      : full-resolution kernel size
    lam         : λ at the finest (scale-1) level
    lambda_mult : λ multiplier between levels  (lambdaMultiplier = 2.1)
    scale_mult  : kernel-size divider between levels (kernelSizeMultiplier = √2)

    Returns
    -------
    fp      : list of downscaled images  [scale-0 … scale-S]
    Mp, Np  : list of image sizes per scale
    MKp, NKp: list of kernel sizes per scale
    lambdas : list of λ values per scale
    scales  : total number of pyramid levels (int)

    Notes
    -----
    MATLAB uses ``round()`` which rounds to nearest, ties to even.
    Python's built-in ``round()`` is banker's rounding (identical behaviour).

    MATLAB ``imresize(f,[M N],'bicubic')`` → ``imresize_matlab(f,(M,N))``.
    """
    M, N = f.shape[:2]
    smallest_scale = 3

    fp = [f]
    Mp = [M]
    Np = [N]
    MKp = [MK]
    NKp = [NK]
    lambdas = [lam]

    num_scales = 1  # current count of levels

    while MKp[num_scales - 1] > smallest_scale and NKp[num_scales - 1] > smallest_scale:
        prev = num_scales - 1  # index of previous level

        # λ decreases toward coarser scales
        lambdas.append(lambdas[prev] / lambda_mult)

        # Kernel dimensions: divide and force odd
        new_mk = round(MKp[prev] / scale_mult)
        new_nk = round(NKp[prev] / scale_mult)
        new_mk = _make_odd(new_mk)
        new_nk = _make_odd(new_nk)

        # Avoid stalling: if a dimension didn't decrease, subtract 2
        if new_nk == NKp[prev]:
            new_nk -= 2
        if new_mk == MKp[prev]:
            new_mk -= 2

        # Floor at smallest_scale
        new_mk = max(new_mk, smallest_scale)
        new_nk = max(new_nk, smallest_scale)

        MKp.append(new_mk)
        NKp.append(new_nk)

        # Image dimensions scale proportionally to kernel change
        factor_m = MKp[prev] / new_mk
        factor_n = NKp[prev] / new_nk

        new_m = round(Mp[prev] / factor_m)
        new_n = round(Np[prev] / factor_n)
        # Force odd
        new_m = _make_odd(new_m)
        new_n = _make_odd(new_n)

        Mp.append(new_m)
        Np.append(new_n)

        # Down-scale image from the *original* f (not recursively)
        fp.append(imresize_matlab(f, (new_m, new_n)))

        num_scales += 1

    return fp, Mp, Np, MKp, NKp, lambdas, num_scales


# ─────────────────────────────────────────────────────────────────────────────
# coarseToFine.m  →  coarse_to_fine
# ─────────────────────────────────────────────────────────────────────────────

def coarse_to_fine(f: np.ndarray, MK: int, NK: int,
                   blind_params: dict, ctf_params: dict,
                   verbose: bool = False, method: str = 'mm',
                   blind_denoise_fn=None,
                   progress_callback=None):
    """
    Multi-scale coarse-to-fine blind deconvolution.

    Equivalent to MATLAB ``coarseToFine.m``.

    Parameters
    ----------
    f : (M, N) blurry image  (already preprocessed: double, odd dims, etc.)
    MK, NK : kernel support at finest level
    blind_params : dict with keys used by ``blind()``:
        outer_iters, inner_iters, tau, k_step, u_step
        (k_step and u_step are **arrays** with one entry per step-phase)
    ctf_params : dict with keys:
        final_lambda    — λ at finest scale
        lambda_mult     — λ multiplier between scales   (default 2.1)
        scale_mult      — kernel-size divider            (default √2)
    verbose : if True, print progress

    Returns
    -------
    u : (M+MK-1, N+NK-1) estimated sharp image (padded)
    k : (MK, NK) estimated kernel

    MATLAB → Python notes
    ---------------------
    * ``padarray(f,[floor(MK/2) floor(NK/2)],'replicate')``
      →  ``pad_replicate(f, MK//2, NK//2)``
    * ``ones(MK,NK)/MK/NK`` = uniform kernel
      → ``np.ones((MK,NK)) / (MK * NK)``

    Multi-step-size handling:
        In ``deblur.m``, ``params.k_step`` and ``params.u_step`` are vectors
        (e.g. [1e-2, 5e-3, 1e-3, 5e-4]).  In ``blind.m`` the outer loop
        ``for i = 1:length(params.k_step)`` runs the *full set of outer_iters*
        once per step-size entry.  Here we replicate that by calling ``blind()``
        once per step-size element at each pyramid level.
    """
    final_lambda = ctf_params.get('final_lambda')
    lambda_mult = ctf_params.get('lambda_mult', 2.1)
    scale_mult = ctf_params.get('scale_mult', np.sqrt(2))

    # Build pyramid
    fp, Mp, Np, MKp, NKp, lambdas, num_scales = build_pyramid(
        f, MK, NK, final_lambda, lambda_mult, scale_mult)

    # Initial estimates
    u = pad_replicate(f, MK // 2, NK // 2)
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    # Extract step-size arrays
    k_steps = blind_params.get('k_step', np.array([1e-3]))
    u_steps = blind_params.get('u_step', np.array([1e-3]))
    if np.isscalar(k_steps):
        k_steps = np.array([k_steps])
    if np.isscalar(u_steps):
        u_steps = np.array([u_steps])

    outer_iters = blind_params.get('outer_iters', 140)
    inner_iters = blind_params.get('inner_iters', 5)
    tau = blind_params.get('tau', 1e-3)

    # ── Process from coarsest to finest ──
    # MATLAB:  for scale = scales:-1:1   (scales is the coarsest, 1 the finest)
    for scale_idx in range(num_scales - 1, -1, -1):
        Ms = Mp[scale_idx]
        Ns = Np[scale_idx]
        MKs = MKp[scale_idx]
        NKs = NKp[scale_idx]

        # Resize current estimates to match this scale's dimensions
        # u has spatial size (Ms + MKs - 1,  Ns + NKs - 1)
        u = imresize_matlab(u, (Ms + MKs - 1, Ns + NKs - 1))
        k = imresize_matlab(k, (MKs, NKs))
        # Project kernel: non-negative, normalised
        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        fs = fp[scale_idx]
        lam = lambdas[scale_idx]

        if verbose:
            print(f"scale: {scale_idx}  lambda: {lam:.4f}  "
                  f"MKs: {MKs}  NKs: {NKs}  outer_iters: {outer_iters}")

        # Run solver for each step-size phase
        # MATLAB outer loop:  for i=1:length(params.k_step)
        # Build a scale-aware callback wrapper so each event carries
        # the current scale index and total number of scales.
        if progress_callback is not None:
            def _scale_cb(ev, _s=scale_idx, _ns=num_scales):
                ev['scale'] = _s
                ev['num_scales'] = _ns
                progress_callback(ev)
            _cb = _scale_cb
        else:
            _cb = None

        for phase in range(len(k_steps)):
            if method == 'mm':
                u, k = blind(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau=tau,
                    k_step=float(k_steps[phase]),
                    u_step=float(u_steps[phase]),
                    blind_denoise_fn=blind_denoise_fn,
                    progress_callback=_cb,
                )
            elif method == 'pd':
                u, k = blind_pd(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    blind_denoise_fn=blind_denoise_fn,
                    k_step=float(k_steps[phase]),
                    pd_tau=blind_params.get('pd_tau', None),
                    pd_sigma=blind_params.get('pd_sigma', None),
                    h_mode=blind_params.get('h_mode', 'closed'),
                    h_lut_size=blind_params.get('h_lut_size', 4096),
                    h_lut_xi_max=blind_params.get('h_lut_xi_max', 4.0),
                    progress_callback=_cb,
                )
            elif method == 'cv':
                u, k = blind_cv(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    blind_denoise_fn=blind_denoise_fn,
                    k_step=float(k_steps[phase]),
                    progress_callback=_cb,
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose 'mm', 'pd', or 'cv'.")

    return u, k


# ═════════════════════════════════════════════════════════════════════════════
# FFT-related helpers for non-blind deconvolution
# ═════════════════════════════════════════════════════════════════════════════

from numpy.fft import fft2, ifft2
from scipy.fft import dstn, idstn

_OPT_FFT_LUT = None


def _build_opt_fft_lut(max_n=4096):
    """Build LUT mapping n -> next efficient FFT size (products of 2,3,5,7)."""
    efficient = set()
    p2 = 1
    while p2 <= max_n:
        p3 = 1
        while p2 * p3 <= max_n:
            p5 = 1
            while p2 * p3 * p5 <= max_n:
                p7 = 1
                while p2 * p3 * p5 * p7 <= max_n:
                    efficient.add(p2 * p3 * p5 * p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    eff_sorted = sorted(efficient)
    lut = np.zeros(max_n + 1, dtype=np.int64)
    idx = 0
    for n_ in range(1, max_n + 1):
        while idx < len(eff_sorted) and eff_sorted[idx] < n_:
            idx += 1
        lut[n_] = eff_sorted[idx] if idx < len(eff_sorted) else n_
    return lut


def opt_fft_size(n) -> np.ndarray:
    """Optimal FFT data length(s) — smallest efficient size >= n."""
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
# wrap_boundary_liu (Liu & Jia ICIP 2008, Cho implementation)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """Solve Laplace eq. with Dirichlet BC via DST-I."""
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
    """Pad image so boundaries are circularly smooth for FFT-based deconv."""
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

        ret[:H, :W, ch] = HG
        ret[H:, :W, ch] = A2[1:-1, :]
        ret[:H, W:, ch] = B2[:, 1:-1]

        if H_w > 0 and W_w > 0:
            r_C = np.zeros((H_w + 2, W_w + 2), dtype=np.float64)
            r_C[0, :-1] = ret[H - 1, W - 1:, ch]
            r_C[0, -1]  = ret[H - 1, 0, ch]
            r_C[-1, :-1] = ret[0, W - 1:, ch]
            r_C[-1, -1]  = ret[0, 0, ch]
            r_C[:-1, 0]  = ret[H - 1:, W - 1, ch]
            r_C[-1, 0]   = ret[0, W - 1, ch]
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
    """TV-l2 deblurring via ADM/Split Bregman with anisotropic TV."""
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
# Bilateral filter
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size, sigma):
    """2-D Gaussian kernel."""
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
# Ringing artifacts removal (Pan et al. CVPR 2014)
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
