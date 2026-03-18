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
          u_step: float = 1e-3) -> tuple:
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
            gradk = convn_valid(np.rot90(u, 2), err)
            alpha = k_step * (k.max() + 1.0 / k.size) / (np.abs(gradk).max() + 1e-30)
            k = k - alpha * gradk
            # projection: non-negative, sum-to-one
            k = np.maximum(k, 0.0)
            k_sum = k.sum()
            if k_sum > 0:
                k /= k_sum

    return u, k


# ─────────────────────────────────────────────────────────────────────────────
# Table 1  →  blind_pd  (Condat-Vũ primal-dual splitting)
# ─────────────────────────────────────────────────────────────────────────────

def blind_pd(f: np.ndarray, MK: int, NK: int, beta: float,
             u: np.ndarray, k: np.ndarray,
             outer_iters: int = 140,
             inner_iters: int = 5,
             tau_param: float = 1e-3,
             k_step: float = 1e-3) -> tuple:
    """
    PD (Primal-Dual) blind deconvolution — Condat-Vũ variant.

    Uses the same MM outer loop as ``blind()`` (majorise the log prior
    with a weighted TV), but solves the resulting weighted-TV subproblem
    with **Condat-Vũ primal-dual splitting** (gradient of data fidelity
    + dual update for the TV term).

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
        synth = convn_valid(u, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u, 2), err)
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
                   verbose: bool = False, method: str = 'mm'):
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
                )
            elif method == 'pd':
                u, k = blind_pd(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    k_step=float(k_steps[phase]),
                )

    return u, k
