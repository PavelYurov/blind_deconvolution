# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

"""
solvers.pyx

Core solver functions for the LIP (Logarithmic Image Prior) blind deconvolution.
(Fast Cython version)
"""

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt

from .utils import (
    shft,
    convn_valid,
    convn_full,
    pad_replicate,
    imresize_matlab,
    psf2otf,
)


# ─────────────────────────────────────────────────────────────────────────────
# gradTVEM.m  →  grad_tv_em (FAST CYTHON VERSION)
# ─────────────────────────────────────────────────────────────────────────────

cdef inline double get_diff(double[:, ::1] arr, int i, int j, int dy, int dx, int M, int N) noexcept:
    """Эквивалент shft() без выделения памяти."""
    if 0 <= i + dy < M and 0 <= j + dx < N:
        return arr[i + dy, j + dx] - arr[i, j]
    return 0.0

def grad_tv_em(cnp.ndarray[cnp.float64_t, ndim=2] u_np, 
               cnp.ndarray[cnp.float64_t, ndim=2] ut_np, 
               double epsilon=1e-3, double tau=1e-1):
    
    u_np = np.ascontiguousarray(u_np, dtype=np.float64)
    ut_np = np.ascontiguousarray(ut_np, dtype=np.float64)

    cdef int M = u_np.shape[0]
    cdef int N = u_np.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] grad_out = np.zeros((M, N), dtype=np.float64)

    cdef double[:, ::1] u = u_np
    cdef double[:, ::1] ut = ut_np
    cdef double[:, ::1] grad = grad_out

    cdef int i, j, d, dx, dy
    cdef double ux1, uy1, TV1, du1, ux2, uy2, TV2, du2, ux3, uy3, TV3, du3
    cdef double utx1, uty1, TVt1, utx2, uty2, TVt2, utx3, uty3, TVt3, sum_val

    cdef int deltas[4][2]
    deltas[0][0] = 1;  deltas[0][1] = 1
    deltas[1][0] = -1; deltas[1][1] = 1
    deltas[2][0] = 1;  deltas[2][1] = -1
    deltas[3][0] = -1; deltas[3][1] = -1

    for i in range(M):
        for j in range(N):
            sum_val = 0.0
            for d in range(4):
                dx = deltas[d][0]
                dy = deltas[d][1]

                # case 1
                ux1 = get_diff(u, i, j, 0, dx, M, N)
                uy1 = get_diff(u, i, j, dy, 0, M, N)
                TV1 = sqrt(epsilon + ux1*ux1 + uy1*uy1)
                du1 = -ux1 - uy1

                utx1 = get_diff(ut, i, j, 0, dx, M, N)
                uty1 = get_diff(ut, i, j, dy, 0, M, N)
                TVt1 = sqrt(epsilon + utx1*utx1 + uty1*uty1)

                # case 2
                ux2 = -get_diff(u, i, j, 0, -dx, M, N)
                uy2 = get_diff(u, i, j, dy, -dx, M, N) - get_diff(u, i, j, 0, -dx, M, N)
                TV2 = sqrt(epsilon + ux2*ux2 + uy2*uy2)
                du2 = ux2

                utx2 = -get_diff(ut, i, j, 0, -dx, M, N)
                uty2 = get_diff(ut, i, j, dy, -dx, M, N) - get_diff(ut, i, j, 0, -dx, M, N)
                TVt2 = sqrt(epsilon + utx2*utx2 + uty2*uty2)

                # case 3
                ux3 = get_diff(u, i, j, -dy, dx, M, N) - get_diff(u, i, j, -dy, 0, M, N)
                uy3 = -get_diff(u, i, j, -dy, 0, M, N)
                TV3 = sqrt(epsilon + ux3*ux3 + uy3*uy3)
                du3 = uy3

                utx3 = get_diff(ut, i, j, -dy, dx, M, N) - get_diff(ut, i, j, -dy, 0, M, N)
                uty3 = -get_diff(ut, i, j, -dy, 0, M, N)
                TVt3 = sqrt(epsilon + utx3*utx3 + uty3*uty3)

                sum_val += (du1 / TV1 / (tau + TVt1) +
                            du2 / TV2 / (tau + TVt2) +
                            du3 / TV3 / (tau + TVt3))

            grad[i, j] = sum_val / 4.0

    return grad_out


# ─────────────────────────────────────────────────────────────────────────────
# blind.m  →  blind (FAST CYTHON VERSION)
# ─────────────────────────────────────────────────────────────────────────────

def blind(f, MK, NK, beta, u, k, outer_iters=140, inner_iters=5, 
          tau=1e-3, k_step=1e-3, u_step=1e-3, blind_denoise_fn=None,
          progress_callback=None):
    
    cdef double epsilon = 1e-3
    u = np.ascontiguousarray(u, dtype=np.float64)
    k = np.ascontiguousarray(k, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)

    for it in range(outer_iters):
        ut = u.copy()
        for itt in range(inner_iters):
            synth = convn_valid(u, k)
            err = synth - f
            gradu = (beta * convn_full(err, np.rot90(k, 2))
                     + grad_tv_em(u, ut, epsilon, tau))
            
            dt = u_step * (u.max() + 1.0 / u.size) / (np.abs(gradu).max() + 1e-30)
            u = u - dt * gradu

            synth = convn_valid(u, k)
            err = synth - f
            u_dk = blind_denoise_fn(u) if blind_denoise_fn is not None else u
            gradk = convn_valid(np.rot90(u_dk, 2), err)
            
            alpha = k_step * (k.max() + 1.0 / k.size) / (np.abs(gradk).max() + 1e-30)
            k = k - alpha * gradk
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
# Condat-Vũ variant on MM-majorant  →  blind_cv  (FAST CYTHON VERSION)
# (own construction — NOT the PD algorithm from the paper; see ``blind_pd``
#  below for the paper-faithful Table 1 variant.)
# ─────────────────────────────────────────────────────────────────────────────

cdef void compute_w(double[:, ::1] u, double[:, ::1] w, double epsilon) noexcept:
    cdef int M = u.shape[0], N = u.shape[1], i, j, next_i, next_j
    cdef double gx, gy
    for i in range(M):
        next_i = i + 1 if i + 1 < M else 0
        for j in range(N):
            next_j = j + 1 if j + 1 < N else 0
            gx = u[i, next_j] - u[i, j]
            gy = u[next_i, j] - u[i, j]
            w[i, j] = sqrt(epsilon + gx*gx + gy*gy)

cdef void compute_pd_dual_step(double[:, ::1] u_bar, double[:, :, ::1] p, double[:, ::1] w, 
                               double tau_param, double sigma_pd) noexcept:
    cdef int M = u_bar.shape[0], N = u_bar.shape[1], i, j, next_i, next_j
    cdef double grad_x, grad_y, ptx, pty, norm_pt, radius, proj_scale
    for i in range(M):
        next_i = i + 1 if i + 1 < M else 0
        for j in range(N):
            next_j = j + 1 if j + 1 < N else 0
            grad_x = u_bar[i, next_j] - u_bar[i, j]
            grad_y = u_bar[next_i, j] - u_bar[i, j]

            ptx = p[0, i, j] + sigma_pd * grad_x
            pty = p[1, i, j] + sigma_pd * grad_y

            radius = 1.0 / (w[i, j] + tau_param)
            norm_pt = sqrt(ptx * ptx + pty * pty + 1e-30)
            proj_scale = radius / norm_pt if radius < norm_pt else 1.0

            p[0, i, j] = ptx * proj_scale
            p[1, i, j] = pty * proj_scale

cdef void compute_pd_primal_update(double[:, ::1] u, double[:, ::1] u_old, double[:, ::1] u_bar,
                                   double[:, ::1] grad_data, double[:, :, ::1] p,
                                   double tau_pd, double theta) noexcept:
    cdef int M = u.shape[0], N = u.shape[1], i, j, prev_i, prev_j
    cdef double div_p, u_new, u_prev
    for i in range(M):
        prev_i = i - 1 if i - 1 >= 0 else M - 1
        for j in range(N):
            prev_j = j - 1 if j - 1 >= 0 else N - 1
            div_p = (p[0, i, j] - p[0, i, prev_j]) + (p[1, i, j] - p[1, prev_i, j])

            u_prev = u[i, j]
            u_new = u_prev - tau_pd * grad_data[i, j] + tau_pd * div_p
            if u_new < 0.0: u_new = 0.0

            u_old[i, j] = u_prev
            u[i, j] = u_new
            u_bar[i, j] = u_new + theta * (u_new - u_prev)

def blind_cv(f, MK, NK, beta, u, k, outer_iters=140, inner_iters=5,
             tau_param=1e-3, k_step=1e-3, blind_denoise_fn=None,
             progress_callback=None):
    """
    Condat-Vũ primal-dual splitting on the MM-majorised weighted-TV
    subproblem (Cython-accelerated, own construction — NOT from the paper).

    Mirrors ``blind_cv`` in ``lip_denoise/solvers.py`` bit-for-bit:
        * outer loop:   MM weights ``w = √(ε + |∇u|²)`` recomputed from u,
                        radius ``1 / (w + τ_param)`` frozen for inner loop.
        * inner loop:   dual step (ball projection) + primal step
                        ``u ← [u − τ·∇f(u) + τ·div(p)]_+``, over-relaxed.
        * kernel step:  OUTSIDE inner loop, projected gradient descent,
                        centre-of-mass re-centring (k, u, p compensated).

    Convergence condition (Condat 2013, Thm. 3.1):
        ``1/τ − σ·‖∇‖² > L_f / 2`` with L_f = β, ‖∇‖² = 8.
        → σ = 0.99·β/16,  τ = 0.99/β.

    Boundary condition: **periodic** (to match the Python reference's
    ``np.roll`` formulation).  The data-fidelity gradient uses spatial
    valid/full convolutions, so those terms are boundary-correct.
    """
    cdef double epsilon = 1e-3
    u = np.ascontiguousarray(u, dtype=np.float64)
    k = np.ascontiguousarray(k, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)
    
    cdef int Mu = u.shape[0], Nu = u.shape[1]
    p = np.zeros((2, Mu, Nu), dtype=np.float64)
    w = np.zeros((Mu, Nu), dtype=np.float64)
    u_old = np.zeros((Mu, Nu), dtype=np.float64)

    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target, cx_target = (MK - 1) / 2.0, (NK - 1) / 2.0

    cdef double L_f = beta
    cdef double sigma_pd = 0.99 * L_f / 16.0
    cdef double tau_pd = 0.99 / L_f
    cdef double theta = 1.0

    for it in range(outer_iters):
        compute_w(u, w, epsilon)
        u_bar = u.copy()

        for itt in range(inner_iters):
            compute_pd_dual_step(u_bar, p, w, tau_param, sigma_pd)

            synth = convn_valid(u, k)
            err = synth - f
            grad_data = np.ascontiguousarray(beta * convn_full(err, np.rot90(k, 2)), dtype=np.float64)

            compute_pd_primal_update(u, u_old, u_bar, grad_data, p, tau_pd, theta)

        u_dk = blind_denoise_fn(u) if blind_denoise_fn is not None else u
        synth = convn_valid(u_dk, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u_dk, 2), err)
        
        alpha_k = k_step * (k.max() + 1.0 / k.size) / (np.abs(gradk).max() + 1e-30)
        k = k - alpha_k * gradk
        k = np.maximum(k, 0.0)
        k_sum = k.sum()
        if k_sum > 0: k /= k_sum

        k_sum_c = k.sum()
        if k_sum_c > 0:
            cy = (ys * k).sum() / k_sum_c
            cx = (xs * k).sum() / k_sum_c
            dy = int(round(cy_target - cy))
            dx = int(round(cx_target - cx))
            if dy != 0 or dx != 0:
                k = np.roll(k, dy, axis=0)
                k = np.roll(k, dx, axis=1)
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
# Implementation is pure NumPy (kept identical to the ``lip_denoise`` version
# — ``точь-в-точь как Python``).  The CV solver above carries the Cython
# speedups.
# ─────────────────────────────────────────────────────────────────────────────

def _grad_neumann(u):
    """Forward differences with Neumann (zero at last row/col) boundary."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div_neumann(px, py):
    """
    Divergence operator — adjoint of ``_grad_neumann`` under
    ``<∇u, p> = -<u, div p>``.  Standard Chambolle-2004 construction.
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


def _h_function(xi, mu, eps, sigma):
    """
    Solve eq. (25) of Perrone & Favaro (2016) per pixel:

        H(ξ, μ, ε, σ) = argmin_ρ  (μ/2σ)·(ρ−1)²·ξ²  +  log(ρ²·ξ² + ε²)

    Cardano closed-form in ρ (depressed cubic).  Identical numerics to the
    ``lip_denoise`` reference implementation.
    """
    xi = np.asarray(xi, dtype=np.float64)
    eps2 = eps * eps

    rho = np.zeros_like(xi)
    safe = xi > 1e-12
    if not np.any(safe):
        return rho

    xi_s = xi[safe]
    xi2 = xi_s * xi_s

    c = eps2 / xi2 + 2.0 * sigma / (mu * xi2)
    d = -eps2 / xi2

    p = c - 1.0 / 3.0
    q = -2.0 / 27.0 + c / 3.0 + d

    disc = q * q / 4.0 + (p ** 3) / 27.0

    rho_s = np.empty_like(xi_s)
    three_real = disc < 0
    one_real = ~three_real

    if np.any(three_real):
        p_t = p[three_real]
        q_t = q[three_real]
        xi2_t = xi2[three_real]
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

        def _obj(rr):
            return (mu / (2.0 * sigma)) * (rr - 1.0) ** 2 * xi2_t \
                   + np.log(rr * rr * xi2_t + eps2)

        o0 = _obj(r0); o1 = _obj(r1); o2 = _obj(r2)
        stack_r = np.stack([r0, r1, r2], axis=0)
        stack_o = np.stack([o0, o1, o2], axis=0)
        best = np.argmin(stack_o, axis=0)
        rho_s[three_real] = np.take_along_axis(stack_r, best[None], axis=0)[0]

    if np.any(one_real):
        p_o = p[one_real]
        q_o = q[one_real]
        disc_o = disc[one_real]
        sqd = np.sqrt(np.maximum(disc_o, 0.0))
        u3 = -q_o / 2.0 + sqd
        v3 = -q_o / 2.0 - sqd
        t = np.cbrt(u3) + np.cbrt(v3)
        rho_s[one_real] = t + 1.0 / 3.0

    rho_s = np.clip(rho_s, 0.0, 1.0)
    rho[safe] = rho_s
    return rho


def _build_h_lut(mu, eps, sigma, xi_max=4.0, n_grid=4096):
    """Pre-compute H(ξ) on a uniform grid for fast interpolation."""
    xi_grid = np.linspace(0.0, xi_max, n_grid, dtype=np.float64)
    h_grid = _h_function(xi_grid, mu=mu, eps=eps, sigma=sigma)
    return xi_grid, h_grid


def _h_lut_apply(xi, xi_grid, h_grid):
    """Vectorised 1-D linear interpolation of a pre-computed H-LUT."""
    return np.interp(xi, xi_grid, h_grid,
                     left=h_grid[0], right=h_grid[-1])


def blind_pd(f, MK, NK, beta, u, k,
             outer_iters=30, inner_iters=50,
             tau_param=1e-3, k_step=1e-3,
             theta=1.0, pd_tau=None, pd_sigma=None,
             h_mode='closed', h_lut_size=4096, h_lut_xi_max=4.0,
             blind_denoise_fn=None,
             progress_callback=None):
    """
    Primal-dual blind deconvolution — Table 1 of Perrone & Favaro (2016).

    Mirrors ``blind_pd`` in ``lip_denoise/solvers.py`` bit-for-bit.  Uses
    Neumann boundary conditions via ``_grad_neumann`` / ``_div_neumann``
    and the H-function from eq. (25).

    Step sizes (balanced, paper-faithful):
        ``pd_tau = pd_sigma = 0.99 / √‖K‖² ≈ 0.33``
        with ``‖K‖² ≤ ‖k*‖² + ‖∇‖² ≤ 1 + 8 = 9``.
    """
    epsilon = float(tau_param)
    mu = float(beta)

    K_norm2 = 9.0
    default_step = 0.99 / np.sqrt(K_norm2)
    tau_pd = float(pd_tau) if pd_tau is not None else default_step
    sigma_pd = float(pd_sigma) if pd_sigma is not None else tau_pd

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

    u = np.ascontiguousarray(u, dtype=np.float64)
    k = np.ascontiguousarray(k, dtype=np.float64)
    f = np.ascontiguousarray(f, dtype=np.float64)

    Mu, Nu = u.shape
    z1 = np.zeros_like(f)
    z2x = np.zeros((Mu, Nu))
    z2y = np.zeros((Mu, Nu))
    u_tilde = u.copy()
    u_bar = u.copy()

    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target = (MK - 1) / 2.0
    cx_target = (NK - 1) / 2.0

    for it in range(outer_iters):
        for itt in range(inner_iters):
            # z₁ ← (z₁ + σ(k*ū − f)) / (1 + σ)
            Kub = convn_valid(u_bar, k)
            z1 = (z1 + sigma_pd * (Kub - f)) / (1.0 + sigma_pd)

            # z₂ ← (1 − H(‖ζ‖/σ, μ, ε, σ)) · ζ   where ζ = z₂ + σ·∇ū
            gx, gy = _grad_neumann(u_bar)
            zx = z2x + sigma_pd * gx
            zy = z2y + sigma_pd * gy
            xi = np.sqrt(zx * zx + zy * zy) / sigma_pd
            H = _H(xi)
            scale = 1.0 - H
            z2x = scale * zx
            z2y = scale * zy

            # ũ ← ũ − τ (k₋ * z₁ + ∇* z₂),   ∇* = −div
            Kstar_z1 = convn_full(z1, np.rot90(k, 2))
            div_z2 = _div_neumann(z2x, z2y)
            u_new = u_tilde - tau_pd * (Kstar_z1 - div_z2)

            u_bar = u_new + theta * (u_new - u_tilde)
            u_tilde = u_new

        u = u_tilde

        # Kernel step (Chan-Wong / blind.m style)
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

        # Kernel centre-of-mass re-centring
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
    M, N = f.shape[:2]
    smallest_scale = 3

    fp = [f]
    Mp = [M]
    Np = [N]
    MKp = [MK]
    NKp = [NK]
    lambdas = [lam]

    num_scales = 1

    while MKp[num_scales - 1] > smallest_scale and NKp[num_scales - 1] > smallest_scale:
        prev = num_scales - 1

        lambdas.append(lambdas[prev] / lambda_mult)

        new_mk = round(MKp[prev] / scale_mult)
        new_nk = round(NKp[prev] / scale_mult)
        new_mk = _make_odd(new_mk)
        new_nk = _make_odd(new_nk)

        if new_nk == NKp[prev]:
            new_nk -= 2
        if new_mk == MKp[prev]:
            new_mk -= 2

        new_mk = max(new_mk, smallest_scale)
        new_nk = max(new_nk, smallest_scale)

        MKp.append(new_mk)
        NKp.append(new_nk)

        factor_m = MKp[prev] / new_mk
        factor_n = NKp[prev] / new_nk

        new_m = round(Mp[prev] / factor_m)
        new_n = round(Np[prev] / factor_n)
        new_m = _make_odd(new_m)
        new_n = _make_odd(new_n)

        Mp.append(new_m)
        Np.append(new_n)

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
    
    final_lambda = ctf_params.get('final_lambda')
    lambda_mult = ctf_params.get('lambda_mult', 2.1)
    scale_mult = ctf_params.get('scale_mult', np.sqrt(2))

    fp, Mp, Np, MKp, NKp, lambdas, num_scales = build_pyramid(
        f, MK, NK, final_lambda, lambda_mult, scale_mult)

    u = pad_replicate(f, MK // 2, NK // 2)
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    k_steps = blind_params.get('k_step', np.array([1e-3]))
    u_steps = blind_params.get('u_step', np.array([1e-3]))
    if np.isscalar(k_steps):
        k_steps = np.array([k_steps])
    if np.isscalar(u_steps):
        u_steps = np.array([u_steps])

    outer_iters = blind_params.get('outer_iters', 140)
    inner_iters = blind_params.get('inner_iters', 5)
    tau = blind_params.get('tau', 1e-3)

    for scale_idx in range(num_scales - 1, -1, -1):
        Ms = Mp[scale_idx]
        Ns = Np[scale_idx]
        MKs = MKp[scale_idx]
        NKs = NKp[scale_idx]

        u = imresize_matlab(u, (Ms + MKs - 1, Ns + NKs - 1))
        k = imresize_matlab(k, (MKs, NKs))
        
        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        fs = fp[scale_idx]
        lam = lambdas[scale_idx]

        if verbose:
            print(f"scale: {scale_idx}  lambda: {lam:.4f}  "
                  f"MKs: {MKs}  NKs: {NKs}  outer_iters: {outer_iters}")

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
                    outer_iters=blind_params.get('pd_outer_iters', 30),
                    inner_iters=blind_params.get('pd_inner_iters', 50),
                    tau_param=tau,
                    k_step=float(k_steps[phase]),
                    theta=blind_params.get('pd_theta', 1.0),
                    pd_tau=blind_params.get('pd_tau', None),
                    pd_sigma=blind_params.get('pd_sigma', None),
                    h_mode=blind_params.get('h_mode', 'closed'),
                    h_lut_size=blind_params.get('h_lut_size', 4096),
                    h_lut_xi_max=blind_params.get('h_lut_xi_max', 4.0),
                    blind_denoise_fn=blind_denoise_fn,
                    progress_callback=_cb,
                )
            elif method == 'cv':
                u, k = blind_cv(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    k_step=float(k_steps[phase]),
                    blind_denoise_fn=blind_denoise_fn,
                    progress_callback=_cb,
                )
            else:
                raise ValueError(
                    f"Unknown method {method!r}. Choose 'mm', 'pd', or 'cv'.")

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