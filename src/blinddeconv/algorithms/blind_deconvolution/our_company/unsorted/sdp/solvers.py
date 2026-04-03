"""
solvers.py

Core solver for the Convex SDP blind deconvolution algorithm.

Ported from MATLAB Convex_src/blindDeconvolve_implicit_2D.m

Reference:
    A. Ahmed, B. Recht, J. Romberg: "Blind Deconvolution Using Convex
    Programming", IEEE Trans. Inform. Theory, 2014. (arXiv:1211.5608)

    S. Burer, R. D. C. Monteiro: "A nonlinear programming algorithm for
    solving semidefinite programs via low-rank factorization",
    Math. Programming (B), Vol. 95, 2003. pp 329-357.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    minFunc (L-BFGS):
        MATLAB's minFunc by Mark Schmidt: unconstrained optimiser.
        Default corrections (Corr) = 100 for L-BFGS.
        Default maxFunEvals = 50000.
        → scipy.optimize.minimize(method='L-BFGS-B')
        We set maxcor=100 to match MATLAB's default.

    [Z(:); H(:)] packing:
        MATLAB's (:) is column-major.  We use ravel(order='F') for
        packing and reshape(order='F') for unpacking.

    norm(Z, 'fro')^2:
        For a matrix, Frobenius norm squared = sum of all |z_ij|^2.
        → np.linalg.norm(Z, 'fro')**2

    y' * dev  (complex inner product):
        MATLAB's ' is conjugate transpose.  y'*dev = conj(y)^T * dev.
        → np.vdot(y, dev)  which computes sum(conj(y) * dev).

    conj(yhat * ones(1, maxrank)):
        Broadcasting conj(yhat) across columns.
        → np.conj(yhat[:, np.newaxis]) * temp1

    fft2 / ifft2:
        MATLAB and numpy use the same unnormalised / 1/N convention.
        No difference.

    Initial infeasibility is computed in the SPATIAL domain:
        dev_spatial = ifft2( sum_i fft2(C1(Z_i)) .* fft2(C2(H_i)) ) - y
    Loop infeasibility is computed in the FREQUENCY domain:
        dev_freq = sum_i fft2(C1(Z_i)) .* fft2(C2(H_i)) / L - fft2(y) / L
    This matches the MATLAB code exactly (scale mismatch is intentional).
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, Callable

from scipy.optimize import minimize as sp_minimize

from .utils import vec, mat


# ═════════════════════════════════════════════════════════════════════════════
# Subproblem cost & gradient  (MATLAB: subproblem_cost)
# ═════════════════════════════════════════════════════════════════════════════

def _subproblem_cost(
    x_opt: np.ndarray,
    C1: Callable, C2: Callable,
    C1T: Callable, C2T: Callable,
    maxrank: int,
    meas_fft: np.ndarray,
    y_lagrange: np.ndarray,
    sigma: float,
    siglen: int,
    n1: int, n2: int,
    L1: int, L2: int,
) -> Tuple[float, np.ndarray]:
    """
    Compute the augmented Lagrangian cost and its gradient.

    This is an exact port of the nested ``subproblem_cost`` function
    inside ``blindDeconvolve_implicit_2D.m``.

    Parameters
    ----------
    x_opt : (n1*maxrank + n2*maxrank,) real ndarray
        Packed optimisation variable [vec(Z); vec(H)].
    C1, C2 : callable
        Forward operators.  C1: (n1,) → (L1, L2),  C2: (n2,) → (L1, L2).
    C1T, C2T : callable
        Adjoint operators.  C1T: (L1, L2) → (n1,),  C2T: (L1, L2) → (n2,).
    maxrank : int
        Factorisation rank.
    meas_fft : (siglen,) complex ndarray
        Frequency-domain measurement: vec(fft2(blurred_image)) / siglen.
    y_lagrange : (siglen,) complex ndarray
        Lagrange multiplier (frequency domain).
    sigma : float
        Penalty parameter.
    siglen : int
        Signal length L1*L2.
    n1, n2 : int
        Subspace dimensions.
    L1, L2 : int
        Image dimensions.

    Returns
    -------
    cost : float
        Augmented Lagrangian value (real).
    grad : (n1*maxrank + n2*maxrank,) real ndarray
        Gradient w.r.t. x_opt.
    """
    # ── Unpack Z, H (Fortran order, matching MATLAB reshape) ─────────────
    Z = x_opt[:n1 * maxrank].reshape((n1, maxrank), order='F')
    H = x_opt[n1 * maxrank:].reshape((n2, maxrank), order='F')

    # ── Equation error in frequency domain ───────────────────────────────
    #  MATLAB: dev = sum_i vec(fft2(C1(Z(:,i))) .* fft2(C2(H(:,i)))) / siglen
    #          dev = dev - meas_fft
    dev = np.zeros(siglen, dtype=np.complex128)
    for i in range(maxrank):
        fft_c1 = np.fft.fft2(C1(Z[:, i]))          # (L1, L2) complex
        fft_c2 = np.fft.fft2(C2(H[:, i]))          # (L1, L2) complex
        dev += vec(fft_c1 * fft_c2) / siglen
    dev -= meas_fft

    # ── Cost ─────────────────────────────────────────────────────────────
    #  MATLAB: mval = norm(Z,'fro')^2 + norm(H,'fro')^2
    #                 - 2*real(y'*dev) + sigma*norm(dev,'fro')^2
    cost = (np.linalg.norm(Z, 'fro') ** 2
            + np.linalg.norm(H, 'fro') ** 2
            - 2.0 * np.real(np.vdot(y_lagrange, dev))
            + sigma * np.linalg.norm(dev) ** 2)

    # ── Gradient ─────────────────────────────────────────────────────────
    #  MATLAB: yhat = y - sigma * dev
    yhat = y_lagrange - sigma * dev

    #  temp1(:,i) = vec(fft2(C2(H(:,i))))
    #  temp2(:,i) = vec(ifft2(C1(Z(:,i))))
    temp1 = np.zeros((siglen, maxrank), dtype=np.complex128)
    temp2 = np.zeros((siglen, maxrank), dtype=np.complex128)
    for i in range(maxrank):
        temp1[:, i] = vec(np.fft.fft2(C2(H[:, i])))
        temp2[:, i] = vec(np.fft.ifft2(C1(Z[:, i])))

    #  temp3 = conj(yhat * ones(1,maxrank)) .* temp1
    #  temp4 = (yhat * ones(1,maxrank)) .* temp2
    yhat_col = yhat[:, np.newaxis]                          # (siglen, 1)
    temp3 = np.conj(yhat_col) * temp1                       # (siglen, maxrank)
    temp4 = yhat_col * temp2                                # (siglen, maxrank)

    #  temp5(:,i) = C1T(fft2(mat(temp3(:,i))))
    #  temp6(:,i) = C2T(ifft2(mat(temp4(:,i))))
    temp5 = np.zeros((n1, maxrank), dtype=np.complex128)
    temp6 = np.zeros((n2, maxrank), dtype=np.complex128)
    for i in range(maxrank):
        img3 = mat(temp3[:, i], L1, L2)
        temp5[:, i] = C1T(np.fft.fft2(img3))
        img4 = mat(temp4[:, i], L1, L2)
        temp6[:, i] = C2T(np.fft.ifft2(img4))

    #  adjoint_times_H = temp5 / siglen
    #  adjoint_times_Z = temp6 * siglen
    adjoint_times_H = temp5 / siglen
    adjoint_times_Z = temp6 * siglen

    #  GradZ = 2*(Z - adjoint_times_H)
    #  GradH = 2*(H - adjoint_times_Z)
    GradZ = 2.0 * (Z - adjoint_times_H)
    GradH = 2.0 * (H - adjoint_times_Z)

    #  g = real([GradZ(:); GradH(:)])
    grad = np.real(np.concatenate([
        GradZ.ravel(order='F'),
        GradH.ravel(order='F'),
    ]))

    return float(np.real(cost)), grad


# ═════════════════════════════════════════════════════════════════════════════
# Main ALM solver  (MATLAB: blindDeconvolve_implicit_2D)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconvolve_implicit_2d(
    conv_zh: np.ndarray,
    C1: Callable, C2: Callable,
    maxrank: int,
    C1T: Callable, C2T: Callable,
    n1: int, n2: int,
    L1: int, L2: int,
    Z_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    pars: Optional[Dict] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the convex blind deconvolution problem via the Augmented
    Lagrangian Method (ALM) with Burer–Monteiro low-rank factorisation.

    Finds matrices Z (n1 × maxrank) and H (n2 × maxrank) such that

        conv_zh ≈ Σ_k circ_conv( C1·Z[:,k] , C2·H[:,k] )

    by solving:

        min  ||Z||_F² + ||H||_F²
        s.t. conv_zh = Σ_k circ_conv( C1·Z[:,k] , C2·H[:,k] )

    This is an exact port of ``blindDeconvolve_implicit_2D.m``.

    Parameters
    ----------
    conv_zh : (L,) real ndarray
        Vectorised blurred image (Fortran order).
    C1 : callable  (n1,) → (L1, L2)
        Image subspace operator (CC).
    C2 : callable  (n2,) → (L1, L2)
        Kernel subspace operator (BB).
    maxrank : int
        Factorisation rank.
    C1T : callable  (L1, L2) → (n1,)
        Adjoint of C1 (CCT).
    C2T : callable  (L1, L2) → (n2,)
        Adjoint of C2 (BBT).
    n1 : int
        Dimension of image subspace (N).
    n2 : int
        Dimension of kernel subspace (K).
    L1, L2 : int
        Image height and width.
    Z_init, H_init : ndarray, optional
        Starting points.  If None, initialised as 1e-2 * randn.
    pars : dict, optional
        Algorithm parameters (see defaults below).
    verbose : bool
        Print iteration diagnostics.

    Returns
    -------
    Z : (n1, maxrank) ndarray
    H : (n2, maxrank) ndarray
    """
    siglen = conv_zh.size
    assert siglen == L1 * L2, (
        f"conv_zh length {siglen} != L1*L2 = {L1 * L2}"
    )

    # ── Default ALM parameters (matching MATLAB) ────────────────────────
    if pars is None:
        pars = {}
    max_out_iter = pars.get('maxOutIter', 25)
    rmse_tol     = pars.get('rmseTol', 1e-8)
    sigma_init   = pars.get('sigmaInit', 1e4)
    lr1          = pars.get('LR1', 0.25)
    lr2          = pars.get('LR2', 10)
    prog_tol     = pars.get('progTol', 1e-3)
    num_bad      = pars.get('numbaditers', 6)
    max_fun      = pars.get('maxFunEvals', 50000)

    # ── Frequency-domain measurement ────────────────────────────────────
    #  MATLAB: meas_fft = vec(fft2(mat(conv_zh))) / siglen
    meas_fft = vec(np.fft.fft2(mat(conv_zh, L1, L2))) / siglen

    # ── Initialise Z, H ─────────────────────────────────────────────────
    if Z_init is not None:
        Z = Z_init.copy()
    else:
        Z = 1e-2 * np.random.randn(n1, maxrank)

    if H_init is not None:
        H = H_init.copy()
    else:
        H = 1e-2 * np.random.randn(n2, maxrank)

    # ── Lagrange multiplier and penalty parameter ───────────────────────
    y_lagrange = np.zeros(siglen, dtype=np.complex128)
    sigma = sigma_init

    # ── Initial infeasibility (spatial domain, matching MATLAB) ─────────
    #  MATLAB:
    #    dev = zeros(p,1);
    #    for i = 1:maxrank
    #        dev = dev + vec(fft2(mat(C1(Z(:,i)))) .* fft2(mat(C2(H(:,i)))));
    #    end
    #    dev = vec(ifft2(mat(dev))) - conv_zh;
    #    vOld = norm(dev,'fro')^2;
    dev_init = np.zeros(siglen, dtype=np.complex128)
    for i in range(maxrank):
        fft_c1 = np.fft.fft2(C1(Z[:, i]))
        fft_c2 = np.fft.fft2(C2(H[:, i]))
        dev_init += vec(fft_c1 * fft_c2)
    dev_spatial = vec(np.fft.ifft2(mat(dev_init, L1, L2))) - conv_zh
    vOld = np.linalg.norm(dev_spatial) ** 2

    v = vOld
    badcnt = 0

    # ── Diagnostics header ──────────────────────────────────────────────
    if verbose:
        print('|      |          |          | iter  | tot   |')
        print('| iter |  rmse    |  sigma   | time  | time  |')
        print('-' * 46)

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════
    #  Outer ALM loop
    # ══════════════════════════════════════════════════════════════════════
    for out_iter in range(1, max_out_iter + 1):
        t1 = time.time()

        # ── Pack Z, H into optimisation variable ────────────────────────
        x0 = np.concatenate([Z.ravel(order='F'), H.ravel(order='F')])

        # ── Inner minimisation: L-BFGS (matches MATLAB minFunc) ─────────
        result = sp_minimize(
            _subproblem_cost,
            x0,
            args=(C1, C2, C1T, C2T, maxrank, meas_fft,
                  y_lagrange, sigma, siglen, n1, n2, L1, L2),
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 500,
                'maxfun': max_fun,
                'maxcor': 100,        # match MATLAB minFunc default
                'ftol': 1e-15,        # very tight to let maxfun control
                'gtol': 1e-12,
                'disp': False,
            },
        )
        x_opt = result.x

        # ── Unpack Z, H ────────────────────────────────────────────────
        Z = x_opt[:n1 * maxrank].reshape((n1, maxrank), order='F')
        H = x_opt[n1 * maxrank:].reshape((n2, maxrank), order='F')

        # ── Equation error (frequency domain) ──────────────────────────
        #  MATLAB:
        #    dev = sum_i vec(fft2(C1(Z(:,i))).*fft2(C2(H(:,i))))/siglen
        #    dev = dev - meas_fft
        dev = np.zeros(siglen, dtype=np.complex128)
        for i in range(maxrank):
            fft_c1 = np.fft.fft2(C1(Z[:, i]))
            fft_c2 = np.fft.fft2(C2(H[:, i]))
            dev += vec(fft_c1 * fft_c2) / siglen
        dev -= meas_fft

        v_last = v
        v = np.linalg.norm(dev) ** 2

        # ── Progress check ──────────────────────────────────────────────
        if v_last > 0 and abs(v_last - v) / v_last < prog_tol:
            badcnt += 1
            if badcnt > num_bad:
                if verbose:
                    print('\nunable to make progress. terminating')
                break
        else:
            badcnt = 0

        # ── Diagnostics ────────────────────────────────────────────────
        rmse = np.sqrt(v / siglen)
        if verbose:
            print(f'| {out_iter:2d}   | {rmse:.2e} | {sigma:.2e} '
                  f'|  {time.time() - t1:3.0f}  |  {time.time() - t0:3.0f}  |')

        # ── Convergence / multiplier / penalty update ──────────────────
        if rmse < rmse_tol:
            break
        elif v < lr1 * vOld:
            # Feasibility improved → update Lagrange multiplier
            y_lagrange = y_lagrange - sigma * dev
            vOld = v
        else:
            # Feasibility not improved → increase penalty
            sigma = lr2 * sigma

    if verbose:
        print(f'elapsed time: {time.time() - t0:.0f} seconds')

    return Z, H
