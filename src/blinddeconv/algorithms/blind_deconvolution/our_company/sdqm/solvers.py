"""
Blind deconvolution problem and Riemannian solvers.

Ported from ROPTLIB:
  - Problems/CFR2BlindDecon2D/CFR2BlindDecon2D.cpp  (objective, gradient)
  - Solvers/LRBFGS.cpp + SolversLS.cpp              (L-RBFGS with Armijo)
  - Solvers/RSD.cpp                                  (steepest descent)

The problem is:
    min_{(G,H) in M}  ||y - diag(F B G (conj(F) C H)^*)||^2 + penalty

where M = C_*^{K x r} x C_*^{N x r} / C_*  is the quotient manifold,
F is the 2-D DFT, B and C are subspace operators, and r = 1 normally.

All arrays use complex128.  G has shape (K, r), H has shape (N, r).
"""

import numpy as np
from numpy.linalg import norm, cholesky, solve
from . import manifold as mf


# =====================================================================
# Problem: objective + Euclidean gradient
# =====================================================================

class CFR2BlindDeconProblem:
    """Blind deconvolution objective on the CFR2 quotient manifold.

    Parameters
    ----------
    y : complex (L,)      — observation vector (FFT of blurred image)
    B_op, BH_op           — forward / adjoint kernel-subspace operators
    C_op, CH_op           — forward / adjoint image-subspace operators
    K, N, L, r            — dimensions
    rho, d, mu            — penalty (incoherence) parameters
    image_shape           — (H, W) for 2-D FFT
    """

    def __init__(self, y, B_op, BH_op, C_op, CH_op,
                 K, N, L, r, rho, d, mu, image_shape):
        self.y = y.astype(np.complex128).ravel()
        self.B_op = B_op
        self.BH_op = BH_op
        self.C_op = C_op
        self.CH_op = CH_op
        self.K = K
        self.N = N
        self.L = L
        self.r = r
        self.rho = rho
        self.d = d
        self.mu = mu
        self.image_shape = image_shape
        self._cache = {}

    # -----------------------------------------------------------------
    def _fft2_vec(self, v):
        """2-D FFT on a vector of length L = H*W."""
        H, W = self.image_shape
        return np.fft.fft2(v.reshape(H, W)).ravel()

    def _ifft2_vec(self, v):
        """2-D IFFT on a vector of length L = H*W."""
        H, W = self.image_shape
        # Умножаем на L, чтобы поведение совпадало с FFTW_BACKWARD из C++
        return (np.fft.ifft2(v.reshape(H, W)) * self.L).ravel()

    # -----------------------------------------------------------------
    def f(self, G, H):
        """Evaluate objective (and cache intermediates for gradient)."""
        K, N, L, r = self.K, self.N, self.L, self.r

        # BU = F( B @ G_col )   for each column of G
        # CV = Finv( C @ H_col )
        BU_cols = []
        CV_cols = []
        for j in range(r):
            bu = self.B_op(G[:, j])        # length L
            bu = self._fft2_vec(bu)         # FFT(B g_j)
            BU_cols.append(bu)

            cv = self.C_op(H[:, j])        # length L
            cv = self._ifft2_vec(cv)        # IFFT(C h_j)  — matches FFTW_BACKWARD
            CV_cols.append(cv)

        # BU, CV: (L, r)
        BU = np.column_stack(BU_cols)
        CV = np.column_stack(CV_cols)

        # residual_l = y_l - sum_j BU[l,j] * conj(CV[l,j])
        diag_BXCH = np.sum(BU * np.conj(CV), axis=1)   # length L
        residual = self.y - diag_BXCH                    # length L

        # Penalty
        penalty = 0.0
        rownorm2BU = None
        normV2 = None
        if self.rho != 0:
            # tmp_coeff = self.L / (8.0 * self.d**2 * self.mu**2)
            tmp_coeff = 1.0 / (8.0 * self.d**2 * self.mu**2)
            rownorm2BU = np.sum(np.abs(BU)**2, axis=1)   # (L,)
            normV2 = np.sum(np.abs(H)**2)
            violations = np.maximum(tmp_coeff * rownorm2BU * normV2 - 1.0, 0.0)
            penalty = self.rho * np.sum(violations**2)

        obj = np.sum(np.abs(residual)**2) + penalty

        # Cache for gradient
        self._cache = {
            'BU': BU, 'CV': CV,
            'residual': residual,
            'rownorm2BU': rownorm2BU,
            'normV2': normV2,
        }
        return obj

    # -----------------------------------------------------------------
    def euc_grad(self, G, H):
        """Euclidean gradient  (gfG, gfH) — raw, before quotient projection.

        Direct port of CFR2BlindDecon2D::EucGrad (adapted to use operators
        instead of explicit matrices).

        The Euclidean gradient is then converted to the Riemannian gradient
        by applying (V^*V)^{-1} and (U^*U)^{-1} scaling (matching the
        quotient metric), and then projecting onto the horizontal space.

        Returns (gfG, gfH) with gfG: (K, r), gfH: (N, r).
        """
        K, N, L, r = self.K, self.N, self.L, self.r
        BU = self._cache['BU']
        CV = self._cache['CV']
        residual = self._cache['residual']   # y - diag(BU CV^*)

        # --- Gradient w.r.t. G  (called EGFV in C++) ---
        # tmpp = diag(residual) @ CV  then IFFT then BH
        EGFG = np.zeros((K, r), dtype=np.complex128)
        for j in range(r):
            tmpp = residual * CV[:, j]                # element-wise
            tmpp = self._ifft2_vec(tmpp)              # IFFT (FFTW_BACKWARD)
            EGFG[:, j] = self.BH_op(tmpp)

        EGFG *= -2.0

        # --- Gradient w.r.t. H  (called EGFTU in C++) ---
        # tmpp = diag(conj(residual)) @ BU  then FFT then CH
        EGFH = np.zeros((N, r), dtype=np.complex128)
        for j in range(r):
            tmpp = np.conj(residual) * BU[:, j]       # element-wise
            tmpp = self._fft2_vec(tmpp)                # FFT (FFTW_FORWARD)
            EGFH[:, j] = self.CH_op(tmpp)

        EGFH *= -2.0

        # --- Penalty gradient ---
        if self.rho != 0:
            rownorm2BU = self._cache['rownorm2BU']
            normV2 = self._cache['normV2']
            # tmp_coeff = self.L / (8.0 * self.d**2 * self.mu**2)
            tmp_coeff = 1.0 / (8.0 * self.d**2 * self.mu**2)

            violations = np.maximum(tmp_coeff * rownorm2BU * normV2 - 1.0, 0.0)

            # Gradient w.r.t. G penalty
            for j in range(r):
                coefs = 4.0 * self.rho * tmp_coeff * violations * normV2
                DFBU = coefs * BU[:, j]                # (L,)
                DFBU = self._ifft2_vec(DFBU)
                EGFG[:, j] += self.BH_op(DFBU)

            # Gradient w.r.t. H penalty
            coef_h = np.sum(2.0 * tmp_coeff * self.rho * rownorm2BU * 2.0 * violations)
            EGFH += coef_h * H

        # --- Apply quotient metric scaling ---
        # gfG = EGFG @ (H^*H)^{-1},  gfH = EGFH @ (G^*G)^{-1}
        HH = H.conj().T @ H   # r x r
        GG = G.conj().T @ G   # r x r

        try:
            invHH = np.linalg.inv(HH)
        except np.linalg.LinAlgError:
            invHH = np.linalg.inv(HH + 1e-12 * np.eye(r))

        try:
            invGG = np.linalg.inv(GG)
        except np.linalg.LinAlgError:
            invGG = np.linalg.inv(GG + 1e-12 * np.eye(r))

        gfG = EGFG @ invHH
        gfH = EGFH @ invGG

        return gfG, gfH


# =====================================================================
# Riemannian Steepest Descent (RSD)  with Armijo line search
# =====================================================================

def solve_rsd(prob: CFR2BlindDeconProblem, G0, H0,
              max_iter=300, tol=1e-6, verbose=False):
    """Riemannian Steepest Descent on the CFR2 quotient manifold.

    Matches the combination of  RSD.cpp + SolversLS.cpp (Armijo line search).

    Returns G_opt, H_opt, history dict.
    """
    G, H = G0.copy(), H0.copy()
    r = G.shape[1]

    history = {'f': [], 'grad_norm': []}

    f_val = prob.f(G, H)
    egfG, egfH = prob.euc_grad(G, H)
    gfG, gfH = mf.euc_grad_to_grad(G, H, egfG, egfH)
    ngf0 = np.sqrt(mf.metric(G, H, gfG, gfH, gfG, gfH))
    ngf = ngf0

    stepsize = 1.0

    for it in range(max_iter):
        history['f'].append(f_val)
        history['grad_norm'].append(ngf)

        if verbose and it % 50 == 0:
            print(f"  RSD iter {it}: f={f_val:.6e}, |gf|={ngf:.6e}")

        if ngf / (ngf0 + 1e-15) < tol and it > 0:
            break

        # Search direction = -gradient
        etaG, etaH = -gfG, -gfH

        # Armijo line search
        initial_slope = mf.metric(G, H, gfG, gfH, etaG, etaH)  # should be negative
        stepsize = _armijo_line_search(
            prob, G, H, etaG, etaH, f_val, initial_slope, stepsize
        )

        # Retraction
        G_new, H_new = mf.retraction(G, H, etaG, etaH, stepsize)

        # Evaluate at new point
        f_new = prob.f(G_new, H_new)
        egfG_new, egfH_new = prob.euc_grad(G_new, H_new)
        gfG_new, gfH_new = mf.euc_grad_to_grad(G_new, H_new, egfG_new, egfH_new)
        ngf_new = np.sqrt(mf.metric(G_new, H_new, gfG_new, gfH_new, gfG_new, gfH_new))

        # BB step size for next iteration
        sG = G_new - G
        sH = H_new - H
        yG = gfG_new - gfG
        yH = gfH_new - gfH
        inpss = mf.metric(G_new, H_new, sG, sH, sG, sH)
        inpsy = mf.metric(G_new, H_new, sG, sH, yG, yH)
        if abs(inpsy) > 1e-30:
            stepsize = abs(inpss / inpsy)
        else:
            stepsize = 1.0

        G, H = G_new, H_new
        f_val = f_new
        gfG, gfH = gfG_new, gfH_new
        ngf = ngf_new

    history['f'].append(f_val)
    history['grad_norm'].append(ngf)
    return G, H, history


# =====================================================================
# L-RBFGS  solver
# =====================================================================

def solve_lrbfgs(prob: CFR2BlindDeconProblem, G0, H0,
                 max_iter=300, tol=1e-6, memory=4, verbose=False):
    """Limited-memory Riemannian BFGS on the CFR2 quotient manifold.

    Matches LRBFGS.cpp + SolversLS.cpp.

    Returns G_opt, H_opt, history dict.
    """
    G, H = G0.copy(), H0.copy()
    r = G.shape[1]
    K, N = prob.K, prob.N

    history = {'f': [], 'grad_norm': []}

    f_val = prob.f(G, H)
    egfG, egfH = prob.euc_grad(G, H)
    gfG, gfH = mf.euc_grad_to_grad(G, H, egfG, egfH)
    ngf0 = np.sqrt(mf.metric(G, H, gfG, gfH, gfG, gfH))
    ngf = ngf0

    # L-BFGS storage
    S_list = []   # list of (sG, sH) pairs
    Y_list = []   # list of (yG, yH) pairs
    RHO_list = []
    gamma = 1.0

    stepsize = 1.0

    for it in range(max_iter):
        history['f'].append(f_val)
        history['grad_norm'].append(ngf)

        if verbose and it % 50 == 0:
            print(f"  LRBFGS iter {it}: f={f_val:.6e}, |gf|={ngf:.6e}")

        if ngf / (ngf0 + 1e-15) < tol and it > 0:
            break

        # Two-loop recursion to get search direction
        etaG, etaH = _lbfgs_two_loop(
            G, H, gfG, gfH, S_list, Y_list, RHO_list, gamma
        )
        etaG, etaH = -etaG, -etaH

        # Armijo line search
        initial_slope = mf.metric(G, H, gfG, gfH, etaG, etaH)
        if initial_slope >= 0:
            # Not a descent direction — fallback to steepest descent
            etaG, etaH = -gfG, -gfH
            initial_slope = mf.metric(G, H, gfG, gfH, etaG, etaH)

        stepsize_init = _compute_initial_stepsize(
            it, f_val, initial_slope, stepsize
        )
        stepsize = _armijo_line_search(
            prob, G, H, etaG, etaH, f_val, initial_slope, stepsize_init
        )

        G_new, H_new = mf.retraction(G, H, etaG, etaH, stepsize)

        f_new = prob.f(G_new, H_new)
        egfG_new, egfH_new = prob.euc_grad(G_new, H_new)
        gfG_new, gfH_new = mf.euc_grad_to_grad(G_new, H_new, egfG_new, egfH_new)
        ngf_new = np.sqrt(mf.metric(G_new, H_new, gfG_new, gfH_new, gfG_new, gfH_new))

        # Update L-BFGS pairs (with vector transport)
        sG = stepsize * etaG
        sH = stepsize * etaH
        # Transport old gradient to new point
        gfG_transported, gfH_transported = mf.vector_transport(
            G, H, etaG, etaH, G_new, H_new, gfG, gfH
        )
        yG = gfG_new - gfG_transported
        yH = gfH_new - gfH_transported
        # Also transport s
        sG_t, sH_t = mf.vector_transport(
            G, H, etaG, etaH, G_new, H_new, sG, sH
        )

        inpsy = mf.metric(G_new, H_new, sG_t, sH_t, yG, yH)
        inpss = mf.metric(G_new, H_new, sG_t, sH_t, sG_t, sH_t)
        inpyy = mf.metric(G_new, H_new, yG, yH, yG, yH)

        # Update Hessian approximation
        if inpsy > 0:
            rho_k = 1.0 / inpsy
            # Transport all stored pairs to new tangent space
            new_S = []
            new_Y = []
            new_RHO = []
            for (sg_old, sh_old), (yg_old, yh_old), rho_old in zip(S_list, Y_list, RHO_list):
                sg_t, sh_t = mf.vector_transport(G, H, etaG, etaH, G_new, H_new, sg_old, sh_old)
                yg_t, yh_t = mf.vector_transport(G, H, etaG, etaH, G_new, H_new, yg_old, yh_old)
                new_S.append((sg_t, sh_t))
                new_Y.append((yg_t, yh_t))
                new_RHO.append(rho_old)

            new_S.append((sG_t, sH_t))
            new_Y.append((yG, yH))
            new_RHO.append(rho_k)

            if len(new_S) > memory:
                new_S = new_S[-memory:]
                new_Y = new_Y[-memory:]
                new_RHO = new_RHO[-memory:]

            S_list = new_S
            Y_list = new_Y
            RHO_list = new_RHO
            gamma = inpsy / inpyy if inpyy > 1e-30 else 1.0
        else:
            # Reject update, clear memory
            S_list = []
            Y_list = []
            RHO_list = []
            gamma = 1.0

        G, H = G_new, H_new
        f_val = f_new
        gfG, gfH = gfG_new, gfH_new
        ngf = ngf_new

    history['f'].append(f_val)
    history['grad_norm'].append(ngf)
    return G, H, history


# =====================================================================
# Internal helpers
# =====================================================================

def _lbfgs_two_loop(G, H, gfG, gfH, S_list, Y_list, RHO_list, gamma):
    """L-BFGS two-loop recursion on the manifold.

    Returns H_k^{-1} gf  (the approximate Newton direction, before negation).
    """
    m = len(S_list)
    if m == 0:
        return gamma * gfG, gamma * gfH

    alphas = [0.0] * m

    # q = gf
    qG, qH = gfG.copy(), gfH.copy()

    # First loop (backwards)
    for i in range(m - 1, -1, -1):
        sG_i, sH_i = S_list[i]
        yG_i, yH_i = Y_list[i]
        rho_i = RHO_list[i]
        alphas[i] = rho_i * mf.metric(G, H, sG_i, sH_i, qG, qH)
        qG = qG - alphas[i] * yG_i
        qH = qH - alphas[i] * yH_i

    # r = gamma * q  (scaling by initial Hessian approximation)
    rG = gamma * qG
    rH = gamma * qH

    # Second loop (forwards)
    for i in range(m):
        sG_i, sH_i = S_list[i]
        yG_i, yH_i = Y_list[i]
        rho_i = RHO_list[i]
        beta = rho_i * mf.metric(G, H, yG_i, yH_i, rG, rH)
        rG = rG + (alphas[i] - beta) * sG_i
        rH = rH + (alphas[i] - beta) * sH_i

    return rG, rH


def _compute_initial_stepsize(it, f_val, initial_slope, prev_stepsize):
    """Heuristic for initial step size (QuadIntMod from ROPTLIB)."""
    if it == 0:
        return 1.0
    # Quadratic interpolation
    s = -2.0 * f_val / initial_slope if initial_slope < 0 else prev_stepsize
    s = min(max(s, 1e-10), 1e10)
    return s


def _armijo_line_search(prob, G, H, etaG, etaH, f0, initial_slope,
                        stepsize_init, ls_ratio=0.5, ls_alpha=1e-4,
                        max_ls=20):
    """Armijo backtracking line search.

    Find stepsize s such that
        f(R(x, s*eta)) <= f(x) + ls_alpha * s * <gf, eta>
    """
    step = stepsize_init

    for _ in range(max_ls):
        G_try, H_try = mf.retraction(G, H, etaG, etaH, step)
        f_try = prob.f(G_try, H_try)

        if f_try <= f0 + ls_alpha * step * initial_slope:
            return step
        step *= ls_ratio

    return step
