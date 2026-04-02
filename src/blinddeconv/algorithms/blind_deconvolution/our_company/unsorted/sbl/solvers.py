"""
solvers.py

Core solver functions for Generalized Sparse Bayesian Learning (SBL).

Ported from MATLAB code by Jan Glaubitz (Jan 2022).
Reference:
    J. Glaubitz, A. Gelb, G. Song: "Generalized sparse Bayesian learning
    and application to image reconstruction",
    SIAM/ASA J. Uncertainty Quantification, 11(1):262-284, 2023.
    arXiv:2201.07061

Contains:
    ADMM_1d              — 1-D L1-regularised reconstruction via ADMM
    ADMM_2d              — 2-D L1-regularised reconstruction via ADMM
    BCD_1d               — 1-D SBL via Bayesian Coordinate Descent
    BCD_2d               — 2-D SBL via Bayesian Coordinate Descent
    BCD_1d_fusion        — 1-D SBL with two data sources (data fusion)
    SBL_evidence_1d      — 1-D SBL via evidence (type-II ML) approach
    IAS_1d               — 1-D MAP estimation via IAS algorithm

MATLAB -> Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    F.'*y  (transpose, not conjugate):
        For real matrices F, MATLAB F.' == F' (ctranspose).
        -> F.T @ y.

    (FtF + rho*RtR) \\ b  (backslash, dense):
        -> np.linalg.solve(A, b).

    sparse(alpha*FtF + R'*B*R):
        In MATLAB the result is sparse only if all operands are sparse.
        Here FtF = F'*F is DENSE (F is a dense Gaussian kernel matrix).
        Therefore C_inv is actually a dense matrix despite the sparse()
        wrapper.  We use dense np.linalg.solve.

    norm(X-X_OLD)^2  for matrix:
        MATLAB norm(M) for a matrix returns the 2-norm (largest singular
        value), NOT the Frobenius norm.  However, the MATLAB code uses
        norm(vec(M-M_OLD)):  when the argument is a VECTOR, norm returns
        the 2-norm of the vector = Frobenius norm of the original matrix.
        In BCD_2d the code explicitly does norm(vec(Mu-Mu_OLD)).
        In ADMM_2d, norm(X-X_OLD) is called on a MATRIX, which gives the
        spectral norm.  But we follow the code literally:
            - BCD_2d: uses vec() -> vector norm -> np.linalg.norm(vec)
            - ADMM_2d: norm(X-X_OLD) on matrix -> np.linalg.norm(X-X_OLD, 2)
              But singular value decomposition is expensive for every iter.
              Since the MATLAB code is measuring convergence only, and the
              spectral norm <= Frobenius norm, we use Frobenius for speed
              (matches the intent and is a safe upper bound).

    R'*B*R  where R is sparse, B = diag(beta):
        -> R.T @ diag(beta) @ R  or  R.T @ sp.diags(beta, 0) @ R.
        Since FtF is dense, the whole C_inv ends up dense anyway.

    D (sparse regularization) * Mu (dense matrix) in BCD_2d:
        D is sparse (k×n), Mu is dense (n×n).
        -> D @ Mu works in scipy (sparse @ dense = dense).
        -> D.T for transpose of sparse CSR.

    ADMM_2d vec/reshape:
        MATLAB reshape(r, n, n) refills column-major.
        -> unvec(r, (n, n)).

    dot(r, Gr) in ADMM_2d / BCD_2d:
        Both r and Gr are 1-D vectors (after vec()).
        MATLAB dot gives inner product. -> np.dot(r, Gr).
"""

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .utils import vec, unvec, shrinkage


# ═════════════════════════════════════════════════════════════════════════════
# ADMM_1d  (from ADMM_1d.m)
# ═════════════════════════════════════════════════════════════════════════════

def ADMM_1d(
    F: np.ndarray,
    y: np.ndarray,
    R,
    lam: float,
    rho: float,
    alpha: float,
    quiet: bool = True,
):
    """
    1-D signal reconstruction with L1-regularisation via ADMM.

    Solves:  min_x  0.5*||F*x - y||^2 + lambda*||R*x||_1

    Parameters
    ----------
    F     : (m, n) ndarray — forward operator (dense).
    y     : (m,) ndarray — noisy indirect measurements.
    R     : (k, n) sparse or dense — regularisation matrix.
    lam   : float — regularisation weight lambda.
    rho   : float — ADMM penalty parameter.
    alpha : float — ADMM relaxation parameter (typically 1.0).
    quiet : bool — suppress iteration log (default True).

    Returns
    -------
    x       : (n,) ndarray — reconstructed signal.
    history : dict with keys 'abs_error', 'rel_error' (lists).
    """
    MAX_ITER = 1000
    ABSTOL = 1e-8
    RELTOL = 1e-4

    n = F.shape[1]
    k = R.shape[0]

    # Ensure R is a dense ndarray for matmul (F is dense → whole system dense)
    if sp.issparse(R):
        R_dense = R.toarray()
    else:
        R_dense = np.asarray(R, dtype=np.float64)

    # Initialise variables
    x = np.zeros(n)
    x_OLD = np.zeros(n)
    z = np.zeros(k)
    u = np.zeros(k)

    # Pre-compute matrices
    RtR = R_dense.T @ R_dense            # (n, n)
    FtF = F.T @ F                        # (n, n)
    Fty = F.T @ y                        # (n,)

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # x-update:  x = (FtF + rho*RtR) \ (F'*y + rho*R'*(z-u))
        A = FtF + rho * RtR
        b = Fty + rho * R_dense.T @ (z - u)
        x = np.linalg.solve(A, b)

        # z-update with relaxation
        zold = z.copy()
        Rx = R_dense @ x
        Fx_hat = alpha * Rx + (1.0 - alpha) * zold
        z = shrinkage(Fx_hat + u, lam / rho)

        # u-update (dual variable)
        u = u + Fx_hat - z

        # Convergence tracking
        abs_err = np.linalg.norm(x - x_OLD) ** 2
        x_norm = np.linalg.norm(x)
        rel_err = (np.linalg.norm(x - x_OLD) / x_norm) ** 2 if x_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        x_OLD = x.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL:
            break

    return x, history


# ═════════════════════════════════════════════════════════════════════════════
# ADMM_2d  (from ADMM_2d.m)
# ═════════════════════════════════════════════════════════════════════════════

def ADMM_2d(
    F_1d: np.ndarray,
    Y: np.ndarray,
    D,
    lam: float,
    rho: float,
    alpha: float,
    quiet: bool = True,
):
    """
    2-D image reconstruction with L1-regularisation via ADMM.

    The forward model is separable:  Y_obs = F_1d * X * F_1d' + noise.
    Regularisation in both directions via D.

    Parameters
    ----------
    F_1d  : (m, n) ndarray — 1-D forward operator (dense).
    Y     : (m, m) ndarray — matrix of indirect measurements.
    D     : (k, n) sparse or dense — 1-D regularisation matrix.
    lam   : float — regularisation weight.
    rho   : float — ADMM penalty parameter.
    alpha : float — ADMM relaxation parameter.
    quiet : bool — suppress log (default True).

    Returns
    -------
    X       : (n, n) ndarray — reconstructed image.
    history : dict with keys 'abs_error', 'rel_error'.
    """
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2
    GRAD_DESC_STEPS = 5

    n = F_1d.shape[1]
    k = D.shape[0]

    # Convert D to dense for efficient matrix products
    if sp.issparse(D):
        D_dense = D.toarray()
    else:
        D_dense = np.asarray(D, dtype=np.float64)

    Dt = D_dense.T  # (n, k)
    Ft = F_1d.T     # (n, m)

    # --- Function handles matching MATLAB ---
    # fun_FTF(X) = vec( F' * (F * X * F') * F )
    def fun_FTF(X):
        return vec(Ft @ (F_1d @ X @ Ft) @ F_1d)

    # fun_RTR(X, rho) = rho * ( vec(D' * (D*X)) + vec((X * D') * D) )
    def fun_RTR(X, rho_):
        return rho_ * (vec(Dt @ (D_dense @ X)) + vec((X @ Dt) @ D_dense))

    # fun_G(X, rho) = fun_FTF(X) + fun_RTR(X, rho)
    def fun_G(X, rho_):
        return fun_FTF(X) + fun_RTR(X, rho_)

    # fun_b(Y, rho, V1, V2) = vec(F'*Y*F) + rho*vec(V1*D + D'*V2)
    #   where V1 = Z1 - U1 (n, k) and V2 = Z2 - U2 (k, n)
    #   MATLAB: V1*D means (n,k)@(k,n) = not right; looking at dimensions:
    #     D is (k,n).  The MATLAB expression is: rho*vec( V1*D + D'*V2 )
    #     but V1 is (n,k) and D is (k,n) -> V1*D is (n,n).  OK.
    #     D' is (n,k) and V2 is (k,n) -> D'*V2 is (n,n).  OK.
    def fun_b(Y_, rho_, V1, V2):
        return vec(Ft @ Y_ @ F_1d) + rho_ * vec(V1 @ D_dense + Dt @ V2)

    # Initialise
    X = np.zeros((n, n))
    X_OLD = np.zeros((n, n))
    Z1 = np.zeros((n, k))
    Z2 = np.zeros((k, n))
    U1 = np.zeros((n, k))
    U2 = np.zeros((k, n))

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # x-update via conjugate-gradient-like gradient descent
        r = fun_b(Y, rho, Z1 - U1, Z2 - U2) - fun_G(X, rho)
        for _ in range(GRAD_DESC_STEPS):
            R_mat = unvec(r, (n, n))
            Gr = fun_G(R_mat, rho)
            r_dot = np.dot(r, r)
            denom = np.dot(r, Gr)
            if denom == 0:
                break
            gamma_step = r_dot / denom
            X = X + gamma_step * R_mat
            r = r - gamma_step * Gr

        # z-update with relaxation
        # aux1 = alpha*vec(X * D') + (1-alpha)*vec(Z1)
        aux1 = alpha * vec(X @ Dt) + (1.0 - alpha) * vec(Z1)
        z1 = shrinkage(aux1 + vec(U1), lam / rho)
        Z1 = unvec(z1, (n, k))

        # aux2 = alpha*vec(D*X) + (1-alpha)*vec(Z2)
        aux2 = alpha * vec(D_dense @ X) + (1.0 - alpha) * vec(Z2)
        z2 = shrinkage(aux2 + vec(U2), lam / rho)
        Z2 = unvec(z2, (k, n))

        # u-update
        U1 = U1 + unvec(aux1, (n, k)) - Z1
        U2 = U2 + unvec(aux2, (k, n)) - Z2

        # Convergence tracking
        diff = X - X_OLD
        abs_err = np.linalg.norm(diff, 'fro') ** 2
        x_norm = np.linalg.norm(X, 'fro')
        rel_err = (np.linalg.norm(diff, 'fro') / x_norm) ** 2 if x_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        X_OLD = X.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL:
            break

    return X, history


# ═════════════════════════════════════════════════════════════════════════════
# BCD_1d  (from BCD_1d.m)
# ═════════════════════════════════════════════════════════════════════════════

def BCD_1d(
    F: np.ndarray,
    y: np.ndarray,
    R,
    c: float,
    d: float,
    quiet: bool = True,
):
    """
    1-D SBL via Bayesian Coordinate Descent (BCD).

    Iterates:
        1. C_inv = alpha*F'F + R'*diag(beta)*R;  mu = C_inv \\ (alpha*F'y)
        2. alpha = (m + 2c) / (||F*mu - y||^2 + 2d)
        3. beta_j = (1 + 2c) / ((R*mu)_j^2 + 2d)

    Parameters
    ----------
    F : (m, n) ndarray — forward operator.
    y : (m,) ndarray — measurements.
    R : (k, n) sparse or dense — regularisation matrix.
    c, d : float — hyper-hyper-parameters of the Gamma prior.
    quiet : bool — suppress log.

    Returns
    -------
    mu    : (n,) ndarray — posterior mean.
    C_inv : (n, n) ndarray — inverse posterior covariance.
    alpha : float — estimated inverse noise variance.
    beta  : (k,) ndarray — estimated inverse prior variances.
    history : dict.
    """
    MIN_ITER = 10
    MAX_ITER = 1000
    ABSTOL = 1e-8
    RELTOL = 1e-4

    m, n = F.shape
    k = R.shape[0]

    if sp.issparse(R):
        R_dense = R.toarray()
    else:
        R_dense = np.asarray(R, dtype=np.float64)

    FtF = F.T @ F    # (n, n) dense
    Fty = F.T @ y    # (n,)

    # Initial values
    alpha_val = 1.0
    beta = np.ones(k)
    mu_OLD = np.zeros(n)

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # 1) Update x:  C_inv = alpha*FtF + R'*diag(beta)*R
        #    MATLAB: B = sparse(diag(beta)); C_inv = sparse(alpha*FtF + R'*B*R)
        #    Since FtF is dense, C_inv is dense.
        RtBR = R_dense.T @ (beta[:, np.newaxis] * R_dense)  # R' * diag(beta) * R
        C_inv = alpha_val * FtF + RtBR
        mu = np.linalg.solve(C_inv, alpha_val * Fty)

        # 2) Update alpha
        residual = F @ mu - y
        alpha_val = (m + 2.0 * c) / (np.dot(residual, residual) + 2.0 * d)

        # 3) Update beta
        Rmu = R_dense @ mu
        beta = (1.0 + 2.0 * c) / (Rmu ** 2 + 2.0 * d)

        # Convergence tracking
        abs_err = np.linalg.norm(mu - mu_OLD) ** 2
        mu_OLD_norm = np.linalg.norm(mu_OLD)
        rel_err = (np.linalg.norm(mu - mu_OLD) / mu_OLD_norm) ** 2 if mu_OLD_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        mu_OLD = mu.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL and counter >= MIN_ITER:
            break

    return mu, C_inv, alpha_val, beta, history


# ═════════════════════════════════════════════════════════════════════════════
# BCD_2d  (from BCD_2d.m)
# ═════════════════════════════════════════════════════════════════════════════

def BCD_2d(
    F_1d: np.ndarray,
    Y: np.ndarray,
    D,
    c: float,
    d: float,
    quiet: bool = False,
):
    """
    2-D SBL via Bayesian Coordinate Descent.

    Iterates (Kronecker-structured):
        1. Gradient descent on X  (5 steps per outer iteration).
        2. alpha = (m^2 + 2c) / (||F*Mu*F' - Y||_F^2 + 2d)
        3. B1 = (1+2c) / ((D*Mu).^2 + 2d),  B2 = (1+2c) / ((Mu*D').^2 + 2d)

    Parameters
    ----------
    F_1d : (m, n) ndarray — 1-D forward operator.
    Y    : (m, m) ndarray — measurement matrix.
    D    : (k, n) sparse or dense — 1-D regularisation matrix.
    c, d : float — hyper-hyper-parameters.
    quiet : bool — suppress log (default False, matching MATLAB QUIET=0).

    Returns
    -------
    Mu      : (n, n) ndarray — posterior mean image.
    alpha   : float — estimated inverse noise variance.
    B1      : (k, n) ndarray — row-direction inverse prior variances.
    B2      : (n, k) ndarray — column-direction inverse prior variances.
    history : dict.
    """
    MIN_ITER = 10
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2
    GRAD_DESC_STEPS = 5

    m = F_1d.shape[0]
    n = F_1d.shape[1]
    k = D.shape[0]

    if sp.issparse(D):
        D_dense = D.toarray()
    else:
        D_dense = np.asarray(D, dtype=np.float64)

    Dt = D_dense.T   # (n, k)
    Ft = F_1d.T      # (n, m)

    # --- Function handles matching MATLAB ---
    # fun_FTAF(X, alpha) = alpha * vec( F' * (F * X * F') * F )
    def fun_FTAF(X, a):
        return a * vec(Ft @ (F_1d @ X @ Ft) @ F_1d)

    # fun_RTBR(X, B1, B2) = vec( D' * (B1 .* (D*X)) ) + vec( (B2 .* (X*D')) * D )
    #   B1 is (k, n), D*X is (k, n) -> element-wise multiply -> (k, n)
    #   B2 is (n, k), X*D' is (n, k) -> element-wise multiply -> (n, k)
    def fun_RTBR(X, B1_, B2_):
        DX = D_dense @ X          # (k, n)
        XDt = X @ Dt              # (n, k)
        return vec(Dt @ (B1_ * DX)) + vec((B2_ * XDt) @ D_dense)

    # fun_G(X, alpha, B1, B2)
    def fun_G(X, a, B1_, B2_):
        return fun_FTAF(X, a) + fun_RTBR(X, B1_, B2_)

    # fun_b(Y, alpha) = alpha * vec( F' * Y * F )
    def fun_b(Y_, a):
        return a * vec(Ft @ Y_ @ F_1d)

    # Initial values
    alpha_val = 1.0
    B1 = np.ones((k, n))
    B2 = np.ones((n, k))
    Mu = np.zeros((n, n))
    Mu_OLD = Mu.copy()

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # 1) Fix alpha, B1, B2 and update X via gradient descent
        r = fun_b(Y, alpha_val) - fun_G(Mu, alpha_val, B1, B2)
        for _ in range(GRAD_DESC_STEPS):
            R_mat = unvec(r, (n, n))
            Gr = fun_G(R_mat, alpha_val, B1, B2)
            r_dot = np.dot(r, r)
            denom = np.dot(r, Gr)
            if denom == 0:
                break
            gamma_step = r_dot / denom
            Mu = Mu + gamma_step * R_mat
            r = r - gamma_step * Gr

        # 2) Update alpha
        # alpha = (m^2 + 2c) / ( ||vec(F*Mu*F' - Y)||^2 + 2d )
        residual = F_1d @ Mu @ Ft - Y
        res_vec = vec(residual)
        alpha_val = (m ** 2 + 2.0 * c) / (np.dot(res_vec, res_vec) + 2.0 * d)

        # 3) Update B1, B2
        DM = D_dense @ Mu         # (k, n)
        MDt = Mu @ Dt             # (n, k)
        B1 = (1.0 + 2.0 * c) / (DM ** 2 + 2.0 * d)
        B2 = (1.0 + 2.0 * c) / (MDt ** 2 + 2.0 * d)

        # Convergence tracking
        diff_vec = vec(Mu - Mu_OLD)
        abs_err = np.dot(diff_vec, diff_vec)
        old_vec = vec(Mu_OLD)
        old_norm = np.linalg.norm(old_vec)
        rel_err = (np.linalg.norm(diff_vec) / old_norm) ** 2 if old_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        Mu_OLD = Mu.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL and counter >= MIN_ITER:
            break

    return Mu, alpha_val, B1, B2, history


# ═════════════════════════════════════════════════════════════════════════════
# BCD_1d_fusion  (from BCD_1d_fusion.m)
# ═════════════════════════════════════════════════════════════════════════════

def BCD_1d_fusion(
    F1: np.ndarray,
    F2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    R,
    c: float,
    d: float,
    quiet: bool = True,
):
    """
    1-D SBL with two data sources via BCD (data fusion).

    Each sensor has its own forward operator and noise level:
        y1 = F1*x + e1,   y2 = F2*x + e2

    Parameters
    ----------
    F1, F2 : ndarray — forward operators for the two sensors.
    y1, y2 : ndarray — measurements from the two sensors.
    R      : (k, n) — regularisation matrix.
    c, d   : float — hyper-hyper-parameters.
    quiet  : bool — suppress log.

    Returns
    -------
    mu      : (n,) ndarray — posterior mean.
    C_inv   : (n, n) ndarray — inverse posterior covariance.
    alpha   : (2,) ndarray — [alpha1, alpha2] inverse noise variances.
    beta    : (k,) ndarray — inverse prior variances.
    history : dict.
    """
    MIN_ITER = 10
    MAX_ITER = 1000
    ABSTOL = 1e-8
    RELTOL = 1e-4

    m1 = F1.shape[0]
    m2 = F2.shape[0]
    n = F1.shape[1]
    k = R.shape[0]

    if sp.issparse(R):
        R_dense = R.toarray()
    else:
        R_dense = np.asarray(R, dtype=np.float64)

    F1tF1 = F1.T @ F1
    F2tF2 = F2.T @ F2
    F1ty = F1.T @ y1
    F2ty = F2.T @ y2

    alpha1 = 1.0
    alpha2 = 1.0
    beta = np.ones(k)
    mu_OLD = np.zeros(n)

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # 1) Update x
        RtBR = R_dense.T @ (beta[:, np.newaxis] * R_dense)
        C_inv = alpha1 * F1tF1 + alpha2 * F2tF2 + RtBR
        mu = np.linalg.solve(C_inv, alpha1 * F1ty + alpha2 * F2ty)

        # 2) Update alpha1, alpha2
        res1 = F1 @ mu - y1
        res2 = F2 @ mu - y2
        alpha1 = (m1 + 2.0 * c) / (np.dot(res1, res1) + 2.0 * d)
        alpha2 = (m2 + 2.0 * c) / (np.dot(res2, res2) + 2.0 * d)

        # 3) Update beta
        Rmu = R_dense @ mu
        beta = (1.0 + 2.0 * c) / (Rmu ** 2 + 2.0 * d)

        # Convergence
        abs_err = np.linalg.norm(mu - mu_OLD) ** 2
        mu_OLD_norm = np.linalg.norm(mu_OLD)
        rel_err = (np.linalg.norm(mu - mu_OLD) / mu_OLD_norm) ** 2 if mu_OLD_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        mu_OLD = mu.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL:
            break

    alpha = np.array([alpha1, alpha2])
    return mu, C_inv, alpha, beta, history


# ═════════════════════════════════════════════════════════════════════════════
# SBL_evidence_1d  (from SBL_evidence_1d.m)
# ═════════════════════════════════════════════════════════════════════════════

def SBL_evidence_1d(
    F: np.ndarray,
    y: np.ndarray,
    c: float,
    d: float,
    quiet: bool = True,
):
    """
    1-D SBL via evidence (type-II maximum likelihood) approach.

    Assumes sparsity in the canonical basis (R = I).

    Iterates:
        1. C_inv = alpha*F'F + diag(beta);  mu = C_inv \\ (alpha*F'y)
        2. C = inv(C_inv);  gamma_i = 1 - beta_i * C_ii
        3. alpha = (n - sum(gamma) + 2c) / (||F*mu - y||^2 + 2d)
        4. beta_i = (gamma_i + 2c) / (mu_i^2 + 2d)

    Parameters
    ----------
    F : (m, n) ndarray.
    y : (m,) ndarray.
    c, d : float — hyper-hyper-parameters.
    quiet : bool.

    Returns
    -------
    mu, C_inv, alpha, beta, history.
    """
    MIN_ITER = 10
    MAX_ITER = 1000
    ABSTOL = 1e-8
    RELTOL = 1e-4

    m, n = F.shape
    FtF = F.T @ F
    Fty = F.T @ y

    alpha_val = 1.0
    beta = np.ones(n)
    mu_OLD = np.zeros(n)

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # 1) Update x
        C_inv = alpha_val * FtF + np.diag(beta)
        mu = np.linalg.solve(C_inv, alpha_val * Fty)

        # 2) Compute C and gamma
        # MATLAB: C = inv(C_inv)
        C = np.linalg.inv(C_inv)
        gamma = 1.0 - beta * np.diag(C)

        # 3) Update alpha
        residual = F @ mu - y
        alpha_val = (n - np.sum(gamma) + 2.0 * c) / (np.dot(residual, residual) + 2.0 * d)

        # 4) Update beta
        beta = (gamma + 2.0 * c) / (mu ** 2 + 2.0 * d)

        # Convergence
        abs_err = np.linalg.norm(mu - mu_OLD) ** 2
        mu_OLD_norm = np.linalg.norm(mu_OLD)
        rel_err = (np.linalg.norm(mu - mu_OLD) / mu_OLD_norm) ** 2 if mu_OLD_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        mu_OLD = mu.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL and counter >= MIN_ITER:
            break

    return mu, C_inv, alpha_val, beta, history


# ═════════════════════════════════════════════════════════════════════════════
# IAS_1d  (from IAS_1d.m)
# ═════════════════════════════════════════════════════════════════════════════

def IAS_1d(
    F: np.ndarray,
    y: np.ndarray,
    variance: float,
    c: float,
    d: float,
    quiet: bool = True,
):
    """
    1-D MAP estimation via Iterative Alternating Sequential (IAS).

    Assumes sparsity in the canonical basis (R = I).  Alpha is fixed
    from the known noise variance.

    Iterates:
        1. C_inv = alpha*F'F + diag(1/beta);  x = C_inv \\ (alpha*F'y)
        2. eta = c - 3/2;  beta_j = d * (eta/2 + sqrt(eta^2/4 + x_j^2/(2d)))

    Parameters
    ----------
    F        : (m, n) ndarray.
    y        : (m,) ndarray.
    variance : float — known noise variance.
    c, d     : float — hyper-hyper-parameters.
    quiet    : bool.

    Returns
    -------
    x       : (n,) ndarray — MAP estimate.
    beta    : (n,) ndarray — estimated prior covariance.
    history : dict.
    """
    MIN_ITER = 10
    MAX_ITER = 1000
    ABSTOL = 1e-8
    RELTOL = 1e-4

    m, n = F.shape
    FtF = F.T @ F
    Fty = F.T @ y

    alpha_val = 1.0 / variance
    beta = np.ones(n)
    x_OLD = np.zeros(n)

    history = {'abs_error': [], 'rel_error': []}

    if not quiet:
        print(f"{'iter':>5s}  {'abs err':>12s}  {'abs tol':>12s}  "
              f"{'rel err':>12s}  {'rel tol':>12s}")

    for counter in range(MAX_ITER):
        # 1) Fix beta and update x
        # MATLAB: D = sparse(diag(1./beta)); C_inv = alpha*FtF + D
        D_inv = np.diag(1.0 / beta)
        C_inv = alpha_val * FtF + D_inv
        x = np.linalg.solve(C_inv, alpha_val * Fty)

        # 2) Fix x and update beta
        eta = c - 1.5  # c - 3/2
        beta = d * (eta / 2.0 + np.sqrt(eta ** 2 / 4.0 + x ** 2 / (2.0 * d)))

        # Convergence
        abs_err = np.linalg.norm(x - x_OLD) ** 2
        x_OLD_norm = np.linalg.norm(x_OLD)
        rel_err = (np.linalg.norm(x - x_OLD) / x_OLD_norm) ** 2 if x_OLD_norm > 0 else 0.0
        history['abs_error'].append(abs_err)
        history['rel_error'].append(rel_err)
        x_OLD = x.copy()

        if not quiet:
            print(f"{counter + 1:5d}  {abs_err:12.2e}  {ABSTOL:12.2e}  "
                  f"{rel_err:12.2e}  {RELTOL:12.2e}")

        if abs_err < ABSTOL and rel_err < RELTOL and counter >= MIN_ITER:
            break

    return x, beta, history
