"""
utils.py

Utility functions for Generalized Sparse Bayesian Learning (SBL).

Ported from MATLAB code by Jan Glaubitz (Jan 2022).
Reference:
    J. Glaubitz, A. Gelb, G. Song: "Generalized sparse Bayesian learning
    and application to image reconstruction",
    SIAM/ASA J. Uncertainty Quantification, 11(1):262-284, 2023.
    arXiv:2201.07061

MATLAB -> Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    linspace(0, 1, n):
        MATLAB returns a 1×n ROW vector.
        NumPy returns a 1-D array of shape (n,).
        When computing grid - grid', MATLAB uses implicit expansion
        (row minus column = n×n matrix).
        -> Use broadcasting: grid[np.newaxis, :] - grid[:, np.newaxis].

    spdiags(B, d, m, n):
        MATLAB: columns of B placed on diagonals d.  For superdiag d>=0,
        B(j,k) goes to S(j, j+d(k));  elements exceeding matrix bounds
        are silently ignored.
        -> scipy.sparse.diags(values, offsets, shape) with scalar values
        for constant diagonals.  Diagonal k of an (m,n) matrix has
        length min(m, n-k) for k>=0, so scipy auto-truncates.

    X(:) (vectorisation):
        MATLAB stacks columns (Fortran / column-major order).
        -> X.ravel(order='F') to match exactly.
        reshape(v, n, n) refills column-by-column.
        -> v.reshape((n, n), order='F').

    mvnrnd(mu, C, N):
        MATLAB: mu may be column vector; returns N×d.
        -> np.random.Generator.multivariate_normal(mu_1d, C, size=N).
        mu must be flattened to 1-D first.

    quantile(y, p):
        Both use linear interpolation; negligible difference for N>=1000.
        -> np.quantile(y, p, axis=0).

    sparse(diag(beta)):
        -> scipy.sparse.diags(beta, 0) for a diagonal sparse matrix.

    A \\ b (backslash):
        Dense solve: np.linalg.solve(A, b).
        Sparse solve: scipy.sparse.linalg.spsolve(A, b).
        The forward operator F from construct_F_deconvolution is dense
        (full Gaussian kernel), so C_inv = alpha*F'F + R'BR is generally
        dense -> use np.linalg.solve.

    norm(x)^2:
        -> np.dot(x, x) for 1-D, or np.linalg.norm(x)**2.
        For matrices: np.linalg.norm(X, 'fro')**2.
"""

import numpy as np
import scipy.sparse as sp


# ═════════════════════════════════════════════════════════════════════════════
# vec / unvectorise  (MATLAB X(:) and reshape)
# ═════════════════════════════════════════════════════════════════════════════

def vec(X: np.ndarray) -> np.ndarray:
    """
    Vectorise a matrix in column-major (Fortran) order, matching MATLAB X(:).

    Parameters
    ----------
    X : (M, N) ndarray

    Returns
    -------
    v : (M*N,) ndarray
    """
    return X.ravel(order='F')


def unvec(v: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Reshape a vector back to a matrix in column-major order,
    matching MATLAB reshape(v, m, n).

    Parameters
    ----------
    v     : (m*n,) ndarray
    shape : (m, n)

    Returns
    -------
    X : (m, n) ndarray
    """
    return v.reshape(shape, order='F')


# ═════════════════════════════════════════════════════════════════════════════
# construct_F_deconvolution  (from construct_F_deconvolution.m)
# ═════════════════════════════════════════════════════════════════════════════

def construct_F_deconvolution(n: int, gamma: float) -> np.ndarray:
    """
    Build the forward operator for convolution with a Gaussian kernel.

    MATLAB equivalent::

        kernel = @(t) exp(-t.^2/(2*gamma^2)) / sqrt(2*pi*gamma^2);
        grid   = linspace(0, 1, n);
        F      = kernel(grid - grid') / n;

    Parameters
    ----------
    n     : int   — number of equidistant grid points on [0, 1].
    gamma : float — blurring parameter (std-dev of the Gaussian kernel).

    Returns
    -------
    F : (n, n) ndarray — dense forward operator.
    """
    grid = np.linspace(0.0, 1.0, n)
    # MATLAB: grid (1×n row) - grid' (n×1 col) = n×n via implicit expansion.
    # Element (i, j) = grid[j] - grid[i].
    diff = grid[np.newaxis, :] - grid[:, np.newaxis]  # (n, n)
    F = np.exp(-diff ** 2 / (2.0 * gamma ** 2)) / np.sqrt(2.0 * np.pi * gamma ** 2) / n
    return F


# ═════════════════════════════════════════════════════════════════════════════
# TV_operator  (from TV_operator.m)
# ═════════════════════════════════════════════════════════════════════════════

def TV_operator(n: int, order: int) -> sp.csr_matrix:
    """
    Discrete Total-Variation operator of a given order.

    MATLAB equivalent::

        e = ones(n,1);
        % order 1:  D = spdiags([e -e],     0:1, n, n);
        % order 2:  D = spdiags([-e 2e -e],  0:2, n, n);
        % order 3:  D = spdiags([e -3e 3e -e], 0:3, n, n);
        R = D(1:n-order, :);

    Parameters
    ----------
    n     : int — number of equidistant grid points.
    order : int — order of the TV operator (1, 2, or 3).

    Returns
    -------
    R : (n - order, n) sparse CSR matrix.
    """
    if order == 1:
        D = sp.diags([1, -1], [0, 1], shape=(n, n), format='csr')
    elif order == 2:
        D = sp.diags([-1, 2, -1], [0, 1, 2], shape=(n, n), format='csr')
    elif order == 3:
        D = sp.diags([1, -3, 3, -1], [0, 1, 2, 3], shape=(n, n), format='csr')
    else:
        raise ValueError(f'TV order {order} not implemented (use 1, 2, or 3)')

    R = D[:n - order, :]
    return R.tocsr()


# ═════════════════════════════════════════════════════════════════════════════
# compute_CI  (from compute_CI.m)
# ═════════════════════════════════════════════════════════════════════════════

def compute_CI(
    mu: np.ndarray,
    C: np.ndarray,
    n_samples: int = 1000,
    seed: int = 0,
) -> tuple:
    """
    Compute 99.8 % confidence intervals from the posterior distribution.

    Draws *n_samples* from N(mu, C) and returns per-component quantiles
    at 0.001 and 0.999 (matching MATLAB code).

    MATLAB equivalent::

        rng default
        Samples = mvnrnd(mu, C, N);
        CI_lower(i) = quantile(Samples(:,i), 0.001);
        CI_upper(i) = quantile(Samples(:,i), 0.999);

    Parameters
    ----------
    mu        : (n,) ndarray — posterior mean.
    C         : (n, n) ndarray — posterior covariance matrix.
    n_samples : int — number of Monte-Carlo samples (default 1000).
    seed      : int — RNG seed for reproducibility (default 0).

    Returns
    -------
    CI_lower : (n,) ndarray — lower bounds of the confidence intervals.
    CI_upper : (n,) ndarray — upper bounds of the confidence intervals.
    """
    rng = np.random.default_rng(seed)
    # mu must be 1-D for multivariate_normal
    mu_flat = np.asarray(mu, dtype=np.float64).ravel()
    C_dense = np.asarray(C, dtype=np.float64)

    samples = rng.multivariate_normal(mu_flat, C_dense, size=n_samples)  # (N, n)

    CI_lower = np.quantile(samples, 0.001, axis=0)
    CI_upper = np.quantile(samples, 0.999, axis=0)
    return CI_lower, CI_upper


# ═════════════════════════════════════════════════════════════════════════════
# shrinkage  (soft-thresholding, from ADMM_1d.m / ADMM_2d.m)
# ═════════════════════════════════════════════════════════════════════════════

def shrinkage(a: np.ndarray, kappa: float) -> np.ndarray:
    """
    Element-wise soft-thresholding (shrinkage) operator.

    MATLAB equivalent::

        y = max(0, a-kappa) - max(0, -a-kappa);

    Equivalent to  sign(a) * max(|a| - kappa, 0).

    Parameters
    ----------
    a     : ndarray — input array (any shape).
    kappa : float   — threshold.

    Returns
    -------
    y : ndarray — same shape as *a*.
    """
    return np.maximum(0.0, a - kappa) - np.maximum(0.0, -a - kappa)
