"""
utils.py

Utility functions for the Convex SDP blind deconvolution algorithm.

Ported from MATLAB code (Convex_src / DeblurAlgorithm1.m).
Reference:
    A. Ahmed, B. Recht, J. Romberg: "Blind Deconvolution Using Convex
    Programming", IEEE Trans. Inform. Theory, 2014. (arXiv:1211.5608)

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    MATLAB vec = @(x) x(:)
        MATLAB linearises matrices in COLUMN-MAJOR (Fortran) order.
        → np.ravel(x, order='F')  or  x.ravel(order='F')
        DO NOT use the default row-major order.

    MATLAB reshape(x, L1, L2):
        Column-major reshape.
        → np.reshape(x, (L1, L2), order='F')

    MATLAB wavedec2(X, level, 'db1'):
        Returns a 1-D coefficient vector [cA | cH | cV | cD] for each
        level, packed in a specific order, plus a bookkeeping matrix.
        → pywt.wavedec2 returns a list:
          [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
        We provide wrappers that exactly replicate the MATLAB 1-D
        packing so that matrix C and vector m are compatible.

    MATLAB waverec2(c, l, 'db1'):
        Reconstructs image from 1-D coefficient vector + bookkeeping.
        → wrapper around pywt.waverec2.

    MATLAB fft2 / ifft2:
        No normalisation.  np.fft.fft2 / ifft2 behave identically.

    MATLAB fftshift:
        np.fft.fftshift — identical for 2-D arrays.

    MATLAB svd(M, 'econ'):
        Thin SVD.
        → np.linalg.svd(M, full_matrices=False)

    MATLAB sparse(L, K):
        SciPy sparse.  We use scipy.sparse.lil_matrix for construction,
        then convert to csc for fast column slicing / multiplication.
"""

import numpy as np
import pywt
from scipy import sparse
from typing import Tuple, List


# ═════════════════════════════════════════════════════════════════════════════
# Column-major helpers (replicate MATLAB vec / mat)
# ═════════════════════════════════════════════════════════════════════════════

def vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorise a matrix in Fortran (column-major) order,
    matching MATLAB's ``x(:)``.

    Returns a 1-D array.
    """
    return x.ravel(order='F')


def mat(x: np.ndarray, L1: int, L2: int) -> np.ndarray:
    """
    Reshape a 1-D vector back into an (L1, L2) matrix
    in Fortran (column-major) order, matching MATLAB's
    ``reshape(x, L1, L2)``.
    """
    return np.reshape(x, (L1, L2), order='F')


# ═════════════════════════════════════════════════════════════════════════════
# 2-D Wavelet transform wrappers (MATLAB-compatible packing)
# ═════════════════════════════════════════════════════════════════════════════
#
# MATLAB wavedec2 returns:
#   [c, S] where c is a ROW vector containing the coefficients packed as:
#       [cA_n | cH_n | cV_n | cD_n | ... | cH_1 | cV_1 | cD_1]
#   and S is a bookkeeping matrix of sizes.
#
# The DETAIL coefficients at each level (cHi, cVi, cDi) are each
# linearised in COLUMN-MAJOR order, matching MATLAB's (:).
#
# pywt.wavedec2 returns:
#   [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
# where each element is a 2-D array.  The detail tuple order is
# (cH, cV, cD) in pywt — BUT NOTE: pywt uses the convention
# ('da', 'ad', 'dd') which maps to (cH, cV, cD) in MATLAB order
# because MATLAB's ordering is:
#   horizontal detail (cH) ↔ pywt 'da'  (rows=detail, cols=approx)
#   vertical   detail (cV) ↔ pywt 'ad'  (rows=approx, cols=detail)
#   diagonal   detail (cD) ↔ pywt 'dd'
# So the tuple order from pywt already matches MATLAB's (cH, cV, cD).
# ═════════════════════════════════════════════════════════════════════════════


def wavedec2(
    x: np.ndarray, level: int, wavelet: str = 'db1'
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    2-D discrete wavelet decomposition returning a 1-D coefficient
    vector and bookkeeping list, matching MATLAB ``wavedec2``.

    Parameters
    ----------
    x : (M, N) ndarray
        Input 2-D signal.
    level : int
        Decomposition level.
    wavelet : str
        Wavelet name (default 'db1' = Haar).

    Returns
    -------
    coeffs_1d : 1-D ndarray
        Packed coefficient vector in MATLAB order:
        [cA_n | cH_n | cV_n | cD_n | ... | cH_1 | cV_1 | cD_1]
        Each 2-D block is linearised in column-major (Fortran) order.
    bookkeeping : list of (rows, cols) tuples
        Sizes for reconstruction.  The list has (level + 2) entries:
        [approx_shape, detail_shape_level_n, ..., detail_shape_level_1,
         original_shape].
    """
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)

    # coeffs[0] = cA_n  (2-D)
    # coeffs[i] = (cH_{n-i+1}, cV_{n-i+1}, cD_{n-i+1}) for i=1..n

    parts = []
    bookkeeping = []

    # Approximation coefficients at deepest level
    cA = coeffs[0]
    parts.append(vec(cA))
    bookkeeping.append(cA.shape)

    # Detail coefficients from deepest to shallowest level
    for detail_tuple in coeffs[1:]:
        cH, cV, cD = detail_tuple
        bookkeeping.append(cH.shape)
        parts.append(vec(cH))
        parts.append(vec(cV))
        parts.append(vec(cD))

    # Original shape (last entry in MATLAB's S bookkeeping)
    bookkeeping.append(x.shape)

    coeffs_1d = np.concatenate(parts)
    return coeffs_1d, bookkeeping


def waverec2(
    coeffs_1d: np.ndarray,
    bookkeeping: List[Tuple[int, int]],
    wavelet: str = 'db1',
) -> np.ndarray:
    """
    2-D discrete wavelet reconstruction from a 1-D coefficient vector
    and bookkeeping list, matching MATLAB ``waverec2``.

    Parameters
    ----------
    coeffs_1d : 1-D ndarray
        Packed coefficients (see wavedec2).
    bookkeeping : list of (rows, cols) tuples
        Bookkeeping from wavedec2.
    wavelet : str
        Wavelet name.

    Returns
    -------
    x_rec : 2-D ndarray
        Reconstructed signal.
    """
    n_levels = len(bookkeeping) - 2  # subtract approx_shape and original_shape

    idx = 0

    # Unpack approximation coefficients
    r, c = bookkeeping[0]
    size_block = r * c
    cA = mat(coeffs_1d[idx:idx + size_block], r, c)
    idx += size_block

    # Unpack detail coefficients
    detail_list = []
    for lev in range(n_levels):
        r, c = bookkeeping[1 + lev]
        size_block = r * c
        cH = mat(coeffs_1d[idx:idx + size_block], r, c)
        idx += size_block
        cV = mat(coeffs_1d[idx:idx + size_block], r, c)
        idx += size_block
        cD = mat(coeffs_1d[idx:idx + size_block], r, c)
        idx += size_block
        detail_list.append((cH, cV, cD))

    # Reconstruct using pywt
    pywt_coeffs = [cA] + detail_list
    x_rec = pywt.waverec2(pywt_coeffs, wavelet=wavelet)

    # pywt may return an array slightly larger than the original due to
    # padding; crop to the original shape stored in bookkeeping[-1].
    orig_shape = bookkeeping[-1]
    x_rec = x_rec[:orig_shape[0], :orig_shape[1]]

    return x_rec


# ═════════════════════════════════════════════════════════════════════════════
# Build selection matrix B (kernel subspace)
# ═════════════════════════════════════════════════════════════════════════════

def build_kernel_subspace(
    w: np.ndarray,
) -> Tuple[sparse.csc_matrix, np.ndarray, int]:
    """
    Construct the kernel selection matrix B and coefficient vector h.

    Given a 2-D blur kernel w, find the support (non-zero positions)
    in Fortran-order vectorisation and build a sparse matrix B such that
        vec(w) = B @ h
    where h contains only the non-zero entries.

    This replicates the MATLAB code in DeblurAlgorithm1.m lines 24–42.

    Parameters
    ----------
    w : (L1, L2) ndarray
        Blur kernel (may have zeros).

    Returns
    -------
    B : (L, K) sparse csc_matrix
        Selection matrix (columns of identity at support indices).
    h : (K,) ndarray
        Non-zero kernel coefficients.
    K : int
        Number of non-zero entries in the kernel.
    """
    w_vec = vec(w)
    L = w_vec.size
    support = np.abs(w_vec) > 0
    K = int(np.sum(support))

    # Build sparse selection matrix
    rows = np.where(support)[0]
    cols = np.arange(K)
    data = np.ones(K, dtype=np.float64)
    B = sparse.csc_matrix((data, (rows, cols)), shape=(L, K))

    h = w_vec[support].copy()

    return B, h, K


# ═════════════════════════════════════════════════════════════════════════════
# Build selection matrix C (image wavelet subspace)
# ═════════════════════════════════════════════════════════════════════════════

def build_image_subspace(
    y: np.ndarray,
    L1: int,
    L2: int,
    level: int = 4,
    wavelet: str = 'db1',
    threshold_ratio: float = 0.0005,
) -> Tuple[sparse.csc_matrix, np.ndarray, int, List[Tuple[int, int]]]:
    """
    Construct the image wavelet selection matrix C, the coefficient
    vector m (from the blurred image), and bookkeeping info.

    Replicates MATLAB DeblurAlgorithm1.m lines 48–82.
    Support is estimated ONLY from the blurred image (no oracle).

    Parameters
    ----------
    y : (L1, L2) ndarray
        Blurred image.
    L1, L2 : int
        Image dimensions.
    level : int
        Wavelet decomposition level (default 4).
    wavelet : str
        Wavelet name.
    threshold_ratio : float
        Coefficients with |c| > threshold_ratio * max(|c|) are kept.

    Returns
    -------
    C : (P, N) sparse csc_matrix
        Wavelet coefficient selection matrix.
    m : (N,) ndarray
        Selected wavelet coefficients from blurred image.
    N : int
        Number of selected coefficients.
    bookkeeping : list
        Wavelet bookkeeping for reconstruction.
    """
    # MATLAB: conv_wx_image = fftshift(mat(y))
    # In the MATLAB code, y is already vectorised; here we take the 2D image,
    # apply fftshift, then wavelet-decompose.
    conv_wx_image = np.fft.fftshift(y)

    # Cap level to the maximum allowed by the image size
    max_level = pywt.dwt_max_level(min(L1, L2), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)

    alpha_conv, bookkeeping = wavedec2(conv_wx_image, level, wavelet)

    # Threshold: keep coefficients whose magnitude exceeds
    # threshold_ratio * max(|alpha_conv|)
    max_val = np.max(np.abs(alpha_conv))
    Ind = np.abs(alpha_conv) > threshold_ratio * max_val

    N = int(np.sum(Ind))
    P = len(alpha_conv)

    # Build sparse selection matrix C
    rows = np.where(Ind)[0]
    cols = np.arange(N)
    data = np.ones(N, dtype=np.float64)
    C = sparse.csc_matrix((data, (rows, cols)), shape=(P, N))

    # Selected coefficients from blurred image
    m = alpha_conv[Ind].copy()

    return C, m, N, bookkeeping


# ═════════════════════════════════════════════════════════════════════════════
# Operator wrappers: CC, CCT, BB, BBT
# ═════════════════════════════════════════════════════════════════════════════

def make_CC_operator(
    C: sparse.csc_matrix,
    bookkeeping: List[Tuple[int, int]],
    wavelet: str = 'db1',
):
    """
    Build the CC operator and its transpose CCT.

    CC:  coefficients (N,) → image (L1, L2)
         CC(x) = waverec2(C @ x, bookkeeping, wavelet)

    CCT: image (L1, L2) → coefficients (N,)
         CCT(x) = C.T @ wavedec2(x, level, wavelet)[0]

    Returns
    -------
    CC : callable (N,) → (L1, L2)
    CCT : callable (L1, L2) → (N,)
    level : int
    """
    # Determine level from bookkeeping: len(bookkeeping) - 2
    level = len(bookkeeping) - 2

    def CC(x):
        full_coeffs = C @ x
        return waverec2(full_coeffs, bookkeeping, wavelet)

    def CCT(x):
        alpha, _ = wavedec2(x, level, wavelet)
        return C.T @ alpha

    return CC, CCT


def make_BB_operator(B: sparse.csc_matrix, L1: int, L2: int):
    """
    Build the BB operator and its transpose BBT.

    BB:  coefficients (K,) → kernel image (L1, L2)
         BB(x) = mat(B @ x, L1, L2)

    BBT: kernel image (L1, L2) → coefficients (K,)
         BBT(x) = B.T @ vec(x)

    Returns
    -------
    BB : callable (K,) → (L1, L2)
    BBT : callable (L1, L2) → (K,)
    """
    def BB(x):
        return mat(B @ x, L1, L2)

    def BBT(x):
        return B.T @ vec(x)

    return BB, BBT


# ═════════════════════════════════════════════════════════════════════════════
# SVD-based extraction of rank-1 estimates
# ═════════════════════════════════════════════════════════════════════════════

def extract_estimates(
    M: np.ndarray, H: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract rank-1 estimates of wavelet coefficients m and kernel
    coefficients h from the factors M and H returned by the solver.

    Replicates MATLAB DeblurAlgorithm1.m lines 94–100:
        [UM,SM,VM] = svd(M,'econ');
        [UH,SH,VH] = svd(H,'econ');
        [U2,S2,V2] = svd(SM*VM'*VH*SH);
        mEst = sqrt(S2(1,1)) * UM * U2(:,1);
        hEst = sqrt(S2(1,1)) * UH * V2(:,1);

    Parameters
    ----------
    M : (n1, maxrank) ndarray
        First factor from the solver.
    H : (n2, maxrank) ndarray
        Second factor from the solver.

    Returns
    -------
    mEst : (n1,) ndarray
        Estimated wavelet coefficients.
    hEst : (n2,) ndarray
        Estimated kernel coefficients.
    """
    UM, SM_diag, VMt = np.linalg.svd(M, full_matrices=False)
    UH, SH_diag, VHt = np.linalg.svd(H, full_matrices=False)

    # MATLAB: SM * VM' * VH * SH
    # SM, SH are diagonal matrices in MATLAB; in numpy they are 1-D arrays.
    # VM is returned as VM (columns are right singular vectors) in MATLAB.
    # np.linalg.svd returns VMt = VM' (VM transposed).
    #
    # MATLAB: SM * VM' = diag(SM_diag) @ VM'  → in numpy: diag(SM_diag) @ VMt
    # MATLAB: VH * SH = VH @ diag(SH_diag)   → in numpy: VHt.T @ diag(SH_diag)
    #
    # So: SM * VM' * VH * SH = diag(SM_diag) @ VMt @ VHt.T @ diag(SH_diag)

    SM = np.diag(SM_diag)
    SH = np.diag(SH_diag)
    VM = VMt.T  # MATLAB's VM (columns = right singular vectors)
    VH = VHt.T  # MATLAB's VH

    cross = SM @ VM.T @ VH @ SH  # same as diag(SM_diag) @ VMt @ VHt.T @ diag(SH_diag)

    U2, S2_diag, V2t = np.linalg.svd(cross, full_matrices=False)

    s1 = np.sqrt(S2_diag[0])
    mEst = s1 * (UM @ U2[:, 0])
    hEst = s1 * (UH @ V2t[0, :])  # V2(:,1) in MATLAB = V2t[0,:] transposed

    return mEst, hEst


# ═════════════════════════════════════════════════════════════════════════════
# L-BFGS optimiser (wrapper around scipy)
# ═════════════════════════════════════════════════════════════════════════════

def lbfgs_minimize(
    func_and_grad,
    x0: np.ndarray,
    maxiter: int = 500,
    maxfun: int = 50000,
    pgtol: float = 1e-12,
    factr: float = 1e1,
) -> np.ndarray:
    """
    Minimise a function using L-BFGS-B (scipy), matching the role of
    MATLAB's ``minFunc`` with Method='lbfgs'.

    Parameters
    ----------
    func_and_grad : callable
        Function that takes x (1-D array) and returns (f, g) where
        f is the scalar objective value and g is the gradient (1-D).
    x0 : 1-D ndarray
        Initial point.
    maxiter : int
        Maximum number of L-BFGS iterations.
    maxfun : int
        Maximum number of function evaluations.
    pgtol : float
        Projected gradient tolerance for convergence.
    factr : float
        Factor for function-value convergence: stops when
        (f^k - f^{k+1})/max(|f^k|,|f^{k+1}|,1) <= factr*eps.

    Returns
    -------
    x_opt : 1-D ndarray
        Optimised parameters.
    """
    from scipy.optimize import minimize as sp_minimize

    result = sp_minimize(
        func_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,  # func_and_grad returns (f, g)
        options={
            'maxiter': maxiter,
            'maxfun': maxfun,
            'ftol': factr * np.finfo(np.float64).eps,
            'gtol': pgtol,
            'disp': False,
        },
    )
    return result.x
