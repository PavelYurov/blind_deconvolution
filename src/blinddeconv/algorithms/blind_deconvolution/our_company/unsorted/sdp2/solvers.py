"""
solvers.py

Core solvers for the SDP-based blind deconvolution algorithm (sdp2).

Ported from MATLAB code (blind-deconvolution-main/src/).
Reference:
    A. Ahmed, B. Recht, J. Romberg: "Blind Deconvolution Using Convex
    Programming", IEEE Trans. Inform. Theory, 2014.

Contains:
    nuclear_norm_minimization — solve min ||X||_* s.t. A·vec(X) = b
                                via ADMM (Alternating Direction Method
                                of Multipliers).
    blind_deconv_1d          — 1-D blind deconvolution
                                (from blind_deconv_convex.m).
    blind_deconv_2d          — 2-D blind deconvolution on images
                                (from blind2d.m).

MATLAB → Python conversion notes:
    ─────────────────────────────────────────────────────────────────────
    CVX  minimize( norm_nuc(X) )  s.t.  A*X(:) == y_hat:
        MATLAB uses CVX with SDP solver for nuclear norm minimization.
        CVX is not available in Python without cvxpy.  We implement an
        ADMM solver that solves the equivalent problem:
            min ||Z||_*
            s.t. A·vec(X) = b,  X = Z
        via the augmented Lagrangian:
            L(X,Z,Y) = ||Z||_* + <Y, X-Z> + (rho/2)||X-Z||_F^2
        The X-update is a constrained least-squares (projection onto
        the affine constraint set), the Z-update is a singular value
        thresholding (SVT), and Y is the dual variable.

    MATLAB X(:) — column-major vectorisation:
        The constraint A·vec(X) == b uses MATLAB's column-major
        vectorisation.  Our utils.mat_vec does the same.
        When reshaping back: np.reshape(v, (K,N), order='F').

    MATLAB svd(X) vs np.linalg.svd:
        MATLAB: [U,S,V] = svd(X) → X = U*S*V'
        NumPy:  U, s, Vh = svd(X) → X = U*diag(s)*Vh
        So MATLAB's V = Vh.conj().T  (or Vh.T for real).
        MATLAB's V(:,1) = Vh[0,:].conj() in Python.

    MATLAB waverec2(C*v, s, 'haar'):
        The MATLAB code reconstructs the image from wavelet coefficients
        C*v using the bookkeeping structure s from wavedec2.
        → utils.waverec2_flat(C @ v, bookkeeping, wavelet='haar')

    MATLAB dftmtx(L)*y vs fft(y):
        dftmtx(L)*y == fft(y) for column vectors.
        dftmtx(L)*B computes fft of each column of B == fft(B, axis=0).
        The DFT matrix is used explicitly in blind2d.m because the
        dimensions are large and the matrix is dense.  For the 2D case,
        this is the bottleneck.  For small 1D cases it's fine.
        For large 2D cases we use fft(B, axis=0) instead.

    MATLAB conv_wx_image = fftshift(mat(conv_wx)):
        The MATLAB code applies fftshift after 2D convolution via FFT.
        This centres the zero-frequency component.
        → np.fft.fftshift(result)
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict

from .utils import (
    mat_vec,
    mat_reshape,
    dftmtx,
    wavedec2_flat,
    waverec2_flat,
    build_B_from_kernel,
    build_C_from_image,
    build_linear_operator_A,
    cyclic_conv_2d,
    place_kernel_in_image,
    recover_from_svd,
    fspecial_motion,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Singular Value Thresholding (SVT)
# ═════════════════════════════════════════════════════════════════════════════

def _svt(M: np.ndarray, tau: float) -> np.ndarray:
    """
    Singular Value Thresholding (proximal operator of the nuclear norm).

    SVT(M, tau) = U * max(S - tau, 0) * Vh

    This is the proximal operator for tau * ||·||_*:
        prox_{tau ||·||_*}(M) = argmin_X  tau||X||_* + (1/2)||X - M||_F^2

    Parameters
    ----------
    M   : (m, n) ndarray (real or complex).
    tau : float > 0  — threshold.

    Returns
    -------
    X : (m, n) ndarray — thresholded matrix.
    """
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return (U * s_thresh[np.newaxis, :]) @ Vh


# ═════════════════════════════════════════════════════════════════════════════
# Nuclear Norm Minimization via ADMM
# ═════════════════════════════════════════════════════════════════════════════

def nuclear_norm_minimization(
    A: np.ndarray,
    b: np.ndarray,
    shape: Tuple[int, int],
    rho: float = 1.0,
    max_iter: int = 500,
    atol: float = 1e-6,
    rtol: float = 1e-4,
    verbose: bool = False,
) -> np.ndarray:
    """
    Solve  min ||X||_*  subject to  A·vec(X) = b

    via ADMM (Alternating Direction Method of Multipliers).

    The problem is reformulated as:
        min  ||Z||_*
        s.t. A·vec(X) = b
             X - Z = 0

    ADMM iterations:
        X^{k+1} = argmin_X  (rho/2)||X - Z^k + U^k||_F^2
                   s.t. A·vec(X) = b
        Z^{k+1} = SVT(X^{k+1} + U^k, 1/rho)
        U^{k+1} = U^k + X^{k+1} - Z^{k+1}

    The X-update is a projection onto {X : A·vec(X) = b} from the
    point (Z^k - U^k).  This is done via:
        vec(X) = v + A^H (A A^H)^{-1} (b - A·v)
    where v = vec(Z^k - U^k).

    For the equality constraint A·vec(X) = b, we precompute
    A^H (A A^H)^{-1} for efficiency.

    Parameters
    ----------
    A     : (L, K*N) complex matrix — measurement operator.
    b     : (L,) complex vector — measurements (y_hat).
    shape : (K, N) — shape of the matrix variable X.
    rho   : float — ADMM penalty parameter (default 1.0).
    max_iter : int — maximum ADMM iterations (default 500).
    atol  : float — absolute tolerance for convergence.
    rtol  : float — relative tolerance for convergence.
    verbose : bool — print iteration info.

    Returns
    -------
    X : (K, N) complex ndarray — optimal matrix.

    Notes
    -----
    This replaces MATLAB's CVX call:
        cvx_begin
            variable X(K,N)
            minimize( norm_nuc(X) )
            subject to
                A*X(:) == y_hat
        cvx_end

    The ADMM approach is equivalent for convex problems and converges
    to the same solution as CVX.  The penalty parameter rho controls
    the trade-off between primal and dual convergence rates.
    """
    K, N = shape
    L = A.shape[0]

    # ── Precompute factorisation for X-update ────────────────────────────
    # X-update: project vec(Z-U) onto {x : Ax = b}
    #   x = v + A^H (A A^H)^{-1} (b - Av)
    # where v = vec(Z - U)
    #
    # Precompute: A^H (A A^H)^{-1}
    # For numerical stability, use pseudo-inverse via Cholesky or SVD.
    AAH = A @ A.conj().T  # (L, L)

    # Use regularised solve for stability
    # (A A^H + eps * I)^{-1} for numerical stability
    AAH_reg = AAH + 1e-12 * np.eye(L, dtype=AAH.dtype)
    try:
        from scipy.linalg import cho_factor, cho_solve
        cho = cho_factor(AAH_reg)
        _solve_AAH = lambda rhs: cho_solve(cho, rhs)
    except np.linalg.LinAlgError:
        # Fallback to LU
        from scipy.linalg import lu_factor, lu_solve
        lu = lu_factor(AAH_reg)
        _solve_AAH = lambda rhs: lu_solve(lu, rhs)

    # A^H (AAH)^{-1}
    # For the projection: x = v + AH_AAHinv @ (b - A @ v)
    AH = A.conj().T  # (K*N, L)

    # ── Initialise ──────────────────────────────────────────────────────
    X = np.zeros((K, N), dtype=np.complex128)
    Z = np.zeros((K, N), dtype=np.complex128)
    U = np.zeros((K, N), dtype=np.complex128)

    for iteration in range(max_iter):
        # ── X-update: project onto affine constraint ────────────────────
        v = mat_vec(Z - U)  # column-major flatten, matching MATLAB X(:)
        residual = b - A @ v
        correction = AH @ _solve_AAH(residual)
        x_vec = v + correction
        X = x_vec.reshape((K, N), order='F')

        # ── Z-update: singular value thresholding ───────────────────────
        Z_old = Z.copy()
        Z = _svt(X + U, 1.0 / rho)

        # ── U-update: dual variable ────────────────────────────────────
        U = U + X - Z

        # ── Convergence check ──────────────────────────────────────────
        r_norm = np.linalg.norm(X - Z, 'fro')       # primal residual
        s_norm = rho * np.linalg.norm(Z - Z_old, 'fro')  # dual residual

        eps_pri = np.sqrt(K * N) * atol + rtol * max(
            np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro')
        )
        eps_dual = np.sqrt(K * N) * atol + rtol * np.linalg.norm(U, 'fro')

        if verbose and (iteration % 50 == 0 or iteration == max_iter - 1):
            nuc = np.linalg.svd(Z, compute_uv=False).sum()
            logger.info(
                f"ADMM iter {iteration:4d}: "
                f"||r||={r_norm:.2e}  ||s||={s_norm:.2e}  "
                f"||X||_*={nuc:.4f}"
            )

        if r_norm < eps_pri and s_norm < eps_dual:
            if verbose:
                logger.info(f"ADMM converged at iteration {iteration}")
            break

    return Z


# ═════════════════════════════════════════════════════════════════════════════
# 1-D Blind Deconvolution  (from blind_deconv_convex.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_1d(
    N: int = 16,
    K: int = 16,
    L: int = 256,
    seed: Optional[int] = None,
    admm_rho: float = 1.0,
    admm_max_iter: int = 500,
    verbose: bool = False,
) -> Dict:
    """
    1-D blind deconvolution demo — exact port of blind_deconv_convex.m.

    Generates random signals h, m, convolves them, and recovers via
    nuclear norm minimization.

    MATLAB code:
        N = 16; K = 16; L = 256;
        h = randn(K,1); h = h/norm(h);
        m = randn(N,1); m = m/norm(m);
        idxB = randperm(L); idxB = idxB(1:K);
        B = eye(L); B = B(:,idxB); w = B*h;
        idxC = randperm(L); idxC = idxC(1:N);
        C = eye(L); C = C(:,idxC); x = C*m;
        y = real(ifft(fft(x).*fft(w)));
        B_hat = fft(B); C_hat = fft(C); y_hat = fft(y);
        A = []; for i=1:N, A_l=diag(sqrt(L)*C_hat(:,i)); A=[A A_l*B_hat]; end
        cvx_begin ... minimize( norm_nuc(X) ) s.t. A*X(:)==y_hat ... cvx_end
        [U,S,V] = svd(X); u=U(:,1); v=V(:,1);
        error = norm(u*v'-h*m','fro')/norm(h*m','fro')

    Parameters
    ----------
    N, K, L : int — signal dimensions.
    seed    : int or None — random seed.
    admm_rho : float — ADMM penalty.
    admm_max_iter : int — max ADMM iterations.
    verbose : bool — print diagnostics.

    Returns
    -------
    result : dict with keys:
        'h', 'm'       — true signals.
        'u', 'v'       — recovered singular vectors.
        'X'            — recovered matrix.
        'error'        — relative recovery error.
        'B', 'C'       — subspace matrices.
        'y'            — observation.
    """
    rng = np.random.default_rng(seed)

    # ── Generate signals ────────────────────────────────────────────────
    h = rng.standard_normal(K)
    h /= np.linalg.norm(h)
    m = rng.standard_normal(N)
    m /= np.linalg.norm(m)

    # ── Build B, C as subsets of identity columns ───────────────────────
    idxB = rng.permutation(L)[:K]
    B = np.eye(L)[:, idxB]
    w = B @ h

    idxC = rng.permutation(L)[:N]
    C = np.eye(L)[:, idxC]
    x = C @ m

    # ── Cyclic convolution ──────────────────────────────────────────────
    # MATLAB: y = real(ifft(fft(x).*fft(w)))
    y = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(w)))

    # ── Fourier domain ──────────────────────────────────────────────────
    # MATLAB: B_hat = fft(B);  →  fft of each column
    B_hat = np.fft.fft(B, axis=0)
    C_hat = np.fft.fft(C, axis=0)
    y_hat = np.fft.fft(y)

    # ── Build linear operator A ─────────────────────────────────────────
    A = build_linear_operator_A(B_hat, C_hat, L, N)

    # ── Nuclear norm minimization ───────────────────────────────────────
    X = nuclear_norm_minimization(
        A, y_hat, (K, N),
        rho=admm_rho, max_iter=admm_max_iter, verbose=verbose,
    )

    # ── SVD recovery ────────────────────────────────────────────────────
    u, v, S = recover_from_svd(X)

    # ── Error ───────────────────────────────────────────────────────────
    error = np.linalg.norm(
        np.outer(u, v) - np.outer(h, m), 'fro'
    ) / np.linalg.norm(np.outer(h, m), 'fro')

    return {
        'h': h, 'm': m,
        'u': u, 'v': v,
        'X': X, 'S': S,
        'error': error,
        'B': B, 'C': C,
        'y': y,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2-D Blind Deconvolution  (from blind2d.m)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_2d(
    blurred_image: np.ndarray,
    blur_kernel: np.ndarray,
    wavelet_level: int = 4,
    wavelet_name: str = 'db1',
    threshold_ratio: float = 0.0,
    admm_rho: float = 1.0,
    admm_max_iter: int = 500,
    use_dftmtx: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    2-D blind deconvolution — exact port of blind2d.m
    (section "Blind deconvolution using convex programming").

    This is the full image pipeline:
        blurred image → nuclear norm minimization → restored image + kernel

    MATLAB pipeline (blind2d.m, lines 118–175):
        1. Flatten blur kernel → build B, h from non-zero entries.
        2. Wavelet decomposition of image → build C, m from non-zero coeffs.
        3. Vectorise blurred image (column-major).
        4. Compute DFT: y_hat = DFT * y,  B_hat = DFT * B,  C_hat = DFT * C.
        5. Build linear operator A.
        6. Solve  min ||X||_*  s.t. A·vec(X) = y_hat.
        7. SVD: X → u, v.
        8. Reconstruct: C·v → wavelet coeffs → waverec2 → restored image.

    Parameters
    ----------
    blurred_image : (L1, L2) ndarray — input blurred grayscale image (float64).
    blur_kernel   : (K1, K2) ndarray — known or estimated blur PSF.
    wavelet_level : int — wavelet decomposition depth (default 4).
    wavelet_name  : str — wavelet family (default 'db1' = Haar).
    threshold_ratio : float — wavelet coefficient threshold.
        0.0 keeps all non-zero coefficients (MATLAB default).
        For blind setting on blurred image, use e.g. 0.00018.
    admm_rho      : float — ADMM penalty parameter.
    admm_max_iter : int — max ADMM iterations.
    use_dftmtx    : bool — if True, build explicit DFT matrix (slow for
                    large images, exact MATLAB match).  If False, use
                    fft(·, axis=0) which is equivalent and much faster.
    verbose       : bool — print diagnostics.

    Returns
    -------
    x_restored : (L1, L2) ndarray — restored image.
    kernel     : (K1, K2) ndarray — normalised blur kernel.

    Notes
    -----
    MATLAB vectorisation is COLUMN-MAJOR.  All flatten/reshape operations
    use order='F' to match MATLAB exactly.

    The sign ambiguity in SVD (MATLAB line 175: C_recover = C*v.*(-1))
    is handled by checking which sign gives a better match.  In general,
    SVD determines singular vectors only up to sign, so we try both
    and pick the one with larger positive energy.
    """
    L1, L2 = blurred_image.shape
    L = L1 * L2

    # ── Step 1: Build B from kernel ─────────────────────────────────────
    # MATLAB (blind2d.m, lines 120-134):
    #   kernel = blur_kernel(:);   (column-major)
    #   K = sum(kernel ~= 0);
    #   B = zeros(length(kernel), K);
    #   ...
    B_kernel, h_kernel = build_B_from_kernel(blur_kernel, (L1, L2))
    K = len(h_kernel)

    # ── Step 2: Build C from wavelet decomposition of image ─────────────
    # MATLAB (blind2d.m, lines 136-148):
    #   [C_haar, s] = wavedec2(x, 4, 'db1');
    #   N = sum(C_haar ~= 0);
    #   C = zeros(length(img), N);
    #   ...
    # NOTE: In the MATLAB code, C is built from the ORIGINAL image x.
    # In a truly blind setting, you'd use the blurred image instead.
    # We use whatever image is passed (blurred_image).
    C_wavelet, m_wavelet, bookkeeping = build_C_from_image(
        blurred_image, wavelet_level, wavelet_name, threshold_ratio
    )
    N_coeffs = len(m_wavelet)

    logger.info(f"K (kernel non-zeros) = {K}, N (wavelet coeffs) = {N_coeffs}, L = {L}")
    logger.info(f"Matrix A will be ({L}, {K * N_coeffs}) — "
                f"{L * K * N_coeffs * 16 / 1e9:.2f} GB complex")

    # ── Step 3: Vectorise blurred image (column-major) ──────────────────
    # MATLAB: y = conv_wx_image(:);
    y = mat_vec(blurred_image)

    # ── Step 4: Compute DFT ─────────────────────────────────────────────
    # MATLAB:
    #   y_hat = dftmtx(L) * y;
    #   B_hat = dftmtx(L) * B;
    #   C_hat = dftmtx(L) * C;
    #
    # For efficiency, dftmtx(L)*v  ==  fft(v)  for a column vector,
    # and dftmtx(L)*M  ==  fft(M, axis=0)  for a matrix.
    if use_dftmtx:
        # Exact MATLAB match (very slow and memory-intensive for large L)
        W = dftmtx(L)
        y_hat = W @ y
        B_hat = W @ B_kernel
        C_hat = W @ C_wavelet
    else:
        # Equivalent but much faster via FFT
        y_hat = np.fft.fft(y)
        B_hat = np.fft.fft(B_kernel, axis=0)
        C_hat = np.fft.fft(C_wavelet, axis=0)

    # ── Step 5: Build linear operator A ─────────────────────────────────
    # MATLAB:
    #   A = [];
    #   for i = 1:N
    #       A_l = diag(sqrt(L) * C_hat(:,i));
    #       A = [A  A_l * B_hat];
    #   end
    A = build_linear_operator_A(B_hat, C_hat, L, N_coeffs)

    # ── Step 6: Nuclear norm minimization ───────────────────────────────
    # MATLAB:
    #   cvx_begin
    #       variable X(K,N)
    #       minimize( norm_nuc(X) )
    #       subject to  A*X(:) == y_hat
    #   cvx_end
    X = nuclear_norm_minimization(
        A, y_hat, (K, N_coeffs),
        rho=admm_rho, max_iter=admm_max_iter, verbose=verbose,
    )

    # ── Step 7: SVD recovery ────────────────────────────────────────────
    # MATLAB:
    #   [U,S,V] = svd(X);
    #   u = U(:,1);
    #   v = V(:,1);
    u, v, S_diag = recover_from_svd(X)

    # ── Step 8: Image reconstruction from wavelet coefficients ──────────
    # MATLAB (blind2d.m):
    #   C_recover = C * v .* (-1);
    #   x_dec = waverec2(C_recover, s, 'haar');
    #
    # The .*(-1) compensates for SVD sign ambiguity.
    # We try both signs and pick the one with more positive energy.
    C_recover_pos = np.real(C_wavelet @ v)
    C_recover_neg = -C_recover_pos

    x_pos = waverec2_flat(C_recover_pos, bookkeeping, wavelet=wavelet_name)
    x_neg = waverec2_flat(C_recover_neg, bookkeeping, wavelet=wavelet_name)

    # Pick the sign that gives a more plausible image
    # (more positive pixel values for a natural image)
    if np.sum(x_pos > 0) >= np.sum(x_neg > 0):
        x_restored = x_pos
    else:
        x_restored = x_neg

    # ── Kernel recovery (optional, from u) ──────────────────────────────
    # MATLAB doesn't explicitly reconstruct the kernel from u in blind2d.m,
    # but u corresponds to h (kernel coefficients).
    # The kernel can be recovered as: B_kernel @ u → then reshape.
    kernel_recovered_flat = np.real(B_kernel @ u)
    # Determine sign consistency with h_kernel
    if np.dot(kernel_recovered_flat, B_kernel @ h_kernel) < 0:
        kernel_recovered_flat = -kernel_recovered_flat

    # Reshape back to kernel shape
    kernel_recovered = kernel_recovered_flat.reshape(blur_kernel.shape, order='F')
    kernel_recovered = np.abs(kernel_recovered)
    s = kernel_recovered.sum()
    if s > 0:
        kernel_recovered /= s

    return x_restored, kernel_recovered


# ═════════════════════════════════════════════════════════════════════════════
# 2-D Blind Deconvolution — full pipeline (from blur to restoration)
# ═════════════════════════════════════════════════════════════════════════════

def blind_deconv_2d_full_pipeline(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    motion_length: int = 5,
    motion_angle: float = 45.0,
    kernel_type: str = 'motion',
    custom_kernel: Optional[np.ndarray] = None,
    wavelet_level: int = 4,
    wavelet_name: str = 'db1',
    threshold_ratio: float = 0.0,
    admm_rho: float = 1.0,
    admm_max_iter: int = 500,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline: original image → blur → blind deconvolution → restored image.

    This replicates the full blind2d.m workflow including the blurring step,
    useful for testing.  For actual blind deconvolution of an already-blurred
    image, use blind_deconv_2d directly.

    MATLAB pipeline:
        1. Load image, convert to grayscale, normalise.
        2. Create motion blur kernel.
        3. Place kernel in image-sized grid, convolve.
        4. Run blind deconvolution on the convolved image.

    Parameters
    ----------
    image           : (L1, L2) or (L1, L2, 3) — input image.
    kernel_size     : (K1, K2) — kernel dimensions (used for 'gaussian').
    motion_length   : int — motion blur length (for 'motion').
    motion_angle    : float — motion blur angle in degrees.
    kernel_type     : str — 'motion', 'gaussian', or 'custom'.
    custom_kernel   : ndarray or None — user-provided kernel.
    wavelet_level   : int — wavelet decomposition depth.
    wavelet_name    : str — wavelet family.
    threshold_ratio : float — wavelet coefficient threshold.
    admm_rho        : float — ADMM penalty parameter.
    admm_max_iter   : int — max ADMM iterations.
    verbose         : bool — print diagnostics.

    Returns
    -------
    x_restored    : (L1, L2) ndarray — restored image.
    blur_kernel   : (K1, K2) ndarray — blur kernel used.
    """
    # ── 1. Normalise image ──────────────────────────────────────────────
    # MATLAB: x = mean(rgb,3); x = double(x)/norm(x,'fro');
    if image.ndim == 3:
        x = np.mean(image.astype(np.float64), axis=2)
    else:
        x = image.astype(np.float64)

    x /= np.linalg.norm(x, 'fro')

    L1, L2 = x.shape

    # ── 2. Create blur kernel ───────────────────────────────────────────
    # MATLAB: blur_kernel = fspecial('motion', 5, 45);
    #         blur_kernel = blur_kernel / norm(blur_kernel, 'fro');
    if custom_kernel is not None:
        blur_kernel = custom_kernel.astype(np.float64)
    elif kernel_type == 'motion':
        blur_kernel = fspecial_motion(motion_length, motion_angle)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    blur_kernel /= np.linalg.norm(blur_kernel, 'fro')

    # ── 3. Convolve ─────────────────────────────────────────────────────
    # MATLAB:
    #   w = zeros(L1,L2);
    #   w(L1/2-(K1+1)/2+2:L1/2+(K1+1)/2, ...) = blur_kernel;
    #   conv_wx = ifft2(fft2(x).*fft2(w));
    #   conv_wx_image = fftshift(conv_wx);
    w = place_kernel_in_image(blur_kernel, (L1, L2))
    conv_wx = cyclic_conv_2d(x, w)
    conv_wx_image = np.fft.fftshift(conv_wx)

    # ── 4. Run blind deconvolution ──────────────────────────────────────
    x_restored, kernel_recovered = blind_deconv_2d(
        conv_wx_image, blur_kernel,
        wavelet_level=wavelet_level,
        wavelet_name=wavelet_name,
        threshold_ratio=threshold_ratio,
        admm_rho=admm_rho,
        admm_max_iter=admm_max_iter,
        verbose=verbose,
    )

    return x_restored, kernel_recovered
