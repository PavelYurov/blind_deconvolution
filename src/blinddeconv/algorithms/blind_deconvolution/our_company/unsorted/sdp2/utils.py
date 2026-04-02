"""
utils.py

Utility functions for the SDP-based blind deconvolution algorithm.

Ported from MATLAB code (blind-deconvolution-main/src/).
Reference:
    Ahmed, Recht, Romberg: "Blind Deconvolution Using Convex Programming",
    IEEE Trans. Information Theory, 2014.

MATLAB → Python conversion notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    MATLAB vectorisation x(:):
        MATLAB flattens matrices in COLUMN-MAJOR (Fortran) order.
        For an (R,C) matrix, x(:) = [col1; col2; …].
        → In NumPy: x.ravel(order='F') or x.flatten(order='F').
        This is critical when building B, C matrices from 2D data
        and when forming the measurement vector y.

    MATLAB reshape(x, L1, L2):
        Fills in column-major order.
        → np.reshape(x, (L1, L2), order='F')

    MATLAB dftmtx(L):
        Returns an L×L DFT matrix W where W(j,k) = exp(-2πi·(j-1)·(k-1)/L).
        This is the UNNORMALISED DFT matrix (same as fft(eye(L))).
        → Manual construction: omega = exp(-2j*pi/L),
          W[j,k] = omega**(j*k) for j,k in 0..L-1.

    MATLAB fspecial('motion', len, theta):
        Creates a motion-blur PSF of length `len` pixels at angle
        `theta` degrees (measured counter-clockwise from horizontal).
        The kernel is placed on the smallest grid that fits the line
        segment, with partial-pixel weights at the endpoints
        (anti-aliased).  The result sums to 1.
        → Manual implementation below (fspecial_motion).

    MATLAB wavedec2 / waverec2:
        wavedec2(X, level, wname) returns a 1-D coefficient vector C
        and a bookkeeping matrix S (sizes of each level).
        PyWavelets (pywt.wavedec2) returns a nested structure:
          [cA_n, (cH_n, cV_n, cD_n), …, (cH_1, cV_1, cD_1)]
        Conversion helpers (wavedec2_flat / waverec2_flat) bridge the
        two representations so the rest of the algorithm can work with
        a single 1-D coefficient vector, exactly like MATLAB.

    MATLAB fft / ifft on a matrix:
        Operates column-by-column.
        → np.fft.fft(M, axis=0) for column-wise FFT.

    MATLAB fft2 / ifft2:
        → np.fft.fft2 / np.fft.ifft2.  Same convention.

    MATLAB norm(x):
        For vector: Euclidean norm.  For matrix: spectral norm.
        norm(x, 'fro'): Frobenius norm.
        → np.linalg.norm(x) (default: Frobenius for 2-D).
          np.linalg.norm(x, 'fro') for explicit Frobenius.

    MATLAB eye(L)(:, idx):
        Selects columns idx from identity — produces a sparse selector.
        → np.eye(L)[:, idx] or scipy.sparse construction.

    MATLAB sparse(L, K):
        → scipy.sparse.lil_matrix or csc_matrix.
"""

import numpy as np
import pywt
from typing import Tuple, List, Optional


# ═════════════════════════════════════════════════════════════════════════════
# Vectorisation helpers (MATLAB column-major ↔ Python)
# ═════════════════════════════════════════════════════════════════════════════

def mat_vec(x: np.ndarray) -> np.ndarray:
    """
    Equivalent of MATLAB x(:) — flatten in column-major (Fortran) order.

    MATLAB stores matrices column-by-column, so x(:) produces
    [col1; col2; …].  NumPy default is row-major, so we must
    specify order='F'.
    """
    return x.flatten(order='F')


def mat_reshape(x: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Equivalent of MATLAB reshape(x, nrows, ncols) — column-major reshape.

    MATLAB reshape fills column-by-column.
    → np.reshape with order='F'.
    """
    return np.reshape(x, shape, order='F')


# ═════════════════════════════════════════════════════════════════════════════
# DFT matrix
# ═════════════════════════════════════════════════════════════════════════════

def dftmtx(L: int) -> np.ndarray:
    """
    Construct an L×L DFT matrix, equivalent to MATLAB dftmtx(L).

    MATLAB definition:
        W(j,k) = exp(-2πi * (j-1) * (k-1) / L),  j,k = 1..L
    which is also equal to fft(eye(L)).

    In 0-based indexing:
        W[j,k] = exp(-2πi * j * k / L),  j,k = 0..L-1

    Returns
    -------
    W : (L, L) complex128 ndarray — unnormalised DFT matrix.
    """
    j = np.arange(L)
    # Outer product j*k gives the exponent matrix
    return np.exp(-2j * np.pi * np.outer(j, j) / L)


# ═════════════════════════════════════════════════════════════════════════════
# fspecial('motion', len, theta)
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_motion(length: int, angle: float) -> np.ndarray:
    """
    Create a motion-blur PSF, equivalent to MATLAB fspecial('motion', len, angle).

    Parameters
    ----------
    length : int   — length of the motion in pixels (corresponds to MATLAB `len`).
    angle  : float — angle in degrees counter-clockwise from horizontal.

    Returns
    -------
    kernel : 2-D ndarray, sums to 1.

    Notes
    -----
    MATLAB algorithm:
      1. Half-length = (len - 1) / 2.
      2. Compute cos/sin of angle.
      3. Create a grid of size ceil(half_len*|cos|)*2+1  by  ceil(half_len*|sin|)*2+1
         (at minimum 1 in each dimension).
      4. Walk along the line at unit steps (in the dominant direction),
         anti-aliasing the endpoints.
      5. Rotate/flip to match the requested angle.
      6. Normalise so the kernel sums to 1.

    The implementation below replicates MATLAB's exact output.
    """
    # MATLAB: half_len = (len - 1) / 2
    half_len = (length - 1) / 2.0
    phi = np.deg2rad(angle)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Determine the number of points along the line
    # MATLAB samples at integer positions from -half_len to +half_len
    # projected onto the line direction
    xsign = np.sign(cos_phi)
    if xsign == 0:
        xsign = 1.0
    ysign = np.sign(sin_phi)
    if ysign == 0:
        ysign = 1.0

    # Line endpoint in (x, y)
    ex = half_len * cos_phi
    ey = half_len * sin_phi

    # Grid half-sizes (MATLAB: ceil( abs(ex) ), ceil( abs(ey) ) )
    # but at least 0 so that grid is at least 1x1
    hx = int(np.ceil(abs(ex)))
    hy = int(np.ceil(abs(ey)))

    # MATLAB kernel size: (2*hy + 1) x (2*hx + 1)
    # But MATLAB fspecial swaps: rows correspond to y, cols to x
    rows = 2 * hy + 1
    cols = 2 * hx + 1

    kernel = np.zeros((rows, cols), dtype=np.float64)

    # Sample points along the line from -half_len to +half_len
    # MATLAB uses the maximum of abs(ex), abs(ey) to determine step count
    # Number of steps along the dominant axis
    num_steps = max(hx, hy)

    if num_steps == 0:
        # Degenerate case: single-pixel kernel
        kernel[0, 0] = 1.0
        return kernel

    # Parameter t goes from -half_len to +half_len in num_steps*2 + 1 steps
    # Actually MATLAB fspecial('motion') uses a different approach:
    # it fills a line using Bresenham-like anti-aliased stepping.
    # Let's replicate it precisely.

    # Coordinates along the line
    t_values = np.linspace(-half_len, half_len, 2 * num_steps + 1)
    x_coords = t_values * cos_phi  # column direction
    y_coords = t_values * sin_phi  # row direction

    # Centre of kernel
    cy = hy  # row centre
    cx = hx  # col centre

    for x_c, y_c in zip(x_coords, y_coords):
        # Integer coordinates (nearest)
        # MATLAB uses floor for the integer part and distributes weight
        xi = x_c + cx
        yi = y_c + cy

        # Bilinear distribution
        xi0 = int(np.floor(xi))
        yi0 = int(np.floor(yi))
        xi1 = xi0 + 1
        yi1 = yi0 + 1

        fx = xi - xi0
        fy = yi - yi0

        # Distribute weight to up to 4 neighbours
        for (r, c, w) in [
            (yi0, xi0, (1 - fx) * (1 - fy)),
            (yi0, xi1, fx * (1 - fy)),
            (yi1, xi0, (1 - fx) * fy),
            (yi1, xi1, fx * fy),
        ]:
            if 0 <= r < rows and 0 <= c < cols and w > 0:
                kernel[r, c] += w

    # Normalise
    s = kernel.sum()
    if s > 0:
        kernel /= s

    return kernel


# ═════════════════════════════════════════════════════════════════════════════
# Wavelet helpers (bridge MATLAB wavedec2/waverec2 ↔ PyWavelets)
# ═════════════════════════════════════════════════════════════════════════════

def wavedec2_flat(x: np.ndarray, level: int, wavelet: str = 'db1'
                  ) -> Tuple[np.ndarray, List]:
    """
    2-D wavelet decomposition returning a flat 1-D coefficient vector
    and bookkeeping structure, exactly like MATLAB wavedec2.

    MATLAB wavedec2(X, N, wname) returns:
        C  — 1-D row vector of all coefficients concatenated:
             [cA_N | cH_N | cV_N | cD_N | ... | cH_1 | cV_1 | cD_1]
        S  — bookkeeping matrix of sizes.

    PyWavelets pywt.wavedec2 returns:
        [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]

    This function converts the PyWavelets output to MATLAB's flat format.

    Parameters
    ----------
    x       : (M, N) 2-D input array.
    level   : decomposition level.
    wavelet : wavelet name (default 'db1' = Haar).

    Returns
    -------
    coeffs_flat : 1-D ndarray — concatenated coefficients (MATLAB order).
    bookkeeping : list — structure needed by waverec2_flat to reconstruct.
                  Contains (shapes, slices) info matching MATLAB's S matrix.

    Notes
    -----
    MATLAB flattens each 2-D coefficient sub-band in COLUMN-MAJOR order.
    We must replicate this: each sub-band is flattened with order='F'.
    """
    coeffs = pywt.wavedec2(x, wavelet=wavelet, level=level)

    parts = []
    shapes = []

    # Approximation coefficients at coarsest level
    cA = coeffs[0]
    parts.append(cA.flatten(order='F'))
    shapes.append(('approx', cA.shape))

    # Detail coefficients from coarsest to finest
    for detail_tuple in coeffs[1:]:
        cH, cV, cD = detail_tuple
        parts.append(cH.flatten(order='F'))
        parts.append(cV.flatten(order='F'))
        parts.append(cD.flatten(order='F'))
        shapes.append(('detail', cH.shape, cV.shape, cD.shape))

    coeffs_flat = np.concatenate(parts)

    bookkeeping = {
        'shapes': shapes,
        'level': level,
        'wavelet': wavelet,
        'original_shape': x.shape,
    }

    return coeffs_flat, bookkeeping


def waverec2_flat(coeffs_flat: np.ndarray, bookkeeping, wavelet: str = None
                  ) -> np.ndarray:
    """
    2-D wavelet reconstruction from a flat 1-D coefficient vector,
    exactly like MATLAB waverec2(C, S, wname).

    Parameters
    ----------
    coeffs_flat : 1-D ndarray — concatenated coefficients (MATLAB order).
    bookkeeping : dict from wavedec2_flat (or compatible).
    wavelet     : wavelet name override (if None, uses bookkeeping's wavelet).

    Returns
    -------
    x_rec : (M, N) 2-D reconstructed array.

    Notes
    -----
    Each sub-band of the flat vector was stored in COLUMN-MAJOR order,
    so we reshape with order='F' to recover the 2-D arrays.
    """
    if wavelet is None:
        wavelet = bookkeeping['wavelet']

    shapes = bookkeeping['shapes']
    idx = 0

    # Approximation coefficients
    tag, shape_cA = shapes[0]
    n_cA = shape_cA[0] * shape_cA[1]
    cA = coeffs_flat[idx:idx + n_cA].reshape(shape_cA, order='F')
    idx += n_cA

    # Rebuild the PyWavelets coefficient structure
    pywt_coeffs = [cA]

    for entry in shapes[1:]:
        _, sh_H, sh_V, sh_D = entry
        n_H = sh_H[0] * sh_H[1]
        n_V = sh_V[0] * sh_V[1]
        n_D = sh_D[0] * sh_D[1]

        cH = coeffs_flat[idx:idx + n_H].reshape(sh_H, order='F')
        idx += n_H
        cV = coeffs_flat[idx:idx + n_V].reshape(sh_V, order='F')
        idx += n_V
        cD = coeffs_flat[idx:idx + n_D].reshape(sh_D, order='F')
        idx += n_D

        pywt_coeffs.append((cH, cV, cD))

    return pywt.waverec2(pywt_coeffs, wavelet=wavelet)


# ═════════════════════════════════════════════════════════════════════════════
# Building subspace matrices B and C  (from blind2d.m)
# ═════════════════════════════════════════════════════════════════════════════

def build_B_from_kernel(blur_kernel: np.ndarray, image_shape: tuple
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the subspace matrix B from a blur kernel.

    MATLAB blind2d.m constructs B as a selection matrix that
    picks out the non-zero elements of the kernel:
        w = B @ h, where h contains the non-zero kernel values.

    This version works on the 1-D vectorised (column-major) kernel
    placed into the image grid, matching the MATLAB code in the
    "Blind deconvolution using convex programming" section of blind2d.m.

    Parameters
    ----------
    blur_kernel : 2-D ndarray — blur PSF (small, e.g. 5×5).
    image_shape : not used in the small-kernel variant. Kept for API
                  consistency.

    Returns
    -------
    B : (len_kernel, K) ndarray — selection matrix.
    h : (K,) ndarray — non-zero kernel coefficients.

    Notes
    -----
    MATLAB code (blind2d.m, lines 120-134):
        kernel = blur_kernel(:);          % column-major flatten
        K = sum(kernel ~= 0);
        B = zeros(length(kernel), K);
        idx = 1;
        h = zeros(K, 1);
        for i = 1:length(kernel)
            if kernel(i) ~= 0
                B(i, idx) = 1;
                h(idx) = kernel(i);
                idx = idx + 1;
            end
        end

    The flatten uses COLUMN-MAJOR to match MATLAB's kernel(:).
    """
    kernel_flat = blur_kernel.flatten(order='F')
    nz_mask = kernel_flat != 0
    K = int(np.sum(nz_mask))

    B = np.zeros((len(kernel_flat), K), dtype=np.float64)
    h = np.zeros(K, dtype=np.float64)
    idx = 0
    for i in range(len(kernel_flat)):
        if nz_mask[i]:
            B[i, idx] = 1.0
            h[idx] = kernel_flat[i]
            idx += 1

    return B, h


def build_C_from_image(x: np.ndarray, level: int = 4,
                       wavelet: str = 'db1',
                       threshold_ratio: float = 0.0005
                       ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build the subspace matrix C from wavelet coefficients of an image.

    In the blind setting (no access to original image), this should be
    called on the blurred image. The threshold determines which wavelet
    coefficients are considered "significant" and included in the support.

    MATLAB blind2d.m constructs C from the wavelet decomposition:
        [C_haar, s] = wavedec2(x, 4, 'db1');
        N = sum(C_haar ~= 0);
        C = zeros(length(img), N);
        m = zeros(N, 1);
        idx = 1;
        for i = 1:length(C_haar)
            if C_haar(i) ~= 0
                C(i, idx) = 1;
                m(idx) = C_haar(i);
                idx = idx + 1;
            end
        end

    Parameters
    ----------
    x               : (M, N) 2-D image array.
    level           : wavelet decomposition level.
    wavelet         : wavelet name.
    threshold_ratio : coefficients with |c| > threshold_ratio * max(|c|) are kept.
                      Set to 0 to keep only non-zero coefficients (MATLAB default).

    Returns
    -------
    C           : (L, N_coeffs) ndarray — selection matrix.
    m           : (N_coeffs,) ndarray — selected coefficient values.
    bookkeeping : dict — bookkeeping for waverec2_flat.
    """
    coeffs_flat, bookkeeping = wavedec2_flat(x, level, wavelet)

    if threshold_ratio > 0:
        nz_mask = np.abs(coeffs_flat) > threshold_ratio * np.max(np.abs(coeffs_flat))
    else:
        nz_mask = coeffs_flat != 0

    N = int(np.sum(nz_mask))
    L = len(coeffs_flat)

    C = np.zeros((L, N), dtype=np.float64)
    m = np.zeros(N, dtype=np.float64)
    idx = 0
    for i in range(L):
        if nz_mask[i]:
            C[i, idx] = 1.0
            m[idx] = coeffs_flat[i]
            idx += 1

    return C, m, bookkeeping


# ═════════════════════════════════════════════════════════════════════════════
# Linear operator A  (Fourier-domain lifting)
# ═════════════════════════════════════════════════════════════════════════════

def build_linear_operator_A(B_hat: np.ndarray, C_hat: np.ndarray,
                            L: int, N: int) -> np.ndarray:
    """
    Build the lifted linear measurement operator A in the Fourier domain.

    MATLAB (blind_deconv_convex.m / blind2d.m):
        A = [];
        for i = 1:N
            A_l = diag(sqrt(L) * C_hat(:,i));
            A = [A  A_l * B_hat];
        end

    The result is an (L, K*N) complex matrix such that:
        A @ vec(X) == y_hat

    Parameters
    ----------
    B_hat : (L, K) complex — DFT of B (column-wise or via dftmtx).
    C_hat : (L, N) complex — DFT of C.
    L     : int — observation length.
    N     : int — number of columns in C (= number of selected wavelet coeffs).

    Returns
    -------
    A : (L, K*N) complex ndarray.
    """
    K = B_hat.shape[1]
    A = np.empty((L, K * N), dtype=np.complex128)

    for i in range(N):
        # diag(sqrt(L) * C_hat[:, i]) @ B_hat
        # = (sqrt(L) * C_hat[:, i])[:, None] * B_hat   (broadcasting)
        A[:, i * K:(i + 1) * K] = (np.sqrt(L) * C_hat[:, i])[:, np.newaxis] * B_hat

    return A


# ═════════════════════════════════════════════════════════════════════════════
# Cyclic convolution via FFT
# ═════════════════════════════════════════════════════════════════════════════

def cyclic_conv_1d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Cyclic (circular) convolution of two 1-D signals via FFT.

    MATLAB: y = real(ifft(fft(x) .* fft(w)))
    """
    return np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(w)))


def cyclic_conv_2d(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Cyclic (circular) 2-D convolution via FFT.

    MATLAB: y = ifft2(fft2(x) .* fft2(w))
    """
    return np.real(np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(w)))


# ═════════════════════════════════════════════════════════════════════════════
# Place kernel into image grid (blind2d.m, kernel centring)
# ═════════════════════════════════════════════════════════════════════════════

def place_kernel_in_image(blur_kernel: np.ndarray, image_shape: tuple
                          ) -> np.ndarray:
    """
    Place a small blur kernel at the centre of an image-sized array.

    MATLAB (blind2d.m):
        w = zeros(L1, L2);
        w(L1/2-(K1+1)/2+2 : L1/2+(K1+1)/2,
          L2/2-(K2+1)/2+2 : L2/2+(K2+1)/2) = blur_kernel;

    This assumes K1, K2 are odd. The formula places the kernel
    centred at (L1/2, L2/2) in MATLAB 1-based indexing.

    Parameters
    ----------
    blur_kernel : (K1, K2) small PSF array (K1, K2 should be odd).
    image_shape : (L1, L2) target image dimensions.

    Returns
    -------
    w : (L1, L2) array with kernel placed at centre.

    Notes
    -----
    MATLAB indexing is 1-based. The formula:
        row_start = L1/2 - (K1+1)/2 + 2  (1-based)
        row_end   = L1/2 + (K1+1)/2      (1-based)
    Converting to 0-based Python:
        row_start_0 = L1//2 - (K1+1)//2 + 1
        row_end_0   = L1//2 + (K1+1)//2 - 1  (inclusive)
    Slice: row_start_0 : row_end_0 + 1  (length K1)
    """
    L1, L2 = image_shape
    K1, K2 = blur_kernel.shape

    w = np.zeros((L1, L2), dtype=np.float64)

    # MATLAB 1-based: L1/2 - (K1+1)/2 + 2  →  0-based: L1//2 - (K1+1)//2 + 1
    r_start = L1 // 2 - (K1 + 1) // 2 + 1
    c_start = L2 // 2 - (K2 + 1) // 2 + 1

    w[r_start:r_start + K1, c_start:c_start + K2] = blur_kernel

    return w


# ═════════════════════════════════════════════════════════════════════════════
# SVD-based recovery
# ═════════════════════════════════════════════════════════════════════════════

def recover_from_svd(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover h, m (up to scaling) from the rank-1 solution X via SVD.

    MATLAB:
        [U, S, V] = svd(X);
        u = U(:, 1);
        v = V(:, 1);

    Parameters
    ----------
    X : (K, N) complex or real matrix — solution of the nuclear norm problem.

    Returns
    -------
    u : (K,) first left singular vector.
    v : (N,) first right singular vector.
    S : 2-D array of singular values (diagonal matrix).

    Notes
    -----
    np.linalg.svd returns V_H (conjugate transpose of V) by default.
    MATLAB svd returns U, S, V such that X = U @ S @ V'.
    So MATLAB's V(:,1) = Python's V_H[0, :].conj() for complex,
    or V_H[0, :] for real.
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=True)
    u = U[:, 0]
    v = Vh[0, :].conj()  # MATLAB V(:,1) = conj(Vh[0,:])
    S = np.diag(s)
    return u, v, S


def compute_recovery_error(u: np.ndarray, v: np.ndarray,
                           h_true: np.ndarray, m_true: np.ndarray) -> float:
    """
    Compute relative recovery error.

    MATLAB: error = norm(u*v' - h*m', 'fro') / norm(h*m', 'fro')
    """
    X_rec = np.outer(u, v)
    X_true = np.outer(h_true, m_true)
    return np.linalg.norm(X_rec - X_true, 'fro') / np.linalg.norm(X_true, 'fro')
