"""
utils.py

Utility / building-block functions for the Bayesian Combination of Sparse
and Non-Sparse Priors Super-Resolution (BCSNSP-SR) algorithm.

Ported from MATLAB code by S. Villena, M. Vega, D. Babacan, J. Mateos,
R. Molina and A. K. Katsaggelos (2011).

References:
    [1] S. D. Babacan, R. Molina, A. K. Katsaggelos,
        "Bayesian Super Resolution Image Reconstruction using an l1 Prior",
        ISPA 2009 / Chapter in Bayesian Inference, 2011.
    [2] J. Salvador, S. Villena, R. Molina, A. K. Katsaggelos,
        "Bayesian Combination of Sparse and Non-Sparse Priors in
        Image Super Resolution", Digital Signal Processing, 2013.

MATLAB -> Python conversion notes:
    ──────────────────────────────────────────────────────────────────
    Sparse matrices:
        MATLAB sparse -> scipy.sparse (CSR/CSC).
        spdiags(d,0,n,n) -> scipy.sparse.diags(d, 0, shape=(n,n)).
        A'  -> A.T  (for real matrices) or A.conj().T.

    Indexing:
        MATLAB 1-based -> Python 0-based.

    fft2 / ifft2:
        numpy.fft.fft2 / ifft2 (identical semantics).

    circshift(A, [dy dx]):
        numpy.roll(A, dy, axis=0) then numpy.roll(A, dx, axis=1).

    meshgrid:
        MATLAB meshgrid(a,b) -> numpy.meshgrid(a,b) with indexing='xy'.
        Note: MATLAB's result has X varying along columns (axis=1),
        Y varying along rows (axis=0), same as numpy default.

    conv2(A,B):
        scipy.signal.convolve2d(A, B, mode='full').

    fspecial('average', n) -> np.ones((n,n)) / n**2
    fspecial('gaussian', n, sigma) -> see _fspecial_gaussian below

    imresize(img, factor, 'bicubic'):
        We use scipy.ndimage.zoom or skimage.transform.resize.
"""

import numpy as np
from numpy.fft import fft2, ifft2
import scipy.sparse as sp
from scipy.signal import convolve2d
from scipy.ndimage import zoom


# ═════════════════════════════════════════════════════════════════════════════
#  PSF / kernel helpers
# ═════════════════════════════════════════════════════════════════════════════

def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert a PSF kernel to an OTF (Optical Transfer Function) via FFT.

    1. Zero-pad *psf* into an array of *shape*.
    2. Circularly shift so that the centre of the PSF lands at index (0,0).
    3. Return fft2.

    Equivalent to MATLAB ``psf2otf(psf, shape)``.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)


def cent_nucleus2fft(kernel: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """
    Centre a convolution kernel and return its 2-D FFT.

    Equivalent to MATLAB ``cent_nucleus2fft.m``:
    - Flips the kernel (flipud + fliplr),
    - Zero-pads into (nr, nc),
    - Circularly shifts so centre is at (0,0),
    - Returns fft2.

    Used by ``restore_sar`` for frequency-domain deconvolution.
    """
    nrk, nck = kernel.shape

    # If kernel is larger than the image, crop it
    if nrk > nr or nck > nc:
        border = np.maximum(0, (np.array([nrk, nck]) - np.array([nr, nc]))) // 2
        interior = kernel[border[0]:nrk - border[0],
                          border[1]:nck - border[1]]
        fac = kernel.sum() / (interior.sum() + 1e-30)
        kernel = interior * fac
        nrk, nck = kernel.shape

    # Flip the kernel (MATLAB: flipdim(flipdim(spkernel,1),2))
    kernel_flipped = kernel[::-1, ::-1]

    h = np.zeros((nr, nc), dtype=np.float64)
    h[:nrk, :nck] = kernel_flipped

    # Circular shift: move centre to (0,0)
    shift_r = (nrk + 1) // 2 - nrk
    shift_c = (nck + 1) // 2 - nck
    h = np.roll(h, shift_r, axis=0)
    h = np.roll(h, shift_c, axis=1)
    return fft2(h)


def tcent_nucleus2fft(kernel: np.ndarray, nr: int, nc: int) -> np.ndarray:
    """
    Transposed variant of ``cent_nucleus2fft``.

    Equivalent to MATLAB ``Tcent_nucleus2fft.m``.
    Note: In the original code, this is identical to cent_nucleus2fft
    but without the kernel flip — effectively computing the conjugate.
    For real symmetric kernels the result is the same.

    We replicate the MATLAB code exactly: pad without flip, shift, fft2.
    """
    nrk, nck = kernel.shape

    if nrk > nr or nck > nc:
        border = np.maximum(0, (np.array([nrk, nck]) - np.array([nr, nc]))) // 2
        interior = kernel[border[0]:nrk - border[0],
                          border[1]:nck - border[1]]
        fac = kernel.sum() / (interior.sum() + 1e-30)
        kernel = interior * fac
        nrk, nck = kernel.shape

    h = np.zeros((nr, nc), dtype=np.float64)
    h[:nrk, :nck] = kernel

    shift_r = (nrk + 1) // 2 - nrk
    shift_c = (nck + 1) // 2 - nck
    h = np.roll(h, shift_r, axis=0)
    h = np.roll(h, shift_c, axis=1)
    return fft2(h)


# ═════════════════════════════════════════════════════════════════════════════
#  Circulant / circular convolution matrix  (circconvmatx2.m)
# ═════════════════════════════════════════════════════════════════════════════

def _circulant_matrix(c: np.ndarray) -> sp.csr_matrix:
    """
    Build a sparse circulant matrix whose first column is *c*.
    Equivalent to MATLAB ``circulant.m``.
    """
    m = len(c)
    rows = np.arange(m)
    indices = np.zeros((m, m), dtype=np.int64)
    for j in range(m):
        indices[:, j] = (rows - j) % m
    data = c[indices]
    return sp.csr_matrix(data)


def circconvmatx2(h: np.ndarray, M: int, N: int) -> sp.csr_matrix:
    """
    Build a sparse circular-convolution matrix of size (M*N, M*N)
    for a 2-D kernel *h* operating on an (M, N) image stored column-major.

    Equivalent to MATLAB ``circconvmatx2.m``.

    Parameters
    ----------
    h : (mh, mh) square convolution kernel (odd size).
    M : image height (number of rows).
    N : image width  (number of columns).

    Returns
    -------
    H : (M*N, M*N) sparse CSR matrix.
    """
    h = np.atleast_2d(h).astype(np.float64)
    if h.ndim == 0 or h.size == 1:
        # Scalar kernel = identity (no blur)
        return sp.eye(M * N, format='csr')

    mh, nh = h.shape
    if mh != nh:
        raise ValueError("Blur kernel must be square")

    centre = (nh + 1) // 2  # 1-based centre column index

    block_rows = []

    # --- right half of kernel columns (including centre) ---
    for i_col in range(centre - 1, nh):  # 0-based: centre-1 .. nh-1
        h0 = h[:, i_col].copy()
        h0 = h0[::-1]  # fliplr equivalent for a column vector

        row = np.zeros(M, dtype=np.float64)
        row[:len(h0)] = h0
        # Pad and circular-shift
        shift_amount = -(centre - 1)
        row = np.roll(row, shift_amount)

        if np.any(np.abs(row) > 0):
            H0 = _circulant_matrix(row)
        else:
            H0 = sp.csr_matrix((M, M))

        block_rows.append(H0)

    # --- left half of kernel columns (before centre, reverse order) ---
    for i_col in range(centre - 2, -1, -1):  # 0-based: centre-2 .. 0
        h0 = h[:, i_col].copy()
        h0 = h0[::-1]

        row = np.zeros(M, dtype=np.float64)
        row[:len(h0)] = h0
        shift_amount = -(centre - 1)
        row = np.roll(row, shift_amount)

        if np.any(np.abs(row) > 0):
            H0 = _circulant_matrix(row)
        else:
            H0 = sp.csr_matrix((M, M))

        block_rows.insert(0, H0)

    # Combine into first block-row: [H_0 H_1 ... H_{nh-1}  0 ... 0]
    # then pad with zero-blocks to width N
    n_zero_cols = N - nh
    if n_zero_cols > 0:
        block_rows.append(sp.csr_matrix((M, M * n_zero_cols)))

    first_block_row = sp.hstack(block_rows, format='csr')  # shape (M, M*N)

    # Circular shift to align centre column-block at position 0
    shift_blocks = -(centre - 1)
    if shift_blocks != 0:
        total_cols = M * N
        shift_cols = shift_blocks * M
        data_dense = first_block_row.toarray()
        data_dense = np.roll(data_dense, shift_cols, axis=1)
        first_block_row = sp.csr_matrix(data_dense)

    # Build full matrix by block-row circular shifts
    rows_list = [first_block_row]
    fbr_dense = first_block_row.toarray()
    for i in range(1, N):
        shifted = np.roll(fbr_dense, M * i, axis=1)
        rows_list.append(sp.csr_matrix(shifted))

    H = sp.vstack(rows_list, format='csr')
    return H


# ═════════════════════════════════════════════════════════════════════════════
#  Downsampling matrix  (dwnsmpl_matrix.m)
# ═════════════════════════════════════════════════════════════════════════════

def dwnsmpl_matrix(M: int, N: int, res: int) -> sp.csr_matrix:
    """
    Build a sparse downsampling matrix.

    Selects every *res*-th pixel along both axes of an (M, N) image
    (stored column-major as a vector of length M*N).

    Equivalent to MATLAB ``dwnsmpl_matrix.m``.

    Parameters
    ----------
    M   : image height.
    N   : image width.
    res : downsampling factor.

    Returns
    -------
    A : (m*n, M*N) sparse matrix, where m = M//res, n = N//res.
    """
    nopixels = M * N
    m = M // res
    n = N // res

    idx_grid = np.arange(nopixels).reshape(M, N, order='F')  # column-major
    dindices = idx_grid[::res, ::res].ravel(order='F')

    if len(dindices) != m * n:
        raise ValueError("dwnsmpl_matrix: size mismatch")

    row = np.arange(m * n)
    col = dindices
    data = np.ones(m * n, dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(m * n, nopixels))
    return A


# ═════════════════════════════════════════════════════════════════════════════
#  Shift matrix  (shift_matrix.m)
# ═════════════════════════════════════════════════════════════════════════════

def shift_matrix(dx: np.ndarray, dy: np.ndarray) -> sp.csr_matrix:
    """
    Build a sparse integer-shift matrix.

    Maps each pixel to a new position shifted by (dx, dy).
    Out-of-range indices are clamped.

    Equivalent to MATLAB ``shift_matrix.m``.

    Parameters
    ----------
    dx : (M, N) integer horizontal shift per pixel.
    dy : (M, N) integer vertical shift per pixel.

    Returns
    -------
    C : (M*N, M*N) sparse shift matrix.
    """
    M, N = dx.shape
    nopixels = M * N

    base = np.arange(1, nopixels + 1, dtype=np.int64)  # 1-based
    dindices = base + dx.ravel(order='F') * M + dy.ravel(order='F')

    # Clamp out-of-range
    dindices = np.clip(dindices, 1, nopixels)
    dindices -= 1  # to 0-based

    row = np.arange(nopixels)
    data = np.ones(nopixels, dtype=np.float64)
    C = sp.csr_matrix((data, (row, dindices)), shape=(nopixels, nopixels))
    return C


# ═════════════════════════════════════════════════════════════════════════════
#  Warp matrix — bilinear interpolation  (warp_matrix_bilinear.m)
# ═════════════════════════════════════════════════════════════════════════════

def warp_matrix_bilinear(sx: float, sy: float, theta: float,
                         M: int, N: int):
    """
    Build the sparse warp (motion) matrix C for a single LR frame.

    Applies an affine transformation:
        [cos θ  -sin θ  sx] [X]
        [sin θ   cos θ  sy] [Y]
                             [1]
    with bilinear interpolation to handle sub-pixel shifts.

    Equivalent to MATLAB ``warp_matrix_bilinear.m``.

    Parameters
    ----------
    sx, sy : sub-pixel translation.
    theta  : rotation angle (radians).
    M, N   : HR image dimensions.

    Returns
    -------
    C    : (M*N, M*N) sparse warp matrix.
    Lbl  : shift matrix (bottom-left corner).
    Lbr  : shift matrix (bottom-right corner).
    Ltl  : shift matrix (top-left corner).
    Ltr  : shift matrix (top-right corner).
    a    : (M*N,) fractional horizontal part.
    b    : (M*N,) fractional vertical part.
    """
    # Build coordinate grids (matching MATLAB meshgrid semantics)
    if M <= N:
        x_range = np.arange(-N // 2, N // 2)
        y_range = np.arange(-N // 2, N // 2)
        X, Y = np.meshgrid(x_range, y_range)
        if M < N:
            X = X[:M, :]
            lo = (N - M) // 2
            hi = lo + M  # should coincide with N - (N-M)//2 for even
            # MATLAB: Y(ceil((N-M)/2)+1 : N-floor((N-M)/2), :)
            lo_m = int(np.ceil((N - M) / 2))
            hi_m = N - int(np.floor((N - M) / 2))
            Y = Y[lo_m:hi_m, :]
    else:  # M > N
        x_range = np.arange(-M // 2, M // 2)
        y_range = np.arange(-M // 2, M // 2)
        X, Y = np.meshgrid(x_range, y_range)
        lo_n = int(np.ceil((M - N) / 2))
        hi_n = M - int(np.floor((M - N) / 2))
        X = X[:, lo_n:hi_n]
        Y = Y[:M, :N]

    # Column-major flattening to match MATLAB (:) operator
    Xf = X.ravel(order='F')
    Yf = Y.ravel(order='F')

    # Affine transform
    indices = np.vstack([Xf, Yf, np.ones(N * M)])
    S = np.array([[np.cos(theta), -np.sin(theta), sx],
                  [np.sin(theta),  np.cos(theta), sy]])
    new_indices = S @ indices

    dx_arr = new_indices[0, :] - indices[0, :]
    dy_arr = new_indices[1, :] - indices[1, :]

    a = dx_arr - np.floor(dx_arr)
    b = dy_arr - np.floor(dy_arr)

    dx_2d = dx_arr.reshape(M, N, order='F')
    dy_2d = dy_arr.reshape(M, N, order='F')

    # Prevent integer-shift degeneracy
    dx_2d = dx_2d + 1e-6
    dy_2d = dy_2d + 1e-6

    floor_dx = np.floor(dx_2d).astype(np.int64)
    ceil_dx = np.ceil(dx_2d).astype(np.int64)
    floor_dy = np.floor(dy_2d).astype(np.int64)
    ceil_dy = np.ceil(dy_2d).astype(np.int64)

    Lbl = shift_matrix(floor_dx, ceil_dy)
    Lbr = shift_matrix(ceil_dx, ceil_dy)
    Ltl = shift_matrix(floor_dx, floor_dy)
    Ltr = shift_matrix(ceil_dx, floor_dy)

    nopix = N * M
    if np.sum(np.abs(a)) == 0 and np.sum(np.abs(b)) == 0:
        C = shift_matrix(floor_dx, floor_dy)
    else:
        Da_inv = sp.diags(1.0 - a, 0, shape=(nopix, nopix))
        Da = sp.diags(a, 0, shape=(nopix, nopix))
        Db_inv = sp.diags(1.0 - b, 0, shape=(nopix, nopix))
        Db = sp.diags(b, 0, shape=(nopix, nopix))

        C = (Db @ Da_inv @ Lbl
             + Db @ Da @ Lbr
             + Db_inv @ Da_inv @ Ltl
             + Db_inv @ Da @ Ltr)

    return C, Lbl, Lbr, Ltl, Ltr, a, b


# ═════════════════════════════════════════════════════════════════════════════
#  unwrapLR  (unwrapLR.m)
# ═════════════════════════════════════════════════════════════════════════════

def unwrap_lr(y: np.ndarray, m: int, n: int, L: int):
    """
    Split a stacked observation vector y into individual LR frames.

    Parameters
    ----------
    y : (L*m*n,) stacked observation vector.
    m : LR image height.
    n : LR image width.
    L : number of LR frames.

    Returns
    -------
    ys   : list of L arrays, each (m, n).
    yvecs: list of L vectors, each (m*n,).
    """
    npix = m * n
    ys = []
    yvecs = []
    for k in range(L):
        vec = y[npix * k: npix * (k + 1)]
        yvecs.append(vec.copy())
        ys.append(vec.reshape(m, n, order='F'))
    return ys, yvecs


# ═════════════════════════════════════════════════════════════════════════════
#  SAR restoration for initialisation  (restoreSARmio.m)
# ═════════════════════════════════════════════════════════════════════════════

def restore_sar(image: np.ndarray, h: np.ndarray,
                tol: float = 1e-6, max_iter: int = 50):
    """
    Frequency-domain SAR (Simultaneous Auto-Regressive) deconvolution.

    Used to initialise hyperparameters (alpha, beta) for the SR solver.

    Equivalent to MATLAB ``restoreSARmio.m``.

    Parameters
    ----------
    image    : (M, N) observed image.
    h        : PSF kernel.
    tol      : convergence tolerance.
    max_iter : maximum number of iterations.

    Returns
    -------
    out   : (M, N) restored image.
    alpha : estimated prior precision.
    beta  : estimated noise precision.
    """
    image = image.astype(np.float64)
    g = fft2(image)
    M, N = image.shape
    npix = M * N

    H = cent_nucleus2fft(h, M, N)
    Ht = tcent_nucleus2fft(h, M, N)
    HtH = Ht * H

    # SAR Laplacian prior kernel
    priorn = np.array([[0, -0.25, 0],
                       [-0.25, 1, -0.25],
                       [0, -0.25, 0]], dtype=np.float64)
    priorn = convolve2d(priorn, priorn, mode='full')
    prior = cent_nucleus2fft(priorn, M, N)

    dif = g - H * g
    denom = np.sum(np.conj(dif) * dif).real
    beta0 = npix * npix / (denom + 1e-30)
    if beta0 > 1e6:
        beta0 = 1.0

    alpha0 = ((npix - 1.0) * npix /
              (np.sum(np.conj(g) * (prior * g)).real + 1e-30))

    Q = beta0 * HtH + alpha0 * prior
    f = beta0 * Ht * g / (Q + 1e-30)
    f0 = f.copy()

    alpha = alpha0
    beta = beta0

    for _ in range(max_iter):
        alpha_new = ((npix - 1.0) /
                     (np.sum(np.conj(f) * (prior * f)).real / npix
                      + np.sum(prior / (Q + 1e-30)).real + 1e-30)).real

        residual = g - H * f
        beta_new = (npix /
                    (np.sum(np.conj(residual) * residual).real / npix
                     + np.sum(HtH / (Q + 1e-30)).real + 1e-30)).real

        Q = beta_new * HtH + alpha_new * prior
        f = beta_new * Ht * g / (Q + 1e-30)

        t3 = (np.sum(np.conj(f - f0) * (f - f0)).real /
              (np.sum(np.conj(f0) * f0).real + 1e-30))
        f0 = f.copy()
        alpha = alpha_new
        beta = beta_new

        if t3 <= tol:
            break

    alpha = float(np.real(alpha))
    beta = float(np.real(beta))
    out = np.real(ifft2(f))
    return out, alpha, beta


# ═════════════════════════════════════════════════════════════════════════════
#  Modified Preconditioned Conjugate Gradient  (pcgmod.m)
# ═════════════════════════════════════════════════════════════════════════════

def pcg_solve(A, b: np.ndarray, tol: float = 1e-10,
              max_iter: int = 100, x0: np.ndarray = None,
              min_iter: int = 10):
    """
    Solve A x = b via (unpreconditioned) Conjugate Gradient.

    *A* can be either a sparse matrix or a callable (A_func(x) -> Ax).
    Includes a *min_iter* parameter to prevent premature stopping.

    Equivalent to MATLAB ``pcgmod.m``.

    Parameters
    ----------
    A        : sparse matrix or callable.
    b        : (n,) right-hand side.
    tol      : relative residual tolerance.
    max_iter : maximum CG iterations.
    x0       : initial guess.
    min_iter : minimum number of iterations before checking convergence.

    Returns
    -------
    x    : (n,) solution.
    flag : 0 = converged, 1 = max_iter reached.
    """
    n = b.shape[0]

    if callable(A):
        matvec = A
    else:
        matvec = lambda v: A @ v

    if x0 is None:
        x = np.zeros(n, dtype=np.float64)
    else:
        x = x0.copy()

    n2b = np.linalg.norm(b)
    if n2b == 0:
        return np.zeros(n), 0

    r = b - matvec(x)
    normr = np.linalg.norm(r)
    tolb = tol * n2b

    if normr <= tolb:
        return x, 0

    normr_min = normr
    x_min = x.copy()
    rho = 1.0
    flag = 1

    for it in range(1, max_iter + 1):
        z = r.copy()  # no preconditioner
        rho_new = np.dot(r, z)

        if it == 1:
            p = z.copy()
        else:
            beta_cg = rho_new / (rho + 1e-30)
            p = z + beta_cg * p

        q = matvec(p)
        pq = np.dot(p, q)

        if pq <= 0:
            break

        alpha_cg = rho_new / pq
        x = x + alpha_cg * p
        r = r - alpha_cg * q
        rho = rho_new

        normr = np.linalg.norm(r)
        if normr < normr_min:
            normr_min = normr
            x_min = x.copy()

        if normr <= tolb and it >= min_iter:
            flag = 0
            break

    if flag == 1:
        x = x_min

    return x, flag


# ═════════════════════════════════════════════════════════════════════════════
#  Average image for initialisation  (get_avg_img.m)
# ═════════════════════════════════════════════════════════════════════════════

def get_avg_img(y: np.ndarray, W) -> np.ndarray:
    """
    Compute the weighted average image from stacked observations.

    x_avg = diag(1 / sum(W, axis=0)) @ W^T @ y

    Equivalent to MATLAB ``get_avg_img.m``.
    """
    col_sums = np.array(W.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1e-30
    e = 1.0 / col_sums
    S = sp.diags(e, 0, shape=(W.shape[1], W.shape[1]))
    return S @ (W.T @ y)


# ═════════════════════════════════════════════════════════════════════════════
#  Atmospheric blur kernel  (fatmosfblur.m)
# ═════════════════════════════════════════════════════════════════════════════

def fatmosfblur(R: float, delta: float, nr: int, nc: int) -> np.ndarray:
    """
    Generate an atmospheric turbulence blur kernel.

    h(r) = (r^2 / R^2 + 1)^{-delta}

    Equivalent to MATLAB ``fatmosfblur.m``.
    """
    if nr % 2 == 0:
        nr += 1
    if nc % 2 == 0:
        nc += 1

    centre_r = nr // 2
    centre_c = nc // 2

    yr = centre_r - np.arange(nr)
    xc = centre_c - np.arange(nc)

    xs, ys = np.meshgrid(xc, yr)
    rs = (xs * xs + ys * ys).astype(np.float64)

    h = (rs / (R ** 2) + 1.0) ** (-delta)
    h /= h.sum()
    return h


# ═════════════════════════════════════════════════════════════════════════════
#  fspecial equivalents
# ═════════════════════════════════════════════════════════════════════════════

def fspecial_average(size: int) -> np.ndarray:
    """Uniform averaging kernel of given size."""
    return np.ones((size, size), dtype=np.float64) / (size * size)


def fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """Gaussian kernel of given size and standard deviation."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fspecial_disk(radius: float) -> np.ndarray:
    """Disk (pillbox) averaging kernel."""
    size = int(2 * radius + 1)
    ax = np.arange(size) - radius
    xx, yy = np.meshgrid(ax, ax)
    mask = (xx ** 2 + yy ** 2) <= radius ** 2
    kernel = mask.astype(np.float64)
    kernel /= kernel.sum()
    return kernel


# ═════════════════════════════════════════════════════════════════════════════
#  Image resize  (wrapper around scipy.ndimage.zoom)
# ═════════════════════════════════════════════════════════════════════════════

def imresize(image: np.ndarray, factor, order: int = 3) -> np.ndarray:
    """
    Resize an image by a scale factor using spline interpolation.

    Parameters
    ----------
    image  : 2-D array.
    factor : int or float zoom factor (e.g. 2 for 2x upscale).
    order  : spline order (3 = bicubic).

    Returns
    -------
    Resized image.
    """
    return zoom(image, float(factor), order=order)


# ═════════════════════════════════════════════════════════════════════════════
#  Differential-operator kernels
# ═════════════════════════════════════════════════════════════════════════════

def get_diff_kernels():
    """
    Return the first-order finite-difference kernels Dx and Dy
    used in the TV prior.

    dx = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
    dy = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
    """
    dx = np.array([[0, 0, 0],
                   [-1, 1, 0],
                   [0, 0, 0]], dtype=np.float64)
    dy = np.array([[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=np.float64)
    return dx, dy


def get_sar_kernel():
    """
    Return the SAR (Simultaneous Auto-Regressive / Laplacian) prior kernel.

    hsar = [[0, -0.25, 0], [-0.25, 1, -0.25], [0, -0.25, 0]]
    """
    return np.array([[0, -0.25, 0],
                     [-0.25, 1, -0.25],
                     [0, -0.25, 0]], dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
#  Coordinate grid for registration  (used in LKvar and solver loops)
# ═════════════════════════════════════════════════════════════════════════════

def build_coord_grid(M: int, N: int):
    """
    Build the (X, Y) coordinate grid used throughout the SR code.

    Handles non-square images exactly as the MATLAB code does.

    Returns
    -------
    X : (M*N,) flattened X-coordinates (column-major).
    Y : (M*N,) flattened Y-coordinates (column-major).
    """
    if M <= N:
        x_range = np.arange(-N // 2, N // 2)
        y_range = np.arange(-N // 2, N // 2)
        X, Y = np.meshgrid(x_range, y_range)
        if M < N:
            X = X[:M, :]
            lo = int(np.ceil((N - M) / 2))
            hi = N - int(np.floor((N - M) / 2))
            Y = Y[lo:hi, :]
    else:
        x_range = np.arange(-M // 2, M // 2)
        y_range = np.arange(-M // 2, M // 2)
        X, Y = np.meshgrid(x_range, y_range)
        lo = int(np.ceil((M - N) / 2))
        hi = M - int(np.floor((M - N) / 2))
        X = X[:, lo:hi]
        Y = Y[:M, :N]

    return X.ravel(order='F'), Y.ravel(order='F')
