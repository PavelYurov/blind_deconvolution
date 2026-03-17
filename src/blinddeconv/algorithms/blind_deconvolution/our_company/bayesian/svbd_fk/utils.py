"""
Mathematical utilities for SVBD-FK.

Sonogashira et al. (2017), "Shift-Variant Blind Deconvolution Using a
Field of Kernels", IEICE Trans. Inf. & Syst., E100-D(9), 1971-1983.
DOI: 10.1587/transinf.2016PCP0013

Contains:
    - FFT-based convolution / correlation (periodic boundary)
    - Finite-difference operators D_h, D_v and their adjoints
    - Discrete Laplacian via FFT spectrum
    - Conjugate-gradient solver for SPD linear systems
    - Weighted Gram matrix and RHS construction for kernel update
    - TV variational bound weights (Babacan et al. [31])
    - Edge tapering for boundary artifact suppression

References:
    [31] Babacan et al. (2009) — VB blind deconvolution with TV prior,
         IEEE Trans. Image Process., 18(1), 12-26.
    [54] Petersen & Pedersen (2012) — "The Matrix Cookbook".
"""

import numpy as np
from typing import Tuple, Callable, Optional, List

# Global numerical stability constant
EPSILON = 1e-12

# ========================================================================
#  FFT convolution / correlation
# ========================================================================

def psf_to_otf(kernel: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Pad PSF to *shape* with circular centering and return its rfft2.

    The kernel center (kh//2, kw//2) is placed at the origin (0,0) via
    circular shift so that FFT-based convolution matches spatial
    convolution with periodic boundaries.
    """
    kh, kw = kernel.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:kh, :kw] = kernel
    # Circular shift: center of kernel → origin
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.rfft2(padded)


def fft_conv2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D convolution via FFT with periodic boundary conditions.

    Implements  K_l x  from Eq. (3):  y = sum_l diag(w_l) K_l x + n.
    """
    F_k = psf_to_otf(kernel, image.shape)
    return np.fft.irfft2(np.fft.rfft2(image) * F_k, s=image.shape)


def fft_corr2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    2-D correlation via FFT (adjoint of convolution).

    Implements  K_l^T u  used in  A^T u = sum_l K_l^T (w_l .* u).
    """
    F_k = psf_to_otf(kernel, image.shape)
    return np.fft.irfft2(np.fft.rfft2(image) * np.conj(F_k), s=image.shape)

# ========================================================================
#  Finite-difference operators  D_h, D_v  and their adjoints
# ========================================================================

def forward_diff_h(x: np.ndarray) -> np.ndarray:
    """Horizontal forward difference:  [D_h x]_{i,j} = x_{i,j+1} - x_{i,j}."""
    return np.roll(x, -1, axis=1) - x


def forward_diff_v(x: np.ndarray) -> np.ndarray:
    """Vertical forward difference:  [D_v x]_{i,j} = x_{i+1,j} - x_{i,j}."""
    return np.roll(x, -1, axis=0) - x


def adjoint_diff_h(z: np.ndarray) -> np.ndarray:
    """Adjoint of forward_diff_h:  D_h^T z."""
    return np.roll(z, 1, axis=1) - z


def adjoint_diff_v(z: np.ndarray) -> np.ndarray:
    """Adjoint of forward_diff_v:  D_v^T z."""
    return np.roll(z, 1, axis=0) - z

# ========================================================================
#  Laplacian in Fourier domain:  L = D_h^T D_h + D_v^T D_v
# ========================================================================

def laplacian_fft_spectrum(shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute  |F_{D_h}|^2 + |F_{D_v}|^2  in rfft2 layout.

    This is the eigenvalue spectrum of the discrete Laplacian L with
    periodic boundary conditions.  Used to apply  L v  via FFT as
    ifft2(spectrum * fft2(v)).

    Ref: Eq. (7) — weight prior  p(w_l | eta_l) ~ N(0, (eta_l L)^{-1}).
    """
    H, W = shape

    # D_h impulse response: [-1, 1] at (0,0) and (0,1)
    dh = np.zeros(shape, dtype=np.float64)
    dh[0, 0] = -1.0
    dh[0, 1 % W] = 1.0
    F_dh = np.fft.rfft2(dh)

    # D_v impulse response: [-1, 1] at (0,0) and (1,0)
    dv = np.zeros(shape, dtype=np.float64)
    dv[0, 0] = -1.0
    dv[1 % H, 0] = 1.0
    F_dv = np.fft.rfft2(dv)

    return np.abs(F_dh) ** 2 + np.abs(F_dv) ** 2


def apply_laplacian_fft(
    v: np.ndarray, F_L: np.ndarray
) -> np.ndarray:
    """
    Apply discrete Laplacian  L v  via precomputed spectrum F_L.

    Used inside CG for weight update (Eq. 13) where the precision
    matrix contains  eta_l * L.
    """
    return np.fft.irfft2(np.fft.rfft2(v) * F_L, s=v.shape)

# ========================================================================
#  Conjugate-gradient solver
# ========================================================================

def conjugate_gradient(
    A_func: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Standard CG for symmetric positive-definite operator A_func.

    Solves  A x = b  where A is provided implicitly as a function
    computing  A @ v  for any vector/image v.

    Used for:
        - Image update (Eq. 9):  Sigma_x^{-1} mu_x = beta * A_bar^T y
        - Weight update (Eq. 13): Sigma_{w_l}^{-1} mu_{w_l} = beta * (f_l .* r_{w_l})

    Parameters
    ----------
    A_func : callable
        Matrix-vector product  v -> A v.
    b : np.ndarray
        Right-hand side.
    x0 : np.ndarray or None
        Initial guess (warm start).
    max_iter : int
        Maximum number of CG iterations.
    tol : float
        Relative residual tolerance for early stopping.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A_func(x)
    p = r.copy()
    rs_old = np.sum(r * r)
    b_norm = np.sqrt(np.sum(b * b)) + EPSILON

    for _ in range(max_iter):
        Ap = A_func(p)
        pAp = np.sum(p * Ap)
        if pAp <= 0:
            # Operator is not positive-definite in this direction; stop.
            break
        alpha_cg = rs_old / (pAp + EPSILON)
        x += alpha_cg * p
        r -= alpha_cg * Ap
        rs_new = np.sum(r * r)
        if np.sqrt(rs_new) / b_norm < tol:
            break
        beta_cg = rs_new / (rs_old + EPSILON)
        p = r + beta_cg * p
        rs_old = rs_new

    return x

# ========================================================================
#  Gram matrix  X_mu^T diag(d) X_mu   for kernel update  (K^2 x K^2)
# ========================================================================

def _shift_offsets(kernel_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Return list of (dy, dx) shifts for each kernel element index
    j = row * kw + col.

    Convolution convention:  pixel i is affected by kernel element j
    through image value  x(i - delta_j), where delta_j is returned here.

    For a kernel of size (kh, kw) centered at (kh//2, kw//2), the shift
    for element (r, c) is  (kh//2 - r, kw//2 - c).
    """
    kh, kw = kernel_shape
    pad_h, pad_w = kh // 2, kw // 2
    offsets = []
    for r in range(kh):
        for c in range(kw):
            offsets.append((pad_h - r, pad_w - c))
    return offsets


def build_weighted_gram(
    mu_x: np.ndarray,
    d: np.ndarray,
    kernel_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Build the K^2 x K^2 weighted Gram matrix for the kernel update.

    G_{j1, j2} = sum_i  d(i) * mu_x(i - delta_{j1}) * mu_x(i - delta_{j2})

    where d(i) = mu_w_l(i)^2 + sigma_w_l(i) is the second moment of the
    weight map at pixel i.

    Ref: Eq. (11) — precision matrix of q(k_l).

    Parameters
    ----------
    mu_x : (H, W)   image mean
    d    : (H, W)   weight second moment (mu_w^2 + sigma_w)
    kernel_shape : (kh, kw)

    Returns
    -------
    G : (K^2, K^2)  symmetric positive semi-definite matrix
    """
    kh, kw = kernel_shape
    K2 = kh * kw
    offsets = _shift_offsets(kernel_shape)

    # Pre-compute shifted images  mu_x(. - delta_j)  for each j
    shifted = [
        np.roll(mu_x, (dy, dx), axis=(0, 1)) for dy, dx in offsets
    ]

    G = np.empty((K2, K2), dtype=np.float64)
    for j1 in range(K2):
        s1 = shifted[j1]
        for j2 in range(j1, K2):
            val = np.sum(d * s1 * shifted[j2])
            G[j1, j2] = val
            G[j2, j1] = val      # symmetry

    return G


def build_weighted_rhs(
    mu_x: np.ndarray,
    w: np.ndarray,
    residual: np.ndarray,
    kernel_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Build the K^2-vector right-hand side for the kernel update.

    b_j = sum_i  w(i) * mu_x(i - delta_j) * residual(i)

    Ref: Eq. (12) — mean of q(k_l).

    Parameters
    ----------
    mu_x     : (H, W)  image mean
    w        : (H, W)  weight map mean  mu_w_l
    residual : (H, W)  r_l = y - sum_{m != l} diag(mu_w_m) K_m mu_x
    kernel_shape : (kh, kw)

    Returns
    -------
    b : (K^2,)
    """
    kh, kw = kernel_shape
    K2 = kh * kw
    offsets = _shift_offsets(kernel_shape)
    wr = w * residual                       # element-wise  w(i) * r(i)

    b = np.empty(K2, dtype=np.float64)
    for j in range(K2):
        dy, dx = offsets[j]
        shifted = np.roll(mu_x, (dy, dx), axis=(0, 1))
        b[j] = np.sum(wr * shifted)

    return b

# ========================================================================
#  TV weights  (Babacan [31])
# ========================================================================

def compute_tv_weights(
    mu_x: np.ndarray, eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute diagonal weights for the quadratic lower bound of the TV prior.

    Lambda_h(i) = 1 / (|D_h mu_x(i)| + eps)
    Lambda_v(i) = 1 / (|D_v mu_x(i)| + eps)

    Ref: Babacan et al. [31], Eq. (10); used in Sonogashira Eq. (5)
         to make the TV prior locally quadratic.
    """
    dh = forward_diff_h(mu_x)
    dv = forward_diff_v(mu_x)
    lam_h = 1.0 / (np.abs(dh) + eps)
    lam_v = 1.0 / (np.abs(dv) + eps)
    return lam_h, lam_v

# ========================================================================
#  Edge tapering
# ========================================================================

def edgetaper(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Hanning-window edge taper to reduce boundary ringing artefacts
    caused by the periodic-boundary assumption of FFT-based convolution.

    Pixels within *kernel_size // 2* of the border are smoothly blended
    toward the image mean.
    """
    h, w = img.shape
    alpha = kernel_size // 2
    if alpha <= 0:
        return img.copy()

    # 1-D Hanning half-windows
    wx = np.ones(w, dtype=np.float64)
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(alpha) / alpha))
    wx[:alpha] = taper
    wx[-alpha:] = taper[::-1]

    wy = np.ones(h, dtype=np.float64)
    taper = 0.5 * (1.0 - np.cos(np.pi * np.arange(alpha) / alpha))
    wy[:alpha] = taper
    wy[-alpha:] = taper[::-1]

    window = np.outer(wy, wx)
    mean_val = np.mean(img)
    return img * window + mean_val * (1.0 - window)
