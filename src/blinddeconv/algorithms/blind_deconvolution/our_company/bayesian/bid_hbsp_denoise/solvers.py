"""
Solver functions for the BID-HBSP algorithm.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg
from typing import Tuple

from .utils import (
    psf2otf,
    otf2psf,
    precompute_gradient_operators,
    forward_diff_x,
    forward_diff_y,
    adjoint_diff_x,
    adjoint_diff_y,
    compute_hs_weights,
    project_kernel,
    threshold_kernel,
    EPSILON,
    edgetaper,
)

def solve_image_cg(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    gamma_x: np.ndarray,
    gamma_y: np.ndarray,
    max_cg_iter: int = 50,
    cg_tol: float = 1e-6,
    jacobi_mode: str = "scalar",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image-space CG solver:
        (β H^T H + D_x^T Γ_x D_x + D_y^T Γ_y D_y) x = β H^T y

    Parameters
    ----------
    jacobi_mode : 'scalar' | 'perpixel'
        'scalar'  — σ²(i) ≈ 1 / (β ||h||² + reg_i + ε)  (fast)
        'perpixel' — σ²(i) ≈ 1 / (β * ifft2(|F_h|²)(i) + reg_i + ε)
    """
    H, W = y.shape
    N = H * W

    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_y = fft2(y)

    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        Av += adjoint_diff_x(gamma_x * forward_diff_x(v))
        Av += adjoint_diff_y(gamma_y * forward_diff_y(v))
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    rhs = beta * np.real(ifft2(F_h_conj * F_y))

    x_flat, _info = cg(A_op, rhs.ravel(), x0=x_init.ravel(),
                        maxiter=max_cg_iter, atol=cg_tol)
    x_out = x_flat.reshape((H, W))

    # Jacobi variance approximation: σ²(i) ≈ 1 / diag(A)(i)
    reg_strength = (gamma_x + np.roll(gamma_x, 1, axis=1) +
                    gamma_y + np.roll(gamma_y, 1, axis=0))

    if jacobi_mode == "perpixel":
        # Per-pixel: diag(H^T H) = ifft2(|F_h|²)  (exact diagonal)
        diag_hth = np.real(ifft2(F_h_sq))
        sigma_sq = 1.0 / (beta * diag_hth + reg_strength + EPSILON)
    else:
        # Scalar: diag(H^T H) ≈ ||h||²
        h_energy = np.sum(h ** 2)
        sigma_sq = 1.0 / (beta * h_energy + reg_strength + EPSILON)

    return np.maximum(x_out, 0.0), sigma_sq


# ═══════════════════════════════════════════════════════════════
#  Filter-space solvers  (Castro-Macías et al. 2024, Sec. IV)
# ═══════════════════════════════════════════════════════════════

def solve_filtered_image_cg(
    y_n: np.ndarray,
    h: np.ndarray,
    x_n_init: np.ndarray,
    beta_n: float,
    theta_n: np.ndarray,
    max_cg_iter: int = 50,
    cg_tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve for one filtered image in the VB filter-space formulation.

    System (Eq. 17-18 of [1]):

    .. math::
        (\beta_n\,H^\top H + \operatorname{diag}(\theta_n))\,m_{x_n}
            = \beta_n\,H^\top y_n

    Parameters
    ----------
    y_n : (H, W) — pseudo-observation :math:`y_n = F_n\,y`.
    h : (kh, kw) — current blur kernel estimate.
    x_n_init : (H, W) — warm-start for CG.
    beta_n : float — noise precision for this filter channel.
    theta_n : (H, W) — diagonal HS weights :math:`E[\omega_n^i]`.
    max_cg_iter, cg_tol : CG stopping criteria.

    Returns
    -------
    x_n : (H, W) — posterior mean :math:`m_{x_n}`.
    sigma_sq_n : (H, W) — Jacobi approximation of
        :math:`\Sigma_{x_n}(i,i)`.
    """
    H, W = y_n.shape
    N = H * W

    F_h = psf2otf(h, (H, W))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    F_yn = fft2(y_n)

    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        # β_n H^T H v  (likelihood term, Fourier domain)
        Av = beta_n * np.real(ifft2(F_h_sq * fft2(v)))
        # diag(θ_n) v  (prior term)
        Av += theta_n * v
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    rhs = beta_n * np.real(ifft2(F_h_conj * F_yn))

    x_flat, _ = cg(A_op, rhs.ravel(), x0=x_n_init.ravel(),
                    maxiter=max_cg_iter, atol=cg_tol)
    x_n = x_flat.reshape((H, W))

    # Jacobi variance approximation: Σ(i,i) ≈ 1 / A(i,i)
    h_energy = np.sum(h ** 2)
    sigma_sq_n = 1.0 / (beta_n * h_energy + theta_n + EPSILON)

    return x_n, sigma_sq_n


def solve_kernel_fourier_filterspace(
    filtered_data: list,
    kernel_shape: Tuple[int, int],
    beta: float,
    lambda_h: float,
    do_threshold: bool = True,
) -> np.ndarray:
    r"""Estimate the blur kernel from *N* filtered-image pairs.

    Uses a Wiener-type closed-form in the Fourier domain with a
    VB uncertainty correction term.  Boundary rows / columns of each
    filtered image are zeroed to suppress wrap-around artefacts.

    .. note::
        This is a *simplified* version.  The full paper [1] solves a
        quadratic programme on the simplex :math:`\Delta^K` with the
        covariance-corrected matrix :math:`C_h` (Eq. 20-22).

    Parameters
    ----------
    filtered_data : list of (y_n, x_n, sigma_sq_n) tuples
        Each tuple contains the pseudo-observation, the posterior mean
        of the filtered image, and its diagonal covariance.
    kernel_shape : (kh, kw)
    beta : noise precision (original image space).
    lambda_h : kernel regularisation weight.
    do_threshold : zero-out small kernel elements and re-normalise.
    """
    H, W = filtered_data[0][0].shape

    numerator = np.zeros((H, W), dtype=np.complex128)
    denominator = np.zeros((H, W), dtype=np.float64)
    uncertainty_total = 0.0

    for y_n, x_n, sigma_sq_n in filtered_data:
        # Boundary masking (last row & col wrap around for finite diffs)
        ym = y_n.copy();  ym[:, -1] = 0.0;  ym[-1, :] = 0.0
        xm = x_n.copy();  xm[:, -1] = 0.0;  xm[-1, :] = 0.0

        F_yn = fft2(ym)
        F_xn = fft2(xm)

        numerator += F_yn * np.conj(F_xn)
        denominator += np.abs(F_xn) ** 2

        # VB covariance correction  (rough: Tr(X^T X Σ) ≈ N·mean(σ²))
        uncertainty_total += np.mean(sigma_sq_n)

    denominator += H * W * uncertainty_total + (lambda_h / beta) + EPSILON

    F_h = numerator / denominator
    h = otf2psf(F_h, kernel_shape)

    if do_threshold:
        h = threshold_kernel(h, ratio=0.05)
    else:
        h = project_kernel(h)

    return h


def solve_kernel_qp_filterspace(
    filtered_data: list,
    kernel_shape: Tuple[int, int],
    lambda_h: float = 0.0,
    do_threshold: bool = True,
    threshold_ratio: float = 0.05,
) -> np.ndarray:
    r"""Estimate the blur kernel by solving a QP on the probability simplex.

    Implements Eq. (20)–(22) of Castro-Macías et al. (2024):

    .. math::
        \hat{h} = \arg\min_{h \in \Delta^K}
                  \bigl\{h^\top C_h\,h \;-\; 2\,h^\top b_h\bigr\}

    :math:`C_h` is the autocorrelation of the VB posterior means
    :math:`m_{x_n}` (summed over N filters) with a diagonal covariance
    correction term.  :math:`b_h` is the cross-correlation of
    :math:`m_{x_n}` with the pseudo-observations :math:`y_n`.

    Both are computed efficiently via FFT; the final system
    :math:`C_h\,h = b_h` is solved by ``np.linalg.solve`` and the
    result is projected onto the simplex
    :math:`\Delta^K = \{h \ge 0,\;\sum h = 1\}`.

    Parameters
    ----------
    filtered_data : list of (y_n, m_xn, sigma_sq_n) tuples
        N pseudo-observations with their VB posterior means and
        diagonal covariances.
    kernel_shape : (kh, kw)
    lambda_h : float
        Optional Tikhonov regularisation added to diagonal of C_h.
    do_threshold : bool
        Zero-out kernel elements below ``threshold_ratio * max(h)``
        before normalising.
    threshold_ratio : float
        Fraction of peak below which kernel elements are zeroed.
    """
    kh, kw = kernel_shape
    K = kh * kw
    H_img, W_img = filtered_data[0][0].shape

    # ── Kernel coordinate arrays ─────────────────────────────────
    idx = np.arange(K)
    a_coords = idx // kw                          # row in kernel
    b_coords = idx % kw                           # col in kernel

    # C_h[i,j] = R_xx[(a_i − a_j) mod H, (b_i − b_j) mod W]
    da_mat = (a_coords[:, None] - a_coords[None, :]) % H_img    # (K, K)
    db_mat = (b_coords[:, None] - b_coords[None, :]) % W_img    # (K, K)

    # b_h offsets relative to kernel centre (kh//2, kw//2)
    a_off = (a_coords - kh // 2) % H_img                        # (K,)
    b_off = (b_coords - kw // 2) % W_img                        # (K,)

    C_h = np.zeros((K, K), dtype=np.float64)
    b_h = np.zeros(K, dtype=np.float64)

    for y_n, m_xn, sigma_sq_n in filtered_data:
        F_xn = fft2(m_xn)
        F_yn = fft2(y_n)

        # Autocorrelation:  R_xx[d1,d2] = Σ_{r,c} x(r+d1, c+d2) x(r, c)
        #   Symmetric, so direction doesn't matter.
        R_xx = np.real(ifft2(np.abs(F_xn) ** 2))

        # Cross-correlation for the CONVOLUTION convention
        #   (psf2otf uses convolution: y = ifft(F_h · F_x)).
        #   b_h(a,b) = Σ_{r,c} x(r,c) · y(r + offset, c + offset)
        #            = ifft2(conj(F_x) · F_y)[offset]
        R_yx = np.real(ifft2(np.conj(F_xn) * F_yn))

        # ── C_h: autocorrelation part  (Eq. 21, first term) ──────
        C_h += R_xx[da_mat, db_mat]

        # ── C_h: VB covariance correction  (Eq. 21, second term) ─
        #   With Jacobi (diagonal) Σ_{x_n}: nonzero only for i == j.
        #   Σ_l Σ_{x_n}(i+l, i+l) = sum(σ²_n) (periodic boundaries).
        C_h[np.diag_indices(K)] += float(np.sum(sigma_sq_n))

        # ── b_h  (Eq. 22) ────────────────────────────────────────
        b_h += R_yx[a_off, b_off]

    # Optional Tikhonov regularisation
    if lambda_h > 0.0:
        C_h[np.diag_indices(K)] += lambda_h

    # Tiny ridge for numerical stability
    C_h[np.diag_indices(K)] += 1e-10

    # ── Solve  C_h h = b_h ───────────────────────────────────────
    try:
        h_flat = np.linalg.solve(C_h, b_h)
    except np.linalg.LinAlgError:
        h_flat, _, _, _ = np.linalg.lstsq(C_h, b_h, rcond=None)

    # ── Project onto simplex Δ^K ─────────────────────────────────
    h_flat = np.maximum(h_flat, 0.0)

    if do_threshold:
        peak = np.max(h_flat)
        if peak > 0:
            h_flat[h_flat < threshold_ratio * peak] = 0.0

    h_sum = h_flat.sum()
    if h_sum > EPSILON:
        h_flat /= h_sum
    else:
        h_flat = np.ones(K, dtype=np.float64) / K

    return h_flat.reshape(kh, kw)


# ═══════════════════════════════════════════════════════════════
#  Legacy solvers  (kept for reference / backward compatibility)
# ═══════════════════════════════════════════════════════════════

def solve_kernel_fourier(
    y: np.ndarray,
    x: np.ndarray,
    sigma_sq: np.ndarray,
    kernel_shape: Tuple[int, int],
    beta: float,
    lambda_h: float,
    do_threshold: bool = True,
) -> np.ndarray:
    """
    Estimates kernel in Gradient Space with Covariance correction and Boundary Masking.
    Ref: Castro-Macías (2024) Section IV.B & Eq (21).
    """
    H, W = y.shape
    
    # 1. Gradient domain
    dy_x = forward_diff_x(y)
    dy_y = forward_diff_y(y)
    
    dx_x = forward_diff_x(x)
    dx_y = forward_diff_y(x)
    
    # Boundary Masking
    dy_x[:, -1] = 0.0
    dx_x[:, -1] = 0.0
    dy_y[-1, :] = 0.0
    dx_y[-1, :] = 0.0
    F_dy_x = fft2(dy_x)
    F_dy_y = fft2(dy_y)
    F_dx_x = fft2(dx_x)
    F_dx_y = fft2(dx_y)
    
    # 2. Wiener filter
    numerator = (F_dy_x * np.conj(F_dx_x)) + (F_dy_y * np.conj(F_dx_y))
    
    # Autocorrelation X
    denominator = (np.abs(F_dx_x)**2) + (np.abs(F_dx_y)**2)
    
    # VB Correction term (Sigma)
    # Trace(T^T T Sigma) approx N * mean(Sigma_grad)
    sigma_grad_mean = 2.0 * np.mean(sigma_sq)
    uncertainty_term = H * W * sigma_grad_mean
    
    # regularization
    denominator += uncertainty_term + (lambda_h / beta) + EPSILON
    
    F_h = numerator / denominator
    
    # 3. Spatial domain
    h = otf2psf(F_h, kernel_shape)
    
    if do_threshold:
        # thresholding
        h = threshold_kernel(h, ratio=0.05) #0.05 0.1
    else:
        h = project_kernel(h)
        
    return h


def update_noise_precision(y: np.ndarray, h: np.ndarray, x: np.ndarray, beta_prev: float, damping: float = 0.5) -> float:
    H, W = y.shape
    N = float(H * W)
    F_h = psf2otf(h, (H, W))
    residual = y - np.real(ifft2(F_h * fft2(x)))
    rss = float(np.sum(residual ** 2))
    beta_new = N / (rss + EPSILON)
    beta = (1.0 - damping) * beta_prev + damping * beta_new
    beta = float(np.clip(beta, 1.0, 1e8))
    return beta


def update_hs_weights(x: np.ndarray, sigma_sq: np.ndarray, b: float) -> Tuple:
    """
    Update HS weights using the variational expectation E[w].
    Needs sigma_sq (variance) as per Eq. 26 in the paper.
    """
    dx = forward_diff_x(x)
    dy = forward_diff_y(x)
    from .utils import compute_hs_weights
    return compute_hs_weights(dx, dy, sigma_sq, b)


def final_deconvolution(y: np.ndarray, h: np.ndarray, beta: float, lambda_reg: float) -> np.ndarray:
    """
    Non-blind deconvolution using IRLS with Padding to remove boundary artifacts.
    Minimizes: 0.5 * ||y - h*x||^2 + lambda * ||grad x||_p^p
    """
    # Padding
    kh, kw = h.shape
    pad_h = kh
    pad_w = kw
    y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    Hp, Wp = y_padded.shape

    x = y_padded.copy()
    
    # IRLS
    p = 0.8
    irls_iters = 15
    
    # Recalculating kernel OTF
    F_h = psf2otf(h, (Hp, Wp))
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2
    
    # (H^T * y)
    F_y = fft2(y_padded)
    rhs_base = beta * np.real(ifft2(F_h_conj * F_y))
    
    for i in range(irls_iters):
        dx = forward_diff_x(x)
        dy = forward_diff_y(x)
        
        # Weights IRLS: w = p * (|grad|^2 + eps)^(p/2 - 1)
        # power = (0.8 - 2) / 2 = -0.6
        power = (p - 2.0) / 2.0
        
        grad_sq_x = dx**2 + 1e-8
        grad_sq_y = dy**2 + 1e-8
        
        wx = p * (grad_sq_x ** power)
        wy = p * (grad_sq_y ** power)
        
        wx = np.clip(wx, 0.0, 1e4)
        wy = np.clip(wy, 0.0, 1e4)
        
        # regularization
        wx *= lambda_reg
        wy *= lambda_reg
        
        x = _solve_image_irls_step(rhs_base, F_h_sq, wx, wy, x, beta, cg_iter=15)
        
        x = np.clip(x, 0.0, 1.0)
    
    x_final = x[pad_h:-pad_h, pad_w:-pad_w]
        
    return x_final

def _solve_image_irls_step(
    rhs: np.ndarray,
    F_h_sq: np.ndarray,
    wx: np.ndarray,
    wy: np.ndarray,
    x_init: np.ndarray,
    beta: float,
    cg_iter: int = 20
) -> np.ndarray:
    """
    Solves one step of IRLS using Conjugate Gradient.
    System: (beta * H^T H + Dx^T Wx Dx + Dy^T Wy Dy) x = rhs
    """
    H, W = x_init.shape
    N = H * W
    
    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        
        # Data term: beta * H^T H x
        Av = beta * np.real(ifft2(F_h_sq * fft2(v)))
        
        # Prior term: Dx^T (Wx * Dx * v)
        dx_v = forward_diff_x(v)
        dy_v = forward_diff_y(v)
        
        dx_v *= wx
        dy_v *= wy
        
        Av += adjoint_diff_x(dx_v)
        Av += adjoint_diff_y(dy_v)
        
        return Av.ravel()

    A_op = LinearOperator(shape=(N, N), matvec=_matvec, dtype=np.float64)
    
    x_flat, _ = cg(A_op, rhs.ravel(), x0=x_init.ravel(), maxiter=cg_iter, atol=1e-5)
    
    return x_flat.reshape((H, W))

# Placeholder for solve_image_irw if needed to avoid import errors, 
# though 'cg' is the preferred solver.
def solve_image_irw(*args, **kwargs):
    raise NotImplementedError("IRW solver is not fully adapted for the new variance tracking. Use 'cg'.")