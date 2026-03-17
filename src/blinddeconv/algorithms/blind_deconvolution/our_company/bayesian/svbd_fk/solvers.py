"""
VB update steps for SVBD-FK.

Sonogashira et al. (2017), "Shift-Variant Blind Deconvolution Using a
Field of Kernels", IEICE Trans. Inf. & Syst., E100-D(9), 1971-1983.
DOI: 10.1587/transinf.2016PCP0013

Each public function corresponds to one block in the VB algorithm (Fig. 3):
    forward_blur / adjoint_blur   — operators  A, A^T  (Eq. 3)
    update_image                  — Eq. (9)-(10)
    update_kernel_l               — Eq. (11)-(12)
    update_weight_l               — Eq. (13)-(14)
    update_beta / update_alpha /
    update_gamma / update_eta     — Eq. (15)-(18)
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from .utils import (
    fft_conv2d, fft_corr2d,
    forward_diff_h, forward_diff_v,
    adjoint_diff_h, adjoint_diff_v,
    laplacian_fft_spectrum, apply_laplacian_fft,
    conjugate_gradient,
    build_weighted_gram, build_weighted_rhs,
    compute_tv_weights, psf_to_otf,
    _shift_offsets,
    EPSILON,
)

# ========================================================================
#  Forward / adjoint blur  (Eq. 3)
#  A x = sum_l  diag(mu_w_l) K_l  x       (L FFT convolutions)
#  A^T y = sum_l  K_l^T (mu_w_l .* y)      (L FFT correlations)
# ========================================================================

def forward_blur(
    x: np.ndarray,
    mu_k: List[np.ndarray],
    mu_w: List[np.ndarray],
) -> np.ndarray:
    """
    Forward operator  A_bar x = sum_l  diag(mu_w_l) K_l x.

    Ref: Eq. (3) — observation model  y = A x + n  with
    A = sum_l diag(w_l) K_l.
    """
    out = np.zeros_like(x)
    for k_l, w_l in zip(mu_k, mu_w):
        out += w_l * fft_conv2d(x, k_l)
    return out


def adjoint_blur(
    y: np.ndarray,
    mu_k: List[np.ndarray],
    mu_w: List[np.ndarray],
) -> np.ndarray:
    """
    Adjoint operator  A_bar^T y = sum_l K_l^T (mu_w_l .* y).

    Ref: transpose of Eq. (3), used in the image update RHS (Eq. 9).
    """
    out = np.zeros_like(y)
    for k_l, w_l in zip(mu_k, mu_w):
        out += fft_corr2d(w_l * y, k_l)
    return out

# ========================================================================
#  Step 2 — Image  q(x)  update  (Eq. 9-10)
# ========================================================================

def update_image(
    y: np.ndarray,
    mu_k: List[np.ndarray],
    mu_w: List[np.ndarray],
    sigma_w: List[np.ndarray],
    sigma_k: List[np.ndarray],
    lam_h: np.ndarray,
    lam_v: np.ndarray,
    beta: float,
    alpha: float,
    mu_x_prev: np.ndarray,
    cg_iter: int,
    F_L: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Image update:  q(x) = N(mu_x, Sigma_x)  via CG.

    Ref: Eq. (9)  Sigma_x^{-1} mu_x  =  beta A_bar^T y
         Eq. (10) Sigma_x^{-1}  =  beta <A^T A>  +  alpha D^T Lambda D

    where <A^T A> = A_bar^T A_bar + sum_l sigma_{w_l}^2(i) ||mu_{k_l}||^2 I.

    Parameters
    ----------
    y            : (H, W)   observed blurry image
    mu_k         : list of L kernels, each (kh, kw)
    mu_w         : list of L weight maps, each (H, W)
    sigma_w      : list of L weight variance maps, each (H, W)
    sigma_k      : list of L kernel covariance diagonals, each (K^2,)
    lam_h, lam_v : TV weights from compute_tv_weights(mu_x_prev)
    beta         : noise precision
    alpha        : image smoothness hyperparameter
    mu_x_prev    : (H, W) previous image mean (CG warm-start)
    cg_iter      : maximum CG iterations
    F_L          : Laplacian FFT spectrum (not used here, passed for API)

    Returns
    -------
    mu_x    : (H, W)  image mean
    sigma_x : (H, W)  pixel-wise variance (diagonal approximation)
    """
    L = len(mu_k)

    # --- Pre-compute scalar kernel energies ||mu_k_l||^2 ---
    # Used in the diagonal variance correction for <A^T A>.
    k_energy = np.array([np.sum(k ** 2) for k in mu_k])  # (L,)

    # --- CG right-hand side: beta * A_bar^T y  (Eq. 9) ---
    rhs = beta * adjoint_blur(y, mu_k, mu_w)

    # --- CG operator: Sigma_x^{-1} v  (Eq. 10) ---
    # = beta * <A^T A> v  +  alpha * D^T Lambda D v
    def matvec_x(v: np.ndarray) -> np.ndarray:
        # Term 1:  A_bar^T A_bar v   (2L FFTs per CG step)
        Av = forward_blur(v, mu_k, mu_w)
        AtAv = adjoint_blur(Av, mu_k, mu_w)

        # Term 2:  Diagonal variance correction
        # sum_l sigma_{w_l}(i) * ||mu_{k_l}||^2  —  accounts for
        # uncertainty in weights when computing <A^T A>.
        diag_corr = np.zeros_like(v)
        for l in range(L):
            diag_corr += sigma_w[l] * k_energy[l]
        AtAv += diag_corr * v

        result = beta * AtAv

        # Term 3:  TV regularisation (quadratic lower bound)
        # alpha * (D_h^T Lambda_h D_h + D_v^T Lambda_v D_v) v
        dh_v = forward_diff_h(v)
        dv_v = forward_diff_v(v)
        tv_term = adjoint_diff_h(lam_h * dh_v) + adjoint_diff_v(lam_v * dv_v)
        result += alpha * tv_term

        return result

    mu_x = conjugate_gradient(matvec_x, rhs, x0=mu_x_prev, max_iter=cg_iter)

    # --- Diagonal approximation of Sigma_x  (Eq. 10 approx.) ---
    # diag( Sigma_x^{-1} )_i  ≈  beta * sum_l E[w_l(i)^2] * ||k_l||^2
    #                            + alpha * diag(D^T Lambda D)_i
    diag_prec = np.zeros_like(mu_x)
    for l in range(L):
        diag_prec += (mu_w[l] ** 2 + sigma_w[l]) * k_energy[l]
    diag_prec *= beta

    # diag(D_h^T Lambda_h D_h)_i = Lambda_h(i,j) + Lambda_h(i,j-1)
    # diag(D_v^T Lambda_v D_v)_i = Lambda_v(i,j) + Lambda_v(i-1,j)
    diag_tv = (
        lam_h + np.roll(lam_h, 1, axis=1)
        + lam_v + np.roll(lam_v, 1, axis=0)
    )
    diag_prec += alpha * diag_tv

    sigma_x = 1.0 / (diag_prec + EPSILON)

    return mu_x, sigma_x

# ========================================================================
#  Step 3a — Kernel  q(k_l)  update  (Eq. 11-12)
# ========================================================================

def update_kernel_l(
    y: np.ndarray,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    mu_w_l: np.ndarray,
    sigma_w_l: np.ndarray,
    mu_k_l_prev: np.ndarray,
    sigma_k_l_prev: np.ndarray,
    other_blur: np.ndarray,
    beta: float,
    gamma_l: float,
    kernel_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kernel update:  q(k_l) = N(mu_{k_l}, Sigma_{k_l})  via direct solve.

    Ref: Eq. (11) Sigma_{k_l}^{-1}  =  beta G_l  +  gamma_l Lambda_{k_l}
         Eq. (12) Sigma_{k_l}^{-1} mu_{k_l}  =  beta b_l

    where G_l is the K^2 × K^2 weighted Gram matrix, b_l is the
    weighted correlation vector, and Lambda_{k_l} is the diagonal
    Laplace prior weight (cf. Babacan [31]).

    The system size is K^2 × K^2  (e.g. 121 for 11×11 kernels),
    so direct solve via np.linalg.solve is used instead of CG.

    Parameters
    ----------
    y          : (H, W)  observed blurry image
    mu_x       : (H, W)  current image mean
    sigma_x    : (H, W)  current pixel-wise image variance
    mu_w_l     : (H, W)  weight map mean for basis l
    sigma_w_l  : (H, W)  weight map variance for basis l
    mu_k_l_prev    : (kh, kw)  previous kernel mean
    sigma_k_l_prev : (K^2,)    previous kernel covariance diagonal
    other_blur : (H, W)  sum_{m != l} diag(mu_w_m) K_m mu_x
    beta       : noise precision
    gamma_l    : kernel sparsity hyperparameter
    kernel_shape : (kh, kw)

    Returns
    -------
    mu_k_l    : (kh, kw)
    sigma_k_l : (K^2,) diagonal of kernel covariance
    """
    K2 = kernel_shape[0] * kernel_shape[1]

    # Second moment of weights:  E[w_l(i)^2] = mu_w_l(i)^2 + sigma_w_l(i)
    d_w = mu_w_l ** 2 + sigma_w_l  # (H, W)

    # --- Gram matrix G_l  (K^2 × K^2) ---
    # G_{j1,j2} = sum_i  d_w(i) * mu_x(i - delta_{j1}) * mu_x(i - delta_{j2})
    G = build_weighted_gram(mu_x, d_w, kernel_shape)

    # Diagonal correction for image uncertainty:
    # G_{j,j} += sum_i  d_w(i) * sigma_x(i - delta_j)
    offsets = _shift_offsets(kernel_shape)
    for j in range(K2):
        dy, dx = offsets[j]
        shifted_sigma = np.roll(sigma_x, (dy, dx), axis=(0, 1))
        G[j, j] += np.sum(d_w * shifted_sigma)

    # --- Sparse prior Lambda_{k_l}  (diagonal Laplace bound) ---
    # Lambda_{k_l,jj} = 1 / (|mu_{k_l,j}| + eps)   [Babacan, 31]
    mu_k_flat = mu_k_l_prev.ravel()
    lam_k = np.diag(1.0 / (np.abs(mu_k_flat) + 1e-6))

    # --- Precision matrix and solve  (Eq. 11 + 12) ---
    Sigma_k_inv = beta * G + gamma_l * lam_k

    residual_l = y - other_blur  # r_l = y - sum_{m!=l} diag(w_m) K_m x
    b_k = beta * build_weighted_rhs(mu_x, mu_w_l, residual_l, kernel_shape)

    # Direct solve  (K^2 × K^2 system, e.g. 121 for 11×11 kernels)
    try:
        mu_k_flat_new = np.linalg.solve(Sigma_k_inv, b_k)
    except np.linalg.LinAlgError:
        mu_k_flat_new = np.linalg.lstsq(Sigma_k_inv, b_k, rcond=None)[0]

    # Kernel covariance diagonal:  diag(Sigma_k) = diag(Sigma_k_inv^{-1})
    try:
        Sigma_k = np.linalg.inv(Sigma_k_inv)
        sigma_k_diag = np.diag(Sigma_k).copy()
    except np.linalg.LinAlgError:
        sigma_k_diag = 1.0 / (np.diag(Sigma_k_inv) + EPSILON)

    sigma_k_diag = np.maximum(sigma_k_diag, 0.0)

    # --- Post-processing: non-negativity + l1 normalisation ---
    mu_k_flat_new = np.maximum(mu_k_flat_new, 0.0)
    s = mu_k_flat_new.sum()
    if s > EPSILON:
        mu_k_flat_new /= s

    mu_k_new = mu_k_flat_new.reshape(kernel_shape)
    return mu_k_new, sigma_k_diag

# ========================================================================
#  Step 3b — Weight map  q(w_l)  update  (Eq. 13-14)
# ========================================================================

def update_weight_l(
    y: np.ndarray,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    mu_k_l: np.ndarray,
    sigma_k_l: np.ndarray,
    mu_w_l_prev: np.ndarray,
    other_weight_contrib: np.ndarray,
    beta: float,
    eta_l: float,
    F_L: np.ndarray,
    cg_iter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weight-map update:  q(w_l) = N(mu_{w_l}, Sigma_{w_l})  via CG.

    Ref: Eq. (13) Sigma_{w_l}^{-1} = beta diag(f_l^2 + sigma_{f_l}) + eta_l L
         Eq. (14) Sigma_{w_l}^{-1} mu_{w_l} = beta (f_l .* r_{w_l})

    where f_l = K_l mu_x  is the filtered image and sigma_{f_l} accounts
    for uncertainty in both x and k_l.

    Parameters
    ----------
    y                    : (H, W)   observed image
    mu_x                 : (H, W)   image mean
    sigma_x              : (H, W)   pixel-wise image variance
    mu_k_l               : (kh, kw) kernel mean for basis l
    sigma_k_l            : (K^2,)   kernel covariance diagonal
    mu_w_l_prev          : (H, W)   previous weight map mean (CG warm-start)
    other_weight_contrib : (H, W)   sum_{m != l} mu_w_m .* f_m
    beta                 : noise precision
    eta_l                : weight smoothness hyperparameter
    F_L                  : Laplacian FFT spectrum
    cg_iter              : maximum CG iterations

    Returns
    -------
    mu_w_l    : (H, W)  weight map mean
    sigma_w_l : (H, W)  pixel-wise weight variance (diagonal approx.)
    """
    # --- Filtered image and its variance ---
    # f_l(i) = [K_l mu_x](i) = sum_j mu_{k_l}(j) * mu_x(i - delta_j)
    f_l = fft_conv2d(mu_x, mu_k_l)                          # (H, W)

    # Var[f_l(i)] = sum_j k(j)^2 sigma_x(i - delta_j)
    #             + sum_j sigma_k(j) * (mu_x(i-delta_j)^2 + sigma_x(i-delta_j))
    mu_k_sq = mu_k_l ** 2
    sigma_k_img = sigma_k_l.reshape(mu_k_l.shape)           # (kh, kw)
    sigma_f = (
        fft_conv2d(sigma_x, mu_k_sq)
        + fft_conv2d(mu_x ** 2 + sigma_x, sigma_k_img)
    )
    sigma_f = np.maximum(sigma_f, 0.0)

    # Residual for basis l:  r_{w_l} = y - sum_{m != l} w_m .* f_m
    r_w_l = y - other_weight_contrib                         # (H, W)

    # --- RHS  (Eq. 14) ---
    rhs = beta * (f_l * r_w_l)

    # --- Second moment of filtered image ---
    # E[f_l(i)^2] = f_l(i)^2 + sigma_f(i)
    f_sq_plus = f_l ** 2 + sigma_f                           # (H, W)

    # --- CG operator: Sigma_{w_l}^{-1} v  (Eq. 13) ---
    # = beta * diag(f_sq_plus) v  +  eta_l * L v
    def matvec_w(v: np.ndarray) -> np.ndarray:
        return beta * f_sq_plus * v + eta_l * apply_laplacian_fft(v, F_L)

    mu_w_new = conjugate_gradient(matvec_w, rhs, x0=mu_w_l_prev, max_iter=cg_iter)

    # --- Diagonal variance approximation ---
    # diag(Sigma_{w_l}^{-1})_i  =  beta * f_sq_plus(i) + eta_l * diag(L)
    # For the periodic discrete Laplacian, diag(L) = 4 (center stencil coeff.)
    sigma_w_new = 1.0 / (beta * f_sq_plus + eta_l * 4.0 + EPSILON)

    # Post-processing: non-negativity constraint on weight maps
    mu_w_new = np.maximum(mu_w_new, 0.0)

    return mu_w_new, sigma_w_new

# ========================================================================
#  Step 4 — Hyperparameters  (Eq. 15-18)
# ========================================================================

def update_beta(
    y: np.ndarray,
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    mu_k: List[np.ndarray],
    mu_w: List[np.ndarray],
    sigma_w: List[np.ndarray],
    a0: float = 1e-6,
    b0: float = 1e-6,
) -> float:
    """
    Noise precision update:  q(beta) = Gamma(a_beta, b_beta).

    Ref: Eq. (15)
        a_beta = a_0 + N/2
        b_beta = b_0 + 1/2 * (<||y - Ax||^2>  +  tr(<A^T A> Sigma_x))
        beta   = a_beta / b_beta   (posterior mean of Gamma)
    """
    N = y.size

    # <||y - Ax||^2> at the posterior mean
    Ax = forward_blur(mu_x, mu_k, mu_w)
    res_sq = np.sum((y - Ax) ** 2)

    # tr(<A^T A> Sigma_x) ≈ sum_i diag(<A^T A>)_i * sigma_x(i)
    # diag(<A^T A>)_i = sum_l E[w_l(i)^2] * ||k_l||^2
    k_energy = np.array([np.sum(k ** 2) for k in mu_k])
    diag_AtA = np.zeros_like(mu_x)
    for l in range(len(mu_k)):
        diag_AtA += (mu_w[l] ** 2 + sigma_w[l]) * k_energy[l]
    trace_term = np.sum(diag_AtA * sigma_x)

    a_beta = a0 + N / 2.0
    b_beta = b0 + 0.5 * (res_sq + trace_term)
    return a_beta / (b_beta + EPSILON)


def update_alpha(
    mu_x: np.ndarray,
    sigma_x: np.ndarray,
    a0: float = 1e-6,
    b0: float = 1e-6,
) -> float:
    """
    Image smoothness update:  q(alpha) = Gamma(a_alpha, b_alpha).

    Ref: Eq. (16)
        a_alpha = a_0 + N
        b_alpha = b_0 + sum_i <|D_h x_i| + |D_v x_i|>

    where <|D x_i|> ≈ sqrt( (D mu_x)_i^2 + Var(D x_i) )  is the
    expected absolute gradient under q(x) (Jaakkola bound).
    """
    N = mu_x.size
    dh = forward_diff_h(mu_x)
    dv = forward_diff_v(mu_x)

    # Variance of gradients
    sigma_dh = sigma_x + np.roll(sigma_x, -1, axis=1)
    sigma_dv = sigma_x + np.roll(sigma_x, -1, axis=0)

    b_sum = np.sum(np.sqrt(dh ** 2 + sigma_dh + EPSILON)
                   + np.sqrt(dv ** 2 + sigma_dv + EPSILON))

    a_alpha = a0 + N
    b_alpha = b0 + b_sum
    return a_alpha / (b_alpha + EPSILON)


def update_gamma(
    mu_k_l: np.ndarray,
    sigma_k_l: np.ndarray,
    a0: float = 1e-6,
    b0: float = 1e-6,
) -> float:
    """
    Kernel sparsity update:  q(gamma_l) = Gamma(a_gamma, b_gamma).

    Ref: Eq. (17)
        a_gamma = a_0 + K^2 / 2
        b_gamma = b_0 + 1/2 * sum_j <|k_{l,j}|>
    """
    K2 = mu_k_l.size
    mu_flat = mu_k_l.ravel()
    b_sum = 0.5 * np.sum(np.sqrt(mu_flat ** 2 + sigma_k_l + EPSILON))

    a_gamma = a0 + K2 / 2.0
    b_gamma = b0 + b_sum
    return a_gamma / (b_gamma + EPSILON)


def update_eta(
    mu_w_l: np.ndarray,
    sigma_w_l: np.ndarray,
    F_L: np.ndarray,
    a0: float = 1e-6,
    b0: float = 1e-6,
) -> float:
    """
    Weight smoothness update:  q(eta_l) = Gamma(a_eta, b_eta).

    Ref: Eq. (18)
        a_eta = a_0 + N / 2
        b_eta = b_0 + 1/2 * <w_l^T L w_l>

    where <w_l^T L w_l> = mu_w^T L mu_w + tr(L Sigma_w)
                        ≈ ||∇ mu_w||^2 + 4 * sum(sigma_w)   (diagonal L approx.)
    """
    N = mu_w_l.size
    dh = forward_diff_h(mu_w_l)
    dv = forward_diff_v(mu_w_l)
    grad_sq = np.sum(dh ** 2 + dv ** 2)

    # tr(L Sigma_w) ~ diag(L) . sigma_w  ≈ 4 * sum(sigma_w)
    trace_L_Sigma = 4.0 * np.sum(sigma_w_l)

    a_eta = a0 + N / 2.0
    b_eta = b0 + 0.5 * (grad_sq + trace_L_Sigma)
    return a_eta / (b_eta + EPSILON)
