import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import LinearOperator, cg
from typing import Tuple, List, Dict

from .utils import (
    EPSILON,
    psf2otf,
    build_rbf_basis,
    build_gradient_filters,
    compute_cross_correlation_matrix,
    compute_cross_correlation_vector,
    compute_covariance_matrix_Df,
    spectral_covariance,
    spectral_diagonal,
    compute_autocovariance,
)

def solve_image_pcg(
    g: np.ndarray,
    h_fft: np.ndarray,
    mu_f_init: np.ndarray,
    beta: float,
    gamma_maps: List[np.ndarray],
    Q_fft: List[np.ndarray],
    Q_fft_sq: List[np.ndarray],
    cg_maxiter: int = 100,
    cg_tol: float = 1e-6,
    prior_weight: float = 1.0,
) -> np.ndarray:

    H, W = g.shape
    N = H * W
    h_fft_sq = np.abs(h_fft) ** 2
    h_fft_conj = np.conj(h_fft)
    K = len(Q_fft)
    pw = prior_weight

    rhs_2d = beta * np.real(ifft2(h_fft_conj * fft2(g)))

    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        Fv = fft2(v)

        out = beta * np.real(ifft2(h_fft_sq * Fv))

        for k in range(K):
            Qk_v = np.real(ifft2(Q_fft[k] * Fv))
            Lambda_Qk_v = gamma_maps[k] * Qk_v
            out += pw * np.real(ifft2(np.conj(Q_fft[k]) * fft2(Lambda_Qk_v)))
        return out.ravel()

    A_op = LinearOperator((N, N), matvec=_matvec, dtype=np.float64)

    gamma_bar = np.array([float(np.mean(gm)) for gm in gamma_maps])
    denom = beta * h_fft_sq
    for k in range(K):
        denom = denom + pw * gamma_bar[k] * Q_fft_sq[k]
    M_inv_spectrum = 1.0 / (denom + EPSILON)

    def _precond(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        return np.real(ifft2(M_inv_spectrum * fft2(v))).ravel()

    M_op = LinearOperator((N, N), matvec=_precond, dtype=np.float64)

    mu_f_flat, info = cg(
        A_op, rhs_2d.ravel(),
        x0=mu_f_init.ravel(),
        M=M_op,
        maxiter=cg_maxiter,
        rtol=cg_tol,
        atol=0.0,
    )
    mu_f = mu_f_flat.reshape((H, W))
    return mu_f

def solve_weights_direct(
    Phi: np.ndarray,
    FtF_plus_Df: np.ndarray,
    Ftg: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:

    M = Phi.shape[1]
    A = np.diag(alpha)

    Sigma_w_inv = beta * (Phi.T @ FtF_plus_Df @ Phi) + A

    Sigma_w_inv += EPSILON * np.eye(M)

    try:
        L_chol = np.linalg.cholesky(Sigma_w_inv)
        Sigma_w = np.linalg.solve(
            L_chol.T, np.linalg.solve(L_chol, np.eye(M))
        )
    except np.linalg.LinAlgError:

        Sigma_w = np.linalg.inv(Sigma_w_inv + 1e-8 * np.eye(M))

    mu_w = beta * Sigma_w @ (Phi.T @ Ftg)
    return mu_w, Sigma_w

def update_alpha(
    mu_w: np.ndarray,
    Sigma_w: np.ndarray,
    a0_alpha: float,
    b0_alpha: float,
) -> np.ndarray:

    M = mu_w.shape[0]
    a_tilde = a0_alpha + 0.5
    E_w_sq = mu_w ** 2 + np.diag(Sigma_w)
    b_tilde = b0_alpha + 0.5 * E_w_sq
    alpha = a_tilde / (b_tilde + EPSILON)
    return alpha

def update_gamma(
    mu_f: np.ndarray,
    Q_fft: List[np.ndarray],
    Sigma_f_hat: np.ndarray,
    Q_fft_sq: List[np.ndarray],
    a0_gamma: float,
    b0_gamma: float,
    gamma_max: float = 1e6,
    prior_weight: float = 1.0,
) -> Tuple[List[np.ndarray], np.ndarray]:

    K = len(Q_fft)
    gamma_maps: List[np.ndarray] = []
    gamma_bar = np.empty(K, dtype=np.float64)
    pw_half = prior_weight * 0.5
    a_tilde = a0_gamma + pw_half

    F_mu = fft2(mu_f)
    for k in range(K):
        Qk_mu_f = np.real(ifft2(Q_fft[k] * F_mu))
        c_k = spectral_diagonal(Q_fft_sq[k], Sigma_f_hat)
        E_eps_sq = Qk_mu_f ** 2 + c_k

        b_tilde = b0_gamma + pw_half * E_eps_sq
        gmap = a_tilde / (b_tilde + EPSILON)

        np.clip(gmap, EPSILON, gamma_max, out=gmap)
        gamma_maps.append(gmap)
        gamma_bar[k] = float(np.mean(gmap))

    return gamma_maps, gamma_bar

def update_beta(
    g: np.ndarray,
    h_fft: np.ndarray,
    mu_f: np.ndarray,
    Sigma_f_hat: np.ndarray,
    Sigma_w: np.ndarray,
    Phi: np.ndarray,
    a0_beta: float,
    b0_beta: float,
) -> float:

    H, W = g.shape
    N = H * W

    H_mu_f = np.real(ifft2(h_fft * fft2(mu_f)))
    residual_sq = float(np.sum((g - H_mu_f) ** 2))

    trace_term = float(np.sum(np.abs(h_fft) ** 2 * Sigma_f_hat))

    mu_f_power = float(np.sum(np.abs(fft2(mu_f)) ** 2)) / N
    kernel_unc_term = float(np.trace(Sigma_w @ (Phi.T @ Phi))) * mu_f_power

    expected_residual = residual_sq + trace_term + kernel_unc_term

    a_tilde = a0_beta + 0.5 * N
    b_tilde = b0_beta + 0.5 * expected_residual
    beta = a_tilde / (b_tilde + EPSILON)
    beta = float(np.clip(beta, 1e-1, 1e8))
    return beta

def solve_image_chantas2008_pcg(
    g: np.ndarray,
    h_fft: np.ndarray,
    mu_f_init: np.ndarray,
    beta: float,
    gamma_maps: List[np.ndarray],
    Q_fft: List[np.ndarray],
    Q_fft_sq: List[np.ndarray],
    prior_weight: float = 0.25,
    cg_maxiter: int = 200,
    cg_tol: float = 1e-7,
) -> Tuple[np.ndarray, List[np.ndarray]]:

    H, W = g.shape
    K = len(Q_fft)
    pw = prior_weight

    h_fft_sq = np.abs(h_fft) ** 2
    h_fft_conj = np.conj(h_fft)

    rhs = (beta * np.real(ifft2(h_fft_conj * fft2(g)))).ravel()
    rhs_norm = np.linalg.norm(rhs)

    def _Av(v_flat):
        v = v_flat.reshape((H, W))
        Fv = fft2(v)
        out = beta * np.real(ifft2(h_fft_sq * Fv))
        qk_list = []
        for k in range(K):
            Qk_v = np.real(ifft2(Q_fft[k] * Fv))
            qk_list.append(Qk_v)
            out += pw * np.real(
                ifft2(np.conj(Q_fft[k]) * fft2(gamma_maps[k] * Qk_v))
            )
        return out.ravel(), qk_list

    g_bar = np.array([float(np.mean(gm)) for gm in gamma_maps])
    denom = beta * h_fft_sq
    for k in range(K):
        denom = denom + pw * g_bar[k] * Q_fft_sq[k]
    Minv_spec = 1.0 / (denom + EPSILON)

    def _Minv(v_flat):
        v = v_flat.reshape((H, W))
        return np.real(ifft2(Minv_spec * fft2(v))).ravel()

    x = mu_f_init.ravel().copy()
    Ax0, _ = _Av(x)
    r = rhs - Ax0
    z = _Minv(r)
    p = z.copy()
    rz = np.dot(r, z)

    var_maps = [np.zeros((H, W)) for _ in range(K)]

    for it in range(cg_maxiter):
        Ap, qk_p = _Av(p)
        pAp = np.dot(p, Ap)
        if pAp < EPSILON:
            break

        step = rz / pAp
        x += step * p
        r -= step * Ap

        inv_pAp = 1.0 / pAp
        for k in range(K):
            var_maps[k] += qk_p[k] ** 2 * inv_pAp

        if np.linalg.norm(r) / (rhs_norm + EPSILON) < cg_tol:
            break

        z = _Minv(r)
        rz_new = np.dot(r, z)
        p = z + (rz_new / (rz + EPSILON)) * p
        rz = rz_new

    mu_f = x.reshape((H, W))
    for k in range(K):
        np.maximum(var_maps[k], EPSILON, out=var_maps[k])

    return mu_f, var_maps

def update_gamma_constrained(
    mu_f: np.ndarray,
    Q_fft: List[np.ndarray],
    var_maps: List[np.ndarray],
    a0_gamma: float,
    b0_gamma: float,
    gamma_max: float = 1e6,
    prior_weight: float = 0.25,
) -> Tuple[List[np.ndarray], np.ndarray]:

    K = len(Q_fft)
    gamma_maps: List[np.ndarray] = []
    gamma_bar = np.empty(K, dtype=np.float64)
    pw_half = prior_weight * 0.5
    a_tilde = a0_gamma + pw_half

    F_mu = fft2(mu_f)
    for k in range(K):
        Qk_mu = np.real(ifft2(Q_fft[k] * F_mu))
        E_eps_sq = Qk_mu ** 2 + var_maps[k]

        b_tilde = b0_gamma + pw_half * E_eps_sq
        gmap = a_tilde / (b_tilde + EPSILON)
        np.clip(gmap, EPSILON, gamma_max, out=gmap)
        gamma_maps.append(gmap)
        gamma_bar[k] = float(np.mean(gmap))

    return gamma_maps, gamma_bar

def prune_ard(
    alpha: np.ndarray,
    Phi: np.ndarray,
    mu_w: np.ndarray,
    threshold: float = 1e10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    keep = alpha < threshold
    if np.all(keep):
        return alpha, Phi, mu_w, keep

    if not np.any(keep):
        best = np.argmin(alpha)
        keep[best] = True
    return alpha[keep], Phi[:, keep], mu_w[keep], keep
