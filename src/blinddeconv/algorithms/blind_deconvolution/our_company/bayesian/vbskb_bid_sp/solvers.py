"""
Core solvers for Variational Bayesian Sparse Kernel-Based
Blind Image Deconvolution with Student's-t Priors (VBSKB-BID-STP).

Based on:
    Tzikas D.G., Likas A.C., Galatsanos N.P. (2009).
    "Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution
     With Student's-t Priors." IEEE Trans. Image Process., 18(1), 200–208.

    Chantas G., Galatsanos N., Likas A., Saunders M. (2008).
    "Variational Bayesian Image Restoration Based on a Product of
     t-Distributions Image Prior." IEEE Trans. Image Process., 17(10), 1795–1805.

    Tipping M.E. (2001). "Sparse Bayesian Learning and the Relevance
     Vector Machine." J. Mach. Learn. Res., 1, 211–244.

    Bishop C.M. (2006). Pattern Recognition and Machine Learning, Ch. 10.

Contains:
    - Preconditioned Conjugate Gradient solver for q(f)  [Eq. (34)–(35), (59)–(60)].
    - Direct (Cholesky) solver for q(w)                  [Eq. (32)–(33)].
    - Variational M-step updates for α, γ, β             [Eq. (36)–(49)].
    - Expected residual computation                      [Eq. (41)].
    - ARD pruning of irrelevant basis functions.
"""

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


# ═══════════════════════════════════════════════════════════════════════════
#  Image estimation  —  Preconditioned CG  (Tzikas 2009, Eq. 34-35, 59-60)
# ═══════════════════════════════════════════════════════════════════════════

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
    r"""Solve for the image posterior mean μ_f via preconditioned CG.

    The linear system (Chantas 2008 Eq. 3.10–3.11 / Tzikas 2009 Eq. 34–35):

        A = ⟨β⟩ H̄^T H̄  +  (1/P) Σ_k (Q^k)^T Λ^k Q^k
        b = ⟨β⟩ H̄^T g

    where (1/P) is *prior_weight* (Chantas 2008, product-of-t normalisation).
    For Tzikas 2009 (blind loop), prior_weight = 1.0.
    For Chantas 2008 (non-blind), prior_weight = 1/K.

    Parameters
    ----------
    g : ndarray (H, W)
    h_fft : ndarray (H, W), complex
    mu_f_init : ndarray (H, W)
    beta : float
    gamma_maps : list of ndarray (H, W), length K
    Q_fft : list of ndarray (H, W), complex, length K
    Q_fft_sq : list of ndarray (H, W), real, length K
    cg_maxiter : int
    cg_tol : float
    prior_weight : float
        Weight on the prior term.  1.0 = Tzikas 2009,  1/K = Chantas 2008.

    Returns
    -------
    mu_f : ndarray (H, W)
    """
    H, W = g.shape
    N = H * W
    h_fft_sq = np.abs(h_fft) ** 2
    h_fft_conj = np.conj(h_fft)
    K = len(Q_fft)
    pw = prior_weight

    # ── Right-hand side  b = β H̄^T g  (Eq. 35 / 3.11) ──────────────────
    rhs_2d = beta * np.real(ifft2(h_fft_conj * fft2(g)))

    # ── Matvec  A·v  (Eq. 59 / 3.10) ─────────────────────────────────────
    def _matvec(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        Fv = fft2(v)
        # Fidelity term:  β H̄^T H̄ v
        out = beta * np.real(ifft2(h_fft_sq * Fv))
        # Prior terms:  (1/P) Σ_k (Q^k)^T Λ^k Q^k v   (Chantas 2008 Eq. 3.10)
        for k in range(K):
            Qk_v = np.real(ifft2(Q_fft[k] * Fv))
            Lambda_Qk_v = gamma_maps[k] * Qk_v
            out += pw * np.real(ifft2(np.conj(Q_fft[k]) * fft2(Lambda_Qk_v)))
        return out.ravel()

    A_op = LinearOperator((N, N), matvec=_matvec, dtype=np.float64)

    # ── Preconditioner  M⁻¹  (Eq. 60) ────────────────────────────────────
    gamma_bar = np.array([float(np.mean(gm)) for gm in gamma_maps])
    denom = beta * h_fft_sq
    for k in range(K):
        denom = denom + pw * gamma_bar[k] * Q_fft_sq[k]
    M_inv_spectrum = 1.0 / (denom + EPSILON)

    def _precond(v_flat: np.ndarray) -> np.ndarray:
        v = v_flat.reshape((H, W))
        return np.real(ifft2(M_inv_spectrum * fft2(v))).ravel()

    M_op = LinearOperator((N, N), matvec=_precond, dtype=np.float64)

    # ── Solve ─────────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════
#  Kernel-weight estimation  —  Direct solve  (Tzikas 2009, Eq. 32-33)
# ═══════════════════════════════════════════════════════════════════════════

def solve_weights_direct(
    Phi: np.ndarray,
    FtF_plus_Df: np.ndarray,
    Ftg: np.ndarray,
    alpha: np.ndarray,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Solve for kernel-weight posterior q(w) = N(μ_w, Σ_w).

    Σ_w = ( β Φ^T (F̄^T F̄ + D_f) Φ  +  A )⁻¹           (Eq. 32)
    μ_w = β Σ_w Φ^T F̄^T g                                (Eq. 33)

    where A = diag(⟨α⟩).

    Parameters
    ----------
    Phi : ndarray (L, M)
        RBF basis matrix.
    FtF_plus_Df : ndarray (L, L)
        F̄^T F̄ + D_f   (autocorrelation + covariance term).
    Ftg : ndarray (L,)
        F̄^T g          (cross-correlation vector).
    alpha : ndarray (M,)
        Expected ARD precisions ⟨α_i⟩.
    beta : float
        Expected noise precision ⟨β⟩.

    Returns
    -------
    mu_w   : ndarray (M,)
    Sigma_w : ndarray (M, M)
    """
    M = Phi.shape[1]
    A = np.diag(alpha)

    # Σ_w^{-1} = β Φ^T (F̄^T F̄ + D_f) Φ + A
    Sigma_w_inv = beta * (Phi.T @ FtF_plus_Df @ Phi) + A

    # Regularise for numerical stability
    Sigma_w_inv += EPSILON * np.eye(M)

    # Cholesky solve (SPD matrix)
    try:
        L_chol = np.linalg.cholesky(Sigma_w_inv)
        Sigma_w = np.linalg.solve(
            L_chol.T, np.linalg.solve(L_chol, np.eye(M))
        )
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse when Cholesky fails
        Sigma_w = np.linalg.inv(Sigma_w_inv + 1e-8 * np.eye(M))

    # μ_w = β Σ_w Φ^T F̄^T g   (Eq. 33)
    mu_w = beta * Sigma_w @ (Phi.T @ Ftg)
    return mu_w, Sigma_w


# ═══════════════════════════════════════════════════════════════════════════
#  VM-step:  Hyperparameter updates  (Tzikas 2009, Eq. 36-49)
# ═══════════════════════════════════════════════════════════════════════════

def update_alpha(
    mu_w: np.ndarray,
    Sigma_w: np.ndarray,
    a0_alpha: float,
    b0_alpha: float,
) -> np.ndarray:
    r"""Update ARD precisions q(α_i) = Gamma(ã_i^α, b̃_i^α).

    ã_i^α = a₀^α + ½                                       (Eq. 36)
    b̃_i^α = b₀^α + ½ ⟨w_i²⟩                              (Eq. 37)
    ⟨α_i⟩  = ã_i^α / b̃_i^α                               (Eq. 42)

    where ⟨w_i²⟩ = μ_{w,i}² + [Σ_w]_{ii}                  (Eq. 48)

    Parameters
    ----------
    mu_w    : ndarray (M,)
    Sigma_w : ndarray (M, M)
    a0_alpha, b0_alpha : float
        Prior hyperparameters for Gamma(α_i).

    Returns
    -------
    alpha : ndarray (M,)
        Updated ⟨α_i⟩.
    """
    M = mu_w.shape[0]
    a_tilde = a0_alpha + 0.5                                           # (Eq. 36)
    E_w_sq = mu_w ** 2 + np.diag(Sigma_w)                             # (Eq. 48)
    b_tilde = b0_alpha + 0.5 * E_w_sq                                 # (Eq. 37)
    alpha = a_tilde / (b_tilde + EPSILON)                              # (Eq. 42)
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
    r"""Update per-pixel gradient precisions q(γ_j^k).

    Chantas 2008 (Eq. 3.14–3.15) with product-of-t normalisation (1/P):

        ã = a₀^γ + (1/P) · ½ = a₀ + prior_weight/2
        b̃ = b₀^γ + (1/P) · ½ ⟨(ε_j^k)²⟩ = b₀ + prior_weight/2 · ⟨ε²⟩
        ⟨γ_j^k⟩ = ã / b̃

    For Tzikas 2009 (blind loop): prior_weight = 1.0 → ã = a₀ + 0.5
    For Chantas 2008 (non-blind): prior_weight = 1/K → ã = a₀ + 1/(2K)

    Parameters
    ----------
    mu_f         : ndarray (H, W)
    Q_fft        : list of complex ndarray (H, W), length K
    Sigma_f_hat  : ndarray (H, W)
    Q_fft_sq     : list of ndarray (H, W), length K
    a0_gamma, b0_gamma : float
    gamma_max    : float
    prior_weight : float
        1.0 for Tzikas 2009; 1/K for Chantas 2008.

    Returns
    -------
    gamma_maps : list of ndarray (H, W), length K
    gamma_bar  : ndarray (K,)
    """
    K = len(Q_fft)
    gamma_maps: List[np.ndarray] = []
    gamma_bar = np.empty(K, dtype=np.float64)
    pw_half = prior_weight * 0.5
    a_tilde = a0_gamma + pw_half                                       # Chantas Eq. 3.14

    F_mu = fft2(mu_f)
    for k in range(K):
        Qk_mu_f = np.real(ifft2(Q_fft[k] * F_mu))
        c_k = spectral_diagonal(Q_fft_sq[k], Sigma_f_hat)
        E_eps_sq = Qk_mu_f ** 2 + c_k

        b_tilde = b0_gamma + pw_half * E_eps_sq                        # Chantas Eq. 3.15
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
    r"""Update noise precision q(β) = Gamma(ã^β, b̃^β).

    ã^β = a₀^β + N/2                                                    (Eq. 40)
    b̃^β = b₀^β + ½ ⟨‖g − Hf‖²⟩                                      (Eq. 41)
    ⟨β⟩ = ã^β / b̃^β                                                   (Eq. 46)

    The kernel-uncertainty term is approximated as:
        tr(Σ_w Φ^T Φ) · (1/N) Σ_ω |FFT(μ_f)|²

    Parameters
    ----------
    g            : ndarray (H, W)
    h_fft        : ndarray (H, W), complex
    mu_f         : ndarray (H, W)
    Sigma_f_hat  : ndarray (H, W)
    Sigma_w      : ndarray (M, M)
    Phi          : ndarray (L, M)
    a0_beta, b0_beta : float

    Returns
    -------
    beta : float
    """
    H, W = g.shape
    N = H * W

    # 1) Data-fit residual  ‖g − H̄ μ_f‖²
    H_mu_f = np.real(ifft2(h_fft * fft2(mu_f)))
    residual_sq = float(np.sum((g - H_mu_f) ** 2))

    # 2) Image-covariance trace:  tr(H̄^T H̄ Σ_f) ≈ Σ_ω |ĥ(ω)|² Σ̂_f(ω)
    trace_term = float(np.sum(np.abs(h_fft) ** 2 * Sigma_f_hat))

    # 3) Kernel-uncertainty (scalar approximation):
    #    tr(Σ_w Φ^T Φ) · (1/N) Σ_ω |FFT(μ_f)|²
    mu_f_power = float(np.sum(np.abs(fft2(mu_f)) ** 2)) / N
    kernel_unc_term = float(np.trace(Sigma_w @ (Phi.T @ Phi))) * mu_f_power

    expected_residual = residual_sq + trace_term + kernel_unc_term

    a_tilde = a0_beta + 0.5 * N                                        # (Eq. 40)
    b_tilde = b0_beta + 0.5 * expected_residual                        # (Eq. 41)
    beta = a_tilde / (b_tilde + EPSILON)                               # (Eq. 46)
    beta = float(np.clip(beta, 1e-1, 1e8))
    return beta


# ═══════════════════════════════════════════════════════════════════════════
#  Non-blind image restoration  (Chantas 2008, Sec. III–IV)
# ═══════════════════════════════════════════════════════════════════════════

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
    r"""Preconditioned CG for Chantas 2008 non-blind restoration with
    simultaneous per-pixel constrained-variance estimation.

    Solves  A m = β H^T g  where

        A = β H^T H  +  (1/P) Σ_{k=1}^{P} Q_k^T diag(⟨γ^k⟩) Q_k

    and simultaneously estimates per-pixel variance maps

        var_k(j) = [Q_k A^{-1} Q_k^T]_{jj}

    using the CG conjugate-direction identity (Chantas 2008, Sec. IV):

        A^{-1} ≈ Σ_i  p_i p_i^T / (p_i^T A p_i)

    so that  var_k(j) ≈ Σ_i (Q_k p_i)_j² / (p_i^T A p_i).

    The Q_k p_i terms are already computed inside the matvec, so the
    variance accumulation adds essentially zero extra cost.

    Parameters
    ----------
    g            : (H, W)
    h_fft        : (H, W), complex
    mu_f_init    : (H, W)
    beta         : float
    gamma_maps   : list of (H, W), length P
    Q_fft        : list of (H, W) complex, length P
    Q_fft_sq     : list of (H, W) real, length P
    prior_weight : float  (1/P, typically 0.25 for P=4)
    cg_maxiter   : int
    cg_tol       : float

    Returns
    -------
    mu_f     : ndarray (H, W)
    var_maps : list of ndarray (H, W), length P  — per-pixel constrained
               variance  [Q_k R Q_k^T]_{jj}  (Eq. 3.8).
    """
    H, W = g.shape
    K = len(Q_fft)
    pw = prior_weight

    h_fft_sq = np.abs(h_fft) ** 2
    h_fft_conj = np.conj(h_fft)

    # ── RHS:  b = β H^T g  (Eq. 3.11) ────────────────────────────────
    rhs = (beta * np.real(ifft2(h_fft_conj * fft2(g)))).ravel()
    rhs_norm = np.linalg.norm(rhs)

    # ── Inline matvec  A·v  (Eq. 3.10) ───────────────────────────────
    #    Returns the product  AND  the K intermediate Q_k v images
    #    so that variance accumulation can reuse them at zero cost.
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

    # ── Spectral preconditioner  M⁻¹  (Eq. 60) ──────────────────────
    g_bar = np.array([float(np.mean(gm)) for gm in gamma_maps])
    denom = beta * h_fft_sq
    for k in range(K):
        denom = denom + pw * g_bar[k] * Q_fft_sq[k]
    Minv_spec = 1.0 / (denom + EPSILON)

    def _Minv(v_flat):
        v = v_flat.reshape((H, W))
        return np.real(ifft2(Minv_spec * fft2(v))).ravel()

    # ── Initial residual ──────────────────────────────────────────────
    x = mu_f_init.ravel().copy()
    Ax0, _ = _Av(x)
    r = rhs - Ax0
    z = _Minv(r)
    p = z.copy()
    rz = np.dot(r, z)

    var_maps = [np.zeros((H, W)) for _ in range(K)]

    # ── Main PCG loop ─────────────────────────────────────────────────
    for it in range(cg_maxiter):
        Ap, qk_p = _Av(p)
        pAp = np.dot(p, Ap)
        if pAp < EPSILON:
            break

        step = rz / pAp
        x += step * p
        r -= step * Ap

        # Accumulate constrained variance  (reuses Q_k p from matvec)
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
    r"""Constrained gamma update  (Chantas 2008, Eq. 3.14–3.15).

    Uses per-pixel constrained variance from
    q(ε_k) = N(Q_k m, Q_k R Q_k^T)   (Eq. 3.8):

        ⟨(ε_{k,j})²⟩ = (Q_k m)_j²  +  [Q_k R Q_k^T]_{jj}

    where [Q_k R Q_k^T]_{jj} comes from ``var_maps`` computed by
    ``solve_image_chantas2008_pcg``.

    Parameters
    ----------
    mu_f       : (H, W)
    Q_fft      : list of (H, W) complex, length P
    var_maps   : list of (H, W), length P
    a0_gamma, b0_gamma : float
    gamma_max  : float
    prior_weight : float  (1/P)

    Returns
    -------
    gamma_maps : list of (H, W), length P
    gamma_bar  : ndarray (P,)
    """
    K = len(Q_fft)
    gamma_maps: List[np.ndarray] = []
    gamma_bar = np.empty(K, dtype=np.float64)
    pw_half = prior_weight * 0.5
    a_tilde = a0_gamma + pw_half

    F_mu = fft2(mu_f)
    for k in range(K):
        Qk_mu = np.real(ifft2(Q_fft[k] * F_mu))
        E_eps_sq = Qk_mu ** 2 + var_maps[k]                           # per-pixel!

        b_tilde = b0_gamma + pw_half * E_eps_sq
        gmap = a_tilde / (b_tilde + EPSILON)
        np.clip(gmap, EPSILON, gamma_max, out=gmap)
        gamma_maps.append(gmap)
        gamma_bar[k] = float(np.mean(gmap))

    return gamma_maps, gamma_bar


# ═══════════════════════════════════════════════════════════════════════════
#  ARD pruning  (Tipping 2001; Tzikas 2009 Sec. III-C)
# ═══════════════════════════════════════════════════════════════════════════

def prune_ard(
    alpha: np.ndarray,
    Phi: np.ndarray,
    mu_w: np.ndarray,
    threshold: float = 1e10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Remove basis functions whose ARD precision exceeds *threshold*.

    When ⟨α_i⟩ > threshold, the corresponding weight w_i is effectively
    zero and the i-th RBF column is pruned from Φ (Tipping 2001, Sec. 3).

    Parameters
    ----------
    alpha     : ndarray (M,)
    Phi       : ndarray (L, M)
    mu_w      : ndarray (M,)
    threshold : float

    Returns
    -------
    alpha_new : ndarray (M',)
    Phi_new   : ndarray (L, M')
    mu_w_new  : ndarray (M',)
    keep_mask : ndarray (M,), bool
    """
    keep = alpha < threshold
    if np.all(keep):
        return alpha, Phi, mu_w, keep
    # Ensure at least one basis function survives
    if not np.any(keep):
        best = np.argmin(alpha)
        keep[best] = True
    return alpha[keep], Phi[:, keep], mu_w[keep], keep


