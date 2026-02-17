"""
Solvers for the MHDM (Multiscale Hierarchical Decomposition Method)
blind deconvolution algorithm.
"""

import numpy as np
from typing import Tuple, List

from .utils import compute_fourier_weights, complex_sign


def mhdm_initial(
    f_four: np.ndarray,
    lambda_val: float,
    mu_val: float,
    r: float,
    s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    f_four : ndarray, shape (m, n), complex128
        2D DFT of the observed blurred-and-noisy image.
    lambda_val : float
        Image regularisation parameter :math:`\lambda`.
    mu_val : float
        Kernel regularisation parameter :math:`\mu`.
    r, s : float
        Sobolev exponents for image (*r*) and kernel (*s*) penalties.
    """
    m, n = f_four.shape
    delta = compute_fourier_weights(m, n)

    abs_f = np.abs(f_four)
    ratio_mu_lam = mu_val / lambda_val                          

    inner_u = (np.sqrt(ratio_mu_lam) * delta ** ((s - r) / 2.0) * abs_f
               - mu_val * delta ** s)
    u_four = np.sqrt(np.maximum(inner_u, 0.0)) * complex_sign(f_four)

    inner_k = (np.sqrt(1.0 / ratio_mu_lam) * delta ** ((r - s) / 2.0) * abs_f
               - lambda_val * delta ** r)
    k_four = np.sqrt(np.maximum(inner_k, 0.0))

    k_four[0, 0] = 1.0
    u_four[0, 0] = f_four[0, 0]

    return u_four, k_four


def _solve_single_frequency(
    u_n: complex,
    k_n: complex,
    f_val: complex,
    a_n: float,
    b_n: float,
    tol: float,
) -> Tuple[complex, complex]:
    """
    Solve the pointwise bilinear Tikhonov sub-problem at one frequency.
    Parameters:
    u_n, k_n : complex
        Current cumulative image / kernel iterates at this frequency.
    f_val : complex
        DFT coefficient of the observation at this frequency.
    a_n : float
        Image regularisation weight`.
    b_n : float
        Kernel regularisation weight.
    tol : float
        Tolerance for admitting roots with slightly negative real part
        (handles numerical noise in the polynomial root-finding).
    Returns:
    u_inc : complex
        Fourier-domain image increment at this frequency.
    k_inc : complex
        Fourier-domain kernel increment at this frequency.
    """
    fu_conj = np.real(f_val * np.conj(u_n))     # Re( f_hat * conj(u_hat_n) )
    re_kn   = np.real(k_n)
    abs_un2 = np.abs(u_n) ** 2
    abs_f2  = np.abs(f_val) ** 2

    c5 = b_n
    c4 = -re_kn * b_n
    c3 = 2.0 * a_n * b_n
    c2 = a_n * fu_conj - 2.0 * a_n * b_n * re_kn
    c1 = a_n ** 2 * abs_un2 - a_n * abs_f2 + a_n ** 2 * b_n
    c0 = -a_n ** 2 * (fu_conj + b_n * re_kn)

    coeffs = np.array([c5, c4, c3, c2, c1, c0], dtype=np.float64)
    roots = np.roots(coeffs)
    q_candidates = roots - k_n

    def objective(q: complex) -> float:
        """Evaluate J at increment q (Eq. 31 in [1])."""
        p = q + k_n
        abs_p2 = np.abs(p) ** 2
        term1 = (a_n / (abs_p2 + a_n)) * np.abs(u_n * p - f_val) ** 2
        term2 = b_n * np.abs(q) ** 2
        return float(np.real(term1 + term2))

    best_q   = q_candidates[0]
    best_obj = objective(best_q)

    for q in q_candidates[1:]:
        obj = objective(q)
        if obj <= best_obj and np.real(q) >= -tol:
            best_q   = q
            best_obj = obj

    p_star     = best_q + k_n
    abs_pstar2 = np.abs(p_star) ** 2
    u_inc = (a_n * u_n + f_val * np.conj(p_star)) / (abs_pstar2 + a_n) - u_n

    return u_inc, best_q

def mhdm_step(
    u_n: np.ndarray,
    k_n: np.ndarray,
    f_four: np.ndarray,
    lambda_val: float,
    mu_val: float,
    r: float,
    s: float,
    tol: float,
    primary_idx: np.ndarray,
    conjugate_idx: np.ndarray,
    self_conj_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
    u_n, k_n : ndarray, shape (m, n), complex128
        Current cumulative Fourier iterates.
    f_four : ndarray, shape (m, n), complex128
        DFT of the observed image.
    lambda_val, mu_val : float
        Regularisation parameters at the current scale.
    r, s : float
        Sobolev exponents for image / kernel penalties.
    tol : float
        Tolerance for polynomial root selection.
    primary_idx : ndarray, shape (N_pair, 2), intp
        Primary 2D indices for conjugate pairs.
    conjugate_idx : ndarray, shape (N_pair, 2), intp
        Corresponding conjugate 2D indices.
    self_conj_idx : ndarray, shape (N_self, 2), intp
        Self-conjugate (Nyquist) indices, excluding DC.

    Returns:
    u_inc : ndarray, shape (m, n), complex128
        Fourier-domain image increment.
    k_inc : ndarray, shape (m, n), complex128
        Fourier-domain kernel increment.
    """
    m, n = f_four.shape
    delta = compute_fourier_weights(m, n)

    u_inc = np.zeros((m, n), dtype=np.complex128)
    k_inc = np.zeros((m, n), dtype=np.complex128)

    num_pairs = primary_idx.shape[0]
    for idx in range(num_pairs):
        pj, pl = primary_idx[idx]
        a_n = lambda_val * delta[pj, pl] ** r      
        b_n = mu_val    * delta[pj, pl] ** s         

        u_val, k_val = _solve_single_frequency(
            u_n[pj, pl], k_n[pj, pl], f_four[pj, pl],
            a_n, b_n, tol,
        )

        u_inc[pj, pl] = u_val
        k_inc[pj, pl] = k_val

        cj, cl = conjugate_idx[idx]
        u_inc[cj, cl] = np.conj(u_val)
        k_inc[cj, cl] = np.conj(k_val)

    num_self = self_conj_idx.shape[0]
    for idx in range(num_self):
        sj, sl = self_conj_idx[idx]
        a_n = lambda_val * delta[sj, sl] ** r
        b_n = mu_val    * delta[sj, sl] ** s

        u_val, k_val = _solve_single_frequency(
            u_n[sj, sl], k_n[sj, sl], f_four[sj, sl],
            a_n, b_n, tol,
        )

        u_inc[sj, sl] = np.real(u_val)
        k_inc[sj, sl] = np.real(k_val)

    k_inc[0, 0] = 1.0 - k_n[0, 0]
    u_inc[0, 0] = f_four[0, 0] - u_n[0, 0]

    return u_inc, k_inc


def blind_deconvolution_mhdm(
    f: np.ndarray,
    f_four: np.ndarray,
    lambda_0: float,
    mu_0: float,
    r: float,
    s: float,
    tol: float,
    stopping: float,
    maxits: int,
    primary_idx: np.ndarray,
    conjugate_idx: np.ndarray,
    self_conj_idx: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray],
           int, List[float]]:
    """
    Run the full blind MHDM loop (Algorithm 2 in [1]).

    Parameters:
    f : ndarray, shape (m, n)
        Observed blurred-and-noisy image (spatial domain, in [0, 1]).
    f_four : ndarray, shape (m, n), complex128
        Pre-computed 2D DFT of *f*.
    lambda_0, mu_0 : float
        Initial regularisation parameters.
    r, s : float
        Sobolev exponents.
    tol : float
        Numerical tolerance for polynomial root selection.
    stopping : float
        Discrepancy-principle threshold.
    maxits : int
        Maximum number of MHDM iterations.
    primary_idx, conjugate_idx : ndarray
        Conjugate-pair index arrays.
    self_conj_idx : ndarray
        Self-conjugate (Nyquist) index array.
    verbose : bool
        Print per-iteration diagnostics.

    Returns:
    u_end : ndarray, shape (m, n)
        Restored image (spatial domain).
    k_end : ndarray, shape (m, n)
        Estimated full-size PSF (spatial domain, fftshift-ed).
    u_four_list : list of ndarray
        Cumulative Fourier image iterates.
    k_four_list : list of ndarray
        Cumulative Fourier kernel iterates.
    its : int
        Total number of iterations performed.
    residuals : list of float
    """
    l2_norm = lambda arr: float(np.sqrt(np.sum(np.abs(arr) ** 2)))

    lam = lambda_0
    mu  = mu_0

    u_four, k_four = mhdm_initial(f_four, lam, mu, r, s)

    u_four_list: List[np.ndarray] = [u_four.copy()]
    k_four_list: List[np.ndarray] = [k_four.copy()]

    residual = l2_norm(f - np.real(np.fft.ifft2(u_four * k_four)))
    residuals: List[float] = [residual]
    its = 1

    if verbose:
        print(f"[MHDM] Iter 0   residual={residual:.6f}   "
              f"stopping={stopping:.6f}")

    while residual > stopping and its <= maxits:
        lam /= 4.0
        mu  /= 4.0

        u_inc, k_inc = mhdm_step(
            u_four_list[-1], k_four_list[-1], f_four,
            lam, mu, r, s, tol,
            primary_idx, conjugate_idx, self_conj_idx,
        )

        its += 1
        u_four_new = u_four_list[-1] + u_inc
        k_four_new = k_four_list[-1] + k_inc

        u_four_list.append(u_four_new)
        k_four_list.append(k_four_new)

        residual = l2_norm(f - np.real(np.fft.ifft2(u_four_new * k_four_new)))
        residuals.append(residual)

        if verbose:
            print(f"[MHDM] Iter {its - 1}   residual={residual:.6f}")

    u_end = np.real(np.fft.ifft2(u_four_list[-1]))

    k_end = np.real(np.fft.ifft2(k_four_list[-1]))
    k_end = np.fft.fftshift(k_end)

    return u_end, k_end, u_four_list, k_four_list, its, residuals
