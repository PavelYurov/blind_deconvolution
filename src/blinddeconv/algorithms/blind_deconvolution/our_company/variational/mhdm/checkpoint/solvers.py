"""
Solvers for the MHDM (Multiscale Hierarchical Decomposition Method)
blind deconvolution algorithm.

Implements the core mathematical operations carried out entirely in the
Fourier domain:
    1. ``mhdm_initial``  — closed-form initialisation (Theorem 3.5 / Eq. 11–12
                           in [1]).
    2. ``mhdm_step``     — one MHDM increment via pointwise polynomial
                           root-finding (Algorithm 1 / Theorem 4.3 in [1]).
    3. ``blind_deconvolution_mhdm`` — outer loop with geometrically decaying
                           regularisation and discrepancy-principle stopping.

References
----------
[1] Wolf, T., Kindermann, S., Resmerita, E., Vese, L.
    "Applications of multiscale hierarchical decomposition to blind
    deconvolution." arXiv:2409.08734v5, 2025.
[2] Justen, L. "Blind Deconvolution: Theory, Regularization and
    Applications." PhD thesis — Sobolev Fourier weights (p. 110).
"""

import numpy as np
from typing import Tuple, List

from .utils import compute_fourier_weights, compute_conjugate_indices, complex_sign


# ===================================================================
# 1.  Initial step  (Theorem 3.5 / Eq. 11–12 in [1])
# ===================================================================

def mhdm_initial(
    f_four: np.ndarray,
    lambda_val: float,
    mu_val: float,
    r: float,
    s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute the initial Fourier-domain iterates (U_0, K_0) of the MHDM.

    For every frequency :math:`\xi` the closed-form minimisers of the
    decoupled penalty functional are (Theorem 3.5 in [1]):

    .. math::
        \hat U_0(\xi) = \sqrt{\max\!\Bigl(
            \sqrt{\tfrac{\mu}{\lambda}}\,
            \delta_\xi^{(s-r)/2}\,|\hat f(\xi)|
            \;-\; \mu\,\delta_\xi^{s},\; 0\Bigr)}
        \;\cdot\;\operatorname{sign}\!\bigl(\hat f(\xi)\bigr),

    .. math::
        \hat K_0(\xi) = \sqrt{\max\!\Bigl(
            \sqrt{\tfrac{\lambda}{\mu}}\,
            \delta_\xi^{(r-s)/2}\,|\hat f(\xi)|
            \;-\; \lambda\,\delta_\xi^{r},\; 0\Bigr)},

    with the DC normalisation
    :math:`\hat K_0(0)=1,\;\hat U_0(0)=\hat f(0)`.

    Parameters
    ----------
    f_four : ndarray, shape (m, n), complex
        2D DFT of the observed (blurred + noisy) image.
    lambda_val : float
        Image regularisation parameter :math:`\lambda`.
    mu_val : float
        Kernel regularisation parameter :math:`\mu`.
    r : float
        Sobolev exponent for the image penalty.
    s : float
        Sobolev exponent for the kernel penalty.

    Returns
    -------
    u_four : ndarray, shape (m, n), complex
        Fourier-domain image iterate.
    k_four : ndarray, shape (m, n), complex
        Fourier-domain kernel iterate.
    """
    m, n = f_four.shape
    delta = compute_fourier_weights(m, n)                       # (m, n)

    abs_f = np.abs(f_four)
    ratio_mu_lam = mu_val / lambda_val                          # μ / λ

    # --- Image estimate (Eq. 11) ---
    inner_u = (np.sqrt(ratio_mu_lam) * delta ** ((s - r) / 2.0) * abs_f
               - mu_val * delta ** s)
    u_four = np.sqrt(np.maximum(inner_u, 0.0)) * complex_sign(f_four)

    # --- Kernel estimate (Eq. 12) ---
    inner_k = (np.sqrt(1.0 / ratio_mu_lam) * delta ** ((r - s) / 2.0) * abs_f
               - lambda_val * delta ** r)
    k_four = np.sqrt(np.maximum(inner_k, 0.0))

    # DC normalisation: sum(k) = 1  ⇔  K̂(0) = 1;  Û(0) = f̂(0)
    k_four[0, 0] = 1.0
    u_four[0, 0] = f_four[0, 0]

    return u_four, k_four


# ===================================================================
# 2.  Single MHDM step  (Algorithm 1 / Theorem 4.3 in [1])
# ===================================================================

def _solve_single_frequency(
    u_n: complex,
    k_n: complex,
    f_val: complex,
    a_n: float,
    b_n: float,
    tol: float,
) -> Tuple[complex, complex]:
    r"""
    Solve the pointwise bilinear Tikhonov sub-problem for one frequency.

    Given the current Fourier iterates :math:`\hat U_n,\hat K_n` and the
    data :math:`\hat f`, the functional at frequency :math:`\xi` reads
    (Eq. 31 in [1]):

    .. math::
        J(p) = \frac{a_n}{|p|^2 + a_n}\,
               |\hat U_n\,p - \hat f|^2
             + b_n\,|p - \hat K_n|^2,

    where :math:`p = q + \hat K_n` is the *total* kernel coefficient and
    :math:`a_n = \lambda\,\delta_\xi^r`,
    :math:`b_n = \mu\,\delta_\xi^s`.

    Setting :math:`\partial J / \partial \bar p = 0` yields a degree-5
    polynomial in :math:`p` with **real** coefficients (after taking the
    real part, following the reference implementation).  All roots are
    found; the one minimising :math:`J` subject to
    :math:`\operatorname{Re}(p) \ge -\text{tol}` is selected.

    The image increment is recovered in closed form (Eq. 30 in [1]):

    .. math::
        \hat u = \frac{a_n\,\hat U_n
                       + \hat f\,\overline{p^*}}
                      {|p^*|^2 + a_n}
               - \hat U_n.

    Parameters
    ----------
    u_n, k_n, f_val : complex
        Current image / kernel iterates and data at this frequency.
    a_n, b_n : float
        Regularisation weights for image and kernel.
    tol : float
        Tolerance for admitting roots with slightly negative real part.

    Returns
    -------
    u_inc : complex
        Image Fourier increment.
    k_inc : complex
        Kernel Fourier increment.
    """
    # ------------------------------------------------------------------
    # Build the degree-5 polynomial whose roots are candidates for p
    # (stationarity condition, see Theorem 4.3 and reference MATLAB code).
    #
    # Polynomial coefficients (highest degree first):
    #   c5 = b_n
    #   c4 = -Re(k_n) * b_n
    #   c3 = 2 * a_n * b_n
    #   c2 = a_n * Re(f * conj(u_n)) - 2 * a_n * b_n * Re(k_n)
    #   c1 = a_n^2 * |u_n|^2 - a_n * |f|^2 + a_n^2 * b_n
    #   c0 = -a_n^2 * (Re(f * conj(u_n)) + b_n * Re(k_n))
    # ------------------------------------------------------------------
    fu_conj = np.real(f_val * np.conj(u_n))  # Re(f · ū_n)
    re_kn = np.real(k_n)
    abs_un2 = np.abs(u_n) ** 2
    abs_f2 = np.abs(f_val) ** 2

    c5 = b_n
    c4 = -re_kn * b_n
    c3 = 2.0 * a_n * b_n
    c2 = a_n * fu_conj - 2.0 * a_n * b_n * re_kn
    c1 = a_n ** 2 * abs_un2 - a_n * abs_f2 + a_n ** 2 * b_n
    c0 = -a_n ** 2 * (fu_conj + b_n * re_kn)

    coeffs = np.array([c5, c4, c3, c2, c1, c0], dtype=np.float64)

    # Solve the polynomial  (np.roots follows the same convention as
    # MATLAB's `roots`: highest-degree coefficient first).
    roots = np.roots(coeffs)

    # Shift roots to get kernel increments:  q = p - k_n
    q_candidates = roots - k_n

    # ------------------------------------------------------------------
    # Objective evaluation for every candidate
    # ------------------------------------------------------------------
    def objective(q: complex) -> float:
        p = q + k_n
        abs_p2 = np.abs(p) ** 2
        term1 = (a_n / (abs_p2 + a_n)) * np.abs(u_n * p - f_val) ** 2
        term2 = b_n * np.abs(q) ** 2
        return float(np.real(term1 + term2))

    best_q = q_candidates[0]
    best_obj = objective(best_q)

    for q in q_candidates[1:]:
        obj = objective(q)
        if obj <= best_obj and np.real(q) >= -tol:
            best_q = q
            best_obj = obj

    # If the initially selected candidate did not pass the tol check either,
    # attempt to replace it with any root passing the constraint.
    if np.real(best_q) < -tol:
        for q in q_candidates:
            if np.real(q) >= -tol:
                best_q = q
                best_obj = objective(q)
                break

    # ------------------------------------------------------------------
    # Image increment  (Eq. 30)
    # ------------------------------------------------------------------
    p_star = best_q + k_n
    abs_pstar2 = np.abs(p_star) ** 2
    u_inc = (a_n * u_n + f_val * np.conj(p_star)) / (abs_pstar2 + a_n) - u_n

    k_inc = best_q
    return u_inc, k_inc


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
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute one MHDM increment pair :math:`(u, k)` in Fourier space.

    For each non-self-conjugate frequency the bilinear sub-problem is
    solved via ``_solve_single_frequency`` (degree-5 polynomial);
    Hermitian symmetry is enforced afterwards (see Proposition 4.2 in [1]).

    The DC component is updated analytically:
    :math:`\hat k_{\mathrm{inc}}(0,0) = 1 - \hat K_n(0,0)`,
    :math:`\hat u_{\mathrm{inc}}(0,0) = \hat f(0,0) - \hat U_n(0,0)`.

    Parameters
    ----------
    u_n, k_n : ndarray, shape (m, n), complex
        Current cumulative Fourier iterates.
    f_four : ndarray, shape (m, n), complex
        DFT of the observed image.
    lambda_val, mu_val : float
        Regularisation parameters at the current scale.
    r, s : float
        Sobolev exponents for image / kernel.
    tol : float
        Tolerance for admitting roots with slightly negative real part.
    primary_idx : ndarray, shape (N, 2), int
        "Primary" 2-D indices (computed).
    conjugate_idx : ndarray, shape (N, 2), int
        Corresponding conjugate indices.

    Returns
    -------
    u_inc : ndarray, shape (m, n), complex
        Fourier-domain image increment.
    k_inc : ndarray, shape (m, n), complex
        Fourier-domain kernel increment.
    """
    m, n = f_four.shape
    delta = compute_fourier_weights(m, n)

    u_inc = np.zeros((m, n), dtype=np.complex128)
    k_inc = np.zeros((m, n), dtype=np.complex128)

    num_pairs = primary_idx.shape[0]

    for idx in range(num_pairs):
        pj, pl = primary_idx[idx]
        a_n = lambda_val * delta[pj, pl] ** r      # image weight
        b_n = mu_val * delta[pj, pl] ** s           # kernel weight

        u_inc_val, k_inc_val = _solve_single_frequency(
            u_n[pj, pl], k_n[pj, pl], f_four[pj, pl],
            a_n, b_n, tol,
        )

        # Primary index
        u_inc[pj, pl] = u_inc_val
        k_inc[pj, pl] = k_inc_val

        # Conjugate index — Hermitian symmetry (Proposition 4.2)
        cj, cl = conjugate_idx[idx]
        u_inc[cj, cl] = np.conj(u_inc_val)
        k_inc[cj, cl] = np.conj(k_inc_val)

    # DC component:  K̂ = 1 ⇒ increment = 1 − K̂_n(0);  Û = f̂ ⇒ increment = f̂ − Û_n
    k_inc[0, 0] = 1.0 - k_n[0, 0]
    u_inc[0, 0] = f_four[0, 0] - u_n[0, 0]

    return u_inc, k_inc


# ===================================================================
# 3.  Full blind-deconvolution loop  (Algorithm 2 in [1])
# ===================================================================

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
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], int,
           List[float]]:
    r"""
    Run the blind MHDM loop (Algorithm 2 in [1]).

    Starting from the initial step, regularisation parameters are reduced
    by a factor of 4 at each iteration:

    .. math::
        \lambda_n = \lambda_0 / 4^n, \qquad
        \mu_n     = \mu_0     / 4^n.

    The loop terminates when the :math:`L^2` residual falls below the
    stopping threshold (discrepancy principle) or the maximum number of
    iterations is reached.

    Parameters
    ----------
    f : ndarray, shape (m, n)
        Observed (blurred + noisy) image (spatial domain, [0, 1]).
    f_four : ndarray, shape (m, n), complex
        Pre-computed 2D DFT of *f*.
    lambda_0, mu_0 : float
        Initial regularisation parameters.
    r, s : float
        Sobolev exponents.
    tol : float
        Numerical tolerance for polynomial root selection.
    stopping : float
        Discrepancy-principle threshold  :math:`\tau \cdot \delta`.
    maxits : int
        Maximum number of MHDM iterations.
    primary_idx, conjugate_idx : ndarray
        Conjugate-pair index arrays (from ``compute_conjugate_indices``).
    verbose : bool
        Print iteration info.

    Returns
    -------
    u_end : ndarray, shape (m, n)
        Restored image (spatial domain).
    k_end : ndarray, shape (m, n)
        Estimated full-size PSF (spatial domain, fftshifted).
    u_four_list : list of ndarray
        Cumulative Fourier image iterates.
    k_four_list : list of ndarray
        Cumulative Fourier kernel iterates.
    its : int
        Number of iterations performed.
    residuals : list of float
        :math:`\|f - \mathcal{F}^{-1}[\hat U_n \hat K_n]\|_{L^2}` per step.
    """
    l2 = lambda arr: float(np.sqrt(np.sum(np.abs(arr) ** 2)))

    lam = lambda_0
    mu = mu_0

    # --- Step 0: initialisation (Theorem 3.5) ---
    u_four, k_four = mhdm_initial(f_four, lam, mu, r, s)

    u_four_list: List[np.ndarray] = [u_four.copy()]
    k_four_list: List[np.ndarray] = [k_four.copy()]

    residual = l2(f - np.real(np.fft.ifft2(u_four * k_four)))
    residuals: List[float] = [residual]
    its = 1

    if verbose:
        print(f"[MHDM] Iter 0  residual={residual:.6f}  "
              f"(stop={stopping:.6f})")

    # --- Iterative steps ---
    while residual > stopping and its <= maxits:
        mu /= 4.0
        lam /= 4.0

        u_inc, k_inc = mhdm_step(
            u_four_list[-1], k_four_list[-1], f_four,
            lam, mu, r, s, tol,
            primary_idx, conjugate_idx,
        )

        its += 1
        u_four_new = u_four_list[-1] + u_inc
        k_four_new = k_four_list[-1] + k_inc

        u_four_list.append(u_four_new)
        k_four_list.append(k_four_new)

        residual = l2(f - np.real(np.fft.ifft2(u_four_new * k_four_new)))
        residuals.append(residual)

        if verbose:
            print(f"[MHDM] Iter {its-1}  residual={residual:.6f}")

    # --- Spatial-domain outputs ---
    u_end = np.real(np.fft.ifft2(u_four_list[-1]))
    k_end = np.real(np.fft.ifft2(k_four_list[-1]))
    # fftshift so that the PSF peak is centred
    k_end = np.fft.fftshift(k_end)

    return u_end, k_end, u_four_list, k_four_list, its, residuals
