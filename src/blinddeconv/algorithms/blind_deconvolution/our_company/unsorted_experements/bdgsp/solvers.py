"""
solvers.py — Core VB solvers for BDGSP (Babacan et al., ECCV 2012).

Implements the three alternating updates of Algorithm 1:

    (i)   image  :  conjugate-gradient solve for E[x_γ]
    (ii)  kernel :  constrained quadratic program for k
    (iii) final  :  one CG solve for the restored image  (eq. 24)

All operators are expressed through FFT-based circular convolution so
that boundaries behave consistently with the variational derivation.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from numpy.fft import fft2, ifft2

from .utils import apply_K, apply_KT, apply_filter, project_simplex, psf2otf


# ════════════════════════════════════════════════════════════════════════════
#  Conjugate gradient  (matrix-free)
# ════════════════════════════════════════════════════════════════════════════
def _cg(
    apply_A,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> np.ndarray:
    """
    Minimal conjugate-gradient solver for a symmetric-positive-definite
    linear operator ``apply_A`` (``apply_A(x) -> A x``).

    The operator must accept and return NumPy arrays with the same shape
    as ``b``.  Convergence is measured by the relative residual norm.
    """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - apply_A(x)
    p = r.copy()
    rs = float(np.sum(r * r))
    b_norm = float(np.sqrt(np.sum(b * b))) + 1e-30
    for _ in range(max_iter):
        Ap = apply_A(p)
        denom = float(np.sum(p * Ap)) + 1e-30
        alpha = rs / denom
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.sum(r * r))
        if np.sqrt(rs_new) / b_norm < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


# ════════════════════════════════════════════════════════════════════════════
#  (i) Image update  —  E[x_γ]  for one γ
# ════════════════════════════════════════════════════════════════════════════
def solve_x_gamma(
    y_gamma: np.ndarray,
    k: np.ndarray,
    xi_gamma: np.ndarray,
    sigma2: float,
    x0: np.ndarray | None = None,
    tol: float = 1e-4,
    max_iter: int = 30,
) -> np.ndarray:
    """
    Solve  ``((1/σ²) T_kᵀ T_k + diag(ξ_γ)) x = (1/σ²) T_kᵀ y_γ``
    via conjugate gradient.

    Returns the posterior mean estimate ``E[x_γ]`` (eq. 18).
    """
    K = psf2otf(k, y_gamma.shape)
    Kc = np.conj(K)
    inv_s2 = 1.0 / sigma2

    def A(x: np.ndarray) -> np.ndarray:
        Fx = fft2(x)
        Tx = np.real(ifft2(Kc * (K * Fx)))  # T_kᵀ T_k x
        return inv_s2 * Tx + xi_gamma * x

    b = inv_s2 * np.real(ifft2(Kc * fft2(y_gamma)))
    return _cg(A, b, x0=x0, tol=tol, max_iter=max_iter)


def diag_Cx_approx(k: np.ndarray, xi_gamma: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Diagonal approximation of the posterior covariance

        C_xγ(i, i) ≈ 1 / ( ‖k‖² / σ²  +  ξ_γ(i) ).

    This is the element-wise inverse of the diagonal of ``C_xγ⁻¹`` as
    advocated in the paper (Section 3.1, last paragraph).
    """
    k2 = float(np.sum(k ** 2))
    return 1.0 / (k2 / sigma2 + xi_gamma)


# ════════════════════════════════════════════════════════════════════════════
#  (ii) Kernel update  —  simplex-constrained quadratic program
# ════════════════════════════════════════════════════════════════════════════
def _apply_kernel_A(
    k: np.ndarray,
    x_list: Sequence[np.ndarray],
    trace_corr: float,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Apply the quadratic coefficient matrix of eq. (15) to a kernel ``k``.

    With the diagonal covariance approximation the action reduces to

        A_k k  =  Σ_γ  T_{x_γ}ᵀ T_{x_γ} k  +  trace_corr · k

    where ``trace_corr = Σ_γ Σ_j C_xγ(j, j)``.  The ``T_{x_γ}ᵀ T_{x_γ}``
    term is evaluated via FFT on the full image grid and then cropped to
    the kernel support.
    """
    kh, kw = k.shape
    # Embed k in the image-sized grid (upper-left, centre-shifted).
    k_full = np.zeros(image_shape, dtype=np.float64)
    k_full[:kh, :kw] = k
    k_full = np.roll(k_full, -(kh // 2), axis=0)
    k_full = np.roll(k_full, -(kw // 2), axis=1)

    out = np.zeros_like(k_full)
    for x in x_list:
        Xf = fft2(x)
        out += np.real(ifft2(np.conj(Xf) * (Xf * fft2(k_full))))

    # Inverse of the centre shift, then restrict to kernel support.
    out = np.roll(out, kh // 2, axis=0)
    out = np.roll(out, kw // 2, axis=1)
    out = out[:kh, :kw]
    return out + trace_corr * k


def _compute_kernel_b(
    x_list: Sequence[np.ndarray],
    y_list: Sequence[np.ndarray],
    kernel_shape: Tuple[int, int],
) -> np.ndarray:
    """Right-hand side of eq. (16): ``Σ_γ T_{x_γ}ᵀ y_γ`` cropped to kernel."""
    kh, kw = kernel_shape
    image_shape = x_list[0].shape
    acc = np.zeros(image_shape, dtype=np.float64)
    for x, y in zip(x_list, y_list):
        acc += np.real(ifft2(np.conj(fft2(x)) * fft2(y)))
    acc = np.roll(acc, kh // 2, axis=0)
    acc = np.roll(acc, kw // 2, axis=1)
    return acc[:kh, :kw].copy()


def estimate_kernel(
    x_list: Sequence[np.ndarray],
    cx_diag_list: Sequence[np.ndarray],
    y_list: Sequence[np.ndarray],
    k_init: np.ndarray,
    num_iter: int = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Solve the constrained QP for the blur kernel (eq. 14):

        k* = argmin_k  kᵀ A_k k − 2 kᵀ b_k
             s.t.     k ≥ 0,   Σ k = 1

    Uses FISTA-style accelerated projected gradient; at every step the
    iterate is projected onto the non-negative simplex.  Note that the
    noise-variance factor  1/σ²  is common to both  A_k  and  b_k  and
    therefore cancels — we drop it for numerical convenience.
    """
    kernel_shape = k_init.shape
    image_shape = x_list[0].shape
    trace_corr = float(sum(c.sum() for c in cx_diag_list))
    b = _compute_kernel_b(x_list, y_list, kernel_shape)

    # Lipschitz constant of the gradient of ``½ kᵀA_k k − kᵀb``.  A tight
    # estimate is essential: the naive bound ``max |fft(x)|²`` overshoots
    # by many orders of magnitude for natural-sized images (dominated by
    # the DC component), which makes the projected-gradient step ``1/L``
    # vanishingly small and leaves the kernel stuck at its initialisation.
    # We instead run a few power iterations on ``A_k`` restricted to the
    # kernel support.
    v = np.random.RandomState(0).rand(*kernel_shape)
    v = v / (np.linalg.norm(v) + 1e-30)
    for _ in range(20):
        Av = _apply_kernel_A(v, x_list, trace_corr, image_shape)
        nrm = np.linalg.norm(Av)
        if nrm < 1e-30:
            break
        v = Av / nrm
    L = float(np.sum(v * _apply_kernel_A(v, x_list, trace_corr, image_shape))) + 1e-8
    L = max(L, 1e-8) * 1.05  # small safety margin

    k = project_simplex(k_init.copy())
    z = k.copy()
    t = 1.0
    prev_obj = np.inf
    for _ in range(num_iter):
        grad = _apply_kernel_A(z, x_list, trace_corr, image_shape) - b
        k_new = project_simplex(z - grad / L)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = k_new + ((t - 1.0) / t_new) * (k_new - k)
        # Monotonic fallback: drop acceleration if objective worsens.
        Ak = _apply_kernel_A(k_new, x_list, trace_corr, image_shape)
        obj = float(np.sum(k_new * Ak) - 2.0 * np.sum(k_new * b))
        if obj > prev_obj and prev_obj != np.inf:
            z = k_new.copy()
            t_new = 1.0
        if abs(prev_obj - obj) < tol * max(1.0, abs(prev_obj)):
            k = k_new
            break
        prev_obj = obj
        k = k_new
        t = t_new
    return project_simplex(k)


# ════════════════════════════════════════════════════════════════════════════
#  (iii) Final image  —  eq. (24)
# ════════════════════════════════════════════════════════════════════════════
def solve_final_image(
    y: np.ndarray,
    k: np.ndarray,
    xi_list: Sequence[np.ndarray],
    filters: Sequence[np.ndarray],
    sigma2: float,
    x0: np.ndarray | None = None,
    tol: float = 1e-5,
    max_iter: int = 200,
) -> np.ndarray:
    """
    Solve the final non-blind deconvolution system

        ( T_kᵀ T_k + σ² Σ_γ T_{f_γ}ᵀ diag(ξ_γ) T_{f_γ} ) x̂  =  T_kᵀ y

    (eq. 24) via conjugate gradient.  The ``ξ_γ`` arrays carry all of the
    sparsity-preserving regularisation learned during the VB iteration.
    """
    K = psf2otf(k, y.shape)
    Kc = np.conj(K)
    Fs = [psf2otf(f, y.shape) for f in filters]
    Fcs = [np.conj(F) for F in Fs]

    def A(x: np.ndarray) -> np.ndarray:
        Fx = fft2(x)
        Tx = np.real(ifft2(Kc * (K * Fx)))
        acc = Tx
        for F, Fc, xi in zip(Fs, Fcs, xi_list):
            fx = np.real(ifft2(F * Fx))
            acc = acc + sigma2 * np.real(ifft2(Fc * fft2(xi * fx)))
        return acc

    b = np.real(ifft2(Kc * fft2(y)))
    return _cg(A, b, x0=x0, tol=tol, max_iter=max_iter)
