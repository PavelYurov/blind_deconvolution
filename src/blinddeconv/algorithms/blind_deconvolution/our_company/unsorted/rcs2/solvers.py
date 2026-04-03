"""
solvers.py

Core variational-Bayesian inference engine for the Fergus et al.
(SIGGRAPH 2006) blind deconvolution algorithm, ported from MATLAB.

This file contains:
  - Ensemble data accessors (get/put for x and lambda)
  - Distribution moment computation (rectified5)
  - FFT-based blind-deconvolution error evaluation (train_blind_deconv)
  - Evidence evaluation and gradient computation (train_ensemble_evidence6)
  - Main VB optimization loop (train_ensemble_main6)
  - Multi-scale initialization (initialize_parameters2)
  - Inter-scale upsampling (move_level)
  - Post-inference Richardson-Lucy deconvolution (fiddle_lucy3, fiddle_lucy4)

Reference:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Transactions on Graphics (SIGGRAPH), 2006.

    J. Miskin, D. J. C. MacKay: "Ensemble Learning for Blind Image
    Separation and Deconvolution", Adv. in ICA, Springer-Verlag, 2000.
"""

import copy
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.special import gammaln
from typing import Tuple, Optional, Dict, List, Any

from . import utils


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble data accessors
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble_get(c: int, dimensions: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Extract class *c* (0-based) data from flat ensemble vector *x*.

    Parameters
    ----------
    c : int
        Class index (0-based; MATLAB code uses 1-based).
    dimensions : np.ndarray, shape (num_classes, >=2)
        Each row: [n_rows, n_cols, n_components, prior_type, ...].
    x : np.ndarray, 1-D
        Flat vector holding data for all classes concatenated.

    Returns
    -------
    np.ndarray, shape (dimensions[c, 0], dimensions[c, 1])
    """
    start = int(np.sum(dimensions[:c, 0] * dimensions[:c, 1])) if c > 0 else 0
    n = int(dimensions[c, 0] * dimensions[c, 1])
    return x[start:start + n].reshape(int(dimensions[c, 0]), int(dimensions[c, 1]))


def train_ensemble_put(c: int, dimensions: np.ndarray, x: np.ndarray,
                       cx: np.ndarray) -> np.ndarray:
    """
    Replace class *c* (0-based) data in flat ensemble vector *x* with *cx*.

    Parameters are symmetric with :func:`train_ensemble_get`.  Returns the
    modified *x* (also modifies in-place for efficiency).
    """
    start = int(np.sum(dimensions[:c, 0] * dimensions[:c, 1])) if c > 0 else 0
    n = int(dimensions[c, 0] * dimensions[c, 1])
    x[start:start + n] = cx.ravel()
    return x


def train_ensemble_get_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray) -> np.ndarray:
    """
    Extract mixture-weight data for class *c* (0-based) from
    *log_lambda_x*.

    Returns shape (dimensions[c,0], dimensions[c,1], dimensions[c,2]).
    """
    if c > 0:
        start = int(np.sum(np.prod(dimensions[:c, 0:3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c, 0:3]))
    return log_lambda_x[start:start + n].reshape(
        int(dimensions[c, 0]), int(dimensions[c, 1]), int(dimensions[c, 2]))


def train_ensemble_put_lambda(c: int, dimensions: np.ndarray,
                              log_lambda_x: np.ndarray,
                              c_log_lambda_x: np.ndarray) -> np.ndarray:
    """
    Replace mixture-weight data for class *c* (0-based) in *log_lambda_x*.
    """
    if c > 0:
        start = int(np.sum(np.prod(dimensions[:c, 0:3], axis=1)))
    else:
        start = 0
    n = int(np.prod(dimensions[c, 0:3]))
    log_lambda_x[start:start + n] = c_log_lambda_x.ravel()
    return log_lambda_x


# ─────────────────────────────────────────────────────────────────────────────
# Distribution moment computation
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble_rectified5(
    x1: np.ndarray, x2: np.ndarray, dist_type: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate expectations under ensemble distributions.

    Returns
    -------
    Hx  : <log Q(x)> minus any constants from P(x)
    mx  : <x>
    mx2 : <x^2>

    Parameters
    ----------
    x1, x2 : np.ndarray
        Distribution parameters.
    dist_type : int
        0 — Gaussian
        1 — Laplacian (posterior is rectified Gaussian)
        2 — Rectified Gaussian
        3 — Discrete (1, -1)
        4 — Laplacian prior (two rectified Gaussians, buggy in MATLAB)
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)

    if dist_type == 0:
        # Gaussian
        mx = x1 / x2
        mx2 = x1 ** 2 / x2 ** 2 + 1.0 / x2
        Hx = -0.5 + 0.5 * np.log(x2)

    elif dist_type in (1, 2, 4):
        # Laplacian / Rectified Gaussian / type-4
        t = -x1 / np.sqrt(2.0 * x2)
        erf_table = utils.erfcx(t)

        small = t <= 25
        large = ~small

        mx = np.empty_like(x1)
        mx2 = np.empty_like(x1)
        Hx = np.empty_like(x1)

        # --- small t branch ---
        mx[small] = (x1[small] / x2[small]
                      + np.sqrt(2.0 / (np.pi * x2[small]))
                      / erf_table[small])
        mx2[small] = (x1[small] ** 2 / x2[small] ** 2
                       + 1.0 / x2[small]
                       + 2.0 * x1[small] / x2[small]
                       / np.sqrt(2.0 * np.pi * x2[small])
                       / erf_table[small])

        t_clamped = np.minimum(t, 25.0)
        # Hx for t < 25
        mask_Hx_small = t < 25
        Hx[mask_Hx_small] = (
            -np.log(erfc_safe(t_clamped[mask_Hx_small]))
            + 0.5 * np.log(2.0 * x2[mask_Hx_small] / np.pi)
            - 0.5
            + x1[mask_Hx_small]
            / np.sqrt(2.0 * np.pi * x2[mask_Hx_small])
            / erf_table[mask_Hx_small]
        )

        # --- large t branch ---
        mx[large] = (-1.0 / x1[large]
                      + 2.0 * x2[large] * x1[large] ** (-3)
                      - 10.0 * x2[large] ** 2 * x1[large] ** (-5))
        mx2[large] = (2.0 * x1[large] ** (-2)
                       - 10.0 * x2[large] * x1[large] ** (-4)
                       + 74.0 * x2[large] ** 2 * x1[large] ** (-6))

        # Hx for t >= 25
        mask_Hx_large = t >= 25
        Hx[mask_Hx_large] = (
            np.log(-x1[mask_Hx_large])
            - 1.0
            + 2.0 * x2[mask_Hx_large] * x1[mask_Hx_large] ** (-2)
            - 15.0 * x2[mask_Hx_large] ** 2
            * x1[mask_Hx_large] ** (-4) / 2.0
            + 148.0 * x2[mask_Hx_large] ** 3
            * x1[mask_Hx_large] ** (-6) / 3.0
        )

        # type 2 (rectified Gaussian) and type 4 add an extra constant
        if dist_type in (2, 4):
            Hx = Hx + 0.5 * np.log(np.pi / 2.0)

    elif dist_type == 3:
        # Discrete (1, -1)
        mx = np.tanh(x1)
        mx2 = np.ones_like(x1)
        Hx = (x1 * mx - np.abs(x1)
              - np.log(1.0 + np.exp(-2.0 * np.abs(x1)))
              + np.log(2.0))
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return Hx, mx, mx2


def erfc_safe(x: np.ndarray) -> np.ndarray:
    """erfc(x) clamped away from zero to avoid log(0)."""
    from scipy.special import erfc
    val = erfc(x)
    return np.maximum(val, np.finfo(np.float64).tiny)


# ─────────────────────────────────────────────────────────────────────────────
# FFT-based blind-deconvolution error evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_blind_deconv(
    dimensions: np.ndarray,
    ensemble: dict,
    D: np.ndarray,
    Dp: np.ndarray,
    I: int, J: int,
    K: int, L: int,
    M: int, N: int,
    *args
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Find the optimal distribution parameters for a deconvolution problem.

    The data D (I×J) is formed from convolution of filter me (K×L) and
    source mx (M×N), observed through binary mask Dp.

    Returns
    -------
    x1, x2 : np.ndarray (flat vectors)
        Optimal distribution parameters [mu, e(:), x(:)].
    error : float
        Reconstruction error.
    data_points : float
        Number of observed data points (sum of Dp).
    """
    # Copy expectations
    mmu = train_ensemble_get(0, dimensions, ensemble['mx']).ravel()
    me = train_ensemble_get(1, dimensions, ensemble['mx']).reshape(K, L)
    mx = train_ensemble_get(2, dimensions, ensemble['mx']).reshape(M, N)
    mmu2 = train_ensemble_get(0, dimensions, ensemble['mx2']).ravel()
    me2 = train_ensemble_get(1, dimensions, ensemble['mx2']).reshape(K, L)
    mx2 = train_ensemble_get(2, dimensions, ensemble['mx2']).reshape(M, N)

    mmu_val = float(mmu[0]) if mmu.size > 0 else 0.0
    mmu2_val = float(mmu2[0]) if mmu2.size > 0 else 0.0

    # Useful FFTs
    fft_mx = fft2(mx, s=(I, J))
    fft_mx2 = fft2(mx2, s=(I, J))
    fft_mx3 = fft2(mx ** 2, s=(I, J))
    fft_me = fft2(me, s=(I, J))
    fft_me2 = fft2(me2, s=(I, J))
    fft_me3 = fft2(me ** 2, s=(I, J))

    fft_Dp = fft2(Dp, s=(I, J))
    mD = np.real(ifft2(fft_mx * fft_me))
    fft_err = fft2(Dp * (D - mD - mmu_val), s=(I, J))

    # Reconstruction error
    data_points = float(np.sum(Dp))
    error = float(
        np.sum(Dp * ((D - mD - mmu_val) ** 2
                      + np.real(ifft2(fft_me2 * fft_mx2
                                       - fft_me3 * fft_mx3))))
        + data_points * (mmu2_val - mmu_val ** 2)
    )

    # Optimal distributions — mu
    mu1 = float(np.sum(Dp * (D - mD)))
    mu2 = data_points

    # Optimal distributions — blur e
    e1 = np.real(ifft2(fft_err * np.conj(fft_mx)))
    corr = np.real(ifft2(fft_Dp * np.conj(fft_mx3)))
    e1[:K, :L] = e1[:K, :L] + me * corr[:K, :L]
    e2 = np.real(ifft2(fft_Dp * np.conj(fft_mx2)))

    # Optimal distributions — image x
    x1_img = np.real(ifft2(fft_err * np.conj(fft_me)))
    corr2 = np.real(ifft2(fft_Dp * np.conj(fft_me3)))
    x1_img[:M, :N] = x1_img[:M, :N] + mx * corr2[:M, :N]
    x2_img = np.real(ifft2(fft_Dp * np.conj(fft_me2)))

    # Pack into flat vectors: [mu, blur, image]
    x1_out = np.concatenate([
        np.array([mu1]),
        e1[:K, :L].ravel(),
        x1_img[:M, :N].ravel()
    ])
    x2_out = np.concatenate([
        np.array([mu2]),
        e2[:K, :L].ravel(),
        x2_img[:M, :N].ravel()
    ])

    return x1_out, x2_out, error, data_points


# ─────────────────────────────────────────────────────────────────────────────
# Evidence evaluation and gradient computation
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble_evidence6(
    step_len: float,
    dimensions: np.ndarray,
    opt_func,  # unused string in MATLAB; kept for interface compat
    ensemble: dict,
    direction: dict,
    state: int,
    D: np.ndarray,
    Dp: np.ndarray,
    I: int, J: int,
    K: int, L: int,
    M: int, N: int,
    priors: Any,
    fft_mode: int,
    blur_mask: np.ndarray,
    image_mask: np.ndarray,
) -> Tuple[dict, dict]:
    """
    Evaluate evidence for the current ensemble after stepping along *direction*.

    Parameters
    ----------
    step_len : float
        Step size along the search direction.
    state : int
        1 — don't update priors or noise
        2 — don't update noise
        3 — update all
    priors : object with fields .pi (array) and .gamma (array)
        Manual prior override structure; may be None.

    Returns
    -------
    ensemble : dict  (updated)
    grad : dict  (search direction for next step)
    """

    # Step to test point
    ensemble['x1'] = ensemble['x1'] + step_len * direction['x1']
    ensemble['x2'] = np.abs(ensemble['x2'] + step_len * direction['x2'])
    ensemble['b_x_2'] = np.abs(ensemble['b_x_2'] + step_len * direction['b_x_2'])
    ensemble['ba_x_2'] = np.abs(ensemble['ba_x_2'] + step_len * direction['ba_x_2'])
    ensemble['pi_x_2'] = np.abs(ensemble['pi_x_2'] + step_len * direction['pi_x_2'])

    # Make a copy of the direction
    grad = {k: v.copy() if isinstance(v, np.ndarray) else v
            for k, v in direction.items()}

    # Check for valid ensemble
    if (np.any(ensemble['x2'] < 0) or np.any(ensemble['b_x_2'] < 0)
            or np.any(ensemble['ba_x_2'] < 0) or np.any(ensemble['pi_x_2'] < 0)):
        ensemble['D_val'] = np.inf
        grad['x1'][:] = np.nan
        grad['x2'][:] = np.nan
        grad['b_x_2'][:] = np.nan
        grad['ba_x_2'][:] = np.nan
        grad['pi_x_2'][:] = np.nan
        return ensemble, grad

    num_classes = dimensions.shape[0]
    ptr = 0

    for c in range(num_classes):
        dim_rows = int(dimensions[c, 0])
        dim_cols = int(dimensions[c, 1])
        dim_comp = int(dimensions[c, 2])
        dim_type = int(dimensions[c, 3])

        # Extract and update expectations
        cx1 = train_ensemble_get(c, dimensions, ensemble['x1'])
        cx2 = train_ensemble_get(c, dimensions, ensemble['x2'])
        cHx, cmx, cmx2 = train_ensemble_rectified5(cx1, cx2, dim_type)

        # Prior parameters for this class
        c_pi_x_2 = ensemble['pi_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)
        c_b_x_2 = ensemble['b_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)
        c_ba_x_2 = ensemble['ba_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)
        c_log_lambda_x = np.zeros((dim_rows, dim_cols, dim_comp))

        if dim_comp > 1:
            if dim_type in (0, 2):
                # Gaussian or Rectified Gaussian prior
                for alpha in range(dim_comp):
                    for k in range(dim_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + 0.5 * np.log(c_ba_x_2[k, alpha])
                            - 0.25 / c_b_x_2[k, alpha]
                            - 0.5 * cmx2[k, :] * c_ba_x_2[k, alpha]
                        )

            elif dim_type == 1:
                # Exponential prior
                for alpha in range(dim_comp):
                    for k in range(dim_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + np.log(c_ba_x_2[k, alpha])
                            - 0.5 / c_b_x_2[k, alpha]
                            - cmx[k, :] * c_ba_x_2[k, alpha]
                        )

            elif dim_type == 3:
                # Discrete prior — no learning
                pass

            elif dim_type == 4:
                # Exponential prior variant
                for alpha in range(dim_comp):
                    for k in range(dim_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + 0.5 * np.log(c_ba_x_2[k, alpha])
                            - 0.25 / c_b_x_2[k, alpha]
                            - 0.5 * np.abs(cmx[k, :]) * c_ba_x_2[k, alpha]
                        )

            # Normalise c_log_lambda_x
            max_c = np.max(c_log_lambda_x, axis=2, keepdims=True)
            c_log_lambda_x = c_log_lambda_x - max_c
            log_sum = np.log(np.sum(np.exp(c_log_lambda_x), axis=2, keepdims=True))
            c_log_lambda_x = c_log_lambda_x - log_sum

        # ── Optimal prior parameters ──
        exp_lambda = np.exp(c_log_lambda_x)

        if dim_type in (0, 2):
            # (Rectified) Gaussian prior
            sum_exp_lambda = np.sum(exp_lambda, axis=1).reshape(dim_rows, dim_comp)
            opt_c_b_x_2 = ensemble['b_x'] + sum_exp_lambda / 2.0
            opt_c_pi_x_2 = ensemble['pi_x'] + sum_exp_lambda
            opt_c_ba_x_2 = np.zeros((dim_rows, dim_comp))
            for alpha in range(dim_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ensemble['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx2, axis=1) / 2.0
                )

        elif dim_type == 1:
            # Exponential prior
            sum_exp_lambda = np.sum(exp_lambda, axis=1).reshape(dim_rows, dim_comp)
            opt_c_b_x_2 = ensemble['b_x'] + sum_exp_lambda
            opt_c_pi_x_2 = ensemble['pi_x'] + sum_exp_lambda
            opt_c_ba_x_2 = np.zeros((dim_rows, dim_comp))
            for alpha in range(dim_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ensemble['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx, axis=1)
                )

        elif dim_type in (3, 4):
            # Discrete / type-4
            opt_c_b_x_2 = ensemble['b_x'] * np.ones((dim_rows, dim_comp))
            opt_c_ba_x_2 = (ensemble['b_x'] / ensemble['a_x']) * np.ones((dim_rows, dim_comp))
            opt_c_pi_x_2 = ensemble['pi_x'] * np.ones((dim_rows, dim_comp))

        elif dim_type == 5:
            # Laplacian (same as exp)
            sum_exp_lambda = np.sum(exp_lambda, axis=1).reshape(dim_rows, dim_comp)
            opt_c_b_x_2 = ensemble['b_x'] + sum_exp_lambda
            opt_c_pi_x_2 = ensemble['pi_x'] + sum_exp_lambda
            opt_c_ba_x_2 = np.zeros((dim_rows, dim_comp))
            for alpha in range(dim_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ensemble['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx, axis=1)
                )
        else:
            opt_c_b_x_2 = ensemble['b_x'] * np.ones((dim_rows, dim_comp))
            opt_c_ba_x_2 = (ensemble['b_x'] / ensemble['a_x']) * np.ones((dim_rows, dim_comp))
            opt_c_pi_x_2 = ensemble['pi_x'] * np.ones((dim_rows, dim_comp))

        # ── Manual override for priors ──
        if int(dimensions[c, 4]) > 0:
            if c == 2:  # Image prior (0-based: class 2 = image)
                if priors is not None and hasattr(priors, 'pi'):
                    opt_c_pi_x_2 = priors.pi[:dim_rows, :] * 1e3
                    opt_c_ba_x_2 = priors.gamma[:dim_rows, :]
                    opt_c_b_x_2 = np.ones((dim_rows, dim_comp)) * 1e-3
            elif c == 1:  # Blur prior
                opt_c_ba_x_2 = np.array([[5.1143e3, 5.0064e3, 173.8885, 50.6538]])
                if opt_c_ba_x_2.shape[1] != dim_comp:
                    opt_c_ba_x_2 = np.tile(opt_c_ba_x_2.ravel()[:dim_comp], (dim_rows, 1))
                opt_c_b_x_2 = np.array([[787.8988, 201.7349, 236.1948, 143.1756]])
                if opt_c_b_x_2.shape[1] != dim_comp:
                    opt_c_b_x_2 = np.tile(opt_c_b_x_2.ravel()[:dim_comp], (dim_rows, 1))

        # ── Optimal Q(x) ──
        opt_cx1 = np.zeros_like(cx1)
        opt_cx2 = np.zeros_like(cx2)

        if dim_type in (0, 2):
            for alpha in range(dim_comp):
                for k in range(dim_rows):
                    opt_cx2[k, :] += (c_ba_x_2[k, alpha]
                                      * exp_lambda[k, :, alpha])
        elif dim_type in (1, 4):
            for alpha in range(dim_comp):
                for k in range(dim_rows):
                    opt_cx1[k, :] -= (c_ba_x_2[k, alpha]
                                      * exp_lambda[k, :, alpha])

        # ── KL divergence ──
        # D_x entry for this class
        dim_sum_before = int(np.sum(dimensions[:c, 0])) if c > 0 else 0
        dim_sum_after = int(np.sum(dimensions[:c + 1, 0]))

        kl_Hx = np.sum(cHx, axis=1)

        kl_b = np.sum(
            gammaln(ensemble['b_x']) - ensemble['b_x'] * np.log(ensemble['a_x'])
            - gammaln(c_b_x_2) + c_b_x_2 * np.log(c_b_x_2 / c_ba_x_2)
            + (c_b_x_2 - opt_c_b_x_2) * (np.log(c_ba_x_2) - 0.5 / c_b_x_2)
            + (opt_c_b_x_2 / opt_c_ba_x_2 - c_b_x_2 / c_ba_x_2) * c_ba_x_2,
            axis=1
        )

        kl_pi = np.sum(
            gammaln(ensemble['pi_x']) - gammaln(c_pi_x_2)
            + (c_pi_x_2 - opt_c_pi_x_2) * (np.log(c_pi_x_2) - 0.5 / c_pi_x_2),
            axis=1
        )

        kl_pi_sum = (
            -gammaln(dim_comp * ensemble['pi_x'])
            + gammaln(np.sum(c_pi_x_2, axis=1))
            + np.sum(c_pi_x_2 - opt_c_pi_x_2, axis=1)
            * (-np.log(np.sum(c_pi_x_2, axis=1))
               + 0.5 / np.sum(c_pi_x_2, axis=1))
        )

        kl_lambda = np.sum(np.sum(c_log_lambda_x * exp_lambda, axis=1), axis=1)

        ensemble['D_x'][dim_sum_before:dim_sum_after] = (
            kl_Hx + kl_b + kl_pi + kl_pi_sum + kl_lambda
        )

        # Store updated parameters
        ensemble['log_lambda_x'] = train_ensemble_put_lambda(
            c, dimensions, ensemble['log_lambda_x'], c_log_lambda_x)
        ensemble['mx'] = train_ensemble_put(c, dimensions, ensemble['mx'], cmx)
        ensemble['mx2'] = train_ensemble_put(c, dimensions, ensemble['mx2'], cmx2)
        ensemble['opt_ba_x_2'][ptr:ptr + dim_rows * dim_comp] = opt_c_ba_x_2.ravel()
        ensemble['opt_b_x_2'][ptr:ptr + dim_rows * dim_comp] = opt_c_b_x_2.ravel()
        ensemble['opt_pi_x_2'][ptr:ptr + dim_rows * dim_comp] = opt_c_pi_x_2.ravel()

        grad['x1'] = train_ensemble_put(c, dimensions, grad['x1'], opt_cx1)
        grad['x2'] = train_ensemble_put(c, dimensions, grad['x2'], opt_cx2)

        ptr += dim_rows * dim_comp

    # ── Only set gradient for priors if state >= 2 ──
    if state >= 2:
        grad['pi_x_2'] = ensemble['opt_pi_x_2'].copy()
        grad['b_x_2'] = ensemble['opt_b_x_2'].copy()
        grad['ba_x_2'] = ensemble['opt_ba_x_2'].copy()
    else:
        grad['pi_x_2'] = ensemble['pi_x_2'].copy()
        grad['b_x_2'] = ensemble['b_x_2'].copy()
        grad['ba_x_2'] = ensemble['ba_x_2'].copy()

    # ── Q(gamma) — reconstruction error via FFT ──
    dx1, dx2, rerror, data_points = train_blind_deconv(
        dimensions, ensemble, D, Dp, I, J, K, L, M, N)

    ensemble['b_sigma_2'] = ensemble['b_sigma'] + data_points / 2.0
    ensemble['opt_ba_sigma_2'] = ensemble['b_sigma_2'] / (
        ensemble['a_sigma'] + rerror / 2.0)

    if state == 3:
        ensemble['ba_sigma_2'] = ensemble['opt_ba_sigma_2']

    # ── Q(x) update ──
    grad['x1'] = grad['x1'] + ensemble['ba_sigma_2'] * dx1
    grad['x2'] = grad['x2'] + ensemble['ba_sigma_2'] * dx2

    # ── KL divergence for noise ──
    total_dims = int(np.sum(dimensions[:, 0]))
    ensemble['D_x'][total_dims] = (
        gammaln(ensemble['b_sigma']) - gammaln(ensemble['b_sigma_2'])
        - ensemble['b_sigma'] * np.log(ensemble['a_sigma'])
        + ensemble['b_sigma_2'] * np.log(ensemble['b_sigma_2'] / ensemble['ba_sigma_2'])
        + (ensemble['b_sigma_2'] / ensemble['opt_ba_sigma_2']
           - ensemble['b_sigma_2'] / ensemble['ba_sigma_2']) * ensemble['ba_sigma_2']
        + data_points * np.log(2.0 * np.pi) / 2.0
    )

    # Normalise from log_e to bits per data point
    ensemble['D_x'] = ensemble['D_x'] / data_points * np.log2(np.e)
    if np.isnan(ensemble['D_val']):
        ensemble['D_val'] = np.inf

    ensemble['D_val'] = float(np.sum(ensemble['D_x']))

    # ── Gradient = optimal - current ──
    grad['x1'] = grad['x1'] - ensemble['x1']
    grad['x2'] = grad['x2'] - ensemble['x2']
    grad['b_x_2'] = grad['b_x_2'] - ensemble['b_x_2']
    grad['ba_x_2'] = grad['ba_x_2'] - ensemble['ba_x_2']
    grad['pi_x_2'] = grad['pi_x_2'] - ensemble['pi_x_2']

    # ── Clamping: don't train classes if dimensions[:,5] == 0 ──
    ptr2 = 0
    for c in range(num_classes):
        n_elem = int(dimensions[c, 0] * dimensions[c, 1])
        if not int(dimensions[c, 5]):
            grad['x1'][ptr2:ptr2 + n_elem] = 0.0
            grad['x2'][ptr2:ptr2 + n_elem] = 0.0
        ptr2 += n_elem

    return ensemble, grad


# ─────────────────────────────────────────────────────────────────────────────
# Main VB optimization loop
# ─────────────────────────────────────────────────────────────────────────────

def _make_ensemble(dimensions: np.ndarray) -> dict:
    """Create initial ensemble structure (matches MATLAB struct)."""
    total_x = int(dimensions[:, 0].dot(dimensions[:, 1]))
    total_lambda = int(np.sum(np.prod(dimensions[:, 0:3], axis=1)))
    total_prior = int(dimensions[:, 0].dot(dimensions[:, 2]))
    total_D = int(np.sum(dimensions[:, 0])) + 1

    ensemble = {
        'x1': 1e4 * np.random.randn(total_x) * np.ceil(np.random.rand(total_x) * 2),
        'x2': 1e4 * np.ones(total_x),
        'mx': np.zeros(total_x),
        'mx2': np.zeros(total_x),
        'log_lambda_x': np.zeros(total_lambda),
        'pi_x': 1.0,
        'pi_x_2': np.ones(total_prior),
        'opt_pi_x_2': np.ones(total_prior),
        'a_x': 1e-3,
        'ba_x_2': np.ones(total_prior),
        'opt_ba_x_2': np.ones(total_prior),
        'b_x': 1e-3,
        'b_x_2': np.ones(total_prior),
        'opt_b_x_2': np.ones(total_prior),
        'a_sigma': 1e-3,
        'ba_sigma_2': 0.0,
        'opt_ba_sigma_2': 0.0,
        'b_sigma': 1e-3,
        'b_sigma_2': 0.0,
        'D_val': 0.0,
        'D_x': np.zeros(total_D),
    }
    return ensemble


def _make_direction(dimensions: np.ndarray) -> dict:
    """Create zero-initialised direction structure."""
    total_x = int(dimensions[:, 0].dot(dimensions[:, 1]))
    total_prior = int(dimensions[:, 0].dot(dimensions[:, 2]))
    return {
        'x1': np.zeros(total_x),
        'x2': np.zeros(total_x),
        'pi_x_2': np.zeros(total_prior),
        'ba_x_2': np.zeros(total_prior),
        'b_x_2': np.zeros(total_prior),
    }


def _deep_copy_ensemble(e: dict) -> dict:
    """Deep copy of the ensemble dict, copying all numpy arrays."""
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in e.items()}


def train_ensemble_main6(
    dimensions: np.ndarray,
    initial_x1: Optional[np.ndarray],
    initial_x2: Optional[np.ndarray],
    opt_func: str,
    text: str,
    options: list,
    D: np.ndarray,
    Dp: np.ndarray,
    I: int, J: int,
    K: int, L: int,
    M: int, N: int,
    priors: Any,
    fft_mode: int,
    blur_mask: np.ndarray,
    image_mask: np.ndarray,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Main ensemble learning loop.

    Parameters
    ----------
    dimensions : np.ndarray, shape (num_classes, 6)
        [n_rows, n_cols, n_components, prior_type, lock_prior, update_flag]
    initial_x1, initial_x2 : np.ndarray or None
        Initial parameters; if None the default random init is used.
    options : list
        [0] converge_criteria
        [1] plot_step (unused in Python)
        [2] initial noise variance
        [3] restart_priors flag
        [4] restart_switched_off flag
        [5] Niter — max iterations
        [6] (unused)

    Returns
    -------
    ensemble : dict
    D_log : np.ndarray, shape (2, n_iters)
    gamma_log : np.ndarray, shape (1, n_iters)
    """
    Niter = int(options[5])
    D_log = np.full((2, Niter), np.nan)
    gamma_log = np.full((1, Niter), np.nan)

    # Create ensemble
    ensemble = _make_ensemble(dimensions)
    direction = _make_direction(dimensions)

    # Set initial values
    if initial_x1 is not None:
        ensemble['x1'] = np.array(initial_x1, dtype=np.float64).ravel()
    if initial_x2 is not None:
        ensemble['x2'] = np.array(initial_x2, dtype=np.float64).ravel()

    # Avoid zeros
    ensemble['x1'][ensemble['x1'] == 0] = 1e-16
    ensemble['x2'][ensemble['x2'] == 0] = 1e-16

    # State machine
    state = 1
    last_change_iter = 0
    alpha_coeff = 1.0
    beta_coeff = 0.9

    # Options
    converge_criteria = float(options[0])
    ensemble['ba_sigma_2'] = float(options[2]) ** (-2)
    restart_priors = bool(options[3])
    restart_switched_off = bool(options[4])

    oD_val = np.nan
    actual_iters = 0

    for iter_idx in range(Niter):
        actual_iters = iter_idx + 1

        # Re-evaluate after state change
        if iter_idx == last_change_iter:
            if state < 3:
                # Evaluate model before updating priors
                ens_copy = _deep_copy_ensemble(ensemble)
                ens_copy, grad = train_ensemble_evidence6(
                    0.0, dimensions, opt_func, ens_copy, direction, state,
                    D, Dp, I, J, K, L, M, N, priors, fft_mode,
                    blur_mask, image_mask)
                ensemble = ens_copy
                direction = {k: v.copy() if isinstance(v, np.ndarray) else v
                             for k, v in grad.items()}

                # Set priors to optimal
                ensemble['pi_x_2'] = ensemble['opt_pi_x_2'].copy()
                ensemble['b_x_2'] = ensemble['opt_b_x_2'].copy()
                ensemble['ba_x_2'] = ensemble['opt_ba_x_2'].copy()

                # Re-initialise coalesced priors
                ptr = 0
                for c in range(dimensions.shape[0]):
                    dim_rows = int(dimensions[c, 0])
                    dim_cols = int(dimensions[c, 1])
                    dim_comp = int(dimensions[c, 2])
                    dim_type = int(dimensions[c, 3])

                    if dim_comp > 1:
                        c_pi_x_2 = ensemble['pi_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)
                        c_b_x_2 = ensemble['b_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)
                        c_ba_x_2 = ensemble['ba_x_2'][ptr:ptr + dim_rows * dim_comp].reshape(dim_rows, dim_comp)

                        for k in range(dim_rows):
                            if dim_type in (0, 1, 2):
                                sorted_scales = np.sort(c_ba_x_2[k, :])
                                need_restart = (
                                    restart_priors
                                    or np.any(c_pi_x_2[k, :] < ensemble['pi_x'] + 1.0 / dim_comp)
                                    or (dim_comp > 1 and np.any(sorted_scales[1:] < 1.5 * sorted_scales[:-1]))
                                )
                                if need_restart:
                                    mean_scale = (np.sum(c_b_x_2[k, :] / c_ba_x_2[k, :])
                                                  / np.sum(c_b_x_2[k, :]))
                                    c_pi_x_2[k, :] = ensemble['pi_x'] + dim_cols / dim_comp
                                    c_b_x_2[k, :] = ensemble['b_x'] + dim_cols / dim_comp
                                    for a in range(dim_comp):
                                        c_ba_x_2[k, a] = c_b_x_2[k, a] / (
                                            ensemble['a_x']
                                            + 0.5 * (a + 1) * mean_scale
                                            * dim_cols / dim_comp
                                        )

                        ensemble['pi_x_2'][ptr:ptr + dim_rows * dim_comp] = c_pi_x_2.ravel()
                        ensemble['b_x_2'][ptr:ptr + dim_rows * dim_comp] = c_b_x_2.ravel()
                        ensemble['ba_x_2'][ptr:ptr + dim_rows * dim_comp] = c_ba_x_2.ravel()

                    ptr += dim_rows * dim_comp

            # Re-evaluate evidence and search direction
            ens_copy = _deep_copy_ensemble(ensemble)
            ens_copy, grad = train_ensemble_evidence6(
                0.0, dimensions, opt_func, ens_copy, direction, state,
                D, Dp, I, J, K, L, M, N, priors, fft_mode,
                blur_mask, image_mask)
            ensemble = ens_copy

            direction['x1'] = alpha_coeff * grad['x1']
            direction['x2'] = alpha_coeff * grad['x2']
            direction['b_x_2'] = alpha_coeff * grad['b_x_2']
            direction['ba_x_2'] = alpha_coeff * grad['ba_x_2']
            direction['pi_x_2'] = alpha_coeff * grad['pi_x_2']
            step_len = 1.0

        # Take a step
        tensemble = _deep_copy_ensemble(ensemble)
        tensemble, tgrad = train_ensemble_evidence6(
            step_len, dimensions, opt_func, tensemble, direction, state,
            D, Dp, I, J, K, L, M, N, priors, fft_mode,
            blur_mask, image_mask)

        if tensemble['D_val'] > ensemble['D_val']:
            # Reset direction and halve step until improvement
            direction['x1'] = alpha_coeff * grad['x1']
            direction['x2'] = alpha_coeff * grad['x2']
            direction['b_x_2'] = alpha_coeff * grad['b_x_2']
            direction['ba_x_2'] = alpha_coeff * grad['ba_x_2']
            direction['pi_x_2'] = alpha_coeff * grad['pi_x_2']
            step_len = 2.0 * step_len

            while tensemble['D_val'] > ensemble['D_val'] + converge_criteria / 1e4:
                step_len = 0.5 * step_len
                tensemble = _deep_copy_ensemble(ensemble)
                tensemble, tgrad = train_ensemble_evidence6(
                    step_len, dimensions, opt_func, tensemble, direction, state,
                    D, Dp, I, J, K, L, M, N, priors, fft_mode,
                    blur_mask, image_mask)

        # Momentum update
        direction['x1'] = alpha_coeff * tgrad['x1'] + beta_coeff * direction['x1']
        direction['x2'] = alpha_coeff * tgrad['x2'] + beta_coeff * direction['x2']
        direction['b_x_2'] = alpha_coeff * tgrad['b_x_2'] + beta_coeff * direction['b_x_2']
        direction['ba_x_2'] = alpha_coeff * tgrad['ba_x_2'] + beta_coeff * direction['ba_x_2']
        direction['pi_x_2'] = alpha_coeff * tgrad['pi_x_2'] + beta_coeff * direction['pi_x_2']
        step_len = min(1.0, 1.1 * step_len)

        # Accept
        dD_val = tensemble['D_val'] - oD_val
        ensemble = tensemble
        grad = tgrad

        D_log[0, iter_idx] = ensemble['D_val']
        gamma_log[0, iter_idx] = ensemble['ba_sigma_2']
        oD_val = ensemble['D_val']

        print(f"{text}  Iteration {iter_idx + 1:4d}  "
              f"Noise={gamma_log[0, iter_idx] ** (-0.5):11.6e}  "
              f"D_val={ensemble['D_val']:.6e}  state={state}")

        # Check convergence
        converged = False
        if iter_idx > 2:
            last_dD = (D_log[0, iter_idx - 2:iter_idx + 1]
                       - D_log[0, iter_idx - 3:iter_idx])
            if np.all(last_dD < converge_criteria / 1e4) and np.all(last_dD > -converge_criteria):
                converged = True

        if converged:
            if state == 3:
                break
            elif state == 2 and ensemble['opt_ba_sigma_2'] < 1.1 * ensemble['ba_sigma_2']:
                state = 3
            else:
                state = 3 - state

            last_change_iter = iter_idx + 1  # next iteration

            # Handle state==1: update noise, possibly reinitialise
            if state == 1:
                ensemble['ba_sigma_2'] = ensemble['opt_ba_sigma_2']
                for c in range(dimensions.shape[0]):
                    cmx = train_ensemble_get(c, dimensions, ensemble['mx'])
                    cmx2 = train_ensemble_get(c, dimensions, ensemble['mx2'])
                    dim_rows = int(dimensions[c, 0])
                    dim_cols = int(dimensions[c, 1])
                    dim_comp = int(dimensions[c, 2])
                    dim_type = int(dimensions[c, 3])

                    if (restart_switched_off
                            and int(dimensions[c, 4])
                            and dim_type < 3):
                        cx1 = train_ensemble_get(c, dimensions, ensemble['x1'])
                        cx2 = train_ensemble_get(c, dimensions, ensemble['x2'])
                        scales = np.mean(cmx ** 2, axis=1) / np.mean(cmx2, axis=1)

                        for k in range(dim_rows):
                            if scales[k] < 0.7:
                                print(f"  Reinitialising class={c} k={k}")
                                if dim_type == 0:
                                    cx1[k, :] = (1e4 * np.random.randn(dim_cols)
                                                  * np.ceil(np.random.rand(dim_cols) * 2))
                                else:
                                    cx1[k, :] = 1e4 * np.abs(np.random.randn(dim_cols))
                                cx2[k, :] = 1e4

                        ensemble['x1'] = train_ensemble_put(
                            c, dimensions, ensemble['x1'], cx1)
                        ensemble['x2'] = train_ensemble_put(
                            c, dimensions, ensemble['x2'], cx2)

    # Shrink logs
    D_log = D_log[:, :actual_iters]
    gamma_log = gamma_log[:, :actual_iters]

    return ensemble, D_log, gamma_log


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale initialization
# ─────────────────────────────────────────────────────────────────────────────

def initialize_parameters2(
    obs: np.ndarray,
    blur: np.ndarray,
    im: np.ndarray,
    true_blur: Optional[np.ndarray],
    true_im: Optional[np.ndarray],
    pres: float,
    prior_type: int,
    prior_num: int,
    mode_im: str,
    mode_blur: str,
    obs_im: np.ndarray,
    big_blur: np.ndarray,
    spatial_mask: np.ndarray,
    priors: Any,
    fft_mode: int,
    color: bool,
    n_layers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialise x1, x2 parameter vectors for a given pyramid scale.

    Parameters
    ----------
    obs : np.ndarray
        Observed gradient image at this scale.
    blur : np.ndarray
        Blur kernel (from previous scale or initial).
    im : np.ndarray
        Image gradients (from previous scale or initial).
    true_blur, true_im : np.ndarray or None
        Ground truth (synthetic mode); may be None.
    pres : float
        Initial precision.
    mode_im, mode_blur : str
        Initialisation mode strings.
    """
    C = 3 if color else 1
    if spatial_mask is not None:
        spatial_mask_flat = np.asarray(spatial_mask).ravel()
    else:
        spatial_mask_flat = np.zeros(0)

    # ── Blur initialisation ──
    if mode_blur == 'direct':
        me2 = blur.copy()
    elif mode_blur == 'true':
        me2 = true_blur.copy()
    elif mode_blur == 'updown':
        K, L2 = true_blur.shape[:2]
        me2 = utils.imresize(utils.imresize(true_blur, 0.5), (K, L2))
    elif mode_blur == 'delta':
        K = true_blur.shape[0] if true_blur is not None else blur.shape[0]
        me2 = utils.delta_kernel(K)
    elif mode_blur == 'hbar':
        K, L2 = true_blur.shape[:2]
        hK = K // 2
        hL = L2 // 2
        me2 = np.zeros((K, L2))
        me2[hK, hL] = 1
        me2[hK, hL - 1] = 1
        me2[hK, hL + 1] = 1
    elif mode_blur == 'vbar':
        K, L2 = true_blur.shape[:2]
        hK = K // 2
        hL = L2 // 2
        me2 = np.zeros((K, L2))
        me2[hK, hL] = 1
        me2[hK - 1, hL] = 1
        me2[hK + 1, hL] = 1
    elif mode_blur == 'star':
        K, L2 = true_blur.shape[:2]
        hK = K // 2
        hL = L2 // 2
        me2 = np.zeros((K, L2))
        me2[hK - 1, hL + 1] = 1
        me2[hK - 1, hL - 1] = 1
        me2[hK + 1, hL - 1] = 1
        me2[hK + 1, hL + 1] = 1
    elif mode_blur == 'random':
        shape = true_blur.shape if true_blur is not None else blur.shape
        me2 = np.random.rand(*shape)
    elif mode_blur == 'variational':
        me2 = blur.copy()
    else:
        raise ValueError(f"Unknown blur init mode: {mode_blur}")

    # ── Image initialisation ──
    if mode_im == 'direct':
        mx2 = im.copy()
    elif mode_im == 'true':
        mx2 = true_im.copy()
    elif mode_im == 'slight_blur_obs':
        M_s, N_s = obs.shape[:2]
        f = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0
        if obs.ndim == 3:
            mx2 = np.zeros_like(obs)
            for cc in range(obs.shape[2]):
                mx2[:, :, cc] = np.real(ifft2(fft2(f, s=(M_s, N_s)) * fft2(obs[:, :, cc], s=(M_s, N_s))))
        else:
            mx2 = np.real(ifft2(fft2(f, s=(M_s, N_s)) * fft2(obs, s=(M_s, N_s))))
    elif mode_im == 'lucy':
        obs2 = utils.edgetaper(obs_im, big_blur)
        im_tmp = utils.deconvlucy(obs2, big_blur)
        if im_tmp.ndim == 2:
            im_tmp = im_tmp[:, :, np.newaxis]
        im_x_list = []
        im_y_list = []
        for cc in range(C):
            from scipy.signal import convolve2d
            im_x_list.append(convolve2d(im_tmp[:, :, cc], np.array([[1, -1]]), mode='valid'))
            im_y_list.append(convolve2d(im_tmp[:, :, cc], np.array([[1], [-1]]), mode='valid'))

        M_s, N_s = obs.shape[:2]
        mx2 = np.zeros((M_s, N_s, C))
        for cc in range(C):
            im_xs = utils.imresize(im_x_list[cc], (M_s, N_s // 2 + 2))
            im_ys = utils.imresize(im_y_list[cc], (M_s, N_s // 2 + 2))
            mx2[:M_s, :N_s // 2, cc] = im_xs[1:-1, 1:-1]
            mx2[:M_s, N_s // 2:N_s, cc] = im_ys[1:-1, 1:-1]
        if C == 1:
            mx2 = mx2[:, :, 0]

    elif mode_im == 'random':
        SCALE_PARAMETER = [7, 6, 4]
        M_s, N_s = im.shape[:2]
        tmp1 = np.random.rand(M_s, N_s, C)
        tmp2 = np.random.rand(M_s, N_s, C)
        mx2 = SCALE_PARAMETER[2] * (-np.log(tmp1) + np.log(tmp2))
        if C == 1:
            mx2 = mx2[:, :, 0]

    elif mode_im == 'variational':
        MAX_ITERATIONS = 5000
        M_s, N_s = im.shape[:2]
        K_b, L_b = blur.shape[:2]
        norm_blur = me2 / np.sum(me2)

        dim_var = np.array([
            [1, 1, 1, 0, 0, 1],
            [1, K_b * L_b, 4, 1, 0, 0],
            [C, M_s * N_s, 4, 0, 1, 1],
        ], dtype=np.float64)

        if fft_mode:
            I_v = M_s * 2
            J_v = N_s * 2
            Dpf = np.zeros((I_v, J_v, C)) if C > 1 else np.zeros((I_v, J_v))
            if C > 1:
                Dpf[K_b - 1:M_s, L_b - 1:N_s // 2, :] = 1
                Dpf[K_b - 1:M_s, L_b - 1 + N_s // 2:N_s, :] = 1
            else:
                Dpf[K_b - 1:M_s, L_b - 1:N_s // 2] = 1
                Dpf[K_b - 1:M_s, L_b - 1 + N_s // 2:N_s] = 1
            Df = np.pad(obs, ((0, M_s), (0, N_s)), mode='constant')
        else:
            I_v = M_s
            J_v = N_s
            hK_b = K_b // 2
            hL_b = L_b // 2
            Dpf = np.zeros((I_v, J_v, C)) if C > 1 else np.zeros((I_v, J_v))
            if C > 1:
                Dpf[hK_b:M_s - hK_b, hL_b:N_s // 2 - hL_b, :] = 1
                Dpf[hK_b:M_s - hK_b, N_s // 2 + hL_b:N_s - hL_b, :] = 1
            else:
                Dpf[hK_b:M_s - hK_b, hL_b:N_s // 2 - hL_b] = 1
                Dpf[hK_b:M_s - hK_b, N_s // 2 + hL_b:N_s - hL_b] = 1
            Df = obs.copy()

        pres_vec_len = len(norm_blur.ravel()) + len(im.ravel()) + 1
        pres_vector = np.ones(pres_vec_len) * pres
        blur_len = len(norm_blur.ravel())
        q_idx = np.where(spatial_mask_flat != 0)[0]
        for qi in q_idx:
            idx = qi + 1 + blur_len
            if idx < pres_vec_len:
                pres_vector[idx] = spatial_mask_flat[qi]

        dummy_blur_mask = np.zeros((int(dim_var[1, 2]), len(norm_blur.ravel())))

        xx1 = np.concatenate([np.array([0.0]), norm_blur.ravel(), im.ravel()]) * pres_vector
        xx2 = pres_vector.copy()

        ens_var, _, _ = train_ensemble_main6(
            dim_var, xx1, xx2, '', '',
            [1e-4, 0, 1, 0, 0, MAX_ITERATIONS, 0],
            Df, Dpf, I_v, J_v, K_b, L_b, M_s, N_s,
            priors, fft_mode, dummy_blur_mask,
            1.0 - (spatial_mask_flat > 0).astype(np.float64))

        mx2 = train_ensemble_get(2, dim_var, ens_var['mx']).reshape(M_s, N_s)
        if C > 1:
            mx2 = mx2.reshape(M_s, N_s, C)

    elif mode_im == 'greenspan':
        s = utils.create_greenspan_settings()
        s['factor'] = 0
        mx2, _ = utils.greenspan(obs, s)

    else:
        raise ValueError(f"Unknown image init mode: {mode_im}")

    # Normalise blur
    me2_sum = np.sum(me2)
    if me2_sum > 0:
        me2 = me2 / me2_sum

    # Build precision vector with spatial mask
    total_len = len(me2.ravel()) + len(mx2.ravel()) + n_layers
    pres_vector = np.ones(total_len) * pres
    blur_len = len(me2.ravel())
    q_idx = np.where(spatial_mask_flat != 0)[0]
    for qi in q_idx:
        idx = qi + n_layers + blur_len
        if idx < total_len:
            pres_vector[idx] = spatial_mask_flat[qi]

    x1 = np.concatenate([
        np.zeros(n_layers), me2.ravel(), mx2.ravel()
    ]) * pres_vector
    x2 = pres_vector.copy()

    return x1, x2


# ─────────────────────────────────────────────────────────────────────────────
# Inter-scale upsampling
# ─────────────────────────────────────────────────────────────────────────────

def move_level(
    mx: np.ndarray,
    me: np.ndarray,
    K: int, L: int,
    M: int, N: int,
    mode: str,
    resize_step: float,
    center: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Upsample blur kernel and image gradients to next pyramid level.

    Parameters
    ----------
    mx : np.ndarray, shape (rows, cols) or (rows, cols, C)
        Image gradients at current scale.
    me : np.ndarray, shape (k, l)
        Blur kernel at current scale.
    K, L : int
        Target blur kernel size.
    M, N : int
        Target image gradient size.
    mode : str
        Resize mode, e.g. 'matlab_bilinear'.
    resize_step : float
        Scale factor between levels.
    center : bool
        Whether to centre the kernel by its centre of mass.
    """
    from scipy.signal import convolve2d

    if center:
        me_norm = me / np.sum(me) if np.sum(me) != 0 else me
        # Centre of mass
        rows = np.arange(me_norm.shape[0])
        cols = np.arange(me_norm.shape[1])
        mu_y = float(np.sum(rows * np.sum(me_norm, axis=1)))
        mu_x = float(np.sum(cols * np.sum(me_norm, axis=0)))

        offset_x = round(me_norm.shape[1] // 2 - mu_x)
        offset_y = round(me_norm.shape[0] // 2 - mu_y)

        shift_kernel = np.zeros((abs(offset_y) * 2 + 1, abs(offset_x) * 2 + 1))
        shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1

        me = convolve2d(me, shift_kernel, mode='same')
        if mx.ndim == 3:
            for cc in range(mx.shape[2]):
                mx[:, :, cc] = convolve2d(
                    mx[:, :, cc], shift_kernel[::-1, ::-1], mode='same')
        else:
            mx = convolve2d(mx, shift_kernel[::-1, ::-1], mode='same')

    if 'matlab' in mode:
        # Use bilinear/nearest/bicubic upsampling
        if mx.ndim == 3:
            mx_new = np.zeros((M, N, mx.shape[2]))
            for cc in range(mx.shape[2]):
                mx_new[:, :, cc] = utils.imresize(mx[:, :, cc], (M, N))
        else:
            mx_new = utils.imresize(mx, (M, N))
        me_new = utils.imresize(me, (K, L))

    elif mode == 'greenspan':
        s = utils.create_greenspan_settings()
        mx_new, _ = utils.greenspan(mx, s)
        if resize_step != 2:
            if mx_new.ndim == 3:
                new_shape_list = []
                for cc in range(mx_new.shape[2]):
                    new_shape_list.append(
                        utils.imresize(mx_new[:, :, cc], (M, N)))
                mx_new = np.stack(new_shape_list, axis=2)
            else:
                mx_new = utils.imresize(mx_new, (M, N))
        me_new = utils.imresize(me, (K, L))

    elif mode == 'bill_filter':
        if mx.ndim == 3:
            mx_new = np.zeros((M, N, mx.shape[2]))
            for cc in range(mx.shape[2]):
                mx_new[:, :, cc] = utils.imresize(mx[:, :, cc], (M, N))
        else:
            mx_new = utils.imresize(mx, (M, N))
        me_new = utils.imresize(me, (K, L))

        bill_coeffs = np.array([-0.0625, -0.25, 1.625, -0.25, -0.0625])
        bill_filter = bill_coeffs[:, None] * bill_coeffs[None, :]
        if mx_new.ndim == 3:
            for cc in range(mx_new.shape[2]):
                mx_new[:, :, cc] = convolve2d(mx_new[:, :, cc], bill_filter, mode='same')
        else:
            mx_new = convolve2d(mx_new, bill_filter, mode='same')

    else:
        # Default: simple bilinear
        if mx.ndim == 3:
            mx_new = np.zeros((M, N, mx.shape[2]))
            for cc in range(mx.shape[2]):
                mx_new[:, :, cc] = utils.imresize(mx[:, :, cc], (M, N))
        else:
            mx_new = utils.imresize(mx, (M, N))
        me_new = utils.imresize(me, (K, L))

        # Crop to exact size
        if me_new.shape[0] > K or me_new.shape[1] > L:
            me_new = me_new[:K, :L]
        if mx_new.ndim == 2:
            if mx_new.shape[0] > M or mx_new.shape[1] > N:
                mx_new = mx_new[:M, :N]
        else:
            if mx_new.shape[0] > M or mx_new.shape[1] > N:
                mx_new = mx_new[:M, :N, :]

    # Normalise blur kernel
    me_sum = np.sum(me_new)
    if me_sum > 0:
        me_new = me_new / me_sum

    return mx_new, me_new


# ─────────────────────────────────────────────────────────────────────────────
# Post-inference Richardson-Lucy deconvolution
# ─────────────────────────────────────────────────────────────────────────────

def fiddle_lucy3(
    me_est: list,
    obs_im: np.ndarray,
    gamma_correction: float,
    prescale: float,
    lucy_its: int = 10,
    scale_offset: int = 0,
    threshold: float = 7.0,
    edge_crop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Richardson-Lucy on all colour channels after kernel inference.

    Parameters
    ----------
    me_est : list of np.ndarray
        Estimated blur kernels at each scale.
    obs_im : np.ndarray
        Original observed image (uint8 or float [0,255]).
    gamma_correction : float
        Gamma value used during inference.
    prescale : float
        Pre-scaling factor applied to image.
    lucy_its : int
        Number of R-L iterations.
    scale_offset : int
        How many scales to step back from finest.
    threshold : float
        Dynamic threshold on kernel (percentage of max).
    edge_crop : bool
        Whether to crop edges.

    Returns
    -------
    out : np.ndarray
        Deblurred image.
    blur_kernel : np.ndarray
        Thresholded blur kernel.
    """
    idx = len(me_est) - 1 - scale_offset
    blur_kernel = me_est[idx] / np.sum(me_est[idx])

    # Threshold kernel
    thresh_val = np.max(blur_kernel) / threshold
    blur_kernel[blur_kernel < thresh_val] = 0
    blur_kernel = blur_kernel / np.sum(blur_kernel)

    obs = np.array(obs_im, dtype=np.float64)

    # Prescale
    if prescale != 1.0 and prescale != 0.0:
        obs = utils.imresize(obs, prescale)

    # Scale offset
    if scale_offset > 0:
        factor = (1.0 / np.sqrt(2.0)) ** scale_offset
        obs = utils.imresize(obs, factor)

    # Gamma correction
    if gamma_correction != 1.0:
        obs_gam = (obs ** gamma_correction) / (256.0 ** (gamma_correction - 1.0))
    else:
        obs_gam = obs.copy()

    # Edgetaper + RL
    obs_gam = utils.edgetaper(obs_gam, blur_kernel)
    out = utils.deconvlucy(obs_gam, blur_kernel, lucy_its)

    # Undo gamma
    if gamma_correction != 1.0:
        out = out ** (1.0 / gamma_correction)

    out = out.astype(np.float64)

    # Normalise to [0, 1]
    out = out - np.min(out)
    mx = np.max(out)
    if mx > 0:
        out = out / mx

    # Histogram matching
    out = utils.histmatch(out, np.clip(obs, 0, 255).astype(np.uint8))

    # Edge handling: blend deconvolved interior with original at borders
    # (MATLAB crops edges; framework needs same-size output)
    eo = blur_kernel.shape[0] // 2
    if eo > 0:
        obs_ref = np.clip(obs, 0, 255).astype(np.float64)
        out_f = out.astype(np.float64)
        H, W = out_f.shape[:2]

        # Build smooth ramp mask: 0 at border → 1 at eo pixels inside
        ramp_y = np.ones(H, dtype=np.float64)
        ramp_x = np.ones(W, dtype=np.float64)
        for i in range(eo):
            alpha = (i + 1.0) / (eo + 1.0)
            ramp_y[i] = alpha
            ramp_y[H - 1 - i] = alpha
            ramp_x[i] = alpha
            ramp_x[W - 1 - i] = alpha
        mask = ramp_y[:, np.newaxis] * ramp_x[np.newaxis, :]

        if out_f.ndim == 3:
            mask = mask[:, :, np.newaxis]

        out = np.clip(mask * out_f + (1.0 - mask) * obs_ref,
                       0, 255).astype(np.uint8)

    return out, blur_kernel


def fiddle_lucy4(
    me_est: list,
    obs_im: np.ndarray,
    gamma_correction: float,
    prescale: float,
    lucy_its: int = 10,
    scale_offset: int = 0,
    threshold: float = 7.0,
    edge_crop: bool = False,
    brighten: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Richardson-Lucy on the intensity channel only.

    Same as :func:`fiddle_lucy3` but uses ``deconvlucy_intens`` to
    operate only on luminance (for small blurs this gives better results).
    """
    idx = len(me_est) - 1 - scale_offset
    blur_kernel = me_est[idx] / np.sum(me_est[idx])

    # Threshold kernel
    thresh_val = np.max(blur_kernel) / threshold
    blur_kernel[blur_kernel < thresh_val] = 0
    blur_kernel = blur_kernel / np.sum(blur_kernel)

    obs = np.array(obs_im, dtype=np.float64)

    # Prescale
    if prescale != 1.0 and prescale != 0.0:
        obs = utils.imresize(obs, prescale)

    # Scale offset
    if scale_offset > 0:
        factor = (1.0 / np.sqrt(2.0)) ** scale_offset
        obs = utils.imresize(obs, factor)

    # Gamma correction
    if gamma_correction != 1.0:
        obs_gam = (obs ** gamma_correction) / (256.0 ** (gamma_correction - 1.0))
    else:
        obs_gam = obs.copy()

    # Edgetaper + RL on intensity
    obs_gam = utils.edgetaper(obs_gam, blur_kernel)
    out = utils.deconvlucy_intens(obs_gam / 255.0, blur_kernel, lucy_its)

    # Undo gamma
    if gamma_correction != 1.0:
        out = out.astype(np.float64) ** (1.0 / gamma_correction)
    else:
        out = out.astype(np.float64)

    # Normalise to [0, 1]
    out = out - np.min(out)
    mx = np.max(out)
    if mx > 0:
        out = out / mx

    # Scale to uint8
    out = np.clip(255.0 * out * brighten, 0, 255).astype(np.uint8)

    # Edge handling: blend deconvolved interior with original at borders
    eo = blur_kernel.shape[0] // 2
    if eo > 0:
        obs_ref = np.clip(obs, 0, 255).astype(np.float64)
        out_f = out.astype(np.float64)
        H, W = out_f.shape[:2]

        ramp_y = np.ones(H, dtype=np.float64)
        ramp_x = np.ones(W, dtype=np.float64)
        for i in range(eo):
            alpha = (i + 1.0) / (eo + 1.0)
            ramp_y[i] = alpha
            ramp_y[H - 1 - i] = alpha
            ramp_x[i] = alpha
            ramp_x[W - 1 - i] = alpha
        mask = ramp_y[:, np.newaxis] * ramp_x[np.newaxis, :]

        if out_f.ndim == 3:
            mask = mask[:, :, np.newaxis]

        out = np.clip(mask * out_f + (1.0 - mask) * obs_ref,
                       0, 255).astype(np.uint8)

    return out, blur_kernel
