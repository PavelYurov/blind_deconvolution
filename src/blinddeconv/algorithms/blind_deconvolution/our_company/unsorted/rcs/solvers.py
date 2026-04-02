"""
solvers.py

Core solver functions for the Fergus et al. (SIGGRAPH 2006) blind deconvolution.

Ported from MATLAB distribution code v1.2 by Rob Fergus, with contributions
from James Miskin, David MacKay, Yair Weiss, and Bryan Russell.

Reference:
    R. Fergus, B. Singh, A. Hertzmann, S. T. Roweis, W. T. Freeman:
    "Removing Camera Shake from a Single Photograph",
    ACM Trans. Graphics (SIGGRAPH), 2006.

Contains:
    train_blind_deconv       — optimal Q(x), Q(k) via FFT deconvolution
                               (train_blind_deconv.m)
    train_ensemble_evidence6 — evidence evaluation, mixture weights, KL
                               divergence, optimal distributions
                               (train_ensemble_evidence6.m)
    train_ensemble_main6     — main variational inference loop with 3-state
                               machine and conjugate gradient search
                               (train_ensemble_main6.m)
    initialize_parameters2   — scale-level initialisation (variational,
                               direct, delta, hbar, vbar, etc.)
                               (initialize_parameters2.m)
    richardson_lucy          — Richardson-Lucy deconvolution with kernel
                               thresholding, edgetaper, gamma correction,
                               histogram matching (fiddle_lucy3.m)
    richardson_lucy_intens   — intensity-only RL for colour images
                               (fiddle_lucy4.m / deconvlucy_intens.m)

MATLAB -> Python notes:
    - dimensions matrix is 0-indexed in axis 0 but class index c is 1-based
      when passed to train_ensemble_get/put helpers (matching MATLAB).
    - MATLAB struct fields become Python dict keys.
    - All FFT operations use numpy.fft.fft2 / ifft2 with s=(I,J) for zero-pad.
    - gammaln from scipy.special.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.special import gammaln
from scipy.signal import convolve2d, fftconvolve

from .utils import (
    train_ensemble_get,
    train_ensemble_put,
    train_ensemble_get_lambda,
    train_ensemble_put_lambda,
    train_ensemble_rectified5,
    delta_kernel,
    edgetaper,
    histmatch,
    imresize,
    move_level,
    reconsEdge3,
    create_greenspan_settings,
    greenspan,
)


# ═════════════════════════════════════════════════════════════════════════════
# train_blind_deconv  (from train_blind_deconv.m)
# ═════════════════════════════════════════════════════════════════════════════

def train_blind_deconv(dimensions, ensemble, D, Dp, I, J, K, L, M, N):
    """
    Find the optimal Q(x) distributions for the deconvolution problem.

    The data D is formed from a convolution of a known filter me and an
    unknown source mx.  The data is I x J pixels, the source is M x N
    pixels and the filter is K x L pixels.  The data has a binary mask
    Dp which is 1 when the data is observed.

    Parameters
    ----------
    dimensions : (nClasses, 6) int array — class description matrix
    ensemble   : dict — current ensemble parameters
    D          : (I, J) or (I, J, C) observed data (zero-padded)
    Dp         : (I, J) or (I, J, C) observation mask
    I, J       : int — padded observation size
    K, L       : int — kernel size
    M, N       : int — image (gradient) size

    Returns
    -------
    x1, x2    : 1-D arrays — optimal natural parameters
    error      : float — reconstruction error
    data_points: float — number of observed data points
    """
    # ------------------------------------------------------------------
    # Copy the expectations
    # ------------------------------------------------------------------
    mmu = train_ensemble_get(1, dimensions, ensemble['mx'])       # (1, 1)
    me = train_ensemble_get(2, dimensions, ensemble['mx']).reshape(K, L)
    mx = train_ensemble_get(3, dimensions, ensemble['mx']).reshape(M, N)
    mmu2 = train_ensemble_get(1, dimensions, ensemble['mx2'])
    me2 = train_ensemble_get(2, dimensions, ensemble['mx2']).reshape(K, L)
    mx2 = train_ensemble_get(3, dimensions, ensemble['mx2']).reshape(M, N)

    # Scalar values
    mmu_val = float(mmu.ravel()[0])
    mmu2_val = float(mmu2.ravel()[0])

    # ------------------------------------------------------------------
    # Evaluate useful FFTs
    # ------------------------------------------------------------------
    fft_mx = fft2(mx, s=(I, J))
    fft_mx2 = fft2(mx2, s=(I, J))
    fft_mx3 = fft2(mx ** 2, s=(I, J))
    fft_me = fft2(me, s=(I, J))
    fft_me2 = fft2(me2, s=(I, J))
    fft_me3 = fft2(me ** 2, s=(I, J))

    fft_Dp = fft2(Dp, s=(I, J))
    mD = np.real(ifft2(fft_mx * fft_me))
    fft_err = fft2(Dp * (D - mD - mmu_val), s=(I, J))

    # ------------------------------------------------------------------
    # Reconstruction error
    # ------------------------------------------------------------------
    data_points = float(np.sum(Dp))
    error_val = float(
        np.sum(
            Dp * ((D - mD - mmu_val) ** 2
                  + np.real(ifft2(fft_me2 * fft_mx2 - fft_me3 * fft_mx3)))
        )
        + data_points * (mmu2_val - mmu_val ** 2)
    )

    # ------------------------------------------------------------------
    # Optimal distributions for each parameter
    # ------------------------------------------------------------------
    mu1 = float(np.sum(Dp * (D - mD)))
    mu2 = data_points

    e1 = np.real(ifft2(fft_err * np.conj(fft_mx)))
    corr = np.real(ifft2(fft_Dp * np.conj(fft_mx3)))
    e1[:K, :L] = e1[:K, :L] + me * corr[:K, :L]

    e2 = np.real(ifft2(fft_Dp * np.conj(fft_mx2)))

    x1_im = np.real(ifft2(fft_err * np.conj(fft_me)))
    corr2 = np.real(ifft2(fft_Dp * np.conj(fft_me3)))
    x1_im[:M, :N] = x1_im[:M, :N] + mx * corr2[:M, :N]

    x2_im = np.real(ifft2(fft_Dp * np.conj(fft_me2)))

    # ------------------------------------------------------------------
    # Return the optimal distributions as flat vectors
    # ------------------------------------------------------------------
    x1_out = np.concatenate([
        np.array([mu1]),
        e1[:K, :L].ravel(),
        x1_im[:M, :N].ravel(),
    ])
    x2_out = np.concatenate([
        np.array([mu2]),
        e2[:K, :L].ravel(),
        x2_im[:M, :N].ravel(),
    ])
    return x1_out, x2_out, error_val, data_points


# ═════════════════════════════════════════════════════════════════════════════
# train_ensemble_evidence6  (from train_ensemble_evidence6.m)
# ═════════════════════════════════════════════════════════════════════════════

def train_ensemble_evidence6(step_len, dimensions, ensemble, direction,
                             state, D, Dp, I, J, K, L, M, N,
                             priors, fft_mode, blur_mask, image_mask):
    """
    Evaluate evidence for the current ensemble.

    Parameters
    ----------
    step_len   : float — size of step along search direction
    dimensions : (nClasses, 6) int array
    ensemble   : dict — current ensemble (will be modified in-place)
    direction  : dict — search direction
    state      : int — 1: don't update priors/noise;
                        2: don't update noise;
                        3: update all
    D, Dp      : observed data / mask
    I, J       : padded size
    K, L       : kernel size
    M, N       : image size
    priors     : dict with 'pi' and 'gamma' — image prior overrides
    fft_mode   : int — 1 = use FFT convolutions
    blur_mask  : ndarray — spatial weighting on blur elements
    image_mask : ndarray — spatial weighting on image elements

    Returns
    -------
    ensemble   : dict — updated ensemble with D_val, D_x, etc.
    grad       : dict — search gradient / optimal direction
    """
    # Deep-copy ensemble values that will be modified
    import copy
    ens = copy.deepcopy(ensemble)

    # Step to the test point
    _EPS = 1e-300  # prevent exact zeros that cause NaN in rectified5
    ens['x1'] = ens['x1'] + step_len * direction['x1']
    ens['x2'] = np.maximum(np.abs(ens['x2'] + step_len * direction['x2']), _EPS)
    ens['b_x_2'] = np.maximum(np.abs(ens['b_x_2'] + step_len * direction['b_x_2']), _EPS)
    ens['ba_x_2'] = np.maximum(np.abs(ens['ba_x_2'] + step_len * direction['ba_x_2']), _EPS)
    ens['pi_x_2'] = np.maximum(np.abs(ens['pi_x_2'] + step_len * direction['pi_x_2']), _EPS)

    # Make a copy of the direction for gradient output
    grad = copy.deepcopy(direction)

    # Check for valid ensemble
    if (np.any(ens['x2'] < 0) or np.any(ens['b_x_2'] < 0)
            or np.any(ens['ba_x_2'] < 0) or np.any(ens['pi_x_2'] < 0)):
        ens['D_val'] = np.inf
        grad['x1'][:] = np.nan
        grad['x2'][:] = np.nan
        grad['b_x_2'][:] = np.nan
        grad['ba_x_2'][:] = np.nan
        grad['pi_x_2'][:] = np.nan
        return ens, grad

    nClasses = dimensions.shape[0]

    # ------------------------------------------------------------------
    # Find evidence by summing over the components
    # ------------------------------------------------------------------
    ptr = 0
    for c in range(1, nClasses + 1):  # 1-based class index
        c_idx = c - 1

        # Update expectations based on updated Q(x)
        cx1 = train_ensemble_get(c, dimensions, ens['x1'])
        cx2 = train_ensemble_get(c, dimensions, ens['x2'])

        cHx, cmx, cmx2 = train_ensemble_rectified5(cx1, cx2,
                                                     int(dimensions[c_idx, 3]))

        # Reshape mixture parameters
        n_rows = int(dimensions[c_idx, 0])
        n_comp = int(dimensions[c_idx, 2])
        n_cols = int(dimensions[c_idx, 1])

        c_pi_x_2 = ens['pi_x_2'][ptr:ptr + n_rows * n_comp].reshape(
            n_rows, n_comp)
        c_b_x_2 = ens['b_x_2'][ptr:ptr + n_rows * n_comp].reshape(
            n_rows, n_comp)
        c_ba_x_2 = ens['ba_x_2'][ptr:ptr + n_rows * n_comp].reshape(
            n_rows, n_comp)
        c_log_lambda_x = np.zeros((n_rows, n_cols, n_comp))

        prior_type = int(dimensions[c_idx, 3])

        # ------------------------------------------------------------------
        # Compute optimal mixture weights c_log_lambda_x
        # ------------------------------------------------------------------
        if n_comp > 1:
            if prior_type == 0 or prior_type == 2:
                # Gaussian or Rectified Gaussian prior
                for alpha in range(n_comp):
                    for k in range(n_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + 0.5 * np.log(c_ba_x_2[k, alpha])
                            - 0.25 / c_b_x_2[k, alpha]
                            - 0.5 * cmx2[k, :] * c_ba_x_2[k, alpha]
                        )
            elif prior_type == 1:
                # Exponential prior
                for alpha in range(n_comp):
                    for k in range(n_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + np.log(c_ba_x_2[k, alpha])
                            - 0.5 / c_b_x_2[k, alpha]
                            - cmx[k, :] * c_ba_x_2[k, alpha]
                        )
            elif prior_type == 3:
                # Discrete prior — no lambda computation
                pass
            elif prior_type == 4:
                # Exponential-like prior (type 4)
                for alpha in range(n_comp):
                    for k in range(n_rows):
                        c_log_lambda_x[k, :, alpha] = (
                            np.log(c_pi_x_2[k, alpha])
                            - 0.5 / c_pi_x_2[k, alpha]
                            + 0.5 * np.log(c_ba_x_2[k, alpha])
                            - 0.25 / c_b_x_2[k, alpha]
                            - 0.5 * np.abs(cmx[k, :]) * c_ba_x_2[k, alpha]
                        )

            # Normalise c_log_lambda_x (log-sum-exp trick)
            max_c_log = np.max(c_log_lambda_x, axis=2, keepdims=True)
            c_log_lambda_x = c_log_lambda_x - max_c_log
            log_sum = np.log(np.sum(np.exp(c_log_lambda_x), axis=2,
                                    keepdims=True))
            c_log_lambda_x = c_log_lambda_x - log_sum

        # ------------------------------------------------------------------
        # Find optimum parameters for prior parameters
        # ------------------------------------------------------------------
        exp_lambda = np.exp(c_log_lambda_x)  # (n_rows, n_cols, n_comp)
        sum_exp_lambda_2 = np.sum(exp_lambda, axis=1).reshape(
            n_rows, n_comp)  # sum over columns

        if prior_type == 0 or prior_type == 2:
            # (Rectified) Gaussian prior
            opt_c_b_x_2 = ens['b_x'] + sum_exp_lambda_2 / 2.0
            opt_c_pi_x_2 = ens['pi_x'] + sum_exp_lambda_2
            opt_c_ba_x_2 = np.zeros((n_rows, n_comp))
            for alpha in range(n_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ens['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx2, axis=1) / 2.0
                )
        elif prior_type == 1:
            # Exponential prior
            opt_c_b_x_2 = ens['b_x'] + sum_exp_lambda_2
            opt_c_pi_x_2 = ens['pi_x'] + sum_exp_lambda_2
            opt_c_ba_x_2 = np.zeros((n_rows, n_comp))
            for alpha in range(n_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ens['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx, axis=1)
                )
        elif prior_type == 3 or prior_type == 4:
            # Discrete prior (no prior parameter learning)
            opt_c_b_x_2 = ens['b_x'] * np.ones((n_rows, n_comp))
            opt_c_ba_x_2 = (ens['b_x'] / ens['a_x']) * np.ones(
                (n_rows, n_comp))
            opt_c_pi_x_2 = ens['pi_x'] * np.ones((n_rows, n_comp))
        elif prior_type == 5:
            # Laplacian prior
            opt_c_b_x_2 = ens['b_x'] + sum_exp_lambda_2
            opt_c_pi_x_2 = ens['pi_x'] + sum_exp_lambda_2
            opt_c_ba_x_2 = np.zeros((n_rows, n_comp))
            for alpha in range(n_comp):
                opt_c_ba_x_2[:, alpha] = opt_c_b_x_2[:, alpha] / (
                    ens['a_x']
                    + np.sum(exp_lambda[:, :, alpha] * cmx, axis=1)
                )

        # ------------------------------------------------------------------
        # Manual override for priors (dimensions(c, 5) > 0)
        # ------------------------------------------------------------------
        if dimensions[c_idx, 4] > 0:
            if c == 3:
                # Image prior — use P9 (priors dict)
                opt_c_pi_x_2 = (
                    priors['pi'][:n_rows].reshape(n_rows, -1)[:, :n_comp]
                    * 1e3
                )
                opt_c_ba_x_2 = (
                    priors['gamma'][:n_rows].reshape(n_rows, -1)[:, :n_comp]
                )
                opt_c_b_x_2 = np.ones((n_rows, n_comp)) * 1e-3
            elif c == 2:
                # Blur prior — hardcoded values from MATLAB
                opt_c_ba_x_2 = np.array(
                    [5.1143e3, 5.0064e3, 173.8885, 50.6538]
                )[:n_comp].reshape(1, -1).repeat(n_rows, axis=0)
                opt_c_b_x_2 = np.array(
                    [787.8988, 201.7349, 236.1948, 143.1756]
                )[:n_comp].reshape(1, -1).repeat(n_rows, axis=0)

        # ------------------------------------------------------------------
        # Optimum Q(x) gradient contributions from the prior
        # ------------------------------------------------------------------
        opt_cx1 = np.zeros_like(cx1)
        opt_cx2 = np.zeros_like(cx2)

        if prior_type == 0 or prior_type == 2:
            for alpha in range(n_comp):
                for k in range(n_rows):
                    opt_cx2[k, :] += (
                        c_ba_x_2[k, alpha]
                        * exp_lambda[k, :, alpha]
                    )
        elif prior_type == 1 or prior_type == 4:
            for alpha in range(n_comp):
                for k in range(n_rows):
                    opt_cx1[k, :] -= (
                        c_ba_x_2[k, alpha]
                        * exp_lambda[k, :, alpha]
                    )

        # ------------------------------------------------------------------
        # KL divergence for distributions trained in this function
        # ------------------------------------------------------------------
        D_x_start = int(np.sum(dimensions[:c_idx, 0])) if c_idx > 0 else 0
        D_x_end = D_x_start + n_rows

        # Term 1: entropy of Q(x)
        term_Hx = np.sum(cHx, axis=1)  # (n_rows,)

        # Term 2: KL for Gamma distributions on precision
        term_gamma = np.sum(
            gammaln(ens['b_x'])
            - ens['b_x'] * np.log(ens['a_x'])
            - gammaln(c_b_x_2)
            + c_b_x_2 * np.log(c_b_x_2 / c_ba_x_2)
            + (c_b_x_2 - opt_c_b_x_2) * (
                np.log(c_ba_x_2) - 0.5 / c_b_x_2)
            + (opt_c_b_x_2 / opt_c_ba_x_2
               - c_b_x_2 / c_ba_x_2) * c_ba_x_2,
            axis=1
        )

        # Term 3: KL for Dirichlet on mixture weights (individual)
        term_pi_ind = np.sum(
            gammaln(ens['pi_x'])
            - gammaln(c_pi_x_2)
            + (c_pi_x_2 - opt_c_pi_x_2) * (
                np.log(c_pi_x_2) - 0.5 / c_pi_x_2),
            axis=1
        )

        # Term 4: KL for Dirichlet on mixture weights (joint normaliser)
        sum_pi = np.sum(c_pi_x_2, axis=1)
        sum_opt_pi = np.sum(c_pi_x_2 - opt_c_pi_x_2, axis=1)
        term_pi_joint = (
            -gammaln(n_comp * ens['pi_x'])
            + gammaln(sum_pi)
            + sum_opt_pi * (-np.log(sum_pi) + 0.5 / sum_pi)
        )

        # Term 5: entropy of mixture assignments
        term_lambda = np.sum(
            np.sum(c_log_lambda_x * exp_lambda, axis=1),
            axis=1  # sum over components
        )

        ens['D_x'][D_x_start:D_x_end] = (
            term_Hx + term_gamma + term_pi_ind + term_pi_joint + term_lambda
        )

        # ------------------------------------------------------------------
        # Store updated parameters
        # ------------------------------------------------------------------
        ens['log_lambda_x'] = train_ensemble_put_lambda(
            c, dimensions, ens['log_lambda_x'], c_log_lambda_x)
        ens['mx'] = train_ensemble_put(c, dimensions, ens['mx'], cmx)
        ens['mx2'] = train_ensemble_put(c, dimensions, ens['mx2'], cmx2)

        ens['opt_ba_x_2'][ptr:ptr + n_rows * n_comp] = (
            opt_c_ba_x_2.ravel())
        ens['opt_b_x_2'][ptr:ptr + n_rows * n_comp] = (
            opt_c_b_x_2.ravel())
        ens['opt_pi_x_2'][ptr:ptr + n_rows * n_comp] = (
            opt_c_pi_x_2.ravel())

        grad['x1'] = train_ensemble_put(
            c, dimensions, grad['x1'], opt_cx1)
        grad['x2'] = train_ensemble_put(
            c, dimensions, grad['x2'], opt_cx2)

        # Increment pointer
        ptr += n_rows * n_comp

    # ------------------------------------------------------------------
    # Set gradient for priors based on state
    # ------------------------------------------------------------------
    if state >= 2:
        grad['pi_x_2'] = ens['opt_pi_x_2'].copy()
        grad['b_x_2'] = ens['opt_b_x_2'].copy()
        grad['ba_x_2'] = ens['opt_ba_x_2'].copy()
    else:
        grad['pi_x_2'] = ens['pi_x_2'].copy()
        grad['b_x_2'] = ens['b_x_2'].copy()
        grad['ba_x_2'] = ens['ba_x_2'].copy()

    # ------------------------------------------------------------------
    # Q(gamma) — noise model
    # ------------------------------------------------------------------
    if fft_mode:
        dx1, dx2, rerror, data_points = train_blind_deconv(
            dimensions, ens, D, Dp, I, J, K, L, M, N)
    else:
        # Non-FFT mode not implemented (rarely used)
        dx1, dx2, rerror, data_points = train_blind_deconv(
            dimensions, ens, D, Dp, I, J, K, L, M, N)

    ens['b_sigma_2'] = ens['b_sigma'] + data_points / 2.0
    ens['opt_ba_sigma_2'] = ens['b_sigma_2'] / (
        ens['a_sigma'] + rerror / 2.0)

    # Set noise to optimum if in state 3
    if state == 3:
        ens['ba_sigma_2'] = ens['opt_ba_sigma_2']

    # Q(x) update
    grad['x1'] = grad['x1'] + ens['ba_sigma_2'] * dx1
    grad['x2'] = grad['x2'] + ens['ba_sigma_2'] * dx2

    # ------------------------------------------------------------------
    # KL divergence for noise parameter
    # ------------------------------------------------------------------
    D_x_noise_idx = int(np.sum(dimensions[:, 0]))
    ens['D_x'][D_x_noise_idx] = (
        gammaln(ens['b_sigma'])
        - gammaln(ens['b_sigma_2'])
        - ens['b_sigma'] * np.log(ens['a_sigma'])
        + ens['b_sigma_2'] * np.log(
            ens['b_sigma_2'] / ens['ba_sigma_2'])
        + (ens['b_sigma_2'] / ens['opt_ba_sigma_2']
           - ens['b_sigma_2'] / ens['ba_sigma_2'])
        * ens['ba_sigma_2']
        + data_points * np.log(2.0 * np.pi) / 2.0
    )

    # Normalise from log_e to bits per data point
    ens['D_x'] = ens['D_x'] / data_points * np.log2(np.e)

    if np.isnan(ens['D_val']):
        ens['D_val'] = np.inf

    # Total KL divergence
    ens['D_val'] = float(np.sum(ens['D_x']))

    # ------------------------------------------------------------------
    # Find the direction (grad = optimal - current)
    # ------------------------------------------------------------------
    grad['x1'] = grad['x1'] - ens['x1']
    grad['x2'] = grad['x2'] - ens['x2']
    grad['b_x_2'] = grad['b_x_2'] - ens['b_x_2']
    grad['ba_x_2'] = grad['ba_x_2'] - ens['ba_x_2']
    grad['pi_x_2'] = grad['pi_x_2'] - ens['pi_x_2']

    # ------------------------------------------------------------------
    # Don't train classes if options specify clamping (dimensions(c,6)==0)
    # ------------------------------------------------------------------
    ptr = 0
    for c in range(1, nClasses + 1):
        c_idx = c - 1
        n_rows = int(dimensions[c_idx, 0])
        n_cols = int(dimensions[c_idx, 1])
        if not dimensions[c_idx, 5]:
            grad['x1'][ptr:ptr + n_rows * n_cols] = 0.0
            grad['x2'][ptr:ptr + n_rows * n_cols] = 0.0
        ptr += n_rows * n_cols

    # Image masking for saturation (commented out in MATLAB, kept as no-op)
    # ptr = 0
    # for c in range(1, nClasses + 1):
    #     c_idx = c - 1
    #     if c == 3:
    #         ...
    #     ptr += int(dimensions[c_idx, 0]) * int(dimensions[c_idx, 1])

    return ens, grad


# ═════════════════════════════════════════════════════════════════════════════
# train_ensemble_main6  (from train_ensemble_main6.m)
# ═════════════════════════════════════════════════════════════════════════════

def _create_ensemble(dimensions):
    """Create an initial ensemble dict matching MATLAB struct."""
    n_x = int(dimensions[:, 0] @ dimensions[:, 1])  # total x elements
    n_lambda = int(np.sum(np.prod(dimensions[:, :3], axis=1)))
    n_prior = int(dimensions[:, 0] @ dimensions[:, 2])

    rng = np.random.default_rng()

    ens = {
        'x1': 1e4 * rng.standard_normal(n_x) * np.ceil(
            rng.random(n_x) * 2),
        'x2': 1e4 * np.ones(n_x),
        'mx': np.zeros(n_x),
        'mx2': np.zeros(n_x),
        'log_lambda_x': np.zeros(n_lambda),
        'pi_x': 1.0,
        'pi_x_2': np.ones(n_prior),
        'opt_pi_x_2': np.ones(n_prior),
        'a_x': 1e-3,
        'ba_x_2': np.ones(n_prior),
        'opt_ba_x_2': np.ones(n_prior),
        'b_x': 1e-3,
        'b_x_2': np.ones(n_prior),
        'opt_b_x_2': np.ones(n_prior),
        'a_sigma': 1e-3,
        'ba_sigma_2': 0.0,
        'opt_ba_sigma_2': 0.0,
        'b_sigma': 1e-3,
        'b_sigma_2': 0.0,
        'D_val': 0.0,
        'D_x': np.zeros(int(np.sum(dimensions[:, 0])) + 1),
    }
    return ens


def _create_direction(dimensions):
    """Create a zero direction dict matching MATLAB struct."""
    n_x = int(dimensions[:, 0] @ dimensions[:, 1])
    n_prior = int(dimensions[:, 0] @ dimensions[:, 2])
    return {
        'x1': np.zeros(n_x),
        'x2': np.zeros(n_x),
        'pi_x_2': np.zeros(n_prior),
        'ba_x_2': np.zeros(n_prior),
        'b_x_2': np.zeros(n_prior),
    }


def train_ensemble_main6(dimensions, initial_x1, initial_x2,
                          options, D, Dp, I, J, K, L, M, N,
                          priors, fft_mode, blur_mask, image_mask):
    """
    Main variational inference loop.

    Parameters
    ----------
    dimensions  : (nClasses, 6) int array — class description matrix
    initial_x1  : 1-D array or None — initial natural params   (can be None)
    initial_x2  : 1-D array or None — initial natural params   (can be None)
    options     : tuple/list of 7 floats:
                  [convergence_criteria, plot_step, noise_init,
                   restart_priors, restart_switched_off, Niter, unused]
    D, Dp       : observed data / mask
    I, J        : padded size
    K, L        : kernel size
    M, N        : image size
    priors      : dict with 'pi', 'gamma' keys
    fft_mode    : int (1 = FFT mode)
    blur_mask   : ndarray
    image_mask  : ndarray

    Returns
    -------
    ensemble    : dict — final ensemble
    D_log       : (2, n_iters) array — divergence log
    gamma_log   : (1, n_iters) array — noise precision log
    """
    dimensions = np.asarray(dimensions, dtype=np.float64)
    nClasses = dimensions.shape[0]

    Niter = int(options[5])
    converge_criteria = float(options[0])
    plot_step = int(options[1])
    restart_priors = int(options[3])
    restart_switched_off = int(options[4])

    D_log = np.full((2, Niter), np.nan)
    gamma_log = np.full((1, Niter), np.nan)

    # Create ensemble
    ensemble = _create_ensemble(dimensions)

    # Set initial values
    if initial_x1 is not None:
        ensemble['x1'] = np.asarray(initial_x1, dtype=np.float64).copy()
    if initial_x2 is not None:
        ensemble['x2'] = np.asarray(initial_x2, dtype=np.float64).copy()

    # Make sure no parameters are exactly zero
    ensemble['x1'] = ensemble['x1'] + 1e-16 * (ensemble['x1'] == 0)
    ensemble['x2'] = ensemble['x2'] + 1e-16 * (ensemble['x2'] == 0)

    direction = _create_direction(dimensions)

    # State machine
    state = 1
    last_change_iter = 0
    alpha = 1.0
    beta = 0.9

    # Noise variance
    ensemble['ba_sigma_2'] = float(options[2]) ** (-2)

    oD_val = np.nan
    rng = np.random.default_rng()

    final_iter = 0

    for iter_idx in range(Niter):
        final_iter = iter_idx

        # ------------------------------------------------------------------
        # Re-evaluate after a state change
        # ------------------------------------------------------------------
        if iter_idx == last_change_iter:
            if state < 3:
                # Evaluate before updating priors
                step_len = 0.0
                ensemble, grad = train_ensemble_evidence6(
                    step_len, dimensions, ensemble, direction, state,
                    D, Dp, I, J, K, L, M, N,
                    priors, fft_mode, blur_mask, image_mask)
                direction = grad

                # Set priors to optimal
                ensemble['pi_x_2'] = ensemble['opt_pi_x_2'].copy()
                ensemble['b_x_2'] = ensemble['opt_b_x_2'].copy()
                ensemble['ba_x_2'] = ensemble['opt_ba_x_2'].copy()

                # Re-initialise coalesced priors
                ptr = 0
                for c in range(1, nClasses + 1):
                    c_idx = c - 1
                    n_rows = int(dimensions[c_idx, 0])
                    n_comp = int(dimensions[c_idx, 2])
                    n_cols = int(dimensions[c_idx, 1])
                    prior_type = int(dimensions[c_idx, 3])

                    if n_comp > 1:
                        c_pi_x_2 = ensemble['pi_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ].reshape(n_rows, n_comp)
                        c_b_x_2 = ensemble['b_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ].reshape(n_rows, n_comp)
                        c_ba_x_2 = ensemble['ba_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ].reshape(n_rows, n_comp)

                        for k in range(n_rows):
                            if (prior_type == 0 or prior_type == 1
                                    or prior_type == 2):
                                sorted_scales = np.sort(c_ba_x_2[k, :])
                                need_restart = (
                                    restart_priors
                                    or np.any(
                                        c_pi_x_2[k, :]
                                        < ensemble['pi_x']
                                        + 1.0 / n_comp)
                                    or np.any(
                                        sorted_scales[1:]
                                        < 1.5 * sorted_scales[:-1])
                                )
                                if need_restart:
                                    mean_scale = (
                                        np.sum(
                                            c_b_x_2[k, :]
                                            / c_ba_x_2[k, :])
                                        / np.sum(c_b_x_2[k, :])
                                    )
                                    c_pi_x_2[k, :] = (
                                        ensemble['pi_x']
                                        + n_cols / n_comp)
                                    c_b_x_2[k, :] = (
                                        ensemble['b_x']
                                        + n_cols / n_comp)
                                    for a in range(n_comp):
                                        c_ba_x_2[k, a] = (
                                            c_b_x_2[k, a]
                                            / (ensemble['a_x']
                                               + 0.5 * (a + 1)
                                               * mean_scale
                                               * n_cols / n_comp)
                                        )
                            # Discrete (type 3, 4): no prior properties

                        # Store new parameters
                        ensemble['pi_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ] = c_pi_x_2.ravel()
                        ensemble['b_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ] = c_b_x_2.ravel()
                        ensemble['ba_x_2'][
                            ptr:ptr + n_rows * n_comp
                        ] = c_ba_x_2.ravel()

                    ptr += n_rows * n_comp

            # Re-evaluate evidence and search direction
            step_len = 0.0
            ensemble, grad = train_ensemble_evidence6(
                step_len, dimensions, ensemble, direction, state,
                D, Dp, I, J, K, L, M, N,
                priors, fft_mode, blur_mask, image_mask)

            direction['x1'] = alpha * grad['x1']
            direction['x2'] = alpha * grad['x2']
            direction['b_x_2'] = alpha * grad['b_x_2']
            direction['ba_x_2'] = alpha * grad['ba_x_2']
            direction['pi_x_2'] = alpha * grad['pi_x_2']
            step_len = 1.0

        # ------------------------------------------------------------------
        # Take a step and evaluate
        # ------------------------------------------------------------------
        tensemble, tgrad = train_ensemble_evidence6(
            step_len, dimensions, ensemble, direction, state,
            D, Dp, I, J, K, L, M, N,
            priors, fft_mode, blur_mask, image_mask)

        # If step worsened, reset direction and do line search
        if tensemble['D_val'] > ensemble['D_val']:
            direction['x1'] = alpha * grad['x1']
            direction['x2'] = alpha * grad['x2']
            direction['b_x_2'] = alpha * grad['b_x_2']
            direction['ba_x_2'] = alpha * grad['ba_x_2']
            direction['pi_x_2'] = alpha * grad['pi_x_2']
            step_len = 2.0 * step_len

            while (tensemble['D_val']
                   > ensemble['D_val'] + converge_criteria / 1e4):
                step_len = 0.5 * step_len
                tensemble, tgrad = train_ensemble_evidence6(
                    step_len, dimensions, ensemble, direction, state,
                    D, Dp, I, J, K, L, M, N,
                    priors, fft_mode, blur_mask, image_mask)

        # Conjugate gradient update
        direction['x1'] = alpha * tgrad['x1'] + beta * direction['x1']
        direction['x2'] = alpha * tgrad['x2'] + beta * direction['x2']
        direction['b_x_2'] = (
            alpha * tgrad['b_x_2'] + beta * direction['b_x_2'])
        direction['ba_x_2'] = (
            alpha * tgrad['ba_x_2'] + beta * direction['ba_x_2'])
        direction['pi_x_2'] = (
            alpha * tgrad['pi_x_2'] + beta * direction['pi_x_2'])
        step_len = min(1.0, 1.1 * step_len)

        # Accept the point
        dD_val = tensemble['D_val'] - oD_val
        ensemble = tensemble
        grad = tgrad

        # Store logs
        D_log[0, iter_idx] = ensemble['D_val']
        gamma_log[0, iter_idx] = ensemble['ba_sigma_2']

        oD_val = ensemble['D_val']

        noise_std = gamma_log[0, iter_idx] ** (-0.5)
        print(f"  Iteration {iter_idx + 1:4d}  "
              f"Noise={noise_std:11.6e}  "
              f"D_val={ensemble['D_val']:.6e}  state={state}")

        # ------------------------------------------------------------------
        # Check convergence
        # ------------------------------------------------------------------
        converged = False
        if iter_idx > 2:
            last_dD = (D_log[0, iter_idx - 2:iter_idx + 1]
                       - D_log[0, iter_idx - 3:iter_idx])
            if np.all((last_dD < converge_criteria / 1e4)
                      & (last_dD > -converge_criteria)):
                converged = True

        if converged:
            if state == 3:
                # Finished
                print(f"  >>> Converged in state 3, exiting.")
                break
            elif (state == 2
                  and ensemble['opt_ba_sigma_2']
                  < 1.1 * ensemble['ba_sigma_2']):
                # Converged — update noise and continue
                print(f"  >>> State 2→3 (noise close to optimal)")
                state = 3
            else:
                # Swap between 1 and 2
                old_state = state
                state = 3 - state
                print(f"  >>> State {old_state}→{state}")

            last_change_iter = iter_idx + 1  # next iteration triggers reset

            # After converging in state 1, update noise + reinitialise
            if state == 1:
                ensemble['ba_sigma_2'] = ensemble['opt_ba_sigma_2']

                # Randomise switched-off components
                for c in range(1, nClasses + 1):
                    c_idx = c - 1
                    n_rows = int(dimensions[c_idx, 0])
                    n_cols = int(dimensions[c_idx, 1])
                    prior_type = int(dimensions[c_idx, 3])

                    cmx = train_ensemble_get(
                        c, dimensions, ensemble['mx'])
                    cmx2 = train_ensemble_get(
                        c, dimensions, ensemble['mx2'])

                    if (restart_switched_off
                            and dimensions[c_idx, 4]
                            and prior_type < 3):
                        cx1 = train_ensemble_get(
                            c, dimensions, ensemble['x1'])
                        cx2 = train_ensemble_get(
                            c, dimensions, ensemble['x2'])
                        scales = (np.mean(cmx ** 2, axis=1)
                                  / np.mean(cmx2, axis=1))

                        for k in range(n_rows):
                            if scales[k] < 0.7:
                                print(f"  Reinitialising class={c}"
                                      f" k={k}")
                                if prior_type == 0:
                                    cx1[k, :] = (
                                        1e4 * rng.standard_normal(n_cols)
                                        * np.ceil(
                                            rng.random(n_cols) * 2)
                                    )
                                else:
                                    cx1[k, :] = (
                                        1e4
                                        * np.abs(
                                            rng.standard_normal(n_cols))
                                    )
                                cx2[k, :] = 1e4

                        ensemble['x1'] = train_ensemble_put(
                            c, dimensions, ensemble['x1'], cx1)
                        ensemble['x2'] = train_ensemble_put(
                            c, dimensions, ensemble['x2'], cx2)

    # Shrink logs
    n = final_iter + 1
    D_log = D_log[:, :n]
    gamma_log = gamma_log[:, :n]

    return ensemble, D_log, gamma_log


# ═════════════════════════════════════════════════════════════════════════════
# initialize_parameters2  (from initialize_parameters2.m)
# ═════════════════════════════════════════════════════════════════════════════

def initialize_parameters2(obs, blur, im, true_blur, true_im,
                           pres, prior_type, num_components,
                           mode_im, mode_blur, obs_im, big_blur,
                           spatial_mask, priors, fft_mode, color,
                           n_layers):
    """
    Initialise natural parameters (x1, x2) for a scale level.

    Parameters
    ----------
    obs         : (M, N) or (M, N, C) — observed gradient image at this scale
    blur        : (kh, kw) — blur estimate from previous scale (or initial)
    im          : (M, N) or (M, N, C) — image estimate (gradient) from prev scale
    true_blur   : (kh, kw) or None — true blur (synthetic) or placeholder
    true_im     : same as *im* or None
    pres        : float — initial precision
    prior_type  : int — image prior type (0=Gaussian, 1=Exponential, etc.)
    num_components : int — number of image mixture components
    mode_im     : str — image init mode
    mode_blur   : str — blur init mode
    obs_im      : original observed image (for 'lucy' mode)
    big_blur    : full-resolution blur kernel (for 'lucy' mode)
    spatial_mask : (M*N,) or (M, N) mask for saturation
    priors      : dict with 'pi', 'gamma' — MoG priors
    fft_mode    : bool/int
    color       : bool/int
    n_layers    : int — number of offset layers (typically 1)

    Returns
    -------
    x1, x2     : 1-D arrays — initial natural parameters
    """
    # ------------------------------------------------------------------
    # Blur initialisation
    # ------------------------------------------------------------------
    if true_blur is None:
        true_blur_size = blur.shape if blur is not None else (3, 3)
    else:
        true_blur_size = true_blur.shape

    if mode_blur == 'direct':
        me2 = blur.copy()
    elif mode_blur == 'true':
        me2 = true_blur.copy()
    elif mode_blur == 'updown':
        KK, LL = true_blur_size
        small = imresize(true_blur, 0.5, 'bilinear')
        me2 = imresize(small, (KK, LL), 'bilinear')
    elif mode_blur == 'delta':
        KK = true_blur_size[0]
        me2 = delta_kernel(KK)
    elif mode_blur == 'hbar':
        KK, LL = true_blur_size
        hK = KK // 2
        hL = LL // 2
        me2 = np.zeros((KK, LL), dtype=np.float64)
        me2[hK, hL] = 1.0
        me2[hK, hL - 1] = 1.0
        me2[hK, hL + 1] = 1.0
    elif mode_blur == 'vbar':
        KK, LL = true_blur_size
        hK = KK // 2
        hL = LL // 2
        me2 = np.zeros((KK, LL), dtype=np.float64)
        me2[hK, hL] = 1.0
        me2[hK - 1, hL] = 1.0
        me2[hK + 1, hL] = 1.0
    elif mode_blur == 'star':
        KK, LL = true_blur_size
        hK = KK // 2
        hL = LL // 2
        me2 = np.zeros((KK, LL), dtype=np.float64)
        me2[hK - 1, hL + 1] = 1.0
        me2[hK - 1, hL - 1] = 1.0
        me2[hK + 1, hL - 1] = 1.0
        me2[hK + 1, hL + 1] = 1.0
    elif mode_blur == 'random':
        me2 = np.random.rand(*true_blur_size)
    elif mode_blur == 'variational':
        me2 = blur.copy()
    else:
        raise ValueError(f"Unknown blur init mode: {mode_blur}")

    # ------------------------------------------------------------------
    # Image initialisation
    # ------------------------------------------------------------------
    C = 3 if color else 1

    if spatial_mask is not None:
        spatial_mask_flat = np.asarray(spatial_mask).ravel()
    else:
        spatial_mask_flat = np.zeros(np.prod(obs.shape), dtype=np.float64)

    if mode_im == 'direct':
        mx2 = im.copy()
    elif mode_im == 'true':
        mx2 = true_im.copy()
    elif mode_im == 'slight_blur_obs':
        M_s, N_s = obs.shape[:2]
        f = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                     dtype=np.float64) / 16.0
        if obs.ndim == 3:
            mx2 = np.zeros_like(obs)
            for ch in range(C):
                mx2[:, :, ch] = np.real(
                    ifft2(fft2(f, s=(M_s, N_s))
                          * fft2(obs[:, :, ch], s=(M_s, N_s))))
        else:
            mx2 = np.real(ifft2(fft2(f, s=(M_s, N_s))
                                * fft2(obs, s=(M_s, N_s))))
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
        KK, LL = blur.shape if blur is not None else me2.shape

        norm_blur = me2 / np.sum(me2)

        var_dimensions = np.array([
            [1,     1,         1,  0, 0, 1],
            [1,     KK * LL,   4,  1, 0, 0],
            [C,     M_s * N_s, 4,  0, 1, 1],
        ], dtype=np.float64)

        if fft_mode:
            I_v = M_s * 2
            J_v = N_s * 2
            Dpf = np.zeros((I_v, J_v, C) if C > 1 else (I_v, J_v))
            if C > 1:
                Dpf[KK - 1:M_s, LL - 1:N_s // 2, :] = 1
                Dpf[KK - 1:M_s, LL - 1 + N_s // 2:N_s, :] = 1
            else:
                Dpf[KK - 1:M_s, LL - 1:N_s // 2] = 1
                Dpf[KK - 1:M_s, LL - 1 + N_s // 2:N_s] = 1
            Df = np.pad(obs,
                        ((0, M_s), (0, N_s))
                        if obs.ndim == 2
                        else ((0, M_s), (0, N_s), (0, 0)),
                        mode='constant')
        else:
            I_v = M_s
            J_v = N_s
            hK = KK // 2
            hL = LL // 2
            Dpf = np.zeros((I_v, J_v, C) if C > 1 else (I_v, J_v))
            if C > 1:
                Dpf[hK:M_s - hK, hL:N_s // 2 - hL, :] = 1
                Dpf[hK:M_s - hK, N_s // 2 + hL:N_s - hL, :] = 1
            else:
                Dpf[hK:M_s - hK, hL:N_s // 2 - hL] = 1
                Dpf[hK:M_s - hK, N_s // 2 + hL:N_s - hL] = 1

            # Shift convolution (MATLAB: conv2(obs, shift_kernel, 'same'))
            shift_kernel = np.zeros((KK, LL), dtype=np.float64)
            shift_kernel[0, 0] = 1.0
            if obs.ndim == 3:
                Df = np.zeros_like(obs)
                for ch in range(C):
                    Df[:, :, ch] = convolve2d(
                        obs[:, :, ch], shift_kernel,
                        mode='same', boundary='fill')
            else:
                Df = convolve2d(
                    obs, shift_kernel,
                    mode='same', boundary='fill')

        # Flatten for 2D case
        if Dpf.ndim == 3 and C == 1:
            Dpf = Dpf[:, :, 0]
        if Df.ndim == 3 and C == 1:
            Df = Df[:, :, 0]

        pres_vector = np.ones(
            1 + len(norm_blur.ravel()) + len(im.ravel()),
            dtype=np.float64) * pres
        q_idx = np.where(spatial_mask_flat != 0)[0]
        if len(q_idx) > 0:
            pres_vector[q_idx + 1 + len(norm_blur.ravel())] = (
                spatial_mask_flat[q_idx])

        dummy_blur_mask = np.zeros(
            (int(var_dimensions[1, 2]), len(norm_blur.ravel())))

        xx1 = np.concatenate([
            np.array([0.0]),
            norm_blur.ravel(),
            im.ravel()
        ]) * pres_vector
        xx2 = pres_vector.copy()

        var_options = [1e-4, 0, 1, 0, 0, MAX_ITERATIONS, 0]

        sat_mask_var = 1.0 - (spatial_mask_flat > 0).astype(np.float64)

        ensemble_v, _, _ = train_ensemble_main6(
            var_dimensions, xx1, xx2, var_options,
            Df, Dpf, I_v, J_v, KK, LL, M_s, N_s,
            priors, fft_mode, dummy_blur_mask, sat_mask_var)

        mx2_flat = train_ensemble_get(3, var_dimensions, ensemble_v['mx'])
        if C > 1:
            mx2 = mx2_flat.reshape(C, M_s, N_s).transpose(1, 2, 0)
        else:
            mx2 = mx2_flat.reshape(M_s, N_s)

    elif mode_im == 'greenspan':
        s = create_greenspan_settings()
        s['factor'] = 0
        en, _ = greenspan(obs if obs.ndim == 2 else obs[:, :, 0], s)
        mx2 = en
    else:
        raise ValueError(f"Unknown image init mode: {mode_im}")

    # ------------------------------------------------------------------
    # Normalise blur
    # ------------------------------------------------------------------
    me2_sum = np.sum(me2)
    if me2_sum > 0:
        me2 = me2 / me2_sum

    # ------------------------------------------------------------------
    # Spatial masking and build output vectors
    # ------------------------------------------------------------------
    pres_vector = np.ones(
        n_layers + len(me2.ravel()) + len(mx2.ravel()),
        dtype=np.float64) * pres
    q_idx = np.where(spatial_mask_flat != 0)[0]
    if len(q_idx) > 0:
        offset = n_layers + len(me2.ravel())
        valid = q_idx[q_idx + offset < len(pres_vector)]
        if len(valid) > 0:
            pres_vector[valid + offset] = spatial_mask_flat[valid]

    x1 = np.concatenate([
        np.zeros(n_layers),
        me2.ravel(),
        mx2.ravel(),
    ]) * pres_vector
    x2 = pres_vector.copy()

    return x1, x2


# ═════════════════════════════════════════════════════════════════════════════
# Richardson-Lucy deconvolution  (from fiddle_lucy3.m)
# ═════════════════════════════════════════════════════════════════════════════

def _deconvlucy(image, psf, iterations=10):
    """
    Richardson-Lucy deconvolution.

    Parameters
    ----------
    image : (H, W) or (H, W, C) — blurred image (float)
    psf   : (kh, kw) — point spread function
    iterations : int

    Returns
    -------
    result : same shape as image
    """
    psf_mirror = psf[::-1, ::-1]
    eps = 1e-12

    if image.ndim == 2:
        estimate = image.copy()
        for _ in range(iterations):
            reblurred = fftconvolve(estimate, psf, mode='same')
            reblurred = np.maximum(reblurred, eps)
            ratio = image / reblurred
            correction = fftconvolve(ratio, psf_mirror, mode='same')
            estimate = estimate * correction
            estimate = np.maximum(estimate, 0.0)
        return estimate
    else:
        estimate = image.copy()
        for _ in range(iterations):
            reblurred = np.stack([
                fftconvolve(estimate[:, :, c], psf, mode='same')
                for c in range(image.shape[2])
            ], axis=2)
            reblurred = np.maximum(reblurred, eps)
            ratio = image / reblurred
            correction = np.stack([
                fftconvolve(ratio[:, :, c], psf_mirror, mode='same')
                for c in range(image.shape[2])
            ], axis=2)
            estimate = estimate * correction
            estimate = np.maximum(estimate, 0.0)
        return estimate


def richardson_lucy(obs_im, kernel_estimates, gamma_correction=2.2,
                    prescale=1.0, lucy_its=10, threshold=7.0,
                    scale_offset=0, resize_step=np.sqrt(2)):
    """
    Threshold kernel → edgetaper → RL deconvolution → gamma → histmatch.

    Equivalent to MATLAB fiddle_lucy3.m.

    Parameters
    ----------
    obs_im           : (H, W, 3) uint8 or float — original blurred image
    kernel_estimates : list of 2-D arrays — estimated kernels per scale
    gamma_correction : float (default 2.2)
    prescale         : float — image prescale factor
    lucy_its         : int — number of RL iterations
    threshold        : float — kernel noise threshold (% of max)
    scale_offset     : int — offset from finest scale
    resize_step      : float (default sqrt(2))

    Returns
    -------
    out         : (H, W, C) uint8 — deblurred image
    blur_kernel : 2-D array — thresholded kernel
    """
    obs_im = np.asarray(obs_im, dtype=np.float64)

    # Get kernel
    idx = len(kernel_estimates) - 1 - scale_offset
    blur_kernel = kernel_estimates[idx].copy()
    print(f"  RL kernel[{idx}] before norm: min={blur_kernel.min():.6g}, "
          f"max={blur_kernel.max():.6g}, sum={blur_kernel.sum():.6g}")
    blur_kernel = blur_kernel / np.sum(blur_kernel)

    # Threshold kernel
    thresh_val = np.max(blur_kernel) / threshold
    blur_kernel[blur_kernel < thresh_val] = 0.0
    bk_sum = np.sum(blur_kernel)
    print(f"  RL kernel after thresh: nonzero={np.count_nonzero(blur_kernel)}, "
          f"sum={bk_sum:.6g}")
    if bk_sum > 0:
        blur_kernel = blur_kernel / bk_sum

    # Resize image
    if prescale != 1.0:
        obs_im = imresize(obs_im, prescale, 'bilinear')

    # Rescale for offset
    if scale_offset > 0:
        sf = (1.0 / resize_step) ** scale_offset
        obs_im = imresize(obs_im, sf, 'bilinear')

    # Gamma correction
    if gamma_correction != 1.0:
        obs_im_gam = (obs_im ** gamma_correction) / (
            256.0 ** (gamma_correction - 1.0))
    else:
        obs_im_gam = obs_im.copy()

    # Edge taper
    if obs_im_gam.ndim == 3:
        for ch in range(obs_im_gam.shape[2]):
            obs_im_gam[:, :, ch] = edgetaper(
                obs_im_gam[:, :, ch], blur_kernel)
    else:
        obs_im_gam = edgetaper(obs_im_gam, blur_kernel)

    # Richardson-Lucy
    out = _deconvlucy(obs_im_gam, blur_kernel, lucy_its)

    # Undo gamma
    if gamma_correction != 1.0:
        out = np.power(np.maximum(out, 0.0), 1.0 / gamma_correction)
    else:
        out = np.asarray(out, dtype=np.float64)

    # Shift and rescale to [0, 1]
    out = out - np.min(out)
    out_max = np.max(out)
    if out_max > 0:
        out = out / out_max

    # Histogram matching
    out = histmatch(out, np.clip(obs_im, 0, 255).astype(np.uint8))

    return out, blur_kernel


def richardson_lucy_intens(obs_im, kernel_estimates, gamma_correction=2.2,
                           prescale=1.0, lucy_its=10, threshold=7.0,
                           scale_offset=0, resize_step=np.sqrt(2)):
    """
    Intensity-channel-only Richardson-Lucy for colour images.

    Equivalent to MATLAB fiddle_lucy4.m / deconvlucy_intens.m.
    Runs RL only on the Y (luminance) channel of YIQ colour space,
    then converts back to RGB.

    Parameters
    ----------
    (same as richardson_lucy)

    Returns
    -------
    out         : (H, W, C) uint8 — deblurred image
    blur_kernel : 2-D array — thresholded kernel
    """
    obs_im = np.asarray(obs_im, dtype=np.float64)

    # Get kernel
    idx = len(kernel_estimates) - 1 - scale_offset
    blur_kernel = kernel_estimates[idx].copy()
    blur_kernel = blur_kernel / np.sum(blur_kernel)

    # Threshold kernel
    thresh_val = np.max(blur_kernel) / threshold
    blur_kernel[blur_kernel < thresh_val] = 0.0
    bk_sum = np.sum(blur_kernel)
    if bk_sum > 0:
        blur_kernel = blur_kernel / bk_sum

    # Resize image
    if prescale != 1.0:
        obs_im = imresize(obs_im, prescale, 'bilinear')

    # Rescale for offset
    if scale_offset > 0:
        sf = (1.0 / resize_step) ** scale_offset
        obs_im = imresize(obs_im, sf, 'bilinear')

    # Gamma correction
    if gamma_correction != 1.0:
        obs_im_gam = (obs_im ** gamma_correction) / (
            256.0 ** (gamma_correction - 1.0))
    else:
        obs_im_gam = obs_im.copy()

    # Edge taper
    if obs_im_gam.ndim == 3:
        for ch in range(obs_im_gam.shape[2]):
            obs_im_gam[:, :, ch] = edgetaper(
                obs_im_gam[:, :, ch], blur_kernel)
    else:
        obs_im_gam = edgetaper(obs_im_gam, blur_kernel)

    # --- Intensity-only RL ---
    if obs_im_gam.ndim == 3 and obs_im_gam.shape[2] == 3:
        # RGB → YIQ
        rgb_norm = obs_im_gam / 255.0
        yiq = _rgb2yiq(rgb_norm)
        # RL on Y channel only
        y_deconv = _deconvlucy(yiq[:, :, 0:1].squeeze(), blur_kernel,
                               lucy_its)
        yiq_out = yiq.copy()
        yiq_out[:, :, 0] = y_deconv
        out_rgb = _yiq2rgb(yiq_out) * 255.0
        out = np.clip(out_rgb, 0, 255)
    else:
        out = _deconvlucy(obs_im_gam, blur_kernel, lucy_its)

    # Undo gamma
    if gamma_correction != 1.0:
        out = np.power(np.maximum(out, 0.0), 1.0 / gamma_correction)
    else:
        out = np.asarray(out, dtype=np.float64)

    # Shift and rescale to [0, 1]
    out = out - np.min(out)
    out_max = np.max(out)
    if out_max > 0:
        out = out / out_max

    # Convert to uint8
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    return out, blur_kernel


# ─────────────────────────────────────────────────────────────────────────────
# Colour-space helpers (NTSC/YIQ)
# ─────────────────────────────────────────────────────────────────────────────

def _rgb2yiq(rgb):
    """RGB [0,1] → YIQ, matching MATLAB rgb2ntsc."""
    T = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312],
    ])
    shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    yiq_flat = flat @ T.T
    return yiq_flat.reshape(shape)


def _yiq2rgb(yiq):
    """YIQ → RGB [0,1], matching MATLAB ntsc2rgb."""
    T = np.array([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312],
    ])
    T_inv = np.linalg.inv(T)
    shape = yiq.shape
    flat = yiq.reshape(-1, 3)
    rgb_flat = flat @ T_inv.T
    return rgb_flat.reshape(shape)
