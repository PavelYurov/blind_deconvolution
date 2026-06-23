"""
solvers.py

Core solvers for the ARD Variational-Bayes blind deconvolution pipeline.

Ported from the MATLAB code accompanying:
    J. Kotera, F. Sroubek, V. Smidl,
    "Blind Deconvolution with Model Discrepancies",
    IEEE Transactions on Image Processing, 2017.

Contains
    psf_estim_ard      — main VB blind PSF estimator
                         (PSFestimARDonAll_newer.m).
    mc_restoration     — multiscale orchestrator that wraps
                         ``psf_estim_ard`` and returns the PSF
                         (MCrestoration.m).
    vb_deconv          — VB non-blind image restoration with a known PSF
                         (VBdeconv.m).

MATLAB → Python notes (CRITICAL differences):
    ─────────────────────────────────────────────────────────────────────
    fft2(A, M, N) ≡ np.fft.fft2(A, s=(M, N)).
    ifft2(A) returns COMPLEX even if A is real-conjugate-symmetric.
    edgetaper       — implemented in :mod:`utils` (see edgetaper docstring).
    repmat(A,[1 1 P]) → np.broadcast_to/np.tile along a new last axis.
    A.^2            → A ** 2 (element-wise).
    sum(reshape(X,[],P)) along columns → X.reshape(-1, P).sum(axis=0).
    psi(x) (digamma) → scipy.special.digamma.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.special import digamma
from scipy.signal import convolve2d

from .utils import (
    vec, unvec, mycg, initblur, dsample, edgetaper, u_constr,
    update_g_prior,
    psf2otf, otf2psf, pad_replicate,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_roi(g: np.ndarray, win, shift=(0, 0)) -> np.ndarray:
    """
    Extract the central ROI used for PSF estimation.  For colour input the
    green channel is selected (matches MATLAB ``getROI`` in MCrestoration.m).
    """
    isize = g.shape
    gsize = np.array(isize[:2])
    win = np.array(win)
    if g.ndim > 2 and g.shape[2] > 1:
        cind = 1   # green channel (MATLAB index 2 → Python 1)
    else:
        cind = 0
    if np.any(gsize - win < 0):
        win = gsize.copy()
        shift = (0, 0)
    margin = np.floor((gsize - win) / 2).astype(int) + np.array(shift, dtype=int)
    if g.ndim == 2:
        return g[margin[0]:margin[0] + win[0],
                 margin[1]:margin[1] + win[1]].copy()
    return g[margin[0]:margin[0] + win[0],
             margin[1]:margin[1] + win[1], cind].copy()


def _upsample_param(P: dict, factor: float, target_shape) -> dict:
    """
    Mirrors local ``upsampleParam`` of MCrestoration.m: scales precisions /
    covariances by the ``‖g‖²`` factor of the anti-alias kernel and resamples
    spatial fields to the next pyramid level.
    """
    if not P:
        return P
    k = float(dsample(None, factor)) / 1e2
    out = dict(P)
    s = (target_shape[0], target_shape[1])
    if 'covH' in P:
        out['covH'] = dsample(P['covH'] / k, 1.0 / factor, 'same', s)
    if 'gamma' in P:
        out['gamma'] = P['gamma'] * k
    if 'gamma_vec' in P:
        out['gamma_vec'] = dsample(P['gamma_vec'] * k, 1.0 / factor, 'same', s)
    if 'beta' in P:
        out['beta'] = P['beta'] * k
    if 'beta_vec' in P:
        out['beta_vec'] = dsample(P['beta_vec'] * k, 1.0 / factor, 'same', s)
    if 'uPrior_prec' in P:
        out['uPrior_prec'] = dsample(P['uPrior_prec'] * k, 1.0 / factor, 'same', s)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# psf_estim_ard  (port of PSFestimARDonAll_newer.m)
# ═════════════════════════════════════════════════════════════════════════════

def psf_estim_ard(G: np.ndarray, iH: np.ndarray, PAR: dict,
                  iParam: dict | None = None):
    """
    Variational-Bayes joint estimation of the latent image, the PSF and the
    per-pixel noise / prior precisions.

    Parameters
    ----------
    G      : ndarray ``(H, W)`` or ``(H, W, P)``.  Blurred input ROI.
    iH     : ndarray ``(kh, kw)`` or ``(kh, kw, P)``.  Initial PSF(s).
    PAR    : dict of algorithm parameters (see :mod:`ard.ard` for keys).
    iParam : optional dict carrying state between pyramid levels.

    Returns
    -------
    H        : estimated PSF.
    U        : estimated latent image.
    Report   : diagnostics dict (alpha, gamma, beta, d, covU, covH histories).
    Param    : empty dict by default; mirrors original sparse return.
    gamma_vec: per-pixel noise precision.
    """
    if iParam is None:
        iParam = {}

    # ── basic shapes ────────────────────────────────────────────────────
    if iH.ndim == 2:
        iH = iH[:, :, None]
    hsize = (iH.shape[0], iH.shape[1])
    if G.ndim == 2:
        G = G[:, :, None]
    P = G.shape[2]
    usize = (G.shape[0], G.shape[1])

    is_mask = int(PAR.get('ARDnoise', 0))
    delta_pdf = int(PAR.get('deltaPDF', 0))
    reltol = float(PAR['reltol'])
    ccreltol = float(PAR['ccreltol'])
    maxiter = int(PAR['maxiter'])
    verbose = int(PAR.get('verbose', 0))

    # ── initial scalars ─────────────────────────────────────────────────
    def _init_param(key, default):
        if key in iParam:
            return np.atleast_1d(np.asarray(iParam[key], dtype=np.float64)).copy()
        return np.full(P, float(default), dtype=np.float64)

    alpha = _init_param('alpha', PAR['alpha'])
    d = _init_param('d', PAR['d'])
    gamma = _init_param('gamma', PAR['gamma'])
    if 'beta' in iParam:
        beta = float(iParam['beta'])
    else:
        beta = float(np.atleast_1d(PAR['gamma'])[0])

    # ── per-pixel γ ─────────────────────────────────────────────────────
    mask = np.zeros(usize, dtype=np.float64)
    hh = (hsize[0] // 2, hsize[1] // 2)
    if is_mask:
        mask[hh[0]:usize[0] - hh[0], hh[1]:usize[1] - hh[1]] = 1.0
    else:
        mask[:] = 1.0

    gamma_vec = np.zeros((usize[0], usize[1], P), dtype=np.float64)
    if 'gamma_vec' in iParam:
        for pp in range(P):
            gamma_vec[:, :, pp] = iParam['gamma_vec'][:, :, pp] * mask
    else:
        for pp in range(P):
            gamma_vec[:, :, pp] = gamma[pp] * mask

    if 'beta_vec' in iParam:
        beta_vec = np.array(iParam['beta_vec'], dtype=np.float64, copy=True)
    else:
        beta_vec = np.full_like(gamma_vec, beta)

    # ── FFT precomputations ─────────────────────────────────────────────
    U = np.zeros(usize, dtype=np.float64)
    H = iH.astype(np.float64).copy()
    FH = fft2(H, s=usize, axes=(0, 1))
    FU = fft2(U)

    FDx = fft2(np.array([[1.0, -1.0]]), s=usize)
    FDy = fft2(np.array([[1.0], [-1.0]]), s=usize)
    FD = np.stack([FDx, FDy], axis=2)               # (H, W, 2)

    FA = np.stack([
        np.conj(fft2(np.array([[1.0, 1.0]]), s=usize)),
        np.conj(fft2(np.array([[1.0], [1.0]]), s=usize)),
    ], axis=2)

    FUx = np.zeros(usize, dtype=np.complex128)
    FUy = np.zeros(usize, dtype=np.complex128)

    eG = np.zeros((usize[0], usize[1], P), dtype=np.float64)

    # ── covariances ─────────────────────────────────────────────────────
    if 'covH' in iParam:
        covH = np.array(iParam['covH'], dtype=np.float64, copy=True)
    else:
        covH = np.zeros((usize[0], usize[1], P), dtype=np.float64)
    covU = np.zeros(usize, dtype=np.float64)

    # ── edge-taper ──────────────────────────────────────────────────────
    for p in range(P):
        if is_mask:
            eG[:, :, p] = G[:, :, p]
        else:
            kuni = np.ones(hsize) / np.prod(hsize)
            eG[:, :, p] = edgetaper(G[:, :, p], kuni)

    # ── hyperprior parameters ───────────────────────────────────────────
    alpha_a0, alpha_b0 = PAR.get('alphamodel', (1e-3, 1e-3))
    d_a0, d_b0 = PAR.get('dmodel', (1e-3, 1e-3))
    gamma_a0, gamma_b0 = PAR.get('gammamodel', (1e-3, 1e-3))
    beta_a0, beta_b0 = PAR.get('betamodel', (1e-3, 1e-3))

    Report: dict = {
        'alpha': [alpha.copy()],
        'd': [d.copy()],
        'gamma': [gamma.copy()],
        'beta': [beta],
        'covU': [],
        'covH': [],
    }

    # ── image prior init  (uprior.type only 0=ARD supported) ────────────
    uprior = PAR.get('uprior', {'type': 0})
    uprior_type = int(uprior.get('type', 0))
    image_model = uprior.get('model', (0.0, 1e-4))
    if 'uPrior_prec' in iParam:
        uPrior_prec = np.array(iParam['uPrior_prec'], dtype=np.float64, copy=True)
    else:
        # init with weakly informative prior — same as MATLAB
        init_model = (0.0, 1.0 / (float(np.atleast_1d(PAR['gamma'])[0]) *
                                  float(np.atleast_1d(PAR['alpha'])[0])))
        uPrior_prec = update_g_prior(np.zeros(FD.shape), np.zeros(usize),
                                     init_model)

    # ════════════════════════════════════════════════════════════════════
    # main iteration
    # ════════════════════════════════════════════════════════════════════
    for mI in range(1, maxiter + 1):
        if verbose:
            print(f"Iteration: {mI}")

        # alphagamma_vec = gamma_vec .* repmat(reshape(alpha,1,1,[]), usize)
        alphagamma_vec = gamma_vec * alpha.reshape(1, 1, P)
        # FFT of (eG .* alphagamma_vec)
        FeGu = fft2(eG * alphagamma_vec, axes=(0, 1))

        FUp = FU.copy()

        # ── Ustep ───────────────────────────────────────────────────────
        FU, U, FUx, FUy, covU = _ustep(
            FH, FeGu, alphagamma_vec, covH, uPrior_prec,
            FA, FD, FDx, FDy, usize, P, FU, reltol, delta_pdf,
        )
        Report['covH'].append(covH.reshape(-1, P).sum(axis=0))

        # ── Hstep ───────────────────────────────────────────────────────
        H, FH, covH = _hstep(
            U, FU, eG, alphagamma_vec, covU, beta_vec, FH, H,
            usize, hsize, P, delta_pdf,
        )
        Report['covU'].append(float(covU.sum()))

        # ── update image prior ──────────────────────────────────────────
        if uprior_type == 0:
            DU = np.real(ifft2(np.stack([FUx, FUy], axis=2), axes=(0, 1)))
            uPrior_prec = update_g_prior(DU, covU, image_model)
        else:
            raise NotImplementedError("Only ARD image prior (type=0) is ported.")

        # ── AlphaGammastep ──────────────────────────────────────────────
        alpha, gamma, gamma_vec, d = _alpha_gamma_step(
            U, H, FU, FH, eG, covU, covH, gamma_vec, mask, alpha, d,
            is_mask, alpha_a0, alpha_b0, gamma_a0, gamma_b0,
            d_a0, d_b0, usize, P, verbose,
        )
        Report['alpha'].append(alpha.copy())
        Report['d'].append(d.copy())
        Report['gamma'].append(gamma.copy())

        # ── Betastep ────────────────────────────────────────────────────
        beta, beta_vec = _beta_step(FH, covH, beta_a0, beta_b0, hsize, P)
        Report['beta'].append(beta)

        # ── convergence ────────────────────────────────────────────────
        denom = np.sqrt(np.sum(np.abs(FU) ** 2))
        if denom == 0:
            relcon = 0.0
        else:
            relcon = float(np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom)
        if relcon < ccreltol:
            break

    # extract PSF support
    H_out = H[:hsize[0], :hsize[1], :].copy()
    if H_out.shape[2] == 1:
        H_out = H_out[:, :, 0]
    Param: dict = {}
    return H_out, np.real(ifft2(FU)), Report, Param, gamma_vec


# ═════════════════════════════════════════════════════════════════════════════
# Sub-step functions (closures in MATLAB are ported as helpers)
# ═════════════════════════════════════════════════════════════════════════════

def _ustep(FH, FeGu, alphagamma_vec, covH, uPrior_prec,
           FA, FD, FDx, FDy, usize, P, FU, reltol, delta_pdf):
    """Update q(u): one CG solve in Fourier domain."""
    # cov(H^T diag(γα) H)
    covHTgammaH = np.sum(
        np.real(ifft2(fft2(alphagamma_vec, axes=(0, 1)) *
                      np.conj(fft2(covH, axes=(0, 1))), axes=(0, 1))),
        axis=2,
    )
    # diag approx of H^T diag(γα) H
    H_real_sq_fft = fft2(np.real(ifft2(FH, axes=(0, 1))) ** 2, axes=(0, 1))
    diagHTgammaH = np.sum(
        np.real(ifft2(np.conj(H_real_sq_fft) *
                      fft2(alphagamma_vec, axes=(0, 1)), axes=(0, 1))),
        axis=2,
    )
    # diag approx of D^T diag(λ) D
    appPrior = np.real(ifft2(np.sum(FA * fft2(uPrior_prec, axes=(0, 1)), axis=2)))

    # Right-hand side b = sum_p conj(FH_p) * FeGu_p
    b = np.sum(np.conj(FH) * FeGu, axis=2)

    def gradcalcFU(x_flat):
        X = unvec(x_flat, usize)
        # H^T diag(γα) H · x
        T = ifft2(FH * X[:, :, None], axes=(0, 1)) * alphagamma_vec
        g = np.sum(np.conj(FH) * fft2(T, axes=(0, 1)), axis=2)
        # cov term
        g = g + fft2(covHTgammaH * ifft2(X))
        # D^T diag(λ) D · x
        DX = FD * X[:, :, None]
        g = g + np.sum(np.conj(FD) *
                       fft2(ifft2(DX, axes=(0, 1)) * uPrior_prec, axes=(0, 1)),
                       axis=2)
        return vec(g)

    xmin, _flag, _relres, _it, _r = mycg(
        gradcalcFU, vec(b), reltol, 100, None, vec(FU)
    )
    FU = unvec(xmin, usize)
    FUx = FDx * FU
    FUy = FDy * FU
    U = np.real(ifft2(FU))

    if not delta_pdf:
        covU = 1.0 / (diagHTgammaH + covHTgammaH + appPrior)
    else:
        covU = np.zeros(usize, dtype=np.float64)
    return FU, U, FUx, FUy, covU


def _hstep(U, FU, eG, alphagamma_vec, covU, beta_vec, FH, H,
           usize, hsize, P, delta_pdf):
    """Update q(h): CG solve in Fourier domain."""
    covUTgammaU = np.real(
        ifft2(np.conj(fft2(covU))[:, :, None] *
              fft2(alphagamma_vec, axes=(0, 1)), axes=(0, 1))
    )
    diagUTgammaU = np.real(
        ifft2(np.conj(fft2(U ** 2))[:, :, None] *
              fft2(alphagamma_vec, axes=(0, 1)), axes=(0, 1))
    )

    FeGu = fft2(eG * alphagamma_vec, axes=(0, 1))
    FUD = FeGu * np.conj(FU)[:, :, None]

    b = FUD

    def gradcalcFH(x_flat):
        X = unvec(x_flat, (usize[0], usize[1], P))
        FUS_X = FU[:, :, None] * X
        T = ifft2(FUS_X, axes=(0, 1)) * alphagamma_vec
        g = np.conj(FU)[:, :, None] * fft2(T, axes=(0, 1))
        g = g + fft2((covUTgammaU + beta_vec) *
                     ifft2(X, axes=(0, 1)), axes=(0, 1))
        return vec(g)

    xmin, _flag, _relres, _it, _r = mycg(
        gradcalcFH, vec(b), 1e-6, 1000, None, vec(FH)
    )
    FH = unvec(xmin, (usize[0], usize[1], P))

    hI = np.real(ifft2(FH, axes=(0, 1)))
    H = hI[:hsize[0], :hsize[1], :].copy()
    # Positivity constraint: PSF energy cannot be negative.
    # When a shadow pixel goes negative (CG fluctuation), forcing it to 0
    # makes EH → covH → 0, which drives beta_vec → ∞ for that pixel.
    # The next CG iteration then sees an enormous prior term there and
    # keeps it near zero — a self-reinforcing sparsification loop.
    # (Mirrors MATLAB's commented-out `H = hConstr(H)` in Hstep.)
    H = np.maximum(H, 0.0)
    FH = fft2(H, s=usize, axes=(0, 1))

    if not delta_pdf:
        covH_full = 1.0 / (diagUTgammaU + covUTgammaU + beta_vec)
        # zero outside support of H
        covH_full[hsize[0]:, :, :] = 0.0
        covH_full[:hsize[0], hsize[1]:, :] = 0.0
    else:
        covH_full = np.zeros((usize[0], usize[1], P), dtype=np.float64)

    return H, FH, covH_full


def _alpha_gamma_step(U, H, FU, FH, eG, covU, covH, gamma_vec, mask,
                      alpha, d, is_mask, alpha_a0, alpha_b0,
                      gamma_a0, gamma_b0, d_a0, d_b0, usize, P, verbose):
    """
    Update α, γ_vec and (for ARDnoise=3) the Student-t degrees of freedom d.
    Mirrors AlphaGammastep in PSFestimARDonAll_newer.m.
    """
    E = np.zeros((usize[0], usize[1], 4, P), dtype=np.float64)
    FcovU = fft2(covU)
    FcovH = fft2(covH, axes=(0, 1))

    # || H*u - g ||^2
    E[:, :, 0, :] = np.abs(
        ifft2(FU[:, :, None] * FH - fft2(eG, axes=(0, 1)), axes=(0, 1))
    ) ** 2
    # tr{ U_i^T U_i cov(H) }
    E[:, :, 1, :] = np.real(
        ifft2(fft2(U ** 2)[:, :, None] * FcovH, axes=(0, 1))
    )
    # tr{ H_i^T H_i cov(U) }
    E[:, :, 2, :] = np.real(
        ifft2(fft2(H ** 2, s=usize, axes=(0, 1)) * FcovU[:, :, None], axes=(0, 1))
    )
    # cov(H) ⊙ cov(U)
    E[:, :, 3, :] = np.real(
        ifft2(FcovH * FcovU[:, :, None], axes=(0, 1))
    )

    sumE = E.sum(axis=2)                              # (H, W, P)
    nnz_mask = float(np.count_nonzero(mask))
    # MATLAB: ns = nnz(mask) — only valid (non-border) pixels.
    # This is the count used in BOTH alpha (ARDnoise=3) and d updates.
    ns = int(nnz_mask)

    # ── α update ────────────────────────────────────────────────────────
    # NOTE: MATLAB AlphaGammastep switch has cases {0,1} and 3 only.
    # ARDnoise=2 has NO case → alpha is NOT updated (stays at previous value).
    if is_mask in (0, 1):
        gE = sumE * gamma_vec
        alpha = (alpha_a0 + 0.5 * nnz_mask) / \
                (alpha_b0 + 0.5 * gE.reshape(-1, P).sum(axis=0))
    elif is_mask == 3:
        gE = sumE * gamma_vec
        alpha = (alpha_a0 + 0.5 * ns) / \
                (alpha_b0 + 0.5 * gE.reshape(-1, P).sum(axis=0))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

    # ── γ update ────────────────────────────────────────────────────────
    aE = alpha.reshape(1, 1, P) * sumE
    aE = aE * mask[:, :, None]
    if is_mask == 2:
        gamma_vec = (gamma_a0 + 0.5) / (gamma_b0 + 0.5 * aE)
        gamma_vec = gamma_vec * mask[:, :, None]
    elif is_mask == 3:
        rd = d.reshape(1, 1, P)
        gamma_vec = (rd + 0.5) / (rd + 0.5 * aE)
        gamma_vec = gamma_vec * mask[:, :, None]
        # ── d update ────────────────────────────────────────────────────
        digamma_term = 1.0 + digamma(d + 0.5)
        # MATLAB/Python difference: MATLAB log(-x) returns complex (real part
        # = log|x|, never NaN); Python np.log(-x) returns NaN which then
        # propagates everywhere.  Use real(log(|arg|)) to match MATLAB.
        log_arg = rd + 0.5 * aE
        log_term = np.real(np.log(np.abs(log_arg) + 1e-300))
        d = (d_a0 + 0.5 * ns) / (
            d_b0
            + gamma_vec.reshape(-1, P).sum(axis=0)
            - ns * digamma_term
            + log_term.reshape(-1, P).sum(axis=0)
        )
        # MATLAB resets d when d<0; also reset NaN/Inf which MATLAB never
        # produces (its complex log keeps d finite).
        if np.any(d < 0) or np.any(~np.isfinite(d)):
            d = np.where((d < 0) | (~np.isfinite(d)), 1.0, d)

    gamma = gamma_vec.reshape(-1, P).mean(axis=0)

    if verbose:
        print(f"alpha: {alpha}, d: {d}")

    return alpha, gamma, gamma_vec, d


def _beta_step(FH, covH, beta_a0, beta_b0, hsize, P):
    """Update β and β_vec for the PSF prior."""
    EH = np.abs(ifft2(FH, axes=(0, 1))) ** 2 + covH
    n_h = int(np.prod(hsize))
    beta = float((beta_a0 + P * n_h / 2.0) /
                 (beta_b0 + 0.5 * EH.sum()))
    beta_vec = (beta_a0 + 0.5) / (beta_b0 + 0.5 * EH)
    return beta, beta_vec


# ═════════════════════════════════════════════════════════════════════════════
# mc_restoration  (port of MCrestoration.m)
# ═════════════════════════════════════════════════════════════════════════════

def mc_restoration(g: np.ndarray, hsize, params: dict):
    """
    Multi-scale orchestrator for blind PSF estimation.

    Parameters
    ----------
    g       : ``(H, W)`` or ``(H, W, C)`` blurred input image.
    hsize   : ``(kh, kw)`` upper bound on PSF support.
    params  : full parameter dict (see :func:`ard.ard.default_params`).

    Returns
    -------
    H         : estimated PSF (2-D or ``(kh, kw, P)``).
    gamma_vec : final per-pixel noise precision.
    """
    PAR = params['PAR']

    # ── gamma correction ────────────────────────────────────────────────
    gamma_corr = float(params.get('gamma_corr', 1.0))
    if gamma_corr != 1.0:
        g = g ** gamma_corr

    do_ARDnoise = int(PAR.get('ARDnoise', 0))
    L = max(1, int(PAR.get('MSlevels', 1)))
    factor = float(PAR.get('factor', 1.5))
    sp = factor ** np.arange(L - 1, -1, -1)            # length L

    # ── ROI per level ───────────────────────────────────────────────────
    ROI = [None] * L
    ROI[L - 1] = _get_roi(g, PAR['maxROIsize'])
    for i in range(L - 2, -1, -1):
        ROI[i] = dsample(ROI[L - 1], sp[i], 'valid')

    # ── PSF size schedule ───────────────────────────────────────────────
    hsize_arr = np.array(hsize, dtype=np.float64)
    hsize_list = np.ceil(np.tile(hsize_arr, (L, 1)) /
                         sp.reshape(-1, 1)).astype(int)
    hs = hsize_list[0]
    hs = 2 * (hs // 2) + 1                              # make odd
    h = initblur((hs[0], hs[1]),
                 ((hs[0] + 1) / 2.0, (hs[1] + 1) / 2.0),
                 (1, 1))

    psf_method = params.get('psf_method', 'ard')
    if psf_method != 'ard':
        raise NotImplementedError(f"psf_method={psf_method!r} is not ported.")

    # pyramid_thresh: fraction of the peak used to suppress sub-threshold pixels
    # BEFORE each upsampling step.  The Gaussian anti-alias filter in dsample
    # would otherwise smear small noise/halo pixels across the finer-level
    # support, accumulating halo across all L levels.  Applying the threshold
    # here (element-wise on the small hsize array) adds negligible computation.
    pyramid_thresh = float(params.get('pyramid_thresh', 0.0))

    param: dict = {}
    gamma_vec = None
    for i in range(L):
        # adjust ARDnoise per level
        if do_ARDnoise:
            PAR['ARDnoise'] = do_ARDnoise
            if do_ARDnoise == 1 and i != L - 1:
                PAR['ARDnoise'] = 0

        h, _u, _report, param, gamma_vec = psf_estim_ard(
            ROI[i], h, PAR, param
        )

        if i < L - 1:
            # Suppress halo pixels before upsampling so the Gaussian anti-alias
            # filter in dsample does not smear them into the finer-level kernel.
            if pyramid_thresh > 0.0:
                h_max = h.max()
                if h_max > 0:
                    h = np.where(h >= h_max * pyramid_thresh, h, 0.0)
                    s = h.sum()
                    if s > 1e-12:
                        h = h / s

            target_h = hsize_list[i + 1]
            target_h = (int(target_h[0]), int(target_h[1]))
            h = dsample(h, 1.0 / factor, 'same', target_h)
            param = _upsample_param(param, factor, ROI[i + 1].shape)

        # positivity + sum-to-1 (replicates MATLAB hard-projection)
        h = np.where(h < 0, 0.0, h)
        s = h.sum()
        if s > 0:
            h = h / s
    return h, gamma_vec


# ═════════════════════════════════════════════════════════════════════════════
# vb_deconv  (port of VBdeconv.m, single-image grayscale path)
# ═════════════════════════════════════════════════════════════════════════════

def vb_deconv(G_list, H_list, params: dict):
    """
    VB non-blind restoration with a known PSF.  Single grayscale frame is
    the path exercised by the ARD wrapper, but the function accepts a list
    of frames / PSFs to mirror the MATLAB API.

    Parameters
    ----------
    G_list : list of 2-D arrays (input blurred frames).
    H_list : list of 2-D arrays (PSFs).
    params : full parameter dict.

    Returns
    -------
    U      : restored image (same shape as the first frame).
    Report : diagnostics dict.
    """
    if not isinstance(G_list, (list, tuple)):
        G_list = [G_list]
    if not isinstance(H_list, (list, tuple)):
        H_list = [H_list]
    assert len(G_list) == len(H_list), "G and H must have equal length"

    PAR = params['PAR']
    gamma = float(PAR['gamma_nonblind'])
    reltol = float(PAR['reltol'])
    ccreltol = float(PAR['ccreltol'])
    maxiter_u = int(PAR['maxiter_u'])

    G0 = np.asarray(G_list[0], dtype=np.float64)
    if G0.ndim == 2:
        G0 = G0[:, :, None]
    H, W, C = G0.shape
    P = len(G_list)
    usize = (H, W)

    # PSF size (use first PSF)
    H0 = np.asarray(H_list[0], dtype=np.float64)
    hsize = H0.shape

    # value range per channel
    vrange = np.zeros((C, 2), dtype=np.float64)
    for c in range(C):
        lo = min(np.asarray(g, dtype=np.float64)[..., c].min()
                 if np.asarray(g).ndim == 3 else np.asarray(g, dtype=np.float64).min()
                 for g in G_list)
        hi = max(np.asarray(g, dtype=np.float64)[..., c].max()
                 if np.asarray(g).ndim == 3 else np.asarray(g, dtype=np.float64).max()
                 for g in G_list)
        vrange[c] = [lo, hi]

    # per-pixel gamma with border zeroed out
    mask = np.zeros((H, W), dtype=np.float64)
    hh = (hsize[0] // 2, hsize[1] // 2)
    mask[hh[0]:H - hh[0], hh[1]:W - hh[1]] = 1.0
    gamma_vec = gamma * np.broadcast_to(
        mask[:, :, None, None], (H, W, C, P)).copy()

    # FFT plans
    FDx = np.broadcast_to(fft2(np.array([[1.0, -1.0]]), s=usize)[:, :, None],
                          (H, W, C)).copy()
    FDy = np.broadcast_to(fft2(np.array([[1.0], [-1.0]]), s=usize)[:, :, None],
                          (H, W, C)).copy()
    FD = np.stack([FDx, FDy], axis=3)            # (H, W, C, 2)
    FA = np.stack([
        np.broadcast_to(np.conj(fft2(np.array([[1.0, 1.0]]), s=usize))[:, :, None],
                        (H, W, C)).copy(),
        np.broadcast_to(np.conj(fft2(np.array([[1.0], [1.0]]), s=usize))[:, :, None],
                        (H, W, C)).copy(),
    ], axis=3)

    # FFT-shift PSF centre to origin (matches MATLAB Fspsf usage)
    hssize = hsize
    hshift = np.zeros((hssize[0] // 2 + 1, hssize[1] // 2 + 1))
    hshift[-1, -1] = 1.0
    Fspsf = np.conj(fft2(hshift, s=usize))

    eG = np.zeros((H, W, C, P), dtype=np.float64)
    eH = np.zeros((hssize[0], hssize[1], 1, P), dtype=np.float64)
    for p, (gp, hp) in enumerate(zip(G_list, H_list)):
        gp = np.asarray(gp, dtype=np.float64)
        if gp.ndim == 2:
            gp = gp[:, :, None]
        eG[:, :, :, p] = gp
        eH[:, :, 0, p] = np.asarray(hp, dtype=np.float64)

    # FHS = repmat(Fspsf, [1 1 1 P]) .* fft2(eH, H, W) → tile across C
    FH_pad = fft2(eH, s=usize, axes=(0, 1))               # (H, W, 1, P)
    FHS = np.broadcast_to(
        Fspsf[:, :, None, None] * FH_pad,
        (H, W, C, P)
    ).copy()

    U = np.zeros((H, W, C), dtype=np.float64)
    FU = fft2(U, axes=(0, 1))
    covU = np.zeros((H, W, C), dtype=np.float64)

    # image prior (only ARD type 0 ported)
    image_model = PAR.get('uprior_nonblind', {}).get('model', (0.0, 1e-4))
    if int(PAR.get('uprior_nonblind', {}).get('type', 0)) != 0:
        raise NotImplementedError("Only ARD non-blind prior (type=0) is ported.")
    uPrior_prec = update_g_prior(np.zeros(FD.shape), np.zeros((H, W, C)),
                                 image_model)

    gamma_a0, gamma_b0 = PAR.get('gammamodel_nonblind', (0.0, 1e-10))

    Report = {'gamma': [gamma]}

    for mI in range(1, maxiter_u + 1):
        FeGu = fft2(eG * gamma_vec, axes=(0, 1))
        FUp = FU.copy()

        # ── Ustep ──────────────────────────────────────────────────────
        # diag approx of H^T diag(γ) H
        diagHTgammaH = np.real(
            ifft2(np.conj(fft2(np.real(ifft2(FHS, axes=(0, 1))) ** 2,
                               axes=(0, 1))) *
                  fft2(gamma_vec, axes=(0, 1)), axes=(0, 1))
        )
        appPrior = np.real(
            ifft2(np.sum(FA * fft2(uPrior_prec, axes=(0, 1)), axis=3),
                  axes=(0, 1))
        )

        b = np.sum(np.conj(FHS) * FeGu, axis=3)         # (H, W, C)

        def gradcalcFU_gammavec(x_flat):
            X = unvec(x_flat, (H, W, C))
            T = FHS * X[:, :, :, None]
            T = fft2(gamma_vec * ifft2(T, axes=(0, 1)), axes=(0, 1))
            g = np.sum(np.conj(FHS) * T, axis=3)        # (H, W, C)
            DX = FD * X[:, :, :, None]
            g = g + np.sum(np.conj(FD) *
                           fft2(ifft2(DX, axes=(0, 1)) * uPrior_prec,
                                axes=(0, 1)),
                           axis=3)
            return vec(g)

        xmin, _flag, _relres, _it, _r = mycg(
            gradcalcFU_gammavec, vec(b), reltol, 100, None, vec(FU)
        )
        FU = unvec(xmin, (H, W, C))
        FUx = FDx * FU
        FUy = FDy * FU

        U = np.real(ifft2(FU, axes=(0, 1)))
        covU = 1.0 / (diagHTgammaH.sum(axis=3) + appPrior)

        # ── update image prior ─────────────────────────────────────────
        DU = np.real(ifft2(np.stack([FUx, FUy], axis=3), axes=(0, 1)))
        uPrior_prec = update_g_prior(DU, covU, image_model)

        # ── Gammastep ──────────────────────────────────────────────────
        E = np.zeros((H, W, C, P, 4), dtype=np.float64)
        E[:, :, :, :, 0] = np.abs(
            ifft2(FU[:, :, :, None] * FHS - fft2(eG, axes=(0, 1)), axes=(0, 1))
        ) ** 2
        FcovU = fft2(covU, axes=(0, 1))
        E[:, :, :, :, 2] = np.real(
            ifft2(fft2(np.real(ifft2(FHS, axes=(0, 1))) ** 2, axes=(0, 1)) *
                  FcovU[:, :, :, None], axes=(0, 1))
        )
        # per-pixel γ: mean over channel + 4 components → shape (H, W, P)
        sum_over_C_E = np.sum(np.sum(E, axis=4), axis=2)   # (H, W, P)
        gv = (gamma_a0 + 0.5 * C) / (gamma_b0 + 0.5 * sum_over_C_E)
        gamma_vec = np.broadcast_to(gv[:, :, None, :], (H, W, C, P)).copy()

        E_masked = E * mask[:, :, None, None, None]
        ns = P * float(mask.sum())
        gamma = float((gamma_a0 + 0.5 * ns) / (gamma_b0 + 0.5 * E_masked.sum()))
        Report['gamma'].append(gamma)

        denom = np.sqrt(np.sum(np.abs(FU) ** 2))
        if denom == 0:
            relcon = 0.0
        else:
            relcon = float(np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom)
        if relcon < ccreltol:
            break

    U = u_constr(U, vrange)
    if C == 1:
        U = U[:, :, 0]
    return U, Report


# ═════════════════════════════════════════════════════════════════════════════
# frils_deb_ubc  ─── Fast IRLS non-blind ℓ_p deconvolution
#
# Ported from FBDHSGP/frils_deb_ubc.m
# Reference:
#   X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
#   "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
#   Digital Signal Processing, 2016.
# ═════════════════════════════════════════════════════════════════════════════

def frils_deb_ubc(y: np.ndarray, h: np.ndarray, opt: dict) -> np.ndarray:
    """
    Fast IRLS non-blind image deblurring with undetermined boundary conditions
    and an ℓ_p (Huber) sparsity prior on first- and second-order derivatives.

    Parameters
    ----------
    y   : observed image, shape (M1, M2), float64 in [0, 1].
    h   : estimated PSF (odd-sized), shape (m1, m2).
    opt : dict with keys:
            lambda        — data-fidelity weight
            alpha         — ℓ_p exponent (e.g. 2/3)
            beta_a        — ADMM penalty (set to lambda*alpha*(20/255)^(alpha-2))
            lambda_u      — FOV constraint penalty
            epsilon_min   — Huber ε floor
            epsilon_max   — Huber ε start (continuation)
            out_iter      — outer continuation iterations
            inner_iter    — inner ADMM iterations per β level
            IF            — β multiplicative continuation factor (e.g. √2)

    Returns
    -------
    x_fov : restored image, shape (M1, M2), float64 in [0, 1].
    """
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = pad_replicate(y, hks1, hks2)

    # derivative filters (3×3)
    dxf  = np.array([[0, 0, 0], [0, 1, -1], [0, 0,  0]], dtype=np.float64)
    dyf  = np.array([[0, 0, 0], [0, 1,  0], [0, -1, 0]], dtype=np.float64)
    dyyf = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)
    dxxf = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
    dxyf = np.array([[0, 0, 0], [0,  1, -1], [0, -1, 1]], dtype=np.float64)

    dxfr  = dxf[::-1, ::-1]
    dyfr  = dyf[::-1, ::-1]
    dxxfr = dxxf[::-1, ::-1]
    dyyfr = dyyf[::-1, ::-1]
    dxyfr = dxyf[::-1, ::-1]

    H_otf  = psf2otf(h, (n1, n2))
    Ht     = np.conj(H_otf)
    Hx_otf = psf2otf(dxf,  (n1, n2))
    Hy_otf = psf2otf(dyf,  (n1, n2))
    Hxx    = psf2otf(dxxf, (n1, n2))
    Hyy    = psf2otf(dyyf, (n1, n2))
    Hxy    = psf2otf(dxyf, (n1, n2))

    HH    = (H_otf * Ht).real
    RR    = ((Hx_otf * np.conj(Hx_otf)).real
             + (Hy_otf * np.conj(Hy_otf)).real
             + (Hxx * np.conj(Hxx)).real
             + (Hyy * np.conj(Hyy)).real
             + (Hxy * np.conj(Hxy)).real)

    lambda_  = float(opt["lambda"])
    alpha    = float(opt["alpha"])
    beta_a   = float(opt["beta_a"])
    lambda_u = float(min(opt["lambda_u"], 5000.0 * beta_a))
    w0       = 0.25
    eps_min  = float(opt["epsilon_min"])
    eps_max  = float(opt["epsilon_max"])
    N1       = int(opt["out_iter"])
    N2       = int(opt["inner_iter"])
    IF_      = float(opt["IF"])

    c        = alpha * lambda_
    beta_min = alpha * lambda_ / (eps_max ** (2.0 - alpha))
    beta_max = alpha * lambda_ / (eps_min ** (2.0 - alpha))
    beta     = beta_min

    def _conv_circ(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        """1-pixel circular-pad then 'valid' conv2 — matches MATLAB conv_circ."""
        ap = np.pad(a, ((1, 1), (1, 1)), mode="wrap")
        return convolve2d(ap, k, mode="valid")

    dx  = _conv_circ(x, dxf)
    dy  = _conv_circ(x, dyf)
    dxx = _conv_circ(x, dxxf)
    dyy = _conv_circ(x, dyyf)
    dxy = _conv_circ(x, dxyf)

    adx  = np.abs(dx)
    ady  = np.abs(dy)
    adxx = np.abs(dxx)
    adyy = np.abs(dyy)
    adxy = np.abs(dxy)

    du   = np.zeros((n1, n2), dtype=np.float64)
    dvx  = np.zeros((n1, n2), dtype=np.float64)
    dvy  = np.zeros((n1, n2), dtype=np.float64)
    dvxx = np.zeros((n1, n2), dtype=np.float64)
    dvyy = np.zeros((n1, n2), dtype=np.float64)
    dvxy = np.zeros((n1, n2), dtype=np.float64)

    X    = np.fft.fft2(x)
    Ax   = np.real(np.fft.ifft2(H_otf * X))
    invA = HH + (beta_a / lambda_u) * RR

    for _outer in range(N1):
        with np.errstate(divide="ignore", invalid="ignore"):
            Wx  = np.minimum(beta, c * adx  ** (alpha - 2.0))
            Wy  = np.minimum(beta, c * ady  ** (alpha - 2.0))
            Wxx = np.minimum(beta, c * adxx ** (alpha - 2.0)) * w0
            Wyy = np.minimum(beta, c * adyy ** (alpha - 2.0)) * w0
            Wxy = np.minimum(beta, c * adxy ** (alpha - 2.0)) * w0
        Wx  = np.nan_to_num(Wx,  nan=beta,      posinf=beta)
        Wy  = np.nan_to_num(Wy,  nan=beta,      posinf=beta)
        Wxx = np.nan_to_num(Wxx, nan=beta * w0, posinf=beta * w0)
        Wyy = np.nan_to_num(Wyy, nan=beta * w0, posinf=beta * w0)
        Wxy = np.nan_to_num(Wxy, nan=beta * w0, posinf=beta * w0)

        for _inner in range(N2):
            # u sub-problem (FOV constraint on inner region)
            u = Ax + du
            inner = u[hks1:n1 - hks1, hks2:n2 - hks2]
            inner = (y + lambda_u * inner) / (1.0 + lambda_u)
            u[hks1:n1 - hks1, hks2:n2 - hks2] = inner

            # v sub-problems
            vx  = beta_a * (dx  + dvx)  / (Wx  + beta_a)
            vy  = beta_a * (dy  + dvy)  / (Wy  + beta_a)
            vxx = beta_a * (dxx + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy + dvxy) / (Wxy + beta_a)

            # dual variable updates
            du   = du   - u   + Ax
            dvx  = dvx  - vx  + dx
            dvy  = dvy  - vy  + dy
            dvxx = dvxx - vxx + dxx
            dvyy = dvyy - vyy + dyy
            dvxy = dvxy - vxy + dxy

            # x sub-problem (in Fourier domain)
            Y_fft = np.fft.fft2(u - du) * Ht

            tx  = _conv_circ(vx  - dvx,  dxfr)
            ty  = _conv_circ(vy  - dvy,  dyfr)
            txx = _conv_circ(vxx - dvxx, dxxfr)
            tyy = _conv_circ(vyy - dvyy, dyyfr)
            txy = _conv_circ(vxy - dvxy, dxyfr)

            X  = (Y_fft + (beta_a / lambda_u) *
                  np.fft.fft2(tx + ty + txx + tyy + txy)) / invA
            Ax = np.real(np.fft.ifft2(H_otf * X))
            x  = np.real(np.fft.ifft2(X))

            dx  = _conv_circ(x, dxf)
            dy  = _conv_circ(x, dyf)
            dxx = _conv_circ(x, dxxf)
            dyy = _conv_circ(x, dyyf)
            dxy = _conv_circ(x, dxyf)
            adx  = np.abs(dx)
            ady  = np.abs(dy)
            adxx = np.abs(dxx)
            adyy = np.abs(dyy)
            adxy = np.abs(dxy)

        beta = min(beta * IF_, beta_max)

    x_fov = x[hks1:n1 - hks1, hks2:n2 - hks2]
    return x_fov
