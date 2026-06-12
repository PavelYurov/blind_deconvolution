"""
solvers.py

Core solver functions for the HTP (Heavy-Tailed Priors) blind
deconvolution algorithm.

Ported from MATLAB code accompanying:
    J. Kotera, F. Sroubek, P. Milanfar,
    "Blind Deconvolution Using Alternating Maximum a Posteriori
     Estimation with Heavy-tailed Priors", CAIP 2013.

Contains:
    psf_estim_lno_rgrad     — single-scale alternating MAP for (u, h)
                              (PSFestimaLnoRgrad.m)
    fft_cg_sr_al            — non-blind deconvolution via split-Bregman
                              in FFT domain (fftCGSRaL.m)
    mc_restoration          — top-level multiscale pipeline
                              (MCrestoration.m)

MATLAB → Python notes:
    fft2(X, M, N)              → np.fft.fft2(X, s=(M, N))
    real(ifft2(...))           → np.real(np.fft.ifft2(...))
    conj(...)                  → np.conj(...)
    repmat(A,[1 1 D])          → np.broadcast_to(A[..., None], shape)
                                 (or np.repeat for writable copy)
    edgetaper(I, PSF)          → utils.edgetaper_matlab
    imresize(I, scale, m)      → utils.imresize_matlab
    centerPSF(H, t)            → utils.center_psf
    calculateMSE(h, hs)        → utils.calculate_mse
    asetupLnormPrior(q,a,b)    → utils.setup_lp_prior
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, List

import numpy as np

from .utils import (
    simpnormimg,
    denormimg,
    get_roi,
    center_psf,
    calculate_mse,
    fft2_pad,
    setup_lp_prior,
    imresize_matlab,
    edgetaper_matlab,
)


# ═════════════════════════════════════════════════════════════════════════════
# psf_estim_lno_rgrad  (PSFestimaLnoRgrad.m)
# ═════════════════════════════════════════════════════════════════════════════

def psf_estim_lno_rgrad(
    G: np.ndarray,
    iH: np.ndarray,
    PAR: Dict,
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Joint estimation of latent image U and PSF H at a single scale via
    half-quadratic splitting + Bregman iterations, with FFT-closed
    sub-problems.

    Solves (eq. (4) of the paper):
        min_{u,h}  gamma/2 ||h*u - g||^2
                  + alpha_u * sum (|D_x u|^p + |D_y u|^p)
                  + alpha_h * ||h||_1     (h >= 0,  zero outside support)

    The h-step is performed in the **gradient domain** for stability
    (Sec. 3.2 of the paper):
        FUD  = FeGx * conj(FUx) + FeGy * conj(FUy)
        FUTU = |FUx|^2 + |FUy|^2

    Parameters
    ----------
    G    : (H, W) blurred image (single channel, float in [0, 1])
    iH   : (kh, kw) initial PSF (e.g. delta impulse at the coarsest level)
    PAR  : parameters dict (see parameters.py)
    Hstar: optional ground-truth PSF for MSE reporting

    Returns
    -------
    H : (kh, kw) estimated PSF (sums to 1, non-negative, centered)
    U : (H, W) latent image estimate at this scale
    Report : dict with diagnostics (per-iteration MSE if Hstar given)
    """
    Report: Dict = {'hstep': {}}

    gamma = float(PAR['gamma'])
    Lp = float(PAR['Lp'])
    ccreltol = float(PAR['ccreltol'])
    maxiter = int(PAR['maxiter'])
    maxiter_u = int(PAR['maxiter_u'])
    maxiter_h = int(PAR['maxiter_h'])
    alpha_u = float(PAR['alpha_u'])
    beta_u = float(PAR['beta_u'])
    alpha_h = float(PAR['alpha_h'])
    beta_h = float(PAR['beta_h'])
    centering_threshold = float(PAR.get('centering_threshold', 20.0 / 255.0))
    # Iterative kernel hard-threshold (Cho‑Lee 2009 / Pan‑Sun 2014 style).
    # If > 0, after each outer iteration's hstep, set H[H < kernel_thresh*max(H)]=0
    # and renormalise.  This sparsifies the PSF and synergises with the L1
    # prior; default 0 (OFF) so behaviour matches MATLAB on complex kernels
    # (rings, fat support) where hard-thresholding would erase valid mass.
    kernel_thresh = float(PAR.get('kernel_thresh', 0.0))
    # If > 0, recenter PSF (utils.center_psf) after every outer iteration
    # rather than only at the end.  Anchors the PSF at window center
    # throughout iterations, preventing accumulated drift on asymmetric
    # / curved PSFs (b-splines, dendric, hook, comet).
    iterative_recenter = bool(PAR.get('iterative_recenter', True))
    verbose = int(PAR.get('verbose', 0))

    # PSF / image sizes
    iH = np.asarray(iH, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    hsize = iH.shape[:2]
    gsize = G.shape[:2]
    usize = gsize  # latent has same size as g (non-blind problem on ROI)
    M, N = usize

    # MSE tracking
    do_mse = Hstar is not None and np.asarray(Hstar).size > 0
    if do_mse:
        Report['hstep']['mse'] = np.zeros(maxiter + 1, dtype=np.float64)

    U = np.zeros(usize, dtype=np.float64)
    H = iH.copy()

    # FFT of derivative operators (unchanged across iterations)
    FDx = fft2_pad(np.array([[1.0, -1.0]]), M, N)
    FDy = fft2_pad(np.array([[1.0], [-1.0]]), M, N)
    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # Auxiliary and Bregman variables
    Vx = np.zeros(usize, dtype=np.float64)
    Vy = np.zeros(usize, dtype=np.float64)
    # Vh / Bh live on the full FFT lattice (usize), then cropped to hsize.
    Vh = np.zeros(usize, dtype=np.float64)
    Bx = np.zeros(usize, dtype=np.float64)
    By = np.zeros(usize, dtype=np.float64)
    Bh = np.zeros(usize, dtype=np.float64)

    if do_mse:
        Report['hstep']['mse'][0] = calculate_mse(H, np.asarray(Hstar))

    # Edge-tapered g and its gradients (used in both u- and h-steps)
    eG = edgetaper_matlab(G, np.ones(hsize, dtype=np.float64) / np.prod(hsize))
    FeGu = np.fft.fft2(eG)
    FeGx = FDx * FeGu
    FeGy = FDy * FeGu

    # Will be filled in by Ustep, used by Hstep:
    state = {'FU': np.fft.fft2(U), 'FUx': np.zeros(usize, dtype=complex),
             'FUy': np.zeros(usize, dtype=complex)}

    def ustep(gamma_local: float):
        FU = state['FU']
        FHS = fft2_pad(H, M, N)
        FHTH = np.conj(FHS) * FHS
        FGs = np.conj(FHS) * FeGu

        beta = beta_u
        alpha = alpha_u

        nonlocal Vx, Vy, Bx, By
        prior_fh = setup_lp_prior(Lp, alpha, beta)

        for i in range(1, maxiter_u + 1):
            FUp = FU
            b = (FGs
                 + (beta / gamma_local) * (
                     np.conj(FDx) * np.fft.fft2(Vx + Bx)
                     + np.conj(FDy) * np.fft.fft2(Vy + By)
                 ))
            FU = b / (FHTH + (beta / gamma_local) * DTD)

            FUx = FDx * FU
            FUy = FDy * FU
            xD = np.real(np.fft.ifft2(FUx))
            yD = np.real(np.fft.ifft2(FUy))
            xDm = xD - Bx
            yDm = yD - By
            nDm = np.sqrt(xDm * xDm + yDm * yDm)
            Vy = prior_fh(yDm, nDm)
            Vx = prior_fh(xDm, nDm)

            Bx = Bx + Vx - xD
            By = By + Vy - yD

            denom = np.sqrt(np.sum(np.abs(FU) ** 2))
            if denom == 0:
                relcon = 0.0
            else:
                relcon = np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom

            if relcon < ccreltol:
                break

        if verbose:
            print(f'  min_U steps: {i}  relcon: {relcon:.3e}')

        state['FU'] = FU
        state['FUx'] = FDx * FU
        state['FUy'] = FDy * FU

    def hstep(gamma_local: float) -> np.ndarray:
        nonlocal Vh, Bh
        FUx = state['FUx']
        FUy = state['FUy']

        FUD = FeGx * np.conj(FUx) + FeGy * np.conj(FUy)
        FUTU = np.conj(FUx) * FUx + np.conj(FUy) * FUy
        FH = fft2_pad(H, M, N)

        beta = beta_h
        alpha = alpha_h

        prior_fh = setup_lp_prior(1.0, alpha, beta)
        H_local = H

        for i in range(1, maxiter_h + 1):
            FHp = FH
            b = (beta / gamma_local) * np.fft.fft2(Vh + Bh) + FUD
            FH = b / (FUTU + beta / gamma_local)

            denom = np.sqrt(np.sum(np.abs(FH) ** 2))
            if denom == 0:
                relcon = 0.0
            else:
                relcon = np.sqrt(np.sum(np.abs(FHp - FH) ** 2)) / denom

            hI = np.real(np.fft.ifft2(FH))
            hIm = hI - Bh
            nIm = np.abs(hIm)
            Vh = prior_fh(hIm, nIm)
            # Positivity (correct way to enforce h >= 0)
            Vh[Vh < 0] = 0.0
            # Zero outside the PSF support
            Vh[hsize[0]:, :] = 0.0
            Vh[:hsize[0], hsize[1]:] = 0.0
            # Update Bregman variable
            Bh = Bh + Vh - hI

            H_local = hI[:hsize[0], :hsize[1]]

            if relcon < ccreltol:
                break

        if verbose:
            print(f'  min_H step {i}  relcon: {relcon:.3e}')
        return H_local

    # Main alternating loop
    for mI in range(1, maxiter + 1):
        ustep(gamma)
        H = hstep(gamma)

        # ── Optional iterative hard-threshold (Cho-Lee / Pan-Sun) ────────
        # Use SPARINGLY (default OFF) — for fat / ring / b-spline kernels
        # this can erase legitimate mass.  When ON it produces visibly
        # cleaner kernels on simple motion / sparse curve PSFs.
        if kernel_thresh > 0.0:
            H_pos = np.maximum(H, 0.0)
            mx = H_pos.max()
            if mx > 0:
                H = np.where(H_pos < kernel_thresh * mx, 0.0, H_pos)
                s = H.sum()
                if s > 0:
                    H = H / s

        # ── Iterative re-centering anchors PSF at window centre ──────────
        # so subsequent iterations don't drift, and so the final non-blind
        # step receives a PSF whose mass centroid coincides with the
        # `hshift` impulse used by fft_cg_sr_al.  This is the single most
        # effective fix for "image plyvyot" on asymmetric kernels.
        if iterative_recenter and centering_threshold > 0 and mI < maxiter:
            H = center_psf(H, centering_threshold)

        if do_mse:
            Report['hstep']['mse'][mI] = calculate_mse(H, np.asarray(Hstar))

        # gamma continuation — helps escape local minima
        gamma = gamma * 1.5

    # Final PSF centering / cleanup (always at end, MATLAB-faithful)
    if centering_threshold > 0:
        H = center_psf(H, centering_threshold)

    U = np.real(np.fft.ifft2(state['FU']))
    return H, U, Report


# ═════════════════════════════════════════════════════════════════════════════
# fft_cg_sr_al  (fftCGSRaL.m)
# ═════════════════════════════════════════════════════════════════════════════

def fft_cg_sr_al(G: np.ndarray, H: np.ndarray, PAR: Dict) -> np.ndarray:
    """
    Fast non-blind deconvolution using augmented Lagrangian / split-
    Bregman, working entirely in the Fourier domain.

    Problem:
        min_u   gamma/2 || g - H * u ||^2 + alpha * || grad(u) ||_p^p

    Works on the full image (mono- or multi-channel).  For multi-channel
    inputs the gradient norm is aggregated across channels (vectorial
    TV / Lp), matching MATLAB's behaviour.

    Parameters
    ----------
    G  : (H, W) or (H, W, C) blurred image, float in [0, 1]
    H  : (kh, kw) PSF (sums to 1, non-negative)
    PAR: parameters dict; uses gamma_nonblind, beta_u_nonblind,
         Lp_nonblind when present (otherwise falls back to gamma,
         beta_u, Lp).

    Returns
    -------
    U : restored image, same shape as G, clipped to per-channel value
        range of G.
    """
    G = np.asarray(G, dtype=np.float64)
    H_psf = np.asarray(H, dtype=np.float64)

    maxiter = int(PAR['maxiter_u'])
    alpha = float(PAR['alpha_u'])
    ccreltol = float(PAR['ccreltol'])
    gamma = float(PAR.get('gamma_nonblind', PAR['gamma']))
    beta = float(PAR.get('beta_u_nonblind', PAR['beta_u']))
    Lp = float(PAR.get('Lp_nonblind', PAR['Lp']))
    verbose = int(PAR.get('verbose', 0))

    if G.ndim == 2:
        G = G[..., None]
        squeeze_out = True
    else:
        squeeze_out = False
    Hh, Ww, C = G.shape

    # Per-channel intensity range for output clipping
    vrange = np.zeros((C, 2), dtype=np.float64)
    for c in range(C):
        ch = G[..., c]
        vrange[c, 0] = ch.min()
        vrange[c, 1] = ch.max()

    # PSF center shift  (so blur is non-shifting in FFT domain)
    hshift = np.zeros_like(H_psf)
    hshift[H_psf.shape[0] // 2, H_psf.shape[1] // 2] = 1.0

    # FFTs
    FDx_2d = fft2_pad(np.array([[1.0, -1.0]]), Hh, Ww)
    FDy_2d = fft2_pad(np.array([[1.0], [-1.0]]), Hh, Ww)
    FDx = np.repeat(FDx_2d[..., None], C, axis=2)
    FDy = np.repeat(FDy_2d[..., None], C, axis=2)

    FH_2d = (np.conj(fft2_pad(hshift, Hh, Ww))
             * fft2_pad(H_psf, Hh, Ww))
    FH = np.repeat(FH_2d[..., None], C, axis=2)
    FHTH = np.conj(FH) * FH

    eG = edgetaper_matlab(G if not squeeze_out else G[..., 0], H_psf)
    if eG.ndim == 2:
        eG = eG[..., None]
    FGu = np.fft.fft2(eG, axes=(0, 1))
    FGs = np.conj(FH) * FGu

    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # Bregman / auxiliary variables
    Bx = np.zeros((Hh, Ww, C), dtype=np.float64)
    By = np.zeros((Hh, Ww, C), dtype=np.float64)
    Vx = np.zeros((Hh, Ww, C), dtype=np.float64)
    Vy = np.zeros((Hh, Ww, C), dtype=np.float64)

    FU = np.zeros((Hh, Ww, C), dtype=complex)
    prior_fh = setup_lp_prior(Lp, alpha, beta)

    for i in range(1, maxiter + 1):
        if verbose:
            print(f'nonblind deconv step {i}')

        FUp = FU
        b = FGs + (beta / gamma) * (
            np.conj(FDx) * np.fft.fft2(Vx + Bx, axes=(0, 1))
            + np.conj(FDy) * np.fft.fft2(Vy + By, axes=(0, 1))
        )
        FU = b / (FHTH + (beta / gamma) * DTD)

        xD = np.real(np.fft.ifft2(FDx * FU, axes=(0, 1)))
        yD = np.real(np.fft.ifft2(FDy * FU, axes=(0, 1)))
        xDm = xD - Bx
        yDm = yD - By
        # Vectorial Lp: norm aggregates across channels
        nDm_2d = np.sqrt(np.sum(xDm ** 2, axis=2) + np.sum(yDm ** 2, axis=2))
        nDm = np.repeat(nDm_2d[..., None], C, axis=2)

        Vy = prior_fh(yDm, nDm)
        Vx = prior_fh(xDm, nDm)

        Bx = Bx + Vx - xD
        By = By + Vy - yD

        denom = np.sqrt(np.sum(np.abs(FU) ** 2))
        if denom == 0:
            relcon = 0.0
        else:
            relcon = np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom
        if verbose:
            print(f'  relcon: {relcon:.3e}')
        if relcon < ccreltol:
            break

    U = np.real(np.fft.ifft2(FU, axes=(0, 1)))

    # Per-channel value-range constraint
    for c in range(C):
        lo, hi = vrange[c, 0], vrange[c, 1]
        ch = U[..., c]
        ch[ch < lo] = lo
        ch[ch > hi] = hi
        U[..., c] = ch

    if squeeze_out:
        U = U[..., 0]
    return U


# ═════════════════════════════════════════════════════════════════════════════
# mc_restoration  (MCrestoration.m)
# ═════════════════════════════════════════════════════════════════════════════

def mc_restoration(
    G: np.ndarray,
    hsize: Tuple[int, int],
    PAR: Dict,
    MSlevels: int = 4,
    maxROIsize: Tuple[int, int] = (1024, 1024),
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Top-level multiscale (coarse-to-fine) blind deconvolution pipeline.

    Mirrors MATLAB MCrestoration.m:

        1. Normalize image intensities to [0, 1].
        2. Build coarse-to-fine pyramid of the central ROI (green channel
           for RGB inputs).
        3. Initialize PSF as a delta impulse at the coarsest level.
        4. For each pyramid level: alternating MAP via
           psf_estim_lno_rgrad, then upsample PSF (lanczos3, ×2).
        5. Constrain PSF (>=0, sum to 1).
        6. Run final non-blind deconvolution on the full (colour) image.
        7. Denormalize back to original intensity range.

    Parameters
    ----------
    G        : input image (mono or RGB), any range (will be normalized).
    hsize    : (kh, kw) upper bound on PSF size at the FINEST scale.
    PAR      : parameters dict.
    MSlevels : number of pyramid levels (>=1).  1 = no pyramid.
    maxROIsize : (h, w) center ROI size used for kernel estimation.
    Hstar    : optional ground-truth PSF (for diagnostics).

    Returns
    -------
    U      : restored image, same shape as G, dtype float64
    H      : estimated PSF (sums to 1)
    report : dict with per-scale Reports and the parameters used.
    """
    G = np.asarray(G, dtype=np.float64)

    # 1. Normalise intensities to [0, 1]
    Gn, norm_m, norm_v = simpnormimg(G)

    # 2. Multiscale ROI pyramid
    L = max(1, int(MSlevels))
    ROI: List[np.ndarray] = [None] * L
    HstarP: List[Optional[np.ndarray]] = [None] * L

    ROI[L - 1] = get_roi(Gn, tuple(maxROIsize))
    if Hstar is not None:
        HstarP[L - 1] = np.asarray(Hstar, dtype=np.float64)

    for i in range(L - 2, -1, -1):
        ROI[i] = imresize_matlab(ROI[i + 1], 0.5, method='bicubic')
        if HstarP[i + 1] is not None:
            HstarP[i] = imresize_matlab(HstarP[i + 1], 0.5, method='bicubic')

    # 3. Initial PSF size (coarsest level) and delta init
    hsize0 = (int(np.ceil(hsize[0] / (2 ** (L - 1)))),
              int(np.ceil(hsize[1] / (2 ** (L - 1)))))
    cen = ((hsize0[0]) // 2, (hsize0[1]) // 2)  # MATLAB: floor((hsize+1)/2)-1
    # MATLAB floor((hsize+1)/2) is 1-based; 0-based equivalent:
    cen = ((hsize0[0] + 1) // 2 - 1, (hsize0[1] + 1) // 2 - 1)
    hi = np.zeros(hsize0, dtype=np.float64)
    hi[cen[0], cen[1]] = 1.0

    verbose = int(PAR.get('verbose', 0))
    if verbose:
        print('Estimating PSFs...')

    report = {'ms': [None] * L}

    # 4. Per-level alternating MAP
    h_current = hi
    for i in range(L):
        if verbose:
            print(f'hsize: {h_current.shape}')
        s = h_current.sum()
        if s != 0:
            h_current = h_current / s
        H_est, _U_est, rep_i = psf_estim_lno_rgrad(
            ROI[i], h_current, PAR, HstarP[i]
        )
        report['ms'][i] = rep_i
        if i < L - 1:
            # upsample by 2 with lanczos3 for the next finer level
            h_current = imresize_matlab(H_est, 2.0, method='lanczos3')
            # Inter-scale anchoring: lanczos3 upsampling can introduce
            # small ringing and shift the centroid by up to 1 pixel; we
            # re-center with the same threshold so the next coarser-to-
            # finer level starts from a properly centered initial PSF.
            ct = float(PAR.get('centering_threshold', 20.0 / 255.0))
            if ct > 0:
                # use a *milder* threshold here (×0.5) so we don't crop
                # legitimate faint extensions of curved PSFs
                h_current = center_psf(h_current, max(ct * 0.5, 1e-3))
        else:
            h_current = H_est

    # 5. Constrain PSF (sum-normalize) — match MATLAB MCrestoration.m exactly:
    #     H = h;
    #     H(H<0) = 0;             % clipped only to compute the normaliser
    #     H = h / sum(H(:));      % numerator is the ORIGINAL h (with negatives)
    H_pos = h_current.copy()
    H_pos[H_pos < 0] = 0.0
    s = H_pos.sum()
    if s != 0:
        H = h_current / s
    else:
        H = h_current.copy()

    if verbose:
        print('PSF estimation done.')

    # 6. Final non-blind deconvolution on the full (colour) image
    U = fft_cg_sr_al(Gn, H, PAR)
    if verbose:
        print('Nonblind deconvolution done.')

    # 7. Denormalize
    U = denormimg(U, norm_m, norm_v)

    report['par'] = PAR
    return U, H, report
