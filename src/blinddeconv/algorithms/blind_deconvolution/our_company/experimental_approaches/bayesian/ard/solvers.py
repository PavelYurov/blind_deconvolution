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

def _get_roi(g: np.ndarray, win, shift=(0, 0)) -> np.ndarray:

    isize = g.shape
    gsize = np.array(isize[:2])
    win = np.array(win)
    if g.ndim > 2 and g.shape[2] > 1:
        cind = 1
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

def psf_estim_ard(G: np.ndarray, iH: np.ndarray, PAR: dict,
                  iParam: dict | None = None):

    if iParam is None:
        iParam = {}

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

    U = np.zeros(usize, dtype=np.float64)
    H = iH.astype(np.float64).copy()
    FH = fft2(H, s=usize, axes=(0, 1))
    FU = fft2(U)

    FDx = fft2(np.array([[1.0, -1.0]]), s=usize)
    FDy = fft2(np.array([[1.0], [-1.0]]), s=usize)
    FD = np.stack([FDx, FDy], axis=2)

    FA = np.stack([
        np.conj(fft2(np.array([[1.0, 1.0]]), s=usize)),
        np.conj(fft2(np.array([[1.0], [1.0]]), s=usize)),
    ], axis=2)

    FUx = np.zeros(usize, dtype=np.complex128)
    FUy = np.zeros(usize, dtype=np.complex128)

    eG = np.zeros((usize[0], usize[1], P), dtype=np.float64)

    if 'covH' in iParam:
        covH = np.array(iParam['covH'], dtype=np.float64, copy=True)
    else:
        covH = np.zeros((usize[0], usize[1], P), dtype=np.float64)
    covU = np.zeros(usize, dtype=np.float64)

    for p in range(P):
        if is_mask:
            eG[:, :, p] = G[:, :, p]
        else:
            kuni = np.ones(hsize) / np.prod(hsize)
            eG[:, :, p] = edgetaper(G[:, :, p], kuni)

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

    uprior = PAR.get('uprior', {'type': 0})
    uprior_type = int(uprior.get('type', 0))
    image_model = uprior.get('model', (0.0, 1e-4))
    if 'uPrior_prec' in iParam:
        uPrior_prec = np.array(iParam['uPrior_prec'], dtype=np.float64, copy=True)
    else:

        init_model = (0.0, 1.0 / (float(np.atleast_1d(PAR['gamma'])[0]) *
                                  float(np.atleast_1d(PAR['alpha'])[0])))
        uPrior_prec = update_g_prior(np.zeros(FD.shape), np.zeros(usize),
                                     init_model)

    for mI in range(1, maxiter + 1):
        if verbose:
            print(f"Iteration: {mI}")

        alphagamma_vec = gamma_vec * alpha.reshape(1, 1, P)

        FeGu = fft2(eG * alphagamma_vec, axes=(0, 1))

        FUp = FU.copy()

        FU, U, FUx, FUy, covU = _ustep(
            FH, FeGu, alphagamma_vec, covH, uPrior_prec,
            FA, FD, FDx, FDy, usize, P, FU, reltol, delta_pdf,
        )
        Report['covH'].append(covH.reshape(-1, P).sum(axis=0))

        H, FH, covH = _hstep(
            U, FU, eG, alphagamma_vec, covU, beta_vec, FH, H,
            usize, hsize, P, delta_pdf,
        )
        Report['covU'].append(float(covU.sum()))

        if uprior_type == 0:
            DU = np.real(ifft2(np.stack([FUx, FUy], axis=2), axes=(0, 1)))
            uPrior_prec = update_g_prior(DU, covU, image_model)
        else:
            raise NotImplementedError("Only ARD image prior (type=0) is ported.")

        alpha, gamma, gamma_vec, d = _alpha_gamma_step(
            U, H, FU, FH, eG, covU, covH, gamma_vec, mask, alpha, d,
            is_mask, alpha_a0, alpha_b0, gamma_a0, gamma_b0,
            d_a0, d_b0, usize, P, verbose,
        )
        Report['alpha'].append(alpha.copy())
        Report['d'].append(d.copy())
        Report['gamma'].append(gamma.copy())

        beta, beta_vec = _beta_step(FH, covH, beta_a0, beta_b0, hsize, P)
        Report['beta'].append(beta)

        denom = np.sqrt(np.sum(np.abs(FU) ** 2))
        if denom == 0:
            relcon = 0.0
        else:
            relcon = float(np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom)
        if relcon < ccreltol:
            break

    H_out = H[:hsize[0], :hsize[1], :].copy()
    if H_out.shape[2] == 1:
        H_out = H_out[:, :, 0]
    Param: dict = {}
    return H_out, np.real(ifft2(FU)), Report, Param, gamma_vec

def _ustep(FH, FeGu, alphagamma_vec, covH, uPrior_prec,
           FA, FD, FDx, FDy, usize, P, FU, reltol, delta_pdf):

    covHTgammaH = np.sum(
        np.real(ifft2(fft2(alphagamma_vec, axes=(0, 1)) *
                      np.conj(fft2(covH, axes=(0, 1))), axes=(0, 1))),
        axis=2,
    )

    H_real_sq_fft = fft2(np.real(ifft2(FH, axes=(0, 1))) ** 2, axes=(0, 1))
    diagHTgammaH = np.sum(
        np.real(ifft2(np.conj(H_real_sq_fft) *
                      fft2(alphagamma_vec, axes=(0, 1)), axes=(0, 1))),
        axis=2,
    )

    appPrior = np.real(ifft2(np.sum(FA * fft2(uPrior_prec, axes=(0, 1)), axis=2)))

    b = np.sum(np.conj(FH) * FeGu, axis=2)

    def gradcalcFU(x_flat):
        X = unvec(x_flat, usize)

        T = ifft2(FH * X[:, :, None], axes=(0, 1)) * alphagamma_vec
        g = np.sum(np.conj(FH) * fft2(T, axes=(0, 1)), axis=2)

        g = g + fft2(covHTgammaH * ifft2(X))

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

    H = np.maximum(H, 0.0)
    FH = fft2(H, s=usize, axes=(0, 1))

    if not delta_pdf:
        covH_full = 1.0 / (diagUTgammaU + covUTgammaU + beta_vec)

        covH_full[hsize[0]:, :, :] = 0.0
        covH_full[:hsize[0], hsize[1]:, :] = 0.0
    else:
        covH_full = np.zeros((usize[0], usize[1], P), dtype=np.float64)

    return H, FH, covH_full

def _alpha_gamma_step(U, H, FU, FH, eG, covU, covH, gamma_vec, mask,
                      alpha, d, is_mask, alpha_a0, alpha_b0,
                      gamma_a0, gamma_b0, d_a0, d_b0, usize, P, verbose):

    E = np.zeros((usize[0], usize[1], 4, P), dtype=np.float64)
    FcovU = fft2(covU)
    FcovH = fft2(covH, axes=(0, 1))

    E[:, :, 0, :] = np.abs(
        ifft2(FU[:, :, None] * FH - fft2(eG, axes=(0, 1)), axes=(0, 1))
    ) ** 2

    E[:, :, 1, :] = np.real(
        ifft2(fft2(U ** 2)[:, :, None] * FcovH, axes=(0, 1))
    )

    E[:, :, 2, :] = np.real(
        ifft2(fft2(H ** 2, s=usize, axes=(0, 1)) * FcovU[:, :, None], axes=(0, 1))
    )

    E[:, :, 3, :] = np.real(
        ifft2(FcovH * FcovU[:, :, None], axes=(0, 1))
    )

    sumE = E.sum(axis=2)
    nnz_mask = float(np.count_nonzero(mask))

    ns = int(nnz_mask)

    if is_mask in (0, 1):
        gE = sumE * gamma_vec
        alpha = (alpha_a0 + 0.5 * nnz_mask) /\
                (alpha_b0 + 0.5 * gE.reshape(-1, P).sum(axis=0))
    elif is_mask == 3:
        gE = sumE * gamma_vec
        alpha = (alpha_a0 + 0.5 * ns) /\
                (alpha_b0 + 0.5 * gE.reshape(-1, P).sum(axis=0))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))

    aE = alpha.reshape(1, 1, P) * sumE
    aE = aE * mask[:, :, None]
    if is_mask == 2:
        gamma_vec = (gamma_a0 + 0.5) / (gamma_b0 + 0.5 * aE)
        gamma_vec = gamma_vec * mask[:, :, None]
    elif is_mask == 3:
        rd = d.reshape(1, 1, P)
        gamma_vec = (rd + 0.5) / (rd + 0.5 * aE)
        gamma_vec = gamma_vec * mask[:, :, None]

        digamma_term = 1.0 + digamma(d + 0.5)

        log_arg = rd + 0.5 * aE
        log_term = np.real(np.log(np.abs(log_arg) + 1e-300))
        d = (d_a0 + 0.5 * ns) / (
            d_b0
            + gamma_vec.reshape(-1, P).sum(axis=0)
            - ns * digamma_term
            + log_term.reshape(-1, P).sum(axis=0)
        )

        if np.any(d < 0) or np.any(~np.isfinite(d)):
            d = np.where((d < 0) | (~np.isfinite(d)), 1.0, d)

    gamma = gamma_vec.reshape(-1, P).mean(axis=0)

    if verbose:
        print(f"alpha: {alpha}, d: {d}")

    return alpha, gamma, gamma_vec, d

def _beta_step(FH, covH, beta_a0, beta_b0, hsize, P):

    EH = np.abs(ifft2(FH, axes=(0, 1))) ** 2 + covH
    n_h = int(np.prod(hsize))
    beta = float((beta_a0 + P * n_h / 2.0) /
                 (beta_b0 + 0.5 * EH.sum()))
    beta_vec = (beta_a0 + 0.5) / (beta_b0 + 0.5 * EH)
    return beta, beta_vec

def mc_restoration(g: np.ndarray, hsize, params: dict):

    PAR = params['PAR']

    gamma_corr = float(params.get('gamma_corr', 1.0))
    if gamma_corr != 1.0:
        g = g ** gamma_corr

    do_ARDnoise = int(PAR.get('ARDnoise', 0))
    L = max(1, int(PAR.get('MSlevels', 1)))
    factor = float(PAR.get('factor', 1.5))
    sp = factor ** np.arange(L - 1, -1, -1)

    ROI = [None] * L
    ROI[L - 1] = _get_roi(g, PAR['maxROIsize'])
    for i in range(L - 2, -1, -1):
        ROI[i] = dsample(ROI[L - 1], sp[i], 'valid')

    hsize_arr = np.array(hsize, dtype=np.float64)
    hsize_list = np.ceil(np.tile(hsize_arr, (L, 1)) /
                         sp.reshape(-1, 1)).astype(int)
    hs = hsize_list[0]
    hs = 2 * (hs // 2) + 1
    h = initblur((hs[0], hs[1]),
                 ((hs[0] + 1) / 2.0, (hs[1] + 1) / 2.0),
                 (1, 1))

    psf_method = params.get('psf_method', 'ard')
    if psf_method != 'ard':
        raise NotImplementedError(f"psf_method={psf_method!r} is not ported.")

    pyramid_thresh = float(params.get('pyramid_thresh', 0.0))

    param: dict = {}
    gamma_vec = None
    for i in range(L):

        if do_ARDnoise:
            PAR['ARDnoise'] = do_ARDnoise
            if do_ARDnoise == 1 and i != L - 1:
                PAR['ARDnoise'] = 0

        h, _u, _report, param, gamma_vec = psf_estim_ard(
            ROI[i], h, PAR, param
        )

        if i < L - 1:

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

        h = np.where(h < 0, 0.0, h)
        s = h.sum()
        if s > 0:
            h = h / s
    return h, gamma_vec

def vb_deconv(G_list, H_list, params: dict):

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

    H0 = np.asarray(H_list[0], dtype=np.float64)
    hsize = H0.shape

    vrange = np.zeros((C, 2), dtype=np.float64)
    for c in range(C):
        lo = min(np.asarray(g, dtype=np.float64)[..., c].min()
                 if np.asarray(g).ndim == 3 else np.asarray(g, dtype=np.float64).min()
                 for g in G_list)
        hi = max(np.asarray(g, dtype=np.float64)[..., c].max()
                 if np.asarray(g).ndim == 3 else np.asarray(g, dtype=np.float64).max()
                 for g in G_list)
        vrange[c] = [lo, hi]

    mask = np.zeros((H, W), dtype=np.float64)
    hh = (hsize[0] // 2, hsize[1] // 2)
    mask[hh[0]:H - hh[0], hh[1]:W - hh[1]] = 1.0
    gamma_vec = gamma * np.broadcast_to(
        mask[:, :, None, None], (H, W, C, P)).copy()

    FDx = np.broadcast_to(fft2(np.array([[1.0, -1.0]]), s=usize)[:, :, None],
                          (H, W, C)).copy()
    FDy = np.broadcast_to(fft2(np.array([[1.0], [-1.0]]), s=usize)[:, :, None],
                          (H, W, C)).copy()
    FD = np.stack([FDx, FDy], axis=3)
    FA = np.stack([
        np.broadcast_to(np.conj(fft2(np.array([[1.0, 1.0]]), s=usize))[:, :, None],
                        (H, W, C)).copy(),
        np.broadcast_to(np.conj(fft2(np.array([[1.0], [1.0]]), s=usize))[:, :, None],
                        (H, W, C)).copy(),
    ], axis=3)

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

    FH_pad = fft2(eH, s=usize, axes=(0, 1))
    FHS = np.broadcast_to(
        Fspsf[:, :, None, None] * FH_pad,
        (H, W, C, P)
    ).copy()

    U = np.zeros((H, W, C), dtype=np.float64)
    FU = fft2(U, axes=(0, 1))
    covU = np.zeros((H, W, C), dtype=np.float64)

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

        diagHTgammaH = np.real(
            ifft2(np.conj(fft2(np.real(ifft2(FHS, axes=(0, 1))) ** 2,
                               axes=(0, 1))) *
                  fft2(gamma_vec, axes=(0, 1)), axes=(0, 1))
        )
        appPrior = np.real(
            ifft2(np.sum(FA * fft2(uPrior_prec, axes=(0, 1)), axis=3),
                  axes=(0, 1))
        )

        b = np.sum(np.conj(FHS) * FeGu, axis=3)

        def gradcalcFU_gammavec(x_flat):
            X = unvec(x_flat, (H, W, C))
            T = FHS * X[:, :, :, None]
            T = fft2(gamma_vec * ifft2(T, axes=(0, 1)), axes=(0, 1))
            g = np.sum(np.conj(FHS) * T, axis=3)
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

        DU = np.real(ifft2(np.stack([FUx, FUy], axis=3), axes=(0, 1)))
        uPrior_prec = update_g_prior(DU, covU, image_model)

        E = np.zeros((H, W, C, P, 4), dtype=np.float64)
        E[:, :, :, :, 0] = np.abs(
            ifft2(FU[:, :, :, None] * FHS - fft2(eG, axes=(0, 1)), axes=(0, 1))
        ) ** 2
        FcovU = fft2(covU, axes=(0, 1))
        E[:, :, :, :, 2] = np.real(
            ifft2(fft2(np.real(ifft2(FHS, axes=(0, 1))) ** 2, axes=(0, 1)) *
                  FcovU[:, :, :, None], axes=(0, 1))
        )

        sum_over_C_E = np.sum(np.sum(E, axis=4), axis=2)
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

def frils_deb_ubc(y: np.ndarray, h: np.ndarray, opt: dict) -> np.ndarray:

    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = pad_replicate(y, hks1, hks2)

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

            u = Ax + du
            inner = u[hks1:n1 - hks1, hks2:n2 - hks2]
            inner = (y + lambda_u * inner) / (1.0 + lambda_u)
            u[hks1:n1 - hks1, hks2:n2 - hks2] = inner

            vx  = beta_a * (dx  + dvx)  / (Wx  + beta_a)
            vy  = beta_a * (dy  + dvy)  / (Wy  + beta_a)
            vxx = beta_a * (dxx + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy + dvxy) / (Wxy + beta_a)

            du   = du   - u   + Ax
            dvx  = dvx  - vx  + dx
            dvy  = dvy  - vy  + dy
            dvxx = dvxx - vxx + dxx
            dvyy = dvyy - vyy + dyy
            dvxy = dvxy - vxy + dxy

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
