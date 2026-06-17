"""
solvers.py

Core solver functions for Variational Bayesian Pansharpening / Super-Resolution.

Ported from MATLAB code:
    Pérez-Bueno, F., Vega, M., Mateos, J., Molina, R., & Katsaggelos, A. K. (2020).
    Variational Bayesian Pansharpening with Super-Gaussian Sparse Image Priors.
    Sensors, 20(18), 5308.

    M. Vega, J. Mateos, R. Molina, and A. K. Katsaggelos, "Super resolution of
    multispectral images using TV image models," KES 2008, pp. 408-415.

Contains:
    restoreSAR         — SAR denoising/deblurring (initial hyperparameter estimation)
    alfaTVpvini        — initial alpha for TV prior
    alfaSGlogvini      — initial alpha for SG log prior
    alfaSGlpvini       — initial alpha for SG lp prior
    restSGME_Sens      — main SG (log / lp) pansharpening algorithm
    TVME_Sens          — TV-prior pansharpening algorithm
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.sparse.linalg import cg as scipy_cg
from scipy.signal import convolve2d
from scipy.ndimage import zoom

from .utils import (
    im_decomp, im_comp,
    blk_fft2, blk_ifft2, blk_fd_conv, blk_fd_trace,
    blk_fft2_DtD, blk_DtY,
    cent_nucleus2fft, Tcent_nucleus2fft,
    cent_nucleus2blk_fft2, Tcent_nucleus2blk_fft2,
    circ_gradient2, Tcirc_gradient2,
    convertToMBVec, convertToMBImg,
    compute_Wpvb, getfilters, getkappa,
    _EPS,
)


# ═════════════════════════════════════════════════════════════════════════════
# restoreSAR  (from restoreSAR.m)
# ═════════════════════════════════════════════════════════════════════════════

def restoreSAR(img, h, term=1e-6, nitermax=100):
    """Bayesian image denoising / deblurring with a SAR (Simultaneous
    Auto-Regressive) prior.  Used for **initial hyperparameter estimation**.

    Parameters
    ----------
    img      : (M, N) observed image
    h        : 2-D convolution kernel  (use ``np.array([[1]])`` for denoising)
    term     : float — convergence threshold
    nitermax : int — max iterations

    Returns
    -------
    out   : (M, N) restored image
    alpha : float — estimated prior precision
    beta  : float — estimated noise precision

    Reference
    ---------
    R. Molina, A.K. Katsaggelos, J. Mateos, "Bayesian and Regularization
    Methods for Hyperparameter Estimation in Image Restoration",
    IEEE TIP, 8(2), 231-246, 1999.
    """
    img = np.asarray(img, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)

    M, N = img.shape[:2]
    tamm = M * N

    g = fft2(img)

    H = cent_nucleus2fft(h, M, N)
    Ht = Tcent_nucleus2fft(h, M, N)
    HtH = Ht * H

    # SAR prior kernel  C^T C
    priorn = np.array([[0, -0.25, 0],
                       [-0.25, 1, -0.25],
                       [0, -0.25, 0]])
    priorn = convolve2d(priorn, priorn, mode='full')
    prior = cent_nucleus2fft(priorn, M, N)

    dif = g - H * g
    dif_energy = np.real(np.sum(np.conj(dif) * dif))

    alpha0 = np.real((tamm - 1.0) * tamm /
                     np.sum(np.conj(g) * (prior * g)))
    if dif_energy > 0:
        beta0 = np.real(tamm * tamm / dif_energy)
    else:
        beta0 = 100.0 * alpha0
    if beta0 > 1e4 * alpha0:
        beta0 = 100.0 * alpha0

    Q = beta0 * HtH + alpha0 * prior
    f = beta0 * Ht * g / Q
    f0 = f.copy()

    alpha = alpha0
    beta = beta0

    for it in range(1, nitermax + 1):
        alpha = np.real(
            (tamm - 1.0) /
            (np.sum(np.conj(f) * (prior * f)) / tamm + np.sum(prior / Q))
        )
        beta = np.real(
            tamm /
            (np.sum(np.conj(g - H * f) * (g - H * f)) / tamm + np.sum(HtH / Q))
        )

        Q = beta * HtH + alpha * prior
        f = beta * Ht * g / Q

        t3 = np.real(np.sum(np.conj(f - f0) * (f - f0)) /
                     np.sum(np.conj(f0) * f0))
        f0 = f.copy()
        alpha0 = alpha
        beta0 = beta

        if t3 <= term:
            break

    alpha = float(np.real(alpha))
    beta = float(np.real(beta))
    out = np.real(ifft2(f))
    return out, alpha, beta


# ═════════════════════════════════════════════════════════════════════════════
# Initial alpha estimators
# ═════════════════════════════════════════════════════════════════════════════

def alfaTVpvini(x, p=2):
    """Initial alpha for the TV prior.

    alpha = sum(p_ij) / (4 * sum( (|grad(x)|^2)^(1/p) ))
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3:
        x = x[:, :, 0]
    M, N = x.shape
    pmat = p * np.ones((M, N))
    Dhx, Dvx = circ_gradient2(x)
    v = Dhx ** 2 + Dvx ** 2
    v = v ** (1.0 / pmat)
    return float(np.sum(pmat) / (4.0 * np.sum(v) + _EPS))


def alfaSGlogvini(Y, filtersetname, epsW=1e-6):
    """Initial alpha values for the SG *log* prior.

    Parameters
    ----------
    Y              : (M, N, nbands) or (M, N)
    filtersetname  : str
    epsW           : float

    Returns
    -------
    alpha : list of (nbands,) arrays — one per filter
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    M, N, nbands = Y.shape
    filters = getfilters(filtersetname)
    nfilters = len(filters)

    kappa = getkappa('log')
    kappa_f = kappa[0]
    alpha_f = kappa[2]

    alpha = [np.zeros(nbands) for _ in range(nfilters)]

    for nu in range(nfilters):
        Fnu = cent_nucleus2fft(filters[nu], M, N)
        for i in range(nbands):
            xF = fft2(Y[:, :, i])
            xnu = ifft2(Fnu * xF)
            u = epsW + np.abs(xnu * xnu) ** 0.5
            W = kappa_f(u)
            val = float(np.sum(np.real(W))) + _EPS
            alpha[nu][i] = alpha_f(val)

    return alpha


def alfaSGlpvini(Y, p, filtersetname, epsW=1e-5):
    """Initial alpha values for the SG *lp* prior.

    Parameters
    ----------
    Y              : (M, N, nbands) or (M, N)
    p              : float — lp exponent
    filtersetname  : str
    epsW           : float

    Returns
    -------
    alpha : list of (nbands,) arrays — one per filter
    """
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    M, N, nbands = Y.shape
    filters = getfilters(filtersetname)
    nfilters = len(filters)

    kappa = getkappa('lp', p)
    kappa_f = kappa[0]
    alpha_f = kappa[2]

    alpha = [np.zeros(nbands) for _ in range(nfilters)]

    for nu in range(nfilters):
        Fnu = cent_nucleus2fft(filters[nu], M, N)
        for i in range(nbands):
            xF = fft2(Y[:, :, i])
            xnu = ifft2(Fnu * xF)
            u = epsW + np.abs(xnu * xnu) ** 0.5
            W = kappa_f(u)
            val = float(np.sum(np.real(W))) + _EPS
            alpha[nu][i] = alpha_f(val)

    return alpha


# ═════════════════════════════════════════════════════════════════════════════
# restSGME_Sens  (from restSGME_Sens.m — MAIN SG ALGORITHM)
# ═════════════════════════════════════════════════════════════════════════════

def restSGME_Sens(
    Y, x, lam, kappa, filtersetname, hnuclei, nbands,
    eps_map=1e-4, itmax_map=50, itmin_map=2,
    gamma_alpha=None, gamma_beta=None, gamma_gamma=0.0,
    alpha_mode=None, beta_mode=None, gamma_mode=1.0,
    eps_y=1e-7, itmax_y=30,
    verbose=False,
):
    """Variational Bayesian Pansharpening with Super-Gaussian priors.

    Solves:
        Y_b = D H y_b + n_b       (MS observation)
        x   = sum_b lam_b y_b + η  (PAN observation)

    using SG (log or lp) sparsity priors on filtered images {F_ν y_b}.

    Parameters
    ----------
    Y              : (lr_h, lr_w, nbands) LR multispectral observation
    x              : (hr_h, hr_w) HR panchromatic observation
    lam            : (nbands,) lambda coefficients
    kappa          : [kappa_f, rho_f, alpha_f] from getkappa()
    filtersetname  : 'fohv' or 'fo'
    hnuclei        : PSF kernel (2-D array) or list of per-band kernels
    nbands         : int
    eps_map        : convergence threshold
    itmax_map      : max outer iterations
    itmin_map      : min outer iterations
    gamma_alpha    : (nbands,) Gamma hyperprior confidence for alpha
    gamma_beta     : (nbands,) Gamma hyperprior confidence for beta
    gamma_gamma    : float — Gamma hyperprior confidence for gamma
    alpha_mode     : list of (nbands,) reference alpha values per filter
    beta_mode      : (nbands,) reference beta values
    gamma_mode     : float — reference gamma value
    eps_y          : CG convergence tolerance
    itmax_y        : CG max iterations
    verbose        : bool

    Returns
    -------
    y      : (hr_h, hr_w, nbands) reconstructed HR MS image
    alpha  : list of (nbands,) estimated alpha per filter
    beta   : (nbands,) estimated beta
    gamma  : float estimated gamma
    W      : list of (hr_h, hr_w, nbands) weight matrices
    """
    Y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)

    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    lr_h, lr_w = Y.shape[:2]
    hr_h, hr_w = x.shape[:2]
    SRratio = hr_h // lr_h

    # Defaults
    vec1 = np.ones(nbands)
    if gamma_alpha is None:
        gamma_alpha = np.zeros(nbands)
    if gamma_beta is None:
        gamma_beta = np.zeros(nbands)
    if beta_mode is None:
        beta_mode = np.ones(nbands)

    filters = getfilters(filtersetname)
    nfilters = len(filters)

    if alpha_mode is None:
        alpha_mode = [np.ones(nbands) for _ in range(nfilters)]

    epsW = float(np.mean(Y)) * 1e-5

    # ── Pre-compute filter FFTs ──────────────────────────────────────────
    filtersT = []
    filterTfilter = []
    Df = []
    Dft = []
    DftDf = []

    for nu in range(nfilters):
        fT = np.flip(np.flip(filters[nu], axis=0), axis=1)
        filtersT.append(fT)
        fTf = convolve2d(fT, filters[nu], mode='full')
        filterTfilter.append(fTf)
        Df.append(cent_nucleus2fft(filters[nu], hr_h, hr_w))
        Dft.append(cent_nucleus2fft(fT, hr_h, hr_w))
        DftDf.append(
            cent_nucleus2blk_fft2(fTf, hr_h, hr_w, SRratio, SRratio)
        )

    # ── Pre-compute observation model operators ──────────────────────────
    DtD = blk_fft2_DtD(hr_h, hr_w, SRratio, SRratio)

    psf_is_list = isinstance(hnuclei, list)
    if psf_is_list:
        H_blk = []
        Ht_blk = []
        HtDtDH = []
        for i in range(nbands):
            Hi = cent_nucleus2blk_fft2(hnuclei[i], hr_h, hr_w, SRratio, SRratio)
            Hti = Tcent_nucleus2blk_fft2(hnuclei[i], hr_h, hr_w, SRratio, SRratio)
            H_blk.append(Hi)
            Ht_blk.append(Hti)
            DtDH = blk_fd_conv(DtD, Hi)
            HtDtDH.append(blk_fd_conv(Hti, DtDH))
    else:
        H_blk = cent_nucleus2blk_fft2(hnuclei, hr_h, hr_w, SRratio, SRratio)
        Ht_blk = Tcent_nucleus2blk_fft2(hnuclei, hr_h, hr_w, SRratio, SRratio)
        DtDH = blk_fd_conv(DtD, H_blk)
        HtDtDH = blk_fd_conv(Ht_blk, DtDH)

    # ── Independent terms (constant across iterations) ───────────────────
    indep_wo_gamma = np.zeros((hr_h, hr_w, nbands))
    indep_wo_beta = np.zeros((hr_h, hr_w, nbands))

    for i in range(nbands):
        indep_wo_gamma[:, :, i] = lam[i] * x
        dtY = blk_DtY(Y[:, :, i], SRratio, SRratio)
        dtY_f = blk_fft2(dtY)
        if psf_is_list:
            v = blk_fd_conv(Ht_blk[i], dtY_f)
        else:
            v = blk_fd_conv(Ht_blk, dtY_f)
        indep_wo_beta[:, :, i] = np.real(im_comp(blk_ifft2(v), SRratio, SRratio))

    # ── Initial estimate: bicubic upsample ───────────────────────────────
    y0_img = np.zeros((hr_h, hr_w, nbands))
    for i in range(nbands):
        y0_img[:, :, i] = zoom(Y[:, :, i], SRratio, order=3)

    trgamma = 0.0
    tralpha = np.zeros(nbands)
    trbeta = np.zeros(nbands)

    y0 = convertToMBVec(y0_img)
    y_img = convertToMBImg(y0, hr_h, hr_w, nbands)

    conv_crit = eps_map + 1.0
    iteration = 0

    # ── Main iteration loop ──────────────────────────────────────────────
    while iteration <= itmin_map or (conv_crit > eps_map and iteration < itmax_map):
        iteration += 1
        if verbose:
            print(f"iter = {iteration}")

        # --- Update hyperparameters ---
        alpha, beta, gamma_val, W = _sg_update_params(
            x, Y, H_blk, y_img, lam, nfilters, Df, kappa,
            tralpha, trbeta, trgamma, SRratio, nbands,
            gamma_alpha, gamma_beta, gamma_gamma,
            alpha_mode, beta_mode, gamma_mode, epsW,
            psf_is_list,
        )

        if verbose:
            for nu in range(nfilters):
                print(f"  alpha[{nu}] = {alpha[nu]}")
            print(f"  beta = {beta}")
            print(f"  gamma = {gamma_val}")

        # --- Build RHS ---
        indep_term = np.zeros((hr_h, hr_w, nbands))
        for i in range(nbands):
            indep_term[:, :, i] = (gamma_val * indep_wo_gamma[:, :, i] +
                                   beta[i] * indep_wo_beta[:, :, i])
        rhs = convertToMBVec(indep_term)

        # --- Solve Σ^{-1} y = rhs via CG ---
        def matvec(v):
            return _sg_multiply_by_invcov(
                v, nfilters, Df, Dft, alpha, W, beta, HtDtDH,
                gamma_val, lam, hr_h, hr_w, SRratio, nbands, psf_is_list,
            )

        from scipy.sparse.linalg import LinearOperator
        n_total = hr_h * hr_w * nbands
        A_op = LinearOperator((n_total, n_total), matvec=matvec, dtype=np.float64)
        y_vec, cg_info = scipy_cg(A_op, rhs, x0=y0, rtol=eps_y, maxiter=itmax_y)

        # --- Convergence check ---
        if iteration > 1:
            denom = np.dot(y0, y0)
            if denom > 0:
                conv_crit = np.dot(y_vec - y0, y_vec - y0) / denom
            else:
                conv_crit = 0.0
            if verbose:
                print(f"  ||y-y0||^2/||y0||^2 = {conv_crit:.6e}")

        y0 = y_vec.copy()

        # --- Trace computation for next iteration ---
        tralpha, trbeta, trgamma = _sg_calc_trazas(
            alpha, nfilters, DftDf, W, beta, HtDtDH, gamma_val, lam,
            (lr_h, lr_w), SRratio, nbands, psf_is_list,
        )

        y_img = convertToMBImg(y_vec, hr_h, hr_w, nbands)

    return y_img, alpha, beta, gamma_val, W


# ── SG helper: update hyperparameters ────────────────────────────────────
def _sg_update_params(
    x, Y, H_blk, y, lam, nfilters, Df, kappa,
    tralpha, trbeta, trgamma, bkn, nbands,
    gamma_alpha, gamma_beta, gamma_gamma,
    alpha_mode, beta_mode, gamma_mode, epsW,
    psf_is_list,
):
    M, N = y.shape[:2]
    ysupport = M * N
    observationsupport = ysupport / bkn / bkn

    kappa_f = kappa[0]
    alpha_f = kappa[2]

    aux4 = x.copy()
    W = [np.zeros((M, N, nbands)) for _ in range(nfilters)]
    E_alpha = [np.zeros(nbands) for _ in range(nfilters)]
    E_beta = np.zeros(nbands)

    for i in range(nbands):
        # Prior weights
        for nu in range(nfilters):
            xnu = ifft2(Df[nu] * fft2(y[:, :, i]))
            u = epsW + np.abs(xnu * xnu + tralpha[i] / ysupport) ** 0.5
            W[nu][:, :, i] = np.real(kappa_f(u))
            val = float(np.mean(W[nu][:, :, i])) + _EPS
            inv_E = gamma_alpha[i] / alpha_mode[nu][i] + (1.0 - gamma_alpha[i]) / alpha_f(val)
            E_alpha[nu][i] = max(1.0 / inv_E, _EPS)

        # Observation error
        YB = blk_fft2(im_decomp(y[:, :, i], bkn, bkn))
        if psf_is_list:
            recon = np.real(im_comp(blk_ifft2(blk_fd_conv(H_blk[i], YB)), bkn, bkn))
        else:
            recon = np.real(im_comp(blk_ifft2(blk_fd_conv(H_blk, YB)), bkn, bkn))
        recon_lr = recon[::bkn, ::bkn]
        err = np.sum((Y[:, :, i] - recon_lr) ** 2) + _EPS

        inv_Eb = gamma_beta[i] / beta_mode[i] + (1.0 - gamma_beta[i]) * (err + trbeta[i]) / observationsupport
        E_beta[i] = max(1.0 / inv_Eb, _EPS)

        aux4 = aux4 - lam[i] * y[:, :, i]

    norma_gam = np.sum(aux4 ** 2)
    inv_Eg = gamma_gamma / gamma_mode + (1.0 - gamma_gamma) * (norma_gam + trgamma) / ysupport
    E_gamma = max(1.0 / inv_Eg, _EPS)

    return E_alpha, E_beta, E_gamma, W


# ── SG helper: matrix-vector product Σ^{-1} y ───────────────────────────
def _sg_multiply_by_invcov(
    y_vec, nfilters, Df, Dft, alpha, W, beta, HtDtDH,
    gamma_val, lam, nr, nc, bkn, nbands, psf_is_list,
):
    yd = convertToMBImg(y_vec, nr, nc, nbands)
    Ay = np.zeros((nr, nc, nbands))

    for i in range(nbands):
        ydbf = blk_fft2(im_decomp(yd[:, :, i], bkn, bkn))
        if psf_is_list:
            v = beta[i] * im_comp(blk_ifft2(blk_fd_conv(HtDtDH[i], ydbf)), bkn, bkn)
        else:
            v = beta[i] * im_comp(blk_ifft2(blk_fd_conv(HtDtDH, ydbf)), bkn, bkn)

        # Prior term: sum_nu alpha[nu][i] * F_nu^T * diag(W[nu][:,:,i]) * F_nu * y_i
        prior_term = np.zeros((nr, nc))
        ydf = fft2(yd[:, :, i])
        for nu in range(nfilters):
            temp = fft2(W[nu][:, :, i] * np.real(ifft2(Df[nu] * ydf)))
            prior_term += alpha[nu][i] * np.real(ifft2(Dft[nu] * temp))

        Ay[:, :, i] = v + prior_term

    # PAN coupling term: gamma * lam_i * lam_j * y_j
    for i in range(nbands):
        for j in range(nbands):
            Ay[:, :, i] += gamma_val * lam[i] * lam[j] * yd[:, :, j]

    return convertToMBVec(Ay)


# ── SG helper: trace computation ─────────────────────────────────────────
def _sg_calc_trazas(
    alpha, nfilters, DftDf, W, beta, HtDtDH, gamma_val, lam,
    lr_size, SRratio, nbands, psf_is_list,
):
    Qinv = _sg_calc_cov_inv(
        alpha, nfilters, DftDf, W, beta, HtDtDH, gamma_val, lam,
        lr_size, SRratio, nbands, psf_is_list,
    )

    M, N = W[0].shape[:2]
    trpancr = 0.0
    trprior = np.zeros(nbands)
    trobs = np.zeros(nbands)

    for i in range(nbands):
        for nu in range(nfilters):
            trprior[i] += blk_fd_trace(blk_fd_conv(Qinv[i][i], DftDf[nu])) / (M * N)
        if psf_is_list:
            trobs[i] = blk_fd_trace(blk_fd_conv(Qinv[i][i], HtDtDH[i]))
        else:
            trobs[i] = blk_fd_trace(blk_fd_conv(Qinv[i][i], HtDtDH))
        for j in range(nbands):
            trpancr += lam[i] * lam[j] * blk_fd_trace(Qinv[i][j])

    return trprior, trobs, trpancr


def _sg_calc_cov_inv(
    alpha, nfilters, DftDf, W, beta, HtDtDH, gamma_val, lam,
    lr_size, SRratio, nbands, psf_is_list,
):
    """Compute the block-inverse of the precision matrix Q for trace calculations.

    Returns a nested list  Qinv[i][j]  of shape (low_nr, low_nc, bkn2, bkn2).
    """
    low_nr, low_nc = lr_size
    bkn2 = SRratio * SRratio

    # Identity block
    I_blk = np.zeros((low_nr, low_nc, bkn2, bkn2))
    for k in range(bkn2):
        I_blk[:, :, k, k] = 1.0

    # Build Q[i][j] blocks  (6-D in MATLAB, here nested list of 4D)
    Q = [[None] * nbands for _ in range(nbands)]

    for i in range(nbands):
        if psf_is_list:
            diag_block = gamma_val * lam[i] * lam[i] * I_blk + beta[i] * HtDtDH[i]
        else:
            diag_block = gamma_val * lam[i] * lam[i] * I_blk + beta[i] * HtDtDH

        for nu in range(nfilters):
            z = float(np.mean(W[nu][:, :, i]))
            diag_block = diag_block + alpha[nu][i] * z * DftDf[nu]
        Q[i][i] = diag_block.copy()

        for j in range(i + 1, nbands):
            off_block = gamma_val * lam[i] * lam[j] * I_blk
            Q[i][j] = off_block.copy()
            Q[j][i] = off_block.copy()

    # Small diagonal regularisation to prevent singular blocks
    reg = _EPS * I_blk

    # For nbands==1 simplify: just invert the (bkn2 x bkn2) block at each freq
    if nbands == 1:
        Qinv = [[np.zeros_like(Q[0][0])]]
        for fi in range(low_nr):
            for fj in range(low_nc):
                mat = Q[0][0][fi, fj, :, :] + reg[fi, fj, :, :]
                Qinv[0][0][fi, fj, :, :] = np.linalg.inv(mat)
        return Qinv

    # General case: for each (fi, fj) frequency, build the full
    # (nbands*bkn2) x (nbands*bkn2) matrix and invert
    Qinv = [[np.zeros((low_nr, low_nc, bkn2, bkn2)) for _ in range(nbands)]
            for _ in range(nbands)]

    full_size = nbands * bkn2
    full_reg = _EPS * np.eye(full_size)
    for fi in range(low_nr):
        for fj in range(low_nc):
            full_mat = np.zeros((full_size, full_size), dtype=complex)
            for bi in range(nbands):
                for bj in range(nbands):
                    if Q[bi][bj] is not None:
                        r0, r1 = bi * bkn2, (bi + 1) * bkn2
                        c0, c1 = bj * bkn2, (bj + 1) * bkn2
                        full_mat[r0:r1, c0:c1] = Q[bi][bj][fi, fj, :, :]

            full_inv = np.linalg.inv(full_mat + full_reg)

            for bi in range(nbands):
                for bj in range(nbands):
                    r0, r1 = bi * bkn2, (bi + 1) * bkn2
                    c0, c1 = bj * bkn2, (bj + 1) * bkn2
                    Qinv[bi][bj][fi, fj, :, :] = full_inv[r0:r1, c0:c1]

    return Qinv


# ═════════════════════════════════════════════════════════════════════════════
# TVME_Sens  (from TVME_Sens.m — TV PRIOR ALGORITHM)
# ═════════════════════════════════════════════════════════════════════════════

def TVME_Sens(
    Y, x, lam, hnuclei, nbands,
    eps_map=1e-4, itmax_map=50, itmin_map=2,
    gamma_alpha=None, gamma_beta=None, gamma_gamma=0.0,
    alpha_mode=None, beta_mode=None, gamma_mode=1.0,
    eps_y=1e-7, itmax_y=30,
    verbose=False,
):
    """Variational Bayesian Pansharpening with Total-Variation prior.

    Parameters
    ----------
    Y              : (lr_h, lr_w, nbands) LR multispectral observation
    x              : (hr_h, hr_w) HR panchromatic observation
    lam            : (nbands,) lambda coefficients
    hnuclei        : PSF kernel (2-D) or list of per-band kernels
    nbands         : int
    eps_map, itmax_map, itmin_map : convergence params
    gamma_alpha, gamma_beta, gamma_gamma : Gamma hyperprior confidences
    alpha_mode, beta_mode, gamma_mode    : Gamma hyperprior reference values
    eps_y, itmax_y : CG params
    verbose        : bool

    Returns
    -------
    y      : (hr_h, hr_w, nbands) reconstructed HR image
    alpha  : (nbands,) estimated alpha
    beta   : (nbands,) estimated beta
    gamma  : float estimated gamma
    W      : (hr_h, hr_w, nbands) weight matrix
    """
    Y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)

    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    lr_h, lr_w = Y.shape[:2]
    hr_h, hr_w = x.shape[:2]
    SRratio = hr_h // lr_h

    epsW = 1e-5

    if gamma_alpha is None:
        gamma_alpha = np.zeros(nbands)
    if gamma_beta is None:
        gamma_beta = np.zeros(nbands)
    if alpha_mode is None:
        alpha_mode = np.ones(nbands)
    if beta_mode is None:
        beta_mode = np.ones(nbands)

    # ── TV difference kernels in block-FFT ──
    DhtDh_kern = np.array([[1.0, -2.0, 1.0]])
    DvtDv_kern = DhtDh_kern.T
    DhtDh = cent_nucleus2blk_fft2(DhtDh_kern, hr_h, hr_w, SRratio, SRratio)
    DvtDv = cent_nucleus2blk_fft2(DvtDv_kern, hr_h, hr_w, SRratio, SRratio)

    # ── Observation model operators ──
    DtD = blk_fft2_DtD(hr_h, hr_w, SRratio, SRratio)

    psf_is_list = isinstance(hnuclei, list)
    if psf_is_list:
        H_blk = []
        Ht_blk = []
        HtDtDH = []
        for i in range(nbands):
            Hi = cent_nucleus2blk_fft2(hnuclei[i], hr_h, hr_w, SRratio, SRratio)
            Hti = Tcent_nucleus2blk_fft2(hnuclei[i], hr_h, hr_w, SRratio, SRratio)
            H_blk.append(Hi)
            Ht_blk.append(Hti)
            DtDH_i = blk_fd_conv(DtD, Hi)
            HtDtDH.append(blk_fd_conv(Hti, DtDH_i))
    else:
        H_blk = cent_nucleus2blk_fft2(hnuclei, hr_h, hr_w, SRratio, SRratio)
        Ht_blk = Tcent_nucleus2blk_fft2(hnuclei, hr_h, hr_w, SRratio, SRratio)
        DtDH_single = blk_fd_conv(DtD, H_blk)
        HtDtDH = blk_fd_conv(Ht_blk, DtDH_single)

    # ── Constant RHS parts ──
    indep_wo_gamma = np.zeros((hr_h, hr_w, nbands))
    indep_wo_beta = np.zeros((hr_h, hr_w, nbands))

    for i in range(nbands):
        indep_wo_gamma[:, :, i] = lam[i] * x
        dtY = blk_DtY(Y[:, :, i], SRratio, SRratio)
        dtY_f = blk_fft2(dtY)
        if psf_is_list:
            v = blk_fd_conv(Ht_blk[i], dtY_f)
        else:
            v = blk_fd_conv(Ht_blk, dtY_f)
        indep_wo_beta[:, :, i] = np.real(im_comp(blk_ifft2(v), SRratio, SRratio))

    # ── Initialise parameters via SAR ──
    alpha, beta, gamma_val, W_mat = _tv_ini_params(
        x, Y, indep_wo_beta, indep_wo_gamma, nbands,
        gamma_alpha, gamma_beta, gamma_gamma,
        alpha_mode, beta_mode, gamma_mode, epsW,
    )

    y0 = convertToMBVec(np.zeros((hr_h, hr_w, nbands)))
    conv_crit = eps_map + 1.0
    iteration = 0

    # ── Main loop ──
    while iteration <= itmin_map or (conv_crit > eps_map and iteration < itmax_map):
        iteration += 1
        if verbose:
            print(f"iter = {iteration}")

        # RHS
        indep_term = np.zeros((hr_h, hr_w, nbands))
        for i in range(nbands):
            indep_term[:, :, i] = (gamma_val * indep_wo_gamma[:, :, i] +
                                   beta[i] * indep_wo_beta[:, :, i])
        rhs = convertToMBVec(indep_term)

        # CG solve
        def matvec(v):
            return _tv_multiply_by_invcov(
                v, alpha, W_mat, beta, HtDtDH, gamma_val, lam,
                hr_h, hr_w, SRratio, nbands, psf_is_list,
            )

        from scipy.sparse.linalg import LinearOperator
        n_total = hr_h * hr_w * nbands
        A_op = LinearOperator((n_total, n_total), matvec=matvec, dtype=np.float64)
        y_vec, _ = scipy_cg(A_op, rhs, x0=y0, rtol=eps_y, maxiter=itmax_y)

        # Convergence
        if iteration > 1:
            denom = np.dot(y0, y0)
            if denom > 0:
                conv_crit = np.dot(y_vec - y0, y_vec - y0) / denom
            else:
                conv_crit = 0.0
            if verbose:
                print(f"  ||y-y0||^2/||y0||^2 = {conv_crit:.6e}")

        y0 = y_vec.copy()

        # Traces
        tralpha, trbeta, trgamma = _tv_calc_trazas(
            alpha, DhtDh, DvtDv, W_mat, beta, HtDtDH, gamma_val, lam,
            (lr_h, lr_w), SRratio, nbands, psf_is_list,
        )

        y_img = convertToMBImg(y_vec, hr_h, hr_w, nbands)

        # Update params
        alpha, beta, gamma_val, W_mat = _tv_update_params(
            x, Y, H_blk, y_img, lam,
            tralpha, trbeta, trgamma, SRratio, nbands,
            gamma_alpha, gamma_beta, gamma_gamma,
            alpha_mode, beta_mode, gamma_mode, epsW,
            psf_is_list,
        )

        if verbose:
            print(f"  alpha = {alpha}, beta = {beta}, gamma = {gamma_val}")

    y_img = convertToMBImg(y_vec, hr_h, hr_w, nbands)
    return y_img, alpha, beta, gamma_val, W_mat


# ── TV helper: initial parameters via SAR ────────────────────────────────
def _tv_ini_params(
    x, Y, indep_wo_beta, indep_wo_gamma, nbands,
    gamma_alpha, gamma_beta, gamma_gamma,
    alpha_mode, beta_mode, gamma_mode, epsW,
):
    M, N = x.shape[:2]
    _, alpha_SAR, gamma_SAR = restoreSAR(x, np.array([[1.0]]))
    alpha_init = alfaTVpvini(np.real(ifft2(fft2(x))), 2)  # on PAN

    E_alpha = np.zeros(nbands)
    E_beta = np.zeros(nbands)

    for i in range(nbands):
        _, _, beta_i = restoreSAR(Y[:, :, i], np.array([[1.0]]))
        E_alpha[i] = gamma_alpha[i] / alpha_mode[i] + (1.0 - gamma_alpha[i]) / alpha_init
        E_beta[i] = gamma_beta[i] / beta_mode[i] + (1.0 - gamma_beta[i]) / beta_i
        E_alpha[i] = 1.0 / E_alpha[i]
        E_beta[i] = 1.0 / E_beta[i]

    E_gamma = gamma_gamma / gamma_mode + (1.0 - gamma_gamma) / gamma_SAR
    E_gamma = 1.0 / E_gamma

    # Initial weights
    indep_term = np.zeros_like(indep_wo_gamma)
    for i in range(nbands):
        indep_term[:, :, i] = E_gamma * indep_wo_gamma[:, :, i] + E_beta[i] * indep_wo_beta[:, :, i]

    W = np.zeros((M, N, nbands))
    for i in range(nbands):
        Dhy, Dvy = circ_gradient2(indep_term[:, :, i])
        v = Dhy ** 2 + Dvy ** 2
        W[:, :, i] = compute_Wpvb(v, 2, epsW)

    return E_alpha, E_beta, E_gamma, W


# ── TV helper: update hyperparameters ────────────────────────────────────
def _tv_update_params(
    x, Y, H_blk, y, lam,
    tralpha, trbeta, trgamma, bkn, nbands,
    gamma_alpha, gamma_beta, gamma_gamma,
    alpha_mode, beta_mode, gamma_mode, epsW,
    psf_is_list,
):
    M, N = y.shape[:2]
    ysupport = M * N
    observationsupport = ysupport / bkn / bkn

    aux4 = x.copy()
    W = np.zeros((M, N, nbands))
    E_alpha = np.zeros(nbands)
    E_beta = np.zeros(nbands)

    for i in range(nbands):
        # TV prior weights
        Dhy, Dvy = circ_gradient2(y[:, :, i])
        v = Dhy ** 2 + Dvy ** 2 + tralpha[i]
        v[v < 0] = 0.0
        W[:, :, i] = compute_Wpvb(v, 2, epsW)

        sum_t = 4.0 * v ** 0.5
        sum_p = 2.0 * ysupport
        normprior_i = np.sum(sum_t)

        # Observation error
        YB = blk_fft2(im_decomp(y[:, :, i], bkn, bkn))
        if psf_is_list:
            recon = np.real(im_comp(blk_ifft2(blk_fd_conv(H_blk[i], YB)), bkn, bkn))
        else:
            recon = np.real(im_comp(blk_ifft2(blk_fd_conv(H_blk, YB)), bkn, bkn))
        recon_lr = recon[::bkn, ::bkn]
        err = np.sum((Y[:, :, i] - recon_lr) ** 2) + _EPS

        E_alpha[i] = gamma_alpha[i] / alpha_mode[i] + (1.0 - gamma_alpha[i]) * normprior_i / sum_p
        E_beta[i] = gamma_beta[i] / beta_mode[i] + (1.0 - gamma_beta[i]) * (err + trbeta[i]) / observationsupport
        E_alpha[i] = max(1.0 / E_alpha[i], _EPS)
        E_beta[i] = max(1.0 / E_beta[i], _EPS)

        aux4 = aux4 - lam[i] * y[:, :, i]

    norma_gam = np.sum(aux4 ** 2)
    inv_Eg = gamma_gamma / gamma_mode + (1.0 - gamma_gamma) * (norma_gam + trgamma) / ysupport
    E_gamma = max(1.0 / inv_Eg, _EPS)

    return E_alpha, E_beta, E_gamma, W


# ── TV helper: matrix-vector product ─────────────────────────────────────
def _tv_multiply_by_invcov(
    y_vec, alpha, W, beta, HtDtDH, gamma_val, lam,
    nr, nc, bkn, nbands, psf_is_list,
):
    yd = convertToMBImg(y_vec, nr, nc, nbands)
    Ay = np.zeros((nr, nc, nbands))

    for i in range(nbands):
        yf = blk_fft2(im_decomp(yd[:, :, i], bkn, bkn))
        if psf_is_list:
            v = beta[i] * im_comp(blk_ifft2(blk_fd_conv(HtDtDH[i], yf)), bkn, bkn)
        else:
            v = beta[i] * im_comp(blk_ifft2(blk_fd_conv(HtDtDH, yf)), bkn, bkn)

        Dhy, Dvy = circ_gradient2(yd[:, :, i])
        F2, _ = Tcirc_gradient2(W[:, :, i] * Dhy)
        _, F3 = Tcirc_gradient2(W[:, :, i] * Dvy)
        Ay[:, :, i] = v + alpha[i] * F2 + alpha[i] * F3

    for i in range(nbands):
        for j in range(nbands):
            Ay[:, :, i] += gamma_val * lam[i] * lam[j] * yd[:, :, j]

    return convertToMBVec(Ay)


# ── TV helper: trace computation ─────────────────────────────────────────
def _tv_calc_trazas(
    alpha, DhtDh, DvtDv, W, beta, HtDtDH, gamma_val, lam,
    lr_size, SRratio, nbands, psf_is_list,
):
    Qinv = _tv_calc_cov_inv(
        alpha, DhtDh, DvtDv, W, beta, HtDtDH, gamma_val, lam,
        lr_size, SRratio, nbands, psf_is_list,
    )

    M, N = W.shape[:2]
    trpancr = 0.0
    trprior = np.zeros(nbands)
    trobs = np.zeros(nbands)
    DhDv_sum = DhtDh + DvtDv

    for i in range(nbands):
        trprior[i] = blk_fd_trace(blk_fd_conv(Qinv[i][i], DhDv_sum)) / (M * N)
        if psf_is_list:
            trobs[i] = blk_fd_trace(blk_fd_conv(Qinv[i][i], HtDtDH[i]))
        else:
            trobs[i] = blk_fd_trace(blk_fd_conv(Qinv[i][i], HtDtDH))
        for j in range(nbands):
            trpancr += lam[i] * lam[j] * blk_fd_trace(Qinv[i][j])

    return trprior, trobs, trpancr


def _tv_calc_cov_inv(
    alpha, DhtDh, DvtDv, W, beta, HtDtDH, gamma_val, lam,
    lr_size, SRratio, nbands, psf_is_list,
):
    low_nr, low_nc = lr_size
    bkn2 = SRratio * SRratio

    I_blk = np.zeros((low_nr, low_nc, bkn2, bkn2))
    for k in range(bkn2):
        I_blk[:, :, k, k] = 1.0

    Q = [[None] * nbands for _ in range(nbands)]
    DhDv = DhtDh + DvtDv

    for i in range(nbands):
        z = float(np.mean(W[:, :, i]))
        if psf_is_list:
            diag = gamma_val * lam[i] * lam[i] * I_blk + alpha[i] * z * DhDv + beta[i] * HtDtDH[i]
        else:
            diag = gamma_val * lam[i] * lam[i] * I_blk + alpha[i] * z * DhDv + beta[i] * HtDtDH
        Q[i][i] = diag.copy()

        for j in range(i + 1, nbands):
            off = gamma_val * lam[i] * lam[j] * I_blk
            Q[i][j] = off.copy()
            Q[j][i] = off.copy()

    # Small diagonal regularisation to prevent singular blocks
    reg = _EPS * I_blk

    # Invert per frequency point
    if nbands == 1:
        Qinv = [[np.zeros_like(Q[0][0])]]
        for fi in range(low_nr):
            for fj in range(low_nc):
                Qinv[0][0][fi, fj, :, :] = np.linalg.inv(Q[0][0][fi, fj, :, :] + reg[fi, fj, :, :])
        return Qinv

    Qinv = [[np.zeros((low_nr, low_nc, bkn2, bkn2)) for _ in range(nbands)]
            for _ in range(nbands)]

    full_size = nbands * bkn2
    full_reg = _EPS * np.eye(full_size)
    for fi in range(low_nr):
        for fj in range(low_nc):
            full_mat = np.zeros((full_size, full_size), dtype=complex)
            for bi in range(nbands):
                for bj in range(nbands):
                    if Q[bi][bj] is not None:
                        r0, r1 = bi * bkn2, (bi + 1) * bkn2
                        c0, c1 = bj * bkn2, (bj + 1) * bkn2
                        full_mat[r0:r1, c0:c1] = Q[bi][bj][fi, fj, :, :]
            full_inv = np.linalg.inv(full_mat + full_reg)
            for bi in range(nbands):
                for bj in range(nbands):
                    r0, r1 = bi * bkn2, (bi + 1) * bkn2
                    c0, c1 = bj * bkn2, (bj + 1) * bkn2
                    Qinv[bi][bj][fi, fj, :, :] = full_inv[r0:r1, c0:c1]

    return Qinv
