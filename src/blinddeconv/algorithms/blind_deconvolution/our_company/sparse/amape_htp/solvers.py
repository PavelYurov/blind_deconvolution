"""
solvers.py

Core solver functions for AMAPE-HTP blind deconvolution.

Ported from C++ code by Suzuki Hironobu (Blind-Deblurring-master).

Reference:
    J. Kotera, F. Sroubek, P. Milanfar:
    "Blind deconvolution using alternating maximum a posteriori estimation
    with heavy-tailed priors", DOI: 10.1007/978-3-642-40246-3_8

C++ -> Python notes:
    - FFTW3 forward: no normalization; backward: 1/(rows*cols).
      numpy fft2/ifft2 match this exactly.
    - C++ Ustep: idft2d(FU, U) where U is MatrixXd -> takes real part only.
      Python: U = np.real(ifft2(FU)).
    - C++ fftCGSRaL: idft2d(FU, U) where U is MatrixXcd -> complex output.
      Python: U = ifft2(FU).
    - C++ Hstep denominator: tmp1.real() = FUTU.real() + beta/gamma.
      Only modifies real part; imag stays 0 (from copy_mat_2_cmat_zeros).
      In Python we replicate by building denom with zero imag part.
    - All copy_mat_2_*_zeros have off-by-one: copy (rows-1) x (cols-1).
      Replicated exactly via utils functions.

Contains:
    Ustep              — image estimation sub-problem (Sec. 3.1)
    Hstep              — kernel estimation sub-problem (Sec. 3.2)
    PSFestimaLnoRgrad  — single-scale blind deconvolution (Sec. 3)
    fftCGSRaL          — non-blind deconvolution with known PSF
"""

import numpy as np
from numpy.fft import fft2, ifft2

from .utils import (
    copy_mat_2_mat_zeros,
    copy_mat_2_cmat_zeros,
    copy_mat_2_cmat,
    copy_cmat_2_cmat_zeros,
    sumabs2,
    asetLnorm,
    aLn,
    set_Vh,
    centerPSF,
    set_vrange,
    uConstr,
    edgetaper,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Ustep  (from blind_deblur.cpp lines 881-990)
# Image estimation via ALM — Sec. 3.1 of paper
# ═══════════════════════════════════════════════════════════════════════════════

def Ustep(q, H, U, FeGu, FU, FUx, FUy, FDx, FDy, Vx, Vy, Bx, By, DTD,
          param, gamma):
    """
    U-estimation sub-problem.

    Alternating minimization of the augmented Lagrangian w.r.t. u
    (latent image) with Lp prior on gradients.

    Corresponds to Algorithm in Sec. 3.1, lines 1-9.

    Parameters
    ----------
    q       : Lp exponent (param.Lp)
    H       : current PSF estimate (hsize_r x hsize_c)
    U       : current image estimate (usize_r x usize_c), real
    FeGu    : fft2(edgetapered blurred image)
    FU      : fft2(U)
    FUx, FUy : fft2 of x,y gradients of U
    FDx, FDy : OTF of derivative filters [1,-1] and [1;-1]
    Vx, Vy  : auxiliary variables (real)
    Bx, By  : Bregman variables (real)
    DTD     : conj(FDx)*FDx + conj(FDy)*FDy
    param   : dict with 'beta_u', 'alpha_u', 'ccreltol', 'maxiter_u'
    gamma   : current gamma value

    Returns
    -------
    U, FU, FUx, FUy, Vx, Vy, Bx, By   (all updated)
    """
    beta = param['beta_u']
    alpha = param['alpha_u']
    ccreltol = param['ccreltol']
    maxiter_u = param['maxiter_u']

    usize = (FU.shape[0], FU.shape[1])

    # ── Pad H to image size and compute FHS ──────────────────────────
    # C++: copy_mat_2_mat_zeros(H, tmpr1, H.rows(), H.cols());
    #      dft2d(tmpr1, FHS);
    tmpr1 = copy_mat_2_mat_zeros(H, usize, H.shape[0], H.shape[1])
    FHS = fft2(tmpr1)

    # FHTH = conj(FHS) .* FHS
    FHTH = np.conj(FHS) * FHS
    # FGs = conj(FHS) .* FeGu
    FGs = np.conj(FHS) * FeGu

    # ── Inner loop (coordinate descent) ──────────────────────────────
    for i in range(maxiter_u):
        FUp = FU.copy()

        # b = FGs + beta/gamma*(conj(FDx).*fft2(Vx+Bx)
        #                     + conj(FDy).*fft2(Vy+By))
        tmp1 = fft2(Vx + Bx)
        tmp2 = fft2(Vy + By)
        b = np.conj(FDx) * tmp1 + np.conj(FDy) * tmp2
        b = (beta / gamma) * b + FGs

        # FU = b ./ (FHTH + beta/gamma * DTD)
        denom = (beta / gamma) * DTD + FHTH
        FU = b / denom

        # FUx = FDx .* FU;  FUy = FDy .* FU
        FUx = FDx * FU
        FUy = FDy * FU

        # xD = real(ifft2(FUx));  yD = real(ifft2(FUy))
        xD = np.real(ifft2(FUx))
        yD = np.real(ifft2(FUy))

        # xDm = xD - Bx;  yDm = yD - By
        xDm = xD - Bx
        yDm = yD - By

        # nDm = sqrt(xDm.^2 + yDm.^2)
        nDm = np.sqrt(xDm ** 2 + yDm ** 2)

        # Lp proximal operator (thresholding)
        v_star, u_star = asetLnorm(q, alpha, beta)
        Vx = aLn(xDm, nDm, v_star, u_star)
        Vy = aLn(yDm, nDm, v_star, u_star)

        # Bregman update
        Bx = Bx + (Vx - xD)
        By = By + (Vy - yD)

        # Convergence check
        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FUp - FU)) / np.sqrt(sumabs2(FU))
        if relcon < ccreltol:
            break

    # U = real(ifft2(FU))
    U = np.real(ifft2(FU))

    return U, FU, FUx, FUy, Vx, Vy, Bx, By


# ═══════════════════════════════════════════════════════════════════════════════
# Hstep  (from blind_deblur.cpp lines 995-1093)
# Kernel estimation via ALM — Sec. 3.2 of paper
# ═══════════════════════════════════════════════════════════════════════════════

def Hstep(q, h, FeGx, FeGy, FUx, FUy, Vh, Bh,
          param, gamma, hsize_r, hsize_c, usize):
    """
    H-estimation sub-problem.

    Alternating minimization of the augmented Lagrangian w.r.t. h
    (blur kernel) with Laplace prior (q=1) and positivity constraint.

    Corresponds to Algorithm in Sec. 3.2, lines 1-8.

    Parameters
    ----------
    q           : Lp exponent for kernel prior (always 1.0)
    h           : current PSF estimate (hsize_r x hsize_c)
    FeGx, FeGy : fft2 of x,y gradients of blurred image
    FUx, FUy   : fft2 of x,y gradients of U (from Ustep)
    Vh          : auxiliary variable (real, usize)
    Bh          : Bregman variable (real, usize)
    param       : dict with 'beta_h', 'alpha_h', 'ccreltol', 'maxiter_h'
    gamma       : current gamma value
    hsize_r, hsize_c : PSF size
    usize       : image size tuple (rows, cols)

    Returns
    -------
    h, Vh, Bh   (all updated)
    """
    beta = param['beta_h']
    alpha = param['alpha_h']
    ccreltol = param['ccreltol']
    maxiter_h = param['maxiter_h']

    # FUD = FeGx .* conj(FUx) + FeGy .* conj(FUy)
    FUD = FeGx * np.conj(FUx) + FeGy * np.conj(FUy)
    # FUTU = conj(FUx).*FUx + conj(FUy).*FUy
    FUTU = np.conj(FUx) * FUx + np.conj(FUy) * FUy

    # FH = fft2(H, usize)
    # C++: copy_mat_2_cmat_zeros(H, tmp1, H.rows(), H.cols());
    #      dft2d(tmp1, FH);
    tmp1 = copy_mat_2_cmat_zeros(h, usize, h.shape[0], h.shape[1])
    FH = fft2(tmp1)

    # Denominator: tmp1.real() = FUTU.real() + beta/gamma
    # C++ only modifies real part; imag stays at 0 (from copy_mat_2_cmat_zeros).
    # After dft2d(tmp1, FH), tmp1 still holds the padded H (imag=0 from
    # copy_mat_2_cmat_zeros). Then C++ overwrites only the real part.
    # FUTU is conj(z)*z so purely real. We build denom with imag=0.
    denom = np.zeros(usize, dtype=np.complex128)
    denom.real = np.real(FUTU) + (beta / gamma)

    # ── Inner loop ───────────────────────────────────────────────────
    for i in range(maxiter_h):
        FHp = FH.copy()

        # b = beta/gamma * fft2(Vh + Bh) + FUD
        b = fft2(Vh + Bh)
        b = (beta / gamma) * b + FUD

        # FH = b ./ denom
        FH = b / denom

        # Convergence check
        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FHp - FH)) / np.sqrt(sumabs2(FH))

        # hI = real(ifft2(FH))
        hI = np.real(ifft2(FH))

        # hIm = hI - Bh;  nIm = abs(hIm)
        hIm = hI - Bh
        nIm = np.abs(hIm)

        # Lp thresholding (q=1 for kernel -> soft thresholding)
        v_star, u_star = asetLnorm(q, alpha, beta)
        Vh = aLn(hIm, nIm, v_star, u_star)

        # Vh(Vh<0) = 0; zero outside kernel region
        Vh = set_Vh(Vh, hsize_r, hsize_c)

        # Bregman update
        Bh = Bh + (Vh - hI)

        # H = hI(0:hsize_r, 0:hsize_c)
        h = hI[:hsize_r, :hsize_c].copy()

        if relcon < ccreltol:
            break

    return h, Vh, Bh


# ═══════════════════════════════════════════════════════════════════════════════
# PSFestimaLnoRgrad  (from blind_deblur.cpp lines 730-875)
# Single-scale PSF estimation — Sec. 3, main loop
# ═══════════════════════════════════════════════════════════════════════════════

def PSFestimaLnoRgrad(h, ROI, param, L):
    """
    Single-scale blind PSF estimation.

    Alternates between Ustep (image estimation) and Hstep (kernel estimation)
    with increasing gamma, then centers the PSF.

    Parameters
    ----------
    h     : initial PSF estimate (hsize_r x hsize_c), real
    ROI   : region of interest (gsize_r x gsize_c), complex
            (converted from real image with off-by-one via copy_mat_2_cmat)
    param : dict with all algorithm parameters
    L     : current scale level (1-based)

    Returns
    -------
    h : updated PSF estimate
    """
    gamma = param['gamma']
    hsize_r = h.shape[0]
    hsize_c = h.shape[1]
    gsize_r = ROI.shape[0]
    gsize_c = ROI.shape[1]
    usize = (gsize_r, gsize_c)

    # ── Initialize all variables to zeros ────────────────────────────
    U = np.zeros((gsize_r, gsize_c), dtype=np.float64)
    FU = fft2(U)   # C++: dft2d(U, FU) where U is all zeros

    FUx = np.zeros(usize, dtype=np.complex128)
    FUy = np.zeros(usize, dtype=np.complex128)

    Vx = np.zeros(usize, dtype=np.float64)
    Vy = np.zeros(usize, dtype=np.float64)
    Vh = np.zeros(usize, dtype=np.float64)
    Bx = np.zeros(usize, dtype=np.float64)
    By = np.zeros(usize, dtype=np.float64)
    Bh = np.zeros(usize, dtype=np.float64)

    # ── FDx, FDy: OTF of derivative filters ─────────────────────────
    # C++: complex matrix, set (0,0)=1, (0,1)=-1, then dft2d
    FDx = np.zeros(usize, dtype=np.complex128)
    FDx[0, 0] = 1.0 + 0j
    FDx[0, 1] = -1.0 + 0j
    FDx = fft2(FDx)

    FDy = np.zeros(usize, dtype=np.complex128)
    FDy[0, 0] = 1.0 + 0j
    FDy[1, 0] = -1.0 + 0j
    FDy = fft2(FDy)

    # DTD = conj(FDx).*FDx + conj(FDy).*FDy
    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # ── Edgetaper blurred image ──────────────────────────────────────
    eG = edgetaper(ROI)
    FeGu = fft2(eG)

    # FeGx = FDx .* FeGu;  FeGy = FDy .* FeGu
    FeGx = FDx * FeGu
    FeGy = FDy * FeGu

    # ── Main alternating loop ────────────────────────────────────────
    maxiter = param['maxiter']
    for ml in range(maxiter):
        # ── U-estimation ─────────────────────────────────────────
        U, FU, FUx, FUy, Vx, Vy, Bx, By = Ustep(
            param['Lp'], h, U, FeGu, FU, FUx, FUy, FDx, FDy,
            Vx, Vy, Bx, By, DTD,
            param, gamma
        )

        # ── H-estimation (always Lp=1 for kernel) ───────────────
        h, Vh, Bh = Hstep(
            1.0, h, FeGx, FeGy, FUx, FUy, Vh, Bh,
            param, gamma, hsize_r, hsize_c, usize
        )

        # ── Increase gamma ───────────────────────────────────────
        gamma *= 1.5

    # ── Center PSF ───────────────────────────────────────────────────
    h = centerPSF(h, param['centering_threshold'])

    return h


# ═══════════════════════════════════════════════════════════════════════════════
# fftCGSRaL  (from blind_deblur.cpp lines 1116-1291)
# Non-blind deconvolution with known PSF
# ═══════════════════════════════════════════════════════════════════════════════

def fftCGSRaL(G, H, param):
    """
    Non-blind deconvolution via ALM with known PSF.

    Same U-step solver as in blind estimation, but with:
    - Known (fixed) PSF with center-shift via hshift
    - Edgetaper preprocessing
    - Value range clamping (uConstr)
    - Different gamma/beta/Lp (nonblind parameters)

    Parameters
    ----------
    G     : blurred image (gsize_r x gsize_c), real float64
    H     : estimated PSF (hsize_r x hsize_c), real float64
    param : dict with 'maxiter_u', 'alpha_u', 'ccreltol',
            'gamma_nonblind', 'beta_u_nonblind', 'Lp_nonblind'

    Returns
    -------
    U : restored image (gsize_r x gsize_c), complex
        (real part is the restored image, clamped to original range)
    """
    maxiter = param['maxiter_u']
    alpha = param['alpha_u']
    ccreltol = param['ccreltol']
    gamma = param['gamma_nonblind']
    beta = param['beta_u_nonblind']
    Lp = param['Lp_nonblind']

    gsize_r = G.shape[0]
    gsize_c = G.shape[1]
    hsize_r = H.shape[0]
    hsize_c = H.shape[1]
    gsize = (gsize_r, gsize_c)

    # ── Value range for clamping ─────────────────────────────────────
    vrange_min, vrange_max = set_vrange(G)

    # ── hshift: delta at center of kernel ────────────────────────────
    # C++: cen_r = floor((H.rows()+1)/2);  cen_c = floor((H.cols()+1)/2);
    # In C++, (int+1)/int is integer division, then floor().
    # Python: (int+1)//2 gives same result.
    cen_r = (hsize_r + 1) // 2
    cen_c = (hsize_c + 1) // 2
    hshift = np.zeros((hsize_r, hsize_c), dtype=np.complex128)
    hshift[cen_r, cen_c] = 1.0 + 0j

    # ── FDx, FDy: OTF of derivative filters ─────────────────────────
    FDx = np.zeros(gsize, dtype=np.complex128)
    FDx[0, 0] = 1.0 + 0j
    FDx[0, 1] = -1.0 + 0j
    FDx = fft2(FDx)

    FDy = np.zeros(gsize, dtype=np.complex128)
    FDy[0, 0] = 1.0 + 0j
    FDy[1, 0] = -1.0 + 0j
    FDy = fft2(FDy)

    # ── FH = conj(fft2(hshift, gsize)) .* fft2(H, gsize) ───────────
    # C++: copy_cmat_2_cmat_zeros(hshift, tmp1, hshift.rows(), hshift.cols());
    #      copy_mat_2_cmat_zeros(H, tmp2, hshift.rows(), hshift.cols());
    #      dft2d(tmp1, tmp1); dft2d(tmp2, tmp2);
    #      FH = conj(tmp1) .* tmp2;
    tmp1 = copy_cmat_2_cmat_zeros(hshift, gsize, hsize_r, hsize_c)
    tmp2 = copy_mat_2_cmat_zeros(H, gsize, hsize_r, hsize_c)
    tmp1 = fft2(tmp1)
    tmp2 = fft2(tmp2)
    FH = np.conj(tmp1) * tmp2
    FHTH = np.conj(FH) * FH

    # ── Edgetaper on G ───────────────────────────────────────────────
    # C++: copy_mat_2_cmat(G, cG, G.rows(), G.cols());
    #      edgetaper(eG, cG);
    #      dft2d(eG, FGu);
    cG = copy_mat_2_cmat(G, gsize, gsize_r, gsize_c)
    eG = edgetaper(cG)
    FGu = fft2(eG)

    # FGs = conj(FH) .* FGu
    FGs = np.conj(FH) * FGu

    # DTD = conj(FDx).*FDx + conj(FDy).*FDy
    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # ── Initialize state ─────────────────────────────────────────────
    FU = np.zeros(gsize, dtype=np.complex128)
    Bx = np.zeros(gsize, dtype=np.float64)
    By = np.zeros(gsize, dtype=np.float64)
    Vx = np.zeros(gsize, dtype=np.float64)
    Vy = np.zeros(gsize, dtype=np.float64)

    # ── U-estimation loop ────────────────────────────────────────────
    for i in range(maxiter):
        FUp = FU.copy()

        # b = FGs + beta/gamma*(conj(FDx).*fft2(Vx+Bx)
        #                     + conj(FDy).*fft2(Vy+By))
        tmp1 = fft2(Vx + Bx)
        tmp2 = fft2(Vy + By)
        b = np.conj(FDx) * tmp1 + np.conj(FDy) * tmp2
        b = (beta / gamma) * b + FGs

        # FU = b ./ (FHTH + beta/gamma * DTD)
        denom = (beta / gamma) * DTD + FHTH
        FU = b / denom

        # xD = real(ifft2(FDx .* FU))
        # yD = real(ifft2(FDy .* FU))
        xD = np.real(ifft2(FDx * FU))
        yD = np.real(ifft2(FDy * FU))

        xDm = xD - Bx
        yDm = yD - By

        # nDm = sqrt(xDm.^2 + yDm.^2)
        nDm = np.sqrt(xDm ** 2 + yDm ** 2)

        # Lp proximal operator
        v_star, u_star = asetLnorm(Lp, alpha, beta)
        Vx = aLn(xDm, nDm, v_star, u_star)
        Vy = aLn(yDm, nDm, v_star, u_star)

        # Bregman update
        Bx = Bx + (Vx - xD)
        By = By + (Vy - yD)

        # Convergence check
        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FUp - FU)) / np.sqrt(sumabs2(FU))
        if relcon < ccreltol:
            break

    # ── U = ifft2(FU) — COMPLEX output (C++: idft2d to MatrixXcd) ───
    U = ifft2(FU)

    # ── Clamp to original value range ────────────────────────────────
    U = uConstr(U, vrange_min, vrange_max)

    return U
