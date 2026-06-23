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

def Ustep(q, H, U, FeGu, FU, FUx, FUy, FDx, FDy, Vx, Vy, Bx, By, DTD,
          param, gamma):

    beta = param['beta_u']
    alpha = param['alpha_u']
    ccreltol = param['ccreltol']
    maxiter_u = param['maxiter_u']

    usize = (FU.shape[0], FU.shape[1])

    tmpr1 = copy_mat_2_mat_zeros(H, usize, H.shape[0], H.shape[1])
    FHS = fft2(tmpr1)

    FHTH = np.conj(FHS) * FHS

    FGs = np.conj(FHS) * FeGu

    for i in range(maxiter_u):
        FUp = FU.copy()

        tmp1 = fft2(Vx + Bx)
        tmp2 = fft2(Vy + By)
        b = np.conj(FDx) * tmp1 + np.conj(FDy) * tmp2
        b = (beta / gamma) * b + FGs

        denom = (beta / gamma) * DTD + FHTH
        FU = b / denom

        FUx = FDx * FU
        FUy = FDy * FU

        xD = np.real(ifft2(FUx))
        yD = np.real(ifft2(FUy))

        xDm = xD - Bx
        yDm = yD - By

        nDm = np.sqrt(xDm ** 2 + yDm ** 2)

        v_star, u_star = asetLnorm(q, alpha, beta)
        Vx = aLn(xDm, nDm, v_star, u_star)
        Vy = aLn(yDm, nDm, v_star, u_star)

        Bx = Bx + (Vx - xD)
        By = By + (Vy - yD)

        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FUp - FU)) / np.sqrt(sumabs2(FU))
        if relcon < ccreltol:
            break

    U = np.real(ifft2(FU))

    return U, FU, FUx, FUy, Vx, Vy, Bx, By

def Hstep(q, h, FeGx, FeGy, FUx, FUy, Vh, Bh,
          param, gamma, hsize_r, hsize_c, usize):

    beta = param['beta_h']
    alpha = param['alpha_h']
    ccreltol = param['ccreltol']
    maxiter_h = param['maxiter_h']

    FUD = FeGx * np.conj(FUx) + FeGy * np.conj(FUy)

    FUTU = np.conj(FUx) * FUx + np.conj(FUy) * FUy

    tmp1 = copy_mat_2_cmat_zeros(h, usize, h.shape[0], h.shape[1])
    FH = fft2(tmp1)

    denom = np.zeros(usize, dtype=np.complex128)
    denom.real = np.real(FUTU) + (beta / gamma)

    for i in range(maxiter_h):
        FHp = FH.copy()

        b = fft2(Vh + Bh)
        b = (beta / gamma) * b + FUD

        FH = b / denom

        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FHp - FH)) / np.sqrt(sumabs2(FH))

        hI = np.real(ifft2(FH))

        hIm = hI - Bh
        nIm = np.abs(hIm)

        v_star, u_star = asetLnorm(q, alpha, beta)
        Vh = aLn(hIm, nIm, v_star, u_star)

        Vh = set_Vh(Vh, hsize_r, hsize_c)

        Bh = Bh + (Vh - hI)

        h = hI[:hsize_r, :hsize_c].copy()

        if relcon < ccreltol:
            break

    return h, Vh, Bh

def PSFestimaLnoRgrad(h, ROI, param, L):

    gamma = param['gamma']
    hsize_r = h.shape[0]
    hsize_c = h.shape[1]
    gsize_r = ROI.shape[0]
    gsize_c = ROI.shape[1]
    usize = (gsize_r, gsize_c)

    U = np.zeros((gsize_r, gsize_c), dtype=np.float64)
    FU = fft2(U)

    FUx = np.zeros(usize, dtype=np.complex128)
    FUy = np.zeros(usize, dtype=np.complex128)

    Vx = np.zeros(usize, dtype=np.float64)
    Vy = np.zeros(usize, dtype=np.float64)
    Vh = np.zeros(usize, dtype=np.float64)
    Bx = np.zeros(usize, dtype=np.float64)
    By = np.zeros(usize, dtype=np.float64)
    Bh = np.zeros(usize, dtype=np.float64)

    FDx = np.zeros(usize, dtype=np.complex128)
    FDx[0, 0] = 1.0 + 0j
    FDx[0, 1] = -1.0 + 0j
    FDx = fft2(FDx)

    FDy = np.zeros(usize, dtype=np.complex128)
    FDy[0, 0] = 1.0 + 0j
    FDy[1, 0] = -1.0 + 0j
    FDy = fft2(FDy)

    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    eG = edgetaper(ROI)
    FeGu = fft2(eG)

    FeGx = FDx * FeGu
    FeGy = FDy * FeGu

    maxiter = param['maxiter']
    for ml in range(maxiter):

        U, FU, FUx, FUy, Vx, Vy, Bx, By = Ustep(
            param['Lp'], h, U, FeGu, FU, FUx, FUy, FDx, FDy,
            Vx, Vy, Bx, By, DTD,
            param, gamma
        )

        h, Vh, Bh = Hstep(
            1.0, h, FeGx, FeGy, FUx, FUy, Vh, Bh,
            param, gamma, hsize_r, hsize_c, usize
        )

        gamma *= 1.5

    h = centerPSF(h, param['centering_threshold'])

    return h

def fftCGSRaL(G, H, param):

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

    vrange_min, vrange_max = set_vrange(G)

    cen_r = (hsize_r + 1) // 2
    cen_c = (hsize_c + 1) // 2
    hshift = np.zeros((hsize_r, hsize_c), dtype=np.complex128)
    hshift[cen_r, cen_c] = 1.0 + 0j

    FDx = np.zeros(gsize, dtype=np.complex128)
    FDx[0, 0] = 1.0 + 0j
    FDx[0, 1] = -1.0 + 0j
    FDx = fft2(FDx)

    FDy = np.zeros(gsize, dtype=np.complex128)
    FDy[0, 0] = 1.0 + 0j
    FDy[1, 0] = -1.0 + 0j
    FDy = fft2(FDy)

    tmp1 = copy_cmat_2_cmat_zeros(hshift, gsize, hsize_r, hsize_c)
    tmp2 = copy_mat_2_cmat_zeros(H, gsize, hsize_r, hsize_c)
    tmp1 = fft2(tmp1)
    tmp2 = fft2(tmp2)
    FH = np.conj(tmp1) * tmp2
    FHTH = np.conj(FH) * FH

    cG = copy_mat_2_cmat(G, gsize, gsize_r, gsize_c)
    eG = edgetaper(cG)
    FGu = fft2(eG)

    FGs = np.conj(FH) * FGu

    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    FU = np.zeros(gsize, dtype=np.complex128)
    Bx = np.zeros(gsize, dtype=np.float64)
    By = np.zeros(gsize, dtype=np.float64)
    Vx = np.zeros(gsize, dtype=np.float64)
    Vy = np.zeros(gsize, dtype=np.float64)

    for i in range(maxiter):
        FUp = FU.copy()

        tmp1 = fft2(Vx + Bx)
        tmp2 = fft2(Vy + By)
        b = np.conj(FDx) * tmp1 + np.conj(FDy) * tmp2
        b = (beta / gamma) * b + FGs

        denom = (beta / gamma) * DTD + FHTH
        FU = b / denom

        xD = np.real(ifft2(FDx * FU))
        yD = np.real(ifft2(FDy * FU))

        xDm = xD - Bx
        yDm = yD - By

        nDm = np.sqrt(xDm ** 2 + yDm ** 2)

        v_star, u_star = asetLnorm(Lp, alpha, beta)
        Vx = aLn(xDm, nDm, v_star, u_star)
        Vy = aLn(yDm, nDm, v_star, u_star)

        Bx = Bx + (Vx - xD)
        By = By + (Vy - yD)

        with np.errstate(divide='ignore', invalid='ignore'):
            relcon = np.sqrt(sumabs2(FUp - FU)) / np.sqrt(sumabs2(FU))
        if relcon < ccreltol:
            break

    U = ifft2(FU)

    U = uConstr(U, vrange_min, vrange_max)

    return U
