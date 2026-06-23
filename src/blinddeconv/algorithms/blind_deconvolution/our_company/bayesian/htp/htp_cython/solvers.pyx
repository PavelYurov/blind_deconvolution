"""
solvers.pyx

Функции-решатели алгоритма слепой деконволюции HTP.

Основные функции:
    psf_estim_lno_rgrad - Чередующаяся оценка скрытого изображения и ядра 
                          на одном масштабе.
    fft_cg_sr_al        - Быстрая неслепая деконволюция в частотной области.
    mc_restoration      - Конвейер обработки изображения по пирамиде масштабов.

Литература:
[1] J. Kotera, F. Sroubek, P. Milanfar,
    "Blind Deconvolution Using Alternating Maximum a Posteriori
     Estimation with Heavy-tailed Priors", CAIP 2013.
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
    imresize,
    edgetaper,
)
from .denoisers import apply_denoiser

import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- Чередующаяся оценка ядра и изображения ---
def psf_estim_lno_rgrad(
    G: np.ndarray,
    iH: np.ndarray,
    PAR: Dict,
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Совместная оценка скрытого изображения U и ядра H на одном масштабе 
    через полуквадратичное расщепление и итерации Брегмана (Уравнение 4 статьи).
    
    Для повышения стабильности шаг оценки H выполняется в пространстве градиентов.
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
    
    # Итеративное жесткое пороговое отсечение ядра. При > 0 увеличивает разреженность ФРТ.
    kernel_thresh = float(PAR.get('kernel_thresh', 0.0))
    # Итеративное центрирование ядра предотвращает накопление ошибки смещения
    # для асимметричных ФРТ в процессе оптимизации.
    iterative_recenter = bool(PAR.get('iterative_recenter', True))
    verbose = int(PAR.get('verbose', 0))

    # Хук промежуточного шумоподавления скрытого изображения.
    pre_kernel = PAR.get('pre_kernel', None)
    pre_kernel_params = PAR.get('pre_kernel_params', None) or {}

    iteration_callback = PAR.get('iteration_callback', None)
    cb_level = int(PAR.get('_cb_level', 0))
    cb_num_levels = int(PAR.get('_cb_num_levels', 1))

    iH = np.asarray(iH, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    hsize = iH.shape[:2]
    gsize = G.shape[:2]
    usize = gsize  
    M, N = usize

    do_mse = Hstar is not None and np.asarray(Hstar).size > 0
    if do_mse:
        Report['hstep']['mse'] = np.zeros(maxiter + 1, dtype=np.float64)

    U = np.zeros(usize, dtype=np.float64)
    H = iH.copy()

    # Фурье-образы операторов производных
    FDx = fft2_pad(np.array([[1.0, -1.0]]), M, N)
    FDy = fft2_pad(np.array([[1.0], [-1.0]]), M, N)
    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # Вспомогательные переменные
    Vx = np.zeros(usize, dtype=np.float64)
    Vy = np.zeros(usize, dtype=np.float64)
    Vh = np.zeros(usize, dtype=np.float64)
    Bx = np.zeros(usize, dtype=np.float64)
    By = np.zeros(usize, dtype=np.float64)
    Bh = np.zeros(usize, dtype=np.float64)

    if do_mse:
        Report['hstep']['mse'][0] = calculate_mse(H, np.asarray(Hstar))

    # Сглаживание краев для устранения краевых эффектов
    eG = edgetaper(G, np.ones(hsize, dtype=np.float64) / np.prod(hsize))
    FeGu = np.fft.fft2(eG)
    FeGx = FDx * FeGu
    FeGy = FDy * FeGu

    state = {'FU': np.fft.fft2(U), 'FUx': np.zeros(usize, dtype=complex),
             'FUy': np.zeros(usize, dtype=complex)}

    def ustep(gamma_local: float):
        """Внутренний шаг оценки скрытого изображения U."""
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
            relcon = 0.0 if denom == 0 else np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom

            if relcon < ccreltol:
                break

        state['FU'] = FU
        state['FUx'] = FDx * FU
        state['FUy'] = FDy * FU

    def hstep(gamma_local: float) -> np.ndarray:
        """Внутренний шаг оценки ядра H."""
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
            relcon = 0.0 if denom == 0 else np.sqrt(np.sum(np.abs(FHp - FH) ** 2)) / denom

            hI = np.real(np.fft.ifft2(FH))
            hIm = hI - Bh
            nIm = np.abs(hIm)
            Vh = prior_fh(hIm, nIm)
            
            # Ограничение ФРТ: обнуление отрицательных значений
            Vh[Vh < 0] = 0.0
            Vh[hsize[0]:, :] = 0.0
            Vh[:hsize[0], hsize[1]:] = 0.0
            
            Bh = Bh + Vh - hI
            H_local = hI[:hsize[0], :hsize[1]]

            if relcon < ccreltol:
                break

        return H_local

    # --- Внешний чередующийся цикл оптимизации ---
    for mI in range(1, maxiter + 1):
        ustep(gamma)

        # Применение промежуточного шумоподавления (Хук 2)
        if pre_kernel is not None and pre_kernel != 'none':
            U_curr = np.real(np.fft.ifft2(state['FU']))
            U_dn = apply_denoiser(U_curr, pre_kernel, **pre_kernel_params)
            FU_dn = np.fft.fft2(U_dn)
            state['FU'] = FU_dn
            state['FUx'] = FDx * FU_dn
            state['FUy'] = FDy * FU_dn

        H = hstep(gamma)

        # Опциональное жесткое пороговое отсечение
        if kernel_thresh > 0.0:
            H_pos = np.maximum(H, 0.0)
            mx = H_pos.max()
            if mx > 0:
                H = np.where(H_pos < kernel_thresh * mx, 0.0, H_pos)
                s = H.sum()
                if s > 0:
                    H = H / s

        # Итеративное центрирование ядра
        if iterative_recenter and centering_threshold > 0 and mI < maxiter:
            H = center_psf(H, centering_threshold)

        if do_mse:
            Report['hstep']['mse'][mI] = calculate_mse(H, np.asarray(Hstar))

        if iteration_callback is not None:
            try:
                U_snap = np.real(np.fft.ifft2(state['FU']))
            except Exception:
                U_snap = None
            iteration_callback({
                'iteration': mI - 1,        
                'scale': cb_level,           
                'num_scales': cb_num_levels,
                'kernel': H.copy(),
                'image': U_snap.copy() if U_snap is not None else None,
                'metrics': {
                    'gamma': float(gamma),
                },
            })

        # Релаксация параметра гамма
        gamma = gamma * 1.5

    if centering_threshold > 0:
        H = center_psf(H, centering_threshold)

    U = np.real(np.fft.ifft2(state['FU']))
    return H, U, Report


# --- Базовый метод неслепой деконволюции ---

def fft_cg_sr_al(G: np.ndarray, H: np.ndarray, PAR: Dict) -> np.ndarray:
    """
    Быстрая неслепая деконволюция на основе метода дополненного лагранжиана 
    с векторной регуляризацией градиентов. Выполняется полностью в Фурье-области.
    """
    G = np.asarray(G, dtype=np.float64)
    H_psf = np.asarray(H, dtype=np.float64)

    maxiter = int(PAR['maxiter_u'])
    alpha = float(PAR['alpha_u'])
    ccreltol = float(PAR['ccreltol'])
    gamma = float(PAR.get('gamma_nonblind', PAR['gamma']))
    beta = float(PAR.get('beta_u_nonblind', PAR['beta_u']))
    Lp = float(PAR.get('Lp_nonblind', PAR['Lp']))

    if G.ndim == 2:
        G = G[..., None]
        squeeze_out = True
    else:
        squeeze_out = False
    Hh, Ww, C = G.shape

    # Фиксация диапазона значений
    vrange = np.zeros((C, 2), dtype=np.float64)
    for c in range(C):
        ch = G[..., c]
        vrange[c, 0] = ch.min()
        vrange[c, 1] = ch.max()

    hshift = np.zeros_like(H_psf)
    hshift[H_psf.shape[0] // 2, H_psf.shape[1] // 2] = 1.0

    FDx_2d = fft2_pad(np.array([[1.0, -1.0]]), Hh, Ww)
    FDy_2d = fft2_pad(np.array([[1.0], [-1.0]]), Hh, Ww)
    FDx = np.repeat(FDx_2d[..., None], C, axis=2)
    FDy = np.repeat(FDy_2d[..., None], C, axis=2)

    FH_2d = (np.conj(fft2_pad(hshift, Hh, Ww))
             * fft2_pad(H_psf, Hh, Ww))
    FH = np.repeat(FH_2d[..., None], C, axis=2)
    FHTH = np.conj(FH) * FH

    eG = edgetaper(G if not squeeze_out else G[..., 0], H_psf)
    if eG.ndim == 2:
        eG = eG[..., None]
    FGu = np.fft.fft2(eG, axes=(0, 1))
    FGs = np.conj(FH) * FGu

    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    Bx = np.zeros((Hh, Ww, C), dtype=np.float64)
    By = np.zeros((Hh, Ww, C), dtype=np.float64)
    Vx = np.zeros((Hh, Ww, C), dtype=np.float64)
    Vy = np.zeros((Hh, Ww, C), dtype=np.float64)

    FU = np.zeros((Hh, Ww, C), dtype=complex)
    prior_fh = setup_lp_prior(Lp, alpha, beta)

    for i in range(1, maxiter + 1):
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
        
        # Агрегирование нормы по каналам для векторного Lp
        nDm_2d = np.sqrt(np.sum(xDm ** 2, axis=2) + np.sum(yDm ** 2, axis=2))
        nDm = np.repeat(nDm_2d[..., None], C, axis=2)

        Vy = prior_fh(yDm, nDm)
        Vx = prior_fh(xDm, nDm)

        Bx = Bx + Vx - xD
        By = By + Vy - yD

        denom = np.sqrt(np.sum(np.abs(FU) ** 2))
        relcon = 0.0 if denom == 0 else np.sqrt(np.sum(np.abs(FUp - FU) ** 2)) / denom
        if relcon < ccreltol:
            break

    U = np.real(np.fft.ifft2(FU, axes=(0, 1)))

    for c in range(C):
        lo, hi = vrange[c, 0], vrange[c, 1]
        ch = U[..., c]
        ch[ch < lo] = lo
        ch[ch > hi] = hi
        U[..., c] = ch

    if squeeze_out:
        U = U[..., 0]
    return U


# --- Альтернативные методы неслепой деконволюции ---

from blinddeconv.algorithms.mod_cython._build_pyd.non_blind import ringing_artifacts_removal as _ringing_artifacts_removal
from blinddeconv.algorithms.mod_cython._build_pyd.non_blind import firls_deconv as _firls_deconv


def nonblind_ringing_removal(G: np.ndarray, H: np.ndarray, PAR: Dict) -> np.ndarray:
    """Обертка для метода удаления краевых артефактов (ringing)."""
    lambda_tv = float(PAR.get('lambda_tv', 4e-3))
    lambda_l0 = float(PAR.get('lambda_l0', 2e-3))
    weight_ring = float(PAR.get('weight_ring', 0.5))
    if G.ndim == 3:
        out = np.empty_like(G, dtype=np.float64)
        for c in range(G.shape[2]):
            out[..., c] = _ringing_artifacts_removal(
                G[..., c].astype(np.float64), H,
                lambda_tv=lambda_tv, lambda_l0=lambda_l0,
                weight_ring=weight_ring,
            )
        return out
    return _ringing_artifacts_removal(
        G.astype(np.float64), H,
        lambda_tv=lambda_tv, lambda_l0=lambda_l0, weight_ring=weight_ring,
    )


def nonblind_firls(G: np.ndarray, H: np.ndarray, PAR: Dict) -> np.ndarray:
    """Обертка для деконволюции методом FIRLS-UBC."""
    fp = dict(PAR.get('firls_params', None) or {})
    fp.setdefault('clip', True)

    H_arr = np.ascontiguousarray(H, dtype=np.float64)
    pad_r = 1 if (H_arr.shape[0] % 2 == 0) else 0
    pad_c = 1 if (H_arr.shape[1] % 2 == 0) else 0
    if pad_r or pad_c:
        H_arr = np.pad(H_arr, ((0, pad_r), (0, pad_c)), mode='constant')
    s = H_arr.sum()
    if s > 0:
        H_arr = H_arr / s

    if G.ndim == 3:
        out = np.empty_like(G, dtype=np.float64)
        for c in range(G.shape[2]):
            out[..., c] = _firls_deconv(
                G[..., c].astype(np.float64), H_arr, **fp,
            )
        return out
    return _firls_deconv(G.astype(np.float64), H_arr, **fp)


# --- Конвейер алгоритма ---

def mc_restoration(
    G: np.ndarray,
    hsize: Tuple[int, int],
    PAR: Dict,
    MSlevels: int = 4,
    maxROIsize: Tuple[int, int] = (1024, 1024),
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Конвейер многомасштабной оценки ФРТ от грубого к точному масштабу."""
    G = np.asarray(G, dtype=np.float64)

    Gn, norm_m, norm_v = simpnormimg(G)

    # Применение шумоподавления до построения пирамиды
    pre_pyramid = PAR.get('pre_pyramid', None)
    pre_pyramid_params = PAR.get('pre_pyramid_params', None) or {}
    if pre_pyramid is not None and pre_pyramid != 'none':
        if Gn.ndim == 3:
            Gn_dn = np.empty_like(Gn)
            for c in range(Gn.shape[2]):
                Gn_dn[..., c] = apply_denoiser(
                    Gn[..., c], pre_pyramid, **pre_pyramid_params
                )
            Gn = Gn_dn
        else:
            Gn = apply_denoiser(Gn, pre_pyramid, **pre_pyramid_params)

    L = max(1, int(MSlevels))
    ROI: List[np.ndarray] = [None] * L
    HstarP: List[Optional[np.ndarray]] = [None] * L

    ROI[L - 1] = get_roi(Gn, tuple(maxROIsize))
    if Hstar is not None:
        HstarP[L - 1] = np.asarray(Hstar, dtype=np.float64)

    for i in range(L - 2, -1, -1):
        ROI[i] = imresize(ROI[i + 1], 0.5, method='bicubic')
        if HstarP[i + 1] is not None:
            HstarP[i] = imresize(HstarP[i + 1], 0.5, method='bicubic')

    hsize0 = (int(np.ceil(hsize[0] / (2 ** (L - 1)))),
              int(np.ceil(hsize[1] / (2 ** (L - 1)))))
    cen = ((hsize0[0] + 1) // 2 - 1, (hsize0[1] + 1) // 2 - 1)
    hi = np.zeros(hsize0, dtype=np.float64)
    hi[cen[0], cen[1]] = 1.0

    report = {'ms': [None] * L}

    h_current = hi
    for i in range(L):
        s = h_current.sum()
        if s != 0:
            h_current = h_current / s
            
        PAR['_cb_level'] = i
        PAR['_cb_num_levels'] = L
        H_est, _U_est, rep_i = psf_estim_lno_rgrad(
            ROI[i], h_current, PAR, HstarP[i]
        )
        report['ms'][i] = rep_i
        
        if i < L - 1:
            h_current = imresize(H_est, 2.0, method='lanczos3')
            ct = float(PAR.get('centering_threshold', 20.0 / 255.0))
            if ct > 0:
                h_current = center_psf(h_current, max(ct * 0.5, 1e-3))
        else:
            h_current = H_est

    H_pos = h_current.copy()
    H_pos[H_pos < 0] = 0.0
    s = H_pos.sum()
    if s != 0:
        H = h_current / s
    else:
        H = h_current.copy()

    pre_nonblind = PAR.get('pre_nonblind', None)
    pre_nonblind_params = PAR.get('pre_nonblind_params', None) or {}
    if pre_nonblind is not None and pre_nonblind != 'none':
        if Gn.ndim == 3:
            Gn_for_nb = np.empty_like(Gn)
            for c in range(Gn.shape[2]):
                Gn_for_nb[..., c] = apply_denoiser(
                    Gn[..., c], pre_nonblind, **pre_nonblind_params
                )
        else:
            Gn_for_nb = apply_denoiser(Gn, pre_nonblind, **pre_nonblind_params)
    else:
        Gn_for_nb = Gn

    nonblind_method = PAR.get('nonblind_method', 'fft_cg_sr_al')
    if nonblind_method == 'fft_cg_sr_al' or nonblind_method in (None, 'none'):
        U = fft_cg_sr_al(Gn_for_nb, H, PAR)
    elif nonblind_method == 'ringing_removal':
        U = nonblind_ringing_removal(Gn_for_nb, H, PAR)
    elif nonblind_method == 'firls':
        U = nonblind_firls(Gn_for_nb, H, PAR)
    else:
        raise ValueError(f"Неизвестный параметр nonblind_method: {nonblind_method}")

    U = denormimg(U, norm_m, norm_v)

    report['par'] = PAR
    return U, H, report