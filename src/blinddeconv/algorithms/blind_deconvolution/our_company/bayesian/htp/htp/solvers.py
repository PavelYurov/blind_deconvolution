"""
solvers.py

Функции-решатели для алгоритма слепой деконволюции с априорными 
распределениями с тяжелыми хвостами (HTP).

Содержит:
    - psf_estim_lno_rgrad: Совместная чередующаяся MAP-оценка для скрытого 
      изображения u и функции рассеяния точки (ФРТ) h на одном масштабе.
    - fft_cg_sr_al: Быстрая неслепая деконволюция с использованием метода 
      расщепления Брегмана в Фурье-области.
    - mc_restoration: Многомасштабный конвейер алгоритма оценки от грубого 
      к точному масштабу.

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


def psf_estim_lno_rgrad(
    G: np.ndarray,
    iH: np.ndarray,
    PAR: Dict,
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Совместная оценка скрытого изображения U и ФРТ H на одном масштабе 
    с использованием полуквадратичного расщепления и итераций Брегмана.

    Решает задачу минимизации (Уравнение 2 из [1]):
        min_{u,h}  (gamma/2) * ||h*u - g||^2 
                  + alpha_u * sum(|D_x u|^p + |D_y u|^p)
                  + alpha_h * ||h||_1     (h >= 0)

    Шаг оценки h выполняется в пространстве градиентов для повышения 
    стабильности (Раздел 3.1 и 3.2 из [1]).

    Параметры
    ----------
    G     : Наблюдаемое размытое изображение, форма (H, W).
    iH    : Начальная оценка ФРТ, форма (kh, kw).
    PAR   : Словарь параметров алгоритма.
    Hstar : Истинная ФРТ (опционально, для вычисления СКО/MSE).

    Возвращает
    -------
    H      : Оцененная ФРТ (сумма равна 1, неотрицательная, отцентрированная).
    U      : Оцененное скрытое изображение на текущем масштабе.
    Report : Словарь с диагностической информацией (например, СКО по итерациям).
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
    kernel_thresh = float(PAR.get('kernel_thresh', 0.0))
    iterative_recenter = bool(PAR.get('iterative_recenter', True))
    verbose = int(PAR.get('verbose', 0))

    # --- Инициализация размеров ---
    iH = np.asarray(iH, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    hsize = iH.shape[:2]
    gsize = G.shape[:2]
    usize = gsize 
    M, N = usize

    # --- Отслеживание СКО (MSE) ---
    do_mse = Hstar is not None and np.asarray(Hstar).size > 0
    if do_mse:
        Report['hstep']['mse'] = np.zeros(maxiter + 1, dtype=np.float64)

    U = np.zeros(usize, dtype=np.float64)
    H = iH.copy()

    # --- Фурье-образы операторов производных ---
    FDx = fft2_pad(np.array([[1.0, -1.0]]), M, N)
    FDy = fft2_pad(np.array([[1.0], [-1.0]]), M, N)
    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # --- Вспомогательные переменные Брегмана ---
    Vx = np.zeros(usize, dtype=np.float64)
    Vy = np.zeros(usize, dtype=np.float64)
    Vh = np.zeros(usize, dtype=np.float64)
    Bx = np.zeros(usize, dtype=np.float64)
    By = np.zeros(usize, dtype=np.float64)
    Bh = np.zeros(usize, dtype=np.float64)

    if do_mse:
        Report['hstep']['mse'][0] = calculate_mse(H, np.asarray(Hstar))

    # --- Подготовка наблюдаемого изображения ---
    eG = edgetaper(G, np.ones(hsize, dtype=np.float64) / np.prod(hsize))
    FeGu = np.fft.fft2(eG)
    FeGx = FDx * FeGu
    FeGy = FDy * FeGu

    state = {
        'FU': np.fft.fft2(U), 
        'FUx': np.zeros(usize, dtype=complex),
        'FUy': np.zeros(usize, dtype=complex)
    }

    def ustep(gamma_local: float):
        """Шаг оценки скрытого изображения u (Раздел 3.1 из [1])."""
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
        """Шаг оценки ФРТ h (Раздел 3.2 из [1])."""
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
            
            # --- Условие неотрицательности ядра ---
            Vh[Vh < 0] = 0.0
            
            # --- Обнуление значений за пределами носителя ФРТ ---
            Vh[hsize[0]:, :] = 0.0
            Vh[:hsize[0], hsize[1]:] = 0.0
            
            # Обновление переменных Брегмана
            Bh = Bh + Vh - hI

            H_local = hI[:hsize[0], :hsize[1]]

            if relcon < ccreltol:
                break

        if verbose:
            print(f'  min_H step {i}  relcon: {relcon:.3e}')
        return H_local

    # --- Внешний чередующийся цикл (Alternating MAP) ---
    for mI in range(1, maxiter + 1):
        ustep(gamma)
        H = hstep(gamma)

        # Итеративное жесткое пороговое ограничение (опционально)
        if kernel_thresh > 0.0:
            H_pos = np.maximum(H, 0.0)
            mx = H_pos.max()
            if mx > 0:
                H = np.where(H_pos < kernel_thresh * mx, 0.0, H_pos)
                s = H.sum()
                if s > 0:
                    H = H / s

        # Итеративное центрирование для предотвращения смещения ФРТ
        if iterative_recenter and centering_threshold > 0 and mI < maxiter:
            H = center_psf(H, centering_threshold)

        if do_mse:
            Report['hstep']['mse'][mI] = calculate_mse(H, np.asarray(Hstar))

        # Продолжение по параметру gamma (помогает избежать локальных минимумов)
        gamma = gamma * 1.5

    # --- Финальное центрирование и очистка ФРТ ---
    if centering_threshold > 0:
        H = center_psf(H, centering_threshold)

    U = np.real(np.fft.ifft2(state['FU']))
    return H, U, Report


def fft_cg_sr_al(G: np.ndarray, H: np.ndarray, PAR: Dict) -> np.ndarray:
    """
    Быстрая неслепая деконволюция с использованием дополненного лагранжиана / 
    расщепления Брегмана в Фурье-области.

    Решает задачу:
        min_u   (gamma/2) * ||g - H * u||^2 + alpha * ||grad(u)||_p^p

    Параметры
    ----------
    G  : Наблюдаемое изображение, форма (H, W) или (H, W, C).
    H  : ФРТ, форма (kh, kw) (нормализованная, неотрицательная).
    PAR: Словарь параметров. Используются gamma_nonblind, beta_u_nonblind,
         Lp_nonblind (при их отсутствии используется fallback на параметры 
         слепого шага).

    Возвращает
    -------
    U : Восстановленное изображение, форма совпадает с G.
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

    # --- Ограничения по яркости для каждого канала ---
    vrange = np.zeros((C, 2), dtype=np.float64)
    for c in range(C):
        ch = G[..., c]
        vrange[c, 0] = ch.min()
        vrange[c, 1] = ch.max()

    # Сдвиг центра ФРТ (чтобы свертка не смещала изображение в Фурье-области)
    hshift = np.zeros_like(H_psf)
    hshift[H_psf.shape[0] // 2, H_psf.shape[1] // 2] = 1.0

    # --- Инициализация Фурье-преобразований ---
    FDx_2d = fft2_pad(np.array([[1.0, -1.0]]), Hh, Ww)
    FDy_2d = fft2_pad(np.array([[1.0], [-1.0]]), Hh, Ww)
    FDx = np.repeat(FDx_2d[..., None], C, axis=2)
    FDy = np.repeat(FDy_2d[..., None], C, axis=2)

    FH_2d = (np.conj(fft2_pad(hshift, Hh, Ww)) * fft2_pad(H_psf, Hh, Ww))
    FH = np.repeat(FH_2d[..., None], C, axis=2)
    FHTH = np.conj(FH) * FH

    eG = edgetaper(G if not squeeze_out else G[..., 0], H_psf)
    if eG.ndim == 2:
        eG = eG[..., None]
    FGu = np.fft.fft2(eG, axes=(0, 1))
    FGs = np.conj(FH) * FGu

    DTD = np.conj(FDx) * FDx + np.conj(FDy) * FDy

    # --- Вспомогательные переменные Брегмана ---
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
        
        # Векторное Lp: норма агрегируется по каналам
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

    # --- Применение ограничений диапазона значений по каналам ---
    for c in range(C):
        lo, hi = vrange[c, 0], vrange[c, 1]
        ch = U[..., c]
        ch[ch < lo] = lo
        ch[ch > hi] = hi
        U[..., c] = ch

    if squeeze_out:
        U = U[..., 0]
    return U


def mc_restoration(
    G: np.ndarray,
    hsize: Tuple[int, int],
    PAR: Dict,
    MSlevels: int = 4,
    maxROIsize: Tuple[int, int] = (1024, 1024),
    Hstar: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Многомасштабный конвейер слепой деконволюции от грубого к точному масштабу.

    Алгоритм:
        1. Нормализация интенсивностей изображения в [0, 1].
        2. Построение пирамиды центральной области (зеленый канал для RGB).
        3. Инициализация ФРТ дельта-функцией на самом грубом уровне.
        4. На каждом уровне пирамиды: чередующаяся MAP-оценка, затем 
           интерполяция ФРТ (увеличение в 2 раза) для следующего уровня.
        5. Финальное ограничение ФРТ (неотрицательность, нормализация суммы).
        6. Финальная неслепая деконволюция полного изображения.
        7. Денормализация в исходный диапазон интенсивностей.

    Параметры
    ----------
    G          : Входное изображение (градации серого или RGB).
    hsize      : Верхняя граница размера ФРТ на самом точном масштабе (kh, kw).
    PAR        : Словарь параметров алгоритма.
    MSlevels   : Количество уровней пирамиды (>=1).
    maxROIsize : Размер центральной области (H, W) для оценки ядра.
    Hstar      : Истинная ФРТ (опционально, для диагностики).

    Возвращает
    -------
    U      : Восстановленное изображение (float64).
    H      : Оцененная ФРТ (сумма равна 1).
    report : Словарь с диагностическими данными и использованными параметрами.
    """
    G = np.asarray(G, dtype=np.float64)

    # --- 1. Нормализация интенсивностей в [0, 1] ---
    Gn, norm_m, norm_v = simpnormimg(G)

    # --- 2. Построение пирамиды масштабов ---
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

    # --- 3. Инициализация ФРТ на самом грубом масштабе ---
    hsize0 = (int(np.ceil(hsize[0] / (2 ** (L - 1)))),
              int(np.ceil(hsize[1] / (2 ** (L - 1)))))
    # Вычисление центрального индекса для инициализации
    cen = ((hsize0[0] + 1) // 2 - 1, (hsize0[1] + 1) // 2 - 1)
    hi = np.zeros(hsize0, dtype=np.float64)
    hi[cen[0], cen[1]] = 1.0

    verbose = int(PAR.get('verbose', 0))
    if verbose:
        print('Estimating PSFs...')

    report = {'ms': [None] * L}

    # --- 4. MAP-оценка по уровням пирамиды ---
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
            # Увеличение ядра для следующего масштаба с интерполяцией
            h_current = imresize(H_est, 2.0, method='lanczos3')
            
            # Повторное центрирование после интерполяции (предотвращает смещение)
            ct = float(PAR.get('centering_threshold', 20.0 / 255.0))
            if ct > 0:
                h_current = center_psf(h_current, max(ct * 0.5, 1e-3))
        else:
            h_current = H_est

    # --- 5. Ограничение неотрицательности и нормализация ФРТ ---
    H_pos = h_current.copy()
    H_pos[H_pos < 0] = 0.0
    s = H_pos.sum()
    if s != 0:
        H = h_current / s
    else:
        H = h_current.copy()

    if verbose:
        print('PSF estimation done.')

    # --- 6. Финальная неслепая деконволюция полного изображения ---
    U = fft_cg_sr_al(Gn, H, PAR)
    if verbose:
        print('Nonblind deconvolution done.')

    # --- 7. Денормализация в исходный диапазон ---
    U = denormimg(U, norm_m, norm_v)

    report['par'] = PAR
    return U, H, report