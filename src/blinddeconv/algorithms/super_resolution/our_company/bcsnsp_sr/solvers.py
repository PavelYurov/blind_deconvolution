"""
solvers.py

Функции-решатели алгоритма сверхразрешения на основе байесовской комбинации 
разреженных и неразреженных априорных распределений (BCSNSP-SR).

Основные функции:
    create_data       - Симуляция наблюдений низкого разрешения (LR) из HR изображения.
    lk_var            - Регистрация Лукаса-Канаде с байесовской оценкой неопределенности.
    solvex_var_l4_sar - Итеративная реконструкция сверхразрешения с комбинированной регуляризацией.
"""

import numpy as np
import scipy.sparse as sp

from .utils import (
    circconvmatx2, dwnsmpl_matrix, warp_matrix_bilinear,
    unwrap_lr, restore_sar, pcg_solve,
    build_coord_grid, get_diff_kernels, get_sar_kernel,
    imresize,
)


# --- Симуляция наблюдений ---
def create_data(xtrue, h, M, N, res, L, sx_true, sy_true, theta_true, sigma):
    """
    Симуляция L наблюдений низкого разрешения на основе изображения высокого разрешения.

    Прямая модель для каждого кадра: y_k = A * H * C_k * x + шум
    Объединенная модель:             y   = W * x + шум

    Параметры
    ----------
    xtrue      : Истинное HR изображение, форма (M, N).
    h          : Ядро размытия 2D.
    M, N       : Размеры HR изображения.
    res        : Коэффициент субдискретизации (даунсэмплинга).
    L          : Количество LR кадров.
    sx_true    : Субпиксельные сдвиги по X для каждого кадра.
    sy_true    : Субпиксельные сдвиги по Y для каждого кадра.
    theta_true : Угол поворота для каждого кадра (в радианах).
    sigma      : Стандартное отклонение аддитивного шума.

    Возвращает
    -------
    y : Объединенный вектор наблюдений, форма (L * m * n,).
    W : Разреженная матрица системы, форма (L * m * n, M * N).
    """
    H = circconvmatx2(h, M, N)
    A = dwnsmpl_matrix(M, N, res)

    blocks = []
    for i in range(L):
        C, *_ = warp_matrix_bilinear(sx_true[i], sy_true[i],
                                      theta_true[i], M, N)
        blocks.append(A @ H @ C)

    W = sp.vstack(blocks, format='csr')

    m, n = M // res, N // res
    x_vec = xtrue.ravel(order='F')
    y = W @ x_vec + sigma * np.random.randn(L * m * n)
    return y, W


# --- Вспомогательные функции для ковариации ---
def _trace_diag(Oi, Oj, sigma_diag):
    """Вычисление следа произведения матриц: trace(Oi^T * Oj * diag(sigma_diag))."""
    d = np.array(Oi.multiply(Oj).sum(axis=0)).ravel()
    return float(np.dot(d, sigma_diag))


# --- Матрицы производных для регистрации ---
def _build_registration_derivatives(theta_k, a, b, Lbl, Lbr, Ltl, Ltr,
                                    A, H, X_coord, Y_coord, nopix):
    """Формирование операторов производных O1, O2, O3 для алгоритма регистрации."""
    P1 = sp.diags(-X_coord * np.sin(theta_k) - Y_coord * np.cos(theta_k),
                   0, shape=(nopix, nopix))
    P2 = sp.diags(X_coord * np.cos(theta_k) - Y_coord * np.sin(theta_k),
                   0, shape=(nopix, nopix))

    O2 = (sp.diags(1 - b, 0, shape=(nopix, nopix)) @ (Ltr - Ltl) +
          sp.diags(b, 0, shape=(nopix, nopix)) @ (Lbr - Lbl))
    O3 = (sp.diags(1 - a, 0, shape=(nopix, nopix)) @ (Lbl - Ltl) +
          sp.diags(a, 0, shape=(nopix, nopix)) @ (Lbr - Ltr))
    O1 = P1 @ O2 + P2 @ O3

    O1 = A @ H @ O1
    O2 = A @ H @ O2
    O3 = A @ H @ O3
    return O1, O2, O3


# --- Оценка параметров регистрации ---
def lk_var(x, yk, sigma_diag, A, H, Lambda_pk, betak,
           M, N, sx_k, sy_k, theta_k,
           sx_init, sy_init, theta_init,
           X_coord, Y_coord, method='variational',
           lk_maxit=20, lk_thr=1e-4):
    """
    Регистрация Лукаса-Канаде с байесовской оценкой неопределенности.

    Параметры
    ----------
    x            : Текущая оценка HR изображения, вектор формы (M*N,).
    yk           : Наблюдение LR для текущего кадра, форма (m*n,).
    sigma_diag   : Диагональ ковариационной матрицы изображения.
    A, H         : Матрицы субдискретизации и размытия.
    Lambda_pk    : Априорная ковариация регистрации, форма (3, 3).
    betak        : Точность (инвертированная дисперсия) шума.
    M, N         : Размеры HR изображения.
    sx_k, sy_k   : Текущие оценки сдвига.
    theta_k      : Текущая оценка поворота.
    sx_init, sy_init, theta_init : Инициализационные значения регистрации.
    X_coord, Y_coord : Векторы координат.
    method       : Метод оптимизации ('variational' или 'degenerate').
    lk_maxit     : Максимальное количество итераций.
    lk_thr       : Порог сходимости.

    Возвращает
    -------
    newsk   : Обновленные параметры [поворот, сдвиг X, сдвиг Y].
    Lambdak : Апостериорная матрица ковариации.
    """
    nopix = M * N

    sk = np.array([theta_k, sx_k, sy_k], dtype=np.float64)
    newsk = sk.copy()
    s_init = np.array([theta_init, sx_init, sy_init], dtype=np.float64)

    if np.sum(np.abs(Lambda_pk)) == 0:
        Lambda_pk_inv = np.zeros((3, 3))
    else:
        Lambda_pk_inv = np.linalg.inv(Lambda_pk)

    Lambdak = Lambda_pk.copy()
    Lambdak_min = Lambda_pk.copy()
    e_min = 1e10
    sk_min = sk.copy()

    for it in range(lk_maxit):
        thetak = sk[0]
        dx_val = sk[1]
        dy_val = sk[2]

        C, Lbl, Lbr, Ltl, Ltr, a, b = warp_matrix_bilinear(
            dx_val, dy_val, thetak, M, N)

        yhat = A @ H @ C @ x
        if it == 0:
            Lambdak = Lambdak_min.copy()

        e_norm = np.linalg.norm(yk - yhat)
        if e_norm <= e_min:
            e_min = e_norm
            sk_min = newsk.copy()
            Lambdak_min = Lambdak.copy()
        else:
            newsk = sk_min.copy()
            Lambdak = Lambdak_min.copy()
            break

        O1, O2, O3 = _build_registration_derivatives(
            thetak, a, b, Lbl, Lbr, Ltl, Ltr,
            A, H, X_coord, Y_coord, nopix)

        O1x = O1 @ x
        O2x = O2 @ x
        O3x = O3 @ x

        # Гессиан согласования данных
        Phik = np.array([
            [O1x @ O1x, O1x @ O2x, O1x @ O3x],
            [O1x @ O2x, O2x @ O2x, O2x @ O3x],
            [O1x @ O3x, O2x @ O3x, O3x @ O3x],
        ])

        Lambdak_inv = Lambda_pk_inv + betak * Phik

        err = yk - yhat
        rhs = (Lambda_pk_inv @ s_init +
               betak * Phik @ sk +
               betak * np.array([err @ O1x, err @ O2x, err @ O3x]))

        # Оценка неопределенности изображения
        if method == 'variational' and sigma_diag is not None:
            Psik = np.array([
                [_trace_diag(O1, O1, sigma_diag),
                 _trace_diag(O1, O2, sigma_diag),
                 _trace_diag(O1, O3, sigma_diag)],
                [_trace_diag(O1, O2, sigma_diag),
                 _trace_diag(O2, O2, sigma_diag),
                 _trace_diag(O2, O3, sigma_diag)],
                [_trace_diag(O1, O3, sigma_diag),
                 _trace_diag(O2, O3, sigma_diag),
                 _trace_diag(O3, O3, sigma_diag)],
            ])

            Lambdak_inv = Lambdak_inv + Psik

            AHC = A @ H @ C
            rhs = rhs + betak * Psik @ sk - betak * np.array([
                _trace_diag(AHC, O1, sigma_diag),
                _trace_diag(AHC, O2, sigma_diag),
                _trace_diag(AHC, O3, sigma_diag),
            ])

        if np.linalg.matrix_rank(Lambdak_inv) < 3:
            if it == 0:
                Lambdak = None
            break

        Lambdak = np.linalg.inv(Lambdak_inv)
        newsk = Lambdak @ rhs

        if np.linalg.norm(newsk - sk) / (np.linalg.norm(sk) + 1e-30) < lk_thr:
            sk = newsk
            break

        sk = newsk.copy()

    newsk = sk_min.copy()
    Lambdak = Lambdak_min.copy()
    return newsk, Lambdak


# --- Основной решатель ---
def solvex_var_l4_sar(y, *, M, N, m, n, res, L, h,
                      sx, sy, theta,
                      sx_init=None, sy_init=None, theta_init=None,
                      xtrue=None,
                      method='variational',
                      lambda_prior=0.5,
                      maxit=50,
                      thr=1e-4,
                      pcg_thr=1e-6,
                      pcg_maxit=100,
                      pcg_minit=10,
                      estimate_registration=True,
                      noise_estimate='SEPARATE',
                      approx_sigma=True,
                      fixed_parameters=False,
                      lk_maxit=20,
                      lk_thr=1e-4,
                      verbose=False):
    """
    Итеративная реконструкция сверхразрешения с комбинированным априорным 
    распределением L1 (анизотропный TV) и SAR.

    Параметры
    ----------
    y              : Вектор наблюдений LR, форма (L*m*n,).
    M, N           : Пространственные размеры HR изображения.
    m, n           : Пространственные размеры LR изображения.
    res            : Коэффициент масштабирования.
    L              : Количество LR кадров.
    h              : Ядро размытия.
    sx, sy, theta  : Начальные оценки регистрации.
    lambda_prior   : Параметр компромисса между L1 (TV) и SAR (диапазон [0, 1]).
    maxit          : Максимальное количество внешних итераций.
    thr            : Порог сходимости относительного изменения изображения.

    Возвращает
    -------
    x   : Вектор восстановленного HR изображения.
    out : Словарь с гиперпараметрами и историей оптимизации.
    """
    sx = np.array(sx, dtype=np.float64).copy()
    sy = np.array(sy, dtype=np.float64).copy()
    theta = np.array(theta, dtype=np.float64).copy()

    if sx_init is None:
        sx_init = sx.copy()
    if sy_init is None:
        sy_init = sy.copy()
    if theta_init is None:
        theta_init = theta.copy()

    nopix = M * N

    # --- Формирование базовых операторов ---
    dx_kern, dy_kern = get_diff_kernels()
    Dx = circconvmatx2(dx_kern, M, N)
    Dy = circconvmatx2(dy_kern, M, N)

    hsar = get_sar_kernel()
    Csar = circconvmatx2(hsar, M, N)
    CtC = Csar.T @ Csar

    H_mat = circconvmatx2(h, M, N)
    A_mat = dwnsmpl_matrix(M, N, res)

    X_coord, Y_coord = build_coord_grid(M, N)

    # --- Формирование начальной матрицы наблюдений W ---
    W_blocks = []
    for k in range(L):
        C, *_ = warp_matrix_bilinear(sx[k], sy[k], theta[k], M, N)
        W_blocks.append(A_mat @ H_mat @ C)
    W = sp.vstack(W_blocks, format='csr')

    # --- Инициализация оценки и гиперпараметров ---
    ys, yvecs = unwrap_lr(y, m, n, L)

    betak = np.zeros(L)
    for kk in range(L):
        _, _, beta_kk = restore_sar(ys[kk].astype(np.float64), h)
        betak[kk] = beta_kk

    # Интерполяция для стартовой оценки
    x = imresize(ys[0], res, order=3).ravel(order='F')

    y_max = np.max(np.abs(y))
    if y_max > 0:
        y = y / y_max
    x_max = np.max(np.abs(x))
    if x_max > 0:
        x = x / x_max

    xtrue_vec = None
    if xtrue is not None:
        xtrue_vec = xtrue.ravel(order='F').astype(np.float64)
        xt_max = np.max(np.abs(xtrue_vec))
        if xt_max > 0:
            xtrue_vec = xtrue_vec / xt_max

    ys, yvecs = unwrap_lr(y, m, n, L)

    u_h = (Dx @ x) ** 2 + np.finfo(float).eps
    u_v = (Dy @ x) ** 2 + np.finfo(float).eps

    T_h = sp.diags(1.0 / np.sqrt(u_h), 0, shape=(nopix, nopix))
    T_v = sp.diags(1.0 / np.sqrt(u_v), 0, shape=(nopix, nopix))

    alpha_h = (nopix / 4.0) / np.sum(np.sqrt(u_h))
    alpha_v = (nopix / 4.0) / np.sum(np.sqrt(u_v))

    Csarx = Csar @ x
    alpha_sar = nopix / (np.dot(Csarx, Csarx) + np.sum(CtC.diagonal()))

    e = y - W @ x
    if noise_estimate == 'SEPARATE':
        _, es = unwrap_lr(e, m, n, L)
        for kk in range(L):
            esk = np.sum(es[kk] ** 2) + 1e-30
            betak[kk] = len(es[kk]) / esk
    else:
        e_sq = np.sum(e ** 2) + 1e-30
        betak[:] = len(y) / e_sq

    Lambdas = [np.zeros((3, 3)) for _ in range(L)]
    Lambdas_p = [np.zeros((3, 3)) for _ in range(L)]

    history = {
        'PSNRs': [], 'MSEs': [], 'alpha_h': [], 'alpha_v': [],
        'alpha_sar': [], 'betak': [], 'xconv': [],
    }

    sigma_diag = None

    # --- Основной итеративный цикл ---
    for it in range(maxit):
        oldx = x.copy()

        # Построение матрицы системы и правой части
        Sigma_inv = sp.csr_matrix((nopix, nopix))
        W_blocks = []
        rhs = np.zeros(nopix)

        for k in range(L):
            C, Lbl, Lbr, Ltl, Ltr, a, b = warp_matrix_bilinear(
                sx[k], sy[k], theta[k], M, N)
            B = A_mat @ H_mat @ C
            W_blocks.append(B)

            rhs += betak[k] * (B.T @ yvecs[k])
            Sigma_inv = Sigma_inv + betak[k] * (B.T @ B)

            if method == 'variational':
                O1, O2, O3 = _build_registration_derivatives(
                    theta[k], a, b, Lbl, Lbr, Ltl, Ltr,
                    A_mat, H_mat, X_coord, Y_coord, nopix)

                O11 = O1.T @ O1
                O22 = O2.T @ O2
                O33 = O3.T @ O3
                O12 = O1.T @ O2
                O13 = O1.T @ O3
                O23 = O2.T @ O3

                Lam = Lambdas[k]
                Sigma_inv = Sigma_inv + betak[k] * (
                    Lam[0, 0] * O11 + Lam[1, 1] * O22 +
                    Lam[2, 2] * O33 +
                    2 * Lam[0, 1] * O12 + 2 * Lam[0, 2] * O13 +
                    2 * Lam[1, 2] * O23)

        W = sp.vstack(W_blocks, format='csr')

        # Добавление регуляризации
        Sigma_inv = Sigma_inv + lambda_prior * (
            alpha_h * Dx.T @ T_h @ Dx + alpha_v * Dy.T @ T_v @ Dy
        ) + (1 - lambda_prior) * alpha_sar * CtC

        # Решение системы методом сопряженных градиентов
        x, flag = pcg_solve(Sigma_inv, rhs, tol=pcg_thr,
                            max_iter=pcg_maxit, x0=x, min_iter=pcg_minit)

        if approx_sigma:
            diag_vals = Sigma_inv.diagonal().copy()
            diag_vals[diag_vals == 0] = 1e-30
            sigma_diag = 1.0 / diag_vals

        # Обновление параметров регистрации
        if estimate_registration:
            for k in range(1, L):
                newsk, Lambdak = lk_var(
                    x, yvecs[k], sigma_diag, A_mat, H_mat,
                    Lambdas_p[k], betak[k], M, N,
                    sx[k], sy[k], theta[k],
                    sx_init[k], sy_init[k], theta_init[k],
                    X_coord, Y_coord,
                    method=method, lk_maxit=lk_maxit, lk_thr=lk_thr)

                theta[k] = newsk[0]
                sx[k] = newsk[1]
                sy[k] = newsk[2]
                if Lambdak is not None:
                    Lambdas[k] = Lambdak

        # Обновление гиперпараметров
        if not fixed_parameters:
            if method == 'variational' and sigma_diag is not None:
                Sigma_mat = sigma_diag.reshape(M, N, order='F')
                DxSigma = Sigma_mat + np.roll(Sigma_mat, -1, axis=1)
                DySigma = Sigma_mat + np.roll(Sigma_mat, -1, axis=0)

                u_h = ((Dx @ x) ** 2 +
                       (1.0 / nopix) * DxSigma.ravel(order='F'))
                u_v = ((Dy @ x) ** 2 +
                       (1.0 / nopix) * DySigma.ravel(order='F'))
            else:
                u_h = (Dx @ x) ** 2
                u_v = (Dy @ x) ** 2

            u_h = np.maximum(u_h, np.finfo(float).eps)
            u_v = np.maximum(u_v, np.finfo(float).eps)

            T_h = sp.diags(1.0 / np.sqrt(u_h), 0, shape=(nopix, nopix))
            T_v = sp.diags(1.0 / np.sqrt(u_v), 0, shape=(nopix, nopix))

            alpha_h = (nopix / 4.0) / np.sum(np.sqrt(u_h))
            alpha_v = (nopix / 4.0) / np.sum(np.sqrt(u_v))

            Csarx = Csar @ x
            sar_denom = np.dot(Csarx, Csarx)
            if sigma_diag is not None:
                sar_denom += np.dot(sigma_diag, CtC.diagonal())
            alpha_sar = nopix / (sar_denom + 1e-30)

            e = y - W @ x
            if noise_estimate == 'SEPARATE':
                _, es = unwrap_lr(e, m, n, L)
                for kk in range(L):
                    esk = np.sum(es[kk] ** 2)

                    if method == 'variational' and sigma_diag is not None:
                        C_k, Lbl, Lbr, Ltl, Ltr, a, b = \
                            warp_matrix_bilinear(
                                sx[kk], sy[kk], theta[kk], M, N)
                        B_k = A_mat @ H_mat @ C_k

                        traceBkS = _trace_diag(B_k, B_k, sigma_diag)
                        esk += traceBkS

                        O1, O2, O3 = _build_registration_derivatives(
                            theta[kk], a, b, Lbl, Lbr, Ltl, Ltr,
                            A_mat, H_mat, X_coord, Y_coord, nopix)

                        O1x = O1 @ x
                        O2x = O2 @ x
                        O3x = O3 @ x

                        Lam = Lambdas[kk]
                        xOx = (Lam[0, 0] * (O1x @ O1x) +
                               Lam[1, 1] * (O2x @ O2x) +
                               Lam[2, 2] * (O3x @ O3x) +
                               2 * Lam[0, 1] * (O1x @ O2x) +
                               2 * Lam[0, 2] * (O1x @ O3x) +
                               2 * Lam[1, 2] * (O2x @ O3x))

                        trOS = (Lam[0, 0] * _trace_diag(O1, O1, sigma_diag) +
                                Lam[1, 1] * _trace_diag(O2, O2, sigma_diag) +
                                Lam[2, 2] * _trace_diag(O3, O3, sigma_diag) +
                                2 * Lam[0, 1] * _trace_diag(O1, O2, sigma_diag) +
                                2 * Lam[0, 2] * _trace_diag(O1, O3, sigma_diag) +
                                2 * Lam[1, 2] * _trace_diag(O2, O3, sigma_diag))

                        esk += trOS + xOx

                    betak[kk] = len(es[kk]) / (esk + 1e-30)
            else:
                e_sq = np.sum(e ** 2) + 1e-30
                betak[:] = len(y) / e_sq

        # Проверка сходимости
        xconv = np.linalg.norm(x - oldx) / (np.linalg.norm(oldx) + 1e-30)

        if xtrue_vec is not None:
            MSE = np.sum((x - xtrue_vec) ** 2) / nopix
            PSNR = 10 * np.log10(1.0 / (MSE + 1e-30))
            history['PSNRs'].append(float(PSNR))
            history['MSEs'].append(float(MSE))

        history['alpha_h'].append(float(alpha_h))
        history['alpha_v'].append(float(alpha_v))
        history['alpha_sar'].append(float(alpha_sar))
        history['betak'].append(betak.tolist())
        history['xconv'].append(float(xconv))

        if verbose:
            msg = (f"it={it + 1:3d}  α_h={alpha_h:.4f}  α_v={alpha_v:.4f}  "
                   f"α_sar={alpha_sar:.4f}  xconv={xconv:.2e}")
            if xtrue_vec is not None:
                msg += f"  PSNR={PSNR:.2f}"
            print(msg)

        if xconv < thr:
            break

    # --- Формирование результатов ---
    out = {
        'betak': betak,
        'alpha_h': alpha_h,
        'alpha_v': alpha_v,
        'alpha_sar': alpha_sar,
        'lambda_prior': lambda_prior,
        'Lambdas': Lambdas,
        'xconv': xconv,
        'theta': theta,
        'sx': sx,
        'sy': sy,
        'iterations': it + 1,
        'history': history,
    }
    return x, out