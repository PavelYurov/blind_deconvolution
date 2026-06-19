"""
solvers.py

Основные функции решателей для алгоритма слепой деконволюции на основе 
логарифмического априорного распределения (LIP).

Основано на методах:
    D. Perrone, R. Diethelm, P. Favaro: "Blind Deconvolution via
    Lower-Bounded Logarithmic Image Priors", EMMCVPR 2015.

Содержит:
    - grad_tv_em : вычисление градиента для мажорированной модели полной вариации.
    - blind : базовый цикл градиентного спуска методом MM (Таблица 2 статьи).
    - blind_cv : прямо-двойственное расщепление Конда-Вю для мажорированной задачи.
    - blind_pd : точная реализация прямо-двойственного решателя (Таблица 1 статьи) 
      для невыпуклого функционала LIP.
    - build_pyramid : построение многомасштабной пирамиды.
    - coarse_to_fine : иерархическая оценка ядра (от грубого масштаба к точному).
"""

import numpy as np

from .utils import (
    shft,
    convn_valid,
    convn_full,
    pad_replicate,
    imresize_matlab,
)



def grad_tv_em(u: np.ndarray, ut: np.ndarray,
               epsilon: float = 1e-3, tau: float = 1e-1) -> np.ndarray:
    """
    Вычисление градиента для мажорированной оценки полной вариации (TV).

    Используется симметричная формулировка с четырьмя структурами окрестностей 
    (диагональные и прямые направления) и тремя вариантами сдвига, что в сумме 
    дает 12 членов, усредненных с коэффициентом 1/4.

    Оцениваемый регуляризатор имеет вид: log(|grad u| + tau).
    Его EM-мажоранта на предыдущей итерации ut формулируется как:
        |grad u| / (|grad ut| + tau) + const

    Параметры
    ---------
    u : ndarray
        Текущая оценка четкого изображения размерности (M, N).
    ut : ndarray
        Оценка изображения на предыдущей итерации (используется в знаменателе 
        мажоранты).
    epsilon : float, по умолчанию 1e-3
        Константа сглаживания для предотвращения деления на ноль под корнем.
    tau : float, по умолчанию 1e-1
        Параметр нижней границы логарифмического распределения.

    Возвращает
    ----------
    grad : ndarray
        Градиент мажорированного регуляризатора TV по переменной u.
    """
    deltas = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

    grad = np.zeros_like(u)

    for dx, dy in deltas:
        ux1 = shft(u, dx, 0) - shft(u, 0, 0)
        uy1 = shft(u, 0, dy) - shft(u, 0, 0)
        TV1 = np.sqrt(epsilon + ux1 ** 2 + uy1 ** 2)
        du1 = -ux1 - uy1

        utx1 = shft(ut, dx, 0) - shft(ut, 0, 0)
        uty1 = shft(ut, 0, dy) - shft(ut, 0, 0)
        TVt1 = np.sqrt(epsilon + utx1 ** 2 + uty1 ** 2)

        ux2 = shft(u, 0, 0) - shft(u, -dx, 0)
        uy2 = shft(u, -dx, dy) - shft(u, -dx, 0)
        TV2 = np.sqrt(epsilon + ux2 ** 2 + uy2 ** 2)
        du2 = ux2

        utx2 = shft(ut, 0, 0) - shft(ut, -dx, 0)
        uty2 = shft(ut, -dx, dy) - shft(ut, -dx, 0)
        TVt2 = np.sqrt(epsilon + utx2 ** 2 + uty2 ** 2)

        ux3 = shft(u, dx, -dy) - shft(u, 0, -dy)
        uy3 = shft(u, 0, 0) - shft(u, 0, -dy)
        TV3 = np.sqrt(epsilon + ux3 ** 2 + uy3 ** 2)
        du3 = uy3

        utx3 = shft(ut, dx, -dy) - shft(ut, 0, -dy)
        uty3 = shft(ut, 0, 0) - shft(ut, 0, -dy)
        TVt3 = np.sqrt(epsilon + utx3 ** 2 + uty3 ** 2)

        grad += du1 / TV1 / (tau + TVt1)
        grad += du2 / TV2 / (tau + TVt2)
        grad += du3 / TV3 / (tau + TVt3)

    grad /= 4.0
    return grad


def blind(f: np.ndarray, MK: int, NK: int, beta: float,
          u: np.ndarray, k: np.ndarray,
          outer_iters: int = 140,
          inner_iters: int = 5,
          tau: float = 1e-3,
          k_step: float = 1e-3,
          u_step: float = 1e-3) -> tuple:
    """
    Основной цикл слепой деконволюции на основе мажоризации-минимизации (MM).

    Минимизирует функцию стоимости:
        min_{u,k} (1 / 2*beta) * ||k * u - f||^2 + sum |grad u_{i,j}| / (|grad ut_{i,j}| + tau)

    Алгоритм поочередно выполняет:
    1. Градиентный спуск для оценки четкого изображения u.
    2. Градиентный спуск для оценки ядра k с последующей проекцией на 
       симплекс (k >= 0, sum(k) = 1).

    Параметры
    ---------
    f : ndarray
        Размытое изображение размерности (M, N).
    MK, NK : int
        Пространственные размеры ядра.
    beta : float
        Вес члена верности данных (соответствует параметру lambda).
    u : ndarray
        Начальная оценка четкого изображения (M+MK-1, N+NK-1).
    k : ndarray
        Начальная оценка ядра (MK, NK).
    outer_iters : int
        Количество внешних итераций (обновлений мажоранты).
    inner_iters : int
        Количество внутренних итераций градиентного спуска.
    tau : float
        Параметр нижней границы логарифмического распределения.
    k_step : float
        Коэффициент масштабирования шага для ядра.
    u_step : float
        Коэффициент масштабирования шага для изображения.

    Возвращает
    ----------
    u : ndarray
        Оцененное четкое изображение.
    k : ndarray
        Оцененное ядро размытия.
    """
    epsilon = 1e-3

    for it in range(outer_iters):
        ut = u.copy()
        for itt in range(inner_iters):
            synth = convn_valid(u, k)
            err = synth - f
            gradu = (beta * convn_full(err, np.rot90(k, 2))
                     + grad_tv_em(u, ut, epsilon, tau))
            dt = u_step * (u.max() + 1.0 / u.size) / (np.abs(gradu).max() + 1e-30)
            u = u - dt * gradu

            synth = convn_valid(u, k)
            err = synth - f
            gradk = convn_valid(np.rot90(u, 2), err)
            alpha = k_step * (k.max() + 1.0 / k.size) / (np.abs(gradk).max() + 1e-30)
            k = k - alpha * gradk
            k = np.maximum(k, 0.0)
            k_sum = k.sum()
            if k_sum > 0:
                k /= k_sum

    return u, k


def blind_cv(f: np.ndarray, MK: int, NK: int, beta: float,
             u: np.ndarray, k: np.ndarray,
             outer_iters: int = 140,
             inner_iters: int = 5,
             tau_param: float = 1e-3,
             k_step: float = 1e-3) -> tuple:
    """
    Прямо-двойственное расщепление Конда-Вю для решения мажорированной 
    подзадачи взвешенной полной вариации.

    Логарифмический априорный регуляризатор сначала мажорируется (как в методе MM), 
    после чего полученная подзадача решается с помощью расщепления Конда-Вю. 
    Верность данных учитывается через градиент с использованием пространственных 
    сверток (во избежание круговых граничных артефактов).

    Размеры шагов удовлетворяют строгому условию сходимости:
        1/tau - sigma * ||grad||^2 > L_f / 2

    Возвращает
    ----------
    u : ndarray
        Оцененное четкое изображение.
    k : ndarray
        Оцененное ядро размытия.
    """
    epsilon = 1e-3
    Mu, Nu = u.shape

    p = np.zeros((2, Mu, Nu))

    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target = (MK - 1) / 2.0
    cx_target = (NK - 1) / 2.0

    for it in range(outer_iters):
        grad_x_ut = np.roll(u, -1, axis=1) - u
        grad_y_ut = np.roll(u, -1, axis=0) - u
        w = np.sqrt(epsilon + grad_x_ut ** 2 + grad_y_ut ** 2)
        radius = 1.0 / (w + tau_param)

        u_bar = u.copy()

        L_f = beta
        sigma_pd = 0.99 * L_f / 16.0
        tau_pd = 0.99 / L_f
        theta = 1.0

        for itt in range(inner_iters):
            grad_x = np.roll(u_bar, -1, axis=1) - u_bar
            grad_y = np.roll(u_bar, -1, axis=0) - u_bar

            ptx = p[0] + sigma_pd * grad_x
            pty = p[1] + sigma_pd * grad_y

            norm_pt = np.sqrt(ptx ** 2 + pty ** 2 + 1e-30)
            proj_scale = np.minimum(1.0, radius / norm_pt)
            p[0] = ptx * proj_scale
            p[1] = pty * proj_scale

            u_old = u.copy()
            synth = convn_valid(u, k)
            err = synth - f
            grad_data = beta * convn_full(err, np.rot90(k, 2))
            div_p = (p[0] - np.roll(p[0], 1, axis=1)) + \
                    (p[1] - np.roll(p[1], 1, axis=0))
            u = u - tau_pd * grad_data + tau_pd * div_p
            u = np.maximum(u, 0.0)
            u_bar = u + theta * (u - u_old)

        synth = convn_valid(u, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u, 2), err)
        alpha_k = k_step * (k.max() + 1.0 / k.size) / \
                  (np.abs(gradk).max() + 1e-30)
        k = k - alpha_k * gradk
        k = np.maximum(k, 0.0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        k_sum_c = k.sum()
        if k_sum_c > 0:
            cy = (ys * k).sum() / k_sum_c
            cx = (xs * k).sum() / k_sum_c
            dy = int(round(cy_target - cy))
            dx = int(round(cx_target - cx))
            if dy != 0 or dx != 0:
                k = np.roll(k, dy, axis=0)
                k = np.roll(k, dx, axis=1)
                u = np.roll(u, -dy, axis=0)
                u = np.roll(u, -dx, axis=1)
                p[0] = np.roll(p[0], -dy, axis=0)
                p[0] = np.roll(p[0], -dx, axis=1)
                p[1] = np.roll(p[1], -dy, axis=0)
                p[1] = np.roll(p[1], -dx, axis=1)

    return u, k


def _grad_neumann(u: np.ndarray):
    """Вычисление прямых разностей с условиями Неймана (нули на краях)."""
    gx = np.zeros_like(u)
    gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def _div_neumann(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Оператор дивергенции, сопряженный к оператору _grad_neumann."""
    dx = np.zeros_like(px)
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, 0]    = px[:, 0]
    dx[:, -1]   = -px[:, -2]

    dy = np.zeros_like(py)
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[0, :]    = py[0, :]
    dy[-1, :]   = -py[-2, :]
    return dx + dy


def _h_function(xi: np.ndarray, mu: float, eps: float,
                sigma: float) -> np.ndarray:
    """
    Решение одномерной подзадачи минимизации для проксимального оператора.

    Аналитически решается кубическое уравнение относительно rho:
        rho^3 - rho^2 + c*rho + d = 0,
        где c = eps^2/xi^2 + 2*sigma/(mu*xi^2), d = -eps^2/xi^2.
    """
    xi = np.asarray(xi, dtype=np.float64)
    eps2 = eps * eps

    rho = np.zeros_like(xi)

    safe = xi > 1e-12
    if not np.any(safe):
        return rho

    xi_s = xi[safe]
    xi2 = xi_s * xi_s

    c = eps2 / xi2 + 2.0 * sigma / (mu * xi2)
    d = -eps2 / xi2

    p = c - 1.0 / 3.0
    q = -2.0 / 27.0 + c / 3.0 + d

    disc = q * q / 4.0 + (p ** 3) / 27.0

    rho_s = np.empty_like(xi_s)
    three_real = disc < 0
    one_real = ~three_real

    if np.any(three_real):
        p_t = p[three_real]
        q_t = q[three_real]
        xi2_t = xi2[three_real]
        r = 2.0 * np.sqrt(-p_t / 3.0)
        arg = (3.0 * q_t) / (2.0 * p_t) * np.sqrt(-3.0 / p_t)
        arg = np.clip(arg, -1.0, 1.0)
        phi = np.arccos(arg) / 3.0
        t0 = r * np.cos(phi)
        t1 = r * np.cos(phi - 2.0 * np.pi / 3.0)
        t2 = r * np.cos(phi - 4.0 * np.pi / 3.0)
        r0 = t0 + 1.0 / 3.0
        r1 = t1 + 1.0 / 3.0
        r2 = t2 + 1.0 / 3.0

        def _obj(rr):
            return (mu / (2.0 * sigma)) * (rr - 1.0) ** 2 * xi2_t \
                   + np.log(rr * rr * xi2_t + eps2)

        o0 = _obj(r0)
        o1 = _obj(r1)
        o2 = _obj(r2)
        stack_r = np.stack([r0, r1, r2], axis=0)
        stack_o = np.stack([o0, o1, o2], axis=0)
        best = np.argmin(stack_o, axis=0)
        rho_s[three_real] = np.take_along_axis(
            stack_r, best[None], axis=0)[0]

    if np.any(one_real):
        p_o = p[one_real]
        q_o = q[one_real]
        disc_o = disc[one_real]
        sqd = np.sqrt(np.maximum(disc_o, 0.0))
        u3 = -q_o / 2.0 + sqd
        v3 = -q_o / 2.0 - sqd
        t = np.cbrt(u3) + np.cbrt(v3)
        rho_s[one_real] = t + 1.0 / 3.0

    rho_s = np.clip(rho_s, 0.0, 1.0)
    rho[safe] = rho_s
    return rho


def _build_h_lut(mu: float, eps: float, sigma: float,
                 xi_max: float = 2.0, n_grid: int = 4096) -> tuple:
    """Предварительное вычисление функции H на равномерной сетке."""
    xi_grid = np.linspace(0.0, xi_max, n_grid, dtype=np.float64)
    h_grid = _h_function(xi_grid, mu=mu, eps=eps, sigma=sigma)
    return xi_grid, h_grid


def _h_lut_apply(xi: np.ndarray, xi_grid: np.ndarray,
                 h_grid: np.ndarray) -> np.ndarray:
    """Векторизованная линейная интерполяция по таблице H-LUT."""
    return np.interp(xi, xi_grid, h_grid,
                     left=h_grid[0], right=h_grid[-1])


def blind_pd(f: np.ndarray, MK: int, NK: int, beta: float,
             u: np.ndarray, k: np.ndarray,
             outer_iters: int = 30,
             inner_iters: int = 50,
             tau_param: float = 1e-3,
             k_step: float = 1e-3,
             theta: float = 1.0,
             pd_tau: float = None,
             pd_sigma: float = None,
             h_mode: str = 'closed',
             h_lut_size: int = 4096,
             h_lut_xi_max: float = 4.0) -> tuple:
    """
    Прямо-двойственная слепая деконволюция на основе алгоритма Шамболя-Пока.

    Решает задачу с невыпуклым регуляризатором LIP напрямую. Для каждой 
    внешней итерации ядро фиксируется, а во внутреннем цикле выполняется 
    прямо-двойственное обновление:
    
        z1^{n+1} = (z1^n + sigma*(k * u_bar^n - f)) / (1 + sigma)
        zeta = z2^n + sigma * grad(u_bar^n)
        xi = ||zeta|| / sigma
        z2^{n+1} = (1 - H(xi, mu, eps, sigma)) * zeta
        u^{n+1} = u^n - tau * (k^T * z1^{n+1} - div(z2^{n+1}))

    Значения шагов по умолчанию (сбалансированные):
        tau = sigma = 0.99 / sqrt(||K||^2) ~ 0.33, 
    что обеспечивает выполнение условия сходимости Шамболя-Пока.
    """
    epsilon = float(tau_param)
    mu = float(beta)

    K_norm2 = 9.0
    default_step = 0.99 / np.sqrt(K_norm2)
    tau_pd = float(pd_tau) if pd_tau is not None else default_step
    sigma_pd = float(pd_sigma) if pd_sigma is not None else tau_pd

    h_mode_l = str(h_mode).lower()
    if h_mode_l == 'lut':
        _xi_grid, _h_grid = _build_h_lut(
            mu=mu, eps=epsilon, sigma=sigma_pd,
            xi_max=float(h_lut_xi_max), n_grid=int(h_lut_size))
        def _H(xi_arr):
            return _h_lut_apply(xi_arr, _xi_grid, _h_grid)
    elif h_mode_l == 'closed':
        def _H(xi_arr):
            return _h_function(xi_arr, mu, epsilon, sigma_pd)
    else:
        raise ValueError(f"h_mode must be 'closed' or 'lut', got {h_mode!r}")

    Mu, Nu = u.shape
    z1 = np.zeros_like(f)
    z2x = np.zeros((Mu, Nu))
    z2y = np.zeros((Mu, Nu))
    u_tilde = u.copy()
    u_bar = u.copy()

    ys, xs = np.mgrid[0:MK, 0:NK]
    cy_target = (MK - 1) / 2.0
    cx_target = (NK - 1) / 2.0

    for it in range(outer_iters):
        for itt in range(inner_iters):
            Kub = convn_valid(u_bar, k)
            z1 = (z1 + sigma_pd * (Kub - f)) / (1.0 + sigma_pd)

            gx, gy = _grad_neumann(u_bar)
            zx = z2x + sigma_pd * gx
            zy = z2y + sigma_pd * gy
            xi = np.sqrt(zx * zx + zy * zy) / sigma_pd
            H = _H(xi)
            scale = 1.0 - H
            z2x = scale * zx
            z2y = scale * zy

            Kstar_z1 = convn_full(z1, np.rot90(k, 2))
            div_z2 = _div_neumann(z2x, z2y)
            u_new = u_tilde - tau_pd * (Kstar_z1 - div_z2)

            u_bar = u_new + theta * (u_new - u_tilde)
            u_tilde = u_new

        u = u_tilde

        synth = convn_valid(u, k)
        err = synth - f
        gradk = convn_valid(np.rot90(u, 2), err)
        alpha_k = k_step * (k.max() + 1.0 / k.size) / \
                  (np.abs(gradk).max() + 1e-30)
        k = k - alpha_k * gradk
        k = np.maximum(k, 0.0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        k_sum_c = k.sum()
        if k_sum_c > 0:
            cy = (ys * k).sum() / k_sum_c
            cx = (xs * k).sum() / k_sum_c
            dy = int(round(cy_target - cy))
            dx = int(round(cx_target - cx))
            if dy != 0 or dx != 0:
                k = np.roll(np.roll(k, dy, axis=0), dx, axis=1)
                u = np.roll(np.roll(u, -dy, axis=0), -dx, axis=1)
                u_tilde = np.roll(np.roll(u_tilde, -dy, axis=0), -dx, axis=1)
                u_bar = np.roll(np.roll(u_bar, -dy, axis=0), -dx, axis=1)
                z2x = np.roll(np.roll(z2x, -dy, axis=0), -dx, axis=1)
                z2y = np.roll(np.roll(z2y, -dy, axis=0), -dx, axis=1)

    return u, k


def _make_odd(val: int) -> int:
    """Округление целого числа до ближайшего нечетного (с вычитанием 1 для четных)."""
    return val - 1 if val % 2 == 0 else val


def build_pyramid(f: np.ndarray, MK: int, NK: int,
                  lam: float, lambda_mult: float,
                  scale_mult: float = 1.4142135623730951):
    """
    Построение пирамиды изображений, размеров ядер и значений параметра 
    lambda для иерархической обработки.

    Параметры
    ---------
    f : ndarray
        Изображение в исходном (максимальном) разрешении.
    MK, NK : int
        Размеры ядра в исходном разрешении.
    lam : float
        Параметр lambda для финального (самого точного) масштаба.
    lambda_mult : float
        Множитель lambda между уровнями.
    scale_mult : float
        Делитель размера ядра между уровнями.
    """
    M, N = f.shape[:2]
    smallest_scale = 3

    fp = [f]
    Mp = [M]
    Np = [N]
    MKp = [MK]
    NKp = [NK]
    lambdas = [lam]

    num_scales = 1

    while MKp[num_scales - 1] > smallest_scale and NKp[num_scales - 1] > smallest_scale:
        prev = num_scales - 1 

        lambdas.append(lambdas[prev] / lambda_mult)

        new_mk = round(MKp[prev] / scale_mult)
        new_nk = round(NKp[prev] / scale_mult)
        new_mk = _make_odd(new_mk)
        new_nk = _make_odd(new_nk)

        if new_nk == NKp[prev]:
            new_nk -= 2
        if new_mk == MKp[prev]:
            new_mk -= 2

        new_mk = max(new_mk, smallest_scale)
        new_nk = max(new_nk, smallest_scale)

        MKp.append(new_mk)
        NKp.append(new_nk)

        factor_m = MKp[prev] / new_mk
        factor_n = NKp[prev] / new_nk

        new_m = round(Mp[prev] / factor_m)
        new_n = round(Np[prev] / factor_n)
        new_m = _make_odd(new_m)
        new_n = _make_odd(new_n)

        Mp.append(new_m)
        Np.append(new_n)

        fp.append(imresize_matlab(f, (new_m, new_n)))

        num_scales += 1

    return fp, Mp, Np, MKp, NKp, lambdas, num_scales


def coarse_to_fine(f: np.ndarray, MK: int, NK: int,
                   blind_params: dict, ctf_params: dict,
                   verbose: bool = False, method: str = 'mm'):
    """
    Многомасштабная слепая деконволюция. Выполняет итерационную оценку 
    перемещаясь от грубого разрешения к точному (исходному).
    """
    final_lambda = ctf_params.get('final_lambda')
    lambda_mult = ctf_params.get('lambda_mult', 2.1)
    scale_mult = ctf_params.get('scale_mult', np.sqrt(2))

    fp, Mp, Np, MKp, NKp, lambdas, num_scales = build_pyramid(
        f, MK, NK, final_lambda, lambda_mult, scale_mult)

    u = pad_replicate(f, MK // 2, NK // 2)
    k = np.ones((MK, NK), dtype=np.float64) / (MK * NK)

    k_steps = blind_params.get('k_step', np.array([1e-3]))
    u_steps = blind_params.get('u_step', np.array([1e-3]))
    if np.isscalar(k_steps):
        k_steps = np.array([k_steps])
    if np.isscalar(u_steps):
        u_steps = np.array([u_steps])

    outer_iters = blind_params.get('outer_iters', 140)
    inner_iters = blind_params.get('inner_iters', 5)
    tau = blind_params.get('tau', 1e-3)

    for scale_idx in range(num_scales - 1, -1, -1):
        Ms = Mp[scale_idx]
        Ns = Np[scale_idx]
        MKs = MKp[scale_idx]
        NKs = NKp[scale_idx]

        u = imresize_matlab(u, (Ms + MKs - 1, Ns + NKs - 1))
        k = imresize_matlab(k, (MKs, NKs))
        k = k * (k > 0)
        k_sum = k.sum()
        if k_sum > 0:
            k /= k_sum

        fs = fp[scale_idx]
        lam = lambdas[scale_idx]

        if verbose:
            print(f"scale: {scale_idx}  lambda: {lam:.4f}  "
                  f"MKs: {MKs}  NKs: {NKs}  outer_iters: {outer_iters}")

        for phase in range(len(k_steps)):
            if method == 'mm':
                u, k = blind(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau=tau,
                    k_step=float(k_steps[phase]),
                    u_step=float(u_steps[phase]),
                )
            elif method == 'pd':
                u, k = blind_pd(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    k_step=float(k_steps[phase]),
                    pd_tau=blind_params.get('pd_tau', None),
                    pd_sigma=blind_params.get('pd_sigma', None),
                    theta=blind_params.get('pd_theta', 1.0),
                    h_mode=blind_params.get('h_mode', 'closed'),
                    h_lut_size=blind_params.get('h_lut_size', 4096),
                    h_lut_xi_max=blind_params.get('h_lut_xi_max', 4.0),
                )
            elif method == 'cv':
                u, k = blind_cv(
                    fs, MKs, NKs, lam,
                    u, k,
                    outer_iters=outer_iters,
                    inner_iters=inner_iters,
                    tau_param=tau,
                    k_step=float(k_steps[phase]),
                )
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Choose 'mm', 'pd', or 'cv'.")

    return u, k
