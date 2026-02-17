"""
Солверы для метода слепой деконволюции на основе дробного порядка
с PMP (Patch-wise Minimal Pixels) prior.

Литература:
    Wu, T., Wan, S., Feng, C., Zhang, H., & Zeng, T. (2024).
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
     Meets Fractional-Order Method."
    Journal of Mathematical Imaging and Vision, 67(1), 2.
    DOI: 10.1007/s10851-024-01221-x

Реализованные солверы:

1. solve_image_admm:
   Оценка латентного изображения f при фиксированном ядре h.
   Используется ADMM (Alternating Direction Method of Multipliers)
   для минимизации энергии с дробным TV и PMP-регуляризатором.

   Энергия f-подзадачи (Раздел 4 статьи):
       min_f  (1/2)||h * f - g||^2
            + lambda * ||nabla^alpha f||_{2,1}
            + mu * sum_i min_{j in P_i} |f_j|

2. solve_kernel:
   Оценка ядра h при фиксированном изображении f.
   Решение в градиентной области через FFT:
       min_h  ||nabla(h * f) - nabla g||^2 + gamma * ||h||^2
   с проекцией на допустимое множество {h >= 0, ||h||_1 = 1}.

3. coarse_to_fine:
   Многомасштабная (coarse-to-fine) схема слепой деконволюции.
   Пирамида Гаусса + последовательная оценка ядра от грубого к точному.

4. final_nonblind_deconv:
   Финальная не-слепая деконволюция методом ADMM с дробным TV.
"""

import numpy as np
from numpy.fft import fft2, ifft2

from .utils import (
    fft_fractional_operators,
    fft_gradient_operators,
    soft_threshold,
    vector_shrinkage,
    pmp_weight_map,
    center_kernel_fft,
    crop_kernel_from_fft,
    resize_kernel,
    kernel_threshold,
    build_gaussian_pyramid,
    edgetaper,
    compute_gradient,
)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Оценка латентного изображения (f-подзадача) через ADMM
# ─────────────────────────────────────────────────────────────────────────────

def solve_image_admm(g: np.ndarray,
                     h: np.ndarray,
                     f_init: np.ndarray,
                     alpha: float,
                     lambda_ftv: float,
                     mu_pmp: float,
                     patch_size: int,
                     beta1: float,
                     beta2: float,
                     num_iter: int,
                     fft_ops: dict = None) -> np.ndarray:
    """
    Оценка латентного изображения f при фиксированном ядре h
    посредством ADMM с двумя регуляризаторами:
        (i)  Изотропный дробный TV порядка alpha
        (ii) PMP (Patch-wise Minimal Pixels) prior

    Минимизируемый функционал:
        E(f) = (1/2)||h * f - g||_2^2
             + lambda * ||nabla^alpha f||_{2,1}
             + mu * sum_i w_i |f_i|

    где w_i — веса PMP, обновляемые итеративно
    (iteratively reweighted l1 аппроксимация PMP prior).

    ADMM формулировка (Раздел 4 статьи):
        Вводятся вспомогательные переменные:
            u1 = D_x^alpha f   (горизонтальная дробная производная)
            u2 = D_y^alpha f   (вертикальная дробная производная)
            v  = f             (для PMP)

        Расширенный Лагранжиан:
            L = (1/2)||h*f - g||^2
              + lambda * ||(u1, u2)||_{2,1}
              + mu * sum_i w_i |v_i|
              + (beta1/2)||D_x^alpha f - u1 + d1||^2
              + (beta1/2)||D_y^alpha f - u2 + d2||^2
              + (beta2/2)||f - v + d3||^2

    ADMM итерации:

    (a) f-обновление (в частотной области):
        F(f) = [conj(F(h)) F(g) + beta1 (conj(F(Dx)) F(z1) + conj(F(Dy)) F(z2)) + beta2 F(z3)]
             / [|F(h)|^2 + beta1 (|F(Dx)|^2 + |F(Dy)|^2) + beta2]
        где z1 = u1 - d1, z2 = u2 - d2, z3 = v - d3

    (b) (u1, u2)-обновление (изотропное векторное сжатие):
        (u1, u2) = VecShrink(D_x^alpha f + d1, D_y^alpha f + d2, lambda/beta1)

    (c) v-обновление (взвешенная мягкая пороговая обработка):
        v = shrink(f + d3, mu * w / beta2)

    (d) Обновление двойственных переменных:
        d1 <- d1 + D_x^alpha f - u1
        d2 <- d2 + D_y^alpha f - u2
        d3 <- d3 + f - v

    (e) Обновление весов PMP:
        w = pmp_weight_map(f)

    Parameters
    ----------
    g : np.ndarray
        Размытое изображение (H x W), float64, [0, 1].
    h : np.ndarray
        Текущая оценка ядра PSF (kh x kw).
    f_init : np.ndarray
        Начальное приближение изображения (H x W).
    alpha : float
        Порядок дробной производной (1 < alpha < 2).
    lambda_ftv : float
        Вес изотропного дробного TV.
    mu_pmp : float
        Вес PMP prior.
    patch_size : int
        Размер патча для PMP.
    beta1 : float
        Штрафной параметр ADMM для дробного TV.
    beta2 : float
        Штрафной параметр ADMM для PMP.
    num_iter : int
        Число итераций ADMM.
    fft_ops : dict or None
        Предвычисленные FFT операторы (для повторного использования).
        Ожидаемые ключи: 'F_Dx', 'F_Dy', 'F_Dx_sq', 'F_Dy_sq'.

    Returns
    -------
    f : np.ndarray
        Восстановленное изображение (H x W).
    """
    H, W = g.shape

    # Предвычисление FFT операторов
    if fft_ops is None:
        F_Dx, F_Dy, F_Dx_sq, F_Dy_sq = fft_fractional_operators((H, W), alpha)
    else:
        F_Dx = fft_ops['F_Dx']
        F_Dy = fft_ops['F_Dy']
        F_Dx_sq = fft_ops['F_Dx_sq']
        F_Dy_sq = fft_ops['F_Dy_sq']

    # FFT ядра
    H_padded = center_kernel_fft(h, (H, W))
    F_h = fft2(H_padded)
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2

    # FFT наблюдения
    F_g = fft2(g)

    # Инициализация
    f = f_init.copy()
    u1 = np.zeros((H, W), dtype=np.float64)
    u2 = np.zeros((H, W), dtype=np.float64)
    v = f.copy()
    d1 = np.zeros((H, W), dtype=np.float64)
    d2 = np.zeros((H, W), dtype=np.float64)
    d3 = np.zeros((H, W), dtype=np.float64)

    # Знаменатель f-обновления (постоянный в ADMM)
    denom = F_h_sq + beta1 * (F_Dx_sq + F_Dy_sq) + beta2

    # Карта весов PMP (инициализация)
    w_pmp = pmp_weight_map(f, patch_size)

    for it in range(num_iter):
        # (a) f-обновление (Частотная область)
        z1 = u1 - d1
        z2 = u2 - d2
        z3 = v - d3

        numer = (F_h_conj * F_g
                 + beta1 * (np.conj(F_Dx) * fft2(z1) + np.conj(F_Dy) * fft2(z2))
                 + beta2 * fft2(z3))

        f = np.real(ifft2(numer / denom))

        # (b) (u1, u2)-обновление: изотропное векторное сжатие
        #     D_x^alpha f + d1, D_y^alpha f + d2
        Dxf = np.real(ifft2(F_Dx * fft2(f)))
        Dyf = np.real(ifft2(F_Dy * fft2(f)))

        u1, u2 = vector_shrinkage(Dxf + d1, Dyf + d2, lambda_ftv / beta1)

        # (c) v-обновление: взвешенная мягкая пороговая обработка (PMP)
        v = soft_threshold(f + d3, mu_pmp * w_pmp / beta2)

        # (d) Обновление двойственных переменных
        d1 = d1 + Dxf - u1
        d2 = d2 + Dyf - u2
        d3 = d3 + f - v

        # (e) Периодическое обновление весов PMP (каждые 3 итерации)
        if (it + 1) % 3 == 0:
            w_pmp = pmp_weight_map(f, patch_size)

    # Ограничение диапазона [0, 1]
    f = np.clip(f, 0.0, 1.0)

    return f


# ─────────────────────────────────────────────────────────────────────────────
#  2. Оценка ядра размытия (h-подзадача) через FFT в градиентной области
# ─────────────────────────────────────────────────────────────────────────────

def solve_kernel(g: np.ndarray,
                 f: np.ndarray,
                 kernel_size: tuple,
                 gamma: float,
                 threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Оценка ядра размытия h при фиксированном латентном изображении f.

    Решается в градиентной области (более устойчиво к артефактам):
        min_h  ||h * nabla_x f - nabla_x g||^2
             + ||h * nabla_y f - nabla_y g||^2
             + gamma * ||h||^2

    Решение в частотной области (замкнутая форма):
        F(h) = [conj(F(f_x)) F(g_x) + conj(F(f_y)) F(g_y)]
             / [|F(f_x)|^2 + |F(f_y)|^2 + gamma]

    где f_x = nabla_x f, g_x = nabla_x g (конечные разности).

    После FFT-вычисления выполняется проекция на допустимое множество:
        h >= 0,  ||h||_1 = 1

    Parameters
    ----------
    g : np.ndarray
        Размытое изображение (H x W).
    f : np.ndarray
        Текущая оценка латентного изображения (H x W).
    kernel_size : tuple of (int, int)
        Размер ядра (kh, kw).
    gamma : float
        Вес регуляризации ядра (Тихоновская регуляризация).
    threshold_ratio : float
        Порог для обнуления малых элементов ядра.

    Returns
    -------
    kernel : np.ndarray
        Оценённое ядро (kh x kw), неотрицательное, нормированное.
    """
    H, W = g.shape

    # Градиенты изображений (конечные разности)
    fx, fy = compute_gradient(f)
    gx, gy = compute_gradient(g)

    # FFT градиентов
    F_fx = fft2(fx)
    F_fy = fft2(fy)
    F_gx = fft2(gx)
    F_gy = fft2(gy)

    # Решение в частотной области
    numer = np.conj(F_fx) * F_gx + np.conj(F_fy) * F_gy
    denom = np.abs(F_fx) ** 2 + np.abs(F_fy) ** 2 + gamma

    F_h = numer / denom
    h_full = np.real(ifft2(F_h))

    # Извлечение компактного ядра
    kernel = crop_kernel_from_fft(h_full, kernel_size)

    # Пороговая обработка
    kernel = kernel_threshold(kernel, threshold_ratio)

    return kernel


# ─────────────────────────────────────────────────────────────────────────────
#  3. Финальная не-слепая деконволюция (дробный TV через ADMM)
# ─────────────────────────────────────────────────────────────────────────────

def final_nonblind_deconv(g: np.ndarray,
                          h: np.ndarray,
                          alpha: float,
                          lambda_ftv: float,
                          beta: float = 1.0,
                          num_iter: int = 30) -> np.ndarray:
    """
    Финальная не-слепая деконволюция с дробным TV.

    Решается задача:
        min_f  (1/2)||h * f - g||^2 + lambda * ||nabla^alpha f||_{2,1}

    посредством ADMM (упрощённая версия без PMP, только дробный TV).

    ADMM итерации:
    (a) f-обновление:
        F(f) = [conj(F(h)) F(g) + beta (conj(F(Dx)) F(u1 - d1) + conj(F(Dy)) F(u2 - d2))]
             / [|F(h)|^2 + beta (|F(Dx)|^2 + |F(Dy)|^2)]

    (b) (u1, u2)-обновление: VecShrink(Dxf + d1, Dyf + d2, lambda/beta)

    (c) Обновление двойственных переменных

    Parameters
    ----------
    g : np.ndarray
        Размытое изображение (H x W), float64, [0, 1].
    h : np.ndarray
        Финальная оценка ядра PSF (kh x kw).
    alpha : float
        Порядок дробной производной.
    lambda_ftv : float
        Вес дробного TV.
    beta : float
        Штрафной параметр ADMM.
    num_iter : int
        Число ADMM итераций.

    Returns
    -------
    f : np.ndarray
        Восстановленное изображение (H x W), [0, 1].
    """
    H, W = g.shape

    # Предвычисление операторов
    F_Dx, F_Dy, F_Dx_sq, F_Dy_sq = fft_fractional_operators((H, W), alpha)

    H_padded = center_kernel_fft(h, (H, W))
    F_h = fft2(H_padded)
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h) ** 2

    F_g = fft2(g)

    # Инициализация
    f = g.copy()
    u1 = np.zeros((H, W), dtype=np.float64)
    u2 = np.zeros((H, W), dtype=np.float64)
    d1 = np.zeros((H, W), dtype=np.float64)
    d2 = np.zeros((H, W), dtype=np.float64)

    denom = F_h_sq + beta * (F_Dx_sq + F_Dy_sq)

    for _ in range(num_iter):
        # (a) f-обновление
        z1 = u1 - d1
        z2 = u2 - d2

        numer = (F_h_conj * F_g
                 + beta * (np.conj(F_Dx) * fft2(z1) + np.conj(F_Dy) * fft2(z2)))
        f = np.real(ifft2(numer / denom))

        # (b) u-обновление
        Dxf = np.real(ifft2(F_Dx * fft2(f)))
        Dyf = np.real(ifft2(F_Dy * fft2(f)))
        u1, u2 = vector_shrinkage(Dxf + d1, Dyf + d2, lambda_ftv / beta)

        # (c) Двойственные переменные
        d1 = d1 + Dxf - u1
        d2 = d2 + Dyf - u2

    f = np.clip(f, 0.0, 1.0)
    return f


# ─────────────────────────────────────────────────────────────────────────────
#  4. Основной алгоритм: coarse-to-fine слепая деконволюция
# ─────────────────────────────────────────────────────────────────────────────

def coarse_to_fine(g: np.ndarray,
                   kernel_size: tuple,
                   alpha: float = 1.5,
                   lambda_ftv: float = 4e-3,
                   mu_pmp: float = 1e-3,
                   gamma_kernel: float = 2.0,
                   patch_size: int = 5,
                   num_scales: int = 5,
                   outer_iter: int = 5,
                   admm_iter: int = 10,
                   beta1: float = 1.0,
                   beta2: float = 1.0,
                   kernel_threshold_ratio: float = 0.05,
                   final_deconv_iter: int = 40,
                   verbose: bool = False) -> tuple:
    """
    Многомасштабная (coarse-to-fine) слепая деконволюция с дробным TV и PMP.

    Алгоритм (Раздел 4 статьи, Algorithm 1):

    1. Построить Гауссову пирамиду g_1, ..., g_S размытого изображения.
    2. Инициализировать ядро h_1 дельта-функцией на грубейшем уровне.
    3. Для каждого масштаба s = 1, ..., S:
       a. Оценить латентное изображение f_s методом ADMM
          (дробный TV + PMP, см. solve_image_admm).
       b. Оценить ядро h_s в градиентной области (см. solve_kernel).
       c. Если s < S, масштабировать h_s к размеру следующего уровня.
    4. Финальная не-слепая деконволюция (дробный TV без PMP).

    Выбор порядка дробной производной alpha:
        alpha in (1, 2)  контролирует баланс между сохранением
        деталей (alpha -> 1, стандартный TV) и подавлением звона
        (alpha -> 2, лапласиан).
        Оптимальное значение alpha ≈ 1.3-1.7 (Раздел 3 статьи).

    Parameters
    ----------
    g : np.ndarray
        Размытое изображение (H x W), float64, [0, 1].
    kernel_size : tuple of (int, int)
        Ожидаемый размер ядра (kh, kw) — должен быть нечётным.
    alpha : float
        Порядок дробной производной (1 < alpha < 2, по умолчанию 1.5).
    lambda_ftv : float
        Вес дробного TV регуляризатора.
    mu_pmp : float
        Вес PMP prior.
    gamma_kernel : float
        Вес регуляризации ядра (Тихонов).
    patch_size : int
        Размер патча для PMP.
    num_scales : int
        Число уровней пирамиды.
    outer_iter : int
        Число чередующихся итераций (f, h) на каждом масштабе.
    admm_iter : int
        Число ADMM итераций для оценки изображения.
    beta1 : float
        Штрафной параметр ADMM для дробного TV.
    beta2 : float
        Штрафной параметр ADMM для PMP.
    kernel_threshold_ratio : float
        Порог для обнуления малых элементов ядра.
    final_deconv_iter : int
        Число итераций финальной не-слепой деконволюции.
    verbose : bool
        Вывод диагностической информации.

    Returns
    -------
    f_final : np.ndarray
        Восстановленное изображение (H x W), [0, 1].
    h_final : np.ndarray
        Оценённое ядро (kh x kw), сумма = 1.
    history : dict
        Словарь с историей оптимизации.
    """
    H, W = g.shape
    kh, kw = kernel_size

    # ── 1. Ограничение числа масштабов ──
    min_dim = min(H, W)
    max_possible_scales = max(int(np.log(min_dim / max(kh, kw)) / np.log(np.sqrt(2))), 1) + 1
    num_scales = min(num_scales, max_possible_scales)

    if verbose:
        print(f"[FO-BID] Image: {H}x{W}, Kernel: {kh}x{kw}, "
              f"Scales: {num_scales}, alpha: {alpha:.2f}")

    # ── 2. Построение пирамиды ──
    pyramid = build_gaussian_pyramid(g, num_scales)

    # ── 3. Инициализация ядра (delta-функция) ──
    h = np.zeros((kh, kw), dtype=np.float64)
    h[kh // 2, kw // 2] = 1.0

    history = {'kernel_diff': [], 'energy': []}

    # ── 4. Цикл по масштабам (от грубого к точному) ──
    for scale_idx, g_s in enumerate(pyramid):
        H_s, W_s = g_s.shape

        # Масштабирование ядра к текущему уровню
        scale_factor_h = H_s / H
        scale_factor_w = W_s / W
        kh_s = max(int(np.round(kh * scale_factor_h)), 3)
        kw_s = max(int(np.round(kw * scale_factor_w)), 3)
        # Принудительно нечётные размеры
        kh_s = kh_s if kh_s % 2 == 1 else kh_s + 1
        kw_s = kw_s if kw_s % 2 == 1 else kw_s + 1

        h_s = resize_kernel(h, (kh_s, kw_s))

        # Edge tapering
        g_tapered = edgetaper(g_s, h_s)

        # Предвычисление FFT операторов дробных производных
        fft_ops = {}
        F_Dx, F_Dy, F_Dx_sq, F_Dy_sq = fft_fractional_operators(
            (H_s, W_s), alpha)
        fft_ops['F_Dx'] = F_Dx
        fft_ops['F_Dy'] = F_Dy
        fft_ops['F_Dx_sq'] = F_Dx_sq
        fft_ops['F_Dy_sq'] = F_Dy_sq

        # Начальное приближение изображения
        f_s = g_tapered.copy()

        # Адаптивные параметры регуляризации по масштабу
        # На грубых масштабах — более сильная регуляризация
        scale_weight = (num_scales - scale_idx) / num_scales
        lambda_s = lambda_ftv * (1.0 + scale_weight)
        mu_s = mu_pmp * (1.0 + 0.5 * scale_weight)
        gamma_s = gamma_kernel * (1.0 + 2.0 * scale_weight)

        if verbose:
            print(f"  Scale {scale_idx + 1}/{num_scales}: "
                  f"{H_s}x{W_s}, kernel {kh_s}x{kw_s}, "
                  f"lambda={lambda_s:.4f}, mu={mu_s:.4f}, gamma={gamma_s:.2f}")

        # ── Чередующиеся итерации на текущем масштабе ──
        for it in range(outer_iter):
            h_prev = h_s.copy()

            # (A) Оценка изображения f при фиксированном h
            f_s = solve_image_admm(
                g_tapered, h_s, f_s,
                alpha=alpha,
                lambda_ftv=lambda_s,
                mu_pmp=mu_s,
                patch_size=patch_size,
                beta1=beta1,
                beta2=beta2,
                num_iter=admm_iter,
                fft_ops=fft_ops
            )

            # (B) Оценка ядра h при фиксированном f
            h_s = solve_kernel(
                g_tapered, f_s,
                kernel_size=(kh_s, kw_s),
                gamma=gamma_s,
                threshold_ratio=kernel_threshold_ratio
            )

            # Мониторинг сходимости
            hdiff = np.linalg.norm(h_s - h_prev) / max(np.linalg.norm(h_prev), 1e-10)
            history['kernel_diff'].append(hdiff)

            if verbose:
                energy = 0.5 * np.sum((np.real(ifft2(
                    fft2(center_kernel_fft(h_s, (H_s, W_s))) * fft2(f_s)))
                    - g_tapered) ** 2)
                history['energy'].append(energy)
                print(f"    Iter {it + 1}/{outer_iter}: "
                      f"dh={hdiff:.6f}, E={energy:.4f}")

            # Критерий ранней остановки
            if hdiff < 1e-4 and it > 1:
                if verbose:
                    print(f"    Converged at iter {it + 1}")
                break

        # Сохранение текущей оценки ядра (масштабированной к полному размеру)
        h = resize_kernel(h_s, (kh, kw))

    # ── 5. Финальная не-слепая деконволюция ──
    if verbose:
        print(f"  Final non-blind deconvolution (iter={final_deconv_iter})")

    g_tapered_final = edgetaper(g, h)
    f_final = final_nonblind_deconv(
        g_tapered_final, h,
        alpha=alpha,
        lambda_ftv=lambda_ftv * 0.5,  # Уменьшенная регуляризация для финала
        beta=beta1 * 2.0,
        num_iter=final_deconv_iter
    )

    return f_final, h, history
