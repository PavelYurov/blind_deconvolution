"""
solvers.py

Основные функции решателей для алгоритма слепой деконволюции 
на основе улучшенной разреженной модели (ESM).

Основано на методе:
    L. Chen, F. Fang, S. Lei, F. Li, G. Zhang: "Enhanced Sparse Model
    for Blind Deblurring", ECCV, 2020.

В отличие от стандартной реализации, данный модуль включает расширенную 
поддержку работы с сильно зашумленными данными:
- Опциональные адаптивные пороги (soft и hard floors) в функциях 
  обновления скрытого изображения и ядра, зависящие от дисперсии шума.
- Возможность внедрения пользовательских пространственных фильтров внутри 
  цикла оценки ядра.
- Предварительное санирование градиентов данных.
- Систему событийной телеметрии для отслеживания метрик сходимости.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import label, zoom, map_coordinates

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    fftconv,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    bilateral_filter,
)



def _circ_diff_x(S: np.ndarray) -> np.ndarray:
    """
    Вычисление прямой циклической разности по оси X (горизонтальный градиент).
    Поддерживает двумерные и трехмерные массивы.
    """
    d = np.diff(S, n=1, axis=1)
    wrap = S[:, 0:1, ...] - S[:, -1:, ...]
    return np.concatenate([d, wrap], axis=1)


def _circ_diff_y(S: np.ndarray) -> np.ndarray:
    """
    Вычисление прямой циклической разности по оси Y (вертикальный градиент).
    """
    d = np.diff(S, n=1, axis=0)
    wrap = S[0:1, :, ...] - S[-1:, :, ...]
    return np.concatenate([d, wrap], axis=0)


def _adjoint_diff_x(h: np.ndarray) -> np.ndarray:
    """
    Вычисление сопряженного оператора (дивергенции) для разности по оси X.
    """
    head = h[:, -1:, ...] - h[:, 0:1, ...]
    rest = -np.diff(h, n=1, axis=1)
    return np.concatenate([head, rest], axis=1)


def _adjoint_diff_y(v: np.ndarray) -> np.ndarray:
    """
    Вычисление сопряженного оператора (дивергенции) для разности по оси Y.
    """
    head = v[-1:, :, ...] - v[0:1, :, ...]
    rest = -np.diff(v, n=1, axis=0)
    return np.concatenate([head, rest], axis=0)


def _fft2_planes(S: np.ndarray) -> np.ndarray:
    """
    Выполнение двумерного быстрого преобразования Фурье по первым двум осям массива.
    """
    if S.ndim == 2:
        return fft2(S)
    return fft2(S, axes=(0, 1))


def _ifft2_planes(F: np.ndarray) -> np.ndarray:
    """
    Выполнение обратного двумерного быстрого преобразования Фурье по первым двум осям массива.
    """
    if F.ndim == 2:
        return ifft2(F)
    return ifft2(F, axes=(0, 1))



def L0Restoration_HS(Im: np.ndarray,
                     kernel: np.ndarray,
                     lambda_data: float,
                     lambda_grad: float,
                     theta: float,
                     kappa: float = 2.0,
                     expert_adapt: dict = None) -> np.ndarray:
    """
    Обновление скрытого изображения для модели ESM (I-подзадача).

    Решает оптимизационную задачу с использованием метода полуквадратичного 
    расщепления (HQS):
        min_I || k * I - B ||_2^2 
              + lambda_grad * (|| grad(I) ||_0 - || grad(I) ||_1)
              + lambda_data * (|| k * grad(I) - grad(B) ||_0 - || k * grad(I) - grad(B) ||_1)

    Параметры
    ---------
    Im : ndarray
        Размытое изображение в формате float64.
    kernel : ndarray
        Ядро размытия.
    lambda_data : float
        Весовой коэффициент разреженности остатков градиентов данных.
    lambda_grad : float
        Весовой коэффициент разреженности градиентов изображения.
    theta : float
        Параметр улучшенной модели L0-L1.
    kappa : float, по умолчанию 2.0
        Фактор геометрического роста для параметров расщепления.
    expert_adapt : dict или None, опционально
        Словарь экспертных параметров для адаптации к шуму. Позволяет задать 
        минимальные пороги (soft и hard floors) для операций сжатия на основе 
        оцененного уровня шума sigma, а также переопределить параметры kappa и betamax.

    Возвращает
    ----------
    S : ndarray
        Восстановленное скрытое изображение, обрезанное до исходного размера.
    """
    sigma_n = 0.0
    qf = 0.0; gf = 0.0; qsf = 0.0; gsf = 0.0
    if expert_adapt:
        sigma_n = float(expert_adapt.get('sigma', 0.0) or 0.0)
        qf = float(expert_adapt.get('q_floor_factor', 0.0) or 0.0)
        gf = float(expert_adapt.get('g_floor_factor', 0.0) or 0.0)
        qsf = float(expert_adapt.get('q_soft_floor_factor', 0.0) or 0.0)
        gsf = float(expert_adapt.get('g_soft_floor_factor', 0.0) or 0.0)
        if 'kappa' in expert_adapt and expert_adapt['kappa']:
            kappa = float(expert_adapt['kappa'])
    sigma_grad2 = 2.0 * sigma_n * sigma_n   
    q_hard_floor = qf * sigma_grad2         
    g_hard_floor = gf * sigma_grad2
    q_soft_floor = qsf * float(np.sqrt(sigma_grad2)) if sigma_grad2 > 0 else 0.0
    g_soft_floor = gsf * float(np.sqrt(sigma_grad2)) if sigma_grad2 > 0 else 0.0

    H, W = Im.shape[:2]

    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    Im = wrap_boundary_liu(Im, tuple(target))

    S = Im.copy()
    betamax = 1e5
    if expert_adapt and expert_adapt.get('betamax'):
        betamax = float(expert_adapt['betamax'])

    fx = np.array([[1.0, -1.0]], dtype=np.float64)
    fy = np.array([[1.0], [-1.0]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
        KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
        Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))
        otfFx_b = np.tile(otfFx[:, :, np.newaxis], (1, 1, D))
        otfFy_b = np.tile(otfFy[:, :, np.newaxis], (1, 1, D))
    else:
        otfFx_b = otfFx
        otfFy_b = otfFy

    Normin1 = np.conj(KER) * _fft2_planes(S)

    beta1 = 2.0 * lambda_grad
    tau1 = 2.0 * lambda_data

    B_h = _circ_diff_x(Im)
    B_v = _circ_diff_y(Im)

    KG_h = otfFx_b * KER
    KG_v = otfFy_b * KER
    KG = np.abs(KG_h) ** 2 + np.abs(KG_v) ** 2

    while beta1 < betamax:
        Denormin = Den_KER + beta1 * Denormin2 + tau1 * KG

        S_h = _circ_diff_x(S)
        S_v = _circ_diff_y(S)

        q_soft = lambda_data * theta / (2.0 * tau1)
        q_hard = lambda_data / tau1
        if q_soft_floor > 0.0 and q_soft_floor > q_soft:
            q_soft = q_soft_floor
        if q_hard_floor > 0.0 and q_hard_floor > q_hard:
            q_hard = q_hard_floor

        q_h = B_h - fftconv(S_h, kernel)
        q_h = np.sign(q_h) * np.maximum(np.abs(q_h) - q_soft, 0.0)
        q_v = B_v - fftconv(S_v, kernel)
        q_v = np.sign(q_v) * np.maximum(np.abs(q_v) - q_soft, 0.0)
        t_h = q_h ** 2 < q_hard
        t_v = q_v ** 2 < q_hard
        q_h[t_h] = 0.0
        q_v[t_v] = 0.0

        g_soft = lambda_grad * theta / (2.0 * beta1)
        g_hard = lambda_grad / beta1
        if g_soft_floor > 0.0 and g_soft_floor > g_soft:
            g_soft = g_soft_floor
        if g_hard_floor > 0.0 and g_hard_floor > g_hard:
            g_hard = g_hard_floor

        g_h = S_h.copy()
        g_v = S_v.copy()
        g_h = np.sign(g_h) * np.maximum(np.abs(g_h) - g_soft, 0.0)
        g_v = np.sign(g_v) * np.maximum(np.abs(g_v) - g_soft, 0.0)
        t_h = g_h ** 2 < g_hard
        t_v = g_v ** 2 < g_hard
        g_h[t_h] = 0.0
        g_v[t_v] = 0.0

        Normin2 = _adjoint_diff_x(g_h) + _adjoint_diff_y(g_v)
        Normin3 = np.conj(KG_h) * _fft2_planes(B_h - q_h) \
                + np.conj(KG_v) * _fft2_planes(B_v - q_v)

        FS = (Normin1 + beta1 * _fft2_planes(Normin2) + tau1 * Normin3) \
             / Denormin
        S = np.real(_ifft2_planes(FS))

        beta1 = beta1 * kappa
        tau1 = tau1 * kappa

    return S[:H, :W, ...]



def L0Restoration(Im: np.ndarray,
                  kernel: np.ndarray,
                  lambda_grad: float,
                  kappa: float = 2.0) -> np.ndarray:
    """
    Неслепое восстановление изображения с использованием классической 
    L0-нормы градиентов. Используется на этапе подавления артефактов звона.

    Параметры
    ---------
    Im : ndarray
        Входное размытое изображение.
    kernel : ndarray
        Ядро размытия.
    lambda_grad : float
        Весовой коэффициент L0-регуляризации градиентов.
    kappa : float, по умолчанию 2.0
        Фактор роста штрафного параметра.

    Возвращает
    ----------
    S : ndarray
        Восстановленное изображение исходного размера.
    """
    H, W = Im.shape[:2]

    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    Im = wrap_boundary_liu(Im, tuple(target))

    S = Im.copy()
    betamax = 1e5

    fx = np.array([[1.0, -1.0]], dtype=np.float64)
    fy = np.array([[1.0], [-1.0]], dtype=np.float64)

    if S.ndim == 2:
        N, M = S.shape
        D = 1
    else:
        N, M, D = S.shape

    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    if D > 1:
        Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
        KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
        Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * _fft2_planes(S)

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        h = _circ_diff_x(S)
        v = _circ_diff_y(S)

        if D == 1:
            t = (h ** 2 + v ** 2) < lambda_grad / beta
        else:
            t = np.sum(h ** 2 + v ** 2, axis=2) < lambda_grad / beta
            t = np.tile(t[:, :, np.newaxis], (1, 1, D))

        h[t] = 0.0
        v[t] = 0.0

        Normin2 = _adjoint_diff_x(h) + _adjoint_diff_y(v)

        FS = (Normin1 + beta * _fft2_planes(Normin2)) / Denormin
        S = np.real(_ifft2_planes(FS))

        beta = beta * kappa

    return S[:H, :W, ...]



def _compute_Ax_psf(x: np.ndarray, p: dict) -> np.ndarray:
    """
    Вычисление произведения матрицы на вектор для решения подзадачи оценки 
    ядра методом сопряженных градиентов.
    """
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y


def estimate_psf_l0(blurred_x: np.ndarray,
                    blurred_y: np.ndarray,
                    latent_x: np.ndarray,
                    latent_y: np.ndarray,
                    weight: float,
                    tau: float,
                    k_prev: np.ndarray,
                    theta: float,
                    expert_adapt: dict = None) -> np.ndarray:
    """
    Обновление ядра размытия (k-подзадача) для модели ESM.

    Решает систему линейных уравнений методом сопряженных градиентов (CG) 
    в частотной области с учетом L0-L1 регуляризации остатков градиентов данных.
    Включает поддержку экспертной адаптации порогов сжатия к уровню шума.

    Параметры
    ---------
    blurred_x, blurred_y : ndarray
        Производные размытого изображения по осям X и Y.
    latent_x, latent_y : ndarray
        Производные скрытого изображения по осям X и Y.
    weight : float
        Весовой коэффициент регуляризации Тихонова для ядра.
    tau : float
        Параметр lambda_data (вес регуляризатора остатков).
    k_prev : ndarray
        Предыдущая оценка ядра (начальное приближение для метода сопряженных градиентов).
    theta : float
        Параметр улучшенной модели L0-L1.
    expert_adapt : dict или None, опционально
        Словарь экспертных параметров для адаптации мягких и жестких порогов сжатия 
        к уровню шума.

    Возвращает
    ----------
    psf : ndarray
        Обновленное ядро размытия, прошедшее пороговую фильтрацию и нормализацию.
    """
    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    psf_size = k_prev.shape
    tau1 = 2.0 * tau
    psf = k_prev.copy()
    iter_max = 5

    g_soft_floor = 0.0
    g_hard_floor = 0.0
    if expert_adapt:
        sigma_n = float(expert_adapt.get('sigma', 0.0) or 0.0)
        gsf = float(expert_adapt.get('g_soft_floor_factor', 0.0) or 0.0)
        gf = float(expert_adapt.get('g_floor_factor', 0.0) or 0.0)
        sigma_grad2 = 2.0 * sigma_n * sigma_n
        g_hard_floor = gf * sigma_grad2
        if sigma_grad2 > 0:
            g_soft_floor = gsf * float(np.sqrt(sigma_grad2))

    for _ in range(iter_max):
        g_soft = tau * theta / (2.0 * tau1)
        g_hard = tau / tau1
        if g_soft_floor > 0.0 and g_soft_floor > g_soft:
            g_soft = g_soft_floor
        if g_hard_floor > 0.0 and g_hard_floor > g_hard:
            g_hard = g_hard_floor

        g_h = blurred_x - fftconv(latent_x, psf)
        g_h = np.sign(g_h) * np.maximum(np.abs(g_h) - g_soft, 0.0)
        g_v = blurred_y - fftconv(latent_y, psf)
        g_v = np.sign(g_v) * np.maximum(np.abs(g_v) - g_soft, 0.0)
        t_h = g_h ** 2 < g_hard
        t_v = g_v ** 2 < g_hard
        g_h[t_h] = 0.0
        g_v[t_v] = 0.0

        temp = np.conj(latent_xf) * fft2(blurred_x - g_h) \
             + np.conj(latent_yf) * fft2(blurred_y - g_v)
        b_f = tau1 * temp + np.conj(latent_xf) * blurred_xf \
                          + np.conj(latent_yf) * blurred_yf
        b = np.real(otf2psf(b_f, psf_size))

        p = {
            'm': (np.conj(latent_xf) * latent_xf
                  + np.conj(latent_yf) * latent_yf) * (1.0 + tau1),
            'img_size': blurred_xf.shape[:2],
            'psf_size': psf_size,
            'lambda': weight,
        }
        psf = conjgrad(psf, b, 8, 1e-5, _compute_Ax_psf, p)

        tau1 = tau1 * 2.0

    max_val = psf.max()
    if max_val > 0:
        psf[psf < max_val * 0.05] = 0.0
        s = psf.sum()
        if s > 0:
            psf = psf / s
    return psf



def blind_deconv_main(blur_B: np.ndarray,
                      k: np.ndarray,
                      lambda_data: float,
                      lambda_grad: float,
                      threshold: float,
                      opts: dict,
                      blind_denoise_fn=None,
                      progress_callback=None,
                      scale_index=None):
    """
    Одна итерация (цикл для одного масштаба) слепой деконволюции ESM.

    Поочередно обновляет скрытое изображение, выполняет адаптивное 
    пороговое ограничение градиентов, обновляет ядро и фильтрует 
    изолированные компоненты ядра.
    Модифицированная версия включает поддержку предварительного санирования 
    градиентов, промежуточной пространственной фильтрации скрытого 
    изображения и передачи метрик через функцию обратного вызова.

    Параметры
    ---------
    blur_B : ndarray
        Размытое изображение.
    k : ndarray
        Текущая оценка ядра размытия.
    lambda_data : float
        Весовой коэффициент регуляризации остатков данных.
    lambda_grad : float
        Весовой коэффициент регуляризации градиентов.
    threshold : float
        Адаптивный порог для ограничения градиентов.
    opts : dict
        Словарь с параметрами оптимизации (xk_iter, theta, expert_adapt).
    blind_denoise_fn : callable или None, опционально
        Пользовательская функция пространственной фильтрации промежуточного 
        скрытого изображения S.
    progress_callback : callable или None, опционально
        Функция для логирования метрик и визуализации ядра на каждой итерации.
    scale_index : int или None, опционально
        Индекс текущего масштаба в пирамиде.

    Возвращает
    ----------
    k : ndarray
        Обновленное ядро размытия.
    lambda_data, lambda_grad : float
        Обновленные (уменьшенные) параметры регуляризации.
    S : ndarray
        Текущая оценка скрытого изображения.
    """
    dx = np.array([[-1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    dy = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float64)

    H, W = blur_B.shape[:2]
    target = opt_fft_size(np.array([H, W]) + np.array(k.shape[:2]) - 1)
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    theta = opts['theta']
    xk_iter = opts['xk_iter']
    expert_adapt = opts.get('expert_adapt', None)

    if expert_adapt and expert_adapt.get('sanitize_grad_fn') is not None:
        try:
            Bx = expert_adapt['sanitize_grad_fn'](Bx)
            By = expert_adapt['sanitize_grad_fn'](By)
        except Exception:
            pass

    S = blur_B.copy()
    for _it in range(xk_iter):
        S = L0Restoration_HS(blur_B, k, lambda_data, lambda_grad, theta,
                             expert_adapt=expert_adapt)

        if blind_denoise_fn is not None:
            try:
                S_dn = blind_denoise_fn(S)
                if S_dn is not None and S_dn.shape == S.shape:
                    S = S_dn
            except Exception:
                pass

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, max(k.shape), threshold
        )

        k_prev = k.copy()
        k = estimate_psf_l0(Bx, By, latent_x, latent_y,
                            2.0, lambda_data, k, theta,
                            expert_adapt=expert_adapt)

        structure = np.ones((3, 3), dtype=np.int32)  
        labeled, n_comp = label(k > 0, structure=structure)
        for ii in range(1, n_comp + 1):
            mask = labeled == ii
            currsum = k[mask].sum()
            if currsum < 0.1:
                k[mask] = 0.0
        k[k < 0] = 0.0
        s = k.sum()
        if s > 0:
            k = k / s
.
        if progress_callback is not None:
            try:
                kdiff = float(np.linalg.norm(k - k_prev))
                progress_callback({
                    'event': 'iter',
                    'scale': scale_index,
                    'iter': _it,
                    'kernel': k.copy(),
                    'kernel_diff': kdiff,
                    'lambda_data': float(lambda_data),
                    'lambda_grad': float(lambda_grad),
                    'kernel_max': float(k.max()),
                    'kernel_sum': float(k.sum()),
                    'kernel_sparsity': float((k > 0).mean()),
                })
            except Exception:
                pass

        if lambda_data != 0:
            lambda_data = max(lambda_data / 1.1, 1e-4)
        else:
            lambda_data = 0.0
        if lambda_grad != 0:
            lambda_grad = max(lambda_grad / 1.1, 1e-4)
        else:
            lambda_grad = 0.0

    k[k < 0] = 0.0
    s = k.sum()
    if s > 0:
        k = k / s
    return k, lambda_data, lambda_grad, S



def _init_kernel(minsize: int) -> np.ndarray:
    """
    Инициализация ядра на самом грубом уровне пирамиды.
    Создает ядро из нулей, помещая в его геометрический центр два элемента со значением 0.5.
    """
    k = np.zeros((minsize, minsize), dtype=np.float64)
    a = (minsize - 1) // 2  
    row = a - 1
    col_start = a - 1
    col_end = a  
    k[row, col_start:col_end + 1] = 0.5
    return k


def _downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:
    """
    Понижающее масштабирование изображения с применением антиалиасингового 
    гауссовского фильтра перед билинейной интерполяцией.
    """
    if ret == 1.0:
        return I

    sig = ret / np.pi
    g0 = np.arange(-50, 51) * 2 * np.pi
    sf = np.exp(-0.5 * g0 * g0 * sig * sig)
    sf = sf / sf.sum()

    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]
    kern_row = sf.reshape(1, -1)      
    kern_col = sf.reshape(-1, 1)        
    kern = kern_col @ kern_row           
    Ic = convolve2d(I, kern, mode='valid')

    Hc, Wc = Ic.shape
    gx = np.arange(1.0, Wc + 1e-12, 1.0 / ret)
    gy = np.arange(1.0, Hc + 1e-12, 1.0 / ret)
    gx0 = gx - 1.0
    gy0 = gy - 1.0
    GX, GY = np.meshgrid(gx0, gy0)
    sI = map_coordinates(Ic, [GY.ravel(), GX.ravel()],
                         order=1, mode='nearest')
    return sI.reshape(gy0.size, gx0.size)


def _fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:
    """
    Корректировка пространственных размеров ядра до целевых значений 
    путем добавления или удаления краевых строк/столбцов со стороны 
    с наименьшей суммой элементов.
    """
    k1, k2 = f.shape
    while (k1 != nk1) or (k2 != nk2):
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]
        if k1 < nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[:k1, :] = f
                f = tf
            else:
                tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
                tf[1:k1 + 1, :] = f
                f = tf
        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]
        if k2 < nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, :k2] = f
                f = tf
            else:
                tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
                tf[:, 1:k2 + 1] = f
                f = tf
        k1, k2 = f.shape
    return f


def _resizeKer(k: np.ndarray, ret_inv: float, k1: int, k2: int) -> np.ndarray:
    """
    Повышение разрешения ядра размытия методом кубической интерполяции 
    с ограничением снизу нулем и последующей нормализацией.
    """
    k = zoom(k, ret_inv, order=3, mode='nearest')
    k = np.maximum(k, 0.0)
    k = _fixsize(k, k1, k2)
    m = k.max()
    if m > 0:
        k = k / k.sum()
    return k


def blind_deconv(y: np.ndarray,
                 lambda_data: float,
                 lambda_grad: float,
                 opts: dict,
                 blind_denoise_fn=None,
                 progress_callback=None):
    """
    Многомасштабная слепая деконволюция на основе модели ESM.

    Параметры
    ---------
    y : ndarray
        Входное полутоновое изображение в диапазоне [0, 1].
    lambda_data : float
        Весовая константа для регуляризации остатков данных.
    lambda_grad : float
        Весовая константа для регуляризации градиентов.
    opts : dict
        Словарь параметров конфигурации (kernel_size, gamma_correct, 
        xk_iter, k_thresh, theta, expert_adapt).
    blind_denoise_fn : callable или None, опционально
        Пользовательский фильтр, вызываемый на скрытом изображении перед обновлением ядра.
    progress_callback : callable или None, опционально
        Функция для сбора метрик и фиксации состояния ядра.

    Возвращает
    ----------
    kernel : ndarray
        Финальная оценка ядра размытия.
    interim_latent : ndarray
        Промежуточное скрытое изображение на наивысшем разрешении.
    """
    if opts.get('gamma_correct', 1.0) != 1.0:
        y = y ** opts['gamma_correct']

    ret = np.sqrt(0.5)
    kernel_size = opts['kernel_size']
    k_thresh = opts.get('k_thresh', 20)
    opts_with_theta = dict(opts)
    if 'theta' not in opts_with_theta:
        opts_with_theta['theta'] = 1.0

    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1

    retv = ret ** np.arange(0, maxitr + 1)  
    k1list = np.ceil(kernel_size * retv).astype(int)
    k1list[k1list % 2 == 0] += 1  

    ks = None
    threshold = 0.0
    interim_latent = None

    for s in range(num_scales - 1, -1, -1):
        k1 = int(k1list[s])
        k2 = k1
        cret = retv[s]

        if s == num_scales - 1:
            ks = _init_kernel(k1)
        else:
            ks = _resizeKer(ks, 1.0 / ret, k1, k2)

        ys = _downSmpImC(y, cret)

        print(f'Processing scale {s + 1}/{num_scales}; '
              f'kernel size {k1}x{k2}; image size {ys.shape[0]}x{ys.shape[1]}',
              flush=True)

        if s == num_scales - 1:
            _, _, threshold = threshold_pxpy_v1(ys, max(ks.shape))

        if progress_callback is not None:
            try:
                progress_callback({
                    'event': 'scale_start',
                    'scale': s,
                    'num_scales': num_scales,
                    'kernel_shape': tuple(ks.shape),
                    'image_shape': tuple(ys.shape),
                })
            except Exception:
                pass

        ks, lambda_data, lambda_grad, interim_latent = blind_deconv_main(
            ys, ks, lambda_data, lambda_grad, threshold, opts_with_theta,
            blind_denoise_fn=blind_denoise_fn,
            progress_callback=progress_callback,
            scale_index=s,
        )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        sk = ks.sum()
        if sk > 0:
            ks = ks / sk

        if progress_callback is not None:
            try:
                progress_callback({
                    'event': 'scale_end',
                    'scale': s,
                    'kernel': ks.copy(),
                })
            except Exception:
                pass

        if s == 0:
            kernel = ks
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            ssum = kernel.sum()
            if ssum > 0:
                kernel = kernel / ssum
            return kernel, interim_latent

    return ks, interim_latent



def _computeDenominator(y: np.ndarray, k: np.ndarray):
    """
    Предварительное вычисление спектральных компонентов знаменателя 
    для метода расщепления Брэгмана (ADM).
    """
    sizey = y.shape
    otfk = psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = np.abs(psf2otf(np.array([[1.0, -1.0]]), sizey)) ** 2 \
           + np.abs(psf2otf(np.array([[1.0], [-1.0]]), sizey)) ** 2
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B: np.ndarray,
                         k: np.ndarray,
                         lambda_tv: float,
                         alpha: float = 1.0) -> np.ndarray:
    """
    Анизотропная TV-деконволюция методом ADM.

    Параметры
    ---------
    B : ndarray
        Размытое изображение.
    k : ndarray
        Ядро размытия.
    lambda_tv : float
        Вес TV-регуляризации.
    alpha : float, по умолчанию 1.0
        Экспонента нормы. Поддерживается только 1.0.

    Возвращает
    ----------
    I : ndarray
        Восстановленное изображение.
    """
    if alpha != 1.0:
        raise NotImplementedError(
            "deblurring_adm_aniso is only ported for alpha = 1 "
            "(the sole branch used by the ESM pipeline)."
        )

    if (k.shape[0] % 2 != 1) or (k.shape[1] % 2 != 1):
        raise ValueError('Blur kernel k must be odd-sized.')

    beta = 1.0 / lambda_tv
    beta_min = 0.001

    m, n = B.shape
    I = B.copy()

    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = _circ_diff_x(I)
    Iy = _circ_diff_y(I)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
        Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)

        Wxx = _adjoint_diff_x(Wx) + _adjoint_diff_y(Wy)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = _circ_diff_x(I)
        Iy = _circ_diff_y(I)

        beta = beta / 2.0

    return I



def ringing_artifacts_removal(y: np.ndarray,
                              kernel: np.ndarray,
                              lambda_tv: float,
                              lambda_l0: float,
                              weight_ring: float) -> np.ndarray:
    """
    Финальная неслепая деконволюция с подавлением артефактов звона.

    Используется комбинация TV-l2 метода и фильтрации высокочастотных 
    структур (вычитание L0-оценки с последующим билатеральным сглаживанием).

    Параметры
    ---------
    y : ndarray
        Размытое изображение.
    kernel : ndarray
        Функция рассеяния точки.
    lambda_tv : float
        Весовой коэффициент TV-регуляризации.
    lambda_l0 : float
        Весовой коэффициент L0-регуляризации градиентов.
    weight_ring : float
        Вес вычитания высокочастотных артефактов звона.

    Возвращает
    ----------
    result : ndarray
        Восстановленное изображение без эффекта звона.
    """
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
        was_2d = True
    else:
        was_2d = False

    H, W, Ch = y.shape

    target = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    y_pad = wrap_boundary_liu(y, tuple(target))
    if y_pad.ndim == 2:
        y_pad = y_pad[:, :, np.newaxis]

    Latent_tv = np.zeros_like(y_pad)
    for c in range(Ch):
        Latent_tv[:, :, c] = deblurring_adm_aniso(
            y_pad[:, :, c], kernel, lambda_tv, 1.0
        )
    Latent_tv = Latent_tv[:H, :W, :]

    if weight_ring == 0:
        result = Latent_tv
        if was_2d:
            return result[:, :, 0]
        return result

    if y_pad.shape[2] == 1:
        Latent_l0 = L0Restoration(y_pad[:, :, 0], kernel, lambda_l0, 2.0)
        Latent_l0 = Latent_l0[:H, :W]
        Latent_l0 = Latent_l0[:, :, np.newaxis]
    else:
        Latent_l0 = L0Restoration(y_pad, kernel, lambda_l0, 2.0)
        Latent_l0 = Latent_l0[:H, :W, :]

    diff = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff, 3.0, 0.1)
    bf_diff = np.asarray(bf_diff, dtype=np.float64)
    if bf_diff.ndim == 2 and diff.ndim == 3:
        bf_diff = bf_diff[:, :, np.newaxis]

    result = Latent_tv - weight_ring * bf_diff

    if was_2d:
        return result[:, :, 0]
    return result
