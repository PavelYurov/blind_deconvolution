"""
utils.py

Вспомогательные функции для BID-HBSP: Байесовской слепой деконволюции
изображений с априорным распределением гиперболического секанса.

Содержит:
    - Утилиты для вычисления сверток на основе FFT (psf2otf, otf2psf)
    - Операторы пространственных градиентов и их сопряженные аналоги
    - Вычисление весов для априорного распределения гиперболического секанса (Gaussian Scale Mixture)
    - Проекция, пороговая обработка и инициализация ядра

Литература:
[1] Castro-Macias, Perez-Bueno, et al. (2024), "Bayesian Blind Image
    Deconvolution using a Hyperbolic-Secant prior", ICIP 2024.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple

EPSILON = 1e-12



def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Перевод функции рассеяния точки (PSF) в оптическую передаточную функцию (OTF).

    PSF дополняется нулями до целевого размера shape и циклически сдвигается так,
    чтобы центр ядра находился по индексу (0, 0) перед применением двумерного БПФ.

    Параметры
    ----------
    psf : ndarray, форма (kh, kw)
        Функция рассеяния точки (ядро размытия).
    shape : tuple (H, W)
        Целевые пространственные размеры OTF.

    Возвращает
    -------
    otf : ndarray, форма (H, W), комплексный
        Оптическая передаточная функция.
    """
    kh, kw = psf.shape
    padded = np.zeros(shape, dtype=psf.dtype)
    padded[:kh, :kw] = psf
    # Центрирование ядра в начале координат для корректной фазы
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return fft2(padded)


def otf2psf(otf: np.ndarray, kernel_shape: Tuple[int, int]) -> np.ndarray:
    """Восстановление пространственной PSF из ее OTF с помощью обратного БПФ и обрезки.

    Параметры
    ----------
    otf : ndarray, форма (H, W), комплексный
    kernel_shape : (kh, kw)

    Возвращает
    -------
    psf : ndarray, форма (kh, kw), вещественный
    """
    kh, kw = kernel_shape
    psf_full = np.real(ifft2(otf))
    psf_full = np.roll(psf_full, kh // 2, axis=0)
    psf_full = np.roll(psf_full, kw // 2, axis=1)
    return psf_full[:kh, :kw]


def precompute_gradient_operators(
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Предварительное вычисление частотных (DFT) представлений операторов первых конечных разностей.

    Прямые разности с периодическими (круговыми) границами:

        (C_x * u)[i, j] = u[i, j+1] - u[i, j]      (горизонтальная)
        (C_y * u)[i, j] = u[i+1, j] - u[i, j]      (вертикальная)

    Возвращает
    -------
    F_dx : ndarray, комплексный - DFT горизонтального ядра разностей
    F_dy : ndarray, комплексный - DFT вертикального ядра разностей
    F_grad_sq : ndarray, вещественный - |F_dx|^2 + |F_dy|^2 (спектр лапласиана)
    """
    H, W = shape

    dx = np.zeros(shape)
    dx[0, 0] = -1
    dx[0, 1] = 1

    dy = np.zeros(shape)
    dy[0, 0] = -1
    dy[1, 0] = 1

    F_dx = fft2(dx)
    F_dy = fft2(dy)
    F_grad_sq = np.abs(F_dx) ** 2 + np.abs(F_dy) ** 2
    return F_dx, F_dy, F_grad_sq


def forward_diff_x(u: np.ndarray) -> np.ndarray:
    """Горизонтальная прямая разность: (C_x * u)[i,j] = u[i, j+1] - u[i, j]."""
    return np.roll(u, -1, axis=1) - u


def forward_diff_y(u: np.ndarray) -> np.ndarray:
    """Вертикальная прямая разность: (C_y * u)[i,j] = u[i+1, j] - u[i, j]."""
    return np.roll(u, -1, axis=0) - u


def adjoint_diff_x(v: np.ndarray) -> np.ndarray:
    """Сопряженный оператор горизонтальной прямой разности.

    (C_x^T * v)[i,j] = v[i, j-1] - v[i, j]
    """
    return np.roll(v, 1, axis=1) - v


def adjoint_diff_y(v: np.ndarray) -> np.ndarray:
    """Сопряженный оператор вертикальной прямой разности.

    (C_y^T * v)[i,j] = v[i-1, j] - v[i, j]
    """
    return np.roll(v, 1, axis=0) - v



def compute_hs_weights(
    dx: np.ndarray, 
    dy: np.ndarray, 
    sigma_x: np.ndarray,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет E[w] для априорного распределения гиперболического секанса с использованием вариационной аппроксимации.
    Источник: Castro-Macias et al. (2024), Уравнение (26).
    """
    sigma_grad = 2.0 * sigma_x
    
    # Второй момент E[u^2] = mean^2 + var
    # Нужен аргумент для tanh: sqrt(E[u^2])
    # См. Приложение C или Уравнение (26), где ksi = sqrt(E[x^2])
    
    nu_x = np.sqrt(dx**2 + sigma_grad + EPSILON)
    nu_y = np.sqrt(dy**2 + sigma_grad + EPSILON)
    
    # alpha = 1/b. Формула: (alpha * tanh(alpha * nu)) / nu
    alpha = 1.0 / b
    
    gamma_x = (alpha * np.tanh(alpha * nu_x)) / nu_x
    gamma_y = (alpha * np.tanh(alpha * nu_y)) / nu_y
    
    return gamma_x, gamma_y


def compute_hs_weights_scalar(
    x_n: np.ndarray,
    sigma_sq_n: np.ndarray,
    alpha_n: float,
) -> np.ndarray:
    """Вычисляет веса HS E[omega] для одного отфильтрованного изображения (в пространстве фильтров).

    В вариационной формулировке в пространстве фильтров априорная модель применяется напрямую
    к пикселям отфильтрованного изображения x_n = F_n * x, поэтому обновление весов 
    выполняется по скалярной (попиксельной) формуле без операторов градиента.

        ksi_n_i = sqrt(m_xn^2(i) + Sigma_xn(i,i)), 
        E[omega_n_i] = (alpha_n * tanh(alpha_n * ksi_n_i)) / ksi_n_i.

    Параметры
    ----------
    x_n : ndarray (H, W)
        Апостериорное среднее n-го отфильтрованного изображения m_xn.
    sigma_sq_n : ndarray (H, W)
        Диагональ апостериорной ковариации Sigma_xn(i,i).
    alpha_n : float
        Параметр масштаба HS alpha_n = 1/b.

    Возвращает
    -------
    theta_n : ndarray (H, W)
        Диагональные веса HS E[omega_n_i].

    Источник: Castro-Macias et al. (2024), Уравнение (26).
    """
    xi = np.sqrt(x_n ** 2 + sigma_sq_n + EPSILON)
    theta = (alpha_n * np.tanh(alpha_n * xi)) / xi
    return theta


def project_kernel(h: np.ndarray) -> np.ndarray:
    """Проецирует ядро на вероятностный симплекс h >= 0, sum(h) = 1."""
    h = np.maximum(h, 0.0)
    h_sum = h.sum()
    if h_sum > EPSILON:
        h /= h_sum
    else:
        h = np.ones_like(h) / h.size
    return h


def threshold_kernel(
    h: np.ndarray,
    ratio: float = 0.05
) -> np.ndarray:
    """Пороговое обнуление малых значений ядра (для разреженности) с последующей нормализацией.

    Элементы ниже ratio * max(h) обнуляются.

    Параметры
    ----------
    h : ndarray - ядро (ожидается неотрицательное)
    ratio : float - доля от пикового значения, ниже которой значения обнуляются
    """
    h = np.maximum(h, 0.0)
    h[h < ratio * np.max(h)] = 0.0
    return project_kernel(h)


def init_gaussian_kernel(
    shape: Tuple[int, int],
    sigma: float = None
) -> np.ndarray:
    """Создает гауссовское ядро, нормализованное к единичной сумме.

    Параметры
    ----------
    shape : (kh, kw)
    sigma : float, опционально
        Стандартное отклонение; по умолчанию равно max(kh, kw) / 6.
    """
    kh, kw = shape
    if sigma is None:
        sigma = max(kh, kw) / 6.0
    cy, cx = kh // 2, kw // 2
    y, x = np.ogrid[-cy: kh - cy, -cx: kw - cx]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def fft_convolve(
    x: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Круговая свертка h * x через БПФ.

    Параметры
    x : ndarray (H, W) - изображение
    h : ndarray (kh, kw) - ядро

    Возвращает
    y : ndarray (H, W) - свернутое изображение
    """
    F_h = psf2otf(h, x.shape)
    return np.real(ifft2(F_h * fft2(x)))


from scipy.signal import fftconvolve

def edgetaper(img: np.ndarray, kernel: np.ndarray, n_taper: int = None) -> np.ndarray:
    """
    Сглаживает края изображения для уменьшения артефактов "звона" (ringing) 
    при деконволюции на основе БПФ.
    Симулирует функцию edgetaper из Matlab.
    """
    h, w = img.shape
    kh, kw = kernel.shape
    
    if n_taper is None:
        n_taper = max(kh, kw)
        
    # Создание весов для сглаживания (подобно окну Ханнинга)
    # 1. Горизонтальные
    dx = np.arange(w)
    wx = np.ones(w)
    # Левый край
    wx[dx < n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx < n_taper] - n_taper) / n_taper))
    # Правый край
    wx[dx >= w - n_taper] = 0.5 * (1 + np.cos(np.pi * (dx[dx >= w - n_taper] - (w - n_taper - 1)) / n_taper))
    
    # 2. Вертикальные
    dy = np.arange(h)
    wy = np.ones(h)
    # Верхний край
    wy[dy < n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy < n_taper] - n_taper) / n_taper))
    # Нижний край
    wy[dy >= h - n_taper] = 0.5 * (1 + np.cos(np.pi * (dy[dy >= h - n_taper] - (h - n_taper - 1)) / n_taper))
    
    # 2D веса
    W = np.outer(wy, wx)
    
    # Размытие изображения ядром (для соответствия граничным условиям)
    blurred = fftconvolve(img, kernel, mode='same')
    
    # Смешивание: Центр - это оригинальное изображение, границы - размытая версия
    # Это делает изображение циклически согласованным для БПФ
    return img * W + blurred * (1 - W)