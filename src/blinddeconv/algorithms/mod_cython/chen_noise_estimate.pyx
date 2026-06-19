"""
chen_noise_estimate.py

Оценка уровня аддитивного белого гауссовского шума (AWGN) на изображении 
с использованием анализа главных компонент (PCA) перекрывающихся блоков.

Основано на методе:
    Chen G., Zhu F., Heng P.A.:
    "An Efficient Statistical Method for Image Noise Level Estimation",
    ICCV 2015.
"""

import numpy as np

__all__ = ['estimate_noise_level']


def _im2patch(im, pch_size, stride=1):
    """
    Извлечение перекрывающихся блоков (патчей) из тензора изображения 
    методом скользящего окна.

    Параметры
    ---------
    im : ndarray
        Тензор изображения размерности (C, H, W).
    pch_size : int
        Размер извлекаемого квадратного блока (ширина и высота).
    stride : int, по умолчанию 1
        Шаг смещения пространственного окна.

    Возвращает
    ----------
    patches : ndarray
        Массив извлеченных блоков размерности (C, pch_size, pch_size, num_patches).
    """
    pch_H = pch_W = int(pch_size)
    stride_H = stride_W = int(stride)

    C, H, W = im.shape
    num_H = len(range(0, H - pch_H + 1, stride_H))
    num_W = len(range(0, W - pch_W + 1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H * pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H - pch_H + ii + 1:stride_H,
                       jj:W - pch_W + jj + 1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))


def estimate_noise_level(image, pch_size=8):
    """
    Оценка среднеквадратичного отклонения (СКО, sigma) аддитивного 
    белого гауссовского шума по единственному изображению.

    Метод формирует набор перекрывающихся блоков, вычисляет их выборочную 
    матрицу ковариации и анализирует ее собственные значения (PCA). 
    Наибольшие собственные значения (энергия полезного сигнала) последовательно 
    отбрасываются. Уровень шума определяется по подмножеству наименьших 
    собственных значений с использованием медианного критерия остановки.

    Параметры
    ---------
    image : ndarray
        Входное изображение: полутоновое размерности (H, W) или цветное (H, W, C).
        Поддерживаются диапазоны значений float [0, 1] и uint8 [0, 255] 
        (масштаб определяется автоматически по максимуму массива).
    pch_size : int, по умолчанию 8
        Размер стороны квадратного блока для анализа.

    Возвращает
    ----------
    sigma : float
        Оценка СКО шума (sigma) в нормализованном масштабе [0, 1]. 
        Для получения значения в пиксельном масштабе результат необходимо 
        умножить на 255. Если оценка не удалась (например, из-за недостатка 
        данных), возвращается 0.0.
    """
    im = np.asarray(image, dtype=np.float64)

    if im.max() > 1.0:
        im = im / 255.0

    if im.ndim == 3:
        im = im.transpose((2, 0, 1))
    elif im.ndim == 2:
        im = im[np.newaxis, :, :]
    else:
        raise ValueError(f"Expected 2D or 3D image, got ndim={im.ndim}")

    pch = _im2patch(im, pch_size, stride=3)
    num_pch = pch.shape[3]
    pch = pch.reshape((-1, num_pch))
    d = pch.shape[0]

    if num_pch < d:
        return 0.0

    mu = pch.mean(axis=1, keepdims=True)
    X = pch - mu
    sigma_X = X @ X.T / num_pch

    sig_values, _ = np.linalg.eigh(sigma_X)
    sig_values.sort()

    for ii in range(-1, -d - 1, -1):
        subset = sig_values[:ii]
        if len(subset) == 0:
            break
        tau = np.mean(subset)
        if tau < 0:
            break
        if np.sum(subset > tau) == np.sum(subset < tau):
            return float(np.sqrt(max(tau, 0.0)))

    min_eig = float(sig_values[0])
    if min_eig > 0:
        return float(np.sqrt(min_eig))
    return 0.0
