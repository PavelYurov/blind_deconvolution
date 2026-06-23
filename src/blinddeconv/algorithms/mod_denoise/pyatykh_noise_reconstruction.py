"""
pyatykh_noise_reconstruction.py

Оценка параметров пуассоновско-гауссовского шума с использованием метода 
анализа главных компонент (PCA), стабилизирующего дисперсию преобразования 
(VST) и минимизации эксцесса.

Основано на методе:
    Pyatykh S., Hesser J., Zheng L.:
    "Image Noise Level Estimation by Principal Component Analysis",
    IEEE Transactions on Image Processing, vol. 22, no. 2, 2014.

Модель шума: y = Poisson(x / a) * a + N(0, b), где
    a - параметр пуассоновского (зависящего от сигнала) шума;
    b - дисперсия гауссовского (не зависящего от сигнала) шума.
"""

import numpy as np
import scipy.linalg
from scipy.stats import kurtosis
from scipy.optimize import fminbound

__all__ = ['estimate_noise_params']


def _im2col(image, m1, m2):
    """
    Извлечение перекрывающихся блоков из изображения методом скользящего окна.

    Параметры
    ---------
    image : ndarray
        Двумерный массив изображения.
    m1, m2 : int
        Размеры извлекаемого блока.

    Возвращает
    ----------
    out : ndarray
        Двумерный массив, где каждый столбец представляет собой вытянутый блок.
    """
    rows, cols = image.shape
    s0, s1 = image.strides
    n_rows = rows - m1 + 1
    n_cols = cols - m2 + 1
    out = np.lib.stride_tricks.as_strided(
        image, shape=(m1, m2, n_rows, n_cols), strides=(s0, s1, s0, s1))
    return out.reshape(m1 * m2, -1)


def _get_valid_block_index(image, m1, m2):
    """
    Поиск индексов валидных блоков, не содержащих константных участков 
    или насыщенных пикселей.

    Пиксель считается невалидным, если он совпадает со значением константного 
    блока, меньше или равен нулю, либо больше или равен 255. Блок признается 
    валидным только в том случае, если все его пиксели удовлетворяют условиям.

    Возвращает
    ----------
    indices : ndarray
        Массив индексов валидных блоков.
    """
    block = _im2col(image, m1, m2)
    minimums = np.min(block, axis=0)
    maximums = np.max(block, axis=0)
    equal_minmax = minimums == maximums
    invalid_grayvalue = np.unique(minimums[equal_minmax])
    invalid_mask = (np.isin(block, invalid_grayvalue) |
                    (block <= 0) | (block >= 255))
    blocks_ok = ~invalid_mask.any(axis=0)
    valid_block_index = np.where(blocks_ok)
    return np.array(valid_block_index).T


def _vst(image, phi):
    """
    Преобразование, стабилизирующее дисперсию (VST), параметризованное 
    углом phi.

    Угол phi определяет соотношение компонент модели:
    - phi = 0 соответствует чисто пуассоновскому шуму (a = 1, b = 0).
    - phi = pi/2 соответствует чисто гауссовскому шуму (a = 0, b = 1).
    """
    a = np.cos(phi)
    b = np.sin(phi)
    if a > np.finfo(float).eps:
        return (2.0 / a) * np.sqrt(np.maximum(a * image + b, 0.0))
    return image / np.sqrt(max(b, np.finfo(float).eps))


def _get_blocks(image, phi, row_parity, valid_block_index, m1, m2):
    """
    Извлечение подмножества блоков из изображения после применения 
    преобразования VST.
    """
    block = _im2col(_vst(image, phi), m1, m2)
    block = block[row_parity - 1::2, valid_block_index]
    return np.squeeze(block).T


def _pca_svd_score(data):
    """
    Вычисление проекций (счетов) главных компонент на основе сингулярного 
    разложения центрированных данных.
    """
    centered = data - np.mean(data, axis=0)
    U, s, _ = scipy.linalg.svd(centered, full_matrices=False,
                                check_finite=False)
    return U * s


def _pca_svd_latent(data):
    """
    Вычисление собственных значений (объясненной дисперсии) главных компонент.
    """
    centered = data - np.mean(data, axis=0)
    s = scipy.linalg.svd(centered, full_matrices=False,
                          compute_uv=False, check_finite=False)
    return (s ** 2) / (data.shape[0] - 1)


def _sort_blocks(image, phi, valid_block_index, m1, m2):
    """
    Сортировка блоков по возрастанию текстурной сложности.

    Наименее текстурированные (гладкие) блоки помещаются в начало списка. 
    Энергия текстуры оценивается как сумма квадратов счетов PCA, начиная 
    с четвертой компоненты.
    """
    block = _get_blocks(image, phi, 2, valid_block_index, m1, m2)
    scores = _pca_svd_score(block)
    energy = np.sum(np.square(scores[:, 3:]), axis=1)
    t = np.column_stack((valid_block_index, energy))
    t = t[np.argsort(t[:, 1])]
    return t[:, 0]


def _compute_kurtosis(phi, image, tau, block_count, m1, m2):
    """
    Вычисление эксцесса (kurtosis) шумовой компоненты в области VST.
    """
    block = _get_blocks(image, phi, 1, tau[:block_count], m1, m2)
    scores = _pca_svd_score(block)
    g = (kurtosis(scores[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return g


def _compute_kurtosis_and_block(phi, image, tau, block_count, m1, m2):
    """
    Вычисление эксцесса шумовой компоненты с одновременным возвратом 
    данных блока для последующего анализа.
    """
    block = _get_blocks(image, phi, 1, tau[:block_count], m1, m2)
    scores = _pca_svd_score(block)
    g = (kurtosis(scores[:, -1], fisher=False) - 3) * np.sqrt(block_count / 24)
    return g, block


def estimate_noise_params(image, blocksize=7):
    """
    Оценка параметров пуассоновско-гауссовского шума по единственному 
    изображению.

    Алгоритм использует анализ главных компонент (PCA), стабилизирующее 
    дисперсию преобразование (VST) и оптимизацию по эксцессу для разделения 
    шума на пуассоновскую (зависящую от сигнала) и гауссовскую 
    (не зависящую от сигнала) компоненты.

    Параметры
    ---------
    image : ndarray
        Зашумленное входное изображение. Поддерживаемые форматы:
        - полутоновое (H, W), uint8 [0, 255];
        - полутоновое (H, W), float [0, 1] (автоматически масштабируется к [0, 255]);
        - цветное (H, W, C) (автоматически конвертируется в полутоновое).
    blocksize : int, по умолчанию 7
        Размер стороны квадратного блока для анализа.

    Возвращает
    ----------
    result : dict
        Словарь с результатами оценки:
        - 'a' : параметр пуассоновского шума (float).
        - 'b' : дисперсия гауссовского шума (float).
        - 'sigma' : эффективное СКО шума для средней яркости в масштабе [0, 255] (float).
        - 'sigma_norm' : эффективное СКО шума в нормализованном масштабе [0, 1] (float).
        - 'sigma_gaussian' : СКО гауссовской компоненты (float).
        - 'noise_type' : строковый идентификатор классифицированного типа шума 
          ('gaussian', 'poisson', 'poisson_gaussian', 'unknown').
    """
    img = np.asarray(image, dtype=np.float64)

    if img.ndim == 3:
        if img.shape[2] == 3:
            img = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1]
                   + 0.1140 * img[:, :, 2])
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[2]}")

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image after conversion, got ndim={img.ndim}")

    if img.max() <= 1.0:
        img = img * 255.0

    m1, m2 = blocksize, blocksize
    valid_block_index = _get_valid_block_index(img, m1, m2)

    _empty = {'a': 0.0, 'b': 0.0, 'sigma': 0.0, 'sigma_norm': 0.0,
              'noise_type': 'unknown'}

    if len(valid_block_index) < 1000:
        return _empty

    tau = _sort_blocks(img, 0.0, valid_block_index, m1, m2).astype(int)

    block_count = min(20000, len(tau))
    curr_phi = 0.0
    curr_sigma = 0.0

    while block_count <= len(tau):
        opt_phi = fminbound(
            _compute_kurtosis, 0.0, np.pi / 2 - 0.001,
            args=(img, tau, block_count, m1, m2),
            xtol=0.01, maxfun=10000, disp=0)

        opt_kurtosis, block = _compute_kurtosis_and_block(
            opt_phi, img, tau, block_count, m1, m2)

        if opt_kurtosis < 3 or curr_phi == 0:
            phi_converged = abs(opt_phi - curr_phi) < 0.0005
            curr_phi = opt_phi
            latent = _pca_svd_latent(block)
            curr_sigma = float(np.sqrt(max(latent[-1], 0.0)))
            if phi_converged:
                break
        else:
            break
        block_count += 5000

    a = curr_sigma ** 2 * np.cos(curr_phi)
    b = curr_sigma ** 2 * np.sin(curr_phi)

    if a < 1e-6 and b < 1e-6:
        noise_type = 'unknown'
    elif a < 1e-6:
        noise_type = 'gaussian'
    elif b / max(a, 1e-10) > 10:
        noise_type = 'gaussian'
    elif a / max(b, 1e-10) > 10:
        noise_type = 'poisson'
    else:
        noise_type = 'poisson_gaussian'

    mean_brightness = float(np.mean(img))
    sigma_255 = float(np.sqrt(max(a * mean_brightness + b, 0.0)))

    return {
        'a': float(a),
        'b': float(b),
        'sigma': sigma_255,
        'sigma_norm': sigma_255 / 255.0,
        'sigma_gaussian': float(np.sqrt(max(b, 0.0))),
        'noise_type': noise_type,
    }
