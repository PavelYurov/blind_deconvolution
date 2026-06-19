"""
screenot.py

Подавление шума на изображениях с использованием алгоритма ScreeNOT 
(оптимальное адаптивное пороговое ограничение сингулярных чисел).

Основано на методе:
    Donoho, Gavish, Romanov:
    "ScreeNOT: Exact MSE-optimal singular value thresholding in correlated noise."
    Annals of Statistics (2023).

Алгоритм ScreeNOT находит оптимальный (в смысле минимума среднеквадратичной 
ошибки) жесткий порог для сингулярных чисел матрицы наблюдаемых данных 
Y = X + Z, где X — низкоранговый полезный сигнал, а Z — аддитивный шум 
с произвольной (неизвестной) корреляционной структурой. Порог вычисляется 
адаптивно на основе распределения наблюдаемых сингулярных чисел без 
необходимости предварительного знания статистики шума.

Режимы работы:
- 'full' : обработка всего изображения как единой матрицы размерности (H, W). 
  Быстрый метод, не создающий блочных артефактов.
- 'patch' : извлечение перекрывающихся блоков (патчей) в единую матрицу, 
  применение ScreeNOT и обратная сборка с усреднением. Более ресурсоемкий 
  метод, но эффективнее для изображений с высокой текстурной сложностью.
"""

import numpy as np
from numpy.linalg import svd

__all__ = [
    'adaptive_hard_thresholding',
    'screenot_denoise',
]

def _Phi(y, fZ):
    """Вычисление функционала Phi(y; fZ)."""
    return np.mean(y / (y ** 2 - fZ ** 2))


def _Phid(y, fZ):
    """Вычисление производной функционала Phi по переменной y."""
    return np.mean(-(y ** 2 + fZ ** 2) / (y ** 2 - fZ ** 2) ** 2)


def _D(y, fZ, gamma):
    """Вычисление функционала D_gamma(y; fZ)."""
    phi = _Phi(y, fZ)
    return phi * (gamma * phi + (1 - gamma) / y)


def _Dd(y, fZ, gamma):
    """Вычисление производной функционала D_gamma по переменной y."""
    phi = _Phi(y, fZ)
    phid = _Phid(y, fZ)
    return (phid * (gamma * phi + (1 - gamma) / y)
            + phi * (gamma * phid - (1 - gamma) / y ** 2))


def _F(y, fZ, gamma):
    """
    Вычисление функционала Psi_gamma(y; fZ). 
    Оптимальный порог удовлетворяет условию F = -4.
    """
    d = _D(y, fZ, gamma)
    dd = _Dd(y, fZ, gamma)
    return y * dd / d


def _create_pseudo_noise(fY, k, strategy='i'):
    """
    Оценка распределения сингулярных чисел шума на основе наблюдаемых 
    сингулярных чисел.

    Параметры
    ---------
    fY : ndarray
        Одномерный массив наблюдаемых сингулярных чисел.
    k : int
        Верхняя граница ранга полезного сигнала.
    strategy : str
        Стратегия оценки распределения шума:
        - 'i' : импутация (заполнение на основе аппроксимации).
        - 'w' : винзоризация (замещение крайних значений).
        - '0' : обнуление.

    Возвращает
    ----------
    fZ : ndarray
        Оцененное распределение сингулярных чисел шума.
    """
    fZ = np.sort(fY)
    p = fZ.size
    if k >= p:
        raise ValueError('k too large: requires k < min(n, p)')

    if k > 0:
        if strategy == '0':
            fZ[-k:] = 0
        elif strategy == 'w':
            fZ[-k:] = fZ[-k - 1]
        elif strategy == 'i':
            if 2 * k + 1 >= p:
                raise ValueError(
                    'k too large for imputation: requires 2*k+1 < min(n, p)')
            diff = fZ[-k - 1] - fZ[-2 * k - 1]
            for l in range(1, k + 1):
                a = (1 - ((l - 1) / k) ** (2 / 3)) / (2 ** (2 / 3) - 1)
                fZ[-l] = fZ[-k - 1] + a * diff
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}', use 'i', 'w', or '0'")
    return fZ


def _compute_opt_threshold(fZ, gamma):
    """
    Двоичный поиск оптимального порога t*, удовлетворяющего условию F(t*; fZ) = -4.
    """
    low = np.max(fZ)
    high = low + 2.0
    while _F(high, fZ, gamma) < -4:
        low = high
        high = 2 * high

    eps = 1e-6
    while high - low > eps:
        mid = (high + low) / 2
        if _F(mid, fZ, gamma) < -4:
            low = mid
        else:
            high = mid
    return (high + low) / 2


def adaptive_hard_thresholding(Y, k, strategy='i'):
    """
    Оптимальное адаптивное жесткое пороговое ограничение сингулярных чисел 
    матрицы (алгоритм ScreeNOT).

    Параметры
    ---------
    Y : ndarray
        Наблюдаемая матрица размерности (n, p), представляющая собой 
        сумму полезного сигнала и шума.
    k : int
        Верхняя граница ранга сигнала (допускается нестрогая оценка).
    strategy : str
        Стратегия оценки распределения шума ('i', 'w' или '0').

    Возвращает
    ----------
    Xest : ndarray
        Восстановленная матрица низкого ранга (формат совпадает с Y).
    Topt : float
        Примененный оптимальный жесткий порог.
    r : int
        Оцененный ранг сигнала (количество сохраненных компонент).
    """
    U, fY, Vt = svd(Y, full_matrices=False)
    gamma = min(Y.shape[0] / Y.shape[1], Y.shape[1] / Y.shape[0])

    fZ = _create_pseudo_noise(fY, k, strategy=strategy)
    Topt = _compute_opt_threshold(fZ, gamma)

    fY_new = fY * (fY > Topt)
    Xest = U @ np.diag(fY_new) @ Vt
    r = int(np.sum(fY_new > 0))

    return Xest, Topt, r


def screenot_denoise(image, k=10, strategy='i', mode='full',
                     patch_size=8, stride=3):
    """
    Подавление шума на двумерном полутоновом изображении.

    Параметры
    ---------
    image : ndarray
        Входное полутоновое изображение (размерность HxW), значения 
        float64 в диапазоне [0, 1].
    k : int
        Верхняя граница ранга сигнала.
        Для режима 'full': ожидаемый ранг матрицы изображения (обычно 10-50).
        Для режима 'patch': ожидаемый ранг матрицы блоков (обычно 5-20).
    strategy : str
        Стратегия оценки шума в ScreeNOT ('i', 'w' или '0'). По умолчанию 'i'.
    mode : str
        Режим обработки:
        - 'full' : применение алгоритма ко всему изображению как к единой матрице.
        - 'patch' : извлечение перекрывающихся блоков, их обработка и агрегация.
    patch_size : int, по умолчанию 8
        Длина стороны квадратного блока (используется только в режиме 'patch').
    stride : int, по умолчанию 3
        Шаг смещения блоков (используется только в режиме 'patch').

    Возвращает
    ----------
    denoised : ndarray
        Восстановленное изображение (формат совпадает со входом).
    info : dict
        Словарь метаданных:
        - 'Topt' : вычисленный оптимальный порог.
        - 'rank' : оцененный ранг полезного сигнала.
        - 'mode' : примененный режим обработки.
    """
    if image.ndim != 2:
        raise ValueError(f'Expected 2D image, got shape {image.shape}')

    if mode == 'full':
        return _denoise_full(image, k, strategy)
    elif mode == 'patch':
        return _denoise_patch(image, k, strategy, patch_size, stride)
    else:
        raise ValueError(f"Unknown mode '{mode}', use 'full' or 'patch'")


def _denoise_full(image, k, strategy):
    """
    Применение алгоритма ScreeNOT напрямую к матрице изображения.
    Значение k автоматически ограничивается допустимым диапазоном для 
    выбранной стратегии импутации.
    """
    H, W = image.shape
    min_dim = min(H, W)

    max_k = min_dim // 2 - 1
    if k > max_k:
        k = max(1, max_k)

    try:
        denoised, Topt, r = adaptive_hard_thresholding(
            image, k, strategy=strategy)
    except ValueError:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'mode': 'full', 'skipped': True,
        }

    denoised = np.clip(denoised, 0.0, 1.0)
    return denoised, {
        'Topt': float(Topt),
        'rank': r,
        'mode': 'full',
        'image_shape': (H, W),
        'skipped': False,
    }


def _extract_patches(image, patch_size, stride):
    """
    Извлечение перекрывающихся блоков изображения в виде строк единой матрицы.
    """
    H, W = image.shape
    positions = []
    rows = []
    for y0 in range(0, H - patch_size + 1, stride):
        for x0 in range(0, W - patch_size + 1, stride):
            patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
            rows.append(patch.ravel())
            positions.append((y0, x0))
    return np.array(rows, dtype=np.float64), positions


def _aggregate_patches(patches, positions, patch_size, image_shape):
    """
    Обратная сборка изображения из набора блоков с усреднением значений 
    в об
    H, W = image_shape
    accum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for i, (y0, x0) in enumerate(positions):
        patch_2d = patches[i].reshape(patch_size, patch_size)
        accum[y0:y0 + patch_size, x0:x0 + patch_size] += patch_2d
        count[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0
    count = np.maximum(count, 1.0)
    return accum / count


def _denoise_patch(image, k, strategy, patch_size, stride):
    """
    Подавление шума с использованием блочного (patch-based) подхода.
    Сформированная матрица блоков подвергается обработке ScreeNOT, после 
    чего изображение восстанавливается с помощью усреднения.
    """
    H, W = image.shape
    if H < patch_size or W < patch_size:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'mode': 'patch', 'skipped': True,
        }

    patches, positions = _extract_patches(image, patch_size, stride)
    n_patches, dim = patches.shape

    max_k = min(n_patches, dim) // 2 - 1
    if k > max_k:
        k = max(1, max_k)

    try:
        denoised_patches, Topt, r = adaptive_hard_thresholding(
            patches, k, strategy=strategy)
    except ValueError:
        return image.copy(), {
            'Topt': 0.0, 'rank': 0, 'n_patches': n_patches,
            'patch_matrix_shape': patches.shape, 'mode': 'patch',
            'skipped': True,
        }

    denoised = _aggregate_patches(
        denoised_patches, positions, patch_size, (H, W))
    denoised = np.clip(denoised, 0.0, 1.0)

    return denoised, {
        'Topt': float(Topt),
        'rank': r,
        'n_patches': n_patches,
        'patch_matrix_shape': patches.shape,
        'mode': 'patch',
        'skipped': False,
    }
