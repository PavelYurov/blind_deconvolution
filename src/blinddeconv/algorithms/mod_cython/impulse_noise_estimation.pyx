"""
impulse_noise_estimation.py

Обнаружение и подавление импульсного шума (типа "соль и перец").

Импульсный шум проявляется в виде изолированных пикселей с экстремальными 
значениями (0 или 255). Подавление такого шума до выполнения слепой 
деконволюции является критически важным шагом. В противном случае алгоритмы 
оценки функции рассеяния точки (PSF) воспринимают импульсы как полезные 
детали изображения, а ядро размытия "размазывает" их, создавая характерные 
звездообразные артефакты.

Метод не требует наличия эталонного изображения и включает два этапа:
1. Обнаружение:
   - Анализ гистограммы для поиска пикселей с экстремальными значениями.
   - Поиск локальных выбросов (пикселей, сильно отличающихся от локальной медианы).
   - Комбинирование сигналов для оценки плотности импульсного шума.
2. Подавление:
   - Использование адаптивного медианного фильтра (AMF). Фильтрация применяется 
     исключительно к обнаруженным зашумленным пикселям, оставляя остальные 
     без изменений, что предотвращает общее размытие изображения.
"""

import numpy as np
from scipy.ndimage import median_filter as _scipy_median_filter

__all__ = [
    'detect_impulse_noise',
    'estimate_impulse_density',
    'adaptive_median_filter',
    'remove_impulse_noise',
]


def _histogram_extremes(image, low_thresh=0.01, high_thresh=0.99):
    """
    Вычисление доли пикселей с экстремальными значениями яркости.

    Шум типа "соль и перец" формирует пики на краях распределения. 
    Для изображений в диапазоне [0, 1] анализируются значения <= low_thresh 
    и >= high_thresh.

    Параметры
    ---------
    image : ndarray
        Массив изображения (в диапазоне [0, 1]).
    low_thresh : float
        Верхняя граница для пикселей типа "перец".
    high_thresh : float
        Нижняя граница для пикселей типа "соль".

    Возвращает
    ----------
    frac_low : float
        Доля пикселей, близких к нулю.
    frac_high : float
        Доля пикселей, близких к максимуму.
    frac_total : float
        Суммарная доля экстремальных пикселей.
    """
    total = image.size
    frac_low = np.count_nonzero(image <= low_thresh) / total
    frac_high = np.count_nonzero(image >= high_thresh) / total
    return frac_low, frac_high, frac_low + frac_high


def _local_outlier_mask(image, window_size=5, threshold=0.15):
    """
    Поиск локальных выбросов (пикселей, значительно отличающихся от 
    медианы своей окрестности).

    Параметры
    ---------
    image : ndarray
        Полутоновое изображение размерности (H, W), значения float [0, 1].
    window_size : int
        Размер окна для локального медианного фильтра (нечетное число).
    threshold : float
        Минимальное отклонение от локальной медианы для классификации 
        пикселя как выброса (в масштабе [0, 1]).

    Возвращает
    ----------
    mask : ndarray
        Булева маска размерности (H, W), где True соответствует 
        подозреваемому импульсному шуму.
    """
    local_med = _scipy_median_filter(image, size=window_size)
    diff = np.abs(image - local_med)
    return diff > threshold


def detect_impulse_noise(image, low_thresh=0.01, high_thresh=0.99,
                         outlier_window=5, outlier_threshold=0.15,
                         density_threshold=0.0005):
    """
    Комплексное обнаружение импульсного шума на изображении.

    Алгоритм комбинирует два подхода:
    1. Поиск экстремумов на границах гистограммы (около 0 и 1).
    2. Поиск локальных выбросов (отклонений от медианы).
    Пиксель помечается как зашумленный, если он удовлетворяет обоим критериям.
    Дополнительно применяется смягченный порог отклонения (0.02) для пикселей 
    с жесткими экстремальными значениями (<= 0.005 или >= 0.995), чтобы 
    избежать их пропуска в очень темных или сильно засвеченных областях.

    Параметры
    ---------
    image : ndarray
        Изображение (H, W) или (H, W, C). Тип float64 [0, 1] или uint8 [0, 255].
    low_thresh : float
        Нижний порог для темных экстремумов ("перец").
    high_thresh : float
        Верхний порог для светлых экстремумов ("соль").
    outlier_window : int
        Размер окна локального медианного фильтра (нечетное число).
    outlier_threshold : float
        Порог отклонения от локальной медианы для классификации выброса.
    density_threshold : float
        Минимальная предполагаемая плотность (по умолчанию 0.05%), при которой 
        принимается решение о наличии импульсного шума на изображении.

    Возвращает
    ----------
    result : dict
        Словарь с результатами анализа:
        - 'has_impulse' : bool, True, если обнаружен шум.
        - 'density' : float, оцененная плотность импульсного шума [0, 1].
        - 'frac_low' : float, доля темных экстремумов.
        - 'frac_high' : float, доля светлых экстремумов.
        - 'outlier_frac' : float, доля локальных выбросов.
        - 'impulse_mask' : ndarray (bool, HxW), пространственная маска шума.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    if img.ndim == 3:
        gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    else:
        gray = img

    frac_low, frac_high, frac_total = _histogram_extremes(
        gray, low_thresh, high_thresh)

    outlier_mask = _local_outlier_mask(
        gray, outlier_window, outlier_threshold)
    outlier_frac = np.count_nonzero(outlier_mask) / gray.size

    extreme_mask = (gray <= low_thresh) | (gray >= high_thresh)

    impulse_mask = extreme_mask & outlier_mask

    hard_extreme = (gray <= 0.005) | (gray >= 0.995)
    hard_outlier = _local_outlier_mask(gray, outlier_window, 0.02)
    impulse_mask = impulse_mask | (hard_extreme & hard_outlier)

    density = np.count_nonzero(impulse_mask) / gray.size

    has_impulse = density >= density_threshold

    return {
        'has_impulse': has_impulse,
        'density': density,
        'frac_low': frac_low,
        'frac_high': frac_high,
        'outlier_frac': outlier_frac,
        'impulse_mask': impulse_mask,
    }


def estimate_impulse_density(image, **kwargs):
    """
    Упрощенная функция для получения только оценки плотности импульсного шума.

    Возвращает
    ----------
    density : float
        Оцененная плотность шума в диапазоне [0, 1]. Возвращает 0.0, если 
        шум не обнаружен (значение ниже порога density_threshold).
    """
    result = detect_impulse_noise(image, **kwargs)
    return result['density'] if result['has_impulse'] else 0.0


def adaptive_median_filter(image, impulse_mask, max_window=7):
    """
    Применение медианного фильтра исключительно к пикселям, отмеченным 
    в маске импульсного шума. Пиксели без шума остаются без изменений.

    Алгоритм итеративно увеличивает размер окна (вплоть до max_window), 
    если вычисленное медианное значение для малого окна само по себе является 
    экстремальным (что характерно для областей с высокой плотностью шума).

    Параметры
    ---------
    image : ndarray
        Полутоновое изображение размерности (H, W), значения float64 [0, 1].
    impulse_mask : ndarray
        Булева маска (H, W), где True соответствует зашумленному пикселю.
    max_window : int
        Максимальный размер окна медианного фильтра (нечетное число).

    Возвращает
    ----------
    filtered : ndarray
        Изображение (H, W) с восстановленными пикселями.
    """
    filtered = image.copy()
    remaining = impulse_mask.copy()

    for wsize in range(3, max_window + 1, 2):
        if not np.any(remaining):
            break
        med = _scipy_median_filter(filtered, size=wsize)
        filtered[remaining] = med[remaining]
        still_extreme = (filtered <= 0.01) | (filtered >= 0.99)
        remaining = remaining & still_extreme

    return filtered


def remove_impulse_noise(image, density_threshold=0.005,
                         max_window=7, outlier_window=5,
                         outlier_threshold=0.15):
    """
    Обнаружение и адаптивное подавление импульсного шума на изображении.

    Если плотность шума оценивается ниже заданного порога, алгоритм 
    завершает работу и возвращает исходное изображение без изменений. 
    Для цветных изображений обнаружение проводится по яркостной компоненте, 
    однако восстановление и проверка поканальных экстремумов осуществляются 
    для каждого цветового канала независимо.

    Параметры
    ---------
    image : ndarray
        Входное изображение (H, W) или (H, W, C), float64 [0, 1] или uint8 [0, 255].
    density_threshold : float
        Минимальная плотность шума для инициации процесса подавления.
    max_window : int
        Максимальный размер окна адаптивного медианного фильтра (AMF).
    outlier_window : int
        Размер окна для поиска локальных выбросов на этапе обнаружения.
    outlier_threshold : float
        Порог отклонения от локальной медианы для этапа обнаружения.

    Возвращает
    ----------
    result : dict
        Словарь с результатами:
        - 'image' : ndarray, отфильтрованное изображение (формат совпадает со входом).
        - 'has_impulse' : bool, флаг обнаружения импульсного шума.
        - 'density' : float, оцененная плотность шума.
        - 'applied' : bool, флаг фактического применения фильтрации (True, 
          если плотность превысила порог).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    is_color = img.ndim == 3

    info = detect_impulse_noise(
        img,
        density_threshold=density_threshold,
        outlier_window=outlier_window,
        outlier_threshold=outlier_threshold,
    )

    if not info['has_impulse']:
        return {
            'image': img,
            'has_impulse': False,
            'density': info['density'],
            'applied': False,
        }

    mask = info['impulse_mask']

    if is_color:
        filtered = img.copy()
        for ch in range(img.shape[2]):
            ch_mask = mask.copy()
            ch_extreme = (img[:, :, ch] <= 0.01) | (img[:, :, ch] >= 0.99)
            ch_mask = ch_mask | (_local_outlier_mask(
                img[:, :, ch], outlier_window, outlier_threshold) & ch_extreme)
            filtered[:, :, ch] = adaptive_median_filter(
                img[:, :, ch], ch_mask, max_window)
    else:
        filtered = adaptive_median_filter(img, mask, max_window)

    return {
        'image': filtered,
        'has_impulse': True,
        'density': info['density'],
        'applied': True,
    }
