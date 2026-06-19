"""
noise_psd_analysis.py

Анализ спектральной плотности мощности (СПМ) шума, спектральная фильтрация 
и обнаружение коррелированных шумовых компонент.

Модуль предоставляет инструменты для:
1. Оценки СПМ шума по единственному изображению на основе спектрального 
   анализа наиболее гладких участков (блоков).
2. Обнаружения периодического шума путем поиска изолированных 
   спектральных пиков.
3. Оценки корреляции шума на основе автокорреляции с задержкой 1 (lag-1) 
   для шумовых остатков.
4. Применения спектральных фильтров: режекторного (notch), 
   режекторного полосового (band-stop) и фильтра обесцвечивания (prewhitening).

Примечание: спектральный наклон beta (где P ~ 1/f^beta), оцениваемый по 
блокам изображения, ненадежен для естественных двумерных изображений из-за 
остаточного присутствия полезного сигнала со спектром ~1/f^2. Значение beta 
вычисляется исключительно в информационных целях. Автоматические решения 
о наличии корреляции принимаются на основе теста автокорреляции lag-1.
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import median_filter as _median_filter_nd

__all__ = [
    'analyze_noise_psd',
    'estimate_noise_psd',
    'classify_noise',
    'prewhiten',
    'notch_filter',
    'bandstop_filter',
]


def _radial_profile(psd_2d):
    """
    Вычисление радиального (азимутально усредненного) профиля спектра мощности.

    Параметры
    ---------
    psd_2d : ndarray
        Центрированный двумерный спектр мощности размерности (H, W).

    Возвращает
    ----------
    radii : ndarray
        Массив частотных бинов (радиусов в пикселях от центрального).
    profile : ndarray
        Среднее значение СПМ для каждого радиуса.
    """
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    radii = np.arange(0, max_r)
    profile = np.zeros(max_r, dtype=np.float64)
    for r in radii:
        mask = R == r
        if mask.any():
            profile[r] = psd_2d[mask].mean()
    return radii, profile


def _extract_smooth_patches(image, pch_size=32, stride=None, n_patches=100):
    """
    Извлечение участков изображения с минимальной текстурой (наиболее гладких).

    Блоки сортируются по возрастанию энергии градиента (дисперсия разностей 
    по горизонтали и вертикали). Самые гладкие блоки содержат преимущественно 
    шум с минимальной долей полезного сигнала, что делает их оптимальными 
    для оценки СПМ шума.

    Параметры
    ---------
    image : ndarray
        Полутоновое изображение размерности (H, W).
    pch_size : int
        Размер стороны квадратного блока.
    stride : int или None
        Шаг смещения окна. Если None, используется значение pch_size // 2.
    n_patches : int
        Количество извлекаемых блоков для возврата.

    Возвращает
    ----------
    patches : list of ndarray
        Список извлеченных блоков размерности (pch_size, pch_size).
    """
    H, W = image.shape
    if stride is None:
        stride = max(1, pch_size // 2)

    candidates = []
    for y0 in range(0, H - pch_size + 1, stride):
        for x0 in range(0, W - pch_size + 1, stride):
            p = image[y0:y0 + pch_size, x0:x0 + pch_size]
            dx = np.diff(p, axis=1)
            dy = np.diff(p, axis=0)
            energy = float(np.var(dx) + np.var(dy))
            candidates.append((energy, p))

    candidates.sort(key=lambda t: t[0])
    return [c[1] for c in candidates[:n_patches]]


def _detrend_patch(patch):
    """
    Удаление плоского тренда вида (ax + by + c) из двумерного блока.

    Устраняет постоянную составляющую и линейные градиенты, оставляя только 
    шум и высокочастотные остатки сигнала. Это необходимо для несмещенной 
    оценки СПМ шума: без удаления тренда градиенты яркости в гладких блоках 
    формируют ложный спектральный спад 1/f^2.

    Параметры
    ---------
    patch : ndarray
        Входной двумерный массив.

    Возвращает
    ----------
    detrended : ndarray
        Блок с вычтенным линейным трендом.
    """
    h, w = patch.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx.ravel().astype(np.float64),
                         yy.ravel().astype(np.float64),
                         np.ones(h * w, dtype=np.float64)])
    coef = np.linalg.lstsq(A, patch.ravel(), rcond=None)[0]
    trend = (A @ coef).reshape(h, w)
    return patch - trend


def _detect_periodic_peaks_2d(psd_2d, threshold_factor=8.0, min_radius=5,
                              max_peaks=20):
    """
    Обнаружение изолированных спектральных пиков, соответствующих 
    периодическому шуму.

    Значение каждого частотного бина сравнивается с медианным значением 
    по кольцу соответствующего радиуса (радиальным средним). Истинные 
    периодические пики значительно превышают этот базовый уровень.
    Для устранения близких срабатываний применяется жадная кластеризация: 
    пики сортируются по мощности, соседние пики подавляются.

    Параметры
    ---------
    psd_2d : ndarray
        Центрированный двумерный спектр мощности размерности (H, W).
    threshold_factor : float
        Минимальное отношение пика к радиальному среднему уровню.
    min_radius : int
        Минимальное расстояние от нулевой частоты для поиска.
    max_peaks : int
        Максимальное количество возвращаемых пиков.

    Возвращает
    ----------
    peaks : list of dict
        Список найденных пиков. Каждый элемент содержит координаты 'u', 'v', 
        расстояние от центра 'radius', мощность 'power' и отношение 'ratio'.
    """
    H, W = psd_2d.shape
    cy, cx = H // 2, W // 2

    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    R_int = R.astype(int)
    max_r = min(cy, cx)

    radial_avg = np.zeros(max_r + 1, dtype=np.float64)
    radial_cnt = np.zeros(max_r + 1, dtype=np.float64)
    for r in range(max_r + 1):
        mask_r = R_int == r
        if mask_r.any():
            radial_avg[r] = np.median(psd_2d[mask_r])
            radial_cnt[r] = mask_r.sum()

    baseline = np.ones_like(psd_2d) * np.median(psd_2d)
    for r in range(max_r + 1):
        mask_r = R_int == r
        baseline[mask_r] = max(radial_avg[r], 1e-30)

    ratio_map = psd_2d / baseline

    border = 2
    peak_mask = (
        (ratio_map > threshold_factor) &
        (R > min_radius) &
        (R < min(cy, cx) * 0.95)
    )
    peak_mask[:border, :] = False
    peak_mask[-border:, :] = False
    peak_mask[:, :border] = False
    peak_mask[:, -border:] = False

    peaks = []
    coords = np.argwhere(peak_mask)
    if len(coords) == 0:
        return peaks
    
    powers = psd_2d[peak_mask]
    order = np.argsort(-powers)
    coords_sorted = coords[order]
    used = np.zeros(len(coords_sorted), dtype=bool)

    for i in range(len(coords_sorted)):
        if used[i]:
            continue
        v, u = coords_sorted[i]
        r = float(np.sqrt((u - cx) ** 2 + (v - cy) ** 2))
        peaks.append({
            'u': int(u),
            'v': int(v),
            'radius': r,
            'power': float(psd_2d[v, u]),
            'ratio': float(ratio_map[v, u]),
        })
        for j in range(i + 1, len(coords_sorted)):
            vj, uj = coords_sorted[j]
            if abs(vj - v) <= 5 and abs(uj - u) <= 5:
                used[j] = True

    return peaks[:max_peaks]


def estimate_noise_psd(image, pch_size=32, n_smooth=100):
    """
    Оценка двумерной СПМ шума на основе наиболее гладких участков изображения.

    Используется метод Бартлетта: извлекаются гладкие блоки, из них вычитается 
    линейный тренд, после чего применяется оконная функция Ханнинга (для 
    уменьшения спектрального просачивания) и усредняются их периодограммы. 
    Итоговый спектр интерполируется до полных размеров исходного изображения.

    Параметры
    ---------
    image : ndarray
        Изображение размерности (H, W). Значения в диапазоне [0, 1] или [0, 255].
    pch_size : int
        Размер стороны блока для анализа.
    n_smooth : int
        Количество используемых гладких блоков.

    Возвращает
    ----------
    psd_2d_full : ndarray
        Оценка СПМ шума, масштабированная до полного размера изображения (H, W).
    norm_freq : ndarray
        Нормализованные радиальные частоты в диапазоне [0, 1].
    radial_psd : ndarray
        Радиально усредненная СПМ шума, полученная по блокам.
    psd_avg : ndarray
        Оригинальная двумерная СПМ размерности (pch_size, pch_size).
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0

    H, W = img.shape[:2]
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    patches = _extract_smooth_patches(img, pch_size=pch_size, n_patches=n_smooth)
    if len(patches) < 5:
        F = fftshift(fft2(img))
        psd_full = np.abs(F) ** 2 / (H * W)
        radii, profile = _radial_profile(psd_full)
        max_r = min(H // 2, W // 2)
        return psd_full, radii / max_r, profile, psd_full

    window = np.outer(np.hanning(pch_size), np.hanning(pch_size))
    window_energy = np.sum(window ** 2)
    psd_avg = np.zeros((pch_size, pch_size), dtype=np.float64)
    for p in patches:
        p_dt = _detrend_patch(p) * window
        F = fftshift(fft2(p_dt))
        psd_avg += np.abs(F) ** 2
    psd_avg /= len(patches)
    psd_avg /= window_energy

    radii, radial_psd = _radial_profile(psd_avg)
    max_r_patch = pch_size // 2
    norm_freq = radii / max(max_r_patch, 1)

    cy, cx = H // 2, W // 2
    max_r_full = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    R_full = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    scale = max_r_patch / max(max_r_full, 1)
    R_patch = R_full * scale
    R_patch = np.clip(R_patch, 0, len(radial_psd) - 1)
    r_floor = np.floor(R_patch).astype(int)
    r_ceil = np.minimum(r_floor + 1, len(radial_psd) - 1)
    frac = R_patch - r_floor
    psd_2d_full = radial_psd[r_floor] * (1 - frac) + radial_psd[r_ceil] * frac

    return psd_2d_full, norm_freq, radial_psd, psd_avg


def _lag1_autocorrelation(image, pch_size=32, n_patches=100):
    """
    Проверка шума на коррелированность через автокорреляцию с задержкой 1 
    для шумовых остатков по горизонтали и вертикали.

    Для белого шума значение ожидается около нуля. Существенные положительные 
    значения указывают на пространственную корреляцию шума. Статистическая 
    значимость оценивается с использованием порога в 3 стандартных отклонения 
    (3 * sigma), где теоретическое СКО для блоков размера pch_size составляет 
    примерно 1 / pch_size.

    Возвращает
    ----------
    lag1_h : float
        Усредненная горизонтальная автокорреляция.
    lag1_v : float
        Усредненная вертикальная автокорреляция.
    is_correlated : bool
        Флаг, указывающий на превышение порога значимости.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    patches = _extract_smooth_patches(img, pch_size=pch_size,
                                      n_patches=n_patches)
    if len(patches) < 5:
        return 0.0, 0.0, False

    lag1_h_list = []
    lag1_v_list = []
    for p in patches:
        p_dt = _detrend_patch(p)
        var = np.var(p_dt)
        if var < 1e-15:
            continue
        h = np.mean(p_dt[:, :-1] * p_dt[:, 1:]) / var
        v = np.mean(p_dt[:-1, :] * p_dt[1:, :]) / var
        lag1_h_list.append(h)
        lag1_v_list.append(v)

    if not lag1_h_list:
        return 0.0, 0.0, False

    lag1_h = float(np.median(lag1_h_list))
    lag1_v = float(np.median(lag1_v_list))

    expected_std = 1.0 / pch_size
    threshold = 3.0 * expected_std
    is_correlated = (lag1_h > threshold) or (lag1_v > threshold)

    return lag1_h, lag1_v, is_correlated


def classify_noise(radial_freq, radial_psd, psd_2d_full=None,
                   peak_threshold=100.0, image=None):
    """
    Определение типа шума на основе спектральных пиков и автокорреляции.

    Возвращает
    ----------
    result : dict
        Словарь параметров классификации:
        - 'noise_class' : тип шума ('white', 'periodic' или 'correlated').
        - 'beta' : оценка спектрального наклона (справочная информация).
        - 'is_correlated' : результат автокорреляционного теста.
        - 'has_periodic' : наличие выраженных периодических пиков.
        - 'periodic_peaks' : список параметров найденных пиков.
        - 'noise_floor' : базовый уровень шума (медиана).
        - 'lag1_h', 'lag1_v' : коэффициенты автокорреляции lag-1.
    """
    valid = (radial_freq > 0.35) & (radial_freq < 0.9)
    if valid.sum() < 3:
        beta = 0.0
    else:
        f = radial_freq[valid]
        p = np.maximum(radial_psd[valid], 1e-30)
        A = np.vstack([np.log(f), np.ones_like(f)]).T
        coeff = np.linalg.lstsq(A, np.log(p), rcond=None)[0]
        beta = float(-coeff[0])

    noise_floor = float(np.median(radial_psd[valid])) if valid.sum() >= 3 \
        else float(np.median(radial_psd))

    peaks = []
    if psd_2d_full is not None:
        peaks = _detect_periodic_peaks_2d(psd_2d_full,
                                          threshold_factor=peak_threshold)
    has_periodic = len(peaks) > 0

    lag1_h, lag1_v, is_corr = 0.0, 0.0, False
    if image is not None:
        lag1_h, lag1_v, is_corr = _lag1_autocorrelation(image)

    if has_periodic:
        noise_class = 'periodic'
    elif is_corr:
        noise_class = 'correlated'
    else:
        noise_class = 'white'

    return {
        'noise_class': noise_class,
        'beta': beta,
        'is_correlated': is_corr,
        'has_periodic': has_periodic,
        'periodic_peaks': peaks,
        'noise_floor': noise_floor,
        'lag1_h': lag1_h,
        'lag1_v': lag1_v,
    }


def analyze_noise_psd(image, pch_size=32, n_smooth=100,
                      peak_threshold=100.0):
    """
    Комплексный спектральный анализ шума на изображении.

    Объединяет функции оценки СПМ шума, тест автокорреляции и итоговую 
    классификацию в единый процесс.

    Параметры
    ---------
    image : ndarray
        Полутоновое изображение размерности (H, W).
    pch_size : int
        Размер блоков для оценки СПМ.
    n_smooth : int
        Количество извлекаемых гладких блоков.
    peak_threshold : float
        Порог обнаружения периодических пиков.

    Возвращает
    ----------
    info : dict
        Словарь, содержащий данные двумерной СПМ, радиального профиля и 
        результаты классификации типа шума.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3:
        img = 0.2989 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    psd_2d, radial_freq, radial_psd, psd_patches = estimate_noise_psd(
        img, pch_size=pch_size, n_smooth=n_smooth)

    F = fftshift(fft2(img))
    psd_full_2d = np.abs(F) ** 2 / (img.shape[0] * img.shape[1])

    classification = classify_noise(
        radial_freq, radial_psd,
        psd_2d_full=psd_full_2d,
        peak_threshold=peak_threshold,
        image=img)

    return {
        'psd_2d': psd_2d,
        'psd_2d_patches': psd_patches,
        'radial_freq': radial_freq,
        'radial_psd': radial_psd,
        **classification,
    }

def notch_filter(image, peaks, notch_radius=3, rolloff=2):
    """
    Подавление периодического шума на основе фильтра Баттерворта.

    Для каждого обнаруженного пика в спектре создается режекторный 
    провал (notch) по координатам (u, v) и симметрично (-u, -v). Если 
    координаты не заданы, применяется кольцевое подавление по заданному 
    радиусу.

    Параметры
    ---------
    image : ndarray
        Изображение размерности (H, W) или (H, W, C).
    peaks : list of dict
        Список характеристик пиков, сформированный функцией analyze_noise_psd.
    notch_radius : int
        Ширина режекторного провала вокруг пика (в пикселях).
    rolloff : int
        Порядок фильтра Баттерворта, определяющий гладкость перехода.

    Возвращает
    ----------
    filtered : ndarray
        Отфильтрованное изображение (тип float64).
    """
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = notch_filter(img[:, :, ch], peaks,
                                         notch_radius, rolloff)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    cy, cx = H // 2, W // 2
    Y, X = np.mgrid[:H, :W]

    mask = np.ones((H, W), dtype=np.float64)
    for pk in peaks:
        if 'u' in pk and 'v' in pk:
            u0, v0 = pk['u'], pk['v']
            for (pu, pv) in [(u0, v0), (2 * cx - u0, 2 * cy - v0)]:
                D = np.sqrt((X - pu) ** 2 + (Y - pv) ** 2)
                D = np.maximum(D, 1e-10)
                notch = 1.0 - 1.0 / (1.0 + (D / max(notch_radius, 1)) ** (2 * rolloff))
                mask *= notch
        else:
            r0 = pk['radius']
            R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            D = np.abs(R - r0)
            D = np.maximum(D, 1e-10)
            notch = 1.0 - 1.0 / (1.0 + (D / max(notch_radius, 1)) ** (2 * rolloff))
            mask *= notch
    F = fftshift(fft2(img))
    F_filtered = F * mask
    filtered = np.real(ifft2(ifftshift(F_filtered)))

    return filtered * 255.0 if was_255 else filtered


def bandstop_filter(image, freq_low, freq_high, order=2):
    """
    Подавление заданного диапазона радиальных частот.

    Реализуется с помощью режекторного полосового фильтра Баттерворта, 
    который ослабляет частоты между freq_low и freq_high.

    Параметры
    ---------
    image : ndarray
        Изображение размерности (H, W).
    freq_low : float
        Нижняя граница подавляемого диапазона (от 0 до 1, где 1 - частота Найквиста).
    freq_high : float
        Верхняя граница подавляемого диапазона.
    order : int
        Порядок фильтра Баттерворта.

    Возвращает
    ----------
    filtered : ndarray
        Отфильтрованное изображение (тип float64).
    """
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = bandstop_filter(img[:, :, ch],
                                            freq_low, freq_high, order)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    Y, X = np.ogrid[:H, :W]
    R_norm = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_r  # [0, ~1.4]

    f_centre = (freq_low + freq_high) / 2.0
    f_width = (freq_high - freq_low) / 2.0

    D = np.abs(R_norm - f_centre)
    D = np.maximum(D, 1e-10)
    mask = 1.0 - 1.0 / (1.0 + (D / max(f_width, 1e-6)) ** (2 * order))

    mask[cy, cx] = 1.0

    F = fftshift(fft2(img))
    F_filtered = F * mask
    filtered = np.real(ifft2(ifftshift(F_filtered)))

    return filtered * 255.0 if was_255 else filtered



def prewhiten(image, psd_2d, reg=1e-3):
    """
    Обесцвечивание изображения путем спектрального деления на корень из СПМ шума.

    Внимание: данная функция требует использования СПМ, состоящей исключительно 
    из шума. Если применяемая СПМ (например, полученная из функции 
    estimate_noise_psd) содержит существенную долю спектральной энергии 
    полезного сигнала, применение фильтра приведет к искажению изображения 
    (подавлению низкочастотной структуры сигнала и чрезмерному усилению 
    высокочастотных компонент).
    Функция оставлена для экспериментального использования и не вызывается 
    автоматически в стандартном конвейере обработки.

    Характеристика фильтра: W(f) = 1 / sqrt(P_n(f) + reg)

    Параметры
    ---------
    image : ndarray
        Изображение размерности (H, W).
    psd_2d : ndarray
        Центрированная двумерная СПМ шума.
    reg : float
        Коэффициент регуляризации Тихонова (по умолчанию 1e-3).

    Возвращает
    ----------
    whitened : ndarray
        Обесцвеченное изображение.
    """
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0
    if was_255:
        img = img / 255.0

    if img.ndim == 3:
        out = np.zeros_like(img)
        for ch in range(img.shape[2]):
            out[:, :, ch] = prewhiten(img[:, :, ch], psd_2d, reg)
        return out * 255.0 if was_255 else out

    H, W = img.shape
    psd = np.asarray(psd_2d, dtype=np.float64)

    if psd.shape != (H, W):
        from scipy.ndimage import zoom
        psd = zoom(psd, (H / psd.shape[0], W / psd.shape[1]), order=1)

    W_filter = 1.0 / np.sqrt(psd + reg)

    med = np.median(W_filter)
    if med > 0:
        W_filter = W_filter / med

    F = fftshift(fft2(img))
    F_whitened = F * W_filter
    whitened = np.real(ifft2(ifftshift(F_whitened)))

    whitened = np.clip(whitened, 0.0, 1.0)

    return whitened * 255.0 if was_255 else whitened


def noise_preprocess(image, pch_size=32, n_smooth=100,
                     peak_threshold=100.0,
                     notch_radius=3):
    """
    Автоматическая спектральная предобработка шума (анализ и фильтрация).

    В автоматическом режиме применяется только режекторная фильтрация для 
    выявленного периодического шума. Процедура обесцвечивания (prewhitening) 
    автоматически не выполняется, так как требует идеально чистой СПМ шума, 
    что затруднительно обеспечить для единственного изображения.

    Параметры
    ---------
    image : ndarray
        Изображение размерности (H, W) или (H, W, C).
    pch_size : int
        Размер блока для оценки СПМ.
    n_smooth : int
        Количество извлекаемых гладких блоков.
    peak_threshold : float
        Порог для обнаружения пиков периодического шума (истинные пики 
        обычно имеют отношение более 2000, порог 100 исключает ложные 
        срабатывания).
    notch_radius : int
        Радиус подавления вокруг пика для фильтрации.

    Возвращает
    ----------
    result : dict
        Словарь с результатами:
        - 'image' : предобработанное изображение (формат совпадает со входом).
        - 'psd_info' : подробный анализ СПМ (из analyze_noise_psd).
        - 'applied' : список примененных фильтров (например, ['notch'] или []).
    """
    img = np.asarray(image, dtype=np.float64)
    was_255 = img.max() > 1.0

    if was_255:
        work = img / 255.0
    else:
        work = img.copy()

    if work.ndim == 3:
        gray = 0.2989 * work[:, :, 0] + 0.587 * work[:, :, 1] + 0.114 * work[:, :, 2]
    else:
        gray = work

    psd_info = analyze_noise_psd(gray, pch_size=pch_size,
                                 n_smooth=n_smooth,
                                 peak_threshold=peak_threshold)

    applied = []
    processed = work.copy()
    if psd_info['has_periodic']:
        processed = notch_filter(processed, psd_info['periodic_peaks'],
                                 notch_radius=notch_radius)
        applied.append('notch')

    if was_255:
        processed = processed * 255.0

    return {
        'image': processed,
        'psd_info': psd_info,
        'applied': applied,
    }
