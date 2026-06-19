import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Callable, Optional, Tuple


def _align_shift(original: np.ndarray,
                 restored: np.ndarray,
                 max_shift: int = 8,
                 border: int = 4,
                 data_range: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Находит целые (dy, dx) в диапозоне [-max_shift, max_shift],
    которые максимизируют метрику между оригинальным изображением
    и восстановленным на пересекающемся центральном участке (без граничных пикселей).

    Возвращает 2 обрезанных массива и лучший сдвиг.
    Работает для 2D и 3D массивов.
    """
    a = np.asarray(original)
    b = np.asarray(restored)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch in _align_shift: {a.shape} vs {b.shape}")

    if data_range is None:
        data_range = 255.0 if a.dtype.kind in 'ui' else 1.0

    H, W = a.shape[:2]
    ms = int(max_shift)
    bd = int(border)
    inner_h = H - 2 * (ms + bd)
    inner_w = W - 2 * (ms + bd)
    if inner_h <= 0 or inner_w <= 0:
        ms = 0
        inner_h = H - 2 * bd
        inner_w = W - 2 * bd
        if inner_h <= 0 or inner_w <= 0:
            return a, b, (0, 0)
    y0 = ms + bd
    x0 = ms + bd
    a_win = a[y0:y0 + inner_h, x0:x0 + inner_w]

    best_mse = np.inf
    best_dy, best_dx = 0, 0
    for dy in range(-ms, ms + 1):
        for dx in range(-ms, ms + 1):
            b_win = b[y0 + dy:y0 + dy + inner_h,
                      x0 + dx:x0 + dx + inner_w]
            diff = a_win.astype(np.float64) - b_win.astype(np.float64)
            mse = float(np.mean(diff * diff))
            if mse < best_mse:
                best_mse = mse
                best_dy, best_dx = dy, dx

    b_aligned = b[y0 + best_dy:y0 + best_dy + inner_h,
                  x0 + best_dx:x0 + best_dx + inner_w]
    return a_win, b_aligned, (best_dy, best_dx)


def PSNR(original: np.ndarray,
         restored: np.ndarray,
         aligned: bool = True,
         max_shift: int = 8,
         border: int = 4) -> float:
    """
    Вычисляет отношение пикового сигнала к шуму (PSNR) между изображениями.

    Авторы: Юров П.И.

    Аргументы:
        original (ndarray): Исходное изображение
        restored (ndarray): Восстановленное/обработанное изображение
        aligned (bool): Если True (по умолчанию), перед вычислением
            PSNR выполняется поиск оптимального целочисленного сдвига
            restored относительно original в окне
            [-max_shift, +max_shift] по обеим осям; это компенсирует
            translation-ambiguity слепой деконволюции.
            Если False — стандартный попиксельный PSNR.
        max_shift (int): Полуширина окна поиска сдвига (только при
            aligned=True).
        border (int): Дополнительный отступ от границы (px), который
            выбрасывается из оценки, чтобы не считать на ringing-артефактах.

    Возвращает:
        Значение PSNR в децибелах (dB)
    """
    if not aligned:
        return peak_signal_noise_ratio(original, restored)

    a_win, b_win, _ = _align_shift(original, restored,
                                   max_shift=max_shift, border=border)
    return peak_signal_noise_ratio(a_win, b_win)


def SSIM(original: np.ndarray,
         restored: np.ndarray,
         data_range: Optional[float] = None,
         aligned: bool = True,
         max_shift: int = 8,
         border: int = 4) -> float:
    """
    Вычисляет индекс структурного сходства (SSIM) между изображениями.

    Авторы: Юров П.И.

    Аргументы:
        original: Исходное изображение
        restored: Восстановленное/обработанное изображение
        data_range (Optional[float]): Верхний предел значений
        aligned (bool): Если True (по умолчанию), перед вычислением
            SSIM выполняется поиск оптимального целочисленного сдвига
            restored относительно original в окне
            [-max_shift, +max_shift] по обеим осям (сдвиг
            подбирается по MSE, что эквивалентно максимизации PSNR).
            Если False — стандартный SSIM.
        max_shift (int): Полуширина окна поиска сдвига (только при
            aligned=True).
        border (int): Дополнительный отступ от границы (px).

    Возвращает:
        Значение SSIM в диапазоне от 0 до 1
    """
    if not aligned:
        return structural_similarity(original, restored, data_range=data_range)

    a_win, b_win, _ = _align_shift(original, restored,
                                   max_shift=max_shift, border=border,
                                   data_range=data_range)
    if a_win.ndim == 3:
        return structural_similarity(a_win, b_win,
                                     data_range=data_range,
                                     channel_axis=-1)
    return structural_similarity(a_win, b_win, data_range=data_range)


def calculate_sml(image: np.ndarray) -> float:
    """
    Вычисляет Sum of Modified Laplacian (SML) для изображения.

    Авторы: Беззаборов А.А.
    Мера общей резкости/количества краев.

    Аргументы:
        image (ndarray): Входное изображение (grayscale или color)

    Возвращает:
        Сумма модифицированного лапласиана
    """

    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kernel_x = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
    kernel_y = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float32)
    ml_x = cv.filter2D(image, -1, kernel_x)
    ml_y = cv.filter2D(image, -1, kernel_y)
    ml = np.abs(ml_x) + np.abs(ml_y)
    return np.sum(ml)


def Sharpness(image: np.ndarray):
    """
    Подсчет резкости через дисперсию Лапласа.

    Авторы: Юров П.И.
    Более высокое значение указывает на большую резкость.

    Аргументы:
        image (ndarray): Входное изображение
    """
    
    return cv.Laplacian(image, -1).var()


def blur_complexity(original: np.ndarray, 
                    blurred: np.ndarray) -> float:
    """
    Вычисляет нормированную меру сложности смаза [0, 1] на основе SML.

    Авторы: Беззаборов А.А.

    Аргументы:
        original: Исходное резкое изображение
        blurred: Смазанное изображение

    Возвращает:
        Нормированная мера смаза:
        0 - нет смаза (идеально резкое)
        1 - максимальный смаз (полностью размытое)
    """

    sml_orig = calculate_sml(original)
    sml_blur = calculate_sml(blurred)
    if sml_orig == 0:
        return 1.0
    blur_measure = 1.0 - (sml_blur / sml_orig)
    return np.clip(blur_measure, 0.0, 1.0)


def calculate_snr(signal: np.ndarray, 
                  noise: np.ndarray) -> float:
    """
    Вычисляет отношение сигнал-шум (SNR) в dB.

    Авторы: Беззаборов А.А.

    Аргументы:
        signal: Сигнал (смазанное изображение без шума)
        noise: Матрица добавленного шума

    Возвращает:
        SNR в децибелах (dB)
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    if noise_power < 1e-10:
        return float("inf")
    return 10.0 * np.log10(signal_power / noise_power)


def noise_complexity(signal: np.ndarray, 
                     noise: np.ndarray, 
                     min_snr: float = 0.0, 
                     max_snr: float = 50.0) -> float:
    """
    Вычисляет нормированную меру сложности шума [0, 1] на основе SNR.

    Авторы: Беззаборов А.А.

    Аргументы:
        signal: Сигнал (смазанное изображение без шума)
        noise: Матрица добавленного шума
        min_snr: Минимальный SNR (соответствует сложности 1)
        max_snr: Максимальный SNR (соответствует сложности 0)

    Возвращает:
        Нормированная мера шума:
        0 - нет шума (SNR = ∞)
        1 - максимальный шум (SNR → 0)
    """
    snr_db = calculate_snr(signal, noise)
    if np.isinf(snr_db):
        return 0.0
    noise_measure = 1.0 - ((snr_db - min_snr) / (max_snr - min_snr))

    return np.clip(noise_measure, 0.0, 1.0)
