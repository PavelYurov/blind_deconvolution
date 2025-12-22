import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Callable, Optional, Tuple


def PSNR(original: np.ndarray, 
         restored: np.ndarray) -> float:
    """
    Вычисляет отношение пикового сигнала к шуму (PSNR) между изображениями.

    Аргументы:
        original (ndarray): Исходное изображение
        restored (ndarray): Восстановленное/обработанное изображение

    Возвращает:
        Значение PSNR в децибелах (dB)
    """

    return peak_signal_noise_ratio(original, restored)


def SSIM(original: np.ndarray, 
         restored: np.ndarray, 
         data_range: Optional[float] = None) -> float:
    """
    Вычисляет индекс структурного сходства (SSIM) между изображениями.

    Аргументы:
        original: Исходное изображение
        restored: Восстановленное/обработанное изображение
        data_range (Optional[float]): Верхний предел значений

    Возвращает:
        Значение SSIM в диапазоне от 0 до 1
    """

    return structural_similarity(original, restored, data_range=data_range)


def calculate_sml(image: np.ndarray) -> float:
    """
    Вычисляет Sum of Modified Laplacian (SML) для изображения.
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
    Более высокое значение указывает на большую резкость.

    Аргументы:
        image (ndarray): Входное изображение
    """
    
    return cv.Laplacian(image, -1).var()


def blur_complexity(original: np.ndarray, 
                    blurred: np.ndarray) -> float:
    """
    Вычисляет нормированную меру сложности смаза [0, 1] на основе SML.

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
