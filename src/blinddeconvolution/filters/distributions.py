"""
Функции распределения для генерации ядер PSF.

Автор: Беззаборов А.А., Юров П.И.
"""

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.stats import multivariate_normal
import cv2


def generate_multivariate_normal_kernel(ksize: int, cov: list) -> np.ndarray:
    """
    Генерирует 2D-ядро на основе многомерного нормального распределения.
    
    Форма, размер и поворот ядра полностью определяются матрицей ковариации.

    Параметры
    ---------
    ksize : int
        Размер выходного ядра (должен быть нечетным).
    cov : list
        Матрица ковариации.

    Возвращает
    ----------
    np.ndarray
        Нормализованное 2D ядро размытия.
    """
    center = ksize // 2
    x, y = np.meshgrid(np.arange(ksize) - center, np.arange(ksize) - center)

    pos = np.dstack((x, y))

    rv = multivariate_normal(mean=[0, 0], cov=cov)

    kernel = rv.pdf(pos)

    if kernel.sum() > 0:
        return kernel / kernel.sum()
    return kernel


def generate_bspline_motion_kernel(ksize: int, points: list, thickness: int = 3) -> np.ndarray:
    """
    Генерирует ядро размытия в движении по кривой, заданной B-сплайном.

    Параметры
    ---------
    ksize : int
        Размер выходного ядра (должен быть нечетным).
    points : list
        Список контрольных точек [(x1, y1), (x2, y2), ...],
        заданных относительно центра ядра (0,0).
    thickness : int
        Толщина кривой в пикселях.

    Возвращает
    ----------
    np.ndarray
        Нормализованное 2D ядро размытия.
    """
    if ksize % 2 == 0:
        ksize += 1

    center = ksize // 2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    points_np = np.array(points).T
    tck, u = splprep(points_np, s=0, k=min(3, len(points) - 1))
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_coords, y_coords = splev(u_new, tck)

    x_abs = np.round(x_coords + center).astype(int)
    y_abs = np.round(y_coords + center).astype(int)

    valid_indices = (x_abs >= 0) & (x_abs < ksize) & (y_abs >= 0) & (y_abs < ksize)
    unique_pixels = np.unique(np.vstack((y_abs[valid_indices], x_abs[valid_indices])).T, axis=0)

    for y, x in unique_pixels:
        kernel[y, x] = 1.0

    if thickness > 1:
        thickness = thickness if thickness % 2 == 1 else thickness + 1
        kernel = cv2.GaussianBlur(kernel, (thickness, thickness), 0)

    if kernel.sum() > 0:
        return kernel / kernel.sum()
    return kernel


def gaussian_distribution(x: np.ndarray, std: float) -> np.ndarray:
    """
    Гауссовская функция распределения.
    
    Применение:
    - Для DefocusBlur: передаем 2D радиус (x = sqrt(x² + y²)).
    - Для MotionBlur: передаем 1D координаты вдоль направления движения.
    
    Параметры
    ---------
    x : np.ndarray
        Входной массив расстояний/координат.
    std : float
        Стандартное отклонение.
        
    Возвращает
    ----------
    np.ndarray
        Ненормализованные значения распределения.
    """
    return np.exp(-x**2 / (2 * std**2))


def uniform_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Равномерная функция распределения.
    
    Применение:
    - Для DefocusBlur: создает диск (disk_psf).
    - Для MotionBlur: создает прямоугольное размытие.
    
    Параметры
    ---------
    x : np.ndarray
        Входной массив расстояний/координат.
    radius : float
        Радиус/полуширина распределения.
        
    Возвращает
    ----------
    np.ndarray
        Ненормализованные значения распределения.
    """
    return (x <= radius).astype(float)


def linear_decay_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Универсальная линейно убывающая функция распределения.
    
    Применение:
    - Для DefocusBlur: создает конус (cone_psf).
    - Для MotionBlur: создает треугольное размытие.
    
    Параметры
    ---------
    x : np.ndarray
        Входной массив расстояний/координат.
    radius : float
        Радиус/полуширина распределения.
        
    Возвращает
    ----------
    np.ndarray
        Ненормализованные значения распределения.
    """
    return np.clip(1 - x/radius, 0, None)


def ring_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Кольцевое распределение (специфично для размытия вне фокуса).
    
    Параметры
    ---------
    x : np.ndarray
        Входной массив расстояний.
    radius : float
        Радиус кольца.
        
    Возвращает
    ----------
    np.ndarray
        Ненормализованные значения распределения.
    """
    return np.exp(-(x - radius)**2 / (0.1 * radius**2))


def exponential_decay_distribution(x: np.ndarray, scale: float) -> np.ndarray:
    """
    Экспоненциально убывающее распределение (специфично для размытия в движении).
    
    Параметры
    ---------
    x : np.ndarray
        1D массив координат вдоль направления движения.
    scale : float
        Параметр масштаба.
        
    Возвращает
    ----------
    np.ndarray
        Ненормализованные значения распределения.
    """
    return np.exp(-np.abs(x)/scale)
