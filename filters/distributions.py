import numpy as np
from scipy.interpolate import splev, splprep
from scipy.stats import multivariate_normal
from scipy.interpolate import splev, splprep
import cv2


def generate_multivariate_normal_kernel(ksize: int, cov: list, **kwargs) -> np.ndarray:
    """
    Генерирует 2D-ядро на основе многомерного нормального (гауссова) распределения.
    
    Форма, размер и поворот ядра полностью определяются матрицей ковариации.
    """
    center = ksize // 2
    x, y = np.meshgrid(np.arange(ksize) - center, np.arange(ksize) - center)
    
    # Упаковываем координаты в формат, понятный для scipy
    pos = np.dstack((x, y))
    
    # Создаем объект двумерного нормального распределения
    rv = multivariate_normal(mean=[0, 0], cov=cov)
    
    # Вычисляем функцию плотности вероятности (PDF) для каждой точки нашей сетки
    kernel = rv.pdf(pos)
    
    # Нормализуем ядро
    if kernel.sum() > 0:
        return kernel / kernel.sum()
    return kernel

def generate_bspline_motion_kernel(ksize: int, points: list, thickness: int = 3, **kwargs) -> np.ndarray:
    """
    Генерирует ядро размытия в движении по кривой, заданной B-сплайном.

    Аргументы:
        ksize (int): Размер выходного ядра (должен быть нечетным).
        points (list): Список контрольных точек [(x1, y1), (x2, y2), ...],
                       заданных относительно центра ядра (0,0).
        thickness (int): Толщина кривой в пикселях.
        **kwargs: Дополнительные аргументы для совместимости.

    Возвращает:
        Нормализованное 2D ядро размытия.
    """
    if ksize % 2 == 0:
        ksize += 1  # Убедимся, что размер нечетный

    center = ksize // 2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)

    # 1. Подготовка контрольных точек
    # Транспонируем и преобразуем в numpy-массив
    points_np = np.array(points).T
    
    # 2. Вычисление B-сплайна
    # splprep находит параметрическое представление сплайна
    # k=3 означает кубический сплайн, наиболее распространенный
    # s=0 означает, что кривая должна пройти точно через все точки
    tck, u = splprep(points_np, s=0, k=min(3, len(points) - 1))
    
    # 3. "Прорисовка" кривой в ядре
    # Генерируем 1000 точек вдоль кривой для гладкости
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_coords, y_coords = splev(u_new, tck)
    
    # Переводим относительные координаты в абсолютные индексы массива
    # и округляем, чтобы "закрасить" пиксели
    x_abs = np.round(x_coords + center).astype(int)
    y_abs = np.round(y_coords + center).astype(int)
    
    # Убираем дубликаты и выходы за границы
    valid_indices = (x_abs >= 0) & (x_abs < ksize) & (y_abs >= 0) & (y_abs < ksize)
    unique_pixels = np.unique(np.vstack((y_abs[valid_indices], x_abs[valid_indices])).T, axis=0)
    
    # Заполняем ядро
    for y, x in unique_pixels:
        kernel[y, x] = 1.0

    # 4. Применение толщины и нормализация
    if thickness > 1:
        # Убедимся, что толщина нечетная
        thickness = thickness if thickness % 2 == 1 else thickness + 1
        kernel = cv2.GaussianBlur(kernel, (thickness, thickness), 0)

    if kernel.sum() > 0:
        return kernel / kernel.sum()
    return kernel

def gaussian_distribution(x: np.ndarray, std: float) -> np.ndarray:
    """
    Гауссовская функция распределения.
    
    Применение:
    - Для DefocusBlur: передаем 2D радиус (x = sqrt(x² + y²))
    - Для MotionBlur: передаем 1D координаты вдоль направления движения
    
    Параметры:
        r: Входной массив расстояний/координат
        std: Стандартное отклонение
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-x**2 / (2 * std**2))

def uniform_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Равномерная функция распределения.
    
    Применение:
    - Для DefocusBlur: создает диск (disk_psf)
    - Для MotionBlur: создает прямоугольное размытие
    
    Параметры:
        x: Входной массив расстояний/координат
        radius: Радиус/полуширина распределения
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return (x <= radius).astype(float)

def linear_decay_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Универсальная линейно убывающая функция распределения.
    
    Применение:
    - Для DefocusBlur: создает конус (cone_psf)
    - Для MotionBlur: создает треугольное размытие
    
    Параметры:
        x: Входной массив расстояний/координат
        radius: Радиус/полуширина распределения
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.clip(1 - x/radius, 0, None)

def ring_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Кольцевое распределение (специфично для размытия вне фокуса).
    
    Параметры:
        x: Входной массив расстояний
        radius: Радиус кольца
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-(x - radius)**2 / (0.1 * radius**2))

def exponential_decay_distribution(x: np.ndarray, scale: float) -> np.ndarray:
    """
    Экспоненциально убывающее распределение (специфично для размытия в движении).
    
    Параметры:
        x: 1D массив координат вдоль направления движения
        scale: Параметр масштаба
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-np.abs(x)/scale)