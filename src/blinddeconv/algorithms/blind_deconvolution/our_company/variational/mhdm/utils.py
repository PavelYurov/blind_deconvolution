"""
Вспомогательные функции для MHDM Blind Deconvolution.
Реализация весов норм Соболева, паддинга и пост-обработки ядра.
"""

import numpy as np
from scipy.ndimage import zoom, label, center_of_mass, shift

def compute_sobolev_weights(shape: tuple, order: float) -> np.ndarray:
    """
    Вычисляет веса дискретной нормы Соболева в частотной области.
    """
    m, n = shape
    i = np.arange(m).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    
    term_i = 2 * (m**2) * (1 - np.cos(2 * np.pi * i / m))
    term_j = 2 * (n**2) * (1 - np.cos(2 * np.pi * j / n))
    
    delta = 1.0 + term_i + term_j
    
    return np.power(delta, order)

def normalize_min_max(x: np.ndarray) -> np.ndarray:
    """Нормализует массив в диапазон [0, 1]."""
    xmin, xmax = x.min(), x.max()
    if xmax > xmin + 1e-8:
        return (x - xmin) / (xmax - xmin)
    return x

def pad_image(image: np.ndarray, pad_size: int, mode: str = 'reflect') -> np.ndarray:
    """
    Расширяет изображение.
    """
    return np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), mode=mode)

def crop_center(image: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Вырезает центральную часть изображения."""
    h, w = image.shape
    th, tw = target_shape
    start_y = (h - th) // 2
    start_x = (w - tw) // 2
    return image[start_y:start_y+th, start_x:start_x+tw]

def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Ресайз изображения (бикубическая интерполяция)."""
    return zoom(image, scale, order=3, prefilter=True)

def resize_kernel(kernel: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Ресайз ядра с сохранением энергии.
    """
    current_h, current_w = kernel.shape
    target_h, target_w = target_shape
    
    zoom_h = target_h / current_h
    zoom_w = target_w / current_w
    
    k_new = zoom(kernel, (zoom_h, zoom_w), order=1, prefilter=False)
    
    k_sum = k_new.sum()
    if k_sum > 1e-12:
        k_new /= k_sum
        
    return k_new

def edgetaper(image: np.ndarray, kernel_shape: tuple) -> np.ndarray:
    """Сглаживает края изображения."""
    from scipy.signal import fftconvolve
    h, w = image.shape
    kh, kw = kernel_shape
    
    wy = np.ones(h)
    if h > kh:
        idx = np.arange(kh)
        vals = 0.5 * (1 - np.cos(np.pi * idx / (kh - 1)))
        wy[:kh] = vals
        wy[-kh:] = vals[::-1]
        
    wx = np.ones(w)
    if w > kw:
        idx = np.arange(kw)
        vals = 0.5 * (1 - np.cos(np.pi * idx / (kw - 1)))
        wx[:kw] = vals
        wx[-kw:] = vals[::-1]
        
    alpha = np.outer(wy, wx)
    
    sigma = min(kh, kw) / 5.0
    y_grid, x_grid = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
    gauss = np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))
    gauss /= gauss.sum()
    
    blurred = fftconvolve(image, gauss, mode='same')
    return alpha * image + (1 - alpha) * blurred

def process_kernel_spatial(k: np.ndarray, threshold_ratio: float = 0.05) -> np.ndarray:
    """
    Очистка ядра в пространственной области:
    1. Hard Thresholding (убирает жирность/туман).
    2. Оставление только самой большой компоненты связности (убирает шум вдалеке).
    3. Центрирование (Center of Mass).
    """
    # 1. Thresholding
    k_max = k.max()
    # Обнуляем все, что меньше 5-10% от пика
    mask = k > (k_max * threshold_ratio)
    k_clean = k * mask
    
    # 2. Largest Connected Component (если ядро разбилось на части)
    labeled, n_components = label(mask)
    if n_components > 1:
        sizes = [np.sum(labeled == i) for i in range(1, n_components + 1)]
        largest_label = np.argmax(sizes) + 1
        k_clean[labeled != largest_label] = 0
    
    # 3. Centering
    # Вычисляем центр масс
    if k_clean.sum() > 1e-12:
        cy, cx = center_of_mass(k_clean)
        h, w = k.shape
        dy = (h // 2) - cy
        dx = (w // 2) - cx
        
        # Сдвигаем ядро в центр массива
        k_clean = shift(k_clean, (dy, dx), order=0, mode='constant', cval=0)
    
    # Нормализация
    k_sum = k_clean.sum()
    if k_sum > 1e-12:
        k_clean /= k_sum
        
    return k_clean