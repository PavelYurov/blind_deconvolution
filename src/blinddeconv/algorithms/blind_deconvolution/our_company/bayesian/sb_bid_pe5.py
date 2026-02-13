"""
Robust Blind Deconvolution (Amizic + Cho&Lee Hybrid).
Версия 4: Устранение граничных артефактов и "крестов" в ядре.

Изменения:
1. Использование Padding + Crop вместо Edgetaper (убирает серую рамку).
2. Windowing градиентов при оценке ядра (убирает крестообразные артефакты).
3. Очистка ядра через поиск связных компонент (убирает шум вокруг ядра).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, center_of_mass, label
from scipy.signal import convolve2d
import time
from typing import Tuple, List, Any, Dict
import sys
from pathlib import Path

# --- Служебный код для путей ---
def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path

try:
    _CURRENT_FILE = Path(__file__).resolve()
    _PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
    _SRC_DIR = _PROJECT_ROOT / "src"
    _ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"
    for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from blinddeconv.algorithms.base import DeconvolutionAlgorithm
except (RuntimeError, ImportError):
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name
        def process(self, image): pass

# --- Вспомогательные функции ---

def psf2otf(psf, shape):
    """Перевод PSF в OTF с центрированием."""
    if np.all(psf == 0): return np.zeros(shape)
    in_shape = psf.shape
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:in_shape[0], :in_shape[1]] = psf
    for axis, axis_size in enumerate(in_shape):
        psf_padded = np.roll(psf_padded, -int(axis_size / 2), axis=axis)
    return fft2(psf_padded)

def otf2psf(otf, out_shape):
    """Перевод OTF в PSF."""
    psf = np.real(ifft2(otf))
    for axis, axis_size in enumerate(out_shape):
        psf = np.roll(psf, int(axis_size / 2), axis=axis)
    return psf[:out_shape[0], :out_shape[1]]

def adjust_kernel_center(kernel):
    """Центрирование ядра по центру масс."""
    kh, kw = kernel.shape
    cy, cx = center_of_mass(kernel)
    if np.isnan(cy) or np.isnan(cx): return kernel
    shift_y = int(round(kh // 2 - cy))
    shift_x = int(round(kw // 2 - cx))
    return np.roll(np.roll(kernel, shift_y, axis=0), shift_x, axis=1)

def keep_largest_component(kernel):
    """Оставляет только самую большую связную область в ядре (убирает шум)."""
    # Бинаризуем ядро для поиска компонент
    mask = kernel > 0.05 * kernel.max()
    labeled, num_features = label(mask)
    if num_features <= 1:
        return kernel
    
    # Ищем самую большую компоненту
    max_size = 0
    max_label = 0
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if size > max_size:
            max_size = size
            max_label = i
            
    # Зануляем все, что не входит в главную компоненту
    kernel_clean = kernel.copy()
    kernel_clean[labeled != max_label] = 0
    return kernel_clean

def solve_image(y, k, lambda_reg):
    """Восстановление изображения (L2/Wiener)."""
    H = psf2otf(k, y.shape)
    H_conj = np.conj(H)
    h, w = y.shape
    
    # Лапласиан для регуляризации
    dx = np.zeros((h, w)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((h, w)); dy[0, 0] = -1; dy[1, 0] = 1
    FDx = fft2(dx)
    FDy = fft2(dy)
    
    numer = H_conj * fft2(y)
    denom = np.abs(H)**2 + lambda_reg * (np.abs(FDx)**2 + np.abs(FDy)**2)
    x = np.real(ifft2(numer / (denom + 1e-8)))
    return np.maximum(x, 0)

def solve_kernel(y, x, kernel_shape, lambda_k):
    """Оценка ядра."""
    h, w = y.shape
    kh, kw = kernel_shape
    
    # 1. Вычисляем градиенты
    dy_h = np.roll(y, -1, axis=1) - y
    dy_v = np.roll(y, -1, axis=0) - y
    dx_h = np.roll(x, -1, axis=1) - x
    dx_v = np.roll(x, -1, axis=0) - x
    
    # 2. Windowing градиентов (УБИРАЕТ КРЕСТЫ)
    # Накладываем окно на градиенты, чтобы убрать разрывы на границах
    win_h = np.hanning(h)
    win_w = np.hanning(w)
    mask = np.outer(win_h, win_w)
    
    dy_h *= mask; dy_v *= mask
    dx_h *= mask; dx_v *= mask
    
    # 3. FFT
    Y_h = fft2(dy_h); Y_v = fft2(dy_v)
    X_h = fft2(dx_h); X_v = fft2(dx_v)
    
    # 4. Tikhonov regularization
    numer = np.conj(X_h) * Y_h + np.conj(X_v) * Y_v
    denom = np.abs(X_h)**2 + np.abs(X_v)**2 + lambda_k
    
    K_otf = numer / (denom + 1e-8)
    k_est = otf2psf(K_otf, kernel_shape)
    
    return k_est

class SB_BID_PE(DeconvolutionAlgorithm):
    def __init__(self, kernel_shape, lambda_1=0.005, lambda_2=20.0, 
                 max_iterations=10, num_scales=5, verbose=True, **kwargs):
        super().__init__(name='Robust_Blind_Deconv_V4')
        
        # Принудительно нечетный размер ядра
        kh, kw = kernel_shape
        if kh % 2 == 0: kh += 1
        if kw % 2 == 0: kw += 1
        self.kernel_shape = (kh, kw)
        
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iterations = max_iterations
        self.num_scales = num_scales
        self.verbose = verbose
        self.history = {}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Нормализация
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        y_raw = (img - img_min) / (img_max - img_min + 1e-8)
        
        # --- PADDING (Вместо Edgetaper) ---
        # Расширяем изображение, чтобы артефакты ушли за край
        pad_h = self.kernel_shape[0] // 2 + 4
        pad_w = self.kernel_shape[1] // 2 + 4
        # mode='edge' (повторение пикселей) или 'reflect' лучше чем wrap для фото
        y_full = np.pad(y_raw, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # Пирамида
        scales = []
        min_dim = min(self.kernel_shape)
        # Начинаем с масштаба, где ядро ~3-5 пикселей
        min_scale = 5.0 / min_dim 
        if self.num_scales > 1 and min_scale < 1.0:
            scales = np.logspace(np.log10(min_scale), 0, self.num_scales)
        else:
            scales = [1.0]
            
        # Инициализация ядра
        k_curr = np.ones(self.kernel_shape)
        k_curr /= k_curr.sum()
        
        x_curr = None
        
        for scale_idx, scale in enumerate(scales):
            # Ресайз
            H_s = int(np.ceil(y_full.shape[0] * scale))
            W_s = int(np.ceil(y_full.shape[1] * scale))
            y_s = zoom(y_full, (H_s/y_full.shape[0], W_s/y_full.shape[1]), order=3)
            
            # Инициализация X и K на текущем уровне
            if x_curr is None:
                x_curr = y_s.copy()
            else:
                x_curr = zoom(x_curr, (H_s/x_curr.shape[0], W_s/x_curr.shape[1]), order=3)
                # Ядро тоже нужно ресайзить, но аккуратно
                # Для простоты в coarse-to-fine часто просто интерполируют
                k_curr = zoom(k_curr, (self.kernel_shape[0]/k_curr.shape[0], 
                                     self.kernel_shape[1]/k_curr.shape[1]), order=1)
                k_curr[k_curr < 0] = 0
                k_curr /= k_curr.sum()

            if self.verbose:
                print(f"Scale {scale_idx+1}: {y_s.shape}")

            # Итерации
            for it in range(self.max_iterations):
                # 1. Prediction (Отбор градиентов)
                dx = np.roll(x_curr, -1, axis=1) - x_curr
                dy = np.roll(x_curr, -1, axis=0) - x_curr
                mag = np.sqrt(dx**2 + dy**2)
                
                # Жесткий порог для отбора краев
                threshold = max(np.max(mag) * 0.3, 0.02)
                mask = (mag > threshold).astype(float)
                
                # Создаем "острую" версию для оценки ядра
                # (Вместо shock filter просто берем градиенты, но в solve_kernel
                # мы передаем x_curr, поэтому маскирование там неявное через lambda)
                # Для улучшения: можно передать маску в solve_kernel, но это усложнит код.
                # Используем трюк Cho&Lee: обновляем x с очень маленькой lambda (резкий)
                # только для шага ядра.
                x_for_kernel = solve_image(y_s, k_curr, lambda_reg=1e-4) # Очень резкий, много шума
                
                # 2. Оценка ядра
                # Используем x_for_kernel который резкий
                k_est = solve_kernel(y_s, x_for_kernel, self.kernel_shape, self.lambda_2)
                
                # 3. Пост-обработка ядра (Constraints)
                k_est = np.maximum(k_est, 0)
                # Thresholding
                k_est[k_est < 0.05 * k_est.max()] = 0
                # Центрирование
                k_est = adjust_kernel_center(k_est)
                # Связные компоненты (Убираем мусор вокруг)
                k_est = keep_largest_component(k_est)
                
                # Нормировка
                if k_est.sum() > 1e-8:
                    k_est /= k_est.sum()
                else:
                    k_est = k_curr # Откат
                
                k_curr = k_est
                
                # 4. Оценка изображения (Сглаженная для следующей итерации)
                x_curr = solve_image(y_s, k_curr, self.lambda_1)
        
        # --- Финальная деконволюция ---
        if self.verbose:
            print("Final Deconvolution...")
            
        # Используем найденное ядро на исходном (padded) изображении
        # Используем меньшую lambda для финальной резкости
        x_final_padded = solve_image(y_full, k_curr, self.lambda_1 * 0.1)
        
        # --- CROP (Убираем padding) ---
        x_final = x_final_padded[pad_h:-pad_h, pad_w:-pad_w]
        
        # Денормализация
        x_final = x_final * (img_max - img_min) + img_min
        x_final = np.clip(x_final, 0, 255 if img_max > 1 else 1)
        
        self.hyperparams = {'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2}
        return x_final, k_curr

    # Boilerplate
    def get_param(self) -> List[Tuple[str, Any]]: return []
    def change_param(self, params: Dict[str, Any]) -> None: pass
    def get_history(self) -> dict: return {}
    def get_hyperparams(self) -> dict: return self.hyperparams

def sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs):
    algo = SB_BID_PE(kernel_shape=kernel_shape, **kwargs)
    return algo.process(y)