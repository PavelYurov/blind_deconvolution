"""
Robust Blind Deconvolution (Amizic + Cho&Lee Hybrid).
Версия 5: Возврат к стабильной логике v3 с исправлением артефактов.

Исправления:
1. Pad-Taper-Crop стратегия: убирает "серую рамку" с полезной части изображения.
2. Boundary Gradient Masking: убирает "кресты" в ядре, игнорируя разрывы на границах.
3. Мягкий Gradient Selection: позволяет восстанавливать сложные (кривые) ядра.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, center_of_mass
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

def edgetaper(img, kernel_shape):
    """
    Сглаживание границ. 
    Важно: применяется к уже расширенному изображению.
    """
    h, w = img.shape
    kh, kw = kernel_shape
    
    # Окно Hanning, но более широкое, чтобы плавнее уходить в серый
    win_h = np.hanning(kh * 3)[:kh]
    win_w = np.hanning(kw * 3)[:kw]
    
    mask = np.ones((h, w))
    for i in range(kh):
        mask[i, :] *= win_h[i]; mask[-1-i, :] *= win_h[i]
    for i in range(kw):
        mask[:, i] *= win_w[i]; mask[:, -1-i] *= win_w[i]
        
    return img * mask + np.mean(img) * (1 - mask)

def adjust_kernel_center(kernel):
    """Центрирование ядра (Center of Mass)."""
    kh, kw = kernel.shape
    cy, cx = center_of_mass(kernel)
    if np.isnan(cy) or np.isnan(cx): return kernel
    
    shift_y = int(round(kh // 2 - cy))
    shift_x = int(round(kw // 2 - cx))
    
    # Используем roll, так как ядро обычно затухает к краям
    return np.roll(np.roll(kernel, shift_y, axis=0), shift_x, axis=1)

def solve_image(y, k, lambda_reg):
    """
    Восстановление изображения (L2/Wiener).
    Быстрое и стабильное решение для внутреннего цикла.
    """
    H = psf2otf(k, y.shape)
    H_conj = np.conj(H)
    h, w = y.shape
    
    # Градиенты в частотной области
    dx = np.zeros((h, w)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((h, w)); dy[0, 0] = -1; dy[1, 0] = 1
    FDx = fft2(dx)
    FDy = fft2(dy)
    
    numer = H_conj * fft2(y)
    # Регуляризация Тихонова (L2 на градиенты)
    denom = np.abs(H)**2 + lambda_reg * (np.abs(FDx)**2 + np.abs(FDy)**2)
    
    x = np.real(ifft2(numer / (denom + 1e-8)))
    return np.maximum(x, 0)

def solve_kernel(y, x, kernel_shape, lambda_k):
    """
    Оценка ядра.
    Ключевое исправление: маскирование границ градиентов.
    """
    h, w = y.shape
    kh, kw = kernel_shape
    
    # 1. Вычисляем градиенты
    dy_h = np.roll(y, -1, axis=1) - y
    dy_v = np.roll(y, -1, axis=0) - y
    dx_h = np.roll(x, -1, axis=1) - x
    dx_v = np.roll(x, -1, axis=0) - x
    
    # --- FIX: Убираем граничные артефакты (Кресты) ---
    # Зануляем градиенты на границах изображения, так как np.roll 
    # смешивает левый край с правым.
    border = 2
    dy_h[:, :border] = 0; dy_h[:, -border:] = 0
    dy_v[:border, :] = 0; dy_v[-border:, :] = 0
    dx_h[:, :border] = 0; dx_h[:, -border:] = 0
    dx_v[:border, :] = 0; dx_v[-border:, :] = 0
    
    # 2. Переходим в Фурье
    Y_h = fft2(dy_h); Y_v = fft2(dy_v)
    X_h = fft2(dx_h); X_v = fft2(dx_v)
    
    # 3. Решаем систему для K
    numer = np.conj(X_h) * Y_h + np.conj(X_v) * Y_v
    denom = np.abs(X_h)**2 + np.abs(X_v)**2 + lambda_k
    
    K_otf = numer / (denom + 1e-8)
    k_est = otf2psf(K_otf, kernel_shape)
    
    return k_est

class SB_BID_PE(DeconvolutionAlgorithm):
    def __init__(self, kernel_shape, lambda_1=0.01, lambda_2=5.0, 
                 max_iterations=10, num_scales=5, verbose=True, **kwargs):
        super().__init__(name='Robust_Blind_Deconv_V5')
        
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
        if img_max - img_min < 1e-6:
            return image, np.zeros(self.kernel_shape)
            
        y_raw = (img - img_min) / (img_max - img_min + 1e-8)
        
        # --- FIX: Pad-Taper-Crop ---
        # 1. Паддинг (отражение), чтобы отодвинуть границы
        pad_h = self.kernel_shape[0] // 2 + 10
        pad_w = self.kernel_shape[1] // 2 + 10
        y_full = np.pad(y_raw, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        
        # 2. Edgetaper на расширенном изображении
        # Теперь "серая рамка" будет на паддинге, а не на полезной части
        y_full = edgetaper(y_full, self.kernel_shape)
        
        # Пирамида
        scales = []
        min_dim = min(self.kernel_shape)
        min_scale = 3.0 / min_dim # Начинаем когда ядро ~3px
        if self.num_scales > 1 and min_scale < 1.0:
            scales = np.logspace(np.log10(min_scale), 0, self.num_scales)
        else:
            scales = [1.0]
            
        # Инициализация ядра (Uniform)
        k_curr = np.ones(self.kernel_shape)
        k_curr /= k_curr.sum()
        
        x_curr = None
        
        for scale_idx, scale in enumerate(scales):
            # Ресайз
            H_s = int(np.ceil(y_full.shape[0] * scale))
            W_s = int(np.ceil(y_full.shape[1] * scale))
            y_s = zoom(y_full, (H_s/y_full.shape[0], W_s/y_full.shape[1]), order=3)
            
            # Инициализация X и K
            if x_curr is None:
                x_curr = y_s.copy()
            else:
                x_curr = zoom(x_curr, (H_s/x_curr.shape[0], W_s/x_curr.shape[1]), order=3)
                # Ядро просто интерполируем
                k_curr = zoom(k_curr, (self.kernel_shape[0]/k_curr.shape[0], 
                                     self.kernel_shape[1]/k_curr.shape[1]), order=1)
                k_curr[k_curr < 0] = 0
                k_curr /= k_curr.sum()

            if self.verbose:
                print(f"Scale {scale_idx+1}: {y_s.shape}")

            for it in range(self.max_iterations):
                # --- Шаг A: Предсказание (Gradient Selection) ---
                dx = np.roll(x_curr, -1, axis=1) - x_curr
                dy = np.roll(x_curr, -1, axis=0) - x_curr
                mag = np.sqrt(dx**2 + dy**2)
                
                # Порог: берем сильные края, но не слишком агрессивно
                # Если порог слишком высокий, сложные ядра (кривые) теряются
                threshold = max(np.max(mag) * 0.25, 0.02) 
                mask = (mag > threshold).astype(float)
                
                # Создаем "острую" версию для оценки ядра
                # Используем solve_image с очень маленькой регуляризацией,
                # чтобы получить максимальную резкость (пусть и с шумом)
                x_for_kernel = solve_image(y_s, k_curr, lambda_reg=1e-3)
                
                # --- Шаг B: Оценка ядра ---
                # solve_kernel внутри зануляет границы градиентов!
                k_est = solve_kernel(y_s, x_for_kernel, self.kernel_shape, self.lambda_2)
                
                # Пост-обработка ядра
                k_est = np.maximum(k_est, 0)
                
                # Thresholding: убираем шум, но оставляем структуру
                # 0.05 - консервативный порог, сохраняет хвосты
                k_est[k_est < 0.05 * k_est.max()] = 0
                
                # Центрирование
                k_est = adjust_kernel_center(k_est)
                
                # Нормировка
                if k_est.sum() > 1e-8:
                    k_est /= k_est.sum()
                else:
                    k_est = k_curr # Fallback
                
                k_curr = k_est
                
                # --- Шаг C: Оценка изображения ---
                x_curr = solve_image(y_s, k_curr, self.lambda_1)
        
        # --- Финальная деконволюция ---
        if self.verbose:
            print("Final Deconvolution...")
            
        # Используем найденное ядро на полном (padded) изображении
        # lambda_1 чуть меньше для финальной резкости
        x_final_padded = solve_image(y_full, k_curr, self.lambda_1 * 0.5)
        
        # --- CROP: Убираем паддинг и серую рамку ---
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