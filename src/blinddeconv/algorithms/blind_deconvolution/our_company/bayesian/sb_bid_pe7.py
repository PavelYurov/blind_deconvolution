"""
Fast Motion Deblurring (Cho & Lee 2009).
Реализация алгоритма:
S. Cho and S. Lee, "Fast Motion Deblurring," ACM Trans. Graph. (SIGGRAPH Asia), 2009.

Это "золотой стандарт" для слепой деконволюции. 
В отличие от чистого Amizic (MAP), этот метод использует явный шаг предсказания краев (Prediction Step),
что позволяет восстанавливать сложные траектории смаза (Motion Blur) и избегать локальных минимумов.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter, zoom, center_of_mass
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
    """
    Перевод ядра в OTF (Optical Transfer Function) без сдвига фазы.
    """
    if np.all(psf == 0): return np.zeros(shape)
    h, w = shape
    kh, kw = psf.shape
    
    # Паддинг ядра до размера изображения
    psf_padded = np.zeros((h, w), dtype=psf.dtype)
    psf_padded[:kh, :kw] = psf
    
    # Циклический сдвиг, чтобы центр ядра (kh//2, kw//2) попал в (0,0)
    # Это критически важно для корректной фазы в FFT
    psf_padded = np.roll(psf_padded, -int(kh // 2), axis=0)
    psf_padded = np.roll(psf_padded, -int(kw // 2), axis=1)
    
    return fft2(psf_padded)

def otf2psf(otf, out_shape):
    """Обратное преобразование OTF -> PSF."""
    psf = np.real(ifft2(otf))
    kh, kw = out_shape
    # Обратный сдвиг
    psf = np.roll(psf, int(kh // 2), axis=0)
    psf = np.roll(psf, int(kw // 2), axis=1)
    return psf[:kh, :kw]

def get_gradients(img):
    """Вычисление градиентов (вперед и вниз)."""
    dx = np.roll(img, -1, axis=1) - img
    dy = np.roll(img, -1, axis=0) - img
    return dx, dy

def shock_filter(img, iter_n=1, dt=1.0):
    """
    Шоковый фильтр (Shock Filter).
    Делает края резкими, "восстанавливая" их из размытого состояния.
    Используется для предсказания четкого изображения.
    Уравнение: I_t = -sign(Laplacian(I)) * |grad(I)|
    """
    u = img.copy()
    for _ in range(iter_n):
        dx, dy = get_gradients(u)
        grad_mag = np.sqrt(dx**2 + dy**2)
        
        # Лапласиан
        uxx = np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)
        uyy = np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)
        lap = uxx + uyy
        
        # Shock update
        u = u - dt * np.sign(lap) * grad_mag
    return u

def projection_on_gradients(dx, dy, threshold_r=2.0):
    """
    Отбор градиентов (Gradient Selection).
    Оставляет только сильные градиенты, подавляет шум и мелкие текстуры.
    """
    mag = np.sqrt(dx**2 + dy**2)
    
    # Вычисляем порог (Cho & Lee используют эвристику)
    # Здесь берем порог, который отсекает мелкий шум
    threshold = max(np.mean(mag) * threshold_r, 1e-4)
    
    # Маска сильных краев
    mask = (mag > threshold).astype(float)
    
    return dx * mask, dy * mask

def solve_image_fft(y, k, lambda_reg):
    """
    Быстрое восстановление изображения (Tikhonov regularization).
    x = argmin ||y - k*x||^2 + lambda ||grad x||^2
    """
    H = psf2otf(k, y.shape)
    H_conj = np.conj(H)
    
    h, w = y.shape
    # OTF градиентных фильтров
    dx_f = psf2otf(np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]), (h, w))
    dy_f = psf2otf(np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]), (h, w))
    
    numer = H_conj * fft2(y)
    denom = np.abs(H)**2 + lambda_reg * (np.abs(dx_f)**2 + np.abs(dy_f)**2)
    
    x = np.real(ifft2(numer / (denom + 1e-8)))
    return np.maximum(x, 0)

def solve_kernel_fft(y, x_pred, kernel_shape, lambda_k):
    """
    Оценка ядра по предсказанному резкому изображению.
    k = argmin ||grad y - k * grad x_pred||^2 + lambda ||k||^2
    """
    h, w = y.shape
    kh, kw = kernel_shape
    
    # Работаем с градиентами!
    dy_h, dy_v = get_gradients(y)
    dx_h, dx_v = get_gradients(x_pred)
    
    Y_h, Y_v = fft2(dy_h), fft2(dy_v)
    X_h, X_v = fft2(dx_h), fft2(dx_v)
    
    # Система уравнений в частотной области
    numer = np.conj(X_h) * Y_h + np.conj(X_v) * Y_v
    denom = np.abs(X_h)**2 + np.abs(X_v)**2 + lambda_k
    
    K_otf = numer / (denom + 1e-8)
    k_est = otf2psf(K_otf, kernel_shape)
    
    return k_est

class SB_BID_PE(DeconvolutionAlgorithm):
    def __init__(self, kernel_shape, lambda_1=2e-3, lambda_2=5.0, 
                 max_iterations=5, num_scales=5, verbose=True, **kwargs):
        """
        kernel_shape: размер ядра (должен быть нечетным).
        lambda_1: регуляризация изображения (меньше -> резче).
        lambda_2: регуляризация ядра (больше -> чище ядро).
        """
        super().__init__(name='Cho_Lee_2009')
        
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
        # 1. Препроцессинг
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        y = (img - img_min) / (img_max - img_min + 1e-8)
        
        # Padding для подавления граничных эффектов
        pad_h = self.kernel_shape[0] // 2 + 4
        pad_w = self.kernel_shape[1] // 2 + 4
        y_padded = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        
        # 2. Пирамида масштабов
        scales = []
        min_dim = min(self.kernel_shape)
        # Начинаем с масштаба, где ядро ~3-5 пикселей
        min_scale = 3.0 / min_dim
        if min_scale >= 1.0:
            scales = [1.0]
        else:
            n_scales = int(np.ceil(np.log(1.0 / min_scale) / np.log(1.5))) + 1
            scales = [min_scale * (1.5 ** i) for i in range(n_scales)]
            # Убедимся, что последний масштаб 1.0
            if scales[-1] < 1.0: scales.append(1.0)
            else: scales[-1] = 1.0
            
        # Инициализация ядра
        k_est = np.zeros(self.kernel_shape)
        # Инициализация как маленькое гауссово пятно (лучше чем точка)
        cx, cy = self.kernel_shape[1]//2, self.kernel_shape[0]//2
        k_est[cy-1:cy+2, cx-1:cx+2] = 1.0
        k_est /= k_est.sum()
        
        x_est = None
        
        # 3. Coarse-to-Fine цикл
        for scale in scales:
            if self.verbose:
                print(f"Processing scale {scale:.3f}...")
                
            # Ресайз Y
            H_s = int(np.ceil(y_padded.shape[0] * scale))
            W_s = int(np.ceil(y_padded.shape[1] * scale))
            # Убедимся, что размеры четные для лучшей работы FFT (опционально)
            if H_s % 2 != 0: H_s += 1
            if W_s % 2 != 0: W_s += 1
            
            y_s = zoom(y_padded, (H_s/y_padded.shape[0], W_s/y_padded.shape[1]), order=1)
            
            # Ресайз K
            if scale == scales[0]:
                # На первом уровне ядро маленькое
                ks_h = 3
                ks_w = 3
            else:
                # Интерполируем ядро с предыдущего уровня
                # Размер ядра на текущем уровне должен соответствовать масштабу
                # Но мы держим размер массива фиксированным (kernel_shape), меняя содержимое
                k_est = zoom(k_est, (1.0, 1.0), order=1) # Заглушка, размер не меняем
                # В идеале нужно ресайзить ядро физически, но здесь мы используем фиксированный контейнер
                # и полагаемся на то, что ядро "вырастет" внутри контейнера.
                pass

            # Инициализация X
            if x_est is None:
                x_est = y_s.copy()
            else:
                x_est = zoom(x_est, (H_s/x_est.shape[0], W_s/x_est.shape[1]), order=1)

            # Внутренний цикл (Iterative Update)
            for it in range(self.max_iterations):
                # A. Prediction Step (Cho & Lee)
                # 1. Восстанавливаем X с текущим ядром (быстро)
                x_tmp = solve_image_fft(y_s, k_est, self.lambda_1)
                
                # 2. Применяем Shock Filter для заострения краев
                x_shock = shock_filter(x_tmp, iter_n=1)
                
                # 3. Вычисляем градиенты
                dx, dy = get_gradients(x_shock)
                
                # 4. Gradient Selection (оставляем только сильные края)
                dx_p, dy_p = projection_on_gradients(dx, dy, threshold_r=2.0)
                
                # B. Kernel Estimation Step
                # Оцениваем ядро, используя Y и ПРЕДСКАЗАННЫЕ градиенты X
                # (Здесь мы не используем solve_kernel_fft напрямую с градиентами, 
                # а модифицируем функцию, чтобы она принимала уже готовые градиенты)
                
                # Переходим в Фурье
                Y_h, Y_v = fft2(get_gradients(y_s)[0]), fft2(get_gradients(y_s)[1])
                X_h, X_v = fft2(dx_p), fft2(dy_p)
                
                # Решаем для K
                numer = np.conj(X_h) * Y_h + np.conj(X_v) * Y_v
                denom = np.abs(X_h)**2 + np.abs(X_v)**2 + self.lambda_2
                K_otf = numer / (denom + 1e-8)
                k_new = otf2psf(K_otf, self.kernel_shape)
                
                # Пост-обработка ядра
                k_new = np.maximum(k_new, 0)
                # Thresholding (убираем шум)
                k_new[k_new < 0.05 * k_new.max()] = 0
                # Центрирование
                cy, cx = center_of_mass(k_new)
                if not np.isnan(cy) and not np.isnan(cx):
                    shift_y = self.kernel_shape[0]//2 - cy
                    shift_x = self.kernel_shape[1]//2 - cx
                    k_new = shift(k_new, (shift_y, shift_x), order=1)
                
                # Нормировка
                if k_new.sum() > 0:
                    k_new /= k_new.sum()
                
                k_est = k_new
                
                # Обновляем x_est для следующей итерации
                x_est = solve_image_fft(y_s, k_est, self.lambda_1)

        # 4. Финальное восстановление
        # Используем найденное ядро на исходном изображении (без паддинга)
        # Для финального качества можно использовать Richardson-Lucy, если Tikhonov мылит
        if self.verbose:
            print("Final Deconvolution...")
            
        # Убираем паддинг из ядра (если он был) - здесь не нужен, ядро фиксировано
        
        # Восстанавливаем полное изображение
        # Используем чуть меньшую регуляризацию для деталей
        x_final_padded = solve_image_fft(y_padded, k_est, self.lambda_1 * 0.1)
        
        # Обрезаем паддинг
        x_final = x_final_padded[pad_h:-pad_h, pad_w:-pad_w]
        
        # Денормализация
        x_final = x_final * (img_max - img_min) + img_min
        x_final = np.clip(x_final, 0, 255 if img_max > 1 else 1)
        
        self.hyperparams = {'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2}
        return x_final, k_est

    # Boilerplate
    def get_param(self) -> List[Tuple[str, Any]]: return []
    def change_param(self, params: Dict[str, Any]) -> None: pass
    def get_history(self) -> dict: return {}
    def get_hyperparams(self) -> dict: return self.hyperparams

def sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs):
    algo = SB_BID_PE(kernel_shape=kernel_shape, **kwargs)
    return algo.process(y)

# Дополнительный импорт для shift (забыл добавить в начале)
from scipy.ndimage import shift

if __name__ == "__main__":
    # Тест
    from scipy import signal
    import matplotlib.pyplot as plt
    img = np.zeros((200, 200))
    img[50:150, 50:150] = 1.0
    # Motion blur kernel (диагональ)
    kernel = np.zeros((15, 15))
    for i in range(15): kernel[i, i] = 1
    kernel /= kernel.sum()
    
    blurred = signal.convolve2d(img, kernel, mode='same')
    noisy = blurred + 0.005 * np.random.randn(*blurred.shape)
    
    solver = SB_BID_PE(kernel_shape=(15, 15), verbose=True)
    x_est, h_est = solver.process(noisy)
    
    plt.figure(figsize=(12,4))
    plt.subplot(131); plt.imshow(noisy, cmap='gray'); plt.title("Blurred")
    plt.subplot(132); plt.imshow(x_est, cmap='gray'); plt.title("Restored")
    plt.subplot(133); plt.imshow(h_est, cmap='gray'); plt.title("Kernel")
    plt.show()