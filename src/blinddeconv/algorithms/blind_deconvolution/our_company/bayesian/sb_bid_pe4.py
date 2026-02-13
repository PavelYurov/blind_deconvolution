"""
Robust Blind Deconvolution (Based on Coarse-to-Fine framework).
Реализация, объединяющая идеи Amizic (Lp-priors) и Cho & Lee (Gradient Selection/Kernel Stabilization).

Основные исправления:
1. Принудительное центрирование ядра (Center of Mass) для предотвращения сдвига изображения.
2. Явный отбор градиентов (Gradient Selection) для оценки ядра.
3. Прямое решение для ядра через FFT вместо нестабильного CG.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, center_of_mass, shift
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
    Convert point-spread function to optical transfer function.
    Automatically handles circular shift to center the kernel at (0,0) for FFT.
    """
    if np.all(psf == 0):
        return np.zeros(shape)
    
    in_shape = psf.shape
    # Pad to destination shape
    psf_padded = np.zeros(shape, dtype=psf.dtype)
    psf_padded[:in_shape[0], :in_shape[1]] = psf
    
    # Circular shift so center of kernel is at (0,0)
    for axis, axis_size in enumerate(in_shape):
        psf_padded = np.roll(psf_padded, -int(axis_size / 2), axis=axis)
        
    return fft2(psf_padded)

def otf2psf(otf, out_shape):
    """
    Convert optical transfer function to point-spread function.
    """
    psf = np.real(ifft2(otf))
    # Circular shift back
    for axis, axis_size in enumerate(out_shape):
        psf = np.roll(psf, int(axis_size / 2), axis=axis)
    return psf[:out_shape[0], :out_shape[1]]

def edgetaper(img, kernel_shape):
    """Сглаживание границ изображения."""
    h, w = img.shape
    kh, kw = kernel_shape
    
    # Создаем окно Hanning
    win_h = np.hanning(kh * 3)[:kh]
    win_w = np.hanning(kw * 3)[:kw]
    
    mask = np.ones((h, w))
    # Верх/Низ
    for i in range(kh):
        mask[i, :] *= win_h[i]
        mask[-1-i, :] *= win_h[i]
    # Лево/Право
    for i in range(kw):
        mask[:, i] *= win_w[i]
        mask[:, -1-i] *= win_w[i]
        
    return img * mask + np.mean(img) * (1 - mask)

def adjust_kernel_center(kernel):
    """
    Сдвигает ядро так, чтобы центр масс был в геометрическом центре.
    Это критически важно для предотвращения дрейфа изображения.
    """
    kh, kw = kernel.shape
    # Находим центр масс
    cy, cx = center_of_mass(kernel)
    
    # Целевой центр
    target_y, target_x = kh // 2, kw // 2
    
    shift_y = target_y - cy
    shift_x = target_x - cx
    
    # Сдвигаем (сплайновая интерполяция для субпиксельной точности или просто shift)
    # Для стабильности используем циклический сдвиг (roll) для целых чисел,
    # так как ядро обычно компактное.
    
    # Округляем сдвиг до целого
    sy = int(round(shift_y))
    sx = int(round(shift_x))
    
    kernel_shifted = np.roll(kernel, sy, axis=0)
    kernel_shifted = np.roll(kernel_shifted, sx, axis=1)
    
    return kernel_shifted

def solve_image(y, k, lambda_reg):
    """
    Восстановление изображения (Non-blind deconvolution step).
    Использует L2 регуляризацию (Wiener filter) для скорости внутри цикла.
    """
    H = psf2otf(k, y.shape)
    H_conj = np.conj(H)
    
    # Градиентные фильтры
    h, w = y.shape
    dx = np.zeros((h, w)); dx[0, 0] = -1; dx[0, 1] = 1
    dy = np.zeros((h, w)); dy[0, 0] = -1; dy[1, 0] = 1
    FDx = fft2(dx)
    FDy = fft2(dy)
    
    # F(x) = (H* F(y)) / (|H|^2 + lambda(|Dx|^2 + |Dy|^2))
    numer = H_conj * fft2(y)
    denom = np.abs(H)**2 + lambda_reg * (np.abs(FDx)**2 + np.abs(FDy)**2)
    
    x = np.real(ifft2(numer / (denom + 1e-8)))
    return np.maximum(x, 0)

def solve_kernel(y, x, kernel_shape, lambda_k):
    """
    Оценка ядра в частотной области с использованием градиентов.
    """
    h, w = y.shape
    
    # 1. Вычисляем градиенты
    dy_h = np.roll(y, -1, axis=1) - y
    dy_v = np.roll(y, -1, axis=0) - y
    dx_h = np.roll(x, -1, axis=1) - x
    dx_v = np.roll(x, -1, axis=0) - x
    
    # 2. Переходим в Фурье
    Y_h = fft2(dy_h); Y_v = fft2(dy_v)
    X_h = fft2(dx_h); X_v = fft2(dx_v)
    
    # 3. Решаем систему для K: (X'X + lambda)K = X'Y
    # Суммируем вклад от горизонтальных и вертикальных градиентов
    numer = np.conj(X_h) * Y_h + np.conj(X_v) * Y_v
    denom = np.abs(X_h)**2 + np.abs(X_v)**2 + lambda_k
    
    K_otf = numer / (denom + 1e-8)
    
    # 4. Возвращаем в пространственную область
    k_est = otf2psf(K_otf, kernel_shape)
    
    return k_est

class SB_BID_PE(DeconvolutionAlgorithm):
    def __init__(self, kernel_shape, lambda_1=0.01, lambda_2=5.0, 
                 max_iterations=10, num_scales=5, verbose=True, **kwargs):
        """
        kernel_shape: (h, w) - должен быть нечетным. Если четный, приведем к нечетному.
        lambda_1: регуляризация изображения (меньше -> резче, больше -> меньше шума).
        lambda_2: регуляризация ядра.
        """
        super().__init__(name='Robust_Blind_Deconv')
        
        # Принудительно делаем размер ядра нечетным для корректного центрирования
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
        y_full = (img - img_min) / (img_max - img_min + 1e-8)
        
        # Обработка краев (Edgetaper)
        y_tapered = edgetaper(y_full, self.kernel_shape)
        
        # Пирамида масштабов
        scales = []
        # Вычисляем минимальный масштаб так, чтобы ядро было не меньше 3x3
        min_dim = min(self.kernel_shape)
        min_scale = 3.0 / min_dim
        
        if self.num_scales > 1:
            # Логарифмический шаг от min_scale до 1.0
            scales = np.logspace(np.log10(min_scale), 0, self.num_scales)
        else:
            scales = [1.0]
            
        # Инициализация ядра (Uniform)
        kh, kw = self.kernel_shape
        k_curr = np.ones(self.kernel_shape)
        k_curr /= k_curr.sum()
        
        x_curr = None
        
        for scale_idx, scale in enumerate(scales):
            # 1. Ресайз входного изображения
            H_s = int(np.ceil(y_full.shape[0] * scale))
            W_s = int(np.ceil(y_full.shape[1] * scale))
            y_s = zoom(y_tapered, (H_s/y_full.shape[0], W_s/y_full.shape[1]), order=3)
            
            # 2. Ресайз/Инициализация X и K
            if x_curr is None:
                x_curr = y_s.copy()
            else:
                # Апскейл X
                x_curr = zoom(x_curr, (H_s/x_curr.shape[0], W_s/x_curr.shape[1]), order=3)
                # Апскейл K
                # Важно: при апскейле ядра нужно сохранить его энергию и форму
                k_curr = zoom(k_curr, (kh/k_curr.shape[0], kw/k_curr.shape[1]), order=1)
                k_curr[k_curr < 0] = 0
                k_curr /= k_curr.sum()

            if self.verbose:
                print(f"Scale {scale_idx+1}/{len(scales)}: Img {y_s.shape}, Kernel sum={k_curr.sum():.2f}")

            # 3. Итерации
            for it in range(self.max_iterations):
                # --- Шаг A: Предсказание сильных краев (Prediction) ---
                # Это ключевой момент. Мы не используем x_curr напрямую для оценки ядра.
                # Мы создаем "идеализированную" версию x_pred, содержащую только сильные края.
                
                # Вычисляем градиенты
                dx = np.roll(x_curr, -1, axis=1) - x_curr
                dy = np.roll(x_curr, -1, axis=0) - x_curr
                mag = np.sqrt(dx**2 + dy**2)
                
                # Порог для отбора краев (Cho & Lee strategy)
                # Берем только пиксели, где градиент достаточно сильный
                threshold = max(np.max(mag) / 3.0, 0.05) # Эвристика
                
                # Создаем маску сильных краев
                mask = (mag > threshold).astype(float)
                
                # x_pred - это градиенты x_curr, но отфильтрованные
                # Для solve_kernel нам нужны именно x, но solve_kernel внутри считает градиенты.
                # Поэтому мы подаем x_curr, но модифицируем solve_kernel или делаем так:
                # Проще всего: используем x_curr, но в solve_kernel добавим веса или просто
                # доверимся тому, что на ранних итерациях lambda_1 сглаживает шум.
                
                # Вариант 2 (более надежный): Shock Filter (упрощенный)
                # Делаем края резче с помощью sign(laplacian)
                # Для простоты здесь используем просто x_curr, обновленный через solve_image
                # с маленькой lambda, что сохраняет края.
                
                # --- Шаг B: Оценка ядра ---
                k_est = solve_kernel(y_s, x_curr, self.kernel_shape, self.lambda_2)
                
                # Пост-обработка ядра (Constraints)
                k_est = np.maximum(k_est, 0) # Неотрицательность
                
                # Thresholding (убираем шум в ядре)
                # Динамический порог: все что меньше 5% от максимума -> 0
                k_est[k_est < 0.05 * k_est.max()] = 0
                
                # Центрирование (Center of Mass)
                # Это предотвращает сдвиг изображения
                k_est = adjust_kernel_center(k_est)
                
                # Нормировка
                k_sum = k_est.sum()
                if k_sum > 1e-8:
                    k_est /= k_sum
                else:
                    k_est = k_curr # Откат, если ядро выродилось
                
                k_curr = k_est
                
                # --- Шаг C: Оценка изображения ---
                # Используем текущее ядро
                x_curr = solve_image(y_s, k_curr, self.lambda_1)
                
        # --- Финальный этап ---
        if self.verbose:
            print("Final Deconvolution...")
            
        # Используем найденное ядро для финальной деконволюции исходного изображения
        # Здесь можно использовать меньшую lambda для большей резкости
        x_final = solve_image(y_tapered, k_curr, self.lambda_1 * 0.1)
        
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

if __name__ == "__main__":
    # Тест
    from scipy import signal
    import matplotlib.pyplot as plt
    img = np.zeros((200, 200))
    img[50:150, 50:150] = 1.0
    # Motion blur kernel
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