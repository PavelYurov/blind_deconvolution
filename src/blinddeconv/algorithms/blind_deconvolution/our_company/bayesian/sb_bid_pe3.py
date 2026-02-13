"""
Разреженная байесовская слепая деконволюция с оценкой параметров (Amizic 2012).
Исправленная и стабилизированная версия.

Ссылки:
1. Amizic, B., et al. (2012). "Sparse Bayesian blind image deconvolution..."
2. Levin, A., et al. (2009). "Understanding and evaluating blind deconvolution algorithms" 
   (Объясняет необходимость правильной инициализации и почему MAP сходится к дельта-функции).
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, gaussian_filter
from scipy.sparse.linalg import LinearOperator, cg
import time
from typing import Tuple, List, Any, Dict
import sys
from pathlib import Path

# --- Служебный код для путей (не меняем) ---
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

def edgetaper(img, kernel_shape):
    """
    Сглаживает края изображения, чтобы уменьшить звон (ringing artifacts) в FFT.
    Аналог MATLAB edgetaper.
    """
    h, w = img.shape
    kh, kw = kernel_shape
    
    # Создаем окно, затухающее к краям
    alpha = 0.5
    
    # Hanning window для краев
    win_h = np.hanning(kh * 2)[:kh] # левая часть
    win_w = np.hanning(kw * 2)[:kw]
    
    # Маска весов
    mask = np.ones((h, w))
    
    # Затухание сверху/снизу
    for i in range(kh):
        mask[i, :] *= win_h[i]
        mask[-1-i, :] *= win_h[i]
        
    # Затухание слева/справа
    for i in range(kw):
        mask[:, i] *= win_w[i]
        mask[:, -1-i] *= win_w[i]
        
    # Размытая версия изображения (среднее значение или гаусс)
    blurred = gaussian_filter(img, sigma=3)
    
    return img * mask + blurred * (1 - mask)

def _compute_gradient_h(x):
    """Горизонтальный градиент (циклический)."""
    return np.roll(x, -1, axis=1) - x

def _compute_gradient_v(x):
    """Вертикальный градиент (циклический)."""
    return np.roll(x, -1, axis=0) - x

def _compute_divergence_h(p):
    """Сопряженный оператор к gradient_h."""
    return np.roll(p, 1, axis=1) - p

def _compute_divergence_v(p):
    """Сопряженный оператор к gradient_v."""
    return np.roll(p, 1, axis=0) - p

def _pad_kernel_for_fft(h, image_shape):
    H, W = image_shape
    kh, kw = h.shape
    h_padded = np.zeros((H, W), dtype=h.dtype)
    h_padded[:kh, :kw] = h
    # Центрируем ядро в (0,0) для корректной фазы
    h_padded = np.roll(h_padded, -kh//2, axis=0)
    h_padded = np.roll(h_padded, -kw//2, axis=1)
    return h_padded

def _extract_kernel_from_padded(h_padded, kernel_shape):
    kh, kw = kernel_shape
    # Обратный сдвиг
    shifted = np.roll(h_padded, kh//2, axis=0)
    shifted = np.roll(shifted, kw//2, axis=1)
    return shifted[:kh, :kw]

def _update_weights(x, p, epsilon):
    """Веса W = p/2 * (|grad|^2 + eps)^(p/2 - 1)."""
    gh = _compute_gradient_h(x)
    gv = _compute_gradient_v(x)
    power = (p / 2.0) - 1.0
    w_h = (p / 2.0) * (gh**2 + epsilon)**power
    w_v = (p / 2.0) * (gv**2 + epsilon)**power
    return w_h, w_v

def _solve_cg_image(y, H_fft, w_h, w_v, alpha, beta, shape):
    """Решение для изображения x с использованием CG."""
    N = shape[0] * shape[1]
    Y_fft = fft2(y)
    # Правая часть: beta * H^T * y
    rhs = np.real(ifft2(beta * np.conj(H_fft) * Y_fft)).flatten()
    
    def matvec(v_flat):
        v = v_flat.reshape(shape)
        # 1. Data term: beta * H^T * H * v
        term1 = np.real(ifft2(beta * (np.abs(H_fft)**2) * fft2(v)))
        
        # 2. Prior term: alpha * div(W * grad(v))
        # Заметьте: в функционале энергии стоит +alpha*|grad|^p.
        # Градиент энергии: alpha * D^T * W * D * x. 
        # D^T (дивергенция) в наших функциях уже имеет правильный знак для сопряжения.
        # Но математически div = -D^T.
        # Уравнение: (beta H^T H + alpha D^T W D) x = ...
        # D^T W D x = -div(W grad x).
        # Наши функции _compute_divergence реализуют именно D^T (сопряженный), 
        # который равен (p[i-1] - p[i]). Это соответствует -div в непрерывном смысле.
        
        gh = _compute_gradient_h(v)
        gv = _compute_gradient_v(v)
        
        # D^T * W * D
        term2_h = _compute_divergence_h(w_h * gh)
        term2_v = _compute_divergence_v(w_v * gv)
        
        term2 = alpha * (term2_h + term2_v)
        return (term1 + term2).flatten()

    A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    x_est, _ = cg(A, rhs, x0=y.flatten(), atol=1e-4, maxiter=20)
    return x_est.reshape(shape)

def _solve_cg_kernel(y, x, w_u, gamma, beta, k_shape, img_shape):
    """Решение для ядра h с использованием CG."""
    N_k = k_shape[0] * k_shape[1]
    X_fft = fft2(x)
    Y_fft = fft2(y)
    
    # Правая часть: beta * X^T * y
    # Обрезаем до размера ядра
    rhs_full = np.real(ifft2(beta * np.conj(X_fft) * Y_fft))
    rhs = _extract_kernel_from_padded(rhs_full, k_shape).flatten()
    
    def matvec(h_flat):
        h = h_flat.reshape(k_shape)
        
        # 1. Data term: beta * X^T * X * h
        h_pad = _pad_kernel_for_fft(h, img_shape)
        term1_full = np.real(ifft2(beta * (np.abs(X_fft)**2) * fft2(h_pad)))
        term1 = _extract_kernel_from_padded(term1_full, k_shape)
        
        # 2. Prior term: gamma * D^T * W * D * h (TV)
        gh = _compute_gradient_h(h)
        gv = _compute_gradient_v(h)
        
        # Для TV веса одинаковы для обоих направлений
        term2 = gamma * (_compute_divergence_h(w_u * gh) + 
                         _compute_divergence_v(w_u * gv))
        
        return (term1 + term2).flatten()
        
    A = LinearOperator((N_k, N_k), matvec=matvec, dtype=np.float64)
    
    # Старт с равномерного распределения, чтобы не застрять в нуле
    h0 = np.ones(N_k) / N_k
    h_est, _ = cg(A, rhs, x0=h0, atol=1e-4, maxiter=15)
    return h_est.reshape(k_shape)

class SB_BID_PE(DeconvolutionAlgorithm):
    def __init__(self, kernel_shape, p=0.8, lambda_1=10.0, lambda_2=1e-3, 
                 max_iterations=20, tolerance: float = 1e-4, num_scales=3, verbose=True, **kwargs):
        """
        lambda_1: увеличен дефолт, чтобы подавить шум на ранних этапах.
        """
        super().__init__(name='Amizic2012')
        self.kernel_shape = tuple(kernel_shape)
        self.p = p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iterations = max_iterations
        self.num_scales = num_scales
        self.verbose = verbose
        self.tolerance = tolerance
        self.history = {}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Нормализация
        img = image.astype(np.float64)
        img_min, img_max = img.min(), img.max()
        if img_max - img_min < 1e-6:
            return image, np.zeros(self.kernel_shape)
        
        y_full = (img - img_min) / (img_max - img_min)
        H_orig, W_orig = y_full.shape
        
        # Обработка краев (критично для деконволюции!)
        y_tapered = edgetaper(y_full, self.kernel_shape)
        
        # Пирамида
        scales = []
        ratio = 0.75  # ~ 1 / sqrt(2)
        for s in range(self.num_scales):
            scales.append(ratio ** (self.num_scales - 1 - s))
            
        # Инициализация
        # ВАЖНО: Инициализируем ядро не точкой, а небольшим пятном или равномерным,
        # чтобы избежать локального минимума "нет размытия".
        kh, kw = self.kernel_shape
        h = np.ones((kh, kw))
        h /= h.sum()
        
        x = None
        
        # Начальные параметры (эмпирические, будут обновлены)
        alpha = 0.1
        beta = 100.0
        gamma = 100.0
        
        self.history = {'alpha': [], 'beta': [], 'gamma': []}

        for scale_idx, scale in enumerate(scales):
            # Точный расчет размеров для текущего масштаба
            H_s = int(np.ceil(H_orig * scale))
            W_s = int(np.ceil(W_orig * scale))
            
            # Масштабирование y
            # Используем order=3 (bicubic) для лучшего качества даунскейла
            y_s = zoom(y_tapered, (H_s / H_orig, W_s / W_orig), order=3)
            
            # Масштабирование x (если есть с предыдущего шага)
            if x is None:
                x_s = y_s.copy()
            else:
                x_s = zoom(x, (H_s / x.shape[0], W_s / x.shape[1]), order=3)
            
            # Масштабирование ядра (интерполяция ядра важна!)
            # Но размер ядра фиксирован в задаче (kernel_shape).
            # В coarse-to-fine подходах обычно размер ядра тоже меняется.
            # Если мы держим размер ядра фиксированным (как в параметрах функции),
            # то на грубых масштабах ядро должно быть "сжато".
            # Однако, в классическом Amizic размер ядра фиксирован.
            # Мы просто используем текущее h как начальное приближение.
            # Сброс в uniform на самом грубом масштабе часто помогает.
            if scale_idx == 0:
                 # На самом грубом масштабе - Uniform initialization
                 h_s = np.ones(self.kernel_shape) / np.prod(self.kernel_shape)
            else:
                 h_s = h.copy()

            if self.verbose:
                print(f"Scale {scale_idx+1}: {y_s.shape}, Kernel: {h_s.shape}")

            # Внутренний цикл
            for it in range(self.max_iterations):
                x_prev = x_s.copy()
                
                # 1. Веса изображения (Lp)
                w_h, w_v = _update_weights(x_s, self.p, epsilon=1e-6)
                
                # 2. Обновление X
                h_pad = _pad_kernel_for_fft(h_s, (H_s, W_s))
                H_fft = fft2(h_pad)
                x_s = _solve_cg_image(y_s, H_fft, w_h, w_v, alpha, beta, (H_s, W_s))
                x_s = np.maximum(x_s, 0.0)
                x_s = np.minimum(x_s, 1.0)
                
                # 3. Веса ядра (TV)
                w_u = _update_weights(h_s, p=1.0, epsilon=1e-6)[0] # w_u общий
                
                # 4. Обновление H
                h_s = _solve_cg_kernel(y_s, x_s, w_u, gamma, beta, self.kernel_shape, (H_s, W_s))
                
                # Проекция ядра
                h_s = np.maximum(h_s, 0.0)
                # Убираем шум (thresholding) - помогает от дельта-решения
                h_s[h_s < 0.05 * h_s.max()] = 0 
                h_sum = h_s.sum()
                if h_sum > 1e-8:
                    h_s /= h_sum
                else:
                    h_s = np.ones_like(h_s) / h_s.size
                
                # 5. Обновление параметров (Amizic Eqs 16-18)
                # Alpha
                grads_x = np.abs(_compute_gradient_h(x_s))**self.p + np.abs(_compute_gradient_v(x_s))**self.p
                alpha_new = (self.lambda_1 * x_s.size) / (self.p * np.sum(grads_x) + 1e-6)
                # Демпфирование изменений параметров для стабильности
                alpha = 0.5 * alpha + 0.5 * alpha_new
                
                # Beta
                # Важно: если beta растет слишком быстро, мы сваливаемся в y=x.
                # Ограничим beta сверху на первых итерациях
                Hx = np.real(ifft2(fft2(_pad_kernel_for_fft(h_s, (H_s, W_s))) * fft2(x_s)))
                mse = np.mean((y_s - Hx)**2)
                beta_new = 1.0 / (mse + 1e-6)
                # Ограничитель роста beta
                beta_new = min(beta_new, 1000.0 * (scale_idx + 1)) 
                beta = 0.5 * beta + 0.5 * beta_new
                
                # Gamma
                grads_h = np.sqrt(_compute_gradient_h(h_s)**2 + _compute_gradient_v(h_s)**2)
                gamma_new = (self.lambda_2 * h_s.size) / (np.sum(grads_h) + 1e-6)
                gamma = 0.5 * gamma + 0.5 * gamma_new
                
                self.history['alpha'].append(alpha)
                self.history['beta'].append(beta)
                self.history['gamma'].append(gamma)
                
                # Сходимость
                diff = np.linalg.norm(x_s - x_prev) / (np.linalg.norm(x_prev) + 1e-9)
                if self.verbose and it % 5 == 0:
                    print(f"  Iter {it}: diff={diff:.1e}, a={alpha:.1e}, b={beta:.1e}, g={gamma:.1e}")
                
                if diff < self.tolerance:
                    break
            
            x = x_s
            h = h_s
            
        # Финальный результат: деконволюция на полном разрешении с найденным ядром
        # Используем не-слепую деконволюцию с найденным h для лучшего качества
        # (Один финальный проход с фиксированным h и более жесткими ограничениями)
        if self.verbose:
            print("Final non-blind deconvolution step...")
            
        # Восстанавливаем x на полном разрешении, используя найденное h
        # h не меняем, только x
        h_final = h.copy()
        h_pad = _pad_kernel_for_fft(h_final, (H_orig, W_orig))
        H_fft = fft2(h_pad)
        
        # Используем чуть более строгую регуляризацию для финала
        alpha_final = alpha * 0.5 
        beta_final = beta * 2.0
        
        # Стартуем с апскейленного x
        x_final = zoom(x, (H_orig / x.shape[0], W_orig / x.shape[1]), order=3)
        
        for _ in range(10): # Несколько итераций для уточнения деталей
            w_h, w_v = _update_weights(x_final, self.p, epsilon=1e-6)
            x_final = _solve_cg_image(y_tapered, H_fft, w_h, w_v, alpha_final, beta_final, (H_orig, W_orig))
            x_final = np.maximum(x_final, 0.0)
            x_final = np.minimum(x_final, 1.0)

        # Денормализация
        x_final = x_final * (img_max - img_min) + img_min
        
        self.hyperparams = {'alpha': alpha, 'beta': beta, 'gamma': gamma}
        return x_final, h_final

    # --- Boilerplate для фреймворка ---
    def get_param(self) -> List[Tuple[str, Any]]:
        return [('kernel_shape', self.kernel_shape), ('p', self.p), 
                ('lambda_1', self.lambda_1), ('lambda_2', self.lambda_2),
                ('max_iterations', self.max_iterations), ('num_scales', self.num_scales),
                ('verbose', self.verbose)]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        if 'kernel_shape' in params: self.kernel_shape = tuple(params['kernel_shape'])
        if 'p' in params: self.p = float(params['p'])
        if 'lambda_1' in params: self.lambda_1 = float(params['lambda_1'])
        if 'lambda_2' in params: self.lambda_2 = float(params['lambda_2'])
        if 'max_iterations' in params: self.max_iterations = int(params['max_iterations'])
        if 'num_scales' in params: self.num_scales = int(params['num_scales'])
        if 'verbose' in params: self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict: return self.history
    def get_hyperparams(self) -> dict: return self.hyperparams

def sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs):
    algo = SB_BID_PE(kernel_shape=kernel_shape, **kwargs)
    return algo.process(y)

if __name__ == "__main__":
    # Тестовый запуск
    from scipy import signal
    import matplotlib.pyplot as plt
    img = np.zeros((128, 128))
    img[30:100, 30:100] = 1.0; img[50:80, 50:80] = 0.0
    kernel = np.eye(7) / 7.0 # Motion blur
    blurred = signal.convolve2d(img, kernel, mode='same')
    noisy = blurred + 0.001 * np.random.randn(*blurred.shape)
    
    solver = SB_BID_PE(kernel_shape=(7, 7), p=0.8, verbose=True)
    x_est, h_est = solver.process(noisy)
    
    plt.figure(figsize=(10,4))
    plt.subplot(131); plt.imshow(noisy, cmap='gray'); plt.title("Input")
    plt.subplot(132); plt.imshow(x_est, cmap='gray'); plt.title("Restored")
    plt.subplot(133); plt.imshow(h_est, cmap='gray'); plt.title("Kernel")
    plt.show()