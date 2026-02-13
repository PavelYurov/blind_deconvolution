"""
Разреженная байесовская слепая деконволюция с оценкой параметров.
Реализация алгоритма на основе статьи:
Amizic, B., Molina, R., & Katsaggelos, A. K. (2012).
"Sparse Bayesian blind image deconvolution with parameter estimation."

Исправления:
1. Устранена ошибка размерности (broadcasting error) путем явного расчета коэффициентов масштабирования.
2. Использован итеративный решатель (CG) для шагов обновления изображения и ядра,
   так как прямая инверсия в частотной области нестабильна для p < 1.
3. Сохранены интерфейсы фреймворка.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom
from scipy.sparse.linalg import LinearOperator, cg
import time
from typing import Tuple, List, Any, Dict

import sys
from pathlib import Path

# --- Блок сохранения путей фреймворка (как в оригинале) ---
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
    # Фолбек для автономного запуска, если не в структуре проекта
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name
        def process(self, image): pass

# --- Константы и вспомогательные функции ---

DEFAULT_P = 0.8           # Amizic рекомендует p < 1 (обычно 0.8)
DEFAULT_LAMBDA_1 = 2/3    # Начальное значение (будет обновляться)
DEFAULT_LAMBDA_2 = 1e-3   # Начальное значение
EPSILON = 1e-6            # Для численной стабильности весов

def _compute_gradient_h(x):
    """Горизонтальный градиент (разности вперед) с периодическими граничными условиями."""
    grad = np.zeros_like(x)
    grad[:, :-1] = x[:, 1:] - x[:, :-1]
    grad[:, -1] = x[:, 0] - x[:, -1]
    return grad

def _compute_gradient_v(x):
    """Вертикальный градиент (разности вперед)."""
    grad = np.zeros_like(x)
    grad[:-1, :] = x[1:, :] - x[:-1, :]
    grad[-1, :] = x[0, :] - x[-1, :]
    return grad

def _compute_divergence_h(p):
    """Сопряженный оператор к gradient_h (div = -grad^T)."""
    div = np.zeros_like(p)
    div[:, 1:] = p[:, :-1] - p[:, 1:]
    div[:, 0] = p[:, -1] - p[:, 0]
    return div

def _compute_divergence_v(p):
    """Сопряженный оператор к gradient_v."""
    div = np.zeros_like(p)
    div[1:, :] = p[:-1, :] - p[1:, :]
    div[0, :] = p[-1, :] - p[0, :]
    return div

def _pad_kernel_for_fft(h, image_shape):
    """Паддинг ядра до размера изображения и циклический сдвиг."""
    H, W = image_shape
    kh, kw = h.shape
    h_padded = np.zeros((H, W), dtype=h.dtype)
    h_padded[:kh, :kw] = h
    # Центрируем для корректной фазы FFT (центр ядра в (0,0))
    h_padded = np.roll(h_padded, -kh//2, axis=0)
    h_padded = np.roll(h_padded, -kw//2, axis=1)
    return h_padded

def _extract_kernel_from_padded(h_padded, kernel_shape):
    """Извлечение ядра после обратного сдвига."""
    kh, kw = kernel_shape
    shifted = np.roll(h_padded, kh//2, axis=0)
    shifted = np.roll(shifted, kw//2, axis=1)
    return shifted[:kh, :kw]

def _update_weights_image(x, p, epsilon=EPSILON):
    """
    Вычисление весов W_d для Lp нормы изображения.
    Формула (10) в статье: w = (p/2) * (Delta_x^2 + eps)^(p/2 - 1)
    """
    gh = _compute_gradient_h(x)
    gv = _compute_gradient_v(x)
    
    # power = p/2 - 1. При p=0.8 power = -0.6.
    # Если градиент мал, вес становится большим -> сохранение краев.
    power = (p / 2.0) - 1.0
    
    w_h = (p / 2.0) * (gh**2 + epsilon)**power
    w_v = (p / 2.0) * (gv**2 + epsilon)**power
    
    return w_h, w_v

def _update_weights_kernel(h, epsilon=EPSILON):
    """
    Вычисление весов для TV нормы ядра (p=1).
    Формула (12) в статье.
    """
    gh = _compute_gradient_h(h)
    gv = _compute_gradient_v(h)
    
    # Для TV веса общие для обоих направлений: 1 / sqrt(grad^2 + eps)
    denom = np.sqrt(gh**2 + gv**2 + epsilon)
    w = 1.0 / np.maximum(denom, epsilon)
    return w

def _solve_image_cg(y, H_fft, w_h, w_v, alpha, beta, image_shape):
    """
    Решает систему для x: (beta * H^T H + alpha * L_w) x = beta * H^T y
    Использует Conjugate Gradient, так как матрица L_w не диагональна в Фурье.
    """
    H_rows, W_cols = image_shape
    N = H_rows * W_cols
    
    # Правая часть: beta * H^T * y
    # H^T в частотной области это conj(H)
    Y_fft = fft2(y)
    RHS_fft = beta * np.conj(H_fft) * Y_fft
    rhs = np.real(ifft2(RHS_fft)).flatten()
    
    # Оператор левой части A*v
    def matvec(v_flat):
        v = v_flat.reshape(image_shape)
        
        # 1. Data term: beta * H^T * H * v
        V_fft = fft2(v)
        term1_fft = beta * (np.abs(H_fft)**2) * V_fft
        term1 = np.real(ifft2(term1_fft))
        
        # 2. Prior term: alpha * D^T * W * D * v
        # Это реализация div(W * grad(v))
        gh = _compute_gradient_h(v)
        wh_gh = w_h * gh
        div_h = _compute_divergence_h(wh_gh)
        
        gv = _compute_gradient_v(v)
        wv_gv = w_v * gv
        div_v = _compute_divergence_v(wv_gv)
        
        # Оператор дивергенции уже включает знак минус по отношению к Лапласиану,
        # но в функционале энергии стоит + alpha * |grad|^p.
        # Уравнение Эйлера-Лагранжа: beta H^T(Hx-y) - alpha div(W grad x) = 0
        # => (beta H^T H - alpha div W grad) x = beta H^T y
        # Наши функции divergence реализуют сопряженный оператор к градиенту (D^T).
        # D^T = -div. Значит term2 = alpha * (D_h^T W_h D_h + D_v^T W_v D_v)
        
        term2 = alpha * (div_h + div_v)
        
        return (term1 + term2).flatten()

    A = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
    
    # Начальное приближение - наблюдаемое изображение
    x_est, info = cg(A, rhs, x0=y.flatten(), atol=1e-5, maxiter=50)
    
    return x_est.reshape(image_shape)

def _solve_blur_cg(y, x, w_u, gamma, beta, kernel_shape, image_shape):
    """
    Решает систему для h: (beta * X^T X + gamma * L_u) h = beta * X^T y
    """
    kh, kw = kernel_shape
    N_k = kh * kw
    
    X_fft = fft2(x)
    Y_fft = fft2(y)
    
    # Правая часть: beta * X^T * y
    # X^T * y (свертка с перевернутым x) -> conj(X)*Y
    RHS_spatial = np.real(ifft2(beta * np.conj(X_fft) * Y_fft))
    # Извлекаем область ядра (с учетом сдвига)
    rhs = _extract_kernel_from_padded(RHS_spatial, kernel_shape).flatten()
    
    def matvec(h_flat):
        h_curr = h_flat.reshape(kernel_shape)
        
        # 1. Data term: beta * X^T * X * h
        # Свертка h с автокорреляцией x
        h_pad = _pad_kernel_for_fft(h_curr, image_shape)
        H_f = fft2(h_pad)
        term1_fft = beta * (np.abs(X_fft)**2) * H_f
        term1_full = np.real(ifft2(term1_fft))
        term1 = _extract_kernel_from_padded(term1_full, kernel_shape)
        
        # 2. Prior term: gamma * D^T * W * D * h (TV)
        gh = _compute_gradient_h(h_curr)
        gv = _compute_gradient_v(h_curr)
        
        # Для TV веса одинаковы для H и V
        term2 = gamma * (_compute_divergence_h(w_u * gh) + 
                         _compute_divergence_v(w_u * gv))
        
        return (term1 + term2).flatten()
        
    A = LinearOperator((N_k, N_k), matvec=matvec, dtype=np.float64)
    
    # Начальное приближение - равномерное или дельта
    h0 = np.zeros(N_k)
    h0[N_k//2] = 1.0
    
    h_est, info = cg(A, rhs, x0=h0, atol=1e-5, maxiter=20)
    
    return h_est.reshape(kernel_shape)

class SB_BID_PE(DeconvolutionAlgorithm):
    """
    Реализация алгоритма Amizic 2012.
    Использует многомасштабный подход и MM (Majorization-Minimization).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        p: float = DEFAULT_P,
        lambda_1: float = DEFAULT_LAMBDA_1,
        lambda_2: float = DEFAULT_LAMBDA_2,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        num_scales: int = 3,
        use_multiscale: bool = True,
        verbose: bool = False,
        # Параметры интерфейса (не используются напрямую в логике, но нужны для совместимости)
        use_spatial_solver: bool = True,
        max_cg_iters: int = 50,
        cg_tol: float = 1e-4
    ):
        super().__init__(name='Amizic2012')
        
        self.kernel_shape = tuple(kernel_shape)
        self.p = p
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.num_scales = num_scales if use_multiscale else 1
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}
    
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # 1. Нормализация изображения [0, 1] (критично для Lp норм)
        img_min, img_max = image.min(), image.max()
        if img_max - img_min < 1e-8:
            # Если изображение однотонное, возвращаем как есть
            return image, np.ones(self.kernel_shape)/np.prod(self.kernel_shape)
            
        y_full = (image - img_min) / (img_max - img_min)
        H_full, W_full = y_full.shape
        
        # 2. Подготовка пирамиды масштабов
        scales = []
        if self.num_scales > 1:
            # Геометрическая прогрессия масштабов
            ratio = (3/2) ** (1/2) # Более мягкий шаг чем 2
            for s in range(self.num_scales):
                scales.append(1.0 / (ratio ** (self.num_scales - 1 - s)))
        else:
            scales = [1.0]
            
        # Инициализация переменных
        # На первом масштабе x инициализируется масштабированным y
        x = None 
        h = np.zeros(self.kernel_shape)
        # Инициализация ядра дельта-функцией
        kh, kw = self.kernel_shape
        h[kh//2, kw//2] = 1.0
        
        # Начальные значения гиперпараметров (будут обновлены на первой итерации)
        alpha = 1e-2
        beta = 100.0  # Высокая точность данных сначала
        gamma = 1e-2
        
        self.history = {'alpha': [], 'beta': [], 'gamma': []}

        # --- Цикл по масштабам ---
        for scale_idx, scale in enumerate(scales):
            if self.verbose:
                print(f"Scale {scale_idx+1}/{len(scales)} (factor {scale:.2f})")
            
            # --- ИСПРАВЛЕНИЕ ОШИБКИ РАЗМЕРНОСТИ ---
            # Вычисляем целевые размеры
            H_s = int(H_full * scale)
            W_s = int(W_full * scale)
            
            # Явно указываем zoom факторы, чтобы получить ровно (H_s, W_s)
            zoom_h = H_s / H_full
            zoom_w = W_s / W_full
            
            y_s = zoom(y_full, (zoom_h, zoom_w), order=1)
            
            # Ресайз текущей оценки x к новому масштабу
            if x is None:
                x_s = y_s.copy()
            else:
                # Масштабируем предыдущий x до текущего размера y_s
                x_s = zoom(x, (H_s / x.shape[0], W_s / x.shape[1]), order=1)
                
            # Ресайз ядра (ядро обычно маленькое, можно не менять размер или менять аккуратно)
            # В Amizic ядро обычно фиксированного размера, но для coarse-to-fine 
            # можно масштабировать. Здесь оставим фиксированный размер support, 
            # но уточняем значения.
            # Если нужно менять размер ядра:
            # kh_s = max(3, int(kh * scale) // 2 * 2 + 1)
            # Но для простоты и стабильности оставим размер ядра постоянным, 
            # так как blind deconvolution часто предполагает фиксированный support.
            h_s = h.copy()
            
            # --- Внутренний цикл (MM итерации) ---
            for it in range(self.max_iterations):
                x_prev = x_s.copy()
                
                # 1. Расчет весов для изображения (Lp)
                w_h, w_v = _update_weights_image(x_s, self.p)
                
                # 2. Обновление изображения x (CG Solver)
                # Подготовка FFT ядра
                h_pad = _pad_kernel_for_fft(h_s, (H_s, W_s))
                H_fft = fft2(h_pad)
                
                x_s = _solve_image_cg(y_s, H_fft, w_h, w_v, alpha, beta, (H_s, W_s))
                x_s = np.maximum(x_s, 0.0) # Проекция на неотрицательность
                
                # 3. Расчет весов для ядра (TV)
                w_u = _update_weights_kernel(h_s)
                
                # 4. Обновление ядра h (CG Solver)
                h_s = _solve_blur_cg(y_s, x_s, w_u, gamma, beta, self.kernel_shape, (H_s, W_s))
                
                # Проекция ядра (неотрицательность и сумма = 1)
                h_s = np.maximum(h_s, 0.0)
                h_sum = h_s.sum()
                if h_sum > 1e-10:
                    h_s /= h_sum
                else:
                    h_s = np.zeros_like(h_s)
                    h_s[kh//2, kw//2] = 1.0
                
                # 5. Оценка параметров (Parameter Estimation)
                # См. Amizic 2012, уравнения (16), (17), (18)
                
                # alpha (Eq 16): lambda_1 * N / (p * sum(|Delta x|^p))
                # Замечание: в статье используется Generalized Gaussian, 
                # знаменатель - это сумма модулей градиентов в степени p.
                grad_x_h = _compute_gradient_h(x_s)
                grad_x_v = _compute_gradient_v(x_s)
                norm_lp_x = np.sum(np.abs(grad_x_h)**self.p) + np.sum(np.abs(grad_x_v)**self.p)
                alpha = (self.lambda_1 * x_s.size) / (self.p * norm_lp_x + 1e-6)
                
                # beta (Eq 17): N / ||y - Hx||^2
                # Вычисляем невязку
                Hx = np.real(ifft2(fft2(_pad_kernel_for_fft(h_s, (H_s, W_s))) * fft2(x_s)))
                residual = np.sum((y_s - Hx)**2)
                beta = x_s.size / (residual + 1e-6)
                
                # gamma (Eq 18): lambda_2 * M / TV(h)
                # TV(h) = sum(sqrt(dh^2 + dv^2))
                grad_h_h = _compute_gradient_h(h_s)
                grad_h_v = _compute_gradient_v(h_s)
                tv_h = np.sum(np.sqrt(grad_h_h**2 + grad_h_v**2 + 1e-8))
                gamma = (self.lambda_2 * h_s.size) / (tv_h + 1e-6)
                
                # Логирование
                self.history['alpha'].append(alpha)
                self.history['beta'].append(beta)
                self.history['gamma'].append(gamma)
                
                # Проверка сходимости
                diff = np.linalg.norm(x_s - x_prev) / (np.linalg.norm(x_prev) + 1e-6)
                if self.verbose and it % 5 == 0:
                    print(f"  Iter {it}: diff={diff:.1e}, a={alpha:.1e}, b={beta:.1e}, g={gamma:.1e}")
                
                if diff < self.tolerance:
                    break
            
            # Сохраняем результат текущего масштаба
            x = x_s
            h = h_s

        # 3. Финальный результат
        # Масштабируем x до исходного размера
        x_final = zoom(x, (H_full / x.shape[0], W_full / x.shape[1]), order=1)
        # Денормализация яркости
        x_final = x_final * (img_max - img_min) + img_min
        x_final = np.clip(x_final, 0, 255) if img_max > 1.0 else np.clip(x_final, 0, 1)
        
        # Сохраняем параметры
        self.hyperparams = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'p': self.p
        }
        
        return x_final, h

    # --- Методы интерфейса (чтобы не ломать фреймворк) ---
    
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('p', self.p),
            ('lambda_1', self.lambda_1),
            ('lambda_2', self.lambda_2),
            ('max_iterations', self.max_iterations),
            ('num_scales', self.num_scales),
            ('verbose', self.verbose)
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        if 'kernel_shape' in params: self.kernel_shape = tuple(params['kernel_shape'])
        if 'p' in params: self.p = float(params['p'])
        if 'lambda_1' in params: self.lambda_1 = float(params['lambda_1'])
        if 'lambda_2' in params: self.lambda_2 = float(params['lambda_2'])
        if 'max_iterations' in params: self.max_iterations = int(params['max_iterations'])
        if 'num_scales' in params: self.num_scales = int(params['num_scales'])
        if 'verbose' in params: self.verbose = bool(params['verbose'])
    
    def get_history(self) -> dict:
        return self.history
    
    def get_hyperparams(self) -> dict:
        return self.hyperparams

# --- Функции обратной совместимости (если используются в старом коде) ---
def sparse_bayesian_blind_deconvolution(y, kernel_shape, **kwargs):
    algo = SB_BID_PE(kernel_shape=kernel_shape, **kwargs)
    return algo.process(y)

# Пример запуска (для отладки)
if __name__ == "__main__":
    from scipy import signal
    import matplotlib.pyplot as plt
    
    # Тест
    img = np.zeros((100, 100))
    img[30:70, 30:70] = 1.0
    kernel = np.ones((5, 5)) / 25.0
    blurred = signal.convolve2d(img, kernel, mode='same')
    noisy = blurred + 0.001 * np.random.randn(*blurred.shape)
    
    solver = SB_BID_PE(kernel_shape=(5, 5), p=0.8, verbose=True)
    x_res, h_res = solver.process(noisy)
    
    plt.figure()
    plt.subplot(131); plt.imshow(noisy, cmap='gray'); plt.title("Input")
    plt.subplot(132); plt.imshow(x_res, cmap='gray'); plt.title("Restored")
    plt.subplot(133); plt.imshow(h_res, cmap='gray'); plt.title("Kernel")
    plt.show()

