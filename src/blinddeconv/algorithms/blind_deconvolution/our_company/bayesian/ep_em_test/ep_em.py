"""
Blind Image Deconvolution using Expectation Propagation (EP-EM).
Implementation Strategy: Fast-Cx (Spectral Uncertainty) + HQS + PGD.

Based on:
    Abdulaziz, A., et al. "Blind deconvolution of images corrupted by Gaussian noise 
    using Expectation Propagation." EUSIPCO 2021.

Modules:
    - utils: FFT helpers and math operators.
    - solvers: Pure functions for Image (HQS), Uncertainty (Spectral), and Kernel (PGD).
    - ep_em: Main class managing the EM loop.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict
from scipy.signal import convolve2d
from .utils import precompute_gradient_operators, compute_spatial_gradient, edgetaper
from .solvers import solve_image_hqs, estimate_uncertainty, solve_kernel_pgd, non_neg_ep

# Robust import of base class
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from base import DeconvolutionAlgorithm
except ImportError:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name

class EP_EM(DeconvolutionAlgorithm):
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_tv: float = 6.7,      # ВЕРНУЛ оригинальное значение
        noise_sigma: float = 0.05,   # Шум
        max_iter: int = 30,          # Итерации EM
        hqs_iter: int = 5,           # Итерации восстановления картинки
        pgd_iter: int = 20,          # Итерации ядра
        pgd_momentum: float = 0.9,
        beta_max: float = 512.0,     # Оптимально для баланса скорость/качество
        strategy: str = 'fast',
        num_probes: int = 10,
        non_neg: bool = True,
        verbose: bool = False
    ):
        super().__init__(name='EP-EM-BID')
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_tv = lambda_tv
        self.noise_sigma = noise_sigma
        self.max_iter = max_iter
        self.verbose = verbose
        
        self.hqs_iter = hqs_iter
        self.pgd_iter = pgd_iter
        self.pgd_momentum = pgd_momentum
        self.beta_max = beta_max
        self.strategy = strategy
        self.num_probes = num_probes
        self.non_neg = non_neg
        
        self.history = {'kernel_diff': []}
        self.hyperparams = {}

    def _build_Dx_fast(self, r: np.ndarray, kh: int, kw: int, H: int, W: int) -> np.ndarray:
        """
        Быстрое построение матрицы Dx (ковариация) без вложенных циклов.
        Математически эквивалентно оригиналу, но работает за доли секунды.
        """
        # Создаем сетку индексов для ядра
        k = kh * kw
        I = np.arange(k)
        y_idx = I // kw
        x_idx = I % kw
        
        # Вычисляем разницы координат (broadcasting)
        # dy[i, j] = y_idx[i] - y_idx[j]
        dy = (y_idx[:, None] - y_idx[None, :]) % H
        dx = (x_idx[:, None] - x_idx[None, :]) % W
        
        # Извлекаем значения из автокорреляции r
        # r имеет размер (H, W) и центрирована в (0,0) по логике ifft
        D_x = (H * W) * r[dy, dx]
        return D_x

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Нормализация
        y_full = image.astype(np.float64)
        if y_full.max() > 1.0:
            y_full /= 255.0
            
        H, W = y_full.shape
        kh, kw = self.kernel_shape

        # 2. Инициализация ядра (Гаусс, чтобы не начинать с точки)
        # Немного расширим сигму, чтобы алгоритм мог "схватить" смаз
        sig = max(kh, kw) / 8.0 
        grid_y, grid_x = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
        h = np.exp(-(grid_x**2 + grid_y**2) / (2 * sig**2))
        h /= h.sum()
        
        # 3. EDGETAPER (ВАЖНО!)
        # Смягчаем края, чтобы убрать артефакты-линии от FFT
        # Используем текущее приближение ядра для taper
        y = edgetaper(y_full, h)
        x = y.copy()

        # Прекомпьют градиентов
        F_ops = precompute_gradient_operators((H, W))
        _, _, F_grad_sq = F_ops
        
        if self.verbose:
            print(f"[{self.name}] Start. Img: {H}x{W}, Ker: {kh}x{kw}")

        # --- EM LOOP ---
        for it in range(self.max_iter):
            h_prev = h.copy()
            
            # --- E-STEP (Image) ---
            # HQS: Восстанавливаем изображение x
            x = solve_image_hqs(
                y, h, x, 
                self.noise_sigma, self.lambda_tv, 
                self.beta_max, self.hqs_iter, F_ops
            )
            
            # Адаптивная lambda (как в статье)
            # Добавил clip, чтобы не делить на 0 и не получать бесконечность
            grad_x, grad_y = compute_spatial_gradient(x)
            mean_grad = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y)) + 1e-6
            lambda_eff = self.lambda_tv / (mean_grad * 0.5)

            # Оценка неопределенности (Variance)
            uncertainty, r = estimate_uncertainty(
                h, self.noise_sigma, lambda_eff, (H, W), F_grad_sq,
                strategy=self.strategy
            )
            
            # Non-negativity constraint (Soft)
            if self.non_neg:
                x = non_neg_ep(x, uncertainty)

            # Строим Dx быстро
            D_x = self._build_Dx_fast(r, kh, kw, H, W)
            
            # --- M-STEP (Kernel) ---
            # Обновляем ядро с учетом Dx
            h = solve_kernel_pgd(
                y, x, h, 
                D_x, self.pgd_iter, momentum=self.pgd_momentum
            )
            
            # Логирование
            diff = np.linalg.norm(h - h_prev)
            if self.verbose:
                print(f"Iter {it+1}: dH={diff:.6f}, Uncert={uncertainty:.2e}")
            
            if diff < 1e-6:
                break
        
        # 4. Финальный проход (Non-blind deconvolution)
        # Используем оригинальное (не taper) изображение, но с найденным ядром
        # Увеличиваем beta_max для максимальной резкости
        # Но сначала нужно снова сделать edgetaper с финальным ядром для лучшего результата,
        # либо использовать граничные условия 'wrap' в HQS (FFT это и делает).
        # Для чистоты результата применим к taper версии, потом обрежем или вернем как есть.
        
        # Повторный edgetaper с найденным ядром для финального восстановления
        y_final_taper = edgetaper(y_full, h)
        x_final = solve_image_hqs(
            y_final_taper, h, x, 
            self.noise_sigma, self.lambda_tv * 0.8, # Чуть слабее регуляризация для деталей
            self.beta_max * 4.0, self.hqs_iter * 2, F_ops
        )
        
        x_final = np.clip(x_final * 255.0, 0, 255).astype(np.uint8)
        return x_final, h

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_tv', self.lambda_tv),
            ('noise_sigma', self.noise_sigma),
            ('max_iter', self.max_iter),
            ('strategy', self.strategy),
            ('non_neg', self.non_neg)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)
    
    def get_history(self) -> dict:
        return self.history
    
    def get_hyperparams(self) -> dict:
        return self.hyperparams

def run_algorithm(g, kernel_shape, **kwargs):
    algo = EP_EM(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history
