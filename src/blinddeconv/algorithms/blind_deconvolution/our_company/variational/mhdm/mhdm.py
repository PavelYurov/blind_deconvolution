"""
Реализация класса MHDMBlind.
Single-Scale версия с Edgetaper и финальной неслепой деконволюцией.
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

# Подгрузка локальных модулей
from .utils import (compute_sobolev_weights, normalize_min_max, 
                   pad_image, crop_center, edgetaper)
from .solvers import solve_step_0, solve_step_n

# --- Boilerplate imports ---
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
except Exception:
    class DeconvolutionAlgorithm:
        def __init__(self, name): self.name = name
# ---------------------------

class MHDMBlind(DeconvolutionAlgorithm):
    """
    Multiscale Hierarchical Decomposition Method (MHDM) for Blind Deconvolution.
    Single-scale implementation with Edgetaper and Final Restoration.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        # Параметры MHDM (для оценки ядра)
        lambda_0: float = 1e-4, 
        mu_0: float = 1e-3,      
        r: float = 0.8,          # Гладкость картинки
        s: float = 0.1,          # Гладкость ядра (низкое значение для остроты)
        scaling_factor: float = 1.3, # Плавное уменьшение параметров
        noise_level: float = 0.01,
        tau: float = 1.01,
        max_iter: int = 50,      # Больше итераций, так как без пирамиды сходимся дольше
        
        # Настройки пост-обработки ядра
        kernel_threshold: float = 0.05, # Срезаем все что ниже 5% от максимума (убирает "жир")
        
        # Настройки финальной деконволюции
        final_deconv_method: str = 'tikhonov', # 'wiener' или 'tikhonov'
        final_alpha: float = 1e-2,             # Регуляризация для финального шага
        
        auto_scale_params: bool = True
    ):
        super().__init__(name='MHDM-Blind-SingleScale')
        self.kernel_shape = kernel_shape
        self.lambda_0 = lambda_0
        self.mu_0 = mu_0
        self.r = r
        self.s = s
        self.scaling_factor = scaling_factor
        self.noise_level = noise_level
        self.tau = tau
        self.max_iter = max_iter
        
        self.kernel_threshold = kernel_threshold
        self.final_deconv_method = final_deconv_method
        self.final_alpha = final_alpha
        self.auto_scale_params = auto_scale_params
        
        self.history = {'residual': []}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        
        # 1. Подготовка данных
        y = image.astype(np.float64)
        if y.max() > 1.0: y /= 255.0
        orig_H, orig_W = y.shape
        
        # --- ЭТАП 1: ОЦЕНКА ЯДРА (MHDM) ---
        
        # Используем Edgetaper. Это критически важно для FFT методов оценки ядра.
        # Он сглаживает края, чтобы убрать разрывы при периодическом продолжении.
        # Без этого в спектре ядра появляется "крест".
        y_for_est = edgetaper(y, self.kernel_shape)
        
        H, W = y_for_est.shape
        F_f = np.fft.fft2(y_for_est)
        
        # Веса Соболева
        W_r = compute_sobolev_weights((H, W), self.r)
        W_s = compute_sobolev_weights((H, W), self.s)
        
        # Начальные параметры
        lambda_curr = self.lambda_0
        mu_curr = self.mu_0
        
        if self.auto_scale_params:
            mean_signal = np.mean(np.abs(F_f))
            mean_Wr = np.mean(W_r)
            mean_Ws = np.mean(W_s)
            # Mu отвечает за штраф ядра. Делаем его маленьким, чтобы ядро могло
            # принять форму линии (Motion) или диска (Defocus), а не точки.
            mu_curr = 0.002 * (mean_signal / mean_Ws)
            # Lambda отвечает за картинку.
            lambda_curr = mu_curr * 0.1

        # Накопители в Фурье
        F_U = np.zeros_like(F_f, dtype=np.complex128)
        F_K = np.zeros_like(F_f, dtype=np.complex128)
        
        stop_threshold = self.tau * (H * W) * (self.noise_level ** 2)
        self.history['residual'] = []
        
        # Итерационный процесс
        converged_iter = self.max_iter
        for n in range(self.max_iter):
            if n == 0:
                # Аналитический первый шаг
                F_u_inc, F_k_inc = solve_step_0(F_f, lambda_curr, mu_curr, W_r, W_s)
            else:
                # Уменьшаем параметры (Geometrical Cooling)
                # Это позволяет сначала найти грубую форму, потом детали
                lambda_curr /= self.scaling_factor
                mu_curr /= self.scaling_factor
                
                # Решение полинома 5-й степени для шага n
                F_u_inc, F_k_inc = solve_step_n(F_f, F_U, F_K, lambda_curr, mu_curr, W_r, W_s)
            
            F_U += F_u_inc
            F_K += F_k_inc
            
            # Проверка сходимости
            res_sq = np.sum(np.abs(F_f - F_K * F_U)**2) / (H * W)
            self.history['residual'].append(res_sq)
            
            # Не выходим слишком рано, даем ядру сформироваться
            if res_sq <= stop_threshold and n > 15:
                converged_iter = n
                # print(f"Converged at iter {n}")
                break
        
        # --- ЭТАП 2: ПОСТ-ОБРАБОТКА ЯДРА ---
        
        # Перевод в пространство
        k_full = np.real(np.fft.ifft2(F_K))
        # Сдвиг центра в середину массива
        k_shifted = np.fft.fftshift(k_full)
        
        # Вырезаем ядро заданного размера
        kh, kw = self.kernel_shape
        cy, cx = H // 2, W // 2
        sy, sx = cy - kh // 2, cx - kw // 2
        k_final = k_shifted[sy:sy+kh, sx:sx+kw]
        
        # 1. Убираем отрицательные значения
        k_final = np.maximum(k_final, 0)
        
        # 2. Thresholding (борьба с "жирным" ядром)
        # Обнуляем все значения меньше N% от максимума.
        # Это делает линии тоньше и убирает туман.
        thresh_val = k_final.max() * self.kernel_threshold
        k_final[k_final < thresh_val] = 0
        
        # 3. Нормализация
        k_sum = k_final.sum()
        if k_sum > 1e-12: 
            k_final /= k_sum
        else:
            # Fallback: если ядро исчезло, возвращаем дельту
            k_final[kh//2, kw//2] = 1.0
        
        # --- ЭТАП 3: ФИНАЛЬНОЕ ВОССТАНОВЛЕНИЕ (NON-BLIND) ---
        
        # Для восстановления используем Padding (отражение), а не Edgetaper.
        # Это сохраняет детали на краях.
        pad_h = kh + 2
        pad_w = kw + 2
        y_padded = pad_image(y, pad_h, mode='reflect')
        
        # Запуск выбранного метода деконволюции
        u_restored_padded = self.run_final_deconv(y_padded, k_final)
        
        # Обрезаем паддинг
        u_final = crop_center(u_restored_padded, (orig_H, orig_W))
        
        # Clip в валидный диапазон
        u_final = np.clip(u_final, 0, 1)
        u_out = (u_final * 255.0).astype(np.uint8)
        
        self.hyperparams['time'] = time.time() - start_time
        self.hyperparams['iterations'] = converged_iter
        
        return u_out, k_final

    def run_final_deconv(self, y: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Выполняет неслепую деконволюцию с известным ядром.
        Решает проблему серости и плохого контраста.
        """
        H, W = y.shape
        kh, kw = k.shape
        
        # Паддинг ядра до размера изображения (центр в 0,0 для FFT)
        k_pad = np.zeros((H, W))
        cy, cx = H//2, W//2
        sy, sx = cy - kh//2, cx - kw//2
        k_pad[sy:sy+kh, sx:sx+kw] = k
        k_pad = np.fft.ifftshift(k_pad)
        
        F_y = np.fft.fft2(y)
        F_k = np.fft.fft2(k_pad)
        
        # Квадрат модуля ядра
        F_k_abs2 = np.abs(F_k)**2
        
        if self.final_deconv_method == 'wiener':
            # Wiener Filter: F_u = (conj(H) * F_y) / (|H|^2 + alpha)
            # alpha (SNR) здесь константа.
            denom = F_k_abs2 + self.final_alpha
            F_u = np.conj(F_k) * F_y / denom
            
        elif self.final_deconv_method == 'tikhonov':
            # Tikhonov с регуляризацией по Лапласиану (гладкость второго порядка)
            # F_u = (conj(H) * F_y) / (|H|^2 + alpha * |L|^2)
            
            # Строим оператор Лапласа в Фурье: k^2
            freq_y = np.fft.fftfreq(H).reshape(-1, 1)
            freq_x = np.fft.fftfreq(W).reshape(1, -1)
            L_sq = (freq_y**2 + freq_x**2)
            # Нормируем, чтобы alpha имела предсказуемый эффект
            L_sq /= (L_sq.max() + 1e-12)
            
            denom = F_k_abs2 + self.final_alpha * L_sq + 1e-9
            F_u = np.conj(F_k) * F_y / denom
            
        else:
            # Если метод не выбран, возвращаем оригинал (fallback)
            return y
            
        return np.real(np.fft.ifft2(F_u))

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('scaling_factor', self.scaling_factor),
            ('s', self.s),
            ('kernel_threshold', self.kernel_threshold),
            ('final_deconv_method', self.final_deconv_method),
            ('final_alpha', self.final_alpha),
            ('max_iter', self.max_iter)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams