"""
Вариационная байесовская компрессионная слепая деконволюция
(Variational Bayesian Compressive Blind Image Deconvolution)

Литература:
    Amizic, B., Spinoulas, L., Molina, R., & Katsaggelos, A. K. (2013).
    Variational Bayesian Compressive Blind Image Deconvolution.
    In 2013 21st European Signal Processing Conference (EUSIPCO) (pp. 1-5). IEEE.
    ID: 1569744671

Алгоритм реализует вариационный байесовский подход к задаче слепой деконволюции,
объединенный с принципами компрессионного сжатия (Compressed Sensing).
Модель наблюдения:
    y = ΦHx + n
В контексте классической деконволюции предполагается Φ = I.

Используемые априорные распределения:
    - Изображение x: Обобщенное гауссово распределение (GGD) на градиенты,
      что соответствует невыпуклой l_p квази-норме (0 < p < 1).
    - Ядро h: Одновременная авторегрессия (SAR), поощряющая гладкость.
    - Коэффициенты a: l_1 норма (распределение Лапласа), при условии Hx ≈ Wa.

Для вывода используется метод множителей Лагранжа (ADMM), интегрированный
в вариационный цикл, что позволяет эффективно разделять переменные x и a.
"""

import numpy as np
from scipy.signal import fftconvolve, convolve2d
from scipy.sparse.linalg import LinearOperator, cg
import pywt
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm

# Константа для численной стабильности
EPSILON = 1e-10


class Amizic2013(DeconvolutionAlgorithm):
    """
    Реализация алгоритма VB-ADMM Blind Deconvolution (Amizic et al., 2013).

    Атрибуты
    --------
    kernel_size : int
        Размер ядра размытия (нечетное число).
    outer_iters : int
        Количество внешних итераций вариационного алгоритма.
    x_cg_iters : int
        Количество итераций метода сопряженных градиентов (CG) для x.
    h_grad_iters : int
        Количество шагов градиентного спуска для h.
    p_val : float
        Параметр формы p для априори изображения (0 < p <= 1).
        В статье рекомендуется значение 0.8.
    eta : float
        Параметр штрафа ADMM (обратная дисперсия шума согласования Hx=Wa).
    lambda1 : float
        Масштабный параметр для априори изображения (по умолчанию 0.5).
    wavelet : str
        Тип вейвлета для разреженного представления (по умолчанию 'haar').
    verbose : bool
        Вывод информации о ходе выполнения.
    """

    def __init__(
        self,
        kernel_size: int = 15,
        outer_iters: int = 15,
        x_cg_iters: int = 25,
        h_grad_iters: int = 10,
        p_val: float = 0.8,
        eta: float = 50.0,
        lambda1: float = 0.5,
        wavelet: str = 'haar',
        verbose: bool = False
    ):
        super().__init__(name='Amizic2013')
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.outer_iters = outer_iters
        self.x_cg_iters = x_cg_iters
        self.h_grad_iters = h_grad_iters
        self.p = p_val
        self.eta = eta
        self.lambda1 = lambda1
        self.wavelet = wavelet
        self.verbose = verbose
        
        # Проверка наличия библиотеки PyWavelets
        if pywt is None:
            raise ImportError("Для работы алгоритма требуется библиотека PyWavelets (pip install PyWavelets)")

        self.history = {}
        self.hyperparams = {}

    def _flip(self, k: np.ndarray) -> np.ndarray:
        """Поворот ядра на 180 градусов для вычисления сопряженного оператора H^T."""
        return np.flip(np.flip(k, 0), 1)

    def _grad(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет пространственные градиенты (первые разности).
        Используется для формирования весов l_p нормы.
        """
        dx = np.zeros_like(x)
        dy = np.zeros_like(x)
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
        dy[:-1, :] = x[1:, :] - x[:-1, :]
        return dx, dy

    def _div(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """
        Вычисляет дивергенцию (отрицательный сопряженный оператор к градиенту).
        Необходим для построения оператора L_w = D^T W D в методе CG.
        """
        out = np.zeros_like(px)
        out[:, :-1] -= px[:, :-1]
        out[:, 1:] += px[:, :-1]
        out[:-1, :] -= py[:-1, :]
        out[1:, :] += py[:-1, :]
        return out

    def _wt(self, x: np.ndarray) -> Tuple[np.ndarray, list, tuple]:
        """Прямое вейвлет-преобразование (Wa)."""
        coeffs = pywt.wavedec2(x, self.wavelet, level=2, mode='periodization')
        arr, slices = pywt.coeffs_to_array(coeffs)
        return arr, slices, x.shape

    def _iwt(self, arr: np.ndarray, slices, shape) -> np.ndarray:
        """Обратное вейвлет-преобразование (W^T a)."""
        coeffs = pywt.array_to_coeffs(arr, slices, output_format='wavedec2')
        return pywt.waverec2(coeffs, self.wavelet, mode='periodization')[:shape[0], :shape[1]]

    def _solve_x(self, target: np.ndarray, k: np.ndarray, x0: np.ndarray, 
                 alpha: float, w_x: np.ndarray, w_y: np.ndarray) -> np.ndarray:
        """
        Обновляет оценку изображения x (Шаг 1.c алгоритма).
        
        Минимизируется квадратичная аппроксимация функционала:
            (η/2)||Wa - Hx + u||^2 + (α/2) x^T L_w x
        
        Это приводит к решению системы линейных уравнений (Ур. 24):
            (η H^T H + α p L_w) x = η H^T (Wa + u)
        
        где L_w — взвешенный лапласиан, аппроксимирующий l_p норму методом 
        мажоризации-минимизации.
        """
        H_rows, W_cols = x0.shape
        k_flip = self._flip(k)

        # Оператор H^T H (свертка с ядром и его сопряженным)
        def HTH(z):
            hz = fftconvolve(z, k, mode='same')
            return fftconvolve(hz, k_flip, mode='same')

        # Оператор L_w (взвешенный Лапласиан)
        def Lw(z):
            dx, dy = self._grad(z)
            return self._div(w_x * dx, w_y * dy)

        def matvec(v_flat):
            v = v_flat.reshape(H_rows, W_cols)
            term1 = self.eta * HTH(v)
            term2 = alpha * self.p * Lw(v)
            return (term1 + term2).ravel()

        A = LinearOperator((x0.size, x0.size), matvec=matvec, dtype=np.float64)
        
        # Правая часть уравнения: η H^T(target), где target = Wa + u
        b = (self.eta * fftconvolve(target, k_flip, mode='same')).ravel()

        # Решение системы методом сопряженных градиентов
        x_new, _ = cg(A, b, x0=x0.ravel(), maxiter=self.x_cg_iters, tol=1e-5)
        return x_new.reshape(H_rows, W_cols)

    def _update_h(self, x: np.ndarray, target: np.ndarray, h: np.ndarray, gamma: float) -> np.ndarray:
        """
        Обновляет оценку ядра размытия h (Шаг 1.d алгоритма).
        
        Минимизирует функционал (Ур. 25):
            (η/2)||Wa - Hx + u||^2 + (γ/2)||Ch||^2
        
        Задача решается методом градиентного спуска с проекцией на симплекс
        (ограничения неотрицательности и нормировки).
        """
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        # Эвристическая оценка шага обучения на основе масштаба данных
        denom = (self.eta * np.mean(x**2) * self.kernel_size**2 + gamma + 1e-5)
        step_size = 0.5 / denom

        x_flip = self._flip(x)
        kh, kw = h.shape
        h_curr = h.copy()
        
        for _ in range(self.h_grad_iters):
            # Градиент члена верности данных: -η * x^T * (Target - Hx)
            xh = fftconvolve(x, h_curr, mode='same')
            residual = target - xh 
            
            grad_data_full = fftconvolve(residual, x_flip, mode='same')
            
            # Обрезка градиента до размера ядра
            cy, cx = grad_data_full.shape[0]//2, grad_data_full.shape[1]//2
            dy, dx = kh//2, kw//2
            grad_data = -self.eta * grad_data_full[cy-dy:cy+dy+1, cx-dx:cx+dx+1]
            
            # Градиент регуляризации (SAR prior): γ * C^T C h
            Ch = convolve2d(h_curr, laplacian_kernel, mode='same')
            grad_reg = gamma * convolve2d(Ch, laplacian_kernel, mode='same')
            
            # Шаг спуска
            h_curr = h_curr - step_size * (grad_data + grad_reg)
            
            # Проекция (неотрицательность и сумма равна 1)
            h_curr = np.maximum(h_curr, 0)
            h_sum = np.sum(h_curr)
            if h_sum > EPSILON:
                h_curr /= h_sum
        
        return h_curr

    def _update_a_analytic(self, y: np.ndarray, b: np.ndarray, beta: float, tau: float, wt_slices) -> np.ndarray:
        """
        Обновляет вейвлет-коэффициенты a (Шаг 4, Ур. 30).
        
        Минимизируемый функционал:
            L(a) = (β/2)||y - Wa||^2 + (η/2)||Wa - b||^2 + τ||a||_1
        
        Поскольку вейвлет-преобразование W ортогонально, задача разделяется
        и имеет аналитическое решение через Soft Thresholding взвешенного 
        среднего наблюдений y и переменной ADMM b.
        """
        # Взвешенное среднее наблюдений
        denom = beta + self.eta
        Z = (beta * y + self.eta * b) / denom
        
        # Эффективный порог
        thresh = tau / denom

        # Прямое вейвлет-преобразование взвешенного среднего
        arr_Z, _, _ = self._wt(Z)
        
        # Soft-thresholding
        a_new = np.sign(arr_Z) * np.maximum(np.abs(arr_Z) - thresh, 0)
        
        return a_new

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск процедуры слепой деконволюции.
        
        Параметры
        ---------
        image : ndarray
            Входное размытое изображение (в градациях серого или цветное).
            
        Возвращает
        ----------
        f_est : ndarray
            Восстановленное изображение (uint8).
        h_est : ndarray
            Оцененное ядро размытия.
        """
        start_time = time.time()

        if image.dtype == np.uint8:
            y_in = image.astype(np.float64) / 255.0
        else:
            y_in = image.astype(np.float64)

        if y_in.ndim == 3:
            # Для цветных изображений ядро оценивается по каналу яркости
            y_gray = 0.299 * y_in[:,:,0] + 0.587 * y_in[:,:,1] + 0.114 * y_in[:,:,2]
            f_gray, h_est = self._process_channel(y_gray)
            
            # Неслепая деконволюция для каждого канала с зафиксированным ядром
            res_channels = []
            for ch in range(3):
                # fix_h=True отключает обновление ядра
                f_ch, _ = self._process_channel(y_in[:,:,ch], h_init=h_est, fix_h=True)
                res_channels.append(f_ch)
            
            f_est = np.dstack(res_channels)
            f_est = np.clip(f_est * 255.0, 0, 255).astype(np.uint8)
        else:
            f_est_float, h_est = self._process_channel(y_in)
            f_est = np.clip(f_est_float * 255.0, 0, 255).astype(np.uint8)

        self.timer = time.time() - start_time
        return f_est, h_est

    def _process_channel(self, y: np.ndarray, h_init=None, fix_h=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Внутренняя логика обработки одного канала изображения.
        
        Выполняет итерационное обновление переменных x, h, a и гиперпараметров.
        """
        H, W = y.shape
        N = H * W
        
        # Инициализация ядра
        if h_init is None:
            h = np.zeros((self.kernel_size, self.kernel_size))
            h[self.kernel_size//2, self.kernel_size//2] = 1.0
        else:
            h = h_init.copy()
            
        # Инициализация переменных
        x = y.copy()
        u = np.zeros_like(y) # Множитель Лагранжа
        
        # Инициализация вейвлет-коэффициентов
        a_arr, slices, shp = self._wt(x)
        
        # Начальные значения гиперпараметров
        alpha = 1.0
        gamma = 1.0
        beta = 100.0  # Высокая начальная точность данных
        tau = 0.1
        
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        for k in range(self.outer_iters):
            
            # Обновление гиперпараметров alpha и gamma (Ур. 21, 22)
            dx, dy = self._grad(x)
            grad_p_norm = np.sum(np.abs(dx)**self.p + np.abs(dy)**self.p)
            alpha = (self.lambda1 * (N / self.p) + 1) / (grad_p_norm + EPSILON)
            
            if not fix_h:
                Ch = convolve2d(h, laplacian_kernel, mode='same')
                gamma = (h.size + 2) / (np.sum(Ch**2) + EPSILON)
            
            # Расчет весов для мажоризации l_p нормы (Ур. 23)
            # w ~ |grad|^(p-2) используется для построения взвешенного Лапласиана
            w_x = (dx**2 + EPSILON)**((self.p - 2.0) / 2.0)
            w_y = (dy**2 + EPSILON)**((self.p - 2.0) / 2.0)
            
            # Обновление изображения x (Ур. 24)
            Wa = self._iwt(a_arr, slices, (H, W))
            target_x = Wa + u
            x = self._solve_x(target_x, h, x, alpha, w_x, w_y)
            x = np.clip(x, 0, 1)
            
            # Обновление ядра размытия h (Ур. 25)
            if not fix_h:
                h = self._update_h(x, target_x, h, gamma)
            
            # Обновление гиперпараметров beta и tau (Ур. 28, 29)
            res_y = y - Wa
            beta = (N + 2) / (np.sum(res_y**2) + EPSILON)
            
            tau_unscaled = (a_arr.size + 1) / (np.sum(np.abs(a_arr)) + EPSILON)
            tau = tau_unscaled
            
            # Обновление коэффициентов a (Ур. 30)
            # b_admm соответствует члену (Hx - u) в целевой функции
            Hx = fftconvolve(x, h, mode='same')
            b_admm = Hx - u
            a_arr = self._update_a_analytic(y, b_admm, beta, tau, slices)
            
            # Обновление множителя Лагранжа u (Ур. 32)
            Wa_new = self._iwt(a_arr, slices, (H, W))
            u = u + (Wa_new - Hx)
            
            if self.verbose:
                print(f"Итерация {k+1:2d}: α={alpha:.2e}, γ={gamma:.2e}, β={beta:.2e}, τ={tau:.2e}")
                
        self.hyperparams = {
            'alpha': float(alpha),
            'gamma': float(gamma),
            'beta': float(beta),
            'tau': float(tau)
        }
        
        return x, h

    def get_param(self) -> List[Tuple[str, Any]]:
        """Возвращает текущие параметры алгоритма."""
        return [
            ('kernel_size', self.kernel_size),
            ('outer_iters', self.outer_iters),
            ('p_val', self.p),
            ('eta', self.eta),
            ('lambda1', self.lambda1),
            ('wavelet', self.wavelet),
            ('verbose', self.verbose)
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """Изменяет параметры алгоритма."""
        if 'kernel_size' in params:
            self.kernel_size = int(params['kernel_size'])
            if self.kernel_size % 2 == 0: self.kernel_size += 1
        if 'outer_iters' in params:
            self.outer_iters = int(params['outer_iters'])
        if 'p_val' in params:
            self.p = float(params['p_val'])
        if 'eta' in params:
            self.eta = float(params['eta'])
        if 'lambda1' in params:
            self.lambda1 = float(params['lambda1'])
        if 'wavelet' in params:
            self.wavelet = str(params['wavelet'])
        if 'verbose' in params:
            self.verbose = bool(params['verbose'])

    def get_history(self) -> dict:
        """Возвращает историю выполнения (пусто, если не реализовано сохранение метрик)."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает финальные значения гиперпараметров."""
        return self.hyperparams