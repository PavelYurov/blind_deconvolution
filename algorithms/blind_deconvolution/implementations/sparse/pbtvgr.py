"""
Слепая деконволюция для изображений с пуассоновским шумом 
с использованием TV и L0-регуляризации градиентов

Литература:
    Dong, W., Tao, S., Xu, G., & Chen, Y. (2021).
    Blind Deconvolution for Poissonian Blurred Image With Total Variation 
    and L0-Norm Gradient Regularizations.
    IEEE Transactions on Image Processing, 30, 1030-1043.
    DOI: 10.1109/TIP.2020.3038518

Алгоритм решает задачу минимизации:
    min_{h,o} (λ/2) * KL(P(ho), g) + ||∇o||_0 + μ * TV(h)
    
    где KL — обобщенная дивергенция Кульбака-Лейблера, соответствующая 
    отрицательному логарифму правдоподобия распределения Пуассона:
    KL(u, g) ≈ sum(u - g * ln(u)).

Используемые методы:
    - Расщепление переменных (Variable Splitting) и метод множителей Лагранжа
    - Альтернативная минимизация (Algorithm 1)
    - IRLS для оценки PSF (Algorithm 2)
    - L0-сглаживание для оценки изображения (Algorithm 4)
    - Не-слепая деконволюция на этапе постобработки (Algorithm 5)

Реализованные уравнения:
    - Ур. (5): Целевая функция
    - Ур. (9), (13)-(15): h-подзадача (IRLS с модифицированным fidelity-членом)
    - Ур. (10), (16)-(18): u-подзадача (аналитическое решение для Пуассона)
    - Ур. (11), (28)-(31): o-подзадача (L0-сглаживание через расщепление)
    - Ур. (36)-(42): не-слепая деконволюция с TV-регуляризацией
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg, LinearOperator
import time
from typing import Tuple, List, Any, Dict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base import DeconvolutionAlgorithm


EPSILON = 1e-10


def _pad_kernel_for_fft(h, image_shape):
    """
    Дополняет ядро h до размера изображения и центрирует для БПФ.
    
    Параметры
    ---------
    h : ndarray (kh, kw)
        Ядро размытия.
    image_shape : tuple (H, W)
        Размер целевого изображения.
    
    Возвращает
    ----------
    h_padded : ndarray (H, W)
        Дополненное и центрированное ядро.
    """
    H, W = image_shape
    kh, kw = h.shape
    h_padded = np.zeros((H, W), dtype=np.float64)
    h_padded[:kh, :kw] = h
    h_padded = np.roll(h_padded, shift=-kh//2, axis=0)
    h_padded = np.roll(h_padded, shift=-kw//2, axis=1)
    return h_padded


def _extract_kernel_from_padded(h_padded, kernel_shape):
    """
    Извлекает ядро из дополненного представления.
    
    Параметры
    ---------
    h_padded : ndarray (H, W)
        Дополненное ядро.
    kernel_shape : tuple (kh, kw)
        Размер исходного ядра.
    
    Возвращает
    ----------
    h : ndarray (kh, kw)
        Извлечённое ядро.
    """
    kh, kw = kernel_shape
    shifted = np.roll(h_padded, shift=kh//2, axis=0)
    shifted = np.roll(shifted, shift=kw//2, axis=1)
    return shifted[:kh, :kw]


def _compute_gradient_operators_fft(image_shape):
    """
    Вычисляет БПФ операторов градиента (форвардные разности).
    Используется в ур. (5), (13) и (31).
    
    Параметры
    ---------
    image_shape : tuple (H, W)
        Размер изображения.
    
    Возвращает
    ----------
    F_dx : ndarray (H, W), complex
        БПФ горизонтального оператора.
    F_dy : ndarray (H, W), complex
        БПФ вертикального оператора.
    """
    H, W = image_shape
    
    # Горизонтальный градиент [0, -1, 1] (оператор D_x)
    dx = np.zeros((H, W), dtype=np.float64)
    dx[0, 0] = -1
    dx[0, 1] = 1
    
    # Вертикальный градиент [[0], [-1], [1]] (оператор D_y)
    dy = np.zeros((H, W), dtype=np.float64)
    dy[0, 0] = -1
    dy[1, 0] = 1
    
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    
    return F_dx, F_dy


class PBTVGR(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции для Пуассоновского шума с TV и L0 регуляризацией (PBTVGR).
    
    Включает в себя два этапа:
    1. Слепая оценка ядра размытия и предварительного изображения (Algorithm 1).
    2. Не-слепая деконволюция для улучшения качества (Algorithm 5).
    
    Атрибуты
    --------
    kernel_shape : tuple (kh, kw)
        Размер оцениваемой PSF.
    lambda_ : float
        Вес fidelity-члена (данных). Основной параметр регуляризации.
    mu : float
        Вес TV-регуляризации для PSF.
    beta_init : float
        Начальное значение штрафного параметра β.
    gamma_init : float
        Начальное значение штрафного параметра γ (для L0 подзадачи).
    xi : float
        Вес регуляризации для не-слепого этапа.
    eta_init : float
        Начальное значение штрафного параметра η (для не-слепого этапа).
    """
    
    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        lambda_: float = 2e4,
        mu: float = 0.04,
        beta_init: float = 1.0,
        gamma_init: float = 2.0,
        xi: float = 1e4,
        eta_init: float = 1.0,
        tau: float = 10.0,
        T: float = 100.0,
        sigma: float = 2.0,
        zeta: float = 10.0,
        max_iter: int = 50,
        nonblind_iter: int = 30,
        verbose: bool = False
    ):
        """
        Инициализация алгоритма PBTVGR.
        
        Параметры
        ---------
        kernel_shape : tuple (kh, kw)
            Размер оцениваемой PSF.
        lambda_ : float
            Параметр регуляризации для u (формула 10).
        mu : float
            Параметр TV регуляризации для PSF (формула 9).
        beta_init : float
            Начальное значение beta (формула 12).
        gamma_init : float
            Начальное значение gamma (формула 19).
        xi : float
            Параметр регуляризации для не-слепой деконволюции (формула 36).
        eta_init : float
            Начальное значение eta (формула 37).
        tau : float
            Скорость обновления beta (формула 12).
        T : float
            Максимальный шаг для beta (формула 12).
        sigma : float
            Коэффициент увеличения gamma (формула 22).
        zeta : float
            Скорость обновления eta (формула 41).
        max_iter : int
            Максимальное число итераций слепого этапа.
        nonblind_iter : int
            Число итераций не-слепого этапа.
        verbose : bool
            Выводить прогресс.
        """
        super().__init__(name='PBTVGR')
        
        self.kernel_shape = tuple(kernel_shape)
        self.lambda_ = lambda_
        self.mu = mu
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.xi = xi
        self.eta_init = eta_init
        self.tau = tau
        self.T = T
        self.sigma = sigma
        self.zeta = zeta
        
        self.max_iter = max_iter
        self.nonblind_iter = nonblind_iter
        self.verbose = verbose
        
        self.history = {}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение слепой деконволюции (Algorithm 1 и Algorithm 5).
        
        Параметры
        ---------
        image : ndarray (H, W)
            Размытое зашумленное изображение (g).
            
        Возвращает
        ----------
        f_est : ndarray (H, W)
            Восстановленное изображение (после не-слепого этапа).
        h_est : ndarray (kh, kw)
            Оцененное ядро размытия.
        """
        start_time = time.time()
        
        if image.ndim != 2:
            raise ValueError("Ожидается 2D изображение в градациях серого")
        
        g = np.asarray(image, dtype=np.float64)
        g = np.maximum(g, EPSILON)  # Для корректного логарифма (KL)
        H, W = g.shape
        kh, kw = self.kernel_shape
        
        # Инициализация (Algorithm 1, step 1)
        # PSF инициализируется гауссианой (как указано в Section V.F)
        sig = max(kh, kw) / 10.0
        y_grid, x_grid = np.ogrid[-kh//2:kh//2, -kw//2:kw//2]
        h_init = np.exp(-(x_grid**2 + y_grid**2) / (2 * sig**2))
        h_init /= h_init.sum()
        
        h = h_init.copy()
        u = g.copy()
        o = g.copy()
        
        beta = self.beta_init
        
        # Предвычисление операторов градиента в Фурье 
        # (используются в h-подзадаче и o-подзадаче)
        F_dx, F_dy = _compute_gradient_operators_fft((H, W))
        
        # История сходимости
        self.history = {'beta': [], 'error': []}
        
        # Основной цикл слепой деконволюции (Algorithm 1)
        for k in range(self.max_iter):
            # 1. Решение h-подзадачи (Algorithm 2)
            # h = argmin (β/2)||d(ho) - du||^2 + μ TV(h)
            h = self._solve_h_subproblem(u, o, h, F_dx, F_dy, beta)
            
            # 2. Решение u-подзадачи (Algorithm 3)
            # u = argmin (λ/2){u - g ln u} + (β/2)||ho - u||^2
            # Первый член - KL-дивергенция (Poisson log-likelihood)
            u = self._solve_u_subproblem(g, h, o, beta, self.lambda_)
            
            # 3. Решение o-подзадачи (Algorithm 4)
            # o = argmin (β/2)||ho - u||^2 + ||∇o||_0
            o = self._solve_o_subproblem(u, h, o, F_dx, F_dy, beta)
            
            # Вычисление ошибки ||ho - u||^2 и обновление beta (Ур. 12)
            h_padded = _pad_kernel_for_fft(h, (H, W))
            F_h = fft2(h_padded)
            ho = np.real(ifft2(F_h * fft2(o)))
            
            diff_norm = np.linalg.norm(ho - u)**2
            
            # Обновление beta: β = β + min(T, τ ||ho - u||^2)
            beta_new = beta + min(self.T, self.tau * diff_norm)
            
            self.history['beta'].append(beta)
            self.history['error'].append(diff_norm)
            
            if self.verbose:
                print(f"Iter {k+1}/{self.max_iter}: β={beta:.2f}, Error={diff_norm:.4e}")
            
            # Критерий остановки (Algorithm 1, line 2: ||ho - u||^2 < ε)
            if diff_norm < 1e-6:
                break
                
            beta = beta_new

        # Не-слепая деконволюция (Algorithm 5)
        # Улучшение качества изображения при фиксированном h
        if self.verbose:
            print("Starting Non-blind deconvolution step...")
            
        f_final = self._nonblind_deconvolution(g, h, o, F_dx, F_dy)
        
        self.hyperparams = {
            'beta_final': beta,
            'lambda': self.lambda_,
            'mu': self.mu
        }
        
        self.timer = time.time() - start_time
        
        return f_final, h

    def _solve_h_subproblem(self, u, o, h_prev, F_dx, F_dy, beta):
        """
        Решение подзадачи для h с помощью IRLS и CG (Algorithm 2).
        
        Минимизирует (Ур. 13-14):
        h = argmin (β/2)||d(ho) - du||^2 + μ TV(h)
        
        Используется модифицированный fidelity term с производными для ускорения (Section III.B).
        """
        H, W = u.shape
        kh, kw = h_prev.shape
        
        # Параметры IRLS (l_max=5 согласно Section V.F)
        l_max = 5
        cg_maxiter = 50
        
        h_curr = h_prev.copy()
        
        # d(u) в частотной области: F(d_x u), F(d_y u)
        F_u = fft2(u)
        F_du_x = F_dx * F_u
        F_du_y = F_dy * F_u
        
        F_o = fft2(o)
        
        for l in range(l_max):
            # Вычисление весовой матрицы H для TV (Ур. 14, 15)
            # H - диагональная матрица sqrt((d_x h)^2 + (d_y h)^2).
            # В IRLS нам нужно H^(-1).
            
            # Операторы градиента в пространственной области (для малого h)
            k_dx = np.array([[0,0,0],[0,-1,1],[0,0,0]])
            k_dy = np.array([[0,0,0],[0,-1,0],[0,1,0]])
            
            dh_x = convolve2d(h_curr, k_dx, mode='same', boundary='wrap')
            dh_y = convolve2d(h_curr, k_dy, mode='same', boundary='wrap')
            
            # Weights = H^(-1)
            grad_mag = np.sqrt(dh_x**2 + dh_y**2 + EPSILON)
            Weights = 1.0 / grad_mag
            
            # Настройка системы Ax = b для CG
            # Ур. 15: A = β (d(O)^T d(O)) + μ (d^T H^(-1) d)
            
            # 1. Fidelity term (постоянная часть A): β * |F(d o)|^2
            # Вычисляется в Фурье для скорости (свертка с o)
            F_term_fidelity = beta * (np.abs(F_dx * F_o)**2 + np.abs(F_dy * F_o)**2)
            
            # 2. Правая часть b (Ур. 15)
            # b = β [ (d_x O)^T d_x u + (d_y O)^T d_y u ]
            # В Фурье: β [ conj(F_dx F_o) * (F_dx F_u) + ... ]
            rhs_freq = beta * (
                np.conj(F_dx * F_o) * F_du_x + 
                np.conj(F_dy * F_o) * F_du_y
            )
            # Перевод в пространственную область и обрезка до размера ядра
            b_spatial = np.real(ifft2(rhs_freq))
            b_vec = _extract_kernel_from_padded(b_spatial, (kh, kw)).flatten()
            
            def matvec(h_vec):
                """Оператор умножения матрицы A на вектор h."""
                h_k = h_vec.reshape((kh, kw))
                
                # 1. Fidelity term (через FFT)
                h_pad = _pad_kernel_for_fft(h_k, (H, W))
                F_h_k = fft2(h_pad)
                res_fidelity_freq = F_term_fidelity * F_h_k
                res_fidelity = _extract_kernel_from_padded(
                    np.real(ifft2(res_fidelity_freq)), (kh, kw)
                )
                
                # 2. Regularization term (μ TV) - пространственная область
                # μ [ d_x^T (Weights * d_x h) + d_y^T (Weights * d_y h) ]
                dx_h = convolve2d(h_k, k_dx, mode='same', boundary='wrap')
                dy_h = convolve2d(h_k, k_dy, mode='same', boundary='wrap')
                
                # Взвешивание
                w_dx = Weights * dx_h
                w_dy = Weights * dy_h
                
                # Сопряженные операторы (транспонированная свертка)
                # Для wrap boundary d_x^T ~ [-1, 1] -> [1, -1] с flip
                k_dx_T = np.array([[0,0,0],[1,-1,0],[0,0,0]])
                k_dy_T = np.array([[0,1,0],[0,-1,0],[0,0,0]])
                
                res_reg = self.mu * (
                    convolve2d(w_dx, k_dx_T, mode='same', boundary='wrap') +
                    convolve2d(w_dy, k_dy_T, mode='same', boundary='wrap')
                )
                
                return (res_fidelity + res_reg).flatten()
            
            A_op = LinearOperator((kh*kw, kh*kw), matvec=matvec)
            
            # Решение системы
            h_flat, _ = cg(A_op, b_vec, x0=h_curr.flatten(), maxiter=cg_maxiter, tol=1e-5)
            h_curr = h_flat.reshape((kh, kw))
            
            # Проекция и нормализация (п.5-6 Algorithm 2)
            h_curr = np.maximum(h_curr, 0)
            h_sum = h_curr.sum()
            if h_sum > EPSILON:
                h_curr /= h_sum
        
        return h_curr

    def _solve_u_subproblem(self, g, h, o, beta, lam):
        """
        Решение подзадачи для u (Algorithm 3).
        
        Минимизирует (Ур. 10):
        u = argmin (λ/2){u - g ln u} + (β/2)||ho - u||^2
        
        Решение через квадратное уравнение (Ур. 17-18).
        """
        H, W = g.shape
        
        # ho = h * o
        h_padded = _pad_kernel_for_fft(h, (H, W))
        F_h = fft2(h_padded)
        F_o = fft2(o)
        ho = np.real(ifft2(F_h * F_o))
        
        # Коэффициенты квадратного уравнения: 2β u^2 + (λ - 2β ho) u - λg = 0
        term_b = lam - 2 * beta * ho
        delta = term_b**2 + 8 * lam * beta * g
        
        # Выбираем положительный корень (Ур. 18)
        u = (2 * beta * ho - lam + np.sqrt(np.maximum(delta, 0))) / (4 * beta)
        
        return np.maximum(u, EPSILON)

    def _solve_o_subproblem(self, u, h, o_prev, F_dx, F_dy, beta):
        """
        Решение подзадачи для o (Algorithm 4).
        
        Минимизирует (Ур. 11):
        o = argmin (β/2)||ho - u||^2 + ||∇o||_0
        
        Использует метод расщепления (Variable Splitting) с параметром γ.
        """
        H, W = u.shape
        
        # Предвычисления для формулы обновления o
        h_padded = _pad_kernel_for_fft(h, (H, W))
        F_h = fft2(h_padded)
        F_h_conj = np.conj(F_h)
        F_u = fft2(u)
        
        # β * |F(h)|^2
        denom_fidelity = beta * (np.abs(F_h)**2)
        
        # |F(dx)|^2 + |F(dy)|^2
        F_grad_sq = np.abs(F_dx)**2 + np.abs(F_dy)**2
        
        o = o_prev.copy()
        
        # Вспомогательные переменные w (градиенты)
        w_x = np.zeros_like(o)
        w_y = np.zeros_like(o)
        
        gamma = self.gamma_init
        # Эмпирическое ограничение gamma_max = 1e4 (Section V.F)
        gamma_max = 1e4
        
        while gamma <= gamma_max:
            
            # 1. Обновление o (Ур. 28, решение через FFT Ур. 31)
            # o = argmin β/2 ||ho - u||^2 + γ/2 ||w - ∇o||^2
            # В Фурье:
            # (β F_h* F_u + γ (F_dx* F_wx + F_dy* F_wy)) / (β |F_h|^2 + γ (|F_dx|^2 + |F_dy|^2))
            
            F_wx = fft2(w_x)
            F_wy = fft2(w_y)
            
            num = beta * F_h_conj * F_u + gamma * (np.conj(F_dx) * F_wx + np.conj(F_dy) * F_wy)
            den = denom_fidelity + gamma * F_grad_sq
            
            o = np.real(ifft2(num / (den + EPSILON)))
            o = np.maximum(o, 0)
            
            # 2. Обновление w (Hard Thresholding, Ур. 29)
            # w = argmin γ/2 ||w - ∇o||^2 + ||w||_0
            # Решение: w = ∇o, если |∇o|^2 > 2/γ, иначе 0.
            
            # Вычисляем градиенты текущего o
            grad_o_x = np.real(ifft2(F_dx * fft2(o)))
            grad_o_y = np.real(ifft2(F_dy * fft2(o)))
            
            grad_mag_sq = grad_o_x**2 + grad_o_y**2
            
            # Порог 2/γ выводится из L0-минимизации (см. Xu et al. 2011)
            mask = grad_mag_sq > (2.0 / gamma)
            
            w_x = np.where(mask, grad_o_x, 0)
            w_y = np.where(mask, grad_o_y, 0)
            
            # 3. Увеличение gamma (Ур. 30)
            gamma *= self.sigma
            
        return o

    def _nonblind_deconvolution(self, g, h, o_init, F_dx, F_dy):
        """
        Не-слепая деконволюция (Algorithm 5).
        
        Решает задачу с TV регуляризацией методом расщепления (Ур. 36-37).
        Минимизирует:
        (ξ/2) KL(x, g) + (η/2)||ho - x||^2 + (η/2)||o - y||^2 + TV(y)
        
        где x ≈ ho, y ≈ o.
        """
        H, W = g.shape
        x = convolve2d(o_init, h, mode='same', boundary='wrap')
        y = o_init.copy()
        o = o_init.copy()
        
        eta = self.eta_init
        
        h_padded = _pad_kernel_for_fft(h, (H, W))
        F_h = fft2(h_padded)
        F_h_conj = np.conj(F_h)
        
        for k in range(self.nonblind_iter):
            # 1. x-подзадача (Ур. 38)
            # Аналогично u-подзадаче, только параметры ξ и η
            # 2η x^2 + (ξ - 2η ho) x - ξ g = 0
            
            h_o = np.real(ifft2(F_h * fft2(o)))
            
            term_b = self.xi - 2 * eta * h_o
            D = term_b**2 + 8 * self.xi * eta * g
            
            x = (2 * eta * h_o - self.xi + np.sqrt(np.maximum(D, 0))) / (4 * eta)
            x = np.maximum(x, EPSILON)
            
            # 2. y-подзадача (Ур. 39) - TV denoising
            # y = argmin η/2 ||o - y||^2 + ||∇y||_1
            # Используем FTVd (Wang et al. 2008), на который ссылается статья [49]
            y = self._solve_y_ftvd(o, eta, F_dx, F_dy)
            
            # 3. o-подзадача (Ур. 40, 42)
            # o = argmin ||ho - x||^2 + ||o - y||^2
            # Решение в Фурье: (F_h* F_x + F_y) / (|F_h|^2 + 1)
            F_x = fft2(x)
            F_y = fft2(y)
            
            num = F_h_conj * F_x + F_y
            den = np.abs(F_h)**2 + 1.0
            o = np.real(ifft2(num / den))
            o = np.maximum(o, 0)
            
            # 4. Обновление eta (Ур. 41)
            diff_norm = np.linalg.norm(o - y)**2
            eta = eta + self.zeta * diff_norm
            
        return o

    def _solve_y_ftvd(self, f, eta, F_dx, F_dy):
        """
        Решение подзадачи y (TV denoising) методом FTVd (Wang et al. 2008).
        Минимизирует: (eta/2)||y - f||^2 + ||∇y||_1
        
        Используется в Algorithm 5 (step 4).
        """
        # Параметры FTVd (внутренние, стандартные для метода)
        beta_tv = 1.0
        beta_max = 2**8
        
        y = f.copy()
        
        # Знаменатель D^T D в Фурье (для w-шага)
        F_DTD = np.abs(F_dx)**2 + np.abs(F_dy)**2
        
        F_f = fft2(f)
        
        while beta_tv < beta_max:
            # Метод расщепления для TV:
            # y = argmin eta/2 ||y-f||^2 + beta_tv/2 ||Dy - w||^2
            # w = argmin beta_tv/2 ||Dy - w||^2 + ||w||_1
            
            # Внутренние итерации
            for _ in range(5):
                # 1. w-шаг (Shrinkage)
                # Вычисляем градиенты
                g_x = np.real(ifft2(F_dx * fft2(y)))
                g_y = np.real(ifft2(F_dy * fft2(y)))
                
                # Soft Thresholding
                g_mag = np.sqrt(g_x**2 + g_y**2)
                # Порог = 1/beta_tv (так как вес TV=1, вес квадратичного члена=beta_tv)
                mask = g_mag > (1.0 / beta_tv)
                scale = 1.0 - 1.0 / (beta_tv * g_mag + EPSILON)
                scale[~mask] = 0
                
                w_x = g_x * scale
                w_y = g_y * scale
                
                # 2. y-шаг (FFT)
                # (eta I + beta D^T D) y = eta f + beta D^T w
                F_DTw = np.conj(F_dx) * fft2(w_x) + np.conj(F_dy) * fft2(w_y)
                
                num = eta * F_f + beta_tv * F_DTw
                den = eta + beta_tv * F_DTD
                
                y = np.real(ifft2(num / (den + EPSILON)))
            
            beta_tv *= 2.0
            
        return y

    def get_param(self) -> List[Tuple[str, Any]]:
        """Возвращает текущие гиперпараметры алгоритма."""
        return [
            ('kernel_shape', self.kernel_shape),
            ('lambda_', self.lambda_),
            ('mu', self.mu),
            ('beta_init', self.beta_init),
            ('gamma_init', self.gamma_init),
            ('xi', self.xi),
            ('eta_init', self.eta_init),
            ('max_iter', self.max_iter),
            ('nonblind_iter', self.nonblind_iter),
            ('verbose', self.verbose),
        ]
    
    def change_param(self, params: Dict[str, Any]) -> None:
        """Изменяет гиперпараметры алгоритма."""
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)
    
    def get_history(self) -> dict:
        """Возвращает историю сходимости."""
        return self.history
    
    def get_hyperparams(self) -> dict:
        """Возвращает оценённые гиперпараметры."""
        return self.hyperparams


# Обратная совместимость
def pbtvgr_blind_deconvolution(g, kernel_shape, **kwargs):
    """
    Обёртка для совместимости со старым API.
    """
    algo = PBTVGR(kernel_shape=kernel_shape, **kwargs)
    f_est, h_est = algo.process(g)
    return f_est, h_est, algo.hyperparams, algo.history


def run_algorithm(g, kernel_shape, **kwargs):
    """Обёртка для pbtvgr_blind_deconvolution()."""
    return pbtvgr_blind_deconvolution(g, kernel_shape, **kwargs)
