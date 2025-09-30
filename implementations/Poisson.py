import numpy as np
from scipy.signal import convolve2d
from numpy.fft import fft2, ifft2, fftshift
from scipy.sparse import diags
from scipy.sparse.linalg import cg
from skimage.restoration import denoise_tv_chambolle

class PoissonBlindDeconvolution:
    def __init__(self, lambda_=1e4, mu=0.04, beta_init=1.0,
                 gamma_init=2.0, xi=1e4, eta_init=1.0,
                 tau=10, T=100, sigma=2.0, zeta=10,
                 gamma_max=1e4, eta_max=2**14):
        """
        Полная реализация алгоритма слепой деконволюции для изображений с шумом Пуассона
        с TV и L0-нормальной регуляризацией градиентов согласно статье.

        Параметры:
        ----------
        lambda_ : float - параметр регуляризации для u (формула 10)
        mu : float - параметр TV регуляризации для PSF (формула 9)
        beta_init : float - начальное значение beta (формула 12)
        gamma_init : float - начальное значение gamma (формула 19)
        xi : float - параметр регуляризации для не-слепой деконволюции (формула 36)
        eta_init : float - начальное значение eta (формула 37)
        tau : float - скорость обновления beta (формула 12)
        T : float - максимальный шаг для beta (формула 12)
        sigma : float - коэффициент увеличения gamma (формула 22)
        zeta : float - скорость обновления eta (формула 41)
        gamma_max : float - максимальное значение gamma (формула 22)
        eta_max : float - максимальное значение eta (формула 22)
        """
        self.lambda_ = lambda_
        self.mu = mu
        self.beta = beta_init
        self.gamma_init = gamma_init
        self.xi = xi
        self.eta_init = eta_init
        self.tau = tau
        self.T = T
        self.sigma = sigma
        self.zeta = zeta
        self.gamma_max = gamma_max
        self.eta_max = eta_max

        # Операторы градиента (3x3 как в статье)
        self.kernel_x = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])  # Горизонтальный градиент
        self.kernel_y = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])   # Вертикальный градиент

    def blind_deconvolution(self, g, h_init, max_iter=100, eps=1e-6):
        """
        Основной алгоритм слепой деконволюции (Algorithm 1 в статье)

        Параметры:
        ----------
        g : ndarray - размытое/зашумленное изображение
        h_init : ndarray - начальная оценка PSF (обычно Гауссова)
        max_iter : int - максимальное число итераций
        eps : float - порог сходимости

        Возвращает:
        ----------
        h : ndarray - оцененная PSF
        o : ndarray - восстановленное изображение
        """
        # Инициализация (шаг 1 Algorithm 1)
        u = o = g.copy()
        h = h_init.copy()

        for _ in range(max_iter):
            # Шаг 1: Оптимизация h (формула 9)
            h = self.solve_h_subproblem(u, o, h)

            # Шаг 2: Оптимизация u (формула 10)
            u = self.solve_u_subproblem(g, h, o)

            # Шаг 3: Оптимизация o (формула 11)
            o = self.solve_o_subproblem(u, h, o)

            # Вычисление ошибки для обновления beta
            ho = convolve2d(o, h, mode='same')
            error = np.sum((ho - u)**2)

            # Проверка сходимости
            if error < eps:
                break

            # Обновление beta (формула 12)
            self.beta += min(self.T, self.tau * error)

        return h, o

    def nonblind_deconvolution(self, g, h, max_iter=100, eps=1e-6):
        """
        Не-слепая деконволюция с TV регуляризацией (Algorithm 5 в статье)

        Параметры:
        ----------
        g : ndarray - размытое/зашумленное изображение
        h : ndarray - оцененная PSF (из blind_deconvolution)
        max_iter : int - максимальное число итераций
        eps : float - порог сходимости

        Возвращает:
        ----------
        o : ndarray - финальное восстановленное изображение
        """
        # Инициализация (формула 37)
        o = g.copy()
        x = convolve2d(o, h, mode='same')
        y = o.copy()
        eta = self.eta_init

        for _ in range(max_iter):
            # Шаг 1: Оптимизация x (формула 38)
            x = self._solve_x_subproblem(g, h, o, eta)

            # Шаг 2: Оптимизация y (формула 39)
            y = self._solve_y_subproblem(o, eta)

            # Шаг 3: Оптимизация o (формула 40, 42)
            o = self._solve_o_nonblind(h, x, y)

            # Проверка сходимости
            error = np.sum((o - y)**2)
            if error < eps:
                break

            # Обновление eta (формула 41)
            eta = min(eta + self.zeta * error, self.eta_max)

        return o

    def solve_h_subproblem(self, u, o, h_init, l_max=5, cg_maxiter=100):
        """
        Реализация Algorithm 2 из статьи (рис. 2) для решения подзадачи h
        Соответствует формулам (13)-(15) из раздела B статьи

        Параметры:
            u: Размытое изображение (формула 13: du)
            o: Текущая оценка изображения (формула 13: do)
            h_init: Инициализация PSF (h^(0) = Gaussian PSF, п.1 Algorithm 2)
            l_max: Максимальное число итераций IRLS (п.2 Algorithm 2)
            cg_maxiter: Максимальное число итераций метода сопряженных градиентов

        Возвращает:
            h: Оценка PSF (п.8 Algorithm 2)
        """
        # =============================================
        # Инициализация (п.1 Algorithm 2)
        # h^(0) = Gaussian PSF
        # =============================================
        h = h_init.copy()

        # =============================================
        # Определение операторов производных
        # d = d_x + d_y (формула 14)
        # =============================================
        # Операторы производных (форвардные разности)
        dx = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])    # d_x оператор
        dy = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])     # d_y оператор

        # Сопряженные операторы (backward differences)
        dxT = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])    # d_x^T оператор
        dyT = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])     # d_y^T оператор

        # =============================================
        # Предварительное вычисление: d*u
        # Градиент размытого изображения u
        # =============================================
        du_dx = convolve2d(u, dx, mode='same', boundary='wrap')
        du_dy = convolve2d(u, dy, mode='same', boundary='wrap')

        # =============================================
        # Основной цикл IRLS (п.2 Algorithm 2)
        # while l ≤ l_max do
        # =============================================
        for l in range(l_max):

            # =========================================
            # Обновление весовой матрицы H (п.4 Algorithm 2)
            # H_jj = sqrt((d_x h)_j^2 + (d_y h)_j^2) (формула 14)
            # =========================================
            d_h_dx = convolve2d(h, dx, mode='same', boundary='wrap')
            d_h_dy = convolve2d(h, dy, mode='same', boundary='wrap')
            H_vals = np.sqrt(d_h_dx**2 + d_h_dy**2).flatten() + 1e-8

            # =========================================
            # Определение системы для CG (формула 15)
            # =========================================

            def A_operator(x_vec):
                """
                Оператор A для метода сопряженных градиентов
                Реализует левую часть уравнения 15
                """
                x = x_vec.reshape(h.shape)
                result = np.zeros_like(x)

                # Член данных: β * [d_x^T(d_x(x*o) - d_x(u)) + d_y^T(d_y(x*o) - d_y(u))]
                # Соответствует β/2 || d*(h*o) - d*u ||² в формуле 15

                # Вычисляем x*o (свертка)
                conv_x_o = convolve2d(o, x, mode='same', boundary='wrap')  # x*o

                # Вычисляем d(x*o) - d(u) (как в формуле 15)
                diff_dx = convolve2d(conv_x_o, dx, mode='same', boundary='wrap') - du_dx
                diff_dy = convolve2d(conv_x_o, dy, mode='same', boundary='wrap') - du_dy

                # Применяем сопряженные операторы
                term_data = self.beta * (
                    convolve2d(diff_dx, dxT, mode='same', boundary='wrap') +
                    convolve2d(diff_dy, dyT, mode='same', boundary='wrap')
                )
                result += term_data

                # Регуляризационный член: μ * [d_x^T(H^{-1}·d_x x) + d_y^T(H^{-1}·d_y x)]
                # Соответствует второму и третьему слагаемым в формуле 15
                d_x_x = convolve2d(x, dx, mode='same', boundary='wrap').flatten()
                d_y_x = convolve2d(x, dy, mode='same', boundary='wrap').flatten()

                # Умножение на H^{-1} (поэлементное деление)
                weighted_d_x = (d_x_x / H_vals).reshape(h.shape)
                weighted_d_y = (d_y_x / H_vals).reshape(h.shape)

                term_reg = self.mu * (
                    convolve2d(weighted_d_x, dxT, mode='same', boundary='wrap') +
                    convolve2d(weighted_d_y, dyT, mode='same', boundary='wrap')
                )
                result += term_reg

                return result.flatten()

            # =========================================
            # Правая часть уравнения: A*x = 0
            # Решаем уравнение A_operator(h) = 0
            # =========================================
            b = np.zeros_like(h.flatten())

            # =========================================
            # Решение системы методом CG (п.5 Algorithm 2)
            # h^(l+1) = argmin_h [формула 15]
            # =========================================
            # SciPy API compatibility: prefer rtol, fallback to tol
            try:
                h_flat, info = cg(A_operator, b, x0=h.flatten(),
                                  maxiter=cg_maxiter, rtol=1e-6)
            except TypeError:
                h_flat, info = cg(A_operator, b, x0=h.flatten(),
                                  maxiter=cg_maxiter, tol=1e-6)
            h = np.maximum(h_flat.reshape(h.shape), 0)

            # =========================================
            # Нормализация PSF (п.6 Algorithm 2)
            # h^(l+1) = h^(l+1) / sum_j h_j^(l+1)
            # =========================================
            h_sum = np.sum(h)
            if h_sum > 0:
                h /= h_sum

        return h

    def solve_u_subproblem(self, g, h, o):
        """
        Реализация Algorithm 3 из статьи (рис. 4) для решения подзадачи u
        Соответствует формулам (16)-(18) из раздела C статьи

        Параметры:
            g: зашумленное наблюдение (исходное размытое изображение)
            h: текущая оценка PSF
            o: текущая оценка изображения

        Возвращает:
            u: вспомогательная переменная, аппроксимирующая ho
        """
        # ======================================================
        # Вычисление ho (формула 16: член (ho)_i)
        # ho = h * o (свертка PSF и текущей оценки изображения)
        # ======================================================
        ho = convolve2d(o, h, mode='same')

        # ======================================================
        # Подготовка коэффициентов для решения уравнения (17)
        # Уравнение: 2β(u)_i^2 + [λ - 2β(ho)_i](u)_i - λg_i = 0
        # ======================================================
        # Коэффициенты квадратного уравнения (формула 17):
        # a = 2β
        # b = λ - 2β(ho)_i
        # c = -λg_i
        a = 2 * self.beta
        b = self.lambda_ - 2 * self.beta * ho
        c = -self.lambda_ * g

        # ======================================================
        # Решение квадратного уравнения (формула 18)
        # (u)_i = [ -b + sqrt(b^2 - 4ac) ] / (2a)
        # Для нашего случая:
        # (u)_i = [2β(ho)_i - λ + sqrt([λ - 2β(ho)_i]^2 + 8λβg_i)] / (4β)
        # ======================================================
        discriminant = b**2 - 4*a*c  # Дискриминант (подкоренное выражение из формулы 18)

        # Вычисление решения по формуле 18 (аналог п.1 Algorithm 3)
        # Обратите внимание, что discriminant всегда неотрицательный, так как:
        # [λ - 2β(ho)_i]^2 + 8λβg_i >= 0 (поскольку λ, β, g_i >= 0)
        sqrt_discr = np.sqrt(np.maximum(discriminant, 0))  # Защита от численных ошибок

        u = (2 * self.beta * ho - self.lambda_ + sqrt_discr) / (4 * self.beta)

        # ======================================================
        # Обеспечение положительности (формула 16 требует (u)_i > 0)
        # ======================================================
        u = np.maximum(u, 1e-6)  # Малое положительное значение вместо 0

        return u  # Возвращаем решение (соответствует п.2 Algorithm 3)

    def solve_o_subproblem(self, u, h, o_init):
        """
        Реализация Algorithm 4 из статьи для решения подзадачи o
        Соответствует формулам (19)-(22) из раздела D

        Решает: (o,w) = argmin { β/2||h*o - u||² + γ/2||w - o||² + ||∇w||₀ }

        Параметры:
        ----------
        u : ndarray
            Входное размытое изображение (формула 19)
        h : ndarray
            Оценка PSF (формула 19)
        o_init : ndarray
            Начальное приближение для o
        beta : float
            Параметр из основной задачи
        gamma_init : float
            Начальное значение параметра штрафа (формула 19)
        gamma_max : float
            Максимальное значение параметра штрафа (формула 22)
        sigma : float
            Коэффициент увеличения gamma (формула 22)

        Возвращает:
        -----------
        ndarray
            Восстановленное изображение w (согласно Algorithm 4)
        """
        # =============================================================
        # Инициализация (Algorithm 4, п.1)
        # w, γ = инициализация
        # =============================================================
        w = o_init.copy()
        gamma = self.gamma_init
        rows, cols = u.shape

        # =============================================================
        # Предварительные вычисления FFT для формулы (23)
        # o = F⁻¹[(βF*(h)◦F(u) + γF(w))/(βF*(h)◦F(h) + γ)]
        # =============================================================
        F_h = fft2(h, s=(rows, cols))
        F_u = fft2(u)
        F_h_conj = np.conj(F_h)
        F_denom_base = self.beta * (F_h_conj * F_h)  # βF*(h)◦F(h)

        # =============================================================
        # Основной цикл (Algorithm 4, п.2-6)
        # while γ ≤ γ_max do
        # =============================================================
        while gamma <= self.gamma_max:
            # =========================================================
            # Шаг 1: Оптимизация o - формула (20) и (23)
            # Computing o with (23)
            # =========================================================
            F_w = fft2(w)
            numerator = self.beta * F_h_conj * F_u + gamma * F_w
            denominator = F_denom_base + gamma

            # Регуляризация для избежания деления на ноль
            denominator = np.maximum(denominator, 1e-12)
            o_freq = numerator / denominator
            o = np.real(ifft2(o_freq))

            # =========================================================
            # Шаг 2: Оптимизация w - формула (21)
            # Computing w with the method in [39]
            # Используем Algorithm 1 из [39] для L0-сглаживания
            # =========================================================
            # Параметр λ для L0-сглаживания: λ = γ/2
            # так как решаем: min_w [γ/2||w - o||² + ||∇w||₀]
            lam_smoothing = 2.0 / gamma
            w = self.L0_gradient_minimization(o, lam=lam_smoothing,
                                      beta_max=1e5, kappa=2.0, max_iter=30)

            # =========================================================
            # Шаг 3: Увеличение γ - формула (22)
            # γ = σγ
            # =========================================================
            if gamma * self.sigma > self.gamma_max:
                break
            gamma *= self.sigma

        return w  # Согласно Algorithm 4: output w

    def L0_gradient_minimization(self, I, lam=0.02, beta_max=1e5, kappa=2.0, max_iter=30):
        """
        Реализация Algorithm 1 из статьи [39]:
        "Image Smoothing via L0 Gradient Minimization" by Xu et al. 2011

        Решает: min_S { Σ_p (S_p - I_p)^2 + λ · C(S) }
        где C(S) = #{p | |∂xS_p| + |∂yS_p| ≠ 0} (формула 5 из [39])

        Параметры:
        ----------
        I : ndarray
            Входное изображение (одноканальное)
        lam : float
            Параметр сглаживания (λ) - контролирует уровень сглаживания
        beta_max : float
            Максимальное значение β (1e5 согласно статье)
        kappa : float
            Коэффициент увеличения β (2.0 согласно статье)
        max_iter : int
            Максимальное количество итераций (20-30 согласно статье)

        Возвращает:
        -----------
        ndarray
            Сглаженное изображение S
        """
        # =============================================================
        # Инициализация (Algorithm 1 из [39], строка 2)
        # S ← I, β ← β₀, i ← 0
        # β₀ = 2λ согласно разделу 3 статьи [39]
        # =============================================================
        S = I.copy()
        beta = 2 * lam  # β₀ = 2λ
        rows, cols = I.shape

        # =============================================================
        # Предварительные вычисления FFT операторов
        # Для формулы (8) из [39]: решение в частотной области
        # =============================================================
        # Операторы градиента (форвардные разности)
        dx_kernel = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])  # ∂/∂x
        dy_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])   # ∂/∂y

        # FFT операторов градиента
        F_dx = fft2(dx_kernel, s=(rows, cols))
        F_dy = fft2(dy_kernel, s=(rows, cols))
        F_dx_conj = np.conj(F_dx)  # F*(∂/∂x)
        F_dy_conj = np.conj(F_dy)  # F*(∂/∂y)

        # FFT единичного оператора (дельта-функция)
        # ВАЖНО: F(1) - не единица, а Фурье-образ дельта-функции!
        delta = np.zeros((rows, cols))
        delta[0, 0] = 1  # Дельта-функция в начале координат
        F_1 = fft2(delta)

        # Предварительный вычисленный знаменатель для формулы (8)
        F_denom = F_1 + beta * (F_dx_conj * F_dx + F_dy_conj * F_dy)

        # =============================================================
        # Основной цикл (Algorithm 1 из [39], строки 3-7)
        # =============================================================
        for iteration in range(max_iter):
            if beta >= beta_max:
                break

            # =========================================================
            # Шаг 1: Решение подзадачи (h, v) - формула (12) из [39]
            # With S, solve for h_p and v_p in Eq. (12)
            # =========================================================
            # Вычисление градиентов S
            grad_x = convolve2d(S, dx_kernel, mode='same', boundary='wrap')
            grad_y = convolve2d(S, dy_kernel, mode='same', boundary='wrap')

            # Вычисление (h, v) согласно формуле (12) из [39]
            threshold = lam / beta
            grad_magnitude_sq = grad_x**2 + grad_y**2

            # Бинарная маска: 1 если сохраняем градиент, 0 если обнуляем
            mask = grad_magnitude_sq > threshold

            # Применение порога (формула 12 из [39])
            h = np.where(mask, grad_x, 0)
            v = np.where(mask, grad_y, 0)
            # Соответствует: (h_p, v_p) = { (0, 0) если (∂xS_p)² + (∂yS_p)² ≤ λ/β
            # { (∂xS_p, ∂yS_p) иначе

            # =========================================================
            # Шаг 2: Решение подзадачи S - формула (8) из [39]
            # With h and v, solve for S with Eq. (8)
            # =========================================================
            # Вычисление числителя для формулы (8)
            F_I = fft2(I)
            F_h = fft2(h)
            F_v = fft2(v)

            numerator = F_I + beta * (F_dx_conj * F_h + F_dy_conj * F_v)

            # Решение в частотной области
            S_freq = numerator / F_denom
            # Соответствует: S = F⁻¹[ (F(I) + β(F*(∂x)F(h) + F*(∂y)F(v))) / (F(1) + β(F*(∂x)F(∂x) + F*(∂y)F(∂y))) ]
            S = np.real(ifft2(S_freq))

            # =========================================================
            # Шаг 3: Увеличение β - формула из раздела 3 [39]
            # β ← κβ
            # =========================================================
            beta = min(beta * kappa, beta_max)

        return S

    def _solve_x_subproblem(self, g, h, o, eta):
        """
        Решение подзадачи x для не-слепой деконволюции (формула 38)
        Аналогично решению подзадачи u в Algorithm 3

        Проблема: x = argmin_x (ξ/2)∑[x_i - g_i ln(x_i)] + (η/2)||ho - x||²

        Параметры:
        ----------
        g : ndarray - наблюдаемое изображение
        h : ndarray - PSF
        o : ndarray - текущая оценка изображения
        eta : float - параметр штрафа

        Возвращает:
        ----------
        ndarray - вспомогательная переменная x
        """
        ho = convolve2d(o, h, mode='same')

        # Нужно решить: ξ(1 - g/x) + η(x - ho) = 0
        # Решаем квадратное уравнение: ηx² + (ξ - ηho)x - ξg = 0

        # Но на самом деле, давайте решим это правильно, используя тот же подход, что и Алгоритм 3
        # Уравнение, которое нужно решить: ξ(1 - g/x) + η(x - ho) = 0
        # Умножить на x: ξ(x - g) + ηx(x - ho) = 0
        # ηx² + (ξ - ηho)x - ξg = 0

        a = eta
        b = self.xi - eta * ho
        c = -self.xi * g

        discriminant = b**2 - 4*a*c
        sqrt_discr = np.sqrt(np.maximum(discriminant, 0))

        # Квадратичная формула: x = [-b ± sqrt(b² - 4ac)] / (2a)
        x = (-b + sqrt_discr) / (2 * a)

        return np.maximum(x, 1e-6)

    def _solve_y_subproblem(self, o, eta):
        """
        Решение TV-регуляризованной задачи шумоподавления для подзадачи y с использованием алгоритма FTVd [49]

        Проблема: y = argmin_y (η/2)||o - y||² + TV(y)
        Соответствует задаче FTVd с K=I, μ=η, f=o

        Параметры:
        ----------
        o : ndarray - входное изображение
        eta : float - параметр штрафа (η)

        Возвращает:
        ----------
        ndarray - очищенное изображение y
        """
        # Параметры из статьи FTVd
        beta_init = 1.0      # β0 из Algorithm 2
        beta_max = 2**14     # βmax из Algorithm 2
        max_iter_inner = 10  # Внутренние итерации для Algorithm 1
        tol = 1e-4           # Допуск сходимости

        # Инициализация переменных
        y = o.copy()
        beta = beta_init

        # Определение операторов градиента (форвардные разности)
        kernel_x = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])  # D^(1)
        kernel_y = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])   # D^(2)

        # Предварительное вычисление FFT операторов градиента с периодическим дополнением
        def pad_kernel_to_shape(kernel, target_shape):
            padded = np.zeros(target_shape)
            h, w = kernel.shape
            padded[:h, :w] = kernel
            return padded

        # Дополнение ядер до размера изображения
        kernel_x_padded = pad_kernel_to_shape(kernel_x, o.shape)
        kernel_y_padded = pad_kernel_to_shape(kernel_y, o.shape)

        # Вычисление FFT операторов градиента (формула 2.4)
        F_D1 = fft2(kernel_x_padded)
        F_D2 = fft2(kernel_y_padded)
        F_D1_conj = np.conj(F_D1)  # F(D^(1))*
        F_D2_conj = np.conj(F_D2)  # F(D^(2))*

        # Предварительное вычисление постоянной части знаменателя (формула 2.4)
        denom_base = F_D1_conj * F_D1 + F_D2_conj * F_D2

        # FFT входного изображения (остается постоянным)
        F_o = fft2(o)

        # Основной алгоритм FTVd (Algorithm 2)
        while beta <= beta_max:
            # Вычисление полного знаменателя для текущего β (адаптированная формула 2.4)
            # Для нашего случая: K = I, поэтому F(K)*◦F(K) = 1, и μ/β = η/β
            full_denom = denom_base + (eta / beta)
            full_denom[full_denom == 0] = 1e-12

            # Инициализация вспомогательных переменных
            w1 = np.zeros_like(o)  # компонента w_1
            w2 = np.zeros_like(o)  # компонента w_2

            # Algorithm 1: Чередующаяся минимизация для фиксированного β
            for _ in range(max_iter_inner):
                y_prev = y.copy()

                # Шаг 1: Вычисление градиентов y
                D1_y = convolve2d(y, kernel_x, mode='same', boundary='wrap')  # D^(1)y
                D2_y = convolve2d(y, kernel_y, mode='same', boundary='wrap')  # D^(2)y

                # Шаг 2: Операция сжатия (формула 2.2)
                norm_Dy = np.sqrt(D1_y**2 + D2_y**2 + 1e-8)
                shrink_factor = np.maximum(1 - 1/(beta * norm_Dy), 0)

                w1 = D1_y * shrink_factor
                w2 = D2_y * shrink_factor

                # Шаг 3: Решение для y через FFT (адаптированная формула 2.4)
                F_w1 = fft2(w1)
                F_w2 = fft2(w2)

                # Числитель: F(D^(1))*◦F(w1) + F(D^(2))*◦F(w2) + (η/β)F(o)
                numerator_fft = (F_D1_conj * F_w1 +
                              F_D2_conj * F_w2 +
                              (eta / beta) * F_o)

                y = np.real(ifft2(numerator_fft / full_denom))

                # Проверка сходимости
                if np.linalg.norm(y - y_prev) < tol:
                    break

            # Продолжение: увеличение β (шаг 2 Algorithm 2)
            beta = min(2 * beta, beta_max)

        return y

    def _solve_o_nonblind(self, h, x, y):
        """Решение подзадачи o для не-слепой деконволюции (формула 42)"""
        F_h = fft2(h, s=x.shape)
        F_x = fft2(x)
        F_y = fft2(y)

        numerator = np.conj(F_h) * F_x + F_y
        denominator = np.conj(F_h) * F_h + 1
        denominator[denominator == 0] = 1e-12

        return np.real(ifft2(numerator / denominator))
