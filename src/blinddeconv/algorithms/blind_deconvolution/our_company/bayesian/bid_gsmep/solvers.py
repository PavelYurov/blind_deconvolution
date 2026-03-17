import numpy as np
from scipy.fft import fft2, ifft2
from scipy.optimize import minimize_scalar
from .utils import *

def solve_image_subproblem(g, h_kernel, filters_w, gsm_params, lambda_reg, beta, num_inner_iter=1):
    """
    Решает Problem 1: Оценка изображения o при фиксированном h.
    Метод: Half-Quadratic Splitting (Eq. 8 - Eq. 14).
    
    Args:
        g: размытое изображение
        h_kernel: текущая оценка ядра
        filters_w: набор фильтров GSM FoE (K, s, s)
        gsm_params: (variances, weights)
        lambda_reg: коэффициент регуляризации
        beta: параметр splitting penalty
    
    Returns:
        o_est: обновленная оценка изображения
    """
    H, W = g.shape
    K = len(filters_w)
    variances, weights = gsm_params
    
    # Предварительные вычисления в частотной области (Eq. 13)
    G_f = fft2(g)
    H_f = psf2otf(h_kernel, (H, W))
    H_conj_H = np.abs(H_f)**2
    
    # Фильтры в частотной области
    W_f = np.array([psf2otf(fw, (H, W)) for fw in filters_w]) # (K, H, W)
    W_conj_W_sum = np.sum(np.abs(W_f)**2, axis=0) # Sum W_k^* W_k
    
    # Знаменатель Eq. (13)
    denominator = lambda_reg * H_conj_H + beta * W_conj_W_sum
    
    # Инициализация o (можно взять g)
    o_est = g.copy()
    
    # Вспомогательные переменные x_k (K, H, W)
    x = np.zeros((K, H, W))
    
    for _ in range(num_inner_iter):
        # Sub-problem 2: Update x_k (Eq. 14)
        # Fix o, optimize x_k.
        # (x_k)_i = argmin beta/2 (w_k * o - x_i)^2 - ln psi(x_i)
        # Это поэлементная оптимизация. 
        # В статье упоминается LUT. Здесь используем численный метод Ньютона 
        # или аппроксимацию, так как LUT требует точных GSM параметров.
        
        # Вычисляем отклики фильтров: v_k = w_k * o
        o_f = fft2(o_est)
        v = np.zeros_like(x)
        for k in range(K):
            v[k] = np.real(ifft2(o_f * W_f[k]))
            
        # Решаем Eq. 14: min beta/2 (v - x)^2 - ln(psi(x))
        # grad = beta(x - v) - (ln psi(x))' = 0
        # beta(x - v) + (-ln psi)'(x) = 0
        # Используем 1 шаг метода Ньютона или просто градиентный спуск, 
        # так как задача выпукла локально.
        # Для простоты и скорости: x_new = v - (1/beta) * deriv(-ln psi(v))
        # Это приближение первого порядка.
        
        grads_penalty = gsm_neg_log_derivative(v, variances, weights)
        x = v - (1.0 / beta) * grads_penalty
        
        # Sub-problem 1: Update o (Eq. 13)
        # Fix x_k, optimize o.
        # O(u) = (lambda H* G + beta sum W_k* X_k) / denominator
        
        numerator_part2 = np.zeros_like(G_f)
        for k in range(K):
            X_f_k = fft2(x[k])
            numerator_part2 += np.conj(W_f[k]) * X_f_k
            
        numerator = lambda_reg * np.conj(H_f) * G_f + beta * numerator_part2
        
        o_f_new = numerator / (denominator + 1e-8)
        o_est = np.real(ifft2(o_f_new))
        
    return o_est

def solve_psf_subproblem(g, o, kernel_size, tau, p_norm, iterations_irls=5, threshold_c_o=0.01):
    """
    Решает Problem 2: Оценка PSF h при фиксированном изображении o.
    Метод: IRLS с выбором градиентов (Eq. 15 - Eq. 18).
    
    Args:
        g: размытое изображение
        o: текущая оценка латентного изображения
        kernel_size: размер ядра (нечетный, например 27)
        tau: вес регуляризации PSF
        p_norm: p для lp-norm (обычно 1.5)
        iterations_irls: количество итераций IRLS
        threshold_c_o: порог для градиентов изображения (Eq. 16)
    
    Returns:
        h_est: оценка PSF
    """
    H, W = g.shape
    ks = kernel_size
    
    # 1. Gradient Selecting Method (текст перед Eq. 15)
    # Вычисляем производные
    d1, d2 = gradient_filters()
    
    # Градиенты g
    g_d1, g_d2 = compute_grads(g)
    
    # Градиенты o
    o_d1, o_d2 = compute_grads(o)
    
    # Порог T1 (Eq. 16)
    # В статье не указан точный алгоритм выбора c_o, обычно это эвристика 
    # или фиксированное значение. Используем переданный параметр.
    o_d1_t = threshold_gradients(o_d1, threshold_c_o)
    o_d2_t = threshold_gradients(o_d2, threshold_c_o)
    
    # Формируем матрицы для системы Eq. 18.
    # Eq. 15: min lambda/2 ||dg - h * do_thresh||^2 + tau ||dh||_p^p
    # В статье (Eq. 18) lambda не указана явно как множитель перед data term,
    # но в Eq. 15 она есть. В Eq. 18 она сокращена или подразумевается баланс.
    # Следуем Eq. 18:
    # (sum M_r^T M_r + tau/lambda * sum D_r^T H_r^-1 D_r) h = sum M_r^T (d_r g)
    # Здесь lambda из Eq. 15 заменена на 1 (относительно tau). 
    # Но в Eq. 18 есть явная 'lambda' перед data term.
    # Проверим Eq. 18: h = argmin lambda/2 ... + tau ...
    # Значит решаем систему (lambda * A + tau * B) h = lambda * b
    
    # Переходим к FFT для построения операторов свертки M_r
    # M_r соответствует свертке с o_dr_t.
    # Поскольку ядро маленькое, решать систему можно через Conjugate Gradient (CG)
    # используя FFT для быстрого умножения матриц.
    
    # Инициализация h (например, дельта-функция или предыдущее значение)
    h_est = np.zeros((ks, ks))
    h_est[ks//2, ks//2] = 1.0
    
    pad_h = (H, W)
    
    # FFT градиентов o (это операторы M_r)
    M1_f = fft2(o_d1_t)
    M2_f = fft2(o_d2_t)
    
    # Правая часть системы (lambda * sum M_r^T d_r g)
    # M^T соответствует корреляции, в частотной области это conj(M_f)
    g_d1_f = fft2(g_d1)
    g_d2_f = fft2(g_d2)
    
    # RHS = lambda * (ifft(conj(M1)*G1) + ifft(conj(M2)*G2))
    # Но нам нужно вырезать центральную часть размером ks x ks, так как h маленькое.
    # ВАЖНО: Eq. 18 формулируется для вектора h.
    rhs_full = np.real(ifft2(np.conj(M1_f) * g_d1_f + np.conj(M2_f) * g_d2_f))
    # Центрируем и вырезаем
    rhs_full = np.roll(rhs_full, (ks//2, ks//2), axis=(0,1)) 
    b = rhs_full[:ks, :ks].flatten() # Вектор b
    
    # Операторы производных для h (D_r)
    # D1, D2 для h (размер ks x ks)
    # Используем разреженные матрицы или свертку в малом окне.
    # Проще сделать через FFT на размере ядра (с паддингом для корректности linear conv)
    
    # Для CG нам нужна функция, вычисляющая Ax
    lambda_val = 1.0 # В Eq. 18 стоит lambda. Поставим 1.0, а tau будет регулировать соотношение.
    # В статье lambda используется в Problem 1 и Problem 2, но обычно это разные lambda?
    # В Eq. 6 общая формула: lambda/2 data_term + regularization.
    # В Eq. 18 коэффициент lambda сохранен.
    # Мы будем использовать переданный tau, но lambda внутри этой функции возьмем большой, 
    # так как data term главнее. В статье (раздел III, текст после Eq. 19): 
    # "assign lambda a relative small value... finally... larger lambda".
    # Пусть здесь lambda = 1.0 (зашито в RHS), а tau масштабируется.
    
    for l in range(iterations_irls):
        # 2. Вычисление весовых матриц H_r (Eq. 17)
        # H_r = diag(|d_r h|^{p-2})
        h_curr = h_est
        
        # Градиенты текущего h
        # Используем валидную свертку, чтобы не вылезти за границы
        # Но для простоты используем ту же функцию compute_grads на маленьком размере
        hd1, hd2 = compute_grads(h_curr) # returns same shape as h_curr
        
        # Добавляем epsilon для стабильности (|x|^(p-2) при x->0 взлетает)
        eps = 1e-6
        w1 = (np.abs(hd1) + eps) ** (p_norm - 2.0)
        w2 = (np.abs(hd2) + eps) ** (p_norm - 2.0)
        
        W1 = w1.flatten()
        W2 = w2.flatten()
        
        # Функция умножения матрицы системы на вектор h_vec
        def matvec(h_vec):
            h_mat = h_vec.reshape((ks, ks))
            
            # Часть 1: lambda * sum M_r^T M_r h
            # Свертка h с o_d1_t, затем корреляция с o_d1_t
            h_pad = np.zeros((H, W))
            h_pad[:ks, :ks] = h_mat
            # shift to center for FFT conv consistency with RHS
            h_pad = np.roll(h_pad, (-ks//2, -ks//2), axis=(0,1))
            
            H_f_loc = fft2(h_pad)
            
            # M^T M h = ifft( |M|^2 H )
            res1_f = (np.abs(M1_f)**2 + np.abs(M2_f)**2) * H_f_loc
            res1 = np.real(ifft2(res1_f))
            
            # Unshift and crop
            res1 = np.roll(res1, (ks//2, ks//2), axis=(0,1))
            res1_crop = res1[:ks, :ks]
            
            term1 = lambda_val * res1_crop.flatten()
            
            # Часть 2: tau * sum D_r^T H_r^-1 D_r
            # H_r в Eq 17 - это матрица весов. В Eq 18 она в -1 степени?
            # В Eq 17: H_r = diag(|dh|^p-2). 
            # В Eq 15 регуляризатор tau * ||dh||_p^p = tau * sum |dh|^p = tau * sum (dh)^T (|dh|^p-2) dh
            # = tau * h^T D^T W D h.
            # В Eq 18 написано H_r^{-1}. Это может быть опечатка OCR или статьи, 
            # так как IRLS обычно приводит к W * x.
            # Если H_r определен как diag(|dh|^{p-2}), то матрица системы A = X^T H_r X.
            # Проверим текст: "matrix H_r ... diag(|d h|^{p-2})".
            # Eq 18: ... + tau (d1 h)^T (H1)^-1 (d1 h) ...
            # Если H определяется как веса, то обычно они идут без инверсии.
            # НО! В OCR тексте: H = diag(|dh|^(p-2)).
            # А в формуле (18) стоит H^-1.
            # Если |dh|^(p-2) стоит в знаменателе, то это |dh|^(2-p).
            # Обычно для L_p (p < 2) веса равны |x|^(p-2).
            # Если мы умножаем на H^-1, то множитель становится |x|^(2-p). Это странно для p=1.5.
            # Скорее всего, в Eq 18 H_r обозначает ковариационную матрицу (как в Gaussian), 
            # тогда обратная ей — это precision matrix (веса).
            # НО текст говорит H_r = diag(...).
            # Предположим, что стандартный IRLS для L_p нормы:
            # ||Ax||_p^p ~ ||W^(1/2) Ax||_2^2 -> A^T W A. где W = |Ax|^(p-2).
            # Значит, нам нужно умножать на веса.
            # Если в формуле (18) стоит H^-1, значит H_r там определено как |dh|^(2-p) (обратное к весам).
            # В коде я буду использовать стандартный IRLS подход: term = D^T * W * D.
            # Где W рассчитано выше как |dh|^(p-2).
            
            # D1 * h
            dh1, dh2 = compute_grads(h_mat)
            
            # W * D * h
            wdh1 = W1 * dh1.flatten()
            wdh2 = W2 * dh2.flatten()
            
            # D^T * (W * D * h). D^T - это свертка с перевернутым фильтром (корреляция)
            # Для простоты аппроксимируем транспонирование градиентного фильтра:
            # D1 = [1, -1], D1^T ~ свертка с [-1, 1] (со сдвигом)
            # Реализуем через те же compute_grads но с учетом знаков.
            # D^T x = - div x.
            # divergence discrete:
            wdh1_r = wdh1.reshape(ks, ks)
            wdh2_r = wdh2.reshape(ks, ks)
            
            # Divergence (обратная операция к градиенту)
            # div([gx, gy]) approx dx(gx) + dy(gy) (с учетом сдвигов)
            # Используем FFT для D^T, так надежнее
            dh1_f = fft2(wdh1_r, s=(ks, ks)) # Zero pad not needed strictly if cyclic
            dh2_f = fft2(wdh2_r, s=(ks, ks))
            
            d1_kern, d2_kern = gradient_filters()
            d1_k_f = psf2otf(d1_kern, (ks, ks))
            d2_k_f = psf2otf(d2_kern, (ks, ks))
            
            # D^T соответствует conj(Filters)
            term2_f = np.conj(d1_k_f) * dh1_f + np.conj(d2_k_f) * dh2_f
            term2 = np.real(ifft2(term2_f)).flatten()
            
            return term1 + tau * term2

        # Solve linear system using CG
        from scipy.sparse.linalg import cg, LinearOperator
        A = LinearOperator((ks*ks, ks*ks), matvec=matvec)
        h_vec, info = cg(A, b, x0=h_est.flatten(), atol=1e-5, maxiter=50)
        h_est = h_vec.reshape((ks, ks))
        
        # 3. Thresholding and Normalization (Eq. 19)
        # В статье это делается в конце или на каждой итерации IRLS?
        # "In each iteration... Eq(18)... We also use a threshold function..."
        # Значит внутри цикла.
        h_est = threshold_psf(h_est, 0.0) # c_h можно поставить small value, но часто просто non-negativity
        h_est[h_est < 0] = 0 # Non-negativity constraint (стандарт для PSF)

    return h_est