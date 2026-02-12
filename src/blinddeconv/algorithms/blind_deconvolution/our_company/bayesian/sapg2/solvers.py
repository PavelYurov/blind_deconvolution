import numpy as np
from numpy.fft import fft2, ifft2
from typing import Tuple, Callable, Union
from .utils import (
    psf2otf, compute_spatial_gradient, tv_prox, tv_norm,
    gaussian_psf, gaussian_psf_deriv_alpha, project_param, EPSILON,
    estimate_lipschitz_alpha
)

# ----------------------------------------------------------------------
# Градиенты правдоподобия (f) для анизотропного случая
# ----------------------------------------------------------------------

def data_fidelity_grad_fft(y: np.ndarray, 
                           alpha: Union[float, Tuple[float, float]],
                           sigma2: float,
                           x: np.ndarray,
                           psf_func: Callable,
                           kernel_shape: Tuple[int, int],
                           boundary: str = 'periodic') -> np.ndarray:
    """Градиент по x: (1/σ²) H^T(α)(H(α)x - y)"""
    h = psf_func(alpha, kernel_shape)
    F_h = psf2otf(h, y.shape, boundary=boundary)
    F_x = fft2(x)
    conv = np.real(ifft2(F_h * F_x))
    residual = conv - y
    F_res = fft2(residual)
    grad = np.real(ifft2(np.conj(F_h) * F_res)) / sigma2
    return grad

def data_fidelity_alpha_grad(y: np.ndarray,
                             alpha: Union[float, Tuple[float, float]],
                             sigma2: float,
                             x: np.ndarray,
                             psf_func: Callable,
                             psf_deriv_func: Callable,
                             kernel_shape: Tuple[int, int],
                             boundary: str = 'periodic') -> Union[float, Tuple[float, float]]:
    """
    Градиент f по α.
    Возвращает скаляр или кортеж (d/dah, d/dav) для анизотропного случая.
    """
    h = psf_func(alpha, kernel_shape)
    deriv = psf_deriv_func(alpha, kernel_shape)
    
    F_h = psf2otf(h, y.shape, boundary=boundary)
    F_x = fft2(x)
    conv_hx = np.real(ifft2(F_h * F_x))
    residual = conv_hx - y
    
    if isinstance(alpha, (int, float)):
        dh_dalpha = deriv
        F_dh = psf2otf(dh_dalpha, y.shape, boundary=boundary)
        conv_dhx = np.real(ifft2(F_dh * F_x))
        grad_alpha = np.sum(residual * conv_dhx) / sigma2
        return grad_alpha
    else:
        dh_dah, dh_dav = deriv
        F_dh_ah = psf2otf(dh_dah, y.shape, boundary=boundary)
        F_dh_av = psf2otf(dh_dav, y.shape, boundary=boundary)
        conv_dahx = np.real(ifft2(F_dh_ah * F_x))
        conv_davx = np.real(ifft2(F_dh_av * F_x))
        grad_ah = np.sum(residual * conv_dahx) / sigma2
        grad_av = np.sum(residual * conv_davx) / sigma2
        return grad_ah, grad_av

def data_fidelity_sigma2_grad(y: np.ndarray,
                              alpha: Union[float, Tuple[float, float]],
                              sigma2: float,
                              x: np.ndarray,
                              psf_func: Callable,
                              kernel_shape: Tuple[int, int],
                              boundary: str = 'periodic') -> float:
    """∇_{σ²} f = - 1/(2σ⁴) ||y - H(α)x||²"""
    h = psf_func(alpha, kernel_shape)
    F_h = psf2otf(h, y.shape, boundary=boundary)
    F_x = fft2(x)
    conv = np.real(ifft2(F_h * F_x))
    residual = conv - y
    return -0.5 * np.sum(residual**2) / (sigma2**2)

# ----------------------------------------------------------------------
# MYULA sampler – общая версия
# ----------------------------------------------------------------------

def myula_sampler(
    x_init: np.ndarray,
    gamma: float,
    lam: float,
    m: int,
    burn_in: int,
    grad_log_potential: Callable,   # функция, возвращающая ∇_x (-log p(x))
    prox: Callable,                 # проксимальный оператор
    args_grad: Tuple = (),
    args_prox: Tuple = (),
    clip: bool = False,
    clip_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Универсальный MYULA sampler.
    grad_log_potential(x, *args_grad) -> градиент логарифма целевой плотности (без знака минус!).
    prox(x, *args_prox) -> проксимальный оператор для сглаженной части.
    """
    x = x_init.copy()
    samples = []
    for k in range(m + burn_in):
        grad = grad_log_potential(x, *args_grad)
        prox_term = prox(x, *args_prox)
        x = (1.0 - gamma / lam) * x \
            - gamma * grad \
            + (gamma / lam) * prox_term \
            + np.sqrt(2.0 * gamma) * np.random.randn(*x.shape)
        if clip:
            x = np.clip(x, *clip_range)
        if k >= burn_in:
            samples.append(x.copy())
    return np.array(samples)

# ----------------------------------------------------------------------
# Специализированные обёртки для posterior и prior
# ----------------------------------------------------------------------

def posterior_grad_factory(y, alpha, sigma2, psf_func, kernel_shape, boundary):
    """Фабрика: возвращает функцию ∇_x [-log p(x|y,θ,α,σ²)] без учёта θ-части."""
    def grad(x):
        return data_fidelity_grad_fft(y, alpha, sigma2, x, psf_func, kernel_shape, boundary)
    return grad

def prior_grad_factory():
    """Для prior: градиент отсутствует (нулевой)."""
    def grad(x):
        return np.zeros_like(x)
    return grad

def prox_factory(theta, prox_g):
    """Фабрика проксимального оператора для сглаженного prior."""
    def prox(x, lam):
        return prox_g(x, lam * theta)
    return prox

# ----------------------------------------------------------------------
# SAPG updates (поддерживают анизотропный alpha)
# ----------------------------------------------------------------------

def sapg_update_theta_homogeneous(
    samples_post: np.ndarray,
    theta: float,
    delta: float,
    d: int,
    q: float,
    bounds: Tuple[float, float]
) -> float:
    """θ update для однородного prior (q-однородного)."""
    avg_g_post = np.mean([tv_norm(s) for s in samples_post])
    grad_theta = d / (q * theta) - avg_g_post
    return project_param(theta + delta * grad_theta, bounds)

def sapg_update_theta_inhomogeneous(
    samples_post: np.ndarray,
    samples_prior: np.ndarray,
    theta: float,
    delta: float,
    bounds: Tuple[float, float]
) -> float:
    """θ update для неоднородного prior."""
    avg_g_post = np.mean([tv_norm(s) for s in samples_post])
    avg_g_prior = np.mean([tv_norm(s) for s in samples_prior])
    grad_theta = avg_g_prior - avg_g_post
    return project_param(theta + delta * grad_theta, bounds)

def sapg_update_alpha(
    samples: np.ndarray,
    y: np.ndarray,
    alpha: Union[float, Tuple[float, float]],
    sigma2: float,
    delta: float,
    psf_func: Callable,
    psf_deriv_func: Callable,
    kernel_shape: Tuple[int, int],
    bounds: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]],
    boundary: str = 'periodic'
) -> Union[float, Tuple[float, float]]:
    """SAPG update для α (скаляр или вектор)."""
    m = len(samples)
    if isinstance(alpha, (int, float)):
        grads = [data_fidelity_alpha_grad(y, alpha, sigma2, s, psf_func, psf_deriv_func,
                                          kernel_shape, boundary) for s in samples]
        avg_grad = np.mean(grads)
        alpha_new = alpha - delta * avg_grad
        return project_param(alpha_new, bounds)
    else:
        # Анизотропный случай: bounds должен быть кортежем из двух пар
        bounds_ah, bounds_av = bounds
        grads_ah = []
        grads_av = []
        for s in samples:
            gah, gav = data_fidelity_alpha_grad(y, alpha, sigma2, s, psf_func, psf_deriv_func,
                                                kernel_shape, boundary)
            grads_ah.append(gah)
            grads_av.append(gav)
        avg_gah = np.mean(grads_ah)
        avg_gav = np.mean(grads_av)
        ah_new, av_new = alpha[0] - delta * avg_gah, alpha[1] - delta * avg_gav
        ah_new = project_param(ah_new, bounds_ah)
        av_new = project_param(av_new, bounds_av)
        return (ah_new, av_new)

def sapg_update_sigma2(
    samples: np.ndarray,
    y: np.ndarray,
    alpha: Union[float, Tuple[float, float]],
    sigma2: float,
    delta: float,
    d: int,
    psf_func: Callable,
    kernel_shape: Tuple[int, int],
    bounds: Tuple[float, float],
    boundary: str = 'periodic'
) -> float:
    """SAPG update для σ²."""
    m = len(samples)
    avg_grad_sigma2 = np.mean([
        data_fidelity_sigma2_grad(y, alpha, sigma2, s, psf_func, kernel_shape, boundary)
        for s in samples
    ]) + d / (2.0 * sigma2)
    sigma2_new = sigma2 - delta * avg_grad_sigma2
    return project_param(sigma2_new, bounds)

# ----------------------------------------------------------------------
# Final MAP via HQS (поддержка анизотропного ядра)
# ----------------------------------------------------------------------

def solve_image_hqs(
    y: np.ndarray,
    h: np.ndarray,
    x_init: np.ndarray,
    sigma: float,
    theta: float,
    max_iter: int = 50,
    beta0: float = 1.0,
    beta_factor: float = 1.1,
    prox_tv_iter: int = 20,
    boundary: str = 'periodic'
) -> np.ndarray:
    """Half-Quadratic Splitting for TV deconvolution."""
    H, W = y.shape
    F_h = psf2otf(h, (H, W), boundary=boundary)
    F_h_conj = np.conj(F_h)
    F_h_sq = np.abs(F_h)**2

    x = x_init.copy()
    z = x.copy()
    beta = beta0

    for _ in range(max_iter):
        F_y = fft2(y)
        F_z = fft2(z)
        numerator = F_h_conj * F_y + beta * sigma**2 * F_z
        denominator = F_h_sq + beta * sigma**2
        x = np.real(ifft2(numerator / (denominator + EPSILON)))

        z = tv_prox(x, theta / beta, max_iter=prox_tv_iter)

        beta *= beta_factor

    return np.clip(x, 0.0, 1.0)