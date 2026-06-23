"""
utils.py

Вспомогательные функции для алгоритма слепой деконволюции с априорными 
распределениями с тяжелыми хвостами (HTP).

Содержит:
    - simpnormimg, denormimg: Нормализация и денормализация интенсивностей.
    - get_roi: Выделение центральной области интереса для оценки ядра.
    - mat2gray: Линейное контрастное масштабирование к диапазону [0, 1].
    - bwmorph_clean: Удаление изолированных пикселей маски.
    - center_psf: Пороговое отсечение и точное центрирование функции рассеяния точки.
    - calculate_mse: Вычисление инвариантной к сдвигу среднеквадратичной ошибки (MSE).
    - fft2_pad: Быстрое преобразование Фурье с дополнением нулями.
    - setup_lp_prior: Построение функции сжатия (shrinkage) для регуляризации.
    - imresize: Изменение размера изображений с предварительным антиалиасингом.
    - edgetaper: Сглаживание границ для устранения краевых эффектов (ringing) при БПФ.

Литература:
[1] J. Kotera, F. Sroubek, P. Milanfar,
    "Blind Deconvolution Using Alternating Maximum a Posteriori Estimation
     with Heavy-tailed Priors", CAIP 2013.
"""

from __future__ import annotations

from typing import Tuple, Callable

import numpy as np
from scipy.optimize import brentq
from scipy.ndimage import label


# --- Нормализация изображения ---
def simpnormimg(G: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Нормализует входное изображение так, чтобы значения интенсивностей 
    лежали в диапазоне [0, 1].

    Возвращает
    -------
    I : Нормализованное изображение типа float64
    m : Минимальное значение (смещение)
    v : Размах (масштаб)
    """
    G = np.asarray(G, dtype=np.float64)
    lb = float(G.min())
    ub = float(G.max())
    v = ub - lb
    if v == 0.0:
        v = 1.0
    I = (G - lb) / v
    return I, lb, v


def denormimg(U: np.ndarray, m: float, v: float) -> np.ndarray:
    """Обратное преобразование нормализации: U * v + m."""
    return U * v + m


# --- Выделение области интереса (ROI) ---
def get_roi(G: np.ndarray, win: Tuple[int, int]) -> np.ndarray:
    """
    Выделяет центральное окно заданного размера из изображения G.

    Для RGB изображений используется зеленый канал (индекс 1). Для полутоновых
    изображений возвращается единственный канал. Если исходное изображение меньше
    указанного размера окна, окно усекается по размеру изображения.
    """
    G = np.asarray(G)
    if G.ndim == 3 and G.shape[2] > 1:
        ch = G[..., 1]
    elif G.ndim == 3:
        ch = G[..., 0]
    else:
        ch = G

    isize = ch.shape
    win = (min(win[0], isize[0]), min(win[1], isize[1]))
    margin = ((isize[0] - win[0]) // 2, (isize[1] - win[1]) // 2)
    return ch[margin[0]:margin[0] + win[0],
              margin[1]:margin[1] + win[1]].copy()


# --- Линейное контрастное масштабирование ---
def mat2gray(A: np.ndarray) -> np.ndarray:
    """Масштабирование интенсивностей массива к [0, 1] по минимуму и максимуму."""
    A = np.asarray(A, dtype=np.float64)
    lo = float(A.min())
    hi = float(A.max())
    if hi == lo:
        return np.zeros_like(A)
    return (A - lo) / (hi - lo)


# --- Морфологическая очистка ---
def bwmorph_clean(BW: np.ndarray) -> np.ndarray:
    """
    Удаление изолированных пикселей (без 8-связных соседей) из бинарной маски.
    """
    BW = np.asarray(BW, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    labeled, _ = label(BW, structure=structure)
    if labeled.max() == 0:
        return BW.copy()
    counts = np.bincount(labeled.ravel())
    isolated = counts == 1
    isolated[0] = False
    out = BW.copy()
    out[isolated[labeled]] = False
    return out


# --- Центрирование ФРТ ---
def center_psf(H: np.ndarray, thresh: float) -> np.ndarray:
    """
    Пороговое отсечение и центрирование функции рассеяния точки в пределах 
    ее опорного окна.

    Шаги:
      1. Выделение положительной части H и применение порога thresh относительно 
         максимума.
      2. Удаление изолированных пикселей (опционально, если не удаляет все).
      3. Определение ограничивающей рамки (bounding box) для центрирования 
         внутри исходного окна (устраняет глобальное смещение).
      4. Интегральное центрирование масс: дополнительный сдвиг на целое число 
         пикселей для выравнивания центра масс сложного/асимметричного ядра 
         с геометрическим центром массива.
    """
    H = np.asarray(H, dtype=np.float64)
    hsize = np.array(H.shape[:2])

    h = np.maximum(H, 0.0)
    m_max = h.max()
    if m_max <= 0:
        s = h.sum()
        return h / s if s != 0 else h
        
    m = h >= (thresh * m_max)
    m2 = bwmorph_clean(m)
    if m2.any():
        m = m2 

    if not m.any():
        s = h.sum()
        return h / s if s != 0 else h

    sum1 = m.any(axis=0)  
    sum2 = m.any(axis=1)  
    L = np.array([np.argmax(sum2), np.argmax(sum1)])                    
    R = np.array([len(sum2) - 1 - np.argmax(sum2[::-1]),
                  len(sum1) - 1 - np.argmax(sum1[::-1])])

    val = (L + R + 3 - hsize) / 2.0
    topleft = np.fix(val).astype(np.int64) - 1  

    src_r0 = max(int(topleft[0]), 0)
    src_c0 = max(int(topleft[1]), 0)
    src_r1 = min(int(topleft[0] + hsize[0]), int(hsize[0]))
    src_c1 = min(int(topleft[1] + hsize[1]), int(hsize[1]))

    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return h / h.sum() if h.sum() != 0 else h

    cropped = h[src_r0:src_r1, src_c0:src_c1]   

    pad_pre = np.maximum(-topleft, 0).astype(int)
    out = np.zeros_like(h)
    out[pad_pre[0]:pad_pre[0] + cropped.shape[0],
        pad_pre[1]:pad_pre[1] + cropped.shape[1]] = cropped

    s = out.sum()
    if s == 0:
        return out
    kh, kw = out.shape
    ys = np.arange(kh, dtype=np.float64)[:, None]
    xs = np.arange(kw, dtype=np.float64)[None, :]
    yc = (out * ys).sum() / s
    xc = (out * xs).sum() / s
    sy = int(round(kh // 2 - yc))
    sx = int(round(kw // 2 - xc))
    if sy != 0 or sx != 0:
        shifted = np.zeros_like(out)
        src_r0 = max(0, -sy);  src_r1 = min(kh, kh - sy)
        src_c0 = max(0, -sx);  src_c1 = min(kw, kw - sx)
        dst_r0 = max(0, sy);   dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, sx);   dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = out[src_r0:src_r1, src_c0:src_c1]
        out = shifted

    s = out.sum()
    if s != 0:
        out = out / s
    return out


# --- Вычисление MSE ---
def calculate_mse(h: np.ndarray, hs: np.ndarray) -> float:
    """
    Вычисляет инвариантную к сдвигу среднеквадратичную ошибку (MSE) между 
    оцененной ФРТ h и истинной ФРТ hs.
    """
    h = np.asarray(h, dtype=np.float64)
    hs = np.asarray(hs, dtype=np.float64)
    sh = np.array(h.shape)
    shs = np.array(hs.shape)

    sum_h = h.sum()
    if sum_h != 0:
        h = h / sum_h * hs.sum()

    n = sh - shs + 1  
    if np.any(n < 1):
        return float(np.sqrt(((h - hs) ** 2).sum()))

    n_total = int(np.prod(n))
    hs_col = hs.flatten(order='F')

    center_idx = int(np.ceil(n_total / 2)) - 1
    cj = center_idx // n[0]   
    ci = center_idx % n[0]

    window = h[ci:ci + shs[0], cj:cj + shs[1]].flatten(order='F')
    return float(np.sqrt(((window - hs_col) ** 2).sum()))


# --- БПФ с дополнением ---
def fft2_pad(X: np.ndarray, M: int, N: int) -> np.ndarray:
    """Быстрое преобразование Фурье (2D) с предварительным дополнением нулями до размера (M, N)."""
    return np.fft.fft2(X, s=(M, N))


# --- Построение оператора сжатия (shrinkage) ---
def setup_lp_prior(q: float, alpha: float, beta: float) -> Callable[
        [np.ndarray, np.ndarray], np.ndarray]:
    """
    Построение оператора полуквадратичного сжатия (shrinkage) для 
    регуляризации в соответствии с Уравнением 3.

    Априорное распределение моделируется как:
        phi(s) = alpha * |s|^q     при |s| > u_star
        phi(s) = (beta/2) * s^2    при |s| <= u_star
    """
    if q == 1.0:
        v_star = 0.0
        u_star = alpha / beta
    elif q == 0.0:
        v_star = np.sqrt(2.0 * alpha / beta)
        u_star = v_star
    else:
        ratio = alpha / beta
        f1 = lambda v: -v + ratio * (v ** (q - 1)) * (1.0 - q) * q
        leftmarker = brentq(f1, np.finfo(float).eps, 10.0)
        f2 = lambda v: -0.5 * v * v + ratio * (v ** q) * (1.0 - q)
        v_star = brentq(f2, leftmarker, 10.0)
        u_star = v_star + ratio * q * (v_star ** (q - 1))

    k = u_star - v_star

    def fh(DU: np.ndarray, normDU: np.ndarray) -> np.ndarray:
        V = np.zeros_like(DU)
        m = normDU > u_star
        nDp = normDU[m]
        V[m] = DU[m] * (nDp - k) / nDp
        return V

    return fh


# --- Функции интерполяции ---
def _kernel_cubic(x: np.ndarray) -> np.ndarray:
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx
    f = ((1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
         + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0)
           * ((absx > 1.0) & (absx <= 2.0)))
    return f


def _kernel_lanczos3(x: np.ndarray) -> np.ndarray:
    f = np.zeros_like(x, dtype=np.float64)
    m = np.abs(x) < 3.0
    xm = x[m]
    f[m] = np.sinc(xm) * np.sinc(xm / 3.0)
    return f


_KERNEL_WIDTHS = {
    'bicubic': 4.0,
    'cubic':   4.0,
    'lanczos3': 6.0,
}

_KERNEL_FUNCS = {
    'bicubic':  _kernel_cubic,
    'cubic':    _kernel_cubic,
    'lanczos3': _kernel_lanczos3,
}


def _contributions(in_length: int, out_length: int, scale: float,
                   kernel: Callable, kernel_width: float):
    x = np.arange(1, out_length + 1, dtype=np.float64)
    u = x / scale + 0.5 * (1.0 - 1.0 / scale)

    if scale < 1.0:
        kernel_width_eff = kernel_width / scale
        kernel_eff = lambda t: scale * kernel(scale * t)
    else:
        kernel_width_eff = kernel_width
        kernel_eff = kernel

    left = np.floor(u - kernel_width_eff / 2.0)
    P = int(np.ceil(kernel_width_eff)) + 2
    indices = left[:, None] + np.arange(P, dtype=np.float64)[None, :]
    weights = kernel_eff(u[:, None] - indices)
    
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1.0
    weights = weights / weights_sum

    indices = indices.astype(np.int64) - 1  
    period = 2 * in_length
    mirror = np.concatenate([np.arange(in_length),
                             np.arange(in_length - 1, -1, -1)])
    indices = mirror[np.mod(indices, period)]

    keep = np.any(weights != 0, axis=0)
    weights = weights[:, keep]
    indices = indices[:, keep]

    return weights, indices


def _resize_along_dim(A: np.ndarray, dim: int,
                      weights: np.ndarray, indices: np.ndarray) -> np.ndarray:
    A = np.moveaxis(A, dim, 0)
    gathered = A[indices, ...]                 
    w_shape = (weights.shape[0], weights.shape[1]) + (1,) * (gathered.ndim - 2)
    out = (gathered * weights.reshape(w_shape)).sum(axis=1)
    return np.moveaxis(out, 0, dim)


def imresize(A: np.ndarray, scale, method: str = 'bicubic') -> np.ndarray:
    """Двумерная интерполяция изображения с учетом эффекта антиалиасинга при уменьшении."""
    A = np.asarray(A, dtype=np.float64)
    if method not in _KERNEL_FUNCS:
        raise ValueError(f'Неподдерживаемый метод: {method}')

    in_h, in_w = A.shape[:2]

    if np.isscalar(scale):
        sy = sx = float(scale)
        out_h = max(1, int(np.round(in_h * sy)))
        out_w = max(1, int(np.round(in_w * sx)))
    else:
        scale = list(scale)
        if (len(scale) == 2 and isinstance(scale[0], (int, np.integer))
                and isinstance(scale[1], (int, np.integer))):
            out_h, out_w = int(scale[0]), int(scale[1])
            sy = out_h / in_h
            sx = out_w / in_w
        else:
            sy, sx = float(scale[0]), float(scale[1])
            out_h = max(1, int(np.round(in_h * sy)))
            out_w = max(1, int(np.round(in_w * sx)))

    kfn = _KERNEL_FUNCS[method]
    kw = _KERNEL_WIDTHS[method]

    order = (0, 1) if sy <= sx else (1, 0)
    sizes = {0: (in_h, out_h, sy), 1: (in_w, out_w, sx)}

    out = A
    for dim in order:
        in_len, out_len, s = sizes[dim]
        if in_len == out_len and s == 1.0:
            continue
        weights, indices = _contributions(in_len, out_len, s, kfn, kw)
        out = _resize_along_dim(out, dim, weights, indices)

    return out


# --- Сглаживание краев изображения ---
def edgetaper(I: np.ndarray, PSF: np.ndarray) -> np.ndarray:
    """
    Сглаживание границ изображения для устранения краевых эффектов 
    перед вычислением быстрого преобразования Фурье.
    """
    I = np.asarray(I, dtype=np.float64)
    PSF = np.asarray(PSF, dtype=np.float64)
    s = PSF.sum()
    if s != 0:
        PSF = PSF / s

    if I.ndim == 2:
        return _edgetaper_2d(I, PSF)
    out = np.empty_like(I)
    for c in range(I.shape[2]):
        out[..., c] = _edgetaper_2d(I[..., c], PSF)
    return out


def _edgetaper_alpha_1d(proj: np.ndarray, N: int) -> np.ndarray:
    if N <= 1:
        return np.zeros(max(N, 0), dtype=np.float64)
    L = N - 1
    F = np.fft.fft(proj, L)
    z = np.real(np.fft.ifft(np.abs(F) ** 2))
    z = np.concatenate([z, z[:1]])  
    zmax = z.max()
    if zmax <= 0:
        return np.ones(N, dtype=np.float64)
    return 1.0 - z / zmax


def _psf2otf_centered(PSF: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    ph, pw = PSF.shape
    H, W = shape
    padded = np.zeros((H, W), dtype=np.float64)
    padded[:ph, :pw] = PSF
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return np.fft.fft2(padded)


def _edgetaper_2d(I: np.ndarray, PSF: np.ndarray) -> np.ndarray:
    H, W = I.shape
    proj_y = PSF.sum(axis=1)             
    proj_x = PSF.sum(axis=0)             

    beta_y = _edgetaper_alpha_1d(proj_y, H)
    beta_x = _edgetaper_alpha_1d(proj_x, W)
    alpha = np.outer(beta_y, beta_x)     
    alpha = np.clip(alpha, 0.0, 1.0)

    OTF = _psf2otf_centered(PSF, (H, W))
    blurred = np.real(np.fft.ifft2(np.fft.fft2(I) * OTF))

    return alpha * I + (1.0 - alpha) * blurred