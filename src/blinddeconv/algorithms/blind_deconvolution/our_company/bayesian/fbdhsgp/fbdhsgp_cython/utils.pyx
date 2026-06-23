"""
utils.pyx

Вспомогательные функции для алгоритма слепой деконволюции FBDHSGP.

Содержит:
    - psf2otf, otf2psf: Преобразования функции рассеяния точки в оптическую передаточную функцию.
    - pad_replicate, pad_zeros: Функции для дополнения краев (паддинга) изображений.
    - fspecial_gaussian, init_kernel: Генерация гауссовского фильтра для начального ядра размытия.
    - imresize_bilinear: Билинейная интерполяция с фильтром антиалиасинга.
    - getindex: Вычисление индексов для неопределенных граничных условий (UBC).
    - shift_kernel_img_space: Пространственное центрирование ядра для предотвращения смещений.

Важные замечания по реализации:
    - Индексация массивов основана на нуле (0-based).
    - При вычислении сверток применяется истинная свертка (с переворотом ядра).
    - Изменение размера изображений (resize) выполняется билинейным методом 
      с фильтрацией от алиасинга для сохранения стабильности на грубых масштабах.

Литература:
[1] X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
    "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
    Digital Signal Processing, 2017.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Библиотека scikit-image указана в зависимостях и предоставляет функцию 
# билинейного изменения размера с антиалиасинговым низкочастотным фильтром,
# который необходим по умолчанию. Мы используем cv2 как запасной вариант, 
# если skimage отсутствует.
try:  # pragma: no cover
    from skimage.transform import resize as _sk_resize
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _HAS_SKIMAGE = False
import cv2


# --- Преобразования PSF <-> OTF ---

def psf2otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Вычисляет оптическую передаточную функцию (OTF) по функции рассеяния точки (PSF).

    Шаги:
        1. Ядро psf дополняется нулями до требуемого размера shape.
        2. Применяется циклический сдвиг на половину размера ядра вдоль каждой оси, 
           чтобы центр PSF оказался в индексе (0, 0).
        3. Возвращается результат fft2.
    """
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    in_h, in_w = psf.shape
    out_h, out_w = shape
    padded = np.zeros((out_h, out_w), dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def otf2psf(otf: np.ndarray, psf_size: Tuple[int, int]) -> np.ndarray:
    """
    Восстанавливает пространственную PSF из ее OTF.

    Шаги:
        1. Выполняется обратное БПФ (ifft2) с взятием вещественной части.
        2. Применяется циклический сдвиг на половину размера ядра вдоль каждой оси.
        3. Результат обрезается до верхнего левого блока размера psf_size.
    """
    full = np.real(np.fft.ifft2(otf))
    ph, pw = psf_size
    full = np.roll(full, ph // 2, axis=0)
    full = np.roll(full, pw // 2, axis=1)
    return full[:ph, :pw]


# --- Вспомогательные функции для заполнения границ ---

def pad_replicate(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Дополняет границы изображения путем дублирования крайних пикселей."""
    return np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")


def pad_zeros(x: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Дополняет границы изображения нулями."""
    return np.pad(x, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")


# --- Генерация гауссовского фильтра ---

def fspecial_gaussian(shape: Sequence[int], sigma: float) -> np.ndarray:
    """
    Двумерный гауссовский фильтр.

    Вычисляется сетка координат с центром в середине фильтра, применяется формула 
    Гаусса, значения ниже машинного эпсилон обнуляются, затем фильтр нормализуется 
    на единичную сумму.
    """
    hr, hc = int(shape[0]), int(shape[1])
    siz_r = (hr - 1) / 2.0
    siz_c = (hc - 1) / 2.0
    y, x = np.mgrid[-siz_r:siz_r + 1, -siz_c:siz_c + 1]
    arg = -(x ** 2 + y ** 2) / (2.0 * sigma ** 2)
    h = np.exp(arg)
    eps = np.finfo(np.float64).eps
    h[h < eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h = h / s
    return h


def init_kernel(minsize: Sequence[int], sigma: float) -> np.ndarray:
    """Создает начальное ядро размытия на самом грубом уровне пирамиды."""
    return fspecial_gaussian(minsize, sigma)


# --- Изменение размера изображения ---

def imresize_bilinear(img: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    """
    Билинейная интерполяция изображения.

    При значительном уменьшении масштаба в пирамиде критически важно применять
    предварительный низкочастотный фильтр (антиалиасинг). Без него высокочастотный 
    шум на грубых масштабах вызывает смещение оценки ядра и появление артефактов 
    в финальном результате. Функция использует skimage.transform.resize, так как 
    она корректно обрабатывает антиалиасинг. Если библиотека недоступна, 
    используется cv2.resize.
    """
    r, c = int(target_shape[0]), int(target_shape[1])
    src = np.ascontiguousarray(img, dtype=np.float64)

    if _HAS_SKIMAGE:
        # skimage автоматически решает, нужен ли антиалиасинг, исходя из масштаба.
        # Мы передаем anti_aliasing=None, поэтому он включается только при уменьшении.
        out = _sk_resize(
            src,
            (r, c),
            order=1,                  # билинейная
            mode="edge",              # обработка границ
            anti_aliasing=None,       # авто: True только при даунсэмплинге
            preserve_range=True,
        )
        return np.asarray(out, dtype=np.float64)

    # Запасной вариант (без антиалиасинга).
    out = cv2.resize(src, (c, r), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float64)


# --- Индексы для 4-плиточных неопределенных граничных условий (UBC) ---

def getindex(n1: int, n2: int, hks1: int, hks2: int):
    """
    Предварительное вычисление четырех наборов индексов областей (плиток), 
    используемых функцией x_admm_ubc_bi для применения операторов H и H^T 
    с неопределенными граничными условиями.

    Возвращает четыре списка. Каждая запись представляет собой кортеж
    (rows, cols) одномерных индексов numpy (с 0-индексацией), подходящих
    для использования с np.ix_::

        block = x[np.ix_(rows, cols)]
    """
    # --- index1 : Px (извлечение областей x с паддингом) ---
    rows_lr = np.r_[n1 - hks1:n1, 0:n1, 0:hks1]
    rows_top = np.r_[n1 - hks1:n1, 0:2 * hks1]
    rows_bot = np.r_[n1 - 2 * hks1:n1, 0:hks1]

    cols_left = np.r_[n2 - hks2:n2, 0:2 * hks2]
    cols_right = np.r_[n2 - 2 * hks2:n2, 0:hks2]
    cols_full = np.arange(n2)

    index1 = [
        (rows_lr,   cols_left),    # лево
        (rows_lr,   cols_right),   # право
        (rows_top,  cols_full),    # верх
        (rows_bot,  cols_full),    # низ
    ]

    # --- index2 : координаты для записи фрагментов Hx в матрицу ye ---
    index2 = [
        (np.arange(n1),                  np.arange(hks2)),
        (np.arange(n1),                  np.arange(n2 - hks2, n2)),
        (np.arange(hks1),                np.arange(hks2, n2 - hks2)),
        (np.arange(n1 - hks1, n1),       np.arange(hks2, n2 - hks2)),
    ]

    # --- index3 : Py (извлечение фрагментов ye для применения H^T) ---
    cols_left3 = np.r_[n2 - hks2:n2, 0:3 * hks2]
    cols_right3 = np.r_[n2 - 3 * hks2:n2, 0:hks2]
    rows_top3 = np.r_[n1 - hks1:n1, 0:3 * hks1]
    rows_bot3 = np.r_[n1 - 3 * hks1:n1, 0:hks1]
    cols_mid = np.arange(hks2, n2 - hks2)

    index3 = [
        (rows_lr,   cols_left3),   # лево
        (rows_lr,   cols_right3),  # право
        (rows_top3, cols_mid),     # верх
        (rows_bot3, cols_mid),     # низ
    ]

    # --- index4 : координаты для записи фрагментов H^T(ye) в матрицу hrye ---
    index4 = [
        (np.arange(n1),                  np.arange(2 * hks2)),
        (np.arange(n1),                  np.arange(n2 - 2 * hks2, n2)),
        (np.arange(2 * hks1),            np.arange(2 * hks2, n2 - 2 * hks2)),
        (np.arange(n1 - 2 * hks1, n1),   np.arange(2 * hks2, n2 - 2 * hks2)),
    ]

    return index1, index2, index3, index4


# --- Центрирование ядра в пространстве изображений ---

def shift_kernel_img_space(
    x: np.ndarray, k: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Центрирует ядро размытия k так, чтобы расстояние до границ его 
    ограничивающей рамки было минимальным, и сдвигает скрытое изображение x 
    на противоположную величину (поскольку ядра инвариантны к сдвигу: 
    сдвиг в k эквивалентен сдвигу в x).
    """
    from scipy.signal import convolve2d

    # Очистка ядра от малого шума перед поиском ограничивающей рамки
    tao = 0.03
    threshold = min(k.max() * tao, 0.002)
    k = k.copy()
    k[k < threshold] = 0.0

    nz = np.argwhere(k > 0)
    if nz.size == 0:
        return x.copy(), k

    y_top, x_left = nz.min(axis=0)
    y_bottom, x_right = nz.max(axis=0)

    ksy, ksx = k.shape
    # Индексация с нуля, поэтому отступ слева равен индексу колонки
    gap_left = x_left
    gap_right = ksx - 1 - x_right
    gap_top = y_top
    gap_bottom = ksy - 1 - y_bottom

    # Вычисление сдвига для центрирования с небольшой поправкой при равенстве
    s_l = k[:, x_left].sum()
    s_r = k[:, x_right].sum()
    ratio_x = s_l / s_r if s_r != 0 else 1.0
    bonus_x = 0.01 if ratio_x >= 1 else -0.01
    shift_x = int(np.round((gap_right - gap_left + bonus_x) / 2.0))

    s_t = k[y_top, :].sum()
    s_b = k[y_bottom, :].sum()
    ratio_y = s_t / s_b if s_b != 0 else 1.0
    bonus_y = 0.01 if ratio_y >= 1 else -0.01
    shift_y = int(np.round((gap_bottom - gap_top + bonus_y) / 2.0))

    # Реализация сдвига через свертку с дельта-фильтром
    hksy = ksy // 2
    hksx = ksx // 2
    shift_filter = np.zeros((ksy, ksx), dtype=np.float64)
    
    rr = hksy + shift_y
    cc = hksx + shift_x
    if 0 <= rr < ksy and 0 <= cc < ksx:
        shift_filter[rr, cc] = 1.0
    else:
        shift_filter[hksy, hksx] = 1.0

    k_shift = convolve2d(k, shift_filter, mode="same")

    # Сдвиг скрытого изображения через свертку с перевернутым фильтром
    x_padded = pad_replicate(x, hksy, hksx)
    flipped = shift_filter[::-1, ::-1]
    x_shift = convolve2d(x_padded, flipped, mode="valid")

    return x_shift, k_shift


def make_odd(n: int) -> int:
    """Возвращает n, если оно нечетное, иначе n + 1 (размеры ядра должны быть нечетными)."""
    return n if n % 2 == 1 else n + 1