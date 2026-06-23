"""
utils.py

Вспомогательные функции алгоритма сверхразрешения SelfExSR.

Функциональность:
    - clamp               - Ограничение массива значений.
    - fspecial_gaussian   - Построение двумерного гауссовского фильтра.
    - im2col_sliding      - Извлечение перекрывающихся патчей (sliding window).
    - update_uvMap        - Обновление карт соответствия по плоским индексам.
    - uvMat_from_uvMap    - Извлечение данных из карт соответствия.
    - get_uvpix           - Вычисление валидных координат центров патчей.
    - scale_tform         - Вычисление масштаба из матрицы гомографии.
    - trans_tform         - Применение вектора трансляции (сдвига) к гомографии.
    - check_valid_pos     - Проверка выхода патчей за границы изображения.
    - prep_plane_prob_acc - Подготовка аккумулятивных вероятностей принадлежности плоскости.
    - draw_plane_id       - Случайный выбор индекса плоскости.
    - vgg_interp2         - Билинейная интерполяция значений в дробных координатах.
    - apply_affine_tform  - Применение аффинных возмущений к матрице гомографии.
    - draw_rand_sample    - Генерация случайных сдвигов и аффинных деформаций (PatchMatch).

Индексация выполняется строго с 0, порядок массивов и обхода — row-major.
Линейный индекс всегда `ind = row * W + col`.

Литература:
[1] J. Huang, A. Singh, and N. Ahuja, 
    "Single Image Super-Resolution from Transformed Self-Exemplars",
    CVPR 2015.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import map_coordinates


# --- Базовые утилиты ---
def clamp(v, lo, hi):
    """Поэлементное ограничение массива в заданных пределах (min/max)."""
    return np.clip(v, lo, hi)


def fspecial_gaussian(size, sigma):
    """
    Построение двумерного изотропного гауссовского фильтра заданного 
    размера с нормализованной суммой.
    """
    if isinstance(size, (list, tuple)):
        h, w = size
    else:
        h = w = int(size)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y = np.arange(h, dtype=np.float64) - cy
    x = np.arange(w, dtype=np.float64) - cx
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def im2col_sliding(img_2d, patch_size):
    """
    Извлечение всех перекрывающихся патчей (скользящее окно) из двумерного массива.
    Возвращает матрицу, где каждый столбец содержит вытянутый в вектор патч.
    Патчи нумеруются построчно (row-major).
    """
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size
        
    windows = sliding_window_view(img_2d, (ph, pw))
    n_rows, n_cols = windows.shape[0], windows.shape[1]
    cols = windows.reshape(n_rows * n_cols, ph * pw).T
    return cols.astype(np.float32)


# --- Работа с картами соответствия (PatchMatch) ---
def update_uvMap(map_arr, data, pix_ind):
    """
    Обновление карты (двух- или трехмерной) на основе одномерного линейного индекса.
    Массив map_arr модифицируется in-place (на месте).
    """
    shape = map_arr.shape
    if map_arr.ndim == 2:
        flat = map_arr.ravel()
        flat[pix_ind] = data if np.isscalar(data) else np.asarray(data).ravel()
    else:
        H, W, C = shape
        flat = map_arr.reshape(H * W, C)
        if np.isscalar(data):
            flat[pix_ind] = data
        else:
            d = np.asarray(data)
            if d.ndim == 1:
                flat[pix_ind] = d[:, None] if C > 1 else d
            else:
                flat[pix_ind] = d
    return map_arr


def uvMat_from_uvMap(uv_map, pix_ind):
    """
    Извлечение данных из карты соответствия по одномерному линейному индексу.
    """
    if uv_map.ndim == 2:
        return uv_map.ravel()[pix_ind]
    else:
        H, W, C = uv_map.shape
        flat = uv_map.reshape(H * W, C)
        return flat[pix_ind]


def get_uvpix(img_size, prad):
    """
    Формирование сетки координат центров всех валидных патчей в изображении.
    Краевые пиксели, для которых патч выходит за границы, исключаются (mask = False).
    """
    H, W = img_size
    mask = np.ones((H, W), dtype=bool)
    
    # Исключение краевой зоны толщиной в радиус патча
    mask[:prad, :] = False
    mask[H - prad:, :] = False
    mask[:, :prad] = False
    mask[:, W - prad:] = False

    rc = np.argwhere(mask) 
    rows = rc[:, 0]
    cols = rc[:, 1]

    sub = np.column_stack([cols, rows]).astype(np.float32)  
    ind = (rows * W + cols).astype(np.int64)

    return {
        'sub': sub,
        'ind': ind,
        'mask': mask,
        'numUvPix': len(ind),
    }


# --- Геометрические преобразования ---
def scale_tform(H):
    """
    Оценка масштабного коэффициента, применяемого гомографией, 
    путем анализа определителя ее аффинной части.
    """
    H = np.asarray(H, dtype=np.float32)
    if H.ndim == 1:
        H = H.reshape(1, -1)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    h4, h5, h6 = H[:, 3], H[:, 4], H[:, 5]
    h7, h8 = H[:, 6], H[:, 7]
    det = (h1 - h7 * h3) * (h5 - h8 * h6) - (h4 - h7 * h6) * (h2 - h8 * h3)
    return np.sqrt(np.abs(det))


def trans_tform(uv_tform, d):
    """
    Применение смещения d = (dx, dy) к матрице трансформации. 
    Изменяются только элементы, отвечающие за трансляцию.
    """
    out = uv_tform.copy()
    d = np.asarray(d, dtype=np.float32)

    if d.ndim == 1:
        dx, dy = d[0], d[1]
        out[:, 6] = uv_tform[:, 0] * dx + uv_tform[:, 3] * dy + uv_tform[:, 6]
        out[:, 7] = uv_tform[:, 1] * dx + uv_tform[:, 4] * dy + uv_tform[:, 7]
        out[:, 8] = uv_tform[:, 2] * dx + uv_tform[:, 5] * dy + uv_tform[:, 8]
    else:
        dx = d[:, 0]
        dy = d[:, 1]
        out[:, 6] = uv_tform[:, 0] * dx + uv_tform[:, 3] * dy + uv_tform[:, 6]
        out[:, 7] = uv_tform[:, 1] * dx + uv_tform[:, 4] * dy + uv_tform[:, 7]
        out[:, 8] = uv_tform[:, 2] * dx + uv_tform[:, 5] * dy + uv_tform[:, 8]

    h9 = out[:, 8] + 1e-10
    out = out / h9[:, None]
    return out


def check_valid_pos(pos, img_size, prad):
    """
    Проверка того, не выходят ли координаты патча за границы изображения 
    с учетом радиуса патча (prad).
    """
    H, W = img_size
    x = pos[:, 0]
    y = pos[:, 1]
    return (x >= prad) & (x <= W - 1 - prad) & (y >= prad) & (y <= H - 1 - prad)


# --- Вероятностные функции (для планарной модели) ---
def prep_plane_prob_acc(plane_prob, pix_ind):
    """
    Формирование массива кумулятивной (накопленной) вероятности принадлежности 
    каждого пикселя к одной из выделенных плоскостей изображения.
    """
    num_plane = plane_prob.shape[2]
    N = len(pix_ind)
    acc = np.zeros((N, num_plane + 1), dtype=np.float32)

    for i in range(num_plane):
        prob_i = plane_prob[:, :, i].ravel()[pix_ind]
        acc[:, i + 1] = prob_i
        if i > 0:
            acc[:, i + 1] += acc[:, i]

    return acc


def draw_plane_id(plane_prob_acc):
    """
    Случайный выбор идентификатора плоскости на основе кумулятивной вероятности.
    Используется метод обратного преобразования (inverse transform sampling).
    Возвращает 0-индексированный массив идентификаторов.
    """
    N = plane_prob_acc.shape[0]
    num_plane = plane_prob_acc.shape[1] - 1
    rand_sample = np.random.rand(N).astype(np.float32)
    plane_id = np.zeros(N, dtype=np.uint8)

    for p in range(num_plane):
        mask = (plane_prob_acc[:, p] < rand_sample) & (plane_prob_acc[:, p + 1] >= rand_sample)
        plane_id[mask] = p 

    return plane_id


# --- Интерполяция субпиксельных значений ---
def vgg_interp2(img, x_coords, y_coords):
    """
    Многоканальная билинейная интерполяция изображения по дробным координатам.
    Значения за пределами границ заполняются нулями.
    """
    x = np.squeeze(x_coords)  
    y = np.squeeze(y_coords)

    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]

    pNumPix, N = x.shape
    C = img.shape[2] if img.ndim == 3 else 1

    result = np.zeros((pNumPix, N, C), dtype=np.float32)

    for c in range(C):
        ch = img[:, :, c] if img.ndim == 3 else img
        coords = np.array([y.ravel(), x.ravel()], dtype=np.float64)
        interped = map_coordinates(ch, coords, order=1, mode='constant', cval=0.0)
        result[:, :, c] = interped.reshape(pNumPix, N).astype(np.float32)

    return result


# --- Случайный поиск PatchMatch ---
def apply_affine_tform(tform_a, tform_d):
    """
    Применение аффинной матрицы возмущения к текущей матрице трансформации 
    путем перемножения 2x2 компонент.
    """
    cand = np.zeros_like(tform_a)
    cand[:, 0] = tform_d[:, 0] * tform_a[:, 0] + tform_d[:, 2] * tform_a[:, 1]
    cand[:, 1] = tform_d[:, 1] * tform_a[:, 0] + tform_d[:, 3] * tform_a[:, 1]
    cand[:, 2] = tform_d[:, 0] * tform_a[:, 2] + tform_d[:, 2] * tform_a[:, 3]
    cand[:, 3] = tform_d[:, 1] * tform_a[:, 2] + tform_d[:, 3] * tform_a[:, 3]
    return cand


def draw_rand_sample(search_pos_rad, num_uv_pix, iter_, opt):
    """
    Генерация кандидатов случайного поиска для PatchMatch:
    формирование пространственного смещения и матрицы аффинного искажения (сдвиг + поворот + масштаб),
    амплитуда которых затухает с увеличением номера итерации.
    """
    N = num_uv_pix

    # Пространственное смещение (offset)
    src_pos_offset = (2.0 * search_pos_rad * (np.random.rand(N, 2) - 0.5)).astype(np.float32)

    # Аффинное искажение (scale, rotation, shear)
    scale = opt['scaleRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    scale = scale + 1.0  
    theta = opt['rotRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    sh_x = opt['shRadA'] * (np.random.rand(N, 1) - 0.5) / iter_
    sh_y = opt['shRadA'] * (np.random.rand(N, 1) - 0.5) / iter_

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    d = np.zeros((N, 4), dtype=np.float32)
    d[:, 0:1] = cos_t - sin_t * sh_y
    d[:, 1:2] = sin_t + cos_t * sh_y
    d[:, 2:3] = cos_t * sh_x - sin_t
    d[:, 3:4] = sin_t * sh_x + cos_t

    d *= scale
    return src_pos_offset, d