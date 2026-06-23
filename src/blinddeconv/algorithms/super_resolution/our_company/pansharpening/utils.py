"""
utils.py

Вспомогательные функции для алгоритма вариационного байесовского сверхразрешения (паншарпенинга).

Содержит:
    Инфраструктура полифазного преобразования:
        im_decomp, im_comp, blk_fft2, blk_ifft2, blk_fd_conv,
        blk_fd_trace, blk_fft2_DtD, blk_DtY

    Операторы в частотной области:
        cent_nucleus2fft, Tcent_nucleus2fft,
        cent_nucleus2blk_fft2, Tcent_nucleus2blk_fft2

    Операторы кругового (периодического) градиента:
        circ_gradient2, Tcirc_gradient2

    Преобразование многоканального изображения в вектор:
        convertToMBVec, convertToMBImg

    Функции вычисления весов априорных распределений:
        compute_Wpvb, getfilters, getkappa, weight_log, weight_lp

    Утилиты модели наблюдений:
        calclam, image_normalize, image_denormalize, get_psf
"""

import numpy as np
from numpy.fft import fft2, ifft2

_EPS = np.finfo(np.float64).eps


# --- Полифазная декомпозиция и композиция ---

def im_decomp(img, bknr, bknc):
    """
    Полифазная декомпозиция 2D изображения.
    Разделяет изображение [nr, nc] на [low_nr, low_nc, bknr*bknc, 1] полифазных
    компонент путем извлечения каждого bknr-го пикселя по строкам и bknc-го по столбцам.
    """
    nr, nc = img.shape[:2]
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))

    pad_r = bknr * low_nr - nr
    pad_c = bknc * low_nc - nc
    if pad_r > 0 or pad_c > 0:
        img = np.pad(img, ((0, pad_r), (0, pad_c)))

    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, 1))
    pos = 0
    for spr in range(bknr):
        for spc in range(bknc):
            out[:, :, pos, 0] = img[spr::bknr, spc::bknc]
            pos += 1
    return out


def im_comp(blk, bknr, bknc):
    """
    Обратная полифазная композиция.
    Восстанавливает изображение [nr, nc] из полифазного представления [low_nr, low_nc, bkn2, 1].
    """
    low_nr, low_nc = blk.shape[:2]
    nr = low_nr * bknr
    nc = low_nc * bknc
    out = np.zeros((nr, nc))
    pos = 0
    for spr in range(bknr):
        for spc in range(bknc):
            out[spr::bknr, spc::bknc] = np.real(blk[:, :, pos, 0])
            pos += 1
    return out


# --- Блочное Быстрое Преобразование Фурье ---

def blk_fft2(data):
    """Блочное двумерное БПФ по первым двум измерениям."""
    out = np.zeros_like(data, dtype=complex)
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            out[:, :, i, j] = fft2(data[:, :, i, j])
    return out


def blk_ifft2(data):
    """Обратное блочное двумерное БПФ (с возвратом вещественной части)."""
    out = np.zeros(data.shape)
    for i in range(data.shape[2]):
        for j in range(data.shape[3]):
            out[:, :, i, j] = np.real(ifft2(data[:, :, i, j]))
    return out


# --- Арифметика блочных матриц в частотной области ---

def blk_fd_conv(Op1, Op2):
    """
    Блочное умножение матриц (свертка) в частотной области.
    Выполняет поэлементное умножение по пространственно-частотным осям (0,1)
    с матричным произведением по блочным осям (2,3).
    """
    assert Op1.shape[3] == Op2.shape[2], "Внутренние размерности блоков не совпадают"
    assert Op1.shape[:2] == Op2.shape[:2], "Пространственные размерности не совпадают"

    dt = complex if (np.iscomplexobj(Op1) or np.iscomplexobj(Op2)) else float
    out = np.zeros((Op2.shape[0], Op2.shape[1], Op1.shape[2], Op2.shape[3]), dtype=dt)
    for i in range(out.shape[2]):
        for j in range(out.shape[3]):
            for k in range(Op1.shape[3]):
                out[:, :, i, j] += Op1[:, :, i, k] * Op2[:, :, k, j]
    return out


def blk_fd_trace(m):
    """
    Вычисление следа блочной матрицы в частотной области.
    Возвращает вещественный скаляр sum_{i,j} trace(m[i,j,:,:]).
    """
    tr = 0.0 + 0.0j
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            tr += np.trace(m[i, j, :, :])
    return float(np.real(tr))


def blk_fft2_DtD(nr, nc, bknr, bknc):
    """
    Оператор D^T D в блочном Фурье-представлении.
    Выделяет полифазную компоненту с индексом (0,0).
    """
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, bkn2))
    out[:, :, 0, 0] = 1.0
    return out


def blk_DtY(Y, bknr, bknc):
    """
    Оператор D^T Y: интерполяция путем вставки нулей.
    Размещает LR-наблюдение Y в полифазную позицию (0,0).
    """
    low_nr, low_nc = Y.shape[:2]
    bkn2 = bknr * bknc
    out = np.zeros((low_nr, low_nc, bkn2, 1))
    out[:, :, 0, 0] = Y
    return out


# --- Перевод ядер свертки в частотную область ---

def cent_nucleus2fft(spkernel, nr, nc):
    """
    Преобразование центрированного ядра свертки в оптическую передаточную функцию (ОТФ).
    Размещает центр ядра в координате (0, 0) с циклическим смещением.
    """
    spkernel = np.asarray(spkernel, dtype=np.float64)
    kh, kw = spkernel.shape

    if kh > nr or kw > nc:
        fac_orig = spkernel.sum()
        if kh > nr:
            c = (kh - nr) // 2
            spkernel = spkernel[c:c + nr, :]
        kh, kw = spkernel.shape
        if kw > nc:
            c = (kw - nc) // 2
            spkernel = spkernel[:, c:c + nc]
        kh, kw = spkernel.shape
        fac_new = spkernel.sum()
        if abs(fac_new) > _EPS:
            spkernel *= fac_orig / fac_new

    cr, cc = kh // 2, kw // 2
    h = np.zeros((nr, nc))
    for i in range(kh):
        for j in range(kw):
            h[(i - cr) % nr, (j - cc) % nc] = spkernel[i, j]
    return fft2(h)


def Tcent_nucleus2fft(spkernel, nr, nc):
    """
    Сопряженная (транспонированная) ОТФ ядра.
    Для вещественных ядер эквивалентно ОТФ пространственно отраженного ядра.
    """
    return np.conj(cent_nucleus2fft(spkernel, nr, nc))


def cent_nucleus2blk_fft2(spkernel, nr, nc, bknr, bknc):
    """Преобразование ядра свертки в полифазное блочно-циркулянтное Фурье-представление."""
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc

    H = np.zeros((low_nr, low_nc, bkn2, bkn2))

    cent_nuc = np.real(ifft2(cent_nucleus2fft(spkernel, nr, nc)))

    col_idx = 0
    inter = cent_nuc.copy()
    for jr in range(bknr):
        for jc in range(bknc):
            col = im_decomp(inter, bknr, bknc)
            H[:, :, :, col_idx] = col[:, :, :, 0]
            col_idx += 1
            inter = np.roll(inter, 1, axis=1)      
        cent_nuc = np.roll(cent_nuc, 1, axis=0)    
        inter = cent_nuc.copy()

    return blk_fft2(H)


def Tcent_nucleus2blk_fft2(spkernel, nr, nc, bknr, bknc):
    """Транспонированное (сопряженное) полифазное представление ядра."""
    low_nr = int(np.ceil(nr / bknr))
    low_nc = int(np.ceil(nc / bknc))
    bkn2 = bknr * bknc

    H = np.zeros((low_nr, low_nc, bkn2, bkn2))

    cent_nuc = np.real(ifft2(Tcent_nucleus2fft(spkernel, nr, nc)))

    col_idx = 0
    inter = cent_nuc.copy()
    for jr in range(bknr):
        for jc in range(bknc):
            col = im_decomp(inter, bknr, bknc)
            H[:, :, :, col_idx] = col[:, :, :, 0]
            col_idx += 1
            inter = np.roll(inter, 1, axis=1)
        cent_nuc = np.roll(cent_nuc, 1, axis=0)
        inter = cent_nuc.copy()

    return blk_fft2(H)


# --- Операторы кругового градиента ---

def circ_gradient2(f):
    """
    Круговой (периодический) градиент на основе обратных разностей.
    Возвращает горизонтальные и вертикальные разности с учетом краев.
    """
    dfh = np.concatenate([f[:, :-1] - f[:, 1:],
                          (f[:, -1] - f[:, 0])[:, np.newaxis]], axis=1)
    dfv = np.concatenate([f[:-1, :] - f[1:, :],
                          (f[-1, :] - f[0, :])[np.newaxis, :]], axis=0)
    return dfh, dfv


def Tcirc_gradient2(f):
    """Сопряженный (транспонированный) оператор кругового градиента."""
    dfhT = np.concatenate([(f[:, 0] - f[:, -1])[:, np.newaxis],
                           f[:, 1:] - f[:, :-1]], axis=1)
    dfvT = np.concatenate([(f[0, :] - f[-1, :])[np.newaxis, :],
                           f[1:, :] - f[:-1, :]], axis=0)
    return dfhT, dfvT


# --- Трансформация массивов (векторизация) ---

def convertToMBVec(x):
    """Преобразование многоканального изображения в одномерный вектор."""
    if x.ndim == 2:
        return x.flatten()
    N, M, nb = x.shape
    parts = [x[:, :, i].flatten() for i in range(nb)]
    return np.concatenate(parts)


def convertToMBImg(x, Ng, Mg, nb):
    """Восстановление многоканального изображения из одномерного вектора."""
    pix = Ng * Mg
    y = np.zeros((Ng, Mg, nb))
    for i in range(nb):
        y[:, :, i] = x[i * pix:(i + 1) * pix].reshape(Ng, Mg)
    return y


# --- Функции весов априорных распределений ---

def compute_Wpvb(u, p, epsW):
    """Вычисление весов для TV-регуляризации при минимизации функции мажоранты (IRLS)."""
    exponent = (p - 1.0) / p
    return 2.0 / (p * (epsW + u ** exponent))


def weight_log(u):
    """Весовая функция kappa(u) для супергауссовского (log) априорного распределения."""
    val = _EPS + (np.abs(u) + _EPS) * np.abs(u)
    return 1.0 / val


def weight_lp(u, p):
    """Весовая функция kappa(u) для супергауссовского (lp) априорного распределения."""
    val = _EPS + np.abs(u) ** (2 - p)
    return 1.0 / val


def getfilters(filtersetname):
    """Инициализация набора фильтров (конечных разностей) для SG-регуляризации."""
    sq2 = np.sqrt(2.0)
    if filtersetname == 'fohv':
        return [
            np.array([[0.0, 1.0, -1.0]]) / sq2,
            np.array([[0.0], [1.0], [-1.0]]) / sq2,
        ]
    elif filtersetname == 'fo':
        return [
            np.array([[0.0, 1.0, -1.0]]) / sq2,
            np.array([[0.0], [1.0], [-1.0]]) / sq2,
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=float) / sq2,
            np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]], dtype=float) / sq2,
        ]
    else:   
        return [np.array([[1.0]])]


def getkappa(sg_prior_name, parameter=None):
    """Возвращает тройку функций-обработчиков (kappa_f, rho_f, alpha_f) для SG-регуляризации."""
    if sg_prior_name == 'log':
        kappa_f = weight_log
        rho_f = lambda nu: np.log(np.abs(nu) + _EPS)
        alpha_f = lambda val: 1.0 + 1.0 / val
    elif sg_prior_name == 'lp':
        if parameter is None:
            raise ValueError("Для априорного распределения 'lp' требуется параметр p")
        p = parameter
        kappa_f = lambda u, _p=p: weight_lp(u, _p)
        rho_f = lambda nu, _p=p: np.abs(nu) ** _p
        alpha_f = lambda val, _p=p: 1.0 / (_EPS + _p * val)
    else:
        raise ValueError(f"Неизвестный тип SG-распределения: {sg_prior_name}")

    return [kappa_f, rho_f, alpha_f]


# --- Модель наблюдений ---

def calclam(hires, pan):
    """
    Оценка коэффициентов масштабирования (lambda) связи между HR и PAN изображениями 
    с помощью метода наименьших квадратов.
    """
    if hires.ndim == 2:
        return np.array([1.0])

    nbands = hires.shape[2]
    if nbands == 1:
        return np.array([1.0])

    M = np.zeros((nbands, nbands))
    r = np.zeros(nbands)

    for ib in range(nbands):
        for jb in range(nbands):
            M[ib, jb] = np.sum(hires[:, :, ib] * hires[:, :, jb])
        r[ib] = np.sum(hires[:, :, ib] * pan)

    lamb = np.linalg.solve(M, r)
    lamb[lamb <= _EPS] = 0.0
    pos = lamb > _EPS
    if np.any(pos):
        lamb[pos] /= np.sum(lamb[pos])
    return lamb


def image_normalize(Y, x):
    """Поканальная нормализация среднего значения для мультиспектрального и PAN изображений."""
    Y = np.asarray(Y, dtype=np.float64).copy()
    x = np.asarray(x, dtype=np.float64).copy()

    if Y.ndim == 2:
        Y = Y[:, :, np.newaxis]

    nbands = Y.shape[2]
    lr_h, lr_w = Y.shape[:2]
    hr_h, hr_w = x.shape[:2]

    facY = np.zeros(nbands)
    for i in range(nbands):
        facY[i] = np.sum(Y[:, :, i]) / (lr_h * lr_w)
        if facY[i] > _EPS:
            Y[:, :, i] /= facY[i]

    facx = np.sum(x) / (hr_h * hr_w)
    if facx > _EPS:
        x /= facx

    return Y, x, facY, facx


def image_denormalize(Y, facY):
    """Обратная поканальная нормализация среднего значения (возврат к исходному масштабу)."""
    Y = np.asarray(Y, dtype=np.float64).copy()
    Y[Y < _EPS] = 0.0

    if Y.ndim == 2:
        Y *= facY[0]
    else:
        for i in range(Y.shape[2]):
            Y[:, :, i] *= facY[i]
    return Y


def get_psf(ratio, sensor='none'):
    """
    Генерация функции рассеяния точки (ФРТ) для заданного коэффициента масштабирования.
    Принимает строку с названием сенсора ('QB', 'IKONOS', 'GeoEye1', 'WV2' или 'none').
    """
    known_sensors = {
        'QB':      [0.34, 0.32, 0.30, 0.22],
        'IKONOS':  [0.26, 0.28, 0.29, 0.28],
        'GeoEye1': [0.23, 0.23, 0.23, 0.23],
        'WV2':     [0.35] * 7 + [0.27],
    }

    if sensor in known_sensors:
        GNyq = known_sensors[sensor]
        N = 41
        fcut = 1.0 / ratio
        psfs = []
        for g in GNyq:
            alpha = np.sqrt(((N - 1) * (fcut / 2)) ** 2 / (-2 * np.log(g)))
            H = _fspecial_gaussian(N, alpha)
            H /= H.sum()
            psfs.append(H)
        return psfs

    psf = np.ones((ratio, ratio), dtype=np.float64) / (ratio * ratio)
    return psf


def _fspecial_gaussian(size, sigma):
    """Построение изотропного двумерного фильтра Гаусса."""
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    h = np.outer(g, g)
    h /= h.sum()
    return h