"""
act_denoise.py

Алгоритм адаптивного порогового шумоподавления в области курвлет-преобразования
(Adaptive Curvelet Thresholding, ACT) для подавления белого и цветного 
гауссовского шума.

Основано на:
    N. Eslahi, A. Aghagolzadeh:
    "Compressive Sensing Image Restoration Using Adaptive Curvelet
     Thresholding and Nonlocal Sparse Regularization",
    IEEE Trans. Image Process., vol. 25, no. 7, pp. 3126-3140, Jul. 2016.
    https://doi.org/10.1109/TIP.2016.2562563

Требования: curvelets (реализация UDCT на чистом Python, pip install curvelets).
Зависимости: numpy, scipy, curvelets.
"""

import numpy as np
from scipy.ndimage import convolve as _ndconvolve
from numpy.fft import ifft2, fftshift

__all__ = ['act_denoise']


def _choose_num_scales(H, W):
    """
    Определение оптимального количества масштабов (включая низкочастотный) 
    для преобразования UDCT.

    Используется эмпирическое правило: ceil(log2(min(H, W))) - 2.
    Значение ограничивается диапазоном от 2 до 4.
    """
    return max(2, min(4, int(np.ceil(np.log2(min(H, W)))) - 2))


def _udct_pad_multiple(num_scales):
    """
    Вычисление множителя, которому должны быть кратны пространственные 
    размеры изображения для точного восстановления после UDCT.

    Чтобы избежать значительных ошибок восстановления и артефактов на 
    границах изображения (особенно для неквадратных или нечетных размеров), 
    каждое измерение должно быть кратно 2^(num_scales-1).
    """
    return 1 << max(num_scales - 1, 0)


def _make_udct(H, W, num_scales=None):
    """
    Создание оператора UDCT. 
    
    Размеры (H, W) предварительно должны быть дополнены (padded) до значений, 
    кратных результату функции _udct_pad_multiple(num_scales).
    """

    from curvelets.numpy import UDCT

    if num_scales is None:
        num_scales = _choose_num_scales(H, W)

    return UDCT(shape=(H, W), num_scales=num_scales, transform_kind='real')


def _compute_curvelet_noise_rootpsd(fft_psd, udct_op):
    """
    Оценка среднеквадратичного отклонения (СКО) шума для каждой подполосы 
    курвлет-преобразования.

    Пространственное ядро окрашивания шума восстанавливается из корня 
    спектральной плотности мощности (FFT_PSD) через обратное БПФ. Затем это 
    ядро трансформируется в курвлет-область. СКО шума в каждой подполосе 
    вычисляется как среднеквадратичное значение (RMS) полученных коэффициентов.

    Параметры
    ---------
    fft_psd : ndarray (H, W)
        Спектральная плотность мощности шума (FFT-PSD) в стандартном порядке 
        FFT (постоянная составляющая в [0,0]). Для белого гауссовского шума 
        (AWGN) с дисперсией sigma^2: FFT_PSD = sigma^2 * H * W.
    udct_op : curvelets.numpy.UDCT
        Инициализированный оператор курвлет-преобразования.

    Возвращает
    ----------
    rootpsd : list[list[list[float]]]
        Вложенный список со значениями СКО шума для каждой подполосы
        (масштаб, направление, клин).
    """
    kernel_noise = fftshift(ifft2(np.sqrt(fft_psd.astype(np.complex128)))).real

    c_struct = udct_op.forward(kernel_noise)

    rootpsd = []
    for scale in c_struct:
        dirs = []
        for direction in scale:
            wedges = []
            for wedge in direction:
                rms = float(np.sqrt(np.mean(np.abs(wedge) ** 2)))
                wedges.append(rms)
            dirs.append(wedges)
        rootpsd.append(dirs)
    return rootpsd


def _ml_estimator(noisy_coeffs, noise_rootpsd, noise_type):
    """
    Оценка СКО чистого сигнала в отдельной курвлет-подполосе с использованием 
    метода максимального правдоподобия.

    Локальная дисперсия зашумленных коэффициентов оценивается путем усреднения 
    квадратов их модулей по пространственному окну (исключая центральный пиксель). 
    Дисперсия чистого сигнала вычисляется как разность между локальной дисперсией 
    и дисперсией шума. Итоговое СКО = sqrt(max(clean_var, 0)).

    Параметры
    ---------
    noisy_coeffs : ndarray (complex)
        Курвлет-коэффициенты текущей подполосы.
    noise_rootpsd : float
        СКО шума (sigma_n) в данной подполосе.
    noise_type : str
        Тип шума: 'white' или 'colored'. Определяет размер окна усреднения:
        7x7 для белого шума, 31x31 для цветного.

    Возвращает
    ----------
    clean_std : ndarray
        Пространственно-зависимая оценка СКО чистого сигнала (совпадает по 
        размеру с noisy_coeffs).
    """
    if noise_type == 'white':
        k = np.ones((7, 7), dtype=np.float64) / 48.0
        k[3, 3] = 0.0
    else:
        k = np.ones((31, 31), dtype=np.float64) / 960.0
        k[15, 15] = 0.0
    power = (np.abs(noisy_coeffs) ** 2).astype(np.float64)
    local_var = _ndconvolve(power, k, mode='wrap')

    clean_var = local_var - noise_rootpsd ** 2
    clean_var = np.maximum(clean_var, 0.0)

    return np.sqrt(clean_var)


def _apply_act(c_struct, rootpsd, threshold_setting, noise_type):
    """
    Применение порогового ограничения ACT ко всем курвлет-подполосам.

    Самый грубый масштаб (J=0, низкочастотная компонента) всегда пропускается 
    без изменений. Для остальных масштабов вычисляется адаптивный порог на 
    основе отношения дисперсии шума к оценке дисперсии чистого сигнала.

    Параметры
    ---------
    c_struct : list[list[list[ndarray]]]
        Курвлет-коэффициенты, полученные из UDCT.forward().
    rootpsd : list[list[list[float]]]
        СКО шума для каждой подполосы (из _compute_curvelet_noise_rootpsd).
    threshold_setting : str
        Режим ограничения:
        's' - мягкое ограничение (Soft ACT), порог: sqrt(2) * sigma_n^2 / sigma_clean.
        'h' - жесткое ограничение (Hard ACT).
        'ksigma' - классическое ограничение k-sigma (Starck, Candes, Donoho 2002).
    noise_type : str
        Тип шума ('white' или 'colored').

    Возвращает
    ----------
    denoised : list[list[list[ndarray]]]
        Структура курвлет-коэффициентов после пороговой обработки.
    """
    nscales = len(c_struct)
    denoised = []

    for J in range(nscales):
        dirs = []
        for D in range(len(c_struct[J])):
            wedges = []
            for W in range(len(c_struct[J][D])):
                coeff = c_struct[J][D][W].copy()
                if J == 0:
                    wedges.append(coeff)
                    continue

                sigma_n = rootpsd[J][D][W]
                mag = np.abs(coeff)

                if threshold_setting in ('s', 'h'):
                    clean_std = _ml_estimator(coeff, sigma_n, noise_type)
                    safe_std = np.maximum(clean_std, 1e-10)

                    if threshold_setting == 's':
                        threshold = np.sqrt(2.0) * (sigma_n ** 2) / safe_std
                        threshold = np.where(clean_std > 0, threshold, np.inf)
                        shrunk = np.maximum(mag - threshold, 0.0)
                        coeff = np.where(
                            mag > 1e-30,
                            coeff * (shrunk / mag),
                            np.zeros_like(coeff),
                        )

                    else:
                        is_finest = float(J == nscales - 1)
                        threshold = ((3.0 + is_finest) * (sigma_n ** 2)
                                     / (np.sqrt(2.0) * safe_std))
                        threshold = np.where(clean_std > 0, threshold, np.inf)

                        coeff = coeff * (mag > threshold)

                else:
                    is_finest = float(J == nscales - 1)
                    threshold = (3.0 + is_finest) * sigma_n

                    coeff = coeff * (mag > threshold)

                wedges.append(coeff)
            dirs.append(wedges)
        denoised.append(dirs)

    return denoised


def act_denoise(image, noise_var=None, threshold_setting='s'):
    """
    Подавление шума на полутоновом изображении с использованием алгоритма ACT 
    (Adaptive Curvelet Thresholding).

    Функция выполняет дополнение (padding) изображения до оптимальных размеров,
    прямое курвлет-преобразование, оценку уровня шума (при необходимости), 
    адаптивную пороговую обработку коэффициентов и обратное преобразование.

    Параметры
    ---------
    image : ndarray (H, W), float64 в диапазоне [0, 1]
        Входное зашумленное полутоновое изображение.
    noise_var : None, float или ndarray (H, W), по умолчанию None
        Оценка дисперсии шума:
        - None : слепая оценка по методу MAD на самом детальном масштабе курвлет-преобразования.
        - float : известная дисперсия белого гауссовского шума (sigma^2).
        - ndarray : спектральная плотность мощности (FFT-PSD) шума в стандартном 
          порядке FFT (постоянная составляющая в [0,0]). Формат масштабирования 
          для AWGN: FFT_PSD = sigma^2 * H * W.
    threshold_setting : str, по умолчанию 's'
        Стратегия порогового ограничения:
        - 's' : мягкое ограничение (обычно дает лучший PSNR).
        - 'h' : жесткое ограничение.
        - 'ksigma' : базовое пороговое ограничение (Starck/Candes/Donoho 2002).

    Возвращает
    ----------
    denoised : ndarray (H, W), float64
        Изображение после шумоподавления.
    info : dict
        Словарь с метаданными о процессе шумоподавления:
        - 'noise_type' : тип шума ('white' или 'colored').
        - 'noise_var' : эффективная дисперсия шума (float) или строка 'fft_psd'.
        - 'threshold_setting' : примененная стратегия.
        - 'blind' : bool, True, если дисперсия оценивалась автоматически.
    """
    if threshold_setting not in ('s', 'h', 'ksigma'):
        raise ValueError(
            f"threshold_setting='{threshold_setting}': "
            f"choose from 's', 'h', 'ksigma'")

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {img.shape}")
    H, W = img.shape
    num_scales = _choose_num_scales(H, W)
    pad_mult = _udct_pad_multiple(num_scales)
    Hp = H + (-H) % pad_mult
    Wp = W + (-W) % pad_mult
    pad_h = Hp - H
    pad_w = Wp - W
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    N = Hp * Wp

    udct_op = _make_udct(Hp, Wp, num_scales=num_scales)

    c_struct = udct_op.forward(img)

    blind = noise_var is None
    if blind:
        mads = []
        for direction in c_struct[-1]:
            for wedge in direction:
                vals = wedge.real.ravel() if np.iscomplexobj(wedge) \
                    else wedge.ravel()
                mads.append(
                    np.median(np.abs(vals - np.median(vals))) / 0.6745)
        noise_std = float(np.median(mads))
        noise_var = noise_std ** 2

    scalar_var = (np.isscalar(noise_var)
                  or (isinstance(noise_var, np.ndarray)
                      and noise_var.size == 1))
    if scalar_var:
        sigma2 = float(np.ravel(noise_var)[0]
                        if not np.isscalar(noise_var)
                        else noise_var)
        fft_psd = np.full((Hp, Wp), sigma2 * N, dtype=np.float64)
        noise_type = 'white'
    else:
        fft_psd = np.asarray(noise_var, dtype=np.float64)
        if fft_psd.shape == (H, W) and (pad_h or pad_w):
            fft_psd = np.pad(fft_psd, ((0, pad_h), (0, pad_w)), mode='wrap')
        if fft_psd.shape != (Hp, Wp):
            raise ValueError(
                f"FFT-PSD shape {fft_psd.shape} != image ({Hp}, {Wp})")
        psd_range = float(fft_psd.max() - fft_psd.min())
        noise_type = 'white' if psd_range < 0.015 * N else 'colored'

    rootpsd = _compute_curvelet_noise_rootpsd(fft_psd, udct_op)

    denoised_struct = _apply_act(
        c_struct, rootpsd, threshold_setting, noise_type)

    denoised = udct_op.backward(denoised_struct)
    if np.iscomplexobj(denoised):
        denoised = denoised.real

    denoised = denoised[:H, :W]

    info = {
        'noise_type': noise_type,
        'noise_var': sigma2 if scalar_var else 'fft_psd',
        'threshold_setting': threshold_setting,
        'blind': blind,
    }
    return denoised, info
