from typing import Union, Iterable, Optional, Tuple
from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq, irfft2
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum
import numpy as np

"""
    над кодом работал:
    Юров П.И.
"""

def powerlaw_psd_gaussian(
        exponent: float, 
        size: Union[int, Iterable[int]], 
        fmin: float = 0.0, 
        random_state: Optional[Union[int, Generator, RandomState]] = None
    ) -> np.ndarray:
    """
        над кодом работал:
        Юров П.И.
    """
    """
    Гауссов (1/f)**beta шум.

    Аргументы:

    exponent (float): Экспонента шума
        S(f) = (1 / f)**beta
        розовый шум:    exponent beta = 1
        коричневый шум: exponent beta = 2
    shape (int or iterable): Размер выходной марицы
    fmin (Optional[float]): Порог минимальной частоты от 0.0 до 0.5
    random_state (Optional[Union[int, Generator, RandomState]]): \
        Генератор случайных чисел

    Возвращает:
        матрицу шума
    """
    
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")
    
    samples = size[-1]
    
    f = rfftfreq(samples)
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")
    
    s_scale = f    
    ix   = npsum(s_scale < fmin)
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)
    
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.
    sigma = 2 * sqrt(npsum(w**2)) / samples
    
    size[-1] = len(f)

    dims_to_add = len(size) - 1
    s_scale     = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]
    
    normal_dist = _get_normal_distribution(random_state)

    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)
    
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)
    
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)
    
    s  = sr + 1J * si

    y = irfft(s, n=samples, axis=-1) / sigma
    
    return y


def _get_normal_distribution(random_state: Optional[Union[int, Generator, RandomState]]):
    """
        над кодом работал:
        Юров П.И.
    """
    """Возвращает генератор случайных чисел с нормальным распределением"""
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, "
            "numpy.random.Randomstate"
        )
    return normal_dist


def pink_noise_2d(shape: Tuple[int, int], 
                  alpha: float = 1.0):
    """
        над кодом работал:
        Юров П.И.
    """
    """
    Генерирует 2D 1/f^a шум.
    
    Аргументы:
        shape (Tuple[int, int]): Размер выходного массива шума
        alpha (float): Экспонента 'a' 1/f^a шума
    
    Возвращает:
        Матрица шума
    """
    height, width = shape
    
    white_noise = np.random.randn(height, width)

    white_fft = np.fft.fft2(white_noise)

    freqs_y = np.fft.fftfreq(height)
    freqs_x = np.fft.fftfreq(width)
    freq_mesh = np.meshgrid(freqs_y, freqs_x, indexing='ij')

    radial_freq = np.sqrt(freq_mesh[0]**2 + freq_mesh[1]**2)

    radial_freq_shifted = np.fft.fftshift(radial_freq)
    white_fft_shifted = np.fft.fftshift(white_fft)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.where(radial_freq_shifted == 0, 1.0, radial_freq_shifted**(-alpha/2))
        pink_fft_shifted = white_fft_shifted * scaling
    
    pink_fft = np.fft.ifftshift(pink_fft_shifted)
    pink_noise = np.real(np.fft.ifft2(pink_fft))
    
    pink_noise = pink_noise / np.std(pink_noise)*255.0
    
    return pink_noise