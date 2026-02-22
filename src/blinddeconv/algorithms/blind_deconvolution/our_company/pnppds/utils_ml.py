import numpy as np
from scipy.ndimage import center_of_mass

def fft2(x):
    return np.fft.fft2(x, axes=(-2, -1))

def ifft2(x):
    return np.fft.ifft2(x, axes=(-2, -1))

def psf2otf(psf, shape):
    """
    Преобразует PSF в OTF с центрированием.
    """
    if psf.ndim == 2:
        psf_pad = np.zeros(shape)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        psf_pad = np.roll(psf_pad, -kh // 2, axis=0)
        psf_pad = np.roll(psf_pad, -kw // 2, axis=1)
        otf = fft2(psf_pad)
    else:
        psf_pad = np.zeros(shape)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        psf_pad = np.roll(psf_pad, -kh // 2, axis=0)
        psf_pad = np.roll(psf_pad, -kw // 2, axis=1)
        otf = fft2(psf_pad)
        otf = otf[np.newaxis, :, :]
    return otf

def get_blur_operator_ml(x, h):
    """
    Оператор для ML части (X-step).
    """
    H, W = x.shape[-2:]
    h_fft = psf2otf(h, (H, W))

    def phi(img):
        img_fft = fft2(img)
        return np.real(ifft2(img_fft * h_fft))

    def adj_phi(img):
        img_fft = fft2(img)
        return np.real(ifft2(img_fft * np.conj(h_fft)))

    return phi, adj_phi

def proj_l2_ball(x, alpha_n, gaussian_nl, sp_nl, x_0, r=1):
    """Проекция на L2 шар"""
    epsilon = np.sqrt(x.size * (1 - sp_nl)) * r * alpha_n * gaussian_nl
    diff = x - x_0
    norm = np.linalg.norm(diff)
    if norm > epsilon:
        return x_0 + epsilon * diff / norm
    return x

def proj_box(x):
    return np.clip(x, 0., 1.)

def align_kernel_and_image(k, x):
    """
    Критически важная функция: центрирует ядро и сдвигает изображение обратно.
    Предотвращает дрейф (сдвиг) картинки в угол.
    """
    kh, kw = k.shape
    
    # Центр масс
    cy, cx = center_of_mass(k)
    target_y, target_x = kh // 2, kw // 2
    
    # Смещение
    shift_y = int(round(target_y - cy))
    shift_x = int(round(target_x - cx))
    
    if shift_y == 0 and shift_x == 0:
        return k, x
        
    # Сдвиг ядра
    k_aligned = np.roll(k, shift_y, axis=0)
    k_aligned = np.roll(k_aligned, shift_x, axis=1)
    
    # Обратный сдвиг изображения
    if x.ndim == 3:
        x_aligned = np.roll(x, -shift_y, axis=1)
        x_aligned = np.roll(x_aligned, -shift_x, axis=2)
    else:
        x_aligned = np.roll(x, -shift_y, axis=0)
        x_aligned = np.roll(x_aligned, -shift_x, axis=1)
        
    return k_aligned, x_aligned