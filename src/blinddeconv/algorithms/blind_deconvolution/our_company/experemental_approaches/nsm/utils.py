import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve

def conv2_valid(A, B):

    return fftconvolve(A, B, mode='valid')

def conv2_same(A, B):

    return fftconvolve(A, B, mode='same')

def conv2_full(A, B):

    return fftconvolve(A, B, mode='full')

def psf2otf(psf, shape):

    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)

    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf

    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)

def compute_constants(f, k, dx, dy):

    shape = f.shape[:2]
    otfk = psf2otf(k, shape)

    Ktf = np.conj(otfk) * fft2(f)
    KtK = np.abs(otfk) ** 2

    otf_dx = psf2otf(dx, shape)
    otf_dy = psf2otf(dy, shape)
    DtD = np.abs(otf_dx) ** 2 + np.abs(otf_dy) ** 2

    return Ktf, KtK, DtD

def init_kernel(minsize):

    k = np.zeros((minsize, minsize), dtype=np.float64)

    c = (minsize - 1) // 2
    k[c - 1, c - 1] = 0.5
    k[c - 1, c] = 0.5
    return k

def center_kernel_separate(x, y, k):

    rows = np.arange(1, k.shape[0] + 1, dtype=np.float64)
    cols = np.arange(1, k.shape[1] + 1, dtype=np.float64)

    mu_y = np.sum(rows * k.sum(axis=1))
    mu_x = np.sum(cols * k.sum(axis=0))

    offset_x = int(round(np.floor(k.shape[1] / 2.0) + 1 - mu_x))
    offset_y = int(round(np.floor(k.shape[0] / 2.0) + 1 - mu_y))

    sh_rows = abs(offset_y) * 2 + 1
    sh_cols = abs(offset_x) * 2 + 1
    shift_kernel = np.zeros((sh_rows, sh_cols), dtype=np.float64)

    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    k_shifted = conv2_same(k, shift_kernel)

    flipped_sk = shift_kernel[::-1, ::-1]
    x_shifted = conv2_same(x, flipped_sk)
    y_shifted = conv2_same(y, flipped_sk)

    return x_shifted, y_shifted, k_shifted

def edgetaper(img, psf):

    sn, sm = psf.shape
    n, m = img.shape

    proj_y = psf.sum(axis=1)
    proj_x = psf.sum(axis=0)

    z_y = np.correlate(proj_y, proj_y, mode='full')
    z_x = np.correlate(proj_x, proj_x, mode='full')

    z_y = z_y / z_y.max()
    z_x = z_x / z_x.max()

    w_y = np.zeros(n, dtype=np.float64)
    if len(z_y) <= n:
        w_y[:len(z_y)] = z_y
    else:
        w_y[:] = z_y[sn - 1 : sn - 1 + n]
    w_y = np.roll(w_y, -(sn - 1))
    w_y = np.maximum(w_y, 0)

    w_x = np.zeros(m, dtype=np.float64)
    if len(z_x) <= m:
        w_x[:len(z_x)] = z_x
    else:
        w_x[:] = z_x[sm - 1 : sm - 1 + m]
    w_x = np.roll(w_x, -(sm - 1))
    w_x = np.maximum(w_x, 0)

    beta = 1.0 - np.outer(w_y, w_x)

    blurred = np.real(ifft2(fft2(img) * psf2otf(psf, img.shape)))

    return beta * img + (1.0 - beta) * blurred
