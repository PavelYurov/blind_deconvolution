import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import zoom
from skimage.transform import resize as sk_resize


def convn_valid(u: np.ndarray, k: np.ndarray) -> np.ndarray:


    return fftconvolve(u, k, mode='valid')


def convn_full(u: np.ndarray, k: np.ndarray) -> np.ndarray:


    return fftconvolve(u, k, mode='full')


def pad_replicate(f: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:


    return np.pad(f, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')


def shft(u: np.ndarray, dx: int, dy: int) -> np.ndarray:


    M, N = u.shape
    us = np.zeros_like(u)


    r0 = max(-dy, 0)
    r1 = min(M, M - dy)
    c0 = max(-dx, 0)
    c1 = min(N, N - dx)


    sr0 = max(dy, 0)
    sr1 = min(dy + M, M)
    sc0 = max(dx, 0)
    sc1 = min(dx + N, N)

    us[r0:r1, c0:c1] = u[sr0:sr1, sc0:sc1] - u[r0:r1, c0:c1]
    return us


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:


    return np.power(img, gamma)


def make_size_odd(f: np.ndarray) -> np.ndarray:


    if f.shape[0] % 2 == 0:
        f = f[:-1, :]
    if f.shape[1] % 2 == 0:
        f = f[:, :-1]
    return f


def imresize_matlab(img: np.ndarray, target_shape: tuple,
                    order: int = 3) -> np.ndarray:


    th, tw = int(target_shape[0]), int(target_shape[1])
    oh, ow = img.shape[:2]

    if oh == th and ow == tw:
        return img.copy()

    return sk_resize(
        img, (th, tw),
        order=order,
        anti_aliasing=True,
        preserve_range=True,
        mode='edge',
    )


def psf2otf(psf: np.ndarray, shape: tuple) -> np.ndarray:


    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf

    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return np.fft.fft2(padded)


def get_gradient_operators(shape: tuple):


    kx = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]], dtype=np.float64)
    ky = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]], dtype=np.float64)
    OTF_dx = psf2otf(kx, shape)
    OTF_dy = psf2otf(ky, shape)
    return OTF_dx, OTF_dy, np.conj(OTF_dx), np.conj(OTF_dy)


def wiener_filter(img: np.ndarray, kernel: np.ndarray,
                  noise_snr: float = 0.01) -> np.ndarray:


    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (np.abs(otf) ** 2 + noise_snr)) * F_img
    return np.real(np.fft.ifft2(F_res))


def tikhonov_filter(img: np.ndarray, kernel: np.ndarray,
                    alpha: float = 0.01) -> np.ndarray:


    H, W = img.shape
    otf = psf2otf(kernel, (H, W))
    OTF_dx, OTF_dy, _, _ = get_gradient_operators((H, W))
    reg_term = np.abs(OTF_dx) ** 2 + np.abs(OTF_dy) ** 2
    denominator = np.abs(otf) ** 2 + alpha * reg_term
    F_img = np.fft.fft2(img)
    F_res = (np.conj(otf) / (denominator + 1e-12)) * F_img
    return np.real(np.fft.ifft2(F_res))


def edgetaper(img: np.ndarray, kernel: np.ndarray,
              n_tapers: int = 3) -> np.ndarray:


    H, W = img.shape
    kh, kw = kernel.shape


    acf = fftconvolve(kernel, kernel[::-1, ::-1], mode='full')
    acf_max = acf.max()
    if acf_max > 0:
        acf /= acf_max


    cy, cx = kh - 1, kw - 1
    z_col = acf[:, cx]
    z_row = acf[cy, :]


    beta_y = np.ones(H, dtype=np.float64)
    beta_x = np.ones(W, dtype=np.float64)

    half_ky = kh - 1
    if half_ky > 0:
        taper = z_col[:half_ky]
        n = min(len(taper), H // 2)
        beta_y[:n] = taper[:n]
        beta_y[-n:] = taper[:n][::-1]

    half_kx = kw - 1
    if half_kx > 0:
        taper = z_row[:half_kx]
        n = min(len(taper), W // 2)
        beta_x[:n] = taper[:n]
        beta_x[-n:] = taper[:n][::-1]


    alpha = beta_y[:, np.newaxis] * beta_x[np.newaxis, :]


    otf = psf2otf(kernel, (H, W))

    result = img.copy()
    for _ in range(n_tapers):
        blurred = np.real(np.fft.ifft2(otf * np.fft.fft2(result)))
        result = alpha * result + (1.0 - alpha) * blurred

    return result


def pad_image(img: np.ndarray, kernel_shape: tuple) -> np.ndarray:

    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='symmetric')


def crop_image(img: np.ndarray, original_shape: tuple,
               kernel_shape: tuple) -> np.ndarray:

    pad_h = kernel_shape[0]
    pad_w = kernel_shape[1]
    h, w = original_shape
    return img[pad_h:pad_h + h, pad_w:pad_w + w]
