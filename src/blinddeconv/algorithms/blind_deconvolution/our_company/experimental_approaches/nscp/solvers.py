import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from .utils import pad_kernel_centered, extract_kernel_center

EPS = 1e-8

def _crop_center(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:

    H, W = img.shape[:2]
    cy, cx = H // 2, W // 2
    y0 = cy - out_h // 2
    x0 = cx - out_w // 2
    return img[y0 : y0 + out_h, x0 : x0 + out_w]

def _pad_img(img: np.ndarray, h2: int, w2: int) -> np.ndarray:

    if img.ndim == 3:
        out = np.zeros((h2, w2, img.shape[2]), dtype=np.float32)
        out[: img.shape[0], : img.shape[1], :] = img
    else:
        out = np.zeros((h2, w2), dtype=np.float32)
        out[: img.shape[0], : img.shape[1]] = img
    return out

def update_l(
    l: np.ndarray,
    k: np.ndarray,
    b: np.ndarray,
    g: tuple,
    p: np.ndarray,
    lam: float,
    xi: float,
) -> np.ndarray:

    if b.ndim == 3:
        H, W, _C = b.shape
    else:
        H, W = b.shape
    fft_axes = (0, 1)

    kh, kw = k.shape
    H2 = H + kh - 1
    W2 = W + kw - 1

    bpad = _pad_img(b, H2, W2)
    ppad = _pad_img(p, H2, W2)

    Fb = fft2(bpad, s=(H2, W2), axes=fft_axes)
    Fp = fft2(ppad, s=(H2, W2), axes=fft_axes)

    kpad = pad_kernel_centered(k, (H2, W2))
    Fk = fft2(kpad, s=(H2, W2))
    if b.ndim == 3:
        Fk = Fk[:, :, np.newaxis]

    Dh_sp = np.zeros((H2, W2), dtype=np.float32)
    Dh_sp[0, 0] = -1.0
    if W2 > 1:
        Dh_sp[0, -1] = 1.0

    Dv_sp = np.zeros((H2, W2), dtype=np.float32)
    Dv_sp[0, 0] = -1.0
    if H2 > 1:
        Dv_sp[-1, 0] = 1.0

    FDh = fft2(Dh_sp, s=(H2, W2))
    FDv = fft2(Dv_sp, s=(H2, W2))

    if b.ndim == 3:
        FDh = FDh[:, :, np.newaxis]
        FDv = FDv[:, :, np.newaxis]

    g_h, g_v = g

    gh_pad = _pad_img(g_h, H2, W2)
    gv_pad = _pad_img(g_v, H2, W2)

    Fgh = fft2(gh_pad, s=(H2, W2), axes=fft_axes)
    Fgv = fft2(gv_pad, s=(H2, W2), axes=fft_axes)

    Fg = np.conj(FDh) * Fgh + np.conj(FDv) * Fgv

    numerator = np.conj(Fk) * Fb + lam * Fg + xi * Fp
    denominator = (
        np.abs(Fk) ** 2
        + lam * (np.abs(FDh) ** 2 + np.abs(FDv) ** 2)
        + xi
    )
    denominator = np.maximum(denominator, 1e-2)

    Fl = numerator / denominator
    l_full = np.real(ifft2(Fl, axes=fft_axes))

    l_new = l_full[:H, :W]
    if l_new.ndim == 3 and b.ndim == 3 and l_new.shape[2] != b.shape[2]:
        l_new = l_new[:, :, :b.shape[2]]
    l_new = np.clip(l_new, 0.0, 1.0)
    return l_new

def update_kernel(
    l: np.ndarray,
    b: np.ndarray,
    gamma: float,
    image_shape: tuple,
    prev_k: np.ndarray = None,
) -> np.ndarray:

    H, W = image_shape[:2]

    def _get_grad(img):
        gh = np.zeros_like(img)
        gv = np.zeros_like(img)
        gh[:, :-1] = img[:, 1:] - img[:, :-1]
        gv[:-1, :] = img[1:, :] - img[:-1, :]
        return gh, gv

    grad_l_h, grad_l_v = _get_grad(l)
    grad_b_h, grad_b_v = _get_grad(b)

    fft_axes = (0, 1)
    Flh = fft2(grad_l_h, axes=fft_axes)
    Flv = fft2(grad_l_v, axes=fft_axes)
    Fbh = fft2(grad_b_h, axes=fft_axes)
    Fbv = fft2(grad_b_v, axes=fft_axes)

    num_ch = np.conj(Flh) * Fbh + np.conj(Flv) * Fbv
    denom_ch = np.conj(Flh) * Flh + np.conj(Flv) * Flv

    if l.ndim == 3:
        numerator = np.sum(num_ch, axis=2)
        denominator = np.sum(denom_ch, axis=2)
    else:
        numerator = num_ch
        denominator = denom_ch

    denominator += gamma
    Fk = numerator / (denominator + EPS)

    k_full = np.real(ifft2(Fk, axes=fft_axes))

    expected = prev_k.shape if prev_k is not None else None
    k_small = extract_kernel_center(k_full, expected_size=expected)

    if k_small.sum() <= 1e-12:
        if prev_k is not None and prev_k.sum() > 1e-12:
            k_small = prev_k.copy()
        else:
            k_small = np.zeros((3, 3), dtype=np.float32)
            k_small[1, 1] = 1.0

    return k_small

def final_restore(
    img: np.ndarray,
    kernel: np.ndarray,
    snr_const: float = 0.015,
) -> np.ndarray:

    kernel = kernel / (np.sum(kernel) + 1e-8)

    H, W = img.shape[:2]
    kh, kw = kernel.shape

    k_pad = np.zeros((H, W), dtype=np.float32)
    y_off = (H - kh) // 2
    x_off = (W - kw) // 2
    k_pad[y_off : y_off + kh, x_off : x_off + kw] = kernel
    k_pad = ifftshift(k_pad)

    K_f = fft2(k_pad)
    denom = np.abs(K_f) ** 2 + snr_const

    def _apply_wiener(channel_img):
        Y_f = fft2(channel_img)
        numer = np.conj(K_f) * Y_f
        return np.real(ifft2(numer / denom))

    if img.ndim == 3:
        final = np.zeros_like(img)
        for c in range(img.shape[2]):
            final[:, :, c] = _apply_wiener(img[:, :, c])
    else:
        final = _apply_wiener(img)

    return np.clip(final, 0.0, 1.0)
