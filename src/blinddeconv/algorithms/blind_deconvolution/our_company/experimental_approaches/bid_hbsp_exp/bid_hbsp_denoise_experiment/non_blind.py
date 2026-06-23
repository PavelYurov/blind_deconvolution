import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.stats import entropy
from scipy.fft import dstn, idstn

__all__ = ['adaptive_lp_deconv', 'ringing_artifacts_removal', 'firls_deconv']

_LUT_RANGE = 10
_LUT_STEP = 0.0001
_XX = np.arange(-_LUT_RANGE, _LUT_RANGE + _LUT_STEP, _LUT_STEP)

def _compute_w1(v, beta):

    return np.sign(v) * np.maximum(np.abs(v) - 1.0 / beta, 0.0)

def _compute_w23(v, beta):

    eps = 1e-6
    m = np.full_like(v, 8.0 / (27.0 * beta ** 3))
    t1 = (-9.0 / 8.0) * v ** 2
    t2 = (1.0 / 4.0) * v ** 3
    t3 = (-1.0 / 8.0) * m * v ** 2
    t4 = -t3 / 2.0 + np.sqrt(
        (-m ** 3 / 27.0 + m ** 2 * v ** 4 / 256.0).astype(np.complex128))
    t5 = t4 ** (1.0 / 3.0)
    t6 = 2.0 * (-5.0 / 18.0 * t1 + t5 + m / (3.0 * t5))
    t7 = np.sqrt((t1 / 3.0 + t6).astype(np.complex128))

    root = np.zeros(v.shape + (4,), dtype=np.complex128)
    root[:, 0] = 0.75 * v + (
        t7 + np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 1] = 0.75 * v + (
        t7 - np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 2] = 0.75 * v + (
        -t7 + np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0
    root[:, 3] = 0.75 * v + (
        -t7 - np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0

    c1 = np.abs(np.imag(root)) < eps
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 &= (c23 > 0.5 * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)

def _compute_w12(v, beta):

    eps = 1e-6
    m = -np.sign(v) / (4.0 * beta ** 2)
    t1 = (2.0 / 3.0) * v
    inner = (27.0 * m ** 2 + 4.0 * m * v ** 3).astype(np.complex128)
    t2 = (-27.0 * m - 2.0 * v ** 3 + 3.0 ** 1.5 * np.sqrt(inner)) ** (1.0 / 3.0)
    t2 = np.where(np.abs(t2) < eps, eps, t2)
    t3 = v ** 2 / t2

    root = np.zeros(v.shape + (3,), dtype=np.complex128)
    root[:, 0] = t1 + (2 ** (1.0 / 3.0)) / 3.0 * t3 + t2 / (3.0 * 2 ** (1.0 / 3.0))
    root[:, 1] = (
        t1
        - ((1.0 + 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 - 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)
    root[:, 2] = (
        t1
        - ((1.0 - 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3
        - ((1.0 + 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)

    root = np.where(np.isfinite(root), root, 0)
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 = np.abs(np.imag(root)) < eps
    c1 &= (c23 > (2.0 / 3.0) * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)

def _newton_w(v, beta, alpha):

    w = v.copy().astype(np.float64)
    for _ in range(4):
        df = alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)
        ddf = alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta
        w -= df / ddf
    w = np.where(np.isfinite(w), w, 0)
    cost0 = (beta / 2.0) * v ** 2
    costw = np.abs(w) ** alpha + (beta / 2.0) * (w - v) ** 2
    return np.where(costw < cost0, w, 0)

def _compute_w(v, beta, alpha):
    eps = 1e-9
    if abs(alpha - 1.0) < eps:
        return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < eps:
        return _compute_w23(v, beta)
    if abs(alpha - 0.5) < eps:
        return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)

_lut_cache = {}

def _solve_img(v, beta, alpha):

    key = (beta, alpha)
    if key not in _lut_cache:
        _lut_cache[key] = _compute_w(_XX, beta, alpha)
    lut = _lut_cache[key]
    return np.interp(v.ravel(), _XX, lut).reshape(v.shape)

def _clear_lut_cache():
    _lut_cache.clear()

def _psf2otf(psf, shape):

    if psf.size == 0 or np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    padded = np.roll(padded, -(ph // 2), axis=0)
    padded = np.roll(padded, -(pw // 2), axis=1)
    return fft2(padded)

def _fast_deconv_adaptive(yin, kernel, alpha, alpha_n, lam):

    M, N = yin.shape

    K = _psf2otf(kernel, (M, N))
    Y = fft2(yin)
    Nomin1 = np.conj(K) * Y
    Denom1 = np.abs(K) ** 2

    gx = np.array([[1, -1]], dtype=np.float64)
    gy = np.array([[1], [-1]], dtype=np.float64)
    Gx = _psf2otf(gx, (M, N))
    Gy = _psf2otf(gy, (M, N))
    Denom2 = np.abs(Gx) ** 2 + np.abs(Gy) ** 2

    yout = yin.copy()

    youtx = np.roll(yout, -1, axis=1) - yout
    youty = np.roll(yout, -1, axis=0) - yout
    youtn = yin - np.real(ifft2(fft2(yout) * K))

    betas = np.geomspace(1, 2 ** 8, num=9)
    gamma = 1.0 / 50.0
    beta_n = betas * lam / gamma
    beta_g = betas

    for i in range(len(betas)):

        Wn = _solve_img(youtn, beta_n[i], alpha_n)
        Wx = _solve_img(youtx, beta_g[i], alpha)
        Wy = _solve_img(youty, beta_g[i], alpha)

        Wxx = np.roll(Wx, 1, axis=1) - Wx
        Wyy = np.roll(Wy, 1, axis=0) - Wy
        Wnn = np.real(ifft2(fft2(Wn) * np.conj(K)))

        W = -Wnn + gamma * (Wxx + Wyy)
        Denom = Denom1 + Denom2 * gamma
        Fyout = (fft2(W) + Nomin1) / Denom
        yout = np.real(ifft2(Fyout))
        yout = np.clip(yout, 0, 1)

        youtx = np.roll(yout, -1, axis=1) - yout
        youty = np.roll(yout, -1, axis=0) - yout
        youtn = yin - np.real(ifft2(fft2(yout) * K))

    return yout

def _dwt_hh(img):

    import pywt
    _, (_, _, HH) = pywt.dwt2(img, 'db2')
    return HH

def _local_std(grad_map, L=10):

    win = 2 * L + 1
    k = np.ones((win, win), dtype=np.float64) / (win ** 2)
    ms_local = convolve2d(grad_map ** 2, k, mode='same', boundary='symm')
    return np.sqrt(ms_local)

def _x_grad(img):
    gx = np.array([[1, 0, -1]], dtype=np.float64)
    return convolve2d(img, gx, mode='same', boundary='symm')

def _y_grad(img):
    gy = np.array([[1, 0, -1]], dtype=np.float64).reshape(3, 1)
    return convolve2d(img, gy, mode='same', boundary='symm')

def _find_turning_point(sorted_std, M, N):

    original = sorted_std.copy()
    d = max(1, int(M * N / 2000))
    smooth = np.ones(2 * d + 1, dtype=np.float64)
    smooth[0] = 0
    smooth[d + 1:] = -1
    filtered = convolve1d(sorted_std, smooth, mode='reflect')

    ind = 0
    for i in range(1, len(filtered)):
        if filtered[i] < filtered[ind]:
            break
        ind = i - 1
    return original[ind]

def _estimate_noise_std(image):

    M, N = image.shape
    HH = _dwt_hh(image)
    Bgx, Bgy = _x_grad(HH), _y_grad(HH)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)
    sort_sgx = np.sort(sgx.ravel())
    sort_sgy = np.sort(sgy.ravel())
    sgx_n = _find_turning_point(sort_sgx, M, N)
    sgy_n = _find_turning_point(sort_sgy, M, N)
    Eg = np.sum(np.array([1, 0, -1], dtype=np.float64) ** 2)
    sigma_n = np.sqrt((sgx_n ** 2 + sgy_n ** 2 + 1e-8) / Eg)
    return sigma_n

def _compute_lambda_map(image, sigma_n, alpha):

    eps = 1e-8
    M, N = image.shape

    HH = _dwt_hh(image)
    Bgx_hh, Bgy_hh = _x_grad(HH), _y_grad(HH)
    sgx_hh, sgy_hh = _local_std(Bgx_hh, 10), _local_std(Bgy_hh, 10)
    sgx_n = _find_turning_point(np.sort(sgx_hh.ravel()), M, N)
    sgy_n = _find_turning_point(np.sort(sgy_hh.ravel()), M, N)

    Bgx, Bgy = _x_grad(image), _y_grad(image)
    sgx, sgy = _local_std(Bgx, 10), _local_std(Bgy, 10)

    sigma_gsx_sq = sgx ** 2 - sgx_n ** 2
    sigma_gsx_sq[sigma_gsx_sq < eps] = eps
    sigma_gsx = np.sqrt(sigma_gsx_sq)

    sigma_gsy_sq = sgy ** 2 - sgy_n ** 2
    sigma_gsy_sq[sigma_gsy_sq < eps] = eps
    sigma_gsy = np.sqrt(sigma_gsy_sq)

    lam_map = (np.sqrt(2 * sigma_n ** 2 / (
        sigma_gsx ** 2 + sigma_gsy ** 2 + eps))) ** alpha
    return lam_map

def _estimate_alpha_n(blurred, restored, kernel, sigma_n):

    import math

    noise_observed = blurred - restored

    threshold = 0.3
    best_alpha, best_kl = 0.5, np.inf

    for i in range(1, 10):
        alpha_n = round(0.1 * i, 2)

        rng = np.random.default_rng(0)
        beta_hl = sigma_n * np.sqrt(
            math.gamma(1.0 / alpha_n) / math.gamma(3.0 / alpha_n))
        T = rng.gamma(shape=1.0 / alpha_n, scale=1.0, size=blurred.shape)
        S = rng.choice([-1.0, 1.0], size=blurred.shape)
        noise_ref = beta_hl * S * (T ** (1.0 / alpha_n))

        mask = (restored >= threshold) & (restored <= 1.0 - threshold)
        noise_sample = noise_observed[mask]
        noise_ref_masked = noise_ref[mask]

        if noise_sample.size < 100:
            continue

        dx = 0.01
        bins = np.arange(-threshold, threshold + dx, dx)
        hist_s, _ = np.histogram(noise_sample, bins)
        hist_s = hist_s.astype(np.float64) / hist_s.sum() + 1e-12
        hist_r, _ = np.histogram(noise_ref_masked, bins)
        hist_r = hist_r.astype(np.float64) / hist_r.sum() + 1e-12

        kl = entropy(hist_s, hist_r)
        if kl < best_kl:
            best_alpha, best_kl = alpha_n, kl

    return best_alpha

def _build_lambda_library(alpha, C, lam_N):

    i = np.arange(lam_N, dtype=np.float64)
    return C * (2 ** ((alpha / 3.0) * i))

def _interpolate_library(blurred, kernel, alpha, alpha_n, lam_map, lam_library):

    C = lam_library[0]

    I_library = {}
    sat = False
    prev_I = None
    for idx in range(len(lam_library)):
        if not sat:
            I_library[idx] = _deconv_with_padding(
                blurred, kernel, alpha, alpha_n, lam_library[idx])
        else:
            I_library[idx] = prev_I.copy()
            continue
        if prev_I is not None and np.array_equal(I_library[idx], prev_I):
            sat = True
        prev_I = I_library[idx].copy()

    M, N = lam_map.shape
    raw_idx = np.ceil((3.0 / alpha) * np.log2(
        np.maximum(lam_map / C, 1.0))).astype(int)
    raw_idx = np.clip(raw_idx, 0, len(lam_library) - 2)

    w_map = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            denom = np.log(lam_library[i + 1] / max(lam_library[i], 1e-12))
            if abs(denom) < 1e-12:
                w_map[m, n] = 0.5
            else:
                w_map[m, n] = (np.log(
                    lam_library[i + 1] / max(lam_map[m, n], 1e-12)
                ) / denom) ** 1.4

    I_opt = np.zeros_like(lam_map)
    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            w = w_map[m, n]
            I_opt[m, n] = (
                w * I_library[i][m, n]
                + (1.0 - w) * I_library[i + 1][m, n])

    return np.clip(I_opt, 0, 1)

def _mirror_pad(image, pad):

    return np.pad(image, pad, mode='reflect')

def _mirror_unpad(image, pad, orig_shape):

    return image[pad:pad + orig_shape[0], pad:pad + orig_shape[1]]

def _deconv_with_padding(blurimg, kernel, alpha, alpha_n, lam):

    M, N = blurimg.shape
    k_size = kernel.shape[0]
    padded = _mirror_pad(blurimg, k_size)
    result = _fast_deconv_adaptive(padded, kernel, alpha, alpha_n, lam)
    return result[k_size:k_size + M, k_size:k_size + N]

def adaptive_lp_deconv(blurred, kernel, alpha=0.8, sigma_n=None,
                       two_stage=True):

    kernel = kernel.astype(np.float64)
    kernel = np.maximum(kernel, 1e-10)
    kernel /= kernel.sum()

    blurred = blurred.astype(np.float64)
    if blurred.max() > 1.0:
        blurred /= 255.0

    M, N = blurred.shape

    if sigma_n is None:
        sigma_n = _estimate_noise_std(blurred)
    sigma_n = max(sigma_n, 1e-8)

    lam_map = _compute_lambda_map(blurred, sigma_n, alpha)

    C = max(lam_map.min(), 1e-12)
    lam_N = int(np.ceil(3.0 / alpha * np.log2(
        max(lam_map.max() / C, 1.0))) + 2)
    lam_N = max(lam_N, 3)
    lam_library = _build_lambda_library(alpha, C, lam_N)

    alpha_n = alpha
    _clear_lut_cache()
    I_opt = _interpolate_library(
        blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    if two_stage:
        alpha_n = _estimate_alpha_n(blurred, I_opt, kernel, sigma_n)

        if sigma_n > 0.025 or alpha_n == 0.5:
            alpha_n = max(alpha_n, 0.6)
        center_val = kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]
        if center_val < 1e-4:
            alpha_n = 0.8

        _clear_lut_cache()
        I_opt = _interpolate_library(
            blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    return np.clip(I_opt, 0, 1)

def _firls_deb_core(y, h, lam, alpha, beta_a, epsilon_min,
                    out_iter, inner_iter):

    n1, n2 = y.shape

    H = _psf2otf(h, (n1, n2))
    Hx = _psf2otf(np.array([[1.0, -1.0]]), (n1, n2))
    Hy = _psf2otf(np.array([[1.0], [-1.0]]), (n1, n2))

    HH = H * np.conj(H)
    HHx = Hx * np.conj(Hx)
    HHy = Hy * np.conj(Hy)

    Y = np.conj(H) * fft2(y)
    RR = HHx + HHy
    invA = HH + beta_a * RR

    c = alpha * lam
    beta = alpha * lam / (epsilon_min ** (2.0 - alpha))

    x = y.copy()

    dx = np.concatenate([np.diff(x, n=1, axis=1),
                         x[:, 0:1] - x[:, -1:]], axis=1)
    dy = np.concatenate([np.diff(x, n=1, axis=0),
                         x[0:1, :] - x[-1:, :]], axis=0)
    adx = np.abs(dx)
    ady = np.abs(dy)

    dvx = np.zeros_like(x)
    dvy = np.zeros_like(x)

    eps_pow = 1e-12

    for _ in range(out_iter):

        Wx = np.minimum(beta, c * np.maximum(adx, eps_pow) ** (alpha - 2.0))
        Wy = np.minimum(beta, c * np.maximum(ady, eps_pow) ** (alpha - 2.0))

        for _ in range(inner_iter):

            vx = beta_a * (dx + dvx) / (Wx + beta_a)
            vy = beta_a * (dy + dvy) / (Wy + beta_a)

            dvx = dvx - vx + dx
            dvy = dvy - vy + dy

            tempx = vx - dvx
            tempy = vy - dvy

            adj_x = np.concatenate(
                [tempx[:, -1:] - tempx[:, 0:1],
                 -np.diff(tempx, n=1, axis=1)], axis=1)
            adj_y = np.concatenate(
                [tempy[-1:, :] - tempy[0:1, :],
                 -np.diff(tempy, n=1, axis=0)], axis=0)

            X = Y + beta_a * fft2(adj_x + adj_y)
            X = X / invA
            x = np.real(ifft2(X))

            dx = np.concatenate([np.diff(x, n=1, axis=1),
                                 x[:, 0:1] - x[:, -1:]], axis=1)
            dy = np.concatenate([np.diff(x, n=1, axis=0),
                                 x[0:1, :] - x[-1:, :]], axis=0)
            adx = np.abs(dx)
            ady = np.abs(dy)

    return x

def firls_deconv(blurred, kernel, lam=2e-5, alpha=0.8,
                 epsilon_min=2.0 / 255.0, epsilon_max=20.0 / 255.0,
                 beta_a=None, out_iter=5, inner_iter=3,
                 boundary='wrap', clip=True, use_edgetaper=None):

    kernel = kernel.astype(np.float64)
    kernel = np.maximum(kernel, 0.0)
    s = kernel.sum()
    if s <= 0:
        raise ValueError("firls_deconv: kernel has zero sum.")
    kernel = kernel / s

    y = blurred.astype(np.float64)
    if y.max() > 1.0:
        y = y / 255.0

    if beta_a is None:
        beta_a = lam * alpha * (epsilon_max ** (alpha - 2.0))

    if use_edgetaper is True:
        boundary = 'reflect'
    elif use_edgetaper is False:
        boundary = 'none'

    H_orig, W_orig = y.shape
    kh, kw = kernel.shape

    if boundary == 'wrap':

        target_size = _opt_fft_size(
            np.array([H_orig, W_orig]) + np.array([kh, kw]) - 1)
        y_pad = _wrap_boundary_liu(y, tuple(target_size))
        x_pad = _firls_deb_core(
            y_pad, kernel, lam, alpha, beta_a,
            epsilon_min, int(out_iter), int(inner_iter))
        x = x_pad[:H_orig, :W_orig]

    elif boundary == 'reflect':
        pad_h, pad_w = kh, kw
        y_pad = np.pad(y, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        x_pad = _firls_deb_core(
            y_pad, kernel, lam, alpha, beta_a,
            epsilon_min, int(out_iter), int(inner_iter))
        x = x_pad[pad_h:pad_h + H_orig, pad_w:pad_w + W_orig]

    elif boundary == 'none':
        x = _firls_deb_core(
            y, kernel, lam, alpha, beta_a,
            epsilon_min, int(out_iter), int(inner_iter))

    else:
        raise ValueError(
            f"firls_deconv: unknown boundary '{boundary}'. "
            "Use 'wrap', 'reflect', or 'none'.")

    if clip:
        x = np.clip(x, 0.0, 1.0)
    return x

_OPT_FFT_LUT = None

def _build_opt_fft_lut(lut_size=4096):
    lut = np.zeros(lut_size + 1, dtype=np.int64)
    e2 = 1
    while e2 <= lut_size:
        e3 = e2
        while e3 <= lut_size:
            e5 = e3
            while e5 <= lut_size:
                e7 = e5
                while e7 <= lut_size:
                    if e7 <= lut_size:
                        lut[e7] = e7
                    if e7 * 11 <= lut_size:
                        lut[e7 * 11] = e7 * 11
                    if e7 * 13 <= lut_size:
                        lut[e7 * 13] = e7 * 13
                    e7 *= 7
                e5 *= 5
            e3 *= 3
        e2 *= 2
    nn = 0
    for i in range(lut_size, 0, -1):
        if lut[i] != 0:
            nn = i
        else:
            lut[i] = nn
    return lut

def _opt_fft_size(n):
    global _OPT_FFT_LUT
    if _OPT_FFT_LUT is None:
        _OPT_FFT_LUT = _build_opt_fft_lut()
    n = np.asarray(n, dtype=np.int64)
    scalar_input = n.ndim == 0
    n = np.atleast_1d(n)
    lut_size = len(_OPT_FFT_LUT) - 1
    m = np.zeros_like(n)
    for i in range(n.size):
        nn = n.flat[i]
        if 1 <= nn <= lut_size:
            m.flat[i] = _OPT_FFT_LUT[nn]
        else:
            m.flat[i] = -1
    if scalar_input:
        return int(m.flat[0])
    return m

def _solve_min_laplacian(boundary_image):
    H, W = boundary_image.shape
    boundary_image = boundary_image.copy()
    boundary_image[1:-1, 1:-1] = 0.0
    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H-1, 1:W-1] = (
        -4.0 * boundary_image[1:H-1, 1:W-1]
        + boundary_image[1:H-1, 2:W]
        + boundary_image[1:H-1, 0:W-2]
        + boundary_image[0:H-2, 1:W-1]
        + boundary_image[2:H,   1:W-1]
    )
    f1 = -f_bp
    f2 = f1[1:H-1, 1:W-1]
    f2sin = dstn(f2, type=1)
    x = np.arange(1, W - 1)
    y = np.arange(1, H - 1)
    xx, yy = np.meshgrid(x, y)
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) +\
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)
    f3 = f2sin / denom
    img_tt = idstn(f3, type=1)
    img_direct = boundary_image.copy()
    img_direct[1:H-1, 1:W-1] = img_tt
    return img_direct

def _wrap_boundary_liu(img, img_size):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    H, W, Ch = img.shape
    H_out, W_out = img_size[0], img_size[1]
    H_w = H_out - H
    W_w = W_out - W
    ret = np.zeros((H_out, W_out, Ch), dtype=np.float64)
    for ch in range(Ch):
        alpha = 1
        HG = img[:, :, ch]
        r_A = np.zeros((alpha * 2 + H_w, W), dtype=np.float64)
        r_A[:alpha, :] = HG[-alpha:, :]
        r_A[-alpha:, :] = HG[:alpha, :]
        if H_w > 1:
            a = np.arange(H_w, dtype=np.float64) / (H_w - 1)
        else:
            a = np.array([0.0])
        r_A[alpha:alpha + H_w, 0] = (
            (1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0])
        r_A[alpha:alpha + H_w, -1] = (
            (1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1])
        A2 = _solve_min_laplacian(r_A)
        r_A = A2
        A = r_A

        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]
        if W_w > 1:
            a = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            a = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (
            (1 - a) * r_B[0, alpha - 1] + a * r_B[0, -alpha])
        r_B[-1, alpha:alpha + W_w] = (
            (1 - a) * r_B[-1, alpha - 1] + a * r_B[-1, -alpha])
        B2 = _solve_min_laplacian(r_B)
        r_B = B2
        B = r_B

        r_C = np.zeros((alpha * 2 + H_w, alpha * 2 + W_w), dtype=np.float64)
        r_C[:alpha, :] = B[-alpha:, :]
        r_C[-alpha:, :] = B[:alpha, :]
        r_C[:, :alpha] = A[:, -alpha:]
        r_C[:, -alpha:] = A[:, :alpha]
        C2 = _solve_min_laplacian(r_C)
        r_C = C2
        C = r_C

        A = A[:H_w, :]
        B = B[:, 1:W_w + 1]
        C = C[1:H_w + 1, 1:W_w + 1]
        ret[:, :, ch] = np.block([[HG, B], [A, C]])

    if ret.shape[2] == 1:
        return ret[:, :, 0]
    return ret

def _rr_psf2otf(psf, shape):
    if np.all(psf == 0):
        return np.zeros(shape, dtype=np.complex128)
    in_h, in_w = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:in_h, :in_w] = psf
    padded = np.roll(padded, -(in_h // 2), axis=0)
    padded = np.roll(padded, -(in_w // 2), axis=1)
    return fft2(padded)

def _computeDenominator(y, k):
    sizey = y.shape[:2]
    otfk = _rr_psf2otf(k, sizey)
    Nomin1 = np.conj(otfk) * fft2(y)
    Denom1 = np.abs(otfk) ** 2
    Denom2 = (np.abs(_rr_psf2otf(np.array([[1, -1]], dtype=np.float64), sizey)) ** 2
              + np.abs(_rr_psf2otf(np.array([[1], [-1]], dtype=np.float64), sizey)) ** 2)
    return Nomin1, Denom1, Denom2

def _deblurring_adm_aniso(B, k, lambda_tv, alpha):
    beta = 1.0 / lambda_tv
    beta_min = 0.001
    m, n = B.shape
    I = B.copy()
    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)
    Ix = np.concatenate([np.diff(I, n=1, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, n=1, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)
    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2
        if alpha == 1:
            Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
            Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)
        else:
            raise NotImplementedError(
                f"deblurring_adm_aniso: alpha={alpha} not implemented")
        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, n=1, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, n=1, axis=0)], axis=0)
        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))
        Ix = np.concatenate([np.diff(I, n=1, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, n=1, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)
        beta = beta / 2.0
    return I

def _L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    H_orig, W_orig = Im.shape[0], Im.shape[1]
    target_size = _opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1)
    Im = _wrap_boundary_liu(Im, tuple(target_size))
    S = Im.copy()
    betamax = 1e5
    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)
    N, M = Im.shape[:2]
    sizeI2D = (N, M)
    otfFx = _rr_psf2otf(fx, sizeI2D)
    otfFy = _rr_psf2otf(fy, sizeI2D)
    KER = _rr_psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Normin1 = np.conj(KER) * fft2(S)
    beta = 2 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2
        h = np.concatenate([np.diff(S, n=1, axis=1),
                            S[:, 0:1] - S[:, -1:]], axis=1)
        v = np.concatenate([np.diff(S, n=1, axis=0),
                            S[0:1, :] - S[-1:, :]], axis=0)
        t = (h ** 2 + v ** 2) < lambda_grad / beta
        h[t] = 0.0
        v[t] = 0.0
        Normin2_val = np.concatenate([h[:, -1:] - h[:, 0:1],
                                      -np.diff(h, n=1, axis=1)], axis=1)
        Normin2_val = Normin2_val + np.concatenate(
            [v[-1:, :] - v[0:1, :],
             -np.diff(v, n=1, axis=0)], axis=0)
        FS = (Normin1 + beta * fft2(Normin2_val)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa
    S = S[:H_orig, :W_orig]
    return S

def _fspecial_gaussian(size, sigma):
    radius = (size - 1) / 2.0
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return g / g.sum()

def _bilateral_filter(img, sigma_s, sigma):
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    was_2d = img.shape[2] == 1
    h, w, d = img.shape
    img = img.astype(np.float32)
    lab = img.copy()
    sigma = sigma * np.sqrt(d)
    fr = int(np.ceil(sigma_s * 3))
    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    p_lab = np.pad(lab, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)
    spatial_weight = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * sigma
    for y in range(-fr, fr + 1):
        for x in range(-fr, fr + 1):
            w_s = spatial_weight[y + fr, x + fr]
            n_img = p_img[fr + y:fr + y + h, fr + x:fr + x + w, :]
            n_lab = p_lab[fr + y:fr + y + h, fr + x:fr + x + w, :]
            f_diff = lab - n_lab
            f_dist = np.sum(f_diff ** 2, axis=2)
            w_f = np.exp(-0.5 * f_dist / ss)
            w_t = w_s * w_f
            r_img += n_img * w_t[:, :, np.newaxis]
            w_sum += w_t
    r_img = r_img / w_sum[:, :, np.newaxis]
    if was_2d:
        return r_img[:, :, 0]
    return r_img

def ringing_artifacts_removal(y, kernel, lambda_tv=4e-3, lambda_l0=2e-3,
                              weight_ring=0.5):

    H, W = y.shape[:2]
    target_size = _opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    y_pad = _wrap_boundary_liu(y, tuple(target_size))

    Latent_tv = _deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    Latent_l0 = _L0Restoration(y_pad, kernel, lambda_l0, 2)
    Latent_l0 = Latent_l0[:H, :W]

    diff_img = Latent_tv - Latent_l0
    bf_diff = _bilateral_filter(diff_img, 3, 0.1)

    result = Latent_tv - weight_ring * bf_diff
    return result
