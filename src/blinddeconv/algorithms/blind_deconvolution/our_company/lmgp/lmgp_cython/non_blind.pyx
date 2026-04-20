# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from libc.math cimport log, pow
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import convolve1d
from scipy.stats import entropy

_LUT_RANGE = 10
_LUT_STEP = 0.0001
_XX = np.arange(-_LUT_RANGE, _LUT_RANGE + _LUT_STEP, _LUT_STEP)

def _compute_w1(v, beta): return np.sign(v) * np.maximum(np.abs(v) - 1.0 / beta, 0.0)

def _compute_w23(v, beta):
    eps = 1e-6
    m = np.full_like(v, 8.0 / (27.0 * beta ** 3))
    t1, t2, t3 = (-9.0 / 8.0) * v ** 2, (1.0 / 4.0) * v ** 3, (-1.0 / 8.0) * m * v ** 2
    t4 = -t3 / 2.0 + np.sqrt((-m ** 3 / 27.0 + m ** 2 * v ** 4 / 256.0).astype(np.complex128))
    t5 = t4 ** (1.0 / 3.0)
    t6 = 2.0 * (-5.0 / 18.0 * t1 + t5 + m / (3.0 * t5))
    t7 = np.sqrt((t1 / 3.0 + t6).astype(np.complex128))
    root = np.zeros(v.shape + (4,), dtype=np.complex128)
    root[:, 0] = 0.75 * v + (t7 + np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 1] = 0.75 * v + (t7 - np.sqrt(-(t1 + t6 + t2 / t7).astype(np.complex128))) / 2.0
    root[:, 2] = 0.75 * v + (-t7 + np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0
    root[:, 3] = 0.75 * v + (-t7 - np.sqrt(-(t1 + t6 - t2 / t7).astype(np.complex128))) / 2.0
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
    t2 = np.where(np.abs((-27.0 * m - 2.0 * v ** 3 + 3.0 ** 1.5 * np.sqrt(inner)) ** (1.0 / 3.0)) < eps, eps, (-27.0 * m - 2.0 * v ** 3 + 3.0 ** 1.5 * np.sqrt(inner)) ** (1.0 / 3.0))
    t3 = v ** 2 / t2
    root = np.zeros(v.shape + (3,), dtype=np.complex128)
    root[:, 0] = t1 + (2 ** (1.0 / 3.0)) / 3.0 * t3 + t2 / (3.0 * 2 ** (1.0 / 3.0))
    root[:, 1] = (t1 - ((1.0 + 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3 - ((1.0 - 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)
    root[:, 2] = (t1 - ((1.0 - 1j * np.sqrt(3.0)) / (3.0 * 2 ** (2.0 / 3.0))) * t3 - ((1.0 + 1j * np.sqrt(3.0)) / (6.0 * 2 ** (1.0 / 3.0))) * t2)
    root = np.where(np.isfinite(root), root, 0)
    vv = v[:, None]
    c23 = np.real(root) * np.sign(vv)
    c1 = np.abs(np.imag(root)) < eps
    c1 &= (c23 > (2.0 / 3.0) * np.abs(vv)) & (c23 < np.abs(vv))
    root[~c1] = 0
    return np.max(np.real(root), axis=1)

def _newton_w(v, beta, alpha):
    w = v.copy().astype(np.float64)
    for _ in range(4): w -= (alpha * np.sign(w) * np.abs(w) ** (alpha - 1) + beta * (w - v)) / (alpha * (alpha - 1) * np.abs(w) ** (alpha - 2) + beta)
    w = np.where(np.isfinite(w), w, 0)
    return np.where((np.abs(w) ** alpha + (beta / 2.0) * (w - v) ** 2) < ((beta / 2.0) * v ** 2), w, 0)

def _compute_w(v, beta, alpha):
    if abs(alpha - 1.0) < 1e-9: return _compute_w1(v, beta)
    if abs(alpha - 2.0 / 3.0) < 1e-9: return _compute_w23(v, beta)
    if abs(alpha - 0.5) < 1e-9: return _compute_w12(v, beta)
    return _newton_w(v, beta, alpha)

_lut_cache = {}
def _solve_img(v, beta, alpha):
    key = (beta, alpha)
    if key not in _lut_cache: _lut_cache[key] = _compute_w(_XX, beta, alpha)
    return np.interp(v.ravel(), _XX, _lut_cache[key]).reshape(v.shape)

def _clear_lut_cache(): _lut_cache.clear()

def _psf2otf(psf, shape):
    if psf.size == 0 or np.all(psf == 0): return np.zeros(shape, dtype=np.complex128)
    ph, pw = psf.shape
    padded = np.zeros(shape, dtype=np.float64)
    padded[:ph, :pw] = psf
    return fft2(np.roll(np.roll(padded, -(ph // 2), axis=0), -(pw // 2), axis=1))

def _fast_deconv_adaptive(yin, kernel, alpha, alpha_n, lam):
    M, N = yin.shape
    K, Y = _psf2otf(kernel, (M, N)), fft2(yin)
    Nomin1, Denom1 = np.conj(K) * Y, np.abs(K) ** 2
    Denom2 = np.abs(_psf2otf(np.array([[1, -1]], dtype=np.float64), (M, N))) ** 2 + np.abs(_psf2otf(np.array([[1], [-1]], dtype=np.float64), (M, N))) ** 2
    yout = yin.copy()
    youtx, youty = np.roll(yout, -1, axis=1) - yout, np.roll(yout, -1, axis=0) - yout
    youtn = yin - np.real(ifft2(fft2(yout) * K))
    betas, gamma = np.geomspace(1, 2 ** 8, num=9), 1.0 / 50.0
    beta_n, beta_g = betas * lam / gamma, betas

    for i in range(len(betas)):
        Wn, Wx, Wy = _solve_img(youtn, beta_n[i], alpha_n), _solve_img(youtx, beta_g[i], alpha), _solve_img(youty, beta_g[i], alpha)
        Wxx, Wyy = np.roll(Wx, 1, axis=1) - Wx, np.roll(Wy, 1, axis=0) - Wy
        Wnn = np.real(ifft2(fft2(Wn) * np.conj(K)))
        yout = np.clip(np.real(ifft2((fft2(-Wnn + gamma * (Wxx + Wyy)) + Nomin1) / (Denom1 + Denom2 * gamma))), 0, 1)
        youtx, youty = np.roll(yout, -1, axis=1) - yout, np.roll(yout, -1, axis=0) - yout
        youtn = yin - np.real(ifft2(fft2(yout) * K))
    return yout

def _dwt_hh(img):
    import pywt
    _, (_, _, HH) = pywt.dwt2(img, 'db2')
    return HH

def _local_std(grad_map, L=10):
    win = 2 * L + 1
    return np.sqrt(convolve2d(grad_map ** 2, np.ones((win, win), dtype=np.float64) / (win ** 2), mode='same', boundary='symm'))

def _x_grad(img): return convolve2d(img, np.array([[1, 0, -1]], dtype=np.float64), mode='same', boundary='symm')
def _y_grad(img): return convolve2d(img, np.array([[1, 0, -1]], dtype=np.float64).reshape(3, 1), mode='same', boundary='symm')

def _find_turning_point(sorted_std, M, N):
    original = sorted_std.copy()
    d = max(1, int(M * N / 2000))
    smooth = np.ones(2 * d + 1, dtype=np.float64)
    smooth[0], smooth[d + 1:] = 0, -1
    filtered = convolve1d(sorted_std, smooth, mode='reflect')
    ind = 0
    for i in range(1, len(filtered)):
        if filtered[i] < filtered[ind]: break
        ind = i - 1
    return original[ind]

def _estimate_noise_std(image):
    M, N = image.shape
    HH = _dwt_hh(image)
    sgx_n, sgy_n = _find_turning_point(np.sort(_local_std(_x_grad(HH), 10).ravel()), M, N), _find_turning_point(np.sort(_local_std(_y_grad(HH), 10).ravel()), M, N)
    return np.sqrt((sgx_n ** 2 + sgy_n ** 2 + 1e-8) / np.sum(np.array([1, 0, -1], dtype=np.float64) ** 2))

def _compute_lambda_map(image, sigma_n, alpha):
    eps = 1e-8
    M, N = image.shape
    HH = _dwt_hh(image)
    sgx_n, sgy_n = _find_turning_point(np.sort(_local_std(_x_grad(HH), 10).ravel()), M, N), _find_turning_point(np.sort(_local_std(_y_grad(HH), 10).ravel()), M, N)
    sgx, sgy = _local_std(_x_grad(image), 10), _local_std(_y_grad(image), 10)
    sigma_gsx_sq, sigma_gsy_sq = sgx ** 2 - sgx_n ** 2, sgy ** 2 - sgy_n ** 2
    sigma_gsx_sq[sigma_gsx_sq < eps], sigma_gsy_sq[sigma_gsy_sq < eps] = eps, eps
    return (np.sqrt(2 * sigma_n ** 2 / (np.sqrt(sigma_gsx_sq) ** 2 + np.sqrt(sigma_gsy_sq) ** 2 + eps))) ** alpha

def _estimate_alpha_n(blurred, restored, kernel, sigma_n):
    import math
    noise_observed, threshold, best_alpha, best_kl = blurred - restored, 0.3, 0.5, np.inf
    mask = (restored >= threshold) & (restored <= 1.0 - threshold)
    noise_sample = noise_observed[mask]

    for i in range(1, 10):
        alpha_n = round(0.1 * i, 2)
        rng = np.random.default_rng(0)
        noise_ref = (sigma_n * np.sqrt(math.gamma(1.0 / alpha_n) / math.gamma(3.0 / alpha_n))) * rng.choice([-1.0, 1.0], size=blurred.shape) * (rng.gamma(shape=1.0 / alpha_n, scale=1.0, size=blurred.shape) ** (1.0 / alpha_n))
        noise_ref_masked = noise_ref[mask]
        if noise_sample.size < 100: continue
        bins = np.arange(-threshold, threshold + 0.01, 0.01)
        hist_s, hist_r = np.histogram(noise_sample, bins)[0].astype(np.float64), np.histogram(noise_ref_masked, bins)[0].astype(np.float64)
        kl = entropy(hist_s / hist_s.sum() + 1e-12, hist_r / hist_r.sum() + 1e-12)
        if kl < best_kl: best_alpha, best_kl = alpha_n, kl
    return best_alpha

def _build_lambda_library(alpha, C, lam_N): return C * (2 ** ((alpha / 3.0) * np.arange(lam_N, dtype=np.float64)))

# ─────────────────────────────────────────────────────────────────────────────
# C-ОПТИМИЗИРОВАННАЯ ИНТЕРПОЛЯЦИЯ ДЛЯ NON-BLIND
# ─────────────────────────────────────────────────────────────────────────────

cdef void _do_interpolate(double[:, ::1] lam_map, double[::1] lam_lib, int[:, ::1] raw_idx,
                          double[:, :, ::1] I_lib, double[:, ::1] I_opt) noexcept:
    cdef int M = lam_map.shape[0]
    cdef int N = lam_map.shape[1]
    cdef int m, n, i
    cdef double denom, w, val_map, val_lib_i, val_lib_next

    for m in range(M):
        for n in range(N):
            i = raw_idx[m, n]
            val_lib_i = lam_lib[i]
            if val_lib_i < 1e-12: val_lib_i = 1e-12
            val_lib_next = lam_lib[i+1]

            denom = log(val_lib_next / val_lib_i)
            if denom < 1e-12 and denom > -1e-12: w = 0.5
            else:
                val_map = lam_map[m, n]
                if val_map < 1e-12: val_map = 1e-12
                w = pow(log(val_lib_next / val_map) / denom, 1.4)

            I_opt[m, n] = w * I_lib[i, m, n] + (1.0 - w) * I_lib[i+1, m, n]

def _interpolate_library(blurred, kernel, alpha, alpha_n, lam_map, lam_library):
    C = lam_library[0]
    I_library = {}
    sat, prev_I = False, None
    
    for idx in range(len(lam_library)):
        if not sat: I_library[idx] = _deconv_with_padding(blurred, kernel, alpha, alpha_n, lam_library[idx])
        else:
            I_library[idx] = prev_I.copy()
            continue
        if prev_I is not None and np.array_equal(I_library[idx], prev_I): sat = True
        prev_I = I_library[idx].copy()

    cdef int M = lam_map.shape[0], N = lam_map.shape[1]
    raw_idx = np.clip(np.ceil((3.0 / alpha) * np.log2(np.maximum(lam_map / C, 1.0))).astype(np.int32), 0, len(lam_library) - 2)
    
    I_lib_arr = np.zeros((len(lam_library), M, N), dtype=np.float64)
    for idx in range(len(lam_library)): I_lib_arr[idx] = I_library[idx]

    I_opt = np.zeros((M, N), dtype=np.float64)
    _do_interpolate(np.ascontiguousarray(lam_map, dtype=np.float64), np.ascontiguousarray(lam_library, dtype=np.float64), 
                    np.ascontiguousarray(raw_idx, dtype=np.int32), np.ascontiguousarray(I_lib_arr, dtype=np.float64), I_opt)

    return np.clip(I_opt, 0, 1)

def _mirror_pad(image, pad): return np.pad(image, pad, mode='reflect')
def _mirror_unpad(image, pad, orig_shape): return image[pad:pad + orig_shape[0], pad:pad + orig_shape[1]]
def _deconv_with_padding(blurimg, kernel, alpha, alpha_n, lam):
    k_size = kernel.shape[0]
    return _fast_deconv_adaptive(_mirror_pad(blurimg, k_size), kernel, alpha, alpha_n, lam)[k_size:k_size + blurimg.shape[0], k_size:k_size + blurimg.shape[1]]

def adaptive_lp_deconv(blurred, kernel, alpha=0.8, sigma_n=None, two_stage=True):
    kernel = np.maximum(kernel.astype(np.float64), 1e-10)
    kernel /= kernel.sum()
    blurred = blurred.astype(np.float64)
    if blurred.max() > 1.0: blurred /= 255.0

    if sigma_n is None: sigma_n = _estimate_noise_std(blurred)
    sigma_n = max(sigma_n, 1e-8)

    lam_map = _compute_lambda_map(blurred, sigma_n, alpha)
    C = max(lam_map.min(), 1e-12)
    lam_N = max(int(np.ceil(3.0 / alpha * np.log2(max(lam_map.max() / C, 1.0))) + 2), 3)
    lam_library = _build_lambda_library(alpha, C, lam_N)

    _clear_lut_cache()
    I_opt = _interpolate_library(blurred, kernel, alpha, alpha, lam_map, lam_library)

    if two_stage:
        alpha_n = _estimate_alpha_n(blurred, I_opt, kernel, sigma_n)
        if sigma_n > 0.025 or alpha_n == 0.5: alpha_n = max(alpha_n, 0.6)
        if kernel[kernel.shape[0] // 2, kernel.shape[1] // 2] < 1e-4: alpha_n = 0.8
        _clear_lut_cache()
        I_opt = _interpolate_library(blurred, kernel, alpha, alpha_n, lam_map, lam_library)

    return np.clip(I_opt, 0, 1)