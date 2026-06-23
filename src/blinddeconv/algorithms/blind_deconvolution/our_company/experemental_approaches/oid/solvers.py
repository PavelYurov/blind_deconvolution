import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.ndimage import correlate as nd_correlate
from scipy.ndimage import convolve as nd_convolve
from scipy.ndimage import label, map_coordinates

from .utils import (
    psf2otf,
    otf2psf,
    opt_fft_size,
    wrap_boundary_liu,
    conjgrad,
    adjust_psf_center,
    threshold_pxpy_v1,
    fftconv,
)

def L0Restoration(Im: np.ndarray, kernel: np.ndarray,
                  lambda_grad: float, kappa: float = 2.0) -> np.ndarray:

    H_orig, W_orig = Im.shape

    target = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape) - 1
    )
    Im_w = wrap_boundary_liu(Im, tuple(target))

    S = Im_w.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

    N, M = S.shape
    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    KER = psf2otf(kernel, sizeI2D)
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2

    Normin1 = np.conj(KER) * fft2(S)

    beta = 2.0 * lambda_grad
    while beta < betamax:
        Denormin = Den_KER + beta * Denormin2

        h = np.concatenate(
            [np.diff(S, axis=1), (S[:, 0] - S[:, -1])[:, None]],
            axis=1,
        )

        v = np.concatenate(
            [np.diff(S, axis=0), (S[0, :] - S[-1, :])[None, :]],
            axis=0,
        )

        t = (h ** 2 + v ** 2) < (lambda_grad / beta)
        h[t] = 0.0
        v[t] = 0.0

        Normin2 = np.concatenate(
            [(h[:, -1] - h[:, 0])[:, None], -np.diff(h, axis=1)],
            axis=1,
        )

        Normin2 = Normin2 + np.concatenate(
            [(v[-1, :] - v[0, :])[None, :], -np.diff(v, axis=0)],
            axis=0,
        )

        FS = (Normin1 + beta * fft2(Normin2)) / Denormin
        S = np.real(ifft2(FS))
        beta = beta * kappa

    return S[:H_orig, :W_orig]

def estimate_weightmatrix(blur: np.ndarray, latent: np.ndarray,
                          psf: np.ndarray, is_previous: bool) -> np.ndarray:

    alpha = 1.8e-3
    beta = 2e-4

    ww = np.ones_like(blur, dtype=np.float64)
    if is_previous:
        return ww

    bb = fftconv(latent, psf)
    temp = (blur - bb) ** 2
    w_matrix = 1.0 / (1.0 + np.exp((temp - alpha) / beta))

    max_blur = np.max(blur)
    min_blur = np.min(blur)
    w_matrix[blur == min_blur] = 0.0
    w_matrix[blur == max_blur] = 0.0

    eps = np.finfo(np.float64).eps
    w_matrix[w_matrix <= 0] = eps
    w_matrix[w_matrix >= 1] = 1.0 - eps

    ww = w_matrix

    ww[bb > 1] = 0.0
    ww[bb < 0] = 0.0
    return ww

_DXF  = np.array([[0.0, -1.0, 1.0]])
_DYF  = np.array([[0.0], [-1.0], [1.0]])
_DXXF = np.array([[-1.0, 2.0, -1.0]])
_DYYF = np.array([[-1.0], [2.0], [-1.0]])
_DXYF = np.array([[-1.0, 1.0, 0.0],
                  [ 1.0, -1.0, 0.0],
                  [ 0.0,  0.0, 0.0]])

def _Ax_deconv(x: np.ndarray, p: dict) -> np.ndarray:

    x = x.reshape(p['img_size'])
    x_f = fft2(x)

    y = np.real(ifft2(
        fft2(p['data_we'] * np.real(ifft2(p['psf_f'] * x_f))) * np.conj(p['psf_f'])
    ))

    L2 = p['L2_we']

    for f, w in (
        (_DXF,  p['weight_x']),
        (_DYF,  p['weight_y']),
        (_DXXF, p['weight_xx']),
        (_DYYF, p['weight_yy']),
        (_DXYF, p['weight_xy']),
    ):
        y = y + L2 * nd_convolve(
            w * nd_correlate(x, f, mode='wrap'),
            f, mode='wrap',
        )
    return y.ravel()

def deconv_L2(blurred: np.ndarray, latent0: np.ndarray, psf: np.ndarray,
              data_we: np.ndarray, L2_we: float,
              weight_x=None, weight_y=None,
              weight_xx=None, weight_yy=None, weight_xy=None) -> np.ndarray:

    if weight_x is None:
        weight_x  = np.ones_like(blurred)
        weight_y  = np.ones_like(blurred)
        weight_xx = np.zeros_like(blurred)
        weight_yy = np.zeros_like(blurred)
        weight_xy = np.zeros_like(blurred)

    img_size = blurred.shape
    psf_f = psf2otf(psf, img_size)

    b = np.real(ifft2(fft2(data_we * blurred) * np.conj(psf_f)))
    b = b.ravel()

    x = latent0.ravel().copy()

    p = {
        'psf': psf,
        'L2_we': L2_we,
        'data_we': data_we,
        'img_size': img_size,
        'psf_f': psf_f,
        'weight_x':  weight_x,
        'weight_y':  weight_y,
        'weight_xx': weight_xx,
        'weight_yy': weight_yy,
        'weight_xy': weight_xy,
    }

    x = conjgrad(x, b, 25, 1e-4, _Ax_deconv, p)
    return x.reshape(img_size)

def image_estimate(blurred: np.ndarray, psf: np.ndarray,
                   reg_strength: float, is_previous: bool):

    if blurred.ndim == 3:
        H, W, C = blurred.shape
        latent = np.empty_like(blurred, dtype=np.float64)
        w_out = np.empty_like(blurred, dtype=np.float64)
        for c in range(C):
            latent[:, :, c], w_out[:, :, c] = image_estimate(
                blurred[:, :, c], psf, reg_strength, is_previous
            )
        return latent, w_out

    w0 = 0.1
    exp_a = 0.8
    thr_e = 0.01
    N_iters = 15

    H, W = blurred.shape
    target = opt_fft_size(np.array([H, W]) + np.array(psf.shape) - 1)
    blurred_w = wrap_boundary_liu(blurred, tuple(target))
    w_matrix = wrap_boundary_liu(np.zeros_like(blurred), tuple(target))

    w_matrix[:H, :W] = 1.0

    mask = np.zeros_like(blurred_w)
    mask[:H, :W] = 1.0

    eps = np.finfo(np.float64).eps
    w_matrix[(1 - mask) == 1] = eps
    w_matrix[w_matrix >= 1] = 1.0 - eps
    w_matrix[w_matrix <= 0] = eps

    latent_w = deconv_L2(blurred_w, blurred_w, psf, w_matrix, reg_strength)

    ww = w_matrix
    for _ in range(N_iters):
        w_matrix = estimate_weightmatrix(blurred_w, latent_w, psf, is_previous)
        ww = w_matrix * mask

        dx  = nd_correlate(latent_w, _DXF,  mode='wrap')
        dy  = nd_correlate(latent_w, _DYF,  mode='wrap')
        dxx = nd_correlate(latent_w, _DXXF, mode='wrap')
        dyy = nd_correlate(latent_w, _DYYF, mode='wrap')
        dxy = nd_correlate(latent_w, _DXYF, mode='wrap')

        weight_x  =        w0 * np.maximum(np.abs(dx),  thr_e) ** (exp_a - 2)
        weight_y  =        w0 * np.maximum(np.abs(dy),  thr_e) ** (exp_a - 2)
        weight_xx = 0.25 * w0 * np.maximum(np.abs(dxx), thr_e) ** (exp_a - 2)
        weight_yy = 0.25 * w0 * np.maximum(np.abs(dyy), thr_e) ** (exp_a - 2)
        weight_xy = 0.25 * w0 * np.maximum(np.abs(dxy), thr_e) ** (exp_a - 2)

        latent_w = deconv_L2(
            blurred_w, latent_w, psf, ww, reg_strength,
            weight_x, weight_y, weight_xx, weight_yy, weight_xy,
        )

    latent = latent_w[:H, :W]
    w_out = ww[:H, :W]
    return latent, w_out

def _Ax_psf_coarse(x: np.ndarray, p: dict) -> np.ndarray:
    x_f = psf2otf(x, p['img_size'])
    y = otf2psf(p['m'] * x_f, p['psf_size'])
    y = y + p['lambda'] * x
    return y

def psf_coarse(blurred_x, blurred_y, latent_x, latent_y,
               weight: float, psf_size: tuple) -> np.ndarray:

    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)
    blurred_xf = fft2(blurred_x)
    blurred_yf = fft2(blurred_y)

    b_f = np.conj(latent_xf) * blurred_xf + np.conj(latent_yf) * blurred_yf
    b = np.real(otf2psf(b_f, psf_size))

    p = {
        'm': np.conj(latent_xf) * latent_xf + np.conj(latent_yf) * latent_yf,
        'img_size': blurred_xf.shape,
        'psf_size': psf_size,
        'lambda': weight,
    }

    psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
    psf = conjgrad(psf, b, 20, 1e-5, _Ax_psf_coarse, p)

    psf[psf < psf.max() * 0.05] = 0.0
    psf = psf / psf.sum()
    return psf

def _Ax_psf_fine(x: np.ndarray, p: dict) -> np.ndarray:
    x_f = psf2otf(x, p['img_size'])
    Ixconvk = p['ww_x'] * np.real(ifft2(p['latent_xf'] * x_f))
    Iyconvk = p['ww_y'] * np.real(ifft2(p['latent_yf'] * x_f))
    y = otf2psf(
        np.conj(p['latent_xf']) * fft2(Ixconvk) +
        np.conj(p['latent_yf']) * fft2(Iyconvk),
        p['psf_size'],
    )
    y = y + p['lambda'] * x
    return y

def psf_fine(blurred: np.ndarray, latent: np.ndarray, weight: float,
             psf: np.ndarray, threshold: float, is_previous: bool):

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H, W = blurred.shape
    target = opt_fft_size(np.array([H, W]) + np.array(psf.shape) - 1)
    blur_B_w = wrap_boundary_liu(blurred, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]

    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    latent_x, latent_y, _ = threshold_pxpy_v1(
        latent, np.max(psf.shape), threshold
    )
    psf_size = psf.shape

    latent_xf = fft2(latent_x)
    latent_yf = fft2(latent_y)

    ww_x = np.ones_like(Bx)
    for _ in range(15):
        ww_x = estimate_weightmatrix(Bx, latent_x, psf, is_previous)
        ww_y = estimate_weightmatrix(By, latent_y, psf, is_previous)

        b_f = (np.conj(latent_xf) * fft2(ww_x * Bx) +
               np.conj(latent_yf) * fft2(ww_y * By))
        b = np.real(otf2psf(b_f, psf_size))

        p = {
            'latent_xf': latent_xf,
            'latent_yf': latent_yf,
            'img_size': Bx.shape,
            'psf_size': psf_size,
            'lambda': weight,
            'ww_x': ww_x,
            'ww_y': ww_y,
        }

        psf = np.ones(psf_size, dtype=np.float64) / np.prod(psf_size)
        psf = conjgrad(psf, b, 21, 1e-5, _Ax_psf_fine, p)

        psf[psf < psf.max() * 0.05] = 0.0
        psf = psf / psf.sum()

    return psf, ww_x

def _prune_isolated(k: np.ndarray) -> np.ndarray:

    structure = np.ones((3, 3), dtype=np.int64)
    lbl, n_comp = label(k > 0, structure=structure)
    for ii in range(1, n_comp + 1):
        mask = (lbl == ii)
        if k[mask].sum() < 0.1:
            k[mask] = 0.0
    return k

def coarse_deblur(blur_B: np.ndarray, k: np.ndarray, lambda_grad: float,
                  threshold: float, opts: dict):

    dx = np.array([[-1, 1], [0, 0]], dtype=np.float64)
    dy = np.array([[-1, 0], [1, 0]], dtype=np.float64)

    H, W = blur_B.shape
    target = opt_fft_size(np.array([H, W]) + np.array(k.shape) - 1)
    blur_B_w = wrap_boundary_liu(blur_B, tuple(target))
    blur_B_tmp = blur_B_w[:H, :W]
    Bx = convolve2d(blur_B_tmp, dx, mode='valid')
    By = convolve2d(blur_B_tmp, dy, mode='valid')

    S = blur_B
    predeblur = str(opts.get('predeblur', 'L0'))

    for _ in range(int(opts['xk_iter'])):
        if predeblur == 'L0':
            S = L0Restoration(blur_B, k, lambda_grad, 2.0)
        else:
            S, _ = image_estimate(blur_B, k, lambda_grad, True)

        latent_x, latent_y, threshold = threshold_pxpy_v1(
            S, np.max(k.shape), threshold
        )
        k = psf_coarse(Bx, By, latent_x, latent_y, 5, k.shape)

        k = _prune_isolated(k)
        k[k < 0] = 0.0
        k = k / k.sum()

    k[k < 0] = 0.0
    k = k / k.sum()
    return k, lambda_grad, S

def fine_deblur(blur_B: np.ndarray, k: np.ndarray, lambda_grad: float,
                threshold: float, opts: dict):

    S = blur_B
    for it in range(int(opts['xk_iter'])):
        k_prev = k
        S, _wi = image_estimate(blur_B, k, lambda_grad, False)
        k, _wk = psf_fine(blur_B, S, 5, k, threshold, False)

        err = k_prev - k
        if np.linalg.norm(err.ravel(), 2) < 1e-3:
            break

        k = _prune_isolated(k)
        k[k < 0] = 0.0
        k = k / k.sum()

        if (it + 1) % 5 == 0:
            k = adjust_psf_center(k)
            k[k < 0] = 0.0
            k = k / k.sum()
            k_thresh = float(opts.get('k_thresh', 0))
            if k_thresh > 0:
                k[k < k.max() / k_thresh] = 0.0
            else:
                k[k < 0] = 0.0
            k = k / k.sum()

    k[k < 0] = 0.0
    k = k / k.sum()
    return k, lambda_grad, S

def _init_kernel(minsize: int) -> np.ndarray:

    k = np.zeros((minsize, minsize), dtype=np.float64)

    row_m = (minsize - 1) // 2
    col_m = (minsize - 1) // 2

    r = row_m - 1
    c0 = col_m - 1
    k[r, c0:c0 + 2] = 0.5
    return k

def _downSmpImC(I: np.ndarray, ret: float) -> np.ndarray:

    if ret == 1:
        return I.copy()

    sig = 1.0 / np.pi * ret
    g0 = np.arange(-50, 51, dtype=np.float64) * 2.0 * np.pi
    sf = np.exp(-0.5 * g0 ** 2 * sig ** 2)
    sf = sf / sf.sum()
    csf = np.cumsum(sf)
    csf = np.minimum(csf, csf[::-1])
    ii = np.where(csf > 0.05)[0]
    sf = sf[ii]

    tmp = convolve2d(I, sf[None, :], mode='valid')
    tmp = convolve2d(tmp, sf[:, None], mode='valid')

    step = 1.0 / ret
    H2, W2 = tmp.shape

    gx = np.arange(1.0, W2 + 1e-9, step) - 1.0
    gy = np.arange(1.0, H2 + 1e-9, step) - 1.0
    XX, YY = np.meshgrid(gx, gy)
    sI = map_coordinates(
        tmp, [YY.ravel(), XX.ravel()],
        order=1, mode='constant', cval=0.0,
    ).reshape(YY.shape)
    return sI

def _fixsize(f: np.ndarray, nk1: int, nk2: int) -> np.ndarray:

    k1, k2 = f.shape
    while (k1 != nk1) or (k2 != nk2):
        if k1 > nk1:
            s = f.sum(axis=1)
            if s[0] < s[-1]:
                f = f[1:, :]
            else:
                f = f[:-1, :]
        if k1 < nk1:
            s = f.sum(axis=1)
            tf = np.zeros((k1 + 1, f.shape[1]), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:k1, :] = f
            else:
                tf[1:k1 + 1, :] = f
            f = tf
        if k2 > nk2:
            s = f.sum(axis=0)
            if s[0] < s[-1]:
                f = f[:, 1:]
            else:
                f = f[:, :-1]
        if k2 < nk2:
            s = f.sum(axis=0)
            tf = np.zeros((f.shape[0], k2 + 1), dtype=f.dtype)
            if s[0] < s[-1]:
                tf[:, :k2] = f
            else:
                tf[:, 1:k2 + 1] = f
            f = tf
        k1, k2 = f.shape
    return f

def _resizeKer(k: np.ndarray, ret: float, k1: int, k2: int) -> np.ndarray:

    in_h, in_w = k.shape
    out_h = int(round(in_h * ret))
    out_w = int(round(in_w * ret))

    ys = np.linspace(0, in_h - 1, out_h)
    xs = np.linspace(0, in_w - 1, out_w)
    XX, YY = np.meshgrid(xs, ys)
    k_up = map_coordinates(
        k, [YY.ravel(), XX.ravel()], order=3, mode='constant', cval=0.0,
    ).reshape(out_h, out_w)

    k_up = np.maximum(k_up, 0.0)
    k_up = _fixsize(k_up, k1, k2)
    if k_up.max() > 0:
        k_up = k_up / k_up.sum()
    return k_up

def blind_deconv(y: np.ndarray, lambda_grad: float, opts: dict):

    if opts.get('gamma_correct', 1.0) != 1.0:
        y = y ** opts['gamma_correct']

    ret = np.sqrt(0.5)
    kernel_size = int(opts['kernel_size'])

    maxitr = max(int(np.floor(np.log(5.0 / kernel_size) / np.log(ret))), 0)
    num_scales = maxitr + 1
    print(f'Maximum iteration level is {num_scales}')

    retv = ret ** np.arange(0, maxitr + 1)

    k1list = np.ceil(kernel_size * retv).astype(np.int64)
    k1list = k1list + (k1list % 2 == 0).astype(np.int64)
    k2list = k1list.copy()

    ks = None
    threshold = None
    interim_latent = None
    kernel = None

    for s in range(num_scales, 0, -1):
        idx = s - 1

        if s == num_scales:
            ks = _init_kernel(int(k1list[idx]))
            k1 = int(k1list[idx])
            k2 = k1
        else:
            k1 = int(k1list[idx])
            k2 = k1
            ks = _resizeKer(ks, 1.0 / ret, int(k1list[idx]), int(k2list[idx]))

        cret = float(retv[idx])
        ys = _downSmpImC(y, cret)

        print(f'Processing scale {s}/{num_scales}; kernel size {k1}x{k2}; '
              f'image size {ys.shape[0]}x{ys.shape[1]}')

        if s == num_scales:
            _, _, threshold = threshold_pxpy_v1(ys, np.max(ks.shape))

        if s <= 1:

            opts_fine = dict(opts)
            opts_fine['xk_iter'] = int(opts.get('last_iter', opts['xk_iter']))
            ks, lambda_grad, interim_latent = fine_deblur(
                ys, ks, lambda_grad, threshold, opts_fine,
            )
        else:

            if opts.get('isnoisy', 0):

                from scipy.ndimage import gaussian_filter
                ys = gaussian_filter(ys, sigma=1.0, truncate=2.0, mode='nearest')
            ks, lambda_grad, interim_latent = coarse_deblur(
                ys, ks, lambda_grad, threshold, opts,
            )

        ks = adjust_psf_center(ks)
        ks[ks < 0] = 0.0
        sumk = ks.sum()
        if sumk > 0:
            ks = ks / sumk

        if s == 1:
            kernel = ks
            k_thresh = float(opts.get('k_thresh', 0))
            if k_thresh > 0:
                kernel[kernel < kernel.max() / k_thresh] = 0.0
            else:
                kernel[kernel < 0] = 0.0
            kernel = kernel / kernel.sum()

    return kernel, interim_latent
