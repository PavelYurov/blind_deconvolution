import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
import cv2

from .utils import (
    conv2_valid,
    conv2_same,
    conv2_full,
    psf2otf,
    compute_constants,
    init_kernel,
    center_kernel_separate,
    edgetaper,
)

_LUT_RANGE = 10.0
_LUT_STEP = 0.0001
_lut_xx = np.linspace(-_LUT_RANGE, _LUT_RANGE, int(2 * _LUT_RANGE / _LUT_STEP) + 1)
_lut_cache = {}

def _compute_w1(v, beta_val):

    return np.maximum(np.abs(v) - 1.0 / beta_val, 0) * np.sign(v)

def _compute_w23(v, beta_val):

    epsilon = 1e-6
    k_const = 8.0 / (27.0 * beta_val ** 3)
    m = np.full_like(v, k_const, dtype=np.float64)

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    m2 = m * m
    m3 = m2 * m

    a = -1.125 * v2
    b2 = 0.25 * v3

    q = -0.125 * (m * v2)
    r1_inner = (-m3 / 27.0 + (m2 * v4) / 256.0).astype(np.complex128)
    r1 = -q / 2.0 + np.sqrt(r1_inner)

    u = np.exp(np.log(r1.astype(np.complex128)) / 3.0)
    y_var = 2.0 * (-5.0 / 18.0 * a + u + (m / (3.0 * u)))

    W = np.sqrt((a / 3.0 + y_var).astype(np.complex128))

    inner1 = (-(a + y_var + b2 / W)).astype(np.complex128)
    inner2 = (-(a + y_var - b2 / W)).astype(np.complex128)
    roots = np.zeros((len(v), 4), dtype=np.complex128)
    roots[:, 0] = 0.75 * v + 0.5 * (W + np.sqrt(inner1))
    roots[:, 1] = 0.75 * v + 0.5 * (W - np.sqrt(inner1))
    roots[:, 2] = 0.75 * v + 0.5 * (-W + np.sqrt(inner2))
    roots[:, 3] = 0.75 * v + 0.5 * (-W - np.sqrt(inner2))

    v_rep = np.tile(v[:, None], (1, 4))
    sv = np.sign(v_rep)
    rsv = np.real(roots) * sv

    valid = ((np.abs(np.imag(roots)) < epsilon) &
             (rsv > (np.abs(v_rep) / 2.0)) &
             (rsv < np.abs(v_rep)))

    scored = np.where(valid, rsv, 0.0)
    best_idx = np.argmax(scored, axis=1)
    w = scored[np.arange(len(v)), best_idx] * np.sign(v)
    return np.real(w)

def _compute_w12(v, beta_val):

    epsilon = 1e-6
    k_const = -0.25 / beta_val ** 2
    m = k_const * np.sign(v)

    t1 = (2.0 / 3.0) * v
    v2 = v * v
    v3 = v2 * v

    inner = (27.0 * m ** 2 + 4.0 * m * v3).astype(np.complex128)
    t2_arg = (-27.0 * m - 2.0 * v3 + 3.0 * np.sqrt(3.0) * np.sqrt(inner)).astype(np.complex128)
    t2 = np.exp(np.log(t2_arg) / 3.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        t3 = v2.astype(np.complex128) / t2
    t3 = np.where(np.isnan(t3) | np.isinf(t3), 0.0, t3)

    cbrt2 = 2.0 ** (1.0 / 3.0)
    roots = np.zeros((len(v), 3), dtype=np.complex128)
    roots[:, 0] = t1 + cbrt2 / 3.0 * t3 + t2 / (3.0 * cbrt2)
    roots[:, 1] = t1 - ((1 + 1j * np.sqrt(3.0)) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3\
                      - ((1 - 1j * np.sqrt(3.0)) / (6.0 * cbrt2)) * t2
    roots[:, 2] = t1 - ((1 - 1j * np.sqrt(3.0)) / (3.0 * 2.0 ** (2.0 / 3.0))) * t3\
                      - ((1 + 1j * np.sqrt(3.0)) / (6.0 * cbrt2)) * t2

    roots = np.where(np.isnan(roots) | np.isinf(roots), 0.0, roots)

    v_rep = np.tile(v[:, None], (1, 3))
    sv = np.sign(v_rep)
    rsv = np.real(roots) * sv

    valid = ((np.abs(np.imag(roots)) < epsilon) &
             (rsv > (2.0 * np.abs(v_rep) / 3.0)) &
             (rsv < np.abs(v_rep)))

    scored = np.where(valid, rsv, 0.0)
    best_idx = np.argmax(scored, axis=1)
    w = scored[np.arange(len(v)), best_idx] * np.sign(v)
    return np.real(w)

def _newton_w(v, beta_val, alpha_val):

    x = v.copy()
    for _ in range(4):
        with np.errstate(divide='ignore', invalid='ignore'):
            fd = alpha_val * np.sign(x) * np.abs(x) ** (alpha_val - 1) + beta_val * (x - v)
            fdd = alpha_val * (alpha_val - 1) * np.abs(x) ** (alpha_val - 2) + beta_val
            x = x - fd / fdd
    x = np.where(np.isnan(x), 0.0, x)

    z = beta_val / 2.0 * v ** 2
    f = np.abs(x) ** alpha_val + beta_val / 2.0 * (x - v) ** 2
    w = np.where(f < z, x, 0.0)
    return w

def _compute_w(xx, beta_val, alpha_val):

    if abs(alpha_val - 1.0) < 1e-9:
        return _compute_w1(xx, beta_val)
    if abs(alpha_val - 2.0 / 3.0) < 1e-9:
        return _compute_w23(xx, beta_val)
    if abs(alpha_val - 0.5) < 1e-9:
        return _compute_w12(xx, beta_val)
    return _newton_w(xx, beta_val, alpha_val)

def _interp_extrap(xp, fp, x):

    result = np.interp(x, xp, fp)

    below = x < xp[0]
    if np.any(below):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        result[below] = fp[0] + slope * (x[below] - xp[0])
    above = x > xp[-1]
    if np.any(above):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        result[above] = fp[-1] + slope * (x[above] - xp[-1])
    return result

def solve_image_bregman(v, beta_val, alpha_val):

    key = (beta_val, alpha_val)
    if key not in _lut_cache:
        _lut_cache[key] = _compute_w(_lut_xx.copy(), beta_val, alpha_val)
    lut = _lut_cache[key]
    w = _interp_extrap(_lut_xx, lut, v.ravel())
    return w.reshape(v.shape)

def fast_deconv_bregman(f, k, lambda_, alpha):

    beta = 400.0
    initer_max = 1
    outiter_max = 20

    g = f.copy()

    dx = np.array([[1.0, -1.0]])
    dy = np.array([[1.0], [-1.0]])
    dxt = np.array([[-1.0, 1.0]])
    dyt = np.array([[-1.0], [1.0]])

    Ktf, KtK, DtD = compute_constants(f, k, dx, dy)

    gx = conv2_valid(g, dx)
    gy = conv2_valid(g, dy)

    bx = np.zeros_like(gx)
    by = np.zeros_like(gy)
    wx = gx.copy()
    wy = gy.copy()

    for outiter in range(outiter_max):
        for initer in range(initer_max):

            if abs(alpha - 1.0) < 1e-9:

                tmpx = gx + bx
                tmpy = gy + by
                wx = np.maximum(np.abs(tmpx) - 1.0 / beta, 0.0) * np.sign(tmpx)
                wy = np.maximum(np.abs(tmpy) - 1.0 / beta, 0.0) * np.sign(tmpy)
            else:
                wx = solve_image_bregman(gx + bx, beta, alpha)
                wy = solve_image_bregman(gy + by, beta, alpha)

            bx = bx - wx + gx
            by = by - wy + gy

            wx1 = conv2_full(wx - bx, dxt)
            wy1 = conv2_full(wy - by, dyt)

            num = lambda_ * Ktf + beta * fft2(wx1 + wy1)
            denom = lambda_ * KtK + beta * DtD

            with np.errstate(divide='ignore', invalid='ignore'):
                Fg = num / denom
            Fg = np.nan_to_num(Fg, nan=0.0, posinf=0.0, neginf=0.0)

            g = np.real(ifft2(Fg))

            gx = conv2_valid(g, dx)
            gy = conv2_valid(g, dy)

    return g

def pcg_kernel_core_irls_conv(k, X, flipX, weights):

    out_l2 = np.zeros_like(k)

    for i in range(len(X)):
        tmp1 = conv2_valid(X[i], k)
        tmp2 = conv2_valid(flipX[i], tmp1)
        out_l2 = out_l2 + tmp2

    out_l1 = weights * k
    return out_l1 + out_l2

def local_cg(k, X, flipX, weights, rhs, tol, max_its):

    k_out = k.copy()
    Ak = pcg_kernel_core_irls_conv(k_out, X, flipX, weights)
    r = rhs - Ak

    rho_1 = 0.0
    p = np.zeros_like(r)

    for it in range(max_its):
        rho = np.sum(r * r)

        if it > 0:
            beta_cg = rho / rho_1 if rho_1 != 0 else 0.0
            p = r + beta_cg * p
        else:
            p = r.copy()

        Ap = pcg_kernel_core_irls_conv(p, X, flipX, weights)
        q = Ap
        pq = np.sum(p * q)
        alpha_cg = rho / pq if pq != 0 else 0.0

        k_out = k_out + alpha_cg * p
        r = r - alpha_cg * q
        rho_1 = rho

        if rho < tol:
            break

    return k_out

def pcg_kernel_irls_conv(k_init, X, Y, opts):

    lambda_ = opts.get('lambda', 0.0)
    pcg_tol = opts.get('pcg_tol', 1e-4)
    pcg_its = opts.get('pcg_its', 1)

    flipX = [x[::-1, ::-1] for x in X]
    rhs_list = []
    for i in range(len(X)):
        rhs_i = conv2_valid(flipX[i], Y[i])
        rhs_list.append(rhs_i)

    rhs = np.zeros_like(rhs_list[0])
    for r_item in rhs_list:
        rhs = rhs + r_item

    k_out = k_init.copy()

    for _ in range(1):
        k_prev = k_out.copy()

        weights_l1 = lambda_ * (np.maximum(np.abs(k_prev), 0.0001) ** (-1.0))
        k_out = local_cg(k_prev, X, flipX, weights_l1, rhs, pcg_tol, pcg_its)

    return k_out

def ss_blind_deconv(y, x, k, lambda_, delta, x_in_iter, x_out_iter,
                    xk_iter, k_reg_wt):

    error_flag = 0
    khs = k.shape[0] // 2

    m, n = y.shape
    m2 = n // 2

    y1 = [y[:, :m2].copy(), y[:, m2:].copy()]

    if khs > 0:
        y2 = [y1[0][khs:-khs, khs:-khs].copy(), y1[1][khs:-khs, khs:-khs].copy()]
    else:
        y2 = [y1[0].copy(), y1[1].copy()]

    x1 = [x[:, :m2].copy(), x[:, m2:].copy()]

    tmp_0 = conv2_same(x1[0], k) - y1[0]
    tmp_1 = conv2_same(x1[1], k) - y1[1]
    n0 = np.linalg.norm(tmp_0)
    n1 = np.linalg.norm(tmp_1)
    lcost = (lambda_ / 2.0) * (n0 ** 2 + n1 ** 2)
    nx0 = np.linalg.norm(x1[0])
    nx1 = np.linalg.norm(x1[1])
    pcost_0 = np.sum(np.abs(x1[0])) / nx0 if nx0 > 0 else 0.0
    pcost_1 = np.sum(np.abs(x1[1])) / nx1 if nx1 > 0 else 0.0
    pcost = pcost_0 + pcost_1

    normy = [np.linalg.norm(y1[0]), np.linalg.norm(y1[1])]

    lambda_orig = lambda_

    for iter_ in range(xk_iter):
        lambda_ = lambda_orig

        cost_before_x = lcost + pcost
        x2 = [x1[0].copy(), x1[1].copy()]

        while delta > 1e-4:
            for out_iter in range(x_out_iter):

                normx = [np.linalg.norm(x1[0]), np.linalg.norm(x1[1])]
                beta_ista = [lambda_ * normx[0], lambda_ * normx[1]]

                for inn_iter in range(x_in_iter):
                    x1prev = [x1[0].copy(), x1[1].copy()]
                    flipped_k = k[::-1, ::-1]

                    for ch in range(2):

                        fwd = conv2_same(x1prev[ch], k)
                        residual = y1[ch] - fwd
                        adj = conv2_same(residual, flipped_k)
                        v = x1prev[ch] + beta_ista[ch] * delta * adj

                        x1[ch] = np.maximum(np.abs(v) - delta, 0.0) * np.sign(v)

                        nx = np.linalg.norm(x1[ch])
                        if nx > 0:
                            x1[ch] = x1[ch] * normy[ch] / nx

                    tmp_0 = conv2_same(x1[0], k) - y1[0]
                    tmp_1 = conv2_same(x1[1], k) - y1[1]
                    n0 = np.linalg.norm(tmp_0)
                    n1 = np.linalg.norm(tmp_1)
                    lcost = (lambda_ / 2.0) * (n0 ** 2 + n1 ** 2)
                    nx0 = np.linalg.norm(x1[0])
                    nx1 = np.linalg.norm(x1[1])
                    pcost_0 = np.sum(np.abs(x1[0])) / nx0 if nx0 > 0 else 0.0
                    pcost_1 = np.sum(np.abs(x1[1])) / nx1 if nx1 > 0 else 0.0
                    pcost = pcost_0 + pcost_1

            cost_after_x = lcost + pcost
            if cost_after_x > 3.0 * cost_before_x:
                x1 = [x2[0].copy(), x2[1].copy()]
                delta = delta / 2.0
            else:
                break

        k_opts = {'lambda': k_reg_wt, 'pcg_tol': 1e-4, 'pcg_its': 1}

        k_prev = k.copy()
        k = pcg_kernel_irls_conv(k_prev, x1, y2, k_opts)
        k[k < 0] = 0.0
        sk = k.sum()
        if sk > 0:
            k = k / sk

    x_out = np.concatenate([x1[0], x1[1]], axis=1)
    return x_out, k, error_flag

def ms_blind_deconv(y_input, opts):

    kernel_size = opts.get('kernel_size', 31)
    prescale = opts.get('prescale', 1.0)
    gamma_correct = opts.get('gamma_correct', 1.0)
    min_lambda = opts.get('min_lambda', 100.0)
    k_reg_wt = opts.get('k_reg_wt', 0.0)
    delta = opts.get('delta', 0.001)
    x_in_iter = opts.get('x_in_iter', 2)
    x_out_iter = opts.get('x_out_iter', 2)
    xk_iter = opts.get('xk_iter', 21)
    k_thresh = opts.get('k_thresh', 0.0)
    nb_lambda = opts.get('nb_lambda', 3000.0)
    nb_alpha = opts.get('nb_alpha', 1.0)
    use_ycbcr = opts.get('use_ycbcr', True)

    if prescale != 1.0:
        if y_input.ndim == 3:
            y = np.stack([cv2.resize(y_input[:, :, ch], None, fx=prescale, fy=prescale,
                                     interpolation=cv2.INTER_LINEAR)
                          for ch in range(y_input.shape[2])], axis=-1)
        else:
            y = cv2.resize(y_input, None, fx=prescale, fy=prescale,
                           interpolation=cv2.INTER_LINEAR)
    else:
        y = y_input.copy()

    yorig = y.copy()

    if gamma_correct != 1.0:
        y = np.power(np.clip(y, 0, None), gamma_correct)

    if y.ndim == 3 and y.shape[2] == 3:

        ygray = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
    elif y.ndim == 3:
        ygray = y[:, :, 0]
    else:
        ygray = y.copy()

    dx = np.array([[-1.0, 1.0], [0.0, 0.0]])
    dy = np.array([[-1.0, 0.0], [1.0, 0.0]])

    minsize = max(5, 2 * int(np.floor((kernel_size - 1) / 16)) + 1)

    resize_step = np.sqrt(2.0)

    ksize = []
    tmp = minsize
    while tmp < kernel_size:
        ksize.append(tmp)
        tmp = int(np.ceil(tmp * resize_step))
        if tmp % 2 == 0:
            tmp += 1
    ksize.append(kernel_size)
    num_scales = len(ksize)

    ks_list = [None] * num_scales
    ls_list = [None] * num_scales

    error_flag = 0

    for s in range(num_scales):
        if s == 0:
            ks_list[s] = init_kernel(ksize[0])
            k1 = ksize[0]
        else:
            k1 = ksize[s]

            tmp_k = ks_list[s - 1].copy()
            tmp_k[tmp_k < 0] = 0.0
            sk = tmp_k.sum()
            if sk > 0:
                tmp_k = tmp_k / sk
            ks_list[s] = cv2.resize(tmp_k, (k1, k1), interpolation=cv2.INTER_LINEAR)
            ks_list[s][ks_list[s] < 0] = 0.0
            sk = ks_list[s].sum()
            if sk > 0:
                ks_list[s] = ks_list[s] / sk

        r = int(np.floor(ygray.shape[0] * k1 / kernel_size))
        c = int(np.floor(ygray.shape[1] * k1 / kernel_size))
        if s == num_scales - 1:
            r = ygray.shape[0]
            c = ygray.shape[1]

        ys = cv2.resize(ygray, (c, r), interpolation=cv2.INTER_LINEAR)

        yx = conv2_valid(ys, dx)
        yy = conv2_valid(ys, dy)

        c_grad = min(yx.shape[1], yy.shape[1])
        r_grad = min(yx.shape[0], yy.shape[0])
        yx = yx[:r_grad, :c_grad]
        yy = yy[:r_grad, :c_grad]

        g = np.concatenate([yx, yy], axis=1)

        if s == 0:
            ls_list[s] = g.copy()
        else:
            if error_flag != 0:
                ls_list[s] = g.copy()
            else:

                c1 = ls_list[s - 1].shape[1] // 2
                tmp1 = ls_list[s - 1][:, :c1]
                tmp2 = ls_list[s - 1][:, c1:]
                tmp1_up = cv2.resize(tmp1, (c_grad, r_grad), interpolation=cv2.INTER_LINEAR)
                tmp2_up = cv2.resize(tmp2, (c_grad, r_grad), interpolation=cv2.INTER_LINEAR)
                ls_list[s] = np.concatenate([tmp1_up, tmp2_up], axis=1)

        ls_list[s], ks_list[s], error_flag = ss_blind_deconv(
            g, ls_list[s], ks_list[s], min_lambda,
            delta, x_in_iter, x_out_iter, xk_iter, k_reg_wt
        )

        if error_flag < 0:
            ks_list[s] = np.zeros_like(ks_list[s])
            center = ks_list[s].shape[0] // 2
            ks_list[s][center, center] = 1.0

        c1 = ls_list[s].shape[1] // 2
        tmp1 = ls_list[s][:, :c1]
        tmp2 = ls_list[s][:, c1:]
        tmp1_shifted, tmp2_shifted, ks_list[s] = center_kernel_separate(
            tmp1, tmp2, ks_list[s]
        )
        ls_list[s] = np.concatenate([tmp1_shifted, tmp2_shifted], axis=1)

        if s == num_scales - 1:
            kernel = ks_list[s].copy()
            kernel[kernel < 0] = 0.0
            if k_thresh > 0:
                kernel[kernel < k_thresh * kernel.max()] = 0.0
            sk = kernel.sum()
            if sk > 0:
                kernel = kernel / sk

    bhs = kernel_size // 2

    def _deconv_channel(ch_img, alpha=nb_alpha):

        ypad = cv2.copyMakeBorder(ch_img, bhs, bhs, bhs, bhs,
                                  cv2.BORDER_REPLICATE)
        for _ in range(4):
            ypad = edgetaper(ypad, kernel)
        tmp = fast_deconv_bregman(ypad, kernel, nb_lambda, alpha)
        if bhs > 0:
            return tmp[bhs:-bhs, bhs:-bhs]
        return tmp

    if use_ycbcr and yorig.ndim == 3 and yorig.shape[2] == 3:

        _T = np.array([[65.481, 128.553, 24.966],
                       [-37.797, -74.203, 112.0],
                       [112.0, -93.786, -18.214]]) / 255.0
        _off = np.array([16.0, 128.0, 128.0]) / 255.0
        _T_inv = np.linalg.inv(_T)

        flat = yorig.reshape(-1, 3)
        ycbcr = (flat @ _T.T) + _off
        ycbcr = ycbcr.reshape(yorig.shape)

        y_channel = _deconv_channel(ycbcr[:, :, 0], alpha=1.0)
        ycbcr[:, :, 0] = y_channel

        flat2 = (ycbcr.reshape(-1, 3) - _off) @ _T_inv.T
        deblurred = flat2.reshape(yorig.shape)
    elif yorig.ndim == 2:
        deblurred = _deconv_channel(yorig)
    else:
        channels = [yorig[:, :, ch] for ch in range(yorig.shape[2])]
        deblurred = np.stack([_deconv_channel(ch) for ch in channels], axis=-1)

    return deblurred, kernel
