import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport exp

cnp.import_array()

from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.special import digamma, polygamma, gammaln
from scipy.fft import dstn, idstn

from .utils import psf2otf, valid_conv_by_fft

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _compute_Ad_core(double[:, :] Ad, double[:, :] xx, int m1, int m2, int M1, int M2):
    cdef int rows_len = M1 - m1 + 1
    cdef int cols_len = M2 - m2 + 1
    cdef int i, j, r, c
    cdef double s
    for i in range(m1):
        for j in range(m2):
            s = 0.0
            for r in range(rows_len):
                for c in range(cols_len):
                    s += xx[i + r, j + c]
            Ad[m1 - 1 - i, m2 - 1 - j] += s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _update_v(double[:, :] temp, double[:, :] v, double[:, :] s, double lam, int N2):
    cdef int M = temp.shape[0]
    cdef int N = temp.shape[1]
    cdef int i, r, c
    cdef double G, beta, val
    cdef double total_pixels = <double>(M * N)
    for i in range(N2):
        G = 0.0
        for r in range(M):
            for c in range(N):
                G += v[r, c]
        G /= total_pixels

        for r in range(M):
            for c in range(N):
                val = v[r, c]
                beta = lam * G / ((val + G)*(val + G) + 1e-30)
                val = temp[r, c] - beta
                if val < 0.0:
                    val = 0.0
                v[r, c] = val

    for r in range(M):
        for c in range(N):
            v[r, c] = s[r, c] * v[r, c]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _bilateral_filter_core(float[:, :, :] p_img_v, float[:, :, :] r_img_v, float[:, :] w_sum_v, double[:, :] sw_v, double ss, int h, int w, int d, int fr):
    cdef int r, c, ch, yy, xx
    cdef double w_s, w_f, f_dist, f_diff, w_t
    for r in range(h):
        for c in range(w):
            for yy in range(-fr, fr + 1):
                for xx in range(-fr, fr + 1):
                    w_s = sw_v[yy + fr, xx + fr]
                    f_dist = 0.0
                    for ch in range(d):
                        f_diff = p_img_v[r + fr, c + fr, ch] - p_img_v[r + fr + yy, c + fr + xx, ch]
                        f_dist += f_diff * f_diff

                    w_f = exp(-0.5 * f_dist / ss)
                    w_t = w_s * w_f

                    for ch in range(d):
                        r_img_v[r, c, ch] += p_img_v[r + fr + yy, c + fr + xx, ch] * w_t

                    w_sum_v[r, c] += w_t

    for r in range(h):
        for c in range(w):
            for ch in range(d):
                r_img_v[r, c, ch] /= w_sum_v[r, c]

def center_kernel_img_space(x, k, verbose=False):
    rows = np.arange(1, k.shape[0] + 1, dtype=np.float64)
    cols = np.arange(1, k.shape[1] + 1, dtype=np.float64)

    mu_y = np.sum(rows * np.sum(k, axis=1))
    mu_x = np.sum(cols * np.sum(k, axis=0))

    offset_x = int(np.round(k.shape[1] // 2 + 1 - mu_x))
    offset_y = int(np.round(k.shape[0] // 2 + 1 - mu_y))

    shift_kernel = np.zeros((abs(offset_y) * 2 + 1, abs(offset_x) * 2 + 1), dtype=np.float64)
    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    k_shift = convolve2d(k, shift_kernel, 'same')
    xshift = convolve2d(x, np.rot90(shift_kernel, 2), 'same')

    return xshift, k_shift, shift_kernel

def dirichlet_Adbc_fft(x_list, y_list, m1, m2, lambda_C=0, C=None, verbose=False):
    L = len(x_list)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Ad = np.zeros((m1, m2), dtype=np.float64)
    b = np.zeros((m1, m2), dtype=np.float64)

    X_fft =[]
    Xr_fft = []
    x_spatial =[]

    for t in range(L):
        xt = x_list[t]
        M1, M2 = xt.shape

        X_fft.append(fft2(xt))
        flipxt = np.rot90(xt, 2)
        Xr_fft.append(fft2(flipxt))
        x_spatial.append(xt)

        b -= 2.0 * valid_conv_by_fft(Xr_fft[t], y_list[t])

        xx = xt ** 2
        _compute_Ad_core(Ad, xx, m1, m2, M1, M2)

    if lambda_C > 0 and C is not None:
        CtC = convolve2d(C, C, 'same')
        Cd = CtC[CtC.shape[0] // 2, CtC.shape[1] // 2]
        Ad = Ad + lambda_C * Cd

    def xtAx_func(alpha):
        y_val = 0.0
        Xalpha =[None] * (L + 1)
        N = alpha.size
        for i in range(L):
            if N > 600:
                Xalpha[i] = valid_conv_by_fft(X_fft[i], alpha)
            else:
                Xalpha[i] = convolve2d(x_spatial[i], alpha, 'valid')
            y_val += np.sum(Xalpha[i] ** 2)
        if lambda_C > 0 and C is not None:
            Xalpha[L] = convolve2d(alpha, C, 'same')
            y_val += lambda_C * np.sum(Xalpha[L] ** 2)
        return y_val, Xalpha

    def Ax_func(Xalpha_list):
        y_val = np.zeros((m1, m2), dtype=np.float64)
        for i in range(L):
            y_val += valid_conv_by_fft(Xr_fft[i], Xalpha_list[i])
        if lambda_C > 0 and C is not None:
            y_val += lambda_C * convolve2d(Xalpha_list[L], C, 'same')
        return y_val

    return Ad, b, xtAx_func, Ax_func

def _dirichlet_cost_by_fft(alpha, lam, xtAx_func, Ad, b):
    _EPS = 1e-30
    Sa = np.sum(alpha)
    den1 = 2.0 * Sa * (Sa + 1.0) + _EPS
    atAa, Xalpha = xtAx_func(alpha)
    Adta = np.sum(Ad * alpha)
    bta = np.sum(b * alpha)
    temp = (alpha - 1.0) * (digamma(np.maximum(alpha, _EPS)) - digamma(max(Sa, _EPS)))
    f = (lam * (np.sum(temp) + gammaln(max(Sa, _EPS)) - np.sum(gammaln(np.maximum(alpha.ravel(), _EPS))))
         + (atAa + Adta) / den1 + bta / (2.0 * Sa + _EPS))
    return f, atAa, Adta, bta, Sa, den1, Xalpha

def kernel_estimation_filter_space_fft(k, x_list, y_list, opt, verbose=False):
    lam = opt['lambda']
    lambda_C = opt.get('lambda_C', 0)
    C = opt.get('Laplacian_filter', None)
    max_iter = opt['max_iter']
    ba = opt['back_alpha']
    bb = opt['back_beta']
    lb = opt['lower_bound']
    ng_min = opt['ng_min']
    cost_display = opt.get('cost_display', False)

    ks1, ks2 = k.shape
    alpha = opt['alpha0'].copy()

    Ad, b, xtAx_func, Ax_func = dirichlet_Adbc_fft(
        x_list, y_list, ks1, ks2, lambda_C, C, verbose=verbose)

    f0, atAa, Adta, bta, Sa, den1, Xalpha = _dirichlet_cost_by_fft(
        alpha, lam, xtAx_func, Ad, b)
    fx = [f0]
    costcalls = 1

    stepsize = Sa
    itr = 0

    while itr < max_iter:
        _EPS_g = 1e-30
        den2 = den1 ** 2 / (4.0 * Sa + 2.0 + _EPS_g)
        g = lam * (alpha - 1.0) * (polygamma(1, np.maximum(alpha, _EPS_g)) - polygamma(1, max(Sa, _EPS_g)))
        Aa = Ax_func(Xalpha)
        g = (g + (2.0 * Aa + Ad) / den1 + b / (2.0 * Sa + _EPS_g)
             - (atAa + Adta) / den2 - bta / (2.0 * Sa ** 2 + _EPS_g))

        d = -g
        tmax = min(Sa, stepsize * 1.2)
        t = tmax
        temp = np.maximum(alpha + t * d, lb)
        ftemp, atAa, Adta, bta, Sa, den1, Xalpha = _dirichlet_cost_by_fft(
            temp, lam, xtAx_func, Ad, b)
        costcalls += 1
        dg = np.sum((temp - alpha) * g)
        fx1 = fx[itr] + ba * dg

        while ftemp > fx1:
            t = bb * t
            temp = np.maximum(alpha + t * d, lb)
            ftemp, atAa, Adta, bta, Sa, den1, Xalpha = _dirichlet_cost_by_fft(
                temp, lam, xtAx_func, Ad, b)
            costcalls += 1
            dg = np.sum((temp - alpha) * g)
            fx1 = fx[itr] + ba * dg

        itr += 1
        stepsize = t
        alpha = temp
        rf = abs((ftemp - fx[itr - 1]) / ftemp) if ftemp != 0 else 0.0
        fx.append(ftemp)

        if t < 1e-2 or rf < ng_min:
            break

    return alpha, fx, stepsize

def nbid_ngm_ubc_admm(B, k, pars):

    cdef cnp.ndarray[cnp.float64_t, ndim=2] temp1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] temp2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v2
    cdef cnp.ndarray[cnp.float64_t, ndim=2] s1
    cdef cnp.ndarray[cnp.float64_t, ndim=2] s2

    lambda1 = pars['lambda1']
    lambda_min = pars.get('lambda_min', lambda1 * 5)
    lambda_max = pars.get('lambda_max', 1.0)
    cost_display = pars.get('cost_display', 0)

    IF_val = pars.get('IF', np.sqrt(2))

    N2 = pars.get('N2', 2)
    N1 = pars.get('N1', 20)
    lambda_u = pars.get('lambda_u', 0.1)
    xv_iter = pars.get('xv_iter', 1)

    lambda_v = lambda_min

    dx = np.array([[1.0, -1.0]])
    dy = dx.T

    m, n = B.shape
    hks1 = k.shape[0] // 2
    hks2 = k.shape[1] // 2
    M = m + 2 * hks1
    N = n + 2 * hks2

    K = psf2otf(k, (M, N))
    Kt = np.conj(K)

    Dx = psf2otf(dx, (M, N))
    Dy = psf2otf(dy, (M, N))
    Dxt = np.conj(Dx)
    Dyt = np.conj(Dy)
    DtD = Dxt * Dx + Dyt * Dy
    KtK = Kt * K

    x = pars['x0'].copy()
    if x.shape == B.shape:
        x = np.pad(x, ((hks1, hks1), (hks2, hks2)), mode='edge')

    MtB = np.pad(B, ((hks1, hks1), (hks2, hks2)), mode='constant', constant_values=0)
    MtM = np.pad(np.ones((m, n), dtype=np.float64),
                 ((hks1, hks1), (hks2, hks2)), mode='constant', constant_values=0)

    x1 = np.concatenate([np.diff(x, axis=1), x[:, 0:1] - x[:, -1:]], axis=1)
    x2 = np.concatenate([np.diff(x, axis=0), x[0:1, :] - x[-1:, :]], axis=0)

    u = np.pad(B, ((hks1, hks1), (hks2, hks2)), mode='edge')
    du = np.zeros_like(u)

    X = fft2(x)
    i = 1

    while i <= N1:
        Ax = np.real(ifft2(X * K))
        u = (MtB + lambda_u * (Ax + du)) / (MtM + lambda_u)

        du = du + Ax - u
        Ktu = fft2(u - du) * Kt
        invA = 1.0 / (KtK + lambda_v / lambda_u * DtD + 1e-30)
        lam = lambda1 / lambda_v

        for _xv in range(xv_iter):

            temp1 = np.abs(x1)
            temp2 = np.abs(x2)
            v1 = temp1.copy()
            v2 = temp2.copy()
            s1 = np.sign(x1)
            s2 = np.sign(x2)

            _update_v(temp1, v1, s1, lam, N2)
            _update_v(temp2, v2, s2, lam, N2)

        temp1 = -np.concatenate([v1[:, 0:1] - v1[:, -1:], np.diff(v1, axis=1)], axis=1)
        temp2 = -np.concatenate([v2[0:1, :] - v2[-1:, :], np.diff(v2, axis=0)], axis=0)

        X = Ktu + lambda_v / lambda_u * fft2(temp1 + temp2)
        X = invA * X
        x = np.real(ifft2(X))

        x1 = np.concatenate([np.diff(x, axis=1), x[:, 0:1] - x[:, -1:]], axis=1)
        x2 = np.concatenate([np.diff(x, axis=0), x[0:1, :] - x[-1:, :]], axis=0)

        lambda_v = min(lambda_v * IF_val, lambda_max)
        i += 1

    if hks1 == 0 and hks2 == 0:
        x_fov = x.copy()
    elif hks1 == 0:
        x_fov = x[:, hks2:hks2 + n].copy()
    elif hks2 == 0:
        x_fov = x[hks1:hks1 + m, :].copy()
    else:
        x_fov = x[hks1:hks1 + m, hks2:hks2 + n].copy()

    np.nan_to_num(x_fov, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    return x_fov, x

def ss_ngm_dirichlet_ubc_img(y, x, k, alpha0, pars, blind_denoise_fn=None, verbose=False):
    m, n = y.shape
    k1, k2 = k.shape
    khs1 = k1 // 2
    khs2 = k2 // 2

    xk_iter = pars['xk_iter']
    img_pars = pars['img_pars'].copy()
    img_pars['x0'] = x.copy()

    if khs1 > 0 and khs2 > 0:
        y_crop = y[khs1:-khs1, khs2:-khs2]
    elif khs1 > 0:
        y_crop = y[khs1:-khs1, :]
    elif khs2 > 0:
        y_crop = y[:, khs2:-khs2]
    else:
        y_crop = y

    y2 =[np.diff(y_crop, axis=0), np.diff(y_crop, axis=1)]

    alpha = alpha0.copy()
    ker_opts = pars['kernel_pars'].copy()
    lambda1 = img_pars['lambda1']
    lambda_min = img_pars['lambda_min']

    delta_lambda = 0.00005 if lambda1 < 0.0005 else 0.0
    k_old = None

    for i in range(xk_iter):
        img_pars['lambda1'] = lambda1 + delta_lambda * max(6 - (i + 1), 0)
        img_pars['lambda_min'] = lambda_min / max(lambda1, 1e-30) * img_pars['lambda1']

        x_fov, x_full = nbid_ngm_ubc_admm(y, k, img_pars)
        img_pars['x0'] = x_full

        x_for_grad = blind_denoise_fn(x_fov) if blind_denoise_fn is not None else x_fov
        x1 =[np.diff(x_for_grad, axis=0), np.diff(x_for_grad, axis=1)]

        ker_opts['alpha0'] = alpha.copy()
        alpha, fcost, _ = kernel_estimation_filter_space_fft(k, x1, y2, ker_opts, verbose=verbose)
        alpha0 = alpha.reshape(k1, k2)

        if ker_opts.get('mode', 0):
            denom_mode = np.sum(alpha) - k1 * k2
            if abs(denom_mode) < 1e-30:
                denom_mode = 1e-30
            k = (alpha0 - 1.0) / denom_mode
        else:
            k = alpha0 / max(np.sum(alpha), 1e-30)

        if np.any(np.isnan(k)):
            if k_old is not None:
                k = k_old.copy()
            else:
                k = np.ones((k1, k2), dtype=np.float64) / (k1 * k2)

        if i >= 4 and k_old is not None:
            r_k = np.max(np.abs(k - k_old)) / 1.0
            if len(fcost) <= 2 or r_k <= pars.get('k_tol', 1e-4):
                break

        k_old = k.copy()

    return x_fov, k, alpha0

def firls_deb_ubc(y, h, opt, verbose=False):
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = np.pad(y, ((hks1, hks1), (hks2, hks2)), mode='edge')

    dxf  = np.array([[ 0, 0, 0],[ 0, 1,-1],[ 0, 0, 0]], dtype=np.float64)
    dyf  = np.array([[ 0, 0, 0], [ 0, 1, 0],[ 0,-1, 0]], dtype=np.float64)
    dyyf = np.array([[ 0,-1, 0], [ 0, 2, 0],[ 0,-1, 0]], dtype=np.float64)
    dxxf = np.array([[ 0, 0, 0], [-1, 2,-1], [ 0, 0, 0]], dtype=np.float64)
    dxyf = np.array([[ 0, 0, 0], [ 0, 1,-1], [ 0,-1, 1]], dtype=np.float64)

    dxfr  = np.rot90(dxf, 2)
    dyfr  = np.rot90(dyf, 2)
    dxxfr = np.rot90(dxxf, 2)
    dyyfr = np.rot90(dyyf, 2)
    dxyfr = np.rot90(dxyf, 2)

    H   = psf2otf(h, (n1, n2))
    Ht  = np.conj(H)
    Hx  = psf2otf(dxf, (n1, n2))
    Hy  = psf2otf(dyf, (n1, n2))
    Hxx = psf2otf(dxxf, (n1, n2))
    Hyy = psf2otf(dyyf, (n1, n2))
    Hxy = psf2otf(dxyf, (n1, n2))

    HH   = H * Ht
    HHx  = Hx * np.conj(Hx)
    HHy  = Hy * np.conj(Hy)
    HHxx = Hxx * np.conj(Hxx)
    HHyy = Hyy * np.conj(Hyy)
    HHxy = Hxy * np.conj(Hxy)

    RR = HHx + HHy + HHxx + HHyy + HHxy

    lam = opt['lambda']
    w0 = 0.25
    alpha_p = opt.get('alpha', 2.0 / 3.0)
    beta_a = opt.get('beta_a', lam * alpha_p * (20.0 / 255.0) ** (alpha_p - 2))
    lambda_u = opt.get('lambda_u', min(0.1, 5000 * lam))
    N2 = opt.get('inner_iter', 4)
    N1 = opt.get('out_iter', 5)
    epsilon = opt.get('epsilon', 0.01)

    c = alpha_p * lam
    beta = alpha_p * lam / epsilon ** (2 - alpha_p)

    xpad = np.pad(x, ((1, 1), (1, 1)), mode='wrap')
    dx_  = convolve2d(xpad, dxf, 'valid')
    dy_  = convolve2d(xpad, dyf, 'valid')
    dxx_ = convolve2d(xpad, dxxf, 'valid')
    dyy_ = convolve2d(xpad, dyyf, 'valid')
    dxy_ = convolve2d(xpad, dxyf, 'valid')

    adx  = np.abs(dx_)
    ady  = np.abs(dy_)
    adxx = np.abs(dxx_)
    adyy = np.abs(dyy_)
    adxy = np.abs(dxy_)

    du   = np.zeros((n1, n2), dtype=np.float64)
    dvx  = np.zeros_like(du)
    dvy  = np.zeros_like(du)
    dvxx = np.zeros_like(du)
    dvyy = np.zeros_like(du)
    dvxy = np.zeros_like(du)

    X_ = fft2(x)
    Ax_ = np.real(ifft2(H * X_))
    invA = HH + beta_a / lambda_u * RR + 1e-30

    opt_out = dict(opt)

    outer = 0
    while outer < N1:
        outer += 1

        _eps_d = 1e-10
        Wx  = np.minimum(beta, c * np.maximum(adx,  _eps_d) ** (alpha_p - 2))
        Wy  = np.minimum(beta, c * np.maximum(ady,  _eps_d) ** (alpha_p - 2))
        Wxx = np.minimum(beta, c * np.maximum(adxx, _eps_d) ** (alpha_p - 2)) * w0
        Wyy = np.minimum(beta, c * np.maximum(adyy, _eps_d) ** (alpha_p - 2)) * w0
        Wxy = np.minimum(beta, c * np.maximum(adxy, _eps_d) ** (alpha_p - 2)) * w0

        inner = 0
        while inner < N2:
            inner += 1

            u = Ax_ + du
            u[hks1:hks1 + M1, hks2:hks2 + M2] = (
                (y + lambda_u * u[hks1:hks1 + M1, hks2:hks2 + M2])
                / (1.0 + lambda_u)
            )

            vx  = beta_a * (dx_  + dvx)  / (Wx  + beta_a)
            vy  = beta_a * (dy_  + dvy)  / (Wy  + beta_a)
            vxx = beta_a * (dxx_ + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy_ + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy_ + dvxy) / (Wxy + beta_a)

            du   = du   - u   + Ax_
            dvx  = dvx  - vx  + dx_
            dvy  = dvy  - vy  + dy_
            dvxx = dvxx - vxx + dxx_
            dvyy = dvyy - vyy + dyy_
            dvxy = dvxy - vxy + dxy_

            Y_ = fft2(u - du) * Ht

            tempx  = convolve2d(np.pad(vx  - dvx,  ((1,1),(1,1)), mode='wrap'), dxfr,  'valid')
            tempy  = convolve2d(np.pad(vy  - dvy,  ((1,1),(1,1)), mode='wrap'), dyfr,  'valid')
            tempxx = convolve2d(np.pad(vxx - dvxx, ((1,1),(1,1)), mode='wrap'), dxxfr, 'valid')
            tempyy = convolve2d(np.pad(vyy - dvyy, ((1,1),(1,1)), mode='wrap'), dyyfr, 'valid')
            tempxy = convolve2d(np.pad(vxy - dvxy, ((1,1),(1,1)), mode='wrap'), dxyfr, 'valid')

            X_ = Y_ + beta_a / lambda_u * fft2(tempx + tempy + tempxx + tempyy + tempxy)
            X_ = X_ / invA
            Ax_ = np.real(ifft2(H * X_))
            x = np.real(ifft2(X_))

            xpad = np.pad(x, ((1, 1), (1, 1)), mode='wrap')
            dx_  = convolve2d(xpad, dxf, 'valid')
            dy_  = convolve2d(xpad, dyf, 'valid')
            dxx_ = convolve2d(xpad, dxxf, 'valid')
            dyy_ = convolve2d(xpad, dyyf, 'valid')
            dxy_ = convolve2d(xpad, dxyf, 'valid')
            adx  = np.abs(dx_)
            ady  = np.abs(dy_)
            adxx = np.abs(dxx_)
            adyy = np.abs(dyy_)
            adxy = np.abs(dxy_)

    x_fov = x[hks1:hks1 + M1, hks2:hks2 + M2]
    return x_fov, x, opt_out

_OPT_FFT_LUT = None

def _build_opt_fft_lut(max_n=4096):
    efficient = set()
    p2 = 1
    while p2 <= max_n:
        p3 = 1
        while p2 * p3 <= max_n:
            p5 = 1
            while p2 * p3 * p5 <= max_n:
                p7 = 1
                while p2 * p3 * p5 * p7 <= max_n:
                    efficient.add(p2 * p3 * p5 * p7)
                    p7 *= 7
                p5 *= 5
            p3 *= 3
        p2 *= 2
    eff_sorted = sorted(efficient)
    lut = np.zeros(max_n + 1, dtype=np.int64)
    idx = 0
    for n_ in range(1, max_n + 1):
        while idx < len(eff_sorted) and eff_sorted[idx] < n_:
            idx += 1
        lut[n_] = eff_sorted[idx] if idx < len(eff_sorted) else n_
    return lut

def opt_fft_size(n) -> np.ndarray:
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
            m.flat[i] = int(nn)
    if scalar_input:
        return int(m.flat[0])
    return m

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    H, W = boundary_image.shape
    bi = boundary_image.copy()
    bi[1:-1, 1:-1] = 0.0

    f_bp = np.zeros((H, W), dtype=np.float64)
    f_bp[1:H-1, 1:W-1] = (
        -4.0 * bi[1:H-1, 1:W-1]
        + bi[1:H-1, 2:W] + bi[1:H-1, 0:W-2]
        + bi[0:H-2, 1:W-1] + bi[2:H, 1:W-1]
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

    result = bi.copy()
    result[1:H-1, 1:W-1] = img_tt
    return result

def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
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
        r_A[alpha:alpha + H_w, 0] = ((1 - a) * r_A[alpha - 1, 0] + a * r_A[-alpha, 0])
        r_A[alpha:alpha + H_w, -1] = ((1 - a) * r_A[alpha - 1, -1] + a * r_A[-alpha, -1])

        A2 = _solve_min_laplacian(r_A[alpha - 1: alpha + H_w + 1, :])

        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]
        if W_w > 1:
            b = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            b = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = ((1 - b) * r_B[0, alpha - 1] + b * r_B[0, -alpha])
        r_B[-1, alpha:alpha + W_w] = ((1 - b) * r_B[-1, alpha - 1] + b * r_B[-1, -alpha])

        B2 = _solve_min_laplacian(r_B[:, alpha - 1: alpha + W_w + 1])

        ret[:H, :W, ch] = HG
        ret[H:, :W, ch] = A2[1:-1, :]
        ret[:H, W:, ch] = B2[:, 1:-1]

        if H_w > 0 and W_w > 0:
            r_C = np.zeros((H_w + 2, W_w + 2), dtype=np.float64)
            r_C[0, :-1] = ret[H - 1, W - 1:, ch]
            r_C[0, -1]  = ret[H - 1, 0, ch]
            r_C[-1, :-1] = ret[0, W - 1:, ch]
            r_C[-1, -1]  = ret[0, 0, ch]
            r_C[:-1, 0]  = ret[H - 1:, W - 1, ch]
            r_C[-1, 0]   = ret[0, W - 1, ch]
            r_C[:-1, -1] = ret[H - 1:, 0, ch]
            r_C[-1, -1]  = ret[0, 0, ch]
            C2 = _solve_min_laplacian(r_C)
            ret[H:, W:, ch] = C2[1:-1, 1:-1]

    if Ch == 1:
        return ret[:, :, 0]
    return ret

def _computeDenominator(B, k):
    m, n = B.shape
    otf_k = psf2otf(k, (m, n))
    Nomin1 = np.conj(otf_k) * fft2(B)
    Denom1 = np.abs(otf_k) ** 2

    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    Denom2 = (np.abs(psf2otf(dx, (m, n))) ** 2 + np.abs(psf2otf(dy, (m, n))) ** 2)
    return Nomin1, Denom1, Denom2

def deblurring_adm_aniso(B, k, lambda_tv, alpha):
    beta = 1.0 / lambda_tv
    beta_min = 0.001
    m, n = B.shape
    I = B.copy()
    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, axis=0), I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
        Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1], -np.diff(Wx, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :], -np.diff(Wy, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, axis=1), I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, axis=0), I[0:1, :] - I[-1:, :]], axis=0)
        beta = beta / 2.0

    return I

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    H_orig, W_orig = Im.shape[0], Im.shape[1]
    target_size = opt_fft_size(np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1)
    Im_w = wrap_boundary_liu(Im, tuple(target_size))

    if Im_w.ndim == 2:
        Im_w = Im_w[:, :, np.newaxis]
    N, M, D = Im_w.shape

    S = Im_w.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1],[-1]], dtype=np.float64)

    otfFx = psf2otf(fx, (N, M))
    otfFy = psf2otf(fy, (N, M))
    KER = psf2otf(kernel, (N, M))
    Den_KER = np.abs(KER) ** 2
    Denormin2 = np.abs(otfFx) ** 2 + np.abs(otfFy) ** 2
    Denormin2 = np.tile(Denormin2[:, :, np.newaxis], (1, 1, D))
    KER = np.tile(KER[:, :, np.newaxis], (1, 1, D))
    Den_KER = np.tile(Den_KER[:, :, np.newaxis], (1, 1, D))

    Normin1 = np.conj(KER) * fft2(S, axes=(0, 1))

    beta_val = 2 * lambda_grad
    while beta_val < betamax:
        Denormin = Den_KER + beta_val * Denormin2

        h = np.concatenate([np.diff(S, axis=1), S[:, 0:1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, axis=0), S[0:1, :, :] - S[-1:, :, :]], axis=0)

        grad_sq = np.sum(h ** 2 + v ** 2, axis=2)
        t = grad_sq < lambda_grad / beta_val
        t3 = np.tile(t[:, :, np.newaxis], (1, 1, D))
        h[t3] = 0
        v[t3] = 0

        Normin2 = np.concatenate([h[:, -1:, :] - h[:, 0:1, :], -np.diff(h, axis=1)], axis=1)
        Normin2 += np.concatenate([v[-1:, :, :] - v[0:1, :, :], -np.diff(v, axis=0)], axis=0)

        FS = (Normin1 + beta_val * fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))
        beta_val *= kappa

    S = S[:H_orig, :W_orig, :]
    if D == 1:
        S = S[:, :, 0]
    return S

def _fspecial_gaussian(size, sigma):
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    h = np.outer(g, g)
    return h / h.sum()

def bilateral_filter(img, sigma_s, sigma):
    was_2d = img.ndim == 2
    if was_2d:
        img = img[:, :, np.newaxis]

    h = img.shape[0]
    w = img.shape[1]
    d = img.shape[2]

    img = img.astype(np.float32)
    fr = int(np.ceil(sigma_s * 3))

    p_img = np.pad(img, ((fr, fr), (fr, fr), (0, 0)), mode='edge')
    r_img = np.zeros((h, w, d), dtype=np.float32)
    w_sum = np.zeros((h, w), dtype=np.float32)

    sw = _fspecial_gaussian(2 * fr + 1, sigma_s)
    ss = sigma * np.sqrt(d)
    ss = ss * ss

    _bilateral_filter_core(p_img, r_img, w_sum, sw, ss, h, w, d, fr)

    if was_2d:
        return r_img[:, :, 0]
    return r_img

def ringing_artifacts_removal(y, kernel, lambda_tv=1e-3, lambda_l0=2e-3, weight_ring=1.0):
    H, W = y.shape[:2]
    target_size = opt_fft_size(np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
    y_pad = wrap_boundary_liu(y, tuple(target_size))

    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv, 1)
    Latent_tv = Latent_tv[:H, :W]

    if weight_ring == 0:
        return Latent_tv

    Latent_l0 = L0Restoration(y, kernel, lambda_l0, 2)

    diff_img = Latent_tv - Latent_l0
    bf_diff = bilateral_filter(diff_img, 3, 0.1)
    result = Latent_tv - weight_ring * bf_diff
    return result
