import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.special import digamma, polygamma, gammaln

from .utils import psf2otf, valid_conv_by_fft

def center_kernel_img_space(x, k):

    rows = np.arange(1, k.shape[0] + 1, dtype=np.float64)
    cols = np.arange(1, k.shape[1] + 1, dtype=np.float64)

    mu_y = np.sum(rows * np.sum(k, axis=1))

    mu_x = np.sum(cols * np.sum(k, axis=0))

    offset_x = int(np.round(k.shape[1] // 2 + 1 - mu_x))
    offset_y = int(np.round(k.shape[0] // 2 + 1 - mu_y))

    print(f'CenterKernel: weightedMean[{mu_x - 1:.6f} {mu_y - 1:.6f}] '
          f'offset[{offset_x} {offset_y}]')

    shift_kernel = np.zeros((abs(offset_y) * 2 + 1, abs(offset_x) * 2 + 1),
                            dtype=np.float64)

    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    k = convolve2d(k, shift_kernel, 'same')
    xshift = convolve2d(x, np.rot90(shift_kernel, 2), 'same')

    return xshift, k, shift_kernel

def dirichlet_Adbc_fft(x_list, y_list, m1, m2, lambda_C=0, C=None):

    L = len(x_list)
    Ad = np.zeros((m1, m2), dtype=np.float64)
    b = np.zeros((m1, m2), dtype=np.float64)

    X_fft = []
    Xr_fft = []
    x_spatial = []

    for t in range(L):
        xt = x_list[t]
        M1, M2 = xt.shape

        X_fft.append(fft2(xt))
        flipxt = np.rot90(xt, 2)
        Xr_fft.append(fft2(flipxt))
        x_spatial.append(xt)

        b -= 2.0 * valid_conv_by_fft(Xr_fft[t], y_list[t])

        xx = xt ** 2
        rows_len = M1 - m1 + 1
        cols_len = M2 - m2 + 1
        for i in range(m1):
            for j in range(m2):

                Ad[m1 - 1 - i, m2 - 1 - j] += np.sum(xx[i:i + rows_len, j:j + cols_len])

    print(f'Ad_min={Ad.min():.6f},Ad_max={Ad.max():.6f}')

    if lambda_C > 0 and C is not None:
        CtC = convolve2d(C, C, 'same')
        Cd = CtC[CtC.shape[0] // 2, CtC.shape[1] // 2]
        Ad = Ad + lambda_C * Cd

    def xtAx_func(alpha):
        y_val = 0.0
        Xalpha = [None] * (L + 1)
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

    Sa = np.sum(alpha)
    den1 = 2.0 * Sa * (Sa + 1.0)
    atAa, Xalpha = xtAx_func(alpha)
    Adta = np.sum(Ad * alpha)
    bta = np.sum(b * alpha)
    temp = (alpha - 1.0) * (digamma(alpha) - digamma(Sa))
    f = (lam * (np.sum(temp) + gammaln(Sa) - np.sum(gammaln(alpha.ravel())))
         + (atAa + Adta) / den1 + bta / (2.0 * Sa))
    return f, atAa, Adta, bta, Sa, den1, Xalpha

def kernel_estimation_filter_space_fft(k, x_list, y_list, opt):

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
        x_list, y_list, ks1, ks2, lambda_C, C)

    f0, atAa, Adta, bta, Sa, den1, Xalpha = _dirichlet_cost_by_fft(
        alpha, lam, xtAx_func, Ad, b)
    fx = [f0]
    costcalls = 1
    if cost_display:
        print(f'iteration=0,cost={fx[0]:.6f}')

    stepsize = Sa
    itr = 0

    while itr < max_iter:

        den2 = den1 ** 2 / (4.0 * Sa + 2.0)
        g = lam * (alpha - 1.0) * (polygamma(1, alpha) - polygamma(1, Sa))
        Aa = Ax_func(Xalpha)
        g = (g + (2.0 * Aa + Ad) / den1 + b / (2.0 * Sa)
             - (atAa + Adta) / den2 - bta / (2.0 * Sa ** 2))

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

        if cost_display:
            print(f'iteration={itr},costcalls={costcalls},cost={fx[itr]:.6f},'
                  f'rf={rf:.6f},step_size={t:.6f}')

        if t < 1e-2 or rf < ng_min:
            break

    print(f'DIter={itr},costcalls={costcalls},cost={fx[0]:.4f},'
          f'cost={fx[-1]:.4f},rf={rf:.6f},Sa={Sa:.0f}')

    return alpha, fx, stepsize

def nbid_ngm_ubc_admm(B, k, pars):

    lambda1 = pars['lambda1']
    lambda_min = pars.get('lambda_min', lambda1 * 5)
    lambda_max = pars.get('lambda_max', 1.0)
    cost_display = pars.get('cost_display', 0)
    IF = pars.get('IF', np.sqrt(2))
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

        invA = 1.0 / (KtK + lambda_v / lambda_u * DtD)
        lam = lambda1 / lambda_v

        for _xv in range(xv_iter):
            temp1 = np.abs(x1)
            temp2 = np.abs(x2)
            v1 = temp1.copy()
            v2 = temp2.copy()
            s1 = np.sign(x1)
            s2 = np.sign(x2)
            for _t in range(N2):
                G1 = np.mean(v1)
                G2 = np.mean(v2)
                beta1 = lam * G1 / (v1 + G1) ** 2
                beta2 = lam * G2 / (v2 + G2) ** 2
                v1 = np.maximum(temp1 - beta1, 0.0)
                v2 = np.maximum(temp2 - beta2, 0.0)
            v1 = s1 * v1
            v2 = s2 * v2

        temp1 = -np.concatenate([v1[:, 0:1] - v1[:, -1:],
                                 np.diff(v1, axis=1)], axis=1)

        temp2 = -np.concatenate([v2[0:1, :] - v2[-1:, :],
                                 np.diff(v2, axis=0)], axis=0)
        X = Ktu + lambda_v / lambda_u * fft2(temp1 + temp2)
        X = invA * X
        x = np.real(ifft2(X))

        x1 = np.concatenate([np.diff(x, axis=1), x[:, 0:1] - x[:, -1:]], axis=1)
        x2 = np.concatenate([np.diff(x, axis=0), x[0:1, :] - x[-1:, :]], axis=0)

        if cost_display:
            t1 = np.abs(x1.ravel())
            t2 = np.abs(x2.ravel())
            G1 = np.mean(t1)
            G2 = np.mean(t2)
            pcost_i = lambda1 * (np.sum(t1 / (t1 + G1)) + np.sum(t2 / (t2 + G2)))
            Ax_fov = Ax[hks1:hks1 + m, hks2:hks2 + n]
            fcost_i = 0.5 * np.sum((Ax_fov - B) ** 2) + pcost_i
            print(f'Outer iteration {i}: fcost={fcost_i:.6f} pcost={pcost_i:.6f}')

        lambda_v = min(lambda_v * IF, lambda_max)
        i += 1

    if hks1 == 0 and hks2 == 0:
        x_fov = x.copy()
    elif hks1 == 0:
        x_fov = x[:, hks2:hks2 + n].copy()
    elif hks2 == 0:
        x_fov = x[hks1:hks1 + m, :].copy()
    else:
        x_fov = x[hks1:hks1 + m, hks2:hks2 + n].copy()

    return x_fov, x

def ss_ngm_dirichlet_ubc_img(y, x, k, alpha0, pars):

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
    y2 = [np.diff(y_crop, axis=0), np.diff(y_crop, axis=1)]

    alpha = alpha0.copy()
    ker_opts = pars['kernel_pars'].copy()
    lambda1 = img_pars['lambda1']
    lambda_min = img_pars['lambda_min']

    if lambda1 < 0.0005:
        delta_lambda = 0.00005
    else:
        delta_lambda = 0.0

    k_old = None
    for i in range(xk_iter):

        img_pars['lambda1'] = lambda1 + delta_lambda * max(6 - (i + 1), 0)
        img_pars['lambda_min'] = lambda_min / lambda1 * img_pars['lambda1']

        x_fov, x_full = nbid_ngm_ubc_admm(y, k, img_pars)
        img_pars['x0'] = x_full

        x1 = [np.diff(x_fov, axis=0), np.diff(x_fov, axis=1)]

        ker_opts['alpha0'] = alpha.copy()
        alpha, fcost, _ = kernel_estimation_filter_space_fft(k, x1, y2, ker_opts)
        alpha0 = alpha.reshape(k1, k2)

        if ker_opts.get('mode', 0):

            k = (alpha0 - 1.0) / (np.sum(alpha) - k1 * k2)
        else:

            k = alpha0 / np.sum(alpha)

        print(f'Iteration={i + 1}')

        if i >= 4 and k_old is not None:
            r_k = np.max(np.abs(k - k_old)) / 1.0
            if len(fcost) <= 2 or r_k <= pars.get('k_tol', 1e-4):
                break

        k_old = k.copy()

    return x_fov, k, alpha0

def firls_deb_ubc(y, h, opt):

    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = np.pad(y, ((hks1, hks1), (hks2, hks2)), mode='edge')

    dxf  = np.array([[ 0, 0, 0], [ 0, 1,-1], [ 0, 0, 0]], dtype=np.float64)
    dyf  = np.array([[ 0, 0, 0], [ 0, 1, 0], [ 0,-1, 0]], dtype=np.float64)
    dyyf = np.array([[ 0,-1, 0], [ 0, 2, 0], [ 0,-1, 0]], dtype=np.float64)
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
    isnr_display = opt.get('isnr_display', 0)
    cost_display_flag = opt.get('cost_display', 0)
    N2 = opt.get('inner_iter', 4)
    N1 = opt.get('out_iter', 5)
    epsilon = opt.get('epsilon', 0.01)

    if isnr_display == 1:
        I_gt = opt['groundtruth']

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
    invA = HH + beta_a / lambda_u * RR

    opt_out = dict(opt)

    outer = 0
    while outer < N1:
        outer += 1

        eps_w = 1e-12
        exp_w = alpha_p - 2.0
        with np.errstate(divide='ignore', invalid='ignore'):
            Wx  = np.minimum(beta, c * np.maximum(adx,  eps_w) ** exp_w)
            Wy  = np.minimum(beta, c * np.maximum(ady,  eps_w) ** exp_w)
            Wxx = np.minimum(beta, c * np.maximum(adxx, eps_w) ** exp_w) * w0
            Wyy = np.minimum(beta, c * np.maximum(adyy, eps_w) ** exp_w) * w0
            Wxy = np.minimum(beta, c * np.maximum(adxy, eps_w) ** exp_w) * w0

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

        if cost_display_flag == 1:
            r = Ax_[hks1:hks1 + M1, hks2:hks2 + M2] - y
            cost1 = 0.5 * np.sum(r ** 2)
            cost2 = lam * np.sum(
                adx ** alpha_p + ady ** alpha_p
                + adxx ** alpha_p + adyy ** alpha_p + adxy ** alpha_p)
            cost3 = cost1 + cost2
            opt_out.setdefault('cost1', []).append(cost1)
            opt_out.setdefault('cost2', []).append(cost2)
            opt_out.setdefault('cost3', []).append(cost3)
            msg = f'Outiter={outer},costf={cost3:.6f},'
        else:
            msg = ''

        if isnr_display == 1:
            x_fov_tmp = x[hks1:hks1 + M1, hks2:hks2 + M2]
            isnr_val = 20 * np.log10(
                np.linalg.norm(y - I_gt, 'fro')
                / np.linalg.norm(x_fov_tmp - I_gt, 'fro'))
            opt_out.setdefault('isnr', []).append(isnr_val)
            print(f'{msg}isnr={isnr_val:.6f},beta={beta:.6f}')
        else:
            print(f'{msg}beta={beta:.6f}')

    x_fov = x[hks1:hks1 + M1, hks2:hks2 + M2]
    return x_fov, x, opt_out
