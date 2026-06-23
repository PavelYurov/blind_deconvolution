import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d

from .utils import (
    psf2otf,
    imresize,
    edgetaper,
    center_kernel_separate,
    solve_image_bregman,
)

def optimizex_cry(x1, x2, k, y1, y2, lambda_, imax, jmax, t):

    n1 = np.linalg.norm(x1.ravel())
    n2 = np.linalg.norm(x2.ravel())
    if n1 > 0:
        x1 = x1 * 6.0 / n1
    if n2 > 0:
        x2 = x2 * 6.0 / n2

    r1 = convolve2d(x1, k, 'same') - y1
    r2 = convolve2d(x2, k, 'same') - y2
    costLS0 = lambda_ / 2.0 * (np.sum(r1 ** 2) + np.sum(r2 ** 2))

    n1 = np.linalg.norm(x1.ravel())
    n2 = np.linalg.norm(x2.ravel())
    costR0 = (np.linalg.norm(x1.ravel(), 1) / max(n1, 1e-30)
              + np.linalg.norm(x2.ravel(), 1) / max(n2, 1e-30))
    cost0 = costLS0 + costR0

    tp = 1e-4
    k180 = np.rot90(k, 2)

    while t > tp:
        x10, x20 = x1.copy(), x2.copy()

        for _i in range(imax):
            l21 = np.linalg.norm(x1.ravel())
            l22 = np.linalg.norm(x2.ravel())

            for _j in range(jmax):

                grad1 = lambda_ * convolve2d(
                    convolve2d(x1, k, 'same') - y1, k180, 'same')
                grad2 = lambda_ * convolve2d(
                    convolve2d(x2, k, 'same') - y2, k180, 'same')

                tmp1 = x1 - t * l21 * grad1
                tmp2 = x2 - t * l22 * grad2
                x1 = np.maximum(0, np.abs(tmp1) - t) * np.sign(tmp1)
                x2 = np.maximum(0, np.abs(tmp2) - t) * np.sign(tmp2)

        r1 = convolve2d(x1, k, 'same') - y1
        r2 = convolve2d(x2, k, 'same') - y2
        costLS1 = lambda_ / 2.0 * (np.sum(r1 ** 2) + np.sum(r2 ** 2))

        n1 = np.linalg.norm(x1.ravel())
        n2 = np.linalg.norm(x2.ravel())
        costR1 = (np.linalg.norm(x1.ravel(), 1) / max(n1, 1e-30)
                  + np.linalg.norm(x2.ravel(), 1) / max(n2, 1e-30))
        cost1 = costLS1 + costR1

        if cost1 > 2.0 * cost0 or np.isnan(cost1):
            t = t / 2.0
            x1, x2 = x10, x20
        else:
            break

    return x1, x2

def optimizek(x1, x2, k, y1, y2, imax, mu):

    ep = 1e-3
    x1_180 = np.rot90(x1, 2)
    x2_180 = np.rot90(x2, 2)
    k0 = k.copy()

    Ak = (2.0 * convolve2d(x1_180, convolve2d(x1, k, 'valid'), 'valid')
          + 2.0 * convolve2d(x2_180, convolve2d(x2, k, 'valid'), 'valid')
          + 2.0 * mu * k)

    b = (2.0 * convolve2d(x1_180, y1, 'valid')
         + 2.0 * convolve2d(x2_180, y2, 'valid')
         + 2.0 * mu * k0)

    r = b - Ak
    d = r.copy()
    e1 = np.sum(r ** 2)
    e0 = e1

    i = 0
    while e1 > ep * e0 and i < imax:

        Ad = (2.0 * convolve2d(x1_180, convolve2d(x1, d, 'valid'), 'valid')
              + 2.0 * convolve2d(x2_180, convolve2d(x2, d, 'valid'), 'valid')
              + 2.0 * mu * d)

        q = Ad
        dq = d.ravel() @ q.ravel()
        if abs(dq) < 1e-30:
            break
        alpha = e1 / dq
        k = k + alpha * d

        if i % 50 == 0 and i > 0:
            gradLS = (2.0 * convolve2d(x1_180,
                                       convolve2d(x1, k, 'valid') - y1,
                                       'valid')
                      + 2.0 * convolve2d(x2_180,
                                         convolve2d(x2, k, 'valid') - y2,
                                         'valid')
                      + 2.0 * mu * (k - k0))
            r = -gradLS
        else:
            r = r - alpha * q

        e0 = e1
        e1 = np.sum(r ** 2)
        if e0 < 1e-30:
            break
        beta = e1 / e0
        d = r + beta * d
        i += 1

    return k

def optimizerank_new(k0, imax, tau, delta):

    X = k0.copy()

    w = np.ones(X.shape[0])

    L = X
    for _ in range(imax):
        U, s, Vh = np.linalg.svd(X, full_matrices=True)

        s_thresh = np.maximum(s - tau * w, 0.0)
        L = (U * s_thresh[np.newaxis, :]) @ Vh

        sv = np.linalg.svd(L, compute_uv=False)
        w = 1.0 / (sv + delta)

    return L

def blinddeconv_new2_cry(y1, y2, x1, x2, lambda_, sigma, k,
                         imax, ximax, xjmax, kimax, rimax,
                         iterkrank, tx, mu, tau, delta,
                         threshold, L2norm):

    if np.sum(np.abs(x1)) == 0:
        x1 = y1.copy()
    if np.sum(np.abs(x2)) == 0:
        x2 = y2.copy()

    n1 = np.linalg.norm(x1.ravel())
    n2 = np.linalg.norm(x2.ravel())
    if n1 > 0:
        x1 = x1 * L2norm / n1
    if n2 > 0:
        x2 = x2 * L2norm / n2

    ksz = k.shape[0]
    bhs = ksz // 2
    if bhs > 0:
        y1v = y1[bhs:-bhs, bhs:-bhs].copy()
        y2v = y2[bhs:-bhs, bhs:-bhs].copy()
    else:
        y1v = y1.copy()
        y2v = y2.copy()

    for i in range(1, imax + 1):

        x1, x2 = optimizex_cry(x1, x2, k, y1, y2,
                                lambda_, ximax, xjmax, tx)

        for it in range(1, iterkrank + 1):

            if it == 1:
                tmpmu = 0.0
            else:
                tmpmu = mu * np.exp(it) / np.exp(iterkrank)

            k = optimizek(x1, x2, k, y1v, y2v, kimax, tmpmu)

            if sigma > 0:
                k = optimizerank_new(k, rimax, tau, delta)

            k[k < 0] = 0.0
            ks = k.sum()
            if ks > 0:
                k = k / ks

        if threshold:
            k[k < k.max() * threshold * i / imax] = 0.0
        else:
            k[k < 0] = 0.0
        ks = k.sum()
        if ks > 0:
            k = k / ks

    return x1, x2, k

def fast_deconv_bregman(f, k, lambda_, alpha):

    beta = 400.0
    initer_max = 1
    outiter_max = 50

    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("Blur kernel must be odd-sized.")

    dx = np.array([[1.0, -1.0]])
    dy = np.array([[1.0], [-1.0]])
    dxt = dx[::-1, ::-1]
    dyt = dy[::-1, ::-1]

    sizef = f.shape
    otfk = psf2otf(k, sizef)
    Ktf = np.conj(otfk) * fft2(f)
    KtK = np.abs(otfk) ** 2
    Fdx = np.abs(psf2otf(dx, sizef)) ** 2
    Fdy = np.abs(psf2otf(dy, sizef)) ** 2
    DtD = Fdx + Fdy

    g = f.copy()
    gx = convolve2d(g, dx, 'valid')
    gy = convolve2d(g, dy, 'valid')

    bx = np.zeros_like(gx)
    by = np.zeros_like(gy)
    wx = gx.copy()
    wy = gy.copy()

    for _outer in range(outiter_max):
        for _inner in range(initer_max):

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

            wx1 = convolve2d(wx - bx, dxt, 'full')
            wy1 = convolve2d(wy - by, dyt, 'full')

            num = lambda_ * Ktf + beta * fft2(wx1 + wy1)
            denom = lambda_ * KtK + beta * DtD
            Fg = num / denom
            g = np.real(ifft2(Fg))

            gx = convolve2d(g, dx, 'valid')
            gy = convolve2d(g, dy, 'valid')

    return g

def multiscaled_cry(y, K, params):

    assert K % 2 == 1, "Kernel size K must be odd."

    minscale = max(2 * int(np.floor((K - 1) / 32)) + 1, 3)
    scales = []
    layer = minscale
    step = np.sqrt(2.0)

    while layer < K:
        scales.append(int(layer))
        layer = int(np.floor(layer * step))
        if layer % 2 == 0:
            layer += 1

    scales.append(K)

    k = np.zeros((minscale, minscale), dtype=np.float64)
    c = int(np.ceil(minscale / 2))
    k[c - 1, c - 2: c] = 0.5

    x1 = np.zeros((minscale, minscale), dtype=np.float64)
    x2 = np.zeros((minscale, minscale), dtype=np.float64)

    dx = np.array([[1.0, -1.0],
                    [0.0,  0.0]])
    dy = np.array([[1.0,  0.0],
                    [-1.0, 0.0]])

    num_scales = len(scales)

    for idx, Ki in enumerate(scales):
        print(f'Processing ksize = {Ki}')
        ratio = Ki / K

        hw = np.floor(np.array(y.shape[:2], dtype=np.float64) * ratio).astype(int)
        smally = imresize(y, (int(hw[0]), int(hw[1])), 'bilinear')

        x1 = imresize(x1, (int(hw[0]) - 1, int(hw[1]) - 1), 'bilinear')
        x2 = imresize(x2, (int(hw[0]) - 1, int(hw[1]) - 1), 'bilinear')

        if idx != 0:
            k = imresize(k, (Ki, Ki), 'bilinear')

        L2norm = 6.0 * Ki / scales[0]

        y1 = convolve2d(smally, dx, 'valid')
        y2 = convolve2d(smally, dy, 'valid')

        x1, x2, k = blinddeconv_new2_cry(
            y1, y2, x1, x2,
            params['lambda_'], params['sigma'], k,
            params['imax'], params['ximax'], params['xjmax'],
            params['kmax'], params['rmax'], params['iterkrank'],
            params['tx'], params['mu'],
            params['tau'] * (idx + 1) / num_scales,
            params['delta'], params['threshold'],
            L2norm,
        )

        y1, x1, k = center_kernel_separate(y1, x1, k)
        y2, x2, k = center_kernel_separate(y2, x2, k)

    k[k < k.max() * params['threshold']] = 0.0
    ks = k.sum()
    if ks > 0:
        k = k / ks

    bhs = K // 2
    nb_lambda = params.get('nb_lambda', 3000)
    nb_alpha = params.get('nb_alpha', 1.0)

    ypad = np.pad(y, bhs, mode='edge')
    for _ in range(4):
        ypad = edgetaper(ypad, k)

    tmp = fast_deconv_bregman(ypad, k, nb_lambda, nb_alpha)
    x = tmp[bhs:tmp.shape[0] - bhs, bhs:tmp.shape[1] - bhs]

    return x, k
