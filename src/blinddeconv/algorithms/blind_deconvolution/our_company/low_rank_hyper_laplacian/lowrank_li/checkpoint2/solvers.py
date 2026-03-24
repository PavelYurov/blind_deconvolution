"""
solvers.py

Core solver functions for Low-Rank Kernel blind deconvolution.

Ported from MATLAB code by Li Siyao et al.
Reference:
    Li Siyao, Shiyu Zhao, Wenzhe Wang, Ping Tan:
    "Understanding Kernel Size in Blind Deconvolution", WACV 2019.

Non-blind deconvolution (fast_deconv_bregman) from Krishnan & Fergus
"Fast Image Deconvolution using Hyper-Laplacian Priors", NIPS 2009.

Contains:
    optimizex_cry        — X-step:  ISTA with l1/l2 normalised sparsity
                           (optimizex_cry.m)
    optimizek            — K-step:  CG for kernel least-squares
                           (optimizek.m)
    optimizerank_new     — Low-rank regularisation of kernel via
                           weighted SVT  (optimizerank_new.m)
    blinddeconv_new2_cry — Alternating x-k minimisation for one scale
                           (blinddeconv_new2_cry.m)
    fast_deconv_bregman  — Non-blind Split-Bregman deconvolution
                           (fast_deconv_bregman.m)
    multiscaled_cry      — Multi-scale coarse-to-fine pipeline
                           (multiscaled_cry.m)

MATLAB → Python critical differences handled:
    - conv2(A,B,mode)   → scipy.signal.convolve2d(A,B,mode)
      (both perform true convolution — kernel is flipped)
    - rot90(k,2)        → np.rot90(k,2)
    - norm(x(:))        → np.linalg.norm(x.ravel())
    - x(:)'*x(:)        → np.sum(x**2)  or  x.ravel() @ x.ravel()
    - [U,S,V]=svd(X)    → U,s,Vh = np.linalg.svd(X)
      MATLAB S = diag matrix, V not transposed
      NumPy  s = 1-D vector , Vh = V^H  (transposed)
    - MATLAB 1-based index → Python 0-based
    - sign(0): 0 in both MATLAB and NumPy
"""

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


# ═════════════════════════════════════════════════════════════════════════════
# optimizex_cry  (from optimizex_cry.m)
#
# X-step: ISTA with normalised sparsity  l1/l2  (Krishnan CVPR 2011)
# ═════════════════════════════════════════════════════════════════════════════

def optimizex_cry(x1, x2, k, y1, y2, lambda_, imax, jmax, t):
    """
    Estimate latent gradient images *x1*, *x2* using ISTA with the
    normalised sparsity measure  ||x||_1 / ||x||_2.

    Parameters
    ----------
    x1, x2  : current estimates of horizontal/vertical gradients
    k       : current blur kernel
    y1, y2  : blurred gradient images
    lambda_ : data-fidelity weight
    imax    : outer ISTA iterations
    jmax    : inner gradient-descent iterations
    t       : initial step size (halved on cost increase)

    Returns
    -------
    x1, x2 : updated gradient estimates
    """
    # Normalise to L2 = 6
    n1 = np.linalg.norm(x1.ravel())
    n2 = np.linalg.norm(x2.ravel())
    if n1 > 0:
        x1 = x1 * 6.0 / n1
    if n2 > 0:
        x2 = x2 * 6.0 / n2

    # Initial cost
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
                # Gradient of data-fidelity term
                grad1 = lambda_ * convolve2d(
                    convolve2d(x1, k, 'same') - y1, k180, 'same')
                grad2 = lambda_ * convolve2d(
                    convolve2d(x2, k, 'same') - y2, k180, 'same')

                # Proximal gradient step (ISTA with l1/l2)
                tmp1 = x1 - t * l21 * grad1
                tmp2 = x2 - t * l22 * grad2
                x1 = np.maximum(0, np.abs(tmp1) - t) * np.sign(tmp1)
                x2 = np.maximum(0, np.abs(tmp2) - t) * np.sign(tmp2)

        # Check cost — halve step if diverging
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


# ═════════════════════════════════════════════════════════════════════════════
# optimizek  (from optimizek.m)
#
# K-step: Conjugate Gradient for  ||x*k - y||^2 + mu*||k - k0||^2
# ═════════════════════════════════════════════════════════════════════════════

def optimizek(x1, x2, k, y1, y2, imax, mu):
    """
    Estimate blur kernel *k* via Conjugate Gradient.

    Solves:
        min_k  ||x1*k - y1||^2 + ||x2*k - y2||^2 + mu*||k - k0||^2

    Parameters
    ----------
    x1, x2 : latent gradient images
    k      : current kernel estimate
    y1, y2 : blurred gradient images (valid-cropped to kernel size)
    imax   : maximum CG iterations
    mu     : proximity weight

    Returns
    -------
    k : updated kernel
    """
    ep = 1e-3
    x1_180 = np.rot90(x1, 2)
    x2_180 = np.rot90(x2, 2)
    k0 = k.copy()

    # A*k  (operator applied to current k)
    Ak = (2.0 * convolve2d(x1_180, convolve2d(x1, k, 'valid'), 'valid')
          + 2.0 * convolve2d(x2_180, convolve2d(x2, k, 'valid'), 'valid')
          + 2.0 * mu * k)

    # Right-hand side b
    b = (2.0 * convolve2d(x1_180, y1, 'valid')
         + 2.0 * convolve2d(x2_180, y2, 'valid')
         + 2.0 * mu * k0)

    # Initial residual
    r = b - Ak
    d = r.copy()
    e1 = np.sum(r ** 2)
    e0 = e1

    i = 0
    while e1 > ep * e0 and i < imax:
        # A*d
        Ad = (2.0 * convolve2d(x1_180, convolve2d(x1, d, 'valid'), 'valid')
              + 2.0 * convolve2d(x2_180, convolve2d(x2, d, 'valid'), 'valid')
              + 2.0 * mu * d)

        q = Ad
        dq = d.ravel() @ q.ravel()
        if abs(dq) < 1e-30:
            break
        alpha = e1 / dq
        k = k + alpha * d

        # Recompute residual every 50 iterations to avoid round-off drift
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


# ═════════════════════════════════════════════════════════════════════════════
# optimizerank_new  (from optimizerank_new.m)
#
# Low-rank regularisation via weighted Singular Value Thresholding
# ═════════════════════════════════════════════════════════════════════════════

def optimizerank_new(k0, imax, tau, delta):
    """
    Minimise  (1/2*tau)*||k - k0||^2  +  rank(k)
    using a log-det proxy for rank and iterative re-weighted SVT.

    CRITICAL MATLAB→Python difference:
        MATLAB  [U, S, V] = svd(X)  →  S is diagonal *matrix*, V not transposed.
        NumPy   U, s, Vh = svd(X)   →  s is 1-D *vector*, Vh = V^H (transposed).

    Parameters
    ----------
    k0    : input kernel (2-D, typically square)
    imax  : SVT iterations  (typically 3)
    tau   : singular-value threshold step
    delta : stabilisation constant for log-det  (typically 1e-5)

    Returns
    -------
    k : low-rank-regularised kernel
    """
    X = k0.copy()
    # Initial weights  (mu=1 in MATLAB)
    w = np.ones(X.shape[0])

    L = X  # fallback if imax == 0
    for _ in range(imax):
        U, s, Vh = np.linalg.svd(X, full_matrices=True)
        # Soft-threshold singular values with per-value weights
        s_thresh = np.maximum(s - tau * w, 0.0)
        L = (U * s_thresh[np.newaxis, :]) @ Vh   # U @ diag(s_thresh) @ Vh
        # NOTE: MATLAB does NOT update X here — always decomposes the
        # original k0.  Only the weights w are updated from L.

        # Update weights:  w_i = 1 / (sigma_i(L) + delta)
        sv = np.linalg.svd(L, compute_uv=False)
        w = 1.0 / (sv + delta)

    return L


# ═════════════════════════════════════════════════════════════════════════════
# blinddeconv_new2_cry  (from blinddeconv_new2_cry.m)
#
# Alternating x–k minimisation at a single scale
# ═════════════════════════════════════════════════════════════════════════════

def blinddeconv_new2_cry(y1, y2, x1, x2, lambda_, sigma, k,
                         imax, ximax, xjmax, kimax, rimax,
                         iterkrank, tx, mu, tau, delta,
                         threshold, L2norm):
    """
    One-scale blind deconvolution: alternating between the x-step
    (latent gradient estimation) and the k-step (kernel estimation with
    optional low-rank regularisation).

    Parameters
    ----------
    y1, y2      : blurred gradient images (full)
    x1, x2      : initial latent gradient estimates
    lambda_     : data weight for x-step
    sigma       : flag (>0 → apply low-rank regularisation to k)
    k           : initial kernel
    imax        : total x–k alternations
    ximax       : outer ISTA iterations (x-step)
    xjmax       : inner ISTA iterations (x-step)
    kimax       : CG iterations (k-step)
    rimax       : SVT iterations  (rank step)
    iterkrank   : k-step + rank repetitions per alternation
    tx          : initial ISTA step size
    mu          : proximity weight for k-step
    tau         : SVT threshold
    delta       : log-det stabiliser
    threshold   : kernel threshold factor  (frac of max)
    L2norm      : target L2 norm for gradient images

    Returns
    -------
    x1, x2, k : updated estimates
    """
    # Initialise x from y if zero
    if np.sum(np.abs(x1)) == 0:
        x1 = y1.copy()
    if np.sum(np.abs(x2)) == 0:
        x2 = y2.copy()

    # Normalise gradient images
    n1 = np.linalg.norm(x1.ravel())
    n2 = np.linalg.norm(x2.ravel())
    if n1 > 0:
        x1 = x1 * L2norm / n1
    if n2 > 0:
        x2 = x2 * L2norm / n2

    # Crop y to 'valid' region (remove kernel-border pixels)
    ksz = k.shape[0]
    bhs = ksz // 2
    if bhs > 0:
        y1v = y1[bhs:-bhs, bhs:-bhs].copy()
        y2v = y2[bhs:-bhs, bhs:-bhs].copy()
    else:
        y1v = y1.copy()
        y2v = y2.copy()

    for i in range(1, imax + 1):
        # ── x-step ──────────────────────────────────────────────────────
        x1, x2 = optimizex_cry(x1, x2, k, y1, y2,
                                lambda_, ximax, xjmax, tx)

        # ── k-step (iterkrank repetitions) ──────────────────────────────
        for it in range(1, iterkrank + 1):
            # MATLAB: tmpmu = 0 on first iter, then exponential ramp
            if it == 1:
                tmpmu = 0.0
            else:
                tmpmu = mu * np.exp(it) / np.exp(iterkrank)

            k = optimizek(x1, x2, k, y1v, y2v, kimax, tmpmu)

            if sigma > 0:
                k = optimizerank_new(k, rimax, tau, delta)

            # Project: non-negative, sum-to-one
            k[k < 0] = 0.0
            ks = k.sum()
            if ks > 0:
                k = k / ks

        # Progressive kernel thresholding
        if threshold:
            k[k < k.max() * threshold * i / imax] = 0.0
        else:
            k[k < 0] = 0.0
        ks = k.sum()
        if ks > 0:
            k = k / ks

    return x1, x2, k


# ═════════════════════════════════════════════════════════════════════════════
# fast_deconv_bregman  (from fast_deconv_bregman.m)
#
# Non-blind deconvolution via Split Bregman / ADMM
# (Krishnan & Fergus, NIPS 2009)
# ═════════════════════════════════════════════════════════════════════════════

def fast_deconv_bregman(f, k, lambda_, alpha):
    """
    Non-blind image deconvolution using Hyper-Laplacian priors.

    min_g  (lambda/2)||g*k - f||^2  +  ||nabla g||^alpha

    Solved via Split Bregman (ADMM).

    Parameters
    ----------
    f       : observed (padded) blurred image  (2-D)
    k       : blur kernel (odd-sized)
    lambda_ : data-fidelity weight
    alpha   : sparsity exponent  (0.5, 2/3, or 1)

    Returns
    -------
    g : deblurred image, same size as f
    """
    beta = 400.0
    initer_max = 1
    outiter_max = 50

    # Check kernel is odd-sized
    if k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise ValueError("Blur kernel must be odd-sized.")

    # Derivative kernels
    dx = np.array([[1.0, -1.0]])          # (1, 2)
    dy = np.array([[1.0], [-1.0]])         # (2, 1)
    dxt = dx[::-1, ::-1]                   # [-1, 1]
    dyt = dy[::-1, ::-1]                   # [-1; 1]

    # Pre-compute frequency-domain constants
    sizef = f.shape
    otfk = psf2otf(k, sizef)
    Ktf = np.conj(otfk) * fft2(f)
    KtK = np.abs(otfk) ** 2
    Fdx = np.abs(psf2otf(dx, sizef)) ** 2
    Fdy = np.abs(psf2otf(dy, sizef)) ** 2
    DtD = Fdx + Fdy

    # Initialise
    g = f.copy()
    gx = convolve2d(g, dx, 'valid')
    gy = convolve2d(g, dy, 'valid')

    bx = np.zeros_like(gx)
    by = np.zeros_like(gy)
    wx = gx.copy()
    wy = gy.copy()

    for _outer in range(outiter_max):
        for _inner in range(initer_max):
            # ── w-step: proximal operator ────────────────────────────────
            if abs(alpha - 1.0) < 1e-9:
                # Soft thresholding (alpha = 1)
                tmpx = gx + bx
                tmpy = gy + by
                wx = np.maximum(np.abs(tmpx) - 1.0 / beta, 0.0) * np.sign(tmpx)
                wy = np.maximum(np.abs(tmpy) - 1.0 / beta, 0.0) * np.sign(tmpy)
            else:
                wx = solve_image_bregman(gx + bx, beta, alpha)
                wy = solve_image_bregman(gy + by, beta, alpha)

            # Bregman update
            bx = bx - wx + gx
            by = by - wy + gy

            # ── g-step: solve in Fourier domain ─────────────────────────
            wx1 = convolve2d(wx - bx, dxt, 'full')
            wy1 = convolve2d(wy - by, dyt, 'full')

            num = lambda_ * Ktf + beta * fft2(wx1 + wy1)
            denom = lambda_ * KtK + beta * DtD
            Fg = num / denom
            g = np.real(ifft2(Fg))

            gx = convolve2d(g, dx, 'valid')
            gy = convolve2d(g, dy, 'valid')

    return g


# ═════════════════════════════════════════════════════════════════════════════
# multiscaled_cry  (from multiscaled_cry.m)
#
# Multi-scale coarse-to-fine blind deconvolution pipeline
# ═════════════════════════════════════════════════════════════════════════════

def multiscaled_cry(y, K, params):
    """
    Full multi-scale blind deconvolution: kernel estimation in gradient
    domain followed by non-blind restoration.

    Parameters
    ----------
    y      : blurred grayscale image  (2-D, float64, [0, 1])
    K      : expected kernel size  (odd integer >= 3)
    params : dict with keys:
             lambda_, sigma, imax, ximax, xjmax, kmax, rmax,
             iterkrank, tx, mu, tau, delta, threshold,
             nb_lambda, nb_alpha

    Returns
    -------
    x : restored image  (2-D, float64)
    k : estimated kernel (2-D, float64, non-negative, sums to 1)
    """
    assert K % 2 == 1, "Kernel size K must be odd."

    # ── Build scale pyramid ──────────────────────────────────────────────
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

    # ── Initial kernel: horizontal 2-pixel line ──────────────────────────
    #   MATLAB: k(ceil(ms/2), ceil(ms/2)-1 : ceil(ms/2)) = 0.5
    #   Python 0-based: row = ceil(ms/2)-1, cols = [ceil(ms/2)-2, ceil(ms/2)-1]
    k = np.zeros((minscale, minscale), dtype=np.float64)
    c = int(np.ceil(minscale / 2))          # MATLAB 1-based centre
    k[c - 1, c - 2: c] = 0.5               # Python 0-based

    x1 = np.zeros((minscale, minscale), dtype=np.float64)
    x2 = np.zeros((minscale, minscale), dtype=np.float64)

    # Derivative kernels (same as MATLAB: dx=[1,-1;0,0], dy=[1,0;-1,0])
    dx = np.array([[1.0, -1.0],
                    [0.0,  0.0]])
    dy = np.array([[1.0,  0.0],
                    [-1.0, 0.0]])

    num_scales = len(scales)

    for idx, Ki in enumerate(scales):
        print(f'Processing ksize = {Ki}')
        ratio = Ki / K

        # Resize blurred image and gradient estimates
        hw = np.floor(np.array(y.shape[:2], dtype=np.float64) * ratio).astype(int)
        smally = imresize(y, (int(hw[0]), int(hw[1])), 'bilinear')

        # Gradient images are 1 pixel smaller per axis (conv2 valid with 2×2 kernel)
        x1 = imresize(x1, (int(hw[0]) - 1, int(hw[1]) - 1), 'bilinear')
        x2 = imresize(x2, (int(hw[0]) - 1, int(hw[1]) - 1), 'bilinear')

        if idx != 0:
            k = imresize(k, (Ki, Ki), 'bilinear')

        L2norm = 6.0 * Ki / scales[0]

        # Gradient images of blurred input
        y1 = convolve2d(smally, dx, 'valid')
        y2 = convolve2d(smally, dy, 'valid')

        # ── Blind deconvolution at this scale ────────────────────────────
        x1, x2, k = blinddeconv_new2_cry(
            y1, y2, x1, x2,
            params['lambda_'], params['sigma'], k,
            params['imax'], params['ximax'], params['xjmax'],
            params['kmax'], params['rmax'], params['iterkrank'],
            params['tx'], params['mu'],
            params['tau'] * (idx + 1) / num_scales,   # MATLAB: tau*i/length(scale)
            params['delta'], params['threshold'],
            L2norm,
        )

        # Centre kernel (and shift gradient images accordingly)
        y1, x1, k = center_kernel_separate(y1, x1, k)
        y2, x2, k = center_kernel_separate(y2, x2, k)

    # ── Final kernel thresholding ────────────────────────────────────────
    k[k < k.max() * params['threshold']] = 0.0
    ks = k.sum()
    if ks > 0:
        k = k / ks

    # ── Non-blind deconvolution ──────────────────────────────────────────
    bhs = K // 2
    nb_lambda = params.get('nb_lambda', 3000)
    nb_alpha = params.get('nb_alpha', 1.0)

    ypad = np.pad(y, bhs, mode='edge')
    for _ in range(4):
        ypad = edgetaper(ypad, k)

    tmp = fast_deconv_bregman(ypad, k, nb_lambda, nb_alpha)
    x = tmp[bhs:tmp.shape[0] - bhs, bhs:tmp.shape[1] - bhs]

    return x, k
