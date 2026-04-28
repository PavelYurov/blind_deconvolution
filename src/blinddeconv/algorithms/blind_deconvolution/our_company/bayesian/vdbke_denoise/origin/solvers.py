"""
solvers.py

Core solver functions for VDBKE (Variational Dirichlet Blur Kernel Estimation).

Ported from MATLAB code by X. Zhou et al.
Reference:
    X. Zhou, J. Mateos, F. Zhou, R. Molina, A.K. Katsaggelos:
    "Variational Dirichlet Blur Kernel Estimation",
    IEEE TIP, vol. 24, no. 12, pp. 5127-5139, 2015.
"""

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.special import digamma, polygamma, gammaln
from scipy.fft import dstn, idstn

from .utils import psf2otf, valid_conv_by_fft


# ═════════════════════════════════════════════════════════════════════════════
# center_kernel_img_space  ←  center_kernel_img_space.m
# ═════════════════════════════════════════════════════════════════════════════

def center_kernel_img_space(x, k, verbose=False):
    """
    Centre the kernel by translation so that boundary issues are mitigated.
    If the kernel is shifted, the image must also be shifted in the opposite
    direction.

    Ported from ``center_kernel_img_space.m``.

    MATLAB (key lines)::

        mu_y = sum([1:size(k,1)] .* sum(k,2)');
        mu_x = sum([1:size(k,2)] .* sum(k,1));
        offset_x = round(floor(size(k,2)/2) + 1 - mu_x);
        offset_y = round(floor(size(k,1)/2) + 1 - mu_y);
        shift_kernel = zeros(abs(offset_y*2)+1, abs(offset_x*2)+1);
        shift_kernel(abs(offset_y)+1+offset_y, abs(offset_x)+1+offset_x) = 1;
        kshift = conv2(k, shift_kernel, 'same');
        xshift = conv2(x, rot90(shift_kernel,2), 'same');

    Parameters
    ----------
    x : 2-D ndarray — latent image estimate.
    k : 2-D ndarray — blur kernel.

    Returns
    -------
    xshift       : 2-D ndarray — shifted image.
    k            : 2-D ndarray — centred kernel.
    shift_kernel : 2-D ndarray — the translation kernel used.
    """
    # Centre of mass (MATLAB uses 1-based indices [1:size(k,1)])
    rows = np.arange(1, k.shape[0] + 1, dtype=np.float64)  # [1, 2, ..., k1]
    cols = np.arange(1, k.shape[1] + 1, dtype=np.float64)  # [1, 2, ..., k2]

    # MATLAB: mu_y = sum([1:size(k,1)] .* sum(k,2)')
    #   sum(k,2) is column-vector of row-sums → .' makes it row-vector
    mu_y = np.sum(rows * np.sum(k, axis=1))
    # MATLAB: mu_x = sum([1:size(k,2)] .* sum(k,1))
    #   sum(k,1) is row-vector of column-sums
    mu_x = np.sum(cols * np.sum(k, axis=0))

    # MATLAB: offset_x = round(floor(size(k,2)/2) + 1 - mu_x)
    offset_x = int(np.round(k.shape[1] // 2 + 1 - mu_x))
    offset_y = int(np.round(k.shape[0] // 2 + 1 - mu_y))

    if verbose:
        print(f'CenterKernel: weightedMean[{mu_x - 1:.6f} {mu_y - 1:.6f}] '
              f'offset[{offset_x} {offset_y}]')

    # Build shift kernel (delta at the offset position)
    shift_kernel = np.zeros((abs(offset_y) * 2 + 1, abs(offset_x) * 2 + 1),
                            dtype=np.float64)
    # MATLAB 1-indexed: shift_kernel(abs(offset_y)+1+offset_y, abs(offset_x)+1+offset_x) = 1
    # Python 0-indexed: row = abs(offset_y)+offset_y, col = abs(offset_x)+offset_x
    shift_kernel[abs(offset_y) + offset_y, abs(offset_x) + offset_x] = 1.0

    # Shift both kernel and image
    k = convolve2d(k, shift_kernel, 'same')
    xshift = convolve2d(x, np.rot90(shift_kernel, 2), 'same')

    return xshift, k, shift_kernel


# ═════════════════════════════════════════════════════════════════════════════
# dirichlet_Adbc_fft  ←  dirichlet_Adbc_fft.m
# ═════════════════════════════════════════════════════════════════════════════

def dirichlet_Adbc_fft(x_list, y_list, m1, m2, lambda_C=0, C=None, verbose=False):
    """
    Pre-compute Ad, b, and function handles (closures) for X'X products
    needed by the Dirichlet kernel estimation.

    Ported from ``dirichlet_Adbc_fft.m``.

    MATLAB (main loop)::

        for t = 1:length(x)
            X{t}  = fft2(x{t});
            Xr{t} = fft2(rot90(x{t},2));
            b = b - 2*valid_conv_by_fft(Xr{t}, y{t});
            xx = x{t}.^2;
            for i = m1:-1:1
                for j = m2:-1:1
                    Ad(m1+1-i, m2+1-j) += sum(sum(xx(i:i+M1-m1, j:j+M2-m2)));
                end
            end
        end

    Parameters
    ----------
    x_list   : list of 2-D ndarrays — derivative images of the latent estimate.
    y_list   : list of 2-D ndarrays — derivative images of the blurred observation.
    m1, m2   : int — kernel size (rows, cols).
    lambda_C : float — weight on the Laplacian regulariser (default 0).
    C        : 2-D ndarray or None — Laplacian filter (e.g. [[0,-1,0],[-1,4,-1],[0,-1,0]]).

    Returns
    -------
    Ad       : (m1, m2) ndarray
    b        : (m1, m2) ndarray
    xtAx_func: callable(alpha) → (scalar, list_of_Xalpha)
    Ax_func  : callable(Xalpha_list) → ndarray
    """
    L = len(x_list)
    Ad = np.zeros((m1, m2), dtype=np.float64)
    b = np.zeros((m1, m2), dtype=np.float64)

    X_fft = []      # fft2 of each x
    Xr_fft = []     # fft2 of rot180(x)
    x_spatial = []   # keep spatial copies for conv fallback

    for t in range(L):
        xt = x_list[t]
        M1, M2 = xt.shape

        X_fft.append(fft2(xt))
        flipxt = np.rot90(xt, 2)
        Xr_fft.append(fft2(flipxt))
        x_spatial.append(xt)

        # b = b - 2*valid_conv_by_fft(Xr{t}, y{t})
        b -= 2.0 * valid_conv_by_fft(Xr_fft[t], y_list[t])

        # Build Ad via sliding-window sums of x^2
        # MATLAB (1-indexed): Ad(m1+1-i, m2+1-j) += sum(sum(xx(i:i+M1-m1, j:j+M2-m2)))
        # Python (0-indexed): for i in [1..m1], j in [1..m2]:
        #   Ad[m1-i, m2-j] += xx[i-1 : i-1+M1-m1+1, j-1 : j-1+M2-m2+1].sum()
        xx = xt ** 2
        rows_len = M1 - m1 + 1  # number of rows in the valid window
        cols_len = M2 - m2 + 1  # number of cols in the valid window
        for i in range(m1):      # i = 0..m1-1  (corresponds to MATLAB i=m1..1)
            for j in range(m2):  # j = 0..m2-1
                # MATLAB: Ad(m1+1-i, m2+1-j) += ... with i from m1 downto 1
                # Python: map i→m1-1-i, j→m2-1-j to match the reversed indexing
                Ad[m1 - 1 - i, m2 - 1 - j] += np.sum(xx[i:i + rows_len, j:j + cols_len])

    if verbose:
        print(f'Ad_min={Ad.min():.6f},Ad_max={Ad.max():.6f}')

    # Optional Laplacian regularisation
    if lambda_C > 0 and C is not None:
        CtC = convolve2d(C, C, 'same')
        Cd = CtC[CtC.shape[0] // 2, CtC.shape[1] // 2]  # MATLAB: CtC(2,2) for 3x3
        Ad = Ad + lambda_C * Cd

    # ── closure: xtAx(alpha) ──
    # MATLAB: function [y, Xalpha] = xtAx(alpha, X, x, L, lambda_C, C)
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

    # ── closure: Ax(Xalpha_list) ──
    # MATLAB: function y = Ax(Xalpha, Xr, L, lambda_C, C)
    def Ax_func(Xalpha_list):
        y_val = np.zeros((m1, m2), dtype=np.float64)
        for i in range(L):
            y_val += valid_conv_by_fft(Xr_fft[i], Xalpha_list[i])
        if lambda_C > 0 and C is not None:
            y_val += lambda_C * convolve2d(Xalpha_list[L], C, 'same')
        return y_val

    return Ad, b, xtAx_func, Ax_func


# ═════════════════════════════════════════════════════════════════════════════
# dirichlet_cost_by_fft  ←  inner function of kernel_estimation_filter_space_fft.m
# ═════════════════════════════════════════════════════════════════════════════

def _dirichlet_cost_by_fft(alpha, lam, xtAx_func, Ad, b):
    """
    Compute the Dirichlet cost and auxiliary quantities.

    MATLAB::

        Sa   = sum(alpha(:));
        den1 = 2*Sa*(Sa+1);
        [atAa, Xalpha] = xtAx(alpha);
        Adta = sum(sum(Ad.*alpha));
        bta  = sum(sum(b.*alpha));
        temp = (alpha-1).*(psi(alpha)-psi(Sa));
        f = lambda*(sum(temp(:)) + gammaln(Sa) - sum(gammaln(alpha(:))))
            + (atAa+Adta)/den1 + bta/(2*Sa);

    Returns
    -------
    f, atAa, Adta, bta, Sa, den1, Xalpha
    """
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


# ═════════════════════════════════════════════════════════════════════════════
# kernel_estimation_filter_space_fft  ←  kernel_estimation_filter_space_fft.m
# ═════════════════════════════════════════════════════════════════════════════

def kernel_estimation_filter_space_fft(k, x_list, y_list, opt, verbose=False):
    """
    Gradient projection method for Dirichlet minimisation (FFT version).

    Ported from ``kernel_estimation_filter_space_fft.m``.

    Parameters
    ----------
    k      : (ks1, ks2) ndarray — current kernel estimate.
    x_list : list of 2-D ndarrays — gradient images of the latent estimate.
    y_list : list of 2-D ndarrays — gradient images of the blurred observation.
    opt    : dict with keys:
        'lambda', 'lambda_C', 'Laplacian_filter', 'max_iter',
        'back_alpha', 'back_beta', 'lower_bound', 'ng_min',
        'cost_display', 'alpha0'

    Returns
    -------
    alpha  : (ks1, ks2) ndarray — Dirichlet parameters.
    fx     : list of float — cost at each iteration.
    stepsize : float — last step size used.
    """
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

    # Compute A, Ad and b
    Ad, b, xtAx_func, Ax_func = dirichlet_Adbc_fft(
        x_list, y_list, ks1, ks2, lambda_C, C, verbose=verbose)

    # Initial cost
    f0, atAa, Adta, bta, Sa, den1, Xalpha = _dirichlet_cost_by_fft(
        alpha, lam, xtAx_func, Ad, b)
    fx = [f0]
    costcalls = 1
    if cost_display:
        print(f'iteration=0,cost={fx[0]:.6f}')

    stepsize = Sa
    itr = 0

    while itr < max_iter:
        # Compute gradient
        _EPS_g = 1e-30
        den2 = den1 ** 2 / (4.0 * Sa + 2.0 + _EPS_g)
        g = lam * (alpha - 1.0) * (polygamma(1, np.maximum(alpha, _EPS_g)) - polygamma(1, max(Sa, _EPS_g)))
        Aa = Ax_func(Xalpha)
        g = (g + (2.0 * Aa + Ad) / den1 + b / (2.0 * Sa + _EPS_g)
             - (atAa + Adta) / den2 - bta / (2.0 * Sa ** 2 + _EPS_g))

        # Backtracking line search
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

    if verbose:
        print(f'DIter={itr},costcalls={costcalls},cost={fx[0]:.4f},'
              f'cost={fx[-1]:.4f},rf={rf:.6f},Sa={Sa:.0f}')

    return alpha, fx, stepsize


# ═════════════════════════════════════════════════════════════════════════════
# nbid_ngm_ubc_admm  ←  nbid_ngm_ubc_admm.m  (Algorithm 1)
# ═════════════════════════════════════════════════════════════════════════════

def nbid_ngm_ubc_admm(B, k, pars):
    """
    Non-blind image deconvolution using NGM prior and undetermined boundary
    conditions (Algorithm 1 of the paper).

    Ported from ``nbid_ngm_ubc_admm.m``.

    Parameters
    ----------
    B    : (m, n) ndarray — blurred image.
    k    : 2-D ndarray — blur kernel.
    pars : dict with keys:
        'lambda1'     — weight on the NGM term
        'x0'          — initial latent image
      optional:
        'lambda_min'  (default lambda1*5)
        'lambda_max'  (default 1)
        'cost_display'(default 0)
        'IF'          (default sqrt(2))
        'N2'          (default 2)
        'N1'          (default 20)
        'lambda_u'    (default 0.1)
        'xv_iter'     (default 1)

    Returns
    -------
    x_fov : (m, n) ndarray — deblurred image (field of view).
    x     : (M, N) ndarray — full deblurred image (with border).
    """
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
        # MATLAB: padarray(x,[hks1 hks2],'replicate','both')
        x = np.pad(x, ((hks1, hks1), (hks2, hks2)), mode='edge')

    # MATLAB: MtB = padarray(B,[hks1 hks2],0,'both')
    MtB = np.pad(B, ((hks1, hks1), (hks2, hks2)), mode='constant', constant_values=0)
    MtM = np.pad(np.ones((m, n), dtype=np.float64),
                 ((hks1, hks1), (hks2, hks2)), mode='constant', constant_values=0)

    # Circular gradients
    # MATLAB: x1 = [diff(x,1,2), x(:,1)-x(:,end)]
    x1 = np.concatenate([np.diff(x, axis=1), x[:, 0:1] - x[:, -1:]], axis=1)
    # MATLAB: x2 = [diff(x,1,1); x(1,:)-x(end,:)]
    x2 = np.concatenate([np.diff(x, axis=0), x[0:1, :] - x[-1:, :]], axis=0)

    # MATLAB: u = padarray(B,[hks1 hks2],'replicate','both')
    u = np.pad(B, ((hks1, hks1), (hks2, hks2)), mode='edge')
    du = np.zeros_like(u)  # dual variable

    X = fft2(x)
    i = 1

    while i <= N1:
        # Update u
        Ax = np.real(ifft2(X * K))
        u = (MtB + lambda_u * (Ax + du)) / (MtM + lambda_u)

        # Update dual variable
        du = du + Ax - u

        # Ktu = fft2(u-du) .* Kt
        Ktu = fft2(u - du) * Kt

        invA = 1.0 / (KtK + lambda_v / lambda_u * DtD + 1e-30)
        lam = lambda1 / lambda_v

        # Update v
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
                beta1 = lam * G1 / ((v1 + G1) ** 2 + 1e-30)
                beta2 = lam * G2 / ((v2 + G2) ** 2 + 1e-30)
                v1 = np.maximum(temp1 - beta1, 0.0)
                v2 = np.maximum(temp2 - beta2, 0.0)
            v1 = s1 * v1
            v2 = s2 * v2

        # Update x
        # MATLAB: temp1 = -[v1(:,1)-v1(:,end), diff(v1,1,2)]
        temp1 = -np.concatenate([v1[:, 0:1] - v1[:, -1:],
                                 np.diff(v1, axis=1)], axis=1)
        # MATLAB: temp2 = -[v2(1,:)-v2(end,:); diff(v2,1,1)]
        temp2 = -np.concatenate([v2[0:1, :] - v2[-1:, :],
                                 np.diff(v2, axis=0)], axis=0)
        X = Ktu + lambda_v / lambda_u * fft2(temp1 + temp2)
        X = invA * X
        x = np.real(ifft2(X))

        # Recompute circular gradients
        x1 = np.concatenate([np.diff(x, axis=1), x[:, 0:1] - x[:, -1:]], axis=1)
        x2 = np.concatenate([np.diff(x, axis=0), x[0:1, :] - x[-1:, :]], axis=0)

        if cost_display:
            t1 = np.abs(x1.ravel())
            t2 = np.abs(x2.ravel())
            G1 = np.mean(t1)
            G2 = np.mean(t2)
            pcost_i = lambda1 * (np.sum(t1 / (t1 + G1 + 1e-30)) + np.sum(t2 / (t2 + G2 + 1e-30)))
            Ax_fov = Ax[hks1:hks1 + m, hks2:hks2 + n]
            fcost_i = 0.5 * np.sum((Ax_fov - B) ** 2) + pcost_i
            print(f'Outer iteration {i}: fcost={fcost_i:.6f} pcost={pcost_i:.6f}')

        lambda_v = min(lambda_v * IF, lambda_max)
        i += 1

    # Crop to field of view
    # MATLAB 1-indexed: x(hks1+1:m+hks1, hks2+1:n+hks2)
    # Python 0-indexed: x[hks1:hks1+m, hks2:hks2+n]
    if hks1 == 0 and hks2 == 0:
        x_fov = x.copy()
    elif hks1 == 0:
        x_fov = x[:, hks2:hks2 + n].copy()
    elif hks2 == 0:
        x_fov = x[hks1:hks1 + m, :].copy()
    else:
        x_fov = x[hks1:hks1 + m, hks2:hks2 + n].copy()

    # Replace NaN/Inf with 0 to prevent propagation
    np.nan_to_num(x_fov, copy=False, nan=0.0, posinf=1.0, neginf=0.0)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    return x_fov, x


# ═════════════════════════════════════════════════════════════════════════════
# ss_ngm_dirichlet_ubc_img  ←  ss_ngm_dirichlet_ubc_img.m
# ═════════════════════════════════════════════════════════════════════════════

def ss_ngm_dirichlet_ubc_img(y, x, k, alpha0, pars, blind_denoise_fn=None, verbose=False):
    """
    Single-scale blind deconvolution: alternating latent-image (x) and
    kernel (k) estimation.

    Ported from ``ss_ngm_dirichlet_ubc_img.m``.

    Parameters
    ----------
    y      : (m, n) ndarray — blurred image at current scale.
    x      : (m, n) ndarray — initial latent image estimate.
    k      : (k1, k2) ndarray — initial kernel estimate.
    alpha0 : (k1, k2) ndarray — initial Dirichlet parameters.
    pars   : dict with keys:
        'xk_iter'     — number of alternating iterations
        'img_pars'    — dict of parameters for nbid_ngm_ubc_admm
        'kernel_pars' — dict of parameters for kernel_estimation_filter_space_fft
        'k_tol'       — convergence tolerance on kernel change
    blind_denoise_fn : callable or None
        Optional denoiser applied to x_fov **before** gradient
        computation for kernel estimation.  Signature: f(ndarray) -> ndarray.
        Default None — no denoising (original behaviour).

    Returns
    -------
    x      : (m, n) ndarray — estimated latent image (FOV).
    k      : (k1, k2) ndarray — estimated kernel.
    alpha0 : (k1, k2) ndarray — final Dirichlet parameters.
    """
    m, n = y.shape
    k1, k2 = k.shape
    khs1 = k1 // 2
    khs2 = k2 // 2

    xk_iter = pars['xk_iter']
    img_pars = pars['img_pars'].copy()
    img_pars['x0'] = x.copy()

    # Gradient images of y for kernel estimation
    # MATLAB: y2{1} = diff(y(khs1+1:end-khs1, khs2+1:end-khs2), 1, 1)
    # Python: crop y then diff
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
        # Adjust lambda with schedule
        img_pars['lambda1'] = lambda1 + delta_lambda * max(6 - (i + 1), 0)
        img_pars['lambda_min'] = lambda_min / max(lambda1, 1e-30) * img_pars['lambda1']

        # Latent image estimation (Alg. 1)
        x_fov, x_full = nbid_ngm_ubc_admm(y, k, img_pars)
        img_pars['x0'] = x_full  # use full x as init for next iteration

        # Optional inter-iteration denoising before gradient computation
        x_for_grad = blind_denoise_fn(x_fov) if blind_denoise_fn is not None else x_fov

        # Gradient images of x for kernel estimation
        x1 = [np.diff(x_for_grad, axis=0), np.diff(x_for_grad, axis=1)]

        # Kernel estimation
        ker_opts['alpha0'] = alpha.copy()
        alpha, fcost, _ = kernel_estimation_filter_space_fft(k, x1, y2, ker_opts, verbose=verbose)
        alpha0 = alpha.reshape(k1, k2)

        # Update kernel from Dirichlet parameters
        if ker_opts.get('mode', 0):
            # mode estimator
            denom_mode = np.sum(alpha) - k1 * k2
            if abs(denom_mode) < 1e-30:
                denom_mode = 1e-30
            k = (alpha0 - 1.0) / denom_mode
        else:
            # expectation estimator (default)
            k = alpha0 / max(np.sum(alpha), 1e-30)

        # Guard against NaN — fall back to previous kernel
        if np.any(np.isnan(k)):
            if k_old is not None:
                k = k_old.copy()
            else:
                k = np.ones((k1, k2), dtype=np.float64) / (k1 * k2)

        if verbose:
            print(f'Iteration={i + 1}')

        # Convergence check (starts at iteration 5, MATLAB i>=5 is 1-indexed)
        if i >= 4 and k_old is not None:
            r_k = np.max(np.abs(k - k_old)) / 1.0  # max_k = 1
            if len(fcost) <= 2 or r_k <= pars.get('k_tol', 1e-4):
                break

        k_old = k.copy()

    return x_fov, k, alpha0


# ═════════════════════════════════════════════════════════════════════════════
# firls_deb_ubc  ←  firls_deb_ubc.m
# ═════════════════════════════════════════════════════════════════════════════

def firls_deb_ubc(y, h, opt, verbose=False):
    """
    Fast IRLS for image deblurring with undetermined boundary conditions.
    Uses 1st and 2nd order derivative filters + ADMM.

    Ported from ``firls_deb_ubc.m``.

    Parameters
    ----------
    y   : (M1, M2) ndarray — blurred image.
    h   : (m1, m2) ndarray — blur kernel (odd size).
    opt : dict with keys:
        'lambda'       — regularisation weight
      optional:
        'alpha'        (default 2/3)
        'beta_a'       (default lambda*alpha*(20/255)^(alpha-2))
        'lambda_u'     (default min(0.1, 5000*lambda))
        'inner_iter'   (default 4)
        'out_iter'     (default 5)
        'epsilon'      (default 0.01)
        'cost_display' (default 0)
        'isnr_display' (default 0)
        'groundtruth'  — needed if isnr_display==1

    Returns
    -------
    x_fov : (M1, M2) ndarray — deblurred image (field of view).
    x     : (n1, n2) ndarray — full deblurred image.
    opt   : dict — updated with cost/isnr if requested.
    """
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = np.pad(y, ((hks1, hks1), (hks2, hks2)), mode='edge')

    # 1st and 2nd order derivative filters (3x3)
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

    # Initial derivatives via circular-padded convolution
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

    # Dual variables (initialised to zero)
    du   = np.zeros((n1, n2), dtype=np.float64)
    dvx  = np.zeros_like(du)
    dvy  = np.zeros_like(du)
    dvxx = np.zeros_like(du)
    dvyy = np.zeros_like(du)
    dvxy = np.zeros_like(du)

    X_ = fft2(x)
    Ax_ = np.real(ifft2(H * X_))
    invA = HH + beta_a / lambda_u * RR + 1e-30

    # Output containers
    opt_out = dict(opt)

    outer = 0
    while outer < N1:
        outer += 1

        # Clamp derivatives away from zero before negative power (alpha_p - 2 < 0)
        _eps_d = 1e-10
        Wx  = np.minimum(beta, c * np.maximum(adx,  _eps_d) ** (alpha_p - 2))
        Wy  = np.minimum(beta, c * np.maximum(ady,  _eps_d) ** (alpha_p - 2))
        Wxx = np.minimum(beta, c * np.maximum(adxx, _eps_d) ** (alpha_p - 2)) * w0
        Wyy = np.minimum(beta, c * np.maximum(adyy, _eps_d) ** (alpha_p - 2)) * w0
        Wxy = np.minimum(beta, c * np.maximum(adxy, _eps_d) ** (alpha_p - 2)) * w0

        # Inner ADMM loop
        inner = 0
        while inner < N2:
            inner += 1

            # u sub-problem
            u = Ax_ + du
            u[hks1:hks1 + M1, hks2:hks2 + M2] = (
                (y + lambda_u * u[hks1:hks1 + M1, hks2:hks2 + M2])
                / (1.0 + lambda_u)
            )

            # v sub-problems
            vx  = beta_a * (dx_  + dvx)  / (Wx  + beta_a)
            vy  = beta_a * (dy_  + dvy)  / (Wy  + beta_a)
            vxx = beta_a * (dxx_ + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy_ + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy_ + dvxy) / (Wxy + beta_a)

            # Update dual variables
            du   = du   - u   + Ax_
            dvx  = dvx  - vx  + dx_
            dvy  = dvy  - vy  + dy_
            dvxx = dvxx - vxx + dxx_
            dvyy = dvyy - vyy + dyy_
            dvxy = dvxy - vxy + dxy_

            # x sub-problem
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

            # Recompute derivatives
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
            if verbose:
                print(f'{msg}isnr={isnr_val:.6f},beta={beta:.6f}')
        elif verbose:
            print(f'{msg}beta={beta:.6f}')

    x_fov = x[hks1:hks1 + M1, hks2:hks2 + M2]
    return x_fov, x, opt_out


# ═════════════════════════════════════════════════════════════════════════════
# FFT-related helpers for non-blind deconvolution
# ═════════════════════════════════════════════════════════════════════════════

_OPT_FFT_LUT = None


def _build_opt_fft_lut(max_n=4096):
    """Build LUT mapping n -> next efficient FFT size (products of 2,3,5,7)."""
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
    """Optimal FFT data length(s) — smallest efficient size >= n."""
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


# ═════════════════════════════════════════════════════════════════════════════
# wrap_boundary_liu (Liu & Jia ICIP 2008)
# ═════════════════════════════════════════════════════════════════════════════

def _solve_min_laplacian(boundary_image: np.ndarray) -> np.ndarray:
    """Solve Laplace eq. with Dirichlet BC via DST-I."""
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
    denom = (2.0 * np.cos(np.pi * xx / (W - 1)) - 2.0) + \
            (2.0 * np.cos(np.pi * yy / (H - 1)) - 2.0)

    f3 = f2sin / denom
    img_tt = idstn(f3, type=1)

    result = bi.copy()
    result[1:H-1, 1:W-1] = img_tt
    return result


def wrap_boundary_liu(img: np.ndarray, img_size: tuple) -> np.ndarray:
    """Pad image so boundaries are circularly smooth for FFT-based deconv."""
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

        A2 = _solve_min_laplacian(
            r_A[alpha - 1: alpha + H_w + 1, :])

        r_B = np.zeros((H, alpha * 2 + W_w), dtype=np.float64)
        r_B[:, :alpha] = HG[:, -alpha:]
        r_B[:, -alpha:] = HG[:, :alpha]
        if W_w > 1:
            b = np.arange(W_w, dtype=np.float64) / (W_w - 1)
        else:
            b = np.array([0.0])
        r_B[0, alpha:alpha + W_w] = (
            (1 - b) * r_B[0, alpha - 1] + b * r_B[0, -alpha])
        r_B[-1, alpha:alpha + W_w] = (
            (1 - b) * r_B[-1, alpha - 1] + b * r_B[-1, -alpha])

        B2 = _solve_min_laplacian(
            r_B[:, alpha - 1: alpha + W_w + 1])

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


# ═════════════════════════════════════════════════════════════════════════════
# TV deblurring (ADM anisotropic — Split Bregman)
# ═════════════════════════════════════════════════════════════════════════════

def _computeDenominator(B, k):
    """Pre-compute frequency-domain terms for ADM TV deblurring."""
    m, n = B.shape
    otf_k = psf2otf(k, (m, n))
    Nomin1 = np.conj(otf_k) * fft2(B)
    Denom1 = np.abs(otf_k) ** 2

    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    Denom2 = (np.abs(psf2otf(dx, (m, n))) ** 2 +
              np.abs(psf2otf(dy, (m, n))) ** 2)
    return Nomin1, Denom1, Denom2


def deblurring_adm_aniso(B, k, lambda_tv, alpha):
    """TV-l2 deblurring via ADM/Split Bregman with anisotropic TV."""
    beta = 1.0 / lambda_tv
    beta_min = 0.001
    m, n = B.shape
    I = B.copy()
    Nomin1, Denom1, Denom2 = _computeDenominator(B, k)

    Ix = np.concatenate([np.diff(I, axis=1),
                         I[:, 0:1] - I[:, -1:]], axis=1)
    Iy = np.concatenate([np.diff(I, axis=0),
                         I[0:1, :] - I[-1:, :]], axis=0)

    while beta > beta_min:
        gamma = 1.0 / (2.0 * beta)
        Denom = Denom1 + gamma * Denom2

        Wx = np.maximum(np.abs(Ix) - beta * lambda_tv, 0.0) * np.sign(Ix)
        Wy = np.maximum(np.abs(Iy) - beta * lambda_tv, 0.0) * np.sign(Iy)

        Wxx = np.concatenate([Wx[:, -1:] - Wx[:, 0:1],
                              -np.diff(Wx, axis=1)], axis=1)
        Wxx = Wxx + np.concatenate([Wy[-1:, :] - Wy[0:1, :],
                                     -np.diff(Wy, axis=0)], axis=0)

        Fyout = (Nomin1 + gamma * fft2(Wxx)) / Denom
        I = np.real(ifft2(Fyout))

        Ix = np.concatenate([np.diff(I, axis=1),
                             I[:, 0:1] - I[:, -1:]], axis=1)
        Iy = np.concatenate([np.diff(I, axis=0),
                             I[0:1, :] - I[-1:, :]], axis=0)
        beta = beta / 2.0

    return I


# ═════════════════════════════════════════════════════════════════════════════
# L0 gradient restoration
# ═════════════════════════════════════════════════════════════════════════════

def L0Restoration(Im, kernel, lambda_grad, kappa=2.0):
    """Image restoration with L0 gradient prior."""
    H_orig, W_orig = Im.shape[0], Im.shape[1]
    target_size = opt_fft_size(
        np.array([H_orig, W_orig]) + np.array(kernel.shape[:2]) - 1)
    Im_w = wrap_boundary_liu(Im, tuple(target_size))

    if Im_w.ndim == 2:
        Im_w = Im_w[:, :, np.newaxis]
    N, M, D = Im_w.shape

    S = Im_w.copy()
    betamax = 1e5

    fx = np.array([[1, -1]], dtype=np.float64)
    fy = np.array([[1], [-1]], dtype=np.float64)

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

        h = np.concatenate([np.diff(S, axis=1),
                            S[:, 0:1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, axis=0),
                            S[0:1, :, :] - S[-1:, :, :]], axis=0)

        grad_sq = np.sum(h ** 2 + v ** 2, axis=2)
        t = grad_sq < lambda_grad / beta_val
        t3 = np.tile(t[:, :, np.newaxis], (1, 1, D))
        h[t3] = 0
        v[t3] = 0

        Normin2 = np.concatenate([h[:, -1:, :] - h[:, 0:1, :],
                                  -np.diff(h, axis=1)], axis=1)
        Normin2 += np.concatenate([v[-1:, :, :] - v[0:1, :, :],
                                   -np.diff(v, axis=0)], axis=0)

        FS = (Normin1 + beta_val * fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))
        beta_val *= kappa

    S = S[:H_orig, :W_orig, :]
    if D == 1:
        S = S[:, :, 0]
    return S


# ═════════════════════════════════════════════════════════════════════════════
# Bilateral filter
# ═════════════════════════════════════════════════════════════════════════════

def _fspecial_gaussian(size, sigma):
    """2-D Gaussian kernel."""
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    h = np.outer(g, g)
    return h / h.sum()


def bilateral_filter(img, sigma_s, sigma):
    """Bilateral filter for grayscale images."""
    was_2d = img.ndim == 2
    if was_2d:
        img = img[:, :, np.newaxis]
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

    for yy in range(-fr, fr + 1):
        for xx in range(-fr, fr + 1):
            w_s = spatial_weight[yy + fr, xx + fr]
            n_img = p_img[fr + yy:fr + yy + h, fr + xx:fr + xx + w, :]
            n_lab = p_lab[fr + yy:fr + yy + h, fr + xx:fr + xx + w, :]
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


# ═════════════════════════════════════════════════════════════════════════════
# Ringing artifacts removal (Pan et al. CVPR 2014)
# ═════════════════════════════════════════════════════════════════════════════

def ringing_artifacts_removal(y, kernel, lambda_tv=1e-3,
                              lambda_l0=2e-3, weight_ring=1.0):
    """
    Non-blind deconvolution with ringing suppression.

    Uses TV deconv + L0 deconv + bilateral filter on their difference
    to identify and subtract ringing artifacts.

    Parameters
    ----------
    y           : (H, W) blurred image (single channel, float [0,1])
    kernel      : blur kernel
    lambda_tv   : TV regularisation weight
    lambda_l0   : L0 gradient prior weight
    weight_ring : ringing suppression strength (0 = TV only)

    Returns
    -------
    result : (H, W) deblurred image
    """
    H, W = y.shape[:2]
    target_size = opt_fft_size(
        np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
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
