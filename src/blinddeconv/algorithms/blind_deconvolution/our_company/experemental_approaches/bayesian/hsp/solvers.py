"""
solvers.py

Variational solvers for the HSP (Hyperbolic-Secant Prior) blind
deconvolution algorithm.

Ported from MATLAB files in the original toolbox of
    Castro-Macias et al., "Bayesian Blind Image Deconvolution using a
    Hyperbolic-Secant Prior" (2024).

Mapping:
    update_xf_alpha.m              -> update_xf_alpha
    update_kden.m                  -> update_kden
    single_stage_deconv_alphaden.m -> single_stage_deconv
    multi_stage_deconv_alphaden.m  -> multi_stage_deconv
    frils_deb_ubc.m                -> frils_deb_ubc (FIRLS+ADMM non-blind)

Only the 'Hysec' (hyperbolic-secant) prior branch is implemented; the
remaining priors (log, lp, exp, MoG, ECCV31) and the extra filter modes
(Bilateral / Mean / rmap / NLM) are deliberately omitted because the HSP
pipeline never invokes them.

Faithfulness notes (MATLAB vs Python):
    * MATLAB `eps` (== machine epsilon for double, 2.2204e-16) is used
      inside `tanh(eps + a*sqrt(g))/(eps + a*sqrt(g))` for numerical
      stability.  We replicate this exactly with `np.finfo(float).eps`.
    * `1/sigma2^2` appears literally throughout the MATLAB code
      (i.e. the inverse-square of the noise variance, *not* the usual
      `1/sigma2`).  This is preserved verbatim because the entire
      toolbox is calibrated to this convention.
    * `gammaf = alpha^2 * w^2` — likewise reproduces the MATLAB
      expression literally (not w directly as one might infer from
      eq. (26) of the paper).
    * Kernel reshape uses Fortran/column-major order because MATLAB
      `reshape` is column-major.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from .utils import (
    cg_solve,
    center_kernel,
    clean_kernel_ecp,
    conv2_same,
    conv2_valid,
    flipprev,
    im2col_sliding,
    imresize_bilinear,
    imresize_bicubic,
    otf2psf,
    pad_circular,
    pad_replicate,
    psf2otf,
    quadprog_nonneg,
)


_EPS = float(np.finfo(np.float64).eps)


# ═══════════════════════════════════════════════════════════════════════════
# update_xf_alpha.m  →  update_xf_alpha
# ═══════════════════════════════════════════════════════════════════════════

def update_xf_alpha(vars_: Dict, alpha: Sequence[float]) -> Dict:
    """
    E-step for the filtered image coefficients ``x_f`` (HS prior).

    Parameters
    ----------
    vars_ : dict
        Must contain: ``yf`` (list of 2-D arrays — filtered blurred image),
        ``k`` (current kernel), ``sigma2`` (noise variance), ``prior``
        (dict with ``name='Hysec'``).  May optionally provide
        ``MAX_INNER_ITER`` (default 5).
    alpha : sequence of float
        Per-filter prior shape parameter ``alpha[i]``.

    Returns
    -------
    vars_ : dict (modified)
        With updated ``xf``, ``gammaf``, ``xf_cov``.
    """
    yf_in = vars_['yf']
    k = vars_['k']
    sigma2 = float(vars_['sigma2'])
    prior = vars_['prior']

    if prior.get('name') != 'Hysec':
        raise NotImplementedError(
            "Only the 'Hysec' prior branch is implemented in this port."
        )

    tol = 1e-7
    max_inner_iter = int(vars_.get('MAX_INNER_ITER', 5))

    nofilters = len(yf_in)
    kN, kM = k.shape  # MATLAB: [kN, kM] = size(k)   (rows, cols)
    ph, pw = (kN - 1) // 2, (kM - 1) // 2

    # Pad each yf with replicate boundary and build the FOV mask
    yf_pad = [pad_replicate(y, ph, pw) for y in yf_in]
    N, M = yf_pad[0].shape
    mask = np.zeros((N, M), dtype=np.float64)
    mask[ph:N - ph, pw:M - pw] = 1.0

    # Initialisation (PROP_IMG_WITHIN_STAGES is always 0 in HSP defaults)
    xf = [y.copy() for y in yf_pad]

    rhs = [conv2_same(yf_pad[i] * mask, flipprev(k)) for i in range(nofilters)]
    gammaf = [1e4 * np.ones((N, M), dtype=np.float64) for _ in range(nofilters)]

    x_cov_k = (1.0 / sigma2 ** 2) * conv2_same(mask, np.abs(k) ** 2)

    inv_sigma2_sq = 1.0 / sigma2 ** 2
    flip_k = flipprev(k)

    xf_cov: list = [None] * nofilters  # type: ignore

    # CG parameters
    CG_TOL = 1e-5
    CG_MAX_ITER = 15

    for _ in range(max_inner_iter):
        x_conv = np.empty(nofilters, dtype=np.float64)

        for i in range(nofilters):
            xf_old = xf[i]
            weights = gammaf[i]

            def a_func(x, _w=weights, _flip_k=flip_k, _k=k, _mask=mask,
                       _scale=inv_sigma2_sq):
                inner = conv2_same(x, _k) * _mask
                return _scale * conv2_same(inner, _flip_k) + _w * x

            xf[i], _ = cg_solve(
                xf_old.copy(), a_func, inv_sigma2_sq * rhs[i],
                max_iter=CG_MAX_ITER, tol=CG_TOL,
            )

            # Covariance
            xf_cov[i] = 1.0 / (x_cov_k + gammaf[i])

            # Variational weight update (HS prior)
            sq = np.abs(xf[i]) ** 2 + xf_cov[i]
            sqr_gamma = np.sqrt(sq)
            val = alpha[i] * sqr_gamma
            w = np.tanh(_EPS + val) / (_EPS + val)
            gammaf[i] = (alpha[i] ** 2) * (w ** 2)

            # Convergence (relative change within FOV)
            denom = np.linalg.norm(xf_old * mask)
            if denom > 0:
                x_conv[i] = np.linalg.norm((xf[i] - xf_old) * mask) / denom
            else:
                x_conv[i] = np.inf

        if float(x_conv.mean()) < tol:
            break

    # Strip padding from xf, gammaf, xf_cov
    xf_out = [a[ph:N - ph, pw:M - pw] for a in xf]
    gammaf_out = [a[ph:N - ph, pw:M - pw] for a in gammaf]
    xf_cov_out = [a[ph:N - ph, pw:M - pw] for a in xf_cov]

    vars_['xf'] = xf_out
    vars_['gammaf'] = gammaf_out
    vars_['xf_cov'] = xf_cov_out
    return vars_


# ═══════════════════════════════════════════════════════════════════════════
# update_kden.m  →  update_kden
# ═══════════════════════════════════════════════════════════════════════════

def update_kden(vars_: Dict, enforce_sparsity: bool = True) -> Dict:
    """
    M-step for the blur kernel (positivity-constrained QP + IRLS).

    Builds the cumulative covariance system
        Ck =  Σ_i  (Tx_i Tx_i' + diag(Tcovx_i))
        bk =  Σ_i  (sum over patches of  Ty_i .* Tx_i)
    then solves min_h  0.5 h' Ck h - bk' h  s.t.  h ≥ 0.

    Optionally performs 2 IRLS reweighting iterations enforcing
    sparsity on the kernel with weight ``λ · max(|k|, thr0)^(-2)``,
    ``λ = 0.001``, ``thr0 = 1e-4``.
    """
    xf = vars_['xf']
    yf = vars_['yf']
    xf_cov = vars_['xf_cov']
    k_size = tuple(vars_['k_size'])           # (kN, kM)
    nofilters = int(vars_['nofilters'])

    kN, kM = int(k_size[0]), int(k_size[1])
    total_elem = kN * kM

    Ck = np.zeros((total_elem, total_elem), dtype=np.float64)
    bk = np.zeros(total_elem, dtype=np.float64)

    for i in range(nofilters):
        Tcovx = im2col_sliding(xf_cov[i], (kN, kM)).sum(axis=1)   # (kN*kM,)
        Tx = im2col_sliding(xf[i], (kN, kM))                       # (kN*kM, P)
        Ck += Tx @ Tx.T
        Ck[np.arange(total_elem), np.arange(total_elem)] += Tcovx

        # MATLAB: Ty = yf{i}((kN+1)/2 : end-(kN-1)/2, (kM+1)/2 : end-(kM-1)/2)
        # That is the valid centre region of yf with the same number of
        # patches as Tx.  (kN+1)/2 - 1 = (kN-1)//2 (0-based).
        ph, pw = (kN - 1) // 2, (kM - 1) // 2
        Ty = yf[i][ph:yf[i].shape[0] - ph, pw:yf[i].shape[1] - pw]
        # MATLAB-style column-major flatten — must match im2col patch order
        ty = Ty.flatten(order='F')
        bk += Tx @ ty

    # Solve positivity-constrained QP
    h = quadprog_nonneg(Ck, -bk)

    if enforce_sparsity:
        lam = 0.001
        thr0 = 1e-4
        for _ in range(2):
            diag_w = lam * np.maximum(np.abs(h), thr0) ** (-2)
            Ck_w = Ck.copy()
            Ck_w[np.arange(total_elem), np.arange(total_elem)] += diag_w
            h = quadprog_nonneg(Ck_w, -bk, x0=h)

    # MATLAB column-major reshape
    k = np.reshape(h, (kN, kM), order='F')
    k = flipprev(k)
    k[k < 0] = 0.0
    k = clean_kernel_ecp(k)
    s = k.sum()
    if s > 0:
        k = k / s
    vars_['k'] = k
    return vars_


# ═══════════════════════════════════════════════════════════════════════════
# single_stage_deconv_alphaden.m  →  single_stage_deconv
# ═══════════════════════════════════════════════════════════════════════════

def single_stage_deconv(vars_: Dict,
                        alpha: Sequence[float],
                        options: Dict | None = None,
                        ) -> Tuple[Dict, np.ndarray]:
    """
    Alternate ``update_xf_alpha`` and ``update_kden`` for one pyramid
    stage, looping over ``vars_['sigma2_vec']``.

    Returns
    -------
    vars_   : updated state dict
    k_hist  : 3-D ndarray of shape (kN, kM, MAX_ITER) — kernel history
    """
    if options is None:
        options = {}
    verbose = bool(options.get('verbose', False))

    sigma2_vec = np.asarray(vars_['sigma2_vec'], dtype=np.float64)
    max_iter = sigma2_vec.size

    k_hist = np.zeros(vars_['k'].shape + (max_iter,), dtype=np.float64)

    for it in range(max_iter):
        vars_['sigma2'] = float(sigma2_vec[it])
        if verbose:
            print(f'\tIteration {it + 1}/{max_iter}')

        vars_ = update_xf_alpha(vars_, alpha)
        vars_ = update_kden(vars_, enforce_sparsity=True)
        k_hist[:, :, it] = vars_['k']

    return vars_, k_hist


# ═══════════════════════════════════════════════════════════════════════════
# multi_stage_deconv_alphaden.m  →  multi_stage_deconv
# ═══════════════════════════════════════════════════════════════════════════

def _mat_round(x: float) -> int:
    """MATLAB-style round (half away from zero)."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def multi_stage_deconv(y: np.ndarray,
                       k_size: Tuple[int, int],
                       prior: Dict,
                       filters: Sequence[np.ndarray],
                       noise_var: float,
                       alpha: Sequence[float],
                       options: Dict | None = None,
                       ) -> Tuple[Dict, list]:
    """
    Coarse-to-fine pyramid blind deconvolution (BDGSP/HSP).

    Parameters
    ----------
    y : 2-D ndarray of float in [0, 1] — blurred grayscale image.
    k_size : (kN, kM)  — final kernel support, expected odd.
    prior  : {'name': 'Hysec', 'filter_name': 'None'}.
    filters : list of 2-D high-pass filters F_n.
    noise_var : initial noise variance (σ²).
    alpha : per-filter HS shape parameter, e.g. ``[10**2.4, 10**2.15]``.
    options : optional dict — supports
        ``verbose``, ``MAX_ITER`` (default 10), ``MAX_INNER_ITER`` (5),
        ``no_stages`` (None => auto), ``init_k``.

    Returns
    -------
    vars_     : final state dict (contains 'k', 'xf', ...).
    k_history : list of per-stage ``k_hist`` 3-D arrays.
    """
    if options is None:
        options = {}
    opt = {
        'verbose': False,
        'tol': 1e-8,
        'no_stages': None,
        'SHOW_IMGS': False,
        'UPDATE_NOISE': False,
        'PROP_IMG_BET_STAGES': False,
        'PROP_IMG_WITHIN_STAGES': False,
        'MAX_INNER_ITER': 5,
        'MAX_ITER': 10,
    }
    opt.update(options)

    vars_: Dict = dict(opt)

    upfac = np.sqrt(2.0)

    if opt['no_stages'] is None:
        NOSTAGES = max(int(np.floor(np.log(min(k_size) / 5.0) / np.log(upfac))), 1) + 1
    else:
        NOSTAGES = int(opt['no_stages'])

    if not filters:
        filters = [
            np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=np.float64),
            np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64),
        ]
    vars_['filters'] = list(filters)
    vars_['nofilters'] = len(filters)
    vars_['prior'] = prior

    # Kernel sizes at each pyramid level (rounded up, forced odd)
    all_kernel_sizes = np.zeros((NOSTAGES, 2), dtype=np.int64)
    for dim in range(2):
        for s in range(NOSTAGES):
            sz = int(np.ceil(k_size[dim] / upfac ** (NOSTAGES - 1 - s)))
            if sz % 2 == 0:
                sz += 1
            all_kernel_sizes[s, dim] = sz

    # Initial kernel — 2×2 block of ones in (approximately) the centre
    if 'init_k' in opt and opt['init_k'] is not None:
        k_init = np.asarray(opt['init_k'], dtype=np.float64)
        k_init = k_init / k_init.sum()
    else:
        k_init = np.zeros(tuple(all_kernel_sizes[0]), dtype=np.float64)
        c1 = _mat_round(all_kernel_sizes[0, 0] / 2.0)  # 1-based MATLAB index
        c2 = _mat_round(all_kernel_sizes[0, 1] / 2.0)
        # Convert to 0-based and place a 2×2 block at (c-1, c-1)..(c, c)
        a, b = c1 - 1, c2 - 1
        k_init[a, b] = 1.0
        k_init[a, b + 1] = 1.0
        k_init[a + 1, b] = 1.0
        k_init[a + 1, b + 1] = 1.0
        k_init = k_init / k_init.sum()

    vars_['k'] = k_init
    vars_['k_size'] = tuple(all_kernel_sizes[0])

    # Annealed noise-variance schedule (decreasing across inner iters)
    MAX_ITER = int(opt['MAX_ITER'])
    vars_['sigma2_vec'] = noise_var * (1.15 ** np.arange(MAX_ITER, 0, -1))

    k_history: list = []

    for it in range(NOSTAGES):
        if opt['verbose']:
            print(f'Stage {it + 1}/{NOSTAGES}')

        # Downsample observation to current scale.
        ys = imresize_bilinear(y, upfac ** (it - (NOSTAGES - 1)))

        # Filtered blurred image yf{i}: replicate-pad then valid conv.
        yf = []
        for f in vars_['filters']:
            fN, fM = f.shape
            ph, pw = (fN - 1) // 2, (fM - 1) // 2
            ys_pad = pad_replicate(ys, ph, pw)
            yf.append(conv2_valid(ys_pad, f))
        vars_['yf'] = yf

        if it > 0:
            # Upsample kernel to next-stage size, clip negatives, normalise.
            # MATLAB calls `imresize(k, sz)` without method => default bicubic.
            k_up = imresize_bicubic(vars_['k'], tuple(all_kernel_sizes[it]))
            k_up[k_up < 0] = 0.0
            s = k_up.sum()
            if s > 0:
                k_up = k_up / s
            vars_['k'] = k_up
            vars_['k_size'] = tuple(all_kernel_sizes[it])

            # Centre the kernel.  PROP_IMG_BET_STAGES is False by default,
            # so we don't propagate xf between stages.
            vars_['k'], _ = center_kernel(vars_['k'])

        vars_, k_hist = single_stage_deconv(vars_, alpha, opt)
        k_history.append(k_hist)

    return vars_, k_history


# ═══════════════════════════════════════════════════════════════════════════
# frils_deb_ubc.m  →  frils_deb_ubc
# ═══════════════════════════════════════════════════════════════════════════

# First- and second-order derivative filters used by FIRLS.
_DXF = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float64)
_DYF = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
_DXXF = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
_DYYF = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)
_DXYF = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]], dtype=np.float64)


def _conv2_valid_circ(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Equivalent to MATLAB conv2(padarray(x,[1 1],'circular'), f, 'valid')."""
    xp = pad_circular(x, 1, 1)
    return conv2_valid(xp, f)


def frils_deb_ubc(y: np.ndarray,
                  h: np.ndarray,
                  opt: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Fast IRLS + ADMM non-blind deconvolution with undetermined boundary.

    Direct port of ``frils_deb_ubc.m``.  Kernel must have odd dimensions.

    Required ``opt`` keys:
        lambda, alpha, beta_a, lambda_u,
        epsilon_min, epsilon_max,
        out_iter, inner_iter, IF.
    Optional keys: ``isnr_display``, ``cost_display`` (booleans;
    diagnostic-only — ignored here for speed).
    """
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1, hks2 = m1 // 2, m2 // 2
    n1, n2 = M1 + m1 - 1, M2 + m2 - 1

    x = pad_replicate(y, hks1, hks2)

    dxfr = np.rot90(_DXF, 2)
    dyfr = np.rot90(_DYF, 2)
    dxxfr = np.rot90(_DXXF, 2)
    dyyfr = np.rot90(_DYYF, 2)
    dxyfr = np.rot90(_DXYF, 2)

    H = psf2otf(h, (n1, n2))
    Ht = np.conj(H)
    Hx = psf2otf(_DXF, (n1, n2))
    Hy = psf2otf(_DYF, (n1, n2))
    Hxx = psf2otf(_DXXF, (n1, n2))
    Hyy = psf2otf(_DYYF, (n1, n2))
    Hxy = psf2otf(_DXYF, (n1, n2))

    HH = H * Ht
    HHx = Hx * np.conj(Hx)
    HHy = Hy * np.conj(Hy)
    HHxx = Hxx * np.conj(Hxx)
    HHyy = Hyy * np.conj(Hyy)
    HHxy = Hxy * np.conj(Hxy)

    RR = HHx + HHy + HHxx + HHyy + HHxy

    lam = float(opt['lambda'])
    alpha = float(opt['alpha'])
    beta_a = float(opt['beta_a'])
    lambda_u = min(float(opt['lambda_u']), 5000.0 * beta_a)
    w0 = 0.25

    epsilon_min = float(opt['epsilon_min'])
    epsilon_max = float(opt['epsilon_max'])
    N1 = int(opt['out_iter'])
    N2 = int(opt['inner_iter'])
    IF = float(opt['IF'])

    c = alpha * lam
    beta_min = alpha * lam / (epsilon_max ** (2.0 - alpha))
    beta_max = alpha * lam / (epsilon_min ** (2.0 - alpha))
    beta = beta_min

    dx = _conv2_valid_circ(x, _DXF)
    dy = _conv2_valid_circ(x, _DYF)
    dxx = _conv2_valid_circ(x, _DXXF)
    dyy = _conv2_valid_circ(x, _DYYF)
    dxy = _conv2_valid_circ(x, _DXYF)

    adx, ady = np.abs(dx), np.abs(dy)
    adxx, adyy, adxy = np.abs(dxx), np.abs(dyy), np.abs(dxy)

    du = np.zeros((n1, n2), dtype=np.float64)
    dvx = du.copy(); dvy = du.copy()
    dvxx = du.copy(); dvyy = du.copy(); dvxy = du.copy()

    X = np.fft.fft2(x)
    Ax = np.real(np.fft.ifft2(H * X))
    invA = HH + (beta_a / lambda_u) * RR

    eps_small = 1e-12  # guard against pure-zero adx^(α-2) → ∞

    for _ in range(N1):
        Wx = np.minimum(beta, c * np.maximum(adx, eps_small) ** (alpha - 2.0))
        Wy = np.minimum(beta, c * np.maximum(ady, eps_small) ** (alpha - 2.0))
        Wxx = np.minimum(beta, c * np.maximum(adxx, eps_small) ** (alpha - 2.0)) * w0
        Wyy = np.minimum(beta, c * np.maximum(adyy, eps_small) ** (alpha - 2.0)) * w0
        Wxy = np.minimum(beta, c * np.maximum(adxy, eps_small) ** (alpha - 2.0)) * w0

        for _i in range(N2):
            # u sub-problem
            u = Ax + du
            u[hks1:n1 - hks1, hks2:n2 - hks2] = (
                (y + lambda_u * u[hks1:n1 - hks1, hks2:n2 - hks2])
                / (1.0 + lambda_u)
            )

            # v sub-problems
            vx = beta_a * (dx + dvx) / (Wx + beta_a)
            vy = beta_a * (dy + dvy) / (Wy + beta_a)
            vxx = beta_a * (dxx + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy + dvxy) / (Wxy + beta_a)

            # Dual updates
            du = du - u + Ax
            dvx = dvx - vx + dx
            dvy = dvy - vy + dy
            dvxx = dvxx - vxx + dxx
            dvyy = dvyy - vyy + dyy
            dvxy = dvxy - vxy + dxy

            # x sub-problem
            Y = np.fft.fft2(u - du) * Ht

            tempx = _conv2_valid_circ(vx - dvx, dxfr)
            tempy = _conv2_valid_circ(vy - dvy, dyfr)
            tempxx = _conv2_valid_circ(vxx - dvxx, dxxfr)
            tempyy = _conv2_valid_circ(vyy - dvyy, dyyfr)
            tempxy = _conv2_valid_circ(vxy - dvxy, dxyfr)

            X = Y + (beta_a / lambda_u) * np.fft.fft2(
                tempx + tempy + tempxx + tempyy + tempxy
            )
            X = X / invA
            Ax = np.real(np.fft.ifft2(H * X))
            x = np.real(np.fft.ifft2(X))

            dx = _conv2_valid_circ(x, _DXF)
            dy = _conv2_valid_circ(x, _DYF)
            dxx = _conv2_valid_circ(x, _DXXF)
            dyy = _conv2_valid_circ(x, _DYYF)
            dxy = _conv2_valid_circ(x, _DXYF)
            adx, ady = np.abs(dx), np.abs(dy)
            adxx, adyy, adxy = np.abs(dxx), np.abs(dyy), np.abs(dxy)

        beta = min(beta * IF, beta_max)

    x_fov = x[hks1:n1 - hks1, hks2:n2 - hks2]
    return x_fov, x, opt


__all__ = [
    'update_xf_alpha',
    'update_kden',
    'single_stage_deconv',
    'multi_stage_deconv',
    'frils_deb_ubc',
]
