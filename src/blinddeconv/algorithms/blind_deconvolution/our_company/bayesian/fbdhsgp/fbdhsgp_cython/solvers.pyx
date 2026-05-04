"""
solvers.py

Core solver functions for the FBDHSGP blind-deconvolution algorithm.

Ported from the MATLAB reference implementation (folder ``FBDHSGP/``)
accompanying the paper:

    X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
    "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
    Digital Signal Processing, 2016.

Mapping to MATLAB sources
-------------------------
    h_update          ←  h_update.m              (KKT fixed-point QP)
    h_admm_ubc_bi     ←  h_admm_ubc_bi.m         (Algorithm 2: kernel ADMM)
    x_admm_ubc_bi     ←  x_admm_ubc_bi.m         (Algorithm 1: image ADMM + IRLS)
    ss_deb            ←  ss_deb.m                (single-scale alternating BID)
    frils_deb_ubc     ←  frils_deb_ubc.m         (final non-blind ℓ_p deconvolution)

MATLAB → Python conversion notes
--------------------------------
    * MATLAB ``conv2(A, B, 'valid')`` performs TRUE convolution (flips B).
      Equivalent to ``scipy.signal.convolve2d(A, B, mode='valid')``.
    * MATLAB ``rot90(A, 2)`` = both-axis flip → ``A[::-1, ::-1]``.
    * Forward difference with circular wrap (``dx`` in MATLAB)::
          dx = [diff(x,1,2),  x(:,1)-x(:,n2)]
      In NumPy::
          dx = np.concatenate([np.diff(x, axis=1),
                               x[:, :1] - x[:, -1:]], axis=1)
      Similarly for ``dy`` along axis 0.
    * Backward difference with circular wrap (``tempx`` in MATLAB)::
          tempx = [tempx(:,1)-tempx(:,n2),  diff(tempx,1,2)]
      In NumPy::
          tempx = np.concatenate([tempx[:, :1] - tempx[:, -1:],
                                  np.diff(tempx, axis=1)], axis=1)
    * MATLAB struct fields are stored as dict entries.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d

# ---------------------------------------------------------------------------
# MOG (Mixture-of-Gaussians) prior parameters — Levin et al., CVPR 2009.
# Originally stored in MOGparams.mat shipped with the MATLAB reference code.
# Values extracted once so the Python port has no dependency on any .mat file.
# pis   : mixture weights (sum ≈ 1)
# ivars : inverse variances of each Gaussian component
# ---------------------------------------------------------------------------
_MOG_PIS = np.array(
    [0.304710113647604, 0.4343635506463882, 0.2609263357060055],
    dtype=np.float64,
)
_MOG_IVARS = np.array(
    [7021.680498597095, 471.8414210224951, 41.848208684280294],
    dtype=np.float64,
)

from .utils import (
    psf2otf,
    otf2psf,
    pad_replicate,
    pad_zeros,
    getindex,
)


# =============================================================================
# Difference helpers (forward / backward with circular wrap)
# =============================================================================

def _fdiff_x(x: np.ndarray) -> np.ndarray:
    """Forward x-difference with circular wrap.  MATLAB: ``[diff(x,1,2), x(:,1)-x(:,n2)]``."""
    return np.concatenate([np.diff(x, axis=1), x[:, :1] - x[:, -1:]], axis=1)


def _fdiff_y(x: np.ndarray) -> np.ndarray:
    """Forward y-difference with circular wrap.  MATLAB: ``[diff(x,1,1); x(1,:)-x(n1,:)]``."""
    return np.concatenate([np.diff(x, axis=0), x[:1, :] - x[-1:, :]], axis=0)


def _bdiff_x(x: np.ndarray) -> np.ndarray:
    """Backward x-difference with circular wrap.  MATLAB: ``[x(:,1)-x(:,n2), diff(x,1,2)]``."""
    return np.concatenate([x[:, :1] - x[:, -1:], np.diff(x, axis=1)], axis=1)


def _bdiff_y(x: np.ndarray) -> np.ndarray:
    """Backward y-difference with circular wrap.  MATLAB: ``[x(:,1)-x(n1,:); diff(x,1,1)]``."""
    return np.concatenate([x[:1, :] - x[-1:, :], np.diff(x, axis=0)], axis=0)


def _circshift_left(a: np.ndarray) -> np.ndarray:
    """``[a(:,2:n2), a(:,1)]`` — shift columns left by one with wrap."""
    return np.concatenate([a[:, 1:], a[:, :1]], axis=1)


def _circshift_up(a: np.ndarray) -> np.ndarray:
    """``[a(2:n1,:); a(1,:)]`` — shift rows up by one with wrap.

    Note
    ----
    The MATLAB source actually has a typo at the *post-loop* update: it writes
    ``[cov_img(n1,:); cov_img(1:n1-1,:)]`` (a downward shift) for one branch
    and ``[cov_img(2:n1,:); cov_img(1,:)]`` (an upward shift) for another.
    We reproduce the in-loop ("upward") variant via this helper.
    """
    return np.concatenate([a[1:, :], a[:1, :]], axis=0)


def _circshift_right(a: np.ndarray) -> np.ndarray:
    """``[a(:,n2), a(:,1:n2-1)]`` — shift columns right by one with wrap."""
    return np.concatenate([a[:, -1:], a[:, :-1]], axis=1)


def _circshift_down(a: np.ndarray) -> np.ndarray:
    """``[a(n1,:); a(1:n1-1,:)]`` — shift rows down by one with wrap."""
    return np.concatenate([a[-1:, :], a[:-1, :]], axis=0)


# =============================================================================
# h_update  (h_update.m)
# =============================================================================

def h_update(
    d: np.ndarray,
    b: np.ndarray,
    tol: float,
    max_iter: int,
    alpha: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve the quadratic program::

        argmin_h  0.5 h^T diag(d) h - b^T h
        subject to  h >= 0  and  sum(h) = 1

    by a fixed-point iteration on the KKT multipliers ``alpha`` (>= 0,
    one per element, for non-negativity) and ``beta`` (scalar, for the
    sum-to-one constraint).  Direct port of MATLAB ``h_update.m``.

    Returns
    -------
    h     : the (signed) solution ``temp + beta/d`` — caller is expected
            to clip & renormalise (mirrors MATLAB code path).
    alpha : final non-negativity multipliers (warm-start for next call).
    beta  : final sum-to-one multiplier (returned for parity, unused).
    """
    mb = -b
    if alpha is not None:
        alpha = np.maximum(alpha, 0.0)
    else:
        alpha = np.maximum(mb, 0.0)

    sd = np.sum(1.0 / d)
    beta = 0.0
    beta_old = np.nan
    temp = (b + alpha) / d  # first eval; will be re-computed inside loop

    for it in range(1, max_iter + 1):
        temp = (b + alpha) / d
        beta = (1.0 - temp.sum()) / sd
        alpha = np.maximum(mb - beta, 0.0)
        if it > 1 and abs(beta_old - beta) < tol:
            break
        beta_old = beta

    h = temp + beta / d
    return h, alpha, beta


# =============================================================================
# h_admm_ubc_bi  (h_admm_ubc_bi.m)  ─── Algorithm 2 ────────────────────────────
# =============================================================================

def h_admm_ubc_bi(
    y: np.ndarray,
    X: np.ndarray,
    Cx: np.ndarray,
    vars_h: Dict,
) -> Dict:
    """
    Fast kernel estimation by ADMM (Algorithm 2 of the paper).

    Solves
    ------
        min_h  ||H ∘ X − F y||² + h^T D_x h
        s.t.   h(i) >= 0,  Σ h(i) = 1,
        with the splitting  H = F P h.

    Parameters
    ----------
    y       : ``yye`` — observed image augmented with the boundary tile
              ``ye``, shape (n1, n2).
    X       : Fourier transform of the padded latent image, shape (n1, n2).
    Cx      : (ks1, ks2) diagonal-approx of the data-driven matrix ``D_x``.
    vars_h  : dict with at least ``h`` (current PSF, ks1×ks2),
              ``beta_H`` (penalty), ``h_iter`` (max iterations),
              ``delta`` (relative-change stop), ``lambda_h`` (None or scalar),
              optional warm-start ``H`` and ``dH``.

    Returns
    -------
    Updated ``vars_h`` (with new ``h`` and ``dH``).
    """
    m, n = X.shape
    beta_H = float(vars_h["beta_H"])
    h_iter = int(vars_h["h_iter"])

    h = vars_h["h"].astype(np.float64, copy=True)
    Xc = np.conj(X)
    XcX = Xc * X
    lambda_h = vars_h.get("lambda_h", None)

    M = float(Cx.mean())
    beta_H = beta_H * M

    ks1, ks2 = h.shape

    H = vars_h.get("H", None)
    if H is None:
        H = psf2otf(h, (m, n))
    dH = vars_h.get("dH", None)
    if dH is None:
        dH = np.zeros((m, n), dtype=np.float64)

    FPh = H.copy()

    D = 1.0 + (1.0 / beta_H) * Cx
    Ye = fft2(y)

    h_old = h.copy()
    alpha0: np.ndarray | None = None
    rv_h = np.inf

    for i in range(1, h_iter + 1):
        # --- update H -------------------------------------------------------
        H = (Xc * Ye + beta_H * (FPh - dH)) / (XcX + beta_H)

        # --- update h -------------------------------------------------------
        b = np.real(otf2psf(H + dH, (ks1, ks2)))

        if lambda_h is None or (np.ndim(lambda_h) == 0 and lambda_h is None):
            N = 100
            tol = 1e-8
            h_new, alpha0, _ = h_update(D, b, tol, N, alpha0)
            h_new = np.where(h_new < 0.0, 0.0, h_new)
            s = h_new.sum()
            if s > 0:
                h_new = h_new / s
            h = h_new
        else:
            h = np.maximum(0.0, (b - float(lambda_h)) / D)

        FPh = psf2otf(h, (m, n))

        # --- update dual ----------------------------------------------------
        dH = dH + H - FPh

        # --- stopping criterion --------------------------------------------
        if i > 1:
            denom = np.linalg.norm(h_old, "fro")
            rv_h = (
                np.linalg.norm(h - h_old, "fro") / denom if denom > 0 else 0.0
            )
            if rv_h < float(vars_h["delta"]):
                break
        h_old = h.copy()

    h = np.where(h < 0.0, 0.0, h)
    s = h.sum()
    if s > 0:
        h = h / s

    vars_h["h"] = h
    vars_h["dH"] = dH
    vars_h["H"] = H
    vars_h["last_rv_h"] = rv_h
    return vars_h


# =============================================================================
# x_admm_ubc_bi  (x_admm_ubc_bi.m)  ─── Algorithm 1 ────────────────────────────
# =============================================================================

def _compute_weights(
    prior: Dict,
    Edx: np.ndarray,
    Edy: np.ndarray,
    lambda_: float,
    beta: float,
    t: float,
    alpha: float,
    iter_idx: int,
    n_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute IRLS weights ``Wx``, ``Wy`` for one of the supported priors.

    Mirrors the ``switch prior.name`` block inside ``x_admm_ubc_bi.m``.
    """
    name = prior["name"]
    const_weight_iter = prior.get("const_weight", 0)

    # The "iter == const_weight" branch — used to seed weights with a large
    # constant (rarely triggered with default settings, kept for parity).
    if const_weight_iter == iter_idx and const_weight_iter > 0:
        Wx = lambda_ * t * 1e4 * np.ones(n_shape, dtype=np.float64)
        Wy = Wx.copy()
        return Wx, Wy

    if name == "Log":
        Wx = np.minimum(beta, lambda_ / Edx) * t
        Wy = np.minimum(beta, lambda_ / Edy) * t
        return Wx, Wy

    if name == "Lp":
        Wx = np.minimum(beta, lambda_ * Edx ** (alpha / 2.0 - 1.0)) * t
        Wy = np.minimum(beta, lambda_ * Edy ** (alpha / 2.0 - 1.0)) * t
        return Wx, Wy

    if name == "MOG":
        # Mixture-of-Gaussians prior — Levin et al. (CVPR 2009) parameters
        # embedded as module constants; no external .mat file required.
        pis = _MOG_PIS
        ivars = _MOG_IVARS
        px = np.zeros_like(Edx)
        py = np.zeros_like(Edy)
        dpx = np.zeros_like(Edx)
        dpy = np.zeros_like(Edy)
        for k in range(len(pis)):
            edxs = np.exp(-Edx * ivars[k] / 2.0)
            edys = np.exp(-Edy * ivars[k] / 2.0)
            const = pis[k] * np.sqrt(ivars[k] / (2.0 * np.pi))
            px = const * edxs + px
            py = const * edys + py
            dpx = (lambda_ * const * ivars[k] * t) * edxs + dpx
            dpy = (lambda_ * const * ivars[k] * t) * edys + dpy
        Wx = dpx / px
        Wy = dpy / py
        return Wx, Wy

    if name == "NL1":
        Ex = 0.01
        Ey = Ex
        Wx = lambda_ * Ex / (Edx * (Edx + Ex) ** 2)
        Wy = lambda_ * Ey / (Edy * (Edy + Ey) ** 2)
        return Wx, Wy

    raise ValueError(f"Unknown prior name: {name!r}")


def x_admm_ubc_bi(
    y: np.ndarray,
    Y: np.ndarray,
    h: np.ndarray,
    vars_x: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    ADMM for image deblurring with general sparse (HSG) priors and
    undetermined boundary conditions (Algorithm 1 of the paper).

    Parameters
    ----------
    y       : observed (single-scale) image, shape (M1, M2).
    Y       : ``fft2`` of zero-padded ``y``, shape (n1, n2).
    h       : current blur kernel (odd sized), shape (m1, m2).
    vars_x  : dict carrying algorithm parameters and warm-start state.

    Returns
    -------
    x_fov   : restored image cropped to the FOV (M1, M2).
    x       : padded restored image (n1, n2).
    vars_x  : updated state (with X, H, ye, dvx, dvy, cov_img, Cx).
    """
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    # --- initialise x (warm-start when available) ---------------------------
    x0 = vars_x.get("x0", None)
    if x0 is None:
        x = pad_replicate(y, hks1, hks2)
    else:
        x = x0.astype(np.float64, copy=True)

    # --- precompute RR (eigenvalues of D_x^T D_x + D_y^T D_y) ---------------
    RR = vars_x.get("RR", None)
    if RR is None:
        dxf = np.array([[1.0, -1.0]])
        dyf = dxf.T
        Hx_otf = psf2otf(dxf, (n1, n2))
        Hy_otf = psf2otf(dyf, (n1, n2))
        RR = (Hx_otf * np.conj(Hx_otf) + Hy_otf * np.conj(Hy_otf)).real

    H = psf2otf(h, (n1, n2))
    Ht = np.conj(H)
    HH = (H * Ht).real
    hr = h[::-1, ::-1]
    prior = vars_x["priors"]

    # --- algorithm parameters ----------------------------------------------
    epsilon_min = float(vars_x["epsilon_min"])
    sigma = float(vars_x["sigma"])
    alpha = float(vars_x["alpha"])
    beta_v = float(vars_x["beta_v"])
    lambda_ = sigma ** 2
    delta_x = float(vars_x["delta_x"])
    if beta_v > 2.0:
        beta_v = beta_v * lambda_

    # --- covariance image (cov_img) initialisation -------------------------
    cov_img = vars_x.get("cov_img", None)
    if cov_img is None:
        h_norm2 = float((h * h).sum())
        cov_img = h_norm2 / max(lambda_, 1e-30) + 4e4 * np.ones((n1, n2))
        cov_img = 1.0 / cov_img

    K1 = int(vars_x["K1"])
    K2 = int(vars_x["K2"])

    beta = lambda_ / (epsilon_min ** (2.0 - alpha))

    dx = _fdiff_x(x)
    dy = _fdiff_y(x)

    # --- dual variables (warm-start) ----------------------------------------
    if "dvx" in vars_x and vars_x["dvx"] is not None:
        dvx = vars_x["dvx"].astype(np.float64, copy=True)
        dvy = vars_x["dvy"].astype(np.float64, copy=True)
    else:
        dvx = np.zeros((n1, n2), dtype=np.float64)
        dvy = np.zeros((n1, n2), dtype=np.float64)

    ye = np.zeros((n1, n2), dtype=np.float64)
    hrye = np.zeros((n1, n2), dtype=np.float64)

    # 4-tile buffers
    tiles = [
        np.zeros((n1, hks2), dtype=np.float64),
        np.zeros((n1, hks2), dtype=np.float64),
        np.zeros((hks1, n2 - 2 * hks2), dtype=np.float64),
        np.zeros((hks1, n2 - 2 * hks2), dtype=np.float64),
    ]

    index1, index2, index3, index4 = getindex(n1, n2, hks1, hks2)

    HtY = Y * Ht
    x_old = x.copy()
    invA = HH + beta_v * RR
    totiter = 0

    Id = np.ones((m1, m2), dtype=np.float64)
    t = float(prior.get("alpha", 1.0))

    Wx = np.zeros((n1, n2), dtype=np.float64)
    Wy = np.zeros((n1, n2), dtype=np.float64)
    X = HtY.copy()  # ensure X is defined for the return path

    for it in range(1, K1 + 1):
        # --- compute E[(∇x)^2] ------------------------------------------
        Edx = dx ** 2 + cov_img + _circshift_left(cov_img)
        Edy = dy ** 2 + cov_img + _circshift_up(cov_img)

        Wx, Wy = _compute_weights(
            prior, Edx, Edy, lambda_, beta, t, alpha, it, (n1, n2)
        )

        # --- update covariance image ------------------------------------
        h_norm2 = float((h * h).sum())
        if int(prior.get("conv", 0)) == 1:
            shift_Wx = _circshift_left(Wx)
            shift_Wy = _circshift_up(Wy)
            cov_img = h_norm2 + Wx + Wy + shift_Wx + shift_Wy
        else:
            cov_img = h_norm2 + 2.0 * (Wx + Wy)
        cov_img = (sigma ** 2) / cov_img

        # --- inner ADMM (K2 steps) --------------------------------------
        for _ in range(K2):
            totiter += 1

            # ye sub-problem (4-tile UBC apply of H)
            mu = 0.0 if totiter == 1 else 1.0
            for i_t in range(4):
                rows, cols = index1[i_t]
                xpadc = x[np.ix_(rows, cols)]
                tile = convolve2d(xpadc, h, mode="valid")
                tiles[i_t] = tile + mu * (tile - tiles[i_t])
                wr, wc = index2[i_t]
                ye[np.ix_(wr, wc)] = tiles[i_t]

            # H^T ye sub-problem (4-tile)
            for i_t in range(4):
                rows, cols = index3[i_t]
                yepadc = ye[np.ix_(rows, cols)]
                wr, wc = index4[i_t]
                hrye[np.ix_(wr, wc)] = convolve2d(yepadc, hr, mode="valid")

            # v sub-problem
            vx = beta_v * (dx - dvx) / (Wx + beta_v)
            vy = beta_v * (dy - dvy) / (Wy + beta_v)

            # dual update
            dvx = dvx + vx - dx
            dvy = dvy + vy - dy

            # x sub-problem
            tempx = vx + dvx
            tempy = vy + dvy
            tempx = _bdiff_x(tempx)
            tempy = _bdiff_y(tempy)

            X = HtY + fft2(hrye - beta_v * (tempx + tempy))
            X = X / invA
            x = np.real(ifft2(X))

            dx = _fdiff_x(x)
            dy = _fdiff_y(x)

        # --- outer (IRLS) stopping check --------------------------------
        denom = np.linalg.norm(x_old, "fro")
        rvx = np.linalg.norm(x - x_old, "fro") / denom if denom > 0 else 0.0

        # --- iter_callback hook (x-loop) --------------------------------
        cb = vars_x.get("iter_callback", None)
        if cb is not None:
            try:
                cb({
                    "scope": "x_admm",
                    "iter": it,
                    "K1": K1,
                    "rvx": float(rvx),
                    "sigma": sigma,
                    "beta_v": beta_v,
                    "Wx_mean": float(Wx.mean()),
                    "Wy_mean": float(Wy.mean()),
                })
            except Exception:
                pass

        if rvx < delta_x:
            break
        x_old = x.copy()

    # --- post-loop weight refresh + cov_img update --------------------------
    Edx = dx ** 2 + cov_img + _circshift_right(cov_img)
    Edy = dy ** 2 + cov_img + _circshift_down(cov_img)

    Wx, Wy = _compute_weights(
        prior, Edx, Edy, lambda_, beta, t, alpha,
        iter_idx=K1 + 1,  # never matches const_weight (default 0)
        n_shape=(n1, n2),
    )

    h_norm2 = float((h * h).sum())
    if int(prior.get("conv", 0)) == 1:
        shift_Wx = _circshift_left(Wx)
        shift_Wy = _circshift_up(Wy)
        cov_img = h_norm2 + Wx + Wy + shift_Wx + shift_Wy
    else:
        cov_img = h_norm2 + 2.0 * (Wx + Wy)
    cov_img = (sigma ** 2) / cov_img
    vars_x["cov_img"] = cov_img

    # diagonal (scalar) approximation of D_x — see Eq. (27) discussion
    Cx_scalar = float(cov_img[hks1:M1 + hks1, hks2:M2 + hks2].sum())
    vars_x["Cx"] = Cx_scalar * Id

    x_fov = x[hks1:n1 - hks1, hks2:n2 - hks2]
    vars_x["X"] = X
    vars_x["H"] = H
    vars_x["ye"] = ye
    vars_x["dvx"] = dvx
    vars_x["dvy"] = dvy
    vars_x["RR"] = RR

    return x_fov, x, vars_x


# =============================================================================
# ss_deb  (ss_deb.m)  ─── single-scale alternating BID ─────────────────────────
# =============================================================================

def ss_deb(
    y: np.ndarray,
    xvars: Dict,
    hvars: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-scale blind deconvolution: alternates ``x_admm_ubc_bi`` and
    ``h_admm_ubc_bi`` for ``xh_iter`` iterations with annealed noise sigma.
    """
    xh_iter = int(xvars["xh_iter"])
    sigma0 = float(xvars["sigma"])
    h = hvars["h"].astype(np.float64, copy=True)
    h_old = h.copy()

    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    dxf = np.array([[1.0, -1.0]])
    dyf = dxf.T
    Hx_otf = psf2otf(dxf, (n1, n2))
    Hy_otf = psf2otf(dyf, (n1, n2))
    RR = (Hx_otf * np.conj(Hx_otf) + Hy_otf * np.conj(Hy_otf)).real
    xvars["RR"] = RR

    ypad = pad_zeros(y, hks1, hks2)
    Y = fft2(ypad)

    sigma2_max = max(sigma0 ** 2, 0.0004)
    eta = (sigma2_max / sigma0 ** 2) ** 0.2

    x_fov = None
    for i in range(1, xh_iter + 1):
        sigma = max(np.sqrt(sigma2_max * eta ** (-i)), sigma0)

        if int(xvars.get("x_warm_start", 1)) == 0:
            xvars.pop("x0", None)

        xvars["sigma"] = sigma
        hvars["sigma"] = sigma

        x_fov, x_pad, xvars = x_admm_ubc_bi(y, Y, hvars["h"], xvars)
        xvars["x0"] = x_pad
        hvars["H"] = xvars["H"]

        yye = xvars["ye"] + ypad
        hvars = h_admm_ubc_bi(yye, xvars["X"], xvars["Cx"], hvars)
        h = hvars["h"]

        rv_h_outer = 0.0
        if i >= 2:
            denom = np.linalg.norm(h_old, "fro")
            rv_h_outer = (
                np.linalg.norm(h - h_old, "fro") / denom if denom > 0 else 0.0
            )

        # --- iter_callback hook (ss_deb outer loop) ---------------------
        cb = xvars.get("iter_callback", None)
        if cb is not None:
            try:
                cb({
                    "scope": "ss_deb",
                    "iter": i,
                    "xh_iter": xh_iter,
                    "sigma": float(sigma),
                    "rv_h": float(rv_h_outer),
                    "h_min": float(h.min()),
                    "h_max": float(h.max()),
                    "h_sum": float(h.sum()),
                    "kernel": h.copy(),
                })
            except Exception:
                pass

        if i >= 2 and rv_h_outer < 0.005:
            h = h_old
            hvars["h"] = h
            break

        h_old = h.copy()

    return x_fov, h


# =============================================================================
# frils_deb_ubc  (frils_deb_ubc.m)  ─── final non-blind ℓ_p deconvolution ─────
# =============================================================================

def frils_deb_ubc(y: np.ndarray, h: np.ndarray, opt: Dict) -> np.ndarray:
    """
    Fast IRLS for non-blind image deblurring with **undetermined boundary
    conditions** and an ℓ_p (Huber) sparsity prior on first- and
    second-order derivatives.

    Used as the final image-reconstruction step (Eq. (31) of the paper).

    Parameters
    ----------
    y    : observed (single-channel) image, shape (M1, M2).
    h    : final blur-kernel estimate (odd sized), shape (m1, m2).
    opt  : dict with keys ``lambda``, ``alpha``, ``beta_a``, ``lambda_u``,
           ``epsilon_min``, ``epsilon_max``, ``out_iter``, ``inner_iter``, ``IF``.

    Returns
    -------
    x_fov : restored image of shape (M1, M2).
    """
    M1, M2 = y.shape
    m1, m2 = h.shape
    hks1 = m1 // 2
    hks2 = m2 // 2
    n1 = M1 + m1 - 1
    n2 = M2 + m2 - 1

    x = pad_replicate(y, hks1, hks2)

    # First- and second-order derivative filters (3×3, MATLAB layout)
    dxf = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=np.float64)
    dyf = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=np.float64)
    dyyf = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]], dtype=np.float64)
    dxxf = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float64)
    dxyf = np.array([[0, 0, 0], [0, 1, -1], [0, -1, 1]], dtype=np.float64)

    dxfr = dxf[::-1, ::-1]
    dyfr = dyf[::-1, ::-1]
    dxxfr = dxxf[::-1, ::-1]
    dyyfr = dyyf[::-1, ::-1]
    dxyfr = dxyf[::-1, ::-1]

    H = psf2otf(h, (n1, n2))
    Ht = np.conj(H)
    Hx = psf2otf(dxf, (n1, n2))
    Hy = psf2otf(dyf, (n1, n2))
    Hxx = psf2otf(dxxf, (n1, n2))
    Hyy = psf2otf(dyyf, (n1, n2))
    Hxy = psf2otf(dxyf, (n1, n2))

    HH = (H * Ht).real
    HHx = (Hx * np.conj(Hx)).real
    HHy = (Hy * np.conj(Hy)).real
    HHxx = (Hxx * np.conj(Hxx)).real
    HHyy = (Hyy * np.conj(Hyy)).real
    HHxy = (Hxy * np.conj(Hxy)).real
    RR = HHx + HHy + HHxx + HHyy + HHxy

    lambda_ = float(opt["lambda"])
    alpha = float(opt["alpha"])
    beta_a = float(opt["beta_a"])
    lambda_u = float(min(opt["lambda_u"], 5000.0 * beta_a))
    w0 = 0.25
    epsilon_min = float(opt["epsilon_min"])
    epsilon_max = float(opt["epsilon_max"])
    N1 = int(opt["out_iter"])
    N2 = int(opt["inner_iter"])
    IF_ = float(opt["IF"])

    c = alpha * lambda_
    beta_min = alpha * lambda_ / (epsilon_max ** (2.0 - alpha))
    beta_max = alpha * lambda_ / (epsilon_min ** (2.0 - alpha))
    beta = beta_min

    # Compute initial gradients via convolution with circular pad
    def _conv_circ(a: np.ndarray, k: np.ndarray) -> np.ndarray:
        ap = np.pad(a, ((1, 1), (1, 1)), mode="wrap")
        return convolve2d(ap, k, mode="valid")

    dx = _conv_circ(x, dxf)
    dy = _conv_circ(x, dyf)
    dxx = _conv_circ(x, dxxf)
    dyy = _conv_circ(x, dyyf)
    dxy = _conv_circ(x, dxyf)

    adx = np.abs(dx)
    ady = np.abs(dy)
    adxx = np.abs(dxx)
    adyy = np.abs(dyy)
    adxy = np.abs(dxy)

    du = np.zeros((n1, n2), dtype=np.float64)
    dvx = np.zeros((n1, n2), dtype=np.float64)
    dvy = np.zeros((n1, n2), dtype=np.float64)
    dvxx = np.zeros((n1, n2), dtype=np.float64)
    dvyy = np.zeros((n1, n2), dtype=np.float64)
    dvxy = np.zeros((n1, n2), dtype=np.float64)

    X = fft2(x)
    Ax = np.real(ifft2(H * X))
    invA = HH + (beta_a / lambda_u) * RR

    for _outer in range(N1):
        # IRLS weights with Huber (ℓ_p) cap at ``beta``
        # Note: when adx contains zeros, adx**(alpha-2) overflows; but min
        # with beta clamps it to a finite value, matching MATLAB behaviour
        # (Inf is clamped to beta).
        with np.errstate(divide="ignore", invalid="ignore"):
            Wx = np.minimum(beta, c * adx ** (alpha - 2.0))
            Wy = np.minimum(beta, c * ady ** (alpha - 2.0))
            Wxx = np.minimum(beta, c * adxx ** (alpha - 2.0)) * w0
            Wyy = np.minimum(beta, c * adyy ** (alpha - 2.0)) * w0
            Wxy = np.minimum(beta, c * adxy ** (alpha - 2.0)) * w0
        # Sanitize NaNs (0**(neg) → inf → min(beta,inf)=beta, OK; 0/0 → NaN)
        Wx = np.nan_to_num(Wx, nan=beta, posinf=beta)
        Wy = np.nan_to_num(Wy, nan=beta, posinf=beta)
        Wxx = np.nan_to_num(Wxx, nan=beta * w0, posinf=beta * w0)
        Wyy = np.nan_to_num(Wyy, nan=beta * w0, posinf=beta * w0)
        Wxy = np.nan_to_num(Wxy, nan=beta * w0, posinf=beta * w0)

        for _inner in range(N2):
            # u sub-problem (FOV constraint applied only on inner region)
            u = Ax + du
            inner = u[hks1:n1 - hks1, hks2:n2 - hks2]
            inner = (y + lambda_u * inner) / (1.0 + lambda_u)
            u[hks1:n1 - hks1, hks2:n2 - hks2] = inner

            # v sub-problem
            vx = beta_a * (dx + dvx) / (Wx + beta_a)
            vy = beta_a * (dy + dvy) / (Wy + beta_a)
            vxx = beta_a * (dxx + dvxx) / (Wxx + beta_a)
            vyy = beta_a * (dyy + dvyy) / (Wyy + beta_a)
            vxy = beta_a * (dxy + dvxy) / (Wxy + beta_a)

            # dual variables
            du = du - u + Ax
            dvx = dvx - vx + dx
            dvy = dvy - vy + dy
            dvxx = dvxx - vxx + dxx
            dvyy = dvyy - vyy + dyy
            dvxy = dvxy - vxy + dxy

            # x sub-problem
            Y_fft = fft2(u - du) * Ht

            tempx = vx - dvx
            tempy = vy - dvy
            tempxx = vxx - dvxx
            tempyy = vyy - dvyy
            tempxy = vxy - dvxy

            tempx = _conv_circ(tempx, dxfr)
            tempy = _conv_circ(tempy, dyfr)
            tempxx = _conv_circ(tempxx, dxxfr)
            tempyy = _conv_circ(tempyy, dyyfr)
            tempxy = _conv_circ(tempxy, dxyfr)

            X = Y_fft + (beta_a / lambda_u) * fft2(
                tempx + tempy + tempxx + tempyy + tempxy
            )
            X = X / invA
            Ax = np.real(ifft2(H * X))
            x = np.real(ifft2(X))

            dx = _conv_circ(x, dxf)
            dy = _conv_circ(x, dyf)
            dxx = _conv_circ(x, dxxf)
            dyy = _conv_circ(x, dyyf)
            dxy = _conv_circ(x, dxyf)
            adx = np.abs(dx)
            ady = np.abs(dy)
            adxx = np.abs(dxx)
            adyy = np.abs(dyy)
            adxy = np.abs(dxy)

        beta = min(beta * IF_, beta_max)

    x_fov = x[hks1:n1 - hks1, hks2:n2 - hks2]
    return x_fov
