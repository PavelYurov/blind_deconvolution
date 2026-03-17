"""
Complex Fixed-Rank 2-Factor quotient manifold.

Ported from ROPTLIB  Manifolds/CFixedRank2Factors/CFixedRank2Factors.cpp

The manifold is  M = C_*^{K x r} x C_*^{N x r} / GL(r),
which represents the set of complex K x N matrices of rank r
via the factorization  X = G H^*   (G: K x r,  H: N x r).

Riemannian metric:
    <(dG1, dH1), (dG2, dH2)>_{(G,H)}
      = tr(dG1^* dG2 (H^* H)) + tr(dH1^* dH2 (G^* G))

For blind deconvolution the typical case is  r = 1, which simplifies
many operations to scalar / vector algebra.
"""

import numpy as np
from numpy.linalg import cholesky, solve, norm, qr


# ─────────────────────────────────────────────────────────────────────
# Metric
# ─────────────────────────────────────────────────────────────────────

def metric(G, H, etaG, etaH, xiG, xiH):
    """Riemannian metric  <eta, xi>_{(G,H)} on the quotient manifold.

    For r = 1 this reduces to
        (H^* H) * (etaG^* xiG) + (G^* G) * (etaH^* xiH)
    all of which are scalars.
    """
    HH = H.conj().T @ H          # r x r
    GG = G.conj().T @ G          # r x r
    val = np.real(
        np.trace(etaG.conj().T @ xiG @ HH) +
        np.trace(etaH.conj().T @ xiH @ GG)
    )
    return val


# ─────────────────────────────────────────────────────────────────────
# Horizontal projection  (ExtrProjection in CFixedRank2Factors.cpp)
# ─────────────────────────────────────────────────────────────────────

def horizontal_projection(G, H, etaG, etaH):
    """Project an ambient tangent vector (etaG, etaH) onto the horizontal
    space at (G, H).

    This is a direct port of CFixedRank2Factors::ExtrProjection.

    Returns projected (pG, pH).
    """
    r = G.shape[1]

    HH = H.conj().T @ H  # r x r
    HV = H.conj().T @ etaH  # r x r

    GG = G.conj().T @ G  # r x r
    GV = G.conj().T @ etaG  # r x r

    # Solve HH^{-1} HV  and  GG^{-1} GV via Cholesky
    try:
        L_H = cholesky(HH)
        inv_HH_HV = solve(L_H.conj().T, solve(L_H, HV))
    except np.linalg.LinAlgError:
        inv_HH_HV = solve(HH + 1e-12 * np.eye(r), HV)

    try:
        L_G = cholesky(GG)
        inv_GG_GV = solve(L_G.conj().T, solve(L_G, GV))
    except np.linalg.LinAlgError:
        inv_GG_GV = solve(GG + 1e-12 * np.eye(r), GV)

    # Lambda = (inv_GG_GV - inv_HH_HV^*) / 2
    Lambda = (inv_GG_GV - inv_HH_HV.conj().T) / 2.0

    pG = etaG - G @ Lambda
    pH = etaH + H @ Lambda.conj().T
    return pG, pH


# ─────────────────────────────────────────────────────────────────────
# EucGradToGrad  (the Riemannian gradient from the Euclidean one)
# ─────────────────────────────────────────────────────────────────────

def euc_grad_to_grad(G, H, egfG, egfH):
    """Convert Euclidean gradient to Riemannian gradient.

    In CFixedRank2Factors this is simply the horizontal projection
    of the Euclidean gradient.
    """
    return horizontal_projection(G, H, egfG, egfH)


# ─────────────────────────────────────────────────────────────────────
# Retraction
# ─────────────────────────────────────────────────────────────────────

def retraction(G, H, etaG, etaH, stepsize=1.0):
    """Default additive retraction: (G, H) + stepsize * (etaG, etaH).

    For the CFixedRank2Factors manifold, the C++ code uses the
    ProductManifold default retraction when working in extrinsic mode,
    which for Euclidean component manifolds is simply addition.
    """
    G_new = G + stepsize * etaG
    H_new = H + stepsize * etaH
    return G_new, H_new


# ─────────────────────────────────────────────────────────────────────
# Vector transport (trivial — identity, matching C++)
# ─────────────────────────────────────────────────────────────────────

def vector_transport(G_old, H_old, etaG, etaH, G_new, H_new, xiG, xiH):
    """Isometric vector transport by differentiated retraction.

    For additive retraction on Euclidean components this is the identity.
    After transport the vector should be re-projected onto the horizontal
    space at the new point.
    """
    pG, pH = horizontal_projection(G_new, H_new, xiG, xiH)
    return pG, pH
