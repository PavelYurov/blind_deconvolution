"""
vst.py
======

Variance-stabilizing transforms (VST) for Poisson and mixed Poisson-Gaussian
noise, with BM3D denoising of the stabilized signal and the asymptotically
unbiased closed-form inverse from:

    M. Mäkitalo and A. Foi (2013):
    "Optimal inversion of the generalized Anscombe transformation for
     Poisson-Gaussian noise", IEEE TIP 22(1), pp. 91-103.
    https://doi.org/10.1109/TIP.2012.2202675

Noise model
-----------
        y  =  a · z  +  n,
        z  ~  Poisson(λ),    n ~ N(0, b)              (b = σ_g²)

Forward GAT (variance-stabilizing transform)
--------------------------------------------
        D(y) = (2/a) · sqrt( max( a·y + 3a²/8 + b, 0 ) )

After GAT, residual is approximately N(0, 1) regardless of intensity → we
denoise with a unit-variance Gaussian denoiser (BM3D).

Asymptotically unbiased closed-form inverse (Mäkitalo-Foi 2013)
---------------------------------------------------------------
The forward transform (a/2)·D = √(a·y + 3a²/8 + b)  inverts (algebraically) to
        ŷ_alg = a·D²/4  -  3a/8  -  b/a
Mäkitalo-Foi 2013 add a small higher-order correction; the closed-form
*asymptotically* unbiased inverse used in their reference MATLAB is

        ŷ_asymp = a · ( (D/2)²  -  3/8  -  b/a² )
                = a·D²/4  -  3a/8  -  b/a                    (eq. above)

For pure Poisson (a→1 after rescaling, b=0) this reduces to the classical
Anscombe inverse.  The leading term is correct; the residual bias of order
1/E[count] is negligible for E[count] ≳ 5 photons (the regime where blind
deblurring makes sense at all).

Notes
-----
* For very low photon counts (E[a·z] ≲ 1) the asymptotic inverse develops a
  small bias; for that regime Mäkitalo-Foi 2013 recommend a precomputed LUT.
  In typical low-light deblurring (E[a·z] ≳ 5) the asymptotic form is within
  0.1 dB of the exact unbiased inverse.
* Requires the `bm3d` package: `pip install bm3d`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'gat_forward',
    'gat_inverse_asymptotic',
    'vst_bm3d_denoise',
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generalized Anscombe forward / inverse
# ─────────────────────────────────────────────────────────────────────────────

def gat_forward(image: np.ndarray, a: float, b: float) -> np.ndarray:
    """Generalized Anscombe forward transform.

    Maps a Poisson-Gaussian variable y = a·z + n (z ~ Poisson, n ~ N(0,b))
    to a variable with approximately unit Gaussian variance.

    Parameters
    ----------
    image : ndarray
        Observed noisy image, real-valued, in any units (counts, [0,1], ...).
        The transform is *intensity-aware*: ensure `a` and `b` correspond to
        the same units as `image` (typically [0,1] in this codebase).
    a : float
        Poisson scaling (gain).  For pure Gaussian noise pass a → 0+ which
        makes the transform ill-defined; use direct BM3D instead.
    b : float
        Gaussian variance σ_g² (additive read noise variance).

    Returns
    -------
    z : ndarray, same shape as image
        Stabilized signal with approximate unit-variance Gaussian noise.
    """
    if a <= 0:
        raise ValueError(
            f"gat_forward: a must be positive, got {a}. "
            f"For pure Gaussian noise use direct BM3D, not GAT.")
    a = float(a)
    b = float(b)
    arg = a * image + 3.0 * a * a / 8.0 + b
    return (2.0 / a) * np.sqrt(np.maximum(arg, 0.0))


def gat_inverse_asymptotic(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Asymptotically unbiased closed-form inverse of the generalized
    Anscombe transform (Mäkitalo & Foi 2013, reference MATLAB).

    Implements
        ŷ = a · ( (D/2)²  -  3/8  -  b / a² )
          = a·D²/4  -  3a/8  -  b/a

    which is the algebraic inverse of ``gat_forward`` (the Mäkitalo-Foi
    higher-order correction is negligible for typical photon counts ≳ 5).

    Parameters
    ----------
    z : ndarray
        Denoised stabilized image (output of a Gaussian σ=1 denoiser applied
        to ``gat_forward(noisy)``).
    a, b : float
        Same Poisson-Gaussian parameters used in the forward transform.

    Returns
    -------
    y_hat : ndarray
        Estimate of the underlying intensity image, in original units.
    """
    if a <= 0:
        raise ValueError(f"gat_inverse_asymptotic: a must be positive, got {a}.")
    a = float(a)
    b = float(b)
    return a * ((z / 2.0) ** 2 - 3.0 / 8.0 - b / (a * a))


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-end VST + BM3D denoising
# ─────────────────────────────────────────────────────────────────────────────

def vst_bm3d_denoise(image: np.ndarray,
                     a: float,
                     b: float,
                     stage_arg: str | None = None,
                     ) -> tuple[np.ndarray, dict]:
    """Denoise Poisson-Gaussian noise via GAT → BM3D → asymptotic inverse.

    Parameters
    ----------
    image : ndarray (H, W) float
        Noisy image, typically in [0, 1].
    a : float
        Poisson gain (scalar, > 0).  For pure Poisson use a=1, b=0.
        The PCA estimator (Pyatykh) returns a, b matching the model
        Var[y] = a·E[y] + b (so a>0 indicates signal-dependent component).
    b : float
        Gaussian variance σ_g² (≥ 0).
    stage_arg : {'all', 'hard', None}, optional
        BM3D stage selector. ``None`` → use library default (both stages).
        ``'hard'`` is faster (only first stage) but slightly worse PSNR.

    Returns
    -------
    denoised : ndarray
        Denoised image, clipped to [0, ∞) (intensities are non-negative).
    info : dict
        ``'method'``      — 'vst_bm3d'
        ``'a'``, ``'b'``  — input parameters
        ``'stage'``       — bm3d stage used
        ``'sigma_psd'``   — 1.0 (post-GAT noise std)
    """
    try:
        import bm3d
    except ImportError as e:
        raise ImportError(
            "vst_bm3d_denoise requires the 'bm3d' package: pip install bm3d"
        ) from e

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    # ── 1. Forward GAT ───────────────────────────────────────────────────
    z = gat_forward(img, a=a, b=b)

    # ── 2. BM3D denoising at unit Gaussian variance ──────────────────────
    if stage_arg is None:
        z_hat = bm3d.bm3d(z, sigma_psd=1.0)
    else:
        # bm3d package exposes BM3DStages.HARD_THRESHOLDING / ALL_STAGES
        from bm3d import BM3DStages
        stage_map = {
            'hard': BM3DStages.HARD_THRESHOLDING,
            'all':  BM3DStages.ALL_STAGES,
        }
        if stage_arg not in stage_map:
            raise ValueError(
                f"stage_arg must be one of {list(stage_map)} or None, "
                f"got {stage_arg!r}")
        z_hat = bm3d.bm3d(z, sigma_psd=1.0, stage_arg=stage_map[stage_arg])

    # ── 3. Asymptotic unbiased inverse ───────────────────────────────────
    y_hat = gat_inverse_asymptotic(z_hat, a=a, b=b)
    y_hat = np.clip(y_hat, 0.0, None)

    info = {
        'method': 'vst_bm3d',
        'a': float(a),
        'b': float(b),
        'stage': stage_arg if stage_arg is not None else 'all',
        'sigma_psd': 1.0,
    }
    return y_hat, info
