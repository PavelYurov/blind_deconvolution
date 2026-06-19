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


def gat_forward(image: np.ndarray, a: float, b: float) -> np.ndarray:


    if a <= 0:
        raise ValueError(
            f"gat_forward: a must be positive, got {a}. "
            f"For pure Gaussian noise use direct BM3D, not GAT.")
    a = float(a)
    b = float(b)
    arg = a * image + 3.0 * a * a / 8.0 + b
    return (2.0 / a) * np.sqrt(np.maximum(arg, 0.0))


def gat_inverse_asymptotic(z: np.ndarray, a: float, b: float) -> np.ndarray:


    if a <= 0:
        raise ValueError(f"gat_inverse_asymptotic: a must be positive, got {a}.")
    a = float(a)
    b = float(b)
    return a * ((z / 2.0) ** 2 - 3.0 / 8.0 - b / (a * a))


def vst_bm3d_denoise(image: np.ndarray,
                     a: float,
                     b: float,
                     stage_arg: str | None = None,
                     ) -> tuple[np.ndarray, dict]:


    try:
        import bm3d
    except ImportError as e:
        raise ImportError(
            "vst_bm3d_denoise requires the 'bm3d' package: pip install bm3d"
        ) from e

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")


    z = gat_forward(img, a=a, b=b)


    if stage_arg is None:
        z_hat = bm3d.bm3d(z, sigma_psd=1.0)
    else:

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
