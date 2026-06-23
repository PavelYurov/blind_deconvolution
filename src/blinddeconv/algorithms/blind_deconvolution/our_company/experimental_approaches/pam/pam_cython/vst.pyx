from __future__ import annotations

import numpy as np

__all__ = ['gat_forward', 'gat_inverse_asymptotic', 'vst_bm3d_denoise']

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
        raise ValueError(
            f"gat_inverse_asymptotic: a must be positive, got {a}.")
    a = float(a)
    b = float(b)
    return a * ((z / 2.0) ** 2 - 3.0 / 8.0 - b / (a * a))

def vst_bm3d_denoise(img,
                     noise_info=None,
                     *,
                     a: float | None = None,
                     b: float | None = None,
                     sigma: float | None = None,
                     stage_arg: str | None = None,
                     verbose: bool = False):

    try:
        import bm3d as _bm3d
    except ImportError as e:
        raise ImportError(
            "vst_bm3d_denoise requires the 'bm3d' package: pip install bm3d"
        ) from e

    img = np.asarray(img, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    ni = noise_info or {}
    a_raw = float(ni.get('a', 0.0)) if a is None else float(a)
    b_raw = float(ni.get('b', 0.0)) if b is None else float(b)
    sigma_eff = sigma
    if sigma_eff is None:
        sigma_eff = ni.get('sigma_norm', None)

    A = a_raw / 255.0
    B = b_raw / (255.0 ** 2)

    stage_kw = {}
    stage_label = 'all'
    if stage_arg is not None:
        from bm3d import BM3DStages
        stage_map = {'hard': BM3DStages.HARD_THRESHOLDING,
                     'all':  BM3DStages.ALL_STAGES}
        if stage_arg not in stage_map:
            raise ValueError(
                f"stage_arg must be one of {list(stage_map)} or None, "
                f"got {stage_arg!r}")
        stage_kw['stage_arg'] = stage_map[stage_arg]
        stage_label = stage_arg

    info = {'a': a_raw, 'b': b_raw,
            'A_norm': float(A), 'B_norm': float(B),
            'stage': stage_label, 'sigma_psd': 1.0}

    if A <= 1e-8:
        if sigma_eff is not None and sigma_eff > 0:
            sig = float(sigma_eff)
        elif B > 0:
            sig = float(np.sqrt(B))
        else:
            sig = 0.05
        sig = max(sig, 1e-4)
        if verbose:
            print(f"[vst_bm3d] No Poisson component (a={a_raw:.4g}); "
                  f"plain BM3D σ={sig:.5f}")
        denoised = _bm3d.bm3d(img, sigma_psd=sig, **stage_kw)
        info.update({'mode': 'gaussian_fallback', 'sigma_used': sig})
        return np.clip(denoised, 0.0, 1.0), info

    z = gat_forward(img, a=A, b=B)

    if verbose:
        print(f"[vst_bm3d] GAT: A={A:.6g}, B={B:.6g}, "
              f"z∈[{z.min():.4f},{z.max():.4f}], "
              f"BM3D sigma_psd=1.0, stage={stage_label}")

    z_hat = _bm3d.bm3d(z, sigma_psd=1.0, **stage_kw)

    y_hat = gat_inverse_asymptotic(z_hat, a=A, b=B)

    info.update({
        'mode': 'gat_bm3d',
        'sigma_used': 1.0,
        'z_min': float(z.min()), 'z_max': float(z.max()),
    })
    return np.clip(y_hat, 0.0, 1.0), info
