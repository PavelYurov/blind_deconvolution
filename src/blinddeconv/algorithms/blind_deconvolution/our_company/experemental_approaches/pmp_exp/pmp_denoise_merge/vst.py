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
