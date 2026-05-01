"""
_sanitation_cascade
===================

Shared, mathematically-grounded noise sanitation cascade for blind
deconvolution algorithms in `our_company/`.

Public API
----------
    sanitize(image, *, profile='auto', verbose=False) -> SanitationResult

Pipeline (delegated to the validated PMP robust orchestrator):
    1. impulse → adaptive median (only on flagged pixels)
    2. periodic peaks → notch filter (gated by lag-1)
    3. PCA / PSD descriptor on residual
    4. branch: VST+BM3D (Poisson family) | ACT-colored (truly correlated)
       | BM3D-white | no-op
    5. residual analysis on cleaned output → SanitationResult

Algorithms can switch to this cascade via `auto_mode='sanitation'` and treat
the returned `image_clean` + `residual_sigma` as a high-quality, low-noise
input to their existing blind-deconv pipeline.

Notes
-----
This package currently bridges to the validated implementation in
`patch_wise_minimum_pixels_prior/pmp_denoise_fix/`.  Long-term, the
underlying detectors will be migrated into this package as the canonical
copy.  For now, callers should depend ONLY on the public API exported here.
"""

from .api import sanitize, SanitationResult

__all__ = ['sanitize', 'SanitationResult']
