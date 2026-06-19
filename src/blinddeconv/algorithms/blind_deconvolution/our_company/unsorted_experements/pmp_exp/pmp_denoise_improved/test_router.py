"""Quick router-only smoke test for PMP_BD_Merged.

Generates the same 6 noise types as test_robust_pmp.py, classifies each
WITHOUT running blind deconv (fast), and prints the routing decision.

Run::
    .\.venv\Scripts\python.exe -X faulthandler -u -m \
        src.blinddeconv.algorithms.blind_deconvolution.our_company.\
patch_wise_minimum_pixels_prior.pmp_denoise_merge.test_router
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from ..pmp_denoise_fix.test_robust_pipeline import (
    _make_clean_image, _add_white_gaussian, _add_colored_gaussian,
    _add_poisson, _add_poisson_gaussian, _add_periodic, _add_impulse,
)
from .pmp_merged import PMP_BD_Merged


def _gaussian_psf(k: int = 21, sigma: float = 2.0) -> np.ndarray:
    ax = np.arange(k) - (k - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def main(seed: int = 0):
    clean = _make_clean_image(256, 256, seed=seed)
    psf = _gaussian_psf(21, 2.0)
    blurred = np.clip(fftconvolve(clean, psf, mode='same'), 0, 1)

    cases = [
        ('AWGN sigma=0.02',                 lambda b: _add_white_gaussian(b, 0.02, seed + 1)),
        ('AWGN sigma=0.04',                 lambda b: _add_white_gaussian(b, 0.04, seed + 2)),
        ('Colored Gauss sigma=0.04',        lambda b: _add_colored_gaussian(b, 0.04, seed + 3, smooth_radius=1.5)),
        ('Poisson photons=120',             lambda b: _add_poisson(b, 120.0, seed + 4)),
        ('Poisson-Gauss P=120 sg=0.02',     lambda b: _add_poisson_gaussian(b, 120.0, 0.02, seed + 5)),
        ('Periodic + AWGN sigma=0.02',      lambda b: _add_periodic(b, 0.02, seed + 6, amplitude=0.06)),
        ('Impulse density=0.03',            lambda b: _add_impulse(b, 0.03, seed + 7)),
        ('No noise',                        lambda b: b.copy()),
    ]

    merged = PMP_BD_Merged(kernel_size=21, verbose=False)
    print('=' * 90)
    print('PMP_BD_Merged router decisions')
    print('=' * 90)
    print(f"{'noise':32s} | {'branch':6s} | {'sig':6s} | {'imp':6s} | {'a_n':9s} | reason")
    print('-' * 90)
    for name, fn in cases:
        noisy = fn(blurred)
        branch, desc = merged._classify(noisy)
        print(f"{name:32s} | {branch:6s} | "
              f"{desc.get('psd_sigma_norm', 0):.4f} | "
              f"{desc.get('impulse_density', 0):.4f} | "
              f"{desc.get('pca_a_norm', 0):.3e} | "
              f"{desc.get('reason', '')}")
    print('=' * 90)


if __name__ == '__main__':
    main()
