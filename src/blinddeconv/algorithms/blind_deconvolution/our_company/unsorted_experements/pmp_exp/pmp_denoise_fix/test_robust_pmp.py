"""
test_robust_pmp.py
==================

End-to-end blind+non-blind comparison harness:
    OLD ``PMP_BD``  vs  NEW ``PMP_BD_Robust``

For each noise type a synthetic experiment is built::

    clean  ── ⊛ PSF_true ──►  blurred
    blurred  ── + noise ──►   noisy

Both algorithms restore ``noisy`` and we report:

    * PSNR_image  =  PSNR(restored, clean)
    * kernel_err  =  ||k_est − k_true||₂ / ||k_true||₂   (centroid-aligned)
    * runtime (seconds)

The clean image and PSFs are deterministic; identical PSF + noise per row
so any winner/loser is fully attributable to the algorithm change.

Run::

    .\.venv\Scripts\python.exe -X faulthandler -u -m \
        src.blinddeconv.algorithms.blind_deconvolution.our_company.\
patch_wise_minimum_pixels_prior.pmp_denoise_fix.test_robust_pmp
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
from scipy.signal import fftconvolve

from .pmp import PMP_BD
from .pmp_robust import PMP_BD_Robust
from .test_robust_pipeline import (
    _make_clean_image,
    _add_white_gaussian,
    _add_colored_gaussian,
    _add_poisson,
    _add_poisson_gaussian,
    _add_periodic,
    _add_impulse,
)


def _gaussian_psf(k: int = 21, sigma: float = 2.0) -> np.ndarray:
    ax = np.arange(k) - (k - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def _motion_psf(k: int = 21, length: int = 13, angle_deg: float = 30.0) -> np.ndarray:
    psf = np.zeros((k, k), dtype=np.float64)
    cx = cy = (k - 1) / 2.0
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th), np.sin(th)
    for t in np.linspace(-length / 2.0, length / 2.0, num=length * 8 + 1):
        x = cx + t * dx
        y = cy + t * dy
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < k and 0 <= iy < k:
            psf[iy, ix] += 1.0
    psf /= psf.sum()
    return psf


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def _kernel_error(k_est: np.ndarray, k_true: np.ndarray) -> float:

    k_est = np.asarray(k_est, dtype=np.float64)
    k_true = np.asarray(k_true, dtype=np.float64)
    s = k_est.sum()
    if s > 0:
        k_est = k_est / s
    s = k_true.sum()
    if s > 0:
        k_true = k_true / s


    H = max(k_est.shape[0], k_true.shape[0]) + 4
    H |= 1
    W = max(k_est.shape[1], k_true.shape[1]) + 4
    W |= 1

    def _embed_centroid(k: np.ndarray, H: int, W: int) -> np.ndarray:

        ys = np.arange(k.shape[0])[:, None]
        xs = np.arange(k.shape[1])[None, :]
        m = k.sum()
        if m <= 0:
            cy, cx = (k.shape[0] - 1) / 2.0, (k.shape[1] - 1) / 2.0
        else:
            cy = float((ys * k).sum() / m)
            cx = float((xs * k).sum() / m)
        out = np.zeros((H, W), dtype=np.float64)

        dy = (H - 1) / 2.0 - cy
        dx = (W - 1) / 2.0 - cx

        idy = int(round(dy))
        idx = int(round(dx))
        y0 = idy
        x0 = idx
        y1 = y0 + k.shape[0]
        x1 = x0 + k.shape[1]

        sy0 = max(0, -y0); sy1 = k.shape[0] - max(0, y1 - H)
        sx0 = max(0, -x0); sx1 = k.shape[1] - max(0, x1 - W)
        ty0 = max(0, y0); ty1 = min(H, y1)
        tx0 = max(0, x0); tx1 = min(W, x1)
        out[ty0:ty1, tx0:tx1] = k[sy0:sy1, sx0:sx1]
        return out

    a = _embed_centroid(k_est, H, W)
    b = _embed_centroid(k_true, H, W)
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(den, 1e-12)


def _make_blurred_noisy(clean: np.ndarray,
                        psf_true: np.ndarray,
                        noise_fn) -> np.ndarray:
    blurred = fftconvolve(clean, psf_true, mode='same')
    blurred = np.clip(blurred, 0.0, 1.0)
    return noise_fn(blurred)


def _run_alg(alg, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    t0 = time.time()
    restored, kernel = alg.process(noisy)
    dt = time.time() - t0
    restored = np.asarray(restored, dtype=np.float64)
    if restored.max() > 1.5:
        restored /= 255.0
    if restored.ndim == 3:
        restored = restored.mean(axis=-1)
    restored = np.clip(restored, 0.0, 1.0)
    kernel = np.asarray(kernel, dtype=np.float64)
    return restored, kernel, dt


def _evaluate(name: str,
              clean: np.ndarray,
              psf_true: np.ndarray,
              noise_fn,
              kernel_size: int = 21,
              verbose: bool = False) -> dict:
    noisy = _make_blurred_noisy(clean, psf_true, noise_fn)
    psnr_in = _psnr(noisy, clean)


    old = PMP_BD(kernel_size=kernel_size, verbose=verbose)
    new = PMP_BD_Robust(kernel_size=kernel_size, verbose=verbose)

    r_old, k_old, t_old = _run_alg(old, noisy)
    r_new, k_new, t_new = _run_alg(new, noisy)

    return {
        'name':        name,
        'psnr_in':     psnr_in,
        'psnr_old':    _psnr(r_old, clean),
        'psnr_new':    _psnr(r_new, clean),
        'ker_old':     _kernel_error(k_old, psf_true),
        'ker_new':     _kernel_error(k_new, psf_true),
        't_old':       t_old,
        't_new':       t_new,
        'branch':      getattr(new, '_last_robust_info',
                               {}).get('branch', '?')
                       if getattr(new, '_last_robust_info', None) else '?',
    }


def main(seed: int = 0, kernel_size: int = 21, image_size: int = 256,
         verbose_alg: bool = False) -> None:

    clean = _make_clean_image(image_size, image_size, seed=seed)
    psf_true = _gaussian_psf(k=kernel_size, sigma=2.0)


    cases = [
        ('AWGN σ=0.02',
         lambda b: _add_white_gaussian(b, 0.02, seed + 1)),
        ('AWGN σ=0.04',
         lambda b: _add_white_gaussian(b, 0.04, seed + 2)),
        ('Colored Gaussian σ=0.04',
         lambda b: _add_colored_gaussian(b, 0.04, seed + 3, smooth_radius=1.5)),
        ('Poisson photons=120',
         lambda b: _add_poisson(b, 120.0, seed + 4)),
        ('Poisson-Gaussian P=120, σg=0.02',
         lambda b: _add_poisson_gaussian(b, 120.0, 0.02, seed + 5)),
        ('Periodic + AWGN σ=0.02',
         lambda b: _add_periodic(b, 0.02, seed + 6, amplitude=0.06)),
        ('Impulse density=0.03',
         lambda b: _add_impulse(b, 0.03, seed + 7)),
        ('No noise',
         lambda b: b.copy()),
    ]

    print('=' * 110)
    print(f"PMP_BD vs PMP_BD_Robust — image {image_size}×{image_size}, "
          f"kernel size {kernel_size}, true PSF=Gauss(σ=2.0)")
    print('=' * 110)
    header = (f"{'noise':38s} | {'PSNR_in':>7s} | "
              f"{'PSNR_old':>8s} {'PSNR_new':>8s} {'Δdb':>6s} | "
              f"{'k_old':>6s} {'k_new':>6s} {'Δkerr':>7s} | "
              f"{'t_old':>5s} {'t_new':>5s} | branch")
    print(header)
    print('-' * len(header))

    results = []
    for name, noise_fn in cases:
        try:
            r = _evaluate(name, clean, psf_true, noise_fn,
                          kernel_size=kernel_size, verbose=verbose_alg)
        except Exception as ex:
            print(f"{name:38s} | FAILED: {type(ex).__name__}: {ex}")
            continue
        results.append(r)
        d_psnr = r['psnr_new'] - r['psnr_old']
        d_kerr = r['ker_new'] - r['ker_old']
        sign_psnr = '+' if d_psnr >= 0 else ''
        sign_kerr = '+' if d_kerr >= 0 else ''
        print(f"{r['name']:38s} | {r['psnr_in']:7.2f} | "
              f"{r['psnr_old']:8.2f} {r['psnr_new']:8.2f} "
              f"{sign_psnr}{d_psnr:5.2f} | "
              f"{r['ker_old']:6.3f} {r['ker_new']:6.3f} "
              f"{sign_kerr}{d_kerr:6.3f} | "
              f"{r['t_old']:5.1f} {r['t_new']:5.1f} | {r['branch']}")

    print('-' * len(header))
    if results:
        d_psnrs = [r['psnr_new'] - r['psnr_old'] for r in results]
        d_kerrs = [r['ker_new'] - r['ker_old'] for r in results]
        wins  = sum(1 for d in d_psnrs if d > 0.1)
        losses = sum(1 for d in d_psnrs if d < -0.1)
        print(f"Δ-PSNR(new−old): mean={np.mean(d_psnrs):+.2f} dB, "
              f"min={min(d_psnrs):+.2f}, max={max(d_psnrs):+.2f} | "
              f"wins(>0.1)={wins}, losses(<-0.1)={losses}, "
              f"ties={len(results) - wins - losses}")
        print(f"Δ-kernel-err(new−old): mean={np.mean(d_kerrs):+.3f}, "
              f"min={min(d_kerrs):+.3f}, max={max(d_kerrs):+.3f}  "
              f"(negative = NEW better)")
    print('=' * 110)


if __name__ == '__main__':
    main(seed=0, kernel_size=21, image_size=256, verbose_alg=False)
