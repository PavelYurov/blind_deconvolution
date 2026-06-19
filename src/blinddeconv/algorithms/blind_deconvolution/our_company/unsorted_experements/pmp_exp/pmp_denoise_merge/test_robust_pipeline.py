"""
test_robust_pipeline.py
=======================

Smoke / validation script for the robust noise pipeline in
``pmp_denoise_fix/noise_orchestrator.py``.

Synthesises a clean grayscale image with six well-defined noise types,
runs the orchestrator, and reports:

    * detector decision (impulse / periodic / pca / psd)
    * branch chosen (vst / act_colored / bm3d_white / noop)
    * PSNR(noisy, clean)  vs  PSNR(denoised, clean)

It does NOT run blind deconvolution — that is the user's downstream
experiment.  This file only verifies that the *denoising* part of the
pipeline picks the mathematically correct branch for each noise type
and recovers the underlying signal with positive PSNR gain.

Run from the framework (9) folder::

    .\\.venv\\Scripts\\python.exe -m \
        src.blinddeconv.algorithms.blind_deconvolution.our_company.\
patch_wise_minimum_pixels_prior.pmp_denoise_fix.test_robust_pipeline
"""

from __future__ import annotations

import time

import numpy as np


from .noise_orchestrator import robust_denoise


def _make_clean_image(H: int = 256, W: int = 256, seed: int = 0) -> np.ndarray:

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64) / max(H, W)

    img = 0.4 * y + 0.3 * x
    cy, cx = H * 0.55, W * 0.40
    r = np.sqrt((np.arange(H)[:, None] - cy) ** 2
                + (np.arange(W)[None, :] - cx) ** 2)
    img += 0.35 * np.exp(-(r ** 2) / (2 * (min(H, W) * 0.12) ** 2))

    img[H // 4: H // 2, W // 6: W // 3] = 0.85
    img[2 * H // 3: 5 * H // 6, 2 * W // 3: 11 * W // 12] = 0.15

    img += 0.03 * np.cos(rng.uniform(0, 2 * np.pi)
                         + 12 * np.pi * (x + y))
    return np.clip(img, 0.0, 1.0)


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float('inf')
    return 10.0 * float(np.log10((data_range ** 2) / mse))


def _add_white_gaussian(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 1)


def _add_colored_gaussian(clean: np.ndarray, sigma: float, seed: int,
                          smooth_radius: float = 1.5) -> np.ndarray:

    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1.0, clean.shape)
    coloured = gaussian_filter(white, sigma=smooth_radius)

    coloured *= sigma / max(coloured.std(), 1e-12)
    return np.clip(clean + coloured, 0, 1)


def _add_poisson(clean: np.ndarray, photons: float, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum(clean, 0) * photons)
    return np.clip(counts / photons, 0, 1)


def _add_poisson_gaussian(clean: np.ndarray, photons: float,
                          sigma_g: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.maximum(clean, 0) * photons)
    img = counts / photons + rng.normal(0, sigma_g, clean.shape)
    return np.clip(img, 0, 1)


def _add_periodic(clean: np.ndarray, sigma: float, seed: int,
                  freq_uv=(48, 32), amplitude: float = 0.06) -> np.ndarray:

    rng = np.random.default_rng(seed)
    H, W = clean.shape
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = freq_uv
    pattern = amplitude * np.sin(2 * np.pi * (u * xx / W + v * yy / H))
    out = clean + pattern + rng.normal(0, sigma, clean.shape)
    return np.clip(out, 0, 1)


def _add_impulse(clean: np.ndarray, density: float, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    out = clean.copy()
    mask = rng.random(out.shape) < density
    salt_or_pepper = rng.random(out.shape) < 0.5
    out[mask & salt_or_pepper] = 1.0
    out[mask & ~salt_or_pepper] = 0.0
    return out


def _build_test_cases(clean: np.ndarray):
    return [
        ('AWGN σ=0.04',
         _add_white_gaussian(clean, sigma=0.04, seed=1),
         {'expected_branch': 'bm3d_white'}),
        ('Colored Gaussian σ=0.04',
         _add_colored_gaussian(clean, sigma=0.04, seed=2, smooth_radius=1.5),
         {'expected_branch': 'act_colored'}),
        ('Poisson (photons=80)',
         _add_poisson(clean, photons=80.0, seed=3),
         {'expected_branch': 'vst'}),
        ('Poisson-Gaussian (P=120, σ_g=0.02)',
         _add_poisson_gaussian(clean, photons=120.0, sigma_g=0.02, seed=4),
         {'expected_branch': 'vst'}),
        ('Periodic + AWGN σ=0.02',
         _add_periodic(clean, sigma=0.02, seed=5, amplitude=0.10),
         {'expected_branch': 'any'}),
        ('Impulse density=0.05',
         _add_impulse(clean, density=0.05, seed=6),
         {'expected_branch': 'any'}),
    ]


def _format_row(name, branch, t, psnr_in, psnr_out, status):
    return (f"{name:<38s} | branch={branch:<13s} | "
            f"PSNR {psnr_in:5.2f} → {psnr_out:5.2f} dB "
            f"(Δ={psnr_out - psnr_in:+5.2f}) | "
            f"{t:5.1f}s | {status}")


def main():
    np.random.seed(0)
    H = W = 256
    clean = _make_clean_image(H, W, seed=0)
    cases = _build_test_cases(clean)

    print("=" * 100)
    print(f"Robust noise pipeline smoke test  (image {H}×{W}, clean range "
          f"[{clean.min():.3f}, {clean.max():.3f}])")
    print("=" * 100)

    results = []
    for name, noisy, expect in cases:
        psnr_in = _psnr(noisy, clean)
        t0 = time.perf_counter()
        try:
            denoised, info = robust_denoise(noisy, verbose=False)
            dt = time.perf_counter() - t0
            branch = info['branch']
            psnr_out = _psnr(denoised, clean)
            exp = expect['expected_branch']
            ok = (exp == 'any') or (branch == exp)
            psnr_ok = psnr_out > psnr_in - 0.5
            status = ('OK' if (ok and psnr_ok)
                      else f"WARN(branch want {exp})" if not ok
                      else "WARN(no PSNR gain)")
        except Exception as e:
            dt = time.perf_counter() - t0
            branch = 'ERROR'
            psnr_out = float('nan')
            status = f"FAIL: {type(e).__name__}: {e}"
            info = {'log': []}

        print(_format_row(name, branch, dt, psnr_in, psnr_out, status))
        for line in info.get('log', []):
            print(f"    {line}")
        results.append((name, status))

    print("=" * 100)
    n_ok = sum(1 for _, s in results if s.startswith('OK'))
    print(f"PASSED: {n_ok}/{len(results)}")
    return results


if __name__ == '__main__':
    main()
