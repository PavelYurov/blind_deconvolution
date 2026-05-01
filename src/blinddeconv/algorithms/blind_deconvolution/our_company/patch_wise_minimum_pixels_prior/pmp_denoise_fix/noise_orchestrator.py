"""
noise_orchestrator.py
=====================

Mathematically grounded noise pipeline for blind deconvolution.

A unified detector + cascade router that fuses three independent estimators
(Pyatykh PCA, smooth-patch PSD analysis, impulse-density detector) into a
single descriptor and dispatches the image to the *correct* denoiser:

    impulse pixels      → adaptive median filter
    periodic interference → notch filter on detected peaks
    Poisson / Poisson-Gaussian → generalized Anscombe VST + BM3D + inverse
    correlated Gaussian   → ACT (curvelet) with 2D-PSD colored branch
    white Gaussian        → BM3D with scalar σ

Each branch is invoked only when its underlying assumption is *empirically
confirmed*; this replaces the previous heuristic that always passed
``noise_var=σ²`` to ACT regardless of noise type.

Public API
----------
    analyze_noise(image)        — run all detectors, return descriptor.
    robust_denoise(image)       — full cascade: detect + denoise.
    psd_to_act_fft_format(...)  — convention adapter for ACT colored branch.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Any

from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from .noise_psd_analysis import analyze_noise_psd, notch_filter
from .pyatykh_noise_reconstruction import estimate_noise_params
from .chen_noise_estimate import estimate_noise_level

__all__ = [
    'analyze_noise',
    'robust_denoise',
    'psd_to_act_fft_format',
]


# ─────────────────────────────────────────────────────────────────────────────
# Convention adapter: noise_psd_analysis  ──→  act_denoise
# ─────────────────────────────────────────────────────────────────────────────
#
# `analyze_noise_psd()['psd_2d']` is centred (DC at the centre, fftshift order)
# and normalised so that for white noise of variance σ² it equals σ² at every
# pixel — see estimate_noise_psd: avg-periodogram divided by window energy.
#
# `act_denoise()` expects FFT-PSD in *standard FFT order* (DC at [0,0]) and
# scaled so that for white σ²: FFT_PSD = σ² · H · W.
#
# Conversion:
#     act_psd  =  ifftshift(centred_psd) · (H · W)
#
# Sanity: for white centred_psd = σ² constant
#         → act_psd = σ² · H · W constant ✓ matches ACT internal formula.
# ─────────────────────────────────────────────────────────────────────────────

def psd_to_act_fft_format(psd_centered: np.ndarray) -> np.ndarray:
    """Convert centred-PSD (σ² for white) to ACT FFT-PSD (σ²·H·W, FFT order).

    Parameters
    ----------
    psd_centered : ndarray (H, W)
        2D PSD as returned by ``analyze_noise_psd()['psd_2d']``.

    Returns
    -------
    fft_psd : ndarray (H, W)
        Same data in FFT order with ACT scaling — feed directly to
        ``act_denoise(image, noise_var=fft_psd, ...)``.
    """
    psd_centered = np.asarray(psd_centered, dtype=np.float64)
    H, W = psd_centered.shape
    return np.fft.ifftshift(psd_centered) * (H * W)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unified noise descriptor
# ─────────────────────────────────────────────────────────────────────────────

def _to_grayscale_norm(image: np.ndarray) -> np.ndarray:
    """Convert to float64 [0,1] grayscale (consistent input to all detectors)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = (0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1]
                   + 0.1140 * img[:, :, 2])
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[2]}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D after conversion, got ndim={img.ndim}")
    if img.max() > 1.5:
        img = img / 255.0
    return img


def analyze_noise(image: np.ndarray,
                  pch_size: int = 32,
                  n_smooth: int = 100,
                  peak_threshold: float = 100.0,
                  pyatykh_blocksize: int = 7,
                  ) -> Dict[str, Any]:
    """Run impulse + PSD + PCA detectors, return unified descriptor.

    Parameters
    ----------
    image : ndarray
        Noisy image (gray or RGB, any scale; auto-normalised to [0,1] gray).
    pch_size, n_smooth, peak_threshold : forwarded to ``analyze_noise_psd``.
    pyatykh_blocksize : forwarded to ``estimate_noise_params``.

    Returns
    -------
    info : dict
        ``'image_norm'``  — grayscale float64 [0,1] used for analysis
        ``'impulse'``     — full output of ``detect_impulse_noise``
        ``'psd'``         — full output of ``analyze_noise_psd``
        ``'pca'``         — full output of ``estimate_noise_params``
                            (a, b on [0,255] scale)
        ``'pca_norm'``    — a, b rescaled to [0,1] image scale:
                              a_norm = a / 255      (Var = a_norm · y + b_norm)
                              b_norm = b / 255**2
        ``'chen_sigma_norm'`` — Chen's σ on [0,1] scale (fallback)
    """
    img = _to_grayscale_norm(image)

    impulse = detect_impulse_noise(img)
    psd = analyze_noise_psd(img,
                            pch_size=pch_size,
                            n_smooth=n_smooth,
                            peak_threshold=peak_threshold)
    pca = estimate_noise_params(img, blocksize=pyatykh_blocksize)
    chen_sigma_norm = float(estimate_noise_level(img))

    a_255 = float(pca.get('a', 0.0))
    b_255 = float(pca.get('b', 0.0))
    pca_norm = {
        'a': a_255 / 255.0,
        'b': b_255 / (255.0 ** 2),
        'sigma_norm': float(pca.get('sigma_norm', 0.0)),
        'noise_type': pca.get('noise_type', 'unknown'),
    }

    return {
        'image_norm': img,
        'impulse': impulse,
        'psd': psd,
        'pca': pca,
        'pca_norm': pca_norm,
        'chen_sigma_norm': chen_sigma_norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Branch denoisers
# ─────────────────────────────────────────────────────────────────────────────

def _denoise_white_bm3d(image: np.ndarray, sigma: float) -> np.ndarray:
    """Standard BM3D for white Gaussian noise."""
    try:
        import bm3d
    except ImportError as e:
        raise ImportError("BM3D path requires `pip install bm3d`") from e
    if sigma <= 0:
        return image.copy()
    return bm3d.bm3d(image, sigma_psd=float(sigma))


def _denoise_colored_act(image: np.ndarray,
                         psd_centered: np.ndarray,
                         threshold_setting: str = 'ksigma',
                         ) -> Tuple[np.ndarray, dict]:
    """ACT curvelet denoising for correlated Gaussian noise.

    Engages ACT's *colored* branch via the 2D-PSD path (31×31 ML window).

    Default ``threshold_setting='ksigma'`` is less aggressive than soft ACT
    (``'s'``); it preserves more high-frequency edges — important for
    downstream blind deconvolution which needs intact edges to estimate
    the kernel.
    """
    from .act_denoise import act_denoise
    fft_psd = psd_to_act_fft_format(psd_centered)
    return act_denoise(image, noise_var=fft_psd,
                       threshold_setting=threshold_setting)


def _is_truly_correlated(psd_info: dict) -> bool:
    """Robust two-stage test for genuinely correlated noise.

    Avoids false positives on textured-but-white-noise images where:
      * patch leakage inflates lag-1 to ~0.2 (32-px patches);
      * radial-PSD upscale produces banding artefacts that the
        peak-detector mistakes for periodic peaks.

    Criteria (BOTH must hold):
      1. ``max(|lag1_h|, |lag1_v|) >= _TRUE_CORRELATION_LAG1`` — true
         spatial coupling, not just patch-leakage baseline.
      2. ``CV(radial_psd) >= _TRUE_CORRELATION_CV``  —  the radial PSD
         deviates substantially from flat (white = CV ≈ 1/√N_radial
         ≈ 0.1 for 32-px patches).
    """
    lag1_max = max(abs(psd_info.get('lag1_h', 0.0)),
                   abs(psd_info.get('lag1_v', 0.0)))
    if lag1_max < _TRUE_CORRELATION_LAG1:
        return False

    radial = psd_info.get('radial_psd', None)
    if radial is None or len(radial) < 4:
        return False
    # Restrict to the mid-band where signal leakage is minimal.
    n = len(radial)
    band = np.asarray(radial[n // 4: 3 * n // 4], dtype=np.float64)
    band = band[band > 0]
    if band.size < 3:
        return False
    cv = float(band.std() / max(band.mean(), 1e-12))
    return cv >= _TRUE_CORRELATION_CV


def _denoise_poisson_vst(image: np.ndarray,
                         a_norm: float,
                         b_norm: float,
                         ) -> Tuple[np.ndarray, dict]:
    """Poisson / Poisson-Gaussian denoising via VST + BM3D + inverse."""
    from .vst import vst_bm3d_denoise
    # GAT requires a > 0; protect against degenerate inputs.
    a_eff = max(float(a_norm), 1e-6)
    b_eff = max(float(b_norm), 0.0)
    return vst_bm3d_denoise(image, a=a_eff, b=b_eff)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cascade router
# ─────────────────────────────────────────────────────────────────────────────

# Decision thresholds (tunable, but each has a calibration justification):
#
#   _MIN_A_FOR_VST          : minimum normalised Poisson gain to engage VST.
#                             For a clean image scaled to [0,1], a_norm =
#                             pyatykh_a/255.  Below 1e-6 the Poisson term
#                             contributes < 1e-6 · mean(I) ≈ 1e-7 to Var,
#                             negligible vs typical b_norm > 1e-5.
#   _MIN_SIGMA_FOR_DENOISE  : if both Pyatykh σ and Chen σ fall below this,
#                             skip denoising (image is essentially clean).
#   _MIN_IMPULSE_DENSITY    : impulse mask is treated as noise only above this.
#                             0.005 (= 0.5 %) matches remove_impulse_noise's
#                             own density_threshold default in the original
#                             impulse_noise_estimation module.
#   _DEFAULT_PEAK_THRESHOLD : periodic-peak ratio threshold.  The peak
#                             detector docstring states real peaks have
#                             ratio > 2000 and false positives < 50; at 100
#                             the detector fires on every textured image.
#                             We set 2000.0 as a conservative production
#                             default; pass a lower value explicitly to be
#                             more aggressive.
_MIN_A_FOR_VST = 1e-6
_MIN_SIGMA_FOR_DENOISE = 1e-3
_MIN_IMPULSE_DENSITY = 5e-3
_DEFAULT_PEAK_THRESHOLD = 2000.0
# lag-1 gate raised from 0.4 → 0.5: on 32-px patches the patch-leakage
# baseline is ≈20% (CLT for n=32), so any lag1 in [0.2, 0.4] is residual
# image content rather than noise correlation.  At 0.5 we are well above
# the leakage floor.
_PERIODIC_LAG1_GATE = 0.5
_TRUE_CORRELATION_LAG1 = 0.5
# Coefficient-of-variation threshold for the radial PSD profile.
# White noise has CV ≈ 1/√N_radial ≈ 0.1; structured colored noise
# (Gaussian-blurred white, 1/f, etc.) has CV ≳ 0.3.
_TRUE_CORRELATION_CV = 0.3


def robust_denoise(image: np.ndarray,
                   verbose: bool = False,
                   pch_size: int = 32,
                   n_smooth: int = 100,
                   peak_threshold: float = _DEFAULT_PEAK_THRESHOLD,
                   notch_radius: int = 3,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """End-to-end mathematically grounded denoising cascade.

    Pipeline:
        1. Impulse detection → adaptive median (only on flagged pixels).
        2. Periodic-peak detection → notch filter; re-estimate residual.
        3. Pyatykh PCA on residual → (a, b, type).
        4. Branch:
              type ∈ {poisson, poisson_gaussian} & a meaningful → GAT-VST + BM3D
              else if PSD class == 'correlated'                 → ACT colored
              else (white Gaussian or unknown)                   → BM3D scalar σ

    Steps 1, 2 short-circuit when the corresponding noise is absent,
    making the pipeline safe on clean images.

    Parameters
    ----------
    image : ndarray, gray or RGB, any scale.
    verbose : bool, print decisions to stdout.
    pch_size, n_smooth, peak_threshold, notch_radius : forwarded to detectors.

    Returns
    -------
    cleaned : ndarray (H, W) float64 [0, 1]
    info : dict
        ``'log'``       — list[str], one line per pipeline step taken
        ``'detector'``  — descriptor returned by ``analyze_noise``
        ``'final_pca'`` — PCA estimate on the *cleaned* output (validation)
        ``'branch'``    — 'vst' | 'act_colored' | 'bm3d_white' | 'noop'
    """
    img = _to_grayscale_norm(image)
    log: list[str] = []

    # ── Stage 1: impulse pre-pass ────────────────────────────────────────
    imp_info = detect_impulse_noise(img)
    imp_density = float(imp_info.get('density', 0.0))
    if imp_info.get('has_impulse', False) and imp_density > _MIN_IMPULSE_DENSITY:
        img = adaptive_median_filter(img, imp_info['impulse_mask'])
        log.append(f"[1] impulse density={imp_density:.4f} → adaptive_median")
        if verbose:
            print(log[-1])
    elif imp_density > 0:
        log.append(f"[1] impulse density={imp_density:.4f} (≤ {_MIN_IMPULSE_DENSITY}) → skip")

    # ── Stage 2: periodic notch ──────────────────────────────────────────
    # Gate: the periodic branch is only safe when the residual lag-1
    # autocorrelation is *low*.  High lag-1 means the noise has broadband
    # correlation (colored Gaussian), in which case the "peaks" are
    # actually a smeared low-frequency mound — notching them destroys the
    # very structure ACT's colored branch is designed to model.
    psd = analyze_noise_psd(img, pch_size=pch_size,
                            n_smooth=n_smooth,
                            peak_threshold=peak_threshold)
    lag1_max = max(abs(psd.get('lag1_h', 0.0)),
                   abs(psd.get('lag1_v', 0.0)))
    if (psd.get('has_periodic', False) and psd.get('periodic_peaks')
            and lag1_max < _PERIODIC_LAG1_GATE):
        n_peaks = len(psd['periodic_peaks'])
        img = notch_filter(img, psd['periodic_peaks'],
                           notch_radius=notch_radius)
        log.append(f"[2] periodic peaks={n_peaks}, lag1_max={lag1_max:.3f} "
                   f"→ notch_filter")
        if verbose:
            print(log[-1])
        # re-estimate residual PSD for subsequent decisions
        psd = analyze_noise_psd(img, pch_size=pch_size,
                                n_smooth=n_smooth,
                                peak_threshold=peak_threshold)
    elif psd.get('has_periodic', False):
        log.append(f"[2] periodic peaks={len(psd.get('periodic_peaks', []))}, "
                   f"lag1_max={lag1_max:.3f} (≥ {_PERIODIC_LAG1_GATE}) "
                   f"→ skip notch (broadband correlation, treat as colored)")

    # ── Stage 3: descriptors on residual ─────────────────────────────────
    pca = estimate_noise_params(img)
    chen_sigma = float(estimate_noise_level(img))
    a_norm = float(pca.get('a', 0.0)) / 255.0
    b_norm = float(pca.get('b', 0.0)) / (255.0 ** 2)
    sigma_norm = float(pca.get('sigma_norm', 0.0))
    pca_type = pca.get('noise_type', 'unknown')
    psd_class = psd.get('noise_class', 'white')

    # Pick the best scalar σ estimate for the white-Gaussian branch.
    # Pyatykh σ_norm is intensity-dependent (eval at mean brightness) and
    # tends to slightly *over*-estimate; Chen is an aggregate MAD.  Use the
    # smaller of the two if they agree to within 2× — otherwise trust Chen
    # which is more robust on textured images.
    if sigma_norm > 0 and chen_sigma > 0 and \
       0.5 * sigma_norm <= chen_sigma <= 2.0 * sigma_norm:
        scalar_sigma = float(min(sigma_norm, chen_sigma))
    else:
        scalar_sigma = float(chen_sigma if chen_sigma > 0 else sigma_norm)

    log.append(
        f"[3] PCA: type={pca_type}, a_norm={a_norm:.4g}, b_norm={b_norm:.4g}, "
        f"σ_pca={sigma_norm:.4g} | Chen σ={chen_sigma:.4g} → σ*={scalar_sigma:.4g} | "
        f"PSD: class={psd_class}, lag1=({psd.get('lag1_h', 0):.3f},"
        f"{psd.get('lag1_v', 0):.3f})")
    if verbose:
        print(log[-1])

    # ── Stage 4: main denoiser branch ────────────────────────────────────
    # Routing rules (priority top → bottom):
    #   (a) Poisson family with measurable Poisson gain → VST + BM3D
    #   (b) Genuinely correlated noise (high lag-1 AND high radial-CV via
    #       _is_truly_correlated) → ACT colored.  Both criteria are
    #       required to avoid over-smoothing AWGN images where patch
    #       leakage produces lag-1 ≈ 0.2 baseline.
    #   (c) White additive noise with measurable σ → BM3D scalar σ
    #       (faster + edge-preserving vs ACT colored 31×31 window).
    #   (d) Otherwise → no-op
    truly_correlated = _is_truly_correlated(psd)

    branch = 'noop'
    branch_info: dict = {}

    poisson_path = (pca_type in ('poisson', 'poisson_gaussian')
                    and a_norm > _MIN_A_FOR_VST)

    # Energy-based σ estimator that *includes* low-frequency correlated
    # power (unlike scalar PCA σ, which is killed by the smooth-patch
    # detrending step for colored noise).  PSD is normalised so its
    # spatial mean ≈ noise variance.
    psd_2d = psd.get('psd_2d', None)
    psd_sigma = float(np.sqrt(max(0.0, psd_2d.mean()))) \
        if psd_2d is not None else 0.0

    # Don't engage ACT-colored on essentially-clean images: the radial-CV
    # criterion can fire on structured *signal* content (textures,
    # gradients) rather than noise.  Require measurable noise level —
    # use the PSD-energy σ which is sensitive to both white and colored
    # noise components.
    correlated_path = (not poisson_path) and truly_correlated \
        and (psd_sigma > _MIN_SIGMA_FOR_DENOISE)

    white_path = (not poisson_path and not correlated_path
                  and scalar_sigma > _MIN_SIGMA_FOR_DENOISE)

    if poisson_path:
        img, branch_info = _denoise_poisson_vst(img, a_norm, b_norm)
        branch = 'vst'
        log.append(f"[4] BRANCH=VST(a={a_norm:.4g}, b={b_norm:.4g}) [BM3D core]")
    elif correlated_path:
        img, branch_info = _denoise_colored_act(img, psd['psd_2d'])
        branch = 'act_colored'
        log.append(
            f"[4] BRANCH=ACT-colored (lag1_h={psd.get('lag1_h', 0):.3f}, "
            f"lag1_v={psd.get('lag1_v', 0):.3f}, radial-CV ok) "
            f"→ ACT info={branch_info}")
    elif white_path:
        img = _denoise_white_bm3d(img, scalar_sigma)
        branch = 'bm3d_white'
        branch_info = {'method': 'bm3d', 'sigma': scalar_sigma}
        log.append(f"[4] BRANCH=BM3D-white (σ={scalar_sigma:.4g}, "
                   f"truly_correlated={truly_correlated})")
    else:
        log.append(f"[4] BRANCH=NOOP (image looks clean: σ={scalar_sigma:.4g})")

    if verbose:
        print(log[-1])

    info = {
        'log': log,
        'branch': branch,
        'branch_info': branch_info,
        'detector': {
            'impulse': imp_info,
            'psd': {k: v for k, v in psd.items()
                    if k not in ('psd_2d', 'psd_2d_patches',
                                 'radial_freq', 'radial_psd')},
            'pca': pca,
            'pca_norm': {'a': a_norm, 'b': b_norm,
                         'sigma_norm': sigma_norm, 'noise_type': pca_type},
            'chen_sigma_norm': chen_sigma,
        },
    }
    return img, info
