from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from .noise_psd_analysis import analyze_noise_psd, notch_filter
from .pyatykh_noise_reconstruction import estimate_noise_params
from .chen_noise_estimate import estimate_noise_level

__all__ = ['sanitize', 'SanitationResult']

_DEFAULT_NOISE_FLOOR = 0.005

_BORDERLINE_SIGMA = 0.010

_BORDERLINE_LAG1_GATE = 0.65

_ACT_COLORED_MIN_SIGMA = 0.020

_MIN_A_FOR_VST = 1e-6
_MIN_IMPULSE_DENSITY = 5e-3
_DEFAULT_PEAK_THRESHOLD = 2000.0
_PERIODIC_LAG1_GATE = 0.5
_TRUE_CORRELATION_LAG1 = 0.5
_TRUE_CORRELATION_CV = 0.3

@dataclass
class SanitationResult:

    image_clean: np.ndarray
    noise_info: Dict[str, Any]
    actions: List[str]
    residual_sigma: float
    residual_type: str
    branch: str
    raw: Dict[str, Any] = field(default_factory=dict)

def _to_grayscale_norm(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = (0.2989 * img[..., 0] + 0.5870 * img[..., 1]
                   + 0.1140 * img[..., 2])
        elif img.shape[2] == 1:
            img = img[..., 0]
        else:
            raise ValueError(f"Expected 1 or 3 channels, got {img.shape[2]}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got ndim={img.ndim}")
    if img.max() > 1.5:
        img = img / 255.0
    return img

def _psd_to_act_fft_format(psd_centered: np.ndarray) -> np.ndarray:

    psd_centered = np.asarray(psd_centered, dtype=np.float64)
    H, W = psd_centered.shape
    return np.fft.ifftshift(psd_centered) * (H * W)

def _is_truly_correlated(psd_info: dict) -> bool:

    lag1_max = max(abs(psd_info.get('lag1_h', 0.0)),
                   abs(psd_info.get('lag1_v', 0.0)))
    if lag1_max < _TRUE_CORRELATION_LAG1:
        return False
    radial = psd_info.get('radial_psd', None)
    if radial is None or len(radial) < 4:
        return False
    n = len(radial)
    band = np.asarray(radial[n // 4: 3 * n // 4], dtype=np.float64)
    band = band[band > 0]
    if band.size < 3:
        return False
    cv = float(band.std() / max(band.mean(), 1e-12))
    return cv >= _TRUE_CORRELATION_CV

def _denoise_white_bm3d(image: np.ndarray, sigma: float) -> np.ndarray:

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

    from .act_denoise import act_denoise
    fft_psd = _psd_to_act_fft_format(psd_centered)
    return act_denoise(image, noise_var=fft_psd,
                       threshold_setting=threshold_setting)

def _denoise_poisson_vst(image: np.ndarray,
                         a_norm: float, b_norm: float,
                         ) -> Tuple[np.ndarray, dict]:

    from .vst import vst_bm3d_denoise
    a_eff = max(float(a_norm), 1e-6)
    b_eff = max(float(b_norm), 0.0)
    return vst_bm3d_denoise(image, a=a_eff, b=b_eff)

def sanitize(image: np.ndarray,
             *,
             profile: str = 'auto',
             noise_floor: float = _DEFAULT_NOISE_FLOOR,
             borderline_sigma: float = _BORDERLINE_SIGMA,
             borderline_lag1_gate: float = _BORDERLINE_LAG1_GATE,
             act_colored_min_sigma: float = _ACT_COLORED_MIN_SIGMA,
             verbose: bool = False) -> SanitationResult:

    if profile != 'auto':
        raise ValueError(f"profile={profile!r} not supported; use 'auto'.")

    log: List[str] = []
    img = _to_grayscale_norm(image)

    chen_pre = float(estimate_noise_level(img))
    if chen_pre < noise_floor:
        log.append(f"[0] JPEG-floor gate: σ_chen={chen_pre:.5f} < "
                   f"floor={noise_floor:.4f} → NO-OP (treated as clean)")
        if verbose:
            print(log[-1])
        return SanitationResult(
            image_clean=img.copy(),
            noise_info={
                'method': 'sanitation', 'sigma_norm': chen_pre,
                'sigma': chen_pre * 255.0, 'sigma_norm_pre': chen_pre,
                'noise_type': 'clean', 'a_norm': 0.0, 'b_norm': 0.0,
                'branch': 'noop',
            },
            actions=log,
            residual_sigma=chen_pre,
            residual_type='clean',
            branch='noop',
            raw={'log': log, 'reason': 'below_noise_floor'},
        )

    imp_info = detect_impulse_noise(img)
    imp_density = float(imp_info.get('density', 0.0))
    if imp_info.get('has_impulse', False) and imp_density > _MIN_IMPULSE_DENSITY:
        img = adaptive_median_filter(img, imp_info['impulse_mask'])
        log.append(f"[1] impulse density={imp_density:.4f} → adaptive_median")
        if verbose:
            print(log[-1])
    elif imp_density > 0:
        log.append(f"[1] impulse density={imp_density:.4f} "
                   f"(≤ {_MIN_IMPULSE_DENSITY}) → skip")

    psd = analyze_noise_psd(img, peak_threshold=_DEFAULT_PEAK_THRESHOLD)
    lag1_max = max(abs(psd.get('lag1_h', 0.0)), abs(psd.get('lag1_v', 0.0)))
    if (psd.get('has_periodic', False) and psd.get('periodic_peaks')
            and lag1_max < _PERIODIC_LAG1_GATE):
        n_peaks = len(psd['periodic_peaks'])
        img = notch_filter(img, psd['periodic_peaks'], notch_radius=3)
        log.append(f"[2] periodic peaks={n_peaks}, lag1_max={lag1_max:.3f} "
                   f"→ notch_filter")
        if verbose:
            print(log[-1])
        psd = analyze_noise_psd(img, peak_threshold=_DEFAULT_PEAK_THRESHOLD)
        lag1_max = max(abs(psd.get('lag1_h', 0.0)),
                       abs(psd.get('lag1_v', 0.0)))

    pca = estimate_noise_params(img)
    chen_sigma = float(estimate_noise_level(img))
    a_norm = float(pca.get('a', 0.0)) / 255.0
    b_norm = float(pca.get('b', 0.0)) / (255.0 ** 2)
    sigma_pca_norm = float(pca.get('sigma_norm', 0.0))
    pca_type = pca.get('noise_type', 'unknown')

    if sigma_pca_norm > 0 and chen_sigma > 0 and\
       0.5 * sigma_pca_norm <= chen_sigma <= 2.0 * sigma_pca_norm:
        scalar_sigma = float(min(sigma_pca_norm, chen_sigma))
    else:
        scalar_sigma = float(chen_sigma if chen_sigma > 0 else sigma_pca_norm)

    log.append(
        f"[3] σ_chen={chen_sigma:.4g}, σ_pca={sigma_pca_norm:.4g} → "
        f"σ*={scalar_sigma:.4g} | type={pca_type}, a={a_norm:.3g}, "
        f"b={b_norm:.3g} | lag1=({psd.get('lag1_h', 0):.3f},"
        f"{psd.get('lag1_v', 0):.3f})")
    if verbose:
        print(log[-1])

    poisson_path = (pca_type in ('poisson', 'poisson_gaussian')
                    and a_norm > _MIN_A_FOR_VST)
    base_correlated = _is_truly_correlated(psd)

    in_borderline = scalar_sigma < borderline_sigma
    if in_borderline and lag1_max < borderline_lag1_gate:
        if base_correlated:
            log.append(f"[4-gate] borderline σ={scalar_sigma:.4g} & "
                       f"lag1_max={lag1_max:.3f} < {borderline_lag1_gate} "
                       f"→ override correlated→white")
        truly_correlated = False
    else:
        truly_correlated = base_correlated

    if truly_correlated and scalar_sigma < act_colored_min_sigma:
        log.append(f"[4-gate] σ={scalar_sigma:.4g} < "
                   f"{act_colored_min_sigma} → ACT-colored disabled, "
                   f"use BM3D-white")
        truly_correlated = False

    correlated_path = (not poisson_path) and truly_correlated
    white_path = (not poisson_path and not correlated_path
                  and scalar_sigma > noise_floor)

    branch = 'noop'
    branch_info: Dict[str, Any] = {}

    if poisson_path:
        img, branch_info = _denoise_poisson_vst(img, a_norm, b_norm)
        branch = 'vst'
        log.append(f"[4] BRANCH=VST(a={a_norm:.4g}, b={b_norm:.4g})")
    elif correlated_path:
        img, branch_info = _denoise_colored_act(img, psd['psd_2d'])
        branch = 'act_colored'
        log.append(f"[4] BRANCH=ACT-colored "
                   f"(lag1=({psd.get('lag1_h', 0):.3f},"
                   f"{psd.get('lag1_v', 0):.3f}), σ={scalar_sigma:.4g})")
    elif white_path:
        img = _denoise_white_bm3d(img, scalar_sigma)
        branch = 'bm3d_white'
        branch_info = {'method': 'bm3d', 'sigma': scalar_sigma}
        log.append(f"[4] BRANCH=BM3D-white (σ={scalar_sigma:.4g})")
    else:
        log.append(f"[4] BRANCH=NOOP (σ={scalar_sigma:.4g} ≤ floor)")

    if verbose:
        print(log[-1])

    residual_sigma = float(estimate_noise_level(img))
    residual_type = {
        'vst': 'awgn',
        'act_colored': 'mild_colored',
        'bm3d_white': 'awgn',
        'noop': 'clean',
    }[branch]

    noise_info = {
        'method': 'sanitation',
        'sigma_norm': residual_sigma,
        'sigma': residual_sigma * 255.0,
        'sigma_norm_pre': chen_pre,
        'noise_type': pca_type,
        'a_norm': a_norm,
        'b_norm': b_norm,
        'branch': branch,
    }

    return SanitationResult(
        image_clean=img.astype(np.float64, copy=False),
        noise_info=noise_info,
        actions=log,
        residual_sigma=residual_sigma,
        residual_type=residual_type,
        branch=branch,
        raw={
            'log': log,
            'branch_info': branch_info,
            'detector': {
                'impulse': imp_info,
                'psd': {k_: v for k_, v in psd.items()
                        if k_ not in ('psd_2d', 'psd_2d_patches',
                                      'radial_freq', 'radial_psd')},
                'pca': pca,
                'chen_sigma_norm_pre': chen_pre,
                'scalar_sigma': scalar_sigma,
            },
        },
    )
