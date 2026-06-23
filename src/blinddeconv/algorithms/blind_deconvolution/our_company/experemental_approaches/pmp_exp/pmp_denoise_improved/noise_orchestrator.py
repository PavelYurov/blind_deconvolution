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


def psd_to_act_fft_format(psd_centered: np.ndarray) -> np.ndarray:


    psd_centered = np.asarray(psd_centered, dtype=np.float64)
    H, W = psd_centered.shape
    return np.fft.ifftshift(psd_centered) * (H * W)


def _to_grayscale_norm(image: np.ndarray) -> np.ndarray:

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
    fft_psd = psd_to_act_fft_format(psd_centered)
    return act_denoise(image, noise_var=fft_psd,
                       threshold_setting=threshold_setting)


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


def _denoise_poisson_vst(image: np.ndarray,
                         a_norm: float,
                         b_norm: float,
                         ) -> Tuple[np.ndarray, dict]:

    from .vst import vst_bm3d_denoise

    a_eff = max(float(a_norm), 1e-6)
    b_eff = max(float(b_norm), 0.0)
    return vst_bm3d_denoise(image, a=a_eff, b=b_eff)


_MIN_A_FOR_VST = 1e-6
_MIN_SIGMA_FOR_DENOISE = 1e-3
_MIN_IMPULSE_DENSITY = 5e-3
_DEFAULT_PEAK_THRESHOLD = 2000.0


_PERIODIC_LAG1_GATE = 0.5
_TRUE_CORRELATION_LAG1 = 0.5


_TRUE_CORRELATION_CV = 0.3


def robust_denoise(image: np.ndarray,
                   verbose: bool = False,
                   pch_size: int = 32,
                   n_smooth: int = 100,
                   peak_threshold: float = _DEFAULT_PEAK_THRESHOLD,
                   notch_radius: int = 3,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:


    img = _to_grayscale_norm(image)
    log: list[str] = []


    imp_info = detect_impulse_noise(img)
    imp_density = float(imp_info.get('density', 0.0))
    if imp_info.get('has_impulse', False) and imp_density > _MIN_IMPULSE_DENSITY:
        img = adaptive_median_filter(img, imp_info['impulse_mask'])
        log.append(f"[1] impulse density={imp_density:.4f} → adaptive_median")
        if verbose:
            print(log[-1])
    elif imp_density > 0:
        log.append(f"[1] impulse density={imp_density:.4f} (≤ {_MIN_IMPULSE_DENSITY}) → skip")


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

        psd = analyze_noise_psd(img, pch_size=pch_size,
                                n_smooth=n_smooth,
                                peak_threshold=peak_threshold)
    elif psd.get('has_periodic', False):
        log.append(f"[2] periodic peaks={len(psd.get('periodic_peaks', []))}, "
                   f"lag1_max={lag1_max:.3f} (≥ {_PERIODIC_LAG1_GATE}) "
                   f"→ skip notch (broadband correlation, treat as colored)")


    pca = estimate_noise_params(img)
    chen_sigma = float(estimate_noise_level(img))
    a_norm = float(pca.get('a', 0.0)) / 255.0
    b_norm = float(pca.get('b', 0.0)) / (255.0 ** 2)
    sigma_norm = float(pca.get('sigma_norm', 0.0))
    pca_type = pca.get('noise_type', 'unknown')
    psd_class = psd.get('noise_class', 'white')


    if sigma_norm > 0 and chen_sigma > 0 and\
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


    truly_correlated = _is_truly_correlated(psd)

    branch = 'noop'
    branch_info: dict = {}

    poisson_path = (pca_type in ('poisson', 'poisson_gaussian')
                    and a_norm > _MIN_A_FOR_VST)


    psd_2d = psd.get('psd_2d', None)
    psd_sigma = float(np.sqrt(max(0.0, psd_2d.mean())))\
        if psd_2d is not None else 0.0


    correlated_path = (not poisson_path) and truly_correlated\
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
