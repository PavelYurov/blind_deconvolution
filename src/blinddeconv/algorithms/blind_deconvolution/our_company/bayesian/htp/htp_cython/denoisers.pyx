"""
denoisers.pyx

Модуль применения алгоритмов шумоподавления для конвейера слепой деконволюции HTP.

Алгоритмы вызываются на разных стадиях работы:
    pre_pyramid  - обработка полного нормализованного изображения до построения пирамиды.
    pre_kernel   - промежуточная обработка скрытого изображения перед обновлением ядра.
    pre_nonblind - финальная обработка перед неслепой деконволюцией.

Доступные методы:
    none            - без фильтрации.
    tv              - полная вариация.
    nlm             - нелокальные средние.
    bilateral       - двусторонняя фильтрация.
    guided          - направляемый фильтр.
    bm3d            - блочная фильтрация 3D.
    act             - адаптивная пороговая обработка курвлетов.
    vst_bm3d        - преобразование Энскомба в связке с BM3D.
    screenot        - сжатие сингулярных значений.
    adaptive_median - адаптивный медианный фильтр.

Если алгоритму требуется оценка дисперсии шума и она не передана явно, 
значение вычисляется автоматически по входному изображению.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

__all__ = ['apply_denoiser']

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root")
        path = path.parent
    return path


_CURRENT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _find_project_root(_CURRENT_FILE)
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# --- Направляемый фильтр ---
def _guided_filter(I: np.ndarray, p: np.ndarray, radius: int, eps: float) -> np.ndarray:
    size = 2 * int(radius) + 1
    mean_I = uniform_filter(I, size)
    mean_p = uniform_filter(p, size)
    corr_Ip = uniform_filter(I * p, size)
    var_I = uniform_filter(I * I, size) - mean_I * mean_I
    a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = uniform_filter(a, size)
    mean_b = uniform_filter(b, size)
    return mean_a * I + mean_b


# --- Диспетчер методов шумоподавления ---
def apply_denoiser(img: np.ndarray, method, **params) -> np.ndarray:
    """
    Применяет выбранный метод шумоподавления к двумерному изображению.
    Изображение ожидается в формате float64 со значениями в диапазоне [0, 1].
    """
    if method is None or method == 'none':
        return img.copy()

    # --- Полная вариация ---
    if method == 'tv':
        from skimage.restoration import denoise_tv_chambolle
        weight = float(params.get('weight', 0.05))
        max_num_iter = int(params.get('max_num_iter', 100))
        return denoise_tv_chambolle(img, weight=weight, max_num_iter=max_num_iter)

    # --- Нелокальные средние ---
    if method == 'nlm':
        from skimage.restoration import denoise_nl_means, estimate_sigma
        sigma = params.get('sigma', None)
        if sigma is None:
            sigma = float(estimate_sigma(img))
        patch_size = int(params.get('patch_size', 5))
        patch_distance = int(params.get('patch_distance', 6))
        h = float(params.get('h', 0.8 * sigma))
        return denoise_nl_means(
            img, h=h, patch_size=patch_size,
            patch_distance=patch_distance, fast_mode=True,
            sigma=sigma,
        )

    # --- Двусторонняя фильтрация ---
    if method == 'bilateral':
        from skimage.restoration import denoise_bilateral, estimate_sigma
        sigma_color = params.get('sigma_color', None)
        if sigma_color is None:
            sigma_color = float(estimate_sigma(img))
        sigma_spatial = float(params.get('sigma_spatial', 1.0))
        return denoise_bilateral(
            img, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

    # --- Направляемый фильтр ---
    if method == 'guided':
        radius = int(params.get('radius', 5))
        eps = float(params.get('eps', 0.01))
        return _guided_filter(img, img, radius, eps)

    # --- BM3D ---
    if method == 'bm3d':
        try:
            import bm3d as bm3d_lib
        except ImportError as e:
            raise ImportError("Требуется пакет bm3d") from e
        from skimage.restoration import estimate_sigma
        sigma_psd = params.get('sigma_psd', None)
        if sigma_psd is None:
            sigma_psd = float(estimate_sigma(img))
        return bm3d_lib.bm3d(img, sigma_psd=sigma_psd)

    # --- Адаптивная курвлет-фильтрация ---
    if method == 'act':
        from blinddeconv.algorithms.mod_cython._build_pyd.act_denoise import act_denoise
        nv = params.get('noise_var', None)
        ts = params.get('threshold_setting', 's')
        result, _ = act_denoise(img, noise_var=nv, threshold_setting=ts)
        return result

    # --- Преобразование Энскомба с BM3D ---
    if method == 'vst_bm3d':
        from blinddeconv.algorithms.mod_cython._build_pyd.vst import vst_bm3d_denoise
        result, _ = vst_bm3d_denoise(
            img,
            noise_info=params.get('noise_info', None),
            a=params.get('a', None),
            b=params.get('b', None),
            sigma=params.get('sigma', None),
            stage_arg=params.get('stage_arg', None),
            verbose=params.get('verbose', False),
        )
        return result

    # --- Сжатие сингулярных значений ---
    if method == 'screenot':
        from blinddeconv.algorithms.mod_cython._build_pyd.screenot import screenot_denoise
        return screenot_denoise(
            img,
            k=int(params.get('k', 10)),
            strategy=params.get('strategy', 'i'),
            mode=params.get('mode', 'full'),
            patch_size=params.get('patch_size', None),
            stride=params.get('stride', None),
        )

    # --- Удаление импульсного шума ---
    if method == 'adaptive_median':
        from blinddeconv.algorithms.mod_cython._build_pyd.impulse_noise_estimation import (
            detect_impulse_noise, adaptive_median_filter,
        )
        max_window = int(params.get('max_window', 7))
        mask = params.get('impulse_mask', None)
        if mask is None:
            mask = detect_impulse_noise(
                img,
                outlier_window=int(params.get('outlier_window', 5)),
                outlier_threshold=float(params.get('outlier_threshold', 0.15)),
            )
        return adaptive_median_filter(img, mask, max_window=max_window)

    raise ValueError(f"Неизвестный метод шумоподавления: {method}")