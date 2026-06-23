"""
htp.pyx

Алгоритм быстрой слепой деконволюции с использованием априорных распределений 
с тяжелыми хвостами (Blind Image Deblurring Using Heavy-Tailed Priors).

Последовательность операций:
    1. Нормализация входного изображения в диапазон [0, 1].
    2. Построение пирамиды изображений от грубого к точному масштабу для 
       центральной области интереса (зеленый канал для RGB, полное изображение для градаций серого).
    3. Многомасштабная чередующаяся MAP-оценка для скрытого изображения u 
       и функции рассеяния точки (ФРТ) h с Lp-регуляризацией на градиенты 
       изображения (p < 1) и L1-регуляризацией на ФРТ. Оптимизация выполняется 
       через полуквадратичное расщепление и итерации Брегмана в частотной области.
    4. Финальная неслепая деконволюция полного изображения с усиленным 
       согласованием данных и TV-подобной регуляризацией (Lp_nonblind = 1).
    5. Возврат восстановленного изображения и оцененного ядра.

Литература:
[1] J. Kotera, F. Sroubek, P. Milanfar,
    "Blind Deconvolution Using Alternating Maximum a Posteriori
     Estimation with Heavy-tailed Priors", CAIP 2013.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict, Optional, Callable

# --- Внутренний импорт базового класса ---
import sys
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

from blinddeconv.algorithms.base import DeconvolutionAlgorithm                  
# -----------------------------------------

from .solvers import mc_restoration


class HTP_BD(DeconvolutionAlgorithm):
    """
    Слепая деконволюция с априорными распределениями с тяжелыми хвостами [1].

    Параметры
    ----------
    kernel_size : int
        Пространственный размер неизвестной ФРТ (квадратная область). По умолчанию 31.
    Lp : float
        Экспонента Lp-нормы для регуляризации градиентов скрытого изображения 
        на этапе оценки ФРТ (0 < p <= 1). По умолчанию 0.3 (тяжелые хвосты).
    gamma : float
        Вес согласования данных при оценке ФРТ. Рекомендуется настраивать 
        в зависимости от уровня шума. По умолчанию 1e2.
    alpha_u : float
        Относительный вес априорного распределения изображения. По умолчанию 1e-2.
    beta_u : float
        Относительный вес штрафа расщепления Брегмана для изображения. По умолчанию 1e0.
    alpha_h : float
        Относительный вес L1-регуляризации ФРТ. По умолчанию 1e1.
    beta_h : float
        Относительный вес штрафа расщепления для ФРТ. По умолчанию 1e4.
    centering_threshold : float
        Порог для центрирования ФРТ между итерациями. По умолчанию 20/255.
    gamma_nonblind : float
        Вес согласования данных для финальной неслепой деконволюции. По умолчанию 2e1.
    beta_u_nonblind : float
        Вес расщепления для финального неслепого шага. По умолчанию 1e-2.
    Lp_nonblind : float
        Экспонента Lp для финального неслепого шага. По умолчанию 1.0.
    maxiter : int
        Внешние чередующиеся итерации на одном масштабе пирамиды. По умолчанию 10.
    maxiter_u : int
        Внутренние итерации оценки скрытого изображения. По умолчанию 10.
    maxiter_h : int
        Внутренние итерации оценки ФРТ. По умолчанию 10.
    ccreltol : float
        Относительный критерий остановки для внутренних циклов. По умолчанию 1e-3.
    MSlevels : int
        Количество масштабов многомасштабной пирамиды (>= 1). По умолчанию 4.
    maxROIsize : tuple[int, int]
        Центральная область интереса, используемая для оценки ядра. По умолчанию (1024, 1024).
    verbose : int
        Уровень детализации вывода. По умолчанию 0.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        Lp: float = 0.3,
        gamma: float = 1e2,
        alpha_u: float = 1e-2,
        beta_u: float = 1e0,
        alpha_h: float = 1e1,
        beta_h: float = 1e4,
        centering_threshold: float = 20.0 / 255.0,
        gamma_nonblind: float = 2e1,
        beta_u_nonblind: float = 1e-2,
        Lp_nonblind: float = 1.0,
        maxiter: int = 10,
        maxiter_u: int = 10,
        maxiter_h: int = 10,
        ccreltol: float = 1e-3,
        MSlevels: int = 4,
        maxROIsize: Tuple[int, int] = (1024, 1024),
        verbose: int = 0,
        kernel_flip: str = 'none',
        auto_recenter: bool = False,
        recenter_mode: str = 'centroid',
        kernel_thresh: float = 0.0,
        iterative_recenter: bool = True,
        pre_pyramid: Optional[str] = None,
        pre_pyramid_params: Optional[Dict[str, Any]] = None,
        pre_kernel: Optional[str] = None,
        pre_kernel_params: Optional[Dict[str, Any]] = None,
        pre_nonblind: Optional[str] = None,
        pre_nonblind_params: Optional[Dict[str, Any]] = None,
        noise_estimation: str = 'none',
        noise_estimation_params: Optional[Dict[str, Any]] = None,
        impulse_preprocess: str = 'none',
        impulse_params: Optional[Dict[str, Any]] = None,
        auto_mode: str = 'off',
        auto_mode_overrides: Optional[Dict[str, Any]] = None,
        iteration_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        nonblind_method: str = 'fft_cg_sr_al',
        lambda_tv: float = 4e-3,
        lambda_l0: float = 2e-3,
        weight_ring: float = 0.5,
        firls_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name='HTP-BD')

        self.kernel_size = int(kernel_size)
        self.Lp = float(Lp)
        self.gamma = float(gamma)
        self.alpha_u = float(alpha_u)
        self.beta_u = float(beta_u)
        self.alpha_h = float(alpha_h)
        self.beta_h = float(beta_h)
        self.centering_threshold = float(centering_threshold)
        self.gamma_nonblind = float(gamma_nonblind)
        self.beta_u_nonblind = float(beta_u_nonblind)
        self.Lp_nonblind = float(Lp_nonblind)
        self.maxiter = int(maxiter)
        self.maxiter_u = int(maxiter_u)
        self.maxiter_h = int(maxiter_h)
        self.ccreltol = float(ccreltol)
        self.MSlevels = int(MSlevels)
        self.maxROIsize = tuple(maxROIsize)
        self.verbose = int(verbose)

        if kernel_flip not in ('none', 'lr', 'ud', 'rot180'):
            raise ValueError(f"Недопустимое значение kernel_flip: {kernel_flip}")
        self.kernel_flip = kernel_flip

        if recenter_mode not in ('centroid', 'peak', 'masscentroid'):
            raise ValueError(f"Недопустимый recenter_mode: {recenter_mode}")
        self.auto_recenter = bool(auto_recenter)
        self.recenter_mode = recenter_mode
        self.kernel_thresh = float(kernel_thresh)
        self.iterative_recenter = bool(iterative_recenter)

        # --- Хуки фильтрации ---
        self.pre_pyramid = pre_pyramid
        self.pre_pyramid_params = dict(pre_pyramid_params or {})
        self.pre_kernel = pre_kernel
        self.pre_kernel_params = dict(pre_kernel_params or {})
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = dict(pre_nonblind_params or {})

        # --- Оценка шума ---
        if noise_estimation not in ('none', 'chen', 'pyatykh'):
            raise ValueError(f"Недопустимый метод оценки шума: {noise_estimation}")
        self.noise_estimation = noise_estimation
        self.noise_estimation_params = dict(noise_estimation_params or {})
        self.noise_sigma: Optional[float] = None
        self.noise_info: Optional[Dict[str, Any]] = None

        # --- Предварительная обработка импульсного шума ---
        if impulse_preprocess not in ('none', 'auto'):
            raise ValueError(f"Недопустимый impulse_preprocess: {impulse_preprocess}")
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = dict(impulse_params or {})
        self.impulse_info: Optional[Dict[str, Any]] = None

        # --- Автоматическая конфигурация (auto_mode) ---
        if auto_mode not in ('off', 'auto'):
            raise ValueError(f"Недопустимый auto_mode: {auto_mode}")
        self.auto_mode = auto_mode
        self.auto_mode_overrides = dict(auto_mode_overrides or {})
        self.auto_mode_applied: Optional[Dict[str, Any]] = None

        self.iteration_callback = iteration_callback

        # --- Альтернативные методы неслепой деконволюции ---
        if nonblind_method not in ('fft_cg_sr_al', 'ringing_removal', 'firls'):
            raise ValueError(f"Недопустимый nonblind_method: {nonblind_method}")
        self.nonblind_method = nonblind_method
        self.lambda_tv = float(lambda_tv)
        self.lambda_l0 = float(lambda_l0)
        self.weight_ring = float(weight_ring)
        self.firls_params = dict(firls_params or {})

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def _build_par(self) -> Dict[str, Any]:
        """Сборка параметров для передачи в решатели."""
        gamma = self.gamma
        gamma_nb = self.gamma_nonblind * gamma
        return {
            'verbose': self.verbose,
            'gamma': gamma,
            'Lp': self.Lp,
            'beta_h': self.beta_h * gamma,
            'alpha_h': self.alpha_h * gamma,
            'centering_threshold': self.centering_threshold,
            'beta_u': self.beta_u * gamma,
            'alpha_u': self.alpha_u * gamma,
            'gamma_nonblind': gamma_nb,
            'beta_u_nonblind': self.beta_u_nonblind * gamma_nb,
            'Lp_nonblind': self.Lp_nonblind,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'maxiter': self.maxiter,
            'ccreltol': self.ccreltol,
            'kernel_thresh': self.kernel_thresh,
            'iterative_recenter': self.iterative_recenter,
            'pre_pyramid': self.pre_pyramid,
            'pre_pyramid_params': self.pre_pyramid_params,
            'pre_kernel': self.pre_kernel,
            'pre_kernel_params': self.pre_kernel_params,
            'pre_nonblind': self.pre_nonblind,
            'pre_nonblind_params': self.pre_nonblind_params,
            'iteration_callback': self.iteration_callback,
            'nonblind_method': self.nonblind_method,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'firls_params': self.firls_params,
        }

    @staticmethod
    def _auto_denoiser_config(noise_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Эвристический подбор методов фильтрации на основе оценки шума.
        Параметры самой слепой деконволюции остаются неизменными.
        """
        sigma = float(noise_info.get('sigma_norm', noise_info.get('sigma', 0.0)) or 0.0)
        ntype = str(noise_info.get('noise_type', 'gaussian')).lower()
        a = float(noise_info.get('a', 0.0) or 0.0)

        cfg: Dict[str, Any] = {}

        # Обработка пуассоновского и смешанного шума
        if 'poisson' in ntype and a > 0.0:
            ni = dict(noise_info)
            cfg['pre_pyramid'] = 'vst_bm3d'
            cfg['pre_pyramid_params'] = {'noise_info': ni}
            cfg['pre_kernel'] = 'bilateral'
            cfg['pre_kernel_params'] = {
                'sigma_color': max(sigma * 0.5, 0.01),
                'sigma_spatial': 1.0,
            }
            cfg['pre_nonblind'] = 'vst_bm3d'
            cfg['pre_nonblind_params'] = {'noise_info': ni}
            return cfg

        # Для слабого шума сохраняется поведение по умолчанию
        if sigma < 0.01:
            return cfg                                  

        # Для гауссовского/неизвестного шума используется адаптивная пороговая обработка
        cfg['pre_pyramid'] = 'act'
        cfg['pre_pyramid_params'] = {'noise_var': float(sigma ** 2),
                                     'threshold_setting': 's'}
        cfg['pre_kernel'] = 'bilateral'
        cfg['pre_kernel_params'] = {
            'sigma_color': float(max(sigma * 0.5, 0.005)),
            'sigma_spatial': 1.0,
        }
        cfg['pre_nonblind'] = 'act'
        cfg['pre_nonblind_params'] = {'noise_var': float(sigma ** 2),
                                      'threshold_setting': 's'}
        return cfg

    @staticmethod
    def _default_params_for(
        method: Optional[str],
        sigma: float,
        noise_info: Optional[Dict[str, Any]] = None,
        hook: str = 'pre_pyramid',
    ) -> Dict[str, Any]:
        """
        Формирование параметров шумоподавления на основе дисперсии шума 
        в зависимости от точки вызова (hook).
        """
        if method in (None, 'none'):
            return {}
        sigma = float(max(sigma or 0.0, 1e-6))
        
        # Снижение интенсивности фильтрации перед оценкой ядра
        scale = 0.5 if hook == 'pre_kernel' else 1.0
        s = sigma * scale

        if method == 'tv':
            return {'weight': float(max(s, 0.005))}

        if method == 'nlm':
            return {
                'sigma': float(s),
                'h': float(0.8 * s),
                'patch_size': 5,
                'patch_distance': 6,
            }

        if method == 'bilateral':
            return {
                'sigma_color': float(max(s, 0.005)),
                'sigma_spatial': 1.0,
            }

        if method == 'guided':
            return {
                'radius': 4 if hook == 'pre_kernel' else 5,
                'eps': float(max(s ** 2, 1e-4)),
            }

        if method == 'bm3d':
            if hook == 'pre_pyramid' and sigma >= 0.05:
                return {'sigma_psd': float(1.1 * sigma)}
            return {'sigma_psd': float(s if hook == 'pre_kernel' else sigma)}

        if method == 'vst_bm3d':
            return {'noise_info': dict(noise_info)} if noise_info else {}

        if method == 'act':
            return {'noise_var': float(sigma ** 2)}

        if method == 'screenot':
            return {}

        if method == 'adaptive_median':
            return {}

        return {}

    def _recenter_kernel_and_image(
        self, H: np.ndarray, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Пространственное центрирование ядра с противоположным сдвигом изображения.
        Сохраняет свертку инвариантной, предотвращая глобальное смещение результата.
        
        Используется ограничивающая рамка порога ядра (bounding box) для робастного 
        вычисления центра, так как прямой центр масс подвержен смещению из-за шума.
        """
        kh, kw = H.shape
        cy_int = kh // 2
        cx_int = kw // 2

        # Определение требуемого сдвига (sy, sx)
        if self.recenter_mode == 'peak':
            iy, ix = np.unravel_index(int(np.argmax(H)), H.shape)
            sy, sx = int(cy_int - iy), int(cx_int - ix)

        elif self.recenter_mode == 'masscentroid':
            Hp = np.maximum(H, 0.0)
            s = Hp.sum()
            if s <= 0:
                return H, U
            ys = np.arange(kh)[:, None]
            xs = np.arange(kw)[None, :]
            iy = (Hp * ys).sum() / s
            ix = (Hp * xs).sum() / s
            sy = int(round((kh - 1) / 2.0 - iy))
            sx = int(round((kw - 1) / 2.0 - ix))

        else:  # 'centroid' на основе рамки
            Hp = np.maximum(H, 0.0)
            m = Hp.max()
            if m <= 0:
                return H, U
            
            # Порог отсечения шумового фона ядра
            tao = 0.03
            thr = min(m * tao, 0.002)
            mask = Hp >= thr
            if not mask.any():
                return H, U
            
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y_top, y_bot = int(rows[0]), int(rows[-1])
            x_left, x_right = int(cols[0]), int(cols[-1])

            gap_left = x_left
            gap_right = (kw - 1) - x_right
            gap_top = y_top
            gap_bot = (kh - 1) - y_bot

            # Поправка в сторону более "тяжелого" края для симметрии
            s_l = Hp[:, x_left].sum()
            s_r = Hp[:, x_right].sum()
            bonus_x = 0.01 if (s_l >= s_r) else -0.01
            s_t = Hp[y_top, :].sum()
            s_b = Hp[y_bot, :].sum()
            bonus_y = 0.01 if (s_t >= s_b) else -0.01

            sx = int(round((gap_right - gap_left + bonus_x) / 2.0))
            sy = int(round((gap_bot - gap_top + bonus_y) / 2.0))

        if sy == 0 and sx == 0:
            return H, U

        # Выполнение сдвига ядра с дополнением нулями
        H_new = np.zeros_like(H)
        src_r0 = max(0, -sy); src_r1 = min(kh, kh - sy)
        src_c0 = max(0, -sx); src_c1 = min(kw, kw - sx)
        dst_r0 = max(0, sy);  dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, sx);  dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            H_new[dst_r0:dst_r1, dst_c0:dst_c1] = H[src_r0:src_r1, src_c0:src_c1]
        s_h = H_new.sum()
        if s_h > 0:
            H_new = H_new / s_h

        # Встречный сдвиг изображения (используется edge-padding для предотвращения артефактов)
        Mh, Mw = U.shape
        py0 = max(0, sy);  py1 = max(0, -sy)
        px0 = max(0, sx);  px1 = max(0, -sx)
        U_padded = np.pad(U, ((py0, py1), (px0, px1)), mode='edge')
        U_new = U_padded[py1:py1 + Mh, px1:px1 + Mw].copy()

        return H_new, U_new

    # --- Основной процесс ---
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # 1. Нормализация к float64 [0, 1]
        y = np.asarray(image, dtype=np.float64)
        if y.max() > 1.0:
            y = y / 255.0

        # 1a. Предварительная фильтрация импульсного шума
        # Применяется до оценки дисперсии, так как импульсные выбросы искажают статистику.
        self.impulse_info = None
        if self.impulse_preprocess == 'auto':
            from blinddeconv.algorithms.mod_cython._build_pyd.impulse_noise_estimation import (
                detect_impulse_noise, adaptive_median_filter,
            )
            ip = dict(self.impulse_params)
            density_threshold = float(ip.pop('density_threshold', 0.005))
            max_window = int(ip.pop('max_window', 7))
            
            def _impulse_one(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
                info = detect_impulse_noise(arr, **ip)
                if info['has_impulse'] and info['density'] >= density_threshold:
                    cleaned = adaptive_median_filter(
                        arr, info['impulse_mask'], max_window=max_window,
                    )
                    return cleaned, info
                return arr, info

            if y.ndim == 3:
                cleaned = np.empty_like(y)
                infos: List[Dict[str, Any]] = []
                for c in range(y.shape[2]):
                    ch_out, ch_info = _impulse_one(y[..., c])
                    cleaned[..., c] = ch_out
                    infos.append(ch_info)
                self.impulse_info = {
                    'per_channel': [
                        {k: v for k, v in d.items() if k != 'impulse_mask'}
                        for d in infos
                    ],
                    'mean_density': float(np.mean([d['density'] for d in infos])),
                }
                y = cleaned
            else:
                y_clean, info = _impulse_one(y)
                self.impulse_info = {
                    k: v for k, v in info.items() if k != 'impulse_mask'
                }
                y = y_clean
            if self.verbose:
                print(f'[HTP_BD] Предварительная очистка импульсного шума \u2192 '
                      f'{self.impulse_info}')

        # 1b. Оценка дисперсии шума
        self.noise_sigma = None
        self.noise_info = None
        if self.noise_estimation != 'none':
            y_lum = y.mean(axis=2) if y.ndim == 3 else y
            if self.noise_estimation == 'chen':
                from blinddeconv.algorithms.mod_cython._build_pyd.chen_noise_estimate import estimate_noise_level
                pch_size = int(self.noise_estimation_params.get('pch_size', 8))
                sigma = float(estimate_noise_level(y_lum, pch_size=pch_size))
                self.noise_sigma = sigma
                self.noise_info = {'method': 'chen', 'sigma': sigma}
            elif self.noise_estimation == 'pyatykh':
                from blinddeconv.algorithms.mod_cython._build_pyd.pyatykh_noise_reconstruction import estimate_noise_params
                blocksize = int(self.noise_estimation_params.get('blocksize', 7))
                result = estimate_noise_params(y_lum, blocksize=blocksize)
                self.noise_sigma = float(result.get('sigma_norm',
                                                    result.get('sigma', 0.0) / 255.0))
                self.noise_info = {
                    'method': 'pyatykh',
                    'a': float(result.get('a', 0.0)),
                    'b': float(result.get('b', 0.0)),
                    'sigma': float(result.get('sigma', 0.0)),
                    'sigma_norm': self.noise_sigma,
                    'noise_type': result.get('noise_type', 'unknown'),
                }
            if self.verbose:
                print(f'[HTP_BD] Оценка шума ({self.noise_estimation}) \u2192 {self.noise_info}')

        # 1c. Автоконфигурация методов шумоподавления
        self.auto_mode_applied = None
        if self.auto_mode == 'auto' and self.noise_info is not None:
            cfg = self._auto_denoiser_config(self.noise_info)
            cfg.update(self.auto_mode_overrides or {})
            sigma = float(self.noise_sigma or 0.0)
            applied: Dict[str, Any] = {}
            for hook in ('pre_pyramid', 'pre_kernel', 'pre_nonblind'):
                user_method = getattr(self, hook)
                if user_method in (None, 'none') and cfg.get(hook) is not None:
                    setattr(self, hook, cfg[hook])
                    applied[hook] = cfg[hook]
                
                method_now = getattr(self, hook)
                if method_now in (None, 'none'):
                    continue
                pkey = hook + '_params'
                user_params = getattr(self, pkey)
                if not user_params:                      
                    if cfg.get(hook) == method_now and cfg.get(pkey):
                        params = dict(cfg[pkey])
                    else:
                        params = self._default_params_for(
                            method_now, sigma, self.noise_info, hook=hook,
                        )
                    setattr(self, pkey, params)
                    applied[pkey] = params
            self.auto_mode_applied = applied
            if self.verbose and applied:
                print(f'[HTP_BD] Применена автоконфигурация шумоподавления: {applied}')

            # Адаптивный выбор метода финальной неслепой деконволюции
            user_locked_nb = ('nonblind_method' in (self.auto_mode_overrides or {}))
            if (self.nonblind_method == 'fft_cg_sr_al' and not user_locked_nb):
                if sigma < 0.01:
                    self.nonblind_method = 'firls'
                else:
                    self.nonblind_method = 'ringing_removal'

        # 2. Выполнение алгоритма многомасштабной оценки ядра
        PAR = self._build_par()
        hsize = (self.kernel_size, self.kernel_size)

        U, H, _report = mc_restoration(
            y,
            hsize=hsize,
            PAR=PAR,
            MSlevels=self.MSlevels,
            maxROIsize=self.maxROIsize,
        )
        U = np.clip(U, 0.0, 1.0)

        # 3. Финальное центрирование ядра (опционально)
        if self.auto_recenter:
            H, U = self._recenter_kernel_and_image(H, U)

        # 4. Формирование результатов
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'Lp': self.Lp,
            'gamma': self.gamma,
            'alpha_u': self.alpha_u,
            'beta_u': self.beta_u,
            'alpha_h': self.alpha_h,
            'beta_h': self.beta_h,
            'gamma_nonblind': self.gamma_nonblind,
            'beta_u_nonblind': self.beta_u_nonblind,
            'Lp_nonblind': self.Lp_nonblind,
            'MSlevels': self.MSlevels,
            'maxROIsize': self.maxROIsize,
            'maxiter': self.maxiter,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'time': time.time() - start_time,
        }

        x_final = U * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        if self.kernel_flip == 'lr':
            H_out = H[:, ::-1].copy()
        elif self.kernel_flip == 'ud':
            H_out = H[::-1, :].copy()
        elif self.kernel_flip == 'rot180':
            H_out = H[::-1, ::-1].copy()
        else:
            H_out = H
        return x_final, H_out

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('Lp', self.Lp),
            ('gamma', self.gamma),
            ('alpha_u', self.alpha_u),
            ('beta_u', self.beta_u),
            ('alpha_h', self.alpha_h),
            ('beta_h', self.beta_h),
            ('centering_threshold', self.centering_threshold),
            ('gamma_nonblind', self.gamma_nonblind),
            ('beta_u_nonblind', self.beta_u_nonblind),
            ('Lp_nonblind', self.Lp_nonblind),
            ('maxiter', self.maxiter),
            ('maxiter_u', self.maxiter_u),
            ('maxiter_h', self.maxiter_h),
            ('ccreltol', self.ccreltol),
            ('MSlevels', self.MSlevels),
            ('maxROIsize', self.maxROIsize),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'maxROIsize':
                    self.maxROIsize = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams