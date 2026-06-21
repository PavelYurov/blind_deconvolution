"""
dcp.py

Слепая деконволюция изображений с использованием априорного распределения 
темного канала (Dark Channel Prior, DCP).
Модифицированная версия с опциональным конвейером шумоподавления и 
интеграцией методов неслепого восстановления.

Основано на методе:
    J. Pan, D. Sun, H. Pfister, M.-H. Yang: "Blind Image Deblurring
    Using Dark Channel Prior", CVPR, 2016.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

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

from .solvers import blind_deconv, ringing_artifacts_removal


class DCP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе априорного распределения темного канала 
    с расширенным конвейером подавления шума.

    Базовые параметры DCP
    ---------------------
    kernel_size : int, по умолчанию 25
        Пространственный размер неизвестной функции рассеяния точки (квадратное, нечетное).
    lambda_dark : float, по умолчанию 4e-3
        Вес для L0-регуляризации интенсивности темного канала.
    lambda_grad : float, по умолчанию 4e-3
        Вес для L0-регуляризации градиентов изображения.
    xk_iter : int, по умолчанию 5
        Количество итераций слепой оценки на каждом уровне пирамиды.
    gamma_correct : float, по умолчанию 1.0
        Экспонента гамма-коррекции, применяемая перед оценкой ядра.
    k_thresh : float, по умолчанию 20.0
        Относительный порог обнуления элементов ядра (max(k) / k_thresh).
    lambda_tv : float, по умолчанию 0.003
        Вес TV-регуляризации для этапа неслепой деконволюции.
    lambda_l0 : float, по умолчанию 5e-4
        Вес L0-регуляризации для этапа неслепой деконволюции.
    weight_ring : float, по умолчанию 1.0
        Коэффициент подавления артефактов звона.

    Параметры конвейера шумоподавления
    ----------------------------------
    impulse_preprocess : str, по умолчанию 'none'
        Метод обработки импульсного шума ('auto' для автоматического обнаружения).
    impulse_params : dict или None
        Параметры обнаружения импульсного шума (density_threshold, outlier_threshold, и т.д.).
    noise_estimation : str, по умолчанию 'none'
        Метод оценки дисперсии шума: 'chen' (на базе PCA), 'pca' (с VST) или 'none'.
    auto_params : dict или None
        Правила автоматической настройки lambda_dark, lambda_grad и параметров 
        неслепого восстановления на основе оцененного уровня шума sigma.
    screenot_preprocess : str, по умолчанию 'none'
        Предварительное шумоподавление на основе метода ScreeNOT ('auto', 'none').
    screenot_params : dict или None
        Параметры метода ScreeNOT.
    act_preprocess : str, по умолчанию 'none'
        Предварительное шумоподавление методом Adaptive Curvelet Thresholding.
    act_params : dict или None
        Параметры метода ACT.
    preprocess : str, по умолчанию 'none'
        Пространственный алгоритм шумоподавления, применяемый до начала 
        слепого цикла оценки ядра ('tv', 'nlm', 'bilateral', 'bm3d' и др.).
    preprocess_params : dict или None
        Специфичные параметры для предварительного пространственного фильтра.
    noise_preprocess : str, по умолчанию 'none'
        Спектральный фильтр для периодического шума ('auto', 'notch', 'bandstop').
    noise_preprocess_params : dict или None
        Параметры спектральной фильтрации.
    histogram_eq : str, по умолчанию 'none'
        Выравнивание гистограммы всего изображения перед началом работы 
        ('clahe', 'global', 'none'). Внимание: может нарушить свойства темного канала.
    histogram_eq_params : dict или None
        Параметры эквализации гистограммы.
    pre_nonblind : str, по умолчанию 'none'
        Алгоритм шумоподавления, применяемый перед финальным неслепым шагом.
    pre_nonblind_params : dict или None
        Параметры для данного алгоритма.

    Параметры внутренних обработчиков (blind_hooks)
    -----------------------------------------------
    blind_hooks : dict или None
        Определяет методы обработки внутри цикла слепой оценки ядра. Возможные ключи:
        - 'latent_denoise': фильтрация скрытого изображения S перед вычислением градиентов.
        - 'latent_denoise_decay': фактор затухания силы фильтрации с каждой итерацией.
        - 'grad_eq': безопасная эквализация гистограммы только для скрытого 
          изображения перед вычислением градиентов.
        - 'kernel_smooth': сглаживание ядра размытия после каждой итерации.

    Параметры финального восстановления
    -----------------------------------
    final_deconv : str, по умолчанию 'ringing_removal'
        Метод финальной неслепой деконволюции ('ringing_removal', 'adaptive_lp', 
        'wiener', 'tikhonov').
    nb_params : dict или None
        Специфические параметры для выбранного метода финального восстановления.
    auto_mode : str, по умолчанию 'off'
        Глобальный оркестратор шумоподавления ('robust' или 'off'). Автоматически 
        переопределяет параметры конвейера на основе оценки шума.
    auto_mode_params : dict или None
        Параметры для настройки порогов срабатывания оркестратора.
    verbose : bool, по умолчанию False
        Флаг вывода диагностической информации в консоль.
    """
    def __init__(
        self,
        kernel_size: int = 25,
        lambda_dark: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.003,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        auto_params: dict = None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        preprocess: str = 'none',
        preprocess_params: dict = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        histogram_eq: str = 'none',
        histogram_eq_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        blind_hooks: dict = None,
        final_deconv: str = 'ringing_removal',
        nb_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
        verbose: bool = False,
    ):
        super().__init__(name='DCP-BD')

        self.kernel_size = kernel_size
        self.lambda_dark = lambda_dark
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring

        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.histogram_eq = histogram_eq
        self.histogram_eq_params = histogram_eq_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params

        self.blind_hooks = blind_hooks

        self.final_deconv = final_deconv.lower()
        self.nb_params = nb_params

        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        self.verbose = verbose

        self._defaults_snapshot = {
            'preprocess':          preprocess,
            'preprocess_params':   preprocess_params,
            'pre_nonblind':        pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'act_preprocess':      act_preprocess,
            'act_params':          act_params,
            'final_deconv':        self.final_deconv,
            'nb_params':           nb_params,
            'blind_hooks':         blind_hooks,
            'ringing_weights': {
                'lambda_tv':   lambda_tv,
                'lambda_l0':   lambda_l0,
                'weight_ring': weight_ring,
            },
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск алгоритма слепой деконволюции с полным конвейером обработки.

        Процесс включает следующие этапы:
        1. Нормализация входного изображения.
        2. Конвертация в полутоновый формат для оценки ядра.
        3. Последовательное выполнение этапов шумоподавления (импульсный шум, 
           оценка параметров, ScreeNOT/ACT, спектральная фильтрация).
        4. Многомасштабная слепая деконволюция с опциональными внутренними 
           обработчиками (хуками).
        5. Неслепое восстановление исходного изображения с применением 
           оцененного ядра.

        Возвращает
        ----------
        x_final : ndarray
            Восстановленное изображение в формате int16 [0, 255].
        kernel : ndarray
            Оцененное ядро размытия.
        """
        start_time = time.time()

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 3 and y.shape[2] == 1:
            yg = y[:, :, 0]
        else:
            yg = y.copy() if y.ndim == 2 else y[:, :, 0]

        impulse_info = None
        if self.impulse_preprocess == 'auto':
            from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import ( 
                detect_impulse_noise, adaptive_median_filter,
            )
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                yg,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
                outlier_window=ip.get('outlier_window', 5),
            )
            if impulse_info['has_impulse']:
                if self.verbose:
                    print(f"[DCP-BD] Impulse noise detected "
                          f"(density={impulse_info['density']:.4f}), "
                          f"applying adaptive median filter")
                yg = adaptive_median_filter(
                    yg, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))

        if self.auto_mode == 'robust' and self.noise_estimation == 'none':
            self.noise_estimation = 'pca'
            if self.verbose:
                print("[DCP-BD] auto_mode='robust' \u2192 "
                      "forcing noise_estimation='pca'")
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
            if self.verbose and noise_info is not None:
                sigma = noise_info.get('sigma_norm', 0)
                print(f"[DCP-BD] Noise estimation ({self.noise_estimation}): "
                      f"σ={sigma:.5f} (σ_255={sigma * 255:.2f})")

        if self.auto_params is not None and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_ld = ap.get('k_lambda_dark', 2.0)
                k_lg = ap.get('k_lambda_grad', 0.2)
                k_tv = ap.get('k_lambda_tv', 0.05)
                k_l0 = ap.get('k_lambda_l0', 0.025)
                self.lambda_dark = max(1e-4, k_ld * sigma_n)
                self.lambda_grad = max(1e-4, k_lg * sigma_n)
                self.lambda_tv = max(1e-4, k_tv * sigma_n)
                self.lambda_l0 = max(1e-5, k_l0 * sigma_n)
                if self.verbose:
                    print(f"[DCP-BD] auto_params(σ={sigma_n:.5f}): "
                          f"λ_dark={self.lambda_dark:.5f}, "
                          f"λ_grad={self.lambda_grad:.5f}, "
                          f"λ_tv={self.lambda_tv:.5f}, "
                          f"λ_l0={self.lambda_l0:.6f}")

        orchestrator_info = self._orchestrate_robust(noise_info)

        screenot_info = None
        if self.screenot_preprocess == 'auto':
            if self.act_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from blinddeconv.algorithms.mod_denoise.screenot import screenot_denoise
            sp = self.screenot_params or {}
            yg, screenot_info = screenot_denoise(
                yg,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if self.verbose:
                print(f"[DCP-BD] ScreeNOT applied "
                      f"(rank={screenot_info.get('rank', '?')})")

        act_info = None
        if self.act_preprocess == 'auto':
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            yg, act_info = act_denoise(
                yg,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if self.verbose:
                print("[DCP-BD] ACT curvelet denoising applied")

        if self.preprocess not in (None, 'none'):
            yg = self._apply_denoise(
                yg, self.preprocess, self.preprocess_params, noise_info)
            if self.verbose:
                print(f"[DCP-BD] Pre-blind denoise: {self.preprocess}")

        psd_info = None
        if self.noise_preprocess != 'none':
            yg, psd_info = self._apply_noise_preprocess(yg)
            if self.verbose:
                print(f"[DCP-BD] PSD noise preprocess: {self.noise_preprocess}")

        yg_for_restore = yg
        if self.histogram_eq not in (None, 'none'):
            yg = self._apply_histogram_eq(yg)
            if self.verbose:
                print(f"[DCP-BD] Histogram equalization: {self.histogram_eq}")

        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
        }

        latent_hook, kernel_hook = self._build_blind_hooks(noise_info)

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_dark, self.lambda_grad, opts,
            latent_hook=latent_hook, kernel_hook=kernel_hook,
            iteration_callback=self._callback,
        )

        y_nb = yg_for_restore
        if self.pre_nonblind not in (None, 'none'):
            y_nb = self._apply_denoise(
                y_nb, self.pre_nonblind, self.pre_nonblind_params, noise_info)
            if self.verbose:
                print(f"[DCP-BD] Pre-nonblind denoise: {self.pre_nonblind}")

        if self.final_deconv == 'ringing_removal':
            Latent = ringing_artifacts_removal(
                y_nb, kernel,
                self.lambda_tv, self.lambda_l0, self.weight_ring,
            )
        elif self.final_deconv == 'adaptive_lp':
            from blinddeconv.algorithms.mod_denoise.non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            Latent = adaptive_lp_deconv(
                y_nb, kernel,
                alpha=nbp.get('alpha', 0.8),
                sigma_n=sigma_n,
                two_stage=nbp.get('two_stage', True),
            )
        elif self.final_deconv == 'wiener':
            Latent = self._wiener_filter(y_nb, kernel)
        elif self.final_deconv == 'tikhonov':
            Latent = self._tikhonov_filter(y_nb, kernel)
        else:
            raise ValueError(
                f"Unknown final_deconv '{self.final_deconv}'. "
                "Choose 'ringing_removal', 'adaptive_lp', "
                "'wiener', or 'tikhonov'.")

        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_dark': self.lambda_dark,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'final_deconv': self.final_deconv,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': {k_: v for k_, v in (impulse_info or {}).items()
                            if k_ != 'impulse_mask'} if impulse_info else None,
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'preprocess': self.preprocess,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': {k_: v for k_, v in (psd_info or {}).items()
                         if k_ != 'psd_2d'} if psd_info else None,
            'histogram_eq': self.histogram_eq,
            'pre_nonblind': self.pre_nonblind,
            'blind_hooks': self.blind_hooks,
            'auto_params': self.auto_params,
            'nb_params': self.nb_params,
            'auto_mode': self.auto_mode,
            'auto_mode_params': self.auto_mode_params,
            'orchestrator_info': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel


    @staticmethod
    def _guided_filter(I, p, r, eps):
        """Реализация алгоритма фильтрации с направляющим изображением (Guided Filter)."""
        from scipy.ndimage import uniform_filter
        size = 2 * r + 1
        def box(x):
            return uniform_filter(x, size=size, mode='reflect')
        mean_I = box(I)
        mean_p = box(p)
        corr_Ip = box(I * p)
        var_I = box(I * I) - mean_I ** 2
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return box(a) * I + box(b)

    def _apply_denoise(self, img, method, params, noise_info):
        """
        Применение пространственного фильтра к одноканальному изображению.
        Поддерживаемые алгоритмы: 'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'act'.
        """
        if method is None or method == 'none':
            return img
        p = dict(params or {})
        sigma = noise_info.get('sigma_norm', None) if noise_info else None

        if method == 'tv':
            from skimage.restoration import denoise_tv_chambolle
            w = p.get('weight', max(0.01, sigma * 2) if sigma else 0.1)
            return denoise_tv_chambolle(img, weight=w)

        elif method == 'nlm':
            from skimage.restoration import (
                denoise_nl_means, estimate_sigma as _est_sig)
            sig = p.get('sigma', sigma)
            if sig is None:
                sig = float(np.mean(_est_sig(img)))
            h = p.get('h', 0.8 * sig)
            return denoise_nl_means(
                img, h=h,
                patch_size=p.get('patch_size', 5),
                patch_distance=p.get('patch_distance', 6),
                fast_mode=True)

        elif method == 'bilateral':
            import cv2
            d = p.get('d', 5)
            sc = p.get('sigma_color', sigma if sigma else 0.1)
            ss = p.get('sigma_space', 5.0)
            return cv2.bilateralFilter(
                img.astype(np.float32), d, float(sc), float(ss)
            ).astype(np.float64)

        elif method == 'guided':
            r = p.get('radius', 4)
            eps = p.get('eps',
                        sigma ** 2 * 4 if sigma else 0.01)
            return self._guided_filter(img, img, r, eps)

        elif method == 'bm3d':
            import bm3d as bm3d_lib
            sig = p.get('sigma', sigma if sigma else 0.05)
            return bm3d_lib.bm3d(img, sigma_psd=sig)

        elif method == 'act':
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            nv = p.get('noise_var', None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = p.get('threshold_setting', 's')
            result, _ = act_denoise(img, noise_var=nv,
                                    threshold_setting=ts)
            return result

        else:
            raise ValueError(
                f"Unknown denoiser='{method}'. Choose from: "
                f"'tv', 'nlm', 'bilateral', 'guided', 'bm3d', "
                f"'act', 'none'")

    def _estimate_noise(self, yg):
        """Оценка параметров шума методами Chen или PCA."""
        if self.noise_estimation == 'chen':
            from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pca':
            from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pca'
            return result
        return None

    def _apply_noise_preprocess(self, yg):
        """Анализ и фильтрация периодического шума в спектральной области."""
        from blinddeconv.algorithms.mod_denoise.noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess,
            notch_filter, bandstop_filter,
        )
        p = self.noise_preprocess_params or {}
        mode = self.noise_preprocess

        if mode == 'auto':
            result = noise_preprocess(
                yg,
                pch_size=p.get('pch_size', 32),
                n_smooth=p.get('n_smooth', 100),
                peak_threshold=p.get('peak_threshold', 100.0),
                notch_radius=p.get('notch_radius', 3),
            )
            return result['image'], result['psd_info']

        psd_info = analyze_noise_psd(
            yg,
            pch_size=p.get('pch_size', 32),
            n_smooth=p.get('n_smooth', 100),
            peak_threshold=p.get('peak_threshold', 100.0),
        )

        if mode == 'notch':
            peaks = psd_info['periodic_peaks']
            if peaks:
                yg_out = notch_filter(yg, peaks,
                                      notch_radius=p.get('notch_radius', 3))
            else:
                yg_out = yg
        elif mode == 'bandstop':
            yg_out = bandstop_filter(
                yg,
                freq_low=p.get('freq_low', 0.3),
                freq_high=p.get('freq_high', 0.5),
                order=p.get('order', 2),
            )
        else:
            raise ValueError(
                f"Unknown noise_preprocess='{mode}'. "
                f"Choose from: 'auto', 'notch', 'bandstop', 'none'")

        return yg_out, psd_info

    def _orchestrate_robust(self, noise_info):
        """
        Автоматическая настройка параметров конвейера на основе оценки шума.

        Алгоритм работы:
        - Режим малого шума (sigma <= sigma_clean): используются оригинальные параметры 
          без изменений.
        - Режим сильного шума (sigma > sigma_clean): плавное перераспределение 
          весов, активация направляющих фильтров, двусторонней фильтрации 
          или алгоритма BM3D/ACT в зависимости от типа (пуассоновский или гауссовский).
        """
        info = {'triggered': False, 'mode': self.auto_mode}

        if self.auto_mode != 'robust':
            return info

        snap = self._defaults_snapshot
        p = self.auto_mode_params or {}
        sigma_clean = float(p.get('sigma_clean', 0.005))
        sigma_heavy = float(p.get('sigma_heavy', 0.05))
        force_heavy_sigma = float(p.get('force_heavy_sigma', 0.01))
        prefer_act = bool(p.get('prefer_act_for_gaussian', False))

        sigma = None
        ntype = 'unknown'
        if noise_info is not None:
            sigma = noise_info.get('sigma_norm', None)
            ntype = noise_info.get('noise_type', 'unknown') or 'unknown'
        if sigma is None:
            sigma = 0.0

        poisson_like = ntype in ('poisson', 'poisson_gaussian', 'unknown')
        heavy = (sigma > sigma_clean) or (poisson_like and sigma > force_heavy_sigma)

        info.update({
            'triggered': True,
            'sigma': float(sigma),
            'noise_type': ntype,
            'poisson_like': poisson_like,
            'sigma_clean': sigma_clean,
            'sigma_heavy': sigma_heavy,
            'force_heavy_sigma': force_heavy_sigma,
            'prefer_act_for_gaussian': prefer_act,
        })

        if not heavy:
            info['branch'] = 'clean'
            self.preprocess          = snap['preprocess']
            self.preprocess_params   = snap['preprocess_params']
            self.pre_nonblind        = snap['pre_nonblind']
            self.pre_nonblind_params = snap['pre_nonblind_params']
            self.act_preprocess      = snap['act_preprocess']
            self.act_params          = snap['act_params']
            self.final_deconv        = snap['final_deconv']
            self.nb_params           = snap['nb_params']
            self.blind_hooks         = snap['blind_hooks']
            rw = snap['ringing_weights']
            self.lambda_tv   = rw['lambda_tv']
            self.lambda_l0   = rw['lambda_l0']
            self.weight_ring = rw['weight_ring']
            if self.verbose:
                print(f"[DCP-BD][orchestrator] clean (\u03c3={sigma:.5f}) "
                      f"\u2192 paper defaults restored")
            return info

        info['branch'] = 'heavy'
        sigma_eff = max(sigma, 1e-3)

        self.final_deconv = snap['final_deconv']
        self.nb_params    = snap['nb_params']

        if poisson_like or prefer_act:
            info['route'] = 'act'
            self.preprocess        = 'none'
            self.preprocess_params = None
            self.act_preprocess    = 'auto'
            self.act_params        = {'noise_var': sigma_eff ** 2}
            self.pre_nonblind      = 'act'
            self.pre_nonblind_params = {'noise_var': sigma_eff ** 2}
            self.blind_hooks = {
                'latent_denoise': 'bilateral',
                'latent_denoise_params': {
                    'sigma_color': sigma_eff * 1.5,
                    'sigma_space': 2.0,
                },
                'latent_denoise_decay': 0.8,
                'grad_eq': 'none',
                'kernel_smooth': 'gaussian',
                'kernel_smooth_params': {'sigma': 0.3},
            }
        else:
            w = (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            info['blend_weight'] = w

            if w < 0.6:
                info['route'] = 'gauss_light'
                self.preprocess = 'bilateral'
                self.preprocess_params = {
                    'sigma_color': sigma_eff * 2.0,
                    'sigma_space': 5.0,
                }
                self.act_preprocess    = 'none'
                self.act_params        = None
                self.pre_nonblind      = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma_eff}
                self.blind_hooks = {
                    'latent_denoise': 'guided',
                    'latent_denoise_params': {
                        'radius': 2,
                        'eps': max(1e-3, sigma_eff ** 2 * 2),
                    },
                    'latent_denoise_decay': 0.7,
                    'grad_eq': 'none',
                    'kernel_smooth': 'none',
                }
            else:
                info['route'] = 'gauss_strong'
                self.preprocess = 'bm3d'
                self.preprocess_params = {'sigma': sigma_eff}
                self.act_preprocess    = 'none'
                self.act_params        = None
                self.pre_nonblind      = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma_eff * 1.5}
                self.blind_hooks = {
                    'latent_denoise': 'guided',
                    'latent_denoise_params': {
                        'radius': 4,
                        'eps': max(5e-3, sigma_eff ** 2 * 4),
                    },
                    'latent_denoise_decay': 0.8,
                    'grad_eq': 'none',
                    'kernel_smooth': 'gaussian',
                    'kernel_smooth_params': {'sigma': 0.4},
                }

        if self.auto_params is None and self.final_deconv == 'ringing_removal':
            w = (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-6)
            w = float(np.clip(w, 0.0, 1.0))
            rw = snap['ringing_weights']
            noisy_tv   = sigma_eff * 0.5
            noisy_l0   = sigma_eff * 0.025
            noisy_ring = float(np.clip(sigma_eff * 50.0, 0.3, 1.0))
            self.lambda_tv   = (1 - w) * rw['lambda_tv']   + w * noisy_tv
            self.lambda_l0   = (1 - w) * rw['lambda_l0']   + w * noisy_l0
            self.weight_ring = (1 - w) * rw['weight_ring'] + w * noisy_ring
            info['nb_blend_applied'] = True
            info['nb_blend_weight'] = w
            info['nb_weights'] = {
                'lambda_tv':   self.lambda_tv,
                'lambda_l0':   self.lambda_l0,
                'weight_ring': self.weight_ring,
            }
        else:
            info['nb_blend_applied'] = False
            info['nb_blend_skipped_reason'] = (
                'user_auto_params_active' if self.auto_params is not None
                else f"final_deconv={self.final_deconv!r}"
            )

        if self.verbose:
            print(f"[DCP-BD][orchestrator] heavy/{info['route']} "
                  f"\u03c3={sigma:.5f} type={ntype} "
                  f"preprocess={self.preprocess} "
                  f"act={self.act_preprocess} "
                  f"pre_nb={self.pre_nonblind} "
                  f"nb_blend={info['nb_blend_applied']}")

        return info

    def _build_blind_hooks(self, noise_info):
        """
        Формирование функций обратного вызова для внедрения фильтрации внутрь 
        цикла оценки ядра (подавление шума в скрытом изображении или ядре 
        на каждой итерации).
        """
        bh = self.blind_hooks
        if bh is None:
            return None, None

        sigma = noise_info.get('sigma_norm', None) if noise_info else None

        latent_method = bh.get('latent_denoise', 'none')
        latent_params = bh.get('latent_denoise_params', None)
        latent_decay = bh.get('latent_denoise_decay', 1.0)
        grad_eq = bh.get('grad_eq', 'none')
        grad_eq_params = bh.get('grad_eq_params', None)

        latent_hook = None
        if latent_method != 'none' or grad_eq != 'none':
            def _latent_hook(S, k, iter_idx, scale_idx):
                result = S
                if latent_method != 'none':
                    p = dict(latent_params or {})
                    if latent_decay < 1.0 and iter_idx > 0:
                        factor = latent_decay ** iter_idx
                        for key in ('weight', 'h', 'sigma_color',
                                    'eps', 'sigma', 'sigma_psd'):
                            if key in p:
                                p[key] = p[key] * factor
                    result = self._apply_denoise(
                        result, latent_method, p, noise_info)
                if grad_eq == 'clahe':
                    from skimage.exposure import equalize_adapthist
                    gp = grad_eq_params or {}
                    result = equalize_adapthist(
                        np.clip(result, 0, 1),
                        clip_limit=gp.get('clip_limit', 0.01),
                        nbins=gp.get('nbins', 256),
                    )
                elif grad_eq == 'global':
                    from skimage.exposure import equalize_hist
                    result = equalize_hist(np.clip(result, 0, 1))
                return result
            latent_hook = _latent_hook

        kernel_method = bh.get('kernel_smooth', 'none')
        kernel_params = bh.get('kernel_smooth_params', {})

        kernel_hook = None
        if kernel_method != 'none':
            def _kernel_hook(k, S, iter_idx, scale_idx):
                if kernel_method == 'gaussian':
                    from scipy.ndimage import gaussian_filter
                    sigma_k = kernel_params.get('sigma', 0.5)
                    k = gaussian_filter(k, sigma=sigma_k)
                return k
            kernel_hook = _kernel_hook

        if self.verbose and (latent_hook or kernel_hook):
            parts = []
            if latent_method != 'none':
                parts.append(f"latent_denoise={latent_method}")
            if grad_eq != 'none':
                parts.append(f"grad_eq={grad_eq}")
            if kernel_method != 'none':
                parts.append(f"kernel_smooth={kernel_method}")
            print(f"[DCP-BD] blind_hooks: {', '.join(parts)}")

        return latent_hook, kernel_hook


    def _apply_histogram_eq(self, img):
        """Глобальное или адаптивное выравнивание гистограммы."""
        from skimage.exposure import equalize_adapthist, equalize_hist
        method = self.histogram_eq
        p = self.histogram_eq_params or {}

        if method == 'clahe':
            return equalize_adapthist(
                img,
                clip_limit=p.get('clip_limit', 0.01),
                nbins=p.get('nbins', 256),
                kernel_size=p.get('kernel_size', None),
            )
        elif method == 'global':
            return equalize_hist(img)
        else:
            raise ValueError(
                f"Unknown histogram_eq='{method}'. "
                f"Choose from: 'clahe', 'global', 'none'")

    @staticmethod
    def _psf2otf(psf, shape):
        padded = np.zeros(shape, dtype=np.float64)
        ph, pw = psf.shape
        padded[:ph, :pw] = psf
        padded = np.roll(padded, -(ph // 2), axis=0)
        padded = np.roll(padded, -(pw // 2), axis=1)
        return np.fft.fft2(padded)

    def _wiener_filter(self, img, kernel):
        """Неслепая деконволюция на основе фильтра Винера."""
        nbp = self.nb_params or {}
        noise_snr = nbp.get('noise_snr', 0.01)
        H_otf = self._psf2otf(kernel, img.shape[:2])
        H_conj = np.conj(H_otf)
        G = np.fft.fft2(img)
        denom = np.abs(H_otf) ** 2 + noise_snr
        result = np.real(np.fft.ifft2(H_conj * G / denom))
        return result

    def _tikhonov_filter(self, img, kernel):
        """Неслепая деконволюция с регуляризацией Тихонова."""
        nbp = self.nb_params or {}
        alpha = nbp.get('alpha', 0.01)
        H_otf = self._psf2otf(kernel, img.shape[:2])
        H_conj = np.conj(H_otf)
        G = np.fft.fft2(img)
        dx = np.array([[1, -1]], dtype=np.float64)
        dy = np.array([[1], [-1]], dtype=np.float64)
        Dx = self._psf2otf(dx, img.shape[:2])
        Dy = self._psf2otf(dy, img.shape[:2])
        reg = np.abs(Dx) ** 2 + np.abs(Dy) ** 2
        denom = np.abs(H_otf) ** 2 + alpha * reg
        result = np.real(np.fft.ifft2(H_conj * G / denom))
        return result

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_dark', self.lambda_dark),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('impulse_preprocess', self.impulse_preprocess),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
            ('screenot_preprocess', self.screenot_preprocess),
            ('act_preprocess', self.act_preprocess),
            ('preprocess', self.preprocess),
            ('noise_preprocess', self.noise_preprocess),
            ('histogram_eq', self.histogram_eq),
            ('pre_nonblind', self.pre_nonblind),
            ('blind_hooks', self.blind_hooks),
            ('final_deconv', self.final_deconv),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
