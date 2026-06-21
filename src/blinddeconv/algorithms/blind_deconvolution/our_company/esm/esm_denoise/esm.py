"""
esm.py

Слепая деконволюция изображений с использованием улучшенной разреженной 
модели (Enhanced Sparse Model, ESM) с расширенным конвейером шумоподавления.

Основано на методе:
    L. Chen, F. Fang, S. Lei, F. Li, G. Zhang: "Enhanced Sparse Model
    for Blind Deblurring", ECCV, 2020.

Конвейер обработки включает модификации для работы с зашумленными данными:
1. Нормализация входного изображения к диапазону float64 [0, 1].
2. Преобразование в полутоновый формат для оценки ядра размытия.
3. Опциональный конвейер обработки шума (обнаружение импульсного шума, 
   оценка дисперсии, пространственная фильтрация и фильтрация спектра мощности).
4. Многомасштабная слепая оценка ядра размытия методом ESM с возможностью 
   промежуточного шумоподавления.
5. Неслепое восстановление полноцветного изображения с подавлением 
   артефактов звона или с использованием адаптивного Lp-регуляризатора.
6. Возврат восстановленного изображения (в формате int16 [0, 255]) и 
   оцененного ядра.

Базовые параметры ESM (веса регуляризации, параметр theta, размер ядра) 
остаются неизменными в любых режимах, в то время как робастный оркестратор 
автоматически адаптирует выбор фильтров и параметры неслепой деконволюции 
на основе оцененного уровня шума. Специфические для ESM экспертные 
адаптации порогов доступны через включение соответствующего флага.
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict, Callable, Optional

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
from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import (
    detect_impulse_noise,
    adaptive_median_filter,
    remove_impulse_noise,
)


class ESM_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе улучшенной разреженной модели 
    с поддержкой адаптивного шумоподавления.

    Если все опции шумоподавления отключены ('none') и отключен 
    автоматический режим, конвейер полностью соответствует оригинальному 
    алгоритму ESM.

    Параметры алгоритма деконволюции
    --------------------------------
    kernel_size : int, по умолчанию 35
        Пространственный размер неизвестной функции рассеяния точки (нечетное число).
    lambda_data : float, по умолчанию 4e-3
        Весовой коэффициент для L0-L1 априорного распределения остатка градиентов данных.
    lambda_grad : float, по умолчанию 4e-3
        Весовой коэффициент для L0-L1 априорного распределения градиентов изображения.
    theta : float, по умолчанию 1.0
        Параметр улучшенного разреженного L0-L1 распределения, контролирующий 
        ширину зоны сжатия.
    xk_iter : int, по умолчанию 5
        Количество итераций слепой оценки на каждом уровне пирамиды масштабов.
    gamma_correct : float, по умолчанию 1.0
        Экспонента гамма-коррекции, применяемая перед оценкой ядра.
    k_thresh : float, по умолчанию 20.0
        Относительный порог обнуления шума в ядре.
    lambda_tv : float, по умолчанию 0.002
        Вес TV-регуляризации для этапа подавления звона.
    lambda_l0 : float, по умолчанию 2e-4
        Вес L0-регуляризации градиентов для этапа подавления звона.
    weight_ring : float, по умолчанию 1.0
        Коэффициент подавления артефактов звона.
    final_deconv : str, по умолчанию 'ringing_removal'
        Метод финальной неслепой деконволюции ('ringing_removal' или 'adaptive_lp').
    verbose : bool, по умолчанию False
        Флаг вывода подробной информации о решениях оркестратора.
    progress_callback : callable, опционально
        Функция обратного вызова для отслеживания прогресса многомасштабного решателя.

    Параметры шумоподавления (по умолчанию отключены)
    -------------------------------------------------
    impulse_preprocess : str, по умолчанию 'none'
        Режим подавления импульсного шума ('auto' или 'none').
    impulse_params : dict, опционально
        Параметры детектора импульсного шума.
    noise_estimation : str, по умолчанию 'none'
        Метод оценки дисперсии шума ('pca', 'chen' или 'none').
    screenot_preprocess : str, по умолчанию 'none'
        Использование SVD-фильтрации ScreeNOT перед слепой оценкой.
    screenot_params : dict, опционально
        Параметры фильтрации ScreeNOT.
    act_preprocess : str, по умолчанию 'none'
        Использование адаптивной фильтрации на основе курвлетов (ACT).
    act_params : dict, опционально
        Параметры ACT-фильтрации.
    preprocess : str, по умолчанию 'none'
        Пространственный фильтр для подготовки полутонового изображения.
    preprocess_params : dict, опционально
        Параметры пространственного фильтра. Можно передать 'use_vst': True 
        для использования обобщенного преобразования Энскомба.
    noise_preprocess : str, по умолчанию 'none'
        Фильтрация на основе анализа спектра мощности (PSD).
    noise_preprocess_params : dict, опционально
        Параметры PSD-фильтрации.
    blind_denoise : str, по умолчанию 'none'
        Метод фильтрации промежуточного скрытого изображения внутри 
        итераций оценки ядра.
    blind_denoise_params : dict, опционально
        Параметры фильтрации скрытого изображения.
    pre_nonblind : str, по умолчанию 'none'
        Метод фильтрации полноцветного изображения перед финальной деконволюцией.
    pre_nonblind_params : dict, опционально
        Параметры фильтра перед неслепым восстановлением.
    nb_params : dict, опционально
        Переопределение параметров финальной деконволюции.
    auto_mode : str, по умолчанию 'off'
        Режим работы оркестратора: 'off' или 'robust'.
    auto_mode_params : dict, опционально
        Параметры настройки оркестратора.

    Экспертные параметры адаптации к шуму
    -------------------------------------
    expert_noise_adapt : bool, по умолчанию False
        Включение специфичных для ESM модификаций, отклоняющихся от оригинальной 
        статьи: введение порогов обнуления шума при оценке скрытого изображения 
        и ядра (основанных на sigma), опциональная фильтрация градиентов данных.
    expert_noise_adapt_params : dict, опционально
        Параметры экспертной адаптации (коэффициенты жестких и мягких порогов, 
        флаги санирования градиентов).
    """

    def __init__(
        self,
        kernel_size: int = 35,
        lambda_data: float = 4e-3,
        lambda_grad: float = 4e-3,
        theta: float = 1.0,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.002,
        lambda_l0: float = 2e-4,
        weight_ring: float = 1.0,
        final_deconv: str = 'ringing_removal',
        verbose: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        impulse_preprocess: str = 'none',
        impulse_params: Optional[dict] = None,
        noise_estimation: str = 'none',
        screenot_preprocess: str = 'none',
        screenot_params: Optional[dict] = None,
        act_preprocess: str = 'none',
        act_params: Optional[dict] = None,
        preprocess: str = 'none',
        preprocess_params: Optional[dict] = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: Optional[dict] = None,
        blind_denoise: str = 'none',
        blind_denoise_params: Optional[dict] = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: Optional[dict] = None,
        nb_params: Optional[dict] = None,
        auto_mode: str = 'off',
        auto_mode_params: Optional[dict] = None,
        expert_noise_adapt: bool = False,
        expert_noise_adapt_params: Optional[dict] = None,
    ):
        super().__init__(name='ESM-BD')

        self.kernel_size = kernel_size
        self.lambda_data = lambda_data
        self.lambda_grad = lambda_grad
        self.theta = theta
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.final_deconv = (final_deconv or 'ringing_removal').lower()
        self.verbose = verbose
        self.progress_callback = progress_callback

        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.blind_denoise = blind_denoise
        self.blind_denoise_params = blind_denoise_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.nb_params = nb_params
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params
        self.expert_noise_adapt = bool(expert_noise_adapt)
        self.expert_noise_adapt_params = expert_noise_adapt_params

        self._defaults_snapshot = {
            'lambda_tv': float(lambda_tv),
            'lambda_l0': float(lambda_l0),
            'weight_ring': float(weight_ring),
            'final_deconv': self.final_deconv,
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'nb_params': nb_params,
        }

        self.history: Dict[str, list] = {
            'kernel_diff': [],
            'iterations': [],
            'scale_kernels': [],
        }
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск алгоритма слепой деконволюции с учетом конвейера шумоподавления.

        Процесс включает:
        1. Инициализацию истории выполнения и нормализацию входного изображения.
        2. Извлечение яркостного канала для оценки ядра размытия.
        3. Обработку дефектов: поиск и устранение импульсного шума, оценку 
           параметров шума (sigma), отработку робастного оркестратора.
        4. Предварительную фильтрацию (ScreeNOT, ACT, классические 
           пространственные и частотные фильтры).
        5. Многомасштабную оценку ядра методом ESM с опциональной фильтрацией 
           скрытого изображения на каждой итерации и использованием 
           экспертных адаптаций.
        6. Финальную фильтрацию полноцветного размытого изображения 
           и неслепую деконволюцию.

        Параметры
        ---------
        image : ndarray
            Входное изображение.

        Возвращает
        ----------
        x_final : ndarray
            Восстановленное изображение в формате int16 [0, 255].
        kernel : ndarray
            Оцененное ядро размытия.
        """
        start_time = time.time()

        self.history = {
            'kernel_diff': [],
            'iterations': [],
            'scale_kernels': [],
        }

        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 3:
            yg = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]
        elif y.ndim == 2:
            yg = y.copy()
        else:
            yg = y[:, :, 0]

        y_full = y

        def _luma(im):
            if im.ndim == 2:
                return im
            if im.shape[2] == 3:
                return (0.2989 * im[:, :, 0]
                        + 0.5870 * im[:, :, 1]
                        + 0.1140 * im[:, :, 2])
            return im[:, :, 0]

        def _per_channel(im, fn):
            if im.ndim == 2:
                return fn(im)
            out = np.empty_like(im)
            for _c in range(im.shape[2]):
                out[:, :, _c] = fn(im[:, :, _c])
            return out

        ks_size = int(self.kernel_size)
        if ks_size % 2 == 0:
            ks_size += 1

        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            res = remove_impulse_noise(
                y_full,
                density_threshold=ip.get('density_threshold', 0.0005),
                max_window=ip.get('max_window', 7),
                outlier_window=ip.get('outlier_window', 5),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
            )
            impulse_info = {
                'has_impulse': res['has_impulse'],
                'density': res['density'],
                'applied': res['applied'],
            }
            if res['applied']:
                y_full = res['image']
                if y_full.ndim == 3 and y_full.shape[2] == 3:
                    yg = (0.2989 * y_full[:, :, 0]
                          + 0.5870 * y_full[:, :, 1]
                          + 0.1140 * y_full[:, :, 2])
                elif y_full.ndim == 3:
                    yg = y_full[:, :, 0]
                else:
                    yg = y_full

        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
        elif self.auto_mode == 'robust':
            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(yg)
        elif self.expert_noise_adapt:
            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(yg)

        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)

        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from blinddeconv.algorithms.mod_denoise.screenot import screenot_denoise
            sp = self.screenot_params or {}
            sn_kw = dict(
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if y_full.ndim == 2:
                y_full, screenot_info = screenot_denoise(y_full, **sn_kw)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = screenot_denoise(
                        y_full[:, :, _c], **sn_kw)
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        screenot_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise as _act
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            act_kw = dict(
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if y_full.ndim == 2:
                y_full, act_info = _act(y_full, **act_kw)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = _act(y_full[:, :, _c], **act_kw)
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        act_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        if self.preprocess not in (None, 'none'):
            y_full = _per_channel(
                y_full,
                lambda im: self._apply_denoise(
                    im, self.preprocess, self.preprocess_params, noise_info),
            )
            yg = _luma(y_full)

        psd_info = None
        if self.noise_preprocess != 'none':
            if y_full.ndim == 2:
                y_full, psd_info = self._apply_noise_preprocess(y_full)
            else:
                _out = np.empty_like(y_full)
                for _c in range(y_full.shape[2]):
                    _cleaned, _info_c = self._apply_noise_preprocess(
                        y_full[:, :, _c])
                    _out[:, :, _c] = _cleaned
                    if _c == 0:
                        psd_info = _info_c
                y_full = _out
            yg = _luma(y_full)

        blind_denoise_fn = None
        if self.blind_denoise not in (None, 'none'):
            def blind_denoise_fn(s_arr, _info=noise_info):
                return self._apply_blind_denoise(s_arr, _info)

        progress_proxy = self._make_progress_proxy()

        expert_adapt = self._build_expert_adapt(noise_info)

        opts = {
            'kernel_size': ks_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'theta': self.theta,
            'expert_adapt': expert_adapt,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_data, self.lambda_grad, opts,
            blind_denoise_fn=blind_denoise_fn,
            progress_callback=progress_proxy,
        )

        if self.pre_nonblind not in (None, 'none'):
            y_full = self._apply_pre_nonblind(y_full, noise_info)

        Latent = self._final_deconv(y_full, kernel, noise_info)
        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': ks_size,
            'lambda_data': self.lambda_data,
            'lambda_grad': self.lambda_grad,
            'theta': self.theta,
            'xk_iter': self.xk_iter,
            'gamma_correct': self.gamma_correct,
            'k_thresh': self.k_thresh,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'final_deconv': self.final_deconv,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': ({k_: v for k_, v in (impulse_info or {}).items()
                              if k_ != 'impulse_mask'}
                             if impulse_info else None),
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'preprocess': self.preprocess,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': ({k_: v for k_, v in (psd_info or {}).items()
                          if k_ != 'psd_2d'} if psd_info else None),
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'expert_noise_adapt': self.expert_noise_adapt,
            'expert_adapt_active': expert_adapt is not None,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def _build_expert_adapt(self, noise_info):
        """
        Формирование словаря экспертных параметров для адаптации к шуму.

        Если флаг expert_noise_adapt установлен в False, возвращает None, 
        что обеспечивает полное соответствие поведения решателей оригинальной статье.
        """
        if not self.expert_noise_adapt:
            return None
        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)
        if sigma <= 0.0:
            return None
        ep = dict(self.expert_noise_adapt_params or {})

        adapt = {
            'sigma': sigma,
            'q_floor_factor': float(ep.get('q_floor_factor', 9.0)),
            'g_floor_factor': float(ep.get('g_floor_factor', 9.0)),
            'q_soft_floor_factor': float(ep.get('q_soft_floor_factor', 3.0)),
            'g_soft_floor_factor': float(ep.get('g_soft_floor_factor', 3.0)),
        }
        if ep.get('kappa') is not None:
            adapt['kappa'] = float(ep['kappa'])
        if ep.get('betamax') is not None:
            adapt['betamax'] = float(ep['betamax'])

        if ep.get('sanitize_grad', False):
            import cv2
            d = int(ep.get('sanitize_grad_d', 5))
            sc = float(max(2.0 * np.sqrt(2.0) * sigma, 0.02))

            def _sanitize_grad(g):
                return cv2.bilateralFilter(
                    g.astype(np.float32), d, sc, 5.0
                ).astype(np.float64)

            adapt['sanitize_grad_fn'] = _sanitize_grad

        if self.verbose:
            print(f"[{self.name}] expert_adapt(σ={sigma:.5f}): "
                  f"q_floor={adapt['q_floor_factor']}, "
                  f"g_floor={adapt['g_floor_factor']}, "
                  f"sanitize_grad={ep.get('sanitize_grad', False)}, "
                  f"kappa={adapt.get('kappa', 'paper')}")
        return adapt

    def _final_deconv(self, y_full, kernel, noise_info):
        """
        Маршрутизация финального этапа неслепой деконволюции.

        Выбирает метод восстановления (ringing_removal или adaptive_lp) 
        и применяет его к изображению.
        """
        nbp = self.nb_params or {}
        method = self.final_deconv

        if method == 'auto':
            method = 'ringing_removal'
            self.final_deconv = method

        if method == 'ringing_removal':
            return ringing_artifacts_removal(
                y_full, kernel,
                nbp.get('lambda_tv', self.lambda_tv),
                nbp.get('lambda_l0', self.lambda_l0),
                nbp.get('weight_ring', self.weight_ring),
            )

        if method == 'adaptive_lp':
            from blinddeconv.algorithms.mod_denoise.non_blind import adaptive_lp_deconv
            sigma_n = None
            if noise_info is not None:
                sigma_n = noise_info.get('sigma_norm', None)
            alpha = nbp.get('alpha', 0.8)
            two_stage = nbp.get('two_stage', True)

            if y_full.ndim == 2:
                return adaptive_lp_deconv(
                    y_full, kernel, alpha=alpha,
                    sigma_n=sigma_n, two_stage=two_stage)
            chans = []
            for c in range(y_full.shape[2]):
                chans.append(adaptive_lp_deconv(
                    y_full[:, :, c], kernel, alpha=alpha,
                    sigma_n=sigma_n, two_stage=two_stage))
            return np.stack(chans, axis=2)

        raise ValueError(
            f"Unknown final_deconv='{self.final_deconv}'. "
            "Choose 'ringing_removal' or 'adaptive_lp'.")

    def _make_progress_proxy(self):
        """
        Обертка для пользовательской функции обратного вызова.

        Записывает эволюцию параметров и ядра в словарь истории 
        и передает форматированные события во внешний логгер.
        """
        user_cb = self._callback 
        history = self.history

        def proxy(event):
            try:
                ev_type = event.get('event')
                if ev_type == 'iter':
                    history['kernel_diff'].append(event.get('kernel_diff'))
                    history['iterations'].append({
                        k: v for k, v in event.items() if k != 'kernel'
                    })
                elif ev_type == 'scale_end':
                    history['scale_kernels'].append({
                        'scale': event.get('scale'),
                        'kernel': event.get('kernel'),
                    })
            except Exception:
                pass
            if user_cb is not None and event.get('event') == 'iter':
                try:
                    _meta = {'event', 'scale', 'num_scales',
                             'kernel', 'image', 'iter'}
                    user_cb({
                        'iteration': event.get('iter', 0),
                        'scale': event.get('scale', 0),
                        'num_scales': event.get('num_scales', 1),
                        'kernel': event.get('kernel'),
                        'image': event.get('image', None),
                        'metrics': {k: v for k, v in event.items()
                                    if k not in _meta},
                    })
                except Exception:
                    pass

        return proxy

    @staticmethod
    def _guided_filter(I, p, r, eps):
        """
        Ускоренная реализация фильтра Guided Filter на основе усредняющих 
        фильтров прямоугольного окна.
        """
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
        """
        if method is None or method == 'none':
            return img
        p = dict(params or {})
        sigma = noise_info.get('sigma_norm', None) if noise_info else None

        if p.pop('use_vst', False) and noise_info is not None:
            return self._apply_denoise_vst(img, method, p, noise_info)

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

        elif method == 'ensemble':
            sig = p.get('sigma', sigma if sigma else 0.05)
            members = p.get('members', ('bm3d', 'nlm', 'bilateral'))
            weights = p.get('weights', None)
            outs = []
            for m in members:
                sub_params = dict(p)
                sub_params['sigma'] = sig
                if m == 'ensemble':
                    continue
                outs.append(self._apply_denoise(
                    img, m, sub_params, noise_info))
            if not outs:
                return img
            if weights is None:
                return np.mean(np.stack(outs, axis=0), axis=0)
            ws = np.asarray(weights, dtype=np.float64)
            ws = ws / ws.sum()
            stacked = np.stack(outs, axis=0)
            return np.tensordot(ws, stacked, axes=(0, 0))

        else:
            raise ValueError(
                f"Unknown denoiser='{method}'. Choose from: "
                f"'tv', 'nlm', 'bilateral', 'guided', 'bm3d', "
                f"'act', 'ensemble', 'none'")

    def _apply_denoise_vst(self, img, method, params, noise_info):
        """
        Обобщенное преобразование Энскомба (VST) для фильтрации данных, 
        подверженных пуассоновско-гауссовскому шуму.
        """
        a = float(noise_info.get('a', 0.0) or 0.0) if noise_info else 0.0
        b = float(noise_info.get('b', 0.0) or 0.0) if noise_info else 0.0
        A = a / 255.0
        B = b / (255.0 ** 2)

        if A < 1e-8:
            sub = dict(params)
            sub.setdefault('sigma',
                           noise_info.get('sigma_norm', None)
                           if noise_info else None)
            return self._apply_denoise(img, method, sub, noise_info)

        inner = np.maximum(A * img + (3.0 / 8.0) * A * A + B, 0.0)
        z = (2.0 / np.sqrt(A)) * np.sqrt(inner)

        zmax = float(max(z.max(), 1e-6))
        z_scaled = z / zmax
        sigma_scaled = 1.0 / zmax

        sub = dict(params)
        sub['sigma'] = sigma_scaled
        sub['weight'] = max(0.005, 2.0 * sigma_scaled)
        sub['sigma_color'] = sigma_scaled
        inner_info = {'sigma_norm': sigma_scaled, 'method': 'vst'}
        denoised_scaled = self._apply_denoise(
            z_scaled, method, sub, inner_info)

        z_clean = denoised_scaled * zmax

        x_rec = (z_clean / 2.0) ** 2 - (3.0 / 8.0) * A - B / max(A, 1e-8)
        return np.clip(x_rec, 0.0, 1.0)

    def _estimate_noise(self, yg):
        """
        Диспетчеризация методов оценки дисперсии шума.
        """
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

    def _orchestrate_robust(self, noise_info):
        """
        Автоматическая адаптация параметров конвейера на основе уровня шума.

        Базовые весовые коэффициенты ESM (lambda_data, lambda_grad, theta, 
        kernel_size, xk_iter) остаются неизменными согласно оригинальной статье. 
        Подстраиваются только параметры финальной деконволюции и 
        выбор алгоритмов шумоподавления в зависимости от интенсивности шума (sigma).
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        self.lambda_tv = snap['lambda_tv']
        self.lambda_l0 = snap['lambda_l0']
        self.weight_ring = snap['weight_ring']
        self.final_deconv = snap['final_deconv']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.blind_denoise = snap['blind_denoise']
        self.blind_denoise_params = snap['blind_denoise_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.nb_params = snap['nb_params']

        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))

        force_heavy = False
        nt = (noise_info or {}).get('noise_type', None)
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.01))
        if nt in ('poisson', 'poisson_gaussian') and sigma >= force_heavy_sigma:
            force_heavy = True

        if sigma <= sigma_clean and not force_heavy:
            if snap['final_deconv'] == 'auto':
                self.final_deconv = 'ringing_removal'
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"defaults kept, final_deconv={self.final_deconv}")
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'final_deconv': self.final_deconv,
                'preprocess': self.preprocess,
                'blind_denoise': self.blind_denoise,
                'pre_nonblind': self.pre_nonblind,
                'lambda_tv': float(self.lambda_tv),
                'lambda_l0': float(self.lambda_l0),
                'weight_ring': float(self.weight_ring),
            }

        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / (sigma_heavy - sigma_clean))
        regime = 'heavy' if w > 0.95 else 'medium'

        noise_type = (noise_info or {}).get('noise_type', 'gaussian')
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        k_lambda_tv = float(amp.get('k_lambda_tv', 0.05))
        k_lambda_l0 = float(amp.get('k_lambda_l0', 0.01))
        k_weight_ring = float(amp.get('k_weight_ring', 1.0))
        k_alpha = float(amp.get('k_alpha', 0.1))

        lam_tv_noisy = max(snap['lambda_tv'], k_lambda_tv * sigma)
        lam_l0_noisy = max(snap['lambda_l0'], k_lambda_l0 * sigma)
        wring_noisy = min(2.0, snap['weight_ring'] + k_weight_ring * sigma)

        self.lambda_tv = (1.0 - w) * snap['lambda_tv'] + w * lam_tv_noisy
        self.lambda_l0 = (1.0 - w) * snap['lambda_l0'] + w * lam_l0_noisy
        self.weight_ring = (1.0 - w) * snap['weight_ring'] + w * wring_noisy

        if w < 0.5:
            self.blind_denoise = 'bilateral'
            self.blind_denoise_params = {
                'sigma_color': float(max(sigma, 0.01)),
                'sigma_space': 5.0,
            }
        else:
            self.blind_denoise = 'bilateral'
            self.blind_denoise_params = {
                'sigma_color': float(max(2.0 * sigma, 0.02)),
                'sigma_space': 7.0,
            }

        if poisson_like:
            self.preprocess = 'act'
            self.preprocess_params = {'threshold_setting': 's'}
        elif w < 0.6:
            self.preprocess = 'bilateral'
            self.preprocess_params = {
                'sigma_color': float(sigma),
                'sigma_space': 5.0,
            }
        else:
            self.preprocess = 'bm3d'
            self.preprocess_params = {'sigma': float(sigma)}

        if poisson_like:
            self.pre_nonblind = 'act'
            self.pre_nonblind_params = {'threshold_setting': 's'}
        elif w < 0.6:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma': float(max(sigma, 0.01))}
        else:
            self.pre_nonblind = 'ensemble'
            self.pre_nonblind_params = {
                'sigma': float(sigma),
                'members': amp.get('ensemble_members',
                                   ('bm3d', 'nlm', 'bilateral')),
            }

        if snap['final_deconv'] == 'auto':
            if poisson_like and w >= 0.5:
                self.final_deconv = 'adaptive_lp'
                if self.nb_params is None:
                    self.nb_params = {
                        'alpha': max(0.5, 0.8 - k_alpha * sigma),
                        'two_stage': True,
                    }
            else:
                self.final_deconv = 'ringing_removal'

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'lambda_tv': float(self.lambda_tv),
            'lambda_l0': float(self.lambda_l0),
            'weight_ring': float(self.weight_ring),
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={noise_type}): "
                  f"λ_tv={self.lambda_tv:.5f}, λ_l0={self.lambda_l0:.5f}, "
                  f"w_ring={self.weight_ring:.3f}, "
                  f"pre={self.preprocess}, blind={self.blind_denoise}, "
                  f"pre_nb={self.pre_nonblind}, "
                  f"final={self.final_deconv}")
        return info

    def _apply_noise_preprocess(self, yg):
        """
        Устранение периодических шумовых структур на основе анализа 
        спектра плотности мощности (PSD).
        """
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

    def _apply_blind_denoise(self, x, noise_info):
        """
        Фильтрация скрытого изображения внутри цикла оценки ядра.
        """
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    def _apply_pre_nonblind(self, img, noise_info):
        """
        Независимое поканальное применение пространственного фильтра 
        к полноцветному изображению перед финальной деконволюцией.
        """
        if img.ndim == 2:
            return self._apply_denoise(
                img, self.pre_nonblind, self.pre_nonblind_params, noise_info)
        chans = []
        for c in range(img.shape[2]):
            chans.append(self._apply_denoise(
                img[:, :, c], self.pre_nonblind,
                self.pre_nonblind_params, noise_info))
        return np.stack(chans, axis=2)

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_data', self.lambda_data),
            ('lambda_grad', self.lambda_grad),
            ('theta', self.theta),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('final_deconv', self.final_deconv),
            ('verbose', self.verbose),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('noise_estimation', self.noise_estimation),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params', self.screenot_params),
            ('act_preprocess', self.act_preprocess),
            ('act_params', self.act_params),
            ('preprocess', self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('blind_denoise', self.blind_denoise),
            ('blind_denoise_params', self.blind_denoise_params),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('nb_params', self.nb_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
            ('expert_noise_adapt', self.expert_noise_adapt),
            ('expert_noise_adapt_params', self.expert_noise_adapt_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in self._defaults_snapshot:
                    self._defaults_snapshot[key] = (
                        float(value)
                        if key in ('lambda_tv', 'lambda_l0', 'weight_ring')
                        else value
                    )

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
