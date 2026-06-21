"""
lmgp.py

Модуль слепой деконволюции изображений на основе априорной информации
о локальном максимальном градиенте с интегрированным конвейером 
робастного шумоподавления и адаптивной оценки параметров.

Основано на методе:
    L. Chen, F. Fang, T. Wang, G. Zhang: "Blind Image Deblurring
    With Local Maximum Gradient Prior", CVPR, 2019.

Общий конвейер обработки включает последовательное выполнение следующих этапов:
1. Предварительное обнаружение и фильтрация импульсного шума.
2. Слепая оценка дисперсии аддитивного гауссова шума.
3. Опциональная низкоранговая или кривлет-фильтрация (SVD/ACT) и 
   пространственное сглаживание до построения пирамиды масштабов.
4. Спектральная обработка для подавления периодических помех (PSD-фильтрация).
5. Итеративная многомасштабная слепая деконволюция с промежуточной 
   регуляризацией градиентов скрытого изображения.
6. Финальная неслепая деконволюция с использованием алгоритмов минимизации 
   полной вариации, L0-нормы градиента или адаптивной фильтрации 
   для подавления краевых артефактов.
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

from .solvers import (
    blind_deconv,
    ringing_artifacts_removal,
    L0Restoration,
    deblurring_adm_aniso,
)
from .non_blind import adaptive_lp_deconv
from blinddeconv.algorithms.mod_cython._build_pyd.impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from .utils import opt_fft_size, wrap_boundary_liu


class LMGP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе априорного знания локального
    максимального градиента с расширенным конвейером робастной оценки.

    Параметры базовой модели
    ------------------------
    kernel_size : int, по умолчанию 27
        Линейный размер квадратного пространственного носителя функции рассеяния точки.
    lambda_lmg : float, по умолчанию 4e-3
        Коэффициент регуляризации априорного члена локального максимального градиента.
    lambda_grad : float, по умолчанию 4e-3
        Коэффициент регуляризации разреженности градиента скрытого изображения.
    xk_iter : int, по умолчанию 5
        Количество итераций попеременной минимизации на каждом масштабном уровне.
    gamma_correct : float, по умолчанию 1.0
        Коэффициент степенного преобразования перед оценкой ядра.
    k_thresh : float, по умолчанию 20.0
        Порог отсечения малых амплитуд в оцененном ядре размытия.
    lambda_tv : float, по умолчанию 0.001
        Вес полной вариации на этапе финального подавления артефактов.
    lambda_l0 : float, по умолчанию 5e-4
        Вес L0-нормы градиента на этапе финального подавления артефактов.
    weight_ring : float, по умолчанию 1.0
        Коэффициент силы компенсации артефактов звона.

    Параметры фильтрации при оценке градиентов
    ------------------------------------------
    denoise_eps : float, опционально
        Параметр регуляризации для направленного фильтра (guided filter).
    denoise_radius : int, по умолчанию 2
        Радиус окна направленного фильтра.
    ensemble_denoise : bool, по умолчанию False
        Использование ансамбля направленных фильтров разных радиусов.
    denoise_type : str, по умолчанию 'guided'
        Тип пространственного фильтра (guided, bilateral, bm3d, nlm).
    denoise_bilateral_sigma_s : float, по умолчанию 2.0
        Пространственное среднеквадратичное отклонение двустороннего фильтра.
    denoise_bilateral_sigma_r : float, по умолчанию 0.1
        Амплитудное среднеквадратичное отклонение двустороннего фильтра.
    denoise_bm3d_sigma : float, по умолчанию 0.01
        Ожидаемое среднеквадратичное отклонение шума для фильтра BM3D.
    denoise_nlm_h : float, по умолчанию 0.01
        Сила сглаживания для фильтра нелокальных средних.

    Параметры фильтрации для оператора локального максимального градиента
    ---------------------------------------------------------------------
    lmg_denoise_eps : float, опционально
        Регуляризация направленного фильтра перед вычислением оператора.
    lmg_denoise_radius : int, по умолчанию 2
        Радиус окна направленного фильтра перед вычислением оператора.
    lmg_denoise_type : str, по умолчанию 'guided'
        Тип фильтра (guided, bilateral, bm3d, nlm).
    lmg_bilateral_sigma_s : float, по умолчанию 2.0
        Пространственное отклонение двустороннего фильтра.
    lmg_bilateral_sigma_r : float, по умолчанию 0.1
        Амплитудное отклонение двустороннего фильтра.
    lmg_bm3d_sigma : float, по умолчанию 0.01
        Уровень шума для BM3D фильтрации.
    lmg_nlm_h : float, по умолчанию 0.01
        Сила сглаживания NLM фильтрации.

    Расширенные параметры робастности
    ---------------------------------
    grad_smooth_sigma : float, опционально
        Отклонение гауссова фильтра для сглаживания градиентов.
    use_soft_threshold : bool, по умолчанию False
        Использование мягкого отсечения на основе L1-нормы.
    softmax_tau : float, опционально
        Температурный параметр для перевода операции локального максимума 
        в непрерывную функцию (мягкий максимум).
    kernel_reg_weight : float, по умолчанию 0.0
        Коэффициент регуляризации Тихонова для матрицы функции рассеяния точки.

    Параметры финального восстановления
    -----------------------------------
    nonblind_method : str, по умолчанию 'ringing_removal'
        Метод финального неслепого восстановления (ringing_removal, tv_adm, 
        l0, adaptive_lp).
    nonblind_params : dict, опционально
        Дополнительные параметры для выбранного метода восстановления.

    Оценка и подавление шума
    ------------------------
    noise_estimation : str, по умолчанию 'none'
        Метод автоматической оценки дисперсии шума (pca, chen, none).
    auto_params : bool, по умолчанию False
        Автоматическое переопределение порогов и параметров фильтрации 
        на основе оцененной дисперсии шума.
    noise_preprocess : str, по умолчанию 'none'
        Метод спектрального подавления шума.
    impulse_preprocess : str, по умолчанию 'none'
        Метод подавления импульсного шума.
    auto_mode : str, по умолчанию 'off'
        Режим работы робастного оркестратора. При значении robust позволяет 
        алгоритму динамически перестраивать конвейер в зависимости от уровня шума.
    """

    def __init__(
        self,
        kernel_size: int = 27,
        lambda_lmg: float = 4e-3,
        lambda_grad: float = 4e-3,
        xk_iter: int = 5,
        gamma_correct: float = 1.0,
        k_thresh: float = 20.0,
        lambda_tv: float = 0.001,
        lambda_l0: float = 5e-4,
        weight_ring: float = 1.0,
        denoise_eps: float = None,
        denoise_radius: int = 2,
        ensemble_denoise: bool = False,
        denoise_type: str = 'guided',
        denoise_bilateral_sigma_s: float = 2.0,
        denoise_bilateral_sigma_r: float = 0.1,
        denoise_bm3d_sigma: float = 0.01,
        denoise_nlm_h: float = 0.01,
        grad_smooth_sigma: float = None,
        lmg_denoise_eps: float = None,
        lmg_denoise_radius: int = 2,
        lmg_denoise_type: str = 'guided',
        lmg_bilateral_sigma_s: float = 2.0,
        lmg_bilateral_sigma_r: float = 0.1,
        lmg_bm3d_sigma: float = 0.01,
        lmg_nlm_h: float = 0.01,
        use_soft_threshold: bool = False,
        softmax_tau: float = None,
        kernel_reg_weight: float = 0.0,
        use_pmp_nonblind: bool = False,
        pmp_lambda: float = 0.1,
        pmp_patch_r: int = 3,
        pmp_quantile: float = 0.0,
        nonblind_method: str = 'ringing_removal',
        nonblind_params: dict = None,
        preprocess: str = 'none',
        preprocess_params: dict = None,
        noise_estimation: str = 'none',
        auto_params: bool = False,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        impulse_preprocess: str = 'none',
        impulse_density_threshold: float = 0.0005,
        impulse_outlier_threshold: float = 0.08,
        impulse_max_window: int = 7,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        histogram_eq: str = 'none',
        histogram_eq_params: dict = None,
        kernel_eq: str = 'none',
        kernel_eq_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
    ):
        super().__init__(name='LMGP-BD')

        self.kernel_size = kernel_size
        self.lambda_lmg = lambda_lmg
        self.lambda_grad = lambda_grad
        self.xk_iter = xk_iter
        self.gamma_correct = gamma_correct
        self.k_thresh = k_thresh
        self.lambda_tv = lambda_tv
        self.lambda_l0 = lambda_l0
        self.weight_ring = weight_ring
        self.denoise_eps = denoise_eps
        self.denoise_radius = denoise_radius
        self.ensemble_denoise = ensemble_denoise
        self.denoise_type = denoise_type
        self.denoise_bilateral_sigma_s = denoise_bilateral_sigma_s
        self.denoise_bilateral_sigma_r = denoise_bilateral_sigma_r
        self.denoise_bm3d_sigma = denoise_bm3d_sigma
        self.denoise_nlm_h = denoise_nlm_h
        self.grad_smooth_sigma = grad_smooth_sigma
        self.lmg_denoise_eps = lmg_denoise_eps
        self.lmg_denoise_radius = lmg_denoise_radius
        self.lmg_denoise_type = lmg_denoise_type
        self.lmg_bilateral_sigma_s = lmg_bilateral_sigma_s
        self.lmg_bilateral_sigma_r = lmg_bilateral_sigma_r
        self.lmg_bm3d_sigma = lmg_bm3d_sigma
        self.lmg_nlm_h = lmg_nlm_h
        self.use_soft_threshold = use_soft_threshold
        self.softmax_tau = softmax_tau
        self.kernel_reg_weight = kernel_reg_weight
        self.use_pmp_nonblind = use_pmp_nonblind
        self.pmp_lambda = pmp_lambda
        self.pmp_patch_r = pmp_patch_r
        self.pmp_quantile = pmp_quantile
        self.nonblind_method = nonblind_method
        self.nonblind_params = nonblind_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.impulse_preprocess = impulse_preprocess
        self.impulse_density_threshold = impulse_density_threshold
        self.impulse_outlier_threshold = impulse_outlier_threshold
        self.impulse_max_window = impulse_max_window
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.histogram_eq = histogram_eq
        self.histogram_eq_params = histogram_eq_params
        self.kernel_eq = kernel_eq
        self.kernel_eq_params = kernel_eq_params

        self.auto_mode = auto_mode
        self.auto_mode_params = auto_mode_params
        self._defaults_snapshot = {
            'preprocess':          preprocess,
            'preprocess_params':   preprocess_params,
            'pre_nonblind':        pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'act_preprocess':      act_preprocess,
            'act_params':          act_params,
            'nonblind_method':     nonblind_method,
            'nonblind_params':     nonblind_params,
            'nb_params': {
                'lambda_tv':   lambda_tv,
                'lambda_l0':   lambda_l0,
                'weight_ring': weight_ring,
            },
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск расширенного конвейера слепой деконволюции.

        Выполняет нормирование, опциональную предварительную фильтрацию,
        слепую оценку функции рассеяния точки и финальное восстановление
        резкости изображения.

        Параметры
        ---------
        image : np.ndarray
            Входное искаженное изображение.

        Возвращаемое значение
        ---------------------
        Tuple[np.ndarray, np.ndarray]
            Кортеж, содержащий восстановленное изображение в целочисленном формате
            и вещественную матрицу оцененного ядра искажения.
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
            impulse_info = detect_impulse_noise(
                yg,
                density_threshold=self.impulse_density_threshold,
                outlier_threshold=self.impulse_outlier_threshold,
            )
            if impulse_info['has_impulse']:
                yg = adaptive_median_filter(
                    yg, impulse_info['impulse_mask'],
                    max_window=self.impulse_max_window)
                if y.ndim == 3:
                    for ch in range(y.shape[2]):
                        ch_info = detect_impulse_noise(
                            y[:, :, ch],
                            density_threshold=self.impulse_density_threshold,
                            outlier_threshold=self.impulse_outlier_threshold,
                        )
                        if ch_info['has_impulse']:
                            y[:, :, ch] = adaptive_median_filter(
                                y[:, :, ch], ch_info['impulse_mask'],
                                max_window=self.impulse_max_window)
                else:
                    y = yg.copy()
        if self.auto_mode == 'robust' and self.noise_estimation == 'none':
            self.noise_estimation = 'pca'
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)

        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)

        overrides = {}
        pp_eff = self.preprocess_params
        if self.auto_params and noise_info is not None:
            sigma = noise_info.get('sigma_norm', 0.0)
            overrides = self._compute_adaptive_params(sigma)
            if self.preprocess not in (None, 'none'):
                pp_eff = self._adapt_preprocess_params(sigma)

        def eff(name):
            return overrides[name] if name in overrides else getattr(self, name)

        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from blinddeconv.algorithms.mod_cython._build_pyd.screenot import screenot_denoise
            sp = self.screenot_params or {}
            yg, screenot_info = screenot_denoise(
                yg,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = screenot_denoise(
                        y[:, :, ch],
                        k=sp.get('k', 10),
                        strategy=sp.get('strategy', 'i'),
                        mode=sp.get('mode', 'full'),
                        patch_size=sp.get('patch_size', 8),
                        stride=sp.get('stride', 3),
                    )
            else:
                y = yg.copy()

        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from blinddeconv.algorithms.mod_cython._build_pyd.act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            yg, act_info = act_denoise(
                yg,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = act_denoise(
                        y[:, :, ch],
                        noise_var=act_noise_var,
                        threshold_setting=ap.get('threshold_setting', 's'),
                    )
            else:
                y = yg.copy()

        if self.preprocess not in (None, 'none'):
            yg = self._apply_preprocess(yg, pp_eff)
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch] = self._apply_preprocess(y[:, :, ch], pp_eff)
            else:
                y = yg.copy()

        psd_info = None
        if self.noise_preprocess != 'none':
            yg, psd_info = self._apply_noise_preprocess(yg)

        if self.histogram_eq not in (None, 'none'):
            yg = self._apply_histogram_eq(yg)

        opts = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'xk_iter': self.xk_iter,
            'k_thresh': self.k_thresh,
            'denoise_eps': eff('denoise_eps'),
            'denoise_radius': self.denoise_radius,
            'ensemble_denoise': self.ensemble_denoise,
            'denoise_type': self.denoise_type,
            'denoise_bilateral_sigma_s': self.denoise_bilateral_sigma_s,
            'denoise_bilateral_sigma_r': eff('denoise_bilateral_sigma_r'),
            'denoise_bm3d_sigma': eff('denoise_bm3d_sigma'),
            'denoise_nlm_h': eff('denoise_nlm_h'),
            'grad_smooth_sigma': self.grad_smooth_sigma,
            'lmg_denoise_eps': eff('lmg_denoise_eps'),
            'lmg_denoise_radius': self.lmg_denoise_radius,
            'lmg_denoise_type': self.lmg_denoise_type,
            'lmg_bilateral_sigma_s': self.lmg_bilateral_sigma_s,
            'lmg_bilateral_sigma_r': eff('lmg_bilateral_sigma_r'),
            'lmg_bm3d_sigma': eff('lmg_bm3d_sigma'),
            'lmg_nlm_h': eff('lmg_nlm_h'),
            'use_soft_threshold': self.use_soft_threshold,
            'softmax_tau': self.softmax_tau,
            'kernel_reg_weight': self.kernel_reg_weight,
            'kernel_eq': self.kernel_eq,
            'kernel_eq_params': self.kernel_eq_params,
        }

        kernel, interim_latent = blind_deconv(
            yg, self.lambda_lmg, self.lambda_grad, opts,
            iteration_callback=self._callback,
        )

        if self.pre_nonblind not in (None, 'none'):
            y = self._apply_pre_nonblind(y, noise_info)

        if self.nonblind_method == 'ringing_removal':
            if y.ndim == 3:
                Latent = np.zeros_like(y)
                for ch in range(y.shape[2]):
                    Latent[:, :, ch] = ringing_artifacts_removal(
                        y[:, :, ch], kernel,
                        eff('lambda_tv'), eff('lambda_l0'), eff('weight_ring'),
                        use_pmp_nonblind=self.use_pmp_nonblind,
                        pmp_lambda=self.pmp_lambda,
                        pmp_patch_r=self.pmp_patch_r,
                        pmp_quantile=self.pmp_quantile)
            else:
                Latent = ringing_artifacts_removal(
                    y, kernel,
                    eff('lambda_tv'), eff('lambda_l0'), eff('weight_ring'),
                    use_pmp_nonblind=self.use_pmp_nonblind,
                    pmp_lambda=self.pmp_lambda,
                    pmp_patch_r=self.pmp_patch_r,
                    pmp_quantile=self.pmp_quantile)
        else:
            nb_p = self.nonblind_params or {}
            if y.ndim == 3:
                Latent = np.zeros_like(y)
                for ch in range(y.shape[2]):
                    Latent[:, :, ch] = self._nonblind_single(
                        y[:, :, ch], kernel, self.nonblind_method, nb_p)
            else:
                Latent = self._nonblind_single(
                    y, kernel, self.nonblind_method, nb_p)

        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'lambda_lmg': self.lambda_lmg,
            'lambda_grad': self.lambda_grad,
            'lambda_tv': self.lambda_tv,
            'lambda_l0': self.lambda_l0,
            'weight_ring': self.weight_ring,
            'denoise_eps': self.denoise_eps,
            'lmg_denoise_eps': self.lmg_denoise_eps,
            'nonblind_method': self.nonblind_method,
            'use_pmp_nonblind': self.use_pmp_nonblind,
            'preprocess': self.preprocess,
            'noise_estimation': self.noise_estimation,
            'auto_params': self.auto_params,
            'noise_preprocess': self.noise_preprocess,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_density_threshold': self.impulse_density_threshold,
            'impulse_outlier_threshold': self.impulse_outlier_threshold,
            'impulse_max_window': self.impulse_max_window,
            'impulse_info': {k: v for k, v in (impulse_info or {}).items()
                            if k != 'impulse_mask'} if impulse_info else None,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_params': self.screenot_params,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_params': self.act_params,
            'act_info': act_info,
            'pre_nonblind': self.pre_nonblind,
            'pre_nonblind_params': self.pre_nonblind_params,
            'histogram_eq': self.histogram_eq,
            'histogram_eq_params': self.histogram_eq_params,
            'kernel_eq': self.kernel_eq,
            'kernel_eq_params': self.kernel_eq_params,
            'noise_info': noise_info,
            'psd_info': {k: v for k, v in (psd_info or {}).items()
                         if k != 'psd_2d'} if psd_info else None,
            'effective_overrides': overrides if overrides else None,
            'auto_mode': self.auto_mode,
            'auto_mode_params': self.auto_mode_params,
            'orchestrator_info': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def _nonblind_single(self, y_ch, kernel, method, params):
        """
        Выполнение финальной деконволюции для выбранного альтернативного метода.
        """
        if method == 'tv_adm':
            H, W = y_ch.shape
            target_size = opt_fft_size(
                np.array([H, W]) + np.array(kernel.shape[:2]) - 1)
            y_pad = wrap_boundary_liu(y_ch, tuple(target_size))
            result = deblurring_adm_aniso(
                y_pad, kernel,
                lambda_tv=params.get('lambda_tv', 2e-3),
                alpha=params.get('alpha', 1))
            return result[:H, :W]
        elif method == 'l0':
            return L0Restoration(
                y_ch, kernel,
                lambda_grad=params.get('lambda_grad', 2e-3),
                kappa=params.get('kappa', 2.0))
        elif method == 'adaptive_lp':
            return adaptive_lp_deconv(
                y_ch, kernel,
                alpha=params.get('alpha', 0.8),
                sigma_n=params.get('sigma_n', None),
                two_stage=params.get('two_stage', True))
        else:
            raise ValueError(
                f"Unknown nonblind_method='{method}'. "
                f"Choose from: 'ringing_removal', 'tv_adm', 'l0', "
                f"'adaptive_lp'")

    def _apply_pre_nonblind(self, y, noise_info):
        """
        Применение пространственной фильтрации перед этапом неслепой деконволюции.
        Подавляет коррелированный шум для предотвращения структурированных артефактов.
        """
        method = self.pre_nonblind
        params = dict(self.pre_nonblind_params or {})

        sigma = None
        if noise_info is not None:
            sigma = noise_info.get('sigma_norm', None)

        if method == 'act':
            from blinddeconv.algorithms.mod_cython._build_pyd.act_denoise import act_denoise
            nv = params.get('noise_var', None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = params.get('threshold_setting', 's')
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = act_denoise(
                        y[:, :, ch], noise_var=nv,
                        threshold_setting=ts)
            else:
                y, _ = act_denoise(y, noise_var=nv,
                                   threshold_setting=ts)
            return y

        if method == 'bm3d':
            if 'sigma' not in params and sigma is not None:
                params['sigma'] = sigma
        elif method == 'nlm':
            if 'h' not in params and sigma is not None:
                params['h'] = 0.8 * sigma
        elif method == 'bilateral':
            if 'sigma_color' not in params and sigma is not None:
                params['sigma_color'] = sigma
        elif method == 'guided':
            if 'eps' not in params and sigma is not None:
                params['eps'] = sigma ** 2 * 4

        saved = self.preprocess
        self.preprocess = method
        try:
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch] = self._apply_preprocess(y[:, :, ch], params)
            else:
                y = self._apply_preprocess(y, params)
        finally:
            self.preprocess = saved
        return y

    def _apply_histogram_eq(self, img):
        """
        Применение эквализации гистограммы к полутоновому изображению
        для усиления контраста структурных элементов.
        """
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

    def _apply_noise_preprocess(self, yg):
        """
        Оценка спектральной плотности мощности шума и фильтрация периодических помех.
        """
        from blinddeconv.algorithms.mod_cython._build_pyd.noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess,
            prewhiten, notch_filter, bandstop_filter,
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

        if mode == 'prewhiten':
            import warnings
            warnings.warn(
                "Prewhitening uses patch-based PSD which contains signal "
                "contamination. This WILL distort the image. Use with caution.",
                stacklevel=2)
            yg_out = prewhiten(yg, psd_info['psd_2d'],
                               reg=p.get('prewhiten_reg', 1e-3))
        elif mode == 'notch':
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
                f"Choose from: 'auto', 'prewhiten', 'notch', 'bandstop', 'none'")

        return yg_out, psd_info

    def _orchestrate_robust(self, noise_info):
        """
        Динамическое управление параметрами конвейера на основе оценки уровня шума.

        Стратегия: восстанавливает параметры из исходного состояния для идемпотентности.
        При превышении заданного порога шума активируются жесткие методы
        предварительной фильтрации (например, кривлет-преобразование или BM3D),
        а веса регуляризаторов финальной деконволюции масштабируются пропорционально
        оцененной дисперсии для предотвращения роста артефактов.
        """
        snap = self._defaults_snapshot

        self.preprocess          = snap['preprocess']
        self.preprocess_params   = snap['preprocess_params']
        self.pre_nonblind        = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.act_preprocess      = snap['act_preprocess']
        self.act_params          = snap['act_params']
        self.nonblind_method     = snap['nonblind_method']
        self.nonblind_params     = snap['nonblind_params']
        self.lambda_tv   = snap['nb_params']['lambda_tv']
        self.lambda_l0   = snap['nb_params']['lambda_l0']
        self.weight_ring = snap['nb_params']['weight_ring']

        if noise_info is None:
            return {'triggered': False, 'reason': 'no_noise_info',
                    'branch': 'clean'}

        sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)
        ntype = noise_info.get('noise_type', 'unknown')
        poisson_like = ntype in ('poisson', 'poisson_gaussian', 'unknown')

        p = self.auto_mode_params or {}
        sigma_clean = float(p.get('sigma_clean', 0.005))
        sigma_heavy = float(p.get('sigma_heavy', 0.05))
        force_sigma = float(p.get('force_heavy_sigma', 0.01))
        prefer_act_gauss = bool(p.get('prefer_act_for_gaussian', False))

        heavy = (sigma > sigma_clean) or (poisson_like and sigma > force_sigma)
        if not heavy:
            return {
                'triggered': True, 'branch': 'clean',
                'sigma': sigma, 'noise_type': ntype,
                'sigma_clean': sigma_clean,
            }

        denom = max(sigma_heavy - sigma_clean, 1e-9)
        w = max(0.0, min(1.0, (sigma - sigma_clean) / denom))

        decisions = {
            'triggered': True, 'branch': 'heavy',
            'sigma': sigma, 'noise_type': ntype, 'w': w,
            'sigma_clean': sigma_clean, 'sigma_heavy': sigma_heavy,
            'poisson_like': poisson_like,
            'prefer_act_for_gaussian': prefer_act_gauss,
        }

        use_act_branch = poisson_like or prefer_act_gauss

        if use_act_branch:
            self.preprocess          = 'none'
            self.preprocess_params   = None
            self.act_preprocess      = 'auto'
            self.act_params = {
                'noise_var': sigma ** 2,
                'threshold_setting': 's',
            }
            self.pre_nonblind        = 'act'
            self.pre_nonblind_params = {
                'noise_var': sigma ** 2,
                'threshold_setting': 's',
            }
            decisions['route'] = 'act'
        else:
            self.act_preprocess = 'none'
            self.act_params     = None
            if w < 0.6:
                self.preprocess        = 'bilateral'
                self.preprocess_params = {
                    'sigma_color': max(0.01, sigma * 2.0),
                    'sigma_space': 5.0,
                }
                self.pre_nonblind        = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma}
                decisions['route'] = 'gauss_light'
            else:
                self.preprocess        = 'bm3d'
                self.preprocess_params = {'sigma': sigma}
                self.pre_nonblind        = 'bm3d'
                self.pre_nonblind_params = {'sigma': sigma * 1.5}
                decisions['route'] = 'gauss_strong'

        if self.nonblind_method == 'ringing_removal':
            base = snap['nb_params']
            noisy = {
                'lambda_tv':   sigma * 0.5,
                'lambda_l0':   sigma * 0.025,
                'weight_ring': max(0.3, min(1.0, sigma * 50.0)),
            }
            self.lambda_tv = (
                (1.0 - w) * float(base['lambda_tv']) + w * noisy['lambda_tv']
            )
            self.lambda_l0 = (
                (1.0 - w) * float(base['lambda_l0']) + w * noisy['lambda_l0']
            )
            self.weight_ring = (
                (1.0 - w) * float(base['weight_ring'])
                + w * noisy['weight_ring']
            )
            decisions['nb_blend'] = {
                'lambda_tv':   self.lambda_tv,
                'lambda_l0':   self.lambda_l0,
                'weight_ring': self.weight_ring,
            }

        return decisions

    def _estimate_noise(self, yg):
        """
        Оценка уровня дисперсии шума из полутонового изображения.
        """
        if self.noise_estimation == 'chen':
            from blinddeconv.algorithms.mod_cython._build_pyd.chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pca':
            from blinddeconv.algorithms.mod_cython._build_pyd.pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pca'
            return result
        return None

    def _compute_adaptive_params(self, sigma):
        """
        Вычисление словаря переопределяемых параметров на основе оценки шума.
        Изменяются только те параметры, которые оставлены в значениях по умолчанию.
        """
        overrides = {}
        if sigma < 1e-6:
            return overrides

        if self.denoise_eps is None:
            overrides['denoise_eps'] = sigma ** 2 * 4
        if self.denoise_bm3d_sigma == 0.01:
            overrides['denoise_bm3d_sigma'] = sigma
        if self.denoise_nlm_h == 0.01:
            overrides['denoise_nlm_h'] = sigma
        if self.denoise_bilateral_sigma_r == 0.1:
            overrides['denoise_bilateral_sigma_r'] = sigma * 2

        if self.lmg_denoise_eps is None:
            overrides['lmg_denoise_eps'] = sigma ** 2 * 4
        if self.lmg_bm3d_sigma == 0.01:
            overrides['lmg_bm3d_sigma'] = sigma
        if self.lmg_nlm_h == 0.01:
            overrides['lmg_nlm_h'] = sigma
        if self.lmg_bilateral_sigma_r == 0.1:
            overrides['lmg_bilateral_sigma_r'] = sigma * 2

        if self.nonblind_method == 'ringing_removal':
            if self.lambda_tv == 0.001:
                overrides['lambda_tv'] = sigma * 0.5
            if self.lambda_l0 == 5e-4:
                overrides['lambda_l0'] = sigma * 0.025
            if self.weight_ring == 1.0:
                overrides['weight_ring'] = min(1.0, max(0.3, sigma * 50))

        return overrides

    def _apply_preprocess(self, img, params=None):
        """
        Применение пространственного фильтра к изображению перед началом
        итеративной слепой деконволюции.
        """
        method = self.preprocess
        if method is None or method == 'none':
            return img
        p = params if params is not None else (self.preprocess_params or {})

        if method == 'tv':
            from skimage.restoration import denoise_tv_chambolle
            weight = p.get('weight', 0.1)
            return denoise_tv_chambolle(img, weight=weight)

        elif method == 'nlm':
            from skimage.restoration import denoise_nl_means, estimate_sigma
            sigma_est = p.get('sigma', None)
            if sigma_est is None:
                sigma_est = float(np.mean(estimate_sigma(img)))
            h = p.get('h', 0.8 * sigma_est)
            return denoise_nl_means(
                img, h=h,
                patch_size=p.get('patch_size', 5),
                patch_distance=p.get('patch_distance', 6),
                fast_mode=True)

        elif method == 'bilateral':
            import cv2
            img_f32 = img.astype(np.float32)
            d = p.get('d', 5)
            sigma_color = p.get('sigma_color', 0.1)
            sigma_space = p.get('sigma_space', 5.0)
            return cv2.bilateralFilter(
                img_f32, d, sigma_color, sigma_space
            ).astype(np.float64)

        elif method == 'guided':
            from .utils import guided_filter
            r = p.get('radius', 4)
            eps = p.get('eps', 0.01)
            return guided_filter(img, img, r, eps)

        elif method == 'bm3d':
            import bm3d as bm3d_lib
            sigma = p.get('sigma', 0.05)
            return bm3d_lib.bm3d(img, sigma_psd=sigma)

        else:
            raise ValueError(
                f"Unknown preprocess='{method}'. "
                f"Choose from: 'tv', 'nlm', 'bilateral', 'guided', "
                f"'bm3d', 'none'")

    def _adapt_preprocess_params(self, sigma):
        """
        Автоматическое заполнение словаря параметров предварительной фильтрации
        на основе оцененной дисперсии шума.
        """
        pp = dict(self.preprocess_params) if self.preprocess_params else {}
        m = self.preprocess
        if sigma < 1e-6 or m is None or m == 'none':
            return pp

        if m == 'tv':
            pp.setdefault('weight', max(0.01, sigma * 2))
        elif m == 'nlm':
            pp.setdefault('h', max(0.001, sigma * 0.8))
            pp.setdefault('sigma', sigma)
        elif m == 'bilateral':
            pp.setdefault('sigma_color', max(0.01, sigma * 2))
            pp.setdefault('sigma_space', 5.0)
        elif m == 'guided':
            pp.setdefault('eps', max(0.001, sigma ** 2 * 4))
            pp.setdefault('radius', 4)
        elif m == 'bm3d':
            pp.setdefault('sigma', max(0.001, sigma))

        return pp

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('lambda_lmg', self.lambda_lmg),
            ('lambda_grad', self.lambda_grad),
            ('xk_iter', self.xk_iter),
            ('gamma_correct', self.gamma_correct),
            ('k_thresh', self.k_thresh),
            ('lambda_tv', self.lambda_tv),
            ('lambda_l0', self.lambda_l0),
            ('weight_ring', self.weight_ring),
            ('denoise_eps', self.denoise_eps),
            ('denoise_radius', self.denoise_radius),
            ('ensemble_denoise', self.ensemble_denoise),
            ('denoise_type', self.denoise_type),
            ('denoise_bilateral_sigma_s', self.denoise_bilateral_sigma_s),
            ('denoise_bilateral_sigma_r', self.denoise_bilateral_sigma_r),
            ('denoise_bm3d_sigma', self.denoise_bm3d_sigma),
            ('denoise_nlm_h', self.denoise_nlm_h),
            ('grad_smooth_sigma', self.grad_smooth_sigma),
            ('lmg_denoise_eps', self.lmg_denoise_eps),
            ('lmg_denoise_radius', self.lmg_denoise_radius),
            ('lmg_denoise_type', self.lmg_denoise_type),
            ('lmg_bilateral_sigma_s', self.lmg_bilateral_sigma_s),
            ('lmg_bilateral_sigma_r', self.lmg_bilateral_sigma_r),
            ('lmg_bm3d_sigma', self.lmg_bm3d_sigma),
            ('lmg_nlm_h', self.lmg_nlm_h),
            ('use_soft_threshold', self.use_soft_threshold),
            ('softmax_tau', self.softmax_tau),
            ('kernel_reg_weight', self.kernel_reg_weight),
            ('use_pmp_nonblind', self.use_pmp_nonblind),
            ('pmp_lambda', self.pmp_lambda),
            ('pmp_patch_r', self.pmp_patch_r),
            ('pmp_quantile', self.pmp_quantile),
            ('nonblind_method', self.nonblind_method),
            ('nonblind_params', self.nonblind_params),
            ('preprocess', self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_density_threshold', self.impulse_density_threshold),
            ('impulse_outlier_threshold', self.impulse_outlier_threshold),
            ('impulse_max_window', self.impulse_max_window),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params', self.screenot_params),
            ('act_preprocess', self.act_preprocess),
            ('act_params', self.act_params),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('histogram_eq', self.histogram_eq),
            ('histogram_eq_params', self.histogram_eq_params),
            ('kernel_eq', self.kernel_eq),
            ('kernel_eq_params', self.kernel_eq_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
