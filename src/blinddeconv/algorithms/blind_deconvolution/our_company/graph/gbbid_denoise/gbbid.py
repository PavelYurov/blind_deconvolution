"""
gbbid.py

Слепая деконволюция изображений с использованием графового априорного 
распределения (Graph-Based RGTV Prior, GBBID).

Основано на методе:
    Y. Bai, G. Cheung, X. Liu, W. Gao:
    "Graph-Based Blind Image Deblurring From a Single Photograph",
    IEEE Transactions on Image Processing, vol. 28, no. 3, pp. 1404-1418, 2019.

Модифицированная версия включает расширенный конвейер предобработки и 
подавления шума (импульсного, периодического, пуассоновского и гауссовского) 
с автоматическим оркестратором параметров.
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
    bid_rgtv_c2f_cg,
    Deconvolution_FHLP,
    apply_denoiser,
    deblurring_adm_aniso,
    L0Restoration,
    ringing_artifacts_removal,
)
from blinddeconv.algorithms.mod_denoise.non_blind import adaptive_lp_deconv
from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from .utils import opt_fft_size, wrap_boundary_liu


class GBBID(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции на основе графовой регуляризации RGTV 
    с интегрированным конвейером обработки шума.

    Параметры
    ---------
    k_estimate_size : int, по умолчанию 69
        Ожидаемый пространственный размер оцениваемого ядра размытия.
    border : int, по умолчанию 20
        Количество краевых пикселей, обрезаемых перед началом оценки ядра.
    preprocess : str
        Алгоритм предварительного шумоподавления перед построением пирамиды:
        'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'none'. По умолчанию 'tv'.
    preprocess_params : dict или None
        Параметры для метода предварительного шумоподавления.
    pre_kernel : str
        Алгоритм шумоподавления перед шагом оценки ядра. По умолчанию 'none'.
    pre_kernel_params : dict или None
        Параметры для метода шумоподавления перед оценкой ядра.
    nonblind_method : str
        Метод финальной неслепой деконволюции:
        - 'fhlp' : быстрое гиперлапласовское априорное распределение (по умолчанию).
        - 'tv_adm' : TV-регуляризация через метод ADM/Split Bregman.
        - 'l0' : использование L0-нормы градиентов.
        - 'ringing_removal' : комбинация TV, L0 и билатерального фильтра.
        - 'adaptive_lp' : пространственно-зависимая Lp-регуляризация.
    nonblind_params : dict или None
        Специфичные параметры для выбранного метода неслепой деконволюции.
    lambda_fhlp : float, по умолчанию 2e3
        Вес члена верности данных для метода FHLP.
    alpha_fhlp : float, по умолчанию 0.5
        Экспонента гиперлапласиана для метода FHLP.
    edgetaper_iters : int, по умолчанию 4
        Количество проходов сглаживания краев (edgetaper) для метода FHLP.
    noise_estimation : str
        Метод оценки дисперсии шума: 'chen', 'pyatykh' или 'none'.
    auto_params : bool
        Если True и оценка шума включена, алгоритм автоматически адаптирует 
        незаданные параметры методов шумоподавления на основе уровня шума.
    noise_preprocess : str
        Спектральный фильтр для периодического шума: 'auto', 'prewhiten', 
        'notch', 'bandstop', 'none'.
    noise_preprocess_params : dict или None
        Параметры спектральной фильтрации шума.
    impulse_preprocess : str
        Метод обработки импульсного шума ('auto', 'none').
    impulse_params : dict или None
        Параметры для обнаружения и фильтрации импульсного шума.
    screenot_preprocess : str
        Предварительное шумоподавление методом ScreeNOT ('auto', 'none').
    screenot_params : dict или None
        Параметры алгоритма ScreeNOT.
    act_preprocess : str
        Предварительное шумоподавление методом Adaptive Curvelet Thresholding. 
        Несовместимо с screenot_preprocess='auto'.
    act_params : dict или None
        Параметры для метода ACT.
    pre_nonblind : str
        Метод шумоподавления, применяемый перед финальным неслепым шагом. 
        Рекомендуется 'bm3d' или 'act' для коррелированного шума.
    pre_nonblind_params : dict или None
        Параметры шумоподавления перед неслепым шагом.
    auto_mode : str
        Глобальный оркестратор шумоподавления ('robust' или 'off'). В режиме 
        'robust' выполняется автоматическая настройка всего конвейера с 
        плавным переходом параметров в зависимости от уровня шума sigma.
    auto_mode_params : dict или None
        Настройки порогов и ограничений для глобального оркестратора.
    """

    def __init__(
        self,
        k_estimate_size: int = 69,
        border: int = 20,
        preprocess: str = 'tv',
        preprocess_params: dict = None,
        pre_kernel: str = 'none',
        pre_kernel_params: dict = None,
        nonblind_method: str = 'fhlp',
        nonblind_params: dict = None,
        lambda_fhlp: float = 2e3,
        alpha_fhlp: float = 0.5,
        edgetaper_iters: int = 4,
        noise_estimation: str = 'none',
        auto_params: bool = False,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        impulse_preprocess: str = 'auto',
        impulse_params: dict = None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
    ):
        super().__init__(name='GBBID')

        self.k_estimate_size = k_estimate_size
        self.border = border
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params
        self.pre_kernel = pre_kernel
        self.pre_kernel_params = pre_kernel_params
        self.nonblind_method = nonblind_method
        self.nonblind_params = nonblind_params
        self.lambda_fhlp = lambda_fhlp
        self.alpha_fhlp = alpha_fhlp
        self.edgetaper_iters = edgetaper_iters
        self.noise_estimation = noise_estimation
        self.auto_params = auto_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        self._defaults_snapshot = {
            'lambda_fhlp': float(lambda_fhlp),
            'alpha_fhlp': float(alpha_fhlp),
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'pre_kernel': pre_kernel,
            'pre_kernel_params': pre_kernel_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'nonblind_method': nonblind_method,
            'nonblind_params': nonblind_params,
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Запуск алгоритма слепой деконволюции.

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
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                yg,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_window=ip.get('outlier_window', 5),
                outlier_threshold=ip.get('outlier_threshold', 0.15),
            )
            if impulse_info['has_impulse']:
                yg = adaptive_median_filter(
                    yg, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))
                if y.ndim == 3:
                    for ch in range(y.shape[2]):
                        ch_info = detect_impulse_noise(
                            y[:, :, ch],
                            density_threshold=ip.get('density_threshold', 0.005),
                            outlier_window=ip.get('outlier_window', 5),
                            outlier_threshold=ip.get('outlier_threshold', 0.15),
                        )
                        if ch_info['has_impulse']:
                            y[:, :, ch] = adaptive_median_filter(
                                y[:, :, ch], ch_info['impulse_mask'],
                                max_window=ip.get('max_window', 7))
                else:
                    y = yg.copy()

        screenot_info = None
        if self.screenot_preprocess == 'auto':
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

        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(yg)
        elif self.auto_mode == 'robust':
            self.noise_estimation = 'pyatykh'
            noise_info = self._estimate_noise(yg)

        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info, image=yg)

        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
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
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = act_denoise(
                        y[:, :, ch],
                        noise_var=act_noise_var,
                        threshold_setting=ap.get('threshold_setting', 's'),
                    )
            else:
                y = yg.copy()

        b = self.border
        if b > 0:
            yg_cropped = yg[b:-b, b:-b]
        else:
            yg_cropped = yg

        psd_info = None
        if self.noise_preprocess != 'none':
            yg, psd_info = self._apply_noise_preprocess(yg)
            if b > 0:
                yg_cropped = yg[b:-b, b:-b]
            else:
                yg_cropped = yg

        eff_pp = self.preprocess_params
        eff_pkp = self.pre_kernel_params
        eff_nbp = self.nonblind_params
        if self.auto_params and noise_info is not None:
            sigma = noise_info.get('sigma_norm', 0.0)
            eff_pp, eff_pkp, eff_nbp = self._compute_adaptive_params(
                sigma, eff_pp, eff_pkp, eff_nbp)

        kernel, _skeleton = bid_rgtv_c2f_cg(
            yg_cropped, self.k_estimate_size,
            show_intermediate=False,
            preprocess=self.preprocess,
            preprocess_params=eff_pp,
            pre_kernel=self.pre_kernel,
            pre_kernel_params=eff_pkp,
            iteration_callback=self._callback,
        )

        if self.pre_nonblind not in (None, 'none'):
            y = self._apply_pre_nonblind(y, noise_info)

        nb = self.nonblind_method
        nb_p = eff_nbp or {}

        if y.ndim == 3:
            Latent = np.zeros_like(y)
            for ch in range(y.shape[2]):
                Latent[:, :, ch] = self._nonblind_single(
                    y[:, :, ch], kernel, nb, nb_p)
        else:
            Latent = self._nonblind_single(y, kernel, nb, nb_p)

        Latent = np.clip(Latent, 0.0, 1.0)

        self.hyperparams = {
            'k_estimate_size': self.k_estimate_size,
            'border': self.border,
            'preprocess': self.preprocess,
            'preprocess_params': self.preprocess_params,
            'pre_kernel': self.pre_kernel,
            'pre_kernel_params': self.pre_kernel_params,
            'nonblind_method': self.nonblind_method,
            'nonblind_params': self.nonblind_params,
            'lambda_fhlp': self.lambda_fhlp,
            'alpha_fhlp': self.alpha_fhlp,
            'edgetaper_iters': self.edgetaper_iters,
            'noise_estimation': self.noise_estimation,
            'auto_params': self.auto_params,
            'noise_preprocess': self.noise_preprocess,
            'noise_preprocess_params': self.noise_preprocess_params,
            'impulse_preprocess': self.impulse_preprocess,
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
            'noise_info': noise_info,
            'psd_info': {k: v for k, v in (psd_info or {}).items()
                         if k != 'psd_2d'} if psd_info else None,
            'effective_preprocess_params': eff_pp,
            'effective_pre_kernel_params': eff_pkp,
            'effective_nonblind_params': eff_nbp,
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'time': time.time() - start_time,
        }

        x_final = Latent * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def _nonblind_single(self, y_ch, kernel, method, params):
        """Выполнение неслепой деконволюции для одного цветового канала."""
        if method == 'fhlp':
            return Deconvolution_FHLP(
                y_ch, kernel,
                lambda_val=params.get('lambda_val', self.lambda_fhlp),
                alpha=params.get('alpha', self.alpha_fhlp),
                edgetaper_iters=params.get('edgetaper_iters',
                                           self.edgetaper_iters))
        elif method == 'tv_adm':
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
        elif method == 'ringing_removal':
            return ringing_artifacts_removal(
                y_ch, kernel,
                lambda_tv=params.get('lambda_tv', 2e-3),
                lambda_l0=params.get('lambda_l0', 2e-3),
                weight_ring=params.get('weight_ring', 0.5))
        elif method == 'adaptive_lp':
            return adaptive_lp_deconv(
                y_ch, kernel,
                alpha=params.get('alpha', 0.8),
                sigma_n=params.get('sigma_n', None),
                two_stage=params.get('two_stage', True))
        else:
            raise ValueError(
                f"Unknown nonblind_method='{method}'. "
                f"Choose from: 'fhlp', 'tv_adm', 'l0', 'ringing_removal', "
                f"'adaptive_lp'")

    def _orchestrate_robust(self, noise_info, image=None):
        """
        Автоматическая конфигурация параметров конвейера на основе оценки шума.

        Стратегия работы:
        - Режим чистого сигнала (sigma <= sigma_clean): параметры по умолчанию 
          остаются без изменений. Метод неслепой деконволюции переводится 
          в режим 'fhlp', если был задан как 'auto'.
        - Режим сильного шума (sigma > sigma_clean): выполняется плавное 
          смешивание параметров. Применяется маршрутизация в зависимости от 
          типа шума: пуассоновский шум направляется на фильтры ACT или VST+BM3D, 
          гауссовский — на билатеральную фильтрацию или BM3D. Для предотвращения 
          потери информативных краев агрессивное сглаживание внутри цикла 
          оценки ядра ограничивается.
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        self.lambda_fhlp = snap['lambda_fhlp']
        self.alpha_fhlp = snap['alpha_fhlp']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.pre_kernel = snap['pre_kernel']
        self.pre_kernel_params = snap['pre_kernel_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.nonblind_method = snap['nonblind_method']
        self.nonblind_params = snap['nonblind_params']

        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.01))

        nt = (noise_info or {}).get('noise_type', None)
        force_heavy = (nt in ('poisson', 'poisson_gaussian')
                       and sigma >= force_heavy_sigma)

        if sigma <= sigma_clean and not force_heavy:
            if snap['nonblind_method'] == 'auto':
                self.nonblind_method = 'fhlp'
            if self.verbose if hasattr(self, 'verbose') else False:
                pass
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': 'clean',
                'nonblind_method': self.nonblind_method,
                'preprocess': self.preprocess,
                'pre_kernel': self.pre_kernel,
                'pre_nonblind': self.pre_nonblind,
                'lambda_fhlp': float(self.lambda_fhlp),
                'alpha_fhlp': float(self.alpha_fhlp),
            }

        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / (sigma_heavy - sigma_clean))
        regime = 'heavy' if w > 0.95 else 'medium'

        noise_type = nt or 'gaussian'
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        k_lambda_fhlp = float(amp.get('k_lambda_fhlp', 50.0))
        k_alpha_fhlp = float(amp.get('k_alpha_fhlp', 0.6))
        lam_noisy = float(np.clip(k_lambda_fhlp / max(sigma, 1e-6),
                                  100.0, 1e5))
        alpha_noisy = max(0.5, k_alpha_fhlp)

        self.lambda_fhlp = (1.0 - w) * snap['lambda_fhlp'] + w * lam_noisy
        self.alpha_fhlp = (1.0 - w) * snap['alpha_fhlp'] + w * alpha_noisy

        poisson_denoiser = str(amp.get('poisson_denoiser', 'act')).lower()
        if poisson_denoiser not in ('act', 'vst_bm3d'):
            poisson_denoiser = 'act'

        gauss_act_preprocess = bool(amp.get('act_preprocess_gaussian', False))
        gauss_act_pre_nonblind = bool(amp.get('act_pre_nonblind_gaussian', False))


        if poisson_like:
            if poisson_denoiser == 'vst_bm3d':
                self.preprocess = 'vst_bm3d'
                self.preprocess_params = {'noise_info': noise_info}
            else:
                self.preprocess = 'bilateral'
                self.preprocess_params = {
                    'sigma_color': float(max(2.0 * sigma, 0.02)),
                    'sigma_spatial': 3.0,
                }
        elif gauss_act_preprocess:
            self.preprocess = 'act'
            self.preprocess_params = {
                'noise_var': float(sigma ** 2),
                'threshold_setting': 's',
            }
        elif w < 0.6:
            self.preprocess = 'bilateral'
            self.preprocess_params = {
                'sigma_color': float(max(sigma, 0.01)),
                'sigma_spatial': 3.0,
            }
        else:
            self.preprocess = 'bm3d'
            self.preprocess_params = {'sigma_psd': float(sigma)}

        if (not poisson_like) and w >= 0.5:
            self.pre_kernel = 'bilateral'
            self.pre_kernel_params = {
                'sigma_color': float(max(0.5 * sigma, 0.005)),
                'sigma_spatial': 2.0,
            }
        if poisson_like:
            if poisson_denoiser == 'vst_bm3d':
                self.pre_nonblind = 'vst_bm3d'
                self.pre_nonblind_params = {'noise_info': noise_info}
            else:
                self.pre_nonblind = 'act'
                self.pre_nonblind_params = {
                    'noise_var': float(sigma ** 2),
                    'threshold_setting': 's',
                }
        elif gauss_act_pre_nonblind:
            self.pre_nonblind = 'act'
            self.pre_nonblind_params = {
                'noise_var': float(sigma ** 2),
                'threshold_setting': 's',
            }
        elif w < 0.6:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma_psd': float(max(sigma, 0.01))}
        else:
            self.pre_nonblind = 'bm3d'
            self.pre_nonblind_params = {'sigma_psd': float(sigma)}

        nb_auto_heavy = amp.get('nonblind_auto_heavy', 'ringing_removal')
        if snap['nonblind_method'] == 'auto':
            self.nonblind_method = nb_auto_heavy

        info = {
            'sigma_norm': sigma, 'w': float(w), 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'poisson_denoiser': poisson_denoiser,
            'act_preprocess_gaussian': gauss_act_preprocess,
            'act_pre_nonblind_gaussian': gauss_act_pre_nonblind,
            'act_noise_var_type': 'pyatykh_scalar',
            'lambda_fhlp': float(self.lambda_fhlp),
            'alpha_fhlp': float(self.alpha_fhlp),
            'preprocess': self.preprocess,
            'pre_kernel': self.pre_kernel,
            'pre_nonblind': self.pre_nonblind,
            'nonblind_method': self.nonblind_method,
        }
        return info

    def _apply_noise_preprocess(self, yg):
        """
        Анализ спектральной плотности мощности шума и применение 
        соответствующей спектральной фильтрации.
        """
        from blinddeconv.algorithms.mod_denoise.noise_psd_analysis import (
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

    def _apply_pre_nonblind(self, y, noise_info):
        """
        Шумоподавление перед неслепой деконволюцией для устранения 
        коррелированных шумовых компонент. Неслепые методы предполагают белый шум, 
        наличие цветного шума приводит к структурным артефактам.
        """
        method = self.pre_nonblind
        params = dict(self.pre_nonblind_params or {})

        sigma = None
        if noise_info is not None:
            sigma = noise_info.get('sigma_norm', None)

        if method == 'act':
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
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

        if method == 'vst_bm3d':
            from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise
            ni = params.get('noise_info', None)
            a = params.get('a', None)
            b = params.get('b', None)
            sig = params.get('sigma', sigma)
            if y.ndim == 3:
                for ch in range(y.shape[2]):
                    y[:, :, ch], _ = vst_bm3d_denoise(
                        y[:, :, ch], noise_info=ni,
                        a=a, b=b, sigma=sig)
            else:
                y, _ = vst_bm3d_denoise(y, noise_info=ni,
                                        a=a, b=b, sigma=sig)
            return y

        if method == 'bm3d' and 'sigma_psd' not in params and sigma is not None:
            params['sigma_psd'] = sigma
        elif method == 'nlm' and 'h' not in params and sigma is not None:
            params['h'] = 0.8 * sigma
        elif method == 'bilateral' and 'sigma_color' not in params and sigma is not None:
            params['sigma_color'] = sigma
        elif method == 'guided' and 'eps' not in params and sigma is not None:
            params['eps'] = sigma ** 2 * 4

        if y.ndim == 3:
            for ch in range(y.shape[2]):
                y[:, :, ch] = apply_denoiser(
                    y[:, :, ch], method, **params)
        else:
            y = apply_denoiser(y, method, **params)
        return y

    def _estimate_noise(self, yg):
        """Оценка уровня шума по полутоновому изображению."""
        if self.noise_estimation == 'chen':
            from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pyatykh':
            from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result['method'] = 'pyatykh'
            return result
        return None

    def _compute_adaptive_params(self, sigma, pp, pkp, nbp):
        """
        Адаптация параметров обработки на основе оцененного уровня шума.
        Заполняются только те параметры, которые не были заданы пользователем.
        """
        if sigma < 1e-6:
            return pp, pkp, nbp

        if pp is None:
            if self.preprocess == 'nlm':
                pp = {'h': sigma}
            elif self.preprocess == 'tv':
                pp = {'mu': sigma * 0.5, 'gamma': 0.1, 'max_it': 10}
            elif self.preprocess == 'bilateral':
                pp = {'sigma_spatial': 3, 'sigma_range': sigma * 2}
            elif self.preprocess == 'guided':
                pp = {'radius': 2, 'eps': sigma ** 2 * 4}

        if pkp is None:
            if self.pre_kernel == 'guided':
                pkp = {'radius': 2, 'eps': sigma ** 2 * 4}
            elif self.pre_kernel == 'nlm':
                pkp = {'h': sigma}
            elif self.pre_kernel == 'bilateral':
                pkp = {'sigma_spatial': 2, 'sigma_range': sigma * 2}

        if nbp is None:
            if self.nonblind_method == 'ringing_removal':
                nbp = {
                    'lambda_tv': sigma * 0.5,
                    'lambda_l0': sigma * 0.025,
                    'weight_ring': min(1.0, max(0.3, sigma * 50)),
                }
            elif self.nonblind_method == 'tv_adm':
                nbp = {'lambda_tv': sigma * 0.5}
            elif self.nonblind_method == 'l0':
                nbp = {'lambda_grad': sigma * 0.1}
            elif self.nonblind_method == 'adaptive_lp':
                nbp = {'alpha': 0.8, 'sigma_n': sigma, 'two_stage': True}

        return pp, pkp, nbp

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('k_estimate_size', self.k_estimate_size),
            ('border', self.border),
            ('preprocess', self.preprocess),
            ('preprocess_params', self.preprocess_params),
            ('pre_kernel', self.pre_kernel),
            ('pre_kernel_params', self.pre_kernel_params),
            ('nonblind_method', self.nonblind_method),
            ('nonblind_params', self.nonblind_params),
            ('lambda_fhlp', self.lambda_fhlp),
            ('alpha_fhlp', self.alpha_fhlp),
            ('edgetaper_iters', self.edgetaper_iters),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
            ('noise_preprocess', self.noise_preprocess),
            ('noise_preprocess_params', self.noise_preprocess_params),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('screenot_preprocess', self.screenot_preprocess),
            ('screenot_params', self.screenot_params),
            ('act_preprocess', self.act_preprocess),
            ('act_params', self.act_params),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in self._defaults_snapshot:
                    self._defaults_snapshot[key] = (
                        float(value) if key in ('lambda_fhlp', 'alpha_fhlp')
                        else value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
