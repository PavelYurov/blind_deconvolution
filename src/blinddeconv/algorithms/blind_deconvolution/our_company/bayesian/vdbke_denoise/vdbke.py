"""
vdbke.py — Variational Dirichlet Blur Kernel Estimation.

Multi-scale blind deconvolution framework wrapper.

Ported from ``ms_ngm_dirichlet_ubc_img.m`` by X. Zhou et al.
Reference:
    X. Zhou, J. Mateos, F. Zhou, R. Molina, A.K. Katsaggelos:
    "Variational Dirichlet Blur Kernel Estimation",
    IEEE TIP, vol. 24, no. 12, pp. 5127-5139, 2015.
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
from .utils import (psf2otf, fspecial_gaussian, imresize,
                    rgb2gray, rgb2ycbcr, ycbcr2rgb,
                    gamma_correction, tikhonov_filter, wiener_filter,
                    pad_image, crop_image, edgetaper)
from .solvers import (center_kernel_img_space, ss_ngm_dirichlet_ubc_img,
                      firls_deb_ubc, ringing_artifacts_removal)
from .impulse_noise_estimation import detect_impulse_noise, adaptive_median_filter
from scipy.signal import convolve2d


class VDBKE(DeconvolutionAlgorithm):
    """Variational Dirichlet Blur Kernel Estimation (VDBKE).

    Multi-scale blind deconvolution followed by non-blind deconvolution.
    Ported from ``ms_ngm_dirichlet_ubc_img.m``.

    Reference
    ---------
    X. Zhou, J. Mateos, F. Zhou, R. Molina, A.K. Katsaggelos,
    "Variational Dirichlet Blur Kernel Estimation",
    IEEE TIP, vol. 24, no. 12, pp. 5127-5139, 2015.

    Parameters
    ----------
    **Общие параметры**

    kernel_size : tuple of int
        Размер оцениваемого ядра (высота, ширина). Должен быть нечётным.
    gamma_correct : float
        Гамма-коррекция входного изображения. 1.0 = выключена.
    use_ycbcr : bool
        Если True, не-слепая деконволюция применяется только к Y-каналу
        (YCbCr). Если False — поканально RGB.
    kernel_est_win : tuple or None
        Окно (r1, c1, r2, c2) для обрезки изображения перед оценкой
        ядра. None = всё изображение.
    verbose : bool
        Выводить промежуточную информацию (шаги, шум, параметры).

    **Параметры оценки ядра (kernel_*)** [★ = итерационные]

    kernel_lambda : float
        Регуляризация Дирихле ядра.
    kernel_max_iter : int  ★
        Макс. число итераций Ньютона для оценки ядра (на каждом
        альтернирующем шаге). HPO: [5, 50].
    kernel_back_alpha : float
        Начальный шаг backtracking line-search.
    kernel_back_beta : float
        Коэфф. уменьшения шага backtracking.
    kernel_lower_bound : float
        Нижняя граница параметров Дирихле (α₀).
    kernel_ng_min : float
        Мин. норма градиента для остановки Ньютона.
    kernel_cost_display : int
        Вывод стоимости ядра каждые N итераций (0 = выкл).
    kernel_mode : int
        Режим: 0 = стандартный, 1 = альтернативный.
    kernel_Laplacian_filter : ndarray or None
        Лапласиан-фильтр для приора на ядро. None = identity (Гауссов приор).
    kernel_lambda_C : float
        Дополнительная регуляризация (Companion term). 0 = выкл.

    **Параметры оценки изображения (img_*)** [★ = итерационные]

    img_lambda1 : float
        Основная регуляризация NGM (Non-Gaussian Model) для изображения.
    img_lambda_min : float
        Мин. значение λ при адаптивной регуляризации.
    img_lambda_max : float
        Макс. значение λ при адаптивной регуляризации.
    img_IF : float or None
        Inflation Factor для λ. None → √2.
    img_N1 : int  ★
        Внешние итерации оценки изображения. HPO: [5, 40].
    img_N2 : int  ★
        Внутренние итерации оценки изображения (CG). HPO: [1, 5].
    img_lambda_u : float
        Начальное значение λ_u.
    img_xv_iter : int  ★
        Итерации x-v подзадачи. HPO: [1, 3].
    img_cost_display : int
        Вывод стоимости изображения каждые N итераций (0 = выкл).

    **Альтернирующие итерации** [★ = итерационные]

    xk_iter : int  ★
        Число x↔k альтернирующих итераций на каждом уровне пирамиды.
        HPO: [5, 30].
    k_tol : float
        Допуск сходимости ядра (ранняя остановка).

    **Не-слепая деконволюция (FIRLS)** [★ = итерационные]

    firls_lambda : float
        Регуляризация FIRLS.
    firls_alpha : float
        Степень Lp-нормы (2/3 ≈ сверхразреженный приор).
    firls_out_iter : int  ★
        Внешние итерации FIRLS. HPO: [2, 10].
    firls_inner_iter : int  ★
        Внутренние итерации FIRLS (CG-шаги). HPO: [2, 8].

    **Пайплайн обработки шума** (все выключены по умолчанию)

    impulse_preprocess : str
        'auto' — обнаружение и удаление импульсного шума,
        'none' — выключено.
    impulse_params : dict or None
        {'density_threshold': 0.0005, 'outlier_threshold': 0.08,
         'max_window': 7}.
    noise_estimation : str
        Метод оценки уровня шума: 'chen', 'pca', 'none'.
    auto_params : bool or dict or None
        Авто-подбор img_lambda1 и firls_lambda из оценённого σ.
        - None / False — выключено.
        - True — включено с коэффициентами по умолчанию.
        - dict — включено с пользовательскими коэффициентами:
          {'k_img_lambda1': 200.0, 'k_firls_lambda': 200.0}.
          Формула: λ = k · σ².
    screenot_preprocess : str
        'auto' — ScreeNOT SVD-шумоподавление, 'none' — выключено.
        Взаимоисключающий с act_preprocess.
    screenot_params : dict or None
        {'k': 10, 'strategy': 'i', 'mode': 'full',
         'patch_size': 8, 'stride': 3}.
    act_preprocess : str
        'auto' — ACT curvelet-шумоподавление, 'none' — выключено.
        Взаимоисключающий с screenot_preprocess.
    act_params : dict or None
        {'noise_var': None, 'threshold_setting': 's'}.
    preprocess : str
        Пространственный денойзер перед пирамидой:
        'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'none'.
    preprocess_params : dict or None
        Параметры для выбранного денойзера (зависят от метода).
    noise_preprocess : str
        PSD-фильтрация: 'auto', 'notch', 'bandstop', 'none'.
    noise_preprocess_params : dict or None
        {'pch_size': 32, 'n_smooth': 100, 'peak_threshold': 100.0,
         'notch_radius': 3, 'freq_low': 0.3, 'freq_high': 0.5,
         'order': 2}.
    blind_denoise : str
        Денойзер внутри blind-цикла (перед grad для ядра):
        'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'none'.
    blind_denoise_params : dict or None
        Параметры для blind-денойзера.
    kernel_threshold : float
        Порог обнуления малых значений ядра (доля от max).
        0.0 = выключено. Типично 0.01–0.15.
    pre_nonblind : str
        Денойзер перед не-слепым шагом (Y-канал): те же варианты.
    pre_nonblind_params : dict or None
        Параметры для pre_nonblind денойзера.
    final_deconv : str
        Метод не-слепой деконволюции:
        'firls' (по умолчанию), 'blend' (FIRLS + ringing_removal),
        'ringing_removal', 'adaptive_lp', 'tikhonov', 'wiener', 'auto'.
        'auto' разрешается оркестратором (auto_mode='robust') в 'blend'
        на чистых/средних данных и 'ringing_removal' на сильном шуме.
    final_alpha : float
        Регуляризация для tikhonov/wiener.
    nb_params : dict or None
        Доп. параметры для final_deconv:
        - ringing_removal: {'lambda_tv': 1e-3, 'lambda_l0': 2e-3,
          'weight_ring': 1.0}
        - adaptive_lp: {'alpha': 0.8, 'two_stage': True}
        - blend: те же ключи + 'blend_weight' (вес ringing_removal,
          по умолчанию 0.5).
    auto_mode : str
        'off' (по умолчанию) — оркестратор выключен.
        'robust' — мягкое автоконфигурирование шумового пайплайна
        по оценённой σ. VDBKE довольно хорошо справляется с шумом
        сам по себе, поэтому оркестратор намеренно консервативный:
        не трогает screenot/act_preprocess/noise_preprocess, лишь
        мягко поднимает λ-регуляризации, включает дешёвый bilateral
        внутри блайнд-цикла и (для poisson-like шума) выбирает ACT
        для pre_nonblind.
    auto_mode_params : dict or None
        Параметры оркестратора:
        {'sigma_clean': 0.005, 'sigma_heavy': 0.05,
         'force_heavy_sigma': 0.012,
         'k_img_lambda1': 200.0, 'k_firls_lambda': 200.0,
         'k_alpha': 0.1, 'blend_weight': 0.5}.
    """

    def __init__(
        self,
        kernel_size=(25, 25),
        gamma_correct: float = 1.0,
        use_ycbcr: bool = True,
        kernel_est_win=None,
        # ── kernel estimation parameters ──
        kernel_lambda: float = 1e-6,
        kernel_max_iter: int = 20,
        kernel_back_alpha: float = 0.01,
        kernel_back_beta: float = 0.5,
        kernel_lower_bound: float = 1.0,
        kernel_ng_min: float = 1e-5,
        kernel_cost_display: int = 0,
        kernel_mode: int = 0,
        kernel_Laplacian_filter=None,
        kernel_lambda_C: float = 0.0,
        # ── image estimation parameters ──
        img_lambda1: float = 0.002,
        img_lambda_min: float = 0.01,
        img_lambda_max: float = 1.0,
        img_IF: float = None,
        img_N1: int = 20,
        img_N2: int = 2,
        img_lambda_u: float = 0.1,
        img_xv_iter: int = 1,
        img_cost_display: int = 0,
        # ── alternating iteration parameters ──
        xk_iter: int = 20,
        k_tol: float = 5e-4,
        # ── non-blind deconvolution (FIRLS) parameters ──
        firls_lambda: float = 0.002,
        firls_alpha: float = 2.0 / 3.0,
        firls_out_iter: int = 5,
        firls_inner_iter: int = 4,
        # ── noise pipeline (all disabled by default) ──
        verbose: bool = False,
        impulse_preprocess: str = 'none',
        impulse_params: dict = None,
        noise_estimation: str = 'none',
        auto_params=None,
        screenot_preprocess: str = 'none',
        screenot_params: dict = None,
        act_preprocess: str = 'none',
        act_params: dict = None,
        preprocess: str = 'none',
        preprocess_params: dict = None,
        noise_preprocess: str = 'none',
        noise_preprocess_params: dict = None,
        blind_denoise: str = 'none',
        blind_denoise_params: dict = None,
        kernel_threshold: float = 0.0,
        pre_nonblind: str = 'none',
        pre_nonblind_params: dict = None,
        final_deconv: str = 'firls',
        final_alpha: float = 0.001,
        nb_params: dict = None,
        auto_mode: str = 'off',
        auto_mode_params: dict = None,
    ):
        super().__init__(name='VDBKE')

        self.kernel_size = tuple(kernel_size) if not isinstance(kernel_size, tuple) else kernel_size
        self.gamma_correct = gamma_correct
        self.use_ycbcr = use_ycbcr
        self.kernel_est_win = kernel_est_win

        # Kernel estimation
        self.kernel_lambda = kernel_lambda
        self.kernel_max_iter = kernel_max_iter
        self.kernel_back_alpha = kernel_back_alpha
        self.kernel_back_beta = kernel_back_beta
        self.kernel_lower_bound = kernel_lower_bound
        self.kernel_ng_min = kernel_ng_min
        self.kernel_cost_display = kernel_cost_display
        self.kernel_mode = kernel_mode
        # Default Laplacian filter: identity (Gaussian prior on kernel)
        if kernel_Laplacian_filter is None:
            self.kernel_Laplacian_filter = np.array(
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float64)
        else:
            self.kernel_Laplacian_filter = np.asarray(kernel_Laplacian_filter, dtype=np.float64)
        self.kernel_lambda_C = kernel_lambda_C

        # Image estimation
        self.img_lambda1 = img_lambda1
        self.img_lambda_min = img_lambda_min
        self.img_lambda_max = img_lambda_max
        self.img_IF = img_IF if img_IF is not None else np.sqrt(2)
        self.img_N1 = img_N1
        self.img_N2 = img_N2
        self.img_lambda_u = img_lambda_u
        self.img_xv_iter = img_xv_iter
        self.img_cost_display = img_cost_display

        # Alternating
        self.xk_iter = xk_iter
        self.k_tol = k_tol

        # FIRLS
        self.firls_lambda = firls_lambda
        self.firls_alpha = firls_alpha
        self.firls_out_iter = firls_out_iter
        self.firls_inner_iter = firls_inner_iter

        # Noise pipeline
        self.verbose = verbose
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
        self.blind_denoise = blind_denoise
        self.blind_denoise_params = blind_denoise_params
        self.kernel_threshold = kernel_threshold
        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.final_deconv = final_deconv
        self.final_alpha = final_alpha
        self.nb_params = nb_params
        self.auto_mode = (auto_mode or 'off').lower()
        self.auto_mode_params = auto_mode_params

        # Snapshot of defaults for the robust orchestrator so soft
        # blending always starts from the values supplied at construction
        # time, not from values overwritten on a previous process() call.
        self._defaults_snapshot = {
            'img_lambda1': float(img_lambda1),
            'firls_lambda': float(firls_lambda),
            'final_alpha': float(final_alpha),
            'final_deconv': final_deconv,
            'preprocess': preprocess,
            'preprocess_params': preprocess_params,
            'blind_denoise': blind_denoise,
            'blind_denoise_params': blind_denoise_params,
            'pre_nonblind': pre_nonblind,
            'pre_nonblind_params': pre_nonblind_params,
            'impulse_preprocess': impulse_preprocess,
            'act_preprocess': act_preprocess,
            'screenot_preprocess': screenot_preprocess,
            'kernel_threshold': float(kernel_threshold),
        }

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    # ─────────────────────────────────────────────────────────────────────
    # process — main entry point  (← ms_ngm_dirichlet_ubc_img.m)
    # ─────────────────────────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # Convert to float64 [0, 1]
        y_in = image.astype(np.float64)
        if y_in.max() > 1.0:
            y_in /= 255.0

        # Gamma correction first (so non-blind sees the same colour space
        # as blind, mirroring LIP's single-image pipeline).
        y_in = y_in ** self.gamma_correct

        # Extract luminance: every denoising step in the chain is applied
        # to a single 2-D channel.  For colour input, we run on Y of YCbCr
        # and rebuild the cleaned RGB image afterwards — that way the
        # non-blind step receives the *same* denoised signal as the kernel
        # estimator (matching LIP semantics, where ``f`` carries every
        # preprocess effect into non-blind).
        has_color = (y_in.ndim == 3 and y_in.shape[2] == 3)
        if has_color:
            ycbcr_full = rgb2ycbcr(y_in)
            y = ycbcr_full[:, :, 0].copy()
        else:
            ycbcr_full = None
            y = y_in if y_in.ndim == 2 else y_in[:, :, 0].copy()

        # ── Robust mode: force protective flags BEFORE the pipeline ─────
        # impulse detection must run on the raw image (it uses the image
        # itself to find spikes), so we cannot enable it retroactively
        # from the orchestrator.  Force it here when robust mode is on
        # and the user did not pick anything explicit.
        if self.auto_mode == 'robust' and self.impulse_preprocess == 'none':
            self.impulse_preprocess = 'auto'

        # ── 4a. Impulse noise detection & removal ───────────────────────
        impulse_info = None
        if self.impulse_preprocess == 'auto':
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                y,
                density_threshold=ip.get('density_threshold', 0.0005),
                outlier_threshold=ip.get('outlier_threshold', 0.08),
            )
            if impulse_info['has_impulse']:
                y = adaptive_median_filter(
                    y, impulse_info['impulse_mask'],
                    max_window=ip.get('max_window', 7))
                if self.verbose:
                    print(f"[{self.name}] impulse noise removed "
                          f"(density={impulse_info['density']:.4f})")

        # ── 4b. Noise estimation ────────────────────────────────────────
        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(y)
            if self.verbose and noise_info is not None:
                print(f"[{self.name}] noise estimation "
                      f"({noise_info.get('method','?')}): "
                      f"σ={noise_info.get('sigma_norm', 0):.5f}")
        elif self.auto_mode == 'robust':
            # Orchestrator needs σ — force PCA if user left it 'none'.
            self.noise_estimation = 'pca'
            noise_info = self._estimate_noise(y)

        # ── 4b¼. Robust orchestrator (soft-weighted auto config) ────────
        orchestrator_info = None
        if self.auto_mode == 'robust':
            orchestrator_info = self._orchestrate_robust(noise_info)

        # ── 4b½. Auto-params from σ ────────────────────────────────────
        if self.auto_params and noise_info is not None:
            sigma_n = noise_info.get('sigma_norm', None)
            if sigma_n is not None and sigma_n > 0:
                ap = self.auto_params if isinstance(self.auto_params, dict) else {}
                k_lambda = ap.get('k_img_lambda1', 200.0)
                k_firls = ap.get('k_firls_lambda', 200.0)
                self.img_lambda1 = max(1e-5, k_lambda * sigma_n ** 2)
                self.firls_lambda = max(1e-5, k_firls * sigma_n ** 2)
                if self.verbose:
                    print(f"[{self.name}] auto_params(σ={sigma_n:.5f}): "
                          f"img_λ1={self.img_lambda1:.6f}, "
                          f"firls_λ={self.firls_lambda:.6f}")

        # ── 4c-1. ScreeNOT SVD denoising ───────────────────────────────
        screenot_info = None
        if self.screenot_preprocess == 'auto':
            from .screenot import screenot_denoise
            sp = self.screenot_params or {}
            y, screenot_info = screenot_denoise(
                y,
                k=sp.get('k', 10),
                strategy=sp.get('strategy', 'i'),
                mode=sp.get('mode', 'full'),
                patch_size=sp.get('patch_size', 8),
                stride=sp.get('stride', 3),
            )
            if self.verbose:
                print(f"[{self.name}] ScreeNOT denoising applied")

        # ── 4c-2. ACT curvelet denoising ───────────────────────────────
        act_info = None
        if self.act_preprocess == 'auto':
            if self.screenot_preprocess == 'auto':
                raise ValueError(
                    "screenot_preprocess and act_preprocess cannot both "
                    "be 'auto'. Choose one denoiser.")
            from .act_denoise import act_denoise
            ap = self.act_params or {}
            act_noise_var = ap.get('noise_var', None)
            if act_noise_var is None and noise_info is not None:
                act_noise_var = noise_info.get('sigma_norm', 0.0) ** 2
            y, act_info = act_denoise(
                y,
                noise_var=act_noise_var,
                threshold_setting=ap.get('threshold_setting', 's'),
            )
            if self.verbose:
                print(f"[{self.name}] ACT curvelet denoising applied")

        # ── 4c-3. PSD-based noise filtering ────────────────────────────
        psd_info = None
        if self.noise_preprocess != 'none':
            y, psd_info = self._apply_noise_preprocess(y)
            if self.verbose:
                print(f"[{self.name}] noise_preprocess="
                      f"'{self.noise_preprocess}' applied")

        # ── 4c-4. Pre-pyramid spatial denoising ────────────────────────
        if self.preprocess not in (None, 'none'):
            y = self._apply_denoise(
                y, self.preprocess, self.preprocess_params, noise_info)
            if self.verbose:
                print(f"[{self.name}] preprocess='{self.preprocess}' applied")

        # ── Rebuild cleaned full-resolution image for non-blind ─────────
        # ``yorig`` carries every step of the denoising chain; the
        # non-blind solver therefore sees the same signal that the
        # kernel was estimated from, instead of the raw blurry input
        # (this fixes the divergence with LIP).
        if has_color:
            ycbcr_clean = ycbcr_full.copy()
            ycbcr_clean[:, :, 0] = y
            yorig = ycbcr2rgb(ycbcr_clean)
        else:
            yorig = y.copy()

        # Optional crop only for kernel estimation (non-blind still uses
        # the full-resolution ``yorig``).
        if self.kernel_est_win is not None:
            w = self.kernel_est_win  # (r1, c1, r2, c2) 0-indexed
            y = y[w[0]:w[2], w[1]:w[3]]

        blur_size = self.kernel_size  # (ks1, ks2)

        # ── Determine kernel sizes at each scale ──
        # MATLAB: [max_ks, ind1] = max(opts.kernel_size)
        max_ks = max(blur_size)
        ind1 = 0 if blur_size[0] >= blur_size[1] else 1
        ind2 = 1 - ind1

        minsize = [0, 0]
        minsize[ind1] = max(3, 2 * ((max_ks - 1) // 64) + 1)
        temp = int(np.floor(blur_size[ind2] / blur_size[ind1] * minsize[ind1]))
        if temp % 2 == 0:
            temp += 1
        minsize[ind2] = max(temp, 3)

        if self.verbose:
            print(f'Kernel size at coarsest level is [{minsize[0]}, {minsize[1]}]')

        resize_step = np.sqrt(2)
        # Build ksize list for each scale
        ksize = []
        tmp = minsize[ind1]
        while tmp < max_ks:
            ks_entry = [0, 0]
            ks_entry[ind1] = tmp
            tmp2 = int(np.ceil(blur_size[ind2] / blur_size[ind1] * tmp))
            if tmp2 % 2 == 0:
                tmp2 += 1
            ks_entry[ind2] = max(tmp2, 3)
            ksize.append(tuple(ks_entry))

            tmp = int(np.ceil(tmp * resize_step))
            if tmp % 2 == 0:
                tmp += 1

        ksize.append(tuple(blur_size))
        num_scales = len(ksize)

        # Storage per scale
        ks = [None] * num_scales
        alphas = [None] * num_scales
        ls = [None] * num_scales

        lambda_C = self.kernel_lambda_C

        # ── Build blind_denoise callback ────────────────────────────────
        blind_denoise_fn = None
        if self.blind_denoise not in (None, 'none'):
            def blind_denoise_fn(u_arr):
                return self._apply_blind_denoise(u_arr, noise_info)

        # ── Multi-scale loop ──
        for s in range(num_scales):
            k1, k2 = ksize[s]

            if s == 0:
                # Coarsest level: initialise kernel as Gaussian
                Gsigma = 1.0 if max_ks > 50 else 0.5
                ks[s] = fspecial_gaussian((k1, k2), Gsigma)
                alphas[s] = ks[s] + self.kernel_lower_bound
            else:
                # Up-sample kernel from previous level
                tmp_k = ks[s - 1].copy()
                tmp_k[tmp_k < 0] = 0
                tmp_k /= tmp_k.sum()
                ks[s] = imresize(tmp_k, (k1, k2), 'bilinear')
                alphas[s] = imresize(alphas[s - 1], (k1, k2), 'bilinear')
                ks[s][ks[s] < 0] = 0
                ks[s] /= ks[s].sum()

            # Image size at this level
            r = int(np.floor(y.shape[0] * k1 / blur_size[0]))
            c = int(np.floor(y.shape[1] * k2 / blur_size[1]))
            if s == num_scales - 1:
                r, c = y.shape[0], y.shape[1]

            if self.verbose:
                print(f'Processing scale {s + 1}/{num_scales}; '
                      f'kernel size {k1}x{k2}; image size {r}x{c}')

            # Resize y to current scale
            ys = imresize(y, (r, c), 'bilinear')

            if s == 0:
                ls[s] = ys.copy()
            else:
                ls[s] = imresize(ls[s - 1], (r, c), 'bilinear')

            # Lambda_C schedule
            if s == num_scales - 1:
                cur_lambda_C = lambda_C
            else:
                cur_lambda_C = (lambda_C * ksize[s][0] * ksize[s][1]
                                / (ksize[-1][0] * ksize[-1][1]))

            # Centre the kernel
            ls[s], ks[s], shift_kernel = center_kernel_img_space(
                ls[s], ks[s], verbose=self.verbose)
            alphas[s] = np.maximum(
                convolve2d(alphas[s], shift_kernel, 'same'),
                self.kernel_lower_bound)

            # Build parameter dicts for this scale
            kernel_pars = {
                'lambda': self.kernel_lambda,
                'max_iter': self.kernel_max_iter,
                'back_alpha': self.kernel_back_alpha,
                'back_beta': self.kernel_back_beta,
                'lower_bound': self.kernel_lower_bound,
                'ng_min': self.kernel_ng_min,
                'cost_display': self.kernel_cost_display,
                'mode': self.kernel_mode,
                'Laplacian_filter': self.kernel_Laplacian_filter,
                'lambda_C': cur_lambda_C,
                'alpha0': alphas[s],
            }

            img_pars = {
                'lambda1': self.img_lambda1,
                'lambda_min': self.img_lambda_min,
                'lambda_max': self.img_lambda_max,
                'IF': self.img_IF,
                'N1': self.img_N1,
                'N2': self.img_N2,
                'lambda_u': self.img_lambda_u,
                'xv_iter': self.img_xv_iter,
                'cost_display': self.img_cost_display,
                'x0': ls[s].copy(),
            }

            pars = {
                'xk_iter': self.xk_iter,
                'img_pars': img_pars,
                'kernel_pars': kernel_pars,
                'k_tol': self.k_tol,
            }

            # Single-scale alternating estimation
            ls[s], ks[s], alphas[s] = ss_ngm_dirichlet_ubc_img(
                ys, ls[s], ks[s], alphas[s], pars,
                blind_denoise_fn=blind_denoise_fn,
                verbose=self.verbose)

            # At finest scale, extract final kernel
            if s == num_scales - 1:
                kernel = alphas[s] - self.kernel_lower_bound
                kernel = kernel / kernel.sum()

        # ── Kernel thresholding ─────────────────────────────────────────
        if self.kernel_threshold > 0:
            kernel[kernel < self.kernel_threshold * kernel.max()] = 0.0
            k_sum = kernel.sum()
            if k_sum > 0:
                kernel /= k_sum
            if self.verbose:
                print(f"[{self.name}] kernel thresholded "
                      f"(thr={self.kernel_threshold:.3f})")

        # ── Pre-nonblind denoising ──────────────────────────────────────
        if self.pre_nonblind not in (None, 'none'):
            yorig = self._apply_pre_nonblind(yorig, noise_info)
            if self.verbose:
                print(f"[{self.name}] pre_nonblind='{self.pre_nonblind}' "
                      f"applied to yorig")

        # ── Non-blind deconvolution ──
        if self.final_deconv == 'ringing_removal':
            nbp = self.nb_params or {}
            deblur = self._nonblind_channel_dispatch(
                yorig, kernel,
                lambda ch, k: ringing_artifacts_removal(
                    ch, k,
                    lambda_tv=nbp.get('lambda_tv', 1e-3),
                    lambda_l0=nbp.get('lambda_l0', 2e-3),
                    weight_ring=nbp.get('weight_ring', 1.0),
                ))

        elif self.final_deconv == 'adaptive_lp':
            from .non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = noise_info.get('sigma_norm', None) if noise_info else None
            deblur = self._nonblind_channel_dispatch(
                yorig, kernel,
                lambda ch, k: adaptive_lp_deconv(
                    ch, k,
                    alpha=nbp.get('alpha', 0.8),
                    sigma_n=sigma_n,
                    two_stage=nbp.get('two_stage', True),
                ))

        elif self.final_deconv in ('tikhonov', 'wiener'):
            MK, NK = kernel.shape
            def _freq_deconv(ch, k):
                ch_pad = pad_image(ch, (MK, NK))
                ch_pad = edgetaper(ch_pad, k)
                if self.final_deconv == 'tikhonov':
                    res = tikhonov_filter(ch_pad, k, alpha=self.final_alpha)
                else:
                    res = wiener_filter(ch_pad, k, noise_snr=self.final_alpha)
                return crop_image(res, ch.shape[:2], (MK, NK))
            deblur = self._nonblind_channel_dispatch(
                yorig, kernel, _freq_deconv)

        else:  # default: 'firls' or 'blend'
            firls_opts = {
                'lambda': self.firls_lambda,
                'alpha': self.firls_alpha,
                'out_iter': self.firls_out_iter,
                'inner_iter': self.firls_inner_iter,
            }
            def _firls(ch, k):
                x_fov, _, _ = firls_deb_ubc(ch, k, firls_opts,
                                             verbose=self.verbose)
                return x_fov

            if self.final_deconv == 'blend':
                # Weighted average of FIRLS (sharp, sparse-prior detail)
                # and ringing_removal (smoother, fewer inverse-filter
                # ripples).  Both already cope with mild noise, so the
                # blend keeps detail while suppressing residual ringing —
                # a safer alternative than tikhonov+rr because tikhonov
                # is much more noise-sensitive.
                nbp = self.nb_params or {}
                blend_w = float(nbp.get('blend_weight', 0.5))  # weight of RR
                def _blend(ch, k):
                    u_firls = _firls(ch, k)
                    u_rr = ringing_artifacts_removal(
                        ch, k,
                        lambda_tv=nbp.get('lambda_tv', 1e-3),
                        lambda_l0=nbp.get('lambda_l0', 2e-3),
                        weight_ring=nbp.get('weight_ring', 1.0),
                    )
                    h = min(u_firls.shape[0], u_rr.shape[0])
                    w_ = min(u_firls.shape[1], u_rr.shape[1])
                    return ((1.0 - blend_w) * u_firls[:h, :w_]
                            + blend_w * u_rr[:h, :w_])
                deblur = self._nonblind_channel_dispatch(yorig, kernel, _blend)
            else:  # 'firls'
                deblur = self._nonblind_channel_dispatch(
                    yorig, kernel, _firls)

        deblur = np.clip(deblur, 0.0, 1.0)

        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'gamma_correct': self.gamma_correct,
            'kernel_lambda': self.kernel_lambda,
            'img_lambda1': self.img_lambda1,
            'firls_lambda': self.firls_lambda,
            'impulse_preprocess': self.impulse_preprocess,
            'impulse_info': {k_: v for k_, v in (impulse_info or {}).items()
                            if k_ != 'impulse_mask'} if impulse_info else None,
            'noise_estimation': self.noise_estimation,
            'noise_info': noise_info,
            'auto_params': self.auto_params,
            'screenot_preprocess': self.screenot_preprocess,
            'screenot_info': screenot_info,
            'act_preprocess': self.act_preprocess,
            'act_info': act_info,
            'noise_preprocess': self.noise_preprocess,
            'psd_info': psd_info,
            'preprocess': self.preprocess,
            'blind_denoise': self.blind_denoise,
            'kernel_threshold': self.kernel_threshold,
            'pre_nonblind': self.pre_nonblind,
            'final_deconv': self.final_deconv,
            'auto_mode': self.auto_mode,
            'orchestrator': orchestrator_info,
            'time': time.time() - start_time,
        }

        # Output: int16 [0, 255], kernel
        x_final = np.clip(deblur * 255.0, 0, 255).astype(np.int16)
        return x_final, kernel

    # ─────────────────────────────────────────────────────────────────────
    # Channel dispatch for non-blind deconvolution
    # ─────────────────────────────────────────────────────────────────────
    def _nonblind_channel_dispatch(self, img, kernel, deconv_fn):
        """Apply a non-blind deconvolution function to each channel.

        Uses YCbCr (luminance only) or per-channel RGB depending on
        ``self.use_ycbcr``.
        """
        if self.use_ycbcr:
            if img.ndim == 3 and img.shape[2] == 3:
                ycbcr = rgb2ycbcr(img)
            else:
                ycbcr = img.copy()
            if ycbcr.ndim == 3 and ycbcr.shape[2] == 3:
                ycbcr[:, :, 0] = deconv_fn(ycbcr[:, :, 0], kernel)
                return np.clip(ycbcr2rgb(ycbcr), 0.0, 1.0)
            return np.clip(deconv_fn(ycbcr, kernel), 0.0, 1.0)
        else:
            if img.ndim == 3:
                out = img.copy()
                for j in range(img.shape[2]):
                    out[:, :, j] = deconv_fn(img[:, :, j], kernel)
                return np.clip(out, 0.0, 1.0)
            return np.clip(deconv_fn(img, kernel), 0.0, 1.0)

    # ─────────────────────────────────────────────────────────────────────
    # PSD-based noise preprocessing
    # ─────────────────────────────────────────────────────────────────────
    def _apply_noise_preprocess(self, yg):
        from .noise_psd_analysis import (
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
                yg_out = notch_filter(
                    yg, peaks,
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

    # ─────────────────────────────────────────────────────────────────────
    # Guided filter (box-filter variant, He et al. 2013)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _guided_filter(I, p, r, eps):
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

    # ─────────────────────────────────────────────────────────────────────
    # Universal denoiser dispatch
    # ─────────────────────────────────────────────────────────────────────
    def _apply_denoise(self, img, method, params, noise_info):
        """Apply a spatial denoiser to a single-channel image [0, 1].

        Supported methods: 'tv', 'nlm', 'bilateral', 'guided', 'bm3d'.
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
            from .act_denoise import act_denoise
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
                f"'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'act', 'none'")

    # ─────────────────────────────────────────────────────────────────────
    # Blind-loop denoiser (x_fov before kernel step)
    # ─────────────────────────────────────────────────────────────────────
    def _apply_blind_denoise(self, x, noise_info):
        p = dict(self.blind_denoise_params or {})
        if self.blind_denoise == 'guided':
            p.setdefault('radius', 2)
        return self._apply_denoise(x, self.blind_denoise, p, noise_info)

    # ─────────────────────────────────────────────────────────────────────
    # Pre-nonblind denoiser
    # ─────────────────────────────────────────────────────────────────────
    def _apply_pre_nonblind(self, img, noise_info):
        """Apply denoiser to image before non-blind step.

        For color images: denoise luminance channel only (YCbCr).
        """
        if img.ndim == 3 and img.shape[2] == 3:
            ycbcr = rgb2ycbcr(img)
            ycbcr[:, :, 0] = self._apply_denoise(
                ycbcr[:, :, 0], self.pre_nonblind,
                self.pre_nonblind_params, noise_info)
            return ycbcr2rgb(ycbcr)
        return self._apply_denoise(
            img, self.pre_nonblind, self.pre_nonblind_params, noise_info)

    # ─────────────────────────────────────────────────────────────────────
    # Noise estimation helper
    # ─────────────────────────────────────────────────────────────────────
    def _estimate_noise(self, yg):
        """Estimate noise level from a grayscale image [0, 1].

        Parameters
        ----------
        yg : ndarray — input image (2-D or 3-D; converted to gray internally).

        Returns
        -------
        dict with keys 'method', 'sigma_norm', 'sigma', or None.
        """
        img = yg
        if img.ndim == 3 and img.shape[2] == 3:
            img = rgb2gray(img)

        if self.noise_estimation == 'chen':
            from .chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(img)
            return {'method': 'chen', 'sigma_norm': sigma,
                    'sigma': sigma * 255.0}
        elif self.noise_estimation == 'pca':
            from .pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(img)
            result['method'] = 'pca'
            return result
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Robust orchestrator (soft-weighted auto config for the noise pipeline)
    # ─────────────────────────────────────────────────────────────────────
    def _orchestrate_robust(self, noise_info):
        """Conservative robust auto-config for VDBKE.

        VDBKE is already noise-tolerant on its own (Dirichlet prior +
        adaptive λ during image estimation).  Empirically, a *minimal*
        intervention reproduces the best hand-tuned config the author
        found:

            impulse_preprocess='auto'  (forced earlier in process())
            act_preprocess='auto'      (curvelet denoising on Y)
            pre_nonblind='guided'      (light edge-preserving filter)
            kernel_threshold ≈ 0.05    (kill spurious tails)

        Aggressive choices that *kill the kernel* (collapse it to a
        single pixel) were observed and removed:

            ✗ ``blind_denoise='bilateral'`` — over-smooths u inside the
              alternating loop, the gradient w.r.t. k vanishes, and α
              concentrates on one pixel.
            ✗ σ-scaling of img_lambda1 / firls_lambda with k≈200 — at
              σ≈0.05 this raises img_lambda1 ~250× over its default and
              suppresses the image step entirely; the kernel update sees
              a zero-content image and collapses.
            ✗ Adding extra ``preprocess`` on top of ``act_preprocess`` —
              double smoothing destroys the high-frequency content the
              kernel estimator relies on.

        Policy:
            • Clean regime (σ ≤ σ_clean) — keep all user defaults.
              ``final_deconv='auto'`` → 'blend'.
            • Heavy regime (σ  > σ_clean) — only enable the protective
              flags above (when the user left them at 'none' / 0.0),
              and route ``final_deconv='auto'`` to 'blend' (medium) or
              'ringing_removal' (heavy).  Do NOT change λ-regularisation
              or add blind_denoise.

        Always resets mutable fields from the __init__ snapshot so
        repeated process() calls are deterministic.
        """
        snap = self._defaults_snapshot
        amp = dict(self.auto_mode_params or {})

        # ── 1) Reset from snapshot ───────────────────────────────────
        self.img_lambda1 = snap['img_lambda1']
        self.firls_lambda = snap['firls_lambda']
        self.final_alpha = snap['final_alpha']
        self.preprocess = snap['preprocess']
        self.preprocess_params = snap['preprocess_params']
        self.blind_denoise = snap['blind_denoise']
        self.blind_denoise_params = snap['blind_denoise_params']
        self.pre_nonblind = snap['pre_nonblind']
        self.pre_nonblind_params = snap['pre_nonblind_params']
        self.act_preprocess = snap['act_preprocess']
        self.kernel_threshold = snap['kernel_threshold']

        # ── 2) σ + thresholds ────────────────────────────────────────
        sigma = 0.0
        if noise_info is not None:
            sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)

        sigma_clean = float(amp.get('sigma_clean', 0.005))
        sigma_heavy = float(amp.get('sigma_heavy', 0.05))
        blend_weight_clean = float(amp.get('blend_weight', 0.5))

        force_heavy = False
        nt = (noise_info or {}).get('noise_type', None)
        force_heavy_sigma = float(amp.get('force_heavy_sigma', 0.012))
        if nt in ('poisson', 'poisson_gaussian') and sigma >= force_heavy_sigma:
            force_heavy = True

        # ── 3) Clean branch — keep user defaults. ────────────────────
        if sigma <= sigma_clean and not force_heavy:
            regime = 'clean'
            if snap['final_deconv'] == 'auto':
                self.final_deconv = 'blend'
                if self.nb_params is None:
                    self.nb_params = {'blend_weight': blend_weight_clean}
                elif 'blend_weight' not in self.nb_params:
                    self.nb_params = dict(self.nb_params)
                    self.nb_params['blend_weight'] = blend_weight_clean
            if self.verbose:
                print(f"[{self.name}] orchestrator(σ={sigma:.5f}, clean): "
                      f"defaults kept, final_deconv={self.final_deconv}")
            return {
                'sigma_norm': sigma, 'w': 0.0, 'regime': regime,
                'final_deconv': self.final_deconv,
                'act_preprocess': self.act_preprocess,
                'pre_nonblind': self.pre_nonblind,
                'kernel_threshold': self.kernel_threshold,
            }

        # ── 4) Heavy/medium branch ───────────────────────────────────
        w = 1.0 if sigma >= sigma_heavy else (
            (sigma - sigma_clean) / max(sigma_heavy - sigma_clean, 1e-9))
        w = float(np.clip(w, 0.0, 1.0))
        regime = 'heavy' if w > 0.85 else 'medium'

        noise_type = (noise_info or {}).get('noise_type', 'gaussian')
        poisson_like = noise_type in ('poisson', 'poisson_gaussian',
                                      'unknown')

        # (a) ACT curvelet denoising — only enable if neither it nor
        #     screenot is already chosen by the user.
        if (snap['act_preprocess'] in (None, 'none')
                and snap['screenot_preprocess'] in (None, 'none')):
            self.act_preprocess = 'auto'

        # (b) pre_nonblind — light guided filter unless user picked
        #     something else.  Guided is fast, edge-preserving, and the
        #     author confirmed it gives the best visual quality on heavy
        #     noise.  For poisson-like noise we still use guided — ACT
        #     is already doing the heavy lifting in (a).
        if snap['pre_nonblind'] in (None, 'none'):
            self.pre_nonblind = 'guided'
            self.pre_nonblind_params = {
                'radius': 4,
                'eps': float(max(sigma, 0.01)) ** 2 * 4.0,
            }

        # (c) kernel_threshold — kill spurious tails on noisy ψ.  Only
        #     bump if the user left it at 0 (off).
        if snap['kernel_threshold'] <= 0.0:
            self.kernel_threshold = float(amp.get('kernel_threshold', 0.05))

        # (d) final_deconv routing.  FIRLS handles moderate noise well,
        #     but its sparse prior amplifies kernel-mismatch ringing on
        #     very noisy data — RR damps that explicitly.
        if snap['final_deconv'] == 'auto':
            self.final_deconv = 'ringing_removal' if w >= 0.85 else 'blend'

        # NOTE: img_lambda1, firls_lambda, blind_denoise, preprocess,
        # noise_preprocess are deliberately NOT touched — see docstring.

        info = {
            'sigma_norm': sigma, 'w': w, 'regime': regime,
            'noise_type': noise_type,
            'poisson_like': bool(poisson_like),
            'act_preprocess': self.act_preprocess,
            'pre_nonblind': self.pre_nonblind,
            'kernel_threshold': self.kernel_threshold,
            'final_deconv': self.final_deconv,
        }
        if self.verbose:
            print(f"[{self.name}] orchestrator(σ={sigma:.5f}, w={w:.2f}, "
                  f"regime={regime}, type={noise_type}): "
                  f"act_pre={self.act_preprocess}, "
                  f"pre_nb={self.pre_nonblind}, "
                  f"k_thr={self.kernel_threshold:.3f}, "
                  f"final={self.final_deconv}")
        return info

    # ─────────────────────────────────────────────────────────────────────
    # Framework interface methods
    # ─────────────────────────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('gamma_correct', self.gamma_correct),
            ('use_ycbcr', self.use_ycbcr),
            ('kernel_lambda', self.kernel_lambda),
            ('kernel_max_iter', self.kernel_max_iter),
            ('kernel_lower_bound', self.kernel_lower_bound),
            ('kernel_lambda_C', self.kernel_lambda_C),
            ('img_lambda1', self.img_lambda1),
            ('img_lambda_min', self.img_lambda_min),
            ('img_lambda_max', self.img_lambda_max),
            ('xk_iter', self.xk_iter),
            ('k_tol', self.k_tol),
            ('firls_lambda', self.firls_lambda),
            ('firls_alpha', self.firls_alpha),
            ('firls_out_iter', self.firls_out_iter),
            ('firls_inner_iter', self.firls_inner_iter),
            ('verbose', self.verbose),
            ('impulse_preprocess', self.impulse_preprocess),
            ('impulse_params', self.impulse_params),
            ('noise_estimation', self.noise_estimation),
            ('auto_params', self.auto_params),
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
            ('kernel_threshold', self.kernel_threshold),
            ('pre_nonblind', self.pre_nonblind),
            ('pre_nonblind_params', self.pre_nonblind_params),
            ('final_deconv', self.final_deconv),
            ('final_alpha', self.final_alpha),
            ('nb_params', self.nb_params),
            ('auto_mode', self.auto_mode),
            ('auto_mode_params', self.auto_mode_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_size':
                    self.kernel_size = tuple(value)
                else:
                    setattr(self, key, value)
                # Keep the orchestrator's default-snapshot in sync with
                # parameters the user updates after construction.
                if key in self._defaults_snapshot:
                    if key in ('img_lambda1', 'firls_lambda', 'final_alpha'):
                        self._defaults_snapshot[key] = float(value)
                    else:
                        self._defaults_snapshot[key] = value

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
