"""
Denoise.py

Универсальный класс-обертка для интеграции алгоритмов шумоподавления 
в инфраструктуру слепой деконволюции.

Конвейер обработки:
1. Нормализация входного изображения к диапазону float64 [0, 1].
2. Усечение пространственных размеров изображения до нечетных значений.
3. Оценка уровня шума с использованием методов на основе PCA или вейвлет-анализа.
4. Применение выбранного алгоритма шумоподавления с параметрами, адаптированными 
   на основе предварительной оценки шума.
5. Возврат отфильтрованного изображения (в формате int16 [0, 255]) и 
   точечного ядра (эквивалента дельта-функции Дирака).

Поддерживаемые алгоритмы шумоподавления:
- bm3d : блочное сопоставление и трехмерная фильтрация (Block Matching 3D).
- guided : фильтр с направляющим изображением (Guided Filter).
- bilateral : билатеральная фильтрация.
- nlm : нелокальное усреднение (Non-Local Means).
- tv : метод полной вариации (модель Шамболя).
- vst+bm3d : преобразование, стабилизирующее дисперсию, совместно с BM3D 
  для подавления пуассоновско-гауссовского шума.
- act : адаптивное пороговое ограничение в кривлет-области.
- median : медианная фильтрация с переключением (для импульсного шума).

Методы оценки шума:
- chen : метод на основе собственных значений в вейвлет-области (ICCV 2015).
- pca : метод главных компонент с применением стабилизации дисперсии (TIP 2013).
- none : оценка отключена, используются параметры по умолчанию.

Литература:
    1. Chen G., Zhu F., Heng P.A., "An Efficient Statistical Method for 
       Image Noise Level Estimation", ICCV 2015.
    2. Pyatykh S., Hesser J., Zheng L., "Image Noise Level Estimation 
       by Principal Component Analysis", IEEE Trans. Image Process., 2013.
    3. Dabov K., Foi A., Katkovnik V., Egiazarian K., "Image Denoising 
       by Sparse 3D Transform-Domain Collaborative Filtering", IEEE Trans. 
       Image Process., 2007.
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

def make_size_odd(image):
    """
    Усечение пространственных размеров изображения до нечетных значений.
    """
    h, w = image.shape[:2]
    h = h if h % 2 == 1 else h - 1
    w = w if w % 2 == 1 else w - 1
    return image[:h, :w]

class DenoiseWrapper(DeconvolutionAlgorithm):
    """
    Алгоритм-обертка для применения методов шумоподавления в рамках 
    инфраструктуры слепой деконволюции.

    Применяет одиночный фильтр с параметрами, которые могут вычисляться 
    адаптивно на основе оценки уровня шума. Возвращаемая функция рассеяния 
    точки представляет собой единичный импульс в центре.

    Параметры
    ---------
    method : str
        Идентификатор метода шумоподавления. Допустимые значения: 'bm3d', 
        'guided', 'bilateral', 'nlm', 'tv', 'vst+bm3d', 'act', 'median'.
    noise_estimation : str
        Идентификатор метода оценки шума: 'chen', 'pca' или 'none'. 
        При значении 'none' используются параметры фильтра по умолчанию.
    denoiser_params : dict, опционально
        Словарь специфических параметров для выбранного метода шумоподавления.
    noise_estimation_params : dict, опционально
        Словарь параметров для алгоритма оценки шума.
    verbose : bool, по умолчанию False
        Флаг вывода диагностической информации в консоль.
    """

    def __init__(
        self,
        method: str = 'bm3d',
        noise_estimation: str = 'chen',
        denoiser_params: dict = None,
        noise_estimation_params: dict = None,
        verbose: bool = False,
    ):
        super().__init__(name='DenoiseWrapper')

        valid_methods = {'bm3d', 'guided', 'bilateral', 'nlm', 'tv',
                         'vst+bm3d', 'act', 'median'}
        method_lower = str(method).lower().strip()
        if method_lower not in valid_methods:
            raise ValueError(
                f"Denoiser method='{method}' not supported. "
                f"Choose from: {sorted(valid_methods)}")
        self.method = method_lower

        valid_noise_est = {'chen', 'pca', 'none'}
        noise_est_lower = str(noise_estimation).lower().strip()
        if noise_est_lower not in valid_noise_est:
            raise ValueError(
                f"noise_estimation='{noise_estimation}' not supported. "
                f"Choose from: {sorted(valid_noise_est)}")
        self.noise_estimation = noise_est_lower

        self.denoiser_params = dict(denoiser_params or {})
        self.noise_estimation_params = dict(noise_estimation_params or {})
        self.verbose = bool(verbose)

        self.history: Dict[str, list] = {}
        self.hyperparams: Dict[str, Any] = {}


    def _estimate_noise(self, image):
        """
        Оценка уровня шума по изображению.

        Возвращает
        ----------
        dict или None
            Словарь с ключами:
            - 'method' : идентификатор использованного метода оценки.
            - 'sigma_norm' : оценка СКО шума в масштабе [0, 1].
            - 'sigma_pix' : оценка СКО шума в масштабе [0, 255].
        """
        if self.noise_estimation == 'none':
            return None

        try:
            if self.noise_estimation == 'chen':
                
                from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level

                sigma_norm = estimate_noise_level(
                    image,
                    pch_size=self.noise_estimation_params.get('pch_size', 8),
                )
                return {
                    'method': 'chen',
                    'sigma_norm': float(sigma_norm),
                    'sigma_pix': float(sigma_norm * 255.0),
                }

            elif self.noise_estimation == 'pca':
                
                from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params

                result = estimate_noise_params(image)
                result['method'] = 'pca'
                if 'sigma_norm' not in result:
                    result['sigma_norm'] = result.get('sigma', 0.0)
                return result

            return None

        except Exception as e:
            if self.verbose:
                print(f"[{self.name}] Warning: noise estimation failed: {e}")
            return None


    def _apply_tv(self, image, noise_info):
        """Подавление шума на основе модели полной вариации (Chambolle)."""
        from skimage.restoration import denoise_tv_chambolle

        p = dict(self.denoiser_params)
        sigma = (noise_info.get('sigma_norm') if noise_info else None)

        weight = p.get('weight', None)
        if weight is None:
            weight = max(0.01, sigma * 2) if sigma else 0.1

        kwargs = dict(weight=weight, eps=p.get('eps', 0.002))
        try:
            return denoise_tv_chambolle(image, **kwargs,
                                        max_num_iter=p.get('max_num_iter', 200))
        except TypeError:
            return denoise_tv_chambolle(image, **kwargs,
                                        n_iter_max=p.get('max_num_iter', 200))

    def _apply_nlm(self, image, noise_info):
        """Нелокальное усреднение (Non-Local Means)."""
        from skimage.restoration import denoise_nl_means, estimate_sigma

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        if sigma is None:
            sigma = float(np.mean(estimate_sigma(image)))

        h = p.get('h', 0.8 * sigma)
        patch_size = p.get('patch_size', 5)
        patch_distance = p.get('patch_distance', 7)

        return denoise_nl_means(
            image, h=h, patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=p.get('fast_mode', True)
        )

    def _apply_bilateral(self, image, noise_info):
        """Билатеральная фильтрация."""
        import cv2

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        d = p.get('d', 5)
        sigma_color = p.get('sigma_color', sigma if sigma else 0.1)
        sigma_space = p.get('sigma_space', 5.0)

        result = cv2.bilateralFilter(
            image.astype(np.float32), d, float(sigma_color),
            float(sigma_space)
        )
        return result.astype(np.float64)

    def _apply_guided(self, image, noise_info):
        """Фильтрация на основе направляющего изображения (Guided Filter)."""
        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        radius = p.get('radius', 4)
        eps = p.get('eps', sigma ** 2 * 4 if sigma else 0.01)

        return self._guided_filter_impl(image, image, radius, eps)

    def _guided_filter_impl(self, I, p, r, eps):
        """
        Реализация алгоритма Guided Filter.

        Параметры
        ---------
        I : ndarray
            Направляющее изображение.
        p : ndarray
            Исходное изображение для фильтрации.
        r : int
            Радиус окна фильтрации.
        eps : float
            Параметр регуляризации.
            
        Возвращает
        ----------
        q : ndarray
            Отфильтрованное изображение.
        """
        from scipy.ndimage import uniform_filter

        I = I.astype(np.float64)
        p = p.astype(np.float64)
        H, W = I.shape

        mean_I = uniform_filter(I, size=2*r+1, mode='reflect')
        mean_p = uniform_filter(p, size=2*r+1, mode='reflect')
        mean_Ip = uniform_filter(I * p, size=2*r+1, mode='reflect')
        mean_II = uniform_filter(I * I, size=2*r+1, mode='reflect')

        var_I = mean_II - mean_I ** 2
        cov_Ip = mean_Ip - mean_I * mean_p

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = uniform_filter(a, size=2*r+1, mode='reflect')
        mean_b = uniform_filter(b, size=2*r+1, mode='reflect')

        q = mean_a * I + mean_b
        return np.clip(q, 0, 1)

    def _apply_bm3d(self, image, noise_info):
        """Блочное сопоставление и трехмерная фильтрация (BM3D)."""
        import bm3d

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        sigma_psd = p.get('sigma', sigma if sigma else 0.05)

        stage_arg = p.get('stage_arg', bm3d.BM3DStages.ALL_STAGES)
        if isinstance(stage_arg, str):
            _map = {
                'all': bm3d.BM3DStages.ALL_STAGES,
                'ht': bm3d.BM3DStages.HARD_THRESHOLDING,
                'hard': bm3d.BM3DStages.HARD_THRESHOLDING,
                'wiener': bm3d.BM3DStages.WIENER_FILTERING,
            }
            stage_arg = _map.get(stage_arg.lower(), bm3d.BM3DStages.ALL_STAGES)

        return bm3d.bm3d(image, sigma_psd=sigma_psd, stage_arg=stage_arg)

    def _apply_vst_bm3d(self, image, noise_info):
        """
        Фильтрация BM3D в области стабилизации дисперсии (VST) для 
        подавления пуассоновско-гауссовского шума.
        """
        from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise

        p = dict(self.denoiser_params)

        result, _ = vst_bm3d_denoise(
            image,
            noise_info=noise_info,
            sigma=p.get('sigma', None),
            a=p.get('a', None),
            b=p.get('b', None),
            stage_arg=p.get('stage_arg', None),
            verbose=self.verbose,
        )
        return result

    def _apply_median(self, image, noise_info):
        """
        Адаптивный медианный фильтр с логическим переключением для 
        подавления импульсного шума.

        Для каждого пикселя вычисляется медиана в заданной локальной окрестности. 
        Пиксель классифицируется как шумовой выброс, если его абсолютное отклонение 
        от локальной медианы превышает порог, зависящий от локального 
        среднеквадратичного отклонения. Замене подлежат только пиксели, 
        классифицированные как выбросы, остальные остаются без изменений.

        Параметры из словаря denoiser_params
        ------------------------------------
        kernel_size : int, по умолчанию 3
            Размер стороны квадратного окна фильтра.
        threshold : float, по умолчанию 0.3
            Относительный порог отклонения для выявления выбросов. Большие 
            значения делают фильтр менее агрессивным.
        """
        from scipy.ndimage import median_filter, uniform_filter

        p = dict(self.denoiser_params)
        ksize     = int(p.get('kernel_size', 3))
        threshold = float(p.get('threshold', 0.3))

        med = median_filter(image, size=ksize, mode='reflect')

        local_mean  = uniform_filter(image, size=ksize, mode='reflect')
        local_sq    = uniform_filter(image ** 2, size=ksize, mode='reflect')
        local_var   = np.clip(local_sq - local_mean ** 2, 0, None)
        local_sigma = np.sqrt(local_var)

        global_sigma = float(np.std(image)) or 1e-6
        scale = np.where(local_sigma > 1e-6, local_sigma, global_sigma)

        mask = np.abs(image - med) > threshold * scale

        result = image.copy()
        result[mask] = med[mask]
        return result

    def _apply_act(self, image, noise_info):
        """Адаптивное пороговое ограничение в кривлет-области (ACT)."""
        from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise

        p = dict(self.denoiser_params)
        sigma = noise_info.get('sigma_norm') if noise_info else None

        noise_var = p.get('noise_var', None)
        if noise_var is None and sigma is not None:
            noise_var = sigma ** 2

        result, _ = act_denoise(
            image,
            noise_var=noise_var,
            threshold_setting=p.get('threshold_setting', 's'),
        )
        return result

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Выполнение процедуры шумоподавления для входного изображения.

        Параметры
        ---------
        image : ndarray
            Входное изображение размерности (H, W) или (H, W, 3).

        Возвращает
        ----------
        x_final : ndarray
            Отфильтрованное изображение в формате int16 [0, 255].
        h : ndarray
            Функция рассеяния точки (матрица 1x1 с единицей).
        """
        start_time = time.time()

        f = image.astype(np.float64)
        if f.max() > 1.0:
            f /= 255.0

        if f.ndim == 3 and f.shape[2] == 3:
            f = np.mean(f, axis=2)
        elif f.ndim > 2:
            raise ValueError(f"Expected 2D or 3D RGB image, got shape={f.shape}")

        M_orig, N_orig = f.shape

        f = make_size_odd(f)
        M, N = f.shape

        noise_info = None
        if self.noise_estimation != 'none':
            noise_info = self._estimate_noise(f)
            if self.verbose and noise_info:
                print(
                    f"[{self.name}] Noise estimation ({noise_info['method']}): "
                    f"σ_norm={noise_info.get('sigma_norm', 0):.5f}")

        if self.verbose:
            print(f"[{self.name}] Applying {self.method} denoiser...")

        if self.method == 'tv':
            x_denoised = self._apply_tv(f, noise_info)
        elif self.method == 'nlm':
            x_denoised = self._apply_nlm(f, noise_info)
        elif self.method == 'bilateral':
            x_denoised = self._apply_bilateral(f, noise_info)
        elif self.method == 'guided':
            x_denoised = self._apply_guided(f, noise_info)
        elif self.method == 'bm3d':
            x_denoised = self._apply_bm3d(f, noise_info)
        elif self.method == 'vst+bm3d':
            x_denoised = self._apply_vst_bm3d(f, noise_info)
        elif self.method == 'act':
            x_denoised = self._apply_act(f, noise_info)
        elif self.method == 'median':
            x_denoised = self._apply_median(f, noise_info)
        else:
            raise RuntimeError(f"Unknown denoiser: {self.method}")

        x_denoised = np.clip(x_denoised, 0, 1)

        x_final = x_denoised * 255.0
        x_final = np.round(x_final).astype(np.int16)

        h = np.zeros((1, 1), dtype=np.int16)
        h[0, 0] = 1

        self.hyperparams = {
            'time': time.time() - start_time,
            'method': self.method,
            'noise_estimation': self.noise_estimation,
            'input_shape': (M_orig, N_orig),
            'output_shape': x_final.shape,
        }
        if noise_info is not None:
            self.hyperparams['noise_sigma_norm'] = \
                noise_info.get('sigma_norm', None)
            self.hyperparams['noise_sigma_pix'] = \
                noise_info.get('sigma_pix', None)

        if self.verbose:
            print(f"[{self.name}] Done in {self.hyperparams['time']:.3f}s")

        return x_final, h


    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('method', self.method),
            ('noise_estimation', self.noise_estimation),
            ('denoiser_params', self.denoiser_params),
            ('noise_estimation_params', self.noise_estimation_params),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if key == 'method':
                valid_methods = {'bm3d', 'guided', 'bilateral', 'nlm', 'tv',
                                 'vst+bm3d', 'act'}
                if str(value).lower() not in valid_methods:
                    raise ValueError(f"Invalid method: {value}")
                self.method = str(value).lower()
            elif key == 'noise_estimation':
                valid_noise = {'chen', 'pca', 'none'}
                if str(value).lower() not in valid_noise:
                    raise ValueError(f"Invalid noise_estimation: {value}")
                self.noise_estimation = str(value).lower()
            elif key == 'denoiser_params':
                self.denoiser_params = dict(value or {})
            elif key == 'noise_estimation_params':
                self.noise_estimation_params = dict(value or {})
            elif key == 'verbose':
                self.verbose = bool(value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams
