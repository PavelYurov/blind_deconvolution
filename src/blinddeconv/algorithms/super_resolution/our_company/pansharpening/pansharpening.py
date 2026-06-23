"""
pansharpening.py

Сверхразрешение одиночного изображения на основе вариационного байесовского 
паншарпенинга с супергауссовскими и TV априорными распределениями.

Содержание алгоритма:
    1. Нормализация входного изображения в диапазон [0, 1].
    2. Построение псевдо-панхроматического (PAN) направляющего изображения
       путем бикубической интерполяции до целевого высокого разрешения.
    3. Выполнение вариационного байесовского вывода (решатели restSGME_Sens или TVME_Sens).
    4. Возврат восстановленного изображения и пустого (фиктивного) ядра размытия.

Литература:
[1] Pérez-Bueno, F., Vega, M., Mateos, J., Molina, R., & Katsaggelos, A. K.
    (2020). Variational Bayesian Pansharpening with Super-Gaussian Sparse
    Image Priors. Sensors, 20(18), 5308.
"""

import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np
from scipy.ndimage import zoom

# --- Интеграция с базовым классом ---
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

from .utils import get_psf, getfilters, getkappa, image_normalize, image_denormalize
from .solvers import (
    restoreSAR, alfaTVpvini, alfaSGlogvini, alfaSGlpvini,
    restSGME_Sens, TVME_Sens,
)


class SGPansharpening(DeconvolutionAlgorithm):
    """
    Байесовское сверхразрешение одиночного изображения с использованием 
    механизма паншарпенинга.

    Входное LR-изображение (в градациях серого) рассматривается как 
    одноканальное мультиспектральное наблюдение. Бикубически увеличенная 
    копия служит псевдо-панхроматическим направляющим изображением.
    Возвращает HR-изображение и фиктивное ядро размытия (алгоритм 
    не оценивает функцию рассеяния точки в слепом режиме).

    Параметры алгоритма
    -------------------
    ratio : int
        Коэффициент масштабирования (сверхразрешения). По умолчанию 2.
    prior_type : str
        Тип априорного распределения: 'log', 'lp' или 'tv'. По умолчанию 'log'.
    filtersetname : str
        Набор фильтров: 'fohv' или 'fo' (только для SG априорных распределений). 
        По умолчанию 'fohv'.
    lp_p : float
        Экспонента для априорного распределения 'lp'. По умолчанию 0.8.
    sensor : str
        Тип сенсора (определяет ФРТ): 'none' (прямоугольное усреднение), 
        'gaussian' и т.д. По умолчанию 'none'.
    eps_map : float
        Порог сходимости внешнего цикла. По умолчанию 1e-4.
    itmax_map : int
        Максимальное количество внешних итераций. По умолчанию 50.
    itmin_map : int
        Минимальное количество внешних итераций. По умолчанию 2.
    eps_y : float
        Порог сходимости для метода сопряженных градиентов (CG). По умолчанию 1e-7.
    itmax_y : int
        Максимальное количество итераций CG. По умолчанию 30.
    gamma_gamma : float
        Уверенность гипер-априорного распределения для параметра связи PAN-изображения. 
        По умолчанию 0.0.
    verbose : bool
        Флаг вывода отладочной информации. По умолчанию False.
    """

    def __init__(
        self,
        ratio: int = 2,
        prior_type: str = 'log',
        filtersetname: str = 'fohv',
        lp_p: float = 0.8,
        sensor: str = 'none',
        eps_map: float = 1e-4,
        itmax_map: int = 50,
        itmin_map: int = 2,
        eps_y: float = 1e-7,
        itmax_y: int = 30,
        gamma_gamma: float = 0.0,
        verbose: bool = False,
    ):
        super().__init__(name='SG-Pansharpening')

        self.ratio = ratio
        self.prior_type = prior_type
        self.filtersetname = filtersetname
        self.lp_p = lp_p
        self.sensor = sensor
        self.eps_map = eps_map
        self.itmax_map = itmax_map
        self.itmin_map = itmin_map
        self.eps_y = eps_y
        self.itmax_y = itmax_y
        self.gamma_gamma = gamma_gamma
        self.verbose = verbose

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной процесс реконструкции сверхразрешения."""
        start_time = time.time()

        # --- 1. Нормализация входных данных ---
        y = image.astype(np.float64)
        if y.max() > 1.0:
            y /= 255.0

        if y.ndim == 3 and y.shape[2] == 1:
            y = y[:, :, 0]
        elif y.ndim == 3 and y.shape[2] == 3:
            y = 0.2989 * y[:, :, 0] + 0.5870 * y[:, :, 1] + 0.1140 * y[:, :, 2]

        lr_h, lr_w = y.shape[:2]
        hr_h, hr_w = lr_h * self.ratio, lr_w * self.ratio
        nbands = 1

        # --- 2. Построение псевдо-PAN и подготовка наблюдений ---
        Y_LR = y[:, :, np.newaxis] 
        x_pan = zoom(y, self.ratio, order=3)
        x_pan = np.clip(x_pan, 0.0, 1.0)

        Y_norm, x_norm, facY, facx = image_normalize(Y_LR, x_pan)

        lam = np.array([1.0])
        psf = get_psf(self.ratio, self.sensor)

        # --- 3. Первичная оценка гиперпараметров ---
        _, alpha_sar, beta_sar = restoreSAR(Y_norm[:, :, 0], np.array([[1.0]]))

        if self.prior_type == 'tv':
            alpha_init = alfaTVpvini(x_norm, 2)
            alpha_mode = np.array([alpha_init])
        elif self.prior_type == 'log':
            alpha_init = alfaSGlogvini(Y_norm, self.filtersetname)
            alpha_mode = alpha_init  
        elif self.prior_type == 'lp':
            alpha_init = alfaSGlpvini(Y_norm, self.lp_p, self.filtersetname)
            alpha_mode = alpha_init
        else:
            raise ValueError(f"Неизвестный тип априорного распределения (prior_type): {self.prior_type}")

        beta_mode = np.array([beta_sar])
        gamma_mode = alpha_sar

        # --- 4. Запуск основного решателя ---
        if self.prior_type == 'tv':
            y_hr, alpha_out, beta_out, gamma_out, W_out = TVME_Sens(
                Y_norm, x_norm, lam, psf, nbands,
                eps_map=self.eps_map, itmax_map=self.itmax_map,
                itmin_map=self.itmin_map,
                alpha_mode=alpha_mode, beta_mode=beta_mode,
                gamma_mode=gamma_mode, gamma_gamma=self.gamma_gamma,
                eps_y=self.eps_y, itmax_y=self.itmax_y,
                verbose=self.verbose,
            )
        else:
            kappa = getkappa(self.prior_type,
                             self.lp_p if self.prior_type == 'lp' else None)
            y_hr, alpha_out, beta_out, gamma_out, W_out = restSGME_Sens(
                Y_norm, x_norm, lam, kappa, self.filtersetname, psf, nbands,
                eps_map=self.eps_map, itmax_map=self.itmax_map,
                itmin_map=self.itmin_map,
                alpha_mode=alpha_mode, beta_mode=beta_mode,
                gamma_mode=gamma_mode, gamma_gamma=self.gamma_gamma,
                eps_y=self.eps_y, itmax_y=self.itmax_y,
                verbose=self.verbose,
            )

        # --- 5. Денормализация и формирование вывода ---
        if y_hr.ndim == 3:
            y_hr = y_hr[:, :, 0]

        y_hr = image_denormalize(y_hr, facY)

        elapsed = time.time() - start_time
        self.hyperparams = {
            'ratio': self.ratio,
            'prior_type': self.prior_type,
            'filtersetname': self.filtersetname,
            'eps_map': self.eps_map,
            'itmax_map': self.itmax_map,
            'alpha': alpha_out,
            'beta': beta_out,
            'gamma': gamma_out,
            'time': elapsed,
        }

        x_final = y_hr * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        dummy_kernel = np.zeros((3, 3), dtype=np.float64)
        
        return x_final, dummy_kernel

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('ratio', self.ratio),
            ('prior_type', self.prior_type),
            ('filtersetname', self.filtersetname),
            ('lp_p', self.lp_p),
            ('sensor', self.sensor),
            ('eps_map', self.eps_map),
            ('itmax_map', self.itmax_map),
            ('itmin_map', self.itmin_map),
            ('eps_y', self.eps_y),
            ('itmax_y', self.itmax_y),
            ('gamma_gamma', self.gamma_gamma),
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