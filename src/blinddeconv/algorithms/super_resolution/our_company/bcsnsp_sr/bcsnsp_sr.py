"""
bcsnsp_sr.py

Сверхразрешение на основе байесовской комбинации разреженных и неразреженных 
априорных распределений (Bayesian Combination of Sparse and Non-Sparse Priors 
Super-Resolution - BCSNSP-SR).

Содержание алгоритма:
    1. Инициализация входного изображения (рассматривается как HR оригинал 
       в режиме тестирования или как базовый кадр в режиме масштабирования).
    2. Симуляция L наблюдений низкого разрешения (LR) через операторы сдвига 
       и размытия.
    3. Выполнение итеративной реконструкции сверхразрешения.
    4. Возврат восстановленного изображения высокого разрешения (HR).

Литература:
[1] S. D. Babacan, R. Molina, A. K. Katsaggelos,
    "Bayesian Super Resolution Image Reconstruction using an l1 Prior",
    ISPA 2009 / Chapter in Bayesian Inference, 2011.
[2] J. Salvador, S. Villena, R. Molina, A. K. Katsaggelos,
    "Bayesian Combination of Sparse and Non-Sparse Priors in
    Image Super Resolution", Digital Signal Processing, 2013.
"""

import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np
from scipy.ndimage import shift as _ndshift

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

from .solvers import create_data, solvex_var_l4_sar
from .utils import fspecial_gaussian


class BCSNSP_SR(DeconvolutionAlgorithm):
    """
    Алгоритм сверхразрешения на основе байесовской комбинации априорных распределений.

    Режимы работы
    -------------
    'upscale' (по умолчанию) - Практическое масштабирование.
        Вход: LR изображение (m × n).
        Выход: HR изображение (m*res × n*res).
        Генерирует L псевдокадров из одного входного изображения с помощью 
        субпиксельных сдвигов, затем реконструирует HR изображение.

    'benchmark' - Режим симуляции и тестирования.
        Вход: HR изображение (M × N).
        Выход: HR изображение (M × N).
        Искусственно ухудшает HR вход до LR кадров, затем восстанавливает.

    Параметры алгоритма
    -------------------
    res : int
        Коэффициент масштабирования (увеличения).
    L : int
        Количество (симулируемых) кадров низкого разрешения.
    sigma : float
        Ожидаемое стандартное отклонение шума наблюдений.
    blur_size : int
        Пространственный размер ядра размытия (нечетное число).
    blur_sigma : float
        Стандартное отклонение гауссовского ядра размытия.
    lambda_prior : float
        Баланс между L1-TV и SAR априорными распределениями (диапазон [0, 1]).
    maxit : int
        Максимальное количество итераций реконструкции.
    thr : float
        Порог сходимости для остановки итераций.
    method : str
        Метод оптимизации: 'variational' или 'degenerate'.
    estimate_reg : bool
        Флаг обновления параметров регистрации на каждой итерации.
    max_shift : float
        Максимальный субпиксельный сдвиг (в пикселях LR для 'upscale', 
        в пикселях HR для 'benchmark').
    max_theta : float
        Максимальный случайный угол поворота (в радианах) для режима 'benchmark'.
    pcg_thr : float
        Допуск для решателя метода сопряженных градиентов (PCG).
    pcg_maxit : int
        Максимальное количество итераций PCG.
    pcg_minit : int
        Минимальное количество итераций PCG.
    mode : str
        Режим работы: 'upscale' или 'benchmark'.
    verbose : bool
        Флаг вывода отладочной информации по итерациям.
    seed : int или None
        Начальное значение генератора случайных чисел для воспроизводимости.
    """

    def __init__(
        self,
        res: int = 2,
        L: int = 4,
        sigma: float = 0.01,
        blur_size: int = 3,
        blur_sigma: float = 0.5,
        lambda_prior: float = 0.5,
        maxit: int = 30,
        thr: float = 1e-4,
        method: str = 'variational',
        estimate_reg: bool = True,
        max_shift: float = 0.5,
        max_theta: float = 0.01,
        pcg_thr: float = 1e-6,
        pcg_maxit: int = 100,
        pcg_minit: int = 10,
        mode: str = 'upscale',
        verbose: bool = False,
        seed: int | None = None,
    ):
        super().__init__(name='BCSNSP-SR')

        self.res = res
        self.L = L
        self.sigma = sigma
        self.blur_size = blur_size
        self.blur_sigma = blur_sigma
        self.lambda_prior = lambda_prior
        self.maxit = maxit
        self.thr = thr
        self.method = method
        self.estimate_reg = estimate_reg
        self.max_shift = max_shift
        self.max_theta = max_theta
        self.pcg_thr = pcg_thr
        self.pcg_maxit = pcg_maxit
        self.pcg_minit = pcg_minit
        self.mode = mode
        self.verbose = verbose
        self.seed = seed

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной процесс реконструкции сверхразрешения."""
        start_time = time.time()

        if self.seed is not None:
            np.random.seed(self.seed)

        # --- 1. Нормализация к float64 [0, 1] в градациях серого ---
        img = image.astype(np.float64)
        if img.ndim == 3:
            if img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + \
                      0.1140 * img[:, :, 2]
        if img.max() > 1.0:
            img /= 255.0

        h = fspecial_gaussian(self.blur_size, self.blur_sigma)

        # --- 2. Выполнение алгоритма в зависимости от режима ---
        if self.mode == 'benchmark':
            x_vec, out, M, N = self._process_benchmark(img, h)
        else:
            x_vec, out, M, N = self._process_upscale(img, h)

        # --- 3. Формирование результатов ---
        x_img = x_vec.reshape(M, N, order='F')
        x_img = np.clip(x_img, 0.0, 1.0)

        self.hyperparams = {
            'res': self.res,
            'L': self.L,
            'sigma': self.sigma,
            'lambda_prior': self.lambda_prior,
            'method': self.method,
            'mode': self.mode,
            'iterations': out['iterations'],
            'input_shape': img.shape,
            'output_shape': (M, N),
            'xconv': out['xconv'],
            'time': time.time() - start_time,
        }
        if out['history']['PSNRs']:
            self.hyperparams['final_psnr'] = out['history']['PSNRs'][-1]
        self.history = out['history']

        x_final = (x_img * 255.0).clip(0, 255).astype(np.int16)
        kernel = np.zeros((3, 3), dtype=np.float64)
        return x_final, kernel

    def _process_upscale(self, img, h):
        """
        Режим практического масштабирования.
        Вход: Одно LR изображение (m × n).
        Генерирует L псевдокадров через субпиксельные сдвиги.
        Выход: Вектор HR изображения размером (m*res × n*res).
        """
        m, n = img.shape
        M = m * self.res
        N = n * self.res

        # Генерация L кадров из одного входного через субпиксельные сдвиги
        sx_lr = np.zeros(self.L)
        sy_lr = np.zeros(self.L)

        frames_vec = [img.ravel(order='F')]
        for k in range(1, self.L):
            sx_lr[k] = (np.random.rand() * 2 - 1) * self.max_shift
            sy_lr[k] = (np.random.rand() * 2 - 1) * self.max_shift
            shifted = _ndshift(img, [sy_lr[k], sx_lr[k]],
                               order=1, mode='reflect')
            frames_vec.append(shifted.ravel(order='F'))

        y = np.concatenate(frames_vec)

        # Конвертация сдвигов из LR-пространства в HR-пространство
        sx_hr = sx_lr * self.res
        sy_hr = sy_lr * self.res
        theta_hr = np.zeros(self.L)

        x_vec, out = solvex_var_l4_sar(
            y, M=M, N=N, m=m, n=n, res=self.res, L=self.L, h=h,
            sx=sx_hr, sy=sy_hr, theta=theta_hr,
            xtrue=None,
            method=self.method,
            lambda_prior=self.lambda_prior,
            maxit=self.maxit,
            thr=self.thr,
            pcg_thr=self.pcg_thr,
            pcg_maxit=self.pcg_maxit,
            pcg_minit=self.pcg_minit,
            estimate_registration=False,  
            verbose=self.verbose,
        )
        return x_vec, out, M, N

    def _process_benchmark(self, img, h):
        """
        Режим симуляции.
        Вход: Истинное HR изображение (M × N).
        Искусственно деградирует изображение до L LR кадров, затем реконструирует.
        Выход: Вектор восстановленного HR изображения.
        """
        M_raw, N_raw = img.shape
        m = M_raw // self.res
        n = N_raw // self.res
        M = m * self.res
        N = n * self.res
        img = img[:M, :N]

        sx_true = np.zeros(self.L)
        sy_true = np.zeros(self.L)
        theta_true = np.zeros(self.L)
        for k in range(1, self.L):
            sx_true[k] = (np.random.rand() * 2 - 1) * self.max_shift
            sy_true[k] = (np.random.rand() * 2 - 1) * self.max_shift
            theta_true[k] = (np.random.rand() * 2 - 1) * self.max_theta

        y, _W = create_data(img, h, M, N, self.res, self.L,
                            sx_true, sy_true, theta_true, self.sigma)

        sx_init = sx_true.copy()
        sy_init = sy_true.copy()
        theta_init = theta_true.copy()
        for k in range(1, self.L):
            sx_init[k] += np.random.randn() * 0.1 * self.max_shift
            sy_init[k] += np.random.randn() * 0.1 * self.max_shift
            theta_init[k] += np.random.randn() * 0.1 * self.max_theta

        x_vec, out = solvex_var_l4_sar(
            y, M=M, N=N, m=m, n=n, res=self.res, L=self.L, h=h,
            sx=sx_init, sy=sy_init, theta=theta_init,
            sx_init=sx_init.copy(), sy_init=sy_init.copy(),
            theta_init=theta_init.copy(),
            xtrue=img,
            method=self.method,
            lambda_prior=self.lambda_prior,
            maxit=self.maxit,
            thr=self.thr,
            pcg_thr=self.pcg_thr,
            pcg_maxit=self.pcg_maxit,
            pcg_minit=self.pcg_minit,
            estimate_registration=self.estimate_reg,
            verbose=self.verbose,
        )
        return x_vec, out, M, N

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('res', self.res),
            ('L', self.L),
            ('sigma', self.sigma),
            ('blur_size', self.blur_size),
            ('blur_sigma', self.blur_sigma),
            ('lambda_prior', self.lambda_prior),
            ('maxit', self.maxit),
            ('thr', self.thr),
            ('method', self.method),
            ('estimate_reg', self.estimate_reg),
            ('max_shift', self.max_shift),
            ('max_theta', self.max_theta),
            ('pcg_thr', self.pcg_thr),
            ('pcg_maxit', self.pcg_maxit),
            ('pcg_minit', self.pcg_minit),
            ('mode', self.mode),
            ('verbose', self.verbose),
            ('seed', self.seed),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams