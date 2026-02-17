"""
Слепая деконволюция изображений на основе дробного порядка
с PMP (Patch-wise Minimal Pixels) prior.

Обёртка над алгоритмом coarse-to-fine для интеграции с фреймворком.

Литература:
    Wu, T., Wan, S., Feng, C., Zhang, H., & Zeng, T. (2024).
    "Blind Image Deconvolution: When Patch-wise Minimal Pixels Prior
     Meets Fractional-Order Method."
    Journal of Mathematical Imaging and Vision, 67(1), 2.
    DOI: 10.1007/s10851-024-01221-x

Краткое описание метода:
    Предложена вариационная модель для слепой деконволюции,
    объединяющая два регуляризатора:

    1. Изотропный дробный Total Variation (FTV) порядка alpha in (1, 2):
       ||nabla^alpha f||_{2,1} = sum_i sqrt((D_x^alpha f_i)^2 + (D_y^alpha f_i)^2)

       Дробная производная определяется по Грюнвальду–Летникову (GL).
       Выбор alpha in (1, 2) обеспечивает более плавную регуляризацию,
       чем стандартный TV (alpha=1), подавляя кольцевые артефакты (ringing).

    2. Patch-wise Minimal Pixels (PMP) prior:
       R_PMP(f) = sum_i min_{j in P_i} |f_j|

       Аналог тёмного канала (dark channel prior): для каждого патча
       наименьшее абсолютное значение должно быть близко к нулю.
       Способствует восстановлению резких границ.

    Энергетический функционал:
       E(f, h) = (1/2)||h * f - g||_2^2
               + lambda * ||nabla^alpha f||_{2,1}
               + mu * R_PMP(f)
               + (gamma/2) * ||h||_2^2

    Ограничения на ядро: h >= 0, ||h||_1 = 1.

    Оптимизация выполняется чередующейся минимизацией (alternating minimization)
    в рамках coarse-to-fine (многомасштабной) схемы:
      - f-подзадача: ADMM с расщеплением на дробный TV и PMP
      - h-подзадача: решение в градиентной области через FFT

Модуль:
    fractional_order.py — обёртка-класс FractionalOrderBID.
    solvers.py          — логика солверов (ADMM, оценка ядра, coarse-to-fine).
    utils.py            — вспомогательный функционал (GL-коэффициенты, PMP, пирамида и пр.).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

from .solvers import coarse_to_fine

# ── Импорт базового класса фреймворка ──
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Поиск корня проекта (по наличию pyproject.toml)."""
    path = start.resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Cannot locate project root (pyproject.toml)")
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


class FractionalOrderBID(DeconvolutionAlgorithm):
    """
    Слепая деконволюция на основе дробного TV с PMP prior.

    Реализация алгоритма Wu et al. (2024):
        - Дробный TV порядка alpha in (1, 2) подавляет ringing-артефакты
        - PMP prior обеспечивает сохранение резких границ
        - Coarse-to-fine схема для устойчивой оценки ядра

    Parameters
    ----------
    kernel_size : tuple of (int, int)
        Размер искомого ядра PSF (должен быть нечётным).
        По умолчанию (25, 25).
    alpha : float
        Порядок дробной производной (1 < alpha < 2).
        По умолчанию 1.5.
    lambda_ftv : float
        Вес дробного TV регуляризатора.
    mu_pmp : float
        Вес PMP prior.
    gamma_kernel : float
        Вес Тихоновской регуляризации ядра.
    patch_size : int
        Размер патча для PMP prior.
    num_scales : int
        Число уровней пирамиды.
    outer_iter : int
        Число чередующихся итераций (f, h) на каждом масштабе.
    admm_iter : int
        Число ADMM итераций для оценки изображения.
    beta1 : float
        Штрафной параметр ADMM для дробного TV.
    beta2 : float
        Штрафной параметр ADMM для PMP.
    verbose : bool
        Вывод диагностической информации.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (25, 25),
        alpha: float = 1.5,
        lambda_ftv: float = 4e-3,
        mu_pmp: float = 1e-3,
        gamma_kernel: float = 2.0,
        patch_size: int = 5,
        num_scales: int = 5,
        outer_iter: int = 5,
        admm_iter: int = 10,
        beta1: float = 1.0,
        beta2: float = 1.0,
        kernel_threshold_ratio: float = 0.05,
        final_deconv_iter: int = 40,
        verbose: bool = False
    ):
        super().__init__(name='FractionalOrder-BID')

        # ── Параметры модели (Раздел 3 статьи) ──
        self.kernel_size = tuple(kernel_size)
        self.alpha = alpha              # Порядок дробной производной
        self.lambda_ftv = lambda_ftv    # Вес дробного TV
        self.mu_pmp = mu_pmp            # Вес PMP prior
        self.gamma_kernel = gamma_kernel  # Регуляризация ядра
        self.patch_size = patch_size    # Размер патча PMP

        # ── Параметры солвера (Раздел 4 статьи) ──
        self.num_scales = num_scales
        self.outer_iter = outer_iter
        self.admm_iter = admm_iter
        self.beta1 = beta1              # Штрафной ADMM для FTV
        self.beta2 = beta2              # Штрафной ADMM для PMP
        self.kernel_threshold_ratio = kernel_threshold_ratio
        self.final_deconv_iter = final_deconv_iter

        self.verbose = verbose

        # ── Сохранённые результаты ──
        self.history = {}
        self.hyperparams = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Основной интерфейс фреймворка: слепая деконволюция изображения.

        Parameters
        ----------
        image : np.ndarray
            Размытое изображение (H x W) или (H x W x C).
            Может быть uint8 [0, 255] или float [0, 1].

        Returns
        -------
        restored : np.ndarray
            Восстановленное изображение в формате int16, [0, 255].
        kernel : np.ndarray
            Оценённое ядро PSF (kh x kw), float64, сумма = 1.
        """
        start_time = time.time()

        # ── 1. Подготовка входных данных ──
        img = image.astype(np.float64)

        # Обработка цветных изображений: работаем с яркостным каналом
        is_color = (img.ndim == 3 and img.shape[2] == 3)

        if is_color:
            # Конвертация BGR -> YCrCb, оценка ядра по яркости (Y)
            if img.max() > 1.0:
                img /= 255.0
            img_y = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        else:
            if img.max() > 1.0:
                img /= 255.0
            img_y = img.copy()

        # ── 2. Запуск coarse-to-fine слепой деконволюции ──
        f_restored, h_estimated, history = coarse_to_fine(
            g=img_y,
            kernel_size=self.kernel_size,
            alpha=self.alpha,
            lambda_ftv=self.lambda_ftv,
            mu_pmp=self.mu_pmp,
            gamma_kernel=self.gamma_kernel,
            patch_size=self.patch_size,
            num_scales=self.num_scales,
            outer_iter=self.outer_iter,
            admm_iter=self.admm_iter,
            beta1=self.beta1,
            beta2=self.beta2,
            kernel_threshold_ratio=self.kernel_threshold_ratio,
            final_deconv_iter=self.final_deconv_iter,
            verbose=self.verbose
        )

        # ── 3. Обработка цветных изображений (применение ядра к каждому каналу) ──
        if is_color:
            from .solvers import final_nonblind_deconv
            from .utils import edgetaper

            restored_color = np.zeros_like(img, dtype=np.float64)
            for c in range(3):
                ch = img[:, :, c]
                ch_tapered = edgetaper(ch, h_estimated)
                restored_color[:, :, c] = final_nonblind_deconv(
                    ch_tapered, h_estimated,
                    alpha=self.alpha,
                    lambda_ftv=self.lambda_ftv * 0.5,
                    beta=self.beta1 * 2.0,
                    num_iter=self.final_deconv_iter
                )
            f_out = np.clip(restored_color * 255.0, 0.0, 255.0)
            f_out = np.round(f_out).astype(np.int16)
        else:
            f_out = np.clip(f_restored * 255.0, 0.0, 255.0)
            f_out = np.round(f_out).astype(np.int16)

        # ── 4. Сохранение метаданных ──
        elapsed = time.time() - start_time
        self.timer = elapsed
        self.history = history
        self.hyperparams = {
            'alpha': self.alpha,
            'lambda_ftv': self.lambda_ftv,
            'mu_pmp': self.mu_pmp,
            'gamma_kernel': self.gamma_kernel,
            'kernel_size': self.kernel_size,
            'num_scales': self.num_scales,
            'elapsed_time': elapsed,
        }

        if self.verbose:
            print(f"[{self.name}] Done in {elapsed:.2f}s")

        return f_out, h_estimated

    def get_param(self) -> List[Tuple[str, Any]]:
        """
        Получение текущих гиперпараметров алгоритма.

        Returns
        -------
        params : list of tuple
            Список (название, значение) для всех гиперпараметров.
        """
        return [
            ('kernel_size', self.kernel_size),
            ('alpha', self.alpha),
            ('lambda_ftv', self.lambda_ftv),
            ('mu_pmp', self.mu_pmp),
            ('gamma_kernel', self.gamma_kernel),
            ('patch_size', self.patch_size),
            ('num_scales', self.num_scales),
            ('outer_iter', self.outer_iter),
            ('admm_iter', self.admm_iter),
            ('beta1', self.beta1),
            ('beta2', self.beta2),
            ('final_deconv_iter', self.final_deconv_iter),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        """
        Изменение гиперпараметров алгоритма.

        Parameters
        ----------
        params : dict
            Словарь {имя_параметра: новое_значение}.
            Допустимые ключи: kernel_size, alpha, lambda_ftv, mu_pmp,
            gamma_kernel, patch_size, num_scales, outer_iter, admm_iter,
            beta1, beta2, final_deconv_iter, verbose.
        """
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_size':
                    self.kernel_size = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        """Получение истории оптимизации (kernel_diff, energy)."""
        return self.history

    def get_hyperparams(self) -> dict:
        """Получение словаря гиперпараметров после последнего запуска."""
        return self.hyperparams
