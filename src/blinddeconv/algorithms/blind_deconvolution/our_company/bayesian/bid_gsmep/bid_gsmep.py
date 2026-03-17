"""
Blind Image Deconvolution Using the Gaussian Scale Mixture Fields of Experts Prior.
Algorithm based on the paper by Shuyin Tao et al. (2017).
"""

import numpy as np
import time
from typing import Tuple, List, Any, Dict

# --- Algorithm-specific imports ---
from .utils import *
from .solvers import *

# ── Framework base class import (DO NOT MODIFY) ─────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────


class BidGsmep(DeconvolutionAlgorithm):
    """
    Blind Image Deconvolution using GSM FoE Prior.
    Reference: Tao et al., "Blind Image Deconvolution Using the Gaussian Scale Mixture Fields of Experts Prior", 2017.
    """

    def __init__(
        self,
        kernel_shape: Tuple[int, int] = (27, 27),
        iterations: int = 5,
        lambda_reg: float = 50.0,   # Регуляризация для data term
        tau: float = 0.5,           # Регуляризация для PSF (Eq. 18)
        p_norm: float = 1.5,        # Norm for PSF prior (p approx 1.5)
        beta_min: float = 1.0,      # Начальное значение beta (splitting param)
        beta_max: float = 256.0,    # Конечное значение beta
        beta_rate: float = 2.0,     # Множитель роста beta
        threshold_co: float = 0.02, # Порог для градиентов изображения (Eq. 16)
    ):
        super().__init__(name='GSM-FoE-Prior')

        self.kernel_shape = kernel_shape
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.p_norm = p_norm
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_rate = beta_rate
        self.threshold_co = threshold_co

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}
        
        # Загрузка / Генерация фильтров GSM FoE
        # В реальном сценарии здесь должна быть загрузка из файла.
        self.filters, self.variances, self.weights = get_gsm_foe_params()

    # ── Main entry point ─────────────────────────────────────────────────
    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main processing loop based on Alternating Minimization (Section III).
        """
        start_time = time.time()
        
        # Нормализация изображения [0, 1] для стабильности
        img_float = image.astype(np.float32)
        if img_float.max() > 1.0:
            img_float /= 255.0
            
        g = img_float
        H, W = g.shape
        kh, kw = self.kernel_shape
        
        # Инициализация
        # 1. Инициализируем o копией g
        o_est = g.copy()
        
        # 2. Инициализируем h (например, Gaussian или просто spike)
        # В статье "h is initialized with...". Обычно 3x3 box или spike.
        h_est = np.zeros(self.kernel_shape, dtype=np.float32)
        center = (kh // 2, kw // 2)
        # Небольшое размытие для старта лучше чем дельта-функция для blind deconv
        sigma_init = 1.0
        y, x = np.ogrid[-center[0]:kh-center[0], -center[1]:kw-center[1]]
        h_est = np.exp(-(x*x + y*y)/(2*sigma_init**2))
        h_est /= h_est.sum()

        # Основной цикл (Alternating Minimization)
        # Статья описывает итерации, где beta растет.
        # Обычно это выглядит так: внешний цикл по beta, внутри Problem 1 и Problem 2.
        # В статье (раздел III): "Problem 1... optimize o... Problem 2... optimize h".
        # "beta is usually initialized with 1 and increased... For each given beta, (10) is split..."
        # Это значит, что цикл по beta является внешним для Sub-problems изображения,
        # но как это соотносится с оценкой ядра?
        # Обычно:
        # Loop t:
        #   1. Solve o (using HQS with increasing beta inside or fixed beta?)
        #      Статья говорит: "In the first phase... alternating minimization... simple gradient selecting...".
        #      Структура обычно такая:
        #      Outer Loop:
        #        Update h (Problem 2)
        #        Update o (Problem 1)
        #      Но HQS для o требует изменения beta. 
        #      Обычно beta сбрасывается или растет глобально.
        #      В [17] (Krishnan), на который ссылаются, beta растет внутри шага восстановления изображения.
        #      Сделаем так: 
        #      Global Iterations (Algorithm loop):
        #         1. PSF Estimation (Problem 2) - использует o с прошлой итерации.
        #         2. Image Estimation (Problem 1) - использует найденный h. Внутри него HQS (цикл по beta).
        
        # В статье есть нюанс: "two phases... PSF estimation and image restoration".
        # Phase 1: Alternating minimization для оценки PSF.
        # Phase 2: Final non-blind restoration с фиксированным PSF и подстройкой lambda.
        
        # PHASE 1: Estimating PSF
        current_lambda = self.lambda_reg
        
        for it in range(self.iterations):
            # Step 1: PSF Estimation (Problem 2)
            # В начале o = g.
            # Используем solver для Problem 2
            h_est = solve_psf_subproblem(
                g, o_est, 
                kernel_size=kh, 
                tau=self.tau, 
                p_norm=self.p_norm,
                threshold_c_o=self.threshold_co
            )
            
            # Step 2: Image Estimation (Problem 1)
            # Используем HQS. Beta растет от min до max.
            # Для этапа оценки ядра (Phase 1) нам не нужно идеальное качество картинки,
            # достаточно выделить структуры. Можно сделать меньше итераций beta.
            
            beta = self.beta_min
            o_temp = o_est
            while beta < self.beta_max:
                o_temp = solve_image_subproblem(
                    g, h_est, 
                    self.filters, (self.variances, self.weights),
                    lambda_reg=current_lambda,
                    beta=beta
                )
                beta *= self.beta_rate
            o_est = o_temp
            
            # Логирование
            self.history['kernel_diff'].append(np.sum(h_est)) # Placeholder metric

        # PHASE 2: Final Image Restoration (Non-blind)
        # "fix the estimated PSF and tune the regularization coefficient... to achieve restored image"
        # "equivalent to solve Problem 1 once more but with a larger lambda" (End of section III)
        
        final_lambda = current_lambda * 5.0 # Увеличиваем lambda как сказано в конце раздела III
        beta = self.beta_min
        x_final = g.copy() # Start from g again or current o_est
        
        while beta < self.beta_max * 2: # Более глубокая оптимизация для финала
            x_final = solve_image_subproblem(
                g, h_est, 
                self.filters, (self.variances, self.weights),
                lambda_reg=final_lambda,
                beta=beta
            )
            beta *= self.beta_rate

        self.hyperparams = {
            'time': time.time() - start_time,
            'final_lambda': final_lambda,
            'p_norm': self.p_norm
        }

        # Возврат к диапазону 0-255
        x_final = np.clip(x_final, 0, 1) * 255.0
        x_final = np.round(x_final).astype(np.uint8) # Используем uint8 для картинки
        
        return x_final, h_est

    # ── Interface methods ────────────────────────────────────────────────
    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_shape', self.kernel_shape),
            ('iterations', self.iterations),
            ('lambda_reg', self.lambda_reg),
            ('tau', self.tau),
            ('threshold_co', self.threshold_co)
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'kernel_shape':
                    self.kernel_shape = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams