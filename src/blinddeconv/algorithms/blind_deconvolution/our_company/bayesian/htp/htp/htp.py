"""
htp.py

Слепая деконволюция изображений с использованием априорных распределений 
с тяжелыми хвостами (Blind Image Deblurring Using Heavy-Tailed Priors - HTP).

Содержит:
    - HTP_BD: Основной класс алгоритма слепой деконволюции. Принимает изображение
      и возвращает восстановленный результат вместе с оцененным ядром размытия.

Содержание алгоритма:
    1. Нормализация входного изображения в диапазон [0, 1] (тип float64).
    2. Построение пирамиды изображений от грубого к точному масштабу для 
       центральной области интереса (ROI).
    3. Многомасштабная чередующаяся MAP-оценка для скрытого изображения u 
       и функции рассеяния точки (ФРТ) h. Используется Lp-априорное 
       распределение на градиенты изображения (p < 1) и L1-априорное 
       распределение на ФРТ. Решается через полуквадратичное расщепление 
       и итерации Брегмана в Фурье-области на каждом масштабе.
    4. Финальная неслепая деконволюция полного изображения с усиленным 
       согласованием данных и TV-подобной регуляризацией (Lp_nonblind = 1).

Литература:
[1] J. Kotera, F. Sroubek, P. Milanfar:
    "Blind Deconvolution Using Alternating Maximum a Posteriori
     Estimation with Heavy-tailed Priors", CAIP 2013.
"""

import time
import sys
from pathlib import Path
from typing import Tuple, List, Any, Dict

import numpy as np


# --- Внутренний импорт базового класса алгоритма ---
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
# ---------------------------------------------------

from .solvers import mc_restoration


class HTP_BD(DeconvolutionAlgorithm):
    """
    Алгоритм слепой деконволюции с использованием априорных распределений 
    с тяжелыми хвостами [1].

    Параметры алгоритма
    -------------------
    kernel_size : int
        Пространственный размер неизвестной функции рассеяния точки 
        (квадратное ядро). По умолчанию 31.
    Lp : float
        Экспонента Lp-нормы для априорного распределения на градиенты 
        скрытого изображения во время оценки ФРТ (0 < p <= 1). 
        По умолчанию 0.3 (тяжелые хвосты).
    gamma : float
        Вес члена согласования данных (data-term) при оценке ФРТ. 
        Настраивается в зависимости от уровня шума (например, 10 дБ -> 1e1). 
        По умолчанию 1e2.
    alpha_u : float
        Относительный вес априорного распределения изображения 
        (умножается на gamma). По умолчанию 1e-2.
    beta_u : float
        Относительный вес штрафа расщепления Брегмана для изображения 
        (умножается на gamma). По умолчанию 1e0.
    alpha_h : float
        Относительный вес L1-априорного распределения для ФРТ 
        (умножается на gamma). По умолчанию 1e1.
    beta_h : float
        Относительный вес штрафа расщепления Брегмана для ФРТ 
        (умножается на gamma). По умолчанию 1e4.
    centering_threshold : float
        Порог, используемый при центрировании ФРТ между итерациями. 
        По умолчанию 20/255. Значение <= 0 отключает центрирование.
    gamma_nonblind : float
        Вес члена согласования данных для финальной неслепой деконволюции 
        (относительно gamma). По умолчанию 2e1.
    beta_u_nonblind : float
        Вес штрафа расщепления для финального неслепого шага 
        (умножается на gamma_nonblind). По умолчанию 1e-2.
    Lp_nonblind : float
        Экспонента Lp для финального неслепого шага. 
        По умолчанию 1.0 (TV-подобная регуляризация).
    maxiter : int
        Внешние чередующиеся итерации на каждом уровне пирамиды масштабов. 
        По умолчанию 10.
    maxiter_u : int
        Внутренние итерации оценки скрытого изображения u. По умолчанию 10.
    maxiter_h : int
        Внутренние итерации оценки ФРТ h. По умолчанию 10.
    ccreltol : float
        Относительный допуск остановки для внутренних циклов. По умолчанию 1e-3.
    MSlevels : int
        Количество масштабов пирамиды (>= 1). По умолчанию 4.
    maxROIsize : tuple[int, int]
        Центральная область интереса (ROI), используемая для оценки ядра. 
        По умолчанию (1024, 1024).
    verbose : int
        0 = без вывода сообщений, 1 = вывод промежуточных шагов. По умолчанию 0.
    kernel_flip : str
        Опция отражения/поворота финального ядра ('none', 'lr', 'ud', 'rot180').
    auto_recenter : bool
        Если True, применяется автоматическое центрирование итогового ядра.
    recenter_mode : str
        Режим центрирования: 'centroid' (центр масс по рамке), 'peak' (максимум) 
        или 'masscentroid' (прямой центр масс). По умолчанию 'centroid'.
    kernel_thresh : float
        Порог жесткого отсечения для ядра. По умолчанию 0.0.
    iterative_recenter : bool
        Если True, центрирование ядра выполняется на каждой итерации.
    """

    def __init__(
        self,
        kernel_size: int = 31,
        Lp: float = 0.3,
        gamma: float = 1e2,
        alpha_u: float = 1e-2,
        beta_u: float = 1e0,
        alpha_h: float = 1e1,
        beta_h: float = 1e4,
        centering_threshold: float = 20.0 / 255.0,
        gamma_nonblind: float = 2e1,
        beta_u_nonblind: float = 1e-2,
        Lp_nonblind: float = 1.0,
        maxiter: int = 10,
        maxiter_u: int = 10,
        maxiter_h: int = 10,
        ccreltol: float = 1e-3,
        MSlevels: int = 4,
        maxROIsize: Tuple[int, int] = (1024, 1024),
        verbose: int = 0,
        kernel_flip: str = 'none',
        auto_recenter: bool = False,
        recenter_mode: str = 'centroid',
        kernel_thresh: float = 0.0,
        iterative_recenter: bool = True,
    ):
        super().__init__(name='HTP-BD')

        self.kernel_size = int(kernel_size)
        self.Lp = float(Lp)
        self.gamma = float(gamma)
        self.alpha_u = float(alpha_u)
        self.beta_u = float(beta_u)
        self.alpha_h = float(alpha_h)
        self.beta_h = float(beta_h)
        self.centering_threshold = float(centering_threshold)
        self.gamma_nonblind = float(gamma_nonblind)
        self.beta_u_nonblind = float(beta_u_nonblind)
        self.Lp_nonblind = float(Lp_nonblind)
        self.maxiter = int(maxiter)
        self.maxiter_u = int(maxiter_u)
        self.maxiter_h = int(maxiter_h)
        self.ccreltol = float(ccreltol)
        self.MSlevels = int(MSlevels)
        self.maxROIsize = tuple(maxROIsize)
        self.verbose = int(verbose)

        if kernel_flip not in ('none', 'lr', 'ud', 'rot180'):
            raise ValueError(
                f"kernel_flip должно быть 'none', 'lr', 'ud' или 'rot180', "
                f"получено {kernel_flip!r}"
            )
        self.kernel_flip = kernel_flip

        if recenter_mode not in ('centroid', 'peak', 'masscentroid'):
            raise ValueError(
                f"recenter_mode должно быть 'centroid', 'peak' или "
                f"'masscentroid', получено {recenter_mode!r}"
            )
        self.auto_recenter = bool(auto_recenter)
        self.recenter_mode = recenter_mode
        self.kernel_thresh = float(kernel_thresh)
        self.iterative_recenter = bool(iterative_recenter)

        self.history: Dict[str, list] = {'kernel_diff': []}
        self.hyperparams: Dict[str, Any] = {}

    def _build_par(self) -> Dict[str, Any]:
        """
        Формирование словаря параметров для решателей.
        Веса априорных распределений масштабируются параметром gamma.
        """
        gamma = self.gamma
        gamma_nb = self.gamma_nonblind * gamma
        return {
            'verbose': self.verbose,
            'gamma': gamma,
            'Lp': self.Lp,
            'beta_h': self.beta_h * gamma,
            'alpha_h': self.alpha_h * gamma,
            'centering_threshold': self.centering_threshold,
            'beta_u': self.beta_u * gamma,
            'alpha_u': self.alpha_u * gamma,
            'gamma_nonblind': gamma_nb,
            'beta_u_nonblind': self.beta_u_nonblind * gamma_nb,
            'Lp_nonblind': self.Lp_nonblind,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'maxiter': self.maxiter,
            'ccreltol': self.ccreltol,
            'kernel_thresh': self.kernel_thresh,
            'iterative_recenter': self.iterative_recenter,
        }

    def _recenter_kernel_and_image(
        self, H: np.ndarray, U: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Пространственное центрирование ядра H с противоположным сдвигом 
        изображения U для сохранения инвариантности свертки g = h * u.

        Используется ограничивающая рамка (bounding box) порогового 
        значения ядра для более робастного определения центра.
        """
        kh, kw = H.shape
        cy_int = kh // 2
        cx_int = kw // 2

        # --- Определение смещения ядра ---
        if self.recenter_mode == 'peak':
            iy, ix = np.unravel_index(int(np.argmax(H)), H.shape)
            sy, sx = int(cy_int - iy), int(cx_int - ix)

        elif self.recenter_mode == 'masscentroid':
            Hp = np.maximum(H, 0.0)
            s = Hp.sum()
            if s <= 0:
                return H, U
            ys = np.arange(kh)[:, None]
            xs = np.arange(kw)[None, :]
            iy = (Hp * ys).sum() / s
            ix = (Hp * xs).sum() / s
            sy = int(round((kh - 1) / 2.0 - iy))
            sx = int(round((kw - 1) / 2.0 - ix))

        else:  # 'centroid' — на основе ограничивающей рамки
            Hp = np.maximum(H, 0.0)
            m = Hp.max()
            if m <= 0:
                return H, U
            
            tao = 0.03
            thr = min(m * tao, 0.002)
            mask = Hp >= thr
            if not mask.any():
                return H, U
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            y_top, y_bot = int(rows[0]), int(rows[-1])
            x_left, x_right = int(cols[0]), int(cols[-1])

            gap_left = x_left
            gap_right = (kw - 1) - x_right
            gap_top = y_top
            gap_bot = (kh - 1) - y_bot

            # Поправка в сторону более "тяжелого" края
            s_l = Hp[:, x_left].sum()
            s_r = Hp[:, x_right].sum()
            bonus_x = 0.01 if (s_l >= s_r) else -0.01
            s_t = Hp[y_top, :].sum()
            s_b = Hp[y_bot, :].sum()
            bonus_y = 0.01 if (s_t >= s_b) else -0.01

            sx = int(round((gap_right - gap_left + bonus_x) / 2.0))
            sy = int(round((gap_bot - gap_top + bonus_y) / 2.0))

        if sy == 0 and sx == 0:
            return H, U

        # --- Сдвиг ядра (дополнение нулями) ---
        H_new = np.zeros_like(H)
        src_r0 = max(0, -sy); src_r1 = min(kh, kh - sy)
        src_c0 = max(0, -sx); src_c1 = min(kw, kw - sx)
        dst_r0 = max(0, sy);  dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, sx);  dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            H_new[dst_r0:dst_r1, dst_c0:dst_c1] = H[src_r0:src_r1, src_c0:src_c1]
        s_h = H_new.sum()
        if s_h > 0:
            H_new = H_new / s_h

        # --- Обратный сдвиг изображения (дополнение краев) ---
        Mh, Mw = U.shape
        py0 = max(0, sy);  py1 = max(0, -sy)
        px0 = max(0, sx);  px1 = max(0, -sx)
        U_padded = np.pad(U, ((py0, py1), (px0, px1)), mode='edge')
        U_new = U_padded[py1:py1 + Mh, px1:px1 + Mw].copy()

        return H_new, U_new

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной процесс слепой деконволюции."""
        start_time = time.time()

        # --- 1. Нормализация входных данных в [0, 1] ---
        y = np.asarray(image, dtype=np.float64)
        if y.max() > 1.0:
            y = y / 255.0

        # --- 2. Формирование словаря параметров и запуск многомасштабной оценки ---
        PAR = self._build_par()
        hsize = (self.kernel_size, self.kernel_size)

        U, H, _report = mc_restoration(
            y,
            hsize=hsize,
            PAR=PAR,
            MSlevels=self.MSlevels,
            maxROIsize=self.maxROIsize,
        )
        U = np.clip(U, 0.0, 1.0)

        # --- 3. Автоматическое центрирование ядра (устранение неоднозначности сдвига) ---
        if self.auto_recenter:
            H, U = self._recenter_kernel_and_image(H, U)

        # --- 4. Формирование результатов ---
        self.hyperparams = {
            'kernel_size': self.kernel_size,
            'Lp': self.Lp,
            'gamma': self.gamma,
            'alpha_u': self.alpha_u,
            'beta_u': self.beta_u,
            'alpha_h': self.alpha_h,
            'beta_h': self.beta_h,
            'gamma_nonblind': self.gamma_nonblind,
            'beta_u_nonblind': self.beta_u_nonblind,
            'Lp_nonblind': self.Lp_nonblind,
            'MSlevels': self.MSlevels,
            'maxROIsize': self.maxROIsize,
            'maxiter': self.maxiter,
            'maxiter_u': self.maxiter_u,
            'maxiter_h': self.maxiter_h,
            'time': time.time() - start_time,
        }

        x_final = U * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)

        if self.kernel_flip == 'lr':
            H_out = H[:, ::-1].copy()
        elif self.kernel_flip == 'ud':
            H_out = H[::-1, :].copy()
        elif self.kernel_flip == 'rot180':
            H_out = H[::-1, ::-1].copy()
        else:
            H_out = H
            
        return x_final, H_out

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ('kernel_size', self.kernel_size),
            ('Lp', self.Lp),
            ('gamma', self.gamma),
            ('alpha_u', self.alpha_u),
            ('beta_u', self.beta_u),
            ('alpha_h', self.alpha_h),
            ('beta_h', self.beta_h),
            ('centering_threshold', self.centering_threshold),
            ('gamma_nonblind', self.gamma_nonblind),
            ('beta_u_nonblind', self.beta_u_nonblind),
            ('Lp_nonblind', self.Lp_nonblind),
            ('maxiter', self.maxiter),
            ('maxiter_u', self.maxiter_u),
            ('maxiter_h', self.maxiter_h),
            ('ccreltol', self.ccreltol),
            ('MSlevels', self.MSlevels),
            ('maxROIsize', self.maxROIsize),
            ('verbose', self.verbose),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == 'maxROIsize':
                    self.maxROIsize = tuple(value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams