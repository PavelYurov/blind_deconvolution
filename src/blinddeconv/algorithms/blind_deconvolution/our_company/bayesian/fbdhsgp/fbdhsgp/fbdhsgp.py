"""
fbdhsgp.py

Быстрая байесовская слепая деконволюция с хуберовскими супергауссовскими 
априорными распределениями (Fast Bayesian Blind Deconvolution with Huber 
Super Gaussian Priors - FBDHSGP).

Содержит:
    - FBDHSGP: Основной класс алгоритма слепой деконволюции. Принимает изображение
      и возвращает восстановленный результат вместе с оцененным ядром размытия.
    - _multiscale_bid: Реализация многомасштабной оценки ядра на основе 
      построения пирамиды изображений.
    - _init_vars, _default_firls_opts: Вспомогательные методы инициализации 
      параметров для решателей и финальной неслепой деконволюции.

Литература:
[1] X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
    "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
    Digital Signal Processing, 2017.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np

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

from .solvers import frils_deb_ubc, ss_deb
from .utils import (
    imresize_bilinear,
    init_kernel,
    pad_replicate,
    shift_kernel_img_space,
)


class FBDHSGP(DeconvolutionAlgorithm):
    """
    Алгоритм быстрой байесовской слепой деконволюции с хуберовскими 
    супергауссовскими априорными распределениями.

    Принимает одно полутоновое изображение и возвращает восстановленное 
    изображение вместе с оцененной функцией рассеяния точки (ФРТ).

    Параметры алгоритма
    -------------------
    kernel_size : tuple[int, int]
        Пространственный размер неизвестного ядра размытия (оба числа 
        должны быть нечетными; по умолчанию (35, 35)).
    sigma : float
        Стандартное отклонение шума (по умолчанию 0.01 согласно Разделу 4 статьи).
    epsilon_min : float
        Параметр eps для априорного распределения Хьюбера (по умолчанию 0.002; 
        рекомендуемый диапазон [0.001, 0.004]).
    beta_v : float
        Параметр штрафа ADMM для v_gamma = F_gamma * x 
        (по умолчанию 0.1; рекомендуемый диапазон [0.1, 1]).
    beta_H : float
        Параметр штрафа ADMM для H = F * P * h (по умолчанию 10.0).
    K1 : int
        Внешние итерации метода IRLS для обновления изображения (по умолчанию 10).
    K2 : int
        Внутренние итерации ADMM для обновления изображения (по умолчанию 1; 
        вместе с большим значением beta_v это реализует сглаженный вариант SADMM).
    xh_iter : int
        Количество чередующихся обновлений изображения/ядра на каждом масштабе 
        (по умолчанию 15).
    h_iter : int
        Итерации ADMM внутри процедуры оценки ядра (по умолчанию 10).
    delta : float
        Допуск остановки для обновлений h внутри Алгоритма 2 (по умолчанию 0.002).
    delta_x : float
        Допуск остановки для цикла IRLS оценки изображения (по умолчанию 0.001).
    x_warm_start : int
        Если 1, используется "теплый старт" для x между масштабами/итерациями 
        (по умолчанию 1).
    gamma_correct : float
        Экспонента гамма-коррекции, применяемая к входу перед деконволюцией 
        (по умолчанию 1.0).
    prior_name : str
        Выбор априорной модели: 'Log', 'Lp', 'MOG' или 'NL1' (по умолчанию 'Log').
    prior_alpha : float
        Множитель силы априорной регуляризации t (по умолчанию 1.0).

    Параметры финального неслепого шага (frils_deb_ubc)
    ---------------------------------------------------
    Контролируют финальную деконволюцию с Lp-регуляризацией (Уравнение 31).
    Настройки по умолчанию соответствуют рекомендациям оригинальной статьи.
    Для сложных (больших или криволинейных) ядер наиболее важными параметрами являются:
        * firls_out_iter — увеличение с 5 до 10..20 позволяет beta достичь beta_max.
        * firls_alpha — 2/3 (по умолчанию, более резкий результат) или 0.8 (мягче).
        * firls_lambda — уменьшить для большей резкости, увеличить при сильном шуме.
        * firls_epsilon_min — меньшее значение приближает модель к истинной Lp-норме.

    firls_out_iter : int
        Внешние итерации продолжения (continuation) по параметру beta (по умолчанию 5).
    firls_inner_iter : int
        Внутренние итерации ADMM на каждый уровень beta (по умолчанию 4).
    firls_IF : float
        Множитель для шага продолжения beta (по умолчанию sqrt(2)).
    firls_lambda : float
        Компромисс между точностью данных и априорным распределением (по умолчанию 2e-4).
    firls_lambda_u : float
        Штраф за ограничение поля зрения (FOV) для подзадачи u (по умолчанию 0.1).
    firls_epsilon_min : float
        Параметр eps для априорного распределения Huber-Lp (по умолчанию 2.55/255).
    firls_epsilon_max : float
        Начальное значение eps (по умолчанию равно firls_epsilon_min; для постепенного
        ужесточения регуляризации можно использовать большее значение).
    firls_alpha : float
        Экспонента Lp для априорного распределения (по умолчанию 2/3).
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (35, 35),
        sigma: float = 0.01,
        epsilon_min: float = 0.002,
        beta_v: float = 0.1,
        beta_H: float = 10.0,
        K1: int = 10,
        K2: int = 1,
        xh_iter: int = 15,
        h_iter: int = 10,
        delta: float = 0.002,
        delta_x: float = 0.001,
        x_warm_start: int = 1,
        gamma_correct: float = 1.0,
        prior_name: str = "Log",
        prior_alpha: float = 1.0,
        firls_out_iter: int = 5,
        firls_inner_iter: int = 4,
        firls_IF: float = float(np.sqrt(2.0)),
        firls_lambda: float = 2e-4,
        firls_lambda_u: float = 0.1,
        firls_epsilon_min: float = 2.55 / 255.0,
        firls_epsilon_max: float | None = None,
        firls_alpha: float = 2.0 / 3.0,
    ):
        super().__init__(name="FBDHSGP")

        self.kernel_size = tuple(int(s) for s in kernel_size)
        self.sigma = float(sigma)
        self.epsilon_min = float(epsilon_min)
        self.beta_v = float(beta_v)
        self.beta_H = float(beta_H)
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.xh_iter = int(xh_iter)
        self.h_iter = int(h_iter)
        self.delta = float(delta)
        self.delta_x = float(delta_x)
        self.x_warm_start = int(x_warm_start)
        self.gamma_correct = float(gamma_correct)
        self.prior_name = str(prior_name)
        self.prior_alpha = float(prior_alpha)

        self.firls_out_iter = int(firls_out_iter)
        self.firls_inner_iter = int(firls_inner_iter)
        self.firls_IF = float(firls_IF)
        self.firls_lambda = float(firls_lambda)
        self.firls_lambda_u = float(firls_lambda_u)
        self.firls_epsilon_min = float(firls_epsilon_min)
        self.firls_epsilon_max = (
            float(firls_epsilon_max)
            if firls_epsilon_max is not None
            else float(firls_epsilon_min)
        )
        self.firls_alpha = float(firls_alpha)

        self.history: Dict[str, list] = {"kernel_diff": []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основной процесс деконволюции."""
        start_time = time.time()

        # Нормализация в float64 [0, 1]
        y = np.asarray(image).astype(np.float64)
        if y.max() > 1.0:
            y = y / 255.0

        # Перевод в градации серого, если на входе RGB
        if y.ndim == 3:
            if y.shape[2] == 1:
                y = y[:, :, 0]
            elif y.shape[2] == 3:
                y = (
                    0.2989 * y[:, :, 0]
                    + 0.5870 * y[:, :, 1]
                    + 0.1140 * y[:, :, 2]
                )

        # Гамма-коррекция (опционально)
        if self.gamma_correct != 1.0:
            y = np.power(y, self.gamma_correct)

        # Построение пирамиды и многомасштабная оценка ядра
        kernel = self._multiscale_bid(y)

        # Финальная неслепая Lp деконволюция
        firls_opts = self._default_firls_opts()
        x_final = frils_deb_ubc(y, kernel, firls_opts)

        # Формирование результатов
        self.hyperparams = {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "epsilon_min": self.epsilon_min,
            "beta_v": self.beta_v,
            "beta_H": self.beta_H,
            "K1": self.K1,
            "K2": self.K2,
            "xh_iter": self.xh_iter,
            "h_iter": self.h_iter,
            "delta": self.delta,
            "delta_x": self.delta_x,
            "prior_name": self.prior_name,
            "prior_alpha": self.prior_alpha,
            "time": time.time() - start_time,
        }

        x_final = np.clip(x_final, 0.0, 1.0) * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def _multiscale_bid(self, y: np.ndarray) -> np.ndarray:
        """Реализация многомасштабной слепой деконволюции от грубого к точному."""
        blur_size = self.kernel_size

        # Выбор минимального размера ядра на самом грубом уровне
        max_ks = max(self.kernel_size)
        ind1 = 0 if self.kernel_size[0] >= self.kernel_size[1] else 1
        ind2 = 1 - ind1

        minsize = [0, 0]
        minsize[ind1] = max(3, 2 * ((max_ks - 1) // 64) + 1)
        tmp = (
            self.kernel_size[ind2] * minsize[ind1]
        ) // self.kernel_size[ind1]
        if tmp % 2 == 0:
            tmp += 1
        minsize[ind2] = max(tmp, 3)

        resize_step = np.sqrt(2.0)

        # Формирование списка масштабов
        ksize: List[List[int]] = []
        tmp = minsize[ind1]
        while tmp < max_ks:
            row = [0, 0]
            row[ind1] = int(tmp)
            tmp2 = int(
                np.ceil(self.kernel_size[ind2] / self.kernel_size[ind1] * tmp)
            )
            if tmp2 % 2 == 0:
                tmp2 += 1
            row[ind2] = max(tmp2, 3)
            ksize.append(row)

            tmp = int(np.ceil(tmp * resize_step))
            if tmp % 2 == 0:
                tmp += 1
        ksize.append([int(self.kernel_size[0]), int(self.kernel_size[1])])
        num_scales = len(ksize)

        ks: List[np.ndarray | None] = [None] * num_scales
        ls: List[np.ndarray | None] = [None] * num_scales

        xvars, hvars = self._init_vars()

        for s in range(num_scales):
            if s == 0:
                Gsigma = 1.0 if max_ks > 50 else 0.5
                ks[s] = init_kernel(ksize[0], Gsigma)
                k1, k2 = ksize[0]
            else:
                k1, k2 = ksize[s]
                tmp_k = ks[s - 1]
                tmp_k = np.where(tmp_k < 0, 0.0, tmp_k)
                s_sum = tmp_k.sum()
                if s_sum > 0:
                    tmp_k = tmp_k / s_sum
                ks[s] = imresize_bilinear(tmp_k, (k1, k2))
                ks[s] = np.where(ks[s] < 0, 0.0, ks[s])
                s_sum = ks[s].sum()
                if s_sum > 0:
                    ks[s] = ks[s] / s_sum

            # Вычисление размера изображения для текущего масштаба
            r = int(np.floor(y.shape[0] * k1 / blur_size[0]))
            c = int(np.floor(y.shape[1] * k2 / blur_size[1]))
            if s == num_scales - 1:
                r, c = y.shape

            ys = imresize_bilinear(y, (r, c))

            if s == 0:
                ls[s] = ys
            else:
                ls[s] = imresize_bilinear(ls[s - 1], (r, c))
                ls[s - 1] = None

            # Центрирование ядра в пространстве изображения
            ls[s], ks[s] = shift_kernel_img_space(ls[s], ks[s])

            hks1 = k1 // 2
            hks2 = k2 // 2

            xvars["x0"] = pad_replicate(ls[s], hks1, hks2)
            
            # Сброс временных параметров при переходе на новый масштаб
            for key in ("RR", "cov_img", "dvx", "dvy", "ye", "X", "H"):
                xvars.pop(key, None)
            for key in ("H", "dH"):
                hvars.pop(key, None)

            hvars["h"] = ks[s]
            hvars["scale_ratio"] = (k1 * k2) / (
                ksize[num_scales - 1][0] * ksize[num_scales - 1][1]
            )

            # Выполнение слепой деконволюции для одного масштаба
            ls[s], ks[s] = ss_deb(ys, xvars, hvars)

        kernel = ks[num_scales - 1]
        kernel = np.where(kernel < 0, 0.0, kernel)
        s_sum = kernel.sum()
        if s_sum > 0:
            kernel = kernel / s_sum

        # Финальное центрирование итогового ядра
        # Гарантирует отсутствие глобального смещения (трансляции) изображения,
        # сохраняя при этом все компоненты диффузных или сложных ядер.
        ks_h, ks_w = kernel.shape
        if kernel.sum() > 0:
            ys_idx, xs_idx = np.indices(kernel.shape)
            mu_y = float((ys_idx * kernel).sum() / kernel.sum())
            mu_x = float((xs_idx * kernel).sum() / kernel.sum())
            shift_y = int(np.round(ks_h // 2 - mu_y))
            shift_x = int(np.round(ks_w // 2 - mu_x))
            if shift_y != 0 or shift_x != 0:
                kernel = np.roll(kernel, shift_y, axis=0)
                kernel = np.roll(kernel, shift_x, axis=1)
                if shift_y > 0:
                    kernel[:shift_y, :] = 0.0
                elif shift_y < 0:
                    kernel[shift_y:, :] = 0.0
                if shift_x > 0:
                    kernel[:, :shift_x] = 0.0
                elif shift_x < 0:
                    kernel[:, shift_x:] = 0.0
                s_sum = kernel.sum()
                if s_sum > 0:
                    kernel = kernel / s_sum

        return kernel

    def _init_vars(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Инициализация словарей с параметрами для решателей."""
        prior = {
            "name": self.prior_name,
            "alpha": self.prior_alpha,
            "conv": 0,
            "const_weight": 0,
        }
        xvars: Dict[str, Any] = {
            "sigma": self.sigma,
            "epsilon_min": self.epsilon_min,
            "x_warm_start": self.x_warm_start,
            "xh_iter": self.xh_iter,
            "K1": self.K1,
            "K2": self.K2,
            "delta_x": self.delta_x,
            "priors": prior,
            "alpha": 0.0,
            "beta_v": self.beta_v,
        }
        hvars: Dict[str, Any] = {
            "beta_H": self.beta_H,
            "delta": self.delta,
            "h_iter": self.h_iter,
            "lambda_h": None,
            "sigma": self.sigma,
        }
        return xvars, hvars

    def _default_firls_opts(self) -> Dict[str, float]:
        """Сборка параметров для финальной неслепой деконволюции.
        
        Вычисление параметра beta_a согласно рекомендациям оригинального алгоритма:
        beta_a = lambda * alpha * (20/255) ** (alpha - 2)
        """
        firls = {
            "out_iter": self.firls_out_iter,
            "inner_iter": self.firls_inner_iter,
            "IF": self.firls_IF,
            "lambda": self.firls_lambda,
            "lambda_u": self.firls_lambda_u,
            "epsilon_min": self.firls_epsilon_min,
            "epsilon_max": self.firls_epsilon_max,
            "alpha": self.firls_alpha,
        }
        firls["beta_a"] = (
            firls["lambda"]
            * firls["alpha"]
            * (20.0 / 255.0) ** (firls["alpha"] - 2.0)
        )
        return firls

    def get_param(self) -> List[Tuple[str, Any]]:
        return [
            ("kernel_size", self.kernel_size),
            ("sigma", self.sigma),
            ("epsilon_min", self.epsilon_min),
            ("beta_v", self.beta_v),
            ("beta_H", self.beta_H),
            ("K1", self.K1),
            ("K2", self.K2),
            ("xh_iter", self.xh_iter),
            ("h_iter", self.h_iter),
            ("delta", self.delta),
            ("delta_x", self.delta_x),
            ("x_warm_start", self.x_warm_start),
            ("gamma_correct", self.gamma_correct),
            ("prior_name", self.prior_name),
            ("prior_alpha", self.prior_alpha),
            ("firls_out_iter", self.firls_out_iter),
            ("firls_inner_iter", self.firls_inner_iter),
            ("firls_IF", self.firls_IF),
            ("firls_lambda", self.firls_lambda),
            ("firls_lambda_u", self.firls_lambda_u),
            ("firls_epsilon_min", self.firls_epsilon_min),
            ("firls_epsilon_max", self.firls_epsilon_max),
            ("firls_alpha", self.firls_alpha),
        ]

    def change_param(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            if hasattr(self, key):
                if key == "kernel_size":
                    self.kernel_size = tuple(int(s) for s in value)
                else:
                    setattr(self, key, value)

    def get_history(self) -> dict:
        return self.history

    def get_hyperparams(self) -> dict:
        return self.hyperparams