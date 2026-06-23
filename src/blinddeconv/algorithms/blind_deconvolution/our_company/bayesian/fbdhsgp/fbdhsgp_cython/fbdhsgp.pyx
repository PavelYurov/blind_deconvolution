"""
fbdhsgp.pyx

Быстрая байесовская слепая деконволюция с хуберовскими супергауссовскими 
априорными распределениями (Fast Bayesian Blind Deconvolution with Huber Super 
Gaussian Priors - FBDHSGP).

Поведение по умолчанию (auto_mode='off' и все фильтры шума в 'none') 
в точности воспроизводит базовый конвейер, описанный в оригинальной статье.

При auto_mode='robust' применяется единая упрощенная политика адаптации к шуму:
    1. Всегда запускается обнаружение импульсного шума.
    2. Всегда оценивается уровень шума (через метод 'pca').
    3. Если sigma_norm < sigma_clean -> ЧИСТОЕ ИЗОБРАЖЕНИЕ: используются
       исходные настройки из статьи, без дополнительного шумоподавления.
    4. Иначе, если обнаружен пуассоновско-гауссовский шум и включен prefer_vst_on_poisson:
       подавление шума через обобщенное преобразование Энскомба (VST) + BM3D 
       перед запуском слепой деконволюции.
    5. Иначе (сильный гауссовский шум): использование оцененной sigma в
       вариационных параметрах деконволюции (по статье: lam = sigma**2).
       Затем ЛИБО:
           - если sigma >= sigma_ringing И включено apply_ringing_on_heavy:
             финальный шаг заменяется на 'ringing_removal' (без pre_nb фильтрации).
           - если пользовательский pre_nonblind не задан:
             включается pre_nonblind='bm3d' перед финальной неслепой деконволюцией.
           (Исключающее ИЛИ: 'ringing_removal' никогда не комбинируется с 'pre_nb').

Остальные параметры ядра алгоритма (epsilon_min, beta_v, beta_H, K1, K2, xh_iter, 
firls_lambda и т.д.) оркестратором не изменяются, так как они не зависят от уровня шума.

Литература:
[1] X. Zhou, M. Vega, F. Zhou, R. Molina, A. K. Katsaggelos,
    "Fast Bayesian Blind Deconvolution with Huber Super Gaussian Priors",
    Digital Signal Processing, 2017.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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

CallbackType = Optional[Callable[[Dict[str, Any]], None]]


class FBDHSGP(DeconvolutionAlgorithm):
    """
    Алгоритм быстрой байесовской слепой деконволюции с хуберовскими 
    супергауссовскими априорными распределениями.

    Базовые параметры алгоритма (не изменяются оркестратором, за исключением sigma)
    --------------------------------------------------------------------
    kernel_size, sigma, epsilon_min, beta_v, beta_H, K1, K2, xh_iter,
    h_iter, delta, delta_x, x_warm_start, gamma_correct, prior_name,
    prior_alpha, firls_*

    Ортогональный конвейер шумоподавления (не изменяется оркестратором)
    --------------------------------------------------------------
    impulse_preprocess  : 'none' | 'auto'
    noise_estimation    : 'none' | 'pca' | 'chen' (при auto_mode='robust' 
                          переключается на 'pca')
    screenot_preprocess : 'none' | 'auto'
    act_preprocess      : 'none' | 'auto'
    noise_preprocess    : 'none' | 'auto' | 'notch' | 'bandstop'
    histogram_eq        : 'none' | 'clahe' | 'global'
    preprocess          : пространственный шумоподавитель ДО слепого цикла
                          ('none' | 'tv' | 'nlm' | 'bilateral'
                           | 'guided' | 'bm3d' | 'act' | 'vst_bm3d')

    Параметры, управляемые оркестратором (при auto_mode='robust')
    -------------------------------------------------------
    pre_nonblind        : те же опции, что и у preprocess; применяется к 
                          ИСХОДНОМУ изображению перед финальным неслепым шагом.
    final_nb            : метод финальной неслепой деконволюции
                          'none' | 'frils' | 'adaptive_lp'
                          | 'ringing_removal' | 'wiener' | 'tikhonov'
                          (по умолчанию 'frils' — оригинальный шаг из статьи)
    nb_params           : словарь параметров для выбранного метода неслепого шага

    Надежный оркестратор
    -------------------
    auto_mode           : 'off' | 'robust' (по умолчанию 'off')
    auto_mode_params    : словарь параметров оркестратора, значения по умолчанию:
        sigma_clean             = 0.005
        sigma_ringing           = 0.05
        apply_ringing_on_heavy  = False
        prefer_vst_on_poisson   = True

    Функции обратного вызова (Callbacks)
    ---------
    iter_callback   : функция(dict) | None
                      Вызывается из ss_deb (внешние итерации) и x_admm_ubc_bi 
                      (внутренние итерации IRLS).
    collect_history : bool
                      Если True, сохраняет события коллбека в self.history.
    """

    def __init__(
        self,
        # --- Базовые параметры алгоритма ---
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
        # --- Ортогональный конвейер шумоподавления ---
        impulse_preprocess: str = "none",
        impulse_params: dict | None = None,
        noise_estimation: str = "none",
        screenot_preprocess: str = "none",
        screenot_params: dict | None = None,
        act_preprocess: str = "none",
        act_params: dict | None = None,
        noise_preprocess: str = "none",
        noise_preprocess_params: dict | None = None,
        histogram_eq: str = "none",
        histogram_eq_params: dict | None = None,
        preprocess: str = "none",
        preprocess_params: dict | None = None,
        # --- Управляемые оркестратором ---
        pre_nonblind: str = "none",
        pre_nonblind_params: dict | None = None,
        final_nb: str = "frils",
        nb_params: dict | None = None,
        # --- Переключатель оркестратора ---
        auto_mode: str = "off",
        auto_mode_params: dict | None = None,
        # --- Коллбеки ---
        iter_callback: CallbackType = None,
        collect_history: bool = False,
        visualize: bool = False,
        # --- Порог ядра ---
        kernel_threshold: float = 0.0,
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

        self.impulse_preprocess = impulse_preprocess
        self.impulse_params = impulse_params
        self.noise_estimation = noise_estimation
        self.screenot_preprocess = screenot_preprocess
        self.screenot_params = screenot_params
        self.act_preprocess = act_preprocess
        self.act_params = act_params
        self.noise_preprocess = noise_preprocess
        self.noise_preprocess_params = noise_preprocess_params
        self.histogram_eq = histogram_eq
        self.histogram_eq_params = histogram_eq_params
        self.preprocess = preprocess
        self.preprocess_params = preprocess_params

        self.pre_nonblind = pre_nonblind
        self.pre_nonblind_params = pre_nonblind_params
        self.final_nb = (final_nb or "none").lower()
        self.nb_params = nb_params

        self.auto_mode = (auto_mode or "off").lower()
        self.auto_mode_params = auto_mode_params

        self.iter_callback = iter_callback
        self.collect_history = bool(collect_history)
        self.visualize = bool(visualize)
        self.kernel_threshold = float(kernel_threshold)

        # Снимок для чистого восстановления состояния оркестратором.
        self._defaults_snapshot = {
            "sigma": self.sigma,
            "preprocess": preprocess,
            "preprocess_params": preprocess_params,
            "pre_nonblind": pre_nonblind,
            "pre_nonblind_params": pre_nonblind_params,
            "final_nb": self.final_nb,
            "nb_params": nb_params,
        }

        self.history: Dict[str, list] = {"x_admm": [], "ss_deb": []}
        self.hyperparams: Dict[str, Any] = {}

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Основная точка входа алгоритма."""
        start_time = time.time()

        # --- 1. Нормализация в оттенки серого float64 [0, 1] ---
        y = np.asarray(image).astype(np.float64)
        if y.max() > 1.0:
            y = y / 255.0
        if y.ndim == 3:
            if y.shape[2] == 1:
                y = y[:, :, 0]
            elif y.shape[2] == 3:
                y = (
                    0.2989 * y[:, :, 0]
                    + 0.5870 * y[:, :, 1]
                    + 0.1140 * y[:, :, 2]
                )
        f_raw = y.copy()  # Копия без изменений для ветви pre_nonblind

        # --- 2. Гамма-коррекция ---
        if self.gamma_correct != 1.0:
            y = np.power(y, self.gamma_correct)

        # --- 3. Обнаружение и удаление импульсного шума ---
        impulse_info = None
        impulse_method = self.impulse_preprocess
        if self.auto_mode == "robust" and impulse_method == "none":
            impulse_method = "auto"
        if impulse_method == "auto":
            from blinddeconv.algorithms.mod_denoise.impulse_noise_estimation import (
                detect_impulse_noise, adaptive_median_filter,
            )
            ip = self.impulse_params or {}
            impulse_info = detect_impulse_noise(
                y,
                density_threshold=ip.get("density_threshold", 0.0005),
                outlier_threshold=ip.get("outlier_threshold", 0.08),
                outlier_window=ip.get("outlier_window", 5),
            )
            if impulse_info["has_impulse"]:
                if self.visualize:
                    print(f"[FBDHSGP] Плотность импульсного шума="
                          f"{impulse_info['density']:.4f} -> медианный фильтр")
                y = adaptive_median_filter(
                    y, impulse_info["impulse_mask"],
                    max_window=ip.get("max_window", 7))
                f_raw = y.copy()

        # --- 4. Оценка шума ---
        if self.auto_mode == "robust" and self.noise_estimation == "none":
            self.noise_estimation = "pca"
            if self.visualize:
                print("[FBDHSGP] auto_mode='robust' -> noise_estimation='pca'")
        noise_info = None
        if self.noise_estimation != "none":
            noise_info = self._estimate_noise(y)
            if self.visualize and noise_info is not None:
                print(f"[FBDHSGP] шум: sigma_norm="
                      f"{noise_info.get('sigma_norm', 0):.5f} "
                      f"a={noise_info.get('a', 0):.4g} "
                      f"b={noise_info.get('b', 0):.4g}")

        # --- 5. Ортогональные шумоподавители ---
        screenot_info = None
        if self.screenot_preprocess == "auto":
            if self.act_preprocess == "auto":
                raise ValueError(
                    "screenot_preprocess и act_preprocess не могут "
                    "быть 'auto' одновременно.")
            from blinddeconv.algorithms.mod_denoise.screenot import screenot_denoise
            sp = self.screenot_params or {}
            y, screenot_info = screenot_denoise(
                y,
                k=sp.get("k", 10),
                strategy=sp.get("strategy", "i"),
                mode=sp.get("mode", "full"),
                patch_size=sp.get("patch_size", 8),
                stride=sp.get("stride", 3),
            )

        act_info = None
        if self.act_preprocess == "auto":
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            ap = self.act_params or {}
            nv = ap.get("noise_var", None)
            if nv is None and noise_info is not None:
                nv = noise_info.get("sigma_norm", 0.0) ** 2
            y, act_info = act_denoise(
                y, noise_var=nv,
                threshold_setting=ap.get("threshold_setting", "s"))

        psd_info = None
        if self.noise_preprocess != "none":
            y, psd_info = self._apply_noise_preprocess(y)

        if self.histogram_eq not in (None, "none"):
            y = self._apply_histogram_eq(y)

        # --- 6. Надежный оркестратор ---
        # Выполняет спектральный анализ для выбора между BM3D и ACT 
        # при наличии коррелированного шума.
        psd_for_orch = psd_info
        if (
            self.auto_mode == "robust"
            and psd_for_orch is None
            and bool((self.auto_mode_params or {}).get("correlated_check", True))
        ):
            try:
                from blinddeconv.algorithms.mod_denoise.noise_psd_analysis import analyze_noise_psd
                psd_for_orch = analyze_noise_psd(y)
                if self.visualize:
                    print(f"[FBDHSGP] PSD: класс={psd_for_orch.get('noise_class')} "
                          f"скоррелирован={psd_for_orch.get('is_correlated')} "
                          f"beta={psd_for_orch.get('beta', 0):.2f}")
            except Exception as exc:
                if self.visualize:
                    print(f"[FBDHSGP] Ошибка анализа PSD: {exc}")
                psd_for_orch = None
        orchestrator_info = self._orchestrate_robust(noise_info, psd_for_orch)

        # --- 7. VST шумоподавление ---
        vst_info = None
        if orchestrator_info.get("apply_vst", False):
            from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise
            y, vst_info = vst_bm3d_denoise(
                y, noise_info=noise_info, verbose=self.visualize)
            f_raw = y.copy()
            if self.visualize:
                print(f"[FBDHSGP] Применен VST: режим={vst_info.get('mode')}")

        # --- 8. Пространственное шумоподавление до слепого цикла ---
        if self.preprocess not in (None, "none"):
            y = self._apply_denoise(y, self.preprocess,
                                    self.preprocess_params, noise_info)

        # --- 9. Многомасштабная слепая деконволюция ---
        sigma_for_bid = orchestrator_info.get("sigma_bid", self.sigma)
        kernel = self._multiscale_bid(y, sigma_override=sigma_for_bid)

        # --- 10. Пороговая обработка ядра ---
        if self.kernel_threshold > 0.0:
            k_max = kernel.max()
            if k_max > 0:
                kernel[kernel < self.kernel_threshold * k_max] = 0.0
                k_sum = kernel.sum()
                if k_sum > 0:
                    kernel /= k_sum

        # --- 11. Финальный неслепой шаг ---
        x_final = self._run_final_nb(
            f_raw, y, kernel, noise_info, sigma_for_bid)

        # --- 12. Сбор результатов ---
        self.hyperparams = {
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "sigma_bid": sigma_for_bid,
            "epsilon_min": self.epsilon_min,
            "beta_v": self.beta_v,
            "beta_H": self.beta_H,
            "K1": self.K1,
            "K2": self.K2,
            "xh_iter": self.xh_iter,
            "h_iter": self.h_iter,
            "prior_name": self.prior_name,
            "prior_alpha": self.prior_alpha,
            "impulse_preprocess": impulse_method,
            "impulse_info": (
                {k: v for k, v in (impulse_info or {}).items()
                 if k != "impulse_mask"} if impulse_info else None
            ),
            "noise_estimation": self.noise_estimation,
            "noise_info": noise_info,
            "screenot_preprocess": self.screenot_preprocess,
            "screenot_info": screenot_info,
            "act_preprocess": self.act_preprocess,
            "act_info": act_info,
            "noise_preprocess": self.noise_preprocess,
            "psd_info": (
                {k: v for k, v in (psd_info or {}).items() if k != "psd_2d"}
                if psd_info else None
            ),
            "histogram_eq": self.histogram_eq,
            "preprocess": self.preprocess,
            "pre_nonblind": self.pre_nonblind,
            "final_nb": self.final_nb,
            "nb_params": self.nb_params,
            "auto_mode": self.auto_mode,
            "auto_mode_params": self.auto_mode_params,
            "orchestrator_info": orchestrator_info,
            "vst_info": vst_info,
            "time": time.time() - start_time,
        }

        x_final = np.clip(x_final, 0.0, 1.0) * 255.0
        x_final = np.clip(x_final, 0, 255).astype(np.int16)
        return x_final, kernel

    def _orchestrate_robust(
        self,
        noise_info: Optional[dict],
        psd_info: Optional[dict] = None,
    ) -> dict:
        """Политика настройки оркестратора конвейера шума."""
        info: Dict[str, Any] = {
            "triggered": False,
            "mode": self.auto_mode,
            "branch": "off",
            "apply_vst": False,
            "sigma_bid": self.sigma,
        }

        if self.auto_mode != "robust":
            return info

        snap = self._defaults_snapshot
        p = self.auto_mode_params or {}
        sigma_clean = float(p.get("sigma_clean", 0.005))
        sigma_ringing = float(p.get("sigma_ringing", 0.05))
        apply_ringing = bool(p.get("apply_ringing_on_heavy", False))
        prefer_vst = bool(p.get("prefer_vst_on_poisson", True))
        correlated_check = bool(p.get("correlated_check", True))
        correlated_use_act_preblind = bool(
            p.get("correlated_use_act_preblind", True))

        sigma_n = 0.0
        if noise_info is not None:
            sigma_n = float(noise_info.get("sigma_norm", 0.0) or 0.0)
        a_pg = 0.0
        if noise_info is not None:
            a_pg = float(noise_info.get("a", 0.0) or 0.0)

        info.update({
            "triggered": True,
            "sigma": sigma_n,
            "a_pg": a_pg,
            "sigma_clean": sigma_clean,
            "sigma_ringing": sigma_ringing,
            "apply_ringing_on_heavy": apply_ringing,
            "prefer_vst_on_poisson": prefer_vst,
        })

        # --- ЧИСТОЕ: восстанавливаем исходные настройки ---
        if sigma_n < sigma_clean:
            info["branch"] = "clean"
            self.sigma = snap["sigma"]
            self.preprocess = snap["preprocess"]
            self.preprocess_params = snap["preprocess_params"]
            self.pre_nonblind = snap["pre_nonblind"]
            self.pre_nonblind_params = snap["pre_nonblind_params"]
            self.final_nb = snap["final_nb"]
            self.nb_params = snap["nb_params"]
            info["sigma_bid"] = self.sigma
            if self.visualize:
                print(f"[FBDHSGP][orch] чистое (sigma={sigma_n:.5f}) "
                      "-> восстановлены настройки по умолчанию")
            return info

        # --- ЗАШУМЛЕННОЕ (выбор параметров на основе sigma) ---
        info["branch"] = "robust"

        poisson_like = (a_pg > 1e-6) and prefer_vst
        info["poisson_like"] = poisson_like
        if poisson_like:
            info["apply_vst"] = True

        sigma_eff = max(sigma_n, snap["sigma"])
        self.sigma = sigma_eff
        info["sigma_bid"] = sigma_eff

        is_correlated = False
        noise_class = "unknown"
        if psd_info is not None:
            is_correlated = bool(psd_info.get("is_correlated", False)) \
                or bool(psd_info.get("has_periodic", False))
            noise_class = str(psd_info.get("noise_class", "unknown"))
        info["is_correlated"] = is_correlated
        info["noise_class"] = noise_class

        heavy = sigma_n >= sigma_ringing
        if heavy and apply_ringing:
            info["route"] = "ringing"
            self.final_nb = "ringing_removal"
            self.nb_params = None
            self.pre_nonblind = "none"
            self.pre_nonblind_params = None
        else:
            info["route"] = "pre_nb"
            self.final_nb = snap["final_nb"]
            self.nb_params = snap["nb_params"]
            if snap["pre_nonblind"] in (None, "none"):
                if is_correlated:
                    self.pre_nonblind = "act"
                    self.pre_nonblind_params = {
                        "noise_var": sigma_eff ** 2,
                        "threshold_setting": "s",
                    }
                    info["route"] = "pre_nb_act"
                else:
                    self.pre_nonblind = "bm3d"
                    self.pre_nonblind_params = {"sigma": sigma_eff}
            else:
                self.pre_nonblind = snap["pre_nonblind"]
                self.pre_nonblind_params = snap["pre_nonblind_params"]

            if (
                is_correlated
                and correlated_use_act_preblind
                and snap["preprocess"] in (None, "none")
            ):
                self.preprocess = "act"
                self.preprocess_params = {
                    "noise_var": sigma_eff ** 2,
                    "threshold_setting": "s",
                }
                info["preblind_act"] = True

        if self.visualize:
            print(f"[FBDHSGP][orch] маршрут={info['route']} "
                  f"sigma_n={sigma_n:.5f} a={a_pg:.4g} "
                  f"vst={info['apply_vst']} "
                  f"final_nb={self.final_nb} "
                  f"pre_nb={self.pre_nonblind} "
                  f"sigma_bid={sigma_eff:.5f}")
        return info

    def _multiscale_bid(
        self,
        y: np.ndarray,
        sigma_override: float | None = None,
    ) -> np.ndarray:
        """Многомасштабная слепая деконволюция."""
        blur_size = self.kernel_size
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

        xvars, hvars = self._init_vars(sigma_override=sigma_override)

        cb = self._make_callback()
        if cb is not None:
            xvars["iter_callback"] = cb
            hvars["iter_callback"] = cb

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

            ls[s], ks[s] = shift_kernel_img_space(ls[s], ks[s])

            hks1 = k1 // 2
            hks2 = k2 // 2

            xvars["x0"] = pad_replicate(ls[s], hks1, hks2)
            xvars["scale"] = s
            for key in ("RR", "cov_img", "dvx", "dvy", "ye", "X", "H"):
                xvars.pop(key, None)
            for key in ("H", "dH"):
                hvars.pop(key, None)

            hvars["h"] = ks[s]
            hvars["scale_ratio"] = (k1 * k2) / (
                ksize[num_scales - 1][0] * ksize[num_scales - 1][1]
            )

            ls[s], ks[s] = ss_deb(ys, xvars, hvars)

        kernel = ks[num_scales - 1]
        kernel = np.where(kernel < 0, 0.0, kernel)
        s_sum = kernel.sum()
        if s_sum > 0:
            kernel = kernel / s_sum

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

    def _run_final_nb(
        self,
        f_raw: np.ndarray,
        f_after_pipeline: np.ndarray,
        kernel: np.ndarray,
        noise_info: Optional[dict],
        sigma_eff: float,
    ) -> np.ndarray:
        """Диспетчеризация финального неслепого шага."""
        method = (self.final_nb or "none").lower()

        if method == "none":
            method = "frils"

        f_in = f_raw
        if self.pre_nonblind not in (None, "none"):
            f_in = self._apply_denoise(
                f_in, self.pre_nonblind, self.pre_nonblind_params, noise_info)
            if self.visualize:
                print(f"[FBDHSGP] pre_nonblind={self.pre_nonblind}")

        if method == "frils":
            opts = self._default_firls_opts()
            return frils_deb_ubc(f_in, kernel, opts)

        if method == "ringing_removal":
            from blinddeconv.algorithms.mod_denoise.non_blind import ringing_removal
            nbp = self.nb_params or {}
            return ringing_removal(
                f_in, kernel,
                lambda_tv=float(nbp.get("lambda_tv", 3e-3)),
                lambda_l0=float(nbp.get("lambda_l0", 5e-4)),
                weight_ring=float(nbp.get("weight_ring", 1.0)),
            )

        if method == "adaptive_lp":
            from blinddeconv.algorithms.mod_denoise.non_blind import adaptive_lp_deconv
            nbp = self.nb_params or {}
            sigma_n = (noise_info or {}).get("sigma_norm", None)
            return adaptive_lp_deconv(
                f_in, kernel,
                alpha=float(nbp.get("alpha", 0.8)),
                sigma_n=sigma_n,
                two_stage=bool(nbp.get("two_stage", True)),
            )

        if method == "wiener":
            nbp = self.nb_params or {}
            return self._wiener_filter(
                f_in, kernel, float(nbp.get("noise_snr", 0.01)))

        if method == "tikhonov":
            nbp = self.nb_params or {}
            return self._tikhonov_filter(
                f_in, kernel, float(nbp.get("alpha", 0.01)))

        raise ValueError(
            f"Неизвестный метод final_nb='{method}'. Варианты: 'none', 'frils', "
            "'ringing_removal', 'adaptive_lp', 'wiener', 'tikhonov'.")

    @staticmethod
    def _wiener_filter(b, k, noise_snr):
        H, W = b.shape
        K = np.fft.fft2(k, s=(H, W))
        B = np.fft.fft2(b)
        K_conj = np.conj(K)
        return np.real(np.fft.ifft2(K_conj * B / (np.abs(K) ** 2 + noise_snr)))

    @staticmethod
    def _tikhonov_filter(b, k, alpha):
        H, W = b.shape
        K = np.fft.fft2(k, s=(H, W))
        B = np.fft.fft2(b)
        cy = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)
        L = np.fft.fft2(cy, s=(H, W))
        K_conj = np.conj(K)
        return np.real(np.fft.ifft2(
            K_conj * B / (np.abs(K) ** 2 + alpha * np.abs(L) ** 2)
        ))

    def _apply_denoise(self, img, method, params, noise_info):
        """Универсальный диспетчер вызовов пространственных шумоподавителей."""
        if method is None or method == "none":
            return img
        p = dict(params or {})
        sigma = (noise_info or {}).get("sigma_norm", None) if noise_info else None

        if method == "tv":
            from skimage.restoration import denoise_tv_chambolle
            w = p.get("weight", max(0.01, sigma * 2) if sigma else 0.1)
            return denoise_tv_chambolle(img, weight=w)

        if method == "nlm":
            from skimage.restoration import (
                denoise_nl_means, estimate_sigma as _est_sig)
            sig = p.get("sigma", sigma)
            if sig is None:
                sig = float(np.mean(_est_sig(img)))
            h = p.get("h", 0.8 * sig)
            return denoise_nl_means(
                img, h=h,
                patch_size=p.get("patch_size", 5),
                patch_distance=p.get("patch_distance", 6),
                fast_mode=True)

        if method == "bilateral":
            import cv2
            d = p.get("d", 5)
            sc = p.get("sigma_color", sigma if sigma else 0.1)
            ss = p.get("sigma_space", 5.0)
            return cv2.bilateralFilter(
                img.astype(np.float32), d, float(sc), float(ss)
            ).astype(np.float64)

        if method == "guided":
            r = p.get("radius", 4)
            eps = p.get("eps", sigma ** 2 * 4 if sigma else 0.01)
            return self._guided_filter(img, img, r, eps)

        if method == "bm3d":
            import bm3d as bm3d_lib
            sig = p.get("sigma", sigma if sigma else 0.05)
            return bm3d_lib.bm3d(img, sigma_psd=sig)

        if method == "act":
            from blinddeconv.algorithms.mod_denoise.act_denoise import act_denoise
            nv = p.get("noise_var", None)
            if nv is None and sigma is not None:
                nv = sigma ** 2
            ts = p.get("threshold_setting", "s")
            result, _ = act_denoise(img, noise_var=nv, threshold_setting=ts)
            return result

        if method == "vst_bm3d":
            from blinddeconv.algorithms.mod_denoise.vst import vst_bm3d_denoise
            result, _ = vst_bm3d_denoise(
                img, noise_info=noise_info, verbose=self.visualize)
            return result

        raise ValueError(
            f"Неизвестный шумоподавитель='{method}'. Доступные варианты: "
            "'tv', 'nlm', 'bilateral', 'guided', 'bm3d', 'act', "
            "'vst_bm3d', 'none'")

    @staticmethod
    def _guided_filter(I, p, r, eps):
        from scipy.ndimage import uniform_filter
        size = 2 * r + 1

        def box(x):
            return uniform_filter(x, size=size, mode="reflect")

        mean_I = box(I)
        mean_p = box(p)
        corr_Ip = box(I * p)
        var_I = box(I * I) - mean_I ** 2
        cov_Ip = corr_Ip - mean_I * mean_p
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        return box(a) * I + box(b)

    def _estimate_noise(self, yg):
        if self.noise_estimation == "chen":
            from blinddeconv.algorithms.mod_denoise.chen_noise_estimate import estimate_noise_level
            sigma = estimate_noise_level(yg)
            return {"method": "chen", "sigma_norm": sigma,
                    "sigma": sigma * 255.0}
        if self.noise_estimation == "pca":
            from blinddeconv.algorithms.mod_denoise.pyatykh_noise_reconstruction import estimate_noise_params
            result = estimate_noise_params(yg)
            result["method"] = "pca"
            return result
        return None

    def _apply_noise_preprocess(self, yg):
        from blinddeconv.algorithms.mod_denoise.noise_psd_analysis import (
            analyze_noise_psd, noise_preprocess as _npp,
        )
        npp = self.noise_preprocess_params or {}
        psd_info = analyze_noise_psd(yg)
        method = self.noise_preprocess
        if method == "auto":
            if psd_info.get("has_periodic", False):
                method = "notch"
            elif psd_info.get("color_label", "white") in ("pink", "brown"):
                method = "bandstop"
            else:
                return yg, psd_info
        return _npp(yg, method, npp), psd_info

    def _apply_histogram_eq(self, yg):
        method = self.histogram_eq
        hp = self.histogram_eq_params or {}
        yg_clipped = np.clip(yg, 0, 1)
        if method == "clahe":
            from skimage.exposure import equalize_adapthist
            return equalize_adapthist(
                yg_clipped,
                clip_limit=hp.get("clip_limit", 0.01),
                nbins=hp.get("nbins", 256))
        if method == "global":
            from skimage.exposure import equalize_hist
            return equalize_hist(yg_clipped)
        raise ValueError(
            f"Неизвестный параметр histogram_eq='{method}'. Доступно: 'clahe', 'global' или 'none'.")

    def _init_vars(
        self,
        sigma_override: float | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sigma_use = (
            float(sigma_override) if sigma_override is not None
            else self.sigma
        )
        prior = {
            "name": self.prior_name,
            "alpha": self.prior_alpha,
            "conv": 0,
            "const_weight": 0,
        }
        xvars: Dict[str, Any] = {
            "sigma": sigma_use,
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
            "sigma": sigma_use,
        }
        return xvars, hvars

    def _default_firls_opts(self) -> Dict[str, float]:
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

    def _make_callback(self) -> CallbackType:
        user_cb = self.iter_callback
        framework_cb = self._callback
        if user_cb is None and framework_cb is None and not self.collect_history:
            return None

        _meta = {'scope', 'iter', 'kernel', 'image'}

        def _wrapper(event: Dict[str, Any]) -> None:
            if self.collect_history:
                scope = event.get("scope", "unknown")
                bucket = self.history.setdefault(scope, [])
                bucket.append(dict(event))
            if user_cb is not None:
                user_cb(event)
            if framework_cb is not None and event.get("scope") == "ss_deb":
                try:
                    framework_cb({
                        'iteration':  event['iter'],
                        'scale':      0,
                        'num_scales': 1,
                        'kernel':     event.get('kernel'),
                        'image':      event.get('image', None),
                        'metrics':    {k: v for k, v in event.items()
                                       if k not in _meta},
                    })
                except Exception:
                    pass

        return _wrapper

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
            ("impulse_preprocess", self.impulse_preprocess),
            ("noise_estimation", self.noise_estimation),
            ("screenot_preprocess", self.screenot_preprocess),
            ("act_preprocess", self.act_preprocess),
            ("noise_preprocess", self.noise_preprocess),
            ("histogram_eq", self.histogram_eq),
            ("preprocess", self.preprocess),
            ("pre_nonblind", self.pre_nonblind),
            ("final_nb", self.final_nb),
            ("nb_params", self.nb_params),
            ("auto_mode", self.auto_mode),
            ("auto_mode_params", self.auto_mode_params),
            ("collect_history", self.collect_history),
            ("visualize", self.visualize),
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