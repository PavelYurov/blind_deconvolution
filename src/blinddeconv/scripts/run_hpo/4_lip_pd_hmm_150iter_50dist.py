"""
Оптимизация гиперпараметров алгоритма LIP-BD (Primal-Dual)
с использованием HMM.

Поиск оптимальных гиперпараметров методом MCMC-FMP (MCMC — Factorized Mixture Proposal)
и параллельного построения 3D Парето-фронта для анализа компромиссов при слепой деконволюции.

Оси Парето-фронта:
    X (Сложность смаза) : метрика только размытого изображения (минимизируется).
    Y (Вклад шума)      : разница метрик размытого и зашумленного изображений (минимизируется).
    Z (Качество)        : абсолютная метрика восстановленного изображения (максимизируется).

Целевая функция MCMC-FMP базируется на вычислении групповой медианы SSIM (с равным весом
для чистых и зашумленных изображений) с применением штрафов за вырожденные решения
(дельта-ядро, потеря дисперсии).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")

import time
import warnings
import json
import numpy as np
import cv2 as cv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.signal import fftconvolve
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label
from scipy.stats import entropy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# --- Настройка путей ---
HPO_RL_ROOT       = Path(__file__).resolve().parent
BLIND_DECONV_ROOT = HPO_RL_ROOT.parent / "blind_deconvolution"
BLIND_DECONV_SRC  = BLIND_DECONV_ROOT / "src"
DATASET_ROOT      = HPO_RL_ROOT / "middle_data_pictures"

for _p in [str(BLIND_DECONV_SRC), str(HPO_RL_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from blinddeconv.algorithms.blind_deconvolution.our_company.logarithmic_pds.lip_denoise.lip import LIP_BD
from hpo_rl.experiments.run_experiment import run_n_experiments
import os
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
# --- Глобальные настройки ---

# Максимальный размер стороны изображения (None = без изменения масштаба)
MAX_IMAGE_SIDE   = None

# Директория для Парето-фронтов
PARETO_OUTPUT_DIR = HPO_RL_ROOT / "4_lip_pd_hmm_150iter_50dist"

# Углы обзора для 4 ракурсов PNG (elev, azim)
_VIEW_ANGLES = [
    (30, -60),   # _1: вид по умолчанию
    (30, 30),    # _2: поворот на 90°
    (30, 120),   # _3: поворот на 180°
    (30, 210),   # _4: поворот на 270°
]

# Известные типы шума в именах файлов датасета
_KNOWN_NOISE_TYPES = frozenset({
    "gaussian", "gauss", "poisson", "brown", "pink", "impulse", "saltpepper",
})


# --- Пространство поиска гиперпараметров (HP Space) ---
# Обоснование диапазонов:
# lambda_val (β) : Вес data-fidelity. Отвечает за баланс между резкостью (высокие значения) и подавлением шума (низкие значения).
# tau (ε)        : Параметр нижней границы log-prior. Управляет разреженностью градиентов (чем меньше, тем жестче штраф за мелкие детали/шум).
# outer_iters    : Количество внешних итераций (шагов MM - Majorization-Minimization) на одном масштабе пирамиды.
# inner_iters    : Количество внутренних Primal-Dual шагов Конда-Вю для оценки изображения.
# lambda_mult / scale_mult : Управление coarse-to-fine пирамидой. Определяют, как быстро меняется вес регуляризации и размер ядра при переходе к мелкому масштабу.
# kernel_threshold: Порог отсечения шума в оцененном ядре. Убирает "хвосты" PSF.

LIP_HP_SPACE: Dict[str, Dict[str, Any]] = {
    # ── Размер ядра ──
    "kernel_shape":     {"type": "categorical", "values": [(15,15), (21,21), (27,27), (31,31), (35,35), (41,41), (51,51), (61,61)]},

    # ── Core параметры blind deconvolution (Primal-Dual) ──
    "lambda_val":       {"type": "float",       "values": [1e3, 5e4]},         # Вес data-fidelity (β)
    "tau":              {"type": "float",       "values": [1e-4, 5e-2]},       # Параметр сглаживания log-prior (ε)
    "outer_iters":      {"type": "int",         "values": [30, 140]},          # Внешние MM-итерации на масштаб
    "inner_iters":      {"type": "int",         "values": [3, 20]},            # Внутренние PD-шаги Конда-Вю
    
    # ── Параметры пирамиды (Coarse-to-fine) ──
    "lambda_mult":      {"type": "float",       "values": [1.5, 3.0]},         # Множитель λ между уровнями пирамиды
    "scale_mult":       {"type": "float",       "values": [1.2, 2.0]},         # Коэффициент масштабирования ядра

    # ── Постобработка ядра и изображения ──
    "kernel_threshold": {"type": "float",       "values": [0.01, 0.25]},       # Порог отсечения шума в PSF (доля от максимума)
    "gamma_correction": {"type": "categorical", "values": [0, 1]},             # Флаг применения гамма-коррекции

    # ── Параметры non-blind шага восстановления (Ringing removal) ──
    "lambda_tv":        {"type": "float",       "values": [1e-4, 5e-2]},       # Вес TV-регуляризации
    "lambda_l0":        {"type": "float",       "values": [1e-4, 1e-2]},       # Вес L0-градиента
    "weight_ring":      {"type": "float",       "values": [0.0, 2.0]},         # Сила подавления артефактов (0 = только TV)

    # ── Noise pipeline (вкл/выкл денойзеров) ──
    "impulse_preprocess": {"type": "categorical", "values": ["none", "auto"]},
    "noise_estimation":   {"type": "categorical", "values": ["none", "chen", "pca"]},
    "act_preprocess":     {"type": "categorical", "values": ["none", "auto"]},
    "preprocess":         {"type": "categorical", "values": ["none", "bm3d", "nlm", "bilateral", "guided", "tv"]},
    "noise_preprocess":   {"type": "categorical", "values": ["none", "auto"]},
    "blind_denoise":      {"type": "categorical", "values": ["none", "guided", "bilateral", "tv"]},
    "pre_nonblind":       {"type": "categorical", "values": ["none", "bm3d", "bilateral", "guided", "tv"]},
}


# --- Структуры данных и глобальное состояние ---
@dataclass
class TestCase:
    """Структура для хранения тестового примера (изображения на разных стадиях деградации)."""
    name: str
    original:  np.ndarray   # чистое
    distorted: np.ndarray   # смазанное и зашумлённое
    blur_only: np.ndarray   # только смазанное (без шума)
    kernel_name: str
    noise_type:  str        # "" для изображений без шума
    has_noise:   bool


_PARETO_STORE: List[Dict[str, Any]] = []
_EVAL_COUNTER = [0]
_TEST_CASES: Optional[List[TestCase]] = None


# --- Утилиты загрузки и подготовки данных ---
def _parse_distorted_filename(
    stem: str,
    original_stems: List[str],
    kernel_names: List[str],
) -> Optional[Tuple[str, str, str]]:
    """Извлечение имён оригинала, ядра и типа шума (orig, kernel, noise) из составного имени файла «original_kernel_noise»."""
    for orig in sorted(original_stems, key=len, reverse=True):
        prefix = orig + "_"
        if not stem.startswith(prefix):
            continue
        remainder = stem[len(prefix):]
        for kname in sorted(kernel_names, key=len, reverse=True):
            if not remainder.startswith(kname):
                continue
            noise_part = remainder[len(kname):]
            if noise_part == "":
                return (orig, kname, "")
            if noise_part.startswith("_"):
                noise = noise_part[1:]
                if noise == "" or noise in _KNOWN_NOISE_TYPES:
                    return (orig, kname, noise)
    return None


def _load_kernel(kernel_name: str) -> Optional[np.ndarray]:
    """Загрузка матрицы ядра (PSF) из ground_truth_filters/{name}_kernel.png."""
    path = DATASET_ROOT / "ground_truth_filters" / f"{kernel_name}_kernel.png"
    if not path.exists():
        return None
    k = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if k is None:
        return None
    kf = k.astype(np.float64)
    kf /= kf.sum() + 1e-12
    return kf


def _make_blur_only(original_f: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Свёртка оригинала с ядром - изображение только со смазом."""
    blurred = fftconvolve(original_f, kernel, mode="same")
    return np.clip(blurred, 0.0, 1.0)


def _resize_if_needed(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    """Пропорциональное уменьшение изображения, если его максимальная сторона превышает лимит."""
    if max_side is None:
        return image
    h, w = image.shape[:2]
    if max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv.resize(image, (int(w * scale), int(h * scale)),
                     interpolation=cv.INTER_AREA)


def load_test_cases() -> List[TestCase]:
    """Загрузка набора данных, формируя тестовые случаи с оригиналами и искажениями."""
    originals_dir = DATASET_ROOT / "originals"
    distorted_dir = DATASET_ROOT / "distorted"
    filters_dir   = DATASET_ROOT / "ground_truth_filters"

    original_stems = [p.stem for p in sorted(originals_dir.glob("*.png"))]
    kernel_names   = [p.stem.replace("_kernel", "")
                      for p in sorted(filters_dir.glob("*_kernel.png"))]

    cases: List[TestCase] = []
    skipped: List[str] = []

    for dist_path in sorted(distorted_dir.glob("*.png")):
        stem = dist_path.stem
        parsed = _parse_distorted_filename(stem, original_stems, kernel_names)
        if parsed is None:
            skipped.append(stem)
            continue

        orig_stem, kernel_name, noise_type = parsed
        has_noise = noise_type != ""

        orig_path = originals_dir / f"{orig_stem}.png"
        original  = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
        distorted = cv.imread(str(dist_path), cv.IMREAD_GRAYSCALE)
        if original is None or distorted is None:
            skipped.append(f"{stem} (imread)")
            continue

        original  = _resize_if_needed(original,  MAX_IMAGE_SIDE)
        distorted = _resize_if_needed(distorted, MAX_IMAGE_SIDE)

        # Версия «только смаз»
        if has_noise:
            kernel = _load_kernel(kernel_name)
            if kernel is None:
                skipped.append(f"{stem} (kernel)")
                continue
            orig_f = original.astype(np.float64)
            if orig_f.max() > 1.0:
                orig_f /= 255.0
            blur_f    = _make_blur_only(orig_f, kernel)
            blur_only = (blur_f * 255.0).astype(np.uint8)
        else:
            # Нет шума → искажённое = только смаз
            blur_only = distorted.copy()

        cases.append(TestCase(
            name=stem, original=original, distorted=distorted,
            blur_only=blur_only, kernel_name=kernel_name,
            noise_type=noise_type, has_noise=has_noise,
        ))

    print(f"Загружено {len(cases)} тестовых случаев, пропущено {len(skipped)}: {skipped}")
    return cases


def _get_test_cases() -> List[TestCase]:
    """Синглтон для ленивой загрузки датасета."""
    global _TEST_CASES
    if _TEST_CASES is None:
        _TEST_CASES = load_test_cases()
    return _TEST_CASES


# --- Вычисление метрик (PSNR / SSIM) ---
def _to_f64(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float64)
    if a.max() > 1.0:
        a /= 255.0
    return np.clip(a, 0.0, 1.0)


def _crop_pair(a: np.ndarray, b: np.ndarray):
    """Обрезка изображений до минимальных общих размеров для корректного сравнения."""
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def compute_psnr(original: np.ndarray, image: np.ndarray) -> float:
    o, i = _crop_pair(_to_f64(original), _to_f64(image))
    v = peak_signal_noise_ratio(o, i, data_range=1.0)
    return float(v) if np.isfinite(v) else 0.0


def compute_ssim(original: np.ndarray, image: np.ndarray) -> float:
    o, i = _crop_pair(_to_f64(original), _to_f64(image))
    v = structural_similarity(o, i, data_range=1.0)
    return float(v) if np.isfinite(v) else 0.0


# --- Построение и визуализация Парето-фронта ---
def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Недоминируемые точки.  X,Y минимизируются; Z максимизируется.

    Returns:  boolean mask длины N.
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        mask = np.arange(n) != i
        mask &= is_pareto
        others = points[mask]
        # j доминирует i ⟺ X_j≤X_i ∧ Y_j≤Y_i ∧ Z_j≥Z_i ∧ хотя бы одно строго
        weak = ((others[:, 0] <= points[i, 0]) &
                (others[:, 1] <= points[i, 1]) &
                (others[:, 2] >= points[i, 2]))
        strict = ((others[:, 0] < points[i, 0]) |
                  (others[:, 1] < points[i, 1]) |
                  (others[:, 2] > points[i, 2]))
        if np.any(weak & strict):
            is_pareto[i] = False
    return is_pareto

def compute_pareto_front_nd(points: np.ndarray) -> np.ndarray:
    """
    Недоминируемые точки (Noise Degradation вариант).
    X минимизируется; Y,Z максимизируются.
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        mask = np.arange(n) != i
        mask &= is_pareto
        others = points[mask]
        weak = ((others[:, 0] <= points[i, 0]) &
                (others[:, 1] >= points[i, 1]) &
                (others[:, 2] >= points[i, 2]))
        strict = ((others[:, 0] < points[i, 0]) |
                  (others[:, 1] > points[i, 1]) |
                  (others[:, 2] > points[i, 2]))
        if np.any(weak & strict):
            is_pareto[i] = False
    return is_pareto

def _interpolate_pareto_surface(pareto_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Интерполяция поверхности Парето-фронта (RBF thin-plate spline).
    Поверхность гладкая, проходит через точки и слегка за них.
    Вместо clamping (плоские полки) — NaN для зон где RBF улетает.
    Возвращает сетку Xg, Yg, Zg.
    """
    xy = pareto_pts[:, :2]
    z  = pareto_pts[:, 2]
    z_min, z_max = z.min(), z.max()

    # Сетка с небольшим запасом (5 %), чтобы поверхность чуть выходила за точки
    x_lo, x_hi = xy[:, 0].min(), xy[:, 0].max()
    y_lo, y_hi = xy[:, 1].min(), xy[:, 1].max()
    x_range = x_hi - x_lo if x_hi > x_lo else 1.0
    y_range = y_hi - y_lo if y_hi > y_lo else 1.0
    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range
    xg = np.linspace(x_lo - x_pad, x_hi + x_pad, 200)
    yg = np.linspace(y_lo - y_pad, y_hi + y_pad, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    # RBF-интерполяция (thin-plate spline — гладкая)
    rbf = RBFInterpolator(xy, z, kernel='thin_plate_spline', smoothing=0.0)
    Zg = rbf(grid).reshape(Xg.shape)

    # 1) Мягкая обрезка по расстоянию — сглаженная граница (без зубцов)
    tree = cKDTree(xy)
    dists, _ = tree.query(grid)
    if len(xy) >= 2:
        nn_dists = tree.query(xy, k=2)[0][:, 1]
        clip_dist = nn_dists.max() * 3.0
    else:
        clip_dist = max(x_range, y_range) * 0.5
    diag = np.sqrt(x_range**2 + y_range**2)
    clip_dist = max(clip_dist, 0.35 * diag)
    dist_mask = (dists.reshape(Zg.shape) <= clip_dist).astype(float)
    dist_mask = gaussian_filter(dist_mask, sigma=4.0)
    Zg[dist_mask < 0.5] = np.nan

    # 2) NaN для зон где RBF улетает за диапазон реальных точек (+100 %)
    z_margin = 1.00 * (z_max - z_min) if z_max > z_min else 0.01
    Zg[(Zg < z_min - z_margin) | (Zg > z_max + z_margin)] = np.nan

    # 3) Убираем оторванные «заплатки» — оставляем только крупнейший кусок
    valid = ~np.isnan(Zg)
    labeled, num_features = label(valid)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        Zg[labeled != sizes.argmax()] = np.nan

    # 4) Лёгкое NaN-safe гауссово сглаживание (sigma=0.8)
    valid = ~np.isnan(Zg)
    if valid.any():
        Zg_filled = np.where(valid, Zg, 0.0)
        w = valid.astype(float)
        Zg_num = gaussian_filter(Zg_filled, sigma=0.8)
        Zg_den = gaussian_filter(w, sigma=0.8)
        with np.errstate(invalid='ignore', divide='ignore'):
            Zg[valid] = np.where(Zg_den[valid] > 1e-12,
                                 Zg_num[valid] / Zg_den[valid], Zg[valid])

    return Xg, Yg, Zg

def _plot_pareto_3d_with_points(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    iteration: int,
    out_dir: Path,
) -> None:
    """3D-график с Парето-фронтом: доминируемые точки (синие), точки фронта (красные),
    и интерполированная поверхность (зелёная, полупрозрачная)."""
    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    # Доминируемые точки (синие)
    dominated = ~pareto_mask
    if dominated.any():
        ax.scatter(all_pts[dominated, 0], all_pts[dominated, 1], all_pts[dominated, 2],
                   c="steelblue", alpha=0.20, s=10, label="Доминируемые")
        
    pareto_pts = all_pts[pareto_mask]
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2],
               c="red", alpha=0.9, s=50, edgecolors="darkred", linewidths=0.6,
               label=f"Парето-фронт (точки: {len(pareto_pts)})")

    if len(pareto_pts) >= 4:
        try:
            Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
            ax.plot_surface(Xg, Yg, Zg, alpha=0.55, color="limegreen", edgecolor="none")
        except Exception as e:
            print(f"  [Warning] Ошибка интерполяции поверхности: {e}")

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"{metric} размытого и зашумлённого → мин", fontsize=13, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    ax.set_title(f"Парето-фронт ({metric}) — итерация {iteration}\n"
                 f"всего точек: {len(all_pts)}, на фронте: {pareto_mask.sum()}", fontsize=15)
    ax.legend(loc="upper left", fontsize=12)

    iter_dir = out_dir / f"{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = iter_dir / f"pareto_{metric}_iter_{iteration:04d}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  [Pareto] сохранен {path.name}")
    plt.close(fig)

def _plot_pareto_surface_only(
    pareto_pts: np.ndarray,
    metric: str,
    iteration: int,
    out_dir: Path,
) -> None:
    """Построение исключительно интерполированной поверхности Парето-фронта."""
    if len(pareto_pts) < 4:
        return

    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    try:
        Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
        ax.plot_surface(Xg, Yg, Zg, alpha=0.9, color="green", edgecolor="darkgreen", linewidth=0.1)
    except Exception as e:
        print(f"  [Warning] Ошибка построения изолированной поверхности: {e}")
        plt.close(fig)
        return

    # Настройка осей (русский язык, но PSNR/SSIM остаются на английском)
    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"{metric} размытого и зашумлённого → мин", fontsize=13, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    ax.set_title(f"Парето-поверхность ({metric}) — итерация {iteration}", fontsize=15)

    iter_dir = out_dir / f"{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = iter_dir / f"pareto_{metric}_surface_only_iter_{iteration:04d}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  [Pareto] сохранен surface-only: {path.name}")
    plt.close(fig)

def _plot_pareto_3d_with_points_nd(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    iteration: int,
    out_dir: Path,
) -> None:
    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    dominated = ~pareto_mask
    if dominated.any():
        ax.scatter(all_pts[dominated, 0], all_pts[dominated, 1], all_pts[dominated, 2],
                   c="steelblue", alpha=0.20, s=10, label="Доминируемые")

    pareto_pts = all_pts[pareto_mask]
    ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], pareto_pts[:, 2],
               c="red", alpha=0.9, s=50, edgecolors="darkred", linewidths=0.6,
               label=f"Парето-фронт (точки: {len(pareto_pts)})")

    if len(pareto_pts) >= 4:
        try:
            Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
            ax.plot_surface(Xg, Yg, Zg, alpha=0.55, color="limegreen", edgecolor="none")
        except Exception as e:
            print(f"  [Warning] Ошибка ND интерполяции поверхности: {e}")

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"Уровень шума\n({metric} размытого − {metric} размыт. и зашумл.) → макс", fontsize=11, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    ax.set_title(f"Парето-фронт ({metric}) — итерация {iteration}\n"
                 f"всего точек: {len(all_pts)}, на фронте: {pareto_mask.sum()}", fontsize=15)
    ax.legend(loc="upper left", fontsize=12)

    iter_dir = out_dir / f"{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = iter_dir / f"pareto_nd_{metric}_iter_{iteration:04d}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  [Pareto ND] сохранен {path.name}")
    plt.close(fig)

def _plot_pareto_surface_only_nd(
    pareto_pts: np.ndarray,
    metric: str,
    iteration: int,
    out_dir: Path,
) -> None:
    if len(pareto_pts) < 4:
        return

    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    try:
        Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
        ax.plot_surface(Xg, Yg, Zg, alpha=0.9, color="green", edgecolor="darkgreen", linewidth=0.1)
    except Exception as e:
        print(f"  [Warning] Ошибка построения изолированной ND поверхности: {e}")
        plt.close(fig)
        return

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"Уровень шума\n({metric} размытого − {metric} размыт. и зашумл.) → макс", fontsize=11, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    ax.set_title(f"Парето-поверхность ({metric}) — итерация {iteration}", fontsize=15)

    iter_dir = out_dir / f"{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = iter_dir / f"pareto_nd_{metric}_surface_only_iter_{iteration:04d}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  [Pareto ND] сохранен surface-only: {path.name}")
    plt.close(fig)

def _build_interactive_figure_nd(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    records: List[Dict[str, Any]],
    iteration: int = 0,
) -> "go.Figure":
    """Построение интерактивного Plotly 3D-графика Парето (ND вариант)."""
    fig = go.Figure()
    dominated = ~pareto_mask
    pareto_pts = all_pts[pareto_mask]

    hover_all = []
    for i, r in enumerate(records):
        txt = (f"test: {r.get('test_name', '?')}<br>"
               f"iter: {r.get('iteration', '?')}<br>"
               f"{metric}_blur: {all_pts[i, 0]:.3f}<br>"
               f"noise_degrad: {all_pts[i, 1]:.3f}<br>"
               f"{metric}_rest: {all_pts[i, 2]:.3f}")
        hover_all.append(txt)

    has_dominated = dominated.any()
    if has_dominated:
        dom_idx = np.where(dominated)[0]
        fig.add_trace(go.Scatter3d(
            x=all_pts[dom_idx, 0], y=all_pts[dom_idx, 1], z=all_pts[dom_idx, 2],
            mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.25),
            name="Доминируемые",
            hovertext=[hover_all[i] for i in dom_idx],
            hoverinfo="text",
        ))

    par_idx = np.where(pareto_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=pareto_pts[:, 0], y=pareto_pts[:, 1], z=pareto_pts[:, 2],
        mode="markers",
        marker=dict(size=5, color="red", opacity=0.9,
                    line=dict(width=1, color="darkred")),
        name=f"Парето-фронт (точки: {len(pareto_pts)})",
        hovertext=[hover_all[i] for i in par_idx],
        hoverinfo="text",
    ))

    has_surface = False
    if len(pareto_pts) >= 4:
        try:
            Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                colorscale=[[0, "rgb(50, 180, 50)"], [1, "rgb(0, 120, 0)"]],
                opacity=0.55, showscale=False,
                name="Поверхность", hoverinfo="skip",
                visible=True,
            ))
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                colorscale=[[0, "rgb(0, 130, 0)"], [1, "rgb(0, 80, 0)"]],
                opacity=0.9, showscale=False,
                name="Парето-поверхность", hoverinfo="skip",
                visible=False,
            ))
            has_surface = True
        except Exception as e:
            print(f"  [Warning] Ошибка ND интерактивной поверхности: {e}")

    n_traces = len(fig.data)
    if has_surface:
        vis_points = [True] * n_traces
        vis_points[-1] = False
        vis_surface = [False] * n_traces
        vis_surface[-1] = True
        buttons = [
            dict(label="Парето точки", method="update",
                 args=[{"visible": vis_points}]),
            dict(label="Только поверхность", method="update",
                 args=[{"visible": vis_surface}]),
        ]
        theme_buttons = [
            dict(label="Светлая тема", method="relayout",
                 args=[{"template": "plotly", "paper_bgcolor": "white",
                        "font.color": "#333"}]),
            dict(label="Тёмная тема", method="relayout",
                 args=[{"template": "plotly_dark",
                        "paper_bgcolor": "#282c34",
                        "plot_bgcolor": "#282c34",
                        "font.color": "#abb2bf"}]),
        ]
        updatemenus = [
            dict(type="buttons", direction="down",
                 x=0.0, y=0.75, xanchor="left", yanchor="top",
                 showactive=True, buttons=buttons,
                 font=dict(size=14)),
            dict(type="buttons", direction="down",
                 x=0.0, y=0.55, xanchor="left", yanchor="top",
                 showactive=True, buttons=theme_buttons,
                 font=dict(size=14)),
        ]
    else:
        theme_buttons = [
            dict(label="Светлая тема", method="relayout",
                 args=[{"template": "plotly", "paper_bgcolor": "white",
                        "font.color": "#333"}]),
            dict(label="Тёмная тема", method="relayout",
                 args=[{"template": "plotly_dark",
                        "paper_bgcolor": "#282c34",
                        "plot_bgcolor": "#282c34",
                        "font.color": "#abb2bf"}]),
        ]
        updatemenus = [
            dict(type="buttons", direction="down",
                 x=0.0, y=0.55, xanchor="left", yanchor="top",
                 showactive=True, buttons=theme_buttons,
                 font=dict(size=14)),
        ]

    iter_label = f" — итерация {iteration}" if iteration else ""
    fig.update_layout(
        title=dict(text=f"Парето-фронт ({metric}){iter_label} — "
                        f"всего точек: {len(all_pts)}, "
                        f"на фронте: {pareto_mask.sum()}",
                   x=0.6, font=dict(size=16)),
        scene=dict(
            domain=dict(x=[0.2, 1.0]),
            xaxis=dict(title=dict(text=f"{metric} размытого → мин", font=dict(size=11))),
            yaxis=dict(title=dict(text=f"Уровень шума<br>({metric} размытого − {metric} размыт. и зашумл.) → макс", font=dict(size=10))),
            zaxis=dict(title=dict(text=f"{metric} восстановленного → макс", font=dict(size=11))),
        ),
        updatemenus=updatemenus,
        width=1200, height=900,
        legend=dict(x=-0.05, y=0.99, xanchor="left", font=dict(size=14)),
    )
    return fig

def _build_interactive_figure(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    records: List[Dict[str, Any]],
    iteration: int = 0,
) -> "go.Figure":
    """Построение интерактивного Plotly 3D-графика Парето с переключением режимов."""
    fig = go.Figure()
    dominated = ~pareto_mask
    pareto_pts = all_pts[pareto_mask]

    # Hover-текст
    hover_all = []
    for i, r in enumerate(records):
        txt = (f"test: {r.get('test_name', '?')}<br>"
               f"iter: {r.get('iteration', '?')}<br>"
               f"{metric}_blur: {all_pts[i, 0]:.3f}<br>"
               f"{metric}_dist: {all_pts[i, 1]:.3f}<br>"
               f"{metric}_rest: {all_pts[i, 2]:.3f}")
        hover_all.append(txt)

    # trace 0: Доминируемые (синие)
    has_dominated = dominated.any()
    if has_dominated:
        dom_idx = np.where(dominated)[0]
        fig.add_trace(go.Scatter3d(
            x=all_pts[dom_idx, 0], y=all_pts[dom_idx, 1], z=all_pts[dom_idx, 2],
            mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.25),
            name="Доминируемые",
            hovertext=[hover_all[i] for i in dom_idx],
            hoverinfo="text",
        ))

    # trace 1: Точки Парето (красные)
    par_idx = np.where(pareto_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=pareto_pts[:, 0], y=pareto_pts[:, 1], z=pareto_pts[:, 2],
        mode="markers",
        marker=dict(size=5, color="red", opacity=0.9,
                    line=dict(width=1, color="darkred")),
        name=f"Парето-фронт (точки: {len(pareto_pts)})",
        hovertext=[hover_all[i] for i in par_idx],
        hoverinfo="text",
    ))

    # trace 2: Поверхность (полупрозрачная, для режима "Парето точки")
    # trace 3: Поверхность (яркая, для режима "Только поверхность")
    has_surface = False
    if len(pareto_pts) >= 4:
        try:
            Xg, Yg, Zg = _interpolate_pareto_surface(pareto_pts)
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                colorscale=[[0, "rgb(50, 180, 50)"], [1, "rgb(0, 120, 0)"]],
                opacity=0.55, showscale=False,
                name="Поверхность", hoverinfo="skip",
                visible=True,
            ))
            fig.add_trace(go.Surface(
                x=Xg, y=Yg, z=Zg,
                colorscale=[[0, "rgb(0, 130, 0)"], [1, "rgb(0, 80, 0)"]],
                opacity=0.9, showscale=False,
                name="Парето-поверхность", hoverinfo="skip",
                visible=False,
            ))
            has_surface = True
        except Exception as e:
            print(f"  [Warning] Ошибка интерактивной поверхности: {e}")

    # Кнопки переключения режимов
    n_traces = len(fig.data)
    if has_surface:
        # Режим "Парето точки": все точки + полупрозрачная поверхность
        vis_points = [True] * n_traces
        vis_points[-1] = False  # яркая поверхность скрыта
        # Режим "Только поверхность": только яркая поверхность
        vis_surface = [False] * n_traces
        vis_surface[-1] = True  # яркая поверхность
        buttons = [
            dict(label="Парето точки", method="update",
                 args=[{"visible": vis_points}]),
            dict(label="Только поверхность", method="update",
                 args=[{"visible": vis_surface}]),
        ]
        theme_buttons = [
            dict(label="Светлая тема", method="relayout",
                 args=[{"template": "plotly", "paper_bgcolor": "white",
                        "font.color": "#333"}]),
            dict(label="Тёмная тема", method="relayout",
                 args=[{"template": "plotly_dark",
                        "paper_bgcolor": "#282c34",
                        "plot_bgcolor": "#282c34",
                        "font.color": "#abb2bf"}]),
        ]
        updatemenus = [
            dict(type="buttons", direction="down",
                 x=0.0, y=0.75, xanchor="left", yanchor="top",
                 showactive=True, buttons=buttons,
                 font=dict(size=14)),
            dict(type="buttons", direction="down",
                 x=0.0, y=0.55, xanchor="left", yanchor="top",
                 showactive=True, buttons=theme_buttons,
                 font=dict(size=14)),
        ]
    else:
        theme_buttons = [
            dict(label="Светлая тема", method="relayout",
                 args=[{"template": "plotly", "paper_bgcolor": "white",
                        "font.color": "#333"}]),
            dict(label="Тёмная тема", method="relayout",
                 args=[{"template": "plotly_dark",
                        "paper_bgcolor": "#282c34",
                        "plot_bgcolor": "#282c34",
                        "font.color": "#abb2bf"}]),
        ]
        updatemenus = [
            dict(type="buttons", direction="down",
                 x=0.0, y=0.55, xanchor="left", yanchor="top",
                 showactive=True, buttons=theme_buttons,
                 font=dict(size=14)),
        ]

    iter_label = f" — итерация {iteration}" if iteration else ""
    fig.update_layout(
        title=dict(text=f"Парето-фронт ({metric}){iter_label} — "
                        f"всего точек: {len(all_pts)}, "
                        f"на фронте: {pareto_mask.sum()}",
                   x=0.6, font=dict(size=16)),
        scene=dict(
            domain=dict(x=[0.2, 1.0]),
            xaxis=dict(title=dict(text=f"{metric} размытого → мин", font=dict(size=11))),
            yaxis=dict(title=dict(text=f"{metric} размытого и зашумленного → мин", font=dict(size=11))),
            zaxis=dict(title=dict(text=f"{metric} восстановленного → макс", font=dict(size=11))),
        ),
        updatemenus=updatemenus,
        width=1200, height=900,
        legend=dict(x=-0.05, y=0.99, xanchor="left", font=dict(size=14)),
    )
    return fig


def update_pareto_plots(iteration: int) -> None:
    """Обновление Парето-графиков: вычисление фронтов и генерация набора статических и интерактивных графиков."""
    if not _PARETO_STORE:
        return
    
    psnr_pts = np.array([[d["psnr_blur"], d["psnr_dist"], d["psnr_rest"]]
                         for d in _PARETO_STORE])
    ssim_pts = np.array([[d["ssim_blur"], d["ssim_dist"], d["ssim_rest"]]
                         for d in _PARETO_STORE])

    psnr_mask = compute_pareto_front(psnr_pts)
    ssim_mask = compute_pareto_front(ssim_pts)

    tag = f"iter_{iteration:04d}"

    # --- Экспорт статических графиков (PNG) ---
    plots_dir = PARETO_OUTPUT_DIR / "pareto_plots"
    psnr_png_dir = plots_dir / "PSNR"
    ssim_png_dir = plots_dir / "SSIM"
    
    _plot_pareto_3d_with_points(psnr_pts, psnr_mask, "PSNR", iteration, psnr_png_dir)
    _plot_pareto_3d_with_points(ssim_pts, ssim_mask, "SSIM", iteration, ssim_png_dir)
    
    pareto_psnr = psnr_pts[psnr_mask]
    pareto_ssim = ssim_pts[ssim_mask]
    _plot_pareto_surface_only(pareto_psnr, "PSNR", iteration, psnr_png_dir)
    _plot_pareto_surface_only(pareto_ssim, "SSIM", iteration, ssim_png_dir)

    # --- Экспорт интерактивных графиков (HTML) ---
    if _HAS_PLOTLY:
        interactive_dir = PARETO_OUTPUT_DIR / "pareto_interactive"
        for metric, pts, mask in [("PSNR", psnr_pts, psnr_mask),
                                   ("SSIM", ssim_pts, ssim_mask)]:
            html_dir = interactive_dir / metric
            html_dir.mkdir(parents=True, exist_ok=True)
            fig = _build_interactive_figure(pts, mask, metric, _PARETO_STORE, iteration)
            html_path = html_dir / f"pareto_{metric}_{tag}.html"
            fig.write_html(str(html_path))
            print(f"  [Pareto] сохранен interactive: {html_path.name}")
    else:
        print(" [Pareto] plotly не установлен — пропускаем интерактивный HTML")

    # Сохраняем данные в JSON для дальнейшего анализа
    json_path = PARETO_OUTPUT_DIR / f"pareto_data_iter_{iteration:04d}.json"
    _save_pareto_json(json_path)

    # --- Экспорт графиков Noise Degradation (ND) ---
    nd_psnr_pts = np.array([[d["psnr_blur"], d["psnr_blur"] - d["psnr_dist"], d["psnr_rest"]]
                            for d in _PARETO_STORE])
    nd_ssim_pts = np.array([[d["ssim_blur"], d["ssim_blur"] - d["ssim_dist"], d["ssim_rest"]]
                            for d in _PARETO_STORE])

    nd_psnr_mask = compute_pareto_front_nd(nd_psnr_pts)
    nd_ssim_mask = compute_pareto_front_nd(nd_ssim_pts)

    # PNG ND
    nd_plots_dir = PARETO_OUTPUT_DIR / "pareto_nd_plots"
    _plot_pareto_3d_with_points_nd(nd_psnr_pts, nd_psnr_mask, "PSNR", iteration, nd_plots_dir / "PSNR")
    _plot_pareto_3d_with_points_nd(nd_ssim_pts, nd_ssim_mask, "SSIM", iteration, nd_plots_dir / "SSIM")
    _plot_pareto_surface_only_nd(nd_psnr_pts[nd_psnr_mask], "PSNR", iteration, nd_plots_dir / "PSNR")
    _plot_pareto_surface_only_nd(nd_ssim_pts[nd_ssim_mask], "SSIM", iteration, nd_plots_dir / "SSIM")

    # HTML ND
    if _HAS_PLOTLY:
        nd_interactive_dir = PARETO_OUTPUT_DIR / "pareto_nd_interactive"
        for metric, pts, mask in [("PSNR", nd_psnr_pts, nd_psnr_mask),
                                   ("SSIM", nd_ssim_pts, nd_ssim_mask)]:
            html_dir = nd_interactive_dir / metric
            html_dir.mkdir(parents=True, exist_ok=True)
            fig = _build_interactive_figure_nd(pts, mask, metric, _PARETO_STORE, iteration)
            html_path = html_dir / f"pareto_nd_{metric}_{tag}.html"
            fig.write_html(str(html_path))
            print(f"  [Pareto ND] сохранен interactive: {html_path.name}")


def _save_pareto_json(path: Path) -> None:
    """Экспорт накопленных метрик и параметров в JSON-формат."""
    records = []
    for d in _PARETO_STORE:
        rec = {k: v for k, v in d.items() if k != "params"}
        rec["params"] = {k: (float(v) if isinstance(v, (np.floating, float))
                             else int(v) if isinstance(v, (np.integer, int, bool))
                             else v)
                         for k, v in d["params"].items()}
        records.append(rec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)


# --- Worker обработки отдельного изображения для ProcessPoolExecutor ---
def _process_single_image(args: tuple) -> dict:
    """
    Выполнение слепой деконволюции для одного тестового случая и вычисление метрик качества.
    Включение системы штрафов для фильтрации вырожденных решений.
    """
    (idx, name, original, distorted, blur_only, lr_params, max_side, has_noise) = args
    
    t0 = time.time()
    original  = _resize_if_needed(original,  max_side)
    distorted = _resize_if_needed(distorted, max_side)
    blur_only = _resize_if_needed(blur_only, max_side)

    # --- Метрики до восстановления ---
    psnr_blur = compute_psnr(original, blur_only)
    ssim_blur = compute_ssim(original, blur_only)
    psnr_dist = compute_psnr(original, distorted)
    ssim_dist = compute_ssim(original, distorted)

    # --- Восстановление ---
    psnr_rest, ssim_rest = psnr_dist, ssim_dist    # fallback
    error_msg = None
    kernel_entropy_norm = None
    std_restored = None
    std_distorted = None
    
    try:
        algo = LIP_BD(**lr_params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            restored, kernel = algo.process(distorted)
        
        if np.all(np.isfinite(restored.astype(np.float64))):
            psnr_rest = compute_psnr(original, restored)
            ssim_rest = compute_ssim(original, restored)
            
            # --- Вычисление штрафов за вырожденные решения ---
            std_restored = np.std(restored.astype(np.float64))
            std_distorted = np.std(distorted.astype(np.float64))
            
            # Вычисление нормализованной энтропии ядра (для детекции дельта-ядер)
            if kernel is not None:
                k = kernel.astype(np.float64)
                k = k / (k.sum() + 1e-12)
                k_flat = k.ravel()
                ent = entropy(k_flat + 1e-12, base=2)
                max_ent = np.log2(len(k_flat))
                kernel_entropy_norm = ent / max_ent 
            
            # Штраф за потерю дисперсии восстановленного изображения (решение вырождается в почти константное изображение)
            if std_restored is not None and std_distorted is not None:
                if std_restored < std_distorted / 20.0:
                    ssim_rest -= 0.4
            
            # Штраф за слишком острое дельта-ядро (решение, игнорирующее смаз)
            if kernel_entropy_norm is not None and kernel_entropy_norm < 0.3:
                ssim_rest -= 0.3

            # Ограничение минимального значения метрики
            ssim_rest = max(ssim_rest, 0.0)
                
            # Не допускаем отрицательных значений SSIM
            ssim_rest = max(ssim_rest, 0.0)
            
        else:
            error_msg = "NaN in restored"
    except Exception as e:
        error_msg = str(e)

    return {
        "idx": idx, "name": name,
        "psnr_blur": psnr_blur, "ssim_blur": ssim_blur,
        "psnr_dist": psnr_dist, "ssim_dist": ssim_dist,
        "psnr_rest": psnr_rest, "ssim_rest": ssim_rest,
        "delta_ssim": ssim_rest - ssim_dist,
        "dt":    time.time() - t0,
        "error": error_msg,
        "has_noise": has_noise,
    }

# --- Целевая функция оптимизатора ObjectiveBackend ---
def objective_function(config: Dict[str, Any], dict_config: Dict[str, Any]) -> float:
    """
    Метрика для HMM_MCMC:  group-median(SSIM_restored)  по всем тестам.
    Возвращает loss = -group_median(SSIM_rest)  (ObjectiveBackend минимизирует).
    """
    _EVAL_COUNTER[0] += 1
    eval_id = _EVAL_COUNTER[0]
    t_start = time.time()

    test_cases = _get_test_cases()

    # Сбор параметров алгоритма LIP_BD из конфигурации
    lr_params: Dict[str, Any] = {}
    for name in dict_config:
        lr_params[name] = config[name]

    # int-параметры
    _INT_KEYS = {"outer_iters", "inner_iters"}
    for ik in _INT_KEYS:
        if ik in lr_params:
            lr_params[ik] = int(round(lr_params[ik]))

    # kernel_shape — tuple из categorical
    kernel_shape = lr_params.pop("kernel_shape", (31, 31))
    if not isinstance(kernel_shape, tuple):
        kernel_shape = tuple(kernel_shape)

    # gamma_correction: 0/1 → bool
    gamma_flag = int(lr_params.pop("gamma_correction", 0))
    use_gamma = bool(gamma_flag)

    # ringing_removal sub-params → nb_params dict
    lambda_tv = float(lr_params.pop("lambda_tv", 1e-3))
    lambda_l0 = float(lr_params.pop("lambda_l0", 2e-3))
    weight_ring = float(lr_params.pop("weight_ring", 1.0))
    nb_params = {
        "lambda_tv": lambda_tv,
        "lambda_l0": lambda_l0,
        "weight_ring": weight_ring,
    }

    # Применение фиксированных параметров алгоритма
    lr_params["kernel_shape"] = kernel_shape
    lr_params["method"] = "pd"
    lr_params["final_deconv"] = "ringing_removal"
    lr_params["gamma_correction"] = use_gamma
    lr_params["auto_params"] = None
    lr_params["nb_params"] = nb_params

    print(f"\n{'='*72}")
    print(f"  [eval {eval_id}] LIP_BD params: {lr_params}")
    print(f"{'='*72}")

    tasks = [(idx, tc.name, tc.original, tc.distorted, tc.blur_only,
              lr_params, MAX_IMAGE_SIDE, tc.has_noise)
             for idx, tc in enumerate(test_cases)]

    # Распараллеленная обработка датасета
    raw_results: List[Dict[str, Any]] = []
    if len(tasks) <= 2:
        for t in tasks:
            raw_results.append(_process_single_image(t))
    else:
        n_workers = min(len(tasks), 20)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_process_single_image, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                raw_results.append(fut.result())
    raw_results.sort(key=lambda r: r["idx"])

    # Агрегация результатов
    delta_ssims: List[float] = []
    for r in raw_results:
        err = f"  ERROR: {r['error']}" if r["error"] else ""
        print(f"    [{eval_id}] {r['idx']+1:2d}/{len(tasks)} \"{r['name']}\"  "
              f"delta_SSIM={r['delta_ssim']:+.4f}  "
              f"(blur={r['ssim_blur']:.3f}  dist={r['ssim_dist']:.3f}  "
              f"rest={r['ssim_rest']:.3f})  {r['dt']:.1f}s{err}")
        delta_ssims.append(r["delta_ssim"])

        # Сохраняем в глобальное хранилище Парето
        _PARETO_STORE.append({
            "psnr_blur": r["psnr_blur"], "psnr_dist": r["psnr_dist"],
            "psnr_rest": r["psnr_rest"],
            "ssim_blur": r["ssim_blur"], "ssim_dist": r["ssim_dist"],
            "ssim_rest": r["ssim_rest"],
            "params": lr_params.copy(),
            "iteration": eval_id,
            "test_name": r["name"],
        })

    # --- Групповая медиана абсолютного SSIM (равные веса) ---
    ssim_clean = [r["ssim_rest"] for r in raw_results if not r["has_noise"]]
    ssim_noisy = [r["ssim_rest"] for r in raw_results if r["has_noise"]]
    median_clean = float(np.median(ssim_clean)) if ssim_clean else 0.0
    median_noisy = float(np.median(ssim_noisy)) if ssim_noisy else 0.0
    group_metric = 0.5 * (median_clean + median_noisy)
    
    dt_total = time.time() - t_start
    print(f"  [eval {eval_id}] group-median(SSIM_rest) = {group_metric:.4f}  "
            f"({len(raw_results)}/{len(test_cases)} tests, {dt_total:.1f}s)\n")
    
    # Обновление Парето-фронтов с заданным шагом
    if eval_id % 5 == 0:
        update_pareto_plots(eval_id)

    # Инверсия метрики (минимизируется ObjectiveBackend)
    return -group_metric


# --- Конфигурация HPO-эксперимента ---
config_HMM = {
    "full_args": {
        "algorithm": {
            "name": "HMM_MCMC",
            "budget": 150,
            "n_init": 10,
            "n_chains": 1,
            "orchestrate_every": 1000,
            "T_mcmc": 0.01,
            "sigma_fraction": 0.0055,
            "wide_sigma_fraction": 0.5,
            "temperature": 0.3,
            "hmm_window": 4,
            "hmm_obs_epsilon": 1e-8,
            "hmm_lambda_noise": 0.01,
            "clone_noise": 0.05,
            "burnin_fraction": 0.0,
            "p_cat_step": 0.0,
            "kde_tau": 0.05,
            "anneal_T": True,
        }
    },
    "backend": {
        "name": "objective",
        "objective_function": objective_function,
        "hp_space": LIP_HP_SPACE,
    },
}


if __name__ == "__main__":
    # Запуск оптимизации
    run_n_experiments(config_HMM, n_experiments=1, inference_only=False)

    # Итоговая генерация графиков для всех накопленных точек
    if _PARETO_STORE:
        update_pareto_plots(_EVAL_COUNTER[0])
        print(f"\nDone: {len(_PARETO_STORE)} points -> {PARETO_OUTPUT_DIR}")
