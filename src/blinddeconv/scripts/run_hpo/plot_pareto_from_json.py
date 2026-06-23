"""
plot_pareto_from_json.py

Модуль для генерации статических 3D-графиков Парето-фронтов (PSNR и SSIM) 
на основе данных, экспортированных в формате JSON в процессе оптимизации.

Использование:
    python plot_pareto_from_json.py <path_to_json>
    python plot_pareto_from_json.py <path_to_json> --outdir ./my_pareto_plots

Ожидаемый формат JSON: список словарей с ключами:
    psnr_blur, psnr_dist, psnr_rest, ssim_blur, ssim_dist, ssim_rest
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Tuple

from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label

import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Углы обзора (elevation, azimuth) для генерации статических проекций
_VIEW_ANGLES = [
    (30, -60),   # Ракурс 1: Базовый вид
    (30, 30),    # Ракурс 2: Поворот на 90°
    (30, 120),   # Ракурс 3: Поворот на 180°
    (30, 210),   # Ракурс 4: Поворот на 270°
]


# --- Вычисление Парето-фронта ---
def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Вычисление маски недоминируемых точек (Парето-фронт).
    Ось X: минимизируется.
    Ось Y: минимизируется.
    Ось Z: максимизируется.
    
    Returns:
        is_pareto: Булев массив размерности N.
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
    Вычисление маски Парето-фронта Noise Degradation (ND).
    Ось X: минимизируется.
    Ось Y: максимизируется.
    Ось Z: максимизируется.
    
    Returns:
        is_pareto: Булев массив размерности N.
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


# --- Интерполяция поверхности Парето-фронта ---
def interpolate_pareto_surface(
    pareto_pts: np.ndarray,
    grid_size: int = 200,
    smooth_sigma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Интерполяция поверхности Парето-фронта методом радиальных базисных функций (RBF thin-plate spline).
    Включение пространственной фильтрации для устранения артефактов экстраполяции.
    """
    xy = pareto_pts[:, :2]
    z  = pareto_pts[:, 2]
    z_min, z_max = z.min(), z.max()

    # Формирование регулярной сетки с 5% запасом по краям
    x_lo, x_hi = xy[:, 0].min(), xy[:, 0].max()
    y_lo, y_hi = xy[:, 1].min(), xy[:, 1].max()
    x_range = x_hi - x_lo if x_hi > x_lo else 1.0
    y_range = y_hi - y_lo if y_hi > y_lo else 1.0
    x_pad = 0.05 * x_range
    y_pad = 0.05 * y_range
    xg = np.linspace(x_lo - x_pad, x_hi + x_pad, grid_size)
    yg = np.linspace(y_lo - y_pad, y_hi + y_pad, grid_size)
    Xg, Yg = np.meshgrid(xg, yg)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])

    # RBF-интерполяция (thin-plate spline)
    rbf = RBFInterpolator(xy, z, kernel='thin_plate_spline', smoothing=0.0)
    Zg = rbf(grid).reshape(Xg.shape)

    # Ограничение области интерполяции по евклидову расстоянию до ближайших точек
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

    # Исключение значений, значительно превышающих диапазон реальных точек
    z_margin = 1.00 * (z_max - z_min) if z_max > z_min else 0.01
    Zg[(Zg < z_min - z_margin) | (Zg > z_max + z_margin)] = np.nan

    # Удаление изолированных сегментов интерполяции
    valid = ~np.isnan(Zg)
    labeled, num_features = label(valid)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        Zg[labeled != sizes.argmax()] = np.nan

    # Лёгкое гауссово сглаживание восстановленной поверхности
    valid = ~np.isnan(Zg)
    if valid.any() and smooth_sigma > 0:
        Zg_filled = np.where(valid, Zg, 0.0)
        w = valid.astype(float)
        Zg_num = gaussian_filter(Zg_filled, sigma=smooth_sigma)
        Zg_den = gaussian_filter(w, sigma=smooth_sigma)
        with np.errstate(invalid='ignore', divide='ignore'):
            Zg[valid] = np.where(Zg_den[valid] > 1e-12,
                                 Zg_num[valid] / Zg_den[valid], Zg[valid])

    return Xg, Yg, Zg


# --- Построение статических 3D-графиков ---
def plot_pareto_3d_with_points(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    out_path: Path,
    tag: str = "",
) -> None:
    """Построение комбинированного графика: точки и интерполированная поверхность."""
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
            Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
            ax.plot_surface(Xg, Yg, Zg, alpha=0.55, color="limegreen", edgecolor="none")
        except Exception as e:
            print(f"  [Warning] Ошибка построения поверхности: {e}")

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"{metric} размытого и зашумлённого → мин", fontsize=13, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    
    title = f"Парето-фронт ({metric})"
    if tag:
        title += f" — {tag}"
    title += f"\nвсего точек: {len(all_pts)}, на фронте: {pareto_mask.sum()}"
    ax.set_title(title, fontsize=15)
    ax.legend(loc="upper left", fontsize=12)

    stem = out_path.stem
    m = re.search(r'iter_(\d+)', stem)
    save_dir = out_path.parent / (m.group(1) if m else "misc")
    save_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = save_dir / f"{stem}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_pareto_surface_only(
    pareto_pts: np.ndarray,
    metric: str,
    out_path: Path,
    tag: str = "",
) -> None:
    """Построение исключительно интерполированной поверхности Парето-фронта."""
    if len(pareto_pts) < 4:
        print(f"  [Информация] Пропуск изолированной поверхности ({metric}): требуется >= 4 точек на фронте.")
        return

    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    try:
        Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
        ax.plot_surface(Xg, Yg, Zg, alpha=0.9, color="green",
                        edgecolor="darkgreen", linewidth=0.1)
    except Exception as e:
        print(f"  [Warning] Ошибка построения изолированной поверхности: {e}")
        plt.close(fig)
        return

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"{metric} размытого и зашумлённого → мин", fontsize=13, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    
    title = f"Парето-поверхность ({metric})"
    if tag:
        title += f" — {tag}"
    ax.set_title(title, fontsize=15)

    stem = out_path.stem
    m = re.search(r'iter_(\d+)', stem)
    save_dir = out_path.parent / (m.group(1) if m else "misc")
    save_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = save_dir / f"{stem}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  Saved: {path}")
    plt.close(fig)


# --- Построение графиков проекции ND (Noise Degradation) ---
def plot_pareto_3d_with_points_nd(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    out_path: Path,
    tag: str = "",
) -> None:
    """Построение комбинированного графика (вариант Noise Degradation)."""
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
            Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
            ax.plot_surface(Xg, Yg, Zg, alpha=0.55, color="limegreen", edgecolor="none")
        except Exception as e:
            print(f"  [Warning] Ошибка построения поверхности ND: {e}")

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"Уровень шума\n({metric} размытого − {metric} размыт. и зашумл.) → макс", fontsize=11, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    title = f"Парето-фронт ({metric})"
    if tag:
        title += f" — {tag}"
    title += f"\nвсего точек: {len(all_pts)}, на фронте: {pareto_mask.sum()}"
    ax.set_title(title, fontsize=15)
    ax.legend(loc="upper left", fontsize=12)

    stem = out_path.stem
    m = re.search(r'iter_(\d+)', stem)
    save_dir = out_path.parent / (m.group(1) if m else "misc")
    save_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = save_dir / f"{stem}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_pareto_surface_only_nd(
    pareto_pts: np.ndarray,
    metric: str,
    out_path: Path,
    tag: str = "",
) -> None:
    """Построение исключительно интерполированной поверхности Парето-фронта ND."""
    if len(pareto_pts) < 4:
        print(f"  [Информация] Пропуск изолированной ND поверхности ({metric}): требуется >= 4 точек на фронте.")
        return

    fig = plt.figure(figsize=(18, 13))
    ax  = fig.add_subplot(111, projection="3d")

    try:
        Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
        ax.plot_surface(Xg, Yg, Zg, alpha=0.9, color="green",
                        edgecolor="darkgreen", linewidth=0.1)
    except Exception as e:
        print(f"  [Warning] Ошибка построения изолированной поверхности ND: {e}")
        plt.close(fig)
        return

    ax.set_xlabel(f"{metric} размытого → мин", fontsize=13, labelpad=14)
    ax.set_ylabel(f"Уровень шума\n({metric} размытого − {metric} размыт. и зашумл.) → макс", fontsize=11, labelpad=14)
    ax.set_zlabel(f"{metric} восстановленного → макс", fontsize=13, labelpad=14)
    title = f"Парето-поверхность ({metric})"
    if tag:
        title += f" — {tag}"
    ax.set_title(title, fontsize=15)

    stem = out_path.stem
    m = re.search(r'iter_(\d+)', stem)
    save_dir = out_path.parent / (m.group(1) if m else "misc")
    save_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, (elev, azim) in enumerate(_VIEW_ANGLES, 1):
        ax.view_init(elev=elev, azim=azim)
        path = save_dir / f"{stem}_{view_idx}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight", pad_inches=0.4)
        print(f"  Saved: {path}")
    plt.close(fig)


# --- Точка входа ---
def main():
    parser = argparse.ArgumentParser(
        description="Построение статических 3D-графиков Парето-фронтов на базе JSON данных.")
    parser.add_argument("json_path", type=str,
                        help="Путь к JSON-файлу с точками Парето.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Директория для сохранения графиков (по умолчанию — рядом с JSON).")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"Файл не найден: {json_path}")
        sys.exit(1)

    out_dir = Path(args.outdir) if args.outdir else json_path.parent / "pareto_plots"

    json_stem = json_path.stem  #
    raw_tag = json_stem.replace("pareto_data_", "") 
    file_tag = raw_tag 

    m = re.search(r'iter_0*(\d+)', raw_tag)
    display_tag = f"итерация {m.group(1)}" if m else raw_tag

    with open(str(json_path), "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Загружено {len(records)} точек из {json_path}")

    psnr_pts = np.array([[r["psnr_blur"], r["psnr_dist"], r["psnr_rest"]]
                         for r in records])
    ssim_pts = np.array([[r["ssim_blur"], r["ssim_dist"], r["ssim_rest"]]
                         for r in records])

    psnr_mask = compute_pareto_front(psnr_pts)
    ssim_mask = compute_pareto_front(ssim_pts)

    print(f"PSNR: {psnr_mask.sum()} точек на фронте из {len(psnr_pts)}")
    print(f"SSIM: {ssim_mask.sum()} точек на фронте из {len(ssim_pts)}")

    psnr_dir = out_dir / "PSNR"
    ssim_dir = out_dir / "SSIM"

    # Построение базовых проекций
    plot_pareto_3d_with_points(psnr_pts, psnr_mask, "PSNR",
                               psnr_dir / f"pareto_PSNR_{file_tag}.png", display_tag)
    plot_pareto_3d_with_points(ssim_pts, ssim_mask, "SSIM",
                               ssim_dir / f"pareto_SSIM_{file_tag}.png", display_tag)

    plot_pareto_surface_only(psnr_pts[psnr_mask], "PSNR",
                             psnr_dir / f"pareto_PSNR_surface_only_{file_tag}.png", display_tag)
    plot_pareto_surface_only(ssim_pts[ssim_mask], "SSIM",
                             ssim_dir / f"pareto_SSIM_surface_only_{file_tag}.png", display_tag)

    # Формирование ND-проекций (Noise Degradation)
    nd_psnr_pts = np.array([[r["psnr_blur"],
                             r["psnr_blur"] - r["psnr_dist"],
                             r["psnr_rest"]] for r in records])
    nd_ssim_pts = np.array([[r["ssim_blur"],
                             r["ssim_blur"] - r["ssim_dist"],
                             r["ssim_rest"]] for r in records])

    nd_psnr_mask = compute_pareto_front_nd(nd_psnr_pts)
    nd_ssim_mask = compute_pareto_front_nd(nd_ssim_pts)

    print(f"PSNR (ND): {nd_psnr_mask.sum()} точек на фронте из {len(nd_psnr_pts)}")
    print(f"SSIM (ND): {nd_ssim_mask.sum()} точек на фронте из {len(nd_ssim_pts)}")

    nd_out_dir = json_path.parent / "pareto_nd_plots"
    nd_psnr_dir = nd_out_dir / "PSNR"
    nd_ssim_dir = nd_out_dir / "SSIM"

    plot_pareto_3d_with_points_nd(nd_psnr_pts, nd_psnr_mask, "PSNR",
                                  nd_psnr_dir / f"pareto_nd_PSNR_{file_tag}.png", display_tag)
    plot_pareto_3d_with_points_nd(nd_ssim_pts, nd_ssim_mask, "SSIM",
                                  nd_ssim_dir / f"pareto_nd_SSIM_{file_tag}.png", display_tag)

    plot_pareto_surface_only_nd(nd_psnr_pts[nd_psnr_mask], "PSNR",
                                nd_psnr_dir / f"pareto_nd_PSNR_surface_only_{file_tag}.png", display_tag)
    plot_pareto_surface_only_nd(nd_ssim_pts[nd_ssim_mask], "SSIM",
                                nd_ssim_dir / f"pareto_nd_SSIM_surface_only_{file_tag}.png", display_tag)

    print(f"\nГотово! Графики сохранены в {out_dir} и {nd_out_dir}")


if __name__ == "__main__":
    main()
