"""
plot_pareto_interactive_rotate.py

Модуль генерации анимированной интерактивной HTML-визуализации 3D Парето-фронта. 
Отображает динамику формирования фронта в процессе оптимизации (переход между итерациями) 
с автоматическим вращением камеры наблюдения. Включает базовый и Noise Degradation (ND) варианты.

Использование:
    python plot_pareto_interactive_rotate.py <indir> [--metric {psnr,ssim,both}]

Модуль осуществляет пакетное чтение файлов `pareto_data_iter_*.json` в указанной директории, 
хронологическую сортировку и формирование последовательности кадров анимации.
"""

import sys
import json
import glob
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, label

try:
    import plotly.graph_objects as go
except ImportError:
    print("[Ошибка] Модуль Plotly не установлен. Выполните: pip install plotly")
    sys.exit(1)


# --- Вычисление Парето-фронта ---
def compute_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    Вычисление маски недоминируемых точек (Парето-фронт).
    Ось X: минимизируется.
    Ось Y: минимизируется.
    Ось Z: максимизируется.
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
    Интерполяция поверхности Парето-фронта (RBF thin-plate spline).
    """
    if len(pareto_pts) < 4:
        return None, None, None

    xy = pareto_pts[:, :2]
    z = pareto_pts[:, 2]
    z_min, z_max = z.min(), z.max()

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

    rbf = RBFInterpolator(xy, z, kernel='thin_plate_spline', smoothing=0.0)
    Zg = rbf(grid).reshape(Xg.shape)

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

    z_margin = 1.00 * (z_max - z_min) if z_max > z_min else 0.01
    Zg[(Zg < z_min - z_margin) | (Zg > z_max + z_margin)] = np.nan

    valid = ~np.isnan(Zg)
    labeled, num_features = label(valid)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        Zg[labeled != sizes.argmax()] = np.nan

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


# --- Построение кадров анимации (frames) ---
def build_animation_frames(
    json_files: List[Path],
    metric: str,
    is_nd: bool,
    start_iter: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Генерация кадров (frames) для анимации: каждый кадр сопоставляется с определенной 
    итерацией оптимизатора, и для каждого шага синтезируются проекции с различными 
    азимутальными углами камеры наблюдения.
    """
    frames = []
    frame_names = []

    for json_path in json_files:
        stem = json_path.stem
        m = re.search(r'iter_0*(\d+)', stem)
        if not m:
            continue
        iter_num = int(m.group(1))
        if iter_num < start_iter:
            continue

        with open(str(json_path), "r", encoding="utf-8") as f:
            records = json.load(f)

        key_prefix = metric.lower()
        if is_nd:
            pts = np.array([[r[f"{key_prefix}_blur"],
                             r[f"{key_prefix}_blur"] - r[f"{key_prefix}_dist"],
                             r[f"{key_prefix}_rest"]] for r in records])
            pareto_mask = compute_pareto_front_nd(pts)
        else:
            pts = np.array([[r[f"{key_prefix}_blur"],
                             r[f"{key_prefix}_dist"],
                             r[f"{key_prefix}_rest"]] for r in records])
            pareto_mask = compute_pareto_front(pts)

        pareto_pts = pts[pareto_mask]

        hover_all = []
        for i, r in enumerate(records):
            txt = (f"test: {r.get('test_name', '?')}<br>"
                   f"iter: {r.get('iteration', '?')}<br>"
                   f"{metric}_blur: {pts[i, 0]:.3f}<br>")
            if is_nd:
                txt += f"noise_degrad: {pts[i, 1]:.3f}<br>"
            else:
                txt += f"{metric}_dist: {pts[i, 1]:.3f}<br>"
            txt += f"{metric}_rest: {pts[i, 2]:.3f}"
            hover_all.append(txt)

        dominated = ~pareto_mask
        dom_idx = np.where(dominated)[0] if dominated.any() else []
        par_idx = np.where(pareto_mask)[0]

        Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)

        for azim in range(0, 360, 10):
            frame_data = []

            if len(dom_idx) > 0:
                frame_data.append(go.Scatter3d(
                    x=pts[dom_idx, 0], y=pts[dom_idx, 1], z=pts[dom_idx, 2],
                    mode="markers",
                    marker=dict(size=3, color="steelblue", opacity=0.25),
                    name="Доминируемые",
                    hovertext=[hover_all[i] for i in dom_idx],
                    hoverinfo="text",
                ))

            frame_data.append(go.Scatter3d(
                x=pareto_pts[:, 0], y=pareto_pts[:, 1], z=pareto_pts[:, 2],
                mode="markers",
                marker=dict(size=5, color="red", opacity=0.9,
                            line=dict(width=1, color="darkred")),
                name=f"Парето-фронт (точки: {len(pareto_pts)})",
                hovertext=[hover_all[i] for i in par_idx],
                hoverinfo="text",
            ))

            if Xg is not None:
                frame_data.append(go.Surface(
                    x=Xg, y=Yg, z=Zg,
                    colorscale=[[0, "rgb(50, 180, 50)"], [1, "rgb(0, 120, 0)"]],
                    opacity=0.55, showscale=False,
                    name="Поверхность", hoverinfo="skip",
                ))

            nd_text = " (ND)" if is_nd else ""
            frames.append(go.Frame(
                data=frame_data,
                layout=go.Layout(
                    scene_camera=dict(eye=dict(x=1.75*np.cos(np.radians(azim)),
                                               y=1.75*np.sin(np.radians(azim)),
                                               z=1.55)),
                    title=f"Парето-фронт ({metric}){nd_text} — итерация {iter_num} ({azim} градусов)"
                ),
                name=f"iter_{iter_num:04d}_azim_{azim:03d}"
            ))
            frame_names.append(f"Итерация {iter_num}, угол {azim} градусов")

    return frames, frame_names


def save_animation(frames: List[Any], metric: str, is_nd: bool, outdir_base: Path):
    """Сборка последовательности кадров в единый графический объект Plotly и экспорт в HTML."""
    if not frames:
        print(f"  [Информация] Отсутствуют кадры для {metric} (ND={is_nd}). Пропуск.")
        return

    first_frame = frames[0]
    fig = go.Figure(data=first_frame.data, frames=frames)

    yaxis_title = (f"Уровень шума<br>({metric} размытого − {metric} размыт. и зашумл.) → макс" 
                   if is_nd else f"{metric} размытого и зашумленного → мин")
    nd_text = " (ND)" if is_nd else ""

    fig.update_layout(
        title=f"Парето-фронт ({metric}){nd_text} — анимация с вращением",
        scene=dict(
            xaxis=dict(title=f"{metric} размытого → мин"),
            yaxis=dict(title=yaxis_title),
            zaxis=dict(title=f"{metric} восстановленного → макс"),
            camera=dict(eye=dict(x=1.75, y=0, z=1.55))
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
                buttons=[
                    dict(label="Продолжить", method="animate",
                         args=[None, dict(frame=dict(duration=150, redraw=True),
                                          transition=dict(duration=0),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="Пауза", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=True),
                                            transition=dict(duration=0),
                                            mode="immediate")]),
                ]
            )
        ],
        width=1200, height=900,
    )

    out_dir_metric = outdir_base / metric
    out_dir_metric.mkdir(parents=True, exist_ok=True)
    prefix = "pareto_nd" if is_nd else "pareto"
    html_path = out_dir_metric / f"{prefix}_{metric}_rotate.html"
    fig.write_html(str(html_path))
    print(f"  [Экспорт] Анимация сохранена: {html_path}")


# --- Точка входа ---
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Анимированный Парето-фронт с вращением и сменой итераций.")
    parser.add_argument("indir", type=str,
                        help="Путь к директории с результатами оптимизации (JSON файлы).")
    parser.add_argument("--metric", type=str, default="both",
                        choices=["psnr", "ssim", "both"],
                        help="Ограничение построения по конкретной метрике.")
    args = parser.parse_args()

    in_dir = Path(args.indir)

    json_pattern = "pareto_data_iter_*.json"
    json_files = sorted(in_dir.glob(json_pattern))
    if not json_files:
        print(f"[Ошибка] Файлы формата {json_pattern} в директории {in_dir} не найдены.")
        sys.exit(1)

    print(f"Инициализация: обнаружено {len(json_files)} файлов в директории {in_dir}.")

    def get_iter(path):
        m = re.search(r'iter_0*(\d+)', path.stem)
        return int(m.group(1)) if m else 0

    json_files.sort(key=get_iter)

    metrics_to_plot = []
    if args.metric in ("psnr", "both"):
        metrics_to_plot.append("PSNR")
    if args.metric in ("ssim", "both"):
        metrics_to_plot.append("SSIM")

    out_dir_std = in_dir / "pareto_interactive_rotate"
    out_dir_nd = in_dir / "pareto_nd_interactive_rotate"

    for metric in metrics_to_plot:
        print(f"\nПостроение анимационной последовательности для {metric} (Обычный фронт)...")
        frames_std, _ = build_animation_frames(json_files, metric=metric, is_nd=False, start_iter=5)
        save_animation(frames_std, metric=metric, is_nd=False, outdir_base=out_dir_std)

        print(f"Построение анимационной последовательности для {metric} (ND фронт)...")
        frames_nd, _ = build_animation_frames(json_files, metric=metric, is_nd=True, start_iter=5)
        save_animation(frames_nd, metric=metric, is_nd=True, outdir_base=out_dir_nd)

    print("\nГенерация завершена. Для просмотра откройте созданные HTML-файлы и используйте кнопку 'Запуск'.")

if __name__ == "__main__":
    main()
