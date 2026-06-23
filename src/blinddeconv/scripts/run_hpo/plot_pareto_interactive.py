"""
plot_pareto_interactive.py

Модуль для генерации интерактивных 3D-моделей Парето-фронтов (PSNR и SSIM) 
в формате HTML с использованием библиотеки Plotly. 
Предоставляет графический интерфейс для масштабирования, вращения и переключения 
режимов отображения (точки, поверхность, выбор темы).

Использование:
    python plot_pareto_interactive.py <path_to_json>
    python plot_pareto_interactive.py <path_to_json> --metric ssim
    python plot_pareto_interactive.py <path_to_json> --metric both
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

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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
    Вычисление маски Парето-фронта для Noise Degradation (ND).
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
    Интерполяция поверхности Парето-фронта методом радиальных базисных функций (RBF thin-plate spline).
    """
    xy = pareto_pts[:, :2]
    z  = pareto_pts[:, 2]
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


# --- Потсроение интерактивной модели (Plotly) ---
def build_interactive_figure(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    records: list,
    tag: str = "",
) -> go.Figure:
    """Построение интерактивного 3D-графика Парето с переключением режимов."""
    fig = go.Figure()

    dominated = ~pareto_mask
    pareto_pts = all_pts[pareto_mask]

    hover_all = []
    for i, r in enumerate(records):
        txt = (f"test: {r.get('test_name', '?')}<br>"
               f"iter: {r.get('iteration', '?')}<br>"
               f"{metric}_blur: {all_pts[i, 0]:.3f}<br>"
               f"{metric}_dist: {all_pts[i, 1]:.3f}<br>"
               f"{metric}_rest: {all_pts[i, 2]:.3f}")
        hover_all.append(txt)

    has_dominated = dominated.any()
    if has_dominated:
        dom_idx = np.where(dominated)[0]
        fig.add_trace(go.Scatter3d(
            x=all_pts[dom_idx, 0],
            y=all_pts[dom_idx, 1],
            z=all_pts[dom_idx, 2],
            mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.25),
            name="Доминируемые",
            hovertext=[hover_all[i] for i in dom_idx],
            hoverinfo="text",
        ))

    par_idx = np.where(pareto_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=pareto_pts[:, 0],
        y=pareto_pts[:, 1],
        z=pareto_pts[:, 2],
        mode="markers",
        marker=dict(size=5, color="red", opacity=0.9,
                    line=dict(width=1, color="darkred")),
        name=f"Парето-фронт (точки: {len(pareto_pts)})",
        hovertext=[hover_all[i] for i in par_idx],
        hoverinfo="text",
    ))

    if len(pareto_pts) >= 4:
        try:
            Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
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
            print(f"  [Warning] Ошибка интерполяции интерактивной поверхности: {e}")

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

    tag_label = f" — {tag}" if tag else ""
    fig.update_layout(
        title=dict(text=f"Парето-фронт ({metric}){tag_label} — "
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
        width=1200,
        height=900,
        legend=dict(x=-0.05, y=0.99, xanchor="left", font=dict(size=14)),
    )
    return fig


def build_interactive_figure_nd(
    all_pts: np.ndarray,
    pareto_mask: np.ndarray,
    metric: str,
    records: list,
    tag: str = "",
) -> go.Figure:
    """Построение интерактивной 3D-модели Парето-фронта (вариант Noise Degradation)."""
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
            x=all_pts[dom_idx, 0],
            y=all_pts[dom_idx, 1],
            z=all_pts[dom_idx, 2],
            mode="markers",
            marker=dict(size=3, color="steelblue", opacity=0.25),
            name="Доминируемые",
            hovertext=[hover_all[i] for i in dom_idx],
            hoverinfo="text",
        ))

    par_idx = np.where(pareto_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=pareto_pts[:, 0],
        y=pareto_pts[:, 1],
        z=pareto_pts[:, 2],
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
            Xg, Yg, Zg = interpolate_pareto_surface(pareto_pts)
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
            print(f"  [Warning] Ошибка интерполяции интерактивной поверхности ND: {e}")

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

    tag_label = f" — {tag}" if tag else ""
    fig.update_layout(
        title=dict(text=f"Парето-фронт ({metric}){tag_label} — "
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
        width=1200,
        height=900,
        legend=dict(x=-0.05, y=0.99, xanchor="left", font=dict(size=14)),
    )
    return fig


# --- Точка входа ---
def main():
    parser = argparse.ArgumentParser(
        description="Построение интерактивного 3D Парето-фронта с вращением (Plotly).")
    parser.add_argument("json_path", type=str,
                       help="Путь к исходному JSON-файлу с результатами.")
    parser.add_argument("--metric", type=str, default="both",
                        choices=["psnr", "ssim", "both"],
                        help="Ограничение построения по конкретной метрике.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Директория сохранения выходных HTML-файлов.")
    parser.add_argument("--save-html", action="store_true",
                        help="Активировать сохранение на диск вместо прямого отображения.")
    args = parser.parse_args()

    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[Ошибка] Исходный файл не найден: {json_path}")
        sys.exit(1)

    out_dir = Path(args.outdir) if args.outdir else json_path.parent / "pareto_interactive"

    json_stem = json_path.stem
    raw_tag = json_stem.replace("pareto_data_", "")
    file_tag = raw_tag
    import re
    m = re.search(r'iter_0*(\d+)', raw_tag)
    display_tag = f"Итерация {m.group(1)}" if m else raw_tag

    with open(str(json_path), "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Инициализация: загружено {len(records)} записей из {json_path}")

    metrics_to_plot = []
    if args.metric in ("psnr", "both"):
        metrics_to_plot.append("PSNR")
    if args.metric in ("ssim", "both"):
        metrics_to_plot.append("SSIM")

    for metric in metrics_to_plot:
        key_prefix = metric.lower()
        pts = np.array([[r[f"{key_prefix}_blur"],
                         r[f"{key_prefix}_dist"],
                         r[f"{key_prefix}_rest"]] for r in records])

        pareto_mask = compute_pareto_front(pts)
        print(f"{metric}: {pareto_mask.sum()} точек на фронте из {len(pts)}")

        fig = build_interactive_figure(pts, pareto_mask, metric, records, display_tag)

        if args.save_html:
            metric_dir = out_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            html_path = metric_dir / f"pareto_{metric}_{file_tag}.html"
            fig.write_html(str(html_path))
            print(f"  Saved: {html_path}")
        else:
            fig.show()

    nd_out_dir = json_path.parent / "pareto_nd_interactive"

    for metric in metrics_to_plot:
        key_prefix = metric.lower()
        nd_pts = np.array([[r[f"{key_prefix}_blur"],
                            r[f"{key_prefix}_blur"] - r[f"{key_prefix}_dist"],
                            r[f"{key_prefix}_rest"]] for r in records])

        nd_mask = compute_pareto_front_nd(nd_pts)
        print(f"{metric} (ND): {nd_mask.sum()} точек на фронте из {len(nd_pts)}")

        fig = build_interactive_figure_nd(nd_pts, nd_mask, metric, records, display_tag)

        if args.save_html:
            metric_dir = nd_out_dir / metric
            metric_dir.mkdir(parents=True, exist_ok=True)
            html_path = metric_dir / f"pareto_nd_{metric}_{file_tag}.html"
            fig.write_html(str(html_path))
            print(f"  Saved: {html_path}")
        else:
            fig.show()

    print("\nПроцесс генерации завершен.")


if __name__ == "__main__":
    main()
