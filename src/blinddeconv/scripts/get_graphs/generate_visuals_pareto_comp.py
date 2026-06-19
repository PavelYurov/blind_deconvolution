#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Строит визуальные сравнения "стандартные параметры vs Парето-параметры"

Для каждой пары (алгоритм, парето-таблица) генерируется:
  1) Фигуры:
        Верхний ряд: Original | Distorted | Restored_standard | Restored_pareto
        Нижний ряд:  GT_kernel | Est_kernel_standard | Est_kernel_pareto
        + подпись с метриками (PSNR, SSIM, error_ratio, time)
  2) Сводная фигура с барами средних метрик (PSNR/SSIM/ER) для стандартных/парето.

Все картинки сохраняются в:
  presentation_graphics_pareto_comp/<ALG_LABEL>/<table_stem>/visuals/
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTPUT_ROOT = PROJECT_ROOT / "presentation_graphics_pareto_comp"
DATASET_DIR = PROJECT_ROOT / "images" / "middle_data_pictures"

_BASE_FONT = 18


def _imread_safe(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def _show(ax, img, title):
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center",
                transform=ax.transAxes, color="red")
        return
    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _crop_kernel(k: np.ndarray, thresh_frac: float = 0.02) -> np.ndarray | None:
    if k is None:
        return None
    gray = k if k.ndim == 2 else k.mean(axis=2)
    gray = gray.astype(float)
    gmax = gray.max()
    if gmax == 0:
        return k
    mask = gray > thresh_frac * gmax
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return k
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return k[r0:r1, c0:c1]


def _normalize_kernels(ks: list) -> list:
    cropped = [_crop_kernel(k) for k in ks]
    valid   = [c for c in cropped if c is not None]
    if not valid:
        return ks
    max_h = max(c.shape[0] for c in valid)
    max_w = max(c.shape[1] for c in valid)
    result = []
    for c in cropped:
        if c is None:
            result.append(None)
            continue
        h, w = c.shape[:2]
        ph = (max_h - h) // 2
        pw = (max_w - w) // 2
        if c.ndim == 2:
            out = np.zeros((max_h, max_w), dtype=c.dtype)
            out[ph:ph + h, pw:pw + w] = c
        else:
            out = np.zeros((max_h, max_w, c.shape[2]), dtype=c.dtype)
            out[ph:ph + h, pw:pw + w] = c
        result.append(out)
    return result


def _per_row_figure(row, out_path: Path):
    test_name   = row["test_name"]
    image_name  = row.get("image_name", "")
    kernel_name = row.get("kernel_name", "")

    dist_file = DATASET_DIR / "distorted"             / f"{test_name}.png"
    orig_file = DATASET_DIR / "originals"             / f"{image_name}.png"
    gt_kfile  = (DATASET_DIR / "ground_truth_filters" / f"{kernel_name}_kernel.png"
                 if kernel_name else None)

    orig     = _imread_safe(orig_file)
    dist     = _imread_safe(dist_file)
    rest_std = _imread_safe(row.get("restored_path__standard", ""))
    rest_par = _imread_safe(row.get("restored_path__pareto",   ""))
    rest_bst = _imread_safe(row.get("restored_path__best_hpo", ""))
    gt_k     = _imread_safe(gt_kfile) if gt_kfile else None
    est_std  = _imread_safe(row.get("kernel_path__standard",   ""))
    est_par  = _imread_safe(row.get("kernel_path__pareto",     ""))
    est_bst  = _imread_safe(row.get("kernel_path__best_hpo",   ""))

    _k_norm  = _normalize_kernels([gt_k, est_std, est_par, est_bst])
    gt_k, est_std, est_par, est_bst = _k_norm

    psnr_b   = row.get("psnr_blurred");   ssim_b   = row.get("ssim_blurred")
    psnr_std = row.get("psnr__standard");  ssim_std = row.get("ssim__standard")
    psnr_par = row.get("psnr__pareto");    ssim_par = row.get("ssim__pareto")
    psnr_bst = row.get("psnr__best_hpo");  ssim_bst = row.get("ssim__best_hpo")

    col_titles = [
        "Оригинал",
        "Искаженное",
        "Стандартные параметры",
        "Параметры Парето",
        "Лучшее оптимизации",
    ]
    images  = [orig,   dist,    rest_std,  rest_par,  rest_bst]
    kernels = [None,   gt_k,    est_std,   est_par,   est_bst]
    metrics = [
        None,
        (psnr_b,   ssim_b),
        (psnr_std, ssim_std),
        (psnr_par, ssim_par),
        (psnr_bst, ssim_bst),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(25, 9))
    for ax, img, title, kern, met in zip(axes, images, col_titles, kernels, metrics):
        ax.set_title(title, fontsize=_BASE_FONT, pad=6)
        ax.axis("off")
        if img is not None:
            cmap = "gray" if img.ndim == 2 else None
            ax.imshow(img, cmap=cmap)
        else:
            ax.text(0.5, 0.5, "missing", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=_BASE_FONT)

        if met is not None:
            p, s = met
            ax.text(0.5, -0.02,
                    f"PSNR = {_fmt(p)}   SSIM = {_fmt(s)}",
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=_BASE_FONT - 2,
                    clip_on=False)

        if kern is not None:
            inset = ax.inset_axes([0.63, 0.63, 0.35, 0.35])
            cmap_k = "gray" if kern.ndim == 2 else None
            inset.imshow(kern, cmap=cmap_k)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_edgecolor("white")
                spine.set_linewidth(2.5)

    fig.tight_layout(pad=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _summary_bars(df: pd.DataFrame, table_stem: str, out_path: Path):
    """Бары средних метрик: standard vs pareto vs best_hpo."""
    modes_colors = [
        ("standard", "#7e57c2"),
        ("pareto",   "#26a69a"),
        ("best_hpo", "#ef6c00"),
    ]
    metric_names = ["psnr", "ssim", "error_ratio", "time_sec"]

    rows_to_plot: list[tuple] = []  # (metric, [(mode, val), ...])
    for m in metric_names:
        vals = []
        for mode, _ in modes_colors:
            col = f"{m}__{mode}"
            if col in df.columns:
                vals.append((mode, float(df[col].mean())))
        if vals:
            rows_to_plot.append((m, vals))

    if not rows_to_plot:
        return

    n = len(rows_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    color_map = dict(modes_colors)
    for ax, (lbl, vals) in zip(axes, rows_to_plot):
        labels = [v[0] for v in vals]
        heights = [v[1] for v in vals]
        colors  = [color_map.get(l, "#888") for l in labels]
        ax.bar(labels, heights, color=colors)
        for i, v in enumerate(heights):
            ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax.set_title(lbl)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{table_stem} — средние по {len(df)} строкам", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _per_metric_scatter(df: pd.DataFrame, table_stem: str, out_path: Path):
    pairs = [("psnr", "PSNR"), ("ssim", "SSIM")]
    compare_modes = [
        ("pareto",   "#26a69a", "pareto"),
        ("best_hpo", "#ef6c00", "best HPO"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (m, name) in zip(axes, pairs):
        cs = f"{m}__standard"
        if cs not in df.columns:
            ax.axis("off"); continue
        x_std = df[cs].to_numpy()
        all_vals = [x_std]
        for cmode, color, label in compare_modes:
            cp = f"{m}__{cmode}"
            if cp not in df.columns:
                continue
            y = df[cp].to_numpy()
            all_vals.append(y)
            ax.scatter(x_std, y, s=32, alpha=0.75, c=color,
                       edgecolor="black", lw=0.4, label=label, zorder=3)
            for xi, yi, lbl in zip(x_std, y, df["test_name"]):
                ax.annotate(lbl, (xi, yi), fontsize=5, alpha=0.65)
        lo = float(np.nanmin([np.nanmin(v) for v in all_vals]))
        hi = float(np.nanmax([np.nanmax(v) for v in all_vals]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="equality")
        ax.set_xlabel(f"{name}  standard")
        ax.set_ylabel(f"{name}  other mode")
        ax.set_title(f"{name}: vs standard")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(table_stem, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _process_table(table_dir: Path):
    combined = table_dir / "results_combined.csv"
    if not combined.exists():
        print(f"  ! {combined} не найден — пропуск")
        return
    df = pd.read_csv(combined)
    print(f"  {table_dir.name}: {len(df)} строк")

    visuals_dir = table_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        prefix = f"{int(row['row_idx']):03d}_{row['test_name']}"
        _per_row_figure(row, visuals_dir / f"{prefix}.png")

    _summary_bars(df,     table_dir.name, visuals_dir / "_summary_bars.png")
    _per_metric_scatter(df, table_dir.name, visuals_dir / "_summary_scatter.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",  default=None, help="Один алгоритм по label")
    parser.add_argument("--table", default=None,
                        help="Один table_stem (например pareto_PSNR_front)")
    args = parser.parse_args()

    if not OUTPUT_ROOT.exists():
        print(f"! {OUTPUT_ROOT} не существует"
              f"run_all_dataset_esm_pareto_comp.py")
        return

    for algo_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not algo_dir.is_dir():
            continue
        if args.algo and algo_dir.name != args.algo:
            continue
        print(f"\n[{algo_dir.name}]")
        for table_dir in sorted(algo_dir.iterdir()):
            if not table_dir.is_dir():
                continue
            if args.table and table_dir.name != args.table:
                continue
            _process_table(table_dir)
    print("\nГОТОВО.")


if __name__ == "__main__":
    main()
