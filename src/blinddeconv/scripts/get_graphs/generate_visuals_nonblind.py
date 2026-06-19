#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация графиков сравнения non-blind методов деконволюции.

Читает результаты из presentation_graphics_nonblind/
Все подписи пропускаются через presentation_labels.json.

Графики:
  1. metrics_by_image_{metric}.png  - средние метрики по изображениям (+среднее)
  2. metrics_by_kernel_{metric}.png - средние метрики по ядрам (+среднее)
  3. visual_{original}.png          - таблица: строки=методы, столбцы=ядра
       В ячейке размытого изображения показывается само размытое + ядро рядом
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.image as mpimg
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

RESULTS_ROOT  = PROJECT_ROOT / "presentation_graphics_nonblind"
DATASET_DIR   = PROJECT_ROOT / "images" / "compare_data" / "anton" / "Grid_Test"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"
KERNELS_DIR   = DATASET_DIR / "ground_truth_filters"
LABELS_CONFIG = PROJECT_ROOT / "presentation_labels.json"

OUT_DIR = RESULTS_ROOT / "comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE  = ["#2176AE", "#E05929", "#57A773", "#B5338A", "#F2C12E", "#1B998B"]
MARKERS  = ["o", "s", "^", "D", "v", "*"]

_LMAP: dict = {}


def _load_labels() -> dict:
    if not LABELS_CONFIG.exists():
        return {}
    try:
        raw = json.loads(LABELS_CONFIG.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items()
                if isinstance(k, str) and not k.startswith("_") and isinstance(v, str)}
    except Exception:
        return {}


def L(name: str) -> str:
    if not isinstance(name, str):
        name = str(name)
    return _LMAP.get(name, name)


def discover_methods() -> list[str]:
    methods = []
    for d in sorted(RESULTS_ROOT.iterdir()):
        if d.is_dir() and not d.name.startswith("comparison") and d.name != "tmp":
            csv = d / f"results_{d.name}.csv"
            if csv.exists():
                methods.append(d.name)
    return methods


def load_data(methods: list[str]) -> dict[str, pd.DataFrame]:
    global_csv = RESULTS_ROOT / "all_results.csv"

    if not methods and global_csv.exists():
        df_all = pd.read_csv(global_csv)
        result = {}
        for m, grp in df_all.groupby("method"):
            valid = grp.dropna(subset=["psnr_out"])
            if not valid.empty:
                result[str(m)] = valid.reset_index(drop=True)
        return result

    result = {}
    for m in methods:
        csv_path = RESULTS_ROOT / m / f"results_{m}.csv"
        if not csv_path.exists():
            if global_csv.exists():
                df_g = pd.read_csv(global_csv)
                sub = df_g[df_g["method"] == m].dropna(subset=["psnr_out"])
                if not sub.empty:
                    result[m] = sub.reset_index(drop=True)
            continue
        df = pd.read_csv(csv_path).dropna(subset=["psnr_out"])
        if not df.empty:
            result[m] = df.reset_index(drop=True)
    return result


def _load_img(path) -> np.ndarray | None:
    if not path:
        return None
    p = Path(str(path))
    if not p.exists():
        return None
    try:
        img = mpimg.imread(str(p))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    except Exception:
        return None


def _load_gray(path) -> np.ndarray | None:
    img = _load_img(path)
    if img is None:
        return None
    if img.ndim == 3:
        return np.mean(img, axis=2).astype(np.float32)
    return img.astype(np.float32)


def _find_original(name: str) -> Path | None:
    for ext in [".png", ".jpg", ".bmp", ".tif"]:
        p = ORIGINALS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _find_blurred(filename: str) -> Path | None:
    p = DISTORTED_DIR / filename
    return p if p.exists() else None


def _find_kernel(kernel_name: str) -> Path | None:
    if not KERNELS_DIR.exists():
        return None
    for f in sorted(KERNELS_DIR.iterdir()):
        if f.is_file() and kernel_name in f.stem:
            return f
    return None


def _restored_path(method: str, filename: str) -> Path:
    return RESULTS_ROOT / method / "restored" / filename


def _show(ax, img, cmap="gray", miss_color="#cccccc"):
    ax.set_xticks([])
    ax.set_yticks([])
    if img is not None:
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
    else:
        ax.set_facecolor(miss_color)


def _add_kernel_inset(ax, kernel_img: np.ndarray,
                      frac: float = 0.25, pad: float = 0.012,
                      border_color: str = "white", border_lw: float = 1.5):
    """Квадратная миниатюра ядра в правом верхнем углу осей ax."""
    if kernel_img is None:
        return
    inset = ax.inset_axes([1 - frac - pad, 1 - frac - pad, frac, frac])
    inset.imshow(kernel_img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    inset.set_aspect("equal", adjustable="box")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_lw)


def _metric_str(psnr, ssim) -> str:
    def _f(v, fmt):
        try:
            fv = float(v)
            return (fmt % fv) if not math.isnan(fv) else "—"
        except Exception:
            return "—"
    return "PSNR %s  SSIM %s" % (_f(psnr, "%.2f"), _f(ssim, "%.4f"))


def _color(i):
    return PALETTE[i % len(PALETTE)]


# ГРАФИК 1 & 2: bar-чарты метрик по изображениям / ядрам

def _bar_metrics(all_data: dict, group_col: str, out_dir: Path, tag: str):
    """
    group_col = 'original' или 'kernel'
    tag       = 'image'    или 'kernel'
    """
    methods = list(all_data.keys())
    df_all = pd.concat(all_data.values(), ignore_index=True)
    groups = sorted(df_all[group_col].dropna().unique())
    groups_plus = list(groups) + ["_mean"]
    x = np.arange(len(groups_plus))
    bw = 0.75 / max(len(methods), 1)

    for metric, ylabel, title_suf in [
        ("psnr_out",  "PSNR, дБ",  "PSNR"),
        ("ssim_out",  "SSIM",       "SSIM"),
        ("isnr",      "ISNR, дБ",   "ISNR"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5.6))

        for mi, method in enumerate(methods):
            df_m = all_data[method]
            vals = []
            for g in groups:
                sub = df_m[df_m[group_col] == g][metric].dropna()
                vals.append(float(sub.mean()) if not sub.empty else float("nan"))
            all_sub = df_m[metric].dropna()
            vals.append(float(all_sub.mean()) if not all_sub.empty else float("nan"))

            offsets = x + mi * bw - bw * (len(methods) - 1) / 2
            bars = ax.bar(offsets, vals, width=bw * 0.92,
                          color=_color(mi), alpha=0.88,
                          label=L(method), edgecolor="grey", linewidth=0.4)

            for xi, v in zip(offsets, vals):
                if not math.isnan(v):
                    fmt = "%.4f" if metric == "ssim_out" else "%.2f"
                    ax.text(xi, v + abs(v) * 0.008, fmt % v,
                            ha="center", va="bottom", fontsize=7)

        ax.axvline(x[-1] - 0.5, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

        xlabels = [L(g) for g in groups] + ["Среднее"]
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{title_suf} по {'изображениям' if tag == 'image' else 'ядрам'}",
                     fontsize=14)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if metric == "isnr":
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

        plt.tight_layout()
        fname = f"metrics_by_{tag}_{metric.replace('_out','')}"
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        fig.savefig(out_dir / f"{fname}.pdf")
        print(f"  -> {fname}")
        plt.close(fig)
    fig_leg, ax_leg = plt.subplots(figsize=(4, 0.45 * len(methods)))
    ax_leg.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_color(mi), alpha=0.88, edgecolor="grey", linewidth=0.4)
        for mi, method in enumerate(methods)
    ]
    labels = [L(m) for m in methods]
    ax_leg.legend(handles, labels, fontsize=11, loc="center",
                  frameon=True, framealpha=0.9, edgecolor="grey")
    plt.tight_layout()
    leg_fname = f"legend_{tag}"
    fig_leg.savefig(out_dir / f"{leg_fname}.png", dpi=150, bbox_inches="tight")
    fig_leg.savefig(out_dir / f"{leg_fname}.pdf", bbox_inches="tight")
    print(f"  -> {leg_fname}")
    plt.close(fig_leg)


# =============================================================================
# ГРАФИК 3: визуальная таблица сранения
#   строки: оригинал | размытое+ядро | method1 | method2 | ...
#   столбцы: ядра
#
#   Ячейка "размытое+ядро" делится 2:1 (размытое слева, ядро справа)
# =============================================================================

def plot_visual_tables(all_data: dict, out_dir: Path):
    methods = list(all_data.keys())
    df_all = pd.concat(all_data.values(), ignore_index=True)
    originals = sorted(df_all["original"].dropna().unique())
    kernels   = sorted(df_all["kernel"].dropna().unique())

    # Строки таблицы: 0=original, 1=blurred+kernel, 2..=methods
    # Ячейка 1 строки делится на 2 вложенных: blurred и kernel (2:1 по ширине)

    n_method_rows = len(methods)
    n_rows = 2 + n_method_rows          # orig + blur/kernel + methods
    n_cols = len(kernels)

    cell = 2.6           # ширина ячейки
    label_w = 3.2        # ширина первого столбца с подписью
    header_h = 0.55      # высота заголовка строки ядра
    row_h = cell         # высота обычной строки
    blur_row_h = cell    # строка размытого (та же высота, делится внутри)

    fig_w = label_w + n_cols * cell
    fig_h = header_h + row_h + blur_row_h + n_method_rows * row_h + 0.4

    for orig in originals:
        fig = plt.figure(figsize=(fig_w, fig_h))
        fig.suptitle(L(orig), fontsize=13, y=0.995)

        height_ratios = [header_h / row_h] + [1.0] * n_rows
        width_ratios  = [label_w / cell] + [1.0] * n_cols

        gs = mgridspec.GridSpec(
            n_rows + 1, n_cols + 1,
            figure=fig,
            left=0.01, right=0.99, top=0.97, bottom=0.02,
            wspace=0.04, hspace=0.06,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
        )

        ax_corner = fig.add_subplot(gs[0, 0])
        ax_corner.set_axis_off()

        for ci, kern in enumerate(kernels):
            ax_hdr = fig.add_subplot(gs[0, ci + 1])
            ax_hdr.text(0.5, 0.5, L(kern), ha="center", va="center",
                        fontsize=9, transform=ax_hdr.transAxes)
            ax_hdr.set_axis_off()

        row_labels = ["Оригинал", "Размытое"] + [L(m) for m in methods]
        for ri, lbl in enumerate(row_labels):
            ax_lbl = fig.add_subplot(gs[ri + 1, 0])
            ax_lbl.text(1.0, 0.5, lbl, ha="right", va="center",
                        fontsize=8.5,
                        transform=ax_lbl.transAxes)
            ax_lbl.set_axis_off()
        for ci, kern in enumerate(kernels):

            # Строка 0: оригинал
            orig_img = _load_gray(_find_original(orig))
            ax_o = fig.add_subplot(gs[1, ci + 1])
            _show(ax_o, orig_img)

            # Строка 1: размытое + ядро
            filename = None
            for m in methods:
                sub = all_data[m][(all_data[m]["original"] == orig) &
                                  (all_data[m]["kernel"] == kern)]
                if not sub.empty:
                    filename = sub.iloc[0]["filename"]
                    break

            blur_img   = _load_gray(_find_blurred(filename)) if filename else None
            kernel_img = _load_gray(_find_kernel(kern))

            ax_blur = fig.add_subplot(gs[2, ci + 1])
            _show(ax_blur, blur_img)
            _add_kernel_inset(ax_blur, kernel_img)

            psnr_b_vals = []
            for m in methods:
                sub = all_data[m][(all_data[m]["original"] == orig) &
                                  (all_data[m]["kernel"] == kern)]["psnr_blurred"].dropna()
                if not sub.empty:
                    psnr_b_vals.append(float(sub.iloc[0]))
                    break
            if psnr_b_vals:
                ax_blur.set_xlabel("PSNR %.2f" % psnr_b_vals[0],
                                   fontsize=6, labelpad=2)

            for mi, method in enumerate(methods):
                df_m = all_data[method]
                sub = df_m[(df_m["original"] == orig) & (df_m["kernel"] == kern)]
                row = sub.iloc[0] if not sub.empty else None

                rest_img = _load_gray(_restored_path(method, row["filename"])) \
                    if row is not None else None

                ax_r = fig.add_subplot(gs[3 + mi, ci + 1])
                _show(ax_r, rest_img)

                if row is not None:
                    ax_r.set_xlabel(
                        _metric_str(row.get("psnr_out"), row.get("ssim_out")),
                        fontsize=6, labelpad=2
                    )

        fname = "visual_%s" % orig
        fig.savefig(out_dir / f"{fname}.png", dpi=150)
        fig.savefig(out_dir / f"{fname}.pdf")
        print(f"  -> {fname}")
        plt.close(fig)


# ГРАФИК 4: горизонтальное сравнение методов для одной пары (orig, kernel)
#   Строка: Оригинал | Размытое+ядро | Метод1 | Метод2 | ...
#   Один файл на пару (original, kernel)


def plot_row_comparison(all_data: dict, out_dir: Path):
    """
    Для каждой пары (original, kernel) строит горизонтальную полосу:
      Оригинал | Размытое+ядро | Метод1 | Метод2 | ... | МетодN

    Файлы: row_{original}_{kernel}.png/pdf
    """
    methods   = list(all_data.keys())
    df_all    = pd.concat(all_data.values(), ignore_index=True)
    originals = sorted(df_all["original"].dropna().unique())
    kernels   = sorted(df_all["kernel"].dropna().unique())

    # Столбцы: orig, blur, method0, method1, ...
    col_labels = ["Оригинал", "Размытое"] + [L(m) for m in methods]
    n_cols     = len(col_labels)

    for orig in originals:
        for kern in kernels:
            any_result = any(
                not all_data[m][
                    (all_data[m]["original"] == orig) &
                    (all_data[m]["kernel"]   == kern)
                ].empty
                for m in methods
            )
            if not any_result:
                continue

            fig = plt.figure(figsize=(14, 2.8))
            gs = mgridspec.GridSpec(
                2, n_cols,
                figure=fig,
                left=0.01, right=0.99, top=0.82, bottom=0.13,
                wspace=0.03, hspace=0.03,
                height_ratios=[0.12, 1.0],
            )

            fig.suptitle(f"{L(orig)} — {L(kern)}", fontsize=11, y=0.97)

            for ci, lbl in enumerate(col_labels):
                ax_hdr = fig.add_subplot(gs[0, ci])
                ax_hdr.text(0.5, 0.5, lbl, ha="center", va="center",
                            fontsize=9.5,
                            transform=ax_hdr.transAxes)
                ax_hdr.set_axis_off()

            orig_img = _load_gray(_find_original(orig))
            ax_o = fig.add_subplot(gs[1, 0])
            _show(ax_o, orig_img)

            filename = None
            for m in methods:
                sub = all_data[m][
                    (all_data[m]["original"] == orig) &
                    (all_data[m]["kernel"]   == kern)
                ]
                if not sub.empty:
                    filename = sub.iloc[0]["filename"]
                    break

            blur_img   = _load_gray(_find_blurred(filename)) if filename else None
            kernel_img = _load_gray(_find_kernel(kern))

            ax_b = fig.add_subplot(gs[1, 1])
            _show(ax_b, blur_img)
            _add_kernel_inset(ax_b, kernel_img)

            psnr_b = float("nan")
            for m in methods:
                sub = all_data[m][
                    (all_data[m]["original"] == orig) &
                    (all_data[m]["kernel"]   == kern)
                ]["psnr_blurred"].dropna()
                if not sub.empty:
                    psnr_b = float(sub.iloc[0])
                    break

            ssim_b = float("nan")
            for m in methods:
                sub = all_data[m][
                    (all_data[m]["original"] == orig) &
                    (all_data[m]["kernel"]   == kern)
                ]["ssim_blurred"].dropna()
                if not sub.empty:
                    ssim_b = float(sub.iloc[0])
                    break
            
            if not math.isnan(psnr_b):
                ax_b.set_xlabel(f"PSNR {psnr_b:.2f} SSIM {ssim_b:.4f}", fontsize=8, labelpad=3)

            for mi, method in enumerate(methods):
                df_m = all_data[method]
                sub  = df_m[
                    (df_m["original"] == orig) &
                    (df_m["kernel"]   == kern)
                ]
                row = sub.iloc[0] if not sub.empty else None

                rest_img = _load_gray(_restored_path(method, row["filename"])) \
                    if row is not None else None

                ax_r = fig.add_subplot(gs[1, 2 + mi])
                _show(ax_r, rest_img, miss_color="#ffe0e0")

                if row is not None:
                    ax_r.set_xlabel(
                        _metric_str(row.get("psnr_out"), row.get("ssim_out")),
                        fontsize=8, labelpad=3,
                    )

            fname = f"row_{orig}_{kern}"
            fig.savefig(out_dir / f"{fname}.png", dpi=150)
            fig.savefig(out_dir / f"{fname}.pdf")
            print(f"  -> {fname}")
            plt.close(fig)


def main():
    global _LMAP
    _LMAP = _load_labels()
    print("Подписи: %d записей" % len(_LMAP))

    methods = discover_methods()
    print("Методы из папок: %s" % methods)

    all_data = load_data(methods)
    if not all_data:
        print("Нет данных в %s" % RESULTS_ROOT)
        return

    methods = list(all_data.keys())
    df_all = pd.concat(all_data.values(), ignore_index=True)
    kernels  = sorted(df_all["kernel"].dropna().unique())
    originals = sorted(df_all["original"].dropna().unique())

    print("Методы: %s" % methods)
    print("Изображения (%d): %s" % (len(originals), originals))
    print("Ядра (%d): %s" % (len(kernels), kernels))
    print("Результаты -> %s\n" % OUT_DIR)

    print("[1] Метрики по изображениям...")
    _bar_metrics(all_data, "original", OUT_DIR, "image")

    print("[2] Метрики по ядрам...")
    _bar_metrics(all_data, "kernel", OUT_DIR, "kernel")

    print("[3] Визуальные таблицы по оригиналам...")
    plot_visual_tables(all_data, OUT_DIR)

    print("[4] Горизонтальное сравнение (orig x kernel)...")
    plot_row_comparison(all_data, OUT_DIR)

    print("\n[OK] Готово. Файлы: %s" % OUT_DIR)


if __name__ == "__main__":
    main()
