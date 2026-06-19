#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация таблиц/графиков сравнения приоров для слепой деконволюции.

Подписи на графиках управляются через presentation_labels.json
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import visualisation as vis

RESULTS_ROOT  = PROJECT_ROOT / "presentation_graphics_priors"
DATASET_NAME  = "priors"
LABELS_CONFIG = PROJECT_ROOT / "presentation_labels.json"

OUT_DIR = RESULTS_ROOT / "comparison_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_PATH_COLS = ["restored_path", "kernel_path", "gt_kernel_path", "original_path"]


def _load_label_map() -> dict:
    if not LABELS_CONFIG.exists():
        return {}
    try:
        raw = json.loads(LABELS_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {k: v for k, v in raw.items()
                if isinstance(k, str) and not k.startswith("_")
                and isinstance(v, str)}
    except Exception as e:
        print(f"  Ошибка чтения {LABELS_CONFIG}: {e}")
        return {}


def _decode(s: object) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    return vis.decode_in_text(s).strip()


def _fix_path(p: object) -> str:
    if p is None:
        return p
    s = str(p)
    if not s or s == "nan":
        return s
    s_norm = s.replace("\\", "/")
    anchor_pg = "presentation_graphics/"
    idx = s_norm.find(anchor_pg)
    if idx >= 0 and not s_norm[idx:].startswith("presentation_graphics_priors/"):
        tail = s_norm[idx + len(anchor_pg):]
        return str(PROJECT_ROOT / "presentation_graphics_priors" / tail)
    anchor_img = "images/compare_data/"
    idx = s_norm.find(anchor_img)
    if idx >= 0:
        return str(PROJECT_ROOT / s_norm[idx:])
    return s


def _rewrite_paths(df: pd.DataFrame) -> None:
    for col in _PATH_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_fix_path)


def load_all_data() -> dict:
    all_data = {}
    for alg_dir in sorted(RESULTS_ROOT.iterdir()):
        if not alg_dir.is_dir() or alg_dir.name.startswith("comparison"):
            continue
        csvs = list(alg_dir.glob("all_results_*.csv"))
        if not csvs:
            csvs = list(alg_dir.glob("*/results_*.csv"))
        if not csvs:
            print(f"  Пропуск {alg_dir.name}: нет CSV")
            continue
        frames = []
        for csv_file in csvs:
            try:
                frames.append(pd.read_csv(csv_file))
            except Exception as e:
                print(f"  Ошибка чтения {csv_file}: {e}")
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        _rewrite_paths(combined)
        if "dataset" not in combined.columns:
            combined["dataset"] = DATASET_NAME
        if "noise_name" in combined.columns:
            combined["noise_name"] = combined["noise_name"].fillna("clean")
        else:
            combined["noise_name"] = "clean"
        all_data[alg_dir.name] = combined
        print(f"  ? {alg_dir.name}: {len(combined)} записей")
    return all_data


def _short_prior_name(full_name: str) -> str:
    if "(" in full_name and ")" in full_name:
        inner = full_name[full_name.rfind("(") + 1: full_name.rfind(")")]
        inner = inner.replace("_", " ")
        return _decode(inner).strip()
    return _decode(full_name.replace("_", " ")).strip()


def _rank_priors(frames_clean: dict) -> list:
    def _mean_psnr(df_a):
        v = df_a["psnr"].dropna()
        return v.mean() if len(v) > 0 else -np.inf
    return sorted(frames_clean.keys(), key=lambda a: _mean_psnr(frames_clean[a]), reverse=True)


def _load_img(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(str(p))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return img
    except Exception:
        return None


def _load_gray(path):
    img = _load_img(path)
    if img is None:
        return None
    if img.ndim == 3:
        return np.mean(img[:, :, :3], axis=2)
    return img


def _load_kernel(path):
    img = _load_img(path)
    if img is None:
        return None
    if img.ndim == 3:
        img = np.mean(img[:, :, :3], axis=2)
    return img


def _pad_square(k: np.ndarray, target: int) -> np.ndarray:
    ph = (target - k.shape[0]) // 2
    pw = (target - k.shape[1]) // 2
    return np.pad(k, ((ph, target - k.shape[0] - ph),
                      (pw, target - k.shape[1] - pw)),
                  mode="constant", constant_values=0)


def _show_cell(ax, img, cmap="gray", fc_miss="#cccccc"):
    ax.set_xticks([])
    ax.set_yticks([])
    if img is not None:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    else:
        ax.set_facecolor(fc_miss)


def _fmt_metrics(psnr, ssim) -> str:
    try:
        pb = f"{float(psnr):.2f}" if pd.notna(psnr) and not np.isnan(float(psnr)) else "—"
    except Exception:
        pb = "—"
    try:
        sb = f"{float(ssim):.4f}" if pd.notna(ssim) and not np.isnan(float(ssim)) else "—"
    except Exception:
        sb = "—"
    return f"PSNR {pb}  SSIM {sb}"


def _blur_path_from_row(row) -> "Path | None":
    orig_p = row.get("original_path")
    dist_f = row.get("distorted_file")
    if orig_p and pd.notna(orig_p) and dist_f and pd.notna(dist_f):
        return Path(orig_p).parent.parent / "distorted" / str(dist_f)
    return None



#ТАБЛИЦА 1: для каждого приора — оригинал / смазанное / восстановленное

def plot_per_prior_visual(all_data: dict, out_dir: Path) -> None:
    for alg_name, df in all_data.items():
        short = _short_prior_name(alg_name)
        df_clean = df[df["noise_name"] == "clean"].copy()
        if df_clean.empty:
            print(f"  [{short}] нет clean-записей, пропуск")
            continue
        images  = sorted(df_clean["image_name"].unique())
        kernels = sorted(df_clean["kernel_name"].unique())
        n_cols  = len(images) * len(kernels)
        cell, label_w = 2.5, 2.0
        fig = plt.figure(figsize=(label_w + n_cols * cell, 3 * cell + 0.7))
        gs = mgridspec.GridSpec(
            3, n_cols + 1, figure=fig,
            left=0.01, right=0.99, top=0.93, bottom=0.08,
            wspace=0.03, hspace=0.06,
            width_ratios=[label_w / cell] + [1.0] * n_cols,
        )
        for r, lbl in enumerate(["Оригинал", "Искажённое", "Восстановленное"]):
            ax = fig.add_subplot(gs[r, 0])
            ax.set_axis_off()
            ax.text(0.95, 0.5, lbl, ha="right", va="center", fontsize=8,
                    transform=ax.transAxes)
        fig.suptitle(short, fontsize=11, y=0.97)
        col = 0
        for img_name in images:
            for ker_name in kernels:
                rows = df_clean[(df_clean["image_name"] == img_name) &
                                (df_clean["kernel_name"] == ker_name)]
                c = col + 1
                if rows.empty:
                    for r in range(3):
                        ax = fig.add_subplot(gs[r, c])
                        _show_cell(ax, None)
                        ax.text(0.5, 0.5, "н/д", ha="center", va="center",
                                fontsize=8, color="gray", transform=ax.transAxes)
                    col += 1
                    continue
                row = rows.iloc[0]
                orig_p  = row.get("original_path")
                rest_p  = row.get("restored_path")
                gt_k_p  = row.get("gt_kernel_path")
                est_k_p = row.get("kernel_path")
                blur_p  = _blur_path_from_row(row)
                gt_k  = _load_kernel(gt_k_p  if pd.notna(gt_k_p)  else None)
                est_k = _load_kernel(est_k_p if pd.notna(est_k_p) else None)
                ks = [k for k in [gt_k, est_k] if k is not None]
                if ks:
                    t = max(max(k.shape) for k in ks)
                    gt_k  = _pad_square(gt_k,  t) if gt_k  is not None else None
                    est_k = _pad_square(est_k, t) if est_k is not None else None
                ax0 = fig.add_subplot(gs[0, c])
                _show_cell(ax0, _load_gray(orig_p if pd.notna(orig_p) else None))
                ax1 = fig.add_subplot(gs[1, c])
                _show_cell(ax1, _load_gray(blur_p))
                ax1.set_xlabel(_fmt_metrics(row.get("psnr_blurred"), row.get("ssim_blurred")),
                               fontsize=7, labelpad=2)
                if gt_k is not None:
                    vis._add_kernel_inset(ax1, gt_k, corner="top-right", frac=0.42)
                ax2 = fig.add_subplot(gs[2, c])
                _show_cell(ax2, _load_gray(rest_p if pd.notna(rest_p) else None),
                           fc_miss="#ffe0e0")
                ax2.set_xlabel(_fmt_metrics(row.get("psnr"), row.get("ssim")),
                               fontsize=7, labelpad=2)
                if est_k is not None:
                    vis._add_kernel_inset(ax2, est_k, corner="top-right", frac=0.42)
                col += 1
        stem = f"visual_{alg_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  > {stem}")
        plt.close(fig)


def _build_cross_prior_grid(frames_clean: dict, prior_names: list,
                             images: list, kernels: list,
                             title: str, out_dir: Path, stem: str) -> None:
    short_names = {a: _short_prior_name(a) for a in prior_names}
    n_prior = len(prior_names)
    n_cols  = len(images) * len(kernels)
    n_rows  = 2 + n_prior
    cell_w, cell_h = 2.0, 1.9
    label_w = 2.8
    fig = plt.figure(figsize=(label_w + n_cols * cell_w, n_rows * cell_h + 0.7))
    gs = mgridspec.GridSpec(
        n_rows, n_cols + 1, figure=fig,
        left=0.01, right=0.99, top=0.95, bottom=0.04,
        wspace=0.03, hspace=0.10,
        width_ratios=[label_w / cell_w] + [1.0] * n_cols,
    )
    fig.suptitle(title, fontsize=12, y=0.98)
    row_labels = ["Оригинал", "Искажённое"] + [short_names[a] for a in prior_names]
    for r, lbl in enumerate(row_labels):
        ax = fig.add_subplot(gs[r, 0])
        ax.set_axis_off()
        ax.text(0.96, 0.5, lbl, ha="right", va="center", fontsize=8,
                transform=ax.transAxes, multialignment="right")
    all_ks_global = []
    for df_a in frames_clean.values():
        for _, row in df_a.iterrows():
            for col_k in ("gt_kernel_path", "kernel_path"):
                k = _load_kernel(row.get(col_k))
                if k is not None:
                    all_ks_global.append(k)
    target_k = max(max(k.shape) for k in all_ks_global) if all_ks_global else 1
    col = 0
    for img_name in images:
        for ker_name in kernels:
            c = col + 1
            orig_p = blur_p = gt_k_p = None
            psnr_b = ssim_b = float("nan")
            for df_a in frames_clean.values():
                r0 = df_a[(df_a["image_name"] == img_name) &
                          (df_a["kernel_name"] == ker_name)]
                if not r0.empty:
                    r0r = r0.iloc[0]
                    orig_p = r0r.get("original_path")
                    gt_k_p = r0r.get("gt_kernel_path")
                    psnr_b = r0r.get("psnr_blurred", float("nan"))
                    ssim_b = r0r.get("ssim_blurred", float("nan"))
                    blur_p = _blur_path_from_row(r0r)
                    break
            gt_k = _load_kernel(gt_k_p if gt_k_p and pd.notna(gt_k_p) else None)
            gt_k = _pad_square(gt_k, target_k) if gt_k is not None else None
            est_ks = {}
            for alg_name in prior_names:
                r0 = frames_clean[alg_name]
                r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                k = _load_kernel(r0.iloc[0]["kernel_path"] if not r0.empty else None)
                est_ks[alg_name] = _pad_square(k, target_k) if k is not None else None
            ax = fig.add_subplot(gs[0, c])
            _show_cell(ax, _load_gray(orig_p if orig_p and pd.notna(orig_p) else None))
            ax = fig.add_subplot(gs[1, c])
            _show_cell(ax, _load_gray(blur_p))
            ax.set_xlabel(_fmt_metrics(psnr_b, ssim_b), fontsize=6.5, labelpad=2)
            if gt_k is not None:
                vis._add_kernel_inset(ax, gt_k, corner="top-right", frac=0.27)
            for ri, alg_name in enumerate(prior_names):
                ax = fig.add_subplot(gs[2 + ri, c])
                r0 = frames_clean[alg_name]
                r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                if not r0.empty:
                    r0r = r0.iloc[0]
                    _show_cell(ax, _load_gray(r0r.get("restored_path")), fc_miss="#ffe0e0")
                    ax.set_xlabel(_fmt_metrics(r0r.get("psnr"), r0r.get("ssim")),
                                  fontsize=6.5, labelpad=2)
                    if est_ks.get(alg_name) is not None:
                        vis._add_kernel_inset(ax, est_ks[alg_name],
                                              corner="top-right", frac=0.27)
                else:
                    _show_cell(ax, None, fc_miss="#ffe0e0")
                    ax.text(0.5, 0.5, "н/д", ha="center", va="center",
                            fontsize=7, color="gray", transform=ax.transAxes)
            col += 1
    fig.savefig(out_dir / f"{stem}.png", dpi=150)
    fig.savefig(out_dir / f"{stem}.pdf")
    print(f"  > {stem}")
    plt.close(fig)


#ТАБЛИЦА 2: кросс-приор сравнение (все оригиналы, все ядра)


def plot_cross_prior_comparison(all_data: dict, out_dir: Path) -> None:
    frames_clean = {a: df[df["noise_name"] == "clean"].copy()
                    for a, df in all_data.items()}
    frames_clean = {a: df for a, df in frames_clean.items() if not df.empty}
    if not frames_clean:
        return
    all_rows    = pd.concat(list(frames_clean.values()), ignore_index=True)
    images      = sorted(all_rows["image_name"].unique())
    kernels     = sorted(all_rows["kernel_name"].unique())
    prior_names = _rank_priors(frames_clean)
    _build_cross_prior_grid(frames_clean, prior_names, images, kernels,
                             title="Сравнение приоров",
                             out_dir=out_dir, stem="comparison_all_priors")

# ТАБЛИЦА 3: кросс-приор сравнение по отдельному оригиналу

def plot_per_image_comparison(all_data: dict, out_dir: Path) -> None:
    frames_clean = {a: df[df["noise_name"] == "clean"].copy()
                    for a, df in all_data.items()}
    frames_clean = {a: df for a, df in frames_clean.items() if not df.empty}
    if not frames_clean:
        return
    all_rows    = pd.concat(list(frames_clean.values()), ignore_index=True)
    images      = sorted(all_rows["image_name"].unique())
    kernels     = sorted(all_rows["kernel_name"].unique())
    prior_names = _rank_priors(frames_clean)
    for img_name in images:
        _build_cross_prior_grid(
            frames_clean, prior_names, [img_name], kernels,
            title=_decode(img_name),
            out_dir=out_dir,
            stem=f"comparison_{img_name}",
        )


#  ТАБЛИЦА 4: маленькие таблицы (1 ядро / 1 оригинал)
#  Строка 1: оригинал | искажённое
#  Строка 2: prior1   | prior2
#  Строка 3: prior3   | prior4

def plot_small_tables_per_kernel(all_data: dict, out_dir: Path) -> None:
    frames_clean = {a: df[df["noise_name"] == "clean"].copy()
                    for a, df in all_data.items()}
    frames_clean = {a: df for a, df in frames_clean.items() if not df.empty}
    if not frames_clean:
        return
    all_rows    = pd.concat(list(frames_clean.values()), ignore_index=True)
    images      = sorted(all_rows["image_name"].unique())
    kernels     = sorted(all_rows["kernel_name"].unique())
    prior_names = _rank_priors(frames_clean)
    short_names = {a: _short_prior_name(a) for a in prior_names}
    pairs = [(prior_names[i], prior_names[i + 1] if i + 1 < len(prior_names) else None)
             for i in range(0, len(prior_names), 2)]
    n_rows = 1 + len(pairs)
    all_ks_global = []
    for df_a in frames_clean.values():
        for _, row in df_a.iterrows():
            for col_k in ("gt_kernel_path", "kernel_path"):
                k = _load_kernel(row.get(col_k))
                if k is not None:
                    all_ks_global.append(k)
    target_k = max(max(k.shape) for k in all_ks_global) if all_ks_global else 1
    cell = 3.2
    for img_name in images:
        for ker_name in kernels:
            fig = plt.figure(figsize=(2 * cell, n_rows * cell + 0.8))
            gs = mgridspec.GridSpec(
                n_rows, 2, figure=fig,
                left=0.01, right=0.99, top=0.93, bottom=0.10,
                wspace=0.04, hspace=0.28,
            )
            fig.suptitle(f"{_decode(img_name)} — {_decode(ker_name)}", fontsize=14, y=0.97)
            orig_p = blur_p = gt_k_p = None
            psnr_b = ssim_b = float("nan")
            for df_a in frames_clean.values():
                r0 = df_a[(df_a["image_name"] == img_name) &
                          (df_a["kernel_name"] == ker_name)]
                if not r0.empty:
                    r0r = r0.iloc[0]
                    orig_p = r0r.get("original_path")
                    gt_k_p = r0r.get("gt_kernel_path")
                    psnr_b = r0r.get("psnr_blurred", float("nan"))
                    ssim_b = r0r.get("ssim_blurred", float("nan"))
                    blur_p = _blur_path_from_row(r0r)
                    break
            gt_k = _load_kernel(gt_k_p if gt_k_p and pd.notna(gt_k_p) else None)
            gt_k = _pad_square(gt_k, target_k) if gt_k is not None else None
            est_ks = {}
            for alg_name in prior_names:
                r0 = frames_clean[alg_name]
                r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                k = _load_kernel(r0.iloc[0]["kernel_path"] if not r0.empty else None)
                est_ks[alg_name] = _pad_square(k, target_k) if k is not None else None
            # Строка 0: оригинал | смазанное
            ax = fig.add_subplot(gs[0, 0])
            _show_cell(ax, _load_gray(orig_p if orig_p and pd.notna(orig_p) else None))
            ax.set_title("Оригинал", fontsize=16, pad=4)
            ax = fig.add_subplot(gs[0, 1])
            _show_cell(ax, _load_gray(blur_p))
            ax.set_title("Искажённое", fontsize=16, pad=4)
            ax.set_xlabel(_fmt_metrics(psnr_b, ssim_b), fontsize=11.5, labelpad=4)
            if gt_k is not None:
                vis._add_kernel_inset(ax, gt_k, corner="top-right", frac=0.42)
            for ri, (p1, p2) in enumerate(pairs):
                row_idx = 1 + ri
                ax = fig.add_subplot(gs[row_idx, 0])
                r0 = frames_clean[p1]
                r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                if not r0.empty:
                    r0r = r0.iloc[0]
                    _show_cell(ax, _load_gray(r0r.get("restored_path")), fc_miss="#ffe0e0")
                    ax.set_title(short_names[p1], fontsize=16, pad=4)
                    ax.set_xlabel(_fmt_metrics(r0r.get("psnr"), r0r.get("ssim")),
                                  fontsize=11.5, labelpad=4)
                    if est_ks.get(p1) is not None:
                        vis._add_kernel_inset(ax, est_ks[p1], corner="top-right", frac=0.42)
                else:
                    _show_cell(ax, None, fc_miss="#ffe0e0")
                    ax.set_title(short_names[p1], fontsize=16, pad=4)
                ax = fig.add_subplot(gs[row_idx, 1])
                if p2 is not None:
                    r0 = frames_clean[p2]
                    r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                    if not r0.empty:
                        r0r = r0.iloc[0]
                        _show_cell(ax, _load_gray(r0r.get("restored_path")), fc_miss="#ffe0e0")
                        ax.set_title(short_names[p2], fontsize=16, pad=4)
                        ax.set_xlabel(_fmt_metrics(r0r.get("psnr"), r0r.get("ssim")),
                                      fontsize=11.5, labelpad=4)
                        if est_ks.get(p2) is not None:
                            vis._add_kernel_inset(ax, est_ks[p2], corner="top-right", frac=0.42)
                    else:
                        _show_cell(ax, None, fc_miss="#ffe0e0")
                        ax.set_title(short_names[p2], fontsize=16, pad=4)
                else:
                    ax.set_axis_off()
            stem = f"small_{img_name}_{ker_name}"
            fig.savefig(out_dir / f"{stem}.png", dpi=200)
            fig.savefig(out_dir / f"{stem}.pdf")
            print(f"  > {stem}")
            plt.close(fig)

#ТАБЛИЦА 5: сравнение ядер

def plot_kernels_comparison(all_data: dict, out_dir: Path) -> None:
    frames_clean = {a: df[df["noise_name"] == "clean"].copy()
                    for a, df in all_data.items()}
    frames_clean = {a: df for a, df in frames_clean.items() if not df.empty}
    if not frames_clean:
        return
    all_rows    = pd.concat(list(frames_clean.values()), ignore_index=True)
    images      = sorted(all_rows["image_name"].unique())
    kernels     = sorted(all_rows["kernel_name"].unique())
    prior_names = _rank_priors(frames_clean)
    short_names = {a: _short_prior_name(a) for a in prior_names}
    for img_name in images:
        n_ker   = len(kernels)
        n_rows  = 1 + len(prior_names)
        cell    = 2.0
        label_w = 2.4
        fig = plt.figure(figsize=(label_w + n_ker * cell, n_rows * cell + 0.5))
        gs = mgridspec.GridSpec(
            n_rows, n_ker + 1, figure=fig,
            left=0.01, right=0.99, top=0.93, bottom=0.05,
            wspace=0.04, hspace=0.08,
            width_ratios=[label_w / cell] + [1.0] * n_ker,
        )
        fig.suptitle(f"Ядра — {_decode(img_name)}", fontsize=11, y=0.97)
        for r, lbl in enumerate(["GT ядро"] + [short_names[a] for a in prior_names]):
            ax = fig.add_subplot(gs[r, 0])
            ax.set_axis_off()
            ax.text(0.95, 0.5, lbl, ha="right", va="center", fontsize=8,
                    transform=ax.transAxes)
        for ki, ker_name in enumerate(kernels):
            c = ki + 1
            gt_k_p = None
            for df_a in frames_clean.values():
                r0 = df_a[(df_a["image_name"] == img_name) &
                          (df_a["kernel_name"] == ker_name)]
                if not r0.empty:
                    gt_k_p = r0.iloc[0].get("gt_kernel_path")
                    break
            gt_k = _load_kernel(gt_k_p if gt_k_p and pd.notna(gt_k_p) else None)
            est_ks = {}
            for alg_name in prior_names:
                r0 = frames_clean[alg_name]
                r0 = r0[(r0["image_name"] == img_name) & (r0["kernel_name"] == ker_name)]
                est_ks[alg_name] = _load_kernel(
                    r0.iloc[0]["kernel_path"] if not r0.empty else None)
            all_ks_cell = [k for k in [gt_k] + list(est_ks.values()) if k is not None]
            if all_ks_cell:
                t = max(max(k.shape) for k in all_ks_cell)
                gt_k   = _pad_square(gt_k, t) if gt_k is not None else None
                est_ks = {a: (_pad_square(k, t) if k is not None else None)
                          for a, k in est_ks.items()}
            def _show_kernel(ax_obj, k, title=""):
                ax_obj.set_xticks([])
                ax_obj.set_yticks([])
                if k is not None:
                    kn = k - k.min()
                    if kn.max() > 0:
                        kn = kn / kn.max()
                    ax_obj.imshow(kn, cmap="hot", vmin=0, vmax=1)
                else:
                    ax_obj.set_facecolor("#cccccc")
                    ax_obj.text(0.5, 0.5, "нет файла", ha="center", va="center",
                                fontsize=7, color="gray", transform=ax_obj.transAxes)
                if title:
                    ax_obj.set_title(title, fontsize=8, pad=2)
            ax = fig.add_subplot(gs[0, c])
            _show_kernel(ax, gt_k, title=_decode(ker_name))
            for ri, alg_name in enumerate(prior_names):
                ax = fig.add_subplot(gs[1 + ri, c])
                _show_kernel(ax, est_ks.get(alg_name))
        stem = f"kernels_{img_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=180)
        fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  > {stem}")
        plt.close(fig)

#ТАБЛИЦА 6: метрики PSNR/SSIM


def plot_metrics_table(all_data: dict, out_dir: Path) -> None:
    frames_clean = {a: df[df["noise_name"] == "clean"]
                    for a, df in all_data.items()}
    frames_clean = {a: df for a, df in frames_clean.items() if not df.empty}
    if not frames_clean:
        return
    prior_names = _rank_priors(frames_clean)
    short_names = [_short_prior_name(a) for a in prior_names]
    all_rows    = pd.concat(list(frames_clean.values()), ignore_index=True)
    kernels     = sorted(all_rows["kernel_name"].unique())
    colors      = plt.cm.tab10(np.linspace(0, 0.9, len(prior_names)))
    x           = np.arange(len(kernels))
    width       = 0.8 / len(prior_names)
    fig, axes = plt.subplots(1, 2, figsize=(max(len(kernels) * 2.5, 8), 5))
    for metric, ax, ylabel in [("psnr", axes[0], "PSNR, дБ"),
                                ("ssim", axes[1], "SSIM")]:
        for i, alg_name in enumerate(prior_names):
            vals = [frames_clean[alg_name][
                        frames_clean[alg_name]["kernel_name"] == k
                    ][metric].dropna().mean() for k in kernels]
            ax.bar(x + i * width, vals, width=width, color=colors[i],
                   alpha=0.85, label=short_names[i],
                   edgecolor="grey", linewidth=0.4)
        ax.set_xticks(x + width * (len(prior_names) - 1) / 2)
        ax.set_xticklabels([_decode(k) for k in kernels], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} по ядрам", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "metrics_by_kernel.png", dpi=150)
    fig.savefig(out_dir / "metrics_by_kernel.pdf")
    print("  > metrics_by_kernel")
    plt.close(fig)
    fig2, axes2 = plt.subplots(1, 2, figsize=(max(len(prior_names) * 1.8, 6), 4))
    for metric, ax, ylabel in [("psnr", axes2[0], "PSNR, дБ"),
                                ("ssim", axes2[1], "SSIM")]:
        vals = [frames_clean[a][metric].dropna().mean() for a in prior_names]
        ax.bar(np.arange(len(prior_names)), vals, color=colors[:len(prior_names)],
               alpha=0.85, edgecolor="grey", linewidth=0.4)
        ax.set_xticks(np.arange(len(prior_names)))
        ax.set_xticklabels(short_names, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Средний {metric.upper()}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig2.savefig(out_dir / "metrics_mean.png", dpi=150)
    fig2.savefig(out_dir / "metrics_mean.pdf")
    print("  > metrics_mean")
    plt.close(fig2)

def main():
    label_map = _load_label_map()
    vis.set_label_map(label_map)
    if label_map:
        print(f"Загружены подписи ({len(label_map)} ключей) из {LABELS_CONFIG.name}")
    else:
        print(f"Карта подписей пуста ({LABELS_CONFIG.name}); подписи без подмен.")

    print("\nЗагрузка данных из", RESULTS_ROOT)
    all_data = load_all_data()
    if not all_data:
        print("Нет данных! Проверьте наличие CSV в", RESULTS_ROOT)
        return

    print(f"\nПриоры ({len(all_data)}): {list(all_data.keys())}")
    print(f"Результаты сохраняются в: {OUT_DIR}\n")

    print("[1] Визуальная таблица для каждого приора (без заголовков столбцов)...")
    plot_per_prior_visual(all_data, OUT_DIR)

    print("\n[2] Кросс-приор сравнение — все оригиналы...")
    plot_cross_prior_comparison(all_data, OUT_DIR)

    print("\n[3] Кросс-приор сравнение — по каждому оригиналу отдельно...")
    plot_per_image_comparison(all_data, OUT_DIR)

    print("\n[4] Маленькие таблицы (1 ядро / 1 оригинал)...")
    plot_small_tables_per_kernel(all_data, OUT_DIR)

    print("\n[5] Сравнение ядер...")
    plot_kernels_comparison(all_data, OUT_DIR)

    print("\n[6] Метрики PSNR/SSIM...")
    plot_metrics_table(all_data, OUT_DIR)

    print("\n[OK] Готово. Результаты:", OUT_DIR)


if __name__ == "__main__":
    main()
