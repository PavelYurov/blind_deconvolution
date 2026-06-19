#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Генерирует таблицы и столбцатые диаграммы для алгоритмов шумоподавления.

Импульсный шум не учитывается.

Charts produced:
  1. chart2_estimator_comparison_{metric}.png
     -- средний PSNR / SSIM / ISNR на комбинацию денойзера/метода оценки шума.

  2. chart3_{estimator}_{noise_type}_{metric}.png
     -- сравнение алгоритмов шумоподавления при фиксированном методе оценки
     на разных интенсивностях шума.

  3. summary_table.csv  (средние метрики за комбинацию)

Использование:
    python generate_visuals_denoise.py
    python generate_visuals_denoise.py --estimator chen   # chart3 только для chen
    python generate_visuals_denoise.py --phase 2          # только chart2
    python generate_visuals_denoise.py --phase 3          # только chart3
"""

import sys
import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)


ALL_RESULTS_CSV = PROJECT_ROOT / "presentation_graphics_denoise" / "all_results.csv"
OUT_DIR         = PROJECT_ROOT / "presentation_graphics_denoise"


LEVELS_ORDER = ["weak", "medium", "strong"]

ESTIMATORS_ORDER = ["none", "pca", "chen"]


ESTIMATOR_LABELS = {"none": "No Est.", "pca": "PCA", "chen": "Chen"}


NOISE_LABELS = {
    "gaussian": "Gaussian",
    "poisson":  "Poisson",
    "pink":     "Pink",
    "brown":    "Brown",
}

METRICS = [
    ("psnr_out", "PSNR out, дб"),
    ("ssim_out", "SSIM out"),
    ("isnr",     "ISNR, дб"),
]


PALETTE = [
    '#2176AE', '#E05929', '#57A773', '#B5338A', '#F2C12E',
    '#1B998B', '#D64045', '#6B4226', '#3D5A80', '#EE6C4D',
]
TITLE_FONTSIZE = 16


def _bar_ymin(vals: list, metric: str) -> float:
    finite = [v for v in vals if v is not None and not math.isnan(v)]
    if not finite:
        return 0.0
    mn = min(finite)
    if "psnr" in metric:
        return max(0.0, math.floor(mn / 5) * 5 - 5)
    elif "ssim" in metric:
        return max(0.0, round(math.floor(mn * 10) / 10 - 0.1, 10))
    elif "isnr" in metric:
        return (math.floor(mn*100) - 30) / 100
    return 0.0


def _bar_ymax(vals: list, metric: str) -> float:
    finite = [v for v in vals if v is not None and not math.isnan(v)]
    if not finite:
        return 1.0
    mx = max(finite)
    if "psnr" in metric:
        return math.ceil(mx / 5) * 5 + 2
    elif "ssim" in metric:
        return min(1.02, round(math.ceil(mx * 10) / 10 + 0.05, 10))
    elif "isnr" in metric:
        return (math.ceil(mx*100) + 30) / 100
    return mx * 1.1


def _colormap_bars(n: int):
    try:
        cmap = plt.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = cm.get_cmap("tab20", max(n, 1))
    return [cmap(i / max(n, 1)) for i in range(n)]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(str(ALL_RESULTS_CSV))
    df = df[df["noise_type"] != "impulse"].copy()
    df = df.dropna(subset=["psnr_out", "ssim_out"]).copy()
    return df



def chart2_estimator_comparison(df: pd.DataFrame, out_dir: Path):
    denoisers = [d for d in df["denoiser"].unique()
                 if d not in ("vst+bm3d",)]
    denoisers = sorted(denoisers)
    has_vst = "vst+bm3d" in df["denoiser"].unique()

    estimators = [e for e in ESTIMATORS_ORDER if e in df["estimator"].unique()]
    all_methods = denoisers + (["vst+bm3d"] if has_vst else [])
    colors = _colormap_bars(len(all_methods))

    for col, ylabel in METRICS:
        data = {}
        for method in denoisers:
            row_vals = []
            for est in estimators:
                sub = df[(df["denoiser"] == method) & (df["estimator"] == est)]
                row_vals.append(sub[col].mean() if len(sub) > 0 else math.nan)
            data[method] = row_vals

        if has_vst:
            sub_vst = df[(df["denoiser"] == "vst+bm3d") & (df["estimator"] == "pca")]
            vst_vals = [math.nan] * len(estimators)
            if "pca" in estimators:
                vst_vals[estimators.index("pca")] = (
                    sub_vst[col].mean() if len(sub_vst) > 0 else math.nan
                )
            data["vst+bm3d"] = vst_vals

        n_groups = len(estimators)
        n_bars   = len(all_methods)
        bar_w    = 0.7 / n_bars
        x_base   = np.arange(n_groups)

        all_vals = [v for vals in data.values() for v in vals if not math.isnan(v)]
        ymin = _bar_ymin(all_vals, col)
        ymax = _bar_ymax(all_vals, col)

        fig, ax = plt.subplots(figsize=(max(n_groups * n_bars * 0.55 + 2, 8), 5))

        for i, method in enumerate(all_methods):
            vals  = data[method]
            xpos  = x_base + i * bar_w - (n_bars - 1) * bar_w / 2
            rects = ax.bar(xpos, vals, width=bar_w * 0.9,
                           color=colors[i], alpha=0.85,
                           edgecolor="grey", linewidth=0.3,
                           label=method)
            for rect, v in zip(rects, vals):
                if not math.isnan(v):
                    height = rect.get_height()
                    if height > ymin:
                        ax.text(
                            rect.get_x() + rect.get_width() / 2,
                            height + (ymax - ymin) * 0.012,
                            f"{v:.2f}" if "ssim" not in col else f"{v:.3f}",
                            ha="center", va="bottom", fontsize=10.5, rotation=0,
                        )

        ax.set_xticks(x_base)
        ax.set_xticklabels([ESTIMATOR_LABELS.get(e, e) for e in estimators],
                           fontsize=14)
        ax.set_xlabel("Оценщик уровня шума", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(
            f"Средний {ylabel.split(',')[0]} по методу шумоподавления и оценщику шума",
            fontsize=TITLE_FONTSIZE,
        )
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="lower right", fontsize=12, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        metric_tag = col.replace("_out", "").replace("_", "")
        fpath = out_dir / f"chart2_estimator_comparison_{metric_tag}.png"
        fig.savefig(str(fpath), dpi=150)
        fig.savefig(str(fpath.with_suffix(".pdf")))
        plt.close(fig)
        print(f"  Saved: {fpath.name}")


def chart3_algo_vs_noise_level(
    df: pd.DataFrame,
    out_dir: Path,
    estimator_filter: str | None = None,
):
    noise_types = [n for n in df["noise_type"].unique() if n != "impulse"]
    noise_types = sorted(noise_types)
    estimators  = [e for e in ESTIMATORS_ORDER if e in df["estimator"].unique()]

    if estimator_filter:
        estimators = [e for e in estimators if e == estimator_filter]

    levels = [l for l in LEVELS_ORDER if l in df["noise_level"].unique()]

    for est in estimators:
        df_est = df[df["estimator"] == est]

        for noise_t in noise_types:
            df_nt = df_est[df_est["noise_type"] == noise_t]
            if df_nt.empty:
                continue
            denoisers_here = sorted(df_nt["denoiser"].unique())

            colors = _colormap_bars(len(denoisers_here))

            for col, ylabel in METRICS:
                n_groups = len(levels)
                n_bars   = len(denoisers_here)
                bar_w    = 0.7 / n_bars
                x_base   = np.arange(n_groups)

                all_vals = []
                table = {}
                for method in denoisers_here:
                    row_v = []
                    for lvl in levels:
                        sub = df_nt[(df_nt["denoiser"] == method) &
                                    (df_nt["noise_level"] == lvl)]
                        v = sub[col].mean() if len(sub) > 0 else math.nan
                        row_v.append(v)
                        if not math.isnan(v):
                            all_vals.append(v)
                    table[method] = row_v

                if not all_vals:
                    continue

                ymin = _bar_ymin(all_vals, col)
                ymax = _bar_ymax(all_vals, col)

                fig, ax = plt.subplots(
                    figsize=(max(n_groups * n_bars * 0.6 + 2, 7), 5)
                )

                for i, method in enumerate(denoisers_here):
                    vals = table[method]
                    xpos = x_base + i * bar_w - (n_bars - 1) * bar_w / 2
                    rects = ax.bar(xpos, vals, width=bar_w * 0.9,
                                   color=colors[i], alpha=0.85,
                                   edgecolor="grey", linewidth=0.3,
                                   label=method)
                    for rect, v in zip(rects, vals):
                        if not math.isnan(v):
                            height = rect.get_height()
                            if height > ymin:
                                ax.text(
                                    rect.get_x() + rect.get_width() / 2,
                                    height + (ymax - ymin) * 0.012,
                                    f"{v:.2f}" if "ssim" not in col else f"{v:.3f}",
                                    ha="center", va="bottom", fontsize=6.5,
                                )

                ax.set_xticks(x_base)
                ax.set_xticklabels([l.capitalize() for l in levels], fontsize=10)
                ax.set_xlabel("Noise level", fontsize=10)
                ax.set_ylabel(ylabel, fontsize=10)
                noise_label = NOISE_LABELS.get(noise_t, noise_t.capitalize())
                ax.set_title(
                    f"Mean {ylabel.split(',')[0]} -- {noise_label} noise"
                    f"\nEstimator: {ESTIMATOR_LABELS.get(est, est)}",
                    fontsize=TITLE_FONTSIZE,
                )
                ax.set_ylim(ymin, ymax)
                ax.legend(loc="lower right", fontsize=8,
                          ncol=max(1, n_bars // 4))
                ax.grid(axis="y", alpha=0.3)
                plt.tight_layout()

                metric_tag = col.replace("_out", "").replace("_", "")
                fname = f"chart3_{est}_{noise_t}_{metric_tag}.png"
                fpath = out_dir / fname
                fig.savefig(str(fpath), dpi=150)
                fig.savefig(str(fpath.with_suffix(".pdf")))
                plt.close(fig)
                print(f"  Saved: {fname}")


def make_summary_table(df: pd.DataFrame, out_dir: Path):
    """Mean metrics per (denoiser, estimator) across all non-impulse noise."""
    grp = df.groupby(["denoiser", "estimator"]).agg(
        psnr_in_mean  = ("psnr_in",  "mean"),
        psnr_out_mean = ("psnr_out", "mean"),
        ssim_out_mean = ("ssim_out", "mean"),
        isnr_mean     = ("isnr",     "mean"),
        n             = ("psnr_out", "count"),
    ).reset_index()
    grp = grp.sort_values(["denoiser", "estimator"])
    csv_path = out_dir / "summary_table.csv"
    grp.to_csv(str(csv_path), index=False, float_format="%.4f")
    print(f"  Saved: {csv_path.name}")

    print("\n  Summary (mean metrics, no impulse):")
    print(f"  {'Denoiser':<12}  {'Estimator':<8}  "
          f"{'PSNR_in':>8}  {'PSNR_out':>8}  {'SSIM_out':>8}  {'ISNR':>7}  {'n':>4}")
    print("  " + "-" * 65)
    for _, row in grp.iterrows():
        print(f"  {row['denoiser']:<12}  {row['estimator']:<8}  "
              f"{row['psnr_in_mean']:8.2f}  {row['psnr_out_mean']:8.2f}  "
              f"{row['ssim_out_mean']:8.4f}  {row['isnr_mean']:7.2f}  "
              f"{int(row['n']):4d}")



def main():
    parser = argparse.ArgumentParser(description="Denoise visuals generator")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 2, 3],
                        help="0=all, 2=chart2 only, 3=chart3 only")
    parser.add_argument("--estimator", type=str, default=None,
                        help="Filter chart3 to one estimator: none/pca/chen")
    args = parser.parse_args()

    if not ALL_RESULTS_CSV.exists():
        print(f"ERROR: {ALL_RESULTS_CSV} not found.")
        print("Run run_all_dataset_denoise.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {ALL_RESULTS_CSV.name} ...")
    df = load_data()
    print(f"  Rows after excluding impulse noise: {len(df)}")
    print(f"  Noise types: {sorted(df['noise_type'].unique())}")

    run_all = (args.phase == 0)

    make_summary_table(df, OUT_DIR)

    if run_all or args.phase == 2:
        print("\n[Chart 2] Estimator comparison...")
        chart2_estimator_comparison(df, OUT_DIR)

    if run_all or args.phase == 3:
        print("\n[Chart 3] Algorithm vs noise level (per estimator, per noise type)...")
        chart3_algo_vs_noise_level(df, OUT_DIR, estimator_filter=args.estimator)

    print("\n[DONE] All figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
