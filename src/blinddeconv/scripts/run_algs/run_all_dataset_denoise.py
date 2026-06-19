#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Подсчет всех комбинаций денойзер x оценщик шума на датасете
images/compare_data/noise/.

Структура комбинаций:
  - 6 денойзеров x 3 оценщика шума = 18 комбинаций
      bm3d, guided, bilateral, nlm, tv, act  x  chen, pca, none
  - vst+bm3d x pca -- только на изображениях с пуассоновским шумом

Результаты:
  presentation_graphics_denoise/
    {denoiser}_{estimator}/          <- папка комбинации
        results_{denoiser}_{estimator}.csv
        restored/                    <- денойзированные изображения
    all_results.csv                  <- сводная таблица

Метрики в CSV:
  psnr_in   -- PSNR(шумное, оригинал)
  ssim_in   -- SSIM(шумное, оригинал)
  psnr_out  -- PSNR(денойзированное, оригинал)
  ssim_out  -- SSIM(денойзированное, оригинал)
  isnr      -- psnr_out - psnr_in  (Improvement SNR)

Запуск:
  python run_all_dataset_denoise.py              # все комбинации
  python run_all_dataset_denoise.py --combo bm3d_chen   # одна комбинация
  python run_all_dataset_denoise.py --list               # напечатать все комбинации
  python run_all_dataset_denoise.py --workers 4         # задать число ядер
"""

import sys
import os
import json
import time
import math
import argparse
import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent

# КОНФИГУРАЦИЯ

DATASET_DIR   = PROJECT_ROOT / "images" / "compare_data" / "noise"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"
DESIGN_JSON   = DATASET_DIR / "dataset_design.json"

RESULTS_ROOT  = PROJECT_ROOT / "presentation_graphics_denoise"

NUM_WORKERS = 6

# Денойзеры
BASE_DENOISERS = ["bm3d", "guided", "bilateral", "nlm", "tv", "act", "median"]

# Оценщики шума
ESTIMATORS = ["chen", "pca", "none"]

# median применяется только через estimator='none'
MEDIAN_ESTIMATORS = ["none"]

# Шум-тип для vst+bm3d
VST_NOISE_TYPE = "poisson"

# ПОСТРОЕНИЕ СПИСКА КОМБИНАЦИЙ
def build_combos():
    """Возвращает список кортежей (denoiser, estimator, noise_filter)."""
    combos = []
    for d in BASE_DENOISERS:
        ests = MEDIAN_ESTIMATORS if d == "median" else ESTIMATORS
        for e in ests:
            combos.append((d, e, None))
    combos.append(("vst+bm3d", "pca", VST_NOISE_TYPE))
    return combos


def combo_label(denoiser: str, estimator: str) -> str:
    return f"{denoiser.replace('+', '_')}_{estimator}"


def load_design() -> dict:
    if not DESIGN_JSON.exists():
        return {}
    with open(str(DESIGN_JSON), encoding="utf-8") as f:
        records = json.load(f)
    return {r["filename"]: r for r in records}



def _worker_init(project_root: str):
    import sys, os
    root = project_root
    src  = os.path.join(root, "src")
    for p in (src, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)


def _process_task(task: dict) -> dict:
    import time, math, cv2, numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from blinddeconv.algorithms.denoise.Denoise import DenoiseWrapper

    denoiser      = task["denoiser"]
    estimator     = task["estimator"]
    dist_path     = task["dist_path"]
    orig_path     = task["orig_path"]
    restored_path = task["restored_path"]
    fname         = task["fname"]
    original_name = task["original_name"]
    noise_type    = task["noise_type"]
    noise_level   = task["noise_level"]
    psnr_in       = task["psnr_in"]
    ssim_in       = task["ssim_in"]

    original = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    noisy    = cv2.imread(dist_path, cv2.IMREAD_GRAYSCALE)

    base_row = {
        "filename":    fname,
        "original":    original_name,
        "noise_type":  noise_type,
        "noise_level": noise_level,
        "denoiser":    denoiser,
        "estimator":   estimator,
        "psnr_in":     round(psnr_in, 4),
        "ssim_in":     round(ssim_in, 4),
        "psnr_out":    None,
        "ssim_out":    None,
        "isnr":        None,
        "time_sec":    0.0,
    }

    if original is None or noisy is None:
        base_row["error"] = "imread failed"
        return base_row

    alg = DenoiseWrapper(method=denoiser, noise_estimation=estimator, verbose=False)

    t0 = time.time()
    try:
        restored_raw, _ = alg.process(noisy)
    except Exception as e:
        base_row["time_sec"] = round(time.time() - t0, 3)
        base_row["error"] = str(e)
        return base_row
    elapsed = time.time() - t0

    restored_u8 = np.clip(restored_raw, 0, 255).astype(np.uint8)

    orig_f = original.astype(np.float64) / 255.0
    img_f  = np.clip(restored_u8.astype(np.float64) / 255.0, 0.0, 1.0)
    try:
        psnr_out = float(peak_signal_noise_ratio(orig_f, img_f, data_range=1.0))
    except Exception:
        psnr_out = math.nan
    try:
        ssim_out = float(structural_similarity(orig_f, img_f, data_range=1.0))
    except Exception:
        ssim_out = math.nan

    isnr = (psnr_out - psnr_in
            if not math.isnan(psnr_out) and not math.isnan(psnr_in)
            else math.nan)

    cv2.imwrite(restored_path, restored_u8)

    return {
        "filename":    fname,
        "original":    original_name,
        "noise_type":  noise_type,
        "noise_level": noise_level,
        "denoiser":    denoiser,
        "estimator":   estimator,
        "psnr_in":     round(psnr_in, 4),
        "ssim_in":     round(ssim_in, 4),
        "psnr_out":    round(psnr_out, 4),
        "ssim_out":    round(ssim_out, 4),
        "isnr":        round(isnr, 4),
        "time_sec":    round(elapsed, 3),
    }


def build_tasks(combos: list, design: dict) -> list[dict]:
    """Лист всех комбинаций задач"""
    dist_files = sorted(DISTORTED_DIR.glob("*.png"))
    tasks = []

    for denoiser, estimator, noise_filter in combos:
        label        = combo_label(denoiser, estimator)
        restored_dir = RESULTS_ROOT / label / "restored"
        restored_dir.mkdir(parents=True, exist_ok=True)

        for dist_path in dist_files:
            fname = dist_path.name
            info  = design.get(fname, {})
            original_name = info.get("original", dist_path.stem.split("_")[0])
            noise_type    = info.get("noise_type", "")
            noise_level   = info.get("noise_level", "")

            if noise_filter is not None and noise_type != noise_filter:
                continue

            orig_path = ORIGINALS_DIR / f"{original_name}.png"
            if not orig_path.exists():
                continue

            psnr_in = float(info["psnr"]) if "psnr" in info else None
            ssim_in = float(info["ssim"]) if "ssim" in info else None

            if psnr_in is None:
                original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
                noisy    = cv.imread(str(dist_path), cv.IMREAD_GRAYSCALE)
                if original is not None and noisy is not None:
                    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                    orig_f = original.astype(np.float64) / 255.0
                    noisy_f = np.clip(noisy.astype(np.float64) / 255.0, 0.0, 1.0)
                    try:
                        psnr_in = float(peak_signal_noise_ratio(orig_f, noisy_f, data_range=1.0))
                        ssim_in = float(structural_similarity(orig_f, noisy_f, data_range=1.0))
                    except Exception:
                        psnr_in = math.nan
                        ssim_in = math.nan
                else:
                    psnr_in = math.nan
                    ssim_in = math.nan

            out_name = f"{dist_path.stem}_{label}.png"
            tasks.append({
                "denoiser":      denoiser,
                "estimator":     estimator,
                "dist_path":     str(dist_path),
                "orig_path":     str(orig_path),
                "restored_path": str(restored_dir / out_name),
                "fname":         fname,
                "original_name": original_name,
                "noise_type":    noise_type,
                "noise_level":   noise_level,
                "psnr_in":       psnr_in,
                "ssim_in":       ssim_in,
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Denoise dataset runner")
    parser.add_argument("--combo", type=str, default=None,
                        help="Run only one combo, e.g. bm3d_chen")
    parser.add_argument("--list", action="store_true",
                        help="Print all available combos and exit")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Number of parallel workers (default: {NUM_WORKERS})")
    args = parser.parse_args()

    combos = build_combos()

    if args.list:
        print("Available combos:")
        for d, e, nf in combos:
            lbl  = combo_label(d, e)
            note = f" (only {nf})" if nf else ""
            print(f"  {lbl}{note}")
        return

    if args.combo:
        combos = [(d, e, nf) for d, e, nf in combos
                  if combo_label(d, e) == args.combo]
        if not combos:
            print(f"Combo '{args.combo}' not found. Use --list.")
            return

    if not DISTORTED_DIR.exists():
        print(f"ERROR: distorted dir not found: {DISTORTED_DIR}")
        return

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    design = load_design()

    tasks = build_tasks(combos, design)
    n_total = len(tasks)

    print("=" * 70)
    print(f"  Dataset   : {DATASET_DIR.name}")
    print(f"  Combos    : {len(combos)}")
    print(f"  Tasks     : {n_total}")
    print(f"  Workers   : {args.workers}")
    print(f"  Results   : {RESULTS_ROOT}")
    print("=" * 70)

    t_all = time.time()
    all_rows = []
    n_done = 0
    n_err  = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(PROJECT_ROOT),),
    ) as pool:
        futures = {pool.submit(_process_task, t): t for t in tasks}

        for fut in as_completed(futures):
            task = futures[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {
                    "filename":    task["fname"],
                    "original":    task["original_name"],
                    "noise_type":  task["noise_type"],
                    "noise_level": task["noise_level"],
                    "denoiser":    task["denoiser"],
                    "estimator":   task["estimator"],
                    "psnr_in":     task["psnr_in"],
                    "ssim_in":     task["ssim_in"],
                    "psnr_out":    None,
                    "ssim_out":    None,
                    "isnr":        None,
                    "time_sec":    0.0,
                    "error":       str(e),
                }
                n_err += 1

            all_rows.append(row)
            n_done += 1

            err_tag = f"  ERROR: {row.get('error', '')}" if row.get("error") else ""
            psnr_out = row.get("psnr_out")
            isnr     = row.get("isnr")
            psnr_in  = row.get("psnr_in", math.nan)
            lbl      = combo_label(row["denoiser"], row["estimator"])

            if psnr_out is not None:
                print(f"  [{n_done:4d}/{n_total}] {lbl:30s} | "
                      f"{row['filename']:40s} | "
                      f"{psnr_in:5.2f}->{psnr_out:5.2f} dB  ISNR={isnr:+5.2f}")
            else:
                print(f"  [{n_done:4d}/{n_total}] {lbl:30s} | "
                      f"{row['filename']:40s} | FAILED{err_tag}")

    df_all = pd.DataFrame(all_rows)
    for (d, e), grp in df_all.groupby(["denoiser", "estimator"]):
        lbl      = combo_label(d, e)
        csv_path = RESULTS_ROOT / lbl / f"results_{lbl}.csv"
        grp.to_csv(csv_path, index=False)

    all_csv = RESULTS_ROOT / "all_results.csv"
    df_all.to_csv(all_csv, index=False)

    elapsed = time.time() - t_all
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)

    print(f"\n  Global CSV: {all_csv}")
    print(f"\n  {'Combo':<35s}  {'PSNR_in':>8s}  {'PSNR_out':>8s}  "
          f"{'ISNR':>7s}  {'SSIM_out':>8s}")
    print("  " + "-" * 74)
    for (d, e), grp in df_all.groupby(["denoiser", "estimator"]):
        valid = grp.dropna(subset=["psnr_out"])
        if valid.empty:
            continue
        lbl = combo_label(d, e)
        print(f"  {lbl:<35s}  "
              f"{valid['psnr_in'].mean():8.2f}  "
              f"{valid['psnr_out'].mean():8.2f}  "
              f"{valid['isnr'].mean():+7.2f}  "
              f"{valid['ssim_out'].mean():8.4f}")

    print(f"\n{'=' * 70}")
    print(f"  [DONE] {n_done} tasks, {n_err} errors, time: {mins}m {secs}s")
    print(f"  Results: {RESULTS_ROOT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
