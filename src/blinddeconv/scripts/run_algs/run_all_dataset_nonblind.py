#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сравнение non-blind шагов деконволюции на датасете Grid_Test.
Запускается только на чистых изображениях (noise_type == "clean") с
истинными ядрами из ground_truth_filters/.


Результаты:
  presentation_graphics_nonblind/
    {method}/
        restored/                  — PNG, uint8
        results_{method}.csv
    all_results.csv                — сводная таблица

Метрики:
  psnr_blurred — PSNR(размытое, оригинал)
  ssim_blurred — SSIM(размытое, оригинал)
  psnr_out     — PSNR(восстановленное, оригинал)
  ssim_out     — SSIM(восстановленное, оригинал)
  isnr         — psnr_out - psnr_blurred

Запуск:
  python run_all_dataset_nonblind.py                  # все методы
  python run_all_dataset_nonblind.py --method firls   # конкретный метод
  python run_all_dataset_nonblind.py --list           # список методов
  python run_all_dataset_nonblind.py --workers 2      # потоки
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent

# КОНФИГУРАЦИЯ

DATASET_DIR   = PROJECT_ROOT / "images" / "compare_data" / "anton" / "Grid_Test"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"
KERNELS_DIR   = DATASET_DIR / "ground_truth_filters"
DESIGN_JSON   = DATASET_DIR / "dataset_design.json"

RESULTS_ROOT = PROJECT_ROOT / "presentation_graphics_nonblind"

NUM_WORKERS = 6

# Сравниваемые методы
METHODS = [
    "firls",
    "irls",
    "tikhonov",
    "ringing_removal",
    "adaptive_lp",
]

_FIRLS_ALPHA  = 2.0 / 3.0
_FIRLS_LAMBDA = 2e-4
_FIRLS_BETA_A = _FIRLS_LAMBDA * _FIRLS_ALPHA * (20.0 / 255.0) ** (_FIRLS_ALPHA - 2.0)

FIRLS_OPTS = {
    "out_iter":    5,
    "inner_iter":  4,
    "IF":          float(np.sqrt(2.0)),
    "lambda":      _FIRLS_LAMBDA,
    "lambda_u":    0.1,
    "epsilon_min": 2.55 / 255.0,
    "epsilon_max": 2.55 / 255.0,
    "alpha":       _FIRLS_ALPHA,
    "beta_a":      _FIRLS_BETA_A,
}

IRLS_BETA       = 1.0
IRLS_LAMBDA_REG = 2e-4

TIKHONOV_ALPHA = 0.001

RINGING_LAMBDA_TV = 1e-3
RINGING_LAMBDA_L0 = 2e-3
RINGING_WEIGHT    = 1.0

ADAPTIVE_LP_ALPHA = 0.8


def load_design() -> dict:
    """Загружает dataset_design.json - {filename: record}."""
    if not DESIGN_JSON.exists():
        return {}
    with open(str(DESIGN_JSON), encoding="utf-8") as f:
        records = json.load(f)
    return {r["filename"]: r for r in records}



def _worker_init(project_root: str):
    root = project_root
    src  = os.path.join(root, "src")
    for p in (src, root):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)


def _make_odd_kernel(h: np.ndarray) -> np.ndarray:
    m1, m2 = h.shape
    if m1 % 2 == 0:
        h = np.pad(h, ((0, 1), (0, 0)))
    if m2 % 2 == 0:
        h = np.pad(h, ((0, 0), (0, 1)))
    return h


def _run_solver(method: str, y: np.ndarray, h: np.ndarray,
                firls_opts: dict, irls_beta: float, irls_lambda: float,
                tikhonov_alpha: float = 0.001,
                ringing_lambda_tv: float = 1e-3,
                ringing_lambda_l0: float = 2e-3,
                ringing_weight: float = 1.0,
                adaptive_lp_alpha: float = 0.8,
                ) -> np.ndarray:
    """
    Вызывает нужный non-blind решатель.

    Параметры:
    y : float64 [0, 1], форма (H, W)
    h : float64 нормализованное ядро, сумма = 1

    Возвращает:
    x : float64 [0, 1], форма (H, W)
    """
    if method == "firls":
        from blinddeconv.algorithms.blind_deconvolution.our_company.\
            bayesian.fbdhsgp.fbdhsgp.solvers import frils_deb_ubc
        return frils_deb_ubc(y, _make_odd_kernel(h), firls_opts)

    if method == "irls":
        from blinddeconv.algorithms.blind_deconvolution.our_company.\
            bayesian.bid_hbsp_denoise.solvers import final_deconvolution
        return final_deconvolution(y, h, irls_beta, irls_lambda)

    if method == "tikhonov":
        from blinddeconv.algorithms.blind_deconvolution.our_company.\
            logarithmic_pds.lip_denoise.utils import (
                pad_image, edgetaper, tikhonov_filter, crop_image)
        MK, NK = h.shape
        M_orig, N_orig = y.shape
        y_pad = pad_image(y, (MK, NK))
        y_pad = edgetaper(y_pad, h)
        x_out = tikhonov_filter(y_pad, h, alpha=tikhonov_alpha)
        return crop_image(x_out, (M_orig, N_orig), (MK, NK))

    if method == "ringing_removal":
        from blinddeconv.algorithms.blind_deconvolution.our_company.\
            logarithmic_pds.lip_denoise.solvers import ringing_artifacts_removal
        return ringing_artifacts_removal(
            y, h,
            lambda_tv=ringing_lambda_tv,
            lambda_l0=ringing_lambda_l0,
            weight_ring=ringing_weight,
        )

    if method == "adaptive_lp":
        from blinddeconv.algorithms.blind_deconvolution.our_company.\
            logarithmic_pds.lip_denoise.non_blind import adaptive_lp_deconv
        return adaptive_lp_deconv(y, h, alpha=adaptive_lp_alpha)

    raise ValueError(f"Unknown method: {method!r}")


def _process_task(task: dict) -> dict:
    """Обрабатывает одну (метод х изображение) задачу в дочернем процессе."""
    import math
    import time as _time

    import cv2
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    method        = task["method"]
    dist_path     = task["dist_path"]
    orig_path     = task["orig_path"]
    kernel_path   = task["kernel_path"]
    restored_path = task["restored_path"]
    fname         = task["fname"]
    original_name = task["original_name"]
    kernel_name   = task["kernel_name"]
    psnr_blurred  = task["psnr_blurred"]
    ssim_blurred  = task["ssim_blurred"]
    firls_opts         = task["firls_opts"]
    irls_beta          = task["irls_beta"]
    irls_lambda        = task["irls_lambda"]
    tikhonov_alpha     = task["tikhonov_alpha"]
    ringing_lambda_tv  = task["ringing_lambda_tv"]
    ringing_lambda_l0  = task["ringing_lambda_l0"]
    ringing_weight     = task["ringing_weight"]
    adaptive_lp_alpha  = task["adaptive_lp_alpha"]

    base_row = {
        "filename":     fname,
        "original":     original_name,
        "kernel":       kernel_name,
        "method":       method,
        "psnr_blurred": round(psnr_blurred, 4),
        "ssim_blurred": round(ssim_blurred, 4),
        "psnr_out":     None,
        "ssim_out":     None,
        "isnr":         None,
        "time_sec":     0.0,
    }

    blurred    = cv2.imread(dist_path,   cv2.IMREAD_GRAYSCALE)
    original   = cv2.imread(orig_path,   cv2.IMREAD_GRAYSCALE)
    kernel_img = cv2.imread(kernel_path, cv2.IMREAD_GRAYSCALE)

    if blurred is None or original is None or kernel_img is None:
        base_row["error"] = "imread failed"
        return base_row

    y = blurred.astype(np.float64) / 255.0

    h = kernel_img.astype(np.float64)
    h = np.rot90(h, 2)
    h_sum = h.sum()
    if h_sum > 0:
        h /= h_sum
    else:
        base_row["error"] = "kernel sums to zero"
        return base_row

    t0 = _time.time()
    try:
        x_out = _run_solver(
            method, y, h,
            firls_opts, irls_beta, irls_lambda,
            tikhonov_alpha,
            ringing_lambda_tv, ringing_lambda_l0, ringing_weight,
            adaptive_lp_alpha,
        )
    except Exception as exc:
        base_row["time_sec"] = round(_time.time() - t0, 3)
        base_row["error"] = str(exc)
        return base_row
    elapsed = _time.time() - t0

    x_clipped = np.clip(x_out, 0.0, 1.0)
    x_u8 = (x_clipped * 255.0).astype(np.uint8)
    cv2.imwrite(restored_path, x_u8)

    orig_f = original.astype(np.float64) / 255.0
    try:
        psnr_out = float(peak_signal_noise_ratio(orig_f, x_clipped, data_range=1.0))
    except Exception:
        psnr_out = math.nan
    try:
        ssim_out = float(structural_similarity(orig_f, x_clipped, data_range=1.0))
    except Exception:
        ssim_out = math.nan

    isnr = (
        psnr_out - psnr_blurred
        if not (math.isnan(psnr_out) or math.isnan(psnr_blurred))
        else math.nan
    )

    return {
        "filename":     fname,
        "original":     original_name,
        "kernel":       kernel_name,
        "method":       method,
        "psnr_blurred": round(psnr_blurred, 4),
        "ssim_blurred": round(ssim_blurred, 4),
        "psnr_out":     round(psnr_out, 4),
        "ssim_out":     round(ssim_out, 4),
        "isnr":         round(isnr, 4),
        "time_sec":     round(elapsed, 3),
    }


def build_tasks(methods: list, design: dict) -> list:
    """
    Строит список задач: (метод х чистое изображение).

    Фильтр: только файлы вида *_clean.png (noise_type == "clean").
    Ядро: ground_truth_filters/{kernel_name}.png
    """
    clean_files = sorted(DISTORTED_DIR.glob("*_clean.png"))
    tasks = []

    for method in methods:
        restored_dir = RESULTS_ROOT / method / "restored"
        restored_dir.mkdir(parents=True, exist_ok=True)

        for dist_path in clean_files:
            fname = dist_path.name
            info  = design.get(fname, {})

            stem_parts    = dist_path.stem.split("_")
            original_name = info.get("original", stem_parts[0] if stem_parts else "")
            kernel_name   = info.get("kernel",   stem_parts[1] if len(stem_parts) > 1 else "")

            orig_path   = ORIGINALS_DIR / f"{original_name}.png"
            kernel_path = KERNELS_DIR   / f"{kernel_name}.png"

            if not orig_path.exists():
                for ext in [".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
                    cand = ORIGINALS_DIR / f"{original_name}{ext}"
                    if cand.exists():
                        orig_path = cand
                        break

            if not orig_path.exists():
                print(f"  [WARN] original not found: {orig_path}")
                continue
            if not kernel_path.exists():
                print(f"  [WARN] kernel not found: {kernel_path}")
                continue

            psnr_blurred = float(info.get("blur_psnr",
                              info.get("final_psnr", math.nan)))
            ssim_blurred = float(info.get("blur_ssim",
                              info.get("final_ssim", math.nan)))

            tasks.append({
                "method":        method,
                "dist_path":     str(dist_path),
                "orig_path":     str(orig_path),
                "kernel_path":   str(kernel_path),
                "restored_path": str(restored_dir / fname),
                "fname":         fname,
                "original_name": original_name,
                "kernel_name":   kernel_name,
                "psnr_blurred":  psnr_blurred,
                "ssim_blurred":  ssim_blurred,
                "firls_opts":          FIRLS_OPTS,
                "irls_beta":           IRLS_BETA,
                "irls_lambda":         IRLS_LAMBDA_REG,
                "tikhonov_alpha":      TIKHONOV_ALPHA,
                "ringing_lambda_tv":   RINGING_LAMBDA_TV,
                "ringing_lambda_l0":   RINGING_LAMBDA_L0,
                "ringing_weight":      RINGING_WEIGHT,
                "adaptive_lp_alpha":   ADAPTIVE_LP_ALPHA,
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Non-blind deconvolution comparison"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Run only one method (e.g. firls)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available methods and exit"
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"Number of parallel worker processes (default: {NUM_WORKERS})"
    )
    args = parser.parse_args()

    if args.list:
        print("Available methods:")
        for m in METHODS:
            print(f"  {m}")
        return

    methods = list(METHODS)
    if args.method:
        if args.method not in METHODS:
            print(f"ERROR: unknown method {args.method!r}. Available: {METHODS}")
            sys.exit(1)
        methods = [args.method]

    if not DISTORTED_DIR.exists():
        print(f"ERROR: distorted dir not found: {DISTORTED_DIR}")
        sys.exit(1)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    design = load_design()
    tasks  = build_tasks(methods, design)
    n_total = len(tasks)

    if n_total == 0:
        print("No tasks found. Check that *_clean.png files exist in distorted/")
        sys.exit(1)

    print("=" * 70)
    print(f"  Dataset   : {DATASET_DIR.name}  (clean images only)")
    print(f"  Methods   : {', '.join(methods)}")
    print(f"  Tasks     : {n_total}  ({n_total // len(methods)} images x {len(methods)} methods)")
    print(f"  Workers   : {args.workers}")
    print(f"  Results   : {RESULTS_ROOT}")
    print("=" * 70)

    t_all    = time.time()
    all_rows = []
    n_done   = 0
    n_err    = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(PROJECT_ROOT),),
    ) as pool:
        futures = {pool.submit(_process_task, t): t for t in tasks}

        for fut in as_completed(futures):
            n_done += 1
            try:
                row = fut.result()
            except Exception as exc:
                n_err += 1
                task = futures[fut]
                print(f"  [ERROR] {task['fname']} / {task['method']}: {exc}")
                continue

            if "error" in row:
                n_err += 1
                print(f"  [SKIP]  {row['filename']} / {row['method']}: {row['error']}")
            else:
                isnr_str = f"{row['isnr']:+6.2f}" if not math.isnan(row['isnr']) else "   nan"
                print(
                    f"  [{n_done:4d}/{n_total}]"
                    f"  {row['filename']:<40s}"
                    f"  {row['method']:<26s}"
                    f"  PSNR {row['psnr_blurred']:5.2f} -> {row['psnr_out']:5.2f}"
                    f"  ({isnr_str})"
                    f"  {row['time_sec']:.1f}s"
                )

            all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    if not df_all.empty and "method" in df_all.columns:
        for method, grp in df_all.groupby("method"):
            csv_path = RESULTS_ROOT / method / f"results_{method}.csv"
            grp.to_csv(csv_path, index=False)

    all_csv = RESULTS_ROOT / "all_results.csv"
    df_all.to_csv(all_csv, index=False)

    elapsed = time.time() - t_all
    mins    = int(elapsed // 60)
    secs    = int(elapsed  % 60)

    print(f"\n  Global CSV: {all_csv}")
    print(f"\n  {'Method':<28s}  {'PSNR_blur':>9s}  {'PSNR_out':>8s}"
          f"  {'ISNR':>7s}  {'SSIM_out':>8s}  {'Time_total':>10s}")
    print("  " + "-" * 82)
    if not df_all.empty and "method" in df_all.columns:
        for method, grp in df_all.groupby("method"):
            valid = grp.dropna(subset=["psnr_out"])
            if valid.empty:
                print(f"  {method:<28s}  (no valid results)")
                continue
            print(
                f"  {method:<28s}"
                f"  {valid['psnr_blurred'].mean():9.2f}"
                f"  {valid['psnr_out'].mean():8.2f}"
                f"  {valid['isnr'].mean():+7.2f}"
                f"  {valid['ssim_out'].mean():8.4f}"
                f"  {valid['time_sec'].sum():9.1f}s"
            )

    print(f"\n{'=' * 70}")
    print(f"  [DONE]  {n_done} tasks, {n_err} errors, time: {mins}m {secs}s")
    print(f"  Results : {RESULTS_ROOT}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
