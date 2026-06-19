#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Подсчет алгоритма на датасете middle_data_pictures для
сравнения "стандартных" параметров (из статьи + модификации) и параметров,
выбранных как точки Парето-фронта (после TPE-оптимизации).

Для каждой строки каждой Парето-таблицы выполняются ТРИ подсчета:
  1) pareto    — с параметрами из строки (kernel_size, lambda_*, theta, и т.д.);
  2) standard  — со стандартными параметрами;
  3) best_hpo  — с параметрами лучшей итерации TPE (min Objective из
                 history_0.csv — задача минимизации, loss = -SSIM).

Результаты сохраняются ОТДЕЛЬНО для каждой из 4 таблиц
(pareto_PSNR_front, pareto_PSNR_ND_front, pareto_SSIM_front,
pareto_SSIM_ND_front) — таблицы между собой не смешиваются.

если файл восстановленного изображения уже существует и
запись есть в results_<mode>.csv, подсчет пропускается. 
Флаг --force отключает пропуск и пересчитывает всё.

Структура вывода:
  presentation_graphics_pareto_comp/
    <ALG_LABEL>/
      <table_stem>/
        pareto/restored/<idx>_<test_name>.png
        pareto/kernels/<idx>_<test_name>_kernel.png
        standard/restored/...   standard/kernels/...
        best_hpo/restored/...   best_hpo/kernels/...
        results_pareto.csv
        results_standard.csv
        results_best_hpo.csv
        results_combined.csv
"""
from __future__ import annotations

import sys
import os
import time
import math
import argparse
import importlib
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import cv2 as cv

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)


#ГЛОБАЛЬНЫЕ НАСТРОЙКИ
NUM_WORKERS = 6
ALIGNED=True

#Датасет для сравнения
DATASET_DIR  = PROJECT_ROOT / "images" / "middle_data_pictures"
DISTORTED_DIR = DATASET_DIR / "distorted"
ORIGINALS_DIR = DATASET_DIR / "originals"
KERNELS_DIR   = DATASET_DIR / "ground_truth_filters"

#Корневая директория для всех результатов сравнения
OUTPUT_ROOT = PROJECT_ROOT / "presentation_graphics_pareto_comp"

# Управление kernel_size:
# - int  - подсчет всегда с этим kernel_size;
# - None - подсчет берёт kernel_size из строки таблицы.      
KERNEL_SIZE_FIXED: int | None = 51

# Список таблиц парето-фронта, которые надо обработать.
PARETO_TABLE_FILES = [
    "pareto_PSNR_front.csv",
    "pareto_PSNR_ND_front.csv",
    "pareto_SSIM_front.csv",
    "pareto_SSIM_ND_front.csv",
]

#   КОНФИГУРАЦИЯ АЛГОРИТМОВ

# Каждый словарь описывает один алгоритм. Поля:
#   label            — имя для папок/CSV
#   alg_module       — путь к модулю
#   alg_class        — имя класса
#   kernel_param     — имя параметра, отвечающего за размер ядра
#   kernel_is_tuple  — True, если параметр ожидает (k,k) вместо int
#   standard_kwargs  — параметры из статьи + модификации
#   pareto_tables_dir— папка с парето-CSV (содержит PARETO_TABLE_FILES)
#   int_keys         — параметры, которые надо округлять до int при чтении
#                      из CSV
#   odd_keys         — параметры, которые должны быть нечётными (как
#                      kernel_size)
#   tunable_keys     — параметры, которые ИЩЕТ оптимизатор (берутся из
#                      строки CSV для pareto-подсчета; для standard
#                      берутся из standard_kwargs либо из defaults)
ALGORITHM_CONFIGS: list[dict[str, Any]] = [
    {
        "label":            "Enhanced_Sparse_Model",
        "alg_module":       ("src.blinddeconv.algorithms.blind_deconvolution."
                             "our_company.esm_cython._build_pyd.esm"),
        "alg_class":        "ESM_BD",
        "kernel_param":     "kernel_size",
        "kernel_is_tuple":  False,
        "standard_kwargs": {
            #статья + модификации
            "kernel_size":        51,
            "impulse_preprocess": "auto",
            "auto_mode":          "robust",
            "verbose":            False,
            # gamma_correct=1.0, final_deconv='ringing_removal'
            # lambda_data=4e-3, lambda_grad=4e-3, theta=1.0,
            # xk_iter=5, k_thresh=20, lambda_tv=2e-3, lambda_l0=2e-4,
            # weight_ring=1.0
        },
        "pareto_tables_dir":
            PROJECT_ROOT / "presentation_graphics_pareto"
            / "pareto_15_esm_tpe_150iter_50dist",
        "int_keys":     ["kernel_size", "xk_iter"],
        "odd_keys":     ["kernel_size"],
        "tunable_keys": ["kernel_size", "lambda_data", "lambda_grad",
                         "theta", "xk_iter", "k_thresh",
                         "lambda_tv", "lambda_l0", "weight_ring"],
        # Фиксированные параметры HPO:
        "pareto_fixed_kwargs": {
            "gamma_correct":      1.0,
            "final_deconv":       "ringing_removal",
            "impulse_preprocess": "auto",
            "auto_mode":          "robust",
            "verbose":            False,
        },
        # Файл истории TPE-оптимизации (Iteration, params..., Objective).
        # Строка с минимальным Objective берётся как best_hpo-подсчет.
        "history_path": (
            PROJECT_ROOT / "presentation_graphics_pareto"
            / "pareto_15_esm_tpe_150iter_50dist"
            / "logs" / "TPE" / "20260429_113330" / "history_0.csv"
        ),
    },
]

def _build_dataset_index() -> tuple[list[str], list[str]]:
    """Список названий оригиналов и имён ядер (как в HPO)."""
    originals = sorted(p.stem for p in ORIGINALS_DIR.glob("*.png"))
    kernels = sorted(p.stem.replace("_kernel", "")
                     for p in KERNELS_DIR.glob("*_kernel.png"))
    return originals, kernels


def _parse_test_name(stem: str,
                     orig_stems: list[str],
                     kernel_names: list[str]) -> tuple[str, str, str]:
    """<orig>_<kernel>[_<noise>] - (orig, kernel, noise)."""
    for orig in sorted(orig_stems, key=len, reverse=True):
        prefix = orig + "_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix):]
        for kname in sorted(kernel_names, key=len, reverse=True):
            if not rest.startswith(kname):
                continue
            noise_part = rest[len(kname):]
            if noise_part == "":
                return orig, kname, ""
            if noise_part.startswith("_"):
                return orig, kname, noise_part[1:]
    return stem, "", ""


def _coerce_params(raw: dict[str, Any], int_keys, odd_keys) -> dict[str, Any]:
    """Округлить int-ключи и сделать odd-ключи нечётными (как в HPO)."""
    p = dict(raw)
    for k in int_keys:
        if k in p and p[k] is not None and not (isinstance(p[k], float) and math.isnan(p[k])):
            p[k] = int(round(float(p[k])))
    for k in odd_keys:
        if k in p and isinstance(p[k], int):
            v = p[k]
            if v % 2 == 0:
                v += 1
            p[k] = max(v, 3)
    return p


def _adapt_kernel_param(value: int, kernel_is_tuple: bool):
    return (value, value) if kernel_is_tuple else value


def _load_best_params(
    history_path: Path,
    int_keys: list[str],
    odd_keys: list[str],
    fixed_kwargs: dict[str, Any],
    tunable_keys: list[str],
    kernel_param: str,
    kernel_is_tuple: bool,
) -> dict[str, Any] | None:
    """Читает history CSV и возвращает kwargs для строки с min(Objective)."""
    if history_path is None or not Path(history_path).exists():
        print(f"  ! history_path не найден: {history_path}")
        return None
    df = pd.read_csv(history_path)
    if "Objective" not in df.columns:
        print(f"  ! В {history_path} нет колонки Objective")
        return None
    best_row = df.loc[df["Objective"].idxmin()]
    best_iter = int(best_row.get("Iteration", -1))
    obj_val = float(best_row["Objective"])
    print(f"  best_hpo: iteration={best_iter}, Objective={obj_val:.6f}")

    raw = {k: best_row[k] for k in tunable_keys if k in df.columns}
    params = _coerce_params(raw, int_keys, odd_keys)

    kwargs = dict(fixed_kwargs)
    kwargs.update(params)
    if kernel_is_tuple and kernel_param in kwargs:
        kwargs[kernel_param] = _adapt_kernel_param(kwargs[kernel_param], True)

    kwargs["_best_hpo_iteration"] = best_iter
    kwargs["_best_hpo_objective"] = obj_val
    return kwargs


def _load_existing_results(out_dir: Path) -> dict[tuple, dict]:
    """Загружает результаты из уже существующих results_<mode>.csv."""
    existing: dict[tuple, dict] = {}
    for mode, fname in [("pareto",   "results_pareto.csv"),
                        ("standard", "results_standard.csv"),
                        ("best_hpo", "results_best_hpo.csv")]:
        csv_p = out_dir / fname
        if not csv_p.exists():
            continue
        try:
            df = pd.read_csv(csv_p)
        except Exception:
            continue
        for _, r in df.iterrows():
            restored = r.get("restored_path", "")
            if pd.isna(restored) or not Path(str(restored)).exists():
                continue
            key = (int(r["row_idx"]), mode)
            existing[key] = r.to_dict()
    return existing


def _init_worker(project_root: str):
    pr = str(project_root)
    src = str(Path(project_root) / "src")
    if pr not in sys.path:
        sys.path.insert(0, pr)
    if src not in sys.path:
        sys.path.insert(0, src)


def _error_ratio_nonblind(original, blurred, true_kernel, est_kernel):
    from numpy.fft import fft2, ifft2
    h, w = blurred.shape[:2]
    reg = 1e-3

    def _wr(y, k):
        K = fft2(k, s=(h, w))
        Y = fft2(y, s=(h, w))
        return np.real(ifft2(np.conj(K) * Y / (np.abs(K) ** 2 + reg)))

    x_true = _wr(blurred, true_kernel)
    x_est = _wr(blurred, est_kernel)
    orig = original.astype(np.float64)
    mse_true = np.mean((orig - x_true) ** 2)
    mse_est = np.mean((orig - x_est) ** 2)
    return mse_est / mse_true if mse_true > 1e-12 else 1.0


def process_one(task: dict) -> dict:
    """Запуск алгоритма для одной (row, mode) пары."""
    _init_worker(task["project_root"])
    from src.blinddeconv.processing.utils import (
        imread, prepare_image_for_metric, calculate_metrics,
    )

    dist_file   = Path(task["dist_file"])
    orig_path   = Path(task["orig_path"])     if task["orig_path"]   else None
    gt_k_path   = Path(task["gt_kernel_path"]) if task["gt_kernel_path"] else None
    save_rest   = Path(task["save_restored"])
    save_kernel = Path(task["save_kernel"])
    save_rest.parent.mkdir(parents=True, exist_ok=True)
    save_kernel.parent.mkdir(parents=True, exist_ok=True)

    blurred = imread(str(dist_file), False)
    if blurred is None:
        return {**task["row_meta"], "mode": task["mode"],
                "error": f"cannot read {dist_file}"}

    original = imread(str(orig_path), False) if orig_path and orig_path.exists() else None
    gt_kernel = imread(str(gt_k_path), False) if gt_k_path and gt_k_path.exists() else None

    mod = importlib.import_module(task["alg_module"])
    alg_cls = getattr(mod, task["alg_class"])
    alg_kwargs = dict(task["alg_kwargs"])

    t0 = time.time()
    try:
        restored, est_kernel = alg_cls(**alg_kwargs).process(blurred.copy())
    except Exception as e:
        import traceback
        return {**task["row_meta"], "mode": task["mode"],
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=3)}
    elapsed = time.time() - t0

    cv.imwrite(str(save_rest), restored)
    k_save = np.rot90(est_kernel.copy(), 2)
    if k_save.max() > 0:
        k_save = (k_save / k_save.max() * 255).astype(np.uint8)
    cv.imwrite(str(save_kernel), k_save)

    row = {**task["row_meta"], "mode": task["mode"],
           "time_sec":      round(elapsed, 3),
           "restored_path": str(save_rest),
           "kernel_path":   str(save_kernel),
           "kernel_size_used": alg_kwargs.get(task["kernel_param"], None),
           "psnr":          math.nan, "ssim":          math.nan,
           "psnr_blurred":  math.nan, "ssim_blurred":  math.nan,
           "error_ratio":   math.nan, "error":         None}

    if original is not None:
        orig_m = prepare_image_for_metric(np.atleast_3d(original))
        rest_m = prepare_image_for_metric(np.atleast_3d(restored))
        psnr_val, ssim_val = calculate_metrics(orig_m, rest_m, data_range=1.0, aligned=ALIGNED)
        row["psnr"] = round(psnr_val, 4)
        row["ssim"] = round(ssim_val, 4)

        blur_m = prepare_image_for_metric(np.atleast_3d(blurred))
        p_b, s_b = calculate_metrics(orig_m, blur_m, data_range=1.0, aligned=ALIGNED)
        row["psnr_blurred"] = round(p_b, 4)
        row["ssim_blurred"] = round(s_b, 4)

    if original is not None and gt_kernel is not None:
        try:
            orig_gray = original if original.ndim == 2 else original[:, :, 0]
            blur_gray = blurred  if blurred.ndim  == 2 else blurred[:, :, 0]
            gt_k = gt_kernel.astype(np.float64)
            if gt_k.ndim > 2:
                gt_k = gt_k[:, :, 0]
            gt_k = gt_k / (gt_k.sum() + 1e-12)
            est_k = est_kernel.astype(np.float64)
            if est_k.ndim > 2:
                est_k = est_k[:, :, 0]
            est_k = est_k / (est_k.sum() + 1e-12)
            er = _error_ratio_nonblind(
                orig_gray.astype(np.float64) / 255.0,
                blur_gray.astype(np.float64) / 255.0,
                gt_k, est_k,
            )
            row["error_ratio"] = round(er, 4)
        except Exception:
            pass

    return row


#ПОСТРОЕНИЕ ЗАДАЧ ДЛЯ ОДНОЙ ТАБЛИЦЫ

def _build_tasks_for_table(
    algo: dict[str, Any],
    table_path: Path,
    out_dir: Path,
    orig_stems: list[str],
    kernel_names: list[str],
    best_kwargs: dict[str, Any] | None,
    force: bool,
    existing: dict[tuple, dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Возвращает (tasks, cached_rows, missing_rows).

    Каждая строка - (pareto, standard, best_hpo).
    Если force=False и restored-файл уже есть в existing, задача не создаётся.
    """
    df = pd.read_csv(table_path)
    tasks: list[dict] = []
    cached_rows: list[dict] = []
    missing: list[dict] = []

    for idx, row in df.iterrows():
        test_name = str(row["test_name"])
        dist_file = DISTORTED_DIR / f"{test_name}.png"
        if not dist_file.exists():
            missing.append({"idx": idx, "test_name": test_name,
                            "reason": "distorted file not found"})
            continue

        orig_stem, kname, _ = _parse_test_name(test_name, orig_stems, kernel_names)
        orig_path = ORIGINALS_DIR / f"{orig_stem}.png"
        gt_k_path = KERNELS_DIR / f"{kname}_kernel.png" if kname else None
        orig_path_str = str(orig_path) if orig_path.exists() else ""
        gt_k_path_str = (str(gt_k_path)
                         if gt_k_path is not None and gt_k_path.exists() else "")

        #параметры из строки
        row_params_raw = {k: row[k] for k in algo["tunable_keys"] if k in df.columns}
        row_params = _coerce_params(row_params_raw,
                                    algo["int_keys"], algo["odd_keys"])

        #kwargs для pareto-посчета
        pareto_kwargs = dict(algo.get("pareto_fixed_kwargs", {}))
        pareto_kwargs.update(row_params)
        if algo["kernel_is_tuple"] and algo["kernel_param"] in pareto_kwargs:
            pareto_kwargs[algo["kernel_param"]] = _adapt_kernel_param(
                pareto_kwargs[algo["kernel_param"]], True)

        #kwargs для standard-подсчета
        standard_kwargs = dict(algo["standard_kwargs"])
        if KERNEL_SIZE_FIXED is None:
            if algo["kernel_param"] in row_params:
                ks_val = row_params[algo["kernel_param"]]
                standard_kwargs[algo["kernel_param"]] = _adapt_kernel_param(
                    ks_val, algo["kernel_is_tuple"])
        else:
            standard_kwargs[algo["kernel_param"]] = _adapt_kernel_param(
                KERNEL_SIZE_FIXED, algo["kernel_is_tuple"])

        row_meta = {
            "row_idx":    int(idx),
            "iteration":  int(row["iteration"]) if "iteration" in df.columns else -1,
            "test_name":  test_name,
            "image_name": orig_stem,
            "kernel_name": kname,
        }
        for col in algo["tunable_keys"]:
            if col in df.columns:
                row_meta[f"pareto_{col}"] = row[col]

        prefix = f"{int(idx):03d}_{test_name}"
        common = {
            "project_root":   str(PROJECT_ROOT),
            "alg_module":     algo["alg_module"],
            "alg_class":      algo["alg_class"],
            "kernel_param":   algo["kernel_param"],
            "dist_file":      str(dist_file),
            "orig_path":      orig_path_str,
            "gt_kernel_path": gt_k_path_str,
            "row_meta":       row_meta,
        }

        modes_to_run = [
            ("pareto",   pareto_kwargs),
            ("standard", standard_kwargs),
        ]
        if best_kwargs is not None:
            bk = {k: v for k, v in best_kwargs.items()
                  if not k.startswith("_")}
            modes_to_run.append(("best_hpo", bk))

        for mode, mode_kwargs in modes_to_run:
            key = (int(idx), mode)
            if not force and key in existing:
                cached_rows.append(existing[key])
                print(f"    [skip] {mode:>8} {test_name}")
                continue
            tasks.append({
                **common,
                "mode":          mode,
                "alg_kwargs":    mode_kwargs,
                "save_restored": str(out_dir / mode / "restored" / f"{prefix}.png"),
                "save_kernel":   str(out_dir / mode / "kernels"  / f"{prefix}_kernel.png"),
            })

    return tasks, cached_rows, missing


#ОБРАБОТКА ОДНОЙ ТАБЛИЦЫ

def _run_table(algo: dict[str, Any], table_path: Path, alg_out_root: Path,
               orig_stems: list[str], kernel_names: list[str],
               best_kwargs: dict[str, Any] | None,
               force: bool) -> None:
    table_stem = table_path.stem
    out_dir = alg_out_root / table_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    all_modes = ["pareto", "standard", "best_hpo"]
    for mode in all_modes:
        for sub in ["restored", "kernels"]:
            (out_dir / mode / sub).mkdir(parents=True, exist_ok=True)

    existing = {} if force else _load_existing_results(out_dir)
    if existing:
        print(f"  Найдено {len(existing)} уже готовых результатов (--force чтобы пересчитать)")

    tasks, cached_rows, missing = _build_tasks_for_table(
        algo, table_path, out_dir, orig_stems, kernel_names,
        best_kwargs, force, existing)

    n_rows = len(pd.read_csv(table_path))
    n_modes = 3 if best_kwargs is not None else 2
    print(f"\n[{algo['label']} / {table_stem}] {n_rows} строк × {n_modes} режима "
          f"- {len(tasks)} новых подсчетов, {len(cached_rows)} кэш; "
          f"пропущено строк: {len(missing)}")
    if missing:
        pd.DataFrame(missing).to_csv(out_dir / "skipped.csv", index=False)

    new_rows: list[dict] = []
    if tasks:
        if NUM_WORKERS > 1:
            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=_init_worker,
                initargs=(str(PROJECT_ROOT),),
            ) as pool:
                futures = {pool.submit(process_one, t): t for t in tasks}
                done = 0
                total = len(tasks)
                for fut in as_completed(futures):
                    r = fut.result()
                    new_rows.append(r)
                    done += 1
                    tag = f"[{r.get('mode'):>8}]"
                    err = r.get("error")
                    if err:
                        print(f"  {done}/{total} {tag} {r.get('test_name')}  ✗  {err}")
                    else:
                        print(f"  {done}/{total} {tag} {r.get('test_name')}  "
                              f"PSNR={r.get('psnr')} SSIM={r.get('ssim')}  "
                              f"t={r.get('time_sec')}s")
        else:
            for i, t in enumerate(tasks, 1):
                r = process_one(t)
                new_rows.append(r)
                print(f"  {i}/{len(tasks)} [{r.get('mode')}] {r.get('test_name')}: "
                      f"PSNR={r.get('psnr')} SSIM={r.get('ssim')}")

    all_rows = new_rows + cached_rows
    if not all_rows:
        print(f"  ! Нет результатов для {table_stem}")
        return

    df_all = pd.DataFrame(all_rows)

    mode_dfs: dict[str, pd.DataFrame] = {}
    for mode, fname in [("pareto",   "results_pareto.csv"),
                        ("standard", "results_standard.csv"),
                        ("best_hpo", "results_best_hpo.csv")]:
        df_m = df_all[df_all["mode"] == mode].copy().sort_values("row_idx")
        if not df_m.empty:
            df_m.to_csv(out_dir / fname, index=False)
            mode_dfs[mode] = df_m

    keep_cols = ["row_idx", "iteration", "test_name", "image_name",
                 "kernel_name", "kernel_size_used",
                 "psnr", "ssim", "psnr_blurred", "ssim_blurred",
                 "error_ratio", "time_sec",
                 "restored_path", "kernel_path", "error"]
    base_cols = ["row_idx", "iteration", "test_name", "image_name",
                 "kernel_name", "psnr_blurred", "ssim_blurred"]

    df_comb: pd.DataFrame | None = None
    for mode, df_m in mode_dfs.items():
        avail_keep = [c for c in keep_cols if c in df_m.columns]
        avail_base = [c for c in base_cols if c in df_m.columns]
        part = df_m[avail_keep].rename(columns={
            c: f"{c}__{mode}" for c in avail_keep if c not in avail_base})
        if df_comb is None:
            df_comb = part
        else:
            on_cols = [c for c in avail_base if c in df_comb.columns]
            df_comb = df_comb.merge(part, on=on_cols, how="outer")

    if df_comb is not None:
        # Дельты: каждый режим относительно standard
        for m in ["psnr", "ssim", "error_ratio", "time_sec"]:
            cs = f"{m}__standard"
            if cs not in df_comb.columns:
                continue
            for mode in ["pareto", "best_hpo"]:
                cp = f"{m}__{mode}"
                if cp in df_comb.columns:
                    df_comb[f"delta_{m}__{mode}_vs_standard"] = (
                        df_comb[cp] - df_comb[cs])
        df_comb.to_csv(out_dir / "results_combined.csv", index=False)

    if df_comb is not None:
        print(f"  [{table_stem}] средние:")
        header = f"  {'metric':12s}  {'standard':>10}  {'pareto':>10}  {'best_hpo':>10}"
        print(header)
        for m in ["psnr", "ssim", "error_ratio"]:
            cs = f"{m}__standard"
            cp = f"{m}__pareto"
            cb = f"{m}__best_hpo"
            vs = df_comb[cs].mean() if cs in df_comb.columns else float("nan")
            vp = df_comb[cp].mean() if cp in df_comb.columns else float("nan")
            vb = df_comb[cb].mean() if cb in df_comb.columns else float("nan")
            print(f"  {m:12s}  {vs:>10.4f}  {vp:>10.4f}  {vb:>10.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default=None,
                        help="Запустить только один алгоритм по названию")
    parser.add_argument("--table", default=None,
                        help="Запустить только одну таблицу (имя файла)")
    parser.add_argument("--force", action="store_true",
                        help="Пересчитать все результаты, даже если они "
                             "уже существуют")
    args = parser.parse_args()

    orig_stems, kernel_names = _build_dataset_index()
    print(f"Датасет: {len(orig_stems)} оригиналов, {len(kernel_names)} ядер")
    print(f"Standard kernel_size: {KERNEL_SIZE_FIXED} "
          f"({'fixed' if KERNEL_SIZE_FIXED is not None else 'match Pareto row'})")
    print(f"Параллельных процессов: {NUM_WORKERS}")
    if args.force:
        print("Режим --force: все результаты будут пересчитаны")
    else:
        print("Скип-режим: уже существующие результаты пропускаются "
              "(--force чтобы пересчитать)")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for algo in ALGORITHM_CONFIGS:
        if args.algo and algo["label"] != args.algo:
            continue
        print("\n" + "=" * 80)
        print(f"АЛГОРИТМ: {algo['label']}")
        print("=" * 80)

        hist_path = algo.get("history_path", None)
        best_kwargs: dict[str, Any] | None = None
        if hist_path is not None:
            best_kwargs = _load_best_params(
                history_path=hist_path,
                int_keys=algo["int_keys"],
                odd_keys=algo["odd_keys"],
                fixed_kwargs=algo.get("pareto_fixed_kwargs", {}),
                tunable_keys=algo["tunable_keys"],
                kernel_param=algo["kernel_param"],
                kernel_is_tuple=algo["kernel_is_tuple"],
            )
        else:
            print("  history_path не задан — режим best_hpo отключён")

        alg_out_root = OUTPUT_ROOT / algo["label"]
        alg_out_root.mkdir(parents=True, exist_ok=True)

        for fname in PARETO_TABLE_FILES:
            if args.table and args.table not in (fname, Path(fname).stem):
                continue
            table_path = algo["pareto_tables_dir"] / fname
            if not table_path.exists():
                print(f"  ! Таблица не найдена: {table_path}")
                continue
            _run_table(algo, table_path, alg_out_root, orig_stems, kernel_names,
                       best_kwargs=best_kwargs, force=args.force)

    print("\nГОТОВО.")


if __name__ == "__main__":
    main()
