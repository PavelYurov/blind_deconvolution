#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Сравнение non-blind шагов BID_HBSP (Hyperbolic Secant Prior)
на датасете nonblind.

Для каждого из методов (firls / irls / tikhonov / ringing / adaptive_lp/ weiner)
запускается полный подсчет HBSP с соответствующим final_deconv.
Результаты каждого метода сохраняются в отдельную папку:
  presentation_graphics/Hyperbolic_Secant_Prior_(FIRLS)/
  presentation_graphics/Hyperbolic_Secant_Prior_(IRLS)/
  ...

Запуск:
  python run_all_dataset_hbsp_nonblind.py                  # все методы
  python run_all_dataset_hbsp_nonblind.py --method firls   # только один
"""

import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

import _process_worker
import visualisation as vis

#ОБЩИЕ НАСТРОЙКИ
NUM_WORKERS = 6

DATASETS_ROOT   = PROJECT_ROOT / "images" / "compare_data" / "kostya"
RESULTS_ROOT    = PROJECT_ROOT / "presentation_graphics"

DATASETS_PHASE1 = ["nonblind"]

ALG_MODULE       = ("src.blinddeconv.algorithms.blind_deconvolution."
                    "our_company.bayesian.bid_hbsp_cython._build_pyd.bid_hbsp")
ALG_CLASS        = "BID_HBSP"
ALG_KERNEL_PARAM = "kernel_shape"

BASE_ALG_KWARGS = {
    'kernel_shape':       (35, 35),   # адаптируется под GT ядро
    'impulse_preprocess': 'auto',
    'auto_mode':          'robust',
    'auto_mode_params': {
        'poisson_denoiser': 'vst_bm3d',
    },
}

# NON-BLIND КОНФИГУРАЦИИ: 'irls', 'adaptive_lp', 'wiener','tikhonov', 'ringing', 'firls'
NON_BLIND_CONFIGS = [
    {
        'key':          'firls',
        'label':        'Hyperbolic Secant Prior (FIRLS)',
        'final_deconv': 'firls',
    },
    {
        'key':          'irls',
        'label':        'Hyperbolic Secant Prior (IRLS)',
        'final_deconv': 'irls',
    },
    {
        'key':          'tikhonov',
        'label':        'Hyperbolic Secant Prior (Tikhonov)',
        'final_deconv': 'tikhonov',
    },
    {
        'key':          'ringing',
        'label':        'Hyperbolic Secant Prior (TV+L0)',
        'final_deconv': 'ringing',
    },
    {
        'key':          'adaptive_lp',
        'label':        'Hyperbolic Secant Prior (Adaptive Lp)',
        'final_deconv': 'adaptive_lp',
    },
]


# ПОДСЧЕТ ОДНОГО МЕТОДА

def run_phase1(algorithm_label: str, alg_kwargs: dict):
    """
    Подсчет алгоритма с заданными alg_kwargs на всех датасетах из
    DATASETS_PHASE1. Результаты сохраняются в:
      presentation_graphics/<algorithm_label
    """
    alg_results_dir = RESULTS_ROOT / algorithm_label.replace(" ", "_")
    alg_results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  [{algorithm_label}]  final_deconv={alg_kwargs['final_deconv']!r}")
    print("=" * 80)

    all_dataset_results = []

    for dataset_name in DATASETS_PHASE1:
        dataset_dir = DATASETS_ROOT / dataset_name
        if not dataset_dir.exists():
            print(f"! Датасет {dataset_name} не найден: {dataset_dir}. Пропуск.")
            continue

        dist_dir   = dataset_dir / "distorted"
        orig_dir   = dataset_dir / "originals"
        kernel_dir = dataset_dir / "ground_truth_filters"

        has_originals = orig_dir.exists() and any(orig_dir.iterdir())
        has_kernels   = kernel_dir.exists() and any(kernel_dir.iterdir())

        ds_result_dir = alg_results_dir / dataset_name
        ds_result_dir.mkdir(parents=True, exist_ok=True)
        (ds_result_dir / "restored").mkdir(exist_ok=True)
        (ds_result_dir / "kernels").mkdir(exist_ok=True)

        distorted_files = sorted([f for f in dist_dir.iterdir() if f.is_file()])
        if not distorted_files:
            print(f"! Датасет {dataset_name}: нет файлов в distorted/. Пропуск.")
            continue

        print(f"\n[{dataset_name}] {len(distorted_files)} изображений")

        tasks = [
            {
                'project_root':     str(PROJECT_ROOT),
                'dist_file':        str(dist_file),
                'orig_dir':         str(orig_dir),
                'kernel_dir':       str(kernel_dir),
                'has_originals':    has_originals,
                'has_kernels':      has_kernels,
                'dataset_name':     dataset_name,
                'algorithm_label':  algorithm_label,
                'alg_module':       ALG_MODULE,
                'alg_class':        ALG_CLASS,
                'alg_kwargs':       alg_kwargs,
                'alg_kernel_param': ALG_KERNEL_PARAM,
                'ds_result_dir':    str(ds_result_dir),
                'num_runs':         1,
            }
            for dist_file in distorted_files
        ]

        results_rows = []
        if NUM_WORKERS > 1:
            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=_process_worker.init_worker,
                initargs=(str(PROJECT_ROOT),),
            ) as pool:
                futures = {
                    pool.submit(_process_worker.process_one, t): t
                    for t in tasks
                }
                for future in as_completed(futures):
                    r = future.result()
                    if r and 'error' not in r:
                        results_rows.append(r)
                        print(f"  v {r['distorted_file']:<45s}"
                              f"  PSNR={r.get('psnr', 'n/a')}  t={r['time_sec']}s")
                    elif r:
                        print(f"  x {r.get('dist_file', '?')}: {r['error']}")
        else:
            for t in tasks:
                r = _process_worker.process_one(t)
                if r and 'error' not in r:
                    results_rows.append(r)
                    print(f"  v {r['distorted_file']:<45s}"
                          f"  PSNR={r.get('psnr', 'n/a')}  t={r['time_sec']}s")
                elif r:
                    print(f"  x {r.get('dist_file', '?')}: {r['error']}")

        if not results_rows:
            continue

        df = pd.DataFrame(results_rows)
        csv_path = ds_result_dir / f"results_{algorithm_label}_{dataset_name}.csv"
        df.to_csv(csv_path, index=False)

        vis.save_complex_plots(
            df, dist_dir,
            ds_result_dir / "complex_plots", algorithm_label)

        all_dataset_results.append(df)
        print(f"  CSV: {csv_path}")

    if all_dataset_results:
        df_all = pd.concat(all_dataset_results, ignore_index=True)
        csv_all = alg_results_dir / f"all_results_{algorithm_label}.csv"
        df_all.to_csv(csv_all, index=False)
        print(f"  Общий CSV: {csv_all}")


def main():
    valid_keys = [c['key'] for c in NON_BLIND_CONFIGS]
    parser = argparse.ArgumentParser(
        description="Сравнение non-blind шагов BID_HBSP на датасете nonblind")
    parser.add_argument(
        '--method', type=str, default=None,
        help=f"Запустить только один метод: {valid_keys}")
    args = parser.parse_args()

    configs = NON_BLIND_CONFIGS
    if args.method is not None:
        configs = [c for c in NON_BLIND_CONFIGS if c['key'] == args.method]
        if not configs:
            print(f"Неизвестный метод: {args.method!r}. Доступные: {valid_keys}")
            return

    t_start = time.time()

    for cfg in configs:
        alg_kwargs = dict(BASE_ALG_KWARGS)
        alg_kwargs['auto_mode_params'] = dict(BASE_ALG_KWARGS['auto_mode_params'])
        alg_kwargs['final_deconv'] = cfg['final_deconv']
        run_phase1(cfg['label'], alg_kwargs)

    elapsed = time.time() - t_start
    hours = int(elapsed // 3600)
    mins  = int((elapsed % 3600) // 60)
    print(f"\n{'=' * 80}")
    print(f"[ГОТОВО] {hours}ч {mins}мин — результаты в {RESULTS_ROOT}")


if __name__ == "__main__":
    main()