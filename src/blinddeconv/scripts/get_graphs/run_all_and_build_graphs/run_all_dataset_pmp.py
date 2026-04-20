#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Полный скрипт прогона алгоритма PMP на всех датасетах + итерационное
логирование + гиперпараметрическая сетка.

Три фазы:
  1) Прогон на датасетах (качество + скорость + стресс-тест)
  2) Итерационное логирование (3-5 отобранных картинок)
  3) Гиперпараметрическая сетка (1-2 картинки, 2 параметра)

Запуск:
  python run_all_dataset_pmp.py              # все фазы
  python run_all_dataset_pmp.py --phase 1    # только датасеты
  python run_all_dataset_pmp.py --phase 2    # только итерации
  python run_all_dataset_pmp.py --phase 3    # только гиперпараметры

Для запуска на другом компьютере:
  - Скопируй скрипт, поменяй DATASETS_PHASE1 (например, убери Complexity_Test)
  - Результаты потом скопируй в presentation_graphics/<ALG_LABEL>/
"""

import sys
import os
import math
import time
import argparse
import itertools
import importlib
import pandas as pd
import numpy as np
import cv2 as cv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Автоопределение корня
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

import _process_worker
import visualisation as vis

# ══════════════════════════════════════════════════════════════════════════════
# ОБЩИЕ НАСТРОЙКИ
# ══════════════════════════════════════════════════════════════════════════════
ALGORITHM_LABEL = "Patch-wise Min Pixels"
NUM_WORKERS = 6  # Количество процессов для параллельного прогона датасетов

# Пути
DATASETS_ROOT = PROJECT_ROOT / "images" / "compare_data" / "anton"
RESULTS_ROOT  = PROJECT_ROOT / "presentation_graphics"
ALG_RESULTS_DIR = RESULTS_ROOT / ALGORITHM_LABEL.replace(" ", "_")
ALG_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Модуль алгоритма (для _process_worker)
# >>> УКАЖИ СВОЙ МОДУЛЬ (pmp_denoise или pmp_cython_pyd) <<<
ALG_MODULE = "src.blinddeconv.algorithms.blind_deconvolution.our_company.patch_wise_minimum_pixels_prior.pmp_cython_pyd.pmp"
ALG_CLASS  = "PMP_BD"
ALG_KERNEL_PARAM = "kernel_size"

# Импорт алгоритма — ОБЯЗАТЕЛЬНО pmp_denoise (Python-версия с поддержкой callback).
# pmp_cython_pyd — скомпилированный .pyd без callback, итерации не логируются.
ALG_MODULE_DENOISE = ("src.blinddeconv.algorithms.blind_deconvolution."
                        "our_company.patch_wise_minimum_pixels_prior."
                        "pmp_denoise.pmp")

# ══════════════════════════════════════════════════════════════════════════════
# ПАРАМЕТРЫ АЛГОРИТМА (лучшие найденные)
# >>> УКАЖИ СВОИ ЛУЧШИЕ ПАРАМЕТРЫ <<<
# ══════════════════════════════════════════════════════════════════════════════
ALG_KWARGS = {
    'kernel_size': 51,              # будет адаптирован под GT ядро
    'denoise_eps': 0.006,
    'denoise_radius': 2,
    'lambda_tv': 0.005,
    'lambda_l0': 0.002,
    'grad_smooth_sigma': 0.285,
    'ensemble_denoise': True,
    'impulse_preprocess': 'auto',
    'noise_estimation': 'pca',
    'preprocess': 'bm3d',
    'blind_denoise': 'guided',
    'pre_nonblind': 'bm3d',
}


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 1: ПРОГОН НА ДАТАСЕТАХ
# ══════════════════════════════════════════════════════════════════════════════

# >>> УКАЖИ ДАТАСЕТЫ ДЛЯ ПРОГОНА <<<
# Complexity_Test: проверка скорости (запускать только на одном компьютере!)
# Grid_Test: стресс-тест рабочей области
# Остальные: качество (сравниваются между собой через generate_visuals.py)
DATASETS_PHASE1 = [
    # "Complexity_Test",   # раскомментируй на том компе, где меряешь скорость
    "Grid_Test",
    "Levin",
    "Kohler",
    # "Lai",
    # "Set12",
    # "Sun",
    # >>> добавь свои датасеты качества <<<
]


def run_phase1_datasets():
    """Фаза 1: прогон алгоритма на датасетах через ProcessPoolExecutor."""
    print("=" * 80)
    print(f"ФАЗА 1: Прогон [{ALGORITHM_LABEL}] на датасетах")
    print("=" * 80)

    all_dataset_results = []

    for dataset_name in DATASETS_PHASE1:
        dataset_dir = DATASETS_ROOT / dataset_name
        if not dataset_dir.exists():
            print(f"! Датасет {dataset_name} не найден: {dataset_dir}. Пропуск.")
            continue

        dist_dir = dataset_dir / "distorted"
        orig_dir = dataset_dir / "originals"
        kernel_dir = dataset_dir / "ground_truth_filters"

        has_originals = orig_dir.exists() and any(orig_dir.iterdir())
        has_kernels = kernel_dir.exists() and any(kernel_dir.iterdir())

        ds_result_dir = ALG_RESULTS_DIR / dataset_name
        ds_result_dir.mkdir(parents=True, exist_ok=True)
        (ds_result_dir / "restored").mkdir(exist_ok=True)
        (ds_result_dir / "kernels").mkdir(exist_ok=True)

        distorted_files = sorted([f for f in dist_dir.iterdir() if f.is_file()])
        if not distorted_files:
            print(f"! Датасет {dataset_name}: нет файлов в distorted/. Пропуск.")
            continue

        # Complexity_Test: несколько прогонов для стабильного времени
        runs = 3 if dataset_name == "Complexity_Test" else 1

        print(f"\n[{dataset_name}] {len(distorted_files)} изображений (прогонов: {runs})")

        tasks = []
        for dist_file in distorted_files:
            tasks.append({
                'project_root': str(PROJECT_ROOT),
                'dist_file': str(dist_file),
                'orig_dir': str(orig_dir),
                'kernel_dir': str(kernel_dir),
                'has_originals': has_originals,
                'has_kernels': has_kernels,
                'dataset_name': dataset_name,
                'algorithm_label': ALGORITHM_LABEL,
                'alg_module': ALG_MODULE,
                'alg_class': ALG_CLASS,
                'alg_kwargs': ALG_KWARGS,
                'alg_kernel_param': ALG_KERNEL_PARAM,
                'ds_result_dir': str(ds_result_dir),
                'num_runs': runs,
            })

        results_rows = []
        if NUM_WORKERS > 1:
            with ProcessPoolExecutor(
                max_workers=NUM_WORKERS,
                initializer=_process_worker.init_worker,
                initargs=(str(PROJECT_ROOT),),
            ) as pool:
                futures = {pool.submit(_process_worker.process_one, t): t for t in tasks}
                for future in as_completed(futures):
                    r = future.result()
                    if r and 'error' not in r:
                        results_rows.append(r)
                        print(f"  v {r['distorted_file']}  t={r['time_sec']}с")
                    elif r and 'error' in r:
                        print(f"  x {r['dist_file']}: {r['error']}")
        else:
            for t in tasks:
                r = _process_worker.process_one(t)
                if r and 'error' not in r:
                    results_rows.append(r)
                    print(f"  v {r['distorted_file']}  t={r['time_sec']}с")

        if not results_rows:
            continue

        df_results = pd.DataFrame(results_rows)
        csv_path = ds_result_dir / f"results_{ALGORITHM_LABEL}_{dataset_name}.csv"
        df_results.to_csv(csv_path, index=False)

        # Комплексные графики 2x4
        vis.save_complex_plots(
            df_results, dist_dir,
            ds_result_dir / "complex_plots", ALGORITHM_LABEL)

        all_dataset_results.append(df_results)

    # Общий CSV
    if all_dataset_results:
        df_all = pd.concat(all_dataset_results, ignore_index=True)
        df_all.to_csv(ALG_RESULTS_DIR / f"all_results_{ALGORITHM_LABEL}.csv", index=False)
        print(f"\n[ФАЗА 1 OK] Результаты: {ALG_RESULTS_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 2: ИТЕРАЦИОННОЕ ЛОГИРОВАНИЕ
# ══════════════════════════════════════════════════════════════════════════════

# >>> УКАЖИ ДАТАСЕТ ДЛЯ ИТЕРАЦИОННОГО ЛОГИРОВАНИЯ <<<
# Отдельный датасет с 3-5 отобранными картинками (не участвует в сравнении).
# Структура:  images/compare_data/Iteration_Test/distorted/
#             images/compare_data/Iteration_Test/originals/
#             images/compare_data/Iteration_Test/ground_truth_filters/
# ITERATION_DATASET = "Iteration_Test"
ITERATION_DATASET = "log_test"


# Частота сохранения:
SAVE_KERNEL_EVERY = 1    # сохранять ядро каждую итерацию (они маленькие)
SAVE_IMAGE_EVERY  = 1    # сохранять восстановленное изображение каждую итерацию


def _make_nonblind_func(alg_kwargs):
    """Создаёт callable для non-blind шага PMP (ringing_artifacts_removal)."""
    from src.blinddeconv.algorithms.blind_deconvolution.our_company.\
        patch_wise_minimum_pixels_prior.pmp_denoise.solvers import \
        ringing_artifacts_removal

    lambda_tv = alg_kwargs.get('lambda_tv', 0.005)
    lambda_l0 = alg_kwargs.get('lambda_l0', 0.002)
    weight_ring = alg_kwargs.get('weight_ring', 1.0)

    def nonblind(blurred, kernel):
        return ringing_artifacts_removal(
            blurred, kernel, lambda_tv, lambda_l0, weight_ring)

    return nonblind


def run_phase2_iterations():
    """Фаза 2: прогон с итерационным логированием (последовательно)."""
    print("\n" + "=" * 80)
    print(f"ФАЗА 2: Итерационное логирование [{ALGORITHM_LABEL}]")
    print("=" * 80)

    from src.blinddeconv.algorithms.iteration_logger import IterationLogger
    from src.blinddeconv.processing.utils import imread

    dataset_dir = DATASETS_ROOT / ITERATION_DATASET
    if not dataset_dir.exists():
        print(f"! Датасет {ITERATION_DATASET} не найден: {dataset_dir}")
        print("  Создай папку с 3-5 отобранными картинками и перезапусти.")
        return

    dist_dir = dataset_dir / "distorted"
    orig_dir = dataset_dir / "originals"
    kernel_dir = dataset_dir / "ground_truth_filters"

    distorted_files = sorted([f for f in dist_dir.iterdir() if f.is_file()])
    if not distorted_files:
        print(f"  Нет файлов в {dist_dir}. Пропуск.")
        return

    iter_results_dir = ALG_RESULTS_DIR / ITERATION_DATASET
    iter_results_dir.mkdir(parents=True, exist_ok=True)

    
    mod = importlib.import_module(ALG_MODULE_DENOISE)
    alg_cls = getattr(mod, ALG_CLASS)

    nonblind_func = _make_nonblind_func(ALG_KWARGS)

    print(f"  Найдено {len(distorted_files)} изображений для логирования")

    all_rows = []

    for dist_file in distorted_files:
        stem = dist_file.stem
        parts = stem.split("_")
        img_name = parts[0] if parts else stem
        kernel_name = parts[1] if len(parts) >= 2 else ""

        print(f"\n  [{stem}] Запуск с итерационным логированием...")

        # Загрузка изображений
        blurred = imread(str(dist_file), False)
        if blurred is None:
            print(f"    ! Не удалось прочитать {dist_file}")
            continue

        # Оригинал
        original = None
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            candidate = orig_dir / f"{img_name}{ext}"
            if candidate.exists():
                original = imread(str(candidate), False)
                break

        # GT ядро
        gt_kernel = None
        if kernel_name and kernel_dir.exists():
            for f in kernel_dir.iterdir():
                if f.is_file() and kernel_name in f.stem:
                    gt_kernel = imread(str(f), False)
                    break

        if gt_kernel is not None:
            gt_k = gt_kernel.astype(np.float64)
            if gt_k.ndim > 2:
                gt_k = gt_k[:, :, 0]
            gt_k = gt_k / (gt_k.sum() + 1e-12)
        else:
            gt_k = None

        # Определение размера ядра
        if gt_kernel is not None:
            ks = gt_kernel.shape[0]
            if ks % 2 == 0:
                ks += 1
        else:
            ks = ALG_KWARGS.get('kernel_size', 21)

        # Grayscale для non-blind
        if blurred.ndim == 3:
            blur_gray = (0.2989 * blurred[:, :, 0]
                         + 0.587 * blurred[:, :, 1]
                         + 0.114 * blurred[:, :, 2])
        else:
            blur_gray = blurred.astype(np.float64)
        if blur_gray.max() > 1.0:
            blur_gray = blur_gray / 255.0

        # Создание логгера
        img_save_dir = iter_results_dir / stem
        logger = IterationLogger(
            save_dir=img_save_dir,
            original=original,
            gt_kernel=gt_k,
            blurred=blur_gray,
            nonblind_func=nonblind_func,
            save_kernel_every=SAVE_KERNEL_EVERY,
            save_image_every=SAVE_IMAGE_EVERY,
            only_finest_scale=True,
        )

        # Создание алгоритма с callback
        kwargs = dict(ALG_KWARGS)
        kwargs['kernel_size'] = ks
        alg = alg_cls(**kwargs)
        alg.set_callback(logger)

        t0 = time.time()
        try:
            restored, est_kernel = alg.process(blurred)
        except Exception as e:
            print(f"    ! Ошибка: {e}")
            import traceback; traceback.print_exc()
            continue
        elapsed = time.time() - t0

        # Сохранение CSV итераций
        logger.save_csv()

        # Сохранение финальных результатов
        cv.imwrite(str(img_save_dir / "restored_final.png"), restored)
        k_save = np.rot90(est_kernel.copy(), 2)
        if k_save.max() > 0:
            k_save = (k_save / k_save.max() * 255).astype(np.uint8)
        cv.imwrite(str(img_save_dir / "kernel_final.png"), k_save)

        n_iters = len(logger.log)
        print(f"    OK: {n_iters} итераций залогировано, t={elapsed:.1f}с")
        print(f"    Сохранено в: {img_save_dir}")

        all_rows.append({
            'distorted_file': dist_file.name,
            'image_name': img_name,
            'kernel_name': kernel_name,
            'num_iterations': n_iters,
            'time_sec': round(elapsed, 3),
        })

    # Общий CSV фазы 2
    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            iter_results_dir / f"iteration_summary_{ALGORITHM_LABEL}.csv",
            index=False)
        print(f"\n[ФАЗА 2 OK] Итерационные логи: {iter_results_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# ФАЗА 3: ГИПЕРПАРАМЕТРИЧЕСКАЯ СЕТКА (п.12 + п.15)
# ══════════════════════════════════════════════════════════════════════════════

# >>> УКАЖИ ДАТАСЕТ ДЛЯ ГИПЕРПАРАМЕТРОВ <<<
# Можно тот же Iteration_Test или отдельный — главное 1-2 картинки.
# HYPERPARAM_DATASET = "Iteration_Test"
HYPERPARAM_DATASET = "param_grid_test"


# >>> УКАЖИ ДВА ПАРАМЕТРА ДЛЯ СЕТКИ <<<
# Эти параметры будут варьироваться. Остальные фиксированы из ALG_KWARGS.
GRID_PARAM_1 = "lambda_tv"
GRID_VALUES_1 = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

GRID_PARAM_2 = "lambda_l0"
GRID_VALUES_2 = [0.00005, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]


def run_phase3_hyperparam_grid():
    """Фаза 3: прогон на 2D сетке гиперпараметров (последовательно)."""
    print("\n" + "=" * 80)
    print(f"ФАЗА 3: Гиперпараметрическая сетка [{ALGORITHM_LABEL}]")
    print(f"  {GRID_PARAM_1}: {len(GRID_VALUES_1)} значений")
    print(f"  {GRID_PARAM_2}: {len(GRID_VALUES_2)} значений")
    print(f"  Всего комбинаций: {len(GRID_VALUES_1) * len(GRID_VALUES_2)}")
    print("=" * 80)

    from src.blinddeconv.processing.utils import (
        imread, prepare_image_for_metric, calculate_metrics,
    )

    dataset_dir = DATASETS_ROOT / HYPERPARAM_DATASET
    if not dataset_dir.exists():
        print(f"! Датасет {HYPERPARAM_DATASET} не найден: {dataset_dir}")
        return

    dist_dir = dataset_dir / "distorted"
    orig_dir = dataset_dir / "originals"
    kernel_dir = dataset_dir / "ground_truth_filters"

    distorted_files = sorted([f for f in dist_dir.iterdir() if f.is_file()])
    if not distorted_files:
        print(f"  Нет файлов в {dist_dir}. Пропуск.")
        return

    grid_results_dir = ALG_RESULTS_DIR / "hyperparam_grid"
    grid_results_dir.mkdir(parents=True, exist_ok=True)

    # Импорт алгоритма
    mod = importlib.import_module(ALG_MODULE)
    alg_cls = getattr(mod, ALG_CLASS)

    combos = list(itertools.product(GRID_VALUES_1, GRID_VALUES_2))
    total = len(combos) * len(distorted_files)
    print(f"  Всего запусков: {total} ({len(combos)} комбинаций × "
          f"{len(distorted_files)} изображений)")

    all_results = []
    counter = 0

    for dist_file in distorted_files:
        stem = dist_file.stem
        parts = stem.split("_")
        img_name = parts[0] if parts else stem
        kernel_name = parts[1] if len(parts) >= 2 else ""

        # Загрузка
        blurred = imread(str(dist_file), False)
        if blurred is None:
            continue

        original = None
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            candidate = orig_dir / f"{img_name}{ext}"
            if candidate.exists():
                original = imread(str(candidate), False)
                break

        gt_kernel = None
        if kernel_name and kernel_dir.exists():
            for f in kernel_dir.iterdir():
                if f.is_file() and kernel_name in f.stem:
                    gt_kernel = imread(str(f), False)
                    break

        # Размер ядра
        if gt_kernel is not None:
            ks = gt_kernel.shape[0]
            if ks % 2 == 0:
                ks += 1
        else:
            ks = ALG_KWARGS.get('kernel_size', 21)

        print(f"\n  [{stem}] Прогон {len(combos)} комбинаций...")

        for v1, v2 in combos:
            counter += 1
            kwargs = dict(ALG_KWARGS)
            kwargs['kernel_size'] = ks
            kwargs[GRID_PARAM_1] = v1
            kwargs[GRID_PARAM_2] = v2

            t0 = time.time()
            try:
                alg = alg_cls(**kwargs)
                restored, est_kernel = alg.process(blurred.copy())
                elapsed = time.time() - t0
            except Exception as e:
                print(f"    [{counter}/{total}] {GRID_PARAM_1}={v1}, "
                      f"{GRID_PARAM_2}={v2} → ERROR: {e}")
                all_results.append({
                    'image': stem,
                    GRID_PARAM_1: v1,
                    GRID_PARAM_2: v2,
                    'psnr': None, 'ssim': None,
                    'time_sec': time.time() - t0,
                })
                continue

            # Метрики
            row = {
                'image': stem,
                GRID_PARAM_1: v1,
                GRID_PARAM_2: v2,
                'time_sec': round(elapsed, 3),
            }

            if original is not None:
                orig_m = prepare_image_for_metric(np.atleast_3d(original))
                rest_m = prepare_image_for_metric(np.atleast_3d(restored))
                psnr_val, ssim_val = calculate_metrics(
                    orig_m, rest_m, data_range=1.0)
                row['psnr'] = round(psnr_val, 4)
                row['ssim'] = round(ssim_val, 4)
            else:
                row['psnr'] = None
                row['ssim'] = None

            # Error ratio
            if original is not None and gt_kernel is not None:
                try:
                    o_gray = (original[:, :, 0] if original.ndim == 3
                              else original).astype(np.float64) / 255.0
                    b_gray = (blurred[:, :, 0] if blurred.ndim == 3
                              else blurred).astype(np.float64) / 255.0
                    gt_k = gt_kernel.astype(np.float64)
                    if gt_k.ndim > 2:
                        gt_k = gt_k[:, :, 0]
                    gt_k = gt_k / (gt_k.sum() + 1e-12)
                    est_k = est_kernel.astype(np.float64)
                    if est_k.ndim > 2:
                        est_k = est_k[:, :, 0]
                    est_k = est_k / (est_k.sum() + 1e-12)
                    row['error_ratio'] = round(
                        _process_worker._error_ratio_nonblind(
                            o_gray, b_gray, gt_k, est_k), 4)
                except Exception:
                    row['error_ratio'] = None
            else:
                row['error_ratio'] = None

            all_results.append(row)

            if counter % 10 == 0 or counter == total:
                print(f"    [{counter}/{total}] {GRID_PARAM_1}={v1}, "
                      f"{GRID_PARAM_2}={v2} → PSNR={row.get('psnr', '?')}, "
                      f"SSIM={row.get('ssim', '?')}, t={elapsed:.1f}с")

    # Сохранение
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = (grid_results_dir /
                    f"grid_{GRID_PARAM_1}_{GRID_PARAM_2}_{ALGORITHM_LABEL}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[ФАЗА 3 OK] Гиперпараметрическая сетка: {csv_path}")

        # Лучшие результаты
        valid = df.dropna(subset=['psnr'])
        if not valid.empty:
            best_psnr = valid.loc[valid['psnr'].idxmax()]
            best_ssim = valid.loc[valid['ssim'].idxmax()]
            print(f"\n  Лучший PSNR: {best_psnr['psnr']:.2f} при "
                  f"{GRID_PARAM_1}={best_psnr[GRID_PARAM_1]}, "
                  f"{GRID_PARAM_2}={best_psnr[GRID_PARAM_2]}")
            print(f"  Лучший SSIM: {best_ssim['ssim']:.4f} при "
                  f"{GRID_PARAM_1}={best_ssim[GRID_PARAM_1]}, "
                  f"{GRID_PARAM_2}={best_ssim[GRID_PARAM_2]}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f"Полный прогон алгоритма {ALGORITHM_LABEL}")
    parser.add_argument(
        '--phase', type=int, default=0,
        help='Номер фазы: 1=датасеты, 2=итерации, 3=гиперпараметры, 0=все')
    args = parser.parse_args()

    t_start = time.time()

    if args.phase in (0, 1):
        run_phase1_datasets()

    if args.phase in (0, 2):
        run_phase2_iterations()

    if args.phase in (0, 3):
        run_phase3_hyperparam_grid()

    elapsed_total = time.time() - t_start
    hours = int(elapsed_total // 3600)
    mins = int((elapsed_total % 3600) // 60)
    print(f"\n{'=' * 80}")
    print(f"[ГОТОВО] Общее время: {hours}ч {mins}мин")
    print(f"Результаты: {ALG_RESULTS_DIR}")


if __name__ == "__main__":
    main()

#--phase 0|1|2|3 - (0 - все)