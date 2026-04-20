#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для генерации графиков и TeX-таблиц по всем сохраненным CSV.
Автоматически исключает Complexity_Test из графиков метрик качества.

Запуск из консоли:
python generate_visuals.py
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Автоопределение корня
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

import visualisation as vis

DATASETS_ROOT = PROJECT_ROOT / "images" / "compare_data"
RESULTS_ROOT = PROJECT_ROOT / "presentation_graphics"
TEX_DIR = RESULTS_ROOT / "comparison_tex"
FIG_DIR = RESULTS_ROOT / "comparison_figures"
TEX_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Какие датасеты мы ИГНОРИРУЕМ при оценке КАЧЕСТВА (PSNR/SSIM/ER)
EXCLUDE_FROM_QUALITY = ["Complexity_Test", "Grid_Test"]

def main():
    print("Сканирование результатов...")
    all_data = {}

    # 1. Загрузка всех данных
    for alg_dir in sorted(RESULTS_ROOT.iterdir()):
        if not alg_dir.is_dir() or alg_dir.name.startswith("comparison"):
            continue

        alg_name = alg_dir.name
        csvs = list(alg_dir.glob("all_results_*.csv"))
        if not csvs:
            csvs = list(alg_dir.glob("*/results_*.csv"))
        
        frames =[]
        for csv_file in csvs:
            try:
                df = pd.read_csv(csv_file)
                if 'algorithm' not in df.columns:
                    df['algorithm'] = alg_name
                frames.append(df)
            except Exception as e:
                print(f"  Ошибка чтения {csv_file}: {e}")

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            if 'distorted_file' in combined.columns and 'dataset' in combined.columns:
                combined = combined.drop_duplicates(subset=['distorted_file', 'dataset'], keep='last')
            all_data[alg_name] = combined
            print(f"✓ {alg_name}: {len(combined)} записей")

    if not all_data:
        print("Нет данных для анализа!")
        return

    df_global = pd.concat(all_data.values(), ignore_index=True)
    all_datasets = df_global['dataset'].dropna().unique().tolist()
    
    # 2. Фильтруем датасеты (убираем Complexity для качества)
    quality_datasets =[d for d in all_datasets if d not in EXCLUDE_FROM_QUALITY]
    df_global_quality = df_global[df_global['dataset'].isin(quality_datasets)].copy()

    print(f"\nВсе датасеты: {all_datasets}")
    print(f"Датасеты для качества: {quality_datasets}")

    # =========================================================================
    # ЧАСТЬ I: ГРАФИКИ ДЛЯ КАЖДОГО АЛГОРИТМА ОТДЕЛЬНО
    # =========================================================================
    print("\nГенерация одиночных графиков...")
    for alg_name, df_alg in all_data.items():
        alg_fig_dir = RESULTS_ROOT / alg_name / "figures"
        alg_tex_dir = RESULTS_ROOT / alg_name / "tex"
        prof_dir    = RESULTS_ROOT / alg_name / "kernel_profiles"
        alg_fig_dir.mkdir(exist_ok=True); alg_tex_dir.mkdir(exist_ok=True); prof_dir.mkdir(exist_ok=True)

        # Отсекаем Complexity_Test
        df_alg_qual = df_alg[df_alg['dataset'].isin(quality_datasets)]

        if not df_alg_qual.empty:
            vis.plot_boxplots_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
            vis.plot_error_ratio_histogram_single(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
            vis.plot_success_rate_single(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
            vis.plot_psnr_ssim_bars_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
            vis.plot_psnr_ssim_per_image_all_datasets(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
            vis.plot_kernel_size_dependency_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)

            # Генерация профилей ядер (1D срезы) по датасетам
            for ds in quality_datasets:
                df_ds = df_alg_qual[df_alg_qual['dataset'] == ds]
                if not df_ds.empty:
                    dist_dir = DATASETS_ROOT / ds / "distorted"
                    vis.save_kernel_profiles_and_diff(df_ds, dist_dir, prof_dir, alg_name)
            
            # Суммарная таблица (преобразуем формат под функцию)
            ds_results_list = [{'dataset': d, 'df': df_alg_qual[df_alg_qual['dataset'] == d]} for d in quality_datasets]
            vis.build_summary_single(ds_results_list, alg_name, tex_dir=alg_tex_dir)

    # =========================================================================
    # ЧАСТЬ II: СРАВНИТЕЛЬНЫЕ ГРАФИКИ (КАЧЕСТВО)
    # =========================================================================
    print("\nГенерация сравнительных графиков качества...")
    # Фильтруем словари для передачи в функции
    all_data_quality = {alg: df[df['dataset'].isin(quality_datasets)] for alg, df in all_data.items()}

    vis.plot_success_rate_comparison(all_data_quality, quality_datasets, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    vis.plot_bar_psnr_ssim_comparison(all_data_quality, quality_datasets, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    vis.plot_error_ratio_histogram_comparison(all_data_quality, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    vis.plot_kernel_size_dependency_comparison(df_global_quality, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    vis.plot_boxplots_comparison(all_data_quality, quality_datasets, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    vis.plot_noise_dependency(df_global_quality, fig_dir=FIG_DIR, tex_dir=TEX_DIR)
    
    vis.build_table_mean_psnr_ssim(all_data_quality, quality_datasets, tex_dir=TEX_DIR)
    vis.build_table_full_quantitative(all_data_quality, quality_datasets, results_root=RESULTS_ROOT, tex_dir=TEX_DIR)

    # =========================================================================
    # ЧАСТЬ III: ГРАФИКИ ПРОИЗВОДИТЕЛЬНОСТИ И 3D КАРТЫ (Grid_Test + Complexity)
    # =========================================================================
    print("\nГенерация графиков производительности (3D / Pareto / Scalability)...")
    PERF_FIG_DIR = RESULTS_ROOT / "performance_figures"
    PERF_FIG_DIR.mkdir(exist_ok=True)

    grid_name = "Grid_Test" if "Grid_Test" in all_datasets else all_datasets[0]
    comp_name = "Complexity_Test" if "Complexity_Test" in all_datasets else None

    # Рабочие зоны SSIM
    vis.plot_2d_working_areas(all_data, grid_dataset_name=grid_name, metric='ssim', fig_dir=PERF_FIG_DIR)
    vis.plot_3d_applicability_4_angles(all_data, grid_dataset_name=grid_name, complexity_dataset_name=comp_name, metric='ssim', fig_dir=PERF_FIG_DIR)

    # Рабочие зоны PSNR
    vis.plot_2d_working_areas(all_data, grid_dataset_name=grid_name, metric='psnr', fig_dir=PERF_FIG_DIR)
    vis.plot_3d_applicability_4_angles(all_data, grid_dataset_name=grid_name, complexity_dataset_name=comp_name, metric='psnr', fig_dir=PERF_FIG_DIR)

    # Trade-off (Парето) - если есть Levin
    if "Levin" in all_datasets:
        vis.plot_time_quality_pareto(all_data, dataset_name="Levin", fig_dir=PERF_FIG_DIR)

    # Complexity
    if comp_name:
        vis.plot_scalability_comparison(all_data, complexity_dataset_name=comp_name, fig_dir=PERF_FIG_DIR)
    else:
        print("  Пропуск Scalability: 'Complexity_Test' не найден.")

    # =========================================================================
    # ЧАСТЬ IV: ИТЕРАЦИОННЫЕ ГРАФИКИ (log_test)
    # =========================================================================
    print("\nГенерация итерационных графиков (сходимость, эволюция ядра)...")
    ITER_DATASET_NAME = "log_test"

    for alg_name in all_data:
        iter_dir = RESULTS_ROOT / alg_name / ITER_DATASET_NAME
        if not iter_dir.exists():
            continue

        iter_fig_dir = RESULTS_ROOT / alg_name / "figures"
        iter_tex_dir = RESULTS_ROOT / alg_name / "tex"
        iter_fig_dir.mkdir(exist_ok=True)
        iter_tex_dir.mkdir(exist_ok=True)

        print(f"  [{alg_name}] Итерационные графики из {iter_dir.name}...")
        vis.plot_iteration_convergence(iter_dir, alg_name,
                                       fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)
        vis.plot_kernel_evolution_strip(iter_dir, alg_name,
                                        fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)

    # =========================================================================
    # ЧАСТЬ V: ГИПЕРПАРАМЕТРИЧЕСКИЕ ГРАФИКИ (тепловые карты + чувствительность)
    # =========================================================================
    print("\nГенерация гиперпараметрических графиков (тепловые карты, чувствительность)...")

    for alg_name in all_data:
        grid_dir = RESULTS_ROOT / alg_name / "hyperparam_grid"
        if not grid_dir.exists():
            continue

        grid_csvs = list(grid_dir.glob("grid_*.csv"))
        if not grid_csvs:
            continue

        hp_fig_dir = RESULTS_ROOT / alg_name / "figures"
        hp_tex_dir = RESULTS_ROOT / alg_name / "tex"
        hp_fig_dir.mkdir(exist_ok=True)
        hp_tex_dir.mkdir(exist_ok=True)

        for csv_file in grid_csvs:
            print(f"  [{alg_name}] Гиперпараметры: {csv_file.name}...")
            vis.plot_hyperparam_heatmap(csv_file, alg_name,
                                        fig_dir=hp_fig_dir, tex_dir=hp_tex_dir)
            vis.plot_hyperparam_sensitivity_1d(csv_file, alg_name,
                                               fig_dir=hp_fig_dir, tex_dir=hp_tex_dir)

    print("\n[OK] Все графики успешно сгенерированы!")

if __name__ == "__main__":
    main()