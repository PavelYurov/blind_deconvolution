#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скрипт для генерации графиков и TeX-таблиц по всем сохраненным CSV.
Автоматически исключает Complexity_Test/Grid_Test из графиков метрик качества.

Запуск из консоли:
    python generate_visuals.py                # все фазы
    python generate_visuals.py --phase 0      # все фазы (явно)
    python generate_visuals.py --phase 1      # одиночные графики (per-algorithm)
    python generate_visuals.py --phase 2      # сравнительные графики качества
    python generate_visuals.py --phase 3      # производительность / 3D-карты
    python generate_visuals.py --phase 4      # итерационные графики
    python generate_visuals.py --phase 5      # гиперпараметры
    python generate_visuals.py --skip-kernel-profiles   # фаза 1 без профилей ядер (слишком долгая генерация всех)
    python generate_visuals.py --force-kernel-profiles  # фаза 1 перезапись профилей ядер (по стандарту не перезаписывает для ускорения процесса)
    python generate_visuals.py --noise-only   # только перезаписать графики зависимости от шума

Все пути в CSV переписываются под текущий PROJECT_ROOT,
так что результаты, полученные на другой машине, корректно отображаются здесь.

Подписи (имена алгоритмов / датасетов / шумов) на графиках можно
переопределить через файл presentation_labels.json
"""

import sys
import os
import json
import argparse
import matplotlib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

import visualisation as vis

DATASETS_ROOT = PROJECT_ROOT / "images" / "compare_data"
RESULTS_ROOT = PROJECT_ROOT / "presentation_graphics"
TEX_DIR = RESULTS_ROOT / "comparison_tex"
FIG_DIR = RESULTS_ROOT / "comparison_figures"
LABELS_CONFIG = PROJECT_ROOT / "presentation_labels.json"
TEX_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Какие датасеты мы ИГНОРИРУЕМ при оценке КАЧЕСТВА (PSNR/SSIM/ER)
EXCLUDE_FROM_QUALITY = ["Complexity_Test", "Grid_Test"]

# Колонки CSV, в которых лежат пути к файлам — их нужно перебазировать
# относительно текущего PROJECT_ROOT, чтобы графики строились корректно
# на любой машине, куда скопированы CSV-файлы.
PATH_COLUMNS = [
    "distorted_path",
    "original_path",
    "restored_path",
    "kernel_path",
    "gt_kernel_path",
]

# Якорные подстроки внутри пути — после первого совпадения всё, что идёт
# слева, отбрасывается, а PROJECT_ROOT приклеивается слева.
_PATH_ANCHORS = (
    "images/compare_data/",
    "images\\compare_data\\",
    "presentation_graphics/",
    "presentation_graphics\\",
)


def _rebase_path(p) -> str:
    if p is None:
        return p
    try:
        s = str(p)
    except Exception:
        return p
    if not s or s == "nan":
        return s
    s_norm = s.replace("\\", "/")
    for anchor in _PATH_ANCHORS:
        a = anchor.replace("\\", "/")
        idx = s_norm.find(a)
        if idx >= 0:
            tail = s_norm[idx:]
            return str(PROJECT_ROOT / tail)
    return s


def _rewrite_paths_inplace(df: pd.DataFrame) -> None:
    for col in PATH_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_rebase_path)


def _load_label_map() -> dict:
    """Загружает {fs_name: display_label} из presentation_labels.json (если есть)."""
    if not LABELS_CONFIG.exists():
        try:
            LABELS_CONFIG.write_text(
                json.dumps({
                    "_comment": (
                        "Подмены подписей для графиков. Ключ — слово/имя из "
                        "файловой системы (директория алгоритма, имя датасета, "
                        "тип шума). Значение — текст, который нужно показать на "
                        "графике. Если ключа нет, имя остаётся как есть."
                    ),
                    "_examples": {
                        "pmp_denoise": "PMP",
                        "dcp_with_denoiser": "DCP",
                        "Set12": "Set12",
                        "gaussian": "Гауссов",
                        "poisson": "Пуассонов",
                        "impulse": "Импульсный",
                        "pink": "Розовый",
                        "brown": "Коричневый"
                    }
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  Создан шаблон конфига подписей: {LABELS_CONFIG.name}")
        except Exception as e:
            print(f"  Не удалось создать шаблон {LABELS_CONFIG}: {e}")
        return {}
    try:
        raw = json.loads(LABELS_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        # Игнорируем служебные ключи, начинающиеся с '_'
        return {k: v for k, v in raw.items()
                if isinstance(k, str) and not k.startswith("_")
                and isinstance(v, str)}
    except Exception as e:
        print(f"  Ошибка чтения {LABELS_CONFIG}: {e}")
        return {}


def _parse_args():
    p = argparse.ArgumentParser(description="Генерация презентационных графиков.")
    p.add_argument(
        "--phase", type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
        help=(
            "0 — все фазы (по умолчанию); "
            "1 — одиночные графики per-algorithm; "
            "2 — сравнительные графики качества; "
            "3 — производительность и 3D-карты; "
            "4 — итерационные графики (сходимость, эволюция ядра); "
            "5 — гиперпараметры (heatmap + sensitivity)."
        ),
    )
    p.add_argument(
        "--skip-kernel-profiles", action="store_true",
        help="Не генерировать профили ядер. "
             "По умолчанию save_kernel_profiles_and_diff пропускает уже существующие PNG."
    )
    p.add_argument(
        "--noise-only", action="store_true",
        help="Перезаписать ТОЛЬКО графики зависимости от шума "
             "(noise_dependency_*, noise_level_line_*, noise_bsnr_*). "
             "Все остальные фазы и графики не запускаются."
    )
    p.add_argument(
        "--force-kernel-profiles", action="store_true",
        help="Принудительно перезаписать все профили ядер, игнорируя уже "
             "существующие файлы. Без этого флага существующие профили пропускаются."
    )
    return p.parse_args()


def _load_all_data():
    all_data = {}
    for alg_dir in sorted(RESULTS_ROOT.iterdir()):
        if not alg_dir.is_dir() or alg_dir.name.startswith("comparison") \
                or alg_dir.name == "performance_figures":
            continue

        alg_name = alg_dir.name
        csvs = list(alg_dir.glob("all_results_*.csv"))
        if not csvs:
            csvs = list(alg_dir.glob("*/results_*.csv"))

        frames = []
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
                combined = combined.drop_duplicates(
                    subset=['distorted_file', 'dataset'], keep='last')
            _rewrite_paths_inplace(combined)
            all_data[alg_name] = combined
            print(f"✓ {alg_name}: {len(combined)} записей")
    return all_data


def _get_noise_variants(all_data_quality: dict, dataset_name: str) -> list:
    """Возвращает список noise_name для датасета:
    'clean' + один представитель на каждый тип шума.

    Поддерживает форматы:
      - '{weak|medium|strong}{type}'  - берёт 'medium{type}'
      - '{type}_{float}'              - берёт средний по значению уровень
      - прочие                        - берёт средний по алфавиту
    """
    from collections import defaultdict
    all_noise: set = set()
    for df in all_data_quality.values():
        sub = df[df['dataset'] == dataset_name]
        all_noise.update(sub['noise_name'].dropna().unique())

    variants = ['clean'] if 'clean' in all_noise else []

    _LEVELS = ('weak', 'medium', 'strong')
    level_groups: dict = defaultdict(list)
    float_groups: dict = defaultdict(list)
    misc: list = []

    for n in sorted(all_noise):
        if n == 'clean':
            continue
        matched = False
        for lvl in _LEVELS:
            if n.startswith(lvl):
                noise_type = n[len(lvl):]
                if noise_type:
                    level_groups[noise_type].append(n)
                    matched = True
                    break
        if matched:
            continue
        parts = n.rsplit('_', 1)
        if len(parts) == 2:
            try:
                float(parts[1])
                float_groups[parts[0]].append(n)
                continue
            except ValueError:
                pass
        misc.append(n)

    for noise_type, names in sorted(level_groups.items()):
        preferred = f"medium{noise_type}"
        variants.append(preferred if preferred in names
                        else sorted(names)[len(names) // 2])

    for noise_type, names in sorted(float_groups.items()):
        variants.append(sorted(names)[len(names) // 2])

    for n in misc:
        variants.append(n)

    return variants


# =============================================================================
#  ФАЗЫ
# =============================================================================

_PHASE1_FONT_SCALE = 1.3
_PHASE1_RC = {
    'font.size':        round(10 * _PHASE1_FONT_SCALE),
    'axes.titlesize':   round(12 * _PHASE1_FONT_SCALE),
    'axes.labelsize':   round(10 * _PHASE1_FONT_SCALE),
    'xtick.labelsize':  round(10 * _PHASE1_FONT_SCALE),
    'ytick.labelsize':  round(10 * _PHASE1_FONT_SCALE),
    'legend.fontsize':  round(10 * _PHASE1_FONT_SCALE),
    'legend.title_fontsize': round(10 * _PHASE1_FONT_SCALE),
}


def phase1_per_algorithm(all_data, quality_datasets, skip_kernel_profiles=False, force_kernel_profiles=False):
    print("\n[ФАЗА 1] Одиночные графики per-algorithm...")
    _old_rc = {k: matplotlib.rcParams[k] for k in _PHASE1_RC}
    matplotlib.rcParams.update(_PHASE1_RC)
    _old_title_fs = vis.TITLE_FONTSIZE
    vis.TITLE_FONTSIZE = round(_old_title_fs * _PHASE1_FONT_SCALE)
    try:
        _phase1_body(all_data, quality_datasets, skip_kernel_profiles, force_kernel_profiles)
    finally:
        matplotlib.rcParams.update(_old_rc)
        vis.TITLE_FONTSIZE = _old_title_fs


def _phase1_body(all_data, quality_datasets, skip_kernel_profiles=False, force_kernel_profiles=False):
    for alg_name, df_alg in all_data.items():
        alg_fig_dir = RESULTS_ROOT / alg_name / "figures"
        alg_tex_dir = RESULTS_ROOT / alg_name / "tex"
        prof_dir    = RESULTS_ROOT / alg_name / "kernel_profiles"
        alg_fig_dir.mkdir(exist_ok=True)
        alg_tex_dir.mkdir(exist_ok=True)
        prof_dir.mkdir(exist_ok=True)

        df_alg_qual = df_alg[df_alg['dataset'].isin(quality_datasets)]
        if df_alg_qual.empty:
            continue

        vis.plot_boxplots_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_error_ratio_histogram_single(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_success_rate_single(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_psnr_ssim_bars_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_psnr_ssim_per_image_all_datasets(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_kernel_size_dependency_single(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)

        vis.plot_error_ratio_histogram_single_v2(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_success_rate_single_v2(df_alg_qual['error_ratio'], alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_psnr_ssim_bars_single_v2(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)
        vis.plot_psnr_ssim_per_image_per_dataset_v2(df_alg_qual, alg_name, fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)

        for ds in quality_datasets:
            df_ds = df_alg_qual[df_alg_qual['dataset'] == ds]
            if not df_ds.empty:
                dist_dir = DATASETS_ROOT / ds / "distorted"
                if skip_kernel_profiles:
                    pass  # пропуск по флагу
                else:
                    vis.save_kernel_profiles_and_diff(
                        df_ds, dist_dir, prof_dir, alg_name,
                        skip_existing=not force_kernel_profiles,
                    )

        # Визуальное сравнение (оригинал / смазанное / восстановленное)
        for ds in quality_datasets:
            vis.plot_visual_comparison_per_algorithm(
                alg_name, df_alg, ds,
                fig_dir=alg_fig_dir / "visual_comparison",
            )

        # Лучший / худший случай по PSNR и SSIM
        best_worst_dir = alg_tex_dir
        for ds in quality_datasets:
            for metric in ('psnr', 'ssim'):
                vis.plot_best_worst_comparison(
                    alg_name, df_alg, ds, metric=metric,
                    fig_dir=best_worst_dir,
                )

        ds_results_list = [{'dataset': d, 'df': df_alg_qual[df_alg_qual['dataset'] == d]}
                           for d in quality_datasets]
        vis.build_summary_single(ds_results_list, alg_name, tex_dir=alg_tex_dir)


def _split_groups(all_data: dict) -> dict:
    """Разбивает all_data на три набора по признаку '(Base)' в имени папки:

    * no_base — только модифицированные методы (без '(Base)')
    * base   — только базовые методы ('(Base)')
    * all    — все

    Возвращает {имя_группы: dict[alg, df]} (возможно с пустыми группами).
    """
    no_base = {a: df for a, df in all_data.items() if "(Base)" not in a}
    base = {a: df for a, df in all_data.items() if "(Base)" in a}
    return {"no_base": no_base, "base": base, "all": all_data}


def phase_noise_only(all_data, quality_datasets):
    """Перезаписывает ТОЛЬКО графики зависимости от шума, во всех 3 группах."""
    print("\n[NOISE-ONLY] Перегенерация графиков зависимости от шума...")
    groups = _split_groups(all_data)
    for group_name, group_data in groups.items():
        if not group_data:
            print(f"  [{group_name}] пусто — пропуск")
            continue
        fig_g = FIG_DIR / group_name
        tex_g = TEX_DIR / group_name
        fig_g.mkdir(parents=True, exist_ok=True)
        tex_g.mkdir(parents=True, exist_ok=True)
        all_data_quality = {alg: df[df['dataset'].isin(quality_datasets)]
                            for alg, df in group_data.items()}
        df_global_g = pd.concat(all_data_quality.values(), ignore_index=True) \
                        if all_data_quality else pd.DataFrame()
        print(f"  -- группа [{group_name}]: {len(group_data)} алгоритмов --")
        vis.plot_noise_dependency(df_global_g, fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_noise_dependency_delta(df_global_g, fig_dir=fig_g, tex_dir=tex_g)


def phase2_comparison_quality(all_data, quality_datasets, df_global_quality):
    print("\n[ФАЗА 2] Сравнительные графики качества (3 группы: no_base / base / all)...")
    _old_rc = {k: matplotlib.rcParams[k] for k in _PHASE1_RC}
    matplotlib.rcParams.update(_PHASE1_RC)
    _old_title_fs = vis.TITLE_FONTSIZE
    vis.TITLE_FONTSIZE = round(_old_title_fs * _PHASE1_FONT_SCALE)
    try:
        _phase2_body(all_data, quality_datasets, df_global_quality)
    finally:
        matplotlib.rcParams.update(_old_rc)
        vis.TITLE_FONTSIZE = _old_title_fs


def _phase2_body(all_data, quality_datasets, df_global_quality):
    groups = _split_groups(all_data)

    for group_name, group_data in groups.items():
        if not group_data:
            print(f"  [{group_name}] пусто — пропуск")
            continue

        fig_g = FIG_DIR / group_name
        tex_g = TEX_DIR / group_name
        fig_g.mkdir(parents=True, exist_ok=True)
        tex_g.mkdir(parents=True, exist_ok=True)

        all_data_quality = {alg: df[df['dataset'].isin(quality_datasets)]
                            for alg, df in group_data.items()}
        df_global_g = pd.concat(all_data_quality.values(), ignore_index=True) \
                        if all_data_quality else pd.DataFrame()

        print(f"\n  -- группа [{group_name}]: {len(group_data)} алгоритмов --")

        vis.plot_success_rate_comparison(all_data_quality, quality_datasets,
                                         fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_bar_psnr_ssim_comparison(all_data_quality, quality_datasets,
                                          fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_error_ratio_histogram_comparison(all_data_quality,
                                                  fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_error_ratio_histogram_comparison_sorted(all_data_quality,
                                                         fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_kernel_size_dependency_comparison(df_global_g,
                                                   fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_boxplots_comparison(all_data_quality, quality_datasets,
                                     fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_noise_dependency(df_global_g, fig_dir=fig_g, tex_dir=tex_g)
        vis.plot_noise_dependency_delta(df_global_g, fig_dir=fig_g, tex_dir=tex_g)

        vis.build_table_mean_psnr_ssim(all_data_quality, quality_datasets,
                                       tex_dir=tex_g)
        vis.build_table_full_quantitative(all_data_quality, quality_datasets,
                                          results_root=fig_g.parent,
                                          tex_dir=tex_g)

        #Визуальное сравнение: лучший алгоритм (маленькая таблица)
        for ds in quality_datasets:
            vis.plot_visual_comparison_best_algorithm(
                all_data_quality, ds,
                fig_dir=fig_g / "visual_comparison_best",
                metric='psnr',
            )
            vis.plot_visual_comparison_best_mean_algorithm(
                all_data_quality, ds,
                fig_dir=fig_g / "visual_comparison_best_mean",
                metric='psnr',
            )

        #Большие таблицы
        for ds in quality_datasets:
            vis.plot_big_comparison_vertical(
                all_data_quality, ds, 'clean',
                fig_dir=fig_g / "big_comparison",
            )
            vis.plot_big_comparison_horizontal(
                all_data_quality, ds, 'clean',
                fig_dir=fig_g / "big_comparison",
            )

        # Визуальное сравнение робастности к шуму
        _all_noise: set = set()
        for _df in all_data_quality.values():
            if 'noise_name' in _df.columns:
                _all_noise.update(_df['noise_name'].dropna().unique())
        _all_noise.discard('clean')
        for _noise_nm in sorted(_all_noise):
            for ds in quality_datasets:
                vis.plot_noise_visual_comparison(
                    all_data_quality, ds, _noise_nm,
                    fig_dir=fig_g,
                )


def phase3_performance(all_data, all_datasets):
    print("\n[ФАЗА 3] Производительность / 3D-карты (3 группы: no_base / base / all)...")
    groups = _split_groups(all_data)

    grid_name = "Grid_Test" if "Grid_Test" in all_datasets else all_datasets[0]
    comp_name = "Complexity_Test" if "Complexity_Test" in all_datasets else None

    PERF_BASE = RESULTS_ROOT / "performance_figures"
    PERF_BASE.mkdir(exist_ok=True)

    if comp_name:
        for alg_name, df_alg in all_data.items():
            alg_fig_dir = RESULTS_ROOT / alg_name / "figures"
            alg_tex_dir = RESULTS_ROOT / alg_name / "tex"
            alg_fig_dir.mkdir(exist_ok=True)
            alg_tex_dir.mkdir(exist_ok=True)
            vis.plot_speed_vs_size_single(df_alg, alg_name,
                                          complexity_dataset_name=comp_name,
                                          fig_dir=alg_fig_dir, tex_dir=alg_tex_dir)

    for group_name, group_data in groups.items():
        if not group_data:
            print(f"  [{group_name}] пусто — пропуск")
            continue

        PERF_FIG_DIR = PERF_BASE / group_name
        PERF_FIG_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\n  -- группа [{group_name}]: {len(group_data)} алгоритмов --")

        vis.plot_2d_working_areas(group_data, grid_dataset_name=grid_name,
                                  metric='ssim', fig_dir=PERF_FIG_DIR)
        vis.plot_3d_applicability_4_angles(group_data, grid_dataset_name=grid_name,
                                           complexity_dataset_name=comp_name,
                                           metric='ssim', fig_dir=PERF_FIG_DIR,
                                           show_labels=(group_name != 'all'))

        vis.plot_2d_working_areas(group_data, grid_dataset_name=grid_name,
                                  metric='psnr', fig_dir=PERF_FIG_DIR)
        vis.plot_3d_applicability_4_angles(group_data, grid_dataset_name=grid_name,
                                           complexity_dataset_name=comp_name,
                                           metric='psnr', fig_dir=PERF_FIG_DIR,
                                           show_labels=(group_name != 'all'))

        if "Levin" in all_datasets:
            vis.plot_time_quality_pareto(group_data, dataset_name="Levin",
                                         fig_dir=PERF_FIG_DIR,
                                         complexity_dataset_name=comp_name)

        if comp_name:
            vis.plot_scalability_comparison(group_data,
                                            complexity_dataset_name=comp_name,
                                            fig_dir=PERF_FIG_DIR)
        else:
            print("  Пропуск Scalability: 'Complexity_Test' не найден.")


def phase4_iterations(all_data):
    print("\n[ФАЗА 4] Итерационные графики (сходимость, эволюция ядра)...")
    _old_rc = {k: matplotlib.rcParams[k] for k in _PHASE1_RC}
    matplotlib.rcParams.update(_PHASE1_RC)
    _old_title_fs = vis.TITLE_FONTSIZE
    vis.TITLE_FONTSIZE = round(_old_title_fs * _PHASE1_FONT_SCALE)
    try:
        _phase4_body(all_data)
    finally:
        matplotlib.rcParams.update(_old_rc)
        vis.TITLE_FONTSIZE = _old_title_fs


def _phase4_body(all_data):
    ITER_DATASET_NAME = "log_test"

    for alg_name in all_data:
        iter_dir = RESULTS_ROOT / alg_name / ITER_DATASET_NAME
        if not iter_dir.exists():
            continue

        iter_fig_dir = RESULTS_ROOT / alg_name / "figures"
        iter_tex_dir = RESULTS_ROOT / alg_name / "tex"
        iter_fig_dir.mkdir(exist_ok=True)
        iter_tex_dir.mkdir(exist_ok=True)

        distorted_dirs = [
            _user_dir / ITER_DATASET_NAME / "distorted"
            for _user_dir in sorted(DATASETS_ROOT.iterdir())
            if _user_dir.is_dir() and (_user_dir / ITER_DATASET_NAME / "distorted").exists()
        ]

        print(f"  [{alg_name}] Итерационные графики из {iter_dir.name}...")
        vis.plot_iteration_convergence(iter_dir, alg_name,
                                       fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)
        vis.plot_kernel_evolution_strip(iter_dir, alg_name,
                                        fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)
        vis.plot_iteration_convergence_v2(iter_dir, alg_name,
                                          fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)
        vis.plot_kernel_evolution_strip_v2(iter_dir, alg_name,
                                           fig_dir=iter_fig_dir, tex_dir=iter_tex_dir,
                                           distorted_dir=distorted_dirs)
        vis.plot_kernel_evolution_strip_v3(iter_dir, alg_name,
                                           fig_dir=iter_fig_dir, tex_dir=iter_tex_dir)


def phase5_hyperparams(all_data):
    print("\n[ФАЗА 5] Гиперпараметры (heatmap + sensitivity)...")
    _old_rc = {k: matplotlib.rcParams[k] for k in _PHASE1_RC}
    matplotlib.rcParams.update(_PHASE1_RC)
    _old_title_fs = vis.TITLE_FONTSIZE
    vis.TITLE_FONTSIZE = round(_old_title_fs * _PHASE1_FONT_SCALE)
    try:
        _phase5_body(all_data)
    finally:
        matplotlib.rcParams.update(_old_rc)
        vis.TITLE_FONTSIZE = _old_title_fs


def _phase5_body(all_data):
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
                                        fig_dir=hp_fig_dir, tex_dir=hp_tex_dir,
                                        all_results_df=all_data.get(alg_name))
            vis.plot_hyperparam_sensitivity_1d(csv_file, alg_name,
                                               fig_dir=hp_fig_dir, tex_dir=hp_tex_dir)
            vis.plot_hyperparam_heatmap_3d(csv_file, alg_name,
                                           fig_dir=hp_fig_dir, tex_dir=hp_tex_dir,
                                           all_results_df=all_data.get(alg_name))
            vis.plot_hyperparam_sensitivity_1d_v2(csv_file, alg_name,
                                                  fig_dir=hp_fig_dir, tex_dir=hp_tex_dir)


def main():
    args = _parse_args()
    phase = args.phase

    # Декодер подписей
    label_map = _load_label_map()
    vis.set_label_map(label_map)
    if label_map:
        print(f"Загружены подписи ({len(label_map)} ключей) из {LABELS_CONFIG.name}")
    else:
        print(f"Карта подписей пуста ({LABELS_CONFIG.name}); подписи без подмен.")

    print("\nСканирование результатов...")
    all_data = _load_all_data()
    if not all_data:
        print("Нет данных для анализа!")
        return

    df_global = pd.concat(all_data.values(), ignore_index=True)
    all_datasets = df_global['dataset'].dropna().unique().tolist()

    quality_datasets = [d for d in all_datasets if d not in EXCLUDE_FROM_QUALITY]
    df_global_quality = df_global[df_global['dataset'].isin(quality_datasets)].copy()

    print(f"\nВсе датасеты: {all_datasets}")
    print(f"Датасеты для качества: {quality_datasets}")
    print(f"Запрошенная фаза: {phase} ({'все' if phase == 0 else f'фаза {phase}'})")

    if args.noise_only:
        phase_noise_only(all_data, quality_datasets)
        return

    run_all = (phase == 0)
    if run_all or phase == 1:
        phase1_per_algorithm(all_data, quality_datasets,
                             skip_kernel_profiles=args.skip_kernel_profiles,
                             force_kernel_profiles=args.force_kernel_profiles)
    if run_all or phase == 2:
        phase2_comparison_quality(all_data, quality_datasets, df_global_quality)
    if run_all or phase == 3:
        phase3_performance(all_data, all_datasets)
    if run_all or phase == 4:
        phase4_iterations(all_data)
    if run_all or phase == 5:
        phase5_hyperparams(all_data)

    print("\n[OK] Готово.")


if __name__ == "__main__":
    main()
