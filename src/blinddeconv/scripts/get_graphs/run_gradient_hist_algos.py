"""
Строит гистограммы градиентов (в логарифмическом масштабе)
для нескольких датасетов и нескольких алгоритмов из разных папок результатов.

Каждый алгоритм — отдельная кривая.

На каждый датасет создаётся:
  1. {dataset}_orig_vs_distorted.png       — оригиналы + искажённые
  2. {dataset}_{algo_dir}.png              — оригиналы + искажённые + один алгоритм
  3. {dataset}_all_algos.png               — оригиналы + искажённые + все алгоритмы

Результаты сохраняются в BASE / gradient_hist_algos / {dataset_name} /

Запуск:
    python run_gradient_hist_algos.py
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from gradient_histogram import collect_images, plot_gradient_histograms

# Пути
BASE = Path(r"D:\for_proga\franework_deconvolution\framework (9)")
LABELS_PATH = BASE / 'presentation_labels.json'

# Папки с исходными изображениями (originals / distorted)
IMAGES_ROOT = BASE / 'images/compare_data/kostya'

# Папки с результатами алгоритмов
RESULTS_ROOTS = [
    BASE / 'presentation_graphics',
    BASE / 'presentation_graphics_kostya_FINALLY',
    BASE / 'presentation_graphics_pasha_FINALLY',
]

# Куда сохранять гистограммы
OUTPUT_ROOT = BASE / 'gradient_hist_algos'

# Обрабатываемые датасеты
ALLOWED_DATASETS = {'Levin', 'Kohler', 'Sun', 'Set12'}

# Цвета для кривых (фиксированные, одинаковые на всех графиках)
COLOR_ORIG      = '#E62020'   # — оригиналы
COLOR_DISTORTED = '#66C244'   # — искажённые
COLOR_RESTORED  = '#2176AE'   # — восстановленное


# Настройки гистограммы

HIST_SETTINGS = dict(
    bin_range       = (-150.0, 150.0),
    n_bins          = 300,
    ylim            = (-15.0, 0.0),
    figsize         = (8, 6),
    grad_directions = 'both',
    dpi             = 150,
)


HIST_SETTINGS_ALL = dict(HIST_SETTINGS, figsize=(11, 7))

def load_labels(json_path: Path) -> dict:
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def resolve_label(name: str, labels: dict) -> str:
    """Директория/ключ - название через presentation_labels."""
    if name in labels:
        return labels[name]
    candidate = name.replace('_', ' ')
    if candidate in labels:
        return labels[candidate]
    return candidate


def make_algo_colors(n: int) -> List[str]:
    """
    Возвращает n различимых цветов для линий алгоритмов.
    """
    cmap = matplotlib.colormaps.get_cmap('tab20').resampled(max(n, 1))
    return [matplotlib.colors.to_hex(cmap(i)) for i in range(n)]


def collect_algos_for_dataset(
    results_roots: List[Path],
    dataset_name: str,
) -> Dict[str, Dict[str, List[Path]]]:
    found: Dict[str, Dict[str, List[Path]]] = {}
    for results_root in results_roots:
        if not results_root.is_dir():
            print(f"  [warn] Results root not found: {results_root}")
            continue
        for algo_dir in sorted(results_root.iterdir()):
            if not algo_dir.is_dir():
                continue
            restored_dir = algo_dir / dataset_name / 'restored'
            if not restored_dir.is_dir():
                continue
            imgs = collect_images(restored_dir)
            if not imgs:
                continue
            root_name = results_root.name
            found.setdefault(root_name, {})[algo_dir.name] = imgs
            print(f"    [{root_name} / {algo_dir.name}] {dataset_name}: {len(imgs)} restored")
    return found

def run_dataset(dataset_name: str, labels: dict) -> None:
    orig_dir = IMAGES_ROOT / dataset_name / 'originals'
    dist_dir = IMAGES_ROOT / dataset_name / 'distorted'

    orig_paths = collect_images(orig_dir) if orig_dir.is_dir() else []
    dist_paths = collect_images(dist_dir) if dist_dir.is_dir() else []

    if not orig_paths:
        print(f"  [skip] No originals for {dataset_name}")
        return
    if not dist_paths:
        print(f"  [skip] No distorted for {dataset_name}")
        return

    dist_clean_paths = [p for p in dist_paths if p.stem.endswith('_clean')]
    dist_noisy_paths = [p for p in dist_paths if not p.stem.endswith('_clean')]

    algos_by_root = collect_algos_for_dataset(RESULTS_ROOTS, dataset_name)
    if not algos_by_root:
        print(f"  [warn] No restored images found for {dataset_name} — only orig/dist plot.")

    out_dir = OUTPUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    s_orig = dict(label='Оригинал',   color=COLOR_ORIG,      paths=orig_paths)
    s_dist = dict(label='Искажённое', color=COLOR_DISTORTED, paths=dist_paths)

    print(f"\n  Generating plots → {out_dir}")

    # График 1: Оригинал + Искажённое
    plot_gradient_histograms(
        series=[s_orig, s_dist],
        output_path=out_dir / f'{dataset_name}_orig_vs_distorted.png',
        title='Гистограмма градиентов: Оригинал vs Искажённое',
        **HIST_SETTINGS,
    )

    # График 1б: Оригинал + Размытое без шума (_clean) + Искажённое (шум+смаз)
    if dist_clean_paths and dist_noisy_paths:
        s_clean = dict(label='Размытое',                color=COLOR_RESTORED,  paths=dist_clean_paths)
        s_noisy = dict(label='Размытое и зашумлённое', color=COLOR_DISTORTED, paths=dist_noisy_paths)
        plot_gradient_histograms(
            series=[s_orig, s_clean, s_noisy],
            output_path=out_dir / f'{dataset_name}_orig_clean_noisy.png',
            title='',
            **HIST_SETTINGS,
        )

    if not algos_by_root:
        return

    # Графики 2–4: по каждой папке результатов
    for root_name, algos in algos_by_root.items():
        colors = make_algo_colors(len(algos))

        # 2а. По одному на алгоритм (оригинал + искажённое + восстановленное)
        for algo_name, rest_paths in algos.items():
            algo_label = resolve_label(algo_name, labels)
            s_rest = dict(label='Восстановленное', color=COLOR_RESTORED, paths=rest_paths)
            plot_gradient_histograms(
                series=[s_orig, s_dist, s_rest],
                output_path=out_dir / f'{dataset_name}_{algo_name}.png',
                title=f'Гистограмма градиентов: {algo_label}',
                **HIST_SETTINGS,
            )

        # 2б. Все алгоритмы из этой папки на одном графике
        series_root = [s_orig, s_dist]
        for (algo_name, rest_paths), color in zip(algos.items(), colors):
            series_root.append(dict(
                label=resolve_label(algo_name, labels),
                color=color,
                paths=rest_paths,
            ))
        plot_gradient_histograms(
            series=series_root,
            output_path=out_dir / f'{dataset_name}_{root_name}_all.png',
            title='Гистограмма градиентов: сравнение алгоритмов',
            **HIST_SETTINGS_ALL,
        )

        # 2в. Только не-Base алгоритмы из этой папки
        algos_no_base = {k: v for k, v in algos.items() if not k.endswith('_(Base)')}
        if algos_no_base:
            colors_nb = make_algo_colors(len(algos_no_base))
            series_no_base = [s_orig, s_dist]
            for (algo_name, rest_paths), color in zip(algos_no_base.items(), colors_nb):
                series_no_base.append(dict(
                    label=resolve_label(algo_name, labels),
                    color=color,
                    paths=rest_paths,
                ))
            plot_gradient_histograms(
                series=series_no_base,
                output_path=out_dir / f'{dataset_name}_{root_name}_no_base.png',
                title='Гистограмма градиентов: сравнение алгоритмов',
                **HIST_SETTINGS_ALL,
            )

    # График 5: Все алгоритмы из всех папок на одном
    all_algos_flat = {}
    for algos in algos_by_root.values():
        for algo_name, rest_paths in algos.items():
            key = algo_name
            if key in all_algos_flat:
                for root_name, algos2 in algos_by_root.items():
                    if algo_name in algos2 and algos2[algo_name] is rest_paths:
                        key = f"{algo_name}__{root_name}"
                        break
            all_algos_flat[key] = rest_paths

    colors_all = make_algo_colors(len(all_algos_flat))
    series_all = [s_orig, s_dist]
    for (algo_name, rest_paths), color in zip(all_algos_flat.items(), colors_all):
        series_all.append(dict(
            label=resolve_label(algo_name, labels),
            color=color,
            paths=rest_paths,
        ))
    plot_gradient_histograms(
        series=series_all,
        output_path=out_dir / f'{dataset_name}_all_algos.png',
        title='Гистограмма градиентов: все алгоритмы',
        **HIST_SETTINGS_ALL,
    )


def main() -> None:
    labels = load_labels(LABELS_PATH)
    for dataset_name in sorted(ALLOWED_DATASETS):
        print(f"\n=== Dataset: {dataset_name} ===")
        run_dataset(dataset_name, labels)
    print("\nВсё готово.")


if __name__ == '__main__':
    main()
