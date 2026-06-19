"""
Строит гистограммы градиентов (в логарифмическом масштабе) для
одного или нескольких датасетов деконволюции.

На каждый датасет создаётся:
  1. {name}_orig_vs_distorted.png      — оригиналы + искажённые
  2. {name}_{algo_dir}.png             — оригиналы + искажённые + один регуляризатор (по одному на регуляризатор)
  3. {name}_all_priors.png             — оригиналы + искажённые + все регуляризаторы (сравнение)

Запуск:
    python run_gradient_hist.py

Результаты сохраняются в папку  <results_root>/gradient_hist/
"""

import json
from pathlib import Path
from typing import Dict, List

from gradient_histogram import collect_images, plot_gradient_histograms


# Цветовая палитра (единая для всех графиков и датасетов)
COLOR_ORIG      = '#E62020'   # — оригиналы
COLOR_DISTORTED = '#66C244'   # — искажённые
COLOR_RESTORED  = '#2176AE'   # — восстановленные (на графике конкретного регуляризатора)

# Палитра для линий восстановленных на совмещённом графике всех регуляризаторов.
PRIOR_PALETTE = [
    '#2176AE',
    '#9B59B6',
    '#E67E22',
    '#1ABC9C',
    '#F39C12',
    '#7F8C8D',
    '#8E44AD',
    '#16A085',
]


# Настройки гистограммы

HIST_SETTINGS = dict(
    bin_range       = (-150.0, 150.0),  # диапазон значений градиентов
    n_bins          = 300,              # число бинов
    ylim            = (-15.0, 0.0),     # ось Y (log вероятность)
    figsize         = (8, 6),
    grad_directions = 'both',           # 'h', 'v' или 'both'
    dpi             = 150,
)


# Конфигурации датасетов


BASE = Path(r"D:\for_proga\franework_deconvolution\framework (9)")

LABELS_PATH = BASE / 'presentation_labels.json'

DATASETS = [
    # Датасет 1: priors
    {
        'name': 'priors',

        # Папки с изображениями
        'originals_dir': BASE / 'images/compare_data/kostya/priors/originals',
        'distorted_dir': BASE / 'images/compare_data/kostya/priors/distorted',

        # Папка с результатами алгоритмов (подпапка для каждого алгоритма)
        'results_root': BASE / 'presentation_graphics_priors',
        # Путь ВНУТРИ каждой папки алгоритма до восстановленных изображений
        'restored_subpath': 'priors/restored',

        # Куда сохранять гистограммы
        'output_dir': BASE / 'presentation_graphics_priors/gradient_hist',
    },

    # Датасет 2: Levin
    {
        'name': 'Levin',

        # Папки с изображениями
        'originals_dir': BASE / 'images/compare_data/kostya/Levin/originals',
        'distorted_dir': BASE / 'images/compare_data/kostya/Levin/distorted',

        # Папка с результатами алгоритмов
        'results_root': BASE / 'presentation_graphics_priors_comp',
        'restored_subpath': 'Levin/restored',

        'output_dir': BASE / 'presentation_graphics_priors_comp/gradient_hist',
    },
]




def load_labels(json_path: Path) -> dict:
    """Загружает presentation_labels.json, игнорируя служебные ключи _*."""
    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def resolve_label(dir_name: str, labels: dict) -> str:
    """
    Переводит имя директории алгоритма в читаемое название регуляризатора.

    Порядок поиска:
      1. Прямое совпадение (dir_name - labels)
      2. Замена '_' на ' ' (Fast_BD_... - Fast BD ...)
    Если ничего не найдено — возвращает dir_name с пробелами.
    """
    if dir_name in labels:
        return labels[dir_name]
    candidate = dir_name.replace('_', ' ')
    if candidate in labels:
        return labels[candidate]
    return candidate


def collect_restored_by_algo(
    results_root: Path,
    restored_subpath: str,
) -> Dict[str, List[Path]]:
    """
    Возвращает {algo_dir_name: [image_paths]} для каждой папки алгоритма.

    Директории без нужного подпути (comparison_figures, gradient_hist и т.п.)
    автоматически пропускаются.
    """
    result: Dict[str, List[Path]] = {}
    for algo_dir in sorted(results_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        restored_dir = algo_dir / restored_subpath
        if restored_dir.is_dir():
            found = collect_images(restored_dir)
            print(f"    [{algo_dir.name}] restored: {len(found)} images")
            result[algo_dir.name] = found
    return result



# Основная логика
def run_dataset(cfg: dict, labels: dict) -> None:
    out_dir: Path = cfg['output_dir']
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_paths = collect_images(Path(cfg['originals_dir']))
    dist_paths = collect_images(Path(cfg['distorted_dir']))
    restored_by_algo = collect_restored_by_algo(
        Path(cfg['results_root']),
        cfg['restored_subpath'],
    )

    name = cfg['name']
    total_restored = sum(len(v) for v in restored_by_algo.values())
    print(f"\n=== Dataset: {name} ===")
    print(f"  Originals : {len(orig_paths)}")
    print(f"  Distorted : {len(dist_paths)}")
    print(f"  Restored  : {total_restored} ({len(restored_by_algo)} algorithms)")

    if not orig_paths:
        print("  [skip] No original images found.")
        return
    if not dist_paths:
        print("  [skip] No distorted images found.")
        return

    s_orig = dict(label='Оригинал',    color=COLOR_ORIG,      paths=orig_paths)
    s_dist = dict(label='Искажённое',  color=COLOR_DISTORTED, paths=dist_paths)

    print(f"\n  Generating plots → {out_dir}")

    # График 1: Оригинал + Искажённое
    plot_gradient_histograms(
        series=[s_orig, s_dist],
        output_path=out_dir / f'{name}_orig_vs_distorted.png',
        title='Гистограмма градиентов: Оригинал vs Искажённое',
        **HIST_SETTINGS,
    )

    if not restored_by_algo:
        print("  [warn] No restored images found - per-prior plots skipped.")
        return

    # График 2: по одному на регуляризатор - Оригинал + Искажённое + Восстановленное(регуляризатор)
    for algo_name, rest_paths in restored_by_algo.items():
        prior_label = resolve_label(algo_name, labels)
        s_rest = dict(label='Восстановленное', color=COLOR_RESTORED, paths=rest_paths)
        plot_gradient_histograms(
            series=[s_orig, s_dist, s_rest],
            output_path=out_dir / f'{name}_{algo_name}.png',
            title=f'Гистограмма градиентов: {prior_label}',
            **HIST_SETTINGS,
        )

    # График 3: Все регуляризаторы на одном - Оригинал + Искажённое + каждый регуляризатор
    series_combined = [s_orig, s_dist]
    for i, (algo_name, rest_paths) in enumerate(restored_by_algo.items()):
        prior_label = resolve_label(algo_name, labels)
        series_combined.append(dict(
            label=prior_label,
            color=PRIOR_PALETTE[i % len(PRIOR_PALETTE)],
            paths=rest_paths,
        ))
    plot_gradient_histograms(
        series=series_combined,
        output_path=out_dir / f'{name}_all_priors.png',
        title='Гистограмма градиентов: сравнение приоров',
        **HIST_SETTINGS,
    )


def main() -> None:
    labels = load_labels(LABELS_PATH)
    for dataset_cfg in DATASETS:
        run_dataset(dataset_cfg, labels)
    print("\nВсё готово.")


if __name__ == '__main__':
    main()
