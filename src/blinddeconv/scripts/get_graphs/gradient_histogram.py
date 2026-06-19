"""
Построение гистограмм градиентов по набору данных.

Как строятся:
    Все градиенты из всех изображений категории объединяются в один массив,
    затем нормируются в вероятность. Это даёт эмпирическое маргинальное
    распределение градиентов для данной категории, независимо от числа
    изображений в каждой группе.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple



IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def collect_images(directory: Path, extensions: tuple = IMAGE_EXTENSIONS) -> List[Path]:
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(directory.glob(f'*{ext}'))
        paths.extend(directory.glob(f'*{ext.upper()}'))
    return sorted(set(paths))


def collect_images_recursive(directory: Path, extensions: tuple = IMAGE_EXTENSIONS) -> List[Path]:
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(directory.rglob(f'*{ext}'))
        paths.extend(directory.rglob(f'*{ext.upper()}'))
    return sorted(set(paths))



def compute_pooled_gradient_hist(
    image_paths: List[Path],
    bins: np.ndarray,
    grad_directions: str = 'both',
) -> np.ndarray:
    """
    Вычисляет объединенную гистограмму градиентов для заданного набора изображений.

    Значения градиентов из всех изображений объединяются в единый одномерный массив,
    после чего рассчитывается нормализованная вероятностная гистограмма. Данный подход
    позволяет получить эмпирическое маргинальное распределение градиентов для всего
    набора данных, независимо от количества изображений в нём.

    Параметры:
    image_paths : list of Path
        Список путей к изображениям (цветные изображения конвертируются в полутоновые).
    bins : np.ndarray
        Массив границ интервалов гистограммы (размерность: n_bins + 1).
    grad_directions : {'h', 'v', 'both'}
        Направления конечно-разностных градиентов, учитываемые при расчете:
        'h'    — горизонтальные разности (np.diff по оси 1);
        'v'    — вертикальные разности (np.diff по оси 0);
        'both' — объединение градиентов по обоим направлениям.

    Возвращает:
    prob : np.ndarray, форма (n_bins,)
        Нормализованная вероятностная гистограмма (массив вероятностей).
    """
    if not image_paths:
        raise ValueError("image_paths is empty")
    if grad_directions not in ('h', 'v', 'both'):
        raise ValueError("grad_directions must be 'h', 'v', or 'both'")

    chunks: List[np.ndarray] = []
    n_loaded = 0
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [warn] Could not load image: {path}")
            continue
        img = img.astype(np.float64)
        if grad_directions in ('h', 'both'):
            chunks.append(np.diff(img, axis=1).ravel())
        if grad_directions in ('v', 'both'):
            chunks.append(np.diff(img, axis=0).ravel())
        n_loaded += 1

    if not chunks:
        raise RuntimeError("No valid images could be loaded")

    all_grads = np.concatenate(chunks)
    hist, _ = np.histogram(all_grads, bins=bins)
    total = hist.sum()
    prob = hist / total if total > 0 else hist.astype(float)
    return prob


def plot_gradient_histograms(
    series: List[Dict],
    output_path: Path,
    title: str,
    bin_range: Tuple[float, float] = (-150.0, 150.0),
    n_bins: int = 300,
    ylim: Tuple[float, float] = (-20.0, 0.0),
    figsize: Tuple[float, float] = (8, 6),
    grad_directions: str = 'both',
    dpi: int = 150,
) -> None:
    """
    Вычисляет объединенные гистограммы градиентов для нескольких наборов данных (серий)
    и сохраняет результаты в виде единого графика.

    Параметры:
    series : list of dict
        Список словарей, каждый из которых описывает одну кривую на графике:
            'label'     : str   — подпись для легенды;
            'color'     : str   — цвет линии;
            'paths'     : list  — список путей (объекты Path) к изображениям;
            'linewidth' : float — толщина линии (опционально, по умолчанию 2.5);
            'linestyle' : str   — стиль линии (опционально, по умолчанию '-');
            'zorder'    : int   — порядок (слой) отрисовки Z-index (опционально).
    output_path : Path
        Полный путь для сохранения выходного файла с графиком.
    title : str
        Заголовок графика.
    bin_range : tuple[float, float], по умолчанию (-150.0, 150.0)
        Минимальное и максимальное значения градиента, определяющие диапазон гистограммы.
    n_bins : int, по умолчанию 300
        Количество интервалов (корзин) гистограммы.
    ylim : tuple[float, float], по умолчанию (-20.0, 0.0)
        Пределы оси ординат (y) для логарифмической плотности вероятности.
    figsize : tuple[float, float], по умолчанию (8, 6)
        Размер фигуры Matplotlib (в дюймах).
    grad_directions : {'h', 'v', 'both'}, по умолчанию 'both'
        Направления вычисляемых градиентов ('h' - горизонтальное, 'v' - вертикальное, 'both' - оба).
    dpi : int, по умолчанию 150
        Разрешение сохраняемого изображения (точек на дюйм).
    """
    bins = np.linspace(bin_range[0], bin_range[1], n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=figsize)

    for s in series:
        prob = compute_pooled_gradient_hist(s['paths'], bins, grad_directions)
        log_prob = np.log(prob + 1e-10)
        ax.plot(
            bin_centers,
            log_prob,
            color=s['color'],
            linewidth=s.get('linewidth', 2.5),
            linestyle=s.get('linestyle', '-'),
            zorder=s.get('zorder', 2),
            label=s['label'],
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(r'Модуль градиента яркости $|\nabla x|$ (яркость/пиксель)', fontsize=12)
    ax.set_ylabel(r'Логарифм плотности вероятности $\ln\, p(|\nabla x|)$', fontsize=12)
    ax.set_xlim(bin_range)
    ax.set_ylim(ylim)
    ax.grid(True, linestyle='--', alpha=0.4)

    legend = ax.legend(fontsize=11, loc='upper right', frameon=True, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")
