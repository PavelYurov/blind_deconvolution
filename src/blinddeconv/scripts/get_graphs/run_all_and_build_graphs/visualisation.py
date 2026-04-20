"""
Модуль визуализации результатов слепой деконволюции.

Содержит функции для построения всех графиков и генерации TeX-кода
таблиц и фигур для отчётов и презентаций.

Автор: сгенерировано для фреймворка blind deconvolution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from itertools import cycle
from typing import Optional, Dict, List, Tuple, Any

# ── Палитра цветов ───────────────────────────────────────────────────────────
PALETTE = [
    '#2176AE',  # синий
    '#E05929',  # оранжево-красный
    '#57A773',  # зелёный
    '#B5338A',  # фиолетовый
    '#F2C12E',  # золотой
    '#1B998B',  # бирюзовый
    '#D64045',  # алый
    '#6B4226',  # коричневый
    '#3D5A80',  # тёмно-синий
    '#EE6C4D',  # коралловый
]

TITLE_FONTSIZE = 11

MARKERS =['s', '^', '+', 'D', 'o', 'v', '*']

def _get_marker_cycle():
    return cycle(MARKERS)

def _get_palette_cycle():
    return cycle(PALETTE)


def _colormap_bars(n: int):
    """Возвращает массив из n различных цветов (tab20 colourmap)."""
    cmap = cm.get_cmap('tab20', max(n, 1))
    return [cmap(i) for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
#  Утилиты
# ═══════════════════════════════════════════════════════════════════════════════

def save_tex(filepath: Path, tex_code: str):
    """Сохраняет TeX-код в файл."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tex_code)
    print(f"  TeX сохранён: {filepath}")


def _safe_label(s: str) -> str:
    return s.replace("-", "_").replace(" ", "_")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Доля успешных / отношение ошибки  (Success Rate vs Error Ratio)
# ═══════════════════════════════════════════════════════════════════════════════

# def plot_success_rate_single(
#     er_values: pd.Series,
#     alg_label: str,
#     fig_dir: Optional[Path] = None,
#     tex_dir: Optional[Path] = None,
#     suffix: str = "",
# ):
#     """Кумулятивный график доли успешных для **одного** алгоритма."""
#     er = er_values.dropna()
#     if len(er) == 0:
#         print("  Нет данных отношения ошибки для построения графика.")
#         return

#     x_max = float(np.clip(np.nanpercentile(er, 99) * 1.2, 3.5, 10.0))
#     thresholds = np.arange(1.0, x_max + 0.05, 0.05)
#     sr = [(er <= t).sum() / len(er) * 100 for t in thresholds]

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(thresholds, sr, linewidth=2, color=PALETTE[0], label=alg_label)
#     ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Порог r=3')
#     ax.set_xlabel('Отношение ошибки (порог)')
#     ax.set_ylabel('Доля успешных (%)')
#     ax.set_title(f'Доля успешных / отношение ошибки — {alg_label}', fontsize=TITLE_FONTSIZE)
#     ax.set_xlim(1, x_max)
#     ax.set_ylim(0, 105)
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()

#     fname = f"success_rate{suffix}"
#     if fig_dir:
#         fig.savefig(Path(fig_dir) / f"{fname}.pdf")
#         fig.savefig(Path(fig_dir) / f"{fname}.png")
#     #plt.show()

#     if tex_dir:
#         tex = (
#             r"\begin{figure}[htbp]" "\n"
#             r"\centering" "\n"
#             r"\includegraphics[width=0.8\textwidth]{" + f"figures/{fname}.pdf" + r"}" "\n"
#             r"\caption{Кумулятивный график доли успешных для алгоритма " + alg_label +
#             r". Ось абсцисс --- пороговое значение отношения ошибки, "
#             r"ось ординат --- доля изображений с отношением ошибки ниже порога.}" "\n"
#             r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
#             r"\end{figure}"
#         )
#         save_tex(Path(tex_dir) / f"{fname}.tex", tex)

def plot_success_rate_single(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    suffix: str = "",
):
    """Кумулятивный график доли успешных для **одного** алгоритма."""
    er = er_values.dropna()
    if len(er) == 0:
        print("  Нет данных отношения ошибки для построения графика.")
        return

    x_max = float(np.clip(np.nanpercentile(er, 99) * 1.2, 3.5, 10.0))
    thresholds = np.arange(1.0, x_max + 0.05, 0.05)
    sr = [(er <= t).sum() / len(er) * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    # markevery=20: маркеры только на 1.0, 2.0, 3.0...
    # markerfacecolor='none': внутри маркер прозрачный, цвет рамки = цвет линии
    ax.plot(thresholds, sr, linewidth=2, color=PALETTE[0], label=alg_label, 
            marker='s', markersize=8, markerfacecolor='none', markevery=20)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Порог r=3')
    ax.set_xlabel('Отношение ошибки (порог)')
    ax.set_ylabel('Доля успешных (%)')
    ax.set_title(f'Доля успешных / отношение ошибки — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"success_rate{suffix}"
    if fig_dir:
        fig.savefig(Path(fig_dir) / f"{fname}.pdf")
        fig.savefig(Path(fig_dir) / f"{fname}.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{" + f"figures/{fname}.pdf" + r"}" "\n"
            r"\caption{Кумулятивный график доли успешных для алгоритма " + alg_label +
            r". Ось абсцисс --- пороговое значение отношения ошибки, "
            r"ось ординат --- доля изображений с отношением ошибки ниже порога.}" "\n"
            r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"{fname}.tex", tex)

# def plot_success_rate_comparison(
#     all_data: Dict[str, pd.DataFrame],
#     datasets: List[str],
#     fig_dir: Optional[Path] = None,
#     tex_dir: Optional[Path] = None,
#     fig_prefix: str = "comparison_figures",
# ):
#     """Кумулятивный график доли успешных для **нескольких** алгоритмов."""

#     # По каждому набору данных
#     for ds_name in datasets:
#         # Динамический диапазон по 99-му перцентилю для данного набора
#         _er_ds = pd.concat([
#             df_alg[df_alg['dataset'] == ds_name]['error_ratio'].dropna()
#             for df_alg in all_data.values()
#         ])
#         x_max_ds = float(np.clip(np.nanpercentile(_er_ds, 99) * 1.2, 3.5, 10.0)) if len(_er_ds) > 0 else 6.0
#         thresholds_ds = np.arange(1.0, x_max_ds + 0.05, 0.05)

#         fig, ax = plt.subplots(figsize=(9, 6))
#         colors = _get_palette_cycle()
#         has_any = False

#         for alg_name, df_alg in all_data.items():
#             df_ds = df_alg[df_alg['dataset'] == ds_name]
#             er = df_ds['error_ratio'].dropna()
#             if len(er) == 0:
#                 continue
#             has_any = True
#             sr = [(er <= t).sum() / len(er) * 100 for t in thresholds_ds]
#             ax.plot(thresholds_ds, sr, linewidth=2, color=next(colors), label=alg_name)

#         if has_any:
#             ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Порог r=3')
#             ax.set_xlabel('Отношение ошибки (порог)')
#             ax.set_ylabel('Доля успешных (%)')
#             ax.set_title(f'Доля успешных / отношение ошибки — набор данных {ds_name}',
#                          fontsize=TITLE_FONTSIZE)
#             ax.set_xlim(1, x_max_ds); ax.set_ylim(0, 105)
#             ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
#             plt.tight_layout()
#             if fig_dir:
#                 fig.savefig(Path(fig_dir) / f"success_rate_{ds_name}.pdf")
#                 fig.savefig(Path(fig_dir) / f"success_rate_{ds_name}.png")
#             #plt.show()
#             if tex_dir:
#                 tex = (
#                     r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
#                     r"\includegraphics[width=0.85\textwidth]{"
#                     + f"{fig_prefix}/success_rate_{ds_name}.pdf" + r"}" "\n"
#                     r"\caption{Кумулятивный график доли успешных на наборе данных "
#                     + ds_name + r". Чем левее и выше кривая, тем лучше алгоритм.}" "\n"
#                     r"\label{fig:sr_cmp_" + ds_name + r"}" "\n"
#                     r"\end{figure}"
#                 )
#                 save_tex(Path(tex_dir) / f"success_rate_{ds_name}.tex", tex)
#         else:
#             plt.close(fig)
#             print(f"  Набор данных {ds_name}: нет данных отношения ошибки")

#     # Общий (динамическая обрезка по всем данным)
#     _er_all = pd.concat([df_alg['error_ratio'].dropna() for df_alg in all_data.values()])
#     x_max_all = float(np.clip(np.nanpercentile(_er_all, 99) * 1.2, 3.5, 10.0)) if len(_er_all) > 0 else 6.0
#     thresholds_all = np.arange(1.0, x_max_all + 0.05, 0.05)

#     fig, ax = plt.subplots(figsize=(9, 6))
#     colors = _get_palette_cycle()
#     has_any = False
#     for alg_name, df_alg in all_data.items():
#         er = df_alg['error_ratio'].dropna()
#         if len(er) == 0:
#             continue
#         has_any = True
#         sr = [(er <= t).sum() / len(er) * 100 for t in thresholds_all]
#         ax.plot(thresholds_all, sr, linewidth=2, color=next(colors), label=alg_name)

#     if has_any:
#         ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Порог r=3')
#         ax.set_xlabel('Отношение ошибки (порог)')
#         ax.set_ylabel('Доля успешных (%)')
#         ax.set_title('Доля успешных / отношение ошибки — все наборы данных',
#                      fontsize=TITLE_FONTSIZE)
#         ax.set_xlim(1, x_max_all); ax.set_ylim(0, 105)
#         ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
#         plt.tight_layout()
#         if fig_dir:
#             fig.savefig(Path(fig_dir) / "success_rate_all.pdf")
#             fig.savefig(Path(fig_dir) / "success_rate_all.png")
#         #plt.show()
#         if tex_dir:
#             tex = (
#                 r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
#                 r"\includegraphics[width=0.85\textwidth]{"
#                 + f"{fig_prefix}/success_rate_all.pdf" + r"}" "\n"
#                 r"\caption{Общий кумулятивный график доли успешных по всем наборам данных.}" "\n"
#                 r"\label{fig:sr_cmp_all}" "\n" r"\end{figure}"
#             )
#             save_tex(Path(tex_dir) / "success_rate_all.tex", tex)
#     else:
#         plt.close(fig)
#         print("  Нет данных отношения ошибки ни для одного алгоритма.")


def plot_success_rate_comparison(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Кумулятивный график доли успешных для **нескольких** алгоритмов."""

    # По каждому набору данных
    for ds_name in datasets:
        # Динамический диапазон по 99-му перцентилю для данного набора
        _er_ds = pd.concat([
            df_alg[df_alg['dataset'] == ds_name]['error_ratio'].dropna()
            for df_alg in all_data.values()
        ])
        x_max_ds = float(np.clip(np.nanpercentile(_er_ds, 99) * 1.2, 3.5, 10.0)) if len(_er_ds) > 0 else 6.0
        thresholds_ds = np.arange(1.0, x_max_ds + 0.05, 0.05)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = _get_palette_cycle()
        markers = _get_marker_cycle() # Инициализируем цикл фигурок
        has_any = False

        for alg_name, df_alg in all_data.items():
            df_ds = df_alg[df_alg['dataset'] == ds_name]
            er = df_ds['error_ratio'].dropna()
            if len(er) == 0:
                continue
            has_any = True
            sr =[(er <= t).sum() / len(er) * 100 for t in thresholds_ds]
            
            ax.plot(thresholds_ds, sr, linewidth=2, color=next(colors), label=alg_name,
                    marker=next(markers), markersize=8, markerfacecolor='none', markevery=20)

        if has_any:
            ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Порог r=3')
            ax.set_xlabel('Отношение ошибки (порог)')
            ax.set_ylabel('Доля успешных (%)')
            ax.set_title(f'Доля успешных / отношение ошибки — набор данных {ds_name}',
                         fontsize=TITLE_FONTSIZE)
            ax.set_xlim(1, x_max_ds); ax.set_ylim(0, 105)
            ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if fig_dir:
                fig.savefig(Path(fig_dir) / f"success_rate_{ds_name}.pdf")
                fig.savefig(Path(fig_dir) / f"success_rate_{ds_name}.png")
            #plt.show()
            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/success_rate_{ds_name}.pdf" + r"}" "\n"
                    r"\caption{Кумулятивный график доли успешных на наборе данных "
                    + ds_name + r". Чем левее и выше кривая, тем лучше алгоритм.}" "\n"
                    r"\label{fig:sr_cmp_" + ds_name + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"success_rate_{ds_name}.tex", tex)
        else:
            plt.close(fig)
            print(f"  Набор данных {ds_name}: нет данных отношения ошибки")

    # Общий (динамическая обрезка по всем данным)
    _er_all = pd.concat([df_alg['error_ratio'].dropna() for df_alg in all_data.values()])
    x_max_all = float(np.clip(np.nanpercentile(_er_all, 99) * 1.2, 3.5, 10.0)) if len(_er_all) > 0 else 6.0
    thresholds_all = np.arange(1.0, x_max_all + 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _get_palette_cycle()
    markers = _get_marker_cycle() # Инициализируем цикл фигурок
    has_any = False
    for alg_name, df_alg in all_data.items():
        er = df_alg['error_ratio'].dropna()
        if len(er) == 0:
            continue
        has_any = True
        sr =[(er <= t).sum() / len(er) * 100 for t in thresholds_all]
        
        ax.plot(thresholds_all, sr, linewidth=2, color=next(colors), label=alg_name,
                marker=next(markers), markersize=8, markerfacecolor='none', markevery=20)

    if has_any:
        ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.6, label='Порог r=3')
        ax.set_xlabel('Отношение ошибки (порог)')
        ax.set_ylabel('Доля успешных (%)')
        ax.set_title('Доля успешных / отношение ошибки — все наборы данных',
                     fontsize=TITLE_FONTSIZE)
        ax.set_xlim(1, x_max_all); ax.set_ylim(0, 105)
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if fig_dir:
            fig.savefig(Path(fig_dir) / "success_rate_all.pdf")
            fig.savefig(Path(fig_dir) / "success_rate_all.png")
        #plt.show()
        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=0.85\textwidth]{"
                + f"{fig_prefix}/success_rate_all.pdf" + r"}" "\n"
                r"\caption{Общий кумулятивный график доли успешных по всем наборам данных.}" "\n"
                r"\label{fig:sr_cmp_all}" "\n" r"\end{figure}"
            )
            save_tex(Path(tex_dir) / "success_rate_all.tex", tex)
    else:
        plt.close(fig)
        print("  Нет данных отношения ошибки ни для одного алгоритма.")


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Столбчатая PSNR / SSIM по изображениям (один алгоритм, разноцветные)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_psnr_ssim_bars_single(
    df: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """Столбчатая диаграмма PSNR и SSIM, сгруппированная по оригинальному изображению."""
    df_m = df.dropna(subset=['psnr', 'ssim']).copy()
    if len(df_m) == 0:
        print("  Нет данных PSNR/SSIM для столбчатой диаграммы.")
        return

    # Группируем по имени оригинала (первая часть до '_')
    df_m['_img'] = df_m['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0])
    grp = df_m.groupby('_img').agg(
        psnr_mean=('psnr', 'mean'), ssim_mean=('ssim', 'mean')
    ).reset_index()
    avg = pd.DataFrame([{
        '_img': 'Среднее',
        'psnr_mean': grp['psnr_mean'].mean(),
        'ssim_mean': grp['ssim_mean'].mean(),
    }])
    grp = pd.concat([grp, avg], ignore_index=True)

    n = len(grp)
    colors = _colormap_bars(n)
    colors[-1] = (0.55, 0.55, 0.55, 1.0)  # «Среднее» — серый

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(n * 1.2 + 2, 8), 5))
    x = np.arange(n)

    b1 = ax1.bar(x, grp['psnr_mean'].values, color=colors, alpha=0.88, edgecolor='grey', linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grp['_img'].values, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title(f'PSNR по изображениям — {alg_label}', fontsize=TITLE_FONTSIZE)
    for bar, val in zip(b1, grp['psnr_mean'].values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    b2 = ax2.bar(x, grp['ssim_mean'].values, color=colors, alpha=0.88, edgecolor='grey', linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(grp['_img'].values, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM по изображениям — {alg_label}', fontsize=TITLE_FONTSIZE)
    for bar, val in zip(b2, grp['ssim_mean'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "psnr_ssim_bar.pdf")
        fig.savefig(Path(fig_dir) / "psnr_ssim_bar.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=\textwidth]{figures/psnr_ssim_bar.pdf}" "\n"
            r"\caption{Средние PSNR и SSIM для алгоритма " + alg_label
            + r", сгруппированные по оригинальному изображению.}" "\n"
            r"\label{fig:psnr_ssim_bar_" + _safe_label(alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "psnr_ssim_bar.tex", tex)


def plot_psnr_ssim_per_image_all_datasets(
    df: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    PSNR и SSIM для каждого изображения, сгруппированные по датасетам.
    X — изображения (разделённые по датасетам), Y — метрика.
    Датасеты разделены визуально, каждый датасет — свой цвет.
    """
    df_m = df.dropna(subset=['psnr', 'ssim']).copy()
    if df_m.empty or 'dataset' not in df_m.columns:
        return

    # Имя изображения — первая часть до '_'
    df_m['_img'] = df_m['distorted_file'].apply(lambda x: Path(x).stem.rsplit('_', 1)[0])

    # Средние по (dataset, image)
    grp = df_m.groupby(['dataset', '_img']).agg(
        psnr_mean=('psnr', 'mean'),
        ssim_mean=('ssim', 'mean'),
    ).reset_index()

    datasets = grp['dataset'].unique()
    ds_colors = dict(zip(datasets, _colormap_bars(len(datasets))))

    # Сортируем: сначала по датасету, потом по имени
    grp = grp.sort_values(['dataset', '_img']).reset_index(drop=True)

    n = len(grp)
    if n == 0:
        return

    # Строим label: "img (dataset)" чтобы не путать одинаковые имена
    x_labels = [f"{row['_img']}\n({row['dataset']})" for _, row in grp.iterrows()]
    bar_colors = [ds_colors[row['dataset']] for _, row in grp.iterrows()]

    fig_w = max(n * 0.9 + 3, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 9))

    x = np.arange(n)

    # ── PSNR ─────────────────────────────────────────────────────────────
    b1 = ax1.bar(x, grp['psnr_mean'].values, color=bar_colors, alpha=0.85,
                 edgecolor='grey', linewidth=0.3)
    for bar, val in zip(b1, grp['psnr_mean'].values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=6, rotation=90)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=7)
    ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title(f'PSNR по изображениям (все датасеты) — {alg_label}',
                  fontsize=TITLE_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # ── SSIM ─────────────────────────────────────────────────────────────
    b2 = ax2.bar(x, grp['ssim_mean'].values, color=bar_colors, alpha=0.85,
                 edgecolor='grey', linewidth=0.3)
    for bar, val in zip(b2, grp['ssim_mean'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=7)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM по изображениям (все датасеты) — {alg_label}',
                  fontsize=TITLE_FONTSIZE)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # ── Легенда по датасетам ─────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=ds_colors[ds], edgecolor='grey', label=ds)
                      for ds in datasets]
    ax1.legend(handles=legend_handles, fontsize=7, loc='upper right',
               title='Датасет', title_fontsize=8)

    plt.tight_layout()

    fname = "psnr_ssim_per_image_all"
    if fig_dir:
        fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=\textwidth]{"
            + f"figures/{fname}.pdf" + r"}" "\n"
            r"\caption{PSNR и SSIM для каждого изображения из всех датасетов — "
            + alg_label + r". Цвет столбца соответствует датасету.}" "\n"
            r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"{fname}.tex", tex)


# def plot_bar_psnr_ssim_comparison(
#     all_data: Dict[str, pd.DataFrame],
#     datasets: List[str],
#     fig_dir: Optional[Path] = None,
#     tex_dir: Optional[Path] = None,
#     fig_prefix: str = "comparison_figures",
# ):
#     """Сгруппированная столбчатая диаграмма PSNR/SSIM (X=изображения, группы=алгоритмы)."""
#     for ds_name in datasets:
#         # Алгоритмы с данными для данного набора
#         alg_names = [
#             a for a, df_a in all_data.items()
#             if len(df_a[df_a['dataset'] == ds_name].dropna(subset=['psnr'])) > 0
#         ]
#         if not alg_names:
#             print(f"  Набор данных {ds_name}: нет данных PSNR/SSIM")
#             continue

#         # Уникальные имена оригиналов (union по всем алгоритмам)
#         img_set = set()
#         for a in alg_names:
#             df_ds = all_data[a][all_data[a]['dataset'] == ds_name].dropna(subset=['psnr'])
#             img_set.update(df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0]))
#         image_names = sorted(img_set)
#         x_labels = image_names + ['Среднее']

#         n_img = len(image_names)
#         n_alg = len(alg_names)
#         bar_w = 0.8 / n_alg
#         colors = _colormap_bars(n_alg)
#         x_base = np.arange(n_img + 1)  # +1 для «Среднее»

#         for metric, ylabel, fname_m in [
#             ('psnr', 'Средний PSNR (дБ)', 'psnr'),
#             ('ssim', 'Средний SSIM', 'ssim'),
#         ]:
#             fig, ax = plt.subplots(figsize=(max((n_img + 1) * n_alg * 0.6 + 2, 9), 5))
#             for idx, alg_name in enumerate(alg_names):
#                 df_ds = all_data[alg_name][
#                     all_data[alg_name]['dataset'] == ds_name
#                 ].dropna(subset=[metric]).copy()
#                 df_ds['_img'] = df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0])
#                 grp = df_ds.groupby('_img')[metric].mean()
#                 vals = [grp.get(img, np.nan) for img in image_names]
#                 vals.append(float(np.nanmean(vals)))
#                 ax.bar(
#                     x_base + idx * bar_w, vals, width=bar_w,
#                     color=colors[idx], alpha=0.85, label=alg_name,
#                     edgecolor='grey', linewidth=0.3,
#                 )
#             ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
#             ax.set_xticklabels(x_labels, rotation=30, ha='right')
#             ax.set_ylabel(ylabel)
#             metric_title = 'PSNR' if metric == 'psnr' else 'SSIM'
#             ax.set_title(f'{metric_title} по изображениям — {ds_name}', fontsize=TITLE_FONTSIZE)
#             ax.legend(loc='upper right', fontsize=8)
#             plt.tight_layout()

#             fname = f"bar_{fname_m}_{ds_name}"
#             if fig_dir:
#                 fig.savefig(Path(fig_dir) / f"{fname}.pdf")
#                 fig.savefig(Path(fig_dir) / f"{fname}.png")
#             #plt.show()

#             if tex_dir:
#                 mname = 'PSNR' if metric == 'psnr' else 'SSIM'
#                 tex = (
#                     r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
#                     r"\includegraphics[width=\textwidth]{"
#                     + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
#                     r"\caption{Средние значения " + mname
#                     + r" на наборе данных " + ds_name
#                     + r". Каждая группа столбцов --- одно оригинальное изображение, цвет --- алгоритм.}" "\n"
#                     r"\label{fig:bar_" + fname_m + r"_cmp_" + ds_name + r"}" "\n"
#                     r"\end{figure}"
#                 )
#                 save_tex(Path(tex_dir) / f"{fname}.tex", tex)

def plot_bar_psnr_ssim_comparison(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Сгруппированная столбчатая диаграмма PSNR/SSIM (X=изображения, группы=алгоритмы)."""
    for ds_name in datasets:
        # Алгоритмы с данными для данного набора
        alg_names = [
            a for a, df_a in all_data.items()
            if len(df_a[df_a['dataset'] == ds_name].dropna(subset=['psnr'])) > 0
        ]
        if not alg_names:
            print(f"  Набор данных {ds_name}: нет данных PSNR/SSIM")
            continue

        # Уникальные имена оригиналов (union по всем алгоритмам)
        img_set = set()
        for a in alg_names:
            df_ds = all_data[a][all_data[a]['dataset'] == ds_name].dropna(subset=['psnr'])
            img_set.update(df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0]))
        image_names = sorted(img_set)
        x_labels = image_names + ['Среднее']

        n_img = len(image_names)
        n_alg = len(alg_names)
        bar_w = 0.8 / n_alg
        colors = _colormap_bars(n_alg)
        x_base = np.arange(n_img + 1)  # +1 для «Среднее»

        for metric, ylabel, fname_m in [
            ('psnr', 'Средний PSNR (дБ)', 'psnr'),
            ('ssim', 'Средний SSIM', 'ssim'),
        ]:
            fig, ax = plt.subplots(figsize=(max((n_img + 1) * n_alg * 0.6 + 2, 9), 5))
            for idx, alg_name in enumerate(alg_names):
                df_ds = all_data[alg_name][
                    all_data[alg_name]['dataset'] == ds_name
                ].dropna(subset=[metric]).copy()
                df_ds['_img'] = df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0])
                grp = df_ds.groupby('_img')[metric].mean()
                vals =[grp.get(img, np.nan) for img in image_names]
                vals.append(float(np.nanmean(vals)))
                ax.bar(
                    x_base + idx * bar_w, vals, width=bar_w,
                    # Изменили alpha на 1.0 (плотные цвета) и добавили четкую черную рамку
                    color=colors[idx], alpha=1.0, label=alg_name,
                    edgecolor='black', linewidth=0.7,
                )
            ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
            ax.set_xticklabels(x_labels, rotation=30, ha='right')
            ax.set_ylabel(ylabel)
            metric_title = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(f'{metric_title} по изображениям — {ds_name}', fontsize=TITLE_FONTSIZE)
            
            # Добавлена сетка (как на фото 3)
            ax.grid(axis='y', color='black', alpha=0.3, linestyle='-')
            ax.set_axisbelow(True) # Сетка рисуется "под" столбцами
            
            ax.legend(loc='upper right', fontsize=8)
            plt.tight_layout()

            fname = f"bar_{fname_m}_{ds_name}"
            if fig_dir:
                fig.savefig(Path(fig_dir) / f"{fname}.pdf")
                fig.savefig(Path(fig_dir) / f"{fname}.png")
            #plt.show()

            if tex_dir:
                mname = 'PSNR' if metric == 'psnr' else 'SSIM'
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Средние значения " + mname
                    + r" на наборе данных " + ds_name
                    + r". Каждая группа столбцов --- одно оригинальное изображение, цвет --- алгоритм.}" "\n"
                    r"\label{fig:bar_" + fname_m + r"_cmp_" + ds_name + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. Гистограмма распределения отношения ошибки
# ═══════════════════════════════════════════════════════════════════════════════

def plot_error_ratio_histogram_single(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    er = er_values.dropna()
    if len(er) == 0:
        print("  Нет данных отношения ошибки для гистограммы.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max(er.max(), 5) + 0.5, 0.5)
    ax.hist(er, bins=bins, color=PALETTE[0], edgecolor='black', alpha=0.8)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='Порог r=3')
    ax.set_xlabel('Отношение ошибки')
    ax.set_ylabel('Количество изображений')
    ax.set_title(f'Распределение отношения ошибки — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax.legend()
    plt.tight_layout()

    if fig_dir:
        fig.savefig(Path(fig_dir) / "error_ratio_histogram.pdf")
        fig.savefig(Path(fig_dir) / "error_ratio_histogram.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{figures/error_ratio_histogram.pdf}" "\n"
            r"\caption{Распределение отношения ошибки для алгоритма " + alg_label
            + r". Красная линия --- порог $r=3$, выше которого результаты "
            r"считаются визуально неприемлемыми.}" "\n"
            r"\label{fig:er_hist_" + _safe_label(alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "error_ratio_histogram.tex", tex)


def plot_error_ratio_histogram_comparison(
    all_data: Dict[str, pd.DataFrame],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    er_has = {}
    for a, df_a in all_data.items():
        v = df_a['error_ratio'].dropna()
        if len(v) > 0:
            er_has[a] = v

    if not er_has:
        print("  Нет данных отношения ошибки для сравнительной гистограммы.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    max_er = max(v.max() for v in er_has.values())
    bins = np.arange(0, min(max_er + 0.5, 15), 0.5)
    colors = _get_palette_cycle()
    for alg_name, er_vals in er_has.items():
        ax.hist(er_vals, bins=bins, alpha=0.5, label=alg_name,
                edgecolor='black', linewidth=0.5, color=next(colors))
    ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='Порог r=3')
    ax.set_xlabel('Отношение ошибки')
    ax.set_ylabel('Количество изображений')
    ax.set_title('Распределение отношения ошибки — сравнение алгоритмов',
                 fontsize=TITLE_FONTSIZE)
    ax.legend()
    plt.tight_layout()

    if fig_dir:
        fig.savefig(Path(fig_dir) / "error_ratio_hist_cmp.pdf")
        fig.savefig(Path(fig_dir) / "error_ratio_hist_cmp.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{"
            + f"{fig_prefix}/error_ratio_hist_cmp.pdf" + r"}" "\n"
            r"\caption{Сравнение распределений отношения ошибки для разных алгоритмов. "
            r"Красная линия --- порог $r=3$.}" "\n"
            r"\label{fig:er_hist_cmp}" "\n" r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "error_ratio_hist_cmp.tex", tex)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. Зависимость от шума (PSNR + SSIM)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_noise_dependency(
    df_global: pd.DataFrame,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Строит зависимость PSNR и SSIM от типа шума."""
    if 'noise_name' not in df_global.columns:
        print("  Столбец noise_name отсутствует.")
        return

    df_noise = df_global[df_global['noise_name'].notna() & (df_global['noise_name'] != '')].copy()
    if len(df_noise) == 0:
        print("  Нет информации о шуме в именах файлов.")
        return

    noise_types = df_noise['noise_name'].unique()
    if len(noise_types) < 2:
        print(f"  Только один тип шума ({noise_types[0]}), график не информативен.")
        return

    grouped = df_noise.groupby(['algorithm', 'noise_name']).agg(
        psnr_mean=('psnr', 'mean'),
        ssim_mean=('ssim', 'mean'),
    ).reset_index()

    algorithms = grouped['algorithm'].unique()
    n_alg = len(algorithms)
    n_noise = len(noise_types)
    bar_w = 0.8 / max(n_alg, 1)
    colors = _colormap_bars(n_alg)

    # ── PSNR ──
    fig, ax = plt.subplots(figsize=(max(n_noise * n_alg * 0.7, 8), 5))
    x_base = np.arange(n_noise)
    for idx, alg_name in enumerate(algorithms):
        sub = grouped[grouped['algorithm'] == alg_name].set_index('noise_name')
        vals = [sub.loc[nt, 'psnr_mean'] if nt in sub.index else 0 for nt in noise_types]
        ax.bar(x_base + idx * bar_w, vals, width=bar_w, color=colors[idx],
               alpha=0.85, label=alg_name, edgecolor='grey', linewidth=0.4)
    ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
    ax.set_xticklabels(noise_types, rotation=30, ha='right')
    ax.set_ylabel('Средний PSNR (дБ)')
    ax.set_title('Робастность к шуму — PSNR', fontsize=TITLE_FONTSIZE)
    ax.legend()
    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "noise_dependency_psnr.pdf")
        fig.savefig(Path(fig_dir) / "noise_dependency_psnr.png")
    #plt.show()

    # ── SSIM ──
    fig, ax = plt.subplots(figsize=(max(n_noise * n_alg * 0.7, 8), 5))
    for idx, alg_name in enumerate(algorithms):
        sub = grouped[grouped['algorithm'] == alg_name].set_index('noise_name')
        vals = [sub.loc[nt, 'ssim_mean'] if nt in sub.index else 0 for nt in noise_types]
        ax.bar(x_base + idx * bar_w, vals, width=bar_w, color=colors[idx],
               alpha=0.85, label=alg_name, edgecolor='grey', linewidth=0.4)
    ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
    ax.set_xticklabels(noise_types, rotation=30, ha='right')
    ax.set_ylabel('Средний SSIM')
    ax.set_title('Робастность к шуму — SSIM', fontsize=TITLE_FONTSIZE)
    ax.legend()
    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "noise_dependency_ssim.pdf")
        fig.savefig(Path(fig_dir) / "noise_dependency_ssim.png")
    #plt.show()

    if tex_dir:
        for metric in ['psnr', 'ssim']:
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=0.85\textwidth]{"
                + f"{fig_prefix}/noise_dependency_{metric}.pdf" + r"}" "\n"
                r"\caption{Зависимость среднего " + mname
                + r" от типа шума для разных алгоритмов.}" "\n"
                r"\label{fig:noise_dep_" + metric + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"noise_dependency_{metric}.tex", tex)


# ═══════════════════════════════════════════════════════════════════════════════
#  8. Зависимость от размера ядра
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_kernel_area(s):
    try:
        if isinstance(s, str) and s.startswith('('):
            t = eval(s)
            return t[0] * t[1]
    except Exception:
        pass
    return np.nan


def plot_kernel_size_dependency_single(
    df: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    df_ks = df.dropna(subset=['psnr']).copy()
    if 'kernel_shape' not in df_ks.columns or len(df_ks) == 0:
        return
    df_ks['ks_area'] = df_ks['kernel_shape'].apply(_parse_kernel_area)
    df_ks = df_ks.dropna(subset=['ks_area'])
    if len(df_ks) < 2:
        return

    grouped = df_ks.groupby('ks_area').agg(
        psnr_mean=('psnr', 'mean'), ssim_mean=('ssim', 'mean'), count=('psnr', 'count')
    ).reset_index()

    n = len(grouped)
    colors = _colormap_bars(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(grouped['ks_area'].astype(str), grouped['psnr_mean'], color=colors,
            edgecolor='grey', linewidth=0.4)
    ax1.set_xlabel('Площадь ядра (высота × ширина)')
    ax1.set_ylabel('Средний PSNR (дБ)')
    ax1.set_title(f'PSNR по размеру ядра — {alg_label}', fontsize=TITLE_FONTSIZE)

    ax2.bar(grouped['ks_area'].astype(str), grouped['ssim_mean'], color=colors,
            edgecolor='grey', linewidth=0.4)
    ax2.set_xlabel('Площадь ядра (высота × ширина)')
    ax2.set_ylabel('Средний SSIM')
    ax2.set_title(f'SSIM по размеру ядра — {alg_label}', fontsize=TITLE_FONTSIZE)

    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "kernel_size_dependency.pdf")
        fig.savefig(Path(fig_dir) / "kernel_size_dependency.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=\textwidth]{figures/kernel_size_dependency.pdf}" "\n"
            r"\caption{Зависимость качества восстановления от размера ядра "
            r"для алгоритма " + alg_label + r".}" "\n"
            r"\label{fig:ks_dep_" + _safe_label(alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "kernel_size_dependency.tex", tex)


def plot_kernel_size_dependency_comparison(
    df_global: pd.DataFrame,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    df_w = df_global.dropna(subset=['psnr']).copy()
    if 'kernel_shape' not in df_w.columns or len(df_w) == 0:
        return
    df_w['ks_area'] = df_w['kernel_shape'].apply(_parse_kernel_area)
    df_w = df_w.dropna(subset=['ks_area'])
    if len(df_w) == 0:
        return

    grouped = df_w.groupby(['algorithm', 'ks_area']).agg(
        psnr_mean=('psnr', 'mean')
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _get_palette_cycle()
    for alg_name in grouped['algorithm'].unique():
        sub = grouped[grouped['algorithm'] == alg_name].sort_values('ks_area')
        ax.plot(sub['ks_area'], sub['psnr_mean'], marker='o', linewidth=2,
                color=next(colors), label=alg_name)
    ax.set_xlabel('Площадь ядра (высота × ширина)')
    ax.set_ylabel('Средний PSNR (дБ)')
    ax.set_title('Зависимость PSNR от размера ядра', fontsize=TITLE_FONTSIZE)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if fig_dir:
        fig.savefig(Path(fig_dir) / "kernel_size_psnr_cmp.pdf")
        fig.savefig(Path(fig_dir) / "kernel_size_psnr_cmp.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{"
            + f"{fig_prefix}/kernel_size_psnr_cmp.pdf" + r"}" "\n"
            r"\caption{Зависимость среднего PSNR от размера ядра "
            r"для разных алгоритмов.}" "\n"
            r"\label{fig:ks_psnr_cmp}" "\n" r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "kernel_size_psnr_cmp.tex", tex)


# ═══════════════════════════════════════════════════════════════════════════════
#  9. Боксплоты PSNR / SSIM
# ═══════════════════════════════════════════════════════════════════════════════

# def plot_boxplots_comparison(
#     all_data: Dict[str, pd.DataFrame],
#     datasets: List[str],
#     fig_dir: Optional[Path] = None,
#     tex_dir: Optional[Path] = None,
#     fig_prefix: str = "comparison_figures",
# ):
#     for ds_name in datasets:
#         data_psnr, data_ssim, labels = [], [], []
#         colors = _get_palette_cycle()
#         box_colors = []

#         for alg_name, df_alg in all_data.items():
#             df_ds = df_alg[df_alg['dataset'] == ds_name]
#             p = df_ds['psnr'].dropna()
#             s = df_ds['ssim'].dropna()
#             if len(p) > 0:
#                 data_psnr.append(p.values)
#                 data_ssim.append(s.values)
#                 labels.append(alg_name)
#                 box_colors.append(next(colors))

#         if not labels:
#             continue

#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(len(labels) * 2.5, 8), 5))

#         bp1 = ax1.boxplot(data_psnr, labels=labels, patch_artist=True)
#         for patch, c in zip(bp1['boxes'], box_colors):
#             patch.set_facecolor(c); patch.set_alpha(0.7)
#         ax1.set_ylabel('PSNR (дБ)')
#         ax1.set_title(f'Распределение PSNR — {ds_name}', fontsize=TITLE_FONTSIZE)
#         ax1.tick_params(axis='x', rotation=30)

#         bp2 = ax2.boxplot(data_ssim, labels=labels, patch_artist=True)
#         for patch, c in zip(bp2['boxes'], box_colors):
#             patch.set_facecolor(c); patch.set_alpha(0.7)
#         ax2.set_ylabel('SSIM')
#         ax2.set_title(f'Распределение SSIM — {ds_name}', fontsize=TITLE_FONTSIZE)
#         ax2.tick_params(axis='x', rotation=30)

#         plt.tight_layout()
#         if fig_dir:
#             fig.savefig(Path(fig_dir) / f"boxplot_{ds_name}.pdf")
#             fig.savefig(Path(fig_dir) / f"boxplot_{ds_name}.png")
#         #plt.show()

#         if tex_dir:
#             tex = (
#                 r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
#                 r"\includegraphics[width=\textwidth]{"
#                 + f"{fig_prefix}/boxplot_{ds_name}.pdf" + r"}" "\n"
#                 r"\caption{Боксплоты распределения PSNR и SSIM для разных алгоритмов "
#                 r"на наборе данных " + ds_name + r".}" "\n"
#                 r"\label{fig:boxplot_" + ds_name + r"}" "\n"
#                 r"\end{figure}"
#             )
#             save_tex(Path(tex_dir) / f"boxplot_{ds_name}.tex", tex)


def plot_boxplots_comparison(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    for ds_name in datasets:
        data_psnr, data_ssim, labels = [], [], []
        colors = _get_palette_cycle()
        box_colors = []

        for alg_name, df_alg in all_data.items():
            df_ds = df_alg[df_alg['dataset'] == ds_name]
            p = df_ds['psnr'].dropna()
            s = df_ds['ssim'].dropna()
            if len(p) > 0:
                data_psnr.append(p.values)
                data_ssim.append(s.values)
                labels.append(alg_name)
                box_colors.append(next(colors))

        if not labels:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(len(labels) * 1.5, 8), 5))

        # Функция для применения стиля в духе статьи (изображение 2)
        def style_boxplot(bp, colors):
            for i in range(len(colors)):
                c = colors[i]
                
                # 1. Стилизация самой коробки (прозрачный фон, цветная рамка)
                bp['boxes'][i].set_facecolor('none')
                bp['boxes'][i].set_edgecolor(c)
                bp['boxes'][i].set_linewidth(1.5)
                
                # 2. Стилизация медианы
                bp['medians'][i].set_color(c)
                bp['medians'][i].set_linewidth(1.5)
                
                # 3. Стилизация усов (их по 2 на каждый бокс - верхний и нижний)
                bp['whiskers'][i*2].set_color(c)
                bp['whiskers'][i*2].set_linestyle('--') # Пунктир
                bp['whiskers'][i*2].set_linewidth(1.5)
                bp['whiskers'][i*2 + 1].set_color(c)
                bp['whiskers'][i*2 + 1].set_linestyle('--')
                bp['whiskers'][i*2 + 1].set_linewidth(1.5)
                
                # 4. Стилизация шапок (горизонтальные линии на концах усов)
                bp['caps'][i*2].set_color(c)
                bp['caps'][i*2].set_linewidth(1.5)
                bp['caps'][i*2 + 1].set_color(c)
                bp['caps'][i*2 + 1].set_linewidth(1.5)
                
                # 5. Стилизация выбросов (точки/крестики)
                bp['fliers'][i].set_markeredgecolor(c)
                bp['fliers'][i].set_marker('+') # Крестик как на скрине
                bp['fliers'][i].set_markersize(5)

        # --- График 1: PSNR ---
        # Убираем labels=labels, так как теперь будет легенда
        bp1 = ax1.boxplot(data_psnr, patch_artist=True) 
        style_boxplot(bp1, box_colors)
        
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_xlabel('(a) PSNR', fontsize=12)
        ax1.set_xticks([]) # Убираем подписи на оси X
        ax1.grid(axis='y', linestyle='-', alpha=0.5) # Горизонтальная сетка
        ax1.set_axisbelow(True) # Сетка под графиком

        # --- График 2: SSIM ---
        bp2 = ax2.boxplot(data_ssim, patch_artist=True)
        style_boxplot(bp2, box_colors)
        
        ax2.set_ylabel('SSIM')
        ax2.set_xlabel('(b) SSIM', fontsize=12)
        ax2.set_xticks([]) # Убираем подписи на оси X
        ax2.grid(axis='y', linestyle='-', alpha=0.5)
        ax2.set_axisbelow(True)

        # --- Создание общей легенды ---
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=c, lw=1.5, label=l) for c, l in zip(box_colors, labels)]
        
        # Размещаем легенду в нижнем правом углу (как на скрине 2)
        ax1.legend(handles=legend_elements, loc='lower right', fontsize='small')
        ax2.legend(handles=legend_elements, loc='lower right', fontsize='small')

        plt.tight_layout()
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"boxplot_{ds_name}.pdf")
            fig.savefig(Path(fig_dir) / f"boxplot_{ds_name}.png")
        #plt.show()

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{"
                + f"{fig_prefix}/boxplot_{ds_name}.pdf" + r"}" "\n"
                r"\caption{Боксплоты распределения PSNR и SSIM для разных алгоритмов "
                r"на наборе данных " + ds_name + r".}" "\n"
                r"\label{fig:boxplot_" + ds_name + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"boxplot_{ds_name}.tex", tex)


# ═══════════════════════════════════════════════════════════════════════════════
#  10. Таблица средних PSNR / SSIM по наборам данных
# ═══════════════════════════════════════════════════════════════════════════════

import cv2 as cv

def build_summary_single(
    all_dataset_results: List[Dict],
    alg_label: str,
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Таблица средних метрик по наборам данных для одного алгоритма (с суммарной строкой)."""
    
    def calculate_stats(d: pd.DataFrame, ds_name: str) -> Optional[Dict]:
        if d.empty:
            return None
        
        # --- Поиск худшего и лучшего случая для PSNR ---
        if d['psnr'].notna().any():
            idx_min_psnr = d['psnr'].idxmin()
            idx_max_psnr = d['psnr'].idxmax()
            min_psnr_val = round(d.loc[idx_min_psnr, 'psnr'], 2)
            max_psnr_val = round(d.loc[idx_max_psnr, 'psnr'], 2)
            worst_psnr_case = d.loc[idx_min_psnr, 'distorted_file']
            best_psnr_case = d.loc[idx_max_psnr, 'distorted_file']
        else:
            min_psnr_val = max_psnr_val = '—'
            worst_psnr_case = best_psnr_case = '—'
            
        # --- Поиск худшего и лучшего случая для SSIM ---
        if d['ssim'].notna().any():
            idx_min_ssim = d['ssim'].idxmin()
            idx_max_ssim = d['ssim'].idxmax()
            min_ssim_val = round(d.loc[idx_min_ssim, 'ssim'], 4)
            max_ssim_val = round(d.loc[idx_max_ssim, 'ssim'], 4)
            worst_ssim_case = d.loc[idx_min_ssim, 'distorted_file']
            best_ssim_case = d.loc[idx_max_ssim, 'distorted_file']
        else:
            min_ssim_val = max_ssim_val = '—'
            worst_ssim_case = best_ssim_case = '—'

        # --- Вычисление исходных метрик и дельт ---
        if 'psnr_blurred' in d.columns and d['psnr'].notna().any() and d['psnr_blurred'].notna().any():
            mean_psnr_blur = round(d['psnr_blurred'].mean(), 2)
            mean_delta_psnr = round((d['psnr'] - d['psnr_blurred']).mean(), 2)
            mean_delta_psnr_str = f"{mean_delta_psnr:+.2f}"
        else:
            mean_psnr_blur = mean_delta_psnr_str = '—'

        if 'ssim_blurred' in d.columns and d['ssim'].notna().any() and d['ssim_blurred'].notna().any():
            mean_ssim_blur = round(d['ssim_blurred'].mean(), 4)
            mean_delta_ssim = round((d['ssim'] - d['ssim_blurred']).mean(), 4)
            mean_delta_ssim_str = f"{mean_delta_ssim:+.4f}"
        else:
            mean_ssim_blur = mean_delta_ssim_str = '—'

        # --- Поиск размера изображения ---
        img_size_str = "—"
        for _, r in d.iterrows():
            path_img = r.get('original_path') or r.get('restored_path')
            if path_img and pd.notna(path_img) and Path(path_img).exists():
                im = cv.imread(str(path_img))
                if im is not None:
                    img_size_str = f"{im.shape[0]}x{im.shape[1]}"
                    break

        return {
            'Набор данных': ds_name,
            'Количество искаженных изображений': len(d),
            'Размер изображения': img_size_str,
            'PSNR (исх.)': mean_psnr_blur,
            'Среднее (PSNR)': round(d['psnr'].mean(), 2) if d['psnr'].notna().any() else '—',
            'Δ PSNR': mean_delta_psnr_str,
            'Медиана (PSNR)': round(d['psnr'].median(), 2) if d['psnr'].notna().any() else '—',
            'Максимум (PSNR)': max_psnr_val,
            'Лучший случай (PSNR)': best_psnr_case,
            'Минимум (PSNR)': min_psnr_val,
            'Худший случай (PSNR)': worst_psnr_case,
            'SSIM (исх.)': mean_ssim_blur,
            'Среднее (SSIM)': round(d['ssim'].mean(), 4) if d['ssim'].notna().any() else '—',
            'Δ SSIM': mean_delta_ssim_str,
            'Медиана (SSIM)': round(d['ssim'].median(), 4) if d['ssim'].notna().any() else '—',
            'Максимум (SSIM)': max_ssim_val,
            'Лучший случай (SSIM)': best_ssim_case,
            'Минимум (SSIM)': min_ssim_val,
            'Худший случай (SSIM)': worst_ssim_case,
            'Отн. ошибки': round(d['error_ratio'].mean(), 2) if d['error_ratio'].notna().any() else '—',
            'Время (с)': round(d['time_sec'].mean(), 2),
        }

    rows = []
    all_dfs =[]
    for r in all_dataset_results:
        d = r['df']
        all_dfs.append(d)
        stats = calculate_stats(d, r['dataset'])
        if stats:
            rows.append(stats)
            
    # --- Формирование суммарной строки ---
    if all_dfs:
        d_total = pd.concat(all_dfs, ignore_index=True)
        stats_total = calculate_stats(d_total, "ВСЕ (Суммарно)")
        if stats_total:
            # Чтобы в колонке размера для сборной солянки не писало размер первой картинки:
            stats_total['Размер изображения'] = 'Различные' 
            rows.append(stats_total)

    df_s = pd.DataFrame(rows)

    if tex_dir:
        tex = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Сводные результаты алгоритма " + alg_label
            + r" на тестовых наборах данных (с дельтами, экстремумами и суммарной строкой).}" "\n"
            r"\label{tab:summary_" + _safe_label(alg_label) + r"}" "\n"
            r"\resizebox{\textwidth}{!}{" "\n" 
            r"\begin{tabular}{l c c | c c c c c l c l | c c c c c l c l | c c}" "\n" r"\hline" "\n"
            r"Набор данных & Кол-во & Размер & PSNR (исх) & Среднее (PSNR) & $\Delta$ PSNR & Медиана (PSNR) & Макс. (PSNR) & Лучший случай (PSNR) & Мин. (PSNR) & Худший случай (PSNR) & SSIM (исх) & Среднее (SSIM) & $\Delta$ SSIM & Медиана (SSIM) & Макс. (SSIM) & Лучший случай (SSIM) & Мин. (SSIM) & Худший случай (SSIM) & Отн. ош. & Время \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_s.iterrows():
            def safe_tex(val):
                return str(val).replace('_', r'\_')
            
            # Для итоговой строки добавляем жирный шрифт (опционально)
            prefix = r"\bfseries " if row['Набор данных'] == "ВСЕ (Суммарно)" else ""
            
            tex += (f"{prefix}{safe_tex(row['Набор данных'])} & {prefix}{row['Количество искаженных изображений']} & {prefix}{row['Размер изображения']} & "
                    f"{prefix}{row['PSNR (исх.)']} & {prefix}{row['Среднее (PSNR)']} & {prefix}{row['Δ PSNR']} & {prefix}{row['Медиана (PSNR)']} & "
                    f"{prefix}{row['Максимум (PSNR)']} & {prefix}{safe_tex(row['Лучший случай (PSNR)'])} & "
                    f"{prefix}{row['Минимум (PSNR)']} & {prefix}{safe_tex(row['Худший случай (PSNR)'])} & "
                    f"{prefix}{row['SSIM (исх.)']} & {prefix}{row['Среднее (SSIM)']} & {prefix}{row['Δ SSIM']} & {prefix}{row['Медиана (SSIM)']} & "
                    f"{prefix}{row['Максимум (SSIM)']} & {prefix}{safe_tex(row['Лучший случай (SSIM)'])} & "
                    f"{prefix}{row['Минимум (SSIM)']} & {prefix}{safe_tex(row['Худший случай (SSIM)'])} & "
                    f"{prefix}{row['Отн. ошибки']} & {prefix}{row['Время (с)']} \\\\\n")
        tex += r"\hline" "\n" r"\end{tabular}" "\n" r"}" "\n" r"\end{table}"
        save_tex(Path(tex_dir) / "summary_table.tex", tex)

    return df_s


def build_table_mean_psnr_ssim(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Таблица 10 — средние PSNR / SSIM по наборам данных для разных методов."""
    pivot = []
    for ds_name in datasets:
        row = {'Набор данных': ds_name}
        for alg_name, df_alg in all_data.items():
            df_ds = df_alg[df_alg['dataset'] == ds_name]
            p = df_ds['psnr'].dropna()
            s = df_ds['ssim'].dropna()
            row[f'PSNR_{alg_name}'] = round(p.mean(), 2) if len(p) > 0 else '—'
            row[f'SSIM_{alg_name}'] = round(s.mean(), 4) if len(s) > 0 else '—'
        pivot.append(row)
    df_p = pd.DataFrame(pivot)

    if tex_dir:
        algs = sorted(all_data.keys())
        n = len(algs)
        tex = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Средние значения PSNR (дБ) и SSIM для разных методов "
            r"на тестовых наборах данных.}" "\n"
            r"\label{tab:mean_psnr_ssim}" "\n" r"\small" "\n"
            r"\begin{tabular}{l" + " c c" * n + r"}" "\n" r"\hline" "\n"
            r"\multirow{2}{*}{Набор данных} "
        )
        for a in algs:
            tex += r"& \multicolumn{2}{c}{" + a + "} "
        tex += r" \\" "\n"
        for a in algs:
            tex += r" & PSNR & SSIM"
        tex += r" \\" "\n" r"\hline" "\n"
        for _, row in df_p.iterrows():
            tex += f"{row['Набор данных']}"
            for a in algs:
                tex += f" & {row.get(f'PSNR_{a}', '—')} & {row.get(f'SSIM_{a}', '—')}"
            tex += r" \\" "\n"
        tex += r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table}"
        save_tex(Path(tex_dir) / "table_mean_psnr_ssim.tex", tex)

    return df_p


# ═══════════════════════════════════════════════════════════════════════════════
#  11. Итоговая количественная таблица
# ═══════════════════════════════════════════════════════════════════════════════

def build_table_full_quantitative(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    results_root: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    rows = []
    for ds_name in datasets:
        for alg_name, df_alg in all_data.items():
            df_ds = df_alg[df_alg['dataset'] == ds_name]
            if len(df_ds) == 0:
                continue
            p = df_ds['psnr'].dropna()
            s = df_ds['ssim'].dropna()
            er = df_ds['error_ratio'].dropna()
            t = df_ds['time_sec'].dropna()
            rows.append({
                'Набор данных': ds_name,
                'Алгоритм': alg_name,
                'N': len(df_ds),
                'PSNR (ср.)': round(p.mean(), 2) if len(p) > 0 else '—',
                'PSNR (std)': round(p.std(), 2) if len(p) > 1 else '—',
                'SSIM (ср.)': round(s.mean(), 4) if len(s) > 0 else '—',
                'SSIM (std)': round(s.std(), 4) if len(s) > 1 else '—',
                'Отн. ошибки (ср.)': round(er.mean(), 2) if len(er) > 0 else '—',
                'SR@3 (%)': round((er <= 3).sum() / len(er) * 100, 1) if len(er) > 0 else '—',
                'Время (ср., с)': round(t.mean(), 2) if len(t) > 0 else '—',
            })
    df_f = pd.DataFrame(rows)

    if results_root:
        df_f.to_csv(Path(results_root) / "comparison_summary.csv", index=False)

    if tex_dir:
        tex = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Итоговая сводная таблица количественных метрик "
            r"для всех методов и наборов данных.}" "\n"
            r"\label{tab:full_quantitative}" "\n" r"\small" "\n"
            r"\begin{tabular}{l l c c c c c c c c}" "\n" r"\hline" "\n"
            r"Набор данных & Алгоритм & N & PSNR & $\sigma_{\text{PSNR}}$ & "
            r"SSIM & $\sigma_{\text{SSIM}}$ & Отн.ош. & SR@3 & Время \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_f.iterrows():
            tex += (
                f"{row['Набор данных']} & {row['Алгоритм']} & {row['N']} & "
                f"{row['PSNR (ср.)']} & {row['PSNR (std)']} & "
                f"{row['SSIM (ср.)']} & {row['SSIM (std)']} & "
                f"{row['Отн. ошибки (ср.)']} & {row['SR@3 (%)']} & "
                f"{row['Время (ср., с)']} \\\\\n"
            )
        tex += r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table}"
        save_tex(Path(tex_dir) / "table_full_quantitative.tex", tex)

    return df_f

import math

def _crop_kernel_image(kernel_image: np.ndarray, padding: int = 10) -> np.ndarray:
    if kernel_image is None or kernel_image.size == 0: return kernel_image
    coords = cv.findNonZero(kernel_image)
    if coords is None: return kernel_image
    moments = cv.moments(kernel_image)
    if moments['m00'] == 0: return kernel_image
    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    x, y, w, h = cv.boundingRect(coords)
    radius_x = int(math.ceil(max(cx - x, (x + w) - cx) + padding))
    radius_y = int(math.ceil(max(cy - y, (y + h) - cy) + padding))
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    img_h, img_w = kernel_image.shape[:2]
    src_x1 = max(0, cx_int - radius_x)
    src_y1 = max(0, cy_int - radius_y)
    src_x2 = min(img_w, cx_int + radius_x + 1)
    src_y2 = min(img_h, cy_int + radius_y + 1)
    target_w = 2 * radius_x + 1
    target_h = 2 * radius_y + 1
    dst_x1 = src_x1 - (cx_int - radius_x)
    dst_y1 = src_y1 - (cy_int - radius_y)
    cropped = np.zeros((target_h, target_w), dtype=kernel_image.dtype)
    cropped[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
        kernel_image[src_y1:src_y2, src_x1:src_x2]
    return cropped

def _pad_kernel_to_size(kernel_image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = kernel_image.shape[:2]
    if h >= target_h and w >= target_w: return kernel_image
    padded = np.zeros((target_h, target_w), dtype=kernel_image.dtype)
    y_offset = (target_h - h) // 2
    x_offset = (target_w - w) // 2
    padded[y_offset:y_offset + h, x_offset:x_offset + w] = kernel_image
    return padded

def save_complex_plots(df: pd.DataFrame, dist_dir: Path, save_dir: Path, alg_label: str):
    """
    Генерирует и сохраняет сетку 2x4 со всеми этапами восстановления (как в display.py).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = Path(dist_dir)

    print(f"  Генерация комплексных изображений ({len(df)} шт.). Пожалуйста, подождите...")

    for idx, row in df.iterrows():
        dist_name = row['distorted_file']
        dist_path = dist_dir / dist_name
        orig_path = row.get('original_path', '')
        rest_path = row.get('restored_path', '')
        gt_kernel_path = row.get('gt_kernel_path', '')
        est_kernel_path = row.get('kernel_path', '')

        # Вспомогательная функция для безопасного чтения
        def safe_read(p, is_gray=False):
            if pd.isna(p) or not p or not Path(p).exists(): return None
            return cv.imread(str(p), cv.IMREAD_GRAYSCALE if is_gray else cv.IMREAD_COLOR)

        img_orig = safe_read(orig_path)
        img_dist = safe_read(dist_path)
        img_rest = safe_read(rest_path)
        
        k_gt = safe_read(gt_kernel_path, is_gray=True)
        k_est = safe_read(est_kernel_path, is_gray=True)

        if img_orig is not None: img_orig = cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)
        if img_dist is not None: img_dist = cv.cvtColor(img_dist, cv.COLOR_BGR2RGB)
        if img_rest is not None: img_rest = cv.cvtColor(img_rest, cv.COLOR_BGR2RGB)

        # Центрирование и обрезка ядер
        c_gt = _crop_kernel_image(k_gt) if k_gt is not None else None
        c_est = _crop_kernel_image(k_est) if k_est is not None else None
        
        max_h, max_w = 0, 0
        if c_gt is not None:
            max_h, max_w = max(max_h, c_gt.shape[0]), max(max_w, c_gt.shape[1])
        if c_est is not None:
            max_h, max_w = max(max_h, c_est.shape[0]), max(max_w, c_est.shape[1])
        
        if c_gt is not None: c_gt = _pad_kernel_to_size(c_gt, max_h, max_w)
        if c_est is not None: c_est = _pad_kernel_to_size(c_est, max_h, max_w)

        # Построение сетки
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.2, wspace=0.1)

        for ax in axes.flatten(): ax.axis('off')

        # ── Строка 1: Картинки ──
        if img_orig is not None:
            axes[0, 0].imshow(img_orig)
            axes[0, 0].set_title("Original", fontsize=12)

        if img_dist is not None:
            psnr_b = row.get('psnr_blurred', np.nan)
            ssim_b = row.get('ssim_blurred', np.nan)
            ps_str = f"{psnr_b:.4f}" if pd.notna(psnr_b) else "NaN"
            ss_str = f"{ssim_b:.4f}" if pd.notna(ssim_b) else "NaN"
            
            axes[0, 1].imshow(img_dist)
            axes[0, 1].set_title(f"Distorted\nPSNR: {ps_str} | SSIM: {ss_str}", fontsize=12)
            
            # Preprocessed делаем копией искаженного
            axes[0, 2].imshow(img_dist)
            axes[0, 2].set_title("Preprocessed Image", fontsize=12)

        if img_rest is not None:
            psnr_r = row.get('psnr', np.nan)
            ssim_r = row.get('ssim', np.nan)
            ps_str = f"{psnr_r:.4f}" if pd.notna(psnr_r) else "NaN"
            ss_str = f"{ssim_r:.4f}" if pd.notna(ssim_r) else "NaN"
            
            axes[0, 3].imshow(img_rest)
            axes[0, 3].set_title(f"{alg_label}\nPSNR: {ps_str} | SSIM: {ss_str}", fontsize=12)

        # ── Строка 2: Ядра ──
        if c_gt is not None:
            axes[1, 1].imshow(c_gt, cmap='gray')
            axes[1, 1].set_title("original kernel", fontsize=12)

        if c_est is not None:
            axes[1, 3].imshow(c_est, cmap='gray')
            axes[1, 3].set_title(f"{alg_label} kernel", fontsize=12)

        plt.suptitle(dist_name, y=0.98, fontsize=14)
        plt.tight_layout()
        
        # Сохранение в файл
        fig.savefig(save_dir / f"{Path(dist_name).stem}_complex.png", bbox_inches='tight')
        plt.close(fig)
        
    print(f"  Готово! Сохранено {len(df)} изображений в: {save_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# НОВЫЕ ФУНКЦИИ (БОКСПЛОТЫ И ПРОФИЛИ ЯДЕР)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_boxplots_single(
    df_global: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """9. Боксплоты PSNR / SSIM для одного алгоритма (группировка по датасетам)."""
    if df_global.empty or 'dataset' not in df_global.columns:
        return

    datasets = df_global['dataset'].unique()
    data_psnr, data_ssim, labels = [], [], []

    for ds_name in datasets:
        sub = df_global[df_global['dataset'] == ds_name]
        p = sub['psnr'].dropna()
        s = sub['ssim'].dropna()
        if len(p) > 0:
            data_psnr.append(p.values)
            data_ssim.append(s.values)
            labels.append(ds_name)

    if not labels:
        return

    box_colors = _colormap_bars(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(len(labels) * 2, 8), 5))

    # PSNR
    bp1 = ax1.boxplot(data_psnr, patch_artist=True)
    for box, c in zip(bp1['boxes'], box_colors):
        box.set_facecolor(c); box.set_alpha(0.7); box.set_edgecolor('grey'); box.set_linewidth(1.2)
    for median in bp1['medians']:
        median.set_color('black'); median.set_linewidth(1.5)
    for flier, c in zip(bp1['fliers'], box_colors):
        flier.set(marker='+', color=c, alpha=0.6)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel('PSNR (дБ)')
    ax1.set_title(f'Распределение PSNR — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # SSIM
    bp2 = ax2.boxplot(data_ssim, patch_artist=True)
    for box, c in zip(bp2['boxes'], box_colors):
        box.set_facecolor(c); box.set_alpha(0.7); box.set_edgecolor('grey'); box.set_linewidth(1.2)
    for median in bp2['medians']:
        median.set_color('black'); median.set_linewidth(1.5)
    for flier, c in zip(bp2['fliers'], box_colors):
        flier.set(marker='+', color=c, alpha=0.6)
    ax2.set_xticklabels(labels, rotation=15)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'Распределение SSIM — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "boxplot_single.pdf")
        fig.savefig(Path(fig_dir) / "boxplot_single.png")
    #plt.show()


def save_kernel_profiles_and_diff(
    df: pd.DataFrame, 
    dist_dir: Path, 
    save_dir: Path, 
    alg_label: str
):
    """14. Сравнение истинного и оценённого ядра: 2D разность и 1D профили."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Генерация профилей ядер ({len(df)} шт.). Пожалуйста, подождите...")

    for idx, row in df.iterrows():
        dist_name = row['distorted_file']
        gt_kernel_path = row.get('gt_kernel_path', '')
        est_kernel_path = row.get('kernel_path', '')

        if pd.isna(gt_kernel_path) or not gt_kernel_path: continue
        if pd.isna(est_kernel_path) or not est_kernel_path: continue

        k_gt = cv.imread(str(gt_kernel_path), cv.IMREAD_GRAYSCALE)
        k_est = cv.imread(str(est_kernel_path), cv.IMREAD_GRAYSCALE)
        
        if k_gt is None or k_est is None: continue

        # Центрируем и приводим к одному размеру
        c_gt = _crop_kernel_image(k_gt, padding=5)
        c_est = _crop_kernel_image(k_est, padding=5)
        
        if c_gt is None or c_est is None: continue
        
        max_h = max(c_gt.shape[0], c_est.shape[0])
        max_w = max(c_gt.shape[1], c_est.shape[1])
        
        c_gt = _pad_kernel_to_size(c_gt, max_h, max_w).astype(np.float32)
        c_est = _pad_kernel_to_size(c_est, max_h, max_w).astype(np.float32)

        # Нормализация (сумма = 1), чтобы сравнивать физически корректно
        if c_gt.sum() > 0: c_gt /= c_gt.sum()
        if c_est.sum() > 0: c_est /= c_est.sum()

        diff_map = c_est - c_gt
        
        # 1D Профили через центр
        cy, cx = max_h // 2, max_w // 2
        prof_gt_h, prof_est_h = c_gt[cy, :], c_est[cy, :]
        prof_gt_v, prof_est_v = c_gt[:, cx], c_est[:, cx]

        fig = plt.figure(figsize=(18, 4))
        
        # 1. Истинное ядро
        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(c_gt, cmap='gray')
        ax1.set_title("Истинное ядро")
        ax1.axis('off')

        # 2. Оценённое ядро
        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(c_est, cmap='gray')
        ax2.set_title(f"Оценка ({alg_label})")
        ax2.axis('off')

        # 3. Карта разности
        ax3 = plt.subplot(1, 5, 3)
        vmax = max(abs(diff_map.min()), abs(diff_map.max()))
        im3 = ax3.imshow(diff_map, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax3.set_title("Разность (оценка − истина)")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Горизонтальный профиль
        ax4 = plt.subplot(1, 5, 4)
        ax4.plot(prof_gt_h, label='Истинное', color='black', linestyle='--')
        ax4.plot(prof_est_h, label='Оценка', color='red')
        ax4.set_title("Горизонтальный профиль (центр)")
        ax4.legend(fontsize=8)

        # 5. Вертикальный профиль
        ax5 = plt.subplot(1, 5, 5)
        ax5.plot(prof_gt_v, label='Истинное', color='black', linestyle='--')
        ax5.plot(prof_est_v, label='Оценка', color='red')
        ax5.set_title("Вертикальный профиль (центр)")

        plt.suptitle(f"Анализ ядра: {dist_name}", y=1.05)
        plt.tight_layout()
        fig.savefig(save_dir / f"{Path(dist_name).stem}_kernel_profiles.png", bbox_inches='tight')
        plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════════
#  НОВЫЕ ГРАФИКИ: COMPLEXITY, 3D MAP, PARETO (ДЛЯ СРАВНЕНИЯ АЛГОРИТМОВ)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3d_applicability_map(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    complexity_dataset_name: str = "Complexity_Test",
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures"
):
    """
    3D График (Domain of Applicability): X - Смаз, Y - Шум, Z - Время.
    Размер шарика = Success Rate.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = _get_palette_cycle()

    # Для легенды
    legend_handles =[]
    
    for alg_name, df_alg in all_data.items():
        # 1. Извлекаем данные для X и Y (из Grid_Test)
        df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
        if df_grid.empty:
            continue
            
        # Определяем Успех: (error_ratio < 3.0) ИЛИ (PSNR стал лучше, чем был)
        # Если psnr_blurred нет в таблице, опираемся только на error_ratio
        if 'psnr_blurred' in df_grid.columns:
            df_grid['success'] = (df_grid['error_ratio'] < 3.0) | (df_grid['psnr'] > df_grid['psnr_blurred'])
        else:
            df_grid['success'] = (df_grid['error_ratio'] < 3.0)

        success_rate = df_grid['success'].mean() * 100
        df_success = df_grid[df_grid['success']]
        
        if df_success.empty:
            continue # Алгоритм вообще ничего не смог восстановить

        # Расчет осей X и Y (Сложность).
        # Если в таблице есть ssim_blurred, используем его (1 - ssim_blurred)
        # Если нет, пытаемся достать площадь ядра (ks_area)
        if 'ssim_blurred' in df_success.columns:
            x_vals = 1.0 - df_success['ssim_blurred'].dropna()
            x_coord = np.percentile(x_vals, 90) if len(x_vals) > 0 else 0
        elif 'kernel_shape' in df_success.columns:
            df_success['ks_area'] = df_success['kernel_shape'].apply(_parse_kernel_area)
            x_coord = np.percentile(df_success['ks_area'].dropna(), 90)
        else:
            x_coord = 0

        # Ось Y: Оценка шума. Покажем через деградацию PSNR, если нет специфичных данных
        # (Упрощенно берем 90% перцентиль начального искажения как прокси сложности)
        if 'psnr_blurred' in df_success.columns:
            y_vals = 40.0 - df_success['psnr_blurred'].dropna() # Чем меньше начальный PSNR, тем сложнее
            y_coord = np.percentile(y_vals, 90) if len(y_vals) > 0 else 0
        else:
            y_coord = 0

        # 2. Извлекаем данные для Z (Время из Complexity_Test или среднее)
        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
        if not df_comp.empty:
            z_coord = df_comp['time_sec'].median()
        else:
            z_coord = df_grid['time_sec'].median() # Фолбэк на среднее время

        c = next(colors)
        
        # Строим точку
        scatter = ax.scatter(
            x_coord, y_coord, z_coord, 
            s=max(success_rate * 5, 50), # Размер шарика зависит от SR
            c=[c], alpha=0.8, edgecolors='black', linewidth=1
        )
        
        # Добавляем в легенду (с указанием SR)
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=c, markersize=10, 
                                         label=f"{alg_name} (SR: {success_rate:.1f}%)"))

        # Подпись над шариком
        ax.text(x_coord, y_coord, z_coord + (z_coord * 0.05), alg_name, 
                fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Сложность смаза (X)')
    ax.set_ylabel('Сложность шума (Y)')
    ax.set_zlabel('Время, сек (Z)')
    ax.set_title('Область применимости алгоритмов', fontsize=TITLE_FONTSIZE)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / "3d_applicability_map.pdf")
        fig.savefig(Path(fig_dir) / "3d_applicability_map.png")
    #plt.show()

# def plot_scalability_comparison(
#     all_data: Dict[str, pd.DataFrame],
#     complexity_dataset_name: str = "Complexity_Test",
#     fig_dir: Optional[Path] = None,
#     tex_dir: Optional[Path] = None
# ):
#     """График вычислительной сложности: Время от Мегапикселей."""
#     import cv2 as cv
#     fig, ax = plt.subplots(figsize=(8, 6))
#     colors = _get_palette_cycle()
#     markers = _get_marker_cycle()
    
#     has_data = False
#     for alg_name, df_alg in all_data.items():
#         df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name].copy()
#         if df_comp.empty:
#             continue
            
#         # Пытаемся вычислить мегапиксели
#         megapixels = []
#         times =[]
#         for _, row in df_comp.iterrows():
#             p = row.get('distorted_file') or row.get('original_path')
#             # Заглушка, если путь не полный. В реальности нужно читать файл.
#             # Если в df уже есть img_shape, используйте его.
#             # Здесь предполагается, что разрешение можно вытащить. Для надежности:
#             if 'time_sec' in row and pd.notna(row['time_sec']):
#                 # Если в df нет площади, просто используем индекс или имя файла для сортировки.
#                 # Для корректной работы здесь нужно, чтобы ваш воркер сохранял 'image_area' или 'width'/'height'.
#                 # Сделаем фолбэк на сортировку по размеру файла
#                 megapixels.append(os.path.getsize(row['distorted_path']) / 1024) # KB вместо MP, если нет размеров
#                 times.append(row['time_sec'])
                
#         if megapixels:
#             has_data = True
#             # Сортируем по оси X
#             srt = sorted(zip(megapixels, times))
#             mp_sorted, t_sorted = zip(*srt)
            
#             ax.plot(mp_sorted, t_sorted, marker=next(markers), markersize=8, 
#                     linewidth=2, color=next(colors), label=alg_name)

#     if has_data:
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.set_xlabel('Размер данных (KB / Pixels)')
#         ax.set_ylabel('Время работы, секунды (log scale)')
#         ax.set_title('Масштабируемость алгоритмов (Complexity)', fontsize=TITLE_FONTSIZE)
#         ax.legend()
#         ax.grid(True, which="both", ls="--", alpha=0.5)
        
#         plt.tight_layout()
#         if fig_dir:
#             fig.savefig(Path(fig_dir) / "scalability_plot.pdf")
#             fig.savefig(Path(fig_dir) / "scalability_plot.png")
#         #plt.show()

def plot_time_quality_pareto(
    all_data: Dict[str, pd.DataFrame],
    dataset_name: str = "Grid_Test",
    fig_dir: Optional[Path] = None
):
    """Trade-off: Время vs Качество (Pareto Front)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    points = []
    names = []
    colors_list =[]
    colors_cycle = _get_palette_cycle()
    
    for alg_name, df_alg in all_data.items():
        df_ds = df_alg[df_alg['dataset'] == dataset_name]
        if df_ds.empty:
            continue
            
        mean_ssim = df_ds['ssim'].mean()
        median_time = df_ds['time_sec'].median()
        
        if pd.notna(mean_ssim) and pd.notna(median_time):
            points.append((median_time, mean_ssim))
            names.append(alg_name)
            colors_list.append(next(colors_cycle))
            
    if not points:
        print("Нет данных для Pareto графика")
        return
        
    times, ssims = zip(*points)
    
    # Рисуем точки
    for i in range(len(points)):
        ax.scatter(times[i], ssims[i], color=colors_list[i], s=100, edgecolors='black', label=names[i])
        ax.text(times[i], ssims[i] + 0.01, names[i], fontsize=9, ha='center')
        
    ax.set_xscale('log')
    ax.set_xlabel('Медианное время работы, с (log)')
    ax.set_ylabel('Средний SSIM')
    ax.set_title('Компромисс: Время vs Качество', fontsize=TITLE_FONTSIZE)
    
    # Простейшая отрисовка Парето-фронта (левый верхний угол)
    # Сортируем по времени (от быстрых к долгим)
    sorted_pts = sorted(points, key=lambda x: x[0])
    pareto_front_x =[]
    pareto_front_y =[]
    max_ssim = -1
    
    for t, s in sorted_pts:
        if s > max_ssim:
            pareto_front_x.append(t)
            pareto_front_y.append(s)
            max_ssim = s
            
    ax.plot(pareto_front_x, pareto_front_y, 'r--', linewidth=1.5, alpha=0.7, label='Парето-фронт')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if fig_dir:
        fig.savefig(Path(fig_dir) / "time_quality_pareto.pdf")
        fig.savefig(Path(fig_dir) / "time_quality_pareto.png")
    #plt.show()

def plot_scalability_comparison(
    all_data: Dict[str, pd.DataFrame],
    complexity_dataset_name: str = "Complexity_Test",
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None
):
    """График вычислительной сложности: Время от Мегапикселей."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = _get_palette_cycle()
    markers = _get_marker_cycle()
    
    has_data = False
    for alg_name, df_alg in all_data.items():
        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name].copy()
        if df_comp.empty or 'image_megapixels' not in df_comp.columns:
            continue
            
        # Группируем по размеру картинки и берем среднее время 
        # (вдруг у нас несколько картинок одного разрешения)
        grouped = df_comp.groupby('image_megapixels')['time_sec'].mean().reset_index()
        grouped = grouped.sort_values('image_megapixels')
        
        if not grouped.empty:
            has_data = True
            ax.plot(grouped['image_megapixels'], grouped['time_sec'], 
                    marker=next(markers), markersize=8, linewidth=2, 
                    color=next(colors), label=alg_name)

    if has_data:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Размер изображения (Мегапиксели, log)')
        ax.set_ylabel('Время работы, секунды (log scale)')
        ax.set_title('Масштабируемость алгоритмов', fontsize=TITLE_FONTSIZE)
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        if fig_dir:
            fig.savefig(Path(fig_dir) / "scalability_plot.pdf")
        #plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# ИНТЕРАКТИВНЫЙ 3D-ГРАФИК И 2D КАРТЫ РАБОЧИХ ОБЛАСТЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_xy_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Вычисляет оси X и Y строго по одной метрике.
    Для вычисления "базового" (чистого от шума) смаза использует пути
    из таблицы для оригинального изображения и ядра, выполняя свертку на лету."""
    
    import cv2 as cv
    import numpy as np
    from pathlib import Path
    
    # Импортируем метрики из skimage (стандарт для таких задач)
    try:
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr
        from skimage.metrics import structural_similarity as compare_ssim
    except ImportError:
        print("Внимание: Для вычисления базовых метрик требуется scikit-image (pip install scikit-image)")
        return df

    def calc_ssim_safe(img1, img2):
        # Поддержка старых и новых версий skimage
        try:
            return compare_ssim(img1, img2, channel_axis=-1, data_range=255)
        except TypeError:
            return compare_ssim(img1, img2, multichannel=True, data_range=255)

    df = df.copy()
    
    if 'image_name' not in df.columns:
        df['image_name'] = df['distorted_file']

    def get_clean_ref(row):
        orig_path = row.get('original_path')
        kernel_path = row.get('gt_kernel_path')

        # Если путей нет или они битые, возвращаем NaN
        if pd.isna(orig_path) or pd.isna(kernel_path) or not orig_path or not kernel_path:
            return np.nan
        if not Path(orig_path).exists() or not Path(kernel_path).exists():
            return np.nan

        # Читаем оригинал и ядро
        img_orig = cv.imread(str(orig_path), cv.IMREAD_COLOR)
        kernel = cv.imread(str(kernel_path), cv.IMREAD_GRAYSCALE)

        if img_orig is None or kernel is None:
            return np.nan

        # Нормализуем ядро, чтобы сумма элементов равнялась 1 (иначе изменится яркость)
        kernel = kernel.astype(np.float32)
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        # Применяем идеальный смаз (свертку) к оригинальному изображению
        # Используем float32 для точности, затем обрезаем до 0-255 и переводим в uint8
        img_orig_float = img_orig.astype(np.float32)
        img_blurred_clean = cv.filter2D(img_orig_float, -1, kernel, borderType=cv.BORDER_REPLICATE)
        img_blurred_clean = np.clip(img_blurred_clean, 0, 255).astype(np.uint8)

        # Вычисляем метрику между оригиналом и идеально смазанным (без шума) изображением
        if metric == 'ssim':
            return calc_ssim_safe(img_orig, img_blurred_clean)
        else:
            return compare_psnr(img_orig, img_blurred_clean, data_range=255)

    # Применяем функцию к каждой строке
    df['clean_ref'] = df.apply(get_clean_ref, axis=1)
    
    # На всякий случай, если для каких-то строк не нашлись файлы оригиналов/ядер,
    # заполняем их медианным значением по датасету (чтобы график не падал)
    if df['clean_ref'].isna().any():
        fallback_val = df['clean_ref'].median()
        if pd.isna(fallback_val):
            fallback_val = 1.0 if metric == 'ssim' else 40.0
        df['clean_ref'] = df['clean_ref'].fillna(fallback_val)

    # -------------------------------------------------------------
    # Расчет координат X и Y по вашей формуле
    # -------------------------------------------------------------
    if metric == 'ssim':
        # X: Сложность смаза (насколько чистый смаз убил структуру: от 0 до 1)
        df['X'] = 1.0 - df['clean_ref']
        
        # Y: Влияние шума (ssim_clean - ssim_noisy) / ssim_clean
        df['Y'] = (df['clean_ref'] - df['ssim_blurred']) / (df['clean_ref'] + 1e-6)
        # df['Y'] = (df['clean_ref'] - df['ssim_blurred'])

        
        # Успех
        df['Success'] = (df['error_ratio'] < 3.0) | (df['ssim'] > df['ssim_blurred'])
        
    elif metric == 'psnr':
        # X: Сложность смаза (отталкиваемся от условного "идеала" в 40 дБ)
        df['X'] = 40.0 - df['clean_ref']
        
        # Y: Влияние шума (psnr_clean - psnr_noisy) / psnr_clean
        df['Y'] = (df['clean_ref'] - df['psnr_blurred']) / (df['clean_ref'] + 1e-6)
        # df['Y'] = (df['clean_ref'] - df['psnr_blurred'])

        
        # Успех
        df['Success'] = (df['error_ratio'] < 3.0) | (df['psnr'] > df['psnr_blurred'])

    # Срезаем отрицательные значения Y (иногда добавление шума случайно
    # слегка повышает метрику из-за особенностей вычисления, нам нужен 0)
    df['Y'] = df['Y'].clip(lower=0)

    return df


def plot_2d_working_areas(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    metric: str = "ssim",  # 'ssim' или 'psnr'
    fig_dir: Optional[Path] = None
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    algs_with_data = {name: df[df['dataset'] == grid_dataset_name] 
                      for name, df in all_data.items() 
                      if not df[df['dataset'] == grid_dataset_name].empty}
    
    if not algs_with_data:
        print(f"[2D Area] Нет данных для датасета {grid_dataset_name}")
        return

    n_algs = len(algs_with_data)
    fig, axes = plt.subplots(1, n_algs, figsize=(6 * n_algs, 6), squeeze=False)
    axes = axes.flatten()
    
    x_label = "Сложность смаза (1 - SSIM_blur_clean)" if metric == 'ssim' else "Сложность смаза (40 - PSNR_blur_clean)"
    y_label = "Влияние шума (ΔSSIM / SSIM_blur)" if metric == 'ssim' else "Влияние шума (ΔPSNR / PSNR_blur)"
    
    for i, (alg_name, df_grid) in enumerate(algs_with_data.items()):
        ax = axes[i]
        
        # Считаем X и Y по твоей формуле
        df_grid = _calculate_xy_metrics(df_grid, metric)
        
        success_df = df_grid[df_grid['Success']]
        fail_df = df_grid[~df_grid['Success']]
        
        x_90 = np.percentile(success_df['X'].dropna(), 90) if len(success_df['X'].dropna()) > 0 else 0
        y_90 = np.percentile(success_df['Y'].dropna(), 90) if len(success_df['Y'].dropna()) > 0 else 0
        
        ax.scatter(fail_df['X'], fail_df['Y'], c='red', marker='x', alpha=0.5, label='Провал')
        ax.scatter(success_df['X'], success_df['Y'], c='green', marker='o', alpha=0.8, edgecolor='black', label='Успех')
        
        rect = patches.Rectangle((0, 0), x_90, y_90, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, label='Рабочая зона (90%)')
        ax.add_patch(rect)
        
        ax.set_title(f"Рабочая область: {alg_name} ({metric.upper()})", fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper left')
            
    plt.tight_layout()
    if fig_dir:
        fig.savefig(Path(fig_dir) / f"2d_working_areas_{metric}.pdf")
        fig.savefig(Path(fig_dir) / f"2d_working_areas_{metric}.png")
    #plt.show()


def plot_3d_applicability_map_interactive(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    complexity_dataset_name: str = "Complexity_Test",
    metric: str = "ssim",
    fig_dir: Optional[Path] = None
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("Библиотека plotly не установлена. Запустите: pip install plotly")
        return

    x_data, y_data, z_data = [], [],[]
    sizes, texts, hover_texts, colors_plot = [], [], [], []
    px_colors =['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    x_label = "Сложность смаза (X)"
    y_label = "Сложность шума (Y)"
    
    idx = 0
    for alg_name, df_alg in all_data.items():
        df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
        if df_grid.empty: continue
            
        # Считаем X и Y по твоей формуле
        df_grid = _calculate_xy_metrics(df_grid, metric)

        success_rate = df_grid['Success'].mean() * 100
        df_success = df_grid[df_grid['Success']]
        if df_success.empty: continue

        x_coord = np.percentile(df_success['X'].dropna(), 90) if len(df_success['X'].dropna()) > 0 else 0
        y_coord = np.percentile(df_success['Y'].dropna(), 90) if len(df_success['Y'].dropna()) > 0 else 0

        # Z: Время
        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
        z_coord = df_comp['time_sec'].median() if not df_comp.empty else df_grid['time_sec'].median()

        x_data.append(x_coord)
        y_data.append(y_coord)
        z_data.append(z_coord)
        sizes.append(max(success_rate * 0.4, 15)) 
        texts.append(alg_name)
        
        hover_text = (f"<b>{alg_name}</b><br>"
                      f"Успешность (SR): {success_rate:.1f}%<br>"
                      f"Макс. Смаз (X): {x_coord:.3f}<br>"
                      f"Макс. Шум (Y): {y_coord:.3f}<br>"
                      f"Время (Z): {z_coord:.2f} сек")
        hover_texts.append(hover_text)
        colors_plot.append(px_colors[idx % len(px_colors)])
        idx += 1

    fig = go.Figure(data=[go.Scatter3d(
        x=x_data, y=y_data, z=z_data,
        mode='markers+text',
        text=texts,
        textposition="top center",
        hoverinfo='text',
        hovertext=hover_texts,
        marker=dict(size=sizes, color=colors_plot, line=dict(color='black', width=2), opacity=0.9)
    )])

    fig.update_layout(
        title=f"Интерактивная карта применимости алгоритмов ({metric.upper()})",
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title='Время, сек (Z)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    if fig_dir:
        out_path = Path(fig_dir) / f"3d_applicability_interactive_{metric}.html"
        fig.write_html(str(out_path))
        print(f"  [{metric.upper()}] Интерактивный 3D-график сохранен: {out_path}")
    
    fig.show()



# def plot_3d_applicability_4_angles(
#     all_data: Dict[str, pd.DataFrame],
#     grid_dataset_name: str = "Grid_Test",
#     complexity_dataset_name: str = "Complexity_Test",
#     metric: str = "ssim",
#     fig_dir: Optional[Path] = None
# ):
#     """
#     Строит 4 одинаковых 3D-графика с шариками, но с разных углов обзора (azim), 
#     чтобы избежать перекрытия алгоритмов.
#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     plot_data =[]
#     colors = _get_palette_cycle()
    
#     # 1. Собираем данные (так же, как в прошлых вариантах)
#     for alg_name, df_alg in all_data.items():
#         df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
#         if df_grid.empty: 
#             continue
            
#         df_grid = _calculate_xy_metrics(df_grid, metric)
#         success_rate = df_grid['Success'].mean() * 100
#         df_success = df_grid[df_grid['Success']]
        
#         if df_success.empty: 
#             continue
        
#         x_coord = np.percentile(df_success['X'].dropna(), 90) if len(df_success['X'].dropna()) > 0 else 0
#         y_coord = np.percentile(df_success['Y'].dropna(), 90) if len(df_success['Y'].dropna()) > 0 else 0
        
#         df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
#         z_coord = df_comp['time_sec'].median() if not df_comp.empty else df_grid['time_sec'].median()
        
#         plot_data.append({
#             'alg': alg_name,
#             'x': x_coord,
#             'y': y_coord,
#             'z': z_coord,
#             'sr': success_rate,
#             'color': next(colors)
#         })

#     if not plot_data:
#         print(f"[{metric.upper()}] Нет данных для 3D-графика.")
#         return

#     # 2. Создаем полотно на 4 графика
#     fig = plt.figure(figsize=(16, 14))
    
#     # 4 угла обзора по кругу (высота elev=30, угол поворота azim меняется)
#     angles =[
#         (30, 45),   # Спереди-справа
#         (30, 135),  # Сзади-справа
#         (30, 225),  # Сзади-слева
#         (30, 315)   # Спереди-слева
#     ]
    
#     x_label = "Сложность смаза (X)"
#     y_label = "Влияние шума (Y)"
#     z_label = "Время, сек (Z)"
    
#     legend_handles =[]
    
#     # 3. Рисуем 4 раза
#     for i, (elev, azim) in enumerate(angles):
#         ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
#         for item in plot_data:
#             size = max(item['sr'] * 5, 50)
#             # Рисуем шарик
#             ax.scatter(item['x'], item['y'], item['z'], 
#                        s=size, c=[item['color']], alpha=0.9, edgecolors='black', linewidth=1)
            
#             # Текст подписи (чуть выше шарика)
#             z_offset = item['z'] + (max([d['z'] for d in plot_data]) * 0.05) if plot_data else item['z']
#             ax.text(item['x'], item['y'], z_offset, 
#                     item['alg'], fontsize=9, ha='center', va='bottom')
            
#             # Собираем легенду (только 1 раз)
#             if i == 0:
#                 legend_handles.append(
#                     plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=item['color'], 
#                                markersize=12, label=f"{item['alg']} (SR: {item['sr']:.1f}%)")
#                 )
        
#         # Настраиваем камеру
#         ax.view_init(elev=elev, azim=azim)
#         ax.set_title(f"Поворот {azim}°", fontsize=14, fontweight='bold')
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         ax.set_zlabel(z_label)
    
#     # 4. Оформление
#     fig.suptitle(f"Область применимости алгоритмов ({metric.upper()}) с 4 углов обзора", fontsize=20, y=0.98)
    
#     # Выводим легенду сверху
#     fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.94), 
#                ncol=min(len(plot_data), 4), fontsize=12)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.92], w_pad=3.0, h_pad=3.0)
    
#     if fig_dir:
#         out_path = Path(fig_dir) / f"3d_applicability_4_angles_{metric}.png"
#         fig.savefig(out_path, dpi=300, bbox_inches='tight')
#         print(f"  [{metric.upper()}] 3D-график (4 ракурса) сохранен: {out_path}")
        
#     #plt.show()

def plot_3d_applicability_4_angles(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    complexity_dataset_name: str = "Complexity_Test",
    metric: str = "ssim",
    fig_dir: Optional[Path] = None
):
    """
    Строит и сохраняет 4 ОТДЕЛЬНЫХ 3D-графика с шариками с разных углов обзора (azim),
    чтобы можно было выбрать лучший ракурс для вставки в отчет.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plot_data =[]
    colors = _get_palette_cycle()
    
    # 1. Собираем данные
    for alg_name, df_alg in all_data.items():
        df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
        if df_grid.empty: 
            continue
            
        df_grid = _calculate_xy_metrics(df_grid, metric)
        success_rate = df_grid['Success'].mean() * 100
        df_success = df_grid[df_grid['Success']]
        
        if df_success.empty: 
            continue
        
        x_coord = np.percentile(df_success['X'].dropna(), 90) if len(df_success['X'].dropna()) > 0 else 0
        y_coord = np.percentile(df_success['Y'].dropna(), 90) if len(df_success['Y'].dropna()) > 0 else 0
        
        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
        z_coord = df_comp['time_sec'].median() if not df_comp.empty else df_grid['time_sec'].median()
        
        plot_data.append({
            'alg': alg_name,
            'x': x_coord,
            'y': y_coord,
            'z': z_coord,
            'sr': success_rate,
            'color': next(colors)
        })

    if not plot_data:
        print(f"[{metric.upper()}] Нет данных для 3D-графика.")
        return

    # 4 угла обзора по кругу (высота elev=30, угол поворота azim меняется)
    angles =[
        (30, 45),   # Спереди-справа
        (30, 135),  # Сзади-справа
        (30, 225),  # Сзади-слева
        (30, 315)   # Спереди-слева
    ]
    
    x_label = "Сложность смаза (X)"
    y_label = "Влияние шума (Y)"
    z_label = "Время, сек (Z)"
    
    # 2. Генерируем 4 отдельных файла
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        legend_handles =[]
        
        for item in plot_data:
            size = max(item['sr'] * 5, 50)
            # Рисуем шарик
            ax.scatter(item['x'], item['y'], item['z'], 
                       s=size, c=[item['color']], alpha=0.9, edgecolors='black', linewidth=1)
            
            # Текст подписи (чуть выше шарика)
            z_offset = item['z'] + (max([d['z'] for d in plot_data]) * 0.05) if plot_data else item['z']
            ax.text(item['x'], item['y'], z_offset, 
                    item['alg'], fontsize=9, ha='center', va='bottom')
            
            # Собираем легенду
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=item['color'], 
                           markersize=10, label=f"{item['alg']} (SR: {item['sr']:.1f}%)")
            )
        
        # Настраиваем камеру
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"Область применимости ({metric.upper()}) | Ракурс: {azim}°", fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        
        # Выносим легенду аккуратно вправо за пределы 3D-куба
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=11)
        
        plt.tight_layout()
        
        if fig_dir:
            out_path = Path(fig_dir) / f"3d_applicability_{metric}_angle_{azim}.png"
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  [{metric.upper()}] Сохранен ракурс {azim}° -> {out_path.name}")
            
        # Закрываем фигуру, чтобы не забивать память и не спамить 8 всплывающими окнами
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  ИТЕРАЦИОННЫЕ ГРАФИКИ (Фаза 2): Кривые сходимости, эволюция ядра, kernel MSE
# ═══════════════════════════════════════════════════════════════════════════════

def plot_iteration_convergence(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Графики сходимости по итерациям: PSNR, SSIM, kernel_rmse — для каждого
    изображения в папке log_test.

    Строит один PDF/PNG на каждое изображение (3 подграфика: PSNR, SSIM, RMSE ядра)
    + один суммарный график со всеми изображениями.

    Parameters
    ----------
    iter_results_dir : Path
        Папка с подпапками изображений (каждая содержит iterations_log.csv).
    alg_label : str
        Название алгоритма.
    fig_dir : Path or None
        Куда сохранять графики (создаётся если нет).
    tex_dir : Path or None
        Куда сохранять TeX-обёртки.
    """
    iter_results_dir = Path(iter_results_dir)
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    image_dirs = sorted([
        d for d in iter_results_dir.iterdir()
        if d.is_dir() and (d / "iterations_log.csv").exists()
    ])
    if not image_dirs:
        print("  Нет данных итераций для построения кривых сходимости.")
        return

    all_dfs = {}
    for img_dir in image_dirs:
        csv_path = img_dir / "iterations_log.csv"
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        img_name = img_dir.name
        all_dfs[img_name] = df

        # ── Индивидуальный график для каждого изображения ────────────────
        has_psnr = 'psnr' in df.columns and df['psnr'].notna().any()
        has_ssim = 'ssim' in df.columns and df['ssim'].notna().any()
        has_krmse = 'kernel_rmse' in df.columns and df['kernel_rmse'].notna().any()

        n_subplots = sum([has_psnr, has_ssim, has_krmse])
        if n_subplots == 0:
            continue

        fig, axes = plt.subplots(1, n_subplots, figsize=(5.5 * n_subplots, 4.5))
        if n_subplots == 1:
            axes = [axes]

        ax_idx = 0
        iters = df['local_iter']

        if has_psnr:
            ax = axes[ax_idx]; ax_idx += 1
            mask = df['psnr'].notna()
            ax.plot(iters[mask], df['psnr'][mask], 'o-', color=PALETTE[0],
                    linewidth=2, markersize=5, label='PSNR')
            ax.set_xlabel('Итерация')
            ax.set_ylabel('PSNR (дБ)')
            ax.set_title(f'Сходимость PSNR — {img_name}', fontsize=TITLE_FONTSIZE)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        if has_ssim:
            ax = axes[ax_idx]; ax_idx += 1
            mask = df['ssim'].notna()
            ax.plot(iters[mask], df['ssim'][mask], 'o-', color=PALETTE[2],
                    linewidth=2, markersize=5, label='SSIM')
            ax.set_xlabel('Итерация')
            ax.set_ylabel('SSIM')
            ax.set_title(f'Сходимость SSIM — {img_name}', fontsize=TITLE_FONTSIZE)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        if has_krmse:
            ax = axes[ax_idx]; ax_idx += 1
            mask = df['kernel_rmse'].notna()
            ax.plot(iters[mask], df['kernel_rmse'][mask], '^-', color=PALETTE[4],
                    linewidth=2, markersize=5)
            ax.set_xlabel('Итерация')
            ax.set_ylabel('RMSE ядра')
            ax.set_title(f'Ошибка ядра (RMSE) — {img_name}', fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{alg_label}: Сходимость по итерациям — {img_name}',
                     fontsize=12, y=1.02)
        plt.tight_layout()

        fname = f"convergence_{img_name}"
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n"
                r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{"
                + f"figures/{fname}.pdf" + r"}" "\n"
                r"\caption{Кривые сходимости алгоритма " + alg_label
                + r" на изображении " + img_name.replace("_", r"\_")
                + r": PSNR, SSIM и ошибка ядра в зависимости от номера итерации.}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    # ── Суммарный график по всем изображениям ────────────────────────────
    if len(all_dfs) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = _get_palette_cycle()
        markers = _get_marker_cycle()

        for img_name, df in all_dfs.items():
            c = next(colors)
            m = next(markers)
            iters = df['local_iter']

            # PSNR
            if 'psnr' in df.columns and df['psnr'].notna().any():
                mask = df['psnr'].notna()
                axes[0].plot(iters[mask], df['psnr'][mask], marker=m, color=c,
                             linewidth=1.5, markersize=4, label=img_name)

            # SSIM
            if 'ssim' in df.columns and df['ssim'].notna().any():
                mask = df['ssim'].notna()
                axes[1].plot(iters[mask], df['ssim'][mask], marker=m, color=c,
                             linewidth=1.5, markersize=4, label=img_name)

            # Kernel RMSE
            if 'kernel_rmse' in df.columns and df['kernel_rmse'].notna().any():
                mask = df['kernel_rmse'].notna()
                axes[2].plot(iters[mask], df['kernel_rmse'][mask], marker=m, color=c,
                             linewidth=1.5, markersize=4, label=img_name)

        axes[0].set_xlabel('Итерация'); axes[0].set_ylabel('PSNR (дБ)')
        axes[0].set_title('Сходимость PSNR', fontsize=TITLE_FONTSIZE)
        axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Итерация'); axes[1].set_ylabel('SSIM')
        axes[1].set_title('Сходимость SSIM', fontsize=TITLE_FONTSIZE)
        axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Итерация'); axes[2].set_ylabel('RMSE ядра')
        axes[2].set_title('Ошибка ядра (RMSE)', fontsize=TITLE_FONTSIZE)
        axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{alg_label}: Сходимость по итерациям (все изображения)',
                     fontsize=13, y=1.02)
        plt.tight_layout()

        fname = "convergence_all"
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n"
                r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{"
                + f"figures/{fname}.pdf" + r"}" "\n"
                r"\caption{Суммарные кривые сходимости алгоритма " + alg_label
                + r" на всех тестовых изображениях.}" "\n"
                r"\label{fig:convergence_all}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  Кривые сходимости: {len(all_dfs)} изображений")


def plot_kernel_evolution_strip(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Горизонтальная полоска эволюции ядра: один ряд PNG-ядер от первой до
    последней итерации + финальное ядро справа.

    Строит один PNG/PDF на каждое изображение.
    """
    import cv2 as cv

    iter_results_dir = Path(iter_results_dir)
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    image_dirs = sorted([
        d for d in iter_results_dir.iterdir()
        if d.is_dir() and (d / "kernels").exists()
    ])
    if not image_dirs:
        print("  Нет данных ядер для построения эволюции.")
        return

    for img_dir in image_dirs:
        img_name = img_dir.name
        kernels_dir = img_dir / "kernels"
        kernel_files = sorted(kernels_dir.glob("kernel_s0_iter*.png"))
        if not kernel_files:
            continue

        # Загружаем ядра
        kernels = []
        labels = []
        for kf in kernel_files:
            k = cv.imread(str(kf), cv.IMREAD_GRAYSCALE)
            if k is not None:
                kernels.append(k)
                iter_num = kf.stem.split("iter")[-1].lstrip("0") or "0"
                labels.append(f"Ит. {iter_num}")

        # Добавляем финальное ядро
        final_kernel_path = img_dir / "kernel_final.png"
        if final_kernel_path.exists():
            k_final = cv.imread(str(final_kernel_path), cv.IMREAD_GRAYSCALE)
            if k_final is not None:
                kernels.append(k_final)
                labels.append("Финальное")

        if not kernels:
            continue

        # Выбираем максимум ~12 ядер для читабельности
        max_show = 12
        if len(kernels) > max_show:
            indices = np.linspace(0, len(kernels) - 1, max_show, dtype=int)
            indices = sorted(set(indices))
            kernels = [kernels[i] for i in indices]
            labels = [labels[i] for i in indices]

        n = len(kernels)
        fig, axes = plt.subplots(1, n, figsize=(min(n * 1.8, 22), 2.5))
        if n == 1:
            axes = [axes]

        for i, (k, lbl) in enumerate(zip(kernels, labels)):
            axes[i].imshow(k, cmap='hot', interpolation='nearest')
            axes[i].set_title(lbl, fontsize=8)
            axes[i].axis('off')

        fig.suptitle(f'{alg_label}: Эволюция ядра — {img_name}',
                     fontsize=11, y=1.05)
        plt.tight_layout()

        fname = f"kernel_evolution_{img_name}"
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n"
                r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{"
                + f"figures/{fname}.pdf" + r"}" "\n"
                r"\caption{Эволюция ядра алгоритма " + alg_label
                + r" на изображении " + img_name.replace("_", r"\_")
                + r". Слева направо: от ранних итераций к финальному ядру.}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  Эволюция ядер: {len(image_dirs)} изображений")


# ═══════════════════════════════════════════════════════════════════════════════
#  ГИПЕРПАРАМЕТРИЧЕСКИЕ ГРАФИКИ (Фаза 3): Тепловая карта + 1D чувствительность
# ═══════════════════════════════════════════════════════════════════════════════

def plot_hyperparam_heatmap(
    csv_path: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Тепловая карта PSNR/SSIM по 2D сетке гиперпараметров.

    Читает CSV с колонками: image, <param1>, <param2>, psnr, ssim, ...
    Строит 2 тепловые карты (PSNR и SSIM), усреднённые по всем изображениям.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  Файл не найден: {csv_path}")
        return

    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  Пустой CSV для тепловой карты.")
        return

    # Определяем имена параметров
    known_cols = {'image', 'psnr', 'ssim', 'time_sec', 'error_ratio'}
    param_cols = [c for c in df.columns if c not in known_cols]
    if len(param_cols) < 2:
        print(f"  Не удалось определить 2 параметра в {csv_path.name}")
        return

    p1, p2 = param_cols[0], param_cols[1]

    for metric, metric_label, cmap_name in [
        ('psnr', 'PSNR (дБ)', 'YlOrRd'),
        ('ssim', 'SSIM', 'YlGnBu'),
    ]:
        if metric not in df.columns or df[metric].isna().all():
            continue

        # Усредняем по изображениям
        pivot = df.groupby([p1, p2])[metric].mean().reset_index()
        heatmap = pivot.pivot(index=p1, columns=p2, values=metric)
        heatmap = heatmap.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(max(len(heatmap.columns) * 1.2, 7),
                                        max(len(heatmap.index) * 0.8, 5)))

        im = ax.imshow(heatmap.values, cmap=cmap_name, aspect='auto',
                       interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_label, fontsize=10)

        # Подписи осей
        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels([f"{v:.4g}" for v in heatmap.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels([f"{v:.4g}" for v in heatmap.index])

        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f'{alg_label}: {metric_label} — сетка ({p1} × {p2})',
                     fontsize=TITLE_FONTSIZE)

        # Числа в ячейках
        for i in range(len(heatmap.index)):
            for j in range(len(heatmap.columns)):
                val = heatmap.values[i, j]
                if pd.notna(val):
                    txt = f"{val:.2f}" if metric == 'psnr' else f"{val:.3f}"
                    norm_val = (val - np.nanmin(heatmap.values)) / (
                        np.nanmax(heatmap.values) - np.nanmin(heatmap.values) + 1e-12)
                    txt_color = 'white' if norm_val > 0.7 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=8, color=txt_color, fontweight='bold')

        # Рамка вокруг лучшей ячейки
        best_idx = np.unravel_index(np.nanargmax(heatmap.values), heatmap.values.shape)
        ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5),
                                   1, 1, fill=False, edgecolor='lime',
                                   linewidth=3))

        plt.tight_layout()

        fname = f"heatmap_{metric}_{p1}_{p2}"
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n"
                r"\centering" "\n"
                r"\includegraphics[width=0.9\textwidth]{"
                + f"figures/{fname}.pdf" + r"}" "\n"
                r"\caption{Тепловая карта " + metric_label
                + r" алгоритма " + alg_label
                + r" по сетке гиперпараметров "
                + p1.replace("_", r"\_") + r" и " + p2.replace("_", r"\_")
                + r". Зелёная рамка --- лучшая комбинация.}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  Тепловые карты: {csv_path.name}")


def plot_hyperparam_sensitivity_1d(
    csv_path: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Кривые чувствительности: фиксируем один параметр (лучшее значение),
    варьируем другой. Две пары графиков (по каждому параметру).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  Файл не найден: {csv_path}")
        return

    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    known_cols = {'image', 'psnr', 'ssim', 'time_sec', 'error_ratio'}
    param_cols = [c for c in df.columns if c not in known_cols]
    if len(param_cols) < 2:
        return

    p1, p2 = param_cols[0], param_cols[1]

    # Усредняем по изображениям
    avg = df.groupby([p1, p2])[['psnr', 'ssim']].mean().reset_index()

    # Находим лучшую комбинацию по PSNR
    best_row = avg.loc[avg['psnr'].idxmax()]
    best_p1 = best_row[p1]
    best_p2 = best_row[p2]

    # Два среза: фиксируем p1=best, варьируем p2 и наоборот
    slices = [
        (p2, p1, best_p1, avg[avg[p1] == best_p1].sort_values(p2)),
        (p1, p2, best_p2, avg[avg[p2] == best_p2].sort_values(p1)),
    ]

    for vary_param, fix_param, fix_val, df_slice in slices:
        if df_slice.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = df_slice[vary_param].values

        # PSNR
        ax1.plot(x, df_slice['psnr'].values, 'o-', color=PALETTE[0],
                 linewidth=2, markersize=6)
        ax1.set_xlabel(vary_param)
        ax1.set_ylabel('PSNR (дБ)')
        ax1.set_title(f'Чувствительность PSNR к {vary_param}', fontsize=TITLE_FONTSIZE)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # SSIM
        ax2.plot(x, df_slice['ssim'].values, 's-', color=PALETTE[2],
                 linewidth=2, markersize=6)
        ax2.set_xlabel(vary_param)
        ax2.set_ylabel('SSIM')
        ax2.set_title(f'Чувствительность SSIM к {vary_param}', fontsize=TITLE_FONTSIZE)
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(
            f'{alg_label}: Чувствительность к {vary_param} '
            f'(при {fix_param}={fix_val:.4g})',
            fontsize=12, y=1.02)
        plt.tight_layout()

        fname = f"sensitivity_{vary_param}_fix_{fix_param}"
        if fig_dir:
            fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n"
                r"\centering" "\n"
                r"\includegraphics[width=0.9\textwidth]{"
                + f"figures/{fname}.pdf" + r"}" "\n"
                r"\caption{Чувствительность PSNR и SSIM алгоритма " + alg_label
                + r" к параметру " + vary_param.replace("_", r"\_")
                + r" при фиксированном "
                + fix_param.replace("_", r"\_") + f"={fix_val:.4g}"
                + r".}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  Кривые чувствительности: {csv_path.name}")