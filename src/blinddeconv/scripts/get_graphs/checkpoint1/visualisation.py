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
#     plt.show()

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
    plt.show()

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
#             plt.show()
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
#         plt.show()
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
            plt.show()
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
        plt.show()
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
    plt.show()

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
#             plt.show()

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
            plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
    plt.show()

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
#         plt.show()

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
        plt.show()

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

def build_summary_single(
    all_dataset_results: List[Dict],
    alg_label: str,
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Таблица средних метрик по наборам данных для одного алгоритма."""
    rows = []
    for r in all_dataset_results:
        d = r['df']
        rows.append({
            'Набор данных': r['dataset'],
            'Кол-во': len(d),
            'PSNR (среднее)': round(d['psnr'].mean(), 2) if d['psnr'].notna().any() else '—',
            'SSIM (среднее)': round(d['ssim'].mean(), 4) if d['ssim'].notna().any() else '—',
            'Отн. ошибки (среднее)': round(d['error_ratio'].mean(), 2) if d['error_ratio'].notna().any() else '—',
            'Время (ср., с)': round(d['time_sec'].mean(), 2),
        })
    df_s = pd.DataFrame(rows)

    if tex_dir:
        tex = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Сводные результаты алгоритма " + alg_label
            + r" на тестовых наборах данных.}" "\n"
            r"\label{tab:summary_" + _safe_label(alg_label) + r"}" "\n"
            r"\begin{tabular}{l c c c c c}" "\n" r"\hline" "\n"
            r"Набор данных & Кол-во & PSNR & SSIM & Отн. ошибки & Время (с) \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_s.iterrows():
            tex += (f"{row['Набор данных']} & {row['Кол-во']} & "
                    f"{row['PSNR (среднее)']} & {row['SSIM (среднее)']} & "
                    f"{row['Отн. ошибки (среднее)']} & {row['Время (ср., с)']} \\\\\n")
        tex += r"\hline" "\n" r"\end{tabular}" "\n" r"\end{table}"
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
