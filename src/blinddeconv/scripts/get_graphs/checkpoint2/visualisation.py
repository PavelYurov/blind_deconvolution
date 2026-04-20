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
