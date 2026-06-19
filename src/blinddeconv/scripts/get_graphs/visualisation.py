"""
Модуль визуализации результатов слепой деконволюции.

Содержит функции для построения почти ВСЕХ графиков и генерации TeX-кода
таблиц и фигур для отчётов и презентаций.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from itertools import cycle
from typing import Optional, Dict, List, Tuple, Any

PALETTE = [
    '#2176AE', 
    '#E05929',
    '#57A773',
    '#B5338A',
    '#F2C12E',
    '#1B998B',
    '#D64045',
    '#6B4226',
    '#3D5A80',
    '#EE6C4D',
]

TITLE_FONTSIZE = 11

MARKERS =['s', '^', '+', 'D', 'o', 'v', '*']

def _get_marker_cycle():
    return cycle(MARKERS)

def _get_palette_cycle():
    return cycle(PALETTE)


def _colormap_bars(n: int):
    """Возвращает массив из n различных цветов."""
    cmap = cm.get_cmap('tab20', max(n, 1))
    return [cmap(i) for i in range(n)]


def _bar_ymin(vals, metric: str) -> float:
    """Нижняя граница оси Y для столбцатых диаграмм.

    PSNR  - floor до кратного 5, затем ещё -5  (напр. 20.69 -> 20 -> 15)
    SSIM  - floor до кратного 0.1, затем ещё -0.1 (напр. 0.67 -> 0.6 -> 0.5)
    Результат не уходит ниже 0.
    """
    import math
    finite = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not finite:
        return 0.0
    mn = min(finite)
    if metric in ('psnr', 'psnr_mean'):
        return max(0.0, math.floor(mn / 5) * 5 - 5)
    elif metric in ('ssim', 'ssim_mean'):
        return max(0.0, round(math.floor(mn * 10) / 10 - 0.1, 10))
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Декодер (мэппинг ФС-имён -> подписи на графиках)
# ═══════════════════════════════════════════════════════════════════════════════
#
# В таблицах/файловой системе используются «технические» имена (директории
# алгоритмов, датасетов, изображений, типов шума и т.п.). На презентационных
# графиках хочется видеть человекочитаемые подписи. Этот модуль предоставляет
# универсальный декодер: если имя есть в карте — подменяется, нет — остаётся
# как было.
#
#   set_label_map({"pmp_denoise": "PMP", "Set12": "Set12 (12 изобр.)"})
#   decode("pmp_denoise")  -> "PMP"
#   decode("неизвестное")  -> "неизвестное"
#
# Точечная подмена внутри строк выполняется через decode_in_text (заменяет
# слова целиком, по убыванию длины ключа, чтобы избежать частичных совпадений).
_LABEL_MAP: Dict[str, str] = {}

def set_label_map(mapping: Optional[Dict[str, str]]) -> None:
    """Устанавливает текущую карту подмены подписей. None / {} сбрасывает."""
    global _LABEL_MAP
    _LABEL_MAP = dict(mapping) if mapping else {}

def get_label_map() -> Dict[str, str]:
    """Возвращает копию текущей карты подмены."""
    return dict(_LABEL_MAP)

def decode(name: Any) -> Any:
    """Подменяет имя по карте, затем заменяет нижние подчёркивания пробелами.
    Если имя не входит в карту, всё равно делает замену '_' -> ' '."""
    if name is None:
        return name
    s = str(name)
    s = _LABEL_MAP.get(s, s)
    return s.replace("_", " ")

def decode_in_text(text: str) -> str:
    """Подменяет все ключи карты внутри строки (как отдельные подстроки),
    затем заменяет одиночные '_' пробелами (LaTeX-последовательности '\\_'
    остаются нетронутыми).  Длинные ключи заменяются раньше коротких."""
    if not text:
        return text
    out = str(text)
    if _LABEL_MAP:
        for k in sorted(_LABEL_MAP.keys(), key=len, reverse=True):
            out = out.replace(k, _LABEL_MAP[k])
    out = out.replace(r"\_", "\x00")
    out = out.replace("_", " ")
    out = out.replace("\x00", r"\_")
    return out


def _apply_decoder_to_fig(fig) -> None:
    """Прогоняет все текстовые элементы фигуры через decode_in_text.
    Вызывается автоматически в Figure.savefig (см. monkey-patch ниже)."""
    try:
        sup = getattr(fig, "_suptitle", None)
        if sup is not None and sup.get_text():
            sup.set_text(decode_in_text(sup.get_text()))
        for ax in fig.get_axes():
            for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
                if t and t.get_text():
                    t.set_text(decode_in_text(t.get_text()))
            zax = getattr(ax, 'zaxis', None)
            if zax is not None and zax.label.get_text():
                zax.label.set_text(decode_in_text(zax.label.get_text()))
            # tick labels (если они уже строки, а не число)
            for lab in (list(ax.get_xticklabels()) + list(ax.get_yticklabels())
                        + (list(ax.get_zticklabels()) if zax is not None else [])):
                txt = lab.get_text()
                if txt:
                    lab.set_text(decode_in_text(txt))
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    if t.get_text():
                        t.set_text(decode_in_text(t.get_text()))
            for t in ax.texts:
                if t.get_text():
                    t.set_text(decode_in_text(t.get_text()))
    except Exception:
        pass


#    Figure.savefig: автоматически применяет decoder перед записью.
#    Срабатывает один раз при импорте модуля; идемпотентно благодаря флагу.
import matplotlib.figure as _mpl_fig  
if not getattr(_mpl_fig.Figure.savefig, "_decoder_patched", False):
    _orig_savefig = _mpl_fig.Figure.savefig

    def _patched_savefig(self, *args, **kwargs):
        _apply_decoder_to_fig(self)
        return _orig_savefig(self, *args, **kwargs)

    _patched_savefig._decoder_patched = True
    _mpl_fig.Figure.savefig = _patched_savefig  


#  Утилиты

def save_tex(filepath: Path, tex_code: str):
    """Сохраняет TeX-код в файл."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(tex_code)
    print(f"  TeX сохранён: {filepath}")


def _safe_label(s: str) -> str:
    return s.replace("-", "_").replace(" ", "_")

# Отношение ошибок
def plot_success_rate_single(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    suffix: str = "",
):
    """Кумулятивный график доли успешных для одного алгоритма."""
    er = er_values.dropna()
    if len(er) == 0:
        print("  Нет данных отношения ошибок для построения графика.")
        return

    x_max = float(np.clip(np.nanpercentile(er, 99) * 1.2, 3.5, 10.0))
    thresholds = np.arange(1.0, x_max + 0.05, 0.05)
    sr = [(er <= t).sum() / len(er) * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sr, linewidth=2, color=PALETTE[0], label=alg_label, 
            marker='s', markersize=8, markerfacecolor='none', markevery=20)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Порог r=3')
    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Доля успешных (%)')
    ax.set_title(f'Доля успешных / отношение ошибок — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"success_rate{suffix}"
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"{fname}.pdf")
        fig.savefig(Path(fig_dir) / f"{fname}.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{" + f"figures/{fname}.pdf" + r"}" "\n"
            r"\caption{Кумулятивный график доли успешных для алгоритма " + alg_label +
            r". Ось абсцисс --- пороговое значение отношения ошибок, "
            r"ось ординат --- доля изображений с отношением ошибок ниже порога.}" "\n"
            r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"{fname}.tex", tex)


def plot_success_rate_comparison(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Кумулятивный график доли успешных для нескольких алгоритмов."""

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
        markers = _get_marker_cycle()
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
            ax.set_xlabel('Отношение ошибок')
            ax.set_ylabel('Доля успешных (%)')
            ax.set_title(f'Доля успешных / отношение ошибок — набор данных {ds_name}',
                         fontsize=TITLE_FONTSIZE)
            ax.set_xlim(1, x_max_ds); ax.set_ylim(0, 105)
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"success_rate_{ds_name}.pdf")
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
            print(f"  Набор данных {ds_name}: нет данных отношения ошибок")

    # Общий
    _er_all = pd.concat([df_alg['error_ratio'].dropna() for df_alg in all_data.values()])
    x_max_all = float(np.clip(np.nanpercentile(_er_all, 99) * 1.2, 3.5, 10.0)) if len(_er_all) > 0 else 6.0
    thresholds_all = np.arange(1.0, x_max_all + 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = _get_palette_cycle()
    markers = _get_marker_cycle()
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
        ax.set_xlabel('Отношение ошибок')
        ax.set_ylabel('Доля успешных (%)')
        ax.set_title('Доля успешных / отношение ошибок — все наборы данных',
                     fontsize=TITLE_FONTSIZE)
        ax.set_xlim(1, x_max_all); ax.set_ylim(0, 105)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if fig_dir:
            # fig.savefig(Path(fig_dir) / "success_rate_all.pdf")
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
        print("  Нет данных отношения ошибок ни для одного алгоритма.")


#  Столбчатая PSNR / SSIM по изображениям (один алгоритм, разноцветные)

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
    colors[-1] = (0.55, 0.55, 0.55, 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(n * 1.2 + 2, 8), 5))
    x = np.arange(n)

    b1 = ax1.bar(x, grp['psnr_mean'].values, color=colors, alpha=0.88, edgecolor='grey', linewidth=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grp['_img'].values, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('PSNR, дБ')
    ax1.set_title(f'PSNR по изображениям — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax1.set_ylim(bottom=_bar_ymin(grp['psnr_mean'].values, 'psnr'))
    for bar, val in zip(b1, grp['psnr_mean'].values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    b2 = ax2.bar(x, grp['ssim_mean'].values, color=colors, alpha=0.88, edgecolor='grey', linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(grp['_img'].values, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM по изображениям — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax2.set_ylim(bottom=_bar_ymin(grp['ssim_mean'].values, 'ssim'))
    for bar, val in zip(b2, grp['ssim_mean'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    if fig_dir:
        # fig.savefig(Path(fig_dir) / "psnr_ssim_bar.pdf")
        fig.savefig(Path(fig_dir) / "psnr_ssim_bar.png")
        ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
        # fig.savefig(Path(fig_dir) / "psnr_ssim_bar_full.pdf")
        fig.savefig(Path(fig_dir) / "psnr_ssim_bar_full.png")
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

    df_m['_img'] = df_m['distorted_file'].apply(lambda x: Path(x).stem.rsplit('_', 1)[0])

    grp = df_m.groupby(['dataset', '_img']).agg(
        psnr_mean=('psnr', 'mean'),
        ssim_mean=('ssim', 'mean'),
    ).reset_index()

    datasets = grp['dataset'].unique()
    ds_colors = dict(zip(datasets, _colormap_bars(len(datasets))))

    grp = grp.sort_values(['dataset', '_img']).reset_index(drop=True)

    n = len(grp)
    if n == 0:
        return

    x_labels = [f"{row['_img']}\n({row['dataset']})" for _, row in grp.iterrows()]
    bar_colors = [ds_colors[row['dataset']] for _, row in grp.iterrows()]

    fig_w = max(n * 0.9 + 3, 10)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 9))

    x = np.arange(n)

    #PSNR
    b1 = ax1.bar(x, grp['psnr_mean'].values, color=bar_colors, alpha=0.85,
                 edgecolor='grey', linewidth=0.3)
    for bar, val in zip(b1, grp['psnr_mean'].values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, rotation=90)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=9)
    ax1.set_ylabel('PSNR, дБ')
    ax1.set_title(f'PSNR по изображениям (все датасеты) — {alg_label}',
                  fontsize=TITLE_FONTSIZE)
    ax1.set_ylim(bottom=_bar_ymin(grp['psnr_mean'].values, 'psnr'))
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    #SSIM
    b2 = ax2.bar(x, grp['ssim_mean'].values, color=bar_colors, alpha=0.85,
                 edgecolor='grey', linewidth=0.3)
    for bar, val in zip(b2, grp['ssim_mean'].values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=60, ha='right', fontsize=9)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'SSIM по изображениям (все датасеты) — {alg_label}',
                  fontsize=TITLE_FONTSIZE)
    ax2.set_ylim(bottom=_bar_ymin(grp['ssim_mean'].values, 'ssim'))
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    #Легенда по датасетам
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=ds_colors[ds], edgecolor='grey', label=ds)
                      for ds in datasets]
    ax1.legend(handles=legend_handles, fontsize=9, loc='upper right',
               title='Датасет', title_fontsize=10)

    plt.tight_layout()

    fname = "psnr_ssim_per_image_all"
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)
        # fig.savefig(Path(fig_dir) / f"{fname}_full.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / f"{fname}_full.png", dpi=200, bbox_inches='tight')
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


def plot_bar_psnr_ssim_comparison(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Сгруппированная столбчатая диаграмма PSNR/SSIM (X=изображения, группы=алгоритмы)."""
    for ds_name in datasets:
        alg_names = [
            a for a, df_a in all_data.items()
            if len(df_a[df_a['dataset'] == ds_name].dropna(subset=['psnr'])) > 0
        ]
        if not alg_names:
            print(f"  Набор данных {ds_name}: нет данных PSNR/SSIM")
            continue

        img_set = set()
        for a in alg_names:
            df_ds = all_data[a][all_data[a]['dataset'] == ds_name].dropna(subset=['psnr'])
            img_set.update(df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0]))
        image_names = sorted(img_set)
        x_labels = [decode(img) for img in image_names] + ['Среднее']

        n_img = len(image_names)
        n_alg = len(alg_names)
        bar_w = 0.8 / n_alg
        colors = _colormap_bars(n_alg)
        x_base = np.arange(n_img + 1)

        for metric, ylabel, fname_m in [
            ('psnr', 'Средний PSNR, дБ', 'psnr'),
            ('ssim', 'Средний SSIM', 'ssim'),
        ]:
            fig_w = max((n_img + 1) * 0.55 + 3, 8)
            fig, ax = plt.subplots(figsize=(fig_w, 4.5))
            for idx, alg_name in enumerate(alg_names):
                df_ds = all_data[alg_name][
                    all_data[alg_name]['dataset'] == ds_name
                ].dropna(subset=[metric]).copy()
                df_ds['_img'] = df_ds['distorted_file'].apply(lambda x: Path(x).stem.split('_')[0])
                grp = df_ds.groupby('_img')[metric].mean()
                vals =[grp.get(img, np.nan) for img in image_names]
                vals.append(float(np.nanmean(vals)))
                ax.bar(
                    x_base + idx * bar_w, vals, width=bar_w * 0.9,
                    color=colors[idx], alpha=1.0, label=alg_name,
                    edgecolor='black', linewidth=0.5,
                )
            ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
            ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel(ylabel)
            metric_title = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(f'{metric_title} по изображениям — {ds_name}', fontsize=TITLE_FONTSIZE)
            
            ax.set_ylim(bottom=_bar_ymin(
                [v for vals_list in [
                    [all_data[a][all_data[a]['dataset'] == ds_name].dropna(subset=[metric])[metric].mean()
                     for a in alg_names]
                ] for v in vals_list], metric))
            ax.grid(axis='y', color='black', alpha=0.3, linestyle='-')
            ax.set_axisbelow(True)

            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=10, borderaxespad=0.0)
            plt.tight_layout()

            fname = f"bar_{fname_m}_{ds_name}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf")
                fig.savefig(Path(fig_dir) / f"{fname}.png")
                ax.set_ylim(bottom=0)
                # fig.savefig(Path(fig_dir) / f"{fname}_full.pdf")
                fig.savefig(Path(fig_dir) / f"{fname}_full.png")
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


# Гистограмма распределения отношения ошибок

def plot_error_ratio_histogram_single(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    er = er_values.dropna()
    if len(er) == 0:
        print("  Нет данных отношения ошибок для гистограммы.")
        return
    # Обрезаем выбросы: всё что >=10 — в последний столбец
    er = er.clip(upper=10.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max(er.max(), 5) + 0.5, 0.5)
    ax.hist(er, bins=bins, color=PALETTE[0], edgecolor='black', alpha=0.8)
    ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='Порог r=3')
    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Количество изображений')
    ax.set_title(f'Распределение отношения ошибок — {alg_label}', fontsize=TITLE_FONTSIZE)
    ax.legend()
    plt.tight_layout()

    if fig_dir:
        # fig.savefig(Path(fig_dir) / "error_ratio_histogram.pdf")
        fig.savefig(Path(fig_dir) / "error_ratio_histogram.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{figures/error_ratio_histogram.pdf}" "\n"
            r"\caption{Распределение отношения ошибок для алгоритма " + alg_label
            + r". Красная линия --- порог $r=3$, выше которого результаты "
            r"считаются визуально неприемлемыми.}" "\n"
            r"\label{fig:er_hist_" + _safe_label(alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "error_ratio_histogram.tex", tex)


# Стековая гистограмма: цвета не перекрываются. Сортируем по объёму
# выборки, чтобы крупные распределения лежали внизу.
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
            er_has[a] = v.clip(upper=10.0)

    if not er_has:
        print("  Нет данных отношения ошибок для сравнительной гистограммы.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    max_er = max(v.max() for v in er_has.values())
    bins = np.arange(0, min(max_er + 0.5, 15), 0.5)

    items = sorted(er_has.items(), key=lambda kv: -len(kv[1]))
    palette = _get_palette_cycle()
    color_list = [next(palette) for _ in items]
    data_list = [v.values for _, v in items]
    label_list = [a for a, _ in items]

    ax.hist(data_list, bins=bins, stacked=True,
            color=color_list, label=label_list,
            edgecolor='black', linewidth=0.4)
    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Количество изображений')
    ax.set_title('Распределение отношения ошибок — сравнение алгоритмов',
                 fontsize=TITLE_FONTSIZE)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, borderaxespad=0.0)
    plt.tight_layout()

    if fig_dir:
        # fig.savefig(Path(fig_dir) / "error_ratio_hist_cmp.pdf")
        fig.savefig(Path(fig_dir) / "error_ratio_hist_cmp.png")
    #plt.show()

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{"
            + f"{fig_prefix}/error_ratio_hist_cmp.pdf" + r"}" "\n"
            r"\caption{Сравнение распределений отношения ошибок для разных алгоритмов. "
            r"Красная линия --- порог $r=3$.}" "\n"
            r"\label{fig:er_hist_cmp}" "\n" r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "error_ratio_hist_cmp.tex", tex)


def plot_error_ratio_histogram_comparison_sorted(
    all_data: Dict[str, pd.DataFrame],
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Гистограмма отношения ошибок — «послойные» столбцы.

    Для каждого столюца алгоритмы сортируются по числу изображений по возрастанию.
    Нижний сегмент — алгоритм с наименьшим числом, каждый следующий сегмент —
    разница между следующим и предыдущим.
    """
    er_has = {}
    for a, df_a in all_data.items():
        v = df_a['error_ratio'].dropna()
        if len(v) > 0:
            er_has[a] = v.clip(upper=10.0).values

    if not er_has:
        print("  Нет данных отношения ошибок для послойной гистограммы.")
        return

    alg_names = list(er_has.keys())
    max_er = max(v.max() for v in er_has.values())
    bins = np.arange(0, min(max_er + 0.5, 15), 0.5)
    bin_width = bins[1] - bins[0]

    palette = _get_palette_cycle()
    alg_colors = {a: next(palette) for a in alg_names}

    counts = {a: np.histogram(er_has[a], bins=bins)[0] for a in alg_names}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = (bins[:-1] + bins[1:]) / 2  # центры бинов

    added_to_legend = set()
    for i in range(len(bins) - 1):
        bin_counts = sorted(
            [(a, int(counts[a][i])) for a in alg_names if counts[a][i] > 0],
            key=lambda kv: kv[1]
        )
        prev = 0
        for a, cnt in bin_counts:
            seg_h = cnt - prev
            if seg_h > 0:
                lbl = decode(a) if a not in added_to_legend else '_nolegend_'
                ax.bar(x[i], seg_h, width=bin_width * 0.9, bottom=prev,
                       color=alg_colors[a], edgecolor='black', linewidth=0.4,
                       label=lbl)
                added_to_legend.add(a)
            prev = cnt

    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Количество изображений')
    ax.set_title('Распределение отношения ошибок — сравнение алгоритмов',
                 fontsize=TITLE_FONTSIZE)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=9, borderaxespad=0.0)
    plt.tight_layout()

    fname = "error_ratio_hist_cmp_sorted"
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"{fname}.pdf")
        fig.savefig(Path(fig_dir) / f"{fname}.png")
    plt.close(fig)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{"
            + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
            r"\caption{Послойное сравнение распределений отношения ошибок. "
            r"В каждом бине алгоритмы отсортированы по числу изображений: "
            r"нижний сегмент --- алгоритм с наименьшим числом, "
            r"каждый следующий сегмент --- дополнительные изображения следующего алгоритма.}" "\n"
            r"\label{fig:er_hist_cmp_sorted}" "\n" r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"{fname}.tex", tex)


# Зависимость от шума (PSNR + SSIM)

def _split_noise_name(s: str) -> Tuple[str, str]:
    """Разбирает имя шума вида weakgaussian / clean на (тип, уровень).

    Возвращает (noise_type, level), где
      level ∈ {clean, weak, medium, strong},
      noise_type ∈ {clean, gaussian, poisson, impulse, pink, brown, ...}.
    Для clean оба поля равны 'clean'.
    """
    if not isinstance(s, str) or not s:
        return ("unknown", "unknown")
    sl = s.strip().lower()
    if sl == "clean":
        return ("clean", "clean")
    for lvl in ("weak", "medium", "strong"):
        if sl.startswith(lvl):
            return (sl[len(lvl):] or "unknown", lvl)
    return (sl, "unknown")


def plot_noise_dependency(
    df_global: pd.DataFrame,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Зависимость PSNR/SSIM от шума.

    На каждый тип шума (gaussian/poisson/impulse/pink/brown/…) — отдельный
    столбчатый-график с 4 группами (clean / weak / medium / strong) х алгоритмы.

    Дополнительно строится линейный график зависимости метрики от BSNR
    (psnr_blurred - PSNR смазанного к чистому, ~ BSNR).
    """
    if 'noise_name' not in df_global.columns:
        print("  Столбец noise_name отсутствует.")
        return

    df_n = df_global[df_global['noise_name'].notna() & (df_global['noise_name'] != '')].copy()
    if df_n.empty:
        print("  Нет информации о шуме.")
        return

    parsed = df_n['noise_name'].map(_split_noise_name)
    df_n['noise_type'] = parsed.map(lambda t: t[0])
    df_n['noise_level'] = parsed.map(lambda t: t[1])

    levels_order = ['clean', 'weak', 'medium', 'strong']
    types = sorted(t for t in df_n['noise_type'].unique() if t not in ('clean', 'unknown'))
    if not types:
        print("  Только 'clean' — графики шума не информативны.")
        return

    algorithms = sorted(df_n['algorithm'].unique())
    n_alg = len(algorithms)
    colors = _colormap_bars(n_alg)

    # Столбчатый-графики: один на каждый тип шума × метрику ─────────
    for metric, ylabel in [('psnr', 'Средний PSNR, дБ'),
                            ('ssim', 'Средний SSIM')]:
        for ntype in types:
            sub = df_n[(df_n['noise_type'] == ntype) | (df_n['noise_type'] == 'clean')].copy()
            grouped = (sub.groupby(['algorithm', 'noise_level'])[metric]
                          .mean().reset_index())
            present_levels = [lvl for lvl in levels_order
                              if lvl in grouped['noise_level'].unique()]
            n_lvl = len(present_levels)
            if n_lvl < 2:
                continue
            x_base = np.arange(n_lvl)
            bar_w = 0.7 / max(n_alg, 1)

            fig_w = max(n_lvl * n_alg * 0.32 + 3, 7)
            fig, ax = plt.subplots(figsize=(fig_w, 4.6))
            for idx, alg in enumerate(algorithms):
                sub_alg = grouped[grouped['algorithm'] == alg].set_index('noise_level')
                vals = [sub_alg.loc[lvl, metric] if lvl in sub_alg.index else np.nan
                        for lvl in present_levels]
                ax.bar(x_base + idx * bar_w, vals, width=bar_w * 0.85,
                       color=colors[idx], alpha=0.95, label=decode(alg),
                       edgecolor='black', linewidth=0.4)
            ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
            tick_labels = []
            for lvl in present_levels:
                key = 'clean' if lvl == 'clean' else f"{lvl}{ntype}"
                tick_labels.append(decode(key))
            ax.set_xticklabels(tick_labels, rotation=15, ha='right')
            ax.set_ylabel(ylabel)
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(f'Устойчивость к шуму ({decode(ntype)}) — {mname}',
                         fontsize=TITLE_FONTSIZE)
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=10, borderaxespad=0.0)
            plt.tight_layout()

            fname = f"noise_dependency_{metric}_{ntype}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Зависимость " + mname
                    + r" от уровня шума типа " + ntype
                    + r" для разных алгоритмов.}" "\n"
                    r"\label{fig:nd_" + metric + "_" + ntype + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    # Линейный график: метрика / уровень шума -
    # Один график на тип шума × метрику; X - уровень.
    for metric, ylabel in [('psnr', 'Средний PSNR, дБ'),
                            ('ssim', 'Средний SSIM')]:
        for ntype in types:
            sub = df_n[(df_n['noise_type'] == ntype) | (df_n['noise_type'] == 'clean')].copy()
            grouped = (sub.groupby(['algorithm', 'noise_level'])[metric]
                          .mean().reset_index())
            present_levels = [lvl for lvl in levels_order
                              if lvl in grouped['noise_level'].unique()]
            if len(present_levels) < 2:
                continue
            x_idx = np.arange(len(present_levels))

            fig, ax = plt.subplots(figsize=(8, 5))
            colors_cycle = _get_palette_cycle()
            markers_cycle = _get_marker_cycle()
            for alg in algorithms:
                sub_alg = grouped[grouped['algorithm'] == alg].set_index('noise_level')
                vals = [sub_alg.loc[lvl, metric] if lvl in sub_alg.index else np.nan
                        for lvl in present_levels]
                ax.plot(x_idx, vals, marker=next(markers_cycle), linewidth=2,
                        markersize=7, color=next(colors_cycle), label=decode(alg))
            tick_labels = []
            for lvl in present_levels:
                key = 'clean' if lvl == 'clean' else f"{lvl}{ntype}"
                tick_labels.append(decode(key))
            ax.set_xticks(x_idx)
            ax.set_xticklabels(tick_labels, rotation=15, ha='right')
            ax.set_xlabel('Уровень шума')
            ax.set_ylabel(ylabel)
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(f'{mname} в зависимости от уровня шума ({decode(ntype)})',
                         fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=12, borderaxespad=0.0)
            plt.tight_layout()
            fname = f"noise_level_line_{metric}_{ntype}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Зависимость " + mname
                    + r" от уровня шума типа " + ntype
                    + r" (линейный график).}" "\n"
                    r"\label{fig:nl_" + metric + "_" + ntype + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    # Линейный график: метрика / BSNR (psnr_blurred)
    if 'psnr_blurred' in df_n.columns and df_n['psnr_blurred'].notna().any():
        for metric, ylabel in [('psnr', 'Средний PSNR, дБ'),
                                ('ssim', 'Средний SSIM')]:
            df_b = df_n.dropna(subset=['psnr_blurred', metric]).copy()
            if df_b.empty:
                continue
            df_b['bsnr_bin'] = (df_b['psnr_blurred'] / 2).round() * 2
            grouped = (df_b.groupby(['algorithm', 'bsnr_bin'])[metric]
                           .mean().reset_index().sort_values('bsnr_bin'))

            fig, ax = plt.subplots(figsize=(9, 5))
            colors_cycle = _get_palette_cycle()
            markers = _get_marker_cycle()
            for alg in algorithms:
                sub = grouped[grouped['algorithm'] == alg]
                if sub.empty:
                    continue
                ax.plot(sub['bsnr_bin'], sub[metric],
                        marker=next(markers), linewidth=2, markersize=6,
                        color=next(colors_cycle), label=decode(alg))
            ax.set_xlabel('BSNR (PSNR смазанного к чистому), дБ')
            ax.set_ylabel(ylabel)
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(f'Зависимость {mname} от BSNR', fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=12, borderaxespad=0.0)
            plt.tight_layout()
            fname = f"noise_bsnr_{metric}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Зависимость " + mname
                    + r" от BSNR для разных алгоритмов.}" "\n"
                    r"\label{fig:noise_bsnr_" + metric + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)


#  Устойчивость к шуму - прирост относительно clean

def plot_noise_dependency_delta(
    df_global: pd.DataFrame,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures",
):
    """Аналог plot_noise_dependency, но по оси Y — средний ПРИРОСТ
    метрики относительно clean для той же пары (image, kernel).

    Логика:
        Для каждой строки с шумом подбирается «парная» clean-строка с тем же
        (algorithm, dataset, image_name, kernel_name). Если найдена —
        delta = metric_noise - metric_clean. Затем агрегируется среднее delta
        по (algorithm, noise_level) для каждого типа шума.

    Это снимает несбалансированность датасета по подвыборкам разных шумов.

    Графиков два семейства, как и в plot_noise_dependency:
        (1) столбчатая по уровням внутри одного типа шума;
        (2) линейный график delta / уровень шума;

    Имена файлов: noise_dependency_delta_*, noise_level_line_delta_*.
    """
    if 'noise_name' not in df_global.columns:
        print("  [delta] Столбец noise_name отсутствует.")
        return

    df_n = df_global[df_global['noise_name'].notna() & (df_global['noise_name'] != '')].copy()
    if df_n.empty:
        print("  [delta] Нет информации о шуме.")
        return

    parsed = df_n['noise_name'].map(_split_noise_name)
    df_n['noise_type'] = parsed.map(lambda t: t[0])
    df_n['noise_level'] = parsed.map(lambda t: t[1])

    levels_order = ['clean', 'weak', 'medium', 'strong']
    types = sorted(t for t in df_n['noise_type'].unique() if t not in ('clean', 'unknown'))
    if not types:
        print("  [delta] Только 'clean' — приростовые графики не строятся.")
        return

    pair_keys = ['algorithm', 'dataset', 'image_name', 'kernel_name']
    missing_keys = [k for k in pair_keys if k not in df_n.columns]
    if missing_keys:
        print(f"  [delta] Нет столбцов для пары: {missing_keys}. Пропуск.")
        return

    clean_rows = df_n[df_n['noise_level'] == 'clean'].copy()
    if clean_rows.empty:
        print("  [delta] Нет clean-строк — прирост посчитать не от чего.")
        return

    metrics_cols = [c for c in ('psnr', 'ssim') if c in df_n.columns]
    if not metrics_cols:
        print("  [delta] Нет столбцов psnr/ssim.")
        return

    clean_lookup = (clean_rows
                    .groupby(pair_keys)[metrics_cols]
                    .mean()
                    .reset_index()
                    .rename(columns={m: f"{m}_clean" for m in metrics_cols}))
    df_d = df_n.merge(clean_lookup, on=pair_keys, how='left')
    for m in metrics_cols:
        df_d[f"d_{m}"] = df_d[m] - df_d[f"{m}_clean"]

    df_d_noise = df_d[df_d['noise_level'] != 'clean'].copy()
    df_d_noise = df_d_noise.dropna(subset=[f"d_{m}" for m in metrics_cols], how='all')
    if df_d_noise.empty:
        print("  [delta] После парной привязки шумовых строк не осталось.")
        return

    algorithms = sorted(df_d_noise['algorithm'].unique())
    n_alg = len(algorithms)
    colors = _colormap_bars(n_alg)

    # Столбчатые графики: один на тип шума х метрику
    for metric in metrics_cols:
        dcol = f"d_{metric}"
        ylabel = ('Среднее изменение PSNR, дБ'
                  if metric == 'psnr' else
                  'Среднее изменение SSIM')
        for ntype in types:
            sub = df_d_noise[df_d_noise['noise_type'] == ntype].copy()
            if sub.empty:
                continue
            grouped = (sub.groupby(['algorithm', 'noise_level'])[dcol]
                          .mean().reset_index())
            present_levels = [lvl for lvl in levels_order
                              if lvl != 'clean' and lvl in grouped['noise_level'].unique()]
            n_lvl = len(present_levels)
            if n_lvl < 1:
                continue
            x_base = np.arange(n_lvl)
            bar_w = 0.7 / max(n_alg, 1)

            fig_w = max(n_lvl * n_alg * 0.32 + 3, 7)
            fig, ax = plt.subplots(figsize=(fig_w, 4.6))
            for idx, alg in enumerate(algorithms):
                sub_alg = grouped[grouped['algorithm'] == alg].set_index('noise_level')
                vals = [sub_alg.loc[lvl, dcol] if lvl in sub_alg.index else np.nan
                        for lvl in present_levels]
                ax.bar(x_base + idx * bar_w, vals, width=bar_w * 0.85,
                       color=colors[idx], alpha=0.95, label=decode(alg),
                       edgecolor='black', linewidth=0.4)
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
            ax.set_xticks(x_base + bar_w * (n_alg - 1) / 2)
            tick_labels = [decode(f"{lvl}{ntype}") for lvl in present_levels]
            ax.set_xticklabels(tick_labels, rotation=15, ha='right')
            ax.set_ylabel(ylabel)
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(
                f'Изменение {mname} от роста шума ({decode(ntype)})',
                fontsize=TITLE_FONTSIZE,
            )
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=10, borderaxespad=0.0)
            plt.tight_layout()

            fname = f"noise_dependency_delta_{metric}_{ntype}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Средний прирост " + mname
                    + r" относительно clean при шуме типа " + ntype
                    + r" (парно по (image, kernel)).}" "\n"
                    r"\label{fig:nd_delta_" + metric + "_" + ntype + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    # Линейный график: delta / уровень шума
    for metric in metrics_cols:
        dcol = f"d_{metric}"
        ylabel = ('Среднее изменение PSNR, дБ'
                  if metric == 'psnr' else
                  'Среднее изменение SSIM')
        for ntype in types:
            sub = df_d_noise[df_d_noise['noise_type'] == ntype].copy()
            if sub.empty:
                continue
            grouped = (sub.groupby(['algorithm', 'noise_level'])[dcol]
                          .mean().reset_index())
            present_levels = [lvl for lvl in levels_order
                              if lvl != 'clean' and lvl in grouped['noise_level'].unique()]
            if len(present_levels) < 2:
                continue
            x_idx = np.arange(len(present_levels))

            fig, ax = plt.subplots(figsize=(8, 5))
            colors_cycle = _get_palette_cycle()
            markers_cycle = _get_marker_cycle()
            for alg in algorithms:
                sub_alg = grouped[grouped['algorithm'] == alg].set_index('noise_level')
                vals = [sub_alg.loc[lvl, dcol] if lvl in sub_alg.index else np.nan
                        for lvl in present_levels]
                ax.plot(x_idx, vals, marker=next(markers_cycle), linewidth=2,
                        markersize=7, color=next(colors_cycle), label=decode(alg))
            ax.axhline(0.0, color='black', linewidth=0.8, alpha=0.7)
            tick_labels = [decode(f"{lvl}{ntype}") for lvl in present_levels]
            ax.set_xticks(x_idx)
            ax.set_xticklabels(tick_labels, rotation=15, ha='right')
            ax.set_xlabel('Уровень шума')
            ax.set_ylabel(ylabel)
            mname = 'PSNR' if metric == 'psnr' else 'SSIM'
            ax.set_title(
                f'Изменение {mname} от роста шума ({decode(ntype)})',
                fontsize=TITLE_FONTSIZE,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      fontsize=12, borderaxespad=0.0)
            plt.tight_layout()
            fname = f"noise_level_line_delta_{metric}_{ntype}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{"
                    + f"{fig_prefix}/{fname}.pdf" + r"}" "\n"
                    r"\caption{Прирост " + mname
                    + r" относительно clean в зависимости от уровня шума типа "
                    + ntype + r" (линейный график, парно по (image, kernel)).}" "\n"
                    r"\label{fig:nl_delta_" + metric + "_" + ntype + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)


# Зависимость от размера ядра

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
    ax1.set_xlabel('Размер ядра (k × k), пикс.')
    ax1.set_ylabel('Средний PSNR, дБ')
    ax1.set_title(f'PSNR по размеру ядра — {alg_label}', fontsize=TITLE_FONTSIZE)

    ax2.bar(grouped['ks_area'].astype(str), grouped['ssim_mean'], color=colors,
            edgecolor='grey', linewidth=0.4)
    ax2.set_xlabel('Размер ядра (k × k), пикс.')
    ax2.set_ylabel('Средний SSIM')
    ax2.set_title(f'SSIM по размеру ядра — {alg_label}', fontsize=TITLE_FONTSIZE)

    plt.tight_layout()
    if fig_dir:
        # fig.savefig(Path(fig_dir) / "kernel_size_dependency.pdf")
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

    grouped_ssim = df_w.groupby(['algorithm', 'ks_area']).agg(
        ssim_mean=('ssim', 'mean')
    ).reset_index()

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(16, 5))
    colors = _get_palette_cycle()
    colors_ssim = _get_palette_cycle()
    alg_names_ordered = grouped['algorithm'].unique()
    color_map = {a: next(colors) for a in alg_names_ordered}
    for alg_name in alg_names_ordered:
        next(colors_ssim)
    color_map_ssim = {a: c for a, c in zip(alg_names_ordered, [next(_get_palette_cycle()) for _ in alg_names_ordered])}
    color_map_ssim = color_map

    legend_handles = []
    for alg_name in alg_names_ordered:
        sub = grouped[grouped['algorithm'] == alg_name].sort_values('ks_area')
        line, = ax.plot(sub['ks_area'], sub['psnr_mean'], marker='o', linewidth=2,
                color=color_map[alg_name], label=decode(alg_name))
        legend_handles.append(line)
    ax.set_xlabel('Размер ядра (k × k), пикс.')
    ax.set_ylabel('Средний PSNR, дБ')
    ax.set_title('Зависимость PSNR от размера ядра', fontsize=TITLE_FONTSIZE)
    ax.grid(True, alpha=0.3)

    for alg_name in alg_names_ordered:
        sub = grouped_ssim[grouped_ssim['algorithm'] == alg_name].sort_values('ks_area')
        ax_s.plot(sub['ks_area'], sub['ssim_mean'], marker='o', linewidth=2,
                  color=color_map_ssim[alg_name], label=decode(alg_name))
    ax_s.set_xlabel('Размер ядра (k × k), пикс.')
    ax_s.set_ylabel('Средний SSIM')
    ax_s.set_title('Зависимость SSIM от размера ядра', fontsize=TITLE_FONTSIZE)
    ax_s.grid(True, alpha=0.3)

    fig.legend(handles=legend_handles,
               loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=12, borderaxespad=0.5, title='Алгоритм')
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if fig_dir:
        # fig.savefig(Path(fig_dir) / "kernel_size_psnr_cmp.pdf")
        fig.savefig(Path(fig_dir) / "kernel_size_psnr_cmp.png")
    plt.close(fig)

    # Отдельный график PSNR
    fig_p, ax_p = plt.subplots(figsize=(14, 6))
    legend_handles_p = []
    for alg_name in alg_names_ordered:
        sub = grouped[grouped['algorithm'] == alg_name].sort_values('ks_area')
        line, = ax_p.plot(sub['ks_area'], sub['psnr_mean'], marker='o', linewidth=2,
                          color=color_map[alg_name], label=decode(alg_name))
        legend_handles_p.append(line)
    ax_p.set_xlabel('Размер ядра (k × k), пикс.')
    ax_p.set_ylabel('Средний PSNR, дБ')
    ax_p.set_title('Зависимость PSNR от размера ядра', fontsize=TITLE_FONTSIZE)
    ax_p.grid(True, alpha=0.3)
    fig_p.legend(handles=legend_handles_p,
                 loc='center right', bbox_to_anchor=(1.0, 0.5),
                 fontsize=12, borderaxespad=0.5, title='Алгоритм')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    if fig_dir:
        fig_p.savefig(Path(fig_dir) / "kernel_size_psnr_only.png")
    plt.close(fig_p)

    # Отдельный график SSIM
    fig_s, ax_s2 = plt.subplots(figsize=(14, 6))
    legend_handles_s = []
    for alg_name in alg_names_ordered:
        sub = grouped_ssim[grouped_ssim['algorithm'] == alg_name].sort_values('ks_area')
        line, = ax_s2.plot(sub['ks_area'], sub['ssim_mean'], marker='o', linewidth=2,
                           color=color_map[alg_name], label=decode(alg_name))
        legend_handles_s.append(line)
    ax_s2.set_xlabel('Размер ядра (k × k), пикс.')
    ax_s2.set_ylabel('Средний SSIM')
    ax_s2.set_title('Зависимость SSIM от размера ядра', fontsize=TITLE_FONTSIZE)
    ax_s2.grid(True, alpha=0.3)
    fig_s.legend(handles=legend_handles_s,
                 loc='center right', bbox_to_anchor=(1.0, 0.5),
                 fontsize=12, borderaxespad=0.5, title='Алгоритм')
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    if fig_dir:
        fig_s.savefig(Path(fig_dir) / "kernel_size_ssim_only.png")
    plt.close(fig_s)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{"
            + f"{fig_prefix}/kernel_size_psnr_cmp.pdf" + r"}" "\n"
            r"\caption{Зависимость среднего PSNR и SSIM от размера ядра "
            r"для разных алгоритмов.}" "\n"
            r"\label{fig:ks_psnr_cmp}" "\n" r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "kernel_size_psnr_cmp.tex", tex)


# Боксплоты PSNR / SSIM

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

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(len(labels) * 1.2, 9), 5))

        def style_boxplot(bp, colors):
            for box, c in zip(bp['boxes'], colors):
                box.set_facecolor(c); box.set_alpha(0.7)
                box.set_edgecolor('grey'); box.set_linewidth(1.2)
            for median in bp['medians']:
                median.set_color('black'); median.set_linewidth(1.5)
            for flier, c in zip(bp['fliers'], colors):
                flier.set(marker='.', markerfacecolor=c, markeredgecolor=c, alpha=0.7, markersize=5)

        bp1 = ax1.boxplot(data_psnr, patch_artist=True)
        style_boxplot(bp1, box_colors)
        ax1.set_ylabel('PSNR, дБ')
        ax1.set_xlabel('(a) PSNR', fontsize=12)
        ax1.set_xticks([])
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        ax1.set_axisbelow(True)

        bp2 = ax2.boxplot(data_ssim, patch_artist=True)
        style_boxplot(bp2, box_colors)
        ax2.set_ylabel('SSIM')
        ax2.set_xlabel('(b) SSIM', fontsize=12)
        ax2.set_xticks([])
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        ax2.set_axisbelow(True)

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.7, edgecolor='grey',
                                 linewidth=1.2, label=l)
                           for c, l in zip(box_colors, labels)]
        ax2.legend(handles=legend_elements, loc='center left',
                   bbox_to_anchor=(1.02, 0.5), fontsize=12, borderaxespad=0.0)

        plt.tight_layout()
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"boxplot_{ds_name}.pdf")
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


#  Таблица средних PSNR / SSIM по наборам данных

import cv2 as cv

def build_summary_single(
    all_dataset_results: List[Dict],
    alg_label: str,
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Таблица средних метрик по наборам данных для одного алгоритма."""
    
    def calculate_stats(d: pd.DataFrame, ds_name: str) -> Optional[Dict]:
        if d.empty:
            return None
        
        # Поиск худшего и лучшего случая для PSNR
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
            
        # Поиск худшего и лучшего случая для SSIM
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

        # Вычисление исходных метрик и дельт
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
            'Отн. ошибок': round(d['error_ratio'].mean(), 2) if d['error_ratio'].notna().any() else '—',
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

    if all_dfs:
        d_total = pd.concat(all_dfs, ignore_index=True)
        stats_total = calculate_stats(d_total, "ВСЕ (Суммарно)")
        if stats_total:
            stats_total['Размер изображения'] = 'Различные' 
            rows.append(stats_total)

    def _size_sort_key(row):
        if row['Набор данных'] == 'ВСЕ (Суммарно)':
            return float('inf')
        s = row.get('Размер изображения', '—')
        try:
            parts = [int(x) for x in s.replace('x', 'X').split('X') if x.isdigit()]
            return max(parts) if parts else float('inf')
        except Exception:
            return float('inf')

    rows_sorted = sorted([r for r in rows if r['Набор данных'] != 'ВСЕ (Суммарно)'],
                         key=_size_sort_key)
    rows_total = [r for r in rows if r['Набор данных'] == 'ВСЕ (Суммарно)']
    rows = rows_sorted + rows_total

    df_s = pd.DataFrame(rows)

    if tex_dir:
        def safe_tex(val):
            return str(val).replace('_', r'\_')

        # Таблица PSNR
        tex_psnr = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Сводные результаты по PSNR: алгоритм " + safe_tex(alg_label)
            + r".}" "\n"
            r"\label{tab:summary_psnr_" + _safe_label(alg_label) + r"}" "\n"
            r"\resizebox{\textwidth}{!}{" "\n"
            r"\begin{tabular}{l c c | c c c c c l c l | c c}" "\n" r"\hline" "\n"
            r"Набор данных & Кол-во искаженных & Размер изображений"
            r" & Исходное (PSNR) & Среднее (PSNR) & Среднее $\Delta$ PSNR & Медиана (PSNR)"
            r" & Макс. (PSNR) & Лучший случай (PSNR) & Мин. (PSNR) & Худший случай (PSNR)"
            r" & Отн. ош. & Время \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_s.iterrows():
            p = r"\bfseries " if row['Набор данных'] == "ВСЕ (Суммарно)" else ""
            tex_psnr += (
                f"{p}{safe_tex(row['Набор данных'])} & "
                f"{p}{row['Количество искаженных изображений']} & "
                f"{p}{row['Размер изображения']} & "
                f"{p}{row['PSNR (исх.)']} & "
                f"{p}{row['Среднее (PSNR)']} & "
                f"{p}{row['Δ PSNR']} & "
                f"{p}{row['Медиана (PSNR)']} & "
                f"{p}{row['Максимум (PSNR)']} & "
                f"{p}{safe_tex(row['Лучший случай (PSNR)'])} & "
                f"{p}{row['Минимум (PSNR)']} & "
                f"{p}{safe_tex(row['Худший случай (PSNR)'])} & "
                f"{p}{row['Отн. ошибок']} & "
                f"{p}{row['Время (с)']} \\\\\n"
            )
        tex_psnr += r"\hline" "\n" r"\end{tabular}" "\n" r"}" "\n" r"\end{table}"
        save_tex(Path(tex_dir) / "summary_table_psnr.tex", tex_psnr)

        #  Таблица SSIM
        tex_ssim = (
            r"\begin{table}[htbp]" "\n" r"\centering" "\n"
            r"\caption{Сводные результаты по SSIM: алгоритм " + safe_tex(alg_label)
            + r".}" "\n"
            r"\label{tab:summary_ssim_" + _safe_label(alg_label) + r"}" "\n"
            r"\resizebox{\textwidth}{!}{" "\n"
            r"\begin{tabular}{l c c | c c c c c l c l}" "\n" r"\hline" "\n"
            r"Набор данных & Кол-во искаженных & Размер изображений"
            r" & Исходное (SSIM) & Среднее (SSIM) & Среднее $\Delta$ SSIM & Медиана (SSIM)"
            r" & Макс. (SSIM) & Лучший случай (SSIM) & Мин. (SSIM) & Худший случай (SSIM) \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_s.iterrows():
            p = r"\bfseries " if row['Набор данных'] == "ВСЕ (Суммарно)" else ""
            tex_ssim += (
                f"{p}{safe_tex(row['Набор данных'])} & "
                f"{p}{row['Количество искаженных изображений']} & "
                f"{p}{row['Размер изображения']} & "
                f"{p}{row['SSIM (исх.)']} & "
                f"{p}{row['Среднее (SSIM)']} & "
                f"{p}{row['Δ SSIM']} & "
                f"{p}{row['Медиана (SSIM)']} & "
                f"{p}{row['Максимум (SSIM)']} & "
                f"{p}{safe_tex(row['Лучший случай (SSIM)'])} & "
                f"{p}{row['Минимум (SSIM)']} & "
                f"{p}{safe_tex(row['Худший случай (SSIM)'])} \\\\\n"
            )
        tex_ssim += r"\hline" "\n" r"\end{tabular}" "\n" r"}" "\n" r"\end{table}"
        save_tex(Path(tex_dir) / "summary_table_ssim.tex", tex_ssim)

    return df_s


def build_table_mean_psnr_ssim(
    all_data: Dict[str, pd.DataFrame],
    datasets: List[str],
    tex_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Таблица средних PSNR / SSIM по наборам данных для разных методов."""
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
            r"\caption{Средние значения PSNR, дБ и SSIM для разных методов "
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


#  Итоговая количественная таблица

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
                'PSNR (сред.)': round(p.mean(), 2) if len(p) > 0 else '—',
                'PSNR (СКО)': round(p.std(), 2) if len(p) > 1 else '—',
                'SSIM (сред.)': round(s.mean(), 4) if len(s) > 0 else '—',
                'SSIM (СКО)': round(s.std(), 4) if len(s) > 1 else '—',
                'Отношение ошибок': round(er.mean(), 2) if len(er) > 0 else '—',
                'Доля успешных (%)': round((er <= 3).sum() / len(er) * 100, 1) if len(er) > 0 else '—',
                'Время (ср., сек)': round(t.mean(), 2) if len(t) > 0 else '—',
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
            r"SSIM & $\sigma_{\text{SSIM}}$ & Отн.ош. & Доля усп. & Время \\" "\n"
            r"\hline" "\n"
        )
        for _, row in df_f.iterrows():
            tex += (
                f"{row['Набор данных']} & {row['Алгоритм']} & {row['N']} & "
                f"{row['PSNR (сред.)']} & {row['PSNR (СКО)']} & "
                f"{row['SSIM (сред.)']} & {row['SSIM (СКО)']} & "
                f"{row['Отношение ошибок']} & {row['Доля успешных (%)']} & "
                f"{row['Время (ср., сек)']} \\\\\n"
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
    Генерирует и сохраняет сетку 2х4 со всеми этапами восстановления.
    Данная сетка вспомогательная, помогает сразу визуально оценить метрики и
    качество восстановления определенным методом конкретного изображения.
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

        c_gt = _crop_kernel_image(k_gt) if k_gt is not None else None
        c_est = _crop_kernel_image(k_est) if k_est is not None else None
        
        max_h, max_w = 0, 0
        if c_gt is not None:
            max_h, max_w = max(max_h, c_gt.shape[0]), max(max_w, c_gt.shape[1])
        if c_est is not None:
            max_h, max_w = max(max_h, c_est.shape[0]), max(max_w, c_est.shape[1])
        
        if c_gt is not None: c_gt = _pad_kernel_to_size(c_gt, max_h, max_w)
        if c_est is not None: c_est = _pad_kernel_to_size(c_est, max_h, max_w)

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.2, wspace=0.1)

        for ax in axes.flatten(): ax.axis('off')

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
            
            axes[0, 2].imshow(img_dist)
            axes[0, 2].set_title("Preprocessed Image", fontsize=12)

        if img_rest is not None:
            psnr_r = row.get('psnr', np.nan)
            ssim_r = row.get('ssim', np.nan)
            ps_str = f"{psnr_r:.4f}" if pd.notna(psnr_r) else "NaN"
            ss_str = f"{ssim_r:.4f}" if pd.notna(ssim_r) else "NaN"
            
            axes[0, 3].imshow(img_rest)
            axes[0, 3].set_title(f"{alg_label}\nPSNR: {ps_str} | SSIM: {ss_str}", fontsize=12)

        if c_gt is not None:
            axes[1, 1].imshow(c_gt, cmap='gray')
            axes[1, 1].set_title("original kernel", fontsize=12)

        if c_est is not None:
            axes[1, 3].imshow(c_est, cmap='gray')
            axes[1, 3].set_title(f"{alg_label} kernel", fontsize=12)

        plt.suptitle(dist_name, y=0.98, fontsize=14)
        plt.tight_layout()
        
        fig.savefig(save_dir / f"{Path(dist_name).stem}_complex.png", bbox_inches='tight')
        plt.close(fig)
        
    print(f"  Готово! Сохранено {len(df)} изображений в: {save_dir}")


def plot_boxplots_single(
    df_global: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """Боксплоты PSNR / SSIM для одного алгоритма (группировка по датасетам)."""
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
        flier.set(marker='.', markerfacecolor=c, markeredgecolor=c, alpha=0.7, markersize=5)
    ax1.set_xticklabels([decode(l) for l in labels], rotation=15)
    ax1.set_ylabel('PSNR, дБ')
    ax1.set_title(f'Распределение PSNR — {decode(alg_label)}', fontsize=TITLE_FONTSIZE)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # SSIM
    bp2 = ax2.boxplot(data_ssim, patch_artist=True)
    for box, c in zip(bp2['boxes'], box_colors):
        box.set_facecolor(c); box.set_alpha(0.7); box.set_edgecolor('grey'); box.set_linewidth(1.2)
    for median in bp2['medians']:
        median.set_color('black'); median.set_linewidth(1.5)
    for flier, c in zip(bp2['fliers'], box_colors):
        flier.set(marker='.', markerfacecolor=c, markeredgecolor=c, alpha=0.7, markersize=5)
    ax2.set_xticklabels([decode(l) for l in labels], rotation=15)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'Распределение SSIM — {decode(alg_label)}', fontsize=TITLE_FONTSIZE)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    if fig_dir:
        # fig.savefig(Path(fig_dir) / "boxplot_single.pdf")
        fig.savefig(Path(fig_dir) / "boxplot_single.png")
    #plt.show()


def save_kernel_profiles_and_diff(
    df: pd.DataFrame, 
    dist_dir: Path, 
    save_dir: Path, 
    alg_label: str,
    skip_existing: bool = True,
):
    """Сравнение истинного и оценённого ядра: 2D разность и 1D профили.

    skip_existing=True — не перегенерировать уже существующие PNG-профили.
    Экономия времени, так как КРАЙНЕ много ядер нужно обработать при каждом построении графиков.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        before = len(df)
        mask_todo = df['distorted_file'].apply(
            lambda dn: not (save_dir / f"{Path(str(dn)).stem}_kernel_profiles.png").exists()
        )
        df = df[mask_todo]
        if len(df) < before:
            print(f"  [{alg_label}] профили: пропущено {before - len(df)} уже сгенерированных")
        if df.empty:
            return

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

        c_gt = _crop_kernel_image(k_gt, padding=5)
        c_est = _crop_kernel_image(k_est, padding=5)
        
        if c_gt is None or c_est is None: continue
        
        max_h = max(c_gt.shape[0], c_est.shape[0])
        max_w = max(c_gt.shape[1], c_est.shape[1])
        
        c_gt = _pad_kernel_to_size(c_gt, max_h, max_w).astype(np.float32)
        c_est = _pad_kernel_to_size(c_est, max_h, max_w).astype(np.float32)

        if c_gt.sum() > 0: c_gt /= c_gt.sum()
        if c_est.sum() > 0: c_est /= c_est.sum()

        diff_map = c_est - c_gt
        
        cy, cx = max_h // 2, max_w // 2
        prof_gt_h, prof_est_h = c_gt[cy, :], c_est[cy, :]
        prof_gt_v, prof_est_v = c_gt[:, cx], c_est[:, cx]

        fig = plt.figure(figsize=(18, 4))
        
        # Истинное ядро
        ax1 = plt.subplot(1, 5, 1)
        ax1.imshow(c_gt, cmap='gray')
        ax1.set_title("Истинное ядро")
        ax1.axis('off')

        # Оценённое ядро
        ax2 = plt.subplot(1, 5, 2)
        ax2.imshow(c_est, cmap='gray')
        ax2.set_title(f"Оценка ({decode(alg_label)})")
        ax2.axis('off')

        # Карта разности
        ax3 = plt.subplot(1, 5, 3)
        vmax = max(abs(diff_map.min()), abs(diff_map.max()))
        im3 = ax3.imshow(diff_map, cmap='bwr', vmin=-vmax, vmax=vmax)
        ax3.set_title("Разность (оценка − истина)")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Горизонтальный профиль
        ax4 = plt.subplot(1, 5, 4)
        ax4.plot(prof_gt_h, label='Истинное', color='black', linestyle='--')
        ax4.plot(prof_est_h, label='Оценка', color='red')
        ax4.set_title("Горизонтальный профиль")
        ax4.legend(fontsize=8)

        # Вертикальный профиль
        ax5 = plt.subplot(1, 5, 5)
        ax5.plot(prof_gt_v, label='Истинное', color='black', linestyle='--')
        ax5.plot(prof_est_v, label='Оценка', color='red')
        ax5.set_title("Вертикальный профиль")

        plt.tight_layout()
        fig.savefig(save_dir / f"{Path(dist_name).stem}_kernel_profiles.png", bbox_inches='tight')
        plt.close(fig)


def plot_3d_applicability_map(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    complexity_dataset_name: str = "Complexity_Test",
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    fig_prefix: str = "comparison_figures"
):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = _get_palette_cycle()

    legend_handles =[]
    
    for alg_name, df_alg in all_data.items():
        df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
        if df_grid.empty:
            continue
            
        if 'psnr_blurred' in df_grid.columns:
            df_grid['success'] = (df_grid['error_ratio'] < 3.0) & (df_grid['psnr'] >= df_grid['psnr_blurred'] + 1.0)
        else:
            df_grid['success'] = (df_grid['error_ratio'] < 3.0)

        success_rate = df_grid['success'].mean() * 100
        df_success = df_grid[df_grid['success']]
        
        if df_success.empty:
            continue

        if 'ssim_blurred' in df_success.columns:
            x_vals = 1.0 - df_success['ssim_blurred'].dropna()
            x_coord = np.percentile(x_vals, 90) if len(x_vals) > 0 else 0
        elif 'kernel_shape' in df_success.columns:
            df_success['ks_area'] = df_success['kernel_shape'].apply(_parse_kernel_area)
            x_coord = np.percentile(df_success['ks_area'].dropna(), 90)
        else:
            x_coord = 0
        if 'psnr_blurred' in df_success.columns:
            y_vals = 40.0 - df_success['psnr_blurred'].dropna()
            y_coord = np.percentile(y_vals, 90) if len(y_vals) > 0 else 0
        else:
            y_coord = 0

        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
        if not df_comp.empty:
            if 'image_megapixels' in df_comp.columns:
                mask512 = (df_comp['image_megapixels'] - 0.262144).abs() < 0.05
                df_comp512 = df_comp[mask512] if mask512.any() else df_comp
            else:
                df_comp512 = df_comp
            z_coord = df_comp512['time_sec'].median()
        else:
            z_coord = df_grid['time_sec'].median()

        c = next(colors)
        
        scatter = ax.scatter(
            x_coord, y_coord, z_coord, 
            s=max(success_rate * 5, 50),
            c=[c], alpha=0.8, edgecolors='black', linewidth=1
        )
        
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=c, markersize=10, 
                                         label=f"{alg_name} (SR: {success_rate:.1f}%)"))

        ax.text(x_coord, y_coord, z_coord + (z_coord * 0.05), alg_name, 
                fontsize=8, ha='center', va='bottom')

    ax.set_xlabel('Сложность смаза (X)', labelpad=8)
    ax.set_ylabel('Сложность шума (Y)', labelpad=8)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Время, сек (Z)', labelpad=12, rotation=90)
    ax.tick_params(axis='z', pad=4)
    ax.set_title('Область применимости алгоритмов', fontsize=TITLE_FONTSIZE)
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    fig.subplots_adjust(left=0.02, right=0.78, top=0.95, bottom=0.05)
    if fig_dir:
        # fig.savefig(Path(fig_dir) / "3d_applicability_map.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / "3d_applicability_map.png", bbox_inches='tight')
    #plt.show()


def plot_time_quality_pareto(
    all_data: Dict[str, pd.DataFrame],
    dataset_name: str = "Grid_Test",
    fig_dir: Optional[Path] = None,
    target_size_px: int = 512,
    complexity_dataset_name: Optional[str] = None,
):
    """Trade-off: Время / Качество.

    Метрики (SSIM, PSNR) берутся из dataset_name (у нас был Levin).
    Время берётся из complexity_dataset_name (у нас время было только в Complexity_Test)
    с фильтром на изображения размером target_size_px х target_size_px.
    Если complexity_dataset_name не задан — время берётся из dataset_name.
    """
    fig, (ax, ax_p) = plt.subplots(1, 2, figsize=(14, 6))
    
    points = []
    names = []
    colors_list = []
    colors_cycle = _get_palette_cycle()
    target_mp = (target_size_px ** 2) / 1e6
    
    for alg_name, df_alg in all_data.items():
        df_metrics = df_alg[df_alg['dataset'] == dataset_name].copy()
        if df_metrics.empty:
            continue
        mean_ssim = df_metrics['ssim'].mean()
        mean_psnr = df_metrics['psnr'].mean()

        if complexity_dataset_name:
            df_time = df_alg[df_alg['dataset'] == complexity_dataset_name].copy()
        else:
            df_time = df_metrics.copy()

        if df_time.empty:
            # время из датасета, если нет отдельного теста на время
            df_time = df_metrics.copy()

        df_time_used = df_time
        if 'image_megapixels' in df_time.columns and df_time['image_megapixels'].notna().any():
            mask = (df_time['image_megapixels'] - target_mp).abs() < 0.05
            if mask.any():
                df_time_used = df_time[mask]

        median_time = df_time_used['time_sec'].median()
        
        if pd.notna(mean_ssim) and pd.notna(median_time):
            points.append((median_time, mean_ssim, mean_psnr))
            names.append(alg_name)
            colors_list.append(next(colors_cycle))
            
    if not points:
        print("Нет данных для Pareto графика")
        plt.close(fig)
        return
        
    times, ssims, psnrs = zip(*points)

    legend_handles = []
    for i in range(len(points)):
        for cur_ax, vals in [(ax, ssims), (ax_p, psnrs)]:
            cur_ax.scatter(times[i], vals[i], color=colors_list[i], s=200,
                           edgecolors='black', linewidth=0.8)
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                         markerfacecolor=colors_list[i],
                                         markeredgecolor='black', markersize=12,
                                         label=decode(names[i])))

    for cur_ax in (ax, ax_p):
        cur_ax.set_xscale('log')
        cur_ax.set_xlabel('Медианное время работы, с (log)', fontsize=13)
        cur_ax.grid(True, alpha=0.3)

    ax.set_ylabel('Средний SSIM', fontsize=13)
    ax_p.set_ylabel('Средний PSNR, дБ', fontsize=13)

    ds_label = dataset_name
    sz_label = str(target_size_px)
    ax.set_title(f'Качество ({ds_label}) против скорости ({sz_label}) — SSIM',
                 fontsize=TITLE_FONTSIZE)
    ax_p.set_title(f'Качество ({ds_label}) против скорости ({sz_label}) — PSNR',
                   fontsize=TITLE_FONTSIZE)
    
    for cur_ax, vals in [(ax, ssims), (ax_p, psnrs)]:
        sorted_pts = sorted(zip(times, vals), key=lambda x: x[0])
        pareto_x, pareto_y, max_v = [], [], -1e9
        for t, v in sorted_pts:
            if v > max_v:
                pareto_x.append(t); pareto_y.append(v); max_v = v
        pareto_line = cur_ax.plot(pareto_x, pareto_y, 'r--',
                                  linewidth=1.5, alpha=0.7, label='Парето-фронт')[0]

    legend_handles.append(pareto_line)
    
    fig.legend(handles=legend_handles, loc='center left',
               bbox_to_anchor=(0.84, 0.5), fontsize=12, borderaxespad=0.5)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    
    if fig_dir:
        # fig.savefig(Path(fig_dir) / "time_quality_pareto.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / "time_quality_pareto.png", bbox_inches='tight')
    plt.close(fig)

def plot_scalability_comparison(
    all_data: Dict[str, pd.DataFrame],
    complexity_dataset_name: str = "Complexity_Test",
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None
):
    """График вычислительной сложности: время от размера изображения."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = _get_palette_cycle()
    markers = _get_marker_cycle()
    
    has_data = False
    for alg_name, df_alg in all_data.items():
        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name].copy()
        if df_comp.empty or 'image_megapixels' not in df_comp.columns:
            continue
        df_comp['side_px'] = (df_comp['image_megapixels'] * 1e6).clip(lower=0).pow(0.5).round().astype(int)
        grouped = df_comp.groupby('side_px')['time_sec'].mean().reset_index()
        grouped = grouped.sort_values('side_px')
        
        if not grouped.empty:
            has_data = True
            color = next(colors); marker = next(markers)
            ax.plot(grouped['side_px'], grouped['time_sec'], 
                    marker=marker, markersize=8, linewidth=2, 
                    color=color, label=decode(alg_name))

    if has_data:
        ax.set_xlabel('Размер изображения, пикс.', fontsize=13)
        ax.set_ylabel('Время работы, секунды', fontsize=13)
        ax.set_title('Масштабируемость алгоритмов', fontsize=TITLE_FONTSIZE)
        all_sides = sorted({int(s) for _, df_alg in all_data.items()
                            for s in (df_alg[df_alg['dataset'] == complexity_dataset_name]
                                      .get('image_megapixels', pd.Series(dtype=float))
                                      .dropna() * 1e6).clip(lower=0).pow(0.5).round().astype(int)
                            if int(s) > 0})
        if all_sides:
            ax.set_xticks(all_sides)
            ax.set_xticklabels([str(s) for s in all_sides], fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=12, borderaxespad=0.0)
        ax.grid(True, ls="--", alpha=0.5)
        
        plt.tight_layout()
        if fig_dir:
            # fig.savefig(Path(fig_dir) / "scalability_plot.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / "scalability_plot.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_speed_vs_size_single(
    df_alg: pd.DataFrame,
    alg_label: str,
    complexity_dataset_name: str = "Complexity_Test",
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """График зависимости времени работы от размера изображения для одного
    алгоритма. Источник данных - Complexity_Test.
    """
    df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name].copy()
    if df_comp.empty or 'image_megapixels' not in df_comp.columns:
        return
    df_comp['side_px'] = (df_comp['image_megapixels'] * 1e6).clip(lower=0).pow(0.5).round().astype(int)
    grouped = (df_comp.groupby('side_px')['time_sec']
                       .agg(['mean', 'std', 'count'])
                       .reset_index()
                       .sort_values('side_px'))
    if grouped.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped['side_px'], grouped['mean'],
            marker='o', markersize=7, linewidth=2, color=PALETTE[0],
            label=decode(alg_label))
    if grouped['std'].notna().any():
        ax.fill_between(grouped['side_px'],
                        grouped['mean'] - grouped['std'].fillna(0),
                        grouped['mean'] + grouped['std'].fillna(0),
                        color=PALETTE[0], alpha=0.18, label='±1σ')
    ax.set_xticks(grouped['side_px'].tolist())
    ax.set_xticklabels([str(int(s)) for s in grouped['side_px']])
    ax.set_xlabel('Размер изображения, пикс.')
    ax.set_ylabel('Время работы, с')
    ax.set_title(f'Скорость работы от размера изображения — {decode(alg_label)}',
                 fontsize=TITLE_FONTSIZE)
    ax.grid(True, ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()

    safe = _safe_label(alg_label)
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"speed_vs_size_{safe}.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / f"speed_vs_size_{safe}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.85\textwidth]{figures/"
            + f"speed_vs_size_{safe}" + r".pdf}" "\n"
            r"\caption{Время работы алгоритма " + decode(alg_label)
            + r" в зависимости от размера изображения (датасет "
            + complexity_dataset_name.replace("_", r"\_") + r").}" "\n"
            r"\label{fig:speed_vs_size_" + safe + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"speed_vs_size_{safe}.tex", tex)


def _calculate_xy_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Вычисляет оси X и Y строго по одной метрике.
    Для вычисления "базового" (чистого от шума) смаза использует пути
    из таблицы для оригинального изображения и ядра, выполняя свертку на лету."""
    
    import cv2 as cv
    import numpy as np
    from pathlib import Path
    
    try:
        from skimage.metrics import peak_signal_noise_ratio as compare_psnr
        from skimage.metrics import structural_similarity as compare_ssim
    except ImportError:
        print("Внимание: Для вычисления базовых метрик требуется scikit-image (pip install scikit-image)")
        return df

    def calc_ssim_safe(img1, img2):
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

        if pd.isna(orig_path) or pd.isna(kernel_path) or not orig_path or not kernel_path:
            return np.nan
        if not Path(orig_path).exists() or not Path(kernel_path).exists():
            return np.nan

        img_orig = cv.imread(str(orig_path), cv.IMREAD_COLOR)
        kernel = cv.imread(str(kernel_path), cv.IMREAD_GRAYSCALE)

        if img_orig is None or kernel is None:
            return np.nan

        kernel = kernel.astype(np.float32)
        k_sum = kernel.sum()
        if k_sum > 0:
            kernel /= k_sum

        img_orig_float = img_orig.astype(np.float32)
        img_blurred_clean = cv.filter2D(img_orig_float, -1, kernel, borderType=cv.BORDER_REPLICATE)
        img_blurred_clean = np.clip(img_blurred_clean, 0, 255).astype(np.uint8)

        if metric == 'ssim':
            return calc_ssim_safe(img_orig, img_blurred_clean)
        else:
            return compare_psnr(img_orig, img_blurred_clean, data_range=255)

    df['clean_ref'] = df.apply(get_clean_ref, axis=1)

    if df['clean_ref'].isna().any():
        fallback_val = df['clean_ref'].median()
        if pd.isna(fallback_val):
            fallback_val = 1.0 if metric == 'ssim' else 40.0
        df['clean_ref'] = df['clean_ref'].fillna(fallback_val)

    # успех: (error_ratio < 3) и (PSNR вост. >= PSNR смаз. + 1 дБ)
    # ось X: 40.0 - PSNR смаз без шума
    # ось Y: (PSNR смаз без шума - PSNR смаз с шумом) / PSNR смаз без шума
    # успех: (error_ratio < 3) и (SSIM вост. >= SSIM смаз. + 0.02)
    # ось X: 40.0 - SSIM смаз без шума
    # ось Y: (SSIM смаз без шума - SSIM смаз с шумом) / SSIM смаз без шума
    if metric == 'ssim':
        df['X'] = 1.0 - df['clean_ref']
        
        df['Y'] = (df['clean_ref'] - df['ssim_blurred']) / (df['clean_ref'] + 1e-6)
        # df['Y'] = (df['clean_ref'] - df['ssim_blurred'])

        df['Success'] = (df['error_ratio'] < 3.0) & (df['ssim'] >= df['ssim_blurred'] + 0.02)
        
    elif metric == 'psnr':
        df['X'] = 40.0 - df['clean_ref']
        df['Y'] = (df['clean_ref'] - df['psnr_blurred']) / (df['clean_ref'] + 1e-6)
        # df['Y'] = (df['clean_ref'] - df['psnr_blurred'])

        df['Success'] = (df['error_ratio'] < 3.0) & (df['psnr'] >= df['psnr_blurred'] + 1.0)

    df['Y'] = df['Y'].clip(lower=0)

    return df


def plot_2d_working_areas(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    metric: str = "ssim",  # 'ssim' или 'psnr'
    fig_dir: Optional[Path] = None
):
    """Строит 2D-рабочие зоны для всех алгоритмов."""
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
    
    prepared = {alg: _calculate_xy_metrics(df_grid.copy(), metric)
                for alg, df_grid in algs_with_data.items()}

    for i, (alg_name, df_grid) in enumerate(prepared.items()):
        ax = axes[i]
        success_df = df_grid[df_grid['Success']]
        fail_df = df_grid[~df_grid['Success']]
        
        x_90 = np.percentile(success_df['X'].dropna(), 90) if len(success_df['X'].dropna()) > 0 else 0
        y_90 = np.percentile(success_df['Y'].dropna(), 90) if len(success_df['Y'].dropna()) > 0 else 0
        
        ax.scatter(fail_df['X'], fail_df['Y'], c='red', marker='x', s=22, alpha=0.5, label='Провал')
        ax.scatter(success_df['X'], success_df['Y'], c='green', marker='o', s=22, alpha=0.8, edgecolor='black', linewidth=0.4, label='Успех')
        
        rect = patches.Rectangle((0, 0), x_90, y_90, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, label='Рабочая зона (90%)')
        ax.add_patch(rect)
        
        ax.set_title(f"Рабочая область: {decode(alg_name)} ({metric.upper()})", fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0:
            ax.legend(loc='upper left')
            
    plt.tight_layout()
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"2d_working_areas_{metric}.pdf", bbox_inches='tight')
        fig.savefig(Path(fig_dir) / f"2d_working_areas_{metric}.png", bbox_inches='tight')
    plt.close(fig)

    if fig_dir:
        for alg_name, df_grid in prepared.items():
            success_df = df_grid[df_grid['Success']]
            fail_df = df_grid[~df_grid['Success']]
            x_90 = np.percentile(success_df['X'].dropna(), 90) if len(success_df['X'].dropna()) > 0 else 0
            y_90 = np.percentile(success_df['Y'].dropna(), 90) if len(success_df['Y'].dropna()) > 0 else 0

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(fail_df['X'], fail_df['Y'], c='red', marker='x', s=22, alpha=0.6, label='Провал')
            ax.scatter(success_df['X'], success_df['Y'], c='green', marker='o', s=22, alpha=0.85, edgecolor='black', linewidth=0.4, label='Успех')
            rect = patches.Rectangle((0, 0), x_90, y_90, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.1, label='Рабочая зона (90%)')
            ax.add_patch(rect)
            ax.set_title(f"Рабочая область: {decode(alg_name)} ({metric.upper()})", fontsize=13)
            ax.set_xlabel(x_label); ax.set_ylabel(y_label)
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left')
            plt.tight_layout()
            safe = _safe_label(alg_name)
            # fig.savefig(Path(fig_dir) / f"2d_working_area_{metric}_{safe}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"2d_working_area_{metric}_{safe}.png", bbox_inches='tight')
            plt.close(fig)


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
            
        df_grid = _calculate_xy_metrics(df_grid, metric)

        success_rate = df_grid['Success'].mean() * 100
        df_success = df_grid[df_grid['Success']]
        if df_success.empty: continue

        x_coord = np.percentile(df_success['X'].dropna(), 90) if len(df_success['X'].dropna()) > 0 else 0
        y_coord = np.percentile(df_success['Y'].dropna(), 90) if len(df_success['Y'].dropna()) > 0 else 0

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


def plot_3d_applicability_4_angles(
    all_data: Dict[str, pd.DataFrame],
    grid_dataset_name: str = "Grid_Test",
    complexity_dataset_name: str = "Complexity_Test",
    metric: str = "ssim",
    fig_dir: Optional[Path] = None,
    show_labels: bool = True,
):
    """Строит и сохраняет 4 отдельных 3D-графика области приминимости с разных углов обзора."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plot_data = []
    colors = _get_palette_cycle()
    
    for alg_name, df_alg in all_data.items():
        df_grid = df_alg[df_alg['dataset'] == grid_dataset_name].copy()
        if df_grid.empty: 
            continue
            
        df_grid = _calculate_xy_metrics(df_grid, metric)
        success_rate = df_grid['Success'].mean() * 100

        y_thr = df_grid['Y'].quantile(0.33) if df_grid['Y'].notna().any() else 0.0
        x_thr = df_grid['X'].quantile(0.33) if df_grid['X'].notna().any() else 0.0

        df_x_sub = df_grid[(df_grid['Y'] <= y_thr) & df_grid['Success']]
        df_y_sub = df_grid[(df_grid['X'] <= x_thr) & df_grid['Success']]

        x_coord = (np.percentile(df_x_sub['X'].dropna(), 90)
                   if len(df_x_sub['X'].dropna()) > 0 else 0)
        y_coord = (np.percentile(df_y_sub['Y'].dropna(), 90)
                   if len(df_y_sub['Y'].dropna()) > 0 else 0)

        if not df_grid['Success'].any():
            continue

        df_comp = df_alg[df_alg['dataset'] == complexity_dataset_name]
        if not df_comp.empty and 'image_megapixels' in df_comp.columns:
            mask512 = (df_comp['image_megapixels'] - 0.262144).abs() < 0.05
            df_comp512 = df_comp[mask512] if mask512.any() else df_comp
        else:
            df_comp512 = df_comp
        z_coord = df_comp512['time_sec'].median() if not df_comp512.empty else df_grid['time_sec'].median()
        
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

    if metric == 'ssim':
        x_label = "Сложность смаза\n(1 − SSIM смазанного)"
        y_label = "Влияние шума\n|SSIM смазанного − SSIM искажённого| / SSIM смазанного"
    else:
        x_label = "Сложность смаза\n(40 − PSNR смазанного)"
        y_label = "Влияние шума\n|PSNR смазанного − PSNR искажённого| / PSNR смазанного"
    z_label = "Время, сек"

    angles = [
        (30, 45),
        (30, 135),
        (30, 225),
        (30, 315)
    ]
    
    for i, (elev, azim) in enumerate(angles):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection='3d')
        legend_handles = []
        
        for item in plot_data:
            size = max(item['sr'] * 3, 60)
            ax.scatter(item['x'], item['y'], item['z'], 
                       s=size, c=[item['color']], alpha=0.9, edgecolors='black', linewidth=0.6)
            
            if show_labels:
                z_offset = item['z'] + (max([d['z'] for d in plot_data]) * 0.05) if plot_data else item['z']
                ax.text(item['x'], item['y'], z_offset, 
                        decode(item['alg']), fontsize=9, ha='center', va='bottom')
            
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=item['color'],
                           markersize=11,
                           label=f"SR: {item['sr']:.1f}%")
            )

        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"{metric.upper()}", fontsize=14)
        ax.set_xlabel(x_label, labelpad=12)
        ax.set_ylabel(y_label, labelpad=12)
        ax.set_zlabel(z_label, labelpad=12)
        ax.tick_params(axis='z', pad=6)

        fig.subplots_adjust(left=0.05, right=0.76, top=0.93, bottom=0.10)
        fig.legend(handles=legend_handles, loc='center left',
                   bbox_to_anchor=(0.77, 0.5), fontsize=11, borderaxespad=0,
                   handlelength=1.2, handletextpad=0.5, ncol=1)
        
        if fig_dir:
            out_path = Path(fig_dir) / f"3d_applicability_{metric}_angle_{azim}.png"
            fig.savefig(out_path, dpi=300)
            out_pdf = Path(fig_dir) / f"3d_applicability_{metric}_angle_{azim}.pdf"
            fig.savefig(out_pdf)
            print(f"  [{metric.upper()}] Сохранен ракурс {azim}° -> {out_path.name}")
        plt.close(fig)

    if fig_dir and plot_data:
        _leg_handles = []
        for _item in plot_data:
            _leg_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=_item['color'], markersize=10,
                           label=decode(_item['alg']))
            )
        _leg_handles.append(plt.Line2D([0], [0], color='none', label=''))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Ось X (PSNR):  40 − PSNR смазанного'))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Ось Y (PSNR):  |PSNR смазанного − PSNR искажённого| / PSNR смазанного'))
        _leg_handles.append(plt.Line2D([0], [0], color='none', label=''))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Ось X (SSIM):  1 − SSIM смазанного'))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Ось Y (SSIM):  |SSIM смазанного − SSIM искажённого| / SSIM смазанного'))
        _leg_handles.append(plt.Line2D([0], [0], color='none', label=''))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Ось Z:  медианное время обработки 512×512, с'))
        _leg_handles.append(plt.Line2D([0], [0], color='none', label=''))
        _leg_handles.append(plt.Line2D([0], [0], color='none',
            label='Размер точки — доля успешных восстановлений (SR)'))
        _fig_h = max(len(plot_data) * 0.38 + 3.2, 4.0)
        _fig_leg = plt.figure(figsize=(8.5, _fig_h))
        _ax_leg = _fig_leg.add_subplot(111)
        _ax_leg.set_axis_off()
        _fig_leg.legend(handles=_leg_handles, loc='center', fontsize=9,
                        handletextpad=0.6, borderaxespad=0,
                        frameon=True, edgecolor='#aaaaaa')
        _fig_leg.tight_layout()
        _leg_png = Path(fig_dir) / '3d_applicability_legend.png'
        _leg_pdf = Path(fig_dir) / '3d_applicability_legend.pdf'
        _fig_leg.savefig(_leg_png, dpi=200, bbox_inches='tight')
        _fig_leg.savefig(_leg_pdf, bbox_inches='tight')
        plt.close(_fig_leg)
        print(f"  [{metric.upper()}] Легенда -> {_leg_png.name}")

        _sr_handles = []
        for _item in plot_data:
            _sr_handles.append(
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=_item['color'], markersize=10,
                           label=f"{decode(_item['alg'])}  —  SR: {_item['sr']:.1f}%")
            )
        _sr_fig_h = max(len(plot_data) * 0.38 + 0.6, 2.5)
        _fig_sr = plt.figure(figsize=(6.0, _sr_fig_h))
        _ax_sr = _fig_sr.add_subplot(111)
        _ax_sr.set_axis_off()
        _fig_sr.legend(handles=_sr_handles, loc='center', fontsize=10,
                       handletextpad=0.6, borderaxespad=0,
                       frameon=True, edgecolor='#aaaaaa',
                       title=f"Success Rate ({metric.upper()})", title_fontsize=10)
        _fig_sr.tight_layout()
        _sr_png = Path(fig_dir) / f'3d_applicability_legend_{metric}_sr.png'
        _fig_sr.savefig(_sr_png, dpi=200, bbox_inches='tight')
        plt.close(_fig_sr)
        print(f"  [{metric.upper()}] SR-легенда -> {_sr_png.name}")

    if fig_dir:
        try:
            import plotly.graph_objects as go  # type: ignore
            import matplotlib.colors as mcolors
            traces = []
            for item in plot_data:
                r, g, b = mcolors.to_rgb(item['color'])
                cstr = f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                traces.append(go.Scatter3d(
                    x=[item['x']], y=[item['y']], z=[item['z']],
                    mode='markers+text',
                    marker=dict(size=max(item['sr'] / 4, 6),
                                color=cstr, line=dict(color='black', width=0.5)),
                    text=[f"{decode(item['alg'])} (SR={item['sr']:.1f}%)"],
                    textposition='top center',
                    name=f"{decode(item['alg'])} (SR={item['sr']:.1f}%)",
                ))
            if metric == 'ssim':
                _formula_x = 'X: 1 − SSIM смазанного'
                _formula_y = 'Y: |SSIM смазанного − SSIM искажённого| / SSIM смазанного'
            else:
                _formula_x = 'X: 40 − PSNR смазанного'
                _formula_y = 'Y: |PSNR смазанного − PSNR искажённого| / PSNR смазанного'
            _formula_z = 'Z: время обработки 512×512, с'
            for _lbl in ['', _formula_x, _formula_y, _formula_z]:
                traces.append(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=0, opacity=0),
                    name=_lbl,
                    showlegend=True,
                ))
            fig_html = go.Figure(data=traces)
            fig_html.update_layout(
                title=f"Область применимости ({metric.upper()})",
                scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
                width=1100, height=820,
                legend=dict(itemsizing='constant', tracegroupgap=0),
            )
            html_path = Path(fig_dir) / f"3d_applicability_{metric}_interactive.html"
            fig_html.write_html(str(html_path), include_plotlyjs='cdn')
            print(f"  [{metric.upper()}] Интерактивный HTML -> {html_path.name}")

            _JS_ROTATE = r"""
<script>
(function() {
  var gd = document.querySelector('.plotly-graph-div');
  var azim = 45;
  var elev = 30;
  var speed = 0.4;   // градусов за кадр
  var running = true;

  function step() {
    if (!running) { requestAnimationFrame(step); return; }
    azim = (azim + speed) % 360;
    var rad = azim * Math.PI / 180;
    var elevRad = elev * Math.PI / 180;
    Plotly.relayout(gd, {
      'scene.camera': {
        eye: {
          x: 2.0 * Math.cos(rad) * Math.cos(elevRad),
          y: 2.0 * Math.sin(rad) * Math.cos(elevRad),
          z: 2.0 * Math.sin(elevRad)
        }
      }
    });
    requestAnimationFrame(step);
  }

  // Пауза/продолжение по клику
  gd.addEventListener('click', function() { running = !running; });

  requestAnimationFrame(step);
})();
</script>
"""
            rotating_path = Path(fig_dir) / f"3d_applicability_{metric}_rotating.html"
            raw_html = fig_html.to_html(include_plotlyjs='cdn', full_html=True)
            raw_html = raw_html.replace('</body>', _JS_ROTATE + '\n</body>')
            rotating_path.write_text(raw_html, encoding='utf-8')
            print(f"  [{metric.upper()}] Вращающийся HTML -> {rotating_path.name}")
        except ImportError:
            print(f"  [{metric.upper()}] plotly не установлен; HTML пропущен.")
        except Exception as e:
            print(f"  [{metric.upper()}] Ошибка plotly: {e}")


def _load_image_rgb(path) -> Optional[np.ndarray]:
    """Загружает изображение как RGB-массив uint8. Возвращает None при ошибке."""
    try:
        import cv2
        p = str(path)
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def _load_image_gray(path) -> Optional[np.ndarray]:
    """Загружает изображение как grayscale RGB (3-канальный серый) uint8."""
    try:
        import cv2
        p = str(path)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    except Exception:
        return None


def _load_kernel_img(path) -> Optional[np.ndarray]:
    """Загружает ядро как grayscale float, нормализует в [0,1]."""
    try:
        import cv2
        p = str(path)
        if p.endswith('.npy'):
            k = np.load(p)
            k = k.astype(float)
        else:
            k = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if k is None:
                return None
            k = k.astype(float)
        mn, mx = k.min(), k.max()
        if mx > mn:
            k = (k - mn) / (mx - mn)
        else:
            k = np.zeros_like(k, dtype=float)
        return k
    except Exception:
        return None


def _pad_kernel_to_square(k: np.ndarray, target: int) -> np.ndarray:
    """Центрирует ядро на чёрном фоне размером target х target."""
    h, w = k.shape[:2]
    if h == target and w == target:
        return k
    out = np.zeros((target, target), dtype=k.dtype)
    y0 = (target - h) // 2
    x0 = (target - w) // 2
    out[y0:y0 + h, x0:x0 + w] = k
    return out


def _crop_kernels_to_content(*kernels, threshold: float = 0.02, margin: int = 2):
    """
    Обрезает набор ядер (одинакового размера) по объединённому bbox значимых пикселей.
    Crop делается квадратным (по максимальной стороне bbox).
    None-ядра пропускаются при вычислении bbox и возвращаются как None.
    """
    valid = [k for k in kernels if k is not None]
    if not valid:
        return kernels
    h, w = valid[0].shape[:2]
    combined = np.zeros((h, w), dtype=float)
    for k in valid:
        combined = np.maximum(combined, k.astype(float))
    rows_mask = np.any(combined > threshold, axis=1)
    cols_mask = np.any(combined > threshold, axis=0)
    if not rows_mask.any():
        return kernels
    rmin = int(np.where(rows_mask)[0][0])
    rmax = int(np.where(rows_mask)[0][-1])
    cmin = int(np.where(cols_mask)[0][0])
    cmax = int(np.where(cols_mask)[0][-1])
    rmin = max(0, rmin - margin)
    rmax = min(h - 1, rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(w - 1, cmax + margin)
    size = max(rmax - rmin + 1, cmax - cmin + 1)
    rc = (rmin + rmax) // 2
    cc = (cmin + cmax) // 2
    r0 = max(0, rc - size // 2)
    r1 = min(h, r0 + size)
    r0 = max(0, r1 - size)
    c0 = max(0, cc - size // 2)
    c1 = min(w, c0 + size)
    c0 = max(0, c1 - size)
    return tuple(k[r0:r1, c0:c1] if k is not None else None for k in kernels)


def _add_kernel_inset(ax, kernel_img: np.ndarray, corner: str = 'top-right',
                      frac: float = 0.29, border_color: str = 'white',
                      border_lw: float = 1.5):
    """Добавляет квадратную миниатюру ядра в угол осей ax."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes
    size_pct = f"{int(frac * 100)}%"
    loc_map = {
        'top-right':    1,
        'top-left':     2,
        'bottom-left':  3,
        'bottom-right': 4,
    }
    loc = loc_map.get(corner, 1)
    inset = _inset_axes(ax, width=size_pct, height="100%", loc=loc,
                        bbox_to_anchor=ax.bbox,
                        bbox_transform=ax.figure.transFigure,
                        borderpad=0)
    pad = 0.01
    if corner == 'top-right':
        inset = ax.inset_axes([1 - frac - pad, 1 - frac - pad, frac, frac])
    elif corner == 'top-left':
        inset = ax.inset_axes([pad, 1 - frac - pad, frac, frac])
    elif corner == 'bottom-right':
        inset = ax.inset_axes([1 - frac - pad, pad, frac, frac])
    else:
        inset = ax.inset_axes([pad, pad, frac, frac])

    inset.imshow(kernel_img, cmap='gray', vmin=0, vmax=1,
                 interpolation='nearest')
    inset.set_aspect('equal', adjustable='box')
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(border_lw)


# Предопределённые наборы индексов ядер для 4 датасетов.
# Каждая пара наборов отличается не менее чем на 2 элемента:
_KERNEL_PATTERNS = [
    (0, 1, 3),
    (0, 2, 4),
    (1, 3, 5),
    (2, 4, 5),
]


def plot_visual_comparison_per_algorithm(
    alg_name: str,
    df_alg: pd.DataFrame,
    dataset_name: str,
    fig_dir: Optional[Path] = None,
    seed: int = 42,
):
    """
    Строит визуальную таблицу сравнения для одного алгоритма на одном датасете.

    Структура (строки х столбцы):
      Строка 0 — Оригинал
      Строка 1 — Смазанное + GT-ядро в углу
      Строка 2 — Восстановленное + оценённое ядро в углу

    Ядра выбираются детерминированно по рангу датасета.
    Все ядра в строке приведены к одному размеру.
    """
    import matplotlib.pyplot as plt

    df = df_alg[(df_alg['dataset'] == dataset_name) &
                (df_alg['noise_name'] == 'clean')].copy()
    if df.empty:
        print(f"  [{alg_name}] Нет данных для датасета '{dataset_name}' (clean).")
        return

    images = sorted(df['image_name'].unique())
    kernels_all = sorted(df['kernel_name'].unique())

    n_cols = min(len(images), 3)
    images = images[:n_cols]

    all_datasets_sorted = sorted(df_alg['dataset'].unique())
    ds_rank = all_datasets_sorted.index(dataset_name) if dataset_name in all_datasets_sorted else 0
    pattern = _KERNEL_PATTERNS[ds_rank % len(_KERNEL_PATTERNS)]
    chosen_kernels = [kernels_all[i % len(kernels_all)] for i in pattern[:n_cols]]

    experiments = []
    for img_name, ker_name in zip(images, chosen_kernels):
        row = df[(df['image_name'] == img_name) & (df['kernel_name'] == ker_name)]
        if row.empty:
            print(f"  [{alg_name}] Не найдено: img={img_name}, kernel={ker_name}")
            experiments.append(None)
            continue
        row = row.iloc[0]

        orig_p  = Path(row['original_path'])  if pd.notna(row.get('original_path'))  else None
        rest_p  = Path(row['restored_path'])  if pd.notna(row.get('restored_path'))  else None
        gt_k_p  = Path(row['gt_kernel_path']) if pd.notna(row.get('gt_kernel_path')) else None
        est_k_p = Path(row['kernel_path'])    if pd.notna(row.get('kernel_path'))    else None

        blur_p = None
        if orig_p is not None and pd.notna(row.get('distorted_file')):
            blur_p = orig_p.parent.parent / 'distorted' / str(row['distorted_file'])

        experiments.append({
            'img_name': img_name,
            'ker_name': ker_name,
            'orig_p':   orig_p,
            'blur_p':   blur_p,
            'gt_k_p':   gt_k_p,
            'rest_p':   rest_p,
            'est_k_p':  est_k_p,
            'psnr_b': row.get('psnr_blurred', float('nan')),
            'ssim_b': row.get('ssim_blurred', float('nan')),
            'psnr_r': row.get('psnr',         float('nan')),
            'ssim_r': row.get('ssim',         float('nan')),
        })

    if all(e is None for e in experiments):
        return

    n_rows = 3
    cell_size = 3.2
    label_w   = 1.8

    fig_w = label_w + n_cols * cell_size
    fig_h = n_rows * cell_size + 0.8

    fig = plt.figure(figsize=(fig_w, fig_h))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, n_cols + 1,
                  figure=fig,
                  left=0.01, right=0.99,
                  top=0.93, bottom=0.08,
                  wspace=0.04, hspace=0.06,
                  width_ratios=[label_w / cell_size] + [1.0] * n_cols)

    row_labels = [
        'Оригинал',
        'Искаженное',
        'Восстановленное',
    ]

    for r in range(n_rows):
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.95, 0.5, row_labels[r],
                    ha='right', va='center', fontsize=9,
                    transform=ax_lbl.transAxes, multialignment='right')

    fig.suptitle(f"{decode(alg_name)} — {decode(dataset_name)}", fontsize=12, y=0.97)

    for col_idx, exp in enumerate(experiments):
        c = col_idx + 1

        if exp is None:
            for r in range(n_rows):
                ax = fig.add_subplot(gs[r, c])
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'н/д', ha='center', va='center',
                        fontsize=10, color='gray', transform=ax.transAxes)
            continue

        gt_k  = _load_kernel_img(exp['gt_k_p'])  if exp['gt_k_p']  else None
        est_k = _load_kernel_img(exp['est_k_p']) if exp['est_k_p'] else None
        if gt_k is not None and est_k is not None:
            target_k = max(gt_k.shape[0], gt_k.shape[1],
                           est_k.shape[0], est_k.shape[1])
            gt_k  = _pad_kernel_to_square(gt_k,  target_k)
            est_k = _pad_kernel_to_square(est_k, target_k)
        elif gt_k is not None:
            target_k = max(gt_k.shape)
            gt_k = _pad_kernel_to_square(gt_k, target_k)
        elif est_k is not None:
            target_k = max(est_k.shape)
            est_k = _pad_kernel_to_square(est_k, target_k)
        gt_k, est_k = _crop_kernels_to_content(gt_k, est_k)

        ax0 = fig.add_subplot(gs[0, c])
        img_orig = _load_image_gray(exp['orig_p']) if exp['orig_p'] else None
        if img_orig is not None:
            ax0.imshow(img_orig)
        else:
            ax0.set_facecolor('#cccccc')
            ax0.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=8, transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(decode(exp['img_name']), fontsize=9, pad=3)

        ax1 = fig.add_subplot(gs[1, c])
        img_blur = _load_image_gray(exp['blur_p']) if exp['blur_p'] else None
        if img_blur is not None:
            ax1.imshow(img_blur)
        else:
            ax1.set_facecolor('#cccccc')
            ax1.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=8, transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        psnr_b_str = f"{exp['psnr_b']:.2f}" if not np.isnan(exp['psnr_b']) else '—'
        ssim_b_str = f"{exp['ssim_b']:.4f}" if not np.isnan(exp['ssim_b']) else '—'
        ax1.set_xlabel(f"PSNR {psnr_b_str}  SSIM {ssim_b_str}",
                       fontsize=9.5, labelpad=3)
        if gt_k is not None:
            _add_kernel_inset(ax1, gt_k, corner='top-right')

        ax2 = fig.add_subplot(gs[2, c])
        img_rest = _load_image_gray(exp['rest_p']) if exp['rest_p'] else None
        if img_rest is not None:
            ax2.imshow(img_rest)
        else:
            ax2.set_facecolor('#ffe0e0')
            ax2.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=8, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        psnr_r_str = f"{exp['psnr_r']:.2f}" if not np.isnan(exp['psnr_r']) else '—'
        ssim_r_str = f"{exp['ssim_r']:.4f}" if not np.isnan(exp['ssim_r']) else '—'
        ax2.set_xlabel(f"PSNR {psnr_r_str}  SSIM {ssim_r_str}",
                       fontsize=9.5, labelpad=3)
        if est_k is not None:
            _add_kernel_inset(ax2, est_k, corner='top-right')

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"visual_comparison_{alg_name}_{dataset_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        # fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  [{decode(alg_name)}] visual_comparison_{dataset_name} -> {out_dir}")
    plt.close(fig)


def plot_visual_comparison_best_algorithm(
    all_data: dict,
    dataset_name: str,
    fig_dir: Optional[Path] = None,
    metric: str = 'psnr',
):
    """
    Визуальное сравнение: для каждого изображения в датасете показывает
    результат лучшего алгоритма (по метрике) из all_data.

    Структура (3 строки х n_cols столбцов):
      Строка 0 — Оригинал
      Строка 1 — Искаженное + GT-ядро в углу
      Строка 2 — Лучший алгоритм + оценённое ядро в углу
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    frames = []
    for alg_name, df_alg in all_data.items():
        sub = df_alg[(df_alg['dataset'] == dataset_name) &
                     (df_alg['noise_name'] == 'clean')].copy()
        if not sub.empty:
            sub = sub.copy()
            sub['_alg'] = alg_name
            frames.append(sub)

    if not frames:
        print(f"  [best] Нет данных для датасета '{dataset_name}' (clean).")
        return

    df_all = pd.concat(frames, ignore_index=True)

    images = sorted(df_all['image_name'].unique())
    kernels_all = sorted(df_all['kernel_name'].unique())

    n_cols = min(len(images), 3)
    images = images[:n_cols]

    all_datasets_sorted = sorted({
        ds
        for df_a in all_data.values()
        for ds in df_a['dataset'].dropna().unique()
    })
    ds_rank = all_datasets_sorted.index(dataset_name) if dataset_name in all_datasets_sorted else 0
    pattern = _KERNEL_PATTERNS[ds_rank % len(_KERNEL_PATTERNS)]
    chosen_kernels = [kernels_all[i % len(kernels_all)] for i in pattern[:n_cols]]

    experiments = []
    for img_name, ker_name in zip(images, chosen_kernels):
        sub = df_all[(df_all['image_name'] == img_name) &
                     (df_all['kernel_name'] == ker_name)].copy()
        if sub.empty:
            experiments.append(None)
            continue

        if metric not in sub.columns or sub[metric].isna().all():
            best_row = sub.iloc[0]
        else:
            best_row = sub.loc[sub[metric].idxmax()]

        best_alg = best_row['_alg']

        orig_p  = Path(best_row['original_path'])  if pd.notna(best_row.get('original_path'))  else None
        rest_p  = Path(best_row['restored_path'])  if pd.notna(best_row.get('restored_path'))  else None
        gt_k_p  = Path(best_row['gt_kernel_path']) if pd.notna(best_row.get('gt_kernel_path')) else None
        est_k_p = Path(best_row['kernel_path'])    if pd.notna(best_row.get('kernel_path'))    else None

        blur_p = None
        if orig_p is not None and pd.notna(best_row.get('distorted_file')):
            blur_p = orig_p.parent.parent / 'distorted' / str(best_row['distorted_file'])

        experiments.append({
            'img_name': img_name,
            'ker_name': ker_name,
            'best_alg': best_alg,
            'orig_p':   orig_p,
            'blur_p':   blur_p,
            'gt_k_p':   gt_k_p,
            'rest_p':   rest_p,
            'est_k_p':  est_k_p,
            'psnr_b': best_row.get('psnr_blurred', float('nan')),
            'ssim_b': best_row.get('ssim_blurred', float('nan')),
            'psnr_r': best_row.get('psnr',         float('nan')),
            'ssim_r': best_row.get('ssim',         float('nan')),
        })

    if all(e is None for e in experiments):
        return

    cell_size = 3.5
    label_w   = 2.8

    fig_w = label_w + n_cols * cell_size
    fig_h = 3 * cell_size + 0.25 * cell_size + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(4, n_cols + 1,
                  figure=fig,
                  left=0.01, right=0.99,
                  top=0.94, bottom=0.14,
                  wspace=0.04, hspace=0.06,
                  width_ratios=[label_w / cell_size] + [1.0] * n_cols,
                  height_ratios=[1, 1, 0.03, 1])

    row_labels = ['Оригинал', 'Искаженное', 'Восстновл.']
    _label_rows = [0, 1, 3]
    for r_idx, r in enumerate(_label_rows):
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.95, 0.5, row_labels[r_idx],
                    ha='right', va='center', fontsize=15,
                    transform=ax_lbl.transAxes, multialignment='right')

    for col_idx, exp in enumerate(experiments):
        c = col_idx + 1

        if exp is None:
            for r in _label_rows:
                ax = fig.add_subplot(gs[r, c])
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'н/д', ha='center', va='center',
                        fontsize=16, color='gray', transform=ax.transAxes)
            continue

        gt_k  = _load_kernel_img(exp['gt_k_p'])  if exp['gt_k_p']  else None
        est_k = _load_kernel_img(exp['est_k_p']) if exp['est_k_p'] else None
        if gt_k is not None and est_k is not None:
            target_k = max(gt_k.shape[0], gt_k.shape[1],
                           est_k.shape[0], est_k.shape[1])
            gt_k  = _pad_kernel_to_square(gt_k,  target_k)
            est_k = _pad_kernel_to_square(est_k, target_k)
        elif gt_k is not None:
            gt_k  = _pad_kernel_to_square(gt_k,  max(gt_k.shape))
        elif est_k is not None:
            est_k = _pad_kernel_to_square(est_k, max(est_k.shape))
        gt_k, est_k = _crop_kernels_to_content(gt_k, est_k)

        ax0 = fig.add_subplot(gs[0, c])
        img_orig = _load_image_gray(exp['orig_p']) if exp['orig_p'] else None
        if img_orig is not None:
            ax0.imshow(img_orig)
        else:
            ax0.set_facecolor('#cccccc')
            ax0.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(decode(exp['img_name']), fontsize=15, pad=3)

        ax1 = fig.add_subplot(gs[1, c])
        img_blur = _load_image_gray(exp['blur_p']) if exp['blur_p'] else None
        if img_blur is not None:
            ax1.imshow(img_blur)
        else:
            ax1.set_facecolor('#cccccc')
            ax1.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        psnr_b_str = f"{exp['psnr_b']:.2f}" if not np.isnan(exp['psnr_b']) else '—'
        ssim_b_str = f"{exp['ssim_b']:.4f}" if not np.isnan(exp['ssim_b']) else '—'
        ax1.set_xlabel(f"PSNR {psnr_b_str}  SSIM {ssim_b_str}",
                       fontsize=15, labelpad=3)
        if gt_k is not None:
            _add_kernel_inset(ax1, gt_k, corner='top-right')

        ax2 = fig.add_subplot(gs[3, c])
        img_rest = _load_image_gray(exp['rest_p']) if exp['rest_p'] else None
        if img_rest is not None:
            ax2.imshow(img_rest)
        else:
            ax2.set_facecolor('#ffe0e0')
            ax2.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        psnr_r_str = f"{exp['psnr_r']:.2f}" if not np.isnan(exp['psnr_r']) else '—'
        ssim_r_str = f"{exp['ssim_r']:.4f}" if not np.isnan(exp['ssim_r']) else '—'
        ax2.set_xlabel(
            f"PSNR {psnr_r_str}  SSIM {ssim_r_str}\n{decode(exp['best_alg'])}",
            fontsize=15, labelpad=3)
        if est_k is not None:
            _add_kernel_inset(ax2, est_k, corner='top-right')

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"visual_comparison_best_{dataset_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        # fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  [best/{dataset_name}] -> {out_dir}")
    plt.close(fig)


def plot_visual_comparison_best_mean_algorithm(
    all_data: dict,
    dataset_name: str,
    fig_dir: Optional[Path] = None,
    metric: str = 'psnr',
):
    """
    Визуальное сравнение: один лучший алгоритм по среднему metric на датасете.

    Структура аналогична plot_visual_comparison_best_algorithm.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    best_alg_name = None
    best_mean = -np.inf
    for alg_name, df_alg in all_data.items():
        sub = df_alg[(df_alg['dataset'] == dataset_name) &
                     (df_alg['noise_name'] == 'clean')]
        if sub.empty or metric not in sub.columns:
            continue
        m = sub[metric].dropna()
        if m.empty:
            continue
        val = m.mean()
        if val > best_mean:
            best_mean = val
            best_alg_name = alg_name

    if best_alg_name is None:
        print(f"  [best_mean] Нет данных для датасета '{dataset_name}'.")
        return

    df_best = all_data[best_alg_name]
    df = df_best[(df_best['dataset'] == dataset_name) &
                 (df_best['noise_name'] == 'clean')].copy()
    df['_alg'] = best_alg_name

    images = sorted(df['image_name'].unique())
    kernels_all = sorted(df['kernel_name'].unique())

    n_cols = min(len(images), 3)
    images = images[:n_cols]

    all_datasets_sorted = sorted({
        ds
        for df_a in all_data.values()
        for ds in df_a['dataset'].dropna().unique()
    })
    ds_rank = all_datasets_sorted.index(dataset_name) if dataset_name in all_datasets_sorted else 0
    pattern = _KERNEL_PATTERNS[ds_rank % len(_KERNEL_PATTERNS)]
    chosen_kernels = [kernels_all[i % len(kernels_all)] for i in pattern[:n_cols]]

    experiments = []
    for img_name, ker_name in zip(images, chosen_kernels):
        row = df[(df['image_name'] == img_name) & (df['kernel_name'] == ker_name)]
        if row.empty:
            experiments.append(None)
            continue
        row = row.iloc[0]

        orig_p  = Path(row['original_path'])  if pd.notna(row.get('original_path'))  else None
        rest_p  = Path(row['restored_path'])  if pd.notna(row.get('restored_path'))  else None
        gt_k_p  = Path(row['gt_kernel_path']) if pd.notna(row.get('gt_kernel_path')) else None
        est_k_p = Path(row['kernel_path'])    if pd.notna(row.get('kernel_path'))    else None

        blur_p = None
        if orig_p is not None and pd.notna(row.get('distorted_file')):
            blur_p = orig_p.parent.parent / 'distorted' / str(row['distorted_file'])

        experiments.append({
            'img_name':   img_name,
            'ker_name':   ker_name,
            'orig_p':     orig_p,
            'blur_p':     blur_p,
            'gt_k_p':     gt_k_p,
            'rest_p':     rest_p,
            'est_k_p':    est_k_p,
            'noise_name': row.get('noise_name', 'clean'),
            'psnr_b': row.get('psnr_blurred', float('nan')),
            'ssim_b': row.get('ssim_blurred', float('nan')),
            'psnr_r': row.get('psnr',         float('nan')),
            'ssim_r': row.get('ssim',         float('nan')),
        })

    if all(e is None for e in experiments):
        return

    cell_size = 3.5
    label_w   = 2.8

    fig_w = label_w + n_cols * cell_size
    fig_h = 3 * cell_size + 0.25 * cell_size + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(4, n_cols + 1,
                  figure=fig,
                  left=0.01, right=0.99,
                  top=0.94, bottom=0.14,
                  wspace=0.04, hspace=0.06,
                  width_ratios=[label_w / cell_size] + [1.0] * n_cols,
                  height_ratios=[1, 1, 0.03, 1])

    row_labels = ['Оригинал', 'Искаженное', decode(best_alg_name)]
    _label_rows = [0, 1, 3]
    for r_idx, r in enumerate(_label_rows):
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.95, 0.5, row_labels[r_idx],
                    ha='right', va='center', fontsize=15,
                    transform=ax_lbl.transAxes, multialignment='right')

    for col_idx, exp in enumerate(experiments):
        c = col_idx + 1

        if exp is None:
            for r in _label_rows:
                ax = fig.add_subplot(gs[r, c])
                ax.set_axis_off()
                ax.text(0.5, 0.5, 'н/д', ha='center', va='center',
                        fontsize=16, color='gray', transform=ax.transAxes)
            continue

        gt_k  = _load_kernel_img(exp['gt_k_p'])  if exp['gt_k_p']  else None
        est_k = _load_kernel_img(exp['est_k_p']) if exp['est_k_p'] else None
        if gt_k is not None and est_k is not None:
            target_k = max(gt_k.shape[0], gt_k.shape[1],
                           est_k.shape[0], est_k.shape[1])
            gt_k  = _pad_kernel_to_square(gt_k,  target_k)
            est_k = _pad_kernel_to_square(est_k, target_k)
        elif gt_k is not None:
            gt_k  = _pad_kernel_to_square(gt_k,  max(gt_k.shape))
        elif est_k is not None:
            est_k = _pad_kernel_to_square(est_k, max(est_k.shape))
        gt_k, est_k = _crop_kernels_to_content(gt_k, est_k)

        ax0 = fig.add_subplot(gs[0, c])
        img_orig = _load_image_gray(exp['orig_p']) if exp['orig_p'] else None
        if img_orig is not None:
            ax0.imshow(img_orig)
        else:
            ax0.set_facecolor('#cccccc')
            ax0.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax0.transAxes)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_title(decode(exp['img_name']), fontsize=15, pad=3)

        ax1 = fig.add_subplot(gs[1, c])
        img_blur = _load_image_gray(exp['blur_p']) if exp['blur_p'] else None
        if img_blur is not None:
            ax1.imshow(img_blur)
        else:
            ax1.set_facecolor('#cccccc')
            ax1.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        psnr_b_str = f"{exp['psnr_b']:.2f}" if not np.isnan(exp['psnr_b']) else '—'
        ssim_b_str = f"{exp['ssim_b']:.4f}" if not np.isnan(exp['ssim_b']) else '—'
        ax1.set_xlabel(f"PSNR {psnr_b_str}  SSIM {ssim_b_str}",
                       fontsize=15, labelpad=3)
        if gt_k is not None:
            _add_kernel_inset(ax1, gt_k, corner='top-right')

        ax2 = fig.add_subplot(gs[3, c])
        img_rest = _load_image_gray(exp['rest_p']) if exp['rest_p'] else None
        if img_rest is not None:
            ax2.imshow(img_rest)
        else:
            ax2.set_facecolor('#ffe0e0')
            ax2.text(0.5, 0.5, 'нет файла', ha='center', va='center',
                     fontsize=13, transform=ax2.transAxes)
        ax2.set_xticks([])
        ax2.set_yticks([])
        psnr_r_str = f"{exp['psnr_r']:.2f}" if not np.isnan(exp['psnr_r']) else '—'
        ssim_r_str = f"{exp['ssim_r']:.4f}" if not np.isnan(exp['ssim_r']) else '—'
        ax2.set_xlabel(f"PSNR {psnr_r_str}  SSIM {ssim_r_str}",
                       fontsize=15, labelpad=3)
        if est_k is not None:
            _add_kernel_inset(ax2, est_k, corner='top-right')

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"visual_comparison_best_mean_{dataset_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        # fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  [best_mean/{dataset_name}] {decode(best_alg_name)} -> {out_dir}")
    plt.close(fig)


def plot_best_worst_comparison(
    alg_name: str,
    df_alg: pd.DataFrame,
    dataset_name: str,
    metric: str = 'psnr',
    fig_dir: Optional[Path] = None,
):
    """
    Строит таблицу 3 х 3: лучший (сверху), медианный (середина), худший (снизу).
    Столбцы: Оригинал | Искажённое + GT-ядро | Восстановленное + est-ядро
    Метрики подписаны под изображениями.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    df_ds = df_alg[df_alg['dataset'] == dataset_name].copy()
    if df_ds.empty or metric not in df_ds.columns:
        print(f"  [{alg_name}] Нет данных для {dataset_name}/{metric}")
        return

    df_ds = df_ds.dropna(subset=[metric])
    if df_ds.empty:
        return

    idx_best  = df_ds[metric].idxmax()
    idx_worst = df_ds[metric].idxmin()
    median_val = df_ds[metric].median()
    idx_median = (df_ds[metric] - median_val).abs().idxmin()

    def _build_exp(row):
        orig_p  = Path(row['original_path'])  if pd.notna(row.get('original_path'))  else None
        rest_p  = Path(row['restored_path'])  if pd.notna(row.get('restored_path'))  else None
        gt_k_p  = Path(row['gt_kernel_path']) if pd.notna(row.get('gt_kernel_path')) else None
        est_k_p = Path(row['kernel_path'])    if pd.notna(row.get('kernel_path'))    else None
        blur_p  = None
        if orig_p is not None and pd.notna(row.get('distorted_file')):
            blur_p = orig_p.parent.parent / 'distorted' / str(row['distorted_file'])
        return {
            'orig_p':  orig_p,  'blur_p':  blur_p,
            'gt_k_p':  gt_k_p,  'rest_p':  rest_p,  'est_k_p': est_k_p,
            'noise_name': row.get('noise_name', 'clean'),
            'psnr_b': row.get('psnr_blurred', float('nan')),
            'ssim_b': row.get('ssim_blurred', float('nan')),
            'psnr_r': row.get('psnr',  float('nan')),
            'ssim_r': row.get('ssim',  float('nan')),
            'distorted_file': row.get('distorted_file', ''),
        }

    cases = [_build_exp(df_ds.loc[idx_best]),
             _build_exp(df_ds.loc[idx_median]),
             _build_exp(df_ds.loc[idx_worst])]
    row_labels = ['Лучший', 'Медиана', 'Худший']

    n_rows = 3
    cell_w, cell_h = 5.0, 5.0
    label_w  = 2.8
    gap_w    = 0.20
    spacer_w = 0.05

    fig_w = label_w + 3 * cell_w + gap_w + spacer_w
    fig_h = n_rows * cell_h + 1.6

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(n_rows, 6, figure=fig,
                  left=0.01, right=0.99,
                  top=0.95, bottom=0.08,
                  wspace=0.004, hspace=0.16,
                  width_ratios=[label_w / cell_w, 1.0, gap_w / cell_w, 1.0, spacer_w / cell_w, 1.0])

    col_titles = ['Оригинал', 'Искажённое', 'Восстновл.']
    _col_gs = [1, 3, 5]

    for ri, (exp, lbl) in enumerate(zip(cases, row_labels)):
        gt_k  = _load_kernel_img(exp['gt_k_p'])  if exp['gt_k_p']  else None
        est_k = _load_kernel_img(exp['est_k_p']) if exp['est_k_p'] else None
        if gt_k is not None and est_k is not None:
            target_k = max(gt_k.shape[0], gt_k.shape[1],
                           est_k.shape[0], est_k.shape[1])
            gt_k  = _pad_kernel_to_square(gt_k,  target_k)
            est_k = _pad_kernel_to_square(est_k, target_k)
        elif gt_k is not None:
            gt_k  = _pad_kernel_to_square(gt_k,  max(gt_k.shape))
        elif est_k is not None:
            est_k = _pad_kernel_to_square(est_k, max(est_k.shape))
        gt_k, est_k = _crop_kernels_to_content(gt_k, est_k)

        psnr_r = exp['psnr_r']
        ssim_r = exp['ssim_r']
        val_str = (f"PSNR {psnr_r:.2f}" if not np.isnan(psnr_r) else "PSNR —") \
                  if metric == 'psnr' else \
                  (f"SSIM {ssim_r:.4f}" if not np.isnan(ssim_r) else "SSIM —")
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.95, 0.58, lbl, ha='right', va='center',
                    fontsize=17, fontweight='normal', transform=ax_lbl.transAxes)
        ax_lbl.text(0.95, 0.38, val_str, ha='right', va='center',
                    fontsize=15, color='#555555', transform=ax_lbl.transAxes)

        ax = fig.add_subplot(gs[ri, _col_gs[0]])
        img = _load_image_gray(exp['orig_p']) if exp['orig_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(col_titles[0], fontsize=17, pad=3)

        ax = fig.add_subplot(gs[ri, _col_gs[1]])
        img = _load_image_gray(exp['blur_p']) if exp['blur_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        psnr_b = exp['psnr_b']; ssim_b = exp['ssim_b']
        psnr_b_s = f"{psnr_b:.2f}" if not np.isnan(psnr_b) else '—'
        ssim_b_s = f"{ssim_b:.4f}" if not np.isnan(ssim_b) else '—'
        _nn = exp.get('noise_name', 'clean') or 'clean'
        _noise_str = 'Шум: нет' if _nn in ('clean', 'unknown', '') else f"Шум: {decode(_nn)}"
        ax.set_xlabel(f"PSNR {psnr_b_s}  SSIM {ssim_b_s}\n{_noise_str}", fontsize=16, labelpad=4)
        if gt_k is not None:
            _add_kernel_inset(ax, gt_k, corner='top-right')
        if ri == 0:
            ax.set_title(col_titles[1], fontsize=17, pad=3)

        ax = fig.add_subplot(gs[ri, _col_gs[2]])
        img = _load_image_gray(exp['rest_p']) if exp['rest_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#ffe0e0')
        ax.set_xticks([]); ax.set_yticks([])
        psnr_r_s = f"{psnr_r:.2f}" if not np.isnan(psnr_r) else '—'
        ssim_r_s = f"{ssim_r:.4f}" if not np.isnan(ssim_r) else '—'
        ax.set_xlabel(f"PSNR {psnr_r_s}  SSIM {ssim_r_s}", fontsize=16, labelpad=4)
        if est_k is not None:
            _add_kernel_inset(ax, est_k, corner='top-right')
        if ri == 0:
            ax.set_title(col_titles[2], fontsize=17, pad=3)

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"best_worst_{metric}_{dataset_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        print(f"  [{alg_name}] best_worst_{metric}_{dataset_name} -> {out_dir}")
    plt.close(fig)


def plot_noise_visual_comparison(
    all_data: dict,
    dataset_name: str,
    noise_name: str,
    fig_dir: Optional[Path] = None,
):
    """
    Таблица: робастность к шуму.

    Строки:
      0 —          | Оригинал | Искажённое (без шума) | Искажённое (с шумом)
      1 — Без шума | Algo1 clean | Algo2 clean | …
      2 — С шумом  | Algo1 noisy | Algo2 noisy | …

    Выбор: сначала случайная пара (original, gt_kernel) из шумовых данных,
    затем подбирается та же пара из чистых.
    """
    import hashlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    alg_order = sorted(all_data.keys())
    clean_frames: dict = {}
    noisy_frames: dict = {}
    for alg in alg_order:
        df = all_data[alg]
        ds = df[df['dataset'] == dataset_name]
        c = ds[ds['noise_name'] == 'clean']
        n = ds[ds['noise_name'] == noise_name]
        if not c.empty:
            clean_frames[alg] = c.copy()
        if not n.empty:
            noisy_frames[alg] = n.copy()

    alg_names = [a for a in alg_order if a in clean_frames and a in noisy_frames]
    if not alg_names:
        return

    def _pairs_from(frames_dict, keys):
        sets = []
        for alg in keys:
            df = frames_dict[alg]
            pairs = set()
            for _, row in df.iterrows():
                o = str(row.get('original_path', '') or '')
                k = str(row.get('gt_kernel_path', '') or '')
                if o and o != 'nan':
                    pairs.add((o, k))
            sets.append(pairs)
        common = sets[0]
        for s in sets[1:]:
            common &= s
        return common if common else sets[0]

    noisy_pairs = _pairs_from(noisy_frames, alg_names)
    clean_pairs  = _pairs_from(clean_frames,  alg_names)
    valid_pairs = noisy_pairs & clean_pairs
    if not valid_pairs:
        valid_pairs = noisy_pairs

    seed = int(hashlib.md5(f"{dataset_name}_{noise_name}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sorted_pairs = sorted(valid_pairs)
    chosen_orig, chosen_gtk = sorted_pairs[rng.integers(0, len(sorted_pairs))]

    def _pick_row(df_sub: pd.DataFrame) -> pd.Series:
        """Выбирает строку с совпадающим original_path и, если есть, gt_kernel_path."""
        r = df_sub[df_sub['original_path'].map(str) == chosen_orig]
        if chosen_gtk:
            r2 = r[r['gt_kernel_path'].map(str) == chosen_gtk]
            if not r2.empty:
                r = r2
        return r.iloc[0] if not r.empty else df_sub.iloc[0]

    def _blur_path(row: pd.Series) -> Optional[Path]:
        p = row.get('distorted_path')
        if pd.notna(p) and p:
            return Path(str(p))
        orig = row.get('original_path')
        dfile = row.get('distorted_file')
        if pd.notna(orig) and pd.notna(dfile) and dfile:
            return Path(str(orig)).parent.parent / 'distorted' / str(dfile)
        return None

    def _metric_str(row: pd.Series, blurred: bool = False) -> str:
        pk = 'psnr_blurred' if blurred else 'psnr'
        sk = 'ssim_blurred' if blurred else 'ssim'
        p = row.get(pk, float('nan'))
        s = row.get(sk, float('nan'))
        ps = f"{float(p):.2f}" if pd.notna(p) else '—'
        ss = f"{float(s):.4f}" if pd.notna(s) else '—'
        return f"PSNR {ps}  SSIM {ss}"

    first_noisy = _pick_row(noisy_frames[alg_names[0]])
    first_clean = _pick_row(clean_frames[alg_names[0]])

    orig_img       = _load_image_gray(first_clean.get('original_path'))
    blur_clean_img = _load_image_gray(_blur_path(first_clean))
    blur_noisy_img = _load_image_gray(_blur_path(first_noisy))

    gt_k = _load_kernel_img(chosen_gtk) if chosen_gtk and chosen_gtk != 'nan' else None
    if gt_k is None:
        gt_k = _load_kernel_img(first_noisy.get('gt_kernel_path'))

    _FS_TITLE  = 26   # заголовки ячеек
    _FS_METRIC = 22   # метки метрик
    _FS_ALG    = 24   # названия алгоритмов
    _FS_LABEL  = 26   # метки строк слева

    n_alg      = len(alg_names)
    n_img_cols = max(n_alg, 3)   # минимум 3: orig / blur_clean / blur_noisy
    n_cols     = 1 + n_img_cols  # col 0 = узкий label, cols 1..n_img_cols = изображения
    cell_w, cell_h = 4.5, 4.0
    label_w = 2.6
    fig_w = label_w + n_img_cols * cell_w
    fig_h = 3 * cell_h + 1.2
    noise_display = decode(noise_name)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(4, n_cols, figure=fig,
                  left=0.01, right=0.99,
                  top=0.94, bottom=0.05,
                  wspace=0.02, hspace=0.14,
                  height_ratios=[1.0, 0.25, 1.0, 1.0],
                  width_ratios=[label_w / cell_w] + [1.0] * n_img_cols)

    def _show(ax, img):
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])

    fig.add_subplot(gs[0, 0]).axis('off')

    ax = fig.add_subplot(gs[0, 1])
    _show(ax, orig_img)
    ax.set_title('Оригинал', fontsize=_FS_TITLE, pad=4)

    ax = fig.add_subplot(gs[0, 2])
    _show(ax, blur_clean_img)
    ax.set_title('Искажённое\n(без шума)', fontsize=_FS_TITLE, pad=4)
    ax.set_xlabel(_metric_str(first_clean, blurred=True), fontsize=_FS_METRIC, labelpad=3)
    if gt_k is not None:
        _add_kernel_inset(ax, gt_k, corner='top-right')

    if n_img_cols >= 3:
        ax = fig.add_subplot(gs[0, 3])
        _show(ax, blur_noisy_img)
        ax.set_title(f'Искажённое\n({noise_display})', fontsize=_FS_TITLE, pad=4)
        ax.set_xlabel(_metric_str(first_noisy, blurred=True), fontsize=_FS_METRIC, labelpad=3)
        if gt_k is not None:
            _add_kernel_inset(ax, gt_k, corner='top-right')

    for c in range(4, n_cols):
        fig.add_subplot(gs[0, c]).axis('off')

    for c in range(n_cols):
        fig.add_subplot(gs[1, c]).axis('off')

    for ri, (frames, row_label) in enumerate(
        [(clean_frames, 'Без\nшума'),
         (noisy_frames, 'С\nшумом')],
        start=2,
    ):
        ax_lbl = fig.add_subplot(gs[ri, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.95, 0.55, row_label, ha='right', va='center',
                    fontsize=_FS_LABEL, transform=ax_lbl.transAxes)

        for ci, alg in enumerate(alg_names, start=1):
            ax = fig.add_subplot(gs[ri, ci])
            row = _pick_row(frames[alg])
            rest_p = row.get('restored_path')
            _show(ax, _load_image_gray(rest_p) if pd.notna(rest_p) else None)
            ax.set_xlabel(_metric_str(row), fontsize=_FS_METRIC, labelpad=3)
            if ri == 2:
                ax.set_title(decode(alg), fontsize=_FS_ALG, pad=4)
            est_k = _load_kernel_img(row.get('kernel_path'))
            if est_k is not None:
                _add_kernel_inset(ax, est_k, corner='top-right')

        for ci in range(n_alg + 1, n_cols):
            fig.add_subplot(gs[ri, ci]).axis('off')

    if fig_dir:
        out_dir = Path(fig_dir) / "visual_comparison_noise"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"noise_visual_{dataset_name}_{noise_name}"
        fig.savefig(out_dir / f"{stem}.png", dpi=200)
        print(f"  noise_visual {dataset_name}/{noise_name} -> {out_dir}")
    plt.close(fig)


def _build_big_comparison_data(all_data: dict, dataset_name: str, noise_name: str):
    """
    Собирает данные для большой таблицы.
    Возвращает (alg_names, col_specs) или (None, None).

    col_specs: 6 записей — 3 изображения × 2 ядра каждое:
      {img_name, ker_name, orig_p, blur_p, gt_k_p, psnr_b, ssim_b,
       alg_results: {alg_name: {rest_p, est_k_p, psnr_r, ssim_r} | None}}

    alg_names: отсортированы по убыванию среднего PSNR.
    """
    frames = {}
    for alg_name, df_alg in all_data.items():
        sub = df_alg[(df_alg['dataset'] == dataset_name) &
                     (df_alg['noise_name'] == noise_name)].copy()
        if not sub.empty:
            frames[alg_name] = sub

    if not frames:
        return None, None

    all_rows = pd.concat(list(frames.values()), ignore_index=True)
    images  = sorted(all_rows['image_name'].unique())[:3]
    kernels = sorted(all_rows['kernel_name'].unique())

    # 6 пар: img[i] х ker[i*2+j]
    pairs = [(images[i], kernels[(i * 2 + j) % len(kernels)])
             for i in range(len(images)) for j in range(2)]

    alg_means = {a: frames[a]['psnr'].dropna().mean()
                 for a in frames if 'psnr' in frames[a].columns}
    alg_names = sorted(frames.keys(),
                       key=lambda a: alg_means.get(a, -np.inf),
                       reverse=True)

    col_specs = []
    for img_name, ker_name in pairs:
        spec: dict = {
            'img_name': img_name, 'ker_name': ker_name,
            'orig_p': None, 'blur_p': None, 'gt_k_p': None,
            'psnr_b': float('nan'), 'ssim_b': float('nan'),
        }
        for df_a in frames.values():
            row = df_a[(df_a['image_name'] == img_name) &
                       (df_a['kernel_name'] == ker_name)]
            if not row.empty:
                r = row.iloc[0]
                spec['orig_p'] = Path(r['original_path'])  if pd.notna(r.get('original_path'))  else None
                spec['gt_k_p'] = Path(r['gt_kernel_path']) if pd.notna(r.get('gt_kernel_path')) else None
                spec['psnr_b'] = r.get('psnr_blurred', float('nan'))
                spec['ssim_b'] = r.get('ssim_blurred',  float('nan'))
                if spec['orig_p'] is not None and pd.notna(r.get('distorted_file')):
                    spec['blur_p'] = spec['orig_p'].parent.parent / 'distorted' / str(r['distorted_file'])
                break

        alg_results = {}
        for alg_name in alg_names:
            row = frames[alg_name][(frames[alg_name]['image_name'] == img_name) &
                                   (frames[alg_name]['kernel_name'] == ker_name)]
            if row.empty:
                alg_results[alg_name] = None
            else:
                r = row.iloc[0]
                alg_results[alg_name] = {
                    'rest_p':  Path(r['restored_path']) if pd.notna(r.get('restored_path')) else None,
                    'est_k_p': Path(r['kernel_path'])   if pd.notna(r.get('kernel_path'))   else None,
                    'psnr_r':  r.get('psnr',  float('nan')),
                    'ssim_r':  r.get('ssim',  float('nan')),
                }
        spec['alg_results'] = alg_results
        col_specs.append(spec)

    return alg_names, col_specs


def _load_col_kernels(spec: dict, alg_names: list):
    """Загружает и нормализует ядра для одного столбца (GT + все алгоритмы)."""
    gt_k = _load_kernel_img(spec['gt_k_p']) if spec['gt_k_p'] else None
    est_ks = {a: (_load_kernel_img(spec['alg_results'][a]['est_k_p'])
                  if spec['alg_results'].get(a) and spec['alg_results'][a].get('est_k_p')
                  else None)
              for a in alg_names}

    all_ks = [k for k in [gt_k] + list(est_ks.values()) if k is not None]
    if all_ks:
        target = max(max(k.shape) for k in all_ks)
        if gt_k is not None:
            gt_k = _pad_kernel_to_square(gt_k, target)
        est_ks = {a: (_pad_kernel_to_square(k, target) if k is not None else None)
                  for a, k in est_ks.items()}
    all_k_vals = [gt_k] + [est_ks[a] for a in alg_names]
    cropped = _crop_kernels_to_content(*all_k_vals)
    gt_k = cropped[0]
    est_ks = {a: cropped[i + 1] for i, a in enumerate(alg_names)}
    return gt_k, est_ks


def _fmt_metrics(psnr, ssim) -> str:
    def _f(v, fmt):
        try:
            f = float(v)
            return (fmt % f) if not np.isnan(f) else '—'
        except Exception:
            return '—'
    return f"PSNR {_f(psnr, '%.2f')}  SSIM {_f(ssim, '%.4f')}"


def plot_big_comparison_vertical(
    all_data: dict,
    dataset_name: str,
    noise_name: str = 'clean',
    fig_dir: Optional[Path] = None,
):
    """
    Большая вертикальная таблица сравнения.

    Строки: Оригинал | Искажённое | Алг1 | Алг2 | ...
    Столбцы: 6 пар - 3 изображения × 2 ядра каждое.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    alg_names, col_specs = _build_big_comparison_data(all_data, dataset_name, noise_name)
    if alg_names is None:
        print(f"  [big_vert] Нет данных: {dataset_name}/{noise_name}")
        return

    n_cols = len(col_specs)          # 6
    n_rows = 2 + len(alg_names)      # orig + distorted + algs
    cell_w, cell_h = 1.8, 1.7
    label_w = 2.4

    fig = plt.figure(figsize=(label_w + n_cols * cell_w,
                               n_rows * cell_h + 0.6))
    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  left=0.01, right=0.99, top=0.95, bottom=0.04,
                  wspace=0.03, hspace=0.12,
                  width_ratios=[label_w / cell_w] + [1.0] * n_cols)

    noise_label = decode(noise_name) if noise_name != 'clean' else 'без шума'
    fig.suptitle(f"{decode(dataset_name)} — {noise_label}", fontsize=11, y=0.98)

    row_labels = ['Оригинал', 'Искажённое'] + [decode(a) for a in alg_names]
    for r, lbl in enumerate(row_labels):
        ax = fig.add_subplot(gs[r, 0])
        ax.set_axis_off()
        ax.text(0.96, 0.5, lbl, ha='right', va='center', fontsize=7.5,
                transform=ax.transAxes, multialignment='right')

    for ci, spec in enumerate(col_specs):
        c = ci + 1
        gt_k, est_ks = _load_col_kernels(spec, alg_names)

        # Строка 0: оригинал
        ax = fig.add_subplot(gs[0, c])
        img = _load_image_gray(spec['orig_p']) if spec['orig_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(decode(spec['img_name']), fontsize=7.5, pad=2)

        # Строка 1: искажённое + GT ядро
        ax = fig.add_subplot(gs[1, c])
        img = _load_image_gray(spec['blur_p']) if spec['blur_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(_fmt_metrics(spec['psnr_b'], spec['ssim_b']),
                      fontsize=6.5, labelpad=2)
        if gt_k is not None:
            _add_kernel_inset(ax, gt_k, corner='top-right', frac=0.27)

        # Строки 2 и т.д.: алгоритмы
        for ri, alg_name in enumerate(alg_names):
            ax = fig.add_subplot(gs[2 + ri, c])
            alg_r = spec['alg_results'].get(alg_name)
            if alg_r and alg_r.get('rest_p'):
                img = _load_image_gray(alg_r['rest_p'])
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.set_facecolor('#ffe0e0')
            else:
                ax.set_facecolor('#ffe0e0')
                ax.text(0.5, 0.5, 'н/д', ha='center', va='center',
                        fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if alg_r:
                ax.set_xlabel(_fmt_metrics(alg_r['psnr_r'], alg_r['ssim_r']),
                              fontsize=6.5, labelpad=2)
            est_k = est_ks.get(alg_name)
            if est_k is not None:
                _add_kernel_inset(ax, est_k, corner='top-right', frac=0.27)

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = noise_name.replace('/', '_')
        stem = f"big_vertical_{dataset_name}_{safe}"
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        # fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  [big_vert] {dataset_name}/{noise_name} -> {out_dir}")
    plt.close(fig)


def plot_big_comparison_horizontal(
    all_data: dict,
    dataset_name: str,
    noise_name: str = 'clean',
    fig_dir: Optional[Path] = None,
):
    """
    Большая горизонтальная таблица сравнения (для презентации).

    Строки: 6 пар — 3 изображения × 2 ядра каждое.
    Столбцы: Лейбл | Оригинал | Искажённое | Алг1 | Алг2 | ...
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    alg_names, col_specs = _build_big_comparison_data(all_data, dataset_name, noise_name)
    if alg_names is None:
        print(f"  [big_horiz] Нет данных: {dataset_name}/{noise_name}")
        return

    n_data_rows = len(col_specs)          # 6
    n_data_cols = 2 + len(alg_names)      # orig + distorted + algs
    cell_w, cell_h = 1.8, 1.7
    label_w = 1.4

    fig = plt.figure(figsize=(label_w + n_data_cols * cell_w,
                               n_data_rows * cell_h + 0.6))
    gs = GridSpec(n_data_rows, n_data_cols + 1, figure=fig,
                  left=0.01, right=0.99, top=0.94, bottom=0.04,
                  wspace=0.03, hspace=0.12,
                  width_ratios=[label_w / cell_w] + [1.0] * n_data_cols)

    noise_label = decode(noise_name) if noise_name != 'clean' else 'без шума'
    fig.suptitle(f"{decode(dataset_name)} — {noise_label}", fontsize=11, y=0.98)

    col_titles = ['Оригинал', 'Искажённое'] + [decode(a) for a in alg_names]

    for ri, spec in enumerate(col_specs):
        gt_k, est_ks = _load_col_kernels(spec, alg_names)
        ax = fig.add_subplot(gs[ri, 0])
        ax.set_axis_off()
        ax.text(0.96, 0.5,
                decode(spec['img_name']),
                ha='right', va='center', fontsize=7.5,
                transform=ax.transAxes, multialignment='right')

        # Оригинал
        ax = fig.add_subplot(gs[ri, 1])
        img = _load_image_gray(spec['orig_p']) if spec['orig_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(col_titles[0], fontsize=7.5, pad=3)

        # Искажённое + GT ядро
        ax = fig.add_subplot(gs[ri, 2])
        img = _load_image_gray(spec['blur_p']) if spec['blur_p'] else None
        if img is not None:
            ax.imshow(img)
        else:
            ax.set_facecolor('#cccccc')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(_fmt_metrics(spec['psnr_b'], spec['ssim_b']),
                      fontsize=6.5, labelpad=2)
        if gt_k is not None:
            _add_kernel_inset(ax, gt_k, corner='top-right', frac=0.27)
        if ri == 0:
            ax.set_title(col_titles[1], fontsize=7.5, pad=3)

        # Алгоритмы
        for ai, alg_name in enumerate(alg_names):
            ax = fig.add_subplot(gs[ri, 3 + ai])
            alg_r = spec['alg_results'].get(alg_name)
            if alg_r and alg_r.get('rest_p'):
                img = _load_image_gray(alg_r['rest_p'])
                if img is not None:
                    ax.imshow(img)
                else:
                    ax.set_facecolor('#ffe0e0')
            else:
                ax.set_facecolor('#ffe0e0')
                ax.text(0.5, 0.5, 'н/д', ha='center', va='center',
                        fontsize=6, color='gray', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            if alg_r:
                ax.set_xlabel(_fmt_metrics(alg_r['psnr_r'], alg_r['ssim_r']),
                              fontsize=6.5, labelpad=2)
            est_k = est_ks.get(alg_name)
            if est_k is not None:
                _add_kernel_inset(ax, est_k, corner='top-right', frac=0.27)
            if ri == 0:
                ax.set_title(col_titles[2 + ai], fontsize=7.5, pad=3)

    if fig_dir:
        out_dir = Path(fig_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = noise_name.replace('/', '_')
        stem = f"big_horizontal_{dataset_name}_{safe}"
        fig.savefig(out_dir / f"{stem}.png", dpi=150)
        # fig.savefig(out_dir / f"{stem}.pdf")
        print(f"  [big_horiz] {dataset_name}/{noise_name} -> {out_dir}")
    plt.close(fig)


_MAX_PLOT_ITERS = 20  # максимум отображаемых точек на графике сходимости

def _subsample_iters(df: "pd.DataFrame", max_points: int = _MAX_PLOT_ITERS) -> "pd.DataFrame":
    """Возвращает не более max_points равномерно распределённых строк из df."""
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[sorted(set(int(i) for i in idx))].reset_index(drop=True)


def plot_iteration_convergence(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Графики сходимости по итерациям: PSNR, SSIM, kernel_rmse — для каждого
    изображения в папке log_test.

    3 подграфика: PSNR, SSIM, RMSE ядра
    + один суммарный график со всеми изображениями.
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
        df = _subsample_iters(df)
        img_name = img_dir.name
        all_dfs[img_name] = df

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
            ax.set_ylabel('PSNR, дБ')
            ax.set_title(f'Сходимость PSNR — {decode(img_name)}', fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)

        if has_ssim:
            ax = axes[ax_idx]; ax_idx += 1
            mask = df['ssim'].notna()
            ax.plot(iters[mask], df['ssim'][mask], 'o-', color=PALETTE[2],
                    linewidth=2, markersize=5, label='SSIM')
            ax.set_xlabel('Итерация')
            ax.set_ylabel('SSIM')
            ax.set_title(f'Сходимость SSIM — {decode(img_name)}', fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)

        if has_krmse:
            ax = axes[ax_idx]; ax_idx += 1
            mask = df['kernel_rmse'].notna()
            ax.plot(iters[mask], df['kernel_rmse'][mask], '^-', color=PALETTE[4],
                    linewidth=2, markersize=5)
            ax.set_xlabel('Итерация')
            ax.set_ylabel('RMSE ядра')
            ax.set_title(f'Ошибка ядра (RMSE) — {decode(img_name)}', fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{decode(alg_label)}: Сходимость по итерациям — {decode(img_name)}',
                     fontsize=12, y=1.02)
        plt.tight_layout()

        fname = f"convergence_{img_name}"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
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
                             linewidth=1.5, markersize=4, label=decode_in_text(img_name))

            # SSIM
            if 'ssim' in df.columns and df['ssim'].notna().any():
                mask = df['ssim'].notna()
                axes[1].plot(iters[mask], df['ssim'][mask], marker=m, color=c,
                             linewidth=1.5, markersize=4, label=decode_in_text(img_name))

            # Kernel RMSE
            if 'kernel_rmse' in df.columns and df['kernel_rmse'].notna().any():
                mask = df['kernel_rmse'].notna()
                axes[2].plot(iters[mask], df['kernel_rmse'][mask], marker=m, color=c,
                             linewidth=1.5, markersize=4, label=decode_in_text(img_name))

        axes[0].set_xlabel('Итерация'); axes[0].set_ylabel('PSNR, дБ')
        axes[0].set_title('Сходимость PSNR', fontsize=TITLE_FONTSIZE)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Итерация'); axes[1].set_ylabel('SSIM')
        axes[1].set_title('Сходимость SSIM', fontsize=TITLE_FONTSIZE)
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Итерация'); axes[2].set_ylabel('RMSE ядра')
        axes[2].set_title('Ошибка ядра (RMSE)', fontsize=TITLE_FONTSIZE)
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'{decode(alg_label)}: Сходимость по итерациям (все изображения)',
                     fontsize=13, y=1.02)
        # Общая легенда под всеми графиками
        for _ax in axes:
            _h, _l = _ax.get_legend_handles_labels()
            if _h:
                fig.legend(_h, _l, loc='lower center', ncol=min(len(_h), 4),
                           fontsize=14, bbox_to_anchor=(0.5, 0.01),
                           frameon=True, edgecolor='#888888',
                           fancybox=False, handlelength=2.5,
                           markerscale=2.0, handletextpad=0.6)
                break
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)

        fname = "convergence_all"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
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
    Горизонтальная полоска эволюции ядра: один ряд от первой до
    последней итерации + итоговое ядро.
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

        kernels = []
        labels = []
        for kf in kernel_files:
            k = cv.imread(str(kf), cv.IMREAD_GRAYSCALE)
            if k is not None:
                kernels.append(k)
                iter_num = kf.stem.split("iter")[-1].lstrip("0") or "0"
                labels.append(f"Итерация {iter_num}")


        if not kernels:
            continue

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
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
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
                + r". Слева направо: от ранних итераций к итоговому ядру.}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  Эволюция ядер: {len(image_dirs)} изображений")



def _compute_psnr_blurred_map(df_grid: "pd.DataFrame",
                               all_results_df: Optional["pd.DataFrame"]) -> dict:
    """
    Возвращает {image_stem: psnr_blurred} по данным результатов или файлам.
    Используется для добавления ISNR в тепловые карты гиперпараметров,
    когда колонка psnr_blurred отсутствует в grid-CSV.
    """
    psnr_map: dict = {}
    if all_results_df is None or all_results_df.empty:
        return psnr_map


    if ('psnr_blurred' in all_results_df.columns
            and 'distorted_file' in all_results_df.columns):
        tmp = all_results_df.dropna(subset=['psnr_blurred'])
        for stem, grp in tmp.groupby(
                tmp['distorted_file'].apply(lambda f: Path(str(f)).stem)):
            psnr_map[stem] = float(grp['psnr_blurred'].mean())
        if psnr_map:
            return psnr_map

    import cv2 as _cv
    datasets_root = Path('images') / 'compare_data'
    seen: set = set()
    for _, row in all_results_df.drop_duplicates(subset=['distorted_file']).iterrows():
        distorted_file = str(row.get('distorted_file', '') or '')
        stem = Path(distorted_file).stem
        if not stem or stem in seen:
            continue
        seen.add(stem)
        op = str(row.get('original_path', '') or '')
        if not op or not Path(op).exists():
            continue
        dataset = str(row.get('dataset', '') or '')
        distorted_path = None
        if datasets_root.exists():
            for user_dir in datasets_root.iterdir():
                cand = user_dir / dataset / 'distorted' / distorted_file
                if cand.exists():
                    distorted_path = cand
                    break
        if distorted_path is None:
            continue
        img_d = _cv.imread(str(distorted_path), _cv.IMREAD_GRAYSCALE)
        img_o = _cv.imread(op, _cv.IMREAD_GRAYSCALE)
        if img_d is None or img_o is None or img_d.shape != img_o.shape:
            continue
        mse = float(np.mean((img_d.astype(np.float64) - img_o.astype(np.float64)) ** 2))
        if mse > 0:
            psnr_map[stem] = 10.0 * np.log10(255.0 ** 2 / mse)
    return psnr_map


def plot_hyperparam_heatmap(
    csv_path: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    all_results_df: Optional["pd.DataFrame"] = None,
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

    if 'psnr_blurred' not in df.columns and all_results_df is not None:
        _pmap = _compute_psnr_blurred_map(df, all_results_df)
        if _pmap:
            df = df.copy()
            df['psnr_blurred'] = df['image'].map(
                lambda s: _pmap.get(Path(str(s)).stem)
            )

    known_cols = {'image', 'psnr', 'ssim', 'time_sec', 'error_ratio'}
    param_cols = [c for c in df.columns if c not in known_cols]
    if len(param_cols) < 2:
        print(f"  Не удалось определить 2 параметра в {csv_path.name}")
        return

    p1, p2 = param_cols[0], param_cols[1]

    for metric, metric_label, cmap_name in [
        ('psnr', 'PSNR, дБ', 'YlOrRd'),
        ('ssim', 'SSIM', 'YlGnBu'),
    ] + ([('isnr', 'ISNR, дБ', 'RdYlGn')]
         if 'psnr_blurred' in df.columns and df['psnr_blurred'].notna().any()
         and (df.assign(isnr=df['psnr'] - df['psnr_blurred'])['isnr'].notna().any())
         else []):
        if metric == 'isnr' and 'isnr' not in df.columns:
            df = df.copy()
            df['isnr'] = df['psnr'] - df['psnr_blurred']

        if metric not in df.columns or df[metric].isna().all():
            continue

        pivot = df.groupby([p1, p2])[metric].mean().reset_index()
        heatmap = pivot.pivot(index=p1, columns=p2, values=metric)
        heatmap = heatmap.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(max(len(heatmap.columns) * 1.2, 7),
                                        max(len(heatmap.index) * 0.8, 5)))

        im = ax.imshow(heatmap.values, cmap=cmap_name, aspect='auto',
                       interpolation='nearest')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric_label, fontsize=13)

        ax.set_xticks(range(len(heatmap.columns)))
        ax.set_xticklabels([f"{v:.4g}" for v in heatmap.columns], rotation=45, ha='right', fontsize=11)
        ax.set_yticks(range(len(heatmap.index)))
        ax.set_yticklabels([f"{v:.4g}" for v in heatmap.index], fontsize=11)

        ax.set_xlabel(p2, fontsize=13)
        ax.set_ylabel(p1, fontsize=13)
        ax.set_title(f'{alg_label}: {metric_label} — сетка ({p1}, {p2})',
                     fontsize=TITLE_FONTSIZE)

        for i in range(len(heatmap.index)):
            for j in range(len(heatmap.columns)):
                val = heatmap.values[i, j]
                if pd.notna(val):
                    txt = f"{val:.2f}" if metric == 'psnr' else f"{val:.3f}"
                    norm_val = (val - np.nanmin(heatmap.values)) / (
                        np.nanmax(heatmap.values) - np.nanmin(heatmap.values) + 1e-12)
                    txt_color = 'white' if norm_val > 0.7 else 'black'
                    ax.text(j, i, txt, ha='center', va='center',
                            fontsize=11, color=txt_color, fontweight='bold')

        best_idx = np.unravel_index(np.nanargmax(heatmap.values), heatmap.values.shape)
        ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5),
                                   1, 1, fill=False, edgecolor='lime',
                                   linewidth=3))

        plt.tight_layout()

        fname = f"heatmap_{metric}_{p1}_{p2}"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
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

    avg = df.groupby([p1, p2])[['psnr', 'ssim']].mean().reset_index()

    best_row = avg.loc[avg['psnr'].idxmax()]
    best_p1 = best_row[p1]
    best_p2 = best_row[p2]

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
        ax1.set_xlabel(vary_param, fontsize=13)
        ax1.set_ylabel('PSNR, дБ', fontsize=13)
        ax1.set_title(f'Чувствительность PSNR к {vary_param}', fontsize=TITLE_FONTSIZE)
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)

        # SSIM
        ax2.plot(x, df_slice['ssim'].values, 's-', color=PALETTE[2],
                 linewidth=2, markersize=6)
        ax2.set_xlabel(vary_param, fontsize=13)
        ax2.set_ylabel('SSIM', fontsize=13)
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
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
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


def plot_error_ratio_histogram_single_v2(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    plot_error_ratio_histogram_single, но без линии r=3.
    Сохраняется в файл error_ratio_histogram_v2.
    """
    er = er_values.dropna()
    if len(er) == 0:
        print("  [v2] Нет данных отношения ошибок для гистограммы.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, max(er.max(), 5) + 0.5, 0.5)
    ax.hist(er, bins=bins, color=PALETTE[0], edgecolor='black', alpha=0.85)
    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Количество изображений')
    ax.set_title(f'Распределение отношения ошибок — {decode(alg_label)}',
                 fontsize=TITLE_FONTSIZE)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if fig_dir:
        # fig.savefig(Path(fig_dir) / "error_ratio_histogram_v2.pdf")
        fig.savefig(Path(fig_dir) / "error_ratio_histogram_v2.png", dpi=200)
    plt.close(fig)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{figures/error_ratio_histogram_v2.pdf}" "\n"
            r"\caption{Распределение отношения ошибок для алгоритма "
            + decode(alg_label) + r".}" "\n"
            r"\label{fig:er_hist_v2_" + _safe_label(alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / "error_ratio_histogram_v2.tex", tex)


def plot_success_rate_single_v2(
    er_values: pd.Series,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    suffix: str = "",
):
    """
    plot_success_rate_single, но без линии r=3.
    Сохраняется в файл success_rate{suffix}_v2.
    """
    er = er_values.dropna()
    if len(er) == 0:
        print("  [v2] Нет данных отношения ошибок для success-rate.")
        return

    x_max = float(np.clip(np.nanpercentile(er, 99) * 1.2, 3.5, 10.0))
    thresholds = np.arange(1.0, x_max + 0.05, 0.05)
    sr = [(er <= t).sum() / len(er) * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, sr, linewidth=2, color=PALETTE[0],
            label=decode(alg_label), marker='s', markersize=8,
            markerfacecolor='none', markevery=20)
    ax.set_xlabel('Отношение ошибок')
    ax.set_ylabel('Доля успешных (%)')
    ax.set_title(f'Доля успешных / отношение ошибок — {decode(alg_label)}',
                 fontsize=TITLE_FONTSIZE)
    ax.set_xlim(1, x_max); ax.set_ylim(0, 105)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"success_rate{suffix}_v2"
    if fig_dir:
        # fig.savefig(Path(fig_dir) / f"{fname}.pdf")
        fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200)
    plt.close(fig)

    if tex_dir:
        tex = (
            r"\begin{figure}[htbp]" "\n"
            r"\centering" "\n"
            r"\includegraphics[width=0.8\textwidth]{" + f"figures/{fname}.pdf" + r"}" "\n"
            r"\caption{Кумулятивная кривая доли успешных для алгоритма "
            + decode(alg_label) + r".}" "\n"
            r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
            r"\end{figure}"
        )
        save_tex(Path(tex_dir) / f"{fname}.tex", tex)


def plot_psnr_ssim_bars_single_v2(
    df: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Раздельные столбчатые диаграммы PSNR и SSIM (по одному файлу на метрику).

    Сохраняет два файла: psnr_bar_v2.{pdf,png} и ssim_bar_v2.{pdf,png}.
    """
    df_m = df.dropna(subset=['psnr', 'ssim']).copy()
    if df_m.empty:
        print("  [v2] Нет данных PSNR/SSIM для столбчатой диаграммы.")
        return

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
    colors[-1] = (0.55, 0.55, 0.55, 1.0)
    img_labels = [decode(v) for v in grp['_img'].values]

    for metric, ylabel, fmt, fname, label_for_tex in [
        ('psnr_mean', 'PSNR, дБ', '{:.1f}', 'psnr_bar_v2', 'PSNR'),
        ('ssim_mean', 'SSIM',      '{:.3f}', 'ssim_bar_v2', 'SSIM'),
    ]:
        fig, ax = plt.subplots(figsize=(max(n * 0.7 + 2, 8), 5))
        x = np.arange(n)
        bars = ax.bar(x, grp[metric].values, color=colors,
                      alpha=0.9, edgecolor='grey', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(img_labels, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{label_for_tex} по изображениям — {decode(alg_label)}',
                     fontsize=TITLE_FONTSIZE)
        ax.set_ylim(bottom=_bar_ymin(grp[metric].values, metric))
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        offset = grp[metric].max() * 0.01 if grp[metric].max() else 0.01
        for bar, val in zip(bars, grp[metric].values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    fmt.format(val), ha='center', va='bottom', fontsize=10)
        plt.tight_layout()

        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf")
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200)
            ax.set_ylim(bottom=0)
            # fig.savefig(Path(fig_dir) / f"{fname}_full.pdf")
            fig.savefig(Path(fig_dir) / f"{fname}_full.png", dpi=200)
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{figures/" + fname + r".pdf}" "\n"
                r"\caption{Средний " + label_for_tex
                + r" по изображениям для алгоритма " + decode(alg_label) + r".}" "\n"
                r"\label{fig:" + fname + r"_" + _safe_label(alg_label) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)


def plot_psnr_ssim_per_image_per_dataset_v2(
    df: pd.DataFrame,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Для каждого датасета строит две отдельные картинки: PSNR и SSIM.
    Столбцы узкие, столбцы цветные по датасету.

    Файлы: per_image_{psnr|ssim}_{dataset}_v2.{pdf,png}.
    """
    df_m = df.dropna(subset=['psnr', 'ssim']).copy()
    if df_m.empty or 'dataset' not in df_m.columns:
        return
    df_m['_img'] = df_m['distorted_file'].apply(lambda x: Path(x).stem.rsplit('_', 1)[0])

    datasets = df_m['dataset'].unique()

    for ds_name in datasets:
        sub = df_m[df_m['dataset'] == ds_name]
        grp = sub.groupby('_img').agg(
            psnr_mean=('psnr', 'mean'), ssim_mean=('ssim', 'mean')
        ).reset_index().sort_values('_img')
        if grp.empty:
            continue

        n = len(grp)
        x = np.arange(n)
        bar_w = 0.55

        grp['_orig'] = grp['_img'].str.split('_').str[0]
        orig_names = grp['_orig'].unique()
        orig_colors = dict(zip(sorted(orig_names), _colormap_bars(len(orig_names))))
        bar_colors = [orig_colors[o] for o in grp['_orig']]
        labels = [decode_in_text(v) for v in grp['_img'].values]

        for metric, ylabel, fmt, m_short, m_label in [
            ('psnr_mean', 'PSNR, дБ', '{:.1f}', 'psnr', 'PSNR'),
            ('ssim_mean', 'SSIM',      '{:.3f}', 'ssim', 'SSIM'),
        ]:
            fig, ax = plt.subplots(figsize=(max(n * 0.5 + 2, 6), 4.5))
            bars = ax.bar(x, grp[metric].values, width=bar_w, color=bar_colors,
                          edgecolor='grey', linewidth=0.4, alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{m_label} по изображениям — {decode(alg_label)} '
                         f'/ {decode(ds_name)}', fontsize=TITLE_FONTSIZE)
            ax.set_ylim(bottom=_bar_ymin(grp[metric].values, metric))
            ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True)
            offset = grp[metric].max() * 0.01 if grp[metric].max() else 0.01
            for bar, val in zip(bars, grp[metric].values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + offset,
                        fmt.format(val), ha='center', va='bottom',
                        fontsize=10, rotation=0)
            plt.tight_layout()

            fname = f"per_image_{m_short}_{ds_name}_v2"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
                ax.set_ylim(bottom=0)
                # fig.savefig(Path(fig_dir) / f"{fname}_full.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}_full.png", dpi=200, bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=\textwidth]{figures/"
                    + fname + r".pdf}" "\n"
                    r"\caption{" + m_label + r" по изображениям, алгоритм "
                    + decode(alg_label) + r", набор данных " + decode(ds_name) + r".}" "\n"
                    r"\label{fig:" + _safe_label(fname + "_" + alg_label) + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)


def plot_iteration_convergence_v2(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    Расширенная версия: сетка 2 х 2 на одно изображение.

    Сверху  — PSNR (слева) и SSIM (справа) восстановленного изображения.
    Снизу — ошибка ядра: RMSE (L2-норма, слева) и MAE (L1-норма, справа).

    Файлы: convergence_{img}_v2.{pdf,png} и общий convergence_all_v2.
    """
    iter_results_dir = Path(iter_results_dir)
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    image_dirs = sorted([
        d for d in iter_results_dir.iterdir()
        if d.is_dir() and (d / "iterations_log.csv").exists()
    ])
    if not image_dirs:
        print("  [v2] Нет данных итераций.")
        return

    all_dfs = {}
    for img_dir in image_dirs:
        df = pd.read_csv(img_dir / "iterations_log.csv")
        if df.empty:
            continue
        df = _subsample_iters(df)
        all_dfs[img_dir.name] = df

        iters = df['local_iter']
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        ax_psnr, ax_ssim = axes[0, 0], axes[0, 1]
        ax_l2,   ax_l1   = axes[1, 0], axes[1, 1]

        if 'psnr' in df.columns and df['psnr'].notna().any():
            m = df['psnr'].notna()
            ax_psnr.plot(iters[m], df['psnr'][m], 'o-', color=PALETTE[0],
                         linewidth=2, markersize=4, label='PSNR')
        ax_psnr.set_xlabel('Итерация'); ax_psnr.set_ylabel('PSNR, дБ')
        ax_psnr.set_title('Сходимость PSNR изображения', fontsize=TITLE_FONTSIZE)
        ax_psnr.grid(True, alpha=0.3)

        if 'ssim' in df.columns and df['ssim'].notna().any():
            m = df['ssim'].notna()
            ax_ssim.plot(iters[m], df['ssim'][m], 'o-', color=PALETTE[2],
                         linewidth=2, markersize=4, label='SSIM')
        ax_ssim.set_xlabel('Итерация'); ax_ssim.set_ylabel('SSIM')
        ax_ssim.set_title('Сходимость SSIM изображения', fontsize=TITLE_FONTSIZE)
        ax_ssim.grid(True, alpha=0.3)

        if 'kernel_rmse' in df.columns and df['kernel_rmse'].notna().any():
            m = df['kernel_rmse'].notna()
            ax_l2.plot(iters[m], df['kernel_rmse'][m], '^-', color=PALETTE[4],
                       linewidth=2, markersize=4, label='RMSE (L2)')
        ax_l2.set_xlabel('Итерация'); ax_l2.set_ylabel('RMSE ядра')
        ax_l2.set_title('Ошибка ядра — L2-норма (RMSE)', fontsize=TITLE_FONTSIZE)
        ax_l2.grid(True, alpha=0.3)

        if 'kernel_mae' in df.columns and df['kernel_mae'].notna().any():
            m = df['kernel_mae'].notna()
            ax_l1.plot(iters[m], df['kernel_mae'][m], 'D-', color=PALETTE[6],
                       linewidth=2, markersize=4, label='MAE (L1)')
            ax_l1.set_ylabel('MAE ядра')
            ax_l1.set_title('Ошибка ядра — L1-норма (MAE)', fontsize=TITLE_FONTSIZE)
        else:
            ax_l1.text(0.5, 0.5, 'kernel_mae отсутствует в логах\n'
                       '(перезапустите прогон с обновлённым логгером)',
                       ha='center', va='center', transform=ax_l1.transAxes,
                       fontsize=9, color='grey')
            ax_l1.set_title('Ошибка ядра — L1-норма (MAE)', fontsize=TITLE_FONTSIZE)
        ax_l1.set_xlabel('Итерация')
        ax_l1.grid(True, alpha=0.3)

        fig.suptitle(f'{decode(alg_label)}: Сходимость по итерациям — '
                     f'{decode(img_dir.name)}', fontsize=12, y=1.00)
        plt.tight_layout()

        fname = f"convergence_{img_dir.name}_v2"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{figures/" + fname + r".pdf}" "\n"
                r"\caption{Сходимость алгоритма " + decode(alg_label)
                + r" на изображении " + img_dir.name.replace("_", r"\_")
                + r": PSNR/SSIM изображения (сверху) и L2/L1 норма ошибки ядра (снизу).}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    if len(all_dfs) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
        ax_psnr, ax_ssim = axes[0, 0], axes[0, 1]
        ax_l2,   ax_l1   = axes[1, 0], axes[1, 1]
        colors = _get_palette_cycle(); markers = _get_marker_cycle()

        for name, df in all_dfs.items():
            c = next(colors); mk = next(markers)
            iters = df['local_iter']
            lbl = decode_in_text(name)
            for ax, col in [(ax_psnr, 'psnr'), (ax_ssim, 'ssim'),
                            (ax_l2, 'kernel_rmse'), (ax_l1, 'kernel_mae')]:
                if col in df.columns and df[col].notna().any():
                    m = df[col].notna()
                    ax.plot(iters[m], df[col][m], marker=mk, color=c,
                            linewidth=1.4, markersize=4, label=lbl)

        for ax, t, y in [
            (ax_psnr, 'Сходимость PSNR', 'PSNR, дБ'),
            (ax_ssim, 'Сходимость SSIM', 'SSIM'),
            (ax_l2,   'Ошибка ядра — L2-норма (RMSE)', 'RMSE ядра'),
            (ax_l1,   'Ошибка ядра — L1-норма (MAE)', 'MAE ядра'),
        ]:
            ax.set_xlabel('Итерация'); ax.set_ylabel(y)
            ax.set_title(t, fontsize=TITLE_FONTSIZE)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'{decode(alg_label)}: Сходимость по итерациям '
                     f'(все изображения)', fontsize=13, y=1.00)
        for _ax in [ax_psnr, ax_ssim, ax_l2, ax_l1]:
            _h, _l = _ax.get_legend_handles_labels()
            if _h:
                fig.legend(_h, _l, loc='lower center', ncol=min(len(_h), 4),
                           fontsize=14, bbox_to_anchor=(0.5, 0.01),
                           frameon=True, edgecolor='#888888',
                           fancybox=False, handlelength=2.5,
                           markerscale=2.0, handletextpad=0.6)
                break
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.22)

        fname = "convergence_all_v2"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=\textwidth]{figures/" + fname + r".pdf}" "\n"
                r"\caption{Суммарные кривые сходимости алгоритма "
                + decode(alg_label) + r" на всех тестовых изображениях.}" "\n"
                r"\label{fig:" + fname + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  [v2] Кривые сходимости 2x2: {len(all_dfs)} изображений")


def plot_kernel_evolution_strip_v2(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    distorted_dir=None,  # Optional[Union[Path, List[Path]]]
):
    """
    Сокращённая эволюция: ровно 4 ядра — три равномерно распределённых
    итерации + итоговое. Для каждого изображения сохраняется до 8 вариантов:

    стили (hot — красно-жёлтый, viridis — сине-жёлтый),
    компоновки (strip 1×N, grid 2×N) и
    режимы — с включённым в фигуру смазанным изображением (with_blur)
    и без него (only).

    Файлы: kernel_evo4_{style}_{layout}_{mode}_{img}.{pdf,png}.

    distorted_dir : Optional[Path]
        Каталог со смазанными изображениями. Если None, ищем сначала по пути
        images/compare_data/<user>/<iter_results_dir.name>/distorted
        (перебираем все поддиректории пользователей), затем напрямую.
    """

    iter_results_dir = Path(iter_results_dir)
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    if distorted_dir is None:
        _ds_name = iter_results_dir.name
        _base = Path("images") / "compare_data"
        _all: list = []
        if _base.exists():
            for _user_dir in sorted(_base.iterdir()):
                if _user_dir.is_dir():
                    _g = _user_dir / _ds_name / "distorted"
                    if _g.exists():
                        _all.append(_g)
        _direct = _base / _ds_name / "distorted"
        if _direct.exists() and _direct not in _all:
            _all.append(_direct)
        distorted_dirs_list: list = _all
    elif isinstance(distorted_dir, (list, tuple)):
        distorted_dirs_list = [Path(d) for d in distorted_dir]
    else:
        distorted_dirs_list = [Path(distorted_dir)]

    image_dirs = sorted([
        d for d in iter_results_dir.iterdir()
        if d.is_dir() and (d / "kernels").exists()
    ])
    if not image_dirs:
        print("  [v2] Нет данных ядер для построения эволюции.")
        return

    def _find_blurred(img_name: str) -> Optional[np.ndarray]:
        if not distorted_dirs_list:
            return None

        def _load_and_thumb(path: Path) -> Optional[np.ndarray]:
            im = cv.imread(str(path), cv.IMREAD_UNCHANGED)
            if im is None:
                return None
            if im.ndim == 3 and im.shape[2] >= 3:
                im = cv.cvtColor(im[:, :, :3], cv.COLOR_BGR2RGB)
            h, w = im.shape[:2]
            max_dim = 256
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                im = cv.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv.INTER_AREA)
            return im

        for _ddir in distorted_dirs_list:
            if not Path(_ddir).exists():
                continue
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                p = Path(_ddir) / f"{img_name}{ext}"
                if p.exists():
                    return _load_and_thumb(p)
            cands = sorted(Path(_ddir).glob(f"{img_name}_*"))
            if cands:
                return _load_and_thumb(cands[0])
        return None

    for img_dir in image_dirs:
        img_name = img_dir.name
        kernels_dir = img_dir / "kernels"
        kernel_files = sorted(kernels_dir.glob("kernel_s0_iter*.png"))
        final_kernel_path = img_dir / "kernel_final.png"
        if not kernel_files or not final_kernel_path.exists():
            continue

        n_iter = len(kernel_files)
        if n_iter >= 3:
            idx = np.linspace(0, n_iter - 1, 3, dtype=int)
            idx = sorted(set(int(i) for i in idx))
            while len(idx) < 3 and len(idx) < n_iter:
                for i in range(n_iter):
                    if i not in idx:
                        idx.append(i); break
            sel = [kernel_files[i] for i in idx[:3]]
        else:
            sel = list(kernel_files)

        kernels = []
        labels = []
        for kf in sel:
            k = cv.imread(str(kf), cv.IMREAD_GRAYSCALE)
            if k is None:
                continue
            kernels.append(k)
            it_str = kf.stem.split('iter')[-1].lstrip('0') or '0'
            labels.append(f"Итерация {it_str}")

        # k_final = cv.imread(str(final_kernel_path), cv.IMREAD_GRAYSCALE)

        n = len(kernels)
        if n == 0:
            continue

        blurred_img = _find_blurred(img_name)

        gt_kernel_img: Optional[np.ndarray] = None
        _gt_candidates = [img_dir / 'gt_kernel.png', img_dir / 'kernel_gt.png',
                          img_dir / 'kernels' / 'gt_kernel.png']
        for _gtp in _gt_candidates:
            if _gtp.exists():
                _k = cv.imread(str(_gtp), cv.IMREAD_GRAYSCALE)
                if _k is not None:
                    gt_kernel_img = _k
                    break
        if gt_kernel_img is None:
            _log_csv = img_dir / 'iterations_log.csv'
            if _log_csv.exists():
                try:
                    _log_df = pd.read_csv(_log_csv)
                    for _col in ('gt_kernel_path', 'gt_kernel'):
                        if _col in _log_df.columns:
                            _vals = _log_df[_col].dropna()
                            if not _vals.empty:
                                _k = cv.imread(str(_vals.iloc[0]), cv.IMREAD_GRAYSCALE)
                                if _k is not None:
                                    gt_kernel_img = _k
                                    break
                except Exception:
                    pass

        styles = [('hot', 'hot'), ('viridis', 'viridis')]
        layouts = ['strip', 'grid']
        modes = ['only']
        if blurred_img is not None:
            modes.append('with_blur')

        restored_dir = img_dir / "restored"
        restored_at_iter: Dict[str, Optional[np.ndarray]] = {}
        if restored_dir.exists():
            for lbl, kf in zip(labels, sel):
                it_str = kf.stem.split('iter')[-1].lstrip('0') or '0'
                for ext in ('.png', '.jpg', '.jpeg'):
                    cands = list(restored_dir.glob(f"*iter{it_str}{ext}")) + \
                            list(restored_dir.glob(f"*iter{it_str.zfill(4)}{ext}"))
                    if cands:
                        im = cv.imread(str(cands[0]), cv.IMREAD_UNCHANGED)
                        if im is not None:
                            if im.ndim == 3 and im.shape[2] >= 3:
                                im = cv.cvtColor(im[:, :, :3], cv.COLOR_BGR2RGB)
                            restored_at_iter[lbl] = im
                            break
                else:
                    restored_at_iter[lbl] = None
        if len(restored_at_iter) == len(labels) and any(v is not None for v in restored_at_iter.values()):
            modes.append('with_restored')

        for style_tag, cmap_name in styles:
            for layout in layouts:
                for mode in modes:
                    include_blur = (mode == 'with_blur')
                    use_gt_in_grid = (layout == 'grid' and gt_kernel_img is not None
                                      and not include_blur)
                    n_panels = n + (1 if include_blur else 0) + (1 if use_gt_in_grid else 0)

                    if layout == 'strip':
                        nrows, ncols = 1, n + (1 if include_blur else 0)
                        fw = max(nrows * ncols * 1.9, 5)
                        fh = 2.7
                    else:
                        ncols = max(2, int(np.ceil(np.sqrt(n_panels))))
                        nrows = int(np.ceil(n_panels / ncols))
                        fw = ncols * 2.4
                        fh = nrows * 2.4 + 0.3

                    if mode == 'with_restored':
                        n_panels = n
                        nrows, ncols = 2, n_panels
                        fw = max(n_panels * 1.9, 5)
                        fh = 4.5
                        fig, axes2d = plt.subplots(nrows, ncols, figsize=(fw, fh))
                        axes_top = np.atleast_1d(axes2d[0]).ravel()
                        axes_bot = np.atleast_1d(axes2d[1]).ravel()
                        for i in range(n):
                            axes_top[i].imshow(kernels[i], cmap=cmap_name,
                                               interpolation='nearest')
                            axes_top[i].set_title(labels[i], fontsize=9)
                            axes_top[i].axis('off')
                            rest_im = restored_at_iter.get(labels[i])
                            if rest_im is not None:
                                if rest_im.ndim == 2:
                                    axes_bot[i].imshow(rest_im, cmap='gray')
                                else:
                                    axes_bot[i].imshow(rest_im)
                            else:
                                axes_bot[i].text(0.5, 0.5, 'нет\nданных',
                                                 ha='center', va='center',
                                                 transform=axes_bot[i].transAxes, fontsize=8)
                            axes_bot[i].axis('off')
                        fig.suptitle(f'{decode(alg_label)}: Ядра / восстановление — '
                                     f'{decode(img_name)}', fontsize=11, y=1.02)
                        plt.tight_layout()
                        fname = f"kernel_evo4_{style_tag}_strip_restored_{img_name}"
                        if fig_dir:
                            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                            fig.savefig(Path(fig_dir) / f"{fname}.png",
                                        dpi=200, bbox_inches='tight')
                        plt.close(fig)
                        if tex_dir:
                            tex = (
                                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                                r"\includegraphics[width=0.85\textwidth]{figures/"
                                + fname + r".pdf}" "\n"
                                r"\caption{Ядра (сверху) и восстановленные изображения (снизу) "
                                + r"алгоритма " + decode(alg_label)
                                + r" на изображении "
                                + img_name.replace("_", r"\_")
                                + r" (промежуточные итерации).}" "\n"
                                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                                r"\end{figure}"
                            )
                            save_tex(Path(tex_dir) / f"{fname}.tex", tex)
                        continue

                    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
                    axes = np.atleast_1d(axes).ravel()

                    panel_idx = 0
                    if include_blur:
                        ax = axes[panel_idx]
                        if blurred_img.ndim == 2:
                            ax.imshow(blurred_img, cmap='gray')
                        else:
                            ax.imshow(blurred_img)
                        ax.set_title('Смазанное', fontsize=10)
                        ax.axis('off')
                        panel_idx += 1

                    for i in range(n):
                        ax = axes[panel_idx]
                        ax.imshow(kernels[i], cmap=cmap_name,
                                  interpolation='nearest')
                        ax.set_title(labels[i], fontsize=10)
                        ax.axis('off')
                        panel_idx += 1

                    if use_gt_in_grid and panel_idx < len(axes):
                        ax = axes[panel_idx]
                        ax.imshow(gt_kernel_img, cmap=cmap_name,
                                  interpolation='nearest')
                        ax.set_title('Истинное ядро', fontsize=10)
                        ax.axis('off')
                        panel_idx += 1

                    for j in range(panel_idx, len(axes)):
                        axes[j].axis('off')

                    fig.suptitle(f'{decode(alg_label)}: Эволюция ядра — '
                                 f'{decode(img_name)}', fontsize=11, y=1.02)
                    plt.tight_layout()

                    fname = f"kernel_evo4_{style_tag}_{layout}_{mode}_{img_name}"
                    if fig_dir:
                        # fig.savefig(Path(fig_dir) / f"{fname}.pdf",
                                    # bbox_inches='tight')
                        fig.savefig(Path(fig_dir) / f"{fname}.png",
                                    dpi=200, bbox_inches='tight')
                    plt.close(fig)

                    if tex_dir:
                        tex = (
                            r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                            r"\includegraphics[width=0.85\textwidth]{figures/"
                            + fname + r".pdf}" "\n"
                            r"\caption{Эволюция ядра алгоритма "
                            + decode(alg_label) + r" на изображении "
                            + img_name.replace("_", r"\_")
                            + r" (3 промежуточные итерации + итоговое ядро"
                            + (r", вместе со смазанным изображением)."
                               if include_blur else r").")
                            + r"}" "\n"
                            r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                            r"\end{figure}"
                        )
                        save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  [v2] Эволюция ядра (4 панели, 4–8 вариантов на изображение): "
          f"{len(image_dirs)} изображений")


def plot_kernel_evolution_strip_v3(
    iter_results_dir: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    n_frames: int = 5,
):
    """
    Расширенная эволюция: до n_frames равномерно распределённых ядер с
    восстановленным изображением под каждым.

    Все ядра масштабируются до размера финального ядра,
    чтобы ядра разных уровней пирамиды отображались одинакового размера.

    Строки:
      0 — ядра (colormap hot)
      1 — восстановленные изображения (grayscale / RGB)

    Файлы: kernel_evo6_restored_{img_name}.{pdf,png}.
    """

    iter_results_dir = Path(iter_results_dir)
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    image_dirs = sorted([
        d for d in iter_results_dir.iterdir()
        if d.is_dir() and (d / "kernels").exists()
    ])
    if not image_dirs:
        print("  [v3] Нет данных ядер.")
        return

    _ds_name = iter_results_dir.name
    _base = Path("images") / "compare_data"
    _dataset_dirs: list = []
    if _base.exists():
        for _user_dir in sorted(_base.iterdir()):
            if _user_dir.is_dir():
                _cand = _user_dir / _ds_name
                if _cand.is_dir():
                    _dataset_dirs.append(_cand)
    _direct = _base / _ds_name
    if _direct.is_dir() and _direct not in _dataset_dirs:
        _dataset_dirs.append(_direct)

    def _find_in_subdirs(img_stem: str, subdir_name: str,
                         is_gray: bool = False) -> Optional[np.ndarray]:
        """
        Ищет файл в <dataset_dir>/<subdir_name>/ по всем датасетам.

        Порядок попыток:
        1. Точное совпадение: <img_stem>.*
        2. Совпадение-префикс: img_stem начинается с <file_stem>_  (для originals)
        3. Совпадение-фрагмент: img_stem содержит _<file_stem>_ или _<file_stem>  (для kernels)
        """
        def _load(path: Path) -> Optional[np.ndarray]:
            flags = cv.IMREAD_GRAYSCALE if is_gray else cv.IMREAD_UNCHANGED
            return cv.imread(str(path), flags)

        for _ddir in _dataset_dirs:
            _folder = _ddir / subdir_name
            if not _folder.exists():
                continue
            _exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            for ext in _exts:
                _p = _folder / f"{img_stem}{ext}"
                if _p.exists():
                    im = _load(_p)
                    if im is not None:
                        return im
            try:
                candidates = [f for f in _folder.iterdir()
                              if f.suffix.lower() in _exts]
            except Exception:
                continue
            for f in candidates:
                fs = f.stem.lower()
                if img_stem.lower().startswith(fs + '_') or img_stem.lower() == fs:
                    im = _load(f)
                    if im is not None:
                        return im
            for f in candidates:
                fs = f.stem.lower()
                parts = img_stem.lower().split('_')
                if fs in parts:
                    im = _load(f)
                    if im is not None:
                        return im
                for seg_len in range(2, len(parts)):
                    for start in range(len(parts) - seg_len + 1):
                        if '_'.join(parts[start:start+seg_len]) == fs:
                            im = _load(f)
                            if im is not None:
                                return im
        return None

    def _thumb(im: np.ndarray, max_dim: int = 320) -> np.ndarray:
        h, w = im.shape[:2]
        if max(h, w) <= max_dim:
            return im
        scale = max_dim / max(h, w)
        return cv.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv.INTER_AREA)

    for img_dir in image_dirs:
        img_name = img_dir.name
        kernels_dir = img_dir / "kernels"
        kernel_files = sorted(kernels_dir.glob("kernel_s0_iter*.png"))
        if not kernel_files:
            continue

        final_k_path = img_dir / "kernel_final.png"
        _ref_k = cv.imread(str(final_k_path if final_k_path.exists()
                               else kernel_files[-1]), cv.IMREAD_GRAYSCALE)
        if _ref_k is None:
            continue
        target_h, target_w = _ref_k.shape

        n_total = len(kernel_files)
        n_sel = min(n_frames, n_total)
        idx = np.linspace(0, n_total - 1, n_sel, dtype=int)
        sel = [kernel_files[int(i)] for i in sorted(set(int(i) for i in idx))]

        kernels = []
        labels = []
        iter_nums = []
        for kf in sel:
            k = cv.imread(str(kf), cv.IMREAD_GRAYSCALE)
            if k is None:
                continue
            if k.shape != (target_h, target_w):
                canvas = np.zeros((target_h, target_w), dtype=k.dtype)
                kh, kw = k.shape
                y0 = (target_h - kh) // 2
                x0 = (target_w - kw) // 2
                ky0 = max(0, -y0); kx0 = max(0, -x0)
                cy0 = max(0, y0);  cx0 = max(0, x0)
                ch = min(target_h - cy0, kh - ky0)
                cw = min(target_w - cx0, kw - kx0)
                canvas[cy0:cy0+ch, cx0:cx0+cw] = k[ky0:ky0+ch, kx0:kx0+cw]
                k = canvas
            kernels.append(k)
            it_str = kf.stem.split('iter')[-1].lstrip('0') or '0'
            labels.append(f"Итерация {it_str}")
            iter_nums.append(it_str)

        if not kernels:
            continue

        n = len(kernels)

        restored_dir = img_dir / "restored"
        restored_imgs: list = []
        for it_str in iter_nums:
            found = None
            if restored_dir.exists():
                for ext in ('.png', '.jpg', '.jpeg'):
                    cands = (list(restored_dir.glob(f"*iter{it_str}{ext}"))
                             + list(restored_dir.glob(f"*iter{it_str.zfill(4)}{ext}")))
                    if cands:
                        im = cv.imread(str(cands[0]), cv.IMREAD_UNCHANGED)
                        if im is not None:
                            if im.ndim == 3 and im.shape[2] >= 3:
                                im = cv.cvtColor(im[:, :, :3], cv.COLOR_BGR2RGB)
                            found = _thumb(im)
                        break
            restored_imgs.append(found)

        has_restored = any(v is not None for v in restored_imgs)

        blurred_img: Optional[np.ndarray] = None
        _bi = _find_in_subdirs(img_name, 'distorted', is_gray=False)
        if _bi is not None:
            if _bi.ndim == 3 and _bi.shape[2] >= 3:
                _bi = cv.cvtColor(_bi[:, :, :3], cv.COLOR_BGR2RGB)
            blurred_img = _thumb(_bi)
        if blurred_img is None:
            for _fn in ('blurred.png', 'blurred.jpg', 'distorted.png'):
                _p = img_dir / _fn
                if _p.exists():
                    _im = cv.imread(str(_p), cv.IMREAD_UNCHANGED)
                    if _im is not None:
                        if _im.ndim == 3 and _im.shape[2] >= 3:
                            _im = cv.cvtColor(_im[:, :, :3], cv.COLOR_BGR2RGB)
                        blurred_img = _thumb(_im)
                        break

        gt_kernel_img: Optional[np.ndarray] = None
        for _gtp in (img_dir / 'gt_kernel.png', img_dir / 'kernel_gt.png',
                     img_dir / 'kernels' / 'gt_kernel.png'):
            if _gtp.exists():
                _k = cv.imread(str(_gtp), cv.IMREAD_GRAYSCALE)
                if _k is not None:
                    gt_kernel_img = _k
                    break
        if gt_kernel_img is None:
            gt_kernel_img = _find_in_subdirs(img_name, 'ground_truth_filters', is_gray=True)
        _log_csv = img_dir / 'iterations_log.csv'
        if gt_kernel_img is None and _log_csv.exists():
            try:
                _log_df = pd.read_csv(_log_csv)
                for _col in ('gt_kernel_path', 'gt_kernel'):
                    if _col in _log_df.columns:
                        _vals = _log_df[_col].dropna()
                        if not _vals.empty:
                            _k = cv.imread(str(_vals.iloc[0]), cv.IMREAD_GRAYSCALE)
                            if _k is not None:
                                gt_kernel_img = _k
                                break
            except Exception:
                pass

        orig_img: Optional[np.ndarray] = None
        for _fn in ('original.png', 'original.jpg', 'orig.png'):
            _p = img_dir / _fn
            if _p.exists():
                _im = cv.imread(str(_p), cv.IMREAD_UNCHANGED)
                if _im is not None:
                    if _im.ndim == 3 and _im.shape[2] >= 3:
                        _im = cv.cvtColor(_im[:, :, :3], cv.COLOR_BGR2RGB)
                    orig_img = _thumb(_im)
                    break
        if orig_img is None:
            _im = _find_in_subdirs(img_name, 'originals', is_gray=False)
            if _im is not None:
                if _im.ndim == 3 and _im.shape[2] >= 3:
                    _im = cv.cvtColor(_im[:, :, :3], cv.COLOR_BGR2RGB)
                orig_img = _thumb(_im)
        if orig_img is None and _log_csv.exists():
            try:
                _log_df = pd.read_csv(_log_csv)
                for _col in ('original_path', 'orig_path'):
                    if _col in _log_df.columns:
                        _vals = _log_df[_col].dropna()
                        if not _vals.empty:
                            _p2 = Path(str(_vals.iloc[0]))
                            if _p2.exists():
                                _im = cv.imread(str(_p2), cv.IMREAD_UNCHANGED)
                                if _im is not None:
                                    if _im.ndim == 3 and _im.shape[2] >= 3:
                                        _im = cv.cvtColor(_im[:, :, :3], cv.COLOR_BGR2RGB)
                                    orig_img = _thumb(_im)
                                    break
            except Exception:
                pass

        has_gt_col = (gt_kernel_img is not None) or (orig_img is not None)

        for cmap_name, cmap_tag in [('hot', 'hot'), ('viridis', 'viridis')]:
            nrows = 2 if has_restored else 1
            n_cols = 1 + n + (1 if has_gt_col else 0)
            fw = max(n_cols * 2.8, 8.0)
            fh = nrows * 3.2 + 1.0

            fig, axes2d = plt.subplots(nrows, n_cols, figsize=(fw, fh),
                                       squeeze=False)
            axes_top = axes2d[0]
            axes_bot = axes2d[1] if has_restored else None

            axes_top[0].axis('off')
            if has_restored:
                if blurred_img is not None:
                    if blurred_img.ndim == 2:
                        axes_bot[0].imshow(blurred_img, cmap='gray')
                    else:
                        axes_bot[0].imshow(blurred_img)
                    axes_bot[0].set_title('Искажённое', fontsize=16)
                else:
                    axes_bot[0].text(0.5, 0.5, 'нет\nданных',
                                     ha='center', va='center',
                                     transform=axes_bot[0].transAxes, fontsize=16,
                                     color='grey')
                axes_bot[0].axis('off')

            for i in range(n):
                col = i + 1
                axes_top[col].imshow(kernels[i], cmap=cmap_name, interpolation='nearest')
                axes_top[col].set_title(labels[i], fontsize=16)
                axes_top[col].axis('off')

                if has_restored:
                    rim = restored_imgs[i]
                    if rim is not None:
                        if rim.ndim == 2:
                            axes_bot[col].imshow(rim, cmap='gray')
                        else:
                            axes_bot[col].imshow(rim)
                    else:
                        axes_bot[col].text(0.5, 0.5, 'нет\nданных',
                                           ha='center', va='center',
                                           transform=axes_bot[col].transAxes, fontsize=16,
                                           color='grey')
                    axes_bot[col].axis('off')

            if has_gt_col:
                gt_col = 1 + n
                ax_gt_top = axes_top[gt_col]
                if gt_kernel_img is not None:
                    _gk = gt_kernel_img
                    if _gk.shape != (target_h, target_w):
                        canvas = np.zeros((target_h, target_w), dtype=_gk.dtype)
                        kh, kw = _gk.shape
                        y0 = (target_h - kh) // 2
                        x0 = (target_w - kw) // 2
                        ky0 = max(0, -y0); kx0 = max(0, -x0)
                        cy0 = max(0, y0);  cx0 = max(0, x0)
                        ch = min(target_h - cy0, kh - ky0)
                        cw_ = min(target_w - cx0, kw - kx0)
                        canvas[cy0:cy0+ch, cx0:cx0+cw_] = _gk[ky0:ky0+ch, kx0:kx0+cw_]
                        _gk = canvas
                    ax_gt_top.imshow(_gk, cmap=cmap_name, interpolation='nearest')
                else:
                    ax_gt_top.text(0.5, 0.5, 'нет\nданных',
                                   ha='center', va='center',
                                   transform=ax_gt_top.transAxes, fontsize=16,
                                   color='grey')
                ax_gt_top.set_title('Истинное\nядро', fontsize=16)
                ax_gt_top.axis('off')
                for _spine in ax_gt_top.spines.values():
                    _spine.set_visible(True)
                    _spine.set_edgecolor('#00cc44')
                    _spine.set_linewidth(2.5)

                if has_restored:
                    ax_gt_bot = axes_bot[gt_col]
                    if orig_img is not None:
                        if orig_img.ndim == 2:
                            ax_gt_bot.imshow(orig_img, cmap='gray')
                        else:
                            ax_gt_bot.imshow(orig_img)
                    else:
                        ax_gt_bot.text(0.5, 0.5, 'нет\nданных',
                                       ha='center', va='center',
                                       transform=ax_gt_bot.transAxes, fontsize=16,
                                       color='grey')
                    ax_gt_bot.set_title('Истинное\nизображение', fontsize=16)
                    ax_gt_bot.axis('off')
                    for _spine in ax_gt_bot.spines.values():
                        _spine.set_visible(True)
                        _spine.set_edgecolor('#00cc44')
                        _spine.set_linewidth(2.5)

            fig.suptitle(f'{decode(alg_label)}: Ядра / восстановление — '
                         f'{decode(img_name)}', fontsize=22)
            plt.tight_layout()

            fname = f"kernel_evo6_{cmap_tag}_restored_{img_name}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
                fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=\textwidth]{figures/" + fname + r".pdf}" "\n"
                    r"\caption{Ядра (сверху) и восстановленные изображения (снизу) "
                    + r"алгоритма " + decode(alg_label) + r" на изображении "
                    + img_name.replace("_", r"\_")
                    + r" (до " + str(n_frames) + r" равномерных итераций, "
                    + cmap_tag + r").}" "\n"
                    r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  [v3] Эволюция ядра ({n_frames} панелей + восстановление): "
          f"{len(image_dirs)} изображений")


def plot_hyperparam_sensitivity_1d_v2(
    csv_path: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
):
    """
    PSNR (синяя ось слева) и SSIM (зелёная ось справа) на одном полотне.

    Файлы: sensitivity_v2_{vary}_fix_{fix}.{pdf,png}.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
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
    avg = df.groupby([p1, p2])[['psnr', 'ssim']].mean().reset_index()
    if avg.empty or avg['psnr'].isna().all():
        return
    best_row = avg.loc[avg['psnr'].idxmax()]
    best_p1 = best_row[p1]; best_p2 = best_row[p2]

    slices = [
        (p2, p1, best_p1, avg[avg[p1] == best_p1].sort_values(p2)),
        (p1, p2, best_p2, avg[avg[p2] == best_p2].sort_values(p1)),
    ]

    BLUE  = '#1F77B4'  # — PSNR
    GREEN = '#2CA02C'  # — SSIM

    for vary_param, fix_param, fix_val, df_slice in slices:
        if df_slice.empty:
            continue

        x = df_slice[vary_param].values
        fig, ax_p = plt.subplots(figsize=(8.5, 5))

        ax_p.plot(x, df_slice['psnr'].values, 'o-',
                  color=BLUE, linewidth=2, markersize=6, label='PSNR')
        ax_p.set_xlabel(vary_param, fontsize=13)
        ax_p.set_ylabel('PSNR, дБ', color=BLUE, fontsize=13)
        ax_p.tick_params(axis='y', labelcolor=BLUE)
        ax_p.set_xscale('log')
        ax_p.grid(True, alpha=0.3)

        ax_s = ax_p.twinx()
        ax_s.plot(x, df_slice['ssim'].values, 's-',
                  color=GREEN, linewidth=2, markersize=6, label='SSIM')
        ax_s.set_ylabel('SSIM', color=GREEN, fontsize=13)
        ax_s.tick_params(axis='y', labelcolor=GREEN)

        h1, l1 = ax_p.get_legend_handles_labels()
        h2, l2 = ax_s.get_legend_handles_labels()
        ax_p.legend(h1 + h2, l1 + l2, loc='best', fontsize=12)

        ax_p.set_title(
            f'{decode(alg_label)}: Чувствительность к {vary_param} '
            f'(при {fix_param}={fix_val:.4g})',
            fontsize=TITLE_FONTSIZE)
        plt.tight_layout()

        fname = f"sensitivity_v2_{vary_param}_fix_{fix_param}"
        if fig_dir:
            # fig.savefig(Path(fig_dir) / f"{fname}.pdf", bbox_inches='tight')
            fig.savefig(Path(fig_dir) / f"{fname}.png", dpi=200, bbox_inches='tight')
        plt.close(fig)

        if tex_dir:
            tex = (
                r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                r"\includegraphics[width=0.9\textwidth]{figures/"
                + fname + r".pdf}" "\n"
                r"\caption{Чувствительность PSNR и SSIM алгоритма "
                + decode(alg_label) + r" к параметру "
                + vary_param.replace("_", r"\_")
                + r" при фиксированном " + fix_param.replace("_", r"\_")
                + f"={fix_val:.4g}" + r". Левая ось --- PSNR (синяя), "
                r"правая --- SSIM (зелёная).}" "\n"
                r"\label{fig:" + _safe_label(fname) + r"}" "\n"
                r"\end{figure}"
            )
            save_tex(Path(tex_dir) / f"{fname}.tex", tex)

    print(f"  [v2] Чувствительность (две оси): {csv_path.name}")


def plot_hyperparam_heatmap_3d(
    csv_path: Path,
    alg_label: str,
    fig_dir: Optional[Path] = None,
    tex_dir: Optional[Path] = None,
    all_results_df: Optional["pd.DataFrame"] = None,
):
    """
    3D-визуализации тепловой карты гиперпараметров.

    Для каждой из доступных метрик (PSNR, SSIM, и ISNR) строятся:

    - 3D-бары с 4 ракурсами — heatmap3d_bars_{metric}_{p1}_{p2}_4angles
    - 3D-поверхность с 4 ракурсами — heatmap3d_surface_{metric}_{p1}_{p2}_4angles
    - Интерактивный HTML с поверхностью (plotly) —
      heatmap3d_surface_{metric}_{p1}_{p2}_interactive.html
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  [v2/3D] Файл не найден: {csv_path}")
        return
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    if 'psnr_blurred' not in df.columns and all_results_df is not None:
        _pmap = _compute_psnr_blurred_map(df, all_results_df)
        if _pmap:
            df = df.copy()
            df['psnr_blurred'] = df['image'].map(
                lambda s: _pmap.get(Path(str(s)).stem)
            )

    known_cols = {'image', 'psnr', 'ssim', 'time_sec', 'error_ratio',
                  'psnr_blurred', 'ssim_blurred', 'isnr'}
    param_cols = [c for c in df.columns if c not in known_cols]
    if len(param_cols) < 2:
        return
    p1, p2 = param_cols[0], param_cols[1]

    metrics = [
        ('psnr', 'PSNR, дБ', 'YlOrRd'),
        ('ssim', 'SSIM',      'YlGnBu'),
    ]
    if 'psnr_blurred' in df.columns and df['psnr_blurred'].notna().any():
        df = df.copy()
        df['isnr'] = df['psnr'] - df['psnr_blurred']
        metrics.append(('isnr', 'ISNR, дБ', 'RdYlGn'))

    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]

    for metric, mlabel, cmap_name in metrics:
        if metric not in df.columns or df[metric].isna().all():
            continue

        pivot = df.groupby([p1, p2])[metric].mean().reset_index()
        Z = pivot.pivot(index=p1, columns=p2, values=metric).sort_index()
        x_vals = Z.columns.values  # p2
        y_vals = Z.index.values    # p1
        Zv = Z.values

        nx = len(x_vals); ny = len(y_vals)
        cmap = cm.get_cmap(cmap_name)

        xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
        xpos = xi.ravel().astype(float)
        ypos = yi.ravel().astype(float)
        zpos = np.zeros_like(xpos)
        dx = dy = 0.7
        dz = np.nan_to_num(Zv.ravel(), nan=0.0)
        zmin, zmax = np.nanmin(Zv), np.nanmax(Zv)
        if zmax - zmin < 1e-12:
            norm = np.zeros_like(dz)
        else:
            norm = (dz - zmin) / (zmax - zmin)
        bar_colors = cmap(norm)

        for elev, azim in angles:
            fig = plt.figure(figsize=(14, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=bar_colors,
                     shade=True, edgecolor='grey', linewidth=0.2)
            ax.set_xticks(np.arange(nx) + dx / 2)
            ax.set_xticklabels([f"{v:.3g}" for v in x_vals],
                               fontsize=10, rotation=30)
            ax.set_yticks(np.arange(ny) + dy / 2)
            ax.set_yticklabels([f"{v:.3g}" for v in y_vals], fontsize=10)
            ax.set_xlabel(p2, fontsize=13, labelpad=10)
            ax.set_ylabel(p1, fontsize=13, labelpad=10)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(mlabel, fontsize=13, labelpad=20, rotation=0)
            ax.tick_params(axis='z', pad=6, labelsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{decode(alg_label)}: {mlabel}",
                         fontsize=TITLE_FONTSIZE)
            fig.subplots_adjust(left=0.05, right=0.65, top=0.92, bottom=0.08)

            fname_bars = f"heatmap3d_bars_{metric}_{p1}_{p2}_angle_{azim}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname_bars}.pdf")
                fig.savefig(Path(fig_dir) / f"{fname_bars}.png", dpi=180)
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{figures/"
                    + fname_bars + r".pdf}" "\n"
                    r"\caption{3D-бары " + mlabel + r" алгоритма "
                    + decode(alg_label) + r" по сетке гиперпараметров "
                    + p1.replace("_", r"\_") + r" и "
                    + p2.replace("_", r"\_") + r" (угол " + str(azim) + r"$^\circ$).}" "\n"
                    r"\label{fig:" + _safe_label(fname_bars) + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname_bars}.tex", tex)

        Xs, Ys = np.meshgrid(np.arange(nx), np.arange(ny))
        Zs = np.where(np.isnan(Zv), np.nanmin(Zv) if not np.isnan(np.nanmin(Zv)) else 0.0, Zv)

        for elev, azim in angles:
            fig = plt.figure(figsize=(14, 9))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(Xs, Ys, Zs, cmap=cmap_name,
                            linewidth=0.3, antialiased=True,
                            edgecolor='grey', alpha=0.95,
                            rcount=max(10, ny), ccount=max(10, nx))
            ax.set_xticks(np.arange(nx))
            ax.set_xticklabels([f"{v:.3g}" for v in x_vals],
                               fontsize=10, rotation=30)
            ax.set_yticks(np.arange(ny))
            ax.set_yticklabels([f"{v:.3g}" for v in y_vals], fontsize=10)
            ax.set_xlabel(p2, fontsize=13, labelpad=10)
            ax.set_ylabel(p1, fontsize=13, labelpad=10)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(mlabel, fontsize=13, labelpad=20, rotation=0)
            ax.tick_params(axis='z', pad=6, labelsize=10)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{decode(alg_label)}: {mlabel}",
                         fontsize=TITLE_FONTSIZE)
            fig.subplots_adjust(left=0.05, right=0.65, top=0.92, bottom=0.08)

            fname_surf = f"heatmap3d_surface_{metric}_{p1}_{p2}_angle_{azim}"
            if fig_dir:
                # fig.savefig(Path(fig_dir) / f"{fname_surf}.pdf")
                fig.savefig(Path(fig_dir) / f"{fname_surf}.png", dpi=180)
            plt.close(fig)

            if tex_dir:
                tex = (
                    r"\begin{figure}[htbp]" "\n" r"\centering" "\n"
                    r"\includegraphics[width=0.85\textwidth]{figures/"
                    + fname_surf + r".pdf}" "\n"
                    r"\caption{3D-поверхность " + mlabel + r" алгоритма "
                    + decode(alg_label) + r" по сетке гиперпараметров "
                    + p1.replace("_", r"\_") + r" и "
                    + p2.replace("_", r"\_") + r" (угол " + str(azim) + r"$^\circ$).}" "\n"
                    r"\label{fig:" + _safe_label(fname_surf) + r"}" "\n"
                    r"\end{figure}"
                )
                save_tex(Path(tex_dir) / f"{fname_surf}.tex", tex)

        try:
            import plotly.graph_objects as go  # type: ignore
            X3, Y3 = np.meshgrid(x_vals, y_vals)
            fig_html = go.Figure(data=[go.Surface(
                x=X3, y=Y3, z=Zv, colorscale=cmap_name,
                colorbar=dict(title=mlabel),
            )])
            fig_html.update_layout(
                title=f'{decode(alg_label)}: {mlabel} ({p1} × {p2})',
                scene=dict(xaxis_title=p2, yaxis_title=p1, zaxis_title=mlabel),
                width=1000, height=800,
            )
            if fig_dir:
                html_path = (Path(fig_dir)
                             / f"heatmap3d_surface_{metric}_{p1}_{p2}_interactive.html")
                fig_html.write_html(str(html_path), include_plotlyjs='cdn')
        except ImportError:
            print("  [v2/3D] plotly не установлен; HTML-версия пропущена.")
        except Exception as e:
            print(f"  [v2/3D] Ошибка создания HTML: {e}")

    print(f"  [v2/3D] Heatmap 3D ({len(metrics)} метрик): {csv_path.name}")
