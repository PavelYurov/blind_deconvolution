import pandas as pd
import numpy as np
import re
import os
from typing import Optional, List, Dict

def escape_latex(val):
    """
    Экранирует спецсимволы (_, %, &).
    """
    if not isinstance(val, str):
        return val
    if val.startswith('$') and val.endswith('$'):
        return val
    
    val = val.replace('\\', r'\textbackslash')
    val = val.replace('_', r'\_')
    val = val.replace('%', r'\%')
    val = val.replace('&', r'\&')
    val = val.replace('#', r'\#')
    return val

def dataframe_to_latex(
    df: pd.DataFrame, 
    caption: str = "", 
    label: str = "", 
    column_format: Optional[str] = None
) -> str:
    export_df = df.copy()
    if export_df.index.name:
        export_df.reset_index(inplace=True)
    export_df.columns = [escape_latex(c) for c in export_df.columns]
    
    for col in export_df.select_dtypes(include=['object']):
        export_df[col] = export_df[col].apply(escape_latex)

    if column_format is None:
        col_formats = []
        for col in export_df.columns:
            col_formats.append('c' if 'Algorithm' not in col else 'l')
        column_format = "".join(col_formats)

    latex_body = export_df.to_latex(
        index=False,
        escape=False,
        column_format=column_format,
        na_rep='-',
        caption=caption,
        label=label,
        position="h!"
    )

    latex_body = latex_body.replace(
        "\\begin{table}[h!]",
        "\\begin{table}[h!]\n\\centering"
    )
    
    if '\\toprule' not in latex_body:
        latex_body = latex_body.replace('\\hline', '')
    
    return latex_body

def clean_filter_names(df: pd.DataFrame, col_name: str = 'filter') -> pd.DataFrame:
    if col_name not in df.columns: return df
    def cleaner(text):
        if not isinstance(text, str): return text
        name = os.path.basename(text).split('.')[0]
        return name.replace('_', ' ').title()
    
    df_out = df.copy()
    df_out[col_name] = df_out[col_name].apply(cleaner)
    return df_out

def prepare_params_table(json_data: List[Dict]) -> pd.DataFrame:
    rows = []
    for entry in json_data:
        alg = entry.get('algorithm', 'Unknown')
        params = entry.get('algorithm parametrs', [])
        if isinstance(params, list):
             p_dict = {item[0]: item[1] for item in params if len(item) == 2}
        else: p_dict = params
        p_dict['Algorithm'] = alg
        rows.append(p_dict)
    
    if not rows: return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    cols = ['Algorithm'] + [c for c in df.columns if c != 'Algorithm']
    df = df[cols]

    df = df.set_index('Algorithm').T
    df.index.name = "Parameter"
    return df

def _highlight_max(s):
    """Вспомогательная функция: делает максимум жирным для LaTeX."""
    is_max = s == s.max()
    return [f"\\textbf{{{v:.2f}}}" if is_m else f"{v:.2f}" for v, is_m in zip(s, is_max)]

def prepare_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сводная таблица: Mean +/- Std для PSNR, SSIM, Time.
    """
    metrics = [m for m in ['psnr', 'ssim', 'time'] if m in df.columns]
    
    if not metrics:
        return pd.DataFrame()

    grouped = df.groupby('algorithm')[metrics].agg(['mean', 'std'])
    final_df = pd.DataFrame(index=grouped.index)
    
    for metric in metrics:
        mean = grouped[metric]['mean']
        std = grouped[metric]['std']
        
        col_data = []
        for m, s in zip(mean, std):
            if pd.isna(s) or s == 0:
                col_data.append(f"${m:.3f}$")
            else:
                col_data.append(rf"${m:.2f} \pm {s:.2f}$")
        
        final_df[metric.upper()] = col_data

    final_df.index.name = "Algorithm"
    return final_df

def prepare_comparison_pivot(df: pd.DataFrame, metric: str = 'psnr', highlight_best: bool = True) -> pd.DataFrame:
    """
    Кросс-таблица. Строки - фильтры/картинки, Столбцы - Алгоритмы.
    metric: 'psnr' или 'ssim'
    """
    if metric not in df.columns:
        return pd.DataFrame()
    if df['image'].nunique() > 1:
        df['row_id'] = df['image'] + " (" + df['filter'] + ")"
    else:
        df['row_id'] = df['filter']

    pivot = df.pivot_table(index='row_id', columns='algorithm', values=metric, aggfunc='mean')
    
    pivot.index.name = "Test Case"
    pivot.columns.name = None
    
    if highlight_best and len(pivot.columns) > 1:
        pivot = pivot.apply(_highlight_max, axis=1)
    else:
        pivot = pivot.round(3)

    return pivot

def prepare_summary_improvement_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица 1: Сводная статистика по ПРИРОСТУ (Improvement).
    Колонки: Algorithm, Mean PSNR Improv, Mean Time (если есть).
    """
    metrics = []
    if 'psnr_improvement' in df.columns: metrics.append('psnr_improvement')
    if 'ssim_improvement' in df.columns: metrics.append('ssim_improvement')
    if 'time' in df.columns: metrics.append('time')
    
    if not metrics:
        return pd.DataFrame()

    grouped = df.groupby('algorithm')[metrics].agg(['mean', 'std'])
    final_df = pd.DataFrame(index=grouped.index)
    
    rename_map = {
        'psnr_improvement': 'PSNR Gain (dB)',
        'ssim_improvement': 'SSIM Gain',
        'time': 'Time (s)'
    }

    for metric in metrics:
        col_name = rename_map.get(metric, metric)
        mean = grouped[metric]['mean']
        std = grouped[metric]['std']
        
        col_data = []
        for m, s in zip(mean, std):
            if pd.isna(s) or s == 0:
                col_data.append(f"${m:.2f}$")
            else:
                col_data.append(rf"${m:.2f} \pm {s:.2f}$")
        
        final_df[col_name] = col_data

    final_df.index.name = "Algorithm"
    return final_df

def prepare_detailed_comparison(df: pd.DataFrame, metric: str = 'psnr') -> pd.DataFrame:
    """
    Таблица 2: Детальное сравнение.
    Строки: Test Case (Картинка + Фильтр)
    Столбец 1: Blurred Metric (База)
    Остальные столбцы: Алгоритмы
    """
    pivot = df.pivot_table(index='test_case_id', columns='algorithm', values=metric, aggfunc='mean')
    
    blurred_col_name = f"blurred_{metric}"
    if blurred_col_name in df.columns:
        blurred_vals = df.groupby('test_case_id')[blurred_col_name].first()
        pivot.insert(0, 'Blurred', blurred_vals)
    
    pivot = pivot.round(2)
    pivot.index.name = "Test Case"
    pivot.columns.name = None
    
    return pivot

def prepare_best_worst_table(df: pd.DataFrame, metric: str = 'psnr') -> pd.DataFrame:
    """
    Таблица экстремумов (Лучший/Худший кейс) для конкретной метрики.
    
    Args:
        df: DataFrame с данными.
        metric: 'psnr' или 'ssim'.
        
    Returns:
        DataFrame с колонками [Best {Metric} Gain, Worst {Metric} Gain]
    """
    col_name = f"{metric}_improvement"
    
    if col_name not in df.columns:
        return pd.DataFrame()

    rows = []
    
    fmt = ".2f" if metric == 'psnr' else ".3f"
    label = metric.upper()

    for algo_name, group in df.groupby('algorithm'):
        row_data = {'Algorithm': algo_name}
        
        best_idx = group[col_name].idxmax()
        best_row = group.loc[best_idx]
        best_val = best_row[col_name]
        row_data[f'Best {label} Gain'] = f"{best_row['filter']} (+{best_val:{fmt}})"
        
        worst_idx = group[col_name].idxmin()
        worst_row = group.loc[worst_idx]
        worst_val = worst_row[col_name]
        row_data[f'Worst {label} Gain'] = f"{worst_row['filter']} ({worst_val:{fmt}})"

        rows.append(row_data)
    
    res_df = pd.DataFrame(rows)
    
    if not res_df.empty and 'Algorithm' in res_df.columns:
        res_df.set_index('Algorithm', inplace=True)
        
    return res_df