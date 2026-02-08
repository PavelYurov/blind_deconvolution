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

def prepare_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ['psnr', 'ssim', 'time']
    available = [m for m in metrics if m in df.columns]
    
    grouped = df.groupby('algorithm')[available].agg(['mean', 'std'])
    
    final_df = pd.DataFrame(index=grouped.index)
    
    for metric in available:
        mean_col = grouped[metric]['mean']
        std_col = grouped[metric]['std']
        
        formatted_col = []
        for m, s in zip(mean_col, std_col):
            if pd.isna(m):
                formatted_col.append("-")
            elif pd.isna(s) or s == 0:
                formatted_col.append(f"${m:.2f}$")
            else:
                formatted_col.append(rf"${m:.2f} \pm {s:.2f}$")
        
        final_df[metric.upper()] = formatted_col
    
    final_df.index.name = "Algorithm"
    return final_df

def prepare_comparison_pivot(df: pd.DataFrame, metric: str = 'psnr') -> pd.DataFrame:
    df_clean = clean_filter_names(df, 'filter')
    pivot = df_clean.pivot_table(index='filter', columns='algorithm', values=metric)
    pivot = pivot.round(3)
    
    pivot.columns.name = None
    pivot.index.name = "Filter Type"
    return pivot

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