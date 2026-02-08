import pandas as pd
from typing import List, Dict, Any

def prepare_boxplot_data(df: pd.DataFrame, value_col: str, group_col: str) -> Dict[str, Any]:
    """
    Рассчитывает статистику для boxplot (min, q1, median, q3, max) для каждой группы.
    """
    groups = df.groupby(group_col)[value_col]
    plots_data = []
    labels = []
    
    for i, (name, group) in enumerate(groups):
        desc = group.describe()
        plots_data.append({
            'median': f"{desc['50%']:.4f}",
            'q1': f"{desc['25%']:.4f}",
            'q3': f"{desc['75%']:.4f}",
            'min': f"{desc['min']:.4f}",
            'max': f"{desc['max']:.4f}",
            'name': str(name)
        })
        labels.append(str(name).replace('_', r'\_'))

    return {
        'plots': plots_data,
        'xtick_indices': ",".join([str(i+1) for i in range(len(labels))]),
        'xtick_labels': ",".join(labels)
    }

def prepare_scatter_data(df: pd.DataFrame, x_col: str, y_col: str, group_col: str) -> List[Dict]:
    """
    Подготавливает координаты (x, y) для scatter plot, группируя по алгоритму.
    """
    groups = []
    for name, group in df.groupby(group_col):
        coords_str = ""
        for _, row in group.iterrows():
            # Формат: (x, y)
            coords_str += f"({row[x_col]:.4f}, {row[y_col]:.4f}) "
        
        groups.append({
            'name': str(name).replace('_', r'\_'),
            'coords': coords_str
        })
    return groups