import os
import logging
import pandas as pd
from typing import Dict, Optional, List, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .tables import dataframe_to_latex
from .plotting import prepare_boxplot_data, prepare_scatter_data

logger = logging.getLogger(__name__)

class LatexExporter:
    def __init__(self, output_dir: str, template_dir: Optional[str] = None):
        """
        Инициализация экспортера LaTeX.

        Args:
            output_dir: Путь, куда будут сохраняться .tex файлы и графики.
            template_dir: Путь к пользовательским шаблонам (если есть).
                          По умолчанию использует встроенные шаблоны пакета.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if template_dir is None:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['tex']),
            block_start_string='\\BLOCK{',
            block_end_string='}',
            variable_start_string='\\VAR{',
            variable_end_string='}',
            comment_start_string='\\#{',
            comment_end_string='}',
            line_statement_prefix='%%',
            line_comment_prefix='%#',
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def save_table(self, df: pd.DataFrame, filename: str, caption: str = "", label: str = ""):
        """
        Сохраняет pandas DataFrame как отдельный .tex файл с таблицей.
        
        Args:
            df: Данные.
            filename: Имя файла. Без расширения или с ним.
            caption: Подпись таблицы.
            label: Метка для ссылок.
        """
        if filename.endswith('.tex'):
            filename = filename[:-4]
            
        tex_content = dataframe_to_latex(df, caption=caption, label=label)
        
        output_path = os.path.join(self.output_dir, f"{filename}.tex")
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(tex_content)
            logger.info(f"Table saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save table {filename}: {e}")
            raise

    def save_plot(self, df: pd.DataFrame, plot_type: str, filename: str, **kwargs):
        """
        Генерирует график pgfplots.
        
        Args:
            df: DataFrame с данными.
            plot_type: 'boxplot' или 'scatter'.
            filename: Имя выходного файла (без расширения).
            **kwargs: Параметры графика (x_col, y_col, title, ylabel и т.д.)
        """
        context = {}
        template_name = ""

        if plot_type == 'boxplot':
            # Ожидаем: value_col (что меряем), group_col (кто меряет)
            value_col = kwargs.get('value_col', 'psnr')
            group_col = kwargs.get('group_col', 'algorithm')
            
            data = prepare_boxplot_data(df, value_col, group_col)
            context = {
                'title': kwargs.get('title', f'Distribution of {value_col}'),
                'ylabel': kwargs.get('ylabel', value_col.upper()),
                'plots': data['plots'],
                'xtick_indices': data['xtick_indices'],
                'xtick_labels': data['xtick_labels']
            }
            template_name = "plots/boxplot.tex"

        elif plot_type == 'scatter':
            x_col = kwargs.get('x_col', 'time')
            y_col = kwargs.get('y_col', 'psnr')
            group_col = kwargs.get('group_col', 'algorithm')
            
            groups = prepare_scatter_data(df, x_col, y_col, group_col)
            context = {
                'title': kwargs.get('title', f'{y_col} vs {x_col}'),
                'xlabel': kwargs.get('xlabel', x_col),
                'ylabel': kwargs.get('ylabel', y_col),
                'groups': groups
            }
            template_name = "plots/scatterplot.tex"
            
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        try:
            template = self.env.get_template(template_name)
            rendered_tex = template.render(**context)
            
            output_path = os.path.join(self.output_dir, f"{filename}.tex")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_tex)
            logger.info(f"Plot saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate plot {filename}: {e}")
            raise

    def generate_report(self, report_name: str, template_name: str, context: Dict):
        """
        Генерирует полный отчет на основе шаблона.
        
        Args:
            report_name: Имя выходного файла (без расширения).
            template_name: Имя файла шаблона (например, 'article.tex').
            context: Словарь данных для подстановки в шаблон.
        """
        try:
            template = self.env.get_template(template_name)
            rendered_tex = template.render(**context)
            
            output_path = os.path.join(self.output_dir, f"{report_name}.tex")
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_tex)
                
            logger.info(f"Report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise