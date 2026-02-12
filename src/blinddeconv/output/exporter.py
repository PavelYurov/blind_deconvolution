import os
import logging
import pandas as pd
from typing import Dict, Optional, List, Union
from jinja2 import Environment, FileSystemLoader, select_autoescape
from .tables import dataframe_to_latex

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
        pass

    def generate_report(self, report_name: str, content_order: List[Dict]):
        """
        Главный метод сборки.
        
        Args:
            report_name: имя файла (report.pdf)
            content_order: Список словарей, описывающих структуру.
            Пример:
            [
                {'type': 'text', 'content': 'Введение...'},
                {'type': 'include', 'file': 'table_summary'},
                {'type': 'include', 'file': 'visuals_best'},
            ]
        """
        sections_tex = ""
        
        for item in content_order:
            if item['type'] == 'section':
                sections_tex += f"\\section{{{item['title']}}}\n"
            elif item['type'] == 'subsection':
                sections_tex += f"\\subsection{{{item['title']}}}\n"
            elif item['type'] == 'text':
                sections_tex += f"{item['content']}\n\n"
            elif item['type'] == 'include':
                filename = item['file'].replace('.tex', '')
                sections_tex += f"\\input{{{filename}}}\n"
            elif item['type'] == 'raw':
                sections_tex += f"{item['content']}\n"
                
        context = {
            'title': "Report",
            'content': sections_tex
        }
        
        template = self.env.get_template("base_report.tex")
        rendered = template.render(**context)
        self._write_file(f"{report_name}.tex", rendered)

    def _write_file(self, filename, content):
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)

    def save_visuals(self, context: Dict, filename: str):
        """
        Рендерит блок визуального сравнения через шаблон visuals.tex.
        
        Args:
            context: Словарь данных (результат prepare_visual_comparison_data)
            filename: Имя выходного файла (без расширения)
        """
        try:
            template = self.env.get_template("visuals.tex")
            rendered_tex = template.render(**context)
            
            output_path = os.path.join(self.output_dir, f"{filename}.tex")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(rendered_tex)
            logger.info(f"Visuals saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visuals {filename}: {e}")
            raise