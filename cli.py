"""
Интерфейс командной строки (CLI) на основе Click для фреймворка Blind Image Deconvolution.

Предоставляет команды для обработки отдельных изображений, выполнения пайплайнов 
по конфигурационным файлам, управления конфигурациями и интерактивной настройки экспериментов.

Использование::

    python cli.py process --input image.jpg --algorithm vabid
    python cli.py run --config experiment.yaml
    python cli.py generate-config --template medical --output my_config.yaml
    python cli.py view-config experiment.yaml
    python cli.py interactive
    python cli.py list-algorithms
    python cli.py list-filters

Автодополнение в shell::

    eval "$(_CLI_COMPLETE=bash_source python cli.py)"   # bash
    eval "$(_CLI_COMPLETE=zsh_source python cli.py)"    # zsh
    _CLI_COMPLETE=fish_source python cli.py | source    # fish

Автор: Беззаборов А.А.
"""

import sys
import os
import json
import logging
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Path configuration
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:
    import click
except ImportError:
    print(
        "Ошибка: библиотека 'click' не установлена.\n"
        "Установите: pip install click\n"
        "Или: pip install -e .[cli]",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Imports from run.py
from run import (
    ALGORITHM_REGISTRY,
    FILTER_REGISTRY,
    PSF_REGISTRY,
    load_config,
    validate_config,
    run_pipeline,
    create_algorithm,
    setup_logging,
)


logger = logging.getLogger("blinddeconv.cli")


# Configuration templates

CONFIG_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "basic": {
        "experiment": {
            "name": "Basic Deconvolution",
            "description": "Базовый пример восстановления изображений",
        },
        "input": {
            "images_folder": "images/original",
            "blurred_folder": "images/distorted",
            "color": False,
            "load_mode": "bind",
            "bindings": [
                {
                    "original": "images/original/example.png",
                    "blurred": "images/distorted/example_blurred.png",
                    "filter_description": "unknown",
                }
            ],
        },
        "output": {
            "restored_folder": "results/restored",
            "data_folder": "results/data",
            "kernel_folder": "results/kernels",
        },
        "processing": {"mode": "process", "metadata": True,
                        "unique_paths": True},
        "algorithms": [
            {"name": "vabid", "params": {"max_iter": 100, "kernel_size": 21}}
        ],
        "report": {"generate": False, "format": "latex"},
    },
    "medical": {
        "experiment": {
            "name": "Medical Imaging Deconvolution",
            "description": "Восстановление медицинских изображений",
        },
        "input": {
            "images_folder": "images/medical",
            "color": False,
            "load_mode": "all",
        },
        "output": {
            "restored_folder": "results/medical/restored",
            "data_folder": "results/medical/data",
            "kernel_folder": "results/medical/kernels",
        },
        "processing": {"mode": "full_process", "metadata": True,
                        "unique_paths": True},
        "algorithms": [
            {"name": "vabid", "params": {"max_iter": 200, "kernel_size": 15}},
            {"name": "vbbid_tv",
             "params": {"max_iter": 150, "kernel_size": 15}},
        ],
        "filters": [
            {
                "chain": [
                    {"type": "defocus_blur",
                     "params": {"psf": "gaussian", "param": 2.0}},
                    {"type": "gaussian_noise", "params": {"param": 3.0}},
                ]
            }
        ],
        "report": {"generate": True, "format": "latex",
                    "output_path": "results/medical/report.tex"},
    },
    "satellite": {
        "experiment": {
            "name": "Satellite Image Restoration",
            "description": "Восстановление спутниковых снимков",
        },
        "input": {
            "images_folder": "images/satellite",
            "color": True,
            "load_mode": "all",
        },
        "output": {
            "restored_folder": "results/satellite/restored",
            "data_folder": "results/satellite/data",
            "kernel_folder": "results/satellite/kernels",
        },
        "processing": {"mode": "full_process", "metadata": True,
                        "unique_paths": True},
        "algorithms": [
            {"name": "vabid",
             "params": {"max_iter": 300, "kernel_size": 25}},
        ],
        "filters": [
            {
                "chain": [
                    {"type": "defocus_blur",
                     "params": {"psf": "gaussian", "param": 4.0,
                                "kernel_size": 25}},
                    {"type": "gaussian_noise", "params": {"param": 8.0}},
                ]
            },
            {
                "chain": [
                    {"type": "motion_blur",
                     "params": {"psf": "uniform", "param": 3.0,
                                "angle": 30, "kernel_length": 15}},
                    {"type": "gaussian_noise", "params": {"param": 5.0}},
                ]
            },
        ],
        "report": {"generate": True, "format": "latex",
                    "output_path": "results/satellite/report.tex"},
    },
    "empty": {
        "experiment": {"name": "", "description": ""},
        "input": {
            "images_folder": "",
            "color": False,
            "load_mode": "bind",
            "bindings": [],
        },
        "output": {
            "restored_folder": "results/restored",
            "data_folder": "results/data",
            "kernel_folder": "results/kernels",
        },
        "processing": {"mode": "process", "metadata": True,
                        "unique_paths": True},
        "algorithms": [],
        "filters": [],
        "report": {"generate": False, "format": "latex"},
    },
}


# Formatting utilities

def _format_table(headers: List[str], rows: List[List[str]],
                  col_widths: Optional[List[int]] = None) -> str:
    """
    Форматирование таблицы в текстовом виде.

    Параметры
    ---------
    headers : List[str]
        Заголовки столбцов.
    rows : List[List[str]]
        Строки таблицы.
    col_widths : Optional[List[int]]
        Ширина столбцов (авто, если None).

    Возвращает
    ----------
    str
        Отформатированная таблица.
    """
    if col_widths is None:
        col_widths = []
        for i in range(len(headers)):
            max_w = len(headers[i])
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(str(row[i])))
            col_widths.append(min(max_w, 50))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt_row(values: List[str]) -> str:
        cells = []
        for i, v in enumerate(values):
            w = col_widths[i] if i < len(col_widths) else 20
            cells.append(f" {str(v):<{w}} ")
        return "|" + "|".join(cells) + "|"

    lines = [separator, fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(separator)

    return "\n".join(lines)


def _format_table_latex(headers: List[str],
                        rows: List[List[str]]) -> str:
    """
    Форматирование таблицы в LaTeX.

    Параметры
    ---------
    headers : List[str]
        Заголовки столбцов.
    rows : List[List[str]]
        Строки таблицы.

    Возвращает
    ----------
    str
        LaTeX-таблица.
    """
    col_spec = "l" * len(headers)
    header_row = " & ".join(f"\\textbf{{{h}}}" for h in headers)
    data_rows = "\n".join(
        "    " + " & ".join(str(v) for v in row) + " \\\\"
        for row in rows
    )

    return (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"    \\toprule\n"
        f"    {header_row} \\\\\n"
        f"    \\midrule\n"
        f"{data_rows}\n"
        f"    \\bottomrule\n"
        f"\\end{{tabular}}"
    )


# Main command group

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(version="1.0.0", prog_name="blinddeconv-cli")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Подробный вывод (уровень DEBUG).")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    Интерфейс командной строки для фреймворка слепой деконволюции.

    Используйте --help после любой команды для подробной справки.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose=verbose)


# Команда: process

@cli.command()
@click.option("--input", "-i", "input_path", required=True,
              type=click.Path(exists=True),
              help="Путь к входному изображению.")
@click.option("--algorithm", "-a", "algo_name", required=True,
              type=click.Choice(sorted(ALGORITHM_REGISTRY.keys()),
                                case_sensitive=False),
              help="Имя алгоритма восстановления.")
@click.option("--blurred", "-b", "blurred_path", default=None,
              type=click.Path(exists=True),
              help="Путь к смазанному изображению (если отличается от input).")
@click.option("--kernel", "-k", "kernel_path", default=None,
              type=click.Path(exists=True),
              help="Путь к ядру размытия (.npy).")
@click.option("--output-dir", "-o", default="results",
              help="Директория для результатов.")
@click.option("--color/--grayscale", default=False,
              help="Цветной или ч/б режим.")
@click.option("--params", "-p", default=None,
              help='Параметры алгоритма в формате JSON, например: '
                   '\'{"max_iter": 100, "kernel_size": 21}\'')
@click.pass_context
def process(ctx: click.Context, input_path: str, algo_name: str,
            blurred_path: Optional[str], kernel_path: Optional[str],
            output_dir: str, color: bool, params: Optional[str]) -> None:
    """
    Быстрая обработка одного изображения.

    Примеры:

    \b
      python cli.py process --input image.jpg --algorithm vabid
      python cli.py process -i img.png -a richardson_lucy -p '{"max_iter":50}'
      python cli.py process -i orig.png -b blurred.png -a vabid -o results/
    """
    from blinddeconv.processing.core import Processing

    # Парсинг параметров алгоритма
    algo_params = {}
    if params:
        try:
            algo_params = json.loads(params)
        except json.JSONDecodeError as exc:
            click.echo(f"Ошибка: некорректный JSON в --params: {exc}",
                       err=True)
            sys.exit(1)

    click.echo(f"Алгоритм: {algo_name}")
    click.echo(f"Входное изображение: {input_path}")
    if algo_params:
        click.echo(f"Параметры: {algo_params}")

    # Создание алгоритма
    try:
        algo = create_algorithm({"name": algo_name, "params": algo_params})
    except (ValueError, ImportError) as exc:
        click.echo(f"Ошибка создания алгоритма: {exc}", err=True)
        sys.exit(1)

    # Настройка Processing
    images_folder = str(Path(input_path).parent)
    actual_blurred = blurred_path or input_path

    p = Processing(
        images_folder=images_folder,
        blurred_folder=str(Path(actual_blurred).parent),
        restored_folder=str(Path(output_dir) / "restored"),
        data_path=str(Path(output_dir) / "data"),
        color=color,
        kernel_dir=str(Path(output_dir) / "kernels"),
    )

    # Связывание изображений
    if blurred_path:
        p.bind(
            original_image_path=input_path,
            blurred_image_path=blurred_path,
            original_kernel_path=kernel_path,
            filter_description="cli_input",
            color=color,
        )
    else:
        # Если нет отдельного blurred — используем input как blurred
        p.bind(
            original_image_path=input_path,
            blurred_image_path=input_path,
            original_kernel_path=kernel_path,
            filter_description="cli_input",
            color=color,
        )

    # Обработка
    click.echo("Запуск обработки...")
    start = datetime.datetime.now()
    p.process(algorithm_processor=algo, metadata=True)
    elapsed = (datetime.datetime.now() - start).total_seconds()

    click.echo(f"Обработка завершена за {elapsed:.1f} сек.")
    click.echo(f"Результаты: {output_dir}/")


# Команда: run

@cli.command()
@click.option("--config", "-c", required=True,
              type=click.Path(exists=True),
              help="Путь к файлу конфигурации (.yaml/.json).")
@click.option("--output-dir", "-o", default=None,
              help="Переопределение директории результатов.")
@click.option("--generate-report", is_flag=True, default=False,
              help="Генерировать LaTeX-отчёт.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Проверка без выполнения обработки.")
@click.option("--validate-only", is_flag=True, default=False,
              help="Только валидация конфигурации.")
@click.pass_context
def run(ctx: click.Context, config: str, output_dir: Optional[str],
        generate_report: bool, dry_run: bool, validate_only: bool) -> None:
    """
    Запуск пайплайна по конфигурационному файлу.

    Альтернатива run.py с идентичной функциональностью.

    Примеры:

    \b
      python cli.py run --config configs/experiment.yaml
      python cli.py run -c configs/experiment.yaml --dry-run
      python cli.py run -c configs/experiment.yaml --generate-report
      python cli.py run -c configs/experiment.yaml --validate-only
    """
    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        click.echo(f"Ошибка загрузки конфигурации: {exc}", err=True)
        sys.exit(1)

    errors = validate_config(cfg)
    if errors:
        click.echo("Ошибки валидации конфигурации:", err=True)
        for err in errors:
            click.echo(f"  • {err}", err=True)
        sys.exit(1)

    click.echo("Конфигурация валидна.")

    if validate_only:
        click.echo("Режим --validate-only: проверка пройдена.")
        return

    try:
        result = run_pipeline(
            config=cfg,
            output_dir=output_dir,
            generate_report=generate_report,
            dry_run=dry_run,
        )

        if result["status"] == "completed":
            click.echo(
                f"Готово! Обработано: {result['images_processed']} изобр., "
                f"время: {result['duration_seconds']:.1f} сек."
            )
        elif result["status"] == "dry_run":
            click.echo("Dry-run завершён.")

    except Exception as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)


# Команда: generate-config

@cli.command("generate-config")
@click.option("--template", "-t",
              type=click.Choice(sorted(CONFIG_TEMPLATES.keys()),
                                case_sensitive=False),
              default="basic",
              help="Шаблон конфигурации.")
@click.option("--output", "-o", default=None,
              type=click.Path(),
              help="Путь для сохранения конфигурации.")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["yaml", "json"]),
              default="yaml",
              help="Формат выходного файла.")
def generate_config(template: str, output: Optional[str], fmt: str) -> None:
    """
    Генерация файла конфигурации из шаблона.

    Доступные шаблоны: basic, medical, satellite, empty.

    Примеры:

    \b
      python cli.py generate-config --template medical --output my_config.yaml
      python cli.py generate-config -t satellite -f json -o config.json
      python cli.py generate-config -t empty
    """
    config_data = CONFIG_TEMPLATES.get(template, CONFIG_TEMPLATES["basic"])

    if output is None:
        output = f"config_{template}.{fmt}"

    output_path = Path(output)

    if output_path.exists():
        if not click.confirm(f"Файл '{output}' уже существует. Перезаписать?"):
            click.echo("Отменено.")
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if fmt == "yaml":
            if not HAS_YAML:
                click.echo(
                    "Ошибка: PyYAML не установлен. "
                    "Используйте --format json или установите: pip install pyyaml",
                    err=True,
                )
                sys.exit(1)
            yaml.dump(config_data, f, allow_unicode=True,
                      default_flow_style=False, sort_keys=False)
        else:
            json.dump(config_data, f, ensure_ascii=False, indent=4)

    click.echo(f"Конфигурация сохранена: {output_path}")
    click.echo(f"Шаблон: {template}, формат: {fmt}")


# Команда: view-config

@cli.command("view-config")
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--format", "-f", "fmt",
              type=click.Choice(["table", "latex", "json", "yaml"]),
              default="table",
              help="Формат отображения.")
def view_config(config_path: str, fmt: str) -> None:
    """
    Просмотр конфигурации в структурированном виде.

    Примеры:

    \b
      python cli.py view-config configs/experiment.yaml
      python cli.py view-config configs/experiment.yaml --format latex
      python cli.py view-config configs/experiment.yaml --format json
    """
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        click.echo(f"Ошибка: {exc}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps(config, ensure_ascii=False, indent=4))
        return

    if fmt == "yaml":
        if not HAS_YAML:
            click.echo("Ошибка: PyYAML не установлен.", err=True)
            sys.exit(1)
        click.echo(yaml.dump(config, allow_unicode=True,
                             default_flow_style=False, sort_keys=False))
        return

    # Табличный / LaTeX вывод
    exp = config.get("experiment", {})
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    processing_cfg = config.get("processing", {})

    click.echo()
    click.echo(f"  {exp.get('name', 'Без имени')}")
    if exp.get("description"):
        click.echo(f"  {exp['description']}")
    click.echo()

    input_headers = ["Параметр", "Значение"]
    input_rows = [
        ["Папка изображений", input_cfg.get("images_folder", "---")],
        ["Режим загрузки", input_cfg.get("load_mode", "all")],
        ["Цветовой режим",
         "Цветной" if input_cfg.get("color", False) else "Ч/Б"],
    ]
    bindings = input_cfg.get("bindings", [])
    if bindings:
        input_rows.append(["Связей", str(len(bindings))])

    click.echo("  Входные данные:")
    if fmt == "latex":
        click.echo(_format_table_latex(input_headers, input_rows))
    else:
        click.echo(_format_table(input_headers, input_rows))

    # Параметры выхода
    output_rows = [
        ["Восстановленные", output_cfg.get("restored_folder", "restored")],
        ["Данные", output_cfg.get("data_folder", "data")],
        ["Ядра", output_cfg.get("kernel_folder", "kernels")],
    ]
    click.echo("\n  Выходные директории:")
    if fmt == "latex":
        click.echo(_format_table_latex(["Параметр", "Путь"], output_rows))
    else:
        click.echo(_format_table(["Параметр", "Путь"], output_rows))

    click.echo("\n  Обработка:")
    proc_rows = [
        ["Режим", processing_cfg.get("mode", "process")],
        ["Метаданные",
         "Да" if processing_cfg.get("metadata", True) else "Нет"],
        ["Уникальные пути",
         "Да" if processing_cfg.get("unique_paths", True) else "Нет"],
    ]
    if fmt == "latex":
        click.echo(_format_table_latex(["Параметр", "Значение"], proc_rows))
    else:
        click.echo(_format_table(["Параметр", "Значение"], proc_rows))

    # Алгоритмы
    algo_headers = ["Имя", "Категория", "Параметры"]
    algo_rows = []
    for algo in config.get("algorithms", []):
        name = algo.get("name", "?")
        entry = ALGORITHM_REGISTRY.get(name, {})
        cat = entry.get("category", "custom")
        params_str = ", ".join(
            f"{k}={v}" for k, v in algo.get("params", {}).items()
        )
        algo_rows.append([name, cat, params_str or "---"])

    click.echo("\n  Алгоритмы:")
    if fmt == "latex":
        click.echo(_format_table_latex(algo_headers, algo_rows))
    else:
        click.echo(_format_table(algo_headers, algo_rows))

    # Фильтры
    filters_cfg = config.get("filters", [])
    if filters_cfg:
        click.echo("\n  Цепочки фильтров:")
        for i, chain_cfg in enumerate(filters_cfg):
            chain_items = chain_cfg.get("chain", [])
            chain_desc = " → ".join(
                f"{f.get('type', '?')}"
                f"({', '.join(f'{k}={v}' for k, v in f.get('params', {}).items())})"
                for f in chain_items
            )
            click.echo(f"    [{i + 1}] {chain_desc}")

    # Валидация
    errors = validate_config(config)
    if errors:
        click.echo(f"\n  ОШИБКИ ВАЛИДАЦИИ ({len(errors)}):")
        for err in errors:
            click.echo(f"    • {err}")
    else:
        click.echo("\n  Конфигурация валидна.")

    click.echo()


# Команда: interactive

@cli.command()
@click.pass_context
def interactive(ctx: click.Context) -> None:
    """
    Интерактивный режим создания и запуска эксперимента.

    Пошагово задаёт вопросы для настройки и запуска пайплайна.
    Подходит для новичков.
    """
    click.echo("Интерактивный режим настройки эксперимента\n")

    # Эксперимент
    exp_name = click.prompt(
        "Название эксперимента",
        default="My Experiment",
    )
    exp_desc = click.prompt(
        "Описание (необязательно)",
        default="",
    )

    # Входные данные
    click.echo("\n--- Входные данные ---")
    images_folder = click.prompt(
        "Папка с изображениями",
        default="images/original",
    )
    color = click.confirm("Цветные изображения?", default=False)

    load_mode = click.prompt(
        "Режим загрузки (all / bind / bind_state)",
        type=click.Choice(["all", "bind", "bind_state"],
                          case_sensitive=False),
        default="all",
    )

    bindings: List[Dict[str, str]] = []
    bind_state_path = ""

    if load_mode == "bind":
        click.echo("Добавьте связи оригинал → смазанное:")
        while True:
            original = click.prompt("  Путь к оригиналу")
            blurred = click.prompt("  Путь к смазанному")
            kernel = click.prompt(
                "  Путь к ядру (Enter — пропустить)", default=""
            )
            desc = click.prompt(
                "  Описание фильтра", default="unknown"
            )
            binding: Dict[str, str] = {
                "original": original,
                "blurred": blurred,
                "filter_description": desc,
            }
            if kernel:
                binding["kernel"] = kernel
            bindings.append(binding)

            if not click.confirm("Добавить ещё связь?", default=False):
                break

    elif load_mode == "bind_state":
        bind_state_path = click.prompt("Путь к JSON-файлу связей")

    #  Режим обработки 
    click.echo("\n--- Обработка ---")
    mode = click.prompt(
        "Режим обработки (process / full_process)",
        type=click.Choice(["process", "full_process"],
                          case_sensitive=False),
        default="full_process" if load_mode == "all" else "process",
    )

    # Алгоритмы 
    click.echo("\n--- Алгоритмы ---")
    click.echo("Доступные алгоритмы:")
    for name, entry in sorted(ALGORITHM_REGISTRY.items()):
        click.echo(f"  • {name:20s} — {entry['description']}")

    algorithms: List[Dict[str, Any]] = []
    while True:
        algo_name = click.prompt(
            "\nВыберите алгоритм",
            type=click.Choice(sorted(ALGORITHM_REGISTRY.keys()),
                              case_sensitive=False),
        )
        params_str = click.prompt(
            "Параметры (JSON, Enter — по умолчанию)",
            default="{}",
        )
        try:
            algo_params = json.loads(params_str)
        except json.JSONDecodeError:
            click.echo("  Некорректный JSON, параметры сброшены в {}.")
            algo_params = {}

        algorithms.append({"name": algo_name, "params": algo_params})

        if not click.confirm("Добавить ещё алгоритм?", default=False):
            break

    # Фильтры (для full_process)
    filters_cfg: List[Dict[str, Any]] = []
    if mode == "full_process":
        click.echo("\n--- Фильтры ---")
        click.echo("Доступные фильтры:")
        for name, entry in sorted(FILTER_REGISTRY.items()):
            click.echo(f"  • {name:20s} — {entry['description']}")

        while True:
            click.echo(f"\nЦепочка фильтров [{len(filters_cfg) + 1}]:")
            chain: List[Dict[str, Any]] = []
            while True:
                filt_type = click.prompt(
                    "  Тип фильтра",
                    type=click.Choice(sorted(FILTER_REGISTRY.keys()),
                                      case_sensitive=False),
                )
                params_str = click.prompt(
                    "  Параметры (JSON)", default="{}"
                )
                try:
                    filt_params = json.loads(params_str)
                except json.JSONDecodeError:
                    filt_params = {}
                chain.append({"type": filt_type, "params": filt_params})

                if not click.confirm("  Добавить ещё фильтр в цепочку?",
                                     default=False):
                    break

            filters_cfg.append({"chain": chain})

            if not click.confirm("Добавить ещё цепочку?", default=False):
                break

    # Выходные данные
    click.echo("\n--- Выходные данные ---")
    output_dir = click.prompt("Директория результатов", default="results")

    gen_report = click.confirm("Генерировать LaTeX-отчёт?", default=False)

    # Сборка конфигурации
    config: Dict[str, Any] = {
        "experiment": {"name": exp_name, "description": exp_desc},
        "input": {
            "images_folder": images_folder,
            "color": color,
            "load_mode": load_mode,
        },
        "output": {
            "restored_folder": f"{output_dir}/restored",
            "data_folder": f"{output_dir}/data",
            "kernel_folder": f"{output_dir}/kernels",
        },
        "processing": {"mode": mode, "metadata": True, "unique_paths": True},
        "algorithms": algorithms,
        "report": {"generate": gen_report, "format": "latex"},
    }

    if bindings:
        config["input"]["bindings"] = bindings
    if bind_state_path:
        config["input"]["bind_state_path"] = bind_state_path
    if filters_cfg:
        config["filters"] = filters_cfg
    if gen_report:
        config["report"]["output_path"] = f"{output_dir}/report.tex"

    click.echo("\nКонфигурация:")
    click.echo(json.dumps(config, ensure_ascii=False, indent=2))

    # Сохранение / Запуск
    action = click.prompt(
        "\nДействие (run / save / both / cancel)",
        type=click.Choice(["run", "save", "both", "cancel"],
                          case_sensitive=False),
        default="both",
    )

    if action in ("save", "both"):
        save_path = click.prompt(
            "Путь для сохранения конфига",
            default=f"configs/{exp_name.replace(' ', '_').lower()}.yaml",
        )
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            if save_path.suffix in (".yaml", ".yml"):
                if HAS_YAML:
                    yaml.dump(config, f, allow_unicode=True,
                              default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config, f, ensure_ascii=False, indent=4)
            else:
                json.dump(config, f, ensure_ascii=False, indent=4)

        click.echo(f"Конфигурация сохранена: {save_path}")

    if action in ("run", "both"):
        errors = validate_config(config)
        if errors:
            click.echo("Ошибки валидации:")
            for err in errors:
                click.echo(f"  • {err}")
            click.echo("Запуск отменён — исправьте конфигурацию.")
            return

        click.echo("\nЗапуск пайплайна...")
        try:
            result = run_pipeline(config=config)
            if result["status"] == "completed":
                click.echo(
                    f"Готово! Обработано: {result['images_processed']} изобр.,"
                    f" время: {result['duration_seconds']:.1f} сек."
                )
        except Exception as exc:
            click.echo(f"Ошибка: {exc}", err=True)

    if action == "cancel":
        click.echo("Отменено.")


# Команда: list-algorithms

@cli.command("list-algorithms")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["table", "latex"]),
              default="table",
              help="Формат вывода.")
def list_algorithms(fmt: str) -> None:
    """
    Вывод списка доступных алгоритмов деконволюции.

    Примеры:

    \b
      python cli.py list-algorithms
      python cli.py list-algorithms --format latex
    """
    headers = ["Имя", "Категория", "Описание"]
    rows = []
    for name in sorted(ALGORITHM_REGISTRY.keys()):
        entry = ALGORITHM_REGISTRY[name]
        rows.append([
            name,
            entry.get("category", "---"),
            entry.get("description", "---"),
        ])

    click.echo("\nДоступные алгоритмы деконволюции:")
    click.echo()
    if fmt == "latex":
        click.echo(_format_table_latex(headers, rows))
    else:
        click.echo(_format_table(headers, rows))
    click.echo(f"\nВсего: {len(rows)} алгоритмов")


# Команда: list-filters

@cli.command("list-filters")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["table", "latex"]),
              default="table",
              help="Формат вывода.")
def list_filters(fmt: str) -> None:
    """
    Вывод списка доступных фильтров.

    Примеры:

    \b
      python cli.py list-filters
      python cli.py list-filters --format latex
    """
    headers = ["Имя", "Описание", "Требует PSF"]
    rows = []
    for name in sorted(FILTER_REGISTRY.keys()):
        entry = FILTER_REGISTRY[name]
        rows.append([
            name,
            entry.get("description", "---"),
            "Да" if entry.get("requires_psf") else "Нет",
        ])

    click.echo("\nДоступные фильтры:")
    click.echo()
    if fmt == "latex":
        click.echo(_format_table_latex(headers, rows))
    else:
        click.echo(_format_table(headers, rows))

    click.echo(f"\nВсего: {len(rows)} фильтров")

    # PSF-функции
    click.echo("\nДоступные PSF-функции (для фильтров с PSF):")
    for name in sorted(PSF_REGISTRY.keys()):
        click.echo(f"  • {name}")


# Entry point

if __name__ == "__main__":
    cli()
