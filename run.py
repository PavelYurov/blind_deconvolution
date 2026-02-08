"""
Конфигурационно-управляемый исполнитель пайплайнов обработки изображений для системы слепой деконволюции.

Выполняет полный цикл обработки изображений из YAML/JSON конфигурационного файла:
загрузка конфига, валидация, построение пайплайна, обработка изображений, 
сохранение результатов и опциональная генерация LaTeX отчётов.

Использование::

    python run.py --config configs/experiment.yaml
    python run.py --config configs/experiment.yaml --output-dir results/
    python run.py --config configs/experiment.yaml --generate-report
    python run.py --config configs/experiment.yaml --dry-run
    python run.py --config configs/experiment.yaml --validate-only

Автор: Беззаборов А.А.
"""

import argparse
import json
import logging
import sys
import os
import importlib
import importlib.util
import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Path configuration
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
_ALGORITHMS_DIR = _SRC_DIR / "blinddeconv" / "algorithms"

# Extend sys.path for package discovery
for _path in [str(_SRC_DIR), str(_ALGORITHMS_DIR)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Optional dependencies
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


logger = logging.getLogger("blinddeconv.run")


# Algorithm, filter, and PSF registries
ALGORITHM_REGISTRY: Dict[str, Dict[str, str]] = {
    "richardson_lucy": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.classic.richardson_lucy",
        "class_name": "RichardsonLucy",
        "description": "Алгоритм Richardson-Lucy",
        "category": "classic",
    },
    "em": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.classic.expectation_maximization",
        "class_name": "EMBlindDeconvolution",
        "description": "EM-алгоритм для слепой деконволюции",
        "category": "classic",
    },
    "map": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.classic.alternating_minimization",
        "class_name": "MAPDeconvolution",
        "description": "MAP с регуляризацией (alternating minimization)",
        "category": "classic",
    },
    "vbbid_tv": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.bayesian.vbbid_tv",
        "class_name": "VBBID_TV",
        "description": "Вариационная байесовская деконволюция с TV априори",
        "category": "bayesian",
    },
    "bbd_deip": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.bayesian.bbd_deip",
        "class_name": "BBD_DEIP",
        "description": "Байесовская деконволюция с разными экспозициями",
        "category": "bayesian",
    },
    "sb_bid_pe": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.bayesian.sb_bid_pe",
        "class_name": "SB_BID_PE",
        "description": "Разреженная байесовская слепая деконволюция",
        "category": "bayesian",
    },
    "vapibe": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.variational.vapibe",
        "class_name": "VAPIBE",
        "description": "Вариационный подход к оценке параметров",
        "category": "variational",
    },
    "vabid": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.variational.vabid",
        "class_name": "VABID",
        "description": "Вариационный байесовский подход (Likas2004)",
        "category": "variational",
    },
    "vbsk_sid_st": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.variational.vbsk_sid_st",
        "class_name": "VBSK_SID_ST",
        "description": "Вариационная байесовская деконволюция со Student's-t",
        "category": "variational",
    },
    "vbc_bid": {
        "module": "blinddeconv.algorithms.blind_deconvolution"
                  ".our_company.sparse.vbc_bid",
        "class_name": "VBC_BID",
        "description": "Компрессивная байесовская деконволюция",
        "category": "sparse",
    },
    "hqmbr": {
        "module": "blinddeconv.algorithms.nonblind_deconvolution"
                  ".third_party_company.HQMotionBlurRestoration.HQMBR",
        "class_name": "HQMBR",
        "description": "High-Quality Motion Blur Restoration",
        "category": "nonblind",
    },
}

FILTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "defocus_blur": {
        "module": "blinddeconv.filters.blur",
        "class_name": "DefocusBlur",
        "requires_psf": True,
        "description": "Размытие вне фокуса (2D)",
    },
    "motion_blur": {
        "module": "blinddeconv.filters.blur",
        "class_name": "MotionBlur",
        "requires_psf": True,
        "description": "Размытие в движении (1D)",
    },
    "bspline_blur": {
        "module": "blinddeconv.filters.blur",
        "class_name": "BSpline_blur",
        "requires_psf": False,
        "description": "Криволинейное размытие (B-spline)",
    },
    "kernel_convolution": {
        "module": "blinddeconv.filters.blur",
        "class_name": "Kernel_convolution",
        "requires_psf": False,
        "description": "Свёртка с ядром из .npy файла",
    },
    "gaussian_noise": {
        "module": "blinddeconv.filters.noise",
        "class_name": "GaussianNoise",
        "requires_psf": False,
        "description": "Аддитивный гауссовский шум",
    },
    "poisson_noise": {
        "module": "blinddeconv.filters.noise",
        "class_name": "PoissonNoise",
        "requires_psf": False,
        "description": "Пуассоновский шум",
    },
    "salt_pepper_noise": {
        "module": "blinddeconv.filters.noise",
        "class_name": "SaltAndPepperNoise",
        "requires_psf": False,
        "description": "Импульсный шум (соль и перец)",
    },
    "mean_blur": {
        "module": "blinddeconv.filters.smooth",
        "class_name": "MeanBlur",
        "requires_psf": False,
        "description": "Сглаживание средним",
    },
    "median_blur": {
        "module": "blinddeconv.filters.smooth",
        "class_name": "MedianBlur",
        "requires_psf": False,
        "description": "Медианное сглаживание",
    },
    "gaussian_blur": {
        "module": "blinddeconv.filters.smooth",
        "class_name": "GaussianBlur",
        "requires_psf": False,
        "description": "Гауссово сглаживание",
    },
    "bilateral_filter": {
        "module": "blinddeconv.filters.smooth",
        "class_name": "BilateralFilter",
        "requires_psf": False,
        "description": "Билатеральный фильтр",
    },
}

PSF_REGISTRY: Dict[str, str] = {
    "gaussian": "blinddeconv.filters.distributions.gaussian_distribution",
    "uniform": "blinddeconv.filters.distributions.uniform_distribution",
    "linear_decay": "blinddeconv.filters.distributions.linear_decay_distribution",
    "ring": "blinddeconv.filters.distributions.ring_distribution",
    "exponential_decay": "blinddeconv.filters.distributions.exponential_decay_distribution",
}


# JSON Schema for configuration validation

CONFIG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["input", "algorithms"],
    "properties": {
        "experiment": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
            },
        },
        "input": {
            "type": "object",
            "required": ["images_folder"],
            "properties": {
                "images_folder": {"type": "string"},
                "blurred_folder": {"type": "string"},
                "color": {"type": "boolean"},
                "load_mode": {
                    "type": "string",
                    "enum": ["all", "bind", "bind_state"],
                },
                "bindings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["original", "blurred"],
                        "properties": {
                            "original": {"type": "string"},
                            "blurred": {"type": "string"},
                            "kernel": {"type": "string"},
                            "filter_description": {"type": "string"},
                        },
                    },
                },
                "bind_state_path": {"type": "string"},
            },
        },
        "output": {
            "type": "object",
            "properties": {
                "restored_folder": {"type": "string"},
                "data_folder": {"type": "string"},
                "kernel_folder": {"type": "string"},
            },
        },
        "algorithms": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "module": {"type": "string"},
                    "class_name": {"type": "string"},
                    "params": {"type": "object"},
                },
            },
        },
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["chain"],
                "properties": {
                    "chain": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type"],
                            "properties": {
                                "type": {"type": "string"},
                                "params": {"type": "object"},
                            },
                        },
                    }
                },
            },
        },
        "processing": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["process", "full_process"],
                },
                "metadata": {"type": "boolean"},
                "unique_paths": {"type": "boolean"},
            },
        },
        "report": {
            "type": "object",
            "properties": {
                "generate": {"type": "boolean"},
                "format": {"type": "string", "enum": ["latex"]},
                "output_path": {"type": "string"},
            },
        },
    },
}


# Configuration loading and validation

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML или JSON файла.

    Параметры
    ---------
    config_path : Union[str, Path]
        Путь к файлу конфигурации (.yaml, .yml или .json).

    Возвращает
    ----------
    Dict[str, Any]
        Словарь с параметрами конфигурации.

    Raises
    ------
    FileNotFoundError
        Если файл конфигурации не найден.
    ValueError
        Если формат файла не поддерживается.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {config_path}"
        )

    suffix = config_path.suffix.lower()
    logger.info("Загрузка конфигурации из %s", config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError(
                    "Для загрузки YAML-конфигов требуется библиотека PyYAML. "
                    "Установите: pip install pyyaml"
                )
            config = yaml.safe_load(f)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(
                f"Неподдерживаемый формат: {suffix}. "
                "Используйте .yaml, .yml или .json"
            )

    if config is None:
        raise ValueError(f"Файл конфигурации пуст: {config_path}")

    logger.info("Конфигурация загружена: %s",
                config.get("experiment", {}).get("name", "без имени"))
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Валидация конфигурации по JSON Schema и логическим правилам.

    Параметры
    ---------
    config : Dict[str, Any]
        Словарь конфигурации.

    Возвращает
    ----------
    List[str]
        Список ошибок валидации (пустой, если ошибок нет).
    """
    errors: List[str] = []

    # Валидация через JSON Schema
    if HAS_JSONSCHEMA:
        validator = jsonschema.Draft7Validator(CONFIG_SCHEMA)
        for error in sorted(validator.iter_errors(config), key=str):
            errors.append(f"[schema] {error.message} (путь: {list(error.path)})")
    else:
        logger.warning(
            "jsonschema не установлен — schema-валидация пропущена. "
            "Установите: pip install jsonschema"
        )
        # Базовая проверка без jsonschema
        if "input" not in config:
            errors.append("Отсутствует обязательная секция 'input'")
        if "algorithms" not in config:
            errors.append("Отсутствует обязательная секция 'algorithms'")

    # Логическая валидация
    processing = config.get("processing", {})
    mode = processing.get("mode", "process")
    load_mode = config.get("input", {}).get("load_mode", "all")

    if mode == "process" and load_mode not in ("bind", "bind_state"):
        errors.append(
            "Режим 'process' требует load_mode 'bind' или 'bind_state' "
            "(нужны пары оригинал + смазанное)"
        )

    if load_mode == "bind":
        bindings = config.get("input", {}).get("bindings", [])
        if not bindings:
            errors.append(
                "load_mode='bind', но список bindings пуст или не указан"
            )

    if load_mode == "bind_state":
        bind_state_path = config.get("input", {}).get("bind_state_path")
        if not bind_state_path:
            errors.append(
                "load_mode='bind_state', но bind_state_path не указан"
            )

    if mode == "full_process":
        filters_cfg = config.get("filters", [])
        if not filters_cfg:
            errors.append(
                "Режим 'full_process' требует хотя бы одну цепочку фильтров "
                "в секции 'filters'"
            )

    # Проверка алгоритмов
    for i, algo in enumerate(config.get("algorithms", [])):
        algo_name = algo.get("name", "")
        if algo_name not in ALGORITHM_REGISTRY and "module" not in algo:
            errors.append(
                f"Алгоритм [{i}] '{algo_name}': неизвестное имя и "
                "не указан полный путь (module + class_name). "
                f"Доступные: {', '.join(sorted(ALGORITHM_REGISTRY.keys()))}"
            )

    # Проверка фильтров
    for i, chain_cfg in enumerate(config.get("filters", [])):
        for j, filt in enumerate(chain_cfg.get("chain", [])):
            filt_type = filt.get("type", "")
            if filt_type not in FILTER_REGISTRY:
                errors.append(
                    f"Фильтр [{i}][{j}] '{filt_type}': неизвестный тип. "
                    f"Доступные: {', '.join(sorted(FILTER_REGISTRY.keys()))}"
                )

    return errors


# Dynamic module import

def _import_class(module_path: str, class_name: str) -> type:
    """
    Динамический импорт класса по пути модуля и имени класса.

    Используются две стратегии:
    1. Стандартный импорт через importlib.import_module.
    2. Импорт из файла через importlib.util (fallback для модулей
       без __init__.py в промежуточных директориях).

    Параметры
    ---------
    module_path : str
        Путь к модулю (например, 'blinddeconv.algorithms...vabid').
    class_name : str
        Имя класса внутри модуля.

    Возвращает
    ----------
    type
        Импортированный класс.

    Raises
    ------
    ImportError
        Если модуль или класс не удалось загрузить.
    """
    # Стратегия 1: стандартный импорт
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, ModuleNotFoundError):
        pass

    # Стратегия 2: импорт из файла
    file_path = _SRC_DIR / Path(*module_path.split("."))
    file_path = file_path.with_suffix(".py")

    if not file_path.exists():
        raise ImportError(
            f"Не удалось импортировать модуль '{module_path}': "
            f"файл {file_path} не найден"
        )

    logger.debug("Импорт из файла: %s", file_path)
    spec = importlib.util.spec_from_file_location(module_path, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Не удалось создать spec для '{module_path}' из {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[module_path]
        raise ImportError(
            f"Ошибка при загрузке модуля '{module_path}': {exc}"
        ) from exc

    if not hasattr(module, class_name):
        raise ImportError(
            f"Класс '{class_name}' не найден в модуле '{module_path}'"
        )

    return getattr(module, class_name)


# Algorithm and filter factories

def create_algorithm(algo_config: Dict[str, Any]) -> Any:
    """
    Создание экземпляра алгоритма деконволюции по конфигурации.

    Параметры
    ---------
    algo_config : Dict[str, Any]
        Конфигурация алгоритма с ключами:
        - name: краткое имя из реестра или произвольное имя
        - module: (опционально) полный путь модуля
        - class_name: (опционально) имя класса в модуле
        - params: (опционально) словарь параметров конструктора

    Возвращает
    ----------
    DeconvolutionAlgorithm
        Экземпляр алгоритма.

    Raises
    ------
    ValueError
        Если алгоритм не найден в реестре и не указан module/class_name.
    """
    name = algo_config.get("name", "unknown")
    params = algo_config.get("params", {})

    # Определяем модуль и класс
    if "module" in algo_config and "class_name" in algo_config:
        module_path = algo_config["module"]
        class_name = algo_config["class_name"]
    elif name in ALGORITHM_REGISTRY:
        entry = ALGORITHM_REGISTRY[name]
        module_path = entry["module"]
        class_name = entry["class_name"]
    else:
        raise ValueError(
            f"Алгоритм '{name}' не найден в реестре и не указан "
            "module/class_name для пользовательского импорта. "
            f"Доступные алгоритмы: {', '.join(sorted(ALGORITHM_REGISTRY.keys()))}"
        )

    logger.info("Создание алгоритма: %s (%s.%s)", name, module_path, class_name)
    cls = _import_class(module_path, class_name)

    try:
        instance = cls(**params)
    except TypeError as exc:
        raise ValueError(
            f"Ошибка создания алгоритма '{name}': {exc}. "
            f"Проверьте параметры: {params}"
        ) from exc

    logger.info("Алгоритм '%s' создан успешно (параметры: %s)", name, params)
    return instance


def _resolve_psf_function(psf_name: str) -> Any:
    """
    Получение PSF-функции по имени из реестра.

    Параметры
    ---------
    psf_name : str
        Имя PSF-функции (например, 'gaussian', 'uniform').

    Возвращает
    ----------
    Callable
        PSF-функция.
    """
    if psf_name not in PSF_REGISTRY:
        raise ValueError(
            f"PSF-функция '{psf_name}' не найдена. "
            f"Доступные: {', '.join(sorted(PSF_REGISTRY.keys()))}"
        )

    full_path = PSF_REGISTRY[psf_name]
    module_path, func_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def create_filter(filter_config: Dict[str, Any]) -> Any:
    """
    Создание экземпляра фильтра по конфигурации.

    Параметры
    ---------
    filter_config : Dict[str, Any]
        Конфигурация фильтра с ключами:
        - type: имя фильтра из реестра
        - params: словарь параметров конструктора

    Возвращает
    ----------
    FilterBase
        Экземпляр фильтра.
    """
    filt_type = filter_config.get("type", "unknown")
    params = dict(filter_config.get("params", {}))

    if filt_type not in FILTER_REGISTRY:
        raise ValueError(
            f"Фильтр '{filt_type}' не найден. "
            f"Доступные: {', '.join(sorted(FILTER_REGISTRY.keys()))}"
        )

    entry = FILTER_REGISTRY[filt_type]
    module_path = entry["module"]
    class_name = entry["class_name"]

    # Подстановка PSF-функции для фильтров размытия
    if entry.get("requires_psf") and "psf" in params:
        params["psf"] = _resolve_psf_function(params["psf"])

    logger.info("Создание фильтра: %s (%s)", filt_type, params)
    cls = _import_class(module_path, class_name)

    try:
        instance = cls(**params)
    except TypeError as exc:
        raise ValueError(
            f"Ошибка создания фильтра '{filt_type}': {exc}. "
            f"Проверьте параметры: {params}"
        ) from exc

    return instance


def create_filter_chains(
    filters_config: List[Dict[str, Any]]
) -> List[List[Any]]:
    """
    Создание цепочек фильтров из конфигурации.

    Параметры
    ---------
    filters_config : List[Dict[str, Any]]
        Список конфигураций цепочек фильтров.

    Возвращает
    ----------
    List[List[FilterBase]]
        Список цепочек фильтров [[filter1, filter2], [filter3]].
    """
    chains: List[List[Any]] = []
    for chain_cfg in filters_config:
        chain: List[Any] = []
        for filt_cfg in chain_cfg.get("chain", []):
            chain.append(create_filter(filt_cfg))
        if chain:
            chains.append(chain)
    return chains


# LaTeX report generation

def generate_latex_report(
    config: Dict[str, Any],
    results_dir: Path,
    output_path: Path,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    """
    Генерация LaTeX-отчёта с результатами эксперимента.

    Параметры
    ---------
    config : Dict[str, Any]
        Конфигурация эксперимента.
    results_dir : Path
        Директория с результатами.
    output_path : Path
        Путь для сохранения .tex файла.
    start_time : datetime.datetime
        Время начала эксперимента.
    end_time : datetime.datetime
        Время окончания эксперимента.
    """
    experiment = config.get("experiment", {})
    exp_name = experiment.get("name", "Blind Deconvolution Experiment")
    exp_desc = experiment.get("description", "")
    duration = (end_time - start_time).total_seconds()

    # Собираем информацию об алгоритмах
    algo_rows = []
    for algo in config.get("algorithms", []):
        name = algo.get("name", "unknown")
        params = algo.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        entry = ALGORITHM_REGISTRY.get(name, {})
        desc = entry.get("description", name)
        category = entry.get("category", "---")
        algo_rows.append((name, category, desc, params_str))

    # Формируем LaTeX-документ
    algo_table_rows = "\n".join(
        f"        {n} & {cat} & {d} & {p} \\\\"
        for n, cat, d, p in algo_rows
    )

    latex_content = rf"""\documentclass[a4paper,12pt]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T2A]{{fontenc}}
\usepackage[russian]{{babel}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{geometry}}
\geometry{{margin=2cm}}

\title{{{exp_name}}}
\author{{Автоматически сгенерировано blinddeconv}}
\date{{{start_time.strftime('%d.%m.%Y %H:%M')}}}

\begin{{document}}
\maketitle

\section{{Описание эксперимента}}
{exp_desc}

\begin{{itemize}}
    \item \textbf{{Время запуска:}} {start_time.strftime('%d.%m.%Y %H:%M:%S')}
    \item \textbf{{Время завершения:}} {end_time.strftime('%d.%m.%Y %H:%M:%S')}
    \item \textbf{{Длительность:}} {duration:.1f} сек.
    \item \textbf{{Режим:}} {config.get('processing', {}).get('mode', 'process')}
    \item \textbf{{Цветовой режим:}} {{'Цветной' if config.get('input', {{}}).get('color', False) else 'Ч/Б'}}
    \item \textbf{{Директория результатов:}} \verb|{results_dir}|
\end{{itemize}}

\section{{Алгоритмы}}

\begin{{longtable}}{{llll}}
    \toprule
    \textbf{{Имя}} & \textbf{{Категория}} & \textbf{{Описание}} & \textbf{{Параметры}} \\
    \midrule
{algo_table_rows}
    \bottomrule
\end{{longtable}}

\section{{Результаты}}
Результаты сохранены в директории \verb|{results_dir}|.

Для детального анализа используйте CSV-таблицы в директории
\verb|{config.get('output', {}).get('data_folder', 'data')}|.

\end{{document}}
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    logger.info("LaTeX-отчёт сохранён: %s", output_path)


# Experiment metadata export

def export_metadata(
    config: Dict[str, Any],
    results_dir: Path,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    """
    Экспорт метаданных эксперимента в metadata.json.

    Параметры
    ---------
    config : Dict[str, Any]
        Конфигурация эксперимента.
    results_dir : Path
        Директория результатов.
    start_time : datetime.datetime
        Время начала.
    end_time : datetime.datetime
        Время окончания.
    """
    metadata = {
        "experiment": config.get("experiment", {}),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "config": config,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    metadata_path = results_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4, default=str)

    logger.info("Метаданные сохранены: %s", metadata_path)


# Main pipeline

def run_pipeline(
    config: Dict[str, Any],
    output_dir: Optional[str] = None,
    generate_report: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Запуск пайплайна обработки по конфигурации.

    Параметры
    ---------
    config : Dict[str, Any]
        Словарь конфигурации эксперимента.
    output_dir : Optional[str]
        Переопределение директории результатов.
    generate_report : bool
        Генерировать LaTeX-отчёт (перекрывает config.report.generate).
    dry_run : bool
        Если True — проверка без выполнения обработки.

    Возвращает
    ----------
    Dict[str, Any]
        Словарь с информацией о результатах.
    """
    from blinddeconv.processing.core import Processing

    start_time = datetime.datetime.now()
    exp_name = config.get("experiment", {}).get("name", "Эксперимент")
    logger.info("Запуск: %s", exp_name)

    # Извлечение параметров
    input_cfg = config.get("input", {})
    output_cfg = config.get("output", {})
    processing_cfg = config.get("processing", {})
    report_cfg = config.get("report", {})

    images_folder = input_cfg.get("images_folder", "images")
    blurred_folder = input_cfg.get("blurred_folder", "blurred")
    color = input_cfg.get("color", False)
    load_mode = input_cfg.get("load_mode", "all")
    mode = processing_cfg.get("mode", "process")
    metadata_flag = processing_cfg.get("metadata", True)
    unique_paths = processing_cfg.get("unique_paths", True)

    # Переопределение выходных директорий
    if output_dir:
        restored_folder = str(Path(output_dir) / "restored")
        data_folder = str(Path(output_dir) / "data")
        kernel_folder = str(Path(output_dir) / "kernels")
    else:
        restored_folder = output_cfg.get("restored_folder", "restored")
        data_folder = output_cfg.get("data_folder", "data")
        kernel_folder = output_cfg.get("kernel_folder", "kernels")

    results_dir = Path(restored_folder).parent

    # Dry-run: только вывод плана
    if dry_run:
        logger.info("[DRY-RUN] Пайплайн НЕ будет выполнен")
        logger.info("[DRY-RUN] Режим: %s", mode)
        logger.info("[DRY-RUN] Изображения: %s (load_mode=%s)",
                     images_folder, load_mode)
        logger.info("[DRY-RUN] Цветовой режим: %s",
                     "цветной" if color else "ч/б")
        logger.info("[DRY-RUN] Результаты: %s", restored_folder)

        for i, algo in enumerate(config.get("algorithms", [])):
            logger.info("[DRY-RUN] Алгоритм [%d]: %s (params=%s)",
                         i, algo.get("name"), algo.get("params", {}))

        for i, chain_cfg in enumerate(config.get("filters", [])):
            chain_desc = " → ".join(
                f.get("type", "?") for f in chain_cfg.get("chain", [])
            )
            logger.info("[DRY-RUN] Цепочка фильтров [%d]: %s",
                         i, chain_desc)

        logger.info("[DRY-RUN] Проверка завершена.")
        return {"status": "dry_run", "config": config}

    # Создание экземпляров алгоритмов
    logger.info("Инициализация алгоритмов...")
    algorithms = []
    for algo_cfg in config.get("algorithms", []):
        alg = create_algorithm(algo_cfg)
        algorithms.append(alg)
    logger.info("Создано алгоритмов: %d", len(algorithms))

    # Создание Processing 
    logger.info("Инициализация Processing...")
    processing = Processing(
        images_folder=images_folder,
        blurred_folder=blurred_folder,
        restored_folder=restored_folder,
        data_path=data_folder,
        color=color,
        kernel_dir=kernel_folder,
    )

    # Загрузка изображений
    if load_mode == "all":
        logger.info("Загрузка всех изображений из '%s'...", images_folder)
        processing.read_all()
        logger.info("Загружено изображений: %d", len(processing.images))

    elif load_mode == "bind":
        bindings = input_cfg.get("bindings", [])
        logger.info("Связывание %d пар изображений...", len(bindings))
        for binding in bindings:
            processing.bind(
                original_image_path=binding["original"],
                blurred_image_path=binding["blurred"],
                original_kernel_path=binding.get("kernel"),
                filter_description=binding.get("filter_description", "unknown"),
                color=color,
            )
        logger.info("Связано пар: %d", len(bindings))

    elif load_mode == "bind_state":
        bind_state_path = input_cfg["bind_state_path"]
        logger.info("Загрузка состояния из '%s'...", bind_state_path)
        processing.load_bind_state(bind_state_path)
        logger.info("Загружено изображений: %d", len(processing.images))

    if len(processing.images) == 0:
        logger.warning("Не загружено ни одного изображения!")
        return {"status": "no_images", "config": config}

    if mode == "full_process":
        logger.info("Запуск полного пайплайна (full_process)...")
        filter_chains = create_filter_chains(config.get("filters", []))
        logger.info("Создано цепочек фильтров: %d", len(filter_chains))

        processing.full_process(
            filters=filter_chains,
            methods=algorithms,
        )

    elif mode == "process":
        logger.info("Запуск обработки (process)...")
        for alg in algorithms:
            logger.info("Обработка алгоритмом '%s'...", alg.get_name())
            processing.process(
                algorithm_processor=alg,
                metadata=metadata_flag,
                unique_path=unique_paths,
            )

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info("Обработка завершена за %.1f сек.", duration)

    # Экспорт метаданных
    export_metadata(config, results_dir, start_time, end_time)

    # Генерация отчёта
    should_generate = generate_report or report_cfg.get("generate", False)
    if should_generate:
        # report_path = Path(
        #     report_cfg.get("output_path", str(results_dir / "report.tex"))
        # )
        yaml_output_path = report_cfg.get("output_path")
        if output_dir:
            filename = Path(yaml_output_path).name if yaml_output_path else "report.tex"
            report_path = results_dir / filename
        else:
            report_path = Path(yaml_output_path) if yaml_output_path else results_dir / "report.tex"
        logger.info("Генерация LaTeX-отчёта: %s", report_path)
        generate_latex_report(config, results_dir, report_path,
                              start_time, end_time)

    return {
        "status": "completed",
        "duration_seconds": duration,
        "results_dir": str(results_dir),
        "images_processed": len(processing.images),
        "algorithms_used": [a.get_name() for a in algorithms],
    }


# Logging setup

def setup_logging(verbose: bool = False) -> None:
    """
    Настройка логирования для скрипта.

    Параметры
    ---------
    verbose : bool
        Если True — уровень DEBUG, иначе INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(level)

    root_logger = logging.getLogger("blinddeconv")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Подавляем лишний вывод от библиотек
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


# CLI entry point

def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов командной строки.

    Возвращает
    ----------
    argparse.Namespace
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Автоматизация пайплайна слепой деконволюции "
            "по конфигурационному файлу."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python run.py --config configs/basic_deconvolution.yaml
  python run.py --config configs/experiment.yaml --output-dir results/
  python run.py --config configs/experiment.yaml --generate-report
  python run.py --config configs/experiment.yaml --dry-run
  python run.py --config configs/experiment.yaml --validate-only
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Путь к файлу конфигурации (.yaml или .json)",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Переопределение директории для результатов",
    )

    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=False,
        help="Генерировать LaTeX-отчёт по результатам",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Проверка конфигурации без выполнения обработки",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        help="Только валидация конфигурации (без запуска)",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Подробный вывод (уровень DEBUG)",
    )

    return parser.parse_args()


def main() -> None:
    """Главная точка входа скрипта."""
    args = parse_args()
    setup_logging(verbose=args.verbose)

    try:
        # Загрузка конфигурации
        config = load_config(args.config)

        # Валидация
        errors = validate_config(config)
        if errors:
            logger.error("Ошибки валидации конфигурации:")
            for err in errors:
                logger.error("  • %s", err)
            sys.exit(1)

        logger.info("Конфигурация валидна.")

        if args.validate_only:
            logger.info("Режим --validate-only: конфигурация корректна.")
            sys.exit(0)

        # Запуск пайплайна
        result = run_pipeline(
            config=config,
            output_dir=args.output_dir,
            generate_report=args.generate_report,
            dry_run=args.dry_run,
        )

        if result["status"] == "completed":
            logger.info(
                "Готово! Обработано изображений: %d, "
                "время: %.1f сек., результаты: %s",
                result["images_processed"],
                result["duration_seconds"],
                result["results_dir"],
            )

    except FileNotFoundError as exc:
        logger.error("Файл не найден: %s", exc)
        sys.exit(1)
    except ImportError as exc:
        logger.error("Ошибка импорта: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Ошибка значения: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
        sys.exit(130)
    except Exception as exc:
        logger.exception("Непредвиденная ошибка: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
