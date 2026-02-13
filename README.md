# Исследование методов Blind Image Deconvolution

[![CI](https://github.com/PavelYurov/blind_deconvolution/actions/workflows/ci.yml/badge.svg?branch=feature/automation-system)](https://github.com/PavelYurov/blind_deconvolution/actions/workflows/ci.yml)

## Руководитель проекта 

- Парфенов Денис Васильевич, promasterden@yandex.ru

## Участники проекта

- Беззаборов А.А., КМБО-01-22, antonbezzaborov929@gmail.com - Тимлид-разработчик, Variational Bayesian (Robust & Sparse Priors) алгормитмы и фреймворк
- Юров П.И., КМБО-01-22, pavel.yurov0425@gmail.com - Разработчик, Non-Local Low-Rank Tensor Approximation алгормитмы и фреймворк
- Куропатов К.Л., КМБО-01-22, konstantinkuropatov@gmail.com -  Разработчик-теоретик, Variational Bayesian (Total Variation & Structured Priors) алгоритмы
- Малыш Я.В., КМБО-03-22, mrgeroixyu@gmail.com - Разработчик, L0/L1-Regularized Optimization алгормитмы и фреймворк

## Описание проекта

Данный проект посвящен исследованию методов слепой деконволюции (blind deconvolution) с интегрированной системой автоматической оптимизации гиперпараметров. Основное внимание уделяется разработке и сравнению алгоритмов, способных восстанавливать исходное изображение без априорной информации о функции искажения. Проект обеспечивает комплексное исследование алгоритмов с систематической оценкой качества восстановления и подбором оптимальных гиперпараметров.

### Цель исследования

Разработка, сравнительный анализ предельных возможностей методов слепой деконволюции и выявление наиболее эффективных подходов для восстановления изображений, искаженных различными типами размытия и шумов.

### Основные задачи

- **Разработка системы автоматического подбора гиперпараметров** для методов слепой деконволюции
- **Разработка пайплайна** генерации реалистичных искажений изображений
- **Реализация и сравнение** классических и современных методов восстановления
- **Построение многомерных Парето-фронтов** для анализа компромиссов между качеством и производительностью
- **Систематическая оценка** устойчивости алгоритмов к шумам и смазам

### Функциональность фреймворка

- **Обработка**: монохромные и цветные изображения (JPEG, BMP, PNG), пакетная обработка
- **Генерация искажений**: расфокус, motion blur, B-spline траектории, гауссов/пуассонов/импульсный шум
- **Методы восстановления**: классические, байесовские, вариационные, разреженные
- **Метрики качества**: PSNR, SSIM, SML, Sharpness
- **Оптимизация**: байесовская (TPE, GP), случайный поиск, NSGA-II
- **Визуализация**: 3D Парето-фронты, сравнительный анализ, тепловые карты

---

## Техническое описание

### 1. Клонирование репозитория и настройка окружения

```bash
git clone https://github.com/PavelYurov/blind_deconvolution.git
cd blind_deconvolution

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Установка пакета

Базовая установка (минимальные зависимости)
```bash
pip install -e .
```

С поддержкой CLI-интерфейса (run.py, cli.py)
```bash
pip install -e ".[cli]"
```
Для разработки (все зависимости и инструменты разработки)
```bash
pip install -e ".[cli,dev,docs]"
```

**Альтернативный способ — без клонирования**

Если вам нужна только библиотека без исходного кода:

```bash
pip install git+https://github.com/PavelYurov/blind_deconvolution.git
```

### 3. Использование

**В качестве Python-библиотеки**

```python
from blinddeconv.processing import Processing
from blinddeconv.algorithms.task_type.company_type.algorithm_type.algorithm_name import ALGORITHM
```
Инициализация обработчика
```python
proc = Processing(images_folder="images/original", color=False)
```
Бинд оригинального и искажённого изображений
```python
proc.bind("images/original/image.png",
          "images/distorted/image_blurred.png")
```
Восстановление изображения и визуализация
```python
proc.process(ALGORITHM, metadata=True)
proc.show()
```

**Через командную строку (CLI)**

Все команды выполняются в активированном виртуальном окружении:

Запуск полного пайплайна с использованием конфигурации
```bash
python run.py --config configs/experiment.yaml
```
Быстрая обработка одного изображени
```bash
python cli.py process \
  --input image.jpg \
  --algorithm algorithm_name
```
Интерактивный мастер настройки
```bash
python cli.py interactive
```
Просмотр доступных команд
```bash
python cli.py --help
```

---

## Структура проекта

Подробнее об алгоритмах: [Путеводитель по алгоритмам](src/blinddeconv/algorithms/README.md)

```
blind_deconvolution/
├── src/                                 # Исходники Python-пакета
│   └── blinddeconv/                     # Python-пакет blinddeconv
│       ├── algorithms/                  # Алгоритмы и обёртки
│       │   ├── base.py                  # DeconvolutionAlgorithm (ABC)
│       │   ├── blind_deconvolution/     # Алгоритмы восстановления изображений вслепую
│       │   │   ├── our_company/         # Собственные реализации
│       │   │   │   ├── bayesian/
│       │   │   │   ├── classic/
│       │   │   │   ├── sparse/
│       │   │   │   └── variational/
│       │   │   └── third_party_company/
│       │   ├── nonblind_deconvolution/  # Алгоритмы восстановления с известным PSF
│       │   ├── kernel_estimation/       # Алгоритмы оценки PSF
│       │   ├── unsorted/                # Экспериментальные
│       │   └── README.md                # Путеводитель по алгоритмам
│       ├── filters/                     # Генерация искажений
│       │   ├── base.py                  # FilterBase (ABC)
│       │   ├── blur.py                  # DefocusBlur, MotionBlur, и др.
│       │   ├── noise.py                 # GaussianNoise, PoissonNoise, и др.
│       │   ├── smooth.py                # MeanBlur, GaussianBlur, и др.
│       │   └── distributions.py         # PSF-функции
│       ├── processing/                  # Ядро фреймворка
│       │   ├── core.py                  # Processing (фасад)
│       │   ├── utils.py                 # Image, утилиты
│       │   ├── metrics.py               # PSNR, SSIM
│       │   ├── reader.py                # Загрузка изображений
│       │   ├── display.py               # Визуализация
│       │   ├── applyfilter.py           # Применение фильтров
│       │   ├── restore.py               # Восстановление (один алгоритм)
│       │   ├── restorepipeline.py       # Полный пайплайн восстановления
│       │   ├── preprocessing.py         # Выравнивание гистограмм
│       │   ├── tables.py                # Экспорт в CSV-таблицы
│       │   ├── clear.py                 # Очистка
│       │   └── extensions/              # Расширения
│       │       ├── base.py
│       │       ├── hyperparameter_optimization.py
│       │       └── pareto_analysis.py
│       ├── system/                      # Служебные модули
│       │   ├── octave/                  # Octave/MATLAB-обвязка
│       │   │   ├── octaveconfig.py
│       │   │   └── octavewrapper.py
│       │   └── cython/                
│       └── scripts/                     # Генераторы данных
│           ├── dataset_generator.py
│           └── kernel_generator.py
│
├── scripts/                             # Утилиты проекта
│   ├── install.py                       # Интерактивный установщик
│   └── uninstall.py                     # Интерактивное удаление
│
├── configs/                             # Конфигурационные файлы
│   ├── basic_deconvolution.yaml         # Базовый пример (YAML)
│   ├── ...
│   └── experiment_template.json         # Полный шаблон (JSON)
├── run.py                               # Автоматизация по конфигам
├── cli.py                               # CLI-интерфейс
│
├── docs/                                # Документация (Sphinx + Markdown)
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── *.md                         # Markdown-документация
│   └── tools/
│       └── build_docs.py                # Сборка документации
│
├── images/                              # Примеры изображений/искажения
│   ├── dataset_bind.json
│   ├── distorted/
│   │   └── ...
│   └── ...
├── references/                          # PDF-материалы/статьи
│   └── *.pdf
├── tests/                               # Тестовые данные/выходы прогонов
│   └── ...
├── utils/                               # Вспомогательные утилиты
│   └── preflight/                       # Проверка зависимостей
│       ├── config.py, report.py
│       └── checks/
├── pyproject.toml
├── requirements.txt
├── setup.cfg
└── README.md
```

---

## Подробная документация

Полная документация доступна в `docs/source/`:

| Раздел | Описание |
|--------|----------|
| [Установка и настройка](docs/source/installation.md) | Все способы установки, профили зависимостей, интерактивный установщик, настройка Octave, Cython |
| [Руководство пользователя](docs/source/usage_guide.md) | `run.py`, `cli.py`, все команды, примеры Python-кода, сборка документации |
| [Конфигурационные файлы](docs/source/configuration.md) | Полная структура YAML/JSON-конфигов, валидация, реестры алгоритмов и фильтров |
| [Архитектура системы](docs/source/architecture.md) | Компоненты, паттерны проектирования, организация модулей |
| [Поток данных](docs/source/data_flow.md) | Схемы `process` и `full_process`, `bind()`, `filter()`, формат `dataset.json` |
| [API Reference](docs/source/api_reference.md) | Полная справка по классам, методам и функциям |
| [Для разработчиков](docs/source/CONTRIBUTING.md) | Стандарты кода, добавление алгоритмов/фильтров, линтинг |

Собранная HTML-документация: [pavelyurov.github.io/blind_deconvolution](https://pavelyurov.github.io/blind_deconvolution/)
