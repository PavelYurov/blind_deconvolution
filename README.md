# Исследование методов Blind Image Deconvolution

## Руководитель проекта 

- Парфенов Денис Васильевич, promasterden@yandex.ru

## Участники проекта

- Беззаборов А.А., КМБО-01-22, antonbezzaborov929@gmail.com - Тимлид
- Юров П.И., КМБО-01-22, pavel.yurov0425@gmail.com - Программист
- Куропатов К.Л., КМБО-01-22, konstantinkuropatov@gmail.com - Теоретик
- Малыш Я.В., КМБО-03-22, mrgeroixyu@gmail.com - Программист

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

## Модульная архитектура

Проект построен по модульному принципу с центральным классом-фасадом `Processing`:

```
Processing (фасад)
├── reader          — загрузка изображений, создание связей
├── display         — визуализация результатов
├── apply_filter    — применение фильтров (blur, noise, denoise)
├── module_process  — восстановление (один алгоритм)
├── process_pipeline— полный пайплайн (фильтры → восстановление → анализ)
├── histogram       — предобработка (выравнивание гистограмм)
├── tables          — экспорт метрик (CSV)
├── clear           — очистка связей и файлов
├── optimizer       — оптимизация гиперпараметров (Optuna)
└── analyzer        — Парето-анализ
```

### Функциональность

- **Обработка**: монохромные и цветные изображения (JPEG, BMP, PNG), пакетная обработка
- **Генерация искажений**: расфокус, motion blur, B-spline траектории, гауссов/пуассонов/импульсный шум
- **Методы восстановления**: классические, байесовские, вариационные, разреженные
- **Метрики качества**: PSNR, SSIM, SML, Sharpness
- **Оптимизация**: байесовская (TPE, GP), случайный поиск, NSGA-II
- **Визуализация**: 3D Парето-фронты, сравнительный анализ, тепловые карты

---

## Быстрый старт

### Установка

```bash
pip install git+https://github.com/PavelYurov/blind_deconvolution.git
```

### Использование как Python-библиотеки

```python
from blinddeconv.processing import Processing
from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID

# Создаём фреймворк
proc = Processing(images_folder="images/original", color=False)

# Связываем оригинал с искажённым
proc.bind("images/original/airplane.png",
          "images/distorted/airplane_blurred.png",
          "images/kernel_data/kernel.npy",
          "gaussian_blur")

# Восстанавливаем
proc.process(VABID(max_iter=100, kernel_size=21), metadata=True)

# Визуализация
proc.show()
```

### Использование через CLI

```bash
# Установка CLI-зависимостей
pip install pyyaml jsonschema click

# Запуск по конфигурационному файлу
python run.py --config configs/basic_deconvolution.yaml

# Быстрая обработка одного изображения
python cli.py process --input image.jpg --algorithm vabid

# Интерактивный режим
python cli.py interactive
```

---

## Структура проекта

Подробнее об алгоритмах: [Путеводитель по алгоритмам](src/blinddeconv/algorithms/README.md)

```
blind_deconvolution/
├── src/                               # Исходники Python-пакета
│   └── blinddeconv/                   # Python-пакет `blinddeconv`
│       ├── algorithms/                # Алгоритмы и обёртки
│       │   ├── base.py                # DeconvolutionAlgorithm
│       │   ├── blind_deconvolution/
│       │   │   ├── our_company/       # Собственные реализации
│       │   │   │   ├── bayesian/
│       │   │   │   ├── classic/
│       │   │   │   ├── sparse/
│       │   │   │   └── variational/
│       │   │   └── third_party_company/  # Внешние реализации
│       │   │       └── ...
│       │   ├── kernel_estimation/
│       │   │   └── ...
│       │   ├── nonblind_deconvolution/
│       │   │   └── ...
│       │   ├── octave/                # Octave/Matlab-обвязка
│       │   │   └── ...
│       │   ├── unsorted/              # Экспериментальные алгоритмы
│       │   │   └── ...
│       │   ├── README.md              # Путеводитель по алгоритмам
│       │   └── __init__.py
│       ├── filters/                   # Генерация искажений (blur/noise/denoise)
│       │   ├── blur.py
│       │   ├── noise.py
│       │   ├── denoise.py
│       │   ├── distributions.py
│       │   ├── colored_noise.py
│       │   ├── smooth.py
│       │   └── __init__.py
│       ├── processing/                # Основной функционал пайплайна
│       │   ├── core.py
│       │   ├── reader.py
│       │   ├── restorepipeline.py
│       │   ├── metrics.py
│       │   ├── tables.py
│       │   ├── utils.py
│       │   └── ...
│       ├── scripts/                   # Вспомогательные скрипты
│       │   ├── dataset_generator.py
│       │   ├── kernel_generator.py
│       │   └── __init__.py
│       └── __init__.py
│
├── configs/                           # Конфигурационные файлы
│   ├── basic_deconvolution.yaml       # Базовый пример
│   ├── medical_imaging.yaml           # Для медицинских изображений
│   ├── satellite_images.yaml          # Для спутниковых снимков
│   └── experiment_template.json       # Полный шаблон (JSON)
│
├── run.py                             # Автоматизация по конфигам
├── cli.py                             # CLI-интерфейс
│
├── docs/                              # Документация (Sphinx + Markdown)
│   ├── source/
│   │   ├── conf.py
│   │   ├── index.rst
│   │   ├── installation.md
│   │   ├── usage_guide.md
│   │   ├── configuration.md
│   │   ├── architecture.md
│   │   ├── data_flow.md
│   │   ├── api_reference.md
│   │   ├── CONTRIBUTING.md
│   │   ├── CHANGELOG.md
│   │   └── theory/
│   └── tools/
│       └── build_docs.py
├── images/                            # Примеры изображений/артефакты
│   ├── dataset_bind.json
│   ├── distorted/
│   │   └── ...
│   └── ...
├── references/                        # PDF-материалы/статьи
│   └── *.pdf
├── tests/                             # Тестовые данные/выходы прогонов
│   └── ...
├── utils/                             # Вспомогательные утилиты
│   └── preflight/
│       ├── __main__.py
│       ├── config.py
│       ├── report.py
│       └── checks/
│           ├── python.py
│           └── packages.py
│
├── requirements.txt
├── pyproject.toml
├── setup.cfg
└── README.md
```

---

## Подробная документация

Полная документация доступна в `docs/source/` (читается на GitHub, собирается через Sphinx):

| Раздел | Описание |
|--------|----------|
| [Установка и настройка](docs/source/installation.md) | Все способы установки, профили зависимостей, интерактивный установщик, виртуальное окружение |
| [Руководство пользователя](docs/source/usage_guide.md) | `run.py`, `cli.py`, все команды и аргументы, примеры Python-кода |
| [Конфигурационные файлы](docs/source/configuration.md) | Полная структура YAML/JSON-конфигов, валидация, реестры алгоритмов и фильтров |
| [Архитектура системы](docs/source/architecture.md) | Компоненты, паттерны проектирования, организация модулей |
| [Поток данных](docs/source/data_flow.md) | Схемы `process` и `full_process`, формат `dataset.json` |
| [API Reference](docs/source/api_reference.md) | Полная справка по классам, методам и функциям |
| [Для разработчиков](docs/source/CONTRIBUTING.md) | Стандарты кода, добавление алгоритмов/фильтров, Git-конвенции, сборка документации, линтинг |
| [История изменений](docs/source/CHANGELOG.md) | Версии и список изменений |

Собранная HTML-документация: [pavelyurov.github.io/blind_deconvolution](https://pavelyurov.github.io/blind_deconvolution/)
