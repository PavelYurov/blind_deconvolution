# Исследование методов Blind Image Deconvolution

[![CI](https://github.com/PavelYurov/blind_deconvolution/actions/workflows/ci.yml/badge.svg?branch=feature/module-redesign)](https://github.com/PavelYurov/blind_deconvolution/actions/workflows/ci.yml)

## Участники проекта

**Руководитель проекта**  

Парфенов Денис Васильевич, promasterden@yandex.ru

**Тимлид-разработчик**

Беззаборов А.А., КМБО-01-22, antonbezzaborov929@gmail.com  
- Variational Bayesian Inference методы  
- Super Resolution алгоритмы  
- Оптимизация гиперпараметров, визуализация экспериментов и качества методов
- Архитектура фреймворка
- Исследование и координация разработки

**Разработчик (оптимизация и архитектура)**

Юров П.И., КМБО-01-22, pavel.yurov0425@gmail.com  
- Low-Rank, Primal-Dual, Majorization-Minimization методы  
- Noise Estimation, Denoising алгоритмы
- Модели всех видов искажений изображений (noise & blur)
- Проектирование и создание архитектуры фреймворка
- Ускорение и оптимизация методов

**Разработчик (теоретик)**

Куропатов К.Л., КМБО-01-22, konstantinkuropatov@gmail.com  
- Higher-Order, Multiscale методы 
- Теоретическое обоснование методов фреймворка
- Формирование наборов данных разных категорий изображений для экспериментов
- Углубленное исследование имплементированных методов

**Разработчик (эксперименты и интеграция)**  

Малыш Я.В., КМБО-03-22, mrgeroixyu@gmail.com    
- Интеграция third-party алгоритмов во фреймворк
- Разработка Wrapper-ов для Open-Source методов Blind Deconvolution
- Проведение экспериментов для third-party методов

## Описание проекта

Данный проект посвящен исследованию методов Blind Image Deconvolution - восстановления изображений без априорной информации о функции искажения PSF с интегрированной системой автоматической оптимизации гиперпараметров. Основное внимание уделяется разработке и сравнению алгоритмов, способных реконструировать исходное изображение в условиях высокой неопределенности относительно характера искажений.

### Цель исследования

Реализация и многокритериальный анализ предельных возможностей методов восстановления изображений, искаженных различными типами размытия и шумов. В рамках исследования предполагается: систематическая оценка качества восстановления, автоматический подбор оптимальных гиперпараметров, сопоставление разработанных алгоритмов с современными подходами.

### Основные задачи
- **Исследование современных методов Blind Deconvolution, PSF estimation, Super-Resolution**
- **Разработка и имплементация алгоритмов восстановления изображений:** байесовские; вариационные и отимизационные; тензорные
- **Создание интегрированной системы автоматической оптимизации гиперпараметров и многокритериального анализа (multiobjectivization):** применение байесовской оптимизации (GP, TPE), эволюционные алгоритмов (NSGA‑II), случайного поиска; построение многомерных Парето‑фронтов для визуализации компромиссов между качеством и устойчивостью к искажениям.
- **Сравнительный анализ с современными подходами** — оценка точности восстановления, качества оценки PSF и вычислительной эффективности разработанных методов в сопоставлении с state‑of‑the‑art алгоритмами 

## Функциональность фреймворка

### Обработка изображений
- Поддержка монохромных и цветных изображений в форматах JPEG, BMP, PNG (возможность добавления RAW)
- Пакетная обработка групп изображений с единым конвейером
- Настройка экспериментов через конфигурационные файлы (YAML/JSON)

### Генерация реалистичных искажений
- **Типы размытия:**
  - Расфокус (defocus) — 2D аппаратная функция с центральной симметрией
  - Смаз от движения (motion blur)  — 1D ядро вдоль заданного направления (колоколообразная форма)
  - Комбинированные траектории — смазы по B‑сплайновым кривым
- **Типы шумов:**
  - Гауссов шум (белый/цветной)
  - Пуассонов шум
  - Импульсный шум (salt & pepper)
- Возможность последовательного наложения нескольких искажений с контролируемой энергией относительно исходного изображения

### Метрики качества
- **PSNR, SSIM** — сравнение восстановленного изображения с оригиналом
- Оценка устойчивости к шумам
- Измерение времени выполнения алгоритмов

### Оптимизация гиперпараметров
- **Байесовская оптимизация**: Tree‑Structured Parzen Estimator (TPE), Gaussian Processes (GP)
- **Эволюционные алгоритмы**: NSGA‑II, генетические алгоритмы
- **Случайный поиск** с адаптивным распределением

### Визуализация
- **Построение многомерных Парето-фронтов** (качество, сложность смаза, уровень шума)

## Методы восстановления
- **Классические методы**: LPA‑ICI, L0/L1‑регуляризация
- **Байесовские методы**:
  - Робастные модели с тяжёлыми хвостами (Student‑t, Poisson‑Square Cauchy)
  - Разреженный вариационный байес (Scale Mixtures of Gaussians)
  - TV‑приоры и структурированная аппроксимация ковариации
- **Вариационные и оптимизационные методы**:
  - Сглаживающие априоры (Relative Total Variation)
  - Методы дробного порядка (Fractional‑Order Variation)
  - L0‑регуляризация интенсивностей и градиентов
- **Тензорные методы**:
  - Нелокальная низкоранговая тензорная аппроксимация (t‑SVD, WNNM)
  - Многоуровневое иерархическое разложение (MHDM)           

Подробнее об алгоритмах: [Путеводитель по алгоритмам](src/blinddeconv/algorithms/README.md)

## Техническое описание

### 1. Клонирование репозитория и настройка окружения

```bash
git clone https://github.com/PavelYurov/blind_deconvolution.git
cd blind_deconvolution
python -m venv .venv
```
Windows PowerShell
```bash
.venv\Scripts\Activate.ps1
```
Linux / macOS
```bash
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

