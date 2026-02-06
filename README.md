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

## Функциональность фреймворка

### Обработка изображений

- Поддержка монохромных и цветных изображений (JPEG, BMP, PNG)
- Пакетная обработка групп изображений
- Автоматизация экспериментального конвейера

### Генерация искажений

- **Типы размытия**:
  - Расфокус (2D гауссово ядро)
  - Motion blur (1D линейное ядро)
  - Комбинированные смазы (B-spline траектории)
- **Типы шумов**:
  - Гауссов шум
  - Пуассонов шум
  - Импульсный шум (salt & pepper)

### Методы восстановления

- **Классические алгоритмы** (blind deconvolution) и регуляризационные подходы
- **Оценка ядра размытия** (kernel estimation): слепые методы оценки PSF
- **Non-blind deconvolution**: восстановление с известным ядром

## Система оценки

- **Метрики качества**: PSNR, SSIM
- **Производительность**: время выполнения

## Автоматическая оптимизация гиперпараметров

### Методы оптимизации

- **Байесовская оптимизация** c Gаussіаn Рrосеssеs
- **Эвoлюциoнныe aлгopитмы**  (Gеnеtіс Аlgоrіthms)
- **Случайный поиск** с адаптивным распределением

### Оптимизируемые параметры

Для каждого алгоритма определено пространство поиска гиперпараметров:
- **Регуляризационные параметры**
- **Количество итераций** и пороги сходимости
- **Размеры ядер** размытия
- **Параметры шумоподавления**

## Визуализация

### Многомерные Парето-фронты

- **3D визуализация**: качество, сложность смаза, уровень шума
- **Сравнительный анализ** производительности алгоритмов
- **Анализ чувствительности** параметров к различным типам искажений

## Установка

1. Установить последнюю версию пакета:

```bash
pip install git+https://github.com/PavelYurov/blind_deconvolution.git
```

Будут установлен пакет `blinddeconv` и остальные зависимости (`numpy`, `scipy`, `opencv-python`, `scikit-image`, и др.)

```py
import blinddeconv
```

2. Установка с дополнительными зависимостями:

Включает инструменты для тестирования, линтинга и интерактивной работы (`pytest`, `flake8`, `ipython`, `setuptools`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[dev]"
```

Включает CLI-интерфейс и автоматизацию (`click`, `pyyaml`, `jsonschema`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[cli]"
```

Необходимо для генерации документации через Sphinx (`sphinx`, `sphinx-rtd-theme`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[docs]"
```

Всё вместе:

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[cli,dev,docs]"
```

## Удаление

```bash
pip uninstall blinddeconv
```

---

## Автоматизация и CLI

Проект предоставляет два интерфейса для автоматизации запуска экспериментов, которые делают работу с фреймворком удобнее и приближают его к полноценной библиотеке.

| Интерфейс | Назначение | Когда использовать |
|-----------|------------|-------------------|
| `run.py` | Запуск по конфигурационному файлу | Воспроизводимые эксперименты, автоматизация |
| `cli.py` | Быстрые команды из терминала | Тестирование, быстрые операции, генерация конфигов |

### Установка зависимостей CLI

```bash
pip install pyyaml jsonschema click
```

Или через optional-зависимости:

```bash
pip install -e ".[cli]"
```

---

### run.py — Автоматизация через конфигурационные файлы

Главный скрипт для автоматического запуска пайплайна. Пользователь создаёт конфиг один раз, а потом запускает одной командой. Идеален для воспроизводимых экспериментов.

#### Базовый запуск

```bash
# Запуск по конфигу
python run.py --config configs/basic_deconvolution.yaml

# С переопределением директории результатов
python run.py --config configs/experiment.yaml --output-dir my_results/

# С генерацией LaTeX-отчёта
python run.py --config configs/experiment.yaml --generate-report

# Dry-run: проверка без выполнения (показывает план)
python run.py --config configs/experiment.yaml --dry-run

# Только валидация конфигурации
python run.py --config configs/experiment.yaml --validate-only

# Подробный вывод (уровень DEBUG)
python run.py --config configs/experiment.yaml --verbose
```

#### Структура конфигурационного файла (YAML)

```yaml
# Метаданные эксперимента
experiment:
  name: "My Experiment"
  description: "Описание эксперимента"

# Входные данные
input:
  images_folder: "images/original"      # Папка с оригинальными изображениями
  blurred_folder: "images/distorted"    # Папка со смазанными (для mode: process)
  color: false                          # true — цветные, false — ч/б

  # Режим загрузки:
  #   "all"        — все изображения из images_folder (для full_process)
  #   "bind"       — конкретные пары оригинал + смазанное
  #   "bind_state" — загрузка из ранее сохранённого JSON
  load_mode: "bind"

  # Связи (для load_mode: "bind")
  bindings:
    - original: "images/original/airplane.png"
      blurred: "images/distorted/airplane_blurred.png"
      kernel: "images/kernel_data/kernel.npy"       # опционально
      filter_description: "gaussian_blur"            # опционально

  # Путь к JSON (для load_mode: "bind_state")
  # bind_state_path: "dataset/dataset.json"

# Выходные директории
output:
  restored_folder: "results/restored"
  data_folder: "results/data"
  kernel_folder: "results/kernels"

# Режим обработки
processing:
  # "process"      — восстановление по связям (оригинал + смазанное)
  # "full_process" — полный пайплайн: фильтры → восстановление → анализ
  mode: "process"
  metadata: true        # Сохранять метаданные
  unique_paths: true    # Генерировать уникальные пути

# Алгоритмы (минимум один обязателен)
algorithms:
  - name: "vabid"                # Краткое имя из реестра
    params:
      max_iter: 100
      kernel_size: 21

  - name: "custom"               # Пользовательский алгоритм
    module: "my_module.my_algo"  # Полный путь к модулю
    class_name: "MyAlgorithm"    # Имя класса
    params:
      param1: value1

# Цепочки фильтров (для mode: "full_process")
filters:
  - chain:
      - type: "defocus_blur"
        params:
          psf: "gaussian"        # PSF-функция: gaussian, uniform, ring и др.
          param: 5.0
      - type: "gaussian_noise"
        params:
          param: 10.0

# Генерация отчёта
report:
  generate: false
  format: "latex"
  output_path: "results/report.tex"
```

#### Доступные алгоритмы (реестр)

| Имя | Категория | Описание |
|-----|-----------|----------|
| `richardson_lucy` | classic | Алгоритм Richardson-Lucy |
| `em` | classic | EM-алгоритм для слепой деконволюции |
| `map` | classic | MAP с регуляризацией (alternating minimization) |
| `vbbid_tv` | bayesian | Вариационная байесовская с TV априори |
| `bbd_deip` | bayesian | Байесовская с разными экспозициями |
| `sb_bid_pe` | bayesian | Разреженная байесовская деконволюция |
| `vapibe` | variational | Вариационный подход к оценке параметров |
| `vabid` | variational | Вариационный байесовский подход (Likas2004) |
| `vbsk_sid_st` | variational | Вариационная со Student's-t |
| `vbc_bid` | sparse | Компрессивная байесовская деконволюция |

#### Доступные фильтры

| Имя | Описание | Требует PSF |
|-----|----------|-------------|
| `defocus_blur` | Размытие вне фокуса (2D) | Да |
| `motion_blur` | Размытие в движении (1D) | Да |
| `bspline_blur` | Криволинейное размытие (B-spline) | Нет |
| `kernel_convolution` | Свёртка с ядром из .npy файла | Нет |
| `gaussian_noise` | Аддитивный гауссовский шум | Нет |
| `poisson_noise` | Пуассоновский шум | Нет |
| `salt_pepper_noise` | Импульсный шум (соль и перец) | Нет |
| `mean_blur` | Сглаживание средним | Нет |
| `median_blur` | Медианное сглаживание | Нет |
| `gaussian_blur` | Гауссово сглаживание | Нет |
| `bilateral_filter` | Билатеральный фильтр | Нет |

#### PSF-функции (для defocus_blur и motion_blur)

| Имя | Описание |
|-----|----------|
| `gaussian` | Гауссовское распределение |
| `uniform` | Равномерное (диск / прямоугольник) |
| `linear_decay` | Линейно убывающее (конус / треугольник) |
| `ring` | Кольцевое распределение |
| `exponential_decay` | Экспоненциально убывающее |

#### Готовые конфиги

В папке `configs/` находятся готовые примеры:

- `basic_deconvolution.yaml` — базовый пример с режимом `process`
- `medical_imaging.yaml` — обработка медицинских изображений (режим `full_process`)
- `satellite_images.yaml` — обработка спутниковых снимков (режим `full_process`)
- `experiment_template.json` — полный шаблон в JSON со всеми полями

#### Что генерирует run.py

После запуска в директории результатов создаются:

```
results/
├── restored/           # Восстановленные изображения
├── data/               # CSV-таблицы с метриками (PSNR, SSIM)
├── kernels/            # Восстановленные ядра размытия
├── metadata.json       # Метаданные эксперимента (конфиг, время, платформа)
└── report.tex          # LaTeX-отчёт (при --generate-report)
```

---

### cli.py — Интерфейс командной строки

CLI-интерфейс для быстрых операций без конфигурационных файлов.

#### Справка

```bash
python cli.py --help
python cli.py <команда> --help
```

#### Команды

##### 1. `process` — Быстрая обработка одного изображения

```bash
# Обработка изображения алгоритмом VABID
python cli.py process --input image.jpg --algorithm vabid

# С указанием смазанного изображения и параметров
python cli.py process \
  --input original.png \
  --blurred blurred.png \
  --algorithm richardson_lucy \
  --params '{"max_iter": 50, "kernel_size": 15}'

# Цветной режим с указанием выходной директории
python cli.py process -i photo.png -a vabid --color -o my_results/

# С ядром размытия
python cli.py process -i orig.png -b blur.png -k kernel.npy -a vabid
```

##### 2. `run` — Запуск по конфигурации (аналог run.py)

```bash
# Запуск по конфигу
python cli.py run --config configs/experiment.yaml

# Dry-run (проверка без выполнения)
python cli.py run -c configs/experiment.yaml --dry-run

# Только валидация
python cli.py run -c configs/experiment.yaml --validate-only

# С генерацией отчёта
python cli.py run -c configs/experiment.yaml --generate-report
```

##### 3. `generate-config` — Генерация конфига из шаблона

```bash
# Базовый шаблон (YAML)
python cli.py generate-config --template basic --output my_config.yaml

# Медицинский шаблон (JSON)
python cli.py generate-config -t medical -f json -o config.json

# Спутниковый шаблон
python cli.py generate-config -t satellite -o satellite.yaml

# Пустой шаблон (для заполнения вручную)
python cli.py generate-config -t empty
```

Доступные шаблоны: `basic`, `medical`, `satellite`, `empty`.

##### 4. `view-config` — Просмотр конфига в табличном виде

```bash
# Табличный вид (по умолчанию)
python cli.py view-config configs/experiment.yaml

# В формате LaTeX
python cli.py view-config configs/experiment.yaml --format latex

# В формате JSON
python cli.py view-config configs/experiment.yaml --format json

# В формате YAML
python cli.py view-config configs/experiment.yaml --format yaml
```

##### 5. `interactive` — Интерактивный режим для новичков

```bash
python cli.py interactive
```

Режим пошагово задаёт вопросы:
1. Название и описание эксперимента
2. Входные данные (папка, режим загрузки, связи)
3. Выбор алгоритмов из списка
4. Настройка фильтров (для full_process)
5. Параметры вывода
6. Предпросмотр конфигурации
7. Действие: запустить / сохранить / оба / отменить

##### 6. `list-algorithms` — Список доступных алгоритмов

```bash
# Табличный вид
python cli.py list-algorithms

# В формате LaTeX
python cli.py list-algorithms --format latex
```

##### 7. `list-filters` — Список доступных фильтров

```bash
python cli.py list-filters
python cli.py list-filters --format latex
```

#### Автодополнение команд (Tab completion)

Для включения автодополнения в оболочке:

**Bash:**
```bash
eval "$(_CLI_COMPLETE=bash_source python cli.py)"
```

**Zsh:**
```bash
eval "$(_CLI_COMPLETE=zsh_source python cli.py)"
```

**Fish:**
```bash
_CLI_COMPLETE=fish_source python cli.py | source
```

Для постоянного автодополнения добавьте соответствующую строку в `.bashrc`, `.zshrc` или `config.fish`.

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
├── docs/                              # Документация (Sphinx)
│   ├── source/
│   │   ├── conf.py
│   │   └── index.rst
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

## Качество кода

В проекте настроен линтер **flake8** для проверки стиля и качества Python-кода. Конфигурация находится в файле `setup.cfg`.

**Локальная работа:**
1.  Установите зависимости: `pip install flake8 pre-commit`
2.  Для ручного запуска проверки выполните в корне проекта: `flake8`
3.  Для автоматической проверки перед каждым коммитом выполните: `pre-commit install`

**Интеграция с IDE:**
Настройте свою среду разработки на использование виртуального окружения проекта и чтение `setup.cfg`, чтобы видеть предупреждения линтера прямо в редакторе.

**На сервере (CI):**
При каждом push в репозиторий CI-система запускает те же проверки `flake8`, чтобы поддерживать основную ветку (`main`/`master`) в чистоте.

## Документация

HTML-документация проекта генерируется с помощью **Sphinx** и публикуется [по ссылке](https://pavelyurov.github.io/blind_deconvolution/)

### Структура документации

- `docs/source/` — исходники Sphinx (`.rst`, `conf.py`)
- `docs/tools/build_docs.py` — скрипт сборки (генерация API + сборка HTML)
- `docs/_build/html/` — результат локальной сборки (появляется после первого билда)

Документация в основном строится из docstrings Python-модулей (автодокументация).

### Требования

В активном окружении Python должны быть доступны утилиты:
- `sphinx-build`
- `sphinx-apidoc`

Опционально: тема `sphinx_rtd_theme` (если не установлена, используется fallback `alabaster`).

### Локальная сборка (рекомендуемый способ)

```bash
python docs/tools/build_docs.py
```

Скрипт делает два шага:
1) генерирует `.rst` для API (через `sphinx-apidoc`) в `docs/source/`;
2) собирает HTML в `docs/_build/html/`.

Открывайте результат: `docs/_build/html/index.html`.

### Ручная сборка (если нужен контроль шагов)

Из корня репозитория:
```bash
sphinx-apidoc -o docs/source .
sphinx-build -b html docs/source docs/_build/html
```

### Полезные замечания

- `docs/tools/build_docs.py` удаляет все `docs/source/*.rst`, кроме `index.rst`, и генерирует заново — не храните важные ручные правки в авто-генерируемых `.rst`.
- Если `sphinx-build`/`sphinx-apidoc` не найдены, установите Sphinx в активное окружение Python (например, в venv проекта).
