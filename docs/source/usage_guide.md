# Руководство пользователя

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

## run.py — Автоматизация через конфигурационные файлы

Главный скрипт для автоматического запуска пайплайна. Пользователь создаёт конфиг один раз, а потом запускает одной командой. Идеален для воспроизводимых экспериментов.

### Базовый запуск

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

### Аргументы командной строки

| Аргумент | Короткий | Описание |
|---|---|---|
| `--config` | `-c` | Путь к конфигу (обязательный) |
| `--output-dir` | `-o` | Переопределение директории результатов |
| `--generate-report` | — | Генерировать LaTeX-отчёт |
| `--dry-run` | — | Проверка без выполнения |
| `--validate-only` | — | Только валидация конфига |
| `--verbose` | `-v` | Подробный вывод (DEBUG) |

### Структура конфигурационного файла (YAML)

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

Подробнее о каждой секции конфигурации — в разделе [Конфигурационные файлы](configuration.md).

### Доступные алгоритмы (реестр)

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

### Доступные фильтры

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

### PSF-функции (для defocus_blur и motion_blur)

| Имя | Описание |
|-----|----------|
| `gaussian` | Гауссовское распределение |
| `uniform` | Равномерное (диск / прямоугольник) |
| `linear_decay` | Линейно убывающее (конус / треугольник) |
| `ring` | Кольцевое распределение |
| `exponential_decay` | Экспоненциально убывающее |

### Готовые конфиги

В папке `configs/` находятся готовые примеры:

- `basic_deconvolution.yaml` — базовый пример с режимом `process`
- `medical_imaging.yaml` — обработка медицинских изображений (режим `full_process`)
- `satellite_images.yaml` — обработка спутниковых снимков (режим `full_process`)
- `experiment_template.json` — полный шаблон в JSON со всеми полями

### Что генерирует run.py

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

## cli.py — Интерфейс командной строки

CLI-интерфейс для быстрых операций без конфигурационных файлов.

### Справка

```bash
python cli.py --help
python cli.py <команда> --help
```

### Команды

#### 1. `process` — Быстрая обработка одного изображения

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

#### 2. `run` — Запуск по конфигурации (аналог run.py)

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

#### 3. `generate-config` — Генерация конфига из шаблона

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

#### 4. `view-config` — Просмотр конфига в табличном виде

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

#### 5. `interactive` — Интерактивный режим для новичков

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

#### 6. `list-algorithms` — Список доступных алгоритмов

```bash
# Табличный вид
python cli.py list-algorithms

# В формате LaTeX
python cli.py list-algorithms --format latex
```

#### 7. `list-filters` — Список доступных фильтров

```bash
python cli.py list-filters
python cli.py list-filters --format latex
```

### Автодополнение команд (Tab completion)

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

## Сборка документации (`docs/tools/build_docs.py`)

Утилита `build_docs.py` автоматизирует генерацию HTML-документации из исходного кода и Markdown-файлов.

### Требования

```bash
pip install ".[docs]"
```

### Запуск

```bash
python docs/tools/build_docs.py
```

Скрипт выполняет два шага:
1. **Генерация API** — вызывает `sphinx-apidoc --separate` для создания `.rst` файлов из docstrings модулей в `src/`. Все существующие `.rst` (кроме `index.rst`) удаляются и пересоздаются.
2. **Сборка HTML** — вызывает `sphinx-build -b html` для компиляции документации в `docs/_build/html/`.

Результат: `docs/_build/html/index.html`.

### Ручная сборка

Если нужен контроль над отдельными шагами:

```bash
sphinx-apidoc --separate -o docs/source src/
sphinx-build -b html docs/source docs/_build/html
```

### Важно

- Скрипт удаляет и пересоздаёт все `.rst` файлы в `docs/source/` (кроме `index.rst`) — не храните ручные правки в автогенерируемых `.rst`.
- Markdown-файлы (`.md`) в `docs/source/` включаются через MyST-Parser автоматически и не затрагиваются.

---

## Использование как Python-библиотеки

### Восстановление одного изображения

```python
from blinddeconv.processing import Processing

proc = Processing(
    images_folder="images/original",
    blurred_folder="blurred",
    restored_folder="restored",
    color=False,
)

# Связывание оригинала с искажённым
proc.bind(
    original_image_path="images/original/airplane.png",
    blurred_image_path="images/distorted/airplane_blurred.png",
    original_kernel_path="images/kernel_data/kernel.npy",
    filter_description="convolved_kernel",
)

# Восстановление
from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID

algo = VABID(max_iter=100, kernel_size=21)
proc.process(algorithm_processor=algo, metadata=True)

# Визуализация
proc.show()
```

### Сравнение нескольких алгоритмов

```python
proc = Processing(color=False)
proc.bind("original.png", "blurred.png", "kernel.npy", "gaussian_blur")

from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID
from blinddeconv.algorithms.blind_deconvolution.our_company.classic.richardson_lucy import RichardsonLucy

for algo in [VABID(max_iter=100, kernel_size=21),
             RichardsonLucy(max_iter=50, kernel_size=15)]:
    proc.process(algo, metadata=True)

proc.show()
proc.get_table("comparison.csv", display_table=True)
```

### Полный пайплайн (фильтры + восстановление)

```python
from blinddeconv.processing import Processing
from blinddeconv.filters.blur import DefocusBlur
from blinddeconv.filters.distributions import gaussian_distribution
from blinddeconv.filters.noise import GaussianNoise

proc = Processing(images_folder="images/original", color=False)
proc.read_all()

filters = [
    [DefocusBlur(psf=gaussian_distribution, param=5.0),
     GaussianNoise(param=10.0)],
    [DefocusBlur(psf=gaussian_distribution, param=3.0),
     GaussianNoise(param=5.0)],
]

methods = [algo1, algo2]
proc.full_process(filters=filters, methods=methods)
```

### Оптимизация гиперпараметров

```python
algo = VABID(max_iter=100, kernel_size=21)

result = proc.process_hyperparameter_optimization(
    algorithm_processor=algo,
    param_ranges={"max_iter": (50, 500), "kernel_size": (11, 31)},
    n_trials=50,
    metric="PSNR",
)

print(f"Лучшие параметры: {result.best_params}")
print(f"Лучший PSNR: {result.best_value:.2f} dB")
```

### Сохранение и загрузка состояния

```python
# Сохранение
proc.save_bind_state("dataset/my_experiment.json")

# Загрузка в другом сеансе
proc2 = Processing(color=False)
proc2.load_bind_state("dataset/my_experiment.json")
proc2.show()
```

### Очистка данных

```python
proc.clear_restored()          # Удалить восстановленные
proc.unbind_restored()         # Убрать связи с восстановленными
proc.clear_output()            # Удалить файлы + сброс
proc.clear_all()               # Полная очистка
proc.clear_output_directory()  # Очистка директорий (с подтверждением)
```
