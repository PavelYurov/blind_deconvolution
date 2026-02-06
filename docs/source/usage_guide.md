# Руководство пользователя

## Быстрый старт

### Использование как Python-библиотеки

```python
from blinddeconv.processing import Processing

# 1. Создаём экземпляр фреймворка
proc = Processing(
    images_folder="images/original",     # Папка с оригиналами
    blurred_folder="blurred",            # Папка для смазанных
    restored_folder="restored",          # Папка для результатов
    color=False,                         # Ч/Б режим
)

# 2. Связываем оригинал с искажённым изображением
proc.bind(
    original_image_path="images/original/airplane.png",
    blurred_image_path="images/distorted/airplane_blurred.png",
    original_kernel_path="images/kernel_data/kernel.npy",
    filter_description="convolved_kernel",
)

# 3. Создаём алгоритм и запускаем восстановление
from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID

algo = VABID(max_iter=100, kernel_size=21)
proc.process(algorithm_processor=algo, metadata=True)

# 4. Визуализация результатов
proc.show()
```

### Использование через run.py (автоматизация)

```bash
# Запуск по конфигу
python run.py --config configs/basic_deconvolution.yaml

# С генерацией LaTeX-отчёта
python run.py --config configs/experiment.yaml --generate-report

# Dry-run (проверка плана без обработки)
python run.py --config configs/experiment.yaml --dry-run
```

### Использование через cli.py (быстрые команды)

```bash
# Быстрая обработка одного изображения
python cli.py process --input image.jpg --algorithm vabid

# Список доступных алгоритмов
python cli.py list-algorithms

# Интерактивный режим
python cli.py interactive
```

---

## Сценарии использования

### Сценарий 1: Восстановление одного изображения

Самый простой случай: есть смазанное изображение, нужно восстановить.

```python
from blinddeconv.processing import Processing

proc = Processing(color=False)

# Связываем: оригинал нужен для расчёта метрик
proc.bind("original.png", "blurred.png")

# Восстановление
from blinddeconv.algorithms.blind_deconvolution.our_company.classic.richardson_lucy import RichardsonLucy

algo = RichardsonLucy(max_iter=50, kernel_size=15)
proc.process(algo, metadata=True)

# Результаты в restored/
proc.show()
```

Или через CLI:

```bash
python cli.py process -i blurred.png -a richardson_lucy -p '{"max_iter": 50}'
```

### Сценарий 2: Сравнение нескольких алгоритмов

```python
from blinddeconv.processing import Processing

proc = Processing(color=False)
proc.bind("original.png", "blurred.png", "kernel.npy", "gaussian_blur")

# Последовательно применяем разные алгоритмы
from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID
from blinddeconv.algorithms.blind_deconvolution.our_company.classic.richardson_lucy import RichardsonLucy
from blinddeconv.algorithms.blind_deconvolution.our_company.bayesian.vbbid_tv import VBBID_TV

for algo in [VABID(max_iter=100, kernel_size=21),
             RichardsonLucy(max_iter=50, kernel_size=15),
             VBBID_TV(max_iter=150, kernel_size=21)]:
    proc.process(algo, metadata=True)

# Визуализация: все алгоритмы в одной таблице
proc.show()

# Экспорт метрик
proc.get_table("comparison.csv", display_table=True)
```

### Сценарий 3: Полный пайплайн (фильтры → восстановление)

```python
from blinddeconv.processing import Processing
from blinddeconv.filters.blur import DefocusBlur
from blinddeconv.filters.distributions import gaussian_distribution, ring_distribution
from blinddeconv.filters.noise import GaussianNoise

proc = Processing(images_folder="images/original", color=False)
proc.read_all()

# Цепочки фильтров
filters = [
    [DefocusBlur(psf=gaussian_distribution, param=5.0),
     GaussianNoise(param=10.0)],
    [DefocusBlur(psf=ring_distribution, param=3.0),
     GaussianNoise(param=5.0)],
]

methods = [algo1, algo2]

# Запуск
proc.full_process(filters=filters, methods=methods)
```

### Сценарий 4: Оптимизация гиперпараметров

```python
proc = Processing(color=False)
proc.bind("original.png", "blurred.png")

algo = VABID(max_iter=100, kernel_size=21)

# Определяем пространство поиска
param_ranges = {
    "max_iter": (50, 500),
    "kernel_size": (11, 31),
}

# Запуск оптимизации (Optuna)
result = proc.process_hyperparameter_optimization(
    algorithm_processor=algo,
    param_ranges=param_ranges,
    n_trials=50,
    metric="PSNR",
    method="tpe",
)

print(f"Лучшие параметры: {result.best_params}")
print(f"Лучший PSNR: {result.best_value:.2f} dB")
```

### Сценарий 5: Пакетная обработка с конфигом

Создайте `my_experiment.yaml`:

```yaml
experiment:
  name: "Batch Processing"

input:
  images_folder: "images/original"
  color: false
  load_mode: "all"

output:
  restored_folder: "results/batch/restored"
  data_folder: "results/batch/data"

processing:
  mode: "full_process"

filters:
  - chain:
      - type: "defocus_blur"
        params: {psf: "gaussian", param: 3.0}
      - type: "gaussian_noise"
        params: {param: 5.0}

algorithms:
  - name: "vabid"
    params: {max_iter: 200, kernel_size: 21}
  - name: "richardson_lucy"
    params: {max_iter: 100, kernel_size: 15}

report:
  generate: true
  output_path: "results/batch/report.tex"
```

```bash
python run.py --config my_experiment.yaml --generate-report
```

### Сценарий 6: Предобработка с выравниванием гистограмм

```python
proc = Processing(color=False)
proc.bind("original.png", "blurred.png")

# Выравнивание гистограмм перед восстановлением
proc.histogram_equalization_CLAHE(clip_limit=0.01, view_histogram=True)

# Восстановление
proc.process(algo, metadata=True)

# Обратная гистограммная коррекция
proc.inverse_histogram_equalization()

# Визуализация
proc.show()
```

### Сценарий 7: Генерация датасета

```python
from blinddeconv.processing import Processing
from blinddeconv.scripts.dataset_generator import DatasetGenerator

proc = Processing(color=False)

generator = DatasetGenerator(
    processing_instance=proc,
    input_dir="images/original",
    output_dir="images/distorted",
    kernel_dir="images/ground_truth_filters",
    kernel_data_dir="images/kernel_data",
)

# Генерация датасета с различными комбинациями искажений
generator.generate()
```

### Сценарий 8: Сохранение и загрузка состояния

```python
# Сохранение после обработки
proc.save_bind_state("dataset/my_experiment.json")

# Позже: загрузка и продолжение
proc2 = Processing(color=False)
proc2.load_bind_state("dataset/my_experiment.json")
proc2.show()  # Все данные восстановлены
```

---

## Команды cli.py

### `process` — быстрая обработка

```bash
python cli.py process --input image.jpg --algorithm vabid
python cli.py process -i orig.png -b blur.png -a vabid -p '{"max_iter":100}'
python cli.py process -i img.png -a richardson_lucy --color -o results/
```

### `run` — запуск по конфигу

```bash
python cli.py run --config configs/experiment.yaml
python cli.py run -c configs/experiment.yaml --dry-run
python cli.py run -c configs/experiment.yaml --validate-only
```

### `generate-config` — генерация конфига

```bash
python cli.py generate-config --template basic --output my_config.yaml
python cli.py generate-config -t medical -f json -o config.json
python cli.py generate-config -t empty
```

Шаблоны: `basic`, `medical`, `satellite`, `empty`.

### `view-config` — просмотр конфига

```bash
python cli.py view-config configs/experiment.yaml
python cli.py view-config configs/experiment.yaml --format latex
python cli.py view-config configs/experiment.yaml --format json
```

### `interactive` — пошаговый мастер

```bash
python cli.py interactive
```

### `list-algorithms` / `list-filters`

```bash
python cli.py list-algorithms
python cli.py list-filters
python cli.py list-algorithms --format latex
```

---

## Аргументы run.py

| Аргумент | Короткий | Описание |
|---|---|---|
| `--config` | `-c` | Путь к конфигу (обязательный) |
| `--output-dir` | `-o` | Переопределение директории результатов |
| `--generate-report` | — | Генерировать LaTeX-отчёт |
| `--dry-run` | — | Проверка без выполнения |
| `--validate-only` | — | Только валидация конфига |
| `--verbose` | `-v` | Подробный вывод (DEBUG) |

---

## Очистка данных

```python
# Удаление восстановленных (файлы + связи)
proc.clear_restored()

# Удаление только связей (файлы остаются)
proc.unbind_restored()

# Сброс до оригиналов (восстановленные + смазанные удалены)
proc.clear_output()

# Полная очистка
proc.clear_all()

# Очистка выходных директорий (с подтверждением)
proc.clear_output_directory()
```
