# Поток данных (Dataflow) через Pipeline

Все операции с изображениями выполняются через модуль `blinddeconv.processing`. Центральный класс `Processing` предоставляет единый интерфейс, а внутренние модули (`reader`, `applyfilter`, `restore`, `restorepipeline` и др.) реализуют конкретные шаги.

## Общая схема

```
                        Входные изображения (.png, .jpg, .bmp)
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              read_all()        bind()        load_bind_state()
              (все из папки)  (конкретные пары)  (из JSON)
                    │               │               │
                    └───────┬───────┘───────────────┘
                            ▼
                   Processing.images[]
                   (массив объектов Image)
                            │
         ┌──────────────────┼───────────────────────┐
         │                  │                       │
         ▼                  ▼                       ▼
   Режим "full_process"  Режим "process"     Режим "filter"
         │                  │                       │
         ▼                  ▼                       ▼
   filter() + save_filter() │                   filter(kernel)
   (цепочки фильтров)       │                    (одно ядро)
         │                  │                       │
         │                  │                       │             
         │                  │◄──────────────────────┘
         ▼                  ▼                       │
   full_process(filters,    process(algorithm)      │
      methods)              (один алгоритм)         │     
         │                  │                       │     
         │                  │                       │     
         └──────────────────┼───────────────────────┘
                            ▼
               Восстановленные изображения
               + оценённые ядра + метрики
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
       show()     get_table()   save_bind_state()
       pareto()   (CSV-экспорт)  (сохранение связей)
```


## Ключевые функции модуля `processing`

| Функция | Модуль | Назначение |
|---|---|---|
| `bind()` | `reader.py` | Связывает оригинал с искажённым изображением |
| `read_all()` | `reader.py` | Загружает все изображения из директории |
| `read_one()` | `reader.py` | Загружает одно изображение |
| `filter()` | `applyfilter.py` | Применяет один фильтр ко всем изображениям |
| `save_filter()` | `core.py` | Фиксирует текущее смазанное в массив |
| `process()` | `restore.py` | Восстанавливает все изображения одним алгоритмом |
| `full_process()` | `restorepipeline.py` | Полный пайплайн: фильтры → восстановление → анализ |
| `show()` | `display.py` | Визуализация результатов |
| `get_table()` | `tables.py` | Экспорт метрик в CSV |
| `save_bind_state()` | `reader.py` | Сохранение связей в JSON |
| `load_bind_state()` | `reader.py` | Загрузка связей из JSON |

## Режим 1: `process` — восстановление по связям

Используется когда уже есть пары «оригинал + искажённое изображение».

### Шаг 1. Загрузка связей

```python
proc = Processing(images_folder="images/original",
                  blurred_folder="images/distorted",
                  restored_folder="results/restored",
                  color=False)

proc.bind(
    original_image_path="images/original/airplane.png",
    blurred_image_path="images/distorted/airplane_blurred.png",
    original_kernel_path="images/kernels/kernel_airplane.png",
    filter_description="gaussian_blur_sigma5"
)
```

Функция `bind()` принимает пути к `.png` (или другим растровым форматам) файлам оригинального и искажённого изображения. Параметр `original_kernel_path` — опциональный путь к изображению ядра (ground truth PSF), также в формате `.png`.

Внутри `bind()`:
1. Загружаются оригинал и смазанное через `cv2.imread()`
2. Вычисляются метрики деградации (PSNR, SSIM)
3. Создаётся объект `Image` с заполненными атрибутами
4. Объект добавляется в `Processing.images[]`

### Шаг 2. Предобработка (опционально)

```python
proc.histogram_equalization()
proc.histogram_equalization_CLAHE(clip_limit=0.01)
```
Предобработанные изображения сохраняются в preprocess/ и привязываются через preprocessed_blurred_path

### Шаг 3. Восстановление

```python
from blinddeconv.algorithms.blind_deconvolution.our_company.variational.vabid import VABID

algorithm = VABID(max_iter=100, kernel_size=21)
proc.process(algorithm_processor=algorithm, metadata=True)
```

Для каждого изображения в `Processing.images[]`:
1. Загружается смазанное (или предобработанное)
2. Вызывается `algorithm.process(blurred_image)` → `(restored, kernel)`
3. Результаты сохраняются на диск
4. Вычисляются метрики (PSNR, SSIM) между оригиналом и восстановленным
5. Метаданные записываются в JSON (при `metadata=True`)

### Шаг 4. Визуализация и экспорт

```python
# Визуализация: оригинал | смазанное | восстановленные + метрики + ядра
proc.show()

# Экспорт метрик в CSV
proc.get_table("results/metrics.csv", display_table=True)

# Сохранение связей для повторного использования
proc.save_bind_state("results/dataset.json")
```

## Применение фильтров отдельно от `full_process`

Функцию `filter()` (модуль `applyfilter.py`) можно использовать самостоятельно для пошагового построения искажений:

```python
from blinddeconv.filters.blur import DefocusBlur
from blinddeconv.filters.distributions import gaussian_distribution
from blinddeconv.filters.noise import GaussianNoise

proc = Processing(images_folder="images/original", color=False)
proc.read_all()

# Применяем размытие
proc.filter(DefocusBlur(psf=gaussian_distribution, param=5.0))

# Фиксируем результат
proc.save_filter()

# Применяем шум
proc.filter(GaussianNoise(param=10.0))
proc.save_filter()

# Восстанавливаем
proc.process(algorithm, metadata=True)
proc.show()
```

Каждый вызов `filter()` применяет фильтр ко всем загруженным изображениям. Вызов `save_filter()` фиксирует текущее состояние буфера в массив смазанных изображений.

## Режим 2: `full_process` — полный пайплайн

Используется для автоматического создания искажений и последующего восстановления за один вызов:

```python
from blinddeconv.filters.blur import DefocusBlur
from blinddeconv.filters.distributions import gaussian_distribution
from blinddeconv.filters.noise import GaussianNoise

# Загрузка оригиналов
proc = Processing(images_folder="images/original", color=False)
proc.read_all()

# Определение цепочек фильтров
filters = [
     # Цепочка 1: размытие + шум
    [DefocusBlur(psf=gaussian_distribution, param=5.0),
     GaussianNoise(param=10.0)],
     # Цепочка 2: другой тип размытия
    [DefocusBlur(psf=gaussian_distribution, param=3.0),
     GaussianNoise(param=5.0)],
]

# Определение алгоритмов
methods = [algorithm1, algorithm2]

# Запуск полного пайплайна
proc.full_process(filters=filters, methods=methods)
```

Внутри `full_process()`:

1. **Применение фильтров** — для каждого изображения и цепочки:
   - Последовательное применение: `filter1.filter(image)` → `filter2.filter(result)` → ...
   - Параллельная обработка ядра (те же фильтры к delta-функции, кроме шумовых)
   - Сохранение смазанного и ядра на диск, запись метрик
2. **Восстановление** — для каждого смазанного и каждого алгоритма:
   - `algorithm.process(blurred)` → `(restored, kernel)`
   - Сохранение + метрики
3. **Визуализация**
   - Визуализация `show()` 
   - Экспорт таблиц по алгоритмам в CSV, 
   - Построение фронта Парето `pareto()`

## Автоматизация через run.py

```
YAML/JSON конфиг
       │
       ▼
  load_config()            ← Загрузка (PyYAML / json)
       │
       ▼
  validate_config()        ← JSON Schema + логические проверки
       │
       ▼
  create_algorithm()       ← Динамический импорт из реестра
  create_filter_chains()   ← Подстановка PSF-функций
       │
       ▼
  Processing()             ← Создание экземпляра фреймворка
       │
       ▼
  bind() / read_all()      ← Загрузка изображений
       │
       ▼
  process() или            ← Обработка
  full_process()
       │
       ▼
  export_metadata()        ← metadata.json
  generate_latex_report()  ← report.tex (опционально)
```

## Формат сохранённых связей (dataset.json)

```json
{
    "1": {
        "original_path": "images/original/airplane.png",
        "blurred_path": "blurred/airplane.png",
        "original_kernel": "blurred/kernel_airplane.png",
        "filters": "|defocus_gaussian_5.0_None|gaussiannoise_10.0",
        "blurred_psnr": 24.5,
        "blurred_ssim": 0.78,
        "is_color": false,
        "algorithm": ["vabid", "richardson_lucy"],
        "restored_paths": {
            "vabid": "restored/airplane_vabid.png",
            "richardson_lucy": "restored/airplane_richardson_lucy.png"
        },
        "kernel_paths": {
            "vabid": "restored/airplane_vabid_kernel.png",
            "richardson_lucy": "restored/airplane_richardson_lucy_kernel.png"
        },
        "psnr": {"vabid": 28.3, "richardson_lucy": 26.1},
        "ssim": {"vabid": 0.89, "richardson_lucy": 0.84}
    }
}
```

## Обратная гистограммная коррекция

При использовании `histogram_equalization()` или `histogram_equalization_CLAHE()` алгоритм работает с выровненным изображением. Для корректного сравнения метрик необходимо вернуть результат к исходному распределению:

```python
proc.histogram_equalization()           # Выравнивание → preprocess/
proc.process(algorithm)                 # Восстановление выровненного
proc.inverse_histogram_equalization()   # Обратная коррекция
proc.show()                             # Визуализация
```

`inverse_histogram_equalization()` использует `match_histograms()` из `scikit-image` для приведения гистограммы восстановленного к гистограмме исходного смазанного.
