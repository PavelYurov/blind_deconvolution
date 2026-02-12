# API Reference

Справочник по публичным классам, методам и функциям пакета `blinddeconv`.

---

## `blinddeconv.processing.core`

### Класс `Processing`

Центральный класс-фасад фреймворка. Управляет загрузкой, фильтрацией, восстановлением и анализом изображений.

```python
from blinddeconv.processing import Processing
```

#### Конструктор

```python
Processing(
    images_folder: str = "images",
    blurred_folder: str = "blurred",
    restored_folder: str = "restored",
    data_path: str = "data",
    color: bool = False,
    kernel_dir: str = "kernels",
    preprocess_dir: str = "preprocess",
    dataset_path: str = "dataset",
)
```

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `images_folder` | `str` | `"images"` | Директория с исходными изображениями |
| `blurred_folder` | `str` | `"blurred"` | Директория со смазанными изображениями |
| `restored_folder` | `str` | `"restored"` | Директория для восстановленных |
| `data_path` | `str` | `"data"` | Директория для CSV/метаданных |
| `color` | `bool` | `False` | `True` — цветные, `False` — ч/б |
| `kernel_dir` | `str` | `"kernels"` | Директория для ядер |
| `preprocess_dir` | `str` | `"preprocess"` | Директория для предобработанных |
| `dataset_path` | `str` | `"dataset"` | Директория для метаданных датасетов |

При инициализации автоматически создаются все директории.

#### Методы загрузки

| Метод | Описание |
|---|---|
| `read_all()` | Загружает все изображения из `images_folder` |
| `read_one(path)` | Загружает одно изображение по пути |
| `bind(original, blurred, kernel=None, filter_description="unknown", color=True) → Image` | Создаёт связь между оригиналом и смазанным |
| `load_bind_state(bind_path)` | Загружает связи из JSON |
| `save_bind_state(file_path=None)` | Сохраняет связи в JSON |

#### Методы обработки

| Метод | Описание |
|---|---|
| `filter(filter_processor)` | Применяет фильтр ко всем изображениям |
| `save_filter()` | Сохраняет текущее смазанное в массив |
| `load_filter(index)` | Загружает смазанное из массива в буфер |
| `custom_filter(kernel_image_path, kernel_npy_path)` | Применяет пользовательское ядро |
| `process(algorithm, metadata=False, unique_path=True)` | Восстанавливает все изображения одним алгоритмом |
| `full_process(filters, methods, size=0.75, kernel_intencity_scale=10.0)` | Полный пайплайн: фильтры → восстановление → визуализация |

#### Предобработка

| Метод | Описание |
|---|---|
| `histogram_equalization(view_histogram=False)` | Выравнивание гистограмм |
| `histogram_equalization_CLAHE(view_histogram=False, clip_limit=0.01)` | Адаптивное (CLAHE) |
| `inverse_histogram_equalization(view_histogram=False)` | Обратная коррекция |

#### Визуализация и экспорт

| Метод | Описание |
|---|---|
| `show(size=1.0, kernel_intencity_scale=1.0, kernel_size=1.0)` | Визуализация сеткой |
| `show_line(window_scale=1.0, fontsize=8)` | Визуализация строкой |
| `get_table(table_path, display_table=False)` | Экспорт метрик в CSV |
| `pareto()` | Построение фронта Парето |
| `process_hyperparameter_optimization(*args, **kwargs)` | Запуск оптимизации гиперпараметров |

#### Очистка

| Метод | Описание |
|---|---|
| `clear_input()` | Удаляет все связи |
| `reset()` | Сброс состояний до исходного |
| `clear_output()` | Удаляет файлы + сброс |
| `clear_output_directory(warning)` | Очистка директорий (с подтверждением) |
| `clear_restored()` | Удаляет восстановленные файлы |
| `unbind_restored()` | Убирает связи с восстановленными |
| `clear_all()` | Полная очистка |

#### Утилиты

| Метод | Описание |
|---|---|
| `changescale(color: bool)` | Переключение цветного/ч/б |
| `len_blur() → int` | Количество вариантов размытия |

### Функция `merge`

```python
merge(frame1: Processing, frame2: Processing) → Processing
```

Объединяет массивы изображений из двух экземпляров `Processing`.

### Функция `show_from_table`

```python
show_from_table(table_path: Path, alg_name: str, window_scale: float = 1.0)
```

Визуализирует сетку из CSV-таблицы для указанного алгоритма.

---

## `blinddeconv.processing.utils`

### Класс `Image`

Структура данных «связь», хранящая граф оригинал → смазанные → восстановленные с метриками.

```python
Image(original_path: str, is_color: bool)
```

**Ключевые методы:**

| Метод | Возвращает | Описание |
|---|---|---|
| `get_original()` | `str` | Путь к оригиналу |
| `get_blurred()` | `str` | Путь к текущему смазанному (буфер) |
| `get_blurred_array()` | `np.ndarray` | Все сохранённые смазанные |
| `get_restored()` | `Dict[(str,str), str]` | Восстановленные `{(blur, alg): path}` |
| `get_PSNR()` | `Dict[(str,str), float]` | PSNR восстановленных |
| `get_SSIM()` | `Dict[(str,str), float]` | SSIM восстановленных |
| `get_algorithm()` | `np.ndarray` | Список применённых алгоритмов |
| `get_original_image()` | `np.ndarray` | Оригинал как массив |
| `get_blurred_image()` | `np.ndarray` | Смазанное как массив |
| `save_filter()` | — | Перемещает буфер в массив |

### Утилитарные функции

| Функция | Описание |
|---|---|
| `imread(path, color) → np.ndarray` | Загрузка изображения через OpenCV |
| `float_img_to_int(image) → np.ndarray` | Конвертация `[0,1]` → `[0,255]` |
| `prepare_image_for_metric(image) → np.ndarray` | Нормализация для метрик |
| `generate_unique_file_path(directory, filename) → Path` | Генерация уникального пути |
| `calculate_metrics(original, restored, data_range=None) → (psnr, ssim)` | Расчёт PSNR и SSIM |

---

## `blinddeconv.processing.metrics`

| Функция | Сигнатура | Описание |
|---|---|---|
| `PSNR` | `(original, restored) → float` | Peak Signal-to-Noise Ratio (dB) |
| `SSIM` | `(original, restored, data_range=None) → float` | Structural Similarity Index [0,1] |
| `Sharpness` | `(image) → float` | Дисперсия Лапласа (мера резкости) |
| `calculate_sml` | `(image) → float` | Sum of Modified Laplacian |
| `blur_complexity` | `(original, blurred) → float` | Мера сложности смаза [0,1] |
| `calculate_snr` | `(signal, noise) → float` | Signal-to-Noise Ratio (dB) |
| `noise_complexity` | `(signal, noise, min_snr, max_snr) → float` | Мера сложности шума [0,1] |

---

## `blinddeconv.algorithms.base`

### Класс `DeconvolutionAlgorithm` (ABC)

```python
from blinddeconv.algorithms.base import DeconvolutionAlgorithm
```

| Метод | Тип | Описание |
|---|---|---|
| `process(image) → (restored, kernel)` | abstract | Обработка изображения |
| `change_param(param: dict)` | abstract | Изменение гиперпараметров |
| `get_param() → List[Tuple[str, Any]]` | abstract | Получение параметров |
| `get_name() → str` | | Название алгоритма |
| `get_timer() → float` | | Время работы (-1 если не запускался) |
| `import_param_from_file(file: str)` | | Загрузка параметров из JSON |

### Собственные реализации

| Класс | Модуль | Категория |
|---|---|---|
| `RichardsonLucy` | `...classic.richardson_lucy` | Классический |
| `EMBlindDeconvolution` | `...classic.expectation_maximization` | Классический |
| `MAPDeconvolution` | `...classic.alternating_minimization` | Классический |
| `VBBID_TV` | `...bayesian.vbbid_tv` | Байесовский |
| `BBD_DEIP` | `...bayesian.bbd_deip` | Байесовский |
| `SB_BID_PE` | `...bayesian.sb_bid_pe` | Байесовский |
| `VAPIBE` | `...variational.vapibe` | Вариационный |
| `VABID` | `...variational.vabid` | Вариационный |
| `VBSK_SID_ST` | `...variational.vbsk_sid_st` | Вариационный |
| `VBC_BID` | `...sparse.vbc_bid` | Разреженный |

Полный путь модуля: `blinddeconv.algorithms.blind_deconvolution.our_company.<category>.<module>`

---

## `blinddeconv.filters`

### Класс `FilterBase` (ABC)

```python
from blinddeconv.filters.base import FilterBase
```

| Метод | Описание |
|---|---|
| `filter(image: np.ndarray) → np.ndarray` | Применение фильтра |
| `description() → str` | Описание для файловой системы |
| `get_type() → str` | Тип: `"blur"`, `"noise"`, `"denoise"`, `"custom_kernel"` |

### `blinddeconv.filters.blur`

| Класс | Описание | Параметры |
|---|---|---|
| `DefocusBlur` | Размытие вне фокуса (2D) | `psf: Callable`, `param: float`, `kernel_size: int=None` |
| `MotionBlur` | Размытие в движении (1D) | `psf: Callable`, `param: float`, `angle: float=0`, `kernel_length: int=None` |
| `BSpline_blur` | Криволинейное движение (B-spline) | `shape_points`, `intensity_points`, `output_size`, `shape_degree`, `intensity_degree`, `n_samples` |
| `Kernel_convolution` | Свёртка с ядром из .npy | `npy_file_path: str` |
| `Identical_kernel` | Тождественный фильтр | — |

### `blinddeconv.filters.noise`

| Класс | Описание | Параметры |
|---|---|---|
| `GaussianNoise` | Аддитивный гауссовский | `param: float` (стд. отклонение) |
| `PoissonNoise` | Пуассоновский (дробления) | `param: float` (интенсивность, 0-1) |
| `SaltAndPepperNoise` | Импульсный | `param: float` (вероятность) |
| `OldPhotoNoise` | Имитация старой фотографии | `param: float` |
| `ColoredNoise` | Цветной шум (powerlaw) | `param: float` |
| `Pink_Noise` | Розовый шум (1/f) | `param: float` |
| `Brown_Noise` | Коричневый шум (1/f²) | `param: float` |

### `blinddeconv.filters.smooth`

| Класс | Описание |
|---|---|
| `MeanBlur` | Сглаживание средним |
| `MedianBlur` | Медианное сглаживание |
| `GaussianBlur` | Гауссово сглаживание |
| `BilateralFilter` | Билатеральный фильтр |

### `blinddeconv.filters.distributions`

PSF-функции для `DefocusBlur` и `MotionBlur`:

| Функция | Сигнатура | Описание |
|---|---|---|
| `gaussian_distribution` | `(x, std) → np.ndarray` | Гауссова PSF |
| `uniform_distribution` | `(x, radius) → np.ndarray` | Равномерная (диск/прямоугольник) |
| `linear_decay_distribution` | `(x, radius) → np.ndarray` | Линейно убывающая (конус) |
| `ring_distribution` | `(x, radius) → np.ndarray` | Кольцевая |
| `exponential_decay_distribution` | `(x, scale) → np.ndarray` | Экспоненциально убывающая |
| `generate_multivariate_normal_kernel` | `(ksize, cov) → np.ndarray` | 2D-ядро из многомерного нормального |
| `generate_bspline_motion_kernel` | `(ksize, points, thickness) → np.ndarray` | Ядро по B-spline кривой |

---

## `blinddeconv.processing.extensions`

### Класс `ProcessingExtension` (ABC)

Базовый класс для расширений:

| Метод | Описание |
|---|---|
| `execute(*args, **kwargs)` | Основной метод расширения (abstract) |
| `_prepare_image(image)` | Нормализация изображения для метрик |
| `_validate_processing()` | Проверка корректности `processing` |

### Класс `HyperparameterOptimizer`

```python
result = proc.process_hyperparameter_optimization(
    algorithm_processor=algo,
    param_ranges={"max_iter": (50, 500), "kernel_size": (11, 31)},
    n_trials=50,
    metric="PSNR",           # "PSNR" | "SSIM" | "SHARPNESS"
    method=OptimizationMethod.TPE,  # TPE | RANDOM | GP | NSGA2
    timeout=3600,
    n_jobs=1,
    seed=42,
    save_results=True,
)
```

Возвращает `OptimizationResult` с полями: `best_params`, `best_value`, `n_trials`, `history`, `elapsed_time`.

### Класс `ParetoFrontAnalyzer`

Многокритериальный анализ: 3D фронт Парето, 2D проекции, тепловые карты.

```python
proc.pareto()
```

### Вспомогательные типы

| Класс | Описание |
|---|---|
| `ParameterRange` | Диапазон гиперпараметра (name, min, max, log_scale, step) |
| `OptimizationResult` | Результат оптимизации |
| `ParetoPoint` | Точка в многокритериальном пространстве |
| `OptimizationMethod` | Enum: `TPE`, `RANDOM`, `GP`, `NSGA2` |
| `MetricType` | Enum: `PSNR`, `SSIM`, `SHARPNESS` |

---

## `blinddeconv.scripts`

### Класс `DatasetGenerator`

```python
from blinddeconv.scripts.dataset_generator import DatasetGenerator

generator = DatasetGenerator(
    processing_instance=proc,
    input_dir="images/original",
    output_dir="images/distorted",
    kernel_dir="images/ground_truth_filters",
    kernel_data_dir="images/kernel_data",
)
generator.generate()
```

### Класс `KernelGenerator`

```python
from blinddeconv.scripts.kernel_generator import KernelGenerator

gen = KernelGenerator(
    kernel_dir="kernels/png",
    kernel_data_dir="kernels/npy",
    kernel_size=51,
)
```

---

## `run.py` — программный API

### `load_config(config_path) → dict`

Загрузка YAML/JSON конфигурации.

### `validate_config(config) → List[str]`

Валидация по JSON Schema + логические проверки. Возвращает список ошибок.

### `run_pipeline(config, output_dir=None, generate_report=False, dry_run=False) → dict`

Запуск пайплайна программно:

```python
from run import load_config, validate_config, run_pipeline

config = load_config("configs/experiment.yaml")
errors = validate_config(config)
if not errors:
    result = run_pipeline(config, generate_report=True)
```

### `create_algorithm(algo_config) → DeconvolutionAlgorithm`

Фабрика алгоритмов из конфигурации.

### `create_filter(filter_config) → FilterBase`

Фабрика фильтров из конфигурации.

### Реестры

- `ALGORITHM_REGISTRY` — словарь `{name: {module, class_name, description, category}}`
- `FILTER_REGISTRY` — словарь `{name: {module, class_name, requires_psf, description}}`
- `PSF_REGISTRY` — словарь `{name: full_module_path}`
