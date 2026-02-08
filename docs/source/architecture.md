# Архитектура системы

## Модульная структура

BlindDeconvolution — модульный фреймворк для исследования методов слепой деконволюции изображений. Архитектура построена по принципу **фасада**: центральный класс `Processing` делегирует операции специализированным модулям.

Проект построен по модульному принципу с центральным классом-фасадом `Processing`:

```
┌──────────────────────────────────────────────────────────────┐
│                       Processing (Facade)                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  Core Modules                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │  │
│  │  │  Reader  │ │ Display  │ │  Restore  │ │ Pipeline │  │  │
│  │  └──────────┘ └──────────┘ └───────────┘ └──────────┘  │  │
│  │                                                        │  │
│  │  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌──────────┐  │  │
│  │  │  Filter  │ │  Clear   │ │  Tables   │ │   Prep   │  │  │
│  │  └──────────┘ └──────────┘ └───────────┘ └──────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │               Optimization & Analysis                  │  │
│  │  ┌────────────────────────┐ ┌────────────────────────┐ │  │
│  │  │   HyperparamOptimizer  │ │  ParetoFrontAnalyzer   │ │  │
│  │  └────────────────────────┘ └────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
  ┌───────────┐       ┌────────────┐       ┌─────────────┐
  │  filters/ │       │ algorithms/│       │   core.py   │
  │  • blur   │       │ • our_co.  │       │   utils.py  │
  │  • noise  │       │ • 3rd_party│       │   metrics.py│
  │  • smooth │       │ • octave/  │       │   image.py  │
  │  • denoise│       │ • cython/  │       └─────────────┘
  └───────────┘       └────────────┘
```

## Основные компоненты

### 1. `Processing` — центральный класс (фасад)

**Модуль:** `blinddeconv.processing.core`

Единая точка входа для всех операций. При инициализации автоматически создаёт все рабочие директории и подключает модули.

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `images_folder` | `"images"` | Директория с оригиналами |
| `blurred_folder` | `"blurred"` | Директория со смазанными |
| `restored_folder` | `"restored"` | Директория с восстановленными |
| `data_path` | `"data"` | Директория для CSV/метаданных |
| `color` | `False` | `True` — цветные, `False` — ч/б |
| `kernel_dir` | `"kernels"` | Директория для ядер |

Внутренние модули:

| Атрибут | Класс | Назначение |
|---|---|---|
| `reader` | `ModuleReader` | Загрузка изображений, создание связей |
| `display` | `ModuleDisplay` | Визуализация результатов |
| `histogram` | `ModulePreprocessing` | Предобработка (выравнивание гистограмм) |
| `tables` | `ModuleData` | Экспорт метрик в CSV |
| `clear` | `ModuleClear` | Очистка связей и файлов |
| `apply_filter` | `ModuleFilter` | Применение фильтров |
| `module_process` | `ModuleProcess` | Восстановление (один алгоритм) |
| `process_pipeline` | `ModuleProcessPipeline` | Полный пайплайн (фильтры + восстановление) |
| `optimizer` | `HyperparameterOptimizer` | Оптимизация гиперпараметров |
| `analyzer` | `ParetoFrontAnalyzer` | Парето-анализ |

### 2. `Image` — связь изображений

**Модуль:** `blinddeconv.processing.utils`

Ключевая структура данных. Хранит граф связей между оригиналом, смазанными вариантами, восстановленными результатами и метриками.

```
Image (связь)
│
├── original_path          ← Путь к оригиналу
│
├── blurred_path           ← Буфер (текущее смазанное)
├── blurred_array[]        ← Сохранённые смазанные варианты
│   ├── filters{}          ← Описания фильтров
│   ├── blurred_psnr{}     ← PSNR смазанных
│   ├── blurred_ssim{}     ← SSIM смазанных
│   └── original_kernels{} ← Ядра (GT PSF)
│
└── Для каждого (blurred_path, algorithm_name):
    ├── restored_paths{}   ← Восстановленные изображения
    ├── kernel_paths{}     ← Восстановленные ядра
    ├── psnr{}             ← PSNR восстановленных
    └── ssim{}             ← SSIM восстановленных
```

### 3. `DeconvolutionAlgorithm` — базовый класс алгоритмов

**Модуль:** `blinddeconv.algorithms.base`

Абстрактный класс, определяющий интерфейс для всех алгоритмов деконволюции:

- `process(image) → (restored, kernel)` — обработка изображения
- `change_param(params)` — изменение гиперпараметров
- `get_param() → [(name, value), ...]` — получение текущих параметров
- `get_name() → str` — название алгоритма
- `get_timer() → float` — время последнего запуска

### 4. `FilterBase` — базовый класс фильтров

**Модуль:** `blinddeconv.filters.base`

Абстрактный класс для фильтров искажений:

- `filter(image) → image` — применение фильтра
- `description() → str` — описание для файловой системы
- `get_type() → str` — тип фильтра (`"blur"`, `"noise"`, `"denoise"`)

### 5. Система расширений (extensions)

**Модуль:** `blinddeconv.processing.extensions`

Расширения наследуют `ProcessingExtension` и добавляют специализированный функционал:

- **`HyperparameterOptimizer`** — байесовская оптимизация гиперпараметров через Optuna (TPE, Random, GP, NSGA-II)
- **`ParetoFrontAnalyzer`** — многокритериальный анализ с построением 3D фронта Парето

## Организация  исхожного кода алгоритмов

```
algorithms/
├── base.py                           # DeconvolutionAlgorithm (ABC)
├── blind_deconvolution/
│   ├── our_company/                  # Собственные реализации
│   │   ├── bayesian/                 # Байесовские методы
│   │   │   ├── vbbid_tv.py          #   VBBID_TV
│   │   │   ├── bbd_deip.py          #   BBD_DEIP
│   │   │   └── sb_bid_pe.py         #   SB_BID_PE
│   │   ├── classic/                  # Классические методы
│   │   │   ├── richardson_lucy.py   #   RichardsonLucy
│   │   │   ├── expectation_maximization.py  # EM
│   │   │   └── alternating_minimization.py  # MAP
│   │   ├── variational/             # Вариационные методы
│   │   │   ├── vabid.py             #   VABID (Likas2004)
│   │   │   ├── vapibe.py            #   VAPIBE
│   │   │   └── vbsk_sid_st.py       #   VBSK_SID_ST
│   │   └── sparse/                  # Разреженные методы
│   │       ├── vbc_bid.py           #   VBC_BID
│   │       └── pbtvgr.py            #   PBTVGR
│   └── third_party_company/         # Внешние обёртки
├── nonblind_deconvolution/           # Не-слепая деконволюция
├── kernel_estimation/                # Оценка ядра
├── octave/                           # Octave/MATLAB-обвязка
└── unsorted/                         # Экспериментальные
```

## Организация фильтров

| Модуль | Фильтры | Тип |
|---|---|---|
| `filters.blur` | `DefocusBlur`, `MotionBlur`, `BSpline_blur`, `Kernel_convolution` | `blur` |
| `filters.noise` | `GaussianNoise`, `PoissonNoise`, `SaltAndPepperNoise`, `OldPhotoNoise`, `ColoredNoise`, `Pink_Noise`, `Brown_Noise` | `noise` |
| `filters.smooth` | `MeanBlur`, `MedianBlur`, `GaussianBlur`, `BilateralFilter` | `denoise` |
| `filters.distributions` | `gaussian_distribution`, `uniform_distribution`, `ring_distribution`, `linear_decay_distribution`, `exponential_decay_distribution` | (PSF-функции) |

## Вспомогательные скрипты

| Модуль | Класс | Назначение |
|---|---|---|
| `scripts.dataset_generator` | `DatasetGenerator` | Автоматическая генерация датасета с комбинациями фильтров |
| `scripts.kernel_generator` | `KernelGenerator` | Генерация ядер размытия (PSF) для экспериментов |

## Система автоматизации

Два интерфейса для автоматизации в корне проекта:

| Файл | Назначение | Зависимости |
|---|---|---|
| `run.py` | Запуск пайплайна по YAML/JSON-конфигу | `pyyaml`, `jsonschema` |
| `cli.py` | CLI-интерфейс (click) для быстрых команд | `click`, `pyyaml` |

Подробнее — в разделах [Конфигурационные файлы](configuration.md) и [Руководство пользователя](usage_guide.md).

## Паттерны проектирования

1. **Facade** — `Processing` скрывает сложность подсистем за единым интерфейсом.
2. **Strategy** — `DeconvolutionAlgorithm` и `FilterBase` позволяют подменять алгоритмы без изменения клиентского кода.
3. **Template Method** — `ModuleProcessPipeline.full_process()` определяет скелет пайплайна, делегируя шаги подметодам.
4. **Registry** — `run.py` содержит реестры (`ALGORITHM_REGISTRY`, `FILTER_REGISTRY`) для динамического создания экземпляров по имени.
5. **Extension** — `ProcessingExtension` (ABC) позволяет подключать новый функционал без изменения ядра.
