# Конфигурационные файлы

## Обзор

Система конфигурации позволяет описать эксперимент целиком в одном файле (YAML или JSON) и запустить одной командой. Конфиги обеспечивают **воспроизводимость экспериментов** — один конфиг всегда даёт одинаковый результат.

```bash
python run.py --config configs/basic_deconvolution.yaml
```

## Формат файла

Поддерживаются два формата:

- **YAML** (`.yaml`, `.yml`) — рекомендуется, удобен для чтения и редактирования
- **JSON** (`.json`) — строгий формат, подходит для программной генерации

## Структура конфигурации

### Полный пример (YAML)

```yaml
# ── Метаданные эксперимента ──
experiment:
  name: "My Experiment"
  description: "Описание эксперимента"

# ── Входные данные ──
input:
  images_folder: "images/original"
  blurred_folder: "images/distorted"
  color: false
  load_mode: "bind"          # "all" | "bind" | "bind_state"

  bindings:                  # Для load_mode: "bind"
    - original: "images/original/airplane.png"
      blurred: "images/distorted/airplane_blurred.png"
      kernel: "images/kernel_data/kernel.npy"     # опционально
      filter_description: "gaussian_blur"          # опционально

  # bind_state_path: "dataset/dataset.json"  # Для load_mode: "bind_state"

# ── Выходные директории ──
output:
  restored_folder: "results/restored"
  data_folder: "results/data"
  kernel_folder: "results/kernels"

# ── Режим обработки ──
processing:
  mode: "process"            # "process" | "full_process"
  metadata: true
  unique_paths: true

# ── Алгоритмы (обязательно) ──
algorithms:
  - name: "vabid"
    params:
      max_iter: 100
      kernel_size: 21

# ── Фильтры (для mode: "full_process") ──
filters:
  - chain:
      - type: "defocus_blur"
        params:
          psf: "gaussian"
          param: 5.0
      - type: "gaussian_noise"
        params:
          param: 10.0

# ── Отчёт ──
report:
  generate: false
  format: "latex"
  output_path: "results/report.tex"
```

## Описание секций

### `experiment` (опционально)

| Поле | Тип | Описание |
|---|---|---|
| `name` | `string` | Название эксперимента |
| `description` | `string` | Описание |

### `input` (обязательно)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `images_folder` | `string` | **обязательно** | Папка с оригинальными изображениями |
| `blurred_folder` | `string` | `"blurred"` | Папка со смазанными |
| `color` | `bool` | `false` | `true` — цветные, `false` — ч/б |
| `load_mode` | `string` | `"all"` | Режим загрузки (см. ниже) |
| `bindings` | `array` | `[]` | Список связей (для `bind`) |
| `bind_state_path` | `string` | — | Путь к JSON (для `bind_state`) |

**Режимы загрузки (`load_mode`):**

| Режим | Описание | Когда использовать |
|---|---|---|
| `"all"` | Загрузить все файлы из `images_folder` | Режим `full_process` |
| `"bind"` | Связать конкретные пары оригинал + смазанное | Режим `process` |
| `"bind_state"` | Загрузить из сохранённого `dataset.json` | Продолжение эксперимента |

### `output` (опционально)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `restored_folder` | `string` | `"restored"` | Куда сохранять восстановленные |
| `data_folder` | `string` | `"data"` | Куда сохранять CSV/метаданные |
| `kernel_folder` | `string` | `"kernels"` | Куда сохранять ядра |

### `processing` (опционально)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `mode` | `string` | `"process"` | `"process"` или `"full_process"` |
| `metadata` | `bool` | `true` | Сохранять JSON-метаданные |
| `unique_paths` | `bool` | `true` | Генерировать уникальные имена файлов |

### `algorithms` (обязательно)

Минимум один алгоритм. Каждый элемент:

| Поле | Тип | Описание |
|---|---|---|
| `name` | `string` | Имя из реестра или произвольное |
| `module` | `string` | (опц.) Полный путь к модулю |
| `class_name` | `string` | (опц.) Имя класса |
| `params` | `object` | (опц.) Параметры конструктора |

**Алгоритмы из реестра:**

| Имя | Категория | Класс |
|---|---|---|
| `richardson_lucy` | classic | `RichardsonLucy` |
| `em` | classic | `EMBlindDeconvolution` |
| `map` | classic | `MAPDeconvolution` |
| `vbbid_tv` | bayesian | `VBBID_TV` |
| `bbd_deip` | bayesian | `BBD_DEIP` |
| `sb_bid_pe` | bayesian | `SB_BID_PE` |
| `vapibe` | variational | `VAPIBE` |
| `vabid` | variational | `VABID` |
| `vbsk_sid_st` | variational | `VBSK_SID_ST` |
| `vbc_bid` | sparse | `VBC_BID` |

**Пользовательский алгоритм:**

```yaml
algorithms:
  - name: "my_algorithm"
    module: "blinddeconv.algorithms.blind_deconvolution.our_company.classic.richardson_lucy"
    class_name: "RichardsonLucy"
    params:
      max_iter: 50
```

### `filters` (для `full_process`)

Список цепочек. Каждая цепочка — последовательность фильтров, применяемых к оригиналу.

| Поле фильтра | Тип | Описание |
|---|---|---|
| `type` | `string` | Имя из реестра фильтров |
| `params` | `object` | Параметры конструктора |

**Фильтры из реестра:**

| Имя | Тип | Параметры |
|---|---|---|
| `defocus_blur` | blur | `psf` (имя PSF-функции), `param`, `kernel_size` |
| `motion_blur` | blur | `psf`, `param`, `angle`, `kernel_length` |
| `gaussian_noise` | noise | `param` (стд. отклонение) |
| `poisson_noise` | noise | `param` (интенсивность) |
| `salt_pepper_noise` | noise | `param` (вероятность) |
| `mean_blur` | smooth | см. документацию `MeanBlur` |
| `gaussian_blur` | smooth | см. документацию `GaussianBlur` |

**PSF-функции для blur-фильтров:**

`gaussian`, `uniform`, `linear_decay`, `ring`, `exponential_decay`

### `report` (опционально)

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `generate` | `bool` | `false` | Генерировать LaTeX-отчёт |
| `format` | `string` | `"latex"` | Формат отчёта |
| `output_path` | `string` | `"results/report.tex"` | Путь к файлу |

## Валидация

Конфигурация валидируется автоматически по JSON Schema и логическим правилам:

```bash
# Только валидация (без запуска)
python run.py --config configs/experiment.yaml --validate-only

# Dry-run: план выполнения без обработки
python run.py --config configs/experiment.yaml --dry-run
```

Проверяемые правила:

- Обязательные секции `input` и `algorithms` присутствуют
- Режим `process` требует `load_mode: "bind"` или `"bind_state"`
- Режим `full_process` требует непустой секции `filters`
- Все имена алгоритмов и фильтров — из реестра или с указанием `module`/`class_name`

## Готовые конфиги

В папке `configs/` находятся примеры:

| Файл | Описание | Режим |
|---|---|---|
| `basic_deconvolution.yaml` | Базовый пример с парой изображений | `process` |
| `medical_imaging.yaml` | Медицинские изображения | `full_process` |
| `satellite_images.yaml` | Спутниковые снимки | `full_process` |
| `experiment_template.json` | Полный шаблон со всеми полями | — |

## Генерация конфигов через CLI

```bash
# Базовый шаблон
python cli.py generate-config --template basic --output my_config.yaml

# Медицинский шаблон (JSON)
python cli.py generate-config -t medical -f json -o config.json

# Просмотр конфига в таблице
python cli.py view-config configs/experiment.yaml

# Интерактивный режим (пошаговые вопросы)
python cli.py interactive
```
