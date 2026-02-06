# История изменений (CHANGELOG)

Все существенные изменения проекта документируются в этом файле.
Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.0.0/).

---

## [1.1.0] — 2026-02-06

### Добавлено

- **Система автоматизации**: `run.py` для запуска пайплайна по YAML/JSON-конфигурации.
- **CLI-интерфейс**: `cli.py` на базе Click с командами `process`, `run`, `generate-config`, `view-config`, `interactive`, `list-algorithms`, `list-filters`.
- **Конфигурационные файлы**: папка `configs/` с предготовыми шаблонами:
  - `basic_deconvolution.yaml` — базовый пример
  - `medical_imaging.yaml` — медицинские изображения
  - `satellite_images.yaml` — спутниковые снимки
  - `experiment_template.json` — полный шаблон
- **Валидация конфигов**: JSON Schema + логические проверки.
- **Dry-run режим**: проверка конфига и плана без выполнения обработки.
- **Генерация LaTeX-отчётов**: автоматическое создание `.tex` файла с результатами.
- **Экспорт метаданных**: `metadata.json` с настройками и результатами эксперимента.
- **Динамические реестры**: `ALGORITHM_REGISTRY`, `FILTER_REGISTRY`, `PSF_REGISTRY` для загрузки классов по имени.
- **Профиль зависимостей `cli`**: `pyyaml`, `jsonschema`, `click`.
- **Markdown-документация**: полный набор `.md` файлов в `docs/source/` с поддержкой MyST-Parser для Sphinx.

### Изменено

- `pyproject.toml`: добавлены зависимости `pyyaml>=6.0` и `jsonschema>=4.0.0`.
- `requirements.txt`: добавлены `click`, `pyyaml`, `jsonschema`.
- `docs/source/conf.py`: добавлена поддержка MyST-Parser.
- `docs/source/index.rst`: расширен с включением новых `.md` файлов.
- `README.md`: добавлен раздел «Автоматизация и CLI».

## [1.0.0] — 2024-12-01

### Добавлено

- **Фреймворк `Processing`**: центральный класс для управления конвейером обработки.
- **Система связей**: класс `Image` для графа оригинал → смазанные → восстановленные.
- **Модули обработки**: `reader`, `display`, `clear`, `applyfilter`, `restore`, `restorepipeline`, `preprocessing`, `tables`.
- **Базовый класс алгоритмов**: `DeconvolutionAlgorithm` (ABC).
- **Собственные алгоритмы**:
  - Классические: `RichardsonLucy`, `EMBlindDeconvolution`, `MAPDeconvolution`
  - Байесовские: `VBBID_TV`, `BBD_DEIP`, `SB_BID_PE`
  - Вариационные: `VABID`, `VAPIBE`, `VBSK_SID_ST`
  - Разреженные: `VBC_BID`
- **Система фильтров**: `FilterBase` (ABC) с реализациями для blur, noise, smooth.
- **PSF-функции**: `gaussian`, `uniform`, `ring`, `linear_decay`, `exponential_decay` и др.
- **Метрики**: PSNR, SSIM, Sharpness, SML, blur complexity, noise complexity.
- **Расширения**: `HyperparameterOptimizer` (Optuna), `ParetoFrontAnalyzer`.
- **Генераторы**: `DatasetGenerator`, `KernelGenerator`.
- **Установщик**: `scripts/install.py` с профилями зависимостей.
- **Sphinx-документация**: `conf.py`, `build_docs.py`.
- **Предобработка**: выравнивание гистограмм (обычное и CLAHE), обратная коррекция.
