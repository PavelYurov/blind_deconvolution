# Руководство для разработчиков

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

## Настройка окружения для разработки

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
Установка
```bash
pip install -e ".[dev,cli,docs]"
python -c "from blinddeconv.processing import Processing"
```

## Стандарты кода

### Общие правила

- **Язык кода**: английский (имена переменных, классов, функций)
- **Язык сообщений**: русский (строки для пользователя, print, logging)
- **Type hints**: обязательны для всех публичных функций и методов
- **Docstrings**: обязательны для всех классов и публичных методов

### Стиль Docstrings

Используется **NumPy-style**:

```python
def process(self,
            image: np.ndarray,
            kernel_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Восстанавливает изображение методом слепой деконволюции.

    Параметры
    ---------
    image : np.ndarray
        Входное смазанное изображение.
    kernel_size : int, по умолчанию 21
        Размер ядра размытия (нечётное число).

    Возвращает
    ----------
    Tuple[np.ndarray, np.ndarray]
        Кортеж (восстановленное изображение, оценённое ядро).

    Raises
    ------
    ValueError
        Если kernel_size чётный.
    """
```

### Именование

| Элемент | Стиль | Пример |
|---|---|---|
| Классы | `PascalCase` | `ModuleReader`, `DeconvolutionAlgorithm` |
| Функции/методы | `snake_case` | `read_all()`, `get_blurred_image()` |
| Константы | `UPPER_SNAKE_CASE` | `ALGORITHM_REGISTRY` |
| Приватные | `_prefix` | `_apply_single_filter()` |
| Модули | `snake_case` | `reader.py`, `applyfilter.py` |

### Логирование

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Запуск обработки изображения")
logger.warning(f"Файл не найден: {path}")
logger.error(f"Ошибка восстановления: {e}")
```

Не использовать `print()` для отладочных сообщений в новом коде. Для вывода пользователю допускается `print()` только в CLI-скриптах.

## Добавление нового алгоритма

### 1. Создайте модуль

Путь: `src/blinddeconv/algorithms/blind_deconvolution/our_company/<category>/<name>.py`

```python
"""
Реализация алгоритма <Name>.

Автор: <Ваше имя>
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from blinddeconv.algorithms.base import DeconvolutionAlgorithm


class MyAlgorithm(DeconvolutionAlgorithm):
    """
    Описание алгоритма.

    Параметры
    ---------
    max_iter : int
        Максимальное число итераций.
    kernel_size : int
        Размер ядра.
    """

    def __init__(self, max_iter: int = 100, kernel_size: int = 21):
        self.name = "my_algorithm"
        self.max_iter = max_iter
        self.kernel_size = kernel_size
        self.timer = -1

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Восстановление изображения."""
        # ... реализация ...
        return restored, kernel

    def change_param(self, param: Dict[str, Any]) -> None:
        """Изменение гиперпараметров."""
        if "max_iter" in param:
            self.max_iter = param["max_iter"]
        if "kernel_size" in param:
            self.kernel_size = param["kernel_size"]

    def get_param(self) -> List[Tuple[str, Any]]:
        """Текущие параметры."""
        return [
            ("max_iter", self.max_iter),
            ("kernel_size", self.kernel_size),
        ]
```

### 2. Зарегистрируйте в `run.py`

Добавьте запись в `ALGORITHM_REGISTRY`:

```python
"my_algorithm": {
    "module": "blinddeconv.algorithms.blind_deconvolution.our_company.<cat>.<name>",
    "class_name": "MyAlgorithm",
    "description": "Мой алгоритм",
    "category": "<category>",
},
```

### 3. Проверьте

```bash
python cli.py list-algorithms
python cli.py process -i test_image.png -a my_algorithm
```

## Добавление нового фильтра

### 1. Наследуйте от `FilterBase`

```python
from blinddeconv.filters.base import FilterBase

class MyFilter(FilterBase):
    def __init__(self, param: float = 1.0):
        self.param = param

    def filter(self, image: np.ndarray) -> np.ndarray:
        # ...
        return filtered_image

    def description(self) -> str:
        return f"|my_filter_{self.param}"

    def get_type(self) -> str:
        return "blur"  # или "noise", "denoise"
```

### 2. Зарегистрируйте в `run.py`

```python
"my_filter": {
    "module": "blinddeconv.filters.<module>",
    "class_name": "MyFilter",
    "requires_psf": False,
    "description": "Мой фильтр",
},
```

## Сборка документации
Установка зависимостей
```bash
pip install ".[docs]"
```
Генерация API и сборка HTML
```bash
python docs/tools/build_docs.py
```
Или вручную:
```bash
cd docs/source
sphinx-apidoc --separate -o docs/source src/
sphinx-build -b html docs/source docs/_build/html
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

## Документация (Sphinx)

HTML-документация проекта генерируется с помощью **Sphinx** и публикуется [по ссылке](https://pavelyurov.github.io/blind_deconvolution/).

### Структура документации

- `docs/source/` — исходники Sphinx (`.rst`, `.md`, `conf.py`)
- `docs/tools/build_docs.py` — скрипт сборки (генерация API + сборка HTML)
- `docs/_build/html/` — результат локальной сборки (появляется после первого билда)

Документация строится из:
- **Docstrings Python-модулей** (автодокументация через `sphinx.ext.autodoc`)
- **Markdown-файлы** в `docs/source/` (через MyST-Parser)

### Требования

В активном окружении Python должны быть доступны утилиты:
- `sphinx-build`
- `sphinx-apidoc`

Установка:
```bash
pip install ".[docs]"
```

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
