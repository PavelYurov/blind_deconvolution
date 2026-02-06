# Руководство для разработчиков

## Структура проекта

```
blind_deconvolution/
├── src/blinddeconv/              # Исходный код пакета
│   ├── __init__.py
│   ├── processing/               # Ядро фреймворка
│   │   ├── core.py               # Processing (фасад)
│   │   ├── utils.py              # Image, утилиты
│   │   ├── metrics.py            # PSNR, SSIM, Sharpness
│   │   ├── reader.py             # Загрузка изображений
│   │   ├── display.py            # Визуализация
│   │   ├── preprocessing.py      # Выравнивание гистограмм
│   │   ├── clear.py              # Очистка
│   │   ├── applyfilter.py        # Применение фильтров
│   │   ├── restore.py            # Восстановление (один алгоритм)
│   │   ├── restorepipeline.py    # Полный пайплайн
│   │   ├── tables.py             # Экспорт в CSV
│   │   └── extensions/           # Расширения
│   │       ├── base.py
│   │       ├── hyperparameter_optimization.py
│   │       └── pareto_analysis.py
│   ├── algorithms/               # Алгоритмы деконволюции
│   │   ├── base.py               # DeconvolutionAlgorithm (ABC)
│   │   ├── blind_deconvolution/
│   │   │   └── our_company/      # Собственные реализации
│   │   ├── nonblind_deconvolution/
│   │   ├── kernel_estimation/
│   │   └── octave/               # Octave/MATLAB-обвязка
│   ├── filters/                  # Фильтры искажений
│   │   ├── base.py               # FilterBase (ABC)
│   │   ├── blur.py               # DefocusBlur, MotionBlur, ...
│   │   ├── noise.py              # GaussianNoise, PoissonNoise, ...
│   │   ├── smooth.py             # MeanBlur, GaussianBlur, ...
│   │   └── distributions.py      # PSF-функции
│   └── scripts/                  # Утилиты
│       ├── dataset_generator.py
│       └── kernel_generator.py
├── configs/                      # Конфигурационные файлы
├── docs/                         # Документация
├── run.py                        # Автоматизация по конфигам
├── cli.py                        # CLI-интерфейс
├── scripts/install.py            # Установщик
├── pyproject.toml                # Конфигурация пакета
└── requirements.txt              # Зависимости
```

## Настройка окружения для разработки

```bash
# 1. Клонирование
git clone https://github.com/PavelYurov/blind_deconvolution.git
cd blind_deconvolution

# 2. Виртуальное окружение
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Linux/macOS

# 3. Установка в режиме разработки
pip install -e ".[dev,cli,docs]"

# 4. Проверка
python -c "from blinddeconv.processing import Processing; print('OK')"
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
python cli.py list-algorithms  # Должен появиться в таблице

# Тест
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
        # ... реализация ...
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
    "requires_psf": False,  # True если нужна PSF-функция
    "description": "Мой фильтр",
},
```

## Сборка документации

```bash
# Установка зависимостей
pip install ".[docs]"
pip install myst-parser

# Генерация API + сборка HTML
python docs/tools/build_docs.py

# Или вручную:
cd docs/source
sphinx-apidoc --separate -o . ../../src
sphinx-build -b html . ../_build/html
```

Документация появится в `docs/_build/html/index.html`.

## Тестирование

```bash
# Запуск тестов
pytest tests/

# С покрытием
pytest --cov=blinddeconv tests/

# Только определённый модуль
pytest tests/test_processing.py -v
```

## Git-конвенции

### Ветки

| Ветка | Назначение |
|---|---|
| `main` | Стабильная версия |
| `develop` | Текущая разработка |
| `feature/<name>` | Новый функционал |
| `bugfix/<name>` | Исправление ошибок |
| `docs/<name>` | Обновление документации |

### Коммиты

Формат: `<тип>: <описание>`

| Тип | Назначение |
|---|---|
| `feat` | Новый функционал |
| `fix` | Исправление бага |
| `refactor` | Рефакторинг без изменения поведения |
| `docs` | Только документация |
| `test` | Добавление/изменение тестов |
| `chore` | Конфигурация, зависимости |

Примеры:

```
feat: добавлен алгоритм VBBID_TV
fix: исправлен расчёт PSNR для цветных изображений
docs: обновлена API-справка для Processing
refactor: выделен ModuleClear из Processing
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

## Контакты

Если у вас есть вопросы или предложения:

- Создайте Issue на GitHub
- Напишите авторам проекта
