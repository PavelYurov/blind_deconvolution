# Установка и настройка

## Системные требования

| Требование | Минимум | Рекомендуется |
|---|---|---|
| Python | 3.10 | 3.11+ |
| ОС | Windows 10 / Linux / macOS | — |
| ОЗУ | 4 ГБ | 8+ ГБ |
| Диск | 500 МБ | 2+ ГБ (с данными) |

## Установка пакета

### Способ 1: Установка из GitHub (рекомендуется)

```bash
pip install git+https://github.com/PavelYurov/blind_deconvolution.git
```

Будут установлены основные зависимости: `numpy`, `scipy`, `opencv-python`, `scikit-image`, `pandas`, `matplotlib`, `pyyaml`, `jsonschema` и др.

```python
import blinddeconv
from blinddeconv.processing import Processing
```

### Способ 2: Установка для разработки (editable mode)

```bash
git clone https://github.com/PavelYurov/blind_deconvolution.git
cd blind_deconvolution
pip install -e .
```

### Способ 3: Через install.py (интерактивный установщик)

Скрипт `scripts/install.py` предоставляет интерактивный установщик с профилями:

```bash
# Просмотр доступных профилей
python scripts/install.py list-profiles

# Проверка установленных зависимостей
python scripts/install.py check base

# Установка базового профиля
python scripts/install.py install base

# Установка с автоподтверждением
python scripts/install.py install base -y
```

Установщик автоматически:
- определяет виртуальное окружение (или предлагает создать);
- проверяет уже установленные пакеты;
- устанавливает только недостающие.

## Профили зависимостей

| Профиль | Что включает | Команда |
|---|---|---|
| `base` | Основные зависимости проекта | `pip install .` |
| `cli` | CLI-интерфейс (`click`, `pyyaml`, `jsonschema`) | `pip install ".[cli]"` |
| `full` | Расширенные версии (`scikit-image>=0.19`, `optuna>=3.0`) | `pip install ".[full]"` |
| `dev` | Инструменты разработки (`pytest`, `flake8`, `ipython`) | `pip install ".[dev]"` |
| `docs` | Генерация документации (`sphinx`, `sphinx-rtd-theme`, `myst-parser`) | `pip install ".[docs]"` |

Комбинирование профилей:

```bash
# CLI + разработка
pip install ".[cli,dev]"

# Всё вместе
pip install ".[cli,dev,docs,full]"
```

## Зависимости для CLI и автоматизации

Для работы `run.py` и `cli.py` необходимы дополнительные пакеты:

```bash
pip install pyyaml jsonschema click
```

Или через профиль:

```bash
pip install ".[cli]"
```

## Настройка виртуального окружения

```bash
# Создание
python -m venv .venv

# Активация (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Активация (Linux / macOS)
source .venv/bin/activate

# Установка проекта
pip install -e ".[cli,dev]"
```

## Опциональные зависимости

### GNU Octave (для MATLAB-обёрток)

Некоторые внешние алгоритмы требуют GNU Octave:

1. Установите [GNU Octave](https://octave.org/download)
2. Убедитесь, что `octave` доступен в `PATH`
3. Установите Python-обёртку: `pip install oct2py`

### CUDA / GPU (для нейросетевых методов)

Для алгоритмов на базе PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Проверка установки

```python
import blinddeconv
from blinddeconv.processing import Processing

print(f"Версия: {blinddeconv.processing.__version__}")

# Создание экземпляра Processing
p = Processing(images_folder="images/original", color=False)
print("Установка прошла успешно!")
```

## Удаление

```bash
pip uninstall blinddeconv
```

## Решение проблем

### `ImportError: No module named 'cv2'`

```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'blinddeconv'`

Убедитесь, что пакет установлен в текущее виртуальное окружение:

```bash
pip show blind-deconvolution
```

### Ошибки при сборке документации

```bash
pip install ".[docs]"
pip install myst-parser
```
