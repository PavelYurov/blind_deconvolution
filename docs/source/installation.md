# Установка и настройка

## Системные требования

| Требование | Минимум | Рекомендуется |
|---|---|---|
| Python | 3.10 | 3.11+ |
| ОС | Windows 10 / Linux / macOS | — |
| ОЗУ | 4 ГБ | 8+ ГБ |
| Диск | 500 МБ | 2+ ГБ (с данными) |

---

## Установка

1. Установить последнюю версию пакета:

```bash
pip install git+https://github.com/PavelYurov/blind_deconvolution.git
```

Будут установлен пакет `blinddeconv` и остальные зависимости (`numpy`, `scipy`, `opencv-python`, `scikit-image`, и др.)

```py
import blinddeconv
```

2. Установка с дополнительными зависимостями:

Включает инструменты для тестирования, линтинга и интерактивной работы (`pytest`, `flake8`, `ipython`, `setuptools`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[dev]"
```

Включает CLI-интерфейс и автоматизацию (`click`, `pyyaml`, `jsonschema`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[cli]"
```

Необходимо для генерации документации через Sphinx (`sphinx`, `sphinx-rtd-theme`)

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[docs]"
```

Всё вместе:

```bash
pip install "git+https://github.com/PavelYurov/blind_deconvolution.git[cli,dev,docs]"
```

## Удаление

```bash
pip uninstall blinddeconv
```

---

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

---

## Интерактивное развертывание окружения

### Интерактивный установщик зависимостей (`install.py`)

Скрипт обеспечивает автоматизированную настройку рабочего окружения. Основные функции:
- создает и настраивает виртуальное окружение (по умолчанию `.venv`);
- проверяет совместимость версии Python и наличие системных библиотек;
- разрешает зависимости указанного профиля из `pyproject.toml` (секция `[tool.preflight.profiles]`);
- устанавливает недостающие пакеты и регистрирует их в файле состояния `.dependency_state.json`.

**Показать доступные профили:**
```bash
python install.py list-profiles
```

**Проверить статус зависимостей (без установки):**
```bash
python install.py check base
```

**Установить зависимости профиля (интерактивно):**
```bash
python install.py install base
```

**Автоматический режим (без запросов подтверждения):**
```bash
python install.py install base -y
```

**Указание пути к виртуальному окружению:**
Скрипт автоматически использует указанное окружение (создает его при отсутствии).
```bash
python install.py --venv my_env install base
```

### Интерактивное удаление зависимостей (`uninstall.py`)

Скрипт выполняет безопасное удаление пакетов, опираясь на историю установок (`.dependency_state.json`). Алгоритм учитывает пересечения зависимостей между профилями:
- удаляет пакеты, относящиеся *только* к выбранному профилю;
- сохраняет пакеты, которые используются другими активными профилями;
- поддерживает полную очистку окружения.

**Удалить зависимости конкретного профиля:**
```bash
python uninstall.py base
```
*Пример: если пакет `numpy` используется и в профиле `base`, и в профиле `dev`, то при удалении `base` пакет `numpy` останется в системе.*

**Полная очистка (сброс проекта):**
Удаляет каталог виртуального окружения и файл истории установок.
```bash
python uninstall.py --clean-all
```

---

## Настройка виртуального окружения (вручную)

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

---

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

---

## Проверка установки

```python
import blinddeconv
from blinddeconv.processing import Processing

print(f"Версия: {blinddeconv.processing.__version__}")

# Создание экземпляра Processing
p = Processing(images_folder="images/original", color=False)
print("Установка прошла успешно!")
```

---

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
