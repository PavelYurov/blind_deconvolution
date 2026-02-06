"""
Конфигурация Sphinx для автодокументации.

Использование:
    sphinx-apidoc -o . ..
    sphinx-build -b html . _build/html

Поддерживает:
    - reStructuredText (.rst) через встроенные парсеры Sphinx
    - Markdown (.md) через MyST-Parser
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "BlindDeconvolution"
copyright = "2024-2026, Юров П.И., Беззаборов А.А., Куропатов К.Л., Малыш Я.В."
author = "Юров П.И., Беззаборов А.А., Куропатов К.Л., Малыш Я.В."
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # Автодокументация из docstrings
    "sphinx.ext.napoleon",      # Поддержка NumPy/Google style docstrings
    "sphinx.ext.viewcode",      # Ссылки на исходный код
    "sphinx.ext.intersphinx",   # Ссылки на внешнюю документацию
    "myst_parser",              # Поддержка Markdown через MyST
]

# -- MyST-Parser configuration -----------------------------------------------

# Поддерживаемые форматы файлов
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Расширения MyST для расширенного Markdown
myst_enable_extensions = [
    "colon_fence",          # ::: вместо ``` для директив
    "deflist",              # Списки определений
    "fieldlist",            # Списки полей
    "tasklist",             # Чекбоксы [ ] / [x]
    "substitution",         # Подстановки {{ }}
]

# Глубина автоматической генерации якорей для заголовков
myst_heading_anchors = 3

# Napoleon settings для NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Язык документации
language = "ru"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"  # Read the Docs тема (pip install sphinx-rtd-theme)
html_static_path = ["_static"]

# Fallback если тема не установлена
try:
    import sphinx_rtd_theme  # noqa: F401
except ImportError:
    html_theme = "alabaster"

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Не импортировать модули при документировании (для тяжёлых зависимостей)
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "cv2",
    "matlab",
    "matlab.engine",
]

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

