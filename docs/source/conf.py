"""
Sphinx configuration for auto-documentation.

Usage:
    sphinx-apidoc -o . ..
    sphinx-build -b html . _build/html

Supports:
    - reStructuredText (.rst) via built-in Sphinx parsers
    - Markdown (.md) via MyST-Parser
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# Project information 
project = "BlindDeconvolution"
copyright = "2024-2026, Юров П.И., Беззаборов А.А., Куропатов К.Л., Малыш Я.В."
author = "Юров П.И., Беззаборов А.А., Куропатов К.Л., Малыш Я.В."
release = "1.0.0"

# General configuration

extensions = [
    "sphinx.ext.autodoc",       
    "sphinx.ext.napoleon",      
    "sphinx.ext.viewcode",     
    "sphinx.ext.intersphinx",  
]

# Markdown support via MyST-Parser (optional)
try:
    import myst_parser 
    extensions.append("myst_parser")

    source_suffix = {
        ".rst": "restructuredtext",
        ".md": "markdown",
    }

    myst_enable_extensions = [
        "colon_fence",          
        "deflist",              
        "fieldlist",           
        "tasklist",             
        "substitution",         
    ]
    myst_heading_anchors = 3

except ImportError:
    pass

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "ru"

# Options for HTML output 
html_theme = "sphinx_rtd_theme" 
html_static_path = ["_static"]

# Fallback
try:
    import sphinx_rtd_theme  
except ImportError:
    html_theme = "alabaster"

# Options for autodoc 

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "tensorflow",
    "keras",
    "cv2",
    "matlab",
    "matlab.engine",
]

# Options for intersphinx

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

