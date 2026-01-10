"""
HQMotionBlurRestoration - Python Wrapper

Original Implementation:
    gpl27 (GitHub)
    GitHub Repository: https://github.com/gpl27/deblur
    Language/Framework: Python

Reference Paper (if applicable):
    Based on method described in the repository without published paper

Algorithm Description:
    - Motion blur kernel estimation (often parametric) with restoration loop
    - Outputs a restored image (and kernel estimate when available)
    - Outputs a restored image (and kernel estimate when available)

This Python Wrapper Provides:
    - DeconvolutionAlgorithm-compatible Python class
    - Bridges and calls bundled original implementation
    - Basic parameter coercion and sanity checks

Wrapper Features:
    - Works with NumPy arrays and common image dtypes
    - Integrates into the BlindDeconvolution algorithm registry
    - Keeps bundled original source code untouched

Example:
    >>> from my_wrappers import HQMotionBlurRestoration
    >>> processor = HQMotionBlurRestoration()
    >>> result = processor.process(input_image)

Important Notes:
    1. Requires: numpy
    2. Original Python code remains unchanged
    3. This is purely an interface wrapper
    4. Check original repository for license information

Author: AUTHOR_PROJECT
Wrapper Version: 1.0.0
Original Author: gpl27
Original License: Unknown (check repository)
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, Tuple

import numpy as np

from algorithms.base import DeconvolutionAlgorithm
from .convolve import create_line_psf
from .deblur import computeLocalPrior, updatePsi, computeL, updatef

__all__ = ["HQMotionBlindDeconvolution", "HQMotionParams"]
