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

Author: AUTHOR_PROJECT
Wrapper Version: 1.0.0
Original Author: gpl27"""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, Tuple

import numpy as np

from src.algorithms.base import DeconvolutionAlgorithm
from .convolve import create_line_psf
from .deblur import computeLocalPrior, updatePsi, computeL, updatef

__all__ = ["HQMotionBlindDeconvolution", "HQMotionParams"]
