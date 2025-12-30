from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, Tuple

import numpy as np

from algorithms.base import DeconvolutionAlgorithm
from .convolve import create_line_psf
from .deblur import computeLocalPrior, updatePsi, computeL, updatef

__all__ = ["HQMotionBlindDeconvolution", "HQMotionParams"]
