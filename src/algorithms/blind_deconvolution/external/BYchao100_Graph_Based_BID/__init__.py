"""
BYchao100GraphBasedBlindImageDeblurring - Python Wrapper

Original Implementation:
    BYchao100 (GitHub)
    GitHub Repository: https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring/
    Language/Framework: MATLAB

Reference Paper (if applicable):
    Based on method described in the repository without published paper

Algorithm Description:
    - Graph-based formulation/regularization for blind deblurring
    - Outputs a restored image (and kernel estimate when available)
    - Outputs a restored image (and kernel estimate when available)

Author: AUTHOR_PROJECT
Wrapper Version: 1.0.0
Original Author: BYchao100"""

from __future__ import annotations
import os
from time import time
from typing import Any, Tuple

import cv2
import numpy as np
from src.algorithms.octavewrapper import OctaveEngine

from src.algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
MATLAB_ROOT = os.path.join(SOURCE_PATH, "Graph_Based_BID")
MATLAB_CODE_PATH = os.path.join(MATLAB_ROOT, "Graph_Based_BID_v1.1")
OCTAVE_WRAPPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "octave_wrapper")
ALGORITHM_NAME = "BYchao100_Graph_Based_Blind_Image_Deblurring"


def _as_odd_positive(value: Any, *, default: int) -> int:
	try:
		parsed = int(value)
	except Exception:
		return int(default)
	if parsed <= 0:
		return int(default)
	return parsed if (parsed % 2 == 1) else (parsed + 1)


class BYchao100GraphBasedBlindImageDeblurring(DeconvolutionAlgorithm):
	def __init__(
		self,
		k_estimate_size: int = 69,
		border: int = 20,
		show_intermediate: bool = False,
	):
		super().__init__(ALGORITHM_NAME)

		self.k_estimate_size = _as_odd_positive(k_estimate_size, default=69)
		self.border = max(0, int(border))
		self.show_intermediate = bool(show_intermediate)

		self._oc = OctaveEngine.get_instance()
		self._oc.addpath(self._oc.genpath(MATLAB_CODE_PATH))
		self._oc.addpath(self._oc.genpath(OCTAVE_WRAPPER_PATH))

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "k_estimate_size" in param and param["k_estimate_size"] is not None:
			self.k_estimate_size = _as_odd_positive(
				param["k_estimate_size"], 
				default=self.k_estimate_size
			)
		if "border" in param and param["border"] is not None:
			self.border = max(0, int(param["border"]))
		if "show_intermediate" in param and param["show_intermediate"] is not None:
			self.show_intermediate = bool(param["show_intermediate"])

	def get_param(self) -> list[str, Any]:
		return [
			("k_estimate_size", self.k_estimate_size),
			("border", self.border),
			("show_intermediate", self.show_intermediate),
		]

	def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		else:
			image_bgr = image

		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		image_rgb_f = image_rgb.astype(np.float64) / 255.0
		image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

		border = int(self.border)
		if border * 2 >= min(image_gray.shape[:2]):
			border = 0
		if border > 0:
			image_gray = image_gray[border:-border, border:-border]

		
		start = time()

		I_FHLP, k_estimate = self._oc.feval(
            "run_bid",
            image_gray,
            image_rgb_f,
            float(self.k_estimate_size),
            float(1 if self.show_intermediate else 0),
            nout=2,
        )

		self.timer = time() - start

		restored_rgb = np.array(I_FHLP, dtype=np.float64)
		kernel = np.array(k_estimate, dtype=np.float64)

		restored_rgb = np.clip(restored_rgb, 0.0, 1.0)
		restored_uint8 = (restored_rgb * 255.0).round().astype(np.uint8)

		if restored_uint8.ndim == 2:
			restored_bgr = cv2.cvtColor(restored_uint8, cv2.COLOR_GRAY2BGR)
		else:
			restored_bgr = cv2.cvtColor(restored_uint8, cv2.COLOR_RGB2BGR)

		k_sum = float(kernel.sum())
		if k_sum > 0:
			kernel = kernel / k_sum

		return restored_bgr, kernel
