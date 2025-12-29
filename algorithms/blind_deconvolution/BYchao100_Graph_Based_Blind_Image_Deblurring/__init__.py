# https://github.com/BYchao100/Graph-Based-Blind-Image-Deblurring/
from __future__ import annotations
import os
from time import time
from typing import Any, Tuple

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
MATLAB_ROOT = os.path.join(SOURCE_PATH, "Graph_Based_BID")
MATLAB_CODE_PATH = os.path.join(MATLAB_ROOT, "Graph_Based_BID_p1.1")
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

		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(MATLAB_CODE_PATH), nargout=0)
		self._eng.cd(MATLAB_CODE_PATH, nargout=0)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "k_estimate_size" in param and param["k_estimate_size"] is not None:
			self.k_estimate_size = _as_odd_positive(param["k_estimate_size"], default=self.k_estimate_size)
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

		self._eng.workspace["Y_b_py"] = matlab.double(image_gray.tolist())
		self._eng.workspace["I_py"] = matlab.double(image_rgb_f.tolist())
		self._eng.workspace["k_estimate_size_py"] = float(self.k_estimate_size)
		self._eng.workspace["show_intermediate_py"] = float(1 if self.show_intermediate else 0)

		start = time()
		self._eng.eval(
			"Y_b = Y_b_py;"
			"I_blur = I_py;"
			"k_estimate_size = k_estimate_size_py;"
			"show_intermediate = logical(show_intermediate_py);"
			"[k_estimate, ~] = bid_rgtv_c2f_cg(Y_b, k_estimate_size, show_intermediate);"
			"I_FHLP = I_blur;"
			"if ndims(I_blur) == 3 && size(I_blur, 3) == 3;"
			"  for c = 1:3;"
			"    I_FHLP(:,:,c) = Deconvolution_FHLP(I_blur(:,:,c), k_estimate);"
			"  end;"
			"else;"
			"  I_FHLP = Deconvolution_FHLP(I_blur, k_estimate);"
			"end;",
			nargout=0,
		)
		self.timer = time() - start

		restored_rgb = np.array(self._eng.workspace["I_FHLP"], dtype=np.float64)
		kernel = np.array(self._eng.workspace["k_estimate"], dtype=np.float64)

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

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass
