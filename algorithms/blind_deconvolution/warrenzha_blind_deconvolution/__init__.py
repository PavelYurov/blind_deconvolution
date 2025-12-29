# https://github.com/warrenzha/blind-deconvolution
from __future__ import annotations

import os
from typing import Any, Literal

import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "warrenzha_blind_deconvolution"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KERNEL_PLACEHOLDER = np.array([[0]])

class WarrenzhaBlindDeconvolution(DeconvolutionAlgorithm):
	def __init__(
		self,
		length: int = 15,
		theta: int = 0,
		iter: int = 20,
		type: Literal[
		"motion",
		"average",
		"disk",
		"gaussian",
		"laplacian",
		"log",
		"prewitt",
		"sobel",
		"unsharp",
	] = "motion"):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.psf_len = length
		self.psf_theta = theta
		self.deconv_iter = iter
		self.kernel_type = type

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "psf_len" in param:
			self.psf_len = int(param["psf_len"])

		if "psf_theta" in param:
			self.psf_theta = float(param["psf_theta"])

		if "deconv_iter" in param:
			self.deconv_iter = int(param["deconv_iter"])

		if "kernel_type" in param:
			self.kernel_type = param["kernel_type"]

	def get_param(self) -> list[str, Any]:
		return [
			("kernel_type", self.kernel_type),
			("psf_len", self.psf_len),
			("psf_theta", self.psf_theta),
			("deconv_iter", self.deconv_iter),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		image_gray = image.astype(np.float64) / 255.0

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace['I'] = I_mat

		self._eng.workspace["kernel"] = self.kernel_type
		self._eng.workspace['len'] = float(self.psf_len)
		self._eng.workspace['theta'] = float(self.psf_theta)

		if self.kernel_type == "motion":
			self._eng.eval("PSF_init = fspecial(kernel, len, theta);", nargout=0)
		else:
			self._eng.eval("PSF_init = fspecial(kernel);", nargout=0)

		self._eng.eval(f"[J, PSF] = deconvblind(I, PSF_init, {int(self.deconv_iter)});", nargout=0)

		J_mat = self._eng.workspace['J']
		PSF_mat = self._eng.workspace['PSF']

		J_np = np.array(J_mat, dtype=np.float64)
		J_np = np.clip(J_np, 0.0, 1.0)

		J_uint8 = (J_np * 255.0).astype(np.uint8)
		J_rgb = cv2.cvtColor(J_uint8, cv2.COLOR_GRAY2RGB)
		J_bgr = cv2.cvtColor(J_rgb, cv2.COLOR_RGB2BGR)

		kernel = np.array(PSF_mat, dtype=np.float64)

		return J_bgr, kernel

	def __del__(self):
		self._eng.quit()
