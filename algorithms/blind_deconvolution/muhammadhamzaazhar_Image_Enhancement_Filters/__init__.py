# https://github.com/muhammadhamzaazhar/Image-Enhancement-Filters
from __future__ import annotations
import os
from typing import Any, Literal

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "muhammadhamzaazhar_Image_Enhancement_Filters"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KERNEL_PLACEHOLDER = np.array([[0.0]])

class MuhammadhamzaazharImageEnhancementFilters(DeconvolutionAlgorithm):
	def __init__(
		self,
		method: Literal["blind", "lucy", "wiener"] = "blind",
		
		psf_size: int = 7,
		psf_sigma: float = 10.0,
		
		lucy_iterations: int = 10,
		lucy_noise_variance: float = 0.001,
		
		wiener_psf_size: int = 5,
		wiener_psf_sigma: float = 10.0,
		wiener_noise_variance: float = 0.01,
	):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.method = method

		self.psf_size = int(psf_size)
		self.psf_sigma = float(psf_sigma)

		self.lucy_iterations = int(lucy_iterations)
		self.lucy_noise_variance = float(lucy_noise_variance)

		self.wiener_psf_size = int(wiener_psf_size)
		self.wiener_psf_sigma = float(wiener_psf_sigma)
		self.wiener_noise_variance = float(wiener_noise_variance)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "method" in param:
			self.method = param["method"]

		if "psf_size" in param:
			self.psf_size = int(param["psf_size"])

		if "psf_sigma" in param:
			self.psf_sigma = float(param["psf_sigma"])

		if "lucy_iterations" in param:
			self.lucy_iterations = int(param["lucy_iterations"])

		if "lucy_noise_variance" in param:
			self.lucy_noise_variance = float(param["lucy_noise_variance"])

		if "wiener_psf_size" in param:
			self.wiener_psf_size = int(param["wiener_psf_size"])

		if "wiener_psf_sigma" in param:
			self.wiener_psf_sigma = float(param["wiener_psf_sigma"])

		if "wiener_noise_variance" in param:
			self.wiener_noise_variance = float(param["wiener_noise_variance"])

	def get_param(self) -> list[str, Any]:
		return [
			("method", self.method),
			("psf_size", self.psf_size),
			("psf_sigma", self.psf_sigma),
			("lucy_iterations", self.lucy_iterations),
			("lucy_noise_variance", self.lucy_noise_variance),
			("wiener_psf_size", self.wiener_psf_size),
			("wiener_psf_sigma", self.wiener_psf_sigma),
			("wiener_noise_variance", self.wiener_noise_variance),
		]

	def _prepare_image_gray(self, image: np.ndarray) -> np.ndarray:
		if image.ndim == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = image.astype(np.float64) / 255.0
		return image

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if self.method == "blind":
			return self._process_blind(image)
		elif self.method == "lucy":
			return self._process_lucy(image)
		elif self.method == "wiener":
			return self._process_wiener(image)
		else:
			raise ValueError("Wrrong method: ",self.method)

	def _process_blind(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		image_gray = self._prepare_image_gray(image)

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["I"] = I_mat
		self._eng.workspace["psf_size"] = float(self.psf_size)
		self._eng.workspace["psf_sigma"] = float(self.psf_sigma)

		self._eng.eval(
			"PSF = fspecial('gaussian', [psf_size psf_size], psf_sigma);",
			nargout=0,
		)

		self._eng.eval("[J, PSF_rec] = deconvblind(I, PSF);", nargout=0)

		J_mat = self._eng.workspace["J"]
		PSF_rec_mat = self._eng.workspace["PSF_rec"]

		J_np = np.array(J_mat, dtype=np.float64)
		J_np = np.clip(J_np, 0.0, 1.0)
		J_uint8 = (J_np * 255.0).astype(np.uint8)

		J_rgb = cv2.cvtColor(J_uint8, cv2.COLOR_GRAY2RGB)
		J_bgr = cv2.cvtColor(J_rgb, cv2.COLOR_RGB2BGR)

		kernel = np.array(PSF_rec_mat, dtype=np.float64)

		return J_bgr, kernel

	def _process_lucy(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		image_gray = self._prepare_image_gray(image)

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["I"] = I_mat

		self._eng.workspace["psf_size"] = float(self.psf_size)
		self._eng.workspace["psf_sigma"] = float(self.psf_sigma)
		self._eng.workspace["V"] = float(self.lucy_noise_variance)
		self._eng.workspace["NUMIT"] = float(self.lucy_iterations)

		self._eng.eval(
			"PSF = fspecial('gaussian', [psf_size psf_size], psf_sigma);",
			nargout=0,
		)
		self._eng.eval("Blurred = imfilter(I, PSF, 'symmetric', 'conv');", nargout=0)
		self._eng.eval("BlurredNoisy = imnoise(Blurred, 'gaussian', 0, V);", nargout=0)
		self._eng.eval("luc1 = deconvlucy(BlurredNoisy, PSF, NUMIT);", nargout=0)

		luc_mat = self._eng.workspace["luc1"]
		luc_np = np.array(luc_mat, dtype=np.float64)
		luc_np = np.clip(luc_np, 0.0, 1.0)
		luc_uint8 = (luc_np * 255.0).astype(np.uint8)

		luc_rgb = cv2.cvtColor(luc_uint8, cv2.COLOR_GRAY2RGB)
		luc_bgr = cv2.cvtColor(luc_rgb, cv2.COLOR_RGB2BGR)

		return luc_bgr, KERNEL_PLACEHOLDER

	def _process_wiener(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		image_gray = self._prepare_image_gray(image)

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["originalImage"] = I_mat

		self._eng.workspace["psf_size"] = float(self.wiener_psf_size)
		self._eng.workspace["psf_sigma"] = float(self.wiener_psf_sigma)
		self._eng.workspace["noise_var"] = float(self.wiener_noise_variance)

		self._eng.eval(
			"h = fspecial('gaussian', [psf_size psf_size], psf_sigma);",
			nargout=0,
		)
		self._eng.eval(
			"im_blurred = imfilter(originalImage, h, 'conv', 'symmetric');",
			nargout=0,
		)
		self._eng.eval(
			"im_blur = imnoise(im_blurred, 'gaussian', 0, noise_var);",
			nargout=0,
		)

		self._eng.eval("restoredImage = deconvwnr(im_blur, h);", nargout=0)

		restored_mat = self._eng.workspace["restoredImage"]
		restored_np = np.array(restored_mat, dtype=np.float64)
		restored_np = np.clip(restored_np, 0.0, 1.0)
		restored_uint8 = (restored_np * 255.0).astype(np.uint8)

		restored_rgb = cv2.cvtColor(restored_uint8, cv2.COLOR_GRAY2RGB)
		restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

		return restored_bgr, KERNEL_PLACEHOLDER

	def __del__(self):
		self._eng.quit()
