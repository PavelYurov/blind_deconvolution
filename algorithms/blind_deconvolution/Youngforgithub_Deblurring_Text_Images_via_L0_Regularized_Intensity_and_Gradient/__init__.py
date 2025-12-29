import os
from dataclasses import dataclass, asdict
from typing import Any
import matlab.engine

import cv2
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

@dataclass
class _Params:
	# Kernel estimation
	gamma_correct: float = 2.2
	kernel_size: int = 35
	lambda_pixel: float = 4e-3
	lambda_grad: float = 4e-3
	xk_iter: int = 5
	k_thresh: int = 20

	# Non-blind deconvolution
	lambda_tv: float = 2e-3
	lambda_l0: float = 2e-4
	weight_ring: float = 1.0


class YoungforgithubL0TextDeblurring(DeconvolutionAlgorithm):
	def __init__(
		self,
		gamma_correct: float = 2.2,
		kernel_size: int = 35,
		lambda_pixel: float = 4e-3,
		lambda_grad: float = 4e-3,
		xk_iter: int = 5,
		k_thresh: int = 20,
		lambda_tv: float = 2e-3,
		lambda_l0: float = 2e-4,
		weight_ring: float = 1.0,
	):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)
		self._eng.eval("close all; clc;", nargout=0)
		self._eng.eval("set(0,'DefaultFigureVisible','off');", nargout=0)

		self.params = _Params(
			gamma_correct=float(gamma_correct),
			kernel_size=int(kernel_size),
			lambda_pixel=float(lambda_pixel),
			lambda_grad=float(lambda_grad),
			xk_iter=int(xk_iter),
			k_thresh=int(k_thresh),
			lambda_tv=float(lambda_tv),
			lambda_l0=float(lambda_l0),
			weight_ring=float(weight_ring),
		)

	def change_param(self, param: dict[str, Any]):
		for key, value in (param or {}).items():
			if not hasattr(self.params, key):
				continue
			current = getattr(self.params, key)
			if isinstance(current, int):
				setattr(self.params, key, int(value))
			elif isinstance(current, float):
				setattr(self.params, key, float(value))
			else:
				setattr(self.params, key, value)

	def get_param(self):
		return asdict(self.params)

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		import matlab  # type: ignore

		if image.ndim == 3 and image.shape[2] == 3:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			image_gray = image

		image_gray = image_gray.astype(np.float64) / 255.0

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["y"] = I_mat

		self._eng.workspace["lambda_pixel"] = float(self.params.lambda_pixel)
		self._eng.workspace["lambda_grad"] = float(self.params.lambda_grad)

		self._eng.workspace["kernel_size"] = float(self.params.kernel_size)
		self._eng.workspace["gamma_correct"] = float(self.params.gamma_correct)
		self._eng.workspace["xk_iter"] = float(self.params.xk_iter)
		self._eng.workspace["k_thresh"] = float(self.params.k_thresh)

		self._eng.eval(
			"opts = struct("
			"'prescale', 1, "
			"'xk_iter', xk_iter, "
			"'gamma_correct', gamma_correct, "
			"'k_thresh', k_thresh, "
			"'kernel_size', kernel_size"
			");",
			nargout=0,
		)

		self._eng.eval("[kernel, interim_latent] = blind_deconv(y, lambda_pixel, lambda_grad, opts);", nargout=0)

		self._eng.workspace["lambda_tv"] = float(self.params.lambda_tv)
		self._eng.workspace["lambda_l0"] = float(self.params.lambda_l0)
		self._eng.workspace["weight_ring"] = float(self.params.weight_ring)
		self._eng.eval(
			"Latent = ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring);",
			nargout=0,
		)

		latent_mat = self._eng.workspace["Latent"]
		kernel_mat = self._eng.workspace["kernel"]

		latent_np = np.array(latent_mat, dtype=np.float64)
		latent_np = np.clip(latent_np, 0.0, 1.0)
		latent_uint8 = (latent_np * 255.0).astype(np.uint8)

		if latent_uint8.ndim == 2:
			latent_bgr = cv2.cvtColor(latent_uint8, cv2.COLOR_GRAY2BGR)
		else:
			latent_bgr = cv2.cvtColor(latent_uint8, cv2.COLOR_RGB2BGR)

		kernel_np = np.array(kernel_mat, dtype=np.float64)

		return latent_bgr, kernel_np

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass
