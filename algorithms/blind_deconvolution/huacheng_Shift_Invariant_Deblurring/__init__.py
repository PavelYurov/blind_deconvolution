# https://github.com/huacheng/Shift-Invariant-Deblurring
from __future__ import annotations
import os
from typing import Any, Sequence

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "huacheng_Shift_Invariant_Deblurring"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


def _as_matlab_row_vector(values: Sequence[float]) -> "matlab.double":
	return matlab.double([list(map(float, values))])


class HuachengShiftInvariantDeblurring(DeconvolutionAlgorithm):
	def __init__(
		self,
		lambda_coarse: float = 0.001,
		gamma: float = 10.0,
		final_lambda: float = 0.05,
		ratio: Sequence[float] = (10, 7, 5, 3, 2, 1.5, 1),
		ks: Sequence[int] = (1, 2, 4, 7, 9, 12, 17),
		gaussian_sigma: float = 2.0,
		edgetaper_iters: int = 3,
		perona_iter: int = 5,
		shock_iter: int = 5,
		shock_dt: float = 0.1,
		shock_h: float = 1.0,
	):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.lambda_coarse = float(lambda_coarse)
		self.gamma = float(gamma)
		self.final_lambda = float(final_lambda)
		self.ratio = tuple(float(x) for x in ratio)
		self.ks = tuple(int(x) for x in ks)
		self.gaussian_sigma = float(gaussian_sigma)
		self.edgetaper_iters = int(edgetaper_iters)
		self.perona_iter = int(perona_iter)
		self.shock_iter = int(shock_iter)
		self.shock_dt = float(shock_dt)
		self.shock_h = float(shock_h)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "lambda_coarse" in param:
			self.lambda_coarse = float(param["lambda_coarse"])
		if "gamma" in param:
			self.gamma = float(param["gamma"])
		if "final_lambda" in param:
			self.final_lambda = float(param["final_lambda"])
		if "ratio" in param:
			self.ratio = tuple(float(x) for x in param["ratio"])
		if "ks" in param:
			self.ks = tuple(int(x) for x in param["ks"])
		if "gaussian_sigma" in param:
			self.gaussian_sigma = float(param["gaussian_sigma"])
		if "edgetaper_iters" in param:
			self.edgetaper_iters = int(param["edgetaper_iters"])
		if "perona_iter" in param:
			self.perona_iter = int(param["perona_iter"])
		if "shock_iter" in param:
			self.shock_iter = int(param["shock_iter"])
		if "shock_dt" in param:
			self.shock_dt = float(param["shock_dt"])
		if "shock_h" in param:
			self.shock_h = float(param["shock_h"])

	def get_param(self) -> list[str, Any]:
		return [
			("lambda_coarse", self.lambda_coarse),
			("gamma", self.gamma),
			("final_lambda", self.final_lambda),
			("ratio", list(self.ratio)),
			("ks", list(self.ks)),
			("gaussian_sigma", self.gaussian_sigma),
			("edgetaper_iters", self.edgetaper_iters),
			("perona_iter", self.perona_iter),
			("shock_iter", self.shock_iter),
			("shock_dt", self.shock_dt),
			("shock_h", self.shock_h),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 3:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			image_gray = image

		image_gray = image_gray.astype(np.float64) / 255.0

		self._eng.workspace["I_py"] = matlab.double(image_gray.tolist())

		params = {
			"lambda_coarse": float(self.lambda_coarse),
			"gamma": float(self.gamma),
			"final_lambda": float(self.final_lambda),
			"ratio": _as_matlab_row_vector(self.ratio),
			"ks": _as_matlab_row_vector(self.ks),
			"gaussian_sigma": float(self.gaussian_sigma),
			"edgetaper_iters": float(self.edgetaper_iters),
			"perona_iter": float(self.perona_iter),
			"shock_iter": float(self.shock_iter),
			"shock_dt": float(self.shock_dt),
			"shock_h": float(self.shock_h),
		}

		self._eng.workspace["params_py"] = params

		self._eng.eval("[I_latent_py, k_py] = deblur_wrapper(I_py, params_py);", nargout=0)

		I_latent = np.array(self._eng.workspace["I_latent_py"], dtype=np.float64)
		I_latent = np.clip(I_latent, 0.0, 1.0)
		I_latent_uint8 = (I_latent * 255.0).astype(np.uint8)
		I_latent_bgr = cv2.cvtColor(I_latent_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(self._eng.workspace["k_py"], dtype=np.float64)

		return I_latent_bgr, kernel

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass
