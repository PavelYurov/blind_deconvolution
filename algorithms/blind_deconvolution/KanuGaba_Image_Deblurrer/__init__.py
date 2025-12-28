import os
from typing import Optional

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


class KanuGabaImageDeblurrer(DeconvolutionAlgorithm):
	def __init__(
		self,
		length: int = 40,
		theta: float = 20.0,
		noise_var: float = 1e-4,
		nsr: Optional[float] = None,
	):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.length = int(length)
		self.theta = float(theta)
		self.noise_var = float(noise_var)
		self.nsr = None if nsr is None else float(nsr)

	def change_param(self, param):
		if "length" in param:
			self.length = int(param["length"])
		if "theta" in param:
			self.theta = float(param["theta"])
		if "noise_var" in param:
			self.noise_var = float(param["noise_var"])
		if "nsr" in param:
			self.nsr = None if param["nsr"] is None else float(param["nsr"])

	def get_param(self):
		return {
			"length": self.length,
			"theta": self.theta,
			"noise_var": self.noise_var,
			"nsr": self.nsr,
		}

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image_gray = image
		else:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image_gray = image_gray.astype(np.float64) / 255.0

		self._eng.workspace["I"] = matlab.double(image_gray.tolist())
		self._eng.workspace["LEN"] = float(self.length)
		self._eng.workspace["THETA"] = float(self.theta)
		self._eng.workspace["noise_var"] = float(self.noise_var)
		self._eng.workspace["nsr_override"] = float(self.nsr) if self.nsr is not None else float("nan")

		self._eng.eval("[J, PSF] = kanugaba_deconvwnr(I, LEN, THETA, noise_var, nsr_override);", nargout=0)

		J_mat = self._eng.workspace["J"]
		PSF_mat = self._eng.workspace["PSF"]

		J_np = np.array(J_mat, dtype=np.float64)
		J_np = np.clip(J_np, 0.0, 1.0)
		J_uint8 = (J_np * 255.0).astype(np.uint8)
		J_bgr = cv2.cvtColor(J_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(PSF_mat, dtype=np.float64)

		return J_bgr, kernel

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass

