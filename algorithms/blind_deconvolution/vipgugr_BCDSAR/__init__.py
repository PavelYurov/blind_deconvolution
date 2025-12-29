# https://github.com/vipgugr/BCDSAR
from __future__ import annotations
import os
from typing import Any

import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "vipgugr_BCDSAR"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

class VipgugrBCDSAR(DeconvolutionAlgorithm):
	def __init__(self, epsilon: float = 2.0e-5, niter: int = 1000, ns: int = 2):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.epsilon = float(epsilon)
		self.niter = int(niter)
		self.ns = int(ns)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "epsilon" in param:
			self.epsilon = float(param["epsilon"])

		if "niter" in param:
			self.niter = int(param["niter"])

		if "ns" in param:
			self.ns = int(param["ns"])

	def get_param(self) -> list[str, Any]:
		return [
			("epsilon", self.epsilon),
			("niter", self.niter),
			("ns", self.ns),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		elif image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image = image.astype(np.float64) / 255.0

		I_mat = matlab.double(image.tolist())
		self._eng.workspace["I"] = I_mat

		self._eng.eval("load('MLandini','RM');", nargout=0)

		self._eng.workspace["term"] = float(self.epsilon)
		self._eng.workspace["nitermax"] = float(self.niter)
		self._eng.workspace["ns"] = float(self.ns)

		self._eng.eval(
			"[CT, M, alpha, beta, gamma] = BCDHE(im2double(I), RM(:,1:ns), term, nitermax);",
			nargout=0,
		)

		self._eng.eval("[m,n,nc] = size(I);", nargout=0)
		self._eng.eval("ns = size(M,2);", nargout=0)
		self._eng.eval("concentrations = reshape(CT', m, n, ns);", nargout=0)

		self._eng.eval("Hrec_OD  = reshape((M(:,1)*CT(1,:))', m, n, nc);", nargout=0)
		self._eng.eval("Hrec_RGB = OD2intensities(Hrec_OD);", nargout=0)

		self._eng.eval("Erec_OD  = reshape((M(:,2)*CT(2,:))', m, n, nc);", nargout=0)
		self._eng.eval("Erec_RGB = OD2intensities(Erec_OD);", nargout=0)

		self._eng.eval("OD_rec = reshape((M * CT)', m, n, nc);", nargout=0)
		self._eng.eval("I_rec = OD2intensities(OD_rec);", nargout=0)

		I_rec_mat = self._eng.workspace["I_rec"]
		I_rec_rgb = np.array(I_rec_mat, dtype=np.float64)
		I_rec_rgb = np.clip(I_rec_rgb, 0.0, 1.0)

		I_rec_uint8 = (I_rec_rgb * 255.0).astype(np.uint8)
		I_rec_bgr = cv2.cvtColor(I_rec_uint8, cv2.COLOR_RGB2BGR)

		M_mat = self._eng.workspace["M"]
		kernel = np.array(M_mat, dtype=np.float64)

		return I_rec_bgr, kernel

	def __del__(self):
		self._eng.quit()
