# https://github.com/TobiasWolf-math/Blind-Deconvolution-MHDM
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "TobiasWolf_math_Blind_Deconvolution_MHDM"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KernelSpec = Tuple[int, int]


class TobiasWolfMathBlindDeconvolutionMHDM(DeconvolutionAlgorithm):
	def __init__(
		self,
		lambda0: float = 14e-5,
		mu0: float = 63e4,
		r: float = 1.0,
		s: float = 1e-1,
		tol: float = 1e-10,
		maxits: int = 30,
		stopping: float = 0.0,
		tau: float = 1.001,
		noise_level: Optional[float] = None,
	):
		super().__init__(ALGORITHM_NAME)

		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.lambda0 = float(lambda0)
		self.mu0 = float(mu0)
		self.r = float(r)
		self.s = float(s)
		self.tol = float(tol)
		self.maxits = int(maxits)
		self.stopping = float(stopping)
		self.tau = float(tau)
		self.noise_level = None if noise_level is None else float(noise_level)

	def _compute_stopping(self, image_gray: np.ndarray) -> float:
		if self.stopping > 0:
			return self.stopping

		if self.noise_level is not None and self.noise_level > 0:
			N = image_gray.size
			return self.tau * self.noise_level * np.sqrt(N)

		return 0.0

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "lambda0" in param and param["lambda0"] is not None:
			self.lambda0 = float(param["lambda0"])
		if "mu0" in param and param["mu0"] is not None:
			self.mu0 = float(param["mu0"])
		if "r" in param and param["r"] is not None:
			self.r = float(param["r"])
		if "s" in param and param["s"] is not None:
			self.s = float(param["s"])
		if "tol" in param and param["tol"] is not None:
			self.tol = float(param["tol"])
		if "maxits" in param and param["maxits"] is not None:
			self.maxits = int(param["maxits"])
		if "stopping" in param and param["stopping"] is not None:
			self.stopping = float(param["stopping"])
		if "tau" in param and param["tau"] is not None:
			self.tau = float(param["tau"])
		if "noise_level" in param:
			val = param["noise_level"]
			self.noise_level = None if val is None else float(val)

	def get_param(self) -> list[str, Any]:
		return [
			("lambda0", self.lambda0),
			("mu0", self.mu0),
			("r", self.r),
			("s", self.s),
			("tol", self.tol),
			("maxits", self.maxits),
			("stopping", self.stopping),
			("tau", self.tau),
			("noise_level", self.noise_level),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 3 and image.shape[2] == 3:
			image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			image_gray = image

		image_gray = image_gray.astype(np.float64)
		if image_gray.max() > 1.5:
			image_gray /= 255.0

		m, n = image_gray.shape
		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["f_py"] = I_mat
		self._eng.workspace["lambda0_py"] = float(self.lambda0)
		self._eng.workspace["mu0_py"] = float(self.mu0)
		self._eng.workspace["r_py"] = float(self.r)
		self._eng.workspace["s_py"] = float(self.s)
		self._eng.workspace["tol_py"] = float(self.tol)
		self._eng.workspace["maxits_py"] = float(self.maxits)

		stopping_val = self._compute_stopping(image_gray)
		self._eng.workspace["stopping_py"] = float(stopping_val)

		self._eng.eval(
			"""
			f = f_py;
			[m_py,n_py] = size(f);
			f_four = fft2(f);

			[rr, c] = ismember(f_four, conj(f_four));
			c = reshape(c, m_py*n_py, 1);
			M = [c, c(c)];
			sortedM = sort(M, 2);
			[~, uniqueIdx] = unique(sortedM, 'rows', 'stable');
			indices = M(uniqueIdx, :);

			[u_end_py, k_end_py, ~, ~, ~] = blind_deconvolution_MHDM( ...
				f, f_four, ...
				lambda0_py, mu0_py, ...
				r_py, s_py, ...
				tol_py, stopping_py, maxits_py, indices);
			""",
			nargout=0,
		)

		u_end_mat = self._eng.workspace["u_end_py"]
		k_end_mat = self._eng.workspace["k_end_py"]

		u_np = np.array(u_end_mat, dtype=np.float64)
		u_np = np.clip(u_np, 0.0, 1.0)
		u_uint8 = (u_np * 255.0).astype(np.uint8)
		u_bgr = cv2.cvtColor(u_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(k_end_mat, dtype=np.float64)

		return u_bgr, kernel

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass


__all__ = ["TobiasWolfMathBlindDeconvolutionMHDM"]
