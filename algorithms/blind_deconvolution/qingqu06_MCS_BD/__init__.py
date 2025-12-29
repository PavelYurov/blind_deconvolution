# https://github.com/qingqu06/MCS-BD
from __future__ import annotations
import os
from typing import Any, Tuple

import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "qingqu06_MCS_BD"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

class Qingqu06MCSBD(DeconvolutionAlgorithm):
	def __init__(
		self,
		kernel_size: Tuple[int, int] = (10, 10),
		mu: float = 1e-2,
		tau: float = 1e-2,
		max_iter: int = 200,
		use_linesearch: bool = True,
		use_rounding: bool = False,
	):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.kernel_height, self.kernel_width = int(kernel_size[0]), int(kernel_size[1])
		self.mu = float(mu)
		self.tau = float(tau)
		self.max_iter = int(max_iter)
		self.use_linesearch = bool(use_linesearch)
		self.use_rounding = bool(use_rounding)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "kernel_height" in param:
			self.kernel_height = int(param["kernel_height"])
		if "kernel_width" in param:
			self.kernel_width = int(param["kernel_width"])
		if "kernel_size" in param and isinstance(param["kernel_size"], (list, tuple)):
			kh, kw = param["kernel_size"]
			self.kernel_height = int(kh)
			self.kernel_width = int(kw)

		if "mu" in param:
			self.mu = float(param["mu"])
		if "tau" in param:
			self.tau = float(param["tau"])
		if "max_iter" in param:
			self.max_iter = int(param["max_iter"])
		if "use_linesearch" in param:
			self.use_linesearch = bool(param["use_linesearch"])
		if "use_rounding" in param:
			self.use_rounding = bool(param["use_rounding"])

	def get_param(self) -> list[str, Any]:
		return [
			("kernel_height", self.kernel_height),
			("kernel_width", self.kernel_width),
			("kernel_size", (self.kernel_height, self.kernel_width)),
			("mu", self.mu),
			("tau", self.tau),
			("max_iter", self.max_iter),
			("use_linesearch", self.use_linesearch),
			("use_rounding", self.use_rounding),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

		if image.ndim == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		else:
			gray = image

		gray = gray.astype(np.float64) / 255.0

		if gray.shape != (self.kernel_height, self.kernel_width):
			gray_small = cv2.resize(
				gray,
				(self.kernel_width, self.kernel_height),
				interpolation=cv2.INTER_AREA,
			)
		else:
			gray_small = gray
			
		Y_np = gray_small.reshape(self.kernel_height, self.kernel_width, 1)
		Y_mat = matlab.double(Y_np.tolist())
		self._eng.workspace["Y"] = Y_mat

		self._eng.workspace["mu"] = float(self.mu)
		self._eng.workspace["tau"] = float(self.tau)
		self._eng.workspace["MaxIter"] = float(self.max_iter)
		self._eng.workspace["use_linesearch"] = bool(self.use_linesearch)
		self._eng.workspace["use_rounding"] = bool(self.use_rounding)

		self._eng.eval(
			"""
[n1,n2,p] = size(Y);

% Предобработка данных (как в test_2D.m)
V = (1/(n1*n2*p) * sum(abs(fft2(Y)).^2, 3)).^(-1/2);
Y_p = ifft2(bsxfun(@times, fft2(Y), V));

% Выбор функции потерь (берём Huber, как в примере)
f = func_huber_2D(Y_p, mu);

% Настройка опций оптимизации
opts = struct();
opts.islinesearch = logical(use_linesearch);
opts.isprint = false;
opts.tol = 1e-10;
opts.tau = tau;
opts.MaxIter = MaxIter;
opts.rounding = logical(use_rounding);
opts.NumReinit = 1;
opts.truth = false;

Z_init = randn(n1, n2);
opts.Z_init = Z_init / norm(Z_init(:));

% Фаза 1: Riemannian gradient descent
[Z_r, F_val, Err] = grad_descent_2D(f, opts);

% Фаза 2: rounding (опционально)
if opts.rounding
	opts_r = struct();
	opts_r.MaxIter = 200;
	R = Z_r;
	f_l1 = func_l1_2D(Y_p);
	Z = rounding_2D(f_l1, R, opts_r);
else
	Z = Z_r;
end

% Восстановление ядра (preconditioned Z)
precond_Z = real(ifft2(V .* fft2(Z)));
			""",
			nargout=0,
		)

		precond_Z_mat = self._eng.workspace["precond_Z"]
		kernel = np.array(precond_Z_mat, dtype=np.float64)

		kernel = np.maximum(kernel, 0.0)
		s = kernel.sum()
		if s > 0:
			kernel /= s

		restored = image.copy()

		return restored, kernel

	def __del__(self):
		self._eng.quit()
