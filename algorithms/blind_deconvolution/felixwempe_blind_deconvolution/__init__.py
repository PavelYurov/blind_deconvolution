# https://github.com/felixwempe/blind_deconvolution/
from __future__ import annotations

import os
from typing import Any

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

ALGORITHM_NAME = "felixwempe_blind_deconvolution"
SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


class FelixwempeBlindDeconvolution(DeconvolutionAlgorithm):
	def __init__(
		self,
		kernel_size: int = 15,
		wavelet: str = "haar",
		wavelet_level: int = 1,
		coeff_threshold: float = 0.0,
		cvx_quiet: bool = True,
	):
		super().__init__(ALGORITHM_NAME)
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.kernel_size = int(kernel_size)
		self.wavelet = str(wavelet)
		self.wavelet_level = int(wavelet_level)
		self.coeff_threshold = float(coeff_threshold)
		self.cvx_quiet = bool(cvx_quiet)

	def change_param(self, param: Any):
		if not isinstance(param, dict):
			return

		if "kernel_size" in param:
			self.kernel_size = int(param["kernel_size"])
		if "wavelet" in param:
			self.wavelet = str(param["wavelet"])
		if "wavelet_level" in param:
			self.wavelet_level = int(param["wavelet_level"])
		if "coeff_threshold" in param:
			self.coeff_threshold = float(param["coeff_threshold"])
		if "cvx_quiet" in param:
			self.cvx_quiet = bool(param["cvx_quiet"])

	def get_param(self) -> list[str, Any]:
		return [
			("kernel_size", self.kernel_size),
			("wavelet", self.wavelet),
			("wavelet_level", self.wavelet_level),
			("coeff_threshold", self.coeff_threshold),
			("cvx_quiet", self.cvx_quiet),
		]

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		else:
			image_bgr = image

		image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
		I_mat = matlab.double(image_gray.tolist())

		self._eng.workspace["I_py"] = I_mat
		self._eng.workspace["kernel_size_py"] = float(self.kernel_size)
		self._eng.workspace["wavelet_py"] = self.wavelet
		self._eng.workspace["wavelet_level_py"] = float(self.wavelet_level)
		self._eng.workspace["coeff_threshold_py"] = float(self.coeff_threshold)
		self._eng.workspace["cvx_quiet_py"] = float(1 if self.cvx_quiet else 0)

		self._eng.eval(
			"""
			I = I_py;
			[s1, s2] = size(I);
			y = I(:);
			L = numel(y);

			k = round(kernel_size_py);
			k = max(1, min(k, min(s1, s2)));
			mask = zeros(s1, s2);
			c1 = floor(s1/2) + 1;
			c2 = floor(s2/2) + 1;
			r1 = max(1, c1 - floor(k/2));
			r2 = min(s1, r1 + k - 1);
			r1 = max(1, r2 - k + 1);
			c1s = max(1, c2 - floor(k/2));
			c2s = min(s2, c1s + k - 1);
			c1s = max(1, c2s - k + 1);
			mask(r1:r2, c1s:c2s) = 1;
			idxB = find(mask(:));
			K = numel(idxB);
			B = sparse(L, K);
			for j = 1:K
				B(idxB(j), j) = 1;
			end

			wave = wavelet_py;
			lev = round(wavelet_level_py);
			[Cw, S] = wavedec2(reshape(y, s1, s2), lev, wave);
			thr = coeff_threshold_py;
			if thr > 0
				idxC = find(abs(Cw) > thr);
			else
				idxC = find(Cw ~= 0);
			end
			N = numel(idxC);
			C = sparse(L, N);
			m_gt = zeros(N, 1);
			for j = 1:N
				C(idxC(j), j) = 1;
				m_gt(j) = Cw(idxC(j));
			end

			A = zeros(L, K*N);
			B_full = full(B);
			for i = 1:N
				Del = circular(C(:, i));
				A(:, (i-1)*K+1:i*K) = Del * B_full;
			end

			if cvx_quiet_py
				cvx_quiet(true);
			else
				cvx_quiet(false);
			end
			cvx_begin
				variable X(K, N)
				minimise( norm_nuc(X) )
				subject to
					A * X(:) == y
			cvx_end

			[U, Sv, V] = svd(full(X), 'econ');
			sigma1 = Sv(1, 1);
			h_opt = U(:, 1) * sqrt(sigma1);
			m_opt = V(:, 1) * sqrt(sigma1);

			C_dec = C * m_opt;
			x_dec = waverec2(C_dec, S, wave);

			w_vec = B * h_opt;
			w_img = reshape(w_vec, s1, s2);
			kernel = w_img(r1:r2, c1s:c2s);
			""",
			nargout=0,
		)

		x_mat = self._eng.workspace["x_dec"]
		k_mat = self._eng.workspace["kernel"]

		x_np = np.array(x_mat, dtype=np.float64)
		x_np = np.nan_to_num(x_np, nan=0.0, posinf=1.0, neginf=0.0)
		x_np = np.clip(x_np, 0.0, 1.0)
		x_uint8 = (x_np * 255.0).astype(np.uint8)
		x_bgr = cv2.cvtColor(x_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(k_mat, dtype=np.float64)

		return x_bgr, kernel

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass
