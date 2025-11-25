import os
import cv2
import numpy as np
import matlab.engine
from typing import Literal

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KERNEL_PLACEHOLDER = np.array([[0.0]], dtype=np.float64)

class MaxMBImageRestorationWienerBlind(DeconvolutionAlgorithm):
	"""
	Обёртка над MATLAB-реализацией Максимилиано Барроса:
	- режим 'blind' использует deconvblind с гауссовским PSF, как в MAIN.m;
	- режим 'wiener_snr' использует wiener_filter_SNR с тем же PSF.

	Возвращает восстановленное изображение и ядро смаза (PSF).
	"""

	def __init__(
		self,
		mode: Literal["blind", "wiener_snr"] = "blind",
		psf_dim: int = 32,
		psf_sigma: float = 3.0,
		snr: float = 10.0,
		deconv_iter: int = 20,
	):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.mode = mode
		self.psf_dim = int(psf_dim)
		self.psf_sigma = float(psf_sigma)
		self.snr = float(snr)
		self.deconv_iter = int(deconv_iter)

	def change_param(self, param):
		if "mode" in param:
			self.mode = param["mode"]

		if "psf_dim" in param:
			self.psf_dim = int(param["psf_dim"])

		if "psf_sigma" in param:
			self.psf_sigma = float(param["psf_sigma"])

		if "snr" in param:
			self.snr = float(param["snr"])

		if "deconv_iter" in param:
			self.deconv_iter = int(param["deconv_iter"])

	def get_param(self):
		return {
			"mode": self.mode,
			"psf_dim": self.psf_dim,
			"psf_sigma": self.psf_sigma,
			"snr": self.snr,
			"deconv_iter": self.deconv_iter,
		}

	def _prepare_image_gray(self, image: np.ndarray) -> np.ndarray:
		if image.ndim == 3 and image.shape[2] == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image = image.astype(np.float64) / 255.0
		return image

	def _build_gaussian_psf_in_matlab(self):
		self._eng.workspace["dim_h"] = float(self.psf_dim)
		self._eng.workspace["var_h"] = float(self.psf_sigma)
		self._eng.eval(
			"h = fspecial('gaussian', [dim_h, dim_h], var_h);", nargout=0
		)

	def _run_wiener_snr(self, g_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		g_mat = matlab.double(g_np.tolist())
		self._eng.workspace["g"] = g_mat
		self._eng.workspace["SNR"] = float(self.snr)

		self._build_gaussian_psf_in_matlab()
		self._eng.eval("fe = wiener_filter_SNR(h, g, SNR);", nargout=0)

		fe_mat = self._eng.workspace["fe"]
		h_mat = self._eng.workspace["h"]

		fe_np = np.array(fe_mat, dtype=np.float64)
		fe_np = np.clip(fe_np, 0.0, 1.0)
		fe_uint8 = (fe_np * 255.0).astype(np.uint8)
		fe_bgr = cv2.cvtColor(fe_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(h_mat, dtype=np.float64)
		return fe_bgr, kernel

	def _run_blind(self, g_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		g_mat = matlab.double(g_np.tolist())
		self._eng.workspace["g"] = g_mat
		self._eng.workspace["num_iter"] = float(self.deconv_iter)

		self._build_gaussian_psf_in_matlab()
		self._eng.eval(
			"[fe_blind, PSF] = deconvblind(g, h, num_iter);", nargout=0
		)

		fe_mat = self._eng.workspace["fe_blind"]
		psf_mat = self._eng.workspace["PSF"]

		fe_np = np.array(fe_mat, dtype=np.float64)
		fe_np = np.clip(fe_np, 0.0, 1.0)
		fe_uint8 = (fe_np * 255.0).astype(np.uint8)
		fe_bgr = cv2.cvtColor(fe_uint8, cv2.COLOR_GRAY2BGR)

		kernel = np.array(psf_mat, dtype=np.float64)
		return fe_bgr, kernel

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		g_np = self._prepare_image_gray(image)

		if self.mode == "wiener_snr":
			restored, kernel = self._run_wiener_snr(g_np)
		else:
			restored, kernel = self._run_blind(g_np)

		if kernel is None or kernel.size == 0:
			kernel = KERNEL_PLACEHOLDER.copy()

		return restored, kernel

	def __del__(self):
		self._eng.quit()