import os
import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


class LisiyaoATbnuLowRankKernel(DeconvolutionAlgorithm):
	def __init__(
		self,
		tx: float = 1e-2,
		tau: float = 1e-5,
		delta: float = 1e-5,
		imax: int = 5,
		ximax: int = 2,
		xjmax: int = 2,
		kmax: int = 3,
		rmax: int = 3,
		sigma: int = 1,
		lambda_: float = 80.0,
		threshold: float = 0.05,
		mu: float = 1.0,
		iterkrank: int = 10,
		kernel_size: int = 185,
		verbose: bool = True,
	):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.tx = float(tx)
		self.tau = float(tau)
		self.delta = float(delta)
		self.imax = int(imax)
		self.ximax = int(ximax)
		self.xjmax = int(xjmax)
		self.kmax = int(kmax)
		self.rmax = int(rmax)
		self.sigma = int(sigma)
		self.lambda_ = float(lambda_)
		self.threshold = float(threshold)
		self.mu = float(mu)
		self.iterkrank = int(iterkrank)
		self.kernel_size = int(kernel_size)
		self.verbose = bool(verbose)

	def change_param(self, param):
		if "tx" in param:
			self.tx = float(param["tx"])
		if "tau" in param:
			self.tau = float(param["tau"])
		if "delta" in param:
			self.delta = float(param["delta"])
		if "imax" in param:
			self.imax = int(param["imax"])
		if "ximax" in param:
			self.ximax = int(param["ximax"])
		if "xjmax" in param:
			self.xjmax = int(param["xjmax"])
		if "kmax" in param:
			self.kmax = int(param["kmax"])
		if "rmax" in param:
			self.rmax = int(param["rmax"])
		if "sigma" in param:
			self.sigma = int(param["sigma"])
		if "lambda" in param:
			self.lambda_ = float(param["lambda"])
		if "threshold" in param:
			self.threshold = float(param["threshold"])
		if "mu" in param:
			self.mu = float(param["mu"])
		if "iterkrank" in param:
			self.iterkrank = int(param["iterkrank"])
		if "kernel_size" in param:
			self.kernel_size = int(param["kernel_size"])
		if "verbose" in param:
			self.verbose = bool(param["verbose"])

	def get_param(self):
		return {
			"tx": self.tx,
			"tau": self.tau,
			"delta": self.delta,
			"imax": self.imax,
			"ximax": self.ximax,
			"xjmax": self.xjmax,
			"kmax": self.kmax,
			"rmax": self.rmax,
			"sigma": self.sigma,
			"lambda": self.lambda_,
			"threshold": self.threshold,
			"mu": self.mu,
			"iterkrank": self.iterkrank,
			"kernel_size": self.kernel_size,
			"verbose": self.verbose,
		}
	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		else:
			image_bgr = image

		image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
		image_gray = image_gray.astype(np.float64) / 255.0

		I_mat = matlab.double(image_gray.tolist())
		self._eng.workspace["y_py"] = I_mat
		self._eng.workspace["K_py"] = float(self.kernel_size)

		self._eng.workspace["tx"] = float(self.tx)
		self._eng.workspace["tau"] = float(self.tau)
		self._eng.workspace["delta"] = float(self.delta)
		self._eng.workspace["imax"] = float(self.imax)
		self._eng.workspace["ximax"] = float(self.ximax)
		self._eng.workspace["xjmax"] = float(self.xjmax)
		self._eng.workspace["kmax"] = float(self.kmax)
		self._eng.workspace["rmax"] = float(self.rmax)
		self._eng.workspace["sigma"] = float(self.sigma)
		self._eng.workspace["lambda_param"] = float(self.lambda_)
		self._eng.workspace["threshold"] = float(self.threshold)
		self._eng.workspace["mu"] = float(self.mu)
		self._eng.workspace["iterkrank"] = float(self.iterkrank)
		self._eng.workspace["verbose"] = float(1 if self.verbose else 0)

		self._eng.eval(
			"params = struct("
			"'tx', tx, "
			"'tau', tau, "
			"'delta', delta, "
			"'imax', imax, "
			"'ximax', ximax, "
			"'xjmax', xjmax, "
			"'kmax', kmax, "
			"'rmax', rmax, "
			"'sigma', sigma, "
			"'lambda', lambda_param, "
			"'threshold', threshold, "
			"'mu', mu, "
			"'iterkrank', iterkrank, "
			"'verbose', logical(verbose) "
			");",
			nargout=0,
		)

		self._eng.eval(
			"[x_py, k_py] = multiscaled_cry(y_py, K_py, params);",
			nargout=0,
		)

		x_mat = self._eng.workspace["x_py"]
		k_mat = self._eng.workspace["k_py"]

		x_np = np.array(x_mat, dtype=np.float64)
		x_np = np.clip(x_np, 0.0, 1.0)
		x_uint8 = (x_np * 255.0).astype(np.uint8)

		if x_uint8.ndim == 2:
			x_bgr = cv2.cvtColor(x_uint8, cv2.COLOR_GRAY2BGR)
		else:
			x_rgb = x_uint8
			x_bgr = cv2.cvtColor(x_rgb, cv2.COLOR_RGB2BGR)

		kernel = np.array(k_mat, dtype=np.float64)

		return x_bgr, kernel

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass
