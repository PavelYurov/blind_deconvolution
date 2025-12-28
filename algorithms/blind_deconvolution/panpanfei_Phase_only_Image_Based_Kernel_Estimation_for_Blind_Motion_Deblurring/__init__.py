import os
from typing import Any

import cv2
import matlab.engine
import numpy as np

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")


class PanpanfeiPhaseOnlyKernelEstimationBlindMotionDeblurring(DeconvolutionAlgorithm):
	def __init__(
		self,
		needsys: int = 0,
		motion: int = 1,
		fast: int = 1,
		kernel_size: int = 35,
		auto_size: int | None = None,
		iter_num: int = 10,
		lambda_grad: float | None = None,
		lambda_l0: float | None = None,
		lambda_tv: float | None = None,
		synth_len: float = 30.0,
		synth_theta: float = 30.0,
		ifdisplay: int = 0,
	):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

		self.needsys = int(needsys)
		self.motion = int(motion)
		self.fast = int(fast)
		self.kernel_size = int(kernel_size)
		self.auto_size = None if auto_size is None else int(auto_size)
		self.iter_num = int(iter_num)
		self.lambda_grad = None if lambda_grad is None else float(lambda_grad)
		self.lambda_l0 = None if lambda_l0 is None else float(lambda_l0)
		self.lambda_tv = None if lambda_tv is None else float(lambda_tv)
		self.synth_len = float(synth_len)
		self.synth_theta = float(synth_theta)
		self.ifdisplay = int(ifdisplay)

	def change_param(self, param: dict[str, Any]):
		if "needsys" in param:
			self.needsys = int(param["needsys"])
		if "motion" in param:
			self.motion = int(param["motion"])
		if "fast" in param:
			self.fast = int(param["fast"])
		if "kernel_size" in param:
			self.kernel_size = int(param["kernel_size"])
		if "auto_size" in param:
			self.auto_size = None if param["auto_size"] is None else int(param["auto_size"])
		if "iter_num" in param:
			self.iter_num = int(param["iter_num"])
		if "lambda_grad" in param:
			self.lambda_grad = None if param["lambda_grad"] is None else float(param["lambda_grad"])
		if "lambda_l0" in param:
			self.lambda_l0 = None if param["lambda_l0"] is None else float(param["lambda_l0"])
		if "lambda_tv" in param:
			self.lambda_tv = None if param["lambda_tv"] is None else float(param["lambda_tv"])
		if "synth_len" in param:
			self.synth_len = float(param["synth_len"])
		if "synth_theta" in param:
			self.synth_theta = float(param["synth_theta"])
		if "ifdisplay" in param:
			self.ifdisplay = int(param["ifdisplay"])

	def get_param(self):
		return {
			"needsys": self.needsys,
			"motion": self.motion,
			"fast": self.fast,
			"kernel_size": self.kernel_size,
			"auto_size": self.auto_size,
			"iter_num": self.iter_num,
			"lambda_grad": self.lambda_grad,
			"lambda_l0": self.lambda_l0,
			"lambda_tv": self.lambda_tv,
			"synth_len": self.synth_len,
			"synth_theta": self.synth_theta,
			"ifdisplay": self.ifdisplay,
		}

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		if image.ndim == 2:
			image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		if image.shape[2] == 4:
			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_rgb = image_rgb.astype(np.float64) / 255.0

		I_mat = matlab.double(image_rgb.tolist())
		self._eng.workspace["blur_imagec"] = I_mat

		self._eng.eval("S = load('para.mat'); para = S.para;", nargout=0)
		self._eng.workspace["needsys"] = float(self.needsys)
		self._eng.workspace["motionb"] = float(self.motion)
		self._eng.workspace["fastb"] = float(self.fast)
		self._eng.workspace["kernel_size"] = float(self.kernel_size)
		self._eng.workspace["iter_num"] = float(self.iter_num)
		self._eng.workspace["synth_len"] = float(self.synth_len)
		self._eng.workspace["synth_theta"] = float(self.synth_theta)
		self._eng.workspace["ifdisplay"] = float(self.ifdisplay)

		self._eng.eval("para.needsys = needsys;", nargout=0)
		self._eng.eval("para.motion = motionb;", nargout=0)
		self._eng.eval("para.fast = fastb;", nargout=0)
		self._eng.eval("para.iter_num = iter_num;", nargout=0)
		self._eng.eval("para.kernel_size = kernel_size;", nargout=0)

		if self.lambda_grad is not None:
			self._eng.workspace["lambda_grad"] = float(self.lambda_grad)
			self._eng.eval("para.lambda_grad = lambda_grad;", nargout=0)
		if self.lambda_l0 is not None:
			self._eng.workspace["lambda_l0"] = float(self.lambda_l0)
			self._eng.eval("para.lambda_l0 = lambda_l0;", nargout=0)
		if self.lambda_tv is not None:
			self._eng.workspace["lambda_tv"] = float(self.lambda_tv)
			self._eng.eval("para.lambda_tv = lambda_tv;", nargout=0)

		auto_size = max(30, self.kernel_size) if self.auto_size is None else self.auto_size
		self._eng.workspace["auto_size"] = float(auto_size)

		self._eng.eval(
			"""
			if size(blur_imagec, 3) == 1
				blur_imagec = repmat(blur_imagec, [1 1 3]);
			end
			[blur, blurc] = data2blurim(blur_imagec, synth_len, synth_theta, para.needsys);
			[p_aut, text_aut, centrh, centrw] = im2auto_corr(blur, auto_size, ifdisplay);
			[blurlen, bluranle] = auto2motion(text_aut);
			sn = floor(para.kernel_size/2);
			if para.motion == 1
				cPSF = fspecial('motion', blurlen, bluranle);
			else
				sn = max(sn, floor(blurlen/2));
				cPSF = text_aut(centrh-sn:centrh+sn, centrw-sn:centrw+sn);
			end
			[PSF, deblurring] = kernelwithLatent(blurc, cPSF, para);
			""",
			nargout=0,
		)

		deblurring = np.array(self._eng.workspace["deblurring"], dtype=np.float64)
		PSF = np.array(self._eng.workspace["PSF"], dtype=np.float64)

		deblurring = np.clip(deblurring, 0.0, 1.0)
		deblurring_uint8 = (deblurring * 255.0).astype(np.uint8)
		deblurring_bgr = cv2.cvtColor(deblurring_uint8, cv2.COLOR_RGB2BGR)

		return deblurring_bgr, PSF

	def __del__(self):
		try:
			self._eng.quit()
		except Exception:
			pass

