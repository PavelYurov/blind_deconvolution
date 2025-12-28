import os
import cv2
import numpy as np
import matlab.engine

from algorithms.base import DeconvolutionAlgorithm

SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")

KERNEL_PLACEHOLDER = np.array([[0]])

class ZalteckBCDTV(DeconvolutionAlgorithm):
	def __init__(self):
		self._eng = matlab.engine.start_matlab()
		self._eng.addpath(self._eng.genpath(SOURCE_PATH), nargout=0)
		self._eng.cd(SOURCE_PATH, nargout=0)

	def change_param(self, param):
		return None

	def get_param(self):
		return {}

	def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		'''image: cv2.COLOR_BGR2RGB'''

		image = image.astype(np.float64) / 255.0

		m, n, nc = image.shape

		I_mat = matlab.double(image.tolist())
		self._eng.workspace['I'] = I_mat

		self._eng.eval("load('RMImageSet','RM');", nargout=0)

		self._eng.eval("[m,n,nc] = size(I);", nargout=0)
		self._eng.eval("[CT, M, alpha, beta, gamma] = BCDHETV(I, RM);", nargout=0)  # I уже double
		self._eng.eval("ns = size(M,2);", nargout=0)

		self._eng.eval("concentrations = reshape(CT', m, n, ns);", nargout=0)

		self._eng.eval("Hrec_OD  = reshape((M(:,1)*CT(1,:))', m, n, nc);", nargout=0)
		self._eng.eval("Hrec_RGB = OD2intensities(Hrec_OD);", nargout=0)

		self._eng.eval("Erec_OD  = reshape((M(:,2)*CT(2,:))', m, n, nc);", nargout=0)
		self._eng.eval("Erec_RGB = OD2intensities(Erec_OD);", nargout=0)

		self._eng.eval("OD_rec = reshape((M * CT)', m, n, nc);", nargout=0)
		self._eng.eval("I_rec = OD2intensities(OD_rec);", nargout=0)

		I_rec_mat = self._eng.workspace['I_rec']
		I_rec_rgb = np.array(I_rec_mat)

		I_rec_rgb_uint8 = np.clip(I_rec_rgb * 255.0, 0, 255).astype(np.uint8)
		I_rec_bgr = cv2.cvtColor(I_rec_rgb_uint8, cv2.COLOR_RGB2BGR)

		return I_rec_bgr,KERNEL_PLACEHOLDER

	def __del__(self):
		self._eng.quit()
