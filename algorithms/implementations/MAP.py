from algorithms.base import DeconvolutionAlgorithm
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2 as cv

import metrics

class MAP(DeconvolutionAlgorithm):
	"""
	Реализация алгоритма MAP для слепой деконволюции
	
	Параметры:
	predict_kernel -
	max_iter -
	relaxation_factor = param -
	operator - 
	Huber_threshold -
	eps - 
	"""
	
	def __init__(self, max_iter, Huber_threshold, predict_kernel, operator,relaxation_factor, eps = 1e-8):
		super().__init__('MAP') 
		self.param = relaxation_factor
		self.predict_kernel = predict_kernel
		self.operator = operator
		self.Huber_threshold = Huber_threshold
		self.eps = eps
		self.max_iter = max_iter
	
	def convolve(self, image, psf):
		h, w = image.shape
		kh, kw = psf.shape

		img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

		psf_padded = np.zeros_like(img_padded)
		h_pad, w_pad = img_padded.shape
		start_h = (h_pad - kh) // 2
		start_w = (w_pad - kw) // 2
		psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
		psf_padded = ifftshift(psf_padded)
	
		img_fft = fft2(img_padded)
		psf_fft = fft2(psf_padded)
		result_fft = img_fft * np.conj(psf_fft)
		result = np.real(ifft2(result_fft))
		
		result = result[kh//2+1:h+kh//2+1, kw//2+1:w+kw//2+1]
		
		return np.clip(result, 0.0, 1.0)
	
	def deconvolve(self, image, psf):
		h, w = image.shape
		kh, kw = psf.shape

		img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

		psf_padded = np.zeros_like(img_padded)
		h_pad, w_pad = img_padded.shape
		start_h = (h_pad - kh) // 2
		start_w = (w_pad - kw) // 2
		psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
		psf_padded = ifftshift(psf_padded)

		blurred_fft = fft2(img_padded)
		psf_fft = fft2(psf_padded)
		psf_fft_conj = np.conj(psf_fft)
		# wiener_filter = psf_fft_conj / (np.abs(psf_fft) + self.eps)
		wiener_filter = psf_fft_conj / (psf_fft + self.eps)

		result_fft = blurred_fft * wiener_filter
		result = np.real(ifft2(result_fft))

		return np.clip(result[kh//2:h+kh//2, kw//2:w+kw//2], 0.0, 1.0)
	
	def dumb_deconvolve(self, image, psf):
		h, w = image.shape
		kh, kw = psf.shape

		img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='constant')

		psf_padded = np.zeros_like(img_padded)
		h_pad, w_pad = img_padded.shape
		start_h = (h_pad - kh) // 2
		start_w = (w_pad - kw) // 2
		psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
		psf_padded = ifftshift(psf_padded)

		blurred_fft = fft2(img_padded)
		psf_fft = fft2(psf_padded)
		result_fft = blurred_fft / (psf_fft + self.eps)
		result = np.real(ifft2(result_fft))

		return np.clip(result[kh//2:h+kh//2, kw//2:w+kw//2], 0.0, 1.0)
	
	def dumb_convolution(self, img, kernel):
		kh, kw = kernel.shape
		pad_h, pad_w = kh//2, kw//2
		padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
	
		result = np.zeros_like(img)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				result[i,j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
		return result

	def p(self,u):
		if(abs(u) <= self.Huber_threshold):
			return 2*u
		else:
			return 2*self.Huber_threshold*u/abs(u)

	def process(self, image):
		"""Основной метод восстановления изображения."""
		if len(image.shape) != 2:
			raise Exception("Currently supports only grayscale images")
		
		hi, wi = image.shape
		kh,kw = self.predict_kernel.shape
		

		original_image = image.copy() / 255.0
		processed_image = original_image.copy()

		h = np.full((hi,wi), 0.0)
		x_center = (wi - kw) // 2
		y_center = (hi - kh) // 2
		h[y_center:y_center+kh, x_center:x_center+kw] = np.rot90(self.predict_kernel,2)

		# o = self.deconvolve(processed_image, h)
		# o = self.dumb_deconvolve(processed_image, h)
		o = processed_image

		for i in range(0, self.max_iter):
			if i % 2 == 0:
				print(f'[{i}/{self.max_iter}]')

			conv_ho = self.convolve(o,h)
			# conv_ho = self.dumb_convolution(o,h)

			h = h * self.convolve((original_image/(conv_ho+self.eps)),np.flip(o))
			

			h = h / np.sum(h)

			conv_ho = self.convolve(o,h)
			# conv_ho = self.dumb_convolution(o,h)

			# conv_ato = self.dumb_convolution(o, self.operator.T)
			conv_ato = self.convolve(o, self.operator.T)

			# cv.imshow('tmp',conv_ato)
			# cv.waitKey()
			# cv.destroyAllWindows()

			pato = np.vectorize(self.p)(conv_ato)

			conv_apato = self.dumb_convolution(pato,self.operator)
			# conv_apato = self.convolve(pato,self.operator)

			o = o * self.convolve((original_image / (conv_ho+self.eps)),np.flip(h)) / (1 + self.param * np.sum(conv_apato))
			# cv.imshow('tmp',o)
			# cv.waitKey()
			# cv.destroyAllWindows()


		res = o[:hi,:wi].real*255.0
		return res.astype(np.int16)
	
	def get_param():
		return []

	def change_param():
		return None
