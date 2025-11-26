import numpy as np
import random
from typing import Tuple
from random import sample
from .base import FilterBase
from scipy.signal import lfilter, butter, sosfilt
from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv

from .colored_noise import powerlaw_psd_gaussian, pink_noise_2d
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_wavelet
from scipy.ndimage import median_filter

class LinearSmoothing(FilterBase):
    """
    """
    def __init__(self, param=3):
        """
        """
        self.param = param
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|linearsmoothingdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        """ 
        denoised_image = cv.GaussianBlur(image,(self.param,self.param),0)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    

class AnisotropicDiffusion(FilterBase):
    """
    """
    def __init__(self, niter=20, kappa=50, gamma=0.1, option=2):
        """
        """
        self.niter = niter
        self.kappa = kappa
        self.gamma = gamma
        self.option = option
        super().__init__(0, 'blur')

    def discription(self) -> str:
        return f"|anisotropicdiffusiondenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        """ 
        denoised_image = anisotropic_diffusion(image, niter=self.niter, kappa=self.kappa, gamma=self.gamma, option=self.option)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)


class NonLocalMeans(FilterBase):
    """
    """
    def __init__(self, fast = False, sigma_coef = 0.01, patch_size=5, patch_distance=6):
        """
        """
        self.sigma_coef = sigma_coef
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.fast = fast
        super().__init__(0, 'blur')

    def discription(self) -> str:
        return f"|nonlocalmeansdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        """ 

        sigma_est = np.mean(estimate_sigma(image, channel_axis=-1))

        denoised_image = denoise_nl_means(image,patch_size=self.patch_size,patch_distance=self.patch_distance, h=self.sigma_coef * sigma_est , fast_mode=self.fast) * 255
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    
class MedianFilter(FilterBase):
    """
    """
    def __init__(self, param=3):
        """
        """
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|mediandenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        """ 
        denoised_image = median_filter(image, size=self.param)
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    
class Wavelet(FilterBase):
    """
    """
    def __init__(self,method='BayesShrink',mode='soft',rescale_sigma=True):
        """
        mode{‘soft’, ‘hard’}
        method{‘BayesShrink’, ‘VisuShrink’}
        """
        self.method = method
        self.mode = mode
        self.rescale_sigma = rescale_sigma
        super().__init__(0, 'blur')

    def discription(self) -> str:
        return f"|waveletdenoise_"
    
    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        """ 
        sigma_est = estimate_sigma(image, channel_axis=-1, average_sigmas=True)
        denoised_image = denoise_wavelet(image,method=self.method,mode=self.mode,rescale_sigma=self.rescale_sigma, sigma=sigma_est)*255
        return np.clip(denoised_image,0.0,255.0).astype(image.dtype)
    
class BilateralFilter(FilterBase):
    pass

class TV_bregman(FilterBase):
    pass

class TV_Chambolle(FilterBase):
    pass

class WienerFilter(FilterBase):
    pass






