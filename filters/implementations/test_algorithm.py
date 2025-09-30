from .base import DeconvolutionAlgorithm
import cv2 as cv
import numpy as np
from time import time


class TestAlgorithm(DeconvolutionAlgorithm):

    def __init__(self,param = 0):
        self.param = param
        super().__init__('TEST')
    
    def change_param(self,param):
        self.param = param['param']
        return super().change_param(param)

    def process(self,image):
        timer1 = time()
        processed_image = cv.flip(image,0) + self.param
        kernel = np.zeros((10,10))
        timer2 = time()
        self.timer = timer2-timer1
        return processed_image, kernel
    
    
    def get_param(self):
        return [('param1',self.param),('param2',1)]
    