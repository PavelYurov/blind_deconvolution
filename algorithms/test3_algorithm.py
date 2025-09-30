from .base import DeconvolutionAlgorithm
import cv2 as cv
import numpy as np
from time import time

class TestAlgorithm3(DeconvolutionAlgorithm):

    def __init__(self):
        super().__init__('TEST3')

    def change_param(self,param):
        return super().change_param(param)
    
    def process(self,image):
        timer1 = time()
        res_image = 255.0 - image
        kernel = np.zeros((10,10))
        timer2 = time()
        self.timer = timer2 - timer1
        return res_image,kernel
    
    def get_param(self):
        return [('param5',self.param),('param6',3)]