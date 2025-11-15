# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:54:14 2025

@author: Dragos
"""
import matplotlib.pyplot as plt
from total

if __name__  ==  "__main__" :
     #kernel
     kernel_size=(5,5)
     kernel_matrix_gaussian_blur = blur_kernel(sigma = 0.5, shape = kernel_size)
     kernel_matrix_motion_blur = rotated_anisotropic_gaussian_kernel(kernel_size[0] , sigma_x = 2 , sigma_y=10, theta=pi/4)
     
     fig1,axs1 = plt.subplots(1,2)
     fig1.suptitle("Blur Kernels")
     
     #Axes 0
     axs1[ 0].set_title("Original Image")
     axs1[ 0].imshow(kernel_matrix_gaussian_blur , cmap='gray')
     axs1[ 0].axis("off")
     
     #---------------------------------
     #Axes 1
     axs1[ 1].set_title("Blurred Image")
     axs1[ 1].imshow(kernel_matrix_motion_blur , cmap='gray')
     axs1[ 1].axis("off")
     
     
     plt.show()
     