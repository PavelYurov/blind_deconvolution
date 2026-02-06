# -*- coding: utf-8 -*-
"""
Created on Sun May 11 09:55:28 2025

@author: Dragos
"""

import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.signal import convolve2d
import cv2 

def rotated_anisotropic_gaussian_kernel(size, sigma_x, sigma_y, theta):
    kernel = np.zeros((size, size))
    center = size // 2
    
    # Rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            
            # Apply rotation
            x_rot = cos_theta * x - sin_theta * y
            y_rot = sin_theta * x + cos_theta * y
            
            kernel[i, j] = np.exp(-(x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)))

    kernel /= np.sum(kernel)  # Normalize
    return kernel

def blur_kernel(sigma, shape):
    kernel = np.zeros(shape)
    center_x, center_y = shape[0] // 2, shape[1] // 2
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            x = i - center_x
            y = j - center_y
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)  # Normalize the kernel

def show_kernels(kernel_size  , sigma_g , sigma_x , sigma_y, theta):
    
    #kernel
    kernel_matrix_gaussian_blur = blur_kernel(sigma_g, shape = kernel_size)
    kernel_matrix_motion_blur = rotated_anisotropic_gaussian_kernel( kernel_size[0] , sigma_x  , sigma_y , theta  )
    
    fig1,axs1 = plt.subplots(1,2)
    fig1.suptitle("Blur Kernels")
    
    #Axes 0
    axs1[ 0].set_title("Original Image")
    axs1[ 0].imshow(kernel_matrix_gaussian_blur , cmap='gray')
    axs1[ 0].axis("off")
  
    #Axes 1
    axs1[ 1].set_title("Blurred Image")
    axs1[ 1].imshow(kernel_matrix_motion_blur , cmap='gray')
    axs1[ 1].axis("off")
    plt.show()
    
    return kernel_matrix_gaussian_blur , kernel_matrix_motion_blur

def show_blurred_image(image , kernel , title = "Blurred Image"):
    
    #blurred iamge
    blurred_image = convolve2d( image , kernel , mode="same" )
    blurred_image = np.array( blurred_image , dtype = int )
  
    #show the kernel
    plt.imshow( blurred_image , cmap='gray' )
    plt.axis('off')
    plt.title(title)
    plt.show()
    
    return blurred_image
    
if __name__ =="__main__" :
    
    kernel_size = (15,15)
    sigma_g = 2
    sigma_x = 2
    sigma_y =6
    theta =pi/4
    
    #Gaussian and Motion 
    k_gaussian, k_motion = show_kernels( kernel_size , sigma_g , sigma_x , sigma_y , theta)

    #Read an image
    image = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
    motion_blurred_image = show_blurred_image( image , k_motion, title = "Motion Blurred Image")
    
    
    
    