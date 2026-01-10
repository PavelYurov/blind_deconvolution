# -*- coding: utf-8 -*-
"""
Created on Sat May 17 19:37:14 2025

@author: Dragos
"""
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from blur_kernels import show_kernels
import  cv2
from skimage.measure import block_reduce
#from total_variation_deconvolution_recompute import conv2d_operator_sparse
import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import convolve2d



#all params
kernel_size = (15,15)
sigma_g = 2
sigma_x = 2
sigma_y = 8
theta = pi/4

#kernel
kernel_matrix_gaussian_blur , kernel_matrix_motion_blur = show_kernels( kernel_size , sigma_g , sigma_x , sigma_y, theta)

#Read an image
frame = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
frame = block_reduce(frame , block_size=4,func= np.mean)
plt.imshow(frame,cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

#input shape
input_shape=frame.shape


#convolution matrix H
#H = conv2d_operator_sparse(kernel_matrix_gaussian_blur,mode="same", input_shape=input_shape)

h_padded=np.zeros(input_shape)
dir1=15 // 2
dir2=15 // 2 + 1
k1=np.arange(input_shape[0] // 2 - dir1 ,  input_shape[0]//2 + dir2 ,1)
k2=np.arange( input_shape[1] // 2 - dir1, input_shape[1]//2 + dir2 ,1)
h_padded[input_shape[0] //2 - dir1 : input_shape[0]//2 + dir2 , input_shape[1]//2 - dir1 : input_shape[1]//2 + dir2] = kernel_matrix_gaussian_blur

plt.imshow(h_padded , cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()