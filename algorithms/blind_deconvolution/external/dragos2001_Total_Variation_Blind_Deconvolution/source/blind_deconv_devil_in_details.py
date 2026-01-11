# -*- coding: utf-8 -*-
"""
Created on Mon May 26 23:12:22 2025

@author: Dragos
"""

#imports
#from scipy.sparse import lil_matrix
import cv2 
from skimage.filters import gaussian
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
folder_b_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lab1'))
sys.path.append(folder_b_path)
from L1_support import add_gaussian_noise , calculate_ssim , peak_signal_noise_ratio
from scipy.signal import  convolve2d
from scipy.signal import fftconvolve
from math import pi,e
from numpy.fft import fft2, ifft2, fftshift
from blur_kernels import show_kernels
from scipy.sparse import lil_matrix
from scipy.ndimage import convolve

#flip kernel
def flip_kernel(kernel):
    return np.flip(np.flip(kernel, axis=0), axis=1)

#divergence
def divergence (image_components,kernel):
    
    image_component_differentiate_x = convolve2d( image_components[0] , kernel, mode='same')
    image_component_differentiate_y = convolve2d( image_components[1] , flip_kernel(kernel) , mode='same')
    
    return image_component_differentiate_x + image_component_differentiate_y


#divergence image kernel term
def divergence_image_kernel_term ( gradient, norm , gradient_kernel ):
    
    grad_div_term = [ grad / norm for grad in gradient ]
    divergence_image = divergence( grad_div_term , gradient_kernel )
    
    return divergence_image
    
#derivative respects to theta    
def derivative_respects_to_theta(gradient_norm, epsilon, theta ):
    #term
    term = gradient_norm**2 + epsilon
    integrand  = (term)**theta * np.log10(term)
    result = np.sum(integrand)

    return result

#compute norm
def compute_norm(gradient,epsilon=0.002):
    #gradient norm
    gradient_norm = np.zeros(gradient.shape[1:3])
    gradient_norm = np.sqrt(gradient[0]**2 + gradient[1]**2+epsilon)
            
    return gradient_norm
   
#divergence image smoothing term
def divergence_image_smoothing_term(image , epsilon , theta):
    
    #--filters--
    nabla_north_filter = np.array([[0,1,0],[0,-1,0],[0,0,0]])
    nabla_south_filter = np.array([[0,0,0],[0,-1,0],[0,1,0]])
    nabla_east_filter = np.array([[0,0,0],[0,-1,1],[0,0,0]])
    nabla_west_filter = np.array([[0,0,0],[1,-1,0],[0,0,0]])
    
    #--convolutions-
    nabla_north = convolve2d(image, nabla_north_filter,mode="same")
    nabla_south = convolve2d(image, nabla_south_filter,mode="same")
    nabla_east = convolve2d(image, nabla_east_filter,mode="same")
    nabla_west = convolve2d(image, nabla_west_filter,mode="same")
    
    #--terms-
    func_term_north = theta * ( nabla_north * nabla_north +  epsilon )**( theta - 1 )
    funct_term_south = theta * ( nabla_south * nabla_south + epsilon )**( theta - 1 )
    funct_term_east = theta * ( nabla_east * nabla_east + epsilon )**( theta - 1 )
    funct_term_west = theta * ( nabla_west * nabla_west + epsilon )**( theta - 1 )
   
    divergence = func_term_north * nabla_north + funct_term_south * nabla_south + funct_term_east * nabla_east +  funct_term_west * nabla_west
    
    return divergence


from .total_variation_deconvolution_recompute import tv_weights, weighted_laplacian , pad_image_dirichlet , pad_image_neuman
#first and then u
def convolve_fft(h, u,mode="image"):
    
    # Zero-pad h (Dirichlet), reflect-pad u (Neumann)
    full_shape = (u.shape[0] + h.shape[0] - 1, u.shape[1] + h.shape[1] - 1)
    
    start_x = ( full_shape[0] - h.shape[0] )//2
    start_y = ( full_shape[1] - h.shape[1] )//2
    increment_x = h.shape[0]
    increment_y = h.shape[1]
    
    
    #u pad
    u_pad = pad_image_neuman( u,( ( ( full_shape[0] - u.shape[0] ) //2 , ( full_shape[1]-u.shape[1] ) //2  ) ) )
    
    # Pad h with zeros (Dirichlet BCs)
    h_pad = np.zeros(u_pad.shape)
    h_pad[ start_x : start_x + increment_x , start_y : start_y + increment_y ] = h
 
    if mode == "image":
        start_x = (full_shape[0] - u.shape[0]) // 2
        start_y = (full_shape[1] - u.shape[1]) // 2
        increment_x = u.shape[0]
        increment_y = u.shape[1]
        
    return np.real(np.fft.ifftshift(ifft2((fft2(h_pad)) *fft2(u_pad))))[start_x : start_x + increment_x, start_y : start_y + increment_y]

def fft_deconvolution(blurred, kernel, epsilon=1e-6):
    blurred = blurred.astype(np.float32) / 255.0
    kernel = kernel.astype(np.float32)
    kernel /= np.sum(kernel)

    ih, iw = blurred.shape
    kh, kw = kernel.shape

    pad_kernel = np.zeros_like(blurred)
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -kh // 2, axis=0)
    pad_kernel = np.roll(pad_kernel, -kw // 2, axis=1)

    B = fft2(blurred)
    K = fft2(pad_kernel)

    K_conj = np.conj(K)
    I_hat = B * K_conj / (np.abs(K)**2 + epsilon)

    I_rec = np.real(ifft2(I_hat))
    I_rec = np.clip(I_rec, 0, 1)
    return (I_rec * 255).astype(np.uint8)

#inner loop for h
def optimize_h_kernel(n_iterations, u, h , z, deviation = 0.05, step=0.05):
    
    u_image = u.copy()
    h_old = h.copy()
    h_current = h.copy()
    for it in range (n_iterations):
      
        # Kernel smoothing term
        first_conv = convolve_fft( h_old , u_image , mode ='image' ) 
        #first_conv = convolve( h_old , u_image , mode = 'constant')
        second_term = (first_conv - z)
    
        # Convolution term
        u_transpose = flip_kernel(u_image)
       
        # Convolution of current kernel with image
        df_dh = convolve_fft( u_transpose , second_term , mode = 'kernel')
        start_x = (df_dh.shape[0] - h.shape[0])//2
        start_y = (df_dh.shape[1] - h.shape[1])//2
        df_dh_crop = df_dh[start_x : start_x + h.shape[0] , start_y : start_y + h.shape[1]]
        #df_dh = convolve( u_transpose , second_term , mode = 'kernel')
        # Update kernel 
        h_current = h_old - step * df_dh_crop 
        
        # Set negative elements to zero
        h_current[h_current < 0] = 0
        
        # Normalize h_kernel
        h_current = ( h_current + flip_kernel(h_current) )/2
        
        # Sum k 
        sum_k = np.sum(h_current)
        
        if sum_k > 0:
            h_current /= sum_k
        else:
            h_current = np.zeros_like(h_current)
            h_current[h_current.shape[0]//2, h_current.shape[1]//2] = 1
        
        if deviation != 0:
            # Check convergence
            diff = np.linalg.norm(h_current - h_old)
            
            if diff < deviation:
                break
            h_old = h_current.copy()
        
    return h_current

#inner loop for u
def optimize_u_image( n_iterations , u , h , z , epsilon , lambda_v=0.006 , deviation = 0.05 , step = 0.05):
    
    u_old = u.copy()
    u_current = u.copy()
    h_kernel = h.copy()
 
    
    for it in range (n_iterations):
        # Tv weights
        w = tv_weights(u_current)
        
        # Laplacian
        laplacian_term = weighted_laplacian(u_current , w)
        
        # First conv term
        first_conv = convolve_fft(h_kernel , u, mode="image") 
        
        #First term
        first_term = first_conv - z
        
        #Convolution term
        h_transpose = flip_kernel(h_kernel)
       
        #Second Conv
        second_conv = convolve_fft( h_transpose , first_term, mode = "image")
        
        #Derivative respects to u
        df_du = second_conv - lambda_v * laplacian_term
        
        # Update image
        u_current = u_old - step * df_du
        
        # Set negative elements to 0
        u_current[u_current < 0]=0
        
        if deviation != 0:
            # Check convergence
            diff = np.linalg.norm(u_current - u_old)
            
            if diff < deviation:
                break
            u_old = u_current.copy()
        
    return u_current
    
#total variation deconvolution   
def total_variation_deconvolution( noisy_blurred_image , kernel_size = (5,5)  , lambda_v = 0.1 , deviation=0 , step=0.1  , criterium = 0.2 , num_iterations_primary = 200 , num_iterations_secondary = 60):
     
    #1.Define the kernel 
   
    kernel_matrix = np.zeros(kernel_size)
    kernel_matrix[ kernel_size[0]//2 , kernel_size[1]//2 ] = 1
    
    #2.Timestamp 0
    u_image_previous = noisy_blurred_image.copy()
    h_kernel_previous = kernel_matrix.copy()
    
    #4.Initial Image
    z_image = noisy_blurred_image

    for it in range(num_iterations_primary):
        
        #6.Current Iteration
        print("Iteration number:",it)
        
        #7.Update kernel
        h_kernel = optimize_h_kernel( num_iterations_secondary , u_image_previous , h_kernel_previous , z_image , deviation , step)
        
        #8.Update image
        u_image = optimize_u_image( num_iterations_secondary , u_image_previous , h_kernel_previous , z_image  , deviation , step)
        
        
        #11.Delta image
        delta_img = np.linalg.norm(u_image - u_image_previous) / np.linalg.norm(u_image_previous)
        delta_kernel = np.linalg.norm(h_kernel - h_kernel_previous) / np.linalg.norm(h_kernel_previous)
        
        #12.Check stop conditions
        if delta_img < criterium :
            break
        
        #13.Otherwise update
        else:
            u_image_previous = u_image
            h_kernel_previous = h_kernel
        
    return u_image,h_kernel

def fft_deconvolution(blurred, kernel, epsilon=1e-6):
    blurred = blurred.astype(np.float32) / 255.0
    kernel = kernel.astype(np.float32)
    kernel /= np.sum(kernel)

    ih, iw = blurred.shape
    kh, kw = kernel.shape

    pad_kernel = np.zeros_like(blurred)
    pad_kernel[:kh, :kw] = kernel
    pad_kernel = np.roll(pad_kernel, -kh // 2, axis=0)
    pad_kernel = np.roll(pad_kernel, -kw // 2, axis=1)

    B = fft2(blurred)
    K = fft2(pad_kernel)

    K_conj = np.conj(K)
    I_hat = B * K_conj / (np.abs(K)**2 + epsilon)

    I_rec = np.real(ifft2(I_hat))
    I_rec = np.clip(I_rec, 0, 1)
    return (I_rec * 255).astype(np.uint8)
      
#compute the gradient
def compute_gradient_in_2D(image , mask):
    
    gradient_x = convolve2d(image, mask, mode='same')
    gradient_y = convolve2d(image, flip_kernel(mask), mode='same')
    
    
    return np.array([gradient_x, gradient_y])

if __name__  ==  "__main__" :
    
    #all params
    kernel_size = (15,15)
    sigma_g = 6
    sigma_x = 2
    sigma_y = 8
    theta = pi/4
    
    #kernel
    kernel_matrix_gaussian_blur , kernel_matrix_motion_blur = show_kernels( kernel_size , sigma_g , sigma_x , sigma_y , theta)
    
    #Read an image
    frame = cv2.imread('lena.png',cv2.IMREAD_GRAYSCALE)
    blurred_image = convolve2d(frame, kernel_matrix_gaussian_blur,mode="same")
    blurred_image = np.array( blurred_image , dtype = int )
    size=frame.shape[:2]
    
    #Img
    img_motion_blur = convolve2d(frame , kernel_matrix_motion_blur,mode="same",boundary='wrap')
    
    #---------------------------------
    #Figure
    fig,axs=plt.subplots(2,2)
    
    #---------------------------------
    #Axes 0
    axs[0 , 0].set_title("Original Image")
    axs[0 , 0].imshow(frame , cmap='gray')
    axs[0 , 0].axis("off")
    
    #---------------------------------
    #Axes 1
    axs[0 , 1].set_title("Blurred Image")
    axs[0 , 1].imshow(blurred_image , cmap='gray')
    axs[0 , 1].axis("off")
    
    #---------------------------------
    #Add noise 
    noisy_blurred_image = add_gaussian_noise(blurred_image,0,30)
    
    #---------------------------------
    #Axes2
    axs[1,0].set_title("Noisy Blurred Image")
    axs[1,0].imshow(noisy_blurred_image , cmap='gray')
    axs[1,0].axis("off")

    #total variation deconv-----------
    u_image, h_kernel = total_variation_deconvolution(noisy_blurred_image , kernel_size = kernel_size  ,lambda_v=2, deviation = 0,criterium=0.001, step = 0.2  , num_iterations_primary = 100 , num_iterations_secondary = 1)
    u_image = u_image.astype(int)
    
    #Axes2
    axs[1,1].set_title("Reconstructed Image")
    axs[1,1].imshow(u_image , cmap='gray')
    axs[1,1].axis("off")

    #ssim
    ssim_original = calculate_ssim(frame , frame)
    ssim_blurred = calculate_ssim(frame , blurred_image)
    ssim_noisy_blurred = calculate_ssim(frame, noisy_blurred_image)
    ssim_rec = calculate_ssim(frame, u_image)
    
    #psnr noisy blurred
    psnr_noisy_blurred = peak_signal_noise_ratio( frame , noisy_blurred_image)
    psnr_reconstructed = peak_signal_noise_ratio( frame , u_image)
    plt.show()
    
    
    #plot imshow motion blur
    plt.imshow(img_motion_blur , cmap='gray')
    plt.show()
    
    
    #deconvolve the image 
    original_image = fft_deconvolution(img_motion_blur , kernel_matrix_motion_blur) 
    plt.imshow(original_image , cmap='gray')
    plt.show()