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

def build_main_K_2 ( kernel , image_size ):
    
    kernel_size = kernel.shape[0]
    t = image_size - kernel_size + 1
    rows_K = t 
    cols_K = kernel_size * (kernel_size +t-1)
    matrix_K = lil_matrix((rows_K, cols_K)) 
    
    for r in range (matrix_K.shape[0]):
        for i in range(kernel.shape[0]):
            start_index =  ( kernel_size + t - 1) * i + r 
            end_index = (start_index + kernel_size )
            matrix_K[r , start_index : end_index] = kernel[i,:]
            
    return matrix_K


def build_toeplitz(matrix_K,kernel_size,image_size):
     
    t = image_size - kernel_size + 1 
    
    toeplitz_k=[]
    for h in range (t):
        
        s_matrix = np.zeros((t,image_size*(t-1-h)))
        if h>0:
            f_matrix = np.zeros((t,h))
            row = np.concatenate((f_matrix , matrix_K , s_matrix),axis=1)
        else:
            row = np.concatenate((matrix_K , s_matrix),axis=1)
        toeplitz_k.append(row)
        
    
    toeplitz_concat=np.concatenate(toeplitz_k,axis=0)
    return toeplitz_concat



def build_sparse_toeplitz(matrix_K, kernel_size, image_size):
    t = image_size - kernel_size + 1 
    rows_K = matrix_K.shape[0] * t 
    cols_K = image_size * (t - 1) + matrix_K.shape[1]
    
    toeplitz = lil_matrix((rows_K, cols_K))  # sparse instead of dense
    print(toeplitz.size)
    
    for row in range(t):
       
        #Start_x
        start_x = row*t
        
        #End_x
        end_x = start_x + matrix_K.shape[0]
        
        #Start_y
        start_y = row * image_size
        
        #End_y
        end_y = start_y + matrix_K.shape[1]
        
        #Toeplitz
        toeplitz[start_x : end_x  , start_y : end_y ]=matrix_K
    
    return toeplitz


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




#inner loop for h
def iterative_inner_loop_h_kernel(n_iterations, u, h , z, alfa2, deviation = 0.05, step=0.05):
    
    u_image=u.copy()
    h_kernel_old = h.copy()
    gradient_kernel_mask  = np.array([[1,-1]])
    
    for it in range (n_iterations):
      
        # Kernel smoothing term
        gradient_kernel = compute_gradient_in_2D( h_kernel_old , gradient_kernel_mask)
        gradient_norm = compute_norm(gradient_kernel)
        divergence_kernel_smooth = divergence_image_kernel_term(gradient_kernel, gradient_norm ,gradient_kernel_mask )

        # Convolution term
        u_transpose = flip_kernel(u_image)
       
        # Convolution of current kernel with image
        conv_1 = fftconvolve( u_image , h_kernel_old , mode = 'same')
        
        # Second term
        second_term = conv_1 - z
        
        # The convolution
        conv_2 = fftconvolve( u_transpose , second_term , mode = 'same' )
        
        #first term gradient
        first_term_gradient = conv_2[ conv_2.shape[0]//2 - h.shape[0]//2 : conv_2.shape[0]//2 + (h.shape[0]+1)//2 , conv_2.shape[1]// 2 - h.shape[1]// 2 : conv_2.shape[1]// 2 + (h.shape[1]+1)// 2 ] 
        
        # Gradients
        df_dh = first_term_gradient - alfa2 * divergence_kernel_smooth
   
        # Update kernel 
        h_kernel_new = h_kernel_old - step * df_dh
        
        # Set negative elements to zero
        h_kernel_new[h_kernel_new < 0] = 0
        
        # Normalize h_kernel
        h_kernel_new = (h_kernel_new + flip_kernel(h_kernel_new))/2
        
        # Sum k 
        sum_k = np.sum(h_kernel_new)
        
        if sum_k > 0:
            h_kernel_new /= sum_k
        else:
            h_kernel_new = np.zeros_like(h_kernel_new)
            h_kernel_new[h_kernel_new.shape[0]//2, h_kernel_new.shape[1]//2] = 1
        
        # Check convergence
        diff = np.linalg.norm(h_kernel_new - h_kernel_old)
        
        if diff < deviation:
            break
        h_kernel_old = h_kernel_new.copy()
        
    return h_kernel_old 

#inner loop for u
def iterative_inner_loop_u_image( n_iterations , u , h , z , epsilon,  theta,  alfa1 , deviation = 0.05 , step=0.05):
    
    u_image_old = u.copy()
    h_kernel = h.copy()
    #kernel_mask  = np.array([[1,-1]])
    
    for it in range (n_iterations):
      

        # Image smoothing term
        divergence_img_smooth = divergence_image_smoothing_term(u_image_old, epsilon, theta)
        
        # Convolution term
        h_transpose = flip_kernel(h_kernel)
       
        # Convolution of kernel with current_image
        conv_1 = fftconvolve( u_image_old, h_transpose , mode='same')
        
        # Second term
        second_term = conv_1 - z
        #print("shape:" , second_term.shape)
        
        # The convolution
        conv_2 = fftconvolve( second_term ,h_transpose , mode='same' )
        
        #print("shape2:" , conv_2.shape)
        
        #print("shape3:" , divergence_img_smooth.shape)
        
        # Gradients
        df_du = conv_2 - alfa1 * divergence_img_smooth
   
        # Update image
        u_image_new = u_image_old - step * df_du
        
        # Set negative elements to 0
        u_image_new[u_image_new < 0]=0
        
        # Check convergence
        diff = np.linalg.norm(u_image_new - u_image_old)
        
        if diff < deviation:
            break
        u_image_old = u_image_new.copy()
        
    return u_image_new
    
#total variation deconvolution   
def total_variation_deconvolution( noisy_blurred_image , sigma = 2 , kernel_size = (5,5) , alfa1 = 0.005 , alfa2 = 0.005  , theta=1/2 , deviation=0.0001, step=0.1 , epsilon=0.002 , criterium=0.2, num_iterations_primary = 200 , num_iterations_secondary = 60):
     
    #1.Define the kernel 
   
    kernel_matrix = np.zeros(kernel_size)
    kernel_matrix[ kernel_size[0]//2 , kernel_size[1]//2 ] = 1
    
    #2.Timestamp 0
    u_image_previous = noisy_blurred_image.copy()
    h_kernel_previous = kernel_matrix.copy()
    
    #4.Initial Image
    z_image = noisy_blurred_image
    
    #5.Set Theta
    theta_computed = theta
    
    #6.Define the gradient mask
    gradient_mask  = np.array([[1,-1]])
    for it in range(num_iterations_primary):
        #7.Update kernel
        h_kernel = iterative_inner_loop_h_kernel( num_iterations_secondary , u_image_previous , h_kernel_previous , z_image , alfa2 , deviation , step)
        
        #8.Update image
        u_image = iterative_inner_loop_u_image( num_iterations_secondary , u_image_previous , h_kernel_previous , z_image , epsilon , theta_computed , alfa1 , deviation , step)
        
        #9.Compute the image gradient norm
        u_image_gradient = compute_gradient_in_2D( u_image , gradient_mask )
        u_image_gradient_norm = compute_norm( u_image_gradient , epsilon )
        
        #10.Theta computed 
        #theta_computed = theta_computed - step * derivative_respects_to_theta ( u_image_gradient_norm , epsilon, theta)
        theta_computed = theta
        
        #11.Delta image
        delta_img = np.linalg.norm(u_image - u_image_previous) / np.linalg.norm(u_image_previous)
        delta_kernel = np.linalg.norm(h_kernel - h_kernel_previous) / np.linalg.norm(h_kernel_previous)
        
        #12.Check stop conditions
        if delta_img < criterium and delta_kernel < criterium:
            break
        
        #13.Otherwise update
        else:
            u_image_previous = u_image
            h_kernel_previous = h_kernel
        
    return u_image,h_kernel
         
#compute the gradient
def compute_gradient_in_2D(image , mask):
    
    gradient_x = convolve2d(image, mask, mode='same')
    gradient_y = convolve2d(image, flip_kernel(mask), mode='same')
    
    
    return np.array([gradient_x, gradient_y])

if __name__  ==  "__main__" :
    
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
    blurred_image = convolve2d(frame, kernel_matrix_gaussian_blur,mode="same")
    blurred_image = np.array( blurred_image , dtype = int )
    size=frame.shape[:2]
    
    #img
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
    axs[1,0].set_title("Noisy Image")
    axs[1,0].imshow(noisy_blurred_image , cmap='gray')
    axs[1,0].axis("off")

    #total variation deconv-----------
    u_image, h_kernel = total_variation_deconvolution(noisy_blurred_image ,sigma=2, kernel_size=(5,5) , alfa1 =  2*10^-6 , alfa2 =  (10^-5) , theta = 1/2 , deviation = 0.1 , step = 0.05 , epsilon = 0.001 , criterium=0.1 , num_iterations_primary = 3 , num_iterations_secondary = 10)
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