import numpy as np
from scipy.signal import convolve2d
from scipy.sparse.linalg import cg
from scipy.ndimage import convolve
from blur_kernels import show_kernels,show_blurred_image
from math import pi,e
from numpy.fft import fft2, ifft2, fftshift
import cv2
import matplotlib.pyplot as plt
from L1_support import add_gaussian_noise , calculate_ssim , peak_signal_noise_ratio
from .blind_deconvolution import fft_deconvolution
from scipy.sparse.linalg import LinearOperator
from scipy.fftpack import dct
from scipy.fft import dctn, idctn
from scipy.sparse import coo_matrix
from skimage.measure import block_reduce


def conv2d_operator_sparse(kernel, input_shape, mode='same',mode_adjoint=False):
    kH, kW = kernel.shape
    iH, iW = input_shape
    
    if mode == 'full':
        pad_h, pad_w = kH - 1, kW - 1
        oH, oW = iH + kH - 1, iW + kW - 1
    elif mode == 'same':
        pad_h = kH // 2
        pad_w = kW // 2
        oH, oW = iH, iW
    elif mode == 'valid':
        pad_h = pad_w = 0
        oH, oW = iH - kH + 1, iW - kW + 1
    else:
        raise ValueError("Mode must be 'valid', 'same', or 'full'.")

# =============================================================================
#     # Pad input size
#     padded_iH = iH + 2 * pad_h
#     padded_iW = iW + 2 * pad_w
#=============================================================================

    # Flip kernel
    if mode_adjoint == False :
        kernel = np.flipud(np.fliplr(kernel))

    data = []
    rows = []
    cols = []

    for ki in range(kH):
        for kj in range(kW):
            val = kernel[ki, kj]
            for i in range(oH):
                for j in range(oW):
                    row = i * oW + j
                    ii = i + ki
                    jj = j + kj

                    if 0 <= ii - pad_h < iH and 0 <= jj - pad_w < iW:
                        col = (ii - pad_h) * iW + (jj - pad_w)
                        rows.append(row)
                        cols.append(col)
                        data.append(val)

    A = coo_matrix((data, (rows, cols)), shape=(oH * oW, iH * iW))
    return A.tocsr()

#first and then u
def convolve_fft(h, u,mode="kernel"):
    
    # Zero-pad h (Dirichlet), reflect-pad u (Neumann)
    full_shape = (u.shape[0] + h.shape[0] - 1, u.shape[1] + h.shape[1] - 1)
    # Pad h with zeros (Dirichlet BCs)
    h_pad = pad_image_dirichlet( h,( (( full_shape[0] - h.shape[0] ) //2 , ( full_shape[1]-h.shape[1] )//2 ) ) )
    u_pad = pad_image_neuman( u,( (( full_shape[0] - u.shape[0] ) //2 , ( full_shape[1]-u.shape[1] )//2 ) ) )
    
    if mode == "kernel":
        start_x = (full_shape[0] - h.shape[0]) // 2
        start_y = (full_shape[1] - h.shape[1]) // 2
        increment_x = h.shape[0]
        increment_y = h.shape[1]
    else : 
        start_x = (full_shape[0] - u.shape[0]) // 2
        start_y = (full_shape[1] - u.shape[1]) // 2
        increment_x = u.shape[0]
        increment_y = u.shape[1]
        
    return np.real(ifft2(fft2(h_pad) * fft2(u_pad))[start_x : start_x + increment_x, start_y : start_y + increment_y])


def compute_gradients_dirichlet(u):
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy

def divergence_neuman(px, py):
    pxm = px[ 1 : -1 , 0 : -2 ] - px[ 1:-1 , 1:-1 ]
    pym = py[ 0 : -2 , 1 : -1 ] - py[ 1:-1 , 1:-1 ]
    
    return pxm + pym

def tv_weights(u, beta = 1e-3):
    ux , uy = compute_gradients_neuman(u , increment=2)
    return 1.0 / np.sqrt( ux**2 + uy**2 + beta )

def pad_image_neuman(u, pad_width=1):
    """Pad image with Neumann (reflective) boundary conditions."""
    return np.pad(u, pad_width, mode='reflect')  # or mode='symmetric'

def pad_image_dirichlet(u, pad_width=1):
    """Pad image with Neumann (reflective) boundary conditions."""
    return np.pad(u, pad_width, mode='constant')  # or mode='symmetric'

def compute_gradients_neuman(u,increment=1):
    u_padded = pad_image_neuman(u,increment)
    u_x = u_padded[1:-1, 2:] - u_padded[1:-1 , 1:-1]  # Forward difference in x
    u_y = u_padded[2:, 1:-1] - u_padded[1:-1, 1:-1]  # Forward difference in y
    return u_x , u_y

def laplacian_spectrum(shape):
    """Eigenvalues of standard Laplacian with Neumann BCs (DCT-II basis)"""
    M, N = shape
    x = np.cos(np.pi * np.arange(M) / (M - 1))  # DCT-II frequencies
    y = np.cos(np.pi * np.arange(N) / (N - 1))
    return 2 * (1 - x[:, None]) + 2 * (1 - y[None, :])  # Λ_L = 2(1-cos(πk/M)) + 2(1-cos(πl/N))

def weighted_laplacian(u_img,w):
    gradient_ux , gradient_uy = compute_gradients_neuman(u_img,increment=2)
    #print("grad:" , gradient_ux.shape)
    div_grad = divergence_neuman(w * gradient_ux , w * gradient_uy)
    #print("div:" , div_grad.shape)
    return div_grad

def solve_u(f, h, u0, lambda1, max_fp_iter=3,epsilon=1e-8, rtol=1e-3,atol=0.1):
    u = u0.copy()
    image_shape = u.shape
    h_padded = np.zeros(image_shape)
    if h.shape != image_shape:
        dir1 = h.shape[0] // 2
        dir2 = h.shape[1] // 2 + 1
        h_padded [f.shape[0] // 2 - dir1 : f.shape[0] // 2 + dir2 , f.shape[1] // 2 - dir1 : f.shape[1] // 2 + dir2] = h
    else :
        h_padded=h
            
    h_flipped = np.flip(np.flip(h_padded, axis = 0), axis=1)
    image_shape = f.shape
    for i in range(max_fp_iter):
        print(f"Solve u: {i+1} iteration")
        w = tv_weights(u)
            
        def A(u_flat):
            u_img = u_flat.reshape(image_shape)
            #Hu = convolve2d(u_img, h, mode='same', boundary='symm')
            Hu = convolve_fft(u_img , h_padded)
            #HTHu = convolve2d(Hu, h_flipped, mode='same', boundary='symm')
            HTHu = convolve_fft(Hu , h_flipped)
            
            #Laplacian term
            laplacian_term = weighted_laplacian(u_img,w)
            
            return ( HTHu - lambda1 * laplacian_term ).flatten()
        
        def cosine_preconditioner(h_padded, shape, w,epsilon=1e-8,average=True):
                
                # 2. H spectrum
                H_spectrum = abs(dctn(h_padded , norm='ortho'))**2
                
                # 3. Mean
                w_mean = np.mean(w)
                
                # 4. L spectrum
                L_spectrum = laplacian_spectrum(shape) * w_mean  # Λ_L = DCT(L_u)
                
                # 5. DCT spectrum of HᵗH
                spectrum = H_spectrum + lambda1 * L_spectrum
                
                # 6. Inverse (diagonal of preconditioner)
                inv_spectrum = 1.0 / spectrum
                
                return inv_spectrum
            
        def apply_preconditioner(v, inv_spectrum, shape):
            v_img = v.reshape(shape)
            V = dctn(v_img, norm='ortho')
            Y = inv_spectrum * V  # element-wise multiplication
            y_img = idctn(Y, norm='ortho')
            return y_img.flatten()
            
        #Inv spectrum
        inv_spectrum = cosine_preconditioner(h_padded, shape = image_shape  , w = w, epsilon = epsilon, average = True)
        A_shape=(image_shape[0]**2,image_shape[0]**2)
        
        #A preconditioner operator
        A_operator = LinearOperator(shape = A_shape, matvec=A,dtype = np.float16)
        
        #M preconditioner operator
        M_operator = LinearOperator(shape =  A_shape, matvec=lambda v: apply_preconditioner( v , inv_spectrum, shape = image_shape),dtype = np.float16)
        
        #Convolution
        b_img = convolve2d(f, h_flipped, mode='same', boundary='symm')
        b_flat = b_img.flatten()
        x0_flat = u.flatten()
        print("x0_flat shape:",x0_flat.shape)
        u_flat, info = cg(A_operator, b=b_flat, x0=x0_flat, M=M_operator, rtol=rtol, atol=atol)

        
        u = u_flat.reshape(image_shape)
        u[u < 0] = 0
    
    return u

def Lu_operator(v_img, w):
    # v_img shape = (iH, iW)
    ux = np.roll(v_img, -1, axis=1) - v_img
    uy = np.roll(v_img, -1, axis=0) - v_img
    
    wx_inv = 1.0 / (w + 1e-8)  # To avoid division by zero
    
    px = wx_inv * ux
    py = wx_inv * uy
    
    div_p = (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
    return -div_p

# =============================================================================
# 
# def solve_u_toeplitz(f, h, u0, lambda1, max_fp_iter=3, rtol=1e-3,atol=0.1):
#     u = u0.copy()
#     shape_image=u.shape
# 
#     for _ in range(max_fp_iter):
#         w = tv_weights(u)
# 
#    
#         H_star = conv2d_operator_sparse(h,shape_image , mode='same' , mode_adjoint=False)
#         H = conv2d_operator_sparse(h,shape_image , mode='same' , mode_adjoint=True)
#         Lu =    Lu_operator(u,w) 
#         
#         
#         def A(u_flat):
#             A_mat = (H_star + H +Lu)
#             
#         #u_img = u_flat.reshape(f.shape)
#         def DCT_preconditioned_A(u_flat):
#             # Apply DCT on the matrix-vector product
#             spatial_product = A(u_flat)
#             # Transform to frequency domain
#             return dctn(spatial_product)
#         
#         #Creating the LinearOperator with DCT preconditioning for A
#         A_operator = LinearOperator ( shape = (f.size, f.size) , matvec = DCT_preconditioned_A)
#         
#         #Img
#         b_img = convolve2d (f, h_flipped, mode='same', boundary='symm')
#         
#         #Dctn
#         b_dct = dctn(b_img)
#         b_dct_flat = b_dct.flatten()
#         x0_dct = dctn(u)
#         x0_dct_flat = x0_dct.flatten()
#         
#         #Conjugate Gradient
#         u_flat , _ = cg(A_operator , b = b_dct_flat , x0 = x0_dct_flat , rtol = rtol , atol=atol)
#         
#         #Reshape the solution back to image space
#         u = idctn(u_flat.reshape(f.shape))
#         
#         #Normalize
#         #u[u<0]=0
#               
#         #return u
# =============================================================================
   
def crop(image, crop_size):
    
    croped_image= image[ image.shape[0]//2-crop_size//2 : image.shape[0]//2 + crop_size//2 + 1  ,  image.shape[1] // 2 - crop_size//2 : image.shape[1]//2 + crop_size//2 + 1]

    return croped_image





def solve_h(f, u, h0, lambda2, max_fp_iter=3,epsilon=1e-8, rtol=1e-3,atol=0.1):
    h = h0.copy()
    u_flipped = np.flip(np.flip(u, axis=0), axis=1)
    image_shape = f.shape
    
    h_padded = np.zeros(image_shape)
    if h.shape != image_shape:
        dir1 = h.shape[0] // 2
        dir2 = h.shape[1] // 2 + 1
        h_padded [f.shape[0] // 2 - dir1 : f.shape[0] // 2 + dir2 , f.shape[1] // 2 - dir1 : f.shape[1] // 2 + dir2] = h
    else :
        h_padded = h
               
    
    for i in range(max_fp_iter):
        print(f"Solve h: {i+1} iteration")
        w = tv_weights(h_padded)

        def A(h_flat):
            h_img = h_flat.reshape(h_padded)
            UH = convolve2d(h_img, u, mode='same', boundary='symm')
            UH = convolve_fft(h_img , u)
            UTUH = convolve2d(UH, u_flipped, mode='same', boundary='symm')
            UTUH = convolve_fft(UH,u_flipped)
            laplace_term = weighted_laplacian(h_img)
            
            return (UTUH - lambda2 * laplace_term).flatten()
        
 
# =============================================================================
#          
#         def cosine_preconditioner(u, shape, w , epsilon=1e-8 , average=True):
#                 # 1. Delta impulse image
#                 delta = np.zeros(shape)
#                 delta[shape[0]//2,shape[1]//2] = 1
#             
#                 # 2. Simulate HᵗH delta
#                 u_flip = np.flipud(np.fliplr(u))
#                 Ud = convolve2d(delta, u, mode='same', boundary='symm')
#                 UTUd = convolve2d(Ud, u_flip, mode='same', boundary='symm')
#             
#                 # 3. Total variation diagonal approx (optional)
#                 gradient_deltax , gradient_deltay = gradient(delta)
#                 
#                 if average == False:
#                     div_grad_delta = divergence(w * gradient_deltax , w * gradient_deltay)
#                 else:
#                     w_avg = np.mean(w)
#                     div_grad_delta = divergence(w_avg * gradient_deltax , w_avg * gradient_deltay)
#                
#                 # 4. DCT spectrum of HᵗH
#                 spectrum = dctn(UTUd, norm='ortho') + dctn(div_grad_delta,norm ='ortho') + epsilon
#                 
#                 # 6. Inverse (diagonal of preconditioner)
#                 inv_spectrum = 1.0 / spectrum
#                 return inv_spectrum
# =============================================================================
          
        def cosine_preconditioner(u, shape, w , epsilon = 1e-8 , average=True):
                  
           
                  # 2. Simulate HᵗH delta
                  u_flip = np.flipud(np.fliplr(u))
                 
                  # 2. H spectrum
                  U_spectrum = dctn(u_flip , norm='ortho') * dctn(u , norm='ortho')
                  
                  # 3. Mean
                  w_mean = np.mean(w)
                  
                  # 4. L spectrum
                  L_spectrum = laplacian_spectrum(shape) * w_mean  # Λ_L = DCT(L_u)
                  
                  # 5. DCT spectrum of HᵗH
                  spectrum = U_spectrum + lambda2 * L_spectrum
                  
                  # 6. Inverse (diagonal of preconditioner)
                  inv_spectrum = 1.0 / spectrum
                  
                  return inv_spectrum    
              
                
        def apply_preconditioner(v, inv_spectrum, shape):
            v_img = v.reshape(shape)
            V = dctn(v_img, norm='ortho')
            Y = inv_spectrum * V  # element-wise multiplication
            y_img = idctn(Y, norm='ortho')
            return y_img.flatten()
        
        
        #Inv spectrum
        inv_spectrum = cosine_preconditioner(u, shape= u.shape , w = w, epsilon=epsilon, average=True)
        A_shape=(image_shape[0]**2,image_shape[0]**2)
        
        #Creating the LinearOperator with DCT preconditioning for A
        A_operator = LinearOperator ( shape = A_shape , matvec = A , dtype = np.float16)
        
        #M preconditioner operator 
        M_operator = LinearOperator( shape = A_shape, matvec = lambda v: apply_preconditioner( v , inv_spectrum, shape = image_shape ) , dtype = np.float16)
       
        #Img
        b_img = convolve2d(f, u_flipped, mode='same', boundary='symm')
        
        #B image
        b_flat = b_img.flatten()
        
        #Initial solution
        x0 = h
        
        #Flatten version
        x0_flat = x0.flatten()

        #Conjugate Gradient 
        h_flat , _ = cg(A_operator , b = b_flat , x0 = x0_flat , M = M_operator ,rtol = rtol , atol=atol)        
        #Reshape the solution back to iamge space
        h = h_flat.reshape(h.shape)

        # Normalize and project
        h[h < 0] = 0
        h= (h + h.transpose)/2
        h /= (np.sum(h) + 1e-8)

    h = crop(h,h0.shape[0])
    
    return h
    
def blind_deconvolution_am(f, kernel_shape, lambda1, lambda2, num_am_iter=10):
    u = f.copy()
    h = np.zeros(kernel_shape)
    h[kernel_shape[0] // 2, kernel_shape[1] // 2] = 1.0  # delta init

    for i in range(num_am_iter):
        print(f"AM Iteration {i+1}")
        u = solve_u(f, h, u, lambda1)
        h = solve_h(f, u, h, lambda2)

    return u, h

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
    frame = block_reduce(frame , block_size=4,func= np.mean)
    plt.imshow(frame,cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    #input shape
    input_shape=frame.shape
    
    #convolution matrix H
    #H = conv2d_operator_sparse(kernel_matrix_gaussian_blur, input_shape)
    
    #Gaussian and Motion Blurred Image
    gaussian_blurred_image = show_blurred_image( frame , kernel_matrix_gaussian_blur  , title = "Gaussian Blurred" )
    motion_blurred_image = show_blurred_image( frame , kernel_matrix_motion_blur  , title = "Motion Blurred" )
    
    #Add noise 
    noisy_blurred_image = add_gaussian_noise( gaussian_blurred_image , 0 , 30 )
    plt.imshow(noisy_blurred_image , cmap='gray')
    plt.title("Noisy Blurred Image")
    plt.axis("off")
    plt.show()

    #Total Variation deconvolution
    print("Blind TV started: ")
    alfa1 = 2*10 **(-6)
    alfa2 = 1.5*10 **(-5)
    u_image, h_kernel = blind_deconvolution_am( noisy_blurred_image , kernel_size , alfa1 , alfa2)
   
    #Axes2
    plt.imshow(u_image , cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis("off")
    plt.show()
    
    #ssim
    ssim_blurred = calculate_ssim(frame , gaussian_blurred_image)
    ssim_noisy_blurred = calculate_ssim(frame, noisy_blurred_image)
    ssim_rec = calculate_ssim(frame, u_image)
    
    #psnr
    psnr_noisy_blurred = peak_signal_noise_ratio( frame , noisy_blurred_image)
    psnr_reconstructed = peak_signal_noise_ratio( frame , u_image)
    plt.show()
    
    

 