import numpy as np
from numpy.fft import fft2, ifft2
from scipy.ndimage import zoom, center_of_mass, shift

def psf2otf(psf, shape):
    """
    Convert PSF to OTF with cyclic shift.
    Maps kernel center to frequency origin (0,0).
    """
    in_shape = psf.shape
    # Pad to image size
    otf = np.zeros(shape, dtype=psf.dtype)
    otf[:in_shape[0], :in_shape[1]] = psf
    
    # Circular shift to place center at (0,0)
    for axis, axis_size in enumerate(in_shape):
        otf = np.roll(otf, -int(axis_size // 2), axis=axis)
        
    return fft2(otf)

def precompute_gradient_operators(shape):
    """
    Returns FFT of gradient filters [1, -1] and denominator for ADMM.
    """
    h, w = shape
    # Horizontal difference
    dx = np.zeros((h, w), dtype=np.float32)
    dx[0, 0] = 1; dx[0, 1] = -1
    
    # Vertical difference
    dy = np.zeros((h, w), dtype=np.float32)
    dy[0, 0] = 1; dy[1, 0] = -1
    
    F_dx = fft2(dx)
    F_dy = fft2(dy)
    
    # Eigenvalues of D^T D
    F_dtd = np.abs(F_dx)**2 + np.abs(F_dy)**2
    
    return F_dx, F_dy, F_dtd

def init_kernel(shape):
    """
    Initialize with a TIGHT Gaussian.
    Avoids the local minimum of a wide blob.
    """
    k = np.zeros(shape, dtype=np.float32)
    mid_h, mid_w = shape[0]//2, shape[1]//2
    # Small sigma to be close to delta, but smooth enough to calculate gradients
    sigma = 1.0 
    y, x = np.ogrid[-mid_h:mid_h+1, -mid_w:mid_w+1]
    
    if y.shape[0] > shape[0]: y = y[:-1]
    if x.shape[1] > shape[1]: x = x[:-1]
    
    h = np.exp(-(x**2 + y**2) / (2.*sigma**2))
    return h / h.sum()

def center_kernel(kernel):
    """
    Center of Mass alignment to prevent drift.
    """
    h, w = kernel.shape
    cy, cx = center_of_mass(kernel)
    ty, tx = h // 2, w // 2
    dy = ty - cy
    dx = tx - cx
    
    shifted = shift(kernel, (dy, dx), order=1, mode='constant', cval=0.0)
    shifted = np.maximum(shifted, 0)
    return shifted / (np.sum(shifted) + 1e-12)

def project_simplex(v):
    """
    Projects vector v onto the probability simplex (sum=1, >=0).
    Reference: Duchi et al. (2008).
    """
    v_flat = v.flatten()
    n = len(v_flat)
    u = np.sort(v_flat)[::-1]
    cssv = np.cumsum(u)
    ind = np.arange(n) + 1
    cond = u - (cssv - 1) / ind > 0
    
    if not np.any(cond):
        return np.maximum(v, 0) / (np.sum(np.maximum(v, 0)) + 1e-12)
        
    rho = ind[cond][-1]
    theta = (cssv[rho - 1] - 1) / rho
    w = np.maximum(v_flat - theta, 0)
    return w.reshape(v.shape)

def build_pyramid(image, n_scales, scale_factor):
    """Gaussian pyramid."""
    pyramid = [image]
    current = image
    for _ in range(n_scales - 1):
        h_new = int(current.shape[0] * scale_factor)
        w_new = int(current.shape[1] * scale_factor)
        if h_new < 16 or w_new < 16: break
        current = zoom(current, (h_new / current.shape[0], w_new / current.shape[1]), order=3)
        pyramid.append(current)
    return pyramid[::-1]

def pad_image(img, pad_size):
    """Reflective padding to handle boundaries."""
    return np.pad(img, ((pad_size, pad_size), (pad_size, pad_size)), mode='edge')

def crop_image(img, pad_size):
    """Crop back to original size."""
    return img[pad_size:-pad_size, pad_size:-pad_size]