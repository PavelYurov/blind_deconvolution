import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import center_of_mass, shift, label

def psf2otf(psf, shape):
    in_shape = psf.shape
    padded = np.zeros(shape)
    center_y, center_x = shape[0] // 2, shape[1] // 2
    kh, kw = in_shape
    start_y = center_y - kh // 2
    start_x = center_x - kw // 2
    padded[start_y:start_y+kh, start_x:start_x+kw] = psf
    padded = ifftshift(padded)
    return fft2(padded)

def otf2psf(otf, shape):
    psf = np.real(ifft2(otf))
    psf = fftshift(psf)
    return psf

def keep_largest_component(kernel, threshold=0.05):
    max_val = kernel.max()
    if max_val <= 1e-12: return kernel
    binary_mask = kernel > (max_val * threshold)
    labeled_array, num_features = label(binary_mask)
    if num_features <= 1: return kernel * binary_mask
    sums = [np.sum(kernel[labeled_array == i]) for i in range(1, num_features + 1)]
    if not sums: return kernel
    largest_label = np.argmax(sums) + 1
    return kernel * (labeled_array == largest_label)

def force_center_mass(kernel, threshold=0.05):
    kernel_clean = keep_largest_component(kernel, threshold=threshold)
    if kernel_clean.sum() == 0: kernel_clean = kernel
    kh, kw = kernel.shape
    cy, cx = center_of_mass(kernel_clean)
    target_y, target_x = kh // 2, kw // 2
    dy = target_y - cy
    dx = target_x - cx
    shifted = shift(kernel, shift=(dy, dx), order=3, mode='constant', cval=0.0)
    shifted = np.maximum(shifted, 0)
    s = np.sum(shifted)
    if s > 0: shifted /= s
    return shifted

def edgetaper(img, kernel_shape):
    rows, cols = img.shape
    kh, kw = kernel_shape
    nt_rows = min(rows, kh)
    nt_cols = min(cols, kw)
    taper_x = np.ones(cols)
    taper_y = np.ones(rows)
    if nt_cols > 0:
        x_ramp = np.sin(np.linspace(0, np.pi/2, nt_cols))**2
        taper_x[:nt_cols] = x_ramp
        taper_x[-nt_cols:] = x_ramp[::-1]
    if nt_rows > 0:
        y_ramp = np.sin(np.linspace(0, np.pi/2, nt_rows))**2
        taper_y[:nt_rows] = y_ramp
        taper_y[-nt_rows:] = y_ramp[::-1]
    window = np.outer(taper_y, taper_x)
    mean_val = np.mean(img)
    return img * window + mean_val * (1 - window)

def get_gradient_operators(shape):
    dh = np.zeros(shape); dh[0, 0] = -1; dh[0, 1] = 1
    dv = np.zeros(shape); dv[0, 0] = -1; dv[1, 0] = 1
    return fft2(dh), fft2(dv)

def conjugate_gradient(A_func, b, x0, max_iter=20, tol=1e-6):
    x = x0.copy()
    r = b - A_func(x)
    p = r.copy()
    rsold = np.sum(r * r)
    for i in range(max_iter):
        Ap = A_func(p)
        denom = np.sum(p * Ap)
        if abs(denom) < 1e-12: break
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.sum(r * r)
        if np.sqrt(rsnew) < tol: break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def wiener_deconvolution(img, kernel, snr=50.0):
    rows, cols = img.shape
    H = psf2otf(kernel, (rows, cols))
    G = fft2(img)
    H_conj = np.conj(H)
    H_sq = np.abs(H)**2
    F_hat = (G * H_conj) / (H_sq + (1.0 / snr))
    restored = np.real(ifft2(F_hat))
    return np.clip(restored, 0, 1)

def tikhonov_deconvolution(img, kernel, alpha=0.05):
    """
    Non-blind deconvolution using Tikhonov regularization.
    Minimizes ||H*x - y||^2 + alpha * ||L*x||^2
    where L is the Laplacian operator (penalizing roughness).
    """
    rows, cols = img.shape
    H = psf2otf(kernel, (rows, cols))
    G = fft2(img)
    
    # Laplacian operator for regularization
    laplacian = np.array([[0, -1, 0], 
                          [-1, 4, -1], 
                          [0, -1, 0]])
    L = psf2otf(laplacian, (rows, cols))
    
    H_conj = np.conj(H)
    H_sq = np.abs(H)**2
    L_sq = np.abs(L)**2
    
    # Tikhonov formula
    # F = (H* G) / (|H|^2 + alpha * |L|^2)
    numerator = H_conj * G
    denominator = H_sq + alpha * L_sq
    
    F_hat = numerator / (denominator + 1e-12)
    restored = np.real(ifft2(F_hat))
    
    return np.clip(restored, 0, 1)