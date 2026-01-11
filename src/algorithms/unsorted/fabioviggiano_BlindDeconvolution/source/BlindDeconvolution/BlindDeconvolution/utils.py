# utils.py (Versione 2.0 - Aggiornata con blur gaussiano)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def loadImage(path, grayscale=True, normalize=True):
    try:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Impossibile trovare l'immagine al percorso: {path}")
            
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if normalize:
            img = img.astype(np.float32) / 255.0
            
        return img
    except Exception as e:
        print(f"Errore durante il caricamento dell'immagine: {e}")
        return None

def psf2otf(psf, output_shape):
    """
    Converte una Point Spread Function (PSF, il nostro kernel) in una Optical Transfer Function (OTF)
    della dimensione richiesta per la convoluzione nel dominio della frequenza.
    
    Args:
        psf (np.ndarray): Il kernel di blur.
        output_shape (tuple): La dimensione dell'immagine target (es. `immagine.shape`).
        
    Returns:
        np.ndarray: L'OTF (complessa) pronta per la moltiplicazione in F-domain.
    """
    psf_shape = psf.shape
    pad_height = output_shape[0] - psf_shape[0]
    pad_width = output_shape[1] - psf_shape[1]
    
    # Esegue il padding per portare il kernel alla stessa dimensione dell'immagine
    padded_psf = np.pad(psf, 
                        ((0, pad_height), (0, pad_width)), 
                        'constant')
    
    # Sposta il centro del kernel all'origine (0,0) per la FFT
    padded_psf = np.roll(padded_psf, -int(psf_shape[0] // 2), axis=0)
    padded_psf = np.roll(padded_psf, -int(psf_shape[1] // 2), axis=1)
    
    # Calcola la Fast Fourier Transform 2D
    return np.fft.fft2(padded_psf)

def showResults(original, kernel, deblurred):
    """Visualizza l'immagine sfocata, il kernel stimato e l'immagine deblurred."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Immagine Sfocata (Input)")
    axes[0].axis('off')
    
    # Normalizza il kernel per una visualizzazione ottimale
    kernel_display = kernel / kernel.max() if kernel.max() > 0 else kernel
    axes[1].imshow(kernel_display, cmap='gray')
    axes[1].set_title("Kernel Stimato")
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(deblurred, 0, 1), cmap='gray')
    axes[2].set_title("Immagine Ricostruita (Output)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# --- NUOVA FUNZIONE PER CREARE UN KERNEL GAUSSIANO ---
def createGaussianKernel(kernel_size=35, sigma=5):
    """
    Crea un kernel di blur Gaussiano 2D.
    
    Args:
        kernel_size (int): La dimensione del kernel (deve essere dispari).
        sigma (float): La deviazione standard della gaussiana. Più è alta, più il blur è esteso.
        
    Returns:
        np.ndarray: Il kernel gaussiano normalizzato.
    """
    # Crea un kernel 1D per l'asse X
    kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
    # Crea un kernel 1D per l'asse Y
    kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
    # Combina i due kernel 1D per ottenere un kernel 2D tramite prodotto esterno
    kernel = kernel_x * kernel_y.T
    
    # Assicura che la somma del kernel sia 1 per conservare la luminosità dell'immagine
    kernel /= kernel.sum()
    
    return kernel

# --- FUNZIONE createSyntheticBlur AGGIORNATA PER GESTIRE PIÙ TIPI DI BLUR ---
def createSyntheticBlur(image, kernel_size=35, blur_type='motion', motion_angle=45, motion_len=50, gaussian_sigma=5):
    """
    Crea un'immagine sfocata artificialmente a partire da una nitida.
    Restituisce l'immagine sfocata e il kernel ground truth.
    
    Args:
        image (np.ndarray): L'immagine di input nitida.
        kernel_size (int): La dimensione del kernel da generare.
        blur_type (str): Tipo di blur da applicare ('motion' o 'gaussian').
        motion_angle (float): Angolo del motion blur in gradi (usato solo se blur_type='motion').
        motion_len (int): Lunghezza della scia del motion blur (usato solo se blur_type='motion').
        gaussian_sigma (float): Deviazione standard per il blur gaussiano (usato solo se blur_type='gaussian').

    Returns:
        tuple[np.ndarray, np.ndarray]: Una tupla contenente (immagine_sfocata, kernel_ground_truth).
    """
    if blur_type == 'motion':
        # Crea il kernel di motion blur
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        
        dx = np.cos(np.deg2rad(motion_angle))
        dy = np.sin(np.deg2rad(motion_angle))
        
        for i in range(motion_len):
            x = int(round(center + i * dx))
            y = int(round(center + i * dy))
            if 0 <= y < kernel_size and 0 <= x < kernel_size:
                kernel[y, x] = 1
                
        if kernel.sum() > 0:
            kernel /= kernel.sum() # Normalizza il kernel
            
    elif blur_type == 'gaussian':
        # Chiama la nuova funzione per creare il kernel gaussiano
        kernel = createGaussianKernel(kernel_size, gaussian_sigma)
        
    else:
        raise ValueError("Tipo di blur non supportato. Scegli tra 'motion' e 'gaussian'.")
    
    # Applica il kernel all'immagine tramite convoluzione 2D
    blurred_image = cv2.filter2D(image, -1, kernel)
    
    return blurred_image, kernel

def calculateMetrics(ground_truth, reconstructed):
    """
    Calcola PSNR e SSIM tra l'immagine originale e quella ricostruita.
    Le immagini devono essere in float [0, 1].
    
    Args:
        ground_truth (np.ndarray): L'immagine nitida originale.
        reconstructed (np.ndarray): L'immagine ottenuta dopo la deconvolution.
        
    Returns:
        dict: Un dizionario contenente i valori di 'psnr' e 'ssim'.
    """
    # Assicurati che le immagini abbiano lo stesso tipo di dati per le metriche
    gt = ground_truth.astype(reconstructed.dtype)

    psnr = peak_signal_noise_ratio(gt, reconstructed, data_range=1.0)
    ssim = structural_similarity(gt, reconstructed, data_range=1.0, channel_axis=None) # channel_axis=None per grayscale
    
    metrics = {
        'psnr': float(psnr),
        'ssim': float(ssim)
    }
    
    print(f"Metriche di Valutazione (vs Ground Truth):")
    print(f"  - PSNR: {metrics['psnr']:.2f} dB")
    print(f"  - SSIM: {metrics['ssim']:.4f}")
    return metrics

def calculateSharpness(image):
    """
    Calcola un indice di nitidezza usando la varianza del Laplaciano.
    Un valore più alto indica maggiore nitidezza.
    """
    # Usiamo ddepth=-1 per far sì che l'output abbia lo stesso tipo di dati dell'input (float32),
    # che è sufficiente per gestire valori negativi ed evita problemi di compatibilità.
    return cv2.Laplacian(image, -1).var()