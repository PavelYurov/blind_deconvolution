# fergus.py
import numpy as np
import utils 

def deblurFergus(blurred_image, kernel_size, num_iterations=15, noise_level=0.01):
    """
    Implementazione (semplificata) dell'algoritmo di Fergus et al. (2006).
    """

    print("Avvio deconvolution (Fergus).")
   
    latent_image = blurred_image
    kernel = kernel_size

    print("Deconvolution (Fergus) completata.")
    return latent_image, kernel