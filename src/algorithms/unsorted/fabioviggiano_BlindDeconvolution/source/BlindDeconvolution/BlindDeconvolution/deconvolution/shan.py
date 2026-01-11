# shan.py (Versione 2.1 - con iperparametro configurabile)
import numpy as np
import cv2
import utils

# MODIFICA 1: Aggiunto "lambda_kernel_reg" agli argomenti della funzione
def deblurShan(blurred_image, kernel_size, num_iterations, lambda_prior, lambda_kernel_reg, initial_kernel):
    """
    Funzione "operaia" che esegue i passi di ottimizzazione alternata 
    per la stima dell'immagine latente e del kernel.
    """
    kernel = initial_kernel.copy()
    
    latent_image = blurred_image.copy()
    B_fft = np.fft.fft2(blurred_image)
    dx = np.array([[-1, 1]]); dy = np.array([[-1], [1]])
    Dx_otf = utils.psf2otf(dx, blurred_image.shape)
    Dy_otf = utils.psf2otf(dy, blurred_image.shape)
    
    # MODIFICA 2: La riga "lambda_kernel = 1e-3" è stata rimossa.
    # Ora usiamo il valore passato come argomento.

    for i in range(num_iterations):
        # --- Step 1: Stima dell'immagine latente (L) ---
        K_otf = utils.psf2otf(kernel, blurred_image.shape)
        prior_term = lambda_prior * (np.conj(Dx_otf) * Dx_otf + np.conj(Dy_otf) * Dy_otf)
        numerator = np.conj(K_otf) * B_fft
        denominator = np.conj(K_otf) * K_otf + prior_term
        L_fft = numerator / (denominator + 1e-8) # Aggiunto epsilon per stabilità
        latent_image = np.real(np.fft.ifft2(L_fft))
        
        # --- Step 2: Stima del kernel (K) ---
        L_otf = utils.psf2otf(latent_image, blurred_image.shape)
        numerator_grad_K = np.conj(L_otf) * B_fft
        # MODIFICA 3: Usiamo "lambda_kernel_reg" invece del valore fisso
        denominator_grad_K = np.conj(L_otf) * L_otf + lambda_kernel_reg
        K_fft_est = numerator_grad_K / (denominator_grad_K + 1e-8) # Aggiunto epsilon per stabilità
        kernel_est = np.real(np.fft.ifft2(K_fft_est))
        
        # --- Step 3: Proiezione e normalizzazione del kernel ---
        k_h, k_w = kernel.shape
        center_y, center_x = kernel_est.shape[0] // 2, kernel_est.shape[1] // 2
        
        kernel = kernel_est[center_y - k_h//2 : center_y + k_h//2 + 1, center_x - k_w//2 : center_x + k_w//2 + 1]
        kernel[kernel < 0] = 0 # Proiezione (il kernel non può avere valori negativi)
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-6: 
            kernel /= kernel_sum # Normalizzazione (l'energia totale è 1)

    return latent_image, kernel

# MODIFICA 4: Aggiunto "lambda_kernel_reg" con un valore di default
def deblurShanPyramidal(blurred_image, kernel_size, num_iterations=15, lambda_prior=5e-3, lambda_kernel_reg=1e-3, num_levels=4):
    """
    Versione piramidale che orchestra la stima del kernel a diverse risoluzioni.
    """
    print("Inizio deconvolution con l'algoritmo di Shan (Piramidale v2.1)...")

    # Creazione della piramide di immagini
    pyramid = [blurred_image]
    for i in range(num_levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    pyramid.reverse()

    # Inizializzazione del kernel al livello più basso (un singolo impulso)
    k = np.zeros((kernel_size, kernel_size)); 
    k[kernel_size//2, kernel_size//2] = 1.0

    # Loop attraverso i livelli della piramide, dal più piccolo al più grande
    for level, image_level in enumerate(pyramid):
        print(f"--- Processando livello piramidale {level+1}/{num_levels} (Dimensioni: {image_level.shape}) ---")
        
        # Eseguiamo l'algoritmo a questa scala, partendo dal kernel precedente
        # MODIFICA 5: Passiamo "lambda_kernel_reg" alla funzione operaia
        _, k = deblurShan(image_level, k.shape[0], num_iterations, lambda_prior, lambda_kernel_reg, k)

        # Se non siamo all'ultimo livello, ingrandiamo il kernel per il prossimo
        if level < num_levels - 1:
            # Upscaling del kernel per il livello successivo
            k_up = cv2.pyrUp(k) * 4 # Moltiplichiamo per 4 per conservare l'energia
            
            new_h, new_w = k_up.shape
            center_y, center_x = new_h // 2, new_w // 2
            
            # Ritagliamo per assicurarci che non superi la dimensione massima originale
            crop_h, crop_w = kernel_size // 2, kernel_size // 2
            k = k_up[center_y - crop_h : center_y + crop_h + 1,
                     center_x - crop_w : center_x + crop_w + 1]
            
            if k.sum() > 0: 
                k /= k.sum()

    print("Stima finale del kernel completata. Eseguo deconvolution finale sull'immagine originale...")
    
    # Usiamo il kernel stimato per la deconvolution finale non-blind sull'immagine full-res
    # MODIFICA 6: Passiamo "lambda_kernel_reg" anche nella chiamata finale
    final_image, final_kernel = deblurShan(blurred_image, k.shape[0], num_iterations, lambda_prior, lambda_kernel_reg, k)

    return final_image, final_kernel