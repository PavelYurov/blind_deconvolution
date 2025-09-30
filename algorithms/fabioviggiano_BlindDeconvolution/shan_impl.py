# shan.py (Versione 2.1 - con iperparametro configurabile)
import numpy as np
import cv2
try:
    # package context
    from . import utils
except Exception:  # pragma: no cover
    # script/run-from-folder context
    import utils


def _ensure_odd(n: int) -> int:
    n = int(max(1, n))
    return n if (n % 2 == 1) else n - 1


def _center_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    target_h = min(h, target_h)
    target_w = min(w, target_w)
    cy, cx = h // 2, w // 2
    th2, tw2 = target_h // 2, target_w // 2
    return arr[cy - th2 : cy + th2 + 1, cx - tw2 : cx + tw2 + 1]

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
    # Costruzione adattiva della piramide: fermarsi se l'immagine diventa troppo piccola
    for i in range(num_levels - 1):
        nxt = cv2.pyrDown(pyramid[-1])
        if nxt.shape[0] < 2 or nxt.shape[1] < 2:
            break
        pyramid.append(nxt)
    pyramid.reverse()
    # num_levels effettivi
    num_levels = len(pyramid)

    # Inizializzazione del kernel al livello più basso (un singolo impulso)
    # Assicuriamoci che la dimensione del kernel non superi l'immagine corrente
    min_side = min(pyramid[0].shape[:2])
    k_side = _ensure_odd(min(kernel_size, min_side))
    k = np.zeros((k_side, k_side), dtype=np.float32)
    k[k_side // 2, k_side // 2] = 1.0

    # Loop attraverso i livelli della piramide, dal più piccolo al più grande
    for level, image_level in enumerate(pyramid):
        print(f"--- Processando livello piramidale {level+1}/{num_levels} (Dimensioni: {image_level.shape}) ---")
        
        # Prima di eseguire l'algoritmo, garantiamo che il kernel non sia più grande dell'immagine
        max_k_side = _ensure_odd(min(image_level.shape[0], image_level.shape[1]))
        if k.shape[0] > max_k_side:
            k = _center_crop(k, max_k_side, max_k_side)
            if k.sum() > 0:
                k = k / k.sum()

        # Eseguiamo l'algoritmo a questa scala, partendo dal kernel precedente
        # MODIFICA 5: Passiamo "lambda_kernel_reg" alla funzione operaia
        _, k = deblurShan(image_level, k.shape[0], num_iterations, lambda_prior, lambda_kernel_reg, k)

        # Se non siamo all'ultimo livello, ingrandiamo il kernel per il prossimo
        if level < num_levels - 1:
            # Upscaling del kernel per il livello successivo
            k_up = cv2.pyrUp(k) * 4  # Conserva l'energia in media

            # Ritagliamo al lato target desiderato ma mai oltre la dimensione del kernel richiesto
            target_side = _ensure_odd(min(kernel_size, max(pyramid[level + 1].shape[:2])))
            k = _center_crop(k_up, target_side, target_side)

            if k.sum() > 0:
                k = k / k.sum()

    print("Stima finale del kernel completata. Eseguo deconvolution finale sull'immagine originale...")
    
    # Usiamo il kernel stimato per la deconvolution finale non-blind sull'immagine full-res
    # MODIFICA 6: Passiamo "lambda_kernel_reg" anche nella chiamata finale
    # Garantiamo che il kernel finale non superi l'immagine originale
    max_k_side_final = _ensure_odd(min(blurred_image.shape[0], blurred_image.shape[1]))
    if k.shape[0] > max_k_side_final:
        k = _center_crop(k, max_k_side_final, max_k_side_final)
        if k.sum() > 0:
            k = k / k.sum()

    final_image, final_kernel = deblurShan(
        blurred_image, k.shape[0], num_iterations, lambda_prior, lambda_kernel_reg, k
    )

    return final_image, final_kernel
