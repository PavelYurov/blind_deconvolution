# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 17:07:51 2025

@author: Romulus Terebes
"""


import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from scipy.ndimage import median_filter
from skimage.util import random_noise
from skimage.filters import gaussian

def kuwahara_filter(image, window_size=5):
    rows, cols = image.shape
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    filtered_image = np.zeros_like(image)
    
    for i in range(rows):
        for j in range(cols):
            subregions = []
            variances = []
            
            for k in range(2):
                for l in range(2):
                    r_start = i + k * pad_size
                    c_start = j + l * pad_size
                    subregion = padded_image[r_start:r_start+window_size, c_start:c_start+window_size]
                    subregions.append(subregion)
                    variances.append(np.var(subregion))
            
            min_variance_index = np.argmin(variances)
            filtered_image[i, j] = np.mean(subregions[min_variance_index])
    
    return filtered_image

def calculate_ssim(imageA, imageB):
    # Compute SSIM between the two images
    ssim_value, ssim_map = ssim(imageA, imageB, full=True, data_range=255)
    return ssim_value

def mean_filter(image, size):
    # Apply the mean filter using OpenCV's blur function
    filtered_image = cv2.blur(image, (size, size))
    return filtered_image

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Create a copy of the image
    noisy_image = np.copy(image)
    
    # Add salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    
    # Add pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    
    return noisy_image

def add_gaussian_noise(image, mean=0, std=20):
    gaussian_noise = np.random.normal(mean, std, image.shape)
    noisy_image = image + gaussian_noise
    # Clip the values to be in the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def harmonic_mean_filter(image, window_size):
    rows, cols = image.shape
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    
    filtered_image = np.zeros_like(image)
    
    for i in range(rows):
        for j in range(cols):
            neighborhood = padded_image[i:i+window_size, j:j+window_size]
            # Calculate the harmonic mean
            harmonic_mean = len(neighborhood.flatten()) / np.sum(1.0 / (neighborhood + 1e-10))
            filtered_image[i, j] = harmonic_mean
    
    return filtered_image