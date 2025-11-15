# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for blind deconvolution algorithms."""
import numpy as np
from scipy import signal, ndimage
from typing import Tuple, Callable


def generate_test_image(size: int = 64) -> np.ndarray:
    """Generate a synthetic test image with clear features."""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    
    # Create a test pattern with various frequencies
    image = np.zeros((size, size))
    image += 0.5 * np.exp(-(x**2 + y**2) / 0.2)  # Central Gaussian
    image += 0.3 * np.sin(8 * np.pi * x) * np.exp(-y**2 / 0.1)  # Horizontal stripes
    image += 0.2 * np.sin(8 * np.pi * y) * np.exp(-x**2 / 0.1)  # Vertical stripes
    
    # Add some point sources
    image[size//4, size//4] += 0.8
    image[3*size//4, 3*size//4] += 0.6
    
    return np.clip(image, 0, 1)


def generate_blur_kernel(kernel_type: str, size: int = 15) -> np.ndarray:
    """Generate different types of blur kernels."""
    if kernel_type == "gaussian":
        sigma = size / 6
        kernel = np.zeros((size, size))
        center = size // 2
        for i in range(size):
            for j in range(size):
                kernel[i, j] = np.exp(-((i - center)**2 + (j - center)**2) / (2 * sigma**2))
        return kernel / np.sum(kernel)
    
    elif kernel_type == "motion":
        kernel = np.zeros((size, size))
        center = size // 2
        length = size // 2
        for i in range(length):
            kernel[center, center - length//2 + i] = 1
        return kernel / np.sum(kernel)
    
    elif kernel_type == "defocus":
        kernel = np.zeros((size, size))
        center = size // 2
        radius = size // 4
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        kernel[mask] = 1
        return kernel / np.sum(kernel)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, noise_level, image.shape)
    return np.clip(image + noise, 0, 1)


def convolve_with_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve image with blur kernel."""
    return signal.convolve2d(image, kernel, mode='same', boundary='symm')


def lucy_richardson_iteration(current_estimate: np.ndarray, 
                            observed_image: np.ndarray,
                            psf: np.ndarray) -> np.ndarray:
    """Single Lucy-Richardson iteration."""
    # Forward convolution
    convolved = signal.convolve2d(current_estimate, psf, mode='same', boundary='symm')
    
    # Avoid division by zero
    convolved = np.maximum(convolved, 1e-10)
    
    # Ratio image
    ratio = observed_image / convolved
    
    # Backproject with flipped PSF
    psf_flipped = np.flipud(np.fliplr(psf))
    correction = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='symm')
    
    # Update estimate
    return current_estimate * correction


def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_image_gradient_norm(image: np.ndarray) -> float:
    """Compute the L2 norm of the image gradient."""
    grad_x = np.gradient(image, axis=1)
    grad_y = np.gradient(image, axis=0)
    return np.sqrt(np.mean(grad_x**2 + grad_y**2))


def compute_residual_norm(current: np.ndarray, previous: np.ndarray) -> float:
    """Compute normalized change between iterations."""
    diff = np.abs(current - previous)
    return np.mean(diff) / (np.mean(np.abs(current)) + 1e-10)


def run_lucy_richardson_with_stopping_criteria(
    observed_image: np.ndarray,
    psf: np.ndarray,
    stopping_function: Callable,
    max_iterations: int = 200
) -> Tuple[np.ndarray, int, float]:
    """Run Lucy-Richardson with custom stopping criteria.
    
    Returns:
        final_image: The deconvolved image
        iterations_used: Number of iterations before stopping
        final_psnr: PSNR if ground truth available, otherwise -1
    """
    current_estimate = observed_image.copy()
    
    for iteration in range(max_iterations):
        previous_estimate = current_estimate.copy()
        current_estimate = lucy_richardson_iteration(current_estimate, observed_image, psf)
        
        # Check stopping criteria
        if stopping_function(current_estimate, previous_estimate, iteration, psf, observed_image):
            return current_estimate, iteration + 1, -1
    
    # Max iterations reached
    return current_estimate, max_iterations, -1