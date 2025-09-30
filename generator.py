import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil
import math
from typing import Tuple
from filters.distributions import *
from filters.blur import *
from filters.noise import *

class Generator():
    def __init__(self, input_dir='HR', output_dir='filtered', kernel_dir='kernel for bluring'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.kernel_dir = kernel_dir

        self.blur_configs = [
            {'name': 'defocus_gaussian', 'class': DefocusBlur, 'params': {'psf': gaussian_distribution, 'param': 8.0}},
            {'name': 'defocus_disk', 'class': DefocusBlur, 'params': {'psf': uniform_distribution, 'param': 10.0}},
            {'name': 'defocus_cone', 'class': DefocusBlur, 'params': {'psf': linear_decay_distribution, 'param': 12.0}},
            {'name': 'defocus_ring', 'class': DefocusBlur, 'params': {'psf': ring_distribution, 'param': 9.0}},
            {'name': 'motion_linear_uniform', 'class': MotionBlur, 'params': {'psf': uniform_distribution, 'param': 15.0, 'angle': 30}},
            {'name': 'motion_linear_exp', 'class': MotionBlur, 'params': {'psf': exponential_decay_distribution, 'param': 10.0, 'angle': 120}},
        ]

        self.noise_configs = [
            {'name': 'gaussian', 'class': GaussianNoise, 'params': {'param': 3.0}},
            {'name': 'poisson', 'class': PoissonNoise, 'params': {'param': 0.05}},
            {'name': 'saltpepper', 'class': SaltAndPepperNoise, 'params': {'param': (1, 1, 5000)}}
        ]

    def _apply_all_filters(self, image: np.ndarray) -> Tuple[dict, dict]:
        """
        Примените к изображению всех фильтров размытия и шума
        """
        results, kernels = {}, {}

        for blur_config in self.blur_configs:
            blur_name = blur_config['name']
            blur_filter = blur_config['class'](**blur_config['params'])
            kernel = blur_filter.generate_kernel()
            kernels[blur_name] = (kernel, f"{blur_name}\nSize: {kernel.shape[0]}x{kernel.shape[1]}")
            blurred_image = blur_filter.filter(image)

            for noise_config in self.noise_configs:
                noise_name = noise_config['name']
                noise_filter = noise_config['class'](**noise_config['params'])
                noisy_image = noise_filter.filter(blurred_image)
                final_name = f"{blur_name}_{noise_name}"
                results[final_name] = noisy_image
            
        return results, kernels

    def _show_plot(self, kernels: dict):
        """
        Отображение сгенерированных ядер в виде сетки
        """
        kernel_items = list(kernels.items())
        num_kernels = len(kernel_items)
        cols = 4
        rows = math.ceil(num_kernels / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        plt.style.use('dark_background')
        fig.suptitle('Generated Blur Kernels', fontsize=16)
        axes = axes.flatten()

        for i, (name, (kernel, description)) in enumerate(kernel_items):
            axes[i].imshow(kernel, cmap='gray')
            axes[i].set_title(description)
            axes[i].axis('off')

        for i in range(num_kernels, len(axes)):
            axes[i].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def _save_kernels(self, kernels: dict):
        """
        Сохранение ядер в виде файлов изображений
        """
        print("\n--- Saving kernels as images ---")
        for name, (kernel, _) in kernels.items():
            if np.max(kernel) > 0:
                kernel_img = (kernel / np.max(kernel)) * 255
            else:
                kernel_img = kernel
            kernel_img = kernel_img.astype(np.uint8)
            output_path = os.path.join(self.kernel_dir, f"{name}_kernel.png")
            cv2.imwrite(output_path, kernel_img)
            print(f"  -> Saved kernel: {output_path}")

    def process_images(self):
        """
        Основной метод обработки всех изображений во входном каталоге
        """
        for directory in [self.output_dir, self.kernel_dir]:
            if os.path.exists(directory):
                print(f"Removing existing directory: '{directory}'...")
                shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"Created new directory: '{directory}'")

        image_paths = glob.glob(os.path.join(self.input_dir, '*.[pP][nN][gG]')) + \
                      glob.glob(os.path.join(self.input_dir, '*.[jJ][pP][gG]')) + \
                      glob.glob(os.path.join(self.input_dir, '*.[jJ][pP][eE][gG]'))
        if not image_paths:
            print(f"Error: No images found in directory '{self.input_dir}'.")
            return
        print(f"Found images to process: {len(image_paths)}")

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, kernels_for_plot = self._apply_all_filters(dummy_image)
        self._save_kernels(kernels_for_plot)
        print("\nDisplaying all generated kernels...")
        self._show_plot(kernels_for_plot)
        
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"\n--- Processing image: {os.path.basename(image_path)} ---")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}. Skipping.")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            original_path = os.path.join(self.output_dir, f"{base_name}_original.png")
            cv2.imwrite(original_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            print(f"  -> Saved: {original_path}")

            filtered_results, _ = self._apply_all_filters(image_rgb)
            
            for filter_name, result_image in filtered_results.items():
                output_filename = f"{base_name}_{filter_name}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, result_image_bgr)
                print(f"  -> Saved: {output_path}")

        print("\nAll images processed successfully.")

processor = Generator(
        input_dir='HR', 
        output_dir='filtered', 
        kernel_dir='kernel for bluring'
    )
processor.process_images()