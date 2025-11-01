import cv2
import numpy as np
import os
import glob
import shutil
import math
from typing import Tuple
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# --- Импорты из вашего фреймворка ---
import processing as pr
from filters.base import FilterBase
from filters.blur import DefocusBlur, MotionBlur, BSpline_blur
from filters.distributions import gaussian_distribution, uniform_distribution, ring_distribution, exponential_decay_distribution
# --- ИЗМЕНЕНИЕ: Импортируем правильные классы шума ---
from filters.noise import GaussianNoise, PoissonNoise, SaltAndPepperNoise, Pink_Noise, Brown_Noise

class DatasetGenerator:
    """
    Класс для генерации датасета с использованием фреймворка Processing.
    Объединяет логику из ImageFilterProcessor и методы связывания из Processing.
    """
    def __init__(self, processing_instance: pr.Processing,
                 input_dir='images_dataset/original',
                 output_dir='images_dataset/distorted',
                 kernel_dir='images_dataset/ground_truth_filters',
                 kernel_data_dir='images_dataset/kernel_data'):

        self.proc = processing_instance
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.kernel_dir = kernel_dir
        self.kernel_data_dir = kernel_data_dir
        
        self.max_kernel_size = 31
        self.kernel_ksize = 31

        shape_points = np.column_stack([[-1.0,2.0,0.0,-1.0,3.0],[0.0,2.0,-1.0,-1.5,1.5]])
        intensity_points = np.column_stack([[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],[0.0,0.9,0.1,0.3,0.7,0.7,0.1,0.0,0.2,0.35,0.0]])

        _defocus_disk_config = {'name': 'defocusdisk', 'class': DefocusBlur, 'params': {'psf': uniform_distribution, 'param': 4.0}}
        _motion_linear_config = {'name': 'motionlinearuniform', 'class': MotionBlur, 'params': {'psf': uniform_distribution, 'param': 7.0, 'angle': 30}}
        
        _bspline_config_simple = {
            'name': 'motionbsplinesimplecurve',
            'class': BSpline_blur,
            'params': {
                'output_size': (self.kernel_ksize, self.kernel_ksize),
                'shape_points': shape_points,
                'intensity_points': intensity_points,
                'shape_degree': 2,
                'intensity_degree': 3,
            }
        }

        _motion_major_axis = {
            'name': '_major_axis', 'class': MotionBlur,
            'params': {'psf': gaussian_distribution, 'param': 8.0, 'angle': 25}
        }
        _motion_minor_axis = {
            'name': '_minor_axis', 'class': MotionBlur,
            'params': {'psf': gaussian_distribution, 'param': 3.0, 'angle': 115}
        }

        self.blur_configs = [
            {'name': 'defocusgaussian', 'class': DefocusBlur, 'params': {'psf': gaussian_distribution, 'param': 2.0}},
            {'name': 'defocusdisk', 'class': DefocusBlur, 'params': {'psf': uniform_distribution, 'param': 4.0}},
            {'name': 'defocusring', 'class': DefocusBlur, 'params': {'psf': ring_distribution, 'param': 4.0}},
            {'name': 'motionlinearuniform', 'class': MotionBlur, 'params': {'psf': uniform_distribution, 'param': 7.0, 'angle': 30}},
            {'name': 'motionlinearexp', 'class': MotionBlur, 'params': {'psf': exponential_decay_distribution, 'param': 7.0, 'angle': 120}},
            _bspline_config_simple,
            {
                'name': 'stretched_gaussian_kernel',
                'generator_func': self._generate_convolved_kernel,
                'params': {
                    'ksize': self.kernel_ksize,
                    'configs': [_motion_major_axis, _motion_minor_axis]
                }
            },
            {
                'name': 'convolved_kernel',
                'generator_func': self._generate_convolved_kernel,
                'params': {
                    'ksize': self.kernel_ksize,
                    'configs': [_defocus_disk_config, _motion_linear_config, _bspline_config_simple]
                }
            },
        ]
        
        # --- ИЗМЕНЕНИЕ: Используем правильные классы шума ---
        self.noise_configs = [
            {'name': 'gaussian', 'class': GaussianNoise, 'params': {'param': 3.0}},
            {'name': 'poisson', 'class': PoissonNoise, 'params': {'param': 0.03}},
            {'name': 'saltpepper', 'class': SaltAndPepperNoise, 'params': {'param': (1, 1, 1000)}},
            {'name': 'pink', 'class': Pink_Noise, 'params': {'noise_level': 0.02}}, # Использует параметры по умолчанию
            {'name': 'brown', 'class': Brown_Noise, 'params': {'noise_level': 0.05}},
        ]

    def _generate_convolved_kernel(self, ksize: int, configs: list, **kwargs) -> np.ndarray:
        base_kernels = []
        for config in configs:
            k = None
            params = config['params'].copy()
            blur_class = config['class']
            if blur_class == DefocusBlur:
                ksize_calc = int(6 * params.get('param', 1.0)) | 1
                params['kernel_size'] = min(ksize_calc, self.max_kernel_size)
                k = blur_class(**params).generate_kernel()
            elif blur_class == MotionBlur:
                ksize_calc = max(int(4 * params.get('param', 1.0)) | 1, 3)
                params['kernel_length'] = min(ksize_calc, self.max_kernel_size)
                k = blur_class(**params).generate_kernel()
            elif blur_class == BSpline_blur:
                params['output_size'] = (ksize, ksize)
                k = blur_class(**params).create_dual_bspline_psf()
            if k is not None: base_kernels.append(k)
        if not base_kernels: return np.zeros((ksize, ksize))
        total_size = sum(s.shape[0] for s in base_kernels) | 1
        convolved_kernel = np.zeros((total_size, total_size))
        center = total_size // 2
        convolved_kernel[center, center] = 1.0
        for k_conv in base_kernels:
            convolved_kernel = cv2.filter2D(convolved_kernel, -1, k_conv)
        start = center - (ksize // 2)
        end = start + ksize
        final_kernel = convolved_kernel[start:end, start:end]
        return final_kernel / final_kernel.sum() if final_kernel.sum() > 0 else final_kernel

    def _visualize_kernels(self, visualizations: list):
        if not visualizations: return
        print("\nОтображение сгенерированных ядер...")
        num_kernels = len(visualizations)
        cols = int(math.ceil(math.sqrt(num_kernels)))
        rows = int(math.ceil(num_kernels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        fig.suptitle('Сгенерированные ядра (PSF)', fontsize=16)
        axes = np.array(axes).flatten()
        for i, (name, kernel_img) in enumerate(visualizations):
            ax = axes[i]
            h, w = kernel_img.shape[:2]
            ax.imshow(kernel_img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f"{name}\n(Размер: {h}x{w})", fontsize=10)
            ax.axis('off')
        for i in range(num_kernels, len(axes)):
            axes[i].axis('off')
        plt.subplots_adjust(hspace=0.25, top=0.9)
        plt.show()

    def generate_and_save_kernels(self) -> Tuple[dict, list]:
        print("--- Генерация и сохранение ядер ---")
        kernel_paths = {}
        visualizations = []
        for config in self.blur_configs:
            name = config['name']
            kernel = None
            if 'generator_func' in config:
                params = config.get('params', {}).copy()
                kernel = self._generate_convolved_kernel(**params)
            elif 'class' in config:
                params = config['params'].copy()
                blur_class = config['class']
                if blur_class == DefocusBlur:
                    ksize = int(6 * params.get('param', 1.0)) | 1
                    params['kernel_size'] = min(ksize, self.max_kernel_size)
                    kernel = blur_class(**params).generate_kernel()
                elif blur_class == MotionBlur:
                    ksize = max(int(4 * params.get('param', 1.0)) | 1, 3)
                    params['kernel_length'] = min(ksize, self.max_kernel_size)
                    kernel = blur_class(**params).generate_kernel()
                elif blur_class == BSpline_blur:
                    params['output_size'] = (self.kernel_ksize, self.kernel_ksize)
                    kernel = blur_class(**params).create_dual_bspline_psf()
            npy_path = os.path.join(self.kernel_data_dir, f"{name}.npy")
            np.save(npy_path, kernel)
            kernel_img = (kernel / np.max(kernel)) * 255 if np.max(kernel) > 0 else kernel
            kernel_img = kernel_img.astype(np.uint8)
            png_path = os.path.join(self.kernel_dir, f"{name}.png")
            cv2.imwrite(png_path, kernel_img)
            kernel_paths[name] = {'npy': npy_path, 'png': png_path}
            print(f"  -> Ядро '{name}' сохранено (размер: {kernel.shape[0]}x{kernel.shape[1]}).")
            visualizations.append((name, kernel_img))
        return kernel_paths, visualizations

    def process_and_bind_images(self):
        for directory in [self.output_dir, self.kernel_dir, self.kernel_data_dir]:
            if os.path.exists(directory): shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"Создана папка: '{directory}'")
        
        all_kernels, visualizations = self.generate_and_save_kernels()
        image_paths = glob.glob(os.path.join(self.input_dir, '*.[pP][nN][gG]')) + \
                      glob.glob(os.path.join(self.input_dir, '*.[jJ][pP][gG]'))
        
        if not image_paths:
            print("\nВ папке 'images_dataset/original' не найдено изображений для обработки. Завершение работы.")
            self._visualize_kernels(visualizations)
            return

        print(f"\nНайдено изображений для обработки: {len(image_paths)}\n")
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"--- Обработка изображения: {os.path.basename(image_path)} ---")
            
            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"  ! Не удалось прочитать изображение: {image_path}")
                continue
            
            for blur_name, paths in all_kernels.items():
                kernel = np.load(paths['npy'])
                blurred_gray = cv2.filter2D(image_gray, -1, kernel)
                output_filename = f"{base_name}_{blur_name}.png"
                output_path_blurred = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path_blurred, blurred_gray)
                print(f"  -> Сохранено: {output_path_blurred}")
                self.proc.bind(
                    original_image_path=image_path,
                    blurred_image_path=output_path_blurred,
                    original_kernel_path=paths['png'],
                    filter_description=blur_name,
                    color=self.proc.color
                )

                for noise_config in self.noise_configs:
                    noise_name = noise_config['name']
                    noise_filter = noise_config['class'](**noise_config['params'])
                    noisy_gray = noise_filter.filter(blurred_gray)
                    output_filename_noisy = f"{base_name}_{blur_name}_{noise_name}.png"
                    output_path_noisy = os.path.join(self.output_dir, output_filename_noisy)
                    cv2.imwrite(output_path_noisy, noisy_gray)
                    print(f"  -> Сохранено: {output_path_noisy}")
                    self.proc.bind(
                        original_image_path=image_path,
                        blurred_image_path=output_path_noisy,
                        original_kernel_path=paths['png'],
                        filter_description=f"{blur_name}_{noise_name}",
                        color=self.proc.color
                    )
        self._visualize_kernels(visualizations)

if __name__ == "__main__":
    print("Запуск генерации датасета с использованием фреймворка...\n")
    
    proc_instance = pr.Processing(
        images_folder='images_dataset/original', 
        blurred_folder='images_dataset/distorted',
        restored_folder='restored', 
        data_path='data', 
        color=False, 
        kernel_dir='images_dataset/ground_truth_filters',
        dataset_path='images_dataset'
    )
    proc_instance.clear_input()
    generator = DatasetGenerator(processing_instance=proc_instance)
    generator.process_and_bind_images()
    output_json_path = os.path.join("images_dataset", "dataset_bind.json")
    if proc_instance.images.size > 0:
        proc_instance.save_bind_state(output_json_path)
        print(f"\nОбработка завершена. Вся информация о датасете сохранена в '{output_json_path}'")
    else:
        print("\nОбработка завершена, но не было создано ни одного искаженного изображения. JSON-файл не сохранен.")