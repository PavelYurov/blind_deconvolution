"""
Модуль генерации датасета для экспериментов по слепой деконволюции.

Класс DatasetGenerator для автоматической генерации
датасета с различными типами размытия и шумов.

Авторы: Куропатов К.Л.
"""

import cv2
import numpy as np
import os
import glob
import shutil
import math
from typing import Tuple, Dict, List, Any
import matplotlib.pyplot as plt

# Импорты из фреймворка
import processing as pr
from filters.blur import DefocusBlur, MotionBlur, BSpline_blur
from filters.distributions import (
    gaussian_distribution,
    uniform_distribution,
    ring_distribution,
    exponential_decay_distribution
)
from filters.noise import (
    GaussianNoise,
    PoissonNoise,
    SaltAndPepperNoise,
    Pink_Noise,
    Brown_Noise
)


class DatasetGenerator:
    """
    Генератор датасета для экспериментов по слепой деконволюции.

    Класс автоматически создаёт искажённые изображения с различными
    комбинациями размытия и шума, сохраняет ядра (PSF) и связывает
    все данные через Processing.bind().

    Параметры
    ---------
    processing_instance : pr.Processing
        Экземпляр класса Processing для связывания данных.
    input_dir : str, optional
        Путь к папке с исходными изображениями.
    output_dir : str, optional
        Путь к папке для сохранения искажённых изображений.
    kernel_dir : str, optional
        Путь к папке для сохранения PNG-изображений ядер.
    kernel_data_dir : str, optional
        Путь к папке для сохранения NPY-файлов ядер.

    Атрибуты
    --------
    blur_configs : List[Dict]
        Конфигурации фильтров размытия.
    noise_configs : List[Dict]
        Конфигурации шумовых фильтров.
    """

    def __init__(
        self,
        processing_instance: pr.Processing,
        input_dir: str = 'images_dataset/original',
        output_dir: str = 'images_dataset/distorted',
        kernel_dir: str = 'images_dataset/ground_truth_filters',
        kernel_data_dir: str = 'images_dataset/kernel_data'
    ) -> None:
        self.proc = processing_instance
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.kernel_dir = kernel_dir
        self.kernel_data_dir = kernel_data_dir

        # Максимальный размер ядра
        self.max_kernel_size = 31
        self.kernel_ksize = 31

        # Контрольные точки для B-spline
        shape_points = np.column_stack([
            [-1.0, 2.0, 0.0, -1.0, 3.0],
            [0.0, 2.0, -1.0, -1.5, 1.5]
        ])
        intensity_points = np.column_stack([
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.0, 0.9, 0.1, 0.3, 0.7, 0.7, 0.1, 0.0, 0.2, 0.35, 0.0]
        ])

        # Базовые конфигурации для составных ядер
        _defocus_disk_config = {
            'name': 'defocusdisk',
            'class': DefocusBlur,
            'params': {'psf': uniform_distribution, 'param': 4.0}
        }
        _motion_linear_config = {
            'name': 'motionlinearuniform',
            'class': MotionBlur,
            'params': {'psf': uniform_distribution, 'param': 7.0, 'angle': 30}
        }
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
            'name': '_major_axis',
            'class': MotionBlur,
            'params': {'psf': gaussian_distribution, 'param': 8.0, 'angle': 25}
        }
        _motion_minor_axis = {
            'name': '_minor_axis',
            'class': MotionBlur,
            'params': {'psf': gaussian_distribution, 'param': 3.0, 'angle': 115}
        }

        # Конфигурации размытия
        self.blur_configs: List[Dict[str, Any]] = [
            {
                'name': 'defocusgaussian',
                'class': DefocusBlur,
                'params': {'psf': gaussian_distribution, 'param': 2.0}
            },
            {
                'name': 'defocusdisk',
                'class': DefocusBlur,
                'params': {'psf': uniform_distribution, 'param': 4.0}
            },
            {
                'name': 'defocusring',
                'class': DefocusBlur,
                'params': {'psf': ring_distribution, 'param': 4.0}
            },
            {
                'name': 'motionlinearuniform',
                'class': MotionBlur,
                'params': {'psf': uniform_distribution, 'param': 7.0, 'angle': 30}
            },
            {
                'name': 'motionlinearexp',
                'class': MotionBlur,
                'params': {
                    'psf': exponential_decay_distribution,
                    'param': 7.0,
                    'angle': 120
                }
            },
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
                    'configs': [
                        _defocus_disk_config,
                        _motion_linear_config,
                        _bspline_config_simple
                    ]
                }
            },
        ]

        # Конфигурации шума
        self.noise_configs: List[Dict[str, Any]] = [
            {'name': 'gaussian', 'class': GaussianNoise, 'params': {'param': 3.0}},
            {'name': 'poisson', 'class': PoissonNoise, 'params': {'param': 0.03}},
            {
                'name': 'saltpepper',
                'class': SaltAndPepperNoise,
                'params': {'param': (1, 1, 1000)}
            },
            {'name': 'pink', 'class': Pink_Noise, 'params': {'noise_level': 0.02}},
            {'name': 'brown', 'class': Brown_Noise, 'params': {'noise_level': 0.05}},
        ]

    def _generate_convolved_kernel(
        self,
        ksize: int,
        configs: List[Dict],
        **kwargs
    ) -> np.ndarray:
        """
        Генерация составного ядра путём свёртки нескольких базовых ядер.

        Параметры
        ---------
        ksize : int
            Размер выходного ядра.
        configs : List[Dict]
            Список конфигураций базовых ядер.

        Возвращает
        ----------
        np.ndarray
            Нормализованное составное ядро.
        """
        base_kernels = []

        for config in configs:
            kernel = None
            params = config['params'].copy()
            blur_class = config['class']

            if blur_class == DefocusBlur:
                ksize_calc = int(6 * params.get('param', 1.0)) | 1
                params['kernel_size'] = min(ksize_calc, self.max_kernel_size)
                kernel = blur_class(**params).generate_kernel()

            elif blur_class == MotionBlur:
                ksize_calc = max(int(4 * params.get('param', 1.0)) | 1, 3)
                params['kernel_length'] = min(ksize_calc, self.max_kernel_size)
                kernel = blur_class(**params).generate_kernel()

            elif blur_class == BSpline_blur:
                params['output_size'] = (ksize, ksize)
                kernel = blur_class(**params).create_dual_bspline_psf()

            if kernel is not None:
                base_kernels.append(kernel)

        if not base_kernels:
            return np.zeros((ksize, ksize))

        # Свёртка всех ядер
        total_size = sum(k.shape[0] for k in base_kernels) | 1
        convolved_kernel = np.zeros((total_size, total_size))
        center = total_size // 2
        convolved_kernel[center, center] = 1.0

        for kernel in base_kernels:
            convolved_kernel = cv2.filter2D(convolved_kernel, -1, kernel)

        # Обрезка до нужного размера
        start = center - (ksize // 2)
        end = start + ksize
        final_kernel = convolved_kernel[start:end, start:end]

        # Нормализация
        if final_kernel.sum() > 0:
            return final_kernel / final_kernel.sum()
        return final_kernel

    def _visualize_kernels(self, visualizations: List[Tuple[str, np.ndarray]]) -> None:
        """
        Визуализация сгенерированных ядер.

        Параметры
        ---------
        visualizations : List[Tuple[str, np.ndarray]]
            Список кортежей (имя, изображение ядра).
        """
        if not visualizations:
            return

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

    def generate_and_save_kernels(self) -> Tuple[Dict[str, Dict], List]:
        """
        Генерация и сохранение всех ядер размытия.

        Возвращает
        ----------
        kernel_paths : Dict[str, Dict]
            Словарь с путями к файлам ядер.
        visualizations : List[Tuple[str, np.ndarray]]
            Список данных для визуализации.
        """
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

            # Сохранение .npy
            npy_path = os.path.join(self.kernel_data_dir, f"{name}.npy")
            np.save(npy_path, kernel)

            # Сохранение .png
            kernel_img = kernel / np.max(kernel) * 255 if np.max(kernel) > 0 else kernel
            kernel_img = kernel_img.astype(np.uint8)
            png_path = os.path.join(self.kernel_dir, f"{name}.png")
            cv2.imwrite(png_path, kernel_img)

            kernel_paths[name] = {'npy': npy_path, 'png': png_path}
            print(f"  -> Ядро '{name}' сохранено ({kernel.shape[0]}x{kernel.shape[1]})")
            visualizations.append((name, kernel_img))

        return kernel_paths, visualizations

    def process_and_bind_images(self) -> None:
        """
        Основной метод: генерация датасета.

        Создаёт искажённые изображения со всеми комбинациями
        размытия и шума, сохраняет результаты и связывает через Processing.
        """
        # Создание директорий
        for directory in [self.output_dir, self.kernel_dir, self.kernel_data_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"Создана папка: '{directory}'")

        # Генерация ядер
        all_kernels, visualizations = self.generate_and_save_kernels()

        # Поиск изображений
        image_paths = (
            glob.glob(os.path.join(self.input_dir, '*.[pP][nN][gG]')) +
            glob.glob(os.path.join(self.input_dir, '*.[jJ][pP][gG]'))
        )

        if not image_paths:
            print("\nНе найдено изображений в папке 'original'. Завершение.")
            self._visualize_kernels(visualizations)
            return

        print(f"\nНайдено изображений: {len(image_paths)}\n")

        # Обработка каждого изображения
        for image_path in image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print(f"--- Обработка: {os.path.basename(image_path)} ---")

            image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                print(f"  ! Ошибка чтения: {image_path}")
                continue

            # Применение размытия
            for blur_name, paths in all_kernels.items():
                kernel = np.load(paths['npy'])
                blurred = cv2.filter2D(image_gray, -1, kernel)

                # Сохранение размытого изображения
                output_filename = f"{base_name}_{blur_name}.png"
                output_path = os.path.join(self.output_dir, output_filename)
                cv2.imwrite(output_path, blurred)
                print(f"  -> {output_filename}")

                # Связывание
                self.proc.bind(
                    original_image_path=image_path,
                    blurred_image_path=output_path,
                    original_kernel_path=paths['png'],
                    filter_description=blur_name,
                    color=self.proc.color
                )

                # Добавление шума
                for noise_config in self.noise_configs:
                    noise_name = noise_config['name']
                    noise_filter = noise_config['class'](**noise_config['params'])
                    noisy = noise_filter.filter(blurred)

                    # Сохранение зашумлённого изображения
                    output_filename_noisy = f"{base_name}_{blur_name}_{noise_name}.png"
                    output_path_noisy = os.path.join(self.output_dir, output_filename_noisy)
                    cv2.imwrite(output_path_noisy, noisy)
                    print(f"  -> {output_filename_noisy}")

                    # Связывание
                    self.proc.bind(
                        original_image_path=image_path,
                        blurred_image_path=output_path_noisy,
                        original_kernel_path=paths['png'],
                        filter_description=f"{blur_name}_{noise_name}",
                        color=self.proc.color
                    )

        self._visualize_kernels(visualizations)


def main() -> None:
    """Точка входа для запуска генерации датасета."""
    print("Запуск генерации датасета...\n")

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
        print(f"\nДатасет сохранён в '{output_json_path}'")
    else:
        print("\nНе создано ни одного изображения. JSON не сохранён.")


if __name__ == "__main__":
    main()

