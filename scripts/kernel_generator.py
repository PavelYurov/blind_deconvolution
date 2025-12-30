"""
Модуль генерации ядер размытия (PSF) для экспериментов.

Авторы: Куропатов К.Л.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import math
from typing import Dict, List, Tuple, Any, Optional

from filters.blur import DefocusBlur, MotionBlur, BSpline_blur
from filters.distributions import (
    gaussian_distribution,
    uniform_distribution,
    ring_distribution,
    exponential_decay_distribution,
    generate_bspline_motion_kernel,
    generate_multivariate_normal_kernel
)


class KernelGenerator:
    """
    Генератор ядер размытия (PSF) для экспериментов по деконволюции.

    Параметры
    ---------
    kernel_dir : str
        Путь для сохранения PNG-изображений ядер.
    kernel_data_dir : str
        Путь для сохранения NPY-файлов ядер.
    kernel_size : int
        Размер генерируемых ядер.
    """

    def __init__(
        self,
        kernel_dir: str = 'images_dataset/ground_truth_filters',
        kernel_data_dir: str = 'images_dataset/kernel_data',
        kernel_size: int = 51
    ) -> None:
        self.kernel_dir = kernel_dir
        self.kernel_data_dir = kernel_data_dir
        self.kernel_size = kernel_size
        self.max_kernel_size = 51

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
        _bspline_config = {
            'name': 'motionbsplinesimplecurve',
            'generator_func': generate_bspline_motion_kernel,
            'params': {
                'ksize': self.kernel_size,
                'thickness': 3,
                'points': [(-5, 5), (0, -5), (5, 5)]
            }
        }

        # Конфигурации ядер
        self.kernel_configs: List[Dict[str, Any]] = [
            # Расфокус
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
            # Motion blur
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
            # B-spline траектория
            {
                'name': 'motionbsplinesimplecurve',
                'generator_func': generate_bspline_motion_kernel,
                'params': {
                    'ksize': self.kernel_size,
                    'thickness': 3,
                    'points': [(-5, 5), (0, -5), (5, 5)]
                }
            },
            # Многомерное нормальное распределение (эллипс)
            {
                'name': 'mvnellipse45deg',
                'generator_func': generate_multivariate_normal_kernel,
                'params': {
                    'ksize': self.kernel_size,
                    'cov': [[40.0, 25.0], [25.0, 40.0]]
                }
            },
            # Составное ядро
            {
                'name': 'convolved_kernel',
                'generator_func': self._generate_convolved_kernel,
                'params': {
                    'ksize': self.kernel_size,
                    'configs': [
                        _defocus_disk_config,
                        _motion_linear_config,
                        _bspline_config
                    ]
                }
            },
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
        total_size = 0

        for config in configs:
            kernel = None

            if 'generator_func' in config:
                config['params']['ksize'] = max(
                    config['params'].get('ksize', 0), ksize
                )
                kernel = config['generator_func'](**config['params'])

            elif 'class' in config:
                blur_filter = config['class'](**config['params'])
                kernel = blur_filter.generate_kernel()

            if kernel is not None:
                base_kernels.append(kernel)
                total_size += kernel.shape[0]

        if not base_kernels:
            return np.zeros((ksize, ksize))

        # Свёртка всех ядер
        convolved_kernel = np.zeros((total_size, total_size))
        center = total_size // 2
        convolved_kernel[center, center] = 1.0

        for k in base_kernels:
            convolved_kernel = cv2.filter2D(convolved_kernel, -1, k)

        # Обрезка до нужного размера
        start = center - (ksize // 2)
        end = start + ksize
        final_kernel = convolved_kernel[start:end, start:end]

        if final_kernel.sum() > 0:
            return final_kernel / final_kernel.sum()
        return final_kernel

    def generate_all(self) -> Dict[str, np.ndarray]:
        """
        Генерация всех ядер согласно конфигурациям.

        Возвращает
        ----------
        Dict[str, np.ndarray]
            Словарь {имя_ядра: массив_ядра}.
        """
        kernels = {}

        for config in self.kernel_configs:
            name = config['name']
            kernel = None

            if 'generator_func' in config:
                if config['generator_func'] == self._generate_convolved_kernel:
                    kernel = self._generate_convolved_kernel(**config['params'])
                else:
                    kernel = config['generator_func'](**config['params'])

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

            if kernel is not None:
                kernels[name] = kernel
                print(f"  Сгенерировано: {name} ({kernel.shape[0]}x{kernel.shape[1]})")

        return kernels

    def save(self, kernels: Dict[str, np.ndarray]) -> None:
        """
        Сохранение ядер в файлы.

        Параметры
        ---------
        kernels : Dict[str, np.ndarray]
            Словарь ядер для сохранения.
        """
        # Создание директорий
        for directory in [self.kernel_dir, self.kernel_data_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"Создана папка: {directory}")

        # Сохранение
        for name, kernel in kernels.items():
            # NPY
            npy_path = os.path.join(self.kernel_data_dir, f"{name}.npy")
            np.save(npy_path, kernel)

            # PNG
            if np.max(kernel) > 0:
                kernel_img = (kernel / np.max(kernel)) * 255
            else:
                kernel_img = kernel
            kernel_img = kernel_img.astype(np.uint8)
            png_path = os.path.join(self.kernel_dir, f"{name}.png")
            cv2.imwrite(png_path, kernel_img)

            print(f"  Сохранено: {name}")

    def visualize(self, kernels: Dict[str, np.ndarray]) -> None:
        """
        Визуализация ядер.

        Параметры
        ---------
        kernels : Dict[str, np.ndarray]
            Словарь ядер для визуализации.
        """
        num_kernels = len(kernels)
        cols = 4
        rows = math.ceil(num_kernels / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        fig.suptitle('Сгенерированные ядра размытия (PSF)', fontsize=16)
        axes = np.array(axes).flatten()

        for i, (name, kernel) in enumerate(kernels.items()):
            axes[i].imshow(kernel, cmap='gray')
            axes[i].set_title(f"{name}\n({kernel.shape[0]}x{kernel.shape[1]})")
            axes[i].axis('off')

        for i in range(num_kernels, len(axes)):
            axes[i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def run(self, save: bool = True, visualize: bool = True) -> Dict[str, np.ndarray]:
        """
        Запуск генерации ядер.

        Параметры
        ---------
        save : bool
            Сохранять ли ядра в файлы.
        visualize : bool
            Показывать ли визуализацию.

        Возвращает
        ----------
        Dict[str, np.ndarray]
            Сгенерированные ядра.
        """
        print("--- Генерация ядер ---")
        kernels = self.generate_all()

        if save:
            print("\n--- Сохранение ядер ---")
            self.save(kernels)

        if visualize:
            print("\n--- Визуализация ---")
            self.visualize(kernels)

        return kernels


def main() -> None:
    """Точка входа."""
    generator = KernelGenerator()
    generator.run()


if __name__ == "__main__":
    main()

