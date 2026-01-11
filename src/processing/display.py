"""
Модуль вывода связей из фреймворка на экран.

Выводит оригинальное изображение, смазанное, метрики смазанного,
ядро смаза, предобработанное, восстановленное, ядро восстановленного,
метрики восстановленного, зашифрованный фильтр смаза.

Авторы: Юров П.И., Беззаборов А.А., Куропатов К.Л.
"""

import numpy as np
import os
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import math
from typing import Tuple, Dict, Any
from processing.utils import (
    Image,
    imread
)

class ModuleDisplay:
    """
    Модуль вывода связей из фреймворка на экран.

    Выводит оригинальное изображение, смазанное, метрики смазанного,
    ядро смаза, предобработанное, восстановленное, ядро восстановленного,
    метрики восстановленного, зашифрованный фильтр смаза.

    Возможности:
        - Вывод ядер под изображением.
        - Вывод ядер сбоку.
    """
    def __init__(self, processing_instance: Any) -> None:
        """
        Инициализация.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing с изображениями.
        """
        self.processing = processing_instance
    
    def show(self, 
             size: float = 1.0, 
             kernel_intencity_scale: float = 1.0, 
             kernel_size: float = 1.0) -> None:
        """Вывод всех изображений: оригинал, размытые, восстановленные + метрики."""
        if not self.processing.images.size:
            print("Нет изображений для отображения")
            return

        h = 0
        alg_arr = []
        for img_obj in self.processing.images:
            img_obj.save_filter()
            h += max(1,img_obj.get_len_filter())
            alg_arr.extend(img_obj.get_algorithm())
        
        alg_arr = list(set(alg_arr))
        w = len(alg_arr) + 2 + 1
        
        fig, axes = plt.subplots(2 * h, w, figsize=(5 * w * size, 8 * h * size))
        line = 0
        
        for img_obj in self.processing.images:
            line = self._plot_single_image(img_obj, alg_arr, axes, 
                                           line, kernel_intencity_scale, kernel_size)
            img_obj.load(img_obj.get_len_filter() - 1)

        plt.suptitle(" ", y=1.02, fontsize=14 * size)
        plt.tight_layout()
        plt.show()

    def _plot_single_image(self, 
                           img_obj: Image, 
                           alg_arr: np.ndarray, 
                           axes: Any, 
                           line: int, 
                           kernel_intencity_scale: float,
                           kernel_size: Any) -> int:
        """Отрисовка одного изображения со всеми его размытыми вариантами."""
        original_image = img_obj.get_original_image()
        restored_psnr = img_obj.get_PSNR()
        restored_ssim = img_obj.get_SSIM()
        restored_paths = img_obj.get_restored()
        blurred_psnr = img_obj.get_blurred_PSNR()
        blurred_ssim = img_obj.get_blurred_SSIM()
        original_kernels = img_obj.get_original_kernels()
        kernels = img_obj.get_kernels()
        filter_name = img_obj.get_filters()

        for blurred_path in img_obj.get_blurred_array():
            preprocess_path = img_obj.get_preprocessed_blurred_path().get(blurred_path, blurred_path)
            line = self._plot_images_line(img_obj, blurred_path, preprocess_path, original_image, alg_arr, 
                                        restored_psnr, restored_ssim, restored_paths,
                                        blurred_psnr, blurred_ssim, axes, line, filter_name)
            
            line = self._plot_kernels_line(img_obj, blurred_path, alg_arr, original_kernels,
                                        kernels, axes, line, kernel_intencity_scale,kernel_size)
        
        return line

    def _plot_images_line(self, 
                          img_obj: Image, 
                          blurred_path: str, 
                          preprocess_path: str,
                          original_image: np.ndarray, 
                          alg_arr: np.ndarray, 
                          restored_psnr: Dict[Tuple[str, str], str], 
                          restored_ssim: Dict[Tuple[str, str], str], 
                          restored_paths: Dict[Tuple[str, str], str], 
                          blurred_psnr: Dict[str, float], 
                          blurred_ssim: Dict[str, float], 
                          axes: Any, 
                          line: int, 
                          filters_name: Dict[str, str]) -> int:
        """Отрисовка строки с изображениями."""
        blurred_image = imread(str(blurred_path), img_obj.get_color())
        preprocess_image = imread(str(preprocess_path), img_obj.get_color())
        
        plt.subplots_adjust(hspace=0.5)

        axes[line, 0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
        axes[line, 0].set_title("Original", fontsize=12)
        axes[line, 0].axis('off')
        
        psnr_val = blurred_psnr.get(str(blurred_path), math.nan)
        ssim_val = blurred_ssim.get(str(blurred_path), math.nan)
        filter_name = filters_name.get(str(blurred_path), 'missing')

        axes[line, 1].imshow(cv.cvtColor(blurred_image, cv.COLOR_BGR2RGB))
        axes[line, 1].set_title(f"{filter_name}\n\nDistorted\nPSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}", 
                                fontsize=10)
        axes[line, 1].axis('off')

        axes[line, 2].imshow(cv.cvtColor(preprocess_image, cv.COLOR_BGR2RGB))
        axes[line, 2].set_title(f"Prepreocessed Image", 
                                fontsize=10)
        axes[line, 2].axis('off')
        
        for col, alg_name in enumerate(alg_arr, 3):
            axes[line, col].axis('off')
            self._plot_restored_image(img_obj, blurred_path, alg_name, restored_psnr, 
                                    restored_ssim, restored_paths, axes, line, col)
        
        return line + 1

    def _plot_restored_image(self, 
                             img_obj: Image, 
                             blurred_path: str, 
                             alg_name: str, 
                             restored_psnr: Dict[Tuple[str, str], str], 
                             restored_ssim: Dict[Tuple[str, str], str], 
                             restored_paths: Dict[Tuple[str, str], str], 
                             axes: Any, 
                             line: int, 
                             col: int) -> None:
        """Отрисовка одного восстановленного изображения."""
        try:
            restored_path = restored_paths.get((str(blurred_path), str(alg_name)))
            if restored_path:
                restored_image = imread(restored_path, img_obj.get_color())
                if restored_image is not None:
                    psnr_val = restored_psnr.get((str(blurred_path), str(alg_name)), math.nan)
                    ssim_val = restored_ssim.get((str(blurred_path), str(alg_name)), math.nan)
                    
                    axes[line, col].imshow(cv.cvtColor(restored_image, cv.COLOR_BGR2RGB))
                    axes[line, col].set_title(f"{alg_name}\nPSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}", fontsize=10)
        except Exception as e:
            pass

    def _crop_kernel_image(self, kernel_image: np.ndarray, padding: int = 10) -> np.ndarray:
        """Обрезает изображение ядра до его содержимого с добавлением отступа."""
        if kernel_image is None or kernel_image.size == 0:
            return kernel_image

        coords = cv.findNonZero(kernel_image)
        if coords is None:
            return kernel_image

        x, y, w, h = cv.boundingRect(coords)
        
        img_h, img_w = kernel_image.shape[:2]

        start_x = max(0, x - padding)
        start_y = max(0, y - padding)
        end_x = min(img_w, x + w + padding)
        end_y = min(img_h, y + h + padding)

        cropped_kernel = kernel_image[start_y:end_y, start_x:end_x]

        return cropped_kernel
    
    def _plot_kernels_line(self, 
                           img_obj: Image, 
                           blurred_path: str, 
                           alg_arr: np.ndarray, 
                           original_kernels: Dict[str, str], 
                           kernels: Dict[Tuple[str, str], str], 
                           axes: Any, 
                           line: int, 
                           kernel_intencity_scale: float, 
                           kernel_size) -> int:
        """Отрисовка строки с ядрами с сохранением оригинальных пропорций."""
        axes[line, 0].axis('off')
        
        original_kernel_path = original_kernels.get(str(blurred_path))
        if original_kernel_path:
            original_kernel = cv.imread(str(original_kernel_path), cv.IMREAD_GRAYSCALE)
            
            if original_kernel is not None:
                cropped_kernel = self._crop_kernel_image(original_kernel)

                if cropped_kernel is not None and cropped_kernel.size > 0:
                    axes[line, 1].imshow(cropped_kernel, cmap='gray')
                    axes[line, 1].set_title("original kernel", fontsize=10)
                    axes[line, 1].set_aspect('equal', adjustable='box')
                    axes[line, 1].axis('off')

        axes[line, 2].axis('off')
        
        for col, alg_name in enumerate(alg_arr, 3):
            axes[line, col].axis('off')
            self._plot_restored_kernel(img_obj, blurred_path, alg_name, kernels, 
                                    axes, line, col, kernel_intencity_scale)
        
        return line + 1

    def _plot_restored_kernel(self, 
                              img_obj: Image, 
                              blurred_path: str, 
                              alg_name: str, 
                              kernels: Dict[Tuple[str, str], str],
                              axes: Any, 
                              line: int, 
                              col: int, 
                              kernel_intencity_scale: float) -> None:
        """Отрисовка одного восстановленного ядра с сохранением пропорций."""
        try:
            kernel_path = kernels.get((str(blurred_path), str(alg_name)))
            if kernel_path:
                kernel = cv.imread(str(kernel_path), cv.IMREAD_GRAYSCALE)
                if kernel is not None:
                    cropped_kernel = self._crop_kernel_image(kernel)
                    if cropped_kernel is not None and cropped_kernel.size > 0:
                        axes[line, col].imshow(cropped_kernel, cmap='gray')
                        axes[line, col].set_title(f"{alg_name} kernel", fontsize=10)
                        axes[line, col].set_aspect('equal', adjustable='box')
        except Exception as e:
            pass
    
    def show_line(self, window_scale: float = 1.0, fontsize: int = 8) -> None:
        """
        Вывод изображений в строчку.
        
        Параметры
        ---------
        window_scale : float
            Регулирует размер окна.
        fontsize : int
            Размер шрифта.
        """
        for img in self.processing.images:
            self._show_line_single_image(img,window_scale,fontsize)

    def _show_line_single_image(self, 
                                img, 
                                window_scale: float=1.0, 
                                fontsize: int = 8) -> None:
        """Рисует одну строчку из изображений."""
        alg_array = img.get_algorithm()
        w = len(alg_array)
        w = 2*w + 3
        fig, axes = plt.subplots(1, w, figsize=(5 * max(window_scale,0.1), 5 * w * max(window_scale,0.1)))

        blurred_path = img.get_blurred()
        psnr_array = img.get_PSNR()
        ssim_array = img.get_SSIM()
        kernel_restored_array = img.get_kernels()
        restored_array = img.get_restored()

        axes[0].imshow(self._read_image_from_file(img.get_original()), cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(self._read_image_from_file(blurred_path), cmap='gray')
        axes[1].set_title(
            f"PSNR: {img.get_blurred_PSNR().get(str(blurred_path),math.nan):.4f}\nSSIM: {img.get_blurred_SSIM().get(str(blurred_path),math.nan):.4f}",
            fontsize=fontsize)
        axes[1].axis('off')

        axes[2].imshow(self._read_image_from_file(img.get_original_kernels().get(str(blurred_path), None)), cmap='gray')
        axes[2].axis('off')

        for idx, alg_one in enumerate(alg_array):
            axes[3+idx].imshow(self._read_image_from_file(
                restored_array.get((str(blurred_path),str(alg_one)),None)),
                  cmap='gray')
            axes[3+idx].set_title(
            f"PSNR: {psnr_array.get((str(blurred_path),str(alg_one)),math.nan):.4f}\nSSIM: {ssim_array.get((str(blurred_path),str(alg_one)),math.nan):.4f}",
            fontsize=fontsize)
            axes[3+idx].axis('off')

            axes[4+idx].imshow(self._read_image_from_file(
                kernel_restored_array.get((str(blurred_path),str(alg_one)),None)),
                cmap='gray')
            axes[4+idx].axis('off')
        plt.show()
    
    def _read_image_from_file(self, path: Path) -> np.ndarray:
        """Читает изображение из файла."""
        if os.path.exists(path):
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        else:
            print(f"  - ПРЕДУПРЕЖДЕНИЕ: Файл не найден, будет показана заглушка: {path}")
            image = np.zeros((100, 100), dtype=np.uint8) # Черный квадрат-заглушка
        return image

