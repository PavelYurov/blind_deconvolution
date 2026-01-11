"""
Модуль применения к изображению фильтров.

Возможности:
    - Применить заданный фильтр.
    - Создать или обновить ядро смаза.
Авторы: Юров П.И., Беззаборов А.А.
"""
import numpy as np
import cv2 as cv
import math
from pathlib import Path
from typing import Any

from processing.utils import (
    Image,
    imread,
    generate_unique_file_path,
    calculate_metrics
)
import filters.base as filters

from filters.blur import Identical_kernel

class ModuleFilter:
    """
    Модуль применения к изображению фильтров.

    Возможности:
        - Применить заданный фильтр.
        - Создать или обновить ядро смаза.
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

    def filter(self, filter_processor: filters.FilterBase) -> None:
        """Применение фильтра ко всем изображениям."""
        for img_obj in self.processing.images:
            self._apply_single_filter(img_obj, filter_processor)
        
    def _apply_single_filter(self, 
                             img_obj: Image, 
                             filter_processor: filters.FilterBase) -> None:
        """Применение фильтра к одному изображению."""
        blurred_path = img_obj.get_blurred()
        preprocess_path = img_obj.get_preprocessed_blurred_path().get(blurred_path, None)
        denoising_filter = (filter_processor.get_type() == 'denoise')
        if blurred_path is None:
            current_image = img_obj.get_original_image()
        elif denoising_filter and preprocess_path is not None:
            current_image = imread(preprocess_path, img_obj.get_color())
        else:
            current_image = img_obj.get_blurred_image()
        
        if current_image is None:
            raise Exception("Не удалось загрузить изображение")
        
        filtered_image = filter_processor.filter(current_image)
        
        if not denoising_filter:
            if blurred_path is None:
                original_filename = Path(img_obj.get_original()).name
                new_path = generate_unique_file_path(self.processing.folder_path_blurred, 
                                                           original_filename)
            else:
                original_filename = Path(blurred_path).name
                new_path = self.processing.folder_path_blurred / original_filename

            psnr_val, ssim_val = calculate_metrics(img_obj.get_original_image(), 
                                                         filtered_image)
            
            img_obj.add_blurred_PSNR(psnr_val, str(new_path))
            img_obj.add_blurred_SSIM(ssim_val, str(new_path))
            cv.imwrite(str(new_path), filtered_image)
            img_obj.set_blurred(str(new_path))
            img_obj.add_to_current_filter(filter_processor.description())
            
            self._process_kernel(img_obj, 
                                 filter_processor, 
                                 new_path, 
                                 original_filename)
            return
        else:
            if blurred_path is None:
                self._copy_original_to_blurred(img_obj)
                

            if preprocess_path is None:
                original_filename = Path(img_obj.get_blurred()).name
                new_path = generate_unique_file_path(self.processing.preprocess_dir, original_filename)
            else:
                original_filename = Path(img_obj.get_blurred()).name
                new_path = self.processing.preprocess_dir / original_filename
            
            cv.imwrite(str(new_path), filtered_image)
            img_obj.add_preprocessed_blurred_path(str(img_obj.get_blurred()), str(new_path))
            img_obj.add_to_current_filter(filter_processor.discription()) 

    def _copy_original_to_blurred(self, img_obj: Image) -> None:
        """Копирует оригинальное изображение в смазанное, ядро - единичное."""
        original_filename = Path(img_obj.get_original()).name
        new_path = generate_unique_file_path(self.processing.folder_path_blurred, original_filename)

        psnr_val, ssim_val = math.nan, math.nan
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        cv.imwrite(str(new_path), img_obj.get_original_image())
        img_obj.set_blurred(str(new_path))
        self._process_kernel(img_obj, Identical_kernel(), new_path, original_filename)

    def _process_kernel(self, 
                        img_obj: Image, 
                        filter_processor: filters.FilterBase, 
                        new_path: Path, 
                        original_filename: str) -> None:
        """Обработка ядра для фильтра."""
        kernels = img_obj.get_original_kernels()
        kernel_path = kernels.get(str(new_path))
        
        if kernel_path is None:
            kernel_image = img_obj.get_original_image().copy()
            kernel_image *= 0
            h, w = kernel_image.shape[:2]
            kernel_image[h//2, w//2] = 255
            
            new_kernel_path = generate_unique_file_path(self.processing.folder_path_blurred, 
                                                        f"kernel_{original_filename}")
        else:
            kernel_image = imread(str(kernel_path), img_obj.get_color())
            new_kernel_path = Path(kernel_path)
        
        if filter_processor.get_type() != 'noise':
            filtered_kernel = filter_processor.filter(kernel_image)
        else:
            filtered_kernel = kernel_image
        cv.imwrite(str(new_kernel_path), filtered_kernel)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path))
    
    def custom_filter(self, 
                      kernel_image_path: Path, 
                      kernel_npy_path: Path) -> None:
        """Применение созданного фильтра ко всем оригинальным изображениям."""
        for img_obj in self.processing.images:
            self._apply_single_custom_filter(img_obj, kernel_image_path, kernel_npy_path)

    def _apply_single_custom_filter(self, 
                                    img_obj: Image, 
                                    kernel_image_path: Path, 
                                    kernel_npy_path: Path) -> None:
        """Применение созданного фильтра к одному изображению."""
        current_image = img_obj.get_original_image()
        kernel = np.load(kernel_npy_path)
        if current_image is None:
            raise Exception("Не удалось загрузить изображение")
        filtered_image = cv.filter2D(current_image, -1, kernel)
        original_filename = Path(img_obj.get_original()).stem
        blur_filename = Path(kernel_image_path).stem
        filtered_filename =  self.processing.folder_path_blurred / f"{original_filename}_{blur_filename}.png"

        psnr_val, ssim_val = calculate_metrics(current_image, filtered_image)

        img_obj.add_blurred_PSNR(psnr_val, str(filtered_filename))
        img_obj.add_blurred_SSIM(ssim_val, str(filtered_filename))
        cv.imwrite(str(filtered_filename), filtered_image)
        img_obj.set_blurred(str(filtered_filename))
        img_obj.add_to_current_filter(blur_filename)
        img_obj.add_original_kernel(str(kernel_image_path), str(filtered_filename))



   