"""
Модуль предобработки смазанного изображения.

Возможности:
    - Применить выравнивание гистограмм.
    - Применить адаптивное выравнивание гистограмм.
    - Обратить выравнивание гистограмм.

Автор: Юров П.И.
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from skimage.exposure import (
    equalize_hist, 
    equalize_adapthist, 
    histogram, 
    match_histograms, 
    cumulative_distribution
)

from processing.utils import (
    imread,
    float_img_to_int,
    prepare_image_for_metric,
)


class ModulePreprocessing:
    """
    Модуль предобработки смазанного изображения.

    Возможности:
        - Применить выравнивание гистограмм.
        - Применить адаптивное выравнивание гистограмм.
        - Обратить выравнивание гистограмм.
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
    
    def histogram_equalization(self, view_histogram: bool = False) -> None:
        """Выполняет выравнивание гистограмм."""
        for img_obj in self.processing.images:
            blurred_path = img_obj.get_blurred()
            if blurred_path is None:
                self._copy_original_to_blurred(img_obj)
                blurred_path = img_obj.get_blurred()

            current_image = img_obj.get_blurred_image()
            
            filtered_image = equalize_hist(current_image, nbins=256)
            filtered_image = float_img_to_int(filtered_image)
            
            original_filename = Path(img_obj.get_blurred()).name
            new_path_preprocess = self.processing.preprocess_dir / f'{str(original_filename)}'

            cv.imwrite(str(new_path_preprocess), filtered_image)

            img_obj.add_preprocessed_blurred_path(blurred_path, new_path_preprocess)

            if (view_histogram):
                hist1 = histogram(current_image)
                hist2 = histogram(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.bar(hist1[1], hist1[0], alpha=0.5, color='blue')
                plt.bar(hist2[1], hist2[0], alpha=0.5, color='red')
                plt.grid(alpha=0.3)
                plt.show()
                
                cdf1 = cumulative_distribution(current_image)
                cdf2 = cumulative_distribution(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.plot(cdf1[0], cdf1[1], color='blue')
                plt.plot(cdf2[0], cdf2[1], color='red')
                plt.show()

    def histogram_equalization_CLAHE(self, 
                                     view_histogram: bool = False, 
                                     clip_limit: float = 0.01) -> None:
        """Выполняет адаптивное выравнивание гистограмм с ограничением контрастности."""
        for img_obj in self.processing.images:
            blurred_path = img_obj.get_blurred()
            if blurred_path is None:
                self._copy_original_to_blurred(img_obj)
                blurred_path = img_obj.get_blurred()

            current_image = img_obj.get_blurred_image()
            filtered_image = equalize_adapthist(current_image, 
                                                nbins=256, 
                                                clip_limit=clip_limit)
            filtered_image = float_img_to_int(filtered_image)
            
            original_filename = Path(img_obj.get_blurred()).name
            new_path_preprocess = self.processing.preprocess_dir / f'{str(original_filename)}'

            cv.imwrite(str(new_path_preprocess), filtered_image)

            img_obj.add_preprocessed_blurred_path(blurred_path, new_path_preprocess)

            if (view_histogram):
                hist1 = histogram(current_image)
                hist2 = histogram(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.bar(hist1[1], hist1[0], alpha=0.5, color='blue')
                plt.bar(hist2[1], hist2[0], alpha=0.5, color='red')
                plt.grid(alpha=0.3)
                plt.show()
                
                cdf1 = cumulative_distribution(current_image)
                cdf2 = cumulative_distribution(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.plot(cdf1[0], cdf1[1], color='blue')
                plt.plot(cdf2[0], cdf2[1], color='red')
                plt.show()
    
    def inverse_histogram_equalization(self, view_histogram: bool = False) -> None:
        """Обращает выравнивание гистограмм."""
        for img_obj in self.processing.images:

            blurred_path = img_obj.get_blurred()
            
            preprocessed_image_paths = img_obj.get_preprocessed_blurred_path()
            preprocessed_image_path = preprocessed_image_paths.get(blurred_path, None)
            if preprocessed_image_path is None:
                raise Exception('Image didn\'t preprocessed or Image not found')
            current_image = imread(preprocessed_image_path, img_obj.get_color())

            restored_array = img_obj.get_restored()
            original_blurred_image = img_obj.get_blurred_image()
            if original_blurred_image is None:
                raise Exception("Оригинальное смазанное изображение не найдено")
            
            filtered_image = self._inverse_histogram_equalization_one(current_image, 
                                                                original_blurred_image, 
                                                                view_histogram)
            
            cv.imwrite(str(preprocessed_image_path), filtered_image)

            original_image = img_obj.get_original_image()
            original_image = prepare_image_for_metric(original_image)
            for alg_name in img_obj.get_algorithm():
                current_image_path = restored_array[(blurred_path, alg_name)]
                current_image = imread(current_image_path, img_obj.get_color())
                
                filtered_image = self._inverse_histogram_equalization_one(current_image, 
                                                                         original_blurred_image, 
                                                                         view_histogram)
                
                cv.imwrite(str(current_image_path), filtered_image)
                
                filtered_image = prepare_image_for_metric(filtered_image)
                
                psnr_val, ssim_val = self._calculate_metrics(original_image, 
                                                             filtered_image, 
                                                             data_range=1.0)
                
                # blurred_ref = img_obj.get_blurred()
                
                img_obj.add_PSNR(psnr_val, blurred_path, alg_name)
                img_obj.add_SSIM(ssim_val, blurred_path, alg_name)
                img_obj.add_algorithm(alg_name)
                img_obj.add_restored(str(current_image_path), str(blurred_path), alg_name)
                
    def _inverse_histogram_equalization_one(self, 
                                           current_image: np.ndarray, 
                                           original_blurred_image: np.ndarray, 
                                           view_histogram: bool = False) -> np.ndarray:
        """Выполняет обращение выравнивания для одного изображения."""

        filtered_image = match_histograms(image=current_image, 
                                          reference=original_blurred_image)

        if (view_histogram):
            hist1 = histogram(current_image)
            hist2 = histogram(filtered_image)
            plt.figure(figsize=(12, 6))
            plt.bar(hist1[1], hist1[0], alpha=0.5, color='blue')
            plt.bar(hist2[1], hist2[0], alpha=0.5, color='red')
            plt.grid(alpha=0.3)
            plt.show()
            
            cdf1 = cumulative_distribution(current_image)
            cdf2 = cumulative_distribution(filtered_image)
            plt.figure(figsize=(12, 6))
            plt.plot(cdf1[0], cdf1[1], color='blue')
            plt.plot(cdf2[0], cdf2[1], color='red')
        
            plt.show()
        return filtered_image
