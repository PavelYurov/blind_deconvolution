"""
Модуль конвейера обработки изображения.

Возможности:
    - Применение нескольких фильтров.
    - Применение методов восстановления.
    - Анализ результатов.

Авторы: Юров П.И., Беззаборов А.А.
"""
import numpy as np
import cv2 as cv
import math
import pandas as pd
from pathlib import Path
from typing import Any


from processing.utils import (
    Image,
    imread,
    prepare_image_for_metric,
    generate_unique_file_path,
    calculate_metrics
)
import processing.metrics as metrics
import algorithms.base as base


from IPython.display import display

from processing.restore import ModuleProcess

class ModuleProcessPipeline(ModuleProcess):
    """
    Модуль конвейера обработки изображения.

    Возможности:
        - Применение нескольких фильтров.
        - Применение методов восстановления.
        - Анализ результатов.
    """

    def __init__(self, processing_instance: Any) -> None:
        """
        Инициализация.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing с изображениями.
        """
        super().__init__(processing_instance)

    def full_process(self, 
                     filters: list, 
                     methods: list, 
                     size: float = 0.75, 
                     kernel_intencity_scale: float = 10.0) -> None:
        """
        Пайплайн применения фильтров с последующим восстановлением.
        
        Параметры
        ---------
        filters : list
            Массив массивов объектов FilterBase [[],[]].
        methods : list
            Массив объектов DeconvolutionAlgorithm.
        size : float
            Размер таблицы.
        kernel_intencity_scale : float
            Цвет пикселей PSF при выводе.
        """
        data_dict, analysis_dicts = self._initialize_data_structures()
        for img_obj in self.processing.images:
            self._process_image_pipeline(img_obj, 
                                         filters, 
                                         methods, 
                                         data_dict, 
                                         analysis_dicts)
        
        self._finalize_processing(data_dict, 
                                  analysis_dicts, 
                                  size, 
                                  kernel_intencity_scale)
        
    def _initialize_data_structures(self):
        """Инициализация структур данных для сбора результатов."""
        data_dict = {}
        analysis_dicts = {
            'images_dict': {},
            'ssim_dict': {},
            'psnr_dict': {},
            'kernels_dict': {}
        }
        return data_dict, analysis_dicts    
    
    def _process_image_pipeline(self, 
                                img_obj: Image, 
                                filters: list, 
                                methods: list, 
                                data_dict : dict, 
                                analysis_dicts: dict) -> None:
        """Полный пайплайн обработки для одного изображения."""
        self._apply_filter_chains(img_obj, filters)
        
        self._apply_restoration_and_collect_data(img_obj, 
                                                 methods, 
                                                 data_dict, 
                                                 analysis_dicts)   
    
    def _apply_filter_chains(self, 
                             img_obj: Image, 
                             filters: list) -> None:
        """Применение цепочек фильтров к изображению."""
        for filter_chain in filters:
            self._apply_single_filter_chain(img_obj, 
                                            filter_chain)
            img_obj.save_filter()

    def _apply_single_filter_chain(self, 
                                   img_obj: Image, 
                                   filter_chain: list) -> None:
        """Применение одной цепочки фильтров."""
        original_image = img_obj.get_original_image()
        current_image = img_obj.get_blurred_image()
        
        if current_image is None:
            current_image = original_image
        
        if current_image is None:
            raise Exception("Failed to load image")
        
        filtered_image = self._apply_filters_sequential(current_image, 
                                                        filter_chain, 
                                                        img_obj)
        
        new_path = self._generate_blurred_image_path(img_obj)
        cv.imwrite(new_path, filtered_image)
        
        self._update_blurred_image_data(img_obj, 
                                        original_image, 
                                        filtered_image, 
                                        new_path)
        
        self._process_kernel_chain(img_obj, filter_chain, new_path)
    
    def _apply_filters_sequential(self, 
                                  image: np.ndarray, 
                                  filter_chain: list, 
                                  img_obj: Image) -> np.ndarray:
        """Последовательное применение фильтров цепочки."""
        filtered_image = image.copy()
        for filter_processor in filter_chain:
            filtered_image = filter_processor.filter(filtered_image)
            img_obj.add_to_current_filter(filter_processor.description())
        return filtered_image

    def _generate_blurred_image_path(self, img_obj: Image) -> Path:
        """Генерация пути для размытого изображения."""
        if img_obj.get_blurred() is None:
            original_filename = Path(img_obj.get_original()).name
            return generate_unique_file_path(self.processing.folder_path_blurred, original_filename)
        else:
            original_filename = Path(img_obj.get_blurred()).name
            return self.processing.folder_path_blurred / original_filename
    
    def _update_blurred_image_data(self, 
                                   img_obj: Image, 
                                   original_image: np.ndarray, 
                                   filtered_image: np.ndarray, 
                                   new_path: str) -> None:
        """Обновление данных размытого изображения."""
        psnr_val, ssim_val = calculate_metrics(original_image, filtered_image)
        
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        img_obj.set_blurred(str(new_path))
    
    def _process_kernel_chain(self, 
                              img_obj: Image, 
                              filter_chain: list, 
                              new_path: str) -> None:
        """Обработка ядра для цепочки фильтров."""
        kernels = img_obj.get_original_kernels()
        kernel_path = kernels.get(str(new_path))
        
        if kernel_path is None:
            kernel_image = self._create_delta_kernel(img_obj)
            new_kernel_path = self.processing.folder_path_blurred / f"kernel_{Path(new_path).name}"
        else:
            kernel_image = imread(str(kernel_path), img_obj.get_color())
            new_kernel_path = Path(kernel_path)
        
        for filter_processor in filter_chain:
            if filter_processor.get_type() != 'noise':
                kernel_image = filter_processor.filter(kernel_image)
            else:
                kernel_image = kernel_image
        
        cv.imwrite(str(new_kernel_path), kernel_image)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path)) 
        
    def _create_delta_kernel(self, img_obj: Image) -> np.ndarray:
        """Создание дельта-функции (единичного импульса)."""
        kernel_image = img_obj.get_original_image().copy()
        kernel_image *= 0
        h, w = kernel_image.shape[:2]
        kernel_image[h//2, w//2] = 255
        return kernel_image   
    
    def _apply_restoration_and_collect_data(self, 
                                            img_obj: Image, 
                                            methods: list, 
                                            data_dict: dict, 
                                            analysis_dicts: dict) -> None:
        """Восстановление изображений и сбор данных."""
        for blurred_path in img_obj.get_blurred_array():
            blurred_image = imread(str(blurred_path), img_obj.get_color())
            
            for algorithm in methods:
                self._restore_single_image(img_obj, blurred_path, blurred_image, algorithm, data_dict)
            
            self._collect_analysis_data(img_obj, blurred_path, analysis_dicts)
            
    def _restore_single_image(self, 
                              img_obj: Image, 
                              blurred_path: Path, 
                              blurred_image: np.ndarray, 
                              algorithm: base.DeconvolutionAlgorithm, 
                              data_dict: dict) -> None:
        """Восстановление одного изображения алгоритмом."""
        alg_name = algorithm.get_name()
        original_image = img_obj.get_original_image()
        
        try:
            restored_image, restored_kernel = algorithm.process(blurred_image)
            
            restored_path, kernel_path = self._generate_restoration_paths(blurred_path, alg_name)
            
            cv.imwrite(restored_path, restored_image)
            cv.imwrite(kernel_path, restored_kernel)

            restored_image = prepare_image_for_metric(restored_image)
            original_image = prepare_image_for_metric(original_image)
            psnr_val, ssim_val = calculate_metrics(original_image, restored_image, data_range=1.0)
            
            self._update_restoration_data(img_obj, blurred_path, alg_name, 
                                        restored_path, kernel_path, psnr_val, ssim_val)
            
            self._collect_algorithm_data(data_dict, alg_name, algorithm, img_obj,
                                    blurred_path, restored_path, kernel_path,
                                    psnr_val, ssim_val, restored_image, blurred_image)
            
        except Exception as e:
            print(f"Restore error {blurred_path} of {alg_name}: {e}")
    
    def _generate_restoration_paths(self, 
                                    blurred_path: Path, 
                                    alg_name: str) -> tuple[str, str]:
        """Генерация путей для восстановленных изображений и ядер."""
        blurred_basename = Path(blurred_path).stem
        ext = Path(blurred_path).suffix
        
        restored_path = generate_unique_file_path(
            self.processing.folder_path_restored, 
            f"{blurred_basename}_{alg_name}{ext}"
        )
        
        kernel_path = generate_unique_file_path(
            self.processing.folder_path_restored,
            f"{blurred_basename}_{alg_name}_kernel{ext}"
        )
        
        return str(restored_path), str(kernel_path)
  
    def _update_restoration_data(self, 
                                 img_obj: Image, 
                                 blurred_path: Path, 
                                 alg_name: str, 
                                 restored_path: Path, 
                                 kernel_path: Path, 
                                 psnr_val: float, 
                                 ssim_val: float) -> None:
        """Обновление данных восстановления."""
        img_obj.add_PSNR(psnr_val, blurred_path, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_path, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(restored_path, blurred_path, alg_name)
        img_obj.add_kernel(kernel_path, blurred_path, alg_name)

    def _collect_algorithm_data(self, 
                                data_dict: dict, 
                                alg_name: str, 
                                algorithm: base.DeconvolutionAlgorithm, 
                                img_obj: Image, 
                                blurred_path: Path, 
                                restored_path: Path, 
                                kernel_path: Path, 
                                psnr_val: float, 
                                ssim_val: float, 
                                restored_image: np.ndarray, 
                                blurred_image: np.ndarray) -> None:
        """Сбор данных для анализа алгоритмов."""
        alg_data = data_dict.setdefault(alg_name, {})
        
        metrics_data = [
            ('image', img_obj.get_original()),
            ('filter', img_obj.get_filters().get(blurred_path, 'unknown')),
            ('time', algorithm.get_timer()),
            ('blurred_sharpness', metrics.Sharpness(blurred_image)),
            ('restored_sharpness', metrics.Sharpness(restored_image)),
            ('blurred_psnr', img_obj.get_blurred_PSNR().get(blurred_path, math.nan)),
            ('blurred_ssim', img_obj.get_blurred_SSIM().get(blurred_path, math.nan)),
            ('restored_psnr', psnr_val),
            ('restored_ssim', ssim_val),
            ('restored_path', restored_path),
            ('kernel_path', kernel_path)
        ]
        
        for key, value in metrics_data:
            alg_data[key] = np.append(alg_data.setdefault(key, []), value)
        
        for param_name, param_value in algorithm.get_param():
            alg_data[param_name] = np.append(alg_data.setdefault(param_name, []), param_value)
 
    def _collect_analysis_data(self, 
                               img_obj: Image, 
                               blurred_path: Path, 
                               analysis_dicts: dict) -> None:
        """Сбор данных для общего анализа."""
        images_dict, ssim_dict, psnr_dict, kernels_dict = analysis_dicts.values()
        
        original = img_obj.get_original()
        blurred_filter = img_obj.get_filters()
        restored_array = img_obj.get_restored()
        blurred_psnr = img_obj.get_blurred_PSNR()
        blurred_ssim = img_obj.get_blurred_SSIM()
        restored_psnr = img_obj.get_PSNR()
        restored_ssim = img_obj.get_SSIM()

        images_dict.setdefault('original', []).append(original)
        psnr_dict.setdefault('original', []).append(original)
        ssim_dict.setdefault('original', []).append(original)

        images_dict.setdefault('filter', []).append(blurred_filter.get(str(blurred_path), ''))
        psnr_dict.setdefault('filter', []).append(blurred_filter.get(str(blurred_path), ''))
        ssim_dict.setdefault('filter', []).append(blurred_filter.get(str(blurred_path), ''))

        images_dict.setdefault('blurred', []).append(blurred_path)
        psnr_dict.setdefault('blurred', []).append(blurred_psnr.get(str(blurred_path), math.nan))
        ssim_dict.setdefault('blurred', []).append(blurred_ssim.get(str(blurred_path), math.nan))

        for alg_name in img_obj.get_algorithm():
            images_dict.setdefault(alg_name, []).append(
                restored_array.get((str(blurred_path), str(alg_name)), 'None')
            )
            psnr_dict.setdefault(alg_name, []).append(
                restored_psnr.get((str(blurred_path), str(alg_name)), math.nan)
            )
            ssim_dict.setdefault(alg_name, []).append(
                restored_ssim.get((str(blurred_path), str(alg_name)), math.nan)
            )
    
    def _finalize_processing(self, 
                             data_dict: dict, 
                             analysis_dicts: dict, 
                             size: float, 
                             kernel_intencity_scale: float) -> None:
        """Финальная обработка и визуализация результатов."""
        self.processing.show(size, kernel_intencity_scale)
        
        for alg_name, alg_data in data_dict.items():
            df_data = pd.DataFrame(alg_data)
            display(df_data)
            df_data.to_csv(self.processing.data_path / f"{alg_name}.csv", index=False)
        
        self.processing.pareto()


    