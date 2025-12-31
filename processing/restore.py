"""
Модуль обработки смазанного изображения.

Авторы: Юров П.И., Беззаборов А.А.
"""
import numpy as np
import os
import cv2 as cv
from pathlib import Path
import json
from typing import Tuple, Any

from processing.utils import (
    Image,
    imread,
    prepare_image_for_metric,
    generate_unique_file_path,
    calculate_metrics
)
import algorithms.base as base

class ModuleProcess:
    """
    Модуль обработки смазанного изображения.

    Применияет заданный метод деконволюции к изображению.
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

    def process(self, 
                algorithm_processor: base.DeconvolutionAlgorithm, 
                metadata: bool = False, 
                unique_path: bool = True) -> None:
        """
        Восстановление всех изображений.
        
        Параметры
        ---------
        algorithm_processor : DeconvolutionAlgorithm
            Метод восстановления изображения.
        metadata : bool
            Сохранять метаданные или нет.
        unique_path : bool
            Генерировать уникальные пути.
        """
        alg_name = algorithm_processor.get_name()
        
        for img_obj in self.processing.images:
            self._process_single_image(img_obj, 
                                       algorithm_processor, 
                                       alg_name, 
                                       metadata = metadata, 
                                       unique_path=unique_path)
        
    def _process_single_image(self, 
                              img_obj: Image, 
                              algorithm_processor: base.DeconvolutionAlgorithm, 
                              alg_name: str, 
                              metadata: bool = False, 
                              unique_path: bool = True) -> None:
        """Восстановление одного изображения."""        
        original_image = img_obj.get_original_image()
        blurred_image_path = img_obj.get_blurred()
        preprocessed_image_path = img_obj.get_preprocessed_blurred_path().get(blurred_image_path, None)
        
        if blurred_image_path is None:
            blurred_image = original_image

        elif preprocessed_image_path is not None:
            blurred_image = imread(preprocessed_image_path, 
                                         img_obj.get_color())
        else:
            blurred_image = img_obj.get_blurred_image()
        
        if blurred_image is None:
            print(f"Pass: Unable to load image for {img_obj.get_original()}")
            return
    
        try:
            restored_image, kernel = algorithm_processor.process(blurred_image)
        except Exception as e:
            print(f"Restore error {img_obj.get_original()}: {e}")
            return
        
        restored_path, kernel_path = self._generate_output_paths(img_obj, 
                                                                 alg_name, 
                                                                 unique_path=unique_path)
        
        try:
            cv.imwrite(str(restored_path), restored_image)
            cv.imwrite(str(kernel_path), (kernel*255))
        except Exception as e:
            print(f"Saving results error {restored_path}: {e}")
            return
        

        original_image = np.atleast_3d(original_image)
        restored_image = np.atleast_3d(restored_image)
        self._calculate_and_save_metrics(img_obj, original_image, restored_image, 
                                   restored_path, kernel_path, alg_name, 
                                   algorithm_processor,metadata=metadata)
    
    def _generate_output_paths(self, 
                               img_obj: Image, 
                               alg_name: str, 
                               unique_path: bool = True) -> Tuple[Path, Path]:
        """Генерация уникальных путей для сохранения результатов."""
        if img_obj.get_blurred():
            base_path = Path(img_obj.get_blurred())
        else:
            base_path = Path(img_obj.get_original())
        
        base_name = base_path.stem
        
        if unique_path:
            restored_path = generate_unique_file_path(
                self.processing.folder_path_restored, 
                f"{base_name}_{alg_name}{base_path.suffix}"
            )
            
            kernel_path = generate_unique_file_path(
                self.processing.folder_path_restored,
                f"{base_name}_{alg_name}_kernel{base_path.suffix}"
            )
        else:
            restored_path =self.processing.folder_path_restored / f"{base_name}_{alg_name}{base_path.suffix}"

            kernel_path = self.processing.kernel_dir / f"{base_name}_{alg_name}_kernel{base_path.suffix}"
        
        return restored_path, kernel_path
    
    def _calculate_and_save_metrics(self, 
                                    img_obj: Image, 
                                    original_image: np.ndarray, 
                                    restored_image: np.ndarray, 
                                    restored_path: Path, 
                                    kernel_path: Path, 
                                    alg_name: str, 
                                    processor: base.DeconvolutionAlgorithm, 
                                    metadata: bool = False) -> None:
        """Расчет метрик и обновление данных изображения."""
        original_image = prepare_image_for_metric(original_image)
        restored_image = prepare_image_for_metric(restored_image)
        
        psnr_val, ssim_val = calculate_metrics(original_image, restored_image, data_range=1.0)
        
        blurred_ref = img_obj.get_blurred()
        
        img_obj.add_PSNR(psnr_val, blurred_ref, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_ref, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(str(restored_path), blurred_ref, alg_name)
        img_obj.add_kernel(str(kernel_path), blurred_ref, alg_name)

        if metadata:
            metadata_path = os.path.splitext(restored_path)[0] + '.json'

            data = dict()
            data['original'] = img_obj.get_original()
            blurred_path = img_obj.get_blurred()
            data['blurred'] = blurred_path
            data['filter'] = img_obj.get_current_filter()
            data['blurred kernel'] = img_obj.get_original_kernels()[blurred_path]
            data['blurred psnr'] = img_obj.get_blurred_PSNR()[blurred_path]
            data['blurred ssim'] = img_obj.get_blurred_SSIM()[blurred_path]
            data['algorithm'] = alg_name
            data['restored'] = str(restored_path)
            data['restored kernel'] = str(kernel_path)
            data['restored psnr'] = psnr_val
            data['restored ssim'] = ssim_val
            data['algorithm parametrs'] = processor.get_param()
            print(data)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Restored: {Path(restored_path).name} (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})")
