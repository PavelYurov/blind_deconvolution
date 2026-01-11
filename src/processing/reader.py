"""
Модуль загрузки изображений из директории в фреймворк.

Возможности:
    - Загрузить все изображения из директории.
    - Загрузить одно изображение по заданному пути.
    - Загрузить связь из оригинального, смазанного изображения и ядра.
    - Сохранить связи из фреймворка в файл.
    - Загрузить связи в фреймворк из файла.

Авторы: Юров П.И., Беззаборов А.А.
"""
import numpy as np
import os
import json
from pathlib import Path
from typing import Optional, Any
from processing.utils import (
    Image,
    imread,
    calculate_metrics
)


class ModuleReader:
    """
    Модуль загрузки изображений из директории в фреймворк.

    Возможности:
        - Загрузить все изображения из директории.
        - Загрузить одно изображение по заданному пути.
        - Загрузить связь из оригинального, смазанного изображения и ядра.
        - Сохранить связи из фреймворка в файл.
        - Загрузить связи в фреймворк из файла.
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
    
    def read_all(self) -> None:
        """Загрузка всех изображений из директории."""
        for image_file in self.processing.folder_path.iterdir():
            if image_file.is_file():
                self._load_image(image_file)
    
    def read_one(self, path: Path) -> None:
        """Загрузка одного изображения."""
        image_path = self.processing.folder_path / path
        self._load_image(image_path)

    def _load_image(self, image_path: str) -> None:
        """Внутренний метод загрузки изображения."""
        image = imread(str(image_path), self.processing.color)
        if image is not None:
            self.processing.images = np.append(self.processing.images, 
                                    Image(
                                        str(image_path), 
                                        self.processing.color))
    
    def bind(self, 
             original_image_path: Path, 
             blurred_image_path: Path, 
             original_kernel_path: Optional[Path] = None, 
             filter_description: str = "unknown", 
             color: bool = True) -> Image:
        """
        Связывает оригинальное изображение с искаженной версией.
        
        Параметры
        ---------
        original_image_path : Path
            Путь к оригинальному изображению.
        blurred_image_path : Path
            Путь к смазанному изображению.
        original_kernel_path : Optional[Path]
            Путь к ядру размытия.
        filter_description : str
            Описание примененного фильтра.
        color : bool
            Способ загрузки (True - цветное, False - ч/б).
            
        Возвращает
        ----------
        Image
            Объект связи изображений.
        """
        if color is None:
            color = self.processing.color
        
        if not all(os.path.exists(p) for p in [original_image_path, blurred_image_path] if p):
            missing = [p for p in [original_image_path, blurred_image_path] if p and not os.path.exists(p)]
            raise FileNotFoundError(f"Files not found: {missing}")
        
        original = imread(original_image_path, color)
        blurred = imread(blurred_image_path, color)
        
        if original is None:
            raise ValueError(f"Failed to load original image: {original_image_path}")
        if blurred is None:
            raise ValueError(f"Failed to load blurred image: {blurred_image_path}")
    
        img_obj = Image(original_image_path, color)
        img_obj.set_blurred(blurred_image_path)
        img_obj.set_current_filter(filter_description)

        psnr_blured, ssim_blured = calculate_metrics(original, blurred)

        img_obj.add_blurred_PSNR(psnr_blured,blurred_image_path)
        img_obj.add_blurred_SSIM(ssim_blured,blurred_image_path)

        
        if original_kernel_path:
            if not os.path.exists(original_kernel_path):
                print(f"Kernel not found: {original_kernel_path}")
            else:
                img_obj.add_original_kernel(original_kernel_path, blurred_image_path)
        
        self.processing.images = np.append(self.processing.images, img_obj)
        return img_obj

    def save_bind_state(self, file_path: Optional[Path] = None) -> None:
        """Сохраняет состояние связей в JSON файл."""
        if file_path is None:
            file_path = os.path.join(self.processing.dataset_path, 'dataset.json')
        data = dict()
        counter = 1
        self.processing.save_filter()

        for img in self.processing.images:
            data[f'{counter}'] = {}

            original_path = img.get_original()
            
            restored_kernels = img.get_kernels()    #(bl_p , alg_p)
            original_kernels = img.get_original_kernels() #bl_p
            blurred_ssim = img.get_blurred_SSIM() #bl_p
            blurred_psnr = img.get_blurred_PSNR() #bl_p
            filters = img.get_filters() #bl_p
            algorithms = img.get_algorithm() #array
            restored_psnr = img.get_PSNR() #(bl_p,alg_p)
            restored_ssim = img.get_SSIM() #(bl_p,alg_p)
            is_color = img.get_color() #Bool

            restored_path = img.get_restored() #(bl_p,alg_p)

            for blurred_path in img.get_blurred_array():
                tmp_dict = dict()
                tmp_dict['original_path'] = original_path
                tmp_dict['blurred_path'] = blurred_path
                tmp_dict['original_kernel'] = original_kernels[blurred_path]
                tmp_dict['filters'] = filters[blurred_path]
                tmp_dict['blurred_psnr'] = blurred_psnr[blurred_path]
                tmp_dict['blurred_ssim'] = blurred_ssim[blurred_path]
                tmp_dict['is_color'] = is_color
                tmp_dict['restored_paths'] = dict()
                tmp_dict['algorithm'] = algorithms.tolist()
                tmp_dict['kernel_paths'] = dict()
                tmp_dict['psnr'] = dict()
                tmp_dict['ssim'] = dict()


                for alg in algorithms:
                    tmp_dict['restored_paths'][alg] = restored_path[(blurred_path,alg)]
                    tmp_dict['kernel_paths'][alg] = restored_kernels[(blurred_path,alg)]
                    tmp_dict['psnr'][alg] = restored_psnr[(blurred_path,alg)]
                    tmp_dict['ssim'][alg] = restored_ssim[(blurred_path,alg)]
                
                data[f'{counter}'] = tmp_dict
                counter+=1

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        pass

    def load_bind_state(self, bind_path: Path) -> None:
        """Загружает состояние связей из JSON файла."""
        with open(bind_path, 'r', encoding='utf-8') as f:
            data_global = json.load(f)
        
        for i in data_global.keys():
            
            data = data_global[i]

            original_path = data['original_path']
            blurred_path = data['blurred_path']
            original_kernels = {blurred_path: data['original_kernel']}
            filters = data['filters']
            blurred_psnr = {blurred_path: data['blurred_psnr']}
            blurred_ssim = {blurred_path: data['blurred_ssim']}
            is_color = data['is_color']
            restored_paths = dict()
            algorithms = np.array(data['algorithm'])
            kernel_paths = dict()
            psnr = dict()
            ssim = dict()

            tmp_image = Image(original_path,is_color)

            for alg in algorithms:
                restored_paths[(blurred_path,alg)] = data['restored_paths'][alg]
                kernel_paths[(blurred_path,alg)] = data['kernel_paths'][alg]
                psnr[(blurred_path,alg)] = data['psnr'][alg]
                ssim[(blurred_path,alg)] = data['ssim'][alg]

            tmp_image.set_blurred(blurred_path)
            tmp_image.set_blurred_PSNR(blurred_psnr)
            tmp_image.set_blurred_SSIM(blurred_ssim)
            tmp_image.set_current_filter(filters)
            tmp_image.set_original_kernels(original_kernels)

            tmp_image.set_restored(restored_paths)
            tmp_image.set_algorithm(algorithms)
            tmp_image.set_PSNR(psnr)
            tmp_image.set_SSIM(ssim)
            tmp_image.set_kernels(kernel_paths)

            self.processing.images = np.append(self.processing.images, tmp_image)


