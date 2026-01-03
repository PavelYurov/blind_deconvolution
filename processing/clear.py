"""
Модуль очищения связей из фреймворка и удаления
изображений из указанных директорий.

Возможности:
    - Удалить все связи.
    - Удалить обработанные изображения.
    - Удалить восстановленные изображения.
    - Удалить все из указанных директорий.

Авторы: Юров П.И., Беззаборов А.А.
"""
import numpy as np
import os
import glob
from pathlib import Path
from typing import  Any

from processing.utils import Image


class ModuleClear:
    """
    Модуль очищения связей из фреймворка и удаления
    изображений из указанных директорий.

    Возможности:
        - Удалить все связи.
        - Удалить обработанные изображения.
        - Удалить восстановленные изображения.
        - Удалить все из указанных директорий.
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

    def clear_input(self) -> None:
        """Убирает привязку ко всем загруженным изображениям."""
        self.processing.images = np.array([])
    
    def reset(self) -> None:
        """Сброс состояний всех изображений до исходного."""
        for img_obj in self.processing.images:
            self._reset_single_image(img_obj)
    
    def _reset_single_image(self, img_obj: Image) -> None:
        """Сброс состояния одного изображения."""
        reset_operations = [
            ('blurred', None),
            ('restored', {}),
            ('kernels', {}),
            ('PSNR', {}),
            ('SSIM', {}),
            ('original_kernels', {}),
            ('algorithm', np.array([])),
            ('blurred_array', np.array([])),
            ('current_filter', None),
            ('filters', np.array([])),
            ('preprocessed_blurred_path',{})
        ]
        
        for attr, default_value in reset_operations:
            getattr(img_obj, f'set_{attr}')(default_value)
    
    def clear_output(self) -> None:
        """Удаление всех сгенерированных файлов."""
        for img_obj in self.processing.images:
            self._delete_image_files(img_obj)
        self.reset()
    
    def _delete_image_files(self, img_obj: Image) -> None:
        """Удаление всех файлов связанных с одним изображением."""
        files_to_delete = []
        
        file_sources = [
            [img_obj.get_blurred()] if img_obj.get_blurred() else [],
            img_obj.get_restored().values(),
            img_obj.get_kernels().values(), 
            img_obj.get_original_kernels().values(),
            img_obj.get_blurred_array(),
            img_obj.get_preprocessed_blurred_path().values()
        ]
    
        for file_source in file_sources:
            files_to_delete.extend(
                file_path for file_path in file_source 
                if file_path and os.path.exists(file_path)
            )
        
        for file_path in set(files_to_delete): 
            try:
                os.remove(file_path)
                print(f"Удален файл: {file_path}")
            except Exception as e:
                print(f"Ошибка удаления {file_path}: {e}")
            
    def clear_output_directory(self, warning: str = 'IT WILL DELETE EVERYTHING!') -> None:
        """Полная очистка выходных директорий."""
        if not self._confirm_deletion():
            print("Operation canceled")
            return
        
        directories = [self.processing.folder_path_blurred, 
                       self.processing.folder_path_restored, 
                       self.processing.preprocess_dir]
        
        for directory in directories:
            self._clear_directory(directory)
        
        self.reset()  
    
    def clear_restored(self) -> None:
        """Удаляет восстановленные изображения из каждой связи."""
        for img in self.processing.images:
            files_to_delete = []
        
            file_sources = [
                img.get_restored().values(),
                img.get_kernels().values()
            ]
        
            for file_source in file_sources:
                files_to_delete.extend(
                    file_path for file_path in file_source 
                    if file_path and os.path.exists(file_path)
                )
            
            for file_path in set(files_to_delete): 
                try:
                    os.remove(file_path)
                    print(f"Удален файл: {file_path}")
                except Exception as e:
                    print(f"Ошибка удаления {file_path}: {e}")
            
            img.set_kernels({})
            img.set_restored({})
            img.set_algorithm(np.array([]))

    def unbind_restored(self) -> None:
        """Разрывает связь, убирая все восстановленные."""
        for img in self.processing.images:
            img.set_kernels({})
            img.set_restored({})
            img.set_algorithm(np.array([]))

    def _confirm_deletion(self) -> bool:
        """Подтверждение опасной операции."""
        message = (
            f"YOU SURE, YOU WANT TO DELETE EVERY SINGLE FILE IN DIRECTORIES\n"
            f"{self.processing.folder_path_blurred} И {self.processing.folder_path_restored}?\n"
            f"To confirm enter 'YES': "
        )
        return input(message) == 'YES'
    
    def _clear_directory(self, directory_path: Path) -> None:
        """Очистка содержимого директории."""
        if not os.path.exists(directory_path):
            print(f"Warning: directory {directory_path} does not exist")
            return
        
        try:
            os.makedirs(directory_path, exist_ok=True)
            
            for file_path in glob.glob(os.path.join(directory_path, '*')):
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Deletion error {file_path}: {e}")
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                    print(f"Directory deleted: {file_path}")
                    
        except Exception as e:
            print(f"Critical error while cleaning {directory_path}: {e}")
    
    def clear_all(self) -> None:
        """Полная очистка: файлы + состояния + загруженные изображения."""
        self.clear_output()  
        self.clear_input()   
        print("Full cleaning completed")
