import numpy as np
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import math
import pandas as pd
from pathlib import Path

import utils
import filters.base as filter
import metrics 
import algorithms.base as base

import optuna

from IPython.display import display
from extensions import HyperparameterOptimizer
from extensions import ParetoFrontAnalyzer

class Processing:

    '''
    Фреймворк для обработки изображений
    '''
    
    def __init__(self, images_folder='images', blurred_folder='blurred', 
                 restored_folder='restored', data_path='data', color=False, kernel_dir='kernels'):
        '''
            Инициализация фреймворка

            Аргументы:
            -color: тип загрузки изображений цветное/черно-белое
            -folder_path: директория с исходными изображениями
            -folder_path_blurred: директория со смазанными изображениями
            -folder_path_restored: директория с восстановленными изображениями
            -images: массив связей изображений с их смазанными и восстановленными версиями
            -data_path: директория, куда сохранять анализ данных
        '''
        self.color = color
        self.folder_path = Path(images_folder)
        self.folder_path_blurred = Path(blurred_folder)
        self.folder_path_restored = Path(restored_folder)
        self.data_path = Path(data_path)
        self.images = np.array([])
        self.amount_of_blurred = 1
        self.optimizer = HyperparameterOptimizer(self)
        self.analyzer = ParetoFrontAnalyzer(self)
        self.kernel_dir = Path(kernel_dir)
        
        for folder in [self.folder_path, self.folder_path_blurred, 
                    self.folder_path_restored, self.data_path, self.kernel_dir ]:
            folder.mkdir(parents=True, exist_ok=True)

    def changescale(self, color: bool):
        '''
        Изменение способа загрузки изображений
        Аргументы:
            - color: True - цветное, False - черно-белое
        '''
        self.color = color

    def read_all(self):
        '''Загрузка всех изображений из директории'''
        for image_file in self.folder_path.iterdir():
            if image_file.is_file():
                self._load_image(image_file)
    
    def read_one(self, path):
        '''Загрузка одного изображения'''
        image_path = self.folder_path / path
        self._load_image(image_path)

    def _load_image(self, image_path):
        '''Внутренний метод загрузки изображения'''
        image = cv.imread(str(image_path), 
                         cv.IMREAD_COLOR if self.color else cv.IMREAD_GRAYSCALE)
        if image is not None:
            self.images = np.append(self.images, utils.Image(str(image_path), self.color))

    def show(self, size: float = 1.0, kernel_intencity_scale: float = 1.0, kernel_size=1.0):
        '''
        Вывод всех изображений: оригинал, размытые, восстановленные + метрики
        '''
        if not self.images.size:
            print("Нет изображений для отображения")
            return

        h = 0
        alg_arr = []
        for img_obj in self.images:
            img_obj.save_filter()
            h += max(1,img_obj.get_len_filter())
            alg_arr.extend(img_obj.get_algorithm())
        
        alg_arr = list(set(alg_arr))
        w = len(alg_arr) + 2
        

        fig, axes = plt.subplots(2 * h, w, figsize=(5 * w * size, 8 * h * size))
        line = 0
        
        for img_obj in self.images:
            line = self._plot_single_image(img_obj, alg_arr, axes, line, kernel_intencity_scale,kernel_size)
            img_obj.load(img_obj.get_len_filter() - 1)

        # plt.suptitle("ANALYTICS", y=1.02, fontsize=14 * size)
        plt.suptitle(" ", y=1.02, fontsize=14 * size)
        plt.tight_layout()
        plt.show()

    def _plot_single_image(self, img_obj, alg_arr, axes, line, kernel_intencity_scale,kernel_size):
        '''
        Отрисовка одного изображения со всеми его размытыми вариантами
        '''
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
            line = self._plot_images_line(img_obj, blurred_path, original_image, alg_arr, 
                                        restored_psnr, restored_ssim, restored_paths,
                                        blurred_psnr, blurred_ssim, axes, line, filter_name)
            
            line = self._plot_kernels_line(img_obj, blurred_path, alg_arr, original_kernels,
                                        kernels, axes, line, kernel_intencity_scale,kernel_size)
        
        return line

    def _plot_images_line(self, img_obj, blurred_path, original_image, alg_arr,
                        restored_psnr, restored_ssim, restored_paths,
                        blurred_psnr, blurred_ssim, axes, line, filters_name):
        '''
        Отрисовка строки с изображениями
        '''
        blurred_image = cv.imread(str(blurred_path), 
                                cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
        
        plt.subplots_adjust(hspace=0.5)

        axes[line, 0].imshow(cv.cvtColor(original_image, cv.COLOR_BGR2RGB))
        axes[line, 0].set_title("Original", fontsize=12)
        axes[line, 0].axis('off')
        
        psnr_val = blurred_psnr.get(str(blurred_path), math.nan)
        ssim_val = blurred_ssim.get(str(blurred_path), math.nan)
        filter_name = filters_name.get(str(blurred_path), 'missing')


        
        axes[line, 1].imshow(cv.cvtColor(blurred_image, cv.COLOR_BGR2RGB))
        axes[line, 1].set_title(f"{filter_name}\n\nDistorted\nPSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}", fontsize=10)
        axes[line, 1].axis('off')
        
        for col, alg_name in enumerate(alg_arr, 2):
            axes[line, col].axis('off')
            self._plot_restored_image(img_obj, blurred_path, alg_name, restored_psnr, 
                                    restored_ssim, restored_paths, axes, line, col)
        
        return line + 1

    def _plot_restored_image(self, img_obj, blurred_path, alg_name, restored_psnr,
                            restored_ssim, restored_paths, axes, line, col):
        '''
        Отрисовка одного восстановленного изображения
        '''

        try:
            restored_path = restored_paths.get((str(blurred_path), str(alg_name)))
            if restored_path:
                restored_image = cv.imread(restored_path, 
                                        cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
                if restored_image is not None:
                    psnr_val = restored_psnr.get((str(blurred_path), str(alg_name)), math.nan)
                    ssim_val = restored_ssim.get((str(blurred_path), str(alg_name)), math.nan)
                    
                    axes[line, col].imshow(cv.cvtColor(restored_image, cv.COLOR_BGR2RGB))
                    axes[line, col].set_title(f"{alg_name}\nPSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}", fontsize=10)
        except Exception as e:
            pass

    def _plot_kernels_line(self, img_obj, blurred_path, alg_arr, original_kernels,
                        kernels, axes, line, kernel_intencity_scale,kernel_size):
        '''
        Отрисовка строки с ядрами
        '''
        axes[line, 0].axis('off')
        
        original_kernel_path = original_kernels.get(str(blurred_path))
        if original_kernel_path:
            original_kernel = cv.imread(str(original_kernel_path), 
                                    cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
            
            if img_obj.get_color():
                h,w,_ = original_kernel.shape
            else:
                h,w = original_kernel.shape
            
            new_k = round(h/2*(1.0-kernel_size))
            new_w = round(w/2*(1.0-kernel_size))
            if img_obj.get_color():
                original_kernel = original_kernel[new_k:-new_k-1,new_w:-new_w-1,:]
            else:
                original_kernel = original_kernel[new_k:-new_k-1,new_w:-new_w-1]

            if original_kernel is not None:
                axes[line, 1].imshow(np.clip(cv.cvtColor(original_kernel, cv.COLOR_BGR2RGB) * 
                                        kernel_intencity_scale, 0, 255).astype(np.uint8))
                axes[line, 1].set_title("original kernel", fontsize=10)
                axes[line, 1].axis('off')
        
        for col, alg_name in enumerate(alg_arr, 2):
            axes[line, col].axis('off')
            self._plot_restored_kernel(img_obj, blurred_path, alg_name, kernels, 
                                    axes, line, col, kernel_intencity_scale)
        
        return line + 1

    def _plot_restored_kernel(self, img_obj, blurred_path, alg_name, kernels,
                            axes, line, col, kernel_intencity_scale):
        '''Отрисовка одного восстановленного ядра'''
        try:
            kernel_path = kernels.get((str(blurred_path), str(alg_name)))
            if kernel_path:
                kernel = cv.imread(str(kernel_path), 
                                cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
                if kernel is not None:
                    axes[line, col].imshow(np.clip(cv.cvtColor(kernel, cv.COLOR_BGR2RGB) * 
                                                kernel_intencity_scale, 0, 255).astype(np.uint8))
                    axes[line, col].set_title(f"{alg_name} kernel", fontsize=10)
        except Exception as e:
            pass

    def get_metrics(self):
        '''
        Получение метрик в структурированном виде
        '''
        metrics_data = []
        for img_obj in self.images:
            psnr_values = img_obj.get_PSNR()
            ssim_values = img_obj.get_SSIM()
            
            for key, psnr_val in psnr_values.items():
                ssim_val = ssim_values.get(key, math.nan)
                blurred_path, alg_name = key
                metrics_data.append({
                    'image': img_obj.get_original(),
                    'blurred': blurred_path,
                    'algorithm': alg_name,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                })
        
        return pd.DataFrame(metrics_data)

    def clear_input(self):
        '''
        Убирает привязку ко всем загруженным изображениям
        '''
        self.images = np.array([])
    
    def reset(self):
        '''
        Сброс состояний всех изображений до исходного
        Убирает привязку к отфильтрованным и восстановленным изображениями
        '''
        for img_obj in self.images:
            self._reset_single_image(img_obj)
    
    def _reset_single_image(self, img_obj):
        '''
        Сброс состояния одного изображения
        '''
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
            ('filters', np.array([]))
        ]
        
        for attr, default_value in reset_operations:
            getattr(img_obj, f'set_{attr}')(default_value)
    
    def clear_output(self):
        '''
        Удаление всех сгенерированных файлов привязанных отфильрованных и восстановленных изображений
        '''
        for img_obj in self.images:
            self._delete_image_files(img_obj)
        self.reset()
    
    def _delete_image_files(self, img_obj):
        '''
        Удаление всех файлов связанных с одним изображением
        '''
        files_to_delete = []
        
        file_sources = [
            [img_obj.get_blurred()] if img_obj.get_blurred() else [],
            img_obj.get_restored().values(),
            img_obj.get_kernels().values(), 
            img_obj.get_original_kernels().values(),
            img_obj.get_blurred_array()
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
            
    def clear_output_directory(self, warning = 'IT WILL DELETE EVERYTHING!'):
        '''
        Полная очистка выходных директорий с отфильрованными и восстановленными изображениями
        '''
        if not self._confirm_deletion():
            print("Operation canceled")
            return
        
        directories = [self.folder_path_blurred, self.folder_path_restored]
        
        for directory in directories:
            self._clear_directory(directory)
        
        self.reset()  
    
    def clear_restored(self):
        for img in self.images:
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

    def unbind_restored(self):
        for img in self.images:
            img.set_kernels({})
            img.set_restored({})
            img.set_algorithm(np.array([]))

    def _confirm_deletion(self):
        '''
        Подтверждение опасной операции
        '''
        message = (
            f"YOU SURE, YOU WANT TO DELETE EVERY SINGLE FILE IN DIRECTORIES\n"
            f"{self.folder_path_blurred} И {self.folder_path_restored}?\n"
            f"To confirm enter 'YES': "
        )
        return input(message) == 'YES'
    
    def _clear_directory(self, directory_path):
        '''
        Очистка содержимого директории
        '''
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
    
    def clear_all(self):
        '''
        Полная очистка: файлы + состояния + загруженные изображения
        '''
        self.clear_output()  
        self.clear_input()   
        print("Full cleaning completed")
    
    def bind(self, original_image_path, blurred_image_path, original_kernel_path = None, filter_description = "unknown", color:bool = True):
        '''
        Связывает оригинальное изображение с искаженной версией и соответствующим ядром размытия
        
        Аргументы:
            -original_image_path: путь к оригинальному изображению (полностью)
            -blurred_image_path: путь к смазанному изображению (полностью)
            -original_kernel_path: путь к ядру размытия 
            -filter_description: описание примененного фильтра искажения (смаза)
            -color: способ загрузки (True - цветное, False - черно-белое, None - авто)
        '''
        if color is None:
            color = self.color
        
        if not all(os.path.exists(p) for p in [original_image_path, blurred_image_path] if p):
            missing = [p for p in [original_image_path, blurred_image_path] if p and not os.path.exists(p)]
            raise FileNotFoundError(f"Files not found: {missing}")
        
        original = cv.imread(original_image_path, cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE)
        blurred = cv.imread(blurred_image_path, cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE)
        
        if original is None:
            raise ValueError(f"Failed to load original image: {original_image_path}")
        if blurred is None:
            raise ValueError(f"Failed to load blurred image: {blurred_image_path}")
    
        img_obj = utils.Image(original_image_path, color)
        img_obj.set_blurred(blurred_image_path)
        img_obj.set_current_filter(filter_description)

        try:
            psnr_blured = metrics.PSNR(original,blurred)
        except:
            psnr_blured = math.nan

        try:
            ssim_blured = metrics.SSIM(original,blurred)
        except:
            ssim_blured = math.nan

        img_obj.add_blurred_PSNR(psnr_blured,blurred_image_path)
        img_obj.add_blurred_SSIM(ssim_blured,blurred_image_path)

        
        if original_kernel_path:
            if not os.path.exists(original_kernel_path):
                print(f"Kernel not found: {original_kernel_path}")
            else:
                img_obj.add_original_kernel(original_kernel_path, blurred_image_path)
        
        self.images = np.append(self.images, img_obj)
        return img_obj

    def filter(self, filter_processor: filter.FilterBase):
        '''
        Примерние фильтра ко всем изображениям
        '''
        for img_obj in self.images:
            self._apply_single_filter(img_obj, filter_processor)
        
    def _apply_single_filter(self, img_obj, filter_processor):
        '''
        Применение фильтра к одному изображению
        '''
        if img_obj.get_blurred() is not None:
            current_image = img_obj.get_blurred_image()
        else:
            current_image = img_obj.get_original_image()
        
        if current_image is None:
            raise Exception("Не удалось загрузить изображение")
        
        filtered_image = filter_processor.filter(current_image)
        
        if img_obj.get_blurred() is None:
            original_filename = Path(img_obj.get_original()).name
            new_path = self._generate_unique_file_path(self.folder_path_blurred, original_filename)
        else:
            original_filename = Path(img_obj.get_blurred()).name
            new_path = self.folder_path_blurred / original_filename
        
        try:
            psnr_val = metrics.PSNR(current_image, filtered_image)
        except:
            psnr_val = math.nan
        
        try:
            ssim_val = metrics.SSIM(current_image, filtered_image)
        except:
            ssim_val = math.nan
        
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        cv.imwrite(str(new_path), filtered_image)
        img_obj.set_blurred(str(new_path))
        img_obj.add_to_current_filter(filter_processor.discription())
        
        self._process_kernel(img_obj, filter_processor, new_path, original_filename)

    def _process_kernel(self, img_obj, filter_processor, new_path, original_filename):
        '''
        Обработка ядра для фильтра
        '''
        kernels = img_obj.get_original_kernels()
        kernel_path = kernels.get(str(new_path))
        
        if kernel_path is None:
            kernel_image = img_obj.get_original_image().copy()
            kernel_image *= 0
            h, w = kernel_image.shape[:2]
            kernel_image[h//2, w//2] = 255
            
            # new_kernel_path = self.folder_path_blurred / f"kernel_{original_filename}"
            new_kernel_path = self._generate_unique_file_path(self.folder_path_blurred, f"kernel_{original_filename}")
        else:
            kernel_image = cv.imread(str(kernel_path), 
                                cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
            new_kernel_path = Path(kernel_path)
        
        if filter_processor.get_type() != 'noise':
            filtered_kernel = filter_processor.filter(kernel_image)
        else:
            filtered_kernel = kernel_image
        cv.imwrite(str(new_kernel_path), filtered_kernel)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path))

    def _generate_unique_file_path(self, directory, filename):
        '''
        Генерация уникального пути к файлу
        '''
        path = directory / filename
        counter = 1
        
        while path.exists():
            stem, suffix = Path(filename).stem, Path(filename).suffix
            path = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        
        return path
   
    def save_filter(self):
        '''
        Сохранение текущего состояния фильтров
        Переносит изображение из буфера в список
        к изображениям в списке не применяются фильтры
        '''
        for img_obj in self.images:
            img_obj.save_filter()
        self.amount_of_blurred += 1

    def load_filter(self, index):
        '''
        Загрузка состояния фильтров.
        Достает изображение из списка в буфер для изменения
        Аргументы:
            -index: индекс доставаемого изображения
        '''
        for img_obj in self.images:
            img_obj.load(index)
        self.amount_of_blurred -= 1

    def len_blur(self):
        '''
        Количество вариантов размытия
        '''
        return self.amount_of_blurred
    
    def process(self, algorithm_processor: base.DeconvolutionAlgorithm):
        '''
        Восстановление всех изображений
        
        Аргументы:
            -algorithm_processor: метод восстановления изображения
        '''
        alg_name = algorithm_processor.get_name()
        
        for img_obj in self.images:
            self._process_single_image(img_obj, algorithm_processor, alg_name)
        
    def _process_single_image(self, img_obj, algorithm_processor, alg_name):
        '''
        Восстановление одного изображения
        '''        
        original_image = img_obj.get_original_image()
        blurred_image = img_obj.get_blurred_image()
        
        if blurred_image is None:
            blurred_image = original_image
        
        if blurred_image is None:
            print(f"Pass: Unable to load image for {img_obj.get_original()}")
            return
    
        try:
            restored_image, kernel = algorithm_processor.process(blurred_image)
        except Exception as e:
            print(f"Restore error {img_obj.get_original()}: {e}")
            return
        
        restored_path, kernel_path = self._generate_output_paths(img_obj, alg_name)
        
        try:
            cv.imwrite(str(restored_path), restored_image)
            cv.imwrite(str(kernel_path), (kernel*255))
        except Exception as e:
            print(f"Saving results error {restored_path}: {e}")
            return
        

        original_image = np.atleast_3d(original_image)
        restored_image = np.atleast_3d(restored_image)
        self._calculate_and_save_metrics(img_obj, original_image, restored_image, 
                                   restored_path, kernel_path, alg_name)
    
    def _generate_output_paths(self, img_obj, alg_name):
        '''
        Генерация уникальных путей для сохранения результатов
        '''
        if img_obj.get_blurred():
            base_path = Path(img_obj.get_blurred())
        else:
            base_path = Path(img_obj.get_original())
        
        base_name = base_path.stem
        
        restored_path = self._generate_unique_file_path(
            self.folder_path_restored, 
            f"{base_name}_{alg_name}{base_path.suffix}"
        )
        
        kernel_path = self._generate_unique_file_path(
            self.folder_path_restored,
            f"{base_name}_{alg_name}_kernel{base_path.suffix}"
        )
        
        return restored_path, kernel_path
    
    def _calculate_and_save_metrics(self, img_obj, original_image, restored_image, 
                              restored_path, kernel_path, alg_name):
        '''
        Расчет метрик и обновление данных изображения
        '''
        try:
            psnr_val = metrics.PSNR(original_image, restored_image)
        except Exception as e:
            print(f"PSNR calculation error: {e}")
            psnr_val = math.nan
        
        try:
            ssim_val = metrics.SSIM(original_image, restored_image)
        except Exception as e:
            print(f"SSIM calculation error: {e}")
            ssim_val = math.nan
        
        blurred_ref = img_obj.get_blurred()
        
        img_obj.add_PSNR(psnr_val, blurred_ref, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_ref, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(str(restored_path), blurred_ref, alg_name)
        img_obj.add_kernel(str(kernel_path), blurred_ref, alg_name)
        
        print(f"Restored: {Path(restored_path).name} (PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f})")
            
    def param_process(self, algorithm_processor: base.DeconvolutionAlgorithm, param: dict):
        '''
        Восстановление всех изображений с заданными гиперпараметрами, гиперпараметр остается заданным
        
        Аргументы:
            -algorithm_processor: метод восстановления изображения
            -param: гиперпараметр метода
        '''
        algorithm_processor.change_param(param)
        return self.process(algorithm_processor)

    def full_process(self, filters: list, methods: list, size: float = 0.75, kernel_intencity_scale = 10.0):
        '''
        Пайплайн применения фильтров с последующим восстановлением и анализов результатов
        
        Аргументы:
            -filters: массив массивов объектов класса FilterBase (фильтры к изображению) [[],[]]
            -methods: массив объектов класса DeconvolutionAlgorithm (методы восстановления) []
            -size: размер таблицы лучших/худших метрик
            -kernel_intencity_scale: цвет пиксилей psf при выводе
        '''
        data_dict, analysis_dicts = self._initialize_data_structures()
        for img_obj in self.images:
            self._process_image_pipeline(img_obj, filters, methods, data_dict, analysis_dicts)
        
        self._finalize_processing(data_dict, analysis_dicts, size, kernel_intencity_scale)
        
    def _initialize_data_structures(self):
        '''
        Инициализация структур данных для сбора результатов
        '''
        data_dict = {}
        analysis_dicts = {
            'images_dict': {},
            'ssim_dict': {},
            'psnr_dict': {},
            'kernels_dict': {}
        }
        return data_dict, analysis_dicts    
    
    def _process_image_pipeline(self, img_obj, filters, methods, data_dict, analysis_dicts):
        '''
        Полный пайплайн обработки для одного изображения
        '''
        self._apply_filter_chains(img_obj, filters)
        
        self._apply_restoration_and_collect_data(img_obj, methods, data_dict, analysis_dicts)   
    
    def _apply_filter_chains(self, img_obj, filters):
        '''
        Применение цепочек фильтров к изображению
        '''
        for filter_chain in filters:
            self._apply_single_filter_chain(img_obj, filter_chain)
            img_obj.save_filter()

    def _apply_single_filter_chain(self, img_obj, filter_chain):
        '''
        Применение одной цепочки фильтров
        '''
        original_image = img_obj.get_original_image()
        current_image = img_obj.get_blurred_image()
        
        if current_image is None:
            current_image = original_image
        
        if current_image is None:
            raise Exception("Failed to load image")
        
        filtered_image = self._apply_filters_sequential(current_image, filter_chain, img_obj)
        
        new_path = self._generate_blurred_image_path(img_obj)
        cv.imwrite(new_path, filtered_image)
        
        self._update_blurred_image_data(img_obj, original_image, filtered_image, new_path)
        
        self._process_kernel_chain(img_obj, filter_chain, new_path)
    
    def _apply_filters_sequential(self, image, filter_chain, img_obj):
        '''
        Последовательное применение фильтров цепочки
        '''
        filtered_image = image.copy()
        for filter_processor in filter_chain:
            filtered_image = filter_processor.filter(filtered_image)
            img_obj.add_to_current_filter(filter_processor.discription())
        return filtered_image

    def _generate_blurred_image_path(self, img_obj):
        '''
        Генерация пути для размытого изображения
        '''
        if img_obj.get_blurred() is None:
            original_filename = Path(img_obj.get_original()).name
            return self._generate_unique_file_path(self.folder_path_blurred, original_filename)
        else:
            original_filename = Path(img_obj.get_blurred()).name
            return self.folder_path_blurred / original_filename
    
    def _update_blurred_image_data(self, img_obj, original_image, filtered_image, new_path):
        '''
        Обновление данных размытого изображения
        '''
        try:
            psnr_val = metrics.PSNR(original_image, filtered_image)
            ssim_val = metrics.SSIM(original_image, filtered_image)
        except:
            psnr_val = ssim_val = math.nan
        
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        img_obj.set_blurred(str(new_path))
    
    def _process_kernel_chain(self, img_obj, filter_chain, new_path):
        '''
        Обработка ядра для цепочки фильтров
        '''
        kernels = img_obj.get_original_kernels()
        kernel_path = kernels.get(str(new_path))
        
        if kernel_path is None:
            kernel_image = self._create_delta_kernel(img_obj)
            new_kernel_path = self.folder_path_blurred / f"kernel_{Path(new_path).name}"
        else:
            kernel_image = cv.imread(str(kernel_path), 
                                cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
            new_kernel_path = Path(kernel_path)
        
        for filter_processor in filter_chain:
            kernel_image = filter_processor.filter(kernel_image)
        
        cv.imwrite(str(new_kernel_path), kernel_image)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path)) 
        
    def _create_delta_kernel(self, img_obj):
        '''Создание дельта-функции (единичного импульса)'''
        kernel_image = img_obj.get_original_image().copy()
        kernel_image *= 0
        h, w = kernel_image.shape[:2]
        kernel_image[h//2, w//2] = 255
        return kernel_image   
    
    def _apply_restoration_and_collect_data(self, img_obj, methods, data_dict, analysis_dicts):
        '''Восстановление изображений и сбор данных'''
        for blurred_path in img_obj.get_blurred_array():
            blurred_image = cv.imread(str(blurred_path), 
                                    cv.IMREAD_COLOR if img_obj.get_color() else cv.IMREAD_GRAYSCALE)
            
            for algorithm in methods:
                self._restore_single_image(img_obj, blurred_path, blurred_image, algorithm, data_dict)
            
            self._collect_analysis_data(img_obj, blurred_path, analysis_dicts)
            
    def _restore_single_image(self, img_obj, blurred_path, blurred_image, algorithm, data_dict):
        '''
        Восстановление одного изображения алгоритмом
        '''
        alg_name = algorithm.get_name()
        original_image = img_obj.get_original_image()
        
        try:
            restored_image, restored_kernel = algorithm.process(blurred_image)
            
            restored_path, kernel_path = self._generate_restoration_paths(blurred_path, alg_name)
            
            cv.imwrite(restored_path, restored_image)
            cv.imwrite(kernel_path, restored_kernel)
            
            psnr_val, ssim_val = self._calculate_restoration_metrics(original_image, restored_image)
            
            self._update_restoration_data(img_obj, blurred_path, alg_name, 
                                        restored_path, kernel_path, psnr_val, ssim_val)
            
            self._collect_algorithm_data(data_dict, alg_name, algorithm, img_obj,
                                    blurred_path, restored_path, kernel_path,
                                    psnr_val, ssim_val, restored_image, blurred_image)
            
        except Exception as e:
            print(f"Restore error {blurred_path} of {alg_name}: {e}")
    
    def _generate_restoration_paths(self, blurred_path, alg_name):
        '''
        Генерация путей для восстановленных изображений и ядер
        '''
        blurred_basename = Path(blurred_path).stem
        ext = Path(blurred_path).suffix
        
        restored_path = self._generate_unique_file_path(
            self.folder_path_restored, 
            f"{blurred_basename}_{alg_name}{ext}"
        )
        
        kernel_path = self._generate_unique_file_path(
            self.folder_path_restored,
            f"{blurred_basename}_{alg_name}_kernel{ext}"
        )
        
        return str(restored_path), str(kernel_path)
        
    def _calculate_restoration_metrics(self, original_image, restored_image):
        '''Расчет метрик восстановления'''
        try:
            psnr_val = metrics.PSNR(original_image, restored_image)
            ssim_val = metrics.SSIM(original_image, restored_image)
            return psnr_val, ssim_val
        except:
            return math.nan, math.nan  
    
    def _update_restoration_data(self, img_obj, blurred_path, alg_name, 
                           restored_path, kernel_path, psnr_val, ssim_val):
        '''
        Обновление данных восстановления
        '''
        img_obj.add_PSNR(psnr_val, blurred_path, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_path, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(restored_path, blurred_path, alg_name)
        img_obj.add_kernel(kernel_path, blurred_path, alg_name)

    def _collect_algorithm_data(self, data_dict, alg_name, algorithm, img_obj,
                            blurred_path, restored_path, kernel_path,
                            psnr_val, ssim_val, restored_image, blurred_image):
        '''
        Сбор данных для анализа алгоритмов
        '''
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
            
    def _collect_analysis_data(self, img_obj, blurred_path, analysis_dicts):
        '''
        Сбор данных для общего анализа
        '''
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
    
    def _finalize_processing(self, data_dict, analysis_dicts, size, kernel_intencity_scale):
        '''
        Финальная обработка и визуализация результатов
        '''
        self.show(size, kernel_intencity_scale)
        
        for alg_name, alg_data in data_dict.items():
            df_data = pd.DataFrame(alg_data)
            display(df_data)
            df_data.to_csv(self.data_path / f"{alg_name}.csv", index=False)
        
        self.pareto()

    
    def process_hyperparameter_optimization(self, *args, **kwargs):
        return self.optimizer.execute(*args, **kwargs)
    
    def pareto(self):
        return self.analyzer.execute()
    
     
        


