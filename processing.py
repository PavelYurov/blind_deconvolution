"""
Фреймворк для обработки изображений
Основной интерфейс

Содержит:
    - Методы считывания изображений из директорий
    - Методы применения фильтров смаза и шума к изображениям
    - Методы применения алгоритмов восстановления к смазанным изображениям
    - Автоматический посчет метрик для смазанных и восстановленных изображений
    - Методы сохранения результатов, метрик в виде json и csv файлов
    - Автоматическое отслеживание связей между изображениями

Авторы: Юров П.И. Беззаборов А.А. Куропатов К.Л.
"""
import numpy as np
import os
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
from pathlib import Path
import json
from typing import Callable, Optional, Tuple, Dict, Any
from skimage.exposure import equalize_hist, equalize_adapthist, histogram, match_histograms, cumulative_distribution

import utils
import filters.base as filters
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
    
    def __init__(self, 
                 images_folder: str = 'images',
                 blurred_folder: str = 'blurred', 
                 restored_folder: str = 'restored', 
                 data_path: str = 'data', 
                 color: bool = False, 
                 kernel_dir: str = 'kernels', 
                 dataset_path: str = 'dataset') -> None:
        '''
            Инициализация фреймворка

            Аргументы:
            -color: тип загрузки изображений цветное/черно-белое
            -folder_path: директория с исходными изображениями
            -folder_path_blurred: директория со смазанными изображениями
            -folder_path_restored: директория с восстановленными изображениями
            -images: массив связей изображений с их смазанными и восстановленными версиями
            -data_path: директория, куда сохранять анализ данных
            -dataset_path: директория, куда сохраняются сшифки метаданных датасетов
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
        self.dataset_path = Path(dataset_path)
        
        for folder in [self.folder_path, 
                       self.folder_path_blurred, 
                       self.folder_path_restored, 
                       self.data_path, 
                       self.kernel_dir, 
                       self.dataset_path ]:
            folder.mkdir(parents=True, exist_ok=True)

    def changescale(self, color: bool) -> None:
        '''
        Изменение способа загрузки изображений
        Аргументы:
            - color: True - цветное, False - черно-белое
        '''
        self.color = color

    def _imread(self, path: str, color: bool) -> Optional[np.ndarray]:
        '''Загружает изображение соответствующего цветового формата'''
        return cv.imread(path, cv.IMREAD_COLOR if color else cv.IMREAD_GRAYSCALE)

    def read_all(self) -> None:
        '''Загрузка всех изображений из директории'''
        for image_file in self.folder_path.iterdir():
            if image_file.is_file():
                self._load_image(image_file)
    
    def read_one(self, path: Path) -> None:
        '''Загрузка одного изображения'''
        image_path = self.folder_path / path
        self._load_image(image_path)

    def _load_image(self, image_path: str) -> None:
        '''Внутренний метод загрузки изображения'''
        image = self._imread(str(image_path), self.color)
        if image is not None:
            self.images = np.append(self.images, 
                                    utils.Image(
                                        str(image_path), 
                                        self.color))

    def show(self, 
             size: float = 1.0, 
             kernel_intencity_scale: float = 1.0, 
             kernel_size: float = 1.0) -> None:
        
        '''Вывод всех изображений: оригинал, размытые, восстановленные + метрики'''

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
            line = self._plot_single_image(img_obj, alg_arr, axes, 
                                           line, kernel_intencity_scale, kernel_size)
            img_obj.load(img_obj.get_len_filter() - 1)

        plt.suptitle(" ", y=1.02, fontsize=14 * size)
        plt.tight_layout()
        plt.show()

    def _plot_single_image(self, 
                           img_obj: utils.Image, 
                           alg_arr: np.ndarray, 
                           axes: Any, 
                           line: int, 
                           kernel_intencity_scale: float,
                           kernel_size: Any) -> int:
        '''Отрисовка одного изображения со всеми его размытыми вариантами'''
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

    def _plot_images_line(self, 
                          img_obj: utils.Image, 
                          blurred_path: str, 
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
        '''Отрисовка строки с изображениями'''
        blurred_image = self._imread(str(blurred_path), img_obj.get_color())
        
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
        
        for col, alg_name in enumerate(alg_arr, 2):
            axes[line, col].axis('off')
            self._plot_restored_image(img_obj, blurred_path, alg_name, restored_psnr, 
                                    restored_ssim, restored_paths, axes, line, col)
        
        return line + 1

    def _plot_restored_image(self, 
                             img_obj: utils.Image, 
                             blurred_path: str, 
                             alg_name: str, 
                             restored_psnr: Dict[Tuple[str, str], str], 
                             restored_ssim: Dict[Tuple[str, str], str], 
                             restored_paths: Dict[Tuple[str, str], str], 
                             axes: Any, 
                             line: int, 
                             col: int) -> None:
        
        '''Отрисовка одного восстановленного изображения'''

        try:
            restored_path = restored_paths.get((str(blurred_path), str(alg_name)))
            if restored_path:
                restored_image = self._imread(restored_path, img_obj.get_color())
                if restored_image is not None:
                    psnr_val = restored_psnr.get((str(blurred_path), str(alg_name)), math.nan)
                    ssim_val = restored_ssim.get((str(blurred_path), str(alg_name)), math.nan)
                    
                    axes[line, col].imshow(cv.cvtColor(restored_image, cv.COLOR_BGR2RGB))
                    axes[line, col].set_title(f"{alg_name}\nPSNR: {psnr_val:.4f} | SSIM: {ssim_val:.4f}", fontsize=10)
        except Exception as e:
            pass

    def _crop_kernel_image(self, kernel_image: np.ndarray, padding: int = 10) -> np.ndarray:
        '''Обрезает изображение ядра до его содержимого с добавлением отступа.'''
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
                           img_obj: utils.Image, 
                           blurred_path: str, 
                           alg_arr: np.ndarray, 
                           original_kernels: Dict[str, str], 
                           kernels: Dict[Tuple[str, str], str], 
                           axes: Any, 
                           line: int, 
                           kernel_intencity_scale: float, 
                           kernel_size) -> int:
        '''
        Отрисовка строки с ядрами с сохранением их оригинальных пропорций и обрезкой.
        '''
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
        
        for col, alg_name in enumerate(alg_arr, 2):
            axes[line, col].axis('off')
            self._plot_restored_kernel(img_obj, blurred_path, alg_name, kernels, 
                                    axes, line, col, kernel_intencity_scale)
        
        return line + 1

    def _plot_restored_kernel(self, 
                              img_obj: utils.Image, 
                              blurred_path: str, 
                              alg_name: str, 
                              kernels: Dict[Tuple[str, str], str],
                              axes: Any, 
                              line: int, 
                              col: int, 
                              kernel_intencity_scale: float) -> None:
        '''Отрисовка одного восстановленного ядра с сохранением пропорций и обрезкой'''
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

    def _float_img_to_int(self, image: np.ndarray) -> np.ndarray:
        '''Переводит изображение из диапозона [0.0 1.0] в диапозон [0 255]'''
        return np.clip(image*255.0, 0.0, 255.0).astype(np.int16)
    
    def histogram_equalization(self, 
                               view_histogram: bool = False) -> None:
        '''Выполняет выравнивание гистограмм'''
        for img_obj in self.images:
            current_image = img_obj.get_blurred_image()
            original_blurred_image = current_image.copy()
            filtered_image = equalize_hist(current_image, nbins=256)
            filtered_image = self._float_img_to_int(filtered_image)
            
            original_filename = Path(img_obj.get_blurred()).name
            new_path = self.folder_path_blurred / f'hist_eq_orig_{str(original_filename)}'

            cv.imwrite(str(new_path), original_blurred_image)
            cv.imwrite(str(img_obj.get_blurred()), filtered_image)

            img_obj.set_he_data(new_path)

            if (view_histogram):
                hist1 = histogram(original_blurred_image)
                hist2 = histogram(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.bar(hist1[1], hist1[0], alpha=0.5, color='blue')
                plt.bar(hist2[1], hist2[0], alpha=0.5, color='red')
                plt.grid(alpha=0.3)
                plt.show()
                
                cdf1 = cumulative_distribution(original_blurred_image)
                cdf2 = cumulative_distribution(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.plot(cdf1[0], cdf1[1], color='blue')
                plt.plot(cdf2[0], cdf2[1], color='red')
                plt.show()

    def histogram_equalization_CLAHE(self, 
                                     view_histogram: bool = False, 
                                     clip_limit: float = 0.01) -> None:
        '''Выполняет адаптивное выравнивание гистограмм с ограничением контрастности'''
        for img_obj in self.images:
            current_image = img_obj.get_blurred_image()
            original_blurred_image = current_image.copy()
            filtered_image = equalize_adapthist(current_image, 
                                                nbins=256, 
                                                clip_limit=clip_limit)
            filtered_image = self._float_img_to_int(filtered_image)
            
            original_filename = Path(img_obj.get_blurred()).name
            new_path = self.folder_path_blurred / f'hist_eq_orig_{str(original_filename)}'

            cv.imwrite(str(new_path), original_blurred_image)
            cv.imwrite(str(img_obj.get_blurred()), filtered_image)

            img_obj.set_he_data(new_path)

            if (view_histogram):
                hist1 = histogram(original_blurred_image)
                hist2 = histogram(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.bar(hist1[1], hist1[0], alpha=0.5, color='blue')
                plt.bar(hist2[1], hist2[0], alpha=0.5, color='red')
                plt.grid(alpha=0.3)
                plt.show()
                
                cdf1 = cumulative_distribution(original_blurred_image)
                cdf2 = cumulative_distribution(filtered_image)
                plt.figure(figsize=(12, 6))
                plt.plot(cdf1[0], cdf1[1], color='blue')
                plt.plot(cdf2[0], cdf2[1], color='red')
                plt.show()
    
    def inverse_histogram_equalization(self, 
                                       view_histogram: bool = False) -> None:
        '''Обращает выравнивание гистограмм'''
        for img_obj in self.images:
            current_image = img_obj.get_blurred_image()
            blurred_path = img_obj.get_blurred()
            restored_array = img_obj.get_restored()
            original_blurred_image = self._imread(img_obj.get_he_data(), img_obj.get_color())
            
            filtered_image = self.inverse_histogram_equalization_one(current_image, 
                                                                original_blurred_image, 
                                                                view_histogram)
            
            cv.imwrite(str(img_obj.get_blurred()), filtered_image)

            for alg_name in img_obj.get_algorithm():
                current_image_path = restored_array[(blurred_path, alg_name)]
                current_image = self._imread(current_image_path, img_obj.get_color())
                
                filtered_image = self.inverse_histogram_equalization_one(current_image, 
                                                                         original_blurred_image, 
                                                                         view_histogram)
                cv.imwrite(str(current_image_path), filtered_image)
                     
    def inverse_histogram_equalization_one(self, 
                                           current_image: np.ndarray, 
                                           original_blurred_image: np.ndarray, 
                                           view_histogram: bool = False) -> np.ndarray:
        '''Выподняет обращение выравнивания для одного изображения'''

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

    def get_table(self,
                   table_path: Path, 
                   display_table: bool = False) -> None:
        '''
        Получение метрик в структурированном виде
        '''
        data = {}
        data = self._collect_data(data) 
        self._save_data_to_csv(data, table_path, display_table)

    def _collect_data(self, data: Dict[str, Any]) -> Dict[str, Any]: #_collect_analysis_data надо поменять на это
        """
        собирает и сохраняет информацию для общего анализа
        выдает 1 таблицу с метриками, ядрами и ссылками на изображениями
        """
        for img in self.images:
            original_image = img.get_original()
            img.save_filter()

            blurred_kernel_array = img.get_original_kernels()
            blurred_psnr_array = img.get_blurred_PSNR()
            blurred_ssim_array = img.get_blurred_SSIM()

            algorithm_kernel = img.get_kernels()
            algorithm_restored_image = img.get_restored()
            algorithm_restored_psnr = img.get_PSNR()
            algorithm_restored_ssim = img.get_SSIM()
            
            #линия за линией
            for blurred_image in img.get_blurred_array(): #подразумеваем, что она точно существует
                data.setdefault('original', []).append(original_image)
                data.setdefault('kernel blur', []).append(blurred_kernel_array.get(str(blurred_image), 'missing'))
                data.setdefault('blurred', []).append(blurred_image)

                data.setdefault('blurred psnr', []).append(blurred_psnr_array.get(str(blurred_image), math.nan))
                data.setdefault('blurred ssim', []).append(blurred_ssim_array.get(str(blurred_image), math.nan))
                for algorithm_name in img.get_algorithm():
                    data.setdefault(f"kernel_{algorithm_name}", []).append(
                        algorithm_kernel.get((str(blurred_image), str(algorithm_name)), 'missing')
                    )
                    data.setdefault(algorithm_name, []).append(
                        algorithm_restored_image.get((str(blurred_image), str(algorithm_name)), 'missing')
                    )
                    data.setdefault(f"psnr_{algorithm_name}", []).append(
                        algorithm_restored_psnr.get((str(blurred_image), str(algorithm_name)), math.nan)
                    )
                    data.setdefault(f"ssim_{algorithm_name}", []).append(
                        algorithm_restored_ssim.get((str(blurred_image), str(algorithm_name)), math.nan)
                    )
        return data

    def _save_data_to_csv(self, 
                          data: Dict[str, Any], 
                          path: Path, 
                          display_table: bool = False) -> None:
        """
        сохраняет словарь в csv файл
        display: bool - выводит сохраненный датафрейм
        """
        df_data = pd.DataFrame(data)
        if display_table:
            display(df_data)
        df_data.to_csv(path, index=False)

    def clear_input(self) -> None:
        '''
        Убирает привязку ко всем загруженным изображениям
        '''
        self.images = np.array([])
    
    def reset(self) -> None:
        '''
        Сброс состояний всех изображений до исходного
        Убирает привязку к отфильтрованным и восстановленным изображениями
        '''
        for img_obj in self.images:
            self._reset_single_image(img_obj)
    
    def _reset_single_image(self, img_obj: utils.Image) -> None:
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
    
    def clear_output(self) -> None:
        '''
        Удаление всех сгенерированных файлов привязанных отфильрованных и восстановленных изображений
        '''
        for img_obj in self.images:
            self._delete_image_files(img_obj)
        self.reset()
    
    def _delete_image_files(self, img_obj: utils.Image) -> None:
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
            
    def clear_output_directory(self, warning: str = 'IT WILL DELETE EVERYTHING!') -> None:
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
    
    def clear_restored(self) -> None:
        '''
        Удаляет восстановленные изображения из каждой связи
        '''
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

    def unbind_restored(self) -> None:
        '''
        Разрывает связь, убирая все восстановленные
        '''
        for img in self.images:
            img.set_kernels({})
            img.set_restored({})
            img.set_algorithm(np.array([]))

    def _confirm_deletion(self) -> bool:
        '''
        Подтверждение опасной операции
        '''
        message = (
            f"YOU SURE, YOU WANT TO DELETE EVERY SINGLE FILE IN DIRECTORIES\n"
            f"{self.folder_path_blurred} И {self.folder_path_restored}?\n"
            f"To confirm enter 'YES': "
        )
        return input(message) == 'YES'
    
    def _clear_directory(self, directory_path: Path) -> None:
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
    
    def clear_all(self) -> None:
        '''
        Полная очистка: файлы + состояния + загруженные изображения
        '''
        self.clear_output()  
        self.clear_input()   
        print("Full cleaning completed")
    
    def bind(self, 
             original_image_path: Path, 
             blurred_image_path: Path, 
             original_kernel_path: Optional[Path] = None, 
             filter_description: str = "unknown", 
             color: bool = True) -> utils.Image:
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
        
        original = self._imread(original_image_path, color)
        blurred = self._imread(blurred_image_path, color)
        
        if original is None:
            raise ValueError(f"Failed to load original image: {original_image_path}")
        if blurred is None:
            raise ValueError(f"Failed to load blurred image: {blurred_image_path}")
    
        img_obj = utils.Image(original_image_path, color)
        img_obj.set_blurred(blurred_image_path)
        img_obj.set_current_filter(filter_description)

        psnr_blured, ssim_blured = self._calculate_metrics(original, blurred)

        img_obj.add_blurred_PSNR(psnr_blured,blurred_image_path)
        img_obj.add_blurred_SSIM(ssim_blured,blurred_image_path)

        
        if original_kernel_path:
            if not os.path.exists(original_kernel_path):
                print(f"Kernel not found: {original_kernel_path}")
            else:
                img_obj.add_original_kernel(original_kernel_path, blurred_image_path)
        
        self.images = np.append(self.images, img_obj)
        return img_obj

    def save_bind_state(self, file_path: Optional[Path] = None) -> None:
        """
        пусть сохраняет images полностью
        ФАЙЛ - JSON!!!
        """
        if file_path is None:
            file_path = os.path.join(self.dataset_path, 'dataset.json')
        data = dict()
        counter = 1
        self.save_filter()

        for img in self.images:
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
        '''
        Загружает images из файла
        Файл - Json
        '''
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

            tmp_image = utils.Image(original_path,is_color)

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

            self.images = np.append(self.images, tmp_image)

    def custom_filter(self, 
                      kernel_image_path: Path, 
                      kernel_npy_path: Path) -> None:
        '''
        Примерние созданного фильтра ко всем оригинальным изображениям
        '''
        for img_obj in self.images:
            self._apply_single_custom_filter(img_obj, kernel_image_path, kernel_npy_path)

    def _apply_single_custom_filter(self, 
                                    img_obj: utils.Image, 
                                    kernel_image_path: Path, 
                                    kernel_npy_path: Path) -> None:
        '''
        Примерние созданного фильтра ко одному изображению
        '''
        current_image = img_obj.get_original_image()
        kernel = np.load(kernel_npy_path)
        if current_image is None:
            raise Exception("Не удалось загрузить изображение")
        filtered_image = cv.filter2D(current_image, -1, kernel)
        original_filename = Path(img_obj.get_original()).stem
        blur_filename = Path(kernel_image_path).stem
        filtered_filename =  self.folder_path_blurred / f"{original_filename}_{blur_filename}.png"

        psnr_val, ssim_val = self._calculate_metrics(current_image, filtered_image)

        img_obj.add_blurred_PSNR(psnr_val, str(filtered_filename))
        img_obj.add_blurred_SSIM(ssim_val, str(filtered_filename))
        cv.imwrite(str(filtered_filename), filtered_image)
        img_obj.set_blurred(str(filtered_filename))
        img_obj.add_to_current_filter(blur_filename)
        img_obj.add_original_kernel(str(kernel_image_path), str(filtered_filename))

    def show_line(self, window_scale: float = 1.0, fontsize: int = 8) -> None:
        """
        self.show но в строчку
        window_scale - регулирует размер окна
        """
        for img in self.images:
            self._show_line_single_image(img,window_scale,fontsize)

    def _show_line_single_image(self, 
                                img, 
                                window_scale: float=1.0, 
                                fontsize: int = 8) -> None:
        '''
        Рисует одну строчку из изображений
        '''
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
        '''Рисует изображение из файла'''
        if os.path.exists(path):
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
        else:
            print(f"  - ПРЕДУПРЕЖДЕНИЕ: Файл не найден, будет показана заглушка: {path}")
            image = np.zeros((100, 100), dtype=np.uint8) # Черный квадрат-заглушка
        return image

    def filter(self, filter_processor: filters.FilterBase) -> None:
        '''
        Примерние фильтра ко всем изображениям
        '''
        for img_obj in self.images:
            self._apply_single_filter(img_obj, filter_processor)
        
    def _apply_single_filter(self, 
                             img_obj: utils.Image, 
                             filter_processor: filters.FilterBase) -> None:
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

        psnr_val, ssim_val = self._calculate_metrics(img_obj.get_original_image(), filtered_image)
        
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        cv.imwrite(str(new_path), filtered_image)
        img_obj.set_blurred(str(new_path))
        img_obj.add_to_current_filter(filter_processor.discription())
        
        self._process_kernel(img_obj, filter_processor, new_path, original_filename)

    def _process_kernel(self, 
                        img_obj: utils.Image, 
                        filter_processor: filters.FilterBase, 
                        new_path: Path, 
                        original_filename: str) -> None:
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
            
            new_kernel_path = self._generate_unique_file_path(self.folder_path_blurred, f"kernel_{original_filename}")
        else:
            kernel_image = self._imread(str(kernel_path), img_obj.get_color())
            new_kernel_path = Path(kernel_path)
        
        if filter_processor.get_type() != 'noise':
            filtered_kernel = filter_processor.filter(kernel_image)
        else:
            filtered_kernel = kernel_image
        cv.imwrite(str(new_kernel_path), filtered_kernel)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path))

    def _generate_unique_file_path(self, 
                                   directory: Path, 
                                   filename: str) -> Path:
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
   
    def save_filter(self) -> None:
        '''
        Сохранение текущего состояния фильтров
        Переносит изображение из буфера в список
        к изображениям в списке не применяются фильтры
        '''
        for img_obj in self.images:
            img_obj.save_filter()
        self.amount_of_blurred += 1

    def load_filter(self, index: int) -> None:
        '''
        Загрузка состояния фильтров.
        Достает изображение из списка в буфер для изменения
        Аргументы:
            -index: индекс доставаемого изображения
        '''
        for img_obj in self.images:
            img_obj.load(index)
        self.amount_of_blurred -= 1

    def len_blur(self) -> int:
        '''
        Количество вариантов размытия
        '''
        return self.amount_of_blurred
    
    def process(self, 
                algorithm_processor: base.DeconvolutionAlgorithm, 
                metadata: bool = False, 
                unique_path: bool = True) -> None:
        '''
        Восстановление всех изображений
        
        Аргументы:
            -algorithm_processor: метод восстановления изображения
            -metadata: сохранять метаданные или нет
        '''
        alg_name = algorithm_processor.get_name()
        
        for img_obj in self.images:
            self._process_single_image(img_obj, 
                                       algorithm_processor, 
                                       alg_name, 
                                       metadata = metadata, 
                                       unique_path=unique_path)
        
    def _process_single_image(self, 
                              img_obj: utils.Image, 
                              algorithm_processor: base.DeconvolutionAlgorithm, 
                              alg_name: str, 
                              metadata: bool = False, 
                              unique_path: bool = True) -> None:
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
                               img_obj: utils.Image, 
                               alg_name: str, 
                               unique_path: bool = True) -> Tuple[Path, Path]:
        '''
        Генерация уникальных путей для сохранения результатов
        '''
        if img_obj.get_blurred():
            base_path = Path(img_obj.get_blurred())
        else:
            base_path = Path(img_obj.get_original())
        
        base_name = base_path.stem
        
        if unique_path:
            restored_path = self._generate_unique_file_path(
                self.folder_path_restored, 
                f"{base_name}_{alg_name}{base_path.suffix}"
            )
            
            kernel_path = self._generate_unique_file_path(
                self.folder_path_restored,
                f"{base_name}_{alg_name}_kernel{base_path.suffix}"
            )
        else:
            restored_path =self.folder_path_restored / f"{base_name}_{alg_name}{base_path.suffix}"

            kernel_path = self.kernel_dir / f"{base_name}_{alg_name}_kernel{base_path.suffix}"
        
        return restored_path, kernel_path
    
    def _calculate_and_save_metrics(self, 
                                    img_obj: utils.Image, 
                                    original_image: np.ndarray, 
                                    restored_image: np.ndarray, 
                                    restored_path: Path, 
                                    kernel_path: Path, 
                                    alg_name: str, 
                                    processor: base.DeconvolutionAlgorithm, 
                                    metadata: bool = False) -> None:
        '''
        Расчет метрик и обновление данных изображения
        '''
        original_image = self._prepare_image_for_metric(original_image)
        restored_image = self._prepare_image_for_metric(restored_image)
        
        psnr_val, ssim_val = self._calculate_metrics(original_image, restored_image, data_range=1.0)
        
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

    def _prepare_image_for_metric(self, image: np.ndarray) -> np.ndarray:
        """
        Подготовка изображения для расчета метрик с нормализацией
        Нормируем в диапазон [0, 1] если нужно
        """
        image = np.array(image.copy(), dtype=np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        if len(image.shape) == 3:
            image = image.mean(axis=2)
        
        image = np.clip(image, 0.0, 1.0)

        return image

    def full_process(self, 
                     filters: list, 
                     methods: list, 
                     size: float = 0.75, 
                     kernel_intencity_scale: float = 10.0) -> None:
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
    
    def _process_image_pipeline(self, 
                                img_obj: utils.Image, 
                                filters: list, 
                                methods: list, 
                                data_dict : dict, 
                                analysis_dicts: dict) -> None:
        '''
        Полный пайплайн обработки для одного изображения
        '''
        self._apply_filter_chains(img_obj, filters)
        
        self._apply_restoration_and_collect_data(img_obj, 
                                                 methods, 
                                                 data_dict, 
                                                 analysis_dicts)   
    
    def _apply_filter_chains(self, 
                             img_obj: utils.Image, 
                             filters: list) -> None:
        '''
        Применение цепочек фильтров к изображению
        '''
        for filter_chain in filters:
            self._apply_single_filter_chain(img_obj, 
                                            filter_chain)
            img_obj.save_filter()

    def _apply_single_filter_chain(self, 
                                   img_obj: utils.Image, 
                                   filter_chain: list) -> None:
        '''
        Применение одной цепочки фильтров
        '''
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
                                  img_obj: utils.Image) -> np.ndarray:
        '''
        Последовательное применение фильтров цепочки
        '''
        filtered_image = image.copy()
        for filter_processor in filter_chain:
            filtered_image = filter_processor.filter(filtered_image)
            img_obj.add_to_current_filter(filter_processor.discription())
        return filtered_image

    def _generate_blurred_image_path(self, img_obj: utils.Image) -> Path:
        '''
        Генерация пути для размытого изображения
        '''
        if img_obj.get_blurred() is None:
            original_filename = Path(img_obj.get_original()).name
            return self._generate_unique_file_path(self.folder_path_blurred, original_filename)
        else:
            original_filename = Path(img_obj.get_blurred()).name
            return self.folder_path_blurred / original_filename
    
    def _update_blurred_image_data(self, 
                                   img_obj: utils.Image, 
                                   original_image: np.ndarray, 
                                   filtered_image: np.ndarray, 
                                   new_path: str) -> None:
        '''
        Обновление данных размытого изображения
        '''

        psnr_val, ssim_val = self._calculate_metrics(original_image, filtered_image)
        
        img_obj.add_blurred_PSNR(psnr_val, str(new_path))
        img_obj.add_blurred_SSIM(ssim_val, str(new_path))
        img_obj.set_blurred(str(new_path))
    
    def _process_kernel_chain(self, 
                              img_obj: utils.Image, 
                              filter_chain: list, 
                              new_path: str) -> None:
        '''
        Обработка ядра для цепочки фильтров
        '''
        kernels = img_obj.get_original_kernels()
        kernel_path = kernels.get(str(new_path))
        
        if kernel_path is None:
            kernel_image = self._create_delta_kernel(img_obj)
            new_kernel_path = self.folder_path_blurred / f"kernel_{Path(new_path).name}"
        else:
            kernel_image = self._imread(str(kernel_path), img_obj.get_color())
            new_kernel_path = Path(kernel_path)
        
        for filter_processor in filter_chain:
            if filter_processor.get_type() != 'noise':
                kernel_image = filter_processor.filter(kernel_image)
            else:
                kernel_image = kernel_image
        
        cv.imwrite(str(new_kernel_path), kernel_image)
        img_obj.add_original_kernel(str(new_kernel_path), str(new_path)) 
        
    def _create_delta_kernel(self, img_obj: utils.Image) -> np.ndarray:
        '''Создание дельта-функции (единичного импульса)'''
        kernel_image = img_obj.get_original_image().copy()
        kernel_image *= 0
        h, w = kernel_image.shape[:2]
        kernel_image[h//2, w//2] = 255
        return kernel_image   
    
    def _apply_restoration_and_collect_data(self, 
                                            img_obj: utils.Image, 
                                            methods: list, 
                                            data_dict: dict, 
                                            analysis_dicts: dict) -> None:
        '''Восстановление изображений и сбор данных'''
        for blurred_path in img_obj.get_blurred_array():
            blurred_image = self._imread(str(blurred_path), img_obj.get_color())
            
            for algorithm in methods:
                self._restore_single_image(img_obj, blurred_path, blurred_image, algorithm, data_dict)
            
            self._collect_analysis_data(img_obj, blurred_path, analysis_dicts)
            
    def _restore_single_image(self, 
                              img_obj: utils.Image, 
                              blurred_path: Path, 
                              blurred_image: np.ndarray, 
                              algorithm: base.DeconvolutionAlgorithm, 
                              data_dict: dict) -> None:
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

            restored_image = self._prepare_image_for_metric(restored_image)
            original_image = self._prepare_image_for_metric(original_image)
            psnr_val, ssim_val = self._calculate_metrics(original_image, restored_image, data_range=1.0)
            
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
  
    def _calculate_metrics(self, 
                           original_image: np.ndarray, 
                           restored_image: np.ndarray,
                           data_range: Optional[float] = None) -> tuple[float, float]:
        '''Расчет метрик'''
        try:
            psnr_val = metrics.PSNR(original_image, restored_image)
        except:
            psnr_val = math.nan
        try:
            ssim_val = metrics.SSIM(original_image, restored_image, data_range=data_range)
        except:
            ssim_val = math.nan 
        return psnr_val, ssim_val 
    
    def _update_restoration_data(self, 
                                 img_obj: utils.Image, 
                                 blurred_path: Path, 
                                 alg_name: str, 
                                 restored_path: Path, 
                                 kernel_path: Path, 
                                 psnr_val: float, 
                                 ssim_val: float) -> None:
        '''
        Обновление данных восстановления
        '''
        img_obj.add_PSNR(psnr_val, blurred_path, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_path, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(restored_path, blurred_path, alg_name)
        img_obj.add_kernel(kernel_path, blurred_path, alg_name)

    def _collect_algorithm_data(self, 
                                data_dict: dict, 
                                alg_name: str, 
                                algorithm: base.DeconvolutionAlgorithm, 
                                img_obj: utils.Image, 
                                blurred_path: Path, 
                                restored_path: Path, 
                                kernel_path: Path, 
                                psnr_val: float, 
                                ssim_val: float, 
                                restored_image: np.ndarray, 
                                blurred_image: np.ndarray) -> None:
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
 
    def _collect_analysis_data(self, 
                               img_obj: utils.Image, 
                               blurred_path: Path, 
                               analysis_dicts: dict) -> None:
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
    
    def _finalize_processing(self, 
                             data_dict: dict, 
                             analysis_dicts: dict, 
                             size: float, 
                             kernel_intencity_scale: float) -> None:
        '''
        Финальная обработка и визуализация результатов
        '''
        self.show(size, kernel_intencity_scale)
        
        for alg_name, alg_data in data_dict.items():
            df_data = pd.DataFrame(alg_data)
            display(df_data)
            df_data.to_csv(self.data_path / f"{alg_name}.csv", index=False)
        
        self.pareto()

    def process_hyperparameter_optimization(self, *args, **kwargs) -> Any:
        return self.optimizer.execute(*args, **kwargs)
    
    def pareto(self) -> Any:
        return self.analyzer.execute()
    


def merge(frame1: Processing, frame2: Processing)->Processing:
         """
        объединяет массивы обработанных изображений
         """
         frame_res = Processing(images_folder=frame1.folder_path,
                                blurred_folder=frame1.folder_path_blurred,
                                restored_folder=frame1.folder_path_restored,
                                data_path=frame1.data_path,
                                color=frame1.color,
                                kernel_dir=frame1.kernel_dir,
                                dataset_path = frame1.dataset_path)
         frame_res.images = np.append(frame1.images.copy(),frame2.images.copy())
         return frame_res

def show_from_table(table_path: Path, alg_name: str, window_scale: float = 1.0) -> None:
    """
    выводит сетку из смазанных и восстановленных изображений, их ядер
    table_path - путь к .csv файлу. таблица, по которой строить сетку
    alg_name - имя алгоритма, для которого строить сетку
    window_scale -  регулирует размер окна
    """
    df = pd.read_csv(table_path)
    h, w = df.shape
    fig, axes = plt.subplots(4, h, figsize=(5 * h * max(window_scale,0.1), 
                                            18*max(window_scale,0.1)))
    axes = np.atleast_2d(axes)
    for idx, line in enumerate(df.iloc):
        paths = {
            "Original": line['blurred'],
            "Restored": line[alg_name],
            "Estimated Kernel": line[f'kernel_{alg_name}'],
            "Ground Truth Kernel": line['kernel blur']
        }
        images = {}
        
        for title, path in paths.items():
            if os.path.exists(path):
                images[title] = mpimg.imread(path)
            else:
                print(f" Файл не найден: {path}")
                images[title] = np.zeros((100, 100), dtype=np.uint8) # Черный квадрат-заглушка
        axes[0, idx].imshow(images["Original"], cmap='gray')

        axes[1, idx].imshow(images["Restored"], cmap='gray')

        axes[2, idx].imshow(images["Estimated Kernel"], cmap='gray')

        axes[3, idx].imshow(images["Ground Truth Kernel"], cmap='gray')
        for row in range(4):
            axes[row, idx].axis('off')
    fig.suptitle(f"Сравнение результатов восстановления изображений ({alg_name})", fontsize=20, y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
        

        


