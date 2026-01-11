"""
Основной модуль фреймворка обработки изображений.

Содержит класс Processing для управления конвейером обработки изображений:
загрузка, фильтрация, восстановление, анализ метрик.

Авторы: Юров П.И., Беззаборов А.А., Куропатов К.Л.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path
from typing import Optional, Any

from processing.utils import Image
import filters.base as filters
import algorithms.base as base

from processing.extensions.hyperparameter_optimization import HyperparameterOptimizer
from processing.extensions.pareto_analysis import ParetoFrontAnalyzer
from processing.reader import ModuleReader
from processing.display import ModuleDisplay
from processing.preprocessing import ModulePreprocessing
from processing.tables import ModuleData
from processing.clear import ModuleClear
from processing.applyfilter import ModuleFilter
from processing.restore import ModuleProcess
from processing.restorepipeline import ModuleProcessPipeline


class Processing:

    """
    Фреймворк для обработки изображений.
    
    Предоставляет полный конвейер для работы с изображениями:
    загрузка, применение фильтров, восстановление, анализ метрик.

    Атрибуты
    --------
    color : bool
        Тип загрузки изображений (цветное/черно-белое).
    folder_path : Path
        Директория с исходными изображениями.
    folder_path_blurred : Path
        Директория со смазанными изображениями.
    folder_path_restored : Path
        Директория с восстановленными изображениями.
    data_path : Path
        Директория для сохранения анализа данных.
    images : np.ndarray
        Массив связей изображений.
    kernel_dir : Path
        Директория для ядер.
    dataset_path : Path
        Директория для метаданных датасетов.
    preprocess_dir: str
        Директория для обработанных смазанных изображений.
    """
    
    def __init__(self, 
                 images_folder: str = 'images',
                 blurred_folder: str = 'blurred', 
                 restored_folder: str = 'restored', 
                 data_path: str = 'data', 
                 color: bool = False, 
                 kernel_dir: str = 'kernels',
                 preprocess_dir: str = 'preprocess', 
                 dataset_path: str = 'dataset') -> None:
        """
        Инициализация фреймворка.

        Параметры
        ---------
        images_folder : str
            Директория с исходными изображениями.
        blurred_folder : str
            Директория со смазанными изображениями.
        restored_folder : str
            Директория с восстановленными изображениями.
        data_path : str
            Директория для сохранения анализа данных.
        color : bool
            Тип загрузки изображений (True - цветное, False - ч/б).
        kernel_dir : str
            Директория для ядер.
        dataset_path : str
            Директория для метаданных датасетов.
        preprocess_dir: str
            Директория для обработанных смазанных изображений.
        """
        self.color = color
        self.folder_path = Path(images_folder)
        self.folder_path_blurred = Path(blurred_folder)
        self.folder_path_restored = Path(restored_folder)
        self.data_path = Path(data_path)
        self.images = np.array([])
        self.amount_of_blurred = 1

        self.optimizer = HyperparameterOptimizer(self)
        self.analyzer = ParetoFrontAnalyzer(self)
        self.reader = ModuleReader(self)
        self.display = ModuleDisplay(self)
        self.histogram = ModulePreprocessing(self)
        self.tables = ModuleData(self)
        self.clear = ModuleClear(self)
        self.apply_filter = ModuleFilter(self)
        self.module_process = ModuleProcess(self)
        self.process_pipeline = ModuleProcessPipeline(self)
        #разные, если дочерний будет переопределять методы из родительского.

        self.kernel_dir = Path(kernel_dir)
        self.dataset_path = Path(dataset_path)
        self.preprocess_dir = Path(preprocess_dir)
        
        for folder in [self.folder_path, 
                       self.folder_path_blurred, 
                       self.folder_path_restored, 
                       self.data_path, 
                       self.kernel_dir, 
                       self.dataset_path, 
                       self.preprocess_dir]:
            folder.mkdir(parents=True, exist_ok=True)

    def changescale(self, color: bool) -> None:
        """
        Изменение способа загрузки изображений.
        
        Параметры
        ---------
        color : bool
            True - цветное, False - черно-белое.
        """
        self.color = color

    def read_all(self) -> None:
        """Загрузка всех изображений из директории."""
        self.reader.read_all()
    
    def read_one(self, path: Path) -> None:
        """Загрузка одного изображения."""
        self.reader.read_one(path)

    def show(self, 
             size: float = 1.0, 
             kernel_intencity_scale: float = 1.0, 
             kernel_size: float = 1.0) -> None:
        """Вывод всех изображений: оригинал, размытые, восстановленные + метрики."""
        self.display.show(size, kernel_intencity_scale, kernel_size)
    
    def histogram_equalization(self, view_histogram: bool = False) -> None:
        """Выполняет выравнивание гистограмм."""
        self.histogram.histogram_equalization(view_histogram)

    def histogram_equalization_CLAHE(self, 
                                     view_histogram: bool = False, 
                                     clip_limit: float = 0.01) -> None:
        """Выполняет адаптивное выравнивание гистограмм с ограничением контрастности."""
        self.histogram.histogram_equalization_CLAHE(view_histogram, clip_limit)
    
    def inverse_histogram_equalization(self, view_histogram: bool = False) -> None:
        """Обращает выравнивание гистограмм."""
        self.histogram.inverse_histogram_equalization(view_histogram)
    
    def get_table(self,
                   table_path: Path, 
                   display_table: bool = False) -> None:
        """Получение метрик в структурированном виде."""
        self.tables.get_table(table_path, display_table)

    def clear_input(self) -> None:
        """Убирает привязку ко всем загруженным изображениям."""
        self.clear.clear_input()
    
    def reset(self) -> None:
        """Сброс состояний всех изображений до исходного."""
        self.clear.reset()
    
    def clear_output(self) -> None:
        """Удаление всех сгенерированных файлов."""
        self.clear.clear_output()
   
    def clear_output_directory(self, warning: str = 'IT WILL DELETE EVERYTHING!') -> None:
        """Полная очистка выходных директорий."""
        self.clear.clear_output_directory(warning)  
    
    def clear_restored(self) -> None:
        """Удаляет восстановленные изображения из каждой связи."""
        self.clear.clear_restored()

    def unbind_restored(self) -> None:
        """Разрывает связь, убирая все восстановленные."""
        self.clear.unbind_restored()

    def clear_all(self) -> None:
        """Полная очистка: файлы + состояния + загруженные изображения."""
        self.clear.clear_all()
    
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
        return self.reader.bind(original_image_path, 
                                blurred_image_path, 
                                original_kernel_path, 
                                filter_description, 
                                color)

    def save_bind_state(self, file_path: Optional[Path] = None) -> None:
        """Сохраняет состояние связей в JSON файл."""
        self.reader.save_bind_state(file_path)

    def load_bind_state(self, bind_path: Path) -> None:
        """Загружает состояние связей из JSON файла."""
        self.reader.load_bind_state(bind_path)

    def custom_filter(self, 
                      kernel_image_path: Path, 
                      kernel_npy_path: Path) -> None:
        """Применение созданного фильтра ко всем оригинальным изображениям."""
        self.apply_filter.custom_filter(kernel_image_path, kernel_npy_path)

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
        self.display.show_line(window_scale, fontsize)
    
    def filter(self, filter_processor: filters.FilterBase) -> None:
        """Применение фильтра ко всем изображениям."""
        self.apply_filter.filter(filter_processor)
   
    def save_filter(self) -> None:
        """Сохранение текущего состояния фильтров в список."""
        for img_obj in self.images:
            img_obj.save_filter()
        self.amount_of_blurred += 1

    def load_filter(self, index: int) -> None:
        """
        Загрузка состояния фильтров из списка.
        
        Параметры
        ---------
        index : int
            Индекс доставаемого изображения.
        """
        for img_obj in self.images:
            img_obj.load(index)
        self.amount_of_blurred -= 1

    def len_blur(self) -> int:
        """Количество вариантов размытия."""
        return self.amount_of_blurred
    
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
        self.module_process.process(algorithm_processor, 
                                   metadata, 
                                   unique_path)
    
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
        self.process_pipeline.full_process(filters, 
                                           methods, 
                                           size, 
                                           kernel_intencity_scale)

    def process_hyperparameter_optimization(self, *args, **kwargs) -> Any:
        """Запуск оптимизации гиперпараметров."""
        return self.optimizer.execute(*args, **kwargs)
    
    def pareto(self) -> Any:
        """Построение фронта Парето."""
        return self.analyzer.execute()
    
    


def merge(frame1: Processing, frame2: Processing)->Processing:
        """
        Объединяет массивы обработанных изображений.
        
        Параметры
        ---------
        frame1 : Processing
            Первый фреймворк.
        frame2 : Processing
            Второй фреймворк.
            
        Возвращает
        ----------
        Processing
            Объединённый фреймворк.
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
    Выводит сетку из смазанных и восстановленных изображений.
    
    Параметры
    ---------
    table_path : str
        Путь к .csv файлу.
    alg_name : str
        Имя алгоритма для визуализации.
    window_scale : float
        Регулирует размер окна.
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
        

        


