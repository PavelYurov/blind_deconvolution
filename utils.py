"""
Вспомогательные классы для управления изображениями и метриками.

Автор: Юров П.И.
"""

import cv2 as cv
import numpy as np
from typing import Optional, Tuple, Dict


class Image:
    """
    Класс для управления путями к изображениям и метриками качества.

    В последующем называется "связью".

    Атрибуты
    --------
    original_path : str
        Путь к исходному изображению.
    blurred_path : Optional[str]
        Путь к размытому изображению (буфер).
    blurred_array : np.ndarray
        Несколько вариантов одного размытого изображения.
    restored_paths : Dict[Tuple[str, str], str]
        Список путей к восстановленным изображениям.
    is_color : bool
        Цветное или черно-белое изображение.
    psnr : Dict[Tuple[str, str], float]
        Значения PSNR для восстановленных изображений.
    ssim : Dict[Tuple[str, str], float]
        Значения SSIM для восстановленных изображений.
    algorithm : np.ndarray
        Названия использованных алгоритмов восстановления.
    filters : Dict[str, str]
        Названия использованных фильтров зашумления.
    kernel_paths : Dict[Tuple[str, str], str]
        Путь к полученным psf.
    original_kernels_path : Dict[str, str]
        Путь к изначальным psf.
    current_filter : Optional[str]
        Текущий фильтр.
    blurred_psnr : Dict[str, float]
        Значение PSNR для смазанных изображений.
    blurred_ssim : Dict[str, float]
        Значение SSIM для смазанных изображений.
    preprocessed_blurred_path : Dict[str, str]
        Изображения после выравнивания гистограмм и/или денойзинга.

    Notes
    -----
    Структура связи:
    
    Основное изображение original_path - то, от чего начинается связь.

    От него идет блок смазанных изображений:
        - blurred_path - буфер смазанного изображения
        - blurred_array - список всех смазанных изображений (кроме буфера)
        - original_kernels_path[путь] - список всех ядер
        - blurred_psnr, blurred_ssim[путь] - метрики для смазанных изображений
        - current_filter, filters - буфер и список фильтров

    От каждого смазанного идет блок восстановленных изображений:
        - restored_paths[путь, алгоритм] - пути к восстановленным изображениям
        - algorithm - все алгоритмы, которые применялись к связи
        - psnr, ssim[путь, алгоритм] - метрики восстановленного изображения
        - kernel_paths[путь, алгоритм] - восстановленные psf
    
    Класс необходим для того, чтобы автоматически определять последовательность и порядок картинок,
    к которым применялись фильтры и методы восстановления
    """

    def __init__(self, original_path: str, is_color: bool) -> None:
        """
        Инициализация с путем к исходному изображению.

        Параметры
        ---------
        original_path : str
            Путь к исходному изображению.
        is_color : bool
            Флаг цветного изображения.
        """
        self.original_path = original_path
        self.blurred_path = None

        self.restored_paths = {}
        self.kernel_paths = {}
        self.original_kernels_path = {}
        self.is_color = is_color
        self.psnr = {}
        self.ssim = {}
        self.algorithm = np.array([])

        self.filters = {}
        self.blurred_array = np.array([])
        self.current_filter = None
        self.blurred_psnr = {}
        self.blurred_ssim = {}
        self.preprocessed_blurred_path = {}
    
    def set_preprocessed_blurred_path(self, preprocessed_blurred_path: Dict[str, str]) -> None:
        """Переопределяет изображение для предобработки."""
        self.preprocessed_blurred_path = preprocessed_blurred_path

    def add_preprocessed_blurred_path(self, blurred_path: str, preprocessed_blurred_path: str) -> None:
        """Добавляет изображение для предобработки."""
        self.preprocessed_blurred_path[blurred_path] = preprocessed_blurred_path

    def get_preprocessed_blurred_path(self) -> Dict[str, str]:
        """Возвращает путь до изображения для предобработки."""
        return self.preprocessed_blurred_path

    def set_blurred_PSNR(self, psnr: Dict[str, float])  -> None:
        """Полностью переопределяет PSNR смазанных изображений."""
        self.blurred_psnr = psnr

    def get_blurred_PSNR(self) -> Dict[str, float]:
        """Возвращает список значений PSNR для смазанных изображений."""
        return self.blurred_psnr.copy()

    def add_blurred_PSNR(self, psnr: float, blurred_path: str) -> None:
        """Добавляет/переопределяет значение psnr для конкретного смазанного изображения."""
        self.blurred_psnr[blurred_path] = psnr

    def set_blurred_SSIM(self, ssim: Dict[str, float]) -> None:
        """Полностью переопределяет SSIM смазанных изображений."""
        self.blurred_ssim = ssim

    def get_blurred_SSIM(self) -> Dict[str, float]:
        """Возвращает список значений SSIM смазанных изображений."""
        return self.blurred_ssim.copy()

    def add_blurred_SSIM(self, ssim: float, blurred_path: str) -> None:
        """Добавляет/переопределяет значений SSIM для конкретного смазанного изображения."""
        self.blurred_ssim[blurred_path] = ssim
    
    def get_original_kernels(self) -> Dict[str, str]:
        """Возвращает список путей к ядрам смазанных изображений."""
        return self.original_kernels_path.copy()

    def set_original_kernels(self, kernels: Dict[str, str]) -> None:
        """Полностью переопределяет пути к ядрам смазанных изображений."""
        self.original_kernels_path = kernels

    def add_original_kernel(self, kernel: str, blur_path: str) -> None:
        """Добавляет/переопределяет путь к ядру конкретного смазанного изображения."""
        self.original_kernels_path[blur_path] = kernel

    def get_kernels(self) -> Dict[Tuple[str, str], str]:
        """Возвращает список путей к ядрам восстановленных изображений."""
        return self.kernel_paths.copy()

    def set_kernels(self, kernels: Dict[Tuple[str, str], str]) -> None:
        """Полностью переопределяет пути к ядрам восстановленных изображений."""
        self.kernel_paths = kernels

    def add_kernel(self, 
                   kernel: str, 
                   blur_path: str, 
                   alg_path: str) -> None:
        """Добавляет/переопределяет путь к ядру конкретного метода для смазанного изображения."""
        self.kernel_paths[(blur_path, alg_path)] = kernel

    def save_filter(self) -> None:
        """Сохраняет буфер смазанного изображения в общий список."""
        if self.blurred_path is not None:
            self.filters[self.blurred_path] = self.current_filter
            self.blurred_array = np.append(self.blurred_array, self.blurred_path)
        
        self.current_filter = None
        self.blurred_path = None

    def load(self, index: int) -> None:
        """Загружает в буфер из общего списка смазанное изображение."""
        if index >= self.get_len_filter():
            if self.current is not None:
                print("index out of bounds, load empty")
                self.current_filter = None
                self.blurred_path = None
                return
        if self.current_filter is not None:
            print(
                "current blurred image is not empty; "
                "save it if you do not want to lose it"
            )

        self.blurred_path = self.blurred_array[index]
        if self.blurred_path is not None:
            self.current_filter = self.filters[self.blurred_path]
        else:
            self.current_filter = None
        self.blurred_array = np.delete(self.blurred_array, index, 0)

    def get_len_filter(self) -> int:
        """Возращает длину общего списка смазанных изображений (не учитывает буфер)."""
        if len(self.filters) != len(self.blurred_array):
            raise Exception(
                "filters and blurred images have different counts "
                f"({len(self.filters)} vs {len(self.blurred_array)})"
            )
        return len(self.filters)

    def get_len_algorithms(self) -> int:
        """Возвращает количество примененных алгоритмов."""
        return len(self.algorithm)  # +1 current

    def get_blurred_array(self) -> np.ndarray:
        """Возвращает список путей к смазанным изображениям (кроме буфера)."""
        return self.blurred_array.copy()

    def set_blurred_array(self, array: np.ndarray) -> None:
        """Полностью переопределяет пути к смазанным изображениям (кроме буфера)."""
        self.blurred_array = array

    def get_filters(self) -> Dict[str, str]:
        """Возвращает список фильтров."""
        return self.filters.copy()

    def set_filters(self, filters: Dict[str, str]) -> None:
        """Полностью переопределяет список фильтров."""
        self.filters = filters

    def set_current_filter(self, filter_str: str) -> None:
        """Полностью переопределяет буфер фильров."""
        self.current_filter = filter_str

    def get_current_filter(self) -> str:
        """Возвращает буфер фильтров."""
        return self.current_filter

    def add_to_current_filter(self, filter_str: str) -> None:
        """Добавляет фильтр к списку в буфере фильтров."""
        if self.current_filter is None:
            self.current_filter = filter_str
        else:
            self.current_filter = f"{self.current_filter}{filter_str}"

    def set_original(self, original_path: str) -> None:
        """Полностью переопределяет оригинальное изображение."""
        self.original_path = original_path

    def set_blurred(self, blurred_path: Optional[str]) -> None:
        """Полностью переопределяет буфер смазанных изображений."""
        self.blurred_path = blurred_path

    def set_restored(self, restored_paths: Dict[Tuple[str, str], str]) -> None:
        """Полностью переопределяет восстановленные изображения."""
        self.restored_paths = restored_paths

    def add_restored(self, 
                     restored_paths: str, 
                     blur_path: str, 
                     alg_name: str) -> None:
        """Добавляет/переопределяет путь восстановленного изображения."""
        self.restored_paths[(blur_path, alg_name)] = restored_paths

    def get_original(self) -> str:
        """Возвращает путь к оригинальному изображению."""
        return self.original_path

    def get_blurred(self) -> str:
        """Возвращает путь к смазанному изображению из буфера."""
        return self.blurred_path

    def get_restored(self) -> str:
        """Возвращает список восстановленных изображений."""
        return self.restored_paths.copy()  

    def get_color(self) -> bool:
        """Возвращает флаг цвета."""
        return self.is_color

    def set_PSNR(self, psnr: Dict[Tuple[str, str], str]) -> None:
        """Полностью переопределяет PSNR восстановленных изображений."""
        self.psnr = psnr

    def set_SSIM(self, ssim: Dict[Tuple[str, str], str]) -> None:
        """Полностью переопределяет SSIM восстановленных изображений."""
        self.ssim = ssim

    def add_PSNR(self, psnr: float, blur_path: str, alg_name: str) -> None:
        """Добавляет/переопределяет PSNR конкретного восстановленного изображения."""
        self.psnr[(blur_path, alg_name)] = psnr

    def add_SSIM(self, ssim: float, blur_path: str, alg_name: str) -> None:
        """Добавляет/переопределяет SSIM конкретного воссстановленного изображения."""
        self.ssim[(blur_path, alg_name)] = ssim

    def get_PSNR(self) -> Dict[Tuple[str, str], str]:
        """Возвращает список PSNR восстановленных изображений."""
        return self.psnr.copy()

    def get_SSIM(self) -> Dict[Tuple[str, str], str]:
        """Возвращает список SSIM восстановленных изображений."""
        return self.ssim.copy() 

    def set_algorithm(self, algorithm: np.ndarray) -> None:
        """Полностью переопределяет список алгоритмов восстановления."""
        self.algorithm = algorithm

    def add_algorithm(self, algorithm: str) -> None:
        """Добавляет алгоритм для восстановления изображения в список."""
        self.algorithm = np.array(
            list(set(np.append(self.algorithm, algorithm)))
        )

    def get_algorithm(self) -> str:
        """Возвращает список алгоритмов, примененных для восстановления."""
        return self.algorithm.copy()

    def get_original_image(self) -> np.ndarray:
        """Возвращает оригинальное изображение как массив."""
        return cv.imread(
            self.original_path,
            cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE,
        )

    def get_blurred_image(self) -> Optional[np.ndarray]:
        """Возвращает смазанное изображение из буфера как массив."""
        if self.blurred_path is None:
            return None
        return cv.imread(
            self.blurred_path, 
            cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
        )

    def get_all_blurred_images(self)->Optional[np.ndarray]:
        """Возвращает все смазанные изображения как массивы."""
        res = [
            cv.imread(path, cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE)
            for path in self.blurred_array
        ]
        if self.blurred_path is not None:
            res = np.append(
                res,
                cv.imread(
                    self.blurred_path,
                    cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE,
                ),
            )
        return res

    def get_all_filters(self)->Optional[np.ndarray]:
        """Возращает список всех фильтров смаза."""
        res = self.filters.copy()
        if self.current_filter is not None:
            res = np.append(res, self.current_filter)
        return res
        
