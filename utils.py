import cv2 as cv
import numpy as np
from typing import Callable, Optional, Tuple, Dict


class Image:
    """
    Класс для управления путями к изображениям и метриками качества.

    Атрибуты:
        original_path (str): Путь к исходному изображению
        blurred_path (Optional[str]): Путь к размытому изображению
        blurred_array (np.ndarray): Несколько вариантов одной размытого изображения
        restored_paths (Dict[str, str]): Список путей к восстановленным изображениям
        is_color (bool): Цветное или черно-белое изображение
        psnr (Dict[str, float]): Значения PSNR для восстановленных изображений
        ssim (Dict[str, float]): Значения SSIM для восстановленных изображений
        algorithm (np.ndarray): Названия использованных алгоритмов восстановления
        filters (Dict[str, str]): Названия использованных фильров зашумления
        curent_filter (str): Текущий фильтр
        kernel_paths (Dict[str, str]): Путь к полученным psf
        original_kernels_path (Dict[str, str]): Путь к изначальным psf
        current_filter (Optional(str)): Текущий фильтр
        blurred_psnr (Dict[str, float]): Значение PSNR для смазанных изображений
        blurred_ssim (Dict[str, float]): Значение SSIM для смазанных изображений
    """

    def __init__(self, original_path: str, is_color: bool) -> None:
        """
        Инициализация с путем к исходному изображению.

        Аргументы:
            original_path (str): Путь к исходному изображению
            is_color (bool): Флаг цветного изображения
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

    # def set_mapping_data(self, mapping_data: ) -> None:
    #     self.mapping_data = mapping_data

    # def get_mapping_data(self):
    #     return self.mapping_data

    def set_blurred_PSNR(self, psnr: Dict[str, float])  -> None:
        """Переопределяет ..."""
        self.blurred_psnr = psnr

    def get_blurred_PSNR(self) -> Dict[str, float]:
        return self.blurred_psnr.copy()

    def add_blurred_PSNR(self, psnr: float, blurred_path: str) -> None:
        self.blurred_psnr[blurred_path] = psnr

    def set_blurred_SSIM(self, ssim: Dict[str, float]) -> None:
        self.blurred_ssim = ssim

    def get_blurred_SSIM(self) -> Dict[str, float]:
        return self.blurred_ssim.copy()

    def add_blurred_SSIM(self, ssim: float, blurred_path: str) -> None:
        self.blurred_ssim[blurred_path] = ssim
    
    def get_original_kernels(self) -> Dict[str, str]:
        return self.original_kernels_path.copy()

    def set_original_kernels(self, kernels: Dict[str, str]) -> None:
        self.original_kernels_path = kernels

    def add_original_kernel(self, kernel: str, blur_path: str) -> None:
        self.original_kernels_path[blur_path] = kernel

    def get_kernels(self) -> Dict[str, str]:
        return self.kernel_paths.copy()

    def set_kernels(self, kernels: Dict[str, str]) -> None:
        self.kernel_paths = kernels

    def add_kernel(self, kernel, blur_path, alg_path) -> None:
        self.kernel_paths[(blur_path, alg_path)] = kernel

    def save_filter(self) -> None:

        if self.blurred_path is not None:
            self.filters[self.blurred_path] = self.current_filter
            self.blurred_array = np.append(self.blurred_array, self.blurred_path)
        
        self.current_filter = None
        self.blurred_path = None

    def load(self, index: int) -> None:
        if index >= self.get_len_filter():
            # raise(Exception("index out of bounds"))
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
        if len(self.filters) != len(self.blurred_array):
            raise Exception(
                "filters and blurred images have different counts "
                f"({len(self.filters)} vs {len(self.blurred_array)})"
            )
        return len(self.filters)

    def get_len_algorithms(self) -> int:
        return len(self.algorithm)  # +1 current

    def get_blurred_array(self) -> np.ndarray:
        return self.blurred_array.copy()

    def set_blurred_array(self, array: np.ndarray) -> None:
        self.blurred_array = array

    def get_filters(self) -> Dict[str, str]:
        return self.filters.copy()

    def set_filters(self, filters: Dict[str, str]) -> None:
        self.filters = filters

    def set_current_filter(self, filter_str: str) -> None:
        self.current_filter = filter_str

    def get_current_filter(self) -> str:
        return self.current_filter

    def add_to_current_filter(self, filter_str: str) -> None:
        if self.current_filter is None:
            self.current_filter = filter_str
        else:
            self.current_filter = f"{self.current_filter}{filter_str}"

    def set_original(self, original_path: str) -> None:
        self.original_path = original_path

    def set_blurred(self, blurred_path: Optional[str]) -> None:
        self.blurred_path = blurred_path

    def set_restored(self, restored_paths: Dict[str, str]) -> None:
        self.restored_paths = restored_paths

    def add_restored(self, restored_paths: str, blur_path: str, alg_name: str) -> None:
        self.restored_paths[(blur_path, alg_name)] = restored_paths

    def get_original(self) -> str:
        return self.original_path

    def get_blurred(self) -> str:
        return self.blurred_path

    def get_restored(self) -> str:
        return (
            self.restored_paths.copy()
        )  

    def get_color(self) -> bool:
        return self.is_color

    def set_PSNR(self, psnr) -> None:
        self.psnr = psnr

    def set_SSIM(self, ssim) -> None:
        self.ssim = ssim

    def add_PSNR(self, psnr: float, blur_path, alg_name) -> None:
        self.psnr[(blur_path, alg_name)] = psnr

    def add_SSIM(self, ssim: float, blur_path, alg_name) -> None:
        self.ssim[(blur_path, alg_name)] = ssim

    def get_PSNR(self) -> float:
        return (
            self.psnr.copy()
        )

    def get_SSIM(self) -> float:
        return (
            self.ssim.copy()
        ) 

    def set_algorithm(
        self, algorithm
    ) -> None:  # название алгоритма, которым обрабатывалось
        self.algorithm = algorithm

    def add_algorithm(
        self, algorithm
    ) -> None:  # название алгоритма, которым обрабатывалось
        self.algorithm = np.array(
            list(set(np.append(self.algorithm, algorithm)))
        )  # теперь без повторов

    def get_algorithm(self) -> str:  # название алгоритма, которым обрабатывалось
        return self.algorithm.copy()

    def get_original_image(self) -> np.ndarray:
        return cv.imread(
            self.original_path,
            cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE,
        )

    def get_blurred_image(self) -> Optional[np.ndarray]:
        if self.blurred_path is None:
            return None
        return cv.imread(
            self.blurred_path, cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
        )

    def get_all_blurred_images(self):
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

    def get_all_filters(self):
        res = self.filters.copy()
        if self.current_filter is not None:
            res = np.append(res, self.current_filter)
        return res
        
