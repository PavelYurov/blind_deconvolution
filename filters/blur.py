import cv2 as cv
import numpy as np
from typing import Callable, Optional
from .base import FilterBase

from scipy.interpolate import BSpline, make_interp_spline
from mpl_toolkits.mplot3d import Axes3D

class DefocusBlur(FilterBase):
    """
    Фильтр размытия вне фокуса, имитирующий эффект расфокусировки камеры.
    
    Создает 2D размытие в форме колокола с использованием настраиваемой PSF-функции.
    
    Атрибуты:
        psf (Callable): Функция распределения точки (PSF), генерирующая ядро размытия
        param (float): Параметр интенсивности размытия
        kernel_size (Optional[int]): Фиксированный размер ядра размытия (None для автоопределения)
    """

    def __init__(self, 
                 psf: Callable[[np.ndarray, float], np.ndarray],
                 param: float = 5.0, 
                 kernel_size: Optional[int] = None) -> None:
        """
        Инициализация фильтра размытия вне фокуса.
        
        Аргументы:
            psf: Функция, принимающая (radius, param) и возвращающая значения ядра
            param: Параметр контроля интенсивности размытия
            kernel_size: Опциональный фиксированный размер ядра (должен быть нечетным если указан)
        """
        self.psf = psf
        self.param = param
        self.kernel_size = kernel_size
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|defocus_{self.psf.__name__}_{self.param}_{self.kernel_size}"

    def generate_kernel(self) -> np.ndarray:
        """Генерация ядра размытия."""
        size = self.kernel_size or self._calculate_kernel_size()
        y, x = np.ogrid[-size//2:size//2+1, -size//2:size//2+1]
        radius = np.sqrt(x**2 + y**2)

        kernel = self.psf(radius, self.param)
        kernel[kernel < np.finfo(float).eps] = 0
        return kernel / kernel.sum()

    def _calculate_kernel_size(self) -> int:
        """Вычисление оптимального размера ядра на основе параметра размытия."""
        return int(6 * self.param) | 1 

    def filter(self, image: np.ndarray) -> np.ndarray:
        """
        Применение размытия вне фокуса к изображению.
        
        Аргументы:
            image: Входное изображение для размытия
            
        Возвращает:
            Размытое изображение
        """
        return cv.filter2D(image, -1, self.generate_kernel())
    
class MotionBlur(FilterBase):
    """
    Фильтр размытия в движении, имитирующий линейное движение камеры.
    
    Создает одномерное направленное размытие, которое можно повернуть на любой угол.
    
    Атрибуты:
        psf (Callable): PSF-функция для одномерного движения
        param (float): Параметр контроля длины/интенсивности размытия
        angle (float): Направление движения в градусах (0 = горизонтальное)
        kernel_length (Optional[int]): Фиксированная длина ядра размытия
    """
    
    def __init__(self, 
                 psf: Callable[[np.ndarray, float], np.ndarray],
                 param: float = 1.0,
                 angle: float = 0,
                 kernel_length: Optional[int] = None) -> None:
        """
        Инициализация фильтра размытия в движении.
        
        Аргументы:
            psf: Функция (x, param) -> значения ядра, где x - 1D массив координат
            param: Параметр для PSF-функции
            angle: Угол направления размытия (в градусах)
            kernel_length: Длина размытия (нечетное число)
        """
        self.psf = psf
        self.param = param
        self.angle = angle
        self.kernel_length = kernel_length
        super().__init__(param, 'blur')

    def discription(self) -> str:
        return f"|motion_{self.psf.__name__}_{self.param}_{self.angle}_{self.kernel_length}"

    def generate_kernel(self) -> np.ndarray:
        """Генерация ядра размытия в движении"""
        length = self.kernel_length or self._calculate_kernel_length()
        if length % 2 == 0:
            raise ValueError("Длина ядра должна быть нечетной")

        x_coords = np.linspace(-length//2, length//2, length)
        psf_1d = self.psf(x_coords, self.param)
        psf_1d = psf_1d / psf_1d.sum()
        
        kernel = np.zeros((length, length))
        center = length // 2
        kernel[center, :] = psf_1d

        rotation_matrix = cv.getRotationMatrix2D((center, center), self.angle, 1)
        rotated_kernel = cv.warpAffine(kernel, rotation_matrix, (length, length))
        
        rotated_kernel[rotated_kernel < np.finfo(float).eps] = 0
        return rotated_kernel / rotated_kernel.sum()

    def _calculate_kernel_length(self) -> int:
        """Вычисление длины ядра на основе параметра размытия"""
        return max(int(4 * self.param) | 1, 3) 

    def filter(self, image: np.ndarray) -> np.ndarray:
        """Применение размытия в движении к изображению."""
        return cv.filter2D(image, -1, self.generate_kernel())
    

class BSpline_blur(FilterBase):

    def __init__(self,shape_points, intensity_points, output_size=(15, 15), shape_degree=3, intensity_degree=2, n_samples=1000):
        """
        Создает PSF используя два B-spline: для формы и для интенсивности.
        Применяет их, как фильтр
        
        Parameters:
        -----------
        shape_points : array-like, shape (n, 2)
            Точки, задающие форму кривой [x, y]
        intensity_points : array-like, shape (m, 2)
            Точки, задающие интенсивность вдоль кривой [param, intensity]
            где param - параметр вдоль кривой (0..1)
        output_size : tuple, (width, height)
            Размер выходной матрицы PSF
        shape_degree : int
            Степень B-spline для формы
        intensity_degree : int
            Степень B-spline для интенсивности
        n_samples : int
            Количество точек для дискретизации кривой
        """
        super().__init__(1, 'blur')

        self.shape_points = shape_points
        self.intensity_points = intensity_points
        self.output_size = output_size
        self.shape_degree = shape_degree
        self.intensity_degree = intensity_degree
        self.n_samples = n_samples


    def filter(self, image):
        kernel = self.create_dual_bspline_psf()
        res = cv.filter2D(src=image,ddepth=-1,kernel=kernel)
        return res
    
    def discription(self) -> str:
        return f"|Bspline_motion_"
    
    def create_dual_bspline_psf(self):
        """
        Создает PSF используя два B-spline: для формы и для интенсивности.
        
        Returns:
        --------
        psf : ndarray
            Нормализованная матрица PSF
        shape_spline : BSpline
            B-spline для формы кривой
        intensity_spline : BSpline
            B-spline для интенсивности
        sampled_points : ndarray
            Дискретизированные точки кривой с интенсивностями
        """
        
        shape_points = np.array(self.shape_points)
        intensity_points = np.array(self.intensity_points)
        
        # 1. Создаем B-spline для формы кривой
        t_shape = np.linspace(0, 1, len(shape_points))
        shape_spline_x = make_interp_spline(t_shape, shape_points[:, 0], k=self.shape_degree)
        shape_spline_y = make_interp_spline(t_shape, shape_points[:, 1], k=self.shape_degree)
        
        # 2. Создаем B-spline для интенсивности
        t_intensity = np.linspace(0, 1, len(intensity_points))
        intensity_spline = make_interp_spline(t_intensity, intensity_points[:, 1], k=self.intensity_degree)
        
        # 3. Дискретизируем кривую с интенсивностями
        t_samples = np.linspace(0, 1, self.n_samples)
        x_samples = shape_spline_x(t_samples)
        y_samples = shape_spline_y(t_samples)
        intensity_samples = intensity_spline(t_samples)
        
        # Убеждаемся, что интенсивности неотрицательные
        intensity_samples = np.maximum(intensity_samples, 0)
        
        sampled_points = np.column_stack([x_samples, y_samples, intensity_samples])
        
        # 4. Проецируем на матрицу
        width, height = self.output_size
        
        # Находим границы для нормализации координат
        x_min, x_max = x_samples.min(), x_samples.max()
        y_min, y_max = y_samples.min(), y_samples.max()
        
        # Масштабируем координаты к диапазону [0, size-1]
        x_scaled = (x_samples - x_min) / (x_max - x_min) * (width - 1)
        y_scaled = (y_samples - y_min) / (y_max - y_min) * (height - 1)
        
        # Создаем пустую PSF матрицу
        psf = np.zeros(self.output_size)
        
        # Распределяем интенсивности по матрице (простая бининг)
        for i in range(len(x_scaled)):
            x_idx = int(round(x_scaled[i]))
            y_idx = int(round(y_scaled[i]))
            
            # Проверяем границы
            if 0 <= x_idx < width and 0 <= y_idx < height:
                psf[y_idx, x_idx] += intensity_samples[i]
        
        # # 5. Применяем гауссово размытие для сглаживания
        # from scipy.ndimage import gaussian_filter
        # psf = gaussian_filter(psf, sigma=0.7)
        
        # 6. Нормализуем PSF (сумма = 1)
        psf = np.maximum(psf, 0)  # Убеждаемся в неотрицательности
        if np.sum(psf) > 0:
            psf = psf / np.sum(psf)
        
        return psf
    

class Kernel_convolution(FilterBase):
    param = None
    def __init__(self, npy_file_path) -> None:
        """
        задает произвольный фильтр, сохраненный в .npy файл
        Parameters:
        npy_file_path - путь до .npy файла с ядром
        """
        self.npy_file_path = npy_file_path
        super().__init__(1, 'custom_kernel')

    def get_type(self):
        return self.type

    def generate_kernel(self) -> np.ndarray:
        return np.load(self.npy_file_path)

    def discription(self):
        return f"|custom_kelner_"

    def filter(self, image: np.ndarray) -> np.ndarray:
        kernel = np.load(self.npy_file_path)
        blurred = cv.filter2D(image, -1, kernel)
        return blurred