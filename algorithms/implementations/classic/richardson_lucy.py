from algorithms.base import DeconvolutionAlgorithm
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import cv2 as cv
from typing import Tuple, Optional


class RichardsonLucy(DeconvolutionAlgorithm):
    """
    Реализация алгоритма Richardson-Lucy для слепой деконволюции.

    Алгоритм Richardson-Lucy (RL) — итерационный метод для восстановления
    изображений, основанный на максимизации правдоподобия при пуассоновском
    шуме. Слепая версия чередует оценку изображения и ядра.

    Параметры
    ----------
    param : dict
        Словарь параметров:
        - 'psf': начальное приближение PSF (np.ndarray)
        - 'iter': количество внешних итераций (int)
        - 'eps': малая константа для численной стабильности (float)
        - 'm': количество итераций обновления ядра (int)
        - 'r': количество итераций обновления изображения (int)

    Литература
    ----------
    .. [1] Richardson, W. H. (1972). Bayesian-Based Iterative Method of
           Image Restoration. JOSA, 62(1), 55-59.
    .. [2] Lucy, L. B. (1974). An iterative technique for the rectification
           of observed distributions. AJ, 79, 745.
    """
    
    def __init__(self, param):
        super().__init__('Richardson_Lucy')
        self.param = param
        self._estimated_kernel: Optional[np.ndarray] = None
        self._estimated_image: Optional[np.ndarray] = None
    
    def convolve(self, image, psf):
        h, w = image.shape
        kh, kw = psf.shape

        img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

        psf_padded = np.zeros_like(img_padded)
        h_pad, w_pad = img_padded.shape
        start_h = (h_pad - kh) // 2
        start_w = (w_pad - kw) // 2
        psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
        psf_padded = ifftshift(psf_padded)
    
        img_fft = fft2(img_padded)
        psf_fft = fft2(psf_padded)
        result_fft = img_fft * np.conj(psf_fft)
        result = np.real(ifft2(result_fft))
        
        result = result[kh//2+1:h+kh//2+1, kw//2+1:w+kw//2+1]
        
        return np.clip(result, 0.0, 1.0)
    
    def deconvolve(self, image, psf):
        h, w = image.shape
        kh, kw = psf.shape

        img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

        psf_padded = np.zeros_like(img_padded)
        h_pad, w_pad = img_padded.shape
        start_h = (h_pad - kh) // 2
        start_w = (w_pad - kw) // 2
        psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
        psf_padded = ifftshift(psf_padded)

        blurred_fft = fft2(img_padded)
        psf_fft = fft2(psf_padded)
        psf_fft_conj = np.conj(psf_fft)
        wiener_filter = psf_fft_conj / (np.abs(psf_fft) + self.param['eps'])
        result_fft = blurred_fft * wiener_filter
        result = np.real(ifft2(result_fft))

        return np.clip(result[kh//2:h+kh//2, kw//2:w+kw//2], 0.0, 1.0)
    
    def process(self, image) -> Tuple[np.ndarray, np.ndarray]:
        """Основной метод восстановления изображения.
        
        Параметры
        ----------
        image : np.ndarray
            Входное зашумленное изображение
            
        Возвращает
        -------
        tuple
            (восстановленное_изображение, оценённое_ядро)
        """
        if len(image.shape) != 2:
            raise Exception("Currently supports only grayscale images")
        
        h, w = image.shape
        kh, kw = self.param['psf'].shape
        eps = self.param['eps']

        original_image = image.copy() / 255.0
        processed_image = original_image.copy()

        g = np.full((h, w), 0.0)
        x_center = (w - kw) // 2
        y_center = (h - kh) // 2
        g[y_center:y_center+kh, x_center:x_center+kw] = np.rot90(self.param['psf'], 2)

        f = self.deconvolve(processed_image, g)

        for i in range(0, self.param['iter']):
            # Обновление ядра (m итераций)
            for j in range(0, self.param['m']):
                conv_gf = self.convolve(g, f)
                ratio = original_image / (conv_gf + eps)
                g *= self.convolve(ratio, ifft2(np.conj(fft2(f))))
                g = np.abs(g) / np.sum(np.abs(g))

            # Обновление изображения (r итераций)
            for j in range(0, self.param['r']):
                conv_fg = self.convolve(f, g)
                ratio = original_image / (conv_fg + eps)
                f *= self.convolve(ratio, ifft2(np.conj(fft2(g))))
                f = np.clip(f, 0.0, 1.0)

        res = f[:h, :w].real * 255.0
        
        self._estimated_image = res.astype(np.int16)
        
        estimated_kernel = g[y_center:y_center+kh, x_center:x_center+kw]
        self._estimated_kernel = estimated_kernel
        
        return self._estimated_image, self._estimated_kernel
    
    def get_estimated_kernel(self) -> Optional[np.ndarray]:
        """Возвращает оцененное ядро размытия.
        
        Возвращает
        -------
        np.ndarray или None
            Оцененное ядро размытия (PSF) или None, если обработка
            еще не выполнялась
        """
        return self._estimated_kernel
    
    def get_estimated_image(self) -> Optional[np.ndarray]:
        """Возвращает восстановленное изображение.
        
        Возвращает
        -------
        np.ndarray или None
            Восстановленное изображение или None, если обработка
            еще не выполнялась
        """
        return self._estimated_image
    
    def reset_estimations(self) -> None:
        """Сбрасывает сохраненные оценки изображения и ядра."""
        self._estimated_kernel = None
        self._estimated_image = None

    def get_param(self):
        """Возвращает текущие параметры алгоритма."""
        return [
            ('psf_shape', self.param['psf'].shape if self.param.get('psf') is not None else None),
            ('iter', self.param.get('iter', 0)),
            ('eps', self.param.get('eps', 1e-8)),
            ('m', self.param.get('m', 1)),
            ('r', self.param.get('r', 1)),
            ('has_estimation', self._estimated_kernel is not None),
        ]

    def change_param(self, param):
        """Изменяет параметры алгоритма."""
        if not isinstance(param, dict):
            return
        
        param_changed = False
        for key in ['psf', 'iter', 'eps', 'm', 'r']:
            if key in param and param[key] is not None:
                if key in self.param and self.param[key] is not None:
                    if (isinstance(self.param[key], np.ndarray) and 
                        isinstance(param[key], np.ndarray)):
                        if not np.array_equal(self.param[key], param[key]):
                            param_changed = True
                    elif self.param[key] != param[key]:
                        param_changed = True
                self.param[key] = param[key]
        
        if param_changed:
            self.reset_estimations()


class NonBlindRichardsonLucy:
    """
    Реализация алгоритма Richardson-Lucy для не слепой деконволюции.
    
    Параметры
    ----------
    psf : np.ndarray
        Известная функция рассеяния точки (Point Spread Function)
    eps : float
        Малая константа для предотвращения деления на ноль
    iter : int
        Количество итераций алгоритма
    
    Примечания
    ----------
    Для слепой деконволюции используйте класс RichardsonLucy.
    """
    
    def __init__(self, psf, eps, iter):
        self.psf = psf
        self.eps = eps
        self.iter = iter
        self._estimated_image: Optional[np.ndarray] = None
    
    def convolve(self, image, psf):
        h, w = image.shape
        kh, kw = psf.shape

        img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

        psf_padded = np.zeros_like(img_padded)
        h_pad, w_pad = img_padded.shape
        start_h = (h_pad - kh) // 2
        start_w = (w_pad - kw) // 2
        psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
        psf_padded = ifftshift(psf_padded)
    
        img_fft = fft2(img_padded)
        psf_fft = fft2(psf_padded)
        result_fft = img_fft * np.conj(psf_fft)
        result = np.real(ifft2(result_fft))
        
        result = result[kh//2+1:h+kh//2+1, kw//2+1:w+kw//2+1]
        
        return np.clip(result, 0.0, 1.0)
    
    def deconvolve(self, image, psf):
        h, w = image.shape
        kh, kw = psf.shape

        img_padded = np.pad(image, ((kh//2, kh//2), (kw//2, kw//2)), mode='reflect')

        psf_padded = np.zeros_like(img_padded)
        h_pad, w_pad = img_padded.shape
        start_h = (h_pad - kh) // 2
        start_w = (w_pad - kw) // 2
        psf_padded[start_h:start_h+kh, start_w:start_w+kw] = np.rot90(psf, 2)
        psf_padded = ifftshift(psf_padded)

        blurred_fft = fft2(img_padded)
        psf_fft = fft2(psf_padded)
        psf_fft_conj = np.conj(psf_fft)
        wiener_filter = psf_fft_conj / (np.abs(psf_fft) + self.eps)
        result_fft = blurred_fft * wiener_filter
        result = np.real(ifft2(result_fft))

        return np.clip(result[kh//2:h+kh//2, kw//2:w+kw//2], 0.0, 1.0)
    
    def process(self, image):
        """Основной метод восстановления изображения."""
        if len(image.shape) != 2:
            raise Exception("Currently supports only grayscale images")
        
        h, w = image.shape
        kh, kw = self.psf.shape
        eps = self.eps

        original_image = image.copy() / 255.0
        processed_image = original_image.copy()

        g = np.full((h, w), 0.0)
        x_center = (w - kw) // 2
        y_center = (h - kh) // 2
        g[y_center:y_center+kh, x_center:x_center+kw] = np.rot90(self.psf, 2)

        f = self.deconvolve(processed_image, g)

        for i in range(0, self.iter):
            conv_fg = self.convolve(f, g)
            ratio = original_image / (conv_fg + eps)
            f *= self.convolve(ratio, np.rot90(np.rot90(g)))
            f = np.clip(f, 0.0, 1.0)

        res = f[:h, :w].real * 255.0
        self._estimated_image = res.astype(np.int16)
        return self._estimated_image
    
    def get_estimated_image(self) -> Optional[np.ndarray]:
        """Возвращает восстановленное изображение."""
        return self._estimated_image


def richardson_lucy_blind_deconvolution(
    image: np.ndarray,
    psf: np.ndarray,
    max_iter: int = 50,
    eps: float = 1e-8,
    m_iter: int = 1,
    r_iter: int = 1,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Обёртка для совместимости со старым API.
    
    Возвращает (восстановленное изображение, оценённое ядро).
    
    Параметры
    ----------
    image : np.ndarray
        Зашумленное изображение
    psf : np.ndarray
        Начальное приближение PSF
    max_iter : int, optional
        Количество внешних итераций (по умолчанию 50)
    eps : float, optional
        Малая константа для численной стабильности (по умолчанию 1e-8)
    m_iter : int, optional
        Количество итераций обновления ядра (по умолчанию 1)
    r_iter : int, optional
        Количество итераций обновления изображения (по умолчанию 1)
    **kwargs : dict
        Дополнительные параметры
    
    Возвращает
    -------
    tuple
        (восстановленное_изображение, оценённое_ядро)
    """
    param = {
        'psf': psf,
        'iter': max_iter,
        'eps': eps,
        'm': m_iter,
        'r': r_iter
    }
    algo = RichardsonLucy(param, **kwargs)
    result = algo.process(image)
    
    if isinstance(result, tuple) and len(result) == 2:
        return result
    else:
        kernel = algo.get_estimated_kernel()
        if kernel is None:
            kernel = psf
        return result, kernel
