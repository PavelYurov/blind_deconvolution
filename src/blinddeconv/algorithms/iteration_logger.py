"""
Универсальный логгер итераций для алгоритмов слепой деконволюции.

Используется как callback: alg.set_callback(logger) перед alg.process().
Собирает промежуточные ядра, изображения, метрики и сохраняет их.

Пример использования:
    from blinddeconv.algorithms.iteration_logger import IterationLogger

    logger = IterationLogger(
        save_dir=Path("results/iterations/img1"),
        original=original_image,       # для PSNR/SSIM
        gt_kernel=gt_kernel,           # для kernel MSE / error ratio
        blurred=blurred_image,         # для non-blind восстановления
        nonblind_func=my_nonblind,     # callable(blurred, kernel) -> restored
        save_kernel_every=5,
        save_image_every=10,
    )
    alg.set_callback(logger)
    restored, kernel = alg.process(blurred)
    logger.save_csv()
"""

import numpy as np
import cv2 as cv
import pandas as pd
from pathlib import Path
from typing import Optional, Callable


class IterationLogger:
    """
    Callback-логгер для сбора промежуточных результатов по итерациям.

    Вызывается алгоритмом как self._callback(state), где state — dict:
        'iteration'  : int   — номер итерации (внутри текущего масштаба)
        'scale'      : int   — индекс масштаба пирамиды (0 = финальный)
        'num_scales' : int   — всего масштабов
        'kernel'     : ndarray — текущая оценка ядра
        'image'      : ndarray or None — текущее латентное изображение
        'metrics'    : dict  — доп. метрики из алгоритма (kernel_diff, beta, ...)
    """

    def __init__(
        self,
        save_dir: Path,
        original: Optional[np.ndarray] = None,
        gt_kernel: Optional[np.ndarray] = None,
        blurred: Optional[np.ndarray] = None,
        nonblind_func: Optional[Callable] = None,
        save_kernel_every: int = 5,
        save_image_every: int = 10,
        only_finest_scale: bool = True,
    ):
        """
        Parameters
        ----------
        save_dir : Path
            Папка для сохранения (создаётся автоматически).
        original : ndarray or None
            Оригинальное изображение float64 [0,1] для вычисления PSNR/SSIM.
        gt_kernel : ndarray or None
            Истинное ядро float64 (нормализованное, sum=1) для kernel MSE.
        blurred : ndarray or None
            Размытое изображение float64 [0,1] для non-blind восстановления.
        nonblind_func : callable or None
            Функция (blurred, kernel) -> restored_float64 для полного
            non-blind шага на каждой сохраняемой итерации.
        save_kernel_every : int
            Сохранять ядро как PNG каждые N итераций (только finest scale).
        save_image_every : int
            Сохранять восстановленное изображение каждые N итераций.
            Требует nonblind_func и blurred.
        only_finest_scale : bool
            Если True — логируем только finest scale (scale=0).
            Если False — логируем все масштабы.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "kernels").mkdir(exist_ok=True)
        (self.save_dir / "restored").mkdir(exist_ok=True)

        self.original = original
        self.gt_kernel = gt_kernel
        self.blurred = blurred
        self.nonblind_func = nonblind_func
        self.save_kernel_every = save_kernel_every
        self.save_image_every = save_image_every
        self.only_finest_scale = only_finest_scale

        # Глобальный счётчик итераций (через все масштабы)
        self.global_iter = 0
        # Счётчик итераций внутри finest scale
        self._finest_iter = 0

        # Лог: список словарей, каждый — одна итерация
        self.log = []

    def __call__(self, state: dict):
        """Вызывается алгоритмом на каждой итерации."""
        scale = state.get('scale', 0)
        num_scales = state.get('num_scales', 1)
        iteration = state.get('iteration', self.global_iter)
        kernel = state.get('kernel', None)
        image = state.get('image', None)
        metrics = state.get('metrics', {})

        is_finest = (scale == 0)

        if self.only_finest_scale and not is_finest:
            self.global_iter += 1
            return

        if is_finest:
            self._finest_iter += 1
            local_iter = self._finest_iter
        else:
            local_iter = iteration

        # ── Базовая строка лога ──────────────────────────────────────────
        row = {
            'global_iter': self.global_iter,
            'scale': scale,
            'num_scales': num_scales,
            'local_iter': local_iter,
        }
        row.update(metrics)

        # ── Kernel MSE (если есть GT ядро) ───────────────────────────────
        if kernel is not None and self.gt_kernel is not None:
            k_est = kernel.astype(np.float64)
            if k_est.ndim > 2:
                k_est = k_est[:, :, 0]
            k_est_norm = k_est / (k_est.sum() + 1e-12)

            k_gt = self.gt_kernel.astype(np.float64)
            if k_gt.ndim > 2:
                k_gt = k_gt[:, :, 0]
            k_gt_norm = k_gt / (k_gt.sum() + 1e-12)

            # Привести к одному размеру (pad меньшее)
            k_est_p, k_gt_p = self._match_kernel_sizes(k_est_norm, k_gt_norm)

            row['kernel_mse'] = float(np.mean((k_est_p - k_gt_p) ** 2))
            row['kernel_rmse'] = float(np.sqrt(row['kernel_mse']))
            row['kernel_mae'] = float(np.mean(np.abs(k_est_p - k_gt_p)))

        # ── Сохранение ядра как PNG ──────────────────────────────────────
        if kernel is not None and local_iter % self.save_kernel_every == 0:
            k_save = np.rot90(kernel.copy(), 2)  # Поворот на 180° (корреляция → свёртка)
            if k_save.max() > 0:
                k_save = (k_save / k_save.max() * 255).astype(np.uint8)
            fname = f"kernel_s{scale}_iter{local_iter:04d}.png"
            cv.imwrite(str(self.save_dir / "kernels" / fname), k_save)

        # ── Non-blind восстановление: метрики КАЖДУЮ итерацию, PNG — каждые N ─
        if (kernel is not None
                and self.nonblind_func is not None
                and self.blurred is not None):
            try:
                restored = self.nonblind_func(self.blurred, kernel)
                restored = np.clip(restored, 0.0, 1.0)

                # Сохранение PNG (только каждые save_image_every итераций)
                if local_iter % self.save_image_every == 0:
                    r_save = (restored * 255).astype(np.uint8)
                    fname = f"restored_s{scale}_iter{local_iter:04d}.png"
                    cv.imwrite(str(self.save_dir / "restored" / fname), r_save)

                # PSNR / SSIM — считаем КАЖДУЮ итерацию
                if self.original is not None:
                    psnr, ssim = self._compute_metrics(self.original, restored)
                    row['psnr'] = round(psnr, 4)
                    row['ssim'] = round(ssim, 4)
            except Exception as e:
                row['nonblind_error'] = str(e)

        self.log.append(row)
        self.global_iter += 1

    def save_csv(self, filename: str = "iterations_log.csv"):
        """Сохранить лог итераций в CSV."""
        if not self.log:
            return
        df = pd.DataFrame(self.log)
        df.to_csv(self.save_dir / filename, index=False)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.log)

    def reset(self):
        """Сброс логгера для повторного использования."""
        self.log = []
        self.global_iter = 0
        self._finest_iter = 0

    # ── Вспомогательные методы ────────────────────────────────────────────

    @staticmethod
    def _match_kernel_sizes(k1: np.ndarray, k2: np.ndarray):
        """Привести два ядра к одному размеру (pad нулями по центру)."""
        h = max(k1.shape[0], k2.shape[0])
        w = max(k1.shape[1], k2.shape[1])

        def _center_pad(k, th, tw):
            ph = th - k.shape[0]
            pw = tw - k.shape[1]
            return np.pad(k, ((ph // 2, ph - ph // 2),
                              (pw // 2, pw - pw // 2)))

        return _center_pad(k1, h, w), _center_pad(k2, h, w)

    @staticmethod
    def _compute_metrics(original: np.ndarray, restored: np.ndarray):
        """PSNR и SSIM между двумя float64 [0,1] изображениями."""
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        orig = original.astype(np.float64)
        if orig.max() > 1.0:
            orig /= 255.0
        rest = restored.astype(np.float64)
        if rest.max() > 1.0:
            rest /= 255.0

        # Привести к одному количеству каналов
        if orig.ndim == 3 and rest.ndim == 2:
            rest = np.stack([rest] * orig.shape[2], axis=-1)
        elif orig.ndim == 2 and rest.ndim == 3:
            orig = np.stack([orig] * rest.shape[2], axis=-1)

        # Обрезать до минимального размера
        h = min(orig.shape[0], rest.shape[0])
        w = min(orig.shape[1], rest.shape[1])
        orig = orig[:h, :w]
        rest = rest[:h, :w]

        psnr = peak_signal_noise_ratio(orig, rest, data_range=1.0)
        ssim = structural_similarity(
            orig, rest, data_range=1.0,
            channel_axis=2 if orig.ndim == 3 else None)
        return psnr, ssim
