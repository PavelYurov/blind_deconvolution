"""
Воркер для параллельной обработки изображений слепой деконволюцией.

Используется из ноутбуков presentation_graphics_*.ipynb через ProcessPoolExecutor.
Все функции определены на уровне модуля, чтобы быть picklable на Windows (spawn).
"""
import sys
import importlib
import math
import time as _time
import numpy as np
import cv2 as cv
from pathlib import Path


def init_worker(project_root: str):
    """Initializer для ProcessPoolExecutor — настраивает sys.path в дочернем процессе."""
    pr = str(project_root)
    src = str(Path(project_root) / "src")
    if pr not in sys.path:
        sys.path.insert(0, pr)
    if src not in sys.path:
        sys.path.insert(0, src)


def _error_ratio_nonblind(original, blurred, true_kernel, est_kernel):
    from numpy.fft import fft2, ifft2
    h, w = blurred.shape[:2]
    reg = 1e-3

    def _wr(y, k):
        K = fft2(k, s=(h, w))
        Y = fft2(y, s=(h, w))
        return np.real(ifft2(np.conj(K) * Y / (np.abs(K) ** 2 + reg)))

    x_true = _wr(blurred, true_kernel)
    x_est = _wr(blurred, est_kernel)
    orig = original.astype(np.float64)
    mse_true = np.mean((orig - x_true) ** 2)
    mse_est = np.mean((orig - x_est) ** 2)
    return mse_est / mse_true if mse_true > 1e-12 else 1.0


def process_one(task: dict) -> dict | None:
    """
    Обрабатывает одно искажённое изображение.

    task — словарь с ключами:
        project_root, dist_file, orig_dir, kernel_dir,
        has_originals, has_kernels, dataset_name, algorithm_label,
        alg_module, alg_class, alg_kwargs, alg_kernel_param, ds_result_dir
    """
    # ── imports (в каждом вызове, т.к. дочерний процесс) ──────────────────────
    init_worker(task['project_root'])
    from src.blinddeconv.processing.utils import (
        imread, prepare_image_for_metric, calculate_metrics,
    )

    dist_file = Path(task['dist_file'])
    algorithm_label = task['algorithm_label']
    dataset_name = task['dataset_name']
    ds_result_dir = Path(task['ds_result_dir'])
    has_originals = task['has_originals']
    has_kernels = task['has_kernels']
    orig_dir = Path(task['orig_dir'])
    kernel_dir = Path(task['kernel_dir'])

    # ── Парсинг имени файла ──────────────────────────────────────────────────
    stem = dist_file.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        img_name, kernel_name = parts[0], parts[1]
        noise_name = "_".join(parts[2:]) if len(parts) > 2 else ""
    else:
        img_name, kernel_name, noise_name = stem, "", ""

    # ── Загрузка изображений ─────────────────────────────────────────────────
    blurred = imread(str(dist_file), False)
    if blurred is None:
        return None

    original = None
    orig_path = None
    if has_originals:
        for ext in ['.png', '.jpg', '.bmp', '.tif']:
            candidate = orig_dir / f"{img_name}{ext}"
            if candidate.exists():
                orig_path = candidate
                break
        if orig_path:
            original = imread(str(orig_path), False)

    gt_kernel = None
    gt_kernel_path = None
    if has_kernels and kernel_name:
        if kernel_dir.exists():
            for f in kernel_dir.iterdir():
                if f.is_file() and kernel_name in f.stem:
                    gt_kernel_path = f
                    break
        if gt_kernel_path:
            gt_kernel = imread(str(gt_kernel_path), False)

    # ── Размер ядра ──────────────────────────────────────────────────────────
    if gt_kernel is not None:
        ks = gt_kernel.shape[:2]
        ks = (ks[0] if ks[0] % 2 == 1 else ks[0] + 1,
              ks[1] if ks[1] % 2 == 1 else ks[1] + 1)
    else:
        ks = (21, 21)

    # # ── Создание и запуск алгоритма ──────────────────────────────────────────
    # mod = importlib.import_module(task['alg_module'])
    # alg_cls = getattr(mod, task['alg_class'])
    # alg_kwargs = dict(task['alg_kwargs'])
    # alg_kwargs[task['alg_kernel_param']] = ks
    # alg = alg_cls(**alg_kwargs)

    # t0 = _time.time()
    # try:
    #     restored_image, est_kernel = alg.process(blurred)
    # except Exception as e:
    #     return {'error': str(e), 'dist_file': dist_file.name}
    # elapsed = _time.time() - t0

    # # ── Сохранение ───────────────────────────────────────────────────────────
    # restored_path = ds_result_dir / "restored" / f"{stem}_{algorithm_label}.png"
    # kernel_path = ds_result_dir / "kernels" / f"{stem}_{algorithm_label}_kernel.png"
    # cv.imwrite(str(restored_path), restored_image)
    # k_save = est_kernel.copy()
    # if k_save.max() > 0:
    #     k_save = (k_save / k_save.max() * 255).astype(np.uint8)
    # cv.imwrite(str(kernel_path), k_save)

    # # ── Формирование строки результата ───────────────────────────────────────
    # row = {
    #     'dataset': dataset_name,
    #     'distorted_file': dist_file.name,
    #     'image_name': img_name,
    #     'kernel_name': kernel_name or '',
    #     'noise_name': noise_name or '',
    #     'kernel_shape': str(ks),
    #     'time_sec': round(elapsed, 3),
    #     'restored_path': str(restored_path),
    #     'kernel_path': str(kernel_path),
    #     'gt_kernel_path': str(gt_kernel_path) if gt_kernel_path else '',
    #     'original_path': str(orig_path) if orig_path else '',
    # }
    # ── Создание и запуск алгоритма ──────────────────────────────────────────
    mod = importlib.import_module(task['alg_module'])
    alg_cls = getattr(mod, task['alg_class'])
    alg_kwargs = dict(task['alg_kwargs'])

    # === УМНАЯ АДАПТАЦИЯ ТИПА ЯДРА ===
    k_param = task['alg_kernel_param']
    original_k_val = alg_kwargs.get(k_param)

    if isinstance(original_k_val, int):
        # Если алгоритм ожидает int (как PMP_BD), берем бóльшую сторону ядра
        adapted_ks = max(ks[0], ks[1])
    elif isinstance(original_k_val, list):
        adapted_ks = list(ks)
    else:
        # По умолчанию передаем как tuple (как LIP_BD)
        adapted_ks = tuple(ks)

    alg_kwargs[k_param] = adapted_ks
    alg = alg_cls(**alg_kwargs)

    # Читаем количество прогонов из задачи (по умолчанию 1)
    num_runs = task.get('num_runs', 1)
    times =[]
    
    restored_image = None
    est_kernel = None

    for i in range(num_runs):
        t0 = _time.time()
        try:
            # На последнем прогоне сохраняем результат
            res_img, res_k = alg.process(blurred.copy()) 
            if i == num_runs - 1:
                restored_image = res_img
                est_kernel = res_k
        except Exception as e:
            return {'error': str(e), 'dist_file': dist_file.name}
        elapsed = _time.time() - t0
        times.append(elapsed)

    # Берем медианное время для стабильности
    median_time = float(np.median(times))

    # ── Сохранение ───────────────────────────────────────────────────────────
    restored_path = ds_result_dir / "restored" / f"{stem}_{algorithm_label}.png"
    kernel_path = ds_result_dir / "kernels" / f"{stem}_{algorithm_label}_kernel.png"
    cv.imwrite(str(restored_path), restored_image)
    k_save = np.rot90(est_kernel.copy(), 2)  # fix: 180° rotation (conv vs correlation)
    if k_save.max() > 0:
        k_save = (k_save / k_save.max() * 255).astype(np.uint8)
    cv.imwrite(str(kernel_path), k_save)

    # Вычисляем мегапиксели для графика масштабируемости
    h, w = blurred.shape[:2]
    megapixels = round((h * w) / 1_000_000, 3)

    # ── Формирование строки результата ───────────────────────────────────────
    row = {
        'dataset': dataset_name,
        'distorted_file': dist_file.name,
        'image_name': img_name,
        'kernel_name': kernel_name or '',
        'noise_name': noise_name or '',
        'kernel_shape': str(ks),
        'image_megapixels': megapixels, # <--- ДОБАВИЛИ РАЗМЕР КАРТИНКИ
        'time_sec': round(median_time, 3), # <--- ТЕПЕРЬ ТУТ МЕДИАНА
        'runs_count': num_runs,
        'restored_path': str(restored_path),
        'kernel_path': str(kernel_path),
        'gt_kernel_path': str(gt_kernel_path) if gt_kernel_path else '',
        'original_path': str(orig_path) if orig_path else '',
    }

    # ── PSNR / SSIM ──────────────────────────────────────────────────────────
    if original is not None:
        orig_m = prepare_image_for_metric(np.atleast_3d(original))
        rest_m = prepare_image_for_metric(np.atleast_3d(restored_image))
        psnr_val, ssim_val = calculate_metrics(orig_m, rest_m, data_range=1.0)
        row['psnr'] = round(psnr_val, 4)
        row['ssim'] = round(ssim_val, 4)

        blur_m = prepare_image_for_metric(np.atleast_3d(blurred))
        psnr_blur, ssim_blur = calculate_metrics(orig_m, blur_m, data_range=1.0)
        row['psnr_blurred'] = round(psnr_blur, 4)
        row['ssim_blurred'] = round(ssim_blur, 4)
    else:
        row['psnr'] = math.nan
        row['ssim'] = math.nan
        row['psnr_blurred'] = math.nan
        row['ssim_blurred'] = math.nan

    # ── Error ratio ──────────────────────────────────────────────────────────
    if original is not None and gt_kernel is not None:
        try:
            orig_gray = original if original.ndim == 2 else original[:, :, 0]
            blur_gray = blurred if blurred.ndim == 2 else blurred[:, :, 0]
            gt_k = gt_kernel.astype(np.float64)
            if gt_k.ndim > 2:
                gt_k = gt_k[:, :, 0]
            gt_k = gt_k / (gt_k.sum() + 1e-12)
            est_k = est_kernel.astype(np.float64)
            if est_k.ndim > 2:
                est_k = est_k[:, :, 0]
            est_k = est_k / (est_k.sum() + 1e-12)
            er = _error_ratio_nonblind(
                orig_gray.astype(np.float64) / 255.0,
                blur_gray.astype(np.float64) / 255.0,
                gt_k, est_k,
            )
            row['error_ratio'] = round(er, 4)
        except Exception:
            row['error_ratio'] = math.nan
    else:
        row['error_ratio'] = math.nan

    return row
