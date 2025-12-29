"""
Модуль оптимизации гиперпараметров.

Реализует оптимизацию гиперпараметров алгоритмов слепой деконволюции
с использованием байесовских методов оптимизации.

Поддерживаемые методы:
    - TPE (Tree-structured Parzen Estimator)
    - Случайный поиск
    - Гауссовские процессы
    - NSGA-II для многокритериальной оптимизации

Автор: Беззаборов А.А.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable, Union

import numpy as np
import cv2 as cv

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Hyperparameter optimization disabled.")

try:
    from optuna.integration import BoTorchSampler
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False

import algorithms.base as base
import metrics

from extensions.base import (
    ProcessingExtension,
    OptimizationResult,
    OptimizationMethod,
    logger
)


class HyperparameterOptimizer(ProcessingExtension):
    """
    Оптимизация гиперпараметров с использованием байесовской оптимизации.
    
    Реализует эффективный поиск гиперпараметров для алгоритмов слепой 
    деконволюции с использованием фреймворка Optuna.
    
    Поддерживаемые стратегии оптимизации:
        - TPE (Tree-structured Parzen Estimator)
        - Случайный поиск
        - Гауссовские процессы (при наличии BoTorch)
        - NSGA-II для многокритериальной оптимизации
    """
    
    def __init__(self, processing_instance: Any, output_folder: str = "parameters"):
        """
        Инициализация оптимизатора.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing с изображениями.
        output_folder : str
            Директория для сохранения результатов оптимизации.
        """
        super().__init__(processing_instance, output_folder)
        
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter optimization. "
                "Install via: pip install optuna"
            )
    
    def execute(
        self,
        algorithm_processor: base.DeconvolutionAlgorithm,
        param_ranges: Dict[str, Tuple[Union[int, float], Union[int, float]]],
        n_trials: int = 50,
        metric: str = 'PSNR',
        timeout: Optional[int] = 3600,
        method: OptimizationMethod = OptimizationMethod.TPE,
        n_jobs: int = 1,
        seed: int = 42,
        show_progress: bool = True,
        save_results: bool = True
    ) -> OptimizationResult:
        """
        Оптимизация гиперпараметров для заданного алгоритма.
        
        Параметры
        ---------
        algorithm_processor : DeconvolutionAlgorithm
            Алгоритм деконволюции для оптимизации.
        param_ranges : Dict[str, Tuple]
            Словарь соответствия имен параметров кортежам (min, max).
        n_trials : int, по умолчанию 50
            Количество испытаний оптимизации.
        metric : str, по умолчанию 'PSNR'
            Метрика оптимизации: 'PSNR', 'SSIM' или 'SHARPNESS'.
        timeout : Optional[int], по умолчанию 3600
            Максимальное время оптимизации в секундах (None без ограничения).
        method : OptimizationMethod, по умолчанию TPE
            Метод оптимизации.
        n_jobs : int, по умолчанию 1
            Количество параллельных задач (-1 для всех ядер).
        seed : int, по умолчанию 42
            Seed для воспроизводимости.
        show_progress : bool, по умолчанию True
            Отображать индикатор прогресса.
        save_results : bool, по умолчанию True
            Сохранять лучшие параметры в JSON-файл.
            
        Возвращает
        ----------
        OptimizationResult
            Контейнер с лучшими параметрами и историей оптимизации.
        """
        start_time = time.time()
        algorithm_name = algorithm_processor.get_name()
        
        logger.info(f"Starting hyperparameter optimization for {algorithm_name}")
        logger.info(f"Method: {method.value}, Trials: {n_trials}, Metric: {metric}")
        
        sampler = self._create_sampler(method, seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"{algorithm_name}_optimization"
        )
        
        objective = self._create_objective(algorithm_processor, param_ranges, metric)
        
        if not show_progress:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress
        )
        
        elapsed_time = time.time() - start_time
        
        result = OptimizationResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            study=study,
            history=[
                {'params': t.params, 'value': t.value, 'state': str(t.state)}
                for t in study.trials
            ],
            elapsed_time=elapsed_time
        )
        
        logger.info("OPTIMIZATION COMPLETE")
        logger.info(f"Best {metric}: {result.best_value:.4f}")
        logger.info(f"Best parameters:")
        for param, value in result.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Total trials: {result.n_trials}")
        logger.info(f"Elapsed time: {elapsed_time:.2f} s")
        
        self._apply_and_process(algorithm_processor, result.best_params, algorithm_name)
        
        if save_results:
            self._save_results(result, algorithm_name)
        
        return result
    
    def _create_sampler(
        self, 
        method: OptimizationMethod, 
        seed: int
    ) -> optuna.samplers.BaseSampler:
        """
        Создание семплера Optuna на основе метода оптимизации.
        
        Параметры
        ---------
        method : OptimizationMethod
            Выбранный метод оптимизации.
        seed : int
            Seed для генератора случайных чисел.
            
        Возвращает
        ----------
        BaseSampler
            Настроенный семплер Optuna.
        """
        if method == OptimizationMethod.TPE:
            return TPESampler(
                seed=seed,
                n_startup_trials=10,
                multivariate=True
            )
        elif method == OptimizationMethod.RANDOM:
            return RandomSampler(seed=seed)
        elif method == OptimizationMethod.GP:
            if not BOTORCH_AVAILABLE:
                logger.warning("BoTorch not available, falling back to TPE")
                return TPESampler(seed=seed)
            return BoTorchSampler(seed=seed)
        elif method == OptimizationMethod.NSGA2:
            return NSGAIISampler(seed=seed)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _create_objective(
        self,
        algorithm_processor: base.DeconvolutionAlgorithm,
        param_ranges: Dict[str, Tuple],
        metric: str
    ) -> Callable:
        """
        Создание целевой функции для Optuna.
        
        Параметры
        ---------
        algorithm_processor : DeconvolutionAlgorithm
            Алгоритм для оптимизации.
        param_ranges : Dict
            Диапазоны поиска параметров.
        metric : str
            Целевая метрика.
            
        Возвращает
        ----------
        Callable
            Целевая функция для Optuna.
        """
        def objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif isinstance(min_val, float) or isinstance(max_val, float):
                    if max_val / max(min_val, 1e-10) > 100:
                        params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val, log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val
                        )
                else:
                    params[param_name] = trial.suggest_categorical(
                        param_name, [min_val, max_val]
                    )
            
            algorithm_processor.change_param(params)
            
            total_score = 0.0
            n_images = 0
            
            for img_obj in self.processing.images:
                original = img_obj.get_original_image()
                blurred = img_obj.get_blurred_image()
                
                if blurred is None:
                    blurred = original
                if original is None:
                    continue
                
                try:
                    restored, _ = algorithm_processor.process(blurred)
                    
                    original_prep = self._prepare_image(original)
                    restored_prep = self._prepare_image(restored)
                    
                    if np.max(restored_prep) < 1e-6:
                        logger.warning("Degenerate solution detected (near-zero image)")
                        continue
                    
                    score = self._compute_metric(original_prep, restored_prep, metric)
                    total_score += score
                    n_images += 1
                    
                except Exception as e:
                    logger.warning(f"Error evaluating parameters {params}: {e}")
                    continue
            
            if n_images == 0:
                return float('-inf')
            
            return total_score / n_images
        
        return objective
    
    def _compute_metric(
        self, 
        original: np.ndarray, 
        restored: np.ndarray, 
        metric: str
    ) -> float:
        """
        Вычисление метрики качества.
        
        Параметры
        ---------
        original : np.ndarray
            Оригинальное изображение.
        restored : np.ndarray
            Восстановленное изображение.
        metric : str
            Название метрики.
            
        Возвращает
        ----------
        float
            Значение метрики (большее значение лучше).
        """
        metric_upper = metric.upper()
        
        if metric_upper == 'PSNR':
            return metrics.PSNR(original, restored)
        elif metric_upper == 'SSIM':
            return metrics.SSIM(original, restored, data_range=1.0)
        elif metric_upper == 'SHARPNESS':
            return metrics.Sharpness(restored)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _apply_and_process(
        self,
        algorithm_processor: base.DeconvolutionAlgorithm,
        best_params: Dict[str, Any],
        algorithm_name: str
    ) -> None:
        """
        Применение лучших параметров и обработка всех изображений.
        
        Параметры
        ---------
        algorithm_processor : DeconvolutionAlgorithm
            Алгоритм для применения.
        best_params : Dict
            Лучшие гиперпараметры.
        algorithm_name : str
            Имя алгоритма для именования файлов.
        """
        algorithm_processor.change_param(best_params)
        
        for img_obj in self.processing.images:
            original = img_obj.get_original_image()
            blurred = img_obj.get_blurred_image()
            
            if blurred is None:
                blurred = original
            if original is None:
                continue
            
            try:
                restored, kernel = algorithm_processor.process(blurred)
                
                restored_path, kernel_path = self._generate_output_paths(
                    img_obj, algorithm_name
                )
                
                cv.imwrite(restored_path, restored)
                cv.imwrite(kernel_path, kernel)
                
                self._save_metrics(
                    img_obj, original, restored,
                    restored_path, kernel_path, algorithm_name
                )
                
            except Exception as e:
                logger.error(f"Error processing with optimized params: {e}")
    
    def _generate_output_paths(
        self, 
        img_obj: Any, 
        algorithm_name: str
    ) -> Tuple[str, str]:
        """
        Генерация уникальных путей для выходных файлов.
        
        Параметры
        ---------
        img_obj : Any
            Объект изображения.
        algorithm_name : str
            Имя алгоритма.
            
        Возвращает
        ----------
        Tuple[str, str]
            Пути для восстановленного изображения и ядра.
        """
        base_path = Path(img_obj.get_blurred() or img_obj.get_original())
        base_name = base_path.stem
        suffix = base_path.suffix
        
        restored_path = self.processing._generate_unique_file_path(
            self.processing.folder_path_restored,
            f"{base_name}_{algorithm_name}_optimized{suffix}"
        )
        
        kernel_path = self.processing._generate_unique_file_path(
            self.processing.folder_path_restored,
            f"{base_name}_{algorithm_name}_optimized_kernel{suffix}"
        )
        
        return str(restored_path), str(kernel_path)
    
    def _save_metrics(
        self,
        img_obj: Any,
        original: np.ndarray,
        restored: np.ndarray,
        restored_path: str,
        kernel_path: str,
        algorithm_name: str
    ) -> None:
        """
        Вычисление и сохранение метрик качества.
        
        Параметры
        ---------
        img_obj : Any
            Объект изображения.
        original : np.ndarray
            Оригинальное изображение.
        restored : np.ndarray
            Восстановленное изображение.
        restored_path : str
            Путь к сохраненному восстановленному изображению.
        kernel_path : str
            Путь к сохраненному ядру.
        algorithm_name : str
            Имя алгоритма.
        """
        original_prep = self._prepare_image(original)
        restored_prep = self._prepare_image(restored)
        
        try:
            psnr_val = metrics.PSNR(original_prep, restored_prep)
        except Exception as e:
            logger.warning(f"PSNR calculation failed: {e}")
            psnr_val = float('nan')
        
        try:
            ssim_val = metrics.SSIM(original_prep, restored_prep, data_range=1.0)
        except Exception as e:
            logger.warning(f"SSIM calculation failed: {e}")
            ssim_val = float('nan')
        
        blurred_ref = img_obj.get_blurred()
        img_obj.add_PSNR(psnr_val, blurred_ref, algorithm_name)
        img_obj.add_SSIM(ssim_val, blurred_ref, algorithm_name)
        img_obj.add_algorithm(algorithm_name)
        img_obj.add_restored(restored_path, blurred_ref, algorithm_name)
        img_obj.add_kernel(kernel_path, blurred_ref, algorithm_name)
        
        logger.info(f"Optimized: {Path(restored_path).name}")
        logger.info(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
    
    def _save_results(
        self, 
        result: OptimizationResult, 
        algorithm_name: str
    ) -> None:
        """
        Сохранение результатов оптимизации в JSON-файл.
        
        Параметры
        ---------
        result : OptimizationResult
            Результаты оптимизации.
        algorithm_name : str
            Имя алгоритма.
        """
        output_data = {
            'algorithm': algorithm_name,
            'best_params': result.best_params,
            'best_value': result.best_value,
            'n_trials': result.n_trials,
            'elapsed_time': result.elapsed_time
        }
        
        filepath = self.output_folder / f"{algorithm_name}_optimal_params.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Results saved to {filepath}")
