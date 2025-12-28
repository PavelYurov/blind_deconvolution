import algorithms.base as base

from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import optuna
from optuna.samplers import TPESampler, RandomSampler, NSGAIISampler
from optuna.pruners import MedianPruner
import metrics 
import math

import warnings
import os as os
from IPython.display import display
import logging
import json

"""
    над кодом работал:
    Беззаборов А.А
"""

class ProcessingExtension(ABC):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    '''
    Базовый класс для расширения функционала Processing
    '''
    
    def __init__(self, processing_instance):
        self.processing = processing_instance
        self.save_folder = 'parametrs'
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        '''
        Основной метод выполнения
        '''
        pass


try:
    from optuna.integration import BoTorchSampler
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    
class OptimizationMethod(Enum):
    TPE = "tpe"  # Tree-structured Parzen Estimator (Байесовская оптимизация)
    RANDOM = "random"  # Random Search
    GP = "gp"  # Gaussian Process (Байесовская оптимизация)
    GA = "ga"  # Genetic Algorithm (Эволюционный алгоритм)

class HyperparameterOptimizer(ProcessingExtension):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    '''
    Класс для оптимизации гиперпараметров
    '''
    
    def execute(self, algorithm_processor: base.DeconvolutionAlgorithm, 
                param_ranges: dict, n_trials: int = 50, 
                metric: str = 'PSNR', timeout: int = 3600,
                method: OptimizationMethod = OptimizationMethod.TPE,
                logs: bool = True,
                **optimization_kwargs):
            '''
            Оптимизация гиперпараметров с использованием Optuna
            
            Аргументы:
                - algorithm_processor: метод восстановления изображения
                - param_ranges: словарь с диапазонами параметров {param: (min, max)}
                - n_trials: количество испытаний
                - metric: метрика для оптимизации ('PSNR', 'SSIM', 'Sharpness')
                - timeout: максимальное время оптимизации в секундах
                - method: метод оптимизации (TPE, Random, GP, GA)
                - logs: выводить логи или нет
                - optimization_kwargs: дополнительные параметры для оптимизатора
            '''
            alg_name = algorithm_processor.get_name()
            
            best_params = self._optimize_with_optuna(
                    algorithm_processor, param_ranges, n_trials, metric, timeout, method, logs, optimization_kwargs
                )
            for img_obj in self.processing.images:
                original_image = img_obj.get_original_image()
                blurred_image = img_obj.get_blurred_image()
                if blurred_image is None:
                    blurred_image = original_image

                self._process_with_optimized_params(
                        img_obj, algorithm_processor, best_params, 
                        blurred_image, original_image, alg_name
                    )
            self.save_to_json(best_params,alg_name)
            
    def _optimize_with_optuna(self, algorithm_processor, param_ranges, n_trials, metric, timeout, method, logs, optimization_kwargs):
        '''
        Оптимизация гиперпараметров с помощью Optuna
        '''
        
        def objective(trial):
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                elif isinstance(min_val, float) or isinstance(max_val, float):
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_categorical(param_name, [min_val, max_val]) #по идее это работать не будет, если только не поменять на что-то умное
            
            algorithm_processor.change_param(params)
            score = 0
            for img_obj in self.processing.images:

                original_image = img_obj.get_original_image()
                blurred_image = img_obj.get_blurred_image()
                if blurred_image is None:
                    blurred_image = original_image

                try:
                    test_restored, _ = algorithm_processor.process(blurred_image)
                    original_prepared = self._prepare_image_for_metric(original_image)
                    restored_prepared = self._prepare_image_for_metric(test_restored)

                    if np.max(test_restored) < 1e-6:
                        print("WARNING: Restored image is almost zero!")
                        score += 0
                    if metric.upper() == 'SSIM':
                        score += metrics.SSIM(original_prepared, restored_prepared,data_range=1.0)
                    elif metric.upper() == 'SHARPNESS':
                        score += metrics.Sharpness(restored_prepared)
                    else:  # PSNR по умолчанию
                        score += metrics.PSNR(original_prepared, restored_prepared)
                    
                    
                except Exception as e:
                    print(f"Error while testing parameters {params}: {e}")
                    score +=0
            
            #пусть будет среднее...
            #потом можно будет переделать на что-то умное, я не знаю на что
            score = score / len(self.processing.images)
            return score
        
        if not logs:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner()  # Prunning для ранней остановки
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=logs)
        
        print(f"Optimization completed:")
        print(f"   Best {metric}: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        print(f"   Number of tests: {len(study.trials)}")
        
        return study.best_params

    def save_to_json(self, best_params, alg_name):
        """
        сохраняет параметры в .json
        """
        os.makedirs(self.save_folder, exist_ok=True)
        filename = alg_name + '.json'
        file_path = os.path.join(self.save_folder,filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, ensure_ascii=False, indent=4)


    # def execute(self, algorithm_processor: base.DeconvolutionAlgorithm, 
    #             param_ranges: dict, n_trials: int = 50, 
    #             metric: str = 'PSNR', timeout: int = 3600,
    #             method: OptimizationMethod = OptimizationMethod.TPE,
    #             logs: bool = True,
    #             **optimization_kwargs):
    #         '''
    #         Оптимизация гиперпараметров с использованием Optuna
            
    #         Аргументы:
    #             - algorithm_processor: метод восстановления изображения
    #             - param_ranges: словарь с диапазонами параметров {param: (min, max)}
    #             - n_trials: количество испытаний
    #             - metric: метрика для оптимизации ('PSNR', 'SSIM', 'Sharpness')
    #             - timeout: максимальное время оптимизации в секундах
    #             - method: метод оптимизации (TPE, Random, GP, GA)
    #             - logs: выводить логи или нет
    #             - optimization_kwargs: дополнительные параметры для оптимизатора
    #         '''
    #         alg_name = algorithm_processor.get_name()
            
    #         for img_obj in self.processing.images:
    #             self._optimize_and_process_single_image(
    #                 img_obj, algorithm_processor, param_ranges, 
    #                 n_trials, metric, timeout, alg_name, method, logs, optimization_kwargs
    #             )

    # def _optimize_and_process_single_image(self, img_obj, algorithm_processor, 
    #                                     param_ranges, n_trials, metric, 
    #                                     timeout, alg_name, method, logs, optimization_kwargs):
    #     '''
    #     Оптимизация и обработка одного изображения
    #     '''
    #     original_image = img_obj.get_original_image()
    #     blurred_image = img_obj.get_blurred_image()
        
    #     if blurred_image is None:
    #         blurred_image = original_image
            
    #     if blurred_image is None:
    #         print(f"Pass: Unable to load image {img_obj.get_original()}")
    #         return
        
    #     print(f"Oprimizig hyperparameters for {Path(img_obj.get_original()).name}")
        
    #     best_params = self._optimize_with_optuna(
    #             algorithm_processor, param_ranges, blurred_image, 
    #             original_image, n_trials, metric, timeout, method, logs, optimization_kwargs
    #         )
        
    #     self._process_with_optimized_params(
    #             img_obj, algorithm_processor, best_params, 
    #             blurred_image, original_image, alg_name
    #         )

    # def _optimize_with_optuna(self, algorithm_processor, param_ranges, 
    #                             blurred_image, original_image, n_trials, 
    #                             metric, timeout, method, logs, optimization_kwargs):
    #     '''
    #     Оптимизация гиперпараметров с помощью Optuna
    #     '''
        
    #     def objective(trial):
    #         params = {}
    #         for param_name, (min_val, max_val) in param_ranges.items():
    #             if isinstance(min_val, int) and isinstance(max_val, int):
    #                 params[param_name] = trial.suggest_int(param_name, min_val, max_val)
    #             elif isinstance(min_val, float) or isinstance(max_val, float):
    #                 params[param_name] = trial.suggest_float(param_name, min_val, max_val)
    #             else:
    #                 params[param_name] = trial.suggest_categorical(param_name, [min_val, max_val]) #по идее это работать не будет, если только не поменять на что-то умное
            
    #         algorithm_processor.change_param(params)
            
    #         try:
    #             test_restored, _ = algorithm_processor.process(blurred_image)
    #             original_prepared = self._prepare_image_for_metric(original_image)
    #             restored_prepared = self._prepare_image_for_metric(test_restored)

    #             if np.max(test_restored) < 1e-6:
    #                 print("WARNING: Restored image is almost zero!")
    #                 return -float('inf')
    #             if metric.upper() == 'SSIM':
    #                 score = metrics.SSIM(original_prepared, restored_prepared)
    #             elif metric.upper() == 'SHARPNESS':
    #                 score = metrics.Sharpness(restored_prepared)
    #             else:  # PSNR по умолчанию
    #                 score = metrics.PSNR(original_prepared, restored_prepared)
    #             return score
                
    #         except Exception as e:
    #             print(f"Error while testing parameters {params}: {e}")
    #             return -float('inf')
        
    #     if not logs:
    #         optuna.logging.set_verbosity(optuna.logging.WARNING)
    #     study = optuna.create_study(
    #         direction="maximize",
    #         sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    #         pruner=optuna.pruners.MedianPruner()  # Prunning для ранней остановки
    #     )
        
    #     study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=logs)
        
    #     print(f"Optimization completed:")
    #     print(f"   Best {metric}: {study.best_value:.4f}")
    #     print(f"   Best params: {study.best_params}")
    #     print(f"   Number of tests: {len(study.trials)}")
        
    #     return study.best_params


    def _prepare_image_for_metric(self, image):
        """
        Подготовка изображения для расчета метрик с нормализацией
        """
        image = np.array(image.copy(), dtype=np.float32)

        
        # Нормализуем в диапазон [0, 1] если нужно
        if image.max() > 1.0:
            image = image / 255.0
        
        # # Обеспечиваем 3 канала для цветных изображений
        # if len(image.shape) == 2:
        #     image = np.stack([image] * 3, axis=-1)
        # elif image.shape[2] == 1:
        #     image = np.repeat(image, 3, axis=2)

        if len(image.shape) == 3:
            image = np.dot(image[...,:3], [1.0, 0.0, 0.0])
        
        image = np.clip(image, 0.0, 1.0)
        
        return image

    def _process_with_optimized_params(self, img_obj, algorithm_processor, best_params,
                                    blurred_image, original_image, alg_name):
        '''
        Обработка изображения с оптимизированными параметрами
        '''
        
        algorithm_processor.change_param(best_params)
        
        print("Best params:")
        for param_name, param_value in algorithm_processor.get_param():
            print(f"   {param_name}: {param_value}")
        
        try:
            restored_image, kernel = algorithm_processor.process(blurred_image)
            
            restored_path, kernel_path = self._generate_optimized_output_paths(img_obj, alg_name)
            
            print(np.max(restored_image))

            cv.imwrite(restored_path, restored_image)
            cv.imwrite(kernel_path, kernel)
            
            self._calculate_and_save_optimized_metrics(img_obj, original_image, restored_image,
                                                    restored_path, kernel_path, alg_name)
            
        except Exception as e:
            print(f"Error recovering with optimized parameters: {e}")

    def _generate_optimized_output_paths(self, img_obj, alg_name):
        '''
        Генерация путей для оптимизированных результатов
        '''
        if img_obj.get_blurred():
            base_path = Path(img_obj.get_blurred())
        else:
            base_path = Path(img_obj.get_original())
        
        base_name = base_path.stem
        suffix = base_path.suffix
        
        restored_path = self.processing._generate_unique_file_path(
            self.processing.folder_path_restored,
            f"{base_name}_{alg_name}_optimized{suffix}"
        )
        
        kernel_path = self.processing._generate_unique_file_path(
            self.processing.folder_path_restored,
            f"{base_name}_{alg_name}_optimized_kernel{suffix}"
        )
        
        return str(restored_path), str(kernel_path)

    def _calculate_and_save_optimized_metrics(self, img_obj, original_image, restored_image,
                                            restored_path, kernel_path, alg_name):
        '''
        Расчет метрик для оптимизированного результата
        '''
        original_prepared = self._prepare_image_for_metric(original_image)
        restored_prepared = self._prepare_image_for_metric(restored_image)
        try:
            psnr_val = metrics.PSNR(original_prepared, restored_prepared)
        except Exception as e:
            print(f"Error calculating PSNR: {e}")
            psnr_val = math.nan
        
        try:
            ssim_val = metrics.SSIM(original_prepared, restored_prepared,data_range=1.0)
        except Exception as e:
            print(f"Error calculating SSIM: {e}")
            ssim_val = math.nan
        
        blurred_ref = img_obj.get_blurred()
        img_obj.add_PSNR(psnr_val, blurred_ref, alg_name)
        img_obj.add_SSIM(ssim_val, blurred_ref, alg_name)
        img_obj.add_algorithm(alg_name)
        img_obj.add_restored(restored_path, blurred_ref, alg_name)
        img_obj.add_kernel(kernel_path, blurred_ref, alg_name)
        
        print(f"Optimized: {Path(restored_path).name}")
        print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        

class ParetoFrontAnalyzer(ProcessingExtension):
    """
        над кодом работал:
        Беззаборов А.А.
    """
    """
    Класс для анализа результатов с построением фронта Парето
    """
    
    def execute(self):
        '''
        Анализ результатов с помощью построения фронта Парето
        Оптимизация по качеству восстановления (PSNR), степени размытия, интенсивности шума
        '''
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            from scipy import stats, interpolate
            from mpl_toolkits.mplot3d import Axes3D
            warnings.filterwarnings('ignore')
            
            # Сбор данных для анализа
            analysis_data = self._collect_pareto_data()
            
            if not analysis_data:
                print("No data for analysis")
                return
            
            df = pd.DataFrame(analysis_data)
            print(f"Collected data: {len(df)} records")
            
            # Проверяем, достаточно ли данных для 3D анализа
            if len(df) >= 20: 
                self._plot_3d_pareto_fronts(df)
            else:
                print(f"Insufficient data for 3D Pareto fronts. Need at least 20 points, got {len(df)}")
            
            # Дополнительный 2D анализ
            self._plot_comprehensive_pareto_analysis(df)
            self._multi_criteria_analysis(df)
            self._statistical_analysis(df)
            self._algorithm_comparison(df)
            
        except ImportError as e:
            print(f"Missing required libraries: {e}")
        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback
            traceback.print_exc()

    def _plot_3d_pareto_fronts(self, df):
        """
        Построение трехмерные Парето фронтов для каждого алгоритма
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from scipy.interpolate import griddata, Rbf
            
            algorithms = df['algorithm'].unique()
            
            # Создаем 3D визуализацию
            fig = plt.figure(figsize=(20, 15))
            
            # График 1: 3D поверхности Парето
            ax1 = fig.add_subplot(231, projection='3d')
            self._plot_3d_surfaces(ax1, df, algorithms)
            
            # График 2: 2D проекция - Шум vs PSNR
            ax2 = fig.add_subplot(232)
            self._plot_2d_projection(ax2, df, 'noise_intensity', 'psnr', 
                                   'Noise Intensity', 'PSNR (dB)', 'Noise vs PSNR')
            
            # График 3: 2D проекция - Смаз vs PSNR
            ax3 = fig.add_subplot(233)
            self._plot_2d_projection(ax3, df, 'blur_intensity', 'psnr',
                                   'Blur Intensity', 'PSNR (dB)', 'Blur vs PSNR')
            
            # График 4: Сравнение алгоритмов по типам фильтров
            ax4 = fig.add_subplot(234)
            self._plot_algorithm_comparison_by_filter(ax4, df)
            
            # График 5: Тепловая карта производительности
            ax5 = fig.add_subplot(235)
            self._plot_performance_heatmap(ax5, df)
            
            # График 6: Улучшение качества
            ax6 = fig.add_subplot(236)
            self._plot_quality_improvement(ax6, df)
            
            plt.tight_layout()
            plt.savefig('3d_pareto_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Количественный анализ
            self._quantitative_benchmark_analysis(df)
            
        except Exception as e:
            print(f"Error in 3D Pareto front plotting: {e}")

    def _plot_3d_surfaces(self, ax, df, algorithms):
        """
        Построение трехмерных поверхностей Парето для каждого алгоритма
        """
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, algorithm in enumerate(algorithms):
            algorithm_data = df[df['algorithm'] == algorithm]
            
            if len(algorithm_data) < 5: 
                continue
                
            # Извлекаем координаты
            x = algorithm_data['noise_intensity'].values  # Шум
            y = algorithm_data['blur_intensity'].values   # Смаз
            z = algorithm_data['psnr'].values             # PSNR (качество)
            
            # Создаем сетку для интерполяции
            xi = np.linspace(x.min(), x.max(), 20)
            yi = np.linspace(y.min(), y.max(), 20)
            xi, yi = np.meshgrid(xi, yi)
            
            try:
                # Интерполяция с использованием радиальных базисных функций
                rbf = Rbf(x, y, z, function='thin_plate', smooth=0.1)
                zi = rbf(xi, yi)
                
                # Строим поверхность
                surface = ax.plot_surface(xi, yi, zi, alpha=0.6, 
                                        color=colors[i], label=algorithm)
                surface._facecolors2d = surface._facecolors3d
                surface._edgecolors2d = surface._edgecolors3d
                
            except Exception as e:
                # Резервный вариант - точечный график если интерполяция не удалась
                print(f"Interpolation failed for {algorithm}, using scatter plot: {e}")
                ax.scatter(x, y, z, c=[colors[i]], label=algorithm, s=50, alpha=0.8)
        
        ax.set_xlabel('Noise Intensity')
        ax.set_ylabel('Blur Intensity')
        ax.set_zlabel('PSNR (dB)')
        ax.set_title('3D Pareto Fronts: Noise-Blur-Quality')
        ax.legend()

    def _plot_2d_projection(self, ax, df, x_metric, y_metric, x_label, y_label, title):
        """Plot 2D projections of Pareto fronts"""
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, algorithm in enumerate(algorithms):
            algorithm_data = df[df['algorithm'] == algorithm]
            x = algorithm_data[x_metric].values
            y = algorithm_data[y_metric].values
            
            if len(x) > 2:  # Need points for interpolation
                try:
                    # Сортируем для правильного построения линий
                    sort_idx = np.argsort(x)
                    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
                    
                    # Простая интерполяция для гладкой линии
                    from scipy.interpolate import interp1d
                    f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
                    x_interp = np.linspace(x_sorted.min(), x_sorted.max(), 100)
                    y_interp = f(x_interp)
                    
                    ax.plot(x_interp, y_interp, color=colors[i], label=algorithm, linewidth=2)
                    ax.scatter(x_sorted, y_sorted, color=colors[i], alpha=0.6, s=30)
                    
                except Exception as e:
                    print(f"2D interpolation failed for {algorithm}: {e}")
                    ax.scatter(x, y, color=colors[i], label=algorithm, alpha=0.7, s=50)
            else:
                ax.scatter(x, y, color=colors[i], label=algorithm, alpha=0.7, s=50)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_algorithm_comparison_by_filter(self, ax, df):
        """
        Сравнение алгоритмов по типу фильтра
        """
        if 'filter' not in df.columns:
            print("No filter data available for comparison")
            return
            
        filter_types = df['filter'].unique()
        algorithms = df['algorithm'].unique()
        
        # Группируем по фильтрам и алгоритмам
        comparison_data = []
        for filter_type in filter_types:
            for algorithm in algorithms:
                filter_data = df[(df['filter'] == filter_type) & (df['algorithm'] == algorithm)]
                if len(filter_data) > 0:
                    avg_psnr = filter_data['psnr'].mean()
                    comparison_data.append({'filter': filter_type, 'algorithm': algorithm, 'psnr': avg_psnr})
        
        if not comparison_data:
            print("No comparison data available")
            return
            
        comp_df = pd.DataFrame(comparison_data)
        
        # Создаем группированную столбчатую диаграмму
        x_pos = np.arange(len(filter_types))
        bar_width = 0.8 / len(algorithms)
        
        for i, algorithm in enumerate(algorithms):
            algorithm_psnr = []
            for filter_type in filter_types:
                data = comp_df[(comp_df['filter'] == filter_type) & (comp_df['algorithm'] == algorithm)]
                algorithm_psnr.append(data['psnr'].values[0] if len(data) > 0 else 0)
            
            ax.bar(x_pos + i * bar_width, algorithm_psnr, bar_width, label=algorithm)
        
        ax.set_xlabel('Filter Type')
        ax.set_ylabel('Average PSNR (dB)')
        ax.set_title('Algorithm Performance by Filter Type')
        ax.set_xticks(x_pos + bar_width * len(algorithms) / 2)
        ax.set_xticklabels(filter_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_heatmap(self, ax, df):
        """Create performance heatmap"""
        try:
            # Сводная таблица для тепловой карты
            pivot_table = df.pivot_table(values='psnr', 
                                       index='algorithm', 
                                       columns='filter', 
                                       aggfunc='mean')
            
            im = ax.imshow(pivot_table.values, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(pivot_table.columns)))
            ax.set_yticks(range(len(pivot_table.index)))
            ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
            ax.set_yticklabels(pivot_table.index)
            ax.set_title('Performance Heatmap (PSNR)')
            plt.colorbar(im, ax=ax, label='PSNR (dB)')
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")

    def _plot_quality_improvement(self, ax, df):
        """
        Улучшение качества по сравнению с оригиналом
        """
        if 'original_psnr' not in df.columns:
            print("No original PSNR data available")
            return
            
        algorithms = df['algorithm'].unique()
        improvements = []
        
        for algorithm in algorithms:
            algorithm_data = df[df['algorithm'] == algorithm]
            improvement = (algorithm_data['psnr'] - algorithm_data['original_psnr']).mean()
            improvements.append(improvement)
        
        bars = ax.bar(range(len(algorithms)), improvements, color=plt.cm.Set3(np.linspace(0, 1, len(algorithms))))
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Average PSNR Improvement (dB)')
        ax.set_title('Quality Improvement from Blurred Image')
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        
        # Добавляем подписи значений на столбцах
        for bar, improvement in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{improvement:.1f}dB', ha='center', va='bottom')

    def _quantitative_benchmark_analysis(self, df):
        """
        Количественный сравнительный анализ
        """
        
        # Общая производительность
        print("\nOverall performance by algorithm:")
        overall_stats = df.groupby('algorithm')['psnr'].agg(['mean', 'std', 'count'])
        print(overall_stats.round(3))
        
        # Производительность по типам фильтров
        if 'filter' in df.columns:
            print("\nPerformance by filter type:")
            filter_stats = df.groupby(['algorithm', 'filter'])['psnr'].mean().unstack()
            print(filter_stats.round(3))
        
        # Лучший алгоритм для разных условий
        print("\nBest algorithm for specific conditions:")
        
        # Условия с высоким шумом
        high_noise_threshold = df['noise_intensity'].quantile(0.75)
        high_noise_data = df[df['noise_intensity'] > high_noise_threshold]
        if len(high_noise_data) > 0:
            best_high_noise = high_noise_data.groupby('algorithm')['psnr'].mean().idxmax()
            print(f"High noise conditions: {best_high_noise}")
        else:
            print("No high noise data available")
        
        high_blur_threshold = df['blur_intensity'].quantile(0.75)
        high_blur_data = df[df['blur_intensity'] > high_blur_threshold]
        if len(high_blur_data) > 0:
            best_high_blur = high_blur_data.groupby('algorithm')['psnr'].mean().idxmax()
            print(f"High blur conditions: {best_high_blur}")
        else:
            print("No high blur data available")

    def _collect_pareto_data(self):
        """
        Сбор данных для анализа по Парето с оценкой размытости, шума и качества
        """
        analysis_data = []
        
        for img in self.processing.images:
            try:
                original_image = img.get_original_image()
                if original_image is None:
                    continue
                    
                blurred_array = img.get_blurred_array()
                algorithms = img.get_algorithm()
                
                for blurred_path in blurred_array:
                    blurred_image = self._load_image_for_analysis(blurred_path)
                    if blurred_image is None:
                        print(f"Failed to load blurred image: {blurred_path}")
                        continue
                    
                    blur_intensity = self._calculate_blur_intensity(blurred_image)
                    noise_intensity = self._calculate_noise_intensity(blurred_image)
                    
                    blurred_psnr = img.get_blurred_PSNR().get(str(blurred_path), math.nan)
                    blurred_ssim = img.get_blurred_SSIM().get(str(blurred_path), math.nan)
                    
                    for alg_name in algorithms:
                        restored_psnr = img.get_PSNR().get((str(blurred_path), str(alg_name)), math.nan)
                        restored_ssim = img.get_SSIM().get((str(blurred_path), str(alg_name)), math.nan)
                        
                        restored_path = img.get_restored().get((str(blurred_path), str(alg_name)))
                        restoration_quality = math.nan
                        
                        if restored_path and os.path.exists(restored_path):
                            restored_image = self._load_image_for_analysis(restored_path)
                            if restored_image is not None:
                                restoration_quality = self._calculate_restoration_quality(
                                    original_image, blurred_image, restored_image
                                )
                            else:
                                print(f"Failed to load restored image: {restored_path}")
                        
                        process_time = math.nan
                        if hasattr(img, 'get_process_time'):
                            try:
                                time_data = img.get_process_time()
                                if isinstance(time_data, dict):
                                    process_time = time_data.get((str(blurred_path), str(alg_name)), math.nan)
                                else:
                                    process_time = time_data
                            except Exception as e:
                                print(f"Error getting process time: {e}")
                        
                        if (not math.isnan(restored_psnr) and not math.isnan(blur_intensity) and
                            not math.isnan(noise_intensity)):
                            
                            analysis_data.append({
                                'image': os.path.basename(img.get_original()),
                                'algorithm': alg_name,
                                'filter': img.get_filters().get(str(blurred_path), 'unknown'),
                                'blur_intensity': blur_intensity,      
                                'noise_intensity': noise_intensity,    
                                'psnr': restored_psnr,              
                                'ssim': restored_ssim,     
                                'restoration_quality': restoration_quality,  
                                'psnr_improvement': restored_psnr - blurred_psnr,
                                'ssim_improvement': restored_ssim - blurred_ssim,
                                'time': process_time,
                                'original_psnr': blurred_psnr
                            })
                        else:
                            print(f"Skipping invalid data for {alg_name} on {blurred_path}")
                            
            except Exception as e:
                print(f"Error processing image {img.get_original()}: {e}")
                continue
        
        return analysis_data

    def _load_image_for_analysis(self, image_path):
        """
        Загрузка изображения для анализа
        """
        try:
            if not os.path.exists(image_path):
                print(f"Image path does not exist: {image_path}")
                return None
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {image_path}")
                return None
            return image.astype(np.float32)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def _calculate_blur_intensity(self, image):
        """
        Рассчет интенсивности размытия, используя дисперсию Лапласа
        Высокие значения = меньшее размытие, низкие значения = большее размытие
        """
        try:
            laplacian_var = cv.Laplacian(image.astype(np.uint8), cv.CV_64F).var()
            return min(laplacian_var / 1000.0, 1.0) 
        except Exception as e:
            print(f"Error calculating blur intensity: {e}")
            return math.nan

    def _calculate_noise_intensity(self, image):
        """
        Рассчет интенсивности шума, используя стандартное отклонение в гладких областях
        """
        try:
            h, w = image.shape
            block_size = 32
            std_devs = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    std_devs.append(np.std(block))

            avg_std = np.mean(std_devs) if std_devs else 0
            return min(avg_std / 50.0, 1.0)
        except Exception as e:
            print(f"Error calculating noise intensity: {e}")
            return math.nan

    def _calculate_restoration_quality(self, original, blurred, restored):
        """
        Всесторонняя оценка качества восстановления
        Учитывает улучшение по сравнению с размытыми изображениями и приближение к оригиналу
        """
        try:
            psnr_to_original = metrics.PSNR(original, restored)
            psnr_to_blurred = metrics.PSNR(blurred, restored)
            
            quality = min((psnr_to_original / 50.0 + psnr_to_blurred / 30.0) / 2, 1.0)
            return quality
        except Exception as e:
            print(f"Error calculating restoration quality: {e}")
            return math.nan

    def _plot_comprehensive_pareto_analysis(self, df):
        """
        Построение графиков комплексного анализа по Парето
        """
        import matplotlib.pyplot as plt
        
        pareto_combinations = [
            ('psnr', 'time', 'PSNR (dB)', 'Execution Time (sec)', 
             'Pareto Front: Quality vs Speed'),
            
            ('psnr', 'blur_intensity', 'PSNR (dB)', 'Blur Intensity', 
             'Pareto Front: Quality vs Blur Level'),
            
            ('psnr', 'noise_intensity', 'PSNR (dB)', 'Noise Intensity', 
             'Pareto Front: Quality vs Noise Level'),
            
            ('restoration_quality', 'time', 'Restoration Quality', 'Execution Time (sec)',
             'Pareto Front: Complex Quality vs Speed')
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (x_metric, y_metric, x_label, y_label, title) in enumerate(pareto_combinations):
            if idx >= len(axes):
                break
                
            if x_metric in df.columns and y_metric in df.columns:
                valid_data = df[[x_metric, y_metric, 'algorithm']].dropna()
                
                if len(valid_data) > 0:
                    self._plot_single_pareto_front(
                        valid_data, x_metric, y_metric, x_label, y_label, title, axes[idx]
                    )
                else:
                    print(f"No valid data for {title}")
        
        plt.tight_layout()
        plt.savefig('comprehensive_pareto_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_single_pareto_front(self, df, x_metric, y_metric, x_label, y_label, title, ax):
        """
        Построение одного графика Парето фронта
        """
        try:
            maximize_x = x_metric in ['psnr', 'ssim', 'restoration_quality']
            maximize_y = y_metric in ['psnr', 'ssim', 'restoration_quality']
            
            points = df[[x_metric, y_metric]].values
            pareto_mask = self._find_pareto_points(points, maximize_x, maximize_y)
            pareto_points = df[pareto_mask]
            
            algorithms = df['algorithm'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
            
            for i, algorithm in enumerate(algorithms):
                algorithm_data = df[df['algorithm'] == algorithm]
                ax.scatter(algorithm_data[x_metric], algorithm_data[y_metric], 
                          c=[colors[i]], label=algorithm, alpha=0.7, s=60)
            
            if len(pareto_points) > 0:
                sort_idx = np.argsort(pareto_points[x_metric])
                sorted_pareto = pareto_points.iloc[sort_idx]
                ax.plot(sorted_pareto[x_metric], sorted_pareto[y_metric], 
                       'r--', linewidth=2, label='Pareto Front')
                
                for _, point in pareto_points.iterrows():
                    ax.annotate(f"{point['algorithm']}", 
                               (point[x_metric], point[y_metric]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            else:
                print(f"No Pareto points found for {title}")
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {title}: {e}")

    def _find_pareto_points(self, points, maximize_x=True, maximize_y=True):
        """
        Find Pareto-optimal points
        """
        if maximize_x and maximize_y:
            sorted_indices = np.lexsort((points[:, 1], points[:, 0]))[::-1]
        elif not maximize_x and not maximize_y:
            sorted_indices = np.lexsort((points[:, 1], points[:, 0]))
        elif maximize_x and not maximize_y:
            sorted_indices = np.lexsort((-points[:, 1], points[:, 0]))[::-1]
        else:
            sorted_indices = np.lexsort((points[:, 1], -points[:, 0]))
        
        pareto_mask = np.zeros(len(points), dtype=bool)
        best_y = -np.inf if maximize_y else np.inf
        
        for idx in sorted_indices:
            current_y = points[idx, 1]
            if (maximize_y and current_y >= best_y) or (not maximize_y and current_y <= best_y):
                pareto_mask[idx] = True
                best_y = current_y
        
        return pareto_mask

    def _multi_criteria_analysis(self, df):
        """Multi-criteria analysis"""
        print("\n=== MULTI-CRITERIA ANALYSIS ===")
        
        metrics_to_analyze = ['psnr', 'blur_intensity', 'noise_intensity', 'restoration_quality', 'time']
        
        for metric in metrics_to_analyze:
            if metric in df.columns:
                if metric in ['time', 'noise_intensity', 'blur_intensity']:
                    best = df.loc[df[metric].idxmin()] 
                    print(f"Best by {metric}: {best['algorithm']} = {best[metric]:.3f}")
                else:
                    best = df.loc[df[metric].idxmax()] 
                    print(f"Best by {metric}: {best['algorithm']} = {best[metric]:.3f}")
            else:
                print(f"Metric {metric} not available")

    def _statistical_analysis(self, df):
        """Statistical analysis"""
        
        if 'algorithm' in df.columns:
            grouped = df.groupby('algorithm')
            for metric in ['psnr', 'time', 'restoration_quality']:
                if metric in df.columns:
                    print(f"\n{metric.upper()} by algorithms:")
                    for name, group in grouped:
                        if metric in group.columns:
                            values = group[metric].dropna()
                            if len(values) > 0:
                                print(f"  {name}: mean={values.mean():.3f}, std={values.std():.3f}")
                            else:
                                print(f"  {name}: no data")
                else:
                    print(f"Metric {metric} not available")
        else:
            print("No algorithm data available")

    def _algorithm_comparison(self, df):
        """Сравнение алгоритмов"""
        
        comparison_metrics = ['psnr', 'time', 'restoration_quality']
        available_metrics = [m for m in comparison_metrics if m in df.columns]
        
        if available_metrics and 'algorithm' in df.columns:
            comparison = df.groupby('algorithm')[available_metrics].mean()
            print("\nAverage metrics by algorithms:")
            print(comparison.round(3))
        else:
            print("No data available for algorithm comparison")