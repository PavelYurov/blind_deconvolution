"""
Модуль многокритериального анализа на основе фронта Парето.

Реализует анализ производительности алгоритмов слепой деконволюции
по множеству критериев с построением и визуализацией фронта Парето.

Возможности:
    - 3D визуализация поверхности Парето
    - 2D проекции для попарного анализа критериев
    - Статистическое сравнение алгоритмов
    - Тепловые карты производительности

Автор: Беззаборов А.А.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import cv2 as cv

from processing.extensions.base import ProcessingExtension, logger


class ParetoFrontAnalyzer(ProcessingExtension):
    """
    Многокритериальный анализ производительности с построением фронта Парето.
    
    Предоставляет комплексный анализ производительности алгоритмов
    по нескольким конкурирующим критериям (качество vs скорость,
    устойчивость к шуму vs размытию).
    
    Фронт Парето представляет множество недоминируемых решений, где
    ни один критерий не может быть улучшен без ухудшения другого.
    
    Возможности:
        - 3D визуализация поверхности Парето (шум x размытие x качество)
        - 2D проекции для попарного анализа критериев
        - Статистическое сравнение алгоритмов
        - Тепловые карты производительности
    
    Точка x* является Парето-оптимальной, если не существует другой точки x,
    такой что f_i(x) >= f_i(x*) для всех критериев i, и f_j(x) > f_j(x*) 
    хотя бы для одного критерия j.
    """
    
    # Параметры визуализации
    FIGURE_DPI = 300
    COLORMAP = 'viridis'
    MARKER_SIZE = 60
    LINE_WIDTH = 2
    ALPHA = 0.7
    
    def __init__(
        self, 
        processing_instance: Any, 
        output_folder: str = "pareto_analysis"
    ):
        """
        Инициализация анализатора Парето.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing.
        output_folder : str
            Директория для сохранения результатов анализа.
        """
        super().__init__(processing_instance, output_folder)
        
        # Подавление предупреждений matplotlib
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Настройка стиля графиков
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except OSError:
            plt.style.use('ggplot')
    
    def execute(self, save_figures: bool = True) -> pd.DataFrame:
        """
        Выполнение комплексного анализа фронта Парето.
        
        Параметры
        ---------
        save_figures : bool, по умолчанию True
            Сохранять сгенерированные графики.
            
        Возвращает
        ----------
        pd.DataFrame
            DataFrame со всеми данными анализа.
        """
        logger.info("=" * 60)
        logger.info("PARETO FRONT ANALYSIS")
        logger.info("=" * 60)
        
        # Сбор данных
        data = self._collect_analysis_data()
        
        if not data:
            logger.error("No data available for analysis")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.info(f"Collected {len(df)} data points for analysis")
        
        # Генерация визуализаций
        if len(df) >= 10:
            self._plot_3d_pareto_surfaces(df, save_figures)
        else:
            logger.warning(
                f"Insufficient data for 3D analysis ({len(df)} points, need >= 10)"
            )
        
        self._plot_2d_pareto_fronts(df, save_figures)
        self._plot_algorithm_comparison(df, save_figures)
        self._plot_performance_heatmap(df, save_figures)
        
        # Статистический анализ
        self._print_statistical_summary(df)
        self._print_pareto_optimal_solutions(df)
        
        # Сохранение данных в CSV
        if save_figures:
            csv_path = self.output_folder / 'analysis_data.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"Data saved to {csv_path}")
        
        return df
    
    def _collect_analysis_data(self) -> List[Dict[str, Any]]:
        """
        Сбор данных производительности из обработанных изображений.
        
        Возвращает
        ----------
        List[Dict]
            Список словарей с метриками производительности.
        """
        analysis_data = []
        
        for img in self.processing.images:
            try:
                original = img.get_original_image()
                if original is None:
                    continue
                
                blurred_array = img.get_blurred_array()
                algorithms = img.get_algorithm()
                
                for blurred_path in blurred_array:
                    blurred = self._load_image(blurred_path)
                    if blurred is None:
                        continue
                    
                    # Вычисление характеристик деградации
                    blur_intensity = self._compute_blur_intensity(blurred)
                    noise_intensity = self._compute_noise_intensity(blurred)
                    
                    blurred_psnr = img.get_blurred_PSNR().get(str(blurred_path), np.nan)
                    blurred_ssim = img.get_blurred_SSIM().get(str(blurred_path), np.nan)
                    
                    for alg_name in algorithms:
                        restored_psnr = img.get_PSNR().get(
                            (str(blurred_path), str(alg_name)), np.nan
                        )
                        restored_ssim = img.get_SSIM().get(
                            (str(blurred_path), str(alg_name)), np.nan
                        )
                        
                        # Получение времени обработки при наличии
                        process_time = self._get_process_time(
                            img, blurred_path, alg_name
                        )
                        
                        # Пропуск некорректных данных
                        if np.isnan(restored_psnr):
                            continue
                        
                        analysis_data.append({
                            'image': os.path.basename(img.get_original()),
                            'algorithm': alg_name,
                            'filter': img.get_filters().get(str(blurred_path), 'unknown'),
                            'blur_intensity': blur_intensity,
                            'noise_intensity': noise_intensity,
                            'psnr': restored_psnr,
                            'ssim': restored_ssim,
                            'psnr_improvement': restored_psnr - blurred_psnr,
                            'ssim_improvement': restored_ssim - blurred_ssim,
                            'original_psnr': blurred_psnr,
                            'time': process_time
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing {img.get_original()}: {e}")
                continue
        
        return analysis_data
    
    def _load_image(self, path: str) -> Optional[np.ndarray]:
        """
        Загрузка изображения по указанному пути.
        
        Параметры
        ---------
        path : str
            Путь к файлу изображения.
            
        Возвращает
        ----------
        Optional[np.ndarray]
            Загруженное изображение или None при ошибке.
        """
        try:
            if not os.path.exists(path):
                return None
            image = cv.imread(path, cv.IMREAD_GRAYSCALE)
            return image.astype(np.float32) if image is not None else None
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None
    
    def _compute_blur_intensity(self, image: np.ndarray) -> float:
        """
        Вычисление интенсивности размытия через дисперсию лапласиана.
        
        Дисперсия лапласиана является мерой резкости изображения.
        Низкие значения указывают на большее размытие.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение.
            
        Возвращает
        ----------
        float
            Нормализованная интенсивность размытия в диапазоне [0, 1].
        """
        try:
            laplacian = cv.Laplacian(image.astype(np.uint8), cv.CV_64F)
            variance = laplacian.var()
            # Нормализация: высокая дисперсия = меньше размытия, инвертируем
            return 1.0 - min(variance / 1000.0, 1.0)
        except Exception:
            return np.nan
    
    def _compute_noise_intensity(self, image: np.ndarray) -> float:
        """
        Оценка интенсивности шума по гладким областям изображения.
        
        Использует блочную оценку стандартного отклонения в
        предположительно гладких областях изображения.
        
        Параметры
        ---------
        image : np.ndarray
            Входное изображение.
            
        Возвращает
        ----------
        float
            Нормализованная интенсивность шума в диапазоне [0, 1].
        """
        try:
            h, w = image.shape
            block_size = 32
            std_values = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = image[i:i+block_size, j:j+block_size]
                    # MAD (медианное абсолютное отклонение) для робастной оценки
                    median = np.median(block)
                    mad = np.median(np.abs(block - median))
                    std_values.append(mad * 1.4826)  # Масштабирование к эквиваленту std
            
            if not std_values:
                return np.nan
            
            # Используем минимальное std как оценку шума (самая гладкая область)
            noise_estimate = np.percentile(std_values, 10)
            return min(noise_estimate / 50.0, 1.0)
            
        except Exception:
            return np.nan
    
    def _get_process_time(
        self, 
        img: Any, 
        blurred_path: str, 
        alg_name: str
    ) -> float:
        """
        Получение времени обработки при наличии.
        
        Параметры
        ---------
        img : Any
            Объект изображения.
        blurred_path : str
            Путь к размытому изображению.
        alg_name : str
            Имя алгоритма.
            
        Возвращает
        ----------
        float
            Время обработки или NaN при отсутствии данных.
        """
        if not hasattr(img, 'get_process_time'):
            return np.nan
        
        try:
            time_data = img.get_process_time()
            if isinstance(time_data, dict):
                return time_data.get((str(blurred_path), str(alg_name)), np.nan)
            return float(time_data)
        except Exception:
            return np.nan
    
    def _find_pareto_front(
        self,
        points: np.ndarray,
        maximize: List[bool]
    ) -> np.ndarray:
        """
        Поиск Парето-оптимальных точек.
        
        Использует алгоритм недоминируемой сортировки.
        
        Параметры
        ---------
        points : np.ndarray
            Массив размера (n_points, n_objectives).
        maximize : List[bool]
            Направление оптимизации для каждого критерия.
            
        Возвращает
        ----------
        np.ndarray
            Булева маска, указывающая на Парето-оптимальные точки.
        """
        n_points = len(points)
        is_pareto = np.ones(n_points, dtype=bool)
        
        # Преобразование к задаче максимизации
        points_max = points.copy()
        for i, m in enumerate(maximize):
            if not m:
                points_max[:, i] = -points_max[:, i]
        
        for i in range(n_points):
            if not is_pareto[i]:
                continue
            
            for j in range(n_points):
                if i == j or not is_pareto[j]:
                    continue
                
                # Проверка доминирования j над i
                if np.all(points_max[j] >= points_max[i]) and \
                   np.any(points_max[j] > points_max[i]):
                    is_pareto[i] = False
                    break
        
        return is_pareto
    
    def _plot_3d_pareto_surfaces(
        self, 
        df: pd.DataFrame, 
        save: bool = True
    ) -> None:
        """
        Построение 3D визуализации поверхности Парето.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        save : bool
            Сохранять график в файл.
        """
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import Rbf
        
        fig = plt.figure(figsize=(20, 16))
        
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        
        # График 1: 3D точки с поверхностями
        ax1 = fig.add_subplot(221, projection='3d')
        
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm'] == alg]
            
            if len(alg_data) < 4:
                ax1.scatter(
                    alg_data['noise_intensity'],
                    alg_data['blur_intensity'],
                    alg_data['psnr'],
                    c=[colors[i]], label=alg, s=self.MARKER_SIZE, alpha=0.8
                )
                continue
            
            x = alg_data['noise_intensity'].values
            y = alg_data['blur_intensity'].values
            z = alg_data['psnr'].values
            
            try:
                # Создание интерполированной поверхности
                xi = np.linspace(x.min(), x.max(), 15)
                yi = np.linspace(y.min(), y.max(), 15)
                xi, yi = np.meshgrid(xi, yi)
                
                rbf = Rbf(x, y, z, function='thin_plate', smooth=0.5)
                zi = rbf(xi, yi)
                
                ax1.plot_surface(
                    xi, yi, zi, alpha=0.4, color=colors[i]
                )
                ax1.scatter(x, y, z, c=[colors[i]], label=alg, s=30, alpha=0.9)
                
            except Exception as e:
                logger.debug(f"Surface interpolation failed for {alg}: {e}")
                ax1.scatter(x, y, z, c=[colors[i]], label=alg, s=self.MARKER_SIZE)
        
        ax1.set_xlabel('Noise Intensity', fontsize=12)
        ax1.set_ylabel('Blur Intensity', fontsize=12)
        ax1.set_zlabel('PSNR (dB)', fontsize=12)
        ax1.set_title('3D Pareto Surface: Degradation vs Quality', fontsize=14)
        ax1.legend(loc='upper left', fontsize=10)
        
        # График 2: Поверхность улучшения PSNR
        ax2 = fig.add_subplot(222, projection='3d')
        
        if 'psnr_improvement' in df.columns:
            for i, alg in enumerate(algorithms):
                alg_data = df[df['algorithm'] == alg]
                ax2.scatter(
                    alg_data['noise_intensity'],
                    alg_data['blur_intensity'],
                    alg_data['psnr_improvement'],
                    c=[colors[i]], label=alg, s=self.MARKER_SIZE, alpha=0.8
                )
            
            ax2.set_xlabel('Noise Intensity')
            ax2.set_ylabel('Blur Intensity')
            ax2.set_zlabel('PSNR Improvement (dB)')
            ax2.set_title('PSNR Improvement by Degradation Level')
        
        # График 3: 2D проекция - Шум vs PSNR
        ax3 = fig.add_subplot(223)
        self._plot_2d_with_pareto(
            ax3, df, 'noise_intensity', 'psnr',
            'Noise Intensity', 'PSNR (dB)',
            'Pareto Front: Noise Robustness',
            maximize_x=False, maximize_y=True
        )
        
        # График 4: 2D проекция - Размытие vs PSNR
        ax4 = fig.add_subplot(224)
        self._plot_2d_with_pareto(
            ax4, df, 'blur_intensity', 'psnr',
            'Blur Intensity', 'PSNR (dB)',
            'Pareto Front: Blur Robustness',
            maximize_x=False, maximize_y=True
        )
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_folder / '3d_pareto_analysis.png'
            plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Saved 3D analysis to {filepath}")
        
        plt.show()
    
    def _plot_2d_with_pareto(
        self,
        ax: Axes,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        x_label: str,
        y_label: str,
        title: str,
        maximize_x: bool = True,
        maximize_y: bool = True
    ) -> None:
        """
        Построение 2D графика с выделенным фронтом Парето.
        
        Параметры
        ---------
        ax : Axes
            Объект осей matplotlib.
        df : pd.DataFrame
            Данные.
        x_col, y_col : str
            Имена столбцов для осей x и y.
        x_label, y_label : str
            Подписи осей.
        title : str
            Заголовок графика.
        maximize_x, maximize_y : bool
            Направление оптимизации для каждой оси.
        """
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        
        # Отображение всех точек
        for i, alg in enumerate(algorithms):
            alg_data = df[df['algorithm'] == alg]
            ax.scatter(
                alg_data[x_col], alg_data[y_col],
                c=[colors[i]], label=alg, s=self.MARKER_SIZE, alpha=self.ALPHA
            )
        
        # Поиск и выделение фронта Парето
        valid_data = df[[x_col, y_col]].dropna()
        if len(valid_data) > 0:
            points = valid_data.values
            pareto_mask = self._find_pareto_front(points, [maximize_x, maximize_y])
            pareto_points = df.loc[valid_data.index[pareto_mask]]
            
            if len(pareto_points) > 1:
                # Сортировка для построения линии
                pareto_sorted = pareto_points.sort_values(x_col)
                ax.plot(
                    pareto_sorted[x_col], pareto_sorted[y_col],
                    'r--', linewidth=self.LINE_WIDTH, label='Pareto Front',
                    zorder=10
                )
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    def _plot_2d_pareto_fronts(
        self, 
        df: pd.DataFrame, 
        save: bool = True
    ) -> None:
        """
        Построение 2D визуализаций фронта Парето.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        save : bool
            Сохранять график в файл.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        pareto_configs = [
            ('psnr', 'time', 'PSNR (dB)', 'Time (s)',
             'Quality vs Speed', True, False),
            ('psnr', 'noise_intensity', 'PSNR (dB)', 'Noise Intensity',
             'Quality vs Noise Robustness', True, False),
            ('ssim', 'blur_intensity', 'SSIM', 'Blur Intensity',
             'Structural Similarity vs Blur', True, False),
            ('psnr_improvement', 'noise_intensity', 'PSNR Improvement (dB)', 
             'Noise Intensity', 'Improvement vs Noise', True, False)
        ]
        
        for idx, (x_col, y_col, x_label, y_label, title, max_x, max_y) in enumerate(pareto_configs):
            ax = axes[idx // 2, idx % 2]
            
            if x_col in df.columns and y_col in df.columns:
                valid = df[[x_col, y_col, 'algorithm']].dropna()
                if len(valid) > 0:
                    self._plot_2d_with_pareto(
                        ax, valid, x_col, y_col, x_label, y_label, title, max_x, max_y
                    )
            else:
                ax.text(0.5, 0.5, f"Data not available\n({x_col}, {y_col})",
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_folder / '2d_pareto_fronts.png'
            plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Saved 2D fronts to {filepath}")
        
        plt.show()
    
    def _plot_algorithm_comparison(
        self, 
        df: pd.DataFrame, 
        save: bool = True
    ) -> None:
        """
        Построение визуализаций для сравнения алгоритмов.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        save : bool
            Сохранять график в файл.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
        
        # График 1: Box plot PSNR по алгоритмам
        ax1 = axes[0, 0]
        df.boxplot(column='psnr', by='algorithm', ax=ax1)
        ax1.set_title('PSNR Distribution by Algorithm', fontsize=12)
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('PSNR (dB)')
        plt.sca(ax1)
        plt.xticks(rotation=45, ha='right')
        
        # График 2: Среднее PSNR с доверительными интервалами
        ax2 = axes[0, 1]
        grouped = df.groupby('algorithm')['psnr'].agg(['mean', 'std'])
        x_pos = range(len(grouped))
        ax2.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
               color=colors[:len(grouped)], capsize=5, alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Mean PSNR with Standard Deviation', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # График 3: Улучшение PSNR по алгоритмам
        ax3 = axes[1, 0]
        if 'psnr_improvement' in df.columns:
            improvement = df.groupby('algorithm')['psnr_improvement'].mean()
            bars = ax3.bar(range(len(improvement)), improvement.values,
                          color=colors[:len(improvement)], alpha=0.8)
            ax3.set_xticks(range(len(improvement)))
            ax3.set_xticklabels(improvement.index, rotation=45, ha='right')
            ax3.set_ylabel('Average PSNR Improvement (dB)')
            ax3.set_title('Restoration Improvement', fontsize=12)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Добавление подписей значений
            for bar, val in zip(bars, improvement.values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # График 4: Производительность по типам фильтров
        ax4 = axes[1, 1]
        if 'filter' in df.columns:
            pivot = df.pivot_table(
                values='psnr', index='algorithm', columns='filter', aggfunc='mean'
            )
            pivot.plot(kind='bar', ax=ax4, width=0.8, alpha=0.8)
            ax4.set_ylabel('PSNR (dB)')
            ax4.set_title('Performance by Filter Type', fontsize=12)
            ax4.legend(title='Filter', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.sca(ax4)
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_folder / 'algorithm_comparison.png'
            plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Saved comparison to {filepath}")
        
        plt.show()
    
    def _plot_performance_heatmap(
        self, 
        df: pd.DataFrame, 
        save: bool = True
    ) -> None:
        """
        Построение тепловых карт производительности.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        save : bool
            Сохранять график в файл.
        """
        if 'filter' not in df.columns:
            logger.warning("No filter data for heatmap")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Тепловая карта 1: PSNR
        ax1 = axes[0]
        try:
            pivot_psnr = df.pivot_table(
                values='psnr', index='algorithm', columns='filter', aggfunc='mean'
            )
            im1 = ax1.imshow(pivot_psnr.values, cmap='RdYlGn', aspect='auto')
            ax1.set_xticks(range(len(pivot_psnr.columns)))
            ax1.set_yticks(range(len(pivot_psnr.index)))
            ax1.set_xticklabels(pivot_psnr.columns, rotation=45, ha='right')
            ax1.set_yticklabels(pivot_psnr.index)
            ax1.set_title('PSNR Heatmap (dB)', fontsize=12)
            
            # Добавление текстовых аннотаций
            for i in range(len(pivot_psnr.index)):
                for j in range(len(pivot_psnr.columns)):
                    val = pivot_psnr.values[i, j]
                    if not np.isnan(val):
                        ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                                fontsize=9, color='black')
            
            plt.colorbar(im1, ax=ax1, label='PSNR (dB)')
            
        except Exception as e:
            logger.warning(f"PSNR heatmap failed: {e}")
        
        # Тепловая карта 2: SSIM
        ax2 = axes[1]
        if 'ssim' in df.columns:
            try:
                pivot_ssim = df.pivot_table(
                    values='ssim', index='algorithm', columns='filter', aggfunc='mean'
                )
                im2 = ax2.imshow(pivot_ssim.values, cmap='RdYlGn', aspect='auto')
                ax2.set_xticks(range(len(pivot_ssim.columns)))
                ax2.set_yticks(range(len(pivot_ssim.index)))
                ax2.set_xticklabels(pivot_ssim.columns, rotation=45, ha='right')
                ax2.set_yticklabels(pivot_ssim.index)
                ax2.set_title('SSIM Heatmap', fontsize=12)
                
                for i in range(len(pivot_ssim.index)):
                    for j in range(len(pivot_ssim.columns)):
                        val = pivot_ssim.values[i, j]
                        if not np.isnan(val):
                            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                                    fontsize=9, color='black')
                
                plt.colorbar(im2, ax=ax2, label='SSIM')
                
            except Exception as e:
                logger.warning(f"SSIM heatmap failed: {e}")
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_folder / 'performance_heatmaps.png'
            plt.savefig(filepath, dpi=self.FIGURE_DPI, bbox_inches='tight')
            logger.info(f"Saved heatmaps to {filepath}")
        
        plt.show()
    
    def _print_statistical_summary(self, df: pd.DataFrame) -> None:
        """
        Вывод статистической сводки результатов.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        """
        print("\n" + "=" * 70)
        print("STATISTICAL SUMMARY")
        print("=" * 70)
        
        # Общая статистика по алгоритмам
        print("\nPerformance by Algorithm:")
        print("-" * 50)
        
        stats = df.groupby('algorithm').agg({
            'psnr': ['mean', 'std', 'min', 'max', 'count']
        }).round(3)
        stats.columns = ['Mean PSNR', 'Std', 'Min', 'Max', 'Count']
        print(stats.to_string())
        
        # Статистика SSIM
        if 'ssim' in df.columns:
            print("\nSSIM Statistics:")
            print("-" * 50)
            ssim_stats = df.groupby('algorithm')['ssim'].agg(['mean', 'std']).round(4)
            print(ssim_stats.to_string())
        
        # Лучшие алгоритмы для различных условий
        print("\nBest Algorithm for Each Condition:")
        print("-" * 50)
        
        # Высокий шум
        if 'noise_intensity' in df.columns:
            high_noise = df[df['noise_intensity'] > df['noise_intensity'].median()]
            if len(high_noise) > 0:
                best_noise = high_noise.groupby('algorithm')['psnr'].mean().idxmax()
                score = high_noise.groupby('algorithm')['psnr'].mean().max()
                print(f"  High noise conditions: {best_noise} (PSNR: {score:.2f} dB)")
        
        # Сильное размытие
        if 'blur_intensity' in df.columns:
            high_blur = df[df['blur_intensity'] > df['blur_intensity'].median()]
            if len(high_blur) > 0:
                best_blur = high_blur.groupby('algorithm')['psnr'].mean().idxmax()
                score = high_blur.groupby('algorithm')['psnr'].mean().max()
                print(f"  High blur conditions:  {best_blur} (PSNR: {score:.2f} dB)")
        
        # Лучший в среднем
        best_overall = df.groupby('algorithm')['psnr'].mean().idxmax()
        score_overall = df.groupby('algorithm')['psnr'].mean().max()
        print(f"  Overall best:          {best_overall} (PSNR: {score_overall:.2f} dB)")
    
    def _print_pareto_optimal_solutions(self, df: pd.DataFrame) -> None:
        """
        Вывод Парето-оптимальных решений.
        
        Параметры
        ---------
        df : pd.DataFrame
            Данные для анализа.
        """
        print("\n" + "=" * 70)
        print("PARETO-OPTIMAL SOLUTIONS")
        print("=" * 70)
        
        # Поиск фронта Парето для PSNR vs время обработки
        if 'time' in df.columns:
            valid = df[['psnr', 'time', 'algorithm']].dropna()
            if len(valid) > 0:
                points = valid[['psnr', 'time']].values
                pareto_mask = self._find_pareto_front(
                    points, maximize=[True, False]
                )
                pareto_solutions = valid[pareto_mask]
                
                print("\nQuality vs Speed Pareto Front:")
                print("-" * 50)
                for _, row in pareto_solutions.iterrows():
                    print(f"  {row['algorithm']}: PSNR={row['psnr']:.2f} dB, "
                          f"Time={row['time']:.3f} s")
        
        # Поиск фронта Парето для PSNR vs устойчивость к шуму
        if 'noise_intensity' in df.columns:
            # Группировка по алгоритмам и вычисление средней производительности
            algo_stats = df.groupby('algorithm').agg({
                'psnr': 'mean',
                'noise_intensity': 'mean'
            }).reset_index()
            
            points = algo_stats[['psnr', 'noise_intensity']].values
            pareto_mask = self._find_pareto_front(points, maximize=[True, False])
            pareto_algos = algo_stats[pareto_mask]
            
            print("\nQuality vs Noise Robustness Pareto Front:")
            print("-" * 50)
            for _, row in pareto_algos.iterrows():
                print(f"  {row['algorithm']}: PSNR={row['psnr']:.2f} dB")
        
        print("\n" + "=" * 70)

