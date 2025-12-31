"""
Модуль загрузки связей и метрик в таблицы.

Автор: Юров П.И.
"""
import math
import pandas as pd
from pathlib import Path

from typing import Dict, Any

from IPython.display import display


class ModuleData:
    """
    Модуль загрузки связей и метрик в таблицы.
    """

    def __init__(self, processing_instance: Any) -> None:
        """
        Инициализация.
        
        Параметры
        ---------
        processing_instance : Any
            Ссылка на объект Processing с изображениями.
        """
        self.processing = processing_instance

    def get_table(self,
                   table_path: Path, 
                   display_table: bool = False) -> None:
        """Получение метрик в структурированном виде."""
        data = {}
        data = self._collect_data(data) 
        self._save_data_to_csv(data, table_path, display_table)

    def _collect_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Сбор и сохранение информации для общего анализа."""
        for img in self.processing.images:
            original_image = img.get_original()
            img.save_filter()

            blurred_kernel_array = img.get_original_kernels()
            blurred_psnr_array = img.get_blurred_PSNR()
            blurred_ssim_array = img.get_blurred_SSIM()

            algorithm_kernel = img.get_kernels()
            algorithm_restored_image = img.get_restored()
            algorithm_restored_psnr = img.get_PSNR()
            algorithm_restored_ssim = img.get_SSIM()
            
            #линия за линией
            for blurred_image in img.get_blurred_array(): #подразумеваем, что она точно существует
                data.setdefault('original', []).append(original_image)
                data.setdefault('kernel blur', []).append(blurred_kernel_array.get(str(blurred_image), 'missing'))
                data.setdefault('blurred', []).append(blurred_image)

                data.setdefault('blurred psnr', []).append(blurred_psnr_array.get(str(blurred_image), math.nan))
                data.setdefault('blurred ssim', []).append(blurred_ssim_array.get(str(blurred_image), math.nan))
                for algorithm_name in img.get_algorithm():
                    data.setdefault(f"kernel_{algorithm_name}", []).append(
                        algorithm_kernel.get((str(blurred_image), str(algorithm_name)), 'missing')
                    )
                    data.setdefault(algorithm_name, []).append(
                        algorithm_restored_image.get((str(blurred_image), str(algorithm_name)), 'missing')
                    )
                    data.setdefault(f"psnr_{algorithm_name}", []).append(
                        algorithm_restored_psnr.get((str(blurred_image), str(algorithm_name)), math.nan)
                    )
                    data.setdefault(f"ssim_{algorithm_name}", []).append(
                        algorithm_restored_ssim.get((str(blurred_image), str(algorithm_name)), math.nan)
                    )
        return data

    def _save_data_to_csv(self, 
                          data: Dict[str, Any], 
                          path: Path, 
                          display_table: bool = False) -> None:
        """Сохраняет словарь в CSV файл."""
        df_data = pd.DataFrame(data)
        if display_table:
            display(df_data)
        df_data.to_csv(path, index=False)