# Запуск экспериментов

Здесь разбирается запуск скриптов подсчета экспериментальных данных.

## Общие положения

- запуск происходит из корня проекта
- нужен набор данных `/images`
- перед запуском нужно выполнить все пункты из общего readme

## Скрипты

- `_process_worker.py` вспомогательный скрипт по запуску алгоритмов, загрузке и сохранению наборов данных, подсчету метрик и сохранению таблиц.

- `run_all_dataset_{имя алгоритма}.py`
   - скрипт требует дополнительные скрипты `_process_worker.py` и `visualisation.py`
   - скрипт представлен в 2-х вариациях: "_base" и без приписка "_base". Отличаются только набором параметров алгоритма: base вариант имеет параметры из статьи и не имеет модификации в  виде оркестратора, второй наоборот улучшенный наш метод
   - скрипт разделен на 3 фазы:
      - фаза 1 - подсчет качества восстановления изображений для указанных датасетов. Вычисляются метрики PSNR, SSIM, Error Ratio и время работы. Результаты сохраняются в CSV файлы и генерируются превью восстановления. Используемые наборы данных: `Levin`, `Kohler`, `Sun`, `Set12`, `Complexity_test` (10 запусков для исключения неустойчивости), `Grid_Test`.
      - фаза 2 - детальный анализ сходимости по итерациям. На каждой итерации алгоритма сохраняется текущая оценка ядра и промежуточное изображение. Рассчитываются L1/L2 ошибки ядра относительно точного ядра. Используемые наборы данных: `log_test`
      - фаза 3 - поиск по заданной сетки гиперпараметров. Применяется для построения тепловых карт качества и кривых чувствительности алгоритма. Используемые наборы данных: `param_grid_test`
   - запуск (пример):
      ```bash
      python run_all_dataset_pmp.py              # все фазы
      python run_all_dataset_pmp.py --phase 0    # все фазы (явно)
      python run_all_dataset_pmp.py --phase 1    # только датасеты
      python run_all_dataset_pmp.py --phase 2    # только итерации
      python run_all_dataset_pmp.py --phase 3    # только гиперпараметры
      ```
   - настройки:
      - ALGORITHM_LABEL - название алгоритма в файловой системе и на графиках
      - NUM_WORKERS - число ядер
      - ALG_KWARGS - параметры алгоритма
      - DATASETS_PHASE1 - наборы данных для фазы 1
   - результаты сохраняются в файл `/presentation_graphics/{название алгоритма}/{название датасета}/` в следующем формате:
      - фаза 1:
         - kernels - оцененные ядра (названия `kernels/{название искаженного изображения полностью}_{название алгоритма}_kernel.png`)
         - restored - восстановленные изображения (названия `restored/{название искаженного изображения полностью}_{название алгоритма}.png`)
         - `complex_plots` - превью восстановдения с метриками. удобно оценивать результат алгоритма и одновременно смотреть на оцененное ядро (названия `complex_plots/{название искаженного изображения полностью}_complex.png`)
         - `/resoults_{название алгоритма}_{название датасета}.csv` - таблица с метриками, путями к изображениям и ядрам (истинным и оцененным)
      - фаза 2:
         - `/{название искаженного изображения}/kernels/` - все ядра, собранные с итераций
         - `/{название искаженного изображения}/restored/` - все изображения, полученные восстановлением с помощью ядер, полученных на промежуточных итерациях
         - `/{название искаженного изображения}/iterations_log.csv` - таблица с метриками для 1 изображения
         - `/{название искаженного изображения}/kernel_final.png` и `/{название искаженного изображения}/restored_final.png` - результат восстановления
         - `iteration_summary_{название алгоритма}.csv` - таблица со всеми исследуемыми изображениями
      - фаза 3:
         - `grid_{параметр 1}_{параметр 2}_{название алгоритма}.csv` - сводная таблица результатов (метрики $\times$ параметры). Изображений нет.
      - `all_results_{название алгоритма}.csv`  - общая таблица с метриками по всем наборам данных. отсюда берутся данные для построения графиков.
- `fix_combined_table.py` вспомогательный скрипт собирающий/пересобирающий общую таблицу метрик и результатов восстановленния. Gрименялось, если результаты были получены на разных машинах, так как таблица собирается автоматически в конце подсчета. Перезаписывает файлы из `/presentation_graphics/`
``` bash
   python fix_combined_table.py
```
- `fix_er.py` вспомогательный скрипт для пересчета error ratio. Из-за того что нет информации, как корректно ее считать для ситуации с шумом, было реализовано несколько вариантов подсчета данной метрики. В заключительном варианте искользовался ringing_removal (ringing) с денойз конвейером.
``` bash
   python fix_er.py                              # presentation_graphics/
   python fix_er.py --root presentation_graphics_pasha # выбрать папку с результатами
   python fix_er.py --dry-run                    # посчитать без сохранения
   python fix_er.py --no-impulse                 # пропустить шаг удаления импульсного шума
   python fix_er.py --datasets all               # задать конкретный набор данных или все
   python fix_er.py --workers 4                  # число потоков, пересчет долгий
   python fix_er.py --no-bm3d                    # пропустить денойз
   python fix_er.py --datasets all --solver ringing --lambda-tv 1e-3 --lambda-l0 2e-3 --weight-ring 1.0 #пример запуска с параметрами (у нас это был финальный). Для других вариантов параметров см. сам скрипт.
```
- `run_all_dataset_denoise.py` скрипт восстановления методов шумоподавления (BM3D, NLM, Guided, Bilateral, VST, ACT, adaptive median) в комбинации с различными оценщиками шума (PCA, Chen и без них). VST применяется только в связке с PCA. Adaptive median только с None, так как оценка там своя. набор данных `noise_1` (`noise`) имеет только шум (без смаза). Сохраняет результаты в `/presentation_graphics_denoise`
``` bash
   python run_all_dataset_denoise.py                     # все комбинации
   python run_all_dataset_denoise.py --combo bm3d_chen   # одна комбинация
   python run_all_dataset_denoise.py --list              # напечатать все комбинации
   python run_all_dataset_denoise.py --workers 4         # задать число ядер
```
- `run_all_dataset_denoise_with_blur.py` скрипт аналогичен `run_all_dataset_denoise.py` с разницей в датасете: использует набор данных `Grid_Test` со смазом и шумом. Сохраняет результаты в `/presentation_graphics_denoise_with_blur`
``` bash
   python run_all_dataset_denoise_with_blur.py                    # все комбинации
   python run_all_dataset_denoise_with_blur.py --combo bm3d_chen  # одна комбинация
   python run_all_dataset_denoise_with_blur.py --list             # напечатать все комбинации
   python run_all_dataset_denoise_with_blur.py --workers 4        # задать число ядер
```
- `run_all_dataset_{название метода}_pareto_comp.py` скрипт подсчета результатов стандартных параметров, парето и лучших оптимизации для соответственного сравнения. Берет данные из файла `presentation_graphics_pareto/pareto_{название алгоритма}_tpe_150iter_50dist` (файл с результатами оптимизации), наборы данных `/large_data_pictures` и `/middle_data_pictures`,
сохраняет в папку `/presentation_graphics_pareto_comp`
``` bash
   # пример esm
   python run_all_dataset_esm_pareto_comp.py          
   python run_all_dataset_esm_pareto_comp.py --table "pareto_PSNR_front.csv" # Запустить только одну таблицу
   python run_all_dataset_esm_pareto_comp.py --force # Пересчитать все результаты
```
- `run_all_dataset_fbdhsgp.py` дополнительное сравнение работы алгоритма с разными регуляризаторами. Использует логику `run_all_dataset_{имя алгоритма}.py` полностью
``` bash
   python run_all_dataset_fbdhsgp.py --phase 1 # (так как использует логику обычного run_all_dataset_fbdhsgp.py, то нужно поменять prior_name в параметрах, поменять набор данных на "prior" и в ALGORITHM_LABEL добавить приписку регуляризатора (пример: "Fast BD Hyper-Sparse Gradient (log prior)"))
```

- `run_all_dataset_hbsp_nonblind.py` скрипт сравнения качества восстановления различных неслепых шагов с оценненным ядром методом hbsp. Использует набор данных `/nonblind`, сохраняет результат `/presentation_graphics`
``` bash
   python run_all_dataset_hbsp_nonblind.py                  # все методы
   python run_all_dataset_hbsp_nonblind.py --method firls   # только один
```
- `run_all_dataset_nonblind.py` скрипт сравнения качества восстановления различных неслепых методов со знанием точного ядра смаза. набор данных: `Grid_Test` (только изображения без шума), сохраняет в папку `/presentation_graphics_nonblind`
``` bash
   python run_all_dataset_nonblind.py                  # все методы
   python run_all_dataset_nonblind.py --method firls   # конкретный метод
   python run_all_dataset_nonblind.py --list           # список методов
   python run_all_dataset_nonblind.py --workers 4      # количество потоков
```


