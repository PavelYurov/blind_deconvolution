# Визуализация

Разберем, как получить практически все визуализации, графики, таблицы и иллюстрации, использованные в ВКР, а также, куда они сохраняются. Будут разобраны скрипты для генерации визуализации, а также, что и куда они создают. Все скрипты запускались из корня проекта.

## Скрипты

Главным подскриптом для генерации графиков является `visualisation.py`. Большая часть всей графики генерируется здесь. Так как самой главной задачей было сравнение алгоритмов между собой, то именно здесь представлены почти все графики (наравне с `run_all_datasets_{название алгоритма}.py`)

`presentation_labels.json` (конфигурационный файл) список имен, на которые декодор заменит технические названия (названия директорий и путей из таблиц). Любые названия проходят через декодер. Если конфигурационного файла декодера не будет, то создастся пустой

- `generate_visuals.py` - мастер-скрипт `visualisation.py`. Запускает чтение результатов из таблиц, генерацию графиков и их сохранение.

  - Необходимо:
    - `/presentation_graphics` файл, откуда будут браться результаты и куда будут записаны графики
    - набор данных по адресу, откуда был запуск алгоритмов. все пути приводятся к локальному, поэтому, если запуск был на разных машинах, но из корня проекта, все должно работать правильно
  - Скрипт разделен на несколько фаз, чтобы сэкономить время, нужное для генерации (перегенерации) всех графиков:
    - фаза 1:
      - строит графики качества для одного алгоритма (для каждого отдельно)
      - сохраняется в директориях самих алгоритмов `presentation_graphics/{название алгоритма}`
      - использует данные из `Levin`, `Kohler`, `Set12`, `Sun`
      - результаты в папках `tex`, `kernel_profiles`, `figures`
    - фаза 2:
      - строит графики для сравнения качества всех алгоритмов (сравнение качества)
      - сравнение разделено на 3 категории: base, nobase и all, для разностороннего сравнения различных конфигураций алгоритмов
      - сохраняется в директории всех алгоритмов (общей) `presentation_graphics`
      - использует данные из `Levin`, `Kohler`, `Set12`, `Sun`
      - результаты в папках `comparison_tex`, `comparison_figures`
    - фаза 3:
      - строит графики производительности (область приминимости и скорость работы)
      - сравнение разделено на 3 категории: base, nobase и all, для разностороннего сравнения различных конфигураций алгоритмов
      - сохраняется в директории всех алгоритмов (общей) `presentation_graphics`
      - использует данные из `Grid_Test`, `Complexity_Test`
      - результаты в папке `performance_figures`
    - фаза 4:
      - строит графики итерационной сходимости
      - сохраняется в директории самих алгоритмов `presentation_graphics/{название алгоритма}`
      - использует данные из `log_test`
      - результаты в папке `figures`
    - фаза 5:
      - строит графики устойчивости к изменению гиперпараметров
      - сохраняется в директории самих алгоритмов `presentation_graphics/{название алгоритма}`
      - использует данные из `hyperparam_grid`
      - результаты в папке `figures`
  - Запуск:
  ``` bash
    python generate_visuals.py                # все фазы
    python generate_visuals.py --phase 0      # все фазы (явно)
    python generate_visuals.py --phase 1      # одиночные графики (per-algorithm)
    python generate_visuals.py --phase 2      # сравнительные графики качества
    python generate_visuals.py --phase 3      # производительность / 3D-карты
    python generate_visuals.py --phase 4      # итерационные графики
    python generate_visuals.py --phase 5      # гиперпараметры
    python generate_visuals.py --skip-kernel-profiles   # фаза 1 без профилей ядер (слишком долгая генерация всех)
    python generate_visuals.py --force-kernel-profiles  # фаза 1 перезапись профилей ядер (по стандарту не перезаписывает для ускорения процесса)
    python generate_visuals.py --noise-only   # только перезаписать графики зависимости от шума
  ```

- `generate_visuals_trio.py` графики визуального сравнения $firls$ и $TV+l_0$ неслепых методов для восстановления изображения `drawing_heliod_mediumgaussian`. использовались в презентации.
  - используемые наборы данных: `presentation_graphics_nonblind_with_estimated_kernels` (переименованный `presentation_graphics`, полученный из `run_all_dataset_hbsp_nonblind.py`)
  - куда сохраняется результат: `presentation_graphics_nonblind/comparison_figures/trio_drawing_heliod.png`
  - запуск:
  ``` bash
    python generate_visuals_trio.py
  ```
- `generate_visuals_denoise.py` графики качества восстановления. chart2 - средний *PSNR* / *SSIM* / *ISNR* на комбинацию денойзера/метода оценки шума. chart3 - сравнение алгоритмов шумоподавления при фиксированном методе оценки на разных интенсивностях шума
  - используемые наборы данных: `presentation_graphics_denoise`
  - куда сохраняется результат: `presentation_graphics_denoise`
  - запуск:
  ``` bash
    python generate_visuals_denoise.py
    python generate_visuals_denoise.py --estimator chen   # chart3 только для chen
    python generate_visuals_denoise.py --phase 2          # только chart2
    python generate_visuals_denoise.py --phase 3          # только chart3
  ```
- `generate_visuals_nonblind.py` графики сравнения неслепых методов деконволюции.
  - используемые наборы данных: `presentation_graphics_nonblind`
  - куда сохраняется результат: `presentation_graphics_nonblind/comparison_figures`
  - запуск:
  ``` bash
    python generate_visuals_nonblind.py
  ```
- `generate_visuals_pareto_comp.py` визуальное сравнение стандартных параметров / Парето параметров / лучших параметров оптимизации
  - используемые наборы данных: `presentation_graphics_pareto_comp`
  - куда сохраняется результат: `presentation_graphics_pareto_comp/{название алгоритма}/{название типа парето-фронта}/visuals`
  - запуск:
  ``` bash
    python generate_visuals_pareto_comp.py
    python generate_visuals_pareto_comp.py --table pareto_PSNR_front # только одна таблица
  ```
- `generate_visuals_priors.py` таблицы и графики сравнения регуляризаторов для метода fbdhsgp
  - используемые наборы данных: `presentation_graphics_priors`
  - куда сохраняется результат: `presentation_graphics_prior/comparison_figures`
  - запуск:
  ``` bash
    python generate_visuals_priors.py
  ```

- `make_dataset_mosaic.py` полотна с примерами изображений датасетов и ядрами
  - используемые наборы данных: `Levin`, `Set12`, `Kohler`, `Sun` для `anton`, `kostya`, `pasha`
  - куда сохраняется результат: `images/compare_data/{имя}/`
  - запуск:
  ``` bash
    python make_dataset_mosaic.py
  ```

- `make_flat_dataset_mosaic.py` полотна с примерами изображений датасетов и ядрами
  - используемые наборы данных: `images/large_data_pictures`, `images/middle_data_pictures`
  - куда сохраняется результат: `images/large_data_pictures`, `images/middle_data_pictures`
  - запуск:
  ``` bash
    python make_flat_dataset_mosaic.py
  ```

- `gradient_histogram.py` вспомогательный скрипт для подсчета маргинального распределения градиентов изображения. Используется алгоритмами далее.

- `run_gradient_hist.py` гистограммы градиентов (в логарифмическом масштабе) для одного или нескольких датасетов деконволюции. Использует `gradient_histogram.py`
  - используемые наборы данных: `priors`, `kostya/Levin`
  - куда сохраняется результат: `presentation_graphics_priors/gradient_hist`, `presentation_graphics_priors_comp/gradient_hist`
  - запуск:
  ``` bash
    python run_gradient_hist.py
  ```
- `run_gradient_hist_algos.py` гистограммы градиентов (в логарифмическом масштабе)
для нескольких датасетов и нескольких алгоритмов из разных папок результатов. Использует `gradient_histogram.py`
  - используемые наборы данных: `presentation_graphics`
  - куда сохраняется результат: `gradient_hist_algos`
  - запуск:
  ``` bash
    python run_gradient_hist_algos.py
  ```
- `run_gradient_hist_log_test.py` Гистограммы градиентов для двух изображений из `log_test` `bridge_losso_clean` и `boy_losso_clean`. Использует `gradient_histogram.py`
  - используемые наборы данных: `Fast_BD_Hyper-Sparse_Gradient/log_test/boy_losso_clean/restored_final.png`, `Fast_BD_Hyper-Sparse_Gradient/log_test/bridge_losso_clean/restored_final.png`
  - куда сохраняется результат: `gradient_hist_log_test`
  - запуск:
  ``` bash
    python run_gradient_hist_log_test.py
  ```