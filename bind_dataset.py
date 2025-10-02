# bind_dataset.py (исправленная версия)

import os
from pathlib import Path
from processing import Processing

def bind_single_entry(
    processing_instance: Processing,
    original_image_name: str,
    blur_filter_name: str,
    noise_name: str = "none"
):
    print(f"\n--- Попытка связать: {original_image_name} | {blur_filter_name} | {noise_name} ---")

    base_name = Path(original_image_name).stem
    ext = Path(original_image_name).suffix

    original_path = processing_instance.folder_path / original_image_name
    
    if noise_name.lower() == 'none':
        distorted_filename = f"{base_name}_{blur_filter_name}{ext}"
        filter_description = blur_filter_name
    else:
        distorted_filename = f"{base_name}_{blur_filter_name}_{noise_name}{ext}"
        filter_description = f"{blur_filter_name}_{noise_name}"
        
    distorted_path = processing_instance.folder_path_blurred / distorted_filename
    
    # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
    kernel_filename = f"{blur_filter_name}.png"
    kernel_path = processing_instance.kernel_dir / kernel_filename
    
    paths_to_check = {
        'images/benchmark/original': original_path,
        'images/benchmark/distorted': distorted_path,
        'images/benchmark/ground_truth_filters': kernel_path
    }
    
    all_files_exist = True
    for key, path in paths_to_check.items():
        if not path.exists():
            print(f"Ошибка: Файл не найден по пути ({key}): {path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("Связывание отменено из-за отсутствия файлов.")
        return

    try:
        processing_instance.bind(
            original_image_path=str(original_path),
            blurred_image_path=str(distorted_path),
            original_kernel_path=str(kernel_path),
            filter_description=filter_description
        )
        print(f"Успешно связано: {distorted_filename}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Ошибка при вызове метода bind: {e}")

# ... (функция bind_full_dataset остается без изменений) ...

def bind_full_dataset(processing_instance: Processing):
    # ... (код функции без изменений) ...
    print("\n===================================================")
    print("=== Начало автоматического связывания датасета ===")
    print(f"Сканирование папки: {processing_instance.folder_path_blurred}")
    print("===================================================")

    if not processing_instance.folder_path_blurred.exists():
        print(f"Ошибка: Директория с искаженными изображениями не найдена.")
        return
        
    original_images = {p.name: p for p in processing_instance.folder_path.iterdir() if p.is_file()}
    if not original_images:
        print(f"Внимание: Не найдено ни одного оригинального изображения в {processing_instance.folder_path}")

    for distorted_path in processing_instance.folder_path_blurred.iterdir():
        if not distorted_path.is_file():
            continue

        filename = distorted_path.name
        parts = Path(filename).stem.split('_')
        
        if len(parts) < 2:
            print(f"Пропуск: Некорректный формат имени файла '{filename}'")
            continue

        original_name_candidate = ""
        blur_filter_name = ""
        noise_name = "none"

        for i in range(1, len(parts)):
            temp_original_base = "_".join(parts[:i])
            
            found_original = None
            for orig_name in original_images:
                if Path(orig_name).stem == temp_original_base:
                    found_original = orig_name
                    break
            
            if found_original:
                remaining_parts = parts[i:]
                if remaining_parts[-1] in ['gaussian', 'poisson', 'saltpepper']:
                    noise_name = remaining_parts[-1]
                    blur_filter_name = "_".join(remaining_parts[:-1])
                else:
                    blur_filter_name = "_".join(remaining_parts)
                
                original_name_candidate = found_original
                break
        
        if not original_name_candidate or not blur_filter_name:
            print(f"Пропуск: Не удалось найти оригинал или разобрать имя фильтра для '{filename}'")
            continue

        bind_single_entry(
            processing_instance=processing_instance,
            original_image_name=original_name_candidate,
            blur_filter_name=blur_filter_name,
            noise_name=noise_name
        )
    
    print("\n==============================================")
    print("=== Автоматическое связывание завершено ===")
    print("==============================================")