"""presentation_generate_datasetcomp_light.py

Облегчённая генерация датасета:
  10 изображений × 6 ядер × 4 варианта = 240 картинок.

4 варианта на каждую пару (изображение × ядро):
  1) clean  — только смаз, без шума
  2) weak   — слабый шум  (тип round-robin по 5 типам)
  3) medium — средний шум  (тип round-robin со сдвигом +1)
  4) strong — сильный шум  (тип round-robin со сдвигом +2)

Типы шума (5 шт.) равномерно распределяются между парами,
так что каждый тип покрыт на каждом уровне примерно одинаково.

Использует классы фильтров из фреймворка blinddeconv.
"""

import sys
import json
import itertools
import shutil
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# from framework import framework

# ── Путь к фреймворку ───────────────────────────────────────────
FRAMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from blinddeconv.filters.blur import Kernel_convolution
from blinddeconv.filters.noise import (
    GaussianNoise, PoissonNoise, SaltAndPepperNoise, Pink_Noise, Brown_Noise,
)

# ═══════════════════════════════════════════════════════════════════
#   Конфигурация директорий (ОТРЕДАКТИРУЙТЕ ПОД СЕБЯ)
# ═══════════════════════════════════════════════════════════════════

# 1. Откуда берем исходные файлы (только PNG)
# UNIVERSAL_PART = "Levin"; SIZE_IMAGE = 255*255
# UNIVERSAL_PART = "Lai"; SIZE_IMAGE = 1024*682
# UNIVERSAL_PART = "Sun"; SIZE_IMAGE = 924*668
# UNIVERSAL_PART = "Kohler"; SIZE_IMAGE = 800*800
UNIVERSAL_PART = "Set12"; SIZE_IMAGE = 512*512
# UNIVERSAL_PART = "Grid_Test"; SIZE_IMAGE = 256*256

INPUT_ORIGINALS_DIR = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART / "originals"
INPUT_KERNELS_DIR   = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART / "ground_truth_filters"

# 2. Куда сохраняем готовый датасет
OUTPUT_DATASET_DIR  = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART

# -------------------------------------------------------------------
# Внутренняя структура датасета (сохраняем как было)
ORIGINALS_DIR = OUTPUT_DATASET_DIR / "originals"
DISTORTED_DIR = OUTPUT_DATASET_DIR / "distorted"
FILTERS_DIR   = OUTPUT_DATASET_DIR / "ground_truth_filters"
TMP_DIR       = OUTPUT_DATASET_DIR / "tmp"

# ═══════════════════════════════════════════════════════════════════
#   Конфигурация шумов: 5 типов × 3 уровня = 15 вариантов
#   Уровни распределяются round-robin по парам (изображение × ядро),
#   а не полным перебором — экономия ×3 при полном покрытии типов.
# ═══════════════════════════════════════════════════════════════════

NOISE_TYPES = ["gaussian", "poisson", "impulse", "pink", "brown"]

# Для каждого типа — три уровня: слабый / средний / сильный
NOISE_LEVELS = {
    "gaussian": [
        {"label": "weak",   "param": 2.0},
        {"label": "medium", "param": 5.0},
        {"label": "strong", "param": 10.0},
    ],
    "poisson": [
        {"label": "weak",   "param": 0.15},
        {"label": "medium", "param": 0.35},
        {"label": "strong", "param": 0.60},
    ],
    "impulse": [
        {"label": "weak",   "param": [1, 1, SIZE_IMAGE // 1200]},   # мало пикселей
        {"label": "medium", "param": [1, 1, SIZE_IMAGE // 600]},
        {"label": "strong", "param": [1, 1, SIZE_IMAGE // 300]},    # много пикселей
    ],
    "pink": [
        {"label": "weak",   "param": 0.01},
        {"label": "medium", "param": 0.02},
        {"label": "strong", "param": 0.04},
    ],
    "brown": [
        {"label": "weak",   "param": 0.01},
        {"label": "medium", "param": 0.02},
        {"label": "strong", "param": 0.04},
    ],
}


def make_noise_filter(noise_type, level_param):
    """Создаёт фильтр шума по типу и параметру уровня."""
    return {
        "gaussian": lambda p: GaussianNoise(p),
        "poisson":  lambda p: PoissonNoise(p),
        "impulse":  lambda p: SaltAndPepperNoise(p),
        "pink":     lambda p: Pink_Noise(noise_level=p),
        "brown":    lambda p: Brown_Noise(noise_level=p),
    }[noise_type](level_param)


# ═══════════════════════════════════════════════════════════════════
#   Подготовка ядер: PNG → npy + Kernel_convolution
# ═══════════════════════════════════════════════════════════════════

def prepare_kernels():
    """Считывает все PNG из папки ядер, конвертирует в .npy, возвращает dict фильтров."""
    npy_dir = TMP_DIR / "kernels_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)

    filters = {}
    kernel_paths = list(INPUT_KERNELS_DIR.glob("*.png"))
    
    if not kernel_paths:
        raise FileNotFoundError(f"В папке {INPUT_KERNELS_DIR} не найдено PNG файлов ядер!")

    for png_src in kernel_paths:
        kname = png_src.stem  # Имя файла без расширения
        
        # Загрузка и нормализация
        k = cv.imread(str(png_src), cv.IMREAD_GRAYSCALE).astype(np.float64)
        if k is None:
            print(f"  [Предупреждение] Не удалось прочитать ядро: {png_src}")
            continue
            
        k /= k.sum() + 1e-12

        # Сохраняем .npy для Kernel_convolution
        npy_path = npy_dir / f"{kname}.npy"
        np.save(str(npy_path), k)

        # Копируем PNG в ground_truth_filters датасета
        dst_png = FILTERS_DIR / png_src.name
        if not dst_png.exists():
            shutil.copy2(str(png_src), str(dst_png))

        filters[kname] = Kernel_convolution(str(npy_path))
        print(f"  Ядро: {kname:25s} ({k.shape[1]}x{k.shape[0]})")

    return filters


# ═══════════════════════════════════════════════════════════════════
#   Матрица назначений: ВСЕ на ВСЕ
# ═══════════════════════════════════════════════════════════════════

def build_design_matrix(originals_list, kernels_list):
    """
    Облегчённая матрица назначений: 4 варианта на пару.

    На каждую пару (оригинал × ядро):
      1) clean  (без шума)
      2) weak   — тип = NOISE_TYPES[idx % 5]
      3) medium — тип = NOISE_TYPES[(idx + 1) % 5]
      4) strong — тип = NOISE_TYPES[(idx + 2) % 5]

    idx — порядковый номер пары (0, 1, 2, …).
    Сдвиг +1/+2 гарантирует, что на одну пару попадают разные типы шума.

    Возвращает список кортежей: (orig, kernel, noise_type, level_label, level_param)
    """
    pairs = list(itertools.product(originals_list, kernels_list))
    n_types = len(NOISE_TYPES)

    design = []
    for idx, (orig, kern) in enumerate(pairs):
        # 1) clean
        design.append((orig, kern, "", "", None))

        # 2) weak — один тип шума
        nt_w = NOISE_TYPES[idx % n_types]
        lvl_w = NOISE_LEVELS[nt_w][0]           # weak
        design.append((orig, kern, nt_w, lvl_w["label"], lvl_w["param"]))

        # 3) medium — другой тип шума
        nt_m = NOISE_TYPES[(idx + 1) % n_types]
        lvl_m = NOISE_LEVELS[nt_m][1]           # medium
        design.append((orig, kern, nt_m, lvl_m["label"], lvl_m["param"]))

        # 4) strong — третий тип шума
        nt_s = NOISE_TYPES[(idx + 2) % n_types]
        lvl_s = NOISE_LEVELS[nt_s][2]           # strong
        design.append((orig, kern, nt_s, lvl_s["label"], lvl_s["param"]))

    return design


# ═══════════════════════════════════════════════════════════════════
#   Применение фильтров + метрики
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(original, image):
    """PSNR и SSIM между original и image (оба uint8)."""
    h = min(original.shape[0], image.shape[0])
    w = min(original.shape[1], image.shape[1])
    orig_f = original[:h, :w].astype(np.float64) / 255.0
    img_f  = np.clip(image[:h, :w].astype(np.float64) / 255.0, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(orig_f, img_f, data_range=1.0))
    ssim = float(structural_similarity(orig_f, img_f, data_range=1.0))
    return psnr, ssim


def process_one(orig_path, blur_filter, noise_type, level_param, output_path):
    """Применяет blur (+ noise) через фреймворк, сохраняет, возвращает метрики."""
    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить: {orig_path}")

    # Смаз через фреймворк
    blurred = blur_filter.filter(original)
    blur_psnr, blur_ssim = compute_metrics(original, blurred)

    # Шум через фреймворк (если требуется)
    if noise_type:
        noise_filter = make_noise_filter(noise_type, level_param)
        result = noise_filter.filter(blurred)
        result = np.clip(result, 0, 255).astype(np.uint8)
        final_psnr, final_ssim = compute_metrics(original, result)
    else:
        result = blurred
        final_psnr, final_ssim = blur_psnr, blur_ssim

    cv.imwrite(str(output_path), result)

    return {
        "blur_psnr": blur_psnr, "blur_ssim": blur_ssim,
        "final_psnr": final_psnr, "final_ssim": final_ssim,
        "delta_psnr": blur_psnr - final_psnr,
        "delta_ssim": blur_ssim - final_ssim,
    }


def plot_dataset_distribution(records, save_path):
    """2D scatter: X = blur metric, Y = delta (blur - final). SSIM и PSNR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    noisy  = [r for r in records if r["noise_type"] != "clean"]
    clean  = [r for r in records if r["noise_type"] == "clean"]

    # --- SSIM ---
    ax = axes[0]
    if noisy:
        ax.scatter([r["blur_ssim"] for r in noisy],
                   [r["delta_ssim"] for r in noisy],
                   c="tab:blue", alpha=0.3, label=f"Шумное ({len(noisy)})")
    if clean:
        ax.scatter([r["blur_ssim"] for r in clean],
                   [r["delta_ssim"] for r in clean],
                   c="tab:orange", marker="x", s=80, label=f"Чистое ({len(clean)})")
    ax.set_xlabel("SSIM размытого")
    ax.set_ylabel("ΔSSIM (размытое − размытое и зашумленное)")
    ax.set_title("Распределение данных (SSIM)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- PSNR ---
    ax = axes[1]
    if noisy:
        ax.scatter([r["blur_psnr"] for r in noisy],[r["delta_psnr"] for r in noisy],
                   c="tab:blue", alpha=0.3, label=f"Шумное ({len(noisy)})")
    if clean:
        ax.scatter([r["blur_psnr"] for r in clean],
                   [r["delta_psnr"] for r in clean],
                   c="tab:orange", marker="x", s=80, label=f"Чистое ({len(clean)})")
    ax.set_xlabel("PSNR размытого")
    ax.set_ylabel("ΔPSNR (размытое − размытое и зашумленное)")
    ax.set_title("Распределение данных (PSNR)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  График сохранен: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Генерация датасета (LIGHT: 4 варианта на пару)")
    print("=" * 70)

    # Проверка исходных директорий
    if not INPUT_ORIGINALS_DIR.exists():
        print(f"ОШИБКА: Исходная папка с оригиналами не найдена: {INPUT_ORIGINALS_DIR}")
        return
    if not INPUT_KERNELS_DIR.exists():
        print(f"ОШИБКА: Исходная папка с ядрами не найдена: {INPUT_KERNELS_DIR}")
        return

    # Создаём структуру целевого датасета
    for d in[ORIGINALS_DIR, DISTORTED_DIR, FILTERS_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Считываем и копируем оригиналы
    orig_files = list(INPUT_ORIGINALS_DIR.glob("*.png"))
    if not orig_files:
        print(f"ОШИБКА: В {INPUT_ORIGINALS_DIR} нет PNG файлов!")
        return

    originals_names =[]
    print("\n--- Подготовка оригиналов ---")
    for src in orig_files:
        orig_name = src.stem
        originals_names.append(orig_name)
        dst = ORIGINALS_DIR / src.name
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
    print(f"  Найдено и скопировано оригиналов: {len(originals_names)}")

    # Подготовка ядер
    print("\n--- Подготовка ядер ---")
    blur_filters = prepare_kernels()
    kernels_names = list(blur_filters.keys())

    # Матрица назначений
    print("\n--- Матрица назначений ---")
    design = build_design_matrix(originals_names, kernels_names)

    # Статистика
    n_total = len(design)
    n_clean = sum(1 for _, _, noise, _, _ in design if not noise)
    n_noisy = n_total - n_clean

    # Подсчёт по уровням
    level_counts = Counter()
    for _, _, noise, lvl, _ in design:
        if noise:
            level_counts[f"{lvl}/{noise}"] += 1

    print(f"  Оригиналов:  {len(originals_names)}")
    print(f"  Ядер:        {len(kernels_names)}")
    print(f"  Вариантов на пару: 4 (clean + weak + medium + strong)")
    print(f"  Типы шумов: {len(NOISE_TYPES)} (round-robin со сдвигом)")
    print(f"  Всего изображений будет сгенерировано: {n_total}")
    print(f"    - из них чистых (только смаз): {n_clean}")
    print(f"    - из них шумных:               {n_noisy}")
    print(f"  Распределение по уровням и типам шума:")
    for key in sorted(level_counts):
        print(f"    {key:25s} → {level_counts[key]}")

    # Очищаем distorted перед новой генерацией
    for f in DISTORTED_DIR.glob("*.png"):
        f.unlink()

    # Генерация
    print("\n--- Генерация искаженных изображений ---")
    records = []
    for i, (orig, kernel, noise, lvl_label, lvl_param) in enumerate(design):
        orig_path = ORIGINALS_DIR / f"{orig}.png"

        # Имя файла: {orig}_{kernel}_{noise_type}_{level}.png
        # Если шума нет → "clean"
        if noise:
            noise_str = f"{lvl_label}{noise}"
        else:
            noise_str = "clean"
        out_name = f"{orig}_{kernel}_{noise_str}.png"
        out_path = DISTORTED_DIR / out_name

        metrics = process_one(orig_path, blur_filters[kernel], noise, lvl_param, out_path)

        delta_s = f"  Δ={metrics['delta_ssim']:.3f}" if noise else ""
        print(f"  [{i+1:4d}/{n_total}] {out_name:55s}  "
              f"blur:SSIM={metrics['blur_ssim']:.3f}  "
              f"final:SSIM={metrics['final_ssim']:.3f}{delta_s}")

        records.append({
            "filename": out_name,
            "original": orig,
            "kernel": kernel,
            "noise_type": noise if noise else "clean",
            "noise_level": lvl_label if noise else "",
            "noise_param": lvl_param if noise else None,
            "blur_psnr":  round(metrics["blur_psnr"], 4),
            "blur_ssim":  round(metrics["blur_ssim"], 4),
            "final_psnr": round(metrics["final_psnr"], 4),
            "final_ssim": round(metrics["final_ssim"], 4),
            "delta_psnr": round(metrics["delta_psnr"], 4),
            "delta_ssim": round(metrics["delta_ssim"], 4),
        })

    # Сохраняем дизайн в JSON
    json_path = OUTPUT_DATASET_DIR / "dataset_design.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # График распределения
    plot_path = OUTPUT_DATASET_DIR / "dataset_distribution.png"
    plot_dataset_distribution(records, plot_path)

    # Итог
    print(f"\n{'=' * 70}")
    print(f"  Готово! {len(records)} изображений сгенерировано.")
    print(f"  Датасет сохранен в:  {OUTPUT_DATASET_DIR}")
    print(f"  Design JSON:         {json_path}")
    print(f"  Plot:                {plot_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()