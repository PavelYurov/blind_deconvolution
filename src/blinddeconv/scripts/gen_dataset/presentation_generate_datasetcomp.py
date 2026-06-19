"""
generate_dataset_all_to_all.py

Генерация датасета с равномерным распределением уровней шума:
- По ТИПАМ шума: все на все (каждая пара изображение х ядро получает каждый тип).
- По УРОВНЯМ шума: сдвиг (weak - medium - strong - weak - …).

5 типов шума × 3 уровня каждый. На одну пару (изображение × ядро) приходится
6 искажённых версий: 1 clean + 5 типов шума (каждый с одним из 3 уровней).
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


FRAMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from blinddeconv.filters.blur import Kernel_convolution
from blinddeconv.filters.noise import (
    GaussianNoise, PoissonNoise, SaltAndPepperNoise, Pink_Noise, Brown_Noise,
)

#   Конфигурация директорий 

# Откуда берем исходные файлы (только PNG)
# UNIVERSAL_PART = "Levin"; SIZE_IMAGE = 255*255
# UNIVERSAL_PART = "Lai"; SIZE_IMAGE = 1024*682
# UNIVERSAL_PART = "Sun"; SIZE_IMAGE = 924*668
# UNIVERSAL_PART = "Kohler"; SIZE_IMAGE = 800*800
# UNIVERSAL_PART = "Set12"; SIZE_IMAGE = 512*512
UNIVERSAL_PART = "Grid_Test"; SIZE_IMAGE = 256*256

INPUT_ORIGINALS_DIR = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART / "originals"
INPUT_KERNELS_DIR   = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART / "ground_truth_filters"

# Куда сохраняем готовый датасет
OUTPUT_DATASET_DIR  = FRAMEWORK_ROOT / "images" / "compare_data" / "anton" / UNIVERSAL_PART

# Внутренняя структура датасета
ORIGINALS_DIR = OUTPUT_DATASET_DIR / "originals"
DISTORTED_DIR = OUTPUT_DATASET_DIR / "distorted"
FILTERS_DIR   = OUTPUT_DATASET_DIR / "ground_truth_filters"
TMP_DIR       = OUTPUT_DATASET_DIR / "tmp"


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
        {"label": "weak",   "param": [1, 1, SIZE_IMAGE // 1200]},
        {"label": "medium", "param": [1, 1, SIZE_IMAGE // 600]},
        {"label": "strong", "param": [1, 1, SIZE_IMAGE // 300]},
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
    return {
        "gaussian": lambda p: GaussianNoise(p),
        "poisson":  lambda p: PoissonNoise(p),
        "impulse":  lambda p: SaltAndPepperNoise(p),
        "pink":     lambda p: Pink_Noise(noise_level=p),
        "brown":    lambda p: Brown_Noise(noise_level=p),
    }[noise_type](level_param)



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
        kname = png_src.stem
        
        k = cv.imread(str(png_src), cv.IMREAD_GRAYSCALE).astype(np.float64)
        if k is None:
            print(f"  [Предупреждение] Не удалось прочитать ядро: {png_src}")
            continue
            
        k /= k.sum() + 1e-12

        npy_path = npy_dir / f"{kname}.npy"
        np.save(str(npy_path), k)

        dst_png = FILTERS_DIR / png_src.name
        if not dst_png.exists():
            shutil.copy2(str(png_src), str(dst_png))

        filters[kname] = Kernel_convolution(str(npy_path))
        print(f"  Ядро: {kname:25s} ({k.shape[1]}x{k.shape[0]})")

    return filters



def build_design_matrix(originals_list, kernels_list):
    """
    Генерирует матрицу назначений с равномерным распределением уровней шума.

    Стратегия:
      - Чистый вариант (без шума): каждый (оригинал × ядро).
      - Для каждого типа шума: все пары (оригинал × ядро) получают этот тип,
        но УРОВЕНЬ циклически чередуется (weak - medium - strong - weak - …).

    Итого на одну пару: 1 (clean) + 5 (типов шума) = 6 искажённых изображений.

    Возвращает список кортежей: (orig, kernel, noise_type, level_label, level_param)
    Для чистого варианта: noise_type="", level_label="", level_param=None
    """
    pairs = list(itertools.product(originals_list, kernels_list))

    design = []

    for orig, kern in pairs:
        design.append((orig, kern, "", "", None))

    for noise_type in NOISE_TYPES:
        levels = NOISE_LEVELS[noise_type]
        for idx, (orig, kern) in enumerate(pairs):
            lvl = levels[idx % len(levels)]
            design.append((orig, kern, noise_type, lvl["label"], lvl["param"]))

    return design


def compute_metrics(original, image):
    h = min(original.shape[0], image.shape[0])
    w = min(original.shape[1], image.shape[1])
    orig_f = original[:h, :w].astype(np.float64) / 255.0
    img_f  = np.clip(image[:h, :w].astype(np.float64) / 255.0, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(orig_f, img_f, data_range=1.0))
    ssim = float(structural_similarity(orig_f, img_f, data_range=1.0))
    return psnr, ssim


def process_one(orig_path, blur_filter, noise_type, level_param, output_path):
    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить: {orig_path}")

    blurred = blur_filter.filter(original)
    blur_psnr, blur_ssim = compute_metrics(original, blurred)

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



def main():
    print("=" * 70)
    print("  Генерация датасета (Режим: все на все)")
    print("=" * 70)

    if not INPUT_ORIGINALS_DIR.exists():
        print(f"ОШИБКА: Исходная папка с оригиналами не найдена: {INPUT_ORIGINALS_DIR}")
        return
    if not INPUT_KERNELS_DIR.exists():
        print(f"ОШИБКА: Исходная папка с ядрами не найдена: {INPUT_KERNELS_DIR}")
        return

    for d in[ORIGINALS_DIR, DISTORTED_DIR, FILTERS_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

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

    print("\n--- Подготовка ядер ---")
    blur_filters = prepare_kernels()
    kernels_names = list(blur_filters.keys())

    print("\n--- Матрица назначений ---")
    design = build_design_matrix(originals_names, kernels_names)

    n_total = len(design)
    n_clean = sum(1 for _, _, noise, _, _ in design if not noise)
    n_noisy = n_total - n_clean

    level_counts = Counter()
    for _, _, noise, lvl, _ in design:
        if noise:
            level_counts[f"{lvl}/{noise}"] += 1

    print(f"  Оригиналов:  {len(originals_names)}")
    print(f"  Ядер:        {len(kernels_names)}")
    print(f"  Типов шумов: {len(NOISE_TYPES)} × 3 уровня (round-robin)")
    print(f"  Всего изображений будет сгенерировано: {n_total}")
    print(f"    - из них чистых (только смаз): {n_clean}")
    print(f"    - из них шумных:               {n_noisy}")
    print(f"  Распределение уровней шума:")
    for key in sorted(level_counts):
        print(f"    {key:25s} → {level_counts[key]}")

    for f in DISTORTED_DIR.glob("*.png"):
        f.unlink()

    print("\n--- Генерация искаженных изображений ---")
    records = []
    for i, (orig, kernel, noise, lvl_label, lvl_param) in enumerate(design):
        orig_path = ORIGINALS_DIR / f"{orig}.png"

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

    json_path = OUTPUT_DATASET_DIR / "dataset_design.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    plot_path = OUTPUT_DATASET_DIR / "dataset_distribution.png"
    plot_dataset_distribution(records, plot_path)

    print(f"\n{'=' * 70}")
    print(f"  Готово! {len(records)} изображений сгенерировано.")
    print(f"  Датасет сохранен в:  {OUTPUT_DATASET_DIR}")
    print(f"  Design JSON:         {json_path}")
    print(f"  Plot:                {plot_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()