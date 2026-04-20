"""
generate_dataset_mid.py

Генерация СРЕДНЕГО датасета: 50 смазанных изображений.
15 оригиналов, 10 ядер (5 классов x 2).
Каждый оригинал получает 3–4 ядра из разных классов = 50 пар.
17 clean (34%) + 33 noisy (66%).
5 типов шума: gaussian, poisson, impulse, pink, brown.

Использует классы фильтров из фреймворка blinddeconv.
"""

import sys
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── Путь к фреймворку ───────────────────────────────────────────
FRAMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from blinddeconv.filters.blur import Kernel_convolution
from blinddeconv.filters.noise import (
    GaussianNoise, PoissonNoise, SaltAndPepperNoise, Pink_Noise, Brown_Noise,
)

# ═══════════════════════════════════════════════════════════════════
#   Конфигурация датасета
# ═══════════════════════════════════════════════════════════════════

DATASET_DIR   = FRAMEWORK_ROOT / "images" / "new_dataset_mid_pics"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"
FILTERS_DIR   = DATASET_DIR / "ground_truth_filters"
TMP_DIR       = DATASET_DIR / "tmp"

# Откуда берём PNG ядер (мастер-копия)
KERNEL_SOURCE_DIR = FRAMEWORK_ROOT / "images" / "new_dataset_small_pics" / "ground_truth_filters"

# Откуда копировать оригиналы (256×256, уже обрезанные)
ORIGINALS_SOURCE_DIR = FRAMEWORK_ROOT / "images" / "new_dataset_big_pics_test" / "originals"

# 15 оригиналов (те же, что в big)
ORIGINALS = [
    "airplane", "boat", "boy", "bridge", "cameraman",
    "childs", "drawing", "house", "lenna", "monarch",
    "parrot", "people", "pepper", "scarf", "star",
]

# 10 ядер: 5 классов × 2 (drop thick_motion, hook, pacman, defocusmotion, zspiral)
KERNEL_CLASSES = [
    ["linear", "largemation"],        # Linear
    ["dendric", "heliod"],            # Trajectory
    ["defocus", "moon"],              # Area
    ["comet", "hookdefocus"],         # Mixed
    ["spiral", "wspiral"],            # Complex
]

CLASS_NAMES = ["Linear", "Trajectory", "Area", "Mixed", "Complex"]

KERNEL_FILES = {
    "linear":        "linear_kernel.png",
    "largemation":   "largemation_kernel.png",
    "dendric":       "dendric_kernel.png",
    "heliod":        "heliod_kernel.png",
    "defocus":       "defocus_kernel.png",
    "moon":          "moon_kernel.png",
    "comet":         "comet_kernel.png",
    "hookdefocus":   "hookdefocus_kernel.png",
    "spiral":        "spiral_kernel.png",
    "wspiral":       "wspiral_kernel.png",
}

NOISE_TYPES = ["gaussian", "poisson", "impulse", "pink", "brown"]

def make_noise_filter(noise_type):
    return {
        "gaussian": lambda: GaussianNoise(5.0),
        "poisson":  lambda: PoissonNoise(0.35),
        "impulse":  lambda: SaltAndPepperNoise([1, 1, 600]),
        "pink":     lambda: Pink_Noise(noise_level=0.02),
        "brown":    lambda: Brown_Noise(noise_level=0.02),
    }[noise_type]()

N_TOTAL = 50
N_CLEAN = 17   # из 50 -> 34% clean, 66% noisy
SEED = 42


# ═══════════════════════════════════════════════════════════════════
#   Подготовка ядер
# ═══════════════════════════════════════════════════════════════════

def prepare_kernels():
    import shutil

    npy_dir = TMP_DIR / "kernels_npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)

    filters = {}
    all_kernel_names = set()
    for cls in KERNEL_CLASSES:
        all_kernel_names.update(cls)

    for kname in sorted(all_kernel_names):
        kfile = KERNEL_FILES[kname]
        png_src = KERNEL_SOURCE_DIR / kfile

        if not png_src.exists():
            raise FileNotFoundError(f"Ядро не найдено: {png_src}")

        k = cv.imread(str(png_src), cv.IMREAD_GRAYSCALE).astype(np.float64)
        k /= k.sum() + 1e-12

        npy_path = npy_dir / f"{kname}.npy"
        np.save(str(npy_path), k)

        dst_png = FILTERS_DIR / kfile
        if not dst_png.exists():
            shutil.copy2(str(png_src), str(dst_png))

        filters[kname] = Kernel_convolution(str(npy_path))
        print(f"  Ядро: {kname:20s} ({k.shape[1]}x{k.shape[0]})")

    return filters


# ═══════════════════════════════════════════════════════════════════
#   Матрица назначений
# ═══════════════════════════════════════════════════════════════════

def build_design_matrix():
    """
    15 оригиналов, 10 ядер (5 классов x 2), выбираем 50 пар.
    Каждый оригинал получает 3–4 ядра из разных классов.
    Каждое ядро используется 5 раз.
    17 clean + 33 noisy.
    """
    np.random.seed(SEED)
    n_orig = len(ORIGINALS)       # 15
    n_classes = len(KERNEL_CLASSES)  # 5
    n_kernels = sum(len(c) for c in KERNEL_CLASSES)  # 10

    # Шаг 1: Полная матрица 15 x 5 классов = 75 пар
    full_pairs = []
    for i, orig in enumerate(ORIGINALS):
        for c, cls_kernels in enumerate(KERNEL_CLASSES):
            k_idx = (i + c) % len(cls_kernels)
            full_pairs.append((orig, cls_kernels[k_idx], c, i))

    # Шаг 2: Отбираем N_TOTAL=50 из 75.
    # Каждый оригинал получает 3 или 4 класса (50/15 ~ 3.33).
    n_drop = len(full_pairs) - N_TOTAL   # 25
    base_drop = n_drop // n_orig          # 1
    extra_drop = n_drop % n_orig          # 10
    drop_per_orig = [base_drop + (1 if i < extra_drop else 0)
                     for i in range(n_orig)]
    np.random.shuffle(drop_per_orig)

    pairs = []
    for i, orig in enumerate(ORIGINALS):
        orig_items = [(o, k, c) for o, k, c, oi in full_pairs if oi == i]
        # Убираем drop_per_orig[i] случайных классов
        class_indices = list(range(n_classes))
        np.random.shuffle(class_indices)
        drop_classes = set(class_indices[:drop_per_orig[i]])
        for o, k, c in orig_items:
            if c not in drop_classes:
                pairs.append((o, k, c))

    # Шаг 3: clean/noisy
    n_noisy = len(pairs) - N_CLEAN
    base_clean = N_CLEAN // n_orig
    extra_clean = N_CLEAN % n_orig
    clean_per_orig = [base_clean + (1 if i < extra_clean else 0)
                      for i in range(n_orig)]
    np.random.shuffle(clean_per_orig)

    # Пул шумов
    n_per_type = n_noisy // len(NOISE_TYPES)
    n_extra = n_noisy % len(NOISE_TYPES)
    noise_pool = []
    for j, nt in enumerate(NOISE_TYPES):
        noise_pool.extend([nt] * (n_per_type + (1 if j < n_extra else 0)))
    np.random.shuffle(noise_pool)
    noise_iter = iter(noise_pool)

    result = []
    for i, orig in enumerate(ORIGINALS):
        orig_pairs = [(o, k, c) for o, k, c in pairs if o == orig]
        class_order = [c for _, _, c in orig_pairs]
        np.random.shuffle(class_order)
        clean_set = set(class_order[:clean_per_orig[i]])

        for o, k, c in orig_pairs:
            if c in clean_set:
                result.append((o, k, ""))
                clean_set.discard(c)  # только 1 clean на класс
            else:
                result.append((o, k, next(noise_iter)))

    return result


# ═══════════════════════════════════════════════════════════════════
#   Применение фильтров + метрики
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(original, image):
    h = min(original.shape[0], image.shape[0])
    w = min(original.shape[1], image.shape[1])
    orig_f = original[:h, :w].astype(np.float64) / 255.0
    img_f  = np.clip(image[:h, :w].astype(np.float64) / 255.0, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(orig_f, img_f, data_range=1.0))
    ssim = float(structural_similarity(orig_f, img_f, data_range=1.0))
    return psnr, ssim


def process_one(orig_path, blur_filter, noise_type, output_path):
    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить: {orig_path}")

    blurred = blur_filter.filter(original)
    blur_psnr, blur_ssim = compute_metrics(original, blurred)

    if noise_type:
        noise_filter = make_noise_filter(noise_type)
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    noisy  = [r for r in records if r["noise"]]
    clean  = [r for r in records if not r["noise"]]

    ax = axes[0]
    if noisy:
        ax.scatter([r["blur_ssim"] for r in noisy],
                   [r["delta_ssim"] for r in noisy],
                   c="tab:blue", alpha=0.7, label=f"Шумное ({len(noisy)})")
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
        ax.scatter([r["blur_psnr"] for r in noisy],
                   [r["delta_psnr"] for r in noisy],
                   c="tab:blue", alpha=0.7, label=f"Шумное ({len(noisy)})")
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
    print(f"  Plot saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Генерация датасета: MID (50 изображений)")
    print("  15 оригиналов, 10 ядер, 17 clean + 33 noisy")
    print("=" * 65)

    for d in [ORIGINALS_DIR, DISTORTED_DIR, FILTERS_DIR, TMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Копируем оригиналы из источника (если ещё не скопированы)
    import shutil as _shutil
    if ORIGINALS_SOURCE_DIR.exists():
        for orig in ORIGINALS:
            dst = ORIGINALS_DIR / f"{orig}.png"
            src = ORIGINALS_SOURCE_DIR / f"{orig}.png"
            if not dst.exists() and src.exists():
                _shutil.copy2(str(src), str(dst))
                print(f"  Скопирован: {orig}.png")

    # Проверяем оригиналы
    missing = [o for o in ORIGINALS if not (ORIGINALS_DIR / f"{o}.png").exists()]
    if missing:
        print(f"\nОШИБКА: Не найдены оригиналы ({len(missing)} шт):")
        for m in missing:
            print(f"  {ORIGINALS_DIR / (m + '.png')}")
        print(f"\nПоложите {len(ORIGINALS)} изображений в:\n  {ORIGINALS_DIR}")
        return

    # Ядра
    print("\n--- Подготовка ядер ---")
    blur_filters = prepare_kernels()

    # Матрица
    print("\n--- Матрица назначений ---")
    design = build_design_matrix()

    kernel_counts = Counter(k for _, k, _ in design)
    noise_counts  = Counter(n if n else "CLEAN" for _, _, n in design)
    print(f"  Всего: {len(design)}")
    print(f"  По ядрам: { {k: kernel_counts[k] for k in sorted(kernel_counts)} }")
    print(f"  По шуму:  {dict(noise_counts)}")

    # Очищаем
    for f in DISTORTED_DIR.glob("*.png"):
        f.unlink()

    # Генерация
    print("\n--- Генерация смазанных изображений ---")
    records = []
    for i, (orig, kernel, noise) in enumerate(design):
        orig_path = ORIGINALS_DIR / f"{orig}.png"

        out_name = f"{orig}_{kernel}_{noise}.png"
        out_path = DISTORTED_DIR / out_name

        metrics = process_one(orig_path, blur_filters[kernel], noise, out_path)

        label = f"blur+{noise}" if noise else "blur only"
        delta_s = f"  Δ={metrics['delta_ssim']:.3f}" if noise else ""
        print(f"  [{i+1:3d}/50] {out_name:50s}  "
              f"blur:PSNR={metrics['blur_psnr']:5.1f}/SSIM={metrics['blur_ssim']:.3f}  "
              f"final:PSNR={metrics['final_psnr']:5.1f}/SSIM={metrics['final_ssim']:.3f}"
              f"{delta_s}  ({label})")

        records.append({
            "filename": out_name,
            "original": orig,
            "kernel": kernel,
            "noise": noise if noise else "",
            "blur_psnr":  round(metrics["blur_psnr"], 4),
            "blur_ssim":  round(metrics["blur_ssim"], 4),
            "final_psnr": round(metrics["final_psnr"], 4),
            "final_ssim": round(metrics["final_ssim"], 4),
            "delta_psnr": round(metrics["delta_psnr"], 4),
            "delta_ssim": round(metrics["delta_ssim"], 4),
        })

    # JSON
    json_path = DATASET_DIR / "dataset_design.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # График распределения
    plot_path = DATASET_DIR / "dataset_distribution.png"
    plot_dataset_distribution(records, plot_path)

    n_clean = noise_counts.get("CLEAN", 0)
    n_noisy = len(design) - n_clean
    print(f"\n{'=' * 65}")
    print(f"  Готово! {len(records)} изображений сгенерировано.")
    print(f"  Clean: {n_clean}, Noisy: {n_noisy}")
    print(f"  Distorted:  {DISTORTED_DIR}")
    print(f"  Design JSON: {json_path}")
    print(f"  Plot:        {plot_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
