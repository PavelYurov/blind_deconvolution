"""
generate_dataset_big.py

Генерация БОЛЬШОГО датасета: 75 смазанных изображений.
15 оригиналов × 5 ядер (по 1 из каждого класса) = 75 пар.
25 clean (33%) + 50 noisy (67%).
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

DATASET_DIR   = FRAMEWORK_ROOT / "images" / "new_dataset_big_pics"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"
FILTERS_DIR   = DATASET_DIR / "ground_truth_filters"
TMP_DIR       = DATASET_DIR / "tmp"

# Откуда берём PNG ядер (мастер-копия)
KERNEL_SOURCE_DIR = FRAMEWORK_ROOT / "images" / "new_dataset_small_pics" / "ground_truth_filters"

# Откуда копировать оригиналы (256×256, уже обрезанные)
ORIGINALS_SOURCE_DIR = FRAMEWORK_ROOT / "images" / "new_dataset_big_pics_test" / "originals"

# 15 оригиналов (должны лежать в ORIGINALS_DIR как {name}.png)
ORIGINALS = [
    "airplane", "boat", "boy", "bridge", "cameraman",
    "childs", "drawing", "house", "lenna", "monarch",
    "parrot", "people", "pepper", "scarf", "star",
]

# 15 ядер: 5 классов × 3
KERNEL_CLASSES = [
    ["thick_motion", "linear", "largemation"],     # Linear
    ["dendric", "hook", "heliod"],                  # Trajectory
    ["defocus", "moon", "pacman"],                  # Area
    ["comet", "defocusmotion", "hookdefocus"],      # Mixed
    ["spiral", "zspiral", "wspiral"],               # Complex
]

CLASS_NAMES = ["Linear", "Trajectory", "Area", "Mixed", "Complex"]

# Файлы ядер (PNG) — маппинг имя → файл
KERNEL_FILES = {
    "thick_motion":  "thick_motion_kernel.png",
    "linear":        "linear_kernel.png",
    "largemation":   "largemation_kernel.png",
    "dendric":       "dendric_kernel.png",
    "hook":          "hook_kernel.png",
    "heliod":        "heliod_kernel.png",
    "defocus":       "defocus_kernel.png",
    "moon":          "moon_kernel.png",
    "pacman":        "pacman_kernel.png",
    "comet":         "comet_kernel.png",
    "defocusmotion": "defocusmotion_kernel.png",
    "hookdefocus":   "hookdefocus_kernel.png",
    "spiral":        "spiral_kernel.png",
    "zspiral":       "zspiral_kernel.png",
    "wspiral":       "wspiral_kernel.png",
}

# Шумы (те же параметры, что в dataset_gen_small_pics.ipynb)
NOISE_TYPES = ["gaussian", "poisson", "impulse", "pink", "brown"]

def make_noise_filter(noise_type):
    """Создаёт фильтр шума по типу."""
    return {
        "gaussian": lambda: GaussianNoise(5.0),
        "poisson":  lambda: PoissonNoise(0.35),
        "impulse":  lambda: SaltAndPepperNoise([1, 1, 600]),
        "pink":     lambda: Pink_Noise(noise_level=0.02),
        "brown":    lambda: Brown_Noise(noise_level=0.02),
    }[noise_type]()

N_CLEAN = 25   # из 75 → 33% clean, 67% noisy
SEED = 42


# ═══════════════════════════════════════════════════════════════════
#   Подготовка ядер: PNG → npy + Kernel_convolution
# ═══════════════════════════════════════════════════════════════════

def prepare_kernels():
    """Загружает PNG ядер, конвертирует в .npy, возвращает dict фильтров."""
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

        # Загрузка и нормализация
        k = cv.imread(str(png_src), cv.IMREAD_GRAYSCALE).astype(np.float64)
        k /= k.sum() + 1e-12

        # Сохраняем .npy для Kernel_convolution
        npy_path = npy_dir / f"{kname}.npy"
        np.save(str(npy_path), k)

        # Копируем PNG в ground_truth_filters датасета
        dst_png = FILTERS_DIR / kfile
        if not dst_png.exists():
            shutil.copy2(str(png_src), str(dst_png))

        filters[kname] = Kernel_convolution(str(npy_path))
        print(f"  Ядро: {kname:20s} ({k.shape[1]}x{k.shape[0]})")

    return filters


# ═══════════════════════════════════════════════════════════════════
#   Матрица назначений (design matrix)
# ═══════════════════════════════════════════════════════════════════

def build_design_matrix():
    """
    Сбалансированное назначение: (original, kernel, noise_or_empty).

    Каждый оригинал получает ровно 1 ядро из каждого класса (= 5 ядер).
    Каждое ядро используется ровно 5 раз.
    N_CLEAN записей — clean (noise=""), остальные — шум (распределён по типам).
    """
    np.random.seed(SEED)
    n_orig = len(ORIGINALS)
    n_classes = len(KERNEL_CLASSES)

    # Шаг 1: Назначение ядер (balanced pattern)
    # Для класса c, оригинал i → индекс ядра в классе
    pairs = []  # (orig, kernel_name, class_idx)
    for i, orig in enumerate(ORIGINALS):
        for c, cls_kernels in enumerate(KERNEL_CLASSES):
            n_k = len(cls_kernels)
            # Разные паттерны для разных классов чтобы избежать повторов
            if c < 3:
                k_idx = (i + c) % n_k
            elif c == 3:
                k_idx = (i * 2) % n_k
            else:
                k_idx = (i * 2 + 1) % n_k
            pairs.append((orig, cls_kernels[k_idx], c))

    # Шаг 2: Распределяем clean/noisy по оригиналам
    n_total = len(pairs)
    n_noisy = n_total - N_CLEAN
    base_clean = N_CLEAN // n_orig          # = 1
    extra_clean = N_CLEAN % n_orig          # = 10
    clean_per_orig = [base_clean + (1 if i < extra_clean else 0)
                      for i in range(n_orig)]
    np.random.shuffle(clean_per_orig)

    # Шаг 3: Пул шумов (сбалансированный по типам)
    n_per_type = n_noisy // len(NOISE_TYPES)
    n_extra = n_noisy % len(NOISE_TYPES)
    noise_pool = []
    for j, nt in enumerate(NOISE_TYPES):
        noise_pool.extend([nt] * (n_per_type + (1 if j < n_extra else 0)))
    np.random.shuffle(noise_pool)
    noise_iter = iter(noise_pool)

    # Шаг 4: Финальная матрица
    result = []
    for i, orig in enumerate(ORIGINALS):
        orig_pairs = [(o, k, c) for o, k, c in pairs if o == orig]
        # Случайно выбираем, какие слоты — clean
        class_order = list(range(n_classes))
        np.random.shuffle(class_order)
        clean_set = set(class_order[:clean_per_orig[i]])

        for o, k, c in orig_pairs:
            if c in clean_set:
                result.append((o, k, ""))
            else:
                result.append((o, k, next(noise_iter)))

    return result


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


def process_one(orig_path, blur_filter, noise_type, output_path):
    """
    Применяет blur (+ noise) через фреймворк, сохраняет.
    Возвращает dict с метриками blur-only и финальными.
    """
    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить: {orig_path}")

    # Смаз через фреймворк
    blurred = blur_filter.filter(original)

    # Метрики чистого смаза (всегда считаем)
    blur_psnr, blur_ssim = compute_metrics(original, blurred)

    # Шум через фреймворк (если есть)
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
    """2D scatter: X = blur metric, Y = delta (blur - final). SSIM и PSNR."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    noisy  = [r for r in records if r["noise"]]
    clean  = [r for r in records if not r["noise"]]

    # --- SSIM ---
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

    # --- PSNR ---
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
    print("  Генерация датасета: BIG (75 изображений)")
    print("  15 оригиналов × 5 ядер, 25 clean + 50 noisy")
    print("=" * 65)

    # Создаём папки
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

    # Подготовка ядер
    print("\n--- Подготовка ядер ---")
    blur_filters = prepare_kernels()

    # Матрица назначений
    print("\n--- Матрица назначений ---")
    design = build_design_matrix()

    # Статистика
    kernel_counts = Counter(k for _, k, _ in design)
    noise_counts  = Counter(n if n else "CLEAN" for _, _, n in design)
    print(f"  Всего: {len(design)}")
    print(f"  По ядрам: { {k: kernel_counts[k] for k in sorted(kernel_counts)} }")
    print(f"  По шуму:  {dict(noise_counts)}")

    # Очищаем distorted
    for f in DISTORTED_DIR.glob("*.png"):
        f.unlink()

    # Генерация
    print("\n--- Генерация смазанных изображений ---")
    records = []
    for i, (orig, kernel, noise) in enumerate(design):
        orig_path = ORIGINALS_DIR / f"{orig}.png"

        # Имя файла: {orig}_{kernel}_{noise}.png   (_ в конце = clean)
        out_name = f"{orig}_{kernel}_{noise}.png"
        out_path = DISTORTED_DIR / out_name

        metrics = process_one(orig_path, blur_filters[kernel], noise, out_path)

        label = f"blur+{noise}" if noise else "blur only"
        delta_s = f"  Δ={metrics['delta_ssim']:.3f}" if noise else ""
        print(f"  [{i+1:3d}/75] {out_name:50s}  "
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

    # Сохраняем дизайн в JSON
    json_path = DATASET_DIR / "dataset_design.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # График распределения
    plot_path = DATASET_DIR / "dataset_distribution.png"
    plot_dataset_distribution(records, plot_path)

    # Итог
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
