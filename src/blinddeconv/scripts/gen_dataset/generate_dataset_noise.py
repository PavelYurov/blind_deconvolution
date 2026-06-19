"""
generate_dataset_noise.py

Генерация датасета зашумленных изображений.

Для каждого оригинала применяются ВСЕ типы шума х ВСЕ уровни интенсивности.
5 типов × 3 уровня = 15 зашумленных версий на каждый оригинал.

Структура датасета:
    images/compare_data/noise/
        originals/     - исходные изображения
        distorted/     - зашумленные изображения

Именование зашумленных файлов: {оригинал}_{уровень}{тип}.png
    Пример: boy_weakgaussian.png, bridge_strongbrown.png
"""

import sys
import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

FRAMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from blinddeconv.filters.noise import (
    GaussianNoise, PoissonNoise, SaltAndPepperNoise, Pink_Noise, Brown_Noise,
)

DATASET_DIR   = FRAMEWORK_ROOT / "images" / "compare_data" / "noise"
ORIGINALS_DIR = DATASET_DIR / "originals"
DISTORTED_DIR = DATASET_DIR / "distorted"

# Конфигурация шумов: 5 типов × 3 уровня = 15 вариантов на каждый оригинал

NOISE_TYPES = ["gaussian", "poisson", "impulse", "pink", "brown"]

# Уровни шума для каждого типа (None для impulse — вычисляется из размера изображения)
NOISE_LEVELS = {
    "gaussian": [
        {"label": "weak",   "param": 10.0},
        {"label": "medium", "param": 25.0},
        {"label": "strong", "param": 50.0},
    ],
    "poisson": [
        {"label": "weak",   "param": 0.5},
        {"label": "medium", "param": 1.0},
        {"label": "strong", "param": 5.0},
    ],
    "impulse": [
        {"label": "weak",   "param": None},
        {"label": "medium", "param": None},
        {"label": "strong", "param": None},
    ],
    "pink": [
        {"label": "weak",   "param": 0.01},
        {"label": "medium", "param": 0.06},
        {"label": "strong", "param": 0.15},
    ],
    "brown": [
        {"label": "weak",   "param": 0.01},
        {"label": "medium", "param": 0.06},
        {"label": "strong", "param": 0.15},
    ],
}

# Множители для импульсного шума: доля пикселей изображения
IMPULSE_FRACTIONS = {
    "weak":   1 / 1200,
    "medium": 1 / 600,
    "strong": 1 / 5,
}


def _get_impulse_param(image_size: int, label: str):
    n_pixels = max(1, int(image_size * IMPULSE_FRACTIONS[label]))
    return [1, 1, n_pixels]


def make_noise_filter(noise_type: str, label: str, level_param, image_size: int):
    if noise_type == "gaussian":
        return GaussianNoise(level_param)
    elif noise_type == "poisson":
        return PoissonNoise(level_param)
    elif noise_type == "impulse":
        param = _get_impulse_param(image_size, label)
        return SaltAndPepperNoise(param)
    elif noise_type == "pink":
        return Pink_Noise(noise_level=level_param)
    elif noise_type == "brown":
        return Brown_Noise(noise_level=level_param)
    else:
        raise ValueError(f"Неизвестный тип шума: {noise_type}")


def compute_metrics(original: np.ndarray, image: np.ndarray):
    """PSNR и SSIM между original и image (оба uint8 grayscale)."""
    orig_f = original.astype(np.float64) / 255.0
    img_f  = np.clip(image.astype(np.float64) / 255.0, 0.0, 1.0)
    psnr = float(peak_signal_noise_ratio(orig_f, img_f, data_range=1.0))
    ssim = float(structural_similarity(orig_f, img_f, data_range=1.0))
    return psnr, ssim


def process_one(orig_path: Path, noise_type: str, label: str, level_param,
                output_path: Path):
    """Загружает оригинал, применяет шум, сохраняет, возвращает метрики."""
    original = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить: {orig_path}")

    image_size = original.shape[0] * original.shape[1]
    noise_filter = make_noise_filter(noise_type, label, level_param, image_size)

    noisy = noise_filter.filter(original)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    cv.imwrite(str(output_path), noisy)

    psnr, ssim = compute_metrics(original, noisy)
    return psnr, ssim



def plot_distribution(records: list, save_path: Path):
    """Scatter-plot: PSNR и SSIM по типам шума."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = {
        "gaussian": "tab:blue",
        "poisson":  "tab:orange",
        "impulse":  "tab:green",
        "pink":     "tab:red",
        "brown":    "tab:purple",
    }
    markers = {"weak": "o", "medium": "s", "strong": "^"}

    by_type = {}
    for r in records:
        nt = r["noise_type"]
        by_type.setdefault(nt, []).append(r)

    for noise_type, recs in by_type.items():
        c = colors.get(noise_type, "gray")
        for lvl, mk in markers.items():
            subset = [r for r in recs if r["noise_level"] == lvl]
            if not subset:
                continue
            axes[0].scatter(
                [r["ssim"] for r in subset],
                [r["psnr"] for r in subset],
                color=c, marker=mk, alpha=0.6, s=60,
                label=f"{noise_type}/{lvl}" if mk == "o" else "_",
            )
            axes[1].scatter(
                [r["ssim"] for r in subset],
                [r["psnr"] for r in subset],
                color=c, marker=mk, alpha=0.6, s=60,
            )

    for ax, title in zip(axes, ["PSNR vs SSIM (все типы)", "PSNR vs SSIM (легенда)"]):
        ax.set_xlabel("SSIM")
        ax.set_ylabel("PSNR, дБ")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close()
    print(f"  График сохранён: {save_path}")


def main():
    print("=" * 70)
    print("  Генерация датасета зашумленных изображений")
    print("=" * 70)

    # Проверка директорий
    if not ORIGINALS_DIR.exists():
        print(f"ОШИБКА: Папка с оригиналами не найдена: {ORIGINALS_DIR}")
        return
    orig_files = list(ORIGINALS_DIR.glob("*.png"))
    if not orig_files:
        print(f"ОШИБКА: В {ORIGINALS_DIR} нет PNG файлов!")
        return

    DISTORTED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Оригиналов: {len(orig_files)}")
    print(f"  Типов шума: {len(NOISE_TYPES)} × 3 уровня = "
          f"{len(NOISE_TYPES) * 3} вариантов на оригинал")
    total = len(orig_files) * len(NOISE_TYPES) * 3
    print(f"  Всего будет сгенерировано: {total} изображений")

    existing = list(DISTORTED_DIR.glob("*.png"))
    if existing:
        print(f"\n  Удаляем {len(existing)} существующих файлов в distorted/...")
        for f in existing:
            f.unlink()

    print("\n--- Генерация ---")
    records = []
    counter = 0

    for orig_path in sorted(orig_files):
        orig_name = orig_path.stem
        print(f"\n  [{orig_name}]")

        for noise_type in NOISE_TYPES:
            for lvl_dict in NOISE_LEVELS[noise_type]:
                label     = lvl_dict["label"]
                level_param = lvl_dict["param"]

                # Имя файла: {оригинал}_{уровень}{тип}.png
                noise_str = f"{label}{noise_type}"
                out_name  = f"{orig_name}_{noise_str}.png"
                out_path  = DISTORTED_DIR / out_name

                psnr, ssim = process_one(
                    orig_path, noise_type, label, level_param, out_path)

                counter += 1
                print(f"    [{counter:4d}/{total}] {out_name:45s}  "
                      f"PSNR={psnr:6.2f} дБ  SSIM={ssim:.4f}")

                records.append({
                    "filename":    out_name,
                    "original":    orig_name,
                    "noise_type":  noise_type,
                    "noise_level": label,
                    "noise_param": level_param,
                    "psnr":        round(psnr, 4),
                    "ssim":        round(ssim, 4),
                })

    json_path = DATASET_DIR / "dataset_design.json"
    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\n  Design JSON сохранён: {json_path}")

    plot_path = DATASET_DIR / "dataset_distribution.png"
    plot_distribution(records, plot_path)

    print(f"\n{'=' * 70}")
    print(f"  Готово! {len(records)} изображений сгенерировано.")
    print(f"  Датасет: {DATASET_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
