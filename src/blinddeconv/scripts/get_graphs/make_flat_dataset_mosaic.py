# -*- coding: utf-8 -*-
"""
Собирает полотна с примерами изображений и ядрами для наборов данных:
  images/large_data_pictures
  images/middle_data_pictures

В каждом из них ожидается структура:
  originals/            - оригинальные изображения (*.png)
  ground_truth_filters/ - ядра (*.png)
  <image>_<kernel>_<noise>.png  - смазанные примеры

Для каждого датасета создаются 4 файла прямо в папке датасета:
  originals_mosaic_vertical.png    - оригиналы: больше строк (3 столбца)
  originals_mosaic_horizontal.png  - оригиналы: больше столбцов (5 столбцов)
  kernels_mosaic_vertical.png      - ядра:      больше строк
  kernels_mosaic_horizontal.png    - ядра:      больше столбцов

"""

from pathlib import Path
import json
import math
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent / "images"
LABELS_CONFIG = Path(__file__).parent / "presentation_labels.json"
DATASETS = [
    "large_data_pictures",
    "middle_data_pictures",
]
DPI       = 200
IMG_THUMB = 256   # макс. размер превью оригинала (пикселей)
KERN_DISP = 31    # сторона холста для ядра (пикселей)


def _load_label_map() -> dict:
    if not LABELS_CONFIG.exists():
        return {}
    try:
        raw = json.loads(LABELS_CONFIG.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return {k: v for k, v in raw.items()
                if isinstance(k, str) and not k.startswith("_")
                and isinstance(v, str)}
    except Exception as e:
        print(f"  Ошибка чтения {LABELS_CONFIG}: {e}")
        return {}


_LABEL_MAP: dict = {}


def _label(name: str) -> str:
    return _LABEL_MAP.get(name, name)



def _load_gray(path: Path, max_dim: int = IMG_THUMB) -> np.ndarray:
    im = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    h, w = im.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        im = cv.resize(im, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv.INTER_AREA)
    return im


def _load_kernel(path: Path, display_size: int = KERN_DISP) -> np.ndarray:
    k = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if k is None:
        raise FileNotFoundError(path)
    kh, kw = k.shape
    canvas = np.zeros((display_size, display_size), dtype=k.dtype)
    if kh > display_size or kw > display_size:
        scale = display_size / max(kh, kw)
        k = cv.resize(k, (max(1, int(kw * scale)), max(1, int(kh * scale))),
                      interpolation=cv.INTER_AREA)
        kh, kw = k.shape
    y0 = (display_size - kh) // 2
    x0 = (display_size - kw) // 2
    canvas[y0:y0 + kh, x0:x0 + kw] = k
    return canvas


def _stem(filename: str) -> str:
    s = Path(filename).stem
    return s.removesuffix("_kernel")


def _best_grid(n: int, prefer_horizontal: bool):
    """
    prefer_horizontal=True - ncols > nrows (широкая сетка).
    prefer_horizontal=False - nrows > ncols (высокая сетка).
    """
    ncols = math.ceil(math.sqrt(n))
    if prefer_horizontal:
        while ncols <= n and n % ncols != 0:
            ncols += 1
        if ncols > n:
            ncols = n
        nrows = math.ceil(n / ncols)
    else:
        nrows = math.ceil(math.sqrt(n))
        ncols = math.ceil(n / nrows)
        if ncols > nrows:
            nrows, ncols = ncols, nrows
    return nrows, ncols


def _render_images(files: list, nrows: int, ncols: int,
                   out_path: Path, title: str):
    cell_in = IMG_THUMB / DPI * 1.1
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell_in + 1.2,
                                      nrows * cell_in + 0.5),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.25})
    axes_flat = np.atleast_1d(axes).ravel()

    for i, f in enumerate(files):
        img = _load_gray(f)
        ax = axes_flat[i]
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(_label(_stem(f.name)), fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for j in range(len(files), len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def build_originals(ds_dir: Path):
    files = sorted((ds_dir / "originals").glob("*.png"))
    if not files:
        print(f"  No originals found in {ds_dir}")
        return

    n = len(files)
    title = f"{ds_dir.name} — originals ({n})"

    # Вертикальный (больше строк)
    nr_v, nc_v = _best_grid(n, prefer_horizontal=False)
    _render_images(files, nr_v, nc_v,
                   ds_dir / "originals_mosaic_vertical.png",
                   title)

    # Горизонтальный (больше столбцов)
    nr_h, nc_h = _best_grid(n, prefer_horizontal=True)
    _render_images(files, nr_h, nc_h,
                   ds_dir / "originals_mosaic_horizontal.png",
                   title)


def _render_kernels(files: list, nrows: int, ncols: int,
                    out_path: Path, title: str):
    cell_in = KERN_DISP / DPI * 1.8 
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * (cell_in + 0.9),
                                      nrows * (cell_in + 0.9)),
                             gridspec_kw={"wspace": 0.2, "hspace": 0.4})
    axes_flat = np.atleast_1d(axes).ravel()

    for i, f in enumerate(files):
        k = _load_kernel(f, KERN_DISP)
        ax = axes_flat[i]
        ax.imshow(k, cmap='gray', interpolation='nearest')
        ax.set_title(_label(_stem(f.name)), fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for j in range(len(files), len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def build_kernels(ds_dir: Path):
    files = sorted((ds_dir / "ground_truth_filters").glob("*.png"))
    if not files:
        print(f"  No kernels found in {ds_dir}")
        return

    n = len(files)
    title = f"{ds_dir.name} — kernels ({n})"

    nr_v, nc_v = _best_grid(n, prefer_horizontal=False)
    _render_kernels(files, nr_v, nc_v,
                    ds_dir / "kernels_mosaic_vertical.png",
                    title)

    nr_h, nc_h = _best_grid(n, prefer_horizontal=True)
    _render_kernels(files, nr_h, nc_h,
                    ds_dir / "kernels_mosaic_horizontal.png",
                    title)


def main():
    global _LABEL_MAP
    _LABEL_MAP = _load_label_map()
    if _LABEL_MAP:
        print(f"Загружены подписи ({len(_LABEL_MAP)} ключей) из {LABELS_CONFIG.name}")

    for ds_name in DATASETS:
        ds_dir = BASE_DIR / ds_name
        if not ds_dir.exists():
            print(f"[SKIP] {ds_dir} not found")
            continue
        print(f"\n[{ds_name}]")
        build_originals(ds_dir)
        build_kernels(ds_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
