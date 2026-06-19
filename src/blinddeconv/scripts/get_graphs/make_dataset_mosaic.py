# -*- coding: utf-8 -*-
"""
Собирает полотна с примерами изображений датасетов и ядрами.

Для каждого набора данных создаётся 4 файла в images/compare_data/<user>/:
  dataset_mosaic_vertical.png    - вертикальная компоновка (4 строки × 3 изображения)
  dataset_mosaic_horizontal.png  - горизонтальная компоновка (3 изображения × 4 столбца)
  kernels_mosaic_vertical.png    - ядра в сетке 3 строки × 2 столбца
  kernels_mosaic_horizontal.png  - ядра в сетке 2 строки × 3 столбца
"""

from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE_DIR   = Path(__file__).parent / "images" / "compare_data"
USERS      = ["anton", "kostya", "pasha"]
DATASETS   = ["Levin", "Set12", "Kohler", "Sun"]
N_IMAGES   = 3
DPI        = 200
IMG_THUMB  = 256  
KERN_DISP  = 31   

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


def _orig_size_label(ds_dir: Path) -> str:
    origs_dir = ds_dir / "originals"
    files = sorted(origs_dir.glob("*.png")) if origs_dir.exists() else []
    if not files:
        for ext in ("*.jpg", "*.jpeg", "*.bmp", "*.tif"):
            files = sorted(origs_dir.glob(ext)) if origs_dir.exists() else []
            if files:
                break
    if files:
        im = cv.imread(str(files[0]))
        if im is not None:
            h, w = im.shape[:2]
            return f"{ds_dir.name} ({w} × {h})"
    return ds_dir.name


def _collect_images(ds_dir: Path, n: int = N_IMAGES):
    origs_dir = ds_dir / "originals"
    files = sorted(origs_dir.glob("*.png")) if origs_dir.exists() else []
    if not files:
        return []
    return files[:n]


def _collect_kernels(ds_dir: Path):
    kdir = ds_dir / "ground_truth_filters"
    return sorted(kdir.glob("*.png")) if kdir.exists() else []



def build_vertical(user_dir: Path):
    n_ds = len(DATASETS)
    n_im = N_IMAGES

    rows = []
    for ds_name in DATASETS:
        ds_dir = user_dir / ds_name
        files  = _collect_images(ds_dir, n_im)
        label  = _orig_size_label(ds_dir)
        imgs   = [_load_gray(f) for f in files] if files else []
        rows.append((label, imgs))

    fig, axes = plt.subplots(n_ds, n_im,
                             figsize=(n_im * 2.8 + 1.4, n_ds * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    for r, (label, imgs) in enumerate(rows):
        for c in range(n_im):
            ax = axes[r][c]
            if c < len(imgs):
                ax.imshow(imgs[c], cmap='gray', vmin=0, vmax=255)
            else:
                ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[r][0].set_ylabel(label, fontsize=10, labelpad=6)

    out = user_dir / "dataset_mosaic_vertical.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def build_horizontal(user_dir: Path):
    n_ds = len(DATASETS)
    n_im = N_IMAGES

    cols = []
    for ds_name in DATASETS:
        ds_dir = user_dir / ds_name
        files  = _collect_images(ds_dir, n_im)
        label  = _orig_size_label(ds_dir)
        imgs   = [_load_gray(f) for f in files] if files else []
        cols.append((label, imgs))

    fig, axes = plt.subplots(n_im, n_ds,
                             figsize=(n_ds * 2.8, n_im * 2.8),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.15})

    for c, (label, imgs) in enumerate(cols):
        for r in range(n_im):
            ax = axes[r][c]
            if r < len(imgs):
                ax.imshow(imgs[r], cmap='gray', vmin=0, vmax=255)
            else:
                ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0][c].set_title(label, fontsize=10, pad=5)

    out = user_dir / "dataset_mosaic_horizontal.png"
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def _render_kernels(kernel_files: list, nrows: int, ncols: int, user_dir: Path, out_name: str):
    n = len(kernel_files)
    cell = (KERN_DISP / DPI) * 1.8 
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * (cell + 0.8), nrows * (cell + 0.8)),
                             gridspec_kw={"wspace": 0.15, "hspace": 0.35})
    axes_flat = np.atleast_1d(axes).ravel()

    for i, kf in enumerate(kernel_files):
        k = _load_kernel(kf, KERN_DISP)
        axes_flat[i].imshow(k, cmap='gray', interpolation='nearest')
        axes_flat[i].set_xticks([]); axes_flat[i].set_yticks([])
        for spine in axes_flat[i].spines.values():
            spine.set_visible(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    out = user_dir / out_name
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def build_kernels(user_dir: Path):
    kernel_files = []
    for ds_name in DATASETS:
        kf = _collect_kernels(user_dir / ds_name)
        if kf:
            kernel_files = kf
            break

    if not kernel_files:
        print(f"  No kernels found for {user_dir.name}, skipping.")
        return

    n = len(kernel_files)
    nrows_v = 3; ncols_v = int(np.ceil(n / nrows_v))
    nrows_h = 2; ncols_h = int(np.ceil(n / nrows_h))

    _render_kernels(kernel_files, nrows_v, ncols_v, user_dir, "kernels_mosaic_vertical.png")
    _render_kernels(kernel_files, nrows_h, ncols_h, user_dir, "kernels_mosaic_horizontal.png")


def main():
    for user in USERS:
        user_dir = BASE_DIR / user
        if not user_dir.exists():
            print(f"[SKIP] {user_dir} not found")
            continue
        print(f"\n[{user.upper()}]")
        build_vertical(user_dir)
        build_horizontal(user_dir)
        build_kernels(user_dir)
        _old = user_dir / "kernels_mosaic.png"
        if _old.exists():
            _old.unlink()
    print("\nDone.")


if __name__ == "__main__":
    main()
