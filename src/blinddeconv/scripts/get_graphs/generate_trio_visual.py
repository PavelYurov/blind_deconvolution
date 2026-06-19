#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Одна строка из 3 изображений:
  1) Оцененное ядро
  2) Восстановленное FIRLS
  3) Восстановленное TV/l0
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import matplotlib.mathtext


PROJECT_ROOT = Path(os.path.abspath(__file__)).parent

BASE = PROJECT_ROOT / "presentation_graphics_nonblind_with_estimated_kernels"

PANELS = [
    {
        "img_path": BASE / "Hyperbolic_Secant_Prior_(FIRLS)" / "nonblind" / "kernels"
                         / "drawing_heliod_mediumgaussian_Hyperbolic Secant Prior (FIRLS)_kernel.png",
        "label":    None,           
        "metric":   None,           
        "cmap":     "gray",
        "zoom":     4.0,            
    },
    {
        "img_path": BASE / "Hyperbolic_Secant_Prior_(FIRLS)" / "nonblind" / "restored"
                         / "drawing_heliod_mediumgaussian_Hyperbolic Secant Prior (FIRLS).png",
        "label":    "FIRLS",
        "metric":   "PSNR 15.69   SSIM 0.3689",
        "cmap":     "gray",
        "zoom":     1.0,
    },
    {
        "img_path": BASE / "Hyperbolic_Secant_Prior_(TV+L0)" / "nonblind" / "restored"
                         / "drawing_heliod_mediumgaussian_Hyperbolic Secant Prior (TV+L0).png",
        "label":    r"TV/$l_0$",
        "metric":   "PSNR 18.08   SSIM 0.4475",
        "cmap":     "gray",
        "zoom":     1.0,
    },
]

OUT_PATH = PROJECT_ROOT / "presentation_graphics_nonblind" / "comparison_figures" / "trio_drawing_heliod.png"
OUT_PDF  = OUT_PATH.with_suffix(".pdf")

FIG_W = 10.0
FIG_H = 3.5

LABEL_FONTSIZE  = 11
METRIC_FONTSIZE = 11

# ══════════════════════════════════════════════════════════════════════════════


def _load(path, zoom=1.0):
    p = Path(str(path))
    if not p.exists():
        print(f"  [WARN] не найден файл: {p}")
        return None
    img = mpimg.imread(str(p))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if zoom > 1.0:
        z = int(zoom)
        if img.ndim == 2:
            img = np.repeat(np.repeat(img, z, axis=0), z, axis=1)
        else:
            img = np.repeat(np.repeat(img, z, axis=0), z, axis=1)
    return img


def main():
    n = len(PANELS)
    fig, axes = plt.subplots(1, n, figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.10,
                        wspace=0.04)

    for ax, panel in zip(axes, PANELS):
        img = _load(panel["img_path"], panel.get("zoom", 1.0))

        ax.set_xticks([])
        ax.set_yticks([])

        if img is not None:
            kw = {"cmap": panel.get("cmap", "gray")}
            if img.ndim == 2:
                kw["vmin"] = 0.0
                kw["vmax"] = 1.0
            ax.imshow(np.clip(img, 0, 1), **kw)
        else:
            ax.set_facecolor("#cccccc")
            ax.text(0.5, 0.5, "нет файла", ha="center", va="center",
                    fontsize=9, color="red", transform=ax.transAxes)

        if panel.get("label"):
            ax.set_title(panel["label"], fontsize=LABEL_FONTSIZE, pad=4)

        if panel.get("metric"):
            ax.set_xlabel(panel["metric"], fontsize=METRIC_FONTSIZE, labelpad=4)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight")
    fig.savefig(str(OUT_PDF),  bbox_inches="tight")
    print(f"Сохранено: {OUT_PATH}")
    print(f"Сохранено: {OUT_PDF}")
    plt.close(fig)


if __name__ == "__main__":
    main()
