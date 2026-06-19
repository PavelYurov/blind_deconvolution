"""
Гистограммы градиентов для двух изображений из log_test:
  - bridge_losso_clean
  - boy_losso_clean

Каждый график: Оригинал (красный) / Искажённое (зелёный) / Восстановленное (синий)

Результаты сохраняются в:
  BASE / gradient_hist_log_test /
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')

from gradient_histogram import plot_gradient_histograms

BASE = Path(r"D:\for_proga\franework_deconvolution\framework (9)")
OUT_DIR = BASE / 'gradient_hist_log_test'
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_ORIG      = '#E62020'
COLOR_DISTORTED = '#66C244'
COLOR_RESTORED  = '#2176AE'

HIST_SETTINGS = dict(
    bin_range       = (-150.0, 150.0),
    n_bins          = 300,
    ylim            = (-12.0, 0.0),
    figsize         = (8, 6),
    grad_directions = 'both',
    dpi             = 150,
)

CASES = [
    {
        'name':      'bridge',
        'orig':      BASE / 'images/compare_data/kostya/log_test/originals/bridge.png',
        'distorted': BASE / 'images/compare_data/kostya/log_test/distorted/bridge_losso_clean.png',
        'restored':  BASE / 'presentation_graphics_anton_FINALLY_2/Fast_BD_Hyper-Sparse_Gradient/log_test/bridge_losso_clean/restored_final.png',
    },
    {
        'name':      'boy',
        'orig':      BASE / 'images/compare_data/kostya/log_test/originals/boy.png',
        'distorted': BASE / 'images/compare_data/kostya/log_test/distorted/boy_losso_clean.png',
        'restored':  BASE / 'presentation_graphics_anton_FINALLY_2/Fast_BD_Hyper-Sparse_Gradient/log_test/boy_losso_clean/restored_final.png',
    },
]


def main() -> None:
    for case in CASES:
        name = case['name']
        for key in ('orig', 'distorted', 'restored'):
            if not case[key].exists():
                print(f"  [warn] Not found: {case[key]}")

        series = [
            dict(label='Оригинал',        color=COLOR_ORIG,      paths=[case['orig']]),
            dict(label='Искажённое',       color=COLOR_DISTORTED, paths=[case['distorted']]),
            dict(label='Восстановленное',  color=COLOR_RESTORED,  paths=[case['restored']]),
        ]

        out_path = OUT_DIR / f'{name}_grad_hist.png'
        print(f"Building histogram for {name}...")
        plot_gradient_histograms(
            series=series,
            output_path=out_path,
            title='',
            **HIST_SETTINGS,
        )

    print("\nГотово.")


if __name__ == '__main__':
    main()
