"""Восстанавливает (пере)собирает общий файл all_results_<alg>.csv
из per-dataset файлов results_<alg>_<dataset>.csv в той же папке алгоритма.

Использование:
    python fix_combined_table.py                       # все алгоритмы в presentation_graphics
    python fix_combined_table.py "Extreme_Channel_prior_(Base)"
    python fix_combined_table.py --root presentation_graphics "Extreme Channel prior (Base)"

Логика:
- Папка алгоритма — каждая подпапка ROOT, содержащая хотя бы один
  results_*_<dataset>.csv в подкаталогах (Set12, Sun, Levin, Kohler,
  Complexity_Test, Grid_Test и т.д.).
- Имя алгоритма извлекается из имени файла results_<alg>_<dataset>.csv.
- Старый all_results_<alg>.csv (если есть) бэкапится в .bak.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

RESULTS_RE = re.compile(r"^results_(?P<alg>.+?)_(?P<dataset>[^_]+(?:_[^_]+)*)\.csv$")


def find_per_dataset_files(alg_dir: Path) -> list[tuple[str, str, Path]]:
    """Возвращает список (alg_name, dataset_name, path) для results_*.csv
    в подпапках алгоритма."""
    out: list[tuple[str, str, Path]] = []
    for sub in alg_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.glob("results_*.csv"):
            m = RESULTS_RE.match(f.name)
            if not m:
                continue
            ds_name = sub.name
            out.append((m.group("alg"), ds_name, f))
    return out


def fix_one_algorithm(alg_dir: Path, dry_run: bool = False) -> bool:
    files = find_per_dataset_files(alg_dir)
    if not files:
        print(f"  [skip] {alg_dir.name}: нет per-dataset results_*.csv")
        return False

    alg_name = files[0][0]
    combined_path = alg_dir / f"all_results_{alg_name}.csv"

    frames: list[pd.DataFrame] = []
    datasets_found: list[str] = []
    for _, ds, p in files:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"    ! не удается прочитать {p.name}: {e}")
            continue
        if df.empty:
            continue
        if 'dataset' not in df.columns or df['dataset'].isna().all():
            df['dataset'] = ds
        else:
            df['dataset'] = df['dataset'].fillna(ds)
        frames.append(df)
        datasets_found.append(f"{ds}({len(df)})")

    if not frames:
        print(f"  [skip] {alg_dir.name}: все per-dataset csv пустые")
        return False

    combined = pd.concat(frames, ignore_index=True, sort=False)

    print(f"  {alg_dir.name}: {len(combined)} строк | {', '.join(datasets_found)}")

    if dry_run:
        return True

    if combined_path.exists():
        bak = combined_path.with_suffix(combined_path.suffix + ".bak")
        try:
            combined_path.replace(bak)
            print(f"    backup -> {bak.name}")
        except Exception as e:
            print(f"    ! не удалось сделать бэкап старого файла: {e}")

    combined.to_csv(combined_path, index=False, encoding='utf-8-sig')
    print(f"    записан -> {combined_path.name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("alg", nargs="?", default=None,
                    help="Имя папки алгоритма. "
                         "Если не задано — обработать все.")
    ap.add_argument("--root", default="presentation_graphics",
                    help="Корень с папками алгоритмов.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"Не найден корень: {root.resolve()}")
        return 2

    if args.alg:
        candidates = [
            root / args.alg,
            root / args.alg.replace(" ", "_"),
        ]
        alg_dir = next((c for c in candidates if c.is_dir()), None)
        if alg_dir is None:
            print(f"Папка алгоритма не найдена: {args.alg}")
            return 2
        targets = [alg_dir]
    else:
        targets = [d for d in root.iterdir() if d.is_dir()]

    ok = 0
    for d in targets:
        if fix_one_algorithm(d, dry_run=args.dry_run):
            ok += 1
    print(f"\nГотово: пересобрано {ok} из {len(targets)} алгоритмов.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
