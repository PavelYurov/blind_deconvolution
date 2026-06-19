#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_er.py — пересчёт error_ratio в presentation_graphics/ через
noise-aware non-blind конвейер.
определение метрики размытое. конкретных деталей по ней нет,
пришлось несколько раз пересчитывать...
дополнительно фикс бага с перевернутыми ядрами.

Запуск:
    python fix_er.py                              # presentation_graphics/
    python fix_er.py --root presentation_graphics_pasha
    python fix_er.py --dry-run
    python fix_er.py --no-impulse                 # пропустить шаг 1
    python fix_er.py --no-bm3d                    # пропустить денойз
    python fix_er.py --lambda-tv 1e-3 --lambda-l0 2e-3 --weight-ring 1.0
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import math
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from blinddeconv.algorithms.blind_deconvolution.our_company.\
    logarithmic_pds.lip_denoise.impulse_noise_estimation import (  # noqa: E402
        detect_impulse_noise, adaptive_median_filter,
    )
from blinddeconv.algorithms.blind_deconvolution.our_company.\
    logarithmic_pds.lip_denoise.pyatykh_noise_reconstruction import (  # noqa: E402
        estimate_noise_params,
    )
from blinddeconv.algorithms.blind_deconvolution.our_company.\
    logarithmic_pds.lip_denoise.vst import vst_bm3d_denoise  # noqa: E402
from blinddeconv.algorithms.blind_deconvolution.our_company.\
    logarithmic_pds.lip_denoise.solvers import (  # noqa: E402
        ringing_artifacts_removal,
    )

_PATH_ANCHORS = (
    "images/compare_data/",
    "presentation_graphics/",
)


def _rebase(p: object) -> str:
    if p is None:
        return ""
    s = str(p)
    if not s or s == "nan":
        return ""
    s_norm = s.replace("\\", "/")
    for anchor in _PATH_ANCHORS:
        idx = s_norm.find(anchor)
        if idx >= 0:
            return str(PROJECT_ROOT / s_norm[idx:])
    return s


def _blurred_path_from_row(row: pd.Series) -> Path | None:
    """images/compare_data/<...>/<dataset>/distorted/<distorted_file>"""
    orig = _rebase(row.get("original_path", ""))
    dist_file = str(row.get("distorted_file", "") or "")
    if not orig or not dist_file:
        return None
    p = Path(orig).parent.parent / "distorted" / dist_file
    return p if p.exists() else None



def _load_gray01(path: Path | str) -> np.ndarray | None:
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img.astype(np.float64) / 255.0


def _load_kernel_rot180_norm(path: Path | str) -> np.ndarray | None:
    img = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[:, :, 0]
    k = img.astype(np.float64)
    k = np.rot90(k, 2)
    s = k.sum()
    if s <= 1e-12:
        return None
    return k / s


def _estimate_blurred_noise(y: np.ndarray, do_impulse: bool
                            ) -> tuple[np.ndarray, dict]:
    y_work = y
    if do_impulse:
        info = detect_impulse_noise(
            y_work, density_threshold=0.0005, outlier_threshold=0.08)
        if info.get('has_impulse'):
            y_work = adaptive_median_filter(
                y_work, info['impulse_mask'], max_window=7)
    try:
        ni = estimate_noise_params(y_work) or {}
    except Exception:
        ni = {}
    ni.setdefault('sigma_norm', 0.0)
    ni.setdefault('noise_type', 'gaussian')
    ni.setdefault('a', 0.0)
    ni.setdefault('b', 0.0)
    return y_work, ni


def _denoise(y: np.ndarray, noise_info: dict, sigma_floor: float
             ) -> np.ndarray:
    """BM3D (gauss) или VST+BM3D (poisson / poisson-gauss).

    Под порогом sigma_floor ничего не делаем — y и так почти чистый.
    """
    sigma = float(noise_info.get('sigma_norm', 0.0) or 0.0)
    if sigma < sigma_floor:
        return y
    nt = noise_info.get('noise_type', 'gaussian')
    if nt in ('poisson', 'poisson_gaussian', 'unknown'):
        out, _ = vst_bm3d_denoise(y, noise_info=noise_info)
        return np.clip(out, 0.0, 1.0)
    import bm3d as bm3d_lib
    out = bm3d_lib.bm3d(y, sigma_psd=max(sigma, 1e-4))
    return np.clip(out, 0.0, 1.0)


def _psf2otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    kh, kw = psf.shape
    H, W = shape
    pad = np.zeros((H, W), dtype=np.float64)
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -(kh // 2), axis=0)
    pad = np.roll(pad, -(kw // 2), axis=1)
    return np.fft.fft2(pad)


def _tikhonov_deconv(y: np.ndarray, k: np.ndarray,
                     lam: float) -> np.ndarray:
    """Тихоновский non-blind в Фурье"""
    K = _psf2otf(k, y.shape)
    Y = np.fft.fft2(y)
    K2 = (K.conj() * K).real
    X = K.conj() * Y / (K2 + lam)
    return np.real(np.fft.ifft2(X))


def _ringing_solve(y_clean: np.ndarray, k: np.ndarray,
                   lambda_tv: float, lambda_l0: float,
                   weight_ring: float) -> np.ndarray:
    """Тяжёлый ringing_artifacts_removal (TV + L0 + bilateral diff)."""
    return ringing_artifacts_removal(
        y_clean, k,
        lambda_tv=lambda_tv,
        lambda_l0=lambda_l0,
        weight_ring=weight_ring,
    )


def _solve(y_clean: np.ndarray, k: np.ndarray, *, solver: str,
           lambda_tv: float, lambda_l0: float, weight_ring: float,
           lambda_tikh: float) -> np.ndarray:
    if solver == "tikhonov":
        return _tikhonov_deconv(y_clean, k, lambda_tikh)
    if solver == "ringing":
        return _ringing_solve(y_clean, k, lambda_tv, lambda_l0, weight_ring)
    raise ValueError(f"unknown solver: {solver!r}")


def _error_ratio(original: np.ndarray, y_clean: np.ndarray,
                 gt_k: np.ndarray, est_k: np.ndarray, *,
                 solver: str,
                 lambda_tv: float, lambda_l0: float,
                 weight_ring: float, lambda_tikh: float) -> float:
    x_true = _solve(y_clean, gt_k, solver=solver,
                    lambda_tv=lambda_tv, lambda_l0=lambda_l0,
                    weight_ring=weight_ring, lambda_tikh=lambda_tikh)
    x_est = _solve(y_clean, est_k, solver=solver,
                   lambda_tv=lambda_tv, lambda_l0=lambda_l0,
                   weight_ring=weight_ring, lambda_tikh=lambda_tikh)
    H = min(original.shape[0], x_true.shape[0], x_est.shape[0])
    W = min(original.shape[1], x_true.shape[1], x_est.shape[1])
    o = original[:H, :W]
    xt = np.clip(x_true[:H, :W], 0.0, 1.0)
    xe = np.clip(x_est[:H, :W], 0.0, 1.0)
    mse_true = float(np.mean((o - xt) ** 2))
    mse_est = float(np.mean((o - xe) ** 2))
    return mse_est / mse_true if mse_true > 1e-12 else 1.0



def _process_row(payload: dict) -> tuple[int, float, str | None]:
    """Обработать одну строку CSV. Возвращает (idx, er, fail_reason | None)."""
    idx = payload["idx"]
    gt_p = payload["gt_p"]
    est_p = payload["est_p"]
    orig_p = payload["orig_p"]
    blur_p = payload["blur_p"]
    do_impulse = payload["do_impulse"]
    do_denoise = payload["do_denoise"]
    sigma_floor = payload["sigma_floor"]
    solver = payload["solver"]
    lambda_tv = payload["lambda_tv"]
    lambda_l0 = payload["lambda_l0"]
    weight_ring = payload["weight_ring"]
    lambda_tikh = payload["lambda_tikh"]
    verbose_errors = payload["verbose_errors"]

    if not (gt_p and est_p and orig_p):
        return idx, math.nan, "missing path in CSV"
    if blur_p is None:
        return idx, math.nan, "blurred not found"
    if not Path(gt_p).exists():
        return idx, math.nan, "gt_kernel file missing"
    if not Path(est_p).exists():
        return idx, math.nan, "est_kernel file missing"
    if not Path(orig_p).exists():
        return idx, math.nan, "original file missing"

    original = _load_gray01(orig_p)
    blurred = _load_gray01(blur_p)
    gt_k = _load_kernel_rot180_norm(gt_p)
    est_k = _load_kernel_rot180_norm(est_p)

    if original is None or blurred is None:
        return idx, math.nan, "imread failed (image)"
    if gt_k is None or est_k is None:
        return idx, math.nan, "kernel sums to zero / unreadable"

    if original.shape != blurred.shape:
        h = min(original.shape[0], blurred.shape[0])
        w = min(original.shape[1], blurred.shape[1])
        original = original[:h, :w]
        blurred = blurred[:h, :w]

    try:
        y_clean, noise_info = _estimate_blurred_noise(
            blurred, do_impulse=do_impulse)
        if do_denoise:
            y_clean = _denoise(y_clean, noise_info, sigma_floor)
        er = _error_ratio(
            original, y_clean, gt_k, est_k,
            solver=solver,
            lambda_tv=lambda_tv, lambda_l0=lambda_l0,
            weight_ring=weight_ring, lambda_tikh=lambda_tikh,
        )
        return idx, round(float(er), 4), None
    except Exception as exc:
        if verbose_errors:
            traceback.print_exc()
        return idx, math.nan, f"solver exception: {type(exc).__name__}"


def process_csv(csv_path: Path, *, do_impulse: bool, do_denoise: bool,
                sigma_floor: float, solver: str,
                lambda_tv: float, lambda_l0: float,
                weight_ring: float, lambda_tikh: float,
                workers: int = 1,
                allowed_datasets: set[str] | None = None,
                dry_run: bool = False,
                verbose_errors: bool = False) -> dict:
    df = pd.read_csv(csv_path)
    if "error_ratio" not in df.columns:
        return {"csv": csv_path.name, "skipped": True, "reason": "no error_ratio column"}

    n_total = len(df)
    if allowed_datasets and "dataset" in df.columns:
        ds_lower = df["dataset"].astype(str).str.lower()
        allowed_lower = {d.lower() for d in allowed_datasets}
        active_mask = ds_lower.isin(allowed_lower).to_numpy()
    else:
        active_mask = np.ones(n_total, dtype=bool)
    n_active = int(active_mask.sum())

    payloads: list[dict] = []
    for i, row in df.iterrows():
        if not active_mask[i]:
            continue
        blur_p = _blurred_path_from_row(row)
        payloads.append({
            "idx": int(i),
            "gt_p": _rebase(row.get("gt_kernel_path", "")),
            "est_p": _rebase(row.get("kernel_path", "")),
            "orig_p": _rebase(row.get("original_path", "")),
            "blur_p": str(blur_p) if blur_p is not None else None,
            "do_impulse": do_impulse,
            "do_denoise": do_denoise,
            "sigma_floor": sigma_floor,
            "solver": solver,
            "lambda_tv": lambda_tv,
            "lambda_l0": lambda_l0,
            "weight_ring": weight_ring,
            "lambda_tikh": lambda_tikh,
            "verbose_errors": verbose_errors,
        })

    new_er = [math.nan] * n_total
    fail_reasons: dict[str, int] = {}
    n_ok = 0
    n_fail = 0
    if "error_ratio" in df.columns:
        old_er_num = pd.to_numeric(df["error_ratio"], errors="coerce")
        for i in range(n_total):
            if not active_mask[i]:
                v = old_er_num.iloc[i]
                new_er[i] = float(v) if pd.notna(v) else math.nan

    def _bump(reason: str) -> None:
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    if workers <= 1:
        for p in payloads:
            idx, er, reason = _process_row(p)
            new_er[idx] = er
            if reason is None:
                n_ok += 1
            else:
                n_fail += 1
                _bump(reason)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_row, p) for p in payloads]
            for fut in as_completed(futures):
                idx, er, reason = fut.result()
                new_er[idx] = er
                if reason is None:
                    n_ok += 1
                else:
                    n_fail += 1
                    _bump(reason)
                done += 1
                if done % 25 == 0 or done == n_active:
                    print(f"    [progress] {done}/{n_active}", flush=True)

    new_er_s = pd.Series(new_er, index=df.index)

    if not dry_run:
        bak = csv_path.with_suffix(csv_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(csv_path, bak)
        if "error_ratio_orig" not in df.columns:
            df["error_ratio_orig"] = df["error_ratio"]
        df["error_ratio"] = new_er_s
        df["error_ratio_fixed"] = new_er_s
        df.to_csv(csv_path, index=False)

    valid_old = pd.to_numeric(df.get("error_ratio_orig", df["error_ratio"]),
                              errors="coerce").dropna()
    valid_new = pd.to_numeric(new_er_s, errors="coerce").dropna()
    return {
        "csv": csv_path.name,
        "rows": n_total,
        "active": n_active,
        "recomputed": n_ok,
        "failed": n_fail,
        "old_median": float(valid_old.median()) if len(valid_old) else math.nan,
        "new_median": float(valid_new.median()) if len(valid_new) else math.nan,
        "old_mean":   float(valid_old.mean())   if len(valid_old) else math.nan,
        "new_mean":   float(valid_new.mean())   if len(valid_new) else math.nan,
        "fail_reasons": dict(fail_reasons),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="presentation_graphics",
                        help="Папка с результатами (по умолчанию presentation_graphics)")
    parser.add_argument("--no-impulse", action="store_true",
                        help="Пропустить детекцию импульсного шума")
    parser.add_argument("--no-bm3d", action="store_true",
                        help="Пропустить шаг денойза (BM3D / VST+BM3D)")
    parser.add_argument("--sigma-floor", type=float, default=0.005,
                        help="Под этим σ денойз пропускается (default 0.005)")
    parser.add_argument("--solver", choices=["tikhonov", "ringing"],
                        default="tikhonov",
                        help="non-blind solver: 'tikhonov' (быстрый, default) "
                             "или 'ringing' (TV+L0+bilateral, медленный)")
    parser.add_argument("--lambda-tikh", type=float, default=1e-3,
                        help="Tikhonov regularization (default 1e-3). "
                             "Чем меньше - тем острее, но больше ringing.")
    parser.add_argument("--lambda-tv", type=float, default=1e-3,
                        help="ringing_artifacts_removal: lambda_tv (default 1e-3)")
    parser.add_argument("--lambda-l0", type=float, default=2e-3,
                        help="ringing_artifacts_removal: lambda_l0 (default 2e-3)")
    parser.add_argument("--weight-ring", type=float, default=1.0,
                        help="ringing_artifacts_removal: weight_ring (default 1.0)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Кол-во параллельных процессов на строку (default 6)")
    parser.add_argument("--datasets", nargs="+",
                        default=["Levin", "Kohler", "Sun", "Set12"],
                        help="Какие датасеты пересчитывать (по колонке 'dataset', "
                             "регистронезависимо). Default: Levin Kohler Sun Set12. "
                             "'all' чтобы пересчитать все.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Только посчитать, ничего не писать")
    parser.add_argument("--verbose-errors", action="store_true",
                        help="Печатать stack-trace при ошибках в строке")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    if not root.is_dir():
        print(f"[FATAL] Папка не найдена: {root}")
        sys.exit(1)

    print(f"[INFO] Корень       : {root}")
    print(f"[INFO] impulse step : {not args.no_impulse}")
    print(f"[INFO] denoise      : {not args.no_bm3d}  (BM3D / VST+BM3D, sigma_floor={args.sigma_floor})")
    print(f"[INFO] solver       : {args.solver}")
    if args.solver == "tikhonov":
        print(f"[INFO] λ_tikh       : {args.lambda_tikh}")
    else:
        print(f"[INFO] ringing      : λ_tv={args.lambda_tv}  λ_l0={args.lambda_l0}  w_ring={args.weight_ring}")
    print(f"[INFO] workers      : {args.workers}")
    print(f"[INFO] Dry-run      : {args.dry_run}")

    csvs = sorted(root.glob("*/all_results_*.csv"))
    if not csvs:
        print(f"[WARN] Не найдено all_results_*.csv в {root}/<algo>/")
        sys.exit(0)

    print(f"[INFO] Найдено CSV: {len(csvs)}\n")
    raw_ds = list(args.datasets) if args.datasets else []
    if any(d.lower() == "all" for d in raw_ds):
        allowed = None
    else:
        allowed = set(raw_ds)
    print(f"[INFO] Datasets    : {'ALL' if allowed is None else sorted(allowed)}")

    summary_rows = []
    for csv_path in csvs:
        algo = csv_path.parent.name
        print(f"[{algo}] {csv_path.name}")
        info = process_csv(
            csv_path,
            do_impulse=not args.no_impulse,
            do_denoise=not args.no_bm3d,
            sigma_floor=args.sigma_floor,
            solver=args.solver,
            lambda_tv=args.lambda_tv,
            lambda_l0=args.lambda_l0,
            weight_ring=args.weight_ring,
            lambda_tikh=args.lambda_tikh,
            workers=args.workers,
            allowed_datasets=allowed,
            dry_run=args.dry_run,
            verbose_errors=args.verbose_errors,
        )
        summary_rows.append({"algo": algo, **{k: v for k, v in info.items()
                                              if k != "fail_reasons"}})
        print(f"  rows={info.get('rows')}  active={info.get('active')}  "
              f"recomputed={info.get('recomputed')}  "
              f"failed={info.get('failed')}  "
              f"ER median: {info.get('old_median'):.3f} → {info.get('new_median'):.3f}  "
              f"mean: {info.get('old_mean'):.3f} → {info.get('new_mean'):.3f}")
        if info.get("fail_reasons"):
            for r, c in info["fail_reasons"].items():
                print(f"    fail: {r} × {c}")

    print()
    print("=" * 78)
    print("СВОДКА:")
    print("=" * 78)
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))

    if not args.dry_run:
        out = root / "fix_er_summary.csv"
        summary.to_csv(out, index=False)
        print(f"\n[OK] Сводка сохранена: {out}")
    else:
        print("\n[DRY-RUN] CSV не модифицированы.")


if __name__ == "__main__":
    main()
