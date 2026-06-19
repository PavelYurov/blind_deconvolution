"""
build_all_cython.py 
Находит и запускает все build_cython.py в проекте.

Сканирует дерево проекта на наличие скриптов называющихся build_cython.py
и запускает каждый из них в своем собственнои подпроцессе. Работает из 
любой позиции внутри проекта - корень проета детектится автоматически.

Использование:
    python scripts/build_cython/build_all_cython.py            # собрать все
    python scripts/build_cython/build_all_cython.py --list     # список найденных скриптов
    python scripts/build_cython/build_all_cython.py --filter gbbid lip   # фильтр подстроки в пути к скрипту
    python scripts/build_cython/build_all_cython.py --jobs 4   # параллельные потоки исполнения
    python scripts/build_cython/build_all_cython.py --keep-going   # не прекращать сборку при возникновении ошибки

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_MARKERS = ("pyproject.toml", "setup.cfg", "requirements.txt", ".git")


def find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(10):
        if any((cur / m).exists() for m in _MARKERS):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parents[1] if len(start.parents) >= 2 else start


PROJECT_ROOT = find_project_root(_HERE)


_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    "build", "dist", ".mypy_cache", ".pytest_cache",
    "_build_pyd", "_build_c",
}


def find_build_scripts(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if "build_cython.py" in filenames:
            p = Path(dirpath) / "build_cython.py"
            if p.resolve() == Path(__file__).resolve():
                continue
            found.append(p)
    return sorted(found)


def run_one(script: Path) -> tuple[Path, int, float, str]:
    t0 = time.time()
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return script, proc.returncode, time.time() - t0, out
    except Exception as e:
        return script, 1, time.time() - t0, f"[runner-error] {e!r}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build every build_cython.py in the project.")
    parser.add_argument("--list", action="store_true",
                        help="Only list discovered build_cython.py scripts and exit.")
    parser.add_argument("--filter", nargs="+", default=None,
                        help="Run only scripts whose path contains ANY of these substrings.")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of parallel builds (default: 1, sequential).")
    parser.add_argument("--keep-going", "-k", action="store_true",
                        help="Continue after a failure instead of stopping.")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT,
                        help="Override project root directory.")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"[build-all] Project root: {root}")

    scripts = find_build_scripts(root)
    if args.filter:
        needles = [s.lower() for s in args.filter]
        scripts = [p for p in scripts
                   if any(n in str(p).lower() for n in needles)]

    if not scripts:
        print("[build-all] No build_cython.py scripts found.")
        return 0

    print(f"[build-all] Found {len(scripts)} build script(s):")
    for p in scripts:
        print(f"   - {p.relative_to(root) if root in p.parents else p}")

    if args.list:
        return 0

    results: list[tuple[Path, int, float, str]] = []
    failed: list[Path] = []

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, s): s for s in scripts}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                script, rc, elapsed, out = res
                tag = "OK" if rc == 0 else f"FAIL ({rc})"
                print(f"\n[{tag}] {script}  ({elapsed:.1f}s)")
                print(out.rstrip())
                if rc != 0:
                    failed.append(script)
    else:
        for s in scripts:
            print(f"\n{'═' * 80}\n[build-all] >>> {s}\n{'═' * 80}")
            res = run_one(s)
            results.append(res)
            script, rc, elapsed, out = res
            print(out.rstrip())
            tag = "OK" if rc == 0 else f"FAIL ({rc})"
            print(f"[{tag}] {script}  ({elapsed:.1f}s)")
            if rc != 0:
                failed.append(script)
                if not args.keep_going:
                    print("[build-all] Aborting — use --keep-going to continue past failures.")
                    break

    print(f"\n{'═' * 80}\n[build-all] SUMMARY")
    total_time = sum(r[2] for r in results)
    ok = sum(1 for r in results if r[1] == 0)
    print(f"  total scripts run : {len(results)}")
    print(f"  ok                : {ok}")
    print(f"  failed            : {len(failed)}")
    print(f"  wall time         : {total_time:.1f}s")
    if failed:
        print("\n[build-all] Failed scripts:")
        for p in failed:
            print(f"   - {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
