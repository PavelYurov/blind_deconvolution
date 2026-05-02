"""
build_cython.py  —  Build all .pyx modules in fbdhsgp_cython.

Usage (from any directory):
    python  <path-to>/fbdhsgp_cython/build_cython.py

The script:
  1. Compiles every .pyx → .c   (Cython.Compiler)
  2. Builds every .c → .pyd     (setuptools build_ext)
  3. Copies .pyd → _build_pyd/  (separate output folder)
  4. Copies .c  → _build_c/     (separate output folder)
  5. Removes .c from source dir + temp build dirs

Output layout:
    fbdhsgp_cython/
        *.pyx                  ← sources (untouched)
        build_cython.py        ← this script
        __init__.py
        _build_pyd/            ← compiled .pyd files
        _build_c/              ← generated .c files

Windows MAX_PATH workaround: compilation happens in %TEMP%.
"""

import os
import sys
import shutil
import tempfile
import sysconfig
from pathlib import Path

# ── Locate package directory (same folder as this script) ─────
PKG_DIR = Path(__file__).resolve().parent
PYD_DIR = PKG_DIR / "_build_pyd"
C_DIR   = PKG_DIR / "_build_c"
# Рядом лежит пакет fbdhsgp_cython_pyd, из которого реально импортируется
# скомпилированный код.  Кладём туда же, чтобы новый билд сразу применялся.
INSTALL_DIR = PKG_DIR.parent / "fbdhsgp_cython_pyd"

# ── All .pyx modules to compile ──────────────────────────────
PYX_MODULES = sorted(
    p.stem for p in PKG_DIR.glob("*.pyx")
)

print(f"[build] Package dir : {PKG_DIR}")
print(f"[build] .pyd output : {PYD_DIR}")
print(f"[build] .c   output : {C_DIR}")
print(f"[build] Modules     : {PYX_MODULES}")


def cythonize_all():
    """Stage 1: .pyx → .c via Cython compiler."""
    from Cython.Compiler.Main import compile as cy_compile, CompilationOptions
    from Cython.Compiler import Options

    # Глобальные директивы компиляции Cython:
    # ── производительность ──────────────────────────────────────────────
    #   boundscheck=False       — отключает проверки границ массивов
    #   wraparound=False        — отключает поддержку отрицательных индексов
    #   initializedcheck=False  — не проверять инициализацию memoryview
    #   nonecheck=False         — не проверять None при доступе к атрибутам
    #   cdivision=True          — C-семантика для / и % (для ЦЕЛЫХ чисел;
    #                             на float64 НЕ ВЛИЯЕТ, семантика IEEE та же).
    # Эти директивы не меняют результаты с плавающей точкой, они только
    # убирают Python-обёртки вокруг обращений к памяти.
    directives = {
        'language_level': 3,
        'boundscheck': False,
        'wraparound': True,   # MUST be True — disabling causes ACCESS_VIOLATION
                              # on negative indices like list[-1] / arr[-1].
        'cdivision': True,
        'initializedcheck': False,
        'nonecheck': False,
        # infer_types disabled: Cython sometimes promotes scalar locals to
        # complex when an FFT result flows into them, breaking comparisons
        # like `min(beta * IF_, beta_max)` in solvers.pyx.
        'infer_types': False,
    }
    opts = CompilationOptions(
        language_level=3,
        compiler_directives=directives,
    )
    c_files = {}
    for mod in PYX_MODULES:
        pyx = str(PKG_DIR / f"{mod}.pyx")
        print(f"  cython  {mod}.pyx → {mod}.c")
        result = cy_compile(pyx, opts)
        if result.num_errors:
            raise RuntimeError(f"Cython errors in {mod}.pyx")
        c_files[mod] = str(PKG_DIR / f"{mod}.c")
    return c_files


def build_extensions(c_files):
    """Stage 2: .c → .pyd (or .so) via setuptools."""
    import numpy as np
    from setuptools import Extension, Distribution
    from setuptools.command.build_ext import build_ext

    # Use short temp dir to dodge Windows MAX_PATH
    tmp_root = Path(tempfile.gettempdir()) / "cython_fbdhsgp"
    tmp_root.mkdir(exist_ok=True)

    # ── Compiler optimisation flags ──────────────────────────
    # NOTE: никаких fast-math флагов. Они реассоциируют FP-операции,
    # контрактят FMA, меняют обработку денормалов — и через десятки
    # итераций деконволюции результат расходится с чистым Python.
    # /O2 и -O3 сами по себе строго IEEE-754 совместимы.
    if sys.platform.startswith('win'):
        # /fp:precise — дефолт MSVC, ставим явно на случай унаследованных настроек.
        extra_compile_args = ['/O2', '/fp:precise']
        extra_link_args    = []
    else:
        # -fno-fast-math на всякий случай (перебивает возможный -ffast-math из env).
        # -fno-finite-math-only оставляет корректную обработку NaN/Inf.
        extra_compile_args = ['-O3', '-fno-fast-math', '-fno-finite-math-only']
        extra_link_args    = []

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    built = {}

    for mod, c_path in c_files.items():
        print(f"  cc      {mod}.c → {mod}{ext_suffix}")

        ext = Extension(
            name=mod,
            sources=[c_path],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[
                # Silence numpy deprecation warnings from cimport numpy.
                ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
            ],
        )

        dist = Distribution({"ext_modules": [ext]})
        dist.parse_config_files()

        cmd = build_ext(dist)
        cmd.build_lib = str(tmp_root / "lib")
        cmd.build_temp = str(tmp_root / "tmp")
        cmd.inplace = False
        cmd.ensure_finalized()
        cmd.run()

        # Find the built .pyd/.so
        for root, _, files in os.walk(str(tmp_root / "lib")):
            for f in files:
                if f.startswith(mod) and f.endswith(ext_suffix):
                    built[mod] = os.path.join(root, f)
                    break

    return built, tmp_root


def install_and_cleanup(built, c_files, tmp_root):
    """Stage 3–5: .pyd → _build_pyd/, .c → _build_c/, clean temps."""
    PYD_DIR.mkdir(exist_ok=True)
    C_DIR.mkdir(exist_ok=True)

    # Copy .pyd into _build_pyd/ AND into sibling dcp_cython_pyd/
    for mod, pyd_path in built.items():
        dest = PYD_DIR / Path(pyd_path).name
        shutil.copy2(pyd_path, dest)
        print(f"  install {dest.relative_to(PKG_DIR)}")
        if INSTALL_DIR.exists():
            # Удаляем старые .pyd этого модуля (с любым суффиксом)
            # чтобы Python подхватил свежий, а не закешированный.
            for old in INSTALL_DIR.glob(f"{mod}.*.pyd"):
                try:
                    old.unlink()
                except OSError:
                    pass
            dest_install = INSTALL_DIR / Path(pyd_path).name
            shutil.copy2(pyd_path, dest_install)
            print(f"  install ../{dest_install.relative_to(INSTALL_DIR.parent)}")

    # Move .c into _build_c/
    for mod, c_path in c_files.items():
        c = Path(c_path)
        if c.exists():
            dest_c = C_DIR / c.name
            shutil.move(str(c), str(dest_c))
            print(f"  archive {dest_c.relative_to(PKG_DIR)}")

    # Remove temp build tree
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Remove any stray .pyd left in source dir from previous builds
    for pyd in PKG_DIR.glob("*.pyd"):
        pyd.unlink()

    print("[build] Done — .pyd in _build_pyd/, .c in _build_c/, "
          "source dir clean.")


def main():
    print("\n══ Stage 1: Cython (.pyx → .c) ══")
    c_files = cythonize_all()

    print("\n══ Stage 2: Compile (.c → .pyd) ══")
    built, tmp_root = build_extensions(c_files)

    if len(built) != len(PYX_MODULES):
        missing = set(PYX_MODULES) - set(built.keys())
        print(f"[build] WARNING: failed modules: {missing}")

    print("\n══ Stage 3: Install & Cleanup ══")
    install_and_cleanup(built, c_files, tmp_root)

    print(f"\n[build] Successfully built {len(built)}/{len(PYX_MODULES)} modules")
    for mod in sorted(built.keys()):
        print(f"  ✓ {mod}")


if __name__ == "__main__":
    main()
